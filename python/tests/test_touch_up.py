import tempfile
import unittest

import numpy as np
from PIL import Image

from model_maker import bake, inspection, mesh_tools as M, pipeline, touch_up

from . import shapes

SIX_VIEWS = ((0, 0.0), (90, 0.0), (180, 0.0), (270, 0.0), (0, 89.99), (0, -89.99))
GREY = (0.5, 0.5, 0.5)
BRASS = (0.72, 0.55, 0.25)


def candle_in_a_bowl():
    """A holder's dish (its rim 10 cm high, 24 cm across) with a candle standing in its middle."""
    candle = ((-0.03, 0.0, -0.03), (0.03, 0.25, 0.03))
    corners, triangles = shapes.surface_of([((-0.12, 0.0, -0.12), (0.12, 0.02, 0.12)),
                                            ((-0.12, 0.0, -0.12), (-0.1, 0.1, 0.12)),
                                            ((0.1, 0.0, -0.12), (0.12, 0.1, 0.12)),
                                            ((-0.12, 0.0, -0.12), (0.12, 0.1, -0.1)),
                                            ((-0.12, 0.0, 0.1), (0.12, 0.1, 0.12)), candle], 0.005)
    return corners, triangles


def painted_from_six(corners, triangles, colour=GREY, grain=0.0):
    """The colour cloud the six views would leave: `colour` (with a little grain, as paint has) wherever one of them
    sees the surface squarely."""
    points, normals, _ = M.sampled(corners, triangles, 400000, seed=2)
    seen = np.zeros(len(points), dtype=bool)
    for turn, tilt in SIX_VIEWS:
        seen |= touch_up.squarely_seen(corners, triangles, points, normals, turn, tilt)
    colours = np.tile(colour, (int(seen.sum()), 1))
    colours += np.random.default_rng(5).uniform(-grain, grain, colours.shape)
    return points[seen], normals[seen], np.clip(colours, 0, 1)


WAX = (0.93, 0.9, 0.84)
# A pale grey-green, as the fifth trial's inpainter painted a glass cup round the candle's foot.
GLASS = (0.55, 0.62, 0.6)
LINING = (0.86, 0.79, 0.62)


def on_candle(points):
    """Which points lie on the candle (its surface half a voxel out from its box), not on the dish's floor."""
    return (np.abs(points[:, 0]) <= 0.036) & (np.abs(points[:, 2]) <= 0.036) & (points[:, 1] > 0.026)


def wax_and_brass(corners, triangles):
    """The candle in a bowl as the six views would leave it in two colours, wax on the candle and brass on the dish:
    the colour cloud, and the corners' colours as the paint stage leaves them. Those the cloud reaches take its colour;
    the candle's foot, hidden by the rim, takes wax or brass corner by corner (as the fifth trial's candle's corners,
    given colours by views that saw them at a slant: its foot spiked with both); the rest are spread from their
    neighbours (`paint.spread`)."""
    from model_maker import paint
    from model_maker.mesh_tools import corner_normals

    points, normals, _ = painted_from_six(corners, triangles)
    colours = np.where(on_candle(points)[:, None], WAX, BRASS)
    cloud = (points, normals, colours)
    known = ~touch_up.unpainted(corners, corner_normals(corners, triangles), cloud)
    corner_colours = np.where(on_candle(corners)[:, None], WAX, BRASS) * known[:, None]
    foot = on_candle(corners) & ~known
    mottled = np.random.default_rng(3).random(len(corners)) < 0.5
    corner_colours[foot] = np.where(mottled[foot, None], WAX, BRASS)
    known |= foot
    return cloud, paint.spread(corner_colours, known, triangles)


def glass_round_the_foot(grain=0.02):
    """Paints the masked pixels brass, but those not drawn brass (the candle's foot, its corners wax and brass) a
    glass cup's grey-green."""
    def inpaint(picture, mask, text, seed):
        image = np.asarray(picture).astype(np.float64) / 255
        masked = np.asarray(mask) > 127
        waxen = np.linalg.norm(image - BRASS, axis=2) > 0.08
        noise = np.random.default_rng(seed).uniform(-grain, grain, image.shape)
        image[masked] = np.clip(np.where(waxen[..., None], GLASS, BRASS) + noise, 0, 1)[masked]
        return Image.fromarray((image * 255).round().astype(np.uint8))
    return inpaint


def a_cup():
    """A cup 10 cm high and 24 cm across, its walls 2 cm thick: the candle's dish without the candle."""
    return shapes.surface_of([((-0.12, 0.0, -0.12), (0.12, 0.02, 0.12)),
                              ((-0.12, 0.0, -0.12), (-0.1, 0.1, 0.12)), ((0.1, 0.0, -0.12), (0.12, 0.1, 0.12)),
                              ((-0.12, 0.0, -0.12), (0.12, 0.1, -0.1)), ((-0.12, 0.0, 0.1), (0.12, 0.1, 0.12))], 0.005)


def painting(colour, grain=0.0):
    """A stand-in inpainter painting every masked pixel `colour` (with grain)."""
    def inpaint(picture, mask, text, seed):
        image = np.asarray(picture).astype(np.float64) / 255
        masked = np.asarray(mask) > 127
        noise = np.random.default_rng(seed).uniform(-grain, grain, image.shape)
        image[masked] = np.clip(np.asarray(colour) + noise[masked], 0, 1)
        return Image.fromarray((image * 255).round().astype(np.uint8))
    return inpaint


def with_dark_blobs(colour, grain=0.0):
    """Paints the masked pixels `colour`, but with a dark blob in the middle of each masked area's bounding box (as
    SDXL painted holes into the candle's dish)."""
    good = painting(colour, grain)

    def inpaint(picture, mask, text, seed):
        image = np.asarray(good(picture, mask, text, seed)).copy()
        masked = np.asarray(mask) > 127
        rows, cols = np.nonzero(masked)
        r, c = np.ogrid[:image.shape[0], :image.shape[1]]
        for centre_row, centre_col in ((np.percentile(rows, 30), np.percentile(cols, 30)),
                                       (np.percentile(rows, 70), np.percentile(cols, 70))):
            blob = ((r - centre_row) ** 2 + (c - centre_col) ** 2 < 40 ** 2) & masked
            image[blob] = (25, 18, 12)
        return Image.fromarray(image)
    return inpaint


class Recording:
    """Paints as `inpaint` does and keeps every picture it was given."""

    def __init__(self, inpaint):
        self.inpaint, self.given = inpaint, []

    def __call__(self, picture, mask, text, seed):
        self.given.append((np.asarray(picture).copy(), np.asarray(mask) > 127, seed))
        return self.inpaint(picture, mask, text, seed)


def quiet(_):
    pass


class TouchUp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corners, cls.triangles = candle_in_a_bowl()
        cls.grey_cloud = painted_from_six(cls.corners, cls.triangles)
        cls.brass_cloud = painted_from_six(cls.corners, cls.triangles, BRASS, grain=0.04)
        cls.brass = np.tile(BRASS, (len(cls.corners), 1))

    def test_the_views_chosen_look_down_into_the_bowl_and_paint_what_the_six_missed(self):
        corners, triangles, cloud = self.corners, self.triangles, self.grey_cloud
        colours = np.full((len(corners), 3), 0.5)
        before = bake.baked(corners, triangles, colours, cloud, side=512)
        self.assertLess(before["coverage"], 0.93, "the six views leave the dish's inside and the candle's foot")
        # A fill like the paint round it: its grey, a shade lighter so the new colour can be told apart.
        touched, record, sources = touch_up.touched_up(painting((0.53, 0.53, 0.53)), corners, triangles, colours,
                                                       cloud, "a candle", report=quiet)
        self.assertTrue(record["views"])
        self.assertTrue(all(v["tilt_deg"] > 0 for v in record["views"]), record)
        self.assertGreater(len(touched[0]), len(cloud[0]))
        after = bake.baked(corners, triangles, colours, touched, side=512)
        self.assertGreater(after["coverage"], 0.97, (before["coverage"], after["coverage"], record))
        # The new colour landed in the dish (a little on the rim's bevelled edges, which the six views saw at a slant).
        added = touched[0][len(cloud[0]):]
        self.assertTrue((added[:, 1] < 0.11).mean() > 0.9)
        self.assertGreater((np.abs(added[:, [0, 2]]).max(1) < 0.105).mean(), 0.9)
        # Each point is marked with where its colour came from: the six views (0), or touch-up view k (k + 1).
        self.assertEqual(len(sources), len(touched[0]))
        self.assertTrue((sources[:len(cloud[0])] == 0).all())
        self.assertEqual(set(np.unique(sources[len(cloud[0]):])), set(range(1, len(record["views"]) + 1)))

    def test_a_shape_the_six_views_see_whole_needs_no_view(self):
        corners, triangles = shapes.surface_of([((0.0, 0.0, 0.0), (0.3, 0.2, 0.3))], 0.01)
        views, before = touch_up.chosen_views(corners, triangles, painted_from_six(corners, triangles))
        self.assertEqual(views, [])
        self.assertLess(before, 0.02, "only its bevelled edges and corners")

    def test_a_colour_found_nowhere_round_the_fill_is_foreign(self):
        """The fill check's measure: each filled point's colour against the painted points nearest it on the same
        surface; brass among brass is near, black or a pale grey among brass is far."""
        points, normals, colours = self.brass_cloud
        pick = np.random.default_rng(1).choice(len(points), 600, replace=False)
        ring = tuple(np.delete(part, pick, axis=0) for part in (points, normals, colours))
        near = touch_up.misfit(points[pick], normals[pick], colours[pick] * 0.97, ring)
        self.assertLess(np.percentile(near, 95), 6.0)
        black = touch_up.misfit(points[pick], normals[pick], np.full((600, 3), 0.08), ring)
        grey = touch_up.misfit(points[pick], normals[pick], np.full((600, 3), 0.8), ring)
        self.assertGreater(black.min(), touch_up.FOREIGN_DELTA_E)
        self.assertGreater(grey.min(), touch_up.FOREIGN_DELTA_E)

    def test_the_fill_check_fails_dark_blobs_and_passes_a_fill_like_its_surroundings(self):
        good, good_record, good_from = touch_up.touched_up(
            painting(BRASS, 0.03), self.corners, self.triangles, self.brass, self.brass_cloud, "a candle",
            report=quiet, judge=False)
        blobbed, blob_record, blob_from = touch_up.touched_up(
            with_dark_blobs(BRASS, 0.03), self.corners, self.triangles, self.brass, self.brass_cloud, "a candle",
            report=quiet, judge=False)
        good_fit = touch_up.fill_fit(*good, good_from, self.corners, good_record["views"])
        blob_fit = touch_up.fill_fit(*blobbed, blob_from, self.corners, blob_record["views"])
        self.assertTrue(inspection.fill_check(good_fit).passed, good_fit)
        failed = inspection.fill_check(blob_fit)
        self.assertFalse(failed.passed, blob_fit)
        self.assertEqual(failed.name, "fill matches the paint round it")
        self.assertGreater(blob_fit["largest_patch"], touch_up.LARGEST_FOREIGN_PATCH)
        self.assertLess(good_fit["largest_patch"], touch_up.LARGEST_FOREIGN_PATCH / 2)

    def test_the_fill_is_judged_and_a_foreign_colour_is_not_put_on_the_surface(self):
        """With the judge on (as the pipeline runs it), dark blobs are tried again, and what is still foreign is
        left out: those places keep their spread colour and count as unpainted."""
        recording = Recording(with_dark_blobs(BRASS, 0.03))
        touched, record, sources = touch_up.touched_up(recording, self.corners, self.triangles, self.brass,
                                                       self.brass_cloud, "a candle", report=quiet)
        fit = touch_up.fill_fit(*touched, sources, self.corners, record["views"])
        self.assertTrue(inspection.fill_check(fit).passed, fit)
        self.assertEqual(fit["largest_patch"], 0.0)
        added = touched[2][len(self.brass_cloud[0]):]
        self.assertLess(added.sum(1).min(), 3.0)
        self.assertGreater(added.sum(1).min(), 0.6, "no dark blob's colour reached the surface")
        view = record["views"][0]
        # A blobbed fill is tried again: past `FILL_TRIES` while its kept try was, as painted, more than
        # `MOST_FOREIGN` foreign.
        more = view["as_painted"]["foreign"] > touch_up.MOST_FOREIGN
        self.assertEqual(view["tries"], touch_up.MOST_FILL_TRIES if more else touch_up.FILL_TRIES, view)
        self.assertGreater(view["foreign"], 0.0)
        self.assertGreater(view["left_out"], 0)
        self.assertEqual(len({seed for _, _, seed in recording.given}), len(recording.given), "each try its seed")

    def test_a_foreign_fill_on_every_try_adds_nothing(self):
        touched, record, sources = touch_up.touched_up(painting((0.85, 0.85, 0.85)), self.corners, self.triangles,
                                                       self.brass, self.brass_cloud, "a candle", report=quiet)
        self.assertEqual(len(touched[0]), len(self.brass_cloud[0]), record)
        self.assertFalse(sources.any())

    def test_a_good_second_try_is_kept(self):
        calls = []

        def first_grey_then_brass(picture, mask, text, seed):
            calls.append(seed)
            colour = (0.85, 0.85, 0.85) if len(calls) % 2 else BRASS
            return painting(colour, 0.03)(picture, mask, text, seed)

        touched, record, sources = touch_up.touched_up(first_grey_then_brass, self.corners, self.triangles,
                                                       self.brass, self.brass_cloud, "a candle", report=quiet, most=1)
        self.assertEqual(record["views"][0]["tries"], 2)
        self.assertEqual(record["views"][0]["kept_try"], 1)
        self.assertLess(record["views"][0]["foreign"], 0.01)
        self.assertGreater(len(touched[0]), len(self.brass_cloud[0]))

    def test_each_view_sees_the_paint_and_the_fills_before_it(self):
        """The picture given to the inpainter shows the colour cloud's paint (and the earlier views' fills), not only
        the colours spread to the corners: the fill continues what is there."""
        fill = (0.78, 0.6, 0.27)
        recording = Recording(painting(fill))
        touch_up.touched_up(recording, self.corners, self.triangles, self.brass, self.brass_cloud, "a candle",
                            report=quiet)
        later = [picture for picture, _, _ in recording.given[1:]]
        self.assertTrue(later)
        fill_rgb = np.round(np.asarray(fill) * 255)
        shows_fill = [np.abs(picture.astype(float) - fill_rgb).max(-1) <= 2 for picture in later]
        self.assertTrue(any(found.sum() > 200 for found in shows_fill), [int(f.sum()) for f in shows_fill])

    def test_only_pixels_the_inpainter_repaints_count_as_painted(self):
        """SDXL's pipeline shrinks the mask to its latents (one cell for 8 by 8 pixels, the cell's corner pixel
        deciding): a masked pixel in an unmasked cell comes back as it went in, the spread colour. Only pixels in a
        masked cell are put on the surface."""
        fill = (0.66, 0.5, 0.22)

        def like_the_pipeline(picture, mask, text, seed):
            image = np.asarray(picture).copy()
            cells = touch_up.latent_masked(np.asarray(mask) > 127)
            image[cells] = np.round(np.asarray(fill) * 255)
            return Image.fromarray(image)

        touched, record, sources = touch_up.touched_up(like_the_pipeline, self.corners, self.triangles, self.brass,
                                                       self.brass_cloud, "a candle", report=quiet)
        added = touched[2][len(self.brass_cloud[0]):]
        self.assertGreater(len(added), 1000)
        self.assertLess(np.abs(added - np.round(np.asarray(fill) * 255) / 255).max(), 0.01)

    def test_the_mask_covers_a_latent_cell_round_every_wanted_pixel(self):
        wanted = np.zeros((touch_up.SIDE, touch_up.SIDE), dtype=bool)
        wanted[100:103, 205:207] = True
        wanted[400, 401] = True
        hit = np.ones_like(wanted)
        mask = touch_up.grown(wanted, hit)
        self.assertTrue(touch_up.latent_masked(mask)[wanted].all())

    def test_an_older_cloud_is_split_by_its_record(self):
        record = {"views": [{"pixels": 3}, {"pixels": 2}]}
        painted = {"cloud_points": np.zeros((10, 3))}
        self.assertEqual(touch_up.fill_sources(painted, record).tolist(), [0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
        self.assertIsNone(touch_up.fill_sources(painted, None))
        marked = {"cloud_points": np.zeros((5, 3)), "cloud_from": np.array([0, 1, 1, 2, 2])}
        self.assertEqual(touch_up.fill_sources(marked, record).tolist(), [0, 1, 1, 2, 2])

    def test_a_skipped_views_number_is_put_in_the_records_order(self):
        # A chosen view with nothing to paint was skipped without a record entry, and the
        # next view kept its own number (3 for the record's second): the finish measured it from another's camera.
        record = {"views": [{"pixels": 2}, {"pixels": 0}, {"pixels": 1}]}
        marked = {"cloud_points": np.zeros((4, 3)), "cloud_from": np.array([0, 1, 1, 4])}
        self.assertEqual(touch_up.fill_sources(marked, record).tolist(), [0, 1, 1, 3])
        numbered = {"views": [{"pixels": 2, "source": 1}, {"pixels": 1, "source": 2}]}
        self.assertEqual(touch_up.fill_sources({"cloud_from": np.array([0, 1, 1, 2])}, numbered).tolist(),
                         [0, 1, 1, 2])

    def test_the_fill_as_painted_is_shown_not_counted(self):
        """The fill as the inpainter painted it never reaches the model: what was foreign was left out. A pale grey
        painted into a brass dish on every try is left out whole; the model shows the brass spread into those places
        from round them, one colour, and passes. The as-painted share is shown, and the places count as unpainted, so
        the coverage stays true (as without the touch-up)."""
        views = tempfile.mkdtemp(prefix="touch-up-views-")
        touched, record, sources = touch_up.touched_up(painting((0.85, 0.85, 0.85)), self.corners, self.triangles,
                                                       self.brass, self.brass_cloud, "a candle", report=quiet,
                                                       views_out=views)
        painted = {"colours": self.brass, "cloud_points": touched[0], "cloud_normals": touched[1],
                   "cloud_colours": touched[2], "cloud_from": sources}
        fit, _ = pipeline.judged_fill(views, painted, record, self.corners, self.triangles)
        self.assertEqual(fit["judged"], "as kept")
        self.assertGreater(min(fit["as_painted"]["views"]), touch_up.MOST_FOREIGN, fit)
        self.assertGreater(fit["left_out"]["largest_place"], touch_up.LARGEST_FOREIGN_PATCH, fit)
        self.assertEqual(fit["left_out"]["largest_smear"], 0.0, fit)
        check = inspection.fill_check(fit)
        self.assertTrue(check.passed, fit)
        self.assertIn("shown, not counted", check.words)
        without = bake.baked(self.corners, self.triangles, self.brass, self.brass_cloud, side=512)["coverage"]
        with_it = bake.baked(self.corners, self.triangles, self.brass, touched, side=512)["coverage"]
        self.assertEqual(with_it, without, "a place left out is not counted painted")
        # A record made before the touch-up kept its judgement: the saved tries are judged again, alike.
        older = {**record, "views": [{k: v for k, v in view.items() if k not in ("as_painted", "source")}
                                     for view in record["views"]]}
        replayed = touch_up.replayed_as_painted(self.corners, self.triangles, painted, older, views)
        self.assertEqual(replayed, touch_up.as_painted(record))

    def test_a_lining_painted_another_colour_passes(self):
        """A boot's lining, a shelf's boards, a leg's inner face need not look like the paint round them. A cup whose
        inside the inpainter paints a pale lining: the lining's colour is found nowhere round it, so the touch-up
        leaves it out, and the model shows the cup's own colour spread from its rim and floor. It passes."""
        corners, triangles = a_cup()
        cloud = painted_from_six(corners, triangles, BRASS, grain=0.04)
        colours = np.tile(BRASS, (len(corners), 1))
        views = tempfile.mkdtemp(prefix="touch-up-views-")
        touched, record, sources = touch_up.touched_up(painting(LINING, 0.02), corners, triangles, colours,
                                                       cloud, "a cup", report=quiet, views_out=views)
        self.assertTrue(record["views"])
        painted = {"colours": colours, "cloud_points": touched[0], "cloud_normals": touched[1],
                   "cloud_colours": touched[2], "cloud_from": sources}
        fit, shown = pipeline.judged_fill(views, painted, record, corners, triangles)
        self.assertGreater(max(fit["as_painted"]["views"]), touch_up.MOST_FOREIGN, "the lining is foreign as painted")
        self.assertGreater(fit["left_out"]["largest_place"], touch_up.LARGEST_FOREIGN_PATCH, fit["left_out"])
        self.assertTrue(inspection.fill_check(fit).passed, fit)
        self.assertIsNotNone(shown)

    def test_a_smear_left_out_still_fails(self):
        """Leaving a place out does not mend it. The fifth trial's candle: the inpainter painted a glass cup round the
        candle's foot, which the touch-up left out; the model shows there the colours spread from the wax above and
        the brass dish below, neither one colour: a smear, and the check fails."""
        cloud, colours = wax_and_brass(self.corners, self.triangles)
        views = tempfile.mkdtemp(prefix="touch-up-views-")
        touched, record, sources = touch_up.touched_up(glass_round_the_foot(), self.corners, self.triangles,
                                                       colours, cloud, "a candle in a brass holder", report=quiet,
                                                       views_out=views)
        painted = {"colours": colours, "cloud_points": touched[0], "cloud_normals": touched[1],
                   "cloud_colours": touched[2], "cloud_from": sources}
        fit, shown = pipeline.judged_fill(views, painted, record, self.corners, self.triangles)
        self.assertGreater(fit["left_out"]["largest_smear"], touch_up.LARGEST_FOREIGN_PATCH, fit["left_out"])
        failed = inspection.fill_check(fit)
        self.assertFalse(failed.passed, fit)
        self.assertIn("a smear", failed.words)
        smears = [p for view in fit["left_out"]["views"] for p in view["places"] if p["smear"]]
        self.assertTrue(all(p["ring_one_colour"] < touch_up.ONE_COLOUR and p["inside_one_colour"] < touch_up.ONE_COLOUR
                            for p in smears))
        # Shown in the report's picture: the smear ringed magenta.
        picture = np.asarray(shown[0])
        self.assertTrue((np.abs(picture.astype(int) - (210, 40, 200)).max(-1) == 0).any())

    def test_a_smear_needs_its_ring_and_its_inside_both_mixed(self):
        self.assertTrue(touch_up.is_smear(0.61, 0.69))
        self.assertFalse(touch_up.is_smear(0.6, 1.0), "a place showing one colour spread from a mixed ring")
        self.assertFalse(touch_up.is_smear(1.0, 0.6), "a place ringed by one colour passes")
        many = np.tile(BRASS, (100, 1))
        self.assertEqual(touch_up.one_colour_share(many), 1.0)
        half = np.concatenate([np.tile(BRASS, (50, 1)), np.tile(WAX, (50, 1))])
        self.assertLess(touch_up.one_colour_share(half), touch_up.ONE_COLOUR)
        self.assertEqual(touch_up.one_colour_share(np.zeros((0, 3))), 0.0)

    def test_a_good_fill_passes_as_painted(self):
        views = tempfile.mkdtemp(prefix="touch-up-views-")
        touched, record, sources = touch_up.touched_up(painting(BRASS, 0.03), self.corners, self.triangles,
                                                       self.brass, self.brass_cloud, "a candle", report=quiet,
                                                       views_out=views)
        painted = {"colours": self.brass, "cloud_points": touched[0], "cloud_normals": touched[1],
                   "cloud_colours": touched[2], "cloud_from": sources}
        fit, _ = pipeline.judged_fill(views, painted, record, self.corners, self.triangles)
        self.assertTrue(inspection.fill_check(fit).passed, fit)


    def test_a_place_left_out_ringed_by_one_colour_passes(self):
        """A clay jug's fill: a view whose fill held one foreign place larger than the limit (a shadow
        behind its handle, 7.2 % of the fill and 4.7 % across, as painted), which the touch-up left out. The model shows
        there the clay spread from round it: the place passes, and its as-painted measure is shown."""
        views = tempfile.mkdtemp(prefix="touch-up-views-")
        touched, record, sources = touch_up.touched_up(painting(BRASS, 0.03), self.corners, self.triangles,
                                                       self.brass, self.brass_cloud, "a candle", report=quiet,
                                                       views_out=views)
        painted = {"colours": self.brass, "cloud_points": touched[0], "cloud_normals": touched[1],
                   "cloud_colours": touched[2], "cloud_from": sources}
        one_place = {**record, "views": [dict(view) for view in record["views"]]}
        one_place["views"][0]["as_painted"] = {"foreign": 0.072, "patch": 0.047}
        fit, _ = pipeline.judged_fill(views, painted, one_place, self.corners, self.triangles)
        self.assertEqual(fit["as_painted"]["patches"][0], 0.047)
        self.assertEqual(fit["largest_patch"], 0.0, "judged as kept")
        check = inspection.fill_check(fit)
        self.assertTrue(check.passed, fit)
        self.assertIn("7.2%", check.words)

    def test_another_seed_while_the_kept_try_was_foreign_as_painted(self):
        """The as-painted share is the trigger for more seeds: a fill foreign on every try is painted
        `MOST_FILL_TRIES` times; a fill a little foreign (more than `GOOD_ENOUGH`, under `MOST_FOREIGN`) `FILL_TRIES`."""
        recording = Recording(painting((0.85, 0.85, 0.85)))
        _, record, _ = touch_up.touched_up(recording, self.corners, self.triangles, self.brass, self.brass_cloud,
                                           "a candle", report=quiet, most=1)
        self.assertEqual(record["views"][0]["tries"], touch_up.MOST_FILL_TRIES)
        self.assertEqual(len({seed for _, _, seed in recording.given}), touch_up.MOST_FILL_TRIES, "each try its seed")

        def a_few_specks(picture, mask, text, seed):
            image = np.asarray(painting(BRASS, 0.03)(picture, mask, text, seed)).copy()
            rows, cols = np.nonzero(np.asarray(mask) > 127)
            specks = np.random.default_rng(seed).choice(len(rows), len(rows) // 20, replace=False)
            image[rows[specks], cols[specks]] = (20, 20, 20)
            return Image.fromarray(image)

        _, record, _ = touch_up.touched_up(a_few_specks, self.corners, self.triangles, self.brass, self.brass_cloud,
                                           "a candle", report=quiet, most=1)
        view = record["views"][0]
        self.assertGreater(view["foreign"], touch_up.GOOD_ENOUGH)
        self.assertLessEqual(view["as_painted"]["foreign"], touch_up.MOST_FOREIGN)
        self.assertEqual(view["tries"], touch_up.FILL_TRIES)


if __name__ == "__main__":
    unittest.main()
