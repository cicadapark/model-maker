import unittest

import numpy as np

from model_maker import carving, inspection, mesh_tools as M, object_checks as O, render, words

from . import shapes


def picture_of(corners, triangles):
    """The front view's outline, as a picture's cut-out."""
    return render.silhouette(corners, triangles, 0, 512)


def outline(mask, corners, triangles):
    checks, measures, _ = inspection.shape_checks(words.subject_of("a clay mug"), mask, corners, triangles)
    return next(c for c in checks if c.name == "outline matches the picture"), measures["outline"]


class Carved(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mask = picture_of(*shapes.mug())

    def carved_and_judged(self, web):
        corners, triangles = shapes.mug(web)
        before, _ = outline(self.mask, corners, triangles)
        carved_corners, carved_triangles, record = carving.carved(self.mask, corners, triangles)
        after, measured = outline(self.mask, carved_corners, carved_triangles)
        return before, after, measured, record, (carved_corners, carved_triangles), (corners, triangles)

    def assert_whole(self, corners, triangles):
        open_edges, _, torn = M.open_edge_loops(triangles)
        self.assertEqual((open_edges, torn), (0, 0))
        self.assertEqual(M.pieces(corners, triangles)[1], 1)
        self.assertGreater(M.volume_and_centre(corners, triangles)[0], 0)

    def test_a_web_across_part_of_the_loop_is_cut_and_the_cut_closed(self):
        before, after, measured, record, carved, made = self.carved_and_judged("lower")
        self.assertFalse(before.passed, before.words)
        self.assertTrue(record["carved"], record)
        self.assertTrue(after.passed, after.words)
        self.assert_whole(*carved)
        self.assertLess(measured["apart"], O.MOST_OUTLINE_APART_SHARE / 2)

    def test_a_web_across_the_whole_loop_is_cut_and_both_faces_joined(self):
        before, after, measured, record, carved, _ = self.carved_and_judged("whole")
        self.assertFalse(before.passed, before.words)
        self.assertTrue(record["carved"], record)
        self.assertTrue(after.passed, after.words)
        self.assert_whole(*carved)

    def test_the_carved_loop_is_open_seen_from_the_front(self):
        _, _, _, _, carved, _ = self.carved_and_judged("lower")
        seen = picture_of(*carved)
        # The loop's middle (x 0.125, y 0.10, in the lower part the web was) shows through.
        framing = render.Framing(carved[0], 0, 512)
        (col, row), = framing.pixels(np.array([[0.125, 0.10, 0.0]]))[0]
        self.assertFalse(seen[int(row), int(col)])

    def test_an_open_loop_is_left_as_made(self):
        corners, triangles = shapes.mug()
        carved_corners, carved_triangles, record = carving.carved(self.mask, corners, triangles)
        self.assertFalse(record["carved"], record)
        self.assertEqual(len(carved_triangles), len(triangles))

    def test_material_as_deep_as_the_body_is_never_carved(self):
        before, after, _, record, carved, made = self.carved_and_judged("solid")
        self.assertFalse(record["carved"], record)
        self.assertIn("body, not a web", record["words"])
        self.assertEqual(len(carved[1]), len(made[1]))
        self.assertFalse(after.passed)

    def test_a_chair_has_no_hole_to_carve(self):
        corners, triangles = shapes.chair()
        mask = picture_of(corners, triangles)
        _, carved_triangles, record = carving.carved(mask, corners, triangles)
        self.assertFalse(record["carved"])
        self.assertEqual(len(carved_triangles), len(triangles))

    def test_the_record_shows_beside_the_checks_uncounted(self):
        _, _, _, record, _, _ = self.carved_and_judged("lower")
        check = carving.carve_check(record)
        self.assertFalse(check.counted)
        self.assertEqual(check.value, 1)


class Stitched(unittest.TestCase):
    def test_runs_shorter_than_a_tenth_of_an_opening_join_their_neighbours(self):
        sides = [True] * 12 + [False] + [True] * 3 + [False] * 16 + [True] * 2 + [False] * 6
        runs = carving.smoothed_runs(sides)
        self.assertEqual(len(runs), 2)

    def test_a_slot_is_closed_by_a_strip_running_the_loop_s_way(self):
        # A slot seen from the side: the front edge 0..4 along z = 0, the back edge back along z = -1.
        corners = np.array([[x, 0.0, 0.0] for x in range(5)] + [[x, 0.0, -1.0] for x in range(4, -1, -1)], float)
        loop = list(range(10))
        walls = carving.zipped(corners, loop, [(0, 4, True), (4, 6, False)])
        self.assertEqual(len(walls), 8)
        edges = {(a, b) for t in walls for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
        for k in range(10):
            self.assertIn((loop[k], loop[(k + 1) % 10]), edges, "every loop edge used the loop's way")



def coloured(shown, cut=None, side=1024):
    """A picture as the regression's made-up cases draw it: the object grey on a pale backdrop, what the cut-out
    left out of it (`cut`) in a blue of its own."""
    picture = np.where(shown[..., None], 110, 214).astype(np.uint8).repeat(3, -1)
    if cut is not None:
        picture[shown & ~cut] = (60, 90, 160)
    return picture


def holes(picture_shape, model_shape, turn=0, tilt=0):
    corners, triangles = model_shape
    shown = render.silhouette(*picture_shape, turn, 1024, tilt)
    return carving.holes_open(shown, corners, triangles, coloured(shown))


class HolesOpen(unittest.TestCase):
    """A jug's handle filled: a hole the picture shows, filled in the model, under the carving's 7.8 %
    and the outline's 8 % gap piece, was caught by nothing. Each hole of the picture with an inner part is judged by
    how much of it the model covers; a hole the model has and the picture lacks too."""

    def test_a_ring_filled_by_a_web_fails_and_open_passes(self):
        filled = holes(shapes.ringed_box(), shapes.ringed_box(web=True))
        self.assertEqual(filled["judged"], 1, filled)
        self.assertFalse(carving.holes_check(filled).passed, filled)
        opened = holes(shapes.ringed_box(), shapes.ringed_box())
        self.assertTrue(carving.holes_check(opened).passed, opened)
        self.assertLess(opened["worst"], carving.MOST_HOLE_COVERED_SHARE / 2)

    def test_a_panel_between_a_chairs_spindles_fails(self):
        panel = holes(shapes.spindle_chair(), shapes.spindle_chair(panel=True), 20, 12)
        self.assertGreaterEqual(panel["judged"], 3, panel)
        self.assertFalse(carving.holes_check(panel).passed)
        spindles = holes(shapes.spindle_chair(), shapes.spindle_chair(), 20, 12)
        self.assertTrue(carving.holes_check(spindles).passed, spindles)

    def test_a_hole_the_picture_lacks_fails(self):
        extra = holes(shapes.ringed_box(web=True), shapes.ringed_box())
        self.assertEqual(extra["judged"], 0)
        check = carving.holes_check(extra)
        self.assertTrue(check.counted)
        self.assertFalse(check.passed, extra)

    def test_a_hole_too_small_is_not_judged_and_not_passed(self):
        # A box with a loop on its side whose hole is 6 mm high (2.4 % of the height): no inner part.
        mug = shapes.surface_of([((-0.1, 0.0, -0.1), (0.1, 0.25, 0.1)), ((0.1, 0.17, -0.01), (0.124, 0.176, 0.01)),
                                 ((0.1, 0.182, -0.01), (0.124, 0.188, 0.01)),
                                 ((0.118, 0.17, -0.01), (0.124, 0.188, 0.01))], 0.002)
        found = holes(mug, mug)
        self.assertTrue(found["holes"], "the picture has a hole")
        self.assertTrue(all(not h["judged"] for h in found["holes"]))
        self.assertIn("not judged", found["holes"][0]["words"])
        self.assertFalse(carving.holes_check(found).counted)

    def test_a_thin_panel_the_cut_out_missed_is_not_carved(self):
        # A panel BiRefNet cuts out (a mirror's glass) is as thin as a web; the hole's
        # colour in the picture tells: the backdrop's for a hole, the panel's own for a panel.
        corners, triangles = O.set_down(*shapes.framed_panel()[:1], 0.8), shapes.framed_panel()[1]
        frame = render.silhouette(O.set_down(shapes.framed_panel(panel=False)[0], 0.8),
                                  shapes.framed_panel(panel=False)[1], 0, 1024)
        shown = render.silhouette(corners, triangles, 0, 1024)
        _, _, record = carving.carved(frame, corners, triangles, coloured(shown, frame))
        self.assertFalse(record["carved"], record)
        self.assertIn("not a hole", record["words"])
        _, _, as_hole = carving.carved(frame, corners, triangles, coloured(frame))
        self.assertTrue(as_hole["carved"], "the same, its hole showing the backdrop, is a web")


class Ranked(unittest.TestCase):
    def test_a_carved_try_ranks_below_an_uncarved_one_passing_as_many(self):
        checks = [inspection.Check("a", 1, 1, "", True, ""), inspection.Check("b", 1, 1, "", False, "")]
        carved = inspection.score(checks, {"carved": {"carved": True}, "outline": {"iou": 0.95}})
        uncarved = inspection.score(checks, {"carved": {"carved": False}, "outline": {"iou": 0.90}})
        self.assertGreater(uncarved, carved)


if __name__ == "__main__":
    unittest.main()
