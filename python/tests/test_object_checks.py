import unittest

import numpy as np

from model_maker import inspection, object_checks as O, render, words

from . import shapes


def front_mask(corners, triangles, turn=0):
    return render.silhouette(corners, triangles, turn, 512)


class Wholeness(unittest.TestCase):
    def test_a_chair_is_one_closed_piece(self):
        corners, triangles = shapes.chair()
        measures = O.wholeness(corners, triangles)
        self.assertEqual((measures["holes"], measures["pieces"]), (0, 1))
        self.assertTrue(all(c.passed for c in O.whole_checks(measures, weapon=False)))

    def test_a_hole_is_counted(self):
        corners, triangles = shapes.chair()
        measures = O.wholeness(corners, triangles[5:])
        self.assertGreaterEqual(measures["holes"], 1)
        self.assertFalse(O.whole_checks(measures, weapon=False)[0].passed)

    def test_a_floating_pommel_fails_a_weapon(self):
        corners, triangles = shapes.sword(pommel_gap=0.01)
        measures = O.wholeness(corners, triangles)
        self.assertEqual(measures["pieces"], 2)
        self.assertGreater(measures["loose"][0]["share"], O.LEAST_PART_SHARE)
        self.assertFalse(O.whole_checks(measures, weapon=True)[1].passed)


class Standing(unittest.TestCase):
    def test_a_chair_stands_level_with_its_centre_over_its_legs(self):
        corners, triangles = shapes.chair()
        base = O.base_measures(corners, triangles)
        self.assertLess(base["tilt_deg"], 0.5)
        self.assertGreater(base["margin"], 0.4)
        self.assertTrue(base["enclosed"])
        self.assertTrue(all(c.passed for c in O.base_checks(base)))

    def test_a_leg_3_cm_short_does_not_reach_the_ground(self):
        corners, triangles = shapes.chair(short_leg=0.03)
        base = O.base_measures(corners, triangles)
        self.assertEqual(len(base["lifted_m"]), 1)
        self.assertAlmostEqual(base["lifted_m"][0], 0.03, delta=0.013)
        feet, level, centre = O.base_checks(base)
        self.assertFalse(feet.passed)
        # It stands on its other three, its centre toward the back between them.
        self.assertTrue(level.passed and centre.passed)

    def test_a_chest_on_small_feet_has_no_lifted_feet(self):
        feet = [((x, 0.0, z), (x + 0.06, 0.03, z + 0.06)) for x in (-0.4, 0.34) for z in (-0.27, 0.21)]
        body = ((-0.4, 0.03, -0.27), (0.4, 0.5, 0.27))
        corners, triangles = shapes.surface_of(feet + [body], 0.01)
        base = O.base_measures(corners, triangles)
        self.assertEqual(base["lifted_m"], [])
        self.assertTrue(all(c.passed for c in O.base_checks(base)))

    def test_a_ridge_under_the_body_and_a_slight_tilt_are_no_lifted_feet(self):
        feet = [((x, 0.0, z), (x + 0.06, 0.03, z + 0.06)) for x in (-0.4, 0.34) for z in (-0.27, 0.21)]
        body = ((-0.4, 0.03, -0.27), (0.4, 0.5, 0.27))
        ridge = ((-0.3, 0.012, -0.03), (0.3, 0.03, 0.03))
        corners, triangles = shapes.surface_of(feet + [body, ridge], 0.006)
        turn = np.radians(0.5)
        tilted = corners @ np.array([[1, 0, 0], [0, np.cos(turn), -np.sin(turn)], [0, np.sin(turn), np.cos(turn)]])
        base = O.base_measures(O.set_down(tilted, 0.8), triangles)
        self.assertEqual(base["lifted_m"], [])
        self.assertTrue(all(c.passed for c in O.base_checks(base)))

    def test_a_leg_3_mm_short_still_stands(self):
        corners, triangles = shapes.chair(short_leg=0.003, cell=0.003)
        self.assertTrue(all(c.passed for c in O.base_checks(O.base_measures(corners, triangles))))

    def test_a_chair_leaning_past_its_legs_tips_and_settles(self):
        corners, triangles = shapes.chair(lean=0.45)
        _, level, centre = O.base_checks(O.base_measures(corners, triangles))
        self.assertFalse(level.passed)
        self.assertFalse(centre.passed)
        self.assertIn("falls onto another side", centre.words)


class Symmetry(unittest.TestCase):
    def test_a_chair_mirrors_left_to_right(self):
        corners, triangles = shapes.chair()
        share, turn, _, _ = O.mirror_measure(corners, triangles)
        self.assertLess(share, O.MOST_MIRROR_DISTANCE_SHARE)
        # Its mirror plane is upright across X (turn 0) or Z (turn 90): the back is on -Z, so only X.
        self.assertTrue(turn < 3 or turn > 177, turn)

    def test_a_part_missing_out_of_place_or_on_one_side_fails(self):
        # The mean diluted a local fault: a leg missing read 1.67 %, an armrest on one side
        # only 1.97 %, a leg 8 cm out 0.95 %, all under the 2 % limit. A high percentile of the mirrored distances,
        # measured to the surface's triangles, catches each.
        for fault in (dict(missing_leg=True), dict(moved_leg=0.08), dict(armrest=True)):
            corners, triangles = shapes.chair(**fault)
            share, turn, _, far = O.mirror_measure(corners, triangles)
            self.assertLess(share, O.MOST_MIRROR_DISTANCE_SHARE, (fault, "the mean alone misses it"))
            check = O.mirror_check(share, turn, far)
            self.assertFalse(check.passed, (fault, check.words))
        corners, triangles = shapes.chair()
        share, turn, _, far = O.mirror_measure(corners, triangles)
        self.assertTrue(O.mirror_check(share, turn, far).passed)
        self.assertLess(far, O.MOST_MIRROR_FAR_SHARE / 2)

    def test_a_thing_with_a_post_in_one_corner_does_not(self):
        # A post in one corner with an arm reaching out along +X from its top: no upright plane mirrors it.
        tilted = shapes.surface_of([((-0.3, 0.0, -0.3), (0.3, 0.1, 0.3)), ((0.1, 0.1, 0.1), (0.3, 0.9, 0.3)),
                                    ((0.3, 0.75, 0.15), (0.8, 0.85, 0.25))], 0.012)
        share, _, _, _ = O.mirror_measure(*tilted)
        self.assertGreater(share, O.MOST_MIRROR_DISTANCE_SHARE)


def outline_passes(picture, corners, triangles):
    """The outline check's verdict, its view and its values."""
    view, mean, far, apart = O.outline_judged(picture, corners, triangles)
    return O.outline_checks(view, mean, far, apart)[0].passed, view, (mean, far, apart)


class Outline(unittest.TestCase):
    def test_the_shapes_own_outline_matches_from_the_painted_turn(self):
        corners, triangles = shapes.chair()
        mask = front_mask(corners, triangles, turn=30)
        view = O.outline_search(mask, corners, triangles)
        self.assertGreater(view.iou, 0.95)
        self.assertEqual(view.turn, 30)
        self.assertIsNone(view.distance, "a picture drawn straight on is matched straight on")
        self.assertEqual(O.overlay_of(view.painted, view.made).size, (O.OUTLINE_SIDE * 2, O.OUTLINE_SIDE))

    def test_another_shape_does_not_match(self):
        corners, triangles = shapes.chair()
        stool = shapes.surface_of([((-0.2, 0.0, -0.2), (0.2, 0.05, 0.2)), ((-0.05, 0.05, -0.05), (0.05, 0.9, 0.05))],
                                  0.012)
        passed, view, values = outline_passes(front_mask(*stool), corners, triangles)
        self.assertLess(view.iou, O.LEAST_OUTLINE_IOU)
        self.assertFalse(passed, values)

    def test_a_thin_shape_a_pixel_off_its_picture_still_matches_by_its_edges(self):
        corners, triangles = shapes.chair()
        shifted = np.roll(front_mask(corners, triangles, turn=30), 2, axis=1)
        passed, _, values = outline_passes(shifted, corners, triangles)
        self.assertTrue(passed, values)

    def test_an_outline_more_than_twice_as_wide_as_high_is_compared_at_its_own_width(self):
        # A chest-like box 2.2 times as wide as high, against a picture of one 2.5 times: 12 % narrower, a misfit at
        # both ends. Squeezed to a canvas twice as wide as high, both would match.
        def box(width):
            return shapes.surface_of([((-width / 2, 0.0, -0.1), (width / 2, 0.36, 0.1))], 0.01)

        passed, _, values = outline_passes(front_mask(*box(0.9)), *box(0.79))
        self.assertFalse(passed, values)

    def test_a_picture_through_a_lens_matches_its_own_shape_through_the_lens(self):
        corners, triangles = shapes.surface_of([((-0.4, 0.06, -0.24), (0.4, 0.36, 0.24))] +
                                               [((x, 0.0, z), (x + 0.06, 0.06, z + 0.06)) for x in (-0.4, 0.34)
                                                for z in (-0.24, 0.18)], 0.01)
        picture = render.silhouette(corners, triangles, 20, 512, 10, distance=3.0)
        passed, view, values = outline_passes(picture, corners, triangles)
        self.assertEqual(view.distance, 3.0)
        self.assertTrue(passed, values)

    def test_a_web_between_legs_fails_through_any_lens(self):
        # A table's top on three legs, and the same with a web filling the space between two legs under the top: the
        # web's edges lie near the legs' edges, its thickness gives it away.
        top = ((-0.35, 0.6, -0.35), (0.35, 0.65, 0.35))
        legs = [((x, 0.0, -0.02), (x + 0.04, 0.6, 0.02)) for x in (-0.3, 0.26)]
        web = ((-0.26, 0.42, -0.02), (0.26, 0.6, 0.02))
        corners, triangles = shapes.surface_of([top, web] + legs, 0.01)
        passed, _, (mean, far, apart) = outline_passes(front_mask(*shapes.surface_of([top] + legs, 0.01)), corners,
                                                       triangles)
        self.assertGreater(apart, O.MOST_OUTLINE_APART_SHARE, (mean, far, apart))
        self.assertFalse(passed, (mean, far, apart))

    def test_a_long_bench_matches_its_own_picture_from_every_view(self):
        # The search refined only its best coarse view, at that view's camera distance, so
        # a coarse grid that ranked a wrong lens first kept it. A bench 3.3 times as wide as high, pictured from its
        # own surface at 10 degrees round and 12 above straight on, was matched through a lens 3 sizes away (overlap
        # 0.74, edges 2.6 % and 9.4 % apart) and failed; and through a lens 6 sizes away at -30 and 8, matched 4 sizes
        # away at -23 and 11.
        corners, triangles = shapes.bench()
        for turn, tilt, distance in ((10, 12, None), (-30, 8, 6.0), (0, 0, None), (25, 20, None), (40, 24, 4.0)):
            picture = render.silhouette(corners, triangles, turn, 1024, tilt, distance)
            passed, view, (mean, far, apart) = outline_passes(picture, corners, triangles)
            self.assertTrue(passed, (turn, tilt, distance, view.turn, view.tilt, view.distance, mean, far, apart))

    def test_a_plank_wider_than_the_canvas_is_compared_whole(self):
        # A 10:1 plank pictured from the front was cut to the 6:1 canvas, which it filled,
        # and "matched" a view of the plank nearly end-on (cut to fill it too) with an overlap of 1.000. Now the canvas
        # is as wide as the picture needs: its own picture matches it from the front, and a plank 30 % shorter fails.
        def plank(length):
            return shapes.surface_of([((-length / 2, 0.0, -0.04), (length / 2, 0.15, 0.04))], 0.01)

        corners, triangles = plank(1.5)
        picture = render.silhouette(corners, triangles, 0, 2048)
        passed, view, values = outline_passes(picture, corners, triangles)
        self.assertTrue(passed, values)
        self.assertLess(abs(view.turn), 5, view.turn)
        self.assertGreater(view.painted.shape[1], O.OUTLINE_SIDE * 10)
        self.assertFalse(outline_passes(picture, *plank(1.05))[0])

    def test_a_speck_the_picture_checks_let_through_does_not_fail_an_exact_shape(self):
        # A speck of 1 % of the cut-out above the object passes "one piece" (99 %), and the
        # outline was judged against the raw cut-out, speck and all: the exact chair failed (mean 1.66 %, 95th 5.7 %).
        # It is judged against the cut-out the shape model was given (`cutout.for_shape`: the largest piece).
        corners, triangles = O.set_down(shapes.chair()[0], 0.95), shapes.chair()[1]
        picture = front_mask(corners, triangles, 30)
        rows, cols = np.flatnonzero(picture.any(1)), np.flatnonzero(picture.any(0))
        side = int(np.sqrt(0.01 * picture.sum()))
        speckled = picture.copy()
        speckled[max(0, rows[0] - side - 20):rows[0] - 20, cols[0]:cols[0] + side] = True
        self.assertTrue(speckled[:rows[0] - 10].any())
        subject = words.subject_of("an old looking chair")
        checks, measures, _ = inspection.shape_checks(subject, speckled, corners, triangles)
        outline = next(c for c in checks if c.name == "outline matches the picture")
        self.assertTrue(outline.passed, outline.words)
        self.assertLess(measures["outline"]["edge_95"], 0.01)


class AllTogether(unittest.TestCase):
    def test_a_chair_from_its_own_outline_passes_every_shape_check(self):
        corners, triangles = shapes.chair()
        corners = O.set_down(corners, 0.95)
        subject = words.subject_of("an old looking chair")
        checks, measures, _ = inspection.shape_checks(subject, front_mask(corners, triangles, 15), corners, triangles)
        self.assertTrue(all(c.passed for c in checks), [c.words for c in checks if not c.passed])
        self.assertAlmostEqual(max(measures["size_m"]), 0.95, places=3)

    def test_a_weapon_made_at_a_slant_is_set_to_its_length(self):
        # The box's largest side was scaled to the size, so a spear the shape
        # model made at 45 degrees came out 1.4 times its length.
        from model_maker import weapon_checks as W

        corners, triangles = shapes.spear(2.0, 0.03)
        turn = np.radians(45)
        slanted = corners @ np.array([[np.cos(turn), -np.sin(turn), 0], [np.sin(turn), np.cos(turn), 0], [0, 0, 1]]).T
        measures, _ = W.measured(O.set_down(slanted, 2.0, along_length=True), triangles, words.subject_of("a spear"))
        self.assertAlmostEqual(measures["length_m"], 2.0, delta=0.02)
        measures, _ = W.measured(O.set_down(slanted, 2.0), triangles, words.subject_of("a spear"))
        self.assertGreater(measures["length_m"], 2.7, "by the box, as before")

    def test_set_down_stands_it_on_the_ground_at_its_size(self):
        corners, _ = shapes.chair()
        placed = O.set_down(corners * 3 + 7, 0.5)
        self.assertAlmostEqual(float(placed[:, 1].min()), 0.0)
        self.assertAlmostEqual(float(np.ptp(placed, axis=0).max()), 0.5)


if __name__ == "__main__":
    unittest.main()
