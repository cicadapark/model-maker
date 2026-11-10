import unittest

import numpy as np

from model_maker import inspection, weapon_checks as W, words

from . import shapes

SWORD = words.subject_of("an old iron sword")
AXE = words.subject_of("a double-bitted battle axe")


def named(checks):
    return {c.name: c for c in checks}


class Sword(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.measures, cls.checks = W.measured(*shapes.sword(), SWORD)

    def test_a_straight_sword_passes_its_line_grip_and_symmetry(self):
        failed = [c.words for c in self.checks if not c.passed]
        self.assertFalse(failed)
        self.assertIn("blade straight", named(self.checks))
        self.assertIn("symmetric across the blade", named(self.checks))

    def test_its_length_is_measured(self):
        self.assertAlmostEqual(self.measures["length_m"], 0.95, delta=0.01)

    def test_its_grip_is_the_hilt_between_guard_and_pommel(self):
        grip = self.measures["grip"]
        hold_height = grip["hold"][1]
        self.assertTrue(0.06 < hold_height < 0.16, hold_height)
        self.assertLess(grip["thickness_m"], 0.035)

    def test_its_tip_is_the_blades_point_and_the_grip_points_to_it(self):
        self.assertAlmostEqual(self.measures["tip"][1], 0.95, delta=0.01)
        self.assertGreater(self.measures["toward_tip"][1], 0.99)

    def test_a_bent_blade_fails_straight(self):
        _, checks = W.measured(*shapes.sword(bend=0.02), SWORD)
        found = named(checks)
        self.assertFalse(found["blade straight"].passed)
        self.assertFalse(found["no kink"].passed)


class Rapier(unittest.TestCase):
    """A rapier whose blade comes out 2 cm by 1 cm is round enough to hold; the longest
    such stretch was taken for its grip (83 cm along the blade), its tip mark at the pommel and pointing down."""

    def test_its_grip_is_beside_its_guard_and_its_tip_the_blades_point(self):
        measures, checks = W.measured(*shapes.rapier(), words.subject_of("a thin-bladed rapier"))
        self.assertTrue(named(checks)["grip found"].passed, named(checks)["grip found"].words)
        hold_height = measures["grip"]["hold"][1]
        self.assertTrue(0.05 < hold_height < 0.17, hold_height)
        self.assertLessEqual(abs(measures["grip"]["to"][1] - measures["grip"]["from"][1]), W.BLADE_GRIP_MOST_M + 0.01)
        self.assertAlmostEqual(measures["tip"][1], 1.05, delta=0.01)
        self.assertGreater(measures["toward_tip"][1], 0.99)

    def test_a_blade_with_no_hilt_has_no_grip(self):
        measures, checks = W.measured(*shapes.rapier(hilt=False), words.subject_of("a thin-bladed rapier"))
        grip = named(checks)["grip found"]
        self.assertFalse(grip.passed, grip.words)
        self.assertIn("blade", grip.words)


class Shaft(unittest.TestCase):
    """A perfectly round shaft read 3.57 % of its width off itself mirrored (3 m by 3 cm),
    the samples' own spacing (about 1 mm) over a 3 cm width, and failed the 3 % limit."""

    def test_a_perfectly_round_spear_is_symmetric_across_its_shaft(self):
        spear = words.subject_of("a spear")
        for length, width in ((3.0, 0.03), (3.0, 0.04), (2.0, 0.03), (1.9, 0.035)):
            measures, checks = W.measured(*shapes.spear(length, width), spear)
            check = named(checks)["symmetric across the shaft"]
            self.assertTrue(check.passed, (length, width, check.words))
            self.assertLess(measures["across_share"], W.MOST_ACROSS_SHARE / 3, (length, width))

    def test_a_lug_on_one_side_is_not(self):
        measures, checks = W.measured(*shapes.spear(2.0, 0.03, lug=((0.015, 1.6), (0.065, 1.75))),
                                      words.subject_of("a spear"))
        self.assertFalse(named(checks)["symmetric across the shaft"].passed, measures["across_share"])


class SquareHaft(unittest.TestCase):
    def test_a_square_haft_is_as_thick_turned_as_not(self):
        # Widths along the frame's axes made a 4 cm square haft turned 45
        # degrees 5.7 cm across, too thick for a grip, and the same haft unturned 4 cm.
        hammer = words.subject_of("a war hammer")
        found = []
        for degrees in (0.0, 45.0, 30.0):
            corners, triangles = shapes.surface_of([((-0.02, 0.0, -0.02), (0.02, 0.8, 0.02)),
                                                    ((-0.08, 0.68, -0.03), (0.08, 0.78, 0.03))], 0.004)
            t = np.radians(degrees)
            turned = corners @ np.array([[np.cos(t), 0, -np.sin(t)], [0, 1, 0], [np.sin(t), 0, np.cos(t)]]).T
            measures, checks = W.measured(turned, triangles, hammer)
            self.assertTrue(named(checks)["grip found"].passed, (degrees, named(checks)["grip found"].words))
            found.append(measures["grip"]["thickness_m"])
        self.assertLess(max(found) - min(found), 0.003, found)


class Axe(unittest.TestCase):
    def test_a_double_axe_has_a_straight_haft_a_grip_at_its_butt_and_its_head_as_the_tip(self):
        measures, checks = W.measured(*shapes.double_axe(), AXE)
        failed = [c.words for c in checks if not c.passed]
        self.assertFalse(failed)
        self.assertLess(measures["grip"]["hold"][1], 0.2)
        self.assertGreater(measures["tip"][1], 0.85)
        self.assertIn("symmetric across the haft", named(checks))

    def test_a_bent_haft_fails_straight_and_kink(self):
        _, checks = W.measured(*shapes.double_axe(kink_deg=6.0), AXE)
        found = named(checks)
        self.assertFalse(found["haft straight"].passed, found["haft straight"].words)
        self.assertFalse(found["no kink"].passed, found["no kink"].words)

    def test_a_beard_beside_the_haft_is_not_read_as_haft(self):
        # A remade axe: a slice's largest piece under the head was a corner of the head, not the haft, and a haft
        # straight within 4 mm failed. A bearded axe's beard is a larger piece than the haft beside it, and narrow.
        bearded = words.subject_of("a bearded axe")
        measures, checks = W.measured(*shapes.bearded_axe(), bearded)
        found = named(checks)
        self.assertTrue(found["haft straight"].passed, found["haft straight"].words)
        self.assertTrue(found["no kink"].passed, found["no kink"].words)
        self.assertLess(measures["bend_m"], 0.002)
        self.assertTrue(found["grip found"].passed, found["grip found"].words)
        self.assertLess(abs(measures["grip"]["hold"][0]), 0.005, "the grip is on the haft's line")

    def test_the_grip_is_on_the_haft_above_a_butt_cap(self):
        # A remade axe: the grip mark sat 14 cm from the end on the butt piece's neck, thin enough to hold.
        measures, checks = W.measured(*shapes.double_axe(butt_cap=True), AXE)
        self.assertFalse([c.words for c in checks if not c.passed])
        grip = measures["grip"]
        collar_top = 0.14
        self.assertGreater(min(grip["from"][1], grip["to"][1]), collar_top - 0.01, grip)
        self.assertGreater(grip["hold"][1], collar_top + 0.05, grip)
        self.assertLess(grip["hold"][1], collar_top + 0.12, grip)


class Shield(unittest.TestCase):
    SHIELD = words.subject_of("a round wooden shield with an iron boss")

    def measured(self, **args):
        measures, checks = W.measured(*shapes.shield(**args), self.SHIELD)
        return measures, named(checks)["handle found"]

    def test_with_two_handles_the_hand_holds_the_one_on_the_bearers_right(self):
        # A remade shield: two D-handles left and right of the middle, the grip on the boards between them.
        measures, check = self.measured()
        self.assertTrue(check.passed, check.words)
        self.assertFalse(check.counted)
        self.assertEqual(len(measures["handles"]), 2)
        grip = measures["grip"]
        self.assertAlmostEqual(grip["hold"][0], -0.15, delta=0.01)
        self.assertAlmostEqual(grip["hold"][1], 0.0, delta=0.01)
        self.assertLess(grip["hold"][2], -0.035, "on the bar, off the board")
        self.assertAlmostEqual(abs(grip["to"][1] - grip["from"][1]), 0.18, delta=0.02)
        self.assertFalse(grip["by_convention"])
        self.assertGreater(measures["toward_tip"][1], 0.99, "the fingers close round the bar")

    def test_handles_one_above_the_other_are_held_by_the_upper(self):
        measures, check = self.measured(handles=((0.0, -0.15), (0.0, 0.15)), across=True)
        self.assertTrue(check.passed, check.words)
        self.assertAlmostEqual(measures["grip"]["hold"][1], 0.15, delta=0.01)
        self.assertGreater(measures["toward_tip"][0], 0.99, "a bar lying across: the fingers close round +X")

    def test_one_handle_is_held(self):
        measures, check = self.measured(handles=((0.05, 0.02),))
        self.assertTrue(check.passed, check.words)
        self.assertAlmostEqual(measures["grip"]["hold"][0], 0.05, delta=0.01)

    def test_no_handle_is_said_not_found(self):
        for args in ({"handles": ()}, {"solid": True}):
            measures, check = self.measured(**args)
            self.assertFalse(check.passed, args)
            self.assertFalse(check.counted)
            self.assertIn("no handle", check.words)
            self.assertEqual(measures["grip"]["handle"], "not found")
            self.assertTrue(measures["grip"]["by_convention"])


class BowString(unittest.TestCase):
    def test_a_curved_bow_gets_a_string_between_its_tips_on_its_inner_side(self):
        corners, triangles = shapes.bow()
        string = W.bow_string(corners, triangles)
        self.assertIn("corners", string)
        start, end = np.array(string["from"]), np.array(string["to"])
        self.assertAlmostEqual(abs(end[1] - start[1]), 1.4, delta=0.03)
        # The tips are at x 0; the curve bulges to +x: the string runs straight between the tips, inside the curve.
        self.assertLess(abs((start[0] + end[0]) / 2), 0.03)
        self.assertAlmostEqual(string["brace_m"], 0.18, delta=0.03)
        from model_maker import mesh_tools as M
        self.assertGreater(M.volume_and_centre(string["corners"], string["triangles"])[0], 0)
        self.assertEqual(M.open_edge_loops(string["triangles"])[:2], (0, 0))

    def test_a_bow_curving_both_ways_gets_none(self):
        string = W.bow_string(*shapes.bow(s_shape=True))
        self.assertNotIn("corners", string)
        self.assertIn("both ways", string["why"])

    def test_a_waving_bow_gets_none(self):
        boxes = []
        for k in range(70):
            y = 0.02 * k
            x = 0.18 * abs(np.sin(2 * np.pi * y / 1.4))
            boxes.append(((x - 0.015, y, -0.015), (x + 0.015, y + 0.025, 0.015)))
        string = W.bow_string(*shapes.surface_of(boxes, 0.006))
        self.assertNotIn("corners", string)
        self.assertIn("waves", string["why"])

    def test_a_straight_stave_gets_none(self):
        string = W.bow_string(*shapes.bow(bend=0.0))
        self.assertNotIn("corners", string)
        self.assertIn("too straight", string["why"])


class Marks(unittest.TestCase):
    def test_the_export_marks_the_grip_line_and_the_tip(self):
        measures, _ = W.measured(*shapes.sword(), SWORD)
        marks = inspection.marks_of({"weapon": measures})
        self.assertEqual(set(marks), {"grip", "grip_start", "grip_end", "tip"})
        start, end = np.array(marks["grip_start"][0]), np.array(marks["grip_end"][0])
        self.assertGreater(np.linalg.norm(end - start), W.GRIP_LEAST_LENGTH_M - 0.01)


if __name__ == "__main__":
    unittest.main()
