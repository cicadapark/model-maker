import unittest

import numpy as np

from model_maker import inspection, lying, picture_checks, upright, weapon_checks as W, words

from . import shapes

MACE = words.subject_of("an iron mace")
SWORD = words.subject_of("an old iron sword")
UP = np.array([0.0, 1.0, 0.0])


def mace(cell=0.004):
    """A mace 0.6 m long: a haft 3 cm thick, a head 12 cm across at its top."""
    haft = ((-0.015, 0.0, -0.015), (0.015, 0.50, 0.015))
    head = ((-0.06, 0.48, -0.06), (0.06, 0.60, 0.06))
    return shapes.surface_of([haft, head], cell)


def turned_about(corners, axis, degrees):
    """The corners turned `degrees` about the unit `axis` through their middle (a picture painted tilted)."""
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    a = np.radians(degrees)
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rotation = np.eye(3) + np.sin(a) * cross + (1 - np.cos(a)) * cross @ cross
    middle = corners.mean(0)
    return (corners - middle) @ rotation.T + middle


def haft_and_head(corners, triangles):
    """The haft's direction from butt to head, measured as upright measures it, and the head's height over the butt's."""
    head, _ = upright.head_direction(corners, triangles)
    return head


class Upright(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.made = mace()

    def standing(self, corners, triangles):
        stood, _, record = lying.laid(MACE, corners, triangles)
        head = haft_and_head(stood, triangles)
        return stood, record, float(np.degrees(np.arccos(np.clip(head @ UP, -1, 1))))

    def test_an_upright_mace_hardly_moves(self):
        stood, record, off = self.standing(*self.made)
        self.assertTrue(record["upright"])
        self.assertLess(record["turned_deg"], 1.0)
        self.assertLess(off, 1.0)

    def test_a_tilted_mace_stands_its_haft_upright_head_up(self):
        for degrees, axis in ((40, (0, 0, 1)), (70, (1, 0, 0)), (25, (1, 0, 1))):
            corners = turned_about(self.made[0], axis, degrees)
            stood, record, off = self.standing(corners, self.made[1])
            self.assertAlmostEqual(record["turned_deg"], degrees, delta=2.0, msg=(degrees, axis))
            self.assertLess(off, 2.0, (degrees, axis))
            # Head up: the widest end is at the top, the butt on the floor.
            top = stood[stood[:, 1] > stood[:, 1].max() - 0.05]
            self.assertGreater(np.ptp(top[:, 0]), 0.1, (degrees, axis))
            self.assertAlmostEqual(stood[:, 1].min(), 0.0, places=6)

    def test_a_mace_painted_head_down_is_righted(self):
        corners = turned_about(self.made[0], (0, 0, 1), 180)
        stood, record, off = self.standing(corners, self.made[1])
        self.assertAlmostEqual(record["turned_deg"], 180, delta=2.0)
        self.assertLess(off, 2.0)

    def test_the_grip_and_tip_stay_where_they_were_along_the_haft(self):
        tilted = turned_about(self.made[0], (0, 0, 1), 40)
        before, _ = W.measured(tilted, self.made[1], MACE)
        stood, _, record = lying.laid(MACE, tilted, self.made[1])
        after, _ = W.measured(stood, self.made[1], MACE)
        self.assertAlmostEqual(before["grip"]["from_butt_m"], after["grip"]["from_butt_m"], delta=0.002)
        self.assertAlmostEqual(before["length_m"], after["length_m"], delta=0.002)
        # The marks are the same points of the weapon, turned with it.
        back = lying.to_made(np.array([after["grip"]["hold"], after["tip"]]), record)
        self.assertLess(np.linalg.norm(back[0] - before["grip"]["hold"]), 0.003)
        self.assertLess(np.linalg.norm(back[1] - before["tip"]), 0.003)

    def test_the_outline_is_still_judged_in_the_pictures_frame(self):
        tilted = turned_about(self.made[0], (0, 0, 1), 40)
        stood, _, record = lying.laid(MACE, tilted, self.made[1])
        self.assertTrue(np.allclose(lying.to_made(stood, record), tilted, atol=1e-6))

    def test_only_hafted_weapons_are_turned(self):
        sword = shapes.sword()
        corners = turned_about(sword[0], (0, 0, 1), 30)
        same, _, record = lying.laid(SWORD, corners, sword[1])
        self.assertFalse(record["laid"])
        self.assertTrue(np.array_equal(same, corners))

    def test_the_turn_is_shown_not_counted(self):
        _, _, record = lying.laid(MACE, turned_about(self.made[0], (0, 0, 1), 40), self.made[1])
        check = lying.lay_check(record)
        self.assertEqual(check.name, "haft upright")
        self.assertFalse(check.counted)


class Ranking(unittest.TestCase):
    def bar(self, degrees):
        side = 200
        rows, cols = np.mgrid[0:side, 0:side]
        a = np.radians(degrees)
        along = (cols - side / 2) * np.sin(a) + (rows - side / 2) * np.cos(a)
        across = (cols - side / 2) * np.cos(a) - (rows - side / 2) * np.sin(a)
        return (np.abs(along) < 80) & (np.abs(across) < 6)

    def test_the_lean_of_a_cut_out(self):
        self.assertLess(upright.lean_deg(self.bar(0)), 1.0)
        self.assertAlmostEqual(upright.lean_deg(self.bar(40)), 40, delta=1.0)
        self.assertAlmostEqual(upright.lean_deg(self.bar(-70)), 70, delta=1.0)

    def result(self, score, lean):
        found = picture_checks.PictureResult(score=score)
        found.checks.append(upright.standing_check(self.bar(lean)))
        return found

    def test_an_upright_try_is_chosen_before_a_better_scored_tilted_one(self):
        tries = [self.result(0.95, 40), self.result(0.80, 5)]
        self.assertEqual(picture_checks.best_try(tries), 1)

    def test_a_tilted_try_is_never_refused(self):
        tries = [self.result(0.9, 40)]
        self.assertTrue(tries[0].passed)
        self.assertEqual(picture_checks.best_try(tries), 0)
        failing = picture_checks.PictureResult(score=0.99)
        from model_maker.picture_checks import Check

        failing.checks.append(Check("one piece", 50, 97, "%", False, "in pieces"))
        self.assertEqual(picture_checks.best_try([failing, self.result(0.5, 60)]), 1)


class Finish(unittest.TestCase):
    def test_the_finish_judges_a_tilted_mace_upright(self):
        made = mace()
        corners = turned_about(made[0], (0, 0, 1), 40)
        corners[:, 1] -= corners[:, 1].min()
        mask = np.zeros((256, 256), dtype=bool)
        mask[40:220, 120:136] = True
        judged = inspection.judged(MACE, mask, corners, made[1])
        self.assertTrue(judged.laid["upright"])
        self.assertIn("haft upright", {c.name for c in judged.checks})
        head, _ = upright.head_direction(judged.corners, judged.triangles)
        self.assertLess(float(np.degrees(np.arccos(np.clip(head @ UP, -1, 1)))), 2.0)


if __name__ == "__main__":
    unittest.main()
