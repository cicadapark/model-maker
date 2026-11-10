import unittest

import numpy as np

from model_maker import levelling, mesh_tools as M, object_checks as O

from . import shapes


def turned(corners, degrees, axis):
    """The corners turned whole about the X (axis 0) or Z (axis 2) axis through the origin."""
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    if axis == 0:
        matrix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return corners @ matrix.T


def base_passes(corners, triangles):
    return all(c.passed for c in O.base_checks(O.base_measures(corners, triangles)))


class Cut(unittest.TestCase):
    def test_a_box_cut_across_stays_closed_and_stands_on_the_cut(self):
        corners, triangles = shapes.surface_of([((0.0, 0.0, 0.0), (0.3, 0.2, 0.3))], 0.01)
        volume_before, _ = M.volume_and_centre(corners, triangles)
        cut_corners, cut_triangles = levelling.cut_below(corners, triangles, 0.05)
        self.assertEqual(M.open_edge_loops(cut_triangles)[0], 0)
        self.assertAlmostEqual(float(cut_corners[:, 1].min()), 0.0, places=9)
        self.assertAlmostEqual(float(cut_corners[:, 1].max()), float(corners[:, 1].max()) - 0.05, places=6)
        volume_after, _ = M.volume_and_centre(cut_corners, cut_triangles)
        self.assertAlmostEqual(volume_after / volume_before, 0.75, delta=0.03)
        normals, _ = M.face_normals_and_areas(cut_corners, cut_triangles)
        on_floor = np.abs(cut_corners[cut_triangles].mean(1)[:, 1]) < 1e-9
        self.assertTrue((normals[on_floor][:, 1] < -0.99).all(), "the cap faces down")

    def test_legs_cut_across_each_get_a_cap(self):
        corners, triangles = shapes.three_legged_table()
        cut_corners, cut_triangles = levelling.cut_below(corners, triangles, 0.02)
        self.assertEqual(M.open_edge_loops(cut_triangles)[0], 0)
        self.assertEqual(M.pieces(cut_corners, cut_triangles)[1], 1)


    def test_a_cut_foot_ring_is_capped_as_a_ring(self):
        # Each loop the cut leaves was capped on its own, so a foot ring's
        # outer and inner loops became two discs, one over the other, a sheet of no thickness across its hollow.
        body = ((-0.1, 0.03, -0.1), (0.1, 0.2, 0.1))
        ring = ((-0.08, 0.0, -0.08), (0.08, 0.03, 0.08))
        hollow = ((-0.055, -0.01, -0.055), (0.055, 0.029, 0.055))
        corners, triangles = shapes.surface_of([body, ring], 0.005, cut=[hollow])
        cut_corners, cut_triangles = levelling.cut_below(corners, triangles, 0.012)
        self.assertEqual(M.open_edge_loops(cut_triangles)[:3:2], (0, 0))
        middles = cut_corners[cut_triangles].mean(1)
        on_floor = np.abs(middles[:, 1]) < 1e-9
        self.assertTrue(on_floor.any())
        across_hollow = on_floor & (np.abs(middles[:, 0]) < 0.05) & (np.abs(middles[:, 2]) < 0.05)
        self.assertFalse(across_hollow.any(), "no cap across the ring's hollow")
        normals, _ = M.face_normals_and_areas(cut_corners, cut_triangles)
        self.assertTrue((normals[on_floor][:, 1] < -0.99).all(), "the ring's cap faces down")


class Levelled(unittest.TestCase):
    def test_a_table_with_a_leg_2_5_cm_short_is_trimmed_to_stand_level(self):
        corners, triangles = shapes.three_legged_table(short_leg=0.025)
        self.assertFalse(base_passes(corners, triangles))
        level_corners, level_triangles, record = levelling.levelled(corners, triangles)
        self.assertTrue(record["trimmed"], record)
        self.assertAlmostEqual(record["trimmed_m"], 0.025, delta=0.011)
        self.assertTrue(base_passes(level_corners, level_triangles), record)
        self.assertEqual(M.open_edge_loops(level_triangles)[0], 0)
        self.assertAlmostEqual(float(np.ptp(level_corners[:, 1])), 0.65 - record["trimmed_m"], delta=0.012)
        self.assertGreater(record["before"]["tilt_deg"], 2.0)

    def test_a_chair_with_a_leg_3_cm_short_is_trimmed_so_every_foot_reaches_the_ground(self):
        corners, triangles = shapes.chair(short_leg=0.03)
        level_corners, level_triangles, record = levelling.levelled(corners, triangles)
        self.assertTrue(record["trimmed"], record)
        base = O.base_measures(level_corners, level_triangles)
        self.assertEqual(base["lifted_m"], [])
        self.assertTrue(base_passes(level_corners, level_triangles))

    def test_a_leg_beyond_the_limit_is_left_as_made_and_still_fails(self):
        corners, triangles = shapes.three_legged_table(short_leg=0.06)
        level_corners, level_triangles, record = levelling.levelled(corners, triangles)
        self.assertFalse(record["trimmed"])
        self.assertEqual(record["trimmed_m"], 0.0)
        self.assertIn("more than", record["words"])
        self.assertTrue(np.array_equal(level_corners, corners))
        self.assertFalse(base_passes(level_corners, level_triangles))

    def test_a_thing_that_stands_is_left_as_made(self):
        for corners, triangles in (shapes.chair(), shapes.three_legged_table()):
            level_corners, _, record = levelling.levelled(corners, triangles)
            self.assertFalse(record["trimmed"])
            self.assertTrue(np.array_equal(level_corners, corners))
            self.assertIn("stands", record["words"])

    def test_a_table_leaning_whole_stays_a_failure(self):
        # The three-legged table turned whole by 3 degrees fails "base level" and "centre
        # over the base" as made; the levelling trimmed 21.7 mm and every base check passed, its top still 3 degrees
        # off level. Its top is square to the floor its feet settle it on, so no foot is short: it is left as made.
        for degrees, axis in ((3.0, 0), (3.0, 2), (2.0, 0), (4.0, 2)):
            corners, triangles = shapes.three_legged_table()
            corners = O.set_down(turned(corners, degrees, axis), float(np.ptp(corners, axis=0).max()))
            self.assertFalse(base_passes(corners, triangles), (degrees, axis))
            level_corners, level_triangles, record = levelling.levelled(corners, triangles)
            self.assertFalse(record["trimmed"], (degrees, axis, record))
            self.assertTrue(record["top"]["leans"], record)
            self.assertAlmostEqual(record["top"]["made_deg"], degrees, delta=0.3)
            self.assertIn("leans", record["words"])
            self.assertFalse(base_passes(level_corners, level_triangles), (degrees, axis))

    def test_a_short_leg_under_a_level_top_is_no_lean(self):
        corners, triangles = shapes.three_legged_table(short_leg=0.025)
        lean = levelling.leaning(corners, triangles)
        self.assertFalse(lean["leans"], lean)
        self.assertLess(lean["made_deg"], 0.3)
        self.assertGreater(lean["settled_deg"], 2.0)

    def test_the_limit_is_a_share_of_the_height(self):
        corners, triangles = shapes.three_legged_table()
        self.assertAlmostEqual(levelling.trim_limit_m(corners), 0.05 * float(np.ptp(corners[:, 1])))
