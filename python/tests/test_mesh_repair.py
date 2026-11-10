import unittest

import numpy as np

from model_maker import mesh_repair as R, mesh_tools as M, object_checks as O

from . import shapes


def closed_box():
    return shapes.surface_of([((0.0, 0.0, 0.0), (0.2, 0.2, 0.2))], 0.02)


def cube(low, side):
    """An exact cube of 12 triangles wound outward (its edges sharp, as two touching parts' are)."""
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float64) * side + low
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    triangles = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    return corners, triangles


def defects(corners, triangles):
    return M.open_edge_loops(triangles)


class Repair(unittest.TestCase):
    def test_a_clean_surface_is_left_as_it_is(self):
        corners, triangles = closed_box()
        fixed_corners, fixed_triangles, done = R.repaired(corners, triangles)
        self.assertEqual(len(fixed_triangles), len(triangles))
        self.assertEqual(defects(fixed_corners, fixed_triangles), (0, 0, 0))

    def test_a_seam_of_unjoined_corners_is_closed(self):
        corners, triangles = closed_box()
        # One corner copied: the triangles round half of it use the copy, a seam opens.
        copy = len(corners)
        corners = np.concatenate([corners, corners[:1]])
        users = np.flatnonzero((triangles == 0).any(1))[:2]
        triangles = triangles.copy()
        triangles[users] = np.where(triangles[users] == 0, copy, triangles[users])
        self.assertGreater(defects(corners, triangles)[0], 0)
        fixed_corners, fixed_triangles, done = R.repaired(corners, triangles)
        self.assertEqual(defects(fixed_corners, fixed_triangles), (0, 0, 0))
        self.assertEqual(done["corners_joined"], 1)

    def test_a_fold_of_no_thickness_is_dropped(self):
        corners, triangles = closed_box()
        fold = triangles[:3]
        triangles = np.concatenate([triangles, fold, fold[:, ::-1]])
        self.assertGreater(defects(corners, triangles)[2], 0)
        fixed_corners, fixed_triangles, done = R.repaired(corners, triangles)
        self.assertEqual(defects(fixed_corners, fixed_triangles), (0, 0, 0))
        self.assertEqual(done["folds_dropped"], 3)

    def test_two_boxes_touching_along_an_edge_are_split_and_still_one_piece(self):
        a = cube(np.zeros(3), 0.2)
        self.assertGreater(M.volume_and_centre(*a)[0], 0)
        b = (a[0] + np.array([0.2, 0.2, 0.0]), a[1] + len(a[0]))
        corners, triangles = np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]])
        corners, triangles = M.welded(corners, triangles, 1e-9)
        self.assertGreater(defects(corners, triangles)[2], 0)
        fixed_corners, fixed_triangles, done = R.repaired(corners, triangles)
        self.assertEqual(defects(fixed_corners, fixed_triangles), (0, 0, 0))
        self.assertGreater(done["corners_split"], 0)
        # They touch: one piece to the check.
        self.assertEqual(O.wholeness(fixed_corners, fixed_triangles)["pieces"], 1)

    def test_a_small_hole_is_filled_and_a_large_one_left(self):
        corners, triangles = shapes.surface_of([((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))], 0.01)
        centre = corners[triangles].mean(1)
        small = np.zeros(len(triangles), dtype=bool)
        small[np.argmin(np.linalg.norm(centre - np.array([0.5, 1.0, 0.5]), axis=1))] = True
        fixed_corners, fixed_triangles, done = R.repaired(corners, triangles[~small])
        self.assertEqual(done["holes_filled"], 1)
        self.assertEqual(defects(fixed_corners, fixed_triangles), (0, 0, 0))
        large = np.linalg.norm(centre - np.array([0.5, 1.0, 0.5]), axis=1) < 0.2
        _, _, done = R.repaired(corners, triangles[~large])
        self.assertEqual((done["holes_filled"], done["holes_left"]), (0, 1))

    def test_a_hollow_inside_is_removed_and_a_loose_part_kept(self):
        outer = shapes.surface_of([((0.0, 0.0, 0.0), (0.4, 0.4, 0.4))], 0.02)
        inner = shapes.surface_of([((0.15, 0.15, 0.15), (0.25, 0.25, 0.25))], 0.02)
        loose = shapes.surface_of([((0.6, 0.0, 0.0), (0.7, 0.1, 0.1))], 0.02)
        parts = [outer, (inner[0], inner[1][:, ::-1]), loose]
        corners, triangles, offset = [], [], 0
        for c, t in parts:
            corners.append(c)
            triangles.append(t + offset)
            offset += len(c)
        fixed_corners, fixed_triangles, done = R.repaired(np.concatenate(corners), np.concatenate(triangles))
        self.assertEqual(done["hollows_removed"], 1)
        self.assertEqual(M.pieces(fixed_corners, fixed_triangles)[1], 2)


if __name__ == "__main__":
    unittest.main()
