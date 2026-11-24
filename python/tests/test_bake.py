import unittest
from unittest import mock

import numpy as np

from model_maker import bake, glb, render

from . import shapes

COLOUR = np.array([0.8, 0.5, 0.3])
# A sliver this many metres wide: about a fifth of a texel at the small texture below.
SLIVER_WIDTH_M = 0.002
SMALL_SIDE = 128
CLOSE = 2 / 255


def square_with_sliver():
    """A flat square facing +z and, along its +x edge, a sliver facing +x, far thinner than a texel: its own chart."""
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0.5, -SLIVER_WIDTH_M]], dtype=np.float64)
    triangles = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 2]], dtype=np.int64)
    return corners, triangles


def as_before(function, *args, **kwargs):
    """The bake as it was before the fix: no touched texels and no fill to the end."""
    nothing = np.zeros(0, dtype=np.int64)
    with mock.patch.object(bake, "touched_texels", lambda *a: (nothing, nothing, nothing, np.zeros((0, 3)))), \
            mock.patch.object(bake, "filled_to_the_end", lambda texture, filled: None):
        return function(*args, **kwargs)


def middle_reads(result):
    """Each triangle's colour at its middle, as the report's renderer reads the texture."""
    uvs = result["uvs"][result["triangles"]]
    lookup = render.texture_lookup(uvs, result["texture"])
    count = len(result["triangles"])
    return lookup(np.arange(count), np.full((count, 3), 1 / 3))


def held_by_picture(result, side):
    """Each texel's triangle by the bake's own rasteriser on the result's layout (-1 where none)."""
    _, held_by, _ = render.rasterised(result["uvs"] * side, np.zeros(len(result["uvs"])), result["triangles"], side,
                                    side)
    return held_by


def owned_texels(result, side):
    return held_by_picture(result, side) >= 0


def cloud_on(corners, triangles, count=20000, seed=3):
    """Points on the surface with colours that change across it, and their faces' normals."""
    rng = np.random.default_rng(seed)
    normals, areas = bake.M.face_normals_and_areas(corners, triangles)
    picked = rng.choice(len(triangles), count, p=areas / areas.sum())
    weights = rng.dirichlet(np.ones(3), count)
    points = np.einsum("pk,pkd->pd", weights, corners[triangles[picked]])
    low, high = corners.min(0), corners.max(0)
    colours = 0.2 + 0.6 * (points - low) / np.maximum(high - low, 1e-9)
    return points, normals[picked], colours


class SliverColour(unittest.TestCase):
    def test_a_sliver_thinner_than_a_texel_reads_its_own_colour(self):
        corners, triangles = square_with_sliver()
        colours = np.tile(COLOUR, (len(corners), 1))
        before = as_before(bake.baked, corners, triangles, colours, None, SMALL_SIDE)
        after = bake.baked(corners, triangles, colours, None, SMALL_SIDE)
        sliver = int(np.flatnonzero(before["original"][before["triangles"]].max(1) == 4)[0])
        self.assertIn(sliver, bake.starved_triangles(held_by_picture(before, SMALL_SIDE), len(triangles)))
        np.testing.assert_array_equal(middle_reads(before)[sliver], 0.0)
        self.assertLessEqual(np.abs(middle_reads(after)[sliver] - COLOUR).max(), CLOSE)
        # Its own texel, not a neighbour's colour carried there by the fill.
        held_by = held_by_picture(after, SMALL_SIDE)
        rows, cols, which, _ = bake.touched_texels(after["uvs"] * SMALL_SIDE, after["triangles"],
                                                   bake.starved_triangles(held_by, len(triangles)), held_by >= 0,
                                                   SMALL_SIDE)
        middle = after["uvs"][after["triangles"][sliver]].mean(0) * SMALL_SIDE
        mine = set(zip(rows[which == sliver].tolist(), cols[which == sliver].tolist()))
        self.assertIn((int(middle[1]), int(middle[0])), mine)

    def test_every_triangle_reads_its_colour_and_no_texel_is_left_empty(self):
        for name, (corners, triangles) in (("chair", shapes.chair(cell=0.03)), ("mug", shapes.mug(cell=0.01)),
                                           ("sword", shapes.sword(cell=0.005))):
            with self.subTest(shape=name):
                colours = np.tile(COLOUR, (len(corners), 1))
                result = bake.baked(corners, triangles, colours, None, 256)
                self.assertLessEqual(np.abs(middle_reads(result) - COLOUR).max(), CLOSE)
                self.assertLessEqual(np.abs(result["texture"] - COLOUR).max(), CLOSE)


class NothingElseChanged(unittest.TestCase):
    def test_owned_texels_and_every_measure_are_as_before(self):
        chair_corners, chair_triangles = shapes.chair(cell=0.03)
        # The chair with the square and its sliver beside it, so some triangles own no texel centre.
        square_corners, square_triangles = square_with_sliver()
        corners = np.concatenate([chair_corners, square_corners * 0.3 + [0.5, 0.0, 0.0]])
        triangles = np.concatenate([chair_triangles, square_triangles + len(chair_corners)])
        cloud = cloud_on(corners, triangles)
        colours = np.tile(np.array([0.5, 0.5, 0.5]), (len(corners), 1))
        before = as_before(bake.baked, corners, triangles, colours, cloud, 256)
        after = bake.baked(corners, triangles, colours, cloud, 256)
        np.testing.assert_array_equal(before["uvs"], after["uvs"])
        np.testing.assert_array_equal(before["triangles"], after["triangles"])
        for key in ("coverage", "coverage_all_texels", "overlap", "texels_per_m"):
            self.assertEqual(before[key], after[key], key)
        held_by = held_by_picture(before, 256)
        self.assertGreater(len(bake.starved_triangles(held_by, len(before["triangles"]))), 0)
        owned = held_by >= 0
        np.testing.assert_array_equal(before["texture"][owned], after["texture"][owned])
        self.assertFalse(np.array_equal(before["texture"], after["texture"]))

    def test_the_same_texels_on_two_runs(self):
        corners, triangles = square_with_sliver()
        colours = np.tile(COLOUR, (len(corners), 1))
        first = bake.baked(corners, triangles, colours, None, SMALL_SIDE)["texture"]
        second = bake.baked(corners, triangles, colours, None, SMALL_SIDE)["texture"]
        np.testing.assert_array_equal(first, second)


class TouchedTexels(unittest.TestCase):
    def test_the_larger_of_two_starved_triangles_wins_a_shared_texel(self):
        # Two slivers across texel (row 5, column 5), the second twice as wide.
        xy = np.array([[2.0, 5.4], [9.0, 5.4], [5.5, 5.5], [2.0, 5.6], [9.0, 5.6], [5.5, 5.8]])
        triangles = np.array([[0, 1, 2], [3, 4, 5]])
        owned = np.zeros((16, 16), dtype=bool)
        runs = [bake.touched_texels(xy, triangles, np.array([0, 1]), owned, 16) for _ in range(2)]
        for rows, cols, which, weights in runs:
            at = np.flatnonzero((rows == 5) & (cols == 5))
            self.assertEqual(len(at), 1)
            self.assertEqual(int(which[at[0]]), 1)
            np.testing.assert_allclose(weights.sum(1), 1.0)
            self.assertEqual(len(set(zip(rows.tolist(), cols.tolist()))), len(rows))
        for a, b in zip(runs[0], runs[1]):
            np.testing.assert_array_equal(a, b)

    def test_ties_go_to_the_lowest_index_and_owned_texels_are_left_alone(self):
        xy = np.array([[2.0, 5.4], [9.0, 5.4], [5.5, 5.5]])
        triangles = np.array([[0, 1, 2], [0, 1, 2]])
        owned = np.zeros((16, 16), dtype=bool)
        owned[5, 3] = True
        rows, cols, which, _ = bake.touched_texels(xy, triangles, np.array([0, 1]), owned, 16)
        self.assertTrue((which == 0).all())
        self.assertNotIn((5, 3), set(zip(rows.tolist(), cols.tolist())))


class FillToTheEnd(unittest.TestCase):
    def test_every_empty_texel_takes_the_nearest_filled_one(self):
        texture = np.zeros((40, 40, 3))
        filled = np.zeros((40, 40), dtype=bool)
        texture[2, 2], texture[37, 37] = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
        filled[2, 2] = filled[37, 37] = True
        bake.filled_to_the_end(texture, filled)
        self.assertTrue(filled.all())
        np.testing.assert_array_equal(texture[0, 39], (1.0, 0.0, 0.0))
        np.testing.assert_array_equal(texture[10, 5], (1.0, 0.0, 0.0))
        np.testing.assert_array_equal(texture[30, 35], (0.0, 0.0, 1.0))


class Repaired(unittest.TestCase):
    def test_a_stored_texture_gets_its_sliver_coloured_and_keeps_the_rest(self):
        corners, triangles = square_with_sliver()
        colours = np.tile(COLOUR, (len(corners), 1))
        before = as_before(bake.baked, corners, triangles, colours, None, SMALL_SIDE)
        stored = np.round(before["texture"] * 255) / 255
        mended = bake.repaired(before["corners"], before["triangles"], before["uvs"], stored)
        result = {**before, "texture": mended}
        self.assertLessEqual(np.abs(middle_reads(result) - COLOUR).max(), CLOSE)
        owned = owned_texels(before, SMALL_SIDE)
        np.testing.assert_array_equal(mended[owned], stored[owned])
        self.assertTrue((mended.sum(2) > 0).all())

    def test_a_glb_keeps_everything_but_its_image(self):
        corners, triangles = square_with_sliver()
        result = bake.baked(corners, triangles, np.tile(COLOUR, (len(corners), 1)), None, 64)
        normals = np.tile([0.0, 0.0, 1.0], (len(result["corners"]), 1))
        data = glb.glb_bytes("square", result["corners"], result["triangles"], normals, result["uvs"],
                             result["texture"])
        replaced = glb.with_image(data, glb.png_bytes(np.full((64, 64, 3), 0.25)))
        old, new = glb.painted_mesh(data), glb.painted_mesh(replaced)
        for a, b in zip(old[:3], new[:3]):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(new[3], round(0.25 * 255) / 255)


if __name__ == "__main__":
    unittest.main()
