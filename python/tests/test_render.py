import unittest

import numpy as np

from model_maker import render

from . import shapes


def drawn_in_full(corners, triangles, turn, side, tilt=0.0, distance=None):
    """Where the full rasteriser (depth, owner and places) draws the surface: what `silhouette` must cover."""
    framing = render.Framing(corners, turn, side, tilt, distance=distance)
    xy, depth = framing.pixels(corners)
    zbuf, _, _ = render.rasterised(xy, depth, triangles, side, side)
    return np.isfinite(zbuf)


class Silhouette(unittest.TestCase):
    def assert_same_pixels(self, corners, triangles, turn, side, tilt=0.0, distance=None):
        fast = render.silhouette(corners, triangles, turn, side, tilt, distance)
        full = drawn_in_full(corners, triangles, turn, side, tilt, distance)
        self.assertEqual(fast.shape, full.shape)
        self.assertEqual(int((fast != full).sum()), 0, (turn, tilt, distance, int(full.sum())))

    def test_it_covers_what_the_full_rasteriser_draws_from_every_view(self):
        corners, triangles = shapes.chair()
        for turn, tilt, distance in ((0, 0.0, None), (30, 12.0, None), (-75, 24.0, 6.0), (47.5, 3.0, 3.0),
                                     (90, 0.0, 4.0), (-13, 7.0, None)):
            self.assert_same_pixels(corners, triangles, turn, 448, tilt, distance)

    def test_thin_parts_and_large_triangles_are_covered_alike(self):
        # A sword (a thin blade, a pixel or two wide edge-on) and a box of two triangles a face (each larger than the
        # rasteriser's batched span: drawn whole here, split there).
        self.assert_same_pixels(*shapes.sword(), 90, 448, 12.0)
        self.assert_same_pixels(*shapes.sword(), 20, 448, 0.0, 4.0)
        box = np.array([[x, y, z] for x in (0, 1) for y in (0, 0.5) for z in (0, 0.3)], dtype=np.float64)
        faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                          [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int64)
        for turn in (0, 33, 60):
            self.assert_same_pixels(box, faces, turn, 300, 15.0)
            self.assert_same_pixels(box, faces, turn, 300, 15.0, 3.0)

    def test_a_flat_triangle_draws_nothing(self):
        line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        self.assertFalse(render.silhouette(line, np.array([[0, 1, 2]]), 0, 64).any())


if __name__ == "__main__":
    unittest.main()
