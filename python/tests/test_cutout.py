import unittest

import numpy as np

from model_maker import cutout


def jug_mask():
    """A 1024-pixel cut-out: a body, a handle's loop round a hole 40 by 80 pixels, a pinhole of 4 by 4 in the body and
    a speck apart from it."""
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[300:900, 400:700] = True
    mask[350:550, 300:400] = True
    mask[400:480, 330:370] = False  # the handle's loop
    mask[600:604, 500:504] = False  # a pinhole of the cut
    mask[100:105, 100:105] = True  # a speck
    return mask


class ForShape(unittest.TestCase):
    def test_the_hole_in_a_handle_is_kept_for_the_shape_model(self):
        given = cutout.for_shape(jug_mask())
        self.assertFalse(given[400:480, 330:370].any(), "the loop shows the backdrop, as the picture does")

    def test_a_pinhole_of_the_cut_is_filled_and_a_speck_dropped(self):
        given = cutout.for_shape(jug_mask())
        self.assertTrue(given[600:604, 500:504].all())
        self.assertFalse(given[100:105, 100:105].any())

    def test_a_hole_just_under_the_pinhole_size_is_filled_and_one_at_it_kept(self):
        side = int(np.sqrt(cutout.LEAST_HOLE_PIXELS))
        mask = np.zeros((400, 400), dtype=bool)
        mask[50:350, 50:350] = True
        mask[100:100 + side, 100:100 + side - 1] = False  # under
        mask[200:200 + side + 1, 200:200 + side] = False  # over
        given = cutout.for_shape(mask)
        self.assertTrue(given[100:100 + side, 100:100 + side - 1].all())
        self.assertFalse(given[200:200 + side + 1, 200:200 + side].any())


if __name__ == "__main__":
    unittest.main()
