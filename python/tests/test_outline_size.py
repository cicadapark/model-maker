"""The outline is measured against its own size: its height for a compact or tall thing, the square root of
height x width for a thing wider than high (the bed frame and rolling pin refused for a few pixels, 15 Nov 2026)."""
import unittest

import numpy as np

from model_maker import object_checks as O


class OutlineSize(unittest.TestCase):
    def test_compact_and_tall_keep_their_height_a_long_low_thing_its_own_size(self):
        square = np.zeros((100, 300), dtype=bool)
        square[10:60, 10:60] = True
        tall = np.zeros((100, 300), dtype=bool)
        tall[0:90, 10:20] = True
        long_low = np.zeros((100, 300), dtype=bool)
        long_low[40:50, 10:100] = True
        self.assertEqual(O.outline_size(square), 1.0)
        self.assertEqual(O.outline_size(tall), 1.0)
        self.assertAlmostEqual(O.outline_size(long_low), 3.0)
        self.assertEqual(O.outline_size(np.zeros((4, 4), dtype=bool)), 1.0)


if __name__ == "__main__":
    unittest.main()
