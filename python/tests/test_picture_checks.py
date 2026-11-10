import unittest

import numpy as np

from model_maker import picture_checks as P

SIDE = 512
BACKDROP = 205


def picture(boxes, backdrop=BACKDROP):
    """A grey picture with dark boxes (row0, col0, row1, col1) on it, and its cut-out."""
    rgb = np.full((SIDE, SIDE, 3), backdrop, dtype=np.uint8)
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    for r0, c0, r1, c1 in boxes:
        rgb[r0:r1, c0:c1] = (110, 80, 50)
        mask[r0:r1, c0:c1] = True
    return rgb, mask


def named(result):
    return {c.name: c.passed for c in result.checks}


class PictureChecks(unittest.TestCase):
    def test_one_object_on_a_plain_backdrop_passes(self):
        rgb, mask = picture([(120, 180, 400, 330)])
        result = P.checked(rgb, mask)
        self.assertTrue(result.passed, result.faults())

    def test_two_things_fail_one_piece(self):
        rgb, mask = picture([(120, 60, 400, 200), (120, 300, 300, 440)])
        result = P.checked(rgb, mask)
        self.assertFalse(named(result)["one piece"])

    def test_a_cast_shadow_under_it_fails(self):
        rgb, mask = picture([(120, 180, 400, 330)])
        rows, cols = np.mgrid[0:SIDE, 0:SIDE]
        shadow = ((rows - 420) / 18.0) ** 2 + ((cols - 255) / 110.0) ** 2 <= 1
        rgb[shadow & ~mask] = 160
        self.assertFalse(named(P.checked(rgb, mask))["shadow on the backdrop"])

    def test_a_shadow_cast_to_one_side_fails(self):
        rgb, mask = picture([(120, 180, 400, 330)])
        rows, cols = np.mgrid[0:SIDE, 0:SIDE]
        shadow = ((rows - 380) / 25.0) ** 2 + ((cols - 380) / 60.0) ** 2 <= 1
        rgb[shadow & ~mask] = 165
        self.assertFalse(named(P.checked(rgb, mask))["shadow on the backdrop"])

    def test_a_wide_objects_side_shadow_fails_too(self):
        rgb, mask = picture([(150, 100, 380, 400)])
        rows, cols = np.mgrid[0:SIDE, 0:SIDE]
        shadow = ((rows - 360) / 25.0) ** 2 + ((cols - 410) / 30.0) ** 2 <= 1
        rgb[shadow & ~mask] = 165
        self.assertFalse(named(P.checked(rgb, mask))["shadow on the backdrop"])
        clean_rgb, clean_mask = picture([(150, 100, 380, 400)])
        self.assertTrue(named(P.checked(clean_rgb, clean_mask))["shadow on the backdrop"])

    def test_a_shadow_on_the_floor_far_below_a_floating_object_fails(self):
        rgb, mask = picture([(150, 180, 260, 330)])
        rows, cols = np.mgrid[0:SIDE, 0:SIDE]
        shadow = ((rows - 400) / 12.0) ** 2 + ((cols - 290) / 90.0) ** 2 <= 1
        rgb[shadow & ~mask] = 160
        self.assertFalse(named(P.checked(rgb, mask))["shadow on the backdrop"])

    def test_a_floor_lit_unevenly_front_to_back_is_not_a_shadow(self):
        rgb, mask = picture([(150, 180, 300, 330)])
        floor = np.linspace(200, 230, SIDE - 300).astype(np.uint8)
        rgb[300:][~mask[300:]] = np.repeat(floor[:, None], SIDE, 1)[~mask[300:]][:, None]
        self.assertTrue(named(P.checked(rgb, mask))["shadow on the backdrop"])

    def test_a_shadow_or_floor_the_cut_out_leaves_out_does_not_fail_the_picture(self):
        rgb, mask = picture([(150, 180, 260, 330)])
        rows, cols = np.mgrid[0:SIDE, 0:SIDE]
        rgb[(((rows - 400) / 12.0) ** 2 + ((cols - 290) / 90.0) ** 2 <= 1) & ~mask] = 160
        rgb[450:][~mask[450:]] = 180
        result = P.checked(rgb, mask)
        self.assertFalse(named(result)["shadow on the backdrop"])
        self.assertTrue(result.passed, result.faults())

    def test_a_shadow_taken_into_the_cut_out_fails(self):
        rgb, mask = picture([(150, 180, 380, 330)])
        # A grey shadow on the floor from the object's foot to its right, half of it cut out with the object.
        rgb[380:400, 180:420] = 160
        mask[380:400, 180:330] = True
        result = P.checked(rgb, mask)
        self.assertFalse(named(result)["no shadow or floor in the cut-out"])

    def test_a_dark_foot_on_its_own_contact_shadow_passes(self):
        rgb, mask = picture([(150, 180, 380, 330)])
        rgb[360:380, 180:330] = (60, 55, 50)
        rows, cols = np.mgrid[0:SIDE, 0:SIDE]
        rgb[(((rows - 385) / 8.0) ** 2 + ((cols - 255) / 100.0) ** 2 <= 1) & ~mask] = 150
        self.assertTrue(named(P.checked(rgb, mask))["no shadow or floor in the cut-out"])

    def test_a_floor_line_fails(self):
        rgb, mask = picture([(120, 180, 400, 330)])
        rgb[350:][~mask[350:]] = 165
        self.assertFalse(named(P.checked(rgb, mask))["floor line on the backdrop"])

    def test_a_slow_gradient_is_not_a_floor(self):
        rgb, mask = picture([(120, 180, 400, 330)])
        gradient = np.linspace(215, 190, SIDE).astype(np.uint8)
        rgb[~mask] = np.repeat(gradient[:, None], SIDE, 1)[~mask][:, None]
        result = P.checked(rgb, mask)
        self.assertTrue(named(result)["floor line on the backdrop"], result.faults())

    def test_cut_off_at_the_edge_fails(self):
        rgb, mask = picture([(0, 180, 400, 330)])
        self.assertFalse(named(P.checked(rgb, mask))["not cut off"])

    def test_scenery_round_it_fails_the_plain_backdrop(self):
        rgb, mask = picture([(120, 180, 400, 330)])
        rgb[20:120, 20:160] = (40, 120, 40)
        self.assertFalse(named(P.checked(rgb, mask))["plain backdrop"])

    def test_a_thin_upright_spear_is_big_enough(self):
        rgb, mask = picture([(20, 250, 490, 258)])
        result = P.checked(rgb, mask)
        self.assertTrue(result.passed, result.faults())

    def test_the_best_try_passes_first(self):
        good_rgb, good_mask = picture([(120, 180, 400, 330)])
        bad_rgb, bad_mask = picture([(0, 180, 400, 330)])
        results = [P.checked(bad_rgb, bad_mask), P.checked(good_rgb, good_mask), P.checked(good_rgb, good_mask)]
        self.assertEqual(P.best_try(results), 1)


if __name__ == "__main__":
    unittest.main()
