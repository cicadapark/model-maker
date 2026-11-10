import os
import unittest

import numpy as np

from model_maker import picture_checks, state_judge, words


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


class AskedShare(unittest.TestCase):
    def test_a_picture_nearer_the_asked_words_scores_over_half(self):
        picture = unit([1.0, 0.2, 0.0])
        wanted, unwanted = np.array([unit([1.0, 0.0, 0.0])]), np.array([unit([0.0, 1.0, 0.0])])
        self.assertGreater(state_judge.asked_share(picture, wanted, unwanted, scale=100.0), 0.99)
        self.assertLess(state_judge.asked_share(picture, unwanted, wanted, scale=100.0), 0.01)

    def test_each_pair_of_wordings_counts_alike(self):
        picture = unit([1.0, 1.0, 0.0])
        wanted = np.array([unit([1.0, 0.0, 0.0]), unit([0.0, 1.0, 0.0])])
        unwanted = np.array([unit([0.0, 1.0, 0.0]), unit([1.0, 0.0, 0.0])])
        self.assertAlmostEqual(state_judge.asked_share(picture, wanted, unwanted, scale=50.0), 0.5)


class TheSecondView(unittest.TestCase):
    def test_the_crop_is_a_square_round_the_cut_out_with_a_margin(self):
        from PIL import Image

        mask = np.zeros((400, 600), dtype=bool)
        mask[100:200, 300:340] = True
        crop = state_judge.cropped(Image.new("RGB", (600, 400)), mask)
        self.assertEqual(crop.size[0], crop.size[1])
        self.assertAlmostEqual(crop.size[0], 99 * (1 + 2 * state_judge.CROP_MARGIN), delta=1.5)


class TheCheck(unittest.TestCase):
    def test_a_chest_painted_open_fails_and_a_try_painted_shut_is_chosen(self):
        chest = words.subject_of("a wooden treasure chest")
        open_try, shut_try = picture_checks.PictureResult(score=0.9), picture_checks.PictureResult(score=0.6)
        state_judge.add_state_check(open_try, chest, 0.29)
        state_judge.add_state_check(shut_try, chest, 0.98)
        self.assertFalse(open_try.passed)
        self.assertIn("lid shut", open_try.faults()[0])
        self.assertTrue(shut_try.passed)
        self.assertEqual(picture_checks.best_try([open_try, shut_try]), 1)

    def test_a_thing_with_nothing_to_open_gets_no_check(self):
        result = picture_checks.PictureResult(score=0.5)
        state_judge.add_state_check(result, words.subject_of("a clay jug"), None)
        self.assertEqual(result.checks, [])

    def test_the_check_reads_back_from_a_job_record(self):
        result = picture_checks.PictureResult()
        state_judge.add_state_check(result, words.subject_of("a leather-bound book, closed"), 0.37)
        again = state_judge.check_from_record(result.as_dict())
        self.assertEqual(again.as_dict(), result.checks[0].as_dict())
        self.assertIsNone(state_judge.check_from_record({"checks": []}))


class TheModel(unittest.TestCase):
    def test_siglip_2_reads_with_the_pinned_transformers(self):
        folder = os.path.join(os.environ.get("MODEL_MAKER_MODELS", "models"), state_judge.MODEL_FOLDER)
        if not os.path.exists(os.path.join(folder, "model.safetensors")):
            self.skipTest("the state judge's model is not downloaded")
        from PIL import Image

        judge = state_judge.StateJudge(folder, device="cpu")
        blank = Image.new("RGB", (384, 384), (200, 200, 200))
        mask = np.zeros((384, 384), dtype=bool)
        mask[100:300, 150:250] = True
        share = judge.share(blank, mask, ["a closed book"], ["an open book"])
        self.assertTrue(0.0 <= share <= 1.0)


if __name__ == "__main__":
    unittest.main()
