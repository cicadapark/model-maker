"""The shape model's inputs the same on every machine and with every library version: its starting noise is drawn on
the CPU (drawn on the card, two cards of 12 GB and 16 GB made other shapes from one seed), and its picture prepared by
model-maker's own steps (transformers 5 resized it otherwise)."""

import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from model_maker import shape_net


@unittest.skipUnless(HAS_TORCH, "needs torch (installed by setup.sh)")
class StartNoise(unittest.TestCase):
    def test_a_seed_gives_the_same_noise_on_any_machine(self):
        noise = shape_net.start_noise(42)
        self.assertEqual(tuple(noise.shape), (1, shape_net.TOKENS, 64))
        self.assertEqual(noise.dtype, torch.float32)
        self.assertEqual(noise.device.type, "cpu")
        # torch's CPU generator is the same on every machine: these are its first draws for seed 42.
        self.assertEqual([round(v, 6) for v in noise[0, 0, :4].tolist()], [1.926915, 1.487284, 0.900717, -2.105521])
        self.assertTrue(torch.equal(noise, shape_net.start_noise(42)))
        self.assertFalse(torch.equal(noise, shape_net.start_noise(43)))


@unittest.skipUnless(HAS_TORCH, "needs torch (installed by setup.sh)")
class PicturePixels(unittest.TestCase):
    def test_the_picture_is_resized_cropped_and_normalised_by_its_config_alone(self):
        # The steps BitImageProcessor took in transformers 4.x, by Pillow and numpy: the same pixels whatever
        # transformers version is installed.
        import json
        import os
        import tempfile

        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "preprocessor_config.json"), "w") as file:
                json.dump({"size": {"shortest_edge": 256}, "crop_size": {"height": 224, "width": 224},
                           "resample": 3, "rescale_factor": 1 / 255, "image_mean": [0.485, 0.456, 0.406],
                           "image_std": [0.229, 0.224, 0.225]}, file)
            pixels = shape_net.PicturePixels(folder)
            grey = pixels(Image.new("RGB", (300, 300), (128, 128, 128)))
            self.assertEqual(tuple(grey.shape), (1, 3, 224, 224))
            self.assertEqual(grey.dtype, torch.float32)
            expected = (np.float32(128 / 255) - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
                [0.229, 0.224, 0.225], np.float32)
            self.assertTrue(np.allclose(grey[0, :, 100, 100].numpy(), expected, atol=1e-6))
            # A picture with a left half and a right half: resized to 256 high and cut to its middle 224.
            halves = np.zeros((300, 600, 3), np.uint8)
            halves[:, 300:] = 255
            cut = pixels(Image.fromarray(halves))
            self.assertEqual(tuple(cut.shape), (1, 3, 224, 224))
            self.assertLess(float(cut[0, 0, 0, 0]), 0)
            self.assertGreater(float(cut[0, 0, 0, -1]), 0)


if __name__ == "__main__":
    unittest.main()
