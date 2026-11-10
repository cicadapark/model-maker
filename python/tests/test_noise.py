"""The shape's starting noise is drawn on the CPU, so every card starts a seed from the same noise (drawn on the card,
two cards of 12 GB and 16 GB made other shapes from one seed)."""

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


if __name__ == "__main__":
    unittest.main()
