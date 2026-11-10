import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


def card():
    import torch

    return torch.cuda.is_available()


@unittest.skipUnless(HAS_TORCH and card(), "needs PyTorch and an NVIDIA card")
class BlockOffloadOnTheCard(unittest.TestCase):
    """The view painter's UNet blocks taking turns on the card (paint.BlockOffload) must give the same sums."""

    def setUp(self):
        import torch
        from torch import nn

        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256)),
                                   nn.Linear(256, 8)).half().eval()
        self.x = torch.randn(32, 64).half()

    def run_on_card(self):
        import torch

        with torch.no_grad():
            return self.model(self.x.to("cuda")).cpu()

    def test_the_same_output_bit_for_bit(self):
        import torch

        from model_maker.paint import BlockOffload

        self.model.to("cuda")
        whole = self.run_on_card()
        self.model.to("cpu")
        blocks = [self.model[0], self.model[2]]
        BlockOffload(blocks, "cuda")
        self.model[3].to("cuda")
        offloaded = self.run_on_card()
        self.assertTrue(torch.equal(whole, offloaded))

    def test_a_block_is_off_the_card_between_runs(self):
        from model_maker.paint import BlockOffload

        BlockOffload([self.model[0], self.model[2]], "cuda")
        self.model[3].to("cuda")
        self.run_on_card()
        self.assertEqual(self.model[0].weight.device.type, "cpu")
        self.assertTrue(self.model[0].weight.is_pinned())
        self.assertEqual(self.model[2][0].weight.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
