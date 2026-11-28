import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


def card():
    import torch

    return torch.cuda.is_available()


# cudaMemoryType: what cudaPointerGetAttributes says of a pointer.
CUDA_UNREGISTERED_MEMORY = 0
CUDA_HOST_MEMORY = 1


def memory_type(tensor):
    """The CUDA runtime's memory type of the tensor's first byte (the runtime PyTorch loaded, found by its name)."""
    import ctypes

    class Attributes(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("device", ctypes.c_int), ("device_pointer", ctypes.c_void_p),
                    ("host_pointer", ctypes.c_void_p), ("reserved", ctypes.c_long * 8)]

    for name in ("libcudart.so.12", "libcudart.so.13", "cudart64_12.dll", "cudart64_13.dll"):
        try:
            runtime = ctypes.CDLL(name)
        except OSError:
            continue
        found = Attributes()
        error = runtime.cudaPointerGetAttributes(ctypes.byref(found), ctypes.c_void_p(tensor.data_ptr()))
        if error != 0:
            raise AssertionError(f"cudaPointerGetAttributes: error {error}")
        return found.type
    raise unittest.SkipTest("no CUDA runtime library found by name")


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

        offload = BlockOffload([self.model[0], self.model[2]], "cuda")
        self.model[3].to("cuda")
        self.run_on_card()
        self.assertEqual(self.model[0].weight.device.type, "cpu")
        self.assertEqual(self.model[2][0].weight.device.type, "cpu")
        # Page-locked, asked of the CUDA runtime: PyTorch's is_pinned() answers for its own allocator only and is False
        # for memory registered with cudaHostRegister (torch 2.10). Holds for a block pinned alone too.
        for weight in (self.model[0].weight, self.model[2][0].weight):
            self.assertEqual(memory_type(weight.data), CUDA_HOST_MEMORY)
        registered = bool(offload.registered)
        offload.close()
        if registered:
            self.assertEqual(memory_type(self.model[0].weight.data), CUDA_UNREGISTERED_MEMORY, "given back by close")


@unittest.skipUnless(HAS_TORCH, "needs PyTorch")
class BlockOffloadPinnedOnce(unittest.TestCase):
    """Each block's tensors in one exact-size buffer, page-locked once (registration mocked: runs on the CPU alone)."""

    def setUp(self):
        import torch
        from torch import nn

        torch.manual_seed(0)
        # Odd sizes and three types, so the 64-byte boundaries and the typed views are tested, not only one float type.
        self.model = nn.Sequential(nn.Linear(13, 7), nn.Sequential(nn.Linear(7, 5), nn.BatchNorm1d(5)), nn.Linear(5, 3))
        self.model[0].half()
        self.model.eval()
        self.before = {name: tensor.detach().clone() for name, tensor in self.state().items()}
        self.registered, self.unregistered = [], []
        self.made = []
        self.addCleanup(self.close_all)

    def close_all(self):
        from unittest import mock

        from model_maker import paint

        with mock.patch.object(paint, "_unregister", self.unregistered.append):
            for offload in self.made:
                offload.close()

    def state(self):
        return self.model.state_dict(keep_vars=True)

    def offload(self, registers=True):
        from unittest import mock

        from model_maker import paint

        def register(pointer, size):
            self.registered.append((pointer, size))
            return registers

        from model_maker import device

        # The processor stands in for the card: its blocks are page-locked as a CUDA card's are (device.pins_memory).
        with mock.patch.object(device, "PINNING_BACKENDS", ("cuda", "cpu")), \
                mock.patch.object(paint, "_register", register), \
                mock.patch.object(paint, "_unregister", self.unregistered.append), \
                mock.patch.object(paint, "_pinned_alone", lambda tensor: tensor.clone()):
            offload = paint.BlockOffload([self.model[0], self.model[1]], "cpu")
        self.made.append(offload)
        return offload, paint

    def test_the_views_hold_the_same_bytes_in_one_buffer_a_block(self):
        import mmap

        offload, paint = self.offload()
        for name, tensor in self.state().items():
            old = self.before[name]
            self.assertEqual((tensor.dtype, tensor.shape), (old.dtype, old.shape), name)
            self.assertEqual(tensor.data.view(-1).view(torch_uint8()).tolist(), old.view(-1).view(torch_uint8()).tolist(),
                             name)
        # The first block (Linear 13 -> 7, half) and the second (Linear 7 -> 5 and its BatchNorm, buffers included):
        # one storage each, every tensor at a 64-byte boundary of it, the range registered whole pages of its own.
        self.assertEqual(len(self.registered), 2)
        for block, (pointer, size) in zip((self.model[0], self.model[1]), self.registered):
            tensors = list(block.parameters()) + list(block.buffers())
            storages = {tensor.data.untyped_storage().data_ptr() for tensor in tensors}
            self.assertEqual(len(storages), 1)
            for tensor in tensors:
                self.assertEqual((tensor.data.data_ptr() - pointer) % paint.PINNED_ALIGN, 0)
                self.assertTrue(pointer <= tensor.data.data_ptr()
                                and tensor.data.data_ptr() + tensor.numel() * tensor.element_size() <= pointer + size)
            self.assertEqual(pointer % mmap.PAGESIZE, 0)
            self.assertEqual(size % mmap.PAGESIZE, 0)
            exact = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
            self.assertLess(size, exact + len(tensors) * paint.PINNED_ALIGN + mmap.PAGESIZE, "exact size, not rounded")

    def test_the_same_output_through_the_hooks(self):
        import torch

        x = torch.randn(4, 13)
        with torch.no_grad():
            whole = self.model[2](self.model[1](self.model[0](x.half()).float()))
        self.offload()
        with torch.no_grad():
            offloaded = self.model[2](self.model[1](self.model[0](x.half()).float()))
        self.assertTrue(torch.equal(whole, offloaded))

    def test_close_gives_each_range_back_once(self):
        offload, _ = self.offload()
        from unittest import mock

        from model_maker import paint

        with mock.patch.object(paint, "_unregister", self.unregistered.append):
            offload.close()
            offload.close()
        self.assertEqual(sorted(self.unregistered), sorted(p for p, _ in self.registered))
        self.assertEqual(self.model[0].weight.data.tolist(), self.before["0.weight"].tolist(), "values kept after close")

    def test_a_refused_registration_pins_each_tensor_alone(self):
        offload, _ = self.offload(registers=False)
        self.assertEqual(len(self.registered), 2)
        self.assertEqual(offload.registered, [])
        tensors = list(self.model[1].parameters())
        storages = {tensor.data.untyped_storage().data_ptr() for tensor in tensors}
        self.assertEqual(len(storages), len(tensors), "pinned one by one, not in the shared buffer")
        for name, tensor in self.state().items():
            self.assertTrue(tensor.data.equal(self.before[name]), name)


@unittest.skipUnless(HAS_TORCH, "needs PyTorch")
class BlocksTakeTurnsOnlyOnASmallCard(unittest.TestCase):
    """On a card of 15 GiB or more the view painter keeps its UNet whole: no BlockOffload, and closing the painter
    touches no CUDA runtime call (the card's size faked: runs on the CPU alone)."""

    def wanted_with_card_of(self, gib):
        import os
        from unittest import mock

        import torch

        from model_maker import device, paint

        card = mock.Mock(total_memory=gib * 2 ** 30)
        with mock.patch.dict(os.environ), mock.patch.object(torch.cuda, "get_device_properties", return_value=card):
            os.environ.pop("MODEL_MAKER_VIEW_OFFLOAD", None)
            return paint.offload_wanted(device.Device("cuda"))

    def test_a_16_gib_card_keeps_the_unet_whole(self):
        self.assertFalse(self.wanted_with_card_of(16))

    def test_a_12_gib_card_takes_the_blocks_in_turns(self):
        self.assertTrue(self.wanted_with_card_of(12))

    def test_closing_a_painter_without_offload_calls_no_cuda_runtime(self):
        from unittest import mock

        import torch

        from model_maker import paint

        painter = paint.ViewPainter.__new__(paint.ViewPainter)
        painter.offload = None
        refused = AssertionError("a CUDA runtime call on a painter that does not offload")
        with mock.patch.object(torch.cuda, "cudart", side_effect=refused) as cudart, \
                mock.patch.object(torch.cuda, "synchronize", side_effect=refused) as synchronize, \
                mock.patch.object(paint, "_unregister", side_effect=refused) as unregister, \
                mock.patch.object(paint, "_unregister_all", side_effect=refused) as unregister_all:
            painter.close()
        for call in (cudart, synchronize, unregister, unregister_all):
            call.assert_not_called()


def torch_uint8():
    import torch

    return torch.uint8


if __name__ == "__main__":
    unittest.main()
