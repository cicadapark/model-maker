"""On an NVIDIA card the models are loaded, moved and run with the same calls, in the same order, as before the device
layer: each model class is stood in for and records what it is given, the device is said to be CUDA, and the record
is compared with the calls the code made before (0.3.0, written out here). No card and no weights are needed."""

import os
import types
import unittest
from unittest import mock

import numpy as np

from model_maker import device

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class Log(list):
    def maker(self, name, result=None):
        """A stand-in class whose `from_pretrained` and constructor are recorded."""
        log = self

        class StandIn:
            def __init__(self, *args, **kwargs):
                log.append((name, args, kwargs))

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                log.append((f"{name}.from_pretrained", args, kwargs))
                return result if result is not None else Part(log, name)

        return StandIn


class Part:
    """A model part that records where it is moved."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def to(self, *args, **kwargs):
        self.log.append((f"{self.name}.to", args, kwargs))
        return self


def on_cuda(test, log):
    """The device said to be CUDA, the cache emptying recorded."""
    for patch in (mock.patch.object(device, "current", lambda: device.Device("cuda")),
                  mock.patch.object(torch.cuda, "empty_cache", lambda: log.append(("empty_cache", (), {})))):
        patch.start()
        test.addCleanup(patch.stop)


@unittest.skipUnless(HAS_TORCH, "needs torch (installed by setup.sh)")
class OnCudaAsBefore(unittest.TestCase):
    def test_the_inpainter(self):
        import diffusers

        from model_maker import touch_up

        log = Log()
        on_cuda(self, log)

        class Pipe(Part):
            def __call__(self, **kwargs):
                log.append(("pipe", (), {"generator": kwargs["generator"].device.type}))
                return types.SimpleNamespace(images=["painted"])

        pipe = Pipe(log, "pipe")
        vae = Part(log, "vae")
        with mock.patch.object(diffusers, "AutoencoderKL", log.maker("AutoencoderKL", vae)), \
                mock.patch.object(diffusers, "StableDiffusionXLInpaintPipeline", log.maker("Inpaint", pipe)):
            inpainter = touch_up.Inpainter("sdxl", "vae")
            self.assertEqual(inpainter(None, None, "a pot", 7), "painted")
        self.assertEqual(log, [
            ("AutoencoderKL.from_pretrained", ("vae",), {"torch_dtype": torch.float16}),
            ("Inpaint.from_pretrained", ("sdxl",), {"vae": vae, "variant": "fp16", "torch_dtype": torch.float16}),
            ("pipe.to", ("cuda",), {}), ("pipe", (), {"generator": "cpu"}), ("pipe.to", ("cpu",), {}),
            ("empty_cache", (), {})])

    def test_the_picture_painter(self):
        import inspect

        import diffusers
        import transformers

        from model_maker import painter

        log = Log()
        on_cuda(self, log)

        class Flux(Part):
            def enable_model_cpu_offload(self, **kwargs):
                log.append(("enable_model_cpu_offload", (), kwargs))

            def set_progress_bar_config(self, **kwargs):
                log.append(("set_progress_bar_config", (), kwargs))

        transformer, encoder, flux = Part(log, "transformer"), Part(log, "t5"), Flux(log, "flux")
        patches = [mock.patch.object(device, "four_bit_refusal", lambda chosen: ""),
                   mock.patch.object(diffusers, "BitsAndBytesConfig", log.maker("DiffusersBnb")),
                   mock.patch.object(diffusers, "FluxTransformer2DModel", log.maker("Transformer", transformer)),
                   mock.patch.object(diffusers, "FluxPipeline", log.maker("FluxPipeline", flux)),
                   mock.patch.object(transformers, "BitsAndBytesConfig", log.maker("TransformersBnb")),
                   mock.patch.object(transformers, "T5EncoderModel", log.maker("T5", encoder))]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        painter.Painter("flux")
        nf4 = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16}
        names = [entry[0] for entry in log]
        self.assertEqual(names, ["DiffusersBnb", "Transformer.from_pretrained", "transformer.to", "empty_cache",
                                 "TransformersBnb", "T5.from_pretrained", "t5.to", "empty_cache",
                                 "FluxPipeline.from_pretrained", "enable_model_cpu_offload",
                                 "set_progress_bar_config"])
        self.assertEqual(log[0][2], nf4)
        self.assertEqual(log[4][2], nf4)
        self.assertEqual(log[1][1:], (("flux",), {"subfolder": "transformer", "quantization_config": mock.ANY,
                                                  "torch_dtype": torch.bfloat16}))
        self.assertEqual(log[5][1:], (("flux",), {"subfolder": "text_encoder_2", "quantization_config": mock.ANY,
                                                  "torch_dtype": torch.bfloat16}))
        self.assertEqual((log[2][1], log[6][1]), (("cpu",), ("cpu",)))
        self.assertEqual(log[8][2], {"transformer": transformer, "text_encoder_2": encoder,
                                     "torch_dtype": torch.bfloat16})
        # Before, the offload was called with no device: diffusers' default is the device named now, "cuda" on a CUDA
        # computer (0.31 wrote "cuda"; 0.38 writes None and asks its get_device(), which answers "cuda" first).
        default = inspect.signature(diffusers.DiffusionPipeline.enable_model_cpu_offload).parameters["device"].default
        if default is None:
            from diffusers.utils.torch_utils import get_device

            # get_device is cached (functools.lru_cache, first asked when diffusers is imported): ask its body.
            with mock.patch.object(torch.cuda, "is_available", lambda: True):
                default = getattr(get_device, "__wrapped__", get_device)()
        self.assertEqual((default, log[9][2]), ("cuda", {"device": "cuda"}))

    def test_the_picture_painter_refuses_before_loading_where_four_bits_cannot_run(self):
        from model_maker import painter

        with mock.patch.object(device, "current", lambda: device.Device("mps")), \
                mock.patch.object(device, "four_bit_refusal", lambda chosen: "no 4-bit route on Metal"):
            with self.assertRaisesRegex(RuntimeError, "no 4-bit route on Metal"):
                painter.Painter("flux")

    def test_the_shape_model(self):
        from model_maker import shape_net

        log = Log()
        on_cuda(self, log)
        encoder, denoiser, vae = Part(log, "encoder"), Part(log, "denoiser"), Part(log, "vae")

        def kept_models(dtype, choices=None):
            log.append(("kept_models", (dtype, choices), {}))
            return {"encoder": ("processor", encoder), "denoiser": denoiser, "vae": vae}

        def recorded(name, result):
            def call(*args):
                log.append((name, args[-2:] if name != "surface" else (), {}))
                return result
            return call

        corners, triangles = np.zeros((3, 3)), np.zeros((1, 3), dtype=int)
        with mock.patch.object(shape_net, "kept_models", kept_models), \
                mock.patch.object(shape_net, "prepared", lambda image, mask: "image"), \
                mock.patch.object(shape_net, "picture_features", recorded("picture_features", "features")), \
                mock.patch.object(shape_net, "sampled_latents", recorded("sampled_latents", "latents")), \
                mock.patch.object(shape_net, "surface", recorded("surface", (corners, triangles, []))), \
                mock.patch("builtins.print"):
            shape_net.make(np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8), dtype=bool), seed=43)
        self.assertEqual([entry[:2] for entry in log], [
            ("kept_models", (torch.float16, None)),
            ("encoder.to", ("cuda",)), ("picture_features", ("cuda", torch.float16)), ("encoder.to", ("cpu",)),
            ("empty_cache", ()),
            ("denoiser.to", ("cuda",)), ("sampled_latents", ("cuda", torch.float16)), ("denoiser.to", ("cpu",)),
            ("empty_cache", ()),
            ("vae.to", ("cuda",)), ("surface", ()), ("vae.to", ("cpu",)), ("empty_cache", ())])

    def test_the_shape_models_are_kept_in_the_float_type_make_asks_for(self):
        from model_maker import kept, pipeline, registry, shape_net

        asked = []
        with mock.patch.object(device, "current", lambda: device.Device("cuda")), \
                mock.patch.object(shape_net, "kept_models", lambda dtype: asked.append(dtype)), \
                mock.patch.object(registry, "verify_folder", lambda model_id, folder, checked=None: None), \
                mock.patch.dict(os.environ):
            kept.Kept().shape_models(pipeline.Settings(models="models", sources="sources", verify=True))
        self.assertEqual(asked, [torch.float16])

    def test_the_view_painter(self):
        from model_maker import paint

        log = Log()
        on_cuda(self, log)

        class Unet(Part):
            def named_children(self):
                return [("down_blocks", Part(log, "down")), ("conv_in", Part(log, "conv_in"))]

        class Pipe:
            unet, vae, cond_encoder = Unet(log, "unet"), Part(log, "vae"), Part(log, "cond_encoder")
            text_encoder, text_encoder_2 = Part(log, "text_encoder"), Part(log, "text_encoder_2")

            def encode_prompt(self, **kwargs):
                log.append(("encode_prompt", (), {"device": kwargs["device"]}))
                return "embeds", "negative", "pooled", "negative pooled"

            def __call__(self, **kwargs):
                log.append(("pipe", (), {"device": self._execution_device, "control": kwargs["control_image"],
                                         "generator": kwargs["generator"].device.type}))
                return types.SimpleNamespace(images=["views"])

        class Control:
            def permute(self, *order):
                return self

            def to(self, *args):
                log.append(("control.to", args, {}))
                return "control"

        painter = object.__new__(paint.ViewPainter)
        painter.pipe, painter.offload = Pipe(), None
        painter.device, painter.dtype = device.Device("cuda"), torch.float16
        with mock.patch.object(torch, "from_numpy", lambda array: Control()):
            self.assertEqual(painter.views(np.zeros((6, 4, 4, 6)), "a pot", "reference", 0, steps=2), ["views"])
        self.assertEqual([entry[:2] if entry[0] != "pipe" else entry for entry in log], [
            ("unet.to", ("cuda",)), ("vae.to", ("cuda",)), ("cond_encoder.to", ("cuda",)),
            ("text_encoder.to", ("cuda",)), ("text_encoder_2.to", ("cuda",)), ("encode_prompt", ()),
            ("text_encoder.to", ("cpu",)), ("text_encoder_2.to", ("cpu",)), ("empty_cache", ()),
            ("control.to", ("cuda", torch.float16)),
            ("pipe", (), {"device": torch.device("cuda"), "control": "control", "generator": "cpu"}),
            ("unet.to", ("cpu",)), ("vae.to", ("cpu",)), ("cond_encoder.to", ("cpu",)), ("empty_cache", ())])
        self.assertEqual(log[5][2], {"device": "cuda"})

    def test_the_view_painters_offloaded_blocks_are_page_locked_on_cuda(self):
        from torch import nn

        from model_maker import paint

        registered = []

        def register(pointer, size):
            registered.append(size)
            return True

        with mock.patch.object(paint, "_register", register), mock.patch.object(paint, "_unregister", lambda _: None):
            on_card = paint.BlockOffload([nn.Linear(3, 2)], "cuda")
            self.assertEqual(len(registered), 1, "a block's buffer page-locked once on CUDA, as before")
            on_processor = paint.BlockOffload([nn.Linear(3, 2)], "cpu")
            self.assertEqual(len(registered), 1, "nothing page-locked for the processor")
            self.assertFalse(on_processor.registered)
            on_card.close()
            on_processor.close()


if __name__ == "__main__":
    unittest.main()
