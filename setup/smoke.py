#!/usr/bin/env python3
"""setup.sh's last check: every stage's code imports in the environment, as the pipeline imports it, without loading
any weights (a few seconds). It catches a package the lock lacks before the first object is made: the first clean
install lacked setuptools, which triton (under bitsandbytes) imports when FLUX.1 [schnell] is quantised.

    .venv/bin/python -I setup/smoke.py MODELS SOURCES
"""

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))


def main(models, sources):
    os.environ["MV_ADAPTER_SRC"] = os.path.join(sources, "mv-adapter")
    import torch

    from model_maker import pipeline, registry

    # Pictures: FLUX.1 [schnell] in 4 bits, BiRefNet's own code (checked first), SigLIP 2.
    from diffusers import BitsAndBytesConfig, FluxPipeline, FluxTransformer2DModel  # noqa: F401
    from transformers import BitsAndBytesConfig as TransformersBnb, SiglipModel, T5EncoderModel  # noqa: F401
    import bitsandbytes  # noqa: F401

    birefnet = os.path.join(models, pipeline.MODEL_FOLDERS["birefnet"])
    registry.verify_folder("birefnet", birefnet)
    package = types.ModuleType("birefnet_folder")
    package.__path__ = [birefnet]
    sys.modules["birefnet_folder"] = package
    from birefnet_folder.birefnet import BiRefNet  # noqa: F401

    # Shape: TripoSG's weights with model-maker's own denoiser and decoder (on diffusers' attention parts).
    from model_maker import shape_net, shape_vae  # noqa: F401
    from diffusers.models.attention import FeedForward  # noqa: F401
    from diffusers.models.attention_processor import Attention  # noqa: F401
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps  # noqa: F401
    from diffusers.models.normalization import FP32LayerNorm  # noqa: F401
    from transformers import BitImageProcessor, Dinov2Model  # noqa: F401
    from skimage import measure  # noqa: F401

    # Paint: MV-Adapter's pipeline on SDXL.
    from model_maker import paint

    paint._mv_adapter_on_path()
    from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0  # noqa: F401
    from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline  # noqa: F401
    from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler  # noqa: F401

    paint.cameras()
    # Finish: the checks, the bake, the export, the report.
    from model_maker import bake, glb, inspection, levelling, likeness, render, report  # noqa: F401

    print(f"every stage's code imports (PyTorch {torch.__version__})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
