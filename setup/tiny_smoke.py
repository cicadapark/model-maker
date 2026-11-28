#!/usr/bin/env python3
"""The tiny-model smoke test: one object made through every stage on this computer's processor, each model built from
its real configuration shrunk to a few channels and layers, with random weights, at 64 pixels where the size can be
set. Nothing large is downloaded: the only files fetched are two models' own code (BiRefNet's, checked by its sums in
the registry; MV-Adapter's checkout at the registry's commit, checked by `registry.verify_checkout`).

It proves the code paths, on any system: the device and float type chosen, each model's operations present in the
installed wheels, every stage's own steps and files. It never proves an object good (random weights paint noise),
nor that a card's memory suffices.

    .venv/bin/python -I setup/tiny_smoke.py [--code DIR] [--keep DIR]

--code   where the two models' code is kept (fetched there when missing); default: a temporary folder
--keep   keep the tiny models and the made job in DIR (default: removed at the end)

What is not each model's own path, and why:
* pictures: FLUX.1 [schnell] loads in 4 bits through bitsandbytes on CUDA only, and refuses elsewhere before reading
  any weights (device.four_bit_refusal). The test checks that refusal, then gives the stage a stand-in painter that
  draws one fixed object, so the cut-out, the picture checks and the stage's files run as they do after a painting.
* cut-out: BiRefNet's size is written in its own code (checked by sum, never edited), so it is built whole, with
  random weights, without a weights file; it runs on every picture, at 256 pixels. Its mask is random, so the job is
  given the drawn object's mask after the model has run.
* SigLIP 2 (the subject and state judges) and the question judge are optional and not built: the stages say so and go
  on, as on a computer without them.
"""

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPOSITORY, "python"))
sys.path.insert(0, HERE)

# The processor, whatever this computer has: the size of a test every system can run.
os.environ["MODEL_MAKER_DEVICE"] = "cpu"

SIDE = 64
CUT_OUT_SIDE = 256
SHAPE_TOKENS = 64
SHAPE_GRID = 32
PROMPT = "a clay jug"
TINY_SHAPE = dict(width=32, num_layers=3, num_attention_heads=2, cross_attention_dim=32)


def say(text):
    print(f"tiny smoke: {text}", flush=True)


# The two models' own code, the only downloads.

def fetched_code(code):
    """BiRefNet's code files (by the registry's sums) and MV-Adapter's checkout (at the registry's commit) in `code`:
    (BiRefNet's folder, the sources folder)."""
    import fetch_models
    from model_maker import registry

    birefnet = os.path.join(code, "birefnet")
    entry = registry.models()["birefnet"]
    for part in entry["parts"]:
        name = part["url"].rsplit("/", 1)[-1]
        if part["size_mb"] or (os.path.isfile(os.path.join(birefnet, name))
                               and registry.sha256_of(os.path.join(birefnet, name)) == part["sha256"]):
            continue
        fetch_models.downloaded(part["url"], os.path.join(birefnet, name), part["sha256"], None,
                                fetch_models.Progress(0))
    sources = os.path.join(code, "sources")
    os.makedirs(sources, exist_ok=True)
    fetch_models.checked_out(registry.models()["mv-adapter-ig2mv-sdxl"]["source"], sources)
    return birefnet, sources


# The tiny models, written as the real ones are laid out under models/.

def save_shape_model(folder):
    """TripoSG's folder: DINOv2 and its picture processor, the denoiser and the decoder, a few channels each. The
    decoder's field is made to cross zero (random weights rarely give a surface): its query points read only their
    first two frequencies, and its last bias is set to the middle of the field over the shape's room."""
    import torch
    from safetensors.torch import save_file
    from transformers import Dinov2Config, Dinov2Model

    from model_maker import shape_net, shape_vae

    torch.manual_seed(0)
    encoder = Dinov2Config(hidden_size=TINY_SHAPE["cross_attention_dim"], num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=64, patch_size=14, image_size=28)
    Dinov2Model(encoder).save_pretrained(os.path.join(folder, "image_encoder_dinov2"))
    # The processor's config as TripoSG's is laid out; shape_net.PicturePixels reads it (no transformers class).
    os.makedirs(os.path.join(folder, "feature_extractor_dinov2"))
    with open(os.path.join(folder, "feature_extractor_dinov2", "preprocessor_config.json"), "w") as file:
        json.dump({"image_processor_type": "BitImageProcessor", "do_resize": True, "size": {"shortest_edge": 28},
                   "resample": 3, "do_center_crop": True, "crop_size": {"height": 28, "width": 28},
                   "do_rescale": True, "rescale_factor": 1 / 255, "do_normalize": True,
                   "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225],
                   "do_convert_rgb": True}, file, indent=2)
    denoiser = shape_net.Denoiser(**{**shape_net.CHOICES, **TINY_SHAPE})
    os.makedirs(os.path.join(folder, "transformer"))
    save_file(denoiser.state_dict(), os.path.join(folder, "transformer", "diffusion_pytorch_model.safetensors"))
    config = dict(latent_channels=64, width_decoder=32, num_attention_heads=2, num_layers_decoder=2,
                  embed_frequency=8, in_channels=3, embedding_type="frequency", embed_include_pi=False)
    decoder = shape_vae.ShapeDecoder(**shape_vae.decoder_settings(config)).eval()
    with torch.no_grad():
        decoder.decoder["proj_query"].weight[:, 3 + 3 * 2:3 + 3 * 8] = 0
        decoder.decoder["proj_query"].weight[:, 3 + 3 * 8 + 3 * 2:] = 0
        decoder.decoder["blocks"][-1].attn2.to_out[0].weight *= 0.01
        axis = torch.linspace(-0.9, 0.9, 12)
        points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(1, -1, 3)
        field = decoder.field(decoder.tokens(torch.randn(1, SHAPE_TOKENS, 64)), points)
        decoder.decoder["proj_out"].bias += field.median()
    os.makedirs(os.path.join(folder, "vae"))
    with open(os.path.join(folder, "vae", "config.json"), "w") as file:
        json.dump(config, file)
    save_file(decoder.state_dict(), os.path.join(folder, "vae", "diffusion_pytorch_model.safetensors"))


def tiny_tokenizer(folder):
    """A CLIP tokenizer of single characters (CLIP's own byte-pair code with no merges)."""
    from transformers import CLIPTokenizer

    characters = [chr(c) for c in range(33, 127)]
    vocabulary = {token: k for k, token in enumerate(characters + [c + "</w>" for c in characters]
                                                     + ["<|startoftext|>", "<|endoftext|>"])}
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "vocab.json"), "w") as file:
        json.dump(vocabulary, file)
    with open(os.path.join(folder, "merges.txt"), "w") as file:
        file.write("#version: 0.2\n")
    return CLIPTokenizer(os.path.join(folder, "vocab.json"), os.path.join(folder, "merges.txt"), model_max_length=77)


def save_sdxl(models, scratch):
    """SDXL base (both text encoders, their tokenizers, the UNet, the scheduler) and its fp16-safe autoencoder,
    written in fp16 with the "fp16" variant, as the real files are. The UNet has SDXL's block types and first two
    levels (320 and 640 channels, the widths MV-Adapter's condition encoder adds into them) without the third (1280):
    the third alone held three quarters of the weights, and the encoder's third output then goes unused."""
    import torch
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

    torch.manual_seed(0)
    tokenizer = tiny_tokenizer(os.path.join(scratch, "tokenizer"))
    text = dict(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                num_attention_heads=2, max_position_embeddings=77, bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id)
    unet = UNet2DConditionModel(
        sample_size=SIDE // 8, in_channels=4, out_channels=4, layers_per_block=1,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(320, 640), attention_head_dim=(5, 10), transformer_layers_per_block=1,
        cross_attention_dim=32, use_linear_projection=True, addition_embed_type="text_time",
        addition_time_embed_dim=8, projection_class_embeddings_input_dim=6 * 8 + 16)
    autoencoder = dict(in_channels=3, out_channels=3, latent_channels=4, layers_per_block=1, norm_num_groups=8,
                       block_out_channels=(8, 16, 16, 16), sample_size=SIDE, scaling_factor=0.13025,
                       down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4)
    scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                       num_train_timesteps=1000, prediction_type="epsilon", steps_offset=1,
                                       timestep_spacing="leading", interpolation_type="linear")
    pipe = StableDiffusionXLPipeline(
        vae=AutoencoderKL(**autoencoder), text_encoder=CLIPTextModel(CLIPTextConfig(**text)),
        text_encoder_2=CLIPTextModelWithProjection(CLIPTextConfig(**text, projection_dim=16)), tokenizer=tokenizer,
        tokenizer_2=tokenizer, unet=unet, scheduler=scheduler)
    pipe.to(dtype=torch.float16)
    sdxl = os.path.join(models, "sdxl-base-1.0")
    pipe.save_pretrained(sdxl, variant="fp16")
    AutoencoderKL(**autoencoder).to(torch.float16).save_pretrained(os.path.join(models, "sdxl-vae-fp16-fix"))
    return sdxl


def save_adapter(models, sdxl, sources):
    """MV-Adapter's weights for the tiny UNet: its own pipeline's adapter (the UNet's multi-view and reference
    attention, the condition encoder) made and saved by its own code, as its file holds them."""
    import torch
    from safetensors.torch import save_file

    from model_maker import paint

    os.environ["MV_ADAPTER_SRC"] = os.path.join(sources, "mv-adapter")
    paint._mv_adapter_on_path()
    from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
    from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline

    torch.manual_seed(0)
    pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(sdxl, variant="fp16", torch_dtype=torch.float32)
    pipe.init_custom_adapter(num_views=6, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0)
    state = pipe._save_custom_adapter(include_keys=["_mv", "_ref"])
    folder = os.path.join(models, "mv-adapter")
    os.makedirs(folder)
    save_file({name: tensor.to(torch.float16).contiguous() for name, tensor in state.items()},
              os.path.join(folder, "mvadapter_ig2mv_sdxl.safetensors"))
    del pipe, state
    gc.collect()


# The stand-ins where a model cannot run here (see the module's note).

def drawn_object():
    """A jug-like object (a body and a neck) on the painter's grey backdrop: (picture, its mask)."""
    import numpy as np
    from PIL import Image

    from model_maker import painter

    side = painter.SIDE
    rows, cols = np.mgrid[0:side, 0:side] / side
    body = (rows - 0.62) ** 2 / 0.06 + (cols - 0.5) ** 2 / 0.05 < 1
    neck = (abs(cols - 0.5) < 0.09) & (rows > 0.2) & (rows < 0.45)
    mask = body | neck
    picture = np.full((side, side, 3), 200, np.uint8)
    picture[mask] = (150 + 60 * rows[mask, None] * np.array([1.0, 0.6, 0.3])).astype(np.uint8)
    return Image.fromarray(picture), mask


class DrawingPainter:
    """In FLUX.1's place on a computer it refuses: every try is the drawn object."""

    def paint(self, prompt, seed):
        return drawn_object()[0]

    def close(self):
        pass


class RunCutter:
    """BiRefNet's cutter, run on every picture as in the stage; the job gets the drawn object's mask, since random
    weights cut nothing meaningful."""

    def __init__(self, cutter):
        self.cutter, self.ran = cutter, 0

    def mask(self, image):
        found = self.cutter.mask(image)
        if found.shape != (image.height, image.width) or found.dtype != bool:
            raise AssertionError(f"the cutter's mask is {found.shape} {found.dtype}, not the picture's size in bool")
        self.ran += 1
        return drawn_object()[1]

    def close(self):
        self.cutter.model.to("meta")


# Checks on what ran.

def on_the_processor_in_float32(name, module):
    import torch

    kinds = {(p.device.type, p.dtype) for p in module.parameters() if p.is_floating_point()}
    if kinds != {("cpu", torch.float32)}:
        raise AssertionError(f"{name}: its weights are {sorted(map(str, kinds))}, not the processor's float32")


def birefnet_cutter(folder):
    """`cutout.Cutter` on BiRefNet's own code with random weights: its loader builds the model from the folder's
    configuration instead of reading a weights file (there is none here)."""
    import types

    from model_maker import cutout

    package = types.ModuleType("birefnet_folder")
    package.__path__ = [folder]
    sys.modules["birefnet_folder"] = package
    from birefnet_folder.birefnet import BiRefNet

    BiRefNet.from_pretrained = classmethod(lambda cls, path, **_: cls(config=cls.config_class.from_pretrained(path)))
    return cutout.Cutter(folder)


def checked_refusal():
    """FLUX.1's 4-bit loading refused on this device before any weights are read, saying why."""
    from model_maker import device, painter

    chosen = device.current()
    if chosen.name != "cpu":
        raise AssertionError(f"the device layer chose {chosen.name}, not the processor MODEL_MAKER_DEVICE names")
    why = device.stage_refusal("pictures", chosen)
    if not why or "4-bit" not in why and "4 bit" not in why:
        raise AssertionError(f"the pictures stage is not refused on the processor: {why!r}")
    with tempfile.TemporaryDirectory() as empty:
        try:
            painter.Painter(empty)
        except RuntimeError as error:
            if str(error) != device.four_bit_refusal(chosen):
                raise
        else:
            raise AssertionError("FLUX.1's painter loaded on the processor")
    say(f"pictures: FLUX.1's 4-bit loading refused before reading weights, as it should: {why.splitlines()[0]}")


def shrunk():
    """The sizes the stages take from their modules, made small: the pictures' cut-out, the shape's latent and grid,
    the painted views and touch-up views."""
    import functools

    from model_maker import cutout, paint, shape_net, touch_up

    cutout.SIDE = CUT_OUT_SIDE
    shape_net.TOKENS = SHAPE_TOKENS
    shape_net.CHOICES = {**shape_net.CHOICES, **TINY_SHAPE}
    shape_net.surface = functools.partial(shape_net.surface, coarse=SHAPE_GRID)
    paint.SIDE = touch_up.SIDE = SIDE


# The tiny models' folders, by registry id and the pipeline's folder name (kept.py loads each by these).
TINY_FOLDERS = (("triposg", "triposg"), ("sdxl-base-1.0", "sdxl-base-1.0"), ("sdxl-vae-fp16-fix", "sdxl-vae"),
                ("mv-adapter-ig2mv-sdxl", "mv-adapter"))


def as_set_up(model_id, folder):
    """Only the configurations and code files the registry lists for `model_id` left in `folder`, as setup writes a
    model folder: a library's save_pretrained writes more than a model's release holds (transformers 5 adds a
    tokenizer.json beside CLIP's vocab.json and merges.txt), and the pipeline refuses an unlisted one even with the
    sums unchecked (registry.refuse_unlisted). The weights and other data files stay: their names are the real ones."""
    from model_maker import registry

    listed = registry.listed_sums(model_id)
    for root, _, names in os.walk(folder):
        for name in names:
            path = os.path.join(root, name)
            inside = os.path.relpath(path, folder).replace(os.sep, "/")
            if inside not in listed and name.lower().endswith(registry.CONFIG_ENDINGS + registry.CODE_ENDINGS):
                os.remove(path)
                say(f"{model_id}: {inside} removed (the release has no such file)")


def made(work, code_dir):
    from model_maker import credentials, device, kept, pipeline, shape_net, touch_up

    os.environ.setdefault(credentials.TEST_FOLDER_VARIABLE, os.path.join(work, "credentials"))
    os.environ.setdefault("MODEL_MAKER_LEDGER", os.path.join(work, "metrics"))
    say(f"device: {device.current().name}; float types: " + ", ".join(
        f"{model} {device.current().dtype(model)}" for model in device.CARD_DTYPES))
    checked_refusal()
    birefnet, sources = fetched_code(code_dir)
    models = os.path.join(work, "models")
    started = time.time()
    save_shape_model(os.path.join(models, "triposg"))
    sdxl = save_sdxl(models, os.path.join(work, "scratch"))
    save_adapter(models, sdxl, sources)
    for model_id, name in TINY_FOLDERS:
        as_set_up(model_id, os.path.join(models, pipeline.MODEL_FOLDERS[name]))
    say(f"tiny models written in {time.time() - started:.0f} s{held()}")
    shrunk()

    settings = pipeline.Settings(models=models, sources=sources, tries=2, shape_tries=1, triangles=1500,
                                 texture=256, verify=False, paint_steps=2, picture_rounds=1, touch_up_views=1)
    job = pipeline.Job(pipeline.job_folder(os.path.join(work, "made"), PROMPT), PROMPT, settings=settings)
    models_kept = kept.Kept()

    # Pictures: the stage as the pipeline runs it, past the refusal checked above.
    cutter = RunCutter(birefnet_cutter(birefnet))
    on_the_processor_in_float32("cut-out", cutter.cutter.model)
    models_kept.models.update({"picture painter": DrawingPainter(), "cutter": cutter})
    timed("pictures", lambda: past_the_refusal(pipeline, lambda: pipeline.pictures_stage(
        [job], settings, report, kept=models_kept)))
    if cutter.ran < settings.tries:
        raise AssertionError(f"the cutter ran on {cutter.ran} pictures, not {settings.tries}")

    # Shape: TripoSG's tiny folder, loaded in the float type the stage asks for (the stage finds them loaded).
    shape_net.WEIGHTS = os.path.join(models, "triposg")
    loaded = shape_net.kept_models(device.current().dtype("shape"))
    for name in ("denoiser", "vae"):
        on_the_processor_in_float32(f"shape {name}", loaded[name])
    on_the_processor_in_float32("shape encoder", loaded["encoder"][1])
    timed("shape", lambda: pipeline.shape_stage([job], settings, report))

    # Paint: the view painter loaded by the stage; the inpainter given with few steps.
    inpainter = touch_up.Inpainter(sdxl, os.path.join(models, "sdxl-vae-fp16-fix"), steps=3,
                                   strength=settings.touch_up_strength)
    on_the_processor_in_float32("inpainter", inpainter.pipe.unet)
    models_kept.models[f"inpainter {settings.touch_up_strength}"] = inpainter
    views = models_kept.view_painter(settings)
    on_the_processor_in_float32("view painter", views.pipe.unet)
    if views.offload is not None:
        raise AssertionError("the view painter offloads its blocks on the processor")
    timed("paint", lambda: pipeline.paint_stage([job], settings, report, kept=models_kept))

    # Finish: the bake, the checks, the export, the report (the question judge is not here).
    timed("finish", lambda: pipeline.finish_stage([job], settings, report))
    return job


def report(text):
    print(f"  {text}", flush=True)


def past_the_refusal(pipeline, run):
    """The stage run with its card check passed: the refusal itself was checked by `checked_refusal`."""
    checked = pipeline.needs_card
    pipeline.needs_card = lambda stage: None if stage == "pictures" else checked(stage)
    try:
        run()
    finally:
        pipeline.needs_card = checked


def timed(stage, run):
    started = time.time()
    run()
    say(f"{stage}: done in {time.time() - started:.0f} s{held()}")


def held():
    """The process's memory now and at its peak so far, where the system says (memory.py)."""
    from model_maker import memory

    now, peak = memory.in_use_mb(), memory.peak_mb()
    return f" (memory {now:.0f} MB, peak {peak:.0f} MB)" if now and peak else ""


def checked_job(job):
    """Every stage's file there, the object exported as a glTF binary."""
    for name in ("picture.png", "mask.png", "shape.npz", "painted.npz", f"{job['slug']}.glb", "job.json"):
        if not job.has(name):
            raise AssertionError(f"{name} was not made")
    with open(job.path(f"{job['slug']}.glb"), "rb") as file:
        data = file.read()
    if data[:4] != b"glTF":
        raise AssertionError("the .glb is not a glTF binary")
    with open(job.path("job.json")) as file:
        record = json.load(file)
    say(f"made {job['slug']}.glb ({len(data) // 1024} KB); status: {record.get('status')}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--code", help="where the two models' code is kept (fetched there when missing)")
    parser.add_argument("--keep", help="keep the tiny models and the made job in this folder")
    options = parser.parse_args(argv)
    started = time.time()
    work = options.keep or tempfile.mkdtemp(prefix="model-maker-tiny-smoke-")
    os.makedirs(work, exist_ok=True)
    code = options.code or os.path.join(work, "code")
    try:
        checked_job(made(work, code))
    finally:
        if not options.keep:
            shutil.rmtree(work, ignore_errors=True)
    say(f"every stage ran on the processor with tiny models in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
