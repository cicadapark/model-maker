#!/usr/bin/env python3
"""The model code's inputs and outputs under two Python environments (the old pins and the new), on the CPU, compared
against bars fixed before the run (a dependency update's first test: no card).

    python stack_compare.py dump OUT PART...        record PART under this environment into the folder OUT
    python stack_compare.py compare OLD NEW [--exact]  every part both folders hold, against its bar; exit 1 on a miss
                                                     (--exact: every bar byte for byte, for a change on one environment)

Run `dump` once with each environment's Python, the same code, the same models (MODEL_MAKER_MODELS, and
MODEL_MAKER_SOURCES for the view painter's code), then `compare` with either. The parts and their bars:

  tokens    every prompt of the trial lists and case files, the picture-word dictionary and the words the pipeline
            builds from each (the picture model's CLIP and T5 words, the view painter's and the touch-up's two CLIP
            tokenizers, the judge's Gemma words): token ids, exactly equal
  inputs    each labelled picture of the regression data (REGRESSION_DATA): the pixels BiRefNet, SigLIP 2 (whole and
            cropped) and DINOv2 are given, exactly equal; SigLIP 2's features within 1e-5 (and the regression's
            verdicts, compared from its own runs); DINOv2's features within 1e-4
  noise     the seeded starting noise of the four samplers (the picture model's, the shape model's, the view
            painter's, the touch-up's), each drawn by its own pipeline code: exactly equal
  cutout    BiRefNet's mask of a few pictures on the CPU: exactly equal
  shape     two of the shape model's denoising steps on the CPU, from the same picture: within 1e-4
  view      one step of the view painter (MV-Adapter on SDXL) at 128 pixels on the CPU: within 1.5 of 255 on average
  siglip    SigLIP 2's part of `inputs` alone (pixels and image features), compared with either record
  judge     the question judge's test cases (llama.cpp, not Python's stack): shares, verdicts and words exactly equal
  glbs      (compare only) stored jobs finished again under each environment, in OLD/jobs and NEW/jobs: each .glb's
            document, geometry, uvs and texture pixels exactly equal

Nothing is fetched: every model is read from its folder, offline.
"""

import glob
import hashlib
import json
import os
import sys
import time
import tomllib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(REPOSITORY, "python"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

LABELLED = ("states", "prompts", "heads", "qualities", "lettering")
# Bars, fixed in the plan before anything was run (never loosened).
EXACT = 0.0
SIGLIP_FEATURES = 1e-5
DINO_FEATURES = 1e-4
SHAPE_STEPS = 1e-4
VIEW_STEP_OF_255 = 1.5
SEED = 42


def digest(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.view(dtype=__import__("torch").int16)
        value = value.numpy()
    if hasattr(value, "convert") and hasattr(value, "size"):
        value = np.asarray(value)
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes() + str(array.shape).encode() + str(array.dtype).encode()).hexdigest()


def models():
    return os.environ["MODEL_MAKER_MODELS"]


def versions():
    import importlib.metadata as meta

    found = {}
    for name in ("torch", "torchvision", "transformers", "diffusers", "accelerate", "huggingface-hub", "tokenizers",
                 "pillow", "numpy", "safetensors", "timm", "kornia", "peft", "bitsandbytes"):
        try:
            found[name] = meta.version(name)
        except meta.PackageNotFoundError:
            found[name] = None
    return found


# ---------------------------------------------------------------------------------------------------------- tokens

def prompt_lists():
    prompts = set()
    for path in sorted(glob.glob(os.path.join(HERE, "*.txt"))):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.add(line)

    def walk(value):
        if isinstance(value, dict):
            for key, inner in value.items():
                if key == "prompt" and isinstance(inner, str):
                    prompts.add(inner)
                else:
                    walk(inner)
        elif isinstance(value, list):
            for inner in value:
                walk(inner)

    for path in sorted(glob.glob(os.path.join(HERE, "*.toml"))):
        with open(path, "rb") as file:
            walk(tomllib.load(file))
    for path in (os.path.join(REPOSITORY, "python", "tests", "prompt_cases.json"),
                 os.path.join(REPOSITORY, "python", "tests", "question-judge-cases", "cases.toml")):
        if path.endswith(".json"):
            walk(json.load(open(path)))
        else:
            with open(path, "rb") as file:
                walk(tomllib.load(file))
    return sorted(prompts)


def texts_by_tokenizer():
    """{tokenizer: sorted texts}: each prompt and the words the pipeline builds from it, and the dictionary."""
    from model_maker import paint, touch_up, words

    prompts = prompt_lists()
    dictionary = json.load(open(os.path.join(REPOSITORY, "python", "model_maker", "picture_words.json")))["entries"]
    # Each prompt as the pipeline reads it (words.cleaned: markup and stray signs gone); the raw text never reaches a
    # tokenizer, and those that differ are tokenized apart, as a note ("raw").
    plain = ({words.cleaned(p) for p in prompts} | {e["name"] for e in dictionary} | {e["words"] for e in dictionary}
             | set(words.known_things()))
    flux, sdxl, gemma = set(plain), set(plain), {t.lower() for t in plain}
    sdxl |= {paint.NEGATIVE, f"{paint.NEGATIVE}, {touch_up.FILL_NEGATIVE}"}
    refused = []
    for prompt in prompts:
        try:
            subject = words.subject_of(prompt)
        except Exception as error:  # noqa: BLE001 - a prompt the words refuse is listed, not hidden
            refused.append(f"{prompt}: {type(error).__name__}")
            continue
        flux |= {words.picture_prompt(subject, lead) for lead in words.LEADS}
        said = words.paint_prompt(subject)
        sdxl |= {said, f"{said}, {touch_up.FILL_WORDS}"}
        wanted, unwanted = words.state_texts(subject)
        asked, others = words.subject_contrast(subject)
        judged = [*wanted, *unwanted, *asked, *(t for texts in others.values() for t in texts)]
        for _, yes, no in words.quality_texts(subject):
            judged += [*yes, *no]
        gemma |= {t.lower() for t in judged}
    raw = sorted(p for p in prompts if words.cleaned(p) != p)
    return {"flux.clip": sorted(flux), "flux.t5": sorted(flux), "sdxl.clip": sorted(sdxl),
            "sdxl.clip2": sorted(sdxl), "siglip.gemma": sorted(gemma), "raw": raw}, len(prompts), refused


def tokenizer_of(folder, sub):
    import transformers

    index = json.load(open(os.path.join(folder, "model_index.json")))
    _, name = index[sub]
    return getattr(transformers, name).from_pretrained(os.path.join(folder, sub)), name


def dump_tokens(out):
    from transformers import GemmaTokenizerFast

    from model_maker import painter, state_judge

    flux = os.path.join(models(), "flux1-schnell")
    sdxl = os.path.join(models(), "sdxl-base-1.0")
    loaded = {"flux.clip": tokenizer_of(flux, "tokenizer"), "flux.t5": tokenizer_of(flux, "tokenizer_2"),
              "sdxl.clip": tokenizer_of(sdxl, "tokenizer"), "sdxl.clip2": tokenizer_of(sdxl, "tokenizer_2"),
              "siglip.gemma": (GemmaTokenizerFast.from_pretrained(
                  os.path.join(models(), state_judge.MODEL_FOLDER)), "GemmaTokenizerFast")}
    # The pipelines' own calls: padded to the length each model reads, cut there (diffusers' encode_prompt,
    # state_judge.features); and the whole text without padding or cutting, which must agree too.
    longest = {"flux.clip": 77, "flux.t5": painter.MOST_TOKENS, "siglip.gemma": state_judge.TEXT_TOKENS}
    texts, prompt_count, refused = texts_by_tokenizer()
    found = {}
    for name, (tokenizer, kind) in loaded.items():
        most = longest.get(name, tokenizer.model_max_length)
        rows, raw = {}, {}
        for text in texts["raw"]:
            raw[text] = list(tokenizer(text)["input_ids"])
        for text in texts[name]:
            padded = tokenizer(text, padding="max_length", max_length=most, truncation=True, return_tensors="np")
            whole = tokenizer(text)["input_ids"]
            rows[text] = [np.asarray(padded["input_ids"]).ravel().tolist(), list(whole)]
        found[name] = {"class": type(tokenizer).__name__, "named": kind, "length": most, "ids": rows, "raw": raw}
    json.dump({"versions": versions(), "prompts": prompt_count, "refused": refused, "tokens": found},
              open(os.path.join(out, "tokens.json"), "w"))
    print(f"tokens: {prompt_count} prompts, " + ", ".join(f"{k} {len(v['ids'])} texts" for k, v in found.items()))


def compare_tokens(old, new):
    a = json.load(open(os.path.join(old, "tokens.json")))["tokens"]
    b = json.load(open(os.path.join(new, "tokens.json")))["tokens"]
    rows = []
    for name in a:
        texts = a[name]["ids"]
        missing = set(texts) ^ set(b[name]["ids"])
        padded = [t for t in texts if t in b[name]["ids"] and texts[t][0] != b[name]["ids"][t][0]]
        whole = [t for t in texts if t in b[name]["ids"] and texts[t][1] != b[name]["ids"][t][1]]
        value = f"{len(texts)} texts; {len(padded)} padded and {len(whole)} whole differ; {len(missing)} unmatched"
        if padded or whole:
            value += f"; e.g. {(padded or whole)[0]!r}"
        rows.append((f"token ids, {name} ({a[name]['class']} / {b[name]['class']})", "exactly equal", value,
                     not (padded or whole or missing)))
        raw = a[name].get("raw", {})
        moved = [t for t in raw if raw[t] != b[name].get("raw", {}).get(t)]
        rows.append((f"note: raw prompts the pipeline cleans first, {name}", "shown, not a test",
                     f"{len(moved)} of {len(raw)} raw texts tokenize differently" + (f": {moved!r}" if moved else ""),
                     None))
    return rows


# ---------------------------------------------------------------------------------------------------------- inputs

def labelled_pictures(data):
    found = []
    for folder in LABELLED:
        for path in sorted(glob.glob(os.path.join(data, folder, "*.png"))):
            if not path.endswith("-mask.png"):
                found.append(path)
    return found


class Captured:
    """An encoder that hands back what it was given and the real encoder's output."""

    def __init__(self, encoder):
        self.encoder, self.pixels = encoder, None

    def __call__(self, pixels):
        self.pixels = pixels
        return self.encoder(pixels)


def birefnet_pixels(rgb):
    """BiRefNet's input as cutout.Cutter.mask makes it."""
    import torch
    from PIL import Image

    from model_maker import cutout

    small = rgb.resize((cutout.SIDE, cutout.SIDE), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return ((x - torch.tensor(cutout.MEAN)[:, None, None]) / torch.tensor(cutout.STD)[:, None, None])[None]


def siglip_pixels(judge, image):
    """What the judge gives SigLIP 2: its own steps (state_judge.PicturePixels), or transformers' processor before."""
    if hasattr(judge, "pixels"):
        return judge.pixels(image)
    return judge.images(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]


def dump_inputs(out):
    import torch
    from PIL import Image

    from model_maker import shape_net, state_judge

    data = os.environ["REGRESSION_DATA"]
    shape_net.WEIGHTS = os.path.join(models(), "triposg")
    processor, encoder = shape_net.load_encoder("cpu", torch.float32)
    judge = state_judge.StateJudge(os.path.join(models(), state_judge.MODEL_FOLDER), device="cpu")
    texts, _, _ = texts_by_tokenizer()
    os.makedirs(os.path.join(out, "dino"), exist_ok=True)
    pixels, siglip = {}, {}
    started = time.time()
    for path in labelled_pictures(data):
        name = os.path.relpath(path, data)
        rgb = Image.open(path).convert("RGB")
        mask = np.asarray(Image.open(path[:-4] + "-mask.png").convert("L")) > 127
        row = {"birefnet": digest(birefnet_pixels(rgb)), "siglip.whole": digest(siglip_pixels(judge, rgb))}
        views = {"whole": judge.features(image=rgb)[0]}
        if mask.any():
            crop = state_judge.cropped(rgb, mask)
            row["siglip.crop"] = digest(siglip_pixels(judge, crop))
            views["crop"] = judge.features(image=crop)[0]
            captured = Captured(encoder)
            features = shape_net.picture_features(processor, captured, shape_net.prepared(np.asarray(rgb), mask),
                                                  "cpu", torch.float32)
            row["dino"] = digest(captured.pixels)
            np.save(os.path.join(out, "dino", name.replace("/", "__") + ".npy"), features[0].numpy())
        pixels[name] = row
        siglip[name] = {k: v.tolist() for k, v in views.items()}
    words = texts["siglip.gemma"]
    text_features = np.concatenate([judge.features(texts=words[k:k + 32]) for k in range(0, len(words), 32)])
    np.save(os.path.join(out, "siglip-texts.npy"), text_features)
    json.dump({"versions": versions(), "pixels": pixels, "siglip": siglip, "texts": words},
              open(os.path.join(out, "inputs.json"), "w"))
    print(f"inputs: {len(pixels)} pictures, {len(words)} texts in {time.time() - started:.0f} s")


def compare_inputs(old, new):
    a = json.load(open(os.path.join(old, "inputs.json")))
    b = json.load(open(os.path.join(new, "inputs.json")))
    rows = []
    names = sorted(a["pixels"])
    same_set = names == sorted(b["pixels"])
    for key, label in (("birefnet", "BiRefNet"), ("siglip.whole", "SigLIP 2 (whole)"),
                       ("siglip.crop", "SigLIP 2 (cropped)"), ("dino", "DINOv2")):
        had = [n for n in names if key in a["pixels"][n]]
        differ = [n for n in had if a["pixels"][n][key] != b["pixels"].get(n, {}).get(key)]
        rows.append((f"input pixels, {label}", "exactly equal",
                     f"{len(differ)} of {len(had)} pictures differ" + (f" (e.g. {differ[0]})" if differ else ""),
                     same_set and not differ))
    worst, where = 0.0, ""
    for name in names:
        for view, value in a["siglip"][name].items():
            gap = float(np.max(np.abs(np.asarray(value) - np.asarray(b["siglip"][name][view]))))
            if gap > worst:
                worst, where = gap, f"{name} ({view})"
    rows.append(("SigLIP 2 image features (unit vectors)", f"within {SIGLIP_FEATURES:g}",
                 f"{worst:.3g} at most ({where or 'none'}) over {len(names)} pictures", worst <= SIGLIP_FEATURES))
    ta, tb = np.load(os.path.join(old, "siglip-texts.npy")), np.load(os.path.join(new, "siglip-texts.npy"))
    gap = float(np.max(np.abs(ta - tb))) if a["texts"] == b["texts"] else float("inf")
    rows.append(("SigLIP 2 text features (unit vectors)", f"within {SIGLIP_FEATURES:g}",
                 f"{gap:.3g} at most over {len(a['texts'])} texts", gap <= SIGLIP_FEATURES))
    worst, where, count, total = 0.0, "", 0, 0.0
    for path in sorted(glob.glob(os.path.join(old, "dino", "*.npy"))):
        other = os.path.join(new, "dino", os.path.basename(path))
        x, y = np.load(path), np.load(other)
        gap = float(np.max(np.abs(x - y)))
        total += float(np.mean(np.abs(x - y)))
        count += 1
        if gap > worst:
            worst, where = gap, os.path.basename(path)[:-4]
    rows.append(("DINOv2 features (last hidden state)", f"within {DINO_FEATURES:g}",
                 f"{worst:.3g} at most ({where}), {total / max(count, 1):.2g} on average, over {count} pictures",
                 count > 0 and worst <= DINO_FEATURES))
    return rows


def dump_siglip(out):
    """SigLIP 2's part of `inputs` alone: each labelled picture's pixels (whole and cropped) and image features."""
    from PIL import Image

    from model_maker import state_judge

    data = os.environ["REGRESSION_DATA"]
    judge = state_judge.StateJudge(os.path.join(models(), state_judge.MODEL_FOLDER), device="cpu")
    pixels, siglip = {}, {}
    for path in labelled_pictures(data):
        name = os.path.relpath(path, data)
        rgb = Image.open(path).convert("RGB")
        mask = np.asarray(Image.open(path[:-4] + "-mask.png").convert("L")) > 127
        row = {"siglip.whole": digest(siglip_pixels(judge, rgb))}
        views = {"whole": judge.features(image=rgb)[0]}
        if mask.any():
            crop = state_judge.cropped(rgb, mask)
            row["siglip.crop"] = digest(siglip_pixels(judge, crop))
            views["crop"] = judge.features(image=crop)[0]
        pixels[name] = row
        siglip[name] = {k: v.tolist() for k, v in views.items()}
    json.dump({"versions": versions(), "pixels": pixels, "siglip": siglip},
              open(os.path.join(out, "siglip.json"), "w"))
    print(f"siglip: {len(pixels)} pictures")


def compare_siglip(old, new):
    """SigLIP 2's pixels and image features; either side may be a full `inputs` record."""
    def load(folder):
        path = os.path.join(folder, "siglip.json")
        return json.load(open(path if os.path.exists(path) else os.path.join(folder, "inputs.json")))

    a, b = load(old), load(new)
    names = sorted(a["pixels"])
    rows = []
    for key, label in (("siglip.whole", "SigLIP 2 (whole)"), ("siglip.crop", "SigLIP 2 (cropped)")):
        had = [n for n in names if key in a["pixels"][n]]
        differ = [n for n in had if a["pixels"][n][key] != b["pixels"].get(n, {}).get(key)]
        rows.append((f"input pixels, {label}", "exactly equal",
                     f"{len(differ)} of {len(had)} pictures differ" + (f" (e.g. {differ[0]})" if differ else ""),
                     names == sorted(b["pixels"]) and not differ))
    worst, where = 0.0, ""
    for name in names:
        for view, value in a["siglip"][name].items():
            gap = float(np.max(np.abs(np.asarray(value) - np.asarray(b["siglip"][name][view]))))
            if gap > worst:
                worst, where = gap, f"{name} ({view})"
    rows.append(("SigLIP 2 image features (unit vectors)", f"within {SIGLIP_FEATURES:g}",
                 f"{worst:.3g} at most ({where or 'none'}) over {len(names)} pictures", worst <= SIGLIP_FEATURES))
    return rows


# ----------------------------------------------------------------------------------------------------------- noise

class StubVae:
    """A VAE whose encoding is a fixed Gaussian (mean and log-variance from a fixed draw): the touch-up's draws from
    its generator (the picture's latents sampled, then the noise) without the VAE's arithmetic."""

    def __init__(self, latents_shape):
        import torch
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

        fixed = torch.Generator("cpu").manual_seed(7)
        self.parameters = torch.randn((1, latents_shape[1] * 2, *latents_shape[2:]), generator=fixed)
        self.distribution = DiagonalGaussianDistribution
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.config = type("Config", (), {"force_upcast": False, "scaling_factor": 0.13025, "shift_factor": None,
                                          "latents_mean": None, "latents_std": None})()

    def encode(self, image, return_dict=True):
        return type("Encoded", (), {"latent_dist": self.distribution(self.parameters)})()

    def to(self, *args, **kwargs):
        return self


def unweighted_vae(folder):
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_config(AutoencoderKL.load_config(folder))


def empty_unet(folder):
    """SDXL's UNet of its real configuration on no device (no weights, no memory): its constructor reads its config."""
    from accelerate import init_empty_weights
    from diffusers import UNet2DConditionModel

    with init_empty_weights():
        return UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(folder))


def dump_noise(out):
    import torch
    from diffusers import (EulerDiscreteScheduler, FluxPipeline,
                           StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline)

    from model_maker import paint, painter, shape_net, touch_up

    sys.path.insert(0, os.environ["MODEL_MAKER_SOURCES"] + "/mv-adapter")
    from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

    flux_dir = os.path.join(models(), "flux1-schnell")
    sdxl_dir = os.path.join(models(), "sdxl-base-1.0")
    found = {}
    cpu = torch.device("cpu")

    def generator():
        return torch.Generator("cpu").manual_seed(SEED)

    # The picture model: FluxPipeline.prepare_latents, 16 channels, 1024 pixels, bfloat16, the generator on the CPU. The
    # pipelines are built by their own constructors, with a VAE of the real configuration (its weights not read) and no
    # other model (SDXL's UNet on no device), so each version's own arithmetic runs.
    flux = FluxPipeline(scheduler=None, vae=unweighted_vae(os.path.join(flux_dir, "vae")), text_encoder=None,
                        tokenizer=None, text_encoder_2=None, tokenizer_2=None, transformer=None)
    latents, ids = flux.prepare_latents(1, 16, painter.SIDE, painter.SIDE, torch.bfloat16, cpu, generator())
    found["picture"] = latents
    found["picture.ids"] = ids
    # The shape model: our own draw.
    found["shape"] = shape_net.start_noise(SEED)
    # The view painter: SDXL's prepare_latents under MV-Adapter's shifted scheduler, six views, float16.
    shifted = ShiftSNRScheduler.from_scheduler(EulerDiscreteScheduler.from_pretrained(sdxl_dir, subfolder="scheduler"),
                                               shift_mode="interpolated", shift_scale=8.0)
    shifted.set_timesteps(paint.STEPS, device=cpu)
    sdxl_vae = unweighted_vae(os.path.join(models(), "sdxl-vae-fp16-fix"))
    unet = empty_unet(os.path.join(sdxl_dir, "unet"))
    views = StableDiffusionXLPipeline(vae=sdxl_vae, text_encoder=None, text_encoder_2=None, tokenizer=None,
                                      tokenizer_2=None, unet=unet, scheduler=shifted)
    found["views"] = views.prepare_latents(6, 4, paint.SIDE, paint.SIDE, torch.float16, cpu, generator())
    # The touch-up: SDXL inpainting's prepare_latents at its strength (the picture's latents sampled, then noised).
    inpaint_scheduler = EulerDiscreteScheduler.from_pretrained(sdxl_dir, subfolder="scheduler")
    inpaint_scheduler.set_timesteps(30, device=cpu)
    start = int(30 * touch_up.STRENGTH)
    timesteps = inpaint_scheduler.timesteps[30 - start:]
    shape = (1, 4, touch_up.SIDE // 8, touch_up.SIDE // 8)
    pipe = StableDiffusionXLInpaintPipeline(vae=sdxl_vae, text_encoder=None, text_encoder_2=None, tokenizer=None,
                                            tokenizer_2=None, unet=unet, scheduler=inpaint_scheduler)
    object.__setattr__(pipe, "vae", StubVae(shape))
    image = torch.zeros((1, 3, touch_up.SIDE, touch_up.SIDE))
    made = pipe.prepare_latents(
        1, 4, touch_up.SIDE, touch_up.SIDE, torch.float16, cpu, generator(), image=image,
        timestep=timesteps[:1].repeat(1), is_strength_max=False, add_noise=True, return_noise=True,
        return_image_latents=True)
    for key, value in zip(("touch_up", "touch_up.noise", "touch_up.image"), made):
        found[key] = value
    os.makedirs(os.path.join(out, "noise"), exist_ok=True)
    for key, value in found.items():
        value = value.detach().cpu()
        np.save(os.path.join(out, "noise", key + ".npy"), value.float().numpy())
    json.dump({"versions": versions(), "noise": {k: {"sha256": digest(v), "shape": list(v.shape),
                                                     "dtype": str(v.dtype)} for k, v in found.items()}},
              open(os.path.join(out, "noise.json"), "w"), indent=1)
    print("noise: " + ", ".join(f"{k} {tuple(v.shape)} {v.dtype}" for k, v in found.items()))


def compare_noise(old, new):
    a = json.load(open(os.path.join(old, "noise.json")))["noise"]
    b = json.load(open(os.path.join(new, "noise.json")))["noise"]
    rows = []
    for key in a:
        same = key in b and a[key] == b[key]
        value = "identical (sha256, shape, dtype)"
        if not same:
            x, y = np.load(os.path.join(old, "noise", key + ".npy")), np.load(os.path.join(new, "noise", key + ".npy"))
            value = (f"differs: {a[key]['dtype']} {a[key]['shape']} against {b.get(key, {}).get('dtype')} "
                     f"{b.get(key, {}).get('shape')}")
            if x.shape == y.shape:
                value += f", {float(np.max(np.abs(x - y))):.3g} at most"
        rows.append((f"seeded noise, {key}", "exactly equal", value, same))
    return rows


# ---------------------------------------------------------------------------------------------------------- cutout

def cutout_pictures():
    data = os.environ["REGRESSION_DATA"]
    cases = sorted(p for p in glob.glob(os.path.join(data, "*", "picture.png")))
    return cases[:: max(1, len(cases) // 8)][:8]


def dump_cutout(out):
    from PIL import Image

    from model_maker import cutout

    cutter = cutout.Cutter(os.path.join(models(), "birefnet"), device="cpu")
    os.makedirs(os.path.join(out, "cutout"), exist_ok=True)
    found = {}
    for path in cutout_pictures():
        name = os.path.basename(os.path.dirname(path))
        mask = cutter.mask(Image.open(path))
        np.save(os.path.join(out, "cutout", name + ".npy"), mask)
        found[name] = digest(mask)
    json.dump({"versions": versions(), "masks": found}, open(os.path.join(out, "cutout.json"), "w"), indent=1)
    print(f"cutout: {len(found)} masks")


def compare_cutout(old, new):
    a = json.load(open(os.path.join(old, "cutout.json")))["masks"]
    b = json.load(open(os.path.join(new, "cutout.json")))["masks"]
    differ = []
    for name in a:
        if a[name] != b.get(name):
            x = np.load(os.path.join(old, "cutout", name + ".npy"))
            y = np.load(os.path.join(new, "cutout", name + ".npy"))
            differ.append(f"{name}: {int((x != y).sum())} pixels")
    return [("BiRefNet's mask on the CPU", "exactly equal",
             f"{len(a) - len(differ)} of {len(a)} masks identical" + (f" ({'; '.join(differ)})" if differ else ""),
             not differ and set(a) == set(b))]


# ----------------------------------------------------------------------------------------------------------- shape

def shape_case():
    return os.path.join(os.environ["REGRESSION_DATA"], os.environ.get("SHAPE_CASE", "barrel-seams-and-folds"))


def dump_shape(out):
    """Two denoising steps (sampled_latents with 2 steps: sigma 1 to 0.5 to 0) at the pipeline's guidance, float32 on
    the CPU: once from this environment's own DINOv2 features, once from the features stored by `--features`."""
    import torch
    from PIL import Image

    from model_maker import shape_net

    torch.set_grad_enabled(False)
    shape_net.WEIGHTS = os.path.join(models(), "triposg")
    case = shape_case()
    rgb = np.asarray(Image.open(os.path.join(case, "picture.png")).convert("RGB"))
    mask = np.asarray(Image.open(os.path.join(case, "mask.png")).convert("L")) > 127
    processor, encoder = shape_net.load_encoder("cpu", torch.float32)
    features = shape_net.picture_features(processor, encoder, shape_net.prepared(rgb, mask), "cpu", torch.float32)
    del encoder
    denoiser = shape_net.load_denoiser("cpu", torch.float32, **shape_net.CHOICES)
    own = shape_net.sampled_latents(denoiser, features, SEED, 2, 7.0, "cpu", torch.float32)
    np.save(os.path.join(out, "shape-features.npy"), features.numpy())
    np.save(os.path.join(out, "shape-own.npy"), own.numpy())
    common = os.environ.get("SHAPE_FEATURES")
    if common:
        given = torch.from_numpy(np.load(common))
        np.save(os.path.join(out, "shape-common.npy"),
                shape_net.sampled_latents(denoiser, given, SEED, 2, 7.0, "cpu", torch.float32).numpy())
    json.dump({"versions": versions(), "case": os.path.basename(case), "common_features": common},
              open(os.path.join(out, "shape.json"), "w"))
    print(f"shape: two steps of {os.path.basename(case)}" + (" (and from the given features)" if common else ""))


def compare_shape(old, new):
    rows = []
    for key, label in (("own", "each environment's own features"), ("common", "the same features (the old ones)")):
        x, y = os.path.join(old, f"shape-{key}.npy"), os.path.join(new, f"shape-{key}.npy")
        if not (os.path.exists(x) and os.path.exists(y)):
            continue
        x, y = np.load(x), np.load(y)
        gap = float(np.max(np.abs(x - y)))
        rows.append((f"two shape steps on the CPU, from {label}", f"within {SHAPE_STEPS:g}",
                     f"{gap:.3g} at most, {float(np.mean(np.abs(x - y))):.2g} on average (latents up to "
                     f"{float(np.max(np.abs(x))):.2f})", gap <= SHAPE_STEPS))
    return rows


# ------------------------------------------------------------------------------------------------------------ view

def dump_view(out):
    """One step of the view painter at 128 pixels on the CPU, in float16 as on the card, from the stored case's shape
    and picture: the six views as the pipeline returns them."""
    import torch
    from PIL import Image

    from model_maker import paint, words

    side = 128
    os.environ["MV_ADAPTER_SRC"] = os.path.join(os.environ["MODEL_MAKER_SOURCES"], "mv-adapter")
    paint.SIDE = side  # surface_maps, reference and the cameras' pixels
    torch.set_grad_enabled(False)
    case = shape_case()
    job = json.load(open(os.path.join(case, "job.json")))
    shape = np.load(os.path.join(case, "shape.npz"))
    rgb = np.asarray(Image.open(os.path.join(case, "picture.png")).convert("RGB"))
    mask = np.asarray(Image.open(os.path.join(case, "mask.png")).convert("L")) > 127
    text = words.paint_prompt(words.subject_of(job["prompt"]))

    class OneStepOnCpu:
        def __init__(self):
            from diffusers import AutoencoderKL

            paint._mv_adapter_on_path()
            from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
            from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
            from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

            # Each part straight in float16 (the card's pipeline loads the VAE in float32 and turns it to float16 at
            # once: the same weights), so the computer's memory holds one copy.
            vae = AutoencoderKL.from_pretrained(os.path.join(models(), "sdxl-vae-fp16-fix"), torch_dtype=torch.float16)
            pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(os.path.join(models(), "sdxl-base-1.0"), vae=vae,
                                                             variant="fp16", torch_dtype=torch.float16)
            pipe.scheduler = ShiftSNRScheduler.from_scheduler(pipe.scheduler, shift_mode="interpolated",
                                                              shift_scale=8.0)
            # Without copying the UNet's attention weights into the new layers first: that copy clones the whole
            # UNet (over 12 GB of the computer's memory at once), and the adapter's file holds every one of the 700
            # weights it would fill (and the 38 of the condition encoder), so the loaded model is the same.
            pipe.init_custom_adapter(num_views=6, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0,
                                     copy_attn_weights=False)
            pipe.load_custom_adapter(os.path.join(models(), "mv-adapter"),
                                     weight_name="mvadapter_ig2mv_sdxl.safetensors")
            pipe.to(device="cpu", dtype=torch.float16)
            pipe.cond_encoder.to(device="cpu", dtype=torch.float16)
            pipe.set_progress_bar_config(disable=True)
            self.pipe = pipe
            print("view: loaded", flush=True)

        def views(self, control, words_, reference_image, seed, steps):
            control = torch.from_numpy(control.clip(0, 1)).permute(0, 3, 1, 2).to("cpu", torch.float16)
            return self.pipe(prompt=words_, negative_prompt=paint.NEGATIVE, height=side, width=side,
                             num_inference_steps=steps, guidance_scale=paint.GUIDANCE, num_images_per_prompt=6,
                             control_image=control, control_conditioning_scale=1.0, reference_image=reference_image,
                             reference_conditioning_scale=1.0, cross_attention_kwargs={"scale": 1.0},
                             generator=torch.Generator(device="cpu").manual_seed(seed)).images

    painter = OneStepOnCpu()
    captured = []

    def keep(control, words_, reference_image, seed, steps):
        made = painter.views(control, words_, reference_image, seed, steps)
        captured.extend(made)
        return made

    holder = type("Holder", (), {"views": staticmethod(keep)})()
    paint.painted(holder, shape["corners"], shape["triangles"], rgb, mask, text, seed=0, steps=1)
    stacked = np.stack([np.asarray(v.convert("RGB")) for v in captured])
    np.save(os.path.join(out, "view.npy"), stacked)
    json.dump({"versions": versions(), "case": os.path.basename(case), "text": text, "side": side},
              open(os.path.join(out, "view.json"), "w"))
    print(f"view: one step, {stacked.shape}")


def compare_view(old, new):
    x = np.load(os.path.join(old, "view.npy")).astype(np.float64)
    y = np.load(os.path.join(new, "view.npy")).astype(np.float64)
    mean = float(np.mean(np.abs(x - y)))
    return [("one view step at 128 pixels on the CPU (six views)", f"within {VIEW_STEP_OF_255} of 255 on average",
             f"{mean:.3f} of 255 on average, {float(np.max(np.abs(x - y))):.0f} at most, "
             f"{float(np.mean(np.abs(x - y) > 8)) * 100:.2f} % of values over 8", mean <= VIEW_STEP_OF_255)]


# ----------------------------------------------------------------------------------------------------------- judge

# What the judge's record measures of the run rather than answers: its time, a restart, the server's memory.
MEASURED = ("seconds", "restarted", "server_memory_mb")


def answered(case):
    return {**case, "record": {k: v for k, v in case["record"].items() if k not in MEASURED}}


def dump_judge(out):
    """The question judge's test cases (python/tests/question-judge-cases), asked of the real judge (llama.cpp on the
    CPU): its share of yes, its verdict and its words for each."""
    from PIL import Image

    from model_maker import pipeline
    from model_maker import question_judge as Q

    folder = os.path.join(REPOSITORY, "python", "tests", "question-judge-cases")
    with open(os.path.join(folder, "cases.toml"), "rb") as file:
        cases = tomllib.load(file)["case"]
    where = Q.found(pipeline.MODEL_MAKER_ROOT, models())
    if not where.usable:
        raise SystemExit(f"the question judge is not usable here: {where.why}")
    found = {}
    with Q.Judge(where) as judge:
        for case in cases:
            with Image.open(os.path.join(folder, case["file"])) as strip:
                check, record = Q.object_check(judge, strip.convert("RGB"), case["prompt"], required=True)
            record = {k: v for k, v in record.items() if k not in MEASURED}
            found[case["file"]] = {"passed": check.passed, "words": check.words, "labelled": case["passes"],
                                   "record": record}
    json.dump({"versions": versions(), "cases": found}, open(os.path.join(out, "judge.json"), "w"), indent=1)
    print(f"judge: {len(found)} cases, {sum(v['passed'] == v['labelled'] for v in found.values())} as labelled")


def compare_judge(old, new):
    a = json.load(open(os.path.join(old, "judge.json")))["cases"]
    b = json.load(open(os.path.join(new, "judge.json")))["cases"]
    differ = [name for name in a if name not in b or answered(a[name]) != answered(b[name])]
    labelled = sum(v["passed"] == v["labelled"] for v in b.values())
    return [("the question judge's cases (shares, verdicts, words, the record but its time and memory)", "exactly equal",
             f"{len(a) - len(differ)} of {len(a)} identical; {labelled} of {len(b)} as labelled"
             + (f"; differ: {', '.join(differ)}" if differ else ""), not differ and set(a) == set(b))]


# ------------------------------------------------------------------------------------------------------------ glbs

def glb_parts(path):
    """A finished .glb's content: its document, every accessor's values and the texture's pixels (decoded). The
    texture's PNG carries signed content credentials, whose signature differs each time it is made, so its bytes are
    compared as pixels."""
    import io

    from PIL import Image

    from model_maker import glb

    with open(path, "rb") as file:
        data = file.read()
    document, binary = glb.read_document(data)
    accessors = [np.asarray(glb.accessor_values(document, binary, k)) for k in range(len(document["accessors"]))]
    textures = []
    for image in document.get("images", []):
        view = document["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        textures.append(np.asarray(Image.open(io.BytesIO(binary[start:start + view["byteLength"]])).convert("RGBA")))
    plain = json.loads(json.dumps(document))
    for view in plain.get("bufferViews", []):
        view.pop("byteOffset", None)
    for image in plain.get("images", []):
        plain["bufferViews"][image["bufferView"]].pop("byteLength", None)
    plain.get("buffers", [{}])[0].pop("byteLength", None)
    return data, plain, accessors, textures


def compare_glbs(old, new):
    """`old` and `new` hold finished job folders by the same names (GLB_JOBS: their parent folders' names)."""
    rows = []
    a_root, b_root = os.path.join(old, "jobs"), os.path.join(new, "jobs")
    names = sorted(n for n in os.listdir(a_root) if os.path.isdir(os.path.join(a_root, n)))
    same_bytes, same_content, notes = 0, 0, []
    for name in names:
        slug = json.load(open(os.path.join(a_root, name, "job.json")))["slug"]
        a = glb_parts(os.path.join(a_root, name, slug + ".glb"))
        b = glb_parts(os.path.join(b_root, name, slug + ".glb"))
        same_bytes += a[0] == b[0]
        content = (a[1] == b[1] and len(a[2]) == len(b[2]) and all(x.shape == y.shape and np.array_equal(x, y)
                                                                   for x, y in zip(a[2], b[2]))
                   and len(a[3]) == len(b[3]) and all(np.array_equal(x, y) for x, y in zip(a[3], b[3])))
        same_content += content
        if not content:
            notes.append(name)
    rows.append(("the finish stage's .glb of stored jobs (document, geometry, uvs, texture pixels)", "exactly equal",
                 f"{same_content} of {len(names)} identical ({same_bytes} of {len(names)} byte for byte; the texture's "
                 f"signed credentials differ each signing)" + (f"; differ: {', '.join(notes)}" if notes else ""),
                 same_content == len(names) and len(names) > 0))
    return rows


PARTS = {"tokens": (dump_tokens, compare_tokens), "inputs": (dump_inputs, compare_inputs),
         "noise": (dump_noise, compare_noise), "cutout": (dump_cutout, compare_cutout),
         "shape": (dump_shape, compare_shape), "view": (dump_view, compare_view), "glbs": (None, compare_glbs),
         "judge": (dump_judge, compare_judge), "siglip": (dump_siglip, compare_siglip)}


def main(argv):
    if argv[:1] == ["dump"]:
        out = argv[1]
        os.makedirs(out, exist_ok=True)
        import torch

        torch.manual_seed(0)
        for part in argv[2:]:
            started = time.time()
            PARTS[part][0](out)
            print(f"  {part}: {time.time() - started:.0f} s", flush=True)
        return 0
    if argv[:1] == ["compare"]:
        old, new = argv[1], argv[2]
        if "--exact" in argv[3:]:
            # The ports' proof on one environment: every bar is byte for byte.
            global SIGLIP_FEATURES, DINO_FEATURES, SHAPE_STEPS, VIEW_STEP_OF_255
            SIGLIP_FEATURES = DINO_FEATURES = SHAPE_STEPS = VIEW_STEP_OF_255 = EXACT
        rows = []
        for part, (_, compare) in PARTS.items():
            marker = {"tokens": "tokens.json", "inputs": "inputs.json", "noise": "noise.json",
                      "cutout": "cutout.json", "shape": "shape.json", "view": "view.json", "glbs": "jobs",
                      "judge": "judge.json", "siglip": "siglip.json"}[part]
            if part == "siglip":
                # Either side may be a full `inputs` record; compared when one side holds SigLIP's record alone.
                has = [os.path.exists(os.path.join(f, "siglip.json")) or os.path.exists(os.path.join(f, "inputs.json"))
                       for f in (old, new)]
                if all(has) and any(os.path.exists(os.path.join(f, "siglip.json")) for f in (old, new)):
                    rows += compare(old, new)
            elif os.path.exists(os.path.join(old, marker)) and os.path.exists(os.path.join(new, marker)):
                rows += compare(old, new)
        for test, bar, value, passed in rows:
            print(f"{'NOTE' if passed is None else 'PASS' if passed else 'FAIL'} | {test} | {bar} | {value}")
        json.dump([dict(test=t, bar=b, value=v, passed=p) for t, b, v, p in rows],
                  open(os.path.join(new, "compare.json"), "w"), indent=1)
        return 0 if rows and all(r[3] is not False for r in rows) else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
