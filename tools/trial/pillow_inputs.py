#!/usr/bin/env python3
"""Every model input Pillow makes, hashed, so two Pillow versions can be compared byte for byte (docs/objects.md,
"Pillow 13"): run it with each environment over the same pictures and compare the two outputs with `--compare`.

    python pillow_inputs.py OUT.json PICTURE.png ...          hash each picture's inputs under this environment
    python pillow_inputs.py --compare A.json B.json           how many differ, and where

The inputs are the pipeline's own Pillow steps, as written in python/model_maker: the decode; the cut-out's resize to
its model's side and back (cutout.py); the judges' crop round the cut-out (state_judge.cropped) and SigLIP 2's image
processor, which resizes with Pillow inside transformers (state_judge); the shape model's input (shape_net.prepared);
the paint's reference (paint.reference); the touch-up's view resize (touch_up); and the lettering reader's scaling
(lettering). Each is hashed as its raw bytes or array. The models' folders come from MODEL_MAKER_MODELS.
"""

import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))


def digest(value):
    if isinstance(value, Image.Image):
        value = np.asarray(value)
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes() + str(array.shape).encode() + str(array.dtype).encode()).hexdigest()[:16]


def inputs(path, processor):
    from model_maker import cutout, paint, shape_net, state_judge

    found = {}
    image = Image.open(path)
    rgb = image.convert("RGB")
    found["decode"] = digest(rgb)
    small = rgb.resize((cutout.SIDE, cutout.SIDE), Image.Resampling.BILINEAR)
    found["cutout.small"] = digest(small)
    back = small.convert("L").resize(rgb.size, Image.Resampling.BILINEAR)
    found["cutout.back"] = digest(back)
    mask = np.asarray(rgb.convert("L")) < 240
    if mask.any():
        crop = state_judge.cropped(rgb, mask)
        found["judge.crop"] = digest(crop)
        if processor is not None:
            found["judge.pixels"] = digest(processor(images=crop, return_tensors="pt")["pixel_values"])
    if processor is not None:
        found["judge.whole"] = digest(processor(images=rgb, return_tensors="pt")["pixel_values"])
    array = np.asarray(rgb)
    try:
        found["shape.input"] = digest(shape_net.prepared(array, mask))
    except Exception as error:  # noqa: BLE001 - a picture the shape stage would not take is recorded, not hidden
        found["shape.input"] = f"refused: {type(error).__name__}"
    try:
        made = paint.reference(array, mask)
        found["paint.reference"] = digest(made[0] if isinstance(made, tuple) else made)
    except Exception as error:  # noqa: BLE001
        found["paint.reference"] = f"refused: {type(error).__name__}"
    found["touch_up.resize"] = digest(rgb.resize((768, 768)))
    found["lettering.scale"] = digest(rgb.convert("L").resize((rgb.width * 2, rgb.height * 2), Image.Resampling.LANCZOS))
    return found


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    pictures = sorted(set(a["inputs"]) & set(b["inputs"]))
    differ = {}
    for picture in pictures:
        for key, value in a["inputs"][picture].items():
            if b["inputs"][picture].get(key) != value:
                differ.setdefault(key, []).append(os.path.basename(picture))
    print(f"{a['pillow']} against {b['pillow']}: {len(pictures)} pictures, "
          f"{sum(len(v) for v in a['inputs'].values())} inputs each side")
    if not differ:
        print("every input identical, byte for byte")
        return 0
    for key, names in sorted(differ.items()):
        print(f"DIFFERS {key}: {len(names)} pictures, e.g. {', '.join(names[:3])}")
    return 1


def main(argv):
    if argv[:1] == ["--compare"]:
        return compare(argv[1], argv[2])
    out, pictures = argv[0], argv[1:]
    processor = None
    models = os.environ.get("MODEL_MAKER_MODELS")
    if models:
        from transformers import AutoImageProcessor

        folder = os.path.join(models, "siglip2-so400m-patch14-384")
        if os.path.isdir(folder):
            processor = AutoImageProcessor.from_pretrained(folder)
    import PIL

    result = {"pillow": PIL.__version__, "processor": processor is not None,
              "inputs": {path: inputs(path, processor) for path in pictures}}
    json.dump(result, open(out, "w"), indent=1)
    print(f"{len(pictures)} pictures hashed under Pillow {PIL.__version__} (processor: {processor is not None})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
