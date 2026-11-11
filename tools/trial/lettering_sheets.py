#!/usr/bin/env python3
"""Blind sheets for labelling painted lettering by eye: each picture cropped round its cut-out (its own proportions,
large enough to read small letters), shuffled, and shown under a random code only, twelve to a sheet. The key (code
to picture) is written apart, to be opened only once every label is written (as quality_sheets.py; docs/objects.md,
"Painted lettering").

    lettering_sheets.py OUT LIST [--seed N]

LIST holds one picture path a line (its cut-out beside it: <name>-mask.png, or a job's mask.png beside its
picture.png). Writes OUT/sheet-NN.png and OUT/key.json.
"""

import argparse
import json
import os
import random
import string

CELL = 440
PER_ROW = 3
PER_SHEET = 12
MARGIN = 0.06
LABEL_HEIGHT = 34


def mask_of(path):
    """A try's cut-out (<name>-mask.png), or a job's (mask.png beside its picture.png)."""
    beside = path[:-4] + "-mask.png"
    return beside if os.path.exists(beside) else os.path.join(os.path.dirname(path), "mask.png")


def region(picture, mask):
    """The picture cropped round its cut-out, `MARGIN` of its larger side added each side, fitted into a cell."""
    import numpy as np

    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    pad = int(max(rows[-1] - rows[0], cols[-1] - cols[0]) * MARGIN)
    crop = picture.crop((max(cols[0] - pad, 0), max(rows[0] - pad, 0), min(cols[-1] + pad + 1, picture.width),
                         min(rows[-1] + pad + 1, picture.height)))
    crop.thumbnail((CELL, CELL))
    return crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("list")
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    with open(args.list) as file:
        pictures = [line.strip() for line in file if line.strip()]
    chance = random.Random(args.seed)
    chance.shuffle(pictures)
    codes = set()
    while len(codes) < len(pictures):
        codes.add("".join(chance.choice(string.ascii_uppercase) for _ in range(3)))
    codes = sorted(codes, key=lambda _: chance.random())
    os.makedirs(args.out, exist_ok=True)
    try:
        font = ImageFont.load_default(size=26)
    except TypeError:
        font = ImageFont.load_default()
    key = {}
    for start in range(0, len(pictures), PER_SHEET):
        batch = pictures[start:start + PER_SHEET]
        rows = (len(batch) + PER_ROW - 1) // PER_ROW
        sheet = Image.new("RGB", (PER_ROW * CELL, rows * (CELL + LABEL_HEIGHT)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, path in enumerate(batch):
            code = codes[start + k]
            key[code] = path
            picture = Image.open(path).convert("RGB")
            mask_image = Image.open(mask_of(path)).convert("L")
            if mask_image.size != picture.size:
                mask_image = mask_image.resize(picture.size, Image.NEAREST)
            crop = region(picture, np.asarray(mask_image) > 127)
            x, y = (k % PER_ROW) * CELL, (k // PER_ROW) * (CELL + LABEL_HEIGHT)
            sheet.paste(crop, (x + (CELL - crop.width) // 2, y + (CELL - crop.height) // 2))
            draw.text((x + 8, y + CELL + 2), code, fill="black", font=font)
        sheet.save(os.path.join(args.out, f"sheet-{start // PER_SHEET + 1:02d}.png"))
    with open(os.path.join(args.out, "key.json"), "w") as file:
        json.dump(key, file, indent=1)
    print(f"{len(pictures)} pictures on {(len(pictures) + PER_SHEET - 1) // PER_SHEET} sheets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
