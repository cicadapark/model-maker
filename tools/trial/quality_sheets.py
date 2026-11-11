#!/usr/bin/env python3
"""Blind sheets of whole cut-outs for labelling a quality by eye (curved or straight, round or square, tall or squat):
the pictures of the given prompts, each cropped square round its cut-out, shuffled, and shown under a random code
only, twenty to a sheet. The key (code to picture and prompt) is written apart, to be opened only once every label is
written (docs/objects.md, "Qualities the words name"; as head_sheets.py for the war hammers).

    quality_sheets.py OUT --prompts "a curved bronze dagger|a hunting knife|..." TRIES_FOLDER...

TRIES_FOLDER holds a tries.json of paint_tries.py. Writes OUT/sheet-NN.png and OUT/key.json.
"""

import argparse
import glob
import json
import os
import random
import string

SIDE = 260
PER_ROW = 5
PER_SHEET = 20
MARGIN = 0.1


def whole_crop(picture, mask):
    """The picture cropped to a square round its cut-out, `MARGIN` of its size added each side."""
    import numpy as np

    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    side = int(max(rows[-1] - rows[0], cols[-1] - cols[0]) * (1 + 2 * MARGIN)) + 1
    middle_row, middle_col = (rows[0] + rows[-1]) // 2, (cols[0] + cols[-1]) // 2
    left, top = middle_col - side // 2, middle_row - side // 2
    return picture.crop((left, top, left + side, top + side)).resize((SIDE, SIDE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    wanted = set(args.prompts.split("|"))
    entries = []
    for folder in args.folders:
        for record_path in sorted(glob.glob(os.path.join(folder, "**", "tries.json"), recursive=True)):
            with open(record_path) as file:
                for entry in json.load(file):
                    if entry["prompt"] in wanted:
                        picture = os.path.join(os.path.dirname(record_path), entry["picture"])
                        entries.append({"prompt": entry["prompt"], "picture": picture,
                                        "mask": picture[:-4] + "-mask.png", "picture_words": entry["picture_words"],
                                        "seed": entry["seed"]})
    chance = random.Random(args.seed)
    chance.shuffle(entries)
    codes = set()
    while len(codes) < len(entries):
        codes.add("".join(chance.choice(string.ascii_uppercase) for _ in range(3)))
    codes = sorted(codes, key=lambda _: chance.random())
    os.makedirs(args.out, exist_ok=True)
    try:
        font = ImageFont.load_default(size=26)
    except TypeError:
        font = ImageFont.load_default()
    key = {}
    for start in range(0, len(entries), PER_SHEET):
        batch = entries[start:start + PER_SHEET]
        rows = (len(batch) + PER_ROW - 1) // PER_ROW
        sheet = Image.new("RGB", (PER_ROW * SIDE, rows * (SIDE + 32)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, entry in enumerate(batch):
            code = codes[start + k]
            key[code] = entry
            picture = Image.open(entry["picture"]).convert("RGB")
            mask = np.asarray(Image.open(entry["mask"]).convert("L")) > 127
            x, y = (k % PER_ROW) * SIDE, (k // PER_ROW) * (SIDE + 32)
            sheet.paste(whole_crop(picture, mask), (x, y))
            draw.text((x + 8, y + SIDE + 2), code, fill="black", font=font)
        sheet.save(os.path.join(args.out, f"sheet-{start // PER_SHEET + 1:02d}.png"))
    with open(os.path.join(args.out, "key.json"), "w") as file:
        json.dump(key, file, indent=1)
    print(f"{len(entries)} pictures on {(len(entries) + PER_SHEET - 1) // PER_SHEET} sheets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
