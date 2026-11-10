#!/usr/bin/env python3
"""Blind sheets of head crops for labelling by eye: each picture's head (the top of its cut-out, where a hafted
thing painted upright has it) cropped square, shuffled, and shown under a random code only, sixteen to a sheet. The key
(code to picture) is written apart, to be opened only once every label is written (docs/objects.md, "War hammers").

    head_sheets.py JUDGED.json OUT [--seed N]     JUDGED.json from judge_tries.py; writes OUT/sheet-NN.png, OUT/key.json
"""

import argparse
import json
import os
import random
import string

HEAD_SHARE = 0.35  # the top of the cut-out taken as the head, as a share of its height
SIDE = 300
PER_ROW = 4
PER_SHEET = 16


def head_crop(picture, mask):
    import numpy as np

    rows = np.flatnonzero(mask.any(1))
    top, height = rows[0], rows[-1] - rows[0] + 1
    band = mask[top:top + max(int(height * HEAD_SHARE), 8)]
    cols = np.flatnonzero(band.any(0))
    left, right, bottom = cols[0], cols[-1] + 1, top + band.shape[0]
    side = int(max(right - left, bottom - top) * 1.15)
    middle_col, middle_row = (left + right) // 2, (top + bottom) // 2
    box = (middle_col - side // 2, middle_row - side // 2, middle_col - side // 2 + side, middle_row - side // 2 + side)
    return picture.crop(box).resize((SIDE, SIDE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("judged")
    parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    with open(args.judged) as file:
        entries = json.load(file)
    chance = random.Random(args.seed)
    chance.shuffle(entries)
    codes = set()
    while len(codes) < len(entries):
        codes.add("".join(chance.choice(string.ascii_uppercase) for _ in range(3)))
    codes = sorted(codes, key=lambda _: chance.random())
    os.makedirs(args.out, exist_ok=True)
    key = {}
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:
        font = ImageFont.load_default()
    for start in range(0, len(entries), PER_SHEET):
        batch = entries[start:start + PER_SHEET]
        rows = (len(batch) + PER_ROW - 1) // PER_ROW
        sheet = Image.new("RGB", (PER_ROW * SIDE, rows * (SIDE + 36)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, entry in enumerate(batch):
            code = codes[start + k]
            key[code] = {"sha256": entry["sha256"], "picture": entry["picture"], "prompt": entry["prompt"]}
            picture = Image.open(entry["picture"]).convert("RGB")
            mask = np.asarray(Image.open(entry["mask"]).convert("L")) > 127
            x, y = (k % PER_ROW) * SIDE, (k // PER_ROW) * (SIDE + 36)
            sheet.paste(head_crop(picture, mask), (x, y))
            draw.text((x + 8, y + SIDE + 2), code, fill="black", font=font)
        sheet.save(os.path.join(args.out, f"sheet-{start // PER_SHEET + 1:02d}.png"))
    with open(os.path.join(args.out, "key.json"), "w") as file:
        json.dump(key, file, indent=1)
    print(f"{len(entries)} heads on {(len(entries) + PER_SHEET - 1) // PER_SHEET} sheets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
