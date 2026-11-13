#!/usr/bin/env python3
"""Blind sheets for labelling by eye (docs/trials/2026-11-11-030-diagnosis.md).

    blind_sheets.py ROOT OUT [--by-prompt]     ROOT holds paint_tries.py runs, one folder a wording, each with its
                                               tries.json; --by-prompt: one sheet a prompt (its name shown)

Each try's whole cut-out is cropped square on white, shuffled within its group (maces, clubs, spinning wheels, water
wheels) and shown under a random code only: no prompt, wording or score. The key is written apart (OUT/key.json), to
be opened only once every label is written."""
import glob, json, os, random, string, sys
import numpy as np
from PIL import Image, ImageDraw

by_prompt = "--by-prompt" in sys.argv
root, out = [a for a in sys.argv[1:] if a != "--by-prompt"][:2]
os.makedirs(out, exist_ok=True)
GROUPS = {"mace": ("mace",), "club": ("club",), "spinning": ("spinning wheel",), "water": ("water wheel",)}
entries = []
for tries in glob.glob(f"{root}/*/tries.json"):
    wording = os.path.basename(os.path.dirname(tries))
    for e in json.load(open(tries)):
        path = os.path.join(os.path.dirname(tries), e["picture"])
        group = e["prompt"].replace(" ", "-") if by_prompt else \
            next(g for g, words in GROUPS.items() if any(w in e["prompt"] for w in words))
        entries.append({"group": group, "prompt": e["prompt"], "wording": wording, "seed": e["seed"], "picture": path})
# The sealed runs' own tries of the four sealed prompts (the same seeds as "old"), not again.
rng = random.Random(20261111)
codes = set()
key = []
for group in (sorted({e["group"] for e in entries}) if by_prompt else GROUPS):
    items = [e for e in entries if e["group"] == group]
    rng.shuffle(items)
    tiles = []
    for e in items:
        code = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
        while code in codes:
            code = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
        codes.add(code)
        e["code"] = code
        key.append(e)
        pic = Image.open(e["picture"]).convert("RGB")
        mask = np.asarray(Image.open(e["picture"][:-4] + "-mask.png").convert("L")) > 127
        rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        side = int(max(rows[-1] - rows[0], cols[-1] - cols[0]) * 1.1) + 2
        cr, cc = (rows[0] + rows[-1]) // 2, (cols[0] + cols[-1]) // 2
        white = Image.new("RGB", pic.size, (255, 255, 255))
        cut = Image.composite(pic, white, Image.fromarray((mask * 255).astype(np.uint8)))
        tile = cut.crop((cc - side // 2, cr - side // 2, cc - side // 2 + side, cr - side // 2 + side)).resize((220, 220))
        canvas = Image.new("RGB", (220, 244), (255, 255, 255))
        canvas.paste(tile, (0, 24))
        try:
            from PIL import ImageFont
            font = ImageFont.load_default(size=22)
        except TypeError:
            font = None
        ImageDraw.Draw(canvas).text((6, 0), code, fill=(0, 0, 0), font=font)
        tiles.append(canvas)
    per_row = 8
    rows_n = (len(tiles) + per_row - 1) // per_row
    sheet = Image.new("RGB", (per_row * 222, rows_n * 246), (200, 200, 200))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % per_row) * 222, (k // per_row) * 246))
    sheet.save(os.path.join(out, f"sheet-{group}.png"))
json.dump(key, open(os.path.join(out, "key.json"), "w"), indent=1)
print(len(key), "tiles")
