#!/usr/bin/env python3
"""The results picture of a sealed run: each object as made (the report's first side), its prompt and its verdict.

    sealed_grid.py BATCH [BATCH ...] VERDICTS.json OUT.webp [--columns N]

Each BATCH is a `model-maker batch` folder (its job folders, each with job.json and report/; a run made in halves
gives two). VERDICTS maps each prompt to {"checks": "pass" | "fail" | "not made", "eye": "good" | "bad"}: the checks' verdict is the job's own, the eye's
was written blind. Objects are laid out in the prompts' order, the good ones that pass first.
"""

import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

TILE = 260
CAPTION = 80
GAP = 40
BACKGROUND = (246, 246, 243)
TILE_BACKGROUND = (238, 238, 234)
# Colours of the verdicts as 0.2.0's picture had them: green fine, red wrong, blue a false refusal, grey not made.
KINDS = {
    ("pass", "good"): ("fine", (46, 125, 50)),
    ("pass", "bad"): ("passes, but bad", (198, 40, 40)),
    ("fail", "good"): ("fails a check, looks right", (61, 90, 154)),
    ("fail", "bad"): ("fails, bad", (191, 110, 20)),
    ("not made", "bad"): ("not made", (120, 120, 120)),
}
ORDER = list(KINDS)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def first_side(job):
    """The first of the report's turntable sides, its square cut from the strip."""
    strip = Image.open(os.path.join(job, "report", "turntable.png")).convert("RGB")
    side = strip.width // 8
    return strip.crop((0, 0, side, strip.height))


def tile(job, prompt, kind, font, small):
    label, colour = KINDS[kind]
    out = Image.new("RGB", (TILE, TILE + CAPTION), BACKGROUND)
    picture = Image.new("RGB", (TILE, TILE), TILE_BACKGROUND)
    if job:
        made = first_side(job)
        made.thumbnail((TILE, TILE))
        picture.paste(made, ((TILE - made.width) // 2, (TILE - made.height) // 2))
    out.paste(picture, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.text((0, TILE + 10), prompt, fill=(40, 40, 40), font=font)
    draw.rectangle((0, TILE + 40, 11, TILE + 51), fill=colour)
    draw.text((18, TILE + 36), label, fill=colour, font=small)
    return out


def main(argv):
    columns = 5
    if "--columns" in argv:
        at = argv.index("--columns")
        columns = int(argv[at + 1])
        argv = argv[:at] + argv[at + 2:]
    *batches, verdicts_path, out = argv
    with open(verdicts_path) as file:
        verdicts = json.load(file)
    jobs = {}
    for batch, name in sorted((batch, name) for batch in batches for name in os.listdir(batch)):
        job = os.path.join(batch, name)
        if os.path.isfile(os.path.join(job, "job.json")):
            with open(os.path.join(job, "job.json")) as file:
                made = any(entry.endswith(".glb") for entry in os.listdir(job))
                jobs[json.load(file)["prompt"]] = job if made else None
    font = ImageFont.truetype(FONT, 17)
    small = ImageFont.truetype(FONT, 15)
    entries = sorted(verdicts.items(), key=lambda item: ORDER.index((item[1]["checks"], item[1]["eye"])))
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * TILE + (columns + 1) * GAP // 2, rows * (TILE + CAPTION + GAP // 2) + GAP // 2),
                      BACKGROUND)
    for at, (prompt, verdict) in enumerate(entries):
        kind = (verdict["checks"], verdict["eye"])
        x = GAP // 2 + (at % columns) * (TILE + GAP // 2)
        y = GAP // 2 + (at // columns) * (TILE + CAPTION + GAP // 2)
        sheet.paste(tile(jobs.get(prompt), prompt, kind, font, small), (x, y))
    sheet.save(out, quality=82, method=6)
    print(f"{len(entries)} objects in {out}, {sheet.width} x {sheet.height}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
