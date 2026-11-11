#!/usr/bin/env python3
"""The lettering check measured on picture tries (docs/objects.md, "Painted lettering").

    judge_lettering.py read LIST OUT.jsonl [--workers N]   every word tesseract reads in each picture's cut-out
    judge_lettering.py sum OUT.jsonl LABELS.json            what the check (lettering.py's limits) refuses and
                                                             passes, against labels by eye

LIST holds one picture path a line (its cut-out beside it: <name>-mask.png, or a job's mask.png beside its
picture.png). The reading is kept raw (every word,
whatever its confidence), so the limits can be weighed without reading again. LABELS maps a picture's path to
"letters", "none" or "unclear" (and, with "asked", to whether its prompt asks for lettering: {"label": ..,
"asked": ..}). `read` refuses to start without a usable tesseract (version 5 or newer, its English data), and each
record keeps the version that read it.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
sys.path.insert(0, HERE)


def read_one(path):
    import numpy as np
    from PIL import Image

    from model_maker import lettering

    picture = Image.open(path).convert("RGB")
    from lettering_sheets import mask_of

    mask = Image.open(mask_of(path)).convert("L")
    if mask.size != picture.size:
        mask = mask.resize(picture.size, Image.NEAREST)
    mask = np.asarray(mask) > 127
    words = lettering.every_reading(np.asarray(picture), mask) if mask.any() else []
    return {"picture": path, "side": max(mask.shape), "words": [w.as_dict() for w in words],
            "tesseract": lettering.reader().version}


def read(listing, out, workers):
    from model_maker import lettering

    found = lettering.reader()
    if not found.usable:
        raise SystemExit(f"judge_lettering.py: {found.why}")
    with open(listing) as file:
        pictures = [line.strip() for line in file if line.strip()]
    with ProcessPoolExecutor(workers) as pool, open(out, "w") as file:
        for record in pool.map(read_one, pictures, chunksize=4):
            file.write(json.dumps(record) + "\n")
    print(f"{len(pictures)} pictures read into {out}")


def words_of(record):
    """The words of a picture's readings that the check makes (lettering.lettering_of keeps each counted word once)."""
    from model_maker import lettering

    ways = {lettering.reading_name(*reading) for reading in lettering.READINGS}
    return [lettering.Word(w["text"], w["confidence"], *w["box"], w["way"]) for w in record["words"]
            if w["way"] in ways]


def verdict(record, asked=False):
    """Whether the check refuses the picture (its words not asking for lettering), and what it counted."""
    from model_maker import lettering

    counted, characters = lettering.lettering_of(words_of(record), record["side"])
    return characters > lettering.MOST_CHARACTERS and not asked, counted


def summed(readings, labels_path):
    with open(labels_path) as file:
        labels = json.load(file)
    tally = {}
    with open(readings) as file:
        for line in file:
            record = json.loads(line)
            label = labels.get(record["picture"])
            if label is None:
                continue
            if isinstance(label, str):
                label = {"label": label, "asked": False}
            refused, counted = verdict(record, label["asked"])
            key = (label["label"], "asked" if label["asked"] else "not asked", "refused" if refused else "passed")
            tally.setdefault(key, []).append((record["picture"], [w.trimmed for w in counted]))
    for key in sorted(tally):
        print(f"{' / '.join(key)}: {len(tally[key])}")
        if key in (("none", "not asked", "refused"), ("letters", "not asked", "passed")):
            for picture, counted in tally[key]:
                print(f"    {picture} {counted}")


def main():
    parser = argparse.ArgumentParser()
    steps = parser.add_subparsers(dest="step", required=True)
    reading = steps.add_parser("read")
    reading.add_argument("list")
    reading.add_argument("out")
    reading.add_argument("--workers", type=int, default=4)
    summing = steps.add_parser("sum")
    summing.add_argument("readings")
    summing.add_argument("labels")
    args = parser.parse_args()
    if args.step == "read":
        read(args.list, args.out, args.workers)
    else:
        summed(args.readings, args.labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
