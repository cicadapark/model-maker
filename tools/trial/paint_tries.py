#!/usr/bin/env python3
"""Pictures only: each prompt's tries painted, cut out and checked as the pipeline's picture stage does them (the same
picture words, the same consecutive seeds from `painter.seed_of`), without choosing one and without shapes. For
measuring a change of the picture words on the card (docs/objects.md, "War hammers").

    paint_tries.py OUT --tries N [--words FILE] [--view TEXT] PROMPT...

`--words FILE` reads the picture words from another copy of words.py (`git show REV:python/model_maker/words.py`), so
an older wording is painted exactly as it was; `--view TEXT` puts TEXT in place of the view the words choose. Writes
OUT/<slug>/<seed>.png, <seed>-mask.png and OUT/tries.json (each try's prompt, picture words, seed and checks). Needs
an NVIDIA card; models from MODEL_MAKER_MODELS, each checked against the registry first.
"""

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))


def words_module(path):
    if not path:
        from model_maker import words

        return words
    spec = importlib.util.spec_from_file_location("words_compared", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def picture_words(words, prompt, view):
    said = words.picture_prompt(words.subject_of(prompt))
    if view:
        for name in ("HAFTED_VIEW", "WEAPON_VIEW", "OBJECT_VIEW", "TOOL_VIEW"):
            chosen = getattr(words, name, None)
            if chosen and f", {chosen}, " in said:
                return said.replace(f", {chosen}, ", f", {view}, ")
        raise SystemExit(f"no view found to replace in {said!r}")
    return said


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("prompts", nargs="+")
    parser.add_argument("--tries", type=int, default=8)
    parser.add_argument("--words")
    parser.add_argument("--view")
    args = parser.parse_args()

    import numpy as np
    from PIL import Image

    from model_maker import painter, pipeline

    words = words_module(args.words)
    settings = pipeline.Settings()
    paint = painter.Painter(settings.verified("flux1-schnell", "flux1-schnell")[0])
    from model_maker import cutout

    cutter = cutout.Cutter(settings.verified("birefnet", "birefnet")[0])
    record_path = os.path.join(args.out, "tries.json")
    records = json.load(open(record_path)) if os.path.exists(record_path) else []
    for prompt in args.prompts:
        said = picture_words(words, prompt, args.view)
        first = painter.seed_of(said)
        folder = os.path.join(args.out, words.subject_of(prompt).slug)
        os.makedirs(folder, exist_ok=True)
        for seed, picture, mask, result in painter.tries(paint, cutter, said, range(first, first + args.tries)):
            picture.save(os.path.join(folder, f"{seed}.png"))
            Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(folder, f"{seed}-mask.png"))
            records.append({"prompt": prompt, "picture_words": said, "seed": seed,
                            "picture": os.path.relpath(os.path.join(folder, f"{seed}.png"), args.out),
                            **result.as_dict()})
            with open(record_path, "w") as file:
                json.dump(records, file, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
