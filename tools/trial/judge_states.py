#!/usr/bin/env python3
"""The state judge (python/model_maker/state_judge.py) on pictures already painted, on the CPU or the card.

    python3 tools/trial/judge_states.py MODELS "a wooden treasure chest" FOLDER... [--sheet OUT.png]

Each FOLDER's pictures (its `pictures/` folder's tries, or the PNGs in it) are judged against the state the prompt asks
for; each try's share and verdict is printed, and --sheet draws them all on one picture with their verdicts.
"""

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))


def pictures_in(folder):
    inside = os.path.join(folder, "pictures") if os.path.isdir(os.path.join(folder, "pictures")) else folder
    return sorted(p for p in glob.glob(os.path.join(inside, "*.png")) if not p.endswith("-mask.png"))


def main():
    import numpy as np
    from PIL import Image, ImageDraw

    from model_maker import state_judge, words

    parser = argparse.ArgumentParser()
    parser.add_argument("models")
    parser.add_argument("prompt")
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--sheet")
    args = parser.parse_args()
    subject = words.subject_of(args.prompt)
    wanted, unwanted = words.state_texts(subject)
    if not wanted:
        raise SystemExit(f"{args.prompt!r} asks for no state")
    judge = state_judge.StateJudge(os.path.join(args.models, state_judge.MODEL_FOLDER))
    judged = []
    for folder in args.folders:
        for path in pictures_in(folder):
            mask_path = path[:-4] + "-mask.png"
            mask = np.asarray(Image.open(mask_path)) > 127 if os.path.exists(mask_path) else None
            share = judge.share(Image.open(path), mask, wanted, unwanted)
            passed = share >= state_judge.LEAST_STATE_SHARE
            judged.append((path, share, passed))
            print(f"{'passes' if passed else 'FAILS '} {share:.3f} {os.path.relpath(path)}", flush=True)
    print(f"{sum(p for _, _, p in judged)} of {len(judged)} pass ({subject.state} asked)")
    if args.sheet:
        side, across = 200, 8
        sheet = Image.new("RGB", (side * across, side * ((len(judged) + across - 1) // across)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, (path, share, passed) in enumerate(judged):
            x, y = side * (k % across), side * (k // across)
            sheet.paste(Image.open(path).convert("RGB").resize((side, side)), (x, y))
            draw.rectangle([x, y, x + side - 1, y + side - 1], outline=(40, 160, 40) if passed else (210, 40, 40),
                           width=4)
            draw.text((x + 8, y + 6), f"{subject.state} {share:.2f}", fill=(0, 0, 0))
        sheet.save(args.sheet)


if __name__ == "__main__":
    main()
