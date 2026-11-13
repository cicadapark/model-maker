#!/usr/bin/env python3
"""The candidate check "the paint matches the picture where the picture shows it", measured on finished jobs
(docs/trials/2026-11-11-030-diagnosis.md): the likeness made local.

    local_likeness.py ROOT OUT.json [--keep-shift] [--maps OUT.png SHEET...]

Read from each job's likeness sheet (report/coverage-likeness.png: the picture, the painted model drawn unlit from
the picture's view, their difference; `likeness.likeness`), where both are. Both images are blurred before their
colours are compared, as S-CIELAB filters an image before CIELAB (Zhang and Wandell 1996), so fine texture a little
out of place does not count as a colour error; the blur is 2 % of the object's height, within where both are (a
normalised blur), a 3-pixel edge left out. The object's own shift (the median difference: the picture lit, the model
unlit, a view a little lighter all over) is taken out unless --keep-shift. Then the largest joined place whose
difference is over 20, 30 and 40 (CIE76 delta E), each as a share of where both are. --maps draws the difference of
the given sheets (green alike, red unlike, magenta over 30), one row each.
"""

import glob
import json
import os
import sys

import numpy as np

BLUR_SHARE = 0.02
EDGE_PIXELS = 3
LIMITS = (20, 30, 40)


def panels(path):
    from PIL import Image

    sheet = np.asarray(Image.open(path).convert("RGB")).astype(float) / 255
    third = sheet.shape[1] // 3
    picture, made, heat = sheet[:, :third], sheet[:, third:2 * third], sheet[:, 2 * third:3 * third]
    # The difference panel is green to red where both are, black elsewhere.
    return picture, made, heat[..., :2].sum(-1) > 0.5


def measured(picture, made, both, keep_shift=False):
    """({limit: the largest place over it, "mean": the mean difference}, the difference, where it is judged), or None."""
    from scipy import ndimage
    from skimage.color import rgb2lab

    rows = np.flatnonzero(both.any(1))
    if len(rows) < 2:
        return None
    inner = ndimage.binary_erosion(both, iterations=EDGE_PIXELS)
    if inner.sum() < 50:
        inner = both
    sigma = max(1.0, BLUR_SHARE * (rows[-1] - rows[0] + 1))
    weight = np.maximum(ndimage.gaussian_filter(inner.astype(float), sigma), 1e-6)
    blurred = [np.stack([ndimage.gaussian_filter(lab[..., c] * inner, sigma) for c in range(3)], -1)
               / weight[..., None] for lab in (rgb2lab(picture), rgb2lab(made))]
    shift = blurred[0] - blurred[1]
    if not keep_shift:
        shift = shift - np.median(shift[inner], axis=0)
    difference = np.where(inner, np.linalg.norm(shift, axis=-1), 0.0)
    found = {"mean": float(difference[inner].mean())}
    for limit in LIMITS:
        places, count = ndimage.label(difference > limit)
        found[str(limit)] = float(np.bincount(places.ravel())[1:].max() / inner.sum()) if count else 0.0
    return found, difference, inner


def maps(out, sheets):
    from PIL import Image

    rows = []
    for sheet in sheets:
        picture, made, both = panels(sheet)
        _, difference, inner = measured(picture, made, both)
        shown = np.clip(difference / 40, 0, 1)
        heat = np.stack([shown, 1 - shown, np.zeros_like(shown)], -1) * inner[..., None]
        heat[(difference > 30) & inner] = [1, 0, 1]
        rows.append(np.concatenate([picture, made, heat], 1))
    wide = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 4), (0, wide - r.shape[1]), (0, 0))) for r in rows], 0)
    Image.fromarray((sheet * 255).astype(np.uint8)).save(out)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--maps":
        maps(args[1], args[2:])
        return 0
    root, out = args[0], args[1]
    keep_shift = "--keep-shift" in args
    rows = []
    for sheet in sorted(glob.glob(os.path.join(root, "**", "report", "coverage-likeness.png"), recursive=True)):
        job = os.path.dirname(os.path.dirname(sheet))
        found = measured(*panels(sheet), keep_shift=keep_shift)
        if found is None:
            continue
        status = ""
        for record in glob.glob(os.path.join(job, "*.provenance.json")):
            with open(record) as file:
                status = json.load(file).get("status", "")
        rows.append({"job": os.path.relpath(job, root), "status": status, **found[0]})
    with open(out, "w") as file:
        json.dump(rows, file, indent=0)
    for row in sorted(rows, key=lambda r: -r["30"]):
        print(f'{row["20"]:.3f} {row["30"]:.3f} {row["40"]:.3f} mean {row["mean"]:5.1f}  {row["job"]}  '
              f'{row["status"][:40]}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
