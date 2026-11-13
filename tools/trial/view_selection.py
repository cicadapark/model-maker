#!/usr/bin/env python3
"""The bake as made beside the bake with one view chosen per place (docs/trials/2026-11-11-030-diagnosis.md).

    view_selection.py JOB OUT.png [MARGIN]
    view_selection.py --coverage OUT.json JOB...     the coverage as made and with one view per place, no pictures

As made, each texel is the inverse-distance mix of its four nearest painted points, from whichever views they came:
where two views see a surface and paint it differently, their pixels interleave into a stipple. Here a view's painted
point is left out where another view painted the same place more squarely by over MARGIN (default 0.05,
`view_agreement.one_view_per_place`, the pipeline's own): one view per place, as texturing from photographs chooses (Lempitsky and Ivanov 2007; Waechter,
Moehrle and Goesele 2014) and as MV-Adapter's own projection weights its views by the cosine to the surface
(`ExponentialBlend`). OUT.png: the turntable's first four turns, as made above, chosen below. Prints the coverage of
each. CPU; a measure, not the pipeline's bake.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
sys.path.insert(0, HERE)

AXES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)


def coverage(folders, out):
    import json

    rows = []
    for folder in folders:
        try:
            found = compared(folder, None, pictures=False)
        except Exception as error:  # a job whose cloud does not split, or an old record: shown, not fatal
            print(f"{folder}: {error}", flush=True)
            continue
        rows.append({"job": folder, **found})
        print(json.dumps(rows[-1]), flush=True)
    with open(out, "w") as file:
        json.dump(rows, file, indent=0)


def main():
    if sys.argv[1] == "--coverage":
        coverage(sys.argv[3:], sys.argv[2])
        return 0
    compared(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.05)
    return 0


def compared(folder, out, margin=0.05, pictures=True):
    from PIL import Image

    from model_maker import bake, inspection, render, touch_up
    from model_maker.pipeline import Job
    from view_agreement import split

    job = Job(folder)
    shape = np.load(job.path("shape.npz"))
    made_corners, made_triangles = shape["corners"].astype(np.float64), shape["triangles"].astype(np.int64)
    mask = np.asarray(Image.open(job.path("mask.png"))) > 127
    picture = np.asarray(Image.open(job.path("picture.png")).convert("RGB"))
    judgement = inspection.judged(job.subject, mask, made_corners, made_triangles, picture)
    painted = np.load(job.path("painted.npz"))
    colours, cloud = judgement.paint_followed(made_corners, painted)
    sources = touch_up.fill_sources(painted, job.record.get("paint", {}).get("touch_up"))
    if sources is None:
        sources = np.zeros(len(painted["cloud_points"]), dtype=np.int16)
    normals = painted["cloud_normals"].astype(np.float64)
    views = split(painted["cloud_points"].astype(np.float64), normals, sources)
    if views is None:
        raise ValueError("the colour cloud does not split into its six views")
    from model_maker import view_agreement

    # Each point's view, and each view's direction (its points' mean normal, the nearest axis), as paint.painted now
    # records them; then the pipeline's own selection.
    view_of = np.full(len(sources), -1)
    directions = np.zeros((6, 3))
    for v, points in enumerate(views):
        view_of[points] = v
        directions[v] = AXES[np.argmax(AXES @ normals[points].mean(0))]
    keep = view_agreement.one_view_per_place(painted["cloud_points"], normals, view_of, directions, margin)
    rows, found = [], {}
    for name, chosen in (("as_made", cloud), ("one_view", tuple(np.asarray(part)[keep] for part in cloud))):
        baked = bake.baked(judgement.corners, judgement.triangles, colours, chosen, 2048)
        found[name] = round(baked["coverage"], 4)
        if pictures:
            at = render.texture_lookup(baked["uvs"][baked["triangles"]], baked["texture"])
            rows.append(np.asarray(render.turntable(baked["corners"], baked["triangles"], at, side=300,
                                                    turns=(0, 45, 90, 135))))
            print(os.path.basename(os.path.normpath(folder)), "coverage", found[name], "points", len(chosen[0]))
    if pictures:
        Image.fromarray(np.concatenate(rows, 0)).save(out)
    return found


if __name__ == "__main__":
    sys.exit(main())
