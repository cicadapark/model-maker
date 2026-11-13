#!/usr/bin/env python3
"""Where two neighbouring painted views see the same surface, do they paint it alike? (docs/trials/
2026-11-11-030-diagnosis.md). The photo-consistency idea of texturing from photographs (Waechter, Moehrle and Goesele
2014: a view whose colour is an outlier among the views of a face is dropped), between the view painter's views.

    view_agreement.py ROOT OUT.json     every job under ROOT with a painted.npz

The colour cloud (painted.npz) is split into its views (`split`). For each pair of neighbouring side views (front and
right, right and back, ...), each point of one is matched with the nearest point of the other within 2.5 of the
cloud's spacings and facing the same way (normals within about 25 degrees), and their colours compared (CIE76 delta
E): the pair's share over 30. A view's disagreement is the lesser of its two pairs' (a view out of family disagrees
with both its neighbours). Prints one row a job, the most disagreeing view first. CPU.
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

# A side view's rows run from the top down: its points' height never rises by more than this share of the object's
# height; the next view starts where it does.
JUMP_SHARE = 0.05


def split(points, normals, sources):
    """The six views' points (index arrays) of a colour cloud in the made frame (y up), or None when it does not split.
    The six views come first (`sources` 0) in paint.painted's order: front, the sides and back (each in rows from the
    top), then from above (facing up), then from below (facing down); the touch-up's fill after them."""
    six = np.flatnonzero(sources == 0)
    p, n = points[six], normals[six]
    jumps = np.flatnonzero(np.diff(p[:, 1]) > JUMP_SHARE * np.ptp(p[:, 1]))
    if len(jumps) < 4:
        return None
    above = jumps[3] + 1
    down = np.flatnonzero(n[above:, 1] < 0)
    below = above + (down[0] if len(down) else len(p) - above)
    bounds = [0, *(jumps[:3] + 1), above, below, len(p)]
    return [six[bounds[k]:bounds[k + 1]] for k in range(6)]


def judged(painted_path):
    """The pipeline's own measure (model_maker.view_agreement.measured) on a job's paint, its views found by `split`;
    {"pairs", "views" (as "per_view"), "worst", "counts"}, or None when the cloud does not split."""
    from model_maker import view_agreement

    painted = np.load(painted_path)
    points, normals = painted["cloud_points"].astype(np.float64), painted["cloud_normals"].astype(np.float64)
    sources = painted["cloud_from"] if "cloud_from" in painted.files else np.zeros(len(points), dtype=np.int16)
    views = split(points, normals, sources)
    if views is None:
        return None
    view_of = np.full(len(points), -1)
    for v, members in enumerate(views):
        view_of[members] = v
    found = view_agreement.measured(points, normals, painted["cloud_colours"], view_of)
    return {"counts": [len(v) for v in views], "pairs": found["pairs"], "per_view": found["views"],
            "worst": found["worst"]}


def main():
    root, out = sys.argv[1], sys.argv[2]
    results = []
    for path in sorted(glob.glob(os.path.join(root, "**", "painted.npz"), recursive=True)):
        job = os.path.dirname(path)
        found = judged(path)
        if found is None:
            print("not split:", job, file=sys.stderr)
            continue
        status = ""
        for record in glob.glob(os.path.join(job, "*.provenance.json")):
            with open(record) as file:
                status = json.load(file).get("status", "")
        found.update(job=os.path.relpath(job, root), status=status)
        results.append(found)
        scored = [(s, v) for v, s in enumerate(found["per_view"]) if s is not None]
        worst = max(scored) if scored else (float("nan"), -1)
        pairs = " ".join(f"{k}:{p:.2f}" if p is not None else f"{k}:-" for k, p in found["pairs"].items())
        print(f"{worst[0]:.3f} view{worst[1]} {pairs}  {found['job']}  {status[:40]}", flush=True)
    with open(out, "w") as file:
        json.dump(results, file, indent=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
