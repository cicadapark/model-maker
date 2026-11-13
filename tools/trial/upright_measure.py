#!/usr/bin/env python3
"""The proof for standing hafted weapons upright (python/model_maker/upright.py), on made shapes.

    upright_measure.py OUT.json FOLDER...     every job under the folders with a shape.npz whose prompt is hafted

Each job's shape as the shape model made it (shape.npz, repaired as the finish repairs it) is turned as the finish
turns it (`lying.laid`). Measured, independently of how the turn was found where it can be:
* how far it turned;
* how far the haft leans from the up axis before and after, found by level slabs of the shape (not by the weapon
  checks' own frame, which the turn uses);
* head up: the turned shape's widest spread across in its top quarter against its bottom quarter;
* the grip and the tip: the weapon measures before and after (the grip's distance from the butt, the length), and the
  marks of the turned shape turned back, against those of the made shape.
CPU."""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

UP = np.array([0.0, 1.0, 0.0])


def haft_lean_deg(points, slabs=60):
    """How far the haft leans from the up axis, found independently of the weapon checks' frame: the stood shape cut
    into level slabs, the narrow ones (no wider than 1.5 times the median slab) taken as haft, and the straight line
    through their middles (x and z against height) measured against the vertical."""
    y = points[:, 1]
    edges = np.linspace(y.min(), y.max(), slabs + 1)
    rows = []
    for k in range(slabs):
        inside = (y >= edges[k]) & (y < edges[k + 1])
        if inside.sum() < 6:
            continue
        rows.append(((edges[k] + edges[k + 1]) / 2, points[inside][:, [0, 2]].mean(0), spread(points[inside])))
    if len(rows) < 6:
        return float("nan")
    usual = np.median([r[2] for r in rows])
    haft = [r for r in rows if r[2] <= 1.5 * usual]
    heights = np.array([r[0] for r in haft])
    middles = np.array([r[1] for r in haft])
    slope = np.polyfit(heights, middles, 1)[0]
    return float(np.degrees(np.arctan(np.hypot(*slope))))


def spread(points):
    return float(max(np.ptp(points[:, 0]), np.ptp(points[:, 2]))) if len(points) else 0.0


def measured(folder):
    from model_maker import lying, mesh_repair, mesh_tools as M, weapon_checks as W
    from model_maker.pipeline import Job

    job = Job(folder)
    subject = job.subject
    if subject.weapon_kind != "hafted":
        return None
    shape = np.load(job.path("shape.npz"))
    corners, triangles, _ = mesh_repair.repaired(shape["corners"].astype(np.float64), shape["triangles"].astype(np.int64))
    before, _ = W.measured(corners, triangles, subject)
    stood, _, record = lying.laid(subject, corners, triangles)
    after, _ = W.measured(stood, triangles, subject)
    points, _, _ = M.sampled(stood, triangles, W.SAMPLES, seed=3)
    off = haft_lean_deg(points)
    made_points, _, _ = M.sampled(corners, triangles, W.SAMPLES, seed=3)
    made_off = haft_lean_deg(made_points)
    height = np.ptp(stood[:, 1])
    top = spread(stood[stood[:, 1] >= stood[:, 1].max() - height / 4])
    bottom = spread(stood[stood[:, 1] <= stood[:, 1].min() + height / 4])
    found = {"job": folder, "prompt": job["prompt"], "label": (job.record.get("picture") or {}).get("label"),
             "turned_deg": record["turned_deg"], "haft_off_up_deg": round(off, 2), "made_off_up_deg": round(made_off, 2),
             "top_spread_m": round(top, 4), "bottom_spread_m": round(bottom, 4),
             "length_m": [round(before["length_m"], 4), round(after["length_m"], 4)]}
    if "grip" in before and "grip" in after:
        back = lying.to_made(np.array([after["grip"]["hold"], after["tip"]]), record)
        found["from_butt_m"] = [round(before["grip"]["from_butt_m"], 4), round(after["grip"]["from_butt_m"], 4)]
        found["hold_moved_m"] = round(float(np.linalg.norm(back[0] - before["grip"]["hold"])), 4)
        found["tip_moved_m"] = round(float(np.linalg.norm(back[1] - before["tip"])), 4)
    return found


def main():
    out, folders = sys.argv[1], sys.argv[2:]
    rows = []
    for root in folders:
        for record in sorted(glob.glob(os.path.join(root, "**", "shape.npz"), recursive=True)):
            folder = os.path.dirname(record)
            try:
                found = measured(folder)
            except Exception as error:  # shown, not fatal: one job's fault is not the run's
                print(f"{folder}: {error}", flush=True)
                continue
            if found:
                rows.append(found)
                print(json.dumps(found), flush=True)
    with open(out, "w") as file:
        json.dump(rows, file, indent=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
