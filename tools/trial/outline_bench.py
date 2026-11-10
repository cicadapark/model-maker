#!/usr/bin/env python3
"""The outline check's values and seconds on made objects, as the shape stage and the finish judge them.

    python3 tools/trial/outline_bench.py OUT.json FOLDER...

Each FOLDER is a made object (job.json, mask.png, shape.npz and its tries shape-SEED.npz) or a regression case
(job.json, mask.png, shape.npz). For each try, the shape as the shape stage judges it (laid when it lies); for the
kept shape, as the finish judges it (repaired, laid, levelled). Recorded for each: the view found (turn, tilt, the
camera's distance), the overlap, the edges' mean and 95th percentile, the thickest piece in a gap, the verdict, and the
seconds the search and the distances took. Run it before and after a change to the outline check and compare.
"""

import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))


def judged(mask, corners, triangles):
    from model_maker import object_checks as O

    began = time.perf_counter()
    if hasattr(O, "outline_judged"):
        # The code as it is: judged against the shape model's cut-out, the view refined at every distance.
        from model_maker.cutout import for_shape

        view = O.outline_search(for_shape(mask), corners, triangles)
        iou, turn, tilt, distance = view.iou, view.turn, view.tilt, view.distance
        O.overlay_of(view.painted, view.made)
        searched = time.perf_counter()
        mean, far, apart = O.outline_distance(view)
        passed = O.outline_checks(view, mean, far, apart)[0].passed
    elif hasattr(O, "outline_search"):
        view = O.outline_search(mask, corners, triangles)
        iou, turn, tilt, distance = view.iou, view.turn, view.tilt, view.distance
        O.overlay_of(view.painted, view.made)
        searched = time.perf_counter()
        mean, far, apart = O.outline_distance(mask, corners, triangles, turn, tilt, distance, view)
        passed = O.outline_checks(iou, turn, tilt, mean, far, apart, distance)[0].passed
    else:
        # Older code, before the search kept its outlines: the distances drew the best view again.
        iou, turn, tilt, distance, _ = O.outline_overlap(mask, corners, triangles)
        searched = time.perf_counter()
        mean, far, apart = O.outline_distance(mask, corners, triangles, turn, tilt, distance)
        passed = O.outline_checks(iou, turn, tilt, mean, far, apart, distance)[0].passed
    done = time.perf_counter()
    return {"iou": iou, "turn": turn, "tilt": tilt, "distance": distance, "mean": mean, "far": far, "apart": apart,
            "passed": passed, "search_s": searched - began, "distance_s": done - searched}


def object_rows(folder):
    import numpy as np
    from PIL import Image

    from model_maker import levelling, lying, mesh_repair, words

    with open(os.path.join(folder, "job.json")) as file:
        job = json.load(file)
    subject = words.subject_of(job["prompt"])
    mask = np.asarray(Image.open(os.path.join(folder, "mask.png"))) > 127
    rows = []
    # The shape stage: each try as saved (set down and repaired), laid when it lies.
    for path in sorted(glob.glob(os.path.join(folder, "shape-*.npz"))):
        shape = np.load(path)
        corners, triangles, laid = lying.laid(subject, shape["corners"].astype(np.float64),
                                              shape["triangles"].astype(np.int64))
        rows.append({"stage": "shape", "try": os.path.basename(path)[:-4],
                     **judged(mask, lying.to_made(corners, laid), triangles)})
    # The finish: the kept shape repaired, laid, levelled.
    shape = np.load(os.path.join(folder, "shape.npz"))
    corners, triangles, _ = mesh_repair.repaired(shape["corners"].astype(np.float64),
                                                 shape["triangles"].astype(np.int64))
    corners, triangles, laid = lying.laid(subject, corners, triangles)
    if subject.stands:
        corners, triangles, _ = levelling.levelled(corners, triangles)
    rows.append({"stage": "finish", "try": "shape", **judged(mask, lying.to_made(corners, laid), triangles)})
    return rows


def main():
    out, folders = sys.argv[1], sys.argv[2:]
    found = {}
    for folder in folders:
        name = "/".join(os.path.normpath(folder).split(os.sep)[-2:])
        found[name] = object_rows(folder)
        for row in found[name]:
            print(f"{name:52s} {row['stage']:6s} {row['try']:9s} {row['mean']:.4%} {row['far']:.4%} "
                  f"{row['apart']:.3%} {row['turn']:+6.1f} {row['tilt']:4.1f} {row['distance']} "
                  f"{'pass' if row['passed'] else 'FAIL'} search {row['search_s']:.1f}s dist {row['distance_s']:.2f}s",
                  flush=True)
    with open(out, "w") as file:
        json.dump(found, file, indent=1)


if __name__ == "__main__":
    main()
