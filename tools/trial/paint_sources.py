#!/usr/bin/env python3
"""Where each colour of a finished object came from (docs/trials/2026-11-11-030-diagnosis.md).

    paint_sources.py JOB OUT.png [odd|dark|grey]

The job is baked as the finish bakes it (`pipeline.finished`), then baked again with each painted point's colour
replaced by its source's marker colour: the six views split apart (`view_agreement.split`), the touch-up's fill, and,
where no painted point is near and the bake falls back to the corners' own colours, whether a view saw that corner at
a slant (`known`: facing from paint.LEAST_FACING, under the cloud's 0.55) or it was spread from its neighbours. OUT.png
shows both from the turntable's four quarter turns and, below, looking into the object from above at a slant.

Then the object is drawn unlit from six directions, and the pixels picked by the mode are counted by their source:
"odd" (more than 25 delta E from the mean colour round them, a window a fifth of the object across), "dark" (lightness
25 under the object's median) or "grey" (under 0.4 of the object's median chroma). Prints each source's share of all
the pixels seen and of the picked ones. CPU; needs the job's painted.npz and shape.npz.
"""

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
sys.path.insert(0, HERE)

# Views 0-3 greens and teals, from above yellow, from below grey, the fill blue; corners seen at a slant magenta,
# spread red.
MARKERS = np.array([[0.1, 0.6, 0.1], [0.1, 0.8, 0.8], [0.2, 0.9, 0.2], [0.1, 0.5, 0.6], [1.0, 0.85, 0.0],
                    [0.6, 0.6, 0.6], [0.2, 0.2, 1.0]])
SLANT = np.array([0.9, 0.2, 0.9])
SPREAD = np.array([0.9, 0.1, 0.1])
NAMES = ["view 0", "view 1", "view 2", "view 3", "above", "below", "fill", "slant", "spread"]
DIRECTIONS = ((315, 55), (135, 55), (0, 15), (90, 15), (180, 15), (270, 15))
ODD_DELTA_E = 25.0


def picked(mode, lab, hit):
    from scipy import ndimage

    if mode == "dark":
        return hit & (lab[..., 0] < np.median(lab[..., 0][hit]) - ODD_DELTA_E)
    if mode == "grey":
        chroma = np.hypot(lab[..., 1], lab[..., 2])
        return hit & (chroma < 0.4 * np.median(chroma[hit]))
    window = max(9, int(0.2 * np.sqrt(hit.sum()))) / 4
    weight = np.maximum(ndimage.gaussian_filter(hit.astype(float), window), 1e-6)
    around = np.stack([ndimage.gaussian_filter(lab[..., c] * hit, window) for c in range(3)], -1) / weight[..., None]
    return hit & (np.linalg.norm(lab - around, axis=-1) > ODD_DELTA_E)


def main():
    from scipy.spatial import cKDTree
    from skimage.color import rgb2lab

    from model_maker import bake, inspection, likeness, lying, render, touch_up
    from model_maker.pipeline import Job
    from view_agreement import split

    folder, out = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "odd"
    job = Job(folder)
    shape = np.load(job.path("shape.npz"))
    made_corners, made_triangles = shape["corners"].astype(np.float64), shape["triangles"].astype(np.int64)
    mask = np.asarray(Image.open(job.path("mask.png"))) > 127
    picture = np.asarray(Image.open(job.path("picture.png")).convert("RGB"))
    judgement = inspection.judged(job.subject, mask, made_corners, made_triangles, picture)
    corners, triangles = judgement.corners, judgement.triangles
    painted = np.load(job.path("painted.npz"))
    colours, cloud = judgement.paint_followed(made_corners, painted)
    baked = bake.baked(corners, triangles, colours, cloud, 1024)
    sources = touch_up.fill_sources(painted, job.record.get("paint", {}).get("touch_up"))
    if sources is None:
        sources = np.zeros(len(painted["cloud_points"]), dtype=np.int16)
    label = np.full(len(sources), 6)
    views = split(painted["cloud_points"].astype(np.float64), painted["cloud_normals"].astype(np.float64), sources)
    if views is None:
        raise SystemExit(f"{folder}: the colour cloud does not split into its six views")
    for v, idx in enumerate(views):
        label[idx] = v
    nearest = cKDTree(made_corners).query(lying.to_made(corners + judgement.lowered, judgement.laid))[1]
    known = np.asarray(painted["known"])[nearest]
    marked = bake.baked(corners, triangles, np.where(known[:, None], SLANT, SPREAD), (cloud[0], cloud[1], MARKERS[label]),
                        1024)
    real_at = render.texture_lookup(baked["uvs"][baked["triangles"]], baked["texture"])
    mark_at = render.texture_lookup(marked["uvs"][marked["triangles"]], marked["texture"])
    rows = [np.asarray(render.turntable(baked["corners"], baked["triangles"], at, side=300, turns=(0, 90, 180, 270)))
            for at in (real_at, mark_at)]
    into = [np.asarray(render.shaded(baked["corners"], baked["triangles"], 315, 300, at, tilt_deg=55))
            for at in (real_at, mark_at)]
    rows.append(np.concatenate(into + [np.full_like(into[0], 255)] * 2, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(out)
    every_marker = np.vstack([MARKERS, SLANT, SPREAD])
    counts, every = np.zeros(len(NAMES)), np.zeros(len(NAMES))
    for turn, tilt in DIRECTIONS:
        real, hit = likeness.unlit(baked["corners"], baked["triangles"], turn, 400, real_at, tilt)
        source, _ = likeness.unlit(marked["corners"], marked["triangles"], turn, 400, mark_at, tilt)
        chosen = picked(mode, rgb2lab(real), hit)
        for found, where in ((every, hit), (counts, chosen)):
            which = np.linalg.norm(source[where][:, None, :] - every_marker[None], axis=2).argmin(1)
            found += np.bincount(which, minlength=len(NAMES))
    print(os.path.basename(os.path.normpath(folder)), mode, "pixels:", int(counts.sum()), "of", int(every.sum()))
    for k, name in enumerate(NAMES):
        print(f"  {name:7s} of all seen {every[k] / every.sum():.3f}   of the {mode} ones "
              f"{counts[k] / max(counts.sum(), 1):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
