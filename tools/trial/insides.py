#!/usr/bin/env python3
"""The bake again from stored paint data, four ways, for what a vessel's inner walls show (docs/objects.md, "Insides,
bake-only"). Nothing is painted again and nothing is written into the jobs.

    insides.py sheets OUT JOB [JOB ...]     as made and with the slanted colour left out (way 1): the coverage both
                                            ways, and pictures looking into each object for labelling by eye, blind
                                            (OUT/sheets/<code>.png, the code saying nothing of the way; OUT/key.json)
    insides.py ways OUT JOB [JOB ...]       the four ways side by side, each with its coverage (OUT/<job>.png)

The ways, for the corners the colour cloud does not reach (every view saw them at a slant, or not at all):
  as made   the colour the view painter's views gave them from a facing of 0.2 (`paint.LEAST_FACING`);
  way 1     the colour spread from the corners the cloud reaches round them (`paint.spread`);
  way 2     the same, never spread across a crease (corner normals more than 45 degrees apart);
  way 3     way 2 spread from the six views' paint only: the touch-up's fill left out where the six views did not
            paint, so the bake covers less.
Each JOB is a job folder holding shape.npz and painted.npz. CPU only.
"""

import hashlib
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import sparse
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("MM_PYTHON") or os.path.join(HERE, "..", "..", "python"))

TEXTURE = 1024
SIDE = 300
TURNS = (0, 90, 180, 270)
TILT = 55.0
CREASE_COS = np.cos(np.radians(45))


def reached_by(points, normals, corners, corner_normals):
    """Which corners the cloud (points, normals) reaches, as the bake reaches a texel."""
    from model_maker import bake

    if len(points) == 0:
        return np.zeros(len(corners), dtype=bool)
    reach = bake.CLOUD_REACH_SPACINGS * bake.cloud_spacing(points)
    distances, nearest = cKDTree(points).query(corners, k=bake.CLOUD_NEIGHBOURS, distance_upper_bound=reach)
    found = np.isfinite(distances)
    safe = np.where(found, nearest, 0)
    return (found & (np.einsum("pd,pkd->pk", corner_normals, normals[safe]) > bake.SAME_FACING)).any(1)


def crease_spread(colours, known, triangles, corner_normals, rounds=400):
    """`paint.spread`, along edges whose corners' normals lie within 45 degrees of each other only."""
    n = len(colours)
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edges = edges[np.einsum("ij,ij->i", corner_normals[edges[:, 0]], corner_normals[edges[:, 1]]) >= CREASE_COS]
    adjacency = sparse.coo_matrix((np.ones(len(edges) * 2), (np.r_[edges[:, 0], edges[:, 1]],
                                                             np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    colours, weight = colours.copy(), known.astype(float)
    for _ in range(rounds):
        summed = adjacency @ (colours * weight[:, None])
        counted = adjacency @ weight
        fresh = (weight == 0) & (counted > 0)
        if not fresh.any():
            break
        colours[fresh] = summed[fresh] / counted[fresh, None]
        weight[fresh] = 1.0
    return colours


def ways(folder):
    """{way: (corner colours, cloud)} for a job, and its corners and triangles."""
    from model_maker import mesh_tools as M, touch_up, view_agreement
    from model_maker.paint import spread

    with open(os.path.join(folder, "job.json")) as file:
        record = json.load(file)
    shape = np.load(os.path.join(folder, "shape.npz"))
    corners, triangles = shape["corners"].astype(np.float64), shape["triangles"].astype(np.int64)
    painted = np.load(os.path.join(folder, "painted.npz"))
    points, normals = (painted[k].astype(np.float64) for k in ("cloud_points", "cloud_normals"))
    cloud_colours = np.asarray(painted["cloud_colours"])
    keep = np.ones(len(points), dtype=bool)
    if "cloud_view" in painted.files:
        # One view per place, as the finish bakes it.
        keep = view_agreement.one_view_per_place(painted["cloud_points"], painted["cloud_normals"],
                                                 painted["cloud_view"], painted["view_directions"])
    sources = touch_up.fill_sources(painted, record.get("paint", {}).get("touch_up"))
    sources = np.zeros(len(points), dtype=np.int16) if sources is None else sources
    corner_normals = M.corner_normals(corners, triangles)
    made = np.asarray(painted["colours"], dtype=np.float64)
    reached = reached_by(points, normals, corners, corner_normals)
    six = sources == 0
    reached_six = reached_by(points[six], normals[six], corners, corner_normals)
    nearest_corner = cKDTree(corners).query(points)[1]
    keep_six = keep & (six | reached_six[nearest_corner])
    cloud = (points[keep], normals[keep], cloud_colours[keep])
    found = {"as made": (made, cloud),
             "way 1": (spread(made, reached, triangles, rounds=400), cloud),
             "way 2": (crease_spread(made, reached, triangles, corner_normals), cloud),
             "way 3": (crease_spread(made, reached_six, triangles, corner_normals),
                       (points[keep_six], normals[keep_six], cloud_colours[keep_six]))}
    return found, corners, triangles, float((~reached).mean())


def sheet(corners, triangles, colours, cloud, label=None):
    from model_maker import bake, render

    baked = bake.baked(corners, triangles, colours, cloud, TEXTURE)
    colour_at = render.texture_lookup(baked["uvs"][baked["triangles"]], baked["texture"])
    extent = float(np.ptp(corners, axis=0).max()) * 1.05
    top = 30 if label else 0
    out = Image.new("RGB", (SIDE * len(TURNS), SIDE + top), (255, 255, 255))
    for k, turn in enumerate(TURNS):
        out.paste(render.shaded(baked["corners"], baked["triangles"], turn, SIDE, colour_at, tilt_deg=TILT,
                                extent=extent), (k * SIDE, top))
    if label:
        ImageDraw.Draw(out).text((6, 8), f"{label}: coverage {baked['coverage']:.1%}", fill=(0, 0, 0))
    return out, baked["coverage"]


def blind_sheets(out, jobs):
    os.makedirs(os.path.join(out, "sheets"), exist_ok=True)
    key_path = os.path.join(out, "key.json")
    key = json.load(open(key_path)) if os.path.exists(key_path) else {}
    rng = random.Random(len(key) + 11)
    for folder in jobs:
        found, corners, triangles, not_reached = ways(folder)
        row = {"job": folder, "corners_not_reached": round(not_reached, 4)}
        for way in rng.sample(["as made", "way 1"], 2):
            picture, coverage = sheet(corners, triangles, *found[way])
            row[way] = round(coverage, 5)
            code = hashlib.sha256(f"{folder}|{way}|{rng.random()}".encode()).hexdigest()[:6]
            picture.save(os.path.join(out, "sheets", f"{code}.png"))
            key[code] = {"job": folder, "way": way}
        with open(key_path, "w") as file:
            json.dump(key, file, indent=1)
        with open(os.path.join(out, "measures.jsonl"), "a") as file:
            file.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


def side_by_side(out, jobs):
    os.makedirs(out, exist_ok=True)
    for folder in jobs:
        found, corners, triangles, _ = ways(folder)
        rows, coverages = [], {}
        for way, (colours, cloud) in found.items():
            picture, coverages[way] = sheet(corners, triangles, colours, cloud, way)
            rows.append(picture)
        page = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), (255, 255, 255))
        for k, picture in enumerate(rows):
            page.paste(picture, (0, k * picture.height))
        # The run's folder, its batch and the job: two cards' runs hold jobs of one name.
        page.save(os.path.join(out, "__".join(os.path.abspath(folder).split(os.sep)[-3:]) + ".png"))
        print(folder, json.dumps({k: round(v, 4) for k, v in coverages.items()}), flush=True)


def main():
    step, out, jobs = sys.argv[1], sys.argv[2], sys.argv[3:]
    if step == "sheets":
        blind_sheets(out, jobs)
    elif step == "ways":
        side_by_side(out, jobs)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
