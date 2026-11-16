#!/usr/bin/env python3
"""Blind sheets of finished objects, for judging by eye (docs/stats.md in the studio: "would you use it as it is?").

    object_sheets.py OUT RUN=FOLDER [RUN=FOLDER ...] [--seed N]

Each RUN names a batch's folder (its job folders, each with its .glb and report/) under a label kept only in the key
(for example 030-3080=.../batch). One sheet per object made: its prompt and a random code at the top; its first
picture and the report's turntable (8 sides); below, the object painted as it is, seen from below at four turns, so
an underside's holes, floating parts and smears are judged too; beside them, the bare shape set down on a floor as it
would settle, at four turns, so a short leg or a rocking base shows (on 13 Nov a bedside table judged good by eye had
its back legs 85 mm short: the turntable has no floor). The sheets are shuffled within each prompt and named
by prompt and code only, never by run; the key (OUT/key.json: code -> run and folder) is written apart, to be opened
only once every verdict is written. A job that made no object has no sheet; the key lists it as not made.
"""

import io
import json
import os
import random
import string
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
from model_maker import glb, object_checks, render  # noqa: E402

SIDE = 360
BELOW_TILT = -60.0  # seen from below, 60 degrees under the horizon
BELOW_TURNS = (0, 90, 180, 270)
HEADER = 40


def mesh_of(path):
    """The .glb's triangles (corners, triangles), each triangle's corner UVs and its texture (0-1), all primitives."""
    with open(path, "rb") as file:
        document, binary = glb.read_document(file.read())
    corners, triangles, uvs, texture = [], [], [], None
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            points = glb.accessor_values(document, binary, primitive["attributes"]["POSITION"])
            faces = glb.accessor_values(document, binary, primitive["indices"]).reshape(-1, 3)
            uv = glb.accessor_values(document, binary, primitive["attributes"]["TEXCOORD_0"]) \
                if "TEXCOORD_0" in primitive["attributes"] else np.zeros((len(points), 2), np.float32)
            triangles.append(faces + sum(len(c) for c in corners))
            corners.append(points)
            uvs.append(uv)
            if texture is None and "material" in primitive:
                texture = texture_of(document, binary, primitive["material"])
    corners = np.concatenate(corners).astype(np.float64)
    triangles = np.concatenate(triangles).astype(np.int64)
    triangle_uvs = np.concatenate(uvs)[triangles]
    return corners, triangles, triangle_uvs, texture


def texture_of(document, binary, material_index):
    material = document["materials"][material_index]
    found = material.get("pbrMetallicRoughness", {}).get("baseColorTexture")
    if found is None:
        return None
    image = document["images"][document["textures"][found["index"]]["source"]]
    view = document["bufferViews"][image["bufferView"]]
    data = binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float64) / 255


def below(path):
    corners, triangles, triangle_uvs, texture = mesh_of(path)
    colour_at = render.texture_lookup(triangle_uvs, texture) if texture is not None else None
    extent = float(np.ptp(corners, axis=0).max()) * 1.05
    return [render.shaded(corners, triangles, turn, SIDE, colour_at, tilt_deg=BELOW_TILT, extent=extent)
            for turn in BELOW_TURNS]


FLOOR_SHARE = 0.012  # the floor plate's thickness, as a share of the object's height


def set_down(corners, triangles):
    """The shape turned onto the face it settles on (object_checks.resting_face), standing on a floor plate."""
    normal = object_checks.resting_face(corners, triangles)[0]
    down = np.array([0.0, -1.0, 0.0])
    if normal is not None and np.linalg.norm(np.cross(normal, down)) > 1e-9:
        axis = np.cross(normal, down)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.clip(normal @ down, -1, 1))
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        turn = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
        corners = corners @ turn.T
    corners = corners - [0.0, corners[:, 1].min(), 0.0]
    low, high = corners.min(0), corners.max(0)
    reach = 0.3 * float(max(high[0] - low[0], high[2] - low[2]))
    thick = FLOOR_SHARE * float(high[1] - low[1])
    (x0, z0), (x1, z1) = (low[0] - reach, low[2] - reach), (high[0] + reach, high[2] + reach)
    box = np.array([[x, y, z] for x in (x0, x1) for y in (-thick, 0.0) for z in (z0, z1)])
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]) + len(corners)
    return np.concatenate([corners, box]), np.concatenate([triangles, faces])


def standing(path):
    corners, triangles, _, _ = mesh_of(path)
    corners, triangles = set_down(corners, triangles)
    extent = float(np.ptp(corners, axis=0).max()) * 1.05
    return [render.shaded(corners, triangles, turn, SIDE, None, tilt_deg=0.0, extent=extent) for turn in BELOW_TURNS]


def sheet(job, prompt, code):
    report = os.path.join(job, "report")
    picture = Image.open(os.path.join(report, "picture.png")).convert("RGB").resize((SIDE, SIDE))
    turntable = Image.open(os.path.join(report, "turntable.png")).convert("RGB")
    turntable = turntable.resize((turntable.width * SIDE // turntable.height, SIDE))
    model = next(os.path.join(job, name) for name in sorted(os.listdir(job)) if name.endswith(".glb"))
    views = below(model)
    stood = standing(model)
    width = SIDE + turntable.width
    out = Image.new("RGB", (width, HEADER + 2 * SIDE), "white")
    ImageDraw.Draw(out).text((10, 12), f"{code}   {prompt}   (top: the picture and 8 sides; below: from underneath, "
                             "then the bare shape set down on a floor)", fill="black")
    out.paste(picture, (0, HEADER))
    out.paste(turntable, (SIDE, HEADER))
    for at, view in enumerate(views + stood):
        out.paste(view, (SIDE + at * SIDE, HEADER + SIDE))
    return out


def jobs_of(folder):
    for name in sorted(os.listdir(folder)):
        job = os.path.join(folder, name)
        if os.path.isfile(os.path.join(job, "job.json")):
            yield job


def main(argv):
    seed = 20261113
    if "--seed" in argv:
        at = argv.index("--seed")
        seed = int(argv[at + 1])
        argv = argv[:at] + argv[at + 2:]
    out, runs = argv[0], [item.partition("=")[::2] for item in argv[1:]]
    os.makedirs(out, exist_ok=True)
    rng = random.Random(seed)
    by_prompt, key, codes = {}, {}, set()
    for label, folder in runs:
        for job in jobs_of(folder):
            with open(os.path.join(job, "job.json")) as file:
                prompt = json.load(file)["prompt"]
            made = any(name.endswith(".glb") for name in os.listdir(job))
            by_prompt.setdefault(prompt, []).append((label, job, made))
    for prompt in sorted(by_prompt):
        entries = by_prompt[prompt]
        rng.shuffle(entries)
        for label, job, made in entries:
            code = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
            while code in codes:
                code = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
            codes.add(code)
            key[code] = {"run": label, "folder": job, "prompt": prompt, "made": made}
            if made:
                name = f"{prompt.replace(' ', '-').replace(chr(39), '')}--{code}.png"
                sheet(job, prompt, code).save(os.path.join(out, name))
    with open(os.path.join(out, "key.json"), "w") as file:
        json.dump(key, file, indent=1, sort_keys=True)
    made = sum(1 for one in key.values() if one["made"])
    print(f"{made} sheets in {out} ({len(key) - made} not made); the key in {os.path.join(out, 'key.json')}, "
          "to be opened once every verdict is written")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
