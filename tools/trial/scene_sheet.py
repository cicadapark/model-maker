#!/usr/bin/env python3
"""A .glb drawn as its scene places it: every root node of the scene that shows the object's mesh, with the node's
scale and translation (a pair's second, mirrored or copied), from eight turns and from below. For looking at what a
viewer shows, not only the mesh as made.

    scene_sheet.py MODEL.glb OUT.png
"""

import io
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

from model_maker import glb, render  # noqa: E402


def placed(path):
    """The scene's object meshes as placed: (corners, triangles, each triangle's UVs, the texture 0-1)."""
    with open(path, "rb") as file:
        document, binary = glb.read_document(file.read())
    primitive = document["meshes"][0]["primitives"][0]
    points = glb.accessor_values(document, binary, primitive["attributes"]["POSITION"]).astype(np.float64)
    faces = glb.accessor_values(document, binary, primitive["indices"]).reshape(-1, 3)
    uv = glb.accessor_values(document, binary, primitive["attributes"]["TEXCOORD_0"])
    view = document["bufferViews"][document["images"][0]["bufferView"]]
    start = view.get("byteOffset", 0)
    texture = np.asarray(Image.open(io.BytesIO(binary[start:start + view["byteLength"]])).convert("RGB")) / 255.0
    corners, triangles, uvs = [], [], []
    for index in document["scenes"][0]["nodes"]:
        node = document["nodes"][index]
        if node.get("mesh") != 0:
            continue
        scale = np.array(node.get("scale", [1.0, 1.0, 1.0]))
        shift = np.array(node.get("translation", [0.0, 0.0, 0.0]))
        corners.append(points * scale + shift)
        triangles.append(faces + sum(len(c) for c in corners[:-1]))
        uvs.append(uv[faces])
    return np.concatenate(corners), np.concatenate(triangles), np.concatenate(uvs), texture


def main(path, out):
    corners, triangles, uvs, texture = placed(path)
    colour_at = render.texture_lookup(uvs, texture)
    extent = float(np.ptp(corners, axis=0).max()) * 1.05
    side = 360
    row = [render.shaded(corners, triangles, t, side, colour_at, extent=extent) for t in range(0, 360, 45)]
    below = [render.shaded(corners, triangles, t, side, colour_at, tilt_deg=-60.0, extent=extent) for t in (0, 90, 180, 270)]
    sheet = Image.new("RGB", (side * 8, side * 2), render.BACKGROUND)
    for k, picture in enumerate(row):
        sheet.paste(picture, (k * side, 0))
    for k, picture in enumerate(below):
        sheet.paste(picture, (k * side, side))
    sheet.save(out)
    print(f"{out}: {len(corners)} corners placed from {path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
