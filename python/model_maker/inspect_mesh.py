"""The object checks on a mesh made elsewhere (any format trimesh reads), against the picture it was made from: the
same report as a made object's, without the paint. Runs on the CPU.
"""

import json
import os

import numpy as np
from PIL import Image

from . import inspection, object_checks, picture_checks, render, words
from . import report as pages

# A pixel further than this from the backdrop's colour (the picture's border, 0-255 largest channel) is the object,
# when no cut-out is given and the picture has no transparency.
BACKDROP_WITHIN = 24
# `--recut` (for a model made on another computer): the picture is cut again by our own BiRefNet on the CPU and
# every check is judged against our cut-out, never the one sent, so a cut-out made to fit the model (a web between legs
# left out) cannot pass the outline. The one sent must also be ours (their overlap over their union): on the trials'
# pictures our CPU cut and the card's agreed to at least LEAST_CUTOUT_AGREEMENT's margin (docs/objects.md).
CUTOUT_AGREEMENT = "the cut-out sent is the picture's"
LEAST_CUTOUT_AGREEMENT = 0.97


def backdrop_cut(rgba):
    """The object told from a plain backdrop by colour: its transparency when it has some, else every pixel far from
    the border's middle colour."""
    from scipy import ndimage

    if rgba[..., 3].min() < 250:
        return rgba[..., 3] > 127
    rgb = rgba[..., :3].astype(np.int32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    backdrop = np.median(border, axis=0)
    mask = np.abs(rgb - backdrop).max(-1) > BACKDROP_WITHIN
    return ndimage.binary_opening(mask, iterations=2)


def loaded_mesh(path):
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    corners, triangles = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)
    from .mesh_tools import welded

    # Written per triangle or split at texture seams, a surface is welded again before its pieces are counted.
    extent = float(np.ptp(corners, axis=0).max())
    return welded(corners, triangles, extent * 1e-5)


def our_cut(picture):
    """The picture cut out by our own BiRefNet on the CPU (its folder under MODEL_MAKER_MODELS, checked against the
    registry first; registry.ModelMissing when it is not downloaded)."""
    from . import cutout, registry

    folder = registry.model_folder("birefnet", "birefnet")
    return cutout.Cutter(folder, device="cpu").mask(picture)


def agreement_check(sent, ours):
    """The cut-out sent against ours: their overlap over their union."""
    if sent.shape != ours.shape:
        return picture_checks.Check(CUTOUT_AGREEMENT, 0.0, LEAST_CUTOUT_AGREEMENT * 100, "%", False,
                                    f"the cut-out sent is {sent.shape[1]}x{sent.shape[0]}, the picture "
                                    f"{ours.shape[1]}x{ours.shape[0]}")
    union = np.logical_or(sent, ours).sum()
    agreement = float(np.logical_and(sent, ours).sum() / union) if union else 0.0
    return picture_checks.Check(
        CUTOUT_AGREEMENT, agreement * 100, LEAST_CUTOUT_AGREEMENT * 100, "%", agreement >= LEAST_CUTOUT_AGREEMENT,
        f"the cut-out sent and our own cut of the picture overlap {agreement:.1%} of their union (at least "
        f"{LEAST_CUTOUT_AGREEMENT:.0%}); every check is judged against ours")


def inspected(mesh_path, picture_path, prompt, out, size_m=None, mask_path=None, recut=False):
    subject = words.subject_of(prompt)
    picture = Image.open(picture_path).convert("RGBA")
    sent = (np.asarray(Image.open(mask_path).convert("L")) > 127) if mask_path else None
    agreement = None
    if recut:
        mask = our_cut(picture)
        if sent is not None:
            agreement = agreement_check(sent, mask)
    else:
        mask = sent if sent is not None else backdrop_cut(np.asarray(picture))
    corners, triangles = loaded_mesh(mesh_path)
    corners = object_checks.set_down(corners, size_m or subject.size_m, along_length=subject.is_weapon)
    picture_result = picture_checks.checked(np.asarray(picture.convert("RGB")), mask)
    if agreement is not None:
        picture_result.checks.insert(0, agreement)
    checks, measures, overlay = inspection.shape_checks(subject, mask, corners, triangles,
                                                        picture=np.asarray(picture.convert("RGB")))
    grouped = {"picture": [c.as_dict() for c in picture_result.checks], "shape": [c.as_dict() for c in checks]}
    every = [c for group in grouped.values() for c in group]
    every = [c for c in every if c.get("counted", True)]
    status = "passes every check" if all(c["passed"] for c in every) else \
        "fails: " + "; ".join(c["name"] for c in every if not c["passed"])
    job = {"prompt": subject.prompt, "slug": subject.slug, "subject": subject.__dict__,
           "size_m": size_m or subject.size_m, "seconds": {}, "status": status}
    pictures = {"picture.png": (pages.thumbnail(picture.convert("RGB"), 360), "the picture it was made from"),
                "picture-cut.png": (pages.thumbnail(pages.mask_picture(mask), 360), "its cut-out"),
                "outline.png": (overlay, "outlines: green both, red the picture's alone, blue the model's alone"),
                "bare.png": (render.turntable(corners, triangles), "the shape, turned in steps of 45° from the front")}
    if measures.get("weapon"):
        pictures["marks.png"] = (pages.drawn_marks(corners, triangles, None, measures),
                                 "the grip (blue line, the ring where the hand goes) and the tip (red)")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "checks.json"), "w") as file:
        json.dump({"prompt": subject.prompt, "mesh": os.path.basename(mesh_path), "status": status,
                   "checks": grouped, "measures": measures}, file, indent=1, default=float)
    return pages.written(out, job, grouped, measures, pictures, [])
