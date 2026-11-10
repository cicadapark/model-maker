"""How like its picture the painted object looks: the object drawn in its baked colours (unlit: the picture is lit
evenly) from the view its outline matches best, laid over the picture at the same size, and the colours compared
where both are: the mean colour difference (CIE76 delta E in Lab; about 2 is barely seen, 10 plainly) and the share
of pixels within 10. A measure for comparing ways of painting, not a check: no limit is set until one is measured.
"""

import numpy as np
from PIL import Image

from . import render
from .cutout import for_shape
from .object_checks import OUTLINE_CANVAS_ASPECT, OUTLINE_SIDE, canvas_aspect, placed, render_side

PLAIN_DIFFERENCE = 10.0


def unlit(corners, triangles, turn_deg, side, colour_at, tilt_deg, distance=None):
    """The object's own colours (no light) and where it is, seen from a turn (through a lens `distance` sizes away)."""
    framing = render.Framing(corners, turn_deg, side, tilt_deg, distance=distance)
    xy, depth = framing.pixels(corners)
    _, owner, bary = render.rasterised(xy, depth, triangles, side, side)
    hit = owner >= 0
    image = np.zeros((side, side, 3))
    image[hit] = colour_at(owner[hit], bary[hit])
    return image, hit


def framed(rgb, mask, aspect=OUTLINE_CANVAS_ASPECT):
    """The picture framed as the outline check frames its outline (`object_checks.placed`): cropped to the mask,
    scaled to `OUTLINE_SIDE` high, its area's middle centred on a canvas `aspect` heights wide, the outline's own, so
    a wide object is cut in neither or in both."""
    canvas_mask, _, (_, _, _, _, left) = placed(mask, aspect)
    wide = canvas_mask.shape[1]
    canvas_rgb = np.zeros((OUTLINE_SIDE, wide, 3))
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if len(rows) == 0:
        return canvas_rgb, canvas_mask
    crop = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
    width = max(1, round((cols[-1] - cols[0] + 1) * OUTLINE_SIDE / (rows[-1] - rows[0] + 1)))
    small_rgb = np.asarray(Image.fromarray((np.clip(rgb[crop], 0, 1) * 255).astype(np.uint8)).resize(
        (width, OUTLINE_SIDE), Image.Resampling.BILINEAR)) / 255.0
    span = slice(max(0, left), min(wide, left + width))
    if span.stop > span.start:
        canvas_rgb[:, span] = small_rgb[:, span.start - left:span.stop - left]
    return canvas_rgb, canvas_mask


def likeness(picture_rgb, picture_mask, corners, triangles, colour_at, turn_deg, tilt_deg, distance=None):
    """(mean colour difference, the share within `PLAIN_DIFFERENCE`, a picture: the picture, the painted object and
    their difference side by side)."""
    from skimage.color import deltaE_cie76, rgb2lab

    # The cut-out the outline check judged by, so the picture is framed as when its view was found.
    picture_mask = for_shape(picture_mask)
    aspect = canvas_aspect(picture_mask)
    drawn, hit = unlit(corners, triangles, turn_deg, render_side(aspect), colour_at, tilt_deg, distance)
    painted_rgb, painted_mask = framed(picture_rgb.astype(np.float64) / 255.0, picture_mask, aspect)
    made_rgb, made_mask = framed(drawn, hit, aspect)
    both = painted_mask & made_mask
    if not both.any():
        return float("nan"), 0.0, None
    difference = deltaE_cie76(rgb2lab(painted_rgb), rgb2lab(made_rgb))
    mean = float(difference[both].mean())
    within = float((difference[both] <= PLAIN_DIFFERENCE).mean())
    shown = np.clip(difference / (3 * PLAIN_DIFFERENCE), 0, 1)
    heat = np.stack([shown, 1 - shown, np.zeros_like(shown)], -1) * both[..., None]
    # Shown cropped to the columns either uses, a few pixels to spare.
    used = np.flatnonzero((painted_mask | made_mask).any(0))
    middle = painted_mask.shape[1] // 2
    half = max(OUTLINE_SIDE, middle - int(used[0]) + 8, int(used[-1]) + 9 - middle)
    columns = slice(max(0, middle - half), middle + half)
    sheet = np.concatenate([painted_rgb[:, columns], made_rgb[:, columns], heat[:, columns]], 1)
    return mean, within, Image.fromarray((sheet * 255).astype(np.uint8))
