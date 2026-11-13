"""The checks a painted picture passes before it is shaped, each measured from the picture and its cut-out:

* one whole object: the cut-out is one piece (a second thing, or the object in parts, is a second piece), and the
  backdrop is plain (nothing else painted round it);
* no cast shadow and no floor: the backdrop under and beside the object is as light as the backdrop far from it at
  the same heights, and no line runs across the backdrop where a floor would meet a wall;
* not cut off: the object keeps a margin from every edge of the picture.

A picture's score orders the tries that pass: the more of the cut-out in one piece and the plainer the backdrop,
the better. The limits are set from what each fault looks like (a shadow is a darkening of a tenth or more; a floor
line is a step in the backdrop's light), not from any one picture.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

# The cut-out's largest piece holds at least this share of it; a second piece over `LEAST_LISTED_PIECE` is listed.
LEAST_ONENESS = 0.97
LEAST_LISTED_PIECE = 0.003
# The object covers at least this share of the picture (a spear painted upright covers little), at most this.
LEAST_COVER = 0.004
MOST_COVER = 0.8
# Backdrop pixels further than this from the backdrop's middle colour (0-1, the largest channel) are clutter; at
# most this share of the backdrop may be clutter.
CLUTTER_FROM_BACKDROP = 0.18
MOST_CLUTTER = 0.02
# The backdrop straight under the object, against the backdrop beside it at the same heights: at most this much
# darker (a share of its light). A soft cast shadow darkens it by a tenth or more.
MOST_SHADOW_DARKENING = 0.06
# The band under the object that is looked at: this share of the object's height, at least this many pixels.
SHADOW_BAND_SHARE = 0.12
LEAST_SHADOW_BAND = 8
# A floor: a step in the backdrop's light from one row to rows a little lower (rows averaged across the backdrop
# beside the object), at most this much (0-1 light). A seamless backdrop's gradient changes slowly.
MOST_FLOOR_STEP = 0.035
FLOOR_STEP_ROWS = 6
# The object keeps at least this share of the picture's side from every edge.
LEAST_MARGIN = 0.01
# A shadow in the cut-out: where the cut-out's lowest tenth meets a shadow on the backdrop, its edge compared across
# (pixels from 1 to `EDGE_RING` in and out); a step under `NO_EDGE_STEP` (delta E) is the edge running through the
# shadow. From there, the cut-out's pixels coloured like the backdrop (Lab chroma at most `SHADOW_CHROMA_SLACK` over
# it) and darkened as a shadow darkens (from the backdrop check's 6 % to `SHADOW_AT_MOST_DARKENING`; darker is the
# object's own) are the shadow taken in. At most `MOST_SHADOW_IN_CUTOUT` of the cut-out may be.
CUTOUT_BASE_SHARE = 0.1
EDGE_RING = 3
NO_EDGE_STEP = 6.0
LEAST_EDGE_PIXELS = 20
SHADOW_AT_MOST_DARKENING = 0.5
TAKEN_IN_LEAST_DARKENING = -0.06
SHADOW_CHROMA_SLACK = 4.0
MOST_SHADOW_IN_CUTOUT = 0.005
# The backdrop far from the object: the picture's outer columns, this share of its width each side.
FAR_COLUMNS_SHARE = 0.12
FAR_LEAST_PIXELS = 200
# Pixels this close to the object are neither "under" nor "beside" it (the cut-out's soft edge).
EDGE_AWAY = 4


@dataclass
class Check:
    name: str
    value: float
    limit: float
    unit: str
    passed: bool
    words: str
    # A measure shown beside the checks, not counted: it judges nothing that harms the model.
    counted: bool = True

    def as_dict(self):
        return {"name": self.name, "value": round(float(self.value), 4), "limit": self.limit, "unit": self.unit,
                "passed": bool(self.passed), "words": self.words, "counted": self.counted}


@dataclass
class PictureResult:
    checks: list = field(default_factory=list)
    score: float = 0.0

    @property
    def passed(self):
        return all(c.passed for c in self.checks if c.counted)

    def faults(self):
        return [c.words for c in self.checks if c.counted and not c.passed]

    def as_dict(self):
        return {"passed": self.passed, "score": round(self.score, 4), "checks": [c.as_dict() for c in self.checks]}


def light_of(rgb):
    """Each pixel's light (0-1): the mean of its channels."""
    return rgb.astype(np.float64).mean(-1) / 255 if rgb.dtype == np.uint8 else rgb.mean(-1)


def one_object(mask):
    """The share of the cut-out in its largest piece, the other pieces' shares (largest first), and the cover."""
    cover = float(mask.mean())
    labels, count = ndimage.label(mask)
    if count == 0:
        return 0.0, [], cover
    sizes = np.sort(np.bincount(labels.ravel())[1:])[::-1].astype(np.float64)
    shares = sizes / sizes.sum()
    return float(shares[0]), [float(s) for s in shares[1:] if s >= LEAST_LISTED_PIECE], cover


def backdrop_of(rgb, mask):
    """The backdrop (away from the object's soft edge) and its middle colour (0-1)."""
    away = ~ndimage.binary_dilation(mask, iterations=EDGE_AWAY)
    colours = rgb[away].astype(np.float64) / (255 if rgb.dtype == np.uint8 else 1)
    middle = np.median(colours, axis=0) if len(colours) else np.array([0.5, 0.5, 0.5])
    return away, middle


def clutter_share(rgb, mask):
    """The share of the backdrop that is something else: pixels far from the backdrop's colour and more colourful than
    it (in Lab). Its own grey in any light (a shadow, a floor, a vignette's dark corners) is judged where it could
    harm, in the cut-out (`backdrop_in_cutout`). A grey thing beside the object is not counted here; the cut-out's
    "one piece" check finds it if it is cut out with the object."""
    from skimage.color import rgb2lab

    away, middle = backdrop_of(rgb, mask)
    if not away.any():
        return 1.0
    colour = rgb.astype(np.float64) / (255 if rgb.dtype == np.uint8 else 1)
    different = np.abs(colour - middle).max(-1) > CLUTTER_FROM_BACKDROP
    lab = rgb2lab(colour)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    backdrop_tone = chroma <= float(np.median(chroma[away])) + SHADOW_CHROMA_SLACK
    return float((different & ~backdrop_tone)[away].mean())

def far_light_by_row(light, away, width):
    """The backdrop's light far from the object at each height: the median of the picture's outer columns in that row
    (the backdrop's median everywhere when they hold none of it)."""
    outer = max(1, int(FAR_COLUMNS_SHARE * width))
    far = np.zeros_like(away)
    far[:, :outer] = True
    far[:, width - outer:] = True
    far &= away
    height = len(light)
    per_row = np.array([np.median(light[r][far[r]]) if far[r].sum() >= 4 else np.nan for r in range(height)])
    known = np.flatnonzero(np.isfinite(per_row))
    if len(known) == 0:
        return np.full(height, np.median(light[away]) if away.any() else 1.0)
    return np.interp(np.arange(height), known, per_row[known])


def backdrop_in_cutout(rgb, mask):
    """The share of the cut-out that is a cast shadow taken in with the object: inside its lowest tenth, pixels
    coloured like the backdrop (no more colourful in Lab), darkened only as a shadow darkens it, and joined to a
    stretch of the cut-out's lower edge with no colour step across it (the backdrop outside as the cut-out inside).
    The edge is not required to run through a shadow on the backdrop: the check is the stricter one. That is
    what would be meshed as a dark slab under the object or baked into its foot. A shadow the cut-out leaves out harms
    nothing: in the first two trials (212 pictures, 156 with a shadow on the backdrop) none was inside a cut-out by
    eye, and the models made from shadowed pictures (the book, chest, shield, jug) had no slab and no dark foot."""
    rows = np.flatnonzero(mask.any(1))
    if len(rows) == 0:
        return 0.0
    from skimage.color import rgb2lab

    height, width = mask.shape
    colour = rgb.astype(np.float64) / (255 if rgb.dtype == np.uint8 else 1)
    light = colour.mean(-1)
    lab = rgb2lab(colour)
    lab_chroma = np.hypot(lab[..., 1], lab[..., 2])
    away = ~ndimage.binary_dilation(mask, iterations=EDGE_AWAY)
    far = far_light_by_row(light, away, width)[:, None]
    darkened = (far - light) / np.maximum(far, 1e-6)
    backdrop_chroma = float(np.median(lab_chroma[away])) if away.any() else 0.0
    shadow_coloured = (darkened > TAKEN_IN_LEAST_DARKENING) & (darkened < SHADOW_AT_MOST_DARKENING) & \
        (lab_chroma <= backdrop_chroma + SHADOW_CHROMA_SLACK)
    ring_out = ndimage.binary_dilation(mask, iterations=EDGE_RING + 1) & ~ndimage.binary_dilation(mask, iterations=1)
    base_from = rows[-1] - max(1, int(CUTOUT_BASE_SHARE * (rows[-1] - rows[0] + 1)))
    outer = ring_out & ~mask
    outer[:base_from] = False
    inner = ndimage.binary_erosion(mask, iterations=1) & ~ndimage.binary_erosion(mask, iterations=EDGE_RING + 1)
    if outer.sum() < LEAST_EDGE_PIXELS or not inner.any():
        return 0.0
    # Where the edge runs through the shadow: an outer pixel with no colour step to the nearest inner one.
    _, nearest = ndimage.distance_transform_edt(~inner, return_indices=True)
    out_rows, out_cols = np.nonzero(outer)
    in_rows, in_cols = nearest[0][out_rows, out_cols], nearest[1][out_rows, out_cols]
    step = np.linalg.norm(lab[out_rows, out_cols] - lab[in_rows, in_cols], axis=1)
    seeds = np.zeros_like(mask)
    seeds[in_rows[step < NO_EDGE_STEP], in_cols[step < NO_EDGE_STEP]] = True
    # The shadow-coloured pixels inside the cut-out's base joined to those seeds.
    inside = shadow_coloured & mask
    inside[:base_from] = False
    grown, _ = ndimage.label(inside | (seeds & mask))
    taken = np.isin(grown, np.unique(grown[seeds & inside])) & (grown > 0) & inside
    return float(taken.sum() / mask.sum())

def shadow_darkening(rgb, mask):
    """How much darker the backdrop near the object's lower part is than the backdrop far from it at the same height
    (a share of the far backdrop's light). Near it: under it down to the picture's foot, and either side of it from
    its middle down, out to half its width (a shadow cast to one side, or on a floor below an object painted
    floating, lies there); far: the picture's outer columns (the backdrop above the object when it reaches them).
    1 (a fail) when there is no far backdrop to compare with."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if len(rows) == 0:
        return 0.0
    height, width = mask.shape
    top, bottom, left, right = rows[0], rows[-1], cols[0], cols[-1]
    band = max(LEAST_SHADOW_BAND, int(SHADOW_BAND_SHARE * (bottom - top + 1)))
    # Down to the picture's foot: an object painted floating casts its shadow on a floor well below it.
    first, last = (top + bottom) // 2, height
    light = light_of(rgb)
    away = ~ndimage.binary_dilation(mask, iterations=EDGE_AWAY)
    reach = max(band, (right - left + 1) // 2)
    under = np.zeros_like(mask)
    under[bottom + 1:last, left:right + 1] = True
    outer = max(1, int(FAR_COLUMNS_SHARE * width))
    sides = [np.zeros_like(mask), np.zeros_like(mask)]
    sides[0][first:last, max(outer, left - reach):left] = True
    sides[1][first:last, right + 1:min(width - outer, right + 1 + reach)] = True
    far = np.zeros_like(mask)
    far[first:last, :outer] = True
    far[first:last, width - outer:] = True
    far &= away
    if far.sum() < FAR_LEAST_PIXELS:
        # The object reaches the outer columns: the backdrop above it is the far backdrop.
        far = np.zeros_like(mask)
        far[:max(0, top - EDGE_AWAY), :] = True
        far &= away
    if far.sum() < FAR_LEAST_PIXELS:
        return 1.0  # no backdrop to compare with: not shown to be free of shadow
    # The far backdrop's light at each height (a floor lit unevenly from front to back is not a shadow); the backdrop
    # above the object, when that is the far backdrop, is one light for all.
    far_rows = np.flatnonzero(far.any(1))
    if far[first:last].any():
        per_row = np.array([np.median(light[r][far[r]]) if far[r].any() else np.nan for r in range(height)])
        known = np.flatnonzero(np.isfinite(per_row))
        far_light = np.interp(np.arange(height), known, per_row[known])
    else:
        far_light = np.full(height, np.median(light[far_rows][far[far_rows]]))
    # The backdrop's light averaged over patches about the band's size (the object's own pixels left out), so a
    # shadow over part of a region shows and the brush grain does not.
    window = max(LEAST_SHADOW_BAND, band)
    weight = ndimage.uniform_filter(away.astype(np.float64), window)
    patches = ndimage.uniform_filter(light * away, window) / np.maximum(weight, 1e-6)
    near = (under | sides[0] | sides[1]) & away & (weight > 0.5)
    if near.sum() < 10:
        return 0.0
    rows_near = np.nonzero(near)[0]
    darkening = (far_light[rows_near] - patches[near]) / np.maximum(far_light[rows_near], 1e-6)
    return float(max(0.0, darkening.max()))


def floor_step(rgb, mask):
    """The largest step in the backdrop's light down the picture below the object's middle (rows averaged over the
    backdrop beside the object): a floor meeting a wall is a step, a seamless backdrop a slow gradient."""
    rows = np.flatnonzero(mask.any(1))
    if len(rows) == 0:
        return 0.0
    light = light_of(rgb)
    away = ~ndimage.binary_dilation(mask, iterations=EDGE_AWAY)
    start = (rows[0] + rows[-1]) // 2
    means = []
    for r in range(start, mask.shape[0]):
        seen = away[r]
        means.append(light[r, seen].mean() if seen.sum() >= mask.shape[1] // 8 else np.nan)
    means = np.array(means)
    if np.isfinite(means).sum() <= FLOOR_STEP_ROWS:
        return 0.0
    filled = np.where(np.isfinite(means), means, np.nanmedian(means))
    smooth = ndimage.uniform_filter1d(filled, 3)
    steps = np.abs(smooth[FLOOR_STEP_ROWS:] - smooth[:-FLOOR_STEP_ROWS])
    return float(steps.max()) if len(steps) else 0.0


def margin_share(mask):
    """The object's least distance to an edge of the picture, as a share of the picture's shorter side (0 when it
    touches one)."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if len(rows) == 0:
        return 0.0
    height, width = mask.shape
    gaps = [rows[0], height - 1 - rows[-1], cols[0], width - 1 - cols[-1]]
    return float(min(gaps) / min(height, width))


def checked(rgb, mask):
    """Every picture check of a picture (H x W x 3, uint8) and its cut-out (H x W, bool)."""
    result = PictureResult()
    oneness, others, cover = one_object(mask)
    shown = ", ".join(f"{s:.1%}" for s in others[:4])
    result.checks.append(Check(
        "one piece", oneness * 100, LEAST_ONENESS * 100, "%", oneness >= LEAST_ONENESS,
        f"the cut-out's largest piece holds {oneness:.1%} of it" + (f"; other pieces {shown}" if others else "")))
    result.checks.append(Check(
        "object's size in the picture", cover * 100, LEAST_COVER * 100, "%", LEAST_COVER <= cover <= MOST_COVER,
        f"the object covers {cover:.1%} of the picture (from {LEAST_COVER:.1%} to {MOST_COVER:.0%})"))
    clutter = clutter_share(rgb, mask)
    result.checks.append(Check(
        "plain backdrop", clutter * 100, MOST_CLUTTER * 100, "%", clutter <= MOST_CLUTTER,
        f"{clutter:.1%} of the backdrop is something else (another thing, scenery; its own tones in shadow or on a "
        "floor are judged in the cut-out)"))
    taken_in = backdrop_in_cutout(rgb, mask)
    result.checks.append(Check(
        "no shadow or floor in the cut-out", taken_in * 100, MOST_SHADOW_IN_CUTOUT * 100, "%",
        taken_in <= MOST_SHADOW_IN_CUTOUT,
        f"{taken_in:.2%} of the cut-out is backdrop taken in with the object (a shadow or a floor, which would be "
        "meshed as a slab under it)"))
    # Shown, not counted (the second trial, 2026-11-09): a shadow or floor the cut-out leaves out harms nothing.
    shadow = shadow_darkening(rgb, mask)
    result.checks.append(Check(
        "shadow on the backdrop", shadow * 100, MOST_SHADOW_DARKENING * 100, "%", shadow <= MOST_SHADOW_DARKENING,
        f"the backdrop under and beside the object is {shadow:.1%} darker than far from it (shown; the cut-out "
        "leaves it out)", counted=False))
    step = floor_step(rgb, mask)
    result.checks.append(Check(
        "floor line on the backdrop", step * 100, MOST_FLOOR_STEP * 100, "% light", step <= MOST_FLOOR_STEP,
        f"the backdrop's light steps by {step:.1%} at most below the object's middle (shown; the cut-out leaves it "
        "out)", counted=False))
    margin = margin_share(mask)
    result.checks.append(Check(
        "not cut off", margin * 100, LEAST_MARGIN * 100, "%", margin >= LEAST_MARGIN,
        "touches the picture's edge" if margin == 0 else f"{margin:.1%} of the picture from the nearest edge"))
    result.score = oneness * (1 - clutter) * (1 - shadow) * min(1.0, margin / LEAST_MARGIN)
    return result


# A shown measure that ranks the tries that pass (never refuses one): a hafted weapon standing upright (upright).
RANKING_CHECKS = ("standing upright",)


def ranked_well(result):
    """Whether a try passes the shown measures that rank tries (`RANKING_CHECKS`); True when it has none."""
    return all(c.passed for c in result.checks if c.name in RANKING_CHECKS)


def best_try(results):
    """The index of the best try: the passing ones first, then those the ranking measures prefer (a hafted weapon
    painted upright), then by score; the first of equals."""
    if not results:
        return None
    return max(range(len(results)), key=lambda k: (results[k].passed, ranked_well(results[k]), results[k].score, -k))
