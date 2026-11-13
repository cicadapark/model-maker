"""The paint touched up: the places the six painted views did not reach, painted from views chosen to see them.

MV-Adapter paints the object from its front, sides, back, above and below. A place seen only at a slant from above
(the inside of a candle holder's bowl, the candle's foot within it: the fix-check's candle, 2026-11-10) is in none
of the six squarely, so it takes colour spread from its neighbours and the paint's coverage fails. Players see those
places: a candle on a table is looked down on.

The established fix, as MV-Adapter's own texturing does (its paper, section 5.4: "view coverage analysis to identify
uncovered regions", the current texture rendered from those views and the gaps inpainted) and LumiTex chooses its
extra views (greedily, by the uncovered area each sees): candidate views round the object and above it are ranked by
how much of the unpainted surface they see squarely, weighted as the coverage check weights it (by how often each
place is seen); the best is taken, the object drawn from it in its colours so far, the unpainted places in it painted
by the picture model's sibling (SDXL inpainting from the base model's own weights, in the registry already), and the
new colours put back onto the surface; then the next, until the views left would add little. Each fill sees the
paint round it, so it continues it. A place no view sees squarely is left to the spread, and counts as not covered.

The fill is judged before it is kept (the touch-up's first run on the card, 2026-11-10: the candle's dish came back
with dark holes and pale grey, mirror-like patches, and every check passed, as the coverage counts a texel painted,
not painted right). The unpainted places are not painted from noise: they start from the colour spread to them from
their painted neighbours and are repainted only partly (TEXTure's and Text2Tex's "refine" or "update" regions, which
diffuse a partly noised start; a fill from pure noise is their "generate" region, for places nothing is known of).
The picture the inpainter sees is drawn from the painted colours and the earlier views' fills. Each filled pixel's
colour is then measured against the painted points nearest it on the same surface (`misfit`); a view whose fill has
colours found nowhere round it is painted again from another seed, the better try kept, and its pixels still foreign
(and a little round them) are left out: they keep the spread colour and count as unpainted, so the coverage stays
true. A view whose kept try was, as painted, more foreign than the fill check allows is painted again from more seeds.

The finish judges only what the model shows (`inspection.fill_check`). The fill kept is judged as kept, by the same
measure (`fill_fit`). A place left out never shows the inpainter's try: it shows the colour spread to it from round
it, which looks right when the paint round it is one colour (a lining spread from the boot's leather is the leather's
brown) and is a smear when it is not (the candle's foot spread from its wax and its brass dish): a large place is
judged by its ring (`left_out_places`). The fill as painted is shown beside them, not counted: an unseen place need
not look like its surroundings (a lining, a shelf's boards, a leg's inner face), so a try's colours found nowhere round
it are no fault of the model, and what was foreign never reached it.
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from . import bake, render
from . import mesh_tools as M

# The candidate views: round the object every 45 degrees, from these heights above it (the six views already see it
# from the sides, straight above and below; what they miss is seen from above at a slant).
CANDIDATE_TURNS = tuple(range(0, 360, 45))
CANDIDATE_TILTS = (20.0, 35.0, 50.0)
# At most this many views are added, each while it adds at least this share of the surface as it is seen (a box seen
# whole by the six views gains 0.6 % from its bevelled corners, the fix-check's candle misses 14 %).
MOST_VIEWS = 4
LEAST_GAIN = 0.01
SIDE = 768
# The surface sampled to find what is unpainted.
SAMPLES = 120000
# A view paints a place it sees at least this squarely (as the painted views' colour cloud is kept).
LEAST_FACING = 0.55
# SDXL's pipeline shrinks the mask to its latents, one cell for 8 by 8 pixels, the cell's first pixel deciding
# (diffusers' nearest interpolation): a masked pixel in a cell left unmasked comes back as it went in. The first run
# counted 3-5 % of the candle's filled pixels and 17-23 % of the chair's so, its spread colour taken as paint. The mask
# is grown by a cell, so every unpainted pixel's cell is masked, and only pixels in a masked cell are kept.
LATENT_CELL = 8
MASK_GROW = LATENT_CELL
BACKDROP = 0.5
# How much of the start the inpainter repaints (1.0: from noise, the start ignored). The first run's 0.99 painted the
# dish's places as if nothing were there: holes, the grey backdrop carried in over the rim, mirror-like streaks.
STRENGTH = 0.6
# The words given beside the object's own, and those it is steered from (beside the view painter's).
FILL_WORDS = "the same surface and material continued, even light"
FILL_NEGATIVE = "hole, opening, gap, dark spot, stain, reflection, mirror, glare, highlight, background"
# The fill's judgement. A filled point's colour is compared with its nearest painted points on the same surface: found
# by position and facing together (the facing weighted this share of the object's size, so a thin wall's other side,
# a few millimetres off, is not taken for its ring), those facing within about 75 degrees of it kept (as the bake).
RING_POINTS = 32
RING_FACING_WEIGHT = 0.05
# The painted points are thinned to every this many first: the candle's 1.02 million make the search a minute; a
# quarter of them, 32 nearest (as far round as 128 of all), agree on 99 % of its filled points and take 3 s.
RING_STRIDE = 4
# Its colour is foreign when it is more than this far (CIE76 delta E; 10 is plainly seen) from every one of them. A point
# none of them faces the same way as (a wall's inside whose like-facing paint is further off) is judged against them all.
FOREIGN_DELTA_E = 20.0
# The touch-up keeps a filled pixel only when it is within this (stricter than the check, so what it keeps is clear of
# the check's limit, not at it).
KEEP_DELTA_E = 15.0
# A view's fill is tried this many times while more than `GOOD_ENOUGH` of it is foreign (each try its own seed), and
# up to `MOST_FILL_TRIES` while the best try, as painted, is more than `MOST_FOREIGN` foreign (see `touched_up`).
FILL_TRIES = 2
MOST_FILL_TRIES = 4
GOOD_ENOUGH = 0.02
# Foreign pixels, and these many round them (a blob's soft edge), are left out.
LEAVE_OUT_GROW = 2
# The fill check: no touch-up view's fill may be more than this share foreign, and no foreign place joined in a view
# larger than a square of this share of the object's size. The first run's candle: 18 to 72 % a view foreign, its
# largest places 7 to 21 %; its chair, whose fill looks like its wood: 2.4 and 4.8 %, 1.4 and 1.5 %.
MOST_FOREIGN = 0.10
LARGEST_FOREIGN_PATCH = 0.03
JUDGED_IN_PIECES = 20000
# A place a view left unpainted (its fill left out, or never painted) shows on the model the colour spread to it from
# round it. Its ring is the painted pixels within this many of it (half a latent cell), those on the surface within
# this many ring widths of it (a slanted surface's pixels lie further apart than a pixel).
RING_PIXELS = 4
RING_REACH = 3.0
# Colours are one colour when at least this share of them lies within `FOREIGN_DELTA_E` of their median. A place
# larger than `LARGEST_FOREIGN_PATCH` is a smear when neither its ring nor what the model shows in it is one colour.
# Measured on the trials' places (docs/objects.md, "The fill judged on the model"): those smeared by eye (a candle's
# foot spread from its wax and its dish, a chair's stretcher spread from grey and brown) read 0.61 to 0.82 for their
# ring and 0.69 to 0.86 inside; those whose ring holds two colours but which show one by eye (a chair's post spread
# brown from its brown and grey ring, a shield's handle) 0.97 to 1.00 inside.
ONE_COLOUR = 0.9


def _cloud_near(points, normals, cloud):
    """Each point's painted neighbours in the colour cloud, as the bake finds them: (distances, indices, which count)."""
    cloud_points, cloud_normals = (np.asarray(c, dtype=np.float64) for c in cloud[:2])
    reach = bake.CLOUD_REACH_SPACINGS * bake.cloud_spacing(cloud_points)
    distances, nearest = cKDTree(cloud_points).query(points, k=bake.CLOUD_NEIGHBOURS, distance_upper_bound=reach)
    found = np.isfinite(distances)
    safe = np.where(found, nearest, 0)
    facing = np.einsum("pd,pkd->pk", normals, cloud_normals[safe]) > bake.SAME_FACING
    return distances, safe, found & facing


def unpainted(points, normals, cloud):
    """Which points the colour cloud does not reach, as the bake judges a texel."""
    if len(cloud[0]) == 0:
        return np.ones(len(points), dtype=bool)
    return ~_cloud_near(points, normals, cloud)[2].any(1)


def cloud_coloured(points, normals, cloud):
    """Each point's colour from the colour cloud as the bake mixes it, and which points it reaches."""
    colours = np.zeros((len(points), 3))
    if len(cloud[0]) == 0:
        return colours, np.zeros(len(points), dtype=bool)
    distances, nearest, use = _cloud_near(points, normals, cloud)
    weight = np.where(use, 1.0 / (np.where(use, distances, 1.0) + 1e-9), 0.0)
    total = weight.sum(1)
    reached = total > 0
    mixed = np.einsum("pk,pkc->pc", weight, np.asarray(cloud[2], dtype=np.float64)[nearest])
    colours[reached] = mixed[reached] / total[reached, None]
    return colours, reached


def framing(corners, turn, tilt):
    extent = float(np.ptp(corners, axis=0).max())
    return render.Framing(corners, turn, SIDE, tilt, extent=extent * 1.15)


def squarely_seen(corners, triangles, points, normals, turn, tilt):
    """Which points a view sees, nearest it and facing it at least `LEAST_FACING`."""
    frame = framing(corners, turn, tilt)
    xy, depth = frame.pixels(corners)
    zbuf, _, _ = render.rasterised(xy, depth, triangles, SIDE, SIDE)
    point_xy, point_depth = frame.pixels(points)
    col = np.clip(point_xy[:, 0].astype(int), 0, SIDE - 1)
    row = np.clip(point_xy[:, 1].astype(int), 0, SIDE - 1)
    toward = render.view_matrix(turn, tilt)[2]
    return (normals @ toward >= LEAST_FACING) & (point_depth >= zbuf[row, col] - bake.SEEN_SLACK_PIXELS / frame.scale)


def chosen_views(corners, triangles, cloud, most=MOST_VIEWS):
    """The views to add, best first: [(turn, tilt, the share of the seen surface each adds)], and the seen share of
    the surface unpainted before them."""
    points, normals, _ = M.sampled(corners, triangles, SAMPLES, seed=11)
    weight = bake.seen_share(corners, triangles, points, normals)
    todo = unpainted(points, normals, cloud)
    total = max(float(weight.sum()), 1e-12)
    before = float((weight * todo).sum() / total)
    seen = {(turn, tilt): squarely_seen(corners, triangles, points, normals, turn, tilt)
            for turn in CANDIDATE_TURNS for tilt in CANDIDATE_TILTS}
    views = []
    while len(views) < most:
        gains = {view: float((weight * (todo & got)).sum() / total) for view, got in seen.items()}
        best = max(gains, key=gains.get)
        if gains[best] < LEAST_GAIN:
            break
        views.append((best[0], best[1], gains[best]))
        todo &= ~seen[best]
    return views, before


def latent_masked(mask):
    """The pixels SDXL's pipeline repaints for a pixel mask: those in a latent cell whose first pixel is masked."""
    cells = mask[::LATENT_CELL, ::LATENT_CELL]
    whole = np.kron(cells, np.ones((LATENT_CELL, LATENT_CELL), dtype=bool))
    return whole[:mask.shape[0], :mask.shape[1]]


def grown(wanted, hit):
    """The mask given to the inpainter: the unpainted pixels grown by a latent cell, on the object only."""
    return ndimage.binary_dilation(wanted, structure=np.ones((3, 3), dtype=bool), iterations=MASK_GROW) & hit


class Seen:
    """The object from a touch-up view in its colours so far (`seen_from`): which pixels it covers (`hit`), each
    pixel's surface point and normal, its colour (the colour cloud's, as the bake mixes it, where the cloud reaches:
    `reached`; the colours spread to the corners elsewhere), the pixels it sees squarely that the cloud does not reach
    (`wanted`), and its framing."""

    def __init__(self, frame, hit, surface, facing, colour, reached, wanted):
        self.frame, self.hit, self.surface, self.facing = frame, hit, surface, facing
        self.colour, self.reached, self.wanted = colour, reached, wanted


def seen_from(corners, triangles, colours, cloud, turn, tilt):
    frame = framing(corners, turn, tilt)
    xy, depth = frame.pixels(corners)
    _, owner, bary = render.rasterised(xy, depth, triangles, SIDE, SIDE)
    hit = owner >= 0
    face_normals, _ = M.face_normals_and_areas(corners, triangles)
    tri = triangles[owner[hit]]
    points = np.einsum("pk,pkd->pd", bary[hit], corners[tri])
    normals = face_normals[owner[hit]]
    shown = np.einsum("pk,pkc->pc", bary[hit], colours[tri])
    painted, reached = cloud_coloured(points, normals, cloud)
    shown[reached] = painted[reached]
    toward = render.view_matrix(turn, tilt)[2]
    surface, facing = np.zeros((SIDE, SIDE, 3)), np.zeros((SIDE, SIDE, 3))
    surface[hit], facing[hit] = points, normals
    colour = np.full((SIDE, SIDE, 3), BACKDROP)
    colour[hit] = shown
    reached_pixels, wanted = np.zeros((SIDE, SIDE), dtype=bool), np.zeros((SIDE, SIDE), dtype=bool)
    reached_pixels[hit] = reached
    wanted[hit] = ~reached & (normals @ toward >= LEAST_FACING)
    return Seen(frame, hit, surface, facing, colour, reached_pixels, wanted)


def drawn(corners, triangles, colours, cloud, turn, tilt):
    """The object from a view in its colours so far: the colour cloud's (the six views' paint and the fills before
    this view) where it reaches, the colours spread to the corners elsewhere. Returns (picture, the unpainted pixels
    this view sees squarely, the mask to paint, each pixel's surface point and normal)."""
    seen = seen_from(corners, triangles, colours, cloud, turn, tilt)
    mask = grown(seen.wanted, seen.hit)
    picture = Image.fromarray((np.clip(seen.colour, 0, 1) * 255).round().astype(np.uint8))
    return picture, seen.wanted, mask, seen.surface, seen.facing


def _lab(rgb):
    from skimage.color import rgb2lab

    return rgb2lab(np.clip(np.asarray(rgb, dtype=np.float64), 0, 1)[None])[0]


def misfit(points, normals, colours, ring):
    """Each point's least colour difference (delta E) from the `RING_POINTS` painted points of `ring` (points,
    normals, colours) nearest it on the same surface (from them all where none faces its way); infinite with no ring."""
    ring = tuple(np.asarray(c)[::RING_STRIDE] for c in ring)
    ring_points, ring_normals = (np.asarray(c, dtype=np.float64) for c in ring[:2])
    least = np.full(len(points), np.inf)
    if len(points) == 0 or len(ring_points) == 0:
        return least
    size = float(np.ptp(ring_points, axis=0).max())
    tree = cKDTree(np.concatenate([ring_points, ring_normals * RING_FACING_WEIGHT * size], 1))
    ring_lab = _lab(ring[2])
    points, normals, lab = np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64), _lab(colours)
    count = min(RING_POINTS, len(ring_points))
    for start in range(0, len(points), JUDGED_IN_PIECES):
        piece = slice(start, start + JUDGED_IN_PIECES)
        _, nearest = tree.query(np.concatenate([points[piece], normals[piece] * RING_FACING_WEIGHT * size], 1),
                                k=count)
        nearest = nearest.reshape(len(points[piece]), count)
        same = np.einsum("pd,pkd->pk", normals[piece], ring_normals[nearest]) > bake.SAME_FACING
        apart = np.linalg.norm(lab[piece, None] - ring_lab[nearest], axis=2)
        facing_its_way = np.where(same, apart, np.inf).min(1)
        least[piece] = np.where(np.isfinite(facing_its_way), facing_its_way, apart.min(1))
    return least


def fill_sources(painted, record):
    """Where each colour cloud point came from: 0 the six views, k the touch-up record's view k (counted from 1). A
    cloud saved before this was marked is split by the touch-up's record (each view's points were added after the six
    views', in order). In a cloud numbered among the views chosen, where a chosen view with nothing to paint was
    skipped without a record entry, the numbers are put in the record's order (the views that
    kept points, in order, are the numbers found, in order). None when there is no record of a fill."""
    views = (record or {}).get("views") or []
    if "cloud_from" in getattr(painted, "files", painted):
        sources = np.asarray(painted["cloud_from"]).astype(np.int16)
        if not views or all("source" in v for v in views):
            return sources
        found = [int(k) for k in np.unique(sources) if k > 0]
        keeping = [k + 1 for k, view in enumerate(views) if view["pixels"] > 0]
        if len(found) != len(keeping):
            raise ValueError(f"the touch-up's record names {len(keeping)} views with points, the cloud {len(found)}")
        renumbered = np.zeros_like(sources)
        for old, new in zip(found, keeping):
            renumbered[sources == old] = new
        return renumbered
    if not views:
        return None
    count = len(painted["cloud_points"])
    sources = np.zeros(count, dtype=np.int16)
    start = count - sum(v["pixels"] for v in views)
    for k, view in enumerate(views):
        sources[start:start + view["pixels"]] = k + 1
        start += view["pixels"]
    return sources


def as_painted(record):
    """Each touch-up view's fill as it was painted, before what was foreign was left out, as the touch-up judged it
    for the finish's check: (shares foreign at `FOREIGN_DELTA_E`, largest foreign places); None when the record
    predates it (`replayed_as_painted` measures it again)."""
    views = (record or {}).get("views") or []
    if not views or not all("as_painted" in v for v in views):
        return None
    return [v["as_painted"]["foreign"] for v in views], [v["as_painted"]["patch"] for v in views]


def replayed_as_painted(corners, triangles, painted, record, views_dir):
    """`as_painted` for a judged record made before the touch-up kept it: each view's kept try (`touch-up-K.png`,
    saved by the touch-up) judged again against the six views' paint, its pixels drawn again from the colour cloud as
    it stood before that view (the six views' and the earlier views' kept points: the views chosen and their pixels
    depend only on those, so the drawing is the one the touch-up made). None when a try's picture is missing."""
    import os

    sources = fill_sources(painted, record)
    points, normals, colours = (np.asarray(painted[k], dtype=np.float64)
                                for k in ("cloud_points", "cloud_normals", "cloud_colours"))
    ring = (points[sources == 0], normals[sources == 0], colours[sources == 0])
    pictures = sorted((name for name in os.listdir(views_dir) if name.startswith("touch-up-")
                       and not name.endswith("-mask.png")), key=lambda name: int(name[len("touch-up-"):-len(".png")]))
    if len(pictures) != len(record["views"]):
        return None
    shares, patches = [], []
    for number, (view, name) in enumerate(zip(record["views"], pictures), start=1):
        before = sources < number
        cloud = (points[before], normals[before], colours[before])
        _, wanted, mask, surface, facing = drawn(corners, triangles, np.asarray(painted["colours"], dtype=np.float64),
                                                 cloud, view["turn_deg"], view["tilt_deg"])
        wanted &= latent_masked(mask)
        try_picture = np.asarray(Image.open(os.path.join(views_dir, name)).convert("RGB")).astype(np.float64) / 255
        _, _, checked = judged_fill(try_picture, wanted, surface, facing, ring)
        shares.append(checked["foreign"])
        patches.append(checked["patch"])
    return shares, patches


def object_pixels():
    """The object's largest size across a touch-up view, in pixels (its framing's fill of the side)."""
    return 0.86 * SIDE / 1.15


def largest_patch(points, foreign, corners, turn, tilt):
    """The largest joined foreign place among one view's filled points, drawn back into that view's pixels: the side
    of a square of its area, as a share of the object's size (the first run's candle: its holes and grey patches 7 to
    21 %; foreign pixels strung along a thin leg's edge are a few pixels wide and stay small: its chair, 1.4 and
    1.5 %)."""
    if not foreign.any():
        return 0.0
    xy, _ = framing(corners, turn, tilt).pixels(points[foreign])
    cols = np.clip(np.round(xy[:, 0] - 0.5).astype(int), 0, SIDE - 1)
    rows = np.clip(np.round(xy[:, 1] - 0.5).astype(int), 0, SIDE - 1)
    grid = np.zeros((SIDE, SIDE), dtype=bool)
    grid[rows, cols] = True
    labels, count = ndimage.label(grid, structure=np.ones((3, 3), dtype=bool))
    biggest = int(np.bincount(labels.ravel())[1:].max()) if count else 0
    return float(np.sqrt(biggest) / object_pixels())


def filled_misfit(points, normals, colours, sources):
    """Each filled point's `misfit` against the six views' paint (sources 0), in the order of the filled points."""
    sources = np.asarray(sources)
    six = sources == 0
    return misfit(points[~six], normals[~six], colours[~six], (points[six], normals[six], colours[six]))


def fill_fit(points, normals, colours, sources, corners=None, views=None, least=None):
    """How well the touch-up's fill matches the paint round it: each view's share of filled points whose colour is
    foreign (`misfit` above `FOREIGN_DELTA_E` against the six views' paint), the worst view's and all together; and,
    given the shape's corners and the touch-up's views (turn_deg, tilt_deg), each view's largest foreign place
    (`largest_patch`) and the largest. `least`: `filled_misfit`, when it is measured already."""
    sources = np.asarray(sources)
    six = sources == 0
    found = {"views": [], "worst": 0.0, "all": 0.0, "patches": [], "largest_patch": 0.0,
             "filled_points": int((~six).sum()), "delta_e": FOREIGN_DELTA_E}
    if not (~six).any():
        return found
    if least is None:
        least = filled_misfit(points, normals, colours, sources)
    foreign = least > FOREIGN_DELTA_E
    filled_from = sources[~six]
    for k in range(1, max(int(filled_from.max()), len(views or ())) + 1):
        mine = filled_from == k
        found["views"].append(round(float(foreign[mine].mean()), 4) if mine.any() else 0.0)
        if corners is not None and views is not None and k <= len(views):
            found["patches"].append(round(largest_patch(points[~six][mine], foreign[mine], corners,
                                                        views[k - 1]["turn_deg"], views[k - 1]["tilt_deg"]), 4))
    found["worst"] = max(found["views"])
    found["largest_patch"] = max(found["patches"], default=0.0)
    found["all"] = round(float(foreign.mean()), 4)
    return found


def judged_fill(painted, wanted, surface, facing, ring):
    """A view's fill judged: (its share foreign at `KEEP_DELTA_E`, the pixels to leave out, and the fill as the
    finish's check judges it: its share foreign at `FOREIGN_DELTA_E` and its largest joined foreign place, a share of
    the object's size)."""
    least = np.full(wanted.shape, 0.0)
    least[wanted] = misfit(surface[wanted], facing[wanted], painted[wanted], ring)
    foreign = wanted & (least > KEEP_DELTA_E)
    share = float(foreign.sum() / max(int(wanted.sum()), 1))
    left_out = ndimage.binary_dilation(foreign, iterations=LEAVE_OUT_GROW) & wanted if foreign.any() else foreign
    checked = wanted & (least > FOREIGN_DELTA_E)
    return share, left_out, {"foreign": round(float(checked.sum() / max(int(wanted.sum()), 1)), 4),
                             "patch": round(largest_place(checked), 4)}


def largest_place(foreign):
    """The largest joined place of a view's foreign pixels: the side of a square of its area, as a share of the
    object's size across the view."""
    if not foreign.any():
        return 0.0
    labels, count = ndimage.label(foreign, structure=np.ones((3, 3), dtype=bool))
    return float(np.sqrt(np.bincount(labels.ravel())[1:].max()) / object_pixels())


def ring_of(place, seen):
    """The painted pixels round a place in a view: within `RING_PIXELS` of it in the picture and, on the surface,
    within `RING_REACH` ring widths of it (a nearer or farther surface beside it in the picture is not its ring)."""
    near = ndimage.binary_dilation(place, structure=np.ones((3, 3), dtype=bool), iterations=RING_PIXELS)
    ring = near & ~place & seen.hit & seen.reached
    if not ring.any():
        return ring
    reach = RING_REACH * RING_PIXELS / seen.frame.scale
    distance, _ = cKDTree(seen.surface[place]).query(seen.surface[ring], distance_upper_bound=reach)
    rows, cols = np.nonzero(ring)
    ring[rows[~np.isfinite(distance)], cols[~np.isfinite(distance)]] = False
    return ring


def one_colour_share(colours):
    """How much of a ring is one colour: the share of its colours within `FOREIGN_DELTA_E` of their median (in CIE
    Lab); 0 for no ring."""
    if len(colours) == 0:
        return 0.0
    lab = _lab(colours)
    return float((np.linalg.norm(lab - np.median(lab, 0), axis=1) <= FOREIGN_DELTA_E).mean())


def judged_places(seen):
    """The places a view sees squarely that the colour cloud does not reach, larger than `LARGEST_FOREIGN_PATCH` of
    the object's size, largest first: [(their pixels, size, the share of their ring that is one colour, the share of
    what the model shows in them that is, the ring's pixel count)], and the largest place's size (any size)."""
    labels, count = ndimage.label(seen.wanted, structure=np.ones((3, 3), dtype=bool))
    if not count:
        return [], 0.0
    sizes = np.sqrt(np.bincount(labels.ravel())[1:]) / object_pixels()
    places = []
    for label in np.argsort(-sizes):
        if sizes[label] <= LARGEST_FOREIGN_PATCH:
            break
        place = labels == label + 1
        ring = ring_of(place, seen)
        places.append((place, float(sizes[label]), one_colour_share(seen.colour[ring]),
                       one_colour_share(seen.colour[place]), int(ring.sum())))
    return places, float(sizes.max())


def is_smear(ring_share, inside_share):
    """A place left unpainted is a smear when the paint round it is not one colour and what the model shows in it,
    spread from that paint, is not one colour either."""
    return ring_share < ONE_COLOUR and inside_share < ONE_COLOUR


def left_out_places(corners, triangles, colours, cloud, views):
    """The places the touch-up's views left unpainted, judged on the finished model: from each view (turn_deg,
    tilt_deg), the pixels it sees squarely that the finished colour cloud does not reach, joined (`judged_places`).
    There the model shows the colour spread from round the place: a place larger than `LARGEST_FOREIGN_PATCH` of the
    object's size is a smear (`is_smear`) when its ring (`ring_of`) is not one colour (`one_colour_share`) and neither
    is what the model shows in it. Returns each view's places (their size, ring, inside and whether a smear), the
    largest place and the largest smear (0 for none)."""
    found = {"views": [], "largest_place": 0.0, "largest_smear": 0.0, "one_colour": ONE_COLOUR}
    for view in views:
        places, largest = judged_places(seen_from(corners, triangles, colours, cloud, view["turn_deg"],
                                                  view["tilt_deg"]))
        judged = [{"size": round(size, 4), "ring_one_colour": round(ring, 4), "inside_one_colour": round(inside, 4),
                   "ring_pixels": ring_pixels, "smear": is_smear(ring, inside)}
                  for _, size, ring, inside, ring_pixels in places]
        found["views"].append({"largest": round(largest, 4), "places": judged})
        found["largest_place"] = max(found["largest_place"], round(largest, 4))
        found["largest_smear"] = max([found["largest_smear"], *(p["size"] for p in judged if p["smear"])])
    return found


def worst_view(fit):
    """The touch-up view to show: the one with the largest smear, else the most foreign kept fill, else the largest
    place left unpainted."""
    left = fit["left_out"]["views"]
    smears = [max([p["size"] for p in view["places"] if p["smear"]], default=0.0) for view in left]
    if max(smears, default=0.0) > 0:
        return int(np.argmax(smears))
    if max(fit["views"], default=0.0) > 0 and len(fit["views"]) == len(left):
        return int(np.argmax(fit["views"]))
    return int(np.argmax([view["largest"] for view in left]))


def touched_up(inpaint, corners, triangles, colours, cloud, text, seed=0, most=MOST_VIEWS, report=print,
               views_out=None, judge=True):
    """The colour cloud with the places the views missed painted from views chosen to see them: (cloud, record,
    where each point came from: 0 the six views, k + 1 touch-up view k). `inpaint(picture, mask, text, seed)` paints
    the masked pixels of a picture and returns it. With `judge`, each view's fill is judged against the six views'
    paint, tried again when foreign, and its foreign pixels left out."""
    views, before = chosen_views(corners, triangles, cloud, most)
    ring = cloud
    sources = [np.zeros(len(cloud[0]), dtype=np.int16)]
    record = {"unpainted_before": round(before, 4), "views": [], "judged": judge}
    for k, (turn, tilt, gain) in enumerate(views):
        picture, wanted, mask, surface, facing = drawn(corners, triangles, colours, cloud, turn, tilt)
        wanted &= latent_masked(mask)
        if not wanted.any():
            continue
        # Its points are numbered by its place in the record, not among the views chosen: a view with nothing to paint
        # is skipped without an entry, and numbered among the views chosen they would go out of step with the record.
        number = len(record["views"]) + 1
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        tries = []
        for attempt in range(MOST_FILL_TRIES if judge else 1):
            painted = np.asarray(inpaint(picture, mask_image, text, seed + k + 100 * attempt)
                                 .convert("RGB").resize((SIDE, SIDE))).astype(np.float64) / 255
            share, left_out, checked = judged_fill(painted, wanted, surface, facing, ring) if judge \
                else (0.0, np.zeros_like(wanted), None)
            tries.append((share, attempt, painted, left_out, checked))
            if share <= GOOD_ENOUGH:
                break
            # After `FILL_TRIES`, another seed only while the try to be kept was, as painted, more foreign than the
            # fill check allows a kept fill: what it left out is left to the spread.
            if len(tries) >= FILL_TRIES and min(tries, key=lambda t: t[0])[4]["foreign"] <= MOST_FOREIGN:
                break
        share, kept_try, painted, left_out, checked = min(tries, key=lambda t: t[0])
        if views_out:
            from . import credentials

            credentials.save_picture(Image.fromarray((painted * 255).round().astype(np.uint8)),
                                     f"{views_out}/touch-up-{k}.png", credentials.OBJECT_MODELS)
            # Where our own code asked the inpainter to paint: no model made it, and its credentials say so.
            credentials.save_picture(Image.fromarray(mask.astype(np.uint8) * 255), f"{views_out}/touch-up-{k}-mask.png",
                                     (), credentials.ALGORITHMIC)
        kept = wanted & ~left_out
        cloud = (np.concatenate([cloud[0], surface[kept]]).astype(np.float32),
                 np.concatenate([cloud[1], facing[kept]]).astype(np.float32),
                 np.concatenate([cloud[2], painted[kept]]).astype(np.float32))
        sources.append(np.full(int(kept.sum()), number, dtype=np.int16))
        entry = {"turn_deg": turn, "tilt_deg": tilt, "adds": round(gain, 4), "pixels": int(kept.sum()),
                 "tries": len(tries), "kept_try": kept_try, "foreign": round(share, 4),
                 "left_out": int(left_out.sum()), "source": number, "filled": int(wanted.sum())}
        if checked is not None:
            # The fill as painted, judged as the finish's check judges a fill, before what was foreign was left out:
            # what is kept is within `KEEP_DELTA_E` of the paint round it by construction, so the check judges this.
            entry["as_painted"] = checked
        record["views"].append(entry)
        report(f"touch-up view {k}: {turn}° round, {tilt:.0f}° above, {gain:.1%} of the surface as seen; "
               f"{len(tries)} tries, {share:.1%} of the fill foreign, {int(left_out.sum())} pixels left out")
    return cloud, record, np.concatenate(sources)


def fill_picture(corners, triangles, cloud, sources, turn, tilt, side=360, least=None, colours=None):
    """The object from a touch-up view: its painted colours dimmed, each filled place green where its colour is found
    round it and red where it is foreign. `least`: `filled_misfit`, when it is measured already. With the corners'
    `colours`, the places the view left unpainted larger than `LARGEST_FOREIGN_PATCH` (`judged_places`) are ringed:
    blue when the paint round them is one colour, magenta when it is not (a smear)."""
    points, normals, cloud_colours = (np.asarray(c, dtype=np.float64) for c in cloud)
    sources = np.asarray(sources) if sources is not None else np.zeros(len(points), dtype=np.int16)
    filled = sources > 0
    if least is None:
        least = filled_misfit(points, normals, cloud_colours, sources) if filled.any() else np.zeros(0)
    each = np.zeros(len(points))
    each[filled] = least
    least = each
    frame = render.Framing(corners, turn, side, tilt, extent=float(np.ptp(corners, axis=0).max()) * 1.15)
    xy, depth = frame.pixels(corners)
    _, owner, bary = render.rasterised(xy, depth, triangles, side, side)
    hit = owner >= 0
    at = np.einsum("pk,pkd->pd", bary[hit], corners[triangles[owner[hit]]])
    nearest = cKDTree(points).query(at)[1]
    image = np.full((side, side, 3), 245.0)
    shown = cloud_colours[nearest] * 255 * 0.45 + 245 * 0.2
    is_fill = filled[nearest]
    strength = np.clip(least[nearest] / (1.5 * FOREIGN_DELTA_E), 0, 1)[:, None]
    shown[is_fill] = (np.array([215.0, 60, 50]) * strength + np.array([70.0, 175, 90]) * (1 - strength))[is_fill]
    image[hit] = shown
    if colours is not None:
        places, _ = judged_places(seen_from(corners, triangles, np.asarray(colours, dtype=np.float64),
                                            (points, normals, cloud_colours), turn, tilt))
        for place, _, ring, inside, _ in places:
            small = np.asarray(Image.fromarray(place).resize((side, side), Image.NEAREST))
            edge = small & ~ndimage.binary_erosion(small, iterations=2)
            image[edge] = (210, 40, 200) if is_smear(ring, inside) else (40, 110, 240)
    return Image.fromarray(image.astype(np.uint8))


class Inpainter:
    """SDXL's base model painting the masked pixels of a picture (diffusers' SDXL inpainting pipeline on the base
    weights: the masked latents denoised from a partly noised start, the rest held to the picture at every step).
    Needs an NVIDIA card."""

    def __init__(self, sdxl_dir, vae_dir, steps=30, guidance=6.0, strength=STRENGTH):
        import torch
        from diffusers import AutoencoderKL, StableDiffusionXLInpaintPipeline

        vae = AutoencoderKL.from_pretrained(vae_dir, torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(sdxl_dir, vae=vae, variant="fp16",
                                                                     torch_dtype=torch.float16)
        self.steps, self.guidance, self.strength = steps, guidance, strength

    def __call__(self, picture, mask, text, seed):
        import torch

        from .paint import NEGATIVE

        self.pipe.to("cuda")
        try:
            return self.pipe(prompt=f"{text}, {FILL_WORDS}", negative_prompt=f"{NEGATIVE}, {FILL_NEGATIVE}",
                             image=picture, mask_image=mask, height=SIDE, width=SIDE, strength=self.strength,
                             num_inference_steps=self.steps, guidance_scale=self.guidance,
                             # Its noise drawn on the CPU, the same on every card (see paint).
                             generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]
        finally:
            self.pipe.to("cpu")
            torch.cuda.empty_cache()
