"""The checks a shaped object passes, each measured on its surface (metres, +Y up, standing on y = 0):

* its outline against the picture's (seen from the turn that matches best: an object is painted three-quarter on);
* the mesh whole: no holes, one main piece, small loose bits listed;
* it stands on its base (when it is a thing that stands): the face of its hull it rests on is level, and its centre
  of mass lies over that face, well inside it;
* mirror-symmetric left to right (when the prompt implies it): the best upright mirror plane, and how far the
  mirrored surface lies from the surface.

Each limit says what it means in the object's own measure; none is set from any one result.
"""

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull, cKDTree

from . import mesh_tools as M
from . import render
from .picture_checks import Check

# The outline: compared at this many pixels high, from these turns (an object is painted three-quarter on), with the
# best kept.
OUTLINE_SIDE = 384
OUTLINE_RENDER_SIDE = 448
OUTLINE_TURNS = tuple(range(-90, 91, 15))
# The painter is asked for a view slightly from above.
OUTLINE_TILTS = (0.0, 12.0, 24.0)
OUTLINE_REFINE = ((5.0, 4.0), (2.0, 2.0), (1.0, 1.0))
# Each refining step size is repeated while it gains, at most this many times.
MOST_PATTERN_MOVES = 8
# The picture is a photograph, drawn through a lens: its near parts larger than its far ones (the fix-check's chest:
# its near front foot lower than its far back foot while its lid shows hardly at all, which no straight-on view of any
# chest can draw). The camera's distance (in the object's largest sizes) is searched with the view: straight on (None),
# then from 6, 4 and 3 sizes away (fields of view of about 20, 30 and 40 degrees for an object filling half the
# picture). A model that matches its picture only through a lens is matched through it.
OUTLINE_DISTANCES = (None, 6.0, 4.0, 3.0)
# The outlines are compared on a canvas this many times as wide as high, so no outline is squeezed to fit (on a canvas
# twice as wide as high, a chest 2.25 times as wide as high in its picture and 2.10 in the model was squeezed to 2 in
# both, and their widths were never compared).
OUTLINE_CANVAS_ASPECT = 6
LEAST_OUTLINE_IOU = 0.85
# The outline's edges: a compact shape whose outline is offset all round by this share of its height overlaps the
# picture's by the 85 % above (the limit figures from prompts were held to), so this is the mean distance allowed;
# a part out of place is judged at the 95th percentile, at twice it. On the trials: the first round table, a web
# between its legs, 1.34 % and 5.47 % (fails); the second chair and table, matching their pictures, 0.44 % and 1.07 %,
# 0.73 % and 2.08 % (pass).
MOST_OUTLINE_MEAN_SHARE = 0.019
MOST_OUTLINE_FAR_SHARE = 0.04
# A piece one outline has in a gap of the other (within the other's convex hull: between a table's legs): its
# thickness (the widest circle inside it) at most twice the 95th percentile's limit, the width such a piece, lying
# between two of the other outline's edges, may have and still read the 95th percentile's limit at its edges. The
# first trial's round table's web between its legs is such a piece (11.5 % of the height, from every camera): its edges
# lie near the legs' edges, so the edge distances alone could miss it. A piece outside the other's hull reaches away
# from every edge of it, and the edge distances judge it.
MOST_OUTLINE_APART_SHARE = 2 * MOST_OUTLINE_FAR_SHARE
# A loose piece with at least this share of the surface is a part, not a speck: it fails "one main piece" for a
# weapon (a guard or a pommel floating) and is listed for an object. Specks under it are listed only.
LEAST_PART_SHARE = 0.002
# Corners this close (a share of the object's size) touch: their pieces are one.
TOUCHING_SHARE = 1e-6
# The main piece holds at least this share of the surface.
LEAST_MAIN_SHARE = 0.98
# The base: its hull's face is level within this angle, and the centre of mass lies inside it by at least this share
# of the face's own radius (a chair whose centre is nine tenths of the way to its edge tips at a touch).
MOST_BASE_TILT_DEG = 2.0
LEAST_BASE_MARGIN = 0.15
# The ground contact: the object's points this share of its height above its lowest (at least this many metres): a
# foot a few millimetres short still touches once the object settles; a leg 3 cm short does not.
CONTACT_BAND_SHARE = 0.005
LEAST_CONTACT_BAND_M = 0.001
# Lifted feet: the underside's height map in cells of this share of the base's size, from this many points; a low
# place up to this share of the height above the ground is a foot that does not reach it.
FOOT_CELL_SHARE = 0.015
FOOT_SAMPLES = 120000
FOOT_SEARCH_SHARE = 0.05
# A foot stands at least this much lower than everything round it (the larger of the two).
LEAST_FOOT_PROMINENCE_M = 0.005
FOOT_PROMINENCE_SHARE = 0.01
# A lifted foot lies outside the touching feet's outline by at least this share of the outline's radius (a chair's
# short leg lies about 1.8 of it outside the other three legs' triangle; a bottom rim round inset feet a tenth).
LEAST_FOOT_OUTSIDE_SHARE = 0.2
# Symmetry: the mirrored surface's mean distance from the surface, as a share of the object's largest size. Two
# hundredths of a metre-high chair is 2 cm: the shape model's noise is less. A part missing or out of place is a local
# fault the mean dilutes (the test chair with a leg missing read 1.67 %, an armrest on one
# side only 1.97 %), so a high percentile of the mirrored surface's distances, measured to the surface's triangles
# (no sampling floor), has its own limit. The 98th: a chair's leg is under 5 % of its surface, so at the 95th a leg
# missing read 1.25 %, as a good chair; at the 98th it reads 23.5 %, a leg 8 cm out 9.6 %, an armrest on one side
# 17.1 %, while the trials' models that pass every check read at most 3.5 % (a book; tables 2.7 %, stools 1.9 %).
# The limit lies in that gap (docs/objects.md, "Defects found by measurement").
MOST_MIRROR_DISTANCE_SHARE = 0.02
MIRROR_FAR_PERCENTILE = 98
MOST_MIRROR_FAR_SHARE = 0.05
MIRROR_SAMPLES = 20000
MIRROR_COARSE_STEP_DEG = 3.0


class OutlineView:
    """The view an outline search found: its overlap, turn, tilt and the camera's distance, the two outlines compared
    from it (the picture's and the surface's, framed alike), and how many of the surface's outline pixels fell beyond
    the canvas's sides (`cut`), so the edges' distances need draw neither again."""

    def __init__(self, iou, turn, tilt, distance, painted, made, cut=0):
        self.iou, self.turn, self.tilt, self.distance = iou, turn, tilt, distance
        self.painted, self.made, self.cut = painted, made, cut


def view_search(painted, aspect, corners, triangles, by_box=False, counted=None, by_overlap=False):
    """The view (turn, tilt, the camera's distance) whose outline lies nearest the picture's: an `OutlineView`.
    `painted` is the picture's outline as `placed` frames it on a canvas `aspect` heights wide (`by_box`: centred by
    its box, else by its area's middle); `counted`, where on the canvas edges count (the carving leaves a hole's inner
    part out; None: everywhere). `by_overlap`: the views compared by their overlap (the carving's camera, where the
    holes are left out and a web must not pull it).

    The views are compared by the check's own measure, the outlines' edges' mean distance both ways (chamfer matching:
    Barrow, Tenenbaum, Bolles and Wolf, 1977; Borgefors, 1988), not by their overlap: a thin part a few pixels off its
    place overlaps nothing, so the overlap falls from 0.94 to 0.5 three degrees from a bench's true view, and a coarse
    grid of views never finds such a needle; its edges move a few pixels and the distance grows smoothly. Every turn,
    tilt and camera distance of the coarse grid is drawn; then the best view at each camera distance is refined, each
    step size repeated while it gains (a pattern search), and the best refined view kept. Comparing the overlap and
    refining only the best coarse view, its distance fixed, matched a bench drawn from its own surface at 10 degrees
    round and 12 above, straight on, through a lens 3 sizes away, and failed it. A view whose outline is wider than the
    canvas comes after every view whose is not.
    Only the distances are kept; the winner is drawn once more."""
    from scipy import ndimage
    from scipy.spatial import cKDTree

    counted = np.ones(painted.shape, dtype=bool) if counted is None else counted
    painted_edge = edge_of(painted)
    to_painted = ndimage.distance_transform_edt(~painted_edge)
    painted_points = np.argwhere(painted_edge & counted)
    seen = {}

    side = render_side(aspect)

    def made_at(turn, tilt, distance):
        drawn = render.silhouette(corners, triangles, turn, side, tilt, distance)
        made, cut, _ = placed(drawn, aspect, by_box)
        return made, cut

    def tried(turn, tilt, distance):
        """(how near: the cut first, then minus the edges' mean distance; the view)."""
        key = (turn, tilt, distance)
        if key not in seen:
            made, cut = made_at(turn, tilt, distance)
            if by_overlap:
                both = int(np.count_nonzero(painted & made & counted))
                union = int(np.count_nonzero((painted | made) & counted)) + cut
                seen[key] = (0, both / max(1, union))
                return (seen[key], turn, tilt, distance)
            made_edge = edge_of(made) & counted
            if not made_edge.any() or not len(painted_points):
                seen[key] = (-1, -np.inf)
            else:
                there = to_painted[made_edge]
                back = cKDTree(np.argwhere(made_edge)).query(painted_points)[0]
                seen[key] = (-int(cut > 0), -float((there.sum() + back.sum()) / (len(there) + len(back))))
        return (seen[key], turn, tilt, distance)

    def refined(best):
        for step_turn, step_tilt in OUTLINE_REFINE:
            for _ in range(MOST_PATTERN_MOVES):
                _, turn, tilt, distance = best
                moved = max([best] + [tried(turn + dt, max(0.0, tilt + de), distance)
                                      for dt in (-step_turn, 0, step_turn) for de in (-step_tilt, 0, step_tilt)
                                      if dt or de], key=lambda t: t[0])
                if moved is best:
                    break
                best = moved
        return best

    coarse = [max((tried(turn, tilt, distance) for turn in OUTLINE_TURNS for tilt in OUTLINE_TILTS),
                  key=lambda t: t[0]) for distance in OUTLINE_DISTANCES]
    _, turn, tilt, distance = max((refined(best) for best in coarse), key=lambda t: t[0])
    made, cut = made_at(turn, tilt, distance)
    both = int(np.count_nonzero(painted & made & counted))
    union = int(np.count_nonzero((painted | made) & counted)) + cut
    return OutlineView(both / max(1, union), turn, tilt, distance, painted, made, cut)


def edge_of(mask):
    """A mask's edge: its pixels with a pixel outside it among their four neighbours."""
    from scipy import ndimage

    return mask & ~ndimage.binary_erosion(mask)


def outline_search(picture_mask, corners, triangles):
    """The view whose outline overlaps the picture's most (`view_search`), the outlines centred by their areas'
    middles on a canvas wide enough for the picture's (`canvas_aspect`)."""
    aspect = canvas_aspect(picture_mask)
    painted, _, _ = placed(picture_mask, aspect)
    return view_search(painted, aspect, corners, triangles)


def outline_judged(picture_mask, corners, triangles):
    """The outline search and the edges' distances from its view: (`OutlineView`, mean, 95th percentile, the
    thickest piece in a gap; see `outline_distance`)."""
    view = outline_search(picture_mask, corners, triangles)
    return (view, *outline_distance(view))


def overlay_of(painted, made):
    """Green both, red the picture's alone, blue the model's alone; cropped to them (at least twice as wide as high)."""
    image = np.full(painted.shape + (3,), 238, dtype=np.uint8)
    image[painted & made] = (120, 190, 120)
    image[painted & ~made] = (220, 60, 60)
    image[made & ~painted] = (60, 90, 220)
    used = np.flatnonzero((painted | made).any(0))
    middle, half = painted.shape[1] // 2, OUTLINE_SIDE
    if len(used):
        half = max(half, middle - int(used[0]) + 8, int(used[-1]) + 9 - middle)
    return Image.fromarray(image[:, max(0, middle - half):middle + half])


# A picture wider than the canvas widens it, with this much room for the surface's outline to be wider still: a 10:1
# plank pictured from the front, cut to a canvas of 6:1, filled the canvas and matched
# a view of the plank seen nearly end-on, itself cut to fill the canvas (overlap 1.000, edges 0.00 % apart).
PICTURE_ASPECT_ROOM = 1.25


def render_side(aspect):
    """The square the surface's outline is drawn on for a canvas `aspect` heights wide: `OUTLINE_RENDER_SIDE`, larger
    for a canvas wider than `OUTLINE_CANVAS_ASPECT`, so a long thing's height keeps its pixels (a 10:1 plank drawn on
    448 pixels is 38 high)."""
    return int(round(OUTLINE_RENDER_SIDE * max(1.0, aspect / OUTLINE_CANVAS_ASPECT)))


def canvas_aspect(picture_mask):
    """The canvas's width in heights for this picture's outline: `OUTLINE_CANVAS_ASPECT`, or the picture's own
    width in heights with `PICTURE_ASPECT_ROOM` when it is wider, so the picture's outline is never cut."""
    rows, cols = np.flatnonzero(picture_mask.any(1)), np.flatnonzero(picture_mask.any(0))
    if len(rows) == 0:
        return OUTLINE_CANVAS_ASPECT
    aspect = (cols[-1] - cols[0] + 1) / (rows[-1] - rows[0] + 1)
    return max(OUTLINE_CANVAS_ASPECT, int(np.ceil(aspect * PICTURE_ASPECT_ROOM)))


def placed(mask, aspect=OUTLINE_CANVAS_ASPECT, by_box=False):
    """The outline cropped to itself, scaled to `OUTLINE_SIDE` high and set on a canvas `aspect` times as wide,
    centred by its area's middle (the outline check, the likeness) or by its box (`by_box`: the carving, where a web
    must not move it). An outline wider than the canvas is cut at its sides. Returns (canvas, the pixels cut off,
    (first row, first column, rows' scale, columns' scale, the canvas column of the crop's left edge)): a point at
    (column x, row y) of the mask (pixel k spans k to k + 1) lands at canvas row (y - first row) * rows' scale and
    column (x - first column) * columns' scale + left."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    wide = OUTLINE_SIDE * aspect
    canvas = np.zeros((OUTLINE_SIDE, wide), dtype=bool)
    if len(rows) == 0:
        return canvas, 0, (0, 0, 1.0, 1.0, 0)
    crop = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    scale = OUTLINE_SIDE / crop.shape[0]
    width = max(1, round(crop.shape[1] * scale))
    scaled = np.asarray(Image.fromarray(crop.astype(np.uint8) * 255).resize((width, OUTLINE_SIDE),
                                                                              Image.Resampling.BILINEAR)) > 127
    if by_box:
        left = wide // 2 - width // 2
    else:
        left = wide // 2 - (int(round(np.nonzero(scaled)[1].mean())) if scaled.any() else width // 2)
    span = slice(max(0, left), min(wide, left + width))
    if span.stop > span.start:
        canvas[:, span] = scaled[:, span.start - left:span.stop - left]
    cut = int(np.count_nonzero(scaled)) - int(np.count_nonzero(canvas))
    return canvas, cut, (int(rows[0]), int(cols[0]), scale, width / crop.shape[1], left)


def normalised(mask, aspect=OUTLINE_CANVAS_ASPECT):
    """The outline as the outline check frames it (`placed`, centred by its area's middle): the canvas alone."""
    return placed(mask, aspect)[0]


def wholeness(corners, triangles):
    """The surface's holes, torn edges and pieces: a dict of measures, with the loose pieces (share of the surface,
    where they are, their size) largest first."""
    open_edges, holes, torn = M.open_edge_loops(triangles)
    # Pieces that touch are one: corners at the same place joined first (a hoop split from the staves it touches is
    # not loose).
    size = float(np.ptp(corners, axis=0).max())
    corners, triangles = M.welded(corners, triangles, TOUCHING_SHARE * size)
    piece, count = M.pieces(corners, triangles)
    _, areas = M.face_normals_and_areas(corners, triangles)
    piece_area = np.bincount(piece, weights=areas, minlength=count)
    order = np.argsort(piece_area)[::-1]
    total = max(piece_area.sum(), 1e-20)
    loose = []
    for p in order[1:]:
        used = np.unique(triangles[piece == p])
        low, high = corners[used].min(0), corners[used].max(0)
        loose.append({"share": float(piece_area[p] / total), "at": [round(float(x), 3) for x in (low + high) / 2],
                      "size_m": round(float((high - low).max()), 3), "triangles": int((piece == p).sum())})
    return {"open_edges": open_edges, "holes": holes, "torn_edges": torn, "pieces": int(count),
            "main_share": float(piece_area[order[0]] / total) if count else 0.0, "loose": loose}


def whole_checks(measures, weapon):
    """The wholeness measures as checks. A weapon fails on any loose part (a guard or pommel floating); an object
    on a main piece under `LEAST_MAIN_SHARE`."""
    parts = [p for p in measures["loose"] if p["share"] >= LEAST_PART_SHARE]
    specks = [p for p in measures["loose"] if p["share"] < LEAST_PART_SHARE]
    listed = "; ".join(f"{p['share']:.2%} ({p['size_m'] * 100:.1f} cm) at {p['at']}" for p in parts[:5])
    checks = [
        Check("no holes", measures["holes"], 0, "holes", measures["holes"] == 0,
              f"{measures['holes']} holes ({measures['open_edges']} open edges), {measures['torn_edges']} torn edges"),
        Check("one main piece", measures["main_share"] * 100, LEAST_MAIN_SHARE * 100, "%",
              measures["main_share"] >= LEAST_MAIN_SHARE and not (weapon and parts),
              f"the main piece is {measures['main_share']:.1%} of the surface; {len(parts)} loose parts"
              + (f": {listed}" if parts else "") + f"; {len(specks)} specks under {LEAST_PART_SHARE:.1%}"),
    ]
    return checks


def resting_face(corners, triangles):
    """The face of the surface's hull it settles onto when set down upright: the hull face straight under the centre
    of mass. Returns (its outward normal, its corners, the centre of mass, whether the volume's centre was used)."""
    volume, centre = M.volume_and_centre(corners, triangles)
    enclosed = volume > 0 and M.open_edge_loops(triangles)[0] == 0
    if not enclosed:
        centre = M.surface_centre(corners, triangles)
    hull = ConvexHull(corners)
    normals, offsets = hull.equations[:, :3], hull.equations[:, 3]
    below = np.flatnonzero(normals[:, 1] < -1e-6)
    best = None
    for f in below:
        n, d = normals[f], offsets[f]
        # Down from the centre: centre - t * y meets the face's plane n.x + d = 0.
        t = (n @ centre + d) / n[1]
        hit = centre - t * np.array([0.0, 1.0, 0.0])
        a, b, c = corners[hull.simplices[f]]
        if inside_triangle_xz(hit, a, b, c) and (best is None or t < best[0]):
            best = (t, f)
    if best is None:
        return None, None, centre, enclosed
    return normals[best[1]], corners[hull.simplices[best[1]]], centre, enclosed


def inside_triangle_xz(p, a, b, c):
    def side(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[2] - p3[2]) - (p2[0] - p3[0]) * (p1[2] - p3[2])
    d1, d2, d3 = side(p, a, b), side(p, b, c), side(p, c, a)
    negative = (d1 < 0) or (d2 < 0) or (d3 < 0)
    positive = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (negative and positive)


def base_measures(corners, triangles):
    """How the object stands: set down, it settles onto the face of its hull straight under its centre of mass. The
    tilt of that face from level (degrees; over the limit, it falls onto another side); its centre of mass's place
    over where it then touches the ground (the hull's corners within the contact band of that face): its least
    distance inside the contact's outline as a share of the outline's own radius (near 0 it rocks, negative it tips);
    and the feet outside that outline that do not reach the ground (`lifted_feet`), measured from the settled face."""
    normal, _, centre, enclosed = resting_face(corners, triangles)
    height = float(np.ptp(corners[:, 1]))
    found = {"tilt_deg": 90.0, "margin": -1.0, "base_m": 0.0, "enclosed": enclosed,
             "centre": [round(float(x), 4) for x in centre], "lifted_m": [], "height_m": height}
    if normal is None:
        found["lifted_m"] = lifted_feet(corners, triangles, np.array([0.0, -1.0, 0.0]), None)
        return found
    found["tilt_deg"] = float(np.degrees(np.arccos(np.clip(-normal[1], -1, 1))))
    band = max(LEAST_CONTACT_BAND_M, CONTACT_BAND_SHARE * height)
    hull_corners = corners[ConvexHull(corners).vertices]
    lowest = (hull_corners @ normal).max()
    touching = hull_corners[hull_corners @ normal >= lowest - band]
    across, other = across_face(normal)
    flat = np.stack([touching @ across, touching @ other], 1)
    try:
        outline = ConvexHull(flat) if len(flat) >= 3 else None
    except Exception:  # the contact is a line or a point: nothing to stand on
        outline = None
    found["lifted_m"] = lifted_feet(corners, triangles, normal, outline)
    if outline is None:
        return found
    # Each edge's equation: n.x + d <= 0 inside; the least distance inside is -max(n.x + d).
    point = np.array([centre @ across, centre @ other])
    inside_by = float(-(outline.equations[:, :2] @ point + outline.equations[:, 2]).max())
    radius = float(np.sqrt(outline.volume / np.pi))
    found.update(margin=inside_by / max(radius, 1e-9), base_m=2 * radius)
    return found


def across_face(normal):
    """Two directions across a face with this normal."""
    across = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(across) < 1e-6:
        across = np.cross(normal, [0.0, 0.0, 1.0])
    across /= np.linalg.norm(across)
    return across, np.cross(normal, across)


def lifted_feet(corners, triangles, normal, outline, search_share=FOOT_SEARCH_SHARE):
    """The gaps (metres) under feet that do not reach the ground once the object has settled onto the face with this
    (downward) normal: the underside seen as a height map from below that face (the lowest surface in each cell), its
    low places that stand lower than all round them, lie above the contact band but within `FOOT_SEARCH_SHARE` of
    the height (higher, a low place is a part such as a stretcher, not a foot), and lie outside the outline of where
    it touches (`outline`, in the face's plane; inside it, a low place under the body is held up by the feet round it
    and can never rock it)."""
    from scipy import ndimage

    points, _, _ = M.sampled(corners, triangles, FOOT_SAMPLES, seed=3)
    across, other = across_face(normal)
    u, v = points @ across, points @ other
    up = (points @ normal).max() - points @ normal
    cell = FOOT_CELL_SHARE * float(max(np.ptp(u), np.ptp(v)))
    cols = ((u - u.min()) / cell).astype(int)
    rows = ((v - v.min()) / cell).astype(int)
    heights = np.full((rows.max() + 1, cols.max() + 1), np.inf)
    np.minimum.at(heights, (rows, cols), up)
    lowest_near = ndimage.minimum_filter(heights, size=3, mode="constant", cval=np.inf)
    labels, count = ndimage.label(np.isfinite(heights) & (heights <= lowest_near))
    height = float(np.ptp(corners[:, 1]))
    band = max(LEAST_CONTACT_BAND_M, CONTACT_BAND_SHARE * height)
    least_prominence = max(LEAST_FOOT_PROMINENCE_M, FOOT_PROMINENCE_SHARE * height)
    gaps = []
    for k in range(1, count + 1):
        place = labels == k
        gap = float(heights[place].min())
        if not band < gap <= search_share * height:
            continue
        if outline is not None:
            # Within the support, or just outside it (a bottom rim round inset feet), a low place cannot be what the
            # object stands on; a foot that should touch lies well outside the others' outline.
            row, col = np.argwhere(place).mean(0)
            middle = np.array([u.min() + (col + 0.5) * cell, v.min() + (row + 0.5) * cell])
            outside_by = (outline.equations[:, :2] @ middle + outline.equations[:, 2]).max()
            if outside_by < LEAST_FOOT_OUTSIDE_SHARE * np.sqrt(outline.volume / np.pi):
                continue
        # A foot stands lower than all round it (a flat underside raised between feet is not one): the low place grown
        # over the cells joined to it that are within the prominence of its bottom, then what lies round that.
        near_bottom, _ = ndimage.label(heights <= gap + least_prominence)
        grown = near_bottom == near_bottom[place].max()
        ring = ndimage.binary_dilation(grown, iterations=2) & ~grown & np.isfinite(heights)
        if not ring.any() or heights[ring].min() - gap >= least_prominence:
            gaps.append(gap)
    return sorted(gaps, reverse=True)


def base_checks(measures):
    tilt, margin = measures["tilt_deg"], measures["margin"]
    mass = "its volume's" if measures["enclosed"] else "its surface's (it is not closed)"
    lifted = measures["lifted_m"]
    band = max(LEAST_CONTACT_BAND_M, CONTACT_BAND_SHARE * measures["height_m"])
    gap = lifted[0] if lifted else 0.0
    return [
        Check("every foot on the ground", gap * 1000, round(band * 1000, 1), "mm", not lifted,
              "every low place of its underside reaches the ground" if not lifted else
              f"{len(lifted)} foot or feet do not reach the ground: "
              + ", ".join(f"{g * 1000:.0f} mm" for g in lifted[:4]) + " short"),
        Check("base level", tilt, MOST_BASE_TILT_DEG, "°", tilt <= MOST_BASE_TILT_DEG,
              f"set down, it settles {tilt:.1f}° onto the face of its hull under its centre of mass"),
        Check("centre over the base", margin * 100, LEAST_BASE_MARGIN * 100, "%",
              margin >= LEAST_BASE_MARGIN and tilt <= MOST_BASE_TILT_DEG,
              "it falls onto another side before it rests" if tilt > MOST_BASE_TILT_DEG else
              f"{mass} centre of mass lies {margin:.0%} of the base's radius inside the edge of where it touches the "
              f"ground ({measures['base_m'] * 100:.0f} cm across)" + (": it rocks" if margin < LEAST_BASE_MARGIN else "")
              if margin >= 0 else f"{mass} centre of mass lies outside where it touches the ground: it tips over"),
    ]


def mirror_measure(corners, triangles, samples=MIRROR_SAMPLES):
    """The best upright mirror plane through the surface's middle: (the mean distance of the mirrored surface from
    the surface as a share of its largest size, the plane's turn in degrees from the X axis, its offset in metres, the
    `MIRROR_FAR_PERCENTILE`th percentile of the mirrored surface's distances from the surface, point to triangle, as a
    share of its size).
    The plane is found by the mean distance to the surface's samples; the percentile is measured exactly (to the
    surface's triangles, `mesh_tools.SurfaceDistance`), so it carries no sampling floor."""
    points, _, chosen = M.sampled(corners, triangles, samples, seed=1)
    tree = cKDTree(points)
    size = float(np.ptp(corners, axis=0).max())
    middle = (points.min(0) + points.max(0)) / 2

    def mirrored(turn_deg, offset):
        t = np.radians(turn_deg)
        n = np.array([np.cos(t), 0.0, np.sin(t)])
        signed = (points - middle) @ n - offset
        return points - 2 * signed[:, None] * n

    def distance(turn_deg, offset):
        return float(tree.query(mirrored(turn_deg, offset)[::4])[0].mean())

    coarse = [(distance(t, 0.0), t) for t in np.arange(0, 180, MIRROR_COARSE_STEP_DEG)]
    _, turn = min(coarse)
    best = (distance(turn, 0.0), turn, 0.0)
    for step_deg, step_m in ((1.0, 0.01 * size), (0.25, 0.0025 * size)):
        for t in turn + step_deg * np.arange(-2, 3):
            for o in best[2] + step_m * np.arange(-2, 3):
                d = distance(t, o)
                if d < best[0]:
                    best = (d, t, o)
        turn = best[1]
    d, turn, offset = best
    far = float(np.percentile(M.SurfaceDistance(corners, triangles, points, chosen)(mirrored(turn, offset)),
                              MIRROR_FAR_PERCENTILE))
    return d / size, float(turn % 180), float(offset), far / size


def mirror_check(share, turn, far=0.0):
    """Mirror symmetry by the mean distance (the shape model's noise against a whole object out of true) and by a high
    percentile (a part missing, out of place or on one side only: a local fault the mean dilutes)."""
    return Check("left-right symmetry", share * 100, MOST_MIRROR_DISTANCE_SHARE * 100, "% of size",
                 share <= MOST_MIRROR_DISTANCE_SHARE and far <= MOST_MIRROR_FAR_SHARE,
                 f"mirrored across its best upright plane (turned {turn:.0f}°), the surface lies {share:.2%} of the "
                 f"object's size from itself on average and {far:.2%} at the {MIRROR_FAR_PERCENTILE}th percentile "
                 f"(limits {MOST_MIRROR_DISTANCE_SHARE:.0%} and {MOST_MIRROR_FAR_SHARE:.0%})")


def outline_distance(view):
    """How far the two outlines of a view (`OutlineView`) lie apart: (the edges' mean distance, their 95th
    percentile, the thickest piece one outline has and the other not), each edge pixel to the other outline's nearest
    edge, both ways, all as shares of the height; the piece in a gap of the other outline, within its hull."""
    from scipy import ndimage

    painted, made = view.painted, view.made
    if not painted.any() or not made.any():
        return 1.0, 1.0, 1.0
    to_made = ndimage.distance_transform_edt(~edge_of(made))[edge_of(painted)]
    to_painted = ndimage.distance_transform_edt(~edge_of(painted))[edge_of(made)]
    both = np.concatenate([to_made, to_painted]) / OUTLINE_SIDE
    apart = max(thickest(made & ~painted & hull_of(painted)), thickest(painted & ~made & hull_of(made)))
    return float(both.mean()), float(np.percentile(both, 95)), apart


def hull_of(mask):
    from skimage.morphology import convex_hull_image

    return convex_hull_image(mask)


def thickest(only):
    """The thickness of the thickest piece of a mask, as a share of the height: the widest circle inside it."""
    from scipy import ndimage

    return 2 * float(ndimage.distance_transform_edt(only).max()) / OUTLINE_SIDE if only.any() else 0.0


def outline_checks(view, mean, far, apart):
    """The outline judged by its edges' distance and the thickest piece of either alone (the overlap, shown, is unfair
    to thin parts: a leg a pixel or two off counts wholly against it; the second trial's chair matched its picture
    closely and overlapped 79 %). A surface whose outline from the view found is wider than the canvas (the picture's
    width with room to spare) fails: its edges at the canvas's sides are the canvas's, not its own."""
    lens = "straight on" if view.distance is None else f"through a lens {view.distance:g} times its size away"
    seen = f"seen from {view.turn:+.0f}° round and {view.tilt:.0f}° above, {lens}"
    cut = getattr(view, "cut", 0)
    return [
        Check("outline matches the picture", mean * 100, MOST_OUTLINE_MEAN_SHARE * 100, "% of height",
              mean <= MOST_OUTLINE_MEAN_SHARE and far <= MOST_OUTLINE_FAR_SHARE and apart <= MOST_OUTLINE_APART_SHARE
              and cut == 0,
              f"the outlines' edges lie {mean:.2%} of the height apart on average, {far:.2%} at the 95th percentile, "
              f"and the thickest piece of one in a gap of the other is {apart:.1%} of the height (limits "
              f"{MOST_OUTLINE_MEAN_SHARE:.1%}, {MOST_OUTLINE_FAR_SHARE:.0%} and {MOST_OUTLINE_APART_SHARE:.0%}), "
              f"{seen}" + (f"; from it the model's outline is wider than the canvas, "
                           f"{view.painted.shape[1] // OUTLINE_SIDE} heights ({cut} pixels beyond its sides)"
                           if cut else "")),
        Check("outlines overlap", view.iou * 100, LEAST_OUTLINE_IOU * 100, "%", view.iou >= LEAST_OUTLINE_IOU,
              f"the outlines overlap {view.iou:.0%} (intersection over union; shown, unfair to thin parts)",
              counted=False),
    ]


def set_down(corners, size_m, along_length=False):
    """The surface scaled so its largest size is `size_m`, its base's middle at the origin, standing on y = 0.
    `along_length` (a thing held in the hand, whose size is its length): its extent along its longest spread (its
    principal axis) is `size_m`, so one the shape model made at a slant is not made too long (a box's largest side
    scaled a spear at 45 degrees 1.4 times its length)."""
    extent = float(np.ptp(corners, axis=0).max())
    if along_length and len(corners) >= 3:
        spread = corners - corners.mean(0)
        axis = np.linalg.svd(spread, full_matrices=False)[2][0]
        extent = float(np.ptp(spread @ axis))
    scaled = corners * (size_m / max(extent, 1e-9))
    low, high = scaled.min(0), scaled.max(0)
    return scaled - np.array([(low[0] + high[0]) / 2, low[1], (low[2] + high[2]) / 2])
