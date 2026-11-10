"""Holes the picture shows through the object, opened in its shape where the shape model filled them with a web.

The picture is the one view of the object we have, and where it shows the backdrop through the object (the loop of a
jug's handle, a key's bow, a lantern's ring) nothing lies along those lines of sight: the object lies within the
silhouette's cone from the camera (Martin and Aggarwal, 1983, "Volumetric descriptions of objects from multiple
views"; Laurentini, 1994, "The visual hull concept for silhouette-based image understanding"; Kutulakos and Seitz,
2000, "A theory of shape by space carving"). The shape model guesses its surface from that picture and can close
such a loop with a thin web; the outline check finds it ("the thickest piece of one in a gap of the other").

This carves the shape by the picture's silhouette, one hole at a time, from the camera that matches the picture's
outline: the shape's surface that lies, seen from it, in the hole, well inside its edge (`MARGIN_SHARE`), is cut away
along that line (the surface cut where the line crosses its triangles, as the levelling cuts along a plane), and each
opening the cut leaves is closed by a wall along the lines of sight, stitched between the cut's front and back edges
(the established way to join two contours by a strip of triangles, the strip of least total edge length found by
dynamic programming: Fuchs, Kedem and Uselton, 1977), so the surface stays closed and in one piece. The camera is
searched as the outline check searches it, with the outlines framed by their boxes and the holes' inner parts left
out of the count, so the web cannot pull it to a view where the web fits. Only a web is carved:

* the hole is one the picture's outline closes round (the backdrop seen through the object, not round it);
* the shape's piece in it is thicker than an edge's misfit (`LEAST_PIECE_SHARE`), so a leg or a spindle a little off
  its picture's place is never shaved;
* along every line of sight through it, the shape's material is no deeper than the hole is wide: a web spans a hole
  and is thinner than it, the object's body behind a hole would be deeper (and the picture would show it there);
* the cut surface must close: each opening is stitched, then the whole is checked closed, with no edge used by more
  than two triangles, no more pieces than before and nothing left in the hole's core seen from the camera; else that
  hole is left as made, and says so.

The shape the model made is kept unchanged in `shape.npz`; the finish carves the shape it exports (and the shape stage
judges each try carved), and the sidecar's `carved` says what was found and cut.
"""

import numpy as np
from PIL import Image
from scipy import ndimage

from . import mesh_repair
from . import mesh_tools as M
from . import object_checks as O
from . import render
from .cutout import for_shape
from .picture_checks import Check

# A piece of the shape in a hole of the picture is a web when it is thicker than this share of the height: the 95th
# percentile's limit on the outline's edges, the most an edge out of its place is allowed. On the trials the pieces
# a misfit leaves along a hole's edge (a chair's spindles, a lantern's ring) are at most 2.1 %; the jug's webs 13.1 to
# 15.1 %.
LEAST_PIECE_SHARE = O.MOST_OUTLINE_FAR_SHARE
# The cut keeps this share of the height inside the hole's edge: the outline's mean limit, so an edge off by what the
# check allows is never cut, and a picture's edge a pixel or two off its shape's does not move the cut onto the object.
MARGIN_SHARE = O.MOST_OUTLINE_MEAN_SHARE
# The shape's outline is moved sideways by up to this many canvas pixels to match the picture's with their holes
# closed (both are framed by their boxes; a part a little wider or narrower than its picture's moves a box's middle).
MOST_SHIFT_PX = 12


# A hole of the picture's outline shows the backdrop through the object when its colour (the median, CIE76 delta E in
# Lab) lies within this of the backdrop's just outside the object; else it is a thin part of the object the cut-out
# missed (a lantern's glass, a mirror, a framed canvas), never carved and never judged open. On the trials every hole
# with an inner part read at most 2.2 from the backdrop (a lantern's glass, the backdrop seen through it), most under 1;
# the made-up mirror whose glass the cut-out missed reads 63 (docs/objects.md, "Holes the picture shows").
MOST_HOLE_BACKDROP_DELTA_E = 10.0
# The backdrop round the object: the picture's pixels this far outside its cut-out (pixels of the picture).
BACKDROP_BAND_PX = (12, 48)


def backdrop_lab(picture_rgb, mask):
    """The median Lab colour of the backdrop just outside the cut-out."""
    from skimage.color import rgb2lab

    near = ndimage.binary_dilation(mask, iterations=BACKDROP_BAND_PX[1]) & \
        ~ndimage.binary_dilation(mask, iterations=BACKDROP_BAND_PX[0])
    pixels = np.asarray(picture_rgb, dtype=np.float64)[near] / 255.0
    return np.median(rgb2lab(pixels[None])[0], axis=0) if len(pixels) else None


def canvas_colours(picture_rgb, region, placement):
    """The picture's colours (0-255) under a canvas region (a mask framed by `object_checks.placed`, its
    `placement`): each canvas pixel's middle looked up in the picture."""
    row0, col0, rows_scale, cols_scale, left = placement
    rows, cols = np.nonzero(region)
    picture_rows = np.clip((row0 + (rows + 0.5) / rows_scale).astype(int), 0, picture_rgb.shape[0] - 1)
    picture_cols = np.clip((col0 + (cols + 0.5 - left) / cols_scale).astype(int), 0, picture_rgb.shape[1] - 1)
    return np.asarray(picture_rgb, dtype=np.float64)[picture_rows, picture_cols]


def backdrop_delta(picture_rgb, backdrop, region, placement):
    """How far (delta E) a canvas region's median colour in the picture lies from the backdrop's; None when either is
    unknown."""
    from skimage.color import rgb2lab

    if picture_rgb is None or backdrop is None or not region.any():
        return None
    colours = canvas_colours(picture_rgb, region, placement) / 255.0
    return float(np.linalg.norm(np.median(rgb2lab(colours[None])[0], axis=0) - backdrop))


def carved(picture_mask, corners, triangles, picture_rgb=None):
    """The shape with the webs in the picture's holes cut away: (corners, triangles, record). `corners` at the object's
    size, in the frame the shape model made it in (the picture's). `picture_rgb`: the picture, so a hole whose colour is
    not the backdrop's (a thin panel the cut-out missed) is never carved."""
    corners = np.asarray(corners, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    record = {"carved": False, "holes": []}
    # The cut-out the shape model was given (its largest piece, its pinholes filled), as the outline check judges it.
    picture_mask = for_shape(picture_mask)
    aspect = O.canvas_aspect(picture_mask)
    painted, _, placement = O.placed(picture_mask, aspect, by_box=True)
    holes, count = ndimage.label(ndimage.binary_fill_holes(painted) & ~painted)
    widths = [O.thickest(holes == k) for k in range(1, count + 1)]
    candidates = [k + 1 for k, width in enumerate(widths) if width > LEAST_PIECE_SHARE + 2 * MARGIN_SHARE]
    margin_px = MARGIN_SHARE * O.OUTLINE_SIDE
    # A hole whose colour in the picture is not the backdrop's is a panel the cut-out missed, not a hole: never carved.
    backdrop = backdrop_lab(picture_rgb, picture_mask) if picture_rgb is not None else None
    for k in list(candidates):
        delta = backdrop_delta(picture_rgb, backdrop, holes == k, placement)
        if delta is not None and delta > MOST_HOLE_BACKDROP_DELTA_E:
            candidates.remove(k)
            record["holes"].append({"width": round(widths[k - 1], 4), "backdrop_delta_e": round(delta, 1),
                                    "words": f"its colour lies {delta:.0f} (delta E) from the backdrop's: a part the "
                                             "cut-out missed, not a hole; left as made"})
    if not candidates:
        record["words"] = ("no hole in the picture's outline wide enough to hold a web: nothing carved"
                           if not record["holes"] else "nothing carved: " +
                           "; ".join(h["words"] for h in record["holes"]))
        return corners, triangles, record
    # The camera found by the outline check's search, but with the outlines framed by their boxes and compared by their
    # overlap, the holes' inner parts (where a web would be) left out of the count: a web cannot pull the camera off the
    # picture's. (Compared by the edges' distance, as the outline check now compares views, the holes' own edges, which
    # a web hides in the shape's outline, pulled the camera to where other edges lie near them: the first trial's round
    # table was seen 9 degrees from above, its webs as deep as the table, and left as made.)
    inside_by = ndimage.distance_transform_edt(np.isin(holes, candidates))
    view = O.view_search(painted, aspect, corners, triangles, by_box=True, counted=~(inside_by > margin_px),
                         by_overlap=True)
    record["view"] = {"turn_deg": view.turn, "tilt_deg": view.tilt, "distance": view.distance,
                      "overlap": round(view.iou, 4)}
    side = O.render_side(aspect)
    framing = render.Framing(corners, view.turn, side, view.tilt, distance=view.distance)
    xy, depth = framing.pixels(corners)
    made_canvas, _, (row0, col0, rows_scale, cols_scale, left) = O.placed(render.covered(xy, triangles, side, side),
                                                                          aspect, by_box=True)
    shift = matched_shift(ndimage.binary_fill_holes(painted), ndimage.binary_fill_holes(made_canvas))
    record["shift_px"] = shift
    made_canvas = np.roll(made_canvas, shift, axis=1)

    def on_canvas(points_xy):
        """Canvas (rows, columns) of render positions (column, row)."""
        return (points_xy[:, 1] - row0) * rows_scale, (points_xy[:, 0] - col0) * cols_scale + left + shift

    middles = (np.stack(np.meshgrid(np.arange(side), np.arange(side)), -1).reshape(-1, 2) + 0.5).astype(np.float64)
    pixel_rows, pixel_cols = (a.reshape(side, side) for a in on_canvas(middles))

    def seen_in(mask):
        """The render pixels whose middles fall in a canvas mask."""
        return looked_up(mask, pixel_rows, pixel_cols)

    depths = {}

    def deep_and_seen():
        """The depth of the shape along each line of sight (its nearest surface against its farthest) and where it is
        seen, drawn once, when a hole first holds a piece."""
        if not depths:
            near, owner, _ = render.rasterised(xy, depth, triangles, side, side)
            far, _, _ = render.rasterised(xy, -depth, triangles, side, side)
            depths["deep"], depths["seen"] = np.where(owner >= 0, near + far, 0.0), owner >= 0
        return depths["deep"], depths["seen"]

    metres_per_canvas_px = 1.0 / (rows_scale * framing.scale)
    for k in candidates:
        hole = holes == k
        found = {"width": round(widths[k - 1], 4)}
        piece = O.thickest(made_canvas & hole)
        found["piece"] = round(piece, 4)
        record["holes"].append(found)
        if piece <= LEAST_PIECE_SHARE:
            found["words"] = "open in the shape as in the picture"
            continue
        inside_by = ndimage.distance_transform_edt(hole)
        inner = inside_by > margin_px
        deep, drawn = deep_and_seen()
        seen = seen_in(inner) & drawn
        width_m = widths[k - 1] * O.OUTLINE_SIDE * metres_per_canvas_px
        deepest = float(deep[seen].max()) if seen.any() else 0.0
        found.update(width_m=round(width_m, 4), deepest_m=round(deepest, 4))
        if deepest > width_m:
            found["words"] = (f"the shape is {deepest * 100:.1f} cm deep in the hole, more than the hole is wide "
                              f"({width_m * 100:.1f} cm): body, not a web, left as made")
            continue
        # Each corner's distance outside the cut's line (canvas pixels; negative inside it, to be cut away).
        beyond = (ndimage.distance_transform_edt(~inner) - ndimage.distance_transform_edt(inner)).astype(np.float64)

        def placed(points):
            rows, cols = on_canvas(framing.pixels(points)[0])
            return rows, cols, ndimage.map_coordinates(beyond, [rows - 0.5, cols - 0.5], order=1, mode="nearest")

        fine_corners, fine_triangles = refined_along_line(corners, triangles, placed)
        field = placed(fine_corners)[2]
        normals, _ = M.face_normals_and_areas(fine_corners, fine_triangles)
        toward = (normals @ framing.matrix.T)[:, 2]
        cut_corners, cut_triangles, side_of, removed = cut_inside(fine_corners, fine_triangles, field, toward)
        if removed == 0:
            found["words"] = "nothing of the shape lies inside the hole's margin: left as made"
            continue
        opened = opened_up(cut_corners, fine_triangles, cut_triangles, side_of)
        if isinstance(opened, str):
            found["words"] = f"left as made: {opened}"
            continue
        cut_corners, opened = opened[0], opened[1:]
        # Seen from the camera, nothing may be left in the hole's core, a margin further in than the cut: there the
        # wall would span the hole.
        core = seen_in(inside_by > 2 * margin_px)
        left_in = render.covered(framing.pixels(cut_corners)[0], opened[0], side, side) & core
        if not core.any() or left_in.any():
            found["words"] = "left as made: " + ("the hole has no core past twice the margin" if not core.any() else
                                                 f"its wall still covers {int(left_in.sum())} pixels of the hole's core")
            continue
        corners, (triangles, walls) = cut_corners, opened
        found.update(carved=True, removed_triangles=int(removed), wall_triangles=int(walls),
                     words=f"a web {piece:.1%} of the height thick in a hole {widths[k - 1]:.1%} wide (at most "
                           f"{deepest * 100:.1f} cm deep) cut away: {int(removed)} triangles, the opening walled "
                           f"with {walls}")
        record["carved"] = True
    corners, triangles = M.compacted(corners, triangles)
    done = [h for h in record["holes"] if h.get("carved")]
    record["words"] = ("; ".join(h["words"] for h in done) if done else
                       "nothing carved: " + "; ".join(h["words"] for h in record["holes"]))
    return corners, triangles, record


# Near the cut's line no triangle edge is longer than this, seen on the canvas (pixels): a large flat triangle of the
# web (the simplifier leaves them where the surface is flat) cut straight across from edge to edge would cut a chord
# of a curved hole, leaving a sliver of web on one face and not the other.
LONGEST_EDGE_NEAR_LINE_PX = 6.0
MOST_REFINE_ROUNDS = 10


def refined_along_line(corners, triangles, placed):
    """The surface with its triangles near the cut's line split until none of their edges is longer than
    `LONGEST_EDGE_NEAR_LINE_PX` on the canvas: each long edge halved at its middle and every triangle on it split to
    match (one, two or three of its edges halved; so the surface stays joined), its shape unchanged. `placed(points)`
    gives their canvas rows, columns and distance outside the line."""
    for _ in range(MOST_REFINE_ROUNDS):
        rows, cols, field = placed(corners)
        r, c, f = rows[triangles], cols[triangles], field[triangles]
        edge_px = np.hypot(np.roll(r, -1, axis=1) - r, np.roll(c, -1, axis=1) - c)  # edge k: corner k to k + 1
        near = ((f.min(1) < 0) & (f.max(1) > 0)) | (np.abs(f).min(1) < edge_px.max(1))
        long_edges = near[:, None] & (edge_px > LONGEST_EDGE_NEAR_LINE_PX)
        if not long_edges.any():
            break
        ends = np.stack([triangles, np.roll(triangles, -1, axis=1)], -1)[long_edges]
        keys = np.unique(np.sort(ends, axis=1), axis=0)
        middle_of = {(int(a), int(b)): len(corners) + k for k, (a, b) in enumerate(keys)}
        corners = np.concatenate([corners, (corners[keys[:, 0]] + corners[keys[:, 1]]) / 2])
        triangles = split_triangles(triangles, middle_of)
    return corners, triangles


def split_triangles(triangles, middle_of):
    """Each triangle split by the middles of its halved edges (`middle_of`: sorted corner pair -> the middle's index),
    wound as it was."""
    made = []
    for t in triangles.tolist():
        middles = [middle_of.get((min(t[k], t[(k + 1) % 3]), max(t[k], t[(k + 1) % 3]))) for k in range(3)]
        halved = [m is not None for m in middles]
        count = sum(halved)
        if count == 0:
            made.append(t)
            continue
        # Turned so the halved edges come first: (a, b) halved, then (b, c), then (c, a).
        start = halved.index(True) if count == 1 else (halved.index(False) + 1) % 3 if count == 2 else 0
        a, b, c = (t[(start + k) % 3] for k in range(3))
        m0, m1, m2 = (middles[(start + k) % 3] for k in range(3))
        if count == 1:
            made += [[a, m0, c], [m0, b, c]]
        elif count == 2:
            made += [[m0, b, m1], [a, m0, m1], [a, m1, c]]
        else:
            made += [[a, m0, m2], [m0, b, m1], [m2, m1, c], [m0, m1, m2]]
    return np.array(made, dtype=np.int64)


# A corner this close to the cut's line (canvas pixels) lies on it, so no sliver is cut off.
ON_LINE_PX = 1e-6


def cut_inside(corners, triangles, field, toward):
    """The surface with what lies where `field` (per corner) is negative cut away, along its zero line (marching
    triangles, as the levelling cuts along a plane): (corners, triangles, each new open edge's side (True: the cut
    part faced the camera), how many triangles were cut away whole or in part)."""
    side = np.where(np.abs(field) <= ON_LINE_PX, 0, np.sign(field)).astype(np.int64)
    sides = side[triangles]
    whole = (sides >= 0).all(1)
    gone = (sides <= 0).all(1) & (sides < 0).any(1)
    crossing = (sides.max(1) > 0) & (sides.min(1) < 0)
    side_of = {}
    for t, facing in zip(triangles[gone].tolist(), toward[gone].tolist()):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            if side[a] == 0 and side[b] == 0:
                side_of[(min(a, b), max(a, b))] = facing > 0
    new_corners, made_at = [], {}

    def crossing_point(a, b):
        key = (min(a, b), max(a, b))
        if key not in made_at:
            share = field[a] / (field[a] - field[b])
            made_at[key] = len(corners) + len(new_corners)
            new_corners.append(corners[a] + share * (corners[b] - corners[a]))
        return made_at[key]

    clipped = []
    on_line = set(np.flatnonzero(side == 0).tolist())
    for t, facing in zip(triangles[crossing].tolist(), toward[crossing].tolist()):
        polygon = []
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            if side[a] >= 0:
                polygon.append(a)
            if side[a] * side[b] < 0:
                point = crossing_point(a, b)
                on_line.add(point)
                polygon.append(point)
        clipped += [[polygon[0], polygon[k], polygon[k + 1]] for k in range(1, len(polygon) - 1)]
        for p, q in zip(polygon, polygon[1:] + polygon[:1]):
            if p in on_line and q in on_line:
                side_of[(min(p, q), max(p, q))] = facing > 0
    kept = [triangles[whole]] + ([np.array(clipped, dtype=np.int64)] if clipped else [])
    if new_corners:
        corners = np.concatenate([corners, np.array(new_corners)])
    return corners, np.concatenate(kept), side_of, int(gone.sum() + crossing.sum())


def matched_shift(painted, made):
    """The sideways move (canvas pixels) of the shape's outline that overlaps the picture's most, holes closed."""
    best, chosen = -1.0, 0
    for shift in sorted(range(-MOST_SHIFT_PX, MOST_SHIFT_PX + 1), key=abs):
        moved = np.roll(made, shift, axis=1)
        both = np.count_nonzero(moved & painted)
        overlap = both / max(1, np.count_nonzero(moved | painted))
        if overlap > best:
            best, chosen = overlap, shift
    return chosen


def looked_up(mask, rows, cols):
    """Whether points (canvas rows and columns, pixel k spanning k to k + 1) fall in the mask."""
    r, c = np.floor(rows).astype(np.int64), np.floor(cols).astype(np.int64)
    inside = (r >= 0) & (c >= 0) & (r < mask.shape[0]) & (c < mask.shape[1])
    found = np.zeros(r.shape, dtype=bool)
    found[inside] = mask[r[inside], c[inside]]
    return found


def opened_up(corners, before, cut, side_of):
    """The cut surface's openings closed by walls: (corners, triangles, the walls' count), or the reason they could not
    be. `before`: the triangles before the cut; `side_of`: each open edge the cut made, and which face of the web it
    was on. Shreds of the web the cut leaves loose are dropped, and a stray small opening on one face is filled as the
    repair fills a small hole, before the faces' openings are paired."""
    before_open, _, before_torn = M.open_edge_loops(before)
    _, before_pieces = M.pieces(corners, before)
    cut = without_shreds(corners, cut)
    known = {tuple(t) for t in np.sort(cut, axis=1).tolist()}
    fronts, backs, slots = [], [], []
    for loop in mesh_repair.boundary_loops(cut):
        sides = [side_of.get((min(a, b), max(a, b))) for a, b in zip(loop, loop[1:] + loop[:1])]
        if any(s is None for s in sides):
            continue  # an opening the shape had before
        runs = smoothed_runs(sides)
        if runs is None:
            return "an opening runs back and forth between the web's faces"
        if len(runs) == 1:
            (fronts if runs[0][2] else backs).append(loop)
        else:
            slots.append((loop, runs))
    walls, middles = [], []
    size = float(np.ptp(corners, axis=0).max())
    while len(fronts) != len(backs):
        more = fronts if len(fronts) > len(backs) else backs
        smallest = min(more, key=len)
        rim = corners[smallest]
        length = float(np.linalg.norm(rim - np.roll(rim, -1, axis=0), axis=1).sum())
        if len(smallest) > mesh_repair.MOST_FILLED_EDGES or length > mesh_repair.MOST_FILLED_RIM_SHARE * size:
            return f"{len(fronts)} openings on the front, {len(backs)} on the back"
        more.remove(smallest)
        middle = len(corners) + len(middles)
        middles.append(rim.mean(0))
        walls += [(smallest[k], smallest[(k + 1) % len(smallest)], middle) for k in range(len(smallest))]
    if middles:
        corners = np.concatenate([corners, np.array(middles)])
    for loop, runs in slots:
        walls += zipped(corners, loop, runs)
    for front, back in paired(corners, fronts, backs):
        walls += bridged(corners, front, back)
    if not walls:
        return "nothing to close"
    walls = np.array(walls, dtype=np.int64)
    if any(tuple(t) in known for t in np.sort(walls, axis=1).tolist()):
        return "the wall would double a triangle"
    result = np.concatenate([cut, walls])
    after_open, _, after_torn = M.open_edge_loops(result)
    _, after_pieces = M.pieces(corners, result)
    if after_open > before_open or after_torn > before_torn:
        return f"the wall left {after_open} open and {after_torn} torn edges"
    if after_pieces > before_pieces:
        return f"the cut split the shape into {after_pieces} pieces"
    volume, _ = M.volume_and_centre(corners, result)
    if volume <= 0:
        return "the carved shape encloses no volume"
    return corners, result, len(walls)


def without_shreds(corners, triangles):
    """The cut surface without its loose shreds: pieces with an open edge and under `object_checks.LEAST_PART_SHARE`
    of the surface (bits of the web the cut left hanging by nothing; a whole part, closed, is kept)."""
    piece, count = M.pieces(corners, triangles)
    if count < 2:
        return triangles
    _, areas = M.face_normals_and_areas(corners, triangles)
    share = np.bincount(piece, weights=areas, minlength=count) / max(areas.sum(), 1e-20)
    keep = np.ones(len(triangles), dtype=bool)
    for p in np.flatnonzero(share < O.LEAST_PART_SHARE):
        mine = piece == p
        if M.open_edge_loops(triangles[mine])[0]:
            keep &= ~mine
    return triangles[keep]


# A run of an opening's edges shorter than this share of the opening, between two longer ones, is noise of the cut's
# edge (where the web's rim, facing sideways, meets the cut) and joins its neighbours.
SHORTEST_RUN_SHARE = 0.1


def smoothed_runs(sides):
    """The opening's edges in runs of one side (front or back): [(first edge, count, side)], one run or two once runs
    shorter than `SHORTEST_RUN_SHARE` of the opening have joined their neighbours, shortest first; None when more
    stay."""
    sides = list(sides)
    n = len(sides)
    for _ in range(n):
        runs = runs_of(sides)
        if len(runs) <= 2:
            return runs
        start, count, side = min(runs, key=lambda r: r[1])
        if count >= SHORTEST_RUN_SHARE * n:
            return None
        for k in range(start, start + count):
            sides[k % n] = not side
    return None


def runs_of(sides):
    n = len(sides)
    if all(s == sides[0] for s in sides):
        return [(0, n, sides[0])]
    start = next(k for k in range(n) if sides[k] != sides[k - 1])
    runs, k = [], start
    while True:
        count = 1
        while sides[(k + count) % n] == sides[k % n] and count < n:
            count += 1
        runs.append((k % n, count, sides[k % n]))
        k += count
        if k % n == start:
            return runs


def zipped(corners, loop, runs):
    """An opening that runs along the web's front edge and back along its back edge (the web ended inside the hole)
    closed by a strip between the two: from the ear at the turn where the front's run starts, each step adds a triangle
    with one new corner from either side, the steps chosen so the strip's new edges are shortest in all (as contours
    are joined by the strip of least total edge length, found over the steps' grid by dynamic programming). Every
    triangle lists its corners in the loop's order, so it runs the loop's way (as a filling must)."""
    n = len(loop)
    start = runs[0][0]
    order = [loop[(start + k) % n] for k in range(n)]
    points = corners[order]

    def length(a, b):
        return float(np.linalg.norm(points[a] - points[b % n]))

    # State (i, j): the strip's open edge from position i (going up from 1) to j (going down from n - 1); it is done
    # when they are neighbours.
    cost = {(1, n - 1): 0.0}
    came = {}
    for gap in range(n - 2, 1, -1):
        for i in range(1, n - gap):
            j = i + gap
            if (i, j) not in cost:
                continue
            for step in ((i + 1, j), (i, j - 1)):
                added = cost[(i, j)] + length(*step)
                if added < cost.get(step, np.inf):
                    cost[step], came[step] = added, (i, j)
    ends = [(i, i + 1) for i in range(1, n - 1) if (i, i + 1) in cost]
    at = min(ends, key=lambda e: cost[e])
    walls = []
    while at != (1, n - 1):
        i0, j0 = came[at]
        walls.append((order[i0], order[i0 + 1], order[j0]) if at[0] == i0 + 1 else (order[i0], order[j0 - 1], order[j0]))
        at = (i0, j0)
    return [(order[n - 1], order[0], order[1])] + walls[::-1]


def paired(corners, fronts, backs):
    """Each opening on the web's front with the one on its back nearest it."""
    left = list(backs)
    for front in fronts:
        middle = corners[front].mean(0)
        nearest = min(range(len(left)), key=lambda k: np.linalg.norm(corners[left[k]].mean(0) - middle))
        yield front, left.pop(nearest)


def bridged(corners, front, back):
    """Two openings, the web's front and back edges (the web closed the hole whole), joined by a strip: the front taken
    forward, the back backward, from their nearest corners, the steps chosen so the strip's new edges are shortest in
    all (as `zipped`)."""
    a, b = corners[front], corners[back]
    i0, j0 = np.unravel_index(np.argmin(np.linalg.norm(a[:, None] - b[None], axis=2)), (len(front), len(back)))
    first = [front[(i0 + k) % len(front)] for k in range(len(front) + 1)]
    second = [back[(j0 - k) % len(back)] for k in range(len(back) + 1)]
    m, n = len(first) - 1, len(second) - 1
    gap = np.linalg.norm(corners[first][:, None] - corners[second][None], axis=2)
    cost = np.full((m + 1, n + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            if i:
                cost[i, j] = min(cost[i, j], cost[i - 1, j] + gap[i, j])
            if j:
                cost[i, j] = min(cost[i, j], cost[i, j - 1] + gap[i, j])
    walls, i, j = [], m, n
    while i or j:
        if i and (not j or cost[i - 1, j] + gap[i, j] <= cost[i, j - 1] + gap[i, j]):
            walls.append((first[i - 1], first[i], second[j]))
            i -= 1
        else:
            walls.append((second[j], second[j - 1], first[i]))
            j -= 1
    return walls[::-1]


def carve_check(record):
    """The carving, shown beside the checks (not counted: the outline and wholeness checks judge what it leaves)."""
    carved_holes = [h for h in record.get("holes", []) if h.get("carved")]
    return Check("webs in the picture's holes cut away", float(len(carved_holes)), 0, "holes", True,
                 record.get("words", ""), counted=False)


# Holes the picture shows are open: each hole of the picture's outline with an inner part (a margin of
# `MARGIN_SHARE` of the height in from its edge), showing the backdrop, is judged from the camera found with the holes'
# inner parts left out (as the carving finds it): the share of its inner part the model's outline covers. A hole too
# small to have an inner part is not judged. Conversely, a hole of the model's outline with an inner part that the
# picture covers is a hole the picture does not have. Limits set from the trials (docs/objects.md, "Holes the picture
# shows"): of the 112 holes judged on the trials and the regression set, the models that pass read at most 8.3 % covered
# (a round table, carved; 6.4 % a chair's back, 2.6 % a jug), the faults 100 % (an early jug, a ring filled by a
# web, a panel between a chair's spindles); no model's own hole was covered by its picture at all. The limits lie in
# the gap, a third.
MOST_HOLE_COVERED_SHARE = 0.33
MOST_EXTRA_HOLE_COVERED_SHARE = 0.33
# An inner part smaller than this (canvas pixels, the outline 384 high: a square of 7 pixels, 1.8 % of the height) is
# not judged: its cover is an edge's misfit as much as a web. The trials' lanterns show the backdrop through a pane of
# glass between two bars as a hole 4.7 % wide whose inner part is 20 pixels; the model's glass covers it, rightly. The
# made-up filled ring's is 108.
LEAST_HOLE_INNER_PX = 50


def holes_open(picture_mask, corners, triangles, picture_rgb=None, outline_view=None):
    """Whether the picture's holes are open in the shape (and the shape has none the picture lacks): a record of each
    hole, judged or not and why, the view and the worst shares. `outline_view`: the outline check's view, used when the
    picture has no hole to judge (its outlines framed alike, so the shape's own holes can be read from it)."""
    corners = np.asarray(corners, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    picture_mask = for_shape(picture_mask)
    aspect = O.canvas_aspect(picture_mask)
    painted, _, placement = O.placed(picture_mask, aspect, by_box=True)
    holes, count = ndimage.label(ndimage.binary_fill_holes(painted) & ~painted)
    margin_px = MARGIN_SHARE * O.OUTLINE_SIDE
    backdrop = backdrop_lab(picture_rgb, picture_mask) if picture_rgb is not None else None
    record = {"holes": [], "extra_holes": [], "judged": 0, "not_judged": 0}
    inner_all = np.zeros_like(painted)
    judged = []
    for k in range(1, count + 1):
        hole = holes == k
        inner = ndimage.distance_transform_edt(hole) > margin_px
        found = {"width": round(O.thickest(hole), 4), "inner_px": int(inner.sum())}
        if not inner.any():
            found.update(judged=False, words="too small to have an inner part: not judged")
        elif inner.sum() < LEAST_HOLE_INNER_PX:
            found.update(judged=False, words=f"its inner part is {int(inner.sum())} pixels, under "
                                             f"{LEAST_HOLE_INNER_PX}: too small to judge")
        else:
            delta = backdrop_delta(picture_rgb, backdrop, inner, placement)
            found["backdrop_delta_e"] = None if delta is None else round(delta, 1)
            if delta is not None and delta > MOST_HOLE_BACKDROP_DELTA_E:
                found.update(judged=False, words=f"its colour lies {delta:.0f} (delta E) from the backdrop's: a part "
                                                 "the cut-out missed, not a hole; not judged")
            else:
                found["judged"] = True
                judged.append((found, inner))
                inner_all |= inner
        record["holes"].append(found)
    if judged or outline_view is None:
        view = O.view_search(painted, aspect, corners, triangles, by_box=True, counted=~inner_all, by_overlap=True)
        made = view.made
        made = np.roll(made, matched_shift(ndimage.binary_fill_holes(painted), ndimage.binary_fill_holes(made)), axis=1)
        picture_solid = painted
    else:
        view, made, picture_solid = outline_view, outline_view.made, outline_view.painted
    record["view"] = {"turn_deg": view.turn, "tilt_deg": view.tilt, "distance": view.distance}
    for found, inner in judged:
        found["covered"] = round(float((made & inner).sum() / inner.sum()), 4)
        found["words"] = f"{found['covered']:.0%} of its inner part covered by the model"
    # The converse: the model's own holes with an inner part, and how much of that inner part the picture covers.
    model_holes, model_count = ndimage.label(ndimage.binary_fill_holes(made) & ~made)
    for k in range(1, model_count + 1):
        inner = ndimage.distance_transform_edt(model_holes == k) > margin_px
        if inner.sum() >= LEAST_HOLE_INNER_PX:
            covered = float((picture_solid & inner).sum() / inner.sum())
            record["extra_holes"].append({"width": round(O.thickest(model_holes == k), 4),
                                          "inner_px": int(inner.sum()), "covered_by_picture": round(covered, 4)})
    record["judged"] = len(judged)
    record["not_judged"] = sum(not h["judged"] for h in record["holes"])
    record["worst"] = max((f["covered"] for f, _ in judged), default=0.0)
    record["worst_extra"] = max((h["covered_by_picture"] for h in record["extra_holes"]), default=0.0)
    return record


def holes_check(record):
    """The holes, as a check: counted when a hole of the picture was judged or the model has a hole of its own; else
    shown, not counted ("not judged" is never a pass)."""
    worst, extra = record["worst"], record["worst_extra"]
    judged = record["judged"]
    passed = worst <= MOST_HOLE_COVERED_SHARE and extra <= MOST_EXTRA_HOLE_COVERED_SHARE
    words = (f"{judged} hole{'s' if judged != 1 else ''} of the picture judged, the most covered by the model "
             f"{worst:.0%} of its inner part (at most {MOST_HOLE_COVERED_SHARE:.0%}); "
             if judged else "no hole of the picture wide enough to judge; ")
    words += f"{record['not_judged']} not judged; "
    words += (f"the model's own holes: the most covered by the picture {extra:.0%} (at most "
              f"{MOST_EXTRA_HOLE_COVERED_SHARE:.0%})" if record["extra_holes"] else "the model has no hole of its own")
    return Check("holes the picture shows are open", worst * 100, MOST_HOLE_COVERED_SHARE * 100, "%", passed, words,
                 counted=bool(judged or record["extra_holes"]))
