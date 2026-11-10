"""The standard clean-up of a shape before it is checked, each step counted for the report.

Measured on the first trial's sixteen shapes (2026-11-09): every hole was a seam of corners at the same place that
were never joined (four open edges each), and every torn edge belonged to a duplicate triangle (the same three
corners twice, as an edge-collapse simplifier leaves where it flattens a thin fold). So, in order:

1. corners at the same place (within `SAME_PLACE_SHARE` of the object's size) joined into one;
2. triangles that lost a corner to that (two corners now one) dropped;
3. duplicate triangles: a pair facing opposite ways is a fold of no thickness and both go; a pair facing the same
   way is one triangle twice and one goes;
4. corners where two sheets touch (an edge of three or more triangles, or a pinched corner) split, one copy for each
   fan of triangles round them;
5. holes still open filled, each with a fan round its middle, when its edge is at most `MOST_FILLED_EDGES` long and
   its rim at most `MOST_FILLED_RIM_SHARE` of the object's size; larger holes are left for the check to fail;
6. hollows dropped: closed shells inside the main piece and wound inward (the trial's chair had one in each leg, its
   lantern a sealed 129 cm3 chamber, 14 % of its surface, which no view could paint).
"""

import numpy as np

from . import mesh_tools as M

# Corners closer than this share of the object's largest size are the same corner (a millionth: well under the
# shape grid's cell, a 1/256 of the size).
SAME_PLACE_SHARE = 1e-6
# A hole is filled when its rim has at most this many edges and is at most this share of the object's size long.
# Measured on the first trial: after joining, the one hole left of sixteen shapes was the barrel's, 8 edges and
# 22.8 mm round (2.5 % of its size, about two triangles); the limits are about twice that. A missing part is larger
# (a lost leg's stump on a chair is some 13 % of its size round) and fails the check.
MOST_FILLED_EDGES = 24
MOST_FILLED_RIM_SHARE = 0.05


def repaired(corners, triangles):
    """The surface cleaned (corners, triangles) and what was done: a dict of counts."""
    corners = np.asarray(corners, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    done = {}
    size = float(np.ptp(corners, axis=0).max()) if len(corners) else 1.0
    corners, triangles, done["corners_joined"] = joined(corners, triangles, SAME_PLACE_SHARE * size)
    triangles, done["degenerate_dropped"] = without_degenerate(triangles)
    triangles, done["folds_dropped"], done["doubles_dropped"] = without_duplicates(triangles)
    corners, triangles, done["corners_split"] = split_at_torn_edges(corners, triangles)
    corners, triangles, done["holes_filled"], done["holes_left"] = holes_filled(corners, triangles, size)
    triangles, hollows = without_hollows(corners, triangles)
    done["hollows_removed"] = len(hollows)
    done["hollows"] = hollows
    corners, triangles = M.compacted(corners, triangles)
    return corners, triangles, done


def without_hollows(corners, triangles):
    """Closed shells inside the main piece and wound inward (a hollow inside a chair's leg, a lantern's sealed glass
    chamber) dropped: no one sees them, no view can paint them, and they are not parts. Returns (triangles, the
    hollows: their volume in cm3 and their middle)."""
    piece, count = M.pieces(corners, triangles)
    if count < 2:
        return triangles, []
    _, areas = M.face_normals_and_areas(corners, triangles)
    main = int(np.bincount(piece, weights=areas).argmax())
    main_triangles = triangles[piece == main]
    keep = np.ones(len(triangles), dtype=bool)
    hollows = []
    for p in range(count):
        if p == main:
            continue
        mine = piece == p
        volume, _ = M.volume_and_centre(corners, triangles[mine])
        middle = corners[np.unique(triangles[mine])].mean(0)
        if volume < 0 and M.open_edge_loops(triangles[mine])[0] == 0 and inside(middle, corners, main_triangles):
            keep &= ~mine
            hollows.append({"volume_cm3": round(-volume * 1e6, 2), "at": [round(float(x), 3) for x in middle]})
    return triangles[keep], hollows


# Rays in directions no surface made on a grid lines up with (a ray along an axis can run down a seam and count a
# crossing twice or not at all); the majority of their answers is taken.
RAY_DIRECTIONS = np.array([[1.0, 0.1234, 0.0567], [-0.0731, 1.0, 0.2113], [0.1871, -0.0913, 1.0]])


def inside(point, corners, triangles):
    """Whether a point lies inside a closed surface: most of three rays from it cross the surface an odd number of
    times."""
    a, b, c = (corners[triangles[:, k]] for k in range(3))
    e1, e2 = b - a, c - a
    s = point - a
    q = np.cross(s, e1)
    odd = 0
    for direction in RAY_DIRECTIONS / np.linalg.norm(RAY_DIRECTIONS, axis=1, keepdims=True):
        h = np.cross(direction, e2)
        det = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(det) > 1e-14
        f = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        u = f * np.einsum("ij,ij->i", s, h)
        v = f * (q @ direction)
        t = f * np.einsum("ij,ij->i", e2, q)
        odd += int((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)).sum() % 2)
    return odd >= 2


def joined(corners, triangles, tolerance):
    """Corners at the same place joined (`mesh_tools.joined`): (corners, triangles, how many were joined)."""
    return M.joined(corners, triangles, tolerance)


def without_degenerate(triangles):
    keep = (triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2]) & \
        (triangles[:, 0] != triangles[:, 2])
    return triangles[keep], int((~keep).sum())


def winding_key(triangle):
    """The triangle's corners turned so the least is first: equal for the same triangle the same way round."""
    k = int(np.argmin(triangle))
    return tuple(triangle[k:].tolist() + triangle[:k].tolist())


def without_duplicates(triangles):
    """Triangles on the same three corners: an opposite pair both dropped, a same-way pair kept once."""
    sorted_corners = np.sort(triangles, axis=1)
    _, group, counts = np.unique(sorted_corners, axis=0, return_inverse=True, return_counts=True)
    group = group.reshape(-1)
    keep = np.ones(len(triangles), dtype=bool)
    folds = doubles = 0
    for g in np.flatnonzero(counts > 1):
        members = np.flatnonzero(group == g)
        ways = {}
        for m in members:
            ways.setdefault(winding_key(triangles[m]), []).append(m)
        sides = list(ways.values())
        # One of each way pairs off as a fold; what is left over of one way is kept once.
        if len(sides) == 2:
            pairs = min(len(sides[0]), len(sides[1]))
            for side in sides:
                keep[side[:pairs]] = False
            folds += pairs
            rest = [m for side in sides for m in side[pairs:]]
        else:
            rest = sides[0]
        if len(rest) > 1:
            keep[rest[1:]] = False
            doubles += len(rest) - 1
    return triangles[keep], folds, doubles


def split_at_torn_edges(corners, triangles):
    """Two sheets that touch along an edge (an edge of three or more triangles) or at a corner made separate: each
    corner gets one copy for each fan of its triangles joined through edges of exactly two triangles (the standard
    split of non-manifold corners). Returns (corners, triangles, how many corners were added)."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    m = len(triangles)
    if m == 0:
        return corners, triangles, 0
    n = len(corners)
    slots = np.repeat(np.arange(3)[None], m, 0)
    first, second = triangles, np.roll(triangles, -1, axis=1)
    owner = np.repeat(np.arange(m), 3)
    a, b = first.reshape(-1), second.reshape(-1)
    slot_a, slot_b = slots.reshape(-1), np.roll(slots, -1, axis=1).reshape(-1)
    keys = np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b)
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    _, starts, counts = np.unique(keys_sorted, return_index=True, return_counts=True)
    pairs = starts[counts == 2]
    one, two = order[pairs], order[pairs + 1]

    def node(occurrence, corner):
        """The node (triangle * 3 + slot) of `corner` in the triangle of each edge occurrence."""
        slot = np.where(a[occurrence] == corner, slot_a[occurrence], slot_b[occurrence])
        return owner[occurrence] * 3 + slot

    low = np.minimum(a[one], b[one])
    high = np.maximum(a[one], b[one])
    rows = np.concatenate([node(one, low), node(one, high)])
    cols = np.concatenate([node(two, low), node(two, high)])
    graph = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(3 * m, 3 * m))
    count, label = connected_components(graph, directed=False)
    flat = triangles.reshape(-1)
    new_corners = np.zeros((count, 3))
    new_corners[label] = corners[flat]
    used = len(np.unique(flat))
    return new_corners, label.reshape(m, 3), int(count - used)


def boundary_loops(triangles):
    """The surface's holes as loops of corners, each in the order a filling triangle runs its edges (the reverse of
    the triangles round it); loops that do not close (a torn rim) are left out."""
    directed = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    unique, counts = M.edge_uses(triangles)
    open_set = {tuple(e) for e in unique[counts == 1].tolist()}
    following = {}
    for a, b in directed.tolist():
        if (min(a, b), max(a, b)) in open_set:
            following[b] = a  # the filling runs b -> a
    loops, seen = [], set()
    for start in list(following):
        if start in seen:
            continue
        loop, at = [], start
        while at not in seen and at in following:
            seen.add(at)
            loop.append(at)
            at = following[at]
        if at == start and len(loop) >= 3:
            loops.append(loop)
    return loops


def holes_filled(corners, triangles, size):
    """Small holes filled with a fan round their middle: (corners, triangles, filled, left open)."""
    loops = boundary_loops(triangles)
    new_corners, new_triangles, filled = [], [], 0
    for loop in loops:
        rim = corners[loop]
        length = float(np.linalg.norm(rim - np.roll(rim, -1, axis=0), axis=1).sum())
        if len(loop) > MOST_FILLED_EDGES or length > MOST_FILLED_RIM_SHARE * size:
            continue
        middle = len(corners) + len(new_corners)
        new_corners.append(rim.mean(0))
        new_triangles += [[loop[k], loop[(k + 1) % len(loop)], middle] for k in range(len(loop))]
        filled += 1
    if new_corners:
        corners = np.concatenate([corners, np.array(new_corners)])
        triangles = np.concatenate([triangles, np.array(new_triangles, dtype=np.int64)])
    return corners, triangles, filled, len(loops) - filled
