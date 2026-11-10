"""A thing that stands, brought onto one floor: when a leg comes out a little short, the others are trimmed to the
plane its foot stands on.

The shape model guesses a thin leg's end from a few pixels, so one leg may end a centimetre or two above the others
(the third trial's round table: one of three legs 24.6 mm short, its top dead level, so it settled 3.6 degrees onto
the short leg). The established fix, as a furniture maker levels a rocking table or a 3D print's base is flattened
(a plane cut, then the cut capped), is to find the floor the object should stand on, a plane square to its up axis,
and cut away what reaches below it. Here that plane is the lowest at which every foot touches and the object stands
(the base checks pass); only a few centimetres are ever cut (`MOST_TRIM_SHARE`), so a leg truly missing stays a
failure, and a shape made leaning whole (its flat top square to the floor its feet settle it on, not to its own up:
`leaning`) is not trimmed at all and stays a failure. The finish does this to the shape it exports; the shape the model made is kept as it
was, and the sidecar says what was cut.
"""

import numpy as np

from . import mesh_repair
from . import mesh_tools as M
from . import object_checks as O
from .picture_checks import Check

# At most this share of the height is cut from the feet: on a table 65 cm high, 3.25 cm. It is the height within
# which the base checks look for a foot (`object_checks.FOOT_SEARCH_SHARE`); a foot short by more is a part missing.
MOST_TRIM_SHARE = O.FOOT_SEARCH_SHARE
# Low places are looked for this far up (a share of the height), so a foot short beyond the limit is named.
FEET_SEARCH_SHARE = 3 * MOST_TRIM_SHARE
# Corners this close to the cutting plane (a share of the object's size) are moved onto it, so no sliver is cut off.
ON_PLANE_SHARE = 1e-7
DOWN = np.array([0.0, -1.0, 0.0])


def trim_limit_m(corners):
    return MOST_TRIM_SHARE * float(np.ptp(corners[:, 1]))


def base_passes(measures):
    return all(c.passed for c in O.base_checks(measures))


def levelled(corners, triangles):
    """The surface trimmed onto one floor when a foot is short by at most the limit: (corners, triangles, record).
    The record says what was found and done: `trimmed`, `trimmed_m` (how much was cut from the other feet; the object
    is that much lower), the feet's heights above the lowest, the limit, the base measures before and words."""
    before = O.base_measures(corners, triangles)
    limit = trim_limit_m(corners)
    record = {"trimmed": False, "trimmed_m": 0.0, "limit_m": round(limit, 4), "feet_short_m": [],
              "before": {"tilt_deg": round(before["tilt_deg"], 3), "margin": round(before["margin"], 4),
                         "lifted_m": [round(g, 4) for g in before["lifted_m"]]}}
    if base_passes(before):
        record["words"] = "it stands as made: nothing trimmed"
        return corners, triangles, record
    lean = leaning(corners, triangles)
    record["top"] = lean
    if lean["leans"]:
        record["words"] = (f"the whole object leans {lean['made_deg']:.1f}° as made: its flat top is square to the floor "
                           f"it settles on ({lean['settled_deg']:.1f}° off it), not to its own up, so no foot is short; "
                           "left as made")
        return corners, triangles, record
    # Every low place of the underside above the lowest, measured with the object upright as made.
    short = O.lifted_feet(corners, triangles, DOWN, None, search_share=FEET_SEARCH_SHARE)
    record["feet_short_m"] = [round(g, 4) for g in sorted(short)]
    for height in sorted(short):
        if height > limit:
            record["words"] = (f"a foot {height * 1000:.0f} mm short, more than the {limit * 1000:.0f} mm the finish "
                               "trims: left as made")
            return corners, triangles, record
        cut_corners, cut_triangles = cut_below(corners, triangles, float(corners[:, 1].min()) + height)
        if base_passes(O.base_measures(cut_corners, cut_triangles)):
            record.update(trimmed=True, trimmed_m=round(float(height), 5),
                          words=f"the other feet trimmed {height * 1000:.1f} mm to the short foot's floor (at most "
                                f"{limit * 1000:.0f} mm), so every foot stands on it")
            return cut_corners, cut_triangles, record
    record["words"] = ("no foot within reach is short: trimming to one floor would not make it stand" if not short else
                       "trimmed to every short foot's floor in turn, it still does not stand: left as made")
    return corners, triangles, record


# A top: the faces within this angle of the object's up as made, or of the up it settles to (without it, a table turned
# 3 degrees whole was trimmed 21.7 mm and passed, its top still 3 degrees off level). Its evidence
# counts when such faces are at least this share of the surface; the object leans when its top is square to the floor
# it settles on and off its own up by at least `LEAST_LEAN_DEG` more.
TOP_CONE_DEG = 10.0
LEAST_TOP_SHARE = 0.03
LEAST_LEAN_DEG = 1.0


def leaning(corners, triangles):
    """Whether the object leans whole as made, rather than standing on a foot too short: a leg too short leaves the
    body square to the object's own up (the third trial's round table: its top dead level, one foot 24.6 mm up), a
    shape turned whole leaves it square to the floor its feet settle it onto. The body's up is read from its flat top
    (the established cue of a man-made thing's upright: its large horizontal faces; Fu, Cohen-Or, Dror and Sheffer,
    2008): the area-weighted mean normal of the faces within `TOP_CONE_DEG` of either up. A thing with no flat top
    gives no evidence and is never called leaning. Returns {"leans", "made_deg", "settled_deg", "share"}."""
    normal, _, _, _ = O.resting_face(corners, triangles)
    found = {"leans": False, "made_deg": 0.0, "settled_deg": 0.0, "share": 0.0}
    if normal is None:
        return found
    settled_up = -np.asarray(normal, dtype=np.float64)
    made_up = np.array([0.0, 1.0, 0.0])
    normals, areas = M.face_normals_and_areas(corners, triangles)
    least = np.cos(np.radians(TOP_CONE_DEG))
    top = (normals @ made_up >= least) | (normals @ settled_up >= least)
    found["share"] = round(float(areas[top].sum() / max(areas.sum(), 1e-20)), 4)
    if found["share"] < LEAST_TOP_SHARE:
        return found
    mean = (normals[top] * areas[top, None]).sum(0)
    mean /= max(np.linalg.norm(mean), 1e-20)
    made_deg = float(np.degrees(np.arccos(np.clip(mean @ made_up, -1, 1))))
    settled_deg = float(np.degrees(np.arccos(np.clip(mean @ settled_up, -1, 1))))
    found.update(made_deg=round(made_deg, 3), settled_deg=round(settled_deg, 3),
                 leans=bool(made_deg - settled_deg >= LEAST_LEAN_DEG))
    return found


def trim_check(record):
    """The trim, shown beside the checks (not counted: the base checks judge the shape it leaves)."""
    beyond = not record["trimmed"] and any(g > record["limit_m"] for g in record["feet_short_m"])
    return Check("feet trimmed to one floor", record["trimmed_m"] * 1000, round(record["limit_m"] * 1000, 1), "mm",
                 not beyond, record["words"], counted=False)


def cut_below(corners, triangles, height):
    """The surface with everything below the level plane at `height` cut away, each opening the cut leaves closed by a
    flat cap, and set down so the plane is the ground (y = 0)."""
    corners = np.asarray(corners, dtype=np.float64).copy()
    triangles = np.asarray(triangles, dtype=np.int64)
    size = float(np.ptp(corners, axis=0).max())
    above = corners[:, 1] - height
    on_plane = np.abs(above) <= ON_PLANE_SHARE * size
    above[on_plane] = 0.0
    corners[on_plane, 1] = height
    side = np.sign(above)
    sides = side[triangles]
    kept = [triangles[(sides >= 0).all(1)]]
    crossing = triangles[(sides.max(1) > 0) & (sides.min(1) < 0)]
    new_corners, made_at = [], {}

    def crossing_point(a, b):
        key = (min(a, b), max(a, b))
        if key not in made_at:
            share = above[a] / (above[a] - above[b])
            point = corners[a] + share * (corners[b] - corners[a])
            point[1] = height
            made_at[key] = len(corners) + len(new_corners)
            new_corners.append(point)
        return made_at[key]

    clipped = []
    for triangle in crossing.tolist():
        # The triangle's part above the plane (a triangle or a quadrilateral), corners in its own order.
        polygon = []
        for k in range(3):
            a, b = triangle[k], triangle[(k + 1) % 3]
            if side[a] >= 0:
                polygon.append(a)
            if side[a] * side[b] < 0:
                polygon.append(crossing_point(a, b))
        clipped += [[polygon[0], polygon[k], polygon[k + 1]] for k in range(1, len(polygon) - 1)]
    if clipped:
        kept.append(np.array(clipped, dtype=np.int64))
    if new_corners:
        corners = np.concatenate([corners, np.array(new_corners)])
    triangles = np.concatenate(kept)
    caps, middles = [], []
    loops = [loop for loop in mesh_repair.boundary_loops(triangles)
             if np.all(np.abs(corners[loop, 1] - height) <= ON_PLANE_SHARE * size)]
    for loop in with_holes_bridged(corners, loops):
        ears = ear_clipped(corners[loop][:, [0, 2]])
        if ears is not None:
            caps += [[loop[a], loop[b], loop[c]] for a, b, c in ears]
        else:
            # A loop that crosses itself: a fan round its middle, as the repair fills a hole.
            middle = len(corners) + len(middles)
            middles.append(corners[loop].mean(0))
            caps += [[loop[k], loop[(k + 1) % len(loop)], middle] for k in range(len(loop))]
    if middles:
        corners = np.concatenate([corners, np.array(middles)])
    if caps:
        triangles = np.concatenate([triangles, np.array(caps, dtype=np.int64)])
    corners, triangles = M.compacted(corners, triangles)
    corners[:, 1] -= height
    return corners, triangles


def inside_polygon(point, polygon):
    """Whether a point lies inside a polygon (x, z rows), by the crossings of a ray to +x."""
    x, z = polygon[:, 0], polygon[:, 1]
    x2, z2 = np.roll(x, -1), np.roll(z, -1)
    crosses = (z > point[1]) != (z2 > point[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        at = x + (point[1] - z) * (x2 - x) / (z2 - z)
    return bool(np.count_nonzero(crosses & (at > point[0])) % 2)


def with_holes_bridged(corners, loops):
    """The cut's loops to cap, each loop inside another (the inner edge of a cut foot ring, a hollow leg) joined to the
    one round it by a bridge, so the two are capped as one ring, not as two discs over each other (each loop capped on
    its own makes a sheet of no thickness over a ring's hollow).
    The bridge runs from the hole's corner furthest along +x to the nearest corner of the loop round it that it can
    reach without crossing an edge (the standard way to clip ears of a polygon with holes: Eberly, "Triangulation by
    ear clipping", 2002)."""
    flat = [corners[loop][:, [0, 2]] for loop in loops]
    around = []
    for k, polygon in enumerate(flat):
        holders = [j for j, other in enumerate(flat) if j != k and inside_polygon(polygon[0], other)]
        # The loop directly round it: the smallest that holds it; a loop held by an even count is an outer one.
        around.append(min(holders, key=lambda j: abs(signed_area(flat[j]))) if len(holders) % 2 else None)
    joined = []
    for k, loop in enumerate(loops):
        if around[k] is not None:
            continue
        merged = list(loop)
        holes = sorted((j for j in range(len(loops)) if around[j] == k), key=lambda j: -flat[j][:, 0].max())
        for j in holes:
            merged = bridged_hole(corners, merged, list(loops[j]))
        joined.append(merged)
    return joined


def signed_area(polygon):
    x, z = polygon[:, 0], polygon[:, 1]
    return float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z) / 2)


def bridged_hole(corners, outer, hole):
    """The outer loop with the hole spliced in along a bridge: outer to P, round the hole from M back to M, P, on."""
    points = corners[:, [0, 2]]
    m = max(range(len(hole)), key=lambda i: points[hole[i], 0])
    start = points[hole[m]]
    edges = [(a, b) for ring in (outer, hole) for a, b in zip(ring, ring[1:] + ring[:1])]
    order = sorted(range(len(outer)), key=lambda i: np.linalg.norm(points[outer[i]] - start))
    chosen = order[0]
    for i in order:
        end = points[outer[i]]
        if not any(segments_cross(start, end, points[a], points[b]) for a, b in edges
                   if hole[m] not in (a, b) and outer[i] not in (a, b)):
            chosen = i
            break
    return outer[:chosen + 1] + hole[m:] + hole[:m + 1] + outer[chosen:]


def segments_cross(p, q, a, b):
    """Whether segments pq and ab cross at a point inside both."""
    def side(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    return side(p, q, a) * side(p, q, b) < 0 and side(a, b, p) * side(a, b, q) < 0


def turn_of(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def ear_clipped(flat):
    """A simple polygon's triangles by clipping ears, wound as the polygon runs, as index triples; None when no ear can
    be found (the polygon crosses itself)."""
    count = len(flat)
    x, y = flat[:, 0], flat[:, 1]
    winding = np.sign(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) or 1.0
    left = list(range(count))
    place = {}
    for i, point in enumerate(map(tuple, flat.tolist())):
        place.setdefault(point, set()).add(i)
    twins = [place[tuple(point)] for point in flat.tolist()]
    ears = []
    while len(left) > 3:
        for k in range(len(left)):
            a, b, c = left[k - 1], left[k], left[(k + 1) % len(left)]
            if winding * turn_of(flat[a], flat[b], flat[c]) <= 0:
                continue  # a reflex or straight corner is no ear
            # Corners at the same place as the ear's (a bridge's two ends, each met twice) do not block it.
            near = twins[a] | twins[b] | twins[c]
            others = np.array([i for i in left if i not in near])
            if len(others) and inside_any(flat[others], flat[a], flat[b], flat[c], winding):
                continue
            ears.append((a, b, c))
            del left[k]
            break
        else:
            return None
    ears.append(tuple(left))
    return ears


def inside_any(points, a, b, c, winding):
    """Whether any point lies inside or on the triangle abc (wound `winding`)."""
    def side(p, q):
        return winding * ((q[0] - p[0]) * (points[:, 1] - p[1]) - (q[1] - p[1]) * (points[:, 0] - p[0]))
    return bool(np.any((side(a, b) >= 0) & (side(b, c) >= 0) & (side(c, a) >= 0)))
