"""Whether the six painted views agree, and one view per place for the bake (docs/trials/2026-11-11-030-diagnosis.md).

The view painter paints the six views together, but with some seeds one view comes back out of family: 0.2.0's sealed
run made a white teapot with its right side black and a pale glass bottle dark on one side, and every check passed.
Neighbouring side views see a strip of the same surface (round 45 degrees), so a view that disagrees with both its
neighbours there is the odd one: the photo-consistency idea of texturing from photographs (Waechter, Moehrle and
Goesele 2014 drop a view whose colour is an outlier among the views of a face), between the view painter's views. The
same views painted from another seed agreed (the teapot 0.91 to 0.03, the bottle 0.81 to 0.02): the pipeline paints
them again from the next seed and keeps the try that agrees best, as the touch-up keeps its best fill.

Where two views see a surface, the bake mixed each texel's nearest painted points from both, and two unlike views
interleaved into a stipple. Each painted point is now used only where its own view sees the surface about as squarely
as any of the six does (Lempitsky and Ivanov 2007, Waechter et al. 2014: one view per place; MV-Adapter's own
projection weights its views by the cosine to the surface).
"""

import numpy as np

# The four side views in paint.painted's order (front, a side, back, the other side), each with its neighbours.
NEIGHBOURS = ((0, 1), (1, 2), (2, 3), (3, 0))
SIDE_VIEWS = 4
# Two views' points are the same place within this many of the cloud's spacings, facing within about 25 degrees.
REACH_SPACINGS = 2.5
SAME_FACING = 0.9
# A pair with fewer matched points than this (a thin thing's edge) is not judged.
LEAST_PAIRS = 200
# Two colours are unlike over this CIE76 delta E (10 is plainly seen; white against black is about 100).
UNLIKE_DELTA_E = 30.0
# A view unlike both its neighbours over this share of their common strip is painted again. Measured on the paint of
# 167 objects of the trials: the four single-view faults found (the teapots, the bottles) scored 0.52 to 0.91; 17
# objects reach it, the 13 others polished metal or iron (copper pots, a sword, spears, a shield's boss, a pickaxe),
# whose views differ by their reflections. A repaint costs time, not quality: the try that agrees best is kept.
REPAINT_FROM = 0.5
# At most this many more seeds.
MOST_REPAINTS = 2
# A painted point is left out of the bake where another view painted the same place more squarely by over this (a
# narrow band at the 45-degree seam, where two views are equally square, keeps both).
SELECTION_MARGIN = 0.05
# Two views' points are on the same surface when their normals are within about 60 degrees (the bake's own test is
# looser, 75 degrees: bake.SAME_FACING).
SAME_SURFACE = 0.5


def spacing(points, sample=3000):
    """The cloud's usual spacing: the median distance to the nearest other point, over a sample."""
    from scipy.spatial import cKDTree

    if len(points) < 2:
        return 0.0
    probe = points[:: max(1, len(points) // sample)]
    return float(np.median(cKDTree(points).query(probe, k=2)[0][:, 1]))


def unlike_share(points, normals, labs, a, b, reach):
    """Of view a's points that view b also painted (b's nearest point within `reach`, facing the same way), the share
    whose colours differ by more than `UNLIKE_DELTA_E`; None when too few are matched."""
    from scipy.spatial import cKDTree

    if len(a) == 0 or len(b) == 0:
        return None
    distances, nearest = cKDTree(points[b]).query(points[a], distance_upper_bound=reach)
    found = np.isfinite(distances)
    nearest = np.where(found, nearest, 0)
    found &= np.einsum("pd,pd->p", normals[a], normals[b][nearest]) > SAME_FACING
    if found.sum() < LEAST_PAIRS:
        return None
    difference = np.linalg.norm(labs[a][found] - labs[b][nearest][found], axis=1)
    return float((difference > UNLIKE_DELTA_E).mean())


def measured(points, normals, colours, view_of):
    """How unlike each side view is to its neighbours. `view_of`: each point's view (0-5; another number for a point no
    view painted, the touch-up's). Returns {"pairs": {"0-1": share or None, ...}, "views": [each side view's lesser
    pair share, or None], "worst": the largest of those (0 when none is judged)}."""
    from skimage.color import rgb2lab

    points, normals = np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64)
    view_of = np.asarray(view_of)
    labs = rgb2lab(np.clip(np.asarray(colours, dtype=np.float64), 0, 1)[None])[0]
    members = [np.flatnonzero(view_of == v) for v in range(SIDE_VIEWS)]
    reach = REACH_SPACINGS * spacing(points[members[0]] if len(members[0]) > 1 else points)
    pairs = {f"{a}-{b}": unlike_share(points, normals, labs, members[a], members[b], reach) for a, b in NEIGHBOURS}
    views = []
    for view in range(SIDE_VIEWS):
        found = [pairs[f"{a}-{b}"] for a, b in NEIGHBOURS if view in (a, b)]
        views.append(min(found) if all(f is not None for f in found) else None)
    judged = [v for v in views if v is not None]
    return {"pairs": {k: (round(v, 4) if v is not None else None) for k, v in pairs.items()},
            "views": [round(v, 4) if v is not None else None for v in views],
            "worst": round(max(judged), 4) if judged else 0.0}


def kept_try(tries):
    """The index of the try whose most unlike view is least unlike (the first of equals). `tries`: `measured`'s."""
    return min(range(len(tries)), key=lambda k: (tries[k]["worst"], k))


def one_view_per_place(points, normals, view_of, directions, margin=SELECTION_MARGIN):
    """Which painted points the bake uses: a view's point is left out where another view painted the same place (a
    point of it within `REACH_SPACINGS` of the cloud's spacing) and saw it more squarely by over `margin`, so the bake
    takes each place from the view that sees it best and two views do not interleave; a place only one view painted
    keeps that view's points, whatever its facing (a squarer view that is blocked there painted nothing to replace
    them). Every point no view painted (the touch-up's fill) is kept. `directions`: each view's unit vector towards
    its camera, in the points' frame."""
    from scipy.spatial import cKDTree

    points, normals = np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64)
    view_of, directions = np.asarray(view_of), np.asarray(directions, dtype=np.float64)
    keep = np.ones(len(points), dtype=bool)
    views = [np.flatnonzero(view_of == v) for v in range(len(directions))]
    painted = np.concatenate(views)
    if len(painted) < 2:
        return keep
    # The spacing within one view (two views' points may fall on the same place).
    reach = REACH_SPACINGS * spacing(points[max(views, key=len)])
    facing = np.zeros(len(points))
    for v, members in enumerate(views):
        facing[members] = normals[members] @ directions[v]
    for u, better in enumerate(views):
        if len(better) == 0:
            continue
        tree = cKDTree(points[better])
        for v, members in enumerate(views):
            if v == u or len(members) == 0:
                continue
            distances, nearest = tree.query(points[members], distance_upper_bound=reach)
            found = np.isfinite(distances)
            nearest = np.where(found, nearest, 0)
            # The same surface, not a thin wall's other side.
            found &= np.einsum("pd,pd->p", normals[members], normals[better[nearest]]) > SAME_SURFACE
            squarer = np.zeros(len(members), dtype=bool)
            squarer[found] = facing[better[nearest[found]]] > facing[members[found]] + margin
            keep[members[squarer]] = False
    return keep
