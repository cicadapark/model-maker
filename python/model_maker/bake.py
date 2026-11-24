"""The object's colours baked into one texture, and how much of it the painted views covered.

The surface is cut into charts: joined triangles facing within about 55 degrees of one of the six axis directions,
each laid flat by projecting it along that axis, so neighbouring texels are neighbouring surface and the seams are
few. The charts are packed in rows into one square, the texels per metre the same everywhere. Each texel takes its
colour from the painted views' pixels near its point on the surface that face the same way (the colour cloud);
where no painted pixel is near, it takes the colour spread to its corners from their neighbours, and counts as not
covered. A triangle thinner than a texel holds no texel's centre; it colours the texels it touches that no triangle
holds, so it still reads its own colour. The margins round each chart are then filled outward, and every texel still
empty takes the nearest coloured texel's colour, so no smaller copy or mip level mixes in black.
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from . import mesh_tools as M
from . import render

DEFAULT_SIDE = 2048
MARGIN_TEXELS = 3
# Packing starts at this share of the square covered by the surface's own area and shrinks until the charts fit.
FIRST_FILL = 0.85
SHRINK = 0.92
MOST_PACKING_TRIES = 60
# A painted pixel counts for a texel when it lies within this many of the cloud's own spacings, and its normal is
# within about 75 degrees of the surface's.
CLOUD_REACH_SPACINGS = 2.5
SAME_FACING = 0.25
CLOUD_NEIGHBOURS = 4
# The fill reaches this far out from each chart.
FILL_ROUNDS = MARGIN_TEXELS + 2
# A triangle that owns no texel centre (thinner than a texel) is walked along its edges in steps this long, and colours
# the texels it touches that no triangle owns: so every triangle reads its own colour. Such slivers, in charts of one
# to three triangles, were left the texture's black.
EDGE_STEP_TEXELS = 0.5
COVERED_COLOUR = (90, 170, 100)
FILLED_COLOUR = (215, 70, 60)
HIDDEN_COLOUR = (200, 200, 200)
# The coverage counts each texel by how much of the time it is seen: the share of these many directions round the
# object it is in view from (drawn at this side; a texel within this many pixels' depth of the nearest surface is in
# view). The first trial's jug left 17 % of its texels unpainted, most of them inside its neck, seen only looking
# into its mouth: one sees them seldom. A texel seen from at least this share of the
# directions is drawn full strength in the coverage picture.
SEEN_DIRECTIONS = 32
SEEN_SIDE = 384
SEEN_SLACK_PIXELS = 2.0
FULLY_SEEN_SHARE = 0.3


def charts(corners, triangles):
    """Each triangle's chart and each chart's axis (0-5: +x, -x, +y, -y, +z, -z)."""
    normals, _ = M.face_normals_and_areas(corners, triangles)
    label = np.argmax(np.concatenate([normals, -normals], 1)[:, [0, 3, 1, 4, 2, 5]], axis=1)
    n_corners = len(corners)
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edges.sort(axis=1)
    keys = edges[:, 0].astype(np.int64) * n_corners + edges[:, 1]
    owner = np.tile(np.arange(len(triangles)), 3)
    order = np.argsort(keys, kind="stable")
    keys, owner = keys[order], owner[order]
    same = np.flatnonzero(keys[1:] == keys[:-1])
    a, b = owner[same], owner[same + 1]
    joined = label[a] == label[b]
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    graph = sparse.coo_matrix((np.ones(joined.sum()), (a[joined], b[joined])),
                              shape=(len(triangles), len(triangles)))
    _, chart = connected_components(graph, directed=False)
    chart_axis = np.zeros(chart.max() + 1, dtype=np.int64)
    chart_axis[chart] = label
    return chart, chart_axis


# Each axis's flat coordinates (two of x, y, z), so a chart lies flat looking along its axis.
FLAT = {0: (2, 1), 1: (2, 1), 2: (0, 2), 3: (0, 2), 4: (0, 1), 5: (0, 1)}


def shelf_pack(sizes, side):
    """Places (column, row) for rectangles of `sizes` (texels, w x h) in rows across a square of `side`, tallest
    first; None when they do not fit."""
    order = np.argsort(-sizes[:, 1], kind="stable")
    places = np.zeros_like(sizes)
    x = y = row_height = 0
    for k in order:
        w, h = sizes[k]
        if w > side:
            return None
        if x + w > side:
            x, y = 0, y + row_height
            row_height = 0
        if y + h > side:
            return None
        places[k] = (x, y)
        x += w
        row_height = max(row_height, h)
    return places


def atlas(corners, triangles, side=DEFAULT_SIDE):
    """The surface laid out on a square: (split corners, split triangles, their places 0-1 (u right, v down), the
    split corners' original corners, the texels per metre)."""
    chart, chart_axis = charts(corners, triangles)
    count = len(chart_axis)
    # Each chart's corners flat, in metres.
    flat_axes = np.array([FLAT[a] for a in chart_axis])
    corner_chart = np.repeat(chart, 3)
    tri_corners = triangles.reshape(-1)
    points = corners[tri_corners]
    picks = flat_axes[corner_chart]
    flat = np.stack([points[np.arange(len(points)), picks[:, 0]], points[np.arange(len(points)), picks[:, 1]]], 1)
    low = np.full((count, 2), np.inf)
    high = np.full((count, 2), -np.inf)
    np.minimum.at(low, corner_chart, flat)
    np.maximum.at(high, corner_chart, flat)
    extent = high - low
    _, areas = M.face_normals_and_areas(corners, triangles)
    density = np.sqrt(FIRST_FILL * side * side / max(areas.sum() * 1.0, 1e-12))
    for _ in range(MOST_PACKING_TRIES):
        sizes = np.ceil(extent * density).astype(np.int64) + 1 + 2 * MARGIN_TEXELS
        places = shelf_pack(sizes, side)
        if places is not None:
            break
        density *= SHRINK
    else:
        raise ValueError("the charts did not fit the texture")
    texel = (places[corner_chart] + MARGIN_TEXELS) + (flat - low[corner_chart]) * density
    uv = texel / side
    # Split corners: one per (corner, chart).
    pair = tri_corners.astype(np.int64) * count + corner_chart
    unique, inverse = np.unique(pair, return_inverse=True)
    inverse = inverse.reshape(-1)
    split_uv = np.zeros((len(unique), 2))
    split_uv[inverse] = uv
    original = unique // count
    return corners[original], inverse.reshape(-1, 3), split_uv, original, density


def cloud_spacing(cloud_points):
    """The cloud's usual spacing: the median distance to the nearest other point (a sample of them)."""
    if len(cloud_points) < 2:
        return 0.0
    tree = cKDTree(cloud_points)
    probe = cloud_points[:: max(1, len(cloud_points) // 5000)]
    distances, _ = tree.query(probe, k=2)
    return float(np.median(distances[:, 1]))


def texel_colours(points, normals, spread_colours, cloud, density):
    """The colours (0-1) of points on the surface: from the painted views' pixels near each that face its way (the
    colour cloud), else `spread_colours` (the colours spread to the corners, read at each point). Returns the colours
    and which points the cloud covered."""
    colours = np.array(spread_colours, dtype=np.float64)
    covered = np.zeros(len(points), dtype=bool)
    if cloud is not None and len(cloud[0]):
        cloud_points, cloud_normals, cloud_colours = (np.asarray(c, dtype=np.float64) for c in cloud)
        reach = CLOUD_REACH_SPACINGS * max(cloud_spacing(cloud_points), 0.5 / density)
        tree = cKDTree(cloud_points)
        distances, nearest = tree.query(points, k=CLOUD_NEIGHBOURS, distance_upper_bound=reach)
        found = np.isfinite(distances)
        safe = np.where(found, nearest, 0)
        facing = np.einsum("pd,pkd->pk", normals, cloud_normals[safe]) > SAME_FACING
        use = found & facing
        weight = np.where(use, 1.0 / (distances + 1e-6 * reach), 0.0)
        total = weight.sum(1)
        covered = total > 0
        mixed = np.einsum("pk,pkc->pc", weight, cloud_colours[safe]) / np.maximum(total, 1e-20)[:, None]
        colours[covered] = mixed[covered]
    return colours, covered


def starved_triangles(held_by, triangle_count):
    """The triangles that own no texel in the rasteriser's `held_by` picture (-1 where none)."""
    owns = np.zeros(triangle_count, dtype=bool)
    owns[held_by[held_by >= 0]] = True
    return np.flatnonzero(~owns)


def touched_texels(xy, triangles, starved, owned, side):
    """The texels the `starved` triangles touch that no triangle owns (`owned`): points along each one's three edges
    every `EDGE_STEP_TEXELS`, and its middle, each in the texel it falls in. A texel several triangles touch goes to
    the one with the largest area in texels (ties: the lowest index), at its point nearest the texel's centre. Returns
    rows, columns, the triangle and its barycentric weights there, one a texel."""
    empty = np.zeros(0, dtype=np.int64)
    if len(starved) == 0:
        return empty, empty, empty, np.zeros((0, 3))
    corners = xy[triangles[starved]]
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    pieces, weights = [np.arange(len(starved))], [np.full((len(starved), 3), 1.0 / 3.0)]
    for k in range(3):
        start, stop = corners[:, k], corners[:, (k + 1) % 3]
        steps = np.ceil(np.linalg.norm(stop - start, axis=1) / EDGE_STEP_TEXELS).astype(np.int64) + 1
        which = np.repeat(np.arange(len(starved)), steps)
        along = (np.arange(len(which)) - np.repeat(np.cumsum(steps) - steps, steps)) / np.repeat(
            np.maximum(steps - 1, 1), steps)
        w = np.zeros((len(which), 3))
        w[:, k], w[:, (k + 1) % 3] = 1.0 - along, along
        pieces.append(which)
        weights.append(w)
    which, w = np.concatenate(pieces), np.concatenate(weights)
    point = np.einsum("pk,pkd->pd", w, corners[which])
    cols, rows = np.floor(point[:, 0]).astype(np.int64), np.floor(point[:, 1]).astype(np.int64)
    inside = (cols >= 0) & (rows >= 0) & (cols < side) & (rows < side)
    which, w, point, cols, rows = which[inside], w[inside], point[inside], cols[inside], rows[inside]
    free = ~owned[rows, cols]
    which, w, point, cols, rows = which[free], w[free], point[free], cols[free], rows[free]
    off_centre = np.linalg.norm(point - (np.stack([cols, rows], 1) + 0.5), axis=1)
    texel = rows * side + cols
    order = np.lexsort((off_centre, starved[which], -area[which], texel))
    first = order[np.r_[True, texel[order][1:] != texel[order][:-1]]] if len(order) else order
    return rows[first], cols[first], starved[which[first]], w[first]


def baked(corners, triangles, corner_colours, cloud=None, side=DEFAULT_SIDE):
    """The baked texture and what it covers. `corner_colours` (0-1) are the colours spread to every corner; `cloud`
    (points, normals, colours) the painted views' pixels on the surface, or None. Returns a dict: corners,
    triangles, uvs (per split corner), texture (side x side x 3, 0-1), coverage (the share of the surface's texels a
    painted pixel coloured), coverage_picture."""
    split_corners, split_triangles, uv, original, density = atlas(corners, triangles, side)
    split_colours = corner_colours[original]
    xy = uv * side
    zbuf, held_by, bary = render.rasterised(xy, np.zeros(len(xy)), split_triangles, side, side)
    inside = held_by >= 0
    rows, cols = np.nonzero(inside)
    starved = starved_triangles(held_by, len(split_triangles))
    touched_rows, touched_cols, touched_triangle, touched_bary = touched_texels(xy, split_triangles, starved, inside,
                                                                             side)
    # The texels whose centre a triangle holds first, then the touched ones: only the first are counted in the
    # coverage, so every measure the report gives stays as it was.
    centred = int(inside.sum())
    texel_triangle = np.concatenate([held_by[inside], touched_triangle])
    texel_bary = np.concatenate([bary[inside], touched_bary])
    rows, cols = np.concatenate([rows, touched_rows]), np.concatenate([cols, touched_cols])
    tri = split_triangles[texel_triangle]
    surface_points = np.einsum("pk,pkd->pd", texel_bary, split_corners[tri])
    normals, _ = M.face_normals_and_areas(split_corners, split_triangles)
    surface_normals = normals[texel_triangle]
    spread = np.einsum("pk,pkc->pc", texel_bary, split_colours[tri])
    colours, covered = texel_colours(surface_points, surface_normals, spread, cloud, density)
    texture = np.zeros((side, side, 3))
    filled = np.zeros((side, side), dtype=bool)
    texture[rows, cols] = np.clip(colours, 0, 1)
    filled[rows, cols] = True
    seen = seen_share(split_corners, split_triangles, surface_points, surface_normals)
    picture = np.full((side, side, 3), 245, dtype=np.uint8)
    shown = np.where(covered[:, None], COVERED_COLOUR, FILLED_COLOUR).astype(np.float64)
    # A texel seldom seen is drawn faint: grey where no direction sees it.
    strength = np.clip(seen / FULLY_SEEN_SHARE, 0, 1)[:, None]
    picture[rows, cols] = (shown * strength + np.array(HIDDEN_COLOUR) * (1 - strength)).astype(np.uint8)
    filled_out(texture, filled)
    filled_to_the_end(texture, filled)
    seen, covered = seen[:centred], covered[:centred]
    # Where a chart folds over itself, two places of the surface share texels: the share of the charts' area drawn
    # over (about; texels along edges are counted once).
    a, b, c = (xy[split_triangles[:, k]] for k in range(3))
    uv_area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    overlap = max(0.0, 1.0 - centred / max(uv_area.sum(), 1e-9))
    seen_total = float(seen.sum())
    return {"corners": split_corners, "triangles": split_triangles, "uvs": uv, "texture": texture,
            "coverage": float((seen * covered).sum() / seen_total) if seen_total > 0 else 0.0,
            "coverage_all_texels": float(covered.mean()) if len(covered) else 0.0,
            "texels_per_m": float(density), "overlap": float(overlap), "coverage_picture": Image.fromarray(picture),
            "original": original}


def seen_share(corners, triangles, points, normals, count=SEEN_DIRECTIONS, side=SEEN_SIDE):
    """For each point on the surface, the share of directions round the object it is seen from: facing that way and
    nearest the viewer there (a surface inside a jug's neck is seen only from the few directions into its mouth)."""
    seen = np.zeros(len(points))
    extent = float(np.ptp(corners, axis=0).max())
    for direction in M.fibonacci_directions(count):
        tilt = np.degrees(np.arcsin(np.clip(direction[1], -1, 1)))
        turn = np.degrees(np.arctan2(direction[0], direction[2]))
        framing = render.Framing(corners, turn, side, tilt, extent=extent * 1.2)
        xy, depth = framing.pixels(corners)
        zbuf, _, _ = render.rasterised(xy, depth, triangles, side, side)
        point_xy, point_depth = framing.pixels(points)
        col = np.clip(point_xy[:, 0].astype(int), 0, side - 1)
        row = np.clip(point_xy[:, 1].astype(int), 0, side - 1)
        # Nearer than the drawn surface by a pixel's depth and a little: in view.
        slack = SEEN_SLACK_PIXELS / framing.scale
        facing = normals @ direction > 0
        seen += facing & (point_depth >= zbuf[row, col] - slack)
    return seen / count


def filled_out(texture, filled, rounds=FILL_ROUNDS):
    """Texels outside the charts take their filled neighbours' mean, `rounds` texels out."""
    kernel = np.ones((3, 3))
    for _ in range(rounds):
        weight = ndimage.convolve(filled.astype(np.float64), kernel, mode="constant")
        fresh = (~filled) & (weight > 0)
        if not fresh.any():
            break
        for c in range(3):
            summed = ndimage.convolve(texture[..., c] * filled, kernel, mode="constant")
            texture[..., c][fresh] = summed[fresh] / weight[fresh]
        filled |= fresh


def filled_to_the_end(texture, filled):
    """Texels still empty take the nearest filled texel's colour, however far."""
    if filled.all() or not filled.any():
        return
    _, (rows, cols) = ndimage.distance_transform_edt(~filled, return_indices=True)
    empty = ~filled
    texture[empty] = texture[rows[empty], cols[empty]]
    filled[:] = True


def neighbour_colours(corners, triangles, uvs, texture, starved):
    """For a model with only its texture (no colour cloud): each starved triangle's colour, the mean of its neighbours'
    across its edges (found by position, as a seam splits corners), each read at the shared edge's middle in the
    neighbour's own place on the texture. A neighbour starved too lends the colour it was given, in later rounds.
    Rows of NaN where no neighbour has a colour."""
    h, w = texture.shape[:2]
    _, weld = np.unique(corners, axis=0, return_inverse=True)
    welded = weld.reshape(-1)[triangles]
    edges = np.concatenate([welded[:, [k, (k + 1) % 3]] for k in range(3)])
    which = np.tile(np.arange(len(triangles)), 3)
    side_of = np.repeat(np.arange(3), len(triangles))
    key = np.sort(edges, axis=1)
    key = key[:, 0].astype(np.int64) * (weld.max() + 1) + key[:, 1]
    order = np.argsort(key, kind="stable")
    key, which, side_of = key[order], which[order], side_of[order]
    same = np.flatnonzero(key[1:] == key[:-1])
    pairs = np.concatenate([np.stack([which[same], which[same + 1], side_of[same + 1]], 1),
                            np.stack([which[same + 1], which[same], side_of[same]], 1)])
    # (triangle, its neighbour, the neighbour's edge): the neighbour's colour read at the middle of that edge.
    middle = (uvs[triangles[pairs[:, 1], pairs[:, 2]]] + uvs[triangles[pairs[:, 1], (pairs[:, 2] + 1) % 3]]) / 2
    read = texture[np.clip((middle[:, 1] * h).astype(np.int64), 0, h - 1),
                   np.clip((middle[:, 0] * w).astype(np.int64), 0, w - 1)]
    colour = np.full((len(triangles), 3), np.nan)
    is_starved = np.zeros(len(triangles), dtype=bool)
    is_starved[starved] = True
    lends = ~is_starved[pairs[:, 1]] & (read.sum(1) > 0)
    known = np.zeros(len(triangles), dtype=bool)
    while True:
        wanted = is_starved[pairs[:, 0]] & ~known[pairs[:, 0]] & (lends | known[pairs[:, 1]])
        if not wanted.any():
            break
        given = np.where(lends[wanted, None], read[wanted], colour[pairs[wanted, 1]])
        total = np.zeros((len(triangles), 3))
        count = np.zeros(len(triangles))
        np.add.at(total, pairs[wanted, 0], given)
        np.add.at(count, pairs[wanted, 0], 1)
        fresh = count > 0
        colour[fresh] = total[fresh] / count[fresh, None]
        known |= fresh
    return colour[starved]


def repaired(corners, triangles, uvs, texture):
    """A finished model's texture (H x W x 3, 0-1) with its starved triangles given colour, for a model that has only
    its .glb: the same layout (`uvs`, v down) read with the same rasteriser; each starved triangle's touched texels
    take its neighbours' colour (`neighbour_colours`); the texels that are empty (black, owned by no triangle) or
    within the fill's reach of a touched texel are filled again, the rest kept as they are."""
    h, w = texture.shape[:2]
    xy = uvs * np.array([w, h])
    _, held_by, _ = render.rasterised(xy, np.zeros(len(xy)), triangles, w, h)
    owned = held_by >= 0
    starved = starved_triangles(held_by, len(triangles))
    rows, cols, which, _ = touched_texels(xy, triangles, starved, owned, max(h, w))
    colour = neighbour_colours(corners, triangles, uvs, texture, starved)
    found = np.full((len(triangles), 3), np.nan)
    found[starved] = colour
    given = np.isfinite(found[which]).all(1)
    rows, cols, which = rows[given], cols[given], which[given]
    out = np.array(texture, dtype=np.float64)
    touched = np.zeros((h, w), dtype=bool)
    touched[rows, cols] = True
    out[rows, cols] = found[which]
    near = ndimage.binary_dilation(touched, structure=np.ones((3, 3), dtype=bool), iterations=FILL_ROUNDS)
    filled = owned | touched | ((out.sum(2) > 0) & ~near)
    out[~filled] = 0.0
    filled_out(out, filled)
    filled_to_the_end(out, filled)
    return out
