"""Our own small rasteriser: a surface seen from a turn round it (orthographic, or through a lens), as a depth buffer
with each pixel's triangle and place in it; the turntable and the texture bake are drawn with it (numpy only). A
silhouette, which needs no depth, is drawn by `covered`, faster, with the same pixels.
"""

import numpy as np
from PIL import Image

# Triangles spanning more pixels than this are drawn one at a time; smaller ones in batches by pixel offset.
BATCHED_SPAN = 12
# A pixel just outside a triangle's edge still counts as inside, so none between two triangles is missed.
EDGE_SLACK = -1e-6
LIGHT = np.array([0.35, 0.8, 0.55])
AMBIENT = 0.45
BACKGROUND = (238, 238, 234)


def view_matrix(turn_deg, tilt_deg=0.0):
    """Rows: the screen's right, its up and the direction toward the viewer, for a camera turned `turn_deg` round
    the object (0 sees its front, +Z, positive turns to its right side, +X) and tilted down by `tilt_deg`."""
    t, e = np.radians(turn_deg), np.radians(tilt_deg)
    toward = np.array([np.sin(t) * np.cos(e), np.sin(e), np.cos(t) * np.cos(e)])
    right = np.array([np.cos(t), 0.0, -np.sin(t)])
    up = np.cross(toward, right)
    return np.stack([right, up, toward])


def rasterised(xy, depth, triangles, width, height):
    """Each pixel's nearest triangle: (depth buffer, inf where none; triangle index, -1 where none; barycentric
    weights). `xy` are pixel positions (column, row) of the corners, larger `depth` nearer the viewer."""
    zbuf = np.full(width * height, -np.inf)
    owner = np.full(width * height, -1, dtype=np.int64)
    bary = np.zeros((width * height, 3))
    if len(triangles) == 0:
        return zbuf.reshape(height, width), owner.reshape(height, width), bary.reshape(height, width, 3)
    # Each triangle as its corners' (column, row, depth), its own index, and its corners' places in it.
    corners = np.concatenate([xy, depth[:, None]], 1)[triangles]
    parent = np.arange(len(triangles))
    places = np.tile(np.eye(3), (len(triangles), 1, 1))
    on_screen = ((corners[:, :, 0].max(1) >= 0) & (corners[:, :, 1].max(1) >= 0)
                 & (corners[:, :, 0].min(1) < width) & (corners[:, :, 1].min(1) < height))
    corners, parent, places = split_large(corners[on_screen], parent[on_screen], places[on_screen])
    tc, tr, tz = corners[:, :, 0], corners[:, :, 1], corners[:, :, 2]
    c0, c1 = np.floor(tc.min(1)).astype(np.int64), np.ceil(tc.max(1)).astype(np.int64)
    r0, r1 = np.floor(tr.min(1)).astype(np.int64), np.ceil(tr.max(1)).astype(np.int64)

    def draw(index, px, py):
        live = (px >= 0) & (py >= 0) & (px < width) & (py < height)
        index, px, py = index[live], px[live], py[live]
        if len(index) == 0:
            return
        (ax, bx, cx), (ay, by, cy) = tc[index].T, tr[index].T
        x, y = px + 0.5, py + 0.5
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        ok = np.abs(det) > 1e-12
        det = np.where(ok, det, 1.0)
        w0 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / det
        w1 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / det
        w2 = 1 - w0 - w1
        inside = ok & (w0 >= EDGE_SLACK) & (w1 >= EDGE_SLACK) & (w2 >= EDGE_SLACK)
        if not inside.any():
            return
        index, w = index[inside], np.stack([w0, w1, w2], 1)[inside]
        z = (w * tz[index]).sum(1)
        pixel = py[inside] * width + px[inside]
        # The nearest of several triangles at one pixel in this batch: sorted so the nearest is written last.
        order = np.argsort(z, kind="stable")
        pixel, z, index, w = pixel[order], z[order], index[order], w[order]
        nearer = z > zbuf[pixel]
        pixel, z, index, w = pixel[nearer], z[nearer], index[nearer], w[nearer]
        zbuf[pixel] = z
        owner[pixel] = parent[index]
        bary[pixel] = np.einsum("pk,pkj->pj", w, places[index])

    # Triangles grouped by their span, each group visited at only as many pixel offsets as its span needs.
    span = np.maximum(c1 - c0, r1 - r0)
    for size in np.unique(span):
        group = np.flatnonzero(span == size)
        for dy in range(size + 1):
            for dx in range(size + 1):
                fits = group[(c0[group] + dx <= c1[group]) & (r0[group] + dy <= r1[group])]
                draw(fits, c0[fits] + dx, r0[fits] + dy)
    return zbuf.reshape(height, width), owner.reshape(height, width), bary.reshape(height, width, 3)


def split_large(corners, parent, places):
    """Triangles spanning more than `BATCHED_SPAN` pixels split in two at the middle of their longest edge until none
    does (a long thin triangle is halved along its length only); each piece keeps its original triangle and its
    corners' places in it."""
    done = [(corners, parent, places)]
    while True:
        corners, parent, places = done[-1]
        span = np.maximum(np.ptp(corners[:, :, 0], axis=1), np.ptp(corners[:, :, 1], axis=1))
        large = span > BATCHED_SPAN
        if not large.any():
            break
        done[-1] = (corners[~large], parent[~large], places[~large])
        c, p, q = corners[large], parent[large], places[large]
        rows = np.arange(len(c))
        # Corner k starts the longest edge (k to k+1); the third corner is k+2.
        edge = np.stack([np.linalg.norm(c[:, (k + 1) % 3, :2] - c[:, k, :2], axis=1) for k in range(3)], 1)
        k = edge.argmax(1)
        a, b, opposite = k, (k + 1) % 3, (k + 2) % 3
        middle_c = (c[rows, a] + c[rows, b]) / 2
        middle_q = (q[rows, a] + q[rows, b]) / 2
        first_c = np.stack([c[rows, a], middle_c, c[rows, opposite]], 1)
        second_c = np.stack([middle_c, c[rows, b], c[rows, opposite]], 1)
        first_q = np.stack([q[rows, a], middle_q, q[rows, opposite]], 1)
        second_q = np.stack([middle_q, q[rows, b], q[rows, opposite]], 1)
        done.append((np.concatenate([first_c, second_c]), np.tile(p, 2), np.concatenate([first_q, second_q])))
    return (np.concatenate([d[0] for d in done]), np.concatenate([d[1] for d in done]),
            np.concatenate([d[2] for d in done]))


class Framing:
    """Where a surface lands on a square picture seen from a turn: its middle centred, its largest extent across
    the view filling `fill` of the side. Seen straight on (orthographic) unless `distance` is given: then through a
    lens, from that many of the object's largest sizes away from its middle (a photograph's perspective)."""

    def __init__(self, corners, turn_deg, side, tilt_deg=0.0, fill=0.86, extent=None, distance=None):
        self.side = side
        self.matrix = view_matrix(turn_deg, tilt_deg)
        self.lens = None
        if distance is not None and np.isfinite(distance):
            self.centre = ((corners.min(0) + corners.max(0)) / 2) @ self.matrix.T
            self.lens = distance * float(np.ptp(corners, axis=0).max())
        flat, _ = self.flattened(corners)
        low, high = flat.min(0), flat.max(0)
        self.middle = (low + high) / 2
        self.scale = fill * side / max(extent or float((high - low).max()), 1e-9)

    def flattened(self, corners):
        """Points on the picture's plane (metres, before framing) and their depth (larger nearer)."""
        seen = corners @ self.matrix.T
        if self.lens is None:
            return seen[:, :2], seen[:, 2]
        offset = seen - self.centre
        # Each point drawn smaller the farther it lies from the lens (it is `lens` metres from the middle).
        shrink = self.lens / np.maximum(self.lens - offset[:, 2], 1e-6 * self.lens)
        return offset[:, :2] * shrink[:, None], seen[:, 2]

    def pixels(self, corners):
        """Pixel positions (column, row; row 0 at the top) and depth (larger nearer) of points."""
        flat, depth = self.flattened(corners)
        cols = (flat[:, 0] - self.middle[0]) * self.scale + self.side / 2
        rows = self.side / 2 - (flat[:, 1] - self.middle[1]) * self.scale
        return np.stack([cols, rows], 1), depth


def silhouette(corners, triangles, turn_deg, side, tilt_deg=0.0, distance=None):
    """Where the surface is, seen from a turn: the pixels `rasterised` draws (an outline needs no depth)."""
    framing = Framing(corners, turn_deg, side, tilt_deg, distance=distance)
    xy, _ = framing.pixels(corners)
    return covered(xy, triangles, side, side)


def covered(xy, triangles, width, height):
    """The pixels whose middle lies inside a triangle (as `rasterised` tests it, `EDGE_SLACK` included): each triangle
    cut by each pixel row's middle line into a span of columns, and the spans summed along their rows: every
    triangle's rows at once, nothing per pixel. On the regression set's surfaces (50 000 to 60 000 triangles, 448
    pixels) it draws the same pixels as `rasterised` 5 to 14 times faster."""
    seen = np.zeros((height, width), dtype=bool)
    if len(triangles) == 0:
        return seen
    corner = xy[triangles]
    x, y = corner[:, :, 0], corner[:, :, 1]
    det = (y[:, 1] - y[:, 2]) * (x[:, 0] - x[:, 2]) + (x[:, 2] - x[:, 1]) * (y[:, 0] - y[:, 2])
    first = np.maximum(np.ceil(y.min(1) - 0.5), 0).astype(np.int64)
    last = np.minimum(np.floor(y.max(1) - 0.5), height - 1).astype(np.int64)
    rows_of = np.where(np.abs(det) > 1e-12, last - first + 1, 0).clip(min=0)
    if rows_of.sum() == 0:
        return seen
    # One entry a (triangle, row) the triangle's span of rows covers.
    owner = np.repeat(np.arange(len(triangles)), rows_of)
    row = first[owner] + np.arange(len(owner)) - np.repeat(np.cumsum(rows_of) - rows_of, rows_of)
    middle = row + 0.5
    left, right = np.full(len(owner), np.inf), np.full(len(owner), -np.inf)
    for k in range(3):
        xa, ya, xb, yb = x[owner, k], y[owner, k], x[owner, (k + 1) % 3], y[owner, (k + 1) % 3]
        rise = yb - ya
        crosses = (middle >= np.minimum(ya, yb)) & (middle <= np.maximum(ya, yb)) & (rise != 0)
        rise = np.where(crosses, rise, 1.0)
        at = xa + (middle - ya) * (xb - xa) / rise
        # The edge's slack in columns: a barycentric weight of EDGE_SLACK is this far outside it along the row.
        slack = -EDGE_SLACK * np.abs(det[owner] / rise)
        left = np.where(crosses, np.minimum(left, at - slack), left)
        right = np.where(crosses, np.maximum(right, at + slack), right)
    start = np.maximum(np.ceil(left - 0.5), 0)
    stop = np.minimum(np.floor(right - 0.5), width - 1)
    some = start <= stop
    row, start, stop = row[some], start[some].astype(np.int64), stop[some].astype(np.int64)
    # Each span +1 where it starts and -1 after it ends, summed along its row.
    stride = width + 1
    steps = (np.bincount(row * stride + start, minlength=height * stride)
             - np.bincount(row * stride + stop + 1, minlength=height * stride))
    return np.cumsum(steps.reshape(height, stride), axis=1)[:, :width] > 0


def shaded(corners, triangles, turn_deg, side, colour_at=None, tilt_deg=15.0, extent=None):
    """The surface lit from above-front, its colour from `colour_at(owner, bary)` (pixel colours 0-1) or plain clay.
    Returns an RGB picture."""
    from .mesh_tools import face_normals_and_areas

    framing = Framing(corners, turn_deg, side, tilt_deg, extent=extent)
    xy, depth = framing.pixels(corners)
    zbuf, owner, bary = rasterised(xy, depth, triangles, side, side)
    hit = owner >= 0
    image = np.empty((side, side, 3))
    image[:] = np.array(BACKGROUND) / 255
    normals, _ = face_normals_and_areas(corners, triangles)
    seen_normals = normals @ framing.matrix.T
    n = seen_normals[owner[hit]]
    # Lit from the viewer's upper left; a back face (normal away) is lit as if it faced the viewer.
    n = np.where(n[:, 2:3] < 0, -n, n)
    light = LIGHT / np.linalg.norm(LIGHT)
    lambert = AMBIENT + (1 - AMBIENT) * np.clip(n @ light, 0, 1)
    if colour_at is None:
        base = np.tile(np.array([0.78, 0.74, 0.68]), (int(hit.sum()), 1))
    else:
        base = colour_at(owner[hit], bary[hit])
    image[hit] = np.clip(base * lambert[:, None], 0, 1)
    return Image.fromarray((image * 255).round().astype(np.uint8))


def texture_lookup(triangle_uvs, texture):
    """Colours from a texture (H x W x 3, 0-1) by each triangle's corner places (m x 3 x 2, 0-1, v down)."""
    h, w = texture.shape[:2]

    def colour_at(owner, bary):
        uv = np.einsum("pk,pkc->pc", bary, triangle_uvs[owner])
        cols = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
        rows = np.clip((uv[:, 1] * h).astype(int), 0, h - 1)
        return texture[rows, cols]
    return colour_at


def turntable(corners, triangles, colour_at=None, side=360, turns=(0, 45, 90, 135, 180, 225, 270, 315)):
    """The object from each turn, one size throughout (its largest extent), in a row."""
    extent = float(np.ptp(corners, axis=0).max()) * 1.05
    pictures = [shaded(corners, triangles, t, side, colour_at, extent=extent) for t in turns]
    sheet = Image.new("RGB", (side * len(pictures), side), BACKGROUND)
    for k, picture in enumerate(pictures):
        sheet.paste(picture, (k * side, 0))
    return sheet
