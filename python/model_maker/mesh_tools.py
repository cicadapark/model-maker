"""A surface's plain measures: normals, areas, pieces, open edges, volume and centre of mass, points on it.

A surface is its corners (n x 3, metres, +Y up) and triangles (m x 3 corner indices).
"""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def face_normals_and_areas(corners, triangles):
    a, b, c = (corners[triangles[:, k]] for k in range(3))
    cross = np.cross(b - a, c - a)
    length = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(length, 1e-20)[:, None]
    return normals, length / 2


def corner_normals(corners, triangles):
    a, b, c = (corners[triangles[:, k]] for k in range(3))
    face = np.cross(b - a, c - a)
    normals = np.zeros_like(corners, dtype=np.float64)
    for k in range(3):
        np.add.at(normals, triangles[:, k], face)
    return normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-20)


def joined(corners, triangles, tolerance):
    """Corners on one grid cell of `tolerance` made one: (corners, triangles, how many corners were joined)."""
    keys = np.round(corners / max(tolerance, 1e-12)).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    return corners[first], inverse[triangles], int(len(corners) - len(first))


def welded(corners, triangles, tolerance):
    """The same surface with corners closer than `tolerance` made one (a surface written per triangle, or split at
    seams, has its pieces joined again before they are counted), the triangles that fold to a line dropped."""
    corners, triangles, _ = joined(corners, triangles, tolerance)
    keep = (triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2]) & \
        (triangles[:, 0] != triangles[:, 2])
    return corners, triangles[keep]


def pieces(corners, triangles):
    """Each triangle's piece (triangles joined through shared corners), and the pieces' count."""
    n = len(corners)
    if len(triangles) == 0:
        return np.zeros(0, dtype=np.int64), 0
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]]])
    graph = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    count, piece_of_corner = connected_components(graph, directed=False)
    piece = piece_of_corner[triangles[:, 0]]
    # Corners no triangle uses are pieces of their own: numbered away.
    used, piece = np.unique(piece, return_inverse=True)
    return piece.reshape(-1), len(used)


def edge_uses(triangles):
    """Each undirected edge (sorted corner pair) and how many triangles use it."""
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique, counts


def open_edge_loops(triangles):
    """The surface's holes: loops of edges only one triangle uses. Returns (open edges, loops, edges used by more than
    two triangles)."""
    unique, counts = edge_uses(triangles)
    open_edges = unique[counts == 1]
    torn = int((counts > 2).sum())
    if len(open_edges) == 0:
        return 0, 0, torn
    corners = np.unique(open_edges)
    index = {c: i for i, c in enumerate(corners.tolist())}
    rows = np.array([index[c] for c in open_edges[:, 0].tolist()])
    cols = np.array([index[c] for c in open_edges[:, 1].tolist()])
    graph = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(corners), len(corners)))
    loops, _ = connected_components(graph, directed=False)
    return len(open_edges), int(loops), torn


def volume_and_centre(corners, triangles):
    """The enclosed volume (m^3) and its centre of mass, by signed tetrahedra from the origin (exact for a closed
    surface wound outward)."""
    a, b, c = (corners[triangles[:, k]] for k in range(3))
    six_volumes = np.einsum("ij,ij->i", a, np.cross(b, c))
    volume = six_volumes.sum() / 6
    if abs(volume) < 1e-15:
        return 0.0, surface_centre(corners, triangles)
    centre = (six_volumes[:, None] * (a + b + c)).sum(0) / (4 * six_volumes.sum())
    return float(volume), centre


def surface_centre(corners, triangles):
    """The area-weighted centre of the surface (a shell's centre of mass)."""
    _, areas = face_normals_and_areas(corners, triangles)
    middles = corners[triangles].mean(1)
    return (middles * areas[:, None]).sum(0) / max(areas.sum(), 1e-20)


def sampled(corners, triangles, count, seed=0):
    """`count` points spread over the surface by area: (points, their triangles' normals, their triangles)."""
    normals, areas = face_normals_and_areas(corners, triangles)
    rng = np.random.default_rng(seed)
    total = areas.sum()
    if total <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    chosen = rng.choice(len(triangles), size=count, p=areas / total)
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    a, b, c = (corners[triangles[chosen, k]] for k in range(3))
    points = a + (b - a) * u[:, None] + (c - a) * v[:, None]
    return points, normals[chosen], chosen


def fibonacci_directions(count):
    """Directions spread evenly round a sphere (a Fibonacci lattice)."""
    k = np.arange(count) + 0.5
    up = 1 - 2 * k / count
    turn = np.pi * (1 + 5 ** 0.5) * k
    ring = np.sqrt(1 - up * up)
    return np.stack([ring * np.sin(turn), up, ring * np.cos(turn)], 1)


def compacted(corners, triangles, *per_corner):
    """Only the corners the triangles use, renumbered; `per_corner` arrays follow them."""
    used, inverse = np.unique(triangles.reshape(-1), return_inverse=True)
    return (corners[used], inverse.reshape(-1, 3), *[a[used] for a in per_corner])


def closest_on_triangles(points, a, b, c):
    """Each point's closest point on its triangle (a, b, c: its corners, one row each), by the triangle's Voronoi
    regions (Ericson, "Real-Time Collision Detection", 2005, 5.1.5): a corner, an edge or the face."""
    ab, ac = b - a, c - a
    ap, bp, cp = points - a, points - b, points - c

    def dot(u, v):
        return np.einsum("ij,ij->i", u, v)

    d1, d2, d3, d4, d5, d6 = dot(ab, ap), dot(ac, ap), dot(ab, bp), dot(ac, bp), dot(ab, cp), dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        total = va + vb + vc
        v = np.where(total != 0, vb / total, 0.0)
        w = np.where(total != 0, vc / total, 0.0)
        found = a + ab * v[:, None] + ac * w[:, None]
        # The edges and corners, each overriding the face where its region holds the point (the corners last).
        on_bc = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        share = np.where(on_bc, (d4 - d3) / np.where(on_bc, (d4 - d3) + (d5 - d6), 1.0), 0.0)
        found = np.where(on_bc[:, None], b + share[:, None] * (c - b), found)
        on_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        share = np.where(on_ac, d2 / np.where(on_ac, d2 - d6, 1.0), 0.0)
        found = np.where(on_ac[:, None], a + share[:, None] * ac, found)
        on_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        share = np.where(on_ab, d1 / np.where(on_ab, d1 - d3, 1.0), 0.0)
        found = np.where(on_ab[:, None], a + share[:, None] * ab, found)
    found = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, found)
    found = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, found)
    found = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, found)
    return found


# The surface's distance from a point is measured to the triangles of this many of its nearest samples.
NEAREST_SAMPLES = 8


class SurfaceDistance:
    """Distances from points to a surface: to its nearest triangle among those under the point's nearest samples
    (`NEAREST_SAMPLES` of the `sampled` points), point to triangle, exactly. A distance to the nearest sample alone
    carries the samples' spacing (about half the spacing, the sampling floor): a perfectly round spear's shaft, 3 cm
    across and 3 m long, mirrored onto 60,000 samples read 3.6 % of its width off itself."""

    def __init__(self, corners, triangles, points, chosen):
        from scipy.spatial import cKDTree

        self.corners, self.triangles, self.chosen = corners, triangles, chosen
        self.tree = cKDTree(points)

    def __call__(self, queries):
        if len(queries) == 0:
            return np.zeros(0)
        count = min(NEAREST_SAMPLES, len(self.chosen))
        _, nearest = self.tree.query(queries, k=count)
        nearest = np.asarray(nearest).reshape(len(queries), count)
        best = np.full(len(queries), np.inf)
        for k in range(count):
            corners = self.corners[self.triangles[self.chosen[nearest[:, k]]]]
            closest = closest_on_triangles(queries, corners[:, 0], corners[:, 1], corners[:, 2])
            best = np.minimum(best, np.linalg.norm(queries - closest, axis=1))
        return best
