"""A thing that lies (a book, a coin, a plate) laid on its broad face, the way it comes to rest when set down.

The shape model makes the object as its picture shows it, in the picture's own frame: its up is the picture's up. A
book painted from well above, "lying shut, only its cover showing" (the fix-check of 2026-11-10), came out as a slab
standing tilted toward the viewer, its cover facing the camera; set down upright as made, it stood on an edge and
settled 54.8 degrees. A thing that lies has no up of its own in the picture: it rests on whichever broad face it
falls on.

The established way to find how a rigid object comes to rest is its stable poses with the quasi-static probability of
each (Wiegley, Rao and Goldberg, "Computing a statistical distribution of stable poses of a polyhedron", 1992;
Kriegman, "Let them fall where they may", 1997; Moll and Erdmann, 2002; the upright-orientation work of Fu, Cohen-Or,
Dror and Sheffer, 2008, starts from the same candidates, the faces of the convex hull). Set down slowly with its
centre of mass at a height h(d) above the floor (d the direction that points down: the hull's support distance from
the centre along d), the object rolls downhill on h until it rests at a local minimum, the normal of a hull face
under its centre of mass. The share of all directions that roll to a pose is its probability. Here the directions
are a Fibonacci lattice, each rolls to its lowest neighbour until none is lower, and minima within a few degrees of
each other (a cover a little warped) are one pose.

Only a thing whose words say it lies (`words.LYING`) is laid, on its most probable pose (of two near-equal ones,
a book's front and back covers, the one whose face looks the way the picture did faces up). A chair, a table, a
barrel stand as made: one that tips must fail its base checks, never be laid on its back.
"""

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from . import mesh_tools as M

# Directions tried (about 2.6 degrees apart) and the neighbours each rolls toward.
DIRECTIONS = 6000
NEIGHBOURS = 8
# Resting directions within this angle are one pose (a broad face a little curved has several shallow minima).
SAME_POSE_DEG = 12.0
# Of the poses at least this share as probable as the likeliest, the one whose face looks most toward the picture's
# viewer is laid face up (a book's two covers come out near-equal).
NEAR_LIKELIEST = 0.6
# The picture's viewer in the shape model's frame: in front (+Z), the outline's turn 0.
TOWARD_VIEWER = np.array([0.0, 0.0, 1.0])
DOWN = np.array([0.0, -1.0, 0.0])


def centre_of_mass(corners, triangles):
    """The volume's centre when the surface is closed and wound outward, else the surface's (as the base checks)."""
    volume, centre = M.volume_and_centre(corners, triangles)
    if volume > 0 and M.open_edge_loops(triangles)[0] == 0:
        return centre
    return M.surface_centre(corners, triangles)


def stable_poses(corners, triangles, count=DIRECTIONS):
    """The poses the object comes to rest in, likeliest first: each a dict with `down` (the unit direction that
    points to the floor, in the object's frame: the resting hull face's outward normal), `share` (its quasi-static
    probability), `height_m` (its centre of mass above the floor)."""
    centre = centre_of_mass(corners, triangles)
    hull = ConvexHull(corners)
    points = corners[hull.vertices] - centre
    directions = M.fibonacci_directions(count)
    height = (points @ directions.T).max(0)
    _, near = cKDTree(directions).query(directions, k=NEIGHBOURS + 1)
    lowest = near[np.arange(count), np.argmin(height[near], axis=1)]
    rolls_to = np.where(height[lowest] < height, lowest, np.arange(count))
    while True:
        further = rolls_to[rolls_to]
        if np.array_equal(further, rolls_to):
            break
        rolls_to = further
    rests, sizes = np.unique(rolls_to, return_counts=True)
    normals = hull.equations[:, :3]
    poses = []
    for rest in rests[np.argsort(height[rests])]:
        share = sizes[rests == rest][0] / count
        # The resting face: the hull face whose outward normal is nearest the direction it rolled to.
        down = normals[np.argmax(normals @ directions[rest])]
        same = next((p for p in poses if np.degrees(np.arccos(np.clip(p["down"] @ down, -1, 1))) <= SAME_POSE_DEG),
                    None)
        if same is not None:
            same["share"] += share
            continue
        poses.append({"down": down, "share": float(share), "height_m": float((points @ down).max())})
    return sorted(poses, key=lambda p: -p["share"])


def turning(down):
    """The rotation (3 x 3) that turns the direction `down` to point straight down, the least turn that does."""
    down = down / np.linalg.norm(down)
    axis = np.cross(down, DOWN)
    sine, cosine = np.linalg.norm(axis), float(down @ DOWN)
    if sine < 1e-9:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    k = axis / sine
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + sine * cross + (1 - cosine) * cross @ cross


def chosen_pose(poses):
    likeliest = poses[0]["share"]
    near = [p for p in poses if p["share"] >= NEAR_LIKELIEST * likeliest]
    return max(near, key=lambda p: float(-p["down"] @ TOWARD_VIEWER))


def laid(subject, corners, triangles):
    """The surface laid on its likeliest resting face when its words say it lies (else as it is): (corners,
    triangles, record). The record says whether it was laid, the pose (its share, how far it turned) and the
    other poses, and holds the rotation and shift that took the made shape to the laid one. A hafted weapon is turned
    instead so its haft stands upright, head up (`upright`), with a record of the same kind."""
    if subject.weapon_kind == "hafted":
        from . import upright

        return upright.turned(corners, triangles)
    if not subject.lies:
        return corners, triangles, {"laid": False, "words": subject.why.get("lies", "not a thing that lies")}
    poses = stable_poses(corners, triangles)
    pose = chosen_pose(poses)
    rotation = turning(pose["down"])
    turned = corners @ rotation.T
    low, high = turned.min(0), turned.max(0)
    shift = -np.array([(low[0] + high[0]) / 2, low[1], (low[2] + high[2]) / 2])
    turned_deg = float(np.degrees(np.arccos(np.clip(pose["down"] @ DOWN, -1, 1))))
    record = {"laid": True, "turned_deg": round(turned_deg, 2), "share": round(pose["share"], 4),
              "poses": [{"down": [round(float(x), 4) for x in p["down"]], "share": round(p["share"], 4),
                         "height_m": round(p["height_m"], 4)} for p in poses[:6]],
              "rotation": rotation.round(9).tolist(), "shift": [float(x) for x in shift],
              "words": f"laid on the face it most often comes to rest on ({pose['share']:.0%} of the ways it can "
                       f"fall), turned {turned_deg:.0f}° from how it was made"}
    return turned + shift, triangles, record


def rotation_of(record):
    return np.array(record["rotation"]) if record and record.get("laid") else np.eye(3)


def to_made(points, record):
    """Points of the laid shape back in the frame the shape was made in."""
    if not record or not record.get("laid"):
        return points
    return (points - np.array(record["shift"])) @ rotation_of(record)


def from_made(points, record, directions=False):
    """Points (or, `directions`, normals) of the made shape's frame in the laid shape's."""
    if not record or not record.get("laid"):
        return points
    turned = points @ rotation_of(record).T
    return turned if directions else turned + np.array(record["shift"])


def lay_check(record):
    """The laying, shown beside the checks (not counted: the base checks judge the pose it leaves); for a hafted
    weapon, its turn upright."""
    from .picture_checks import Check

    return Check("haft upright" if record.get("upright") else "laid on its broad face", record.get("turned_deg", 0.0), 0.0, "°", True, record["words"],
                 counted=False)
