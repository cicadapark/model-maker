"""A hafted weapon turned so its haft stands upright, its head up.

The shape model makes the object as its picture shows it, in the picture's own frame. A mace, a club or an axe is
painted "standing upright, its head at the top", but the picture model often paints a heavy head lying, tilted or
head down: of the 26 maces the 0.3.0 words trial painted, 14 were not upright, and the two made from such pictures
came out leaning about 40 degrees in their own frame (docs/objects.md, "Maces, clubs and things the picture model
knows only loosely"). The weapon checks measure along the weapon's own long axis, so they passed; the object did not
stand along the up axis.

The haft is found as the weapon checks find it (`weapon_checks`: the slices along the weapon's long axis, followed along
its line, its longest narrow stretch), and its direction is the straight line through those slices' middles, so a heavy
head that tilts the long axis a little does not tilt the haft. The head is the end the picture shows at the top (the
words ask for it there), unless the other end is clearly the wider, so a mace painted head down is righted too; the
weapon is turned by the least rotation that brings the haft onto the up axis, head up, and set on the floor, centred, as
a thing that lies is laid (`lying.laid`, whose record and frame changes it shares: the outline is still judged in the
frame the shape was made in, and the paint follows). Only hafted weapons are turned; a blade, a spear, a bow and a
shield are left as made.
"""

import numpy as np

from . import lying
from . import mesh_tools as M
from . import weapon_checks as W

# Fewer slices of haft than this, and the weapon's own long axis is taken for the haft.
LEAST_HAFT_SLICES = 6
# A weapon is turned head down only when its lower end is this much wider than its upper (a head, not a knob or a
# ferrule): on the 0.3.0 words trial's tries made into shapes, the maces painted head down measured their top end 0.14
# to 0.54 of their bottom's width, the plain sticks and clubs painted the right way up 0.88 and over.
HEAD_CLEAR = 1.5
# The picture's tries ranked: a hafted weapon's cut-out whose long axis leans more than this from the picture's
# vertical is ranked after the upright ones (never refused). On the 0.3.0 words trial's labelled maces the upright ones
# leaned 0 to 14 degrees, the tilted ones 18 to 86 (13 of 14; the fourteenth was painted head down, upright), the 59
# rods and sticks 3 at most: the limit is the middle of the gap.
MOST_UPRIGHT_LEAN_DEG = 16.0


def head_direction(corners, triangles):
    """The unit direction from the butt along the haft toward the head, and how it was found."""
    points, _, _ = M.sampled(corners, triangles, W.SAMPLES, seed=2)
    frame = W.frame_of(points)
    local = frame.local(points)
    length = float(np.ptp(local[:, 0]))
    slices = W.tracked_slices(local, length)
    shaft = W.shaft_slices(slices, length, one_stretch=True)
    if len(shaft) >= LEAST_HAFT_SLICES:
        s = np.array([x.s for x in shaft])
        middles = np.array([x.middle for x in shaft])
        slope = np.polyfit(s, middles, 1)[0]
        along = np.array([1.0, slope[0], slope[1]])
        found = f"the line through its haft's {len(shaft)} slices"
    else:
        along = np.array([1.0, 0.0, 0.0])
        found = "its own long axis (too little haft was found to follow)"
    axis = along[0] * frame.length_axis + along[1] * frame.width_axis + along[2] * frame.thickness_axis
    axis /= np.linalg.norm(axis)
    return axis * head_end(slices), found


def head_end(slices):
    """+1 when the head is the end toward the picture's top (the weapon checks' frame points its length that way), -1
    when the other end is clearly the wider: by `HEAD_CLEAR` or more. The words ask for the head at the top, so the
    picture's own up stands unless the shape says otherwise; a plain stick's two ends differ by little."""
    if not slices:
        return 1
    s = np.array([x.s for x in slices])
    low, high = s.min(), s.max()
    third = (high - low) / 3
    widest_low = max((x.width for x in slices if x.s <= low + third), default=0.0)
    widest_high = max((x.width for x in slices if x.s >= high - third), default=0.0)
    return -1 if widest_low > HEAD_CLEAR * widest_high else 1


def turned(corners, triangles):
    """The weapon turned so its haft stands upright, head up, on the floor and centred: (corners, triangles, record);
    the record as `lying.laid`'s, with `upright` set."""
    head, found = head_direction(corners, triangles)
    rotation = lying.turning(-head)
    standing = corners @ rotation.T
    low, high = standing.min(0), standing.max(0)
    shift = -np.array([(low[0] + high[0]) / 2, low[1], (low[2] + high[2]) / 2])
    turned_deg = float(np.degrees(np.arccos(np.clip(head @ np.array([0.0, 1.0, 0.0]), -1, 1))))
    record = {"laid": True, "upright": True, "turned_deg": round(turned_deg, 2), "found_by": found,
              "rotation": rotation.round(9).tolist(), "shift": [float(x) for x in shift],
              "words": f"turned {turned_deg:.0f}° from how it was made so its haft stands upright, head up (the haft "
                       f"from {found})"}
    return standing + shift, triangles, record


def rotation_for_paint(corners, triangles):
    """The rotation that stands a hafted weapon's made shape upright, for the view painter to paint it standing."""
    head, _ = head_direction(corners, triangles)
    return lying.turning(-head)


def lean_deg(mask):
    """How far a cut-out's long axis (its second moments) leans from the picture's vertical, in degrees."""
    rows, cols = np.nonzero(mask)
    if len(rows) < 3:
        return 0.0
    cov = np.cov(np.stack([cols - cols.mean(), rows - rows.mean()]))
    axis = np.linalg.eigh(cov)[1][:, 1]
    return float(np.degrees(np.arctan2(abs(axis[0]), abs(axis[1]))))


def standing_check(mask):
    """A hafted weapon's picture try painted upright, shown and used to rank the tries that pass, not counted: the
    finish turns the weapon upright whatever the picture, but a try painted upright is the shape model's own pose."""
    from .picture_checks import Check

    lean = lean_deg(mask)
    return Check("standing upright", lean, MOST_UPRIGHT_LEAN_DEG, "°", lean <= MOST_UPRIGHT_LEAN_DEG,
                 f"the cut-out's long axis leans {lean:.0f}° from upright (tries leaning more than "
                 f"{MOST_UPRIGHT_LEAN_DEG:.0f}° are chosen after the upright ones; the finish stands the weapon up)",
                 counted=False)
