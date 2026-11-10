"""The checks a shaped weapon (or a long-handled tool) passes, measured along its length:

* its blade or shaft straight along its line: no bend (the slices' middles off the line) and no kink (the line's
  direction turning sharply); a curved blade or a bow is curved by design, so only its kinks are measured;
* mirror-symmetric across the blade or haft where the prompt implies it (a sword, a spear, a double-bitted axe),
  or about its grip (a bow's two limbs);
* a grip: a stretch thin and round enough for a hand, at least a hand wide, and where it lies along the length;
* its length, at the size the user set (the grip's thickness is judged at that size);
* the mesh whole with no loose guard or pommel (in `object_checks.whole_checks`).

The weapon's frame: its length along the surface's longest spread (principal axis), pointing up the picture; its
width across the blade or head; its thickness the third way.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from . import mesh_tools as M
from .picture_checks import Check

SAMPLES = 60000
SLICES = 120
# Slices this near an end are left out of the line (a point or a pommel's round end narrows to nothing).
END_SHARE = 0.03
# In a slice, points further apart than this across (metres, at the weapon's size) are separate parts (a bow and
# its string): the slice's part is its largest.
PART_GAP_M = 0.012
# The shaft or blade: slices whose part is at most this many times the slices' middle width.
SHAFT_WIDTH_RATIO = 1.5
# The line: the shaft's middles at most this share of the length off their straight line, and its direction turning
# at most this much between neighbouring stretches of `KINK_STRETCH_SHARE` of the length.
MOST_BEND_SHARE = 0.01
MOST_KINK_DEG = 4.0
KINK_STRETCH_SHARE = 0.1
# A grip: thick enough to hold and thin enough to close a hand round (across its widest), not flat (a blade is
# flat), and at least a hand wide.
GRIP_LEAST_M = 0.015
GRIP_MOST_M = 0.055
GRIP_MOST_FLATNESS = 2.5
GRIP_LEAST_LENGTH_M = 0.07
GRIP_SWELL = 1.35
# A hafted weapon is held this far in from the end of its haft; a spear at this share of its length from its butt.
HAFT_HOLD_IN_M = 0.08
SPEAR_HOLD_SHARE = 0.4
# A haft's (or a shaft's) grip is on the haft proper, which begins, from the butt, at the first stretch a hand wide
# whose girth stays within this share of the haft's middle girth. A flared butt cap, its neck, a collar or a pommel at
# the end is not haft: the remade axe's grip mark sat 14 cm from its end on the black butt piece, whose neck (2.8 cm
# round, under the haft's 3.3 cm) was thin enough to hold (remake of 10 November 2026). Taking the longest even
# stretch instead cut a spear's shaft at a band 13 % thicker and moved its hold off its 40 %.
HAFT_EVEN_SHARE = 0.1
# Mirror-symmetric across the blade or haft: the mirrored surface's mean distance from the surface (to its triangles)
# at most this share of the weapon's width; about the grip (a bow), at most this share of its length.
MOST_ACROSS_SHARE = 0.03
MOST_ALONG_SHARE = 0.015
# The tip: the surface's points within this distance of its furthest point along the length.
TIP_REACH_M = 0.004


@dataclass
class Frame:
    middle: np.ndarray
    length_axis: np.ndarray  # toward the top of the picture
    width_axis: np.ndarray
    thickness_axis: np.ndarray

    def local(self, points):
        d = points - self.middle
        return np.stack([d @ self.length_axis, d @ self.width_axis, d @ self.thickness_axis], 1)

    def world(self, local):
        return (self.middle + local[..., 0:1] * self.length_axis + local[..., 1:2] * self.width_axis
                + local[..., 2:3] * self.thickness_axis)


def frame_of(points):
    middle = points.mean(0)
    _, _, axes = np.linalg.svd(points - middle, full_matrices=False)
    length, width = axes[0], axes[1]
    if length[1] < 0:
        length = -length
    thickness = np.cross(length, width)
    return Frame(middle, length, width, thickness)


# A slice is measured across in these directions (every 22.5 degrees): its width the widest, its girth the mean (a round
# part's diameter; a square haft's the same however it is turned: what a hand closes round), its flatness the widest
# over the thinnest. Measured along the frame's two axes only, a 4 cm square haft turned 45 degrees read 5.7 cm and was
# too thick for a grip, and 4 cm unturned. A grip is judged by its girth.
CALIPER_DIRECTIONS = np.stack([np.cos(np.arange(8) * np.pi / 8), np.sin(np.arange(8) * np.pi / 8)], 1)


@dataclass
class Slice:
    s: float  # the slice's middle along the length (metres from the frame's middle)
    middle: np.ndarray  # the part's middle across: (width, thickness)
    width: float  # the part's widest across
    flatness: float  # widest across over its thinnest
    count: int
    girth: float = 0.0  # the mean of its widths round it


def slice_parts(local, count=SLICES):
    """The weapon cut across its length into `count` slices: [(s, [part, ...])], each slice's parts (points closer than
    `PART_GAP_M` across are one part, as (width, thickness) arrays), largest first."""
    s = local[:, 0]
    edges = np.linspace(s.min(), s.max(), count + 1)
    found = []
    for k in range(count):
        inside = (s >= edges[k]) & (s < edges[k + 1]) if k < count - 1 else (s >= edges[k])
        across = local[inside, 1:]
        if len(across) < 6:
            continue
        found.append(((edges[k] + edges[k + 1]) / 2, parts_of(across)))
    return found


def measured_part(s, part):
    """A slice's measures of one of its parts."""
    widths = np.ptp(part @ CALIPER_DIRECTIONS.T, axis=0) if len(part) else np.zeros(len(CALIPER_DIRECTIONS))
    thin = max(float(widths.min()), 1e-4)
    return Slice(s, (part.min(0) + part.max(0)) / 2, float(widths.max()), float(widths.max() / thin), len(part),
                 float(widths.mean()))


def slices_of(local, count=SLICES):
    """The weapon cut across its length into `count` slices; each slice's largest part."""
    return [measured_part(s, parts[0]) for s, parts in slice_parts(local, count)]


def parts_of(points):
    """The groups of points chained within `PART_GAP_M` of each other, largest first."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    tree = cKDTree(points)
    pairs = tree.query_pairs(PART_GAP_M, output_type="ndarray")
    if len(pairs) == 0:
        return [points]
    graph = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
    _, label = connected_components(graph, directed=False)
    sizes = np.bincount(label)
    return [points[label == k] for k in np.argsort(-sizes, kind="stable") if sizes[k]]


# The haft (or blade) is followed from slice to slice: in each slice the part nearest where the line of the slices
# before it points, from the last `TRACK_SLICES` narrow ones. A slice's largest part is not always the haft: just
# under a double-bitted axe's head it was the lower corner of one bit, 80 mm off the haft's line and narrow enough to
# pass for haft, and it failed a haft straight within 4 mm (9.4 % off its line, a 28° kink; remake of 10 Nov 2026).
TRACK_SLICES = 5


def tracked_slices(local, length, count=SLICES):
    """Each slice measured on the part on the weapon's line, followed by continuity from the middle of its length
    outward: the part nearest the line the narrow slices before it point to (a guard or a head, where the line runs
    into it, is the part it meets, measured whole and left out of the following). Where a slice has one part, it is
    that part, however far it lies: a bent haft is followed round its bend, and still fails straight."""
    cut = slice_parts(local, count)
    if not cut:
        return []
    s_values = np.array([s for s, _ in cut])
    largest = [measured_part(s, parts[0]) for s, parts in cut]
    usual = np.median([x.width for x in largest])
    narrow = [len(parts) == 1 and x.width <= SHAFT_WIDTH_RATIO * usual for x, (_, parts) in zip(largest, cut)]
    middle = (s_values.min() + s_values.max()) / 2
    seeds = [k for k in range(len(cut)) if narrow[k]]
    if not seeds:
        return largest
    start = min(seeds, key=lambda k: abs(s_values[k] - middle))
    chosen = [None] * len(cut)
    chosen[start] = largest[start]
    for step in (1, -1):
        followed = [largest[start]]
        k = start + step
        while 0 <= k < len(cut):
            s, parts = cut[k]
            recent = followed[-TRACK_SLICES:]
            if len(recent) >= 2:
                fit = np.polyfit([x.s for x in recent], np.array([x.middle for x in recent]), 1)
                expected = fit[0] * s + fit[1]
            else:
                expected = recent[-1].middle
            candidates = [measured_part(s, part) for part in parts]
            nearest = min(candidates, key=lambda x: float(np.linalg.norm(x.middle - expected)))
            chosen[k] = nearest
            if nearest.width <= SHAFT_WIDTH_RATIO * usual:
                followed.append(nearest)
            k += step
    return chosen


def shaft_slices(slices, length, one_stretch=False):
    """The slices of the blade or shaft: away from the ends, no wider than `SHAFT_WIDTH_RATIO` times the middle width
    (a guard, a pommel, an axe's head are wider). `one_stretch` (a haft or a spear's shaft): only its longest unbroken
    stretch, so the haft stops where the head begins (a spike above an axe's head is not haft)."""
    if not slices:
        return []
    s0, s1 = min(x.s for x in slices), max(x.s for x in slices)
    usual = np.median([x.width for x in slices])
    good = [s0 + END_SHARE * length <= x.s <= s1 - END_SHARE * length and x.width <= SHAFT_WIDTH_RATIO * usual
            for x in slices]
    if not one_stretch:
        return [x for x, ok in zip(slices, good) if ok]
    runs = _runs(good)
    if not runs:
        return []
    first, last = max(runs, key=lambda r: slices[r[1]].s - slices[r[0]].s)
    return slices[first:last + 1]


def line_measures(slices, length, curved):
    """The bend (the middles' largest distance off their straight line, metres) and the largest kink (degrees). For
    a curved weapon the bend is not measured and a kink is the turning's change from its neighbours'."""
    if len(slices) < 6:
        return None, None
    s = np.array([x.s for x in slices])
    middles = np.array([x.middle for x in slices])
    fit = np.polyfit(s, middles, 1)
    off = middles - (np.outer(s, fit[0]) + fit[1])
    bend = None if curved else float(np.linalg.norm(off, axis=1).max())
    # The line's direction over each stretch, and its turn from one stretch to the next.
    stretch = KINK_STRETCH_SHARE * length
    bins = np.floor((s - s.min()) / stretch).astype(int)
    directions = []
    for b in np.unique(bins):
        inside = bins == b
        if inside.sum() >= 3:
            slope = np.polyfit(s[inside], middles[inside], 1)[0]
            d = np.array([1.0, slope[0], slope[1]])
            directions.append(d / np.linalg.norm(d))
    if len(directions) < 3:
        return bend, 0.0
    turns = np.array([np.degrees(np.arccos(np.clip(a @ b, -1, 1))) for a, b in zip(directions, directions[1:])])
    if curved:
        kink = float(np.abs(turns[1:-1] - (turns[:-2] + turns[2:]) / 2).max()) if len(turns) >= 3 else 0.0
    else:
        kink = float(turns.max())
    return bend, kink


def _runs(good, offset=0):
    runs, start = [], None
    for k, ok in enumerate(list(good) + [False]):
        if ok and start is None:
            start = k
        elif not ok and start is not None:
            runs.append((start + offset, k - 1 + offset))
            start = None
    return runs


def haft_from_butt(slices, first, last, toward_tip):
    """The run slices[first..last] with its butt end trimmed to where the haft proper begins: the first stretch, from
    the butt, at least a hand wide (`GRIP_LEAST_LENGTH_M`) whose girth stays within `HAFT_EVEN_SHARE` of the run's
    middle girth. Only the butt end is trimmed: a band or a binding further up a shaft is held over. (first, last);
    the run itself when no such stretch is found."""
    girths = np.array([x.girth for x in slices[first:last + 1]])
    even = np.abs(girths / np.median(girths) - 1) <= HAFT_EVEN_SHARE
    stretches = _runs(even, first)
    long_enough = [r for r in stretches if abs(slices[r[1]].s - slices[r[0]].s) >= GRIP_LEAST_LENGTH_M - 1e-9]
    if not long_enough:
        return first, last
    if toward_tip == 1:
        return min(r[0] for r in long_enough), last
    return first, max(r[1] for r in long_enough)


def grip_runs(slices):
    """Stretches of slices thin and round enough for a hand: [(first, last)] slice indices. Each is narrowed to its
    longest stretch no thicker than `GRIP_SWELL` times its usual thickness: a round pommel or a collar beside a grip
    is thin enough to hold but is not where the hand goes."""
    good = [GRIP_LEAST_M <= x.girth <= GRIP_MOST_M and x.flatness <= GRIP_MOST_FLATNESS for x in slices]
    narrowed = []
    for first, last in _runs(good):
        widths = np.array([x.girth for x in slices[first:last + 1]])
        even = _runs(widths <= GRIP_SWELL * np.median(widths), first)
        narrowed.append(max(even, key=lambda r: r[1] - r[0]))
    return narrowed


# A blade's grip is beside its guard (the widest slice), on the guard's shorter side, and at most this long (a
# two-handed sword's is about 30 cm); a thin blade comes out thick enough to hold (a rapier's blade, 2 cm by 1 cm, was
# taken for an 83 cm grip, its tip mark at the pommel). A "grip" over half a blade
# weapon's length is its blade, and fails.
BLADE_GRIP_MOST_M = 0.30
GUARD_REACH_SLICES = 3
GUARD_SHARE = 0.8
MOST_BLADE_GRIP_SHARE = 0.5


def hilt_run(slices, runs):
    """A blade's grip among the stretches thin enough to hold: the one that meets the guard (the widest slice) on the
    guard's shorter side, toward the nearer end; else the longest."""
    widths = np.array([x.width for x in slices])
    widest = int(np.argmax(widths))
    # The guard: the slices round the widest that are nearly as wide (its slices differ by a sample or two).
    wide = widths >= GUARD_SHARE * widths[widest]
    low = high = widest
    while low > 0 and wide[low - 1]:
        low -= 1
    while high < len(slices) - 1 and wide[high + 1]:
        high += 1
    toward_low = low < len(slices) - 1 - high
    beside = [r for r in runs if (toward_low and 0 <= low - r[1] <= GUARD_REACH_SLICES)
              or (not toward_low and 0 <= r[0] - high <= GUARD_REACH_SLICES)]
    return max(beside or runs, key=lambda r: slices[r[1]].s - slices[r[0]].s)


@dataclass
class Grip:
    start: float  # along the length (metres in the frame), the run's end toward the butt
    end: float  # its end toward the tip
    hold: float  # where the hand's middle goes
    middle_across: np.ndarray  # the run's mean middle across
    thickness: float  # the run's largest girth (metres)
    found_length: float = 0.0  # the stretch thin enough to hold, before a blade's grip was cut to its most
    along: np.ndarray = None  # the run's slices along the length, and their middles across
    middles: np.ndarray = None

    def across_at(self, s):
        """The grip's middle across at `s` along: the median of its slices' middles within half a hand of it. A frame
        tilted by a one-sided head (its longest spread is not quite the haft's) put a bearded axe's hold 3 cm off its
        haft at the run's mean middle; a bow's grip curves, so no straight line through it will do."""
        if self.along is None or not len(self.along):
            return self.middle_across
        near = np.abs(self.along - s) <= GRIP_LEAST_LENGTH_M / 2
        if not near.any():
            near = np.abs(self.along - s) == np.abs(self.along - s).min()
        return np.median(self.middles[near], axis=0)


def grip_and_tip(slices, local, weapon_kind, length):
    """The grip (or None) and which end the tip is (+1 the top of the picture, -1 the bottom)."""
    if not slices:
        return None, 1
    s_values = np.array([x.s for x in slices])
    low, high = s_values.min(), s_values.max()
    third = (high - low) / 3
    widest_low = max((x.width for x in slices if x.s <= low + third), default=0.0)
    widest_high = max((x.width for x in slices if x.s >= high - third), default=0.0)
    runs = [r for r in grip_runs(slices) if slices[r[1]].s - slices[r[0]].s >= GRIP_LEAST_LENGTH_M - 1e-9]
    if weapon_kind == "bow":
        tip_end = 1
        middle = (low + high) / 2
        runs = [r for r in runs if slices[r[0]].s <= middle <= slices[r[1]].s] or runs
    if not runs:
        return None, 1 if widest_high >= widest_low else -1
    if weapon_kind == "bow":
        first, last = min(runs, key=lambda r: min(abs(slices[r[0]].s - (low + high) / 2),
                                                  abs(slices[r[1]].s - (low + high) / 2)))
    elif weapon_kind in ("blade", "curved blade"):
        first, last = hilt_run(slices, runs)
    else:
        first, last = max(runs, key=lambda r: slices[r[1]].s - slices[r[0]].s)
        # The haft proper: the run's butt end trimmed of a butt cap, its neck and a collar. A head is at the wider end.
        first, last = haft_from_butt(slices, first, last, 1 if widest_high >= widest_low else -1)
    a, b = slices[first].s, slices[last].s
    found_length = abs(b - a)
    run_middle = (a + b) / 2
    if weapon_kind in ("blade", "curved blade"):
        # The tip is the end away from the hilt.
        tip_end = 1 if (high - run_middle) > (run_middle - low) else -1
        # A blade's grip is at most `BLADE_GRIP_MOST_M`, kept at its guard's end.
        guard_end = b if tip_end == 1 else a
        if found_length > BLADE_GRIP_MOST_M:
            a, b = ((guard_end - BLADE_GRIP_MOST_M, guard_end) if tip_end == 1 else
                    (guard_end, guard_end + BLADE_GRIP_MOST_M))
            run_middle = (a + b) / 2
    elif weapon_kind != "bow":
        # A head is at the wider end.
        tip_end = 1 if widest_high >= widest_low else -1
    butt, toward_tip = (a, b) if tip_end == 1 else (b, a)
    if weapon_kind in ("blade", "curved blade", "bow"):
        hold = run_middle if weapon_kind != "bow" else float(np.clip((low + high) / 2, min(a, b), max(a, b)))
    elif weapon_kind == "spear":
        butt_end = low if tip_end == 1 else high
        hold = butt_end + tip_end * SPEAR_HOLD_SHARE * length
        hold = float(np.clip(hold, min(a, b), max(a, b)))
    else:
        hold = butt + tip_end * min(HAFT_HOLD_IN_M, abs(b - a) / 2)
    inside = [x for x in slices[first:last + 1] if min(a, b) - 1e-9 <= x.s <= max(a, b) + 1e-9]
    along = np.array([x.s for x in inside])
    middles = np.array([x.middle for x in inside])
    thickness = max(x.girth for x in inside)
    return Grip(butt, toward_tip, float(hold), np.mean(middles, axis=0), float(thickness), float(found_length), along,
                middles), tip_end


# A bow's string, drawn by our own code (the shape model sees a bowstring under a pixel wide): a cord this thick
# between the bow's two tips, drawn only when the bow stands at least `LEAST_BRACE_SHARE` of its length from the line
# between them (a strung bow's brace is about a tenth; an unstrung stave, or one seen edge-on, is nearly straight).
STRING_RADIUS_M = 0.0015
STRING_SIDES = 8
LEAST_BRACE_SHARE = 0.05
# The bow may lie this share of its length to the string's other side (its tips' own thickness) and still be a D.
ONE_SIDE_SLACK_SHARE = 0.01
# A bow's curve may dip this share of its length between two humps (its grip, thicker, moves its middle a little).
MOST_DIP_SHARE = 0.02
# The string's ends reach this far into each tip, so it meets the bow.
STRING_INTO_TIP_M = 0.01
STRING_COLOUR = (0.86, 0.81, 0.69)


def bow_string(corners, triangles):
    """The string between a bow's tips: {"from", "to", "brace_m", "corners", "triangles"}, or {"brace_m", "why"} when
    the bow is too straight to string."""
    points, _, _ = M.sampled(corners, triangles, SAMPLES, seed=4)
    frame = frame_of(points)
    local = frame.local(points)
    length = float(np.ptp(local[:, 0]))
    nocks = [tip_point(local, end) for end in (-1, 1)]
    chord = nocks[1] - nocks[0]
    slices = slices_of(local)
    # The bow's middles away from the line between its tips (across it: width and thickness).
    offsets = np.array([np.array(x.middle) - (nocks[0] + chord * (x.s - nocks[0][0]) / chord[0])[1:]
                        for x in slices]).reshape(-1, 2)
    distances = np.linalg.norm(offsets, axis=1)
    brace = float(distances.max()) if len(distances) else 0.0
    if brace < LEAST_BRACE_SHARE * length:
        return {"brace_m": brace, "why": f"the bow stands {brace * 100:.0f} cm from the line between its tips, under "
                                         f"{LEAST_BRACE_SHARE:.0%} of its length: too straight to string"}
    # A strung bow is one curve to one side of its string (a D). A bow curving both ways (an S) would be crossed by
    # it; a bow waving (a W, the first trial's) is no strung bow: the deepest dip between two humps tells.
    outward = offsets[distances.argmax()] / brace
    away = offsets @ outward
    other_side = float(-away.min())
    if other_side > ONE_SIDE_SLACK_SHARE * length:
        return {"brace_m": brace, "why": f"the bow curves both ways, {other_side * 100:.0f} cm to the other side of "
                                         "the line between its tips: a string would cross it"}
    dip = float((np.minimum(np.maximum.accumulate(away), np.maximum.accumulate(away[::-1])[::-1]) - away).max())
    if dip > MOST_DIP_SHARE * length:
        return {"brace_m": brace, "why": f"the bow waves, dipping {dip * 100:.0f} cm between two humps: not one curve "
                                         "to string"}
    direction = chord / np.linalg.norm(chord)
    ends = [nocks[0] - direction * STRING_INTO_TIP_M, nocks[1] + direction * STRING_INTO_TIP_M]
    tube_corners, tube_triangles = tube(frame.world(ends[0]), frame.world(ends[1]), STRING_RADIUS_M, STRING_SIDES)
    return {"from": frame.world(nocks[0]).tolist(), "to": frame.world(nocks[1]).tolist(), "brace_m": brace,
            "corners": tube_corners, "triangles": tube_triangles}


def tube(start, end, radius, sides):
    """A closed tube (a prism of `sides` sides with flat ends) from `start` to `end`, wound outward."""
    axis = end - start
    along = axis / np.linalg.norm(axis)
    across = np.cross(along, [0.0, 0.0, 1.0] if abs(along[2]) < 0.9 else [1.0, 0.0, 0.0])
    across /= np.linalg.norm(across)
    other = np.cross(along, across)
    turns = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.cos(turns)[:, None] * across + np.sin(turns)[:, None] * other
    corners = np.concatenate([start + radius * ring, end + radius * ring, [start, end]])
    triangles = []
    for k in range(sides):
        a, b = k, (k + 1) % sides
        triangles += [[a, b, sides + b], [a, sides + b, sides + a]]
        triangles += [[2 * sides, b, a], [2 * sides + 1, sides + a, sides + b]]
    return corners, np.array(triangles, dtype=np.int64)


def tip_point(local, tip_end):
    s = local[:, 0] * tip_end
    near = local[s >= s.max() - TIP_REACH_M]
    return near.mean(0)


def mirror_across(local, frame_width, surface):
    """The mean distance of the surface mirrored across its length-and-thickness plane through its line, as a share
    of its width; `surface` gives a point's distance from the surface (`mesh_tools.SurfaceDistance`, in the frame)."""
    s = local[:, 0]
    fit = np.polyfit(s, local[:, 1], 1)
    line = fit[0] * s + fit[1]
    mirrored = local.copy()
    mirrored[:, 1] = 2 * line - local[:, 1]
    return float(surface(mirrored[::3]).mean() / max(frame_width, 1e-9))


def mirror_along(local, length, about, surface):
    mirrored = local.copy()
    mirrored[:, 0] = 2 * about - local[:, 0]
    return float(surface(mirrored[::3]).mean() / max(length, 1e-9))


def measured(corners, triangles, subject):
    """Every weapon measure of a surface already at its size: a dict for the report and the export, and the checks."""
    points, _, chosen = M.sampled(corners, triangles, SAMPLES, seed=2)
    frame = frame_of(points)
    local = frame.local(points)
    length = float(np.ptp(local[:, 0]))
    width = float(np.ptp(local[:, 1]))
    out = {"length_m": length, "width_m": width, "thickness_m": float(np.ptp(local[:, 2]))}
    checks = []
    kind = subject.weapon_kind or "hafted"
    if kind == "shield":
        return shield_measures(corners, triangles, subject, out, checks)
    # A bow is one curve with nothing beside it: its slices are their largest parts. Every other weapon is followed
    # along its line, so a head's corner beside the haft is never read as haft.
    slices = slices_of(local) if kind == "bow" else tracked_slices(local, length)
    shaft = shaft_slices(slices, length, one_stretch=kind in ("hafted", "spear"))
    curved = kind in ("curved blade", "bow")
    bend, kink = line_measures(shaft if not curved else slices, length, curved)
    out["bend_m"], out["kink_deg"] = bend, kink
    what = {"blade": "blade", "spear": "shaft", "hafted": "haft"}.get(kind, "line")
    if bend is not None:
        checks.append(Check(f"{what} straight", bend / length * 100, MOST_BEND_SHARE * 100, "% of length",
                            bend <= MOST_BEND_SHARE * length,
                            f"the {what}'s middle strays {bend * 1000:.1f} mm ({bend / length:.2%} of the length) "
                            "from its straight line"))
    if kink is not None:
        checks.append(Check("no kink", kink, MOST_KINK_DEG, "°", kink <= MOST_KINK_DEG,
                            f"the {what}'s direction turns {kink:.1f}° at most between neighbouring tenths of its "
                            "length" + (" beyond its own curve" if curved else "")))
    elif kind != "shield":
        checks.append(Check(f"{what} straight", 0, MOST_BEND_SHARE * 100, "% of length", False,
                            f"too few slices of a {what} were found to follow its line"))
    grip, tip_end = grip_and_tip(slices, local, kind, length)
    if grip is None:
        checks.append(Check("grip found", 0, GRIP_LEAST_LENGTH_M * 100, "cm", False,
                            f"no stretch {GRIP_LEAST_LENGTH_M * 100:.0f} cm long is between {GRIP_LEAST_M * 100:.1f} "
                            f"and {GRIP_MOST_M * 100:.1f} cm thick and round enough for a hand"))
    else:
        run = abs(grip.end - grip.start)
        s_low = local[:, 0].min()
        from_butt = (grip.hold - s_low) if tip_end == 1 else (local[:, 0].max() - grip.hold)
        bladed = kind in ("blade", "curved blade")
        on_blade = bladed and grip.found_length > MOST_BLADE_GRIP_SHARE * length
        checks.append(Check("grip found", run * 100, GRIP_LEAST_LENGTH_M * 100, "cm", not on_blade,
                            f"the stretch thin enough to hold runs {grip.found_length * 100:.0f} cm, over half the "
                            f"{length * 100:.0f} cm length: it is the blade, not a grip" if on_blade else
                            f"a grip {run * 100:.0f} cm long, {grip.thickness * 100:.1f} cm across at most, held "
                            f"{from_butt * 100:.0f} cm ({from_butt / length:.0%} of the length) from the butt"))
        hold_local = np.array([grip.hold, *grip.across_at(grip.hold)])
        start_local = np.array([grip.start, *grip.across_at(grip.start)])
        end_local = np.array([grip.end, *grip.across_at(grip.end)])
        out["grip"] = {"hold": frame.world(hold_local).tolist(), "from": frame.world(start_local).tolist(),
                       "to": frame.world(end_local).tolist(), "thickness_m": grip.thickness,
                       "from_butt_m": float(from_butt)}
    tip = frame.world(tip_point(local, tip_end))
    out["tip"] = tip.tolist()
    out["toward_tip"] = (frame.length_axis * tip_end).tolist()
    out["flat_normal"] = frame.thickness_axis.tolist()
    # Mirrored points are measured to the surface's triangles, not to its samples: a sample cloud's own spacing (about
    # 1 mm at 60,000 samples on a 3 m spear) divided by a 3 cm width failed a perfectly round shaft.
    surface = M.SurfaceDistance(frame.local(corners), triangles, local, chosen)
    if subject.symmetric == "across":
        share = mirror_across(local, width, surface)
        out["across_share"] = share
        checks.append(Check(f"symmetric across the {what}", share * 100, MOST_ACROSS_SHARE * 100, "% of width",
                            share <= MOST_ACROSS_SHARE,
                            f"mirrored across its {what}, the surface lies {share * width * 1000:.1f} mm "
                            f"({share:.1%} of its width) from itself on average"))
    elif subject.symmetric == "along":
        about = grip.hold if grip else float((local[:, 0].min() + local[:, 0].max()) / 2)
        share = mirror_along(local, length, about, surface)
        out["along_share"] = share
        checks.append(Check("limbs mirror each other", share * 100, MOST_ALONG_SHARE * 100, "% of length",
                            share <= MOST_ALONG_SHARE,
                            f"mirrored about its grip, the surface lies {share * length * 1000:.1f} mm "
                            f"({share:.1%} of its length) from itself on average"))
    return out, checks


# A shield's handles: what stands off its back, found against the board's back (a smooth surface, quadratic across
# the shield, fitted to the backmost point of each cell `BACK_CELL_SHARE` of its size wide, the cells standing off it
# by more than `BACK_FIT_MOST_M` left out of the fit, again and again). A handle is a loop: a bar a hand closes round
# (`GRIP_LEAST_M` to `GRIP_MOST_M` round, at least `GRIP_LEAST_LENGTH_M` long) at least `HANDLE_LEAST_STAND_M` off
# the board, with at least `HANDLE_LEAST_GAP_M` of open space between the board and the bar's middle for the fingers.
# The back's middle by convention would put the grip of a shield with two D-handles, left and right of its middle, on
# the bare boards between them.
BACK_CELL_SHARE = 1 / 40
BACK_FIT_MOST_M = 0.01
BACK_FIT_ROUNDS = 4
HANDLE_STAND_M = 0.012
HANDLE_LEAST_STAND_M = 0.03
HANDLE_LEAST_GAP_M = 0.015
HANDLE_BAR_SHARE = 0.7
# With two handles the forearm goes through one and the hand holds the other. By convention the shield is carried on
# the left arm, the forearm across the back: seen from behind (looking through the back toward the front, +Z), the
# elbow's loop is on the left and the hand's on the right, which is -X; handles one above the other (more apart up
# than across) are held with the forearm raised, the hand in the upper one.


def back_surface(points, size):
    """The board's back: a quadratic z(x, y) fitted to the backmost point of each cell, leaving out what stands off
    it (handles, straps): its coefficients."""
    cell = max(size * BACK_CELL_SHARE, 0.005)
    key = np.floor(points[:, :2] / cell).astype(np.int64)
    order = np.lexsort((points[:, 2], key[:, 1], key[:, 0]))
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(key[order][1:] != key[order][:-1], axis=1)
    backmost = points[order[first]]

    def terms(xy):
        x, y = xy[:, 0], xy[:, 1]
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1)

    kept = np.ones(len(backmost), dtype=bool)
    coefficients = np.zeros(6)
    for _ in range(BACK_FIT_ROUNDS):
        coefficients = np.linalg.lstsq(terms(backmost[kept, :2]), backmost[kept, 2], rcond=None)[0]
        kept = terms(backmost[:, :2]) @ coefficients - backmost[:, 2] <= BACK_FIT_MOST_M
    return lambda xy: terms(xy) @ coefficients


def handles_of(points, size):
    """The handles standing off a shield's back: [{"middle", "from", "to", "direction", "girth_m", "stand_m",
    "gap_m", "length_m"}], and what else stood off it that is no handle ([words])."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    back = back_surface(points, size)
    stand = back(points[:, :2]) - points[:, 2]
    off = points[stand > HANDLE_STAND_M]
    off_stand = stand[stand > HANDLE_STAND_M]
    if len(off) < 20:
        return [], []
    pairs = cKDTree(off).query_pairs(PART_GAP_M, output_type="ndarray")
    graph = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(off), len(off)))
    _, label = connected_components(graph, directed=False)
    handles, other = [], []
    for k in np.unique(label):
        piece, piece_stand = off[label == k], off_stand[label == k]
        if len(piece) < 20:
            continue
        found = handle_of(piece, piece_stand)
        (handles if isinstance(found, dict) else other).append(found)
    return handles, other


def handle_of(piece, stand):
    """A piece standing off the back judged as a handle: its measures, or why it is none (words)."""
    most = float(stand.max())
    where = f"a piece standing {most * 100:.1f} cm off the back at ({piece[:, 0].mean():+.2f}, {piece[:, 1].mean():+.2f})"
    if most < HANDLE_LEAST_STAND_M:
        return f"{where}: too low to put a hand through"
    bar = piece[stand >= HANDLE_BAR_SHARE * most]
    middle_xy = bar[:, :2].mean(0)
    spread = bar[:, :2] - middle_xy
    direction = np.linalg.svd(spread, full_matrices=False)[2][0] if len(bar) > 2 else np.array([0.0, 1.0])
    # Along +Y, or +X for a bar lying across: a hand's fingers close round the node's +Y.
    if (abs(direction[1]) >= abs(direction[0]) and direction[1] < 0) or (abs(direction[1]) < abs(direction[0])
                                                                          and direction[0] < 0):
        direction = -direction
    along = spread @ direction
    length = float(np.ptp(along))
    if length < GRIP_LEAST_LENGTH_M - 1e-9:
        return f"{where}: {length * 100:.0f} cm long, too short for a hand"
    # The bar's middle third: its girth (across it in the board's plane, and its depth off the board) and the open
    # space under it, between the board and the bar's inner face.
    across = np.array([-direction[1], direction[0]])
    piece_along = (piece[:, :2] - middle_xy) @ direction
    third = np.abs(piece_along - np.median(along)) <= length / 6
    in_bar = third & (stand >= HANDLE_BAR_SHARE * most)
    if in_bar.sum() < 6:
        return f"{where}: no bar to hold"
    section = np.stack([(piece[in_bar, :2] - middle_xy) @ across, stand[in_bar]], 1)
    girth = float(np.ptp(section @ CALIPER_DIRECTIONS.T, axis=0).mean())
    gap = float(stand[third].min()) if third.any() else 0.0
    if not GRIP_LEAST_M <= girth <= GRIP_MOST_M:
        return f"{where}: its bar is {girth * 100:.1f} cm round, not {GRIP_LEAST_M * 100:.1f} to {GRIP_MOST_M * 100:.1f}"
    if gap < HANDLE_LEAST_GAP_M:
        return f"{where}: solid to the board under its middle ({gap * 100:.1f} cm open), no loop for the fingers"
    z_middle = float(piece[in_bar, 2].mean())
    ends = [middle_xy + direction * float(along.min()), middle_xy + direction * float(along.max())]
    return {"middle": [float(middle_xy[0]), float(middle_xy[1]), z_middle],
            "from": [float(ends[0][0]), float(ends[0][1]), z_middle],
            "to": [float(ends[1][0]), float(ends[1][1]), z_middle],
            "direction": [float(direction[0]), float(direction[1]), 0.0], "girth_m": girth, "stand_m": most,
            "gap_m": gap, "length_m": length}


def held_handle(handles):
    """Which handle the hand holds (index), and the forearm's loop (index or None), by the convention above."""
    if len(handles) == 1:
        return 0, None
    middles = np.array([h["middle"][:2] for h in handles])
    spread = np.ptp(middles, axis=0)
    hand = int(np.argmin(middles[:, 0])) if spread[0] >= spread[1] else int(np.argmax(middles[:, 1]))
    arm = int(np.argmax(np.linalg.norm(middles - middles[hand], axis=1)))
    return hand, arm


def shield_measures(corners, triangles, subject, out, checks):
    """A shield: its face toward the viewer (+Z); its grip on the handle the hand holds, found on its back (a loop
    standing off the board); where none is found, on the back's middle, said so (the shape model sees only the front,
    so a strap behind it may not be made); the tip is not marked, its front's middle is."""
    low, high = corners.min(0), corners.max(0)
    middle = (low + high) / 2
    size = float(max(high[0] - low[0], high[1] - low[1]))
    out["front"] = [float(middle[0]), float(middle[1]), float(high[2])]
    out["toward_tip"] = [0.0, 1.0, 0.0]
    out["flat_normal"] = [0.0, 0.0, 1.0]
    out["size_m"] = size
    points, _, _ = M.sampled(corners, triangles, SAMPLES, seed=2)
    handles, other = handles_of(points, size)
    out["handles"] = handles
    out["not_handles"] = other
    if not handles:
        out["grip"] = {"hold": [float(middle[0]), float(middle[1]), float(low[2])],
                       "from": [float(middle[0]), float(middle[1] - 0.06), float(low[2])],
                       "to": [float(middle[0]), float(middle[1] + 0.06), float(low[2])],
                       "thickness_m": None, "from_butt_m": None, "by_convention": True, "handle": "not found"}
        checks.append(Check("handle found", 0, 1, "handles", False,
                            "no handle was found on the back (a loop standing off the board a hand closes round): "
                            "the grip mark is on the back's middle by convention"
                            + (f"; {len(other)} other piece(s) stand off it: " + "; ".join(other[:3]) if other else ""),
                            counted=False))
        return out, checks
    hand, arm = held_handle(handles)
    held = handles[hand]
    out["grip"] = {"hold": held["middle"], "from": held["from"], "to": held["to"], "thickness_m": held["girth_m"],
                   "from_butt_m": None, "by_convention": False, "handle": hand, "arm_loop": arm}
    out["toward_tip"] = held["direction"]
    side = ("" if arm is None else
            f"; the forearm goes through the other at ({handles[arm]['middle'][0]:+.2f}, "
            f"{handles[arm]['middle'][1]:+.2f}) m")
    checks.append(Check("handle found", len(handles), 1, "handles", True,
                        f"{len(handles)} handle(s) on the back; the hand holds the one at ({held['middle'][0]:+.2f}, "
                        f"{held['middle'][1]:+.2f}) m, {held['length_m'] * 100:.0f} cm long, {held['girth_m'] * 100:.1f} "
                        f"cm round, standing {held['stand_m'] * 100:.1f} cm off the board{side}", counted=False))
    return out, checks
