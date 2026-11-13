"""Known shapes for the tests, built from boxes of voxels and closed by marching cubes, so each is one whole surface
(as the shape model's are): a chair (and one with a short leg, one leaning), a sword (and a bent one, one with its
pommel floating), a double-bitted axe (and one bent, one with a butt cap), a bearded axe, a shield with handles."""

import numpy as np
from skimage import measure


def surface_of(boxes, cell, cut=()):
    """The surface round the union of `boxes` (each ((x0, y0, z0), (x1, y1, z1)) in metres) less the `cut` boxes, at
    `cell` metres a voxel: (corners, triangles)."""
    low = np.min([b[0] for b in boxes], axis=0) - 2 * cell
    high = np.max([b[1] for b in boxes], axis=0) + 2 * cell
    shape = np.ceil((high - low) / cell).astype(int) + 1
    axes = [low[k] + cell * np.arange(shape[k]) for k in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    solid = np.zeros(shape, dtype=bool)

    def inside(box):
        (x0, y0, z0), (x1, y1, z1) = box
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z >= z0) & (z <= z1)

    for box in boxes:
        solid |= inside(box)
    for box in cut:
        solid &= ~inside(box)
    corners, triangles, _, _ = measure.marching_cubes(solid.astype(np.float32), 0.5)
    corners = corners * cell + low
    a, b, c = (corners[triangles[:, k]] for k in range(3))
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        triangles = triangles[:, ::-1]
    return corners.astype(np.float64), np.ascontiguousarray(triangles, dtype=np.int64)


def chair(short_leg=0.0, lean=0.0, cell=0.012, missing_leg=False, moved_leg=0.0, armrest=False):
    """A chair 0.95 m high: four legs 4 cm square, a seat at 45 cm, a back. `short_leg` shortens one front leg;
    `lean` (metres) moves the seat, back and upper legs that far toward +X, as a chair falling over. Faults the
    symmetry check must catch: `missing_leg` leaves the front -X leg out, `moved_leg` (metres) moves it that far
    toward +X, `armrest` puts an armrest on the +X side only."""
    legs = [((x + (moved_leg if (x < 0 and z > 0) else 0.0), 0.0 + (short_leg if (x < 0 and z > 0) else 0.0), z),
             (x + 0.04 + (moved_leg if (x < 0 and z > 0) else 0.0), 0.45, z + 0.04))
            for x in (-0.22, 0.18) for z in (-0.22, 0.18) if not (missing_leg and x < 0 and z > 0)]
    seat = ((-0.22 + lean, 0.42, -0.22), (0.22 + lean, 0.47, 0.22))
    back = ((-0.22 + lean, 0.47, -0.22), (0.22 + lean, 0.95, -0.18))
    arms = [((0.18, 0.47, -0.18), (0.22, 0.68, -0.14)), ((0.18, 0.64, -0.14), (0.22, 0.68, 0.18)),
            ((0.18, 0.47, 0.14), (0.22, 0.64, 0.18))] if armrest else []
    return surface_of(legs + [seat, back] + arms, cell)


def sword(bend=0.0, pommel_gap=0.0, cell=0.0025):
    """A sword 0.95 m long, upright, point up: a blade 5 cm wide and 8 mm thick, a guard 20 cm, a round-ish grip
    12 cm, a pommel. `bend` shifts the blade's upper half sideways (metres); `pommel_gap` leaves the pommel that far
    below the grip, floating."""
    grip = ((-0.0125, 0.05, -0.0125), (0.0125, 0.17, 0.0125))
    pommel = ((-0.025, 0.0 - pommel_gap, -0.02), (0.025, 0.05 - pommel_gap - (cell * 2 if pommel_gap else 0), 0.02))
    guard = ((-0.10, 0.17, -0.015), (0.10, 0.20, 0.015))
    lower_blade = ((-0.025, 0.20, -0.004), (0.025, 0.55, 0.004))
    upper_blade = ((-0.025 + bend, 0.55, -0.004), (0.025 + bend, 0.95, 0.004))
    return surface_of([grip, pommel, guard, lower_blade, upper_blade], cell)


def rapier(cell=0.0025, hilt=True):
    """A rapier 1.05 m long, upright, point up, its blade thin as a shape model makes a thin blade: 2 cm wide and 1 cm
    thick (round enough to hold); a guard 18 cm, a round grip 12 cm, a pommel.
    `hilt=False`: the blade alone."""
    blade = ((-0.01, 0.20, -0.005), (0.01, 1.05, 0.005))
    if not hilt:
        return surface_of([((-0.01, 0.0, -0.005), (0.01, 1.05, 0.005))], cell)
    pommel = ((-0.02, 0.0, -0.02), (0.02, 0.05, 0.02))
    grip = ((-0.0125, 0.05, -0.0125), (0.0125, 0.17, 0.0125))
    guard = ((-0.09, 0.17, -0.012), (0.09, 0.20, 0.012))
    return surface_of([pommel, grip, guard, blade], cell)


def bow(bend=0.18, cell=0.006, s_shape=False):
    """A bow 1.4 m long, upright, its stave 3 cm square bent `bend` metres out at its middle (a strung bow's curve; 0
    is a straight stave; `s_shape`: bent one way above its middle and the other below), made of short boxes."""
    boxes = []
    for k in range(70):
        y = 0.02 * k
        x = bend * np.sin(2 * np.pi * y / 1.4) if s_shape else bend * (1 - ((y - 0.7) / 0.7) ** 2)
        boxes.append(((x - 0.015, y, -0.015), (x + 0.015, y + 0.025, 0.015)))
    return surface_of(boxes, cell)


def double_axe(cell=0.004, kink_deg=0.0, butt_cap=False):
    """A double-bitted axe 0.9 m long: a round-ish haft 3 cm, two equal bits at the top. `kink_deg` bends the haft's
    upper half that many degrees sideways at its middle (a bent haft; the head follows it); `butt_cap` puts a black
    butt piece on its end as a remade axe had: a flared knob, a thin neck, a collar."""
    if not kink_deg:
        haft = [((-0.015, 0.0, -0.015), (0.015, 0.90, 0.015))]
        top = 0.0
    else:
        slope = np.tan(np.radians(kink_deg))
        haft = [((-0.015, 0.0, -0.015), (0.015, 0.46, 0.015))]
        for k in range(22):
            y = 0.45 + 0.02 * k
            x = slope * (y + 0.01 - 0.45)
            haft.append(((x - 0.015, y, -0.015), (x + 0.015, y + 0.025, 0.015)))
        top = slope * (0.78 - 0.45)
    bits = ((top - 0.17, 0.70, -0.01), (top + 0.17, 0.86, 0.01))
    cap = []
    if butt_cap:
        haft = [((-0.015, 0.14, -0.015), (0.015, 0.90, 0.015))]
        cap = [((-0.03, 0.0, -0.03), (0.03, 0.03, 0.03)), ((-0.012, 0.03, -0.012), (0.012, 0.12, 0.012)),
               ((-0.02, 0.12, -0.02), (0.02, 0.14, 0.02))]
    return surface_of(haft + [bits] + cap, cell)


def bearded_axe(cell=0.004):
    """A bearded axe 0.9 m long: a haft 3 cm, one bit on +X at the top whose beard hangs 18 cm down beside the haft, 3
    cm off it: in the slices beside the beard, the beard (4 by 3 cm) is a larger piece than the haft, and narrow."""
    haft = ((-0.015, 0.0, -0.015), (0.015, 0.90, 0.015))
    bit = ((0.0, 0.74, -0.012), (0.17, 0.86, 0.012))
    beard = ((0.045, 0.56, -0.015), (0.085, 0.75, 0.015))
    return surface_of([haft, bit, beard], cell)


def shield(handles=((-0.15, 0.0), (0.15, 0.0)), cell=0.005, across=False, solid=False):
    """A square shield 0.6 m, its face toward +Z (a board 2 cm thick, a boss on its front) and on its back a D-handle
    at each (x, y) of `handles`: a bar 18 cm long, 2.4 by 2.5 cm, on two posts, 3.5 cm of open space under it. The
    bars run up the shield (`across`: across it). `solid` fills the space under each bar (a block, no loop)."""
    boxes = [((-0.3, -0.3, 0.0), (0.3, 0.3, 0.02)), ((-0.075, -0.075, 0.02), (0.075, 0.075, 0.06))]
    for x, y in handles:
        def box(a0, a1, b0, b1, z0, z1):
            return ((x + b0, y + a0, z0), (x + b1, y + a1, z1)) if not across else ((x + a0, y + b0, z0),
                                                                                    (x + a1, y + b1, z1))
        boxes.append(box(-0.09, 0.09, -0.012, 0.012, -0.06, -0.035))
        boxes.append(box(-0.09, -0.07, -0.012, 0.012, -0.06, 0.0))
        boxes.append(box(0.07, 0.09, -0.012, 0.012, -0.06, 0.0))
        if solid:
            boxes.append(box(-0.09, 0.09, -0.012, 0.012, -0.06, 0.0))
    return surface_of(boxes, cell)


def bench(cell=0.01):
    """A bench 1.51 m long and 0.46 m high (3.3 times as long as high): a seat 5 cm thick on four legs 5 cm square."""
    seat = ((-0.75, 0.40, -0.18), (0.75, 0.45, 0.18))
    legs = [((x, 0.0, z), (x + 0.05, 0.40, z + 0.05)) for x in (-0.70, 0.65) for z in (-0.16, 0.11)]
    return surface_of([seat] + legs, cell)


def bed_frame(cell=0.02):
    """A bed frame 2 m long, 1.4 m wide and 0.39 m high (5 times as long as high): a rail round a deck on four legs
    7 cm square. The deck is whole: a picture does not resolve the gaps between a bed's slats seen from its end."""
    rails = [((-0.70, 0.25, -1.00), (0.70, 0.39, -0.95)), ((-0.70, 0.25, 0.95), (0.70, 0.39, 1.00)),
             ((-0.70, 0.25, -1.00), (-0.65, 0.39, 1.00)), ((0.65, 0.25, -1.00), (0.70, 0.39, 1.00))]
    deck = ((-0.65, 0.30, -0.95), (0.65, 0.33, 0.95))
    legs = [((x, 0.0, z), (x + 0.07, 0.25, z + 0.07)) for x in (-0.70, 0.63) for z in (-1.00, 0.93)]
    return surface_of(rails + [deck] + legs, cell)


def three_legged_table(short_leg=0.0, cell=0.01):
    """A small table 0.65 m high, 0.75 m across: a top 3 cm thick on three legs 4 cm square at 120 degrees round it,
    as the third trial's round table. `short_leg` raises the first leg's foot that far off the ground."""
    top = ((-0.375, 0.62, -0.375), (0.375, 0.65, 0.375))
    legs = []
    for k, turn in enumerate(np.radians([0.0, 120.0, 240.0])):
        x, z = 0.28 * np.cos(turn), 0.28 * np.sin(turn)
        foot = short_leg if k == 0 else 0.0
        legs.append(((x - 0.02, foot, z - 0.02), (x + 0.02, 0.62, z + 0.02)))
    return surface_of([top] + legs, cell)


def mug(web=None, cell=0.005):
    """A mug 0.25 m high: a body 20 cm square, a handle on its +X side, a loop 5 cm across and 9 cm high inside it.
    `web`: None (the loop open), "lower" (a web 1 cm thick across its lower 4 cm, the rest open), "whole" (a web
    across all of it) or "solid" (the loop filled as deep as the body, 20 cm)."""
    body = ((-0.10, 0.0, -0.10), (0.10, 0.25, 0.10))
    top = ((0.10, 0.17, -0.02), (0.18, 0.20, 0.02))
    bottom = ((0.10, 0.05, -0.02), (0.18, 0.08, 0.02))
    outer = ((0.15, 0.05, -0.02), (0.18, 0.20, 0.02))
    boxes = [body, top, bottom, outer]
    if web == "lower":
        boxes.append(((0.10, 0.08, -0.005), (0.15, 0.12, 0.005)))
    elif web == "whole":
        boxes.append(((0.10, 0.08, -0.005), (0.15, 0.17, 0.005)))
    elif web == "solid":
        boxes.append(((0.10, 0.08, -0.10), (0.15, 0.17, 0.10)))
    return surface_of(boxes, cell)


def spear(length=3.0, width=0.03, sides=48, rings=60, lug=None):
    """A perfectly round shaft `length` long and `width` across, upright (a prism of `sides` sides in `rings` rings,
    its ends flat), made exactly, not of voxels; `lug` ((x0, y0), (x1, y1)): a box 2 cm thick sticking out of it on one
    side (a fault: a spear is mirror-symmetric across its shaft)."""
    turns = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.stack([np.cos(turns), np.zeros(sides), np.sin(turns)], 1) * width / 2
    heights = np.linspace(0.0, length, rings + 1)
    corners = np.concatenate([ring + [0.0, h, 0.0] for h in heights] + [[[0.0, 0.0, 0.0], [0.0, length, 0.0]]])
    triangles = []
    for r in range(rings):
        for k in range(sides):
            a, b = r * sides + k, r * sides + (k + 1) % sides
            triangles += [[a, b + sides, b], [a, a + sides, b + sides]]
    bottom, top = len(corners) - 2, len(corners) - 1
    for k in range(sides):
        triangles += [[bottom, k, (k + 1) % sides], [top, rings * sides + (k + 1) % sides, rings * sides + k]]
    corners, triangles = np.asarray(corners, dtype=np.float64), np.asarray(triangles, dtype=np.int64)
    a, b, c = (corners[triangles[:, k]] for k in range(3))
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        triangles = triangles[:, ::-1]
    if lug is None:
        return corners, np.ascontiguousarray(triangles)
    (x0, y0), (x1, y1) = lug
    box_corners, box_triangles = surface_of([((x0, y0, -0.01), (x1, y1, 0.01))], 0.004)
    return (np.concatenate([corners, box_corners]),
            np.concatenate([triangles, box_triangles + len(corners)]))


def ringed_box(web=False, cell=0.005):
    """A box 0.5 m high with a ring on top, facing +Z (a lantern's ring): its bars 2 cm, its hole 4 cm square (6.5 % of
    the height, under the carving's 7.8 % and the outline's 8 % gap piece). `web`: a plate 1 cm thick fills the hole."""
    body = ((-0.12, 0.0, -0.12), (0.12, 0.5, 0.12))
    bars = [((-0.04, 0.5, -0.01), (0.04, 0.52, 0.01)), ((-0.04, 0.56, -0.01), (0.04, 0.58, 0.01)),
            ((-0.04, 0.5, -0.01), (-0.02, 0.58, 0.01)), ((0.02, 0.5, -0.01), (0.04, 0.58, 0.01))]
    plate = [((-0.02, 0.52, -0.005), (0.02, 0.56, 0.005))] if web else []
    return surface_of([body] + bars + plate, cell)


def spindle_chair(panel=False, cell=0.006):
    """A chair 0.95 m high whose back is two posts, two rails and four spindles 2 cm wide, the gaps between them 5.6 cm
    (5.9 % of the height, under the carving's 7.8 %). `panel`: a panel 1 cm thick fills the back between its rails."""
    legs = [((x, 0.0, z), (x + 0.04, 0.45, z + 0.04)) for x in (-0.22, 0.18) for z in (-0.22, 0.18)]
    seat = ((-0.22, 0.42, -0.22), (0.22, 0.47, 0.22))
    back = [((-0.22, 0.47, -0.22), (-0.18, 0.95, -0.18)), ((0.18, 0.47, -0.22), (0.22, 0.95, -0.18)),
            ((-0.22, 0.47, -0.22), (0.22, 0.52, -0.18)), ((-0.22, 0.90, -0.22), (0.22, 0.95, -0.18))]
    back += [((x, 0.52, -0.21), (x + 0.02, 0.90, -0.19)) for x in (-0.124, -0.048, 0.028, 0.104)]
    if panel:
        back.append(((-0.18, 0.52, -0.205), (0.18, 0.90, -0.195)))
    return surface_of(legs + [seat] + back, cell)


def framed_panel(panel=True, cell=0.005):
    """A frame 0.6 m wide and 0.8 m high, its bars 6 cm wide and 3 cm deep, standing upright facing +Z; `panel`: a
    panel 5 mm thick inside it (a mirror's glass, a canvas)."""
    bars = [((-0.3, 0.0, -0.015), (0.3, 0.06, 0.015)), ((-0.3, 0.74, -0.015), (0.3, 0.8, 0.015)),
            ((-0.3, 0.0, -0.015), (-0.24, 0.8, 0.015)), ((0.24, 0.0, -0.015), (0.3, 0.8, 0.015))]
    inside = [((-0.24, 0.06, -0.0025), (0.24, 0.74, 0.0025))] if panel else []
    return surface_of(bars + inside, cell)
