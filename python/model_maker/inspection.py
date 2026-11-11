"""Every check of a shaped object or weapon, gathered: the shape's (outline, whole, base, symmetry, the weapon's line
and grip) and, once it is painted, the bake's coverage. Used to choose among shape tries and for the report.
"""

import numpy as np

from . import lying
from .cutout import for_shape
from . import object_checks as O
from . import weapon_checks as W
from .picture_checks import Check

# The painted views' pixels cover at least this share of the surface's texels; the rest takes colour spread from
# its neighbours (inside a hollow, under a base).
LEAST_COVERAGE = 0.9


def shape_checks(subject, picture_mask, corners, triangles, laid=None, picture=None):
    """The shape's checks and measures, the outline's picture. `corners` at the object's size, standing on y = 0;
    `laid`, the record of a thing laid on its broad face (lying.laid): its outline is judged in the frame it was made
    in, the picture's own, so the views searched are the same as for any other object."""
    checks, measures = [], {}
    low, high = corners.min(0), corners.max(0)
    measures["size_m"] = [round(float(x), 4) for x in high - low]
    measures["triangles"] = int(len(triangles))
    as_made = lying.to_made(corners, laid)
    # Judged against the cut-out the shape model was given (its largest piece, its pinholes filled): a speck the picture
    # checks let through (one piece holds 97 %) is nothing the shape model saw, and must not fail an exact shape.
    view, mean, far, apart = O.outline_judged(for_shape(picture_mask), as_made, triangles)
    overlay = O.overlay_of(view.painted, view.made)
    measures["outline"] = {"iou": view.iou, "turn_deg": view.turn, "tilt_deg": view.tilt, "distance": view.distance,
                           "edge_mean": mean, "edge_95": far, "apart": apart, "cut": view.cut}
    checks += O.outline_checks(view, mean, far, apart)
    # The holes the picture shows, open in the model (and none the picture lacks): a web too thin or a hole too narrow
    # for the outline's gap piece and the carving is judged here (docs/objects.md, "Holes the picture shows").
    from . import carving

    holes = carving.holes_open(picture_mask, as_made, triangles, picture, outline_view=view)
    measures["holes"] = holes
    checks.append(carving.holes_check(holes))
    whole = O.wholeness(corners, triangles)
    measures["whole"] = whole
    checks += O.whole_checks(whole, subject.is_weapon)
    if subject.stands:
        base = O.base_measures(corners, triangles)
        measures["base"] = base
        checks += O.base_checks(base)
    if subject.symmetric == "mirror":
        share, mirror_turn, offset, far = O.mirror_measure(corners, triangles)
        measures["mirror"] = {"share": share, "turn_deg": mirror_turn, "offset_m": offset, "far_share": far}
        checks.append(O.mirror_check(share, mirror_turn, far))
    if subject.is_weapon or subject.handled:
        weapon, weapon_checks = W.measured(corners, triangles, subject)
        measures["weapon"] = weapon
        checks += weapon_checks
    return checks, measures, overlay


class Judged:
    """A shape as the finish judges it (`judged`): the shape it exports, its checks and measures, the outline's picture,
    and the records of what was done to it (`laid`, `level`: None when it does not stand)."""

    def __init__(self, corners, triangles, checks, measures, overlay, laid, level):
        self.corners, self.triangles, self.checks, self.measures = corners, triangles, checks, measures
        self.overlay, self.laid, self.level = overlay, laid, level

    @property
    def lowered(self):
        """How far the levelling set it down (the trim), as a move of its corners."""
        return np.array([0.0, self.level["trimmed_m"] if self.level else 0.0, 0.0])

    def paint_followed(self, made_corners, painted):
        """The paint as made, following the shape: (each corner's colour, the colour cloud), the corners' colours from
        the made corner each stands on or nearest (the repair joins, splits and adds corners but moves none, a cut
        crosses triangles), the cloud turned with the laying and lowered with the trim."""
        from scipy.spatial import cKDTree

        nearest = cKDTree(made_corners).query(lying.to_made(self.corners + self.lowered, self.laid))[1]
        colours = np.asarray(painted["colours"], dtype=np.float64)[nearest]
        cloud = (lying.from_made(painted["cloud_points"].astype(np.float64), self.laid) - self.lowered,
                 lying.from_made(painted["cloud_normals"].astype(np.float64), self.laid, directions=True),
                 painted["cloud_colours"])
        return colours, cloud


def judged(subject, picture_mask, corners, triangles, picture=None):
    """A shape (at its size, standing on y = 0, as the shape model made it) judged as the finish judges it, in one
    place for the shape stage's tries, the finish and the regression set, so they cannot drift apart (a shape stage
    that judged its tries without the levelling would let a seed with a leg 2 cm short lose to one failing another
    check): cleaned up (`mesh_repair`), a web in a hole the picture shows carved away (`carving`), a
    thing that lies laid on its broad face (`lying`), a thing that stands levelled when a leg is a little short
    (`levelling`), then every shape check. Returns a `Judged`."""
    from . import carving, levelling, mesh_repair

    corners, triangles, repair = mesh_repair.repaired(corners, triangles)
    corners, triangles, carved = carving.carved(picture_mask, corners, triangles, picture)
    corners, triangles, laid = lying.laid(subject, corners, triangles)
    level = None
    if subject.stands:
        corners, triangles, level = levelling.levelled(corners, triangles)
    checks, measures, overlay = shape_checks(subject, picture_mask, corners, triangles, laid, picture)
    measures["repair"] = repair
    measures["carved"] = carved
    if carved.get("holes"):
        checks.append(carving.carve_check(carved))
    if laid["laid"]:
        measures["laid"] = laid
        checks.append(lying.lay_check(laid))
    if level:
        measures["levelling"] = level
        checks.append(levelling.trim_check(level))
    return Judged(corners, triangles, checks, measures, overlay, laid, level)


def coverage_check(coverage, all_texels=None):
    """`coverage`: the painted share of the surface as it is seen (each texel counted by how often it is in view);
    `all_texels`: the painted share counting every texel alike, shown beside it."""
    beside = f" ({all_texels:.1%} of all its texels, the seldom seen counted alike)" if all_texels is not None else ""
    return Check("paint covers the surface", coverage * 100, LEAST_COVERAGE * 100, "%", coverage >= LEAST_COVERAGE,
                 f"the painted views coloured {coverage:.1%} of the surface as it is seen{beside}; the rest took "
                 "colour spread from its neighbours")


# A painted view is black where its colour's brightest channel is under this (10 of 255); a view more than this share
# black came back broken (a half-precision decode that overflowed paints black). On the trials (654 views) the most
# black was 1.7 %, a view seeing a thin thing's end on its grey backdrop (the coverage alone counts a black view as
# painted).
BLACK_LEVEL = 0.04
MOST_BLACK_VIEW_SHARE = 0.25


def black_share(view):
    """The share of a painted view's pixels (H x W x 3, 0-1) that are black."""
    return float((np.asarray(view).max(-1) < BLACK_LEVEL).mean())


def views_check(shares):
    """The six painted views, each as it came back: none mostly black. `shares`: each view's `black_share`."""
    worst = max(shares)
    return Check("painted views not black", worst * 100, MOST_BLACK_VIEW_SHARE * 100, "%",
                 worst <= MOST_BLACK_VIEW_SHARE,
                 f"the blackest of the {len(shares)} painted views is {worst:.1%} black (each "
                 + ", ".join(f"{s:.1%}" for s in shares) + f"; at most {MOST_BLACK_VIEW_SHARE:.0%})")


def fill_check(fit):
    """`fit`: the touch-up's fill as the finished model shows it (`pipeline.judged_fill`). The coverage counts a texel
    painted; this judges what it shows. The fill kept on the model fails when its colours are found nowhere in the
    paint round it (a hole painted into a dish, the backdrop's grey carried in); a place left unpainted shows the
    colour spread to it from round it, and fails when it is large and neither the paint round it nor what it shows is
    one colour (a smear).
    The fill as the inpainter painted it, before what was foreign was left out, never reaches the model: it is shown,
    not counted."""
    from .touch_up import LARGEST_FOREIGN_PATCH, MOST_FOREIGN

    shares = ", ".join(f"{share:.1%}" for share in fit["views"])
    worst, patch = fit["worst"], fit.get("largest_patch", 0.0)
    left = fit.get("left_out") or {"largest_place": 0.0, "largest_smear": 0.0}
    smear = left["largest_smear"]
    passed = worst <= MOST_FOREIGN and patch <= LARGEST_FOREIGN_PATCH and smear == 0.0
    painted = fit.get("as_painted")
    shown = (f"; as the inpainter painted it, before the touch-up left out what was foreign (shown, not counted), "
             + ", ".join(f"{share:.1%}" for share in painted["views"]) + " of each view's fill was foreign"
             if painted else "")
    smears = (f"a smear {smear:.1%} of the object's size across: neither the paint round it nor the colour spread "
              f"into it is one colour" if smear else
              "no smear: each larger place is ringed by one colour, or shows one"
              if left["largest_place"] > LARGEST_FOREIGN_PATCH else "none larger than the limit")
    return Check("fill matches the paint round it", worst * 100, MOST_FOREIGN * 100, "%", passed,
                 f"of the touch-up views' fill kept on the model, {worst:.1%} of the worst view's has a colour found "
                 f"nowhere in the paint round it (delta E over {fit['delta_e']:.0f}; each view {shares}; at most "
                 f"{MOST_FOREIGN:.0%}); its largest foreign place is {patch:.1%} of the object's size across (at most "
                 f"{LARGEST_FOREIGN_PATCH:.0%}); {fit['filled_points']} filled points kept. Places left unpainted, "
                 f"the largest {left['largest_place']:.1%} across: {smears}{shown}")


def score(checks, measures):
    """How good a shape try is: the checks it passes first, then whether it was left uncarved (a carved web leaves a flat
    wall along the lines of sight where the picture's hole was: an uncarved try passing as many is better), then its
    outline's overlap."""
    counted = [c for c in checks if c.counted]
    return (sum(c.passed for c in counted), 0 if measures.get("carved", {}).get("carved") else 1,
            measures.get("outline", {}).get("iou", 0.0))


def marks_of(measures):
    """The places the export marks: {name: (position, up, facing)}."""
    weapon = measures.get("weapon")
    if not weapon:
        return {}
    up = np.array(weapon.get("toward_tip", [0.0, 1.0, 0.0]))
    facing = np.array(weapon.get("flat_normal", [0.0, 0.0, 1.0]))
    marks = {}
    grip = weapon.get("grip")
    if grip:
        marks["grip"] = (grip["hold"], up, facing)
        marks["grip_start"] = (grip["from"], up, facing)
        marks["grip_end"] = (grip["to"], up, facing)
    if weapon.get("tip"):
        marks["tip"] = (weapon["tip"], up, facing)
    if weapon.get("front"):
        marks["front"] = (weapon["front"], up, facing)
    return marks
