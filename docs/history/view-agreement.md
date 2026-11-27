# The painted views judged against each other

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## The painted views judged against each other, and one view per place (11 November 2026)

In 0.2.0's sealed run a white ceramic teapot came out with its right side black and a pale glass bottle dark on one
side, and every check passed: the view painter had painted one of its six views out of family with the others, and
the bake took it (docs/trials/2026-11-11-030-diagnosis.md). The coverage counts a texel painted whatever its colour,
"painted views not black" looks for a view that came back wholly black, and the likeness sees the object only from
the picture's side.

**The views judged.** Neighbouring side views see a strip of the same surface, round 45 degrees. Where both painted a
place (points within 2.5 of the cloud's spacing, facing within about 25 degrees), their colours are compared; a pair
is unlike where they differ by over 30 delta E, and a side view's disagreement is the lesser of its two pairs' shares
(an odd view disagrees with both its neighbours; its neighbours each agree with their other side). This is the
photo-consistency idea of texturing from photographs (Waechter, Moehrle and Goesele 2014 drop a view whose colour is
an outlier among the views of a face), between the view painter's views (`view_agreement.measured`).

**Painted again.** A view unlike both neighbours over a share of 0.5 has the six views painted again from the next seed,
at most twice, and the try that agrees best is kept (as the touch-up keeps its best fill). The kept try's views are in
`views/`, every try's in `views/seed-N/`, and each try's measure is in the record (`paint.six_views`) and shown in the
report as "painted views agree" (shown, not counted: polished metal's views differ by their reflections whatever the
seed). The kept seed is the sidecar's paint seed. The limit was set on the paint of the 167 trial objects kept: the four
single-view faults scored 0.52 to 0.91; 17 objects reach 0.5, the other 13 polished metal or iron.

**One view per place.** Where two views painted a place, the bake mixed its nearest painted points from both, and two
unlike views interleaved into a fine stipple. A view's painted point is now left out of the bake where another view
painted the same place more squarely by over 0.05 (on the 45-degree seam both stay): one view per place, as texturing
from photographs chooses a view per face (Lempitsky and Ivanov 2007) and as MV-Adapter's own projection weights views by
the cosine to the surface. A place only one view painted keeps its paint, however slanted: a first version compared each
point with all six directions, so a view blocked from a place by the object itself still pushed out the only paint
there: the coverage fell by up to 3 points (a candle in its dish 85.7 % to 82.8 %), and two objects fell under the 90 %
limit (a chair 90.68 % to 89.97 %, a stone well 90.10 % to 89.64 %); comparing only with what another view did paint,
the coverage of the 167 objects changes by 0.04 points down at most (0.45 up), and none crosses the 90 % limit
(`tools/trial/view_selection.py --coverage`).

**End to end** (`2026-11-11-030-views-e2e`, the 16 GB card, the sealed run's own seeds for picture and shape):

| Object | Views from seed 0 | Kept | 0.2.0 | Now |
|---|---|---|---|---|
| ceramic teapot | 0.91 unlike | seed 1, 0.02 | passed, wrong: a black side | white all round; fails the fill check (below) |
| glass bottle | 0.81 | seed 1, 0.02 | passed, flawed: dark on one side | pale all round, passes |
| copper kettle | 0.77 | seed 1, 0.47 | passed | passes; copper, its front still with silvery reflections |
| wooden armchair | 0.01 | seed 0 (painted once) | passed | the same, passes |

The coverage is as before to within 0.03 points (the teapot 97.82 % to 97.80 %). The paint stage took 57 s to 120 s
longer for the three painted again (the view painter's six views, then the touch-up). The teapot now fails "fill
matches the paint round it": with the new views the touch-up left a place unpainted along the handle's foot on the lid,
4.0 % of the object across (the limit 3 %), whose ring is black handle and white lid, so it counts as a smear. By eye
it is a thin line, not a fault: a false refusal of a kind a later version's work takes up, not this one's.

**Polished metal, one limit.** 13 of the 17 trial objects over the limit are metal or iron, whose views differ by their
reflections. Is a limit per material needed? The six views of six of them were painted again from seeds 1 and 2
(`2026-11-11-030-reseed-metal` and `-reseed`, the views alone):

| Object | Seed 0 | Seed 1 | Seed 2 | Kept by the pipeline |
|---|---|---|---|---|
| copper kettle | 0.77 | 0.47 | 0.51 | seed 1, 0.47 (two paints) |
| steel longsword | 0.66 | 0.11 | 0.64 | seed 1, 0.11 (two paints) |
| round shield, iron boss | 0.67 | 0.48 | 0.54 | seed 1, 0.48 (two paints) |
| pickaxe | 0.59 | 0.68 | 0.18 | seed 2, 0.18 (three paints) |
| spear, leaf-shaped head | 0.73 | 0.71 | 0.79 | seed 1, 0.71 (three paints) |
| copper cooking pot | 0.82 | 0.79 | 0.77 | seed 2, 0.77 (three paints) |

Four of the six come under the limit with a repaint; two do not, and cost two paints for little. A limit per material
would have to find the material in the words, and the spear's prompt names none, while the kettle, the sword and the
pickaxe gain from the repaint. One limit is kept; the cost is time on about one object in ten.

All 364 Python tests pass and the regression set holds whole (53 of 53); its cases were made before the views were
recorded, so they are baked as before.
