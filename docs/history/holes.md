# Holes: the jug's handle, and holes the picture shows

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## The jug's handle: a web where the picture shows a hole (10 November 2026)

"a clay jug" failed "outline matches the picture" in every trial since the outline's gap piece was measured, on both
cards and with both shape seeds: the shape model closed the loop of the handle with a web where the picture shows the
backdrop through it. The thickest piece of the model's outline in the picture's gap was 15.1 % of the height on the
fix-check's and the third trial's jug (seed 42), 13.1 % on the fifth trial's (seed 43; seed 42 15.1 %), 13.5 % on the
12 GB card's fourth trial (seed 43); the limit is 8 %. Everything else of the jug matched (edges 0.64 to 0.95 % apart on
average). A mug, a lantern's ring, a key's bow, a shield's grip seen from behind would meet the same fault.

### Why: the shape model was shown a grey surface in the loop

`cutout.for_shape`, the cut-out the shape model is given, filled every hole closed off inside the object. The picture
is laid on white where the cut-out is empty, so inside the filled loop the shape model saw the picture's own backdrop:
light grey, darkened by the jug's shadow (a mean of 214 and 213 of 255 in the loops of the two pictures the webbed jugs
were made from, one on each card), framed by the handle, on white all round. To a model that sees a picture of an
object on white, that is a grey surface, and it made one. The first trial's jug, made from another picture, whose
handle came out open, had a lighter backdrop in its loop (239). TripoSG's own preparation (its `image_process.py`)
keeps the cut-out's holes and puts the object on white; it drops only specks of the cut-out under 200 pixels.

The webs are thin: 0.6 to 1.1 cm deep along the picture's lines of sight, across a loop 4.7 cm wide.

### The fix, in two parts

* **The shape model is given the holes** (`cutout.for_shape`): only a pinhole of the cut-out (under 200 pixels of
  the 1024-pixel picture, TripoSG's own speck size) is filled; the loop of a handle shows white, as the rest of the
  backdrop does. Tried on the CPU (TripoSG in 32-bit floats, its 50 steps, seed 42, the fifth trial's picture; 85
  minutes a shape on 4 threads, so its shape is not the card's, but both runs differ only in what they were shown):
  given the loop filled, the shape model made the web again (the gap piece 12.2 %, the edges 0.74 % and 2.39 %;
  carved: 2.3 %); given the loop as the picture shows it, it made the handle open (0.22 %, 0.52 %, 0.7 %), every
  shape check passing, nothing to carve. How it does on the card, and on every other object with a hole (the chairs'
  backs, the lantern's ring), is for the card's next run to show.
* **A web the shape model still makes is carved away by the picture's silhouette** (`carving.py`, in the shape stage's
  judgement of each try and in the finish). The picture is the one view we have, and where it shows the backdrop
  through the object, nothing lies along those lines of sight: the object lies within its silhouette's cone from the
  camera, its visual hull (Martin and Aggarwal, 1983; Laurentini, 1994), and what lies outside it is carved away
  (space carving, Kutulakos and Seitz, 2000). Only a web is carved, never the object's body:

  1. a hole of the picture's outline, closed round by the object, at least 7.8 % of the height wide (room for a piece
     thicker than an edge's misfit and the margin on both sides);
  2. the camera found as the outline check finds it (the same turns, tilts and lens distances), but with the
     outlines framed by their boxes and the holes' inner parts left out of the count: the web fills the shape's hole,
     and must not pull the camera to a view where it fits;
  3. the shape's piece in the hole thicker than 4 % of the height (the 95th percentile's limit on the edges: the most
     an edge may be out of its place; a chair's spindles a little off their picture's leave pieces up to 2.1 %);
  4. along every line of sight through the hole's inner part, the shape no deeper than the hole is wide: a web spans a
     hole and is thinner than it; the object's body behind the hole would be deeper (a test mug whose loop is filled as
     deep as its body, 20 cm in a loop 5 cm wide, is left as made);
  5. the cut keeps 1.9 % of the height (the edges' mean limit) inside the hole's edge, so an edge off by what the check
     allows is never cut; the triangles near the cut's line are split until none is longer than 3 canvas pixels (the
     simplifier leaves large flat triangles on a web, and one cut straight across cuts a chord of a curved hole), then
     the surface is cut along the line where it crosses them (as the levelling cuts along a plane);
  6. each opening the cut leaves is closed by a wall along the lines of sight: the web's front and back edges joined by
     a strip of triangles, its steps chosen so its new edges are shortest in all (as contours are joined into a surface,
     Fuchs, Kedem and Uselton, 1977), whether the web closed the loop whole (two openings, front and back) or ended
     inside it (one opening that runs along the front and back along the back); shreds of web the cut leaves hanging
     are dropped, a stray small opening on one face is filled as the repair fills a small hole;
  7. kept only if the surface is closed (no open or torn edge more than before), in no more pieces than before,
     encloses a volume, and, seen from the camera, nothing is left in the hole's core (twice the margin in): else that
     hole is left as made, and the sidecar says why.

  The shape the model made is kept unchanged in `shape.npz`; the sidecar's `carved` says what was found in each hole
  and what was cut, and the report shows it beside the checks ("webs in the picture's holes cut away", not counted).
  The painted colours follow as for the levelling: each corner takes the colour of the made corner nearest it.

On every jug the trials made, judged as the finish judges (the outline's edges: mean and 95th percentile; the thickest
piece in a gap; the shape closed and in one piece in every case):

| Jug | As made | Carved | What was cut |
|---|---|---|---|
| the fix-check's (also the third trial's and `jug-handle-web`), 16 GB card, seed 42 | 0.95 %, 3.12 %, 15.1 % (fails) | 0.53 %, 2.03 %, 3.3 % (passes) | a web 1.1 cm deep across the loop's lower part |
| the fifth trial's (`jug-handle-carved`), 16 GB card, seed 43 | 0.69 %, 2.08 %, 13.1 % (fails) | 0.37 %, 1.84 %, 2.6 % (passes) | a web 0.6 cm deep, ending inside the loop |
| the 12 GB card's fourth trial, seed 43 (from its `.glb`, no shape kept) | 0.64 %, 1.40 %, 13.5 % (fails) | 0.36 %, 1.67 %, 2.6 % (passes) | a web 1.3 cm deep |

Finished again on the CPU with their paint (`./model-maker finish` on copies), the fix-check's and the fifth trial's
jugs now pass every check: the paint covers 96.5 % of the surface as it is seen (97.0 % and 96.8 % with the web), the
likeness is as before (mean colour difference 6.0 and 6.5; 5.8 and 6.5). The cut adds about a thousand triangles to a
jug (61 100 to 61 600) where the line was refined.

Of the 67 objects with a kept shape (the regression set, the fix-check, the third and fifth trials, the three fails,
the touch-up), judged before and after: the four jugs above changed (outline fails to passes), and the first trial's
round table (`table-web`), whose picture shows the backdrop through two gaps between its legs and stretchers where the
shape model put webs 0.5 and 0.6 cm thick, is carved too and now matches its picture (1.10 %, 4.33 %, 11.5 % as made;
0.93 %, 2.60 %, 3.7 % carved; 74 300 triangles, the long cut lines refined). Every other object kept every verdict and
every value: the chairs' backs, whose holes the shape keeps open (pieces of at most 2.1 % along the spindles), the
lanterns (the ring's hole open in the shape; the gaps over the base 4.7 % wide, too narrow to hold a web), the
stools, tables, barrels, chests, books, candles, shield, swords, spears and axes (no hole wide enough, or none). The regression set holds 23 of 23:
`jug-handle-web` and `table-web` must fail the outline as made and pass it carved, `jug-handle-carved` is new, and the
chairs, stool, tables, lantern, sword and chest must have nothing carved.

What the carving leaves: a rim of web 1.9 % of the height wide along the loop's inside (the margin), which keeps the colour the
view painter gave the web; on these jugs the painter painted the web nearly black (it read as the loop's shadow), so a
dark sliver shows on the handle's inner edge. A carving sees only what the picture's view sees through: a web seen
edge-on from the picture's camera (the table's webs between its far legs) is not in any hole of the picture's outline,
and stays. With the shape model shown the holes, the web should not be made at all; the carving is the net under it.

### What did not work

* Cutting away whole triangles whose corners all lay in the hole: the simplifier leaves a web's flat face in a few
  large triangles, so the cut stopped far from the hole's edge on one face and close to it on the other.
* Cutting along the line without splitting the triangles near it first: on the fifth trial's jug the front face's cut
  ran as a straight chord across the curved loop and the back face's along it, and the wall between them crossed the
  hole (it was refused: 399 pixels of the hole's core still covered).
* The outline check's own camera, and the camera found from the outlines with their holes closed: a web whole across a
  test mug's loop drew the first 4° above and through a lens (the cut then took the top of the handle's lower bar), and
  on the fifth trial's jug the second turned 11° off (the cut fell beside the loop). The outline check's canvas
  centres an outline on its area's middle, which the web moves (11 canvas pixels on the test mug); the carving frames
  both outlines by their boxes.
* Joining the front and back edges by their arc length: where the two faces' edges are cut into different numbers of
  pieces it twisted the wall; the shortest strip does not.

Not tried: more shape seeds when only the gap piece fails. Both seeds on both cards made the web (four jugs of four),
as the input made it likely; another seed costs a shape stage (about a minute and a half on the 16 GB card) for a fault
that is not the seed's. The shape model's own settings (its guidance and steps) were not changed: the cause was what
it was shown.

## Holes the picture shows (10 November 2026)

An early clay jug (made before the handle fix) was found with its handle's loop filled: it had passed every check. A filled hole thinner than the outline's gap piece (8 % of the height) or narrower than the carving's
least hole (7.8 %) was caught by nothing: a lantern's ring, a key's bow, the gaps between a chair's spindles.

**"holes the picture shows are open"** (counted). Each hole of the cut-out the shape model was given is judged when it
has an inner part (more than 1.9 % of the height, the outline's mean limit, in from its edge) of at least 50 canvas
pixels and shows the backdrop's colour: the share of that inner part the model's outline covers, seen from the camera
the carving finds (the outline check's single view search, the outlines framed by their boxes and compared by their
overlap with the holes' inner parts left out, so a web cannot pull the camera). The converse is judged too: each hole of
the model's own outline with such an inner part, and how much of it the picture covers (a hole the picture does not
have). A hole too small to judge is listed "not judged", never passed; when no hole was judged and the model has none
of its own, the check is shown, not counted. The sidecar's `measures.holes` lists every hole: its width, its inner
part, its colour against the backdrop's, judged or not and why, how much is covered; and the model's own.

Measured on every made job of the trials (on both cards) and the regression set, as the finish judges them:

| | Most covered | |
|---|---|---|
| models that pass every check | 6.4 % (the 12 GB card's chair, a gap of its back), 2.6 % (jugs), 1.7 % (chairs) | pass |
| the first trial's round table, carved | 8.3 % | passes |
| the early jug, its loop filled (as made, as `inspect --recut` judges a model made elsewhere) | 100 % | fails |
| made-up: a ring on a box filled by a web 1 cm thick (its hole 6.5 % of the height) | 100 % | fails |
| made-up: a chair's back with a panel between its spindles (gaps 5.6 cm, 5.9 %) | 100 % (each of five gaps) | fails |
| any model's own hole covered by its picture | 0 % | passes |

The limit, a third of the inner part, lies in that gap (8.3 % to 100 %), for both ways. The trials' lanterns show the
backdrop through a pane of glass between two bars, a hole 4.7 % wide whose inner part is 20 pixels, and the model's
glass covers it, rightly: under 50 pixels an inner part is not judged (its cover is as much an edge's misfit as a web;
the made-up ring's is 108). Its colour tells nothing there: through clear glass it is the backdrop's (2.2 delta E).

**A model's hole too small for the picture's cut-out (15 November 2026).** The cut-out fills every hole under 200
pixels of the 1024-pixel picture as a speck (`cutout.for_shape`), so a picture covering a smaller hole of the model says
nothing about it. A wooden bread board's hanging hole, about 10 pixels across in its picture, was refused on both cards
as "a hole the picture does not have" (100 % covered). A model's own hole whose area, taken back into the picture's
pixels, is under 200 is listed in `measures.holes.extra_holes_too_small` and said "not judged". Judged again on 14
stored runs (342 jobs) and the regression set (54 held), only the bread board's verdict changed.

**Before carving, the hole's colour.** A hole whose colour in the picture (the median over the hole, CIE76 delta E in Lab)
lies more than 10 from the backdrop's just outside the object (12 to 48 pixels out) is a part the cut-out missed (a
mirror, a framed canvas), not a hole: it is never carved, and never judged open. On the trials every hole read at most
2.2 (most under 1; the jugs' loops 0.4, though the shadow darkens them, as the backdrop next to the object is darkened
alike); the made-up mirror whose glass the cut-out missed reads 63 and is left whole (before, its glass was carved
away: 47,680 triangles).

**A carved try ranks below an uncarved one** passing as many checks: the carving leaves a flat wall along the lines of
sight where the web was, and a seed that made the hole itself is better.

The regression set keeps `example-jug-handle-filled` (built from that jug, kept in the development history and not in
this repository: its picture cut again by our BiRefNet on the CPU, its `model.glb` read as the shape), which must fail
the check as made; `made-up-ring-filled` and `made-up-chair-back-panel`, which must
fail it, and their open twins, which must pass it; and `made-up-panel-miscut`, whose panel must not be carved.

What it costs: when the picture has a hole to judge, one more view search per judgement (the shape stage's tries and
the finish), about as long as the outline's; without one, the outline check's own view is read again.
