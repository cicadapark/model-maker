# The fix-check's three faults

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## The fix-check's three faults (10 November 2026)

The ten regression prompts made again after the third trial's fixes passed 7 of 10. Each of the three left had a
different cause.

### The book stood on its edge: the picture's view, and our set-down

"a leather-bound book, closed" was painted shut, as asked, but from well above ("lying shut, only its cover showing"
draws the picture model nearly overhead). The shape model makes a thing as its picture shows it, in the picture's own
frame: its up is the picture's up. So it made a slab standing on its bottom edge, leaning back, its cover to the
viewer; the finish set it down with the shape model's up as its up, and it stood on an edge and settled 54.8° (base
level, centre over the base and the feet failed). The picture was not wrong (a book lying shut is seen from above),
nor the shape (it matches the picture closely, see below): what was wrong was taking the shape model's up as the
book's up.

A thing that lies has no up of its own: it rests on whichever broad face it falls on. The established way to find
how a rigid object comes to rest is its stable poses, each with its quasi-static probability (Wiegley, Rao and
Goldberg, 1992, "Computing a statistical distribution of stable poses of a polyhedron"; Kriegman, 1997; Moll and
Erdmann, 2002; Fu, Cohen-Or, Dror and Sheffer's upright orientation of man-made objects, 2008, starts from the same
candidates, the faces of the convex hull): set down slowly, the object rolls downhill on the height of its centre of
mass over the floor (its hull's support distance along the direction that points down) until it rests on a hull face
under its centre of mass; the share of all directions that roll to a pose is its probability. `lying.py` follows
6,000 directions (a Fibonacci lattice), each to its lowest neighbour until none is lower, joins resting directions
within 12° (a cover a little warped has several shallow minima), and lays the thing on the likeliest pose; of two
near-equal ones (a book's two covers) the face that looked toward the picture's viewer goes up.

Only a thing that lies is laid: `words.LYING` (a book, a coin, a plate, a tray), unless its words stand it up
("standing", "upright", "on its spine"); the report says which word decided. A chair, a table, a chest, a barrel stand
as the shape model made them, and one that tips fails its base checks; it is never laid on its back (a test holds a
leaning chair to that). The laying is done where the levelling is: the shape stage judges each shape try laid, the
finish lays the shape it exports; `shape.npz` keeps the shape as made, the paint's colours follow the turn, the
sidecar's `laid` says how far it turned and the poses found, and the report shows it beside the checks ("laid on its
broad face", not counted). The outline is judged in the frame the shape was made in, the picture's, so the views it
searches are those of any other object.

On the fix-check's book (`./model-maker finish`, the CPU): laid on its back cover (39 % of the ways it can fall; the
front cover 39 %, each edge 5 % or less), turned 56°; it settles 0.6°, its centre 57 % inside its base; closed and in
one piece; paint 97.2 %. It still fails "every foot on the ground": its back cover is not flat. The shape model made
the book in two lobes, after the bookmark's ribbon that splits the page block in the picture, and laid on that cover
it rests on a strip down its middle, the cover's edges 2 to 8 mm above the floor (16 low places 1.6 to 3.2 mm short;
the levelling, which may cut 5 % of the height, 3.7 mm on a book 7.5 cm thick, finds a low place 4 mm up and leaves it
as made). That is the shape's own fault, small, and it stays a
failure; the book is made again on the card.

The regression set keeps it as `book-made-on-its-edge`: as made it must fail "base level" and "centre over the base",
laid it must pass them, turned 50 to 62°.

### The chest's outline: the measure was wrong, twice

"a wooden treasure chest" was shut and looked right from every side, but failed "outline matches the picture" at the
95th percentile (4.95 % against 4 %; the mean 1.40 % passed). The outline picture showed where: the feet. The
picture's near front-right foot stood lower than its far back-right one, while the lid showed only as a sliver; in the
model's best view the feet were at its ends.

Two faults of the measure, not the shape:

* The outlines were compared on a canvas twice as wide as high, and an outline wider than that was squeezed to fit.
  The chest is 2.25 times as wide as high in its picture and 2.10 in the model: both were squeezed to 2, so their
  widths were never compared. Unsqueezed (a canvas six times as wide as high), the same view reads 3.65 % and 8.33 %;
  the view searched again, 2.17 % and 7.55 %. No other case in the regression set is that wide, so none other moved.
* The picture is a photograph, painted through a lens: near parts larger and lower than far ones. The measure drew the
  model straight on (orthographically). From any turn and tilt, straight on, the chest's feet cannot be where the
  picture has them while its lid shows that little. Drawn through a lens 6 times its size away (a field of view of
  about 20°), it matches: 0.68 % and 1.82 %, every foot in place. The book's outline was the same: 1.29 % and 5.47 %
  straight on (failed), 0.51 % and 1.30 % through a lens 4 sizes away; its picture, looking steeply down on a flat
  thing, is a trapezoid.

The established way to compare a model with its photograph is to fit the camera too (render-and-compare: the pose and
the focal length, as camera calibration from a silhouette does). The view search now tries the camera's distance with
the turn and tilt: straight on, and 6, 4 and 3 sizes away. That is more freedom, and it let the first trial's table
with a web between its legs through: its edge distances through a lens 6 sizes away were 1.10 % and 4.33 % (failing
by 0.33 % only), 4 sizes away 0.98 % and 2.66 % (passing). The web's edges lie near the legs' edges, so the edge
distances barely see it. So the check now also measures what they cannot: the thickest piece one outline has in a gap
of the other (within the other's convex hull: between a table's legs), at most 8 % of the height, twice the 95th
percentile's limit (the width such a piece, lying between two edges of the other outline, may have and still read
the 95th percentile's limit at its edges). The web is 11.5 % from every camera; a piece outside the other's hull (a
corner sticking out, the first trial's chest's at 8.1 % through a lens) reaches away from the other outline's edges,
and the edge distances judge it.

On every case (straight on and squeezed, before; now; mean, 95th percentile, thickest piece in a gap):

| Case | Before | Now | Verdict now |
|---|---|---|---|
| chair-good | 0.44 %, 1.07 % | 0.44 %, 1.07 %, 2.6 % (straight on) | passes, as it must |
| chair-short-back-legs | 0.45 %, 2.08 % | 0.52 %, 1.56 %, 2.1 % | passes |
| table-short-leg | 0.80 %, 1.84 % | 0.43 %, 1.09 %, 3.3 % | passes, as it must |
| table-good | 0.73 %, 2.08 % | 0.46 %, 1.30 %, 2.6 % | passes, as it must |
| table-web | 1.34 %, 5.47 % | 1.10 %, 4.33 %, 11.5 % | fails, as it must |
| sword-good | 0.09 %, 0.26 % | 0.09 %, 0.26 %, 1.0 % (straight on) | passes, as it must |
| book-shadow-left-out | 0.12 %, 0.26 % | 0.39 %, 1.30 %, 0.5 % | passes, as it must |
| barrel, stool, jug, lantern, candle | at most 0.49 %, 1.56 % | at most 0.51 %, 1.58 %, 5.2 % | pass |
| chest-shadow-left-out, chest-lid-open, book-painted-open | at most 1.28 %, 3.91 % | at most 0.61 %, 2.08 %, 1.6 % | pass |
| longbows | 0.16 %, 0.52 % and 0.23 %, 0.52 % | 0.19 %, 0.52 % and 0.25 %, 0.74 % | pass (not what fails them) |
| the fix-check's chest | 1.40 %, 4.95 % | 0.68 %, 1.82 %, 2.6 % | passes |
| the fix-check's book | 1.29 %, 5.47 % | 0.51 %, 1.30 %, 2.6 % | passes |

The likeness (not a check) is drawn from the same camera. The regression set keeps the chest as
`chest-through-a-lens`.

What did not work: the thickest piece of either outline alone, anywhere, without the hull. A good model's corner
sticking a little out of its picture's (the first trial's chest, 8.1 %) read like a web; edge distances already judge
those.

### The candle's paint: places seen from above at a slant, which none of the six views sees squarely

"a candle in a brass holder" passed every shape check, but the paint covered 85.8 % of the surface as it is seen
(limit 90 %). The coverage picture and a count by place and by the directions that see it (the check's own 32):

* what was missed is the dish's inner wall and the candle's lower part inside the dish (10.2 % of the surface as seen
  lies between a tenth and three tenths of the height, 3.7 % between three and five tenths), facing sideways (7.9 %) or
  up (6.3 %); the underside, painted by the view from below, is not among it (0.03 %);
* every missed place is seen from above: of the missed weight, 3.8 % of the whole comes from directions 5 to 30° above
  the horizon, 7.0 % from 30 to 60°, 3.4 % from 60 to 90°, and 0.01 % from below. The four side views look level, and
  the dish's rim hides its own inside and the candle's foot from them; the view from straight above sees the inner
  wall and the candle's sides edge-on.

So these places are not hidden: someone looking down on a candle on a table sees exactly them. The check is right
to count them.

The chair sits at the limit too: the fix-check's chair (made on the 16 GB card) reads 90.7 %, and the same prompt made
on the 12 GB card read 89.99 % and failed on that alone. Its misses are of another kind, but also geometric: 8.2 %
of its surface as seen is sideways-facing (the legs' and spindles' sides), hidden in the four level views behind the
legs in front of them (each level view looks straight along the chair's rows of legs); 5.4 % of the missed weight is
seen from below the horizon, 3.5 % from above. Counting only the directions above the horizon would read 92.5 % and
pass it: that would be the limit loosened by another name, so it is not done. The views chosen for it are the
diagonals the four level views lack: 45° and 315° round, 20° above (2.3 % and 1.8 % of the surface as seen). The
lantern (93.2 %, its cage bars' insides) gets two, 1.4 % and 1.0 %; every other object of the fix-check (jug, book,
table, shield, stool, barrel, chest: 0.8 to 4.9 % unpainted) gets none, as no view would add 1 %. A place no direction sees (the candle's foot where it meets the dish) already counts nothing. Counting
only the directions above the horizon, as for a thing that never leaves the floor, was measured: 74.8 %, stricter,
as it drops the underside, which the view from below paints. It is not done: a thing can be picked up or knocked
over, and the question of what can be seen in use is not the one this failure asked.

The established fix is views chosen by what is missed. MV-Adapter's own texturing (its paper, section 5.4) runs "view
coverage analysis to identify uncovered regions", renders the current texture from those views and inpaints them;
LumiTex ranks candidate views greedily by the uncovered area each sees; Hunyuan3D 2.0 paints from denser views (6 of
44 preset). MV-Adapter's image-and-geometry model was trained on the six fixed views only (four level, above and below),
so its views cannot simply be moved up. Measured on the candle's shape, what views could reach squarely (facing at
least 0.55, as the colour cloud keeps): the six, 84.9 % of the surface as seen; with four more at 20° above, 91.7 %;
at 30°, 94.0 %; at 45°, 96.6 %.

The paint stage now touches the paint up (`touch_up.py`, at most 4 views; `Settings(touch_up_views=0)` for none): candidate
views every 45° round, level and 20°, 35° and 50° above and below, are ranked by the unpainted surface each sees squarely, weighted as
the coverage check weights it; the best is taken, the object drawn from it in its colours so far, the unpainted pixels
in it (grown by 3 pixels so the fill meets the paint round it) painted by SDXL's base model through diffusers' SDXL
inpainting pipeline (the base weights, already in the registry; no new model), and only the unpainted pixels' new
colours put back into the colour cloud; then the next view, which sees the first's fill, until a view would add less
than 1 % of the surface as seen. A box the six views see whole needs none. What no view sees squarely is left to the
spread, and counts as not covered.

On the CPU, the views chosen for the fix-check's candle are 180° round and 50° above (4.3 % of the surface as seen),
0° and 50° (4.1 %), 45° and 50° (2.2 %), 225° and 35° (2.0 %). With a stand-in painter in place of the inpainting
(each unpainted pixel the colour of the nearest painted one: a test of the plumbing, not of the paint) the finish
reads 98.4 % (96.6 % of all texels). How well SDXL fills those places is for the card to show; the regression set
keeps the candle as the six views left it (`candle-dish-unpainted`, it must keep failing), so the check still catches
what the six views miss.

What did not work, or was not done: moving MV-Adapter's own cameras up (it was trained on the six level, top and bottom
views only; its paper's arbitrary-view model is another, separately trained model); LaMa, the inpainting model
MV-Adapter's texturing uses (trained on Places, whose terms are for non-commercial research); a dedicated SDXL
inpainting model (a separate download and registry entry; the base model is tried first).
