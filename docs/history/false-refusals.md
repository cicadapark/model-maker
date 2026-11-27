# False refusals

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## False refusals (11 November 2026)

0.2.0's sealed run failed four objects that look right by eye: the wooden bed frame on both cards ("outline matches
the picture" and "holes the picture shows are open"), the iron cauldron on the 16 GB card and the writing desk on the
12 GB card ("fill matches the paint round it"; the desk also "left-right symmetry"); and the teapot painted again (the
repaint in [view-agreement.md](view-agreement.md)) fails the fill on a place 4.0 % of its size across (limit 3 %). Each was traced in its stored data before anything
was changed, and a change was kept only where it measures better on every stored object.

### The bed frame: the view search never drew it from its own tilt (fixed)

The picture shows the bed from its foot end, a few degrees above. The outline check judged it from 79° round and 14°
above, from its side (edges 9.9 % and 25.5 % of the height apart), and from there the gaps between its slats are open
and the picture covers them: so it also failed the holes. Drawn from its end, 6° above, through a lens 4 sizes away,
it lies 1.3 % from its picture with no hole of its own; the search never drew that view.

The search draws a coarse grid (13 turns, the tilts 0, 12 and 24°, 4 camera distances) and refines the best of it. A
bed is 2 m deep and 0.39 m high: seen from a tilt t its outline is h cos t + d sin t high, and grows by its depth times
the tilt's change, so its look changes fastest with the tilt of anything the trials hold. From its end its edges lie
within 10 % of its picture only from 3 to 9° (its outline 17 % lower or higher than at 6°); at 0 and 12° they lie 14 %
and 15 % away (35 % lower, 34 % higher), and a wrong view from its side (about 11 %) won the coarse grid. The slats were not
the cause: from the end at 12° the model's gaps are 1.5 of the picture's pixels wide, and the outline is as far off with
every hole filled (14.0 %).

A coarse-to-fine search finds a view only when its coarse steps are finer than the measure's capture range round it
(Borgefors, 1988, hierarchical chamfer matching). The tilts now follow the thing's proportions
(`object_checks.outline_tilts`): from level to 24°, in whole degrees, never more than 12° apart, and close enough that
its outline's height grows by at most a quarter between neighbours (d taken as its largest size across, the deepest it
can be seen). A thing as deep as high or less keeps 0, 12 and 24°; a bench 3.3 times as long as high is tried at 0, 4, 9,
15, 23 and 24°, the bed at 0, 2, 5, 9, 14, 20 and 24°: about twice as many coarse views for such things; the rest
search as before.

Both beds are now judged from their end, 0° round and 6° above, through a lens 4 sizes away, with no hole of their
own: the holes pass. The outline still fails on both, now by a little and truly. The 16 GB card's lies 1.32 % and
4.43 % from its picture (the thickest piece in a gap 5.2 %): its rail a little shallower than the picture's, its legs a
little thicker, its far corners a little wider. The 12 GB card's (stored only as its finished model) lies
2.24 % and 3.91 %: its legs stand a little inside the picture's, and two short centre feet hang under its rail where
the picture has none. No limit was moved. The search's other views agree that the 16 GB card's bed sits at the limit:
a third of a degree changes its 95th percentile by a point (3.4 % at 6.3° through a lens 3 sizes away, 5.0 % at 6.6°).

What did not work: a finer last step of the tilt (the tilt that grows the outline by the mean's limit, a third of a
degree for the bed). It found a lower mean (1.23 %) but a higher 95th percentile (4.95 %): the search minimises the
mean, and finer steps only move the bed along the edge of the 95th percentile's limit. Not kept.

**Measured on every stored finished object** (`tools/trial/rejudge.py`: 217 jobs of the trials, the sealed run on both
cards and this version's trials among them; each judged by the code before and after, shape checks and fill, nothing written
into the jobs; 41 jobs stored only as their finished model are judged on it): two verdicts change, both beds' holes,
fail to pass; no verdict changes from pass to fail, and no other check changes. Seven jobs are judged from another view,
all deeper than high (the two beds; two birdbaths, two buckets and a cauldron, whose 95th percentiles move by 0.28 points up at
most and 0.62 down, every verdict the same). The regression set gains `bed-frame-from-its-end` (it must pass the
holes; its outline, at the limit, is held neither way), and a test draws a made-up bed frame from its end at 3, 6 and
9° (the search found it from 61° round before).

### The teapot's and the cauldron's fill: the places are smeared; the eye forgives them (not changed)

Every place over 3 % that a touch-up view left unpainted, on the 41 stored jobs that have one (176 places, copies of
a job in several trial folders counted each time; each place's pixels drawn from its touch-up view), was measured again
by three other readings of "smear", each an established one, against the places labelled by eye:

* **A seam between two colours.** A place on the line where two materials meet (a black handle on a white lid) shows
  the colours of both and their blends; the matting equation reads such a pixel as a mix of the two (Porter and Duff
  1984; the colour-line model of Levin, Lischinski and Weiss 2008). The ring was split into its two colours and the
  inside judged as their blends, placed by its distance to each side. It separates nothing: nearly every ring is two
  colours (at least 85 % of it within 20 of one of two medians, in 174 of the 176 places), and the labelled smears'
  insides lie on the line between them too (78 to 100 %; the teapot's 99 %), as a colour spread from a ring is a
  blend of it. Nor is the teapot's place a sharp edge: only 16 % of its pixels show the colour of the side they lie
  nearer, where an edge would show nearly all; the handle's black runs across the white lid and white flecks lie on
  the black.
* **CIEDE2000** (the CIE's difference of 2001, which weighs lightness less away from mid-grey, in darks and lights):
  within 15 of their medians the cauldron's larger place reads 78 % (ring) and 81 % (inside) one colour, the 16 GB
  card's bucket's, smeared by eye, 81 % and 78 %; within 10 or 12 they lie as close.
* **Chromaticity alone** (CIE 1976 u'v', blind to shading, as colour Retinex tells a change of material from shading;
  0.031, which is delta E 20 at mid-grey): the teapot and the cauldron pass, but so do the strength-0.99 candle's three
  smears (grey patches on its brass and wax: a smear of lightness alone), and that candle would pass the check: a new
  miss.

So the places are what the check says: the teapot's is the handle's black spread over the lid with white flecks on it,
on a strip 4.0 % across, and the cauldron's two (14.2 % and 13.0 %) are the grey the view from above paints on an inner
wall seen at a slant, as on the flower pot whose smear is labelled a fault. On the near-black iron and on the handle's
foot the eye does not see them at a turntable's size. No reading measures better, and no limit was moved. The
cauldron's places have another cause, not fixed yet (the slanted colour on inner walls). The 12 GB card's desk's
fill (a place 4.5 % across on a box on its top) could not be measured again: its stored copy holds no paint data.

### The desk's symmetry: its pen pot is part of it (not changed)

The 12 GB card's desk is mirror-symmetric to 0.97 % on average; its 98th percentile reads 8.6 % (limit 5 %). Every
point lying more than 5 % from its mirror image is above 76 % of its height, on the things on its top, most of them on
the pen pot: 2.9 % of its surface. They are not loose: the desk is one piece (100 % of its surface). A
symmetry judged without them would have to tell them from a part of the object. The made-up chair with one armrest,
which must fail, has 4.4 % of its surface unmatched, above its seat (57 to 71 % of its height); with a leg missing,
4.1 %. A pen pot fused to a desk's top and an armrest on one side are the same to the measure: a part on one side,
above the main surface, a few per cent of the surface. The check is right that the desk is not symmetric; whether
things standing on a desk belong to "a small writing desk" is the picture's question, not the shape's. Not changed.

All 382 Python tests pass (5 skipped), and the regression set holds whole: 54 of 54, the 53 cases before and
`bed-frame-from-its-end`.
