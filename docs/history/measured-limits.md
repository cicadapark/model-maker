# Measured limits

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

What the reference's sections state as a rule, and what each was measured on.

## The steps

The clean-up before a shape is checked (corners joined, doubled triangles dropped, small holes filled). On
the first trial every hole was a seam of unjoined corners and every torn edge a doubled triangle
left by the cut to the triangle budget; the one hole left after joining was 8 edges and 2.5 % of
the barrel's size round.

The view painter's steps: until 0.3.3 an option of `make` and `batch`, `--paint-steps` (the view painter's steps, 20 by default: on the second trial 20 painted as
well as 50 by coverage and likeness in under half the time, about 4 minutes, not 8 to 9). The option and six other
trial settings were taken off the commands after 0.3.3: they were not in the job folder's name, so a second run with
another value found its stages done and returned the first object.

## The checks

Shown, not counted: the shadow on the backdrop (how much darker it is near the object than far from
it) and a floor line on it. The picture model puts a floor shadow under nearly everything; the second
trial (212 pictures, 156 with such a shadow) found none inside a cut-out by eye, and the models made
from shadowed pictures had no slab and no dark foot. What would harm the model is a shadow inside
the cut-out, and that is what is checked. The measure cannot tell an object as pale and grey as the
backdrop from a shadow (2 of the 212, both a white table top): those fail safe, and another try is
painted.

## Open underneath

The limit was measured on 309 stored objects made on both cards (the objects of two development lists, the things of
today, and chests, boxes and bins made for other trials), every one of them measured, whatever its kind. Two came over
0.2. A wooden jewellery chest made open underneath (0.69 and 0.78 on the two cards) is caught: it is the case the check
was written for. Its stored shape is refused; made again, the chest gets a second shape try, which was never made or
judged. A wooden toy horse on wheels, which is right, measured about 0.26 on each card, under the limit (and
it is not of a kind that stands, so the check does not run for it). No object judged right came over 0.4.

## Without a card

`--recut`: On the third trial's chair our CPU cut and
the card's agreed 99.9 %; the same chair's cut-out forged (the gap under its seat filled, 58 % more cut-out) agreed
63.3 % and was refused.
