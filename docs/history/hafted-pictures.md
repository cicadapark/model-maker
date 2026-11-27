# Hafted weapons: maces, clubs, and painted upright

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## Maces, clubs and things the picture model knows only loosely (11 November 2026)

In 0.2.0's sealed run the short iron mace was made as a long rod with a small knob, a sceptre, and passed "shows the
thing asked for"; the club found no picture; the spinning wheel came out a wheel on a stand and the water wheel a
ship's helm. The picture model knows these names only loosely, and the words said nothing more. As for the war
hammer, they now describe what shows, after the user's words: a mace's heavy head ringed with flanges or spikes on a
short handle, a club thick and heavy at its top (`HEAD_PICTURE_WORDS`), a spinning wheel's bench, treadle and spindle,
a water wheel's ring of paddles (`THING_PICTURE_WORDS`, for a thing with no kind of its own, when its name is in the
prompt). Measured blind on eight tries a prompt, the four sealed prompts and seven others: maces 26 of 32 tries (0
with the old words), clubs 13 of 24 (0), spinning wheels 16 of 16 (0), paddled water wheels 11 of 16 (4). The judge
was not changed: a sceptre among its kinds refused every quarterstaff, a club of its own passed sticks, and "short"
as a quality refuses short swords. The measurements, the paint faults of the same run (a view painted out of family,
lighting painted into a vessel's inside) and a check tried and rejected are in
`docs/trials/2026-11-11-030-diagnosis.md`.

**End to end** (`2026-11-11-030-words-e2e`, the 16 GB card, nine prompts none of them sealed): an iron, a flanged, a
spiked and a bronze mace all came out maces and passed every check; the old spinning wheel is a spinning wheel (it
fails "holes the picture shows are open" at 33.3 % against 33 %); the mill water wheel came out a spoked wheel with no
paddles (it fails the paint's coverage); the three clubs are thick sticks with a knob at most (two pass every check,
the knotted oak club fails "haft straight" and "no kink": it is knotted). Two of the four maces' chosen pictures were
tilted, and their models lean about 40 degrees in their own frame: the shape checks measure a hafted thing along its
own long axis, so they pass, and the grip and tip marks follow the haft, but the object does not stand along the up
axis. A tilted try can be told from the cut-out alone: its long axis (the mask's second moments) leans 18 to 86 degrees
in 13 of the 14 tilted maces labelled, against 0 to 14 degrees in the 12 upright ones and at most 3 in the 59 rods and
sticks (the fourteenth tilted one is painted head down, upright). The weapon is now turned so its haft stands (below).

## A hafted weapon stood upright, head up (11 November 2026)

A mace, a club or an axe painted lying, tilted or head down is made so: the shape model keeps the picture's frame.
Refusing tilted tries would throw away 14 of the 26 maces the words trial painted, so the weapon is turned instead
(`upright.py`), and the tilt only ranks the tries.

* **The turn.** The haft is found as the weapon checks find it (the slices along the weapon's long axis, followed along
  its line, its longest narrow stretch) and its direction is the straight line through those slices' middles. The head
  is the end the picture shows at the top, as the words ask, unless the other end is at least 1.5 times as wide (the
  maces painted head down measured their top end 0.14 to 0.54 of their bottom's width, the plain sticks and clubs
  painted the right way up 0.88 and over). The least rotation that brings the haft onto the up axis, head up, is applied
  after the shape stage, as a thing that lies is laid (`lying.laid`, whose record it shares: the outline is still judged
  in the frame the shape was made in), and shown as "haft upright" (not counted). The view painter paints the weapon
  standing (its paint kept in the shape's own frame), and the grip and tip marks are measured on the turned weapon.
* **The ranking.** Each try of a hafted weapon is shown its lean: the long axis of its cut-out (its second moments)
  against the picture's vertical. Among the tries that pass, those leaning 16 degrees or less are chosen first (the
  middle of the measured gap between the upright maces, 0 to 14, and the tilted ones, 18 to 86). A tilted try is never
  refused.

**Measured** (`2026-11-11-030-upright`; `tools/trial/upright_measure.py`, the haft's lean found again by level slabs
of the turned shape, not by the frame the turn uses): every labelled mace and club try of the words trial made into a
shape (112, one shape each), the end-to-end maces and clubs, and every hafted object of the earlier trials.

| Shapes | Turned | The haft's lean afterwards | Head up |
|---|---|---|---|
| 14 maces painted tilted or head down | 36 to 179 degrees | 3.0 degrees at most | 14 of 14 |
| 12 maces painted upright | 13 degrees at most | 0.2 at most | 12 of 12 |
| 30 rods and 29 sticks (upright pictures) | 0.9 degrees at most | 0.3 at most | |
| the end-to-end flanged and bronze maces | 40 and 31 degrees | 0.2 and 0.1 | both |
| 23 hafted objects made end to end in the trials (axes, war hammers, a mallet, quarterstaffs, maces, clubs; those two among them) | 1.1 degrees at most but those two | 0.6 at most | |

The grip and the tip are the same points of the weapon before and after: the grip's distance from the butt and the
length are unchanged to the millimetre, and the marks of the turned weapon, turned back, lie on those of the weapon as
made (0.0 mm apart), in all 131 of the 135 where a grip is found. Two hooked sticks the words trial painted as clubs (labelled "other") are turned head
down, their hook being their wider end. The two tilted maces, painted and finished again standing, pass every check,
their paint's coverage as before (99.67 % and 99.35 % made tilted; 99.54 % and 99.29 % standing). All Python tests pass
but one that fails on objects-first as it is (the README's pictures list, `test_credentials`), and the regression set
holds whole (54 of 54, with the bed frame case added since).
