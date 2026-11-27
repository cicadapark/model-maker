# Held things: hafts, grips, a shield's handles, a forked haft

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## Held things: the haft followed, the grip on the haft, a shield's handles (10 November 2026)

Eight objects made again showed three faults in the checks and marks of things held in the hand.

**The axe failed falsely.** "a double-bitted battle axe" failed "haft straight" (9.37 % against 1 %) and "no kink"
(28.4° against 4°), but its haft is straight within 4 mm. Each slice across the length was measured on its largest
piece; in one slice just under the head the largest piece was the lower corner of one bit (its middle 80 mm off the
haft, 44.8 mm wide, narrow enough to pass for haft). Now the haft is followed from slice to slice: from the middle of
the length outward, each slice is measured on the piece nearest where the line of the narrow slices before it points
(a slice with one piece is that piece, however far it lies, so a bent haft is followed round its bend and still
fails). A haft or a shaft is then its longest unbroken stretch: it stops where the head begins (the spike above the
axe's head is not haft). The axe now reads 0.5 mm (0.05 %) and 1.6°.

**The axe's grip mark sat on its butt.** The hand's middle was 14 cm from the end, on the neck of its black butt
piece (a flared knob, a neck 2.8 cm round under the haft's 3.3 cm, a collar), thin and round enough to hold. Now a
haft's grip begins, from the butt, at the first stretch a hand wide whose girth stays within 10 % of the haft's own;
the hand's middle is 8 cm in from there, on the wood: 26 cm from the end. The grip marks also lie on the grip's own
middle where they are (its slices' middles within 3.5 cm), not on the run's mean middle: a frame tilted by a
one-sided head put a made-up bearded axe's hold 3 cm off its haft, and the war hammer's grip line ran off its haft.

**The shield's grip mark was between its handles.** The grip was on the middle of the back by convention; the remade
shield has two D-handles, left and right of its middle, so the grip lay on bare boards. Now the handles are found: the
board's back is a smooth surface (quadratic) fitted to the backmost point of each cell (1/40 of the shield's size),
what stands more than 1 cm off it left out of the fit, four times; what stands more than 1.2 cm off it is grouped into
pieces, and a piece is a handle when it stands at least 3 cm off, its bar (what stands at least 70 % as far off as its
furthest) is at least 7 cm long and 1.5 to 5.5 cm round across its middle third, with at least 1.5 cm of open space
under that middle third for the fingers. With two handles the hand holds the one on the right as seen from behind
(-X), the forearm through the other: the convention of a shield carried on the left arm, the forearm across the back;
of two handles one above the other, the upper. "handle found" (shown, not counted) names the handle held; with none,
it fails, saying so, and the grip stays on the back's middle by convention.

### What did not work

* Taking the haft proper as its longest even stretch (girth within 10 % of its own middle) cut the remade spear's
  shaft at a band 13 % thicker, 61 cm from its butt: its grip shrank from 157 to 53 cm and its hold left 40 % of its
  length. The butt end alone is trimmed now; a band or a binding further up is held over.
* A straight line fitted through the grip's slices placed its marks on the haft's line, but a bow's grip curves: the
  first trial's bow (which fails anyway) had its grip 9 cm off its stave. The marks take the slices' middles near them.

### The trials finished again

Every held thing of the trials with a kept shape (31 jobs: the first trial and its refinish, the fix-check, the
third and fifth trials, the remake; the 12 GB card's trials keep no shape and their bows were not made) finished on the CPU
with the code before and after, and every verdict and mark compared:

| Model (jobs) | Verdict | Marks, judged by eye |
|---|---|---|
| remade axe (1) | fails "haft straight", "no kink" → passes every check | grip from the black butt piece's neck (14 cm from the end) to the wood above its collar (26 cm) |
| the trials' axe (4) | passes, as before | the same: its butt piece has a collar too; 14 → 26 cm |
| war hammer (4) | passes, as before | the hold 4.7 cm further up (12 → 16 cm), still on the wood just above its cap; its grip line now on the haft, before it ran off it |
| shields (6) | passes, as before; "handle found" new | grip from the boards between two handles to the right-hand one seen from behind: 13.5 cm (the trials' shield), 15.5 cm (the remade) |
| broom (4) | passes, as before | 2 to 4 mm across, onto the handle's middle |
| first trial's bow (2) | fails, as before | the hold on the stave where it was 9 cm off it |
| swords, spears (10) | as before | as before (the remade spear's grip 156.6 → 154.9 cm long, its hold where it was) |

The 12 GB card's trials' shield (not re-finishable) has three pieces on its back: a long bar in the middle, which the hand
holds, a handle to its left seen from behind (the forearm's), and a stub 6 cm long, too short for a hand.

The regression set keeps the remade axe (`axe-head-corner-beside-haft`: straight, no kink, its grip 20 to 32 cm from
the butt) and shield (`shield-two-handles`: two handles found, the grip on one), and made-up cases: a haft bent 6° at
its middle (it must fail straight and kink), a bearded axe (its beard a larger piece than the haft beside it: it must
pass), an axe with a butt cap like the remade one's (its grip 19 to 26 cm from the butt) and a shield with two handles.
With the code before, the remade axe, the bearded axe, the butt cap and both shields break; the bent haft fails both.

## A haft that forks, and its socket (14 November 2026)

An iron shovel was refused for "haft straight" (4.2 %) and "no kink" (38.6°) though its haft is straight: the haft
was followed up one leg of its D-handle, and the blade and the handle, more than half its length, set the width the
haft was told from. A haft now ends where its line forks round the point it was heading for (the slice is measured
whole), a haft's usual width is its lower quartile's, and a socket or a collar tapering into it (thicker than the
haft's middle girth by 10 %) is trimmed off its ends: the shovel passes (3.1°, 0.58 %). Rejudged on 53 stored handled
objects: only the shovel's verdicts change, and one knotted club's "haft straight" (its kink still refuses it); the
regression set holds whole. A woodcutter's axe whose haft is S-curved, as such axes are made, is still refused (1.8 %,
5.3°): a known false refusal, since loosening straightness would pass bent hafts.
