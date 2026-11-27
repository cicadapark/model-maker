# War hammers

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## War hammers (10 November 2026)

Every war hammer the trials chose had an axe-like head: a flared blade with a square poll and a spike (seed 141081,
chosen on both cards). It scored best on the other picture checks, and its models passed every other check. The subject
check refused it (0.49 and 0.58, the likeliest other "an axe"), but did not then take part in the choice of try.

### Why

A close look at the four designs shows the cause is mostly our wording:

* The view asked for "its point or head at the top", which invites a spike.
* Seen straight from the side, a hammer face is edge-on, and from there a face and an axe blade have the same outline.
* The concept itself is mixed: many war hammers in pictures have a blade or a beak behind the face.
* FLUX.1 [schnell] runs with no guidance, so nothing can be asked away with negative words.

### The fix

* **The check chooses.** "Shows the thing asked for" (`subject_judge`, at its limit of 0.75) is now one of the picture
  checks of every try. A try it refuses is not chosen, and when no try passes, another round is painted, as with the
  other picture checks. SigLIP 2 is loaded for every job's pictures. Without it, the tries go unjudged, the run says
  so, and a thing with a lid, cover or doors is still not made. The finish keeps the verdict from the record, and the
  sidecar lists SigLIP 2 among the models used.
* **The words.** A head on a haft (an axe, a hammer, a mace) is painted "standing upright, its head at the top, seen
  straight from the side". "point" is dropped; a blade and a spear keep it. The war hammer family (war hammer, maul,
  mallet, sledgehammer) is told what its head looks like, as a chest is told its lid is shut: "its head a heavy blunt
  block with a flat square striking face". The words name no material, so "a wooden mallet" stays wooden. "mallet" and
  "sledgehammer" are now known words (before, a mallet was an unknown object, painted lying on the floor from above).

### Measured

Each prompt was painted with 8 seeds on the 12 GB card, the pipeline's own seeds for its own words, pictures only
(`tools/trial/paint_tries.py`), in three wordings:

* "old": the words before the change;
* "new": the words above;
* "turned": the new words with the view turned a little toward the viewer, so the face shows.

The axe prompts were painted old and new. The first trial's war hammers and battle axes from the other card were added.
The 12 GB card's copies of the trial's tries are bit-identical to the "old" ones. Every head was labelled by eye before its
score or wording was seen: the top of its cut-out cropped, shuffled under a random code (`tools/trial/head_sheets.py`).
The labels are:

* hammer: blunt faces; a spike on top or a pointed beak is allowed;
* axe: a broad sharp blade;
* hybrid: a blade and a face, or flared ends between the two, or a hammer and a blade on one haft;
* other: neither.

Then the codes were unblinded. In eleven pictures the crop missed what mattered: a second blade or block at the foot
of the haft, or a mallet lying head down. Those labels were corrected from the whole picture, and the case set says so
for each (`tools/trial/head-cases.toml`, 152 pictures).

| Hammer prompts (war hammer, maul, mallet, sledgehammer) | Old | New | Turned | First trial, other card |
|---|---|---|---|---|
| clear hammers | 19 of 32 (59 %) | **30 of 32 (94 %)** | 31 of 32 (97 %) | 0 of 4 |
| of them, "a war hammer" | 1 of 8 | 7 of 8 | 8 of 8 | 0 of 4 |
| axe heads | 1 | 0 | 0 | 1 |
| hybrids, others | 9, 3 | 0, 2 | 0, 1 | 3, 0 |
| check at 0.75: passes, refusals | 27, 5 | 32, 0 | 31, 1 | 3, 1 |
| clear hammers passing | 19 of 19 | 30 of 30 | 31 of 31 | |
| axe heads refused | 1 of 1 (0.58) | | | 1 of 1 (0.49) |

| Axe prompts (battle axe, double-bitted battle axe, hatchet) | Old | New | First trial |
|---|---|---|---|
| axe heads, all passing | 18 (0.97 and over) | 20 (0.97 and over) | 3 |
| hybrids, hammers, others | 4, 0, 2 | 3, 1, 0 | 1, 0, 0 |
| check at 0.75: passes, refusals | 22, 2 | 23, 1 (the hammer head, 0.44) | 4, 0 |

Every rule holds:

* Every clear hammer passes: 80 of 80, lowest 0.935.
* Every axe head asked as a war hammer is refused: 2 of 2, at 0.49 and 0.58.
* With the new view words, every axe head asked as an axe still passes: 41 of 41, lowest 0.967.
* The trials' 239 clear pictures stay at 0.926 or over.

By prompt, with the new words: war hammer 7 of 8 clear hammers, maul 7, mallet 8, sledgehammer 8. With the old words
the numbers are 1, 2, 8 and 8.

With the check in place, the first round of four picks a clear hammer for all four prompts with the new words, and an
axe for all three axe prompts. Without the check, the new words' hatchet round would have chosen its hammer-headed try.
With the old words and the check, "a war hammer" gets a hybrid (two flared ends and a spike, 0.93) in place of the
axe head.

Shown, not held: two "other" pictures from the new words pass. One is a war hammer with two blocks on its haft
(1.00). The other is a maul try that is a pole with a brush-like end (0.86). The check judges what kind of thing a
picture shows, not whether it is well made.

### The second layer was not needed

A second layer was considered in case the subject check alone was too weak: "a hammer head with a flat blunt striking
face" against "an axe head with a sharp curved blade" (`tools/trial/judge_tries.py` scores it for every picture). The
subject check alone holds every rule above, so it is not added. The pair would also be a poor check:

* The clear hammers score from 0.21 up and the axe heads at most 0.05, so its limit would have to lie in that narrow
  gap.
* It would catch nothing the subject check misses: the two "other" pictures that pass score 1.00 on it.

No limit was moved.

### What did not work, and what was not kept

* **The three-quarter turn.** It made 31 clear hammers of 32 against 30, but one turned try put a second large block
  at the foot (refused, 0.63). It also changes the view the shape checks expect: a hafted thing's length along the up
  axis, and its width, seen from the side. It is not kept.
* **Labelling from the top of the cut-out alone.** It missed a second head at the foot in six pictures and two mallets
  lying head down. The labels were corrected from the whole picture, and the case set says so for each.
* **Reading seed 141080 by eye, unblinded, as "a real war hammer".** Labelled blind, both cards' copies are hybrids: two thin
  flared ends and a spike, no clear face. Its score (0.95, 0.97) passes either way.

The labelled heads are a regression case, "head-pictures" (`tools/trial/regression.py`). A clear head of the prompt's
own kind must pass, and a clear head of the other kind must fail. Hybrids and others are shown, not held. The old war
hammers in `prompt-cases.toml` keep their "unclear" label, now with their blind labels.
