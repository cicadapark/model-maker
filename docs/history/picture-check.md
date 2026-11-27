# The picture against the prompt

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## The picture against the prompt (10 November 2026)

A model made on another computer comes with the picture it was made from. Before the model is shown, its picture is
checked to show the thing the prompt asks for (`model-maker picture-check`, `subject_judge.py`):
a chair for "an old looking chair", not a table, a vase or a person. A computer that swaps in another picture, or whose
picture model painted something else, is caught here.

### How it judges

This uses established zero-shot image-text classification (CLIP's, SigLIP's): the picture is compared with words for
the thing asked for and with words for every other thing it might show instead, and the thing asked for must be the
likeliest. The other things are every other kind of thing model-maker makes (`words.SUBJECT_KINDS`, 55 rows), each by a
description of its look with its name first ("a table, a flat top standing on legs"; classification by description,
as the state check's words are), and five distractors: a person, an animal, text, an empty backdrop, several things.
The thing asked for is the prompt itself ("a photo of an old looking chair") and its kind's description. Each class's
wordings are averaged into one direction (CLIP's prompt ensembling). A kind any word of the prompt names is never a
contrast ("a chest with a sword carved on it" is judged against neither), and near kinds a prompt may name by either
word share a row (a jug and a vase, a sword and a dagger), so a jug is never refused for looking like a vase. A
likeness of a creature ("a stone statue of a knight") is not judged against a person or an animal, and a prompt that
asks for letters not against text. The share is the sigmoid of the score difference between the thing asked for and
the likeliest other, averaged over the whole picture and its crop round the cut-out, as the state check's is. The
model is the state check's, SigLIP 2, on the CPU.

### Measured

Every distinct picture the trials painted (268, from 13 runs of 16 prompts; `tools/trial/prompt-cases.toml`): each
against its own prompt, and against every other prompt of the trials (the "shuffled" wrong prompts). The 16 longbows
are left out because bows are not made yet. Labelled by eye, 13 pictures do not clearly show their thing:
all 8 war hammers (no clear hammer face; labelled blind later, one is an axe head and three are hybrids, each copied on
two cards: see [War hammers](war-hammers.md)) and 5 tables painted from straight
above (a round top with at most its feet's tips showing, so a table and a stool look alike). They are shown, not held.

| | First try: a row for halberds | Now: halberds in the axe's row |
|---|---|---|
| own prompt, 239 clear pictures | lowest 0.05 (a war hammer, "a halberd") | **lowest 0.926** (a barrel seen from above, "a helmet") |
| another prompt, 3,346 pairs | highest 0.94 (a longbow asked as a spear) | **highest 0.500** (a three-legged stool asked as "a small round table") |
| an empty backdrop, each prompt | | highest 0.041 |

The limit is **0.75**, the state check's: the thing asked for at least three times as likely as the likeliest other.
In score terms it lies near the middle of the gap (logit 1.10 between 0 and 2.53): 0.25 above the highest wrong
prompt and 0.18 under the lowest own prompt, in share. The nearest wrong pairs are a stool and a small round table
(0.50, 0.46, 0.37), as is fair: a three-legged stool is a small round table's look-alike.

The first try kept halberds, glaives and poleaxes as a row of their own. The picture model paints a battle axe on a
long haft with a spike on top, which SigLIP 2 scores as a halberd: four of eight battle axes fell to 0.15 to 0.38.
A halberd is an axe on a pole, so it now shares the axe's row, as a vase shares the jug's.

The war hammers' scores (0.49 to 0.97) show the check sees what they are: the trials' two chosen war hammer pictures
score 0.49 and 0.58 (the copy on the other card 0.59), with the likeliest other "an axe".

### A known miss: the prompt's words written out

A picture of the prompt's words in black letters scores as the thing they name (the typographic attack on
picture-text models): "a wooden treasure chest" written out scored 1.00 against the general "text". Classes naming
the prompt's own words were tried. Each one that caught the written words also pulled real pictures down, because its
words sit next to the thing asked for:

| Written-words class | Own prompt, lowest | Written prompt, highest | Its noun written, highest |
|---|---|---|---|
| none (kept) | 0.926 | 1.000 | 1.000 |
| `the word "chest" written in letters` | 0.744 | 0.999 | 0.120 |
| `the words "<prompt>" written out in letters` | 0.218 | 0.190 | 0.065 |
| both | 0.328 | 0.332 | 0.002 |
| a page of typed text with the word | 0.223 | 1.000 | 0.862 |
| a sign with the word | 0.718 | 0.999 | 0.643 |

None is kept. The regression shows the written pictures and does not hold them. A picture of letters would also have
to pass the other picture checks, and its model an outline matched to it, but that is not measured. The next step is
a question-answering model ("is this a photo of an object or of text?", VQAScore's way).

### In the pipeline's choice of try

Every try is judged, and a try the check refuses is not chosen, as with the other picture checks (see [War
hammers](war-hammers.md)). It refuses the trials' two chosen war hammer pictures, and rightly: they show an axe head (0.49, 0.58).
Measured on every picture the trials painted that could be reached (268, one copy each; the same
pictures as the labelled set and its 16 longbows), the 239 clear pictures still pass, lowest 0.926. Of the 65 pictures
the trials chose, the check refuses the two war hammers with the axe head and one table from an early comparison of
leads (0.54, seen from straight above, labelled unclear), whose job never got past its pictures.

The labelled set is a regression case, "prompt-pictures" (`tools/trial/regression.py`): each clear picture must pass
its own prompt and fail every other, and an empty backdrop must fail every prompt. The whole regression set holds with it
(26 of 26; "prompt-pictures" took 673 s on 12 CPU threads, most of it SigLIP 2 seeing 252 pictures twice). The
written words are shown as they are: 24 of 30 still pass.
