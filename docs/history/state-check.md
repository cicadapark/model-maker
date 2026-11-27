# The state check's words

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## The state check's words: the state's look, not its name (10 November 2026)

On a run of the 12 GB card "a leather-bound book, closed" was painted open: both halves in the cover's leather, side
by side, the page edges in two blocks, a valley down the middle. "Shows what the words ask" scored it 0.88 shut, it was
chosen, the shape model built an open book in two lobes, and only "every foot on the ground" caught it, 9 card minutes
later. The same seed on the 16 GB card paints the book shut (one cover, a ribbon between the page edges), which is why
that card's runs of the same prompts passed it rightly.

### Why

The check compared the picture with the states' names: "a closed book", "a photo of a closed book", "a book lying
shut" against the same words open. Contrastive picture-text models recognise an object's state far worse than the
object (Newman et al. 2024, "Do pre-trained vision-language models encode object states?": about 30 points lower,
which they lay to weak localisation and to the text side), and a state's name alone is scored largely by the noun's
usual look. An open book in pictures is mostly white pages; a picture of brown leather and no white page reads as
"book, cover", so shut. The open books the check caught before were the ones with white pages (0.07 to 0.18); those
painted in the cover's leather were near its limit (0.40, 0.51) or over it (0.88).

### The fix

The established way past a class name's bias is classification by description (Menon and Vondrick 2023, "Visual
classification via description from large language models"; CuPL, Pratt et al. 2023): each class is scored by
sentences naming what it looks like, keeping the class's name in each (without it such descriptors lose most of their
worth). Each state is now described in pairs that name the same evidence both ways, the share of each pair averaged
as before (`words.STATE_JUDGE_TEXTS`):

| A thing's | Shut | Open |
|---|---|---|
| cover | its front cover one flat panel with no fold across it | its two halves side by side with a fold down the middle |
| | the edges of its pages in one even block | the edges of its pages split in two at the middle |
| | its covers pressed together | its covers spread apart |
| | no pages showing | its pages showing |
| | its spine straight | bent open along its spine |
| lid | its lid down on top of it | its lid raised up behind it |
| | its inside hidden | its inside showing |
| doors | its doors shut flat across its front | its doors swung out |
| | its inside hidden | its inside showing |

Each begins "a closed {thing}," or "an open {thing},". The score is the same share as before (a sigmoid of a pair's
score difference, averaged over pairs and over the two views), so the limit stays at 0.75. No cabinet or wardrobe has
been painted yet; the doors' pairs are written the same way and untested.

Measured on every book and chest picture the trials painted, one copy of each (52; 17 open by eye, 35 shut;
`tools/trial/state-cases.toml`), asked shut:

| | Before: the states' names | After: their look |
|---|---|---|
| shut books pass | 15 of 15 (0.98 to 1.00) | 15 of 15 (0.91 to 1.00) |
| open books fail | 7 of 9 (0.07 to 0.51; missed 0.88, 0.90) | 8 of 9 (0.06 to 0.61; missed 0.81) |
| shut chests pass | 20 of 20 (0.93 to 0.99) | 20 of 20 (0.88 to 0.98) |
| open chests fail | 8 of 8 (0.28 to 0.42) | 8 of 8 (0.15 to 0.34) |

The leather-coloured open book goes from 0.88 to 0.61 and fails; its shut twin from the 16 GB card stays at 0.91. The other
two leather-coloured open books fall from 0.51 and 0.40 to 0.41 and 0.36. The book standing open in a V on its spine is
still missed (0.90 to 0.81): it is kept in the labelled set as a known miss, and the run says when it is caught. The
regression set holds (25 of 25 cases, the labelled pictures as one), and with the old words the new cases break (the
open book passes at 0.88).

### What did not work, and what was tried

* Adding the descriptions to the states' names (three names and two descriptions a side): it failed the leather book
  too (0.61), but the names pull towards the noun's usual look, and nothing they add is evidence of a state; the
  descriptions alone are kept.
* Averaging the wordings' text vectors before scoring (CLIP's prompt ensembling) or averaging their scores per state:
  the same verdicts on every picture, the V book scored higher (0.89, 0.90 against 0.81); the pairs stay.
* The two strongest pairs alone (the fold down the middle, the page edges split): the leather book fell to 0.20, but its
  shut twin to 0.80, five hundredths from the limit. Five pairs keep the shut books at 0.91 and over.
* A third lid pair ("its lid resting flush on its box" against "tilted back on its hinges") brought a shut chest down
  to 0.84 and caught nothing more; it was left out.
* A pair for the V ("its two covers flat against each other" against "apart in a V") moved the V book by a hundredth
  and the leather book up to 0.67; it was left out.

Five sets of descriptions were scored in all, each on the same 52 pictures; the limit was not moved. Not tried: a
question-answering model (VQAScore's way), still the next step if the V book or another state gets through.
