# The third trial's two faults

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## The third trial's two faults (10 November 2026)

The third trial passed 11 of 15. Two of the four failures had causes the checks did not name.

### The picture did not show what the words said

"a leather-bound book, closed" was painted open in two tries of four, and "a wooden treasure chest" with its lid
up in one of four. Every picture check passed them (they judge the cut-out's pieces, the backdrop, shadows and
margins, not what the object is), and the open ones scored best, so they were chosen. The book then settled onto
its spread covers (base level failed); the chest's inside could not be painted from the six views (paint coverage
failed). Both failed only through these side effects, after a card's worth of shaping and painting.

The fix, in two parts:

* The words. A thing with a lid, a cover or doors is shut unless the prompt opens it, and the picture model is told
  so right after the prompt. An open lid hides nothing useful from the shape model, but its inside is seen by no
  painted view, and it is not mirror-symmetric.
* A picture check, before any meshing: "shows what the words ask". The established technique is zero-shot
  image-text scoring with a contrast pair (as CLIP classifies without training; VQAScore and TIFA ask a question
  of a picture-and-text model instead, stronger on prompts with many parts, but a state of one thing is a single
  contrast): the picture is compared with words for the state asked for ("a closed chest", "a chest with its lid
  shut", ...) and the same words for the other state, each pair giving a share (the sigmoid of their score
  difference), averaged over the pairs and over two views, the whole picture and its square crop round the
  cut-out. The model is **SigLIP 2** (google/siglip2-so400m-patch14-384, **Apache-2.0**; Google, trained on its
  WebLI data), 4.5 GB, in the registry with every file's sum. It runs in the pinned transformers through its own
  SigLIP model and the Gemma tokenizer it ships (its own processor class came in a later transformers, so the pin
  stays). A picture passes when the asked state is at least three times as likely as the other (a share of 0.75):
  a share near one half is no verdict (moving the crop by one pixel moved one open book from 0.52 to 0.58), and a
  picture in doubt fails safe: another try is painted.

Measured on every book and chest the trials painted (36 pictures; `tools/trial/judge_states.py`):

| | Shut (by eye) | Open (by eye) |
|---|---|---|
| books, asked closed | 10 of 10 pass (0.99 to 1.00) | 5 of 6 fail (0.18 to 0.51) |
| chests | 16 of 16 pass (0.94 to 0.99) | 4 of 4 fail (0.28 to 0.42) |

The third trial's own tries: the book's open 628200 (chosen) and 628201 fail (0.44, 0.18), its shut 628202 and
628203 pass (0.99, 1.00); the chest's open 962714 (chosen) fails (0.28), its three shut tries pass (0.94 to 0.98).
With the check the shut tries are chosen. The objects with no lid, cover or doors are not judged, and their
picture checks are as before. The one open picture it misses is a book standing on its spine, opened in a V (0.90).

What did not work:

* The first SigLIP (google/siglip-so400m-patch14-384, Apache-2.0): it told the chests apart (open 0.14 to 0.26,
  shut 0.96 and over) but not the books: the open ones scored 0.67 to 0.97 for "closed", with every wording tried
  (four sets, from "a closed book" to "a book lying open in two halves, its spine down the middle").
* Other wordings for SigLIP 2 ("its covers together", "a half-open book", "a book opened in a V") moved the scores a
  few hundredths and caught nothing more; the plain wordings stay, so the judge is not fitted to these pictures.
* The whole picture alone missed a small chest with its lid up (0.61); the crop alone scored an open book 0.52. Both
  views averaged catch both.

Not tried: a question-answering model (VQAScore's way). Of the small ones with a commercial licence (Qwen3-VL 2B,
SmolVLM2, moondream2, all Apache-2.0), each is a larger download and slower per picture; it is the next step if
open-or-shut pictures keep getting through.

### A leg short by a little

"a small round table" came out with one of its three legs 24.6 mm short (3.8 % of its 65 cm height) and its top
dead level: set down, it settled 3.6° onto the short leg and failed "base level" and "centre over the base". The
picture showed every leg to the floor; the shape model guessed the thin leg's end from a few pixels. "Every foot on
the ground" had not named it: three legs always all rest on the floor once it tips onto the short one, so only the
tilt shows it.

The fix is the one a furniture maker uses on a rocking table, and a 3D print's base gets (a plane cut, then the
cut capped): find the floor the object should stand on, a plane square to its up axis, and cut away what reaches
below it. The finish (`levelling.py`) looks for the underside's low places (each foot's end, from the base checks'
own height map), and tries the floors at their heights from the lowest up; the first at which the cut object passes
every base check is kept. Each cut leg is closed with a flat cap (its loop of cut edges, ear-clipped), so the
surface stays closed. It cuts at most 5 % of the height (the height the base checks look for a foot within: 3.25 cm
on the table, 4.75 cm on a 95 cm chair); a foot short by more is left as made and stays a failure, as is a thing
the cut does not set level, and a thing made leaning whole, whose flat top is square to the floor its feet settle it
on rather than to its own up (no foot is short; see [Defects found by measurement](defects.md)). The shape the model made is kept unchanged in `shape.npz`; the sidecar's `levelled`
says what was found and cut, and the report shows it beside the checks ("feet trimmed to one floor", not counted).
The painted colours follow the cut: each corner takes the colour of the made corner nearest it.

On the trial's table (`./model-maker finish`, the CPU): the two long legs trimmed 24.6 mm; it settles 0.0°, its
centre 55 % inside its base, closed and in one piece; it now passes every check. The cost: the table is 2.5 cm
lower than its picture, so its outline's edges lie 0.80 % of the height apart on average (0.45 % before; the limit
is 1.9 %) and its likeness is a little lower (mean colour difference 13.2, 11.8 before). The first trial's chair,
its back legs 28.9 mm short, is trimmed the same way and passes; the regression set keeps it failing "base level"
as made, so the check still catches what the shape model got wrong.

What was not done: lengthening the short leg instead. It would keep the table's height, but it means finding that
leg's own cross-section and extending it down, a guess at its shape; the cut is exact, keeps the surface closed,
and costs the object a few centimetres of height at most.
