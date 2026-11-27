# How the checks came to be

The reference, [`docs/objects.md`](../objects.md), says what model-maker does and every check's limit. These files
say what each limit and fix was measured on, what went wrong on the trials and what did not work, dated as written.
They were sections of `objects.md` until it was split after 0.3.3; the text is as it was, only its links point to the
new places.

The measurements come
from the release's trials: prompt sets made end to end on a 16 GB card (an RTX 3080 laptop card) and a 12 GB card (an
RTX 3060), named by their order ("the first trial", "the third trial", "the fifth trial", "the 12 GB card's fourth
trial"), and the "fix-check", the ten regression prompts made again after the third trial's fixes. The notes of the
first two, and the diagnosis of 0.2.0's sealed run, are in `docs/trials/`; the later trials' notes, their made objects
and the regression set's data are not published (they are large, and hold machine-specific records): what they showed
is given here.

| File | What it holds |
|---|---|
| [measured-limits.md](measured-limits.md) | what the reference's steps, picture checks, "Open underneath" and `--recut` were measured on |
| [thin-parts.md](thin-parts.md) | the first trial's thin parts, the longbow, and the string ready for bows |
| [regression-set.md](regression-set.md) | every case of the regression set, as it was added |
| [third-trial.md](third-trial.md) | the third trial's two faults: a picture that did not show the words, a leg short by a little (10 Nov) |
| [fix-check.md](fix-check.md) | the fix-check's three faults: a book on its edge, a chest's outline, a candle's paint (10 Nov) |
| [touch-up-fill.md](touch-up-fill.md) | the touch-up's fill, counted then judged on the model (10 and 11 Nov) |
| [outline-search.md](outline-search.md) | the outline's search made fast (10 Nov) |
| [holes.md](holes.md) | the jug's handle and holes the picture shows (10 Nov) |
| [state-check.md](state-check.md) | the state check's words: a state's look, not its name (10 Nov) |
| [picture-check.md](picture-check.md) | the picture judged against the prompt (10 Nov) |
| [war-hammers.md](war-hammers.md) | war hammers (10 Nov) |
| [defects.md](defects.md) | 23 defects found by measurement and 15 plausible ones, each fixed or shown not to apply |
| [shape-decoder.md](shape-decoder.md) | the shape model's own denoiser and decoder (10 Nov) |
| [held-things.md](held-things.md) | hafts followed, grips on the haft, a shield's handles (10 Nov), a haft that forks (14 Nov) |
| [several-and-qualities.md](several-and-qualities.md) | several things, pairs, and qualities the words name (11 Nov) |
| [lettering.md](lettering.md) | painted lettering (11 Nov) |
| [view-agreement.md](view-agreement.md) | the painted views judged against each other, one view per place (11 Nov) |
| [false-refusals.md](false-refusals.md) | false refusals: the bed frame, the teapot's fill, the desk's symmetry (11 Nov) |
| [insides.md](insides.md) | insides, bake-only: four ways measured (11 Nov) |
| [hafted-pictures.md](hafted-pictures.md) | maces and clubs, and a hafted weapon painted upright (11 Nov) |
| [long-batch.md](long-batch.md) | a long batch: its memory, a resume, a guard (12 Nov) |
| [pillow-13.md](pillow-13.md) | Pillow 13, and the model stack held (13 Nov) |
| [question-judge.md](question-judge.md) | is it the thing asked for? The question judge (14 Nov) |
| [thin-dark-strokes.md](thin-dark-strokes.md) | thin dark strokes: the bake colours every triangle (0.3.2) |

What each release measured is in [`docs/results.md`](../results.md).
