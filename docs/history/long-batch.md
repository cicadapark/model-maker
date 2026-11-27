# A long batch: its memory, a resume, a guard

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## A long batch: its memory, a resume, a guard (12 November 2026)

0.2.0's code ran a 50-prompt baseline as one `model-maker batch` on the 12 GB card's machine; after 6.6 hours the
system's out-of-memory killer ended it at a 36 GB peak, and nothing it had made was kept as a batch could go on from.

**Where the memory went.** Measured on the 16 GB card (`2026-11-12-030-batch`, the batch's resident memory sampled
every 2 s; `tools/trial/batch_memory.py`), eight prompts with the old code, light settings:

| Stage | Peak | Least free | At its jobs' ends |
|---|---|---|---|
| pictures | 21.9 GB | 4.7 GB | 17.2 to 20.9 GB, growing job by job |
| shape | 24.2 GB | 1.5 GB | 8.9 to 14.8 GB |
| paint | 18.4 GB | 6.3 GB | 11.5 to 18.4 GB |
| finish | 18.8 GB | 5.1 GB | 18.0 to 18.2 GB |

The whole batch: 24.2 GB at its peak, over 42 minutes. Sixteen prompts with the old code: 25.1 GB.

Two things held memory, neither of them the jobs' own arrays (each stage reads its job's files from its folder, and the
finish over 40 stored jobs in one process stayed at 0.4 to 0.5 GB between jobs, 1.5 GB at its peak):

* **Memory freed but not handed back.** A stage's models closed at its end are freed, but glibc keeps a freed heap
  for the process, and torch keeps the pinned memory it freed: the shape model closed left 3.5 of its 4.6 GB held
  until `malloc_trim` gave all but 0.3 GB back. So each stage started on top of what the one before had held: the
  shape stage reached 24.2 GB (1.5 GB left free in the container), and the finish, which needs under 3 GB, ran at
  18 GB, the paint's models still held.
* **The picture stage growing job by job,** about 0.5 GB a job (17.2 GB after the first, 20.9 GB after the eighth): the
  picture model's offload moves its parts between the card and the computer's memory for every picture, and the
  freed copies were not handed back. At fifty prompts that is the 36 GB the baseline died at.

**The fix.** After each job and each stage, the memory freed is handed back (`memory.give_back`: the garbage
collector, torch's caches of the card and of pinned memory, glibc's `malloc_trim`); the models of a stage are kept for
its jobs and closed at its end, as before. With it, eight prompts and seventeen:

| Prompts | Pictures | Shape | Paint | Finish | Whole batch |
|---|---|---|---|---|---|
| 8 | 21.6 GB | 16.5 GB | 17.5 GB | 2.7 GB | 21.6 GB over 41 min |
| 16 | 21.1 GB | 17.1 GB | 17.3 GB | 2.3 GB | 21.1 GB over 74 min |

The peak no longer grows with the batch, and the stages no longer stack: the shape stage's peak fell from 24.2 to 16.5 GB
(the least memory left free from 1.5 to 8.9 GB), the finish's from 18.8 to 2.7 GB.

**Going on where it stopped.** Every file a stage writes last is written under a temporary name and renamed when
whole (the picture, the shape, the paint, and now the model and its sidecar), so a stage is done exactly when its file
is there. `model-maker batch` started again with the same `--out` skips each job's done stages, redoes a stage cut
short, and finishes only the jobs not finished from their present paint (the record keeps what each was finished from).

Tried on four prompts (`resume-cut`): the batch was killed while the chair's views were painted (two of six done),
then started again with the same `--out`. It skipped the pictures and shapes done, painted the chair again, and
finished all four: each job's verdict is the same as an uninterrupted run of the same prompts (`resume-whole`), 3 of 4
passing every check on both.

**The guard.** Before each stage the run compares the memory free to it (what the system has free, and what the
process holds) with the stage's need. Against the peak a batch measured in an earlier run in the same folder
(`memory.json`), a stage that does not fit is not started: the batch says what it needs and what is free, keeps
everything done, and exits with 74; run again with the same `--out` when there is room (MODEL_MAKER_MEMORY_CHECK=0
turns the check off). Against the needs measured here alone (pictures 22.4 GB, shape 13.7, paint 20.2, finish 2.8 on
the 16 GB card) it only warns and goes on: those peaks count model file pages the system can drop and read again, and
PCs with less memory made objects with 0.2.0. Each stage's peak is printed as the batch goes.
