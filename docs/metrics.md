# The numbers we keep

Every job writes its own record as it runs: one line of JSON per event in `metrics/jobs.jsonl` (`MODEL_MAKER_LEDGER`
moves the folder). The pipeline appends; nothing edits a line, and failures and aborted jobs are written like the
rest, so the numbers that show progress cannot be picked by hand.

| Event | When | What it holds |
|---|---|---|
| `load` | a stage's models loaded | seconds |
| `stage` | a job's stage done | seconds, the card's peak memory, the tries and their faults |
| `aborted` | a job's stage raised an error, or its models' loading did | the stage, the error |
| `job` | the finish (again on each refinish) | status, the failed checks, every counted check's value and limit, seconds per stage, first tries, the trim; its run |
| `not made` | a prompt refused (a bow, a character, an empty prompt; by `make`, a batch or the warm process alike), or a chest or book when SigLIP 2 is not downloaded | why |
| `warm` | a job of the warm process done | the jobs it made before, what is left on the card and the process's memory |
| `run` | a batch's end | its jobs, its whole seconds and its loading, for them to share |

A stage's seconds never hold a model's loading: each load has its own `load` line (the shape model's too, not counted
in the first shape's time, and the inpainter's, not counted in its job's paint). A job's line
says its **run** (`make`, a batch, or a warm process that keeps its models between jobs, `kept.py`): `warm` (the
process held models already, kept loaded), `whole_seconds` (the job's real time, its loading with it), `load_seconds`
(the loading it paid; none for a warm job unless it needed a model no job before it had) and the card's peak. A
batch's jobs leave the two out (they are known only at its end) and the batch writes them once, in a `run` line; the
summary gives each job an even share. A warm job writes no `load` line, so the ledger's loads are only those that
happened.

Each line has the job's id, folder, batch and prompt, the repository's version, the settings and the machine (the
card's name and memory, its driver, the CPU's threads): a time is never shown without the card it was made on. The
settings are the ones the job was made with, kept in its `job.json`: a refinish (`model-maker finish`), which runs
with the defaults, does not write those over them (only its texture is its own).

No prompt asked goes unwritten: a prompt `make` or the warm process
refuses has its own `not made` line (it has no job folder); in a batch a job whose stage raises, or whose models fail
to load, gets its `aborted` line and is set aside while the others go on, and the batch still writes its `run` line and
`times.csv`; the same prompt twice in a batch is one job. A job none of whose pictures passed its checks ends there with
its `job` line (failed on those checks), not after a shape and a paint the finish would fail. Each job's folder is named
by its slug and a hash of its prompt and options, so a folder never holds another order's job.

`regression.py run` appends a line to `metrics/regression.jsonl`: held, broken, each case's verdict.

    python3 -I tools/trial/summary.py progress [--write FILE]

prints the headline numbers (the unseen set's pass rate, the regression set, models passed this week) and a row per
batch. Each unseen batch is shown as its jobs first ended (the headline: a refinish after a limit moved cannot change
it) and as they last ended, each with the versions that made it; a second unseen run shows beside the first. Then, per
batch: pass rate of prompts asked, objects and objects held in the hand apart (the ledger's `weapon` field), median
minutes (the stages), median whole minutes (the
run's, loading included, of the jobs that say their run; how many were warm), card minutes per passing model, the
first tries, aborted jobs and the top faults. A job finished again counts once, as it last ended.

Trials made before the ledger were read in once with `summary.py backfill`; their lines say `backfilled` and their
card is the one the machine saw when they were read in.

The honesty rules: the regression and unseen sets are
never edited to look better; each release is compared on the same prompts; numbers come only from the ledger; the
unseen set runs once per release, after the code is frozen.
