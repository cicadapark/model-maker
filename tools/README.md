# tools

The repository's own check: `test_notice.py` (`NOTICE` names every model the registry holds). Its test:
`python3 -I -m unittest discover -s tools -p 'test_*.py'`.

`trial/`: the release's trial prompts (`trial-prompts.txt`) and `run_trial.sh`, which makes every one of them
on this PC's card with a time per object; `outline_bench.py`, the outline check's view, values, verdict and seconds
for each shape try and finish of made objects (CPU), to compare before and after a change to it; `regression.py`, the
regression set (shapes, pictures and made-up cases each check must pass or fail by name). `paint_tries.py` paints a
prompt's tries as the picture stage does (pictures only, any wording), `judge_tries.py` judges every picture under
some folders against its prompt on the CPU (and every quality's pair), `head_sheets.py` makes blind sheets of head
crops for labelling by eye (`head-cases.toml`; docs/objects.md, "War hammers"), and `quality_sheets.py` blind sheets of
whole cut-outs for labelling a quality (`quality-cases.toml`; docs/objects.md, "Several things, and qualities the words
name").

The diagnosis of 0.2.0's sealed run (docs/trials/2026-11-11-030-diagnosis.md), all on the CPU from made jobs:
`paint_sources.py` shows and counts where each colour of a finished object came from (each of the six views, the
touch-up's fill, a view at a slant, the spread); `view_agreement.py` measures how unlike neighbouring painted views are
where they see the same surface; `view_selection.py` bakes a job with one view per place beside the bake as made;
`local_likeness.py` measures the paint against the picture place by place (a check tried and not kept);
`judge_hafted.py` judges hafted things' pictures with changes to the judge's kinds and a "short" pair; and
`blind_sheets.py` makes blind sheets of whole cut-outs from `paint_tries.py` runs.

The regression set's cases, labels and data sums are here, but its data (the trials' made shapes and pictures) is not
published, so `regression.py build` and `run` need a copy of the trials' folders to work on (docs/objects.md, "The
regression set").

`sign_pictures.py`: content credentials (C2PA) for pictures made before the pipeline signed its own: each AI-made
picture of a folder that a list names (`docs/images/made-by-ai.toml` for this repository's) gets a signed manifest,
its pixels as they are; every other picture is left alone. Run it with the environment's Python
(`.venv/bin/python tools/sign_pictures.py FOLDER --list LIST.toml`); its tests are in `python/tests/test_credentials.py`.

`ci/pip-audit.lock`: pip-audit and what it needs, every wheel's sha256, for CI's vulnerability audit of
`python/requirements.lock` (`.github/workflows/ci.yml`, the job "vulnerability audit"); setup never installs it. Run it
by hand as CI does: `pip install --no-deps --require-hashes --only-binary :all: -r tools/ci/pip-audit.lock` in a Python
3.11 environment of its own, then `pip-audit --requirement python/requirements.lock --require-hashes --disable-pip
--strict`.
