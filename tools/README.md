# tools

The repository's own check: `test_notice.py` (`NOTICE` names every model the registry holds). Its test:
`python3 -I -m unittest discover -s tools -p 'test_*.py'`.

`trial/`: the release's trial prompts (`trial-prompts.txt`) and `run_trial.sh`, which makes every one of them
on this PC's card with a time per object; `outline_bench.py`, the outline check's view, values, verdict and seconds
for each shape try and finish of made objects (CPU), to compare before and after a change to it; `regression.py`, the
regression set (shapes, pictures and made-up cases each check must pass or fail by name). `paint_tries.py` paints a
prompt's tries as the picture stage does (pictures only, any wording), `judge_tries.py` judges every picture under
some folders against its prompt on the CPU, and `head_sheets.py` makes blind sheets of head crops for labelling by
eye (`head-cases.toml`; docs/objects.md, "War hammers").

The regression set's cases, labels and data sums are here, but its data (the trials' made shapes and pictures) is not
published, so `regression.py build` and `run` need a copy of the trials' folders to work on (docs/objects.md, "The
regression set").
