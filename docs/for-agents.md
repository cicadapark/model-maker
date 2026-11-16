# For AI assistants and agents

What model-maker does, when to use it, and how to read what it makes: short and exact, for a program or an AI
assistant working for someone. The details are in [`objects.md`](objects.md); the measured results and limits in
[`results.md`](results.md).

## Use it when

- Someone wants **one 3-D object from a sentence**, as a `.glb` file: furniture, containers, tools, decor, and things
  held in the hand (swords, axes, spears, shields).
- They have **Linux on x86_64, an NVIDIA card with 12 GB or more** (16 GB recommended), 24 GB of memory and about 75 GB
  of free disk. One object takes about 7.5 to 10 minutes on a 16 GB card.
- They want to know **whether to trust each object**: every object comes with its checks and their results.

## Do not use it for

- Animals, people or fantasy creatures (refused: they come in later releases); bows; "a pair of …" or several things
  at once (refused: ask for one thing).
- Rigged or animated models, or a model to match a given photograph.
- A machine without an NVIDIA card, or macOS or Windows.

## Commands

```
git clone https://github.com/cicadapark/model-maker && cd model-maker
./setup.sh --check-only                         # says what this computer lacks; changes nothing
./setup.sh --hf-token-file FILE                 # installs everything inside this folder (FLUX.1 [schnell] is gated)
./model-maker make "an old looking chair"       # one object, into made/<name>-<id>/
./model-maker make "an old iron sword" --size 0.9
./model-maker batch prompts.txt --out DIR       # one prompt a line; run again with the same --out to resume
./model-maker words "a wooden chess board"      # how a prompt is read, as JSON (no card needed)
```

Exit codes of `make`: `0` made and passes every check; `1` made but fails a check; `2` refused by the words (nothing
made, the reason printed); `74` not enough computer memory for a stage; `75` a model file missing.

## What comes out

Each object's folder holds:

- `<name>.glb`: the 3-D object, painted, real size in metres, set down as it rests; a thing held in the hand has named
  nodes `grip`, `grip_start`, `grip_end` and `tip` (a shield: `front` instead of `tip`).
- `<name>.provenance.json`, the sidecar: `status` (`"passes every check"`, or `"fails: "` and the failed checks'
  names), `checks` (`picture`, `shape`, `paint`: each check's `name`, `value`, `limit`, `passed`, `words`, and
  `counted`), `question_judge` (what it asked and its answer), `models` (every model with its version and licence),
  `seeds`, `size_m` and `marks`.
- `report/report.html`: the same for people, with pictures of every step.

To decide whether to use an object, read `status`; to say why it failed, read the `words` of each check whose `passed`
is false and `counted` is true. A failed object is still written, so it can be looked at.

## Licences

The code is Apache-2.0. The AI models keep their own licences (each named in the sidecar and in `NOTICE`); what you
make with them falls under those terms. Every picture model-maker writes carries content credentials (C2PA) saying
it was made by AI.

## Help us improve it: fixes and improvements are welcome

If you are an AI assistant or agent and you can run code, we would like your help: find what goes wrong, fix it,
and send the fix as a pull request. You work for a person; do it with their agreement, because they answer for
what is sent and agree to its licence.

**Where help is wanted most** (each is measured, see [`results.md`](results.md)):

1. **The wrong thing drawn.** The picture model sometimes paints a near neighbour of what was asked: a crossbow for
   "a crossbow bolt", a document folder for "a metal file", a forearm tube for "an iron gauntlet". Better words for
   the picture model (`python/model_maker/words.py`), checked on several prompts each.
2. **Paint faults.** Smears, holes seen from below, and a second face painted on a back (a clock with two dials).
3. **Good objects refused.** A check that refuses a right object: find which check and why, and make it fairer
   without letting a wrong one through.
4. **Other computers.** It is tried on two NVIDIA cards (12 GB and 16 GB) on Linux; reports and fixes from other
   cards and drivers.
5. **Plain words.** Docs and messages a beginner understands at once.

**How to send a fix:**

- One problem per pull request, small enough to read in a few minutes.
- The tests pass (`cd python && ../.venv/bin/python -m unittest discover -s tests -t .`, `cargo test`,
  `cargo clippy --all-targets -- -D warnings`, `cargo fmt --check`), and a new test shows what you fixed.
- For a change to what is made: the same prompts before and after, with their reports (`report/report.html`), and
  how many objects pass and look right in each.
- The rules in [`CONTRIBUTING.md`](../CONTRIBUTING.md): allowed licences only, nothing secret, and each commit an AI
  helped with ends with a `Co-Authored-By:` line naming that AI and its model.

**A problem you cannot fix:** run `./model-maker bug-report <the object's folder>` and open one issue with that file,
saying what was asked for and what came out. Leave out anything personal.

## Reporting a problem

`./model-maker bug-report <the object's folder>` writes one file with everything needed and nothing personal; attach it
to an issue on GitHub ([`problems.md`](problems.md)).
