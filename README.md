# model-maker

Make a 3-D object from a written description. A picture model paints it, a shape model builds it, a
view model paints it from six sides to fit its shape, and our own tools check it, bake its texture
and write a standard `.glb` file you can open in any 3-D tool.

```
git clone https://github.com/cicadapark/model-maker && cd model-maker
./setup.sh --check-only
./setup.sh --hf-token-file FILE
./model-maker make "an old looking chair"
./model-maker make "an old iron sword" --size 0.9
```

model-maker makes **3-D objects**: furniture, containers, tools, decor, and things held in the hand such as
swords, axes, spears and shields. Animals come in a later release, and characters (people and fantasy creatures)
after the animals.

Every object is checked, and each check is measured: its outline against its picture, the mesh
whole (no holes, no loose pieces), standing on its base, mirror-symmetric where the words imply it,
the paint covering it, and what the paint touch-up filled in matching the paint round it. For a thing held in the hand,
its blade or shaft is checked straight, its grip found and its length set, and its `.glb` marks the grip and the tip
so a game can put it in a hand. Each object
comes with a report of every check and a sidecar naming every model used and its licence. See
`docs/objects.md`.

## What it makes

See them in 3-D, turned round in your browser: **[Model Maker Studio](https://cicadapark.com/model-maker-studio/)** (cicadapark.com).

![Eight 3-D objects made by model-maker, each from the line of text under it: a chair, a barrel, a lantern, a clay jug, a shield, a sword, an axe and a spear](docs/images/examples.webp)

Eight objects made during this release's development, each from the line of text under it, each passing every
check. Every one is a `.glb` you can open in any 3-D tool, turned all the way round:

![The clay jug, turned in steps of 45 degrees: its handle is an open loop from every side](docs/images/clay-jug-turning.webp)

And every one comes with a report of its checks: the picture it was made from, its cut-out, the tries, the outline
against the picture, the paint, and each check's value against its limit.

<img src="docs/images/report-chair.webp" width="600" alt="The top of the chair's report: passes all 16 checks; its picture, cut-out and tries; its outline against the picture; its turntable">

## What to expect

This is an early release: the object path has been through a handful of trials on two cards (`docs/objects.md`,
`docs/trials/`).

**Pass rate: 15 of 25 right; 18 of 25 pass every check.** Twenty-five prompts were written before this version
was frozen and kept sealed: nothing was tuned on them (`tools/trial/unseen-0.2.0.txt`). Each was made once, at the
frozen code, on each of two cards: a 16 GB RTX 3080 laptop card and a 12 GB RTX 3060. On the 3080, 18 passed every
check (19 on the 3060). Every object was then looked at by eye, turned all the way round: on each card 15 of 25 pass
every check and are right. Both cards paint the same pictures from the same seeds, so their objects are near alike.
The whole run took 4.0 hours on the 3080 and 3.5 hours on the 3060: 8 to 10 minutes an object.

![The 25 sealed prompts as made on the RTX 3080, each with its verdict by eye: 15 fine, 1 flawed, 2 wrong though they pass every check, 6 failing a check, 1 not made](docs/images/sealed-0.2.0.webp)

| Prompt | Checks (3080) | By eye |
|---|---|---|
| a wooden armchair | pass | fine |
| a stone garden bench | pass | fine |
| an oak wardrobe | pass | fine |
| a small writing desk | pass | fine |
| a copper kettle | pass | fine |
| a brass oil lamp | pass | fine; its glass chimney painted opaque |
| a wooden ladder | pass | fine |
| a wooden rake | pass | fine |
| a pickaxe | pass | fine |
| a blacksmith's hammer | pass | fine |
| a pair of iron tongs | pass | fine: one object, as it should be |
| a stone birdbath | pass | fine |
| a kite shield | pass | fine |
| a straight steel longsword | pass | fine |
| a wooden quarterstaff | pass | fine |
| a glass bottle | pass | the glass painted dark on one side |
| a ceramic teapot | pass | wrong: large black patches painted on one side |
| a short iron mace | pass | wrong: a long thin rod with a small knob |
| a wooden bed frame | fail (outline, holes) | looks right |
| an iron cauldron | fail (paint fill) | looks right |
| a clay flower pot | fail (paint) | grey smears inside |
| a wooden bucket | fail (paint fill) | a tub with side handles; dark smears inside |
| a wooden spinning wheel | fail (paint fill) | wrong: a wheel on a stand |
| a wooden water wheel | fail (paint) | wrong: a wheel with no paddles |
| a wooden club | not made | no picture try showed a club |

Twenty-five prompts are still a small sample: the true rate of right objects could plausibly be anywhere from about
4 in 10 to 3 in 4. These 25 are now part of the development set. The first version's ten (`unseen-final.txt`) gave
4 of 10 passing every check, 2 of them right by eye.

| | |
|---|---|
| System | Linux on x86_64 |
| Graphics card | NVIDIA, 12 GB or more (tried on 12 GB and 16 GB), a driver with CUDA 12 or newer |
| Computer's memory | 24 GB or more (16 GB to try) |
| Disk | about 60 GB used (models 54 GB, environment 6 GB); keep 75 GB free |
| Setup | about 20 to 35 minutes (57 GB downloaded; clean PCs took 19 to 33 minutes) |
| One object | about 7.5 to 10 minutes on a 16 GB card; 8 to 13 minutes on a 12 GB card (`docs/setup.md`, "The chair and the sword on 12 GB") |
| Also needed | git, curl, tar, a C and C++ compiler; `setup.sh` brings its own Python and, if needed, its own Rust |
| Recommended | tesseract 5 or newer with its English data (`sudo apt install tesseract-ocr tesseract-ocr-eng` on Debian 12 or Ubuntu 24.04; Ubuntu 22.04's is version 4: `docs/setup.md`), to refuse picture tries with letters painted on the object |

**Not yet:**

* bows (a bow prompt is answered that they come later; nothing is made);
* animals (a later release);
* characters, meaning people and fantasy creatures (after the animals); a statue of either is an object, and a prompt
  that names both ("a knight on a horse") is answered as a character;
* Windows and macOS; WSL2 on Windows should work the same but has not been tried;
* making an object without an NVIDIA card: there is no CPU path for the picture, shape and paint models (the checks,
  the bake and the report do run on the CPU: `./model-maker inspect`, `./model-maker finish`).

**Known limits:**

* one object, in one piece: "a pair of ..." or "two ..." is answered before any painting, asking for one thing
  ("a pair of scissors" and "a pair of tongs" are one object, and made);
* of the qualities a prompt names, "curved" and "round" are judged; others ("tall", "broken", "rusty") are not;
* lettering painted on an object is caught when it is written straight across a flat face; lettering round a rim (a
  coin, a medal) and very small marks still get through;
* paint that is wrong where the six views painted it (the teapot's black patches) is not judged: the paint checks
  measure its coverage and the touch-up's fill;
* a near relative can pass for the thing asked for (a rod with a knob for a mace);
* the paint-fill check can fail an object whose inside rightly has another colour (a lining, shelf boards).

## Setting up

```
./setup.sh --check-only
./setup.sh --hf-token-file FILE
```

`setup.sh` brings everything into this folder: its own Python, the pinned packages, Rust if there is none, the view
painter's code and the models, each after its licence is shown and accepted, every file checked against its sha256.
One model, [FLUX.1 [schnell]](https://huggingface.co/black-forest-labs/FLUX.1-schnell), is gated: its hub page asks a
signed-in Hugging Face account to accept its terms, which shares the account's user name and e-mail address with
Black Forest Labs, and setup then needs that account's access token (a file holding the token alone, on one line).

Setup asks seven licence questions, one for each model, each just before that model downloads, so they come over about
20 minutes; to answer them all at once, read the licences first (`registry/models.toml`, `NOTICE`) and run
`./setup.sh --yes`. It uses about 60 GB (models 54 GB, environment 6 GB); keep 75 GB free. The self-test at its end
takes about 2 to 11 minutes. `--check-only` checks the PC, not the models (`.venv/bin/python -I
setup/fetch_models.py --verify` checks those once downloaded), and it does not see what else is using the card: **on a
12 GB card, stop anything else that uses the card first** (the release check's peak on a 12 GB card was 11.85 GB). See
`docs/setup.md`, also for the warnings a make prints that you can ignore.

## What you get

For each object, in its own folder, `made/<name>-<id>/` (`--out DIR` chooses another folder): its `.glb` (open it in
any 3-D tool or game engine), its sidecar (`<name>.provenance.json`: the prompt, every model used with its version and
licence, the seeds, every check) and its report (`report/report.html`: every check, measured, with the pictures it was
judged by).

## What is here

| Folder | What it holds |
|---|---|
| `model-maker` | the command: `make`, `batch`, `finish`, `inspect`, `picture-check`, `words`, `bug-report`, `reproduce` |
| `python/` | the pipeline: pictures, cut-out, shape, paint, the checks, the bake, the `.glb`, the report (`python/tests/` its tests) |
| `crates/` | the Rust crate `model-maker-simplify` (a shape to its triangle budget) |
| `registry/` | the AI models the pipeline uses, their licences and the sha256 of their files |
| `setup.sh`, `setup/` | the one command that sets a PC up (`docs/setup.md`) |
| `docs/` | how the pipeline works |
| `tools/` | the repository's own checks, and `tools/trial/` (the release's trial prompts and how to run them) |

## How to report a problem

Open an issue on GitHub with the **Bug report** form, and attach the bug report file for the job:

```
./model-maker bug-report made/chair-ec4d6029
./model-maker bug-report made/chair-ec4d6029 --with-model    # the .glb too
```

It writes one file (`bug-report-<name>-<id>.zip`; give `--out FILE.tar.gz` for a .tar.gz) and prints its size and
what it left out. It holds the job's record and sidecar, its report and pictures, the picture it was made from and
its cut-out, the job's lines from the ledger, and the versions it was made with: the code, the AI models, Python,
torch, the graphics card and its driver. The prompt is sent whole, as the remake needs it: leave out of a prompt
anything you would not post. It leaves out the large in-between files, your environment and any token, and it
replaces your home folder, your user name, your machine's name and other folder paths in every text it copies.
`README.txt` inside says what it holds. Drag the file into the form's box (GitHub takes files up to 25 MB). For a
question, use the **Question** form; for a security problem, see `SECURITY.md`.

We claim no rights in what you make; the models' terms apply (`NOTICE`). As the code that made an object is on your
machine too, we cannot take the object or its report as proof of anything: we can check a claim only by remaking it.
So a maintainer makes the same prompt with the same seeds, settings and version on a card of their own
(`./model-maker reproduce FILE --out DIR`) and compares the picture, every check's value and the verdict with yours.

## Contact

developer@cicadapark.com. Security problems: see `SECURITY.md`.

## Licence

Apache-2.0 (see `LICENSE`). Third-party code and the AI models used are listed in `NOTICE`. What
model-maker makes is made by AI systems from text prompts; each sidecar says which ones. One of them, Stable Diffusion
XL, comes under the CreativeML Open RAIL++-M licence: the use restrictions of its Attachment A apply to everything
made with it, and so to every object model-maker makes (each sidecar and report says so).

The code was written with AI (Claude); the commits name each model that helped.
