# Setting up a PC

One command sets up everything model-maker needs, inside its own folder:

```
git clone https://github.com/cicadapark/model-maker && cd model-maker
./setup.sh --check-only                       # what this PC has and lacks (changes nothing)
./setup.sh --hf-token-file ~/hf-token         # everything; asks before each model's licence
./model-maker make "an old looking chair"
./model-maker make "an old iron sword" --size 0.9
```

Linux on x86_64 with an NVIDIA card of 12 GB or more (tried on 16 GB and 12 GB, below), a driver with CUDA 12 or
newer, 24 GB or more of the computer's memory (16 GB to try), git, curl, tar and a C/C++ compiler (`build-essential`
on Debian and Ubuntu, for meshoptimizer). Nothing else: setup brings its own Python and, when the PC has none, its own
Rust. It uses about 60 GB (models 54 GB, environment 6 GB); keep 75 GB free.

Optional, recommended: tesseract (Apache-2.0), version 5 or newer with its English data, which reads each picture try
for letters painted on the object (the check "no painted lettering", docs/objects.md, "Painted lettering"). Setup
installs no system package; install it yourself:

* Debian 12 (5.3.0) and Ubuntu 24.04 (5.3.4): `sudo apt install tesseract-ocr tesseract-ocr-eng`.
* Ubuntu 22.04 packages version 4.1.1, which is too old (it has no Sauvola threshold, and the check does not use it).
  Tesseract's own installation page names a PPA with version 5 for Ubuntu
  (https://tesseract-ocr.github.io/tessdoc/Installation.html): `sudo add-apt-repository ppa:alex-p/tesseract-ocr5`,
  `sudo apt update`, then `sudo apt install tesseract-ocr tesseract-ocr-eng` (5.4.1 for 22.04; its page calls it
  unsupported, so it is your choice). Or build tesseract 5 from its source, as that page says.

`./setup.sh --check-only` shows the version found and whether it can be used. Without a usable tesseract a make still
runs, each try says "not judged" and why, and the job says it was not read for painted lettering (it is never counted
as a pass by the trial tools; `--require-lettering` makes such a job "not made").

`--check-only` checks the PC, not the models: once they are downloaded, `.venv/bin/python -I setup/fetch_models.py
--verify` checks every model file against its sum. Nor does it see what else is using the card: on a 12 GB card,
stop anything else that uses the card before a make (the release check's peak on a 12 GB card was 11.85 GB).

Setup asks seven licence questions, one for each model, each just before that model downloads, so they come over
about 20 minutes. To answer them all at once, read the licences first (`registry/models.toml` names each, with its
terms; `NOTICE`) and run `./setup.sh --yes`: each licence is still shown, and the answer is recorded as `--yes`.

What a make writes goes in `made/<name>-<id>/` (the id: a hash of the prompt and its options); `--out DIR` chooses
another folder.

## What it does

| Step | What | Where | Checked by |
|---|---|---|---|
| 1 | the checks of `--check-only`; it stops before changing anything if one fails | | |
| 2 | uv 0.13.5 (MIT or Apache-2.0), the Python package installer | `.tools/uv` | the release's sha256, named in `setup.sh` |
| 3 | CPython 3.11.14 (python-build-standalone, by uv) and the environment | `.tools/python`, `.venv` | uv's own sums of its Pythons |
| 4 | the Python packages, exactly `python/requirements.lock` (`uv pip sync --require-hashes`) | `.venv` | every wheel's sha256 in the lock; then `uv pip check` and `check_licences.py` (no GPL, AGPL, LGPL or non-commercial package; every shared library a wheel bundles on the allow-list, see "What the wheels bundle") |
| 5 | Rust, only when the PC has neither `rustup` nor a `cargo` as new as `Cargo.toml`'s `rust-version` (a distribution's older `cargo` is not used): rustup 1.28.2, then the toolchain `rust-toolchain.toml` names | `.tools/rust` | rustup-init's sha256, named in `setup.sh` |
| 6 | `model-maker-simplify` (`cargo build --release --locked`) | `target/release` | `Cargo.lock` |
| 7 | each model: its licence shown (where it carries use restrictions, SDXL's RAIL++-M, its whole text saved to `models/licences/` to read first), then asked; its code checked out at its commit (MV-Adapter only); its files downloaded at its hub revision | `models/`, `sources/` | every file's sha256 in `registry/models.toml`; a checkout's commit by `git rev-parse`, and again, with nothing changed or added, each time the code is imported |
| 8 | the self-test (about 2 to 11 minutes): the Python tests, every model file present and checked (by its stamp when it passed), every stage's code importing as the pipeline imports it (`setup/smoke.py`, no weights), PyTorch sees the card | | the tests' own output, made-up jobs whose failures they expect, goes to `.tools/self-test.log` |

Everything setup installs lands in the repository's folder (`.tools`, `.venv`, `target`, `models`, `sources`; git
ignores them): delete the folder and it is gone. The system and the shell's setup are left alone (uv's cache is
`.tools/uv-cache`; a Rust setup.sh installed lives in `.tools/rust`). What it touches outside the folder: a PC that
has its own rustup keeps using it, and rustup then installs the pinned toolchain the usual way, under `~/.rustup`,
and that cargo keeps the Rust crates it downloads under `~/.cargo`; and the model fetcher reads the Hugging Face token
from `~/.cache/huggingface/token` when no other is given (below). Setup writes nothing else in the home folder.

A second run checks what is there and carries on: a download stopped half-way goes on from where it stopped (each file
is written to `<name>.part` and renamed once its sum matches); a file already checked is not read again unless its
size or time changed (`models/.verified`); an accepted licence is not asked again (`models/.accepted` keeps which
licence and version, the day and how: asked or `--yes`).

Options:

| Option | What |
|---|---|
| `--yes`, `--accept-licences` | accept every model's licence without asking (each is still shown) and install Rust into `.tools/rust` without asking; for unattended runs by someone who has read them |
| `--hf-token-file FILE` | a Hugging Face access token, for the gated model (below); the file must hold the token alone, on one line; also `HF_TOKEN_FILE`, the hub's own `~/.cache/huggingface/token`, or `HF_TOKEN` |
| `--skip-models` | all but the models: enough for `./model-maker inspect` and `finish` |
| `--skip-tests` | no self-test |

`setup/fetch_models.py` is the models' part on its own (`--only ID ...`, `--verify` to check the files there).

### A gated model

FLUX.1 [schnell] (Apache-2.0) is gated on its hub: the hub hands its files only to a signed-in account that has
accepted its terms on <https://huggingface.co/black-forest-labs/FLUX.1-schnell> (once; approved at once). Accepting
shares the account's user name and e-mail address with Black Forest Labs, as the page says. So:

1. sign in to huggingface.co, open the model's page and accept;
2. make an access token (Settings, Access Tokens; "read" is enough) and put it in a file only you can read
   (`chmod 600`), the token alone, on one line;
3. `./setup.sh --hf-token-file THAT-FILE`.

The token is read from the file and sent only to huggingface.co (not on to the file store the hub redirects to). It is
never printed or written anywhere. Without it, setup says which model it could not fetch and why. The other models
need no account.

### The models it downloads

Only the files the pipeline loads: 57 GB in all. FLUX.1 [schnell] in the diffusers layout (33.7 GB: its transformer
and its second text encoder at full precision; the pipeline quantises them to 4 bits as it loads them), BiRefNet
0.4 GB, TripoSG 7.6 GB, MV-Adapter's image-and-geometry weights 3.4 GB with the SDXL fp16 autoencoder 0.3 GB, SDXL
base 1.0 in fp16 7.0 GB, SigLIP 2 4.4 GB (the sizes setup prints).

The registry lists every file the pipeline reads, the small ones too (configs, tokenizers and index files), so
`verify_folder` checks each.

### The models' code

MV-Adapter's repository (Apache-2.0) at `4277e001` into `sources/mv-adapter`, fetched at its commit alone (`git fetch
--depth 1 origin <commit>`). That is the only model code setup checks out from a repository. BiRefNet's own model
code (`birefnet.py`, `BiRefNet_config.py`) comes with its weights from the hub, at the pinned revision, each file
checked against its sha256 in the registry.

No code of TripoSG's repository is fetched, imported or run: only its weights (MIT, the hub's model card) and their
configuration files are downloaded, at the revision the registry pins, and model-maker runs them with its own code.
The denoiser is `python/model_maker/shape_net.py`; the decoder from the latent to the field the surface is found in,
the attention processor and the query points' embedding are `python/model_maker/shape_vae.py`, written from
diffusers' Apache-2.0 parts (`AttnProcessor2_0`, `Attention`, `FeedForward`, `FP32LayerNorm`; NOTICE), the TripoSG
paper and the weights' tensor names, shapes and configuration. Their behaviour was confirmed by comparing their
outputs with the upstream code's on the same inputs, before that code was removed from setup: they give the same
field (docs/objects.md, "The shape model's own decoder"). They load every tensor of the weights by name (the VAE's
encoder alone is not built: model-maker only decodes). Why: the repository's own `LICENSE` is MIT, but its `NOTICE`
names code derived from HunyuanDiT and FlashVDM under Tencent Hunyuan community licences, and the folder its code sits
in, `triposg/`, holds the text of the Tencent Hunyuan FlashVDM Community License Agreement, which does not apply in
the European Union, the United Kingdom and South Korea. Setup removes a `sources/triposg` folder if it finds one. The
FlashVDM decoder is neither written again nor used.

### Python

The pipeline was tried on Python 3.11 with PyTorch 2.5.1 (CUDA 12.4), diffusers 0.31.0 and transformers 4.46.3.
PyTorch 2.5.1 has no wheels for the Python a new distribution ships (Ubuntu 26.04 has only 3.14), so setup does not
use the PC's Python: uv installs CPython 3.11.14 from the python-build-standalone builds into `.tools/python`. Those
builds link libedit rather than GNU readline and leave gdbm out, so no GPL code comes with them.

`python/requirements.txt` lists the packages the pipeline names, each with its licence; `python/requirements.lock`
pins all 60 that are installed, with the sha256 of each wheel, made with `uv pip compile --generate-hashes` and held
to the versions the pipeline was tried on. torch 2.5.1 from PyPI is the CUDA 12.4 build on Linux x86_64. Left out of
the environment the pipeline was tried in, as it runs nothing of theirs: the TRELLIS and spconv packages (cumm,
spconv, pccm, ccimport, xformers, ...), omegaconf.

## On a 12 GB card

Two changes let a 12 GB card make objects; neither changes what is made.

* **The picture model's two quantised parts** (FLUX.1 [schnell]'s transformer and its T5 text encoder, 4 bits each)
  are quantised on the card as they load. Loading the second beside the first ran out of memory on 12 GB (11.8 GB
  used). Each now goes to the computer's memory as soon as it is loaded; the pipeline's model offload brings it back
  while it works, as before (`painter.py`). The picture stage then needs 7.5 GB at most.
* **The view painter's UNet** (SDXL in fp16 with MV-Adapter's layers) was on the card whole; with the reference
  picture's hidden states for twelve images (six views, with and without guidance) it reached 11.6 GB before the
  first step and ran out. On a card under 15 GB its down, middle and up blocks now come onto the card one at a time
  from pinned memory and leave after (`paint.BlockOffload`, block-level offloading as later diffusers releases do it
  with their group offloading; diffusers 0.31 has none). The paint stage then needs 10.1 GB at most.
  `MODEL_MAKER_VIEW_OFFLOAD=1` or `0` overrides the choice. It is exact: SDXL's UNet run on the same inputs whole and
  offloaded gave the same output bit for bit (a batch of four at 768 pixels: 0.77 s and 5.25 GB whole, 0.96 s and
  2.59 GB offloaded); `python/tests/test_block_offload.py` checks the same on a small model when there is a card.

Nothing was given up: the same models at the same precision, the same steps, the same sizes.

What else was found on the clean PC:

* **setuptools** was missing: triton (which bitsandbytes imports when FLUX.1 [schnell] is quantised) imports it, and
  a venv made by uv has none (one made by `python3 -m venv` has). It is in the lock now, and `setup/smoke.py` imports
  every stage's code so such a gap shows in setup, not in the first make.
* The Python the PC has does not matter: Ubuntu 26.04 has only 3.14, which PyTorch 2.5.1 does not support.
* Another program holding part of a 12 GB card (on the PC tried, another program held 6 GB) must be stopped for a
  make: 6 + 10 GB does not fit.

## Tried on a clean PC


A PC with an RTX 3060 (12 GB), 24 threads, 45 GB of memory, Ubuntu 26.04 (Python 3.14 only, no Rust on the path,
driver CUDA 13.2), on a connection that downloaded from the hub at about 60 MB/s. A fresh `git clone`, then
`./setup.sh` from nothing, twice: first answering each question (piped in), then once more from a new clone with
`--yes`.

### setup.sh

| Step | First run (asked) | Second run (`--yes`) |
|---|---|---|
| the checks, uv and CPython 3.11.14 | a few seconds | a few seconds |
| the Python packages (5.7 GB installed, 60 packages) | 61 s | 107 s |
| rustup and the 1.99.0 toolchain into `.tools/rust` | under a minute | under a minute |
| `model-maker-simplify` (release build) | 20 s | 16 s |
| the models: 57 GB, 66 files, every sum checked | 998 s | 1002 s |
| self-test (the Python tests, the sums, every stage's imports, the card) | 76 s | 98 s (on another clean install, 648 s: about 2 to 11 minutes) |
| in all | 19 min 17 s | 20 min 29 s |

The first run lacked setuptools in the lock and the smoke import (both found by its first make, below, and added
since); the second ran with both, from a new clone, unattended.

A second `./setup.sh` in the same folder takes about as long as its self-test (everything else is found done), asks nothing
again and downloads nothing.

On disk afterwards: `models` 54 GB, `.venv` 5.7 GB, `.tools` 0.8 GB (uv, CPython, Rust, uv's cache; 0.16 GB on a PC
with its own Rust), `sources` 49 MB (MV-Adapter's code), `target` 31 MB.

### The chair and the sword on 12 GB

`./model-maker make "an old looking chair"` and `./model-maker make "an old iron sword" --size 0.9` in the second
run's clone, nothing else on the card; its memory sampled every 250 ms. Run again on the same card, each made the
same picture and the same paint bit for bit.

| Stage | Chair | Sword | Card at most | On a 16 GB card (chair, sword) |
|---|---|---|---|---|
| models checked against their sums and loaded | about 35 s | about 35 s | 6.4 GB | |
| pictures (four tries, about 21 s each) | 88 s | 88 s | 7.5 GB | 71 s, 67 s |
| shape (one seed for the chair; two for the sword, its first failed two checks) | 110 s | 179 s | 3.4 GB | 92 s, 57 s |
| paint (six views, 20 steps, the UNet's blocks taking turns) | 231 s | 221 s | 10.1 GB | 196 s, 192 s |
| finish (on the CPU) | 32 s | 10 s | 0.3 GB | 36 s, 16 s |
| the whole command | 9 min 8 s | 9 min 50 s | 10.1 GB | |

The computer's memory peaked at 32 GB resident while the picture model loaded (much of it the weights' files mapped
into memory).

Checks:

* **The sword passes every check**: outline 0.10 % of the height apart on average (limit 1.9 %), blade straight
  (0.4 mm), no kink (0.5°), grip found (18 cm long, 3.8 cm across), symmetric across the blade (0.4 % of its width),
  one piece, no holes, paint covers 100 % (likeness: mean colour difference 7.8). Its first shape seed failed two
  checks, so a second was made (as the pipeline does), which passed all seven.
* **The chair passes every check but one**: "paint covers the surface", 89.99 % against its limit of 90 %. Its
  picture, shape and base all pass (outline 0.64 % apart, base level 0.4°, centre 86 % inside its base, symmetric
  0.47 %). On the 16 GB card the same prompt passed it at 90.7 %: the picture model's sums come out a little
  different on another card, so the same seed paints a slightly different chair (the two `picture.png` differ),
  and this chair sits on the limit either way. It is not the offloading: the view painter's offloading is exact
  (above), and the chair's picture and shape were made before any view was painted.

### Open

* The chair on the limit: a 12 GB card is not why it fails, but a card is part of what a prompt makes. Whether the
  coverage limit, the views or the chair's picture should change is the pipeline's question, not setup's.
* Only Linux on x86_64 was tried. WSL2 should work the same (not tried). aarch64 has pinned uv and rustup sums, but
  PyTorch 2.5.1 from PyPI has no CUDA build for it, so the check stops there.
* The lock pins wheels for CPython 3.11 on Linux x86_64 only; a newer stack (PyTorch for Python 3.13 and later) would
  need the whole pipeline tried again.

## Warnings you can ignore

A make prints some warnings from the libraries it uses. These four are expected and change nothing:

* diffusers warning that fp16 does not run on the CPU, printed while the view painter's UNet blocks move off the card
  (on a card under 15 GB; nothing is computed on the CPU in fp16);
* accelerate's "You shouldn't move a model that is dispatched", printed when a model the pipeline offloads is moved;
* a note about `low_cpu_mem_usage` while a model loads;
* CLIP's note that the prompt was cut to 77 tokens: the picture model's T5 text encoder still reads the whole prompt.
  What is cut off is the end of the picture words ("no text, no signature"); letters the picture model paints anyway
  are caught by the check "no painted lettering" (with tesseract 5 or newer).

## What the wheels bundle

A package's own licence line is not enough: a wheel can bundle shared libraries under other licences (opencv's bundled
FFmpeg under the LGPL). `python/model_maker/check_licences.py` lists every shared library a package of the environment
bundles (its `<package>.libs/` folder, `torch/lib/`, ...; a package's own Python extension modules are its own code) and
compares each with an allow-list that names its licence; a library not on the list fails the check until its licence
is read and it is added (`--list` prints them all). On the locked environment it finds 93:

| Bundled by | Libraries | Licence |
|---|---|---|
| numpy, scipy | OpenBLAS | BSD-3-Clause |
| numpy, scipy | libgfortran | GPL-3.0 with the GCC Runtime Library Exception (shipping it with any program is what the exception allows) |
| numpy, scipy | **libquadmath** | **LGPL-2.1-or-later: the recorded exception** |
| torch, torchvision, triton, bitsandbytes | their own libraries | BSD-3-Clause, MIT |
| torch | libgomp | GPL-3.0 with the GCC Runtime Library Exception |
| nvidia-*-cu12, torch, torchvision | CUDA, cuBLAS, cuDNN, cuFFT, cuRAND, cuSOLVER, cuSPARSE(Lt), NVRTC, nvJitLink, NVTX, CUPTI, nvJPEG | NVIDIA's own licences: free to use and to ship with an application, not open source |
| nvidia-nccl-cu12 | NCCL | BSD-3-Clause |
| pillow, torchvision | libjpeg-turbo, libpng, zlib, libwebp, libtiff, OpenJPEG, libavif, Little CMS, FreeType, HarfBuzz, Brotli, zstd, xz, libxcb, libXau | IJG/BSD-3, libpng, Zlib, BSD-3, libtiff, BSD-2, BSD-2, MIT, FTL, MIT, MIT, BSD-3, 0BSD, MIT, MIT |

**libquadmath** (GCC's quad-precision maths) comes inside numpy's and scipy's own PyPI wheels, linked dynamically and
replaceable. The rule says no LGPL; this is its one recorded exception (CONTRIBUTING.md, rule 1): an LGPL library
inside a standard package's own wheel, loaded only at run time. NOTICE names it, and the check allows it by name. A
check of the packages' own licence lines alone would report 0 problems here.

CI's licence job installs exactly the lock (the CUDA build of torch and NVIDIA's libraries with it), as setup does, so
it checks the wheels users get.

