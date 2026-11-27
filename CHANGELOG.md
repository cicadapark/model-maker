# Changes

Each version's full note is on its release page (github.com/cicadapark/model-maker/releases); the measurements behind
each are in [`docs/results.md`](docs/results.md) and [`docs/objects.md`](docs/objects.md).

## 0.3.4

A memory fix for 12 GB cards; nothing that is made changes. The 12 GB card's view painter keeps the blocks that wait
for the card in page-locked buffers of exactly their size, not in PyTorch's pinned memory, which rounds every tensor up
to a power of two: building the painter takes 2.7 GiB less of the computer's memory. The same pictures and objects by
content (six objects, and 6 reference objects made again against 0.3.3's). The new path is not taken on 16 GB cards
(by code and test); the memory advice stays as it is. [`docs/results.md`](docs/results.md).

Also in this version: the code that starts and stops programs (the question judge's server) and reads the computer's
memory now also has Windows and macOS paths, to prepare those systems; nothing changes on Linux.

## 0.3.3

A security update: the Python libraries the models run on are raised past GitHub's dependency alerts. No check,
setting, model or step of the pipeline changed.

- PyTorch 2.5.1 (CUDA 12.4) to 2.10.0 (CUDA 12.8), torchvision 0.25.0, transformers 5.10.1, diffusers 0.38.0,
  accelerate 1.15.0, bitsandbytes 0.46.1 and what they pull along: 86 packages, every wheel's sha256 in the lock.
- Closes 30 of the 31 alerts GitHub listed (every critical, high and moderate one, and 5 of 6 low). Two advisories in
  torch 2.10.0 stay open, each with its reason in `tools/ci/pip-audit-ignore.toml`: one low, one high by its score
  (`.pt2` archives, which model-maker never loads); neither has a fixed CUDA 12 build.
- The shape model's and the state judge's pictures are prepared by model-maker's own steps: the same pixels with any
  transformers version.
- The same prompt and seed now give a slightly different object: the new PyTorch paints another picture from the same
  seed. Measured on 50 prompts on a 16 GB card, judged blind by two judges: another draw, not a worse one (no loss of
  8 or more objects, which is what 50 prompts can see). [`docs/results.md`, "0.3.3 against 0.3.2"](docs/results.md).
- Setup: run `./setup.sh` again to install the new environment (7.5 GB, was 6); keep 77 GB free.
- With a 12 GB card, 32 GB of the computer's memory is now advised (0.3.3 peaked at 25.5 GB there; less was not
  tried); with a 16 GB card 24 GB stays.

## 0.3.2

The texture bake colours every thin triangle, so the thin dark strokes on made objects are gone; `repair-texture`
mends a `.glb` made before.

## 0.3.1

Safer model folders (unlisted code and settings files refused, also with `--no-verify`), and a question judge whose
memory no longer grows over a batch.

## 0.3.0

Pairs (one object and its mirror image), things of today, and the fill check as a note shown in the report.

## 0.2.2

The question judge counted in the verdict; results on 50 new sealed prompts.

## 0.2.1

Content credentials on every picture, batch resume, upright weapons, views painted again; an honest results table.

## 0.2.0

Painted lettering, answers for animals and characters, a sealed pass rate.

## 0.1.0

A checked 3-D object from one line of text.
