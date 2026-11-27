# Pillow 13, and the model stack held

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## Pillow 13, and the model stack held (13 November 2026)

CI's audit of the locked packages (`pip-audit`) found 105 known advisories. Pillow, which decodes every picture the
pipeline reads (also one a user hands to `model-maker inspect`), carried 40 of them; filelock, setuptools and certifi
a few more. The model stack (torch, transformers, diffusers, accelerate) carried the rest.

Raising the whole stack changed results: with the same seeds, 3 of 12 runs of six objects on two cards went from
passing to failing (the armchair on both cards, the kettle on the 12 GB one), as the shape stage's numbers drift with a
new torch. So this version raises only Pillow (12.3 to 13.0), filelock, setuptools and certifi, and keeps the model
stack at the versions its results were measured with. Pillow 13 was proven not to move a model's input: every input
Pillow makes (the decode, the cut-out's resizes, the judges' crop and image processor, the shape model's input, the
paint's reference, the touch-up's and the lettering reader's resizes) was hashed over 241 pictures under each version,
and all 2,410 are the same, byte for byte (`tools/trial/pillow_inputs.py`). The model stack's 50 advisories (72 with
their aliases) are listed in `tools/ci/pip-audit-ignore.toml`, each with why it does not reach model-maker (its models
are files whose sha256 the registry pins, loaded offline; nothing is served on a network) and a review date; the
stack moves in a version of its own, measured against a noise floor (two runs of the old environment first).

Then the whole pipeline: the six objects (an armchair, a chair, a kettle, a teapot, a bottle and a sword) made with
the same seeds under Pillow 12.3 and under 13 on the 12 GB card. Each chose the same picture seed, shape seed and
paint seed; the chosen pictures are identical to the pixel; and all 134 checks have the same verdict and the same
value. Both runs took 67 minutes.
