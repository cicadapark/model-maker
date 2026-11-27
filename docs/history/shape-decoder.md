# The shape model's own decoder

From model-maker's development: what was measured, what was fixed and what did not work, dated as it was
written. The reference is [`docs/objects.md`](../objects.md); the other topics are listed in
[`README.md`](README.md).

## The shape model's own decoder (10 November 2026)

TripoSG's weights are MIT, at the revision the registry pins. The folder its repository's code sits in, `triposg/`,
holds the text of the Tencent Hunyuan FlashVDM Community License, which excludes the European Union, the United
Kingdom and South Korea, so no code of that repository is fetched or run. model-maker runs the weights with its own
code only: `python/model_maker/shape_net.py` holds the denoiser, and `python/model_maker/shape_vae.py` the decoder from
the latent to the field the surface is found in, the attention processor and the point embedding (docs/setup.md).

**What it is, and what it follows.** The attention processor follows diffusers' `AttnProcessor2_0` (Apache-2.0), with
each head taking its own run of the joined projections' channels (query, key and value joined in self-attention; key
and value in cross-attention), the order the weights were trained with. The point embedding is the frequency encoding
the TripoSG paper describes: the point, then the sines and cosines of each coordinate times 1, 2, 4, ... 128 (8
frequencies, `vae/config.json`). The decoder is built from the weights' tensor names and shapes on model-maker's own
block and diffusers' `Attention`, `FeedForward` and `FP32LayerNorm`: the latent's 2048 tokens through `post_quant` and
16 self-attention blocks, the query points looking at them in one cross-attention block, a layer norm and `proj_out`.
Only `decode` was ever called (the narrow-band search asks the field at points), so the VAE's encoder is not built: its
tensors (`encoder.*`, `quant.*`) are the only ones of the file not loaded, and a missing or left-over tensor is refused.
The FlashVDM decoder is neither written again nor used.

**How what the weights do not say was settled.** The upstream code was run as a black box, before it was removed
from setup: the arguments its VAE passed to our block were recorded (layer norms with eps 1e-5, no query and key norms, a layer norm on
the cross-attention's input), and its outputs compared with ours on the same inputs. Two things were found that way:
the heads' order (with diffusers' own order the output differed by up to 9.95 where it reached 6.2, on random inputs; with the joined order by 0), and the
field's sign (the old decoder gives the negative of `proj_out`, so the field is positive inside).

**Measured on the CPU**, the old modules against ours, each on the same inputs:

| What | Inputs | Largest difference |
|---|---|---|
| attention processor | random inputs and weights; self- and cross-attention; widths 2048 (16 heads), 1024 (8), 256 (4); no query and key norm, RMS norm, layer norm without bias; 32 and 16 bits (36 cases) | 0 in every case |
| the denoiser, one call | the chair's picture features, noise of seed 42, time 500, every weight real | 0 |
| point embedding | 60,000 points in the shape's cube | 0 |
| decoder's field, 32 bits | random latents (seed 42) and the chair's real latents (below); 60,000 points each | 0, its sign the same at every point |
| decoder's field, 16 bits (as on the card) | the same latents, 4,096 points each | 0 |
| the field with the latent's tokens worked out once (the new `surface`) | 20,000 of those points | 0 |
| a whole shape | the chair's real latents, the narrow-band search as the pipeline runs it (a coarse grid of 256, one finer level: 513) | identical: 296,942 corners and 593,904 triangles in both, every corner and triangle the same (largest corner distance 0); 1,391 s the old way, 1,037 s the new on 8 CPU threads |
| a whole shape, coarser | the same latents, a coarse grid of 128 and one finer level (257) | identical: 73,838 corners and 147,696 triangles, every one the same; 149 s and 110 s on 12 threads |

The real latents are the regression set's `chair-good`: its picture and cut-out through our denoiser on the CPU (32 bits,
seed 42, 50 steps, guidance 7; 830 s). The field is the same to the last bit, so every shape the pipeline made with the
old code it makes again with this code. The new `surface` also works out the latent's tokens (its 16 self-attention
blocks) once per shape rather than once per chunk of points, which gives the same field and saves that work on every
chunk (a quarter less time for the whole shape on the CPU). A full make on the card was not needed to prove the
equivalence. The regression set holds all its 41 cases with the new code (it judges stored shapes, so it checks that
nothing else moved), and the Python tests pass.
