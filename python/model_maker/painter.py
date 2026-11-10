"""The object's pictures painted by FLUX.1 [schnell] (Apache-2.0): several tries from consecutive seeds, each
checked (`picture_checks`), the best kept. Needs an NVIDIA card; the transformer and the second text encoder are
quantised to 4 bits so it fits a 12 GB card, and moved to the card only while they work.
"""

import numpy as np

SIDE = 1024
STEPS = 4  # FLUX.1 [schnell] is made for four steps and no guidance
MOST_TOKENS = 256


class Painter:
    def __init__(self, folder):
        import torch
        from diffusers import BitsAndBytesConfig as DiffusersBnb
        from diffusers import FluxPipeline, FluxTransformer2DModel
        from transformers import BitsAndBytesConfig as TransformersBnb
        from transformers import T5EncoderModel

        nf4 = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        # Each part is quantised on the card as it loads and sent to the computer's memory at once (the offload below
        # brings it back only while it works): on a 12 GB card the two do not fit on it together (the first clean
        # install on one ran out of memory loading the second). The same 4-bit weights, so the same pictures.
        transformer = FluxTransformer2DModel.from_pretrained(
            folder, subfolder="transformer", quantization_config=DiffusersBnb(**nf4), torch_dtype=torch.bfloat16)
        transformer.to("cpu")
        torch.cuda.empty_cache()
        encoder = T5EncoderModel.from_pretrained(
            folder, subfolder="text_encoder_2", quantization_config=TransformersBnb(**nf4), torch_dtype=torch.bfloat16)
        encoder.to("cpu")
        torch.cuda.empty_cache()
        self.pipe = FluxPipeline.from_pretrained(folder, transformer=transformer, text_encoder_2=encoder,
                                                 torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload()
        self.pipe.set_progress_bar_config(disable=True)

    def paint(self, prompt, seed):
        import torch

        return self.pipe(prompt=prompt, max_sequence_length=MOST_TOKENS, num_inference_steps=STEPS,
                         guidance_scale=0.0, height=SIDE, width=SIDE,
                         generator=torch.Generator("cpu").manual_seed(seed)).images[0]

    def close(self):
        import torch

        del self.pipe
        torch.cuda.empty_cache()


def seed_of(text):
    """A steady seed from the prompt, so the same prompt paints the same tries."""
    return sum((k + 1) * ord(c) for k, c in enumerate(text)) * 7919 % 1_000_000


def tries(painter, cutter, prompt, seeds, report=print):
    """Each seed's picture, its raw cut-out and its checks: [(seed, picture, mask, result)]."""
    from .picture_checks import checked

    made = []
    for seed in seeds:
        picture = painter.paint(prompt, seed)
        mask = cutter.mask(picture)
        result = checked(np.asarray(picture.convert("RGB")), mask)
        report(f"picture {seed}: {'passes' if result.passed else 'fails: ' + '; '.join(result.faults())} "
               f"(score {result.score:.3f})")
        made.append((seed, picture, mask, result))
    return made
