"""Whether a picture shows the state its words ask for: a lid shut, a book closed, doors open (words.state_texts).

On the third trial (2026-11-10) "a wooden treasure chest" was painted with its lid up in one try of four and "a
leather-bound book, closed" open in two; the open ones scored best on the other picture checks and were chosen, and
the made objects failed only by the side effects (the chest's paint coverage, the book's base). The established way
to check a picture against its words is zero-shot image-text scoring with a contrast pair: the picture is compared
with words for the state asked for and words for the other ("a closed book" against "an open book"), and the state
asked for must be the likelier of the two (CLIP's zero-shot classification; image-text scores for picture-model output such as
CLIPScore). Each state is described by what it looks like, in pairs that name the same evidence both ways ("a closed
book, the edges of its pages in one even block" against "an open book, the edges of its pages split in two at the
middle"; classification by description, Menon and Vondrick 2023, and CuPL), and each pair gives the share for the
state asked for (a sigmoid of the difference of their scores). The picture is seen twice, whole and cropped square
round the cut-out (the multi-crop averaging of zero-shot evaluation): whole, a small chest with its lid up scored 0.61
for "closed", cropped 0.22. The shares of both views and every pair are averaged.

The model is SigLIP 2 (google/siglip2-so400m-patch14-384, Apache-2.0, trained by Google on its WebLI data). The first
SigLIP (also Apache-2.0) was tried first and could not tell the trial's open books from closed ones: it gave them
0.67 to 0.97 for "closed". A share near one half is no verdict: moving the crop by one pixel moved an open book from
0.52 to 0.58. So a picture must show its state clearly, the state asked for at least three times as likely as the
other (a share of 0.75); one in doubt fails safe, and another try is painted.

The first wordings named the state only ("a closed book", "a photo of a closed book", "a book lying shut"). Such
wordings score the noun's usual look more than its state (contrastive models recognise object states far worse than
objects: Newman et al. 2024), so a book painted open with leather-coloured halves, no white page in sight, scored 0.88
"closed" and was shaped in two lobes. With the descriptions, on the 52 book and chest pictures labelled by eye
(tools/trial/state-cases.toml): every shut one passes (books 0.91 and over, chests 0.88 and over), and 16 of the 17
open ones fail (0.41 or less; the leather-coloured open book 0.61). The one it misses is a book standing open in a V
on its spine (0.81; 0.90 before), as before.
"""

import numpy as np

from .picture_checks import Check

MODEL_FOLDER = "siglip2-so400m-patch14-384"
NAME = "shows what the words ask"
LEAST_STATE_SHARE = 0.75
# The text model reads at most this many tokens, padded to it as it was trained; its words were lowercased.
TEXT_TOKENS = 64
# The second view: the picture cropped to a square round the cut-out, this share of its size added each side.
CROP_MARGIN = 0.15


class StateJudge:
    """SigLIP 2, loaded once from `folder` (checked against the registry by the caller). Its text side is read with
    the Gemma tokenizer it ships, so the pinned transformers runs it (its own processor class came later)."""

    def __init__(self, folder, device=None):
        import torch
        from transformers import GemmaTokenizerFast, SiglipImageProcessor, SiglipModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 32-bit floats on the card too, as the regression judges on the CPU: in 16 bits a picture near the limit
        # could be judged one way when made and the other way in the regression. The
        # model is 1.6 GB in 32 bits and judges a few pictures a job.
        self.dtype = torch.float32
        self.model = SiglipModel.from_pretrained(folder, torch_dtype=self.dtype).to(self.device).eval()
        self.tokenizer = GemmaTokenizerFast.from_pretrained(folder)
        self.images = SiglipImageProcessor.from_pretrained(folder)

    def features(self, image=None, texts=None):
        import torch

        with torch.no_grad():
            if image is not None:
                pixels = self.images(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
                found = self.model.get_image_features(pixel_values=pixels.to(self.device, self.dtype))
            else:
                tokens = self.tokenizer([t.lower() for t in texts], padding="max_length", max_length=TEXT_TOKENS,
                                        truncation=True, return_tensors="pt")
                found = self.model.get_text_features(input_ids=tokens["input_ids"].to(self.device))
        found = found.float().cpu().numpy()
        return found / np.linalg.norm(found, axis=-1, keepdims=True)

    def share(self, picture, mask, wanted, unwanted):
        """The share for the state asked for (0 to 1), from the whole picture and its crop round the cut-out (`mask`,
        None for the whole picture only): `wanted` and `unwanted` are wordings of the two states, paired by place."""
        scale = float(self.model.logit_scale.exp().item())
        wanted, unwanted = self.features(texts=wanted), self.features(texts=unwanted)
        views = [picture] + ([cropped(picture, mask)] if mask is not None and mask.any() else [])
        return float(np.mean([asked_share(self.features(image=view)[0], wanted, unwanted, scale) for view in views]))

    def close(self):
        import torch

        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()


def cropped(picture, mask):
    """The picture cropped to a square round the cut-out, `CROP_MARGIN` of its size added each side (black beyond
    the picture's edge, as the trial probe cropped)."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    middle_row, middle_col = (rows[0] + rows[-1]) / 2, (cols[0] + cols[-1]) / 2
    side = max(rows[-1] - rows[0], cols[-1] - cols[0]) * (1 + 2 * CROP_MARGIN)
    left, top = int(middle_col - side / 2), int(middle_row - side / 2)
    return picture.convert("RGB").crop((left, top, left + int(side), top + int(side)))


def asked_share(picture, wanted, unwanted, scale):
    """The mean over pairs of wordings of the sigmoid of their score difference (unit vectors; `scale` the model's
    logit scale; its bias is the same for both and cancels)."""
    difference = scale * (wanted @ picture - unwanted @ picture)
    return float(np.mean(1.0 / (1.0 + np.exp(-difference))))


def add_state_check(result, subject, share):
    """The state check added to a picture's checks (`picture_checks.PictureResult`) when its words ask for a state."""
    if not subject.state:
        return
    from .words import OPENING_PARTS, STATE_PICTURE_WORDS

    asked = STATE_PICTURE_WORDS[OPENING_PARTS[subject.noun]][subject.state]
    result.checks.append(Check(
        NAME, share * 100, LEAST_STATE_SHARE * 100, "%", share >= LEAST_STATE_SHARE,
        f"the picture shows the {subject.noun} {subject.state} ({asked}) with a share of {share:.0%} against the other "
        f"state (at least {LEAST_STATE_SHARE:.0%}; judged by SigLIP 2)"))


def check_from_record(result, name=NAME):
    """A SigLIP 2 check (the state's, or another `name`) from a try's record in job.json (`PictureResult.as_dict`), or
    None when it has none."""
    for found in result.get("checks", []):
        if found["name"] == name:
            return Check(found["name"], found["value"], found["limit"], found["unit"], found["passed"],
                         found["words"], found.get("counted", True))
    return None
