"""Whether a picture shows the thing its prompt asks for: a chair for "an old looking chair", not a table, a vase or a
person (words.subject_contrast).

A model made on another computer comes with the picture it was made from; a picture swapped for another, or one in
which the picture model painted something else, must be caught before the model is shown. In the pipeline the same
check judges every picture try. The established way is zero-shot image-text classification (CLIP's, and SigLIP's):
the picture is compared with words for the thing asked for and with words for each other thing it might show
instead, and the thing asked for must be the likeliest. The others are every other kind of thing release 1 makes,
each by a description of its look with its name first (classification by description: Menon and Vondrick 2023,
CuPL), and a few things a picture might hold instead of one object: a person, an animal, text, an empty backdrop,
several things.
The thing asked for is the prompt itself ("a photo of an old looking chair") and its kind's description; the words
of each class are averaged into one direction (CLIP's prompt ensembling).

The share is the sigmoid of the score difference between the thing asked for and the likeliest other, as the state
judge's share is for a pair (state_judge.asked_share): 0.75 means the thing asked for is three times as likely as
the best other. It is averaged over the two views the state judge uses, the whole picture and its crop round the
cut-out. The limit comes from every picture the trials painted, each scored against its own prompt and against every
other prompt of the trials (docs/objects.md, "The picture against the prompt"; tools/trial/prompt-cases.toml).

The model is the state judge's: SigLIP 2 (google/siglip2-so400m-patch14-384, Apache-2.0), which runs on the CPU.
"""

import numpy as np

from .picture_checks import Check
from .state_judge import cropped

NAME = "shows the thing asked for"
LEAST_SUBJECT_SHARE = 0.75


def class_directions(judge, texts_by_class):
    """Each class's words embedded and averaged into one unit direction (a row per class, in the dict's order)."""
    rows = []
    for texts in texts_by_class.values():
        found = judge.features(texts=texts).mean(axis=0)
        rows.append(found / np.linalg.norm(found))
    return np.array(rows)


def subject_share(picture_features, asked, others, scale):
    """The share for the thing asked for against the likeliest other (unit vectors; `asked` one direction, `others`
    a row each; `scale` the model's logit scale; its bias is the same for every class and cancels), and which other
    was likeliest (its row)."""
    against = others @ picture_features
    likeliest = int(np.argmax(against))
    difference = scale * (asked @ picture_features - against[likeliest])
    return float(1.0 / (1.0 + np.exp(-difference))), likeliest


def contrast_directions(judge, asked, others):
    """The thing asked for and each other kind as directions (words.subject_contrast's), worked out once a prompt:
    (its direction, the others' rows, the others' names)."""
    return class_directions(judge, {"asked": asked})[0], class_directions(judge, others), list(others)


def judged(judge, picture, mask, directions):
    """The share for the thing asked for (0 to 1), averaged over the whole picture and its crop round the cut-out
    (`mask`, None for the whole picture only), and the likeliest other kind (in the view where it came nearest).
    `directions` from `contrast_directions`."""
    asked, others, names = directions
    scale = float(judge.model.logit_scale.exp().item())
    views = [picture] + ([cropped(picture, mask)] if mask is not None and mask.any() else [])
    found = [subject_share(judge.features(image=view)[0], asked, others, scale) for view in views]
    nearest = min(found)[1]
    return float(np.mean([share for share, _ in found])), names[nearest]


def subject_check(subject, share, likeliest_other):
    """The check as the picture's checks list it (`picture_checks.Check`)."""
    return Check(
        NAME, share * 100, LEAST_SUBJECT_SHARE * 100, "%", share >= LEAST_SUBJECT_SHARE,
        f"the picture shows {subject.prompt} with a share of {share:.0%} against the likeliest other thing, "
        f"{likeliest_other} (at least {LEAST_SUBJECT_SHARE:.0%}; judged by SigLIP 2)")


def add_subject_check(result, subject, share, likeliest_other):
    """The check added to a try's checks (`picture_checks.PictureResult`): in the pipeline's choice of try, so a try
    it refuses is not chosen, as with the other picture checks (docs/objects.md, "War hammers")."""
    result.checks.append(subject_check(subject, share, likeliest_other))


def picture_check(picture_path, prompt, out, mask_path=None):
    """A picture made elsewhere checked against its job's prompt (`model-maker picture-check`): the thing
    asked for and, for a thing with a lid, cover or doors, its state, both judged by SigLIP 2 on the CPU (its folder
    under MODEL_MAKER_MODELS, checked against the registry first). `mask_path` is the cut-out sent, for the crop: run
    it after that cut-out has been held to our own (inspect --recut). Writes
    `out/picture-check.json` and returns its record."""
    import json
    import os

    from PIL import Image

    from . import registry, state_judge, words

    subject = words.subject_of(prompt)
    picture = Image.open(picture_path).convert("RGB")
    mask = (np.asarray(Image.open(mask_path).convert("L")) > 127) if mask_path else None
    judge = state_judge.StateJudge(registry.model_folder("siglip2-so400m", state_judge.MODEL_FOLDER), device="cpu")
    share, likeliest = judged(judge, picture, mask, contrast_directions(judge, *words.subject_contrast(subject)))
    checks = [subject_check(subject, share, likeliest)]
    if subject.state:
        from .picture_checks import PictureResult

        result = PictureResult()
        wanted, unwanted = words.state_texts(subject)
        state_judge.add_state_check(result, subject, judge.share(picture, mask, wanted, unwanted))
        checks += result.checks
    failed = [c.name for c in checks if not c.passed]
    record = {"prompt": subject.prompt, "status": "fails: " + "; ".join(failed) if failed else "passes every check",
              "checks": [c.as_dict() for c in checks], "likeliest_other": likeliest}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "picture-check.json"), "w") as file:
        json.dump(record, file, indent=1)
    return record
