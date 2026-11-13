#!/usr/bin/env python3
"""Pictures of hafted and long things judged by SigLIP 2 on the CPU (docs/trials/2026-11-11-030-diagnosis.md).

    judge_hafted.py OUT.json FOLDER...

Each picture (found as judge_tries.py finds them) whose prompt names a hafted or long thing is judged as the pipeline
judges "shows the thing asked for", with the kinds as they are (words.SUBJECT_KINDS) and with two changes considered: a
club its own kind, apart from the mace; and a sceptre among the kinds. And a "short" quality's pair, judged whatever
the prompt says (each picture's share for "short" against "long"). One row a picture."""
import json, os, sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
sys.path.insert(0, HERE)
from judge_tries import found_pictures
from model_maker import pipeline, state_judge, subject_judge, words

WANTED = ("mace", "club", "flail", "morning star", "staff", "quarterstaff", "spear", "sword", "longsword", "dagger",
          "axe", "hammer", "war hammer", "hatchet", "mallet", "maul", "rod", "sceptre", "scepter", "pick", "pickaxe")
SHORT_PAIRS = [("a short {noun}, about as long as a forearm", "a long {noun}, as long as a person is tall"),
               ("a short stubby {noun}, its head big and heavy for its short handle",
                "a long thin {noun}, a small head on the end of a long pole")]
BASE = words.SUBJECT_KINDS
CLUB_ROW = [((("mace", "flail", "morning star"), "a mace, a heavy spiked or flanged head on a short handle"),
             (("club", "cudgel"), "a club, a short thick wooden cudgel, much heavier at its top"))]
SCEPTRE = (("sceptre", "scepter"), "a sceptre, a long thin ornate rod with a small knob at its top")

def kinds_variant(name):
    rows = list(BASE)
    if name in ("club", "club+sceptre"):
        i = next(k for k, (nouns, _) in enumerate(rows) if "mace" in nouns)
        rows[i:i + 1] = CLUB_ROW[0]
    if name in ("sceptre", "club+sceptre"):
        rows.append(SCEPTRE)
    return tuple(rows)

def main():
    out, folders = sys.argv[1], sys.argv[2:]
    settings = pipeline.Settings()
    judge = state_judge.StateJudge(settings.verified(pipeline.STATE_MODEL, pipeline.STATE_MODEL)[0], device="cpu")
    scale = float(judge.model.logit_scale.exp().item())
    variants = ("as is", "club", "sceptre", "club+sceptre")
    directions, rows, seen = {}, [], set()
    for prompt, picture_path, mask_path, source in found_pictures(folders):
        subject = words.subject_of(prompt)
        if not any(words._found(prompt.lower(), w) for w in WANTED):
            continue
        key = (open(picture_path, "rb").read().__hash__(), prompt)
        if key in seen:
            continue
        seen.add(key)
        if prompt not in directions:
            directions[prompt] = {}
            for v in variants:
                words.SUBJECT_KINDS = kinds_variant(v)
                directions[prompt][v] = subject_judge.directions_of(judge, subject)
            words.SUBJECT_KINDS = BASE
            noun = subject.noun or words.UNKNOWN_THING
            directions[prompt]["short"] = [(judge.features(texts=[a.format(noun=noun)]),
                                            judge.features(texts=[b.format(noun=noun)])) for a, b in SHORT_PAIRS]
        picture = Image.open(picture_path).convert("RGB")
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        views = [picture] + ([state_judge.cropped(picture, mask)] if mask.any() else [])
        feats = [judge.features(image=view)[0] for view in views]
        row = {"prompt": prompt, "picture": picture_path}
        for v in variants:
            share, likeliest, _ = subject_judge.averaged(
                [subject_judge.view_shares(f, directions[prompt][v], scale) for f in feats], directions[prompt][v][2])
            row[v] = [round(share, 4), likeliest]
        row["short"] = round(float(np.mean([state_judge.asked_share(f, a, b, scale) for f in feats
                                            for a, b in directions[prompt]["short"]])), 4)
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(rows, open(out, "w"), indent=0)
    print("END", len(rows))

if __name__ == "__main__":
    main()
