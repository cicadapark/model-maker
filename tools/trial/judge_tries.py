#!/usr/bin/env python3
"""Every picture under the given folders judged against its own prompt on the CPU, as the pipeline's choice of try
judges it ("shows the thing asked for", subject_judge), one copy of each picture (by its sha256). Also the head pair
considered as a second layer ("a hammer head with a flat blunt striking face" against "an axe head with a sharp curved
blade"), for every picture, to see whether it is needed and where its limit would lie (docs/objects.md, "War
hammers").

    judge_tries.py OUT.json FOLDER...

A picture is found as a job's tries (`<job>/pictures/<seed>.png` with its `-mask.png`, the prompt from `job.json`), a
job's chosen `picture.png` and `mask.png`, or a `tries.json` of tools/trial/paint_tries.py. SigLIP 2 from
MODEL_MAKER_MODELS, checked against the registry first.
"""

import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

HEAD_PAIR = ("a hammer head with a flat blunt striking face", "an axe head with a sharp curved blade")


def found_pictures(folders):
    """[(prompt, picture path, mask path, where it came from)] under the folders."""
    found = []
    for folder in folders:
        for record_path in sorted(glob.glob(os.path.join(folder, "**", "job.json"), recursive=True)):
            job = os.path.dirname(record_path)
            with open(record_path) as file:
                prompt = json.load(file)["prompt"]
            for picture in sorted(glob.glob(os.path.join(job, "pictures", "*.png"))):
                if not picture.endswith("-mask.png"):
                    found.append((prompt, picture, picture[:-4] + "-mask.png", picture))
            if os.path.exists(os.path.join(job, "picture.png")):
                found.append((prompt, os.path.join(job, "picture.png"), os.path.join(job, "mask.png"),
                              os.path.join(job, "picture.png")))
        for record_path in sorted(glob.glob(os.path.join(folder, "**", "tries.json"), recursive=True)):
            with open(record_path) as file:
                for entry in json.load(file):
                    picture = os.path.join(os.path.dirname(record_path), entry["picture"])
                    found.append((entry["prompt"], picture, picture[:-4] + "-mask.png", picture))
    return [f for f in found if os.path.exists(f[1]) and os.path.exists(f[2])]


def main():
    out, folders = sys.argv[1], sys.argv[2:]
    import numpy as np
    from PIL import Image

    from model_maker import pipeline, state_judge, subject_judge, words

    settings = pipeline.Settings()
    judge = state_judge.StateJudge(settings.verified(pipeline.STATE_MODEL, pipeline.STATE_MODEL)[0], device="cpu")
    scale = float(judge.model.logit_scale.exp().item())
    hammer, axe = judge.features(texts=[HEAD_PAIR[0]]), judge.features(texts=[HEAD_PAIR[1]])
    directions, judged, seen = {}, [], {}
    for prompt, picture_path, mask_path, source in found_pictures(folders):
        with open(picture_path, "rb") as file:
            sha = hashlib.sha256(file.read()).hexdigest()
        if (sha, prompt) in seen:
            seen[(sha, prompt)]["sources"].append(source)
            continue
        if prompt not in directions:
            directions[prompt] = subject_judge.contrast_directions(judge, *words.subject_contrast(
                words.subject_of(prompt)))
        picture = Image.open(picture_path).convert("RGB")
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        # As subject_judge.judged judges it, each view's features worked out once for both measures.
        views = [picture] + ([state_judge.cropped(picture, mask)] if mask.any() else [])
        features = [judge.features(image=view)[0] for view in views]
        asked, others, names = directions[prompt]
        found = [subject_judge.subject_share(f, asked, others, scale) for f in features]
        share, likeliest = float(np.mean([s for s, _ in found])), names[min(found)[1]]
        head = float(np.mean([state_judge.asked_share(f, hammer, axe, scale) for f in features]))
        entry = {"sha256": sha, "prompt": prompt, "picture": picture_path, "mask": mask_path, "sources": [source],
                 "subject_share": round(share, 4), "likeliest_other": likeliest, "hammer_head_share": round(head, 4)}
        seen[(sha, prompt)] = entry
        judged.append(entry)
        print(f"{share:.3f} {head:.3f} {prompt!r} {source}", flush=True)
    with open(out, "w") as file:
        json.dump(judged, file, indent=1)
    print(f"END-OF-JUDGING {len(judged)} pictures", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
