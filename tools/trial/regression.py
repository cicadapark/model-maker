#!/usr/bin/env python3
"""The regression set (`regression.toml`): what the checks have caught, each case with the verdicts it must keep.

    python3 tools/trial/regression.py build TRIALS DATA     copy each case's files from the trial folders under TRIALS
    python3 tools/trial/regression.py run DATA [--only NAME] judge each case again (CPU); exit 1 if any verdict changed
    python3 tools/trial/regression.py sums DATA              once a case is added: every data file's sum into the
                                                             repository (regression-sums.json), checked by `run`

Run it before any change to the pipeline or its checks. A case is judged as the finish stage judges a made object:
the shape cleaned up and (a thing that stands) levelled, then every shape check, the picture's checks, and the
paint's coverage when the case asks (`fill = true`: the touch-up's fill against the paint round it), and (`state = true`) the state judge's verdict on the picture (its model from
MODEL_MAKER_MODELS). `made_must_fail` and `made_must_pass` judge the shape as the model made it, before
the finish's levelling (the checks must still catch what the model got wrong); `trimmed_mm` = [low, high] is how much
the levelling must cut from the feet; `laid_deg` = [low, high] how far a thing that lies must be turned to lie on its
broad face; `carved_holes` how many holes of the picture's outline the finish must carve a web from (see carving);
`grip_from_butt_cm` = [low, high] where a hafted thing's grip must be held, from its butt; `handles` how many handles a
shield's back must be found to have, its grip on one of them.
A case with `picture_only = true` has only its picture and cut-out, and is judged by the picture's checks alone.

The state judge's labelled pictures (`state-cases.toml`) are built into DATA/states and judged as one more case,
"state-pictures": each must pass or fail "shows what the words ask" as its label says. The picture-against-prompt
judge's (`prompt-cases.toml`) are built into DATA/prompts and judged as "prompt-pictures": each must pass "shows the
thing asked for" against its own prompt and fail it against every other prompt there, and a made-up picture of each
prompt, an empty backdrop, must fail it (its words written out, a known miss, are shown, not held). The war hammer
family's and the axes' pictures, their heads labelled by eye (`head-cases.toml`), are built into DATA/heads and judged
as "head-pictures": a clear head of the prompt's own kind must pass the check as the pipeline's choice of try judges
it, one of the other kind must fail it, and a head between the two is shown. The pictures painted with a quality and
without it, labelled by eye (`quality-cases.toml`), are built into DATA/qualities and judged as "quality-pictures":
judged against a prompt that names the quality, the quality's pair must pass a picture showing it and refuse one
showing its opposite (the whole check "shows the thing asked for" is shown beside it), and one in doubt is shown.
The picture tries labelled by eye for painted lettering (`lettering-cases.toml`) are built into DATA/lettering and
judged as "lettering-pictures": "no painted lettering" must refuse a try showing letters its prompt does not ask for
(but a known miss, `known_miss`, is shown), pass one showing none, and pass every try whose prompt asks for lettering;
one in doubt is shown. It needs tesseract 5 or newer with its English data: without a usable one the case is not
judged, and counts as broken (it never passes unread); the version that read it is printed and kept in the ledger.
"""

import argparse
import json
import os
import shutil
import sys
import time
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
CASES = os.path.join(HERE, "regression.toml")
# Each data case's files and their sha256, written by `sums` once the set is built, checked by `run`: the set's data
# lives outside the repository (it is not published), and these show it was never edited.
SUMS = os.path.join(HERE, "regression-sums.json")
STATE_CASES = os.path.join(HERE, "state-cases.toml")
PROMPT_CASES = os.path.join(HERE, "prompt-cases.toml")
HEAD_CASES = os.path.join(HERE, "head-cases.toml")
QUALITY_CASES = os.path.join(HERE, "quality-cases.toml")
LETTERING_CASES = os.path.join(HERE, "lettering-cases.toml")
FILES = ("job.json", "picture.png", "mask.png", "shape.npz")
PICTURE_FILES = ("picture.png", "mask.png")
STATE_PICTURES = "state-pictures"
PROMPT_PICTURES = "prompt-pictures"
HEAD_PICTURES = "head-pictures"
QUALITY_PICTURES = "quality-pictures"
LETTERING_PICTURES = "lettering-pictures"


def cases():
    with open(CASES, "rb") as file:
        return tomllib.load(file)["case"]


def state_pictures():
    with open(STATE_CASES, "rb") as file:
        return tomllib.load(file)["picture"]


def prompt_pictures():
    with open(PROMPT_CASES, "rb") as file:
        return tomllib.load(file)["picture"]


def head_pictures():
    with open(HEAD_CASES, "rb") as file:
        return tomllib.load(file)["picture"]


def quality_pictures():
    with open(QUALITY_CASES, "rb") as file:
        return tomllib.load(file)["picture"]


def lettering_pictures():
    with open(LETTERING_CASES, "rb") as file:
        return tomllib.load(file)["picture"]


def made_up(case, target):
    """A made-up case's files (its shape from `python/tests/shapes.py`, its picture and cut-out drawn from that shape
    at the view the case names, a grey object on a pale backdrop; its job.json): a fault with a known answer, built
    by code, as the checks were measured."""
    import json as records

    import numpy as np
    from PIL import Image

    from model_maker import object_checks, render, words
    from tests import shapes

    spec = case["made_up"]
    make = getattr(shapes, spec["shape"])

    def shape_of(args):
        corners, triangles = make(**args)
        if spec.get("turned_deg"):
            t = np.radians(spec["turned_deg"])
            corners = corners @ np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]]).T
        return object_checks.set_down(corners, float(np.ptp(corners, axis=0).max())), triangles

    corners, triangles = shape_of(spec.get("args", {}))
    subject = words.subject_of(case["prompt"])
    size = float(np.ptp(corners, axis=0).max())
    turn, tilt = spec.get("view", (20, 12))

    def drawn(args):
        return render.silhouette(*shape_of(args), turn, 1024, tilt, spec.get("distance"))

    # The picture is drawn from `picture_args` (the object as it should be: a ring open where the model has a web),
    # its cut-out from `mask_args` (a cut-out that missed a part: what it left out is painted in another colour).
    shown = drawn(spec.get("picture_args", spec.get("args", {})))
    mask = drawn(spec["mask_args"]) if "mask_args" in spec else shown
    picture = np.where(shown[..., None], 110, 214).astype(np.uint8).repeat(3, -1)
    picture[shown & ~mask] = (60, 90, 160)
    os.makedirs(target, exist_ok=True)
    Image.fromarray(picture).save(os.path.join(target, "picture.png"))
    Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(target, "mask.png"))
    np.savez_compressed(os.path.join(target, "shape.npz"), corners=corners, triangles=triangles)
    with open(os.path.join(target, "job.json"), "w") as file:
        records.dump({"prompt": subject.prompt, "slug": subject.slug, "subject": subject.__dict__, "size_m": size,
                      "made_up": spec}, file, indent=1)


def from_example(case, target):
    """A case built from an example kept in the repository's history (`example`: "<commit>:<folder>", as git names it):
    its picture, the picture cut by our own BiRefNet on the CPU (MODEL_MAKER_MODELS), and its model.glb read as the
    shape the model made, at the size its sidecar says."""
    import json as records
    import subprocess

    import numpy as np
    from PIL import Image

    from model_maker import inspect_mesh, object_checks, words

    spec = case["example"]
    os.makedirs(target, exist_ok=True)
    for name in ("picture.png", "model.glb", "provenance.json"):
        made = subprocess.run(["git", "-C", os.path.join(HERE, "..", ".."), "show", f"{spec}/{name}"],
                              capture_output=True, check=True)
        with open(os.path.join(target, name), "wb") as file:
            file.write(made.stdout)
    with open(os.path.join(target, "provenance.json")) as file:
        size = float(records.load(file)["size_m"])
    picture = Image.open(os.path.join(target, "picture.png")).convert("RGB")
    mask = inspect_mesh.our_cut(picture)
    Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(target, "mask.png"))
    corners, triangles = inspect_mesh.loaded_mesh(os.path.join(target, "model.glb"))
    corners = object_checks.set_down(corners, size)
    np.savez_compressed(os.path.join(target, "shape.npz"), corners=corners, triangles=triangles)
    subject = words.subject_of(case["prompt"])
    with open(os.path.join(target, "job.json"), "w") as file:
        records.dump({"prompt": subject.prompt, "slug": subject.slug, "subject": subject.__dict__, "size_m": size,
                      "example": spec}, file, indent=1)


def build(trials, data):
    for case in cases():
        if case.get("example"):
            from_example(case, os.path.join(data, case["name"]))
            print(f"{case['name']}: from {case['example']}")
            continue
        if case.get("made_up"):
            made_up(case, os.path.join(data, case["name"]))
            print(f"{case['name']}: made up ({case['made_up']['shape']})")
            continue
        source = os.path.join(trials, case["source"])
        target = os.path.join(data, case["name"])
        os.makedirs(target, exist_ok=True)
        files = PICTURE_FILES if case.get("picture_only") else FILES
        for name in files + (("painted.npz",) if case.get("paint") or case.get("fill") else ()):
            shutil.copyfile(os.path.join(source, name), os.path.join(target, name))
        print(f"{case['name']}: from {case['source']}")
    states = os.path.join(data, "states")
    os.makedirs(states, exist_ok=True)
    for picture in state_pictures():
        source = os.path.join(trials, picture["source"])
        shutil.copyfile(source, os.path.join(states, picture["name"] + ".png"))
        shutil.copyfile(source[:-4] + "-mask.png", os.path.join(states, picture["name"] + "-mask.png"))
    print(f"{STATE_PICTURES}: {len(state_pictures())} pictures")
    prompts = os.path.join(data, "prompts")
    os.makedirs(prompts, exist_ok=True)
    for picture in prompt_pictures():
        shutil.copyfile(os.path.join(trials, picture["source"]), os.path.join(prompts, picture["name"] + ".png"))
        shutil.copyfile(os.path.join(trials, picture["mask"]), os.path.join(prompts, picture["name"] + "-mask.png"))
    print(f"{PROMPT_PICTURES}: {len(prompt_pictures())} pictures")
    heads = os.path.join(data, "heads")
    os.makedirs(heads, exist_ok=True)
    for picture in head_pictures():
        source = os.path.join(trials, picture["source"])
        shutil.copyfile(source, os.path.join(heads, picture["name"] + ".png"))
        shutil.copyfile(source[:-4] + "-mask.png", os.path.join(heads, picture["name"] + "-mask.png"))
    print(f"{HEAD_PICTURES}: {len(head_pictures())} pictures")
    qualities = os.path.join(data, "qualities")
    os.makedirs(qualities, exist_ok=True)
    for picture in quality_pictures():
        source = os.path.join(trials, picture["source"])
        shutil.copyfile(source, os.path.join(qualities, picture["name"] + ".png"))
        shutil.copyfile(source[:-4] + "-mask.png", os.path.join(qualities, picture["name"] + "-mask.png"))
    print(f"{QUALITY_PICTURES}: {len(quality_pictures())} pictures")
    lettered = os.path.join(data, "lettering")
    os.makedirs(lettered, exist_ok=True)
    for picture in lettering_pictures():
        source = os.path.join(trials, picture["source"])
        shutil.copyfile(source, os.path.join(lettered, picture["name"] + ".png"))
        shutil.copyfile(source[:-4] + "-mask.png", os.path.join(lettered, picture["name"] + "-mask.png"))
    print(f"{LETTERING_PICTURES}: {len(lettering_pictures())} pictures")


_JUDGE = []


def state_judge_of():
    """The state judge, loaded once, from MODEL_MAKER_MODELS (as the pipeline finds its models)."""
    if not _JUDGE:
        from model_maker import state_judge

        folder = os.path.join(os.environ.get("MODEL_MAKER_MODELS", "models"), state_judge.MODEL_FOLDER)
        if not os.path.exists(folder):
            raise SystemExit(f"{folder}: the state judge's model is not here (set MODEL_MAKER_MODELS)")
        _JUDGE.append(state_judge.StateJudge(folder))
    return _JUDGE[0]


def state_checks(subject, picture, mask):
    """The state judge's check on a picture (an array) and its cut-out, as the pipeline adds it."""
    from PIL import Image

    from model_maker import picture_checks, state_judge, words

    result = picture_checks.PictureResult()
    wanted, unwanted = words.state_texts(subject)
    state_judge.add_state_check(result, subject, state_judge_of().share(Image.fromarray(picture), mask, wanted,
                                                                         unwanted))
    return result.checks


def judged(case, folder):
    """Every verdict of a case: {check name: passed}, and what else it found."""
    import numpy as np
    from PIL import Image

    from model_maker import bake, inspection, mesh_repair, picture_checks, pipeline, words
    from model_maker.weapon_checks import bow_string

    subject = words.subject_of(case["prompt"])
    if case.get("picture_only"):
        picture = np.asarray(Image.open(os.path.join(folder, "picture.png")).convert("RGB"))
        mask = np.asarray(Image.open(os.path.join(folder, "mask.png"))) > 127
        checks = picture_checks.checked(picture, mask).checks
        if case.get("state"):
            checks += state_checks(subject, picture, mask)
        return {c.name: c for c in checks}, {"hollows_removed": 0, "trimmed_m": 0.0, "as_made": {}, "laid_deg": 0.0,
                                              "carved": {}}
    shape = np.load(os.path.join(folder, "shape.npz"))
    made_corners, made_triangles = shape["corners"].astype(np.float64), shape["triangles"].astype(np.int64)
    picture = np.asarray(Image.open(os.path.join(folder, "picture.png")).convert("RGB"))
    mask = np.asarray(Image.open(os.path.join(folder, "mask.png"))) > 127
    found = {"trimmed_m": 0.0, "as_made": {}}
    if case.get("made_must_fail") or case.get("made_must_pass"):
        corners, triangles, _ = mesh_repair.repaired(made_corners, made_triangles)
        made_checks, _, _ = inspection.shape_checks(subject, mask, corners, triangles, picture=picture)
        found["as_made"] = {c.name: c for c in made_checks}
    # Judged as the finish judges it, by the finish's own function (repaired, carved, laid, levelled, checked).
    judgement = inspection.judged(subject, mask, made_corners, made_triangles, picture)
    corners, triangles = judgement.corners, judgement.triangles
    found["hollows_removed"] = judgement.measures["repair"]["hollows_removed"]
    found["carved"] = judgement.measures["carved"]
    found["laid_deg"] = judgement.laid.get("turned_deg", 0.0)
    found["trimmed_m"] = judgement.level["trimmed_m"] if judgement.level else 0.0
    found["weapon"] = judgement.measures.get("weapon") or {}
    checks = list(judgement.checks)
    checks += picture_checks.checked(picture, mask).checks
    if case.get("state"):
        checks += state_checks(subject, picture, mask)
    if case.get("paint") or case.get("fill"):
        painted = np.load(os.path.join(folder, "painted.npz"))
    if case.get("paint"):
        colours, cloud = judgement.paint_followed(made_corners, painted)
        made = bake.baked(corners, triangles, colours, cloud)
        checks.append(inspection.coverage_check(made["coverage"], made["coverage_all_texels"]))
    if case.get("fill"):
        with open(os.path.join(folder, "job.json")) as file:
            touched = json.load(file).get("paint", {}).get("touch_up")
        if (touched or {}).get("views"):
            fit, _ = pipeline.judged_fill(folder, painted, touched, made_corners, made_triangles)
            found["fill"] = fit
            checks.append(inspection.fill_check(fit))
    if subject.weapon_kind == "bow":
        found["string"] = "drawn" if "corners" in bow_string(corners, triangles) else "none"
    return {c.name: c for c in checks}, found


def run(data, only=None):
    from model_maker import ledger

    held = broken = 0
    began = time.time()
    outcomes = {}
    if only and only not in {case["name"] for case in cases()} | {STATE_PICTURES, PROMPT_PICTURES, HEAD_PICTURES,
                                                                  QUALITY_PICTURES, LETTERING_PICTURES}:
        # A mistyped name would judge nothing and say "0 held, 0 broken" with exit 0.
        print(f"no case is named {only!r}", file=sys.stderr)
        return 2
    for case in cases():
        if only and case["name"] != only:
            continue
        started = time.time()
        wrong = changed_files(case, os.path.join(data, case["name"]))
        verdicts, found = judged(case, os.path.join(data, case["name"]))
        for name in case.get("must_fail", []):
            if name not in verdicts or verdicts[name].passed:
                wrong.append(f"'{name}' should fail: {verdicts[name].words if name in verdicts else 'not checked'}")
        for name in case.get("must_pass", []):
            if name not in verdicts or not verdicts[name].passed:
                wrong.append(f"'{name}' should pass: {verdicts[name].words if name in verdicts else 'not checked'}")
        for name in case.get("made_must_fail", []):
            made = found["as_made"].get(name)
            if made is None or made.passed:
                wrong.append(f"'{name}' should fail as made: {made.words if made else 'not checked'}")
        for name in case.get("made_must_pass", []):
            made = found["as_made"].get(name)
            if made is None or not made.passed:
                wrong.append(f"'{name}' should pass as made: {made.words if made else 'not checked'}")
        if "trimmed_mm" in case:
            low, high = case["trimmed_mm"]
            if not low <= found["trimmed_m"] * 1000 <= high:
                wrong.append(f"the feet should be trimmed {low} to {high} mm, were {found['trimmed_m'] * 1000:.1f} mm")
        if "laid_deg" in case:
            low, high = case["laid_deg"]
            if not low <= found["laid_deg"] <= high:
                wrong.append(f"it should be laid down turned {low} to {high} degrees, was {found['laid_deg']:.1f}")
        if "carved_holes" in case:
            carved = sum(1 for h in found["carved"].get("holes", []) if h.get("carved"))
            if carved != case["carved_holes"]:
                wrong.append(f"{case['carved_holes']} holes should be carved, {carved} were: {found['carved']['words']}")
        if "grip_from_butt_cm" in case:
            low, high = case["grip_from_butt_cm"]
            grip = found["weapon"].get("grip") or {}
            grip_cm = (grip.get("from_butt_m") or 0.0) * 100
            if not low <= grip_cm <= high:
                wrong.append(f"the grip should be held {low} to {high} cm from the butt, is {grip_cm:.1f} cm")
        if "handles" in case:
            grip = found["weapon"].get("grip") or {}
            count = len(found["weapon"].get("handles", []))
            if count != case["handles"] or grip.get("handle") in (None, "not found"):
                wrong.append(f"{case['handles']} handles should be found and the grip on one, {count} were "
                             f"(the grip's handle: {grip.get('handle')})")
        if "string" in case and found.get("string") != case["string"]:
            wrong.append(f"the string should be {case['string']}, is {found.get('string')}")
        if found["hollows_removed"] < case.get("hollows_removed", 0):
            wrong.append(f"{case['hollows_removed']} hollows should be removed, {found['hollows_removed']} were")
        seconds = time.time() - started
        if wrong:
            broken += 1
            outcomes[case["name"]] = "broken"
            print(f"BROKEN {case['name']} ({seconds:.0f} s): " + "; ".join(wrong), flush=True)
        else:
            held += 1
            outcomes[case["name"]] = "held"
            print(f"held   {case['name']} ({seconds:.0f} s)", flush=True)
    if not only or only == STATE_PICTURES:
        started = time.time()
        wrong = changed_files({"name": STATE_PICTURES}, os.path.join(data, "states"))
        wrong += judged_states(os.path.join(data, "states"))
        outcomes[STATE_PICTURES] = "broken" if wrong else "held"
        held, broken = held + (not wrong), broken + bool(wrong)
        print(f"{'BROKEN' if wrong else 'held  '} {STATE_PICTURES} ({time.time() - started:.0f} s)"
              + (": " + "; ".join(wrong) if wrong else ""), flush=True)
    if not only or only == PROMPT_PICTURES:
        started = time.time()
        wrong = changed_files({"name": PROMPT_PICTURES}, os.path.join(data, "prompts"))
        wrong += judged_prompts(os.path.join(data, "prompts"))
        outcomes[PROMPT_PICTURES] = "broken" if wrong else "held"
        held, broken = held + (not wrong), broken + bool(wrong)
        print(f"{'BROKEN' if wrong else 'held  '} {PROMPT_PICTURES} ({time.time() - started:.0f} s)"
              + (": " + "; ".join(wrong[:10]) + (f" and {len(wrong) - 10} more" if len(wrong) > 10 else "")
                 if wrong else ""), flush=True)
    if not only or only == HEAD_PICTURES:
        started = time.time()
        wrong = changed_files({"name": HEAD_PICTURES}, os.path.join(data, "heads"))
        wrong += judged_heads(os.path.join(data, "heads"))
        outcomes[HEAD_PICTURES] = "broken" if wrong else "held"
        held, broken = held + (not wrong), broken + bool(wrong)
        print(f"{'BROKEN' if wrong else 'held  '} {HEAD_PICTURES} ({time.time() - started:.0f} s)"
              + (": " + "; ".join(wrong) if wrong else ""), flush=True)
    if not only or only == QUALITY_PICTURES:
        started = time.time()
        wrong = changed_files({"name": QUALITY_PICTURES}, os.path.join(data, "qualities"))
        wrong += judged_qualities(os.path.join(data, "qualities"))
        outcomes[QUALITY_PICTURES] = "broken" if wrong else "held"
        held, broken = held + (not wrong), broken + bool(wrong)
        print(f"{'BROKEN' if wrong else 'held  '} {QUALITY_PICTURES} ({time.time() - started:.0f} s)"
              + (": " + "; ".join(wrong) if wrong else ""), flush=True)
    if not only or only == LETTERING_PICTURES:
        started = time.time()
        wrong = changed_files({"name": LETTERING_PICTURES}, os.path.join(data, "lettering"))
        wrong += judged_lettering(os.path.join(data, "lettering"))
        outcomes[LETTERING_PICTURES] = "broken" if wrong else "held"
        held, broken = held + (not wrong), broken + bool(wrong)
        print(f"{'BROKEN' if wrong else 'held  '} {LETTERING_PICTURES} ({time.time() - started:.0f} s)"
              + (": " + "; ".join(wrong) if wrong else ""), flush=True)
    print(json.dumps({"held": held, "broken": broken}))
    # A run of part of the set (--only) is kept too, marked so: the progress counts whole runs only.
    from model_maker import lettering

    read_by = lettering.reader().version if LETTERING_PICTURES in outcomes and lettering.reader().usable else None
    ledger.regression(held, broken, time.time() - began, outcomes, whole=only is None, tesseract=read_by)
    return 1 if broken else 0


def sha256_of(path):
    import hashlib

    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


def data_files(folder):
    return sorted(name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name)))


def written_sums(data):
    """Every data case's files and their sums (made-up cases are built by code, and are left out), into `SUMS`."""
    found = {case["name"]: {name: sha256_of(os.path.join(data, case["name"], name))
                            for name in data_files(os.path.join(data, case["name"]))}
             for case in cases() if not case.get("made_up") and not case.get("example")}
    for name, sub in ((STATE_PICTURES, "states"), (PROMPT_PICTURES, "prompts"), (HEAD_PICTURES, "heads"),
                      (QUALITY_PICTURES, "qualities"), (LETTERING_PICTURES, "lettering")):
        found[name] = {file: sha256_of(os.path.join(data, sub, file)) for file in data_files(os.path.join(data, sub))}
    with open(SUMS, "w") as file:
        json.dump(found, file, indent=1, sort_keys=True)
        file.write("\n")
    print(f"{len(found)} cases' sums in {SUMS}")


def changed_files(case, folder):
    """What differs from the sums kept for a data case (or the labelled pictures, by their case's name): [] when it
    matches (or is made up)."""
    if case.get("made_up") or case.get("example"):
        return []
    if not os.path.exists(SUMS):
        return ["no sums kept for the set (tools/trial/regression.py sums DATA)"]
    with open(SUMS) as file:
        kept = json.load(file).get(case["name"])
    if kept is None:
        return ["no sums kept for this case"]
    present = {name: sha256_of(os.path.join(folder, name)) for name in data_files(folder)}
    return [f"its {name} is not the one kept" for name in sorted(set(kept) | set(present))
            if kept.get(name) != present.get(name)]


def judged_states(folder):
    """The labelled pictures judged: what went against its label (a known miss is shown, not counted), and the
    confusion printed (pictures showing the state asked for that pass, the other state that fail)."""
    import numpy as np
    from PIL import Image

    from model_maker import state_judge, words

    wrong, tally = [], {"asked shown, passed": 0, "asked shown, failed": 0, "other shown, failed": 0,
                        "other shown, passed": 0}
    for picture in state_pictures():
        subject = words.subject_of(picture["prompt"])
        path = os.path.join(folder, picture["name"] + ".png")
        image = np.asarray(Image.open(path).convert("RGB"))
        mask = np.asarray(Image.open(path[:-4] + "-mask.png")) > 127
        check = next(c for c in state_checks(subject, image, mask) if c.name == state_judge.NAME)
        asked = picture["shows"] == subject.state
        tally[f"{'asked' if asked else 'other'} shown, {'passed' if check.passed else 'failed'}"] += 1
        right = check.passed == asked
        note = "" if right else " (known miss)" if picture.get("known_miss") else " WRONG"
        if right and picture.get("known_miss"):
            note = " (a known miss, now caught: drop its known_miss)"
        print(f"   {check.value / 100:.3f} {'passes' if check.passed else 'fails '} shows {picture['shows']:6s} "
              f"{picture['name']}{note}", flush=True)
        if not right and not picture.get("known_miss"):
            wrong.append(f"{picture['name']} shows {picture['shows']}, {'passed' if check.passed else 'failed'} "
                         f"({check.value:.0f} %)")
    print("   " + ", ".join(f"{k}: {v}" for k, v in tally.items()), flush=True)
    return wrong


# A head's label against the prompt's family: what the pipeline's choice of try must do with it. Only a clear hammer or
# axe head is held; a head between the two, or another thing, is shown.
AXE_NOUNS = ("axe", "battle axe", "battleaxe", "hatchet")
HELD_HEADS = ("hammer", "axe")


def head_verdict_wanted(prompt, head):
    """True when the picture must pass, False when it must fail, None when it is shown only."""
    from model_maker import words

    noun = words.subject_of(prompt).noun
    family = "hammer" if noun in words.HEAD_KINDS else "axe" if noun in AXE_NOUNS else None
    if head not in HELD_HEADS or family is None:
        return None
    return head == family


def judged_heads(folder):
    """The war hammer family's and the axes' pictures, their heads labelled by eye (hammer, axe or hybrid): each judged
    as the pipeline's choice of try judges it. A clear head of the prompt's own kind must pass, one of the other kind
    must fail; a hybrid is shown. The tally says how many of each passed."""
    import numpy as np
    from PIL import Image

    from model_maker import picture_checks, subject_judge, words

    judge = state_judge_of()
    wrong, tally, directions = [], {}, {}
    for picture in head_pictures():
        prompt = picture["prompt"]
        if prompt not in directions:
            directions[prompt] = subject_judge.directions_of(judge, words.subject_of(prompt))
        path = os.path.join(folder, picture["name"] + ".png")
        image = Image.open(path).convert("RGB")
        mask = np.asarray(Image.open(path[:-4] + "-mask.png").convert("L")) > 127
        result = picture_checks.PictureResult()
        subject_judge.add_subject_check(result, words.subject_of(prompt),
                                        *subject_judge.judged(judge, image, mask, directions[prompt]))
        check = result.checks[0]
        wanted = head_verdict_wanted(prompt, picture["head"])
        key = f"{prompt}, {picture['head']}"
        tally.setdefault(key, [0, 0])[0 if check.passed else 1] += 1
        note = "" if wanted is None or wanted == check.passed else " WRONG"
        print(f"   {check.value / 100:.3f} {'passes' if check.passed else 'fails '} {picture['head']:6s} "
              f"{picture['name']}{note}", flush=True)
        if note:
            wrong.append(f"{picture['name']} ({picture['head']} head) {'passed' if check.passed else 'failed'} "
                         f"({check.value:.0f} %)")
    print("   " + "; ".join(f"{k}: {p} pass, {f} fail" for k, (p, f) in sorted(tally.items())), flush=True)
    return wrong


def quality_verdict_wanted(prompt, shows):
    """The quality `prompt` names that a picture showing `shows` is held to, and whether its pair must pass the picture
    (True) or refuse it (False); (None, None) when it is shown only (in doubt, or a quality the prompt does not name)."""
    from model_maker import words

    for quality in words.subject_of(prompt).qualities:
        if shows == quality:
            return quality, True
        if shows == words.QUALITY_OPPOSITES[quality]:
            return quality, False
    return None, None


def judged_qualities(folder):
    """The pictures painted with a quality and without it, labelled by eye: each judged against its case's prompt as
    the pipeline's choice of try judges it. The quality's pair must pass a picture showing the quality and refuse one
    showing its opposite; one in doubt is shown. The whole check is shown beside it: a picture can fail it for the thing
    (a curved dagger seen as a scythe), which the quality's pair does not answer for. The tally says how many of each
    the pair passed, and how many the whole check passed."""
    import numpy as np
    from PIL import Image

    from model_maker import picture_checks, subject_judge, words

    judge = state_judge_of()
    least = subject_judge.LEAST_SUBJECT_SHARE
    wrong, tally, directions = [], {}, {}
    for picture in quality_pictures():
        prompt = picture["prompt"]
        if prompt not in directions:
            directions[prompt] = subject_judge.directions_of(judge, words.subject_of(prompt))
        path = os.path.join(folder, picture["name"] + ".png")
        image = Image.open(path).convert("RGB")
        mask = np.asarray(Image.open(path[:-4] + "-mask.png").convert("L")) > 127
        share, likeliest, qualities = subject_judge.judged(judge, image, mask, directions[prompt])
        result = picture_checks.PictureResult()
        subject_judge.add_subject_check(result, words.subject_of(prompt), share, likeliest, qualities)
        check = result.checks[0]
        quality, wanted = quality_verdict_wanted(prompt, picture["shows"])
        found = dict(qualities).get(quality) if quality else None
        key = f"{prompt}, {picture['shows']}"
        counts = tally.setdefault(key, [0, 0, 0])
        counts[0] += found is not None and found >= least
        counts[1] += found is not None and found < least
        counts[2] += check.passed
        note = "" if wanted is None or wanted == (found >= least) else " WRONG"
        said = f"{quality} {found:.3f}" if found is not None else "shown only"
        print(f"   {said:14s} check {check.value / 100:.3f} {'passes' if check.passed else 'fails '} "
              f"{picture['shows']:8s} {picture['name']}{note}", flush=True)
        if note:
            wrong.append(f"{picture['name']} ({picture['shows']}, judged as {prompt!r}): {quality} "
                         f"{'passed' if found >= least else 'refused'} it ({found:.0%})")
    print("   " + "; ".join(f"{k}: the pair passes {p}, refuses {f}; the whole check passes {c}"
                          for k, (p, f, c) in sorted(tally.items())), flush=True)
    return wrong


def judged_lettering(folder):
    """The picture tries labelled by eye for painted lettering, each judged as the pipeline's choice of try judges
    its prompt: a try showing letters its prompt does not ask for must be refused (a known miss is shown), one showing
    none passed, and every try whose prompt asks for lettering passed; one in doubt is shown. The tally says how many
    of each were refused."""
    import numpy as np
    from PIL import Image

    from model_maker import lettering, words

    found = lettering.reader()
    if not found.usable:
        return [found.why]
    print(f"   read by tesseract {found.version}", flush=True)
    wrong, tally, doubt, misses = [], {}, [], []
    for picture in lettering_pictures():
        path = os.path.join(folder, picture["name"] + ".png")
        rgb = np.asarray(Image.open(path).convert("RGB"))
        mask = np.asarray(Image.open(path[:-4] + "-mask.png").convert("L")) > 127
        asked = words.lettering_asked(picture["prompt"])
        check = lettering.lettering_check(rgb, mask, asked)
        key = f"{picture['shows']}{', asked' if asked else ''}"
        counts = tally.setdefault(key, [0, 0])
        counts[check.passed] += 1
        must_pass = bool(asked) or picture["shows"] == "none"
        if picture["shows"] == "unclear" and not asked:
            doubt.append(f"{picture['name']} {'passed' if check.passed else 'refused'} ({check.value:.0f})")
        elif picture.get("known_miss") and not asked:
            misses.append(f"{picture['name']} {'passed' if check.passed else 'now refused'}")
        elif check.passed != must_pass:
            wrong.append(f"{picture['name']} ({key}) {'passed' if check.passed else 'refused'}: {check.words}")
    print("   " + "; ".join(f"{key}: refused {refused}, passed {passed}"
                          for key, (refused, passed) in sorted(tally.items())), flush=True)
    if doubt:
        print("   in doubt by eye (shown, not held): " + ", ".join(doubt), flush=True)
    if misses:
        print(f"   known misses (shown, not held): {sum('now refused' in m for m in misses)} of {len(misses)} refused: "
              + ", ".join(misses), flush=True)
    return wrong


def made_up_pictures(prompt):
    """Pictures that show nothing of the prompt's thing: an empty backdrop, the prompt's words written out, its noun
    written large (the typographic attack on picture-text models)."""
    from PIL import Image, ImageDraw, ImageFont

    from model_maker import words

    def font(size):
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    empty = Image.new("RGB", (1024, 1024), (225, 225, 225))
    written, noun = empty.copy(), empty.copy()
    lines = prompt.replace(" with ", "\nwith ").replace(", ", ",\n")
    ImageDraw.Draw(written).multiline_text((60, 420), lines, fill=(20, 20, 20), font=font(56))
    ImageDraw.Draw(noun).text((120, 440), words.subject_of(prompt).noun.upper(), fill=(20, 20, 20), font=font(120))
    return {"an empty backdrop": empty, "its words written out": written, "its noun written large": noun}


def judged_prompts(folder):
    """The labelled pictures against their own prompts and every other (a picture marked unclear is shown, not held),
    and the made-up pictures of each prompt: what went against its label, with the margins printed."""
    import numpy as np
    from PIL import Image

    from model_maker import state_judge, subject_judge, words

    judge = state_judge_of()
    scale = float(judge.model.logit_scale.exp().item())
    pictures = prompt_pictures()
    prompts = sorted({picture["prompt"] for picture in pictures})
    classes = {prompt: subject_judge.directions_of(judge, words.subject_of(prompt)) for prompt in prompts}

    def share(views, prompt):
        """The check's value (the thing's share, or a quality's when lower) and the likeliest other thing."""
        found, likeliest, qualities = subject_judge.averaged(
            [subject_judge.view_shares(view, classes[prompt], scale) for view in views], classes[prompt][2])
        return min([found] + [quality for _, quality in qualities]), likeliest

    least = subject_judge.LEAST_SUBJECT_SHARE
    wrong, own, against, unclear = [], [], [], []
    for picture in pictures:
        path = os.path.join(folder, picture["name"] + ".png")
        image = Image.open(path).convert("RGB")
        mask = np.asarray(Image.open(path[:-4] + "-mask.png").convert("L")) > 127
        views = [judge.features(image=image)[0]]
        if mask.any():
            views.append(judge.features(image=state_judge.cropped(image, mask))[0])
        if picture.get("shows") == "unclear":
            unclear.append((share(views, picture["prompt"])[0], picture["name"]))
            continue
        mine, other = share(views, picture["prompt"])
        own.append((mine, picture["name"], other))
        if mine < least:
            wrong.append(f"{picture['name']} fails its own prompt ({mine:.0%}, the likeliest other {other})")
        for prompt in prompts:
            if prompt != picture["prompt"]:
                theirs = share(views, prompt)[0]
                against.append((theirs, picture["name"], prompt))
                if theirs >= least:
                    wrong.append(f"{picture['name']} passes for {prompt!r} ({theirs:.0%})")
    made_up = []
    for prompt in prompts:
        for what, image in made_up_pictures(prompt).items():
            theirs = share([judge.features(image=image)[0]], prompt)[0]
            made_up.append((theirs, what, prompt))
            # Written words are a known miss (the typographic attack; docs/objects.md): shown, not held.
            if theirs >= least and what == "an empty backdrop":
                wrong.append(f"{what} passes for {prompt!r} ({theirs:.0%})")
    own.sort()
    against.sort(reverse=True)
    made_up.sort(reverse=True)
    written = [m for m in made_up if m[1] != "an empty backdrop"]
    print(f"   written words (a known miss, shown, not held): {sum(m[0] >= least for m in written)} of {len(written)} "
          f"pass", flush=True)
    made_up = [m for m in made_up if m[1] == "an empty backdrop"]
    print(f"   own prompt: {len(own)} pictures, lowest {own[0][0]:.3f} ({own[0][1]}, the likeliest other {own[0][2]}); "
          f"another prompt: {len(against)} pairs, highest {against[0][0]:.3f} ({against[0][1]} for {against[0][2]!r}); "
          f"made up: highest {made_up[0][0]:.3f} ({made_up[0][1]} for {made_up[0][2]!r}); limit {least:.2f}",
          flush=True)
    print("   unclear by eye (shown, not held): " + ", ".join(f"{name} {value:.2f}" for value, name in sorted(unclear)),
          flush=True)
    return wrong


def main():
    parser = argparse.ArgumentParser()
    steps = parser.add_subparsers(dest="step", required=True)
    making = steps.add_parser("build")
    making.add_argument("trials")
    making.add_argument("data")
    running = steps.add_parser("run")
    running.add_argument("data")
    running.add_argument("--only")
    summing = steps.add_parser("sums")
    summing.add_argument("data")
    args = parser.parse_args()
    if args.step == "build":
        build(args.trials, args.data)
        return 0
    if args.step == "sums":
        written_sums(args.data)
        return 0
    return run(args.data, args.only)


if __name__ == "__main__":
    sys.exit(main())
