#!/usr/bin/env python3
"""Content credentials for pictures already published: a signed C2PA manifest added to each AI-made picture of a
folder, its pixels as they are.

    .venv/bin/python tools/sign_pictures.py FOLDER --list LIST.toml [--signing-key KEY --signing-cert CHAIN]
                                            [--dry-run] [--again]
    .venv/bin/python tools/sign_pictures.py FOLDER --check

Which pictures are AI-made is never guessed: the list (TOML) names them, each with what it shows and why, and every
other picture of the folder is left alone (and listed as such):

    version = "0.2.0"                     # the model-maker version whose pipeline made them (each may say its own)
    generator = "..."                     # only for pictures another tool made: its plain name (each may say its
                                          # own); `version` is then that tool's, and `models` a list of the models'
                                          # names as written (model-maker's registry need not know them)

    [[picture]]
    path = "examples.webp"                # under FOLDER; a shell pattern names several
    kind = "composite"                    # "ai": only the models' work (trainedAlgorithmicMedia); "composite": the
                                          # models' work with our own drawing or words (compositeWithTrained...)
    models = "object"                     # "picture" (the picture model), "cut-out", "shape" or "object" (every
                                          # model of the pipeline), or a list of the registry's model ids
    why = "renders of eight 3-D objects model-maker made, our captions under them"

Each manifest says `c2pa.created`, its digital source type, model-maker and the version, and the models by name;
never a prompt, a path or a name. Pictures are PNG, WebP or JPEG. A picture already carrying a manifest is left as it
is (`--again` signs over one model-maker wrote, for a new certificate); after signing, each is read back: its
manifest must hold and its pixels must be the ones it had, else the picture is put back as it was and the tool fails.
Without --signing-key and --signing-cert (or MODEL_MAKER_SIGNING_KEY and MODEL_MAKER_SIGNING_CERT) it signs with the
test certificate made on this computer, and says so: checkers call that untrusted.
"""

import argparse
import fnmatch
import os
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from model_maker import credentials  # noqa: E402

PICTURE_ENDINGS = (".png", ".webp", ".jpg", ".jpeg")
KINDS = {"ai": credentials.TRAINED, "composite": credentials.COMPOSITE}
MODEL_SETS = {"picture": credentials.PICTURE_MODELS, "cut-out": credentials.CUT_OUT_MODELS,
              "shape": credentials.SHAPE_MODELS, "object": credentials.OBJECT_MODELS}


class Refused(Exception):
    pass


def pictures_in(folder):
    """Every picture under `folder`, by its path under it ("/" between folders)."""
    found = []
    for root, _, names in os.walk(folder):
        for name in names:
            if name.lower().endswith(PICTURE_ENDINGS) and not name.startswith("."):
                found.append(os.path.relpath(os.path.join(root, name), folder).replace(os.sep, "/"))
    return sorted(found)


def entries_of(list_path):
    """The list's pictures: [(pattern, digital source type, model ids or names, version, why, generator)]; with a
    generator (another tool's picture) the models are names as written, else model-maker's registry ids."""
    with open(list_path, "rb") as file:
        listed = tomllib.load(file)
    entries = []
    for number, entry in enumerate(listed.get("picture", []), 1):
        where = f"{list_path}, picture {number}"
        if not entry.get("path") or not entry.get("why"):
            raise Refused(f"{where}: every picture needs its path and why it is AI-made")
        if entry.get("kind") not in KINDS:
            raise Refused(f"{where}: kind must be one of {', '.join(KINDS)}")
        generator = entry.get("generator") or listed.get("generator")
        models = entry.get("models", None if generator else "object")
        if generator:
            if not isinstance(generator, str) or not generator.strip() or generator.strip() == credentials.GENERATOR:
                raise Refused(f"{where}: generator is another tool's plain name (model-maker's own pictures name none)")
            if not isinstance(models, list) or not models or not all(isinstance(one, str) and one.strip() for one in models):
                raise Refused(f"{where}: with a generator, models is a list of the models' names")
            model_ids = tuple(one.strip() for one in models)
        else:
            model_ids = MODEL_SETS.get(models) if isinstance(models, str) else tuple(models)
            if model_ids is None:
                raise Refused(f"{where}: models must be one of {', '.join(MODEL_SETS)} or a list of model ids")
            credentials.model_names(model_ids)
        version = entry.get("version") or listed.get("version")
        if not version:
            raise Refused(f"{where}: no version (give `version` at the top or in the picture)")
        entries.append((entry["path"], KINDS[entry["kind"]], model_ids, str(version), entry["why"],
                        generator.strip() if generator else None))
    return entries


def chosen_for(names, entries):
    """{name: (digital source type, model ids or names, version, generator)} for the pictures the list names; a
    pattern that names nothing is refused (the list must say what is there)."""
    chosen = {}
    for pattern, source_type, model_ids, version, _, generator in entries:
        named = [name for name in names if fnmatch.fnmatchcase(name, pattern)]
        if not named:
            raise Refused(f"the list names {pattern!r}, and no picture of the folder is that")
        for name in named:
            chosen[name] = (source_type, model_ids, version, generator)
    return chosen


def pixels_of(path):
    from PIL import Image

    with Image.open(path) as picture:
        picture.load()
        return picture.mode, picture.size, picture.tobytes()


def signed_in_place(path, source_type, model_ids, version, signing, generator=None):
    """Signs one picture; it is put back as it was unless its manifest holds and its pixels are the same."""
    with open(path, "rb") as file:
        original = file.read()
    before = pixels_of(path)
    try:
        if generator:
            credentials.sign_file(path, (), source_type, signing, version, generator=generator, models=model_ids)
        else:
            credentials.sign_file(path, model_ids, source_type, signing, version)
        reading = credentials.read(path)
        if not reading.marked or reading.source_type != source_type:
            raise Refused(f"{path}: its manifest does not hold after signing ({reading.state}, {reading.failures})")
        if pixels_of(path) != before:
            raise Refused(f"{path}: its pixels changed in signing")
    except Exception:
        credentials.write_whole(path, original)
        raise


def checked(folder):
    """Every picture of the folder and what its manifest says."""
    for name in pictures_in(folder):
        reading = credentials.read(os.path.join(folder, name))
        if not reading.found:
            print(f"{name}: no content credentials")
        else:
            print(f"{name}: {'marked' if reading.marked else 'NOT HOLDING'} ({reading.state}); "
                  f"{(reading.source_type or '').rsplit('/', 1)[-1]}; {reading.generator} {reading.version}; "
                  f"signed by {reading.issuer}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Add content credentials (C2PA) to AI-made pictures already made.")
    parser.add_argument("folder")
    parser.add_argument("--list", help="the TOML list of the folder's AI-made pictures")
    parser.add_argument("--check", action="store_true", help="only say what each picture's manifest says")
    parser.add_argument("--signing-key", help="a PEM private key (default MODEL_MAKER_SIGNING_KEY; else the test "
                        "certificate)")
    parser.add_argument("--signing-cert", help="its certificate chain, PEM (default MODEL_MAKER_SIGNING_CERT)")
    parser.add_argument("--dry-run", action="store_true", help="say what would be signed; change nothing")
    parser.add_argument("--again", action="store_true",
                        help="sign again a picture whose manifest model-maker wrote (a new certificate)")
    args = parser.parse_args(argv)
    if args.check:
        return checked(args.folder)
    if not args.list:
        parser.error("--list is needed: which pictures are AI-made is never guessed")
    try:
        names = pictures_in(args.folder)
        chosen = chosen_for(names, entries_of(args.list))
        signing = credentials.chosen(args.signing_key, args.signing_cert)
    except (Refused, credentials.NotSigned) as error:
        print(f"sign_pictures: {error}", file=sys.stderr)
        return 1
    print(signing.words)
    counts = {"signed": 0, "already": 0, "left alone": 0}
    for name in names:
        path = os.path.join(args.folder, name)
        if name not in chosen:
            print(f"{name}: left alone (the list does not name it AI-made)")
            counts["left alone"] += 1
            continue
        source_type, model_ids, version, generator = chosen[name]
        made_by = f"{generator or credentials.GENERATOR} {version}"
        reading = credentials.read(path)
        if reading.found and not (args.again and reading.generator == (generator or credentials.GENERATOR)):
            print(f"{name}: left as it is (it carries a manifest already: {reading.generator} {reading.version}, "
                  f"{'holding' if reading.marked else 'NOT holding'})")
            counts["already"] += 1
            continue
        kind = source_type.rsplit("/", 1)[-1]
        if args.dry_run:
            print(f"{name}: would be signed ({kind}, {made_by})")
            continue
        try:
            signed_in_place(path, source_type, model_ids, version, signing, generator)
        except (Refused, credentials.NotSigned) as error:
            print(f"sign_pictures: {error}", file=sys.stderr)
            return 1
        print(f"{name}: signed ({kind}, {made_by}); pixels the same, manifest holds")
        counts["signed"] += 1
    print("; ".join(f"{count} {what}" for what, count in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
