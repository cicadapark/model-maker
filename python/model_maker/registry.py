"""The registry of AI models (registry/models.toml): every model's licence and the sha256 of its files.

Before a model is loaded, every file in its folder that the registry lists must match its sum, and a file that runs
code when loaded (a pickle, a model's own Python) must be listed: anything else is refused. The sidecar of every
object made names each model used with its version and licence.
"""

import hashlib
import os
import subprocess
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.normpath(os.path.join(HERE, "..", "..", "registry", "models.toml"))
# Files that run as code when loaded: only ever with a registry sum.
CODE_ENDINGS = (".ckpt", ".bin", ".pt", ".pth", ".pkl", ".py")
HASH_CHUNK = 8 * 1024 * 1024


def models(path=REGISTRY):
    with open(path, "rb") as file:
        return {m["id"]: m for m in tomllib.load(file)["model"]}


def sums_of(entry):
    """A model's files as the registry lists them: path within its hub repository -> sha256."""
    sums = {}
    for part in entry.get("parts", []):
        url = part["url"]
        if "/resolve/" in url:
            sums[url.split("/resolve/", 1)[1].split("/", 1)[-1]] = part["sha256"]
        else:
            sums[url.rsplit("/", 1)[-1]] = part["sha256"]
    return sums


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class RefusedFile(Exception):
    pass


# A command that needs a model not downloaded here stops with this code (EX_TEMPFAIL), so a caller can tell "not
# available here" from "the files made it fail" (a program that calls model-maker can wait, then try again).
MODEL_MISSING_EXIT = 75


class ModelMissing(Exception):
    pass


def model_folder(model_id, folder_name, under=None):
    """A model's folder under `under` (or MODEL_MAKER_MODELS), every listed file checked against its sum; raises
    ModelMissing when the model is not downloaded there whole."""
    folder = os.path.join(under or os.environ.get("MODEL_MAKER_MODELS", "models"), folder_name)
    missing = [name for name in sums_of(models()[model_id])
               if not os.path.isfile(os.path.join(folder, name))]
    if missing:
        raise ModelMissing(f"{folder}: the model {model_id} is not here whole ({', '.join(missing[:3])} missing; set "
                           "MODEL_MAKER_MODELS or run ./setup.sh)")
    verify_folder(model_id, folder)
    return folder


def without_bytecode(folder):
    """The folder's `__pycache__` folders removed. Python runs a cached `.pyc` in place of its source when the cache's
    stamp matches, so a planted one would run unchecked; the model
    code is compiled again from its checked sources (and nothing is cached: `__main__` sets dont_write_bytecode)."""
    import shutil

    for root, dirs, _ in os.walk(folder):
        if "__pycache__" in dirs:
            shutil.rmtree(os.path.join(root, "__pycache__"))
            dirs.remove("__pycache__")


def verify_folder(model_id, folder, path=REGISTRY, checked=None):
    """Every listed file in the folder matches its sum; no unlisted code (a cached `.pyc` is removed first:
    `without_bytecode`). `checked` (a set) skips files already checked this run. Returns how many files were
    checked."""
    sums = {}
    for entry_id in ([model_id] if isinstance(model_id, str) else model_id):
        sums.update(sums_of(models(path)[entry_id]))
    without_bytecode(folder)
    present = 0
    for root, _, names in os.walk(folder):
        if ".cache" in root.split(os.sep):
            continue
        for name in names:
            file_path = os.path.join(root, name)
            relative = os.path.relpath(file_path, folder).replace(os.sep, "/")
            listed = sums.get(relative)
            if listed is None and name.lower().endswith(CODE_ENDINGS):
                raise RefusedFile(f"{file_path}: code or a pickle the registry does not list; refusing to load it")
            if listed is None:
                continue
            present += 1
            if checked is not None and file_path in checked:
                continue
            if sha256_of(file_path) != listed:
                raise RefusedFile(f"{file_path}: its sha256 is not the registry's; refusing to load it")
            if checked is not None:
                checked.add(file_path)
    if present == 0:
        raise RefusedFile(f"{folder}: none of the files the registry lists for {model_id} is here; refusing to load it")
    return present


def verify_checkout(model_id, folder, path=REGISTRY):
    """A model's own code (its registry entry's `source`) checked before it is imported: the folder is a git checkout
    at the commit the registry names, with nothing changed, added or left over (a cached `.pyc` is removed first).
    Raises RefusedFile otherwise."""
    source = models(path)[model_id]["source"]
    without_bytecode(folder)

    def git(*args):
        try:
            return subprocess.run(["git", "-C", folder, *args], check=True, capture_output=True, text=True).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise RefusedFile(f"{folder}: not a git checkout of {model_id}'s code ({error}); run ./setup.sh")

    if not os.path.isdir(os.path.join(folder, ".git")):
        raise RefusedFile(f"{folder}: not a git checkout of {model_id}'s code; run ./setup.sh")
    found = git("rev-parse", "HEAD").strip()
    if found != source["commit"]:
        raise RefusedFile(f"{folder}: at {found[:12]}, not the registry's {source['commit'][:12]}; refusing to load it")
    removed = set(source.get("remove", []))
    changed = [line[3:] for line in git("status", "--porcelain", "--ignored", "--untracked-files=all").splitlines()
               if not (line.startswith(" D ") and line[3:] in removed)]
    if changed:
        raise RefusedFile(f"{folder}: changed from the registry's commit ({', '.join(changed[:3])}); refusing to load "
                          "it (run ./setup.sh)")


def used_record(model_ids, path=REGISTRY):
    """The sidecar's list of the models used: id, name, role, version, licence and where to read it."""
    known = models(path)
    return [{key: known[m][key] for key in ("id", "name", "role", "version", "licence", "licence_url")}
            for m in model_ids]
