#!/usr/bin/env python3
"""The AI models and their own code, as registry/models.toml lists them: each model's licence shown and accepted first,
then its files downloaded from its hub at the pinned revision into models/<folder>/, each checked against its sha256,
and its code, where the registry names a repository, checked out at its pinned commit into sources/<folder>/.

    .venv/bin/python -I setup/fetch_models.py [--yes] [--hf-token-file FILE] [--models DIR] [--sources DIR]
    .venv/bin/python -I setup/fetch_models.py --verify        # only check what is there against the sums

The Python standard library only. A download stopped half-way goes on from where it stopped. A file already there is
checked again only when its size or time changed since it last passed (models/.verified). The licences accepted are
kept in models/.accepted, so a second run does not ask again.

An access token for a gated model is read from the file named by --hf-token-file (or HF_TOKEN_FILE, else the hub's own
~/.cache/huggingface/token), or from HF_TOKEN. It is sent only to huggingface.co, never printed and never written.
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.dirname(HERE)
REGISTRY = os.path.join(REPOSITORY, "registry", "models.toml")
LICENCE_TEXTS = os.path.join(HERE, "licences")
HUB = "huggingface.co"
CHUNK = 4 * 1024 * 1024
TRIES = 6
PARALLEL = 4
# The pipeline cannot make anything without these; SigLIP 2 judges each picture against its prompt (without it the
# pictures go unjudged) and is needed for a thing with a lid, a cover or doors, and for the regression set.
NEEDED = ("flux1-schnell", "birefnet", "triposg", "mv-adapter-ig2mv-sdxl", "sdxl-base-1.0", "sdxl-vae-fp16-fix")
SPDX_TEXT = {"Apache-2.0": os.path.join(REPOSITORY, "LICENSE"), "MIT": os.path.join(LICENCE_TEXTS, "MIT.txt")}
REPORT_EVERY_S = 15
# A licence's whole text fetched from the hub (SDXL's, with its use restrictions) is kept here under the models' folder,
# named by the model's id, so it can be read in full before the question is answered.
LICENCE_FOLDER = "licences"


class Failed(Exception):
    pass


def say(text=""):
    print(text, flush=True)


def wrapped(text, indent="  "):
    return textwrap.fill(text, width=100, initial_indent=indent, subsequent_indent=indent)


def human(size):
    return f"{size / 1e9:.1f} GB" if size >= 1e9 else f"{size / 1e6:.0f} MB"


# The registry


def models(path=REGISTRY):
    with open(path, "rb") as file:
        return tomllib.load(file)["model"]


def hub_place(url):
    """(repository, revision, path within it) of a hub file's URL."""
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc != HUB or "/resolve/" not in parsed.path:
        raise Failed(f"{url}: not a file on {HUB}")
    repository, rest = parsed.path.strip("/").split("/resolve/", 1)
    revision, inside = rest.split("/", 1)
    return repository, revision, inside


def fetched_parts(entry, models_dir):
    """Each part setup downloads: (url, its place under models_dir, sha256, approximate size in bytes)."""
    parts = []
    for part in entry.get("parts", []):
        if not part.get("fetch", True):
            continue
        repository, revision, inside = hub_place(part["url"])
        if len(revision) != 40:
            raise Failed(f"{part['url']}: not pinned to a revision (a 40-character commit)")
        folder = entry.get("folders", {}).get(repository)
        if folder is None:
            raise Failed(f"{entry['id']}: the registry names no folder for {repository}")
        parts.append((part["url"], os.path.join(models_dir, folder, *inside.split("/")), part["sha256"],
                      max(1, part.get("size_mb", 1)) * 1_000_000))
    return parts


# The token


def read_token(token_file):
    """The hub token, or None. Read from the file only; its value is never shown."""
    candidates = [token_file, os.environ.get("HF_TOKEN_FILE"),
                  os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token")]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            with open(candidate) as file:
                token = file.read().strip()
            if token:
                return token
        elif candidate is token_file and token_file:
            raise Failed(f"{token_file}: no such file")
    return os.environ.get("HF_TOKEN") or None


# Downloads


def opened(url, token, start=0):
    request = urllib.request.Request(url, headers={"User-Agent": "model-maker-setup/1"})
    if start:
        request.add_header("Range", f"bytes={start}-")
    if token and urllib.parse.urlparse(url).netloc == HUB:
        # Unredirected: the hub's redirect to its file store (a signed link) does not carry the token on.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(request, timeout=60)


def sha256_of(path, digest=None):
    digest = digest or hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(CHUNK):
            digest.update(chunk)
    return digest


class Progress:
    def __init__(self, total):
        self.total, self.done, self.lock = total, 0, threading.Lock()
        self.started = self.last = time.time()

    def add(self, count):
        with self.lock:
            self.done += count
            now = time.time()
            if now - self.last >= REPORT_EVERY_S:
                self.last = now
                rate = self.done / max(now - self.started, 1e-3)
                left = (self.total - self.done) / rate if rate > 0 else 0
                say(f"  ... {human(self.done)} of {human(self.total)} downloaded, {rate / 1e6:.0f} MB/s, "
                    f"about {left / 60:.0f} min left")


def downloaded(url, path, sha256, token, progress):
    """The file at `path`, downloaded (or its download carried on) and checked. Raises Failed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = path + ".part"
    for attempt in range(TRIES):
        start = os.path.getsize(partial) if os.path.exists(partial) else 0
        try:
            with opened(url, token, start) as reply:
                if start and reply.status != 206:
                    start = 0  # the server sent it whole: begin again
                with open(partial, "ab" if start else "wb") as file:
                    while chunk := reply.read(CHUNK):
                        file.write(chunk)
                        progress.add(len(chunk))
            break
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise Failed(f"{url}: the hub refused it ({error.code}): see 'A gated model' in docs/setup.md")
            if error.code == 416:  # asked past its end: it is all here
                break
            if attempt == TRIES - 1:
                raise Failed(f"{url}: {error}")
        except (OSError, TimeoutError) as error:
            if attempt == TRIES - 1:
                raise Failed(f"{url}: {error}")
        time.sleep(min(60, 5 * 2 ** attempt))
    if sha256_of(partial).hexdigest() != sha256:
        os.remove(partial)
        raise Failed(f"{url}: its sha256 is not the registry's; the download was removed (run setup again)")
    os.replace(partial, path)


class Verified:
    """models/.verified: each file's size and time when its sum last passed, so it is not read again for nothing."""

    def __init__(self, models_dir):
        self.path = os.path.join(models_dir, ".verified")
        self.lock = threading.Lock()
        try:
            with open(self.path) as file:
                self.known = json.load(file)
        except (OSError, ValueError):
            self.known = {}

    @staticmethod
    def stamp(path, sha256):
        info = os.stat(path)
        return [info.st_size, int(info.st_mtime), sha256]

    def holds(self, path, sha256):
        return os.path.exists(path) and self.known.get(path) == self.stamp(path, sha256)

    def add(self, path, sha256):
        with self.lock:
            self.known[path] = self.stamp(path, sha256)
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path + ".new", "w") as file:
                json.dump(self.known, file, indent=0)
            os.replace(self.path + ".new", self.path)


def checked_or_fetched(parts, token, verified, verify_only=False):
    """Every part present and matching its sum. Returns the problems (empty when all is well)."""
    todo, problems = [], []
    for url, path, sha256, size in parts:
        if verified.holds(path, sha256):
            continue
        if os.path.exists(path):
            if sha256_of(path).hexdigest() == sha256:
                verified.add(path, sha256)
                continue
            if verify_only:
                problems.append(f"{path}: its sha256 is not the registry's")
                continue
            os.remove(path)
        if verify_only:
            problems.append(f"{path}: missing")
            continue
        todo.append((url, path, sha256, size))
    if not todo:
        return problems
    total = sum(size for *_, size in todo)
    say(f"  downloading {len(todo)} files, about {human(total)}")
    progress = Progress(total)

    def one(item):
        url, path, sha256, _ = item
        downloaded(url, path, sha256, token, progress)
        verified.add(path, sha256)

    with concurrent.futures.ThreadPoolExecutor(PARALLEL) as pool:
        for item, outcome in zip(todo, pool.map(lambda item: _caught(one, item), todo)):
            if outcome:
                problems.append(outcome)
    return problems


def _caught(function, item):
    try:
        function(item)
        return None
    except Failed as error:
        return str(error)


# The models' code


# Code model-maker never reads: TripoSG's (model-maker runs the weights with its own code; their repository's code
# folder holds a licence that excludes the European Union, the United Kingdom and South Korea). A folder of it found
# in sources/ is removed at setup.
RETIRED_SOURCES = ("triposg",)


def retired_sources_removed(sources_dir):
    """The folders of RETIRED_SOURCES under sources_dir removed; returns those that were there."""
    removed = []
    for name in RETIRED_SOURCES:
        folder = os.path.join(sources_dir, name)
        if os.path.isdir(folder) and not os.path.islink(folder):
            shutil.rmtree(folder)
        elif os.path.lexists(folder):
            os.remove(folder)
        else:
            continue
        removed.append(folder)
    return removed


def checked_out(source, sources_dir):
    """The model's own code at its commit in sources/<folder>; files the registry says to remove are removed."""
    folder = os.path.join(sources_dir, source["folder"])
    commit = source["commit"]

    def git(*args, capture=False):
        return subprocess.run(["git", "-C", folder, *args], check=True, text=True,
                              stdout=subprocess.PIPE if capture else subprocess.DEVNULL, stderr=subprocess.PIPE)

    if os.path.exists(folder) and not os.path.isdir(os.path.join(folder, ".git")):
        # Not a checkout (an older run's leftovers): made again.
        shutil.rmtree(folder)
    if os.path.isdir(os.path.join(folder, ".git")):
        try:
            if git("rev-parse", "HEAD", capture=True).stdout.strip() == commit:
                _removed(folder, source)
                return f"{folder} at {commit[:12]} (already)"
        except subprocess.CalledProcessError:
            pass
        shutil.rmtree(folder)
    os.makedirs(folder)
    try:
        git("init", "-q")
        git("remote", "add", "origin", source["url"])
        git("fetch", "-q", "--depth", "1", "origin", commit)
        git("-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD")
        found = git("rev-parse", "HEAD", capture=True).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise Failed(f"{source['url']} at {commit}: git failed: {error.stderr.strip()}")
    if found != commit:
        raise Failed(f"{source['url']}: checked out {found}, not {commit}")
    _removed(folder, source)
    return f"{folder} at {commit[:12]}"


def _removed(folder, source):
    for inside in source.get("remove", []):
        path = os.path.join(folder, *inside.split("/"))
        if os.path.exists(path):
            os.remove(path)


# Licences


def licence_text(entry, token):
    """A licence's full text from the hub at the pinned revision, checked against its sum."""
    text = entry["licence_text"]
    with opened(text["url"], token) as reply:
        data = reply.read()
    if hashlib.sha256(data).hexdigest() != text["sha256"]:
        raise Failed(f"{text['url']}: the licence text's sha256 is not the registry's")
    return data.decode("utf-8", "replace")


def short_name(entry):
    """A model's name without its explanation ("TripoSG (shape), its weights run with our own code": "TripoSG
    (shape)")."""
    return entry["name"].split(", ")[0]


def shown(entry, size, token, models_dir):
    say()
    say("=" * 100)
    say(f"{entry['name']}  ({entry['role']}, {human(size)})")
    say(wrapped(entry["job"]))
    say(f"  Version:  {entry['version']}")
    say(f"  Licence:  {entry['licence']}   {entry['licence_url']}")
    say(textwrap.fill(entry.get("conditions", ""), width=100, initial_indent="  Terms:    ",
                      subsequent_indent="            "))
    if entry.get("source"):
        source = entry["source"]
        say(f"  Its code: {source['url']} at {source['commit'][:12]} ({source['licence']})")
    if entry.get("gated"):
        say(f"  The hub hands it out only to a signed-in account that has accepted its terms on "
            f"https://{HUB}/{next(iter(entry['folders']))}")
    if entry.get("licence_text"):
        # The whole text is long (SDXL's: 113 lines): written to a file to read, not scrolled past.
        kept = os.path.join(models_dir, LICENCE_FOLDER, f"{entry['id']}-LICENSE.md")
        os.makedirs(os.path.dirname(kept), exist_ok=True)
        with open(kept, "w") as file:
            file.write(licence_text(entry, token))
        say()
        say(wrapped(f"Its licence carries use restrictions that pass on to everything made with it. Read its whole "
                    f"text before you answer: {kept} (checked against the registry's sum), also at "
                    f"{entry['licence_text']['url']}"))
    else:
        local = SPDX_TEXT.get(entry["licence"])
        if local:
            say(f"  The {entry['licence']} text: {os.path.relpath(local, REPOSITORY)} in model-maker's folder "
                f"({REPOSITORY})")


def asked(question):
    try:
        with open("/dev/tty") as terminal:
            print(question, end=" ", flush=True)
            answer = terminal.readline()
    except OSError:
        print(question, flush=True)
        answer = sys.stdin.readline()
    return answer.strip().lower() in ("y", "yes")


def accepted_record(models_dir):
    path = os.path.join(models_dir, ".accepted")
    try:
        with open(path) as file:
            return path, json.load(file)
    except (OSError, ValueError):
        return path, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", default=os.environ.get("MODEL_MAKER_MODELS", os.path.join(REPOSITORY, "models")))
    parser.add_argument("--sources", default=os.environ.get("MODEL_MAKER_SOURCES",
                                                            os.path.join(REPOSITORY, "sources")))
    parser.add_argument("--yes", "--accept-licences", dest="yes", action="store_true",
                        help="accept every model's licence without asking (each is still shown)")
    parser.add_argument("--hf-token-file", help="a file holding a Hugging Face access token (for a gated model)")
    parser.add_argument("--only", nargs="*", help="these model ids only")
    parser.add_argument("--verify", action="store_true", help="only check the files there against the sums")
    options = parser.parse_args()
    options.models, options.sources = os.path.abspath(options.models), os.path.abspath(options.sources)

    entries = [m for m in models() if not options.only or m["id"] in options.only]
    os.makedirs(options.models, exist_ok=True)
    os.makedirs(options.sources, exist_ok=True)
    verified = Verified(options.models)
    if not options.verify:
        for folder in retired_sources_removed(options.sources):
            say(f"{folder}: code model-maker does not read, removed")
    token = None if options.verify else read_token(options.hf_token_file)
    record_path, accepted = accepted_record(options.models)
    started = time.time()
    to_ask = [e for e in entries if accepted.get(e["id"], {}).get("accepted") != f"{e['licence']} {e['version']}"]
    if to_ask and not options.verify and not options.yes:
        say(wrapped(f"{len(to_ask)} licence question{'' if len(to_ask) == 1 else 's'} follow, each just before its "
                    "model downloads, over about 20 minutes on a fast connection. To answer them all at once, read "
                    "the licences first (registry/models.toml names each; NOTICE) and run ./setup.sh --yes.", ""))
    declined, problems, done = [], [], []
    for entry in entries:
        parts = fetched_parts(entry, options.models)
        size = sum(p[3] for p in parts)
        if options.verify:
            problems += checked_or_fetched(parts, None, verified, verify_only=True)
            continue
        if entry.get("gated") and not token:
            say(f"\n{entry['name']}: the hub needs a signed-in account for it and no token was given "
                "(see 'A gated model' in docs/setup.md)")
            problems.append(f"{entry['id']}: no hub token")
            continue
        key = f"{entry['licence']} {entry['version']}"
        if accepted.get(entry["id"], {}).get("accepted") != key:
            shown(entry, size, token, options.models)
            what = "its weights" if "weights" in entry["name"] else "it"
            if not (options.yes or asked(f"Accept the {entry['licence']} terms of {short_name(entry)} and download "
                                         f"{what}? [y/N]")):
                declined.append(entry["id"])
                say(f"  {entry['name']}: not downloaded")
                continue
            accepted[entry["id"]] = {"accepted": key, "licence": entry["licence"],
                                     "on": datetime.date.today().isoformat(),
                                     "how": "--yes" if options.yes else "asked"}
            with open(record_path, "w") as file:
                json.dump(accepted, file, indent=1)
        say(f"\n{entry['name']}: {entry['licence']} accepted; {len(parts)} file{'' if len(parts) == 1 else 's'}, "
            f"{human(size)}")
        if entry.get("source"):
            try:
                say(f"  its code: {checked_out(entry['source'], options.sources)}")
            except Failed as error:
                problems.append(str(error))
        found = checked_or_fetched(parts, token, verified)
        problems += found
        if not found:
            done.append(entry["id"])
            say(f"  {entry['name']}: every file there and its sha256 the registry's")
    say()
    if options.verify:
        say("every file checked against the registry" if not problems else "problems:")
    for problem in problems:
        say(f"  PROBLEM: {problem}")
    missing = [m for m in NEEDED if m in declined or m not in done]
    if not options.verify and not options.only and missing:
        say(f"model-maker cannot make an object without: {', '.join(missing)}")
    if "siglip2-so400m" in declined:
        say("SigLIP 2 declined: the pictures are not checked to show the thing asked for, and a thing with a lid, a "
            "cover or doors cannot be judged shut or open, so it cannot be made (the rest can)")
    say(f"models took {time.time() - started:.0f} s")
    return 1 if problems or (missing and not options.verify and not options.only) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Failed as error:
        say(f"setup: {error}")
        sys.exit(1)
    except KeyboardInterrupt:
        say("\nstopped; run it again to carry on where it stopped")
        sys.exit(130)
