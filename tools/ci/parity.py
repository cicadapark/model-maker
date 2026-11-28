#!/usr/bin/env python3
"""Does the `model-maker` launcher set a repository up exactly as setup.sh does? (Before setup.sh hands over to it.)

    python3 -I tools/ci/parity.py manifest REPOSITORY > setup-sh.json     # after ./setup.sh ...
    python3 -I tools/ci/parity.py manifest REPOSITORY > launcher.json     # after model-maker setup ... (same folder)
    python3 -I tools/ci/parity.py compare setup-sh.json launcher.json

A manifest is every file under .venv and .tools, and model-maker-simplify, with its sha256 (a link: where it points),
leaving out only what tools/ci/parity-exclusions.txt lists, each with why; text files are compared with the
repository's own path written as <REPOSITORY>. `compare` lists every file that is only in one, or differs, and
exits 1 if there is any. The standard library only.
"""

import fnmatch
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXCLUSIONS = os.path.join(HERE, "parity-exclusions.txt")
ROOTS = (".venv", ".tools")
ALSO = ("target/release/model-maker-simplify", "target/release/model-maker-simplify.exe")
PLACEHOLDER = b"<REPOSITORY>"


def exclusions(path=EXCLUSIONS):
    """(pattern, why) for each line of the list."""
    with open(path) as file:
        lines = [line.rstrip("\n") for line in file if line.strip() and not line.startswith("#")]
    return [tuple(re.split(r"\s{2,}", line, maxsplit=1)) for line in lines]


def pattern_regex(pattern):
    """A list pattern as a regular expression: ** any number of parts, * within one part."""
    parts = []
    for piece in re.split(r"(\*\*/|/\*\*|\*)", pattern):
        if piece == "**/":
            parts.append(r"(?:.*/)?")
        elif piece == "/**":
            parts.append(r"(?:/.*)?")
        elif piece == "*":
            parts.append(r"[^/]*")
        else:
            parts.append(re.escape(piece))
    return re.compile("^" + "".join(parts) + "$")


def excluded(relative, compiled):
    return any(regex.match(relative) for regex in compiled)


def fingerprint(path, repository_bytes):
    if os.path.islink(path):
        return "link:" + os.readlink(path).replace(repository_bytes.decode(), PLACEHOLDER.decode())
    with open(path, "rb") as file:
        data = file.read()
    if repository_bytes in data:
        data = data.replace(repository_bytes, PLACEHOLDER)
    return hashlib.sha256(data).hexdigest()


def manifest(repository):
    repository = os.path.realpath(repository)
    repository_bytes = repository.encode()
    compiled = [pattern_regex(pattern) for pattern, _ in exclusions()]
    files = {}
    for root in ROOTS:
        for folder, folders, names in os.walk(os.path.join(repository, root)):
            for name in names + [f for f in folders if os.path.islink(os.path.join(folder, f))]:
                path = os.path.join(folder, name)
                relative = os.path.relpath(path, repository).replace(os.sep, "/")
                if not excluded(relative, compiled):
                    files[relative] = fingerprint(path, repository_bytes)
    for relative in ALSO:
        if os.path.exists(os.path.join(repository, relative)):
            files[relative] = fingerprint(os.path.join(repository, relative), repository_bytes)
    return files


def differences(first, second):
    only_first = sorted(set(first) - set(second))
    only_second = sorted(set(second) - set(first))
    changed = sorted(name for name in set(first) & set(second) if first[name] != second[name])
    return only_first, only_second, changed


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["manifest"] and len(argv) == 2:
        json.dump(manifest(argv[1]), sys.stdout, indent=0, sort_keys=True)
        return 0
    if argv[:1] == ["compare"] and len(argv) == 3:
        with open(argv[1]) as a, open(argv[2]) as b:
            first, second = json.load(a), json.load(b)
        only_first, only_second, changed = differences(first, second)
        for label, names in ((f"only in {argv[1]}", only_first), (f"only in {argv[2]}", only_second),
                             ("different", changed)):
            for name in names:
                print(f"{label}: {name}")
        print(f"{len(first)} and {len(second)} files compared; {len(only_first) + len(only_second) + len(changed)} "
              f"differences")
        return 1 if only_first or only_second or changed else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
