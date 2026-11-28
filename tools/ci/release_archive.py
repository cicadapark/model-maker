#!/usr/bin/env python3
"""One release archive for one system: the repository at a commit (as `git archive` makes it), the two prebuilt
programs, the third-party licence file, and the archive's sha256 beside it.

    python3 -I tools/ci/release_archive.py SYSTEM VERSION LAUNCHER SIMPLIFY LICENCES OUT_FOLDER [--commit HEAD]
        [--companion COMPANION]

SYSTEM is linux-x86_64, windows-x86_64 or macos-arm64. The programs go where the launcher looks for itself and its
tool: at the repository's top on Windows and macOS (on macOS the launcher replaces the bash `model-maker`, which
never ran there), in bin/ on Linux (setup.sh and the bash command stay the measured path there until the parity
check lets setup.sh hand over). Windows gets a .zip, the others a .tar.gz. The name says "unsigned" (no code-signing
yet: the sum is the check). The companion (the small window, docs/companion.md) goes beside the launcher, which it
starts. The standard library and git only.
"""

import argparse
import hashlib
import io
import os
import subprocess
import sys
import tarfile
import zipfile

LAYOUT = {"linux-x86_64": ("bin/", ""), "windows-x86_64": ("", ".exe"), "macos-arm64": ("", "")}


def repository_tar(commit, repository="."):
    return subprocess.run(["git", "-C", repository, "archive", "--format=tar", commit], check=True,
                          capture_output=True).stdout


def placed(system, launcher, simplify, licences, companion=None):
    """(path in the archive, file on disk, executable) for the files added to the repository's own."""
    folder, suffix = LAYOUT[system]
    files = [(f"{folder}model-maker{suffix}", launcher, True), (f"{folder}model-maker-simplify{suffix}", simplify, True),
             ("THIRD-PARTY-LICENCES.md", licences, False)]
    if companion:
        files.append((f"{folder}model-maker-companion{suffix}", companion, True))
    return files


def made(system, version, launcher, simplify, licences, out, commit="HEAD", repository=".", companion=None):
    top = f"model-maker-{version}"
    name = f"{top}-{system}-unsigned" + (".zip" if system.startswith("windows") else ".tar.gz")
    path = os.path.join(out, name)
    os.makedirs(out, exist_ok=True)
    extra = placed(system, launcher, simplify, licences, companion)
    replaced = {inside for inside, _, _ in extra}
    with tarfile.open(fileobj=io.BytesIO(repository_tar(commit, repository))) as source:
        members = [(m, source.extractfile(m).read() if m.isfile() else None) for m in source.getmembers()
                   if m.name not in replaced]
    if name.endswith(".zip"):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for member, data in members:
                if member.isfile():
                    archive.writestr(f"{top}/{member.name}", data)
            for inside, file, _ in extra:
                archive.write(file, f"{top}/{inside}")
    else:
        with tarfile.open(path, "w:gz") as archive:
            for member, data in members:
                member.name = f"{top}/{member.name}"
                archive.addfile(member, io.BytesIO(data) if data is not None else None)
            for inside, file, executable in extra:
                info = archive.gettarinfo(file, f"{top}/{inside}")
                info.mode = 0o755 if executable else 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with open(file, "rb") as data:
                    archive.addfile(info, data)
    with open(path, "rb") as file:
        digest = hashlib.sha256(file.read()).hexdigest()
    with open(path + ".sha256", "w") as file:
        file.write(f"{digest}  {name}\n")
    return path, digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("system", choices=sorted(LAYOUT))
    for name in ("version", "launcher", "simplify", "licences", "out"):
        parser.add_argument(name)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--companion", help="the companion program, put beside the launcher")
    options = parser.parse_args(argv)
    path, digest = made(options.system, options.version, options.launcher, options.simplify, options.licences,
                        options.out, options.commit, companion=options.companion)
    print(f"{digest}  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
