#!/usr/bin/env python3
"""The Python environment's lock per platform: made, checked and its wheels' licences read, all from Linux.

python/requirements.lock is the measured one (Linux x86_64 with an NVIDIA card, CUDA 12.8) and is never written here.
The others are made from python/requirements.txt with the measured lock's versions, changed only where TARGETS says
why, each compiled by uv for its platform (`--python-platform`), so every one is made and checked on one machine:

    python3 tools/ci/locks.py make [NAME ...]       write the locks (not the measured one)
    python3 tools/ci/locks.py check [NAME ...]      compile each again and compare; a wheel for its platform for each
    python3 tools/ci/locks.py licences [NAME ...]   every bundled library of the wheels each lock installs, judged by
                                                    python/model_maker/check_licences.py's tables (nothing installed)
    python3 tools/ci/locks.py audit-view NAME       the lock as the vulnerability audit reads it (AUDITED_AS)
    python3 tools/ci/locks.py wheels NAME FOLDER    the lock's wheels for its platform into FOLDER, each by its locked
                                                    sum, with SHA256SUMS: for `model-maker setup --wheels` on a PC
                                                    without internet

`uv` is taken from UV (else the PATH): setup.sh's pinned version. `check` and `licences` read the package index
(pypi.org, and download.pytorch.org for the PyTorch builds named below) and exit 1 on any problem.
"""

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

REPOSITORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPOSITORY, "python"))

from model_maker import check_licences  # noqa: E402

REQUIREMENTS = os.path.join(REPOSITORY, "python", "requirements.txt")
MEASURED = "linux-x86_64-cuda128"
PYTHON = "3.11"
PYTORCH_FILES = "https://download.pytorch.org/whl"

def pytorch_file(path, sha256):
    """A file of PyTorch's index (under PYTORCH_FILES) and its sha256, as that index lists them."""
    return path, sha256


# Each platform: uv's name for it, the wheel tags it takes (best first), the system check_licences judges it as, the
# PyTorch builds taken from PyTorch's own index (file and sha256, as that index lists them; elsewhere from PyPI), and
# the measured lock's versions it changes, each with why.
TARGETS = {
    MEASURED: dict(
        lock="python/requirements.lock", uv="x86_64-manylinux_2_28", system="linux",
        about="Linux x86_64 with an NVIDIA card (CUDA 12.8): the measured lock, which setup.sh installs",
        torch={}, changes={}),
    "windows-x86_64-cuda128": dict(
        lock="python/locks/windows-x86_64-cuda128.lock", uv="x86_64-pc-windows-msvc", system="win32",
        about="Windows x86_64 with an NVIDIA card (CUDA 12.8; PyPI's Windows torch is the processor's build only)",
        torch={"torch": pytorch_file("cu128/torch-2.10.0%2Bcu128-cp311-cp311-win_amd64.whl",
                                     sha256="3523fda6e2cfab2b04ae09b1424681358e508bb3faa11ceb67004113d5e7acad"),
               "torchvision": pytorch_file("cu128/torchvision-0.25.0%2Bcu128-cp311-cp311-win_amd64.whl",
                                           sha256="af00b4e0cdb3f490f4393e9a335b622fe1b92fd5afb181033256ccba03b9637c")},
        changes={"c2pa-python": ("0.39.0", "0.38.1 has no Windows wheel; 0.39.0 is the first that has one")}),
    "macos-arm64": dict(
        lock="python/locks/macos-arm64.lock", uv="aarch64-apple-darwin", macos="14.0", system="darwin",
        about="macOS 14 or newer on Apple Silicon (PyTorch's Metal build, from PyPI)",
        torch={}, changes={"bitsandbytes": ("0.50.2", "0.46.1 has no macOS wheel; 0.50.2 has one for macOS 14+ on "
                                                       "Apple Silicon (FLUX.1's 4-bit loading there is untried)")}),
    "linux-x86_64-rocm71": dict(
        lock="python/locks/linux-x86_64-rocm71.lock", uv="x86_64-manylinux_2_28", system="linux",
        about="Linux x86_64 with an AMD card (ROCm 7.1, the newest PyTorch 2.10 is built for)",
        torch={"torch": pytorch_file("rocm7.1/torch-2.10.0%2Brocm7.1-cp311-cp311-manylinux_2_28_x86_64.whl",
                                     sha256="958298b19aceed29a9f3579ef19859c6fa6b7d2a527a67160d8ad5c52e8860e1"),
               "torchvision": pytorch_file("rocm7.1/torchvision-0.25.0%2Brocm7.1-cp311-cp311-manylinux_2_28_x86_64.whl",
                                           sha256="e79577ea367ed1652d70bb18f4dcf97f0e6aa17b503a28020be7dee0895347eb"),
               # PyTorch's index lists this file without a sum; this one is the downloaded file's.
               "triton-rocm": pytorch_file(
                   "triton_rocm-3.6.0-cp311-cp311-linux_x86_64.whl",
                   sha256="3286c59eb97e65ab705e207689b6a47807cb73a27ce53e9e774e46bab01318fe")},
        changes={"bitsandbytes": ("0.50.2", "0.46.1 is built for CUDA and the processor only; 0.50.2 lists AMD cards "
                                            "(untried)")},
        refused=("its PyTorch build bundles libraries under the LGPL (libnuma; libelf, LGPL or GPL), which the licence "
                 "rule refuses, and about 45 of AMD's own libraries and kernel files, libtinfo and MAGMA, which nobody "
                 "has read yet; so it is never installed (not by setup, not by CI) until a ROCm route without them is "
                 "found or the rule says otherwise. It is kept to show the route resolves, and is checked like the "
                 "others")),
    "linux-x86_64-cpu": dict(
        lock="python/locks/linux-x86_64-cpu.lock", uv="x86_64-manylinux_2_28", system="linux",
        about="Linux x86_64 without a card (PyTorch's processor build: CI's tiny-model smoke test)",
        torch={"torch": pytorch_file("cpu/torch-2.10.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl",
                                     sha256="b7cb1ec66cefb90fd7b676eac72cfda3b8d4e4d0cacd7a531963bc2e0a9710ab"),
               "torchvision": pytorch_file("cpu/torchvision-0.25.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl",
                                           sha256="aa016ab73e06a886f72edc8929ed2ed4c85aaaa6e10500ecdef921b03129b19e")},
        changes={}),
}

ENTRY = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:==(\S+)| @ (\S+))", re.MULTILINE)
SHA256 = re.compile(r"--hash=sha256:([0-9a-f]{64})")


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def parsed(text):
    """{name: (version or URL, {sha256})} of a lock's text."""
    found = {}
    for block in re.split(r"\n(?=\S)", text):
        match = ENTRY.match(block)
        if match:
            found[normalized(match.group(1))] = (match.group(2) or match.group(3), set(SHA256.findall(block)))
    return found


def read_lock(name):
    with open(os.path.join(REPOSITORY, TARGETS[name]["lock"])) as file:
        return file.read()


def torch_url(path, sha256):
    return f"{PYTORCH_FILES}/{path}#sha256={sha256}"


# The vulnerability audit (pip-audit) looks a package up by its PyPI release, which it cannot read from a file's
# address: a PyTorch build is audited as the release it is built from (2.10.0+cu128 as torch 2.10.0), and ROCm's build
# of Triton as Triton (triton-rocm is not on PyPI).
AUDITED_AS = {"triton-rocm": "triton"}


def audit_view(name):
    """The lock as pip-audit reads it: each PyTorch file's line as `release==version`, its sums kept."""
    from packaging.utils import parse_wheel_filename

    def release(match):
        file_name = urllib.parse.unquote(match.group(2).split("#")[0].rsplit("/", 1)[-1])
        package, version = parse_wheel_filename(file_name)[:2]
        return f"{AUDITED_AS.get(normalized(package), normalized(package))}=={version.public}"

    return re.sub(r"^([A-Za-z0-9][A-Za-z0-9._-]*) @ (\S+)", release, read_lock(name), flags=re.MULTILINE)


# The package index: each version's files, by sha256.

_FILES = {}
TRIES = 4


def opened(request, timeout=30):
    """urlopen, tried again after a network error (the index is far away; CI's runners see timeouts)."""
    import time

    for attempt in range(TRIES):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code < 500 or attempt == TRIES - 1:
                raise
            time.sleep(2 ** attempt)


def index_files(name, version):
    """{sha256: (file name, url)} of a PyPI release's files."""
    key = (normalized(name), version)
    if key not in _FILES:
        url = f"https://pypi.org/pypi/{urllib.parse.quote(key[0])}/{urllib.parse.quote(version)}/json"
        with opened(url) as response:
            release = json.load(response)
        _FILES[key] = {file["digests"]["sha256"]: (file["filename"], file["url"]) for file in release["urls"]}
    return _FILES[key]


def files_of(name, pinned, hashes):
    """{sha256: (file name, url)} of a lock entry's files: a PyTorch build's one file, or the PyPI release's files
    with those sums (a sum the index does not serve maps to None)."""
    if "://" in pinned:
        url = pinned.split("#")[0]
        return {sha: (urllib.parse.unquote(url.rsplit("/", 1)[1]), url) for sha in hashes}
    known = index_files(name, pinned)
    return {sha: known.get(sha) for sha in hashes}


# Compiling.

def constraints(target):
    """The measured lock's versions, with the target's changes; its PyTorch builds left to their files."""
    pins = {name: version for name, (version, _) in parsed(read_lock(MEASURED)).items()}
    pins.update({normalized(name): version for name, (version, _) in target["changes"].items()})
    return [f"{name}=={version}" for name, version in sorted(pins.items()) if name not in target["torch"]]


def requirements(target):
    """requirements.txt for the target: torch and torchvision from their PyTorch files where it names them, and
    torchvision (BiRefNet's backbone) everywhere, at the measured lock's version if requirements.txt leaves it out."""
    with open(REQUIREMENTS) as file:
        lines = [line.split("#")[0].strip() for line in file]
    wanted = [line for line in lines if line]
    if not any(normalized(re.split(r"[=<>!~ ;@]", line, maxsplit=1)[0]) == "torchvision" for line in wanted):
        wanted.append(f"torchvision=={parsed(read_lock(MEASURED))['torchvision'][0]}")
    torch_files = target["torch"]
    chosen = []
    for line in wanted:
        name = normalized(re.split(r"[=<>!~ ;]", line, maxsplit=1)[0])
        if name in torch_files:
            line = f"{name} @ {torch_url(*torch_files[name])}"
        elif name in target["changes"]:
            line = f"{name}=={target['changes'][name][0]}"
        chosen.append(line)
    chosen += [f"{name} @ {torch_url(*file)}" for name, file in torch_files.items()
               if not any(line.startswith(name + " @") for line in chosen)]
    return chosen


def uv_command(target, folder):
    command = [os.environ.get("UV", "uv"), "pip", "compile", os.path.join(folder, "requirements.in"),
               "--constraint", os.path.join(folder, "constraints.txt"), "--python-version", PYTHON,
               "--python-platform", target["uv"], "--only-binary", ":all:", "--generate-hashes", "--no-header",
               "--no-annotate", "--quiet", "--no-config", "--index-url", "https://pypi.org/simple"]
    return command


def compiled(name):
    """The lock uv makes for `name` now, without its header."""
    target = TARGETS[name]
    with tempfile.TemporaryDirectory() as folder:
        with open(os.path.join(folder, "requirements.in"), "w") as file:
            file.write("\n".join(requirements(target)) + "\n")
        with open(os.path.join(folder, "constraints.txt"), "w") as file:
            file.write("\n".join(constraints(target)) + "\n")
        environment = dict(os.environ, UV_NO_CONFIG="1")
        if target.get("macos"):
            environment["MACOSX_DEPLOYMENT_TARGET"] = target["macos"]
        made = subprocess.run(uv_command(target, folder), env=environment, check=True, capture_output=True,
                              text=True).stdout
    return made


def header(name):
    import textwrap

    target = TARGETS[name]
    platform = f"--python-platform {target['uv']}" + (
        f" with MACOSX_DEPLOYMENT_TARGET={target['macos']}" if target.get("macos") else "")
    text = [f"model-maker's Python environment for {target['about']}: every package pinned with the sha256 of each "
            f"of its wheels, Python {PYTHON}. Not measured: python/requirements.lock is the measured lock, and this "
            f"one has its versions except where said below.",
            f"Made by `python3 tools/ci/locks.py make {name}`: uv pip compile of python/requirements.txt with "
            f"torchvision, {platform}, --only-binary :all:, --generate-hashes, the measured lock's versions as "
            f"constraints. CI checks that this is what the same command makes today "
            f"(`locks.py check`) and reads its wheels' licences (`locks.py licences`)."]
    if target["torch"]:
        text.append("From PyTorch's own index: " + ", ".join(
            urllib.parse.unquote(path.rsplit("/", 1)[-1]) for path, _ in target["torch"].values()) + ".")
    text += [f"{package}=={version}, not the measured lock's: {why}."
             for package, (version, why) in target["changes"].items()]
    if target.get("refused"):
        text.append(f"REFUSED by the licence check: {target['refused']}.")
    text.append("Never: OpenCV (its wheels bundle LGPL code), diso (non-commercial), pymeshlab (GPL), nvdiffrast "
                "(non-commercial).")
    return "".join(textwrap.fill(paragraph, 118, initial_indent="# ", subsequent_indent="# ") + "\n"
                   for paragraph in text)


def make(names):
    for name in names:
        if name == MEASURED:
            print(f"{name}: the measured lock is not written here")
            continue
        path = os.path.join(REPOSITORY, TARGETS[name]["lock"])
        text = header(name) + compiled(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            file.write(text)
        print(f"{name}: {TARGETS[name]['lock']} written")


# Checking.

def tags_of(target):
    """The wheel tags the target takes, best first."""
    from packaging import tags

    interpreter = tuple(int(part) for part in PYTHON.split("."))
    if target["system"] == "win32":
        platforms = ["win_amd64"]
    elif target["system"] == "darwin":
        major = int(target["macos"].split(".")[0])
        platforms = list(tags.mac_platforms((major, 0), "arm64"))
    else:
        glibc = int(target["uv"].rsplit("_", 1)[1])
        platforms = [f"manylinux_2_{minor}_x86_64" for minor in range(glibc, 4, -1)]
        platforms += ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
    return list(tags.cpython_tags(interpreter, platforms=platforms)) + list(
        tags.compatible_tags(interpreter, "cp311", platforms=platforms))


def chosen_wheels(name):
    """{package: (file name, url)}: the wheel the target installs for each locked package (its best tag among the
    locked sums), and the problems: a package with no wheel for the platform, a sum the index does not serve."""
    from packaging.utils import parse_wheel_filename

    ranks = {tag: rank for rank, tag in enumerate(tags_of(TARGETS[name]))}
    chosen, problems = {}, []
    for package, (pinned, hashes) in sorted(parsed(read_lock(name)).items()):
        files = files_of(package, pinned, hashes)
        unknown = [sha for sha, file in files.items() if file is None]
        if unknown:
            problems.append(f"{name}: {package} {pinned}: the index serves no file with sum {unknown[0]}")
        best = None
        for file in filter(None, files.values()):
            if not file[0].endswith(".whl"):
                continue
            rank = min((ranks[tag] for tag in parse_wheel_filename(file[0])[3] if tag in ranks), default=None)
            if rank is not None and (best is None or rank < best[0]):
                best = (rank, file)
        if best is None:
            problems.append(f"{name}: {package} {pinned}: no wheel for {TARGETS[name]['uv']}")
        else:
            chosen[package] = best[1]
    return chosen, problems


def sums_problems(name, package, pinned, locked, made):
    """A sum the lock's recipe no longer gives (the index serves another file) is a problem; so is a new wheel (an
    installer may choose it and then refuse its sum). A new source archive is not: `--only-binary` never takes one."""
    problems = [f"{name}: {package} {pinned}: the index no longer gives the locked sum {sha}" for sha in locked - made]
    if made - locked:
        files = files_of(package, pinned, made - locked)
        problems += [f"{name}: {package} {pinned}: a new file not in the lock: {file[0] if file else sha}"
                     for sha, file in files.items() if not (file and file[0].endswith((".tar.gz", ".zip")))]
    return problems


def check(names):
    problems = []
    for name in names:
        locked, made = parsed(read_lock(name)), parsed(compiled(name))
        for package in sorted(set(locked) | set(made)):
            if package not in made:
                problems.append(f"{name}: {package} is locked but no longer needed")
            elif package not in locked:
                problems.append(f"{name}: {package} {made[package][0]} is needed but not locked")
            elif locked[package][0] != made[package][0]:
                problems.append(f"{name}: {package} {locked[package][0]} is locked, the recipe makes "
                                f"{made[package][0]}")
            else:
                problems += sums_problems(name, package, *locked[package], made[package][1])
        chosen, missing = chosen_wheels(name)
        problems += missing
        print(f"{name}: {len(locked)} packages, a wheel for {TARGETS[name]['uv']} for {len(chosen)}")
    return problems


# Licences, read from the wheels themselves (their zip directory and METADATA, by ranged requests).

# Wheels read at once (each is a few ranged requests).
READERS = 8


class RemoteFile(io.RawIOBase):
    """A file on a web server read by byte ranges, so a wheel's zip directory is read without downloading it."""

    def __init__(self, url):
        request = urllib.request.Request(url, method="HEAD")
        with opened(request) as response:
            self.size, self.url = int(response.headers["Content-Length"]), response.url
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self.position = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence] + offset
        return self.position

    def readinto(self, buffer):
        if self.position >= self.size or not len(buffer):
            return 0
        last = min(self.size, self.position + len(buffer)) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{last}"})
        with opened(request) as response:
            data = response.read()
        if response.status != 206 and self.position:
            raise OSError(f"{self.url}: the server does not serve byte ranges")
        buffer[:len(data)] = data
        self.position += len(data)
        return len(data)


class WheelDistribution:
    """What check_licences.problems_of reads of an installed package, read from a wheel's zip instead."""

    def __init__(self, url):
        with zipfile.ZipFile(io.BufferedReader(RemoteFile(url), buffer_size=1 << 20)) as wheel:
            names = wheel.namelist()
            metadata = next(n for n in names if re.fullmatch(r"[^/]+\.dist-info/METADATA", n))
            from email.parser import HeaderParser

            self.metadata = HeaderParser().parsestr(wheel.read(metadata).decode("utf-8", "replace"))
            # The files as RECORD lists them, as an installed environment's check reads them (importlib.metadata):
            # a wheel built on Windows may write its paths there with backslashes (torchvision's does), which the
            # zip's own names never have.
            record = metadata.rsplit("/", 1)[0] + "/RECORD"
            if record in names:
                import csv

                rows = csv.reader(io.StringIO(wheel.read(record).decode("utf-8", "replace")))
                listed = [row[0] for row in rows if row]
            else:
                listed = names
        self.files = [_WheelFile(name) for name in listed]


class _WheelFile(str):
    @property
    def name(self):
        return self.rsplit("/", 1)[-1]


def licences(names, show=False):
    problems = []
    for name in names:
        system = TARGETS[name]["system"]
        chosen, missing = chosen_wheels(name)
        problems += missing
        libraries = 0
        with ThreadPoolExecutor(READERS) as pool:
            read = dict(zip(sorted(chosen), pool.map(lambda package: WheelDistribution(chosen[package][1]),
                                                     sorted(chosen))))
        found = []
        for package, distribution in read.items():
            with _judged_as(system):
                found += [f"{name}: {problem}" for problem in check_licences.problems_of(distribution)]
                bundled = list(check_licences.bundled_libraries(distribution))
            libraries += len(bundled)
            if show:
                for path, library in bundled:
                    verdict, licence = check_licences.judged_library(path, library)
                    print(f"{name:24s} {package:24s} {library:48s} {verdict:8s} {licence}")
        if not check_licences.system_known(system):
            found.append(f"{name}: check_licences.py does not know {system}'s libraries")
        print(f"{name}: {len(chosen)} wheels read, {libraries} bundled libraries")
        if TARGETS[name].get("refused"):
            # Its problems are its recorded refusal; a refused lock that passes has a stale record.
            print(f"{name}: refused, as recorded ({len(found)} problems, listed above with --list)")
            if not found:
                problems.append(f"{name}: passes the licence check now: remove its refusal from TARGETS")
            continue
        problems += found
    return problems


class _judged_as:
    """check_licences judges file names as on `system` (its `is_library` reads sys.platform by default)."""

    def __init__(self, system):
        self.system = system

    def __enter__(self):
        self.was = check_licences.is_library.__defaults__
        check_licences.is_library.__defaults__ = (self.system,)

    def __exit__(self, *_):
        check_licences.is_library.__defaults__ = self.was


# A wheel folder for a PC without internet.

WHEELS_SUMS = "SHA256SUMS"


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def downloaded(url, path):
    """`url` into `path` (written beside it first, so a broken download never looks like a wheel)."""
    part = path + ".part"
    with opened(url, timeout=120) as response, open(part, "wb") as file:
        for block in iter(lambda: response.read(1 << 20), b""):
            file.write(block)
    os.replace(part, path)


def wheels(name, folder):
    """The wheel `name`'s platform installs for each locked package, into `folder`, each checked against the lock's
    sums (one already there with a right sum is kept, not fetched again), and `folder`/SHA256SUMS; other wheels in it
    are removed. `setup --wheels` hands the folder (or its address) to uv as --find-links with --no-index, and pins a
    package the lock names by its address (PyTorch's builds) by its version there, so it too comes from the folder
    (uv takes an address pin from its address or its own cache only: offline, with the cache gone, it refused one
    already installed; tried 28 Nov 2026). The problems, as `check` gives them."""
    chosen, problems = chosen_wheels(name)
    locked = parsed(read_lock(name))
    os.makedirs(folder, exist_ok=True)
    kept = {}
    for package, (file_name, url) in sorted(chosen.items()):
        pinned, sums = locked[package]
        path = os.path.join(folder, file_name)
        if not (os.path.isfile(path) and sha256_of(path) in sums):
            downloaded(url, path)
            if sha256_of(path) not in sums:
                os.remove(path)
                problems.append(f"{name}: {package} {pinned}: {file_name} came with a sum the lock does not hold")
                continue
        kept[file_name] = sha256_of(path)
    for stale in sorted(set(os.listdir(folder)) - set(kept) - {WHEELS_SUMS}):
        if stale.endswith((".whl", ".part")):
            os.remove(os.path.join(folder, stale))
            print(f"removed {stale} (not this lock's)")
    with open(os.path.join(folder, WHEELS_SUMS), "w") as file:
        file.writelines(f"{sha}  {file_name}\n" for file_name, sha in sorted(kept.items()))
    print(f"{name}: {len(kept)} wheels in {folder}, each by its locked sum")
    return problems


def main(argv):
    if not argv or argv[0] not in ("make", "check", "licences", "audit-view", "wheels"):
        print(__doc__)
        return 2
    if argv[0] == "wheels":
        if len(argv) != 3 or argv[1] not in TARGETS:
            print(f"usage: locks.py wheels NAME FOLDER (NAME one of {list(TARGETS)})")
            return 2
        problems = wheels(argv[1], argv[2])
        for problem in problems:
            print(problem)
        print(f"{len(problems)} problems")
        return 1 if problems else 0
    command, names = argv[0], [n for n in argv[1:] if not n.startswith("--")] or list(TARGETS)
    unknown = [n for n in names if n not in TARGETS]
    if unknown:
        print(f"unknown locks: {unknown}; known: {list(TARGETS)}")
        return 2
    if command == "make":
        make(names)
        return 0
    if command == "audit-view":
        sys.stdout.write(audit_view(names[0]))
        return 0
    problems = check(names) if command == "check" else licences(names, show="--list" in argv)
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
