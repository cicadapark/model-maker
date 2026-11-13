"""Bug reports: one file that holds what we need to check a problem, and nothing personal; and the remake that checks
it. Only Python's standard library.

    model-maker bug-report JOB_FOLDER [--out FILE] [--with-model] [--ledger FILE]
    model-maker reproduce REPORT_FILE --out DIR

Nothing made on a user's own machine can be taken as proof, as they hold the code. So a report is checked by making
its prompt again, with its seeds, settings and version, on a maintainer's card, and comparing. The report file
carries what that needs: the job's record (job.json) and sidecar, its report page and pictures, the chosen picture
and its cut-out, the job's ledger lines, and the versions it was made with (the code, the models, Python, torch, the
card).

It leaves out the large in-between files (.npz), the other tries' pictures, the .glb unless asked (--with-model), the
environment and anything that looks like a token. In every text it copies, the user's home folder, user name, machine
name, e-mail addresses, network addresses (IPv4 and IPv6) and other absolute paths are replaced by placeholders, and
the file is written only when a check of every copied text finds none of them left. Pictures go as PNGs with their
text and metadata chunks dropped, but their content credentials (C2PA: what made the picture, signed; never a name or
a path model-maker wrote) kept; a picture of another kind (.jpg, .webp) is left out. The prompt is sent whole: it is
what the remake makes.
"""

import dataclasses
import datetime
import getpass
import importlib.metadata
import io
import json
import os
import platform
import re
import socket
import struct
import sys
import tarfile
import zipfile
import zlib

from . import ledger, registry

FORMAT = 1
ROOT = "bug-report"
# GitHub takes an issue's attached file up to this size.
ATTACHMENT_LIMIT_MB = 25
# A member of a report read by `reproduce` is never larger than this (a bomb or a mistake: refused).
LARGEST_MEMBER_MB = 64
# The job's folders not sent, and why.
FOLDERS_LEFT_OUT = {"pictures": "every try's picture and cut-out (the report shows them small)",
                    "views": "the painted views (the report shows them small)"}
TEXT_ENDINGS = (".json", ".jsonl", ".html", ".htm", ".txt", ".csv", ".md", ".svg")
# The packages whose versions decide what is made, besides torch (read from their metadata: nothing is imported).
PACKAGES = ("torch", "diffusers", "transformers", "accelerate", "numpy", "pillow", "trimesh")
# Names too common to replace as whole words (a user called "user" has "/home/user" replaced all the same).
GENERIC_NAMES = {"root", "user", "users", "admin", "administrator", "guest", "home", "public", "shared", "default",
                 "ubuntu", "runner", "localhost", "local", "linux", "desktop", "computer", "pc", "me", "nobody"}
# PNG chunks kept: those that say how to show the pixels. Text, EXIF and times (where tools write names) are dropped.
PNG_KEPT_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"gAMA", b"cHRM", b"sRGB", b"iCCP", b"sBIT", b"pHYs",
                   b"bKGD"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# A PNG's content credentials (C2PA's manifest store): kept, as long as every other chunk is, since its signature holds
# for the file's bytes as they were signed (with any chunk dropped, it would no longer hold and is dropped too).
PNG_CREDENTIALS_CHUNK = b"caBX"

HOME_PATH = re.compile(r"(?<![\w.<>-])(?:/home|/Users|/var/home|/export/home)/([^/\\\s\"'<>`]+)"
                       r"|(?<![\w.<>-])/root(?![\w.-])"
                       r"|\b[A-Za-z]:(?:\\\\|\\|/)(?:Users|Documents and Settings)(?:\\\\|\\|/)([^/\\\s\"'<>`]+)")
# Any other absolute path of three parts or more (not a web address's): its folders could name someone.
OTHER_PATH = re.compile(r"(?<![\w.:/<>\\-])(?:/[\w.@~+-]+){2,}/([\w.@~+-]+)")
EMAIL = re.compile(r"\b[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b")
# Not a pinned version ("==1.2.3.4"); loopback and "any address" are no one's.
IPV4 = re.compile(r"(?<![\w.])(?<![=<>~]=)(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![\w.])")
HARMLESS_ADDRESSES = {"127.0.0.1", "0.0.0.0"}
# An IPv6 address: eight groups, or fewer with "::" (no time of day, "14:26:17", has either); not inside a longer word
# or address. Loopback and "any address" are no one's.
IPV6 = re.compile(r"(?<![\w:.])(?:(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}"
                  r"|(?:[0-9a-f]{1,4}:){1,7}:(?:[0-9a-f]{1,4}(?::[0-9a-f]{1,4}){0,6})?"
                  r"|::[0-9a-f]{1,4}(?::[0-9a-f]{1,4}){0,6})(?![\w:])", re.IGNORECASE)
HARMLESS_IPV6 = {"::1"}
SECRETS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{20,}"),
]
SECRET_VARIABLE = re.compile(r"TOKEN|SECRET|PASSW|API_?KEY|ACCESS_?KEY|PRIVATE|AUTH|CREDENTIAL|COOKIE|SESSION",
                             re.IGNORECASE)
# What each placeholder stands for, in the words the command prints.
PLACEHOLDERS = {"<job>": "the job's folder", "<out>": "the folder it was made in", "<home>": "a home folder",
                "<model-maker>": "model-maker's own folder", "<models>": "the models' folder",
                "<sources>": "the models' code folder", "<ledger>": "the ledger's folder", "<here>": "the folder run in",
                "<user>": "a user name", "<machine>": "the machine's name", "<path>": "other absolute paths",
                "<email>": "an e-mail address", "<address>": "a network address", "<token removed>": "a token"}
README = """\
A model-maker bug report
========================

This file was made by `model-maker bug-report` for a problem report (a GitHub issue).

What is inside (folder bug-report/):
  README.txt            this text
  about.json            the versions: model-maker's code, the AI models (their ids and hub revisions), Python,
                        torch and other packages, the graphics card and its driver (this machine's, and the card
                        the job was made on); and what was left out or replaced
  job/job.json          the job's record: the prompt, the options, the settings, the seeds, every try's checks
  job/*.provenance.json the sidecar: every model used, the seeds, every check's value
  job/picture.png       the picture chosen, and job/mask.png its cut-out
  job/report/           the report page (report.html) and its pictures
  job/ledger.jsonl      the job's lines from the ledger (times, the checks, the card)
{model_line}
What was left out:
{left_out}
What was replaced in the texts (so nothing personal is sent):
{replaced}
How we check it:
  We claim no rights in what you make; the models' terms apply (NOTICE). We cannot take a model made on your own
  machine as proof of anything: the code that made it is on your machine. So we check a report by remaking it. We
  make the same prompt with the same seeds, the same settings and the same version on a maintainer's card
  (`model-maker reproduce`), and compare the picture, every check's value and the verdict with yours.
"""


@dataclasses.dataclass
class Replacement:
    """What a text had replaced: how many of each placeholder."""

    counts: dict = dataclasses.field(default_factory=dict)

    def add(self, placeholder, count=1):
        if count:
            self.counts[placeholder] = self.counts.get(placeholder, 0) + count


class Scrubber:
    """Replaces what is personal in a text by placeholders (`clean`) and finds any left (`leaks`).

    `paths`: (absolute path, placeholder), each replaced wherever it starts a path; the longest first, so a job's
    folder is "<job>", not "<home>/...". `users` and `machines`: names replaced as whole words (letters and digits
    round them end the word; "alice-files" holds "alice"). A name that is also a word of the prompt (`kept_words`) is
    replaced only where it is part of a path or an address, so the prompt, which the remake needs, is kept whole.
    `secrets`: values (tokens from the environment) replaced wherever they are."""

    def __init__(self, paths=(), users=(), machines=(), kept_words=(), secrets=()):
        self.paths = []
        for path, placeholder in paths:
            for variant in path_variants(path):
                self.paths.append((variant, placeholder))
        self.paths.sort(key=lambda pair: -len(pair[0]))
        kept = {word.casefold() for word in kept_words}
        self.names = []
        self.names_in_paths_only = []
        for names, placeholder in ((users, "<user>"), (machines, "<machine>")):
            for name in sorted({n for n in names if usable_name(n)}, key=len, reverse=True):
                pattern = name_pattern(name, name.casefold() in kept)
                (self.names_in_paths_only if name.casefold() in kept else self.names).append((pattern, placeholder))
        self.secrets = sorted({s for s in secrets if s and len(s) >= 8}, key=len, reverse=True)
        self.replaced = Replacement()

    def clean(self, text):
        for pattern in SECRETS:
            text, count = pattern.subn("<token removed>", text)
            self.replaced.add("<token removed>", count)
        for secret in self.secrets:
            count = text.count(secret)
            text = text.replace(secret, "<token removed>")
            self.replaced.add("<token removed>", count)
        for path, placeholder in self.paths:
            text, count = exact_path(path).subn(placeholder, text)
            self.replaced.add(placeholder, count)
        text, count = HOME_PATH.subn("<home>", text)
        self.replaced.add("<home>", count)
        text, count = OTHER_PATH.subn(lambda m: "<path>/" + m.group(1), text)
        self.replaced.add("<path>", count)
        text, count = EMAIL.subn("<email>", text)
        self.replaced.add("<email>", count)
        self.replaced.add("<address>", sum(m.group(0) not in HARMLESS_ADDRESSES for m in IPV4.finditer(text)))
        text = IPV4.sub(lambda m: m.group(0) if m.group(0) in HARMLESS_ADDRESSES else "<address>", text)
        self.replaced.add("<address>", sum(m.group(0) not in HARMLESS_IPV6 for m in IPV6.finditer(text)))
        text = IPV6.sub(lambda m: m.group(0) if m.group(0) in HARMLESS_IPV6 else "<address>", text)
        for pattern, placeholder in self.names + self.names_in_paths_only:
            text, count = pattern.subn(placeholder, text)
            self.replaced.add(placeholder, count)
        return text

    def leaks(self, text):
        """What personal things a text still holds: [(what, where in the text)] (the values are never shown)."""
        found = []
        for pattern in SECRETS:
            found += [("a token", m.start()) for m in pattern.finditer(text)]
        found += [("a token from the environment", text.index(s)) for s in self.secrets if s in text]
        for path, _ in self.paths:
            found += [("a folder of this machine", m.start()) for m in exact_path(path).finditer(text)]
        found += [("a home folder", m.start()) for m in HOME_PATH.finditer(text)]
        found += [("an absolute path", m.start()) for m in OTHER_PATH.finditer(text)]
        for pattern, placeholder in self.names + self.names_in_paths_only:
            found += [(PLACEHOLDERS[placeholder], m.start()) for m in pattern.finditer(text)]
        found += [("an e-mail address", m.start()) for m in EMAIL.finditer(text)]
        found += [("a network address", m.start()) for m in IPV4.finditer(text)
                  if m.group(0) not in HARMLESS_ADDRESSES]
        found += [("a network address", m.start()) for m in IPV6.finditer(text) if m.group(0) not in HARMLESS_IPV6]
        return found


def exact_path(path):
    """A path where it starts one (not inside a longer path or an address) and ends at a separator or the text's end."""
    return re.compile(r"(?<![\w.:/\\-])" + re.escape(path) + r"(?![^/\\\s\"'<>`,;)\]}])")


def path_variants(path):
    """A path as texts may hold it: as given, without a trailing separator, and as JSON writes a Windows one."""
    path = path.rstrip("/\\") or path
    return {path, json.dumps(path)[1:-1]}


def usable_name(name):
    return bool(name) and len(name) >= 2 and name.casefold() not in GENERIC_NAMES and not name.isdigit()


def name_pattern(name, in_paths_only):
    escaped = re.escape(name)
    if in_paths_only:
        # Part of a path ("/alice", "\\alice") or an address ("alice@"): never a word of the prompt.
        return re.compile(r"(?<=[/\\@])" + escaped + r"(?![A-Za-z0-9])|(?<![A-Za-z0-9])" + escaped + r"(?=@)",
                          re.IGNORECASE)
    return re.compile(r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])", re.IGNORECASE)


# --- what this machine is -------------------------------------------------------------------------------------------


def this_users():
    """This machine's user's names: the login, the home folder's name, and the words of the full name."""
    names = set()
    try:
        names.add(getpass.getuser())
    except (KeyError, OSError, ImportError):
        pass
    for variable in ("USER", "LOGNAME", "USERNAME"):
        names.add(os.environ.get(variable, ""))
    names.add(os.path.basename(os.path.expanduser("~").rstrip("/\\")))
    try:
        import pwd

        entry = pwd.getpwuid(os.getuid())
        names.add(entry.pw_name)
        names.update(word for word in re.split(r"[\s,]+", entry.pw_gecos or "") if len(word) >= 3)
    except (ImportError, KeyError, AttributeError):
        pass
    return {name for name in names if name}


def this_machines():
    names = set()
    for name in (socket.gethostname(), platform.node()):
        if name:
            names.update({name, name.split(".")[0]})
    try:
        fqdn = socket.getfqdn()
        names.update({fqdn, fqdn.split(".")[0]})
    except OSError:
        pass
    return {name for name in names if name and name.casefold() not in GENERIC_NAMES}


def secret_values(environment=None):
    """The values of the environment's variables whose names say they are secret (a token, a key, a password)."""
    environment = os.environ if environment is None else environment
    return {value for name, value in environment.items() if SECRET_VARIABLE.search(name) and len(value) >= 8}


def users_in(texts):
    """The user names that home folders in these texts name ("/home/alice/..." gives "alice")."""
    names = set()
    for text in texts:
        for match in HOME_PATH.finditer(text):
            names.update(name for name in match.groups() if name)
    return names


def code_version():
    try:
        from .pipeline import repository_version
    except ImportError:
        return "unknown"
    return repository_version()


def package_versions(names=PACKAGES):
    found = {}
    for name in names:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def registry_models(path=registry.REGISTRY):
    """Every model the registry lists: its id, version, the hub revisions of its files and its code's commit."""
    try:
        known = registry.models(path)
    except OSError:
        return []
    listed = []
    for model_id, entry in known.items():
        revisions = sorted({part["url"].split("/resolve/", 1)[1].split("/", 1)[0] for part in entry.get("parts", [])
                            if "/resolve/" in part["url"]})
        listed.append({"id": model_id, "version": entry.get("version"), "revisions": revisions,
                       "source_commit": (entry.get("source") or {}).get("commit")})
    return listed


def file_sha256(path):
    try:
        return registry.sha256_of(path)
    except OSError:
        return None


def about_this_machine(machine=None):
    machine = ledger.machine() if machine is None else machine
    return {"code_version": code_version(), "models": registry_models(),
            "registry_sha256": file_sha256(registry.REGISTRY),
            "python": platform.python_version(), "torch": package_versions(("torch",))["torch"],
            "packages": package_versions(), "system": f"{platform.system()} {platform.release()} {platform.machine()}",
            "card": {"name": machine.get("card"), "memory_mb": machine.get("card_mb"), "driver": machine.get("driver")},
            "cpu_threads": machine.get("cpu_threads")}


# --- the report file -----------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Member:
    name: str
    data: bytes
    text: bool = False


def ledger_files(job_folder, ledger_path=None):
    """The ledgers that may hold the job's lines: --ledger, the ledger's own folder, and a `metrics/` beside the job's
    folder (a batch's own, when MODEL_MAKER_LEDGER pointed there)."""
    candidates = []
    if ledger_path:
        candidates.append(os.path.join(ledger_path, ledger.LEDGER_FILE) if os.path.isdir(ledger_path) else ledger_path)
    candidates.append(os.path.join(ledger.folder(), ledger.LEDGER_FILE))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(job_folder)), "metrics", ledger.LEDGER_FILE))
    seen, found = set(), []
    for path in candidates:
        real = os.path.realpath(path)
        if real not in seen and os.path.isfile(real):
            seen.add(real)
            found.append(path)
    return found


def job_lines(paths, job_id):
    """The ledger lines of one job: its own (by "job") and its batch's run line (which lists it in "jobs")."""
    lines, seen = [], set()
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as file:
            for text in file:
                text = text.strip()
                if not text or job_id not in text or text in seen:
                    continue
                try:
                    line = json.loads(text)
                except ValueError:
                    continue
                if line.get("job") == job_id or job_id in (line.get("jobs") or []):
                    seen.add(text)
                    lines.append(line)
    return lines


def png_without_text(data):
    """A PNG's bytes with only the chunks that say how to show it, and its content credentials: no text, EXIF or time.
    (count of dropped chunks)."""
    if not data.startswith(PNG_SIGNATURE):
        return data, 0
    chunks, at = [], len(PNG_SIGNATURE)
    while at + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[at:at + 8])
        chunks.append((kind, data[at:at + 12 + length]))
        at += 12 + length
        if kind == b"IEND":
            break
    others_dropped = any(kind not in PNG_KEPT_CHUNKS and kind != PNG_CREDENTIALS_CHUNK for kind, _ in chunks)
    kept_kinds = PNG_KEPT_CHUNKS if others_dropped else PNG_KEPT_CHUNKS | {PNG_CREDENTIALS_CHUNK}
    kept = [chunk for kind, chunk in chunks if kind in kept_kinds]
    return PNG_SIGNATURE + b"".join(kept), len(chunks) - len(kept)


def glb_cleaned(data, clean):
    """A .glb with its JSON chunk (names, extras) cleaned by `clean`; its binary chunk (numbers, the texture) kept."""
    if len(data) < 20 or data[:4] != b"glTF":
        return data
    version = struct.unpack("<I", data[4:8])[0]
    length, kind = struct.unpack("<I4s", data[12:20])
    if kind != b"JSON":
        return data
    text = clean(data[20:20 + length].decode("utf-8"))
    body = text.encode("utf-8")
    body += b" " * (-len(body) % 4)
    rest = data[20 + length:]
    return struct.pack("<4sII", b"glTF", version, 12 + 8 + len(body) + len(rest)) + \
        struct.pack("<I4s", len(body), b"JSON") + body + rest


def glb_json(data):
    if len(data) < 20 or data[:4] != b"glTF":
        return ""
    length = struct.unpack("<I", data[12:16])[0]
    return data[20:20 + length].decode("utf-8", "replace")


class Refused(Exception):
    """The report file is not written (a personal thing left in it, or no job here)."""


@dataclasses.dataclass
class Made:
    path: str
    size: int
    left_out: list
    replaced: dict
    members: list


def human_size(size):
    return f"{size / 1e6:.1f} MB" if size >= 100_000 else f"{size / 1e3:.0f} kB"


def folder_size(path):
    total = 0
    for root, _, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def bug_report(job_folder, out=None, with_model=False, ledger_path=None, about=None, users=None, machines=None,
               environment=None):
    """Writes the report file for a job's folder; returns what was made. `about`, `users`, `machines` and
    `environment` are this machine's own unless given (tests give made-up ones)."""
    job_folder = os.path.abspath(job_folder)
    record_path = os.path.join(job_folder, "job.json")
    if not os.path.isfile(record_path):
        raise Refused(f"{job_folder}: no job.json here; give a job's folder (the one with its .glb and report/)")
    with open(record_path, encoding="utf-8") as file:
        record = json.load(file)
    slug = record.get("slug") or "job"
    job_id = record.get("id")
    about = about_this_machine() if about is None else about
    environment = os.environ if environment is None else environment
    ledgers = ledger_files(job_folder, ledger_path)
    lines = job_lines(ledgers, job_id) if job_id else []

    members, left_out = [], []

    def read(path):
        with open(path, "rb") as file:
            return file.read()

    def add(name, path):
        data = read(path)
        if name.endswith(".png"):
            data, dropped = png_without_text(data)
            if dropped:
                left_out.append(f"{name}: {dropped} text or metadata chunks (the pixels are kept)")
        members.append(Member(name, data, name.endswith(TEXT_ENDINGS)))

    add("job/job.json", record_path)
    sidecar = os.path.join(job_folder, f"{slug}.provenance.json")
    if not os.path.isfile(sidecar):
        sidecars = sorted(name for name in os.listdir(job_folder) if name.endswith(".provenance.json"))
        sidecar = os.path.join(job_folder, sidecars[0]) if sidecars else None
    if sidecar:
        add(f"job/{os.path.basename(sidecar)}", sidecar)
    else:
        left_out.append("no sidecar: the job did not reach its finish")
    for name in ("picture.png", "mask.png"):
        if os.path.isfile(os.path.join(job_folder, name)):
            add(f"job/{name}", os.path.join(job_folder, name))
        else:
            left_out.append(f"no {name} in the job's folder")
    report_folder = os.path.join(job_folder, "report")
    if os.path.isdir(report_folder):
        for root, _, names in sorted(os.walk(report_folder)):
            for name in sorted(names):
                path = os.path.join(root, name)
                inside = os.path.relpath(path, job_folder).replace(os.sep, "/")
                if name.endswith(".png") or name.endswith(TEXT_ENDINGS):
                    add(f"job/{inside}", path)
                elif name.lower().endswith((".jpg", ".jpeg", ".webp")):
                    # Only a PNG's metadata is stripped here, so other pictures (where tools write names) stay home.
                    left_out.append(f"{inside}: a picture whose metadata is not stripped (only PNGs are sent)")
                else:
                    left_out.append(f"{inside} ({human_size(os.path.getsize(path))}): not a page or a picture")
    else:
        left_out.append("no report/ folder: the job did not reach its finish")
    model = os.path.join(job_folder, f"{slug}.glb")
    if with_model and os.path.isfile(model):
        members.append(Member(f"job/{slug}.glb", read(model)))
    elif os.path.isfile(model):
        left_out.append(f"{slug}.glb ({human_size(os.path.getsize(model))}): the model (add it with --with-model)")
    for name in sorted(os.listdir(job_folder)):
        path = os.path.join(job_folder, name)
        if name.endswith(".npz"):
            left_out.append(f"{name} ({human_size(os.path.getsize(path))}): an in-between file, made again by a remake")
        elif name in FOLDERS_LEFT_OUT and os.path.isdir(path):
            left_out.append(f"{name}/ ({human_size(folder_size(path))}): {FOLDERS_LEFT_OUT[name]}")
    left_out.append("the environment's variables and any token")
    ledger_text = "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines)
    members.append(Member("job/ledger.jsonl", ledger_text.encode(), True))
    if not lines:
        left_out.append("no ledger lines: none was found for this job (give the ledger with --ledger)")

    # What to replace: this machine's folders and names, and those of the machine the job was made on, as its
    # record and ledger lines name them.
    texts = [m.data.decode("utf-8", "replace") for m in members if m.text]
    texts += [glb_json(m.data) for m in members if m.name.endswith(".glb")]
    out_folder = os.path.dirname(job_folder)
    paths = [(job_folder, "<job>"), (os.path.realpath(job_folder), "<job>"), (out_folder, "<out>"),
             (os.path.realpath(out_folder), "<out>")]
    for line in lines:
        if isinstance(line.get("folder"), str) and os.path.isabs(line["folder"]):
            paths += [(line["folder"], "<job>"), (os.path.dirname(line["folder"]), "<out>")]
    here = os.path.dirname(os.path.abspath(__file__))
    for path, placeholder in ((os.path.normpath(os.path.join(here, "..", "..")), "<model-maker>"),
                              (environment.get("MODEL_MAKER_MODELS"), "<models>"),
                              (environment.get("MODEL_MAKER_SOURCES"), "<sources>"),
                              (environment.get("MODEL_MAKER_LEDGER"), "<ledger>"),
                              (os.path.expanduser("~"), "<home>"), (environment.get("HOME"), "<home>"),
                              (os.getcwd(), "<here>")):
        if path and os.path.isabs(path) and len(path.rstrip("/\\")) > 1:
            paths += [(path, placeholder), (os.path.realpath(path), placeholder)]
    users = (this_users() if users is None else set(users)) | users_in(texts)
    machines = this_machines() if machines is None else set(machines)
    prompt_words = set()
    for key in ("prompt",):
        prompt_words.update(re.findall(r"[A-Za-z0-9]+", str(record.get(key, ""))))
    prompt_words.update(re.findall(r"[A-Za-z0-9]+", str(record.get("picture", {}).get("prompt", ""))))
    scrubber = Scrubber(paths, users, machines, prompt_words, secret_values(environment))
    kept_in_paths_only = sorted({n for n in users | machines if usable_name(n) and n.casefold() in
                                 {w.casefold() for w in prompt_words}})

    for member in members:
        if member.text:
            member.data = scrubber.clean(member.data.decode("utf-8", "replace")).encode("utf-8")
        elif member.name.endswith(".glb"):
            member.data = glb_cleaned(member.data, scrubber.clean)
    left_out = [scrubber.clean(item) for item in left_out]
    made_with = None
    for member in members:
        if member.name.endswith(".provenance.json"):
            try:
                made_with = json.loads(member.data).get("version")
            except ValueError:
                pass
    about_record = json.loads(scrubber.clean(json.dumps({
        "bug_report": FORMAT, "made_at": ledger.now(),
        "job": {"id": job_id, "prompt": record.get("prompt"), "slug": slug, "status": record.get("status"),
                "made_at": record.get("made_at"), "version": made_with,
                "made_on": next((line["machine"] for line in reversed(lines) if line.get("job") == job_id
                                 and isinstance(line.get("machine"), dict)), None)},
        "this_machine": "the versions below are this machine's, where the report file was made", **about})))
    replaced = {PLACEHOLDERS[k]: v for k, v in sorted(scrubber.replaced.counts.items())}
    about_record.update({"left_out": left_out, "replaced": replaced})
    if kept_in_paths_only:
        about_record["note"] = (f"{len(kept_in_paths_only)} of your names is also a word of the prompt: it was "
                                "replaced only in paths, so the prompt is kept whole")
    model_line = f"  job/{slug}.glb         the model (--with-model)\n" if with_model and os.path.isfile(model) else ""
    readme = README.format(model_line=model_line,
                           left_out="\n".join(f"  - {item}" for item in left_out),
                           replaced="\n".join(f"  - {what}: {count} times" for what, count in replaced.items())
                           or "  - nothing")
    members.insert(0, Member("README.txt", scrubber.clean(readme).encode(), True))
    members.insert(1, Member("about.json", (json.dumps(about_record, indent=1) + "\n").encode(), True))

    # The check: no copied text holds any of it any more.
    found = []
    for member in members:
        text = member.data.decode("utf-8", "replace") if member.text else glb_json(member.data)
        found += [(member.name, what) for what, _ in scrubber.leaks(text)]
    if found:
        shown = sorted({f"{name}: {what}" for name, what in found})
        raise Refused("a personal thing would be left in the report file, so it is not written: " + "; ".join(shown))

    out = out or f"bug-report-{slug}-{(job_id or 'job')[:8]}.zip"
    written_archive(out, members)
    return Made(out, os.path.getsize(out), left_out, replaced, [m.name for m in members])


def archive_kind(path):
    lowered = path.lower()
    if lowered.endswith(".zip"):
        return "zip"
    if lowered.endswith((".tar.gz", ".tgz")):
        return "tar.gz"
    raise Refused(f"{path}: the report file's name ends in .zip or .tar.gz")


def written_archive(path, members):
    kind = archive_kind(path)
    partial = path + ".partial"
    stamp = datetime.datetime.now(datetime.timezone.utc)
    if kind == "zip":
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
            for member in members:
                info = zipfile.ZipInfo(f"{ROOT}/{member.name}", stamp.timetuple()[:6])
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, member.data)
    else:
        with tarfile.open(partial, "w:gz") as archive:
            for member in members:
                # A tar entry holds a user and group name and ids: none are written.
                info = tarfile.TarInfo(f"{ROOT}/{member.name}")
                info.size, info.mode, info.mtime = len(member.data), 0o644, int(stamp.timestamp())
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(member.data))
    os.replace(partial, path)


# --- reading a report file and remaking it ---------------------------------------------------------------------------


@dataclasses.dataclass
class Report:
    """A report file's parts, as `reproduce` needs them."""

    about: dict
    record: dict
    sidecar: dict
    picture: bytes
    files: dict


def read_report(path):
    """The members of a report file under bug-report/, read into memory (nothing is extracted, so no name in it can
    reach a folder of this machine); refuses a member larger than LARGEST_MEMBER_MB."""
    limit = LARGEST_MEMBER_MB * 2 ** 20
    files = {}
    kind = archive_kind(path)
    if kind == "zip":
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.startswith(ROOT + "/"):
                    continue
                if info.file_size > limit:
                    raise Refused(f"{info.filename}: larger than {LARGEST_MEMBER_MB} MB")
                files[info.filename[len(ROOT) + 1:]] = archive.read(info)
    else:
        with tarfile.open(path, "r:gz") as archive:
            for info in archive:
                if not info.isfile() or not info.name.startswith(ROOT + "/"):
                    continue
                if info.size > limit:
                    raise Refused(f"{info.name}: larger than {LARGEST_MEMBER_MB} MB")
                files[info.name[len(ROOT) + 1:]] = archive.extractfile(info).read()

    def loaded(name):
        try:
            return json.loads(files[name].decode("utf-8"))
        except KeyError:
            raise Refused(f"{path}: no {name} in it; is it a model-maker bug report?") from None
        except ValueError as error:
            raise Refused(f"{path}: {name} is not JSON ({error})") from None

    record = loaded("job/job.json")
    sidecars = [name for name in files if name.startswith("job/") and name.endswith(".provenance.json")
                and "/" not in name[4:]]
    sidecar = loaded(sidecars[0]) if sidecars else {}
    about = loaded("about.json") if "about.json" in files else {}
    return Report(about, record, sidecar, files.get("job/picture.png", b""), files)


# The settings a remake takes from a report, each with the range it is held to: a report is someone else's file, and
# must not hold the card for hours.
SETTING_RANGES = {"tries": (1, 16), "shape_tries": (1, 8), "triangles": (1000, 500_000), "texture": (256, 8192),
                  "paint_steps": (1, 100), "picture_rounds": (1, 4), "touch_up_views": (0, 8),
                  "touch_up_strength": (0.0, 1.0)}


@dataclasses.dataclass
class Order:
    """What to remake: the prompt, its options and settings as the report's job had them."""

    prompt: str
    size_m: float = None
    stands: bool = None
    symmetric: bool = None
    settings: dict = dataclasses.field(default_factory=dict)
    seeds: dict = dataclasses.field(default_factory=dict)
    version: str = None
    status: str = None
    notes: list = dataclasses.field(default_factory=list)


def order_of(report):
    """The order to remake from a report's job record and sidecar; refuses what is not well formed."""
    from . import words

    record = report.record
    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 1000:
        raise Refused("the report's job.json has no prompt")
    options = record.get("options") or {}
    size_m, stands, symmetric = options.get("size_m"), options.get("stands"), options.get("symmetric")
    if size_m is not None and not (isinstance(size_m, (int, float)) and 0.01 <= size_m <= 100):
        raise Refused(f"the report's size ({size_m!r}) is not a size in metres")
    for name, value in (("stands", stands), ("symmetric", symmetric)):
        if value is not None and not isinstance(value, bool):
            raise Refused(f"the report's option {name} ({value!r}) is not yes or no")
    notes = []
    kept = record.get("settings")
    settings = {}
    if kept is None:
        notes.append("the job's record keeps no settings (made before they were kept): the defaults are used")
    else:
        for name, (lowest, highest) in SETTING_RANGES.items():
            if name not in kept or kept[name] is None:
                continue
            value = kept[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not lowest <= value <= highest:
                raise Refused(f"the report's setting {name} ({value!r}) is outside {lowest}..{highest}")
            settings[name] = value if isinstance(lowest, float) else int(value)
        lead = kept.get("lead")
        if lead is not None:
            if lead not in words.LEADS:
                raise Refused(f"the report's picture lead ({lead!r}) is not one this code knows")
            settings["lead"] = lead
    seeds = dict(report.sidecar.get("seeds") or {})
    if not seeds:
        seeds = {"picture": (record.get("picture") or {}).get("seed"), "shape": (record.get("shape") or {}).get("seed")}
    version = report.sidecar.get("version") or (report.about or {}).get("code_version")
    status = report.sidecar.get("status") or record.get("status")
    return Order(prompt, size_m, stands, symmetric, settings, seeds, version, status, notes)


# --- comparing a remake with its report ------------------------------------------------------------------------------


def png_pixels(data):
    """A PNG's pixels as (width, height, RGB bytes): 8 bits a channel, grey, grey and alpha, RGB, RGBA or a palette,
    not interlaced (as the pipeline writes them). Alpha is dropped, grey made RGB."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG")
    at, idat, palette = len(PNG_SIGNATURE), [], None
    width = height = depth = colour = interlace = None
    while at + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[at:at + 8])
        body = data[at + 8:at + 8 + length]
        if kind == b"IHDR":
            width, height, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", body)
        elif kind == b"PLTE":
            palette = body
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
        at += 12 + length
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(colour)
    if depth != 8 or channels is None or interlace:
        raise ValueError(f"a PNG of {depth} bits, colour type {colour}, interlace {interlace}: not read here")
    raw = zlib.decompress(b"".join(idat))
    stride = width * channels
    rows, previous, at = [], bytearray(stride), 0
    for _ in range(height):
        kind, line = raw[at], bytearray(raw[at + 1:at + 1 + stride])
        at += 1 + stride
        unfiltered(kind, line, previous, channels)
        rows.append(bytes(line))
        previous = line
    pixels = b"".join(rows)
    if colour == 2:
        return width, height, pixels
    if colour == 6:
        return width, height, b"".join(pixels[k:k + 3] for k in range(0, len(pixels), 4))
    if colour == 0:
        return width, height, bytes(v for v in pixels for _ in range(3))
    if colour == 4:
        return width, height, bytes(pixels[k] for k in range(0, len(pixels), 2) for _ in range(3))
    return width, height, b"".join(palette[3 * v:3 * v + 3] for v in pixels)


def unfiltered(kind, line, previous, step):
    """Undoes one row's PNG filter in place (the specification's five filters)."""
    if kind == 0:
        return
    if kind == 1:
        for k in range(step, len(line)):
            line[k] = (line[k] + line[k - step]) & 255
    elif kind == 2:
        for k in range(len(line)):
            line[k] = (line[k] + previous[k]) & 255
    elif kind == 3:
        for k in range(len(line)):
            left = line[k - step] if k >= step else 0
            line[k] = (line[k] + ((left + previous[k]) >> 1)) & 255
    elif kind == 4:
        for k in range(len(line)):
            a = line[k - step] if k >= step else 0
            b = previous[k]
            c = previous[k - step] if k >= step else 0
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            line[k] = (line[k] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
    else:
        raise ValueError(f"PNG filter {kind}: no such filter")


def mean_pixel_difference(first, second):
    """The mean of each pixel channel's absolute difference between two PNGs, in levels of 0..255; None (and why)
    when they cannot be compared."""
    try:
        a, b = png_pixels(first), png_pixels(second)
    except (ValueError, zlib.error, struct.error) as error:
        return None, f"not compared: {error}"
    if a[:2] != b[:2]:
        return None, f"not compared: {a[0]}x{a[1]} and {b[0]}x{b[1]} pixels"
    total = sum(abs(x - y) for x, y in zip(a[2], b[2]))
    return total / len(a[2]), ""


def counted_checks(sidecar):
    """{(stage, check name): check} of the checks that count toward the verdict."""
    found = {}
    for stage, checks in (sidecar.get("checks") or {}).items():
        for check in checks or []:
            if isinstance(check, dict) and check.get("counted", True) and "name" in check:
                found[(stage, check["name"])] = check
    return found


def compared(order, theirs_sidecar, theirs_picture, ours_sidecar, ours_picture, our_version, machine):
    difference, why = mean_pixel_difference(theirs_picture, ours_picture) if theirs_picture and ours_picture \
        else (None, "not compared: a picture is missing")
    theirs, ours = counted_checks(theirs_sidecar), counted_checks(ours_sidecar)
    rows = []
    for key in list(theirs) + [k for k in ours if k not in theirs]:
        a, b = theirs.get(key, {}), ours.get(key, {})
        rows.append({"stage": key[0], "check": key[1], "reported": a.get("value"), "remade": b.get("value"),
                     "limit": b.get("limit", a.get("limit")), "unit": b.get("unit", a.get("unit")),
                     "reported_passed": a.get("passed"), "remade_passed": b.get("passed")})
    our_seeds = ours_sidecar.get("seeds") or {}
    seeds = {name: {"reported": order.seeds.get(name), "remade": our_seeds.get(name)}
             for name in sorted(set(order.seeds) | set(our_seeds))}
    their_models = {m.get("id"): m.get("version") for m in theirs_sidecar.get("models") or []}
    our_models = {m.get("id"): m.get("version") for m in ours_sidecar.get("models") or []}
    models_differ = sorted(k for k in set(their_models) | set(our_models) if their_models.get(k) != our_models.get(k))
    differs = []
    if order.version != our_version:
        differs.append("the code's version")
    if any(s["reported"] != s["remade"] for s in seeds.values()):
        differs.append("the seeds")
    if models_differ:
        differs.append("the models: " + ", ".join(models_differ))
    if difference is None or difference >= 0.5:
        differs.append("the picture")
    if order.status != ours_sidecar.get("status"):
        differs.append("the verdict")
    if any(r["reported_passed"] != r["remade_passed"] for r in rows):
        differs.append("a check's pass or fail")
    return {"remade_on": machine, "version": {"reported": order.version, "remade": our_version}, "seeds": seeds,
            "models_differ": models_differ,
            "picture_mean_difference": None if difference is None else round(difference, 3),
            "picture_note": why, "verdict": {"reported": order.status, "remade": ours_sidecar.get("status")},
            "checks": rows, "differs": differs}


def number(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    return str(value)


def comparison_lines(comparison):
    machine = comparison["remade_on"] or {}
    card = (f"{machine.get('card')} (driver {machine.get('driver')})" if machine.get("card")
            else "no card found (nvidia-smi gave none)")
    version = comparison["version"]
    lines = [f"remade on: {card}",
             f"version: reported {version['reported']}, remade with {version['remade']}"
             + ("" if version["reported"] == version["remade"] else " (they differ)"),
             "seeds: " + ", ".join(f"{name} {s['reported']} / {s['remade']}" for name, s in comparison["seeds"].items())]
    if comparison["models_differ"]:
        lines.append("models that differ: " + ", ".join(comparison["models_differ"]))
    difference = comparison["picture_mean_difference"]
    lines.append(f"picture: mean pixel difference {difference:.2f} of 255 ({difference / 2.55:.2f}%)"
                 if difference is not None else f"picture: {comparison['picture_note']}")
    verdict = comparison["verdict"]
    lines.append(f"verdict: reported \"{verdict['reported']}\"; remade \"{verdict['remade']}\"")
    width = max([len(f"{r['stage']}: {r['check']}") for r in comparison["checks"]] + [5])
    lines.append(f"{'check':{width}}  {'reported':>10}  {'remade':>10}  {'limit':>8}")
    for row in comparison["checks"]:
        mark = "" if row["reported_passed"] == row["remade_passed"] else \
            f"  <- {'passes' if row['reported_passed'] else 'fails'} / {'passes' if row['remade_passed'] else 'fails'}"
        lines.append(f"{row['stage'] + ': ' + row['check']:{width}}  {number(row['reported']):>10}  "
                     f"{number(row['remade']):>10}  {number(row['limit']):>8}  {row['unit'] or ''}{mark}")
    lines.append("the remake agrees with the report" if not comparison["differs"]
                 else "differs: " + "; ".join(comparison["differs"]))
    return lines


def reproduce(report_path, out, make, our_version=None, machine=None, say=print):
    """Remakes a report's job into `out` with `make(order, out) -> the made job's folder` and compares it with the
    report; returns the comparison (also written to out/comparison.json)."""
    report = read_report(report_path)
    order = order_of(report)
    os.makedirs(out, exist_ok=True)
    reported = os.path.join(out, "reported")
    os.makedirs(reported, exist_ok=True)
    for name, data in (("job.json", json.dumps(report.record, indent=1)),
                       ("provenance.json", json.dumps(report.sidecar, indent=1))):
        with open(os.path.join(reported, name), "w") as file:
            file.write(data)
    if report.picture:
        with open(os.path.join(reported, "picture.png"), "wb") as file:
            file.write(report.picture)
    say(f"remaking \"{order.prompt}\" (reported: {order.status}; version {order.version}; seeds "
        + ", ".join(f"{k} {v}" for k, v in order.seeds.items()) + ")")
    for note in order.notes:
        say(f"note: {note}")
    # The remake's ledger lines go to out/metrics: a check of someone's report is not one of our own jobs, and must not
    # count in our progress figures.
    before = os.environ.get("MODEL_MAKER_LEDGER")
    os.environ["MODEL_MAKER_LEDGER"] = os.path.join(os.path.abspath(out), "metrics")
    try:
        folder = make(order, out)
    finally:
        if before is None:
            os.environ.pop("MODEL_MAKER_LEDGER", None)
        else:
            os.environ["MODEL_MAKER_LEDGER"] = before
    if not folder:
        raise Refused("the remake made no job (the prompt was refused; see above)")
    with open(os.path.join(folder, "job.json"), encoding="utf-8") as file:
        made_record = json.load(file)
    sidecar_path = os.path.join(folder, f"{made_record.get('slug')}.provenance.json")
    ours = {}
    if os.path.isfile(sidecar_path):
        with open(sidecar_path, encoding="utf-8") as file:
            ours = json.load(file)
    picture_path = os.path.join(folder, "picture.png")
    picture = b""
    if os.path.isfile(picture_path):
        with open(picture_path, "rb") as file:
            picture = file.read()
    comparison = compared(order, report.sidecar, report.picture, ours, picture,
                          code_version() if our_version is None else our_version,
                          ledger.machine() if machine is None else machine)
    comparison["remade_job"] = os.path.relpath(folder, out)
    with open(os.path.join(out, "comparison.json"), "w") as file:
        json.dump(comparison, file, indent=1)
    for line in comparison_lines(comparison):
        say(line)
    return comparison


def printed_made(made, say=print):
    say(f"bug report: {made.path} ({human_size(made.size)}, {len(made.members)} files)")
    if made.size > ATTACHMENT_LIMIT_MB * 1e6:
        say(f"it is larger than the {ATTACHMENT_LIMIT_MB} MB GitHub takes: leave out --with-model, or attach it "
            "elsewhere and link it")
    say("left out:")
    for item in made.left_out:
        say(f"  - {item}")
    say("replaced in every text copied:" if made.replaced else "replaced: nothing personal was found")
    for what, count in made.replaced.items():
        say(f"  - {what}: {count} times")
    say("attach it to a new issue (Bug report form); see README.txt inside for what it holds")


def main_bug_report(args, say=print):
    try:
        made = bug_report(args.folder, args.out, args.with_model, args.ledger)
    except Refused as refused:
        print(f"model-maker bug-report: {refused}", file=sys.stderr)
        return 1
    printed_made(made, say)
    return 0
