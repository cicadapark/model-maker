"""Whether the finished object is the thing its prompt names, asked of a vision-language model: its eight sides shown
together, "Is it <the thing named>?" (`asked_words`), and the probability of "yes" against "no" in the answer's first token (VQAScore's way:
Lin et al. 2024, "Evaluating text-to-visual generation with image-to-text generation").

Measured (docs/objects.md, "Is it the thing asked for?"; docs/results.md for 0.2.2): on 50 prompts sealed before the work, judged blind, refusing under
0.5 caught 6 of the 8 objects whose picture showed another thing (a digging fork for a hoe, a pick for a scythe) and
refused 3 good ones (long thin things: a pitchfork, a halberd; a barrel). It does not see a shape's or a paint's fault.
Asked of the pictures, the same question is a note only: over 4 tries it would refuse two names that had a right one.

The model is Qwen 3.5 9B (Apache-2.0), quantised (Q4_K_M) with its vision part, run on the CPU by llama.cpp's own
server (MIT), the release archive of tag b11600 that setup puts in model-maker's folder (never one found on the PATH:
another build answers differently, by up to 0.14 on the measured objects). It is optional, like the lettering reader:
without it the check says "not judged" and counts as nothing, except in a run that requires it.
"""

import base64
import ctypes
import hashlib
import http.client
import io
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass

from PIL import Image

from . import processes
from .picture_checks import Check

NAME = "it is the thing asked for"
NOT_JUDGED = "not judged"
LIMIT = 0.5
# Asked about the thing alone, counted (14 Nov 2026). Asked the whole prompt ("Is it an old iron sword?") it refused a
# good sword the regression set holds (0.44), the paint not looking old; asked the thing the words name ("Is it a
# sword?", 0.78) and measured on the next sealed 50 by a plan written before the run, it caught 7 of 16 bad objects
# that passed every other check and refused none of 15 good (12 GB card), 5 of 14 and 2 of 15 (16 GB card), and kept
# every good object of the regression set (docs/objects.md, "Is it the thing asked for?").
COUNTED = True
# The words, the grid and the picture's size are the ones the limit was measured with; the limit belongs to them.
OBJECT_QUESTION = "These are eight views of one object, seen from all round. Is it {prompt}? Answer yes or no."
PICTURE_QUESTION = "Does this picture show {prompt}? Answer yes or no."
LONGEST_SIDE = 512
STRIP_SIDES = 8
TOP_CHOICES = 20
# The server: loopback only, CPU only, its own threads; a start and an answer each have their limit. It keeps no
# earlier prompt (`--cache-ram 0`, and `cache_prompt` false in every request): each picture is asked afresh, so a
# kept prompt is never reused, and llama.cpp's prompt cache (8 GiB by default) grew the server from about 10 GB to
# 17.5 GB over a long batch. The answers are the same to within 1e-3 either way (measured on 160 pictures). The
# weights are still repacked for the processor: not repacking saves memory too, but moves the answers.
THREADS = min(12, os.cpu_count() or 1)
CONTEXT = 4096
START_SECONDS = 120
ANSWER_SECONDS = 120
LLAMA_TAG = "b11600"
SERVER_FOLDER = os.path.join(".tools", f"llama-{LLAMA_TAG}")
MODEL_FOLDER = "qwen3.5-9b-q4"
# Its entry in registry/models.toml (the files' sums the answers are held to).
REGISTRY_ID = "qwen3.5-9b-question"
MODEL_FILE = "Qwen_Qwen3.5-9B-Q4_K_M.gguf"
VISION_FILE = "mmproj-Qwen_Qwen3.5-9B-f16.gguf"


@dataclass(frozen=True)
class Found:
    """Whether the judge can run here: the server program and the model's two files; why not ("" when it can)."""

    usable: bool
    server: str = ""
    model: str = ""
    vision: str = ""
    why: str = ""


def found(root, models):
    """The judge's files as setup put them: the server in model-maker's folder `root` (.tools/llama-b11600/, the
    release archive unpacked), the model under `models`."""
    server = os.path.join(root, SERVER_FOLDER, processes.program_name("llama-server"))
    model = os.path.join(models, MODEL_FOLDER, MODEL_FILE)
    vision = os.path.join(models, MODEL_FOLDER, VISION_FILE)
    missing = [path for path in (server, model, vision) if not os.path.isfile(path)]
    if missing:
        return Found(False, server, model, vision,
                     f"{NOT_JUDGED} (the question judge is not installed: ./setup.sh --with-question-judge; "
                     f"missing {', '.join(os.path.basename(p) for p in missing)})")
    if not os.access(server, os.X_OK):
        return Found(False, server, model, vision, f"{NOT_JUDGED} (llama-server is not executable)")
    return Found(True, server, model, vision, "")


# What a question can fail with: the server gone or wrong (refused, cut off mid-answer, an answer of another shape).
ASK_ERRORS = (OSError, ValueError, KeyError, IndexError, TypeError, http.client.HTTPException)
_PR_SET_PDEATHSIG = 1
_PRCTL = None
if processes.LINUX:
    try:
        # Found here, in model-maker's process: the forked child only calls it (no library loading between fork and
        # exec).
        _PRCTL = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        _PRCTL = None


def _dies_with_parent():
    """In the server's process, before it runs: killed when model-maker dies, however it dies (Linux)."""
    if _PRCTL is not None:
        _PRCTL(_PR_SET_PDEATHSIG, int(signal.SIGKILL))


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def registry_sums():
    """The judge's model files' sha256 as registry/models.toml lists them, by file name."""
    from . import registry

    entry = registry.models().get(REGISTRY_ID, {})
    return {part["url"].rsplit("/", 1)[1]: part["sha256"] for part in entry.get("parts", [])}


def server_environment(where):
    """The server's environment: no card (it runs on the CPU), and the libraries beside it found first."""
    return processes.library_environment(os.path.dirname(where.server), {**os.environ, "CUDA_VISIBLE_DEVICES": ""})


def server_runs(where):
    """Whether the server program runs on this computer at all (`--version`), and its words if not: checked before a
    run that requires the judge spends the card's time on its jobs."""
    try:
        done = subprocess.run([where.server, "--version"], capture_output=True, timeout=60, stdin=subprocess.DEVNULL,
                              env=server_environment(where))
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"{type(error).__name__}: {error}"
    said = (done.stdout + done.stderr).decode(errors="replace").strip().splitlines()
    return done.returncode == 0, (said[-1][:200] if said else f"exit {done.returncode}")


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Judge:
    """The server started once for a stage of a batch, asked for each job, stopped when the stage ends (also on an
    error: use it as a context manager)."""

    def __init__(self, where, threads=THREADS):
        self.where, self.threads = where, threads
        self.port, self.process, self.log, self.restarts, self.sums = None, None, None, 0, None

    def __enter__(self):
        self.start()
        return self

    def start(self):
        self.port = free_port()
        # Its own words, in a file made for it alone (never one a name could point elsewhere); removed when it stops.
        descriptor, self.log = tempfile.mkstemp(prefix="model-maker-llama-", suffix=".log")
        with os.fdopen(descriptor, "wb") as log:
            self.process = subprocess.Popen(
                [self.where.server, "-m", self.where.model, "--mmproj", self.where.vision, "-ngl", "0",
                 "--no-mmproj-offload", "-t", str(self.threads), "-c", str(CONTEXT), "-np", "1", "--cache-ram",
                 "0", "--host", "127.0.0.1", "--port", str(self.port), "--jinja"],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=server_environment(self.where),
                **processes.start_options(_dies_with_parent))
        processes.tie_to_this_process(self.process)
        try:
            deadline = time.time() + START_SECONDS
            while time.time() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"llama-server stopped as it started (exit {self.process.returncode}: "
                                       f"{self.last_words()})")
                try:
                    with urllib.request.urlopen(self.url("/health"), timeout=5) as answer:
                        if answer.status == 200:
                            return
                except OSError:
                    pass
                time.sleep(1)
            raise RuntimeError(f"llama-server did not start in {START_SECONDS} s ({self.last_words()})")
        except BaseException:
            # Stopped however the start ends without a server (Ctrl-C included): nothing left running.
            self.__exit__(None, None, None)
            raise

    def last_words(self):
        try:
            with open(self.log, errors="replace") as file:
                lines = [line.strip() for line in file if line.strip()]
            return lines[-1][:200] if lines else "nothing said"
        except OSError:
            return "no log"

    def __exit__(self, *_):
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            processes.stop(process, wait_seconds=20)
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if self.log is not None:
            try:
                os.remove(self.log)
            except OSError:
                pass
            self.log = None

    def identity(self):
        """What answered: the llama.cpp build, the two model files with their sha256 (summed once) and whether they
        are the registry's, the threads."""
        if self.sums is None:
            self.sums = {"model_sha256": sha256_of(self.where.model), "vision_sha256": sha256_of(self.where.vision)}
            expected = registry_sums()
            self.sums["sums_are_the_registry_s"] = (expected.get(MODEL_FILE) == self.sums["model_sha256"]
                                                    and expected.get(VISION_FILE) == self.sums["vision_sha256"])
        return {"llama_tag": LLAMA_TAG, "model": MODEL_FILE, "vision": VISION_FILE, **self.sums,
                "threads": self.threads}

    def memory_mb(self):
        """The server's memory (MB): its peak and now, and of it what is its own (anonymous: the weights repacked for
        the processor, its caches) and what is the model files mapped in (the kernel can drop and read those again)."""
        if self.process is None:
            return None
        # From /proc on Linux; elsewhere what psutil reads (now, and the peak on Windows).
        found = processes.memory_of(self.process.pid)
        if found is None:
            return None
        return {"peak": found.get("VmHWM"), "now": found.get("VmRSS"), "own": found.get("RssAnon"),
                "model_files": found.get("RssFile")}

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def ask_or_restart(self, picture, question):
        """`ask`, and if the server has died, started once more (once a stage, counted in `restarts`) and asked again.
        A start that fails then is a failed question (OSError), never the job's own error."""
        try:
            return self.ask(picture, question)
        except ASK_ERRORS:
            if self.process is not None:
                try:
                    # A server that is going: a moment to end, so it is not taken for one that answered wrongly.
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
            if self.alive() or self.restarts:
                raise
            self.restarts += 1
            self.__exit__(None, None, None)
            try:
                self.start()
            except RuntimeError as error:
                raise OSError(f"the server stopped and could not start again: {error}") from error
            return self.ask(picture, question)

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def ask(self, picture, question):
        """(P(yes) / (P(yes) + P(no)) in the first answer token, or None when neither is among its top choices;
        P(yes), P(no) and the first token as answered)."""
        body = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": picture}}, {"type": "text", "text": question}]}],
            "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": TOP_CHOICES,
            "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}
        request = urllib.request.Request(self.url("/v1/chat/completions"), data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=ANSWER_SECONDS) as answer:
            first = json.load(answer)["choices"][0]["logprobs"]["content"][0]
        yes, no = probabilities(first["top_logprobs"])
        return Answer(None if yes + no == 0 else yes / (yes + no), yes, no, str(first.get("token", "")))


@dataclass(frozen=True)
class Answer:
    share: float
    yes: float
    no: float
    first_token: str


def probabilities(top):
    yes = sum(math.exp(one["logprob"]) for one in top if one["token"].strip().lower() == "yes")
    no = sum(math.exp(one["logprob"]) for one in top if one["token"].strip().lower() == "no")
    return yes, no


def as_data(image):
    """A picture as the trial sent it: a strip of eight sides laid out four by two, at most 1024 by 512; any other at
    most 512 a side; PNG in a data URL."""
    image = image.convert("RGB")
    if image.width >= STRIP_SIDES * image.height * 0.9:
        side = image.width // STRIP_SIDES
        grid = Image.new("RGB", (4 * side, 2 * image.height), "white")
        for at in range(STRIP_SIDES):
            grid.paste(image.crop((at * side, 0, (at + 1) * side, image.height)),
                       ((at % 4) * side, (at // 4) * image.height))
        image = grid
        image.thumbnail((2 * LONGEST_SIDE, 2 * LONGEST_SIDE))
    else:
        image.thumbnail((LONGEST_SIDE, LONGEST_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# Known things that also name what a thing is made of: "a stone bench" is asked as a bench, "a bone knife" as a knife.
ALSO_A_MATERIAL = ("stone", "bone", "rock", "horn", "rope", "chain", "log", "feather")


def asked_words(prompt):
    """What the judge is asked about: the thing the words name, with its article ("an old iron sword": "a sword"), or
    the whole prompt when the words name no thing model-maker knows ("an iron birdcage"). A name of two things is
    asked whole: "a bow saw" is asked as a bow saw, not a saw (0.3.0: asked "Is it a saw?", the judge refused a right
    bow saw on the 12 GB card, 16 Nov 2026)."""
    from . import words

    noun = words.subject_of(prompt).noun
    if not noun:
        return prompt
    before = word_before(words.cleaned(prompt).lower(), noun)
    if before and before in words.known_things() and before not in ALSO_A_MATERIAL:
        noun = f"{before} {noun}"
    if noun in words.ONE_THING_IN_PAIRS:
        return f"a pair of {noun}"  # "Is it a pair of headphones?", not "a headphones"
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def word_before(text, noun):
    """The word right before the noun's last place in the text ("a wooden bow saw", "saw": "bow"), or ""."""
    import re

    found = list(re.finditer(rf"(?<![a-z])([a-z-]+)\s+{re.escape(noun)}(?![a-z])", text))
    return found[-1].group(1) if found else ""


def object_check(judge, turntable, prompt, required, unavailable=""):
    """The check on the finished object's eight sides (`turntable`, the report's strip): its answer, shown and kept,
    counted when COUNTED. Without a judge (`unavailable` says why): not judged, and only a run that `required` it
    fails."""
    if judge is None:
        why = f"{NOT_JUDGED} ({unavailable or 'the question judge is not installed'})"
        return not_judged(why, required), {"judged": False, "why": why}
    started = time.time()
    asked = asked_words(prompt)
    question = OBJECT_QUESTION.format(prompt=asked)
    identity = judge.identity()
    restarts = judge.restarts
    answer, failure = None, ""
    if not identity["sums_are_the_registry_s"]:
        failure = "its model files are not the registry's (setup/fetch_models.py --verify)"
    else:
        try:
            answer = judge.ask_or_restart(as_data(turntable), question)
        except ASK_ERRORS as error:
            failure = f"{type(error).__name__}: {error}"
    share = None if answer is None else answer.share
    if answer is not None and share is None:
        failure = f"neither yes nor no in the answer (it began \"{answer.first_token}\")"
    record = {"judged": share is not None, "question": question, "yes": share,
              "p_yes": None if answer is None else answer.yes, "p_no": None if answer is None else answer.no,
              "first_token": None if answer is None else answer.first_token, "limit": LIMIT, "counted": COUNTED,
              **identity, "restarted": judge.restarts > restarts, "seconds": round(time.time() - started, 1),
              "server_memory_mb": judge.memory_mb()}
    if share is None:
        why = f"{NOT_JUDGED} ({failure})"
        record["why"] = why
        return not_judged(why, required), record
    if share >= LIMIT:
        words = f"asked of its eight sides \"Is it {asked}?\", the judge answers yes with {share:.0%} (at least {LIMIT:.0%})"
    else:
        words = (f"asked of its eight sides \"Is it {asked}?\", the judge answers yes with only {share:.0%} (at least "
                 f"{LIMIT:.0%}): it does not look like {asked}. If it is one, say what it looks like in the words "
                 f"(\"{prompt}, with ...\")")
    return Check(NAME, round(share * 100, 1), LIMIT * 100, "%", share >= LIMIT, words, counted=COUNTED), record


def not_judged(why, required):
    """Not asked: no pass and no fault, unless the run requires the judge (then a counted failure)."""
    return Check(NAME, 0.0, LIMIT * 100, "%", not required, why, counted=required)


def picture_note(judge, pictures, chosen, prompt):
    """The note on the pictures: the chosen try asked first; the others only when it is under the limit. Nothing is
    refused or chosen by it (measured: no gain in choosing; refusing over 4 tries lost right names)."""
    if judge is None:
        return {"judged": False}
    question = PICTURE_QUESTION.format(prompt=prompt)
    chosen = str(chosen)
    pictures = {str(seed): picture for seed, picture in pictures.items()}
    if chosen not in pictures:
        return {"judged": False, "why": "the chosen picture is not kept"}
    asked = {chosen: judge.ask_or_restart(as_data(pictures[chosen]), question).share}
    if asked[chosen] is None or asked[chosen] < LIMIT:
        for seed, picture in pictures.items():
            if seed != chosen:
                asked[seed] = judge.ask_or_restart(as_data(picture), question).share
    best = max((v for v in asked.values() if v is not None), default=None)
    return {"judged": True, "question": question, "tries": asked, "limit": LIMIT,
            "none_reaches": best is not None and best < LIMIT}


def server_on_path():
    """A llama-server found on the PATH, which the judge never uses (named in `--check-only`'s row)."""
    return shutil.which("llama-server")
