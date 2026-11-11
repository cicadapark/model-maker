"""The job ledger: one line of JSON per event, appended by the pipeline itself and never edited, so the numbers that show
progress (times, failures, pass rates) come from the machine and cannot be picked by hand. Failures and aborted jobs
are written like the rest. See docs/metrics.md.

    a stage done     {"event": "stage", "stage": "shape", "seconds": .., "peak_card_mb": .., "tries": ..}
    a model loaded   {"event": "load", "stage": "pictures", "seconds": ..}
    a job finished   {"event": "job", "status": .., "passed": .., "failed": [check names], "seconds": {..}}
    a stage failed   {"event": "aborted", "stage": .., "error": ..}
    a thing refused  {"event": "not made", "why": ..}
    a warm job's end {"event": "warm", "jobs_before": .., "card_allocated_mb": .., "rss_mb": .., ..}

A job's line also says how its run went ("run": warm or cold, its whole seconds, the seconds spent loading models in
it), so its real time is known whether the models were loaded for it or kept loaded between jobs; a batch writes
its whole time once at its end ("run" line, its jobs' ids), for them to share. A stage's seconds never hold a model's
loading: that has its own "load" line.

Every line carries when, the job (its id, folder, batch and prompt), the repository's version, the settings and the
machine (card, memory, driver, CPU threads), so a time is never shown without the card it was made on.
"""

import contextlib
import datetime
import json
import os
import subprocess
import sys
import time
import uuid

LEDGER_FILE = "jobs.jsonl"
REGRESSION_FILE = "regression.jsonl"
SCHEMA = 1
_MACHINE = None
# The run under way: began by `run_began`, its loads added by `loaded`, written on each job's line.
_RUN = None
# A stage's own key for seconds spent in it that were not its work (a model loading): left out of its time.
NOT_THE_STAGES = "not_the_stage_s"


def folder():
    """Where the ledgers are kept: MODEL_MAKER_LEDGER, else the repository's metrics/ folder."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get("MODEL_MAKER_LEDGER") or os.path.join(here, "..", "..", "metrics")


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def machine():
    """The card (from nvidia-smi, so the finish stage needs no torch), its driver and the CPU's threads."""
    global _MACHINE
    if _MACHINE is None:
        card = {"card": None, "card_mb": None, "driver": None}
        try:
            done = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                   "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                                  check=True)
            first = done.stdout.strip().splitlines()[0]
            name, mb, driver = (part.strip() for part in first.split(","))
            card = {"card": name, "card_mb": int(float(mb)), "driver": driver}
        except (OSError, subprocess.SubprocessError, IndexError, ValueError):
            pass
        _MACHINE = {**card, "cpu_threads": os.cpu_count()}
    return _MACHINE


def append(line, file_name=LEDGER_FILE):
    os.makedirs(folder(), exist_ok=True)
    text = json.dumps({"schema": SCHEMA, "at": now(), **line}, default=float, sort_keys=True)
    # One write of one line in append mode: lines from two runs at once do not interleave.
    with open(os.path.join(folder(), file_name), "a") as file:
        file.write(text + "\n")


def job_id(job):
    """A job's lasting id, kept in its job.json (given on its first ledger line, for jobs made before the ledger)."""
    if "id" not in job.record:
        job.record["id"] = uuid.uuid4().hex
        job.save()
    return job.record["id"]


def about(job, settings=None):
    from .pipeline import repository_version

    absolute = os.path.abspath(job.folder)
    # The settings the job was made with (kept in its record), so a refinish, which runs with the defaults, does not
    # write them over (a job painted at 30 steps must not be recorded at 20).
    made_with = job.record.get("settings")
    if made_with is not None and settings is not None:
        made_with = {**made_with, "texture": settings.texture}
    return {"job": job_id(job), "folder": absolute, "batch": os.path.basename(os.path.dirname(absolute)),
            "prompt": job["prompt"], "slug": job["slug"], "weapon": job["subject"].get("kind") == "weapon",
            "version": repository_version(), "machine": machine(),
            "settings": made_with or (settings_of(settings) if settings is not None else None)}


def settings_of(settings):
    return {"tries": settings.tries, "shape_tries": settings.shape_tries, "triangles": settings.triangles,
            "texture": settings.texture, "lead": settings.lead, "paint_steps": settings.paint_steps,
            "picture_rounds": settings.picture_rounds, "touch_up_views": settings.touch_up_views,
            "touch_up_strength": settings.touch_up_strength,
            "require_lettering": getattr(settings, "require_lettering", False)}


def _card_peak_reset():
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _card_peak_mb():
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 2 ** 20)
    return None


@contextlib.contextmanager
def stage(job, name, settings):
    """Times one job's stage and writes its line; on an error, an "aborted" line with the error, then the error
    goes on. The block may fill the yielded dict with what the stage tried."""
    found = {}
    _card_peak_reset()
    started = time.time()
    try:
        yield found
    except BaseException as error:
        aborted(job, name, error, settings, time.time() - started)
        raise
    not_its_work = found.pop(NOT_THE_STAGES, 0.0)
    peak = _card_peak_mb()
    if _RUN is not None and peak is not None:
        _RUN["peak_card_mb"] = max(_RUN["peak_card_mb"] or 0, peak)
    append({**about(job, settings), "event": "stage", "stage": name,
            "seconds": round(time.time() - started - not_its_work, 1), "peak_card_mb": peak, **found})


def aborted(job, stage_name, error, settings=None, seconds=0.0):
    """A job's stage ended by an error (its models' loading too): an "aborted" line with the error."""
    append({**about(job, settings), "event": "aborted", "stage": stage_name, "seconds": round(seconds, 1),
            "error": f"{type(error).__name__}: {error}"[:2000]})


def loaded(stage_name, seconds):
    if _RUN is not None:
        _RUN["load_seconds"] += seconds
    append({"event": "load", "stage": stage_name, "seconds": round(seconds, 1), "machine": machine()})


def run_began(jobs, warm=False):
    """A run of `jobs` jobs begins (`make`: one; `batch`: its prompts; the warm process: each order). `warm`: the
    process made a job before this one, its models kept loaded."""
    global _RUN
    _RUN = {"warm": warm, "jobs": jobs, "started": time.time(), "load_seconds": 0.0, "peak_card_mb": None}


def run_record():
    """The run so far, as a job's line says it: its whole seconds and its loads. A run of more than one job (a batch)
    leaves them out of its jobs' lines: they are known only at its end, and written then (`run_ended`)."""
    if _RUN is None:
        return None
    one = _RUN["jobs"] == 1
    return {"warm": _RUN["warm"], "jobs": _RUN["jobs"],
            "whole_seconds": round(time.time() - _RUN["started"], 1) if one else None,
            "load_seconds": round(_RUN["load_seconds"], 1) if one else None, "peak_card_mb": _RUN["peak_card_mb"]}


def run_ended(job_ids):
    """A batch's end: its whole seconds and its loading, for its jobs to share evenly (`tools/trial/summary.py`)."""
    if _RUN is None:
        return
    append({"event": "run", "jobs": list(job_ids), "warm": _RUN["warm"],
            "whole_seconds": round(time.time() - _RUN["started"], 1), "load_seconds": round(_RUN["load_seconds"], 1),
            "peak_card_mb": _RUN["peak_card_mb"], "version": _version(), "machine": machine()})


def warm_job(job_id, jobs_before, seconds, memory):
    """A warm process's job ended (one that keeps its models between jobs, kept.py): what it holds now, on the card
    and in the computer's memory, so a creep from job to job shows."""
    append({"event": "warm", "job": job_id, "jobs_before": jobs_before, "seconds": round(seconds, 1), **memory,
            "version": _version(), "machine": machine()})


def finished(job, settings, grouped, measures):
    """The job's end: its status, the checks that failed, and every counted check's value against its limit."""
    every = [c for group in grouped.values() for c in group if c.get("counted", True)]
    failed = [c["name"] for c in every if not c["passed"]]
    append({**about(job, settings), "event": "job", "status": job.record["status"], "passed": not failed,
            "failed": failed, "seconds": job["seconds"],
            "first_tries": first_tries(job.record),
            # Whether the chosen picture was read for painted lettering (None: made before the check): a job not read
            # is not counted as a pass by tools/trial/summary.py.
            "lettering_judged": (job.record.get("lettering") or {}).get("judged"),
            "tesseract_version": (job.record.get("lettering") or {}).get("tesseract"),
            "checks": [{"name": c["name"], "value": c.get("value"), "limit": c.get("limit"), "passed": c["passed"]}
                       for c in every],
            "levelled_mm": round(1000 * measures["levelling"]["trimmed_m"], 1) if measures.get("levelling") else 0.0,
            "made_at": job["made_at"], **({"run": run} if (run := run_record()) else {})})


def not_made(job, why):
    append({**about(job), "event": "not made", "why": why})


def refused(prompt, out, why):
    """A prompt refused before any job was made for it (`make`, the warm process): its own "not made" line, so the
    prompts asked are all counted."""
    from .pipeline import repository_version

    append({"job": uuid.uuid4().hex, "folder": None, "batch": os.path.basename(os.path.abspath(out)),
            "prompt": prompt, "slug": None, "weapon": False, "version": repository_version(), "machine": machine(),
            "settings": None, "event": "not made", "why": why})


def first_tries(record):
    """Whether the first round of pictures gave a passing one, and the first shape seed passed every counted check."""
    picture = record.get("picture", {})
    shape_tries = record.get("shape", {}).get("tries", [])
    first_shape = shape_tries[0] if shape_tries else None
    return {"picture_rounds": rounds_of(picture),
            "shape_tries": len(shape_tries),
            "shape_first_passed": None if first_shape is None else all(
                c["passed"] for c in first_shape["checks"] if c.get("counted", True))}


def rounds_of(picture):
    tries = picture.get("tries", {})
    if not tries:
        return None
    from .pipeline import DEFAULT_TRIES

    # Jobs made before the ledger did not keep their rounds: every try of a round was kept, so count them in rounds.
    return picture.get("rounds") or -(-len(tries) // DEFAULT_TRIES)


def regression(held, broken, seconds, cases, whole=True, tesseract=None):
    """A run of the regression set; `tesseract`: the version that read its lettering case (None: not read)."""
    append({"event": "regression", "held": held, "broken": broken, "seconds": round(seconds, 1),
            "cases": cases, "whole": whole, "machine": machine(), "version": _version(), "tesseract": tesseract},
           REGRESSION_FILE)


def _version():
    from .pipeline import repository_version

    return repository_version()


def read(file_name=LEDGER_FILE):
    path = os.path.join(folder(), file_name)
    if not os.path.exists(path):
        return []
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]
