"""From a prompt to a checked, painted .glb, in stages. Each object has its own folder (its job), and each stage
writes its results there, so a stage done is not done again and a batch runs stage by stage: every picture first
(the picture model loaded once), then every shape, then every painting, then the CPU's part (bake, checks, export,
report). Each stage's time is kept per object.

    pictures: several tries painted, each checked; the best kept (another round when none passes)
    shape:    the shape from the picture, brought to its triangle budget, set to its size, checked; another shape
              seed when a check fails, the best kept
    paint:    six views painted to fit the shape, back onto it
    finish:   the texture baked (its coverage checked), the .glb with its marks, the sidecar, the report
"""

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from . import ledger, registry, words
from .kept import Kept

MODEL_FOLDERS = {"flux1-schnell": "flux1-schnell", "birefnet": "birefnet", "triposg": "triposg",
                 "mv-adapter": "mv-adapter", "sdxl-base-1.0": "sdxl-base-1.0", "sdxl-vae": "sdxl-vae-fp16-fix",
                 "siglip2-so400m": "siglip2-so400m-patch14-384"}
MODELS_USED = ["flux1-schnell", "birefnet", "triposg", "mv-adapter-ig2mv-sdxl", "sdxl-base-1.0", "sdxl-vae-fp16-fix"]
# SigLIP 2: whether each try shows the thing asked for (subject_judge), and for a thing with a lid, cover or doors,
# whether they are shut or open as asked (state_judge). A thing with a lid, cover or doors is not made without it; the
# rest are, their tries unjudged (setup lets it be declined).
STATE_MODEL = "siglip2-so400m"
DEFAULT_TRIES = 4
# One more round of tries when none of the first passes; then the best is taken and the job says it failed.
MOST_PICTURE_ROUNDS = 2
DEFAULT_SHAPE_TRIES = 2
SHAPE_FIRST_SEED = 42
PAINT_SEED = 0
DEFAULT_TRIANGLES = 60000
# Every job's random generators start here (`same_random_state`).
JOB_RANDOM_SEED = 20261110
DEFAULT_TEXTURE = 2048


class Settings:
    def __init__(self, models=None, sources=None, tries=DEFAULT_TRIES, shape_tries=DEFAULT_SHAPE_TRIES,
                 triangles=DEFAULT_TRIANGLES, texture=DEFAULT_TEXTURE, verify=True, lead=words.DEFAULT_LEAD,
                 paint_steps=None, picture_rounds=MOST_PICTURE_ROUNDS, touch_up_views=None, touch_up_strength=None):
        from .paint import STEPS
        from .touch_up import MOST_VIEWS, STRENGTH

        self.models = models or os.environ.get("MODEL_MAKER_MODELS", "models")
        self.sources = sources or os.environ.get("MODEL_MAKER_SOURCES", "sources")
        self.tries, self.shape_tries = tries, shape_tries
        self.triangles, self.texture, self.verify = triangles, texture, verify
        self.lead, self.paint_steps, self.picture_rounds = lead, paint_steps or STEPS, picture_rounds
        self.touch_up_views = MOST_VIEWS if touch_up_views is None else touch_up_views
        self.touch_up_strength = STRENGTH if touch_up_strength is None else touch_up_strength
        self.checked = set()

    def folder(self, name):
        return os.path.join(self.models, MODEL_FOLDERS[name])

    def verified(self, model_id, *names, checked=None):
        """The models' folders, each listed file checked against the registry first (`checked`: the files already
        checked, skipped; the warm process keeps one set for its life, a cold run this Settings' own)."""
        if self.verify:
            for name in names:
                registry.verify_folder(model_id, self.folder(name), checked=self.checked if checked is None else checked)
        return [self.folder(name) for name in names]


class JobMismatch(Exception):
    """A job folder that holds another prompt's job, or the same prompt's made with other options."""


def job_folder(out, prompt, size_m=None, stands=None, symmetric=None):
    """A job's folder under `out`: its prompt's slug and a short hash of the prompt as read and the options given, so
    two prompts or two sizes never share a folder (by the slug alone, "a long wooden dining table, dark oak" and "...,
    pale pine" would share one, and every prompt with no Latin letters would be "object": the second `make` would
    return the first's model as its own)."""
    key = json.dumps([words.cleaned(prompt), size_m, stands, symmetric])
    return os.path.join(out, f"{words.slug_of(prompt)}-{hashlib.sha256(key.encode()).hexdigest()[:8]}")


class Job:
    """One object's folder and its record (job.json)."""

    def __init__(self, folder, prompt=None, size_m=None, stands=None, symmetric=None, settings=None):
        self.folder = folder
        self.set_aside = False
        path = os.path.join(folder, "job.json")
        if os.path.exists(path):
            with open(path) as file:
                self.record = json.load(file)
            if prompt is not None:
                self.same_order(prompt, size_m, stands, symmetric)
            return
        if prompt is None:
            raise FileNotFoundError(f"{path}: no job here")
        subject = words.subject_of(prompt)
        if stands is not None:
            subject.stands = stands
            subject.why["stands"] = f"{'stands' if stands else 'does not stand'}: set by --stands"
        if symmetric is not None:
            subject.symmetric = (subject.symmetric or ("across" if subject.is_weapon else "mirror")) if symmetric \
                else ""
            subject.why["symmetric"] = f"{'checked' if symmetric else 'not checked'}: set by --symmetric"
        self.record = {"prompt": subject.prompt, "slug": subject.slug, "subject": subject.__dict__,
                       "size_m": float(size_m or subject.size_m), "seconds": {}, "status": "started",
                       "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "options": {"size_m": size_m, "stands": stands, "symmetric": symmetric}}
        if settings is not None:
            self.record["settings"] = ledger.settings_of(settings)
        os.makedirs(folder, exist_ok=True)
        self.save()

    def same_order(self, prompt, size_m, stands, symmetric):
        """Refuses a folder whose job was made from another prompt or with other options."""
        asked = {"prompt": words.cleaned(prompt), "size_m": size_m, "stands": stands, "symmetric": symmetric}
        kept = {"prompt": self.record["prompt"], **self.record.get("options", {"size_m": size_m, "stands": stands,
                                                                               "symmetric": symmetric})}
        if "options" not in self.record and size_m is not None and float(size_m) != self.record["size_m"]:
            kept["size_m"] = self.record["size_m"]
        if asked != kept:
            raise JobMismatch(f"{self.folder} holds the job {kept}, not {asked}: give another --out")

    def __getitem__(self, key):
        return self.record[key]

    @property
    def subject(self):
        subject = words.Subject(**self.record["subject"])
        # A choice the record predates is read from the words as they are read now.
        if "lies" not in self.record["subject"]:
            now = words.subject_of(self.record["prompt"])
            subject.lies = now.lies and subject.noun == now.noun
            if "lies" in now.why:
                subject.why["lies"] = now.why["lies"]
        return subject

    def path(self, *parts):
        return os.path.join(self.folder, *parts)

    def has(self, name):
        return os.path.exists(self.path(name))

    def save(self):
        with open(self.path("job.json.new"), "w") as file:
            json.dump(self.record, file, indent=1, default=float)
        os.replace(self.path("job.json.new"), self.path("job.json"))

    def timed(self, stage, seconds):
        self.record["seconds"][stage] = round(seconds, 1)
        self.save()


def written(path, write):
    """`write(partial)` to a file beside `path`, then moved onto `path` in one step, so a kill mid-write leaves no
    file that looks finished (a stage is done when its file is there, so a cut-short archive would make every later
    run skip the stage and crash on it)."""
    base, ending = os.path.splitext(path)
    partial = f"{base}.partial{ending}"
    write(partial)
    os.replace(partial, path)


def same_random_state():
    """Every job, cold (`make`) or warm (the warm process's next job), starts from the same random state: torch's
    generators (the CPU's and every card's), numpy's and Python's own, seeded alike. Every model's own sampling is
    seeded by the job (the picture's, the shape's, the paint's), so this only makes sure that any draw nothing seeds
    is the same both ways; a new process's own torch state is random (torch seeds itself from the system), so a warm
    job could not copy a cold one's otherwise."""
    import random

    random.seed(JOB_RANDOM_SEED)
    np.random.seed(JOB_RANDOM_SEED)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(JOB_RANDOM_SEED)


def needs_card():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("this stage needs an NVIDIA card (CUDA); the finish stage runs without one")


def loaded(models, stage, what, report):
    """A stage's models loaded: said and written to the ledger. A warm process that had them loads nothing and writes
    nothing (docs/metrics.md)."""
    if models.loaded_any:
        report(f"{what} loaded in {models.loaded_seconds:.0f} s")
        ledger.loaded(stage, models.loaded_seconds)


def set_aside(job, stage, error, report):
    """A batch's job whose stage raised: its record says so and the later stages leave it (its ledger line, "aborted",
    is written already); the other jobs go on."""
    job.set_aside = True
    job.record["status"] = f"aborted at {stage}: {type(error).__name__}: {error}"[:400]
    job.save()
    report(f"{job['slug']}: {job.record['status']}")


def each_job(todo, stage, settings, work, report, keep_going, after=None):
    """`work(job, found)` for each job, in its ledger stage. Without `keep_going` (`make`, the warm process) an error
    ends the run; in a batch the job is set aside and the others go on, so later prompts still get their lines, and
    the batch its `run` line and times."""
    for job in todo:
        # Each job's stage starts from the one random state, so a batch's job is drawn as a `make`'s is (a batch
        # interleaves the stages of its jobs, so one state set once would not do).
        same_random_state()
        try:
            with ledger.stage(job, stage, settings) as found:
                result = work(job, found)
        except Exception as error:
            if not keep_going:
                raise
            set_aside(job, stage, error, report)
            continue
        if after is not None:
            after(job, result)


def loading(todo, stage, settings, report, keep_going, load):
    """A stage's models loaded by `load()`; when that raises (out of the card's memory, a file refused), every job
    waiting for them gets an "aborted" line, and in a batch is set aside: None is returned."""
    try:
        return load()
    except Exception as error:
        for job in todo:
            ledger.aborted(job, stage, error, settings)
        if not keep_going:
            raise
        for job in todo:
            set_aside(job, stage, error, report)
        return None


def pictures_stage(jobs, settings, report=print, kept=None, keep_going=False):
    todo = [j for j in jobs if not j.has("picture.png")]
    if not todo:
        return
    needs_card()
    # A cold run's models are this stage's alone; a warm process's are kept between jobs (see kept).
    models = (kept or Kept()).counting()

    def load():
        judge = models.state_judge(settings) if model_present(settings, STATE_MODEL, STATE_MODEL) else None
        return models.picture_painter(settings), models.cutter(settings), judge

    found = loading(todo, "pictures", settings, report, keep_going, load)
    if found is None:
        return
    paint, cutter, judge = found
    loaded(models, "pictures", "picture models", report)
    if judge is None:
        report(NO_SUBJECT_JUDGE)
    each_job(todo, "pictures", settings, lambda job, found: picture_job(job, settings, paint, cutter, judge, found,
                                                                        report), report, keep_going)
    models.done_with("pictures")


def picture_job(job, settings, paint, cutter, judge, found, report):
    from . import painter, picture_checks, state_judge, subject_judge

    started = time.time()
    subject = job.subject
    prompt = words.picture_prompt(subject, settings.lead)
    first = painter.seed_of(prompt)
    os.makedirs(job.path("pictures"), exist_ok=True)
    # Each try is judged to show the thing asked for, and any quality its words name, and one that does not is not
    # chosen (10 November 2026: every war hammer chosen had an axe-like head, which this check refuses; docs/objects.md,
    # "War hammers"; a curved dagger painted straight: "Qualities the words name").
    directions = subject_judge.directions_of(judge, subject) if judge else None
    made = []
    for round_number in range(settings.picture_rounds):
        seeds = range(first + round_number * settings.tries, first + (round_number + 1) * settings.tries)
        tried = painter.tries(paint, cutter, prompt, seeds, report)
        if directions is not None:
            for seed, picture, mask, result in tried:
                subject_judge.add_subject_check(result, subject, *subject_judge.judged(judge, picture, mask,
                                                                                        directions))
                report(f"picture {seed}: {result.checks[-1].words}")
        if subject.state:
            wanted, unwanted = words.state_texts(subject)
            for seed, picture, mask, result in tried:
                state_judge.add_state_check(result, subject, judge.share(picture, mask, wanted, unwanted))
                report(f"picture {seed}: {result.checks[-1].words}")
        made += tried
        if any(result.passed for _, _, _, result in made):
            break
    rounds = round_number + 1
    for seed, picture, mask, result in made:
        picture.save(job.path("pictures", f"{seed}.png"))
        Image.fromarray(mask.astype(np.uint8) * 255).save(job.path("pictures", f"{seed}-mask.png"))
    best = picture_checks.best_try([result for _, _, _, result in made])
    seed, picture, mask, result = made[best]
    job.record["picture"] = {"prompt": prompt, "lead": settings.lead, "seed": seed, "passed": result.passed,
                             "state_judged": bool(subject.state), "subject_judged": directions is not None,
                             "rounds": rounds,
                             "tries": {str(s): r.as_dict() for s, _, _, r in made}}
    job.timed("pictures", time.time() - started)
    # The record and the cut-out first, the picture (which marks the stage done) last, each whole or not at all.
    written(job.path("mask.png"), lambda path: shutil.copyfile(job.path("pictures", f"{seed}-mask.png"), path))
    written(job.path("picture.png"), lambda path: shutil.copyfile(job.path("pictures", f"{seed}.png"), path))
    found.update({"tries": len(made), "tries_passing": sum(r.passed for _, _, _, r in made), "rounds": rounds,
                  "chosen_passed": result.passed,
                  "faults": sorted({c.name for _, _, _, r in made for c in r.checks if not c.passed})})
    report(f"{job['slug']}: picture {seed} chosen ({'passes' if result.passed else 'fails'} its checks)")


def shape_stage(jobs, settings, report=print, kept=None, keep_going=False):
    todo = [j for j in jobs if not j.has("shape.npz")]
    if not todo:
        return
    needs_card()
    # Loaded before the first job's stage, so its time is a load, not counted in the first shape's time.
    models = (kept or Kept()).counting()
    shape_net = loading(todo, "shape", settings, report, keep_going, lambda: models.shape_models(settings))
    if shape_net is None:
        return
    loaded(models, "shape", "shape model", report)
    each_job(todo, "shape", settings, lambda job, found: shape_job(job, settings, shape_net, found, report), report,
             keep_going)
    models.done_with("shape")


def shape_job(job, settings, shape_net, found, report):
    from . import cutout, inspection, mesh_repair, object_checks, simplify

    started = time.time()
    rgb = np.asarray(Image.open(job.path("picture.png")).convert("RGB"))
    mask = np.asarray(Image.open(job.path("mask.png"))) > 127
    subject = job.subject
    tries = []
    for k in range(settings.shape_tries):
        seed = SHAPE_FIRST_SEED + k
        dense_corners, dense_triangles, specks = shape_net.make(rgb, cutout.for_shape(mask), seed=seed)
        corners, triangles, _, simplifier = simplify.simplified(dense_corners, dense_triangles, settings.triangles)
        corners = object_checks.set_down(corners.astype(np.float64), job["size_m"], along_length=subject.is_weapon)
        corners, triangles, _ = mesh_repair.repaired(corners, triangles)
        # Judged as the finish will judge it (inspection.judged: carved, laid, levelled); the shape is kept as made.
        judgement = inspection.judged(subject, mask, corners, triangles, rgb)
        checks, measures = judgement.checks, judgement.measures
        measures["specks_left_out"] = len(specks)
        # Which simplifier brought it to its budget: edge collapses keep the shape, the grid fallback may step a thin
        # part's outline.
        measures["simplified_by"] = simplifier
        written(job.path(f"shape-{seed}.npz"), lambda path: np.savez_compressed(path, corners=corners,
                                                                                triangles=triangles))
        tries.append({"seed": seed, "checks": [c.as_dict() for c in checks], "measures": measures,
                      "score": list(inspection.score(checks, measures)),
                      "dense_triangles": int(len(dense_triangles))})
        counted = [c for c in checks if c.counted]
        report(f"{job['slug']}: shape {seed}: {sum(c.passed for c in counted)} of {len(counted)} checks pass")
        if all(c.passed for c in counted):
            break
    best = max(tries, key=lambda t: tuple(t["score"]))
    job.record["shape"] = {"seed": best["seed"], "tries": tries}
    job.timed("shape", time.time() - started)
    # The record first, the shape (which marks the stage done) last, whole or not at all.
    written(job.path("shape.npz"), lambda path: shutil.copyfile(job.path(f"shape-{best['seed']}.npz"), path))
    found.update({"tries": len(tries), "seed": best["seed"],
                  "first_passed": all(c["passed"] for c in tries[0]["checks"] if c.get("counted", True)),
                  "faults": sorted({c["name"] for t in tries for c in t["checks"]
                                    if not c["passed"] and c.get("counted", True)})})


def paint_stage(jobs, settings, report=print, kept=None, keep_going=False):
    todo = [j for j in jobs if not j.has("painted.npz")]
    if not todo:
        return
    needs_card()
    models = (kept or Kept()).counting()
    painter = loading(todo, "paint", settings, report, keep_going, lambda: models.view_painter(settings))
    if painter is None:
        return
    loaded(models, "paint", "view painter", report)
    inpainter = LazyInpainter(models, settings, report)

    def work(job, found):
        inpainter.load_seconds = 0.0
        found["steps"] = settings.paint_steps
        paint_job(job, settings, painter, report, inpainter)
        found["touch_up_views"] = len(job.record["paint"].get("touch_up", {}).get("views", []))
        # The inpainter's loading is a load, not the paint's time (it has its own ledger line).
        found[ledger.NOT_THE_STAGES] = inpainter.load_seconds

    each_job(todo, "paint", settings, work, report, keep_going)
    models.done_with("paint")


class LazyInpainter:
    """The touch-up's inpainter, loaded the first time a job needs it (a warm process's, the first time any job
    did)."""

    def __init__(self, models, settings, report):
        self.models, self.settings, self.report = models, settings, report
        self.load_seconds = 0.0

    def __call__(self, *args):
        made = self.models.counting().inpainter(self.settings)
        loaded(self.models, "touch-up", "inpainter", self.report)
        # Its files' check and its loading are not the paint's time.
        self.load_seconds += self.models.loaded_seconds + self.models.checked_seconds
        return made(*args)


def paint_job(job, settings, painter, report, inpainter=None):
    from . import paint, touch_up

    started = time.time()
    shape = np.load(job.path("shape.npz"))
    rgb = np.asarray(Image.open(job.path("picture.png")).convert("RGB"))
    mask = np.asarray(Image.open(job.path("mask.png"))) > 127
    colours, known, cloud = paint.painted(painter, shape["corners"], shape["triangles"], rgb, mask,
                                          words.paint_prompt(job.subject), seed=PAINT_SEED,
                                          views_out=job.path("views"), report=report,
                                          steps=settings.paint_steps)
    job.record["paint"] = {"steps": settings.paint_steps}
    sources = np.zeros(len(cloud[0]), dtype=np.int16)
    if inpainter is not None and settings.touch_up_views:
        # What the six views missed, painted from views chosen to see it, the fill judged (see touch_up).
        cloud, touched, sources = touch_up.touched_up(
            inpainter, shape["corners"].astype(np.float64), shape["triangles"], colours, cloud,
            words.paint_prompt(job.subject), seed=PAINT_SEED, most=settings.touch_up_views, report=report,
            views_out=job.path("views"))
        touched["strength"] = settings.touch_up_strength
        job.record["paint"]["touch_up"] = touched
    job.timed("paint", time.time() - started - getattr(inpainter, "load_seconds", 0.0))
    # The record first, the paint (which marks the stage done) last, whole or not at all.
    written(job.path("painted.npz"), lambda path: np.savez_compressed(
        path, colours=colours.astype(np.float32), known=known, cloud_points=cloud[0], cloud_normals=cloud[1],
        cloud_colours=cloud[2], cloud_from=sources))


def repository_version():
    """The repository's version: `git describe` in a checkout; in a tarball made by `git archive` (a release's
    download), the commit git wrote into `version.txt` (its `export-subst` attribute); else "unknown"."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        done = subprocess.run(["git", "-C", here, "describe", "--always", "--dirty"], capture_output=True, text=True,
                              check=True)
        return done.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        with open(os.path.join(here, "version.txt")) as file:
            written = file.read().strip()
    except OSError:
        return "unknown"
    return written if written and "$Format" not in written else "unknown"


def finish_stage(jobs, settings, report=print, keep_going=False):
    """The CPU's part: bake, checks, export, sidecar, report. Runs again whenever asked (it is quick)."""

    def work(job, _):
        started = time.time()
        grouped, measures = finished(job, settings, report)
        job.timed("finish", time.time() - started)
        return grouped, measures

    def after(job, result):
        # The job's line after its stage's, as before.
        ledger.finished(job, settings, *result)
        report(f"{job['slug']}: {job.record['status']} ({job.path(job['slug'] + '.glb')})")

    each_job(jobs, "finish", settings, work, report, keep_going, after)


def judged_fill(folder, painted, touch_record, made_corners, made_triangles):
    """The touch-up's fill judged as the finished model shows it: (its measures, the picture of the worst view or
    None). The fill the touch-up kept is judged as kept (`touch_up.fill_fit`). A place it left out (or never painted)
    shows the colour spread to it from round it, so it is judged by the paint round it on the finished model
    (`touch_up.left_out_places`): a large place whose ring is not one colour is a smear. The fill as the inpainter
    painted it, before what was foreign was left out, is shown beside them and not counted: those pixels never reach
    the model (kept in the record, or measured again from the saved tries for a record made before that)."""
    from . import touch_up

    sources = touch_up.fill_sources(painted, touch_record)
    made_cloud = tuple(painted[k].astype(np.float64) for k in ("cloud_points", "cloud_normals", "cloud_colours"))
    filled = sources is not None and (sources > 0).any()
    least = touch_up.filled_misfit(*made_cloud, sources) if filled else None
    fit = touch_up.fill_fit(*made_cloud, sources if sources is not None else np.zeros(len(made_cloud[0])),
                            made_corners, touch_record["views"], least=least)
    fit["judged"] = "as kept"
    fit["left_out"] = touch_up.left_out_places(made_corners, made_triangles,
                                               np.asarray(painted["colours"], dtype=np.float64), made_cloud,
                                               touch_record["views"])
    if touch_record.get("judged"):
        views = os.path.join(folder, "views")
        found = touch_up.as_painted(touch_record) or (
            touch_up.replayed_as_painted(made_corners, made_triangles, painted, touch_record, views)
            if os.path.isdir(views) else None)
        if found is not None:
            fit["as_painted"] = {"views": found[0], "patches": found[1]}
    fill_shown = None
    if filled or fit["left_out"]["largest_place"] > 0:
        worst = touch_up.worst_view(fit)
        view = touch_record["views"][worst]
        fill_shown = (touch_up.fill_picture(made_corners, made_triangles, made_cloud, sources, view["turn_deg"],
                                            view["tilt_deg"], least=least, colours=painted["colours"]),
                      f"the touch-up's view {worst} ({view['turn_deg']}° round, {view['tilt_deg']:.0f}° above) on the "
                      f"finished model: its kept fill green where its colour is found in the paint round it, red "
                      f"foreign ({(fit['views'] or [0.0])[min(worst, len(fit['views']) - 1)]:.1%} of this view's kept "
                      f"fill); places left unpainted larger than {touch_up.LARGEST_FOREIGN_PATCH:.0%} of the object's "
                      f"size ringed, magenta a smear (neither the paint round it nor the colour spread into it one "
                      f"colour), blue not")
    return fit, fill_shown


def finished(job, settings, report=print):
    from . import bake, glb, inspection, likeness, lying, picture_checks as P, render
    from . import state_judge, subject_judge
    from . import report as pages
    from .mesh_tools import corner_normals

    subject = job.subject
    shape = np.load(job.path("shape.npz"))
    made_corners, made_triangles = shape["corners"].astype(np.float64), shape["triangles"].astype(np.int64)
    mask = np.asarray(Image.open(job.path("mask.png"))) > 127
    # Repaired, carved, laid, levelled and checked as the shape stage judged its tries (inspection.judged); the made
    # shape is kept as it was.
    picture = Image.open(job.path("picture.png")).convert("RGB")
    judgement = inspection.judged(subject, mask, made_corners, made_triangles, np.asarray(picture))
    corners, triangles, laid, level = judgement.corners, judgement.triangles, judgement.laid, judgement.level
    checks, measures, overlay = judgement.checks, judgement.measures, judgement.overlay
    painted = None
    if job.has("painted.npz"):
        painted = np.load(job.path("painted.npz"))
        colours, cloud = judgement.paint_followed(made_corners, painted)
    else:
        report(f"{job['slug']}: not painted; baked in plain clay")
        colours, cloud = np.full((len(corners), 3), 0.75), None
    baked = bake.baked(corners, triangles, colours, cloud, settings.texture)
    paint_checks = [inspection.coverage_check(baked["coverage"], baked["coverage_all_texels"])] if cloud is not None \
        else []
    views = [job.path("views", f"view-{k}.png") for k in range(6)]
    if cloud is not None and all(os.path.exists(path) for path in views):
        # The coverage counts a texel painted whatever its colour: a view that came back black would pass it.
        shares = [inspection.black_share(np.asarray(Image.open(path).convert("RGB")) / 255) for path in views]
        measures["views_black"] = shares
        paint_checks.append(inspection.views_check(shares))
    fill_shown = None
    touch_record = job.record.get("paint", {}).get("touch_up")
    if cloud is not None and (touch_record or {}).get("views"):
        fit, fill_shown = judged_fill(job.folder, painted, touch_record, made_corners, made_triangles)
        measures["fill"] = fit
        paint_checks.append(inspection.fill_check(fit))
    measures["bake"] = {"coverage": baked["coverage"], "coverage_all_texels": baked["coverage_all_texels"],
                        "texels_per_m": baked["texels_per_m"],
                        "overlap": baked["overlap"], "side": settings.texture}
    triangle_uvs = baked["uvs"][baked["triangles"]]
    painted_at = render.texture_lookup(triangle_uvs, baked["texture"])
    likeness_picture = None
    if cloud is not None:
        outline = measures["outline"]
        difference, within, likeness_picture = likeness.likeness(
            np.asarray(picture), mask, lying.to_made(baked["corners"], laid), baked["triangles"], painted_at,
            outline["turn_deg"], outline["tilt_deg"], outline.get("distance"))
        measures["likeness"] = {"mean_delta_e": difference, "share_within_10": within}
    normals = corner_normals(corners, triangles)[baked["original"]]
    marks = inspection.marks_of(measures)
    # A bow's string, by our own code after the checks (they judge the shape model's work): see weapon_checks.
    parts = []
    if subject.weapon_kind == "bow":
        from .weapon_checks import STRING_COLOUR, STRING_RADIUS_M, bow_string

        string = bow_string(corners, triangles)
        if "corners" in string:
            parts.append({"name": "string", "corners": string["corners"], "triangles": string["triangles"],
                          "colour": STRING_COLOUR})
            measures["string"] = {"made_by": "model-maker's own code (a cord between the tips found)",
                                  "from": string["from"], "to": string["to"], "brace_m": string["brace_m"],
                                  "radius_m": STRING_RADIUS_M}
        else:
            measures["string"] = {"drawn": False, "brace_m": string["brace_m"], "why": string["why"]}
    # The chosen picture checked again, so a measure corrected since it was painted counts; SigLIP 2's verdicts (the
    # thing asked for, and the state) need its model and are kept from when the picture was chosen.
    picture_result = P.checked(np.asarray(picture), mask)
    chosen = job.record.get("picture", {})
    chosen_try = chosen.get("tries", {}).get(str(chosen.get("seed")), {})
    for name in (subject_judge.NAME, state_judge.NAME):
        kept_check = state_judge.check_from_record(chosen_try, name)
        if kept_check is not None:
            picture_result.checks.append(kept_check)
    grouped = {"picture": [c.as_dict() for c in picture_result.checks], "shape": [c.as_dict() for c in checks],
               "paint": [c.as_dict() for c in paint_checks]}
    every = [c for group in grouped.values() for c in group]
    every = [c for c in every if c.get("counted", True)]
    job.record["status"] = "passes every check" if all(c["passed"] for c in every) else \
        "fails: " + "; ".join(c["name"] for c in every if not c["passed"])
    judged_by_siglip = chosen.get("state_judged") or chosen.get("subject_judged")
    models_used = registry.used_record(MODELS_USED + ([STATE_MODEL] if judged_by_siglip else []))
    slug = job["slug"]
    provenance = {
        "made_by": "model-maker", "version": repository_version(), "made_at": job["made_at"],
        "prompt": job["prompt"], "picture_words": job.record.get("picture", {}).get("prompt"),
        "paint_words": words.paint_prompt(subject),
        "paint_steps": job.record.get("paint", {}).get("steps"),
        "picture_lead": job.record.get("picture", {}).get("lead", "floating"),
        "seeds": {"picture": job.record.get("picture", {}).get("seed"),
                  "shape": job.record.get("shape", {}).get("seed"), "paint": PAINT_SEED},
        "subject": {k: v for k, v in job["subject"].items() if k != "prompt"},
        "size_m": job["size_m"], "models": models_used,
        # Which simplifier brought the kept shape to its budget (the tool's words; None for a job made before it was
        # recorded).
        "simplified_by": next((t["measures"].get("simplified_by") for t in job.record.get("shape", {}).get("tries", [])
                               if t["seed"] == job.record.get("shape", {}).get("seed")), None),
        "levelled": level if level else {"trimmed": False, "words": "not a thing that stands: not levelled"},
        "laid": laid,
        "carved": measures["carved"],
        "note": "Made by AI models from a text prompt. The use restrictions of the CreativeML Open RAIL++-M licence "
                "(Stable Diffusion XL, its Attachment A) apply to this model.",
        "marks": {name: {"at": [round(float(x), 4) for x in where], "up": [round(float(x), 4) for x in up]}
                  for name, (where, up, _) in marks.items()},
        "status": job.record["status"], "checks": grouped, "seconds": job["seconds"],
        "measures": json.loads(json.dumps(measures, default=float)),
    }
    with open(job.path(f"{slug}.provenance.json"), "w") as file:
        json.dump(provenance, file, indent=1, default=float)
    data = glb.glb_bytes(slug, baked["corners"], baked["triangles"], normals, baked["uvs"], baked["texture"],
                         marks=marks, parts=parts, extras={"made_by": "model-maker", "prompt": job["prompt"],
                                              "provenance": f"{slug}.provenance.json",
                                              "status": job.record["status"]})
    with open(job.path(f"{slug}.glb"), "wb") as file:
        file.write(data)
    pictures = {"picture.png": (pages.thumbnail(picture, 360), "the picture it was made from"),
                "picture-cut.png": (pages.thumbnail(pages.mask_picture(mask), 360), "its cut-out")}
    for seed, result in job.record.get("picture", {}).get("tries", {}).items():
        if os.path.exists(job.path("pictures", f"{seed}.png")):
            words_said = "passes" if result["passed"] else "fails: " + "; ".join(
                c["name"] for c in result["checks"] if not c["passed"] and c.get("counted", True))
            pictures[f"try-{seed}.png"] = (pages.thumbnail(Image.open(job.path("pictures", f"{seed}.png")), 180),
                                           f"try {seed}: {words_said}")
    pictures["outline.png"] = (overlay, "outlines: green both, red the picture's alone, blue the model's alone")
    pictures["turntable.png"] = (render.turntable(baked["corners"], baked["triangles"], painted_at),
                                 "painted, turned in steps of 45° from the front")
    pictures["bare.png"] = (render.turntable(corners, triangles), "the bare shape, the same turns")
    pictures["coverage.png"] = (pages.thumbnail(baked["coverage_picture"], 360),
                                "the texture: green painted by a view, red colour spread from neighbours")
    if fill_shown is not None:
        pictures["fill.png"] = fill_shown
    if likeness_picture is not None:
        found = measures["likeness"]
        pictures["coverage-likeness.png"] = (
            likeness_picture, f"likeness: the picture, the painted model from the same view, their difference (green "
            f"alike, red unlike); mean colour difference {found['mean_delta_e']:.1f}, "
            f"{found['share_within_10']:.0%} of it within 10")
    if measures.get("weapon"):
        pictures["marks.png"] = (pages.drawn_marks(baked["corners"], baked["triangles"], painted_at, measures),
                                 "the grip (blue line, the ring where the hand goes) and the tip (red)")
    for k in range(6):
        if job.has(os.path.join("views", f"view-{k}.png")):
            pictures[f"views-{k}.png"] = (pages.thumbnail(Image.open(job.path("views", f"view-{k}.png")), 200),
                                          f"painted view {k}")
    pages.written(job.path("report"), job.record, grouped, measures, pictures, models_used)
    job.save()
    return grouped, measures


# A thing with a lid, a cover or doors is not made without the state judge (setup lets SigLIP 2 be declined).
NEEDS_STATE_MODEL = ("needs SigLIP 2, which judges whether a lid, a cover or doors are shut or open as asked; it is not "
                     "downloaded here (run ./setup.sh again and accept it)")


# Without SigLIP 2 the other things are made, each try unjudged against its prompt; the run says so.
NO_SUBJECT_JUDGE = ("SigLIP 2 is not downloaded: the pictures are not checked to show the thing asked for (run "
                    "./setup.sh again and accept it)")


def model_present(settings, name, model_id):
    """Whether every file the registry lists for a model is in its folder (its sums are checked when it loads)."""
    folder = settings.folder(name)
    return all(os.path.isfile(os.path.join(folder, path)) for path in registry.sums_of(registry.models()[model_id]))


def not_made(job, why, report):
    job.record["status"] = f"not made: {why}"
    job.save()
    ledger.not_made(job, why)
    report(f"{job['prompt']}: {why}")


def no_picture_passed(job, settings, report):
    """A job none of whose pictures passed its checks ends there: the finish would fail it on them, after the shape
    and the paint had spent some 6 minutes of the card. Its line says which failed."""
    chosen = job.record["picture"]
    checks = chosen.get("tries", {}).get(str(chosen.get("seed")), {}).get("checks", [])
    failed = [c["name"] for c in checks if not c["passed"] and c.get("counted", True)]
    job.record["status"] = "fails: " + "; ".join(failed or ["the picture's checks"])
    job.save()
    job.set_aside = True
    ledger.finished(job, settings, {"picture": checks}, {})
    report(f"{job['slug']}: no picture passed its checks; not shaped ({job.record['status']})")


def run(jobs, settings, report=print, stages=("pictures", "shape", "paint", "finish"), kept=None, keep_going=False):
    """Each stage over the jobs, but for things not made yet (each says why, and is left as it is). `kept`: the warm
    process's models, kept between jobs (see kept); a cold run loads each stage's own. `keep_going` (a batch): a job
    whose stage raises is set aside and the rest go on."""
    for job in jobs:
        if job.subject.not_yet:
            not_made(job, job.subject.not_yet, report)
    jobs = [job for job in jobs if not job.subject.not_yet]
    # Without SigLIP 2 a chest's picture cannot be judged; such jobs are not made, the rest are (the missing folder
    # must not end a whole batch before any job).
    if "pictures" in stages and not model_present(settings, STATE_MODEL, STATE_MODEL):
        for job in jobs:
            if job.subject.state and not job.has("picture.png"):
                not_made(job, NEEDS_STATE_MODEL, report)
                job.set_aside = True
    for stage in stages:
        jobs = [job for job in jobs if not job.set_aside]
        if stage in ("shape", "paint"):
            for job in jobs:
                if not job.has("shape.npz") and job.record.get("picture", {}).get("passed") is False:
                    no_picture_passed(job, settings, report)
            jobs = [job for job in jobs if not job.set_aside]
        if stage == "finish":
            finish_stage(jobs, settings, report, keep_going)
        else:
            {"pictures": pictures_stage, "shape": shape_stage, "paint": paint_stage}[stage](jobs, settings, report, kept,
                                                                                         keep_going)
