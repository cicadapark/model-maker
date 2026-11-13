"""model-maker's command line.

    model-maker make "an old looking chair" [--out DIR] [--size 0.9] [--tries 4] ...
    model-maker batch PROMPTS.txt --out DIR [...]       many prompts, stage by stage, a time per object
    model-maker finish DIR [DIR ...]                     the CPU's part again for made jobs (bake, checks, report)
    model-maker inspect MESH --picture PNG --prompt TEXT --out DIR [--size M] [--mask PNG] [--recut]
                                                         the object checks on a mesh made elsewhere (CPU)
    model-maker picture-check PNG --prompt TEXT --out DIR [--mask PNG]
                                                         does the picture show what the prompt asks for (CPU)
    model-maker words "a double-bitted battle axe"       how a prompt is read: its kind and which checks apply
    model-maker bug-report JOB_FOLDER [--out FILE] [--with-model]
                                                         one file for a problem report, nothing personal in it
    model-maker reproduce REPORT_FILE --out DIR          a report's job made again here, and compared with it
    model-maker credentials [--make-test] [--read PICTURE ...]
                                                         which certificate signs the pictures' content credentials

The models are read from --models (or MODEL_MAKER_MODELS), the view painter's code (MV-Adapter) from --sources (or
MODEL_MAKER_SOURCES): see docs/objects.md. Every picture is written with its content credentials (C2PA), signed with
--signing-key and --signing-cert (or MODEL_MAKER_SIGNING_KEY and MODEL_MAKER_SIGNING_CERT), else with a test
certificate made on this computer: see docs/objects.md, "Content credentials".
"""

import argparse
import csv
import json
import os
import sys
import time

# Every model is read from its checked folder, never fetched: set before any of the hub's libraries is imported, as
# they read it then (the `model-maker` command sets it too).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# No bytecode cached beside the models' own code: their folders hold only what the registry lists (registry.py).
sys.dont_write_bytecode = True

from . import pipeline, words  # noqa: E402


def yes_no(text):
    if text in ("yes", "no"):
        return text == "yes"
    raise argparse.ArgumentTypeError("yes or no")


def common(parser):
    parser.add_argument("--models", help="the models' folder (default MODEL_MAKER_MODELS or ./models)")
    parser.add_argument("--sources", help="the view painter's code, MV-Adapter (default MODEL_MAKER_SOURCES or "
                        "./sources)")
    parser.add_argument("--tries", type=int, default=pipeline.DEFAULT_TRIES, help="pictures painted per round")
    parser.add_argument("--shape-tries", type=int, default=pipeline.DEFAULT_SHAPE_TRIES,
                        help="shapes made at most when a shape check fails")
    parser.add_argument("--triangles", type=int, default=pipeline.DEFAULT_TRIANGLES, help="the triangle budget")
    parser.add_argument("--texture", type=int, default=pipeline.DEFAULT_TEXTURE, help="the texture's side in texels")
    parser.add_argument("--no-verify", action="store_true", help="skip the models' sha256 check (it was done before)")
    parser.add_argument("--stages", default="pictures,shape,paint,finish",
                        help="which stages to run (a stage done is kept)")
    parser.add_argument("--lead", choices=sorted(words.LEADS), default=words.DEFAULT_LEAD,
                        help="the words the picture model is given first (about the light and the backdrop)")
    parser.add_argument("--picture-rounds", type=int, default=pipeline.MOST_PICTURE_ROUNDS,
                        help="rounds of tries when none passes")
    parser.add_argument("--paint-steps", type=int, help="the view painter's steps (default 20; 50 was the first trial's)")
    parser.add_argument("--touch-up-views", type=int,
                        help="views added to paint what the six missed, at most (default 4; 0 for none)")
    parser.add_argument("--touch-up-strength", type=float,
                        help="how much of a touch-up view's start the inpainter repaints (default 0.6; 1 from noise)")
    parser.add_argument("--require-lettering", action="store_true",
                        help="read every picture for painted lettering: without tesseract 5 or newer and its English "
                        "data a job is not made (the measured run uses it)")
    signing_options(parser)


def signing_options(parser):
    parser.add_argument("--signing-key", help="the PEM private key that signs the pictures' content credentials "
                        "(default MODEL_MAKER_SIGNING_KEY; else a test certificate made on this computer)")
    parser.add_argument("--signing-cert", help="its certificate chain, PEM, the signer's own first (default "
                        "MODEL_MAKER_SIGNING_CERT)")


def use_signing(args):
    """The key and certificate that sign every picture of this run; a pair that does not work ends it here."""
    from . import credentials

    missing = credentials.missing_packages()
    if missing:
        print(f"model-maker: {', '.join(missing)} not installed: the pictures' content credentials need them (new in "
              f"0.2.1). Run ./setup.sh again.", file=sys.stderr)
        sys.exit(2)
    try:
        credentials.use(credentials.chosen(getattr(args, "signing_key", None), getattr(args, "signing_cert", None)))
    except credentials.NotSigned as error:
        print(f"model-maker: {error}", file=sys.stderr)
        sys.exit(2)


def warn_unverified(args):
    if args.no_verify:
        print("model-maker: --no-verify: the models' files are not checked against their sha256 sums", file=sys.stderr)


def settings_of(args):
    warn_unverified(args)
    use_signing(args)
    return pipeline.Settings(models=args.models, sources=args.sources, tries=args.tries, shape_tries=args.shape_tries,
                             triangles=args.triangles, texture=args.texture, verify=not args.no_verify,
                             lead=args.lead, paint_steps=args.paint_steps, picture_rounds=args.picture_rounds,
                             touch_up_views=args.touch_up_views, touch_up_strength=args.touch_up_strength,
                             require_lettering=args.require_lettering)


def report(line):
    print(line, flush=True)


def make(args):
    from .parent import die_with_parent

    # Started by another program that named itself (MODEL_MAKER_WORKER_PID): dies with it.
    die_with_parent()
    code, _ = made(args.prompt, args.out, args.size, args.stands, args.symmetric, settings_of(args),
                   args.stages.split(","))
    return code


def made(prompt, out, size, stands, symmetric, settings, stages, kept=None):
    """One object made into `out/<its slug>/`: (the command's exit code, the job or None). `kept`: models kept between
    jobs by a process that makes many (see kept.py); the run is warm when it holds any already."""
    from . import ledger

    # Begun before anything else, so even a thing not made has its own run (not the last job's).
    ledger.run_began(1, warm=bool(kept is not None and kept.models))
    pipeline.same_random_state()
    subject = words.subject_of(prompt)
    if subject.not_yet:
        print(subject.not_yet)
        # A prompt refused has its own line, so every prompt asked is counted.
        ledger.refused(subject.prompt, out, subject.not_yet)
        return 2, None
    folder = pipeline.job_folder(out, prompt, size, stands, symmetric)
    job = pipeline.Job(folder, prompt, size, stands, symmetric, settings=settings)
    started = time.time()
    try:
        pipeline.run([job], settings, report, stages, kept)
    except pipeline.NotEnoughMemory as short:
        print(f"model-maker: {short}", file=sys.stderr)
        return pipeline.NOT_ENOUGH_MEMORY_EXIT, None
    print(f"{job['prompt']}: {job.record['status']}{unread_words(job)} in {time.time() - started:.0f} s; "
          f"{os.path.join(folder, job['slug'] + '.glb')}, report {os.path.join(folder, 'report', 'report.html')}")
    return (0 if job.record["status"] == "passes every check" else 1), job


def unread_words(job):
    """What a job's status leaves out: its chosen picture not read for painted lettering."""
    lettering = job.record.get("lettering")
    return "" if lettering is None or lettering["judged"] else " (not read for painted lettering)"


def prompts_in(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip() and not line.lstrip().startswith("#")]


def batch(args):
    from . import ledger

    settings = settings_of(args)
    jobs = {}
    for prompt in prompts_in(args.prompts):
        folder = pipeline.job_folder(args.out, prompt, args.size)
        if folder in jobs:
            # The same prompt twice is one job, counted once (two Jobs on one folder would overwrite each other).
            print(f"{prompt}: asked again; made once ({folder})")
            continue
        jobs[folder] = pipeline.Job(folder, prompt, args.size, settings=settings)
    jobs = list(jobs.values())
    ledger.run_began(max(1, sum(not job.subject.not_yet for job in jobs)))
    started = time.time()
    # A job whose stage fails is set aside with its own ledger line; the batch goes on, and its times are written.
    try:
        pipeline.run(jobs, settings, report, args.stages.split(","), keep_going=True)
    except pipeline.NotEnoughMemory as short:
        print(f"model-maker: {short}", file=sys.stderr)
        ledger.run_ended([job.record["id"] for job in jobs if "id" in job.record])
        return pipeline.NOT_ENOUGH_MEMORY_EXIT
    # The batch's whole time and loading, once, for its jobs to share (docs/metrics.md).
    ledger.run_ended([job.record["id"] for job in jobs if "id" in job.record])
    with open(os.path.join(args.out, "times.csv"), "w", newline="") as file:
        out = csv.writer(file)
        out.writerow(["prompt", "status", "pictures_s", "shape_s", "paint_s", "finish_s", "total_s"])
        for job in jobs:
            s = job["seconds"]
            out.writerow([job["prompt"], job.record["status"], *(s.get(k, "") for k in
                                                                  ("pictures", "shape", "paint", "finish")),
                          round(sum(s.values()), 1)])
    passed = sum(job.record["status"] == "passes every check" for job in jobs)
    unread = sum(job.record["status"] == "passes every check" and bool(unread_words(job)) for job in jobs)
    print(f"{passed} of {len(jobs)} pass every check"
          f"{f' ({unread} of them not read for painted lettering)' if unread else ''}; "
          f"{time.time() - started:.0f} s in all; "
          f"{os.path.join(args.out, 'times.csv')}")
    return 0


def finish(args):
    use_signing(args)
    jobs = [pipeline.Job(folder) for folder in args.folders]
    # The ledger keeps each job's own settings (its record's); only the texture is this run's.
    pipeline.finish_stage(jobs, pipeline.Settings(texture=args.texture, verify=False), report, keep_going=True)
    return 0


def inspect(args):
    from .inspect_mesh import inspected

    path = inspected(args.mesh, args.picture, args.prompt, args.out, args.size, args.mask, args.recut)
    print(path)
    return 0


def check_picture(args):
    from .subject_judge import picture_check

    record = picture_check(args.picture, args.prompt, args.out, args.mask)
    for check in record["checks"]:
        print(f"{'passes' if check['passed'] else 'FAILS '} {check['name']}: {check['words']}")
    print(record["status"])
    return 0


def read_words(args):
    subject = words.subject_of(args.prompt)
    print(json.dumps({**subject.__dict__, "slug": subject.slug, "picture_words": words.picture_prompt(subject)},
                     indent=1))
    return 0


def bug_report(args):
    from . import bug_report as reports

    return reports.main_bug_report(args)


def reproduce(args):
    from . import bug_report as reports

    warn_unverified(args)
    use_signing(args)

    def remake(order, out):
        settings = pipeline.Settings(models=args.models, sources=args.sources, verify=not args.no_verify,
                                     **order.settings)
        _, job = made(order.prompt, out, order.size_m, order.stands, order.symmetric, settings,
                      ["pictures", "shape", "paint", "finish"])
        return job.folder if job is not None else None

    try:
        reports.reproduce(args.report, args.out, remake)
    except reports.Refused as refused:
        print(f"model-maker reproduce: {refused}", file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="model-maker", description="Make a checked 3-D object from a prompt.")
    steps = parser.add_subparsers(dest="step", required=True)
    one = steps.add_parser("make", help="make one 3-D object")
    one.add_argument("prompt")
    one.add_argument("--out", default="made",
                     help="the folder its job's folder goes in, made/<name>-<id>/ by default (the id: a hash of the "
                          "prompt and options)")
    one.add_argument("--size", type=float, help="its largest side in metres (an object held in the hand: its length)")
    one.add_argument("--stands", type=yes_no, help="yes/no: check that it stands on its base")
    one.add_argument("--symmetric", type=yes_no, help="yes/no: check its mirror symmetry")
    common(one)
    many = steps.add_parser("batch", help="make every prompt of a file (one a line), stage by stage")
    many.add_argument("prompts")
    many.add_argument("--out", required=True, help="the folder the jobs' folders and times.csv go in; run "
                      "again with the same --out, a batch goes on where it stopped (each job's finished stages kept)")
    many.add_argument("--size", type=float, help="every object's largest side (default: each its usual)")
    common(many)
    again = steps.add_parser("finish", help="bake, check, export and report made jobs again (CPU)")
    again.add_argument("folders", nargs="+")
    again.add_argument("--texture", type=int, default=pipeline.DEFAULT_TEXTURE)
    signing_options(again)
    look = steps.add_parser("inspect", help="the object checks on a mesh made elsewhere (CPU)")
    look.add_argument("mesh")
    look.add_argument("--picture", required=True)
    look.add_argument("--mask", help="the picture's cut-out (white the object); else the picture's transparency or "
                      "its backdrop's colour")
    look.add_argument("--prompt", required=True)
    look.add_argument("--out", required=True)
    look.add_argument("--size", type=float)
    look.add_argument("--recut", action="store_true",
                      help="cut the picture again with our own BiRefNet (CPU) and judge by that cut-out; the --mask "
                           "sent must agree with it")
    seen = steps.add_parser("picture-check", help="does the picture show what the prompt asks for (SigLIP 2, CPU)")
    seen.add_argument("picture")
    seen.add_argument("--prompt", required=True)
    seen.add_argument("--out", required=True)
    seen.add_argument("--mask", help="the picture's cut-out (white the object), for the second view, a crop round it")
    said = steps.add_parser("words", help="how a prompt is read")
    said.add_argument("prompt")
    told = steps.add_parser("bug-report", help="one file for a problem report: the job's record, sidecar, report and "
                            "versions, nothing personal")
    told.add_argument("folder", help="the job's folder (the one with its .glb and report/)")
    told.add_argument("--out", help="the file to write, .zip or .tar.gz (default bug-report-<name>-<id>.zip here)")
    told.add_argument("--with-model", action="store_true", help="put the .glb in too")
    told.add_argument("--ledger", help="the ledger (jobs.jsonl, or its folder) if not the usual one")
    again_made = steps.add_parser("reproduce", help="make a bug report's job again here and compare (needs a card)")
    again_made.add_argument("report", help="the bug report file (.zip or .tar.gz)")
    again_made.add_argument("--out", required=True, help="where the remake and the comparison go")
    again_made.add_argument("--models", help="the models' folder (default MODEL_MAKER_MODELS or ./models)")
    again_made.add_argument("--sources", help="the view painter's code, MV-Adapter (default MODEL_MAKER_SOURCES or "
                            "./sources)")
    again_made.add_argument("--no-verify", action="store_true", help="skip the models' sha256 check")
    signing_options(again_made)
    steps.add_parser("credentials", help="which certificate signs the pictures' content credentials, and what a "
                     "picture's say (its own options: model-maker credentials --help)")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["credentials"]:
        from . import credentials

        return credentials.main(argv[1:])
    args = parser.parse_args(argv)
    from .registry import MODEL_MISSING_EXIT, ModelMissing

    try:
        return {"make": make, "batch": batch, "finish": finish, "inspect": inspect, "words": read_words,
                "picture-check": check_picture, "bug-report": bug_report, "reproduce": reproduce}[args.step](args)
    except ModelMissing as missing:
        print(f"model-maker: {missing}", file=sys.stderr)
        return MODEL_MISSING_EXIT


if __name__ == "__main__":
    sys.exit(main())
