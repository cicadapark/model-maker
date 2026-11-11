#!/usr/bin/env python3
"""A trial's results side by side, from its jobs' records and sidecars.

    python3 -I tools/trial/summary.py pictures DIR [DIR ...]     picture tries passing, per folder (one per lead)
    python3 -I tools/trial/summary.py paint DIR [DIR ...]        paint time, coverage and likeness, per folder (one per
                                                                 step count)
    python3 -I tools/trial/summary.py checks DIR                 each object's status
    python3 -I tools/trial/summary.py progress [--write FILE]   the headline numbers and a row per batch, from the
                                                                 ledger (MODEL_MAKER_LEDGER or metrics/)
    python3 -I tools/trial/summary.py backfill DIR [DIR ...]     once: trial folders made before the ledger into it,
                                                                 each line marked backfilled

A job whose chosen picture was not read for painted lettering (its record's and its ledger line's `lettering_judged`
false: no usable tesseract) is never counted as a pass: it is shown apart ("not read"). A job made before the check
(no `lettering_judged`) is counted as it was.
"""

import datetime
import json
import os
import statistics
import sys
import uuid
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
CARD_STAGES = ("pictures", "shape", "paint")
WEEK = datetime.timedelta(days=7)


def jobs(folder):
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name, "job.json")
        if os.path.exists(path):
            with open(path) as file:
                yield name, json.load(file)


def sidecar(folder, name, job):
    path = os.path.join(folder, name, f"{job['slug']}.provenance.json")
    if not os.path.exists(path):
        return None
    with open(path) as file:
        return json.load(file)


def pictures(folders):
    for folder in folders:
        total = passing = 0
        faults = Counter()
        rows = []
        for name, job in jobs(folder):
            tries = job.get("picture", {}).get("tries", {})
            ok = sum(t["passed"] for t in tries.values())
            total, passing = total + len(tries), passing + ok
            # A check shown, not counted (lettering not judged), is no fault.
            faults.update(c["name"] for t in tries.values() for c in t["checks"]
                          if not c["passed"] and c.get("counted", True))
            rows.append(f"  {name:30} {ok}/{len(tries)} pass")
        print(f"{folder}: {passing} of {total} tries pass; faults {dict(faults)}")
        print("\n".join(rows))


def paint(folders):
    print(f"{'folder':40} {'object':30} {'steps':>5} {'paint s':>8} {'coverage':>9} {'delta E':>8} {'within 10':>9}")
    for folder in folders:
        for name, job in jobs(folder):
            found = sidecar(folder, name, job)
            if not found:
                continue
            measures = found["measures"]
            likeness = measures.get("likeness", {})
            print(f"{os.path.basename(folder.rstrip('/')):40} {name:30} {found.get('paint_steps', 50):>5} "
                  f"{job['seconds'].get('paint', float('nan')):>8.0f} {measures['bake']['coverage']:>9.1%} "
                  f"{likeness.get('mean_delta_e', float('nan')):>8.1f} {likeness.get('share_within_10', 0):>9.0%}")


def unread(job):
    """Whether a job's record or ledger line says its chosen picture was not read for painted lettering."""
    if "lettering_judged" in job:
        return job["lettering_judged"] is False
    return (job.get("lettering") or {}).get("judged") is False


def checks(folder):
    passing = not_read = 0
    for name, job in jobs(folder):
        if job["status"] == "passes every check" and unread(job):
            not_read += 1
            print(f"{name:30} {job['status']}, but NOT READ for painted lettering: not counted as a pass")
            continue
        passing += job["status"] == "passes every check"
        print(f"{name:30} {job['status']}")
    print(f"{passing} pass every check" + (f"; {not_read} more pass the rest but were not read for painted lettering "
                                           "(not counted)" if not_read else ""))


def backfill(folders):
    """Jobs made before the ledger, read from their job.json and sidecar. Their id comes from their folder (the
    trial folders are not written to), their card is the one this machine sees, and each line says it was back-filled."""
    from model_maker import ledger

    written = 0
    for folder in folders:
        for name, job in jobs(folder):
            path = os.path.abspath(os.path.join(folder, name))
            found = sidecar(folder, name, job)
            common = {"job": job.get("id") or uuid.uuid5(uuid.NAMESPACE_URL, path).hex, "folder": path,
                      "batch": os.path.basename(os.path.abspath(folder)), "prompt": job["prompt"],
                      "slug": job["slug"], "weapon": job["subject"].get("kind") == "weapon",
                      "version": (found or {}).get("version", "unknown"), "machine": ledger.machine(),
                      "settings": {"paint_steps": (found or {}).get("paint_steps"),
                                   "lead": (found or {}).get("picture_lead")},
                      "backfilled": True}
            if job["status"].startswith("not made"):
                ledger.append({**common, "event": "not made", "why": job["status"]})
            elif found:
                every = [c for group in found["checks"].values() for c in group if c.get("counted", True)]
                failed = [c["name"] for c in every if not c["passed"]]
                ledger.append({**common, "event": "job", "status": job["status"], "passed": not failed,
                               "failed": failed, "seconds": job["seconds"],
                               "first_tries": ledger.first_tries(job),
                               "checks": [{"name": c["name"], "value": c.get("value"), "limit": c.get("limit"),
                                           "passed": c["passed"]} for c in every],
                               "levelled_mm": round(1000 * found.get("levelled", {}).get("trimmed_m", 0.0), 1),
                               "made_at": job["made_at"]})
            else:
                ledger.append({**common, "event": "aborted", "stage": "unknown", "error": f"left at {job['status']}"})
            written += 1
    print(f"{written} jobs back-filled into {os.path.join(ledger.folder(), ledger.LEDGER_FILE)}")


def first_per_job(lines):
    """Each job as it first ended (its first `job`, `not made` or `aborted` line): the unseen set counts its first
    finish, so a refinish after a limit moved cannot change its headline."""
    ends = {}
    for line in lines:
        if line["event"] in ("job", "not made", "aborted") and line["job"] not in ends:
            ends[line["job"]] = line
    return list(ends.values())


def latest_per_job(lines):
    """A job finished again (a refinish) counts once, as it last ended; a refinish has no run of its own (it loads no
    model), so it keeps the run the job was made in. A batch's job takes an even share of its batch's whole time and
    loading, from the line the batch wrote at its end."""
    shares = {}
    for line in lines:
        if line["event"] == "run" and line.get("jobs"):
            for job in line["jobs"]:
                shares[job] = {"whole_seconds": round(line["whole_seconds"] / len(line["jobs"]), 1),
                               "load_seconds": round(line["load_seconds"] / len(line["jobs"]), 1)}
    ends = {}
    for line in lines:
        if line["event"] in ("job", "not made", "aborted"):
            run = line.get("run") or ends.get(line["job"], {}).get("run")
            if run and run.get("whole_seconds") is None and line["job"] in shares:
                run = {**run, **shares[line["job"]]}
            ends[line["job"]] = {**line, **({"run": run} if run else {})}
    return list(ends.values())


def batch_row(ends):
    made = [e for e in ends if e["event"] == "job"]
    # A job not read for painted lettering passes nothing: it is shown apart.
    passed = [e for e in made if e["passed"] and not unread(e)]
    minutes = [sum(e["seconds"].values()) / 60 for e in made]
    # The real time a job took, its models' loading with it (the jobs that say their run).
    whole = [e["run"]["whole_seconds"] / 60 for e in made if (e.get("run") or {}).get("whole_seconds") is not None]
    card_minutes = sum(sum(e["seconds"].get(s, 0) for s in CARD_STAGES) for e in made) / 60
    first = [e["first_tries"] for e in made]
    cards = sorted({(e["machine"] or {}).get("card") or "no card" for e in ends})
    return {
        "asked": len(ends), "made": len(made), "passed": len(passed),
        "objects": f"{sum(not e['weapon'] for e in passed)} of {sum(not e['weapon'] for e in made)}",
        "weapons": f"{sum(e['weapon'] for e in passed)} of {sum(e['weapon'] for e in made)}",
        "median_min": statistics.median(minutes) if minutes else None,
        "median_whole_min": statistics.median(whole) if whole else None,
        "warm": f"{sum(e['run']['warm'] for e in made if (e.get('run') or {}).get('whole_seconds') is not None)} "
                f"of {len(whole)}",
        "card_min_per_pass": card_minutes / len(passed) if passed else None,
        "picture_first_round": sum(f["picture_rounds"] == 1 for f in first),
        "shape_first_seed": sum(bool(f["shape_first_passed"]) for f in first),
        "aborted": sum(e["event"] == "aborted" for e in ends),
        "not_read": sum(unread(e) for e in made),
        "faults": Counter(name for e in made for name in e["failed"]).most_common(3),
        "versions": sorted({e["version"] for e in ends}), "cards": cards,
        "backfilled": any(e.get("backfilled") for e in ends),
    }


def share(part, whole):
    return f"{part} of {whole} ({part / whole:.0%})" if whole else "none"


def progress(write=None):
    from model_maker import ledger

    lines = ledger.read()
    ends = latest_per_job(lines)
    by_batch = defaultdict(list)
    for end in ends:
        by_batch[end["batch"]].append(end)
    regressions = [r for r in ledger.read(ledger.REGRESSION_FILE) if r.get("whole", True)]
    now = datetime.datetime.now(datetime.timezone.utc)
    this_week = [e for e in ends if e["event"] == "job" and e["passed"] and not unread(e)
                 and now - datetime.datetime.strptime(e["made_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
                     tzinfo=datetime.timezone.utc) <= WEEK]
    out = ["# model-maker: progress", "",
           f"From the ledger ({len(ends)} jobs, {len(by_batch)} batches), written by the pipeline. Made "
           f"{now:%Y-%m-%d %H:%M} UTC by `tools/trial/summary.py progress`; do not edit by hand.", "",
           "## Headline", ""]
    unseen = sorted(b for b in by_batch if "unseen" in b)
    if unseen:
        firsts = defaultdict(list)
        for end in first_per_job(lines):
            firsts[end["batch"]].append(end)
        # Every unseen batch is shown, each as its jobs first ended (the headline) and as they last ended, with the
        # versions that made each: a second run, or a refinish after a limit moved, shows beside the first.
        for batch in unseen:
            first, last = batch_row(firsts[batch]), batch_row(by_batch[batch])
            out.append(f"- Unseen set ({batch}): {share(first['passed'], first['asked'])} of prompts asked pass every "
                       f"check as first finished ({', '.join(first['versions'])}); as last finished "
                       f"{share(last['passed'], last['asked'])} ({', '.join(last['versions'])})"
                       + (f"; NOT READ for painted lettering, so not counted as passes: {first['not_read']} as first "
                          f"finished, {last['not_read']} as last" if first["not_read"] or last["not_read"] else ""))
    else:
        out.append("- Unseen set: not run yet (it runs once per release, after the code is frozen)")
    if regressions:
        last = regressions[-1]
        out.append(f"- Regression set: {last['held']} held, {last['broken']} broken ({last['version']}, {last['at']})")
    out.append(f"- Models made this week that pass every check: {len(this_week)}")
    out += ["", "## Per batch", "",
            "| Batch | Versions | Card | Pass of asked | Objects | Held in the hand | Median min | Median whole min (warm) | "
            "Card min per pass | Picture 1st round | Shape 1st seed | Aborted | Not read for lettering | Top faults |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for batch in sorted(by_batch):
        row = batch_row(by_batch[batch])
        median = f"{row['median_min']:.1f}" if row["median_min"] is not None else "-"
        per_pass = f"{row['card_min_per_pass']:.1f}" if row["card_min_per_pass"] is not None else "-"
        whole = f"{row['median_whole_min']:.1f} ({row['warm']})" if row["median_whole_min"] is not None else "-"
        faults = ", ".join(f"{name} {count}" for name, count in row["faults"]) or "-"
        out.append(f"| {batch}{' (back-filled)' if row['backfilled'] else ''} | {', '.join(row['versions'])} | "
                   f"{', '.join(row['cards'])} | {share(row['passed'], row['asked'])} | {row['objects']} | "
                   f"{row['weapons']} | {median} | {whole} | {per_pass} | {row['picture_first_round']} of {row['made']} | "
                   f"{row['shape_first_seed']} of {row['made']} | {row['aborted']} | {row['not_read']} | {faults} |")
    text = "\n".join(out) + "\n"
    print(text, end="")
    if write:
        with open(write, "w") as file:
            file.write(text)


if __name__ == "__main__":
    what, rest = sys.argv[1], sys.argv[2:]
    if what == "progress":
        progress(rest[1] if rest[:1] == ["--write"] else None)
    else:
        {"pictures": pictures, "paint": paint, "checks": lambda f: checks(f[0]), "backfill": backfill}[what](rest)
