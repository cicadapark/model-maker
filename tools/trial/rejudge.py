#!/usr/bin/env python3
"""Every finished job of the trials judged again by the checks as they are now, without writing anything into the jobs:
the shape's checks as the finish judges them (`inspection.judged`) and the touch-up's fill (`pipeline.judged_fill`).
Run it with the code before a change and after it, and compare the verdicts (`compare`) against the labels by eye.

    rejudge.py judge ROOT OUT.jsonl [PART/PARTS]    one line a job (ROOT's folders holding a sidecar); a job made
                                                    before the stored shape (no shape.npz) is judged on its .glb
    rejudge.py compare BEFORE.jsonl AFTER.jsonl     each check whose verdict changed, and the counts

The model-maker code judged is the one this file sits beside, or MM_PYTHON's. CPU only.
"""

import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("MM_PYTHON") or os.path.join(HERE, "..", "..", "python"))


def finished_jobs(root):
    found = set()
    for record in glob.glob(os.path.join(root, "**", "*.provenance.json"), recursive=True):
        folder = os.path.dirname(record)
        if os.path.exists(os.path.join(folder, "job.json")):
            found.add(folder)
    return sorted(found)


def judged(folder):
    from PIL import Image

    from model_maker import inspection, pipeline
    from model_maker.inspect_mesh import loaded_mesh

    job = pipeline.Job(folder)
    mask = np.asarray(Image.open(job.path("mask.png"))) > 127
    picture = np.asarray(Image.open(job.path("picture.png")).convert("RGB"))
    if job.has("shape.npz"):
        shape = np.load(job.path("shape.npz"))
        corners, triangles, source = shape["corners"].astype(np.float64), shape["triangles"].astype(np.int64), "shape"
    else:
        corners, triangles = loaded_mesh(job.path(job["slug"] + ".glb"))
        source = "glb"
    found = inspection.judged(job.subject, mask, corners, triangles, picture)
    checks = {c.name: {"passed": bool(c.passed), "counted": bool(c.counted), "value": float(c.value)}
              for c in found.checks}
    measures = {k: found.measures.get(k) for k in ("outline", "holes", "mirror")}
    touch = job.record.get("paint", {}).get("touch_up")
    if source == "shape" and job.has("painted.npz") and (touch or {}).get("views"):
        painted = np.load(job.path("painted.npz"))
        fit, _ = pipeline.judged_fill(folder, painted, touch, corners, triangles)
        fill = inspection.fill_check(fit)
        checks[fill.name] = {"passed": bool(fill.passed), "counted": True, "value": float(fill.value)}
        measures["fill"] = {k: v for k, v in fit.items() if k != "as_painted"}
    return {"source": source, "prompt": job["prompt"], "checks": checks,
            "measures": json.loads(json.dumps(measures, default=float))}


def judge(root, out, part="0/1"):
    index, parts = (int(x) for x in part.split("/"))
    done = set()
    if os.path.exists(out):
        with open(out) as file:
            # A job that could not be judged (say its mesh was not there yet) is judged again; its later line counts.
            done = {r["job"] for r in (json.loads(line) for line in file if line.strip()) if "error" not in r}
    for k, folder in enumerate(finished_jobs(root)):
        name = os.path.relpath(folder, root)
        if k % parts != index or name in done:
            continue
        started = time.time()
        try:
            row = judged(folder)
        except Exception as error:  # said in the output, never hidden
            row = {"error": f"{type(error).__name__}: {error}"}
        row.update(job=name, seconds=round(time.time() - started, 1))
        with open(out, "a") as file:
            file.write(json.dumps(row) + "\n")
        print(name, row.get("error") or [c for c, v in row["checks"].items() if v["counted"] and not v["passed"]],
              flush=True)


def rows(path):
    with open(path) as file:
        return {r["job"]: r for r in (json.loads(line) for line in file if line.strip())}


def compare(before_path, after_path):
    before, after = rows(before_path), rows(after_path)
    changed, counts = [], {"jobs": 0, "fail to pass": 0, "pass to fail": 0, "errors": 0}
    for name in sorted(set(before) & set(after)):
        b, a = before[name], after[name]
        if "error" in b or "error" in a:
            counts["errors"] += 1
            print("error:", name, b.get("error"), a.get("error"))
            continue
        counts["jobs"] += 1
        for check in sorted(set(b["checks"]) | set(a["checks"])):
            was = b["checks"].get(check, {}).get("passed", True) or not b["checks"].get(check, {}).get("counted", True)
            now = a["checks"].get(check, {}).get("passed", True) or not a["checks"].get(check, {}).get("counted", True)
            if was != now:
                key = "fail to pass" if now else "pass to fail"
                counts[key] += 1
                changed.append((name, check, key))
    for name, check, key in changed:
        print(f"{key:12}  {check:34}  {name}")
    print(json.dumps(counts))


def main():
    if sys.argv[1] == "judge":
        judge(sys.argv[2], sys.argv[3], *(sys.argv[4:5]))
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
