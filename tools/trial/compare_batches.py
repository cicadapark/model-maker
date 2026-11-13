#!/usr/bin/env python3
"""Two batches of the same prompts compared job by job: the seeds chosen (picture, shape, paint), the verdict, and
every check's value and pass, from each job's sidecar (docs/objects.md, "A long batch").

    compare_batches.py BATCH_A BATCH_B     each a batch's --out folder

Prints one line a job, "same" or what differs, and exits 1 when anything differs. A check's value counts as the same
within 0.01 of its unit (a value printed to the sidecar's four places can round either way on another run)."""

import glob
import json
import os
import sys

CLOSE = 0.01


def sidecars(folder):
    found = {}
    for path in glob.glob(os.path.join(folder, "*", "*.provenance.json")):
        with open(path) as file:
            found[os.path.basename(os.path.dirname(path))] = json.load(file)
    return found


def differences(a, b):
    found = []
    for name in ("picture", "shape", "paint"):
        if a["seeds"].get(name) != b["seeds"].get(name):
            found.append(f"{name} seed {a['seeds'].get(name)} against {b['seeds'].get(name)}")
    if a["status"] != b["status"]:
        found.append(f"verdict {a['status']!r} against {b['status']!r}")
    for stage in sorted(set(a["checks"]) | set(b["checks"])):
        mine = {c["name"]: c for c in a["checks"].get(stage, [])}
        theirs = {c["name"]: c for c in b["checks"].get(stage, [])}
        for name in sorted(set(mine) | set(theirs)):
            x, y = mine.get(name), theirs.get(name)
            if x is None or y is None:
                found.append(f"{stage} '{name}' only in {'the first' if y is None else 'the second'}")
            elif x["passed"] != y["passed"] or abs(float(x["value"]) - float(y["value"])) > CLOSE:
                found.append(f"{stage} '{name}' {x['value']} ({'pass' if x['passed'] else 'fail'}) against "
                             f"{y['value']} ({'pass' if y['passed'] else 'fail'})")
    return found


def main():
    first, second = sidecars(sys.argv[1]), sidecars(sys.argv[2])
    differ = False
    for job in sorted(set(first) | set(second)):
        if job not in first or job not in second:
            print(f"{job}: made in only {'the second' if job not in first else 'the first'}")
            differ = True
            continue
        found = differences(first[job], second[job])
        differ |= bool(found)
        print(f"{job}: " + ("same (" + first[job]["status"] + ")" if not found else "; ".join(found)))
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
