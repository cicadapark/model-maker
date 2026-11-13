#!/usr/bin/env python3
"""A batch's memory, stage by stage, from its sampled memory and its log (docs/objects.md, "A long batch").

    batch_memory.py MEMORY.tsv BATCH.log

MEMORY.tsv: seconds, the batch's resident memory (MB), the machine's available memory (MB) and the log's line count at
each sample (the trials' sample.py). Each stage begins at the log's first line of it: the pictures at the start, the
shape at "shape model loaded", the paint at "view painter loaded", the finish at the first job's verdict. For each
stage: its peak, and the memory at each of its jobs' ends (a job's end: its line in the log), so growth from job to
job shows."""

import re
import sys

STARTS = (("shape", re.compile(r"^shape model loaded")), ("paint", re.compile(r"^view painter loaded")),
          ("finish", re.compile(r": (passes every check|fails: )")))
JOB_ENDS = {"pictures": re.compile(r": picture \d+ chosen"), "shape": re.compile(r"^shape: \d+ corners"),
            "paint": re.compile(r"^touch-up view 0|^views from seed"), "finish": re.compile(r": (passes every check|fails: )")}


def main():
    samples = [tuple(int(float(x)) for x in line.split("\t")) for line in open(sys.argv[1]) if line.strip()]
    with open(sys.argv[2], "rb") as file:
        lines = [l.decode("utf-8", "replace").split("\r")[-1].strip() for l in file.read().split(b"\n")]
    starts = [("pictures", 0)]
    for name, pattern in STARTS:
        at = next((k for k, line in enumerate(lines) if pattern.search(line)), None)
        if at is not None:
            starts.append((name, at))
    starts.sort(key=lambda s: s[1])
    for k, (name, begin) in enumerate(starts):
        end = starts[k + 1][1] if k + 1 < len(starts) else len(lines) + 1
        inside = [s for s in samples if begin <= s[3] < end]
        if not inside:
            continue
        peak = max(s[1] for s in inside)
        least_free = min(s[2] for s in inside)
        ends = [k2 for k2 in range(begin, min(end, len(lines))) if JOB_ENDS[name].search(lines[k2])]
        at_ends = []
        for line_number in ends:
            after = [s for s in samples if s[3] > line_number]
            if after:
                at_ends.append(after[0][1])
        print(f"{name:9} peak {peak / 1024:5.1f} GB, least free {least_free / 1024:5.1f} GB; at its jobs' ends (GB): "
              + " ".join(f"{m / 1024:.1f}" for m in at_ends))
    print(f"whole batch: peak {max(s[1] for s in samples) / 1024:.1f} GB over {samples[-1][0] / 60:.0f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
