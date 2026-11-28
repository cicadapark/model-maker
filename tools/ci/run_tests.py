#!/usr/bin/env python3
"""The unit tests with each test module in a process of its own, as many at once as the computer has processors: the
same tests as `python -m unittest discover -s tests -t .` run in python/, the same verdicts, in a fraction of the
time on CI's three- and four-processor machines (whose minutes are what CI costs).

    PYTHON -I tools/ci/run_tests.py [--workers N] [--log FILE] [--module-limit MINUTES] [--progress SECONDS]

Runs the modules with the Python that runs this script. Prints each module's verdict, tests and seconds as it ends
(so a long run shows its progress), every --progress seconds the modules still running and for how long, then every
skipped test with its reason, the whole log of each module that failed, and the totals; exits 1 if any module failed
(or ran no tests). A module still running after --module-limit minutes is stopped (with the processes it started)
and FAILED, named as over the limit, with its output so far: a hung module ends the run instead of holding it for
ever. With --log (setup's self-test) all of that goes to the file, and only the verdicts, the progress lines and the
totals are printed.
"""

import argparse
import locale
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

PYTHON_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python")
SKIPPED = re.compile(r" \.\.\. skipped ")
RAN = re.compile(r"^Ran (\d+) tests? in ", re.MULTILINE)
# Started first, the slowest first, so the last to end is not a long one started late (measured on four processors,
# 25 November 2026: 774, 401, 310 and 309 s of 2,278 s in all). Only the order depends on this list, never a verdict.
SLOWEST = ("test_touch_up", "test_carving", "test_object_checks", "test_finish")


def modules(folder=PYTHON_FOLDER):
    found = sorted(name[:-3] for name in os.listdir(os.path.join(folder, "tests"))
                   if name.startswith("test_") and name.endswith(".py"))
    return [name for name in SLOWEST if name in found] + [name for name in found if name not in SLOWEST]


# The slowest module took 774 s on CI's four processors (25 November 2026); a module still running after this many
# minutes is taken as hung (the Windows test of 26 November 2026 could not tell slow from hung: no time limit).
MODULE_LIMIT_MINUTES = 30
PROGRESS_SECONDS = 60


def descendants(pid):
    """The processes `pid` started, and theirs, where /proc lists them (Linux); none found elsewhere."""
    parents = {}
    for entry in os.listdir("/proc") if os.path.isdir("/proc") else ():
        try:
            with open(f"/proc/{entry}/stat") as file:
                parents.setdefault(int(file.read().rsplit(")", 1)[1].split()[1]), []).append(int(entry))
        except (OSError, ValueError, IndexError):
            continue
    found, waiting = [], [pid]
    while waiting:
        children = parents.get(waiting.pop(), [])
        found += children
        waiting += children
    return found


def stop_with_its_children(process):
    """Stops a module's process and the processes it started (a hung test may wait on one of them)."""
    if os.name == "nt":
        # taskkill /T ends the whole tree; it is part of every Windows.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=60, check=False)
    else:
        for pid in descendants(process.pid):
            try:
                os.kill(pid, 9)
            except OSError:
                pass
    process.kill()
    process.wait()


def run(module, folder=PYTHON_FOLDER, limit_seconds=None):
    """(module, passed, tests ran, skipped lines, seconds, log, over the limit). The output goes to a file, not a pipe,
    so a module stopped at the limit still gives all it wrote, even if a process it started holds the file open."""
    started = time.time()
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen([sys.executable, "-m", "unittest", "-v", f"tests.{module}"], cwd=folder,
                                   stdout=output, stderr=subprocess.STDOUT)
        try:
            process.wait(timeout=limit_seconds)
            over_the_limit = False
        except subprocess.TimeoutExpired:
            stop_with_its_children(process)
            over_the_limit = True
        output.seek(0)
        # What text=True decoded with before: the locale's encoding, Windows' line ends made "\n".
        log = output.read().decode(locale.getpreferredencoding(False), errors="replace").replace("\r\n", "\n")
    seconds = time.time() - started
    if over_the_limit:
        log += (f"\n===== stopped after {seconds:.0f} s, over the limit of {limit_seconds:.0f} s per module: the "
                f"test after the last one above that ended is the one that hung (or was too slow)\n")
    ran = RAN.search(log)
    count = int(ran.group(1)) if ran else 0
    skipped = [line for line in log.splitlines() if SKIPPED.search(line)]
    passed = process.returncode == 0 and count > 0 and not over_the_limit
    return module, passed, count, skipped, seconds, log, over_the_limit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--log", help="the whole report to this file (setup's self-test); then only each module's "
                                      "verdict, the progress lines and the totals are printed")
    parser.add_argument("--module-limit", type=float, default=MODULE_LIMIT_MINUTES, metavar="MINUTES",
                        help="a module still running after this many minutes is stopped and FAILED, named as over "
                             f"the limit (default {MODULE_LIMIT_MINUTES}; 0: no limit)")
    parser.add_argument("--progress", type=float, default=PROGRESS_SECONDS, metavar="SECONDS",
                        help=f"every this many seconds, the modules still running (default {PROGRESS_SECONDS})")
    options = parser.parse_args(argv)
    if options.module_limit < 0 or options.progress <= 0:
        parser.error("--module-limit must be 0 or more and --progress more than 0")
    limit_seconds = options.module_limit * 60 or None
    log = open(options.log, "w", encoding="utf-8") if options.log else None
    lock = threading.Lock()

    def said(line, shown=True):
        with lock:
            if log:
                log.write(line + "\n")
                log.flush()
            if shown or not log:
                print(line, flush=True)

    running = {}

    def timed(module):
        with lock:
            running[module] = time.time()
        try:
            return run(module, limit_seconds=limit_seconds)
        finally:
            with lock:
                running.pop(module, None)

    started = time.time()
    names = modules()
    results = []
    said(f"{len(names)} test modules, {options.workers} at once, "
         + (f"each stopped after {options.module_limit:g} min" if limit_seconds else "no time limit"))
    with ThreadPoolExecutor(options.workers) as pool:
        waiting = {pool.submit(timed, module) for module in names}
        while waiting:
            ended, waiting = wait(waiting, timeout=options.progress, return_when=FIRST_COMPLETED)
            for future in ended:
                module, passed, count, skipped, seconds, output, over_the_limit = result = future.result()
                results.append(result)
                verdict = "ok" if passed else "FAILED"
                if over_the_limit:
                    verdict += f" (stopped: over the limit of {options.module_limit:g} min, hung or too slow)"
                said(f"tests.{module}: {verdict}, {count} tests, {len(skipped)} skipped, "
                     f"{seconds:.0f} s ({len(results)} of {len(names)})")
            if not ended:
                now = time.time()
                with lock:
                    still = sorted(running.items(), key=lambda item: item[1])
                said(f"still running after {now - started:.0f} s: "
                     + ", ".join(f"tests.{module} ({now - since:.0f} s)" for module, since in still)
                     + f" ({len(results)} of {len(names)} ended)")
    results.sort()
    for result in results:
        for line in result[3]:
            said(line, shown=False)
    failed = [result for result in results if not result[1]]
    for module, _, _, _, _, output, _ in failed:
        said(f"\n===== tests.{module} failed:\n{output}", shown=False)
    over = [result[0] for result in failed if result[6]]
    said(f"\n{sum(r[2] for r in results)} tests in {len(results)} modules, "
         f"{sum(len(r[3]) for r in results)} skipped, {len(failed)} modules failed, "
         f"{time.time() - started:.0f} s with {options.workers} at once")
    if over:
        said("over the time limit (stopped): " + ", ".join(f"tests.{module}" for module in over))
    if log:
        log.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
