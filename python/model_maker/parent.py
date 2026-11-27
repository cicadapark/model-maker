"""A pipeline started by another program ends when that program ends.

A program that starts `model-maker make` (a queue that runs jobs one after another, say) names its own process id in
MODEL_MAKER_WORKER_PID. The pipeline then asks Linux for the parent-death signal, so a starter stopped in any way does
not leave it holding the graphics card. The starter cannot set it itself when it is written without `unsafe` code.
Where there is no such signal (macOS, Windows), a thread of the pipeline watches the starter and ends the pipeline
within a second of the starter's end (`processes.watch_parent`).
"""

import os
import signal
import sys

# The starting program's process id, which it gives the pipeline in its environment.
WORKER_PID = "MODEL_MAKER_WORKER_PID"


def die_with_parent(worker=None):
    """When a starting program named itself (in MODEL_MAKER_WORKER_PID, or `worker`), the kernel kills this process
    when that program ends (Linux's parent-death signal). If its parent is already not that program (it ended before
    the signal was set, and this process was given to init or a subreaper), it exits at once. Run by hand (no starter
    named), nothing is set. Elsewhere than Linux, a watch of the starter does the same (`processes.watch_parent`).
    Whether it was set."""
    worker = worker if worker is not None else os.environ.get(WORKER_PID)
    if not worker:
        return False
    if not sys.platform.startswith("linux"):
        return watched(worker)
    try:
        import ctypes

        set_parent_death_signal = 1  # PR_SET_PDEATHSIG
        if ctypes.CDLL(None, use_errno=True).prctl(set_parent_death_signal, int(signal.SIGKILL), 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl")
    except (OSError, AttributeError) as error:
        print(f"model-maker: the parent-death signal could not be set: {error}", file=sys.stderr, flush=True)
        return False
    if str(os.getppid()) != str(worker).strip():
        sys.exit(1)
    return True


def watched(worker):
    """The starter `worker` watched where the kernel has no parent-death signal; whether it is watched. A starter
    already gone (or no longer this process's parent) ends this process at once, as on Linux."""
    from . import processes

    try:
        alive = processes.parent_alive_check(worker)
    except ValueError as error:
        print(f"model-maker: the starting program's id is not a number: {error}", file=sys.stderr, flush=True)
        return False
    if alive is None:
        print("model-maker: the starting program cannot be watched here (psutil is not installed)", file=sys.stderr,
              flush=True)
        return False
    if not alive():
        sys.exit(1)
    return processes.watch_parent(worker, check=alive) is not None
