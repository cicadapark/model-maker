"""A pipeline started by another program ends when that program ends.

A program that starts `model-maker make` (a queue that runs jobs one after another, say) names its own process id in
MODEL_MAKER_WORKER_PID. The pipeline then asks Linux for the parent-death signal, so a starter stopped in any way does
not leave it holding the graphics card. The starter cannot set it itself when it is written without `unsafe` code.
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
    named), nothing is set. Whether it was set."""
    worker = worker if worker is not None else os.environ.get(WORKER_PID)
    if not worker:
        return False
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
