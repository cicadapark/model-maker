"""The computer's memory as a batch uses it: what this process holds, its peak, what is free, and the guard that
refuses to start a stage the memory cannot hold (docs/objects.md, "A long batch").

A stage's need is measured as it runs: the process's peak resident memory while the stage ran (the kernel's
high-water mark, reset at the stage's start where the kernel allows it). Each batch writes the needs it measured
into its own folder (`memory.json`), and a batch started again in that folder (a resumed one) is refused a stage its
own measured need does not fit. `KNOWN_NEEDS_MB`, measured on the trials' machine, only warns: it is the peak with
droppable file pages counted, and PCs with less memory than it ran 0.2.0.
"""

import json
import os

# Each stage's peak, measured on the 16 GB card's batch of eight (2026-11-12-030-batch, the process's resident memory,
# MB, rounded up): the picture model (FLUX.1 kept in the computer's memory between its steps on the card), the shape
# model, the view painter with the inpainter, and the finish's CPU work. A batch measures its own (`record_need`).
KNOWN_NEEDS_MB = {"pictures": 22400, "shape": 13700, "paint": 20200, "finish": 2800}
# The need as measured, with nothing added: the peak resident memory already counts the model files' pages mapped in,
# which the kernel can drop and read again, so it is a little over the memory a stage cannot do without. Measured on
# the 16 GB card's container: its picture stage peaked at 21.6 GB in a run that began with 24.2 GB free, and never had
# less than 6.8 GB free.
HEADROOM = 1.0
NEEDS_FILE = "memory.json"
# Set to 0 to start a stage whatever memory is free (a machine whose memory the kernel cannot report).
CHECK_VARIABLE = "MODEL_MAKER_MEMORY_CHECK"


def _status(field):
    try:
        with open("/proc/self/status") as file:
            for line in file:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def in_use_mb():
    """This process's resident memory now (MB), or None where the kernel does not say."""
    return _status("VmRSS")


def peak_mb():
    """This process's peak resident memory (MB) since it started or since `reset_peak`, or None."""
    return _status("VmHWM")


def reset_peak():
    """The peak set back to what is resident now (Linux 4.0 and later: writing 5 to clear_refs); False when not
    allowed, and the peak then counts from the process's start."""
    try:
        with open("/proc/self/clear_refs", "w") as file:
            file.write("5")
        return True
    except OSError:
        return False


def free_mb():
    """The memory a new allocation can have without the kernel swapping or killing (MemAvailable, MB), or None."""
    try:
        with open("/proc/meminfo") as file:
            for line in file:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def give_back():
    """Freed memory handed back to the system. A stage's models closed and a job's arrays dropped are freed, but glibc
    keeps a freed heap for the process unless told (malloc_trim), and torch keeps the pinned memory it has freed for
    its next copy to the card: measured, closing the shape model left 3.5 of its 4.6 GB held, and malloc_trim gave all
    but 0.3 GB back (docs/objects.md, "A long batch"). Elsewhere than glibc, the trim is skipped."""
    import gc
    import sys

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch._C._host_emptyCache()
        except (AttributeError, RuntimeError):
            pass
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def measured_needs(folder):
    """Each stage's need (MB) as a batch in `folder` measured it before; {} when none was."""
    try:
        with open(os.path.join(folder, NEEDS_FILE)) as file:
            return {stage: int(need) for stage, need in json.load(file).items()}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def needs(folder):
    """Each stage's need (MB): the larger of `KNOWN_NEEDS_MB` and what a batch in `folder` measured before."""
    found = dict(KNOWN_NEEDS_MB)
    try:
        with open(os.path.join(folder, NEEDS_FILE)) as file:
            measured = json.load(file)
    except (OSError, ValueError):
        measured = {}
    for stage, need in measured.items():
        found[stage] = max(found.get(stage, 0), int(need))
    return found


def record_need(folder, stage, need_mb):
    """A stage's measured need written into the batch's folder (the larger of it and what was there)."""
    path = os.path.join(folder, NEEDS_FILE)
    try:
        with open(path) as file:
            measured = json.load(file)
    except (OSError, ValueError):
        measured = {}
    measured[stage] = max(int(measured.get(stage, 0)), int(need_mb))
    with open(path + ".partial", "w") as file:
        json.dump(measured, file, indent=1)
    os.replace(path + ".partial", path)


def room_for(stage, need_mb):
    """(whether the stage may start, the words why): it may when the memory free now and what this process holds
    (it is the process that grows) cover the need with `HEADROOM`, or when the check is off or cannot be made."""
    if os.environ.get(CHECK_VARIABLE, "1") == "0":
        return True, f"the memory check is off ({CHECK_VARIABLE}=0)"
    free, held = free_mb(), in_use_mb()
    if free is None or held is None or not need_mb:
        return True, "the memory free is not known here"
    wanted = need_mb * HEADROOM
    have = free + held
    words = (f"the {stage} stage needs about {wanted / 1024:.1f} GB of the computer's memory; "
             f"{have / 1024:.1f} GB is free to this run ({free / 1024:.1f} GB free, {held / 1024:.1f} GB its own)")
    return have >= wanted, words
