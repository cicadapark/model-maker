"""Processes on each system: a program model-maker starts (the question judge's server) stopped and tied to it, and
model-maker tied to the program that started it.

On Linux everything is as it was measured: the parent-death signal (prctl), a session of the child's own stopped
with SIGTERM then SIGKILL, the server's libraries found through LD_LIBRARY_PATH, its memory read from /proc. On
macOS the child's own session and the signals are as on Linux; there is no parent-death signal, so model-maker
watches its starter instead, and a child is not yet tied to model-maker (a killed model-maker can leave it running).
On Windows a child goes into a job object that kills its members when model-maker's
handle to it closes (however model-maker ends), and is stopped with TerminateProcess.
"""

import ctypes
import os
import signal
import subprocess
import sys
import threading

LINUX = sys.platform.startswith("linux")
WINDOWS = os.name == "nt"
MACOS = sys.platform == "darwin"


def program_name(name, windows=WINDOWS):
    """A program's file name on this system (`.exe` on Windows)."""
    return name + ".exe" if windows and not name.lower().endswith(".exe") else name


def library_environment(folder, environ=None, system=sys.platform):
    """`environ` with `folder` as where a program finds the libraries beside it: LD_LIBRARY_PATH on Linux (as it was,
    the folder alone), DYLD_LIBRARY_PATH on macOS, the front of PATH on Windows."""
    environ = dict(os.environ if environ is None else environ)
    if system.startswith("win"):
        environ["PATH"] = folder + (os.pathsep + environ["PATH"] if environ.get("PATH") else "")
    elif system == "darwin":
        environ["DYLD_LIBRARY_PATH"] = folder
    else:
        environ["LD_LIBRARY_PATH"] = folder
    return environ


# --- a child: started in its own group, tied to model-maker, stopped --------------------------------------------------


def start_options(preexec=None):
    """Popen's options for a child stopped as a group: its own session on Linux and macOS (with `preexec`, run in
    the child before it starts: Linux's parent-death signal), its own process group on Windows."""
    if WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True, "preexec_fn": preexec}


def stop(process, wait_seconds=20):
    """A started child stopped: asked (SIGTERM to its group; on Windows TerminateProcess), then made to (SIGKILL), each
    given `wait_seconds` to end."""
    if WINDOWS:
        for ending in (process.terminate, process.kill):
            try:
                ending()
            except OSError:
                break
            try:
                process.wait(timeout=wait_seconds)
                break
            except subprocess.TimeoutExpired:
                continue
        return
    for sent in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sent)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=wait_seconds)
            break
        except subprocess.TimeoutExpired:
            continue


# Windows' job object: JOBOBJECT_EXTENDED_LIMIT_INFORMATION with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. Defined with
# ctypes' plain types so its layout (144 bytes on 64-bit Windows) can be checked on any system.
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


class _BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount",
                                                      "OtherOperationCount", "ReadTransferCount",
                                                      "WriteTransferCount", "OtherTransferCount")]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


def kill_on_close_limits():
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    return limits


_JOB = None


def _windows_job():
    """model-maker's one job object, made on first use and never closed: when model-maker ends, however it ends,
    Windows closes the handle and kills every process in it."""
    global _JOB
    if _JOB is None:
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = kill_on_close_limits()
        if not kernel32.SetInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits),
                                                ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        _JOB = (kernel32, job)
    return _JOB


def tie_to_this_process(process):
    """On Windows, `process` (a Popen) put into model-maker's job object, so it dies with model-maker. Elsewhere the
    tie is set as the child starts (`start_options`'s preexec), and this does nothing. Whether it was tied here."""
    if not WINDOWS:
        return False
    try:
        kernel32, job = _windows_job()
        if not kernel32.AssignProcessToJobObject(job, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
    except (OSError, AttributeError) as error:
        print(f"model-maker: the started program could not be tied to this one: {error}", file=sys.stderr, flush=True)
        return False
    return True


def memory_of(pid):
    """A process's memory (MB): {"VmHWM", "VmRSS", "RssAnon", "RssFile"} from /proc on Linux; elsewhere what psutil
    gives (now, and the peak on Windows), the rest left out. None when it cannot be read."""
    found = {}
    try:
        with open(f"/proc/{pid}/status") as file:
            for line in file:
                key, _, value = line.partition(":")
                if key in ("VmHWM", "VmRSS", "RssAnon", "RssFile"):
                    found[key] = int(value.split()[0]) // 1024
        return found
    except (OSError, ValueError, IndexError):
        if LINUX:
            return None
    try:
        import psutil
    except ImportError:
        return None
    try:
        info = psutil.Process(pid).memory_info()
    except (OSError, psutil.Error):
        return None
    found["VmRSS"] = info.rss // 2 ** 20
    if getattr(info, "peak_wset", None) is not None:
        found["VmHWM"] = info.peak_wset // 2 ** 20
    return found


# --- model-maker tied to its starter ---------------------------------------------------------------------------------


def parent_alive_check(worker, psutil_module=None, getppid=os.getppid):
    """A function saying whether the starting program `worker` (its process id) still runs: psutil's process (which
    also tells a new process given the same id), else, on Linux and macOS, whether it is still this process's parent
    (an ended parent's children are given to another). None when neither can tell (Windows without psutil)."""
    worker = int(worker)
    psutil = psutil_module
    if psutil is None:
        try:
            import psutil
        except ImportError:
            psutil = None
    if psutil is not None:
        try:
            starter = psutil.Process(worker)
        except (OSError, psutil.Error):  # gone already
            return lambda: False
        return starter.is_running
    if WINDOWS:
        return None
    return lambda: getppid() == worker


def watch_parent(worker, interval_seconds=1.0, ended=None, check=None):
    """A thread that ends this process (`ended`, default: exit at once with status 1) when the starting program
    `worker` stops running: where the kernel has no parent-death signal. A function that stops the watch, or None
    when nothing here can tell whether the starter runs."""
    alive = check or parent_alive_check(worker)
    if alive is None:
        return None
    ended = ended or (lambda: os._exit(1))
    stop_watching = threading.Event()

    def watch():
        while not stop_watching.wait(interval_seconds):
            if not alive():
                ended()
                return

    threading.Thread(target=watch, name="model-maker parent watch", daemon=True).start()
    return stop_watching.set
