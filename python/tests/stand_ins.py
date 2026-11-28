"""Stand-in programs for the tests: Python code that runs as a named program, on every system, and a way to tell
whether a process still runs."""

import importlib.util
import os
import platform
import shutil
import stat
import sys


def program(path, source):
    """A program at `path` (`.exe` added on Windows) that runs the Python `source`; its path is returned. On Linux
    and macOS a script with its #! line. Windows starts only programs, so there it is setuptools' console launcher
    (installed with the lock), which runs `<name>-script.py` beside it with the interpreter its #! line names: the
    bare interpreter, so the stand-in is one process under the launcher (the source may use the standard library
    only)."""
    if os.name != "nt":
        with open(path, "w", encoding="utf-8") as file:
            file.write(f"#!{sys.executable}\n{source}")
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        return path
    setuptools = os.path.dirname(importlib.util.find_spec("setuptools").origin)
    launcher = "cli-arm64.exe" if platform.machine().upper() == "ARM64" else "cli-64.exe"
    shutil.copyfile(os.path.join(setuptools, launcher), path + ".exe")
    interpreter = getattr(sys, "_base_executable", sys.executable)
    with open(path + "-script.py", "w", encoding="utf-8") as file:
        file.write(f'#!"{interpreter}"\n{ENDS_WITH_ITS_LAUNCHER}{source}')
    return path + ".exe"


# On Windows the stand-in runs as the launcher's child, and a stopped launcher leaves its child running: the stand-in
# ends when the launcher does, as the one process of a real program would.
ENDS_WITH_ITS_LAUNCHER = """\
import ctypes as _ctypes, os as _os, threading as _threading
def _end_with_the_launcher(launcher=_os.getppid()):
    kernel = _ctypes.windll.kernel32
    handle = kernel.OpenProcess(0x00100000, False, launcher)  # SYNCHRONIZE
    if handle:
        kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
        _os._exit(1)
_threading.Thread(target=_end_with_the_launcher, daemon=True).start()
"""


def alive(pid):
    """Whether `pid` runs: /proc on Linux (a zombie, ended but not yet reaped, counts as gone), psutil elsewhere."""
    if os.path.isdir("/proc"):
        try:
            with open(f"/proc/{pid}/stat") as file:
                return file.read().rsplit(")", 1)[1].split()[0] != "Z"
        except OSError:
            return False
    import psutil

    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def end(pids):
    """Ends each process of `pids` still running: on Windows, a stand-in left behind by a launcher that was stopped.
    Elsewhere the stand-in is the program itself, which the code under test stops."""
    if os.name != "nt":
        return
    import psutil

    for pid in pids:
        try:
            psutil.Process(pid).kill()
        except psutil.Error:
            pass
