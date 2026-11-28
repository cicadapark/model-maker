#!/usr/bin/env python3
"""Windows only: when the `model-maker` launcher ends, every program it started ends too, however it ends.

    python -I tools/ci/launcher_ends.py LAUNCHER.exe PYTHON.exe [--work FOLDER]

LAUNCHER is a built `model-maker.exe` (with any library it needs beside it); PYTHON is the Python the stand-in side
runs with, best the environment's own `Scripts\\python.exe` (a starter that runs the real interpreter as its child, as
in a real install). The script makes a stand-in repository whose `model_maker` starts a grandchild (detached: no
console, so no console event can end it; only the launcher's job object can) and writes the process ids, then:

  control   the interpreter started without the launcher and killed: its detached grandchild survives (so the
            checks below do not pass by chance)
  hard kill the launcher killed outright (taskkill /F): nothing of the chain is left
  Ctrl+C    Ctrl+C in the launcher's console: Python ends its own way (exit 7) and the launcher returns that code;
            nothing is left
  closed    the launcher's console window closed: nothing is left (not run where the close cannot reach a console)

Exit 0 when all hold. The standard library only (ctypes for the console calls).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time

CREATE_NEW_CONSOLE = 0x10
DETACHED_PROCESS = 0x8
WAIT_S = 30
GONE_S = 15

STAND_IN_MAIN = textwrap.dedent('''
    import os, subprocess, sys, time
    folder = os.environ["ENDS_FOLDER"]
    grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], creationflags=0x8)
    with open(os.path.join(folder, "pids.json"), "w") as file:
        file.write('{"starter": %d, "interpreter": %d, "grandchild": %d}' % (os.getppid(), os.getpid(), grandchild.pid))
    try:
        time.sleep(600)
    except KeyboardInterrupt:
        grandchild.kill()
        open(os.path.join(folder, "interrupted"), "w").close()
        sys.exit(7)
''')

CONDUCTOR = textwrap.dedent('''
    # Runs in a console of its own: starts the launcher in it, then either sends Ctrl+C to the console or waits to
    # be closed. It ignores Ctrl+C itself.
    import ctypes, json, os, signal, subprocess, sys, time
    launcher, folder, how = sys.argv[1], sys.argv[2], sys.argv[3]
    # A program started over ssh (or by a service) inherits "Ctrl+C ignored", and so would everything it starts; a
    # person's console has it on. Turned on here, the launcher and Python inherit it on, as there.
    ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    child = subprocess.Popen([launcher, "make", "a stand-in"])
    pids = os.path.join(folder, "pids.json")
    for _ in range(300):
        if os.path.exists(pids):
            break
        time.sleep(0.1)
    with open(os.path.join(folder, "conductor.json"), "w") as file:
        json.dump({"launcher": child.pid, "window": ctypes.windll.kernel32.GetConsoleWindow()}, file)
    if how == "ctrl-c":
        time.sleep(1)
        ctypes.windll.kernel32.GenerateConsoleCtrlEvent(0, 0)
        try:
            code = child.wait(30)
        except subprocess.TimeoutExpired:
            code = "still running"
        with open(os.path.join(folder, "result.json"), "w") as file:
            json.dump({"launcher_exit": code}, file)
    else:
        time.sleep(600)
''')


def alive(pid):
    listed = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True, text=True)
    return f'"{pid}"' in listed.stdout


def waited_for(path, seconds=WAIT_S):
    end = time.time() + seconds
    while time.time() < end:
        if os.path.exists(path):
            time.sleep(0.2)
            with open(path) as file:
                return json.load(file)
        time.sleep(0.1)
    raise SystemExit(f"{path}: not written in {seconds} s")


def left_after(pids, seconds=GONE_S):
    end = time.time() + seconds
    while time.time() < end:
        left = {name: pid for name, pid in pids.items() if alive(pid)}
        if not left:
            return {}
        time.sleep(0.5)
    return left


def kill_all(pids):
    for pid in pids.values():
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)


def stand_in_repository(work, launcher):
    repository = os.path.join(work, "repository")
    package = os.path.join(repository, "python", "model_maker")
    os.makedirs(package)
    open(os.path.join(package, "__init__.py"), "w").close()
    with open(os.path.join(package, "__main__.py"), "w") as file:
        file.write(STAND_IN_MAIN)
    for name in os.listdir(os.path.dirname(os.path.abspath(launcher))):
        if name.lower().endswith(".dll"):
            shutil.copy(os.path.join(os.path.dirname(os.path.abspath(launcher)), name), repository)
    shutil.copy(launcher, os.path.join(repository, "model-maker.exe"))
    return repository


def fresh(folder):
    for name in ("pids.json", "conductor.json", "result.json", "interrupted"):
        try:
            os.remove(os.path.join(folder, name))
        except OSError:
            pass


def control(python, repository, folder, env):
    fresh(folder)
    child = subprocess.Popen([python, "-m", "model_maker", "make", "x"], env=env, cwd=repository)
    pids = waited_for(os.path.join(folder, "pids.json"))
    subprocess.run(["taskkill", "/F", "/PID", str(child.pid)], capture_output=True)
    left = left_after({"grandchild": pids["grandchild"]}, 5)
    # Not pids["starter"]: here that is this script itself (the interpreter's parent).
    kill_all({"interpreter": pids["interpreter"], "grandchild": pids["grandchild"]})
    return bool(left), f"without the launcher, the detached grandchild {'survived' if left else 'ended'}"


def hard_kill(launcher, folder, env):
    fresh(folder)
    child = subprocess.Popen([launcher, "make", "x"], env=env)
    pids = waited_for(os.path.join(folder, "pids.json"))
    pids["launcher"] = child.pid
    subprocess.run(["taskkill", "/F", "/PID", str(child.pid)], capture_output=True)
    left = left_after(pids)
    kill_all(left)
    return not left, f"hard kill: left {left or 'nothing'}"


def by_console(how, python, launcher, folder, env, work):
    fresh(folder)
    script = os.path.join(work, "conductor.py")
    with open(script, "w") as file:
        file.write(CONDUCTOR)
    conductor = subprocess.Popen([python, "-I", script, launcher, folder, how], env=env,
                                 creationflags=CREATE_NEW_CONSOLE)
    pids = waited_for(os.path.join(folder, "pids.json"))
    started = waited_for(os.path.join(folder, "conductor.json"))
    pids["launcher"] = started["launcher"]
    if how == "ctrl-c":
        result = waited_for(os.path.join(folder, "result.json"), 40)
        left = left_after(pids)
        interrupted = os.path.exists(os.path.join(folder, "interrupted"))
        kill_all(left)
        conductor.kill()
        ok = not left and result["launcher_exit"] == 7 and interrupted
        return ok, (f"Ctrl+C: the launcher returned {result['launcher_exit']} (Python's own 7 expected), Python "
                    f"{'ended its own way' if interrupted else 'did not see it'}, left {left or 'nothing'}")
    window = started["window"]
    if window:
        import ctypes

        ctypes.windll.user32.PostMessageW(window, 0x0010, 0, 0)  # WM_CLOSE: the console's close button
        how_closed = "its window closed"
        try:
            conductor.wait(GONE_S)
        except subprocess.TimeoutExpired:
            # The console's first program ends on a close by its default handler: still there, the close never
            # reached the console (a machine without an interactive desktop, as GitHub's runners). Nothing about
            # the launcher can be read from that; the processes are ended and the check is not counted.
            kill_all(pids)
            conductor.kill()
            return None, ("console closed: NOT RUN here (the close message did not reach the console: no "
                          "interactive desktop); a PC with a desktop runs it")
    else:
        conductor.kill()
        how_closed = "no window (no desktop here): its first program ended instead"
    left = left_after(pids)
    kill_all(left)
    conductor.kill()
    return not left, f"console closed ({how_closed}): left {left or 'nothing'}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("launcher")
    parser.add_argument("python")
    parser.add_argument("--work")
    options = parser.parse_args(argv)
    if sys.platform != "win32":
        print("launcher_ends: Windows only")
        return 0
    work = options.work or tempfile.mkdtemp(prefix="launcher-ends-")
    os.makedirs(work, exist_ok=True)
    repository = stand_in_repository(work, options.launcher)
    launcher = os.path.join(repository, "model-maker.exe")
    env = dict(os.environ, ENDS_FOLDER=work, MODEL_MAKER_PYTHON=os.path.abspath(options.python),
               PYTHONPATH=os.path.join(repository, "python"))
    # The control runs the interpreter itself (this script's), not a starter, so only its own death is tested.
    results = [control(sys.executable, repository, work, env),
               hard_kill(launcher, work, env),
               by_console("ctrl-c", options.python, launcher, work, env, work),
               by_console("close", options.python, launcher, work, env, work)]
    for ok, said in results:
        print({True: "ok     ", False: "FAILED ", None: "not run "}[ok] + said)
    return 0 if all(ok is not False for ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
