"""Process code on each system (`processes`, `parent`, `memory`, the question judge's server, the licence check, the
bug report's user names): on Linux as it was, elsewhere a clean fallback. The other systems are stood in for."""

import ctypes
import os
import subprocess
import sys
import threading
import types
import unittest
from unittest import mock

from model_maker import check_licences, memory, parent, processes

LINUX = sys.platform.startswith("linux")


class Names(unittest.TestCase):
    def test_a_program_has_its_windows_name_there_only(self):
        self.assertEqual(processes.program_name("llama-server", windows=False), "llama-server")
        self.assertEqual(processes.program_name("llama-server", windows=True), "llama-server.exe")
        self.assertEqual(processes.program_name("llama-server.exe", windows=True), "llama-server.exe")

    def test_the_libraries_beside_a_program(self):
        given = {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": ""}
        # Linux: as the question judge set it before, the folder alone.
        self.assertEqual(processes.library_environment("/tools", given, "linux"),
                         {**given, "LD_LIBRARY_PATH": "/tools"})
        self.assertEqual(processes.library_environment("/tools", given, "darwin"),
                         {**given, "DYLD_LIBRARY_PATH": "/tools"})
        self.assertEqual(processes.library_environment("C:\\tools", given, "win32")["PATH"],
                         "C:\\tools" + os.pathsep + "/usr/bin")
        self.assertEqual(given, {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": ""}, "the given environment unchanged")

    @unittest.skipUnless(LINUX, "the judge's environment on Linux")
    def test_the_judges_server_environment_on_linux_is_as_before(self):
        from model_maker import question_judge

        where = question_judge.Found(True, "/repo/.tools/llama-b11600/llama-server", "m", "v")
        before = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": os.path.dirname(where.server)}
        self.assertEqual(question_judge.server_environment(where), before)


class Children(unittest.TestCase):
    @unittest.skipIf(processes.WINDOWS, "POSIX")
    def test_a_child_has_its_own_session_and_the_given_tie_on_posix(self):
        tie = object()
        self.assertEqual(processes.start_options(tie), {"start_new_session": True, "preexec_fn": tie})

    def test_windows_job_limits_have_windows_layout(self):
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            self.skipTest("the layout is checked for 64-bit systems")
        limits = processes.kill_on_close_limits()
        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION is 144 bytes on 64-bit Windows, its LimitFlags at byte 16.
        self.assertEqual(ctypes.sizeof(limits), 144)
        self.assertEqual(processes._BasicLimits.LimitFlags.offset, 16)
        self.assertEqual(limits.BasicLimitInformation.LimitFlags, 0x2000)

    @unittest.skipIf(processes.WINDOWS, "POSIX")
    def test_tying_does_nothing_where_the_start_ties(self):
        self.assertFalse(processes.tie_to_this_process(object()))

    @unittest.skipIf(processes.WINDOWS, "POSIX")
    def test_a_child_that_ignores_sigterm_is_stopped_with_sigkill(self):
        child = subprocess.Popen([sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, "
                                  "signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"],
                                 stdout=subprocess.PIPE, **processes.start_options())
        self.assertEqual(child.stdout.readline().strip(), b"ready")
        processes.stop(child, wait_seconds=1)
        self.assertEqual(child.wait(timeout=10), -9)
        child.stdout.close()

    def test_on_windows_a_child_is_terminated_then_killed(self):
        said = []

        class Child:
            def terminate(self):
                said.append("terminate")

            def kill(self):
                said.append("kill")

            def wait(self, timeout):
                said.append("wait")
                if said.count("wait") == 1:
                    raise subprocess.TimeoutExpired("server", timeout)

        with mock.patch.object(processes, "WINDOWS", True):
            processes.stop(Child(), wait_seconds=1)
        self.assertEqual(said, ["terminate", "wait", "kill", "wait"])

    @unittest.skipUnless(LINUX, "/proc")
    def test_a_processs_memory_on_linux_is_the_kernels(self):
        found = processes.memory_of(os.getpid())
        self.assertEqual(set(found), {"VmHWM", "VmRSS", "RssAnon", "RssFile"})
        self.assertGreater(found["VmRSS"], 0)

    def test_a_processs_memory_elsewhere_is_psutils(self):
        info = types.SimpleNamespace(rss=300 * 2 ** 20, peak_wset=500 * 2 ** 20)
        stand_in = types.SimpleNamespace(Process=lambda pid: types.SimpleNamespace(memory_info=lambda: info),
                                         Error=Exception)
        with mock.patch.object(processes, "LINUX", False), mock.patch.dict(sys.modules, {"psutil": stand_in}), \
                mock.patch("builtins.open", side_effect=FileNotFoundError("/proc")):
            self.assertEqual(processes.memory_of(4242), {"VmRSS": 300, "VmHWM": 500})
        with mock.patch.object(processes, "LINUX", False), mock.patch.dict(sys.modules, {"psutil": None}), \
                mock.patch("builtins.open", side_effect=FileNotFoundError("/proc")):
            self.assertIsNone(processes.memory_of(4242))


class Starter(unittest.TestCase):
    def test_the_starter_is_known_alive_by_psutil(self):
        running = {"yes": True}
        stand_in = types.SimpleNamespace(
            Process=lambda pid: types.SimpleNamespace(is_running=lambda: running["yes"]), Error=LookupError)
        alive = processes.parent_alive_check("77", psutil_module=stand_in)
        self.assertTrue(alive())
        running["yes"] = False
        self.assertFalse(alive())

    def test_a_starter_gone_already_is_not_alive(self):
        def gone(pid):
            raise LookupError(pid)

        stand_in = types.SimpleNamespace(Process=gone, Error=LookupError)
        self.assertFalse(processes.parent_alive_check(77, psutil_module=stand_in)())

    @unittest.skipIf(processes.WINDOWS, "POSIX")
    def test_without_psutil_the_starter_is_this_processs_parent(self):
        parents = [77]
        with mock.patch.dict(sys.modules, {"psutil": None}):
            alive = processes.parent_alive_check(77, getppid=lambda: parents[0])
        self.assertTrue(alive())
        parents[0] = 1  # given to init when the starter ended
        self.assertFalse(alive())

    def test_without_psutil_windows_cannot_tell(self):
        with mock.patch.dict(sys.modules, {"psutil": None}), mock.patch.object(processes, "WINDOWS", True):
            self.assertIsNone(processes.parent_alive_check(77))

    def test_the_watch_ends_the_process_when_the_starter_ends(self):
        running, ended = [True], threading.Event()
        stop = processes.watch_parent(77, interval_seconds=0.01, ended=ended.set, check=lambda: running[0])
        self.addCleanup(stop)
        self.assertFalse(ended.wait(0.1))
        running[0] = False
        self.assertTrue(ended.wait(5))

    def test_elsewhere_than_linux_the_pipeline_watches_its_starter(self):
        watched = []
        with mock.patch.object(sys, "platform", "darwin"), \
                mock.patch.object(processes, "parent_alive_check", lambda worker: lambda: True), \
                mock.patch.object(processes, "watch_parent", lambda worker, check: watched.append(worker) or id):
            self.assertTrue(parent.die_with_parent("4242"))
        self.assertEqual(watched, ["4242"])

    def test_elsewhere_a_starter_gone_already_ends_the_pipeline(self):
        with mock.patch.object(sys, "platform", "win32"), \
                mock.patch.object(processes, "parent_alive_check", lambda worker: lambda: False):
            with self.assertRaises(SystemExit):
                parent.die_with_parent("4242")

    def test_elsewhere_without_a_way_to_watch_it_says_so(self):
        with mock.patch.object(sys, "platform", "win32"), \
                mock.patch.object(processes, "parent_alive_check", lambda worker: None), \
                mock.patch("sys.stderr") as said:
            self.assertFalse(parent.die_with_parent("4242"))
        self.assertIn("cannot be watched", "".join(call.args[0] for call in said.write.call_args_list))


class Memory(unittest.TestCase):
    """memory.py: the kernel's numbers on Linux, psutil's elsewhere, None without either."""

    def stand_in(self):
        info = types.SimpleNamespace(rss=2048 * 2 ** 20, peak_wset=4096 * 2 ** 20)
        return types.SimpleNamespace(Process=lambda: types.SimpleNamespace(memory_info=lambda: info),
                                     virtual_memory=lambda: types.SimpleNamespace(available=8192 * 2 ** 20),
                                     Error=LookupError)

    @unittest.skipUnless(LINUX, "/proc")
    def test_on_linux_the_numbers_are_the_kernels(self):
        # psutil is not asked where /proc answers (two reads of /proc are not compared: memory moves between them).
        asked = mock.Mock(side_effect=self.stand_in)
        with mock.patch.object(memory, "_psutil", asked):
            found = (memory.in_use_mb(), memory.peak_mb(), memory.free_mb())
        self.assertNotIn(None, found)
        self.assertNotEqual(found, (2048, 4096, 8192))
        asked.assert_not_called()

    def test_without_proc_psutil_says(self):
        with mock.patch.object(memory, "_psutil", self.stand_in), \
                mock.patch("builtins.open", side_effect=FileNotFoundError("/proc")):
            self.assertEqual((memory.in_use_mb(), memory.peak_mb(), memory.free_mb()), (2048, 4096, 8192))

    def test_without_proc_or_psutil_nothing_is_known_and_the_guard_lets_a_stage_start(self):
        with mock.patch.object(memory, "_psutil", lambda: None), \
                mock.patch("builtins.open", side_effect=FileNotFoundError("/proc")), \
                mock.patch.dict(os.environ, {memory.CHECK_VARIABLE: "1"}):
            self.assertEqual((memory.in_use_mb(), memory.peak_mb(), memory.free_mb()), (None, None, None))
            self.assertEqual(memory.room_for("shape", 13700), (True, "the memory free is not known here"))


class Licences(unittest.TestCase):
    def test_libraries_are_found_by_each_systems_names(self):
        self.assertTrue(check_licences.is_library("libgomp-a34b3233.so.1", "linux"))
        self.assertFalse(check_licences.is_library("_C.cpython-311-x86_64-linux-gnu.so", "linux"))
        self.assertFalse(check_licences.is_library("libomp.dylib", "linux"), "Linux judged as before")
        self.assertTrue(check_licences.is_library("libomp.dylib", "darwin"))
        self.assertTrue(check_licences.is_library("cudart64_12.dll", "win32"))
        self.assertFalse(check_licences.is_library("_C.cp311-win_amd64.pyd", "win32"))

    def test_the_check_fails_where_it_does_not_know_the_libraries(self):
        self.assertTrue(check_licences.system_known("linux"))
        self.assertFalse(check_licences.system_known("win32"))
        self.assertFalse(check_licences.system_known("darwin"))
        with mock.patch.object(check_licences.importlib.metadata, "distributions", lambda: []), \
                mock.patch.object(check_licences, "judge_libraries", lambda folder: ([], [])), \
                mock.patch.object(check_licences, "system_known", lambda: False), mock.patch("builtins.print"):
            self.assertEqual(check_licences.main([]), 1)


class UserNames(unittest.TestCase):
    def test_without_pwd_the_users_name_is_still_found(self):
        from model_maker import bug_report

        with mock.patch.dict(sys.modules, {"pwd": None}), \
                mock.patch.dict(os.environ, {"USERNAME": "someone-on-windows"}):
            self.assertIn("someone-on-windows", bug_report.this_users())


class SimplifyTool(unittest.TestCase):
    def test_the_built_tool_has_its_windows_name_there(self):
        from model_maker import simplify

        made = []

        def exists(path):
            made.append(path)
            return True

        with mock.patch.dict(os.environ, {"MODEL_MAKER_SIMPLIFY": ""}), mock.patch.object(simplify.os.path, "exists",
                                                                                         exists):
            with mock.patch.object(simplify.os, "name", "nt"):
                self.assertTrue(simplify.tool().endswith("model-maker-simplify.exe"))
            with mock.patch.object(simplify.os, "name", "posix"):
                self.assertTrue(simplify.tool().endswith(os.path.join("release", "model-maker-simplify")))


if __name__ == "__main__":
    unittest.main()
