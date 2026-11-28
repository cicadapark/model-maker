"""Process code on each system (`processes`, `parent`, `memory`, the question judge's server, the licence check, the
bug report's user names): on Linux as it was, elsewhere a clean fallback. The other systems are stood in for."""

import ctypes
import os
import subprocess
import sys
import tempfile
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
    def test_the_copyleft_dbm_module_is_refused_in_setups_interpreter_only(self):
        with tempfile.TemporaryDirectory() as repository:
            base = os.path.join(repository, ".tools", "python", "cpython-3.11.14-linux-x86_64-gnu")
            modules = check_licences.interpreter_modules_folder(base, "linux")
            os.makedirs(modules)
            for name in ("_dbm.cpython-311-x86_64-linux-gnu.so", "_decimal.cpython-311-x86_64-linux-gnu.so"):
                open(os.path.join(modules, name), "w").close()
            problems, notes = check_licences.interpreter_findings(base, "linux", repository)
            self.assertEqual(len(problems), 1)
            self.assertIn("_dbm.cpython-311", problems[0])
            self.assertEqual(notes, [])
            elsewhere = check_licences.interpreter_findings(base, "linux", os.path.join(repository, "other"))
            self.assertEqual((len(elsewhere[0]), len(elsewhere[1])), (0, 1), "another interpreter: reported only")
            os.remove(os.path.join(modules, "_dbm.cpython-311-x86_64-linux-gnu.so"))
            self.assertEqual(check_licences.interpreter_findings(base, "linux", repository), ([], []))
        self.assertTrue(check_licences.interpreter_modules_folder("C:/p", "win32").endswith("DLLs"))

    def test_the_judges_windows_and_macos_libraries_are_known(self):
        with tempfile.TemporaryDirectory() as folder:
            names = ["ggml-base.dll", "ggml-cpu-zen4.dll", "llama.dll", "llama-server-impl.dll", "mtmd.dll",
                     "libomp.dll", "llama-server.exe", "LICENSE-LLVM-OpenMP"]
            for name in names:
                open(os.path.join(folder, name), "w").close()
            with open(os.path.join(folder, "LICENSE"), "w") as file:
                file.write("MIT License\n")
            rows, problems = check_licences.judge_libraries(folder, "win32")
            self.assertEqual(problems, [])
            self.assertEqual(len(rows), 6)
            self.assertIn("LLVM-exception", dict((r[0], r[2]) for r in rows)["libomp.dll"])
            open(os.path.join(folder, "msvcp140.dll"), "w").close()
            self.assertEqual(len(check_licences.judge_libraries(folder, "win32")[1]), 1, "an unknown library refuses")

    def test_nothing_imports_dbm(self):
        """The proof that removing `_dbm` changes nothing: no module of model-maker or of the installed packages
        imports dbm, shelve (which uses it) or _dbm/_gdbm."""
        import re as regex
        pattern = regex.compile(r"^\s*(import|from)\s+(dbm|shelve|_dbm|_gdbm)\b", regex.MULTILINE)
        roots = [os.path.join(check_licences.REPOSITORY, "python"), os.path.join(check_licences.REPOSITORY, "setup")]
        roots += [p for p in sys.path if p.endswith("site-packages")]
        users = []
        for root in roots:
            for folder, _, names in os.walk(root):
                for name in names:
                    if name.endswith(".py"):
                        path = os.path.join(folder, name)
                        with open(path, encoding="utf-8", errors="replace") as file:
                            if pattern.search(file.read()):
                                users.append(path)
        self.assertEqual(users, [])

    def test_libraries_are_found_by_each_systems_names(self):
        self.assertTrue(check_licences.is_library("libgomp-a34b3233.so.1", "linux"))
        self.assertFalse(check_licences.is_library("_C.cpython-311-x86_64-linux-gnu.so", "linux"))
        self.assertFalse(check_licences.is_library("libomp.dylib", "linux"), "Linux judged as before")
        self.assertTrue(check_licences.is_library("libomp.dylib", "darwin"))
        self.assertTrue(check_licences.is_library("cudart64_12.dll", "win32"))
        self.assertFalse(check_licences.is_library("_C.cp311-win_amd64.pyd", "win32"))

    def test_the_check_fails_where_it_does_not_know_the_libraries(self):
        # Linux, Windows and macOS: every library their locks' wheels bundle is on the allow-list (tools/ci/locks.py
        # licences reads them all in CI); any other system fails.
        for system in ("linux", "win32", "darwin"):
            self.assertTrue(check_licences.system_known(system), system)
        self.assertFalse(check_licences.system_known("freebsd14"))
        self.assertFalse(check_licences.system_known("cygwin"))
        with mock.patch.object(check_licences.importlib.metadata, "distributions", lambda: []), \
                mock.patch.object(check_licences, "judge_libraries", lambda folder: ([], [])), \
                mock.patch.object(check_licences, "system_known", lambda: False), mock.patch("builtins.print"):
            self.assertEqual(check_licences.main([]), 1)


    def test_the_windows_and_macos_libraries_are_judged_by_their_own_names(self):
        for path in ("torch/lib/torch_cuda.dll", "torch/lib/cudnn_ops64_9.dll", "numpy.libs/msvcp140-a4c2229b.dll",
                     "torch/lib/libiomp5md.dll", "c2pa/libs/c2pa_c.dll", "torch/lib/libomp.dylib",
                     "scipy/.dylibs/libgcc_s.1.1.dylib", "numpy/.dylibs/libquadmath.0.dylib"):
            name = path.rsplit("/", 1)[1]
            self.assertEqual(check_licences.judged_library(path, name)[0], "allowed", name)
        # Windows' libjpeg.dll (torchvision) is the IJG's libjpeg 9f: its own row, not libjpeg-turbo's by its start.
        self.assertIn("Independent JPEG Group", check_licences.judged_library("torchvision/libjpeg.dll",
                                                                              "libjpeg.dll")[1])
        self.assertIn("libjpeg-turbo", check_licences.judged_library("PIL/libjpeg.so.62", "libjpeg.so.62")[1])
        # What PyTorch's ROCm build bundles under the LGPL is not (tools/ci/locks.py, the ROCm lock is refused).
        for name in ("libnuma.so", "libelf.so", "libcholmod.so"):
            self.assertEqual(check_licences.judged_library(f"torch/lib/{name}", name)[0], "unknown", name)


    def test_a_record_written_with_windows_paths_is_read_by_each_files_own_name(self):
        # torchvision 0.25's Windows wheel writes its RECORD with backslashes; importlib's paths are posix paths, so
        # `.name` keeps the folder in: every library there was "not on the allow-list" on an installed Windows.
        from importlib.metadata import PackagePath

        record = ["torchvision\\cudart64_12.dll", "torchvision\\nvjpeg64_12.dll", "torchvision\\jpeg8.dll",
                  "torchvision\\libjpeg.dll", "torchvision\\libpng16.dll", "torchvision\\libsharpyuv.dll",
                  "torchvision\\libwebp.dll", "torchvision\\python311.dll", "torchvision\\zlib.dll",
                  "torchvision\\_C.pyd", "torch/lib/nvperf_host.dll"]
        from email.message import Message

        metadata = Message()
        metadata["Name"], metadata["License"] = "torchvision", "BSD"
        distribution = types.SimpleNamespace(files=[PackagePath(path) for path in record], metadata=metadata)
        with mock.patch.object(check_licences.is_library, "__defaults__", ("win32",)):
            found = list(check_licences.bundled_libraries(distribution))
            problems = list(check_licences.problems_of(distribution))
        self.assertEqual([name for _, name in found], [path.replace("\\", "/").rsplit("/", 1)[1]
                                                       for path in record if path.endswith(".dll")])
        self.assertEqual(problems, [])


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
                self.assertEqual(simplify.tool(), os.path.join(simplify.REPOSITORY, "model-maker-simplify"))

    def test_the_shipped_simplify_tool_comes_before_a_build(self):
        from model_maker import simplify

        name = "model-maker-simplify"
        with tempfile.TemporaryDirectory() as repository:
            def made(*parts):
                path = os.path.join(repository, *parts)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path, "w").close()
                return path

            def found():
                return next((p for p in simplify.candidates(repository, name) if os.path.exists(p)), None)

            self.assertIsNone(found())
            built = made("target", "release", name)
            self.assertEqual(found(), built)
            linux_archive = made("bin", name)
            self.assertEqual(found(), linux_archive, "the Linux archive's bin/ before a build")
            top = made(name)
            self.assertEqual(found(), top, "the Windows and macOS archives' top folder first")
        with mock.patch.dict(os.environ, {"MODEL_MAKER_SIMPLIFY": "/the/launchers/choice"}):
            self.assertEqual(simplify.tool(), "/the/launchers/choice")


if __name__ == "__main__":
    unittest.main()
