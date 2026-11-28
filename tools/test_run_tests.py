"""tools/ci/run_tests.py: each module's verdict printed as it ends (setup's self-test shows its progress), with
--log the whole report in the file, the modules still running every --progress seconds, and a module over
--module-limit stopped, with the processes it started, and named as FAILED."""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import textwrap
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("run_tests", os.path.join(HERE, "ci", "run_tests.py"))
run_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_tests)


class Report(unittest.TestCase):
    def setUp(self):
        self.saved = run_tests.modules, run_tests.run
        run_tests.modules = lambda: ["test_a", "test_b"]
        results = {"test_a": ("test_a", True, 3, ["test_x (tests.test_a.T) ... skipped 'no card'"], 1.0, "a's log",
                              False),
                   "test_b": ("test_b", False, 2, [], 2.0, "Traceback: b's own log", False)}
        run_tests.run = lambda module, limit_seconds: results[module]

    def tearDown(self):
        run_tests.modules, run_tests.run = self.saved

    def test_verdicts_as_they_end_and_the_whole_report_in_the_log(self):
        with tempfile.TemporaryDirectory() as folder:
            log = os.path.join(folder, "self-test.log")
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                code = run_tests.main(["--workers", "2", "--log", log])
            with open(log, encoding="utf-8") as file:
                kept = file.read()
        self.assertEqual(code, 1)
        shown = printed.getvalue()
        self.assertIn("2 test modules, 2 at once, each stopped after 30 min", shown)
        self.assertIn("tests.test_b: FAILED, 2 tests, 0 skipped, 2 s", shown)
        self.assertIn(" of 2)", shown)
        self.assertIn("5 tests in 2 modules, 1 skipped, 1 modules failed", shown)
        self.assertNotIn("b's own log", shown)
        self.assertNotIn("skipped 'no card'", shown)
        for line in ("tests.test_a: ok, 3 tests, 1 skipped", "skipped 'no card'", "b's own log", "1 modules failed"):
            self.assertIn(line, kept)

    def test_without_a_log_everything_is_printed(self):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            run_tests.main(["--workers", "1"])
        self.assertIn("b's own log", printed.getvalue())
        self.assertIn("skipped 'no card'", printed.getvalue())

    def test_a_module_over_the_limit_is_named_and_fails_the_run(self):
        run_tests.run = lambda module, limit_seconds: (
            (module, False, 1, [], 120.0, "test_first ... ok\n===== stopped after 120 s", True) if module == "test_b"
            else (module, True, 3, [], 1.0, "a's log", False))
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = run_tests.main(["--workers", "2", "--module-limit", "2"])
        shown = printed.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("each stopped after 2 min", shown)
        self.assertIn("tests.test_b: FAILED (stopped: over the limit of 2 min, hung or too slow), 1 tests", shown)
        self.assertIn("tests.test_a: ok, 3 tests", shown)
        self.assertIn("over the time limit (stopped): tests.test_b", shown)
        self.assertIn("test_first ... ok", shown)

    def test_the_modules_still_running_are_said(self):
        def slow(module, limit_seconds):
            time.sleep(0.5 if module == "test_b" else 0.0)
            return module, True, 1, [], 0.5, "", False

        run_tests.run = slow
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = run_tests.main(["--workers", "2", "--progress", "0.1", "--module-limit", "0"])
        shown = printed.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("no time limit", shown)
        self.assertRegex(shown, r"still running after \d+ s: tests\.test_b \(\d+ s\) \(1 of 2 ended\)")


HANGING_MODULE = """\
import subprocess, sys, unittest


class Hangs(unittest.TestCase):
    def test_1_first(self):
        pass

    def test_2_hangs_on_a_child(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        with open({pid_file!r}, "w") as file:
            file.write(str(child.pid))
        child.wait()
"""


def alive(pid):
    """Linux only: a stopped process left unreaped (a zombie) counts as ended."""
    try:
        with open(f"/proc/{pid}/stat") as file:
            return file.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


class TimeLimit(unittest.TestCase):
    """A real module that hangs (three processes at most: this one, the module, the child it waits on)."""

    def test_a_hung_module_is_stopped_with_its_child_and_keeps_its_output_so_far(self):
        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "tests"))
            open(os.path.join(folder, "tests", "__init__.py"), "w").close()
            pid_file = os.path.join(folder, "child.pid")
            with open(os.path.join(folder, "tests", "test_hangs.py"), "w") as file:
                file.write(HANGING_MODULE.format(pid_file=pid_file))
            started = time.time()
            module, passed, count, skipped, seconds, log, over = run_tests.run("test_hangs", folder=folder,
                                                                               limit_seconds=3)
            self.assertLess(time.time() - started, 60)
            with open(pid_file) as file:
                child = int(file.read())
        self.assertEqual((module, passed, count, over), ("test_hangs", False, 0, True))
        self.assertGreaterEqual(seconds, 3)
        self.assertIn("test_1_first (tests.test_hangs.Hangs", log)
        self.assertIn("over the limit of 3 s per module", log)
        if sys.platform.startswith("linux"):
            self.assertFalse(alive(child), "the child the hung test waited on was stopped too")

    def test_a_module_within_the_limit_is_its_own_verdict(self):
        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "tests"))
            open(os.path.join(folder, "tests", "__init__.py"), "w").close()
            with open(os.path.join(folder, "tests", "test_quick.py"), "w") as file:
                file.write("import unittest\n\n\nclass Q(unittest.TestCase):\n    def test_q(self):\n        pass\n\n"
                           "    @unittest.skip('a reason')\n    def test_s(self):\n        pass\n")
            module, passed, count, skipped, seconds, log, over = run_tests.run("test_quick", folder=folder,
                                                                               limit_seconds=60)
        self.assertEqual((passed, count, over, len(skipped)), (True, 2, False, 1), log)
        self.assertNotIn("stopped after", log)


if __name__ == "__main__":
    unittest.main()
