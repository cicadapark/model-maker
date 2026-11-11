"""tools/trial/summary.py never counts a job whose chosen picture was not read for painted lettering as a pass.

    python3 -I -m unittest discover -s tools -p 'test_*.py'
"""

import contextlib
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("summary", ROOT / "tools" / "trial" / "summary.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def ended(passed=True, lettering_judged=None, failed=()):
    line = {"event": "job", "job": "a", "batch": "b", "passed": passed, "failed": list(failed), "weapon": False,
            "seconds": {"pictures": 60.0}, "first_tries": {"picture_rounds": 1, "shape_first_passed": True},
            "version": "v", "machine": {"card": "a card"}, "made_at": "2026-11-11T08:00:00Z"}
    if lettering_judged is not None:
        line["lettering_judged"] = lettering_judged
    return line


class NotRead(unittest.TestCase):
    def test_a_job_not_read_is_not_a_pass_and_is_shown_apart(self):
        row = summary.batch_row([ended(), ended(lettering_judged=True), ended(lettering_judged=False)])
        self.assertEqual(row["made"], 3)
        self.assertEqual(row["passed"], 2)
        self.assertEqual(row["not_read"], 1)

    def test_a_job_made_before_the_check_counts_as_it_did(self):
        self.assertFalse(summary.unread(ended()))
        self.assertEqual(summary.batch_row([ended()])["passed"], 1)

    def test_the_checks_listing_marks_a_job_not_read(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, lettering in (("read", {"judged": True}), ("unread", {"judged": False}), ("before", None)):
                (pathlib.Path(folder) / name).mkdir()
                record = {"status": "passes every check", **({"lettering": lettering} if lettering else {})}
                (pathlib.Path(folder) / name / "job.json").write_text(json.dumps(record))
            shown = io.StringIO()
            with contextlib.redirect_stdout(shown):
                summary.checks(folder)
        self.assertIn("NOT READ for painted lettering", shown.getvalue())
        self.assertIn("2 pass every check; 1 more", shown.getvalue())

    def test_a_check_shown_not_counted_is_no_fault(self):
        with tempfile.TemporaryDirectory() as folder:
            (pathlib.Path(folder) / "job").mkdir()
            tries = {"7": {"passed": True, "checks": [{"name": "no painted lettering", "passed": False,
                                                       "counted": False}]}}
            (pathlib.Path(folder) / "job" / "job.json").write_text(json.dumps({"picture": {"tries": tries}}))
            shown = io.StringIO()
            with contextlib.redirect_stdout(shown):
                summary.pictures([folder])
        self.assertIn("faults {}", shown.getvalue())


if __name__ == "__main__":
    unittest.main()
