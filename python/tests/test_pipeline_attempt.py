"""A later attempt at an object draws every seed after the earlier attempts' (a refused object made again is new)."""
import unittest

from model_maker import pipeline


class Attempts(unittest.TestCase):
    def test_the_first_attempt_is_as_before_and_a_later_one_starts_after_it(self):
        first, second = pipeline.Settings(verify=False), pipeline.Settings(verify=False, attempt=1)
        self.assertEqual(first.attempt, 0)
        used = range(0, first.picture_rounds * first.tries)
        offset = second.attempt * second.picture_rounds * second.tries
        self.assertGreaterEqual(offset, len(used))

    def test_a_later_attempt_is_recorded_and_is_its_own_job(self):
        import tempfile
        from model_maker import ledger
        first, second = pipeline.Settings(verify=False), pipeline.Settings(verify=False, attempt=1)
        self.assertNotIn("attempt", ledger.settings_of(first))  # the records made before keep their settings
        self.assertEqual(ledger.settings_of(second)["attempt"], 1)
        with tempfile.TemporaryDirectory() as out:
            folder = pipeline.job_folder(out, "an old iron sword", attempt=1)
            self.assertNotEqual(folder, pipeline.job_folder(out, "an old iron sword"))
            job = pipeline.Job(folder, "an old iron sword", settings=second)
            self.assertEqual(job.record["settings"]["attempt"], 1)

    def test_the_command_shows_attempt_and_refuses_a_negative_one(self):
        import argparse
        import contextlib
        import io
        from model_maker import __main__ as command
        self.assertEqual(command.attempt_number("2"), 2)
        for text in ("-1", "one"):
            with self.assertRaises(argparse.ArgumentTypeError):
                command.attempt_number(text)
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown), self.assertRaises(SystemExit):
            command.main(["make", "--help"])
        self.assertIn("--attempt", shown.getvalue())



class PairJobs(unittest.TestCase):
    def test_a_pair_s_job_keeps_the_prompt_asked_and_its_subject_the_one(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            job = pipeline.Job(tmp + "/job", "a pair of boots")
            self.assertEqual(job["prompt"], "a pair of boots")
            self.assertEqual((job.subject.prompt, job.subject.pair), ("a boot", "mirror"))
            again = pipeline.Job(tmp + "/job", "a pair of boots")  # the same order: no mismatch
            self.assertEqual(again.subject.pair, "mirror")

if __name__ == "__main__":
    unittest.main()
