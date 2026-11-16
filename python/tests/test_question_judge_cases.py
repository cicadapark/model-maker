"""The question judge on stored objects: the verdicts of question-judge-cases/cases.toml, asked of the real judge.

Skipped without it (./setup.sh --with-question-judge; the models where MODEL_MAKER_MODELS says, or models/). About a
minute and a half on 12 CPU threads.
"""

import os
import tomllib
import unittest

from PIL import Image

from model_maker import pipeline
from model_maker import question_judge as Q

CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "question-judge-cases")


def cases():
    with open(os.path.join(CASES, "cases.toml"), "rb") as file:
        return tomllib.load(file)["case"]


class StoredObjects(unittest.TestCase):
    def test_the_cases_are_whole(self):
        found = cases()
        self.assertEqual(sum(not c["passes"] for c in found), 3)
        self.assertEqual(sum(c["passes"] for c in found), 6)
        for case in found:
            with Image.open(os.path.join(CASES, case["file"])) as strip:
                self.assertEqual(strip.size, (2880, 360), case["file"])

    def test_the_judge_refuses_the_wrong_things_and_keeps_the_right_ones(self):
        models = os.environ.get("MODEL_MAKER_MODELS") or os.path.join(pipeline.MODEL_MAKER_ROOT, "models")
        where = Q.found(pipeline.MODEL_MAKER_ROOT, models)
        if not where.usable:
            self.skipTest(where.why)
        with Q.Judge(where) as judge:
            for case in cases():
                with self.subTest(case["file"]):
                    with Image.open(os.path.join(CASES, case["file"])) as strip:
                        check, record = Q.object_check(judge, strip.convert("RGB"), case["prompt"], required=True)
                    self.assertTrue(record["judged"], record)
                    self.assertEqual(check.passed, case["passes"], f"{case['file']}: {check.words}")


if __name__ == "__main__":
    unittest.main()
