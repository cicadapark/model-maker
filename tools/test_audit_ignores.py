"""CI's vulnerability audit ignores only what tools/ci/pip-audit-ignore.toml lists, each entry complete and not past
its review date."""

import datetime
import os
import re
import tomllib
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
IGNORES = os.path.join(HERE, "ci", "pip-audit-ignore.toml")
WORKFLOW = os.path.join(HERE, "..", ".github", "workflows", "ci.yml")


def entries():
    with open(IGNORES, "rb") as file:
        return tomllib.load(file).get("ignore", [])


class AuditIgnores(unittest.TestCase):
    def test_each_entry_says_what_why_and_when_to_look_again(self):
        for entry in entries():
            with self.subTest(entry.get("id")):
                self.assertRegex(entry["id"], r"^(PYSEC|GHSA|CVE)-")
                for key in ("package", "version", "why", "review"):
                    self.assertTrue(str(entry.get(key, "")).strip(), f"{entry['id']} lacks {key}")
                self.assertGreater(len(entry["why"]), 80, "say why it does not apply or cannot be fixed yet")

    def test_no_review_date_has_passed(self):
        today = datetime.date.today()
        for entry in entries():
            with self.subTest(entry["id"]):
                review = datetime.date.fromisoformat(entry["review"])
                self.assertGreaterEqual(review, today, f"{entry['id']}: its review date passed; read it again")

    def test_the_workflow_ignores_nothing_else(self):
        with open(WORKFLOW) as file:
            workflow = file.read()
        self.assertIsNone(re.search(r"--ignore-vuln[ =]+[\"']?(PYSEC|GHSA|CVE)-", workflow),
                          "the ignored ids come from tools/ci/pip-audit-ignore.toml alone, never written in the workflow")
        self.assertIn("tools/ci/pip-audit-ignore.toml", workflow)


if __name__ == "__main__":
    unittest.main()
