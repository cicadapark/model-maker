"""`./setup.sh --check-only` on the computer's memory: "ok" from MEMORY_GOOD_GIB, "try it" from MEMORY_LEAST_GIB, and
below that a missing thing (it is counted, so the check exits 1)."""

import os
import re
import subprocess
import tempfile
import unittest

REPOSITORY = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
SETUP = os.path.join(REPOSITORY, "setup.sh")
GIB_IN_KB = 1024 * 1024


def figure(name):
    with open(SETUP) as file:
        return int(re.search(rf"^readonly {name}=(\d+)\b", file.read(), re.M).group(1))


def memory_row(total_kb):
    """The check's row about memory, and its whole output, on a computer with `total_kb` of memory."""
    with tempfile.TemporaryDirectory() as folder:
        meminfo = os.path.join(folder, "meminfo")
        with open(meminfo, "w") as file:
            file.write(f"MemTotal:       {total_kb} kB\nMemFree:         1024 kB\n")
        done = subprocess.run(["bash", SETUP, "--check-only"], capture_output=True, text=True, timeout=120,
                              env={**os.environ, "SETUP_MEMINFO": meminfo})
    rows = {line[:22].strip(): line[68:].strip() for line in done.stdout.splitlines() if line.startswith("Memory")}
    return rows, done.stdout


class Memory(unittest.TestCase):
    def test_the_figures_are_what_the_pipeline_runs_with(self):
        # docs/setup.md: 24 GB or more of the computer's memory (16 GB to try).
        self.assertEqual(figure("MEMORY_GOOD_GIB"), 24)
        self.assertEqual(figure("MEMORY_LEAST_GIB"), 16)

    def test_enough_memory_is_ok(self):
        rows, output = memory_row(figure("MEMORY_GOOD_GIB") * GIB_IN_KB)
        self.assertEqual(rows["Memory"], "ok", output)
        self.assertEqual(list(rows), ["Memory"], "one row about memory")

    def test_between_the_two_says_try_it(self):
        rows, _ = memory_row(figure("MEMORY_GOOD_GIB") * GIB_IN_KB - 1)
        self.assertTrue(rows["Memory"].startswith("try it"), rows)

    def test_too_little_is_missing(self):
        rows, _ = memory_row(figure("MEMORY_LEAST_GIB") * GIB_IN_KB - 1)
        self.assertIn("MISSING", rows["Memory"])
        self.assertIn(f"at least {figure('MEMORY_LEAST_GIB')} GB", rows["Memory"])


STUB_TESSERACT = """#!/bin/bash
case "$1" in
    --version) printf '%b' "$STUB_VERSION_OUT"; printf '%b' "$STUB_VERSION_ERR" >&2; exit "${STUB_VERSION_CODE:-0}" ;;
    --list-langs) printf 'List of available languages in "/stub/" (2):\\n%b' "$STUB_LANGUAGES" ;;
esac
"""


def tesseract_row(version_out="", version_err="", version_code=0, languages="eng\\nosd\\n"):
    """The check's row about tesseract (what it found, its result) with a stand-in tesseract first on the PATH, and the
    whole output."""
    with tempfile.TemporaryDirectory() as folder:
        stub = os.path.join(folder, "tesseract")
        with open(stub, "w") as file:
            file.write(STUB_TESSERACT)
        os.chmod(stub, 0o755)
        done = subprocess.run(["bash", SETUP, "--check-only"], capture_output=True, text=True, timeout=120,
                              env={**os.environ, "PATH": f"{folder}:{os.environ['PATH']}",
                                   "STUB_VERSION_OUT": version_out, "STUB_VERSION_ERR": version_err,
                                   "STUB_VERSION_CODE": str(version_code), "STUB_LANGUAGES": languages})
    rows = [(line[23:67].strip(), line[68:].strip()) for line in done.stdout.splitlines()
            if line[:22].strip() == "tesseract"]
    return rows, done.stdout + done.stderr


class Tesseract(unittest.TestCase):
    """The optional tesseract row never ends the check (set -e, pipefail), and says whether it can be used: version 5 or
    newer with its English data (the 0.2.0 review, defect 4 and missing test 9)."""

    def assert_row(self, rows, output, found, result):
        self.assertEqual(len(rows), 1, output)
        self.assertEqual(rows[0][0], found, output)
        self.assertTrue(rows[0][1].startswith(result), rows)
        # The check went on to its table's end and its verdict.
        self.assertIn("git", output)

    def test_a_version_that_prints_nothing_is_unknown(self):
        self.assert_row(*tesseract_row(), "tesseract (version unknown)", "optional, not used (no version found)")

    def test_a_version_that_fails_is_unknown(self):
        self.assert_row(*tesseract_row(version_code=1), "tesseract (version unknown)",
                        "optional, not used (no version found)")

    def test_version_4_is_too_old(self):
        self.assert_row(*tesseract_row(version_err="tesseract 4.1.1\\n leptonica-1.82.0\\n"), "tesseract 4.1.1",
                        "optional, not used (too old)")

    def test_version_5_without_english_data_is_not_used(self):
        self.assert_row(*tesseract_row(version_out="tesseract 5.3.0\\n", languages="osd\\n"),
                        "tesseract 5.3.0, no English data", "optional, not used")

    def test_version_5_with_english_data_is_ok(self):
        rows, output = tesseract_row(version_out="tesseract 5.5.0\\n leptonica-1.86.0\\n")
        self.assert_row(rows, output, "tesseract 5.5.0", "ok")
        self.assertEqual(rows[0][1], "ok")


if __name__ == "__main__":
    unittest.main()
