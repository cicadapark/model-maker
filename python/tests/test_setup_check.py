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


if __name__ == "__main__":
    unittest.main()
