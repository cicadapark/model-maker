"""tools/ci/parity.py: the exclusion list's patterns, and what counts as the same."""

import importlib.util
import json
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("parity", os.path.join(HERE, "ci", "parity.py"))
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


class Parity(unittest.TestCase):
    def test_every_exclusion_says_why(self):
        listed = parity.exclusions()
        self.assertGreater(len(listed), 5)
        for entry in listed:
            self.assertEqual(len(entry), 2, entry)
            self.assertGreater(len(entry[1]), 10, entry)

    def test_patterns(self):
        compiled = [parity.pattern_regex(p) for p, _ in parity.exclusions()]
        for name in (".venv/lib/python3.11/site-packages/torch/__pycache__/x.cpython-311.pyc", ".tools/uv-cache/a/b",
                     ".tools/c2pa-test/key.pem", ".tools/python/.lock", "target/debug/x"):
            self.assertTrue(parity.excluded(name, compiled), name)
        for name in (".venv/pyvenv.cfg", ".venv/lib/python3.11/site-packages/torch/__init__.py", ".tools/uv/uv",
                     ".tools/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11", ".tools/pythonx/.lock"):
            self.assertFalse(parity.excluded(name, compiled), name)

    def test_the_repositorys_own_path_does_not_count_and_anything_else_does(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for repository, extra in ((first, b""), (second, b"")):
                os.makedirs(os.path.join(repository, ".venv", "bin"))
                with open(os.path.join(repository, ".venv", "bin", "tool"), "wb") as file:
                    file.write(b"#!" + os.path.realpath(repository).encode() + b"/.venv/bin/python\n" + extra)
                os.makedirs(os.path.join(repository, ".tools", "uv-cache"))
                with open(os.path.join(repository, ".tools", "uv-cache", os.path.basename(repository)), "w") as file:
                    file.write("cache differs")
            a, b = parity.manifest(first), parity.manifest(second)
            self.assertEqual(parity.differences(a, b), ([], [], []))
            with open(os.path.join(second, ".venv", "bin", "tool"), "ab") as file:
                file.write(b"changed")
            self.assertEqual(parity.differences(a, parity.manifest(second)), ([], [], [".venv/bin/tool"]))
            os.makedirs(os.path.join(second, ".tools", "extra"))
            open(os.path.join(second, ".tools", "extra", "file"), "w").close()
            self.assertEqual(parity.differences(a, parity.manifest(second))[1], [".tools/extra/file"])
            json.dumps(a)


if __name__ == "__main__":
    unittest.main()
