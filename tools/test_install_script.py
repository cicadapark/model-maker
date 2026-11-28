"""tools/ci/install_script.py: install.ps1 gets the release's values, and only values it can use safely."""

import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("install_script", os.path.join(HERE, "ci", "install_script.py"))
install_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install_script)

NAME = "model-maker-0.3.3-windows-x86_64-unsigned.zip"
SUM = "a" * 64
URL = f"https://example.org/0.3.3/{NAME}"
SUMS = "https://example.org/sums.txt"


class InstallScript(unittest.TestCase):
    def setUp(self):
        with open(install_script.TEMPLATE, encoding="utf-8") as file:
            self.template = file.read()

    def test_the_release_values_are_written_in(self):
        text = install_script.filled(self.template, "0.3.3", NAME, SUM, URL, SUMS)
        for line in ("$Version       = '0.3.3'", f"$ArchiveUrl    = '{URL}'", f"$ArchiveSha256 = '{SUM}'",
                     f"$SumsUrl       = '{SUMS}'"):
            self.assertIn(line, text)
        self.assertLess(len(text.splitlines()), 80, "short enough to read before running it")

    def test_unsafe_values_are_refused(self):
        local = f"http://127.0.0.1:8765/{NAME}"
        for args in [("0.3.3", NAME, SUM, f"http://example.org/{NAME}", SUMS),
                     ("0.3.3", NAME, SUM, local, SUMS),
                     ("0.3.3", NAME, SUM, URL + "'; evil", SUMS),
                     ("0.3.3", NAME, SUM, "https://example.org/other.zip", SUMS),
                     ("0.3.3'", NAME, SUM, URL, SUMS),
                     ("0.3.3", NAME, "A" * 64, URL, SUMS)]:
            with self.assertRaises(ValueError, msg=args):
                install_script.filled(self.template, *args)
        self.assertIn(local, install_script.filled(self.template, "0.3.3", NAME, SUM, local, SUMS, test=True))

    def test_the_sums_are_read_from_the_replys_bytes(self):
        # A sums file served without a text type (a release asset: application/octet-stream) has no text in
        # .Content, only bytes (the Windows test of 26 November 2026); CI serves it so (ci.yml, SHA256SUMS).
        self.assertIn("[Text.Encoding]::UTF8.GetString($SumsReply.RawContentStream.ToArray())", self.template)
        self.assertNotIn(".Content ", self.template)
        self.assertNotIn(".Content)", self.template)


if __name__ == "__main__":
    unittest.main()
