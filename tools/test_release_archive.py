"""tools/ci/release_archive.py: the programs where the launcher looks for itself and its tool, and the sum."""

import hashlib
import importlib.util
import os
import subprocess
import tarfile
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("release_archive", os.path.join(HERE, "ci", "release_archive.py"))
release_archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_archive)


class ReleaseArchive(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.repository = os.path.join(self.work, "repository")
        os.makedirs(os.path.join(self.repository, "python", "model_maker"))
        open(os.path.join(self.repository, "python", "model_maker", "__init__.py"), "w").close()
        with open(os.path.join(self.repository, "model-maker"), "w") as file:
            file.write("#!/usr/bin/env bash\n")
        git = ["git", "-C", self.repository, "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run(["git", "init", "-q", self.repository], check=True)
        subprocess.run(git + ["add", "-A"], check=True)
        subprocess.run(git + ["commit", "-q", "-m", "x"], check=True)
        self.files = []
        for name in ("launcher", "simplify", "licences"):
            path = os.path.join(self.work, name)
            with open(path, "w") as file:
                file.write(name)
            self.files.append(path)
        self.companion = os.path.join(self.work, "companion")
        with open(self.companion, "w") as file:
            file.write("companion")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.work)

    def test_linux_puts_the_programs_in_bin_and_keeps_the_bash_command(self):
        path, digest = release_archive.made("linux-x86_64", "0.3.3", *self.files, os.path.join(self.work, "out"),
                                            repository=self.repository)
        self.assertTrue(path.endswith("model-maker-0.3.3-linux-x86_64-unsigned.tar.gz"))
        with tarfile.open(path) as archive:
            names = archive.getnames()
            self.assertEqual(archive.extractfile("model-maker-0.3.3/model-maker").read(), b"#!/usr/bin/env bash\n")
            self.assertEqual(archive.getmember("model-maker-0.3.3/bin/model-maker").mode, 0o755)
        self.assertIn("model-maker-0.3.3/bin/model-maker-simplify", names)
        self.assertIn("model-maker-0.3.3/python/model_maker/__init__.py", names)
        with open(path, "rb") as file:
            self.assertEqual(hashlib.sha256(file.read()).hexdigest(), digest)
        with open(path + ".sha256") as file:
            self.assertEqual(file.read(), f"{digest}  {os.path.basename(path)}\n")

    def test_windows_puts_them_at_the_top_as_exe_in_a_zip(self):
        path, _ = release_archive.made("windows-x86_64", "0.3.3", *self.files, os.path.join(self.work, "out"),
                                       repository=self.repository)
        self.assertTrue(path.endswith("-windows-x86_64-unsigned.zip"))
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        self.assertIn("model-maker-0.3.3/model-maker.exe", names)
        self.assertIn("model-maker-0.3.3/model-maker-simplify.exe", names)
        self.assertIn("model-maker-0.3.3/THIRD-PARTY-LICENCES.md", names)

    def test_macos_replaces_the_bash_command(self):
        path, _ = release_archive.made("macos-arm64", "0.3.3", *self.files, os.path.join(self.work, "out"),
                                       repository=self.repository)
        with tarfile.open(path) as archive:
            self.assertEqual(archive.extractfile("model-maker-0.3.3/model-maker").read(), b"launcher")
            self.assertEqual(archive.getnames().count("model-maker-0.3.3/model-maker"), 1)

    def test_the_companion_goes_beside_the_launcher_when_given(self):
        for system, inside in (("linux-x86_64", "bin/model-maker-companion"),
                               ("windows-x86_64", "model-maker-companion.exe"),
                               ("macos-arm64", "model-maker-companion")):
            out = os.path.join(self.work, "out-" + system)
            path, _ = release_archive.made(system, "0.3.3", *self.files, out, repository=self.repository,
                                           companion=self.companion)
            if path.endswith(".zip"):
                with zipfile.ZipFile(path) as archive:
                    self.assertEqual(archive.read(f"model-maker-0.3.3/{inside}"), b"companion")
            else:
                with tarfile.open(path) as archive:
                    self.assertEqual(archive.getmember(f"model-maker-0.3.3/{inside}").mode, 0o755)
        path, _ = release_archive.made("linux-x86_64", "0.3.3", *self.files, os.path.join(self.work, "none"),
                                       repository=self.repository)
        with tarfile.open(path) as archive:
            self.assertNotIn("model-maker-0.3.3/bin/model-maker-companion", archive.getnames())


if __name__ == "__main__":
    unittest.main()
