"""tools/ci/locks.py without the network: what each lock is made from, how one is compared with what its recipe makes,
which wheels a platform takes, and how the audit reads a PyTorch build. CI's job "per-platform locks" runs these with
`packaging` installed and fails if any is skipped; the network half runs there as `locks.py check` and `licences`."""

import hashlib
import io
import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ci"))

import locks  # noqa: E402

try:
    import packaging  # noqa: F401
    HAS_PACKAGING = True
except ImportError:
    HAS_PACKAGING = False
NEEDS_PACKAGING = unittest.skipUnless(HAS_PACKAGING, "needs packaging (CI's lock job installs it from the lock)")

A_SUM, B_SUM, C_SUM = "a" * 64, "b" * 64, "c" * 64


class Recipe(unittest.TestCase):
    def test_the_measured_lock_is_never_written(self):
        with mock.patch.object(locks, "compiled") as compiled, mock.patch("builtins.open") as opened:
            locks.make([locks.MEASURED])
        compiled.assert_not_called()
        opened.assert_not_called()

    def test_each_new_lock_is_beside_the_measured_one_and_named_for_its_platform(self):
        self.assertEqual(locks.TARGETS[locks.MEASURED]["lock"], "python/requirements.lock")
        for name, target in locks.TARGETS.items():
            if name != locks.MEASURED:
                self.assertEqual(target["lock"], f"python/locks/{name}.lock")
            self.assertTrue(os.path.isfile(os.path.join(locks.REPOSITORY, target["lock"])), name)

    def test_every_change_from_the_measured_versions_says_why(self):
        for name, target in locks.TARGETS.items():
            for package, (version, why) in target["changes"].items():
                with self.subTest(name=name, package=package):
                    self.assertRegex(version, r"^\d+(\.\d+)+$")
                    self.assertGreater(len(why), 30)

    def test_the_constraints_are_the_measured_versions_with_the_targets_changes(self):
        for name, target in locks.TARGETS.items():
            pins = dict(line.split("==") for line in locks.constraints(target))
            measured = locks.parsed(locks.read_lock(locks.MEASURED))
            for package, (version, _) in measured.items():
                if package in target["torch"]:
                    self.assertNotIn(package, pins, f"{name}: {package} comes from its file, not a version")
                elif package in target["changes"]:
                    self.assertEqual(pins[package], target["changes"][package][0])
                else:
                    self.assertEqual(pins[package], version, f"{name}: {package}")

    def test_the_requirements_name_each_pytorch_build_by_its_file_and_sum(self):
        target = locks.TARGETS["linux-x86_64-rocm71"]
        wanted = locks.requirements(target)
        for package, (path, sha) in target["torch"].items():
            line = next(line for line in wanted if line.startswith(package + " @ "))
            self.assertTrue(line.endswith(f"/{path}#sha256={sha}"), line)
        self.assertIn("bitsandbytes==0.50.2", wanted)
        self.assertFalse([line for line in wanted if line.startswith("torch==")])

    def test_each_lock_holds_what_its_target_says(self):
        for name, target in locks.TARGETS.items():
            locked = locks.parsed(locks.read_lock(name))
            with self.subTest(name):
                for package, (path, sha) in target["torch"].items():
                    self.assertEqual(locked[package][0], locks.torch_url(path, sha))
                    self.assertEqual(locked[package][1], {sha})
                for package, (version, _) in target["changes"].items():
                    self.assertEqual(locked[package][0], version)
                if target["system"] != "linux" or target["torch"]:
                    self.assertFalse([p for p in locked if p.startswith("nvidia-") or p == "triton"],
                                     f"{name}: NVIDIA's Linux libraries are only for the measured lock")


    def test_each_locks_bitsandbytes_is_what_its_transformers_loads_in_4_bits(self):
        # The Windows card test, 28 Nov 2026: this branch's locks kept bitsandbytes 0.45.3 beside transformers 5.10.1,
        # which asks 0.46.1 or newer, and FLUX.1's text encoder refused to load in 4 bits.
        from model_maker import device

        for name in locks.TARGETS:
            locked = locks.parsed(locks.read_lock(name))
            with self.subTest(name):
                floor = device.bitsandbytes_floor(locked["transformers"][0])
                if floor:
                    self.assertGreaterEqual(device.release_of(locked["bitsandbytes"][0]), device.release_of(floor),
                                            f"{name}: bitsandbytes {locked['bitsandbytes'][0]} under transformers "
                                            f"{locked['transformers'][0]}, which asks {floor} or newer")

    def test_a_refused_lock_says_so_and_nothing_installs_it(self):
        workflow = open(os.path.join(locks.REPOSITORY, ".github", "workflows", "ci.yml")).read()
        setup = open(os.path.join(locks.REPOSITORY, "setup.sh")).read()
        refused = [name for name, target in locks.TARGETS.items() if target.get("refused")]
        self.assertEqual(refused, ["linux-x86_64-rocm71"])
        for name in refused:
            self.assertIn("REFUSED by the licence check", locks.read_lock(name))
            self.assertNotIn(name, setup)
            for line in workflow.splitlines():
                self.assertFalse(name in line and "lock:" in line, f"the workflow installs {name}: {line}")
        self.assertFalse(locks.TARGETS[locks.MEASURED].get("refused"))


class Comparing(unittest.TestCase):
    def files(self, names):
        return lambda package, pinned, sums: {sha: (names[sha], "url") if names.get(sha) else None for sha in sums}

    def test_a_new_source_archive_is_not_a_problem_but_a_new_wheel_is(self):
        names = {B_SUM: "pkg-1.0.tar.gz", C_SUM: "pkg-1.0-cp311-cp311-win_amd64.whl"}
        with mock.patch.object(locks, "files_of", self.files(names)):
            self.assertEqual(locks.sums_problems("x", "pkg", "1.0", {A_SUM}, {A_SUM, B_SUM}), [])
            problems = locks.sums_problems("x", "pkg", "1.0", {A_SUM}, {A_SUM, C_SUM})
        self.assertEqual(len(problems), 1)
        self.assertIn("win_amd64.whl", problems[0])

    def test_a_locked_sum_the_index_no_longer_gives_is_a_problem(self):
        with mock.patch.object(locks, "files_of", self.files({})):
            problems = locks.sums_problems("x", "pkg", "1.0", {A_SUM, B_SUM}, {A_SUM})
        self.assertEqual(problems, [f"x: pkg 1.0: the index no longer gives the locked sum {B_SUM}"])

    def test_a_lock_is_read_by_name_version_or_file_and_its_sums(self):
        torch = "https://example/torch-2.10.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=" + C_SUM
        text = (f"# a comment\nnumpy==2.4.6 \\\n    --hash=sha256:{A_SUM} \\\n    --hash=sha256:{B_SUM}\n"
                f"    # via\n    #   scipy\ntorch @ {torch} \\\n    --hash=sha256:{C_SUM}\n"
                f"Typing_Extensions==4.16.0 \\\n    --hash=sha256:{A_SUM}\n")
        self.assertEqual(locks.parsed(text), {
            "numpy": ("2.4.6", {A_SUM, B_SUM}),
            "torch": (torch, {C_SUM}),
            "typing-extensions": ("4.16.0", {A_SUM})})


@NEEDS_PACKAGING
class Platforms(unittest.TestCase):
    def chosen(self, name, files):
        lock = "".join(f"{package}=={version} \\\n" + " \\\n".join(f"    --hash=sha256:{sha}" for sha in files_)
                       + "\n" for package, version, files_ in files)
        index = {sha: (file_name, f"https://files/{file_name}") for _, _, files_ in files
                 for sha, file_name in files_.items()}
        with mock.patch.object(locks, "read_lock", lambda _: lock), \
                mock.patch.object(locks, "index_files", lambda package, version: index):
            return locks.chosen_wheels(name)

    def test_a_package_without_a_wheel_for_the_platform_is_a_problem(self):
        # bitsandbytes 0.45.3 has Linux and Windows wheels, none for macOS.
        files = {A_SUM: "bitsandbytes-0.45.3-py3-none-manylinux_2_24_x86_64.whl",
                 B_SUM: "bitsandbytes-0.45.3-py3-none-win_amd64.whl"}
        chosen, problems = self.chosen("macos-arm64", [("bitsandbytes", "0.45.3", files)])
        self.assertEqual(chosen, {})
        self.assertEqual(problems, ["macos-arm64: bitsandbytes 0.45.3: no wheel for aarch64-apple-darwin"])
        chosen, problems = self.chosen("windows-x86_64-cuda128", [("bitsandbytes", "0.45.3", files)])
        self.assertEqual(chosen["bitsandbytes"][0], files[B_SUM])
        self.assertEqual(problems, [])

    def test_the_best_tag_is_chosen_as_an_installer_would(self):
        files = {A_SUM: "numpy-2.4.6-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
                 B_SUM: "numpy-2.4.6-cp311-cp311-musllinux_1_2_x86_64.whl",
                 C_SUM: "numpy-2.4.6.tar.gz"}
        chosen, problems = self.chosen("linux-x86_64-cpu", [("numpy", "2.4.6", files)])
        self.assertEqual((chosen["numpy"][0], problems), (files[A_SUM], []))

    def test_macos_takes_wheels_up_to_its_version_only(self):
        files = {A_SUM: "bitsandbytes-0.50.2-py3-none-macosx_15_0_arm64.whl"}
        chosen, problems = self.chosen("macos-arm64", [("bitsandbytes", "0.50.2", files)])
        self.assertTrue(problems, "a wheel for macOS 15 does not install on the lock's macOS 14")
        files = {A_SUM: "bitsandbytes-0.50.2-py3-none-macosx_14_0_arm64.whl"}
        self.assertEqual(self.chosen("macos-arm64", [("bitsandbytes", "0.50.2", files)])[1], [])

    def test_the_audit_reads_a_pytorch_build_as_its_release(self):
        files = "https://download.pytorch.org/whl"
        lock = (f"torch @ {files}/rocm7.1/torch-2.10.0%2Brocm7.1-cp311-cp311-manylinux_2_28_x86_64.whl#sha256={A_SUM} \\\n"
                f"    --hash=sha256:{A_SUM}\n"
                f"triton-rocm @ {files}/triton_rocm-3.6.0-cp311-cp311-linux_x86_64.whl"
                f"#sha256={B_SUM} \\\n"
                f"    --hash=sha256:{B_SUM}\nnumpy==2.4.6 \\\n    --hash=sha256:{C_SUM}\n")
        with mock.patch.object(locks, "read_lock", lambda _: lock):
            view = locks.audit_view("linux-x86_64-rocm71")
        self.assertEqual(re.findall(r"^\S+==\S+", view, re.MULTILINE),
                         ["torch==2.10.0", "triton==3.6.0", "numpy==2.4.6"])
        self.assertEqual(len(re.findall(r"--hash=sha256:", view)), 3)

    def test_the_measured_lock_is_audited_as_it_is(self):
        self.assertEqual(locks.audit_view(locks.MEASURED), locks.read_lock(locks.MEASURED))



class WheelsRecord(unittest.TestCase):
    def test_a_wheels_files_are_read_from_its_record_as_an_installed_check_reads_them(self):
        # torchvision's Windows wheel: zip names with "/", RECORD with "\\"; the installed check sees RECORD's.
        import io
        import zipfile

        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as wheel:
            wheel.writestr("tv-1.0.dist-info/METADATA", "Name: tv\nVersion: 1.0\n")
            wheel.writestr("tv/zlib.dll", b"")
            wheel.writestr("tv-1.0.dist-info/RECORD", "tv\\zlib.dll,sha256=x,0\ntv-1.0.dist-info/METADATA,,\n")
        data.seek(0)
        with mock.patch.object(locks, "RemoteFile", lambda url: data):
            distribution = locks.WheelDistribution("https://files/tv-1.0-py3-none-win_amd64.whl")
        self.assertEqual(distribution.files, ["tv\\zlib.dll", "tv-1.0.dist-info/METADATA"])
        self.assertEqual(distribution.files[0].name, "tv\\zlib.dll", "as importlib's posix path keeps it")
        self.assertEqual(distribution.metadata["Name"], "tv")


if __name__ == "__main__":
    unittest.main()


class WheelFolder(unittest.TestCase):
    """`locks.py wheels`: the folder `setup --wheels` reads on a PC without internet (the Windows card test, 28 Nov)."""

    GOOD = b"a wheel's bytes"
    NEW = b"another wheel's bytes"
    TORCH = b"PyTorch's build"

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder)
        self.served = {"https://files/numpy.whl": self.GOOD, "https://files/scipy.whl": self.NEW,
                       "https://example/torch-2.10.0%2Bcu128-cp311-cp311-win_amd64.whl": self.TORCH}
        self.asked = []
        good, new, torch_sum = (hashlib.sha256(b).hexdigest() for b in (self.GOOD, self.NEW, self.TORCH))
        torch = "https://example/torch-2.10.0%2Bcu128-cp311-cp311-win_amd64.whl#sha256=" + torch_sum
        self.lock = (f"numpy==2.4.6 \\\n    --hash=sha256:{good} \\\n    --hash=sha256:{A_SUM}\n"
                     f"scipy==1.17.1 \\\n    --hash=sha256:{new}\n"
                     f"torch @ {torch} \\\n    --hash=sha256:{torch_sum}\n")
        self.chosen = {"numpy": ("numpy-2.4.6-cp311-cp311-win_amd64.whl", "https://files/numpy.whl"),
                       "scipy": ("scipy-1.17.1-cp311-cp311-win_amd64.whl", "https://files/scipy.whl"),
                       "torch": ("torch-2.10.0+cu128-cp311-cp311-win_amd64.whl", torch.split("#")[0])}

    def opened(self, url, timeout=30):
        self.asked.append(url)
        return io.BytesIO(self.served[url])

    def made(self):
        with mock.patch.object(locks, "read_lock", lambda name: self.lock), \
                mock.patch.object(locks, "chosen_wheels", lambda name: (dict(self.chosen), [])), \
                mock.patch.object(locks, "opened", self.opened):
            return locks.wheels("windows-x86_64-cuda128", self.folder)

    def test_each_wheel_by_its_locked_sum_with_a_sums_file_pytorch_from_its_address_too(self):
        self.assertEqual(self.made(), [])
        names = ["numpy-2.4.6-cp311-cp311-win_amd64.whl", "scipy-1.17.1-cp311-cp311-win_amd64.whl",
                 "torch-2.10.0+cu128-cp311-cp311-win_amd64.whl"]
        self.assertEqual(sorted(os.listdir(self.folder)), ["SHA256SUMS"] + names)
        sums = open(os.path.join(self.folder, "SHA256SUMS")).read().splitlines()
        self.assertEqual(sums, [f"{hashlib.sha256(content).hexdigest()}  {name}"
                                for name, content in zip(names, (self.GOOD, self.NEW, self.TORCH))])
        self.assertIn(self.chosen["torch"][1], self.asked)

    def test_a_wheel_already_there_is_kept_and_a_stale_one_removed(self):
        self.made()
        open(os.path.join(self.folder, "numpy-2.4.5-cp311-cp311-win_amd64.whl"), "wb").write(b"old")
        self.asked.clear()
        self.assertEqual(self.made(), [])
        self.assertEqual(self.asked, [])
        self.assertNotIn("numpy-2.4.5-cp311-cp311-win_amd64.whl", os.listdir(self.folder))

    def test_a_download_with_a_sum_the_lock_does_not_hold_is_a_problem_and_not_kept(self):
        self.served["https://files/scipy.whl"] = b"changed on the way"
        problems = self.made()
        self.assertEqual(problems, ["windows-x86_64-cuda128: scipy 1.17.1: scipy-1.17.1-cp311-cp311-win_amd64.whl came "
                                    "with a sum the lock does not hold"])
        self.assertNotIn("scipy-1.17.1-cp311-cp311-win_amd64.whl", os.listdir(self.folder))
        self.assertNotIn("scipy", open(os.path.join(self.folder, "SHA256SUMS")).read())
