import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest

from model_maker import pipeline, registry


class TheRegistry(unittest.TestCase):
    def test_every_model_the_pipeline_names_is_listed_with_a_licence_and_its_files(self):
        known = registry.models()
        for model_id in pipeline.MODELS_USED + [pipeline.STATE_MODEL]:
            entry = known[model_id]
            self.assertTrue(entry["licence"] and entry["licence_url"].startswith("https://"), model_id)
            self.assertTrue(entry["parts"], model_id)
            for part in entry["parts"]:
                self.assertRegex(part["sha256"], r"^[0-9a-f]{64}$")
        self.assertNotRegex(" ".join(m["licence"] for m in known.values()), r"GPL|NC|[Nn]on-?commercial")

    def test_a_sidecar_names_each_model_with_its_version_and_licence(self):
        record = registry.used_record(pipeline.MODELS_USED)
        self.assertEqual([m["id"] for m in record], pipeline.MODELS_USED)
        self.assertTrue(all(m["version"] and m["licence"] for m in record))


class TheSumCheck(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.folder = os.path.join(self.work, "m")
        os.makedirs(self.folder)
        self.weights = b"weights"
        with open(os.path.join(self.folder, "model.safetensors"), "wb") as file:
            file.write(self.weights)
        self.registry = os.path.join(self.work, "models.toml")
        digest = hashlib.sha256(self.weights).hexdigest()
        with open(self.registry, "w") as file:
            file.write('[[model]]\nid = "m"\nparts = [{ url = "https://example.org/m/resolve/abc/model.safetensors", '
                       f'sha256 = "{digest}" }}]\n')

    def tearDown(self):
        shutil.rmtree(self.work)

    def test_a_listed_file_with_its_sum_passes(self):
        self.assertEqual(registry.verify_folder("m", self.folder, self.registry), 1)

    def test_a_changed_file_is_refused(self):
        with open(os.path.join(self.folder, "model.safetensors"), "ab") as file:
            file.write(b"!")
        with self.assertRaisesRegex(registry.RefusedFile, "sha256"):
            registry.verify_folder("m", self.folder, self.registry)

    def test_unlisted_code_is_refused(self):
        with open(os.path.join(self.folder, "extra.py"), "w") as file:
            file.write("print('hello')\n")
        with self.assertRaisesRegex(registry.RefusedFile, "does not list"):
            registry.verify_folder("m", self.folder, self.registry)

    def test_a_folder_without_any_listed_file_is_refused(self):
        os.remove(os.path.join(self.folder, "model.safetensors"))
        with self.assertRaisesRegex(registry.RefusedFile, "none of the files"):
            registry.verify_folder("m", self.folder, self.registry)

    def test_cached_bytecode_is_removed_before_the_check(self):
        # A planted .pyc whose stamp matches its source runs in its place.
        cache = os.path.join(self.folder, "__pycache__")
        os.makedirs(cache)
        with open(os.path.join(cache, "birefnet.cpython-311.pyc"), "wb") as file:
            file.write(b"planted")
        registry.verify_folder("m", self.folder, self.registry)
        self.assertFalse(os.path.exists(cache))


class TheCodeCheck(unittest.TestCase):
    """A model's own code (MV-Adapter's) is imported only from a checkout at the registry's commit, unchanged."""

    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.folder = os.path.join(self.work, "code")
        os.makedirs(os.path.join(self.folder, "package"))
        with open(os.path.join(self.folder, "package", "__init__.py"), "w") as file:
            file.write("VALUE = 1\n")
        self.git("init", "-q")
        self.git("add", "-A")
        self.git("-c", "user.name=test", "-c", "user.email=test@example.org", "commit", "-q", "-m", "code")
        commit = self.git("rev-parse", "HEAD").strip()
        self.registry = os.path.join(self.work, "models.toml")
        with open(self.registry, "w") as file:
            file.write(f'[[model]]\nid = "m"\nsource = {{ url = "https://example.org/m.git", commit = "{commit}", '
                       'folder = "code" }\nparts = []\n')

    def tearDown(self):
        shutil.rmtree(self.work)

    def git(self, *args):
        return subprocess.run(["git", "-C", self.folder, *args], check=True, capture_output=True, text=True).stdout

    def test_the_registry_commit_unchanged_passes(self):
        cache = os.path.join(self.folder, "package", "__pycache__")
        os.makedirs(cache)
        registry.verify_checkout("m", self.folder, self.registry)
        self.assertFalse(os.path.exists(cache))

    def test_a_changed_file_is_refused(self):
        with open(os.path.join(self.folder, "package", "__init__.py"), "a") as file:
            file.write("import os\n")
        with self.assertRaisesRegex(registry.RefusedFile, "changed"):
            registry.verify_checkout("m", self.folder, self.registry)

    def test_an_added_file_is_refused(self):
        with open(os.path.join(self.folder, "torch.py"), "w") as file:
            file.write("print('planted')\n")
        with self.assertRaisesRegex(registry.RefusedFile, "changed"):
            registry.verify_checkout("m", self.folder, self.registry)

    def test_another_commit_is_refused(self):
        with open(os.path.join(self.folder, "package", "more.py"), "w") as file:
            file.write("VALUE = 2\n")
        self.git("add", "-A")
        self.git("-c", "user.name=test", "-c", "user.email=test@example.org", "commit", "-q", "-m", "more")
        with self.assertRaisesRegex(registry.RefusedFile, "not the registry's"):
            registry.verify_checkout("m", self.folder, self.registry)

    def test_a_folder_that_is_not_a_checkout_is_refused(self):
        shutil.rmtree(os.path.join(self.folder, ".git"))
        with self.assertRaisesRegex(registry.RefusedFile, "not a git checkout"):
            registry.verify_checkout("m", self.folder, self.registry)


class NoTripoSGCode(unittest.TestCase):
    """No code of TripoSG's repository is fetched (its code folder holds the Tencent Hunyuan FlashVDM licence, which
    excludes the European Union): its weights only, run with model-maker's own code; a `sources/triposg` folder found
    there is removed."""

    def setUp(self):
        import importlib.util

        self.work = tempfile.mkdtemp()
        spec = importlib.util.spec_from_file_location(
            "fetch_models", os.path.join(os.path.dirname(__file__), "..", "..", "setup", "fetch_models.py"))
        self.fetch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fetch)

    def tearDown(self):
        shutil.rmtree(self.work)

    def test_the_registry_fetches_triposgs_weights_and_no_code_of_its_repository(self):
        entry = registry.models()["triposg"]
        self.assertNotIn("source", entry)
        for model in registry.models().values():
            self.assertNotIn("VAST-AI-Research", model.get("source", {}).get("url", ""), model["id"])
            self.assertFalse([part for part in model["parts"] if "VAST-AI-Research" in part["url"]], model["id"])
        for part in entry["parts"]:
            self.assertTrue(part["url"].startswith("https://huggingface.co/VAST-AI/TripoSG/resolve/"), part["url"])
            self.assertTrue(part["url"].endswith((".safetensors", ".json")), part["url"])

    def test_code_an_older_setup_fetched_is_removed(self):
        sources = os.path.join(self.work, "sources")
        old = os.path.join(sources, "triposg", "triposg", "models")
        os.makedirs(old)
        with open(os.path.join(old, "attention_processor.py"), "w") as file:
            file.write("fetched by an older setup\n")
        os.makedirs(os.path.join(sources, "mv-adapter"))
        self.assertEqual(self.fetch.retired_sources_removed(sources), [os.path.join(sources, "triposg")])
        self.assertEqual(os.listdir(sources), ["mv-adapter"])
        self.assertEqual(self.fetch.retired_sources_removed(sources), [])


class BundledLibraries(unittest.TestCase):
    """Every shared library a wheel bundles is judged against an allow-list."""

    def test_each_library_is_allowed_awaited_or_unknown(self):
        from model_maker import check_licences as C

        self.assertEqual(C.judged_library("scipy.libs/libgfortran-040039e1.so.5", "libgfortran-040039e1.so.5")[0],
                         "allowed")
        # The project's one recorded exception (CONTRIBUTING.md, rule 1), allowed with its licence named.
        verdict, licence = C.judged_library("numpy.libs/libquadmath-96973f99.so.0", "libquadmath-96973f99.so.0")
        self.assertEqual(verdict, "allowed")
        self.assertIn("LGPL", licence)
        self.assertEqual(C.judged_library("cv2/libavcodec-1.so.59", "libavcodec-1.so.59")[0], "unknown")
        self.assertEqual(C.judged_library("torchvision/_C.so", "_C.so")[0], "allowed")
        self.assertEqual(C.judged_library("thing/_C.so", "_C.so")[0], "unknown")
        self.assertIsNone(C.EXTENSION_MODULE.search("libtorch.so"))
        self.assertIsNotNone(C.EXTENSION_MODULE.search("_ckdtree.cpython-311-x86_64-linux-gnu.so"))


if __name__ == "__main__":
    unittest.main()
