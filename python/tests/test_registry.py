import contextlib
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

from model_maker import pipeline, registry

FETCH_MODELS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "setup", "fetch_models.py"))

# Files Python or torch would run that a model folder must not hold unlisted: an extension module is imported before
# the source of the same name (cutout.py imports `birefnet` from its folder), bytecode runs outside `__pycache__` too,
# and a pickle runs code under any of its names.
PLANTED_CODE = ("birefnet.so", "birefnet.cpython-311-x86_64-linux-gnu.so", "birefnet.abi3.so", "birefnet.pyd",
                "extra.pyc", "extra.PYO", "extra.pyw", "state.pickle", "state.joblib")
# Code and configurations in the hub's cache folder are not let be: a component's module named as `../.cache/x` in a
# model_index.json is imported from there.
IN_THE_HUBS_CACHE = (os.path.join(".cache", "huggingface", "download", "config.json"),
                     os.path.join(".cache", "huggingface", "x.py"))


def planted(folder, name, text="print('planted')\n"):
    path = os.path.join(folder, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        file.write(text)
    return path



def removed_tree(folder):
    """`folder` removed. Git keeps its objects read-only, and Windows refuses to delete a read-only file until it is
    made writable."""
    def writable_then_again(function, path, _):
        os.chmod(path, stat.S_IWRITE)
        function(path)

    shutil.rmtree(folder, onerror=writable_then_again)

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

    def test_an_unlisted_configuration_is_refused(self):
        # A config.json can name an attention kernel to fetch, a model_index.json custom code to load; a chat
        # template is run by the tokenizer.
        for name in ("config.json", os.path.join("unet", "config.json"), "model_index.json", "chat_template.JINJA"):
            with self.subTest(name):
                path = os.path.join(self.folder, name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as file:
                    file.write("{}\n")
                with self.assertRaisesRegex(registry.RefusedFile, "configuration the registry does not list"):
                    registry.verify_folder("m", self.folder, self.registry)
                with self.assertRaisesRegex(registry.RefusedFile, "configuration the registry does not list"):
                    registry.refuse_unlisted("m", self.folder, self.registry)
                os.remove(path)

    def test_other_unlisted_files_and_the_hubs_cache_files_are_let_be(self):
        # The hub's cache in a model folder holds only these kinds; none runs code or steers a load.
        cache = os.path.join(".cache", "huggingface", "download")
        for path in ("README.md", ".done", ".gitattributes", os.path.join(cache, "config.json.metadata"),
                     os.path.join(cache, "config.json.lock"), os.path.join(cache, "abc.incomplete"),
                     os.path.join(".cache", "huggingface", ".gitignore")):
            planted(self.folder, path, "not read\n")
        self.assertEqual(registry.verify_folder("m", self.folder, self.registry), 1)

    def test_code_python_or_torch_would_load_is_refused(self):
        for name in PLANTED_CODE:
            with self.subTest(name):
                path = planted(self.folder, name)
                with self.assertRaisesRegex(registry.RefusedFile, "code or a pickle the registry does not list"):
                    registry.verify_folder("m", self.folder, self.registry)
                with self.assertRaisesRegex(registry.RefusedFile, "code or a pickle the registry does not list"):
                    registry.refuse_unlisted("m", self.folder, self.registry)
                os.remove(path)

    def test_every_ending_python_imports_a_module_from_is_code(self):
        import importlib.machinery

        for ending in importlib.machinery.all_suffixes() + [".pyd", ".pyw", ".pyo"]:
            with self.subTest(ending):
                self.assertTrue(("module" + ending).lower().endswith(registry.CODE_ENDINGS))

    def test_code_and_configurations_in_the_hubs_cache_are_refused(self):
        for name in IN_THE_HUBS_CACHE:
            with self.subTest(name):
                path = planted(self.folder, name)
                with self.assertRaisesRegex(registry.RefusedFile, "the registry does not list"):
                    registry.verify_folder("m", self.folder, self.registry)
                with self.assertRaisesRegex(registry.RefusedFile, "the registry does not list"):
                    registry.refuse_unlisted("m", self.folder, self.registry)
                os.remove(path)

    def test_a_model_folder_under_a_cache_path_is_walked(self):
        # A models folder in, say, ~/.cache/model-maker/models: the part of the path above the model folder does not
        # turn the walk off.
        folder = os.path.join(self.work, ".cache", "model-maker", "models", "m")
        shutil.move(self.folder, folder)
        self.assertEqual(registry.verify_folder("m", folder, self.registry), 1)
        planted(folder, "evil.py")
        with self.assertRaisesRegex(registry.RefusedFile, "code or a pickle the registry does not list"):
            registry.verify_folder("m", folder, self.registry)
        with self.assertRaisesRegex(registry.RefusedFile, "code or a pickle the registry does not list"):
            registry.refuse_unlisted("m", folder, self.registry)

    def test_a_linked_folder_inside_is_refused(self):
        # The walk does not follow a link, so what is behind one would be neither summed nor refused.
        elsewhere = os.path.join(self.work, "elsewhere")
        os.makedirs(elsewhere)
        os.symlink(elsewhere, os.path.join(self.folder, "unet"))
        for behind in ((), ("README.md",), ("config.json", "evil.py")):
            with self.subTest(behind=behind):
                for name in behind:
                    planted(elsewhere, name)
                with self.assertRaisesRegex(registry.RefusedFile, "a linked folder inside a model folder"):
                    registry.verify_folder("m", self.folder, self.registry)
                with self.assertRaisesRegex(registry.RefusedFile, "a linked folder inside a model folder"):
                    registry.refuse_unlisted("m", self.folder, self.registry)

    def test_a_model_folder_that_is_itself_a_link_is_walked(self):
        # MODEL_MAKER_MODELS/<model> may point at a folder elsewhere: its files are walked and summed.
        linked = os.path.join(self.work, "linked")
        os.symlink(self.folder, linked)
        self.assertEqual(registry.verify_folder("m", linked, self.registry), 1)
        planted(self.folder, "evil.py")
        with self.assertRaisesRegex(registry.RefusedFile, "code or a pickle the registry does not list"):
            registry.refuse_unlisted("m", linked, self.registry)

    # os.geteuid exists only on Linux and macOS: asked on Windows, it would stop this whole module from importing.
    @unittest.skipIf(os.name == "nt", "Windows' chmod sets only the read-only flag: a folder cannot be made unlistable")
    @unittest.skipIf(os.name != "nt" and os.geteuid() == 0, "the superuser reads every folder")
    def test_a_folder_the_check_cannot_read_is_refused(self):
        # A folder that can be entered but not listed: a file in it can still be imported by its name.
        hidden = os.path.join(self.folder, "hidden")
        planted(hidden, "evil.py")
        os.chmod(hidden, 0o100)
        try:
            with self.assertRaisesRegex(registry.RefusedFile, "a folder the check cannot read"):
                registry.verify_folder("m", self.folder, self.registry)
            with self.assertRaisesRegex(registry.RefusedFile, "a folder the check cannot read"):
                registry.refuse_unlisted("m", self.folder, self.registry)
        finally:
            os.chmod(hidden, 0o700)

    def test_a_refusal_says_what_to_do(self):
        planted(self.folder, "config.json", "{}\n")
        with self.assertRaisesRegex(registry.RefusedFile, r"remove it, or set the models up again with \./setup\.sh"):
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


class WithoutTheSumCheck(unittest.TestCase):
    """`--no-verify` skips reading each file for its sum, but the folder is still walked: unlisted code is refused."""

    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.folder = os.path.join(self.work, "birefnet")
        os.makedirs(self.folder)
        with open(os.path.join(self.folder, "model.safetensors"), "wb") as file:
            file.write(b"weights the sums would refuse")

    def tearDown(self):
        shutil.rmtree(self.work)

    def plant(self, name, text="print('planted')\n"):
        with open(os.path.join(self.folder, name), "w") as file:
            file.write(text)

    def test_files_are_not_read_for_their_sums(self):
        registry.refuse_unlisted("birefnet", self.folder)
        self.assertEqual(pipeline.Settings(models=self.work, verify=False).verified("birefnet", "birefnet"),
                         [self.folder])

    def test_unlisted_code_a_pickle_or_a_configuration_is_still_refused(self):
        for name in ("extra.py", "pytorch_model.bin", "weights.PTH", "state.pkl", "model_index.json", "template.jinja",
                     *PLANTED_CODE, *IN_THE_HUBS_CACHE):
            with self.subTest(name):
                path = planted(self.folder, name)
                with self.assertRaisesRegex(registry.RefusedFile, "the registry does not list"):
                    registry.refuse_unlisted("birefnet", self.folder)
                with self.assertRaisesRegex(registry.RefusedFile, "the registry does not list"):
                    pipeline.Settings(models=self.work, verify=False).verified("birefnet", "birefnet")
                os.remove(path)

    def test_a_models_folder_under_a_cache_path_is_walked(self):
        models = os.path.join(self.work, ".cache", "model-maker", "models")
        os.makedirs(models)
        shutil.move(self.folder, os.path.join(models, "birefnet"))
        planted(os.path.join(models, "birefnet"), "evil.py")
        with self.assertRaisesRegex(registry.RefusedFile, "code or a pickle the registry does not list"):
            pipeline.Settings(models=models, verify=False).verified("birefnet", "birefnet")

    def test_a_linked_folder_inside_is_refused(self):
        elsewhere = os.path.join(self.work, "elsewhere")
        planted(elsewhere, "evil.py")
        os.symlink(elsewhere, os.path.join(self.folder, "unet"))
        with self.assertRaisesRegex(registry.RefusedFile, "a linked folder inside a model folder"):
            pipeline.Settings(models=self.work, verify=False).verified("birefnet", "birefnet")

    def test_a_listed_code_file_passes_and_cached_bytecode_is_removed(self):
        self.plant("birefnet.py", "changed, but its sum is not read without the check\n")
        cache = os.path.join(self.folder, "__pycache__")
        os.makedirs(cache)
        with open(os.path.join(cache, "birefnet.cpython-311.pyc"), "wb") as file:
            file.write(b"planted")
        registry.refuse_unlisted("birefnet", self.folder)
        self.assertFalse(os.path.exists(cache))

    def test_a_missing_folder_is_left_to_the_load(self):
        registry.refuse_unlisted("birefnet", os.path.join(self.work, "absent"))


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
        removed_tree(self.work)

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
        removed_tree(os.path.join(self.folder, ".git"))
        with self.assertRaisesRegex(registry.RefusedFile, "not a git checkout"):
            registry.verify_checkout("m", self.folder, self.registry)

    def test_on_windows_a_files_executable_bit_is_not_compared_its_content_is(self):
        # A checkout made on Linux and read on Windows (the test of 26 November 2026: 8 files 100755 -> 100644).
        self.git("config", "core.filemode", "true")
        os.chmod(os.path.join(self.folder, "package", "__init__.py"), 0o755)
        with self.subTest("on Linux the bit is compared"):
            if os.name == "nt":
                self.skipTest("Windows' chmod sets no executable bit, so there is no changed bit to see")
            with self.assertRaisesRegex(registry.RefusedFile, "changed"):
                registry.verify_checkout("m", self.folder, self.registry, platform="linux")
        registry.verify_checkout("m", self.folder, self.registry, platform="win32")
        with open(os.path.join(self.folder, "package", "__init__.py"), "a") as file:
            file.write("import os\n")
        with self.assertRaisesRegex(registry.RefusedFile, "changed"):
            registry.verify_checkout("m", self.folder, self.registry, platform="win32")


class NoTripoSGCode(unittest.TestCase):
    """No code of TripoSG's repository is fetched (its code folder holds the Tencent Hunyuan FlashVDM licence, which
    excludes the European Union): its weights only, run with model-maker's own code; a `sources/triposg` folder found
    there is removed."""

    def setUp(self):
        import importlib.util

        self.work = tempfile.mkdtemp()
        spec = importlib.util.spec_from_file_location("fetch_models", FETCH_MODELS)
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

    def test_the_question_judges_libraries_are_llama_cpps_own_beside_its_licence(self):
        from model_maker import check_licences as C

        folder = tempfile.mkdtemp()
        try:
            self.assertEqual(C.judge_libraries(os.path.join(folder, "absent")), ([], []))
            for name in ("libggml-cpu-haswell.so", "libllama.so.0", "libmtmd.so", "llama-server"):
                open(os.path.join(folder, name), "w").close()
            rows, problems = C.judge_libraries(folder)
            self.assertEqual([r[1] for r in rows], ["allowed"] * 3)
            self.assertEqual(len(problems), 1, problems)
            self.assertIn("LICENSE is missing", problems[0])
            with open(os.path.join(folder, "LICENSE"), "w") as file:
                file.write("MIT License\n")
            open(os.path.join(folder, "libcurl.so.4"), "w").close()
            rows, problems = C.judge_libraries(folder)
            self.assertEqual(problems, [f"{C.JUDGE_FOLDER}: it holds libcurl.so.4, which is not on the allow-list "
                                        "(read its licence first)"])
        finally:
            shutil.rmtree(folder)


class OptionalParts(unittest.TestCase):
    def test_an_optional_parts_problems_are_said_and_set_aside(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("fetch_models", FETCH_MODELS)
        fetch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fetch)
        problems = []
        fetch.optional_or_not({"id": "judge", "optional": "question-judge"}, ["a.gguf: missing"], problems)
        self.assertEqual(problems, [])
        fetch.optional_or_not({"id": "flux"}, ["b.safetensors: missing"], problems)
        self.assertEqual(problems, ["b.safetensors: missing"])


if __name__ == "__main__":
    unittest.main()


class SystemTrust(unittest.TestCase):
    """On Windows setup's downloads are checked by Windows' own verifier (truststore): Python's OpenSSL cannot make a
    fresh Windows fetch a root it does not hold yet. Elsewhere nothing changes."""

    def setUp(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("fetch_models", FETCH_MODELS)
        self.fetch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fetch)

    def test_not_on_linux_or_macos(self):
        def no_import(name):
            raise AssertionError(f"{name} imported")

        for platform in ("linux", "darwin"):
            self.assertFalse(self.fetch.system_trust(platform, no_import))

    def test_on_windows_truststore_once(self):
        calls = []

        class Truststore:
            @staticmethod
            def inject_into_ssl():
                calls.append("inject")

        self.assertTrue(self.fetch.system_trust("win32", lambda name: Truststore))
        self.assertTrue(self.fetch.system_trust("win32", lambda name: Truststore))
        self.assertEqual(calls, ["inject"])

    def test_on_windows_without_truststore_it_says_so(self):
        def missing(name):
            raise ImportError(name)

        with self.assertRaises(self.fetch.Failed) as said:
            self.fetch.system_trust("win32", missing)
        self.assertIn("Windows lock", str(said.exception))

    def test_every_download_goes_through_it(self):
        import inspect

        self.assertIn("system_trust()", inspect.getsource(self.fetch.opened))

    def test_truststore_is_in_the_windows_lock_only(self):
        repository = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
        for lock, wanted in (("python/locks/windows-x86_64-cuda128.lock", True), ("python/requirements.lock", False),
                             ("python/locks/macos-arm64.lock", False), ("python/locks/linux-x86_64-cpu.lock", False)):
            with open(os.path.join(repository, lock)) as file:
                self.assertEqual("\ntruststore==" in file.read(), wanted, lock)


class AGatedModelWithoutAToken(unittest.TestCase):
    """The hub token is for downloading only. The Windows card test of 28 Nov 2026: FLUX.1 [schnell] was complete on a
    read-only models share (the make had just used it), and setup still stopped on "flux1-schnell: no hub token"."""

    WEIGHTS = b"flux weights"

    def setUp(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("fetch_models", FETCH_MODELS)
        self.fetch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fetch)
        self.work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.work)
        self.models = os.path.join(self.work, "models")
        flux = next(m for m in self.fetch.models() if m["id"] == "flux1-schnell")
        self.assertTrue(flux.get("gated"))
        url = flux["parts"][0]["url"]
        self.entry = {**flux, "parts": [{"url": url, "sha256": hashlib.sha256(self.WEIGHTS).hexdigest(), "size_mb": 1}]}
        self.file = self.fetch.fetched_parts(self.entry, self.models)[0][1]

    def run_setup(self):
        def no_network(*_):
            raise AssertionError("the hub was asked")

        output = io.StringIO()
        argv = ["fetch_models.py", "--models", self.models, "--sources", os.path.join(self.work, "sources"), "--yes",
                "--only", "flux1-schnell"]
        with mock.patch.object(self.fetch, "models", lambda: [self.entry]), \
                mock.patch.object(self.fetch, "read_token", lambda _: None), \
                mock.patch.object(self.fetch, "opened", no_network), \
                mock.patch.object(self.fetch.sys, "argv", argv), contextlib.redirect_stdout(output):
            code = self.fetch.main()
        return code, output.getvalue()

    def placed(self, data):
        os.makedirs(os.path.dirname(self.file), exist_ok=True)
        with open(self.file, "wb") as file:
            file.write(data)

    def test_every_file_already_there_needs_no_token(self):
        self.placed(self.WEIGHTS)
        code, said = self.run_setup()
        self.assertEqual(code, 0, said)
        self.assertNotIn("PROBLEM", said)
        self.assertNotIn("no token was given", said)
        self.assertIn("every file already here with the registry's sum (no hub token needed)", said)

    def test_a_read_only_answer_record_fails_nothing(self):
        # A directory in the place of .accepted: it can be neither read nor written (also when the test runs as root).
        self.placed(self.WEIGHTS)
        os.makedirs(os.path.join(self.models, ".accepted"))
        code, said = self.run_setup()
        self.assertEqual(code, 0, said)
        self.assertIn(".accepted cannot be written: the answer is not kept", said)

    def test_a_missing_or_changed_file_without_a_token_says_what_to_do(self):
        for data in (None, b"something else"):
            with self.subTest(data=data):
                if data is None:
                    shutil.rmtree(self.models, ignore_errors=True)
                else:
                    self.placed(data)
                code, said = self.run_setup()
                self.assertEqual(code, 1, said)
                self.assertIn("1 of its 1 files are not here with the registry's sum", said)
                self.assertIn("--hf-token-file (see 'A gated model' in docs/setup.md)", said)
                self.assertIn("PROBLEM: flux1-schnell: no hub token, and 1 of its files are not here", said)
                if data is not None:
                    self.assertTrue(os.path.exists(self.file), "a changed file is not removed without a token")


class ALicenceTextSetupCarries(unittest.TestCase):
    """A licence's whole text (SDXL's) comes from the copy in setup/licences/, never from the internet when that copy
    is there: the Windows card test (28 Nov 2026) ran without internet, on a read-only models share."""

    WEIGHTS = b"sdxl weights"

    def setUp(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("fetch_models", FETCH_MODELS)
        self.fetch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fetch)
        self.work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.work)
        self.models = os.path.join(self.work, "models")
        sdxl = next(m for m in self.fetch.models() if m["id"] == "sdxl-base-1.0")
        part = {"url": sdxl["parts"][0]["url"], "sha256": hashlib.sha256(self.WEIGHTS).hexdigest(), "size_mb": 1}
        self.entry = {**sdxl, "parts": [part]}
        path = self.fetch.fetched_parts(self.entry, self.models)[0][1]
        os.makedirs(os.path.dirname(path))
        with open(path, "wb") as file:
            file.write(self.WEIGHTS)

    def run_setup(self, opened):
        output = io.StringIO()
        argv = ["fetch_models.py", "--models", self.models, "--sources", os.path.join(self.work, "sources"), "--yes",
                "--only", "sdxl-base-1.0"]
        with mock.patch.object(self.fetch, "models", lambda: [self.entry]), \
                mock.patch.object(self.fetch, "read_token", lambda _: None), \
                mock.patch.object(self.fetch, "opened", opened), \
                mock.patch.object(self.fetch.sys, "argv", argv), contextlib.redirect_stdout(output):
            code = self.fetch.main()
        return code, output.getvalue()

    def test_every_licence_text_of_the_registry_is_carried_with_its_sum(self):
        for entry in self.fetch.models():
            if entry.get("licence_text"):
                copy = os.path.join(self.fetch.LICENCE_TEXTS, f"{entry['id']}-LICENSE.md")
                self.assertEqual(self.fetch.sha256_of(copy).hexdigest(), entry["licence_text"]["sha256"], copy)

    def test_offline_on_a_read_only_share_with_no_earlier_answer_the_carried_copy_is_shown(self):
        def offline(*_):
            raise AssertionError("the internet was asked")

        # Read-only stand-ins (also when the test runs as root): no .accepted can be read or written, no licence
        # folder can be made in the models folder.
        os.makedirs(os.path.join(self.models, ".accepted"))
        with open(os.path.join(self.models, self.fetch.LICENCE_FOLDER), "w") as file:
            file.write("not a folder")
        code, said = self.run_setup(offline)
        self.assertEqual(code, 0, said)
        copy = os.path.join(self.fetch.LICENCE_TEXTS, "sdxl-base-1.0-LICENSE.md")
        self.assertIn(f"Read its whole text before you answer: {copy} (checked against the registry's sum)",
                      " ".join(said.split()))

    def test_only_with_no_copy_anywhere_it_is_fetched_and_kept_with_its_sum(self):
        with open(os.path.join(self.fetch.LICENCE_TEXTS, "sdxl-base-1.0-LICENSE.md"), "rb") as file:
            text = file.read()
        asked = []

        def hub(url, token, start=0):
            asked.append(url)
            return io.BytesIO(text)

        with mock.patch.object(self.fetch, "LICENCE_TEXTS", os.path.join(self.work, "no-copies")):
            code, said = self.run_setup(hub)
        self.assertEqual(code, 0, said)
        self.assertEqual(asked, [self.entry["licence_text"]["url"]])
        kept = os.path.join(self.models, self.fetch.LICENCE_FOLDER, "sdxl-base-1.0-LICENSE.md")
        self.assertEqual(self.fetch.sha256_of(kept).hexdigest(), self.entry["licence_text"]["sha256"])


class TheVerifiedCache(unittest.TestCase):
    """models/.verified, the stamps of files whose sums passed: keyed inside the models folder, so a folder moved or
    shared with another computer still matches; a models folder that cannot be written (shared read-only, as in the
    Windows test of 26 November 2026) is checked without it instead of stopping setup."""

    def setUp(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("fetch_models", FETCH_MODELS)
        self.fetch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fetch)
        self.work = tempfile.mkdtemp()
        self.models = os.path.join(self.work, "models")
        self.file = os.path.join(self.models, "flux", "model.bin")
        os.makedirs(os.path.dirname(self.file))
        with open(self.file, "wb") as file:
            file.write(b"weights")
        self.sha256 = hashlib.sha256(b"weights").hexdigest()

    def tearDown(self):
        shutil.rmtree(self.work)

    def checked(self, models):
        verified = self.fetch.Verified(models)
        path = os.path.join(models, "flux", "model.bin")
        return verified, self.fetch.checked_or_fetched([("u", path, self.sha256, 1)], None, verified, verify_only=True)

    def test_keyed_inside_the_models_folder_so_a_moved_folder_still_matches(self):
        self.assertEqual(self.checked(self.models)[1], [])
        with open(os.path.join(self.models, ".verified")) as file:
            self.assertEqual(list(json.load(file)), ["flux/model.bin"])
        moved = os.path.join(self.work, "elsewhere")
        shutil.copytree(self.models, moved)
        self.assertTrue(self.fetch.Verified(moved).holds(os.path.join(moved, "flux", "model.bin"), self.sha256))

    def test_a_folder_that_cannot_be_written_is_checked_without_it(self):
        # Something that cannot be written in the place of the cache's new copy (so also when the test runs as root).
        os.makedirs(os.path.join(self.models, ".verified.new"))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            verified, problems = self.checked(self.models)
            verified.add(self.file, self.sha256)
        self.assertEqual(problems, [])
        self.assertFalse(verified.writable)
        self.assertFalse(os.path.exists(os.path.join(self.models, ".verified")))
        self.assertEqual(output.getvalue().count("going on without it"), 1, output.getvalue())
        self.assertTrue(verified.holds(self.file, self.sha256))
