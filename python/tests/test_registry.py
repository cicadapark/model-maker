import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest

from model_maker import pipeline, registry

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

    @unittest.skipIf(os.geteuid() == 0, "the superuser reads every folder")
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

        spec = importlib.util.spec_from_file_location(
            "fetch_models", os.path.join(os.path.dirname(__file__), "..", "..", "setup", "fetch_models.py"))
        fetch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fetch)
        problems = []
        fetch.optional_or_not({"id": "judge", "optional": "question-judge"}, ["a.gguf: missing"], problems)
        self.assertEqual(problems, [])
        fetch.optional_or_not({"id": "flux"}, ["b.safetensors: missing"], problems)
        self.assertEqual(problems, ["b.safetensors: missing"])


if __name__ == "__main__":
    unittest.main()
