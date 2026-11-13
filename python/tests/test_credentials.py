"""Content credentials (credentials.py): every picture model-maker writes carries a signed C2PA manifest that says it
was made by AI models, and by which; nothing private is in it; the pixels are the same; a bug report keeps it."""

import argparse
import contextlib
import getpass
import importlib.util
import io
import json
import os
import shutil
import socket
import struct
import tempfile
import unittest
import zlib
from unittest import mock

import numpy as np
from PIL import Image

from model_maker import bug_report, credentials, glb, object_checks as O, pipeline, render

from . import shapes

REPOSITORY = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def noise_picture(side=96, seed=0):
    return Image.fromarray((np.random.default_rng(seed).random((side, side, 3)) * 255).astype(np.uint8))


def pixels(path):
    with Image.open(path) as picture:
        return np.asarray(picture.convert("RGB"))


def png_chunk(kind, body):
    return struct.pack(">I4s", len(body), kind) + body + struct.pack(">I", zlib.crc32(kind + body))


def with_text_chunk(data, text):
    """A PNG's bytes with a tEXt chunk put in after its header."""
    header_end = 8 + 8 + 13 + 4
    return data[:header_end] + png_chunk(b"tEXt", text) + data[header_end:]


class Signing(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="model-maker-credentials-")

    def tearDown(self):
        shutil.rmtree(self.folder)

    def test_a_picture_saved_carries_a_manifest_that_holds_and_says_made_by_ai(self):
        path = credentials.save_picture(noise_picture(), os.path.join(self.folder, "try.png"),
                                        credentials.PICTURE_MODELS)
        reading = credentials.read(path)
        self.assertTrue(reading.marked, (reading.state, reading.failures))
        self.assertEqual(reading.source_type,
                         "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia")
        self.assertEqual(reading.generator, "model-maker")
        self.assertTrue(reading.version)
        self.assertEqual(reading.models, ["FLUX.1 [schnell]"])
        created = [action for assertion in reading.manifest["assertions"] if assertion["label"].startswith(
            "c2pa.actions") for action in assertion["data"]["actions"]]
        self.assertEqual([action["action"] for action in created], ["c2pa.created"])

    def test_the_picture_read_back_is_the_same(self):
        picture = noise_picture(seed=3)
        for ending in (".png", ".webp", ".jpg"):
            with self.subTest(ending=ending):
                plain = os.path.join(self.folder, "plain" + ending)
                picture.save(plain, **({"lossless": True} if ending == ".webp" else {}))
                signed = os.path.join(self.folder, "signed" + ending)
                shutil.copyfile(plain, signed)
                credentials.sign_file(signed, credentials.OBJECT_MODELS)
                self.assertTrue(credentials.read(signed).marked)
                self.assertTrue(np.array_equal(pixels(plain), pixels(signed)))
                if ending == ".png":
                    self.assertTrue(np.array_equal(pixels(signed), np.asarray(picture)))
        saved = credentials.save_picture(picture, os.path.join(self.folder, "saved.png"))
        self.assertTrue(np.array_equal(pixels(saved), np.asarray(picture)))

    def test_each_kind_of_picture_says_its_own_source(self):
        self.assertEqual(credentials.job_picture_kind("pictures/123.png"),
                         (credentials.TRAINED, credentials.PICTURE_MODELS))
        self.assertEqual(credentials.job_picture_kind("views/view-3.png")[0], credentials.TRAINED)
        self.assertEqual(credentials.job_picture_kind("views/touch-up-1-mask.png"), (credentials.ALGORITHMIC, ()))
        self.assertEqual(credentials.job_picture_kind("report/picture-cut.png"),
                         (credentials.TRAINED, credentials.CUT_OUT_MODELS))
        self.assertEqual(credentials.job_picture_kind("report/outline.png")[0], credentials.COMPOSITE)
        self.assertEqual(credentials.job_picture_kind("report/coverage-likeness.png")[0], credentials.COMPOSITE)
        self.assertIsNone(credentials.job_picture_kind("shape.npz"))
        mask = credentials.save_picture(Image.new("L", (32, 32), 255), os.path.join(self.folder, "mask.png"), (),
                                        credentials.ALGORITHMIC)
        reading = credentials.read(mask)
        self.assertTrue(reading.marked)
        self.assertEqual(reading.source_type, "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicMedia")
        self.assertEqual(reading.models, [])

    def test_the_models_named_are_the_pipelines(self):
        self.assertEqual(list(credentials.OBJECT_MODELS), pipeline.MODELS_USED)
        for kind in (credentials.PICTURE_MODELS, credentials.CUT_OUT_MODELS, credentials.SHAPE_MODELS):
            self.assertTrue(set(kind) <= set(credentials.OBJECT_MODELS))

    def test_no_path_user_or_machine_name_is_in_a_manifest(self):
        private = os.path.join(self.folder, "a-private-folder-name")
        os.makedirs(private)
        path = credentials.save_picture(noise_picture(), os.path.join(private, "unusual-file-name.png"),
                                        credentials.OBJECT_MODELS)
        with open(path, "rb") as file:
            data = file.read()
        manifest_text = json.dumps(credentials.read(path).manifest)
        home = os.path.expanduser("~")
        names = {getpass.getuser(), socket.gethostname(), home, self.folder, "a-private-folder-name",
                 "unusual-file-name", credentials.test_folder()}
        for name in names:
            if len(name) < 4 or name in credentials.TEST_ORGANISATION:
                continue
            with self.subTest(name=name):
                self.assertNotIn(name.encode(), data)
                self.assertNotIn(name, manifest_text)

    def test_without_a_key_the_test_certificate_signs_and_says_so(self):
        signing = credentials.chosen(environment={})
        self.assertTrue(signing.test)
        self.assertIn("test certificate", signing.words)
        self.assertIn("untrusted", signing.words)
        self.assertEqual(signing.record()["certificate"], "test")
        path = credentials.save_picture(noise_picture(), os.path.join(self.folder, "a.png"), signing=signing)
        reading = credentials.read(path)
        self.assertTrue(reading.marked)
        self.assertFalse(reading.trusted)
        self.assertEqual(reading.issuer, credentials.TEST_ORGANISATION)
        self.assertIn(credentials.UNTRUSTED, reading.failures)
        # Only the user who made the test key can read it.
        key = os.path.join(credentials.test_folder(), credentials.TEST_KEY)
        self.assertEqual(os.stat(key).st_mode & 0o777, 0o600)
        said = []
        pipeline.run([], pipeline.Settings(verify=False), report=said.append)
        self.assertTrue(any("test certificate" in line and "untrusted" in line for line in said), said)

    def test_a_key_and_certificate_given_sign(self):
        given = os.path.join(self.folder, "given")
        key, chain = credentials.made_test_certificate(given)
        signing = credentials.chosen(environment={credentials.KEY_VARIABLE: key, credentials.CERT_VARIABLE: chain})
        self.assertFalse(signing.test)
        self.assertEqual(signing.record()["certificate"], "given")
        path = credentials.save_picture(noise_picture(), os.path.join(self.folder, "b.png"), signing=signing)
        self.assertTrue(credentials.read(path).marked)
        with self.assertRaises(credentials.NotSigned):
            credentials.chosen(key=key, environment={})
        with self.assertRaises(credentials.NotSigned):
            credentials.chosen(key=os.path.join(self.folder, "none.key"), chain=chain, environment={})
        with open(os.path.join(self.folder, "not-a-key.pem"), "w") as file:
            file.write("not a key")
        with self.assertRaises(credentials.NotSigned):
            credentials.chosen(key=os.path.join(self.folder, "not-a-key.pem"), chain=chain, environment={})

    def test_a_picture_changed_after_signing_does_not_hold(self):
        path = credentials.save_picture(noise_picture(), os.path.join(self.folder, "c.png"))
        with open(path, "rb") as file:
            data = bytearray(file.read())
        at = data.index(b"IDAT") + 20
        data[at] ^= 0xFF
        with open(path, "wb") as file:
            file.write(bytes(data))
        reading = credentials.read(path)
        self.assertTrue(reading.found)
        self.assertFalse(reading.marked)

    def test_the_command_reads_a_pictures_credentials(self):
        import contextlib

        from model_maker.__main__ import main

        path = credentials.save_picture(noise_picture(), os.path.join(self.folder, "e.png"), credentials.SHAPE_MODELS)
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            self.assertEqual(main(["credentials", "--read", path]), 0)
            self.assertEqual(main(["credentials"]), 0)
        self.assertIn("marked (Valid); http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
                      said.getvalue())
        self.assertIn("test certificate", said.getvalue())

    def test_a_plain_picture_has_none(self):
        path = os.path.join(self.folder, "plain.png")
        noise_picture().save(path)
        self.assertFalse(credentials.read(path).found)


class BugReport(unittest.TestCase):
    def test_a_signed_png_keeps_its_credentials(self):
        folder = tempfile.mkdtemp()
        try:
            path = credentials.save_picture(noise_picture(), os.path.join(folder, "d.png"))
            with open(path, "rb") as file:
                data = file.read()
            kept, dropped = bug_report.png_without_text(data)
            self.assertEqual(dropped, 0)
            self.assertTrue(credentials.read(data=kept, mime_type="image/png").marked)
            # With a text chunk dropped the signed bytes change: a manifest that would no longer hold goes with it.
            noted, dropped = bug_report.png_without_text(with_text_chunk(data, b"Comment\0from a tool"))
            self.assertEqual(dropped, 2)
            self.assertFalse(credentials.read(data=noted, mime_type="image/png").found)
            self.assertTrue(np.array_equal(np.asarray(Image.open(io.BytesIO(noted)).convert("RGB")),
                                           pixels(path)))
        finally:
            shutil.rmtree(folder)


class Finish(unittest.TestCase):
    """A job made before its pictures were signed: the finish signs every one, and its own report's; the sidecar,
    the report and the .glb say so."""

    def made_job(self, folder):
        job = pipeline.Job(os.path.join(folder, "chair"), "an old looking chair")
        corners, triangles = shapes.chair()
        corners = O.set_down(corners, job["size_m"])
        mask = render.silhouette(corners, triangles, 20, 256)
        Image.fromarray(np.where(mask[..., None], 120, 210).astype(np.uint8).repeat(3, -1)).save(
            job.path("picture.png"))
        Image.fromarray(mask.astype(np.uint8) * 255).save(job.path("mask.png"))
        os.makedirs(job.path("pictures"))
        os.makedirs(job.path("views"))
        shutil.copyfile(job.path("picture.png"), job.path("pictures", "7.png"))
        shutil.copyfile(job.path("mask.png"), job.path("pictures", "7-mask.png"))
        for k in range(6):
            noise_picture(64, k).save(job.path("views", f"view-{k}.png"))
        noise_picture(64, 9).save(job.path("views", "touch-up-0.png"))
        Image.new("L", (64, 64), 255).save(job.path("views", "touch-up-0-mask.png"))
        np.savez_compressed(job.path("shape.npz"), corners=corners, triangles=triangles)
        job.record["picture"] = {"seed": 7, "passed": True, "tries": {"7": {"passed": True, "checks": []}}}
        job.save()
        return job

    def test_every_picture_of_a_finished_job_carries_its_credentials(self):
        folder = tempfile.mkdtemp()
        try:
            job = self.made_job(folder)
            pipeline.finish_stage([job], pipeline.Settings(texture=256, verify=False), report=lambda _: None)
            pictures = credentials.job_pictures(job.folder)
            names = {os.path.relpath(path, job.folder) for path in pictures}
            for expected in ("picture.png", "mask.png", "pictures/7.png", "pictures/7-mask.png", "views/view-0.png",
                             "views/touch-up-0.png", "views/touch-up-0-mask.png", "report/picture.png",
                             "report/try-7.png", "report/outline.png", "report/turntable.png"):
                self.assertIn(expected, names)
            for path, (source_type, model_ids) in pictures.items():
                with self.subTest(picture=os.path.relpath(path, job.folder)):
                    reading = credentials.read(path)
                    self.assertTrue(reading.marked, (reading.state, reading.failures))
                    self.assertEqual(reading.source_type, source_type)
                    self.assertEqual(reading.models, credentials.model_names(model_ids))
            with open(job.path("chair.provenance.json")) as file:
                provenance = json.load(file)
            record = provenance["content_credentials"]
            self.assertEqual(record["certificate"], "test")
            self.assertEqual(record["pictures"], len(pictures))
            self.assertEqual(record["marked"], len(pictures))
            self.assertEqual(record["not_marked"], [])
            # Made before: the job's own pictures (2 + 2 tries + 8 views) signed at the finish.
            self.assertEqual(record["signed_at_finish"], 12)
            with open(job.path("report", "report.html")) as file:
                page = file.read()
            self.assertIn("Made by AI", page)
            self.assertIn("content credentials", page)
            self.assertIn("untrusted", page)
            # The prompt is in the sidecar, never in a picture's manifest.
            for path in pictures:
                with open(path, "rb") as file:
                    self.assertNotIn(b"old looking chair", file.read())
            # The .glb: its texture signed, and a plain note of what made it.
            with open(job.path("chair.glb"), "rb") as file:
                document, binary = glb.read_document(file.read())
            view = document["bufferViews"][document["images"][0]["bufferView"]]
            texture = binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]
            reading = credentials.read(data=bytes(texture), mime_type="image/png")
            self.assertTrue(reading.marked)
            self.assertEqual(reading.source_type, credentials.TRAINED)
            extras = document["asset"]["extras"]
            self.assertIn("AI models", extras["made_with_ai"])
            self.assertEqual(extras["generator"]["name"], "model-maker")
            # A second finish signs nothing again: every picture holds already, the report's written anew.
            pipeline.finish_stage([job], pipeline.Settings(texture=256, verify=False), report=lambda _: None)
            with open(job.path("chair.provenance.json")) as file:
                self.assertEqual(json.load(file)["content_credentials"]["signed_at_finish"], 0)
        finally:
            shutil.rmtree(folder)

    def test_a_picture_whose_mark_broke_is_reported_not_signed_over(self):
        folder = tempfile.mkdtemp()
        try:
            job = self.made_job(folder)
            path = credentials.save_picture(noise_picture(64, 1), job.path("views", "view-1.png"),
                                            credentials.OBJECT_MODELS)
            # A tool wrote a note into it after it was signed: still a picture, its mark no longer holds.
            with open(path, "rb") as file:
                data = with_text_chunk(file.read(), b"Comment\0edited")
            with open(path, "wb") as file:
                file.write(data)
            self.assertFalse(credentials.read(path).marked)
            pipeline.finish_stage([job], pipeline.Settings(texture=256, verify=False), report=lambda _: None)
            with open(job.path("chair.provenance.json")) as file:
                record = json.load(file)["content_credentials"]
            self.assertEqual(record["not_marked"], ["views/view-1.png"])
            self.assertEqual(record["marked"], record["pictures"] - 1)
        finally:
            shutil.rmtree(folder)


def sign_pictures_tool():
    spec = importlib.util.spec_from_file_location("sign_pictures", os.path.join(REPOSITORY, "tools",
                                                                                "sign_pictures.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SignPictures(unittest.TestCase):
    """tools/sign_pictures.py: the AI-made pictures a list names signed, their pixels the same; the rest left alone."""

    def test_the_pictures_named_are_signed_and_the_rest_left_alone(self):
        tool = sign_pictures_tool()
        folder = tempfile.mkdtemp()
        try:
            noise_picture(seed=1).save(os.path.join(folder, "made.webp"), lossless=True)
            os.makedirs(os.path.join(folder, "grids"))
            noise_picture(seed=2).save(os.path.join(folder, "grids", "tries.png"))
            Image.new("RGB", (40, 40), (20, 20, 20)).save(os.path.join(folder, "diagram.png"))
            before = {name: pixels(os.path.join(folder, name)) for name in ("made.webp", "grids/tries.png")}
            with open(os.path.join(folder, "diagram.png"), "rb") as file:
                diagram = file.read()
            listed = os.path.join(folder, "list.toml")
            with open(listed, "w") as file:
                file.write('version = "0.2.0"\n[[picture]]\npath = "made.webp"\nkind = "ai"\nmodels = "object"\n'
                           'why = "a render"\n[[picture]]\npath = "grids/*.png"\nkind = "composite"\n'
                           'models = "picture"\nversion = "0.1.0"\nwhy = "tries with captions"\n')
            self.assertEqual(tool.main([folder, "--list", listed]), 0)
            made = credentials.read(os.path.join(folder, "made.webp"))
            self.assertTrue(made.marked)
            self.assertEqual((made.source_type, made.version), (credentials.TRAINED, "0.2.0"))
            tries = credentials.read(os.path.join(folder, "grids", "tries.png"))
            self.assertEqual((tries.source_type, tries.version, tries.models),
                             (credentials.COMPOSITE, "0.1.0", ["FLUX.1 [schnell]"]))
            for name, seen in before.items():
                self.assertTrue(np.array_equal(pixels(os.path.join(folder, name)), seen), name)
            with open(os.path.join(folder, "diagram.png"), "rb") as file:
                self.assertEqual(file.read(), diagram, "a picture the list does not name is left alone")
            # Run again: what carries a manifest is left as it is.
            with open(os.path.join(folder, "made.webp"), "rb") as file:
                signed = file.read()
            self.assertEqual(tool.main([folder, "--list", listed]), 0)
            with open(os.path.join(folder, "made.webp"), "rb") as file:
                self.assertEqual(file.read(), signed)
            # Again with --again: signed anew over model-maker's own manifest, still holding.
            self.assertEqual(tool.main([folder, "--list", listed, "--again"]), 0)
            self.assertTrue(credentials.read(os.path.join(folder, "made.webp")).marked)
        finally:
            shutil.rmtree(folder)

    def test_another_tools_pictures_name_that_tool_and_their_models_as_written(self):
        tool = sign_pictures_tool()
        folder = tempfile.mkdtemp()
        try:
            noise_picture(seed=3).save(os.path.join(folder, "backdrop.png"))
            noise_picture(seed=4).save(os.path.join(folder, "side-by-side.png"))
            before = pixels(os.path.join(folder, "backdrop.png"))
            listed = os.path.join(folder, "list.toml")
            with open(listed, "w") as file:
                file.write('generator = "Another art tool"\nversion = "1.2.3"\n'
                           '[[picture]]\npath = "backdrop.png"\nkind = "ai"\nmodels = ["Some Model [fast]"]\n'
                           'why = "a backdrop"\n[[picture]]\npath = "side-by-side.png"\nkind = "composite"\n'
                           'models = ["Some Model [fast]"]\nwhy = "a painting beside renders"\n')
            self.assertEqual(tool.main([folder, "--list", listed]), 0)
            backdrop = credentials.read(os.path.join(folder, "backdrop.png"))
            self.assertTrue(backdrop.marked)
            self.assertEqual((backdrop.source_type, backdrop.generator, backdrop.version, backdrop.models),
                             (credentials.TRAINED, "Another art tool", "1.2.3", ["Some Model [fast]"]))
            side = credentials.read(os.path.join(folder, "side-by-side.png"))
            self.assertEqual((side.source_type, side.generator), (credentials.COMPOSITE, "Another art tool"))
            said = json.dumps([side.manifest["claim_generator_info"], side.manifest["assertions"]])
            self.assertNotIn("model-maker", said, "nothing says model-maker made it (only the test certificate's name)")
            self.assertTrue(np.array_equal(pixels(os.path.join(folder, "backdrop.png")), before))
            for wrong in ('generator = "model-maker"\nversion = "1"\n[[picture]]\npath = "x.png"\nkind = "ai"\n'
                          'models = ["A"]\nwhy = "w"\n',
                          'generator = "Another art tool"\nversion = "1"\n[[picture]]\npath = "x.png"\nkind = "ai"\n'
                          'models = "picture"\nwhy = "w"\n'):
                with open(listed, "w") as file:
                    file.write(wrong)
                with self.assertRaises(tool.Refused):
                    tool.entries_of(listed)
        finally:
            shutil.rmtree(folder)

    def test_a_list_naming_a_picture_not_there_is_refused(self):
        tool = sign_pictures_tool()
        folder = tempfile.mkdtemp()
        try:
            noise_picture().save(os.path.join(folder, "a.png"))
            listed = os.path.join(folder, "list.toml")
            with open(listed, "w") as file:
                file.write('version = "0.2.0"\n[[picture]]\npath = "b.png"\nkind = "ai"\nwhy = "a render"\n')
            self.assertEqual(tool.main([folder, "--list", listed]), 1)
            self.assertFalse(credentials.read(os.path.join(folder, "a.png")).found)
        finally:
            shutil.rmtree(folder)

    def test_the_readme_pictures_list_names_every_picture_it_means(self):
        tool = sign_pictures_tool()
        folder = os.path.join(REPOSITORY, "docs", "images")
        chosen = tool.chosen_for(tool.pictures_in(folder), tool.entries_of(os.path.join(folder, "made-by-ai.toml")))
        self.assertNotIn("qualities/several-things.webp", chosen, "a terminal's text: no model made it")
        self.assertEqual(len(chosen), len(tool.pictures_in(folder)) - 1)


class MissingPackages(unittest.TestCase):
    def test_an_update_without_setup_is_told_to_run_setup_not_shown_a_traceback(self):
        from model_maker import __main__ as main

        real = importlib.util.find_spec
        with mock.patch.object(importlib.util, "find_spec",
                               side_effect=lambda name, *rest: None if name == "c2pa" else real(name, *rest)):
            self.assertEqual(credentials.missing_packages(), ["c2pa-python"])
            said = io.StringIO()
            with contextlib.redirect_stderr(said), self.assertRaises(SystemExit) as stopped:
                main.use_signing(argparse.Namespace())
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("c2pa-python not installed", said.getvalue())
        self.assertIn("./setup.sh", said.getvalue())

    def test_nothing_is_missing_where_setup_ran(self):
        self.assertEqual(credentials.missing_packages(), [])


if __name__ == "__main__":
    unittest.main()
