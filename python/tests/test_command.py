import contextlib
import csv
import io
import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from model_maker import __main__ as command, object_checks as O, pipeline, render, words

from . import shapes
from .test_finish import painted_half


def made_up_job(folder, prompt, corners, triangles):
    """A job whose card stages are done: its picture, cut-out, shape and paint written as they would be."""
    job = pipeline.Job(pipeline.job_folder(folder, prompt), prompt)
    corners = O.set_down(corners, job["size_m"])
    mask = render.silhouette(corners, triangles, 0, 512)
    Image.fromarray(np.where(mask[..., None], 110, 205).astype(np.uint8).repeat(3, -1)).save(job.path("picture.png"))
    Image.fromarray(mask.astype(np.uint8) * 255).save(job.path("mask.png"))
    np.savez_compressed(job.path("shape.npz"), corners=corners, triangles=triangles)
    cloud = painted_half(corners, triangles)
    np.savez_compressed(job.path("painted.npz"), colours=np.full((len(corners), 3), 0.5), known=None,
                        cloud_points=cloud[0], cloud_normals=cloud[1], cloud_colours=cloud[2])
    return job


class Batch(unittest.TestCase):
    def test_a_batch_skips_the_stages_done_and_writes_a_time_per_object(self):
        folder = tempfile.mkdtemp()
        try:
            prompts = os.path.join(folder, "prompts.txt")
            with open(prompts, "w") as file:
                file.write("# a note\nan old iron sword\n\na double-bitted battle axe\n")
            made_up_job(folder, "an old iron sword", *shapes.sword())
            made_up_job(folder, "a double-bitted battle axe", *shapes.double_axe())
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                command.main(["batch", prompts, "--out", folder, "--texture", "256", "--no-verify"])
            with open(os.path.join(folder, "times.csv")) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([r["prompt"] for r in rows], ["an old iron sword", "a double-bitted battle axe"])
            self.assertTrue(all(float(r["finish_s"]) > 0 for r in rows))
            for row in rows:
                slug, job = words.slug_of(row["prompt"]), pipeline.job_folder(folder, row["prompt"])
                self.assertTrue(os.path.exists(os.path.join(job, f"{slug}.glb")))
                self.assertTrue(os.path.exists(os.path.join(job, "report", "marks.png")))
            self.assertIn("of 2 pass every check", said.getvalue())
        finally:
            shutil.rmtree(folder)

    def test_a_bow_is_not_made_yet_and_says_why(self):
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            code = command.main(["make", "a wooden longbow", "--out", tempfile.mkdtemp()])
        self.assertEqual(code, 2)
        self.assertIn("Bows are not made yet", said.getvalue())
        folder = tempfile.mkdtemp()
        try:
            prompts = os.path.join(folder, "prompts.txt")
            with open(prompts, "w") as file:
                file.write("a wooden longbow\n")
            with contextlib.redirect_stdout(io.StringIO()):
                command.main(["batch", prompts, "--out", folder, "--stages", "finish", "--no-verify"])
            with open(os.path.join(folder, "times.csv")) as file:
                rows = list(csv.DictReader(file))
            self.assertTrue(rows[0]["status"].startswith("not made: Bows"))
        finally:
            shutil.rmtree(folder)

    def test_a_job_that_breaks_is_set_aside_and_the_batch_goes_on(self):
        # The first error ended a batch; later prompts got no line, and no times or run
        # line were written.
        from model_maker import ledger

        folder = tempfile.mkdtemp()
        kept = os.environ["MODEL_MAKER_LEDGER"]
        os.environ["MODEL_MAKER_LEDGER"] = os.path.join(folder, "metrics")
        try:
            prompts = os.path.join(folder, "prompts.txt")
            with open(prompts, "w") as file:
                file.write("an old iron sword\na double-bitted battle axe\nan old iron sword\n")
            broken = made_up_job(folder, "an old iron sword", *shapes.sword())
            os.remove(broken.path("shape.npz"))
            made_up_job(folder, "a double-bitted battle axe", *shapes.double_axe())
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                command.main(["batch", prompts, "--out", folder, "--texture", "256", "--stages", "finish"])
            with open(os.path.join(folder, "times.csv")) as file:
                rows = {r["prompt"]: r["status"] for r in csv.DictReader(file)}
            self.assertEqual(len(rows), 2, "the same prompt twice is one job")
            self.assertTrue(rows["an old iron sword"].startswith("aborted at finish: FileNotFoundError"), rows)
            self.assertFalse(rows["a double-bitted battle axe"].startswith("aborted"), rows)
            events = [line["event"] for line in ledger.read()]
            self.assertEqual(sorted(events), ["aborted", "job", "run", "stage"])
        finally:
            os.environ["MODEL_MAKER_LEDGER"] = kept
            shutil.rmtree(folder)

    def test_words_says_how_a_prompt_is_read(self):
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            command.main(["words", "a war hammer"])
        self.assertIn('"weapon_kind": "hafted"', said.getvalue())



class Folders(unittest.TestCase):
    """A job folder was named by the slug alone and reused for another prompt or
    other options; the second `make` returned the first's model as its own."""

    def test_two_prompts_or_two_sizes_never_share_a_folder(self):
        out = "made"
        oak = pipeline.job_folder(out, "a long wooden dining table, dark oak")
        pine = pipeline.job_folder(out, "a long wooden dining table, pale pine")
        self.assertNotEqual(oak, pine)
        self.assertNotEqual(pipeline.job_folder(out, "an old iron sword", 0.9),
                            pipeline.job_folder(out, "an old iron sword", 1.2))
        self.assertNotEqual(pipeline.job_folder(out, "古い椅子"), pipeline.job_folder(out, "古いテーブル"))
        self.assertEqual(pipeline.job_folder(out, "an old  chair."), pipeline.job_folder(out, "an old chair"))
        self.assertTrue(os.path.basename(oak).startswith("long-wooden-dining-table-"))

    def test_a_folder_holding_another_order_is_refused(self):
        folder = os.path.join(tempfile.mkdtemp(), "table")
        try:
            pipeline.Job(folder, "a long wooden dining table, dark oak")
            pipeline.Job(folder, "a long wooden dining table, dark oak")
            with self.assertRaises(pipeline.JobMismatch):
                pipeline.Job(folder, "a long wooden dining table, pale pine")
            with self.assertRaises(pipeline.JobMismatch):
                pipeline.Job(folder, "a long wooden dining table, dark oak", size_m=3.0)
            self.assertEqual(pipeline.Job(folder)["prompt"], "a long wooden dining table, dark oak")
        finally:
            shutil.rmtree(os.path.dirname(folder))

    def test_a_file_written_by_halves_is_never_left_looking_finished(self):
        folder = tempfile.mkdtemp()
        try:
            path = os.path.join(folder, "shape.npz")

            def cut_short(partial):
                with open(partial, "wb") as file:
                    file.write(b"PK")
                raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                pipeline.written(path, cut_short)
            self.assertFalse(os.path.exists(path))
            pipeline.written(path, lambda partial: np.savez_compressed(partial, a=np.arange(3)))
            self.assertEqual(np.load(path)["a"].tolist(), [0, 1, 2])
        finally:
            shutil.rmtree(folder)


class Refused(unittest.TestCase):
    def test_a_refused_prompt_and_a_job_without_its_judge_each_have_a_line(self):
        from model_maker import ledger

        folder = tempfile.mkdtemp()
        kept = os.environ["MODEL_MAKER_LEDGER"]
        os.environ["MODEL_MAKER_LEDGER"] = os.path.join(folder, "metrics")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(command.main(["make", "a red dragon", "--out", folder]), 2)
                self.assertEqual(command.main(["make", " . ", "--out", folder]), 2)
            refused = ledger.read()
            self.assertEqual([line["event"] for line in refused], ["not made", "not made"])
            self.assertEqual(refused[1]["why"], words.EMPTY_PROMPT)
            # Without SigLIP 2 a chest is not made, and says why; nothing raises.
            chest = pipeline.Job(pipeline.job_folder(folder, "a wooden treasure chest"), "a wooden treasure chest")
            settings = pipeline.Settings(models=os.path.join(folder, "no-models"), verify=False)
            pipeline.run([chest], settings, report=lambda _: None, stages=("pictures",))
            self.assertTrue(pipeline.Job(chest.folder)["status"].startswith("not made: needs SigLIP 2"))
            self.assertEqual(ledger.read()[-1]["event"], "not made")
        finally:
            os.environ["MODEL_MAKER_LEDGER"] = kept
            shutil.rmtree(folder)


if __name__ == "__main__":
    unittest.main()
