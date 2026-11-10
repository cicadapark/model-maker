import importlib.util
import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from model_maker import ledger, object_checks as O, pipeline, render

from . import shapes
from .test_finish import painted_half

REPOSITORY = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def summary_tool():
    spec = importlib.util.spec_from_file_location("summary", os.path.join(REPOSITORY, "tools", "trial", "summary.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OwnLedger(unittest.TestCase):
    def setUp(self):
        self.kept = os.environ["MODEL_MAKER_LEDGER"]
        self.folder = tempfile.mkdtemp()
        os.environ["MODEL_MAKER_LEDGER"] = os.path.join(self.folder, "metrics")

    def tearDown(self):
        os.environ["MODEL_MAKER_LEDGER"] = self.kept
        shutil.rmtree(self.folder)

    def chair_job(self, name="chair"):
        job = pipeline.Job(os.path.join(self.folder, "batch-a", name), "an old looking chair")
        corners, triangles = shapes.chair()
        corners = O.set_down(corners, job["size_m"])
        mask = render.silhouette(corners, triangles, 20, 512)
        Image.fromarray(np.where(mask[..., None], 120, 210).astype(np.uint8).repeat(3, -1)).save(
            job.path("picture.png"))
        Image.fromarray(mask.astype(np.uint8) * 255).save(job.path("mask.png"))
        np.savez_compressed(job.path("shape.npz"), corners=corners, triangles=triangles)
        cloud = painted_half(corners, triangles)
        np.savez_compressed(job.path("painted.npz"), colours=np.full((len(corners), 3), 0.5), known=None,
                            cloud_points=cloud[0], cloud_normals=cloud[1], cloud_colours=cloud[2])
        return job


class Lines(OwnLedger):
    def test_a_finished_job_writes_its_stage_and_its_end_with_the_machine_and_settings(self):
        job = self.chair_job()
        pipeline.finish_stage([job], pipeline.Settings(texture=256, verify=False), report=lambda _: None)
        lines = ledger.read()
        self.assertEqual([line["event"] for line in lines], ["stage", "job"])
        stage, end = lines
        self.assertEqual(stage["stage"], "finish")
        self.assertGreater(stage["seconds"], 0)
        self.assertEqual(end["job"], job["id"])
        self.assertEqual(end["batch"], "batch-a")
        self.assertFalse(end["weapon"])
        self.assertIn("cpu_threads", end["machine"])
        self.assertEqual(end["settings"]["texture"], 256)
        # Half the chair was painted: the end says it failed and on which check, with its value and limit.
        self.assertFalse(end["passed"])
        self.assertIn("paint covers the surface", end["failed"])
        paint = next(c for c in end["checks"] if c["name"] == "paint covers the surface")
        self.assertLess(paint["value"], paint["limit"])

    def test_a_refinish_keeps_the_settings_the_job_was_made_with(self):
        # A refinish wrote the defaults (20 steps) on a job painted at 30.
        job = self.chair_job()
        job.record["settings"] = ledger.settings_of(pipeline.Settings(paint_steps=30, lead="isolated"))
        job.save()
        pipeline.finish_stage([job], pipeline.Settings(texture=256, verify=False), report=lambda _: None)
        end = ledger.read()[-1]
        self.assertEqual((end["settings"]["paint_steps"], end["settings"]["lead"]), (30, "isolated"))
        self.assertEqual(end["settings"]["texture"], 256)

    def test_a_stage_that_breaks_writes_an_aborted_line_and_the_error_goes_on(self):
        job = self.chair_job()
        os.remove(job.path("shape.npz"))
        with self.assertRaises(FileNotFoundError):
            pipeline.finish_stage([job], pipeline.Settings(texture=256, verify=False), report=lambda _: None)
        line, = ledger.read()
        self.assertEqual((line["event"], line["stage"]), ("aborted", "finish"))
        self.assertIn("shape.npz", line["error"])

    def test_a_thing_not_made_yet_is_written_as_not_made(self):
        job = pipeline.Job(os.path.join(self.folder, "batch-a", "bow"), "a wooden longbow")
        pipeline.run([job], pipeline.Settings(verify=False), report=lambda _: None, stages=())
        line, = ledger.read()
        self.assertEqual(line["event"], "not made")
        self.assertTrue(line["weapon"])


class Progress(OwnLedger):
    def test_a_refinished_job_counts_once_as_it_last_ended(self):
        job = self.chair_job()
        settings = pipeline.Settings(texture=256, verify=False)
        pipeline.finish_stage([job], settings, report=lambda _: None)
        np.savez_compressed(job.path("painted.npz"), **{**dict(np.load(job.path("painted.npz"), allow_pickle=True)),
                                                         "cloud_points": np.zeros((0, 3))})
        pipeline.finish_stage([job], settings, report=lambda _: None)
        ends = summary_tool().latest_per_job(ledger.read())
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0]["at"], ledger.read()[-1]["at"])

    def test_the_batch_row_counts_passes_weapons_and_card_minutes_from_the_lines(self):
        tool = summary_tool()
        machine = {"card": "a card", "card_mb": 16000, "driver": "1", "cpu_threads": 8}

        def end(job, passed, weapon, seconds):
            return {"event": "job", "job": job, "batch": "b", "prompt": job, "weapon": weapon, "passed": passed,
                    "failed": [] if passed else ["base level"], "seconds": seconds, "version": "v1",
                    "machine": machine, "first_tries": {"picture_rounds": 1, "shape_first_passed": passed}}

        ends = [end("a", True, False, {"pictures": 60, "shape": 60, "paint": 180, "finish": 60}),
                end("b", False, False, {"pictures": 60, "shape": 120, "paint": 180, "finish": 60}),
                end("c", True, True, {"pictures": 60, "shape": 60, "paint": 180, "finish": 60}),
                {"event": "not made", "job": "d", "batch": "b", "weapon": True, "version": "v1", "machine": machine}]
        row = tool.batch_row(ends)
        self.assertEqual((row["asked"], row["made"], row["passed"]), (4, 3, 2))
        self.assertEqual((row["objects"], row["weapons"]), ("1 of 2", "1 of 1"))
        self.assertEqual(row["median_min"], 6.0)
        # Card minutes (pictures, shape, paint) of all three made, over the two that passed.
        self.assertEqual(row["card_min_per_pass"], (5 + 6 + 5) / 2)
        self.assertEqual(row["shape_first_seed"], 2)
        self.assertEqual(row["faults"], [("base level", 1)])

    def test_the_whole_time_of_a_job_counts_its_loading_and_a_refinish_keeps_it(self):
        tool = summary_tool()
        machine = {"card": "a card"}
        stages = {"pictures": 60, "shape": 60, "paint": 180, "finish": 60}

        def end(job, run, at):
            line = {"event": "job", "job": job, "batch": "b", "prompt": job, "weapon": False, "passed": True,
                    "failed": [], "seconds": stages, "version": "v1", "machine": machine, "at": at,
                    "first_tries": {"picture_rounds": 1, "shape_first_passed": True}}
            return {**line, "run": run} if run else line

        cold = {"warm": False, "jobs": 1, "whole_seconds": 480.0, "load_seconds": 110.0}
        warm = {"warm": True, "jobs": 1, "whole_seconds": 372.0, "load_seconds": 0.0}
        lines = [end("a", cold, "1"), end("b", warm, "2"), end("c", warm, "3"), end("a", None, "4")]
        ends = tool.latest_per_job(lines)
        refinished = next(e for e in ends if e["job"] == "a")
        self.assertEqual((refinished["at"], refinished["run"]), ("4", cold), "the refinish keeps the job's run")
        row = tool.batch_row(ends)
        self.assertEqual(row["median_min"], 6.0, "the stages alone")
        self.assertEqual(row["median_whole_min"], 6.2, "with the loading: 8, 6.2 and 6.2 minutes")
        self.assertEqual(row["warm"], "2 of 3")


if __name__ == "__main__":
    unittest.main()
