import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from model_maker import memory, pipeline


class Memory(unittest.TestCase):
    def test_the_process_and_the_machine_are_measured(self):
        self.assertGreater(memory.in_use_mb(), 0)
        self.assertGreaterEqual(memory.peak_mb(), memory.in_use_mb() - 1)
        self.assertGreater(memory.free_mb(), 0)

    def test_memory_freed_is_handed_back(self):
        held = memory.in_use_mb()
        block = np.ones((256, 1024, 1024), dtype=np.uint8)  # 256 MB, touched
        self.assertGreater(memory.in_use_mb(), held + 200)
        del block
        memory.give_back()
        self.assertLess(memory.in_use_mb(), held + 100)

    def test_a_stage_needing_more_than_is_free_is_refused_and_says_why(self):
        with mock.patch.object(memory, "free_mb", return_value=2000), \
                mock.patch.object(memory, "in_use_mb", return_value=1000):
            room, why = memory.room_for("paint", 4000)
            self.assertFalse(room)
            self.assertIn("paint", why)
            self.assertIn("3.9 GB", why)
            self.assertTrue(memory.room_for("paint", 2500)[0])
        with mock.patch.dict(os.environ, {memory.CHECK_VARIABLE: "0"}):
            self.assertTrue(memory.room_for("paint", 10 ** 9)[0])

    def test_the_needs_measured_are_kept_in_the_batchs_folder_the_larger_kept(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(memory.needs(folder), memory.KNOWN_NEEDS_MB)
            memory.record_need(folder, "paint", 999999)
            memory.record_need(folder, "paint", 5)
            self.assertEqual(memory.needs(folder)["paint"], 999999)
            with open(os.path.join(folder, memory.NEEDS_FILE)) as file:
                self.assertEqual(json.load(file), {"paint": 999999})


class Resume(unittest.TestCase):
    def job(self, folder, prompt="a clay jug"):
        return pipeline.Job(os.path.join(folder, "a-clay-jug-1234"), prompt)

    def test_a_stage_is_wanted_until_its_last_file_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            job = self.job(folder)
            self.assertTrue(pipeline.stage_wanted(job, "pictures"))
            pipeline.written(job.path("picture.png"), lambda path: open(path, "wb").close())
            self.assertFalse(pipeline.stage_wanted(job, "pictures"))
            # Cut short: only the partial file is there.
            open(job.path("shape.partial.npz"), "wb").close()
            self.assertTrue(pipeline.stage_wanted(job, "shape"))

    def test_the_finish_is_wanted_again_after_a_new_paint(self):
        with tempfile.TemporaryDirectory() as folder:
            job = self.job(folder)
            np.savez(job.path("painted.npz"), colours=np.zeros(3))
            self.assertTrue(pipeline.stage_wanted(job, "finish"))
            open(job.path(f"{job['slug']}.glb"), "wb").close()
            job.record["finished_from"] = pipeline.made_stamp(job)
            self.assertFalse(pipeline.stage_wanted(job, "finish"))
            os.utime(job.path("painted.npz"), ns=(1, 1))
            self.assertTrue(pipeline.stage_wanted(job, "finish"))

    def ran(self, folder, needs=None, room=True, report=lambda *a: None):
        calls = []

        def stage(name):
            def done(jobs, settings, report, *rest):
                calls.append((name, [os.path.basename(j.folder) for j in jobs]))
            return done

        jobs = [self.job(folder)]
        with mock.patch.object(pipeline, "pictures_stage", stage("pictures")), \
                mock.patch.object(pipeline, "shape_stage", stage("shape")), \
                mock.patch.object(pipeline, "paint_stage", stage("paint")), \
                mock.patch.object(pipeline, "finish_stage", stage("finish")), \
                mock.patch.object(pipeline, "model_present", return_value=True), \
                mock.patch.object(memory, "room_for", return_value=(room, "the paint stage needs more")):
            pipeline.run(jobs, pipeline.Settings(verify=False), report=report, keep_going=True)
        return calls

    def test_a_batch_started_again_skips_what_is_done(self):
        with tempfile.TemporaryDirectory() as folder:
            job = self.job(folder)
            for name in ("picture.png", "shape.npz"):
                open(job.path(name), "wb").close()
            calls = self.ran(folder)
            self.assertEqual([c[0] for c in calls], ["paint", "finish"])

    def test_a_stage_this_batch_measured_is_not_started_when_the_memory_cannot_hold_it(self):
        with tempfile.TemporaryDirectory() as folder:
            memory.record_need(folder, "pictures", 22400)
            with self.assertRaises(pipeline.NotEnoughMemory) as raised:
                self.ran(folder, room=False)
            self.assertIn("as this batch measured it", str(raised.exception))
            self.assertIn("run the same command again", str(raised.exception))

    def test_against_the_trials_needs_alone_a_short_memory_only_warns(self):
        # A 16 GB or a busy 24 GB PC: the built-in needs are peaks with droppable pages counted (memory.py).
        with tempfile.TemporaryDirectory() as folder:
            said = []
            calls = self.ran(folder, room=False, report=said.append)
            self.assertEqual([c[0] for c in calls], ["pictures", "shape", "paint", "finish"])
            self.assertTrue(any("going on" in line and "run the same command again" in line for line in said))


if __name__ == "__main__":
    unittest.main()
