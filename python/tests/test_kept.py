"""The models a run keeps (kept.py), a pipeline started by another program (parent.py), the random state every job
starts from, and how a run is written to the ledger."""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from model_maker import kept as K, ledger, parent

from . import shapes
from .test_command import made_up_job
from .test_ledger import OwnLedger

HAS_TORCH = importlib.util.find_spec("torch") is not None
PYTHON_FOLDER = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


class FakeModel:
    def __init__(self):
        self.closed, self.moved = False, []
        self.model = self

    def to(self, device):
        self.moved.append(device)
        return self

    def close(self):
        self.closed = True


class Keeping(unittest.TestCase):
    """`Kept`'s own bookkeeping with stand-in models (`_model` directly); `RealAccessors` below goes through the
    accessors the pipeline calls."""

    def test_the_bookkeeping_loads_once_and_closes_a_stages_models_at_its_end(self):
        kept = K.Kept().counting()
        loads = []
        cutter = kept._model("cutter", lambda: ["folder"], lambda f: loads.append(f) or FakeModel())
        painter = kept._model("picture painter", lambda: ["folder"], lambda f: loads.append(f) or FakeModel())
        shape = kept._model("shape", lambda: ["folder"], lambda f: loads.append(f) or FakeModel())
        self.assertTrue(kept.loaded_any)
        kept.counting()
        self.assertIs(kept._model("cutter", lambda: ["folder"], lambda f: loads.append(f) or FakeModel()),
                      cutter, "loaded once")
        self.assertFalse(kept.loaded_any)
        self.assertEqual(len(loads), 3)
        self.assertEqual(sorted(kept.stage_models("pictures")), ["cutter", "picture painter"])
        kept.done_with("pictures")
        self.assertEqual(list(kept.models), ["shape"], "only the pictures stage's models closed")
        self.assertTrue(painter.closed and cutter.closed)
        self.assertFalse(shape.closed)

    def test_a_model_keyed_by_its_setting_belongs_to_its_stage(self):
        kept = K.Kept()
        kept._model("inpainter 0.6", lambda: ["folder"], lambda f: FakeModel())
        self.assertEqual(kept.stage_models("paint"), ["inpainter 0.6"])
        kept.done_with("paint")
        self.assertEqual(kept.models, {})

    def test_a_model_is_checked_only_when_it_is_loaded(self):
        kept, checked = K.Kept(), []
        for _ in range(3):
            kept._model("shape", lambda: checked.append(1) or ["folder"], lambda f: FakeModel())
        self.assertEqual(checked, [1])


class StandIn:
    """A model class stand-in: built from its folder(s), with a `.model` that records where it is moved."""

    made = []

    def __init__(self, *folders, **settings):
        self.folders, self.settings = folders, settings
        self.model, self.device, self.closed = FakeModel(), "cuda", False
        StandIn.made.append(self)

    def close(self):
        self.closed = True


@unittest.skipUnless(HAS_TORCH, "needs torch (installed by setup.sh)")
class RealAccessors(unittest.TestCase):
    """The accessors the pipeline calls (`picture_painter`, `cutter`, `state_judge`, `shape_models`, `view_painter`,
    `inpainter`) and `done_with`, with each model class replaced by a stand-in (no weights, no card) and the
    registry's check recorded."""

    def setUp(self):
        from unittest import mock

        from model_maker import cutout, paint, painter, pipeline, registry, shape_net, state_judge, touch_up

        StandIn.made = []
        self.checks = []
        self.folder = tempfile.mkdtemp()
        patches = [mock.patch.object(painter, "Painter", StandIn), mock.patch.object(cutout, "Cutter", StandIn),
                   mock.patch.object(state_judge, "StateJudge", StandIn),
                   mock.patch.object(paint, "ViewPainter", StandIn), mock.patch.object(touch_up, "Inpainter", StandIn),
                   mock.patch.object(shape_net, "kept_models", lambda dtype: shape_net._KEPT.setdefault("k", 1)),
                   mock.patch.object(registry, "verify_folder",
                                     lambda model_id, folder, checked=None: self.checks.append((folder, checked)))]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(shutil.rmtree, self.folder)
        self.settings = lambda: pipeline.Settings(models=self.folder, sources=self.folder, verify=True)

    def every_stage(self, kept):
        settings = self.settings()
        got = [kept.picture_painter(settings), kept.cutter(settings), kept.state_judge(settings)]
        kept.done_with("pictures")
        got.append(kept.shape_models(settings))
        kept.done_with("shape")
        got += [kept.view_painter(settings), kept.inpainter(settings)]
        kept.done_with("paint")
        return got

    def test_a_cold_run_closes_each_stages_models_at_its_end(self):
        from model_maker import shape_net

        kept = K.Kept()
        self.every_stage(kept)
        self.assertEqual(kept.models, {})
        self.assertEqual(shape_net._KEPT, {}, "the shape model out of memory before the view painter")
        painter, judge = StandIn.made[0], StandIn.made[2]
        self.assertTrue(painter.closed and judge.closed)
        self.assertTrue(all(checked is kept.checked for _, checked in self.checks), "one checked set for the run")

    def test_the_inpainters_check_and_load_are_kept_out_of_the_paints_time(self):
        from model_maker import pipeline

        kept = K.Kept().counting()
        lazy = pipeline.LazyInpainter(kept, self.settings(), lambda _: None)
        StandIn.__call__ = lambda self, *args: "painted"
        try:
            self.assertEqual(lazy("picture"), "painted")
        finally:
            del StandIn.__call__
        self.assertGreaterEqual(lazy.load_seconds, kept.loaded_seconds + kept.checked_seconds - 1e-9)
        self.assertGreater(len(self.checks), 0)


class Parent(unittest.TestCase):
    """`die_with_parent`, in real processes: a stand-in for the starting program starts a child that sets the
    signal."""

    CHILD = ("import sys, time; sys.path.insert(0, {path!r}); from model_maker import parent; "
             "print('set' if parent.die_with_parent() else 'none', flush=True); time.sleep(60)")

    def child_of_a_parent(self, worker_pid):
        """A parent process (the starter's stand-in) that starts the child and prints the child's id; the child
        prints whether it set the signal. `worker_pid`: what the child is told (None: the parent's own id)."""
        child = self.CHILD.format(path=PYTHON_FOLDER)
        starter = ("import os, subprocess, sys, time; env = dict(os.environ); "
                   f"env['MODEL_MAKER_WORKER_PID'] = {worker_pid!r} or str(os.getpid()); "
                   f"c = subprocess.Popen([sys.executable, '-c', {child!r}], env=env, stdout=subprocess.PIPE, "
                   "text=True); print(c.pid, c.stdout.readline().strip(), flush=True); c.wait(); "
                   "print('ended', c.returncode, flush=True); time.sleep(60)")
        return subprocess.Popen([sys.executable, "-c", starter], stdout=subprocess.PIPE, text=True)

    def gone_soon(self, pid):
        until = time.time() + 10
        while time.time() < until:
            try:
                with open(f"/proc/{pid}/stat") as file:
                    if file.read().rsplit(") ", 1)[1].startswith("Z"):
                        return True
            except OSError:
                return True
            time.sleep(0.05)
        return False

    def test_a_pipeline_dies_with_its_starter(self):
        starter = self.child_of_a_parent(None)
        try:
            pid, said = starter.stdout.readline().split()
            self.assertEqual(said, "set")
            self.assertFalse(self.gone_soon_quick(int(pid)))
            starter.kill()
            starter.wait()
            self.assertTrue(self.gone_soon(int(pid)), "killed with its starter")
        finally:
            starter.kill()
            starter.wait()

    def test_a_pipeline_whose_parent_is_not_its_starter_ends_at_once(self):
        # Told a starter that is not its parent: as when the starter ended and it was given to init or a subreaper.
        starter = self.child_of_a_parent("1")
        try:
            self.assertEqual(len(starter.stdout.readline().split()), 1, "it said nothing")
            self.assertEqual(starter.stdout.readline().split(), ["ended", "1"])
        finally:
            starter.kill()
            starter.wait()

    def test_run_by_hand_it_sets_nothing(self):
        env = {k: v for k, v in os.environ.items() if k != parent.WORKER_PID}
        done = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {PYTHON_FOLDER!r}); "
                               "from model_maker import parent; print(parent.die_with_parent())"],
                              capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(done.stdout.strip(), "False")

    def gone_soon_quick(self, pid):
        time.sleep(0.2)
        return not os.path.exists(f"/proc/{pid}")


@unittest.skipUnless(HAS_TORCH, "needs torch (installed by setup.sh)")
class RandomState(unittest.TestCase):
    DRAWS = ("import random, numpy, torch; "
             "print([round(random.random(), 12), float(numpy.random.rand()), torch.rand(2).tolist()])")

    def test_a_job_after_another_in_one_process_starts_from_the_random_state_a_new_process_does(self):
        setup = f"import sys; sys.path.insert(0, {PYTHON_FOLDER!r}); from model_maker import pipeline; "
        # Two new processes, as two cold jobs: a new process's own torch state is random, so they draw apart...
        cold = [subprocess.run([sys.executable, "-c", self.DRAWS], capture_output=True, text=True, timeout=120)
                .stdout.strip() for _ in range(2)]
        self.assertNotEqual(cold[0], cold[1], "a new process's own random state is not fixed")
        # ...but each job starts from the same state, cold (a new process) or warm (after another job's draws).
        cold_job = subprocess.run([sys.executable, "-c", setup + "pipeline.same_random_state(); " + self.DRAWS],
                                  capture_output=True, text=True, timeout=120).stdout.strip()
        warm_job = subprocess.run([sys.executable, "-c", setup + self.DRAWS + "; pipeline.same_random_state(); "
                                   + self.DRAWS], capture_output=True, text=True, timeout=120).stdout.split("\n")[1]
        self.assertEqual(warm_job.strip(), cold_job)


class RunInTheLedger(OwnLedger):
    def tearDown(self):
        ledger._RUN = None
        super().tearDown()

    def test_a_jobs_line_says_its_run_warm_or_cold_with_its_loads_and_whole_time(self):
        from model_maker import pipeline

        ledger.run_began(1, warm=False)
        ledger.loaded("pictures", 40.0)
        ledger.loaded("paint", 30.0)
        job = self.chair_job()
        pipeline.finish_stage([job], pipeline.Settings(texture=256, verify=False), report=lambda _: None)
        ledger.run_began(1, warm=True)
        again = self.chair_job("chair-again")
        pipeline.finish_stage([again], pipeline.Settings(texture=256, verify=False), report=lambda _: None)
        lines = ledger.read()
        cold, warm_end = [line for line in lines if line["event"] == "job"]
        self.assertEqual((cold["run"]["warm"], cold["run"]["load_seconds"]), (False, 70.0))
        self.assertEqual((warm_end["run"]["warm"], warm_end["run"]["load_seconds"]), (True, 0.0))
        self.assertGreater(warm_end["run"]["whole_seconds"], 0)
        self.assertEqual([line["stage"] for line in lines if line["event"] == "load"], ["pictures", "paint"])

    def test_a_stages_time_leaves_out_a_model_loading_inside_it(self):
        job = self.chair_job()
        with ledger.stage(job, "paint", None) as found:
            time.sleep(0.5)
            found[ledger.NOT_THE_STAGES] = 0.5
        line, = ledger.read()
        self.assertLess(line["seconds"], 0.3)
        self.assertNotIn(ledger.NOT_THE_STAGES, line)


class RunsOfTheirOwn(OwnLedger):
    def tearDown(self):
        ledger._RUN = None
        super().tearDown()

    def test_a_made_job_given_kept_models_is_warm_and_a_refused_prompt_has_its_own_run(self):
        from model_maker import __main__ as command, pipeline

        self.chair_job(os.path.basename(pipeline.job_folder("batch-a", "an old looking chair")))
        out = os.path.join(self.folder, "batch-a")
        settings = pipeline.Settings(texture=256, verify=False)
        kept = K.Kept()
        code, job = command.made("an old looking chair", out, None, None, None, settings, ["finish"], kept=kept)
        self.assertIsNotNone(job)
        self.assertFalse(ledger.run_record()["warm"], "no model kept yet: not warm")
        ledger.loaded("pictures", 99.0)  # as if the job before had loaded a model at its end
        kept.models["picture painter"] = FakeModel()
        code, job = command.made("a wooden longbow", out, None, None, None, settings, ["finish"], kept=kept)
        self.assertEqual((code, job), (2, None), "not made")
        run = ledger.run_record()
        self.assertTrue(run["warm"], "models kept: warm")
        self.assertEqual(run["load_seconds"], 0.0, "its own run, not the job before's")

    def test_a_batchs_jobs_share_its_whole_time_evenly(self):
        from model_maker import __main__ as command

        from .test_ledger import summary_tool

        out = os.path.join(self.folder, "batch-b")
        made_up_job(out, "an old iron sword", *shapes.sword())
        made_up_job(out, "a double-bitted battle axe", *shapes.double_axe())
        prompts = os.path.join(self.folder, "prompts.txt")
        with open(prompts, "w") as file:
            file.write("an old iron sword\na double-bitted battle axe\n")
        command.main(["batch", prompts, "--out", out, "--stages", "finish", "--texture", "256", "--no-verify"])
        lines = ledger.read()
        ends = [line for line in lines if line["event"] == "job"]
        run, = [line for line in lines if line["event"] == "run"]
        self.assertEqual([end["run"]["whole_seconds"] for end in ends], [None, None],
                         "not known at a job's end, so not written there")
        self.assertEqual(sorted(run["jobs"]), sorted(end["job"] for end in ends))
        tool = summary_tool()
        rows = tool.latest_per_job(lines)
        share = round(run["whole_seconds"] / 2, 1)
        self.assertEqual([row["run"]["whole_seconds"] for row in rows], [share, share])
        self.assertAlmostEqual(tool.batch_row(rows)["median_whole_min"], share / 60)


if __name__ == "__main__":
    unittest.main()
