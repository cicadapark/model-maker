"""The models a run has loaded, and how long it waited for them.

A run (`make`, `batch`) keeps each stage's models for that stage only and closes them at its end (`done_with`): the
computer's memory is free for the next stage's. Each model waits in the computer's memory and goes to the card only for
its own step (FLUX.1's parts by its offload, BiRefNet and SigLIP 2 here, TripoSG's by `shape_net.on_the_card`, the view
painter's by its own moves or `paint.BlockOffload`, the inpainter by its own moves). A program that makes many objects
in one process (a warm process) can keep them between jobs instead, by giving `pipeline.run` its own `Kept` whose `done_with` keeps them.

Each model's files are checked against the registry when it is loaded (`Settings.verified`), outside its load time
(`checked_seconds`); a file checked once is not checked again while the `Kept` lives, and a model loaded once is not
read from disk again.
"""

import os
import time


class Kept:
    def __init__(self):
        self.models = {}
        # The files checked against the registry (shared by every job's Settings).
        self.checked = set()
        # The seconds spent loading and checking files since `counting()`, and whether anything was loaded.
        self.loaded_seconds = 0.0
        self.checked_seconds = 0.0
        self.loaded_any = False

    def counting(self):
        self.loaded_seconds, self.checked_seconds, self.loaded_any = 0.0, 0.0, False
        return self

    def _model(self, name, folders, load):
        """The model `name`, loaded with `load(folders())` the first time (its files checked by `folders`)."""
        if name not in self.models:
            checking = time.time()
            found = folders()
            self.checked_seconds += time.time() - checking
            started = time.time()
            self.models[name] = load(found)
            self.loaded_seconds += time.time() - started
            self.loaded_any = True
        return self.models[name]

    # The pictures stage: FLUX.1 [schnell], BiRefNet and SigLIP 2 (each try against its prompt; a lid's state).

    def picture_painter(self, settings):
        from . import painter

        return self._model("picture painter",
                           lambda: settings.verified("flux1-schnell", "flux1-schnell", checked=self.checked),
                           lambda folders: painter.Painter(folders[0]))

    def cutter(self, settings):
        from . import cutout

        return self._model("cutter", lambda: settings.verified("birefnet", "birefnet", checked=self.checked),
                           lambda folders: cutout.Cutter(folders[0]))

    def state_judge(self, settings):
        from . import state_judge
        from .pipeline import STATE_MODEL

        return self._model("state judge",
                           lambda: settings.verified(STATE_MODEL, STATE_MODEL, checked=self.checked),
                           lambda folders: state_judge.StateJudge(folders[0]))

    # The shape stage: TripoSG (kept by shape_net itself, in the computer's memory; on the card a model at a time).

    def shape_models(self, settings):
        def load(folders):
            import torch

            os.environ["TRIPOSG_WEIGHTS"] = folders[0]
            from . import shape_net

            shape_net.kept_models(torch.float16)
            return shape_net

        return self._model("shape", lambda: settings.verified("triposg", "triposg", checked=self.checked), load)

    # The paint stage: MV-Adapter's view painter on SDXL, and SDXL's inpainter for the touch-up (loaded when a job
    # first needs it).

    def view_painter(self, settings):
        def folders():
            sdxl = settings.verified("sdxl-base-1.0", "sdxl-base-1.0", checked=self.checked)[0]
            vae = settings.verified("sdxl-vae-fp16-fix", "sdxl-vae", checked=self.checked)[0]
            adapter = settings.verified("mv-adapter-ig2mv-sdxl", "mv-adapter", checked=self.checked)[0]
            return sdxl, vae, adapter

        def load(found):
            from . import paint

            os.environ.setdefault("MV_ADAPTER_SRC", os.path.join(settings.sources, "mv-adapter"))
            return paint.ViewPainter(*found)

        return self._model("view painter", folders, load)

    def inpainter(self, settings):
        def folders():
            return (settings.verified("sdxl-base-1.0", "sdxl-base-1.0", checked=self.checked)[0],
                    settings.verified("sdxl-vae-fp16-fix", "sdxl-vae", checked=self.checked)[0])

        def load(found):
            from . import touch_up

            return touch_up.Inpainter(*found, strength=settings.touch_up_strength)

        # Keyed by its strength, the one setting it is built with: a Kept that lives across runs with another
        # strength builds its own.
        return self._model(f"inpainter {settings.touch_up_strength}", folders, load)

    # A stage's end.

    STAGE_MODELS = {"pictures": ("picture painter", "cutter", "state judge"), "shape": ("shape",),
                    "paint": ("view painter", "inpainter")}

    def stage_models(self, stage):
        """The names of the loaded models of `stage` ("pictures", "shape" or "paint")."""
        return [name for name in self.models if any(name == n or name.startswith(n + " ")
                                                    for n in self.STAGE_MODELS[stage])]

    def done_with(self, stage):
        """The stage's models closed, so the next stage has the card and the computer's memory to itself."""
        for name in self.stage_models(stage):
            close(self.models.pop(name))
        # Handed back to the system, not only freed: measured, the shape model closed left most of its memory held by
        # the process until trimmed (memory.give_back).
        from . import memory

        memory.give_back()

    def close(self):
        from . import memory

        for name in list(self.models):
            close(self.models.pop(name))
        memory.give_back()


def close(model):
    if hasattr(model, "close"):
        model.close()
    elif getattr(model, "__name__", "") == "model_maker.shape_net":
        # The shape model out of the computer's memory before the view painter comes in.
        model._KEPT.clear()


def empty_card_cache():
    import sys

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
