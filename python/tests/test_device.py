"""The device layer (`device`): the device chosen on each kind of computer (PyTorch stood in for, no card needed), the
float types and settings on each, and that on an NVIDIA card they are what the pipeline wrote before the layer."""

import os
import re
import subprocess
import sys
import types
import unittest
from unittest import mock

from model_maker import device

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.join(HERE, "..", "model_maker")

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class Recorder:
    """A backend's stand-in: says whether it is there, counts the questions and records what it is told to do."""

    def __init__(self, there, log, name):
        self.there, self.log, self.name = there, log, name

    def is_available(self):
        self.log.append(f"{self.name}.is_available")
        return self.there

    def empty_cache(self):
        self.log.append(f"{self.name}.empty_cache")

    def reset_peak_memory_stats(self):
        self.log.append(f"{self.name}.reset_peak_memory_stats")

    def max_memory_allocated(self):
        self.log.append(f"{self.name}.max_memory_allocated")
        return 3 * 2 ** 30


def fake_torch(cuda=False, mps=False, xpu=False, old=False):
    """A stand-in for torch on a computer with the cards named; `old`: a torch without Metal or XPU parts."""
    log = []
    stand_in = types.SimpleNamespace(log=log, cuda=Recorder(cuda, log, "cuda"))
    stand_in._C = types.SimpleNamespace(_host_emptyCache=lambda: log.append("host_emptyCache"))
    if not old:
        stand_in.backends = types.SimpleNamespace(mps=Recorder(mps, log, "mps"))
        stand_in.mps = Recorder(mps, log, "mps")
        stand_in.xpu = Recorder(xpu, log, "xpu")
    return stand_in


class Pick(unittest.TestCase):
    def test_an_nvidia_card_is_cuda_and_nothing_else_is_asked(self):
        torch_ = fake_torch(cuda=True, mps=True, xpu=True)
        self.assertEqual(device.pick(torch_, {}), device.Device("cuda"))
        self.assertEqual(torch_.log, ["cuda.is_available"])

    def test_a_mac_is_metal(self):
        self.assertEqual(device.pick(fake_torch(mps=True), {}).backend, "mps")

    def test_an_intel_card_is_xpu(self):
        self.assertEqual(device.pick(fake_torch(xpu=True), {}).backend, "xpu")

    def test_no_card_is_the_processor_not_named(self):
        chosen = device.pick(fake_torch(), {})
        self.assertEqual(chosen, device.Device("cpu"))
        self.assertFalse(chosen.named)

    def test_a_torch_without_metal_or_xpu_parts_still_chooses(self):
        self.assertEqual(device.pick(fake_torch(old=True), {}).backend, "cpu")
        self.assertEqual(device.pick(fake_torch(cuda=True, old=True), {}).backend, "cuda")

    def test_the_variable_names_the_processor_on_a_card(self):
        chosen = device.pick(fake_torch(cuda=True), {device.VARIABLE: "CPU "})
        self.assertEqual((chosen.backend, chosen.named), ("cpu", True))

    def test_a_named_device_that_is_not_there_is_refused(self):
        with self.assertRaisesRegex(device.DeviceError, "Metal.*not available"):
            device.pick(fake_torch(cuda=True), {device.VARIABLE: "mps"})

    def test_an_unknown_name_is_refused(self):
        with self.assertRaisesRegex(device.DeviceError, "'rocm' is not a device"):
            device.pick(fake_torch(cuda=True), {device.VARIABLE: "rocm"})
        with self.assertRaises(device.DeviceError):
            device.Device("vulkan")

    def test_current_is_chosen_once(self):
        device.forget()
        self.addCleanup(device.forget)
        calls = []
        with mock.patch.object(device, "pick", lambda: calls.append(1) or device.Device("cpu")):
            self.assertIs(device.current(), device.current())
        self.assertEqual(calls, [1])

    def test_a_name_with_an_index_is_its_backend(self):
        self.assertEqual(device.Device.of("cuda:0"), device.Device("cuda"))


# What the pipeline wrote before the device layer, on an NVIDIA card (0.3.0): the device's name, each model's float
# type, the offload and pinning.
BEFORE_ON_CUDA = {"picture painter": "bfloat16",  # painter.py: torch_dtype and bnb_4bit_compute_dtype bfloat16
                  "cutter": "float16",  # cutout.py: float16 if the device is "cuda"
                  "state judge": "float32",  # state_judge.py: float32 on the card too
                  "shape": "float16",  # kept.py kept_models(torch.float16), shape_net.make dtype = torch.float16
                  "view painter": "float16",  # paint.py: torch_dtype, pipe.to, the control image in float16
                  "inpainter": "float16"}  # touch_up.py: the VAE and the pipeline in float16
# What the pipeline did with the card hidden (cutout.py and state_judge.py chose "cpu" and float32 themselves).
BEFORE_ON_CPU = {"cutter": "float32", "state judge": "float32"}


class FloatTypes(unittest.TestCase):
    def test_on_an_nvidia_card_every_model_keeps_its_measured_float_type(self):
        cuda = device.pick(fake_torch(cuda=True), {})
        self.assertEqual({model: cuda.dtype_name(model) for model in BEFORE_ON_CUDA}, BEFORE_ON_CUDA)
        self.assertEqual(set(device.CARD_DTYPES), set(BEFORE_ON_CUDA), "every model the pipeline loads has its type")
        self.assertEqual(cuda.name, "cuda")
        self.assertTrue(cuda.offloads and cuda.pins_memory)

    def test_on_the_processor_every_model_is_float32(self):
        cpu = device.Device("cpu", named=True)
        self.assertEqual({model: cpu.dtype_name(model) for model in device.CARD_DTYPES},
                         dict.fromkeys(device.CARD_DTYPES, "float32"))
        self.assertEqual({model: cpu.dtype_name(model) for model in BEFORE_ON_CPU}, BEFORE_ON_CPU)
        self.assertFalse(cpu.offloads or cpu.pins_memory or cpu.is_card)

    def test_metal_keeps_the_card_types_without_offload_or_pinning(self):
        mps = device.Device("mps")
        self.assertEqual(mps.dtype_name("shape"), "float16")
        self.assertFalse(mps.offloads or mps.pins_memory)

    def test_a_model_without_a_type_is_an_error_not_a_guess(self):
        with self.assertRaises(KeyError):
            device.Device("cuda").dtype_name("a model nobody named")

    @unittest.skipUnless(HAS_TORCH, "needs torch (installed by setup.sh)")
    def test_the_types_are_torchs_own(self):
        self.assertIs(device.Device("cuda").dtype("shape"), torch.float16)
        self.assertIs(device.Device("cuda").dtype("picture painter"), torch.bfloat16)
        self.assertIs(device.Device("cpu").dtype("shape"), torch.float32)
        self.assertEqual(device.Device("cuda").torch_device(), torch.device("cuda"))


class TheCardsMemory(unittest.TestCase):
    def setUp(self):
        # The real torch (if loaded) put back after each test.
        modules = mock.patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)

    def with_torch(self, stand_in):
        sys.modules["torch"] = stand_in
        return stand_in

    def test_on_cuda_the_cache_and_peak_calls_are_cudas(self):
        torch_ = self.with_torch(fake_torch(cuda=True))
        device.Device("cuda").empty_cache()
        device.release_cached(host_too=True)
        device.reset_peak()
        self.assertEqual(device.peak_mb(), 3072)
        calls = [call for call in torch_.log if not call.endswith("is_available")]
        self.assertEqual(calls, ["cuda.empty_cache", "cuda.empty_cache", "host_emptyCache",
                                 "cuda.reset_peak_memory_stats", "cuda.max_memory_allocated"])

    def test_metal_empties_its_own_cache_and_has_no_peak(self):
        torch_ = self.with_torch(fake_torch(mps=True))
        device.Device("mps").empty_cache()
        device.release_cached(host_too=True)
        self.assertIsNone(device.peak_mb())
        calls = [call for call in torch_.log if not call.endswith("is_available")]
        self.assertEqual(calls, ["mps.empty_cache", "mps.empty_cache"])

    def test_the_processor_touches_nothing(self):
        torch_ = self.with_torch(fake_torch())
        device.Device("cpu").empty_cache()
        device.release_cached(host_too=True)
        device.reset_peak()
        self.assertIsNone(device.peak_mb())
        self.assertEqual([call for call in torch_.log if not call.endswith("is_available")], [])

    def test_without_torch_loaded_nothing_imports_it(self):
        sys.modules.pop("torch", None)
        device.release_cached()
        device.reset_peak()
        self.assertIsNone(device.peak_mb())
        device.Device("cuda").empty_cache()
        self.assertNotIn("torch", sys.modules)


class Offload(unittest.TestCase):
    """paint.offload_wanted, the device asked for its memory."""

    class Card:
        def __init__(self, backend, gib):
            self.backend, self.gib = backend, gib
            self.offloads = backend in device.OFFLOADING_BACKENDS

        def total_memory_gib(self):
            return self.gib

    def wanted(self, card, variable=None):
        from model_maker import paint

        environment = {k: v for k, v in os.environ.items() if k != "MODEL_MAKER_VIEW_OFFLOAD"}
        if variable is not None:
            environment["MODEL_MAKER_VIEW_OFFLOAD"] = variable
        with mock.patch.dict(os.environ, environment, clear=True):
            return paint.offload_wanted(card)

    def test_as_before_on_cuda_and_never_where_it_saves_nothing(self):
        from model_maker import paint

        for gib in (11.6, 12.0, 14.9, 15.0, 16.0, 24.0):
            before = gib < paint.WHOLE_UNET_LEAST_GIB  # paint.py before the layer: the card's size alone
            self.assertEqual(self.wanted(self.Card("cuda", gib)), before, gib)
            self.assertFalse(self.wanted(self.Card("mps", gib)))
        self.assertFalse(self.wanted(self.Card("cpu", None)))
        self.assertTrue(self.wanted(self.Card("cpu", None), "1"))
        self.assertFalse(self.wanted(self.Card("cuda", 12.0), "0"))


class WhatCanRun(unittest.TestCase):
    def test_four_bits_on_cuda_with_bitsandbytes(self):
        self.assertEqual(device.four_bit_refusal(device.Device("cuda"), version=lambda _: "0.44.1",
                                                 found=lambda _: object()), "")

    def test_four_bits_on_cuda_without_bitsandbytes_says_so(self):
        why = device.four_bit_refusal(device.Device("cuda"), version=lambda _: None, found=lambda _: None)
        self.assertIn("bitsandbytes, which is not installed", why)

    def test_four_bits_elsewhere_names_what_is_missing(self):
        versions = {"bitsandbytes": "0.44.1", "diffusers": "0.31.0"}
        for backend, words in (("mps", "Apple's Metal"), ("xpu", "An Intel card"), ("cpu", "The processor")):
            why = device.four_bit_refusal(device.Device(backend), version=versions.get, found=lambda _: object())
            self.assertIn("bitsandbytes (0.44.1) with diffusers (0.31.0) quantises only on an NVIDIA card", why)
            self.assertIn(words, why)

    def test_the_card_stages_on_each_device(self):
        refused = "4-bit missing"
        four_bit = lambda chosen: "" if chosen.backend == "cuda" else refused  # noqa: E731
        cuda, mps = device.Device("cuda"), device.Device("mps")
        cpu, cpu_named = device.Device("cpu"), device.Device("cpu", named=True)
        for stage in device.CARD_STAGES:
            self.assertEqual(device.stage_refusal(stage, cuda, four_bit), "", stage)
            self.assertIn(f"{device.VARIABLE}=cpu", device.stage_refusal(stage, cpu, four_bit))
        for stage in ("shape", "paint"):
            self.assertEqual(device.stage_refusal(stage, cpu_named, four_bit), "")
            self.assertEqual(device.stage_refusal(stage, mps, four_bit), "")
        self.assertIn(refused, device.stage_refusal("pictures", cpu_named, four_bit))
        self.assertIn(refused, device.stage_refusal("pictures", mps, four_bit))
        with self.assertRaises(KeyError):
            device.stage_refusal("finish", cuda, four_bit)

    def test_the_pipeline_stops_a_stage_that_cannot_run_and_lets_one_that_can(self):
        from model_maker import pipeline

        with mock.patch.object(device, "current", lambda: device.Device("cpu")):
            with self.assertRaisesRegex(SystemExit, "needs a graphics card"):
                pipeline.needs_card("shape")
        # bitsandbytes there and not there, whatever this computer has: four_bit_refusal's own lookups stubbed (its
        # defaults, which stage_refusal's default holds), so the real refusal is what is tested.
        installed = (lambda package: "0.46.1", lambda package: object())
        missing = (lambda package: None, lambda package: None)
        with mock.patch.object(device, "current", lambda: device.Device("cuda")), \
                mock.patch.object(device.four_bit_refusal, "__defaults__", installed):
            for stage in device.CARD_STAGES:
                pipeline.needs_card(stage)
        with mock.patch.object(device, "current", lambda: device.Device("cuda")), \
                mock.patch.object(device.four_bit_refusal, "__defaults__", missing):
            with self.assertRaisesRegex(SystemExit, "pictures stage cannot run on an NVIDIA card .*bitsandbytes, "
                                                    "which is not installed"):
                pipeline.needs_card("pictures")
            pipeline.needs_card("shape")
            pipeline.needs_card("paint")

        def named_wrong():
            raise device.DeviceError("MODEL_MAKER_DEVICE=mps: not here")

        with mock.patch.object(device, "current", named_wrong):
            with self.assertRaisesRegex(SystemExit, "not here"):
                pipeline.needs_card("paint")


def nvidia_smi(stdout="NVIDIA GeForce RTX 3060, 12288, 580.95.05\n", calls=None):
    def run(command, **options):
        if calls is not None:
            calls.append((command, options))
        if command[0] != "nvidia-smi":
            raise FileNotFoundError(command[0])
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
    return run


def card_before(run):
    """ledger.machine()'s card before the device layer (0.3.0), word for word, run with a stand-in nvidia-smi."""
    card = {"card": None, "card_mb": None, "driver": None}
    try:
        done = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                   check=True)
        first = done.stdout.strip().splitlines()[0]
        name, mb, driver = (part.strip() for part in first.split(","))
        card = {"card": name, "card_mb": int(float(mb)), "driver": driver}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        pass
    return card


class TheRecord(unittest.TestCase):
    def card(self, environ, run, system="Linux", arch="x86_64"):
        with mock.patch.object(device, "intel_card", lambda: None):
            return device.card(environ, system, arch, run)

    def test_on_linux_with_an_nvidia_card_the_card_is_read_as_before(self):
        for stdout in ("NVIDIA GeForce RTX 3060, 12288, 580.95.05\n",
                       "NVIDIA GeForce RTX 3080 Laptop GPU, 16384, 581.1\nNVIDIA T4, 15360, 1\n", "", "garbled\n"):
            calls_now, calls_before = [], []
            now = self.card({}, nvidia_smi(stdout, calls_now))
            before = card_before(nvidia_smi(stdout, calls_before))
            self.assertEqual({k: now[k] for k in before}, before, stdout)
            self.assertEqual(calls_now, calls_before, "the same command, the same options")
            self.assertEqual(now["backend"], "cuda" if before["card"] else "cpu")

    def test_a_hidden_card_is_not_recorded(self):
        for hidden in ("", "-1", " ,0"):
            self.assertEqual(self.card({"CUDA_VISIBLE_DEVICES": hidden}, nvidia_smi()),
                             {**device.NO_CARD, "backend": "cpu"}, hidden)
        self.assertEqual(self.card({"CUDA_VISIBLE_DEVICES": "0"}, nvidia_smi())["card"], "NVIDIA GeForce RTX 3060")

    def test_the_processor_named_records_no_card(self):
        self.assertEqual(self.card({device.VARIABLE: "cpu"}, nvidia_smi()), {**device.NO_CARD, "backend": "cpu"})

    def test_no_nvidia_smi_is_no_card(self):
        def missing(command, **_):
            raise FileNotFoundError(command[0])

        self.assertEqual(self.card({}, missing), {**device.NO_CARD, "backend": "cpu"})

    def test_a_mac_records_its_chip_and_shared_memory(self):
        answers = {"machdep.cpu.brand_string": "Apple M2 Pro\n", "hw.memsize": str(32 * 2 ** 30) + "\n"}

        def sysctl(command, **_):
            return subprocess.CompletedProcess(command, 0, stdout=answers[command[-1]], stderr="")

        with mock.patch.object(device.platform, "mac_ver", lambda: ("15.1", ("", "", ""), "arm64")):
            found = self.card({}, sysctl, "Darwin", "arm64")
        self.assertEqual(found, {"card": "Apple M2 Pro (Metal, memory shared)", "card_mb": 32768,
                                 "driver": "macOS 15.1", "backend": "mps"})
        self.assertEqual(self.card({}, sysctl, "Darwin", "x86_64")["backend"], "cpu")

    def test_the_tier(self):
        self.assertEqual(device.tier("cuda", "Linux", "x86_64"), device.MEASURED)
        for backend, system, arch in (("cuda", "Windows", "AMD64"), ("mps", "Darwin", "arm64"),
                                      ("cpu", "Linux", "x86_64"), ("cuda", "Linux", "aarch64")):
            self.assertIsNone(device.tier(backend, system, arch), (backend, system))

    def test_the_ledger_records_the_backend_and_the_tier(self):
        from model_maker import ledger

        self.addCleanup(setattr, ledger, "_MACHINE", ledger._MACHINE)
        ledger._MACHINE = None
        with mock.patch.object(device, "card", lambda: {"card": "NVIDIA GeForce RTX 3060", "card_mb": 12288,
                                                        "driver": "580.95.05", "backend": "cuda"}), \
                mock.patch.object(device.platform, "system", lambda: "Linux"), \
                mock.patch.object(device.platform, "machine", lambda: "x86_64"):
            found = ledger.machine()
        self.assertEqual(found, {"card": "NVIDIA GeForce RTX 3060", "card_mb": 12288, "driver": "580.95.05",
                                 "backend": "cuda", "tier": "measured", "cpu_threads": os.cpu_count()})

    def test_elsewhere_the_ledger_and_the_bug_report_have_no_tier(self):
        from model_maker import bug_report, ledger

        self.addCleanup(setattr, ledger, "_MACHINE", ledger._MACHINE)
        ledger._MACHINE = None
        with mock.patch.object(device, "card", lambda: {"card": "NVIDIA GeForce RTX 3060", "card_mb": 12288,
                                                        "driver": "581.57", "backend": "cuda"}), \
                mock.patch.object(device.platform, "system", lambda: "Windows"), \
                mock.patch.object(device.platform, "machine", lambda: "AMD64"):
            found = ledger.machine()
        self.assertEqual(found, {"card": "NVIDIA GeForce RTX 3060", "card_mb": 12288, "driver": "581.57",
                                 "backend": "cuda", "cpu_threads": os.cpu_count()})
        self.assertNotIn("tier", bug_report.about_this_machine(found))
        self.assertEqual(bug_report.about_this_machine({**found, "tier": "measured"})["tier"], "measured")


class OnlyHere(unittest.TestCase):
    """The device layer is the one place that names a backend: no other module writes "cuda" or calls torch.cuda."""

    def test_no_other_module_names_a_backend(self):
        named = re.compile(r"""["']cuda["']|torch\.cuda|torch\.mps|torch\.xpu|float16|bfloat16|\.pin_memory\(""")
        found = []
        for name in sorted(os.listdir(PACKAGE)):
            if not name.endswith(".py") or name == "device.py":
                continue
            with open(os.path.join(PACKAGE, name)) as file:
                for number, line in enumerate(file, 1):
                    code = line.split("#")[0]
                    if named.search(code) and "sdxl-vae-fp16-fix" not in code:
                        found.append(f"{name}:{number}: {line.strip()}")
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
