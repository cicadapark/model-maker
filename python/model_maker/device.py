"""Where the models run: the one place that names a backend (an NVIDIA card through CUDA, Apple's Metal, Intel's XPU,
or the processor), the float type each model uses on it, and the card's memory, cache and record.

On an NVIDIA card every choice here is what the pipeline was measured with before this module existed: the device
"cuda", fp16 for the cut-out, the shape model and both SDXL painters, bf16 for FLUX.1's compute, float32 for the state
judge, the view painter's blocks offloaded on cards under 15 GB, FLUX.1 loaded in 4 bits by bitsandbytes. The
processor runs every model in float32: there fp16 and bf16 raised no error but were many times slower on a processor
without native half precision (an SDXL step over 11 minutes against 22 s in float32). Metal and XPU are chosen when
PyTorch offers them; nothing made on them has been measured.

`MODEL_MAKER_DEVICE` (cuda, mps, xpu or cpu) names the device instead of the first one found. The processor runs the
card stages only when named, as it is so slow.

Seeds do not depend on the device: every noise is drawn on the processor and moved over (see paint, shape_net).
"""

import importlib.metadata
import importlib.util
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass

VARIABLE = "MODEL_MAKER_DEVICE"
# In the order they are looked for when none is named.
BACKENDS = ("cuda", "mps", "xpu", "cpu")
WHAT = {"cuda": "an NVIDIA card (CUDA)", "mps": "Apple's Metal (MPS)", "xpu": "an Intel card (XPU)",
        "cpu": "the processor"}

# Each model's float type on a graphics card: on NVIDIA cards, the types 0.3.0 was measured with. The state judge is
# float32 everywhere (state_judge: in 16 bits a picture near its limit could be judged differently when made and in
# the regression).
CARD_DTYPES = {"picture painter": "bfloat16", "cutter": "float16", "state judge": "float32", "shape": "float16",
               "view painter": "float16", "inpainter": "float16"}
# On the processor every model is float32 (see the module's note).
CPU_DTYPE = "float32"

# The view painter's blocks are offloaded only where the card has memory of its own to save (not Metal's shared
# memory, not the processor), and pinned memory (faster copies to the card) exists only for CUDA here.
OFFLOADING_BACKENDS = ("cuda",)
PINNING_BACKENDS = ("cuda",)
# FLUX.1's 4-bit loading (bitsandbytes NF4 through the pinned diffusers and transformers) quantises on CUDA only:
# diffusers 0.38 refuses it without a CUDA or XPU card ("No GPU found. A GPU is needed for quantization."), and the
# locked bitsandbytes (0.46.1) is built for CUDA only, so XPU is refused here too until a lock with an XPU build.
FOUR_BIT_BACKENDS = ("cuda",)
# The oldest bitsandbytes each transformers major version loads in 4 bits (its BITSANDBYTES_MIN_VERSION): below it,
# transformers refuses only after the transformer's weights are read ("requires bitsandbytes"; the Windows card
# test of 28 Nov 2026 lost 10 minutes to a .venv behind the lock). Majors not named have no such floor.
BITSANDBYTES_FLOOR = {5: "0.46.1"}


class DeviceError(RuntimeError):
    """A device named in MODEL_MAKER_DEVICE that is not a backend, or not there."""


def available(torch, backend):
    """Whether PyTorch (`torch`, the module or a stand-in) can run on `backend`."""
    if backend == "cpu":
        return True
    if backend == "cuda":
        return bool(torch.cuda.is_available())
    if backend == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        return bool(mps is not None and mps.is_available())
    if backend == "xpu":
        xpu = getattr(torch, "xpu", None)
        return bool(xpu is not None and xpu.is_available())
    return False


def pick(torch=None, environ=None):
    """The device to run on: the one MODEL_MAKER_DEVICE names (checked to be there), else the first of CUDA, Metal,
    XPU that PyTorch offers, else the processor. On an NVIDIA card only `torch.cuda.is_available()` is asked."""
    environ = os.environ if environ is None else environ
    if torch is None:
        import torch
    named = (environ.get(VARIABLE) or "").strip().lower()
    if named:
        if named not in BACKENDS:
            raise DeviceError(f"{VARIABLE}={named!r} is not a device model-maker knows ({', '.join(BACKENDS)})")
        if not available(torch, named):
            raise DeviceError(f"{VARIABLE}={named}: {WHAT[named]} is not available to PyTorch on this computer")
        return Device(named, named=True)
    for backend in BACKENDS[:-1]:
        if available(torch, backend):
            return Device(backend)
    return Device("cpu")


_CHOSEN = None


def current():
    """The device this process runs its models on, chosen once (`pick`)."""
    global _CHOSEN
    if _CHOSEN is None:
        _CHOSEN = pick()
    return _CHOSEN


def forget():
    """The choice made again on the next `current()` (for tests)."""
    global _CHOSEN
    _CHOSEN = None


@dataclass(frozen=True)
class Device:
    backend: str
    # Whether MODEL_MAKER_DEVICE named it (the processor runs the card stages only then).
    named: bool = False

    def __post_init__(self):
        if self.backend not in BACKENDS:
            raise DeviceError(f"{self.backend!r} is not a device model-maker knows ({', '.join(BACKENDS)})")

    @classmethod
    def of(cls, name):
        """The device of a name torch takes ("cuda", "cuda:0", "cpu", a torch.device)."""
        return cls(str(name).split(":")[0])

    @property
    def name(self):
        """What torch's `.to()` is given: "cuda" on an NVIDIA card, as the pipeline always wrote it."""
        return self.backend

    @property
    def is_card(self):
        return self.backend != "cpu"

    @property
    def offloads(self):
        return self.backend in OFFLOADING_BACKENDS

    @property
    def pins_memory(self):
        return self.backend in PINNING_BACKENDS

    def dtype_name(self, model):
        """The float type's name for `model` (a key of CARD_DTYPES) on this device."""
        if model not in CARD_DTYPES:
            raise KeyError(f"no float type is set for {model!r}")
        return CARD_DTYPES[model] if self.is_card else CPU_DTYPE

    def dtype(self, model):
        import torch

        return getattr(torch, self.dtype_name(model))

    def torch_device(self):
        import torch

        return torch.device(self.name)

    def total_memory_gib(self):
        """The card's own memory (GiB), or None where it has none of its own (Metal, the processor)."""
        import torch

        if self.backend == "cuda":
            return torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        if self.backend == "xpu":
            return torch.xpu.get_device_properties(0).total_memory / 2 ** 30
        return None

    def empty_cache(self):
        """The memory the card's allocator keeps for later handed back (nothing on the processor)."""
        torch = sys.modules.get("torch")
        if torch is None or not self.is_card:
            return
        getattr(torch, self.backend).empty_cache()


def _in_use(torch):
    """The card backend torch has (CUDA first), without choosing: None when there is none or torch is not loaded."""
    if torch is None:
        return None
    for backend in BACKENDS[:-1]:
        try:
            if available(torch, backend):
                return backend
        except (AttributeError, RuntimeError):
            continue
    return None


def release_cached(host_too=False):
    """Whatever the card's allocator keeps, handed back, if torch is loaded (it is never imported for this); with
    `host_too`, CUDA's cache of freed pinned memory too."""
    torch = sys.modules.get("torch")
    backend = _in_use(torch)
    if backend is None:
        return
    try:
        getattr(torch, backend).empty_cache()
        if host_too and backend == "cuda":
            torch._C._host_emptyCache()
    except (AttributeError, RuntimeError):
        pass


def reset_peak():
    """The card's peak memory counter set back (CUDA, XPU; Metal keeps none)."""
    torch = sys.modules.get("torch")
    backend = _in_use(torch)
    if backend in ("cuda", "xpu"):
        reset = getattr(getattr(torch, backend), "reset_peak_memory_stats", None)
        if reset is not None:
            reset()


def peak_mb():
    """The card's peak memory taken by tensors since the last reset (MB), or None (no card, or Metal, which keeps no
    peak)."""
    torch = sys.modules.get("torch")
    backend = _in_use(torch)
    if backend in ("cuda", "xpu"):
        most = getattr(getattr(torch, backend), "max_memory_allocated", None)
        if most is not None:
            return round(most() / 2 ** 20)
    return None


# --- page-locked memory for the copies to an NVIDIA card (paint.BlockOffload, 12 GB cards) ---------------------------


def lock_pages(pointer, size):
    """Whether the CUDA runtime page-locked [pointer, pointer + size) for the card's copies (cudaHostRegister)."""
    import torch

    try:
        cudart = torch.cuda.cudart()
        if cudart.cudaHostRegister(pointer, size, 0) == cudart.cudaError.success:
            return True
    except (AttributeError, RuntimeError):
        return False
    _forget_runtime_error()
    return False


def unlock_pages(pointer):
    import torch

    torch.cuda.cudart().cudaHostUnregister(pointer)


def wait_for_card_copies():
    """The card's copies finished (before page-locked memory is given back), where there is a CUDA card."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def pinned_copy(tensor):
    """The tensor in pinned memory of the computer, alone (where a whole block's buffer could not be page-locked)."""
    return tensor.to("cpu").pin_memory()


def _forget_runtime_error():
    """A refused registration stays the CUDA runtime's last error, which PyTorch's check after its next kernel would
    raise as that kernel's: read once, which clears it (the runtime PyTorch loaded, found by its name)."""
    import ctypes

    for name in ("libcudart.so.12", "libcudart.so.13", "cudart64_12.dll", "cudart64_13.dll"):
        try:
            ctypes.CDLL(name).cudaGetLastError()
            return
        except (OSError, AttributeError):
            continue


# --- what can run here ----------------------------------------------------------------------------------------------


def version_of(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def release_of(text):
    """A version's release numbers ("0.46.1" -> (0, 46, 1); "2.10.0+cu128" -> (2, 10, 0); "1.0rc1" -> (1, 0)), for
    comparing."""
    numbers = []
    for part in text.split("+")[0].split("."):
        digits = re.match(r"\d+", part)
        if digits is None:
            break
        numbers.append(int(digits.group()))
        if digits.end() < len(part):
            break
    return tuple(numbers)


def bitsandbytes_floor(transformers):
    """The oldest bitsandbytes `transformers` (its version, or None) loads in 4 bits, or None when it names none."""
    major = release_of(transformers)[:1] if transformers else ()
    return BITSANDBYTES_FLOOR.get(major[0]) if major else None


def four_bit_refusal(device, version=version_of, found=importlib.util.find_spec):
    """Why FLUX.1 cannot be loaded in 4 bits on `device` ("" when it can): the 4-bit route is bitsandbytes NF4 through
    the pinned diffusers, which quantises only on CUDA."""
    bitsandbytes = version("bitsandbytes")
    if device.backend in FOUR_BIT_BACKENDS:
        if bitsandbytes is None or found("bitsandbytes") is None:
            return ("the picture model (FLUX.1 [schnell]) is loaded in 4 bits by bitsandbytes, which is not installed "
                    "(it is in python/requirements.txt: run ./setup.sh)")
        transformers = version("transformers")
        floor = bitsandbytes_floor(transformers)
        if floor and release_of(bitsandbytes) < release_of(floor):
            return (f"the picture model (FLUX.1 [schnell]) is loaded in 4 bits by bitsandbytes, and the installed "
                    f"bitsandbytes ({bitsandbytes}) is older than transformers {transformers} takes for that ({floor} "
                    f"or newer): this Python environment is behind the lock; run setup again")
        return ""
    return (f"the picture model (FLUX.1 [schnell]) is loaded in 4 bits by bitsandbytes, and this release's "
            f"bitsandbytes ({bitsandbytes or 'not installed'}) with diffusers ({version('diffusers') or 'not installed'})"
            f" quantises only on an NVIDIA card (CUDA). {WHAT[device.backend][0].upper()}{WHAT[device.backend][1:]} "
            f"needs a 4-bit route this release does not have: a bitsandbytes and a diffusers that quantise on it")


CARD_STAGES = ("pictures", "shape", "paint")


def stage_refusal(stage, device, four_bit=None):
    """Why a card stage ("pictures", "shape", "paint") cannot run on `device` ("" when it can). `four_bit` defaults to
    four_bit_refusal as it is when called (so a test's stand-in for it is the one asked)."""
    if stage not in CARD_STAGES:
        raise KeyError(f"{stage!r} is not a card stage")
    if device.backend == "cpu" and not device.named:
        return (f"this stage needs a graphics card, and PyTorch found none (model-maker is measured on an NVIDIA card "
                f"with CUDA); the finish stage runs without one. {VARIABLE}=cpu runs the shape and paint stages on "
                f"the processor in float32, very slowly (about half an hour a shape try on 16 threads)")
    if stage == "pictures":
        why = (four_bit or four_bit_refusal)(device)
        if why:
            return f"the pictures stage cannot run on {WHAT[device.backend]}: {why}"
    return ""


# --- the card, for the ledger and the bug report (no torch needed) ---------------------------------------------------

NO_CARD = {"card": None, "card_mb": None, "driver": None}


def cuda_hidden(environ=None):
    """Whether CUDA_VISIBLE_DEVICES hides every card ("" or "-1"): nvidia-smi does not read it, so it would name a card
    no model runs on."""
    environ = os.environ if environ is None else environ
    value = environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return False
    first = value.split(",")[0].strip()
    return first == "" or first.startswith("-")


def nvidia_card(run=subprocess.run):
    """The first NVIDIA card as nvidia-smi names it: its name, memory (MB) and driver; None when there is none."""
    try:
        done = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True, timeout=10, check=True)
        first = done.stdout.strip().splitlines()[0]
        name, mb, driver = (part.strip() for part in first.split(","))
        return {"card": name, "card_mb": int(float(mb)), "driver": driver}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def apple_chip(run=subprocess.run, mac_version=None):
    """An Apple Silicon Mac's chip (its Metal card) and its memory, shared with the processor; None elsewhere."""
    try:
        chip = run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=10,
                   check=True).stdout.strip()
        memory = int(run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=10,
                         check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if not chip:
        return None
    version = mac_version if mac_version is not None else platform.mac_ver()[0]
    return {"card": f"{chip} (Metal, memory shared)", "card_mb": memory // 2 ** 20, "driver": f"macOS {version}"}


def intel_card():
    """An Intel card as PyTorch sees it, only if torch is loaded already; None otherwise."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and available(torch, "xpu"):
            properties = torch.xpu.get_device_properties(0)
            return {"card": properties.name, "card_mb": properties.total_memory // 2 ** 20,
                    "driver": getattr(properties, "driver_version", None)}
    except (AttributeError, RuntimeError):
        pass
    return None


def card(environ=None, system=None, arch=None, run=subprocess.run):
    """The card the models would run on, for the record: {"card", "card_mb", "driver", "backend"}. On a computer
    with an NVIDIA card it is nvidia-smi's first card, read as before; no card when MODEL_MAKER_DEVICE says cpu or
    CUDA_VISIBLE_DEVICES hides every card. Torch is not imported for it (the finish stage runs without it)."""
    environ = os.environ if environ is None else environ
    system = system or platform.system()
    arch = arch or platform.machine()
    named = (environ.get(VARIABLE) or "").strip().lower()
    if named == "cpu":
        return {**NO_CARD, "backend": "cpu"}
    if system == "Darwin":
        found = apple_chip(run) if arch == "arm64" and named in ("", "mps") else None
        return {**found, "backend": "mps"} if found else {**NO_CARD, "backend": "cpu"}
    if named in ("", "cuda") and not cuda_hidden(environ):
        found = nvidia_card(run)
        if found:
            return {**found, "backend": "cuda"}
    if named in ("", "xpu"):
        found = intel_card()
        if found:
            return {**found, "backend": "xpu"}
    return {**NO_CARD, "backend": "cpu"}


MEASURED = "measured"


def tier(backend, system=None, arch=None):
    """"measured" for Linux with an NVIDIA card, where every published number was made; None (no tier) elsewhere."""
    system = system or platform.system()
    arch = (arch or platform.machine()).lower()
    if backend == "cuda" and system == "Linux" and arch in ("x86_64", "amd64"):
        return MEASURED
    return None
