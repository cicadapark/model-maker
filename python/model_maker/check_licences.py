#!/usr/bin/env python3
"""Check a Python environment against the rule: no GPL, AGPL, LGPL or non-commercial code, ever.

A package's own licence line is not enough: opencv-python-headless says Apache-2.0 and its wheel
bundled FFmpeg under the LGPL. So every shared library a package bundles (a wheel's `<package>.libs/`,
`torch/lib/`, ...) is listed and compared with an allow-list, each entry with its licence: a library not
on it is a problem, so a new one is read before it is let in (numpy's and scipy's wheels bundle libquadmath,
LGPL-2.1-or-later, which a check of the packages' own licence lines reports as 0 problems).

Run with the environment's own Python, which is what it reads:
    .venv/bin/python -I python/model_maker/check_licences.py [--list]
It prints each problem and exits 1 if there is one. `--list` prints every bundled library with its licence.
"""

import importlib.metadata
import os
import re
import sys

# The repository: this file is python/model_maker/check_licences.py.
REPOSITORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REFUSED_LICENCE = re.compile(r"GPL|General Public|non-?commercial|\bNC\b", re.IGNORECASE)
# Longer than this, a License field is the whole text, not a name. Such texts are not searched: scipy's names
# libgfortran's GPL with its runtime exception, and matplotlib's says "GPL compatible". The libraries they speak of
# are judged one by one below.
LONGEST_LICENCE_NAME = 100
# Copyleft libraries wheels have bundled under permissive-looking packages: refused whatever else is said.
REFUSED_BUNDLED = re.compile(r"lib(avcodec|avformat|avutil|swscale|swresample|x264|x265|postproc)[-.]")
# A package's own compiled code (a Python extension module) is under the package's own licence.
EXTENSION_MODULE = re.compile(r"\.(cpython-\d+[^.]*|abi3)\.so$")
# The systems whose bundled libraries the allow-list below knows (by their file names): every library the wheels of
# python/requirements.lock and python/locks/ bundle there, each read (tools/ci/locks.py licences, in CI). Elsewhere the
# check lists what it finds and fails, rather than pass having recognised nothing.
KNOWN_SYSTEMS = ("linux", "win32", "darwin")


def is_library(name, system=sys.platform):
    """Whether a file of a wheel is a shared library it bundles (not the package's own extension module)."""
    lowered = name.lower()
    if system.startswith("win"):
        return lowered.endswith(".dll")
    if system == "darwin" and lowered.endswith(".dylib"):
        return True
    return ".so" in name and not EXTENSION_MODULE.search(name)


def system_known(system=sys.platform):
    return any(system.startswith(known) for known in KNOWN_SYSTEMS)

# Every shared library the locked environment bundles, by the start of its file name, with its licence. A library
# under the GPL with the GCC Runtime Library Exception may be shipped with any program (the exception's purpose).
GCC_RUNTIME = "GPL-3.0-or-later WITH GCC-exception-3.1 (the GCC Runtime Library Exception)"
NVIDIA = ("NVIDIA proprietary, its CUDA, cuDNN or cuSPARSELt licence: free to use and to ship with an application, not "
          "open source")
ALLOWED_BUNDLED = {
    # numpy and scipy (their wheels' .libs folders)
    "libscipy_openblas": "BSD-3-Clause (OpenBLAS)",
    "libgfortran": GCC_RUNTIME,
    # torch, torchvision, triton, bitsandbytes: their own code
    "libc10": "BSD-3-Clause (PyTorch)", "libtorch": "BSD-3-Clause (PyTorch)", "libshm": "BSD-3-Clause (PyTorch)",
    "libcaffe2_nvrtc": "BSD-3-Clause (PyTorch)", "libaoti_custom_ops": "BSD-3-Clause (PyTorch)",
    "libbackend_with_compiler": "BSD-3-Clause (PyTorch)", "libjitbackend_test": "BSD-3-Clause (PyTorch)",
    "libtorchbind_test": "BSD-3-Clause (PyTorch)",
    "torchvision/_C.so": "BSD-3-Clause (torchvision)", "torchvision/image.so": "BSD-3-Clause (torchvision)",
    "libtriton": "MIT (Triton)", "libproton": "MIT (Triton)",
    # Triton's own instrumentation libraries (its wheel's triton/instrumentation/), MIT as the rest of it.
    "triton/instrumentation/libGPUInstrumentationTestLib.so": "MIT (Triton)",
    "triton/instrumentation/libPrintLoadStoreMemSpaces.so": "MIT (Triton)",
    "libbitsandbytes": "MIT (bitsandbytes)",
    "libgomp": GCC_RUNTIME,
    # NVIDIA's runtime libraries (the nvidia-*-cu12 wheels, and in torch's and torchvision's)
    "libcublas": NVIDIA, "libnvblas": NVIDIA, "libcudart": NVIDIA, "libcudnn": NVIDIA, "libcufft": NVIDIA,
    "libcupti": NVIDIA, "libcheckpoint": NVIDIA, "libpcsamplingutil": NVIDIA, "libnvperf": NVIDIA,
    "libcurand": NVIDIA, "libcusolver": NVIDIA, "libcusparse": NVIDIA, "libnvJitLink": NVIDIA,
    "libnvToolsExt": NVIDIA, "libnvrtc": NVIDIA, "libnvjpeg": NVIDIA,
    # torch 2.10's: cuFile (GPUDirect Storage) and NVSHMEM; each wheel's License.txt is the CUDA Toolkit's end user
    # licence, the same text, byte for byte, as cuBLAS's.
    "libcufile": NVIDIA, "libnvshmem": NVIDIA, "nvshmem_bootstrap_": NVIDIA, "nvshmem_transport_": NVIDIA,
    "libnccl": "BSD-3-Clause (NVIDIA NCCL)",
    # pillow's (and torchvision's) picture libraries
    "libjpeg": "IJG and BSD-3-Clause (libjpeg-turbo)", "libpng16": "libpng-2.0", "libz": "Zlib",
    "libwebp": "BSD-3-Clause", "libsharpyuv": "BSD-3-Clause", "libtiff": "libtiff (BSD-like)",
    "libopenjp2": "BSD-2-Clause", "libavif": "BSD-2-Clause", "liblcms2": "MIT", "libfreetype": "FTL (FreeType)",
    "libharfbuzz": "MIT", "libbrotli": "MIT", "libzstd": "BSD-3-Clause", "liblzma": "0BSD (xz)", "libxcb": "MIT",
    "libXau": "MIT",
    # c2pa-python's native library: c2pa-rs's C interface (0.91.3 in c2pa-python 0.38.1, 0.92.0 in 0.39.0, the
    # Windows lock's), MIT OR Apache-2.0, with its Rust crates linked in. Every crate of its tree (`cargo metadata`
    # of c2pa-c-ffi 0.91.3, and of 0.92.0: 424 crates) is under MIT, Apache-2.0, BSD, ISC, Zlib, Unicode-3.0, 0BSD,
    # CC0-1.0, BSL-1.0 or the Unlicense (each with MIT or Apache-2.0 beside it where it is one of the last two;
    # r-efi offers MIT or Apache-2.0 beside the LGPL), but webpki-roots, CDLA-Permissive-2.0 (the root certificates'
    # list, data); cbindgen, MPL-2.0, only writes its C header at build time and is not in the library. No OpenSSL:
    # it signs with ring and RustCrypto.
    "libc2pa_c": "MIT OR Apache-2.0 (c2pa-rs; its crates permissive, see above)",
    "c2pa_c.dll": "MIT OR Apache-2.0 (c2pa-rs; its crates permissive, see above)",
    # Windows (python/locks/windows-x86_64-cuda128.lock): PyTorch's own and what its third_party folder names in the
    # wheel's LICENSE file, NVIDIA's runtime under its Windows names, and two runtimes free to ship but not open source.
    "c10.dll": "BSD-3-Clause (PyTorch)", "c10_cuda.dll": "BSD-3-Clause (PyTorch)",
    "caffe2_nvrtc.dll": "BSD-3-Clause (PyTorch)", "shm.dll": "BSD-3-Clause (PyTorch)",
    "torch.dll": "BSD-3-Clause (PyTorch)", "torch_cpu.dll": "BSD-3-Clause (PyTorch)",
    "torch_cuda.dll": "BSD-3-Clause (PyTorch)", "torch_global_deps.dll": "BSD-3-Clause (PyTorch)",
    "torch_python.dll": "BSD-3-Clause (PyTorch)", "fbgemm.dll": "BSD-3-Clause (FBGEMM, in PyTorch)",
    "asmjit.dll": "Zlib (AsmJit, in PyTorch)", "uv.dll": "MIT (libuv, in PyTorch)", "zlibwapi.dll": "Zlib",
    "zlib.dll": "Zlib", "jpeg8.dll": "IJG and BSD-3-Clause (libjpeg-turbo)",
    "cublas64_": NVIDIA, "cublasLt64_": NVIDIA, "cudart64_": NVIDIA, "cudnn64_": NVIDIA, "cudnn_": NVIDIA,
    "cufft64_": NVIDIA, "cufftw64_": NVIDIA, "cupti64_": NVIDIA, "curand64_": NVIDIA, "cusolver64_": NVIDIA,
    "cusolverMg64_": NVIDIA, "cusparse64_": NVIDIA, "nvJitLink_": NVIDIA, "nvToolsExt64_": NVIDIA,
    "nvrtc-builtins64_": NVIDIA, "nvrtc64_": NVIDIA, "nvjpeg64_": NVIDIA,
    # torch 2.10's (torch/lib/): NVIDIA's Nsight Perf SDK host library (2024.5, its version resource), which CUPTI
    # loads; Linux's libnvperf_host, which NVIDIA's own CUPTI wheel ships, under its Windows name. The CUDA licence's
    # Attachment A names only cupti.dll for CUPTI; proprietary like the rest of NVIDIA's, not copyleft.
    "nvperf_host.dll": NVIDIA,
    # torchvision 0.25's Windows wheel (torchvision\\ in its RECORD), each read from the wheel: the CUDA runtime 12.8.90
    # and nvJPEG 12.3 (NVIDIA, the rows above), libjpeg-turbo 2.0.0 (jpeg8.dll), zlib 1.3.1 (zlib.dll), libpng
    # 1.6.50 and libwebp with its SharpYuv (the libpng16, libwebp and libsharpyuv rows), and these two:
    "libjpeg.dll": ("IJG (the Independent JPEG Group's own libjpeg, 9f of 14 January 2024): permissive; a program "
                    "shipped as a binary says it is based in part on the work of the Independent JPEG Group"),
    "python311.dll": ("PSF-2.0 (the Python Software Foundation License: the CPython 3.11.14 runtime, its version "
                      "resource read; permissive)"),
    "msvcp140": ("Microsoft's Visual C++ runtime: proprietary, its redistributable files free to ship with an "
                 "application (numpy's Windows wheel bundles it), not open source"),
    "libiomp5md.dll": ("Intel's OpenMP runtime, under the Intel End User License Agreement for Developer Tools (PyPI's "
                       "intel-openmp): proprietary, its Redistributables free to ship as part of a product, no "
                       "non-commercial limit and no copyleft; not open source"),
    "libiompstubs5md.dll": "the same as libiomp5md.dll (Intel's OpenMP stubs)",
    # macOS on Apple Silicon (python/locks/macos-arm64.lock)
    "libgcc_s": GCC_RUNTIME,
    "libomp": "Apache-2.0 WITH LLVM-exception (LLVM's OpenMP runtime; older releases MIT or NCSA)",
    "libc++": "Apache-2.0 WITH LLVM-exception (LLVM's libc++)",
}
# The rule's one recorded exception (CONTRIBUTING.md, rule 1): an LGPL library that comes inside a standard package's
# own PyPI wheel, is loaded only at run time and stays replaceable. It is allowed and listed in NOTICE.
ACCEPTED_EXCEPTIONS = {
    "libquadmath": "LGPL-2.1-or-later: the recorded exception (GCC's quad-precision maths, inside numpy's and scipy's "
                   "own wheels, loaded at run time; NOTICE)",
}


def bundled_libraries(distribution):
    """The shared libraries a package bundles: (path within the environment, file name), its own Python extension
    modules left out."""
    for file in distribution.files or []:
        # A wheel's RECORD may write Windows paths (torchvision's does: torchvision\\zlib.dll), which importlib's
        # posix paths keep whole in `.name`: the file's own name is after the last separator of either kind.
        path = str(file).replace("\\", "/")
        name = path.rsplit("/", 1)[-1]
        if is_library(name):
            yield path, name


def judged_library(path, name):
    """(verdict, licence): "allowed" or "unknown". A row naming the whole file is taken before a row naming the start of
    a name (Windows' libjpeg.dll is the IJG's libjpeg, not libjpeg-turbo's libjpeg.so)."""
    path = path.replace("\\", "/")
    for table, verdict in ((ACCEPTED_EXCEPTIONS, "allowed"), (ALLOWED_BUNDLED, "allowed")):
        if name in table:
            return verdict, table[name]
    for table, verdict in ((ACCEPTED_EXCEPTIONS, "allowed"), (ALLOWED_BUNDLED, "allowed")):
        for start, licence in table.items():
            if ("/" in start and path.endswith(start)) or name.startswith(start):
                return verdict, licence
    return "unknown", ""


def problems_of(distribution):
    metadata = distribution.metadata
    name = metadata["Name"]
    named = metadata.get("License") or ""
    licence_words = [named if len(named) <= LONGEST_LICENCE_NAME else "", metadata.get("License-Expression") or ""]
    licence_words += [line for line in metadata.get_all("Classifier", []) if line.startswith("License")]
    licence = " ".join(licence_words)
    if REFUSED_LICENCE.search(licence):
        yield f"{name}: its licence reads {licence[:160]!r}"
    for path, file_name in bundled_libraries(distribution):
        if REFUSED_BUNDLED.search(file_name):
            yield f"{name}: it bundles {file_name}"
        elif judged_library(path, file_name)[0] == "unknown":
            yield f"{name}: it bundles {file_name}, which is not on the allow-list (read its licence first)"



# The question judge's server (question_judge.py): llama.cpp's own release archive, unpacked whole into this folder of
# the repository by setup.sh. Its shared libraries are llama.cpp's and ggml's own (MIT, its LICENSE kept beside them);
# it links the system's libstdc++, libgomp, OpenSSL and zlib, which it does not bundle.
JUDGE_FOLDER = os.path.join(".tools", "llama-b11600")
JUDGE_LIBRARIES = {"libggml": "MIT (ggml, llama.cpp's LICENSE)", "libllama": "MIT (llama.cpp's LICENSE)",
                   "libmtmd": "MIT (llama.cpp's LICENSE)",
                   # The Windows zip's names (no "lib"), and LLVM's OpenMP runtime it carries (its own licence file,
                   # LICENSE-LLVM-OpenMP, beside it); its LICENSE is fetched from the tag's source by setup.
                   "ggml": "MIT (ggml, llama.cpp's LICENSE)", "llama": "MIT (llama.cpp's LICENSE)",
                   "mtmd": "MIT (llama.cpp's LICENSE)",
                   "libomp": "Apache-2.0 WITH LLVM-exception (LLVM's OpenMP runtime, LICENSE-LLVM-OpenMP)"}


def judge_libraries(folder, system=None):
    """(file name, verdict, licence) for each shared library in the judge's folder, and the problems: an unknown
    library, or no MIT LICENSE beside them. Nothing when the folder is not there (the judge is optional)."""
    if not os.path.isdir(folder):
        return [], []
    rows, problems = [], []
    for name in sorted(os.listdir(folder)):
        if ".so" not in name and not is_library(name, system or sys.platform):
            continue
        licence = next((text for start, text in JUDGE_LIBRARIES.items() if name.startswith(start)), "")
        rows.append((name, "allowed" if licence else "unknown", licence))
        if not licence:
            problems.append(f"{JUDGE_FOLDER}: it holds {name}, which is not on the allow-list (read its licence first)")
    try:
        with open(os.path.join(folder, "LICENSE")) as file:
            if not file.readline().startswith("MIT License"):
                problems.append(f"{JUDGE_FOLDER}/LICENSE: not llama.cpp's MIT licence")
    except OSError:
        problems.append(f"{JUDGE_FOLDER}: its LICENSE is missing (unpack the release archive whole)")
    return rows, problems


# The standalone CPython setup installs (uv's, from python-build-standalone) carries one copyleft library: its
# `_dbm` module links Berkeley DB under the Sleepycat licence (Linux build). `_gdbm` would link GNU dbm (GPL); no build
# has it today. Nothing in model-maker or its locks imports `dbm` (Python's `dbm` falls back to `dbm.dumb` without
# them), so setup removes them after installing the interpreter (docs/licences.md), and this check refuses them there.
# In an interpreter setup did not install (a developer's own, CI's), they are only reported: that Python is not ours.
REFUSED_INTERPRETER_MODULES = ("_dbm", "_gdbm")


def interpreter_modules_folder(base, system=sys.platform):
    """Where an interpreter keeps its compiled standard modules."""
    if system == "win32":
        return os.path.join(base, "DLLs")
    return os.path.join(base, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "lib-dynload")


def interpreter_findings(base=sys.base_prefix, system=sys.platform, repository=REPOSITORY):
    """(problems, notes): the refused modules found in the interpreter; problems when setup installed it (it lives in
    the repository's .tools), notes otherwise."""
    folder = interpreter_modules_folder(base, system)
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return [], []
    found = [name for name in names if name.split(".")[0] in REFUSED_INTERPRETER_MODULES]
    words = [f"{folder}: {name} links a copyleft library (Berkeley DB or GNU dbm); setup removes it" for name in found]
    ours = os.path.realpath(base).startswith(os.path.realpath(os.path.join(repository, ".tools")) + os.sep)
    return (words, []) if ours else ([], words)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    distributions = list(importlib.metadata.distributions())
    judge_rows, judge_problems = judge_libraries(os.path.join(REPOSITORY, JUDGE_FOLDER))
    if "--list" in argv:
        for distribution in sorted(distributions, key=lambda d: d.metadata["Name"].lower()):
            for path, file_name in bundled_libraries(distribution):
                verdict, licence = judged_library(path, file_name)
                print(f"{distribution.metadata['Name']:28s} {file_name:52s} {verdict:9s} {licence}")
        for file_name, verdict, licence in judge_rows:
            print(f"{'llama.cpp (question judge)':28s} {file_name:52s} {verdict:9s} {licence}")
    problems = [problem for distribution in distributions for problem in problems_of(distribution)]
    problems += judge_problems
    interpreter_problems, interpreter_notes = interpreter_findings()
    problems += interpreter_problems
    for note in interpreter_notes:
        print(f"note (an interpreter setup did not install): {note}")
    if not system_known():
        problems.append(f"this check knows the libraries of Linux's wheels only; on {sys.platform} it cannot judge "
                        f"what the wheels bundle, so it does not pass")
    for problem in problems:
        print(problem)
    print(f"{sys.prefix}: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
