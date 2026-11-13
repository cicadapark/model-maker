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
import re
import sys

REFUSED_LICENCE = re.compile(r"GPL|General Public|non-?commercial|\bNC\b", re.IGNORECASE)
# Longer than this, a License field is the whole text, not a name. Such texts are not searched: scipy's names
# libgfortran's GPL with its runtime exception, and matplotlib's says "GPL compatible". The libraries they speak of
# are judged one by one below.
LONGEST_LICENCE_NAME = 100
# Copyleft libraries wheels have bundled under permissive-looking packages: refused whatever else is said.
REFUSED_BUNDLED = re.compile(r"lib(avcodec|avformat|avutil|swscale|swresample|x264|x265|postproc)[-.]")
# A package's own compiled code (a Python extension module) is under the package's own licence.
EXTENSION_MODULE = re.compile(r"\.(cpython-\d+[^.]*|abi3)\.so$")

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
    "libbitsandbytes": "MIT (bitsandbytes)",
    "libgomp": GCC_RUNTIME,
    # NVIDIA's runtime libraries (the nvidia-*-cu12 wheels, and in torch's and torchvision's)
    "libcublas": NVIDIA, "libnvblas": NVIDIA, "libcudart": NVIDIA, "libcudnn": NVIDIA, "libcufft": NVIDIA,
    "libcupti": NVIDIA, "libcheckpoint": NVIDIA, "libpcsamplingutil": NVIDIA, "libnvperf": NVIDIA,
    "libcurand": NVIDIA, "libcusolver": NVIDIA, "libcusparse": NVIDIA, "libnvJitLink": NVIDIA,
    "libnvToolsExt": NVIDIA, "libnvrtc": NVIDIA, "libnvjpeg": NVIDIA,
    "libnccl": "BSD-3-Clause (NVIDIA NCCL)",
    # pillow's (and torchvision's) picture libraries
    "libjpeg": "IJG and BSD-3-Clause (libjpeg-turbo)", "libpng16": "libpng-2.0", "libz": "Zlib",
    "libwebp": "BSD-3-Clause", "libsharpyuv": "BSD-3-Clause", "libtiff": "libtiff (BSD-like)",
    "libopenjp2": "BSD-2-Clause", "libavif": "BSD-2-Clause", "liblcms2": "MIT", "libfreetype": "FTL (FreeType)",
    "libharfbuzz": "MIT", "libbrotli": "MIT", "libzstd": "BSD-3-Clause", "liblzma": "0BSD (xz)", "libxcb": "MIT",
    "libXau": "MIT",
    # c2pa-python's native library: c2pa-rs's C interface (0.91.3), MIT OR Apache-2.0, with its Rust crates linked in.
    # Every crate of its tree (`cargo metadata` of c2pa-c-ffi 0.91.3) is under MIT, Apache-2.0, BSD, ISC, Zlib,
    # Unicode-3.0, 0BSD, CC0-1.0, BSL-1.0 or the Unlicense (each with MIT or Apache-2.0 beside it where it is one of the
    # last two), but webpki-roots, CDLA-Permissive-2.0 (the root certificates' list, data); cbindgen, MPL-2.0, only
    # writes its C header at build time and is not in the library. No OpenSSL: it signs with ring and RustCrypto.
    "libc2pa_c": "MIT OR Apache-2.0 (c2pa-rs; its crates permissive, see above)",
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
        name = file.name
        if ".so" in name and not EXTENSION_MODULE.search(name):
            yield str(file), name


def judged_library(path, name):
    """(verdict, licence): "allowed" or "unknown"."""
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



def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    distributions = list(importlib.metadata.distributions())
    if "--list" in argv:
        for distribution in sorted(distributions, key=lambda d: d.metadata["Name"].lower()):
            for path, file_name in bundled_libraries(distribution):
                verdict, licence = judged_library(path, file_name)
                print(f"{distribution.metadata['Name']:28s} {file_name:52s} {verdict:9s} {licence}")
    problems = [problem for distribution in distributions for problem in problems_of(distribution)]
    for problem in problems:
        print(problem)
    print(f"{sys.prefix}: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
