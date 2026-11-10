"""The shape model's dense surface brought to a triangle budget by `model-maker-simplify` (the Rust crate of that
name: meshoptimizer's edge collapses), through two small files."""

import os
import shutil
import struct
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.normpath(os.path.join(HERE, "..", ".."))
FORMAT_VERSION = 1


def tool():
    """The built tool: `MODEL_MAKER_SIMPLIFY`, else the repository's release build, else one on the PATH."""
    named = os.environ.get("MODEL_MAKER_SIMPLIFY")
    if named:
        return named
    built = os.path.join(REPOSITORY, "target", "release", "model-maker-simplify")
    if os.path.exists(built):
        return built
    found = shutil.which("model-maker-simplify")
    if found:
        return found
    raise FileNotFoundError("model-maker-simplify is not built: cargo build --release -p model-maker-simplify")


def surface_bytes(corners, triangles):
    corners = np.ascontiguousarray(corners, dtype="<f4")
    triangles = np.ascontiguousarray(triangles, dtype="<u4")
    return (b"MMSH" + struct.pack("<III", FORMAT_VERSION, len(corners), len(triangles)) + corners.tobytes()
            + triangles.tobytes())


def read_indices(data):
    if data[:4] != b"MMIX":
        raise ValueError("not an MMIX file")
    (count,) = struct.unpack("<I", data[4:8])
    indices = np.frombuffer(data[8:8 + 4 * count], dtype="<u4")
    if len(indices) != count or count % 3:
        raise ValueError("the MMIX file is cut short")
    return indices.reshape(-1, 3).astype(np.int64)


def simplified(corners, triangles, most):
    """The surface with at most `most` triangles: (corners, triangles, which of the dense corners were kept, how: the
    tool's own words, "edge collapse" or "grid clustering (...)", as it prints them)."""
    if len(triangles) <= most:
        return corners, triangles, np.arange(len(corners)), "kept as it was"
    with tempfile.TemporaryDirectory() as work:
        dense, kept = os.path.join(work, "dense.mesh"), os.path.join(work, "kept.indices")
        with open(dense, "wb") as file:
            file.write(surface_bytes(corners, triangles))
        done = subprocess.run([tool(), dense, kept, "--most", str(most)], check=True, capture_output=True, text=True)
        with open(kept, "rb") as file:
            indices = read_indices(file.read())
    used, inverse = np.unique(indices.reshape(-1), return_inverse=True)
    # The tool says "N triangles kept of M by <method>".
    method = done.stdout.strip().rpartition(" by ")[2] or "unknown"
    return corners[used], inverse.reshape(-1, 3), used, method
