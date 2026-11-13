"""The finished object as a binary glTF 2.0 file (.glb): one mesh with its normals, its texture places and its
texture (PNG), and named empty nodes that mark places on it (a held object's "grip" and "tip", a shield's "front"),
so a game can put it in a hand: a hand's palm goes to "grip" with its fingers round the node's +Y, which points
toward the tip, and its +Z is the blade's flat side.
"""

import io
import json
import struct

import numpy as np

GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
FLOAT = 5126
UNSIGNED_INT = 5125
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
LINEAR = 9729
LINEAR_MIPMAP_LINEAR = 9987
CLAMP_TO_EDGE = 33071
# A matte surface: what the paint shows is colour, not shine.
ROUGHNESS = 0.85


def quaternion(columns):
    """A rotation matrix's quaternion (x, y, z, w), its columns the node's x, y and z axes."""
    m = np.asarray(columns, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2 * np.sqrt(trace + 1)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2 * np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2 * np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return (q / np.linalg.norm(q)).tolist()


def mark_rotation(up, facing):
    """The rotation whose +Y is `up` and whose +Z is `facing` made square to it."""
    y = np.asarray(up, dtype=np.float64)
    y /= np.linalg.norm(y)
    z = np.asarray(facing, dtype=np.float64)
    z = z - (z @ y) * y
    if np.linalg.norm(z) < 1e-9:
        z = np.array([0.0, 0.0, 1.0]) - y[2] * y
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    return quaternion(np.stack([x, y, z], 1))


class _Buffer:
    def __init__(self):
        self.data = bytearray()
        self.views = []
        self.accessors = []

    def view(self, raw, target=None):
        while len(self.data) % 4:
            self.data.append(0)
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(raw)}
        if target:
            view["target"] = target
        self.data.extend(raw)
        self.views.append(view)
        return len(self.views) - 1

    def floats(self, values, kind, bounds=False):
        values = np.ascontiguousarray(values, dtype="<f4")
        accessor = {"bufferView": self.view(values.tobytes(), ARRAY_BUFFER), "componentType": FLOAT,
                    "count": len(values), "type": kind}
        if bounds:
            accessor["min"] = values.min(0).tolist()
            accessor["max"] = values.max(0).tolist()
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def indices(self, values):
        values = np.ascontiguousarray(values, dtype="<u4").reshape(-1)
        self.accessors.append({"bufferView": self.view(values.tobytes(), ELEMENT_ARRAY_BUFFER),
                               "componentType": UNSIGNED_INT, "count": len(values), "type": "SCALAR"})
        return len(self.accessors) - 1


def png_bytes(texture):
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray((np.clip(texture, 0, 1) * 255).round().astype(np.uint8)).save(out, format="PNG", optimize=True)
    return out.getvalue()


def glb_bytes(name, corners, triangles, normals, uvs, texture, marks=None, extras=None, parts=None,
              texture_signed=None):
    """The .glb's bytes. `marks`: {name: (position, up, facing)} empty nodes under the object's node. `parts`: meshes
    our own code adds (a bow's string), each {"name", "corners", "triangles", "colour"} in one plain colour, a node
    under the object's node. `texture_signed`: the texture's PNG bytes with its content credentials in them, from its
    plain ones (credentials.signed_bytes)."""
    from .mesh_tools import corner_normals

    buffer = _Buffer()
    attributes = {"POSITION": buffer.floats(corners, "VEC3", bounds=True), "NORMAL": buffer.floats(normals, "VEC3"),
                  "TEXCOORD_0": buffer.floats(uvs, "VEC2")}
    indices = buffer.indices(triangles)
    texture_png = png_bytes(texture)
    image_view = buffer.view(texture_signed(texture_png) if texture_signed else texture_png)
    nodes = [{"name": name, "mesh": 0}]
    meshes = [{"name": name, "primitives": [{"attributes": attributes, "indices": indices, "material": 0}]}]
    materials = [{"name": f"{name} paint", "pbrMetallicRoughness": {
        "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": ROUGHNESS}}]
    for part in parts or []:
        part_attributes = {"POSITION": buffer.floats(part["corners"], "VEC3", bounds=True),
                           "NORMAL": buffer.floats(corner_normals(part["corners"], part["triangles"]), "VEC3")}
        meshes.append({"name": part["name"], "primitives": [{"attributes": part_attributes,
                                                            "indices": buffer.indices(part["triangles"]),
                                                            "material": len(materials)}]})
        materials.append({"name": part["name"], "pbrMetallicRoughness": {
            "baseColorFactor": [*map(float, part["colour"]), 1.0], "metallicFactor": 0.0,
            "roughnessFactor": ROUGHNESS}})
        nodes.append({"name": part["name"], "mesh": len(meshes) - 1})
    for mark, (position, up, facing) in (marks or {}).items():
        nodes.append({"name": mark, "translation": [float(x) for x in position],
                      "rotation": mark_rotation(up, facing)})
    if len(nodes) > 1:
        nodes[0]["children"] = list(range(1, len(nodes)))
    document = {
        "asset": {"version": "2.0", "generator": "model-maker"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": LINEAR, "minFilter": LINEAR_MIPMAP_LINEAR, "wrapS": CLAMP_TO_EDGE,
                      "wrapT": CLAMP_TO_EDGE}],
        "images": [{"bufferView": image_view, "mimeType": "image/png"}],
        "bufferViews": buffer.views,
        "accessors": buffer.accessors,
        "buffers": [{"byteLength": len(buffer.data)}],
    }
    if extras:
        document["asset"]["extras"] = extras
    text = json.dumps(document, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    binary = bytes(buffer.data) + b"\0" * (-len(buffer.data) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    return (struct.pack("<III", GLB_MAGIC, 2, total) + struct.pack("<II", len(text), JSON_CHUNK) + text
            + struct.pack("<II", len(binary), BIN_CHUNK) + binary)


def read_document(data):
    """A .glb's JSON and binary chunk (for tests and checks)."""
    magic, version, total = struct.unpack("<III", data[:12])
    if magic != GLB_MAGIC or version != 2 or total != len(data):
        raise ValueError("not a glTF 2.0 binary")
    length, kind = struct.unpack("<II", data[12:20])
    if kind != JSON_CHUNK:
        raise ValueError("the first chunk is not JSON")
    document = json.loads(data[20:20 + length])
    at = 20 + length
    blength, bkind = struct.unpack("<II", data[at:at + 8])
    if bkind != BIN_CHUNK:
        raise ValueError("the second chunk is not binary")
    return document, data[at + 8:at + 8 + blength]


def accessor_values(document, binary, index):
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    dtype = "<f4" if accessor["componentType"] == FLOAT else "<u4"
    raw = binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]
    return np.frombuffer(raw, dtype=dtype).reshape(accessor["count"], width) if width > 1 else \
        np.frombuffer(raw, dtype=dtype)
