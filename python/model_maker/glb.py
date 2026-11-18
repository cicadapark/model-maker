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


# The gap between the two of a pair: this share of the object's width, at least this many metres.
PAIR_GAP_SHARE = 0.15
PAIR_GAP_M = 0.02


# A footprint's arch: the inner side of a shoe curves in along the middle third of its length, the outer side
# barely. A difference of at least this share of the length tells the inner side.
ARCH_SHARE = 0.005


def arch_side(corners, axis):
    """Which side (+1 or -1 along `axis`) a shoe's inner side, its arch, faces: the footprint (its lowest 8 %) inset
    more along the middle third of its length on that side; 0 when the footprint does not tell (a glove, a gauntlet)."""
    length = 2 if axis == 0 else 0
    height = float(np.ptp(corners[:, 1]))
    bottom = corners[corners[:, 1] <= corners[:, 1].min() + 0.08 * height]
    low, high = float(bottom[:, length].min()), float(bottom[:, length].max())
    middle = bottom[(bottom[:, length] > low + (high - low) / 3) & (bottom[:, length] < high - (high - low) / 3)]
    if len(middle) < 3:
        return 0
    # The middle's edges by its 5th and 95th percentiles, so a few stray points do not decide.
    inset_low = float(np.percentile(middle[:, axis], 5) - bottom[:, axis].min())
    inset_high = float(bottom[:, axis].max() - np.percentile(middle[:, axis], 95))
    if abs(inset_high - inset_low) < ARCH_SHARE * (high - low):
        return 0
    return 1 if inset_high > inset_low else -1


# A thing this flat (its thinner floor width under this share of its other sizes, a dagger standing on its point) is
# placed beside its pair in its own plane, not face to face: two daggers 2 cm apart flat against flat read as one.
FLAT_SHARE = 1 / 3
# Things a footprint tells left from right by its arch.
FOOTWEAR = ("boot", "shoe", "clog", "sandal", "slipper", "sabaton", "sneaker")


def pair_node(name, corners, pair, footwear=False):
    """The second of a pair (`pair`: "mirror" or "copy"): the same mesh placed again beside the first, a gap between
    them. Side by side across the object's narrower width on the floor (x or z): a boot's pair stands beside it, toes
    the same way, not heel to heel (the first pair made, 16 Nov 2026, was placed along the boot's length); a round
    thing's along x. A mirror is the mesh scaled by -1 across that same axis (glTF turns a mirrored node's faces
    itself), so the two are a left and a right; it stands on the side the made one's arch faces, so the two inner
    sides face each other (wooden clogs were placed the other way round, a left on the right, 16 Nov 2026)."""
    widths = np.ptp(corners[:, [0, 2]], axis=0)
    axis = 0 if widths[0] <= widths[1] * 1.1 else 2
    other = 2 if axis == 0 else 0
    if widths[[0, 2].index(axis)] < FLAT_SHARE * min(widths[[0, 2].index(other)], float(np.ptp(corners[:, 1]))):
        axis = other
    low, high = float(np.min(corners[:, axis])), float(np.max(corners[:, axis]))
    gap = max(PAIR_GAP_SHARE * (high - low), PAIR_GAP_M)
    scale, shift = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
    if pair == "mirror":
        scale[axis] = -1.0
        if footwear and arch_side(corners, axis) < 0:
            # v -> -v + shift: the mirror's far side (-low + shift) lies the gap before the first's near side.
            shift[axis] = 2 * low - gap
        else:
            # v -> -v + shift: the mirror's near side (-high + shift) lies the gap past the first's far side.
            shift[axis] = 2 * high + gap
        return {"name": f"{name} (its mirror image)", "mesh": 0, "scale": scale, "translation": shift}
    shift[axis] = high - low + gap
    return {"name": f"{name} (a copy)", "mesh": 0, "translation": shift}


def glb_bytes(name, corners, triangles, normals, uvs, texture, marks=None, extras=None, parts=None,
              texture_signed=None, pair="", pair_footwear=False):
    """The .glb's bytes. `marks`: {name: (position, up, facing)} empty nodes under the object's node. `parts`: meshes
    our own code adds (a bow's string), each {"name", "corners", "triangles", "colour"} in one plain colour, a node
    under the object's node. `texture_signed`: the texture's PNG bytes with its content credentials in them, from its
    plain ones (credentials.signed_bytes). `pair`: "mirror" or "copy" puts the object in the scene twice
    (`pair_node`): one made and checked, the second the same mesh placed again."""
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
        "scenes": [{"nodes": [0] + ([len(nodes)] if pair else [])}],
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
    if pair:
        second = pair_node(name, np.asarray(corners), pair, pair_footwear)
        # The second has the first's marks and parts too (a dagger's grip, a bow's string), as its own children.
        copies = [{k: v for k, v in nodes[child].items()} for child in nodes[0].get("children", [])]
        nodes.append(second)
        if copies:
            second["children"] = list(range(len(nodes), len(nodes) + len(copies)))
            for copy in copies:
                copy["name"] = f"{copy['name']} (pair)"
                nodes.append(copy)
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
