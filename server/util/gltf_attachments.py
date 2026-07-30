"""
Recover the parts of a GLB that a trimesh round-trip silently discards.

Mesh stores its geometry as a trimesh.Trimesh, and trimesh models exactly one UV set
and no skinning at all. Everything Mesh.save() injects on top -- the TEXCOORD_1
panorama UV (util.gltf_uv2) and the sway skeleton (util.gltf_skin) -- therefore exists
ONLY in the exported file and only because `extra_uv` / `skin_bone_heights` happened to
be set in memory at export time.

That is fine within one process. It fails across a resume: PipelineContext loads a
cached mesh from disk with Mesh.load(), trimesh drops the skin and TEXCOORD_1 on the
floor, the in-memory Mesh comes back with those attributes at None, and anything that
re-exports it writes a file missing both. The asset server does exactly that on a cache
miss, so on any resumed run the client was served meshes with no bones (WindSway then
searches for SwayBone_0 forever and every plant stands perfectly still) and terrain with
no TEXCOORD_1 (TerrainSplat.shader's panoUV, so the panorama layer samples one texel).

The file itself was always correct -- inspecting the GLB on disk showed the skin present,
which is why this hid behind every check that read the artifact rather than what the
client received. Reading the attachments back out of the GLB keeps the file as the single
source of truth, with no sidecars to keep in step and nothing added to the .meta schema.
"""

import json
import struct
from pathlib import Path

import numpy as np

from util.gltf_uv2 import _read_glb

_COMPONENT_TYPE_FLOAT = 5126

# Bytes per component, by glTF componentType.
_COMPONENT_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _read_float_accessor(tree: dict, bin_data: bytes, accessor: dict) -> "np.ndarray | None":
    """Decode a float accessor into (count, components). None if it isn't float, is
    sparse, or lives outside the BIN chunk."""
    if accessor.get("componentType") != _COMPONENT_TYPE_FLOAT:
        return None
    if "sparse" in accessor or "bufferView" not in accessor:
        return None
    components = _TYPE_COMPONENTS.get(accessor.get("type"))
    if components is None:
        return None

    view = tree["bufferViews"][accessor["bufferView"]]
    element = components * _COMPONENT_SIZE[_COMPONENT_TYPE_FLOAT]
    stride = view.get("byteStride") or element
    base = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"]

    if base + (count - 1) * stride + element > len(bin_data):
        return None

    if stride == element:
        # Tightly packed, which is what every accessor this module reads actually is
        # (both inject_texcoord1 and trimesh's exporter write contiguous buffers).
        # Worth special-casing: the generic path below is a Python loop, and terrain
        # carries 40k+ vertices that get decoded on every cache load.
        return np.frombuffer(
            bin_data, dtype="<f4", count=count * components, offset=base
        ).reshape(count, components).astype(np.float32, copy=True)

    out = np.empty((count, components), dtype=np.float32)
    fmt = f"<{components}f"
    for i in range(count):
        out[i] = struct.unpack_from(fmt, bin_data, base + i * stride)
    return out


def _first_primitive_with(tree: dict, attribute: str) -> "dict | None":
    for mesh in tree.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if attribute in primitive.get("attributes", {}):
                return primitive
    return None


def read_texcoord1(path: Path) -> "np.ndarray | None":
    """The (N, 2) TEXCOORD_1 array injected by gltf_uv2.inject_texcoord1, returned in
    Mesh.extra_uv's own convention, or None.

    inject_texcoord1 writes `file_v = 1 - v` (see its comment on cancelling trimesh's
    exporter flip), so the raw file value is NOT what extra_uv held. Undo it here, or a
    load -> save round-trip flips V a second time and lands the panorama upside down --
    which is worse than the missing-UV bug this recovery exists to fix, because it looks
    plausible instead of blank. Verified: with the un-flip, read(save(x)) == x exactly.
    """
    try:
        tree, bin_data = _read_glb(Path(path).read_bytes())
    except Exception:
        return None

    primitive = _first_primitive_with(tree, "TEXCOORD_1")
    if primitive is None:
        return None
    accessor = tree["accessors"][primitive["attributes"]["TEXCOORD_1"]]
    uv = _read_float_accessor(tree, bin_data, accessor)
    if uv is None or uv.shape[1] != 2:
        return None

    uv = uv.copy()
    uv[:, 1] = 1.0 - uv[:, 1]
    return uv


def read_skin(path: Path) -> "tuple[list[float], list[str]] | None":
    """The (bone_heights, bone_names) that gltf_skin.inject_skin encoded, or None.

    inject_skin writes each joint as a pure +Y translation RELATIVE to its parent in a
    single chain, so the absolute heights it was given are the running sum -- exactly
    inverting `local_y = bone_heights[i] - bone_heights[i - 1]`.
    """
    try:
        tree, _ = _read_glb(Path(path).read_bytes())
    except Exception:
        return None

    skins = tree.get("skins") or []
    nodes = tree.get("nodes") or []
    if not skins:
        return None

    joints = skins[0].get("joints") or []
    if not joints:
        return None

    heights: list[float] = []
    names: list[str] = []
    running = 0.0
    for index in joints:
        if not (0 <= index < len(nodes)):
            return None
        node = nodes[index]
        # A joint carrying rotation/scale/matrix wasn't produced by inject_skin, so the
        # running-sum inversion doesn't hold and guessing would be worse than declining.
        if any(key in node for key in ("rotation", "scale", "matrix")):
            return None
        running += float((node.get("translation") or (0.0, 0.0, 0.0))[1])
        heights.append(running)
        names.append(node.get("name") or f"SwayBone_{len(names)}")

    return heights, names
