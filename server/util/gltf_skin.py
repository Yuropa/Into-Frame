"""
Injects a simple vertical skinned bone chain (glTF `skins` / `JOINTS_0` /
`WEIGHTS_0` / `inverseBindMatrices`) into an already-exported GLB file, for a
gentle procedural sway animation (wind on foliage) driven at runtime.

trimesh's own glTF exporter has no notion of skinning at all, so -- exactly
like util.gltf_uv2.inject_texcoord1 next to this file -- the GLB container has
to be hand-edited after trimesh has already written it. See that module's own
docstring for the GLB chunk layout; this reuses its chunk read/write helpers
rather than duplicating them.

The chain is deliberately simple: N joints stacked along +Y (base -> tip), each
a child of the previous one, with an identity bind pose (no rotation) -- so
each joint's global bind transform is just a translation, and every vertex's
skin weight blends linearly between its two nearest joints by height alone.
Rotating a joint at runtime (Unity's WindSway component) bends everything
above it in the chain, giving a smooth continuous curve rather than a faceted
per-segment bend.
"""
from __future__ import annotations
import struct
from pathlib import Path

import numpy as np

from util.gltf_uv2 import _read_glb, _write_glb, _pad, _COMPONENT_TYPE_FLOAT

_COMPONENT_TYPE_UNSIGNED_BYTE = 5121


def _read_position_y(tree: dict, bin_data: bytes, pos_accessor: dict) -> "np.ndarray | None":
    """Decode just the Y component of an existing POSITION accessor -- used to
    compute per-vertex skin weights against the bone heights below."""
    if pos_accessor.get("componentType") != _COMPONENT_TYPE_FLOAT or pos_accessor.get("type") != "VEC3":
        return None
    bv = tree["bufferViews"][pos_accessor["bufferView"]]
    stride = bv.get("byteStride") or 12
    base = bv.get("byteOffset", 0) + pos_accessor.get("byteOffset", 0)
    count = pos_accessor["count"]

    ys = np.empty(count, dtype=np.float64)
    for i in range(count):
        _, y, _ = struct.unpack_from("<3f", bin_data, base + i * stride)
        ys[i] = y
    return ys


def _vertex_skin_data(vertex_y: np.ndarray, bone_heights: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex (joints, weights), each (N, 4) -- blends every vertex between
    the two nearest bones by height, weight to the other two padding slots
    left at 0 so they're harmless regardless of which joint index they name."""
    n = len(vertex_y)
    joints = np.zeros((n, 4), dtype=np.uint8)
    weights = np.zeros((n, 4), dtype=np.float32)
    heights = np.asarray(bone_heights, dtype=np.float64)

    for i, y in enumerate(vertex_y):
        if y <= heights[0]:
            weights[i, 0] = 1.0
            continue
        if y >= heights[-1]:
            joints[i, 0] = len(heights) - 1
            weights[i, 0] = 1.0
            continue
        seg = int(np.searchsorted(heights, y)) - 1
        seg = max(0, min(seg, len(heights) - 2))
        span = heights[seg + 1] - heights[seg]
        t = float((y - heights[seg]) / span) if span > 0 else 0.0
        joints[i, 0] = seg
        joints[i, 1] = seg + 1
        weights[i, 0] = 1.0 - t
        weights[i, 1] = t

    return joints, weights


def _inverse_bind_matrix(bind_y: float) -> np.ndarray:
    """Column-major (glTF's own storage order) 4x4 inverse bind matrix for a
    joint whose bind pose is a pure +Y translation -- exactly what the
    identity-rotation vertical chain built by inject_skin produces, so the
    inverse is just the negated translation."""
    m = np.identity(4, dtype="<f4")
    m[1, 3] = -bind_y
    return m.T  # tobytes() below serializes this view in column-major order


def inject_skin(
    path: "str | Path",
    bone_heights: list[float],
    bone_names: "list[str] | None" = None,
) -> bool:
    """
    Add a skin with len(bone_heights) joints (base-to-tip, ascending Y) to the
    GLB at `path`, on whichever mesh primitive's POSITION accessor has no
    JOINTS_0 attribute yet.

    Returns True if injection succeeded, False if the file's structure didn't
    match what trimesh is expected to produce, or there was no eligible
    primitive -- callers should treat False as "left the file alone, not a
    hard error", same convention as inject_texcoord1.
    """
    if len(bone_heights) < 2:
        return False

    path = Path(path)
    tree, bin_data = _read_glb(path.read_bytes())

    buffers = tree.get("buffers") or []
    if len(buffers) != 1 or buffers[0].get("uri") is not None:
        return False
    if buffers[0].get("byteLength") != len(bin_data):
        return False

    scenes = tree.get("scenes") or []
    scene_idx = tree.get("scene", 0)
    if not scenes or scene_idx >= len(scenes):
        return False

    accessors = tree.setdefault("accessors", [])
    buffer_views = tree.setdefault("bufferViews", [])
    nodes = tree.setdefault("nodes", [])
    skins = tree.setdefault("skins", [])
    scene_nodes = scenes[scene_idx].setdefault("nodes", [])
    meshes = tree.get("meshes", [])

    target_node = None
    target_prim = None
    for node in nodes:
        mesh_idx = node.get("mesh")
        if mesh_idx is None:
            continue
        for prim in meshes[mesh_idx].get("primitives", []):
            attrs = prim.get("attributes", {})
            if attrs.get("POSITION") is None or "JOINTS_0" in attrs:
                continue
            target_node, target_prim = node, prim
            break
        if target_prim is not None:
            break

    if target_node is None or target_prim is None:
        return False

    # The bone chain's local Y offsets are measured directly off mesh.vertices
    # (see CategoryMeshRiggingStage), which only lines up with the exported
    # file if the mesh's own node applies no additional transform -- true for
    # Mesh.save()'s bare trimesh.Trimesh export, but checked defensively.
    if any(k in target_node for k in ("translation", "rotation", "scale", "matrix")):
        return False

    pos_accessor = accessors[target_prim["attributes"]["POSITION"]]
    n = pos_accessor["count"]
    vertex_y = _read_position_y(tree, bin_data, pos_accessor)
    if vertex_y is None or len(vertex_y) != n:
        return False

    joints_arr, weights_arr = _vertex_skin_data(vertex_y, bone_heights)

    padded_bin = _pad(bin_data, 4, b"\x00")
    joints_bytes = joints_arr.tobytes()
    joints_offset = len(padded_bin)
    bin_data = padded_bin + joints_bytes
    buffer_views.append({"buffer": 0, "byteOffset": joints_offset, "byteLength": len(joints_bytes)})
    accessors.append({
        "bufferView": len(buffer_views) - 1,
        "componentType": _COMPONENT_TYPE_UNSIGNED_BYTE,
        "count": n,
        "type": "VEC4",
    })
    joints_accessor_idx = len(accessors) - 1

    padded_bin = _pad(bin_data, 4, b"\x00")
    weights_bytes = weights_arr.astype("<f4").tobytes()
    weights_offset = len(padded_bin)
    bin_data = padded_bin + weights_bytes
    buffer_views.append({"buffer": 0, "byteOffset": weights_offset, "byteLength": len(weights_bytes)})
    accessors.append({
        "bufferView": len(buffer_views) - 1,
        "componentType": _COMPONENT_TYPE_FLOAT,
        "count": n,
        "type": "VEC4",
    })
    weights_accessor_idx = len(accessors) - 1

    target_prim["attributes"]["JOINTS_0"] = joints_accessor_idx
    target_prim["attributes"]["WEIGHTS_0"] = weights_accessor_idx

    names = bone_names or [f"SwayBone_{i}" for i in range(len(bone_heights))]
    base_node_idx = len(nodes)
    joint_indices = []
    for i, name in enumerate(names):
        local_y = bone_heights[i] - (bone_heights[i - 1] if i > 0 else 0.0)
        nodes.append({"name": name, "translation": [0.0, float(local_y), 0.0]})
        joint_indices.append(base_node_idx + i)
        if i > 0:
            nodes[base_node_idx + i - 1].setdefault("children", []).append(base_node_idx + i)

    padded_bin = _pad(bin_data, 4, b"\x00")
    ibm_bytes = b"".join(_inverse_bind_matrix(y).tobytes() for y in bone_heights)
    ibm_offset = len(padded_bin)
    bin_data = padded_bin + ibm_bytes
    buffer_views.append({"buffer": 0, "byteOffset": ibm_offset, "byteLength": len(ibm_bytes)})
    accessors.append({
        "bufferView": len(buffer_views) - 1,
        "componentType": _COMPONENT_TYPE_FLOAT,
        "count": len(bone_heights),
        "type": "MAT4",
    })
    ibm_accessor_idx = len(accessors) - 1

    skins.append({
        "joints": joint_indices,
        "inverseBindMatrices": ibm_accessor_idx,
        "skeleton": joint_indices[0],
    })
    target_node["skin"] = len(skins) - 1
    scene_nodes.append(base_node_idx)

    buffers[0]["byteLength"] = len(bin_data)
    _write_glb(path, tree, bin_data)
    return True
