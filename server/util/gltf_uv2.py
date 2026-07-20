"""
Injects a second UV set (glTF `TEXCOORD_1`) into an already-exported GLB file.

trimesh's own glTF exporter (trimesh.visual.TextureVisuals) only ever writes a
single UV set (TEXCOORD_0) -- there is no supported way to hand it a second
one. mesh.vertex_attributes gets exported too, but as an application-specific
attribute (forced to a leading underscore, e.g. `_texcoord_1`, with none of
TEXCOORD_0's own V-flip applied), which generic glTF consumers -- including
GLTFast, the Unity-side importer this project uses -- have no reason to treat
as a second UV set. So a real, standards-named TEXCOORD_1 has to be added by
directly editing the GLB container after trimesh has already written it.

GLB layout (see the glTF 2.0 spec, "Binary glTF Layout"):
    12-byte header:      magic "glTF", version (uint32 LE, =2), total length
    JSON chunk:          8-byte header (length, type=b"JSON") + JSON text,
                          space-padded to a 4-byte boundary
    BIN chunk (optional): 8-byte header (length, type=b"BIN\\0") + binary
                          buffer data, zero-padded to a 4-byte boundary

trimesh's GLB export embeds every mesh's vertex/index data in exactly one
buffer with no `uri` -- i.e. buffers[0]'s bytes *are* the file's own BIN
chunk. This module appends the new UV2 data to the end of that same chunk,
adds one bufferView + accessor describing it, and points the mesh
primitive's `TEXCOORD_1` attribute at that new accessor.
"""
from __future__ import annotations
import json
import struct
from pathlib import Path

import numpy as np

_MAGIC = b"glTF"
_VERSION = 2
_CHUNK_JSON = b"JSON"
_CHUNK_BIN = b"BIN\x00"

_COMPONENT_TYPE_FLOAT = 5126


def _read_glb(data: bytes) -> tuple[dict, bytes]:
    magic, version, total_length = struct.unpack_from("<4sII", data, 0)
    if magic != _MAGIC:
        raise ValueError(f"Not a GLB file (magic={magic!r})")
    if version != _VERSION:
        raise ValueError(f"Unsupported GLB version {version}")

    offset = 12
    tree: dict | None = None
    bin_data = b""
    while offset < total_length:
        chunk_length, chunk_type = struct.unpack_from("<I4s", data, offset)
        chunk_data = data[offset + 8: offset + 8 + chunk_length]
        if chunk_type == _CHUNK_JSON:
            tree = json.loads(chunk_data.decode("utf-8"))
        elif chunk_type == _CHUNK_BIN:
            bin_data = chunk_data
        offset += 8 + chunk_length

    if tree is None:
        raise ValueError("GLB file has no JSON chunk")
    return tree, bin_data


def _pad(data: bytes, boundary: int, fill: bytes) -> bytes:
    remainder = len(data) % boundary
    if remainder == 0:
        return data
    return data + fill * (boundary - remainder)


def _write_glb(path: Path, tree: dict, bin_data: bytes) -> None:
    json_chunk = _pad(json.dumps(tree, separators=(",", ":")).encode("utf-8"), 4, b" ")
    bin_chunk = _pad(bin_data, 4, b"\x00")

    total_length = 12 + 8 + len(json_chunk) + (8 + len(bin_chunk) if bin_chunk else 0)

    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", _MAGIC, _VERSION, total_length))
        f.write(struct.pack("<I4s", len(json_chunk), _CHUNK_JSON))
        f.write(json_chunk)
        if bin_chunk:
            f.write(struct.pack("<I4s", len(bin_chunk), _CHUNK_BIN))
            f.write(bin_chunk)


def inject_texcoord1(path: "str | Path", uv2: np.ndarray) -> bool:
    """
    Add `uv2` ((N, 2) float array) to the GLB at `path` as glTF TEXCOORD_1,
    on whichever mesh primitive's POSITION accessor has N vertices.

    Returns True if injection succeeded, False if the file's structure didn't
    match what trimesh is expected to produce (single embedded buffer, no
    external URI) -- callers should treat False as "left the file alone, not
    a hard error", since the file is still perfectly valid without the extra
    UV set; only the live Unity panorama-UV path degrades to its own
    position-based fallback.
    """
    path = Path(path)
    tree, bin_data = _read_glb(path.read_bytes())

    buffers = tree.get("buffers") or []
    if len(buffers) != 1 or buffers[0].get("uri") is not None:
        return False
    if buffers[0].get("byteLength") != len(bin_data):
        return False

    accessors = tree.setdefault("accessors", [])
    buffer_views = tree.setdefault("bufferViews", [])

    n = len(uv2)
    target_accessor_idx = None
    for mesh in tree.get("meshes", []):
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            pos_idx = attrs.get("POSITION")
            if pos_idx is None or "TEXCOORD_1" in attrs:
                continue
            if accessors[pos_idx].get("count") != n:
                continue
            if target_accessor_idx is None:
                # Build the accessor/bufferView once; every matching primitive
                # (there's normally exactly one) can share it.
                uv_bytes = np.asarray(uv2, dtype="<f4").tobytes()
                padded_bin = _pad(bin_data, 4, b"\x00")
                byte_offset = len(padded_bin)
                bin_data = padded_bin + uv_bytes

                buffer_views.append({
                    "buffer": 0,
                    "byteOffset": byte_offset,
                    "byteLength": len(uv_bytes),
                })
                accessors.append({
                    "bufferView": len(buffer_views) - 1,
                    "componentType": _COMPONENT_TYPE_FLOAT,
                    "count": n,
                    "type": "VEC2",
                })
                target_accessor_idx = len(accessors) - 1
            attrs["TEXCOORD_1"] = target_accessor_idx

    if target_accessor_idx is None:
        return False

    buffers[0]["byteLength"] = len(bin_data)
    _write_glb(path, tree, bin_data)
    return True
