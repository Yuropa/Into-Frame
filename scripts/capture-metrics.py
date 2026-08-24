#!/usr/bin/env python3
"""
Extract the per-capture regression metrics from a .debug bundle.

    python scripts/capture-metrics.py <name>.debug                    # table
    python scripts/capture-metrics.py <name>.debug --json out.json    # machine-readable
    python scripts/capture-metrics.py <name>.debug --compare base.json

Every number here is tied to a specific failure that has actually shipped, and every
one is cheap enough to compute that there is no excuse for tuning a threshold against
a single capture again:

  relief_inflation    post-refinement height span / measured height span. Terrain
                      Noise Refinement invents relief on every capture, but a flat
                      river scene reconstructed into 71 m of hills and a mountain
                      meadow gaining 35 m look identical in absolute metres. The
                      ratio is what separates them.
  sky_on_ground       fraction of terrain-mesh vertices whose panorama UV lands on a
                      SKY-typed pixel, i.e. ground and water painted with sky. Note
                      this is NOT "UV v < 0.5": ground legitimately rises above the
                      camera (half of Rainier's vertices do), and testing v instead
                      of the sky mask condemns the correct captures with the broken
                      ones.
  seams               straight axis-aligned label boundaries in the nadir half of the
                      panorama region map -- the per-tile segmentation failing to
                      reconcile across tile edges. Counted as columns/rows where over
                      a quarter of the perpendicular runs change label at that exact
                      index, which no organic boundary does.
  water_y             percentiles of the water mesh's own Y. A sea draped up a cliff
                      and a level one have the same vertex count and the same
                      shoreline; only the distribution tells them apart.
  region              composition of the refined ground region map. Water vanishing
                      from a river capture shows up here and nowhere else.
  gates               category-mesh rejections by which gate fired, read from
                      mesh_geometry_debug.json where Scene Generation wrote one.
  textured_meshes     category meshes carrying an image rather than vertex colour.

Bundle layout is the one archive.py writes: <bundle>/manifest.json naming each
sample, and <bundle>/context/<uuid>/<Stage Name>/ holding that stage's artifacts.
Missing stages are reported as None rather than failing, so a partial re-run still
produces a usable row.
"""
from __future__ import annotations

import argparse
import collections
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Mirrors RegionType in pipeline/panorama_segmentation/panorama_region_result.py.
# Duplicated rather than imported because that import pulls in torch, and this script
# has to run on a laptop against a bundle copied off the GPU box.
REGION_NAMES = {
    0: "sky", 1: "water", 2: "terrain", 3: "ground", 4: "vegetation",
    5: "built", 6: "other", 7: "road", 8: "trail",
}
SKY = 0


# ── glTF ─────────────────────────────────────────────────────────────────────

def _glb_chunks(path: Path) -> tuple[dict, bytes | None]:
    data = path.read_bytes()
    offset, js, binary = 12, None, None
    while offset < len(data):
        length, kind = struct.unpack("<II", data[offset:offset + 8])
        chunk = data[offset + 8:offset + 8 + length]
        if kind == 0x4E4F534A:
            js = json.loads(chunk)
        else:
            binary = chunk
        offset += 8 + length
    return js, binary


_COMPONENT = {5126: "<f4", 5123: "<u2", 5125: "<u4", 5121: "<u1"}
_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _accessor(js: dict, binary: bytes, index: int) -> np.ndarray:
    acc = js["accessors"][index]
    view = js["bufferViews"][acc["bufferView"]]
    dtype = _COMPONENT[acc["componentType"]]
    width = _COUNT[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride")
    if stride and stride != np.dtype(dtype).itemsize * width:
        # Interleaved: walk it row by row rather than reinterpreting the block.
        rows = [
            np.frombuffer(binary, dtype=dtype, count=width, offset=start + i * stride)
            for i in range(acc["count"])
        ]
        return np.stack(rows)
    return np.frombuffer(
        binary, dtype=dtype, count=acc["count"] * width, offset=start
    ).reshape(-1, width)


def _primitive(path: Path) -> tuple[dict, bytes, dict]:
    js, binary = _glb_chunks(path)
    return js, binary, js["meshes"][0]["primitives"][0]


# ── metrics ──────────────────────────────────────────────────────────────────

def _span(path: Path) -> float | None:
    if not path.exists():
        return None
    a = np.load(path, mmap_mode="r")
    finite = np.isfinite(a)
    if not finite.any():
        return None
    return float(np.nanmax(a) - np.nanmin(a))


def relief_inflation(ctx: Path) -> dict:
    measured = _span(ctx / "Height Map" / "height_map.npy")
    refined = _span(ctx / "Terrain Noise Refinement" / "height_map.npy")
    factor = None
    if measured and refined and measured > 1e-6:
        factor = refined / measured
    return {"measured_m": measured, "refined_m": refined, "factor": factor}


def _sky_masks(ctx: Path) -> dict[str, np.ndarray]:
    """Both sky masks, because they are not the same mask.

    panorama_sky_mask (from Panorama Depth) is what Panorama.mesh_uvs consumes to
    keep terrain UVs off the sky. panorama_region_type_map's SKY class is what the
    region/height stages reason about. Measured across the four landscape captures
    they disagree by 1.9% (Paris) to 12.4% (Iceland), and reporting only the second
    is how a sky-on-ground figure of 4.16% got attributed to Shark Fin, whose rate
    against the mask that actually gates the guard is 0.18%. Report both or the
    number is not interpretable.
    """
    masks: dict[str, np.ndarray] = {}
    depth_sky = ctx / "Panorama Depth" / "panorama_sky_mask.json"
    if depth_sky.exists():
        try:
            masks["sky_mask"] = np.asarray(json.load(open(depth_sky)), dtype=bool)
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    types = ctx / "Panorama Regions" / "panorama_region_type_map.npy"
    if types.exists():
        masks["region_map"] = np.asarray(np.load(types, mmap_mode="r")).astype(int) == SKY
    return masks


def sky_on_ground(ctx: Path) -> dict:
    mesh = ctx / "Terrain Mesh" / "terrain_mesh.glb"
    masks = _sky_masks(ctx)
    empty = {"vs_sky_mask": None, "vs_region_map": None, "mask_disagreement": None,
             "highest_sky_vertex_m": None}
    if not mesh.exists() or not masks:
        return empty

    js, binary, prim = _primitive(mesh)
    if "TEXCOORD_0" not in prim["attributes"]:
        return empty
    uv = _accessor(js, binary, prim["attributes"]["TEXCOORD_0"])
    pos = _accessor(js, binary, prim["attributes"]["POSITION"])

    out = dict(empty)
    for name, mask in masks.items():
        height, width = mask.shape
        col = np.clip((uv[:, 0] * (width - 1)).astype(int), 0, width - 1)
        row = np.clip((uv[:, 1] * (height - 1)).astype(int), 0, height - 1)
        hits = mask[row, col]
        out[f"vs_{name}"] = float(hits.mean())
        if name == "sky_mask" and hits.any():
            out["highest_sky_vertex_m"] = float(pos[hits, 1].max())

    a, b = masks.get("sky_mask"), masks.get("region_map")
    if a is not None and b is not None and a.shape == b.shape:
        out["mask_disagreement"] = float(np.mean(a != b))
    return out


def stored_uv_on_sky(ctx: Path) -> dict | None:
    """How many observed cells' STORED panorama UVs land on sky.

    terrain_generator prefers these over the sky-corrected UVs from mesh_uvs for
    every observed vertex, and nothing applies the sky rescale to them -- so this is
    the rate at which the guard is bypassed at source. Not the same quantity as the
    final-mesh rate above (downstream fixups clean some of it up), which is exactly
    why both are worth having.
    """
    masks = _sky_masks(ctx)
    mask = masks.get("sky_mask")
    if mask is None:
        return None
    hm = ctx / "Height Map"
    try:
        u = np.asarray(np.load(hm / "height_map_pano_u.npy", mmap_mode="r"))
        v = np.asarray(np.load(hm / "height_map_pano_v.npy", mmap_mode="r"))
        observed = np.asarray(np.load(hm / "height_map_observed_mask.npy", mmap_mode="r")) > 0.5
    except (FileNotFoundError, OSError):
        return None
    if not observed.any():
        return {"observed_cells": 0, "on_sky": None}

    uu, vv = u[observed], v[observed]
    valid = np.isfinite(uu) & np.isfinite(vv)
    if not valid.any():
        return {"observed_cells": int(observed.sum()), "on_sky": None}
    height, width = mask.shape
    col = np.clip((uu[valid] * (width - 1)).astype(int), 0, width - 1)
    row = np.clip((vv[valid] * (height - 1)).astype(int), 0, height - 1)
    return {"observed_cells": int(observed.sum()),
            "on_sky": float(mask[row, col].mean())}


def region_composition(ctx: Path) -> dict | None:
    path = ctx / "Region Map Refinement" / "region_map.npy"
    if not path.exists():
        path = ctx / "Region Map" / "region_map.npy"
    if not path.exists():
        return None
    a = np.load(path, mmap_mode="r")
    counts = collections.Counter(np.asarray(a).astype(int).ravel().tolist())
    total = a.size
    return {REGION_NAMES.get(k, str(k)): v / total for k, v in sorted(counts.items())}


def seam_count(ctx: Path, threshold: float = 0.25) -> dict | None:
    """Straight label edges in the nadir half, as a tile-seam proxy.

    An organic region boundary is never straight for a quarter of the image; a tile
    edge always is. Restricted to the lower half because that is where the artifact
    lives -- the sky/horizon half segments cleanly on every capture measured.
    """
    path = ctx / "Panorama Regions" / "panorama_region_type_map.npy"
    if not path.exists():
        return None
    a = np.asarray(np.load(path, mmap_mode="r")).astype(np.uint8)
    lower = a[a.shape[0] // 2:, :]
    vertical = (lower[:, 1:] != lower[:, :-1]).mean(axis=0)
    horizontal = (lower[1:, :] != lower[:-1, :]).mean(axis=1)
    return {
        "vertical": int((vertical > threshold).sum()),
        "horizontal": int((horizontal > threshold).sum()),
        "columns": np.where(vertical > threshold)[0].tolist()[:16],
    }


def water_profile(ctx: Path) -> dict | None:
    path = ctx / "Terrain Mesh" / "water_mesh.glb"
    if not path.exists():
        return None
    js, binary, prim = _primitive(path)
    y = _accessor(js, binary, prim["attributes"]["POSITION"])[:, 1]
    p = np.percentile(y, [0, 25, 50, 75, 90, 99, 100])
    return {
        "vertices": int(len(y)),
        "p0": float(p[0]), "p25": float(p[1]), "p50": float(p[2]),
        "p75": float(p[3]), "p90": float(p[4]), "p99": float(p[5]),
        "max": float(p[6]),
        "above_5m": float((y > 5.0).mean()),
    }


def scene_objects(ctx: Path) -> dict | None:
    files = sorted((ctx / "Scene Generation").glob("metadata_*.json"))
    if not files:
        return None
    classes = collections.Counter()
    for f in files:
        try:
            classes[json.load(open(f)).get("class")] += 1
        except (json.JSONDecodeError, OSError):
            continue
    return {"total": sum(classes.values()), "classes": dict(classes.most_common())}


def mesh_gates(ctx: Path) -> dict | None:
    """Rejections by gate, from whichever stage directory wrote the debug file."""
    for stage in ("Scene Generation", "Panorama Asset Generation"):
        path = ctx / stage / "mesh_geometry_debug.json"
        if not path.exists():
            continue
        try:
            doc = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        tally = collections.Counter()
        placed = rejected = 0
        for rec in doc.get("meshes", {}).values():
            placed += rec.get("instances", 0) - rec.get("rejected", 0)
            rejected += rec.get("rejected", 0)
            for gate, n in (rec.get("gates") or {}).items():
                tally[gate] += n
        return {"placed": placed, "rejected": rejected, "by_gate": dict(tally),
                "thresholds": doc.get("thresholds")}
    return None


def textured_meshes(ctx: Path) -> dict | None:
    globbed = sorted((ctx / "Panorama Asset Generation").glob("category_mesh_*.glb"))
    meshes = [p for p in globbed if not p.stem.endswith("_card")]
    if not meshes:
        return None
    textured = 0
    for path in meshes:
        try:
            js, _ = _glb_chunks(path)
        except (OSError, struct.error, json.JSONDecodeError):
            continue
        if js.get("images"):
            textured += 1
    return {"textured": textured, "total": len(meshes)}


def collect(ctx: Path) -> dict:
    return {
        "relief": relief_inflation(ctx),
        "sky_on_ground": sky_on_ground(ctx),
        "stored_uv": stored_uv_on_sky(ctx),
        "region": region_composition(ctx),
        "seams": seam_count(ctx),
        "water": water_profile(ctx),
        "scene": scene_objects(ctx),
        "gates": mesh_gates(ctx),
        "textures": textured_meshes(ctx),
    }


# ── reporting ────────────────────────────────────────────────────────────────

def _fmt(value, spec="{:.2f}", none="—"):
    return none if value is None else spec.format(value)


def report(results: dict[str, dict]) -> None:
    names = list(results)
    width = max(22, *(len(n) for n in names)) if names else 22

    def row(label, cells):
        print(f"  {label:<28}" + "".join(f"{c:>{min(width, 14)}}" for c in cells))

    print("\nTerrain")
    row("height span (m)", [_fmt(r["relief"]["measured_m"], "{:.1f}") for r in results.values()])
    row("after refinement (m)", [_fmt(r["relief"]["refined_m"], "{:.1f}") for r in results.values()])
    row("relief inflation", [_fmt(r["relief"]["factor"], "{:.2f}x") for r in results.values()])
    row("sky on ground (sky mask)",
        [_fmt(r["sky_on_ground"]["vs_sky_mask"], "{:.2%}") for r in results.values()])
    row("sky on ground (region map)",
        [_fmt(r["sky_on_ground"]["vs_region_map"], "{:.2%}") for r in results.values()])
    row("  the two masks differ by",
        [_fmt(r["sky_on_ground"]["mask_disagreement"], "{:.1%}") for r in results.values()])
    row("stored UV on sky",
        [_fmt((r["stored_uv"] or {}).get("on_sky"), "{:.1%}") for r in results.values()])
    row("highest sky vertex (m)",
        [_fmt(r["sky_on_ground"].get("highest_sky_vertex_m"), "{:.1f}") for r in results.values()])

    print("\nRegion map")
    row("seams (vertical)", [_fmt((r["seams"] or {}).get("vertical"), "{:d}") for r in results.values()])
    row("seams (horizontal)", [_fmt((r["seams"] or {}).get("horizontal"), "{:d}") for r in results.values()])
    for key in ("water", "terrain", "vegetation", "built", "ground"):
        row(f"{key}", [_fmt((r["region"] or {}).get(key), "{:.1%}") for r in results.values()])

    print("\nWater surface")
    row("vertices", [_fmt((r["water"] or {}).get("vertices"), "{:d}") for r in results.values()])
    row("median Y (m)", [_fmt((r["water"] or {}).get("p50"), "{:.2f}") for r in results.values()])
    row("p99 Y (m)", [_fmt((r["water"] or {}).get("p99"), "{:.2f}") for r in results.values()])
    row("above +5 m", [_fmt((r["water"] or {}).get("above_5m"), "{:.1%}") for r in results.values()])

    print("\nObjects")
    row("placed", [_fmt((r["scene"] or {}).get("total"), "{:d}") for r in results.values()])
    row("meshes placed", [_fmt((r["gates"] or {}).get("placed"), "{:d}") for r in results.values()])
    row("meshes rejected", [_fmt((r["gates"] or {}).get("rejected"), "{:d}") for r in results.values()])
    row("textured meshes", [
        "—" if not r["textures"] else f"{r['textures']['textured']}/{r['textures']['total']}"
        for r in results.values()])

    print("\n  columns: " + ", ".join(names) + "\n")

    for name, r in results.items():
        gates = (r["gates"] or {}).get("by_gate")
        if gates:
            print(f"  {name}: mesh rejections by gate — {gates}")


def compare(current: dict, baseline: dict, tolerance: float = 0.02) -> int:
    """Report drift against a saved baseline. Returns a process exit code."""
    print("\nDrift vs baseline")
    drifted = 0
    for name, cur in current.items():
        base = baseline.get(name)
        if base is None:
            print(f"  {name}: no baseline — new capture")
            continue
        notes = []

        for label, path in (("relief inflation", ("relief", "factor")),
                            ("sky on ground", ("sky_on_ground", "vs_sky_mask"))):
            a, b = cur[path[0]].get(path[1]), base.get(path[0], {}).get(path[1])
            if a is None or b is None:
                continue
            if abs(a - b) > max(tolerance, abs(b) * tolerance):
                notes.append(f"{label} {b:.3f} → {a:.3f}")

        a = (cur["scene"] or {}).get("classes", {})
        b = (base.get("scene") or {}).get("classes", {})
        for cls in sorted(set(a) | set(b)):
            if a.get(cls, 0) != b.get(cls, 0):
                notes.append(f"{cls} {b.get(cls, 0)} → {a.get(cls, 0)}")

        if notes:
            drifted += 1
            print(f"  {name}:")
            for n in notes:
                print(f"      {n}")
        else:
            print(f"  {name}: unchanged")
    return 1 if drifted else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path, help="path to a .debug bundle")
    ap.add_argument("--json", type=Path, help="write machine-readable metrics here")
    ap.add_argument("--compare", type=Path, help="baseline .json to diff against")
    ap.add_argument("--only", nargs="*", help="restrict to these capture names")
    args = ap.parse_args()

    manifest_path = args.bundle / "manifest.json"
    if not manifest_path.exists():
        print(f"No manifest.json under {args.bundle}", file=sys.stderr)
        return 2
    manifest = json.load(open(manifest_path))

    results: dict[str, dict] = {}
    for sample in manifest.get("samples", []):
        name = Path(sample.get("source_path", sample["uuid"])).stem
        if args.only and name not in args.only:
            continue
        ctx = args.bundle / "context" / sample["uuid"]
        if not ctx.is_dir():
            print(f"  {name}: no context directory, skipping", file=sys.stderr)
            continue
        results[name] = collect(ctx)

    if not results:
        print("Nothing to report", file=sys.stderr)
        return 2

    report(results)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"  wrote {args.json}")

    if args.compare:
        return compare(results, json.load(open(args.compare)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
