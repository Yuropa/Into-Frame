#!/usr/bin/env python3
"""
Report which category meshes in a .debug bundle actually carry a sway skeleton.

    python scripts/check-rigged.py <name>.debug/context/<uuid>

WindSway drives bones named SwayBone_0..2, baked in by CategoryMeshRiggingStage via
util.gltf_skin.inject_skin. If a mesh has no skin, WindSway's LateUpdate searches for
those bones, never finds them, and returns -- silently, forever. The instance renders
perfectly and simply never moves, with the scene.json still claiming it has sway params.
That failure is invisible from both ends, which is what this script is for.

A mesh key can exist in several stage directories (e.g. written unrigged by Grass Cover,
then rewritten rigged by Category Mesh Rigging). The pipeline context resolves a key to
its LAST writer in _stage_order.json, and that is the copy the asset server hands the
client -- so that is the one whose answer counts. Every copy is listed, with the winner
marked, because "rigged in one directory, unrigged in the one that wins" is a real and
otherwise baffling failure mode.
"""

import argparse
import json
import struct
import sys
from pathlib import Path


def glb_json(path: Path) -> dict | None:
    try:
        with path.open("rb") as f:
            header = f.read(12)
            if len(header) < 12 or header[:4] != b"glTF":
                return None
            chunk_len, _chunk_type = struct.unpack("<II", f.read(8))
            return json.loads(f.read(chunk_len))
    except Exception:
        return None


def describe(path: Path) -> str:
    tree = glb_json(path)
    if tree is None:
        return "unreadable"
    skins = tree.get("skins", [])
    joint_names = []
    for skin in skins:
        for joint in skin.get("joints", []):
            name = tree.get("nodes", [])[joint].get("name") if joint < len(tree.get("nodes", [])) else None
            if name:
                joint_names.append(name)
    attrs = set()
    for mesh in tree.get("meshes", []):
        for prim in mesh.get("primitives", []):
            attrs |= set(prim.get("attributes", {}))
    has_skin_attrs = {"JOINTS_0", "WEIGHTS_0"} <= attrs
    if skins and has_skin_attrs:
        return f"RIGGED   ({', '.join(joint_names[:3])})"
    if skins or has_skin_attrs:
        return f"BROKEN   (skins={len(skins)}, JOINTS_0/WEIGHTS_0={'yes' if has_skin_attrs else 'NO'})"
    return "no skin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("context_dir", type=Path, help="<name>.debug/context/<uuid>")
    ap.add_argument("--all", action="store_true", help="include meshes that aren't category assets")
    args = ap.parse_args()

    ctx: Path = args.context_dir
    order_file = ctx / "_stage_order.json"
    if not order_file.exists():
        sys.exit(f"no _stage_order.json under {ctx} — not a debug context directory?")
    order = json.loads(order_file.read_text())
    rank = {name: i for i, name in enumerate(order)}

    # mesh key -> [(stage, path)], in pipeline order
    found: dict[str, list[tuple[str, Path]]] = {}
    for stage_dir in ctx.iterdir():
        if not stage_dir.is_dir():
            continue
        for glb in stage_dir.glob("*.glb"):
            key = glb.stem
            if not args.all and not key.startswith("category_mesh_"):
                continue
            found.setdefault(key, []).append((stage_dir.name, glb))

    if not found:
        sys.exit("no category meshes found")

    unrigged_winners = []
    for key in sorted(found):
        copies = sorted(found[key], key=lambda sp: rank.get(sp[0], -1))
        winner_stage = copies[-1][0]
        print(f"\n{key}")
        for stage, path in copies:
            mark = "  <- SERVED" if stage == winner_stage else ""
            status = describe(path)
            print(f"    {stage:28s} {status}{mark}")
            if stage == winner_stage and not status.startswith("RIGGED"):
                unrigged_winners.append((key, stage, status))

    print()
    if unrigged_winners:
        print(f"{len(unrigged_winners)} mesh(es) will be served WITHOUT bones — WindSway is a no-op on these:")
        for key, stage, status in unrigged_winners:
            print(f"    {key:38s} (served from '{stage}': {status})")
        print("\nIf the stage that rigs them ran, check CategoryMeshRiggingStage's own filters:")
        print("  - the object's class must be in rig_categories (VEGETATION_CATEGORIES)")
        print("  - its metadata must have stationary=True")
        print("  - it must appear in scene.objects with a non-null source_index")
        print("  - inject_skin bails (silently) if the mesh's glTF node carries a transform")
    else:
        print("Every category mesh is served rigged — if instances still don't move, the")
        print("bones exist and the problem is client-side (WindSway not attached, or frame rate).")


if __name__ == "__main__":
    main()
