#!/usr/bin/env python3
"""
Summarise a generated scene.json: what got placed, and what will actually move.

    python scripts/scene-summary.py path/to/scene.json

Answers the two questions that are otherwise guesswork from inside the headset --
"why isn't this animated?" and "why did that fall?" -- by classifying every object
into exactly one animation state:

  sway     mesh with sway params  -> WindSway drives the baked bone chain
  video    billboard with a clip  -> AnimatedBillboardVideo swaps in the video
  physics  Rigidbody handoff      -> falls under gravity from a server velocity
  STATIC   none of the above      -> renders, never moves

A billboard can only ever be `video` or STATIC: SceneAnimationStage attaches sway to
meshes only, because a billboard is a camera-facing quad with no skeleton to drive.
So a population that renders as billboards is a population that does not move.

Takes scene.json directly, or a .debug bundle's context directory (it will find
'Scene Animation/scene.json' inside).
"""

import argparse
import collections
import json
import math
from pathlib import Path


def load_scene(path: Path) -> dict:
    if path.is_dir():
        candidate = path / "Scene Animation" / "scene.json"
        if not candidate.exists():
            raise SystemExit(f"no 'Scene Animation/scene.json' under {path}")
        path = candidate
    return json.loads(path.read_text())


def animation_state(obj: dict) -> str:
    if obj.get("physics"):
        return "physics"
    if obj.get("videoColor") or obj.get("video_color"):
        return "video"
    if obj.get("sway"):
        return "sway"
    return "STATIC"


def asset_of(obj: dict) -> str:
    """The asset an object renders, with per-instance numbering stripped so that
    thousands of instances of one mesh collapse to a single row."""
    name = obj.get("mesh") or obj.get("texture") or obj.get("name") or "?"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path, help="scene.json, or a .debug context directory")
    ap.add_argument("--limit", type=int, default=25, help="rows per table (default 25)")
    args = ap.parse_args()

    scene = load_scene(args.scene)
    objects = scene.get("objects") or []
    if not objects:
        raise SystemExit("scene has no objects")

    print(f"{len(objects)} object(s)")
    print(f"  skybox={scene.get('skybox')}  terrainCenterY={scene.get('terrainCenterY')}  "
          f"eyeHeight={scene.get('eyeHeightMeters')}  farClip={scene.get('farClipPlane'):.1f}")

    by_type = collections.Counter(o.get("type") for o in objects)
    by_state = collections.Counter(animation_state(o) for o in objects)
    print(f"\n  types:      {dict(by_type)}")
    print(f"  animation:  {dict(by_state)}")

    static = [o for o in objects if animation_state(o) == "STATIC"]
    if static:
        pct = 100 * len(static) / len(objects)
        print(f"\n--- STATIC: {len(static)} object(s), {pct:.1f}% of the scene ---")
        rows = collections.Counter((asset_of(o), o.get("type")) for o in static)
        for (asset, kind), n in rows.most_common(args.limit):
            print(f"  {n:6d}  {kind:10s} {asset}")

    physics = [o for o in objects if animation_state(o) == "physics"]
    if physics:
        print(f"\n--- PHYSICS (these fall under gravity): {len(physics)} object(s) ---")
        rows = collections.Counter(asset_of(o) for o in physics)
        for asset, n in rows.most_common(args.limit):
            print(f"  {n:6d}  {asset}")
        # A Rigidbody on something rooted in the ground is almost always a
        # misclassification rather than an intent -- surface the evidence.
        speeds = []
        for o in physics:
            v = (o.get("physics") or {}).get("velocity") or {}
            if isinstance(v, dict):
                speeds.append(math.sqrt(v.get("x", 0) ** 2 + v.get("y", 0) ** 2 + v.get("z", 0) ** 2))
            elif isinstance(v, (list, tuple)) and len(v) == 3:
                speeds.append(math.sqrt(sum(c * c for c in v)))
        if speeds:
            speeds.sort()
            print(f"  server velocities  min={speeds[0]:.4f}  median={speeds[len(speeds)//2]:.4f}"
                  f"  max={speeds[-1]:.4f} m/s")
            near_zero = sum(1 for s in speeds if s < 0.05)
            if near_zero:
                print(f"  {near_zero}/{len(speeds)} are under 0.05 m/s -- indistinguishable from "
                      f"stationary, so they were classified moving on tracker noise")

    print(f"\n--- assets by instance count ---")
    rows = collections.Counter(asset_of(o) for o in objects)
    for asset, n in rows.most_common(args.limit):
        states = collections.Counter(animation_state(o) for o in objects if asset_of(o) == asset)
        print(f"  {n:6d}  {asset:36s} {dict(states)}")


if __name__ == "__main__":
    main()
