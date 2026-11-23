#!/usr/bin/env python3
"""Edit a built scene with typed actions and render the result, no pipeline rerun
(server/agent/editor.py; docs/agentic-scene-graph-plan.md).

    scripts/scene-edit.py Paris --list                      # the placed objects: name, id, bearing, distance, size
    scripts/scene-edit.py Paris --actions a.json --label try1 --render
    scripts/scene-edit.py Paris --actions a.json --label try1 --accept
    scripts/scene-edit.py Paris --edits                     # accepted edits so far (<run>/scene_edits.json)
    scripts/scene-edit.py Paris --undo 1                    # drop the last accepted edit and replay

a.json is a list of actions, e.g.
    [{"action": "place", "mesh": "category_mesh_boat_0", "class": "boat", "name": "tour boat",
      "at": {"photo_box": [0.38, 0.55, 0.18, 0.09]}, "height_m": 6, "yaw_deg": 100, "why": "the judge: boat missing"},
     {"action": "move", "target": "thing_20", "at": {"bearing_deg": -12, "distance_m": 140}, "why": "..."},
     {"action": "remove", "target": "structure_eiffel_tower", "why": "..."}]

Preview: the actions are applied to a copy of the run's newest scene.json and written to
scratch/edit/<Scene>/<label>/scene.json with records.json; --render draws the photo view
(and --views more) there with scripts/render-scene.py --scene. Accept: the applied
actions are appended to <run>/scene_edits.json, noted in the scene graph, and the Scene
Edits stage is replayed into the run so its scene.json (the newest) carries them;
later stages become dirty for the next `main.py run`. The run directory is otherwise
only read. Rendering refuses while a pipeline run, server or pan render holds the GPU.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def resolve(scene: str) -> tuple[str, Path]:
    p = Path(scene)
    if p.is_dir():
        return p.name, p.resolve()
    from agent.runner import resolve_run
    run = resolve_run(scene)
    if run is None:
        sys.exit(f"no cached run for '{scene}'")
    return scene, run


def metadata(run: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    order = run / "_stage_order.json"
    names = json.loads(order.read_text()) if order.is_file() else sorted(d.name for d in run.iterdir() if d.is_dir())
    if isinstance(names, dict):
        names = names.get("stages") or list(names)
    for n in names:
        d = run / n
        if d.is_dir():
            for f in d.glob("metadata_*.json"):
                try:
                    out[f.stem.split("_", 1)[1]] = json.loads(f.read_text())
                except Exception:
                    pass
    return out


def list_objects(ed, run: Path, photo_size, everything: bool = False) -> None:
    """Populations (a mesh family with 20+ instances) as one line each; everything else one per object."""
    import re
    meta = metadata(run)
    W, H = photo_size
    fam: dict[str, list[dict]] = {}
    for o in ed.objects:
        m = re.match(r"category_mesh_([a-z_]+?)_\d+", str(o.get("mesh") or ""))
        fam.setdefault(m.group(1) if m else "", []).append(o)
    pops = {k: v for k, v in fam.items() if k and len(v) >= 20 and not everything}
    singles = [o for k, v in fam.items() if k not in pops for o in v]
    if pops:
        print(f"{'population':20s} {'count':>6s} {'dist m':>14s} {'h m':>10s}  members are targets by id or thing_<src>")
        for k, v in sorted(pops.items()):
            d = sorted(ed.bearing_distance([o["position"]["x"], o["position"]["y"], o["position"]["z"]])[1] for o in v)
            hs = [float(meta.get(str(o.get("sourceIndex")), {}).get("world_height") or 0) for o in v]
            hs = [h for h in hs if h > 0]
            print(f"pop_{k:16s} {len(v):6d} {d[0]:6.1f}-{d[-1]:6.1f} {(f'{min(hs):4.1f}-{max(hs):4.1f}' if hs else ''):>10s}")
        print()
    print(f"{'name':34s} {'id':8s} {'src':>5s} {'mesh':30s} {'bear':>6s} {'dist':>6s} {'scale':>6s} {'h_m':>5s}  photo_box (fractions)")
    rows = []
    for o in singles:
        b, dist = ed.bearing_distance([o["position"]["x"], o["position"]["y"], o["position"]["z"]])
        if ed.is_env(o):
            b, dist = 0.0, 0.0
        m = meta.get(str(o.get("sourceIndex")), {})
        pb = m.get("photo_box")
        box = f"[{pb[0] / W:.3f}, {pb[1] / H:.3f}, {pb[2] / W:.3f}, {pb[3] / H:.3f}]" if pb and W else ""
        h = m.get("world_height")
        rows.append((dist, f"{str(o.get('name'))[:34]:34s} {str(o.get('id'))[:8]:8s} {str(o.get('sourceIndex') if o.get('sourceIndex') is not None else ''):>5s} "
                     f"{str(o.get('mesh'))[:30]:30s} {b:6.1f} {dist:6.1f} {float(o['scale']['y']):6.2f} "
                     f"{(f'{float(h):5.1f}' if h is not None else '     ')}  {box}"))
    for _, line in sorted(rows, key=lambda r: r[0]):
        print(line)
    print(f"\n{len(ed.objects)} objects ({len(singles)} listed); mesh keys for place/swap_mesh: {', '.join(ed.mesh_keys())}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--actions", type=Path, default=None, help="JSON list of actions")
    ap.add_argument("--label", default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--views", default="photo")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--no-grass", action="store_true")
    ap.add_argument("--accept", action="store_true", help="persist the applied actions in the run and replay the Scene Edits stage")
    ap.add_argument("--who", default="agent")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true", help="--list every population member too")
    ap.add_argument("--edits", action="store_true")
    ap.add_argument("--undo", type=int, default=0)
    ap.add_argument("--config", type=Path, default=ROOT / "server" / "config.yaml")
    args = ap.parse_args()

    from agent.editor import Editor, apply_all, load_context, load_edits, newest_scene_path, save_edits
    name, run = resolve(args.scene)
    scene_path = newest_scene_path(run)
    scene = json.loads(scene_path.read_text())
    print(f"[{name}] {run.name}: {len(scene['objects'])} objects from {scene_path.parent.name}")

    if args.edits:
        for i, e in enumerate(load_edits(run)["edits"]):
            print(f"{i:3d} {e.get('time', '')} {e.get('who', '')}: {e.get('note', '')}  -- {e.get('why', '')}")
        return 0

    if args.undo:
        data = load_edits(run)
        dropped = data["edits"][-args.undo:]
        data["edits"] = data["edits"][:-args.undo]
        save_edits(run, data)
        for e in dropped:
            print(f"dropped: {e.get('note')}")
        return replay(run, args.config)

    t0 = time.time()
    context = load_context(run, args.config)
    print(f"context loaded in {time.time() - t0:.1f} s")
    ed = Editor(scene, context=context, run_dir=run)

    if args.list:
        from PIL import Image
        inp = run / "input.png"
        size = Image.open(inp).size if inp.is_file() else (0, 0)
        list_objects(ed, run, size, everything=args.all)
        return 0

    if args.actions is None:
        ap.error("--actions, --list, --edits or --undo")
    actions = json.loads(args.actions.read_text())
    if isinstance(actions, dict):
        actions = actions.get("actions") or actions.get("edits") or [actions]
    label = args.label or time.strftime("edit-%m%d-%H%M%S")
    out = ROOT / "scratch" / "edit" / name.replace(" ", "_") / label
    out.mkdir(parents=True, exist_ok=True)
    new, applied, failures = apply_all(scene, actions, context=context, run_dir=run, who=args.who)
    (out / "scene.json").write_text(json.dumps(new, indent=1))
    (out / "records.json").write_text(json.dumps({"applied": [{k: v for k, v in r.items() if k != "removed"} for r in applied],
                                                  "failures": failures}, indent=1))
    print(f"{len(applied)} applied, {len(failures)} refused -> {out}")

    if args.render:
        from agent.runner import gpu_busy
        busy = gpu_busy()
        if busy:
            print(f"not rendering: the GPU is held by {busy}", file=sys.stderr)
            return 3
        cmd = [sys.executable, str(ROOT / "scripts" / "render-scene.py"), str(run), "--scene", str(out / "scene.json"),
               "--out", str(out), "--views", args.views, "--width", str(args.width)]
        if args.no_grass:
            cmd.append("--no-grass")
        t0 = time.time()
        res = subprocess.run(cmd, cwd=ROOT / "server", capture_output=True, text=True)
        (out / "render.log").write_text(res.stdout + res.stderr)
        if res.returncode != 0:
            print(res.stdout[-1500:], res.stderr[-1500:])
            return 1
        print(f"rendered {args.views} in {time.time() - t0:.0f} s -> {out}")

    if args.accept and applied:
        data = load_edits(run)
        data["edits"].extend(applied)
        save_edits(run, data)
        note_graph(run, applied, args.who)
        print(f"accepted {len(applied)} actions -> {run / 'scene_edits.json'} ({len(data['edits'])} total)")
        return replay(run, args.config)
    return 0


def note_graph(run: Path, applied: list[dict], who: str) -> None:
    try:
        from pipeline.scene_graph.graph import SceneGraph
        g = SceneGraph.from_run(run)
    except Exception:
        g = None
    if g is None:
        return
    for r in applied:
        inp = r.get("input") or {}
        target = str(inp.get("target") or "")
        node = target if target in g.nodes else None
        if node is None and r.get("action") == "place":
            cls = (inp.get("class") or "").lower()
            node = f"pop_{cls}" if f"pop_{cls}" in g.nodes else None
        text = f"{r.get('action')}: {r.get('note')}" + (f" -- {r.get('why')}" if r.get("why") else "")
        g.annotate(node or "world", text, who)


def replay(run: Path, config: Path) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / "replay-stage.py"), str(run), "Scene Edits", "--save", "--config", str(config)]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    tail = (res.stdout + res.stderr).strip().splitlines()[-3:]
    print("\n".join(tail))
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
