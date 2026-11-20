#!/usr/bin/env python
"""
A slow pan of a 3DCodeVerse scene: the run's eye-level authored camera turned a full
circle about its own position, rendered by the harness's own renderers and encoded
with ffmpeg.

    python scripts/c3d-orbit.py paris_threejs_1112_1647 --frames 240 --fps 24
    python scripts/c3d-orbit.py mount_rainier_blender_1112_2154 --engine eevee --device GPU
    python scripts/c3d-orbit.py <slug> --camera MeadowEye --width 960 --height 540

Three.js scenes go through runtime_js/render_scene.mjs with explicit cameras (CPU
SwiftShader unless --gpu on); Blender scenes through the harness's render_bpy_scene.py
driver on artifacts/scene.blend. Frames and orbit.mp4 land in
scratch/c3d/orbit/<slug>/ (or --out). Run it in the `c3d` env (node + bpy on PATH).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "external" / "3dcodeverse" / "harness"
RUNS = HARNESS / "runs"
DRIVER = HARNESS / "codeverse3d" / "languages" / "wrappers" / "render_bpy_scene.py"


def run_dir(slug: str) -> Path:
    p = Path(slug)
    if p.is_dir():
        return p.resolve()
    p = RUNS / slug
    if not p.is_dir():
        sys.exit(f"no run at {p}")
    return p


def language(run: Path) -> str:
    rec = run / "record.json"
    if rec.is_file():
        try:
            lang = str(json.loads(rec.read_text()).get("language") or "")
            if lang:
                return lang
        except Exception:
            pass
    if (run / "deliverable" / "src" / "scene.py").is_file():
        return "scene_blender"
    return "scene_threejs"


def picked_views(run: Path) -> list[dict]:
    """The authored cameras of the picked round (deliverable/selection.json), else the
    last rendered round's."""
    renders = run / "artifacts" / "renders"
    rounds = sorted(p for p in renders.glob("r*") if (p / "views.json").is_file())
    if not rounds:
        sys.exit(f"no rendered rounds under {renders}")
    chosen = rounds[-1]
    sel = run / "deliverable" / "selection.json"
    if sel.is_file():
        try:
            s = json.loads(sel.read_text())
            r = s.get("round") if isinstance(s, dict) else None
            if r is None and isinstance(s, dict):
                r = (s.get("picked") or {}).get("round")
            if r is not None:
                cand = renders / f"r{int(r):02d}"
                if (cand / "views.json").is_file():
                    chosen = cand
        except Exception:
            pass
    views = json.loads((chosen / "views.json").read_text())
    return [v for v in views if v.get("kind") == "authored" and v.get("position") and v.get("lookAt")]


def eye_camera(views: list[dict], name: str | None) -> dict:
    if name:
        for v in views:
            if v.get("name") == name:
                return v
        sys.exit(f"no authored camera named {name!r}; have {[v.get('name') for v in views]}")
    # Eye level: the lowest authored camera (the harness's overviews are far higher).
    return min(views, key=lambda v: float(v["position"][1]))


def orbit_cameras(eye: dict, frames: int, pitch_deg: float | None) -> list[dict]:
    pos = [float(c) for c in eye["position"]]
    look = [float(c) for c in eye["lookAt"]]
    d = [look[i] - pos[i] for i in range(3)]
    horiz = math.hypot(d[0], d[2]) or 1.0
    if pitch_deg is not None:
        # Keep the authored look distance, set the pitch (negative looks down).
        dist = math.sqrt(sum(c * c for c in d)) or horiz
        d = [d[0] / horiz * dist * math.cos(math.radians(pitch_deg)), dist * math.sin(math.radians(pitch_deg)),
             d[2] / horiz * dist * math.cos(math.radians(pitch_deg))]
    cams = []
    for i in range(frames):
        a = 2.0 * math.pi * i / frames
        ca, sa = math.cos(a), math.sin(a)
        rx = d[0] * ca + d[2] * sa
        rz = -d[0] * sa + d[2] * ca
        cams.append({"name": f"orbit_{i:04d}", "position": pos, "lookAt": [pos[0] + rx, pos[1] + d[1], pos[2] + rz],
                     "fov": float(eye.get("fov") or 50.0)})
    return cams


def to_blender(v: list[float]) -> list[float]:
    x, y, z = v
    return [x, -z, y]


FRAME_PY = Path.home() / "miniconda3" / "envs" / "frame" / "bin" / "python"


def encode(out: Path, fps: int, pattern: str = "orbit_*_t0.png") -> Path | None:
    """H.264 through the frame env's PyAV (scripts/frames_to_mp4.py): no ffmpeg binary here."""
    cmd = [str(FRAME_PY), str(ROOT / "scripts" / "frames_to_mp4.py"), str(out), "--pattern", pattern,
           "--fps", str(fps), "--out", "orbit.mp4"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print((proc.stdout.strip().splitlines() or [proc.stderr.strip()[-300:]])[-1])
    mp4 = out / "orbit.mp4"
    return mp4 if mp4.is_file() else None


def render_threejs(run: Path, cams: list[dict], out: Path, args) -> None:
    node = shutil.which("node")
    if not node:
        sys.exit("node not on PATH (conda activate c3d)")
    ws = run / "deliverable"
    if not (ws / "src" / "scene.js").is_file():
        ws = run
    cmd = [node, str(HARNESS / "runtime_js" / "render_scene.mjs"), "--ws", str(ws), "--out", str(out),
           "--cameras", json.dumps(cams), "--orbit-views", "none", "--times", "0",
           "--width", str(args.width), "--height", str(args.height), "--gpu", args.gpu, "--no-settle",
           "--timeout-ms", str(int(args.timeout_s * 1000))]
    proc = subprocess.run(cmd, cwd=HARNESS / "runtime_js", capture_output=True, text=True, timeout=args.timeout_s + 60)
    last = (proc.stdout.strip().splitlines() or [""])[-1]
    print(last[:400])
    if proc.returncode != 0:
        sys.exit(f"render_scene.mjs failed ({proc.returncode}): {proc.stderr[-800:]}")


def render_blender(run: Path, cams: list[dict], out: Path, args) -> None:
    blend = run / "artifacts" / "scene.blend"
    if not blend.is_file():
        sys.exit(f"no {blend}")
    blender = shutil.which("blender-5.2") or os.path.expanduser("~/.local/bin/blender-5.2")
    job = {
        "engine": args.engine, "samples": args.samples, "device": args.device,
        "width": args.width, "height": args.height, "out": str(out), "result": str(out / "blender_frames.json"),
        "times": [0.0], "authored": False,
        "cameras": [{"name": c["name"], "position": to_blender(c["position"]), "look_at": to_blender(c["lookAt"]),
                     "fov": c["fov"], "kind": "orbit"} for c in cams],
    }
    job_path = out / "blender_job.json"
    job_path.write_text(json.dumps(job, indent=1))
    env = dict(os.environ)
    env.setdefault("C3D_RENDER_ENGINE", args.engine)
    cmd = [blender, "-b", "--factory-startup", str(blend), "--python", str(DRIVER), "--", str(job_path)]
    proc = subprocess.run(cmd, cwd=out, capture_output=True, text=True, env=env, timeout=args.timeout_s)
    res = out / "blender_frames.json"
    if not res.is_file():
        sys.exit(f"blender driver wrote no result ({proc.returncode}): {proc.stderr[-800:]}\n{proc.stdout[-800:]}")
    r = json.loads(res.read_text())
    print(f"blender {r.get('blender')}: {len(r.get('frames') or [])} frame(s), {len(r.get('errors') or [])} error(s)")
    for e in (r.get("errors") or [])[:3]:
        print("  ", e[:200])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--camera", default=None, help="authored camera name (default: the lowest, eye level)")
    ap.add_argument("--pitch", type=float, default=None, help="look pitch in degrees (default: the authored camera's)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gpu", default="off", help="three.js: off (SwiftShader, CPU) | auto | on")
    ap.add_argument("--engine", default="eevee", help="blender: eevee | cycles | workbench")
    ap.add_argument("--device", default="GPU", help="blender: GPU | CPU")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    args = ap.parse_args()

    run = run_dir(args.slug)
    lang = language(run)
    views = picked_views(run)
    eye = eye_camera(views, args.camera)
    cams = orbit_cameras(eye, args.frames, args.pitch)
    out = args.out or (ROOT / "scratch" / "c3d" / "orbit" / run.name)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("orbit_*_t0.png"):
        stale.unlink()
    print(f"{run.name}: {lang}, camera {eye.get('name')} at {eye['position']} looking {eye['lookAt']}, "
          f"{args.frames} frames -> {out}")
    if lang == "scene_blender":
        render_blender(run, cams, out, args)
    else:
        render_threejs(run, cams, out, args)
    n = len(list(out.glob("orbit_*_t0.png")))
    mp4 = encode(out, args.fps)
    print(f"{n} frame(s); video {mp4 if mp4 else 'not encoded'}")
    (out / "orbit.json").write_text(json.dumps({"run": str(run), "language": lang, "camera": eye, "frames": args.frames,
                                                "fps": args.fps, "video": str(mp4) if mp4 else None}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
