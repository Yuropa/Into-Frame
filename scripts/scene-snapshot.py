#!/usr/bin/env python3
"""Freeze a built scene so later work (animation, video, the agent loop) starts from it
without rebuilding.

    scripts/scene-snapshot.py save Paris --tag opus1 [--check opus1] [--note "..."] [--force]
    scripts/scene-snapshot.py restore Paris --tag opus1
    scripts/scene-snapshot.py list [Paris]

The pipeline caches every stage's output in output/<md5(photo)>/<Stage>/ and reruns a
stage only when it is forced (--rerun), its seed changed, or an earlier stage ran. A
snapshot is a full copy of that run directory under output/snapshots/<Scene>/<tag>/run,
with manifest.json (git commit, config.yaml hash, the scene-check scores when --check
names a scratch/check label) and the config.yaml it was built with. `restore` copies it
back over output/<md5(photo)>, so

    cd server && python main.py run "samples/Paris.jpg" --output ../output --rerun "Video Generation"

then runs only the animation stages (Video Generation, Video Object Extraction, Object
Motion Classification, Category Mesh Rigging, Scene Animation) on the frozen scene.
Copies, not links: the filesystem is ext4 (no reflinks) and stages truncate files in
place, so a hard-linked snapshot would be overwritten by the next rerun. A run dir is
2.5-3.5 GB. Refuses to restore while a pipeline run or server is alive.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPS = ROOT / "output" / "snapshots"
sys.path.insert(0, str(ROOT / "server"))


def run_dir(scene: str) -> tuple[Path, Path]:
    photo = ROOT / "server" / "samples" / f"{scene}.jpg"
    if not photo.is_file():
        sys.exit(f"no photo at {photo}")
    digest = hashlib.md5(photo.read_bytes()).digest()  # PipelineInputItem._hash_image
    return photo, ROOT / "output" / str(uuid.UUID(bytes=digest))


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def config_sha() -> str:
    return hashlib.sha256((ROOT / "server" / "config.yaml").read_bytes()).hexdigest()[:12]


def check_scores(scene: str, label: str) -> dict:
    d = ROOT / "scratch" / "check" / scene.replace(" ", "_") / label
    out: dict = {"label": label}
    if (d / "check.json").is_file():
        c = json.load(open(d / "check.json"))
        out.update({k: c.get(k) for k in ("dino_global", "dino_layout", "colour_de", "black", "flat", "oversize", "judge", "acceptance_pass") if k in c})
    if (d / "verdict.json").is_file():
        v = json.load(open(d / "verdict.json"))
        out["judge_model"] = v.get("model")
        out["judge_summary"] = v.get("summary")
    return out


def du(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e9


def save(args) -> int:
    photo, run = run_dir(args.scene)
    if not run.is_dir():
        sys.exit(f"no run for {args.scene} at {run}")
    dest = SNAPS / args.scene / args.tag
    if dest.exists():
        if not args.force:
            sys.exit(f"{dest} exists; --force replaces it")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    stages = sorted(p.name for p in run.iterdir() if p.is_dir())
    print(f"copying {run} ({du(run):.1f} GB, {len(stages)} stages) -> {dest / 'run'}")
    t0 = time.time()
    shutil.copytree(run, dest / "run", symlinks=True)
    shutil.copy2(ROOT / "server" / "config.yaml", dest / "config.yaml")
    plan = run / "Scene Plan" / "scene_plan.json"
    manifest = {
        "scene": args.scene, "tag": args.tag, "run": run.name, "photo": str(photo.relative_to(ROOT)),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_commit(), "config_sha": config_sha(),
        "stages": stages, "last_stage": stages[-1] if stages else None,
        "plan_model": json.load(open(plan)).get("model") if plan.is_file() else None,
        "note": args.note or "",
    }
    if args.check:
        manifest["check"] = check_scores(args.scene, args.check)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved in {time.time() - t0:.0f} s; manifest {dest / 'manifest.json'}")
    if args.check and "dino_global" not in manifest.get("check", {}):
        print(f"  (no check.json under scratch/check/{args.scene.replace(' ', '_')}/{args.check}; scores not recorded)")
    return 0


def restore(args) -> int:
    from agent.runner import gpu_busy
    busy = gpu_busy()
    if busy:
        sys.exit(f"a pipeline holds the run directories: {busy}")
    photo, run = run_dir(args.scene)
    src = SNAPS / args.scene / args.tag
    if not (src / "run").is_dir():
        sys.exit(f"no snapshot at {src}")
    m = json.load(open(src / "manifest.json")) if (src / "manifest.json").is_file() else {}
    if m.get("run") and m["run"] != run.name:
        sys.exit(f"snapshot was taken of run {m['run']} but {photo.name} now hashes to {run.name}; the photo changed")
    print(f"restoring {src / 'run'} -> {run}")
    t0 = time.time()
    run.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(["rsync", "-a", "--delete", f"{src / 'run'}/", f"{run}/"]).returncode
    if rc != 0:
        sys.exit(f"rsync failed ({rc})")
    print(f"restored in {time.time() - t0:.0f} s")
    if m.get("config_sha") and m["config_sha"] != config_sha():
        print(f"  note: server/config.yaml differs from the one this snapshot was built with "
              f"({m['config_sha']} then, {config_sha()} now); the stages you rerun will use the current one. "
              f"The old one is at {src / 'config.yaml'}")
    return 0


def list_(args) -> int:
    scenes = [SNAPS / args.scene] if args.scene else sorted(p for p in SNAPS.iterdir() if p.is_dir()) if SNAPS.is_dir() else []
    n = 0
    for sd in scenes:
        for td in sorted(p for p in sd.iterdir() if p.is_dir()) if sd.is_dir() else []:
            mf = td / "manifest.json"
            m = json.load(open(mf)) if mf.is_file() else {}
            c = m.get("check", {})
            scores = (f"dino {c.get('dino_global')} / {c.get('dino_layout')}  dE {c.get('colour_de')}  judge {c.get('judge')} "
                      f"({c.get('acceptance_pass')})" if c else "no scores")
            print(f"{sd.name:14s} {td.name:12s} {m.get('created', '?'):19s} git {m.get('git', '?'):8s} "
                  f"plan {str(m.get('plan_model')):28s} last stage {str(m.get('last_stage')):26s} {scores}"
                  + (f"  -- {m['note']}" if m.get("note") else ""))
            n += 1
    if not n:
        print(f"no snapshots under {SNAPS}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save"); s.add_argument("scene"); s.add_argument("--tag", required=True)
    s.add_argument("--check", default=None, help="scratch/check label whose scores go in the manifest")
    s.add_argument("--note", default=None); s.add_argument("--force", action="store_true"); s.set_defaults(fn=save)
    r = sub.add_parser("restore"); r.add_argument("scene"); r.add_argument("--tag", required=True); r.set_defaults(fn=restore)
    l = sub.add_parser("list"); l.add_argument("scene", nargs="?"); l.set_defaults(fn=list_)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
