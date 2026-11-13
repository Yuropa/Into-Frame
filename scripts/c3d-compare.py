#!/usr/bin/env python
"""
Score a 3DCodeVerse run's renders against the photograph it was briefed on, with the
same metrics scripts/scene-check.py uses for our own pipeline, and lay them out on
one sheet beside our latest check render.

    python scripts/c3d-compare.py paris_threejs_1106_1148            # run slug or path
    python scripts/c3d-compare.py <slug> --scene Paris --ours round11b
    python scripts/c3d-compare.py <slug> --round 1

Reads <run>/artifacts/renders/rNN/views.json (name, path, time_s, judge) -- the
picked round by default (deliverable/selection.json, else the last rendered round) --
scores every authored-camera t=0 frame (overview_* harness rig views skipped unless
--all) and writes

    scratch/check/<Scene>/c3d_<slug>/sheet.png     photo | ours | their views, scored
    scratch/check/<Scene>/c3d_<slug>/scores.json
    scratch/check/<Scene>/history.jsonl            one line, label c3d_<slug>, so the
                                                    row sits in the same history as ours

Run it in the `frame` env (torch + transformers for DINOv2).
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "external" / "3dcodeverse" / "harness"


def _scene_check():
    spec = importlib.util.spec_from_file_location("scene_check", ROOT / "scripts" / "scene-check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_run(slug: str) -> Path:
    p = Path(slug)
    if p.is_dir():
        return p.resolve()
    for root in (HARNESS / "runs", ROOT / "external" / "3dcodeverse" / "eval" / "bench" / "out"):
        hits = list(root.glob(f"**/{slug}")) if root.is_dir() else []
        if hits:
            return hits[0].resolve()
    sys.exit(f"no run named {slug} under {HARNESS / 'runs'}")


def pick_round(run: Path, wanted: int | None) -> int:
    renders = run / "artifacts" / "renders"
    rounds = sorted(int(p.name[1:]) for p in renders.glob("r[0-9][0-9]") if (p / "views.json").is_file())
    if not rounds:
        sys.exit(f"{run} has no rendered rounds")
    if wanted is not None:
        if wanted not in rounds:
            sys.exit(f"round {wanted} not rendered; have {rounds}")
        return wanted
    sel = run / "deliverable" / "selection.json"
    if sel.is_file():
        try:
            r = json.loads(sel.read_text())
            for key in ("round", "round_index", "picked"):
                if isinstance(r.get(key), int) and r[key] in rounds:
                    return r[key]
        except ValueError:
            pass
    return rounds[-1]


def views(run: Path, rnd: int, all_views: bool) -> list[tuple[str, Path]]:
    d = run / "artifacts" / "renders" / f"r{rnd:02d}"
    out = []
    for v in json.loads((d / "views.json").read_text()):
        name, rel = v.get("name", "?"), v.get("path", "")
        t = v.get("time_s")
        if t not in (None, 0, 0.0):
            continue
        if name.startswith("overview_") and not all_views:
            continue
        p = Path(rel) if Path(rel).is_absolute() else d / rel
        if p.is_file():
            out.append((name, p))
    return out


def brief_of(run: Path) -> tuple[str, str]:
    rec = run / "record.json"
    if rec.is_file():
        try:
            r = json.loads(rec.read_text())
            spec = r.get("spec", {})
            refs = spec.get("references") or []
            return spec.get("prompt", ""), (refs[0].get("path", "") if refs else "")
        except ValueError:
            pass
    return "", ""


def our_render(scene_dir: Path, label: str | None) -> tuple[str, Path] | None:
    dirs = [p for p in scene_dir.iterdir() if p.is_dir() and (p / "photo.png").is_file() and not p.name.startswith("c3d_")]
    if label:
        dirs = [p for p in dirs if p.name == label]
    if not dirs:
        return None
    best = max(dirs, key=lambda p: (p / "photo.png").stat().st_mtime)
    return best.name, best / "photo.png"


def fit(img: Image.Image, w: int) -> Image.Image:
    h = round(img.height * w / img.width)
    return img.convert("RGB").resize((w, h), Image.BICUBIC)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run slug under external/3dcodeverse/harness/runs, or a run directory")
    ap.add_argument("--scene", help="our scene name (default: from the run's --image file name)")
    ap.add_argument("--ours", help="scratch/check/<Scene>/<label> to put beside it (default: newest)")
    ap.add_argument("--round", type=int, help="render round (default: the picked / last one)")
    ap.add_argument("--all", action="store_true", help="include the harness's overview_* rig views")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--no-history", action="store_true", help="do not append to scratch/check/<Scene>/history.jsonl")
    a = ap.parse_args()

    run = find_run(a.run)
    rnd = pick_round(run, a.round)
    brief, ref = brief_of(run)
    scene = a.scene or (Path(ref).stem if ref else None)
    if not scene:
        sys.exit("cannot tell the scene from the run; pass --scene")
    photo_path = Path(ref) if ref and Path(ref).is_file() else ROOT / "server" / "samples" / f"{scene}.jpg"
    photo = Image.open(photo_path).convert("RGB")

    sc = _scene_check()
    scorer = sc.Scorer()
    theirs = views(run, rnd, a.all)
    if not theirs:
        sys.exit(f"round {rnd} has no authored-camera t=0 renders")

    scene_dir = ROOT / "scratch" / "check" / scene.replace(" ", "_")
    scene_dir.mkdir(parents=True, exist_ok=True)
    out = scene_dir / f"c3d_{run.name}"
    out.mkdir(exist_ok=True)

    rows = []
    for name, p in theirs:
        img = Image.open(p).convert("RGB")
        s = scorer.score(photo, img)
        s.update(sc.image_flags(img, None))
        rows.append({"view": name, "path": str(p), **s})
    rows.sort(key=lambda r: -r["dino_global"])
    ours = our_render(scene_dir, a.ours)
    ours_score = None
    if ours:
        ours_score = scorer.score(photo, Image.open(ours[1]))

    # ── sheet: photo | ours | their views in score order ─────────────────────
    w = a.width
    tiles = [("photo", fit(photo, w), None)]
    if ours:
        tiles.append((f"ours {ours[0]}", fit(Image.open(ours[1]), w), ours_score))
    for r in rows[:6]:
        tiles.append((f"c3d {r['view']}", fit(Image.open(r["path"]), w), r))
    th = max(t.height for _, t, _ in tiles) + 34
    sheet = Image.new("RGB", (w * len(tiles), th), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    for i, (label, img, s) in enumerate(tiles):
        sheet.paste(img, (i * w, 0))
        d.text((i * w + 6, img.height + 3), label[: w // 7], fill=(230, 230, 230))
        if s:
            d.text((i * w + 6, img.height + 17),
                   f"g {s['dino_global']:.3f}  l {s['dino_layout']:.3f}  dE {s['colour_de']:.1f}", fill=(180, 220, 180))
    sheet.save(out / "sheet.png")

    best = rows[0]
    result = {
        "run": str(run), "round": rnd, "brief": brief, "scene": scene,
        "best_view": best["view"], "dino_global": best["dino_global"], "dino_layout": best["dino_layout"],
        "colour_de": best["colour_de"], "views": rows,
        "ours": {"label": ours[0], **ours_score} if ours else None,
    }
    (out / "scores.json").write_text(json.dumps(result, indent=2))
    if not a.no_history:
      with (scene_dir / "history.jsonl").open("a") as f:
        f.write(json.dumps({"time": dt.datetime.now().isoformat(timespec="seconds"), "label": f"c3d_{run.name}",
                            "scene": scene, "git": sc.git_rev(), "dino_global": best["dino_global"],
                            "dino_layout": best["dino_layout"], "colour_de": best["colour_de"],
                            "black": best.get("black"), "flat": best.get("flat"), "view": best["view"],
                            "source": "3dcodeverse"}) + "\n")

    print(f"{scene}: 3dcodeverse round {rnd}, {len(rows)} authored view(s)")
    print(f"  {'view':24s} {'dino_g':>7s} {'dino_l':>7s} {'dE':>6s} {'black':>6s} {'flat':>6s}")
    for r in rows:
        print(f"  {r['view'][:24]:24s} {r['dino_global']:7.3f} {r['dino_layout']:7.3f} {r['colour_de']:6.1f} {r['black']:6.3f} {r['flat']:6.3f}")
    if ours:
        print(f"  {('ours ' + ours[0])[:24]:24s} {ours_score['dino_global']:7.3f} {ours_score['dino_layout']:7.3f} {ours_score['colour_de']:6.1f}")
    print(f"  sheet: {out / 'sheet.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
