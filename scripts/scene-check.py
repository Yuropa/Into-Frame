#!/usr/bin/env python
"""
Contact sheets, photo scores and sanity flags for cached runs, kept as a history.

    python scripts/scene-check.py                        # Paris, Mount Rainier, Shark Fin, Iceland
    python scripts/scene-check.py Paris "Mount Rainier" --label treeline-fix
    python scripts/scene-check.py output/<run-id> --width 480

For each scene: renders photo/left/right/back/down with scripts/render-scene.py,
lays them out beside the input photograph with the previous check's row under
them (sheet.png), and appends one line to scratch/check/<scene>/history.jsonl.

Why the other four views: every bad regression of 2026-10-17 was invisible from
the capture camera -- 70 m canopy slabs in the sky behind the camera (Treeline
Backdrop restoring near trees at backdrop depth), a black shadow block on the
horizon, a gravel ring around the viewer.

Scores (photo view against the photograph):
  dino_global   DINOv2 CLS cosine -- does the render show the same kind of scene.
  dino_layout   mean cosine of spatially corresponding DINOv2 patch tokens -- is
                the same thing in the same place.
  colour_de     mean CIE76 delta-E between 24x16 downsamples -- overall colour.
Flags:
  oversize      a placed instance taller than its class could be (tree > 70 m,
                flower > 2.5 m ...).
  boat_tall     a boat/ship whose height exceeds 0.35x its length.
  treeline_tall a treeline quad taller than 40 m.
  black         non-sky pixels near black with no texture, per view, as a
                fraction of the frame (components over 0.3% only).
  flat          non-sky regions with almost no texture (over 2% of the frame);
                the test renderer's water is flat, so read this as a delta.
Scene Plan (docs/hybrid-scene-plan.md): when the run has a Scene Plan, the record
carries its key readings and every prior decision the consumer stages published
(plan_overrides_*.json: field, measured, plan, used, source), printed under the table.
  --judge      a vision model (default anthropic:claude-sonnet-5-5) grades the photo view
               against the photograph and the plan's acceptance list: observe, list
               issues (target, kind, severity), pass/fail each acceptance item, then a
               0-1 score computed in code from the rubric minus penalties. Written to
               <out>/verdict.json and the history row (judge, acceptance_pass). Cents per
               check; honours FRAME_NO_API=1 (no call, no judge columns).
Scene names resolve to output/<uuid>/ by comparing input.png with
server/samples/<name>.jpg.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

VIEWS = ["photo", "left", "right", "back", "down"]
DEFAULT_SCENES = ["Paris", "Mount Rainier", "Shark Fin", "Iceland"]
# Tallest plausible instance per class, metres.
MAX_HEIGHT_M = {
    "tree": 70.0, "flower": 2.5, "grass": 2.0, "bush": 6.0, "shrub": 6.0, "plant": 3.0,
    "rock": 25.0, "boulder": 25.0, "person": 2.3, "car": 3.0, "truck": 5.0, "bus": 4.5,
    "boat": 20.0, "ship": 40.0, "dock": 10.0, "bench": 2.0, "lamp": 12.0, "bird": 2.0,
}


# ── scene resolution ────────────────────────────────────────────────────────────

def _thumb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((64, 48)), dtype=np.float32)


def resolve(scene: str) -> tuple[str, Path]:
    p = Path(scene)
    if p.is_dir():
        return p.name, p.resolve()
    sample = ROOT / "server" / "samples" / f"{scene}.jpg"
    if not sample.exists():
        raise SystemExit(f"no run dir or sample named {scene!r}")
    want = _thumb(sample)
    best, best_err = None, 1e9
    for run in (ROOT / "output").iterdir():
        inp = run / "input.png"
        if not inp.exists() or not (run / "Scene Generation" / "scene.json").exists():
            continue
        err = float(np.abs(_thumb(inp) - want).mean())
        if err < best_err:
            best, best_err = run, err
    if best is None or best_err > 8.0:
        raise SystemExit(f"no cached run matches {scene!r}")
    return scene, best


# ── geometry flags ──────────────────────────────────────────────────────────────


def _stage_dirs_in_order(run: Path) -> list[Path]:
    order_file = run / "_stage_order.json"
    try:
        names = json.loads(order_file.read_text())
        if isinstance(names, dict):
            names = names.get("stages") or list(names)
    except Exception:
        names = sorted(d.name for d in run.iterdir() if d.is_dir())
    return [run / n for n in names if (run / n).is_dir()]

def geometry_flags(run: Path) -> dict:
    import trimesh
    from scene.mesh import Mesh

    # The newest scene.json any stage wrote, the one render-scene draws (Structure
    # Placeholders appends objects after Scene Generation; Scene Edits, replayed into
    # the run by scripts/scene-edit.py --accept, is newer than a stale Scene Animation).
    cands = sorted(run.glob("*/scene.json"), key=lambda p: p.stat().st_mtime)
    scene_file = cands[-1] if cands else run / "Scene Generation" / "scene.json"
    scene = json.loads(scene_file.read_text())
    glbs: dict[str, Path] = {}
    order_file = run / "_stage_order.json"
    order = json.loads(order_file.read_text()) if order_file.exists() else sorted(
        d.name for d in run.iterdir() if d.is_dir())
    for stage in order:
        for g in (run / stage).glob("*.glb"):
            glbs[g.stem] = g  # the latest stage's copy wins, as in the context
    extents: dict[str, np.ndarray | None] = {}

    def ext(key: str):
        if key not in extents:
            g = glbs.get(key)
            try:
                extents[key] = Mesh.load(g).mesh.extents if g else None
            except Exception:
                extents[key] = None
        return extents[key]

    oversize, boat_tall = [], []
    for o in scene["objects"]:
        name = str(o.get("name") or "")
        key = o.get("mesh") or ""
        if not name.startswith("category_mesh_") or not key:
            continue
        cls = name[len("category_mesh_"):].rstrip("0123456789").rstrip("_")
        e = ext(key)
        if e is None:
            continue
        s = o.get("scale") or {}
        sx, sy, sz = (float(s.get(k, 1.0)) for k in "xyz")
        h = sy * float(e[1])
        length = max(sx * float(e[0]), sz * float(e[2]))
        limit = MAX_HEIGHT_M.get(cls)
        if limit and h > limit:
            oversize.append(f"{name} {h:.1f} m")
        if cls in ("boat", "ship") and length > 0 and h > 0.35 * length:
            boat_tall.append(f"{name} {h:.1f}x{length:.1f} m")

    treeline_tall = 0
    tl = glbs.get("treeline_backdrop")
    if tl is not None:
        v = trimesh.load(str(tl), force="mesh").vertices
        quads = v[: (len(v) // 4) * 4].reshape(-1, 4, 3)
        heights = quads[:, :, 1].max(1) - quads[:, :, 1].min(1)
        treeline_tall = int((heights > 40.0).sum())
    return {
        "oversize": len(oversize), "oversize_examples": oversize[:5],
        "boat_tall": len(boat_tall), "boat_tall_examples": boat_tall[:5],
        "treeline_tall": treeline_tall,
    }


# ── image flags and scores ──────────────────────────────────────────────────────

def _local_std(lum: np.ndarray, r: int = 3) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    m = uniform_filter(lum, 2 * r + 1)
    m2 = uniform_filter(lum * lum, 2 * r + 1)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def image_flags(img: Image.Image, depth: np.ndarray | None) -> dict:
    from scipy.ndimage import label
    rgb = np.asarray(img, dtype=np.float32)
    lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    ground = depth > 0 if depth is not None else np.ones(lum.shape, bool)
    std = _local_std(lum)
    total = float(lum.size)

    def area(mask: np.ndarray, min_frac: float) -> float:
        lab, n = label(mask)
        if n == 0:
            return 0.0
        sizes = np.bincount(lab.ravel())[1:]
        return float(sizes[sizes >= min_frac * total].sum() / total)

    return {
        "black": round(area(ground & (lum < 14) & (std < 4), 0.003), 4),
        "flat": round(area(ground & (std < 1.2), 0.02), 4),
    }


class Scorer:
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        name = "facebook/dinov2-base"
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval()

    def _tokens(self, img: Image.Image, size=(448, 308)):
        torch = self.torch
        x = np.asarray(img.convert("RGB").resize(size, Image.BICUBIC), dtype=np.float32) / 255.0
        mean = np.array(self.proc.image_mean, dtype=np.float32)
        std = np.array(self.proc.image_std, dtype=np.float32)
        t = torch.from_numpy(((x - mean) / std).transpose(2, 0, 1))[None]
        with torch.no_grad():
            h = self.model(pixel_values=t).last_hidden_state[0]
        h = torch.nn.functional.normalize(h, dim=-1)
        return h[0], h[1:]

    def score(self, photo: Image.Image, render: Image.Image) -> dict:
        from skimage.color import rgb2lab
        cp, pp = self._tokens(photo)
        cr, pr = self._tokens(render)
        small = (24, 16)
        a = rgb2lab(np.asarray(photo.convert("RGB").resize(small, Image.BOX)) / 255.0)
        b = rgb2lab(np.asarray(render.convert("RGB").resize(small, Image.BOX)) / 255.0)
        return {
            "dino_global": round(float((cp * cr).sum()), 4),
            "dino_layout": round(float((pp * pr).sum(-1).mean()), 4),
            "colour_de": round(float(np.linalg.norm(a - b, axis=-1).mean()), 2),
        }


# ── scene plan ──────────────────────────────────────────────────────────────────

def plan_summary(run: Path) -> dict | None:
    """The run's Scene Plan readings and the prior decisions its consumers published."""
    pf = run / "Scene Plan" / "scene_plan.json"
    if not pf.exists():
        return None
    try:
        rec = json.loads(pf.read_text())
    except ValueError:
        return None
    p = rec.get("plan", {})

    def blk(name, *fields):
        b = p.get(name, {}) or {}
        out = {f: b.get(f) for f in fields if b.get(f) is not None}
        out["evidence"] = b.get("evidence")
        return out

    overrides = []
    for f in sorted(run.glob("*/plan_overrides_*.json")):
        try:
            for o in json.loads(f.read_text()):
                overrides.append({"stage": f.parent.name, "field": o.get("field"), "measured": o.get("measured"),
                                  "plan": o.get("plan"), "used": o.get("used"), "source": o.get("source"),
                                  "evidence": o.get("evidence")})
        except ValueError:
            pass
    return {
        "model": rec.get("model"), "source": rec.get("source"), "created": rec.get("created"),
        "place": blk("place", "kind", "name"), "camera": blk("camera", "height_m", "pitch_deg"),
        "sun": blk("sun", "azimuth_deg", "elevation_deg", "warmth"),
        "water": blk("water", "kind", "width_m"), "terrain": blk("terrain", "kind"),
        "species": [x.get("name") for x in p.get("species", [])],
        "acceptance": p.get("acceptance", []), "overrides": overrides,
    }


JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["summary", "issues", "acceptance", "rubric"],
    "properties": {
        "summary": {"type": "string", "description": "two sentences: what the render shows vs the photograph"},
        "issues": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                   "required": ["target", "kind", "severity", "text"],
                   "properties": {"target": {"type": "string"},
                                  "kind": {"type": "string", "enum": ["composition", "geometry", "material", "scale",
                                                                      "lighting", "atmosphere", "missing", "extra"]},
                                  "severity": {"type": "string", "enum": ["minor", "major", "severe"]},
                                  "text": {"type": "string"}}}},
        "acceptance": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                       "required": ["id", "pass", "evidence"],
                       "properties": {"id": {"type": "string"}, "pass": {"type": "boolean"}, "evidence": {"type": "string"}}}},
        "rubric": {"type": "object", "additionalProperties": False,
                   "required": ["layout", "geometry", "materials", "lighting", "atmosphere", "realism"],
                   "properties": {k: {"type": "integer", "minimum": 0, "maximum": 5}
                                  for k in ["layout", "geometry", "materials", "lighting", "atmosphere", "realism"]}},
    },
}
JUDGE_SYSTEM = """You grade a real-time 3D reconstruction of a photograph. You see the photograph first, then the \
render from the same camera. Observe before you score: say what the render shows, list concrete issues with the \
thing they are about, judge each acceptance item from what is visible, then score the rubric 0-5 (5 = \
indistinguishable from a photograph of the place). The render is a game-engine frame, not a photo: judge whether it \
reads as the same place with believable geometry, surfaces, light and air, not pixel identity."""
_PENALTY = {"minor": 0.02, "major": 0.06, "severe": 0.12}
_WEIGHTS = {"layout": 0.25, "geometry": 0.2, "materials": 0.15, "lighting": 0.15, "atmosphere": 0.1, "realism": 0.15}


def judge(photo: Image.Image, render: Image.Image, plan: dict | None, model_id: str, out: Path,
          run: Path | None = None, base_url: str | None = None) -> dict | None:
    """Vision-model verdict on the photo view (server/agent/critique.py): issues targeted at
    scene-graph nodes when the run carries scene_graph.json; Anthropic or an OpenAI-compatible
    server by the model id's prefix. Written to out/verdict.json."""
    sys.path.insert(0, str(ROOT / "server"))
    from agent.critique import judge as _judge
    graph = None
    if run is not None and (run / "scene_graph.json").exists():
        try:
            from pipeline.scene_graph import SceneGraph
            graph = SceneGraph.load(run)
        except Exception as exc:
            print(f"  judge: scene graph not loaded ({exc}); untargeted issues")
    return _judge(photo, render, plan, model_id, out, graph=graph, base_url=base_url)


# ── sheet ───────────────────────────────────────────────────────────────────────

def _row(images: list[Image.Image], labels: list[str], w: int) -> Image.Image:
    h = int(round(w * images[0].height / images[0].width))
    row = Image.new("RGB", (len(images) * (w + 6), h + 18), (250, 250, 250))
    d = ImageDraw.Draw(row)
    for i, (im, lab) in enumerate(zip(images, labels)):
        row.paste(im.convert("RGB").resize((w, h), Image.LANCZOS), (i * (w + 6), 18))
        d.text((i * (w + 6) + 3, 3), lab, fill=(20, 20, 20))
    return row


def make_sheet(photo: Image.Image, now: dict[str, Image.Image], prev: dict[str, Image.Image] | None,
               label: str, prev_label: str | None, w: int = 360) -> Image.Image:
    rows = [_row([photo] + [now[v] for v in VIEWS], ["input photograph"] + [f"{label}: {v}" for v in VIEWS], w)]
    if prev:
        blank = Image.new("RGB", photo.size, (250, 250, 250))
        rows.append(_row([blank] + [prev.get(v, blank) for v in VIEWS],
                         [""] + [f"{prev_label}: {v}" for v in VIEWS], w))
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows) + 8 * (len(rows) - 1)), (255, 255, 255))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 8
    return sheet


# ── main ────────────────────────────────────────────────────────────────────────

def git_rev() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
        return rev + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="*", default=DEFAULT_SCENES)
    ap.add_argument("--label", default=None, help="name for this check (default: timestamp)")
    ap.add_argument("--out", type=Path, default=ROOT / "scratch" / "check")
    ap.add_argument("--width", type=int, default=720, help="render width per view")
    ap.add_argument("--grass", action="store_true", help="render grass blades (slow)")
    ap.add_argument("--judge", action="store_true", help="vision-model verdict on the photo view (cents per check)")
    ap.add_argument("--judge-model", default="anthropic:claude-sonnet-5-5")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible server for an openai:<model> judge")
    args = ap.parse_args()
    label = args.label or dt.datetime.now().strftime("%m%d-%H%M")
    scorer = Scorer()
    summary = []

    for scene_arg in args.scenes:
        name, run = resolve(scene_arg)
        base = args.out / name.replace(" ", "_")
        out = base / label
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "scripts" / "render-scene.py"), str(run), "--out", str(out),
               "--views", ",".join(VIEWS), "--width", str(args.width), "--save-depth"]
        if not args.grass:
            cmd.append("--no-grass")
        print(f"[{name}] rendering {run.name} ...", flush=True)
        res = subprocess.run(cmd, cwd=ROOT / "server", capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-2000:], res.stderr[-2000:])
            raise SystemExit(f"render failed for {name}")

        photo = Image.open(run / "input.png").convert("RGB")
        now = {v: Image.open(out / f"{v}.png").convert("RGB") for v in VIEWS}
        record = {"label": label, "time": dt.datetime.now().isoformat(timespec="seconds"),
                  "git": git_rev(), "run": run.name}
        record.update(scorer.score(photo, now["photo"]))
        record.update(geometry_flags(run))
        views = {}
        for v in VIEWS:
            dp = out / f"{v}_depth.npy"
            views[v] = image_flags(now[v], np.load(dp).astype(np.float32) if dp.exists() else None)
        record["views"] = views
        record["black"] = round(sum(f["black"] for f in views.values()), 4)
        record["flat"] = round(sum(f["flat"] for f in views.values()), 4)
        plan = plan_summary(run)
        if plan is not None:
            record["plan"] = {k: plan[k] for k in ("model", "source", "camera", "sun", "water", "terrain")}
            record["plan_overrides"] = plan["overrides"]
        if args.judge:
            verdict = judge(photo, now["photo"], plan, args.judge_model, out, run, base_url=args.base_url)
            if verdict is not None:
                record["judge"] = verdict["score"]
                record["acceptance_pass"] = verdict["acceptance_pass"]
                record["judge_issues"] = len(verdict["issues"])
                print(f"[{name}] judge {verdict['score']:.2f} (rubric {verdict['rubric_score']:.2f} - {verdict['penalty']:.2f}"
                      f"{', must failed: ' + ', '.join(verdict['failed_must']) if verdict['failed_must'] else ''}); "
                      f"acceptance {verdict['acceptance_pass']}; {verdict['summary']}", flush=True)
                for i in verdict["issues"]:
                    print(f"    {str(i.get('severity') or '?'):6s} {str(i.get('kind') or '?'):11s} {i.get('node') or '-'} {i.get('target')}: {i.get('text')}")

        hist_path = base / "history.jsonl"
        history = [json.loads(l) for l in hist_path.read_text().splitlines() if l.strip()] if hist_path.exists() else []
        prev = next((h for h in reversed(history) if h["label"] != label), None)
        prev_imgs = None
        if prev and (base / prev["label"]).is_dir():
            prev_imgs = {v: Image.open(base / prev["label"] / f"{v}.png") for v in VIEWS
                         if (base / prev["label"] / f"{v}.png").exists()}
        make_sheet(photo, now, prev_imgs, label, prev["label"] if prev else None).save(out / "sheet.png")
        (out / "check.json").write_text(json.dumps(record, indent=2))
        history = [h for h in history if h["label"] != label] + [record]
        hist_path.write_text("".join(json.dumps(h) + "\n" for h in history))
        summary.append((name, record, prev))
        print(f"[{name}] sheet: {out / 'sheet.png'}", flush=True)

    keys = ["dino_global", "dino_layout", "colour_de", "judge", "black", "flat", "oversize", "boat_tall", "treeline_tall"]
    print()
    print(f"{'scene':<15}" + "".join(f"{k:>20}" for k in keys))
    for name, rec, prev in summary:
        cells = []
        for k in keys:
            v = rec.get(k)
            cell = f"{v}"
            if prev is not None and isinstance(v, (int, float)) and isinstance(prev.get(k), (int, float)):
                d = v - prev[k]
                if abs(d) > 1e-9:
                    cell += f" ({d:+.3g})"
            cells.append(f"{cell:>20}")
        print(f"{name:<15}" + "".join(cells))
        for k in ("oversize_examples", "boat_tall_examples"):
            if rec.get(k):
                print(f"    {k}: {', '.join(rec[k])}")
        if rec.get("plan"):
            pl = rec["plan"]
            print(f"    plan ({pl.get('model')}, {pl.get('source')}): camera {pl['camera']} sun {pl['sun']} water {pl['water']}")
            for o in rec.get("plan_overrides", []):
                if o.get("source") not in (None, "measured", "n/a"):
                    print(f"    prior {o['stage']}: {o['field']} measured {o.get('measured')} -> {o.get('used')} "
                          f"({o.get('source')}, plan {o.get('plan')} {o.get('evidence')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
