"""The issue-driven loop: the judge's issues are the state, scene actions are the moves.

    round 0   render the scene as it stands from the capture camera, judge it against the
              photograph: a list of issues (node, kind, severity, text, photo_box) and the
              plan's acceptance items.
    round n   take the worst open issue; show the fixer (a vision model) the photograph, the
              render, both cropped to the issue's region, the placed objects, the plan's
              inventory and the action vocabulary (agent/editor.py); it answers with one to
              three actions. Apply them to a copy of the scene, render it (a minute, no
              pipeline), judge it with FOCUS on that issue. Keep the actions when the judge
              says the issue is resolved and nothing got worse (no new severe issue, score
              not down by more than the noise, no black frame); otherwise revert and try the
              issue once more with the failure in the history, then move on.
    keep      = append to <run>/scene_edits.json, note in the scene graph, replay the Scene
              Edits stage into the run so its newest scene.json carries the edit.

Scores are a guardrail, not the objective: the loop ends when no open issue is left in
its reach, the round or dollar cap is hit, or `patience` rounds in a row failed.
Output: scratch/fix/<label>/{loop.log, report.md, round_N/{photo.png, verdict.json,
proposal.json, actions.json, scene.json, records.json}}.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SEVERITY_RANK = {"severe": 0, "major": 1, "minor": 2}
# Issue kinds the vocabulary can answer. material/style/motion need generation; the
# loop lists them in the report as out of reach.
ACTIONABLE = {"missing", "extra", "scale", "composition", "geometry", "atmosphere", "lighting"}

FIX_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["diagnosis", "actions", "expect", "give_up", "give_up_reason"],
    "properties": {
        "diagnosis": {"type": "string", "description": "two sentences: what is wrong at the target region and which objects (by name/id) or scene settings are responsible"},
        "actions": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False, "required": ["action", "why"],
            "properties": {
                "action": {"type": "string", "enum": ["place", "move", "scale", "rotate", "remove", "swap_mesh", "set_atmosphere", "set_sun", "set_light"]},
                "target": {"type": "string", "description": "object id, unique name or thing_<sourceIndex> (move/scale/rotate/remove/swap_mesh)"},
                "mesh": {"type": "string", "description": "a mesh key from the list (place/swap_mesh); a swap keeps the object's length in metres when it is longer than tall, else its height"},
                "class": {"type": "string"}, "name": {"type": "string"},
                "at": {"type": "object", "additionalProperties": False,
                       "properties": {"photo_box": {"type": "array", "items": {"type": "number"}, "description": "[x, y, w, h] fractions of the PHOTOGRAPH where the object should appear; the editor casts the box's bottom centre onto the terrain or water"},
                                      "bearing_deg": {"type": "number"}, "distance_m": {"type": "number"},
                                      "world": {"type": "array", "items": {"type": "number"}},
                                      "delta": {"type": "array", "items": {"type": "number"}, "description": "[dx, dy, dz] metres, move only; x right, y up, z forward"}}},
                "height_m": {"type": "number", "description": "place/scale: the object's vertical height in metres"},
                "length_m": {"type": "number", "description": "place/scale: the object's longest horizontal extent in metres; use this for boats, bridges, buildings and anything wider than it is tall"},
                "yaw_deg": {"type": "number"}, "delta_deg": {"type": "number"}, "factor": {"type": "number"},
                "on": {"type": "string", "enum": ["ground", "water", "none"]},
                "haze_near_m": {"type": "number"}, "haze_far_m": {"type": "number"},
                "azimuth_deg": {"type": "number"}, "elevation_deg": {"type": "number"}, "intensity": {"type": "number"}, "color": {"type": "string"},
                "sun_scale": {"type": "number", "description": "set_light: multiplier on the sun"}, "ambient_scale": {"type": "number", "description": "set_light: multiplier on the ambient (fill) light"},
                "why": {"type": "string"}}}},
        "expect": {"type": "string", "description": "what the crop of the target region should show after the actions, one sentence"},
        "give_up": {"type": "boolean", "description": "true when no action in the vocabulary can address this issue (it needs new content: a texture, a mesh the run does not have)"},
        "give_up_reason": {"type": "string"},
    },
}

SYSTEM = """You fix one issue in a real-time 3D reconstruction of a photograph by editing the placed scene. \
You see the photograph, the current render from the same camera, then the photograph and the render cropped \
to the issue's region. You are given the placed objects (name, id, bearing and distance from the camera, size, \
and where each appears in the photograph), the plan's inventory of what the photograph contains, the mesh keys \
this run can place, the history of earlier attempts at this issue, and the action vocabulary.

Rules. Address only the stated issue, with one to three actions. Positions come from the PHOTOGRAPH: give an \
object's photo_box (fractions of the photograph's width and height) where it should stand and let the editor \
find the ground; give bearing and distance only when the thing has no footprint in the photograph. Sizes in \
metres, from the plan or from what the class can be: height_m for things taller than wide, length_m for boats, \
bridges and buildings (each object's listing gives its current length and height). Place only mesh keys from \
the list; a missing object with no suitable mesh is a give_up. Do not touch the terrain, water, treeline or \
backdrop meshes. A region that renders black or far darker than the photograph while its surface is present \
is usually the exposure fit, not missing content: try set_light (sun_scale, ambient_scale above 1) before \
giving up on it. Do not repeat an attempt the history says failed. When the issue cannot be answered with \
these actions, set give_up and say what would be needed."""


def _bool(v) -> bool:
    return bool(v) and str(v).lower() not in ("false", "0", "no")


@dataclass
class FixConfig:
    scene: str
    label: str
    rounds: int = 12
    max_usd: float = 10.0
    model: str = "anthropic:claude-opus-5-5"
    judge_model: str = "anthropic:claude-sonnet-5-5"
    base_url: Optional[str] = None
    width: int = 1200
    grass: bool = True
    patience: int = 3
    attempts_per_issue: int = 2
    score_drop: float = 0.05
    dry_run: bool = False
    only_kinds: Optional[set] = None
    baseline: Optional[str] = None   # an existing scratch/check label: its photo.png and verdict.json stand in for round 0


@dataclass
class Attempt:
    round: int
    issue: dict
    actions: list
    applied: list
    verdict: Optional[dict]
    kept: bool
    reason: str
    usd: float = 0.0


class FixLoop:
    def __init__(self, cfg: FixConfig):
        self.cfg = cfg
        sys.path.insert(0, str(ROOT / "server"))
        from agent.runner import resolve_run
        run = resolve_run(cfg.scene)
        if run is None:
            raise SystemExit(f"no cached run for '{cfg.scene}'")
        self.run = run
        self.out = ROOT / "scratch" / "fix" / cfg.label
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out / "loop.log"
        self.photo = Image.open(run / "input.png").convert("RGB")
        plan_file = run / "Scene Plan" / "scene_plan.json"
        d = json.loads(plan_file.read_text()) if plan_file.is_file() else {}
        self.plan = d.get("plan", d) if isinstance(d, dict) else {}
        self.graph = None
        try:
            from pipeline.scene_graph.graph import SceneGraph
            self.graph = SceneGraph.from_run(run)
        except Exception:
            pass
        self.usd = 0.0
        self.attempts: list[Attempt] = []
        self.issues: list[dict] = []
        self.baseline: Optional[dict] = None
        self.context = None

    # ── helpers ──────────────────────────────────────────────────────────────
    def say(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with self.log_path.open("a") as f:
            f.write(line + "\n")

    def _context(self):
        if self.context is None:
            from agent.editor import load_context
            self.context = load_context(self.run, ROOT / "server" / "config.yaml")
        return self.context

    def render(self, scene_path: Path, out: Path) -> Optional[Image.Image]:
        from agent.runner import gpu_busy
        busy = gpu_busy()
        if busy:
            self.say(f"  GPU is held by {busy}; cannot render")
            return None
        cmd = [sys.executable, str(ROOT / "scripts" / "render-scene.py"), str(self.run), "--scene", str(scene_path),
               "--out", str(out), "--views", "photo", "--width", str(self.cfg.width)]
        if not self.cfg.grass:
            cmd.append("--no-grass")
        env = dict(os.environ, HF_HUB_OFFLINE="1")
        env.setdefault("RENDER_GRASS_MAX", "60000")
        t0 = time.time()
        res = subprocess.run(cmd, cwd=ROOT / "server", capture_output=True, text=True, env=env)
        (out / "render.log").write_text(res.stdout + res.stderr)
        if res.returncode != 0 or not (out / "photo.png").is_file():
            self.say(f"  render failed ({res.returncode}): {res.stderr.strip().splitlines()[-1:] if res.stderr else ''}")
            return None
        self.say(f"  rendered in {time.time() - t0:.0f} s")
        return Image.open(out / "photo.png").convert("RGB")

    def judge(self, render: Image.Image, out: Path, focus: Optional[dict] = None) -> Optional[dict]:
        from agent.critique import judge
        v = judge(self.photo, render, self.plan, self.cfg.judge_model, out, graph=self.graph,
                  base_url=self.cfg.base_url, focus=focus)
        if v:
            self.usd += float(v.get("usd") or 0.0)
            v["black"] = black_fraction(render)
        return v

    @staticmethod
    def open_issues(verdict: Optional[dict]) -> list[dict]:
        issues = [i for i in (verdict or {}).get("issues", []) if isinstance(i, dict)]
        issues.sort(key=lambda i: (SEVERITY_RANK.get(str(i.get("severity")), 3), 0 if i.get("photo_box") else 1))
        return issues

    def _tries(self, issue: dict) -> list[Attempt]:
        key = _issue_key(issue)
        return [a for a in self.attempts if _issue_key(a.issue) == key]

    def pick(self, verdict: dict) -> Optional[dict]:
        kinds = self.cfg.only_kinds or ACTIONABLE
        for i in self.open_issues(verdict):
            if str(i.get("kind")) not in kinds:
                continue
            tries = self._tries(i)
            if any(a.kept for a in tries) or len(tries) >= self.cfg.attempts_per_issue:
                continue
            if any(a.reason.startswith("give up") for a in tries):
                continue
            return i
        return None

    # ── the loop ─────────────────────────────────────────────────────────────
    def run_loop(self) -> dict:
        from agent.editor import newest_scene_path
        cfg = self.cfg
        self.say(f"fix loop {cfg.label}: {cfg.scene} ({self.run.name}); rounds {cfg.rounds}, budget ${cfg.max_usd:.2f}, "
                 f"fixer {cfg.model}, judge {cfg.judge_model}")
        r0 = self.out / "round_0"
        r0.mkdir(exist_ok=True)
        scene_path = newest_scene_path(self.run)
        verdict = None
        if cfg.baseline:
            src = ROOT / "scratch" / "check" / cfg.scene.replace(" ", "_") / cfg.baseline
            if (src / "photo.png").is_file() and (src / "verdict.json").is_file():
                import shutil
                shutil.copy2(src / "photo.png", r0 / "photo.png")
                shutil.copy2(src / "verdict.json", r0 / "verdict.json")
                render = Image.open(r0 / "photo.png").convert("RGB")
                verdict = json.loads((r0 / "verdict.json").read_text())
                verdict["black"] = black_fraction(render)
                self.say(f"  round 0 from scene-check {cfg.baseline}")
            else:
                self.say(f"  no photo.png + verdict.json under {src}; rendering round 0")
        if verdict is None:
            render = self.render(scene_path, r0)
            if render is None:
                return self.report("no baseline render")
            verdict = self.judge(render, r0)
            if verdict is None:
                return self.report("no baseline verdict")
        self.baseline = verdict
        current = verdict
        self.say(f"  round 0: judge {verdict.get('score')} acceptance {verdict.get('acceptance_pass')} "
                 f"issues {len(verdict.get('issues', []))}: " + "; ".join(
                     f"[{i.get('severity')}/{i.get('kind')}] {i.get('target')}" for i in self.open_issues(verdict)[:8]))
        fails = 0
        for r in range(1, cfg.rounds + 1):
            if self.usd >= cfg.max_usd:
                self.say(f"budget ${cfg.max_usd:.2f} reached")
                break
            issue = self.pick(current)
            if issue is None:
                self.say("no open issue in reach")
                break
            rd = self.out / f"round_{r}"
            rd.mkdir(exist_ok=True)
            self.say(f"round {r}: [{issue.get('severity')}/{issue.get('kind')}] {issue.get('target')}: {str(issue.get('text'))[:140]}")
            base_render = Image.open(self.out / f"round_{self._last_kept_round()}" / "photo.png").convert("RGB")
            try:
                proposal, usage = propose_fix(cfg.model, self, issue, base_render, scene_path, rd, base_url=cfg.base_url)
            except Exception as exc:
                self.say(f"  fixer failed: {type(exc).__name__}: {exc}")
                fails += 1
                if fails >= cfg.patience:
                    break
                continue
            self.usd += float(usage.get("usd") or 0.0)
            (rd / "proposal.json").write_text(json.dumps(proposal, indent=1))
            self.say(f"  fixer ({usage.get('seconds')} s, ${usage.get('usd', 0):.2f}): {proposal.get('diagnosis', '')[:200]}")
            if proposal.get("give_up") or not proposal.get("actions"):
                self.attempts.append(Attempt(r, issue, [], [], None, False, f"give up: {proposal.get('give_up_reason', '')}"))
                self.say(f"  give up: {proposal.get('give_up_reason', '')}")
                continue
            if cfg.dry_run:
                self.say("  dry run: stopping after the first proposal")
                break
            # apply
            from agent.editor import apply_all
            scene = json.loads(scene_path.read_text())
            new, applied, failures = apply_all(scene, proposal["actions"], context=self._context(), run_dir=self.run,
                                               who=f"fix:{cfg.label}:r{r}", log=lambda s: self.say("  " + s))
            (rd / "records.json").write_text(json.dumps({"applied": [{k: v for k, v in a.items() if k != "removed"} for a in applied],
                                                         "failures": failures}, indent=1))
            if not applied:
                self.attempts.append(Attempt(r, issue, proposal["actions"], [], None, False,
                                             "all actions refused: " + "; ".join(f["error"] for f in failures)))
                fails += 1
                if fails >= cfg.patience:
                    self.say(f"{fails} rounds failed in a row; stopping")
                    break
                continue
            trial = rd / "scene.json"
            trial.write_text(json.dumps(new, indent=1))
            render = self.render(trial, rd)
            if render is None:
                self.attempts.append(Attempt(r, issue, proposal["actions"], applied, None, False, "render failed"))
                fails += 1
                if fails >= cfg.patience:
                    break
                continue
            verdict = self.judge(render, rd, focus=issue)
            if verdict is None:
                self.attempts.append(Attempt(r, issue, proposal["actions"], applied, None, False, "no verdict"))
                fails += 1
                if fails >= cfg.patience:
                    break
                continue
            kept, reason = self.decide(current, verdict)
            self.attempts.append(Attempt(r, issue, proposal["actions"], applied, verdict, kept, reason))
            self.say(f"  judge {verdict.get('score')} (was {current.get('score')}), focus resolved {((verdict.get('focus') or {}).get('resolved'))}: "
                     f"{((verdict.get('focus') or {}).get('evidence') or '')[:160]}")
            if kept:
                self.accept(applied, r)
                scene_path = newest_scene_path(self.run)
                current = verdict
                fails = 0
                self.say(f"  KEPT: {reason}")
            else:
                fails += 1
                self.say(f"  reverted: {reason}")
                if fails >= cfg.patience:
                    self.say(f"{fails} rounds failed in a row; stopping")
                    break
        return self.report("done")

    def _last_kept_round(self) -> int:
        kept = [a.round for a in self.attempts if a.kept]
        return kept[-1] if kept else 0

    def decide(self, before: dict, after: dict) -> tuple[bool, str]:
        f = after.get("focus") or {}
        if after.get("black", 0.0) > before.get("black", 0.0) + 0.05:
            return False, f"black area grew to {after.get('black'):.2f}"
        sev_before = sum(1 for i in before.get("issues", []) if i.get("severity") == "severe")
        sev_after = sum(1 for i in after.get("issues", []) if i.get("severity") == "severe")
        if sev_after > sev_before:
            return False, f"new severe issue: " + "; ".join(i.get("target", "") for i in after.get("issues", []) if i.get("severity") == "severe")
        if not f.get("resolved"):
            return False, f"focus not resolved: {f.get('evidence', '')}"
        if float(after.get("score") or 0) < float(before.get("score") or 0) - self.cfg.score_drop:
            return False, f"score fell {before.get('score')} -> {after.get('score')}"
        reg = str(f.get("regressions") or "").strip()
        return True, "issue resolved" + (f"; judge notes: {reg}" if reg and reg.lower() not in ("none", "''", "nothing") else "")

    def accept(self, applied: list[dict], r: int) -> None:
        from agent.editor import load_edits, save_edits
        data = load_edits(self.run)
        data["edits"].extend(applied)
        save_edits(self.run, data)
        if self.graph is not None:
            for a in applied:
                inp = a.get("input") or {}
                node = str(inp.get("target") or "")
                node = node if node in self.graph.nodes else "world"
                try:
                    self.graph.annotate(node, f"{a.get('action')}: {a.get('note')} -- {a.get('why')}", f"fix:{self.cfg.label}:r{r}")
                except Exception:
                    pass
        cmd = [sys.executable, str(ROOT / "scripts" / "replay-stage.py"), str(self.run), "Scene Edits", "--save"]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, HF_HUB_OFFLINE="1"))
        if res.returncode != 0:
            self.say(f"  Scene Edits replay failed: {(res.stdout + res.stderr).strip().splitlines()[-1:]}")

    def report(self, status: str) -> dict:
        kept = [a for a in self.attempts if a.kept]
        last = next((a.verdict for a in reversed(self.attempts) if a.kept and a.verdict), None) or self.baseline or {}
        lines = [f"# fix loop {self.cfg.label}: {self.cfg.scene}", "",
                 f"{status}; {len(self.attempts)} attempts, {len(kept)} kept; spent ${self.usd:.2f}; "
                 f"judge {self.baseline.get('score') if self.baseline else '?'} -> {last.get('score')}; "
                 f"acceptance {self.baseline.get('acceptance_pass') if self.baseline else '?'} -> {last.get('acceptance_pass')}", ""]
        for a in self.attempts:
            lines.append(f"## round {a.round}: [{a.issue.get('severity')}/{a.issue.get('kind')}] {a.issue.get('target')} -- "
                         f"{'KEPT' if a.kept else 'reverted'}: {a.reason}")
            lines.append(f"issue: {a.issue.get('text')}")
            for rec in a.applied:
                lines.append(f"- {rec.get('action')}: {rec.get('note')} ({rec.get('why')})")
            if a.verdict:
                f = a.verdict.get("focus") or {}
                lines.append(f"judge {a.verdict.get('score')}: {f.get('evidence', '')}" + (f" | regressions: {f.get('regressions')}" if f.get("regressions") else ""))
            lines.append("")
        rem = [i for i in self.open_issues(last) if not any(a.kept and _issue_key(a.issue) == _issue_key(i) for a in self.attempts)]
        if rem:
            lines.append("## open issues")
            for i in rem:
                reach = "in reach" if str(i.get("kind")) in (self.cfg.only_kinds or ACTIONABLE) else "needs new content"
                lines.append(f"- [{i.get('severity')}/{i.get('kind')}] {i.get('target')} ({reach}): {i.get('text')}")
        (self.out / "report.md").write_text("\n".join(lines) + "\n")
        self.say(f"report: {self.out / 'report.md'}")
        return {"status": status, "attempts": len(self.attempts), "kept": len(kept), "usd": round(self.usd, 2),
                "score": last.get("score"), "baseline": (self.baseline or {}).get("score")}


def _issue_key(i: dict) -> str:
    return f"{i.get('kind')}|{i.get('node') or ''}|{str(i.get('target') or '').lower()[:40]}"


def black_fraction(im: Image.Image) -> float:
    a = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return float((a.max(axis=2) < 12).mean())


# ── the fixer call ───────────────────────────────────────────────────────────────

def objects_text(editor, run: Path, photo_size) -> str:
    """The placed objects for the fixer: populations as one line, the rest one per object."""
    import re
    meta: dict[str, dict] = {}
    for f in run.glob("*/metadata_*.json"):
        try:
            meta[f.stem.split("_", 1)[1]] = json.loads(f.read_text())
        except Exception:
            pass
    W, H = photo_size
    fam: dict[str, list[dict]] = {}
    for o in editor.objects:
        m = re.match(r"category_mesh_([a-z_]+?)_\d+", str(o.get("mesh") or ""))
        fam.setdefault(m.group(1) if m else "", []).append(o)
    lines = []
    singles = []
    for k, v in sorted(fam.items()):
        if k and len(v) >= 20:
            d = sorted(editor.bearing_distance([o["position"]["x"], o["position"]["y"], o["position"]["z"]])[1] for o in v)
            lines.append(f"population pop_{k}: {len(v)} instances at {d[0]:.0f}-{d[-1]:.0f} m (members by id or thing_<src>; a few examples follow)")
            singles.extend(sorted(v, key=lambda o: editor.bearing_distance([o["position"]["x"], o["position"]["y"], o["position"]["z"]])[1])[:3])
        else:
            singles.extend(v)
    for o in singles:
        if editor.is_env(o):
            lines.append(f"{o.get('name')} (environment mesh, not editable)")
            continue
        ext = editor.object_extent(o)
        b, dist = editor.bearing_distance(ext[2] if ext is not None else [o["position"]["x"], o["position"]["y"], o["position"]["z"]])
        m = meta.get(str(o.get("sourceIndex")), {})
        pb = m.get("photo_box")
        box = f" photo_box [{pb[0] / W:.3f}, {pb[1] / H:.3f}, {pb[2] / W:.3f}, {pb[3] / H:.3f}]" if pb and W else ""
        h = m.get("world_height")
        src = f" thing_{o.get('sourceIndex')}" if o.get("sourceIndex") is not None else ""
        size = (f", {ext[0]:.1f} m long x {ext[1]:.1f} m tall" if ext is not None
                else (f", {float(h):.1f} m tall" if h is not None else f", scale {float(o['scale']['y']):.2f}"))
        lines.append(f"{o.get('name')} id {str(o.get('id'))[:8]}{src}: mesh {o.get('mesh')}, bearing {b:.1f} deg, {dist:.1f} m{size}{box}")
    return "\n".join(lines)


def plan_text(plan: dict) -> str:
    parts = []
    for lm in plan.get("landmarks") or []:
        if isinstance(lm, dict):
            parts.append(f"landmark {lm.get('name') or lm.get('phrase')}: {lm.get('category')} bearing {lm.get('bearing_deg')} deg, "
                         f"{lm.get('distance_m')} m, {lm.get('height_m')} m tall ({lm.get('evidence')})")
    for ob in plan.get("objects") or []:
        if isinstance(ob, dict):
            parts.append(f"object {ob.get('phrase') or ob.get('class')}: {ob.get('class')} x{ob.get('count')}, {ob.get('size_m') or ob.get('height_m')} m, "
                         f"{ob.get('where') or ob.get('position')} ({ob.get('evidence')})")
    w = plan.get("water") or {}
    if isinstance(w, dict) and w.get("kind") and w.get("kind") != "none":
        parts.append(f"water: {w.get('kind')}, width {w.get('width_m')} m, level {w.get('level_below_camera_m')} m below the camera")
    for a in plan.get("acceptance") or []:
        if isinstance(a, dict):
            parts.append(f"acceptance {a.get('id')} [{a.get('priority')}]: {a.get('text')}")
    return "\n".join(parts) or "(no plan)"


def review_text(run: Path) -> str:
    """What the Inventory Review gate found: dropped and relabelled detections and the plan
    items no detection covered (with their photo_box), so the fixer knows what the first
    pass already judged absent."""
    f = run / "Inventory Review" / "review.json"
    if not f.is_file():
        return "(no inventory review)"
    try:
        r = json.loads(f.read_text())
    except Exception:
        return "(unreadable inventory review)"
    out = []
    for m in r.get("missing") or []:
        out.append(f"missing after the first pass: {m.get('count')} x {m.get('name')} ({m.get('category')}) at photo_box {m.get('photo_box')}: {m.get('why')}")
    for a in r.get("applied") or []:
        out.append(str(a))
    return "\n".join(out) or "(the review changed nothing)"


def history_text(loop: FixLoop, issue: dict) -> str:
    tries = loop._tries(issue)
    if not tries:
        return "(first attempt at this issue)"
    out = []
    for a in tries:
        acts = "; ".join(f"{r.get('action')}: {r.get('note')}" for r in a.applied) or "; ".join(x.get("action", "") for x in a.actions)
        out.append(f"round {a.round}: {acts} -> {'kept' if a.kept else 'reverted'}: {a.reason}")
    return "\n".join(out)


def propose_fix(model_id: str, loop: FixLoop, issue: dict, render: Image.Image, scene_path: Path, out: Path,
                base_url: Optional[str] = None, timeout_s: float = 600.0) -> tuple[dict, dict]:
    from agent.critique import crop_box
    from agent.editor import Editor
    from pipeline.scene_plan.client import PlanModel, image_to_jpeg_b64, estimate_usd, resolve_anthropic_key
    import agent.editor as editor_mod
    scene = json.loads(scene_path.read_text())
    ed = Editor(scene, context=loop._context(), run_dir=loop.run)
    box = issue.get("photo_box") if isinstance(issue.get("photo_box"), list) and len(issue.get("photo_box")) == 4 else None
    images = [loop.photo, render]
    if box:
        images += [crop_box(loop.photo, box), crop_box(render, box)]
    vocab = editor_mod.__doc__.split("Editor.apply returns")[0]
    user = (f"ISSUE [{issue.get('severity')}/{issue.get('kind')}] target: {issue.get('target')}; node: {issue.get('node') or '-'}; "
            f"photo_box: {box or 'none'}\n{issue.get('text')}\n\n"
            "The first image is the photograph, the second the current render from its camera"
            + (", the third and fourth the photograph and the render cropped to the issue's region." if box else ".")
            + f"\n\nPLACED OBJECTS\n{objects_text(ed, loop.run, loop.photo.size)}\n\n"
            f"MESH KEYS THIS RUN CAN PLACE\n{', '.join(ed.mesh_keys())}\n\n"
            f"PLAN (what the photograph contains)\n{plan_text(loop.plan)}\n\n"
            f"INVENTORY REVIEW (what the first pass dropped, relabelled or found missing)\n{review_text(loop.run)}\n\n"
            f"EARLIER ATTEMPTS AT THIS ISSUE\n{history_text(loop, issue)}\n\n"
            f"ACTION VOCABULARY\n{vocab}\n\nPropose the actions for this issue.")
    (out / "prompt.txt").write_text(user)
    pm = PlanModel(model_id, timeout_s=timeout_s, retries=1, base_url=base_url)
    if pm.no_api:
        raise RuntimeError("paid calls are off (FRAME_NO_API)")
    b64s = [image_to_jpeg_b64(im, 1280)[0] for im in images]
    t0 = time.time()
    if pm.provider == "anthropic":
        import anthropic
        key = resolve_anthropic_key()
        if not key:
            raise RuntimeError("no Anthropic key")
        client = anthropic.Anthropic(api_key=key, timeout=timeout_s, max_retries=1)
        content: list[dict] = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}} for b in b64s]
        content.append({"type": "text", "text": user})
        msg = client.messages.create(model=pm.model, max_tokens=3000, system=SYSTEM,
                                     tools=[{"name": "fix", "description": "Record the actions for this issue.", "input_schema": FIX_SCHEMA}],
                                     tool_choice={"type": "auto"}, messages=[{"role": "user", "content": content}])
        data = next((b.input for b in msg.content if getattr(b, "type", "") == "tool_use"), None)
        if not isinstance(data, dict):
            text = "".join(getattr(b, "text", "") for b in msg.content)
            raise RuntimeError(f"no fix tool call in the reply: {text[:300]}")
        usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens,
                 "usd": estimate_usd(pm.model, msg.usage.input_tokens, msg.usage.output_tokens)}
    else:
        import httpx
        from agent.critique import parse_json_object
        base = (pm.base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1").rstrip("/")
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in b64s]
        content.append({"type": "text", "text": user + "\nReply with the JSON object only."})
        body: dict[str, Any] = {"model": pm.model, "max_tokens": 3000, "temperature": 0.2,
                                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
        strict = dict(body, response_format={"type": "json_schema", "json_schema": {"name": "fix", "schema": FIX_SCHEMA, "strict": True}})
        data, last = None, None
        for mode, payload in (("json_schema", strict), ("plain", body)):
            try:
                r = httpx.post(f"{base}/chat/completions", json=payload, timeout=timeout_s,
                               headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'none')}"})
                r.raise_for_status()
                j = r.json()
                text = j["choices"][0]["message"]["content"].strip()
                (out / "proposal_raw.txt").write_text(f"[{mode}]\n{text}\n")
                data = parse_json_object(text, required=("actions", "diagnosis"))
                u = j.get("usage", {})
                usage = {"input_tokens": int(u.get("prompt_tokens", 0)), "output_tokens": int(u.get("completion_tokens", 0)), "usd": 0.0}
                break
            except Exception as exc:
                last = f"{mode}: {type(exc).__name__}: {exc}"
        if data is None:
            raise RuntimeError(f"OpenAI-compatible fix failed: {last}")
    usage.update({"seconds": round(time.time() - t0, 1), "model": f"{pm.provider}:{pm.model}"})
    data.setdefault("actions", [])
    data.setdefault("give_up", False)
    data.setdefault("give_up_reason", "")
    data["actions"] = [a for a in data["actions"] if isinstance(a, dict)]
    for a in data["actions"]:
        a.setdefault("why", data.get("diagnosis", "")[:200])
    return data, usage
