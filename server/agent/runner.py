"""The runner: graph edits in, a rebuilt scene out (docs/agentic-scene-graph-plan.md §3, §4).

An edit names a graph node, an attribute and a value. `knobs.yaml` says what each editable
attribute drives: config keys of stages (written into a derived config.yaml the run is
launched with) and/or fields of the cached Scene Plan (patched in the run directory, the
original kept beside it). The earliest stage any edit touches is where the pipeline reruns
from; everything before it stays cached, exactly as scripts/hybrid-run.sh does by hand.

Nothing here calls a language model. The GPU rule applies: a run is refused while another
`main.py run|server` or a render-scene pan is alive.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "server" / "config.yaml"
KNOBS = Path(__file__).with_name("knobs.yaml")


# ── knobs ─────────────────────────────────────────────────────────────────────

@dataclass
class Knob:
    id: str
    node: str                       # node id, may contain {class}
    attr: str
    rerun: str                      # stage to rerun from
    type: str = "float"
    min: Optional[float] = None
    max: Optional[float] = None
    unit: str = ""
    choices: Optional[list] = None
    targets: list[dict] = field(default_factory=list)
    doc: str = ""
    cost: str = "cheap"             # cheap | medium | expensive: what the rerun costs

    def matches(self, node_id: str, attr: str) -> Optional[dict]:
        """The {class} binding when (node_id, attr) is this knob, else None."""
        if attr != self.attr:
            return None
        if "{class}" not in self.node:
            return {} if node_id == self.node else None
        pat = "^" + re.escape(self.node).replace(r"\{class\}", r"(?P<cls>[a-z0-9_]+)") + "$"
        m = re.match(pat, node_id)
        return {"class": m.group("cls")} if m else None

    def coerce(self, value: Any) -> Any:
        if self.type == "float":
            v = float(value)
            if self.min is not None and v < self.min or self.max is not None and v > self.max:
                raise ValueError(f"{self.id}: {v} outside [{self.min}, {self.max}]")
            return v
        if self.type == "int":
            v = int(round(float(value)))
            if self.min is not None and v < self.min or self.max is not None and v > self.max:
                raise ValueError(f"{self.id}: {v} outside [{self.min}, {self.max}]")
            return v
        if self.type == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if self.type == "choice":
            if self.choices and value not in self.choices:
                raise ValueError(f"{self.id}: {value!r} not one of {self.choices}")
            return value
        return value


def load_knobs(path: Path = KNOBS) -> list[Knob]:
    raw = yaml.safe_load(path.read_text())["knobs"]
    return [Knob(**{k: v for k, v in r.items()}) for r in raw]


def knob_catalogue(knobs: list[Knob], graph=None) -> str:
    """The edit vocabulary as text for the orchestrator prompt: one line per knob, with the
    nodes that exist in the graph substituted for {class} patterns."""
    lines = []
    for k in knobs:
        nodes = [k.node]
        if graph is not None and "{class}" in k.node:
            nodes = [n for n in graph.node_ids() if k.matches(n, k.attr) is not None]
            if not nodes:
                continue
        rng = ""
        if k.type in ("float", "int") and (k.min is not None or k.max is not None):
            rng = f" [{k.min if k.min is not None else ''}..{k.max if k.max is not None else ''}{(' ' + k.unit) if k.unit else ''}]"
        elif k.type == "choice":
            rng = f" {k.choices}"
        elif k.type == "bool":
            rng = " [true|false]"
        for n in nodes:
            lines.append(f"- {n}.{k.attr}{rng} ({k.type}; rerun {k.rerun}, {k.cost}): {k.doc}")
    return "\n".join(lines)


# ── edits → config + plan ──────────────────────────────────────────────────────

@dataclass
class Edit:
    node: str
    attr: str
    value: Any
    why: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Edit":
        return cls(str(d["node"]), str(d["attr"]), d.get("value"), str(d.get("why", "")))


@dataclass
class Plan:
    """What a set of edits does to the pipeline."""
    config_sets: list[tuple[str, str, Any]] = field(default_factory=list)   # (stage, dotted key, value)
    plan_patches: list[tuple[str, Any]] = field(default_factory=list)       # (dotted path under plan, value)
    rerun_from: Optional[str] = None
    applied: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)


def stage_order(config_path: Path = CONFIG) -> list[str]:
    return [e["name"] for e in yaml.safe_load(config_path.read_text())["stages"]]


def resolve_edits(edits: list[Edit], knobs: list[Knob], order: Optional[list[str]] = None) -> Plan:
    order = order or stage_order()
    plan = Plan()
    earliest: Optional[int] = None
    for e in edits:
        hit = None
        for k in knobs:
            b = k.matches(e.node, e.attr)
            if b is not None:
                hit = (k, b)
                break
        if hit is None:
            plan.rejected.append({"node": e.node, "attr": e.attr, "value": e.value, "reason": "no knob"})
            continue
        k, binding = hit
        try:
            value = k.coerce(e.value)
        except (TypeError, ValueError) as exc:
            plan.rejected.append({"node": e.node, "attr": e.attr, "value": e.value, "reason": str(exc)})
            continue
        for t in k.targets:
            v = t.get("value", value)
            if "plan" in t:
                plan.plan_patches.append((t["plan"].format(**binding), v))
            else:
                plan.config_sets.append((t["stage"], t["key"].format(**binding), v))
        if k.rerun in order:
            i = order.index(k.rerun)
            earliest = i if earliest is None else min(earliest, i)
        plan.applied.append({"node": e.node, "attr": e.attr, "value": value, "knob": k.id, "why": e.why})
    plan.rerun_from = order[earliest] if earliest is not None else None
    return plan


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    for p in parts[:-1]:
        nxt = d.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            d[p] = nxt
        d = nxt
    d[parts[-1]] = value


def write_derived_config(sets: list[tuple[str, str, Any]], out_path: Path,
                         base: Path = CONFIG) -> Path:
    """config.yaml with the sets applied (comments are lost in the derived copy; the base
    file is never touched)."""
    cfg = yaml.safe_load(base.read_text())
    by_name = {e["name"]: e for e in cfg["stages"]}
    for stage, key, value in sets:
        if stage not in by_name:
            raise KeyError(f"derived config: no stage {stage!r}")
        _set_dotted(by_name[stage], key, value)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return out_path


def patch_plan(run_dir: Path, patches: list[tuple[str, Any]], who: str) -> Optional[Path]:
    """Apply (dotted path, value) to the cached Scene Plan in the run directory. The first
    patch keeps the original as scene_plan.orig.json; every patch is listed in the record."""
    if not patches:
        return None
    pf = run_dir / "Scene Plan" / "scene_plan.json"
    if not pf.exists():
        raise FileNotFoundError(f"no cached plan at {pf}")
    orig = pf.with_name("scene_plan.orig.json")
    if not orig.exists():
        shutil.copy2(pf, orig)
    rec = json.loads(pf.read_text())
    plan = rec.setdefault("plan", {})
    for path, value in patches:
        _set_dotted(plan, path, value)
        parts = path.split(".")
        sec = plan.get(parts[0])
        if isinstance(sec, dict) and "evidence" in sec:
            sec["evidence"] = "inferred"
            sec["because"] = f"{sec.get('because', '')} [{who} set {path}={value}]".strip()
    rec.setdefault("agent_patches", []).append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "who": who,
                                                "patches": [{"path": p, "value": v} for p, v in patches]})
    pf.write_text(json.dumps(rec, indent=2))
    return pf


def restore_plan(run_dir: Path) -> bool:
    pf = run_dir / "Scene Plan" / "scene_plan.json"
    orig = pf.with_name("scene_plan.orig.json")
    if orig.exists():
        shutil.copy2(orig, pf)
        return True
    return False


# ── running ───────────────────────────────────────────────────────────────────

def gpu_busy() -> Optional[str]:
    """The command line of a pipeline run, server or pan render that holds the GPU, else None."""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if re.search(r"(^|/)python3? main\.py .*\b(run|server)\b", line) or "render-scene.py" in line:
            if "agent-loop" in line or "runner.py" in line:
                continue
            return line.strip()
    return None


def run_pipeline(photo: Path, rerun_from: str, config_path: Path, log_path: Path,
                 output_dir: Path = ROOT / "output", timeout_s: float = 4 * 3600) -> int:
    """`main.py run <photo> --rerun <stage> --config <derived>`; returns the exit code."""
    busy = gpu_busy()
    if busy:
        raise RuntimeError(f"GPU is held by: {busy}")
    env = dict(os.environ, HF_HUB_OFFLINE="1", RENDER_GRASS_MAX=os.environ.get("RENDER_GRASS_MAX", "60000"))
    # --no-debug-archive: the .debug.frame bundle takes ~6 min a run and nothing in the loop reads it
    cmd = [sys.executable, "main.py", "--log-mode", "plain", "run", str(photo), "--output", str(output_dir),
           "--rerun", rerun_from, "--config", str(config_path), "--no-debug-archive"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        p = subprocess.run(cmd, cwd=str(ROOT / "server"), stdout=log, stderr=subprocess.STDOUT, env=env,
                           timeout=timeout_s)
    return p.returncode


def run_check(scene: str, label: str, judge_model: Optional[str], log_path: Path, grass: bool = True,
              extra: Optional[list[str]] = None) -> int:
    """scripts/scene-check.py for one scene (renders, metrics, sheet, verdict)."""
    busy = gpu_busy()
    if busy:
        raise RuntimeError(f"GPU is held by: {busy}")
    env = dict(os.environ, HF_HUB_OFFLINE="1", RENDER_GRASS_MAX=os.environ.get("RENDER_GRASS_MAX", "60000"))
    cmd = [sys.executable, str(ROOT / "scripts" / "scene-check.py"), scene, "--label", label]
    if grass:
        cmd.append("--grass")
    if judge_model:
        cmd += ["--judge", "--judge-model", judge_model]
    cmd += extra or []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)
    return p.returncode


def read_check(scene: str, label: str) -> tuple[Optional[dict], Optional[dict]]:
    base = ROOT / "scratch" / "check" / scene.replace(" ", "_") / label
    check = json.loads((base / "check.json").read_text()) if (base / "check.json").exists() else None
    verdict = json.loads((base / "verdict.json").read_text()) if (base / "verdict.json").exists() else None
    return check, verdict


def resolve_run(scene: str) -> Optional[Path]:
    """output/<uuid> holding this scene's photograph, by thumbnail (as hybrid-run.sh)."""
    import numpy as np
    from PIL import Image
    photo = ROOT / "server" / "samples" / f"{scene}.jpg"
    if not photo.exists():
        return None

    def thumb(p):
        return np.asarray(Image.open(p).convert("RGB").resize((32, 32), Image.BOX), dtype=np.float32)

    want = thumb(photo)
    best = None
    for run in (ROOT / "output").glob("*-*-*-*-*"):
        inp = run / "input.png"
        if inp.is_file():
            err = float(np.abs(thumb(inp) - want).mean())
            if err < 6 and (best is None or err < best[0]):
                best = (err, run)
    return best[1] if best else None
