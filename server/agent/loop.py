"""The agent loop: perceive -> hypothesise -> build -> observe -> critique -> revise
(docs/agentic-scene-graph-plan.md §4), over one scene.

Round 0 observes the scene as it stands (scene-check: renders, metrics, the critic). Each
later round asks the orchestrator for edits in the knob vocabulary, applies them to the
graph (logged), derives a config and patches the plan, reruns the pipeline from the
earliest touched stage, re-exports the graph, and checks again. It stops on the model's
say-so, on the budget, after two rounds without improvement, or on a pipeline failure.
The best round by critic score (DINO as the tie-break) is reported; nothing is rolled
back automatically, since every round's config and plan patches are on disk.

Everything a round did is under scratch/agent/<label>/round_<n>/: proposal.json, the
derived config.yaml, pipeline.log, check.log, and the scene-check sheet lives at
scratch/check/<Scene_>/<label>-r<n>/.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from agent import runner
from agent.orchestrator import build_user, propose
from agent.runner import Edit, Knob, load_knobs, knob_catalogue, resolve_edits

ROOT = runner.ROOT


@dataclass
class LoopConfig:
    scene: str
    label: str
    rounds: int = 3
    max_usd: float = 10.0
    model: str = "anthropic:claude-opus-5-5"
    judge_model: Optional[str] = "anthropic:claude-sonnet-5-5"
    base_url: Optional[str] = None
    dry_run: bool = False
    allow_expensive: bool = False
    baseline: Optional[str] = None       # an existing scene-check label to use as round 0
    grass: bool = True
    patience: int = 2
    max_edits: int = 5


@dataclass
class RoundRecord:
    round: int
    label: str
    proposal: Optional[dict] = None
    applied: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    rerun_from: Optional[str] = None
    scores: dict = field(default_factory=dict)
    usd: float = 0.0
    outcome: str = ""
    seconds: float = 0.0

    def to_json(self) -> dict:
        return self.__dict__


class AgentLoop:
    def __init__(self, cfg: LoopConfig):
        self.cfg = cfg
        self.run_dir = runner.resolve_run(cfg.scene)
        if self.run_dir is None:
            raise FileNotFoundError(f"no cached run for scene {cfg.scene!r} under output/")
        self.photo = ROOT / "server" / "samples" / f"{cfg.scene}.jpg"
        self.out = ROOT / "scratch" / "agent" / cfg.label
        self.out.mkdir(parents=True, exist_ok=True)
        self.knobs: list[Knob] = load_knobs()
        self.sets: dict[tuple[str, str], Any] = {}     # cumulative config sets, later rounds win
        self.rounds: list[RoundRecord] = []
        self.usd = 0.0
        self.graph = None
        self.log_path = self.out / "loop.log"

    # ── helpers ───────────────────────────────────────────────────────────────
    def say(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with self.log_path.open("a") as f:
            f.write(line + "\n")

    def export_graph(self):
        from pipeline.scene_graph.export import export_graph
        g = export_graph(self.run_dir)
        g.save(self.run_dir)
        # the exporter reads the base config; re-apply the loop's own edits so the graph
        # shows the values the last run actually used
        for (stage, key), value in self.sets.items():
            for k in self.knobs:
                for t in k.targets:
                    if t.get("stage") == stage and "{class}" not in t.get("key", "") and t.get("key") == key and "value" not in t:
                        node = g.find(k.node)
                        if node is not None:
                            node.set(k.attr, value, "fitted", k.unit, 1.0, "agent")
        self.graph = g
        return g

    def check_label(self, r: int) -> str:
        return f"{self.cfg.label}-r{r}"

    def check_dir(self, label: str) -> Path:
        return ROOT / "scratch" / "check" / self.cfg.scene.replace(" ", "_") / label

    def observe(self, r: int, label: Optional[str] = None) -> tuple[Optional[dict], Optional[dict], str]:
        label = label or self.check_label(r)
        if not (self.check_dir(label) / "check.json").exists():
            extra = ["--base-url", self.cfg.base_url] if self.cfg.base_url else None
            rc = runner.run_check(self.cfg.scene, label, self.cfg.judge_model, self.out / f"round_{r}" / "check.log",
                                  grass=self.cfg.grass, extra=extra)
            if rc != 0:
                self.say(f"scene-check failed (exit {rc}); see round_{r}/check.log")
        check, verdict = runner.read_check(self.cfg.scene, label)
        return check, verdict, label

    @staticmethod
    def scores_of(check: Optional[dict], verdict: Optional[dict]) -> dict:
        s = {}
        if check:
            for k in ("dino_global", "dino_layout", "colour_de", "black", "flat", "oversize"):
                if k in check:
                    s[k] = check[k]
        if verdict:
            s["judge"] = verdict.get("score")
            s["acceptance"] = verdict.get("acceptance_pass")
            s["judge_usd"] = verdict.get("usd", 0.0)
        return s

    JUDGE_NOISE = 0.05     # Haiku re-reads of the same render differ by this much (band27 0.33 vs critic1 0.23)
    DINO_NOISE = 0.01

    @classmethod
    def better(cls, a: dict, b: dict) -> bool:
        """a better than b. The critic decides only when it moved by more than its own
        noise; otherwise DINO global, then colour error. agent1 round 1 taught this: judge
        +0.01 with DINO -0.12 is not an improvement."""
        ja, jb = a.get("judge"), b.get("judge")
        if ja is not None and jb is not None and abs(ja - jb) >= cls.JUDGE_NOISE:
            return ja > jb
        da, db = a.get("dino_global"), b.get("dino_global")
        if da is not None and db is not None and abs(da - db) >= cls.DINO_NOISE:
            return da > db
        ca, cb = a.get("colour_de"), b.get("colour_de")
        if ca is not None and cb is not None:
            return ca < cb
        return False

    def images_for(self, label: str) -> list[Image.Image]:
        d = self.check_dir(label)
        ims = [Image.open(self.photo).convert("RGB")]
        for name in ("photo.png", "sheet.png"):
            p = d / name
            if p.exists():
                ims.append(Image.open(p).convert("RGB"))
        return ims

    # ── the loop ──────────────────────────────────────────────────────────────
    def run(self) -> dict:
        cfg = self.cfg
        self.say(f"agent loop {cfg.label}: scene {cfg.scene!r}, run {self.run_dir.name}, rounds {cfg.rounds}, "
                 f"budget ${cfg.max_usd:.2f}, model {cfg.model}, judge {cfg.judge_model}, dry_run={cfg.dry_run}")
        self.export_graph()
        self.say(f"graph exported: {self.graph.counts()}")

        # round 0: where the scene stands
        r0 = RoundRecord(0, cfg.baseline or self.check_label(0))
        t0 = time.time()
        check, verdict, label0 = self.observe(0, cfg.baseline)
        r0.scores = self.scores_of(check, verdict)
        r0.usd = r0.scores.get("judge_usd", 0.0)
        r0.seconds = time.time() - t0
        self.usd += r0.usd
        self.rounds.append(r0)
        self.say(f"round 0 ({label0}): {r0.scores}")
        best = r0
        no_gain = 0
        knobs_text = knob_catalogue([k for k in self.knobs if cfg.allow_expensive or k.cost != "expensive"], self.graph)
        current_label = label0

        for r in range(1, cfg.rounds + 1):
            rec = RoundRecord(r, self.check_label(r))
            rdir = self.out / f"round_{r}"
            rdir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            if self.usd >= cfg.max_usd:
                rec.outcome = f"budget spent (${self.usd:.2f})"
                self.rounds.append(rec)
                self.say(rec.outcome)
                break
            # hypothesise
            history = [{"round": x.round, "scores": x.scores, "applied": x.applied, "outcome": x.outcome} for x in self.rounds]
            user = build_user(self.graph.summary(), knobs_text, check, verdict, history, r,
                              f"Spent ${self.usd:.2f} of ${cfg.max_usd:.2f}. Expensive reruns are "
                              + ("allowed." if cfg.allow_expensive else "NOT allowed this session (not in the vocabulary)."))
            (rdir / "prompt.txt").write_text(user)
            try:
                proposal, usage = propose(cfg.model, user, self.images_for(current_label), base_url=cfg.base_url,
                                         raw_path=self.out / f"round_{r}" / "proposal_raw.txt")
            except Exception as exc:
                rec.outcome = f"orchestrator failed: {type(exc).__name__}: {exc}"
                self.rounds.append(rec)
                self.say(rec.outcome)
                break
            rec.proposal = proposal
            rec.usd += usage.get("usd", 0.0)
            self.usd += usage.get("usd", 0.0)
            (rdir / "proposal.json").write_text(json.dumps({"proposal": proposal, "usage": usage}, indent=2))
            self.say(f"round {r} proposal (${usage.get('usd', 0):.2f}, {usage.get('seconds')} s): {proposal.get('diagnosis')}")
            for e in proposal.get("edits", []):
                self.say(f"    {e.get('node')}.{e.get('attr')} = {e.get('value')}  ({e.get('why')})")
            if proposal.get("stop"):
                rec.outcome = f"model stopped: {proposal.get('stop_reason')}"
                self.rounds.append(rec)
                self.say(rec.outcome)
                break
            # resolve edits
            edits = [Edit.from_dict(e) for e in proposal.get("edits", [])][: cfg.max_edits]
            allowed = [k for k in self.knobs if cfg.allow_expensive or k.cost != "expensive"]
            plan = resolve_edits(edits, allowed)
            rec.applied, rec.rejected, rec.rerun_from = plan.applied, plan.rejected, plan.rerun_from
            for x in plan.rejected:
                self.say(f"    rejected {x['node']}.{x['attr']}={x['value']}: {x['reason']}")
            if not plan.applied:
                rec.outcome = "no applicable edits"
                self.rounds.append(rec)
                self.say(rec.outcome)
                break
            # apply to the graph (logged) and to the pipeline inputs
            for a in plan.applied:
                node = self.graph.find(a["node"])
                if node is not None:
                    self.graph.edit(a["node"], a["attr"], a["value"], f"agent r{r}", a["why"], evidence="fitted")
            for stage, key, value in plan.config_sets:
                self.sets[(stage, key)] = value
            config_path = runner.write_derived_config([(s, k, v) for (s, k), v in self.sets.items()], rdir / "config.yaml")
            if plan.plan_patches:
                runner.patch_plan(self.run_dir, plan.plan_patches, f"agent r{r}")
                self.say(f"    plan patched: {plan.plan_patches}")
            (rdir / "edits.json").write_text(json.dumps({"applied": plan.applied, "rejected": plan.rejected,
                                                         "config_sets": [(s, k, v) for (s, k), v in self.sets.items()],
                                                         "plan_patches": plan.plan_patches,
                                                         "rerun_from": plan.rerun_from}, indent=2))
            if cfg.dry_run:
                rec.outcome = f"dry run: would rerun from {plan.rerun_from!r} with {len(plan.config_sets)} config sets"
                self.rounds.append(rec)
                self.say(rec.outcome)
                break
            # build
            self.say(f"    rerun from {plan.rerun_from!r} ...")
            try:
                rc = runner.run_pipeline(self.photo, plan.rerun_from, config_path, rdir / "pipeline.log")
            except RuntimeError as exc:
                rec.outcome = str(exc)
                self.rounds.append(rec)
                self.say(rec.outcome)
                break
            if rc != 0:
                rec.outcome = f"pipeline failed (exit {rc}); see round_{r}/pipeline.log"
                self.rounds.append(rec)
                self.say(rec.outcome)
                break
            # observe + critique
            self.export_graph()
            check, verdict, label = self.observe(r)
            rec.scores = self.scores_of(check, verdict)
            rec.usd += rec.scores.get("judge_usd", 0.0)
            self.usd += rec.scores.get("judge_usd", 0.0)
            current_label = label
            rec.seconds = time.time() - t0
            if self.better(rec.scores, best.scores):
                rec.outcome = f"improved on round {best.round}"
                best = rec
                no_gain = 0
            else:
                rec.outcome = f"no better than round {best.round}"
                no_gain += 1
            self.rounds.append(rec)
            self.say(f"round {r}: {rec.scores}; {rec.outcome}; {rec.seconds / 60:.0f} min")
            for a in plan.applied:
                if a["node"] in self.graph.nodes:
                    self.graph.annotate(a["node"], f"r{r} {a['attr']}={a['value']}: {rec.outcome}", f"agent r{r}")
                else:
                    self.say(f"    node {a['node']} is gone from the re-exported graph; outcome not annotated")
            if no_gain >= cfg.patience:
                self.say(f"no improvement for {no_gain} rounds; stopping")
                break
        return self.report(best)

    def report(self, best: RoundRecord) -> dict:
        rep = {"label": self.cfg.label, "scene": self.cfg.scene, "run": self.run_dir.name, "usd": round(self.usd, 3),
               "best_round": best.round, "best_label": best.label, "best_scores": best.scores,
               "rounds": [r.to_json() for r in self.rounds],
               "config_sets": [(s, k, v) for (s, k), v in self.sets.items()]}
        (self.out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
        lines = [f"# agent loop {self.cfg.label}: {self.cfg.scene} ({self.run_dir.name})", "",
                 f"spent ${self.usd:.2f}; best round {best.round} ({best.label}) {best.scores}", "",
                 "| round | judge | dino | colour dE | edits | outcome |", "|---|---|---|---|---|---|"]
        for r in self.rounds:
            s = r.scores
            lines.append(f"| {r.round} | {s.get('judge', '')} | {s.get('dino_global', '')}/{s.get('dino_layout', '')} | "
                         f"{s.get('colour_de', '')} | " + "; ".join(f"{a['node']}.{a['attr']}={a['value']}" for a in r.applied)
                         + f" | {r.outcome} |")
        if self.sets:
            lines += ["", "config sets in force (apply to server/config.yaml to keep them):"]
            lines += [f"- {s}: {k} = {v}" for (s, k), v in self.sets.items()]
        (self.out / "report.md").write_text("\n".join(lines) + "\n")
        self.say(f"report: {self.out / 'report.md'}")
        return rep
