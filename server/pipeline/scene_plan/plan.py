"""ScenePlan: read access to the plan plus the one helper that applies the trust rules.

    plan = ScenePlan.from_context(context)
    r = plan.prior("camera.height_m", measured=1.0, measured_source="config", tolerance=0.25)
    r.value, r.source, r.note          # what to use, where it came from, one log line

Trust rules (docs/hybrid-scene-plan.md section 4):
  visible   may move a measurement inside its tolerance (clamp), may stand in for a
            missing or insane measurement; may not place anything by coordinate.
  inferred  stands in only when there is no measurement or it failed sanity; a
            disagreement with a sane measurement is logged, never applied.
  invented  text only; never a number.
Every call is recorded in plan.overrides so a consumer stage can publish them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pipeline.pipeline_context import ContextKey, PipelineContext


@dataclass
class PriorResult:
    value: Any
    source: str          # "measured" | "plan" | "clamped" | "config"
    evidence: Optional[str]
    note: str
    plan_value: Any = None
    measured: Any = None

    @property
    def changed(self) -> bool:
        return self.source in ("plan", "clamped")


class ScenePlan:
    def __init__(self, data: dict):
        self.data = data or {}
        self.overrides: list[dict] = []

    # ── access ────────────────────────────────────────────────────────────────
    @staticmethod
    def from_context(context: PipelineContext) -> Optional["ScenePlan"]:
        try:
            data = context.input_object(ContextKey.SCENE_PLAN)
        except Exception:
            data = None
        if not isinstance(data, dict) or not data.get("plan"):
            return None
        return ScenePlan(data["plan"])

    @property
    def empty(self) -> bool:
        return not self.data

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            else:
                return default
        return default if node is None else node

    def evidence(self, path: str) -> Optional[str]:
        """Evidence label of the claim block that holds `path` (the block itself or its parent)."""
        node = self.get(path)
        if isinstance(node, dict) and "evidence" in node:
            return node["evidence"]
        parent = path.rsplit(".", 1)[0] if "." in path else path
        block = self.get(parent)
        if isinstance(block, dict):
            return block.get("evidence")
        return None

    def because(self, path: str) -> str:
        parent = path.rsplit(".", 1)[0] if "." in path else path
        block = self.get(parent)
        return str(block.get("because", "")) if isinstance(block, dict) else ""

    def items(self, path: str, min_evidence: str = "inferred") -> list[dict]:
        """List entries (landmarks, species, objects, motion) at or above an evidence level."""
        rank = {"visible": 2, "inferred": 1, "invented": 0}
        need = rank.get(min_evidence, 1)
        out = []
        for it in self.get(path, []) or []:
            if isinstance(it, dict) and rank.get(it.get("evidence"), -1) >= need:
                out.append(it)
        return out

    # ── the rule ──────────────────────────────────────────────────────────────
    def prior(
        self,
        field_path: str,
        measured: Optional[float],
        tolerance: float = 0.25,
        sanity: Optional[Callable[[float], bool]] = None,
        measured_source: str = "measured",
        relative: bool = True,
        label: Optional[str] = None,
    ) -> PriorResult:
        """Resolve one scalar between the pipeline's measurement and the plan's claim.

        measured: the pipeline's value, or None when it has none (a config default
        counts as none when measured_source == "config").
        tolerance: relative (fraction of the plan value) unless relative=False (same units).
        sanity: measured -> bool; a False result lets an inferred claim stand in.
        """
        label = label or field_path
        plan_value = self.get(field_path)
        evidence = self.evidence(field_path)
        has_measure = measured is not None and measured_source != "config"
        rec = {"field": label, "measured": measured, "measured_source": measured_source,
               "plan": plan_value, "evidence": evidence, "because": self.because(field_path)}

        def done(value, source, note):
            rec.update({"used": value, "source": source, "note": note})
            self.overrides.append(rec)
            return PriorResult(value, source, evidence, note, plan_value=plan_value, measured=measured)

        if not isinstance(plan_value, (int, float)) or isinstance(plan_value, bool):
            return done(measured, "measured" if has_measure else measured_source, f"{label}: no plan value")
        plan_value = float(plan_value)
        if evidence == "invented" or evidence is None:
            return done(measured, "measured" if has_measure else measured_source,
                        f"{label}: plan value {plan_value:g} is {evidence or 'unlabelled'}; ignored")
        if not has_measure:
            return done(plan_value, "plan",
                        f"{label}: no measurement ({measured_source} {measured!r}); plan {plan_value:g} ({evidence}) used")
        m = float(measured)
        sane = sanity(m) if sanity is not None else True
        if not sane:
            return done(plan_value, "plan",
                        f"{label}: measured {m:g} failed sanity; plan {plan_value:g} ({evidence}) used")
        tol = abs(plan_value) * tolerance if relative else tolerance
        lo, hi = plan_value - tol, plan_value + tol
        if lo <= m <= hi:
            return done(m, "measured", f"{label}: measured {m:g} within {tolerance:g}{'x' if relative else ''} of plan {plan_value:g}")
        if evidence == "visible":
            clamped = min(max(m, lo), hi)
            return done(clamped, "clamped",
                        f"{label}: measured {m:g} moved to {clamped:g} (plan {plan_value:g} visible, tolerance {tolerance:g})")
        return done(m, "measured",
                    f"{label}: measured {m:g} kept; plan {plan_value:g} is inferred and disagrees")

    def publish(self, context: PipelineContext, stage_slug: str) -> None:
        """Record this stage's prior decisions in the context (one JSON per consumer stage)."""
        if self.overrides:
            context.add_object(f"plan_overrides_{stage_slug}", list(self.overrides))
