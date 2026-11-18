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
from pipeline.object_typing.categories import OBJECT_CATEGORIES

# Species classes the plan uses (schema) -> the pipeline's object categories.
_SPECIES_CATEGORY = {"flower": "flower", "tree": "tree", "shrub": "bush", "grass": None, "reed": None, "moss": None}

# Categories that name a region rather than an instance; never prompted from the plan.
_REGION_CATEGORIES = {"forest"}

# v1 plans carry free-text classes only. A conservative reading of them: a known
# category word, or one of these, and nothing else (token matching through
# normalize_category reads "stone arch bridge" as rock).
_CLASS_WORDS = {
    "barge": "boat", "houseboat": "boat", "péniche": "boat", "peniche": "boat", "ferry": "boat",
    "vessel": "boat", "yacht": "boat", "apartment": "building", "apartments": "building",
    "house": "building", "houses": "building", "block": "building", "roofs": "building",
    "church": "building", "viaduct": "bridge", "overpass": "bridge", "spire": "tower",
    "steeple": "tower", "people": "person", "passengers": "person", "tourists": "person",
}


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

    # ── what the scene contains (for the understanding stages) ────────────────
    @staticmethod
    def category_of(item: dict, kind: str) -> Optional[str]:
        """The pipeline object category of a landmark/object/species entry, or None
        for scenery. v2 plans say it; v1 plans are read conservatively."""
        if kind == "species":
            return _SPECIES_CATEGORY.get(str(item.get("class", "")).lower())
        cat = item.get("category")
        if isinstance(cat, str):
            return cat if cat in OBJECT_CATEGORIES else None
        text = str(item.get("class", "")).lower()
        if text in OBJECT_CATEGORIES:
            return text
        for tok in text.replace("/", " ").replace("(", " ").replace(")", " ").split():
            if tok in OBJECT_CATEGORIES:
                return tok
            if tok in _CLASS_WORDS:
                return _CLASS_WORDS[tok]
        return None

    def entries(self, min_evidence: str = "inferred") -> list[tuple[str, dict, str]]:
        """(kind, entry, category) for every landmark, object and species that is an
        object the pipeline can build, at or above the evidence level."""
        out = []
        for kind in ("landmarks", "objects", "species"):
            for it in self.items(kind, min_evidence):
                cat = self.category_of(it, kind)
                if cat:
                    out.append((kind, it, cat))
        return out

    def vocabulary(self, min_evidence: str = "inferred") -> dict[str, str]:
        """Detector phrase -> object category, for Object Labeling. Each entry
        contributes its category word, its `phrase`, and a proper name when it has
        one (the name finds the Eiffel Tower where "tower" alone does not)."""
        vocab: dict[str, str] = {}
        for kind, it, cat in self.entries(min_evidence):
            if cat in _REGION_CATEGORIES:
                # A band, not a thing: a "conifer tree line" box outscored the tree
                # boxes inside it and relabelled 115 of Rainier's tree blobs forest.
                continue
            vocab[cat.replace("_", " ")] = cat
            phrase = str(it.get("phrase") or "").strip().lower()
            if phrase and len(phrase.split()) <= 5:
                vocab[phrase] = cat
            name = str(it.get("name") or "").strip()
            if kind == "landmarks" and name and name[:1].isupper() and len(name.split()) <= 4:
                vocab[name.lower()] = cat
        return vocab

    # Species sizes by the class they stand for: a species 4 m or taller is a tree,
    # one under 1.5 m a flower (the plan's species entries carry no category).
    _SIZE_CLASSES = {"tree": (4.0, None), "flower": (0.05, 1.5), "bush": (0.5, 4.0)}

    def size_prior(self, cls: str, min_evidence: str = "inferred") -> Optional[tuple[float, str, str]]:
        """(size_m, evidence, name): the standing height the plan gives this class,
        from its species sizes and from objects/landmarks of the category (a forest
        entry counts for tree). The strongest evidence level wins, then the median."""
        cands: list[tuple[float, str, str]] = []
        rng = self._SIZE_CLASSES.get(cls)
        if rng:
            for it in self.items("species", min_evidence):
                s = it.get("size_m")
                if isinstance(s, (int, float)) and not isinstance(s, bool) and s > 0 \
                        and s >= rng[0] and (rng[1] is None or s < rng[1]):
                    cands.append((float(s), str(it.get("evidence")), str(it.get("name") or "")))
        for kind in ("objects", "landmarks"):
            for it in self.items(kind, min_evidence):
                cat = self.category_of(it, kind)
                if cat != cls and not (cls == "tree" and cat == "forest"):
                    continue
                s = it.get("size_m") if kind == "objects" else it.get("height_m")
                if isinstance(s, (int, float)) and not isinstance(s, bool) and s > 0:
                    cands.append((float(s), str(it.get("evidence")), str(it.get("name") or cat)))
        if not cands:
            return None
        rank = {"visible": 2, "inferred": 1}
        best = max(rank.get(e, 0) for _, e, _ in cands)
        top = sorted([c for c in cands if rank.get(c[1], 0) == best], key=lambda c: c[0])
        return top[(len(top) - 1) // 2]

    def categories(self, min_evidence: str = "visible") -> set[str]:
        """Object categories the plan says are in the photograph."""
        return {cat for _, _, cat in self.entries(min_evidence)}

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
