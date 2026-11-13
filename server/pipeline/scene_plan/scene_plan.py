"""Scene Plan stage: one vision-model reading of the photograph, cached twice.

Input   ContextKey.INPUT (the photograph), ContextKey.INPUT_CAPTION
Output  ContextKey.SCENE_PLAN  {schema_version, model, photo_sha256, source, created, plan}

Cache layers: the stage's own context output (normal stage skip) and
<cache_dir>/<sha256(photo)>-v<schema>-<model>.json shared across runs and output
folders, so one photograph is read once, ever, unless `refresh: true`.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from pipeline.pipeline_stage import PipelineStage, PipelineStageConfiguration, SemanticKey
from pipeline.pipeline_context import ContextKey, PipelineContext
from pipeline.scene_plan.schema import PLAN_SCHEMA, SCHEMA_VERSION, SYSTEM_PROMPT, USER_PROMPT, EVIDENCE, CATEGORY_ENUM
from pipeline.scene_plan.client import PlanModel, SpendLedger, NoApiError, SpendCapError, ModelReply

_ROOT = Path(__file__).resolve().parents[3]
TOOL_NAME = "scene_plan"


class ScenePlanConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        model: str = "anthropic:claude-opus-5-5",
        cache_dir: str | None = None,
        refresh: bool = False,
        no_api: bool = False,
        max_usd_per_run: float = 2.0,
        timeout_s: float = 600.0,
        retries: int = 1,
        max_tokens: int = 8000,
        base_url: str | None = None,
        key_service: str = "anthropic",
        key_user: str = "3dcodeverse",
        required: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = str(model)
        self.cache_dir = Path(cache_dir) if cache_dir else _ROOT / "cache" / "scene_plan"
        self.refresh = bool(refresh)
        self.no_api = bool(no_api)
        self.max_usd_per_run = float(max_usd_per_run)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.max_tokens = int(max_tokens)
        self.base_url = base_url
        self.key_service = key_service
        self.key_user = key_user
        # False: a failed or forbidden call leaves the run without a plan (every
        # consumer then behaves as today). True: the stage fails the run.
        self.required = bool(required)


def photo_sha256(pil_image) -> str:
    im = pil_image.convert("RGB")
    h = hashlib.sha256()
    h.update(f"{im.width}x{im.height}:".encode())
    h.update(im.tobytes())
    return h.hexdigest()


def validate_plan(data: dict) -> tuple[dict, list[str]]:
    """Enforce provenance: a block without a legal evidence label is marked invented
    (text only); a list entry without one is dropped. Returns (plan, notes)."""
    notes: list[str] = []
    plan = json.loads(json.dumps(data))  # deep copy, plain JSON types only
    blocks = ["place", "camera", "sun", "atmosphere", "horizon", "water", "terrain"]
    for b in blocks:
        node = plan.get(b)
        if not isinstance(node, dict):
            plan[b] = {"evidence": "invented", "because": "missing from the model's reply"}
            notes.append(f"{b}: missing, marked invented")
            continue
        if node.get("evidence") not in EVIDENCE:
            notes.append(f"{b}: evidence {node.get('evidence')!r} not legal, marked invented")
            node["evidence"] = "invented"
        node.setdefault("because", "")
    for lst in ("landmarks", "species", "objects", "motion"):
        items = plan.get(lst)
        if not isinstance(items, list):
            plan[lst] = []
            continue
        kept = []
        for it in items:
            if isinstance(it, dict) and it.get("evidence") in EVIDENCE:
                if lst in ("landmarks", "objects") and it.get("category") not in CATEGORY_ENUM:
                    notes.append(f"{lst}: category {it.get('category')!r} not in the list ({str(it.get('class'))[:40]})")
                    it.pop("category", None)  # read conservatively from the class text instead
                kept.append(it)
            else:
                notes.append(f"{lst}: entry without evidence dropped ({str(it)[:60]})")
        plan[lst] = kept
    if not isinstance(plan.get("behind"), dict):
        plan["behind"] = {"text": ""}
    acc = plan.get("acceptance")
    plan["acceptance"] = [a for a in acc if isinstance(a, dict) and a.get("text")] if isinstance(acc, list) else []
    return plan, notes


class ScenePlanStage(PipelineStage):
    @classmethod
    def config_class(cls):
        return ScenePlanConfiguration

    def __init__(self, config: ScenePlanConfiguration) -> None:
        super().__init__(config)
        self.ledger = SpendLedger(config.cache_dir, config.max_usd_per_run)

    def _keys(self):
        return self.keys({SemanticKey.INPUT: ContextKey.INPUT, SemanticKey.OUTPUT: ContextKey.SCENE_PLAN})

    def _cache_file(self, sha: str) -> Path:
        model_slug = self.config.model.replace(":", "_").replace("/", "_")
        return self.config.cache_dir / f"{sha}-v{SCHEMA_VERSION}-{model_slug}.json"

    # ── run ───────────────────────────────────────────────────────────────────
    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: ScenePlanConfiguration = self.config
        input_key, output_key = self._keys()
        task = self.create_progress(3, "Scene Plan…")
        photo = context.input_image(input_key)
        if photo is None:
            self.log_warning("Scene Plan: no input photograph; no plan")
            self.finish_progress(task)
            return context
        pil = photo.image if hasattr(photo, "image") else photo
        caption = context.input_object(ContextKey.INPUT_CAPTION) or ""
        sha = photo_sha256(pil)
        self.advance_progress(task)

        cache_file = self._cache_file(sha)
        record: Optional[dict] = None
        if cache_file.is_file() and not cfg.refresh:
            try:
                record = json.loads(cache_file.read_text())
                record["source"] = "cache"
                self.log_info(f"Scene Plan: cached reading {cache_file.name} (no call)")
            except ValueError:
                self.log_warning(f"Scene Plan: unreadable cache {cache_file}; re-reading")
                record = None
        self.advance_progress(task)

        if record is None:
            record = self._read(pil, str(caption), sha, cache_file)
            if record is None:
                self.finish_progress(task)
                return context

        context.add_object(output_key, record)
        if self.output is not None:
            (self.output / "plan.json").write_text(json.dumps(record, indent=2))
        self._log_summary(record)
        self.advance_progress(task)
        self.finish_progress(task)
        return context

    def _read(self, pil, caption: str, sha: str, cache_file: Path) -> Optional[dict]:
        cfg: ScenePlanConfiguration = self.config
        model = PlanModel(cfg.model, timeout_s=cfg.timeout_s, retries=cfg.retries, base_url=cfg.base_url,
                          key_service=cfg.key_service, key_user=cfg.key_user, no_api=cfg.no_api)
        user = USER_PROMPT.replace("{caption}", caption.replace('"', "'"))
        try:
            # ~2.5k tokens of image + prompt, ~3k out at Opus prices: well under a dollar.
            self.ledger.check(0.8, "Scene Plan reading")
            self.log_info(f"Scene Plan: reading the photograph with {cfg.model} (cache miss {cache_file.name})")
            reply: ModelReply = model.read_photo(pil, SYSTEM_PROMPT, user, TOOL_NAME, PLAN_SCHEMA, cfg.max_tokens)
        except (NoApiError, SpendCapError) as exc:
            msg = f"Scene Plan: {exc}"
            if cfg.required:
                raise
            self.log_warning(msg + " -- continuing without a plan")
            return None
        except Exception as exc:
            if cfg.required:
                raise
            self.log_warning(f"Scene Plan: reading failed ({type(exc).__name__}: {exc}) -- continuing without a plan")
            return None
        plan, notes = validate_plan(reply.data)
        for n in notes:
            self.log_info(f"Scene Plan: {n}")
        record = {
            "schema_version": SCHEMA_VERSION, "model": f"{reply.provider}:{reply.model}", "photo_sha256": sha,
            "caption": caption, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "api",
            "usage": {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                      "usd": round(reply.usd, 4), "seconds": round(reply.seconds, 1)},
            "validation_notes": notes, "plan": plan,
        }
        self.ledger.record(self.name, reply, cache_file.name, "cache miss")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(record, indent=2))
        self.log_info(f"Scene Plan: read in {reply.seconds:.0f} s, {reply.input_tokens}+{reply.output_tokens} tokens, "
                      f"~${reply.usd:.2f}; cached as {cache_file.name}")
        return record

    def _log_summary(self, record: dict) -> None:
        p = record.get("plan", {})
        cam, sun, water, terrain = p.get("camera", {}), p.get("sun", {}), p.get("water", {}), p.get("terrain", {})
        self.log_info(
            f"Scene Plan: {p.get('place', {}).get('kind', '?')} ({p.get('place', {}).get('name') or 'unnamed'}); "
            f"camera {cam.get('height_m')} m {cam.get('evidence')}; sun az {sun.get('azimuth_deg')} el "
            f"{sun.get('elevation_deg')} {sun.get('warmth')} {sun.get('evidence')}; water {water.get('kind')} "
            f"{water.get('width_m')} m {water.get('evidence')}; terrain {terrain.get('kind')}; "
            f"{len(p.get('species', []))} species, {len(p.get('landmarks', []))} landmarks, "
            f"{len(p.get('objects', []))} object classes, {len(p.get('motion', []))} motions, "
            f"{len(p.get('acceptance', []))} acceptance items"
        )

    # ── cache / report ────────────────────────────────────────────────────────
    def has_expected_output(self, context: PipelineContext) -> bool:
        _, output_key = self._keys()
        rec = context.object(output_key)
        return isinstance(rec, dict) and rec.get("schema_version") == SCHEMA_VERSION and bool(rec.get("plan"))

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        _, output_key = self._keys()
        rec = context.object(output_key)
        if not isinstance(rec, dict):
            return None
        p = rec.get("plan", {})

        def claim(block: str, *fields: str) -> str:
            b = p.get(block, {})
            vals = ", ".join(f"{f} {b.get(f)}" for f in fields if b.get(f) is not None)
            return f"{vals} [{b.get('evidence')}] -- {b.get('because', '')}"

        stats = {
            "Model": f"{rec.get('model')} ({rec.get('source')})",
            "Place": claim("place", "kind", "name"),
            "Camera": claim("camera", "height_m", "pitch_deg", "on_structure"),
            "Sun": claim("sun", "azimuth_deg", "elevation_deg", "warmth", "sky"),
            "Atmosphere": claim("atmosphere", "haze_near_m", "haze_far_m"),
            "Water": claim("water", "kind", "width_m", "level_below_camera_m", "bank_left", "bank_right"),
            "Terrain": claim("terrain", "kind", "ground"),
            "Species": "; ".join(f"{s.get('name')} ({s.get('density')}, {s.get('evidence')})" for s in p.get("species", [])),
            "Landmarks": "; ".join(f"{l.get('name')} {l.get('distance_m')} m at {l.get('bearing_deg')} deg ({l.get('evidence')})"
                                   for l in p.get("landmarks", [])),
            "Motion": "; ".join(f"{m.get('target')}: {m.get('kind')} {m.get('amount')}" for m in p.get("motion", [])),
            "Acceptance": "; ".join(f"{a.get('id')} [{a.get('priority')}] {a.get('text')}" for a in p.get("acceptance", [])),
        }
        return ReportSection(stage_name=self.name, title="Scene Plan",
                             body="One vision-model reading of the photograph, cached by its hash. Every claim is "
                                  "labelled visible, inferred or invented; later stages use it only through the "
                                  "trust rules (visible bounds a measurement, inferred stands in when a measurement "
                                  "is missing or insane, invented is text).",
                             stats=stats)
