"""Inventory Review: the first agent gate on the understanding pass.

Object Labeling leaves a list of detections (metadata_<i>: class, phrase, confidence,
photo_box) that every later stage trusts: Clustering groups them, Asset Generation
builds a mesh per variant, Scene Generation places each one. Its mistakes are the
scene's mistakes: a strip across the whole frame labelled "boat", a bridge labelled
"rock" because the phrase was "stone arch bridge", a sliver of water as an object.
Nothing between the detector and the scene ever looks at the photograph again and asks
whether the inventory is right.

This stage does. One vision model sees the photograph with every reviewable detection
boxed and numbered, the clean photograph, the detector's class and phrase for each box,
and the scene plan's inventory (docs/agentic-scene-graph-plan.md §13a, the first-pass
gates). Per box it decides keep / drop / relabel with a reason; plan items no box covers
are recorded as `missing` with a photo_box so the fix loop can place them later. A drop
demotes the detection to indeterminate exactly as the size veto does (it still occludes
for Foreground Inpainting, it is never placed); a relabel sets class and name. The
reviewer is told to be conservative: a wrong drop is a hole, a wrong keep a stray.

Reviewable: detections with a photo_box (inside the captured cone), an object class
(environment classes are scenery and never placed anyway) and both sides >= min_box_px.
Everything else passes through untouched.

Cached twice, like Scene Plan: in the run's context and across runs in
cache/inventory_review/<sha256(photo, boxes, model)>.json, so a rerun from any earlier
stage that reproduces the same detections costs nothing.

Input   ContextKey.INPUT, OBJECT_COUNT, metadata_<i>, SCENE_PLAN (optional)
Output  metadata_<i> (dropped or relabelled ones), inventory_review {decisions, missing, ...}
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

from pipeline.pipeline_context import ContextKey, PipelineContext
from pipeline.pipeline_stage import PipelineStage, PipelineStageConfiguration
from pipeline.object_typing.categories import ENVIRONMENT_CATEGORIES, OBJECT_CATEGORIES, normalize_category
from pipeline.scene_plan.client import ModelReply, NoApiError, PlanModel, SpendCapError, SpendLedger
from pipeline.scene_plan.plan import ScenePlan
from pipeline.scene_plan.scene_plan import photo_sha256

_ROOT = Path(__file__).resolve().parents[3]
MARKER = "inventory_review"
TOOL_NAME = "inventory_review"
SCHEMA_VERSION = 1
CATEGORIES = sorted(OBJECT_CATEGORIES)

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["decisions", "missing", "summary"],
    "properties": {
        "decisions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["idx", "decision", "class", "name", "why"],
            "properties": {
                "idx": {"type": "integer", "description": "the box number"},
                "decision": {"type": "string", "enum": ["keep", "drop", "relabel"]},
                "class": {"type": "string", "description": "for relabel: the right category from the list; else the detector's class"},
                "name": {"type": "string", "description": "what the thing is, in a few words (a tour boat, the Pont d'Iéna, a plane tree)"},
                "why": {"type": "string", "description": "one sentence; for drop and relabel, what in the photograph shows it"},
            }}},
        "missing": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "category", "count", "photo_box", "why"],
            "properties": {
                "name": {"type": "string"},
                "category": {"type": "string", "description": "a category from the list"},
                "count": {"type": "integer"},
                "photo_box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                              "description": "[x, y, w, h] as fractions of the photograph, the region the missing thing occupies"},
                "why": {"type": "string"},
            }}},
        "summary": {"type": "string", "description": "two sentences on the inventory's quality and what you changed"},
    },
}

SYSTEM = """You review the object inventory a detector found in a photograph before a 3D scene is built from it.
Image 1 is the photograph with the reviewable detections drawn as numbered boxes; image 2 is the clean photograph.
Every numbered box will become a standing 3D object of its class at that place, so the question per box is: is this one
real object (or one group of the same kind) that should stand in the scene, with the right class?

Decide per box:
- keep: a real object of that class (a row of trees in one box is fine; the box may cover several of the same kind).
- drop: not something to stand in the scene: a strip or sliver across the frame, a reflection or shadow, scenery
  (water, sky, ground, a wall, a distant cityscape) labelled as an object, a part of a larger thing that has its own
  box, or nothing recognisable in the box at all.
- relabel: the right thing with the wrong class or name (a bridge called "rock", a houseboat called "building").
  Give the class from the category list; give the name in a few words.

Be conservative. A wrong drop is a hole in the scene; a wrong keep is one stray object. When a box is ambiguous, keep
it. Do not relabel over a correct class to add detail; use `name` for detail.

Then list what the plan or the photograph shows that no box covers: a bridge, a boat, a building row, a monument.
Each `missing` item gets the category it would be built as and its region in the photograph as fractions [x, y, w, h].
Only things a 3D scene would stand as objects; not sky, water, ground, haze.

Categories: """ + ", ".join(CATEGORIES) + "."


class InventoryReviewConfiguration(PipelineStageConfiguration):
    def __init__(self, model: str = "anthropic:claude-sonnet-5-5", cache_dir: str | None = None,
                 refresh: bool = False, no_api: bool = False, max_usd_per_run: float = 1.0, timeout_s: float = 600.0,
                 retries: int = 1, max_tokens: int = 8000, base_url: str | None = None, key_service: str = "anthropic",
                 key_user: str = "3dcodeverse", min_box_px: int = 10, max_boxes: int = 80, apply: bool = True,
                 required: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.model = str(model)
        self.cache_dir = Path(cache_dir) if cache_dir else _ROOT / "cache" / "inventory_review"
        self.refresh = bool(refresh)
        self.no_api = bool(no_api)
        self.max_usd_per_run = float(max_usd_per_run)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.max_tokens = int(max_tokens)
        self.base_url = base_url
        self.key_service, self.key_user = key_service, key_user
        self.min_box_px = int(min_box_px)
        self.max_boxes = int(max_boxes)
        self.apply = bool(apply)
        self.required = bool(required)


def reviewable(meta: dict, min_px: int) -> bool:
    cls = meta.get("class")
    box = meta.get("photo_box")
    if not cls or cls == "indeterminate" or not box or len(box) != 4:
        return False
    if normalize_category(cls) in ENVIRONMENT_CATEGORIES:
        return False
    return float(box[2]) >= min_px and float(box[3]) >= min_px


def draw_boxes(photo: Image.Image, rows: list[tuple[int, dict]]) -> Image.Image:
    """The photograph with each detection's box and number; the number sits just above the
    box (inside the frame) so small boxes stay visible."""
    im = photo.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    lw = max(2, round(min(W, H) / 400))
    try:
        font = ImageFont.load_default(size=max(14, round(min(W, H) / 45)))
    except Exception:
        font = ImageFont.load_default()
    palette = [(255, 60, 60), (60, 200, 255), (255, 200, 0), (120, 255, 120), (255, 120, 255), (255, 150, 60)]
    for n, (idx, m) in enumerate(rows):
        x, y, w, h = (float(v) for v in m["photo_box"])
        col = palette[n % len(palette)]
        d.rectangle([x, y, x + w, y + h], outline=col, width=lw)
        label = str(idx)
        tw, th = d.textbbox((0, 0), label, font=font)[2:]
        tx = min(max(0.0, x), W - tw - 4)
        ty = y - th - 4 if y - th - 4 >= 0 else y + 2
        d.rectangle([tx, ty, tx + tw + 4, ty + th + 2], fill=col)
        d.text((tx + 2, ty), label, fill=(0, 0, 0), font=font)
    return im


def detections_text(rows: list[tuple[int, dict]], W: int, H: int) -> str:
    lines = []
    for idx, m in rows:
        x, y, w, h = (float(v) for v in m["photo_box"])
        name = m.get("name") or m.get("class")
        lines.append(f"{idx:4d}  class={m.get('class'):<12s} phrase={str(name)!r:<34s} conf={float(m.get('confidence') or 0):.2f}  "
                     f"box=[{x / W:.3f}, {y / H:.3f}, {w / W:.3f}, {h / H:.3f}] ({w:.0f}x{h:.0f} px)")
    return "\n".join(lines)


def plan_text(plan: Optional[ScenePlan]) -> str:
    if plan is None or plan.empty:
        return "(no scene plan)"
    out = []
    for lm in plan.items("landmarks"):
        out.append(f"landmark: {lm.get('name')} ({lm.get('class')}, {lm.get('category')}) bearing {lm.get('bearing_deg')} deg, "
                   f"{lm.get('distance_m')} m, {lm.get('height_m')} m tall, {lm.get('side')} [{lm.get('evidence')}]")
    for ob in plan.items("objects"):
        out.append(f"object: {ob.get('count')} x {ob.get('class')} ({ob.get('category')}) {ob.get('where')} [{ob.get('evidence')}]")
    for sp in plan.items("species"):
        out.append(f"species: {sp.get('name') or sp.get('class')} ({sp.get('class')}) [{sp.get('evidence')}]")
    return "\n".join(out) or "(plan lists no objects)"


def review_key(photo_sha: str, rows: list[tuple[int, dict]], model: str) -> str:
    h = hashlib.sha256()
    h.update(f"{photo_sha}:v{SCHEMA_VERSION}:{model}:".encode())
    for idx, m in rows:
        h.update(json.dumps([idx, m.get("class"), m.get("name"), [round(float(v), 1) for v in m["photo_box"]]]).encode())
    return h.hexdigest()


class InventoryReviewStage(PipelineStage):
    @classmethod
    def config_class(cls):
        return InventoryReviewConfiguration

    def __init__(self, config: InventoryReviewConfiguration) -> None:
        super().__init__(config)
        self.ledger = SpendLedger(config.cache_dir, config.max_usd_per_run)

    def has_expected_output(self, context: PipelineContext) -> bool:
        rec = context.object(MARKER)
        return isinstance(rec, dict) and rec.get("schema_version") == SCHEMA_VERSION

    # ── run ───────────────────────────────────────────────────────────────────
    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: InventoryReviewConfiguration = self.config
        task = self.create_progress(3, "Inventory Review…")
        photo = context.input_image(ContextKey.INPUT)
        count = int(context.input_object(ContextKey.OBJECT_COUNT) or 0)
        if photo is None or count == 0:
            self.log_info("Inventory Review: no photograph or no detections; nothing to review")
            context.add_object(MARKER, {"schema_version": SCHEMA_VERSION, "reviewed": 0, "decisions": [], "missing": []})
            self.finish_progress(task)
            return context
        pil = photo.image if hasattr(photo, "image") else photo
        W, H = pil.size
        rows = [(i, m) for i in range(count)
                if isinstance(m := context.object(f"metadata_{i}"), dict) and reviewable(m, cfg.min_box_px)]
        if len(rows) > cfg.max_boxes:
            rows.sort(key=lambda r: -float(r[1]["photo_box"][2]) * float(r[1]["photo_box"][3]))
            rows = sorted(rows[:cfg.max_boxes])
        self.advance_progress(task)
        if not rows:
            self.log_info("Inventory Review: no reviewable detections (none with a photo box and an object class)")
            context.add_object(MARKER, {"schema_version": SCHEMA_VERSION, "reviewed": 0, "decisions": [], "missing": []})
            self.finish_progress(task)
            return context

        plan = ScenePlan.from_context(context)
        sha = photo_sha256(pil)
        key = review_key(sha, rows, cfg.model)
        cache_file = cfg.cache_dir / f"{key}.json"
        record: Optional[dict] = None
        if cache_file.is_file() and not cfg.refresh:
            try:
                record = json.loads(cache_file.read_text())
                record["source"] = "cache"
                self.log_info(f"Inventory Review: cached review {cache_file.name} (no call)")
            except ValueError:
                record = None
        annotated = draw_boxes(pil, rows)
        if self.output is not None:
            annotated.save(self.output / "boxes.png")
        if record is None:
            record = self._review(pil, annotated, rows, plan, W, H, cache_file)
            if record is None:
                context.add_object(MARKER, {"schema_version": SCHEMA_VERSION, "reviewed": 0, "decisions": [], "missing": [],
                                            "note": "no review (call refused or failed)"})
                self.finish_progress(task)
                return context
        self.advance_progress(task)

        applied = self._apply(context, rows, record) if cfg.apply else []
        record = {**record, "applied": applied, "reviewed": len(rows)}
        context.add_object(MARKER, record)
        if self.output is not None:
            (self.output / "review.json").write_text(json.dumps(record, indent=2))
        kinds = {}
        for d in record.get("decisions", []):
            kinds[d.get("decision")] = kinds.get(d.get("decision"), 0) + 1
        self.log_info(f"Inventory Review: {len(rows)} boxes reviewed by {record.get('model')}: "
                      + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())) + f"; {len(record.get('missing', []))} missing; "
                      + (f"{len(applied)} metadata changed" if cfg.apply else "not applied (apply: false)"))
        for a in applied:
            self.log_info(f"  {a}")
        for m in record.get("missing", []):
            self.log_info(f"  missing: {m.get('count')} x {m.get('name')} ({m.get('category')}) at {m.get('photo_box')}: {m.get('why')}")
        self.advance_progress(task)
        self.finish_progress(task)
        return context

    def _review(self, pil, annotated, rows, plan, W, H, cache_file: Path) -> Optional[dict]:
        cfg: InventoryReviewConfiguration = self.config
        model = PlanModel(cfg.model, timeout_s=cfg.timeout_s, retries=cfg.retries, base_url=cfg.base_url,
                          key_service=cfg.key_service, key_user=cfg.key_user, no_api=cfg.no_api)
        user = (f"The photograph is {W}x{H} px. Reviewable detections ({len(rows)}), box as fractions [x, y, w, h]:\n"
                + detections_text(rows, W, H)
                + "\n\nThe scene plan's inventory (a prior reading of this photograph; evidence in brackets):\n"
                + plan_text(plan)
                + "\n\nGive a decision for every numbered box, then the missing items.")
        try:
            self.ledger.check(0.3, "Inventory Review")
            self.log_info(f"Inventory Review: {len(rows)} boxes to {cfg.model} (cache miss {cache_file.name[:12]})")
            reply: ModelReply = model.read_photo(annotated, SYSTEM, user, TOOL_NAME, REVIEW_SCHEMA, cfg.max_tokens,
                                                 extra_images=[pil], purpose="Record the inventory review.")
        except (NoApiError, SpendCapError) as exc:
            if cfg.required:
                raise
            self.log_warning(f"Inventory Review: {exc} -- inventory kept as detected")
            return None
        except Exception as exc:
            if cfg.required:
                raise
            self.log_warning(f"Inventory Review: call failed ({type(exc).__name__}: {exc}) -- inventory kept as detected")
            return None
        data = reply.data if isinstance(reply.data, dict) else {}
        valid = {i for i, _ in rows}
        decisions = [d for d in data.get("decisions", []) if isinstance(d, dict) and d.get("idx") in valid
                     and d.get("decision") in ("keep", "drop", "relabel")]
        missing = [m for m in data.get("missing", []) if isinstance(m, dict) and isinstance(m.get("photo_box"), list)
                   and len(m["photo_box"]) == 4]
        record = {
            "schema_version": SCHEMA_VERSION, "model": f"{reply.provider}:{reply.model}", "source": "api",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "usage": {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                      "usd": round(reply.usd, 4), "seconds": round(reply.seconds, 1)},
            "decisions": decisions, "missing": missing, "summary": str(data.get("summary", "")),
        }
        self.ledger.record(self.name, reply, cache_file.name, "cache miss")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(record, indent=2))
        self.log_info(f"Inventory Review: answered in {reply.seconds:.0f} s, {reply.input_tokens}+{reply.output_tokens} tokens, "
                      f"~${reply.usd:.2f}; {record['summary']}")
        if self.output is not None:
            (self.output / "prompt.txt").write_text(SYSTEM + "\n\n---\n\n" + user)
        return record

    def _apply(self, context: PipelineContext, rows, record: dict) -> list[str]:
        by_idx = {i: m for i, m in rows}
        applied = []
        for d in record.get("decisions", []):
            idx = d.get("idx")
            m = by_idx.get(idx)
            if m is None:
                continue
            why = str(d.get("why", ""))[:200]
            if d.get("decision") == "drop":
                context.add_object(f"metadata_{idx}", {
                    **m, "class": "indeterminate", "confidence": 0.0, "low_confidence": True, "label_source": "review",
                    "vetoed_class": m.get("class"), "veto_reason": f"review: {why}",
                    "review": {"decision": "drop", "why": why, "name": d.get("name")},
                })
                applied.append(f"drop crop_{idx} ({m.get('class')} {str(m.get('name'))!r}): {why}")
            elif d.get("decision") == "relabel":
                raw = str(d.get("class") or "").strip().lower()
                cls = normalize_category(raw) if raw else None
                if not cls or cls not in OBJECT_CATEGORIES or cls == m.get("class"):
                    if cls and cls in ENVIRONMENT_CATEGORIES:
                        # the reviewer says it is scenery: that is a drop
                        context.add_object(f"metadata_{idx}", {
                            **m, "class": "indeterminate", "confidence": 0.0, "low_confidence": True, "label_source": "review",
                            "vetoed_class": m.get("class"), "veto_reason": f"review: scenery ({raw}); {why}",
                            "review": {"decision": "drop", "why": why, "name": d.get("name")},
                        })
                        applied.append(f"drop crop_{idx} ({m.get('class')} -> scenery {raw!r}): {why}")
                    continue
                context.add_object(f"metadata_{idx}", {
                    **m, "class": cls, "name": str(d.get("name") or cls), "label_source": "review",
                    "review": {"decision": "relabel", "from": m.get("class"), "why": why},
                })
                applied.append(f"relabel crop_{idx} {m.get('class')} -> {cls} {str(d.get('name'))!r}: {why}")
        return applied
