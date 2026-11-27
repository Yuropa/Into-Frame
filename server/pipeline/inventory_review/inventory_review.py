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
that is a part of a kept thing says which (`part_of`), and the kept detection is cut again
over itself and its parts (see _grow). A drop
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
SCHEMA_VERSION = 3   # 2: the `terrain` decision; 3: `part_of` on a drop, the kept box grows over its parts
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
                "decision": {"type": "string", "enum": ["keep", "drop", "terrain", "relabel"]},
                "class": {"type": "string", "description": "for relabel: the right category from the list; else the detector's class"},
                "name": {"type": "string", "description": "what the thing is, in a few words (a tour boat, the Pont d'Iéna, a plane tree)"},
                "why": {"type": "string", "description": "one sentence; for drop and relabel, what in the photograph shows it"},
                "part_of": {"type": "integer", "description": "only for a drop that is a part of a larger thing whose own box "
                                                               "you keep: that kept box's number"},
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
  box, or nothing recognisable in the box at all. When it is a part of a thing whose box you keep, give that box's
  number as `part_of`: the kept box is then grown to cover the whole thing, so when a thing is covered only by
  pieces (a boat as a stern box, a side box and a hull box), keep the box that covers most of it and drop the rest
  as parts of it.
- terrain: a landform, not an object: a cliff, bluff, headland, hillside, bank, dune, rock shelf or outcrop joined to
  the land around it, or any rock mass the edge of the photograph cuts through. It stays in the ground, which is built
  with its relief and carries on past the frame; as an object it would stand as a flat cut-out with a straight cut edge
  where the frame ended. A freestanding boulder or sea stack that is whole in the frame is an object: keep it.
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
                 required: bool = False, recover_missing: bool = True, recover_classes: list[str] | None = None,
                 recover_min_px: int = 12, photo_detect_sr_factor: float = 2.0,
                 erase_classes: list[str] | None = None, grow_over_parts: bool = True, grow_reach: float = 0.5,
                 grow_max_area: float = 6.0, **kwargs):
        super().__init__(**kwargs)
        # A kept box is grown over the boxes dropped as parts of it (see _grow): a part
        # counts when its centre lies within `grow_reach` of the kept box's sides around
        # it, and the grown box may be `grow_max_area` times the kept one at most.
        self.grow_over_parts = bool(grow_over_parts)
        self.grow_reach = float(grow_reach)
        self.grow_max_area = float(grow_max_area)
        # Classes the scene does without: taken out of the ground's picture and not
        # placed. People, until there is a way to animate one: a figure frozen
        # mid-step reads worse than an empty beach.
        self.erase_classes = frozenset(str(c).lower() for c in (erase_classes if erase_classes is not None else ["person"]))
        # What the reviewer names as missing, and gives a box for, is segmented in
        # that box and joins the inventory (see _recover_missing).
        self.recover_missing = bool(recover_missing)
        self.recover_classes = frozenset(str(c).lower() for c in (recover_classes if recover_classes is not None else [
            "person", "animal", "bird", "boat", "ship", "rock", "car", "truck", "bus", "bicycle", "motorcycle",
            "bench", "statue", "dock"]))
        self.recover_min_px = int(recover_min_px)
        self.photo_detect_sr_factor = float(photo_detect_sr_factor)
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
        # A version 1 record (no `terrain` decision) is a finished review: a run is not
        # rebuilt from here because the reviewer learnt a new answer.
        return isinstance(rec, dict) and rec.get("schema_version") in (1, SCHEMA_VERSION)

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
        grown: list[dict] = []
        if cfg.apply and cfg.grow_over_parts:
            try:
                grown = self._grow(context, rows, record)
            except Exception as exc:  # the inventory stands as reviewed
                self.log_warning(f"Inventory Review: kept boxes not grown over their parts ({type(exc).__name__}: {exc})")
        recovered: list[dict] = []
        if cfg.apply and cfg.recover_missing and record.get("missing"):
            try:
                recovered = self._recover_missing(context, record, count, W, H)
            except Exception as exc:  # the inventory stands as reviewed
                self.log_warning(f"Inventory Review: missing items not recovered ({type(exc).__name__}: {exc})")
        erased = self._erase(context) if cfg.apply and cfg.erase_classes else []
        record = {**record, "applied": applied, "reviewed": len(rows), "grown": grown, "recovered": recovered,
                  "erased": erased}
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
                     and d.get("decision") in ("keep", "drop", "terrain", "relabel")]
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

    def _grow(self, context: PipelineContext, rows, record: dict) -> list[dict]:
        """A kept detection cut again over itself and the boxes dropped as parts of it.

        The detector boxes what it recognises, which need not be the whole thing:
        Paris's tour boat, seen from the bridge, is a box over its stern and upper deck
        (237 x 202 px of the photograph), and its port side and hull are six more
        boxes. The reviewer drops those as fragments of the boat and keeps the one,
        and everything after took the stern for the boat: it was modelled from that
        cut-out, fitted to that box (11 degrees wide where the boat is 15) and only
        that box was taken off the ground's picture, the port side left painted on
        the water beside the model. The reviewer says which box a dropped part
        belongs to; the kept detection is segmented again inside the box around all
        of them (box-prompted SAM2 on the detection photo, as a missing item is) and
        written over its own crop_<i> / metadata_<i>. A part far from the kept box, a
        grown box many times the kept one, or a mask that is no larger than the box it
        started from changes nothing."""
        import numpy as np
        from pipeline.segmentation.image_segmentation import ImageSeg
        from pipeline.segmentation.photo_detection import detection_photo, store_photo_masks
        from util.captured_cone import captured_cone_from_context
        from util.device_utils import DeviceStrategy, preferred_device

        cfg: InventoryReviewConfiguration = self.config
        by_idx = {i: m for i, m in rows}
        decisions = {d.get("idx"): d for d in record.get("decisions", [])}
        wanted = []
        for k_idx, d in decisions.items():
            if d.get("decision") not in ("keep", "relabel") or k_idx not in by_idx:
                continue
            kx, ky, kw, kh = (float(v) for v in by_idx[k_idx]["photo_box"])
            x0, y0, x1, y1 = kx, ky, kx + kw, ky + kh
            parts = []
            for p_idx, p in decisions.items():
                if p.get("decision") != "drop" or p.get("part_of") != k_idx or p_idx == k_idx or p_idx not in by_idx:
                    continue
                px, py, pw, ph = (float(v) for v in by_idx[p_idx]["photo_box"])
                cx, cy = px + pw / 2.0, py + ph / 2.0
                if not (kx - cfg.grow_reach * kw <= cx <= kx + (1.0 + cfg.grow_reach) * kw
                        and ky - cfg.grow_reach * kh <= cy <= ky + (1.0 + cfg.grow_reach) * kh):
                    self.log_info(f"  crop_{p_idx} is named a part of crop_{k_idx} and lies away from it; not joined")
                    continue
                parts.append(p_idx)
                x0, y0, x1, y1 = min(x0, px), min(y0, py), max(x1, px + pw), max(y1, py + ph)
            if not parts:
                continue
            ratio = (x1 - x0) * (y1 - y0) / max(kw * kh, 1.0)
            if ratio < 1.15:
                continue
            if ratio > cfg.grow_max_area:
                self.log_info(f"  crop_{k_idx} with its {len(parts)} part(s) would be {ratio:.1f}x its box; not grown")
                continue
            wanted.append((k_idx, parts, (x0, y0, x1 - x0, y1 - y0)))
        if not wanted:
            return []
        panorama = context.input_panorama(ContextKey.PANORAMA)
        cone = captured_cone_from_context(context, panorama.width, panorama.height) if panorama is not None else None
        if cone is None:
            self.log_info("Inventory Review: no captured cone; the kept boxes stay as detected")
            return []
        device, _ = preferred_device(DeviceStrategy.MEMORY)
        photo_img, factor = detection_photo(context, device, cfg.photo_detect_sr_factor, self.temp, self.log_info)
        if photo_img is None:
            return []
        boxes = [[int(x * factor), int(y * factor), max(1, int(w * factor)), max(1, int(h * factor))]
                 for _k, _p, (x, y, w, h) in wanted]
        seg = ImageSeg(device)
        try:
            masks = seg.segment_boxes(photo_img, boxes, self.temp)
        finally:
            seg.close()
        grown: list[dict] = []
        for n, mask in enumerate(masks):
            k_idx, parts, _box = wanted[n]
            old = context.object(f"metadata_{k_idx}") or by_idx[k_idx]
            kx, ky, kw, kh = (float(v) for v in by_idx[k_idx]["photo_box"])
            mask = np.asarray(mask, dtype=bool)
            bx, by, bw, bh = boxes[n]
            inside = mask[by:by + bh, bx:bx + bw]
            frac = float(inside.mean()) if inside.size else 0.0
            ys, xs = np.nonzero(mask)
            area = (float(xs.max() - xs.min() + 1) * float(ys.max() - ys.min() + 1) / factor ** 2) if xs.size else 0.0
            if not 0.08 <= frac <= 0.97 or area < 1.1 * kw * kh:
                self.log_info(f"  crop_{k_idx} over its part(s) {parts}: the mask fills {frac:.0%} of the grown box and "
                              f"spans {area / max(kw * kh, 1.0):.1f}x the kept one; not grown")
                continue
            keep = {k: v for k, v in old.items() if k not in ("box", "photo_box", "score", "source", "in_captured_fov")}
            keep["review"] = {**(old.get("review") or {}), "grown_over": parts,
                              "detected_photo_box": [round(v, 1) for v in (kx, ky, kw, kh)]}
            written, _ = store_photo_masks(
                context, cone, photo_img.rgba(), factor, [mask], [float(old.get("score") or 1.0)], k_idx,
                min_area=0.0, max_area=float(cone.photo_w * cone.photo_h), source=str(old.get("source") or "inventory_review"),
                extra=[keep], log=self.log_info,
            )
            if written:
                new = context.object(f"metadata_{k_idx}")
                grown.append({"idx": k_idx, "parts": parts, "photo_box": new.get("photo_box")})
                self.log_info(f"  crop_{k_idx} ({old.get('class')} {str(old.get('name'))!r}) grown over its part(s) "
                              f"{parts}: {kw:.0f}x{kh:.0f} px -> {new['photo_box'][2]:.0f}x{new['photo_box'][3]:.0f} px")
        if grown and self.output is not None:
            cuts = [context.image(f"crop_hires_{g['idx']}") or context.image(f"crop_{g['idx']}") for g in grown]
            for g, c in zip(grown, cuts):
                if c is not None:
                    tile = Image.new("RGBA", c.image.size, (255, 0, 255, 255))
                    tile.alpha_composite(c.image.convert("RGBA"))
                    tile.convert("RGB").save(self.output / f"grown_{g['idx']}.png")
        return grown

    def _recover_missing(self, context: PipelineContext, record: dict, count: int, W: int, H: int) -> list[dict]:
        """Detections for what the reviewer saw and no box covered.

        The detector and the blob pass miss what is small or plain: Shark Fin's three
        people on the sand (17 x 44 px of the photograph), the low rocks in the water,
        Paris's houseboat at the frame's edge. The reviewer names them with a box, and
        until now that went no further than the record, so no scene had a person in
        it. Each missing item of a placeable class is segmented inside its box
        (box-prompted SAM2 on the detection photo, as Object Labeling does for a
        detector box no blob represents) and written as crop_<i> / metadata_<i> like
        any other detection. A mask that fills its whole box, or next to none of it,
        is not the thing named and is left out."""
        import numpy as np
        from pipeline.segmentation.image_segmentation import ImageSeg
        from pipeline.segmentation.photo_detection import detection_photo, store_photo_masks
        from util.captured_cone import captured_cone_from_context
        from util.device_utils import DeviceStrategy, preferred_device

        cfg: InventoryReviewConfiguration = self.config
        wanted = []
        for m in record.get("missing", []):
            cls = normalize_category(str(m.get("category") or "").strip().lower())
            if cls not in cfg.recover_classes or cls not in OBJECT_CATEGORIES:
                continue
            fx, fy, fw, fh = (float(v) for v in m["photo_box"])
            x0, y0 = max(0.0, fx) * W, max(0.0, fy) * H
            x1, y1 = min(1.0, fx + fw) * W, min(1.0, fy + fh) * H
            if min(x1 - x0, y1 - y0) < cfg.recover_min_px:
                continue
            wanted.append((m, cls, (x0, y0, x1 - x0, y1 - y0)))
        if not wanted:
            return []
        panorama = context.input_panorama(ContextKey.PANORAMA)
        cone = captured_cone_from_context(context, panorama.width, panorama.height) if panorama is not None else None
        if cone is None:
            self.log_info("Inventory Review: no captured cone; the missing items stay a record")
            return []
        device, _ = preferred_device(DeviceStrategy.MEMORY)
        photo_img, factor = detection_photo(context, device, cfg.photo_detect_sr_factor, self.temp, self.log_info)
        if photo_img is None:
            return []
        boxes = [[int(x * factor), int(y * factor), max(1, int(w * factor)), max(1, int(h * factor))]
                 for _m, _c, (x, y, w, h) in wanted]
        seg = ImageSeg(device)
        try:
            masks = seg.segment_boxes(photo_img, boxes, self.temp)
        finally:
            seg.close()
        keep = []
        for k, mask in enumerate(masks):
            bx, by, bw, bh = boxes[k]
            inside = np.asarray(mask, dtype=bool)[by:by + bh, bx:bx + bw]
            frac = float(inside.mean()) if inside.size else 0.0
            if 0.08 <= frac <= 0.97:
                keep.append(k)
            else:
                self.log_info(f"  missing {wanted[k][0].get('name')!r}: the mask fills {frac:.0%} of its box; not recovered")
        recovered: list[dict] = []
        idx = count
        for k in keep:
            m, cls, _box = wanted[k]
            n, _ = store_photo_masks(
                context, cone, photo_img.rgba(), factor, [masks[k]], [1.0], idx,
                min_area=0.0, max_area=float(cone.photo_w * cone.photo_h), source="inventory_review", labels=[cls],
                extra=[{"confidence": 1.0, "low_confidence": False, "label_source": "review",
                        "name": str(m.get("name") or cls),
                        "review": {"decision": "missing", "why": str(m.get("why", ""))[:200], "count": m.get("count")}}],
                log=self.log_info,
            )
            if n:
                recovered.append({"idx": idx, "class": cls, "name": m.get("name")})
                idx += n
        if recovered and self.output is not None:
            # The recovered cut-outs side by side, to look at.
            cuts = [context.image(f"crop_hires_{r['idx']}") or context.image(f"crop_{r['idx']}") for r in recovered]
            cuts = [c.image.convert("RGBA") for c in cuts if c is not None]
            if cuts:
                tiles = []
                for c in cuts:
                    k = 256.0 / max(c.size)
                    c = c.resize((max(1, round(c.width * k)), max(1, round(c.height * k))), Image.LANCZOS)
                    tile = Image.new("RGBA", c.size, (255, 0, 255, 255))
                    tile.alpha_composite(c)
                    tiles.append(tile.convert("RGB"))
                sheet = Image.new("RGB", (sum(t.width for t in tiles) + 8 * (len(tiles) - 1), 256), "white")
                x = 0
                for t in tiles:
                    sheet.paste(t, (x, 0))
                    x += t.width + 8
                sheet.save(self.output / "recovered.png")
        if recovered:
            context.add_object(ContextKey.OBJECT_COUNT, idx)
            self.log_info(f"Inventory Review: {len(recovered)} missing item(s) recovered as detections: "
                          + ", ".join(f"crop_{r['idx']} {r['class']} {str(r['name'])!r}" for r in recovered))
        return recovered

    def _erase(self, context: PipelineContext) -> list[dict]:
        """Detections of an erase class leave the inventory and are marked for removal.

        Demoted like a drop (class indeterminate, so nothing reconstructs or places
        them) and flagged `erase`, which Panorama Foreground Inpainting reads: the
        cut-out is removed from the terrain panorama, so the figure is not painted
        on the sand either. One that is part of another object (the crowd aboard a
        boat) stays with it."""
        cfg: InventoryReviewConfiguration = self.config
        erased = []
        for idx in range(int(context.object(ContextKey.OBJECT_COUNT) or 0)):
            m = context.object(f"metadata_{idx}") or {}
            cls = m.get("class")
            if cls not in cfg.erase_classes or m.get("part_of"):
                continue
            context.add_object(f"metadata_{idx}", {
                **m, "class": "indeterminate", "confidence": 0.0, "low_confidence": True, "label_source": "review",
                "vetoed_class": cls, "veto_reason": f"erased: no {cls} is placed; removed from the ground's picture",
                "erase": True,
                "review": {**(m.get("review") or {}), "decision": "erase"},
            })
            erased.append({"idx": idx, "class": cls, "name": m.get("name")})
        if erased:
            self.log_info(f"Inventory Review: {len(erased)} detection(s) erased from the scene: "
                          + ", ".join(f"crop_{e['idx']} {e['class']} {str(e['name'])!r}" for e in erased))
        return erased

    def _apply(self, context: PipelineContext, rows, record: dict) -> list[str]:
        by_idx = {i: m for i, m in rows}
        applied = []
        for d in record.get("decisions", []):
            idx = d.get("idx")
            m = by_idx.get(idx)
            if m is None:
                continue
            why = str(d.get("why", ""))[:200]
            if d.get("decision") in ("drop", "terrain"):
                # A landform is demoted like a drop: Foreground Inpainting removes only
                # object classes, so it stays in the terrain panorama and the height map
                # raises it (Shark Fin's right-hand cliff stood as a 22 m card cut
                # straight down where the photograph ended).
                kind = str(d.get("decision"))
                context.add_object(f"metadata_{idx}", {
                    **m, "class": "indeterminate", "confidence": 0.0, "low_confidence": True, "label_source": "review",
                    "vetoed_class": m.get("class"),
                    "veto_reason": f"review: {'landform, left in the terrain; ' if kind == 'terrain' else ''}{why}",
                    "review": {"decision": kind, "why": why, "name": d.get("name")},
                })
                applied.append(f"{kind} crop_{idx} ({m.get('class')} {str(m.get('name'))!r}): {why}")
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
