"""
Object Labeling: what each segmented blob is, decided by one grounding model.

Replaces the chain Object Classification (BLIP caption → keyword table) →
Object Typing (CLIP over ~90 prompts with three gates, a caption fallback, a
prior fallback, then a Grounding DINO verification pass) → the tag/caption vetoes
in Category Clustering → a separate Object Detection stage that ran Grounding
DINO again with the scene tags. Five models answering one question in sequence,
each patching the one before it.

The question is: given a blob, say what it is or say nothing. Grounding DINO,
prompted with the scene's own RAM++ tags, returns labelled boxes with real
scores. A SAM2 blob that lies mostly inside a labelled box takes that label and
score. A blob no labelled box covers stays `indeterminate`: it still works as an
occluder for Panorama Foreground Inpainting and is never placed. A labelled box
that no blob represents (a tree the automatic segmenter missed) becomes a new
detection, masked by box-prompted SAM2 -- which is all Object Detection did.

Inside the captured cone the detector runs on the photograph (see
segmentation/photo_detection.py, same detection photo as Object Segmentation);
outside it, on the panorama. All boxes are brought into panorama pixel space,
where every blob's box and mask already live.

What is kept from the old chain, because it answers a different question: the
DINOv2 clustering into visual variants (Object Category Clustering) and the
physical vetoes -- size against a metric prior (here) and region type under the
box (Clustering). What is gone: captions, CLIP, the fallbacks, the verification
pass, the caption veto, the scene-category prior.
"""
from __future__ import annotations

import json
import math
from logging import Logger
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image as PILImage

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_detection.grounding_dino import GroundingDino
from pipeline.segmentation.image_segmentation import ImageSeg
from pipeline.segmentation.segmentation_result import SegmentationResult
from pipeline.segmentation.photo_detection import detection_photo, store_photo_masks
from pipeline.segmentation.hires_crops import CAPTURED_FOV_RESTRICTED_KEY
from pipeline.object_typing.categories import (
    OBJECT_CATEGORIES, ENVIRONMENT_CATEGORIES, normalize_category,
)
from pipeline.scene_generation.object_scale import OBJECT_HEIGHT_PRIORS
from util.captured_cone import captured_cone_from_context
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image

LABELING_MARKER = "labeling_complete"


class ObjectLabelingConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        min_coverage: float = 0.6,
        confident_score: float = 0.30,
        default_vocabulary: list[str] | None = None,
        add_unmatched_boxes: bool = True,
        represented_fraction: float = 0.4,
        min_area_fraction: float = 0.0003,
        max_area_fraction: float = 0.20,
        captured_fov_margin_deg: float = 3.0,
        photo_detect_sr_factor: float = 2.0,
        size_veto_ratio: float = 3.0,
        panorama_detect_scale: float = 0.5,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Resolution of the panorama Grounding DINO pass, as a fraction of the
        # panorama's size -- same reasoning as SegmentationConfiguration's
        # panorama_detect_scale: outside the cone only occluder labels are needed.
        self.panorama_detect_scale = float(panorama_detect_scale)
        # Fraction of a blob's mask that must lie inside a labelled box for the blob
        # to take its label.
        self.min_coverage = float(min_coverage)
        # Detector score at or above which a label is trusted for identity (may
        # anchor a visual bucket in Clustering); below it the blob is low_confidence
        # and must earn its class by similarity to a confident one.
        self.confident_score = float(confident_score)
        # Prompted in every scene regardless of the tags: RAM++ tags a landscape
        # "field, flower, mountain" and never "tree", and a scene with no vegetation
        # simply returns no boxes for these. People, vehicles, animals and the built
        # environment are deliberately NOT here -- they come only from the tags.
        self.default_vocabulary = list(
            default_vocabulary if default_vocabulary is not None else ["tree", "bush", "flower", "plant", "rock"]
        )
        # A labelled box whose area is less than represented_fraction covered by
        # blobs becomes a new detection of its own (masked by box-prompted SAM2).
        self.add_unmatched_boxes = bool(add_unmatched_boxes)
        self.represented_fraction = float(represented_fraction)
        # Size bounds for those new detections, as fractions of the detection
        # source's area (same calibration as SegmentationConfiguration).
        self.min_area_fraction = float(min_area_fraction)
        self.max_area_fraction = float(max_area_fraction)
        self.captured_fov_margin_deg = float(captured_fov_margin_deg)
        self.photo_detect_sr_factor = float(photo_detect_sr_factor)
        # A blob labelled with a class that has a metric height prior
        # (OBJECT_HEIGHT_PRIORS) whose measured height is over this many times the
        # prior, or under the prior divided by it, cannot be that class. 0 disables.
        self.size_veto_ratio = float(size_veto_ratio)


class ObjectLabelingStage(PipelineStage):
    @classmethod
    def config_class(cls) -> type[ObjectLabelingConfiguration]:
        return ObjectLabelingConfiguration

    def __init__(self, config: ObjectLabelingConfiguration) -> None:
        super().__init__(config)
        self._detector = None
        self._seg = None
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)

    # ── vocabulary ───────────────────────────────────────────────────────────

    def _vocabulary(self, context: PipelineContext) -> dict[str, str]:
        """phrase -> class. Scene tags normalised onto the known categories, plus
        the always-on defaults. Environment classes are prompted too, so scenery
        blobs get a scenery label rather than competing for an object one."""
        tags_str = context.input_object(ContextKey.RECOGNIZE_TAGS) or ""
        classes: dict[str, str] = {}
        for tag in [t.strip() for t in str(tags_str).split("|") if t.strip()]:
            cls = normalize_category(tag)
            if cls in OBJECT_CATEGORIES or cls in ENVIRONMENT_CATEGORIES:
                classes[cls.replace("_", " ")] = cls
                # A named thing ("Eiffel tower") is prompted by its own name as well as
                # its class. Prompted only as "tower" beside "sky", Grounding DINO gave
                # the Eiffel Tower's box to sky -- its lattice is mostly sky pixels --
                # and the scene lost it; the proper noun is a far stronger match.
                phrase = tag.lower()
                if cls in OBJECT_CATEGORIES and phrase != cls.replace("_", " "):
                    classes[phrase] = cls
        for cls in self.config.default_vocabulary:
            if cls in OBJECT_CATEGORIES:
                classes[cls.replace("_", " ")] = cls
        return classes

    # ── detector passes ──────────────────────────────────────────────────────

    def _detect(self, image: Image, phrases: list[str], label: str) -> list[dict]:
        task = self.create_progress(1, f"Grounding on the {label}…")
        dets = self._detector.detect(
            image.rgb(), phrases, self.temp or self.output, on_progress=self.make_progress_callback(task),
        )
        self.finish_progress(task)
        return dets

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: ObjectLabelingConfiguration = self.config
        object_count = context.input_object(ContextKey.OBJECT_COUNT) or 0
        panorama = context.input_panorama(ContextKey.PANORAMA)
        if panorama is None:
            self.log_warning("No panorama — nothing to label against")
            context.add_object(LABELING_MARKER, True)
            return context
        pano_img = Image(panorama.image)
        pano_w, pano_h = panorama.width, panorama.height

        vocab = self._vocabulary(context)
        phrases = sorted(vocab)
        self.log_info(f"Vocabulary ({len(phrases)}): {', '.join(phrases)}")

        if self._detector is None:
            self._detector = GroundingDino(self.device)

        cone = captured_cone_from_context(context, pano_w, pano_h)
        # Every labelled box in panorama pixel space:
        # {"box", "class", "score", "origin": "photo"|"panorama", "photo_box": native photo box or None}
        labelled: list[dict] = []

        photo_img, factor = (None, 1.0)
        if cone is not None:
            photo_img, factor = detection_photo(
                context, self.preferred_device, cfg.photo_detect_sr_factor, self.temp, self.log_info,
            )
        if photo_img is not None:
            for det in self._detect(photo_img, phrases, "photograph"):
                cls = normalize_category(det["label"])
                if cls not in OBJECT_CATEGORIES and cls not in ENVIRONMENT_CATEGORIES:
                    continue
                x, y, w, h = (float(v) / factor for v in det["box"])
                pu0, pv0 = cone.photo_to_pano(x, y)
                pu1, pv1 = cone.photo_to_pano(x + w, y + h)
                labelled.append({
                    "box": [pu0, pv0, pu1 - pu0, pv1 - pv0], "class": cls,
                    "score": float(det["score"]), "origin": "photo", "photo_box": [x, y, w, h],
                    "name": _phrase(det["label"], phrases),
                })
            # A second, whole-photo pass at a size the detector takes as one tile.
            # The detection photo is super-resolved and tiled at 800 px, so anything
            # taller than a tile is only ever seen truncated and lost: Paris's
            # Eiffel Tower (0.92 on the whole photo) never got a box, and the only
            # "eiffel tower" left was a distant spire.
            whole_scale = min(1.0, 990.0 / max(photo_img.width, photo_img.height))
            whole = Image(photo_img.rgb().resize(
                (max(16, int(photo_img.width * whole_scale)), max(16, int(photo_img.height * whole_scale))),
                PILImage.LANCZOS,
            ))
            tiled = list(labelled)
            extra_boxes = 0
            for det in sorted(self._detect(whole, phrases, "whole photograph"), key=lambda d: -float(d["score"])):
                cls = normalize_category(det["label"])
                if cls not in OBJECT_CATEGORIES:
                    continue
                x, y, w, h = (float(v) / whole_scale / factor for v in det["box"])
                if any(t["class"] == cls and _box_iou(t["photo_box"], [x, y, w, h]) > 0.3 for t in tiled):
                    continue
                tiled.append({"class": cls, "photo_box": [x, y, w, h]})
                pu0, pv0 = cone.photo_to_pano(x, y)
                pu1, pv1 = cone.photo_to_pano(x + w, y + h)
                labelled.append({
                    "box": [pu0, pv0, pu1 - pu0, pv1 - pv0], "class": cls,
                    "score": float(det["score"]), "origin": "photo", "photo_box": [x, y, w, h],
                    "name": _phrase(det["label"], phrases),
                })
                extra_boxes += 1
            self.log_info(f"Photograph: {len(labelled)} labelled box(es) ({extra_boxes} from the whole-photo pass)")

        # A named thing exists once: of its boxes only the best stands (Paris's
        # "eiffel tower" also matched a distant spire at 0.39).
        best_named: dict[str, float] = {}
        for b in labelled:
            if _is_named(b):
                best_named[b["name"]] = max(best_named.get(b["name"], 0.0), b["score"])
        labelled = [b for b in labelled if not _is_named(b) or b["score"] >= best_named[b["name"]]]

        pano_boxes = 0
        scale = cfg.panorama_detect_scale if photo_img is not None else 1.0
        detect_img = pano_img
        if 0.0 < scale < 1.0:
            detect_img = Image(pano_img.rgb().resize(
                (max(16, int(pano_w * scale)), max(16, int(pano_h * scale))), PILImage.LANCZOS,
            ))
        for det in self._detect(detect_img, phrases, "panorama"):
            cls = normalize_category(det["label"])
            if cls not in OBJECT_CATEGORIES and cls not in ENVIRONMENT_CATEGORIES:
                continue
            box = [float(v) / (scale if 0.0 < scale < 1.0 else 1.0) for v in det["box"]]
            # Inside the cone the photograph's pass is the authority.
            if photo_img is not None and cone.contains_box(box, cfg.captured_fov_margin_deg):
                continue
            labelled.append({"box": box, "class": cls, "score": float(det["score"]), "origin": "panorama",
                             "photo_box": None, "name": str(det["label"]).strip().lower()})
            pano_boxes += 1
        self.log_info(f"Panorama: {pano_boxes} labelled box(es)" + (" outside the cone" if photo_img is not None else ""))

        # ── label every blob by containment ───────────────────────────────────
        task = self.create_progress(max(1, object_count), "Labeling blobs…")
        box_arr = np.array([b["box"] for b in labelled], dtype=np.float64).reshape(-1, 4)
        box_area = np.maximum(box_arr[:, 2] * box_arr[:, 3], 1e-6) if len(labelled) else np.zeros(0)
        # Blob area inside each labelled box, for the "represented" test below.
        covered_by_blobs = np.zeros(len(labelled), dtype=np.float64)
        claimed: dict[int, list] = {}   # labelled box -> [(blob area, idx)] that took its label
        counts: dict[str, int] = {}
        unlabelled = 0
        debug = []
        for idx in range(object_count):
            metadata = context.input_object(f"metadata_{idx}") or {}
            crop = context.input_image(f"crop_{idx}")
            box = metadata.get("box")
            if not box or crop is None:
                self.advance_progress(task)
                continue
            bx, by, bw, bh = (float(v) for v in box)
            alpha = np.asarray(crop.image.convert("RGBA").split()[-1]) > 127
            ah, aw = alpha.shape
            total = float(alpha.sum())
            best = None
            if total > 0 and len(labelled):
                # Intersection of each labelled box with this blob's box, in crop pixels.
                ix0 = np.clip(np.floor(box_arr[:, 0] - bx), 0, aw).astype(int)
                iy0 = np.clip(np.floor(box_arr[:, 1] - by), 0, ah).astype(int)
                ix1 = np.clip(np.ceil(box_arr[:, 0] + box_arr[:, 2] - bx), 0, aw).astype(int)
                iy1 = np.clip(np.ceil(box_arr[:, 1] + box_arr[:, 3] - by), 0, ah).astype(int)
                for k in np.flatnonzero((ix1 > ix0) & (iy1 > iy0)):
                    inside = float(alpha[iy0[k]:iy1[k], ix0[k]:ix1[k]].sum())
                    covered_by_blobs[k] += inside
                    coverage = inside / total
                    if coverage < cfg.min_coverage:
                        continue
                    entry = labelled[k]
                    is_object = entry["class"] in OBJECT_CATEGORIES
                    # Object labels beat scenery labels (a flower inside a "field"
                    # box is a flower); then score; then the tighter box.
                    key = (is_object, entry["score"], -box_area[k])
                    if best is None or key > best[0]:
                        best = (key, k, coverage)
            if best is None:
                unlabelled += 1
                new_meta = {
                    **metadata, "class": "indeterminate", "confidence": 0.0,
                    "low_confidence": True, "label_source": "none",
                }
            else:
                _, k, coverage = best
                claimed.setdefault(k, []).append((total, idx))
                entry = labelled[k]
                new_meta = {
                    **metadata, "class": entry["class"], "confidence": round(entry["score"], 4),
                    "low_confidence": entry["score"] < cfg.confident_score,
                    "label_source": entry["origin"], "label_coverage": round(coverage, 3),
                    # The detector's own phrase ("eiffel tower"), for prompts that
                    # need to name the thing rather than its class.
                    "name": entry.get("name") or entry["class"],
                }
                counts[entry["class"]] = counts.get(entry["class"], 0) + 1
            context.add_object(f"metadata_{idx}", new_meta)
            debug.append({"idx": idx, "class": new_meta["class"], "confidence": new_meta["confidence"],
                          "coverage": new_meta.get("label_coverage"), "source": new_meta["label_source"]})
            self.advance_progress(task)
        self.finish_progress(task)
        # A named thing is one object: when its box labelled several blobs (the
        # Eiffel Tower's body and, separately, a sliver of its first platform),
        # the largest keeps the label and the rest are left as parts of it.
        for k, blobs in claimed.items():
            entry = labelled[k]
            if len(blobs) < 2 or not _is_named(entry):
                continue
            for _, idx in sorted(blobs, reverse=True)[1:]:
                m = context.object(f"metadata_{idx}") or {}
                context.add_object(f"metadata_{idx}", {
                    **m, "class": "indeterminate", "confidence": 0.0, "low_confidence": True,
                    "label_source": "none", "part_of": entry.get("name"),
                })
                counts[entry["class"]] -= 1
                unlabelled += 1
        self.log_info(
            f"Labelled {object_count - unlabelled}/{object_count} blob(s): "
            + ", ".join(f"{c} {n}" for c, n in sorted(counts.items(), key=lambda kv: -kv[1]))
            + (f"; {unlabelled} unlabelled (occluders only)" if unlabelled else "")
        )

        # ── boxes no blob represents become detections ────────────────────────
        next_idx = object_count
        added = 0
        if cfg.add_unmatched_boxes and len(labelled):
            represented = covered_by_blobs / box_area
            # A named thing ("eiffel tower") is its own detection unless a blob took
            # its label: Paris's tower is a 9x24 px sliver on the skyline, covered by
            # sky blobs (so "represented") and under the minimum area, and was lost.
            named = {
                k for k in range(len(labelled))
                if k not in claimed and _is_named(labelled[k])
            }
            unmatched = [
                k for k in range(len(labelled))
                if labelled[k]["class"] in OBJECT_CATEGORIES
                and (represented[k] < cfg.represented_fraction or k in named)
                # A named box a blob already took is that blob, however little of
                # its box the blob fills (a lattice tower is mostly sky).
                and not (k in claimed and _is_named(labelled[k]))
            ]
            photo_k = [k for k in unmatched if labelled[k]["origin"] == "photo"]
            pano_k = [k for k in unmatched if labelled[k]["origin"] == "panorama"]
            if photo_k or pano_k:
                if self._seg is None:
                    self._seg = ImageSeg(self.preferred_device)
            native_area = cone.photo_w * cone.photo_h if cone is not None else 0.0
            # Named things skip the minimum area: a landmark far away is small.
            for group, min_area in (
                ([k for k in photo_k if k not in named], cfg.min_area_fraction * native_area),
                ([k for k in photo_k if k in named], 0.0),
            ):
                if not group:
                    continue
                boxes = []
                for k in group:
                    x, y, w, h = labelled[k]["photo_box"]
                    boxes.append([int(x * factor), int(y * factor), max(1, int(w * factor)), max(1, int(h * factor))])
                masks = self._seg.segment_boxes(photo_img, boxes, self.temp)
                n, _ = store_photo_masks(
                    context, cone, photo_img.rgba(), factor, masks,
                    [labelled[k]["score"] for k in group], next_idx,
                    min_area=min_area, max_area=cfg.max_area_fraction * native_area,
                    source="grounding_dino", labels=[labelled[k]["class"] for k in group],
                    extra=[{"low_confidence": labelled[k]["score"] < cfg.confident_score,
                            "confidence": round(labelled[k]["score"], 4), "label_source": "photo",
                            "name": labelled[k].get("name") or labelled[k]["class"]} for k in group],
                    log=self.log_info,
                )
                next_idx += n
                added += n
            if pano_k:
                boxes = []
                for k in pano_k:
                    x, y, w, h = labelled[k]["box"]
                    x1, y1 = max(0, int(x)), max(0, int(y))
                    x2, y2 = min(pano_w, int(x + w)), min(pano_h, int(y + h))
                    boxes.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
                masks = self._seg.segment_boxes(pano_img, boxes, self.temp)
                result = SegmentationResult(masks=masks, boxes=boxes, scores=[labelled[k]["score"] for k in pano_k])
                area_ref = min(pano_w * pano_h, 1024 * 1024)
                for k, crop in zip(pano_k, result.masked_images(pano_img)):
                    x, y, w, h = crop.box
                    if (w * h < cfg.min_area_fraction * area_ref and k not in named) or w * h > cfg.max_area_fraction * area_ref:
                        continue
                    context.add_image(f"crop_{next_idx}", crop.image)
                    context.add_object(f"metadata_{next_idx}", {
                        "box": [float(x), float(y), float(w), float(h)],
                        "score": labelled[k]["score"], "class": labelled[k]["class"],
                        "confidence": round(labelled[k]["score"], 4),
                        "low_confidence": labelled[k]["score"] < cfg.confident_score,
                        "source": "grounding_dino", "label_source": "panorama",
                        "name": labelled[k].get("name") or labelled[k]["class"],
                        "in_captured_fov": False if cone is not None else True,
                    })
                    next_idx += 1
                    added += 1
            self.log_info(f"Added {added} detection(s) for labelled boxes no blob represented")
        if cone is not None:
            context.add_object(CAPTURED_FOV_RESTRICTED_KEY, True)
        context.add_object(ContextKey.OBJECT_COUNT, next_idx)

        self._size_veto(context, next_idx, panorama)

        if self.output is not None:
            (self.output / "labeling_debug.json").write_text(json.dumps({
                "vocabulary": vocab, "boxes": labelled, "blobs": debug,
            }, indent=1, default=float))
        context.add_object(LABELING_MARKER, True)
        return context

    def _size_veto(self, context: PipelineContext, count: int, panorama) -> None:
        ratio = self.config.size_veto_ratio
        if ratio <= 1.0:
            return
        depth_value = context.input_depth(ContextKey.PANORAMA_OBJECT_DEPTH)
        if depth_value is None:
            return
        depth = depth_value.depth
        dh, dw = depth.shape[:2]
        sx, sy = dw / float(panorama.width), dh / float(panorama.height)
        vetoed, examples = 0, []
        for idx in range(count):
            metadata = context.object(f"metadata_{idx}") or {}
            cls = metadata.get("class")
            box = metadata.get("box")
            if cls not in OBJECT_HEIGHT_PRIORS or not box:
                continue
            x, y, w, h = box
            cx, cy = int(round((x + w / 2.0) * sx)), int(round((y + h / 2.0) * sy))
            patch = depth[max(0, cy - 4):cy + 5, max(0, cx - 4):cx + 5]
            valid = patch[np.isfinite(patch) & (patch > 0)]
            if valid.size == 0:
                continue
            d = float(np.median(valid))
            height_m = 2.0 * d * math.tan(0.5 * (h / float(panorama.height)) * math.pi)
            prior = OBJECT_HEIGHT_PRIORS[cls][0]
            if height_m > prior * ratio or height_m < prior / ratio:
                reason = f"{height_m:.1f} m tall at {d:.0f} m, prior {prior:.1f} m"
                context.add_object(f"metadata_{idx}", {
                    **metadata, "class": "indeterminate", "vetoed_class": cls,
                    "veto_reason": f"size: {reason}", "low_confidence": True,
                })
                vetoed += 1
                if len(examples) < 6:
                    examples.append(f"crop_{idx} ({cls}): {reason}")
        if vetoed:
            self.log_info(f"Size veto: {vetoed} crop(s) demoted to indeterminate — " + "; ".join(examples))

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(LABELING_MARKER) is True

    def model_names(self) -> list[str]:
        return GroundingDino.model_names() + ImageSeg.model_names()

    def clean_up(self):
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        if self._seg is not None:
            self._seg.close()
            self._seg = None
        super().clean_up()


def _box_iou(a, b) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _phrase(label: str, phrases: list[str]) -> str:
    """The vocabulary phrase a detector label names. Grounding DINO joins the
    tokens of adjacent phrases ("eiffel tower tower"); the longest vocabulary
    phrase inside the label is the one meant."""
    label = str(label).strip().lower()
    inside = [p for p in phrases if p in label]
    return max(inside, key=len) if inside else label


def _is_named(entry: dict) -> bool:
    """A proper-noun box ("eiffel tower"), not a class word ("tower", "river")."""
    name = str(entry.get("name") or entry["class"]).replace(" ", "_")
    return (entry["class"] in OBJECT_CATEGORIES
            and name not in OBJECT_CATEGORIES and name not in ENVIRONMENT_CATEGORIES
            and normalize_category(name) != name)
