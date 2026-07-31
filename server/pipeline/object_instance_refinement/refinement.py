import json
from logging import Logger
from typing import Any

import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import scipy.ndimage
import skimage.feature
import skimage.segmentation
import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_typing.categories import COALESCE_CATEGORIES, VEGETATION_CATEGORIES, ABSORB_EXEMPT_CATEGORIES
from util.image_utils import Image
from util.instance_merge import (
    cluster_indices,
    wrap_aware_box_iou,
    wrap_aware_containment,
    dilated_masks_touch,
    union_mask_and_box,
)


class ObjectInstanceRefinementConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        # coalesce (merge) tuning -- COALESCE_CATEGORIES
        coalesce_box_iou_threshold: float = 0.20,
        coalesce_containment_threshold: float = 0.65,
        coalesce_touch_dilation_px: int = 15,
        # split (vegetation) tuning -- VEGETATION_CATEGORIES
        split_min_component_px: int = 400,
        split_min_component_fraction: float = 0.05,
        split_outlier_area_multiplier: float = 2.75,
        split_watershed_min_peak_distance_fraction: float = 0.18,
        split_watershed_min_component_fraction: float = 0.12,
        split_max_components: int = 12,
        # absorb (drop structural sub-parts) tuning -- COALESCE_CATEGORIES parents
        absorb_containment_threshold: float = 0.85,
        # dedupe (drop the same instance detected twice) -- runs AFTER the split pass,
        # deliberately high so it only ever collapses near-identical boxes. See
        # _dedupe_pass for why the coalesce pass cannot cover this.
        #
        # 0.95 measured, not guessed. Swept against the Rainier capture's 404
        # classed+boxed instances, counting how many objects the pass would drop:
        #
        #   IoU >= 0.99 -> 19  (17 tree, 1 flower, 1 sky)
        #   IoU >= 0.95 -> 19  (17 tree, 1 flower, 1 sky)
        #   IoU >= 0.90 -> 22  (17 tree, 4 flower, 1 sky)
        #   IoU >= 0.80 -> 31  (17 tree, 13 flower, 1 sky)
        #
        # The tree duplicates are byte-identical boxes (IoU exactly 1.0), so the tree
        # count is flat from 0.99 all the way to 0.80 -- every threshold in that range
        # catches all of them, and the choice is purely about how much collateral it
        # takes with them. Flowers are what moves: a dense meadow has genuinely
        # adjacent, heavily-overlapping blooms, and by 0.80 the pass is deleting
        # thirteen of them. Since the complaint this work started from is too FEW
        # flowers, the flat part of the tree curve is the right place to sit, not the
        # knee of the flower curve.
        dedupe_box_iou_threshold: float = 0.95,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.coalesce_box_iou_threshold = coalesce_box_iou_threshold
        self.coalesce_containment_threshold = coalesce_containment_threshold
        self.coalesce_touch_dilation_px = coalesce_touch_dilation_px
        self.split_min_component_px = split_min_component_px
        self.split_min_component_fraction = split_min_component_fraction
        self.split_outlier_area_multiplier = split_outlier_area_multiplier
        self.split_watershed_min_peak_distance_fraction = split_watershed_min_peak_distance_fraction
        self.split_watershed_min_component_fraction = split_watershed_min_component_fraction
        self.split_max_components = split_max_components
        self.absorb_containment_threshold = absorb_containment_threshold
        self.dedupe_box_iou_threshold = dedupe_box_iou_threshold


class ObjectInstanceRefinementStage(PipelineStage):
    """
    Sits between Object Detection and Object Correlation. Three independent,
    taxonomy-driven passes over the current crop_{i}/metadata_{i} set:

      1. Coalesce (COALESCE_CATEGORIES): same-category fragments of one
         physical structure (building/wall/fence/rock/skyscraper/barrier)
         that overlap, are mostly-contained, or are touching/adjacent are
         unioned into a single metadata/crop entry.
      2. Split (VEGETATION_CATEGORIES): a single mask that is really several
         disjoint or clustered objects (e.g. one SAM2 blob covering several
         nearby trees) is split into multiple crop_{i}/metadata_{i} entries --
         connected components first (cheap, unambiguous), then watershed
         (gated to statistical size-outliers only) for touching canopy that
         forms one connected mask. Capped at split_max_components pieces so
         one blob can't explode into leaf/twig-level fragments.
      3. Absorb (COALESCE_CATEGORIES parents, any-category children): a small
         detection heavily contained within a larger structural detection's
         box -- e.g. a door or window typed differently than its building, so
         the same-class-only coalesce pass above can't reach it -- is dropped
         entirely rather than kept as a spurious separate object. Exempts
         ABSORB_EXEMPT_CATEGORIES (person/car/etc.), which can legitimately
         overlap a structure's box without being part of it.

    Reads:  ContextKey.OBJECT_COUNT, metadata_{i}, crop_{i}, SemanticKey.INPUT
            (default ContextKey.PANORAMA -- needed to rebuild merged crops'
            RGBA pixels for the union footprint; the split path only ever
            subsets an existing crop's own pixels/mask, no re-read needed)
    Writes: metadata_{i} (merged in place, or nulled for merged-away/absorbed
            slots), crop_{i} (merged crop written at the surviving/canonical
            index; new crop_{i} appended for each split sub-instance),
            ContextKey.OBJECT_COUNT (bumped for split-added slots only --
            merges/absorptions never grow or shrink the visible count, they
            just null gaps)
    Debug:  self.output/refinement_debug.json -- before/after counts, merges,
            splits, absorptions, thresholds used
            self.output/debug_before.png / debug_after.png -- panorama with
            colored boxes per category, before vs after this stage
    """

    @classmethod
    def config_class(cls):
        return ObjectInstanceRefinementConfiguration

    def _input_key(self):
        return self._resolve_key(SemanticKey.INPUT, ContextKey.PANORAMA)

    def _resolve_source(self, context: PipelineContext, input_key) -> "Image | None":
        image = context.input_image(input_key)
        if image is not None:
            return image
        panorama = context.input_panorama(input_key)
        if panorama is not None:
            return Image(panorama.image)
        return None

    def run(self, context: PipelineContext) -> PipelineContext:
        object_count = context.input_object(ContextKey.OBJECT_COUNT)
        if not object_count:
            self.log_info("No objects to refine, skipping")
            return context

        source_image = self._resolve_source(context, self._input_key())
        pano_w = source_image.width if source_image is not None else None

        working: dict[int, dict] = {}
        for idx in range(object_count):
            metadata = context.input_object(f"metadata_{idx}")
            if metadata is None:
                continue
            box = metadata.get("box")
            crop = context.input_image(f"crop_{idx}")
            mask = None
            if crop is not None and crop.image.mode == "RGBA":
                mask = np.asarray(crop.image.getchannel("A")) > 0
            working[idx] = {"metadata": metadata, "mask": mask, "box": box}

        before_snapshot = {idx: dict(entry["metadata"]) for idx, entry in working.items()}

        merges = []
        if pano_w is not None:
            merges = self._coalesce_pass(context, working, pano_w, source_image)

        next_idx = object_count
        splits = []
        if pano_w is not None:
            splits, next_idx = self._split_pass(context, working, next_idx)

        # After the split pass on purpose -- the duplicates that survive to here are
        # mostly ones the split pass just created. See _dedupe_pass.
        duplicates = []
        if pano_w is not None:
            duplicates = self._dedupe_pass(context, working, pano_w)

        absorptions = []
        if pano_w is not None:
            absorptions = self._absorb_pass(context, working, pano_w)

        context.add_object(ContextKey.OBJECT_COUNT, next_idx)
        context.add_object("refinement_complete", True)

        self.log_info(
            f"  {len(merges)} coalesce merge(s), {len(splits)} split(s), "
            f"{len(duplicates)} duplicate(s) dropped, "
            f"{len(absorptions)} absorption(s); count {object_count} -> {next_idx}"
        )

        self._write_debug(context, source_image, pano_w, before_snapshot, working, merges, splits, duplicates, absorptions, object_count, next_idx)
        return context

    # ------------------------------------------------------------------
    # Coalesce pass
    # ------------------------------------------------------------------

    def _coalesce_score(self, a: dict, b: dict, pano_w: int) -> float:
        box_iou = wrap_aware_box_iou(a["box"], b["box"], pano_w)
        containment = wrap_aware_containment(a["box"], b["box"], pano_w)
        if containment >= self.config.coalesce_containment_threshold:
            return 1.0
        if a["mask"] is not None and b["mask"] is not None:
            touching = dilated_masks_touch(
                a["mask"], (a["box"][0], a["box"][1]),
                b["mask"], (b["box"][0], b["box"][1]),
                pano_w, self.config.coalesce_touch_dilation_px,
            )
            if touching:
                return 1.0
        return box_iou

    def _coalesce_pass(self, context: PipelineContext, working: dict[int, dict], pano_w: int, source_image) -> list[dict]:
        by_class: dict[str, list[int]] = {}
        for idx, entry in working.items():
            cls = entry["metadata"].get("class")
            if cls in COALESCE_CATEGORIES and entry["box"] is not None:
                by_class.setdefault(cls, []).append(idx)

        merges = []
        for cls, indices in by_class.items():
            if len(indices) < 2:
                continue
            dets = [{"idx": i, "box": working[i]["box"], "mask": working[i]["mask"]} for i in indices]
            clusters = cluster_indices(
                dets,
                overlap_fn=lambda a, b: self._coalesce_score(a, b, pano_w),
                threshold=self.config.coalesce_box_iou_threshold,
            )
            for members in clusters:
                if len(members) < 2:
                    continue
                member_idxs = [dets[m]["idx"] for m in members]
                canonical = min(member_idxs)
                entries = [(working[i]["mask"], working[i]["box"]) for i in member_idxs if working[i]["mask"] is not None]
                if len(entries) < 2:
                    continue

                union_mask, union_box = union_mask_and_box(entries, pano_w)
                crop_x, crop_y = int(union_box[0]), int(union_box[1])
                crop_w, crop_h = int(round(union_box[2])), int(round(union_box[3]))
                if crop_w <= 0 or crop_h <= 0:
                    continue

                merged_metadata = {
                    **working[canonical]["metadata"],
                    "box": [float(crop_x), float(crop_y), float(crop_w), float(crop_h)],
                    "score": max(working[i]["metadata"].get("score", 0.0) for i in member_idxs),
                    "source": "coalesced",
                    "merged_from": sorted(member_idxs),
                }
                confidences = [working[i]["metadata"]["confidence"] for i in member_idxs if "confidence" in working[i]["metadata"]]
                if confidences:
                    merged_metadata["confidence"] = max(confidences)

                region = self._crop_union_region(source_image, union_mask, crop_x, crop_y, crop_w, crop_h, pano_w)

                context.add_image(f"crop_{canonical}", region)
                context.add_object(f"metadata_{canonical}", merged_metadata)
                working[canonical] = {"metadata": merged_metadata, "mask": union_mask, "box": merged_metadata["box"]}
                for i in member_idxs:
                    if i != canonical:
                        context.add_object(f"metadata_{i}", None)
                        del working[i]
                merges.append({"class": cls, "canonical": canonical, "merged_from": sorted(member_idxs)})
        return merges

    def _crop_union_region(self, source_image, union_mask: np.ndarray, x: int, y: int, w: int, h: int, pano_w: int) -> PIL.Image.Image:
        """Re-crop the merged footprint's RGB straight from the shared source
        image (resolved once in run()) -- alpha comes from union_mask -- with
        wrap-aware column indexing so a merged cluster straddling the
        panorama seam still produces one coherent crop."""
        source_rgb = np.asarray(source_image.rgb()) if source_image is not None else None

        if source_rgb is None:
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., 3] = union_mask.astype(np.uint8) * 255
            return PIL.Image.fromarray(rgba, mode="RGBA")

        cols = (np.arange(x, x + w)) % pano_w
        rows = np.clip(np.arange(y, y + h), 0, source_rgb.shape[0] - 1)
        region_rgb = source_rgb[np.ix_(rows, cols)]
        rgba = np.dstack([region_rgb, union_mask.astype(np.uint8) * 255])
        return PIL.Image.fromarray(rgba, mode="RGBA")

    # ------------------------------------------------------------------
    # Split pass
    # ------------------------------------------------------------------

    def _tier1_connected_components(self, mask: np.ndarray) -> "list[np.ndarray] | None":
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        if num_labels <= 2:
            return None
        total = mask.sum()
        floor = max(self.config.split_min_component_px, self.config.split_min_component_fraction * total)
        comps = [(labels == lbl) for lbl in range(1, num_labels)]
        comps = [c for c in comps if c.sum() >= floor]
        return comps if len(comps) >= 2 else None

    def _tier2_watershed(self, mask: np.ndarray) -> "list[np.ndarray] | None":
        distance = scipy.ndimage.distance_transform_edt(mask)
        h, w = mask.shape
        min_dist = max(3, int(self.config.split_watershed_min_peak_distance_fraction * min(h, w)))
        coords = skimage.feature.peak_local_max(distance, min_distance=min_dist, labels=mask)
        if len(coords) < 2:
            return None
        peak_mask = np.zeros_like(mask, dtype=bool)
        peak_mask[tuple(coords.T)] = True
        markers, num_markers = scipy.ndimage.label(peak_mask)
        labels_ws = skimage.segmentation.watershed(-distance, markers, mask=mask)
        total = mask.sum()
        floor = self.config.split_watershed_min_component_fraction * total
        comps = [(labels_ws == lbl) for lbl in range(1, num_markers + 1)]
        comps = [c for c in comps if c.sum() >= floor]
        return comps if len(comps) >= 2 else None

    def _split_pass(self, context: PipelineContext, working: dict[int, dict], next_idx: int) -> tuple[list[dict], int]:
        areas_by_class: dict[str, list[float]] = {}
        for entry in working.values():
            cls = entry["metadata"].get("class")
            if cls in VEGETATION_CATEGORIES and entry["mask"] is not None:
                areas_by_class.setdefault(cls, []).append(float(entry["mask"].sum()))

        splits = []
        for idx in list(working.keys()):
            entry = working[idx]
            cls = entry["metadata"].get("class")
            mask = entry["mask"]
            if cls not in VEGETATION_CATEGORIES or mask is None:
                continue

            sub_masks = self._tier1_connected_components(mask)
            tier = "connected_components" if sub_masks is not None else None
            if sub_masks is None:
                median_area = float(np.median(areas_by_class.get(cls, [mask.sum()])))
                if median_area > 0 and mask.sum() / median_area >= self.config.split_outlier_area_multiplier:
                    sub_masks = self._tier2_watershed(mask)
                    tier = "watershed" if sub_masks is not None else None

            if not sub_masks or len(sub_masks) < 2:
                continue

            # Cap to the biggest split_max_components pieces -- keeps genuinely
            # distinct sub-trees/bushes while dropping the leaf/twig-scale tail
            # a noisy mask can otherwise explode into.
            sub_masks = sorted(sub_masks, key=lambda m: m.sum(), reverse=True)[:self.config.split_max_components]
            box = entry["metadata"]["box"]
            crop = context.input_image(f"crop_{idx}")
            parent_rgba = np.asarray(crop.image)

            new_indices = []
            for i, sub in enumerate(sub_masks):
                ys, xs = np.nonzero(sub)
                x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
                sub_rgba = parent_rgba[y0:y1, x0:x1].copy()
                sub_rgba[..., 3] = np.where(sub[y0:y1, x0:x1], sub_rgba[..., 3], 0)
                sub_box = [box[0] + x0, box[1] + y0, x1 - x0, y1 - y0]
                sub_metadata = {**entry["metadata"], "box": [float(v) for v in sub_box], "source": f"split_{tier}", "split_from": idx}

                target_idx = idx if i == 0 else next_idx
                if i > 0:
                    next_idx += 1
                context.add_image(f"crop_{target_idx}", PIL.Image.fromarray(sub_rgba, mode="RGBA"))
                context.add_object(f"metadata_{target_idx}", sub_metadata)
                new_indices.append(target_idx)

            splits.append({"class": cls, "tier": tier, "parent": idx, "new_indices": new_indices})

        return splits, next_idx

    # ------------------------------------------------------------------
    # Dedupe pass
    # ------------------------------------------------------------------

    def _dedupe_pass(self, context: PipelineContext, working: dict[int, dict], pano_w: int) -> list[dict]:
        """Drop the same instance detected twice: near-identical boxes of the same class.

        This is NOT what _coalesce_pass does, and it cannot be folded into it, for
        two independent reasons:

          - Ordering. Coalesce runs BEFORE _split_pass, so every duplicate the split
            pass itself emits is created after the only thing that could have merged
            it. On the Rainier capture that is where the duplicates come from: eight
            pairs of split children with byte-identical boxes, identical scores and
            the same split_from parent, at a constant index offset from each other.
          - Intent. Coalesce merges *parts of one object* into their union (threshold
            0.20, plus containment and mask-touch rules), which is the right operation
            for a tree detected as trunk + canopy and the wrong one for the same tree
            detected twice -- unioning two copies of one box just reproduces the box
            while keeping the extra instance alive.

        _absorb_pass does not reach them either: it skips any candidate that is itself
        in COALESCE_CATEGORIES, so two identical `tree` boxes are each other's parent
        and neither is ever dropped.

        So: cluster on box IoU alone at a deliberately high threshold, keep the lowest
        index, and drop the rest outright. No mask union, no box growth -- a duplicate
        contributes no new extent by definition. Runs over every class with a box, not
        just COALESCE_CATEGORIES, because the duplicates that matter here are trees and
        flowers, and a duplicated instance is wrong for any class: it doubles that
        object's weight in every downstream population, density and scale statistic,
        and renders two coincident meshes that z-fight.
        """
        by_class: dict[str, list[int]] = {}
        for idx, entry in working.items():
            cls = entry["metadata"].get("class")
            if cls and entry["box"] is not None:
                by_class.setdefault(cls, []).append(idx)

        duplicates = []
        for cls, indices in by_class.items():
            if len(indices) < 2:
                continue
            dets = [{"idx": i, "box": working[i]["box"]} for i in indices]
            clusters = cluster_indices(
                dets,
                overlap_fn=lambda a, b: wrap_aware_box_iou(a["box"], b["box"], pano_w),
                threshold=self.config.dedupe_box_iou_threshold,
            )
            for members in clusters:
                if len(members) < 2:
                    continue
                member_idxs = sorted(dets[m]["idx"] for m in members)
                # Lowest index wins: it is the earliest-created instance, so anything
                # already referring to it (split_from, correlation) stays valid.
                canonical = member_idxs[0]
                for i in member_idxs[1:]:
                    self.log_info(
                        f"  crop_{i} ({cls}) is a duplicate of crop_{canonical}, dropping"
                    )
                    context.add_object(f"metadata_{i}", None)
                    del working[i]
                    duplicates.append({"idx": i, "class": cls, "duplicate_of": canonical})

        return duplicates

    # ------------------------------------------------------------------
    # Absorb pass
    # ------------------------------------------------------------------

    def _absorb_pass(self, context: PipelineContext, working: dict[int, dict], pano_w: int) -> list[dict]:
        """Drop small detections heavily contained within a larger
        COALESCE_CATEGORIES detection's box, regardless of class -- catches
        structural sub-parts (a door/window typed differently than its
        building) that the same-class-only coalesce pass above can't reach,
        without discarding legitimately independent foreground objects
        (ABSORB_EXEMPT_CATEGORIES) that just happen to overlap a structure's
        box."""
        structural = [
            idx for idx, entry in working.items()
            if entry["metadata"].get("class") in COALESCE_CATEGORIES and entry["box"] is not None
        ]
        if not structural:
            return []

        absorbed = []
        for idx in list(working.keys()):
            if idx in structural:
                continue
            entry = working[idx]
            cls = entry["metadata"].get("class")
            box = entry["box"]
            if box is None or cls in ABSORB_EXEMPT_CATEGORIES:
                continue

            for parent_idx in structural:
                parent_box = working[parent_idx]["box"]
                containment = wrap_aware_containment(box, parent_box, pano_w)
                if containment >= self.config.absorb_containment_threshold:
                    self.log_info(
                        f"  crop_{idx} ({cls}) absorbed into crop_{parent_idx} "
                        f"({working[parent_idx]['metadata'].get('class')}), containment {containment:.2f}"
                    )
                    context.add_object(f"metadata_{idx}", None)
                    del working[idx]
                    absorbed.append({
                        "idx": idx, "class": cls, "absorbed_into": parent_idx,
                        "containment": round(containment, 3),
                    })
                    break

        return absorbed

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def _write_debug(self, context, source_image, pano_w, before_snapshot, working, merges, splits, duplicates, absorptions, before_count, after_count):
        if self.output is None:
            return

        payload = {
            "before_count": before_count,
            "after_count": after_count,
            "merges": merges,
            "splits": splits,
            "duplicates": duplicates,
            "absorptions": absorptions,
            "config": {
                "coalesce_box_iou_threshold": self.config.coalesce_box_iou_threshold,
                "coalesce_containment_threshold": self.config.coalesce_containment_threshold,
                "coalesce_touch_dilation_px": self.config.coalesce_touch_dilation_px,
                "split_outlier_area_multiplier": self.config.split_outlier_area_multiplier,
                "split_max_components": self.config.split_max_components,
                "absorb_containment_threshold": self.config.absorb_containment_threshold,
                "dedupe_box_iou_threshold": self.config.dedupe_box_iou_threshold,
            },
        }
        with open(self.output / "refinement_debug.json", "w") as f:
            json.dump(payload, f, indent=2)

        if source_image is None:
            return

        self._draw_debug_panorama(source_image, before_snapshot, self.output / "debug_before.png")
        after_snapshot = {idx: entry["metadata"] for idx, entry in working.items()}
        self._draw_debug_panorama(source_image, after_snapshot, self.output / "debug_after.png")

    def _draw_debug_panorama(self, source_image, metadata_by_idx: dict[int, dict], path):
        base = source_image.rgb().convert("RGBA")
        overlay = PIL.Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = PIL.ImageDraw.Draw(overlay)
        font = PIL.ImageFont.load_default()

        categories = sorted({m.get("class", "unknown") for m in metadata_by_idx.values()})
        import colorsys
        colors = {}
        for i, cat in enumerate(categories):
            h = i / max(len(categories), 1)
            r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.95)
            colors[cat] = (int(r * 255), int(g * 255), int(b * 255))

        for idx, metadata in metadata_by_idx.items():
            box = metadata.get("box")
            if not box:
                continue
            cls = metadata.get("class", "unknown")
            r, g, b = colors[cls]
            x, y, w, h = box
            draw.rectangle([x, y, x + w, y + h], outline=(r, g, b, 220), width=2)
            draw.text((x + 4, y + 4), f"{cls}:{idx}", fill=(r, g, b, 255), font=font)

        composite = PIL.Image.alpha_composite(base, overlay).convert("RGB")
        composite.save(path)

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object("refinement_complete") is not None
