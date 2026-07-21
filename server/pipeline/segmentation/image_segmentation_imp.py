from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2_hf
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from remote_connection.remote_server import RemoteServer
from util.panorama_tiling import build_tiles, extract_tile, to_panorama_box, is_edge_truncated
from util.instance_merge import merge_tiled_instances, wrap_aware_box_iou, wrap_aware_mask_iou

# SAM2's Hiera image encoder always operates at a fixed square resolution
# internally regardless of input size -- 1024 tiles avoid wasted extra
# downscaling on top of that. Segmenting a whole multi-thousand-pixel-wide
# equirectangular panorama in one pass would squash everything down to that
# same fixed resolution, blurring/missing small or pole-adjacent objects the
# same way panorama_segmentation_imp.py's own tiling exists to avoid for
# SegFormer -- but a discrete instance stradding a tile boundary gets
# detected twice (once per tile) rather than smoothly blended, so this needs
# its own cross-tile merge (see util/instance_merge.py) unlike that per-pixel
# label-voting scheme.
_TILE_SIZE = 1024
_TILE_OVERLAP_FRAC = 0.25
# Below this, treat the input as an ordinary (non-panoramic) photo and run
# SAM2 directly in one pass -- tiling only pays for itself on genuinely wide
# panoramic input; a narrow hero photo already fits inside one tile.
_TILING_MIN_SIZE = int(_TILE_SIZE * (1 + _TILE_OVERLAP_FRAC))
# Pixels of margin at a tile's own edge within which a detection is treated
# as truncated (see is_edge_truncated) rather than a genuine small object
# that happens to sit near the boundary.
_TRUNCATION_MARGIN_PX = 12
# Cheap box-IoU pre-filter before the more expensive mask render (see
# util.instance_merge.wrap_aware_mask_iou) -- below this, two tile-local
# detections are clearly unrelated, not worth comparing masks at all.
_MERGE_BOX_PREFILTER_IOU = 0.1
# Mask-IoU threshold for treating two tile-local detections as the same real
# object. Looser than ObjectCorrelationStage's 0.5 SAM-vs-DINO threshold,
# since tile/seam truncation legitimately shrinks apparent overlap even for
# a genuinely identical object.
_MERGE_MASK_IOU = 0.3
# Context margin around each box in _segment_boxes' panorama-scale path, as a
# fraction of the box's own width/height on each side.
_BOX_CROP_MARGIN_FRAC = 0.5
# SAM2's own per-tile region-area floor (pixels, not a fraction -- this runs
# before any of our own area-fraction filtering, which happens later in
# pipeline/segmentation/segmentation.py's SegmentationStage.run() against the
# *merged, panorama-space* box). A raw 50px floor lets an enormous amount of
# texture/shadow noise (a crack, a mottled patch, a shadow sliver on a cliff
# face) survive as a candidate mask on every one of the ~64x64 point-grid
# probes across every tile -- each of which then has to be scored, NMS'd, and
# fed into the O(n^2) cross-tile merge in util/instance_merge.py, so this
# floor has an outsized effect on both result quality and wall-clock time.
# 0.0003 matches SegmentationConfiguration.min_area_fraction's own real-photo
# calibration (see its docstring), applied to one tile's area (_TILE_SIZE^2)
# rather than 50 raw pixels -- consistent with the *intent* of that existing,
# already-validated calibration instead of an arbitrary round number.
_MIN_MASK_REGION_AREA_PX = int(0.0003 * _TILE_SIZE * _TILE_SIZE)


class ImageSegServer(RemoteServer):
    def setup(self):
        self.model = build_sam2_hf(
            "facebook/sam2.1-hiera-large",
            device=self.device
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=64,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.80,
            min_mask_region_area=_MIN_MASK_REGION_AREA_PX,
            box_nms_thresh=0.7,
            crop_n_layers=1,
            crop_overlap_ratio=0.5,
        )
        self.predictor = SAM2ImagePredictor(self.model)

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "segment":
            return self._segment(input)
        if action == "segment_boxes":
            return self._segment_boxes(input["image"], input["boxes"])
        raise ValueError(f"Unknown action: {action}")

    def _segment(self, image: Image.Image) -> dict:
        orig_w, orig_h = image.size
        image_np = np.array(image.convert("RGB"))

        if max(orig_w, orig_h) <= _TILING_MIN_SIZE:
            self.report_progress(0.1, "Running SAM2 segmentation…")
            with torch.no_grad():
                results = self.mask_generator.generate(image_np)
            self.report_progress(1.0, "Done")
            return {
                "masks": [r["segmentation"] for r in results],
                "boxes": [list(r["bbox"]) for r in results],
                "scores": [float(r["predicted_iou"]) for r in results],
            }

        return self._segment_tiled(image_np, orig_w, orig_h)

    def _segment_tiled(self, image_np: np.ndarray, pano_w: int, pano_h: int) -> dict:
        """Panorama-scale path: run the automatic mask generator per tile (see
        module docstring/_TILE_SIZE), translate each surviving (non-edge-
        truncated) detection into panorama pixel space, then merge detections
        that are really the same object seen in two overlapping tiles."""
        tiles = build_tiles(pano_w, pano_h, _TILE_SIZE, _TILE_OVERLAP_FRAC)
        self.report_progress(0.05, f"Segmenting {len(tiles)} tile(s)…")

        # Each entry: box (panorama-space [x,y,w,h]), score, mask (tile-local
        # boolean array, same size as its own tile), and that tile's own
        # panorama-space offset (needed to place/compare the mask later).
        dets: list[dict] = []
        for i, tile in enumerate(tiles):
            crop_np = extract_tile(image_np, tile)
            with torch.no_grad():
                results = self.mask_generator.generate(crop_np)
            for r in results:
                local_box = list(r["bbox"])
                if is_edge_truncated(local_box, tile, pano_h, margin_px=_TRUNCATION_MARGIN_PX):
                    continue
                dets.append({
                    "box": to_panorama_box(local_box, tile, pano_w),
                    "score": float(r["predicted_iou"]),
                    "mask": r["segmentation"],
                    "tile_x0": tile.x0,
                    "tile_y0": tile.y0,
                })
            self.report_progress(0.05 + 0.80 * (i + 1) / len(tiles), f"Segmenting tiles… ({i + 1}/{len(tiles)})")

        def overlap_fn(a: dict, b: dict) -> float:
            if wrap_aware_box_iou(a["box"], b["box"], pano_w) < _MERGE_BOX_PREFILTER_IOU:
                return 0.0
            return wrap_aware_mask_iou(
                a["mask"], a["tile_x0"], a["tile_y0"],
                b["mask"], b["tile_x0"], b["tile_y0"],
                pano_w,
            )

        merged = merge_tiled_instances(dets, overlap_fn, _MERGE_MASK_IOU, pano_w)
        self.report_progress(0.9, f"Pasting {len(merged)} merged mask(s)…")

        masks_out, boxes_out, scores_out = [], [], []
        for d in merged:
            canvas = np.zeros((pano_h, pano_w), dtype=bool)
            mh, mw = d["mask"].shape
            rows = d["tile_y0"] + np.arange(mh)
            cols = (d["tile_x0"] + np.arange(mw)) % pano_w
            row_ok = (rows >= 0) & (rows < pano_h)
            if row_ok.any():
                canvas[np.ix_(rows[row_ok], cols)] = d["mask"][row_ok, :]
            masks_out.append(canvas)
            boxes_out.append(d["box"])
            scores_out.append(d["score"])

        self.report_progress(1.0, "Done")
        return {"masks": masks_out, "boxes": boxes_out, "scores": scores_out}

    def _segment_boxes(self, image: Image.Image, boxes: list) -> dict:
        """Box-prompted SAM2: one mask per given [x, y, w, h] box, in the same image.

        For a large (panorama-scale) image, each box is predicted from a local
        crop around just that box (plus _BOX_CROP_MARGIN_FRAC context) instead
        of a single whole-image encode -- the same fixed-resolution-encoder
        distortion _segment_tiled exists to avoid for the automatic mask
        generator applies here too, and these boxes (from ObjectDetectionStage's
        Grounding DINO detections) can be scattered anywhere across a huge
        panorama. A small (non-panoramic) image keeps the original
        single-encode-many-boxes behaviour, since cropping wouldn't change
        anything there but would lose that shared-encode efficiency.
        """
        image_np = np.array(image.convert("RGB"))
        img_h, img_w = image_np.shape[:2]
        masks = []

        if max(img_w, img_h) <= _TILING_MIN_SIZE:
            self.report_progress(0.1, "Running SAM2 box-prompted segmentation…")
            with torch.no_grad():
                self.predictor.set_image(image_np)
                for i, (x, y, w, h) in enumerate(boxes):
                    box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
                    pred_masks, scores, _ = self.predictor.predict(box=box_xyxy, multimask_output=False)
                    masks.append(pred_masks[0].astype(bool))
                    self.report_progress(0.1 + 0.9 * (i + 1) / len(boxes), f"Segmenting box {i + 1}/{len(boxes)}…")
            self.report_progress(1.0, "Done")
            return {"masks": masks}

        self.report_progress(0.05, f"Segmenting {len(boxes)} box(es) (panorama-scale, per-box crop)…")
        with torch.no_grad():
            for i, (x, y, w, h) in enumerate(boxes):
                mx, my = w * _BOX_CROP_MARGIN_FRAC, h * _BOX_CROP_MARGIN_FRAC
                cx0, cy0 = max(0, int(x - mx)), max(0, int(y - my))
                cx1, cy1 = min(img_w, int(x + w + mx)), min(img_h, int(y + h + my))
                crop_np = image_np[cy0:cy1, cx0:cx1]
                self.predictor.set_image(crop_np)
                box_local = np.array([x - cx0, y - cy0, x - cx0 + w, y - cy0 + h], dtype=np.float32)
                pred_masks, scores, _ = self.predictor.predict(box=box_local, multimask_output=False)
                full_mask = np.zeros((img_h, img_w), dtype=bool)
                full_mask[cy0:cy1, cx0:cx1] = pred_masks[0].astype(bool)
                masks.append(full_mask)
                self.report_progress(0.05 + 0.9 * (i + 1) / len(boxes), f"Segmenting box {i + 1}/{len(boxes)}…")
        self.report_progress(1.0, "Done")
        return {"masks": masks}


if __name__ == "__main__":
    ImageSegServer.run()
