from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from remote_connection.remote_server import RemoteServer
from util.panorama_tiling import build_tiles, extract_tile, to_panorama_box, is_edge_truncated
from util.instance_merge import merge_tiled_instances, wrap_aware_box_iou


_MODEL_ID = "IDEA-Research/grounding-dino-base"
_BOX_THRESHOLD = 0.35

# Grounding DINO's own processor default (shortest_edge cap, well under its
# ~1333px longest-side cap for a square tile) -- a tile this size needs no
# resizing by the processor. Segmenting a whole multi-thousand-pixel-wide
# equirectangular panorama in one pass would downsample it well past this,
# likely missing small objects -- same reasoning as image_segmentation_imp.py's
# SAM2 tiling.
_TILE_SIZE = 800
_TILE_OVERLAP_FRAC = 0.25
_TILING_MIN_SIZE = int(_TILE_SIZE * (1 + _TILE_OVERLAP_FRAC))
_TRUNCATION_MARGIN_PX = 12
# Grounding DINO only gives boxes (no masks), so box IoU is both the cheap
# pre-filter and the actual merge decision -- no separate coarse/fine stages
# like SAM2's mask-based merge.
_MERGE_BOX_IOU = 0.4

# Floor for the "verify" action's own post-processing, well below _BOX_THRESHOLD --
# that action decides on score and coverage jointly (see _verify) and so needs the
# low-scoring boxes visible rather than cut off before it can weigh them.
_VERIFY_FLOOR = 0.05

# Square canvas every crop is letterboxed onto before verification.
#
# Not cosmetic -- without it the model raises. Grounding DINO's two-stage decoder
# selects a fixed `topk` (900) proposals from the flattened multi-scale feature
# maps, and the processor sizes an image by shortest_edge=800 CAPPED at
# longest_edge=1333. A crop with an extreme aspect ratio hits the cap first and
# comes out tiny on its short axis: measured on the Rainier capture, crop_343 is
# 1024x11 (93:1) and lands at roughly 1333x14, whose feature maps hold nowhere near
# 900 positions, and torch.topk fails with "selected index k out of range". That
# killed the whole stage mid-run. The detection path above never sees it because it
# only ever feeds ~800px square tiles.
#
# Letterboxing onto a square fixes the shape problem at the source: aspect ratio is
# preserved, the short axis is never squeezed by the long one, and every crop
# presents the model the same geometry regardless of how the segmenter cut it.
_VERIFY_CANVAS = 800
# Crops smaller than this on either axis after all scaling are not a detection
# question -- there is nothing in them to localize. Reported unverified rather than
# guessed at.
_VERIFY_MIN_SIDE_PX = 4


def _letterbox_square(image: PILImage.Image, size: int) -> tuple[PILImage.Image, tuple[int, int, int, int]]:
    """Scale `image` to fit a `size`x`size` canvas, preserving aspect, centred.

    Returns (canvas, (x, y, w, h)) where the tuple is where the image landed, in
    canvas pixels -- callers measuring how much of the CROP a detected box covers
    need that rect, not the canvas.

    Padding is the image's own mean colour rather than black. A black surround puts
    a hard high-contrast rectangle round the subject, which an object detector will
    happily find edges on; the mean blends it. Same reasoning as
    flatten_alpha_with_mean_fill's, applied to the outside instead of the holes.
    """
    w, h = image.size
    if w <= 0 or h <= 0:
        return PILImage.new("RGB", (size, size)), (0, 0, 0, 0)

    scale = min(size / w, size / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((new_w, new_h), PILImage.LANCZOS)

    mean = tuple(int(c) for c in np.array(resized).reshape(-1, 3).mean(axis=0))
    canvas = PILImage.new("RGB", (size, size), mean)
    ox, oy = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, (ox, oy, new_w, new_h)


class GroundingDinoServer(RemoteServer):
    def setup(self):
        self.processor = AutoProcessor.from_pretrained(_MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(_MODEL_ID).to(self.device)
        self.model.eval()
        print(f"Grounding DINO loaded ({_MODEL_ID}) on {self.device}")

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "detect":
            return self._detect(input["image"], input["tags"])
        if action == "verify":
            return self._verify(input["crops"], input["threshold"], input["min_box_fraction"])
        raise ValueError(f"Unknown action: {action}")

    def _verify(
        self, crops: list[dict], threshold: float, min_box_fraction: float
    ) -> dict:
        """Answer "does `label` actually appear in this crop" for each (image, label).

        Independent corroboration for a label some other model proposed. Grounding
        DINO is the right second opinion here specifically because it has to LOCALIZE
        what it is asked about: a CLIP-style global embedding always returns a ranking
        over whatever labels it was given, and on a capture with no signal that ranking
        is near-uniform noise that still has a winner. This model can decline -- asked
        for "lighthouse" on a crop of empty sky it returns no box at all.

        Each crop is scored against ONLY its own candidate label, not the scene's whole
        tag set. A single-phrase prompt is the actual question ("is this a tree"); a
        multi-label prompt asks a different and easier one ("which of these is it
        most"), which is the failure mode being corrected for.

        A detection counts only if it covers at least `min_box_fraction` of the crop:
        the crop is already tight around one segmented instance, so a real match fills
        much of it, while a small high-scoring box usually means the label was found in
        background that came along inside the mask's bounding rectangle.

        Returns {"verified": [ {label, score, box_fraction, verified}, ... ]} in input
        order -- the scores are kept for the debug record even when the answer is no.
        """
        out: list[dict] = []
        for i, entry in enumerate(crops):
            image, label = entry["image"], entry["label"]
            rgb = image.convert("RGB")
            # Grounding DINO's prompt format: lowercase phrase, trailing period.
            prompt = str(label).lower().strip().replace("_", " ") + "."
            best_score, best_fraction = 0.0, 0.0
            note = None

            canvas, crop_rect = _letterbox_square(rgb, _VERIFY_CANVAS)
            cx0, cy0, cw, ch = crop_rect
            if min(cw, ch) < _VERIFY_MIN_SIDE_PX:
                note = f"crop too small to localize ({rgb.size[0]}x{rgb.size[1]})"
            else:
                crop_area = float(cw * ch)
                # Below this stage's own _BOX_THRESHOLD on purpose: the acceptance
                # rule here is score AND coverage together, so it has to see the
                # boxes a score-only cut would already have discarded.
                for det in self._detect_one(canvas, prompt, threshold=_VERIFY_FLOOR):
                    bx, by, bw, bh = det["box"]
                    # Clip to the pasted crop -- coverage has to be measured against
                    # the crop, not the canvas, or the letterbox padding would count
                    # as area the label failed to fill and every fraction would be
                    # deflated by however square the crop happened to be.
                    ix0, iy0 = max(bx, cx0), max(by, cy0)
                    ix1, iy1 = min(bx + bw, cx0 + cw), min(by + bh, cy0 + ch)
                    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                    fraction = inter / crop_area if crop_area > 0 else 0.0
                    if det["score"] > best_score:
                        best_score, best_fraction = det["score"], fraction

            result = {
                "label": label,
                "score": best_score,
                "box_fraction": best_fraction,
                "verified": (
                    note is None
                    and best_score >= threshold
                    and best_fraction >= min_box_fraction
                ),
            }
            if note is not None:
                result["note"] = note
            out.append(result)
            if (i + 1) % 16 == 0 or i + 1 == len(crops):
                self.report_progress((i + 1) / max(len(crops), 1), f"Verifying… ({i + 1}/{len(crops)})")
        return {"verified": out}

    def _detect_one(
        self, image: PILImage.Image, prompt: str, threshold: float | None = None
    ) -> list[dict]:
        """Run Grounding DINO on a single (already tile-sized-or-smaller) image,
        returning detections with box in that image's own local pixel space.

        `threshold` overrides the detection default for callers that apply their own
        acceptance rule downstream and need the scores below it -- see _verify, which
        decides on score AND coverage together and so must see boxes this stage's own
        threshold would have dropped."""
        rgb = image.convert("RGB")
        inputs = self.processor(images=rgb, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=_BOX_THRESHOLD if threshold is None else threshold,
            target_sizes=[rgb.size[::-1]],  # (height, width)
        )[0]

        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "box": [x1, y1, x2 - x1, y2 - y1],  # absolute xywh
                "score": float(score),
                "label": str(label),
            })
        return detections

    def _detect(self, image: PILImage.Image, tags: list[str]) -> dict:
        # Grounding DINO expects ". "-separated labels with a trailing period
        prompt = ". ".join(t.lower().strip() for t in tags if t.strip()) + "."
        orig_w, orig_h = image.size

        if max(orig_w, orig_h) <= _TILING_MIN_SIZE:
            self.report_progress(0.1, "Preparing inputs…")
            self.report_progress(0.3, "Running Grounding DINO…")
            detections = self._detect_one(image, prompt)
            self.report_progress(1.0, "Done")
            return {"detections": detections}

        return self._detect_tiled(image, prompt, orig_w, orig_h)

    def _detect_tiled(self, image: PILImage.Image, prompt: str, pano_w: int, pano_h: int) -> dict:
        """Panorama-scale path: run Grounding DINO per tile, translate each
        surviving (non-edge-truncated) detection into panorama pixel space,
        then merge detections that are really the same object seen in two
        overlapping tiles (see util/instance_merge.py) -- the same reasoning
        as image_segmentation_imp.py's SAM2 tiling, minus mask comparison
        (Grounding DINO only gives boxes, so box IoU is both the pre-filter
        and the merge decision)."""
        image_np = np.array(image.convert("RGB"))
        tiles = build_tiles(pano_w, pano_h, _TILE_SIZE, _TILE_OVERLAP_FRAC)
        self.report_progress(0.05, f"Detecting {len(tiles)} tile(s)…")

        dets: list[dict] = []
        for i, tile in enumerate(tiles):
            crop = PILImage.fromarray(extract_tile(image_np, tile))
            for det in self._detect_one(crop, prompt):
                if is_edge_truncated(det["box"], tile, pano_h, margin_px=_TRUNCATION_MARGIN_PX):
                    continue
                dets.append({
                    "box": to_panorama_box(det["box"], tile, pano_w),
                    "score": det["score"],
                    "label": det["label"],
                })
            self.report_progress(0.05 + 0.85 * (i + 1) / len(tiles), f"Detecting tiles… ({i + 1}/{len(tiles)})")

        merged = merge_tiled_instances(
            dets,
            overlap_fn=lambda a, b: wrap_aware_box_iou(a["box"], b["box"], pano_w),
            threshold=_MERGE_BOX_IOU,
            pano_w=pano_w,
        )
        self.report_progress(1.0, "Done")
        return {"detections": merged}


if __name__ == "__main__":
    GroundingDinoServer.run()
