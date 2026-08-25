from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from remote_connection.remote_server import RemoteServer
from util.panorama_tiling import Tile as _Tile, build_tiles as _build_tiles, extract_tile as _extract_tile


_MODEL_ID = "nvidia/segformer-b5-finetuned-ade-640-640"

# Tile size matches the model's native trained resolution (640×640) exactly,
# so a tile needs no resizing by the processor and the model sees genuine
# full-resolution pixels — unlike segmenting the whole (multi-thousand-pixel-
# wide) panorama in one pass, where the processor's forced square resize
# squashes it down to 640×640 and the ~160×160 output logits get stretched
# back over the whole panorama, blurring anything smaller than a few dozen
# pixels (tree branches, ridgelines) into whatever's next to it.
_TILE_SIZE = 640
# Fraction of a tile's size that neighbouring tiles overlap by. Needs to be
# wide enough that the feather window (see _feather_window) has room to taper
# before the next tile takes over, or tile-grid seams show up in the result.
_TILE_OVERLAP_FRAC = 0.25
# Tiles per forward pass. Bounds peak GPU memory; a panorama with more tiles
# than this is simply processed in multiple batches.
_MAX_BATCH = 8

# _Tile/_build_tiles/_extract_tile now live in util.panorama_tiling (shared with
# SAM2/Grounding DINO panorama tiling -- see image_segmentation_imp.py,
# grounding_dino_imp.py); re-imported above under their original local names so
# nothing below this line needs to change.


def _feather_window(h: int, w: int) -> np.ndarray:
    """
    Smooth 2D weight peaking at the tile's centre and tapering toward its
    edges, so where tiles overlap, each pixel is taken from whichever tile is
    most confident about it (closest to that tile's own centre) rather than
    handed off at a hard tile-grid boundary.
    """
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    window = np.outer(wy, wx).astype(np.float32)
    # np.hanning goes to exactly 0 at the edges — floor it so a tile can still
    # win at its own edge pixels where no neighbouring tile reaches at all
    # (the outermost rows/columns of the whole panorama).
    return np.clip(window, 0.05, 1.0)


def _paste_tiles(
    canvases: list[np.ndarray],
    tile_values: list[np.ndarray],
    weight_canvas: np.ndarray,
    tile_weight: np.ndarray,
    tile: _Tile,
) -> None:
    """
    Paste one or more per-tile value canvases (label, confidence, runner-up
    label, ...) into their matching full-resolution canvases, all under the
    same per-pixel "does this tile beat what's already there" mask.

    Every canvas for a given tile must be pasted in one call here: `better`
    is computed from weight_canvas before any of it is mutated, and
    weight_canvas itself is only updated afterwards -- so all canvases stay
    mutually consistent about which tile won each pixel. Calling this
    against the same weight_canvas a second time for a different value would
    silently drop that value wherever an earlier call already won (the
    second call's `better` would compare against weight_canvas already
    holding the winning tile's own weight).
    """
    width = weight_canvas.shape[1]
    cols = np.arange(tile.x0, tile.x0 + tile.w) % width
    rows = np.arange(tile.y0, tile.y0 + tile.h)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")

    current_weight = weight_canvas[row_grid, col_grid]
    better = tile_weight > current_weight

    for canvas, tile_value in zip(canvases, tile_values):
        canvas[row_grid, col_grid] = np.where(better, tile_value, canvas[row_grid, col_grid])

    weight_canvas[row_grid, col_grid] = np.where(better, tile_weight, current_weight)


# Blend overlapping tiles by accumulating weighted class probabilities and taking
# the argmax once, at the end, instead of letting the highest-feather tile win each
# pixel outright.
#
# Winner-take-all does not remove a tile seam, it MOVES it. The feather weight is
# highest at a tile's own centre, so the pixel where the winner flips from tile A to
# tile B is the midpoint between their two centres -- a perfectly straight vertical
# line, at which the label changes discontinuously wherever the two tiles disagree.
# Measured on the 2026-08-25 run at _TILE_SIZE 640 / _TILE_OVERLAP_FRAC 0.25 (stride
# 480, centres at 320 + 480k, midpoints at 560 + 480k): straight label edges in the
# nadir half sit at columns 560, 1040, 1520, 2000, 2480 and 3440 on Iceland, Shark
# Fin and Irises alike, and Rainier's single seam is at 3440. Every one of those is
# a feather crossover, not a region boundary.
#
# Accumulating instead makes the handover continuous: near the midpoint both tiles
# contribute about equally, so the winning class changes where the EVIDENCE changes
# rather than where the geometry does.
_BLEND_TILE_PROBABILITIES = True
# Ceiling on how many distinct classes the accumulator will track. Each one costs a
# full-resolution float32 canvas (~34 MB at 2048x4096), and only classes that reach
# some tile's top-2 are ever allocated -- measured at 21-34 distinct classes per
# capture, so this is roughly 4x headroom rather than a limit expected to bind. Past
# it, further new classes fall back to competing on their own weighted score alone,
# which is the old behaviour for those classes only.
_MAX_ACCUMULATED_CLASSES = 128


class _ProbabilityAccumulator:
    """
    Per-class weighted-probability canvases, argmaxed once when every tile is in.

    Sparse by class: SegFormer's ADE20K head has 150 classes and a dense
    (H, W, 150) float32 accumulator is 5 GB at panorama resolution, but a single
    capture only ever puts a couple of dozen of them in a tile's top-2. Canvases are
    therefore allocated lazily, on first sight of a class.

    Only each tile's top-2 classes are accumulated, not its full softmax -- that is
    what the tile loop already computes, and a class that never places in any tile's
    top two cannot win a blended argmax either. The consequence is that a class
    sitting consistently third contributes nothing, which is the intended tradeoff:
    it could not have been the answer anyway.
    """

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.scores: dict[int, np.ndarray] = {}
        # Sum of the feather weights of every tile covering each pixel. Dividing the
        # accumulated scores by it turns them back into a weighted MEAN of the
        # per-tile softmax probabilities, on the same 0..1 scale the old top-1 value
        # had. That scale is load-bearing: downstream reads confidence and runner-up
        # confidence as softmax probabilities and thresholds their difference as a
        # margin (see ambiguity_strategy_for_label), so rescaling them -- e.g.
        # normalising the pair to sum to 1 -- would silently move every one of those
        # thresholds.
        self.weight = np.zeros((height, width), dtype=np.float32)
        self.overflow = 0

    def add(self, tile: _Tile, tile_weight: np.ndarray,
            ids: list[np.ndarray], vals: list[np.ndarray]) -> None:
        cols = np.arange(tile.x0, tile.x0 + tile.w) % self.width
        rows = np.arange(tile.y0, tile.y0 + tile.h)
        row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
        np.add.at(self.weight, (row_grid, col_grid), tile_weight.astype(np.float32))

        for tile_ids, tile_vals in zip(ids, vals):
            weighted = (tile_vals * tile_weight).astype(np.float32)
            for class_id in np.unique(tile_ids):
                key = int(class_id)
                canvas = self.scores.get(key)
                if canvas is None:
                    if len(self.scores) >= _MAX_ACCUMULATED_CLASSES:
                        self.overflow += 1
                        continue
                    canvas = np.zeros((self.height, self.width), dtype=np.float32)
                    self.scores[key] = canvas
                hit = tile_ids == class_id
                # np.add.at rather than plain += : a tile straddling the wrap seam
                # has repeated column indices, and fancy-index += would apply only
                # the last write instead of summing them.
                np.add.at(canvas, (row_grid[hit], col_grid[hit]), weighted[hit])

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(label, confidence, runner-up label, runner-up confidence)."""
        # Running maxima over the per-class canvases rather than np.stack + argmax:
        # stacking 34 classes at 2048x4096 is a 1.1 GB temporary, and masking the
        # winner out for the second pass copies it again. Two sequential sweeps cost
        # the same arithmetic and hold only a handful of (H, W) arrays at once --
        # measured 4.37 GB -> 1.4 GB peak RSS on a full-resolution panorama, which
        # matters because this runs in the same process as the segmentation model.
        class_ids = sorted(self.scores)
        lookup = np.asarray(class_ids, dtype=np.int16)
        shape = (self.height, self.width)

        best_score = np.full(shape, -np.inf, dtype=np.float32)
        best = np.zeros(shape, dtype=np.int16)
        for idx, class_id in enumerate(class_ids):
            canvas = self.scores[class_id]
            wins = canvas > best_score
            best_score = np.where(wins, canvas, best_score)
            best = np.where(wins, np.int16(idx), best)

        second_score = np.full(shape, -np.inf, dtype=np.float32)
        second = np.zeros(shape, dtype=np.int16)
        for idx, class_id in enumerate(class_ids):
            # Skip only the pixels this class actually won, not the whole plane --
            # a class that is the winner somewhere is still a legitimate runner-up
            # everywhere else.
            canvas = np.where(best == idx, -np.inf, self.scores[class_id])
            wins = canvas > second_score
            second_score = np.where(wins, canvas, second_score)
            second = np.where(wins, np.int16(idx), second)

        best_score[~np.isfinite(best_score)] = 0.0
        second_score[~np.isfinite(second_score)] = 0.0

        # Back to the per-tile softmax scale -- see self.weight.
        coverage = np.where(self.weight > 0, self.weight, 1.0)
        return (
            lookup[best],
            (best_score / coverage).astype(np.float32),
            lookup[second],
            (second_score / coverage).astype(np.float32),
        )


class PanoramaSegmentationServer(RemoteServer):
    def setup(self):
        self.processor = SegformerImageProcessor.from_pretrained(_MODEL_ID)
        self.model = SegformerForSemanticSegmentation.from_pretrained(_MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "segment":
            return self._segment(input)
        raise ValueError(f"Unknown action: {action}")

    def _segment(self, image: Image.Image) -> dict:
        orig_w, orig_h = image.size
        img_arr = np.array(image.convert("RGB"))

        tiles = _build_tiles(orig_w, orig_h, _TILE_SIZE, _TILE_OVERLAP_FRAC)
        self.report_progress(0.05, f"Segmenting {len(tiles)} tile(s)…")

        label_canvas = np.zeros((orig_h, orig_w), dtype=np.int16)
        # Top-1/top-2 softmax confidence and top-2 (runner-up) class id per
        # pixel -- kept alongside the argmax label so downstream stages can
        # tell a confident, unambiguous call apart from a close one (their
        # difference is the margin), and know what the model's second-best
        # guess was (see PanoramaRegionStage / panorama_region_result's
        # ambiguity_strategy_for_label for how this is used to catch e.g.
        # "wall" called on what's actually a cliff face, or "tree" called on
        # dark, mottled rock texture).
        confidence_canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
        runnerup_canvas = np.zeros((orig_h, orig_w), dtype=np.int16)
        runnerup_confidence_canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
        weight_canvas = np.full((orig_h, orig_w), -1.0, dtype=np.float32)
        # See _BLEND_TILE_PROBABILITIES. None -> the winner-take-all canvases above
        # are used exactly as before, so the flag bisects cleanly.
        accumulator = _ProbabilityAccumulator(orig_h, orig_w) if _BLEND_TILE_PROBABILITIES else None

        for batch_start in range(0, len(tiles), _MAX_BATCH):
            batch = tiles[batch_start: batch_start + _MAX_BATCH]
            crops = [Image.fromarray(_extract_tile(img_arr, t)) for t in batch]

            inputs = self.processor(images=crops, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits  # (B, num_classes, h/4, w/4)

            for i, tile in enumerate(batch):
                tile_logits = torch.nn.functional.interpolate(
                    logits[i: i + 1],
                    size=(tile.h, tile.w),
                    mode="bilinear",
                    align_corners=False,
                )
                # top-2 subsumes plain argmax: index 0 of the topk is exactly
                # the argmax label, so this is no extra work over what the
                # label alone already cost.
                probs = torch.softmax(tile_logits, dim=1)
                top2_val, top2_idx = probs.topk(2, dim=1)
                tile_label = top2_idx[:, 0].squeeze(0).cpu().numpy().astype(np.int16)
                tile_confidence = top2_val[:, 0].squeeze(0).cpu().numpy().astype(np.float32)
                tile_runnerup = top2_idx[:, 1].squeeze(0).cpu().numpy().astype(np.int16)
                tile_runnerup_confidence = top2_val[:, 1].squeeze(0).cpu().numpy().astype(np.float32)
                tile_weight = _feather_window(tile.h, tile.w)
                if accumulator is not None:
                    accumulator.add(
                        tile, tile_weight,
                        [tile_label, tile_runnerup],
                        [tile_confidence, tile_runnerup_confidence],
                    )
                else:
                    _paste_tiles(
                        [label_canvas, confidence_canvas, runnerup_canvas, runnerup_confidence_canvas],
                        [tile_label, tile_confidence, tile_runnerup, tile_runnerup_confidence],
                        weight_canvas, tile_weight, tile,
                    )

            done = min(len(tiles), batch_start + len(batch))
            self.report_progress(0.05 + 0.85 * done / len(tiles), f"Segmenting tiles… ({done}/{len(tiles)})")

        if accumulator is not None:
            (label_canvas, confidence_canvas,
             runnerup_canvas, runnerup_confidence_canvas) = accumulator.finalize()
            if accumulator.overflow:
                print(
                    f"  tile blending: {accumulator.overflow} class sighting(s) past the "
                    f"{_MAX_ACCUMULATED_CLASSES}-class accumulator ceiling were dropped"
                )

        self.report_progress(0.95, "Done")

        id2label = self.model.config.id2label
        return {
            "label_map": label_canvas.tolist(),
            # Raw ndarrays (not .tolist()): RemoteObject.encode auto-spills
            # these to a temp .npy file instead of inlining them as JSON,
            # which matters at full panorama resolution.
            "confidence_map": confidence_canvas,
            "runnerup_label_map": runnerup_canvas,
            "runnerup_confidence_map": runnerup_confidence_canvas,
            "id2label": {str(k): v for k, v in id2label.items()},
            "width": orig_w,
            "height": orig_h,
        }


if __name__ == "__main__":
    PanoramaSegmentationServer.run()
