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
                _paste_tiles(
                    [label_canvas, confidence_canvas, runnerup_canvas, runnerup_confidence_canvas],
                    [tile_label, tile_confidence, tile_runnerup, tile_runnerup_confidence],
                    weight_canvas, tile_weight, tile,
                )

            done = min(len(tiles), batch_start + len(batch))
            self.report_progress(0.05 + 0.85 * done / len(tiles), f"Segmenting tiles… ({done}/{len(tiles)})")

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
