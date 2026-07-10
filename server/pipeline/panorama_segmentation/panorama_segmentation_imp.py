from path_utils import add_project_paths
add_project_paths()

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from remote_connection.remote_server import RemoteServer


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


@dataclass(frozen=True)
class _Tile:
    x0: int  # may exceed the panorama width for tiles straddling the wrap seam — resolved via modulo on use
    y0: int  # always within [0, height - h]
    w: int
    h: int


def _axis_starts(extent: int, tile: int, overlap_frac: float, wrap: bool) -> list[int]:
    """Tile start positions covering `extent`, evenly spaced with the given overlap."""
    if extent <= tile:
        return [0]

    stride = max(1, int(tile * (1.0 - overlap_frac)))
    starts = list(range(0, extent, stride))

    if wrap:
        # Equirectangular panoramas wrap horizontally — column 0 and column
        # extent-1 are the same seam in the final 360° view. Add a tile
        # explicitly centred on that seam so it's never left straddled
        # between two tile edges no matter how the stride above lands.
        starts.append(extent - tile // 2)
    else:
        # No vertical wrap (top/bottom are the zenith/nadir poles, not a
        # seam) — clamp so the last tile's far edge lands exactly on the
        # image border instead of overshooting it.
        starts = [min(s, extent - tile) for s in starts]
        starts.append(extent - tile)

    # De-dup while preserving order.
    seen: set[int] = set()
    deduped = []
    for s in starts:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def _build_tiles(width: int, height: int, tile_size: int, overlap_frac: float) -> list[_Tile]:
    tw = min(tile_size, width)
    th = min(tile_size, height)
    x_starts = _axis_starts(width, tw, overlap_frac, wrap=True)
    y_starts = _axis_starts(height, th, overlap_frac, wrap=False)
    return [_Tile(x0=x, y0=y, w=tw, h=th) for y in y_starts for x in x_starts]


def _extract_tile(img_arr: np.ndarray, tile: _Tile) -> np.ndarray:
    width = img_arr.shape[1]
    cols = np.arange(tile.x0, tile.x0 + tile.w) % width
    rows = np.arange(tile.y0, tile.y0 + tile.h)
    return img_arr[rows][:, cols]


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


def _paste_tile(
    label_canvas: np.ndarray,
    weight_canvas: np.ndarray,
    tile_label: np.ndarray,
    tile_weight: np.ndarray,
    tile: _Tile,
) -> None:
    width = label_canvas.shape[1]
    cols = np.arange(tile.x0, tile.x0 + tile.w) % width
    rows = np.arange(tile.y0, tile.y0 + tile.h)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")

    current_weight = weight_canvas[row_grid, col_grid]
    better = tile_weight > current_weight
    label_canvas[row_grid, col_grid] = np.where(better, tile_label, label_canvas[row_grid, col_grid])
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
                tile_label = tile_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int16)
                tile_weight = _feather_window(tile.h, tile.w)
                _paste_tile(label_canvas, weight_canvas, tile_label, tile_weight, tile)

            done = min(len(tiles), batch_start + len(batch))
            self.report_progress(0.05 + 0.85 * done / len(tiles), f"Segmenting tiles… ({done}/{len(tiles)})")

        self.report_progress(0.95, "Done")

        id2label = self.model.config.id2label
        return {
            "label_map": label_canvas.tolist(),
            "id2label": {str(k): v for k, v in id2label.items()},
            "width": orig_w,
            "height": orig_h,
        }


if __name__ == "__main__":
    PanoramaSegmentationServer.run()
