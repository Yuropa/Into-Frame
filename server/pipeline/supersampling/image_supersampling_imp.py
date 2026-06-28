from path_utils import add_project_paths
add_project_paths()

import torch
import numpy as np
from pathlib import Path
from typing import Any
from PIL import Image

from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
from remote_connection.remote_server import RemoteServer

_TILE_SIZE = 512   # input pixels per tile side
_TILE_OVERLAP = 32 # input pixels of overlap between adjacent tiles


class SupersamplingServer(RemoteServer):
    def setup(self):
        self.processor = Swin2SRImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
        self.model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64").to(self.device)

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "supersample":
            return self._supersample(input)
        raise ValueError(f"Unknown action: {action}")

    def _supersample(self, image: Image.Image) -> Image.Image:
        W, H = image.size

        # Small images fit in one shot; only tile when the image exceeds the tile size.
        if W <= _TILE_SIZE and H <= _TILE_SIZE:
            self.report_progress(0.1, "Running supersampling…")
            result = self._run_tile(image)
            self.report_progress(1.0, "Done")
            return result

        # Tiled supersampling: split into overlapping tiles, blend with a per-tile
        # Hanning window so seams are invisible, then reassemble.
        scale = 2
        out_W, out_H = W * scale, H * scale
        out_arr     = np.zeros((out_H, out_W, 3), dtype=np.float32)
        weight_arr  = np.zeros((out_H, out_W, 1), dtype=np.float32)

        img_arr = np.array(image.convert("RGB"))
        step    = _TILE_SIZE - _TILE_OVERLAP

        def tile_starts(length: int) -> list[int]:
            starts = list(range(0, length - _TILE_SIZE + 1, step))
            last   = length - _TILE_SIZE
            if not starts or starts[-1] < last:
                starts.append(last)
            return starts

        ys = tile_starts(H)
        xs = tile_starts(W)
        total = len(ys) * len(xs)
        done  = 0

        # Precompute the Hanning blend mask once (all tiles are the same size).
        wy = np.hanning(_TILE_SIZE * scale).astype(np.float32)
        wx = np.hanning(_TILE_SIZE * scale).astype(np.float32)
        blend_mask = (wy[:, None] * wx[None, :])[:, :, None]  # (T*2, T*2, 1)

        for y in ys:
            for x in xs:
                tile_pil = Image.fromarray(img_arr[y : y + _TILE_SIZE, x : x + _TILE_SIZE])
                tile_out = self._run_tile(tile_pil)  # PIL, size (T*2, T*2)
                tile_arr = np.array(tile_out).astype(np.float32) / 255.0

                oy, ox = y * scale, x * scale
                th, tw = tile_arr.shape[:2]
                out_arr   [oy : oy + th, ox : ox + tw] += tile_arr * blend_mask[:th, :tw]
                weight_arr[oy : oy + th, ox : ox + tw] += blend_mask[:th, :tw]

                done += 1
                self.report_progress(0.1 + 0.85 * done / total, f"Tile {done}/{total}…")

        result = np.clip(out_arr / np.maximum(weight_arr, 1e-6), 0.0, 1.0)
        result = (result * 255).astype(np.uint8)
        self.report_progress(1.0, "Done")
        return Image.fromarray(result)

    def _run_tile(self, tile: Image.Image) -> Image.Image:
        inputs = self.processor(tile, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs).reconstruction
        out = output.squeeze().cpu().clamp(0, 1).numpy()
        out = (np.transpose(out, (1, 2, 0)) * 255).astype(np.uint8)
        return Image.fromarray(out)


if __name__ == "__main__":
    SupersamplingServer.run()
