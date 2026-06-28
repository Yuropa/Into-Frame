from path_utils import add_project_paths
add_project_paths()

import torch
import numpy as np
from pathlib import Path
from typing import Any
from PIL import Image

from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
from remote_connection.remote_server import RemoteServer

_TILE_SIZE    = 512  # input pixels per tile side
_TILE_OVERLAP = 64   # input pixels of overlap on each edge that borders a neighbour


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

        if W <= _TILE_SIZE and H <= _TILE_SIZE:
            self.report_progress(0.1, "Running supersampling…")
            result = self._run_tile(image)
            self.report_progress(1.0, "Done")
            return result

        scale   = 2
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

        out_overlap = _TILE_OVERLAP * scale  # overlap in output pixels

        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                tile_pil = Image.fromarray(img_arr[y : y + _TILE_SIZE, x : x + _TILE_SIZE])
                tile_out = self._run_tile(tile_pil)
                tile_arr = np.array(tile_out).astype(np.float32) / 255.0

                th, tw = tile_arr.shape[:2]
                blend = self._blend_mask(
                    th, tw,
                    has_top    = i > 0,
                    has_bottom = i < len(ys) - 1,
                    has_left   = j > 0,
                    has_right  = j < len(xs) - 1,
                    overlap    = out_overlap,
                )

                oy, ox = y * scale, x * scale
                out_arr   [oy : oy + th, ox : ox + tw] += tile_arr * blend
                weight_arr[oy : oy + th, ox : ox + tw] += blend

                done += 1
                self.report_progress(0.1 + 0.85 * done / total, f"Tile {done}/{total}…")

        result = np.clip(out_arr / np.maximum(weight_arr, 1e-6), 0.0, 1.0)
        result = (result * 255).astype(np.uint8)
        self.report_progress(1.0, "Done")
        return Image.fromarray(result)

    @staticmethod
    def _blend_mask(
        h: int, w: int,
        has_top: bool, has_bottom: bool, has_left: bool, has_right: bool,
        overlap: int,
    ) -> np.ndarray:
        """
        Per-pixel blend weight for one tile.

        Weight is 1 everywhere except in the overlap strip on each edge that
        borders a neighbouring tile, where it ramps linearly from 0 (at the
        edge) to 1 (at the inner boundary of the overlap zone).

        Adjacent tiles share the same overlap strip with complementary ramps
        (A fades 1→0 while B fades 0→1), so their weights always sum to 1.
        """
        wy = np.ones(h, dtype=np.float32)
        wx = np.ones(w, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        if has_top:    wy[:overlap]  = ramp
        if has_bottom: wy[-overlap:] = ramp[::-1]
        if has_left:   wx[:overlap]  = ramp
        if has_right:  wx[-overlap:] = ramp[::-1]
        return (wy[:, None] * wx[None, :])[:, :, None]

    def _run_tile(self, tile: Image.Image) -> Image.Image:
        in_w, in_h = tile.size
        inputs = self.processor(tile, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs).reconstruction
        out = output.squeeze().cpu().clamp(0, 1).numpy()
        out = (np.transpose(out, (1, 2, 0)) * 255).astype(np.uint8)
        result = Image.fromarray(out)
        # The processor pads the input to a window-size multiple before inference,
        # so the output can be slightly larger than 2× the original tile size.
        # Crop back to the expected dimensions so tile placement stays exact.
        expected = (in_w * 2, in_h * 2)
        if result.size != expected:
            result = result.crop((0, 0, *expected))
        return result


if __name__ == "__main__":
    SupersamplingServer.run()
