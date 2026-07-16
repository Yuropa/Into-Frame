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
            min_mask_region_area=50,
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
        image_np = np.array(image.convert("RGB"))
        self.report_progress(0.1, "Running SAM2 segmentation…")
        with torch.no_grad():
            results = self.mask_generator.generate(image_np)
        self.report_progress(1.0, "Done")
        return {
            "masks": [r["segmentation"] for r in results],
            "boxes": [list(r["bbox"]) for r in results],
            "scores": [float(r["predicted_iou"]) for r in results],
        }

    def _segment_boxes(self, image: Image.Image, boxes: list) -> dict:
        """Box-prompted SAM2: one mask per given [x, y, w, h] box, in the same image."""
        image_np = np.array(image.convert("RGB"))
        self.report_progress(0.1, "Running SAM2 box-prompted segmentation…")
        masks = []
        with torch.no_grad():
            self.predictor.set_image(image_np)
            for i, (x, y, w, h) in enumerate(boxes):
                box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
                pred_masks, scores, _ = self.predictor.predict(box=box_xyxy, multimask_output=False)
                masks.append(pred_masks[0].astype(bool))
                self.report_progress(0.1 + 0.9 * (i + 1) / len(boxes), f"Segmenting box {i + 1}/{len(boxes)}…")
        self.report_progress(1.0, "Done")
        return {"masks": masks}


if __name__ == "__main__":
    ImageSegServer.run()
