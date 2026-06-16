from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from remote_connection.remote_server import RemoteServer


_MODEL_ID = "nvidia/segformer-b5-finetuned-ade-512-512"
# Maximum resolution to process in a single pass.  The panorama is resized to
# fit within this width while preserving aspect ratio before inference.
_MAX_WIDTH = 2048


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

        # Resize to a manageable width before inference.
        if orig_w > _MAX_WIDTH:
            scale = _MAX_WIDTH / orig_w
            proc_w = _MAX_WIDTH
            proc_h = max(1, int(orig_h * scale))
            proc_img = image.convert("RGB").resize((proc_w, proc_h), Image.LANCZOS)
        else:
            proc_img = image.convert("RGB")
            proc_w, proc_h = orig_w, orig_h

        self.report_progress(0.1, "Running SegFormer…")

        inputs = self.processor(images=proc_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Upsample logits to processing resolution, then back to original.
        logits = outputs.logits  # (1, num_classes, H/4, W/4)
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=(proc_h, proc_w),
            mode="bilinear",
            align_corners=False,
        )
        pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int16)

        # Resize label map to original panorama resolution.
        if orig_w != proc_w or orig_h != proc_h:
            pred_pil = Image.fromarray(pred.astype(np.uint8) if pred.max() < 256 else pred.astype(np.int16))
            pred_pil = pred_pil.resize((orig_w, orig_h), Image.NEAREST)
            pred = np.array(pred_pil).astype(np.int16)

        self.report_progress(0.9, "Done")

        id2label = self.model.config.id2label
        return {
            "label_map": pred.tolist(),
            "id2label": {str(k): v for k, v in id2label.items()},
            "width": orig_w,
            "height": orig_h,
        }


if __name__ == "__main__":
    PanoramaSegmentationServer.run()
