from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

import torch
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from remote_connection.remote_server import RemoteServer


_MODEL_ID = "IDEA-Research/grounding-dino-base"
_BOX_THRESHOLD = 0.35
_TEXT_THRESHOLD = 0.25


class GroundingDinoServer(RemoteServer):
    def setup(self):
        self.processor = AutoProcessor.from_pretrained(_MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(_MODEL_ID).to(self.device)
        self.model.eval()
        print(f"Grounding DINO loaded ({_MODEL_ID}) on {self.device}")

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "detect":
            return self._detect(input["image"], input["tags"])
        raise ValueError(f"Unknown action: {action}")

    def _detect(self, image: PILImage.Image, tags: list[str]) -> dict:
        # Grounding DINO expects ". "-separated labels with a trailing period
        prompt = ". ".join(t.lower().strip() for t in tags if t.strip()) + "."

        self.report_progress(0.1, "Preparing inputs...")
        rgb = image.convert("RGB")
        inputs = self.processor(images=rgb, text=prompt, return_tensors="pt").to(self.device)

        self.report_progress(0.3, "Running Grounding DINO...")
        with torch.no_grad():
            outputs = self.model(**inputs)

        self.report_progress(0.8, "Post-processing...")
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=_BOX_THRESHOLD,
            text_threshold=_TEXT_THRESHOLD,
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

        self.report_progress(1.0, "Done")
        return {"detections": detections}


if __name__ == "__main__":
    GroundingDinoServer.run()
