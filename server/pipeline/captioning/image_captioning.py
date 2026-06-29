from util.image_utils import Image, make_context_composite
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image as PILImage
import numpy as np
import torch


class ImageCaptioning:
    _MODEL_ID = "microsoft/Florence-2-large"
    # <MORE_DETAILED_CAPTION> produces a full sentence with colour, shape, context.
    # <DETAILED_CAPTION> is slightly shorter; <CAPTION> is one-liner.
    _TASK = "<MORE_DETAILED_CAPTION>"

    def __init__(self, device):
        self.device = device

        self.processor = AutoProcessor.from_pretrained(
            self._MODEL_ID, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self._MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16
        ).to(device)
        self.model.eval()

    @classmethod
    def model_names(cls) -> list[str]:
        return [cls._MODEL_ID]

    def _to_rgb(self, image: Image) -> PILImage.Image:
        """Convert to RGB, filling transparent pixels with the mean opaque colour."""
        pil = image.image
        if pil.mode != "RGBA":
            return image.rgb()
        arr = np.array(pil).astype(np.float32)
        alpha = arr[..., 3:4] / 255.0
        rgb = arr[..., :3]
        opaque = arr[..., 3] > 128
        mean_color = rgb[opaque].mean(axis=0) if opaque.any() else np.array([128.0, 128.0, 128.0])
        background = np.ones_like(rgb) * mean_color
        composited = (rgb * alpha + background * (1.0 - alpha)).astype(np.uint8)
        return PILImage.fromarray(composited, mode="RGB")

    def caption(
        self,
        input: Image,
        scene_image: Image | None = None,
        box: list[float] | None = None,
        prompt: str = "",
    ) -> str:
        """Generate a caption for input.

        scene_image and box are accepted for API compatibility but Florence-2
        doesn't use a freeform text prompt — the task token drives output style.
        If scene_image is provided a context composite is still used so the model
        sees the object in context.
        """
        if scene_image is not None:
            pil_input = make_context_composite(self._to_rgb(input), scene_image.rgb(), box)
        else:
            pil_input = self._to_rgb(input)

        inputs = self.processor(
            text=self._TASK,
            images=pil_input,
            return_tensors="pt",
        ).to(self.device, torch.float16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
            )

        raw = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            raw,
            task=self._TASK,
            image_size=(pil_input.width, pil_input.height),
        )
        return parsed.get(self._TASK, "").strip()
