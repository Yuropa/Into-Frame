from util.image_utils import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image as PILImage
import numpy as np
import torch

class ImageCaptioning:
    def __init__(self, device):
        self.device = device

        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    @classmethod
    def model_names(cls) -> list[str]:
        return ["Salesforce/blip-image-captioning-large"]

    def _to_rgb(self, image: Image) -> PILImage.Image:
        """Convert to RGB, filling transparent pixels with the mean opaque color.

        PIL's default RGBA→RGB fills alpha=0 areas with black, creating a dark
        silhouette of masked-out content (e.g. a tower hole in a sky crop) that
        causes BLIP to caption the hole instead of the actual content."""
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

    def caption(self, input: Image, prompt: str = ""):
        inputs = self.processor(self._to_rgb(input), prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs)

        result = self.processor.decode(out[0], skip_special_tokens=True)
        return result
