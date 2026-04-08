from PIL import Image as PILImage
from util.image_utils import Image
from transformers import Sam3Processor, Sam3Model
import torch
from typing import Generator

class SegmentationResult:
    def __init__(self, results):
        self.masks = results['masks']
        self.boxes = results['boxes']
        self.scores = results['scores']

    def masked_images(self, source: Image) -> Generator[Image, None, None]:
        """Yield a masked RGBA crop for each segmented object."""
        source_rgba = source.image.convert("RGBA")
        masks_np = self.masks.cpu().numpy()  # (N, H, W) bool

        for mask in masks_np:
            # Black out every pixel outside the mask
            rgba = source_rgba.copy()
            alpha = PILImage.fromarray((mask * 255).astype(np.uint8), mode="L")
            rgba.putalpha(alpha)

            # Tight crop to the mask's bounding box
            bbox = alpha.getbbox()
            if bbox:
                rgba = rgba.crop(bbox)

            yield Image(rgba)

class ImageSeg:
    def __init__(self, device):
        self.device = device
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    def segment(self, input: Image, prompt: str= "all objects"):
        inputs = self.processor(images=input.image, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        return SegmentationResult(results)