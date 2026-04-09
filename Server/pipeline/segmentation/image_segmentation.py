from PIL import Image as PILImage
from util.image_utils import Image
from sam2.build_sam import build_sam2_hf
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
import torch
from typing import Generator

class CroppedImage:
    def __init__(self, image, box, mask, score) -> None:
        self.image = image
        self.box = box
        self.score = score
        self.mask = mask

class SegmentationResult:
    def __init__(self, results):
        # results is a list of dicts, not a single dict
        self.masks = [r['segmentation'] for r in results]   # list of (H,W) bool np arrays
        self.boxes = [r['bbox'] for r in results]
        self.scores = [r['predicted_iou'] for r in results]

    def masked_images(self, source: Image) -> Generator[CroppedImage, None, None]:
        """Yield a masked RGBA crop for each segmented object."""
        source_rgba = source.image.convert("RGBA")

        for idx in range(len(self.masks)):
            mask = self.masks[idx]
            rgba = source_rgba.copy()
            alpha = PILImage.fromarray((mask * 255).astype(np.uint8), mode="L")
            rgba.putalpha(alpha)

            bbox = alpha.getbbox()
            if bbox:
                rgba = rgba.crop(bbox)
                yield CroppedImage(Image(rgba), self.boxes[idx], alpha, self.scores[idx])


class ImageSeg:
    def __init__(self, device):
        self.device = device

        self.model = build_sam2_hf(
            "facebook/sam2.1-hiera-large",
            device=device
        )

        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.9,
            min_mask_region_area=500,
            box_nms_thresh=0.7,
        )

    @classmethod
    def model_names(cls) -> list[str]:
        return ["facebook/sam2.1-hiera-large"]

    def segment(self, input: Image, prompt: str = "all objects") -> SegmentationResult:
        image_np = np.array(input.image.convert("RGB"))
        with torch.no_grad():
            results = self.mask_generator.generate(image_np)
        return SegmentationResult(results)