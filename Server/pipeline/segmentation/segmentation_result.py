from PIL import Image as PILImage
from util.image_utils import Image
import numpy as np
from typing import Generator

class CroppedImage:
    def __init__(self, image, box, mask, cropped_image, score) -> None:
        self.image = image
        self.box = box
        self.score = score
        self.mask = mask
        self.cropped_image = cropped_image

class SegmentationResult:
    def __init__(self, masks, boxes, scores):
        self.masks = masks if isinstance(masks, list) else [masks]
        self.boxes = boxes if isinstance(boxes, list) else [boxes]
        self.scores = scores if isinstance(scores, list) else [scores]

    @classmethod
    def from_results(cls, results):
        return cls(
            masks=[r['segmentation'] for r in results],
            boxes=[r['bbox'] for r in results],
            scores=[r['predicted_iou'] for r in results],
        )

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
                cropped_image = Image(source.image.crop(bbox))
                yield CroppedImage(Image(rgba), self.boxes[idx], alpha, cropped_image, self.scores[idx])
