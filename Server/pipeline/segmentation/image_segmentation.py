from util.image_utils import Image
from sam2.build_sam import build_sam2_hf
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from pipeline.segmentation.segmentation_result import SegmentationResult
import numpy as np
import torch

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
            min_mask_region_area=100,
            box_nms_thresh=0.7,
        )

    @classmethod
    def model_names(cls) -> list[str]:
        return ["facebook/sam2.1-hiera-large"]

    def segment(self, input: Image) -> SegmentationResult:
        image_np = np.array(input.rgb())
        with torch.no_grad():
            results = self.mask_generator.generate(image_np)
        return SegmentationResult.from_results(results)