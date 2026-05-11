from pipeline.inpainting.inpainting import InPainting
from pipeline.segmentation.foreground_segmentation import ForegroundSeg
from util.image_utils import Image 

class ForegroundInpaint:
    def __init__(self, device, torch_dtype):
        self.inpaint = InPainting(device, torch_dtype)
        self.segment = ForegroundSeg(device)

    def inpaint(self, input: Image) -> Image:
        segmentation = self.segment(input)
        result = self.inpaint(input, segmentation.mask)
        return result

    @classmethod
    def model_names(cls) -> list[str]:
        return InPainting.model_names() + ForegroundSeg.model_names()
