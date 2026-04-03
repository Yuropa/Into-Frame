from pathlib import Path
import torch

from .image_utils import InputImage
from .segmentation import ImageSeg
from .inpainting import InPainting

class PipelineConfiguration:
    input: Path
    output: Path
    input: Path

    def __init__(self, input: str, output: str):
        self.input = Path(input)
        self.output = Path(output)
        self.temp = Path(output + "/build")

        self.output.mkdir(parents=True, exist_ok=True)
        self.temp.mkdir(parents=True, exist_ok=True)    

        if torch.cuda.is_available():
            self.device = "cuda"
            self.torch_dtype = torch.bfloat16
        elif torch.mps.is_available():
            self.device = "mps"
            self.torch_dtype = torch.float
        else:
            self.device = "cpu"
            self.torch_dtype = torch.bfloat16

class Pipeline:
    def __init__(self, config: PipelineConfiguration):
        self.config = config
        self.device = config.device
        self.torch_dtype = config.torch_dtype

    def run(self):
        print(f"Running with input: {self.config.input}")
        image = InputImage(self.config.input)
        # image.show()

        # seg = ImageSeg(self.device)
        # result = seg.segment(input=image)
        # image.show_masks(result.masks)

        inpaint = InPainting(self.device, self.torch_dtype)
        result = inpaint.inpaint(
            #input=image,
            prompt="An image of a cute cat"
        )
        image._show_image(result)
