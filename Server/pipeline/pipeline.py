from pathlib import Path
import torch

from .image_utils import InputImage
from .segmentation import ImageSeg

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
        elif torch.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

class Pipeline:
    def __init__(self, config: PipelineConfiguration):
        self.config = config
        self.device = config.device

    def run(self):
        print(f"Running with input: {self.config.input}")
        image = InputImage(self.config.input)
        # image.show()

        seg = ImageSeg(self.device)
        result = seg.segment(input=image)

        image.show_masks(result.masks)
