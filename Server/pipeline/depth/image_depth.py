import torch
from depth_anything_3.api import DepthAnything3
from util.image_utils import Image

class DepthResult:
    def __init__(self, result) -> None:
        self.depth = result.depth
        self.confidence = result.conf
        self.extrinsics = result.extrinsics
        self.intrinsics = result.intrinsics

class ImageDepth:
    def __init__(self, device):
        self.device = device

        self.model = DepthAnything3.from_pretrained("depth-anything/da3-giant")
        self.model = self.model.to(device=device)

    @classmethod
    def model_names(cls) -> list[str]:
        return ["depth-anything/da3-giant"]

    def depth(self, input: Image) -> DepthResult:
        image = input.image.convert("RGB")
        with torch.no_grad():
            result = self.model.inference(image)
        return DepthResult(result)