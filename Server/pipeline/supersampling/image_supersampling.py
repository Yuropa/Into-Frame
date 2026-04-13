import torch
import PIL
import numpy as np
from util.image_utils import Image
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

class SuperSample:
    def __init__(self, device) -> None:
        self.device = device
        self.processor = Swin2SRImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x4-64")
        self.model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x4-64").to(device)

    def supersample(self, image: Image) -> Image:
        inputs = self.processor(image.image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs).reconstruction

        # Convert to PIL
        output = output.squeeze().cpu().clamp(0, 1).numpy()
        output = (output * 255).astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))
        result = PIL.Image.fromarray(output)
        return Image(result)
    
    @classmethod
    def model_names(cls) -> list[str]:
        return ["caidas/swin2SR-classical-sr-x4-64"]
