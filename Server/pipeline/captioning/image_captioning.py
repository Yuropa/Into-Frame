from util.image_utils import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class ImageCaptioning:
    def __init__(self, device):
        self.device = device

        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    @classmethod
    def model_names(cls) -> list[str]:
        return ["Salesforce/blip-image-captioning-large"]

    def caption(self, input: Image, prompt: str = ""):
        inputs = self.processor(input.image.convert('RGB'), prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs)

        result = self.processor.decode(out[0], skip_special_tokens=True)
        return result
