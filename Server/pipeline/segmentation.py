from .image_utils import InputImage
from transformers import Sam3Processor, Sam3Model
import torch

class SegmentationResult:
    def __init__(self, results):
        self.masks = results['masks']
        self.boxes = results['boxes']
        self.scores = results['scores']

class ImageSeg:
    def __init__(self, device):
        self.device = device
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")


    def segment(self, input: InputImage, prompt: str= "all objects"):
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