from util.image_utils import Image
from pipeline.object_typing.categories import OBJECT_CATEGORIES
from transformers import CLIPProcessor, CLIPModel
import torch


class ImageClipClassifier:
    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self, device: torch.device):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        self.model = CLIPModel.from_pretrained(self.MODEL_NAME).to(device)
        self.model.eval()

        # Pre-compute text embeddings once — reused for every image
        all_prompts: list[str] = []
        self._category_slices: list[tuple[str, int, int]] = []
        for label, prompts in OBJECT_CATEGORIES.items():
            start = len(all_prompts)
            all_prompts.extend(prompts)
            self._category_slices.append((label, start, len(all_prompts)))

        text_inputs = self.processor(text=all_prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            self._text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    @classmethod
    def model_names(cls) -> list[str]:
        return [cls.MODEL_NAME]

    def classify(self, image: Image) -> str:
        inputs = self.processor(images=image.rgb(), return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Per-category score = max cosine similarity across that category's prompts
        sims = (image_features @ self._text_features.T).squeeze(0)
        best_label = max(
            self._category_slices,
            key=lambda t: sims[t[1]:t[2]].max().item()
        )[0]
        return best_label
