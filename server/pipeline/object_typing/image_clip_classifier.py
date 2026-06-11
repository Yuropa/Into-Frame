from util.image_utils import Image
from pipeline.object_typing.categories import OBJECT_CATEGORIES, ENVIRONMENT_CATEGORIES
from transformers import CLIPProcessor, CLIPModel
from PIL import Image as PILImage
import numpy as np
import torch

_OBJECT_LABELS = frozenset(OBJECT_CATEGORIES.keys())
_ALL_CATEGORIES = {**OBJECT_CATEGORIES, **ENVIRONMENT_CATEGORIES}


class ImageClipClassifier:
    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self, device: torch.device, confidence_threshold: float = 0.75):
        self.device = device
        self._confidence_threshold = confidence_threshold
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        self.model = CLIPModel.from_pretrained(self.MODEL_NAME).to(device)
        self.model.eval()

        # Pre-compute text embeddings once — reused for every image
        all_prompts: list[str] = []
        self._category_slices: list[tuple[str, int, int]] = []
        for label, prompts in _ALL_CATEGORIES.items():
            start = len(all_prompts)
            all_prompts.extend(prompts)
            self._category_slices.append((label, start, len(all_prompts)))

        text_inputs = self.processor(text=all_prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_out = self.model.text_model(**text_inputs)
            text_features = self.model.text_projection(text_out.pooler_output)
            self._text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    @classmethod
    def model_names(cls) -> list[str]:
        return [cls.MODEL_NAME]

    def _to_rgb(self, image: Image) -> PILImage.Image:
        """Convert to RGB, filling transparent pixels with the mean opaque color.

        PIL's default RGBA→RGB conversion fills alpha=0 areas with black, which
        creates a dark silhouette of whatever was masked out. For a sky crop whose
        mask has a tower-shaped hole, that silhouette triggers CLIP's "tower"
        classification even though the actual content is sky. Using the mean opaque
        color as the fill makes the hole blend into the dominant content instead."""
        pil = image.image
        if pil.mode != "RGBA":
            return image.rgb()
        arr = np.array(pil).astype(np.float32)
        alpha = arr[..., 3:4] / 255.0
        rgb = arr[..., :3]
        opaque = arr[..., 3] > 128
        mean_color = rgb[opaque].mean(axis=0) if opaque.any() else np.array([128.0, 128.0, 128.0])
        background = np.ones_like(rgb) * mean_color
        composited = (rgb * alpha + background * (1.0 - alpha)).astype(np.uint8)
        return PILImage.fromarray(composited, mode="RGB")

    def classify(self, image: Image) -> tuple[str, str]:
        """Returns (type, class) where type is the winning category label and
        class is 'object', 'environment', or 'indeterminate'.

        'indeterminate' is returned when neither side clears confidence_threshold,
        computed as max(obj_score, env_score) / (obj_score + env_score) — the same
        formula used by the caption-based classifier."""
        inputs = self.processor(images=self._to_rgb(image), return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_out = self.model.vision_model(**inputs)
            image_features = self.model.visual_projection(vision_out.pooler_output)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        sims = (image_features @ self._text_features.T).squeeze(0)

        best_obj_label, best_obj_score = None, -float("inf")
        best_env_label, best_env_score = None, -float("inf")
        for label, start, end in self._category_slices:
            score = sims[start:end].max().item()
            if label in _OBJECT_LABELS:
                if score > best_obj_score:
                    best_obj_score, best_obj_label = score, label
            else:
                if score > best_env_score:
                    best_env_score, best_env_label = score, label

        obj_s = max(0.0, best_obj_score)
        env_s = max(0.0, best_env_score)
        total = obj_s + env_s
        confidence = max(obj_s, env_s) / total if total > 0 else 0.0

        if confidence < self._confidence_threshold:
            return "indeterminate", "indeterminate"
        if best_obj_score >= best_env_score:
            return best_obj_label, "object"
        return best_env_label, "environment"
