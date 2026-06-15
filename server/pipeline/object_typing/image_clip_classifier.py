from util.image_utils import Image, make_context_composite
from pipeline.object_typing.categories import OBJECT_CATEGORIES, ENVIRONMENT_CATEGORIES
from transformers import CLIPProcessor, CLIPModel
from PIL import Image as PILImage
import numpy as np
import torch

_OBJECT_LABELS = frozenset(OBJECT_CATEGORIES.keys())
_ALL_CATEGORIES = {**OBJECT_CATEGORIES, **ENVIRONMENT_CATEGORIES}


class ImageClipClassifier:
    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self, device: torch.device, confidence_threshold: float = 0.1):
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

    def classify(self, image: Image) -> tuple[str, float]:
        """Returns (label, confidence) — see classify_with_details for the full breakdown."""
        label, confidence, _, _criteria = self.classify_with_details(image)
        return label, confidence

    def classify_with_details(
        self,
        image: Image,
        scene_image: Image | None = None,
        box: list[float] | None = None,
        top_n: int = 5,
    ) -> tuple[str, float, list[tuple[str, float]], dict]:
        """Returns (label, confidence, top_candidates, criteria).

        label          — winning category or 'indeterminate'
        confidence     — rescaled to [0, 1]: 0 = perfectly tied, 1 = one-sided
        top_candidates — top_n (label, raw_score) pairs sorted best-first
        criteria       — dict with best_obj_label, best_obj_score, best_env_label,
                         best_env_score for debugging the decision
        scene_image    — full scene for context; if provided, a composite
                         (scene thumbnail + crop) is fed to CLIP instead of the
                         bare crop, giving the model spatial/scene context
        box            — [x, y, w, h] of the crop in scene_image pixel space;
                         drawn as a red highlight on the scene thumbnail
        """
        if scene_image is not None:
            pil_input = make_context_composite(self._to_rgb(image), scene_image.rgb(), box)
        else:
            pil_input = self._to_rgb(image)

        inputs = self.processor(images=pil_input, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_out = self.model.vision_model(**inputs)
            image_features = self.model.visual_projection(vision_out.pooler_output)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        sims = (image_features @ self._text_features.T).squeeze(0)

        best_obj_label, best_obj_score = None, -float("inf")
        best_env_label, best_env_score = None, -float("inf")
        per_label: list[tuple[str, float]] = []
        for label, start, end in self._category_slices:
            score = sims[start:end].max().item()
            per_label.append((label, score))
            if label in _OBJECT_LABELS:
                if score > best_obj_score:
                    best_obj_score, best_obj_label = score, label
            else:
                if score > best_env_score:
                    best_env_score, best_env_label = score, label

        obj_s = max(0.0, best_obj_score)
        env_s = max(0.0, best_env_score)
        total = obj_s + env_s
        # Raw ratio is in [0.5, 1.0]; rescale to [0, 1] so the threshold is meaningful.
        confidence = (max(obj_s, env_s) / total - 0.5) * 2.0 if total > 0 else 0.0

        top_candidates = sorted(per_label, key=lambda x: x[1], reverse=True)[:top_n]

        criteria = {
            "best_obj_label": best_obj_label,
            "best_obj_score": round(best_obj_score, 4),
            "best_env_label": best_env_label,
            "best_env_score": round(best_env_score, 4),
        }

        if confidence < self._confidence_threshold:
            return "indeterminate", confidence, top_candidates, criteria
        if best_obj_score >= best_env_score:
            return best_obj_label, confidence, top_candidates, criteria
        return best_env_label, confidence, top_candidates, criteria
