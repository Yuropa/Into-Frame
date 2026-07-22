import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from util.image_utils import Image, flatten_alpha_with_mean_fill


class DinoV2Embedder:
    """Lightweight, in-process visual-similarity embedder (mirrors
    ImageClipClassifier's usage of `transformers` -- unlike the SAM3D/TRELLIS
    3D generators, which run out-of-process over a conda-env socket, DINOv2 is
    a plain `transformers` vision model with no exotic dependencies).

    Used by ObjectCategoryClusteringStage to sub-cluster same-class detections
    by visual similarity (species/color variants), which CLIP's zero-shot
    *text*-prompted classification isn't suited for -- it can tell "flower"
    from "tree", not one flower's embedding from another's.
    """

    MODEL_NAME = "facebook/dinov2-base"

    def __init__(self, device: torch.device, model_name: str | None = None):
        self.device = device
        self.model_name = model_name or self.MODEL_NAME
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)
        self.model.eval()

    @classmethod
    def model_names(cls) -> list[str]:
        return [cls.MODEL_NAME]

    def embed(self, image: Image) -> np.ndarray:
        """Returns an L2-normalized embedding vector for the crop. Alpha is
        flattened with the mean-opaque-color fill (same as ImageClipClassifier)
        rather than PIL's default black fill, so the masked-out background
        doesn't read as a dark silhouette shape that skews the embedding."""
        pil_input = flatten_alpha_with_mean_fill(image)
        inputs = self.processor(images=pil_input, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            # CLS token -- available regardless of whether this checkpoint
            # exposes a separate pooler head.
            features = out.last_hidden_state[:, 0, :]
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy()
