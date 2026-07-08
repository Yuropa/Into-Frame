from __future__ import annotations
import PIL.Image
from pathlib import Path
import numpy as np
from typing import Self

class IntrinsicImages:
    """
    Per-pixel albedo (unlit reflectance) and surface normal maps predicted by an
    intrinsic image decomposition model (e.g. IntrinsicDiffusion).

    albedo: (H, W, 3) float32, linear RGB reflectance, unbounded (>= 0).
    normal: (H, W, 3) float32, unit vectors in [-1, 1], tangent-space, +Y up (OpenGL convention).
    """

    def __init__(self, obj):
        if isinstance(obj, str):
            obj = Path(obj)

        if isinstance(obj, Path):
            loaded = IntrinsicImages.load(obj)
            self.albedo = loaded.albedo
            self.normal = loaded.normal
            return
        elif isinstance(obj, IntrinsicImages):
            self.albedo = obj.albedo
            self.normal = obj.normal
            return
        elif isinstance(obj, dict):
            self.albedo = np.asarray(obj["albedo"], dtype=np.float32)
            self.normal = np.asarray(obj["normal"], dtype=np.float32)
        else:
            raise TypeError(f"Unsupported type: {type(obj)}")

        if self.albedo.ndim != 3 or self.albedo.shape[2] != 3:
            raise ValueError(f"albedo must be (H, W, 3), got {self.albedo.shape}")
        if self.normal.ndim != 3 or self.normal.shape[2] != 3:
            raise ValueError(f"normal must be (H, W, 3), got {self.normal.shape}")
        if self.albedo.shape[:2] != self.normal.shape[:2]:
            raise ValueError(f"albedo/normal size mismatch: {self.albedo.shape[:2]} vs {self.normal.shape[:2]}")

    @classmethod
    def load(cls, path: Path) -> Self:
        with np.load(path) as data:
            return cls({"albedo": data["albedo"], "normal": data["normal"]})

    @property
    def width(self):
        return self.albedo.shape[1]

    @property
    def height(self):
        return self.albedo.shape[0]

    @property
    def size(self):
        return (self.width, self.height)

    def copy(self) -> IntrinsicImages:
        return IntrinsicImages({"albedo": self.albedo.copy(), "normal": self.normal.copy()})

    def albedo_image(self) -> PIL.Image.Image:
        """Albedo tonemapped to sRGB for viewing (source is linear and unbounded)."""
        srgb = np.clip(self.albedo, 0.0, 1.0) ** (1.0 / 2.2)
        return PIL.Image.fromarray((srgb * 255.0 + 0.5).astype(np.uint8), mode="RGB")

    def normal_image(self) -> PIL.Image.Image:
        """Normal map encoded the standard way: (n + 1) / 2 -> RGB."""
        vis = np.clip((self.normal + 1.0) * 0.5, 0.0, 1.0)
        return PIL.Image.fromarray((vis * 255.0 + 0.5).astype(np.uint8), mode="RGB")

    def save(self, path: Path):
        np.savez(path, albedo=self.albedo, normal=self.normal)

    def save_debug_albedo(self, path: Path):
        self.albedo_image().save(path)

    def save_debug_normal(self, path: Path):
        self.normal_image().save(path)
