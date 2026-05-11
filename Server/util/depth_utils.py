from __future__ import annotations
import PIL
from pathlib import Path
import numpy as np
from typing import Self

class Depth:
    def __init__(self, obj):
        if isinstance(obj, str):
            obj = Path(obj)

        if isinstance(obj, Path):
            if obj.suffix == ".npy":
                self.depth = np.load(obj)
            else:
                # Load image (could be 16-bit PNG)
                self.depth = np.array(PIL.Image.open(obj))
        elif isinstance(obj, Depth):
            self.depth = obj.depth
        elif isinstance(obj, np.ndarray):
            self.depth = obj
        else:
            raise TypeError(f"Unsupported type: {type(obj)}")

        self.depth = self.depth.astype(np.float32)
        if self.depth.ndim == 3 and self.depth.shape[0] == 1:
            self.depth = self.depth.squeeze(0)

        self._gray = None

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls(np.load(path))

    @property
    def width(self):
        return self.depth.shape[1]

    @property
    def height(self):
        return self.depth.shape[0]

    @property
    def size(self):
        return (self.depth.shape[1], self.depth.shape[0])
    
    def gray(self):
        if self._gray is not None:
            return self._gray

        depth = self.depth.copy()
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        dmin, dmax = depth.min(), depth.max()
        if dmax > dmin:
            depth = (depth - dmin) / (dmax - dmin)
        else:
            depth = np.zeros_like(depth)

        self._gray = (depth * 255).astype(np.uint8)  # (H, W), grayscale
        return self._gray

    def copy(self) -> Depth:
        return Depth(self.depth.copy())
    
    def __getitem__(self, key):
        return self.depth[key]

    def save(self, path: Path):
        np.save(path, self.depth)

    def save_debug_image(self, path: Path):
        PIL.Image.fromarray(self.gray(), mode="L").save(path)

    def min(self):
        return self.depth.min()
    
    def max(self):
        return self.depth.max()

    def normalize(self):
        dmin, dmax = self.min(), self.max()
        if dmax > dmin:
            self.depth = (self.depth - dmin) / (dmax - dmin)
        return self