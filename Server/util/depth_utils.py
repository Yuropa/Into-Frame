import PIL
from pathlib import Path
import numpy as np

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

    def save(self, path, scale=None):
        depth = self.depth.copy()

        if scale is not None:
            depth = depth * scale
        else:
            # Normalize to full uint16 range
            dmin, dmax = depth.min(), depth.max()
            if dmax > dmin:
                depth = (depth - dmin) / (dmax - dmin) * 65535.0

        depth = np.clip(depth, 0, np.iinfo(np.uint16).max)
        depth_16 = depth.astype(np.uint16)

        img = PIL.Image.fromarray(depth_16, mode="I;16")
        img.save(path)

    def save_raw(self, path):
        np.save(path, self.depth)

    def min(self):
        return self.depth.min()
    
    def max(self):
        return self.depth.max()

    def normalize(self):
        dmin, dmax = self.min(), self.max()
        if dmax > dmin:
            self.depth = (self.depth - dmin) / (dmax - dmin)
        return self