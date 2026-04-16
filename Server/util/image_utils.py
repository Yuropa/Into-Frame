from __future__ import annotations
import PIL
import matplotlib.pyplot as plt
from matplotlib import colormaps
from pathlib import Path
import numpy as np
import cv2
import torchvision.transforms.functional as F

class Image:
    image: PIL.Image.Image

    def __init__(self, obj):
        if isinstance(obj, str):
            self.image = PIL.Image.open(obj).convert("RGB")
        elif isinstance(obj, Image):
            self.image = obj.image
        elif isinstance(obj, PIL.Image.Image):
            self.image = obj
        elif isinstance(obj, np.ndarray):
            self.image = PIL.Image.fromarray(obj).convert("RGB")
        elif isinstance(obj, Path):
            self.image = PIL.Image.open(str(obj)).convert("RGB")
        else:
            raise TypeError(f"Unsupported type: {type(obj)}")
        
        self._canny = None 
        self._rgb = None
        self._rgba = None
        self._L = None

    def canny(self):
        if self._canny is None:
            self._canny = self._generate_canny(self.image)
        return self._canny

    def rgb(self, copy: bool = False) -> PIL.Image.Image:
        if self._rgb is None:
            self._rgb = self.image.convert("RGB")
        
        if copy:
            return self._rgb.copy()
        else:
            return self._rgb
        
    def rgba(self, copy: bool = False) -> PIL.Image.Image:
        if self._rgba is None:
            self._rgba = self.image.convert("RGBA")
        
        if copy:
            return self._rgba.copy()
        else:
            return self._rgba

    def L(self, copy: bool = False) -> PIL.Image.Image:
        if self._L is None:
            self._L = self.image.convert("L")
        
        if copy:
            return self._L.copy()
        else:
            return self._L

    def save(self, path):
        self.image.save(path)

    def show(self):
        self._show_image(self.image)

    def copy(self) -> Image:
        return Image(self.image.copy())

    @property
    def width(self):
        return self.image.width

    @property
    def height(self):
        return self.image.height

    @property
    def size(self):
        return (self.image.width, self.image.height)
    
    def __getitem__(self, key):
        return self.image[key]

    def show_masks(self, masks):
        image = self.image.convert("RGBA")

        masks = 255 * masks.cpu().numpy().astype(np.uint8)
    
        n_masks = masks.shape[0]
        cmap = colormaps.get_cmap("rainbow").resampled(n_masks)
        colors = [
            tuple(int(c * 255) for c in cmap(i)[:3])
            for i in range(n_masks)
        ]

        for mask, color in zip(masks, colors):
            mask = PIL.Image.fromarray(mask)
            overlay = PIL.Image.new("RGBA", image.size, color + (0,))
            alpha = mask.point(lambda v: int(v * 0.5))
            overlay.putalpha(alpha)
            image = PIL.Image.alpha_composite(image, overlay)

        self._show_image(image)


    def _generate_canny(self, img):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return PIL.Image.fromarray(cv2.Canny(img, 100, 200))
    
    def _show_image(self, image):
        plt.imshow(image)
        plt.axis('off')
        plt.show()
