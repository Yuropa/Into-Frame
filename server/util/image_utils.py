from __future__ import annotations
import PIL
import PIL.ImageDraw
import matplotlib.pyplot as plt
from matplotlib import colormaps
from pathlib import Path
import numpy as np
import cv2
import torchvision.transforms.functional as F


def make_context_composite(
    crop_pil: PIL.Image.Image,
    scene_pil: PIL.Image.Image,
    box: list[float] | None = None,
    size: int = 448,
) -> PIL.Image.Image:
    """Build a square context image for classification models.

    Top half: scene thumbnail with a red rectangle marking the object.
    Bottom half: crop centred on a dark panel.

    At 448×448 the CLIPProcessor (shortest_edge=224, center_crop=224×224)
    scales uniformly to 224×224 with no cropping, giving each half 112 px."""
    half = size // 2

    # Top panel: scene thumbnail + optional highlight
    scene_rgb = scene_pil.convert("RGB")
    s_w, s_h = scene_rgb.size
    scene_panel = scene_rgb.resize((size, half), PIL.Image.LANCZOS)
    if box is not None:
        draw = PIL.ImageDraw.Draw(scene_panel)
        bx, by, bw, bh = box
        sx, sy = size / s_w, half / s_h
        tx, ty = int(bx * sx), int(by * sy)
        tw, th = max(2, int(bw * sx)), max(2, int(bh * sy))
        draw.rectangle([tx, ty, tx + tw, ty + th], outline=(255, 60, 60), width=2)

    # Bottom panel: crop centred on a dark background
    crop_rgb = crop_pil.convert("RGB")
    c_w, c_h = crop_rgb.size
    scale = min(size / c_w, half / c_h)
    new_w, new_h = max(1, int(c_w * scale)), max(1, int(c_h * scale))
    crop_thumb = crop_rgb.resize((new_w, new_h), PIL.Image.LANCZOS)
    crop_panel = PIL.Image.new("RGB", (size, half), (32, 32, 32))
    crop_panel.paste(crop_thumb, ((size - new_w) // 2, (half - new_h) // 2))

    composite = PIL.Image.new("RGB", (size, size), (0, 0, 0))
    composite.paste(scene_panel, (0, 0))
    composite.paste(crop_panel, (0, half))
    return composite

def lab_color_transfer(
    source: PIL.Image.Image,
    target: PIL.Image.Image,
    strength: float = 0.35,
    mask: PIL.Image.Image | None = None,
) -> PIL.Image.Image:
    """Nudge target's per-channel LAB colour statistics toward source's.

    Matches mean/std of each LAB channel (source stats onto target), then
    blends the result back with the original target by `strength` (or a
    per-pixel strength from `mask`, in [0, 255]) so structure/detail from
    target is preserved while only the colour cast shifts toward source.
    """
    if strength <= 0.0:
        return target

    def to_lab(img: PIL.Image.Image) -> np.ndarray:
        rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    src = to_lab(source)
    tgt = to_lab(target)

    transferred = tgt.copy()
    for ch in range(3):
        tgt_mean = tgt[..., ch].mean()
        tgt_std = tgt[..., ch].std() + 1e-6
        src_mean = src[..., ch].mean()
        src_std = src[..., ch].std() + 1e-6
        transferred[..., ch] = (tgt[..., ch] - tgt_mean) / tgt_std * src_std + src_mean

    if mask is not None:
        mask_resized = mask.convert("L").resize((target.width, target.height), PIL.Image.BILINEAR)
        pixel_strength = np.array(mask_resized, dtype=np.float32)[..., None] / 255.0 * strength
    else:
        pixel_strength = strength

    blended = tgt + pixel_strength * (transferred - tgt)

    blended[..., 0] = np.clip(blended[..., 0], 0.0, 100.0)
    blended[..., 1] = np.clip(blended[..., 1], -127.0, 127.0)
    blended[..., 2] = np.clip(blended[..., 2], -127.0, 127.0)

    rgb = cv2.cvtColor(blended, cv2.COLOR_LAB2RGB)
    return PIL.Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))


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

    @classmethod
    def load(cls, path: Path) -> Image:
        return cls(path)

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

    def __eq__(self, other: Image) -> bool:
        if not isinstance(other, Image):
            return False
        if self.image.size != other.image.size:
            return False
        return np.array_equal(np.array(self.image), np.array(other.image))

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
