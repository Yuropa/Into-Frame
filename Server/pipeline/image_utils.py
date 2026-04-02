from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np

class InputImage:
    def __init__(self, path):
        self.image = Image.open(path).convert("RGB")

    def _show_image(self, image):
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def show(self):
        self._show_image(self.image)

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
            mask = Image.fromarray(mask)
            overlay = Image.new("RGBA", image.size, color + (0,))
            alpha = mask.point(lambda v: int(v * 0.5))
            overlay.putalpha(alpha)
            image = Image.alpha_composite(image, overlay)

        self._show_image(image)
