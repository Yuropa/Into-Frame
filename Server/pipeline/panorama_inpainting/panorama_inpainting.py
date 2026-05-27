from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.segmentation.foreground_segmentation import ForegroundSeg
from util.image_utils import Image
from util.panorama_utils import Panorama
from util.device_utils import clean_device_cache
from pathlib import Path
from PIL import Image as PILImage
from scipy.ndimage import binary_dilation
from typing import Optional
import numpy as np
import gc


class PanoramaInpaint:
    """
    Iteratively segments and removes foreground objects from a panorama,
    saving each detected object crop for later 3D model generation, then
    inpaints the removed regions with plausible background content.
    """

    def __init__(self, device, torch_dtype, seed: int = 0):
        self.device = device
        self.torch_dtype = torch_dtype
        self.seed = seed

    def inpaint(self, panorama: Panorama, temp_path: Path) -> tuple[Panorama, list[Image]]:
        result = Image(panorama.image.copy())
        crops: list[Image] = []

        for idx in range(20):
            result = result.copy()

            segment = ForegroundSeg(self.device)
            segmentation = segment.segment(result)
            segment.unload()
            self._clean_up(segment)

            mask = segmentation.masks[0]
            if mask.ndim == 3:
                mask = mask[..., 0]

            fill_pct = float(mask.mean())
            if fill_pct < 0.01:
                break

            crop = self._crop_object(result, mask)
            if crop is not None:
                crops.append(crop)
                crop.save(temp_path / f"pano_crop_{idx}.png")

            dilation_factor = 30
            struct = np.ones((dilation_factor * 2 + 1, dilation_factor * 2 + 1))
            dilated_mask = binary_dilation(mask, structure=struct).astype(np.float32)

            masked_input = self._apply_mask(result, dilated_mask)
            mask_pil = PILImage.fromarray((dilated_mask * 255).astype(np.uint8), mode="L")

            inpainter = InPainting(self.device, self.torch_dtype, InPaintingType.LAMA)
            result_pil = inpainter.inpaint(
                masked_input,
                mask_pil,
                temp_path=temp_path,
                prompt="no objects, clean background, seamless, empty landscape",
                guidance_scale=2.0,
                strength=1.0,
            )
            inpainter.close()
            self._clean_up(inpainter)

            result = Image(result_pil)
            result.save(temp_path / f"pano_inpainted_{idx}.png")

        return Panorama(result), crops

    def _crop_object(self, image: Image, mask: np.ndarray) -> Optional[Image]:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return None
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        img_array = np.array(image.rgb())
        cropped = img_array[y_min:y_max + 1, x_min:x_max + 1]
        return Image(PILImage.fromarray(cropped))

    def _apply_mask(self, image: Image, mask: np.ndarray) -> PILImage:
        image_array = np.array(image.rgb())
        mask_array = (1.0 - mask.astype(np.float32))
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        mask_array = mask_array[..., np.newaxis]
        mask_array = np.broadcast_to(mask_array, image_array.shape)
        masked = (image_array * mask_array).astype(np.uint8)
        return PILImage.fromarray(masked)

    def _clean_up(self, obj):
        del obj
        gc.collect()
        clean_device_cache(self.device)

    @classmethod
    def model_names(cls) -> list[str]:
        return InPainting.model_names(type=InPaintingType.LAMA) + ForegroundSeg.model_names()
