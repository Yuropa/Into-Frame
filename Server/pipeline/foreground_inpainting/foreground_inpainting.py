from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.segmentation.foreground_segmentation import ForegroundSeg
from util.image_utils import Image 
from util.device_utils import clean_device_cache
from pathlib import Path
from PIL import Image as PILImage
from scipy.ndimage import binary_dilation
import numpy as np
import gc
import torch

class ForegroundInpaint:
    def __init__(self, device, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype

    def inpaint(self, input: Image, temp_path: Path) -> Image:
        result = input
        for idx in range(20):
            result = result.copy()

            segment = ForegroundSeg(self.device)
            segmentation = segment.segment(result)

            segment.unload()
            self._clean_up(segment)

            # Apply mask to input image
            mask = segmentation.masks[0]
            if mask.ndim == 3:
                mask = mask[..., 0]

            fill_pct = mask.mean()
            print(f"Mask fill: {(fill_pct * 100):.2f}%  ({int(mask.sum())} / {mask.size} px)", flush=True)
            self._save_mask(mask, temp_path / f"mask_{idx}.png")

            dilation_factor = 50
            struct = np.ones((dilation_factor * 2 + 1, dilation_factor * 2 + 1))
            mask = binary_dilation(mask, structure=struct).astype(np.float32)

            masked_input = self._apply_mask(result, mask)

            # Save the masked input image
            masked_save_path = temp_path / f"masked_input_{idx}.png"
            masked_input.save(masked_save_path)

            if fill_pct < 0.01:
                break

            inpaint = InPainting(self.device, self.torch_dtype)

            mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8), mode="L")

            inpaint = InPainting(
                self.device, 
                self.torch_dtype,
                ForegroundInpaint._preferred_inpainting()
            )
            result = inpaint.inpaint(
                masked_input, 
                mask_pil, 
                temp_path=temp_path,
                prompt="no objects, clean background, seamless, empty landscape",
                guidance_scale=2.0,
                strength=1.0
            )

            result = Image(result)
            result.save(temp_path / f"inpainted_{idx}.png")

            inpaint.close()
            self._clean_up(inpaint)

        return result
    
    def _clean_up(self, obj):
        del obj
        gc.collect()
        clean_device_cache(self.device)

    def _save_mask(self, mask: np.ndarray, save_path: Path):
        mask_array = mask.astype(np.float32)

        # Normalize to [0, 255] if in [0, 1] range
        if mask_array.max() <= 1.0:
            mask_array = mask_array * 255.0

        # If mask is 3D, collapse to 2D by taking first channel
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]

        PILImage.fromarray(mask_array.astype(np.uint8), mode="L").save(save_path)

    def _apply_mask(self, image: Image, mask: np.ndarray) -> Image:
        # Explicitly convert PIL Images to numpy arrays
        image_array = np.array(image.rgb())
        mask_array = mask.astype(np.float32)

        # Mask is already in [0, 1] — no normalization needed
        # If mask is 3D, collapse to 2D by taking first channel
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]

        mask_array = 1 - mask_array

        # Expand mask to (H, W, 1) and broadcast to match image (H, W, C)
        mask_array = mask_array[..., np.newaxis]
        mask_array = np.broadcast_to(mask_array, image_array.shape)

        # Apply mask
        masked_array = (image_array * mask_array).astype(np.uint8)

        return PILImage.fromarray(masked_array)


    @classmethod
    def _preferred_inpainting(cls) -> InPaintingType:
        return InPaintingType.LAMA

    @classmethod
    def model_names(cls) -> list[str]:
        return InPainting.model_names(type=cls._preferred_inpainting()) + ForegroundSeg.model_names()
