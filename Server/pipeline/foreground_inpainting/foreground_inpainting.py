from pipeline.inpainting.inpainting import InPainting
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

    def log_gpu_memory(self, label: str):
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        print(f"[{label}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")


    def inpaint(self, input: Image, temp_path: Path) -> Image:
        self.log_gpu_memory("start")
        segment = ForegroundSeg(self.device)
        self.log_gpu_memory("after ForegroundSeg load")

        segmentation = segment.segment(input)
        self.log_gpu_memory("after segment()")


        segment.unload()
        del segment
        gc.collect()
        clean_device_cache(self.device)
        self.log_gpu_memory("after segment cleanup")

        # Apply mask to input image
        mask = segmentation.masks[0]
        if mask.ndim == 3:
            mask = mask[..., 0]
        struct = np.ones((21, 21))
        mask = binary_dilation(mask, structure=struct).astype(np.float32)

        self._save_mask(mask, temp_path / "mask.png")
        masked_input = self._apply_mask(input, mask)

        # Save the masked input image
        masked_save_path = temp_path / "masked_input.png"
        masked_input.save(masked_save_path)

        self.log_gpu_memory("before InPainting load")
        inpaint = InPainting(self.device, self.torch_dtype)
        self.log_gpu_memory("after InPainting load")

        mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8), mode="L")

        print(f"masked_input size: {masked_input.size}")
        print(f"mask_pil size: {mask_pil.size}")
        print(f"mask_pil mode: {mask_pil.mode}")
        print(f"mask values unique: {np.unique(np.array(mask_pil))}")

        inpaint = InPainting(self.device, self.torch_dtype)
        result = inpaint.inpaint(masked_input, mask_pil)

        del inpaint
        gc.inpaint()
        clean_device_cache(self.device)

        return result

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
    def model_names(cls) -> list[str]:
        return InPainting.model_names() + ForegroundSeg.model_names()
