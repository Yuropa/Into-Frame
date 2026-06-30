import torch
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
from diffusers import FluxFillPipeline
from util.image_utils import Image
from util.device_utils import offload_pipeline

class InPaintingFlux:
    def __init__(self, device, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype
        # FLUX models are heavy; 'dev' is high quality, 'schnell' is faster
        self.model_id = "black-forest-labs/FLUX.1-Fill-dev"

        self.pipeline = FluxFillPipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch_dtype
        )
        
        offload_pipeline(device, self.pipeline)
        self.pipeline.enable_vae_tiling()
        self.pipeline.set_progress_bar_config(disable=True)

    @classmethod
    def model_names(cls) -> list[str]:
        return ["black-forest-labs/FLUX.1-Fill-dev"]

    def inpaint(
        self,
        input_image: PILImage,
        mask_image: PILImage,
        temp_path: Path,
        prompt: str = "",
        num_inference_steps=30,
        guidance_scale=30.0,
        strength=1.0,
        seed: int = 0,
    ) -> PILImage:
        """
        For FLUX.1-Fill:
        - If prompt is "", it performs logic-based background reconstruction.
        - Guidance scale for FLUX Fill typically defaults higher (around 30.0) compared to SD.
        """

        def normalize_dimension(dim):
            return (dim // 16) * 16
        
        # Ensure images are in RGB for the pipeline
        width = normalize_dimension(input_image.width)
        height = normalize_dimension(input_image.height)

        init_img = input_image.resize((width, height), PILImage.LANCZOS)
        mask_img = mask_image.resize((width, height), PILImage.NEAREST)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        output = self.pipeline(
            prompt=prompt,
            image=init_img,
            mask_image=mask_img,
            height=height,
            width=width,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
            generator=generator,
        ).images[0]

        return output

    def close(self):
        import gc
        if self.pipeline is not None:
            # Drain any in-flight GPU work before tearing down the model.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # enable_model_cpu_offload registers accelerate hooks that create
            # reference cycles. Remove them so the del can reach refcount 0
            # without waiting for cyclic GC.
            try:
                from accelerate.hooks import remove_hook_from_module
                remove_hook_from_module(self.pipeline, recurse=True)
            except Exception:
                pass
            # Explicitly move all components to CPU so no GPU tensors are left
            # stranded on-device after accelerate deferred an offload.
            for component in list(getattr(self.pipeline, 'components', {}).values()):
                if hasattr(component, 'to') and callable(component.to):
                    try:
                        component.to('cpu')
                    except Exception:
                        pass
            del self.pipeline
            self.pipeline = None
        # Multiple GC rounds to break deep cycles in accelerate/diffusers
        # internals before releasing the CUDA allocator cache.
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()