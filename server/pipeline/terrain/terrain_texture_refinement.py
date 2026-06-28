from typing import Any, Optional
from logging import Logger

import numpy as np
import PIL.Image
import torch
from scipy.ndimage import binary_dilation, gaussian_filter

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.supersampling.image_supersampling import ImageSupersampling
from util.depth_utils import Depth
from util.image_utils import Image
from util.device_utils import DeviceStrategy, preferred_device


class TerrainTextureRefinementConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        certainty_threshold: float = 0.2,
        dilation_px: int = 8,
        inpainting_type: str = "FLUX",
        num_inference_steps: int = 30,
        guidance_scale: float = 30.0,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.certainty_threshold = certainty_threshold
        self.dilation_px = dilation_px
        self.inpainting_type = InPaintingType[inpainting_type]
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale


class TerrainTextureRefinementStage(PipelineStage):
    """
    Refines the terrain texture by:
      1. Masking low-certainty regions (nadir dead-zone, grazing-horizon pixels,
         unobserved heightmap cells) identified by TERRAIN_TEXTURE_CERTAINTY.
      2. Inpainting those regions with a diffusion model conditioned on the
         surrounding confident panorama pixels.
      3. Blending the inpainted result back with a soft certainty-weighted alpha
         so there are no hard seams at the mask boundary.
      4. Supersampling the blended 1k texture to 2k via Swin2SR.

    The output overwrites TERRAIN_TEXTURE with the refined 2k image.

    Reads:
      ContextKey.TERRAIN_TEXTURE           — baked colour image (Image, 1k²)
      ContextKey.TERRAIN_TEXTURE_CERTAINTY — per-texel certainty (Depth, [0,1])
      ContextKey.INPUT_CAPTION             — optional scene caption for inpainting prompt

    Writes:
      ContextKey.TERRAIN_TEXTURE           — refined + supersampled colour image (Image, 2k²)
    """

    @classmethod
    def config_class(cls) -> type[TerrainTextureRefinementConfiguration]:
        return TerrainTextureRefinementConfiguration

    def __init__(self, config: TerrainTextureRefinementConfiguration) -> None:
        super().__init__(config)
        self._inpainter: Optional[InPainting] = None
        self._sampler: Optional[ImageSupersampling] = None

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: TerrainTextureRefinementConfiguration = self.config

        texture_img = context.input_image(ContextKey.TERRAIN_TEXTURE)
        certainty_depth = context.input_depth(ContextKey.TERRAIN_TEXTURE_CERTAINTY)

        if texture_img is None or certainty_depth is None:
            self.log_warning("Missing terrain texture or certainty — skipping refinement")
            return context

        task = self.create_progress(4, "Terrain Texture Refinement…")

        texture_pil = texture_img.image.convert("RGB")
        certainty = certainty_depth.depth  # (H, W) float32 in [0, 1]

        # ── Build inpainting mask ─────────────────────────────────────────────
        mask_arr = (certainty < cfg.certainty_threshold).astype(np.uint8)
        if cfg.dilation_px > 0:
            mask_arr = binary_dilation(mask_arr, iterations=cfg.dilation_px).astype(np.uint8)
        mask_pil = PIL.Image.fromarray(mask_arr * 255, "L")

        n_masked = int(mask_arr.sum())
        total_px = mask_arr.size
        self.log_info(
            f"Inpainting {n_masked}/{total_px} ({100*n_masked/total_px:.1f}%) "
            f"low-certainty terrain texels"
        )
        self.advance_progress(task)

        # ── Diffusion inpainting ──────────────────────────────────────────────
        prompt = self._build_prompt(context)

        if self._inpainter is None:
            inpaint_device, inpaint_dtype = preferred_device(DeviceStrategy.MEMORY)
            self._inpainter = InPainting(inpaint_device, inpaint_dtype, cfg.inpainting_type)

        inpainted_pil = self._inpainter.inpaint(
            input_image=texture_pil,
            mask_image=mask_pil,
            temp_path=self.temp,
            prompt=prompt,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            seed=self.seed,
        )
        # Free the inpainter's GPU memory before loading the supersampler —
        # both models together exceed available VRAM.
        self._inpainter.close()
        self._inpainter = None
        self.advance_progress(task)

        # ── Soft blend at boundary (smoothstep certainty as alpha) ───────────
        # Gaussian-smooth certainty to remove jagged observation-mask boundaries,
        # then apply smoothstep so the transition feathers naturally without
        # visible linear ramps.
        cert_smooth = gaussian_filter(certainty, sigma=6.0).clip(0.0, 1.0)
        t = cert_smooth[:, :, np.newaxis]  # (H, W, 1)
        t = t * t * (3.0 - 2.0 * t)       # smoothstep
        tex_arr     = np.array(texture_pil,   dtype=np.float32)
        inp_arr     = np.array(inpainted_pil.convert("RGB"), dtype=np.float32)
        blended_arr = t * tex_arr + (1.0 - t) * inp_arr
        blended_pil = PIL.Image.fromarray(np.clip(blended_arr, 0, 255).astype(np.uint8), "RGB")
        self.advance_progress(task)

        # ── Supersample 1k → 2k ──────────────────────────────────────────────
        if self._sampler is None:
            ss_device, _ = preferred_device(DeviceStrategy.MEMORY)
            self._sampler = ImageSupersampling(ss_device)

        supersampled = self._sampler.supersample(Image(blended_pil), self.temp)
        self.log_info(
            f"Terrain texture refined: {supersampled.image.width}×{supersampled.image.height}"
        )
        self.advance_progress(task)

        context.add_image(ContextKey.TERRAIN_TEXTURE, supersampled)

        if self.temp is not None:
            blended_pil.save(self.temp / "terrain_texture_inpainted.png")
            supersampled.image.save(self.temp / "terrain_texture_supersampled.png")

        self.finish_progress(task)
        return context

    def _build_prompt(self, context: PipelineContext) -> str:
        caption = context.input_object(ContextKey.INPUT_CAPTION)
        base = caption if isinstance(caption, str) and caption else "outdoor natural scene"
        return (
            f"{base}, seamless aerial ground texture, top-down orthographic view, "
            "natural terrain surface, consistent with surrounding area, "
            "photorealistic, no shadows, no people, no vehicles"
        )

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.has_stage_output(ContextKey.TERRAIN_TEXTURE)

    def model_names(self) -> list[str]:
        return [
            *InPainting.model_names(self.config.inpainting_type),
            *ImageSupersampling.model_names(),
        ]

    def clean_up(self):
        if self._inpainter is not None:
            self._inpainter.close()
            self._inpainter = None
        if self._sampler is not None:
            self._sampler.close()
            self._sampler = None
        super().clean_up()
