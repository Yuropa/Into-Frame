from typing import Any, Optional
from logging import Logger

import numpy as np
import PIL.Image
import torch
from scipy.ndimage import binary_dilation, zoom

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.inpainting.inpainting import InPainting, InPaintingType
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
        tile_size: int = 512,
        seam_blend_fraction: float = 0.15,
        certainty_threshold: float = 0.2,
        dilation_px: int = 8,
        inpainting_type: str = "FLUX",
        num_inference_steps: int = 30,
        guidance_scale: float = 30.0,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.tile_size = tile_size
        self.seam_blend_fraction = seam_blend_fraction
        self.certainty_threshold = certainty_threshold
        self.dilation_px = dilation_px
        self.inpainting_type = InPaintingType[inpainting_type]
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale


class TerrainTextureRefinementStage(PipelineStage):
    """
    Refines the baked terrain texture into a small, seamlessly tileable tile.

    Strategy
    --------
    1. Resize the panorama-baked reference to *tile_size* × *tile_size*.
    2. Circularly shift by (tile_size//2, tile_size//2) so the original edges
       meet in the centre of the image.
    3. Build an inpainting mask that covers:
         • the seam band in the centre (where the shifted edges now sit)
         • any low-certainty texels from the bake stage (nadir dead-zone,
           grazing-horizon pixels, unobserved heightmap cells)
    4. Run FLUX inpainting once.  The unmasked high-certainty content
       surrounding the mask gives FLUX the visual context it needs to match
       colour / material.
    5. Shift back by (-tile_size//2, -tile_size//2).  The repaired centre
       seam is now at the edges, which tile seamlessly.

    The resulting small tile is stored back in TERRAIN_TEXTURE.  The terrain
    mesh stage reads *texture_tile_factor* from its config and multiplies the
    UV coordinates by that factor so the tile repeats across the full grid at
    high texel density (e.g. tile_factor=8 on a 100 m grid → one tile per
    12.5 m, ~4 cm/px at 512 px/tile vs ~10 cm/px for a 1 : 1 map).

    Reads:
      ContextKey.TERRAIN_TEXTURE           — baked colour image
      ContextKey.TERRAIN_TEXTURE_CERTAINTY — per-texel certainty [0, 1]
      ContextKey.INPUT_CAPTION             — optional scene caption for prompt

    Writes:
      ContextKey.TERRAIN_TEXTURE           — tileable colour tile (tile_size²)
    """

    @classmethod
    def config_class(cls) -> type[TerrainTextureRefinementConfiguration]:
        return TerrainTextureRefinementConfiguration

    def __init__(self, config: TerrainTextureRefinementConfiguration) -> None:
        super().__init__(config)
        self._inpainter: Optional[InPainting] = None

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: TerrainTextureRefinementConfiguration = self.config

        texture_img = context.input_image(ContextKey.TERRAIN_TEXTURE)
        certainty_depth = context.input_depth(ContextKey.TERRAIN_TEXTURE_CERTAINTY)

        if texture_img is None:
            self.log_warning("Missing terrain texture — skipping refinement")
            return context

        task = self.create_progress(3, "Terrain Texture Refinement…")
        T = cfg.tile_size

        # ── Resize reference and certainty to tile_size ───────────────────────
        ref_pil = texture_img.image.convert("RGB").resize((T, T), PIL.Image.LANCZOS)
        ref_arr = np.array(ref_pil, dtype=np.uint8)

        if certainty_depth is not None:
            src_h, src_w = certainty_depth.depth.shape
            cert_arr = zoom(
                certainty_depth.depth,
                (T / src_h, T / src_w),
                order=1,
            ).astype(np.float32)
        else:
            cert_arr = None

        # ── Circular shift: original edges now meet in the centre ─────────────
        half = T // 2
        shifted_ref = np.roll(np.roll(ref_arr, half, axis=0), half, axis=1)
        if cert_arr is not None:
            shifted_cert = np.roll(np.roll(cert_arr, half, axis=0), half, axis=1)
        else:
            shifted_cert = None

        # ── Build inpainting mask ─────────────────────────────────────────────
        # Covers two things:
        #   a) seam cross: the horizontal + vertical band where shifted edges meet
        #   b) low-certainty texels from the original bake
        mask = np.zeros((T, T), dtype=np.uint8)

        seam_half = max(1, int(T * cfg.seam_blend_fraction / 2))
        cx = half  # centre after shift
        # Horizontal band
        mask[max(0, cx - seam_half) : min(T, cx + seam_half), :] = 255
        # Vertical band
        mask[:, max(0, cx - seam_half) : min(T, cx + seam_half)] = 255

        if shifted_cert is not None:
            mask[shifted_cert < cfg.certainty_threshold] = 255

        if cfg.dilation_px > 0:
            mask = binary_dilation(mask, iterations=cfg.dilation_px).astype(np.uint8) * 255

        mask_pil = PIL.Image.fromarray(mask, "L")

        n_masked = int((mask > 0).sum())
        self.log_info(
            f"Terrain texture refinement: {n_masked}/{T*T} "
            f"({100*n_masked/(T*T):.1f}%) texels masked (seam + low-certainty)"
        )
        self.advance_progress(task)

        # ── FLUX inpainting ───────────────────────────────────────────────────
        prompt = self._build_prompt(context)

        if self._inpainter is None:
            inpaint_device, inpaint_dtype = preferred_device(DeviceStrategy.MEMORY)
            self._inpainter = InPainting(inpaint_device, inpaint_dtype, cfg.inpainting_type)

        inpainted = self._inpainter.inpaint(
            input_image=PIL.Image.fromarray(shifted_ref),
            mask_image=mask_pil,
            temp_path=self.temp,
            prompt=prompt,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            seed=self.seed,
        )

        self._inpainter.close()
        self._inpainter = None
        self.advance_progress(task)

        # ── Shift back: repaired seam is now at the edges ─────────────────────
        result_arr = np.array(inpainted.convert("RGB"), dtype=np.uint8)
        result_arr = np.roll(np.roll(result_arr, -half, axis=0), -half, axis=1)
        result_pil = PIL.Image.fromarray(result_arr)

        context.add_image(ContextKey.TERRAIN_TEXTURE, Image(result_pil))

        if self.temp is not None:
            PIL.Image.fromarray(shifted_ref).save(self.temp / "terrain_texture_shifted_input.png")
            PIL.Image.fromarray(mask, "L").save(self.temp / "terrain_texture_seam_mask.png")
            result_pil.save(self.temp / "terrain_texture_tileable.png")

        self.log_info(
            f"Terrain texture refined: {result_pil.width}×{result_pil.height} tileable tile"
        )
        self.advance_progress(task)
        self.finish_progress(task)
        return context

    def _build_prompt(self, context: PipelineContext) -> str:
        caption = context.input_object(ContextKey.INPUT_CAPTION)
        base = caption if isinstance(caption, str) and caption else "outdoor natural scene"
        return (
            f"{base}, seamless tileable aerial top-down terrain texture, "
            "natural ground surface, consistent material throughout, "
            "photorealistic, no shadows, no people, no vehicles, no seams"
        )

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.has_stage_output(ContextKey.TERRAIN_TEXTURE)

    def model_names(self) -> list[str]:
        return InPainting.model_names(self.config.inpainting_type)

    def clean_up(self):
        if self._inpainter is not None:
            self._inpainter.close()
            self._inpainter = None
        super().clean_up()
