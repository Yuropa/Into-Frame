from typing import Any, Optional
from logging import Logger

import numpy as np
import PIL.Image
import torch
from scipy.ndimage import binary_dilation, gaussian_filter, zoom

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.panorama_segmentation.panorama_region_result import RegionType
from util.image_utils import Image
from util.device_utils import DeviceStrategy, preferred_device


_GROUND_TYPES: frozenset[RegionType] = frozenset({
    RegionType.GROUND,
    RegionType.TERRAIN,
    RegionType.VEGETATION,
    RegionType.WATER,
    RegionType.ROAD,
    RegionType.TRAIL,
    RegionType.BUILT,
})

_BASE_PROMPTS: dict[RegionType, str] = {
    RegionType.GROUND:      "natural earth and soil ground surface, dirt and pebbles",
    RegionType.TERRAIN:     "rocky mountain terrain, stone and gravel surface",
    RegionType.VEGETATION:  "lush green grass meadow, small plants and wildflowers",
    RegionType.WATER:       "calm shallow water surface, lake or river bed",
    RegionType.ROAD:        "asphalt road surface, weathered pavement and tarmac",
    RegionType.TRAIL:       "dirt hiking trail, packed earth and fine gravel",
    RegionType.BUILT:       "concrete and stone pavement, urban ground tiles",
}

_TILE_SUFFIX = (
    ", seamless tileable top-down aerial texture, photorealistic, "
    "high detail, no shadows, no people, no vehicles, overhead view"
)


class TerrainTextureGenerationConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        tile_size: int = 512,
        output_size: int = 2048,
        tile_factor: int = 8,
        blend_sigma: float = 0.05,
        min_region_fraction: float = 0.02,
        inpainting_type: str = "FLUX",
        num_inference_steps: int = 28,
        guidance_scale: float = 30.0,
        seam_width_fraction: float = 0.08,
        seam_dilation_px: int = 8,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.tile_size = tile_size
        self.output_size = output_size
        self.tile_factor = tile_factor
        self.blend_sigma = blend_sigma
        self.min_region_fraction = min_region_fraction
        self.inpainting_type = InPaintingType[inpainting_type]
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seam_width_fraction = seam_width_fraction
        self.seam_dilation_px = seam_dilation_px


class TerrainTextureGenerationStage(PipelineStage):
    """
    Generates a high-quality terrain texture from scratch for each ground region.

    For each ground region type present in the REGION_MAP:
      1. Generate a photorealistic seamlessly tileable tile using FLUX (two-pass:
         full-mask generation → circular-shift seam inpainting).
      2. Save the individual tile for future Unity per-region blending.

    All tiles are then composited into a single output_size² terrain texture
    by sampling each tile at tile_factor tiling density and blending with
    gaussian-smoothed region weights.  The composite is applied to the mesh
    at UV scale 1:1 — high-frequency detail comes from the tile sampling.

    Reads:
      ContextKey.REGION_MAP     — top-down region type grid (optional)
      ContextKey.INPUT_CAPTION  — scene caption for prompt context (optional)

    Writes:
      ContextKey.TERRAIN_TEXTURE            — composite texture (output_size²)
      ContextKey.TERRAIN_TEXTURE_TILES      — dict[str, PIL.Image] per-region tiles
      ContextKey.TERRAIN_TEXTURE_TILE_FACTOR — float hint (1.0) for the mesh stage
    """

    @classmethod
    def config_class(cls) -> type[TerrainTextureGenerationConfiguration]:
        return TerrainTextureGenerationConfiguration

    def __init__(self, config: TerrainTextureGenerationConfiguration) -> None:
        super().__init__(config)
        self._inpainter: Optional[InPainting] = None

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: TerrainTextureGenerationConfiguration = self.config
        caption = context.input_object(ContextKey.INPUT_CAPTION)

        region_map_depth = context.input_depth(ContextKey.REGION_MAP)
        present_types, weights_map = self._analyze_region_map(
            region_map_depth, cfg.output_size, cfg.blend_sigma, cfg.min_region_fraction,
        )
        self.log_info(f"Region types: {[rt.label for rt in present_types]}")

        # Two FLUX calls per region (generate + seam fix), plus two bookkeeping steps
        task = self.create_progress(len(present_types) * 2 + 2, "Terrain Texture Generation…")

        inpaint_device, inpaint_dtype = preferred_device(DeviceStrategy.MEMORY)
        self._inpainter = InPainting(inpaint_device, inpaint_dtype, cfg.inpainting_type)

        tiles: dict[RegionType, np.ndarray] = {}
        for idx, rt in enumerate(present_types):
            prompt = self._build_prompt(rt, caption)
            tile = self._generate_tileable_tile(prompt, cfg, seed_offset=idx)
            tiles[rt] = np.array(tile.convert("RGB"), dtype=np.uint8)
            self.log_info(f"Generated {rt.label} tile ({cfg.tile_size}px, seamless)")
            self.advance_progress(task)
            self.advance_progress(task)

        self._inpainter.close()
        self._inpainter = None
        self.advance_progress(task)

        composite = self._blend_tiles(tiles, weights_map, cfg.output_size, cfg.tile_size, cfg.tile_factor)
        composite_pil = PIL.Image.fromarray(composite)

        context.add_image(ContextKey.TERRAIN_TEXTURE, Image(composite_pil))
        context.add_object(ContextKey.TERRAIN_TEXTURE_TILE_FACTOR, 1.0)
        tiles_pil = {rt.label: PIL.Image.fromarray(arr) for rt, arr in tiles.items()}
        context.add_object(ContextKey.TERRAIN_TEXTURE_TILES, tiles_pil)

        if self.temp is not None:
            composite_pil.save(self.temp / "terrain_texture_composite.png")
            for label, tile_img in tiles_pil.items():
                tile_img.save(self.temp / f"terrain_texture_tile_{label}.png")

        self.log_info(
            f"Terrain texture: {cfg.output_size}×{cfg.output_size} composite from "
            f"{len(tiles)} region tile(s) at ×{cfg.tile_factor} density"
        )
        self.advance_progress(task)
        self.finish_progress(task)
        return context

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _analyze_region_map(
        self,
        region_map_depth,
        output_size: int,
        blend_sigma: float,
        min_fraction: float,
    ) -> tuple[list[RegionType], dict[RegionType, np.ndarray]]:
        if region_map_depth is None:
            uniform = np.ones((output_size, output_size), dtype=np.float32)
            return [RegionType.GROUND], {RegionType.GROUND: uniform}

        rm = region_map_depth.depth.astype(np.float32)
        total = rm.size
        present = [
            rt for rt in _GROUND_TYPES
            if (rm == int(rt)).sum() / total >= min_fraction
        ]
        if not present:
            present = [RegionType.GROUND]

        sigma_px = max(4.0, blend_sigma * output_size)
        weights: dict[RegionType, np.ndarray] = {}
        for rt in present:
            mask = zoom(
                (rm == int(rt)).astype(np.float32),
                (output_size / rm.shape[0], output_size / rm.shape[1]),
                order=1,
            )
            weights[rt] = gaussian_filter(mask, sigma=sigma_px)

        total_w = sum(weights.values()) + 1e-6
        for rt in weights:
            weights[rt] /= total_w

        return present, weights

    def _build_prompt(self, rt: RegionType, caption: Any) -> str:
        base = _BASE_PROMPTS.get(rt, "natural outdoor ground surface")
        prefix = f"{caption}, " if isinstance(caption, str) and caption else ""
        return f"{prefix}{base}{_TILE_SUFFIX}"

    def _generate_tileable_tile(
        self,
        prompt: str,
        cfg: TerrainTextureGenerationConfiguration,
        seed_offset: int,
    ) -> PIL.Image.Image:
        T = cfg.tile_size
        seed = cfg.seed + seed_offset

        # Pass 1: pure FLUX generation — 100% mask on neutral gray = text-to-image
        gray = PIL.Image.new("RGB", (T, T), (128, 128, 128))
        full_mask = PIL.Image.new("L", (T, T), 255)
        generated = self._inpainter.inpaint(
            input_image=gray,
            mask_image=full_mask,
            temp_path=self.temp,
            prompt=prompt,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            seed=seed,
        )

        # Pass 2: circular-shift seam fix
        arr = np.array(generated.convert("RGB"), dtype=np.uint8)
        half = T // 2
        shifted = np.roll(np.roll(arr, half, axis=0), half, axis=1)

        seam_mask = np.zeros((T, T), dtype=np.uint8)
        sw = max(2, int(T * cfg.seam_width_fraction / 2))
        cx = half
        seam_mask[max(0, cx - sw): min(T, cx + sw), :] = 255
        seam_mask[:, max(0, cx - sw): min(T, cx + sw)] = 255
        if cfg.seam_dilation_px > 0:
            seam_mask = (
                binary_dilation(seam_mask, iterations=cfg.seam_dilation_px).astype(np.uint8) * 255
            )

        fixed = self._inpainter.inpaint(
            input_image=PIL.Image.fromarray(shifted),
            mask_image=PIL.Image.fromarray(seam_mask, "L"),
            temp_path=self.temp,
            prompt=prompt,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            seed=seed + 1000,
        )

        # Shift back — the repaired seam is now at the tile edges, which tile seamlessly
        result = np.array(fixed.convert("RGB"), dtype=np.uint8)
        result = np.roll(np.roll(result, -half, axis=0), -half, axis=1)
        return PIL.Image.fromarray(result)

    @staticmethod
    def _blend_tiles(
        tiles: dict[RegionType, np.ndarray],
        weights: dict[RegionType, np.ndarray],
        output_size: int,
        tile_size: int,
        tile_factor: int,
    ) -> np.ndarray:
        # For output pixel i, sample tile pixel: (i / output_size * tile_size * tile_factor) % tile_size
        idx = (
            np.arange(output_size, dtype=np.float32) / output_size * tile_size * tile_factor
        ).astype(int) % tile_size

        composite = np.zeros((output_size, output_size, 3), dtype=np.float64)
        for rt, tile_arr in tiles.items():
            if rt not in weights:
                continue
            w = weights[rt][:, :, np.newaxis]
            sampled = tile_arr[idx[:, None], idx[None, :]]
            composite += sampled.astype(np.float64) * w

        return composite.clip(0, 255).astype(np.uint8)

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.has_stage_output(ContextKey.TERRAIN_TEXTURE)

    def model_names(self) -> list[str]:
        return InPainting.model_names(self.config.inpainting_type)

    def clean_up(self):
        if self._inpainter is not None:
            self._inpainter.close()
            self._inpainter = None
        super().clean_up()

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        texture = context.image(ContextKey.TERRAIN_TEXTURE)
        if texture is None:
            return None
        cfg: TerrainTextureGenerationConfiguration = self.config
        tiles_dict = context.object(ContextKey.TERRAIN_TEXTURE_TILES) or {}
        return ReportSection(
            stage_name=self.name,
            title="Terrain Texture Generation",
            body=(
                "High-quality seamlessly tileable textures were generated for each ground "
                "region type using FLUX inpainting (two-pass: full-mask generation then "
                "circular-shift seam repair). The per-region tiles were composited into a "
                f"single {cfg.output_size}×{cfg.output_size} terrain texture by sampling "
                f"each tile at ×{cfg.tile_factor} density and blending with gaussian-smoothed "
                "region weights. Individual tiles are also saved for runtime Unity blending."
            ),
            images=[(texture.image, "Composited terrain texture")],
            stats={
                "Regions": ", ".join(tiles_dict.keys()) or "ground",
                "Tile size": f"{cfg.tile_size} × {cfg.tile_size} px",
                "Output size": f"{cfg.output_size} × {cfg.output_size} px",
                "Tile density": f"×{cfg.tile_factor}",
            },
        )
