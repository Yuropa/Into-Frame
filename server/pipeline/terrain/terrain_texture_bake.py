from typing import Any, Optional
from logging import Logger

import numpy as np
import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.terrain.terrain_generator import TerrainMeshGenerator
from util.depth_utils import Depth
from util.image_utils import Image


class TerrainTextureBakeConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        tex_size: int = 1024,
        nadir_cutoff_deg: float = -35.0,
        nadir_fade_deg: float = 10.0,
        horizon_fade_deg: float = 5.0,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.tex_size = tex_size
        self.nadir_cutoff_deg = nadir_cutoff_deg
        self.nadir_fade_deg = nadir_fade_deg
        self.horizon_fade_deg = horizon_fade_deg


class TerrainTextureBakeStage(PipelineStage):
    """
    Bakes a top-down panorama texture and a per-texel certainty map.

    The certainty map encodes how reliable each texel's colour is based on
    the equirectangular sampling latitude and heightmap observation coverage.
    Low-certainty regions (nadir dead-zone, grazing-horizon pixels, and
    unobserved heightmap cells) are marked for downstream inpainting.

    Reads:
      ContextKey.PANORAMA              — equirectangular panorama (required)
      ContextKey.HEIGHT_MAP            — terrain height grid (required)
      ContextKey.HEIGHT_MAP_PARAMS     — grid_size_meters (optional)
      ContextKey.HEIGHT_MAP_CERTAINTY  — per-cell observation certainty (optional)

    Writes:
      ContextKey.TERRAIN_TEXTURE           — baked colour image (Image, tex_size²)
      ContextKey.TERRAIN_TEXTURE_CERTAINTY — per-texel certainty (Depth, [0,1])
    """

    @classmethod
    def config_class(cls) -> type[TerrainTextureBakeConfiguration]:
        return TerrainTextureBakeConfiguration

    def __init__(self, config: TerrainTextureBakeConfiguration) -> None:
        super().__init__(config)

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: TerrainTextureBakeConfiguration = self.config

        panorama = context.input_panorama(ContextKey.PANORAMA)
        if panorama is None:
            self.log_warning("No panorama found — skipping terrain texture bake")
            return context

        height_map = context.input_depth(ContextKey.HEIGHT_MAP)
        if height_map is None:
            self.log_warning("No height map found — skipping terrain texture bake")
            return context

        params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS)
        grid_size = (params.get("grid_size_meters") if params else None) or 100.0
        x_half = grid_size / 2.0
        z_far = grid_size / 2.0

        height_certainty_depth = context.input_depth(ContextKey.HEIGHT_MAP_CERTAINTY)
        height_certainty = height_certainty_depth.depth if height_certainty_depth is not None else None

        sky_mask = context.input_object(ContextKey.PANORAMA_SKY_MASK)
        if isinstance(sky_mask, list):
            sky_mask = np.array(sky_mask, dtype=bool)

        task = self.create_progress(2, "Terrain Texture Bake…")

        color, certainty = TerrainMeshGenerator.bake_topdown_texture_with_certainty(
            panorama=panorama,
            height_map=height_map.depth,
            x_half=x_half,
            z_far=z_far,
            tex_size=cfg.tex_size,
            height_certainty=height_certainty,
            nadir_cutoff_deg=cfg.nadir_cutoff_deg,
            nadir_fade_deg=cfg.nadir_fade_deg,
            horizon_fade_deg=cfg.horizon_fade_deg,
            sky_mask=sky_mask,
        )
        self.advance_progress(task)

        context.add_image(ContextKey.TERRAIN_TEXTURE, Image(color))
        context.add_depth(ContextKey.TERRAIN_TEXTURE_CERTAINTY, Depth(certainty))

        if self.temp is not None:
            color.save(self.temp / "terrain_texture.png")
            Depth(certainty).save_debug_image(self.temp / "terrain_texture_certainty.png")

        low_cert_pct = float((certainty < 0.2).mean()) * 100.0
        self.log_info(
            f"Terrain texture baked: {cfg.tex_size}×{cfg.tex_size}, "
            f"{low_cert_pct:.1f}% low-certainty (will be inpainted)"
        )

        self.advance_progress(task)
        self.finish_progress(task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.has_stage_output(ContextKey.TERRAIN_TEXTURE)

    def model_names(self) -> list[str]:
        return []

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        texture = context.image(ContextKey.TERRAIN_TEXTURE)
        if texture is None:
            return None
        cfg: TerrainTextureBakeConfiguration = self.config
        return ReportSection(
            stage_name=self.name,
            title="Terrain Texture Bake",
            body=(
                "A top-down terrain texture was baked from the equirectangular panorama "
                "using an orthographic projection aligned with the height map grid. "
                "A per-texel certainty map encodes reliability based on equirectangular "
                "sampling latitude, heightmap observation coverage, and proximity to the "
                "nadir dead-zone. Low-certainty regions are flagged for inpainting in a "
                "subsequent refinement stage."
            ),
            images=[(texture.image, "Baked top-down terrain texture")],
            stats={"Texture resolution": f"{cfg.tex_size} × {cfg.tex_size} px"},
        )
