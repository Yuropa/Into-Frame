from typing import Any
from logging import Logger

import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.depth_utils import Depth


class GroundMapConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)


class GroundMapStage(PipelineStage):
    """
    Fuses the height map and region map using their certainty maps, then writes
    refined versions back so all downstream stages (terrain mesh, object placement)
    work from the best available ground-plane estimates.

    Currently a pass-through: reads the four maps and writes them back unchanged
    under the same keys.  The refinement logic (e.g. certainty-weighted inpainting,
    cross-map consistency, uncertainty propagation) will be implemented here.

    Reads:  ContextKey.HEIGHT_MAP           (Depth, metres)
            ContextKey.HEIGHT_MAP_CERTAINTY  (Depth, [0, 1])
            ContextKey.REGION_MAP            (Depth, uint8 type indices)
            ContextKey.REGION_MAP_CERTAINTY  (Depth, [0, 1])

    Writes: ContextKey.HEIGHT_MAP           (Depth, shadowing the original)
            ContextKey.HEIGHT_MAP_CERTAINTY  (Depth, shadowing the original)
            ContextKey.REGION_MAP            (Depth, shadowing the original)
            ContextKey.REGION_MAP_CERTAINTY  (Depth, shadowing the original)
    """

    @classmethod
    def config_class(cls) -> type[GroundMapConfiguration]:
        return GroundMapConfiguration

    def __init__(self, config: GroundMapConfiguration) -> None:
        super().__init__(config)

    def run(self, context: PipelineContext) -> PipelineContext:
        height_map = context.input_depth(ContextKey.HEIGHT_MAP)
        height_certainty = context.input_depth(ContextKey.HEIGHT_MAP_CERTAINTY)
        region_map = context.input_depth(ContextKey.REGION_MAP)
        region_certainty = context.input_depth(ContextKey.REGION_MAP_CERTAINTY)

        if height_map is None:
            self.log_warning("No height map — skipping ground map fusion")
            return context

        if region_map is None:
            self.log_warning("No region map — skipping ground map fusion")
            return context

        # TODO: implement certainty-guided refinement here.
        # For now, pass through the maps unchanged so downstream stages
        # (TerrainMeshStage, object distribution/placement) read from this
        # stage's slot rather than the original generator slots.
        context.add_depth(ContextKey.HEIGHT_MAP, height_map)
        context.add_depth(ContextKey.REGION_MAP, region_map)

        if height_certainty is not None:
            context.add_depth(ContextKey.HEIGHT_MAP_CERTAINTY, height_certainty)
        if region_certainty is not None:
            context.add_depth(ContextKey.REGION_MAP_CERTAINTY, region_certainty)

        self.log_info(
            f"Ground map fusion: height {height_map.depth.shape}, "
            f"region {region_map.depth.shape}"
        )
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return (
            context.depth(ContextKey.HEIGHT_MAP) is not None
            and context.depth(ContextKey.REGION_MAP) is not None
        )

    def model_names(self) -> list[str]:
        return []
