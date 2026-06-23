import torch
import numpy as np
from typing import Any
from logging import Logger

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.linear_structures.detector import LinearStructureDetector
from pipeline.panorama_segmentation.panorama_region_result import RegionType


class LinearStructureConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        *args,
        modify_rivers: bool = True,
        modify_roads: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.modify_rivers = modify_rivers
        self.modify_roads  = modify_roads


class LinearStructureStage(PipelineStage):
    """
    Converts pre-computed semantic skeletons into a LinearGraph of world-space
    polylines, then applies terrain modifications to the height map.

    Rivers come from ContextKey.WATER_SKELETON (medial axis of water cells,
    produced by RegionMapStage from semantic segmentation).

    Roads/trails: no reliable semantic source yet — omitted until a dedicated
    road segmentation stage is available.

    Input:
      ContextKey.HEIGHT_MAP        (Depth)
      ContextKey.HEIGHT_MAP_PARAMS (dict)
      ContextKey.WATER_SKELETON    (Depth, binary skeleton grid)
      ContextKey.REGION_MAP        (Depth, uint8 type-index grid, for width estimation)

    Output:
      ContextKey.LINEAR_GRAPH  (LinearGraph)
      ContextKey.HEIGHT_MAP    (Depth, modified)
    """

    @classmethod
    def config_class(cls) -> type[LinearStructureConfiguration]:
        return LinearStructureConfiguration

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: LinearStructureConfiguration = self.config

        task = self.create_progress(3, "Linear Structures...")

        height_map = context.input_depth(ContextKey.HEIGHT_MAP)
        params     = context.input_object(ContextKey.HEIGHT_MAP_PARAMS)

        if height_map is None:
            self.log_warning("No height map — skipping linear structure detection")
            self.finish_progress(task)
            return context

        water_skeleton_depth = context.input_depth(ContextKey.WATER_SKELETON)
        region_map_depth     = context.input_depth(ContextKey.REGION_MAP)

        water_skeleton = water_skeleton_depth.depth if water_skeleton_depth is not None else None
        water_mask = None
        if region_map_depth is not None:
            water_mask = region_map_depth.depth.astype(np.uint8) == int(RegionType.WATER)

        self.advance_progress(task)

        graph = LinearStructureDetector.detect(
            height_map=height_map,
            params=params or {},
            water_skeleton=water_skeleton,
            water_mask=water_mask,
        )
        self.advance_progress(task)

        self.log_info(f"Detected: {graph.summary()}")
        context.add_object(ContextKey.LINEAR_GRAPH, graph)

        modified_hm = LinearStructureDetector.modify_height_map(
            height_map=height_map,
            params=params or {},
            graph=graph,
            modify_rivers=cfg.modify_rivers,
            modify_roads=cfg.modify_roads,
        )
        context.add_depth(ContextKey.HEIGHT_MAP, modified_hm)

        if self.temp is not None:
            modified_hm.save_debug_image(self.temp / "heightmap_modified.png")

        self.finish_progress(task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(ContextKey.LINEAR_GRAPH) is not None

    def model_names(self) -> list[str]:
        return []
