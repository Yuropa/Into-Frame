import torch
from typing import Any
from logging import Logger

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.linear_structures.detector import LinearStructureDetector


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
    Detects roads, rivers, and trails from the panorama and height map, then:
      1. Writes a LinearGraph to ContextKey.LINEAR_GRAPH for downstream stages.
      2. Modifies ContextKey.HEIGHT_MAP in-place (valley carving for rivers,
         smoothing for roads) so TerrainMeshStage sees an updated terrain.

    Detection sources (tried in combination):
      • Panorama colour segmentation (equirectangular, ground hemisphere)
      • Height-map topology (valley detection for rivers)

    Input:
      ContextKey.HEIGHT_MAP        (Depth)
      ContextKey.HEIGHT_MAP_PARAMS (dict)
      ContextKey.PANORAMA          (Panorama, optional)
      ContextKey.PANORAMA_DEPTH    (Depth, optional)

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

        panorama       = context.input_panorama(ContextKey.PANORAMA)
        panorama_depth = context.input_depth(ContextKey.PANORAMA_DEPTH)
        self.advance_progress(task)

        if panorama is not None and panorama_depth is not None:
            self.log_info("Linear structures: using panorama colour + height-map topology")
        else:
            self.log_info("Linear structures: using height-map topology only")

        graph = LinearStructureDetector.detect(
            height_map=height_map,
            params=params or {},
            panorama=panorama,
            panorama_depth=panorama_depth,
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
