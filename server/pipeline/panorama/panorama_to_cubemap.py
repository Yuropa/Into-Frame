from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from typing import Any
from logging import Logger
import torch


class PanoramaToCubemapConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        face_w: int = 512,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.face_w = face_w


class PanoramaToCubemapStage(PipelineStage):
    """
    Reprojects the equirectangular panorama in context into a cubemap.

    Useful after any stage that modifies the panorama (e.g. supersampling)
    so that the cubemap stays in sync.

    Input key  (SemanticKey.PANORAMA) → ContextKey.PANORAMA
    Output key (SemanticKey.CUBEMAP)  → ContextKey.PANAORAMA_CUBENAME

    face_w (config): cubemap face size in pixels (default 512).
    """

    @classmethod
    def config_class(cls):
        return PanoramaToCubemapConfiguration

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.PANORAMA: ContextKey.PANORAMA,
            SemanticKey.CUBEMAP:  ContextKey.PANAORAMA_CUBENAME,
        })

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, cubemap_key = self._resolved_keys()
        return context.has_stage_output(cubemap_key)

    def run(self, context: PipelineContext) -> PipelineContext:
        task = self.create_progress(1, "Rebuilding cubemap…")

        panorama_key, cubemap_key = self._resolved_keys()
        panorama = context.input_panorama(panorama_key)

        if panorama is not None:
            cubemap = panorama.to_cubemap(face_w=self.config.face_w)
            context.add_cubemap(cubemap_key, cubemap)

            if self.output is not None:
                cubemap.save(self.output / cubemap_key)

        self.finish_progress(task)
        return context
