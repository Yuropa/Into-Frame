from typing import Any
from logging import Logger
import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.panorama.image_panorama import ImagePanorama, PanoramaGeneratorType
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.depth_utils import Depth
from util.device_utils import DeviceStrategy, preferred_device
from scene.camera import CameraIntrinsics, CameraExtrinsics

_MEMORY_STRATEGY_TYPES = {
    PanoramaGeneratorType.DREAMCUBE,
    PanoramaGeneratorType.FLUX,
    PanoramaGeneratorType.WORLDGEN,
}


class PanoramaConfiguration(PipelineStageConfiguration):
    """
    Configuration for PanoramaStage.

    generator_type: one of CUBEDIFF, DREAMCUBE, FLUX (default), WORLDGEN.
    WORLDGEN extends the input image into a full panorama using WorldGen's depth model and
    FLUX.1-Fill LoRA — no text prompt is used.
    """

    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        generator_type: str = "FLUX",
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.generator_type = PanoramaGeneratorType[generator_type]


class PanoramaStage(PipelineStage):
    """
    Generates a 360° equirectangular panorama and cubemap from the input image,
    guided by the caption and depth map.

    Input key      (SemanticKey.INPUT)      → ContextKey.INPUT             (Image)
    Depth key      (SemanticKey.DEPTH)      → ContextKey.DEPTH             (Depth)
    Caption key    (SemanticKey.CAPTION)    → ContextKey.INPUT_CAPTION     (str)
    Intrinsics key (SemanticKey.INTRINSICS) → ContextKey.INTRINSICS        (CameraIntrinsics)
    Output key     (SemanticKey.OUTPUT)     → ContextKey.PANORAMA          (Image, equirectangular)
    Cubemap key    (SemanticKey.CUBEMAP)    → ContextKey.PANAORAMA_CUBENAME (CubeMap)

    Set generator_type in config.yaml to select the backend:
      FLUX (default), CUBEDIFF, DREAMCUBE, WORLDGEN
    """

    @classmethod
    def config_class(cls) -> type[PanoramaConfiguration]:
        return PanoramaConfiguration

    def __init__(self, config: PanoramaConfiguration) -> None:
        super().__init__(config)
        self._pano = None

        strategy = DeviceStrategy.MEMORY if config.generator_type in _MEMORY_STRATEGY_TYPES else DeviceStrategy.AUTO
        self.preferred_device, _ = preferred_device(strategy)

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.INPUT, 
            SemanticKey.DEPTH: ContextKey.DEPTH,
            SemanticKey.OUTPUT: ContextKey.PANORAMA,
            SemanticKey.CUBEMAP: ContextKey.PANAORAMA_CUBENAME,
            SemanticKey.CAPTION: ContextKey.INPUT_CAPTION,
            SemanticKey.INTRINSICS: ContextKey.INTRINSICS
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        pano_task = self.create_progress(2, "Panorama…")

        input_key, depth_key, output_key, cubemap_key, caption_key, intrinsics_key = self._resolved_keys()

        generator_type = self.config.generator_type
        if self._pano is None:
            self._pano = ImagePanorama(self.preferred_device, type=generator_type)
        self.advance_progress(pano_task)

        intrinsics = context.intrinsics(intrinsics_key)
        caption = context.object(caption_key) or ""
        input_image = context.input_image(input_key)
        depth_image = context.input_depth(depth_key)

        if input_image is not None:
            if generator_type == PanoramaGeneratorType.WORLDGEN:
                full_caption = ""
            else:
                full_caption = "360 view. seamless. outdoor scene. clear in-focus. smooth background. open air. 360 degree panorama. no people." + caption

            fov = intrinsics.fov if intrinsics is not None else 0.0
            depth_gray = depth_image.gray() if depth_image is not None else None

            pano = self._pano.pano(input_image.rgb(), depth_gray, self.temp, fov, full_caption, seed=self.seed, on_progress=self.make_progress_callback(pano_task))
            context.add_panorama(output_key, pano.panorama)
            context.add_cubemap(cubemap_key, pano.cubemap)

        self.advance_progress(pano_task)
        self.finish_progress(pano_task)

        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, _, output_key, cubemap_key, _, _ = self._resolved_keys()
        return (
            context.panorama(output_key) is not None and
            context.cubemap(cubemap_key) is not None
        )

    def model_names(self) -> list[str]:
        return ImagePanorama.model_names(self.config.generator_type)

    def clean_up(self):
        if self._pano is not None:
            self._pano.close()
            self._pano = None
        super().clean_up()