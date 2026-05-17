from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.panorama.image_panorama import ImagePanorama, PanoramaGeneratorType
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.depth_utils import Depth
from util.device_utils import DeviceStrategy, preferred_device
from scene.camera import CameraIntrinsics, CameraExtrinsics

class PanoramaStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._pano = None

        strategy = DeviceStrategy.AUTO
        if PanoramaGeneratorType.default() == PanoramaGeneratorType.DREAMCUBE or PanoramaGeneratorType.default() == PanoramaGeneratorType.FLUX:
            strategy = DeviceStrategy.MEMORY
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
        pano_task = self.create_progress(2, "Panorama...")

        input_key, depth_key, output_key, cubemap_key, caption_key, intrinsics_key = self._resolved_keys()

        if self._pano is None:
            self._pano = ImagePanorama(self.preferred_device)
        self.advance_progress(pano_task)
        intrinsics = context.intrinsics(intrinsics_key)
        caption = context.object(caption_key)

        input_image = context.input_image(input_key)
        depth_image = context.input_depth(depth_key)
        if input_image is not None:
            if caption is None:
                caption = ""
            full_caption = "360 view. seamless. outdoor scene. clear in-focus. smooth background. open air. 360 degree panorama. dirt ground. " + caption

            pano = self._pano.pano(input_image.rgb(), depth_image.gray(), self.temp, intrinsics.fov, full_caption)
            context.add_image(output_key, pano.image)
            context.add_cubemap(cubemap_key, pano.cubemap)

        self.advance_progress(pano_task)
        self.finish_progress(pano_task)

        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, _, output_key, cubemap_key, _, _ = self._resolved_keys()
        return (
            context.image(output_key) is not None and
            context.cubemap(cubemap_key) is not None and 
            False
        )

    def model_names(self) -> list[str]:
        return ImagePanorama.model_names()

    def clean_up(self):
        super().clean_up()
        self._pano = None