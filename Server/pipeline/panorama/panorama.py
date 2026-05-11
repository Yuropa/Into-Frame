from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.panorama.image_panorama import ImagePanorama
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.depth_utils import Depth
from scene.camera import CameraIntrinsics, CameraExtrinsics

class PanoramaStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._pano = None

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.INPUT, 
            SemanticKey.OUTPUT: ContextKey.PANORAMA,
            SemanticKey.CUBEMAP: ContextKey.PANAORAMA_CUBENAME,
            SemanticKey.CAPTION: ContextKey.INPUT_CAPTION,
            SemanticKey.INTRINSICS: ContextKey.INTRINSICS
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        pano_task = self.create_progress(2, "Panorama...")

        input_key, output_key, cubemap_key, caption_key, intrinsics_key = self._resolved_keys()

        if self._pano is None:
            self._pano = ImagePanorama(self.device)
        self.advance_progress(pano_task)
        intrinsics = context.intrinsics(intrinsics_key)
        caption = context.object(caption_key)

        input_image = context.input_image(input_key)
        if input_image is not None:
            pano = self._pano.pano(input_image.rgb(), self.temp, intrinsics.fov, caption)
            context.add_image(output_key, pano.image)
            context.add_cubemap(cubemap_key, pano.cubemap)

        self.advance_progress(pano_task)
        self.finish_progress(pano_task)

        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, output_key, cubemap_key, _, _ = self._resolved_keys()
        return (
            context.depth(output_key) is not None and
            context.intrinsics(cubemap_key) is not None
        )

    def model_names(self) -> list[str]:
        return ImagePanorama.model_names()

    def clean_up(self):
        super().clean_up()
        self._pano = None