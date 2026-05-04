from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.panorama.image_panorama import ImagePanorama
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.depth_utils import Depth
from util.device_utils import preferred_device, DeviceStrategy
from scene.camera import CameraIntrinsics, CameraExtrinsics

class PanoramaStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._pano = None
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)

    def run(self, context: PipelineContext) -> PipelineContext:
        pano_task = self.create_progress(2, "Panorama...")
        if self._pano is None:
            self._pano = ImagePanorama(self.preferred_device)
        self.advance_progress(pano_task)
        intrinsics = context.intrinsics(ContextKey.INTRINSICS)

        input_image = context.input_image(ContextKey.INPUT)
        if input_image is not None:
            pano = self._pano.pano(input_image.rgb(), self.config.temp, intrinsics.fov)
            context.add_image(ContextKey.PANORAMA, pano)

        self.advance_progress(pano_task)
        self.finish_progress(pano_task)

        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return (
            context.depth(ContextKey.PANORAMA) is not None
        )

    def model_names(self) -> list[str]:
        return ImagePanorama.model_names()

    def clean_up(self):
        super().clean_up()
        self._pano = None