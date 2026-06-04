from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.world_gen.world_gen import WorldGen
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.image_utils import Image
from util.device_utils import preferred_device, DeviceStrategy
from scene.camera import CameraIntrinsics, CameraExtrinsics

class WorldGenStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._gen = None
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)

    def run(self, context: PipelineContext) -> PipelineContext:
        gen_task = self.create_progress(2, "WorldGen...")
        if self._gen is None:
            self._gen = WorldGen(self.preferred_device)
        self.advance_progress(gen_task)
        input_image = context.input_image(ContextKey.INPUT)
        if input_image is not None:
            pano = self._gen.generate(input_image, self.temp)
            context.add_image(ContextKey.PANORAMA, pano)

        self.advance_progress(gen_task)
        self.finish_progress(gen_task)

        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return False # context.image(ContextKey.PANORAMA) is not None

    def model_names(self) -> list[str]:
        return WorldGen.model_names()

    def clean_up(self):
        super().clean_up()
        self._gen = None