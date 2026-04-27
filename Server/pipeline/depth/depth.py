from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.depth.image_depth import ImageDepth
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.depth_utils import Depth
from scene.camera import CameraIntrinsics, CameraExtrinsics

class DepthStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._depth = None

    def run(self, context: PipelineContext) -> PipelineContext:
        depth_task = self.create_progress(2, "Depth...")
        if self._depth is None:
            self._depth = ImageDepth(self.device)
        self.advance_progress(depth_task)

        input_image = context.input_image(ContextKey.INPUT)
        if input_image is not None:
            result = self._depth.depth(input_image, self.config.temp)
            depth = Depth(result.depth)

            intrinsics = CameraIntrinsics.from_depth_anything(
                result.intrinsics, 
                color_width=input_image.width, 
                color_height=input_image.height,
                depth_width=depth.width,
                depth_height=depth.height
            )

            extrinsics = CameraExtrinsics.from_depth_anything(
                result.extrinsics
            )

            self.log_info(f"Scene depth {depth.min()} to {depth.max()}")

            context.add_depth(ContextKey.DEPTH, depth)
            context.add_intrinsics(ContextKey.INTRINSICS, intrinsics)
            context.add_extrinsics(ContextKey.EXTRINSICS, extrinsics)

        self.advance_progress(depth_task)
        self.finish_progress(depth_task)

        return context

    def model_names(self) -> list[str]:
        return ImageDepth.model_names()

    def clean_up(self):
        super().clean_up()
        self._depth = None