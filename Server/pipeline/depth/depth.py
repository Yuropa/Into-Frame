from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.depth.image_depth import ImageDepth
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.depth_utils import Depth
from scene.camera import CameraIntrinsics, CameraExtrinsics

class DepthStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._depth = None

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.INPUT, 
            SemanticKey.OUTPUT: ContextKey.DEPTH,
            SemanticKey.INTRINSICS: ContextKey.INTRINSICS,
            SemanticKey.EXTRINSICS: ContextKey.EXTRINSICS,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        depth_task = self.create_progress(2, "Depth...")
        if self._depth is None:
            self._depth = ImageDepth(self.device)
        self.advance_progress(depth_task)

        input_key, output_key, intrinsics_key, extrinsics_key = self._resolved_keys()

        input_image = context.input_image(input_key)
        if input_image is not None:
            result = self._depth.depth(input_image, self.temp)
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

            depth.save_debug_image(self.temp / "depth.png")

            self.log_info(f"Scene depth {depth.min()} to {depth.max()}")

            context.add_depth(output_key, depth)
            context.add_intrinsics(intrinsics_key, intrinsics)
            context.add_extrinsics(extrinsics_key, extrinsics)

        self.advance_progress(depth_task)
        self.finish_progress(depth_task)

        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, output_key, intrinsics_key, extrinsics_key = self._resolved_keys()
        return (
            context.depth(output_key) is not None and
            context.intrinsics(intrinsics_key) is not None and
            context.extrinsics(extrinsics_key) is not None
        )

    def model_names(self) -> list[str]:
        return ImageDepth.model_names()

    def clean_up(self):
        super().clean_up()
        self._depth = None