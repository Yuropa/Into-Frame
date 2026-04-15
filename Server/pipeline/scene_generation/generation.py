from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from scene.scene import Scene
from scene.object import Object3D
from scene.camera import CameraIntrinsics
from util.depth_utils import Depth
import numpy as np

class SceneGenerationStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._gen = None

    def run(self, context: PipelineContext) -> PipelineContext:
        object_count = context.input_object("count")
        intrinsics = context.input_intrinsics(ContextKey.INTRINSICS)
        depth = context.input_depth(ContextKey.DEPTH)

        scene = Scene()
        generation_task = self.create_progress(object_count, "Creating Objects...")
        for idx in range(object_count):
            texture_name = f"crop_{idx}"
            metadata = context.input_object(f"metadata_{idx}")

            result = self.unproject_bbox(metadata["box"], depth_map=depth, intrinsics=intrinsics)
            if result is None:
                self.log_warning(f"Could not unproject bbox for object {idx}, skipping")
                self.advance_progress(generation_task)
                continue

            position, width, height = result

            billboard = Object3D.billboard(
                texture_name, 
                width=width,
                height=height,
                x=position[0],
                y=position[1],
                z=position[2],
            )
            scene.add_object(billboard)
            self.advance_progress(generation_task)

        self.finish_progress(generation_task)

        context.add_scene(ContextKey.SCENE, scene)
        return context
    
    def unproject_bbox(self, bbox, depth_map: Depth, intrinsics: CameraIntrinsics):
        bx, by, bw, bh = bbox
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh

        sx = intrinsics.width  / intrinsics.color_width
        sy = intrinsics.height / intrinsics.color_height

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        dx, dy = cx * sx, cy * sy

        patch_x1 = max(0, int(dx) - 5)
        patch_x2 = min(intrinsics.width,  int(dx) + 5)
        patch_y1 = max(0, int(dy) - 5)
        patch_y2 = min(intrinsics.height, int(dy) + 5)
        patch = depth_map[patch_y1:patch_y2, patch_x1:patch_x2]

        valid = patch[(patch > 0) & np.isfinite(patch)]
        if len(valid) == 0:
            return None
        depth = float(np.median(valid))
        position = intrinsics.unproject(cx, cy, depth)

        left   = intrinsics.unproject(x1, cy, depth)
        right  = intrinsics.unproject(x2, cy, depth)
        top    = intrinsics.unproject(cx, y1, depth)
        bottom = intrinsics.unproject(cx, y2, depth)

        return position, abs(right[0] - left[0]), abs(bottom[1] - top[1])