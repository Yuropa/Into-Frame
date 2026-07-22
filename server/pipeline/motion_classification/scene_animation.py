from logging import Logger
from typing import Any

import numpy as np

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from scene.object import ObjectType

# Objects taller (footprint height / width) than this ratio get a capsule
# collider instead of a box -- reads better for anything person/bird/bottle-shaped.
_CAPSULE_ASPECT_RATIO = 1.5

# A moving object's true depth extent isn't separately estimated anywhere
# upstream (only width/height from its 2D bbox) -- approximating it as the
# larger of the two is the same simplification SceneGenerationStage already
# makes for mesh scale (see its own mesh_scale comment).
def _collider_size(world_width: float, world_height: float) -> tuple[str, list[float]]:
    depth_guess = max(world_width, world_height)
    shape = "capsule" if world_height > world_width * _CAPSULE_ASPECT_RATIO else "box"
    return shape, [world_width, world_height, depth_guess]


class SceneAnimationConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        wind_axis_degrees: "float | None" = None,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # A single wind direction for the whole scene reads more physically
        # plausible than each tree leaning its own random way -- fixed at
        # construction (derived from `seed` below when left None) so re-running
        # the same scene doesn't change which way things lean.
        self.wind_axis_degrees = wind_axis_degrees


class SceneAnimationStage(PipelineStage):
    """
    Final annotation pass: reads the already-built Scene (SceneGenerationStage,
    long before video existed) and attaches animation/physics data to its existing
    Object3D entries, using each object's `source_index` to look back up its own
    detection metadata (ObjectMotionClassificationStage's stationary/sway/motion
    fields). Does not add, remove, or reposition any Object3D -- purely annotates.

    - Stationary billboard  -> videoColor/videoAlpha (the extracted per-object clip),
      replacing the static texture as an animated billboard.
    - Stationary mesh        -> sway params (amplitude/frequencyHz from the object's
      own clip, phase randomized per instance, a shared wind_axis_degrees for the
      whole scene) for Unity's WindSway component to drive the bones
      CategoryMeshRiggingStage baked into that category's mesh asset.
    - Moving billboard/mesh  -> physics (velocity/acceleration plus a collider
      shape/size derived from the object's own world footprint) for Unity's
      PhysicsHandoff component to apply once at spawn, then let Unity's own
      Rigidbody/physics own the motion from then on.

    Reads:  ContextKey.SCENE, metadata_{i} (stationary/sway/motion/world_width/
            world_height, per source_index), object_video_{i}/object_video_alpha_{i}
    Writes: ContextKey.SCENE (same Scene, objects annotated in place)
    Output key (SemanticKey.OUTPUT) -> ContextKey.SCENE_ANIMATION_COUNT (int)
    """

    @classmethod
    def config_class(cls) -> type[SceneAnimationConfiguration]:
        return SceneAnimationConfiguration

    def run(self, context: PipelineContext) -> PipelineContext:
        scene = context.input_scene(ContextKey.SCENE)
        if scene is None or not scene.objects:
            self.log_info("No scene, skipping")
            return context

        wind_axis = self.config.wind_axis_degrees
        if wind_axis is None:
            wind_axis = float(np.random.default_rng(self.seed).uniform(0.0, 360.0))

        annotated = 0
        task = self.create_progress(len(scene.objects), "Animating scene objects…")
        for obj in scene.objects:
            if obj.source_index is None:
                self.advance_progress(task)
                continue

            idx = obj.source_index
            metadata = context.input_object(f"metadata_{idx}") or {}
            if "stationary" not in metadata:
                self.advance_progress(task)
                continue

            if metadata["stationary"]:
                if obj.type == ObjectType.BILLBOARD:
                    color = context.input_video(f"object_video_{idx}")
                    alpha = context.input_video(f"object_video_alpha_{idx}")
                    if color is not None and alpha is not None:
                        obj.video_color = f"object_video_{idx}"
                        obj.video_alpha = f"object_video_alpha_{idx}"
                        annotated += 1
                elif obj.type == ObjectType.MESH:
                    sway = metadata.get("sway") or {}
                    phase_rng = np.random.default_rng((self.seed, idx))
                    obj.sway = {
                        "amplitude": float(sway.get("amplitude", 0.03)),
                        "frequencyHz": float(sway.get("frequencyHz", 0.25)),
                        "phase": float(phase_rng.uniform(0.0, 2.0 * np.pi)),
                        "axisDegrees": wind_axis,
                    }
                    annotated += 1
            else:
                motion = metadata.get("motion")
                world_width = metadata.get("world_width")
                world_height = metadata.get("world_height")
                if motion is not None and world_width and world_height:
                    shape, size = _collider_size(world_width, world_height)
                    obj.physics = {
                        "velocity": motion["velocity"],
                        "acceleration": motion["acceleration"],
                        "colliderShape": shape,
                        "colliderSize": size,
                    }
                    annotated += 1

            self.advance_progress(task)
        self.finish_progress(task)

        context.add_scene(ContextKey.SCENE, scene)
        context.add_object(ContextKey.SCENE_ANIMATION_COUNT, annotated)
        self.log_info(f"Annotated {annotated} scene object(s)")
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(ContextKey.SCENE_ANIMATION_COUNT) is not None

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        count = context.object(ContextKey.SCENE_ANIMATION_COUNT)
        if count is None:
            return None
        return ReportSection(
            stage_name=self.name,
            title="Scene Animation",
            body=(
                "Objects already placed by Scene Generation were annotated with "
                "animated-billboard video, procedural sway parameters, or rigid-body "
                "physics handoff data, based on each object's own motion "
                "classification from its extracted clip."
            ),
            stats={"Objects annotated": str(count)},
        )
