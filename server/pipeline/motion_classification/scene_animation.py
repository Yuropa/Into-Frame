import math
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
        default_sway_amplitude: float = 0.16,
        default_sway_frequency_hz: float = 0.6,
        sway_frequency_scale: float = 0.5,
        sway_max_distance_m: float = 8.0,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Fallback sway for a stationary mesh whose own instance was never tracked
        # in the generated video, so ObjectMotionClassificationStage recorded
        # "stationary" with no measured sway. That is the overwhelming majority of
        # placed instances -- VideoObjectExtractionStage tracks only a handful per
        # category -- so these defaults, not the measured values, are what the
        # scene's motion actually looks like.
        #
        # The previous 0.03 / 0.25 Hz was set as a deliberately conservative
        # placeholder and is invisible in practice: amplitude is a fraction of the
        # object's own height, so 0.03 on a 10 cm asset is a 3 mm excursion at one
        # cycle every four seconds. These are taken from the measured values real
        # tracked vegetation produces instead -- on the Rainier capture, tracked
        # flowers came back at amplitude 0.17-0.35 and 0.57-0.71 Hz.
        self.default_sway_amplitude = float(default_sway_amplitude)
        self.default_sway_frequency_hz = float(default_sway_frequency_hz)
        # Global multiplier on every sway frequency, measured and default alike --
        # a single knob for how fast the whole scene moves. Applied last, after the
        # per-instance jitter, so the spread between instances scales with it.
        #
        # ObjectMotionClassificationStage derives frequency from zero-crossings of a
        # tracked crop's horizontal centroid, which counts a full there-and-back
        # sway as two events and so reads roughly double the real rate; on the
        # Rainier capture that put placed vegetation at a median 1.24 Hz, fast
        # enough to look like it's being shaken rather than blown. 0.5 halves it.
        self.sway_frequency_scale = float(sway_frequency_scale)
        # Radius beyond which a stationary mesh is left un-swayed -- see the
        # sway_limit block in run() for the per-frame cost this bounds. 0 disables.
        self.sway_max_distance_m = float(sway_max_distance_m)
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
        # Sway is a per-frame cost on the CLIENT, paid per instance and paid whether or
        # not anyone can see the motion. Every object carrying `sway` gets a WindSway
        # MonoBehaviour whose LateUpdate samples the wind field at its own position and
        # writes three bone rotations -- and the mesh it drives has to keep its skin, so
        # it instantiates as a SkinnedMeshRenderer, which can be neither GPU-instanced
        # nor static-batched. On the Rainier capture that was 17,129 of 17,475 objects.
        #
        # Amplitude is a fraction of the object's own footprint, so the ANGULAR motion
        # of a 0.35 m grass tuft falls off with distance and is imperceptible well
        # before the 25 m the ground cover reaches. Measured on that scene, swaying
        # objects by distance from the camera: 8% within 5 m, 18% within 8 m, 51%
        # within 15 m. Gating at 8 m therefore drops ~82% of the per-frame work and of
        # the un-batchable skinned renderers, for motion nobody could resolve anyway.
        #
        # 0 disables the gate (every stationary instance sways, the old behaviour).
        sway_limit = self.config.sway_max_distance_m
        camera_xz = (0.0, 0.0)
        if scene.extrinsics is not None:
            camera_xz = (
                float(scene.extrinsics.translation[0]),
                float(scene.extrinsics.translation[2]),
            )
        swaying = 0
        sway_skipped = 0

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
                    # Keyed on the crop the billboard actually DISPLAYS, not on the
                    # detection it was placed from -- see Object3D.texture_source_index.
                    # These differ whenever the billboard drew from its (class, bucket)
                    # pool, and always differ for a synthetic painted instance, which
                    # has no crop or clip of its own but renders a real sibling's.
                    video_idx = obj.texture_source_index
                    if video_idx is None:
                        video_idx = idx
                    color = context.input_video(f"object_video_{video_idx}")
                    alpha = context.input_video(f"object_video_alpha_{video_idx}")
                    if color is not None and alpha is not None:
                        obj.video_color = f"object_video_{video_idx}"
                        obj.video_alpha = f"object_video_alpha_{video_idx}"
                        annotated += 1
                elif obj.type == ObjectType.MESH:
                    if sway_limit > 0:
                        distance = math.hypot(
                            float(obj.position["x"]) - camera_xz[0],
                            float(obj.position["z"]) - camera_xz[1],
                        )
                        if distance > sway_limit:
                            sway_skipped += 1
                            self.advance_progress(task)
                            continue
                    sway = metadata.get("sway") or {}
                    phase_rng = np.random.default_rng((self.seed, idx))
                    amplitude = float(sway.get("amplitude", self.config.default_sway_amplitude))
                    frequency = float(sway.get("frequencyHz", self.config.default_sway_frequency_hz))
                    obj.sway = {
                        # Jitter per instance so a field of instances sharing one
                        # bucket's fallback defaults doesn't move as a single rigid
                        # sheet -- phase alone desynchronises the timing but leaves
                        # every blade tracing an identical arc.
                        "amplitude": amplitude * float(phase_rng.uniform(0.75, 1.25)),
                        "frequencyHz": (
                            frequency * float(phase_rng.uniform(0.8, 1.2))
                            * self.config.sway_frequency_scale
                        ),
                        "phase": float(phase_rng.uniform(0.0, 2.0 * np.pi)),
                        "axisDegrees": wind_axis,
                    }
                    annotated += 1
                    swaying += 1
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
        if sway_skipped:
            self.log_info(
                f"Sway: {swaying} instance(s) within {sway_limit:.0f} m, "
                f"{sway_skipped} beyond it left static "
                f"({sway_skipped / max(swaying + sway_skipped, 1) * 100:.0f}% fewer "
                f"per-frame sway updates on the client)"
            )
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
