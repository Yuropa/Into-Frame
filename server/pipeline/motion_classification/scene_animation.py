import math
from logging import Logger
from typing import Any

import numpy as np

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from scene.object import ObjectType
from pipeline.object_typing.categories import VEGETATION_CATEGORIES

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
        sway_distance_per_height: float = 8.0,
        sway_large_max_distance_m: float = 150.0,
        sway_large_max_instances: int = 250,
        collider_min_size_m: float = 1.0,
        collider_max_instances: int = 32,
        collider_max_faces: int = 600,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Collision proxies for the modelled things that stand still (a sea stack, a
        # moored boat, a building): see _static_colliders. A model at least
        # collider_min_size_m across as placed, placed at most collider_max_instances
        # times, gets a geometry-only proxy of collider_max_faces triangles at most.
        # collider_max_faces 0 turns it off.
        self.collider_min_size_m = float(collider_min_size_m)
        self.collider_max_instances = int(collider_max_instances)
        self.collider_max_faces = int(collider_max_faces)
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
        # The radius grows with the object: a mesh sways out to this many times its own
        # height (an 18 m tree to 144 m, a 0.4 m flower stays at the radius above), no
        # further than sway_large_max_distance_m, and no more than
        # sway_large_max_instances of them beyond the plain radius (nearest first) --
        # each is one more skinned, un-instanced renderer on the client.
        self.sway_distance_per_height = float(sway_distance_per_height)
        self.sway_large_max_distance_m = float(sway_large_max_distance_m)
        self.sway_large_max_instances = int(sway_large_max_instances)
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
        sway_rigid = 0
        # A tree's sway is seen from much further than a tuft's (the motion is a
        # fraction of the object's size): with one radius for everything, 8 m, Paris's
        # 35 mesh trees and every Rainier fir stood still. The objects that sway beyond
        # the plain radius on their size are chosen first, nearest first, up to the cap.
        heights: dict[str, float] = {}

        def world_height(o) -> float:
            if o.mesh not in heights:
                mesh = context.input_mesh(o.mesh) if o.mesh else None
                heights[o.mesh] = float(mesh.extents[1]) if mesh is not None else 0.0
            return heights[o.mesh] * abs(float(o.scale["y"]))

        def camera_distance(o) -> float:
            return math.hypot(float(o.position["x"]) - camera_xz[0], float(o.position["z"]) - camera_xz[1])

        sway_far: set = set()
        if sway_limit > 0 and self.config.sway_distance_per_height > 0 and self.config.sway_large_max_instances > 0:
            far = []
            for o in scene.objects:
                if o.source_index is None or o.type != ObjectType.MESH:
                    continue
                meta = context.input_object(f"metadata_{o.source_index}") or {}
                if not meta.get("stationary") or meta.get("class") not in VEGETATION_CATEGORIES:
                    continue
                d = camera_distance(o)
                if sway_limit < d <= min(self.config.sway_distance_per_height * world_height(o),
                                         self.config.sway_large_max_distance_m):
                    far.append((d, id(o)))
            far.sort()
            sway_far = {i for _d, i in far[: self.config.sway_large_max_instances]}
            if far:
                self.log_info(f"Sway: {len(sway_far)} of {len(far)} large instance(s) sway beyond {sway_limit:.0f} m "
                              f"on their size (out to {far[len(sway_far) - 1][0]:.0f} m)")

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
                    # Wind bends plants. A rock, a moored boat or a bench is "stationary"
                    # too, and with sway parameters the client rocks it on its base
                    # (Shark Fin's boulder carried them).
                    if metadata.get("class") not in VEGETATION_CATEGORIES:
                        sway_rigid += 1
                        self.advance_progress(task)
                        continue
                    if sway_limit > 0:
                        distance = camera_distance(obj)
                        if distance > sway_limit and id(obj) not in sway_far:
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

        if self.config.collider_max_faces > 0:
            try:
                self._static_colliders(context, scene)
            except Exception as exc:  # the scene stands without them
                self.log_warning(f"Static colliders not written ({type(exc).__name__}: {exc})")

        context.add_scene(ContextKey.SCENE, scene)
        context.add_object(ContextKey.SCENE_ANIMATION_COUNT, annotated)
        self.log_info(f"Annotated {annotated} scene object(s)")
        if sway_rigid:
            self.log_info(f"Sway: {sway_rigid} stationary mesh instance(s) that are not vegetation left rigid")
        if sway_skipped:
            self.log_info(
                f"Sway: {swaying} instance(s) within {sway_limit:.0f} m, "
                f"{sway_skipped} beyond it left static "
                f"({sway_skipped / max(swaying + sway_skipped, 1) * 100:.0f}% fewer "
                f"per-frame sway updates on the client)"
            )
        return context

    def _static_colliders(self, context: PipelineContext, scene) -> None:
        """A collision proxy on every modelled object that stands still.

        Only the terrain carried a collider: Shark Fin's sea stack is an 8,000 triangle
        model 25 m across standing in the sea, and for the client's physics nothing was
        there (a thrown object, the player and anything that later floats on the water
        passed through it). Each such model gets `<mesh>_physics`, its own surface
        welded and decimated to a few hundred triangles with no texture, in the model's
        own frame, so the client's MeshCollider takes the object's position, rotation
        and scale with it (SceneObject.physicsMesh, as the terrain's). Plants are left
        out (they bend, and there are thousands), as are the environment's own meshes,
        anything already handed to the rigid-body path, and models placed too often or
        too small to be worth a MeshCollider each."""
        import trimesh
        from scene.mesh import Mesh
        cfg: SceneAnimationConfiguration = self.config
        environment = {"terrain", "water", "water_apron", "treeline", "backdrop"}
        groups: dict[str, list] = {}
        for obj in scene.objects:
            if obj.type != ObjectType.MESH or not obj.mesh or obj.physics_mesh or obj.physics or obj.sway:
                continue
            if obj.name in environment or str(obj.name).startswith("formation_"):
                continue
            # The class: the detection's, an agent placement's own, and the model
            # file's name (category_mesh_<class>_<n>) for a placement with neither.
            classes = {str((getattr(obj, "agent", None) or {}).get("class") or "")}
            if obj.source_index is not None:
                classes.add(str((context.input_object(f"metadata_{obj.source_index}") or {}).get("class") or ""))
            if str(obj.mesh).startswith("category_mesh_"):
                classes.add(str(obj.mesh)[len("category_mesh_"):].rsplit("_", 1)[0])
            if classes & set(VEGETATION_CATEGORIES):
                continue
            groups.setdefault(obj.mesh, []).append(obj)
        written = []
        for key, objs in groups.items():
            if len(objs) > cfg.collider_max_instances:
                continue
            source = context.input_mesh(key)
            if source is None:
                continue
            tm = source.mesh
            extent = float(np.max(tm.extents)) if len(tm.vertices) else 0.0
            largest = max(extent * max(abs(float(o.scale[a])) for a in ("x", "y", "z")) for o in objs)
            if largest < cfg.collider_min_size_m:
                continue
            # A card is a picture of the thing, not its volume (Paris's landmark card,
            # 209 m of flat quad, would have been a wall across the far bank).
            if len(tm.vertices) and float(np.min(tm.extents)) < 0.02 * extent:
                continue
            proxy = trimesh.Trimesh(vertices=np.asarray(tm.vertices, dtype=np.float64),
                                    faces=np.asarray(tm.faces, dtype=np.int64), process=False)
            proxy.merge_vertices(merge_tex=True, merge_norm=True)   # a textured model's seams are split vertices
            how = "its surface"
            if len(proxy.faces) > cfg.collider_max_faces:
                try:
                    import open3d as o3d
                    dense = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(proxy.vertices)),
                                                      o3d.utility.Vector3iVector(np.asarray(proxy.faces)))
                    coarse = dense.simplify_quadric_decimation(cfg.collider_max_faces)
                    coarse.remove_degenerate_triangles()
                    coarse.remove_unreferenced_vertices()
                    proxy = trimesh.Trimesh(vertices=np.asarray(coarse.vertices), faces=np.asarray(coarse.triangles), process=False)
                    how = "its surface decimated"
                except Exception:
                    proxy = proxy.convex_hull
                    how = "its convex hull"
            if len(proxy.faces) < 4 or not np.all(np.isfinite(proxy.vertices)):
                continue
            # The proxy must stay the model's size: a decimation that pulled the
            # silhouette in by a tenth would let things sink that far into the rock.
            if np.any(np.abs(proxy.extents - tm.extents) > 0.1 * max(extent, 1e-9)):
                proxy = trimesh.Trimesh(vertices=np.asarray(tm.vertices, dtype=np.float64),
                                        faces=np.asarray(tm.faces, dtype=np.int64), process=False).convex_hull
                how = "its convex hull"
            physics_key = f"{key}_physics"
            context.add_mesh(physics_key, Mesh(trimesh.Trimesh(vertices=np.asarray(proxy.vertices), faces=np.asarray(proxy.faces), process=False)))
            for o in objs:
                o.physics_mesh = physics_key
            if self.temp is not None:
                context.mesh(physics_key).save(self.temp / f"{physics_key}.glb")
            written.append(f"{key} ({len(objs)} placed, {largest:.1f} m, {len(proxy.faces)} triangles of {how})")
        if written:
            self.log_info(f"Static colliders: {len(written)} model(s): " + "; ".join(written))
        else:
            self.log_info("Static colliders: no modelled object stands still at a size worth one")

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
