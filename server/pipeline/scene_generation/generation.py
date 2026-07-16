from typing import Any
from logging import Logger

import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_typing.categories import ENVIRONMENT_CATEGORIES as _ENV_CATEGORIES
from pipeline.scene_generation.projection import mesh_y_at, unproject_bbox, unproject_bbox_equirect
from scene.scene import Scene
from scene.object import Object3D
import numpy as np


class SceneGenerationConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        eye_height_meters: float = 1.8,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Sent to the client as the target world-space depth of the terrain
        # center below the viewer; the client pushes the whole scene down to match.
        self.eye_height_meters = eye_height_meters

# Objects whose estimated real-world largest dimension exceeds this threshold (meters)
# are assumed to be large scene elements (mountains, hills, sky) and are skipped.
_MAX_OBJECT_SIZE_M = 40.0


class SceneGenerationStage(PipelineStage):
    """
    Assembles the final 3D scene by unprojecting each detected object's bounding box
    into world space (using depth + camera parameters) and placing its mesh or billboard
    at the computed position. Also adds the terrain mesh (if present) at the origin.

    Input keys:
      SemanticKey.INPUT        → ContextKey.INPUT          (Image, used for bbox scale)
      SemanticKey.DEPTH        → ContextKey.DEPTH          (Depth, for world-space placement)
      SemanticKey.INTRINSICS   → ContextKey.INTRINSICS     (CameraIntrinsics)
      SemanticKey.EXTRINSICS   → ContextKey.EXTRINSICS     (CameraExtrinsics)
      SemanticKey.PANORAMA     → ContextKey.PANORAMA       (Image, used as scene skybox)
      SemanticKey.OBJECT_COUNT → ContextKey.OBJECT_COUNT   (int)

    Dynamic context keys per object (index i):
      crop_{i}      → Image   (object texture)
      mesh_{i}      → Mesh    (optional; falls back to billboard if absent)
      metadata_{i}  → object  ({"box": [...], "score": float})

    Optional:
      ContextKey.TERRAIN_MESH        → Mesh        (placed at origin if present)
      ContextKey.WATER_MESH          → Mesh        (placed at origin if present)
      ContextKey.TERRAIN_FORMATIONS  → list[dict]  (each references a dynamic
                                        "terrain_formation_{id}" Mesh key, already
                                        in absolute world space -- placed at origin)

    Output key (SemanticKey.OUTPUT) → ContextKey.SCENE (Scene)
    """

    def __init__(self, config: SceneGenerationConfiguration) -> None:
        super().__init__(config)
        self._gen = None

    @classmethod
    def config_class(cls) -> type[SceneGenerationConfiguration]:
        return SceneGenerationConfiguration

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.OBJECT_COUNT: ContextKey.OBJECT_COUNT,
            SemanticKey.EXTRINSICS: ContextKey.EXTRINSICS,
            SemanticKey.INTRINSICS: ContextKey.INTRINSICS,
            SemanticKey.INPUT: ContextKey.INPUT,
            SemanticKey.DEPTH: ContextKey.DEPTH,
            SemanticKey.PANORAMA: ContextKey.PANORAMA,
            SemanticKey.OUTPUT: ContextKey.SCENE
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        count_key, extrinsics_key, intrinsics_key, input_key, depth_key, panorama_key, output_key = self._resolved_keys()

        object_count = context.input_object(count_key)
        intrinsics = context.input_intrinsics(intrinsics_key)
        extrinsics = context.input_extrinsics(extrinsics_key)
        depth = context.input_depth(depth_key)
        input = context.input_image(input_key)
        # Objects were detected on the ORIGINAL panorama, so their bounding boxes must be
        # unprojected using depth measured on that same original panorama (PANORAMA_OBJECT_DEPTH)
        # — not PANORAMA_DEPTH, which is recomputed after foreground inpainting and no longer
        # has real depth at removed-object locations (it has the reconstructed background
        # instead), which would silently misplace exactly the near-camera objects most likely
        # to have been removed.
        panorama_depth = context.input_depth(ContextKey.PANORAMA_OBJECT_DEPTH)
        panorama = context.input_panorama(panorama_key)
        terrain_mesh = context.input_mesh(ContextKey.TERRAIN_MESH)

        scene = Scene()
        scene.extrinsics = extrinsics
        # camera_height = world Y of the camera = terrain_y_at_nadir + camera_height_meters.
        # Using the extrinsics translation directly is equivalent and simpler.
        scene.camera_height = float(extrinsics.translation[1]) if extrinsics is not None else 0.0

        scene.eye_height_meters = self.config.eye_height_meters
        if terrain_mesh is not None:
            terrain_center_y = mesh_y_at(0.0, 0.0, terrain_mesh)
            scene.terrain_center_y = float(terrain_center_y) if terrain_center_y is not None else 0.0

        # Rotate the skybox to align the panorama center with the camera's forward direction.
        # The panorama center (theta=0) = +Z in camera space; the extrinsics rotation tells
        # us where +Z ended up in world space.  The skybox _Rotation (Y-axis, degrees) must
        # match that yaw so terrain and panorama line up.
        if extrinsics is not None:
            cam_forward = extrinsics.rotation @ np.array([0.0, 0.0, 1.0])
            scene.skybox_rotation = float(np.degrees(np.arctan2(cam_forward[0], cam_forward[2])))

        if depth is not None:
            valid = depth.depth[np.isfinite(depth.depth) & (depth.depth > 0)]
            if len(valid) > 0:
                scene.far_clip_plane = float(np.percentile(valid, 99)) * 1.5
                scene.far_clip_plane = max(10.0, min(scene.far_clip_plane, 1000.0))

        if panorama is not None:
            scene.skybox = panorama_key

        if object_count is not None:
            # Build category → list of crop indices so billboards can be drawn
            # from the full pool for a category rather than only the per-object crop.
            # Synthetic (DistributionSynthesisStage) entries never get their own
            # crop_{idx} image, so they must be excluded here -- otherwise a
            # synthetic point can be chosen as another object's billboard texture,
            # pointing at a crop image that was never created.
            class_to_crop_indices: dict[str, list[int]] = {}
            for idx in range(object_count):
                meta = context.input_object(f"metadata_{idx}") or {}
                cls = meta.get("class")
                if cls and cls not in _ENV_CATEGORIES and cls != "indeterminate" and not meta.get("synthetic"):
                    class_to_crop_indices.setdefault(cls, []).append(idx)

            rng = np.random.default_rng(self.seed)

            generation_task = self.create_progress(object_count, "Creating Objects…")
            for idx in range(object_count):
                metadata = context.input_object(f"metadata_{idx}")

                cls = (metadata or {}).get("class")
                if cls in _ENV_CATEGORIES or cls == "indeterminate":
                    self.log_info(f"Skipping {cls} object {idx}")
                    self.advance_progress(generation_task)
                    continue

                if metadata.get("synthetic"):
                    # Points painted across a region by DistributionSynthesisStage carry
                    # their world XZ + footprint directly — there's no detection bbox to
                    # unproject. Y is a placeholder; it gets replaced by the terrain snap
                    # below, same as every other object.
                    syn_x, _, syn_z = metadata["world_position"]
                    position = (syn_x, 0.0, syn_z)
                    width = metadata["world_width"]
                    height = metadata["world_height"]
                elif panorama_depth is not None and panorama is not None:
                    result = unproject_bbox_equirect(metadata["box"], panorama.width, panorama.height, pano_depth=panorama_depth, extrinsics=extrinsics)
                    if result is None:
                        self.log_warning(f"Could not unproject bbox for object {idx}, skipping")
                        self.advance_progress(generation_task)
                        continue
                    position, width, height = result
                else:
                    result = unproject_bbox(metadata["box"], input.width, input.height, depth_map=depth, intrinsics=intrinsics, extrinsics=extrinsics)
                    if result is None:
                        self.log_warning(f"Could not unproject bbox for object {idx}, skipping")
                        self.advance_progress(generation_task)
                        continue
                    position, width, height = result

                max_dim = max(width, height)
                if max_dim > _MAX_OBJECT_SIZE_M:
                    self.log_info(f"Skipping object {idx} ({cls}): estimated size {max_dim:.1f}m exceeds limit")
                    self.advance_progress(generation_task)
                    continue

                # Snap the object's base to the terrain surface. The Y offset that
                # lands the *bottom* of the object on the terrain depends on the
                # object's actual placed vertical extent, which differs between the
                # billboard (scaled exactly to `height`) and category-mesh (scaled
                # uniformly to `mesh_scale`) branches below — so it's resolved per
                # branch rather than once here.
                terrain_y = None
                if terrain_mesh is not None:
                    terrain_y = mesh_y_at(position[0], position[2], terrain_mesh)

                place_y = terrain_y + height / 2.0 if terrain_y is not None else position[1]

                context.add_object(f"metadata_{idx}", {
                    **(metadata or {}),
                    "world_position": list(map(float, (position[0], place_y, position[2]))),
                    "world_width": float(width),
                    "world_height": float(height),
                })

                category_mesh = context.input_mesh(f"category_mesh_{cls}")
                if category_mesh is not None and rng.integers(2) == 0:
                    # Use the shared category mesh with a random Y rotation.
                    # Mesh.fit_to_box recenters on the mesh's centroid, not its bounding-box
                    # center, so the mesh's lowest vertex is not reliably at -extent/2 --
                    # for a bottom-heavy shape (e.g. a tree trunk) the centroid sits below
                    # the bbox center, and assuming symmetry would still leave a gap above
                    # the terrain. Sample the mesh's actual lowest vertex (bounds[0][1])
                    # and offset by exactly that, so the true bottom -- not an assumed one
                    # -- lands on terrain_y.
                    mesh_scale = float(min(width, height))
                    mesh_min_y = float(category_mesh.mesh.bounds[0][1]) * mesh_scale
                    mesh_place_y = terrain_y - mesh_min_y if terrain_y is not None else position[1]
                    mesh_key = f"category_mesh_{cls}"
                    self.log_info(f"Creating mesh for {idx} ({cls})")
                    mesh_obj = Object3D.mesh(mesh_key, x=position[0], y=mesh_place_y, z=position[2])
                    mesh_obj.set_rotation(0.0, float(rng.uniform(0.0, 360.0)), 0.0)
                    mesh_obj.set_scale(mesh_scale, mesh_scale, mesh_scale)
                    mesh_obj.name = mesh_key
                    scene.add_object(mesh_obj)
                else:
                    # Pick a random billboard crop from this category's pool. Synthetic
                    # points have no crop of their own, so they must draw from a real
                    # detection's pool; if none exists there's nothing to render.
                    crop_pool = class_to_crop_indices.get(cls, [] if metadata.get("synthetic") else [idx])
                    if not crop_pool:
                        self.log_warning(f"No billboard crop available for synthetic object {idx} ({cls}), skipping")
                        self.advance_progress(generation_task)
                        continue
                    chosen_idx = int(rng.choice(crop_pool))
                    self.log_info(f"Creating billboard for {idx} ({cls}) using crop_{chosen_idx}")
                    billboard = Object3D.billboard(
                        f"crop_{chosen_idx}",
                        width=width,
                        height=height,
                        x=position[0],
                        y=place_y,
                        z=position[2],
                    )
                    billboard.name = f"billboard_{idx}"
                    scene.add_object(billboard)
                self.advance_progress(generation_task)

            self.finish_progress(generation_task)

        terrain_mesh = context.input_mesh(ContextKey.TERRAIN_MESH)
        if terrain_mesh is not None:
            self.log_info("Adding terrain mesh to scene")
            terrain = Object3D.mesh(ContextKey.TERRAIN_MESH, x=0.0, y=0.0, z=0.0)
            terrain.name = "terrain"
            scene.add_object(terrain)

        water_mesh = context.input_mesh(ContextKey.WATER_MESH)
        if water_mesh is not None:
            self.log_info("Adding water mesh to scene")
            water = Object3D.mesh(ContextKey.WATER_MESH, x=0.0, y=0.0, z=0.0)
            water.name = "water"
            scene.add_object(water)

        # Non-primary ground components (see TerrainMeshStage / HeightMapGenerator.
        # _label_ground_components) -- a separate landmass, an isolated rock
        # formation, anything real but genuinely disconnected from the base terrain.
        # Each already carries its own absolute-world-space vertices (built by
        # TerrainMeshGenerator.generate_component_mesh) and its own baked texture
        # (TerrainTextureGenerationStage._texture_formations), so it's placed at the
        # origin exactly like terrain/water above.
        #
        # Object3D.name deliberately does NOT contain "terrain" -- the Unity client's
        # TerrainMaterialManager matches meshes to apply the shared, separately-
        # transmitted SplatMaterial by a case-insensitive substring check against
        # "terrain" (confirmed this session), and a formation mesh already carries
        # its own baked-in texture; it must not be swept up by that name match.
        formations = context.input_object(ContextKey.TERRAIN_FORMATIONS) or []
        for formation in formations:
            formation_mesh = context.input_mesh(formation["mesh_key"])
            if formation_mesh is None:
                continue
            self.log_info(f"Adding formation mesh {formation['id']} to scene")
            formation_obj = Object3D.mesh(formation["mesh_key"], x=0.0, y=0.0, z=0.0)
            formation_obj.name = f"formation_{formation['id']}"
            scene.add_object(formation_obj)

        lighting = context.input_lighting(ContextKey.LIGHTING)
        if lighting is not None:
            self.log_info("Adding environment lighting to scene")
            scene.lighting = lighting

        context.add_scene(output_key, scene)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, _, _, _, _, _, output_key = self._resolved_keys()

        return context.scene(output_key) is not None