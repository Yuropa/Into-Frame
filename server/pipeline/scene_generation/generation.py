from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from scene.scene import Scene
from scene.object import Object3D
from scene.camera import CameraIntrinsics, CameraExtrinsics
from util.depth_utils import Depth
import numpy as np

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
      ContextKey.TERRAIN_MESH  → Mesh  (placed at origin if present)

    Output key (SemanticKey.OUTPUT) → ContextKey.SCENE (Scene)
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._gen = None

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
        panorama_depth = context.input_depth(ContextKey.PANORAMA_DEPTH)
        panorama = context.input_panorama(panorama_key)

        scene = Scene()
        scene.extrinsics = extrinsics

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
            generation_task = self.create_progress(object_count, "Creating Objects...")
            for idx in range(object_count):
                texture_name = f"crop_{idx}"
                mesh_name = f"mesh_{idx}"

                metadata = context.input_object(f"metadata_{idx}")

                cls = (metadata or {}).get("class")
                if cls in ("environment", "indeterminate"):
                    self.log_info(f"Skipping {cls} object {idx}")
                    self.advance_progress(generation_task)
                    continue

                if panorama_depth is not None and panorama is not None:
                    result = self.unproject_bbox_equirect(metadata["box"], panorama.width, panorama.height, pano_depth=panorama_depth, extrinsics=extrinsics)
                else:
                    result = self.unproject_bbox(metadata["box"], input.width, input.height, depth_map=depth, intrinsics=intrinsics, extrinsics=extrinsics)
                if result is None:
                    self.log_warning(f"Could not unproject bbox for object {idx}, skipping")
                    self.advance_progress(generation_task)
                    continue

                position, width, height = result

                if context.input_mesh(mesh_name) is not None:
                    updated_mesh = context.input_mesh(mesh_name)
                    updated_mesh.fit_to_box(width=width, height=height)
                    context.add_mesh(mesh_name, updated_mesh)

                    self.log_info(f"Creating mesh for {idx}")
                    mesh_obj = Object3D.mesh(
                        mesh_name,
                        x=position[0],
                        y=position[1],
                        z=position[2],
                    )
                    mesh_obj.name = mesh_name
                    scene.add_object(mesh_obj)
                else:
                    self.log_info(f"Creating billboard for {idx}")
                    billboard = Object3D.billboard(
                        texture_name, 
                        width=width,
                        height=height,
                        x=position[0],
                        y=position[1],
                        z=position[2],
                    )
                    billboard.name = mesh_name
                    scene.add_object(billboard)
                self.advance_progress(generation_task)

            self.finish_progress(generation_task)

        terrain_mesh = context.input_mesh(ContextKey.TERRAIN_MESH)
        if terrain_mesh is not None:
            self.log_info("Adding terrain mesh to scene")
            terrian = Object3D.mesh(ContextKey.TERRAIN_MESH, x=0.0, y=0.0, z=0.0)
            terrian.name = "terrian"
            scene.add_object(terrian)

        lighting = context.input_lighting(ContextKey.LIGHTING)
        if lighting is not None:
            self.log_info("Adding environment lighting to scene")
            scene.lighting = lighting

        context.add_scene(output_key, scene)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, _, _, _, _, _, output_key = self._resolved_keys()

        return context.scene(output_key) is not None
    
    def unproject_bbox(self, bbox, image_width, image_height, depth_map: Depth, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics):
        bx, by, bw, bh = bbox
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh

        sx = depth_map.width  / image_width
        sy = depth_map.height / image_height

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = int(round(cx * sx)), int(round(cy * sy))

        # Sample a patch around the projected center in depth space
        patch_radius = 5
        patch_x1 = max(0, dx - patch_radius)
        patch_x2 = min(depth_map.width,  dx + patch_radius)
        patch_y1 = max(0, dy - patch_radius)
        patch_y2 = min(depth_map.height, dy + patch_radius)

        patch = depth_map.depth[patch_y1:patch_y2, patch_x1:patch_x2]
        valid = patch[(patch > 0) & np.isfinite(patch)]

        if len(valid) == 0:
            return None

        depth = float(np.median(valid))

        # Unproject using color-space coordinates with color intrinsics
        position = extrinsics.transform(intrinsics.unproject(cx, cy, depth))
        left     = extrinsics.transform(intrinsics.unproject(x1, cy, depth))
        right    = extrinsics.transform(intrinsics.unproject(x2, cy, depth))
        top      = extrinsics.transform(intrinsics.unproject(cx, y1, depth))
        bottom   = extrinsics.transform(intrinsics.unproject(cx, y2, depth))

        return position, abs(right[0] - left[0]), abs(bottom[1] - top[1])

    def unproject_bbox_equirect(self, bbox, pano_width, pano_height, pano_depth: Depth, extrinsics: CameraExtrinsics):
        bx, by, bw, bh = bbox
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh

        sx = pano_depth.width  / pano_width
        sy = pano_depth.height / pano_height

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = int(round(cx * sx)), int(round(cy * sy))

        patch_radius = 5
        patch_x1 = max(0, dx - patch_radius)
        patch_x2 = min(pano_depth.width,  dx + patch_radius)
        patch_y1 = max(0, dy - patch_radius)
        patch_y2 = min(pano_depth.height, dy + patch_radius)

        patch = pano_depth.depth[patch_y1:patch_y2, patch_x1:patch_x2]
        valid = patch[(patch > 0) & np.isfinite(patch)]

        if len(valid) == 0:
            return None

        depth = float(np.median(valid))

        def equirect_point(px, py):
            theta = (px / pano_width  - 0.5) * 2.0 * np.pi
            phi   = (0.5 - py / pano_height) * np.pi
            x_cam = depth * np.cos(phi) * np.sin(theta)
            y_cam = depth * np.sin(phi)
            z_cam = depth * np.cos(phi) * np.cos(theta)
            return np.array(extrinsics.transform((x_cam, y_cam, z_cam)))

        position = equirect_point(cx, cy)
        left     = equirect_point(x1, cy)
        right    = equirect_point(x2, cy)
        top      = equirect_point(cx, y1)
        bottom   = equirect_point(cx, y2)

        width  = float(np.linalg.norm(right - left))
        height = float(np.linalg.norm(bottom - top))

        return tuple(position), width, height