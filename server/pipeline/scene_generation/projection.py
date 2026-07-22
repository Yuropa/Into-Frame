import numpy as np

from scene.camera import CameraIntrinsics, CameraExtrinsics
from util.depth_utils import Depth


def mesh_y_at(world_x: float, world_z: float, terrain_mesh) -> float | None:
    """Return world-space terrain Y at (world_x, world_z) by raycasting down into the mesh."""
    import trimesh
    mesh: trimesh.Trimesh = terrain_mesh.mesh
    y_max = float(mesh.bounds[1][1]) + 1.0
    ray_origin = np.array([[world_x, y_max, world_z]], dtype=np.float64)
    ray_dir    = np.array([[0.0, -1.0, 0.0]], dtype=np.float64)
    locs, _, _ = mesh.ray.intersects_location(ray_origin, ray_dir, multiple_hits=True)
    if len(locs) == 0:
        return None
    return float(locs[:, 1].max())


def terrain_local_xz(world_x: float, world_z: float, yaw_degrees: float) -> tuple[float, float]:
    """Undo the yaw SceneGenerationStage sends to the client as scene.skybox_rotation
    (and applies to the terrain/water/formation Object3Ds) to map an object's WORLD-
    space (x, z) -- produced by unproject_bbox/unproject_bbox_equirect, which bake the
    full extrinsics rotation into position -- back into the terrain mesh's own native
    frame (+Z = panorama theta 0, no rotation applied), which is the frame terrain_mesh's
    raw vertices are actually stored in.

    Without this, mesh_y_at(world_x, world_z, terrain_mesh) raycasts against the
    terrain at the wrong (rotated-away) location whenever yaw_degrees != 0 -- missing
    the mesh's finite footprint entirely near its edges (silently falling back to the
    object's raw unprojected Y, i.e. floating/sinking) or hitting an unrelated part of
    the terrain otherwise. Y is unaffected by a yaw rotation, so mesh_y_at's return
    value needs no corresponding correction back the other way.
    """
    theta = np.radians(yaw_degrees)
    local_x = world_x * np.cos(theta) - world_z * np.sin(theta)
    local_z = world_x * np.sin(theta) + world_z * np.cos(theta)
    return local_x, local_z


def unproject_bbox(bbox, image_width, image_height, depth_map: Depth, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics):
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


def unproject_bbox_equirect(bbox, pano_width, pano_height, pano_depth: Depth, extrinsics: CameraExtrinsics):
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
