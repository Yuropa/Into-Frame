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


def terrain_local_xz(
    world_x: float,
    world_z: float,
    yaw_degrees: float,
    camera_x: float = 0.0,
    camera_z: float = 0.0,
) -> tuple[float, float]:
    """Undo the yaw SceneGenerationStage sends to the client as scene.skybox_rotation
    (and applies to the terrain/water/formation Object3Ds) to map an object's WORLD-
    space (x, z) -- produced by unproject_bbox/unproject_bbox_equirect, which bake the
    full extrinsics transform into position -- back into the terrain mesh's own native
    frame (+Z = panorama theta 0, no rotation applied), which is the frame terrain_mesh's
    raw vertices are actually stored in.

    Without this, mesh_y_at(world_x, world_z, terrain_mesh) raycasts against the
    terrain at the wrong (rotated-away) location whenever yaw_degrees != 0 -- missing
    the mesh's finite footprint entirely near its edges (silently falling back to the
    object's raw unprojected Y, i.e. floating/sinking) or hitting an unrelated part of
    the terrain otherwise. Y is unaffected by a yaw rotation, so mesh_y_at's return
    value needs no corresponding correction back the other way.

    camera_x/camera_z are the extrinsics TRANSLATION's own horizontal components, and
    have to come off before the rotation is undone. Every grid the terrain is built
    from -- the height map, the region map, the terrain mesh's own Poisson domain --
    is authored with the camera at the grid origin by construction (see
    grid_cell_panorama_uv and DistributionSynthesisStage._grid_to_world), and the
    terrain/water/formation Object3Ds are correspondingly placed at x=z=0 carrying
    rotation only. Object positions are not: extrinsics.transform adds the full
    translation, so a non-zero t.x/t.z offsets every object horizontally relative to
    the terrain it is about to be snapped against. That offset is invisible on flat
    ground and reads as floating/buried objects exactly where the terrain has relief,
    which is the same symptom the yaw compensation above was added for.
    """
    theta = np.radians(yaw_degrees)
    rel_x = world_x - camera_x
    rel_z = world_z - camera_z
    local_x = rel_x * np.cos(theta) - rel_z * np.sin(theta)
    local_z = rel_x * np.sin(theta) + rel_z * np.cos(theta)
    return local_x, local_z


def height_map_y_at(
    local_x: float,
    local_z: float,
    height_map,
    grid_size_meters: float,
) -> float | None:
    """Bilinearly sample the dense HEIGHT_MAP grid at a terrain-LOCAL (x, z).

    Fallback for when the mesh raycast in mesh_y_at misses. HEIGHT_MAP is the exact
    surface TerrainMeshGenerator.generate sampled to build terrain_mesh's vertices
    (Terrain Reconstruction and Terrain Noise Refinement both write their result back
    under the same key, so what is in the context here is the final one), but it is a
    complete grid rather than a triangulated sheet -- no finite footprint to fall off,
    no water/formation depression carved into it, and no gaps between Poisson samples.
    So wherever the raycast comes back empty this still answers with the reconstructed
    ground height, which is a far better place to stand an object than its own raw
    unprojected Y -- that Y comes from the depth map, has never been reconciled with
    the reconstructed terrain, and is what made a missed raycast render as an object
    floating in the air or sunk into the hillside.

    Row/column convention matches TerrainMeshGenerator.generate's own sampling of the
    same array (rows = Z, cols = X, both spanning [-grid_size/2, +grid_size/2]).
    Returns None outside the grid or where the grid itself has no value.
    """
    hm = height_map.depth
    h, w = hm.shape
    half = grid_size_meters / 2.0
    if not (-half <= local_x <= half and -half <= local_z <= half):
        return None

    row = (local_z + half) / grid_size_meters * (h - 1)
    col = (local_x + half) / grid_size_meters * (w - 1)
    r0, c0 = int(np.floor(row)), int(np.floor(col))
    r1, c1 = min(r0 + 1, h - 1), min(c0 + 1, w - 1)
    r0, c0 = max(r0, 0), max(c0, 0)
    fr, fc = row - r0, col - c0

    top = hm[r0, c0] * (1.0 - fc) + hm[r0, c1] * fc
    bottom = hm[r1, c0] * (1.0 - fc) + hm[r1, c1] * fc
    value = float(top * (1.0 - fr) + bottom * fr)
    return value if np.isfinite(value) else None


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
