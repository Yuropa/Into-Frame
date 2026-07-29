"""
Crossed-card geometry for the far grass LOD.

A grass instance beyond the mesh LOD distance can't afford a reconstructed mesh,
but it also can't use SceneGenerationStage's normal billboard fallback: a
billboard is a single quad that rotates to face the camera (see Billboard.cs),
which is the right behaviour for an upright subject seen from eye level and the
wrong one for ground cover. Grass is looked at from above as often as from the
side, and a camera-facing quad lying under the viewer collapses to a line and
then swings through the ground plane as the head turns.

Two or three quads intersecting along a shared vertical axis solve that: fixed
orientation, plausible silhouette from any yaw, and no per-frame billboard
rotation. Because the result is an ordinary textured mesh it needs no client
change at all -- SceneGenerationStage places it through its existing MESH branch,
CategoryMeshRiggingStage bakes the same 3-bone sway skeleton into it that a
reconstructed vegetation mesh gets (it only needs a Y extent), and WindSway
drives it at runtime. That is what makes the far LOD animate rather than sit
frozen next to swaying near-LOD clumps.

The alpha channel of the supplied patch is what makes this read as grass rather
than as intersecting rectangles, so the texture must be an RGBA cutout. Unlike a
billboard, this carries its own glTF material rather than the shared Billboard
material, so the cutout is declared on it directly (alphaMode MASK, cutoff 0.5).
"""

import numpy as np
import trimesh
from PIL import Image as PILImage

from scene.mesh import Mesh


def crossed_card_mesh(texture: PILImage.Image, plane_count: int = 3) -> Mesh:
    """A unit-height crossed-card mesh carrying `texture` on every plane.

    The mesh is built with its base at y=0 and its top at y=1, spanning
    [-0.5, 0.5] in the horizontal axes, so:

      - CategoryMeshRiggingStage's bone chain (bounds[0][1] -> bounds[1][1])
        lands base-to-tip along the blade, which is what makes the sway pivot at
        the ground instead of the middle.
      - SceneGenerationStage's own `mesh_min_y = bounds[0][1] * mesh_scale` base
        snap resolves to 0, i.e. the blades meet the terrain exactly, with no
        reliance on the centroid-vs-bbox assumption its comment warns about for
        reconstructed meshes.

    Planes are evenly spaced in yaw (2 planes -> a cross, 3 -> a 60-degree fan).
    Each plane carries both windings in addition to the material's own
    doubleSided flag: grass has no meaningful back face, and a mesh that reads
    correctly even if an importer drops or ignores doubleSided costs 6 extra
    triangles per instance here.
    """
    if plane_count < 1:
        raise ValueError(f"plane_count must be >= 1, got {plane_count}")

    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    uvs: list[tuple[float, float]] = []

    for plane in range(plane_count):
        yaw = np.pi * plane / plane_count
        dx, dz = float(np.cos(yaw)) * 0.5, float(np.sin(yaw)) * 0.5
        base = len(vertices)

        # Corner order: bottom-left, bottom-right, top-right, top-left.
        vertices.extend([
            np.array([-dx, 0.0, -dz]),
            np.array([+dx, 0.0, +dz]),
            np.array([+dx, 1.0, +dz]),
            np.array([-dx, 1.0, -dz]),
        ])
        # trimesh's glTF export flips V, so author V with 0 at the image bottom
        # to land the texture upright on the blade -- the same convention
        # Panorama.mesh_uvs applies for its own exported meshes.
        uvs.extend([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

        faces.append((base + 0, base + 1, base + 2))
        faces.append((base + 0, base + 2, base + 3))
        # Reversed winding for the back face.
        faces.append((base + 2, base + 1, base + 0))
        faces.append((base + 3, base + 2, base + 0))

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture.convert("RGBA"),
        # Grass is matte; the shared vegetation look elsewhere in the scene uses
        # the same near-zero smoothness (see _LAYER_SMOOTHNESS for the terrain's
        # own vegetation layer).
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode="MASK",
        alphaCutoff=0.5,
        doubleSided=True,
    )
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        visual=trimesh.visual.TextureVisuals(
            uv=np.asarray(uvs, dtype=np.float64), material=material,
        ),
        process=False,   # merging the coincident axis vertices would weld the planes
    )
    return Mesh(mesh)
