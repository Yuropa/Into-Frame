"""A card texture rasterised from a species mesh (no GPU, no Blender).

Scene Generation and the client's runtime LOD swap a plant for its bucket's card
beyond the mesh distance, and the card has been the photo crop's cutout: delit,
matted with meadow and a different colour from the species mesh it stands in for
(Rainier: 1700 pale cards behind 300 coloured meshes). The species mesh is a
palette-textured low-poly plant whose every face sits in one swatch, so a front
view of it is a painter's-algorithm fill of flat-coloured triangles, which PIL
draws in milliseconds. The result carries the mesh's own colours at every distance.
"""
from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image as PILImage, ImageDraw


def species_card(mesh: trimesh.Trimesh, height_px: int = 256, light=(0.35, 0.8, 0.5)) -> PILImage.Image:
    """RGBA front view (looking along -Z, Y up) of a palette-textured mesh."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    ext = np.maximum(hi - lo, 1e-6)
    ppm = (height_px - 4) / ext[1]
    width_px = int(np.ceil(ext[0] * ppm)) + 4
    img = PILImage.new("RGBA", (max(width_px, 8), height_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    tex = None
    uv = None
    visual = getattr(mesh, "visual", None)
    if visual is not None and getattr(visual, "uv", None) is not None:
        uv = np.asarray(visual.uv, dtype=np.float64)
        mat = getattr(visual, "material", None)
        tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        if tex is not None:
            tex = tex.convert("RGB")
    base = np.array([[120, 140, 80]] * len(faces), dtype=np.float64)
    if tex is not None and uv is not None and len(uv) == len(verts):
        tw, th = tex.size
        tpx = np.asarray(tex, dtype=np.float64)
        c = uv[faces].mean(axis=1)
        px = np.clip((c[:, 0] * tw).astype(int), 0, tw - 1)
        py = np.clip(((1.0 - c[:, 1]) * th).astype(int), 0, th - 1)
        base = tpx[py, px]
    # Flat shading from a key light above and in front, kept gentle: the client
    # lights the card again.
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    l = np.asarray(light, dtype=np.float64)
    l /= np.linalg.norm(l)
    shade = 0.7 + 0.3 * np.abs(n @ l)
    colour = np.clip(base * shade[:, None], 0, 255).astype(int)
    depth = tri[:, :, 2].mean(axis=1)
    order = np.argsort(depth)  # far (−Z) first, near last on top
    cx = (lo[0] + hi[0]) * 0.5
    for i in order:
        pts = [((v[0] - cx) * ppm + img.width * 0.5, (hi[1] - v[1]) * ppm + 2.0) for v in tri[i]]
        r, g, b = (int(v) for v in colour[i])
        draw.polygon(pts, fill=(r, g, b, 255))
    return img
