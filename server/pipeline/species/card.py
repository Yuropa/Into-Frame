"""A card texture rasterised from a species mesh (no GPU, no Blender).

Scene Generation and the client's runtime LOD swap a plant for its bucket's card
beyond the mesh distance, and the card has been the photo crop's cutout: delit,
matted with meadow and a different colour from the species mesh it stands in for
(Rainier: 1700 pale cards behind 300 coloured meshes). The species mesh is a
textured low-poly plant -- every face either sits in one palette swatch or is a
card cut from the species atlas (pipeline/species/atlas.py) -- so a front view of
it is a painter's-algorithm fill of textured triangles, which numpy draws in a
fraction of a second. The result carries the mesh's own colours and cutouts at
every distance.
"""
from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image as PILImage


def species_card(mesh: trimesh.Trimesh, height_px: int = 256, light=(0.35, 0.8, 0.5)) -> PILImage.Image:
    """RGBA front view (looking along -Z, Y up) of a textured mesh. Triangles are
    drawn far to near; each pixel inside a triangle samples the texture at its
    interpolated UV (nearest) and is written when its alpha passes the cutout."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    ext = np.maximum(hi - lo, 1e-6)
    ppm = (height_px - 4) / ext[1]
    width_px = max(int(np.ceil(ext[0] * ppm)) + 4, 8)
    out = np.zeros((height_px, width_px, 4), dtype=np.uint8)

    tex = None
    uv = None
    visual = getattr(mesh, "visual", None)
    if visual is not None and getattr(visual, "uv", None) is not None:
        uv = np.asarray(visual.uv, dtype=np.float64)
        mat = getattr(visual, "material", None)
        tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        if tex is not None:
            tex = np.asarray(tex.convert("RGBA"), dtype=np.uint8)
    if tex is None or uv is None or len(uv) != len(verts):
        tex = np.array([[[120, 140, 80, 255]]], dtype=np.uint8)
        uv = np.zeros((len(verts), 2))
    th, tw = tex.shape[:2]

    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    l = np.asarray(light, dtype=np.float64)
    l /= np.linalg.norm(l)
    # Flat shading from a key light above and in front, kept gentle: the client
    # lights the card again.
    shade = 0.7 + 0.3 * np.abs(n @ l)
    depth = tri[:, :, 2].mean(axis=1)
    order = np.argsort(depth)  # far (−Z) first, near last on top
    cx = (lo[0] + hi[0]) * 0.5
    # Screen coordinates: x right, y down.
    sx = (tri[:, :, 0] - cx) * ppm + width_px * 0.5
    sy = (hi[1] - tri[:, :, 1]) * ppm + 2.0
    fuv = uv[faces]
    for i in order:
        x0, x1 = int(np.floor(sx[i].min())), int(np.ceil(sx[i].max()))
        y0, y1 = int(np.floor(sy[i].min())), int(np.ceil(sy[i].max()))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, width_px - 1), min(y1, height_px - 1)
        if x1 < x0 or y1 < y0:
            continue
        px, py = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, bx, cx_), (ay, by, cy_) = sx[i], sy[i]
        det = (bx - ax) * (cy_ - ay) - (cx_ - ax) * (by - ay)
        if abs(det) < 1e-9:
            continue
        w1 = ((px - ax) * (cy_ - ay) - (cx_ - ax) * (py - ay)) / det
        w2 = ((bx - ax) * (py - ay) - (px - ax) * (by - ay)) / det
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -1e-3) & (w1 >= -1e-3) & (w2 >= -1e-3)
        if not inside.any():
            continue
        u = w0 * fuv[i, 0, 0] + w1 * fuv[i, 1, 0] + w2 * fuv[i, 2, 0]
        v = w0 * fuv[i, 0, 1] + w1 * fuv[i, 1, 1] + w2 * fuv[i, 2, 1]
        tx = np.clip((u * tw).astype(int), 0, tw - 1)
        ty = np.clip(((1.0 - v) * th).astype(int), 0, th - 1)
        sample = tex[ty, tx]
        keep = inside & (sample[..., 3] > 127)
        if not keep.any():
            continue
        rgb = np.clip(sample[..., :3].astype(np.float64) * shade[i], 0, 255).astype(np.uint8)
        block = out[y0:y1 + 1, x0:x1 + 1]
        block[keep, :3] = rgb[keep]
        block[keep, 3] = 255
    return PILImage.fromarray(out, "RGBA")
