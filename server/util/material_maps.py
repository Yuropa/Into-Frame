"""
PBR texture maps for generated assets: tangent-space normal, ambient occlusion,
roughness and metallic, derived from a base-colour texture and, for real
meshes, from the geometry.

Every asset the pipeline builds arrives with a base colour and nothing else
(or, from TRELLIS, with a metallic-roughness texture that calls a meadow flower
polished metal: metallic 0.74, roughness 0.19 on the Rainier lupine). Under the
client's PBR lighting that reads as flat cut-outs, or as chrome. The maps here
are what a material artist would make from a photo texture:

- height: the texture's own luminance, high-passed at a few scales (or a height
  the caller already has, e.g. a terrain tile's alpha), so crevices are low and
  ridges high;
- normal: the height's gradient, glTF convention (+X right, +Y up in texture
  space, i.e. toward decreasing image row, +Z out of the surface);
- occlusion: cavity occlusion from the same height, times, for a mesh, the
  geometric occlusion baked from depth renders in 64 directions and rasterised
  into its UV layout;
- roughness: a per-class base, rougher in crevices;
- metallic: zero for natural classes; kept (never raised) from the source
  otherwise.

All filters can treat the image as a torus (wrap=True) so a seamless tile stays
seamless.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import PIL.Image
from scipy.ndimage import gaussian_filter, sobel, distance_transform_edt

# Roughness base per object class (0 = mirror, 1 = fully diffuse). Anything not
# listed is treated by NATURAL_CLASSES membership: natural 0.85, built 0.6.
_CLASS_ROUGHNESS = {
    "flower": 0.75, "plant": 0.8, "bush": 0.85, "tree": 0.85, "forest": 0.85,
    "grass": 0.8, "grass_tuft": 0.8, "rock": 0.85, "boulder": 0.85, "stone": 0.85,
    "log": 0.85, "stump": 0.85, "animal": 0.8, "person": 0.7,
    "building": 0.7, "house": 0.7, "bridge": 0.75, "tower": 0.65, "fence": 0.8,
    "boat": 0.55, "car": 0.4, "vehicle": 0.45, "bench": 0.7, "sign": 0.5, "lamp": 0.5,
}
NATURAL_CLASSES = frozenset({
    "flower", "plant", "bush", "tree", "forest", "grass", "grass_tuft", "rock", "boulder",
    "stone", "log", "stump", "animal", "person", "mountain", "hill", "snow", "water",
})


def class_roughness(obj_class: Optional[str]) -> float:
    c = (obj_class or "").lower()
    if c in _CLASS_ROUGHNESS:
        return _CLASS_ROUGHNESS[c]
    return 0.85 if c in NATURAL_CLASSES or not c else 0.6


def _blur(a: np.ndarray, sigma: float, wrap: bool) -> np.ndarray:
    return gaussian_filter(a, sigma, mode="wrap" if wrap else "reflect")


def _fill_masked(a: np.ndarray, valid: Optional[np.ndarray]) -> np.ndarray:
    """Replace values outside `valid` by the nearest valid value, so filters do
    not read the transparent padding of a card or the gutter of a UV atlas."""
    if valid is None or valid.all() or not valid.any():
        return a
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def height_from_image(rgb: np.ndarray, valid: Optional[np.ndarray] = None, wrap: bool = False) -> np.ndarray:
    """A height field in [0, 1] from a texture's luminance: band-passed at a few
    scales and summed, so it carries detail (blades, grains, bark) and not the
    texture's lighting gradients."""
    lum = rgb[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lum = _fill_masked(lum / 255.0 if lum.max() > 1.5 else lum, valid)
    size = max(lum.shape)
    h = np.zeros_like(lum)
    for s, w in ((1.0, 0.5), (3.0, 0.3), (8.0, 0.2)):
        s = s * size / 1024.0
        h += w * (_blur(lum, max(s * 0.5, 0.5), wrap) - _blur(lum, s * 4.0, wrap))
    lo, hi = np.percentile(h, [1, 99])
    return ((h - lo) / max(hi - lo, 1e-6)).clip(0.0, 1.0).astype(np.float32)


def normal_from_height(height: np.ndarray, strength: float = 2.0, wrap: bool = False) -> np.ndarray:
    """Tangent-space normal (H, W, 3) in [-1, 1], glTF convention. `strength` is
    the height range in texels per unit at 1024 px (scaled with resolution, so a
    map looks the same at any size)."""
    mode = "wrap" if wrap else "reflect"
    scale = strength * max(height.shape) / 1024.0
    dx = sobel(height, axis=1, mode=mode) / 8.0 * scale      # d h / d column
    dy = sobel(height, axis=0, mode=mode) / 8.0 * scale      # d h / d row (down)
    # +Y is up in texture space = toward decreasing row, so d h / d(+Y) = -dy.
    n = np.stack([-dx, dy, np.ones_like(height)], axis=-1)
    return (n / np.linalg.norm(n, axis=-1, keepdims=True)).astype(np.float32)


def cavity_occlusion(height: np.ndarray, strength: float = 0.6, wrap: bool = False) -> np.ndarray:
    """Occlusion in [0, 1] from how far a texel sits below its neighbourhood."""
    size = max(height.shape)
    occ = np.ones_like(height)
    for s, w in ((4.0, 0.5), (16.0, 0.5)):
        below = (_blur(height, s * size / 1024.0, wrap) - height).clip(0.0, None)
        occ -= strength * w * (below / max(float(np.percentile(below, 99)), 1e-6)).clip(0.0, 1.0)
    return occ.clip(0.0, 1.0).astype(np.float32)


def encode_normal(n: np.ndarray) -> PIL.Image.Image:
    return PIL.Image.fromarray(((n * 0.5 + 0.5).clip(0, 1) * 255.0 + 0.5).astype(np.uint8), "RGB")


# ── geometric occlusion ─────────────────────────────────────────────────────────

def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], axis=-1)


def vertex_occlusion(vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray,
                     directions: int = 64, resolution: int = 512) -> Optional[np.ndarray]:
    """Per-vertex ambient occlusion in [0, 1]: for each of `directions` view
    directions over the sphere, an orthographic depth render of the mesh decides
    which vertices that direction can see; a vertex's occlusion is the
    cosine-weighted fraction of the directions in its normal's hemisphere that
    reach it. Returns None when no OpenGL context can be made."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        import pyrender
        import trimesh
    except Exception:
        return None
    v = np.asarray(vertices, dtype=np.float64)
    centre = (v.max(axis=0) + v.min(axis=0)) / 2.0
    radius = float(np.linalg.norm(v - centre, axis=1).max()) or 1.0
    vc = v - centre
    tm = trimesh.Trimesh(vertices=vc, faces=np.asarray(faces), process=False)
    try:
        renderer = pyrender.OffscreenRenderer(resolution, resolution)
    except Exception:
        return None
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
    mat = pyrender.MetallicRoughnessMaterial(doubleSided=True, baseColorFactor=[1, 1, 1, 1])
    scene.add(pyrender.Mesh.from_trimesh(tm, material=mat, smooth=False))
    znear, zfar = 0.01 * radius, 4.0 * radius
    cam = pyrender.OrthographicCamera(xmag=radius * 1.02, ymag=radius * 1.02, znear=znear, zfar=zfar)
    cam_node = scene.add(cam, pose=np.eye(4))
    dirs = _fibonacci_sphere(directions)
    nrm = np.asarray(normals, dtype=np.float64)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    seen = np.zeros(len(v)); total = np.zeros(len(v))
    eps = 0.01 * radius
    try:
        for d in dirs:
            # Camera at centre + d * 2r looking toward the centre (-Z of its pose = -d).
            z = d / np.linalg.norm(d)
            up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.95 else np.array([1.0, 0.0, 0.0])
            x = np.cross(up, z); x /= np.linalg.norm(x)
            y = np.cross(z, x)
            pose = np.eye(4)
            pose[:3, 0], pose[:3, 1], pose[:3, 2] = x, y, z
            pose[:3, 3] = z * 2.0 * radius
            scene.set_pose(cam_node, pose)
            depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
            # pyrender linearises with the perspective formula; undo it for an
            # orthographic camera.
            valid = depth > 0
            ndc = np.zeros_like(depth)
            ndc[valid] = (zfar + znear - 2.0 * znear * zfar / depth[valid]) / (zfar - znear)
            lin = np.where(valid, znear + (ndc + 1.0) / 2.0 * (zfar - znear), np.inf)
            rel = vc - pose[:3, 3]
            xc, yc, dc = rel @ x, rel @ y, -(rel @ z)
            col = ((xc / (radius * 1.02) + 1.0) / 2.0 * resolution).astype(int).clip(0, resolution - 1)
            row = ((1.0 - yc / (radius * 1.02)) / 2.0 * resolution).astype(int).clip(0, resolution - 1)
            visible = dc <= lin[row, col] + eps
            w = (nrm @ z).clip(0.0, None)
            seen += w * visible
            total += w
    finally:
        renderer.delete()
    occ = np.where(total > 1e-6, seen / np.maximum(total, 1e-6), 1.0)
    return occ.clip(0.0, 1.0).astype(np.float32)


def rasterize_to_uv(uv: np.ndarray, faces: np.ndarray, values: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex scalar `values` interpolated over the UV layout into a
    (size, size) image (row 0 = v 1, glTF image orientation). Returns (image,
    covered mask). Pure numpy: one bounding-box pass per face, fine for the
    few-thousand-face assets this pipeline exports."""
    img = np.zeros((size, size), dtype=np.float32)
    cov = np.zeros((size, size), dtype=bool)
    px = np.stack([uv[:, 0] * size - 0.5, (1.0 - uv[:, 1]) * size - 0.5], axis=-1)
    for f in np.asarray(faces):
        p = px[f]
        x0, y0 = np.floor(p.min(axis=0)).astype(int)
        x1, y1 = np.ceil(p.max(axis=0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, size - 1), min(y1, size - 1)
        if x1 < x0 or y1 < y0:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        (ax, ay), (bx, by), (cx, cy) = p
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        l0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
        l1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -0.02) & (l1 >= -0.02) & (l2 >= -0.02)
        if not inside.any():
            continue
        val = l0 * values[f[0]] + l1 * values[f[1]] + l2 * values[f[2]]
        img[ys[inside], xs[inside]] = val[inside]
        cov[ys[inside], xs[inside]] = True
    if cov.any() and not cov.all():
        img = _fill_masked(img, cov)       # gutter: nearest texel, so mips do not darken seams
    return img, cov


# ── assets ──────────────────────────────────────────────────────────────────────

def image_detail_usable(uv: Optional[np.ndarray], faces: Optional[np.ndarray], size: int,
                        min_face_px: float = 12.0) -> bool:
    """Whether the texture's own detail can be read as surface relief: false for
    a UV layout whose median face spans fewer than `min_face_px` texels (a
    per-face atlas), true without a layout (a card, a tile)."""
    if uv is None or faces is None or len(faces) == 0:
        return True
    p = np.asarray(uv)[np.asarray(faces)] * size
    edge = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    return float(np.median(edge)) >= min_face_px


def mesh_material_maps(
    base: PIL.Image.Image,
    obj_class: Optional[str],
    uv: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    vertex_ao: Optional[np.ndarray] = None,
    source_mr: Optional[PIL.Image.Image] = None,
    normal_strength: float = 2.0,
    cavity_strength: float = 0.5,
    max_size: int = 1024,
) -> dict:
    """Normal, occlusion and metallic-roughness textures (PIL, glTF channel
    layout) for an asset whose base colour is `base`. Alpha-cut cards pass their
    RGBA base; transparent texels are ignored by every filter."""
    rgba = base.convert("RGBA")
    if max(rgba.size) > max_size:
        s = max_size / max(rgba.size)
        rgba = rgba.resize((max(1, int(rgba.width * s)), max(1, int(rgba.height * s))), PIL.Image.LANCZOS)
    arr = np.asarray(rgba)
    valid = arr[..., 3] > 16
    height = height_from_image(arr, valid=valid)
    detail = image_detail_usable(uv, faces, arr.shape[0])
    if detail:
        normal = normal_from_height(height, normal_strength)
        occ = cavity_occlusion(height, cavity_strength)
    else:
        # A per-face atlas (Mesh.bake_vertex_colours): every face is its own few-
        # texel cell, so image gradients are cell borders, not surface detail.
        height = np.full_like(height, 0.5)
        normal = np.zeros(height.shape + (3,), dtype=np.float32); normal[..., 2] = 1.0
        occ = np.ones_like(height)
    if vertex_ao is not None and uv is not None and faces is not None:
        geo, cov = rasterize_to_uv(np.asarray(uv), np.asarray(faces), np.asarray(vertex_ao), arr.shape[0])
        if geo.shape != occ.shape:
            geo = np.asarray(PIL.Image.fromarray(geo).resize((occ.shape[1], occ.shape[0]), PIL.Image.BILINEAR))
        occ = occ * geo
    rough = (class_roughness(obj_class) + 0.12 * (0.5 - height)).clip(0.05, 1.0)
    metal = np.zeros_like(rough)
    if source_mr is not None and (obj_class or "").lower() not in NATURAL_CLASSES:
        mr = np.asarray(source_mr.convert("RGB").resize((rough.shape[1], rough.shape[0]), PIL.Image.BILINEAR),
                        dtype=np.float32) / 255.0
        metal = mr[..., 2]
        rough = np.maximum(rough * 0.5 + mr[..., 1] * 0.5, 0.2)
    occ_img = PIL.Image.fromarray((np.stack([occ] * 3, axis=-1) * 255.0 + 0.5).astype(np.uint8), "RGB")
    # glTF metallicRoughness: G = roughness, B = metallic (R unused; glTFast and
    # some packers read R as occlusion when the textures are shared, so it is set).
    mr_img = PIL.Image.fromarray((np.stack([occ, rough, metal], axis=-1) * 255.0 + 0.5).astype(np.uint8), "RGB")
    return {"normal": encode_normal(normal), "occlusion": occ_img, "metallic_roughness": mr_img,
            "stats": {"image_detail": bool(detail), "occlusion_mean": float(occ[valid[:occ.shape[0], :occ.shape[1]]].mean()) if valid.any() else 1.0,
                      "roughness_mean": float(rough.mean()), "metallic_mean": float(metal.mean())}}


def terrain_layer_pbr(tile: PIL.Image.Image, smoothness: float, normal_strength: float = 1.5,
                      cavity_strength: float = 0.5, wrap: bool = True) -> PIL.Image.Image:
    """One packed RGBA texture for a terrain splat layer, linear: RG = tangent
    normal xy (0.5 = flat), B = occlusion, A = smoothness. The height is the
    tile's own alpha when it has one (TerrainTextureGeneration packs a local
    micro-height there, but it is built from the same luminance), here its
    luminance, band-passed with wrap-around filters."""
    arr = np.asarray(tile.convert("RGBA"))
    # From the colour, not the tile's alpha micro-height: the colour is what the
    # viewer sees the relief of, and its filters here wrap.
    height = height_from_image(arr, wrap=wrap)
    normal = normal_from_height(height, normal_strength, wrap=wrap)
    occ = cavity_occlusion(height, cavity_strength, wrap=wrap)
    smooth = (smoothness + 0.1 * (height - 0.5)).clip(0.0, 1.0)
    out = np.stack([normal[..., 0] * 0.5 + 0.5, normal[..., 1] * 0.5 + 0.5, occ, smooth], axis=-1)
    return PIL.Image.fromarray((out.clip(0, 1) * 255.0 + 0.5).astype(np.uint8), "RGBA")
