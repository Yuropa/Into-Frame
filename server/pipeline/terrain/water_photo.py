"""The photographed water, as the water sheet's texture.

The water sheet used to carry one colour (TerrainMeshStage._colour_water_from_photo):
flat, with none of the photograph's ripple, wake, reflections or its near-to-far
gradient. The terrain already wears the panorama through a baked second UV set
(TEXCOORD_1); this gives the water the same.

    build_water_photo_texture(context)  ->  RGBA panorama-sized image, or None
    attach_water_photo(mesh, texture)   ->  Mesh with TEXCOORD_1 and the texture embedded

The texture is the object-free terrain panorama (ContextKey.PANORAMA_TERRAIN, what the
terrain's own panorama layer samples). Its alpha is the weight: 1 on water-typed
pixels, 0 elsewhere, feathered over a couple of pixels: where the sheet projects onto a
bank, a boat's hull or the sky the photograph says nothing about the water. The RGB
there is the nearby water's colour carried outward (no detail), which a consumer may
show in place of the sheet's single colour factor.

Two corrections make the outpainted part of the panorama usable as a surface:
  * objects the foreground inpainting left in the water (Paris's tour boat came through
    half-smeared) are filled from the water around them;
  * outside the photographed cone the low frequencies are pulled to the photograph's
    own near-to-far water colour profile (per panorama row), so the outpainting's
    large blotches and painted sun paths go while its ripple stays. Inside the cone
    the photograph is kept as it is.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import PIL.Image
import trimesh
from scipy.ndimage import binary_dilation, binary_erosion, find_objects, gaussian_filter, label

from pipeline.pipeline_context import ContextKey, PipelineContext
from scene.mesh import Mesh


def _resized_types(depth, W: int, H: int) -> "np.ndarray | None":
    if depth is None:
        return None
    t = np.rint(np.asarray(depth.depth)).astype(np.int64)
    if t.shape != (H, W):
        rows = (np.arange(H) * t.shape[0]) // H
        cols = (np.arange(W) * t.shape[1]) // W
        t = t[rows][:, cols]
    return t


def _mask(image, W: int, H: int) -> "np.ndarray | None":
    if image is None:
        return None
    img = image.image if hasattr(image, "image") else image
    return np.asarray(img.convert("L").resize((W, H), PIL.Image.NEAREST)) > 127


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of an (H, W) or (H, W, C) panorama array: longitude wraps."""
    if a.ndim == 2:
        return gaussian_filter(a, sigma, mode=("nearest", "wrap"))
    return np.stack([gaussian_filter(a[..., c], sigma, mode=("nearest", "wrap")) for c in range(a.shape[-1])], axis=-1)


def _masked_low(rgb: np.ndarray, mask: np.ndarray, sigma: float, step: int) -> np.ndarray:
    """Low frequencies of rgb over mask only (normalised convolution), computed on a
    grid `step` times coarser and brought back up. Zero-weight pixels come back NaN."""
    small = rgb[::step, ::step]
    m = mask[::step, ::step].astype(np.float32)
    num = _blur(small * m[..., None], sigma / step)
    den = _blur(m, sigma / step)
    low = np.where(den[..., None] > 1e-3, num / np.maximum(den[..., None], 1e-3), np.nan)
    H, W = rgb.shape[:2]
    rows = np.minimum(np.arange(H) // step, low.shape[0] - 1)
    cols = np.minimum(np.arange(W) // step, low.shape[1] - 1)
    return low[rows][:, cols]


def build_water_photo_texture(
    context: PipelineContext,
    panorama_key: str = ContextKey.PANORAMA_TERRAIN,
    max_width: int = 4096,
    outside_flatten: float = 0.9,
    log=None,
) -> Optional[PIL.Image.Image]:
    from pipeline.panorama_segmentation.panorama_region_result import RegionType
    from util.captured_cone import captured_cone_from_context

    pano = context.input_panorama(panorama_key) or context.input_panorama(ContextKey.PANORAMA)
    if pano is None:
        return None
    img = pano.rgb()
    if img.width > max_width:
        img = img.resize((max_width, max_width // 2), PIL.Image.LANCZOS)
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    H, W = rgb.shape[:2]

    t_terrain = _resized_types(context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP_TERRAIN), W, H)
    t_original = _resized_types(context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP), W, H)
    if t_terrain is None:
        t_terrain = t_original
    if t_terrain is None:
        return None
    water = t_terrain == int(RegionType.WATER)
    if water.sum() < 500:
        return None

    # Water the photograph (or the outpainting) shows unobstructed: typed water before
    # the objects were taken out as well, and clear of the foreground masks.
    clean = water.copy()
    if t_original is not None:
        clean &= t_original == int(RegionType.WATER)
    for key in (ContextKey.PANORAMA_FOREGROUND_OCCLUDER_MASK, ContextKey.PANORAMA_FOREGROUND_MASK):
        fg = _mask(context.input_image(key), W, H)
        if fg is not None:
            clean &= ~binary_dilation(fg, iterations=max(2, W // 1024))
            break
    if clean.sum() < 0.4 * water.sum():
        clean = water.copy()  # the two typings disagree on most of it: trust the terrain one
    fill = water & ~clean

    step = max(1, W // 1024)
    med_all = np.median(rgb[clean][:: max(1, int(clean.sum()) // 200000)], axis=0)

    # The photograph's own near-to-far water colour: the median per panorama row of
    # the unobstructed water inside the photographed cone, smoothed and carried past
    # the photographed rows at its end values.
    cone = captured_cone_from_context(context, W, H)
    inside = np.zeros((H, W), dtype=bool)
    target = None
    n_cone = 0
    if cone is not None:
        (u0, u1), (v0, v1) = cone.u_range, cone.v_range
        inside[int(max(v0, 0)):int(min(v1, H)), int(max(u0, 0)):int(min(u1, W))] = True
        cone_water = clean & inside
        n_cone = int(cone_water.sum())
        rows = np.flatnonzero(cone_water.sum(axis=1) >= 24)
        if n_cone >= 2000 and rows.size >= 8:
            target = np.empty((H, 3), dtype=np.float32)
            per_row = np.stack([np.median(rgb[r][cone_water[r]], axis=0) for r in rows])
            for c in range(3):
                target[:, c] = gaussian_filter(np.interp(np.arange(H), rows, per_row[:, c]),
                                               max(2.0, H / 256.0), mode="nearest")

    # Objects left in the water: water from the same panorama rows beside them (the
    # same distance, so the same ripple scale), else the row's water colour where the
    # photograph gives one, else the water around them.
    if fill.any():
        if target is not None:
            around = np.broadcast_to(target[:, None, :], rgb.shape).copy()
        else:
            around = _masked_low(rgb, clean, sigma=W / 160.0, step=step)
            around = np.where(np.isnan(around), med_all, around)
        grown = binary_dilation(fill, iterations=max(3, W // 512))
        labels, _ = label(grown)
        cols = np.arange(W)
        for i, box in enumerate(find_objects(labels), start=1):
            rs, cs = box
            width = cs.stop - cs.start
            if width < 4 or width > W // 4:
                continue
            part = labels[rs] == i
            done = np.zeros_like(part)
            for shift in (-(width + 2), width + 2, -2 * (width + 2), 2 * (width + 2)):
                src = (cols + shift) % W
                ok = part & ~done & clean[rs][:, src] & ~grown[rs][:, src]
                around[rs][ok] = rgb[rs][:, src][ok]
                done |= ok
        soft = _blur(grown.astype(np.float32), max(1.5, W / 1024.0))[..., None]
        rgb = rgb * (1.0 - soft) + around * soft

    # Outside the photographed cone: that profile in place of the outpainting's own
    # low frequencies (its blotches), and its ripple kept no stronger than the
    # photograph's (its painted sun paths).
    if target is not None and outside_flatten > 0:
        low = _masked_low(rgb, water, sigma=W / 256.0, step=step)
        low = np.where(np.isnan(low), target[:, None, :], low)
        high = rgb - low
        limit = float(np.percentile(np.abs(high[clean & inside]), 97.0))
        k = float(outside_flatten)
        flat = np.clip(high, -limit, limit) + (low * (1.0 - k) + target[:, None, :] * k)
        c_w = _blur(inside.astype(np.float32), W / 48.0)[..., None]
        rgb = rgb * c_w + flat * (1.0 - c_w)

    # Weight: the water mask, pulled one pixel in from its edge and feathered.
    alpha = _blur(binary_erosion(water, iterations=1).astype(np.float32), 1.2)
    alpha = np.clip((alpha - 0.15) / 0.7, 0.0, 1.0)
    # Colour under zero weight: the water's own colour carried outward, from the
    # nearest water first and from wider neighbourhoods further out (a bilinear or
    # mip sample at the mask's edge never mixes the bank in, and a consumer that
    # shows the sheet where the photograph had a bank, a hull or a rock has the local
    # water colour to show there rather than one colour for the whole sheet).
    carried = np.broadcast_to(med_all.astype(np.float32), rgb.shape)
    for sigma in (W / 16.0, W / 64.0, W / 256.0):
        st = step * (4 if sigma > W / 100.0 else 1)
        m = water[::st, ::st].astype(np.float32)
        den = _blur(m, sigma / st)
        low = _blur(rgb[::st, ::st] * m[..., None], sigma / st) / np.maximum(den[..., None], 1e-4)
        rr = np.minimum(np.arange(H) // st, low.shape[0] - 1)
        cc = np.minimum(np.arange(W) // st, low.shape[1] - 1)
        c = np.clip(den[rr][:, cc] * 8.0, 0.0, 1.0)[..., None]
        carried = low[rr][:, cc] * c + carried * (1.0 - c)
    a3 = np.clip(_blur(water.astype(np.float32), 2.0), 0.0, 1.0)[..., None]
    rgb = rgb * a3 + carried * (1.0 - a3)

    out = np.dstack([np.clip(rgb, 0, 255), alpha * 255.0]).astype(np.uint8)
    if log is not None:
        log(f"Water photograph: {W}x{H} from {panorama_key}, {water.mean():.0%} of the panorama is water, "
            f"{fill.sum() / max(water.sum(), 1):.1%} of it filled over objects, "
            + (f"{n_cone} photographed water pixels set the colour profile outside the cone" if n_cone >= 2000
               else "no photographed cone water: the outpainting is used as it is"))
    return PIL.Image.fromarray(out, "RGBA")


def water_photo_uv(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Panorama UVs of a water mesh's vertices, in Mesh.extra_uv's convention (the
    terrain's: u = column / (W - 1), v = 1 - row / (H - 1)), with the triangles that
    straddle the longitude seam given their own unwrapped vertices. The sheet is a
    plane, so the projection is exact down to the nadir."""
    from pipeline.terrain.terrain_generator import TerrainMeshGenerator
    from util.panorama_utils import Panorama

    lat, lon = Panorama._lat_lon(np.asarray(vertices, dtype=np.float64))
    uv = np.stack([(lon + np.pi) / (2.0 * np.pi), 1.0 - (0.5 - lat / np.pi)], axis=-1).astype(np.float32)
    return TerrainMeshGenerator._fix_uv_seam(np.asarray(vertices), np.asarray(faces), uv)


def attach_water_photo(mesh: Mesh, texture: Optional[PIL.Image.Image]) -> Mesh:
    """`mesh` with the photographed water on it: TEXCOORD_1 (Mesh.extra_uv) and the
    RGBA texture embedded as its material's base colour texture (Mesh.preview_image),
    the material's own colour factor kept. Returns the mesh unchanged without a texture."""
    if mesh is None or texture is None:
        return mesh
    tm = mesh.mesh
    vertices, faces, uv = water_photo_uv(np.asarray(tm.vertices), np.asarray(tm.faces))
    material = tm.visual.material if hasattr(tm.visual, "material") else None
    out_tm = trimesh.Trimesh(
        vertices=vertices, faces=faces,
        vertex_normals=np.tile(np.array([[0.0, 1.0, 0.0]]), (len(vertices), 1)),
        visual=trimesh.visual.TextureVisuals(material=material), process=False,
    )
    out = Mesh(out_tm)
    out.extra_uv = uv
    out.preview_image = texture
    out.preview_overlay = True
    return out
