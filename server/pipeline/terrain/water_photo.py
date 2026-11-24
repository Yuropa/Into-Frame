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
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

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


def _carried(rgb: np.ndarray, mask: np.ndarray, fallback: np.ndarray, step: int) -> np.ndarray:
    """The colour of `mask`'s pixels carried over the whole image: from the nearest
    masked pixels first and from wider neighbourhoods further out, `fallback` where
    none is in reach."""
    H, W = mask.shape
    carried = np.broadcast_to(np.asarray(fallback, dtype=np.float32), rgb.shape)
    for sigma in (W / 16.0, W / 64.0, W / 256.0):
        st = step * (4 if sigma > W / 100.0 else 1)
        m = mask[::st, ::st].astype(np.float32)
        den = _blur(m, sigma / st)
        low = _blur(rgb[::st, ::st] * m[..., None], sigma / st) / np.maximum(den[..., None], 1e-4)
        rr = np.minimum(np.arange(H) // st, low.shape[0] - 1)
        cc = np.minimum(np.arange(W) // st, low.shape[1] - 1)
        c = np.clip(den[rr][:, cc] * 8.0, 0.0, 1.0)[..., None]
        carried = low[rr][:, cc] * c + carried * (1.0 - c)
    return carried


def _mirror_fill(rgb: np.ndarray, source: np.ndarray, passes: int = 4,
                 min_run: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Water for the pixels that have none, from the same panorama row (the same
    distance, so the same ripple scale): each is given the water mirrored across the
    nearest edge of the row's water, so the ripple runs on from the edge with no seam;
    a gap wider than the water beside it is finished by further passes over what the
    earlier ones filled. Longitude wraps. Returns the filled image, which pixels
    hold water afterwards (rows with almost no water keep their gaps), and each
    pixel's distance along its row to the source water on its nearer and on its
    further side."""
    H, W = source.shape
    out = rgb.copy()
    have = source.copy()
    cols = np.arange(3 * W, dtype=np.int64)[None, :]
    nearer = further = None
    for _ in range(passes):
        if have.all():
            break
        m = np.concatenate([have, have, have], axis=1)
        left = np.maximum.accumulate(np.where(m, cols, -10 * W), axis=1)        # last water at or left of c
        right = np.minimum.accumulate(np.where(m, cols, 10 * W)[:, ::-1], axis=1)[:, ::-1]
        c = cols[:, W:2 * W]
        dl = (c - left[:, W:2 * W])
        dr = (right[:, W:2 * W] - c)
        if nearer is None:
            nearer, further = np.minimum(dl, dr), np.maximum(dl, dr)
        use_left = dl <= dr
        d = np.where(use_left, dl, dr)
        src = np.where(use_left, c - 2 * d + 1, c + 2 * d - 1)
        ok = ~have & (d < W) & (have.sum(axis=1, keepdims=True) >= min_run)
        src_mod = np.mod(src, W)
        rr = np.broadcast_to(np.arange(H)[:, None], (H, W))
        ok &= have[rr, src_mod]
        if not ok.any():
            break
        out[ok] = out[rr[ok], src_mod[ok]]
        have = have | ok
    if nearer is None:
        nearer = further = np.zeros((H, W), dtype=np.int64)
    return out, have, nearer, further



def _open_water_ripple(H: int, W: int, seed: int = 0) -> np.ndarray:
    """A unit-strength ripple pattern (H, W, 1) for open water nothing photographed:
    band-limited noise stretched along the rows, finer and more stretched toward the
    horizon the way a rippled plane foreshortens (three scales, chosen per row by its
    angle under the horizon). Wraps in longitude; the same seed gives the same pattern."""
    rng = np.random.default_rng(seed)
    k = W / 4096.0
    layers = []
    for sy, sx in ((0.7, 6.0), (1.2, 6.0), (2.2, 7.0)):
        n = rng.standard_normal((H, W)).astype(np.float32)
        n = gaussian_filter(n, sigma=(max(0.5, sy * k), max(1.0, sx * k)), mode=("nearest", "wrap"))
        layers.append(n / max(float(n.std()), 1e-6))
    below = np.degrees((np.arange(H) / H - 0.5) * np.pi)
    f = np.clip(np.log2(np.maximum(below, 0.1) / 2.5), 0.0, len(layers) - 1.0)
    out = np.zeros((H, W), dtype=np.float32)
    for i, n in enumerate(layers):
        out += np.clip(1.0 - np.abs(f - i), 0.0, 1.0)[:, None] * n
    # Nothing at or above the horizon (no water plane maps there), full by 1 degree under.
    out *= np.clip(below, 0.0, 1.0)[:, None]
    return np.clip(out, -2.5, 2.5)[..., None]


def build_water_photo_texture(
    context: PipelineContext,
    panorama_key: str = ContextKey.PANORAMA_TERRAIN,
    max_width: int = 4096,
    outside_flatten: float = 0.9,
    mirror_fill: bool = True,
    seed: int = 0,
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
    ripple_amp = None
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
            # And its ripple's strength per row (median absolute deviation from the
            # water's own low frequencies), for the open water nothing photographed.
            low0 = _masked_low(rgb, clean, sigma=max(2.0, W / 256.0), step=step)
            dev = np.abs(rgb - np.where(np.isnan(low0), rgb, low0))
            per_row = np.stack([np.median(dev[r][cone_water[r]], axis=0) for r in rows]) * 1.48
            ripple_amp = np.empty((H, 3), dtype=np.float32)
            # (a row of surf is not a row of ripple: no row above the usual one)
            per_row = np.minimum(per_row, np.median(per_row, axis=0, keepdims=True))
            for c in range(3):
                ripple_amp[:, c] = gaussian_filter(np.interp(np.arange(H), rows, per_row[:, c]),
                                                   max(2.0, H / 256.0), mode="nearest")

    # Where the panorama has no water of its own to show: the objects left in the water,
    # and everything that is not water at all -- under a hull, a rock or a bank the
    # sheet still shows (the reconstructed water is wider than the panorama's, and the
    # object that stood there is drawn somewhere else or smaller). Each such pixel gets
    # the colour of the unobstructed water around it (its low frequencies, from the
    # nearest water first) and the ripple of its own panorama row mirrored across the
    # gap's edge (the same distance, so the same ripple scale). A copy of the water
    # beside the gap carried that water's colour with it (the tower's dark reflection
    # column beside Paris's tour boat came out as a wedge the boat's shape), and one
    # smooth colour in a rippled river reads as the object's ghost.
    have = water
    if mirror_fill:
        source = binary_erosion(clean, iterations=max(2, W // 1024))
        if fill.any():
            source &= ~binary_dilation(fill, iterations=max(3, W // 512))
        if source.sum() >= 500:
            filled, have, nearer, further = _mirror_fill(rgb, source)
            sig = max(2.0, W / 256.0)
            detail = filled - _masked_low(filled, have, sigma=sig, step=1 if W <= 2048 else 2)
            detail = np.where(np.isnan(detail) | ~have[..., None], 0.0, detail)
            # No stronger than the unobstructed water's usual ripple (three times its
            # median amplitude): what else the mirror brings (a moored hull typed as
            # water, a shadow line, a line of surf) is not ripple.
            limit = 3.0 * float(np.median(np.abs(detail[source][:: max(1, int(source.sum()) // 200000)])))
            detail = np.clip(detail, -limit, limit)
            # A gap with water on both sides of it (a hull in a river) is finished
            # whole. Past the water's outer edge the ripple runs a couple of degrees on
            # and fades: the sheet under a bank is covered, a cove is not repeated
            # around the horizon (Shark Fin's 2% of water came back as a kaleidoscope
            # of its own surf).
            enclosed = further <= W // 6
            fade = np.where(enclosed, 1.0, np.clip(1.0 - nearer / (W / 48.0), 0.0, 1.0)).astype(np.float32)
            fade = _blur(fade, max(1.0, W / 1024.0))
            detail = detail * fade[..., None]
            have = have & (fade > 0.02)
            synth = _carried(rgb, source, med_all, step) + detail
            soft = np.clip(_blur(source.astype(np.float32), max(1.5, W / 1024.0)) * 2.0 - 1.0, 0.0, 1.0)[..., None]
            rgb = rgb * soft + synth * (1.0 - soft)
            have = have | water
    elif fill.any():
        if target is not None:
            around = np.broadcast_to(target[:, None, :], rgb.shape).copy()
        else:
            around = _masked_low(rgb, clean, sigma=W / 160.0, step=step)
            around = np.where(np.isnan(around), med_all, around)
        grown = binary_dilation(fill, iterations=max(3, W // 512))
        soft = _blur(grown.astype(np.float32), max(1.5, W / 1024.0))[..., None]
        rgb = rgb * (1.0 - soft) + around * soft

    # Outside the photographed cone: that profile in place of the outpainting's own
    # low frequencies (its blotches), and its ripple kept no stronger than the
    # photograph's (its painted sun paths).
    if target is not None and outside_flatten > 0:
        low = _masked_low(rgb, have, sigma=W / 256.0, step=step)
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
    # Colour under zero weight where no row water reached either. Beside the water:
    # its own colour carried outward, so a bilinear or mip sample at the edge never
    # mixes the bank in. Further off (the sheet runs on where the panorama shows a
    # headland, and to the horizon beside a cove): the photograph's colour for that
    # row with a ripple of that row's strength. The carried colour alone is the mean
    # of the cove, surf included, and unlit and smooth it stood beside Shark Fin's
    # sea as a pale concrete slab.
    a3 = np.clip(_blur(have.astype(np.float32), 2.0), 0.0, 1.0)[..., None]
    outer = _carried(rgb, have, med_all, step)
    if target is not None and ripple_amp is not None:
        beside = np.clip(_blur(have.astype(np.float32), W / 48.0) * 2.0, 0.0, 1.0)[..., None]
        outer = outer * beside + target[:, None, :] * (1.0 - beside)
        # At half the photographed strength: the pattern is smoother than a real
        # ripple, and at the same strength it shows more.
        outer = outer + _open_water_ripple(H, W, seed) * (0.5 * ripple_amp[:, None, :])
    rgb = rgb * a3 + outer * (1.0 - a3)

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
