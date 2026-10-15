"""
Near-ground plate: the terrain material around the camera, baked top-down and
handed to a generative model as the starting image.

The ground the player stands on is where the material is weakest. Within a few
metres the photograph projects onto the ground as a radial smear (one panorama
row covers centimetres at 2 m and metres at 20 m), so beyond ~2 m the pipeline
has nothing real to paint there and falls back to small tiles repeated fifty
times across the terrain, colour-corrected by a blurred macro map. The tiles
carry no structure larger than four metres and the macro carries no structure
smaller than half a metre, so the near field reads as noise at every scale the
eye can resolve from standing height.

This module takes what the material already knows about the near field -- the
blend of panorama projection, region tiles and macro colour that the shader
would draw -- bakes it top-down over a square plate centred on the camera, and
lets a diffusion model repaint that bake as one photograph of the ground seen
from above. The layout survives (where the trail is, where the soil shows, where
the meadow is green) and the detail becomes coherent ground instead of a
repeat. The result is a single world-space layer whose tile factor is an odd
integer: the terrain grid is centred on the camera, so with an odd number of
repeats the centre tile lands exactly on the camera and the shader needs no
offset to place it. Its blend weight is a soft disc, so it fades back into the
tiled layers at its edge.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import PIL.Image
from scipy.ndimage import map_coordinates

from scene.splat_material import SplatLayer, SplatMaterial


def plate_tile_factor(grid_size_m: float, plate_m: float) -> int:
    """The odd repeat count whose cell is closest to plate_m across a grid of
    grid_size_m, at least 1. Odd so the centre cell is centred on the camera."""
    n = max(1.0, grid_size_m / max(plate_m, 1e-3))
    odd = int(round((n - 1.0) / 2.0)) * 2 + 1
    return max(1, odd)


def _sample(img: PIL.Image.Image, u: np.ndarray, v: np.ndarray, wrap: bool) -> np.ndarray:
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    if wrap:
        cols = np.mod(u, 1.0) * w - 0.5
        rows = np.mod(v, 1.0) * h - 0.5
        mode = "wrap"
    else:
        cols = u * (w - 1)
        rows = v * (h - 1)
        mode = "nearest"
    return np.stack([
        map_coordinates(arr[..., c], [rows.ravel(), cols.ravel()], order=1, mode=mode).reshape(u.shape)
        for c in range(4)
    ], axis=-1)


def bake_material_window(
    material: SplatMaterial,
    height_map: np.ndarray,
    half: float,
    size: int,
    plate_half: float,
    height_blend_sharpness: float = 4.0,
    pano_uv_at: Optional[Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = None,
    include_equirect: bool = True,
) -> np.ndarray:
    """TerrainSplat.shader evaluated on a top-down grid over the square
    [-plate_half, plate_half]^2 around the camera (the world origin): blend maps
    at the grid UV, tiled layers at uv * tile_factor, the equirect layer at the
    texel's direction from the camera (or `pano_uv_at(X, Z)` when the caller has
    the mesh's own baked UV), height-biased weights, macro colour on the tiled
    layers that take it. Row 0 is Z = -plate_half, the canvas convention the
    blend maps and the macro map use. Returns (size, size, 3) float32 in [0, 1].

    include_equirect=False leaves the projected panorama out and renormalises
    the tiled layers among themselves: within the plate the projection is the
    radial smear this module exists to replace, and a diffusion model given it
    as the starting image keeps it (the streaks are low-frequency along the
    ray). The tiles and the macro map carry the layout without it."""
    t = (np.arange(size, dtype=np.float64) + 0.5) / size
    PX, PZ = np.meshgrid(t, t)
    X = (PX - 0.5) * 2.0 * plate_half
    Z = (PZ - 0.5) * 2.0 * plate_half
    U = (X + half) / (2.0 * half)
    V = (Z + half) / (2.0 * half)
    hm_h, hm_w = height_map.shape
    Y = map_coordinates(
        height_map, [(V * (hm_h - 1)).ravel(), (U * (hm_w - 1)).ravel()], order=1, mode="nearest",
    ).reshape(size, size)

    layers = list(material.layers)
    weights = []
    for i in range(len(layers)):
        bm = material.blend_maps[i // 4] if i // 4 < len(material.blend_maps) else None
        if bm is None:
            weights.append(np.zeros((size, size), dtype=np.float32))
            continue
        weights.append(_sample(bm, U, V, wrap=False)[..., i % 4])

    texs = []
    biased = []
    for i, layer in enumerate(layers):
        if layer.equirect and not include_equirect:
            texs.append(np.zeros((size, size, 4), dtype=np.float32))
            biased.append(np.zeros((size, size), dtype=np.float32))
            continue
        if layer.equirect:
            if pano_uv_at is not None:
                u, v = pano_uv_at(X, Z)
            else:
                lon = np.arctan2(X, Z)
                lat = np.arctan2(Y, np.sqrt(X ** 2 + Z ** 2).clip(1e-6))
                u = (lon + np.pi) / (2.0 * np.pi)
                v = 0.5 - lat / np.pi
            tex = _sample(layer.tile, u, v, wrap=True)
            tex[..., 3] = 1.0
        else:
            tf = float(layer.tile_factor)
            tex = _sample(layer.tile, U * tf, V * tf, wrap=True)
        texs.append(tex)
        biased.append(weights[i] * np.power(tex[..., 3] + 1e-4, height_blend_sharpness))
    total = sum(biased) + 1e-4

    macro_mod = np.ones((size, size, 3), dtype=np.float32)
    macro = getattr(material, "macro", None)
    strength = float(getattr(material, "macro_strength", 0.0) or 0.0)
    if macro is not None and strength > 0.0:
        mean = np.asarray(getattr(material, "macro_mean", None) or (0.5, 0.5, 0.5), dtype=np.float32)
        m = _sample(macro, U, V, wrap=False)[..., :3]
        macro_mod = 1.0 + strength * (m / np.maximum(mean, 0.02) - 1.0)

    albedo = np.zeros((size, size, 3), dtype=np.float32)
    for i, layer in enumerate(layers):
        w = (biased[i] / total)[..., None]
        colour = texs[i][..., :3]
        if not layer.equirect and getattr(layer, "macro", True):
            colour = colour * macro_mod
        albedo += colour * w
    return albedo.clip(0.0, 1.0).astype(np.float32)


def near_weight_map(size: int, half: float, plate_half: float, full_fraction: float, edge_fraction: float) -> np.ndarray:
    """A soft disc over the blend-map grid (row 0 at Z = -half): 1 within
    full_fraction * plate_half of the camera, smoothstep to 0 at
    edge_fraction * plate_half. The plate is square; the disc stays inside it."""
    t = (np.arange(size, dtype=np.float64) + 0.5) / size
    X, Z = np.meshgrid((t - 0.5) * 2.0 * half, (t - 0.5) * 2.0 * half)
    r = np.sqrt(X ** 2 + Z ** 2)
    r0 = plate_half * full_fraction
    r1 = max(plate_half * edge_fraction, r0 + 1e-3)
    s = ((r - r0) / (r1 - r0)).clip(0.0, 1.0)
    return (1.0 - s * s * (3.0 - 2.0 * s)).astype(np.float32)


def blend_crops(size: int, crop: int, stride: int, generate: Callable[[int, int, PIL.Image.Image], PIL.Image.Image],
                source: PIL.Image.Image, wrap: bool = False) -> PIL.Image.Image:
    """Regenerate `source` (size x size) as overlapping crop x crop windows at
    `stride`, each passed through `generate(x0, y0, crop_image)`, blended back with
    a raised-cosine window so no window edge shows.

    wrap=True treats the image as a torus: windows start every `stride` px all
    the way round and the last ones wrap past the edge, so a seamless input
    stays seamless (with stride = crop / 2 the raised-cosine windows sum to a
    constant everywhere)."""
    acc = np.zeros((size, size, 3), dtype=np.float32)
    wsum = np.zeros((size, size, 1), dtype=np.float32)
    ramp = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(crop) + 0.5) / crop)
    window = (np.outer(ramp, ramp) + 1e-3)[..., None].astype(np.float32)
    src = np.asarray(source.convert("RGB"))
    if wrap:
        starts = list(range(0, size, stride))
    else:
        starts = sorted(set(list(range(0, size - crop + 1, stride)) + [size - crop]))
    for y0 in starts:
        for x0 in starts:
            if wrap:
                rolled = np.roll(src, (-y0, -x0), axis=(0, 1))[:crop, :crop]
                piece = PIL.Image.fromarray(rolled, "RGB")
            else:
                piece = source.crop((x0, y0, x0 + crop, y0 + crop))
            out = np.asarray(generate(x0, y0, piece).convert("RGB").resize((crop, crop), PIL.Image.LANCZOS),
                             dtype=np.float32)
            if wrap:
                a = np.zeros((size, size, 3), dtype=np.float32)
                w = np.zeros((size, size, 1), dtype=np.float32)
                a[:crop, :crop] = out * window
                w[:crop, :crop] = window
                acc += np.roll(a, (y0, x0), axis=(0, 1))
                wsum += np.roll(w, (y0, x0), axis=(0, 1))
            else:
                acc[y0:y0 + crop, x0:x0 + crop] += out * window
                wsum[y0:y0 + crop, x0:x0 + crop] += window
    merged = acc / np.maximum(wsum, 1e-6)
    return PIL.Image.fromarray(merged.clip(0, 255).astype(np.uint8), "RGB")


def make_seamless(image: PIL.Image.Image, repaint: Callable[[PIL.Image.Image, PIL.Image.Image], PIL.Image.Image] | None,
                  band_fraction: float = 0.2, lowpass_fraction: float = 0.05) -> PIL.Image.Image:
    """Make `image` wrap without touching its middle (the ground under the camera).

    Rolled by half, the image's edges meet in a cross through the middle with a
    hard join. Inside a band of `band_fraction` around that cross the rolled
    image is cross-faded with the unrolled one, which is continuous there (it
    is the image's own middle): the classic offset tile, seamless by
    construction but ghosted where the two overlap. `repaint(image, mask)` (an
    inpainting call at low strength; mask white = repaint) then cleans the
    ghosting inside the band, taking only fine detail from it so the band keeps
    the cross-faded brightness. Given the hard join itself instead, the model
    paints a path or ditch along it, which tiles as a dark grid."""
    w, h = image.size
    src = np.asarray(image.convert("RGB"), dtype=np.float32)
    rolled = np.roll(src, (h // 2, w // 2), axis=(0, 1))
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.minimum(np.abs(xx + 0.5 - w / 2.0) / w, np.abs(yy + 0.5 - h / 2.0) / h)  # to the cross, as a fraction
    half_band = band_fraction / 2.0
    t = (1.0 - dist / max(half_band, 1e-6)).clip(0.0, 1.0)
    alpha = (t * t * (3.0 - 2.0 * t))[..., None].astype(np.float32)
    faded = rolled * (1.0 - alpha) + src * alpha
    if repaint is not None:
        mask = (dist <= half_band).astype(np.uint8) * 255
        out = np.asarray(repaint(PIL.Image.fromarray(faded.clip(0, 255).astype(np.uint8), "RGB"),
                                 PIL.Image.fromarray(mask, "L")).convert("RGB").resize((w, h), PIL.Image.LANCZOS),
                         dtype=np.float32)
        if lowpass_fraction > 0.0:
            from scipy.ndimage import gaussian_filter
            sigma = lowpass_fraction * w
            lo = lambda a: np.stack([gaussian_filter(a[..., c], sigma, mode="wrap") for c in range(3)], axis=-1)
            out = out - lo(out) + lo(faded)
        faded = out * alpha + faded * (1.0 - alpha)
    merged = np.roll(faded, (-(h // 2), -(w // 2)), axis=(0, 1))
    return PIL.Image.fromarray(merged.clip(0, 255).astype(np.uint8), "RGB")


def wrap_padded(image: PIL.Image.Image, pad: int, fn: Callable[[PIL.Image.Image], PIL.Image.Image]) -> PIL.Image.Image:
    """Run an image filter that looks at neighbours (supersampling, delighting)
    on `image` padded with its own wrapped-around content, then crop the padding
    off, so a seamless image stays seamless. Handles filters that rescale."""
    arr = np.asarray(image.convert("RGB"))
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="wrap")
    res = fn(PIL.Image.fromarray(padded, "RGB")).convert("RGB")
    scale = res.width / padded.shape[1]
    p = int(round(pad * scale))
    out_w, out_h = int(round(image.width * scale)), int(round(image.height * scale))
    return res.crop((p, p, p + out_w, p + out_h))


def seam_error(image: PIL.Image.Image) -> tuple[float, float]:
    """(wrap-edge difference, interior neighbour difference): mean absolute
    difference between the last and first column/row, and between adjacent
    columns/rows inside. Seamless when the first is close to the second."""
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    edge = (np.abs(a[:, -1] - a[:, 0]).mean() + np.abs(a[-1] - a[0]).mean()) / 2.0
    inner = (np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean()) / 2.0
    return float(edge), float(inner)


def with_layer(material: SplatMaterial, layer: SplatLayer, weight: np.ndarray) -> SplatMaterial:
    """`material` with `layer` appended at blend weight `weight` (blend-map grid,
    [0, 1]); every existing layer's weight is scaled by (1 - weight) so the
    weights still sum to one."""
    weight = weight.astype(np.float32).clip(0.0, 1.0)
    size = weight.shape[0]
    weights: dict[str, np.ndarray] = {}
    for i, existing in enumerate(material.layers):
        bm = material.blend_maps[i // 4]
        arr = np.asarray(bm.convert("RGBA"), dtype=np.float32)[..., i % 4] / 255.0
        if arr.shape != weight.shape:
            arr = np.asarray(PIL.Image.fromarray((arr * 255).astype(np.uint8)).resize((size, size), PIL.Image.BILINEAR),
                             dtype=np.float32) / 255.0
        weights[existing.name] = arr * (1.0 - weight)
    weights[layer.name] = weight
    layers = list(material.layers) + [layer]
    out = SplatMaterial.from_weight_maps(layers=layers, weight_maps=weights)
    out.macro = material.macro
    out.macro_strength = material.macro_strength
    out.macro_mean = material.macro_mean
    return out


def ring_score(image: PIL.Image.Image) -> float:
    """How much the image's texture is arranged in rings or spokes around its
    centre: the coherence-weighted mean of cos(2 x (local orientation - the
    tangent of the circle through that pixel)), in [-1, 1]; |score| near 0 is an
    isotropic ground, toward 1 concentric rings, toward -1 radial spokes.
    FLUX sometimes paints the plate as rings around the camera, which tiles as
    a repeating bullseye."""
    from scipy.ndimage import gaussian_filter, sobel
    g = np.asarray(image.convert("L").resize((512, 512), PIL.Image.LANCZOS), dtype=np.float32)
    g = g - gaussian_filter(g, 8.0)
    gx, gy = sobel(g, axis=1), sobel(g, axis=0)
    jxx = gaussian_filter(gx * gx, 6.0); jyy = gaussian_filter(gy * gy, 6.0); jxy = gaussian_filter(gx * gy, 6.0)
    # Dominant gradient direction; the texture's own orientation is normal to it.
    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy) + np.pi / 2.0
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / (jxx + jyy + 1e-6)
    yy, xx = np.mgrid[0:512, 0:512] - 255.5
    tangent = np.arctan2(yy, xx) + np.pi / 2.0
    r = np.hypot(xx, yy)
    w = coherence * ((r > 40) & (r < 250))
    return float((np.cos(2.0 * (theta - tangent)) * w).sum() / (w.sum() + 1e-6))
