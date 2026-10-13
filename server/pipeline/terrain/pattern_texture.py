"""
Real-material terrain texturing for the region TerrainTextureGenerationStage's
panorama layer rejects (nadir hole, horizon band, occlusion gaps where the
underlying height itself was interpolated rather than measured -- see
_panorama_visibility_weight's observed_mask gate). Adapted from Neyret &
Cani's "Pattern-Based Texturing Revisited" (SIGGRAPH '99 -- see
resources.bib): a small library of triangular samples, each built so its
value and gradient match a fixed condition on every edge and every corner,
tiled non-periodically across the terrain mesh's own triangles with no
visible seams and no global UV unwrap.

Where this diverges from the paper, and why (validated via a standalone
prototype before landing here -- see the plan at
agile-shimmying-jellyfish.md):

  - The paper needs a per-triangle *runtime* texture-mesh + GPU sampling
    scheme because it targets arbitrary closed surfaces with no usable
    global parameterization. Our terrain is a planar heightfield with a
    trivial global top-down parameterization, and the Unity client only
    supports standard "one texture, one UV per vertex" sampling (confirmed
    this session -- true per-triangle GPU sampling would need new custom
    shader work). So this module runs the paper's assignment/continuity
    algorithm entirely server-side and rasterizes the result into an
    ordinary top-down raster, dropped into the same SplatLayer.tile slot
    that today holds a FLUX tile. Zero client changes.

  - The paper never mixes independently-sampled "real" content with
    boundary-matched "library" content -- it doesn't need to, since nothing
    in its target scenes is directly photographed. Real, continuously-varying
    photo content sampled per-pixel in world space (what
    TerrainTextureGenerationStage's panorama layer already does) is
    naturally seamless on its own -- confirmed in the prototype, no
    boundary-matching needed there. The seam where a *library* region meets
    that real region is real, though: a per-triangle color correction toward
    the local real value produced a patchy "confetti" artifact (each
    boundary triangle nudged toward a different local real color creates a
    jagged ring). A simple image-space distance-transform feather across the
    boundary -- the same gaussian_filter(mask, sigma=...) restore-weight
    idiom already used throughout this codebase for observed/synthetic
    transitions (TerrainReconstructionStage, TerrainNoiseRefinementStage,
    TerrainMeshGenerator's vertex-noise gate) -- works cleanly.

  - Library samples use only the paper's minimal degrees of freedom: one
    symmetric edge condition (a shared, region-type-specific border color
    that every sample's edges and corners fade to). This makes assignment
    trivial (Section 2.3's node-value/edge-condition steps collapse to a
    no-op) since ANY sample in ANY rotation matches ANY neighbor by
    construction -- confirmed in the prototype. The border blend needs to be
    WIDE and low-contrast (not a thin flat ring), or adjacent samples read as
    a visibly "faceted" tiling rather than one continuous material -- a
    concrete tuning finding from the prototype, not a guess.

  - Real material, not procedural noise: library samples are built from real
    reference crops (via _extract_reference_patches, same as the FLUX path
    already uses), delit and edge/corner-flattened -- not Perlin/Worley
    synthesis. If no reference crop exists for a region type, the caller
    (TerrainTextureGenerationStage) falls back to today's FLUX generation as
    an explicit last resort.
"""
from __future__ import annotations

import numpy as np
import PIL.Image
from typing import Optional
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from util.panorama_utils import Panorama
from pipeline.panorama_segmentation.panorama_region_result import RegionType


# ── Library construction ────────────────────────────────────────────────────

def region_border_color(reference_patches: list[PIL.Image.Image]) -> np.ndarray:
    """
    Shared corner/edge anchor colour for every library sample of one region
    type -- the mean colour across all its real reference crops, so the
    anchor is representative of this specific material rather than an
    arbitrary constant.
    """
    means = []
    for patch in reference_patches:
        arr = np.asarray(patch.convert("RGB"), dtype=np.float32) / 255.0
        means.append(arr.reshape(-1, 3).mean(axis=0))
    return np.mean(means, axis=0) if means else np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _triangle_geometry(sample_res: int):
    """Shared equilateral-triangle geometry (corners, masks) at sample_res."""
    ys, xs = np.mgrid[0:sample_res, 0:sample_res].astype(np.float64)
    u = xs / sample_res
    v = 1.0 - ys / sample_res
    p0, p1, p2 = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2])

    def edge_dist(p, a, b):
        ab = b - a
        t = ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / (ab[0] ** 2 + ab[1] ** 2)
        t = np.clip(t, 0.0, 1.0)
        cx, cy = a[0] + t * ab[0], a[1] + t * ab[1]
        return np.hypot(p[0] - cx, p[1] - cy)

    dist_to_edge = np.minimum(
        np.minimum(edge_dist((u, v), p0, p1), edge_dist((u, v), p1, p2)),
        edge_dist((u, v), p2, p0),
    )

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1s, d2s, d3s = sign((u, v), p0, p1), sign((u, v), p1, p2), sign((u, v), p2, p0)
    inside = ~(((d1s < 0) | (d2s < 0) | (d3s < 0)) & ((d1s > 0) | (d2s > 0) | (d3s > 0)))

    return dist_to_edge, inside


def make_boundary_matched_sample(
    source: PIL.Image.Image,
    border_color: np.ndarray,
    sample_res: int = 128,
    border_width_frac: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build one triangular library sample from a real reference crop: fades to
    `border_color` (flat, zero-gradient) across a wide band approaching every
    edge and corner, real detail preserved in the interior. Any two samples
    built this way -- in any relative rotation -- match at a shared edge by
    construction, since the border is a shared constant.

    Returns (rgb float32 [0,1] (sample_res, sample_res, 3), inside_mask,
    border_gate). border_gate is 0 at the edges (pure border_color) rising to
    1 in the interior (pure source content) -- exposed so a caller could
    later do a more targeted correction; TerrainTextureGenerationStage
    currently uses the simpler image-space feather instead (see module
    docstring).
    """
    dist_to_edge, inside = _triangle_geometry(sample_res)
    inradius = (np.sqrt(3) / 2) / 3.0

    src = np.asarray(
        source.convert("RGB").resize((sample_res, sample_res), PIL.Image.LANCZOS),
        dtype=np.float32,
    ) / 255.0

    border_gate = np.clip(dist_to_edge / (inradius * border_width_frac), 0.0, 1.0)
    border_gate = border_gate * border_gate * (3 - 2 * border_gate)  # smoothstep

    sample = src * border_gate[..., None] + border_color[None, None, :] * (1 - border_gate[..., None])
    return sample.astype(np.float32), inside, border_gate.astype(np.float32)


def build_library(
    reference_patches: list[PIL.Image.Image],
    sample_res: int = 128,
    border_width_frac: float = 0.45,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One boundary-matched sample per reference crop, sharing one region type's border colour."""
    if not reference_patches:
        return []
    border_color = region_border_color(reference_patches)
    return [
        make_boundary_matched_sample(p, border_color, sample_res, border_width_frac)
        for p in reference_patches
    ]


# ── Real-content baking ──────────────────────────────────────────────────────

def bake_real_layer(
    panorama: Panorama,
    height_map: np.ndarray,
    terrain_half: float,
    x_half: float,
    z_far: float,
    canvas_size: int,
    x_center: float = 0.0,
    z_center: float = 0.0,
    sky_mask: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Dewarped real-photo colour, sampled directly in world space (no
    intermediate grid resampling -- see Panorama.sample_3d) across a
    [-x_half, x_half] x [-z_far, z_far] canvas around (x_center, z_center).

    sky_mask (optional PANORAMA_SKY_MASK) is forwarded to sample_3d so real
    terrain above the horizon (a mountain slope, a nearby rock formation taller
    than the camera) is sampled directly instead of being treated as sky.

    exclude_mask (optional, panorama pixel space) is forwarded alongside it as a
    second class of hole -- see vegetation_bleed_mask, which is what builds it
    for a rock formation.

    This is the same "real content everywhere" bake synthesize_region_layer
    uses as its base layer and feather target; factored out here so it can
    also be used directly for a formation mesh's own texture
    (TerrainTextureGenerationStage) -- a formation is, by construction, made
    entirely of directly-observed cells (component membership in
    HeightMapGenerator._label_ground_components requires real height data),
    so it essentially never needs the library/assignment machinery below;
    it just needs this same dewarped real bake, applied directly.

    terrain_half is the *shared* HEIGHT_MAP's own half-extent (always
    centered at the world origin, regardless of what this call's own canvas
    covers) -- needed separately so a small local canvas can still correctly
    look up real height from the one shared grid.

    Returns (canvas_size, canvas_size, 3) float32 in [0, 1].
    """
    ys, xs = np.mgrid[0:canvas_size, 0:canvas_size].astype(np.float64)
    Z = ys / canvas_size * (2.0 * z_far) - z_far + z_center
    X = xs / canvas_size * (2.0 * x_half) - x_half + x_center
    hm_h, hm_w = height_map.shape
    row_c = ((Z + terrain_half) / (2.0 * terrain_half) * (hm_h - 1)).clip(0, hm_h - 1)
    col_c = ((X + terrain_half) / (2.0 * terrain_half) * (hm_w - 1)).clip(0, hm_w - 1)
    Y = map_coordinates(height_map, [row_c, col_c], order=1, mode="nearest").astype(np.float32)
    Y = np.nan_to_num(Y, nan=0.0)
    world_pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
    return (
        panorama.sample_3d(world_pts, sky_mask=sky_mask, exclude_mask=exclude_mask)[:, :3]
        .astype(np.float32) / 255.0
    ).reshape(canvas_size, canvas_size, 3)


def vegetation_bleed_mask(
    panorama: Panorama,
    height_map: np.ndarray,
    region_type_map: np.ndarray,
    terrain_half: float,
    x_half: float,
    z_far: float,
    canvas_size: int,
    x_center: float,
    z_center: float,
    max_vegetation_fraction: float = 0.4,
) -> Optional[np.ndarray]:
    """Panorama-space mask of foliage that would bleed onto a rock formation.

    The elevation angle bake_real_layer derives for each canvas pixel is
    atan2(Y, horizontal distance) against the FINAL height map, so its
    sensitivity to a height error grows with steepness. On a formation -- which
    is where the steep geometry lives -- a modest error slides the sampled
    panorama row across the tree line and paints forest onto the peak. That is
    the green banding visible in the baked terrain_formation_*_texture.png
    images, and it does not need the height error to be large.

    Rather than trying to correct the angle (the measured-UV route was tried and
    is worse here: a distant formation has only 20-33% of its cells directly
    sampled, so preferring cached UV where it exists produces hard radial seams
    between the two sources), this rejects the *outcome*. A sample landing on a
    VEGETATION-typed panorama pixel is marked as a hole, and sample_3d's existing
    nearest-valid-neighbour fill substitutes real rock colour from nearby -- the
    same treatment sky already gets.

    Returns None, meaning "don't exclude anything", when vegetation accounts for
    more than max_vegetation_fraction of what this formation samples. Past that
    point the foliage is not bleed, it is what the formation is actually covered
    in (a wooded island, a scrubby hillock), and blanking it would erase the
    formation's real appearance rather than repair it.
    """
    ys, xs = np.mgrid[0:canvas_size, 0:canvas_size].astype(np.float64)
    Z = ys / canvas_size * (2.0 * z_far) - z_far + z_center
    X = xs / canvas_size * (2.0 * x_half) - x_half + x_center
    hm_h, hm_w = height_map.shape
    row_c = ((Z + terrain_half) / (2.0 * terrain_half) * (hm_h - 1)).clip(0, hm_h - 1)
    col_c = ((X + terrain_half) / (2.0 * terrain_half) * (hm_w - 1)).clip(0, hm_w - 1)
    Y = np.nan_to_num(
        map_coordinates(height_map, [row_c, col_c], order=1, mode="nearest").astype(np.float32),
        nan=0.0,
    )
    world_pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    pu, pv, _ = panorama._project_vertices(world_pts)
    pano_h, pano_w = region_type_map.shape
    # _project_vertices works in this panorama's own pixel space; the region map
    # may have been produced at a different resolution.
    pu_i = np.clip(np.round(pu / max(1, panorama.width - 1) * (pano_w - 1)).astype(np.int64), 0, pano_w - 1)
    pv_i = np.clip(np.round(pv / max(1, panorama.height - 1) * (pano_h - 1)).astype(np.int64), 0, pano_h - 1)

    sampled_types = region_type_map[pv_i, pu_i]
    foliage = [int(RegionType.VEGETATION), int(RegionType.FOREST)]
    vegetation_fraction = float(np.isin(sampled_types, foliage).mean())
    if vegetation_fraction > max_vegetation_fraction:
        return None

    return np.isin(region_type_map, foliage)


# ── Real-content baking (mesh-projected) ──────────────────────────────────────

def bake_real_layer_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    panorama_uv: np.ndarray,
    panorama: Panorama,
    canvas_size: int,
    x_half: float,
    z_far: float,
    x_center: float = 0.0,
    z_center: float = 0.0,
) -> np.ndarray:
    """
    Real-photo colour for the main terrain mesh's pattern-texture base layer,
    projected from the mesh's own vertices instead of resampled from a
    synthetic top-down world grid.

    bake_real_layer (above) determines each canvas pixel's world Y by
    resampling the *final* HEIGHT_MAP -- after Terrain Reconstruction/Noise
    Refinement's own solving/diffusion/certainty-weighted blending -- at a
    position that has no relationship to any real measurement. Right where
    this function's caller (synthesize_region_layer) actually needs real
    content most (the near-nadir/horizon band the panorama layer's own
    weight rejects -- see TerrainTextureGenerationStage._panorama_visibility_
    weight), a small height error there amplifies into a large panorama-row
    error (Y = depth * sin(phi), phi steep near nadir) -- the exact same
    failure mode that made TerrainMeshGenerator.generate() prefer each
    observed vertex's own *cached* true panorama UV (HeightMapGenerator.
    _panorama_uv_from_height, captured before any of that downstream
    processing) over re-deriving UV from final geometry, for the main
    panorama layer. This function reuses that exact same already-correct,
    already-cached per-vertex UV (panorama_uv -- pass the terrain mesh's own
    Mesh.extra_uv, computed once by TerrainMeshGenerator.generate() and
    validated by the live panorama layer) instead of computing a second,
    independent, and *less* trustworthy position/height estimate of its own.

    Concretely: sample this panorama once per real mesh vertex (via
    Panorama.sample_uv, which expects mesh_uvs' own UV convention -- pass
    panorama_uv through unmodified), then rasterise each triangle into the
    canvas by barycentric-interpolating its own three vertices' colours
    (same triangle rasteriser _rasterize_triangle uses for library samples
    below). Colour interpolation has no seam-wrap ambiguity the way UV
    interpolation does (see TerrainMeshGenerator._fix_uv_seam) -- a colour is
    already a resolved value, not an angle that can wrap the wrong way -- so
    no seam handling is needed here despite reusing the same per-vertex data
    a seam-fixed UV channel would.

    vertices, faces: the terrain mesh's own (already seam-fixed) arrays --
    same ones TerrainMeshGenerator.generate() built panorama_uv against, so
    indices line up 1:1. Only vertices[:, 0]/[:, 2] (X, Z) are used for
    canvas placement; Y is not needed at all here, since colour already came
    from panorama_uv rather than being re-derived from position.

    Returns (canvas_size, canvas_size, 3) float32 in [0, 1].
    """
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)

    vertex_colors = (
        panorama.sample_uv(panorama_uv[:, 0], panorama_uv[:, 1])[:, :3].astype(np.float32) / 255.0
    )

    tri_xz = vertices[faces][:, :, [0, 2]]       # (M, 3, 2) world (X, Z)
    tri_colors = vertex_colors[faces]            # (M, 3, 3)

    for i in range(len(faces)):
        def colors_fn(w0, w1, w2, _tri_xz, c=tri_colors[i]):
            return (
                w0[..., None] * c[0] + w1[..., None] * c[1] + w2[..., None] * c[2]
            )
        _rasterize_triangle(canvas, None, tri_xz[i], colors_fn, x_half, z_far, x_center, z_center)

    return canvas


# ── Rasterization ────────────────────────────────────────────────────────────
#
# x_center/z_center let the canvas domain be [-x_half, x_half] x [-z_far, z_far]
# around an arbitrary world point rather than always the world origin -- the
# main terrain is centered at the origin (defaults below preserve that), but a
# formation mesh (see TerrainMeshGenerator.generate_component_mesh) lives
# wherever its own footprint actually is.

def _world_to_px(
    xy: np.ndarray, x_half: float, z_far: float, canvas_size: int,
    x_center: float = 0.0, z_center: float = 0.0,
) -> np.ndarray:
    u = (xy[..., 0] - x_center + x_half) / (2.0 * x_half) * canvas_size
    v = (xy[..., 1] - z_center + z_far) / (2.0 * z_far) * canvas_size
    return np.stack([u, v], axis=-1)


def _rasterize_triangle(
    canvas: np.ndarray,
    mask_canvas: Optional[np.ndarray],
    tri_xz: np.ndarray,       # (3, 2) world (X, Z)
    colors_fn,                # (w0, w1, w2, tri_xz) -> (H_bbox, W_bbox, 3) colours in [0,1]
    x_half: float,
    z_far: float,
    x_center: float = 0.0,
    z_center: float = 0.0,
) -> None:
    canvas_size = canvas.shape[0]
    px = _world_to_px(tri_xz, x_half, z_far, canvas_size, x_center, z_center)
    x0, y0 = np.floor(px.min(axis=0)).astype(int)
    x1, y1 = np.ceil(px.max(axis=0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, canvas_size - 1), min(y1, canvas_size - 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    p = np.stack([xx, yy], axis=-1).astype(np.float64)

    def sign(p1, p2, p3):
        return (p1[..., 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[..., 1] - p3[1])
    a, b, c = px[0], px[1], px[2]
    d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
    inside = ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))
    if not inside.any():
        return

    area = sign(a[None, None], b, c)[0, 0]
    if abs(area) < 1e-9:
        return
    w0 = sign(p, b, c) / area
    w1 = sign(p, c, a) / area
    w2 = 1.0 - w0 - w1

    colors = colors_fn(w0, w1, w2, tri_xz)
    canvas[y0:y1 + 1, x0:x1 + 1][inside] = colors[inside]
    if mask_canvas is not None:
        mask_canvas[y0:y1 + 1, x0:x1 + 1] |= inside


def synthesize_region_layer(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_needs_synthesis: np.ndarray,
    face_region_mask: np.ndarray,
    reference_patches: list[PIL.Image.Image],
    real_everywhere: np.ndarray,
    x_half: float,
    z_far: float,
    canvas_size: int,
    seed: int,
    x_center: float = 0.0,
    z_center: float = 0.0,
    sample_res: int = 128,
    border_width_frac: float = 0.45,
    feather_band_px: float = 48.0,
    feather_sigma_px: float = 8.0,
    min_coverage_fraction: float = 0.002,
) -> Optional[PIL.Image.Image]:
    """
    Build this region type's texture layer by assigning boundary-matched real
    samples (see build_library) to every triangle that both needs synthesis
    (TerrainTextureGenerationStage's panorama layer rejected it) and belongs
    to this region type, then feathering the seam against real photo content
    everywhere else on the canvas.

    The canvas domain is [-x_half, x_half] x [-z_far, z_far] around
    (x_center, z_center) -- 0, 0 for the main terrain (its own domain is
    centered at the world origin), but a formation mesh (see
    TerrainMeshGenerator.generate_component_mesh) lives wherever its own
    footprint actually is, hence the offset.

    real_everywhere: precomputed by the caller once (see
    bake_real_layer_from_mesh), shared across every region type's own call to
    this function -- it doesn't depend on region_val, so building it fresh
    per call (as an earlier version of this function did, via its own
    panorama/height_map/terrain_half parameters) was pure repeated work,
    on top of computing the same mesh-projection every region type needlessly
    re-triggered.

    Returns None if there are no reference patches (caller should fall back
    to FLUX generation) or the qualifying footprint is smaller than
    min_coverage_fraction of the canvas (not worth a dedicated bake -- the
    panorama layer / other region layers already cover this negligible area
    well enough via the existing blend-weight normalisation).
    """
    library = build_library(reference_patches, sample_res, border_width_frac)
    if not library:
        return None

    qualifying = face_needs_synthesis & face_region_mask
    if not qualifying.any():
        return None

    rng = np.random.default_rng(seed)

    canvas = real_everywhere.copy()
    synth_mask = np.zeros((canvas_size, canvas_size), dtype=bool)

    qualifying_idx = np.flatnonzero(qualifying)
    library_choice = rng.integers(0, len(library), size=len(qualifying_idx))
    rotation_choice = rng.integers(0, 3, size=len(qualifying_idx))
    rotation_orders = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    sample_corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])

    for i, face_idx in enumerate(qualifying_idx):
        tri_verts = vertices[faces[face_idx]]
        tri_xz = tri_verts[:, [0, 2]]
        sample, _, _ = library[library_choice[i]]
        order = rotation_orders[rotation_choice[i]]

        def colors_fn(w0, w1, w2, _tri_xz, sample=sample, order=order):
            su = w0 * sample_corners[order[0], 0] + w1 * sample_corners[order[1], 0] + w2 * sample_corners[order[2], 0]
            sv = w0 * sample_corners[order[0], 1] + w1 * sample_corners[order[1], 1] + w2 * sample_corners[order[2], 1]
            sx = np.clip((su * sample_res).astype(int), 0, sample_res - 1)
            sy = np.clip(((1.0 - sv) * sample_res).astype(int), 0, sample_res - 1)
            return sample[sy, sx]

        _rasterize_triangle(canvas, synth_mask, tri_xz, colors_fn, x_half, z_far, x_center, z_center)

    coverage = float(synth_mask.mean())
    if coverage < min_coverage_fraction:
        return None

    # Image-space boundary feather -- see module docstring for why this
    # (rather than a per-triangle correction) is what's actually used.
    dist_in = distance_transform_edt(synth_mask)
    band = np.clip(1.0 - dist_in / feather_band_px, 0.0, 1.0)
    band = band * band * (3 - 2 * band)
    band = gaussian_filter(band, sigma=feather_sigma_px) * synth_mask

    canvas = canvas * (1 - band[..., None]) + real_everywhere * band[..., None]

    out = (canvas.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return PIL.Image.fromarray(out, "RGB")


def make_tileable(image: PIL.Image.Image, feather_frac: float = 0.3) -> PIL.Image.Image:
    """Make a square photo patch wrap at its borders.

    Classic offset-and-feather: the patch rolled by half its size has the
    original's borders meeting at a cross through the centre and the original's
    seamless centre at its own borders. Blending the ORIGINAL in along that
    central cross (where it is continuous) and the ROLLED copy toward the
    borders (where it is continuous) yields a tile whose edges match. The blend
    bands are soft, which is why the client samples these tiles stochastically
    (SplatLayer.stochastic): a hex grid of random offsets hides both the bands
    and the repeat.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    rolled = np.roll(np.roll(arr, h // 2, axis=0), w // 2, axis=1)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.minimum(np.abs(xs - w / 2.0), np.abs(ys - h / 2.0))
    band = max(1.0, feather_frac * min(h, w) / 2.0)
    t = np.clip(dist / band, 0.0, 1.0)
    keep_original = 1.0 - t * t * (3.0 - 2.0 * t)   # 1 on the cross, 0 past the band
    out = rolled * (1.0 - keep_original[..., None]) + arr * keep_original[..., None]
    return PIL.Image.fromarray(out.clip(0, 255).astype(np.uint8), "RGB")


def _push_pull(colour: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Fill the zero-weight texels of `colour` (H, W, 3) from the weighted ones:
    a pyramid of normalised 2x2 averages pulled down and pushed back up, so a
    hole takes the average of the nearest valid colour at whatever scale first
    reaches it. Valid texels (weight 1) are returned unchanged."""
    from scipy.ndimage import zoom as _zoom
    h, w = weight.shape
    if h <= 2 or w <= 2:
        total = float(weight.sum())
        mean = (colour * weight[..., None]).sum(axis=(0, 1)) / max(total, 1e-6) if total > 0 else colour.mean(axis=(0, 1))
        return np.broadcast_to(mean, colour.shape).astype(np.float32)
    ph, pw = h + (h % 2), w + (w % 2)
    c = np.zeros((ph, pw, 3), dtype=np.float32); c[:h, :w] = colour * weight[..., None]
    m = np.zeros((ph, pw), dtype=np.float32); m[:h, :w] = weight
    c_small = c.reshape(ph // 2, 2, pw // 2, 2, 3).sum(axis=(1, 3))
    m_small = m.reshape(ph // 2, 2, pw // 2, 2).sum(axis=(1, 3))
    c_small = c_small / np.maximum(m_small, 1e-6)[..., None]
    filled_small = _push_pull(c_small, np.clip(m_small, 0.0, 1.0))
    up = _zoom(filled_small, (h / filled_small.shape[0], w / filled_small.shape[1], 1), order=1)
    wgt = np.clip(weight, 0.0, 1.0)[..., None]
    return (colour * wgt + up * (1.0 - wgt)).astype(np.float32)


def bake_projected_colour(
    panorama: Panorama,
    height_map: np.ndarray,
    half: float,
    canvas_size: int,
    type_map: np.ndarray,
    depth: "np.ndarray | None",
    region_map: "np.ndarray | None",
    max_texel_m: float = 0.6,
    depth_tolerance: float = 0.35,
    near_m: float = 3.0,
    blur_texels: float = 1.0,
    near_vegetation_m: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """The photograph projected onto the terrain where that projection is sound,
    and filled from itself everywhere else.

    A top-down bake of the panorama is a starburst wherever it projects a pixel
    onto ground that pixel does not show: a conifer or the mountain lands on the
    flat far field as a spoke metres long. The fix is not blur. Each texel of the
    grid is a world point; the panorama pixel it projects to is only evidence
    about that texel when (a) the pixel shows a surface (not sky, canopy or a
    building), (b) the panorama's own depth there agrees with the texel's
    distance (a mountain pixel at 110 m does not describe a texel at 150 m on
    the plain, nor the horizon band a texel anywhere), and (c) the pixel covers
    at most max_texel_m of ground along the view ray (distance times the pixel's
    angle over the grazing cosine). Texels that fail are holes, filled per region
    type from the valid texels of the same type (push-pull), so the far meadow
    is meadow-coloured from the meadow that was seen, not from the trees behind.

    Returns (canvas (S, S, 3) float32 in [0, 1], valid (S, S) bool), row 0 at Z = -half.
    """
    from scipy.ndimage import map_coordinates as _map, zoom as _zoom
    S = canvas_size
    ys, xs = np.mgrid[0:S, 0:S].astype(np.float64)
    Z = ys / S * (2.0 * half) - half
    X = xs / S * (2.0 * half) - half
    hm_h, hm_w = height_map.shape
    row_c = ((Z + half) / (2.0 * half) * (hm_h - 1)).clip(0, hm_h - 1)
    col_c = ((X + half) / (2.0 * half) * (hm_w - 1)).clip(0, hm_w - 1)
    Y = np.nan_to_num(_map(height_map, [row_c, col_c], order=1, mode="nearest").astype(np.float64), nan=0.0)
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    d = np.linalg.norm(pts, axis=1)
    pu, pv, _lat = panorama._project_vertices(pts.astype(np.float32))
    pu = np.asarray(pu, dtype=np.float64).reshape(S, S); pv = np.asarray(pv, dtype=np.float64).reshape(S, S)
    d = d.reshape(S, S)
    W, H = panorama.width, panorama.height

    # (a) what the pixel shows
    th, tw = type_map.shape
    tu = np.clip(np.rint(pu / max(W - 1, 1) * (tw - 1)).astype(np.int64), 0, tw - 1)
    tv = np.clip(np.rint(pv / max(H - 1, 1) * (th - 1)).astype(np.int64), 0, th - 1)
    shown = type_map[tv, tu].astype(np.int32)
    surface = np.isin(shown, [int(t) for t in (
        RegionType.TERRAIN, RegionType.GROUND, RegionType.VEGETATION, RegionType.SNOW,
        RegionType.ROAD, RegionType.TRAIL, RegionType.WATER,
    )])
    valid = surface
    # A meadow is upright too: a 0.5 m plant at 20 m from a 1 m camera smears
    # plant_height * d / camera_height = 10 m along its ray, so beyond
    # near_vegetation_m the meadow's pixels are not a sound projection either
    # and the meadow colour comes from the near meadow via the fill.
    valid &= ~((shown == int(RegionType.VEGETATION)) & (d > near_vegetation_m))
    # (b) the panorama's depth agrees with the texel's distance
    if depth is not None:
        dh, dw = depth.shape
        du = np.clip(np.rint(pu / max(W - 1, 1) * (dw - 1)).astype(np.int64), 0, dw - 1)
        dv = np.clip(np.rint(pv / max(H - 1, 1) * (dh - 1)).astype(np.int64), 0, dh - 1)
        dp = depth[dv, du].astype(np.float64)
        agree = np.isfinite(dp) & (np.abs(dp - d) <= depth_tolerance * np.maximum(d, near_m))
        valid &= agree
    # (c) the pixel's ground footprint
    texel = 2.0 * half / S
    gy, gx = np.gradient(Y, texel)                     # dY/dZ (rows), dY/dX (cols)
    normal = np.stack([-gx, np.ones_like(Y), -gy], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    ray = -pts.reshape(S, S, 3) / np.maximum(d, 1e-6)[..., None]
    cos_g = np.abs((normal * ray).sum(axis=-1))
    footprint = d * (np.pi / H) / np.maximum(cos_g, 0.05)
    valid &= footprint <= max_texel_m

    rgb = np.asarray(panorama.rgb(), dtype=np.float32) / 255.0
    colour = np.stack([_map(rgb[..., c], [pv, pu], order=1, mode="nearest") for c in range(3)], axis=-1).astype(np.float32)

    # A valid pixel still describes its texel only at the scale of its footprint:
    # an upright flower at 10 m is smeared along its ray over the metre its pixel
    # covers. Blur each texel by its own footprint (normalised convolution over
    # the valid texels, so holes do not darken the edge), coarse where the
    # projection is coarse and crisp where it is not.
    sigmas = np.clip(footprint / texel / 2.0, 0.5, 24.0)
    levels = [0.5, 2.0, 6.0, 16.0]
    w = valid.astype(np.float32)
    blurred = []
    for sg in levels:
        num = np.stack([gaussian_filter(colour[..., c] * w, sigma=sg) for c in range(3)], axis=-1)
        den = gaussian_filter(w, sigma=sg)
        blurred.append(num / np.maximum(den, 1e-4)[..., None])
    idx = np.clip(np.searchsorted(np.asarray(levels), sigmas), 0, len(levels) - 1)
    colour = np.choose(idx[..., None], [b[..., :] for b in blurred]).astype(np.float32)

    # Fill per region type of the TEXEL (where the viewer stands), from the
    # valid texels of that type.
    if region_map is not None:
        rh, rw = region_map.shape
        region = np.rint(_zoom(region_map.astype(np.float32), (S / rh, S / rw), order=0)).astype(np.int32)
    else:
        region = np.zeros((S, S), dtype=np.int32)
    weight = valid.astype(np.float32)
    filled = _push_pull(colour, weight) if valid.any() else colour
    out = filled.copy()
    for t in np.unique(region):
        here = region == t
        w_t = weight * here
        if w_t.sum() < 16:
            continue                                   # too little of this type was seen: global fill
        out[here] = _push_pull(colour, w_t)[here]
    if blur_texels > 0:
        out = np.stack([gaussian_filter(out[..., c], sigma=blur_texels) for c in range(3)], axis=-1)
    return out.clip(0, 1).astype(np.float32), valid


def bake_macro_colour(
    panorama_rgb: np.ndarray,
    type_map: np.ndarray,
    depth: "np.ndarray | None",
    height_map: np.ndarray,
    half: float,
    canvas_size: int,
    angular_sigma_deg: float = 2.0,
    world_blur_texels: float = 1.5,
    near_vegetation_m: float = 8.0,
    work_width: int = 1024,
    region_map: "np.ndarray | None" = None,
    max_texel_m: float = 0.6,
    depth_tolerance: float = 0.35,
) -> np.ndarray:
    """The photograph's large-scale ground colour over the terrain grid: see
    bake_projected_colour, which replaces the angular-blur bake (that still left
    radial spokes wherever the far field was flat, because the smear is the
    projection's, not the image's)."""
    canvas, _valid = bake_projected_colour(
        Panorama(np.asarray(panorama_rgb, dtype=np.uint8)), height_map, half, canvas_size,
        type_map, depth, region_map, max_texel_m=max_texel_m, depth_tolerance=depth_tolerance,
        blur_texels=world_blur_texels,
    )
    return canvas
