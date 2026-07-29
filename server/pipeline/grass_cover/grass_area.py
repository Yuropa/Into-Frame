"""
Where grass belongs, in top-down terrain-grid space.

The short version of why this module exists at all: RegionMapStage reads
PANORAMA_REGION_TYPE_MAP_TERRAIN (see its own `keys:` block in config.yaml), the
region typing of the *object-removed, LoRA-corrected* panorama. That's the right
source for the things RegionMapStage feeds -- ridge/crest extraction, height
solving, terrain texture reference crops all want a panorama with foreground
occluders taken out.

It is the wrong source for "is there grass here", because
PanoramaForegroundInpaintingStage removes everything closer than
foreground_distance_m, and in a ground-level capture the ground itself is closer
than that across the entire lower hemisphere. Measured on an alpine-meadow
capture (Mount Rainier, 4096x2048 panorama): the removal mask covered 48.5% of
the whole panorama and 100% of every row below ~60% height, and the meadow it
erased came back as bare gravel. Sampled through this module's own projection
over the 2091 m2 of grid cells HEIGHT_MAP_OBSERVED_MASK marks as genuinely
measured:

                original panorama      inpainted panorama
    vegetation     1000 m2 (47.8%)          47 m2 ( 2.2%)
    ground          772 m2 (36.9%)         944 m2 (45.2%)
    terrain         206 m2 ( 9.9%)        1040 m2 (49.7%)

i.e. ~95% of the vegetation evidence is destroyed, the meadow is relabelled as
rock/mountain, and the region typing of the inpainted panorama calls that same
area `ground/sand`.

So grass area is derived here from PANORAMA_REGION_TYPE_MAP -- the typing of the
ORIGINAL panorama, which is already computed by the first PanoramaRegionStage
pass and otherwise unused for this. This matches the split the object/region
analysis path already follows elsewhere: geometry from the inpainted panorama,
semantics from the original.

No reprojection is needed. HeightMapGenerator already cached a per-cell panorama
UV (HEIGHT_MAP_PANO_U / HEIGHT_MAP_PANO_V) -- each terrain grid cell's own true
observed panorama pixel -- so this is a direct lookup, and it inherits that
channel's already-validated correctness rather than re-deriving a second
position/height-based projection with its own error (the failure mode
pattern_texture.bake_real_layer_from_mesh's docstring covers at length).
"""

import numpy as np
from scipy import ndimage

from pipeline.panorama_segmentation.panorama_region_result import RegionType


# Coarse region types on the ORIGINAL panorama that count as grass-bearing
# ground. VEGETATION is the meadow itself (ADE20K labels the whole wildflower
# field "plant"; on the Rainier capture that single region is 35.8% of the
# panorama). GROUND covers the "grass"/"field"/"earth" labels that _LABEL_RULES
# collapses together -- bare patches between tufts read as earth, and excluding
# them would punch holes through the middle of a continuous meadow.
#
# TERRAIN is deliberately excluded: that's mountain/cliff/rock, which is where
# the treeline stops and where scattering grass would put tufts up a snowfield.
_GRASS_SOURCE_TYPES: frozenset[RegionType] = frozenset({
    RegionType.VEGETATION, RegionType.GROUND,
})


def grass_area_mask(
    pano_u: np.ndarray,
    pano_v: np.ndarray,
    region_type_map: np.ndarray,
    observed_mask: np.ndarray,
    *,
    max_radius_m: float,
    grid_size_meters: float,
    nadir_fill_radius_m: float = 6.0,
    nadir_fill_min_fraction: float = 0.5,
    close_radius_cells: int = 9,
    min_component_cells: int = 64,
) -> np.ndarray:
    """Boolean (grid_resolution, grid_resolution) mask of cells that should carry grass.

    pano_u/pano_v      -- HEIGHT_MAP_PANO_U/V: per-cell panorama UV in [0, 1].
                          v is PRE-FLIPPED (v=1 is image row 0) -- the convention
                          HeightMapGenerator._panorama_uv_from_height writes and
                          Panorama.sample_uv un-flips. Sampling the region map with
                          `row = v * H` instead of `(1 - v) * H` reads the mirror
                          latitude: for ground a few metres from the camera that
                          lands ~20 degrees ABOVE the horizon, i.e. in sky and
                          mountain, and the whole near field comes back "not grass".
    region_type_map    -- PANORAMA_REGION_TYPE_MAP (the ORIGINAL panorama's, not
                          the _terrain one), (H, W) of RegionType indices.
    observed_mask      -- HEIGHT_MAP_OBSERVED_MASK. The cached UV is only trusted
                          where this is set (see _panorama_uv_from_height's call
                          site); elsewhere it was re-derived from a solved/diffused
                          height and can point anywhere. Cells outside it are
                          resolved by the nadir fill below or dropped.
    max_radius_m       -- hard cap on distance from the camera. Grass is scattered
                          as individual instances with no client-side instancing
                          (see GrassCoverStage.max_radius_m), so the population has
                          to be bounded somewhere; beyond this the terrain texture
                          carries the ground on its own.

    Everything inside nadir_fill_radius_m is unobserved by construction --
    HeightMapStage's own nadir exclusion drops the near-vertical band where the
    equirectangular depth model is unreliable, so on the Rainier capture no cell
    within ~4 m of the camera has a measurement at all. That disc is precisely
    the ground the user is standing on, so it can't just be left bare; it's
    filled from the ring immediately outside it, the same way RegionMapStage
    fills its own nadir exclusion inward rather than flood-filling from afar.
    """
    resolution = pano_u.shape[0]
    observed = observed_mask.astype(bool) & np.isfinite(pano_u) & np.isfinite(pano_v)

    pano_h, pano_w = region_type_map.shape
    row = np.zeros(pano_u.shape, dtype=np.intp)
    col = np.zeros(pano_u.shape, dtype=np.intp)
    # Only index with the finite entries -- casting NaN to an integer is
    # undefined and, in practice, produces a garbage index that silently reads
    # some unrelated corner of the panorama.
    row[observed] = np.clip(((1.0 - pano_v[observed]) * (pano_h - 1)).astype(np.intp), 0, pano_h - 1)
    col[observed] = np.clip((pano_u[observed] * (pano_w - 1)).astype(np.intp), 0, pano_w - 1)

    cell_type = np.full(pano_u.shape, int(RegionType.OTHER), dtype=np.int16)
    cell_type[observed] = region_type_map[row[observed], col[observed]].astype(np.int16)

    mask = observed & np.isin(cell_type, [int(rt) for rt in _GRASS_SOURCE_TYPES])

    # The per-cell UV is an equirectangular projection fanned out along rays, so
    # an unmeasured column between two measured ones shows up as a thin radial
    # gap rather than a real break in the meadow. Closing at a radius wider than
    # those gaps stitches them without merging genuinely separate patches.
    if close_radius_cells > 1:
        mask = ndimage.binary_closing(mask, structure=_disc(close_radius_cells))

    radius = _radius_grid(resolution, grid_size_meters)
    mask = _fill_nadir_disc(
        mask, radius, observed,
        fill_radius_m=nadir_fill_radius_m,
        min_fraction=nadir_fill_min_fraction,
    )

    mask &= radius <= max_radius_m

    # Drop speckle -- isolated cells from a stray misclassified panorama pixel,
    # which after the radial fan-out become a handful of lone tufts standing in
    # open gravel.
    if min_component_cells > 1:
        labels, count = ndimage.label(mask)
        if count:
            sizes = np.bincount(labels.ravel())
            mask &= ~np.isin(labels, np.flatnonzero(sizes < min_component_cells))

    return mask


def _fill_nadir_disc(
    mask: np.ndarray,
    radius: np.ndarray,
    observed: np.ndarray,
    *,
    fill_radius_m: float,
    min_fraction: float,
) -> np.ndarray:
    """Fill the unmeasured near-camera disc from the measured ring around it.

    Grass either covers the ground the user is standing on or it doesn't; there
    is no useful per-cell signal to recover inside the exclusion zone, so this is
    a single all-or-nothing decision taken from the annulus just outside it. The
    ring is one fill_radius_m-wide band rather than the whole scene: a meadow
    that stops 30 m away shouldn't vote on what's underfoot.

    Only cells that are actually unmeasured get filled, so a measured cell inside
    the radius that genuinely isn't grass (standing water, a rock slab) keeps its
    own answer.
    """
    disc = radius <= fill_radius_m
    ring = (radius > fill_radius_m) & (radius <= fill_radius_m * 2.0) & observed
    if not ring.any():
        return mask
    if mask[ring].mean() < min_fraction:
        return mask
    return mask | (disc & ~observed)


def _disc(radius: int) -> np.ndarray:
    span = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(span, span, indexing="ij")
    return (yy * yy + xx * xx) <= radius * radius


def _radius_grid(resolution: int, grid_size_meters: float) -> np.ndarray:
    """Per-cell distance (metres) from the camera, which sits at the grid origin."""
    half = grid_size_meters / 2.0
    axis = (np.arange(resolution, dtype=np.float32) + 0.5) / resolution * grid_size_meters - half
    zz, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.hypot(xx, zz)


def area_square_meters(mask: np.ndarray, grid_size_meters: float) -> float:
    """Total world-space area the mask covers, for logging/debug."""
    cell_area = (grid_size_meters / mask.shape[0]) ** 2
    return float(mask.sum()) * cell_area
