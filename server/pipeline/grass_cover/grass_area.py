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
over the 2091 m2 of grid cells HEIGHT_MAP_OBSERVED_MASK marks as directly
measured (a strict subset of the cells this module actually samples -- see
`sampled_mask` below -- chosen here only because it is the most conservative
evidence available):

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


def cell_greenness(
    panorama_rgb: np.ndarray, row: np.ndarray, col: np.ndarray, sampled: np.ndarray,
    map_shape: "tuple[int, int] | None" = None,
) -> np.ndarray:
    """Per-cell excess green ((2G-R-B)/255) of the panorama pixel each cell observes.

    The coarse region type is too blunt to decide where grass goes, in both
    directions, and the fine ADE20K label it was collapsed from is not persisted
    anywhere this stage can reach. Colour is available, is per-pixel, and settles
    the same question more directly -- a surface that is going to carry grass is
    green in the photograph.

    It fixes the false positives the type map lets through: _LABEL_RULES folds
    `grass`, `earth`, `field`, `sand`, `dirt`, `mud`, `ground`, `soil`, `floor`,
    `snow` and `ice` into one RegionType.GROUND, so a snowfield and a meadow are
    indistinguishable by type. Rainier's own meadow is ringed with melting
    snowpack that types identically to the wildflowers beside it.

    Cells outside `sampled` get -inf: they have no trustworthy UV, so there is no
    pixel to read, and they must never pass a greenness test by accident.

    row/col arrive in the REGION TYPE MAP's pixel space (that is what the caller
    indexed to get each cell's type), and this reads the PANORAMA. The two are
    normally the same size -- PanoramaRegionStage typed that same image -- but
    SupersamplingStage rewrites PANORAMA in place at 2x, so a run whose typing
    happened against the pre-supersampled panorama has them off by a factor of
    two. GrassCoverStage._grass_card_boxes already resamples for exactly this
    (see its type_map.shape check); this path had no equivalent and would either
    raise IndexError or, in the halving direction, silently read the wrong pixel
    and veto the meadow on the colour of somewhere else entirely. Rescaled rather
    than asserted so the mismatch degrades to a slightly blurrier lookup, which
    is what it actually is, instead of failing a stage over a factor of two.
    """
    out = np.full(row.shape, -np.inf, dtype=np.float32)
    if panorama_rgb is None:
        return np.zeros(row.shape, dtype=np.float32)
    rgb = np.asarray(panorama_rgb, dtype=np.float32)[..., :3]
    pano_h, pano_w = rgb.shape[:2]
    r_idx, c_idx = row[sampled], col[sampled]
    if map_shape is not None and tuple(map_shape) != (pano_h, pano_w):
        map_h, map_w = map_shape
        r_idx = np.clip(r_idx * (pano_h / map_h), 0, pano_h - 1).astype(np.intp)
        c_idx = np.clip(c_idx * (pano_w / map_w), 0, pano_w - 1).astype(np.intp)
    r = rgb[..., 0][r_idx, c_idx]
    g = rgb[..., 1][r_idx, c_idx]
    b = rgb[..., 2][r_idx, c_idx]
    out[sampled] = (2.0 * g - r - b) / 255.0
    return out


def grass_area_mask(
    pano_u: np.ndarray,
    pano_v: np.ndarray,
    region_type_map: np.ndarray,
    sampled_mask: np.ndarray,
    *,
    max_radius_m: float,
    grid_size_meters: float,
    panorama_rgb: "np.ndarray | None" = None,
    min_cell_greenness: float = 0.02,
    nadir_fill_radius_m: float = 6.0,
    nadir_fill_min_fraction: float = 0.5,
    close_radius_cells: int = 9,
    min_component_cells: int = 64,
    stats: "dict | None" = None,
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
    sampled_mask       -- HEIGHT_MAP_REAL_SAMPLE_MASK. The cached UV is only
                          trustworthy where a cell came from a genuine depth
                          sample; elsewhere it was re-derived from a solved/
                          diffused height and can point anywhere. This is the
                          right key for that question rather than the narrower
                          HEIGHT_MAP_OBSERVED_MASK, which additionally excludes
                          the nadir ramp band for height-pinning reasons that
                          don't apply to UV (see its ContextKey comment). On the
                          Rainier capture that difference is 30.8% of the grid
                          versus 5.2%, and -- decisively for grass -- 100% versus
                          0% of the 2-6 m ring the user is standing in.
    max_radius_m       -- hard cap on distance from the camera. Grass is scattered
                          as individual instances with no client-side instancing
                          (see GrassCoverStage.max_radius_m), so the population has
                          to be bounded somewhere; beyond this the terrain texture
                          carries the ground on its own.
    stats              -- optional dict, filled with a FRONT/BEHIND breakdown of the
                          funnel below (see _fill_hemisphere_stats). Nothing here is
                          azimuth-dependent -- every projection this path uses is
                          arctan2(X, Z), correct in all four quadrants, and both
                          panorama samplers wrap horizontally -- so grass thinning out
                          behind the camera can only come from the two INPUTS being
                          weaker there: the depth samples (sampled_mask) or the
                          semantic typing (region_type_map). Only ~1/6 of the panorama
                          is the real photograph; the rest is generated, so that is
                          entirely possible and this is what tells the two apart.

    HeightMapStage's nadir exclusion still leaves the innermost cells unsampled
    even under HEIGHT_MAP_REAL_SAMPLE_MASK (46% coverage inside 2 m on the
    Rainier capture), and that disc is precisely the ground the user is standing
    on, so it can't be left bare. Whatever remains unsampled there is filled from
    the ring immediately outside it, the same way RegionMapStage fills its own
    nadir exclusion inward rather than flood-filling from afar.
    """
    resolution = pano_u.shape[0]
    sampled = sampled_mask.astype(bool) & np.isfinite(pano_u) & np.isfinite(pano_v)

    pano_h, pano_w = region_type_map.shape
    row = np.zeros(pano_u.shape, dtype=np.intp)
    col = np.zeros(pano_u.shape, dtype=np.intp)
    # Only index with the finite entries -- casting NaN to an integer is
    # undefined and, in practice, produces a garbage index that silently reads
    # some unrelated corner of the panorama.
    row[sampled] = np.clip(((1.0 - pano_v[sampled]) * (pano_h - 1)).astype(np.intp), 0, pano_h - 1)
    col[sampled] = np.clip((pano_u[sampled] * (pano_w - 1)).astype(np.intp), 0, pano_w - 1)

    cell_type = np.full(pano_u.shape, int(RegionType.OTHER), dtype=np.int16)
    cell_type[sampled] = region_type_map[row[sampled], col[sampled]].astype(np.int16)

    mask = sampled & np.isin(cell_type, [int(rt) for rt in _GRASS_SOURCE_TYPES])

    # Per-cell colour veto.
    #
    # The type gate above is coarse in a way that matters here: RegionType.GROUND
    # is `grass`/`earth`/`field` collapsed together with `sand`, `snow` and `ice`
    # (see _LABEL_RULES), so it cannot tell a meadow from the snowpack ringing it
    # or from a beach. This asks the panorama what colour the cell actually is,
    # which separates them directly. Deliberately a low bar rather than a test for
    # "lush": dry and golden grass sit near zero on this scale, while sand runs
    # about -0.12 and snow lower still, so the threshold sits just above neutral
    # and only removes surfaces that are definitively not vegetation.
    #
    # Applied TWICE, before and after the morphology below -- see the second call
    # for why once is not enough.
    vetoed_cells = np.zeros(pano_u.shape, dtype=bool)
    if panorama_rgb is not None and min_cell_greenness > -np.inf:
        greenness = cell_greenness(
            panorama_rgb, row, col, sampled, map_shape=(pano_h, pano_w)
        )
        # Only a SAMPLED cell has a real measurement behind it. An unsampled cell
        # reads -inf from cell_greenness, which would make this a veto on "no data"
        # -- the opposite of what the nadir fill below is for.
        vetoed_cells = sampled & (greenness < min_cell_greenness)
        if stats is not None:
            stats["colour_vetoed_cells"] = int((mask & vetoed_cells).sum())
            # Handed back so the caller can scatter BY this rather than uniformly
            # over whatever survives the threshold -- see GrassCoverStage._scatter.
            # A veto answers "is this cell vegetation at all"; how much vegetation
            # it shows is a different question, and the same measurement answers it.
            stats["greenness"] = greenness
        mask &= ~vetoed_cells

    # The per-cell UV is an equirectangular projection fanned out along rays, so
    # an unmeasured column between two measured ones shows up as a thin radial
    # gap rather than a real break in the meadow. Closing at a radius wider than
    # those gaps stitches them without merging genuinely separate patches.
    if close_radius_cells > 1:
        mask = ndimage.binary_closing(mask, structure=_disc(close_radius_cells))

    radius = _radius_grid(resolution, grid_size_meters)
    mask = _fill_nadir_disc(
        mask, radius, sampled,
        fill_radius_m=nadir_fill_radius_m,
        min_fraction=nadir_fill_min_fraction,
    )

    mask &= radius <= max_radius_m

    # Re-apply the colour veto. Both steps above ADD cells, and neither can consult
    # the veto while doing it:
    #
    #   binary_closing is a dilation followed by an erosion, so bridging a gap is
    #   the whole point of it -- it cannot distinguish a gap that is an unmeasured
    #   column (what it exists to stitch) from one that is a snow patch the veto
    #   just cut out (what it must not). Running the veto first, as this used to,
    #   does not help: it only decides what the closing starts from, not what it
    #   fills. Measured on the Rainier capture, 34,079 cells came back this way --
    #   7.8% of the final mask, every one of them a cell whose own panorama pixel
    #   was measured and found not green.
    #
    #   _fill_nadir_disc fills unconditionally by design, but only where a cell is
    #   unsampled, so it cannot resurrect a measured cell and is left alone here.
    #
    # 202 of 8,927 placed tufts stood on vetoed cells on that run, 105 of them on
    # snow-bright pixels at a median 6.1 m -- directly in front of the viewer, which
    # is why a 7.8% mask error was so visible. Restricted to `sampled` so this is a
    # veto on measured evidence, never on its absence.
    #
    # Deliberately BEFORE the speckle drop below, not after: cutting these cells
    # fragments components that were only whole because the closing had bridged
    # them, and the speckle drop is what should decide whether the pieces are
    # still worth scattering into. It costs 2,012 genuinely green cells on the
    # Rainier capture (0.5% of the mask) that survived the veto but ended up in a
    # sub-threshold fragment -- which is the speckle filter doing its job.
    mask &= ~vetoed_cells

    # Drop speckle -- isolated cells from a stray misclassified panorama pixel,
    # which after the radial fan-out become a handful of lone tufts standing in
    # open gravel.
    if min_component_cells > 1:
        labels, count = ndimage.label(mask)
        if count:
            sizes = np.bincount(labels.ravel())
            mask &= ~np.isin(labels, np.flatnonzero(sizes < min_component_cells))

    if stats is not None:
        _fill_hemisphere_stats(
            stats, resolution, grid_size_meters, radius, max_radius_m,
            sampled, cell_type, mask,
        )

    return mask


def _fill_hemisphere_stats(
    stats: dict,
    resolution: int,
    grid_size_meters: float,
    radius: np.ndarray,
    max_radius_m: float,
    sampled: np.ndarray,
    cell_type: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Break the grass funnel down by hemisphere, in front of vs behind the camera.

    "In front" is +Z, which is panorama theta 0 -- the direction the original
    photograph was taken in, and the sixth or so of the equirect that is real rather
    than generated. Each stage of the funnel is reported for both halves so a
    shortfall can be attributed: fewer `sampled` cells behind means the panorama
    DEPTH is thin back there, fewer `grass_typed` means the SEGMENTATION doesn't call
    it vegetation, and an even split at both means the shortfall is somewhere else
    entirely.
    """
    half = grid_size_meters / 2.0
    axis = (np.arange(resolution, dtype=np.float32) + 0.5) / resolution * grid_size_meters - half
    z_grid, _ = np.meshgrid(axis, axis, indexing="ij")

    in_range = radius <= max_radius_m
    grass_types = [int(rt) for rt in _GRASS_SOURCE_TYPES]
    cell_area = (grid_size_meters / resolution) ** 2

    for name, hemisphere in (("front", z_grid >= 0), ("behind", z_grid < 0)):
        here = in_range & hemisphere
        stats[name] = {
            "in_range_m2": round(float(here.sum()) * cell_area, 1),
            "sampled_m2": round(float((here & sampled).sum()) * cell_area, 1),
            "grass_typed_m2": round(
                float((here & sampled & np.isin(cell_type, grass_types)).sum()) * cell_area, 1
            ),
            "final_m2": round(float((here & mask).sum()) * cell_area, 1),
        }


def _fill_nadir_disc(
    mask: np.ndarray,
    radius: np.ndarray,
    sampled: np.ndarray,
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

    Only cells that are actually unsampled get filled, so a sampled cell inside
    the radius that genuinely isn't grass (standing water, a rock slab) keeps its
    own answer.
    """
    disc = radius <= fill_radius_m
    ring = (radius > fill_radius_m) & (radius <= fill_radius_m * 2.0) & sampled
    if not ring.any():
        return mask
    if mask[ring].mean() < min_fraction:
        return mask
    return mask | (disc & ~sampled)


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
