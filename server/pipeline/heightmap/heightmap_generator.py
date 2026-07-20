import json
import warnings
import numpy as np
import PIL.Image
from pathlib import Path
from scipy.ndimage import binary_closing, distance_transform_edt, gaussian_filter, generate_binary_structure, label, maximum_filter, median_filter, minimum_filter, zoom
from typing import Optional
from util.depth_utils import Depth
from util.panorama_utils import Panorama
from util.projection_utils import (
    ground_projection_certainty,
    inverse_map_panorama_to_grid,
    nearest_sample_grid,
)
from util.terrain_noise_utils import diffuse_heightmap
from scene.camera import CameraIntrinsics
from pipeline.panorama_segmentation.panorama_region_result import RegionType

# Integer indices of region types that produce reliable ground-plane depth.
_VALID_REGION_INDICES = np.array(
    [rt for rt in RegionType if rt.ground_valid], dtype=np.int32
)


class HeightMapGenerator:
    @staticmethod
    def generate(
        depth: Depth,
        intrinsics: Optional[CameraIntrinsics],
        grid_size_meters: float = 100.0,
        grid_resolution: int = 4096,
        ground_y_max: float = -0.5,
        use_equirectangular: bool = False,
        smooth_sigma: float = 0.0,
        camera_height_meters: float = 1.0,
        sky_mask: Optional[np.ndarray] = None,
        flood_fill: bool = True,
        flood_fill_max_step: float = 1.5,
        panorama_depth: Optional[Depth] = None,
        region_type_mask: Optional[np.ndarray] = None,
        nadir_exclusion_radius: float = 0.0,
        nadir_ramp_width: float = 5.0,
        flat_zone_certainty: float = 0.15,
        certainty_falloff_meters: float = 20.0,
        elevation_distortion_power: float = 1.0,
        min_forward_samples: int = 4,
        fill_boundary_falloff_cells: float = 6.0,
        min_component_area_fraction: float = 0.001,
        despike_threshold_m: float = 0.3,
        despike_window: int = 5,
        despike_reference_distance_m: float = 10.0,
        region_closing_iterations: int = 2,
        single_sample_blur_sigma: float = 1.5,
        debug_dir: Optional[Path] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Project ground points from a depth map onto a top-down height grid.

        Returns (height_array, certainty_array, cell_relief_array, cell_slope_array,
        true_observed_array, component_id_array). The first four are
        (grid_resolution, grid_resolution) float32; true_observed_array is bool;
        component_id_array is int32. certainty is in [0, 1]:
        sin²(depression_angle) for cells with any direct observation (primary or
        panorama depth), 0 for pure interpolation -- but it decays with distance and
        is nonzero even on the synthetic flat-ground-prior cells, so it is a "how
        much do we trust this" signal, not a "is this a real point" signal.
        true_observed_array is that latter signal: True only for cells with a
        genuine direct measurement (primary projection, dense forward-projected
        stats, or panorama fill), independent of distance-based certainty decay and
        excluding both the flat-ground prior and interpolated fill. component_id_array
        labels which connected walkable-ground component each cell belongs to (see
        _label_ground_components): 0 = none/interpolated gap, 1 = the largest
        component (the base terrain), 2, 3, ... = smaller components ranked by size
        -- a separate landmass, an isolated rock formation across water, anything
        genuinely disconnected from the base terrain by a real discontinuity, kept
        as real geometry rather than discarded. Downstream stages route non-primary
        components to their own separate mesh instead of folding them into the base
        terrain. cell_relief is the raw Y range
        (max - min, in metres) among all depth samples that landed in that one grid
        cell before being collapsed to their mean -- evidence of real vertical
        structure (e.g. a cliff face narrower than one grid cell) that survives only
        here, since it is destroyed by the very next step (averaging) and therefore
        invisible to any gradient computed on the output height map. cell_slope is
        the surface tilt (degrees from horizontal, 0-90) measured directly from a
        local least-squares plane fit through the raw points in that cell -- a
        direct 3-D orientation measurement rather than a proxy inferred from either
        a 2-D height gradient or the scalar Y range.

        In pinhole mode, every depth pixel is forward-projected and binned, so
        cell_relief is populated everywhere multiple samples land in one cell;
        cell_slope is not computed in pinhole mode (out of scope for this change)
        and is all zeros. In equirectangular mode, the primary projection
        (inverse_map_panorama_to_grid) takes a single inverse-mapped sample per
        cell -- necessary far from the camera, where one panorama pixel covers a
        huge ground area and forward projection would leave most cells empty --
        so on its own it has no intra-cell distribution to measure. Near the
        camera the opposite is true: many panorama pixels legitimately land in one
        fine grid cell. _forward_project_cell_stats recovers that near-field
        density via true forward projection, and wherever a cell collects
        min_forward_samples or more independent points its mean/relief/slope
        override the single-sample result; elsewhere the inverse-mapped value
        stands as before.

        camera_height_meters: assumed height of the camera above the ground plane
                              (used to derive the Y floor filter and flood-fill seed).
        ground_y_max: upper Y bound in camera space; points with Y <= this are ground
                      candidates (e.g. -0.5 = at least 0.5 m below camera). Only
                      enforced where region_type_mask is unavailable -- a real
                      slope, hill, or cliff can genuinely rise above the camera's
                      own eye level (someone standing partway up a mountain still
                      looks at more mountain going up), so wherever a pixel is
                      semantically classified as ground-valid (see
                      RegionType.ground_valid), that classification alone is
                      trusted and this ceiling is skipped for it. Without a region
                      mask there's no way to distinguish real elevated terrain from
                      a depth-model sky/artifact glitch, so the ceiling still
                      applies as a fallback guard in that case.
        sky_mask: optional bool (H, W) array from the depth model where True = sky.
                  Sky pixels are excluded before projection, preventing the horizon
                  artefacts that come from sky pixels being assigned far depth values.
        flood_fill: if True, label connected walkable-ground components (see
                    _label_ground_components) and keep every one large enough to
                    qualify, instead of every cell in the grid; stops at height
                    discontinuities and empty cells (sky gaps).
        flood_fill_max_step: maximum Y change (metres) between adjacent cells allowed
                             within one component; larger steps are treated as walls.
        min_component_area_fraction: components smaller than this fraction of the
                             grid are folded into the interpolated-gap fallback
                             instead of kept as their own component.
        grid_size_meters: side length of the square grid; both X and Z span ±half.
        use_equirectangular: treat depth as equirectangular (radial distances); otherwise
                             use pinhole unprojection via intrinsics.
        panorama_depth: optional 360° equirectangular depth map (radial metres). After
                        the primary depth is projected and flood-filled, any still-empty
                        grid cells are filled from this source before interpolation.
        nadir_exclusion_radius: when use_equirectangular=True, ground pixels whose
                                horizontal distance from the camera is less than this
                                value (metres) are discarded.  Equirectangular depth
                                models are unreliable near the nadir (bottom of the
                                panorama), where distortion peaks; those bad estimates
                                project directly under the camera and generate a deep
                                bowl.  The excluded cells are later filled by
                                interpolation from the surrounding reliable ring.
        certainty_falloff_meters: distance at which observed-ground certainty decays
                                  to 0.5 (see util.projection_utils.ground_projection_certainty).
                                  This is a depth-model-trust radius, not the physical
                                  camera height — using camera height here (as before)
                                  collapses certainty within a couple of metres, well
                                  under any usable confidence_threshold downstream.
        elevation_distortion_power: exponent on cos(phi) (phi = true camera-relative
                                  elevation angle to each observed cell), multiplied
                                  into certainty. This is the standard equirectangular
                                  areal-distortion factor: 1 at the horizon ("straight
                                  ahead", least warped), falling to 0 at the top/bottom
                                  of the source panorama (looking straight up or down,
                                  where equirectangular stretching -- and therefore raw
                                  depth-estimate reliability -- is worst). Independent
                                  of certainty_falloff_meters: that captures how much a
                                  depth error amplifies into ground-position error at
                                  grazing incidence; this captures source-image
                                  reliability by row, regardless of ground distance.
                                  0 disables (cos^0 == 1, neutral). See _build_certainty.
        min_forward_samples: equirectangular mode only. Minimum number of independent
                            forward-projected panorama pixels that must land in a grid
                            cell before its single inverse-mapped sample is overridden
                            by real per-cell statistics (mean/relief/slope). Below this,
                            a plane fit is either impossible (<3 points) or too noisy to
                            trust over the existing inverse-mapped value.
        fill_boundary_falloff_cells: see _interpolate — how many cells (at each
                            octave's own resolution) beyond a real observation the
                            synthetic fill's injected noise needs before reaching full
                            amplitude. Cells nearer a real observation than this stay
                            close to its diffused trend instead of drifting via noise.
        despike_threshold_m: equirectangular mode only. A single-inverse-mapped-
                            sample cell (i.e. not in dense_mask — see
                            _forward_project_cell_stats) whose height differs from
                            its own despike_window-sized neighbourhood median by
                            more than this is replaced by that median. Single-
                            sample cells have no intra-cell averaging to smooth
                            over source depth-map noise; a "flying pixel" edge
                            where the depth model flickers between two competing
                            surfaces from one equirectangular pixel to the next
                            otherwise lands as an alternating checkerboard of
                            wildly different heights between adjacent grid cells.
                            0 disables. See _despike_single_sample_cells.
        despike_window: odd cell-window size for the local median used above.
        despike_reference_distance_m: equirectangular mode only. Per-pixel depth-
                            model noise grows with sampled radial range, and is
                            further amplified for elevated/steep rays via
                            Y = depth * sin(phi) -- a single fixed
                            despike_threshold_m tuned for near, roughly-flat
                            ground under-catches real noise on longer, steeper
                            single-sample rays. The effective threshold shrinks
                            below despike_threshold_m (never grows past it) for
                            cells sampled beyond this distance, scaled by
                            despike_reference_distance_m / sampled_depth. See
                            _despike_single_sample_cells.
        region_closing_iterations: equirectangular mode only. Binary closing
                            iterations applied to the ground-valid classification
                            (see _ground_valid_mask) before it gates which pixels
                            count as ground. Per-pixel semantic segmentation is
                            noisy right at class boundaries; closing removes
                            single-pixel holes/islands that would otherwise show
                            up as salt-and-pepper speckle in both the observed
                            mask and the height values sampled from it. 0 disables.
        single_sample_blur_sigma: equirectangular mode only. Gaussian sigma (in
                            grid cells) for a partial blur applied only to
                            single-inverse-mapped-sample cells (not backed by
                            _forward_project_cell_stats' dense_mask), blended in
                            per-cell by (1 - certainty) -- confident single-sample
                            cells stay close to their raw value, poorly-
                            conditioned ones (see elevation_distortion_power) lean
                            further toward the local average. Dense cells are
                            never touched: they already average multiple real
                            samples, so there's no single-ray noise to correct.
                            Targets pervasive per-pixel depth-model dither that
                            despiking's outlier test can't catch (neighbouring
                            single-sample cells usually carry similar-magnitude
                            noise, so no individual cell stands out from a clean
                            local median). 0 disables.
        """
        d = depth.depth.astype(np.float32)
        h, w = d.shape
        half = grid_size_meters / 2.0
        ground_y_min = -(camera_height_meters + 5.0)

        if use_equirectangular:
            # Inverse mapping: for each grid cell, look up the panorama pixel that
            # would observe a ground point there, then sample depth directly.
            # Every cell gets exactly one clean lookup, avoiding the spatial stretching
            # that plagues forward projection near the horizon.
            d_masked = d.copy()
            if sky_mask is not None and sky_mask.shape == d.shape:
                d_masked[sky_mask] = np.nan

            sampled_depth, pano_u, pano_v, X_grid, Z_grid = inverse_map_panorama_to_grid(
                Depth(d_masked), grid_size_meters, grid_resolution, camera_height_meters
            )
            r_grid = np.sqrt(
                X_grid.astype(np.float64) ** 2 + Z_grid.astype(np.float64) ** 2
            ).astype(np.float32)
            phi_grid = -np.arctan2(camera_height_meters, np.maximum(r_grid, 1e-6)).astype(np.float32)
            Y_grid = (sampled_depth * np.sin(phi_grid)).astype(np.float32)

            region_valid = None
            if region_type_mask is not None:
                ground_valid_pano = HeightMapGenerator._ground_valid_mask(
                    region_type_mask, d.shape, closing_iterations=region_closing_iterations
                )
                region_valid = nearest_sample_grid(
                    ground_valid_pano.astype(np.uint8), pano_u, pano_v
                ).astype(bool)

            valid = np.isfinite(sampled_depth) & (sampled_depth > 0) & (Y_grid >= ground_y_min)
            if region_valid is not None:
                # A semantically-classified ground/terrain pixel (mountain, hill,
                # cliff, trail, ...) can legitimately sit above the camera's own eye
                # level -- someone standing partway up a slope still looks at more
                # slope going up. ground_y_max below is only a fallback sky/artifact
                # guard for when there's no classification to trust instead.
                valid &= region_valid
            else:
                valid &= Y_grid <= ground_y_max
            if sky_mask is not None and sky_mask.shape == d.shape:
                valid &= ~nearest_sample_grid(sky_mask.astype(np.uint8), pano_u, pano_v).astype(bool)
            if nadir_exclusion_radius > 0:
                valid &= r_grid >= nadir_exclusion_radius

            if not np.any(valid):
                zeros = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
                return zeros, zeros.copy(), zeros.copy(), zeros.copy(), np.zeros((grid_resolution, grid_resolution), dtype=bool)

            height_map = np.full((grid_resolution, grid_resolution), np.nan, dtype=np.float32)
            height_map[valid] = Y_grid[valid]
            # Single inverse-mapped sample per cell -- no intra-cell distribution to measure
            # on its own; overridden below wherever forward projection finds enough real
            # samples in the same cell.
            cell_relief = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)

            dense_mask, fwd_mean_y, fwd_relief, fwd_slope_deg, any_forward_mask = (
                HeightMapGenerator._forward_project_cell_stats(
                    d, sky_mask, region_type_mask, grid_size_meters, grid_resolution,
                    ground_y_max, ground_y_min, nadir_exclusion_radius, min_forward_samples,
                    region_closing_iterations,
                )
            )
            if dense_mask.any():
                height_map[dense_mask] = fwd_mean_y[dense_mask]
                cell_relief[dense_mask] = fwd_relief[dense_mask]
            cell_slope_deg = fwd_slope_deg

            # The primary single-sample ray always assumes its target lies on a
            # flat ground plane at the camera's own height (phi_grid <= 0
            # identically), so Y_grid can never represent a point above camera
            # height even when the real surface there does -- it's a correct
            # position along *some* fixed downward ray, just not necessarily the
            # one that would actually reach whatever is really at this cell's
            # (X, Z). Any real forward-projected sample -- dense or not -- is a
            # genuine unprojection using that pixel's own true ray angle, so it's
            # always at least as correctly positioned/signed as the primary ray's
            # flat-ground guess (an earlier version of this tried to gate this on
            # a flat-ground depth-consistency check first; real outdoor terrain
            # routinely isn't flat -- rolling meadow, a slope, a distant ridge --
            # so that consistency check fired on the majority of ordinary cells
            # and collapsed coverage. Real forward-projected data is simply
            # always preferred instead, unconditionally, whenever it exists).
            if any_forward_mask.any():
                use_weak = valid & ~dense_mask & any_forward_mask
                height_map[use_weak] = fwd_mean_y[use_weak]
                cell_relief[use_weak] = fwd_relief[use_weak]

            if despike_threshold_m > 0:
                height_map, spike_mask = HeightMapGenerator._despike_single_sample_cells(
                    height_map, protected_mask=dense_mask,
                    threshold_m=despike_threshold_m, window=despike_window,
                    distance_m=sampled_depth, reference_distance_m=despike_reference_distance_m,
                )
                if spike_mask.any():
                    cell_relief[spike_mask] = 0.0
                    cell_slope_deg[spike_mask] = 0.0
                    if debug_dir is not None:
                        PIL.Image.fromarray((spike_mask * 255).astype(np.uint8), "L").save(
                            debug_dir / "heightmap_despiked_cells.png"
                        )

        else:
            if sky_mask is not None and sky_mask.shape == d.shape:
                d = d.copy()
                d[sky_mask] = np.nan

            cx = np.arange(w, dtype=np.float32)
            cy = np.arange(h, dtype=np.float32)
            cx, cy = np.meshgrid(cx, cy)
            X = (cx - intrinsics.px) * d / intrinsics.fx
            Y = -((cy - intrinsics.py) * d / intrinsics.fy)
            Z = d

            ground_mask = (
                (Y >= ground_y_min)
                & (np.abs(Z) <= half) & (np.abs(X) <= half)
                & np.isfinite(d)
            )
            if region_type_mask is not None:
                # See the equirectangular path above: classified ground/terrain is
                # allowed above the camera's own eye level; ground_y_max is only a
                # fallback guard for when there's no classification to trust instead.
                ground_mask &= HeightMapGenerator._ground_valid_mask(
                    region_type_mask, d.shape, closing_iterations=region_closing_iterations
                )
            else:
                ground_mask &= Y <= ground_y_max

            Xg, Yg, Zg = X[ground_mask], Y[ground_mask], Z[ground_mask]
            if len(Xg) == 0:
                zeros = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
                return zeros, zeros.copy(), zeros.copy(), zeros.copy(), np.zeros((grid_resolution, grid_resolution), dtype=bool)

            x_edges = np.linspace(-half, half, grid_resolution + 1)
            z_edges = np.linspace(-half, half, grid_resolution + 1)
            xi = np.digitize(Xg, x_edges) - 1
            zi = np.digitize(Zg, z_edges) - 1
            in_bounds = (
                (xi >= 0) & (xi < grid_resolution)
                & (zi >= 0) & (zi < grid_resolution)
            )
            xi, zi, Yg = xi[in_bounds], zi[in_bounds], Yg[in_bounds]

            height_sum = np.zeros((grid_resolution, grid_resolution), dtype=np.float64)
            height_cnt = np.zeros((grid_resolution, grid_resolution), dtype=np.int32)
            np.add.at(height_sum, (zi, xi), Yg)
            np.add.at(height_cnt, (zi, xi), 1)

            height_map = np.full((grid_resolution, grid_resolution), np.nan, dtype=np.float32)
            has_data = height_cnt > 0
            height_map[has_data] = (height_sum[has_data] / height_cnt[has_data]).astype(np.float32)

            # Intra-cell relief: the Y range spanned by every sample that landed in
            # this cell, captured *before* they get collapsed to the mean above. A
            # near-vertical face narrower than one grid cell puts many samples in
            # the same (x, z) column spanning a large Y range; averaging smears
            # that into one flat point, and no gradient computed on height_map
            # downstream can ever recover it. Cells with only one sample get 0
            # (nothing to measure a range from), same as truly flat ground.
            height_max = np.full((grid_resolution, grid_resolution), -np.inf, dtype=np.float64)
            height_min = np.full((grid_resolution, grid_resolution),  np.inf, dtype=np.float64)
            np.maximum.at(height_max, (zi, xi), Yg)
            np.minimum.at(height_min, (zi, xi), Yg)
            cell_relief = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
            cell_relief[has_data] = (height_max[has_data] - height_min[has_data]).astype(np.float32)
            # Plane-fit slope is only computed for the equirectangular path above.
            cell_slope_deg = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)

        # Label connected walkable ground components (see _label_ground_components):
        # component 1 is the largest -- the base terrain -- component 2, 3, ... are
        # smaller components (a separate landmass, a rock formation across water,
        # anything genuinely disconnected from the base terrain by a real
        # discontinuity) kept as real geometry rather than discarded. Cells not in
        # any qualifying component are set to NaN and filled by _interpolate's
        # synthetic fill, same as the old single-component flood fill did for
        # everything outside its one kept component.
        if flood_fill:
            component_id = HeightMapGenerator._label_ground_components(
                height_map, flood_fill_max_step, min_component_area_fraction
            )
            accepted = component_id > 0
            height_map[~accepted] = np.nan
            cell_relief[~accepted] = 0.0
            cell_slope_deg[~accepted] = 0.0
        else:
            component_id = (~np.isnan(height_map)).astype(np.int32)

        # Fill remaining NaN cells from the panorama depth (360° coverage) before
        # falling back to pure interpolation. Only cells that are still empty after
        # the primary projection are touched, so rectilinear data always wins.
        if panorama_depth is not None:
            height_map = HeightMapGenerator._fill_from_panorama_depth(
                height_map=height_map,
                panorama_depth=panorama_depth,
                sky_mask=sky_mask,
                grid_size_meters=grid_size_meters,
                grid_resolution=grid_resolution,
                ground_y_max=ground_y_max,
                ground_y_min=ground_y_min,
                camera_height_meters=camera_height_meters,
                region_type_mask=region_type_mask,
                nadir_exclusion_radius=nadir_exclusion_radius,
                region_closing_iterations=region_closing_iterations,
            )

        # True observed mask: cells with a genuine direct measurement (primary
        # projection, dense forward-projected stats, or panorama fill) that survived
        # flood-fill -- captured before the synthetic flat-ground prior below
        # overwrites cells with a fabricated value, and before _interpolate fills
        # remaining gaps, so it never includes anything but real point-cloud data.
        # Unlike `certainty` (continuous, decays with distance and is nonzero even
        # on the synthetic flat-prior cells), this is what downstream stages should
        # use to decide "is this a real point" rather than "how much do we trust it."
        true_observed = ~np.isnan(height_map)

        # Each observed cell's own panorama UV, captured now from this cell's own
        # (X, height, Z) -- before Terrain Reconstruction's DEM solve, Terrain
        # Noise Refinement, single_sample_blur_sigma below, _smooth_edge_preserving,
        # or the mesh generator's own Poisson-resampling/vertex noise get anywhere
        # near this value. All of those exist to make the *visual*/geometric height
        # look better and are fine to perturb Y by a few centimetres -- but right
        # where a slope's incline happens to match the current viewing angle to it,
        # equirectangular elevation angle is nearly insensitive to distance walked
        # up-slope, which makes the *inverse* -- how much the sampled panorama row
        # shifts per unit of height error -- blow up. Any of those downstream
        # perturbations would otherwise get amplified into a visible mismatch
        # between the mesh's own silhouette and the photographed ridge line right
        # there. Recomputing UV from Y at that point instead of here inherits that
        # amplification; storing it now, from the height as actually measured,
        # avoids it entirely for any cell true_observed marks as real.
        pano_uv_u, pano_uv_v = HeightMapGenerator._panorama_uv_from_height(
            height_map, grid_size_meters, grid_resolution,
        )

        # Ground-under-the-user prior: the equirectangular depth model is unreliable
        # near the nadir, so cells within the nadir exclusion radius get a synthetic
        # value rather than a direct sample. Rather than assuming a hard-flat plane
        # at -camera_height_meters, interpolate inward from the real terrain just
        # outside the disc via pure Laplacian diffusion (no injected noise, unlike
        # _interpolate's synthetic multi-octave fill for large gaps) -- this follows
        # whatever local slope the real surrounding ground actually has, while still
        # deferring entirely to real data at the boundary. These cells receive a low
        # fixed certainty so the solver treats them as a soft prior rather than an
        # observation, letting the Laplacian blend them smoothly into real terrain
        # observations beyond the zone.
        cell_m = grid_size_meters / grid_resolution
        _x_c = np.linspace(-half + cell_m / 2.0, half - cell_m / 2.0, grid_resolution, dtype=np.float32)
        _X_cell, _Z_cell = np.meshgrid(_x_c, _x_c)
        _r_cell = np.sqrt(_X_cell.astype(np.float64) ** 2 + _Z_cell.astype(np.float64) ** 2).astype(np.float32)

        flat_prior_mask = np.zeros((grid_resolution, grid_resolution), dtype=bool)
        if nadir_exclusion_radius > 0:
            flat_prior_mask = _r_cell <= nadir_exclusion_radius

            # The disc is always centred on the grid; diffuse over a small local
            # crop around it (padded well past the disc radius to reach real
            # boundary data) rather than the whole grid -- the disc's resolved
            # value only depends on its immediate surroundings.
            radius_cells = nadir_exclusion_radius / cell_m
            pad = max(int(radius_cells * 6), 64)
            c0 = grid_resolution // 2
            lo, hi = max(0, c0 - pad), min(grid_resolution, c0 + pad)

            crop_hm = height_map[lo:hi, lo:hi]
            crop_disc = flat_prior_mask[lo:hi, lo:hi]
            crop_known = ~np.isnan(crop_hm) & ~crop_disc

            if crop_known.any():
                seed = np.where(np.isnan(crop_hm), 0.0, crop_hm).astype(np.float32)
                diffused = diffuse_heightmap(
                    seed, crop_known, n_iters=max(200, int(radius_cells) * 8), seed_from='nearest',
                )
                height_map[lo:hi, lo:hi] = np.where(crop_disc, diffused, crop_hm)
            else:
                # No real terrain nearby yet to interpolate from -- fall back to
                # the flat assumption.
                height_map[flat_prior_mask] = -camera_height_meters

            # Any measured relief/slope here belonged to a height value we just
            # discarded in favour of the prior -- it's no longer evidence of
            # anything.
            cell_relief[flat_prior_mask] = 0.0
            cell_slope_deg[flat_prior_mask] = 0.0

        # Cells filled by the panorama-depth fallback or the nadir flat-ground prior
        # above weren't part of the walkable-component analysis (they're
        # supplementary data / a synthetic prior, not connectivity evidence), but
        # they're still real content the base terrain owns -- fold them into the
        # primary component rather than leaving them at 0 (which downstream stages
        # would otherwise treat as an untethered, roughly one-cell-wide "component"
        # scattered wherever panorama-fill happened to land).
        component_id[(component_id == 0) & ~np.isnan(height_map)] = 1

        # Certainty: sin²(elevation) × smooth nadir ramp. The ramp rises from 0 at
        # nadir_exclusion_radius to full geometric certainty at nadir_exclusion_radius
        # + nadir_ramp_width, avoiding the hard ring artifact a step boundary creates.
        # Flat-prior cells get a fixed low certainty (flat_zone_certainty).
        observed = ~np.isnan(height_map)
        certainty = HeightMapGenerator._build_certainty(
            observed, height_map, grid_size_meters, grid_resolution, certainty_falloff_meters,
            nadir_exclusion_radius=nadir_exclusion_radius,
            nadir_ramp_width=nadir_ramp_width,
            elevation_distortion_power=elevation_distortion_power,
        )
        if flat_prior_mask.any():
            certainty[flat_prior_mask] = flat_zone_certainty

        # Partial, density-gated blur: single-inverse-mapped-sample cells (not
        # backed by _forward_project_cell_stats' dense_mask) have no intra-cell
        # averaging to fall back on, so per-pixel depth-model noise survives as
        # visible per-cell dither -- not sharp enough for despiking's outlier test
        # (a cell disagreeing with an otherwise-agreeing neighbourhood), since
        # neighbouring single-sample cells usually carry similar-magnitude noise
        # rather than one cell standing out from a clean local median. Blur is
        # blended in per-cell by (1 - certainty), which by this point already
        # reflects both distance and projection-distortion trust (see
        # _build_certainty) -- confident single-sample cells stay close to their
        # raw value, poorly-conditioned ones lean further toward the local
        # average. Dense cells are never blurred: they're already a real
        # multi-sample average, so there's no single-ray noise here to correct.
        if use_equirectangular and single_sample_blur_sigma > 0:
            blurred = gaussian_filter(
                np.where(observed, height_map, 0.0), sigma=single_sample_blur_sigma
            )
            blur_weight = np.where(observed & ~dense_mask, 1.0 - certainty, 0.0)
            height_map = np.where(
                observed, height_map * (1.0 - blur_weight) + blurred * blur_weight, height_map
            ).astype(np.float32)

        if debug_dir is not None:
            PIL.Image.fromarray((observed * 255).astype(np.uint8), "L").save(
                debug_dir / "heightmap_observed_mask.png"
            )
            # Raw height map before interpolation: NaN cells shown as the minimum value.
            raw_viz = height_map.copy()
            fill_val = float(np.nanmin(raw_viz)) if observed.any() else 0.0
            raw_viz[~observed] = fill_val
            Depth(raw_viz).normalize().save_debug_image(debug_dir / "heightmap_raw.png")
            Depth(certainty.copy()).normalize().save_debug_image(
                debug_dir / "heightmap_certainty.png"
            )
            if cell_relief.any():
                Depth(cell_relief.copy()).normalize().save_debug_image(
                    debug_dir / "heightmap_cell_relief.png"
                )
            if cell_slope_deg.any():
                Depth(cell_slope_deg.copy()).normalize().save_debug_image(
                    debug_dir / "heightmap_cell_slope.png"
                )

        result = HeightMapGenerator._interpolate(
            height_map, boundary_falloff_cells=fill_boundary_falloff_cells
        )
        if smooth_sigma > 0:
            result = HeightMapGenerator._smooth_edge_preserving(result, max_sigma=smooth_sigma)

        if debug_dir is not None:
            HeightMapGenerator._save_radial_profile(
                result, certainty, grid_size_meters, debug_dir / "heightmap_radial_profile.json"
            )
            PIL.Image.fromarray((true_observed * 255).astype(np.uint8), "L").save(
                debug_dir / "heightmap_true_observed_mask.png"
            )
            n_components = int(component_id.max())
            if n_components > 0:
                # Distinct intensities per component id (0 stays black); mainly
                # useful to eyeball how many components exist and their relative
                # size/shape, not precise values.
                viz = (component_id.astype(np.float32) / max(n_components, 1) * 255).astype(np.uint8)
                PIL.Image.fromarray(viz, "L").save(debug_dir / "heightmap_component_id.png")

        return result, certainty, cell_relief, cell_slope_deg, true_observed, component_id, pano_uv_u, pano_uv_v

    @staticmethod
    def _panorama_uv_from_height(
        height_map: np.ndarray,
        grid_size_meters: float,
        grid_resolution: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Per-cell normalised [0, 1] panorama UV from each grid cell's own (X,
        height, Z) -- the exact same projection Panorama.mesh_uvs/_lat_lon uses
        per-vertex (same u = (lon+pi)/(2*pi), v = 0.5 + lat/pi, matching the
        V-flip convention mesh_uvs already applies for glTF export), just
        evaluated once here at native grid resolution against this cell's own
        raw height, so it can be cached and reused verbatim wherever a cell is
        genuinely observed (HEIGHT_MAP_OBSERVED_MASK) instead of being re-
        derived later from whatever geometry-only-motivated smoothing/noise
        stages happen to leave in Y by then (see the call site's comment).

        NaN height produces NaN UV -- callers must gate on the observed mask
        (or otherwise treat NaN as "no reasonable UV yet") before using this;
        there is no defensible UV for a height that hasn't been measured.

        Returns (u, v), each (grid_resolution, grid_resolution) float32.
        """
        half = grid_size_meters / 2.0
        cell_m = grid_size_meters / grid_resolution
        x_c = np.linspace(-half + cell_m / 2.0, half - cell_m / 2.0, grid_resolution, dtype=np.float64)
        X_grid, Z_grid = np.meshgrid(x_c, x_c)  # rows = Z, cols = X -- same convention as height_map

        lon = np.arctan2(X_grid, Z_grid)
        r_xz = np.sqrt(X_grid ** 2 + Z_grid ** 2).clip(1e-6)
        lat = np.arctan2(height_map.astype(np.float64), r_xz)

        u = ((lon + np.pi) / (2.0 * np.pi)).astype(np.float32)
        v = (0.5 + lat / np.pi).astype(np.float32)
        return u, v

    @staticmethod
    def _ground_valid_mask(
        region_type_mask: np.ndarray,
        target_shape: tuple[int, int],
        closing_iterations: int = 2,
    ) -> np.ndarray:
        """
        Boolean ground-valid mask (see RegionType.ground_valid) from a per-pixel
        semantic region classification, resized to target_shape with small
        holes/islands closed.

        Per-pixel semantic segmentation is noisy right at class boundaries
        (grass vs rock vs vegetation): a single stray pixel misclassified on
        either side of the boundary otherwise becomes a hole in an
        otherwise-solid ground region, or a false-positive island inside an
        otherwise-solid occluded one. Sampled directly (as every caller of this
        mask does), that shows up as a salt-and-pepper flicker in which cells
        count as observed -- and since each affected cell has no intra-cell
        averaging to fall back on, the flicker shows up as visible speckle in
        the height values themselves, not just the mask. A small binary closing
        (dilate then erode) removes holes/islands up to its own size while
        leaving real, multi-pixel occlusion boundaries (an actual patch of
        vegetation) untouched.
        """
        rm = np.round(region_type_mask).astype(np.int32)
        if rm.shape != target_shape:
            rm = zoom(rm, (target_shape[0] / rm.shape[0], target_shape[1] / rm.shape[1]), order=0)
        valid = np.isin(rm, _VALID_REGION_INDICES)
        if closing_iterations > 0:
            valid = binary_closing(valid, iterations=closing_iterations)
        return valid

    @staticmethod
    def _forward_project_cell_stats(
        d: np.ndarray,
        sky_mask: Optional[np.ndarray],
        region_type_mask: Optional[np.ndarray],
        grid_size_meters: float,
        grid_resolution: int,
        ground_y_max: float,
        ground_y_min: float,
        nadir_exclusion_radius: float,
        min_samples: int,
        region_closing_iterations: int = 2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward-project every panorama pixel (rather than one inverse-mapped ray per
        output cell) and compute real per-cell statistics wherever any independent
        samples land in the same cell to support them.

        Near the camera, many panorama pixels legitimately land in one fine grid
        cell -- that intra-cell distribution is exactly what the single-sample
        inverse mapping in the equirectangular branch above cannot see. Far from
        the camera the opposite is true (one pixel covers a huge ground area, most
        cells stay empty), which is why inverse mapping is used as the primary
        projection there. This recovers:

          mean_y    -- per-cell mean elevation from real forward-projected samples,
                       instead of a single assumed-flat-ground ray. Populated for
                       any cell with >= 1 sample (see any_mask), not just dense
                       ones: unlike the primary inverse-mapped path (which always
                       assumes the target lies on a flat ground plane at the
                       camera's own height, and so can never represent a point
                       above camera height even when the real sample does), every
                       forward-projected point is a genuine, correctly-signed and
                       correctly-positioned 3-D unprojection regardless of count --
                       it just lacks intra-cell averaging below the dense
                       threshold, same as the primary path's single sample.
          relief    -- per-cell Y range (max - min); the same intra-cell-relief
                       signal the pinhole branch already produces, previously
                       always zero here. Also populated for any populated cell
                       (0 for a lone sample, same as "nothing to measure a range
                       from").
          slope_deg -- per-cell surface tilt (0 = flat, 90 = vertical) from the
                       smallest-eigenvalue eigenvector of the points' 3x3
                       covariance matrix (a local least-squares plane fit) -- a
                       direct measurement of true 3-D orientation, not a proxy
                       inferred from a 2-D gradient or a scalar Y range. Requires
                       >= min_samples (dense_mask), since a plane fit needs >= 3
                       points and is noisy just above that.

        Grouping is done via a unique-linear-index groupby (np.unique +
        return_inverse) rather than scattering into full (grid_resolution,
        grid_resolution) accumulators, so cost and memory scale with the number
        of valid points / populated cells, not the square of grid_resolution.

        Returns (dense_mask, mean_y, relief, slope_deg, any_mask), all
        (grid_resolution, grid_resolution) except any_mask/dense_mask which are
        bool. any_mask marks every cell with >= 1 forward-projected sample;
        dense_mask (a subset) marks cells with >= min_samples, the only ones with
        a real slope_deg. mean_y/relief are populated wherever any_mask is set;
        slope_deg only wherever dense_mask is set.
        """
        empty = (
            np.zeros((grid_resolution, grid_resolution), dtype=bool),
            np.zeros((grid_resolution, grid_resolution), dtype=np.float32),
            np.zeros((grid_resolution, grid_resolution), dtype=np.float32),
            np.zeros((grid_resolution, grid_resolution), dtype=np.float32),
            np.zeros((grid_resolution, grid_resolution), dtype=bool),
        )

        d_masked = d.copy()
        if sky_mask is not None and sky_mask.shape == d.shape:
            d_masked[sky_mask] = np.nan

        X, Y, Z = Panorama.equirectangular_unproject(Depth(d_masked))

        half = grid_size_meters / 2.0
        valid = (
            np.isfinite(d_masked) & (d_masked > 0)
            & (Y >= ground_y_min)
            & (np.abs(X) <= half) & (np.abs(Z) <= half)
        )
        if nadir_exclusion_radius > 0:
            valid &= np.hypot(X, Z) >= nadir_exclusion_radius
        if region_type_mask is not None:
            # See the primary equirectangular projection above: a classified
            # ground/terrain pixel is allowed above the camera's own eye level;
            # ground_y_max is only a fallback guard when there's no classification.
            valid &= HeightMapGenerator._ground_valid_mask(
                region_type_mask, d.shape, closing_iterations=region_closing_iterations
            )
        else:
            valid &= Y <= ground_y_max

        Xg, Yg, Zg = X[valid], Y[valid], Z[valid]
        if len(Xg) == 0:
            return empty

        x_edges = np.linspace(-half, half, grid_resolution + 1)
        z_edges = np.linspace(-half, half, grid_resolution + 1)
        xi = np.digitize(Xg, x_edges) - 1
        zi = np.digitize(Zg, z_edges) - 1
        in_bounds = (xi >= 0) & (xi < grid_resolution) & (zi >= 0) & (zi < grid_resolution)
        xi, zi = xi[in_bounds], zi[in_bounds]
        Xg, Yg, Zg = Xg[in_bounds], Yg[in_bounds], Zg[in_bounds]
        if len(Xg) == 0:
            return empty

        lin = zi.astype(np.int64) * grid_resolution + xi.astype(np.int64)
        uniq_lin, inverse, counts = np.unique(lin, return_inverse=True, return_counts=True)
        n = counts.astype(np.float64)

        def scatter_sum(vals: np.ndarray) -> np.ndarray:
            out = np.zeros(len(uniq_lin), dtype=np.float64)
            np.add.at(out, inverse, vals.astype(np.float64))
            return out

        sum_x, sum_y, sum_z = scatter_sum(Xg), scatter_sum(Yg), scatter_sum(Zg)
        sum_xx, sum_yy, sum_zz = scatter_sum(Xg * Xg), scatter_sum(Yg * Yg), scatter_sum(Zg * Zg)
        sum_xy, sum_xz, sum_yz = scatter_sum(Xg * Yg), scatter_sum(Xg * Zg), scatter_sum(Yg * Zg)

        max_y = np.full(len(uniq_lin), -np.inf)
        min_y = np.full(len(uniq_lin), np.inf)
        np.maximum.at(max_y, inverse, Yg)
        np.minimum.at(min_y, inverse, Yg)

        mean_y = sum_y / n
        relief_1d = (max_y - min_y).astype(np.float64)

        # Need >= 3 points for a well-defined plane; min_samples is expected to
        # already be at least that (see HeightMapConfiguration), but never trust
        # a caller-supplied value below the mathematical minimum.
        dense = counts >= max(3, min_samples)
        dense_idx = np.nonzero(dense)[0]

        slope_1d = np.zeros(len(uniq_lin), dtype=np.float64)
        if len(dense_idx) > 0:
            nd = n[dense_idx]
            mx = sum_x[dense_idx] / nd
            my = sum_y[dense_idx] / nd
            mz = sum_z[dense_idx] / nd
            cov_xx = sum_xx[dense_idx] / nd - mx * mx
            cov_yy = sum_yy[dense_idx] / nd - my * my
            cov_zz = sum_zz[dense_idx] / nd - mz * mz
            cov_xy = sum_xy[dense_idx] / nd - mx * my
            cov_xz = sum_xz[dense_idx] / nd - mx * mz
            cov_yz = sum_yz[dense_idx] / nd - my * mz

            cov = np.zeros((len(dense_idx), 3, 3), dtype=np.float64)
            cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2] = cov_xx, cov_yy, cov_zz
            cov[:, 0, 1] = cov[:, 1, 0] = cov_xy
            cov[:, 0, 2] = cov[:, 2, 0] = cov_xz
            cov[:, 1, 2] = cov[:, 2, 1] = cov_yz

            # Ascending eigenvalues; eigvecs[:, :, 0] is the smallest-eigenvalue
            # eigenvector for each matrix in the batch -- the local plane normal.
            _, eigvecs = np.linalg.eigh(cov)
            normal_y = np.clip(np.abs(eigvecs[:, 1, 0]), 0.0, 1.0)
            slope_1d[dense_idx] = np.degrees(np.arccos(normal_y))

        dense_mask = np.zeros((grid_resolution, grid_resolution), dtype=bool)
        any_mask = np.zeros((grid_resolution, grid_resolution), dtype=bool)
        mean_y_grid = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
        relief_grid = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
        slope_grid = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)

        zi_all = (uniq_lin // grid_resolution).astype(np.int64)
        xi_all = (uniq_lin % grid_resolution).astype(np.int64)
        any_mask[zi_all, xi_all] = True
        mean_y_grid[zi_all, xi_all] = mean_y.astype(np.float32)
        relief_grid[zi_all, xi_all] = relief_1d.astype(np.float32)

        d_lin = uniq_lin[dense_idx]
        zi_d = (d_lin // grid_resolution).astype(np.int64)
        xi_d = (d_lin % grid_resolution).astype(np.int64)

        dense_mask[zi_d, xi_d] = True
        slope_grid[zi_d, xi_d] = slope_1d[dense_idx].astype(np.float32)

        return dense_mask, mean_y_grid, relief_grid, slope_grid, any_mask

    @staticmethod
    def _despike_single_sample_cells(
        height_map: np.ndarray,
        protected_mask: np.ndarray,
        threshold_m: float,
        window: int,
        distance_m: Optional[np.ndarray] = None,
        reference_distance_m: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Replace isolated height outliers among single-inverse-mapped-sample cells
        with their local neighbourhood median.

        Single-sample cells (protected_mask is False -- i.e. not backed by
        _forward_project_cell_stats' dense_mask) have no intra-cell averaging to
        smooth over source depth-map noise: a "flying pixel" edge where the depth
        model flickers between two competing surfaces from one equirectangular
        pixel to the next lands directly as an alternating checkerboard of wildly
        different heights between immediately adjacent grid cells, instead of the
        smooth gradient a real slope would produce.

        A cell is only corrected if it diverges from its own window-sized
        neighbourhood median by more than threshold_m. A genuine, spatially
        coherent step edge (a real cliff/ridge) moves many neighbouring cells
        together, so the local median there already reflects the step and isn't
        flagged; only a cell disagreeing with a neighbourhood that mostly agrees
        with itself gets touched -- the same principle as a standard median/
        Hampel despike filter for salt-and-pepper sensor noise. Dense cells
        (protected_mask) are never candidates, even if their value happens to
        differ from neighbours, since they're backed by real multi-sample
        statistics rather than a single noisy ray.

        distance_m (optional): per-cell sampled radial depth (e.g. equirectangular
        sampled_depth), same shape as height_map. Per-pixel depth-model noise
        grows with range, and Y = depth * sin(phi) further amplifies it for
        elevated/steep rays -- a fixed threshold_m tuned for near, roughly-flat
        ground silently under-catches real noise further out. When given, the
        effective threshold shrinks below threshold_m (never above it) for cells
        sampled beyond reference_distance_m, scaled by
        reference_distance_m / distance_m -- near cells keep the base threshold,
        far single-sample cells are held to a stricter one.

        Returns (height_map, spike_mask) -- a new array (input is not mutated)
        and the bool mask of cells that were replaced, so callers can also clear
        any relief/slope measurement that belonged to the discarded value.
        """
        valid = ~np.isnan(height_map)
        candidates = valid & ~protected_mask
        empty_spikes = np.zeros_like(valid)
        if not candidates.any():
            return height_map, empty_spikes

        # The local median needs a real value at every cell to be meaningful --
        # fill NaN gaps with their nearest valid neighbour first (cheap,
        # vectorized) rather than letting missing cells pollute the window.
        filled = height_map
        if not valid.all():
            nearest_idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
            filled = height_map[tuple(nearest_idx)]

        local_median = median_filter(filled, size=window)

        if distance_m is not None:
            scale = np.clip(reference_distance_m / np.maximum(distance_m, 1e-6), 0.0, 1.0)
            effective_threshold = threshold_m * scale
        else:
            effective_threshold = threshold_m

        spike = candidates & (np.abs(height_map - local_median) > effective_threshold)
        if not spike.any():
            return height_map, empty_spikes

        despiked = height_map.copy()
        despiked[spike] = local_median[spike]
        return despiked, spike

    @staticmethod
    def _build_certainty(
        observed: np.ndarray,
        height_map: np.ndarray,
        grid_size_meters: float,
        grid_resolution: int,
        certainty_falloff_meters: float,
        nadir_exclusion_radius: float = 0.0,
        nadir_ramp_width: float = 5.0,
        elevation_distortion_power: float = 1.0,
    ) -> np.ndarray:
        """
        Build a [0, 1] certainty map over the top-down grid.

        Observed cells are scored by three multiplied factors:

          1. Distance-decayed certainty (see ground_projection_certainty;
             falloff_m=certainty_falloff_meters, a depth-model-trust radius, not
             physical camera height).
          2. A smooth nadir ramp rising from 0 at nadir_exclusion_radius to 1 at
             nadir_exclusion_radius + nadir_ramp_width (squared so it has zero
             slope at the start, avoiding a visible ring in the solver output).
          3. cos(phi)^elevation_distortion_power, phi = the true camera-relative
             elevation angle to each cell's observed point (arctan2(Y, horizontal
             distance) -- not the flat-ground-assumed depression angle used
             elsewhere, since dense forward-projected cells can carry real
             elevation above the camera). This is the standard equirectangular
             areal-distortion factor: 1 straight ahead at the horizon (phi=0,
             least warped, one image pixel ~= one true solid angle), falling to 0
             at the top and bottom of the panorama (phi -> +-90 deg, zenith/nadir,
             where equirectangular stretching is worst and a single image pixel
             represents an increasingly ambiguous/compressed true direction).
             This is a real, independent degradation from (1): (1) captures how
             much a given depth error amplifies into ground-position error at
             grazing incidence; this captures how reliable the raw depth
             estimate itself tends to be, purely as a function of where in the
             source image it came from. 0 disables (cos^0 == 1, neutral).

        Unobserved cells get 0.
        """
        half = grid_size_meters / 2.0
        x_centers = np.linspace(-half, half, grid_resolution, endpoint=False, dtype=np.float32) + half / grid_resolution
        z_centers = np.linspace(-half, half, grid_resolution, endpoint=False, dtype=np.float32) + half / grid_resolution
        X_grid, Z_grid = np.meshgrid(x_centers, z_centers)
        r_grid = np.sqrt(X_grid.astype(np.float64) ** 2 + Z_grid.astype(np.float64) ** 2).astype(np.float32)
        nadir_ramp = np.clip(
            (r_grid - nadir_exclusion_radius) / max(float(nadir_ramp_width), 1e-6),
            0.0, 1.0,
        ).astype(np.float32) ** 2
        certainty_field = ground_projection_certainty(X_grid, Z_grid, certainty_falloff_meters) * nadir_ramp

        if elevation_distortion_power > 0:
            phi_true = np.arctan2(
                np.where(observed, height_map, 0.0).astype(np.float64), np.maximum(r_grid, 1e-6)
            )
            distortion = np.clip(np.cos(phi_true), 0.0, 1.0) ** elevation_distortion_power
            certainty_field = certainty_field * distortion.astype(np.float32)

        return np.where(observed, certainty_field, 0.0).astype(np.float32)

    @staticmethod
    def _label_ground_components(
        height_map: np.ndarray,
        max_step: float,
        min_component_area_fraction: float = 0.001,
    ) -> np.ndarray:
        """
        Connected-component labeling of walkable ground, ranked by size.

        A cell is walkable if it has data and its 3x3 neighbourhood height range
        (max - min) does not exceed max_step -- i.e. no discontinuity passes
        through it. Empty cells (NaN) can never bridge a gap, so they are excluded
        from both the max and min before the range is taken (sky-masked regions
        near the horizon naturally stop a component, as before).

        Unlike an earlier single-seed version of this function (which kept only
        the component connected to the camera's own position and discarded
        every other component to NaN, later overwritten by _interpolate's
        synthetic fill), every sufficiently large component is kept here. A
        physically separate landmass or rock formation across water -- or
        across any other genuine discontinuity, real depth data with nothing
        connecting it back to the camera's own footing -- is real observed
        geometry, not noise; discarding it and replacing it with fabricated
        terrain was never correct just because it happens to be unreachable
        from the seed. Downstream stages use the ranking here (see
        HEIGHT_MAP_COMPONENT_ID) to keep the largest component as the base
        terrain mesh and route every other qualifying component to its own
        separate mesh instead.

        Uses 8-connectivity rather than 4-connectivity: 4-connectivity measures
        reachability in Manhattan distance, which grows a diamond/star from any
        seed regardless of the actual terrain -- purely a grid-connectivity
        artifact. 8-connectivity approximates a circular ball much more closely.

        Returns an int32 (H, W) label array: 0 = not part of any qualifying
        component (falls through to _interpolate's synthetic fill, as the
        discarded region always did), 1 = the largest component, 2, 3, ... =
        smaller components ranked by size descending.

        min_component_area_fraction: components smaller than this fraction of
        the grid are folded into 0 (gap) rather than kept as their own
        component -- avoids spurious few-cell noise clusters becoming a
        "formation" of their own.
        """
        grid_h, grid_w = height_map.shape
        has_data = ~np.isnan(height_map)

        # -inf/+inf for missing cells so they never win the max/min at a boundary.
        local_max = maximum_filter(np.where(has_data, height_map, -np.inf), size=3)
        local_min = minimum_filter(np.where(has_data, height_map,  np.inf), size=3)
        walkable = has_data & (local_max - local_min <= max_step)

        structure = generate_binary_structure(2, 2)  # 8-connectivity
        labels, n_labels = label(walkable, structure=structure)
        if n_labels == 0:
            return np.zeros((grid_h, grid_w), dtype=np.int32)

        counts = np.bincount(labels.ravel())
        counts[0] = 0  # background (unwalkable / no data) is never a component
        min_cells = max(1, int(min_component_area_fraction * grid_h * grid_w))

        qualifying = np.flatnonzero(counts >= min_cells)
        if len(qualifying) == 0:
            return np.zeros((grid_h, grid_w), dtype=np.int32)
        qualifying = qualifying[np.argsort(-counts[qualifying])]  # largest first

        remap = np.zeros(n_labels + 1, dtype=np.int32)
        for new_id, old_label in enumerate(qualifying, start=1):
            remap[old_label] = new_id

        return remap[labels]

    @staticmethod
    def _fill_from_panorama_depth(
        height_map: np.ndarray,
        panorama_depth: "Depth",
        sky_mask: Optional[np.ndarray],
        grid_size_meters: float,
        grid_resolution: int,
        ground_y_max: float,
        ground_y_min: float,
        camera_height_meters: float,
        region_type_mask: Optional[np.ndarray] = None,
        nadir_exclusion_radius: float = 0.0,
        region_closing_iterations: int = 2,
    ) -> np.ndarray:
        """
        Fill empty (NaN) grid cells from a 360° equirectangular depth map via inverse
        mapping.  Existing data is never overwritten.
        """
        missing = np.isnan(height_map)
        if not np.any(missing):
            return height_map

        d_excl = panorama_depth.depth.astype(np.float32).copy()
        if sky_mask is not None and sky_mask.shape == d_excl.shape:
            d_excl[sky_mask] = np.nan
        if region_type_mask is not None:
            valid_region = HeightMapGenerator._ground_valid_mask(
                region_type_mask, panorama_depth.depth.shape, closing_iterations=region_closing_iterations
            )
            d_excl[~valid_region] = np.nan

        sampled_depth, _, _, X_grid, Z_grid = inverse_map_panorama_to_grid(
            Depth(d_excl), grid_size_meters, grid_resolution, camera_height_meters
        )
        r_grid = np.sqrt(
            X_grid.astype(np.float64) ** 2 + Z_grid.astype(np.float64) ** 2
        ).astype(np.float32)
        phi_grid = -np.arctan2(camera_height_meters, np.maximum(r_grid, 1e-6)).astype(np.float32)
        Y_grid = (sampled_depth * np.sin(phi_grid)).astype(np.float32)

        pano_valid = (
            np.isfinite(sampled_depth) & (sampled_depth > 0)
            & (Y_grid >= ground_y_min)
            & (r_grid >= nadir_exclusion_radius)
        )
        if region_type_mask is None:
            # d_excl above already NaN'd out non-ground-valid pixels when a region
            # mask was supplied, so classified ground/terrain is already allowed
            # above the camera's own eye level; ground_y_max is only a fallback
            # guard for when there's no classification to trust instead.
            pano_valid &= Y_grid <= ground_y_max
        result = height_map.copy()
        result[missing & pano_valid] = Y_grid[missing & pano_valid]
        return result

    @staticmethod
    def _smooth_edge_preserving(
        height_map: np.ndarray,
        max_sigma: float,
        edge_sensitivity: float = 1.0,
        n_levels: int = 5,
    ) -> np.ndarray:
        """
        Spatially-varying, edge-preserving Gaussian smooth.

        Each pixel's smoothing amount is driven by two weights multiplied together:

          distance weight  — 0 at grid centre (camera position, most reliable),
                             1 at corners (least reliable).  Same as before.

          flatness weight  — 1 where the local gradient is small (likely noise,
                             smooth freely), 0 where the gradient is large (real
                             structure, preserve it).

        The gradient threshold is set adaptively at the 85th percentile of all
        gradient magnitudes so it scales with the scene.  edge_sensitivity > 1
        preserves more structures; < 1 smooths more aggressively.

        Implementation: pre-compute n_levels+1 Gaussian blurs, then for each
        pixel pick between them according to the combined weight.
        """
        h, w = height_map.shape

        y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        dist = np.clip(np.sqrt(y ** 2 + x ** 2) / np.sqrt(2.0), 0.0, 1.0)

        gy, gx = np.gradient(height_map)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)

        scale = np.percentile(grad_mag, 85) / edge_sensitivity + 1e-6
        edge_weight = np.tanh(grad_mag / scale)
        edge_weight = gaussian_filter(edge_weight, sigma=2.0)
        flat_weight = (1.0 - edge_weight).clip(0.0, 1.0)

        smooth_weight = (dist * flat_weight).astype(np.float32)

        sigmas = np.linspace(0.0, max_sigma, n_levels + 1)
        levels = np.stack(
            [height_map if s == 0 else gaussian_filter(height_map, sigma=s)
             for s in sigmas],
            axis=0,
        )

        idx  = smooth_weight * n_levels
        lo   = np.floor(idx).astype(np.int32).clip(0, n_levels - 1)
        hi   = (lo + 1).clip(0, n_levels)
        frac = (idx - lo).astype(np.float32)

        rows, cols = np.mgrid[0:h, 0:w]
        return (levels[lo, rows, cols] * (1.0 - frac) +
                levels[hi, rows, cols] * frac).astype(np.float32)

    @staticmethod
    def _interpolate(
        height_map: np.ndarray,
        n_octaves: int = 4,
        noise_seed: int = 0,
        boundary_falloff_cells: float = 6.0,
    ) -> np.ndarray:
        """
        Multi-scale noise inpainting for unknown (NaN) cells.

        Coarse-to-fine: at the coarsest level, known cells are diffused outward
        via 4-neighbor Laplacian diffusion to seed a structurally coherent base.
        At each finer level the previous result is upsampled bicubically, noise
        with cubically-decreasing amplitude is injected into still-unknown cells
        (front-loading large-scale structure), and diffusion propagates boundary
        values inward.  Known cells are hard constraints throughout.

        boundary_falloff_cells: the injected noise at each level is additionally
        scaled by distance (in that level's own cells) to the nearest real known
        cell — 0 right at the boundary, ramping to full amplitude
        boundary_falloff_cells away. Without this, a cell immediately next to a
        real observation (e.g. the occluded top of a cliff whose base is
        directly observed) could get just as much random noise as a cell deep in
        a large unobserved region with no nearby evidence at all — every level
        already diffuses in the value from known neighbours, but the noise
        term itself didn't defer to how close a real observation actually was.
        Gating it means fill immediately adjacent to real data continues that
        data's trend (diffusion-dominated, "close to known neighbours"), while
        only cells with no nearby observation get real freedom to wander.
        """
        known_mask = ~np.isnan(height_map)
        if np.all(known_mask):
            return height_map
        if not np.any(known_mask):
            return np.zeros_like(height_map)

        rng = np.random.default_rng(noise_seed)
        h, w = height_map.shape

        noise_scale = float(np.std(height_map[known_mask]))

        def nan_downsample(factor: int) -> np.ndarray:
            sh, sw = max(1, h // factor), max(1, w // factor)
            crop = height_map[: sh * factor, : sw * factor]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanmean(
                    crop.reshape(sh, factor, sw, factor), axis=(1, 3)
                ).astype(np.float32)

        ZI: Optional[np.ndarray] = None

        for octave in range(n_octaves - 1, -1, -1):
            factor = 2 ** octave
            ds = nan_downsample(factor) if factor > 1 else height_map.copy()
            ds_mask = ~np.isnan(ds)
            sh, sw = ds.shape
            ds_vals = np.where(ds_mask, ds, 0.0)

            if ZI is None:
                # Coarsest level: seed from nearest known neighbour so diffusion
                # starts at the last-seen boundary height, not the global mean.
                ZI = diffuse_heightmap(ds_vals, ds_mask, n_iters=max(200, sh * 4), seed_from='nearest')
            else:
                # Upsample bicubically and inject noise with cubic amplitude scaling.
                # Cubic front-loads structure at coarse levels; fine levels are near
                # noise-free so diffusion — not randomness — sets the last detail.
                ZI_up = zoom(ZI, (sh / ZI.shape[0], sw / ZI.shape[1]), order=3)
                amplitude = noise_scale * (octave ** 3) * 1e-2
                dist_to_known = distance_transform_edt(~ds_mask)
                boundary_gate = np.clip(
                    dist_to_known / max(boundary_falloff_cells, 1e-6), 0.0, 1.0
                ).astype(np.float32)
                noise = rng.standard_normal((sh, sw)).astype(np.float32) * amplitude * boundary_gate
                ZI = np.where(ds_mask, ds_vals, ZI_up + noise)

            n_iters = max(20, sh * 2 if octave == n_octaves - 1 else sh // 4)
            # 'keep': preserve the upsampled+noise state already set above.
            ZI = diffuse_heightmap(ZI, ds_mask, n_iters=n_iters, seed_from='keep')

        # Restore original known values exactly — diffusion must not drift them.
        return np.where(known_mask, height_map, ZI).astype(np.float32)

    @staticmethod
    def _save_radial_profile(
        height_map: np.ndarray,
        certainty: np.ndarray,
        grid_size_meters: float,
        path: Path,
        n_bins: int = 64,
    ) -> None:
        """
        Compute mean height and certainty as a function of radial distance from the
        camera origin and write the result as JSON.  Useful for spotting concentric
        ring artifacts: a ripple shows up as an oscillation in mean_height_m.
        """
        h, w = height_map.shape
        half = grid_size_meters / 2.0
        x_c = np.linspace(-half, half, w, endpoint=False) + half / w
        z_c = np.linspace(-half, half, h, endpoint=False) + half / h
        X_g, Z_g = np.meshgrid(x_c, z_c)
        r_2d = np.sqrt(X_g ** 2 + Z_g ** 2).ravel()

        hm_flat   = height_map.ravel()
        cert_flat = certainty.ravel()

        max_r    = float(r_2d.max())
        r_edges  = np.linspace(0.0, max_r, n_bins + 1)
        r_mids   = ((r_edges[:-1] + r_edges[1:]) / 2.0).tolist()
        bin_idx  = np.digitize(r_2d, r_edges) - 1
        bin_idx  = bin_idx.clip(0, n_bins - 1)

        height_means: list = []
        cert_means:   list = []
        for i in range(n_bins):
            mask = bin_idx == i
            height_means.append(float(hm_flat[mask].mean()) if mask.any() else None)
            cert_means.append(float(cert_flat[mask].mean()) if mask.any() else None)

        with open(path, "w") as f:
            json.dump({
                "radius_m":       r_mids,
                "mean_height_m":  height_means,
                "mean_certainty": cert_means,
            }, f, indent=2)
