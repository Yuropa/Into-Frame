import json
import warnings
import numpy as np
import PIL.Image
from pathlib import Path
from scipy.ndimage import gaussian_filter, generate_binary_structure, label, maximum_filter, minimum_filter, zoom
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
        debug_dir: Optional[Path] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project ground points from a depth map onto a top-down height grid.

        Returns (height_array, certainty_array, cell_relief_array), all
        (grid_resolution, grid_resolution) float32.  certainty is in [0, 1]:
        sin²(depression_angle) for cells with any direct observation (primary or
        panorama depth), 0 for pure interpolation.  cell_relief is the raw Y range
        (max - min, in metres) among all depth samples that landed in that one grid
        cell before being collapsed to their mean -- evidence of real vertical
        structure (e.g. a cliff face narrower than one grid cell) that survives only
        here, since it is destroyed by the very next step (averaging) and therefore
        invisible to any gradient computed on the output height map. Only populated
        in pinhole mode, where many depth pixels legitimately land in one cell;
        equirectangular mode takes a single inverse-mapped sample per cell, so
        there's no intra-cell distribution to measure and this is all zeros.

        camera_height_meters: assumed height of the camera above the ground plane
                              (used to derive the Y floor filter and flood-fill seed).
        ground_y_max: upper Y bound in camera space; points with Y <= this are ground
                      candidates (e.g. -0.5 = at least 0.5 m below camera).
        sky_mask: optional bool (H, W) array from the depth model where True = sky.
                  Sky pixels are excluded before projection, preventing the horizon
                  artefacts that come from sky pixels being assigned far depth values.
        flood_fill: if True, BFS from the grid centre outward to keep only connected
                    ground; stops at height discontinuities and empty cells (sky gaps).
        flood_fill_max_step: maximum Y change (metres) between adjacent cells allowed
                             during flood-fill; larger steps are treated as walls.
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

            valid = (
                np.isfinite(sampled_depth) & (sampled_depth > 0)
                & (Y_grid <= ground_y_max) & (Y_grid >= ground_y_min)
            )
            if sky_mask is not None and sky_mask.shape == d.shape:
                valid &= ~nearest_sample_grid(sky_mask.astype(np.uint8), pano_u, pano_v).astype(bool)
            if region_type_mask is not None:
                rm = np.round(region_type_mask).astype(np.int32)
                if rm.shape != d.shape:
                    rm = zoom(rm, (d.shape[0] / rm.shape[0], d.shape[1] / rm.shape[1]), order=0)
                valid &= np.isin(nearest_sample_grid(rm, pano_u, pano_v), _VALID_REGION_INDICES)
            if nadir_exclusion_radius > 0:
                valid &= r_grid >= nadir_exclusion_radius

            if not np.any(valid):
                zeros = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
                return zeros, zeros.copy(), zeros.copy()

            height_map = np.full((grid_resolution, grid_resolution), np.nan, dtype=np.float32)
            height_map[valid] = Y_grid[valid]
            # Single inverse-mapped sample per cell -- no intra-cell distribution to measure.
            cell_relief = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)

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
                (Y <= ground_y_max) & (Y >= ground_y_min)
                & (np.abs(Z) <= half) & (np.abs(X) <= half)
                & np.isfinite(d)
            )
            if region_type_mask is not None:
                rm = np.round(region_type_mask).astype(np.int32)
                if rm.shape != d.shape:
                    rm = zoom(rm, (d.shape[0] / rm.shape[0], d.shape[1] / rm.shape[1]), order=0)
                ground_mask &= np.isin(rm, _VALID_REGION_INDICES)

            Xg, Yg, Zg = X[ground_mask], Y[ground_mask], Z[ground_mask]
            if len(Xg) == 0:
                zeros = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
                return zeros, zeros.copy(), zeros.copy()

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

        # Flood-fill from the grid centre (camera XZ = 0,0) outward. Cells connected to
        # the starting point with small height steps are kept; everything else is set to
        # NaN and filled by nearest-neighbour extrapolation from the flood-fill boundary.
        if flood_fill:
            accepted = HeightMapGenerator._flood_fill_ground(
                height_map, camera_height_meters, flood_fill_max_step
            )
            height_map[~accepted] = np.nan
            cell_relief[~accepted] = 0.0

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
            )

        # Flat ground prior: pin all cells within the nadir exclusion radius to
        # -camera_height_meters. The equirectangular depth model is unreliable near
        # the nadir, and the ground directly under the user is expected to be
        # approximately level. These cells receive a low fixed certainty so the solver
        # treats them as a soft prior rather than an observation, letting the Laplacian
        # blend them smoothly into real terrain observations beyond the zone.
        cell_m = grid_size_meters / grid_resolution
        _x_c = np.linspace(-half + cell_m / 2.0, half - cell_m / 2.0, grid_resolution, dtype=np.float32)
        _X_cell, _Z_cell = np.meshgrid(_x_c, _x_c)
        _r_cell = np.sqrt(_X_cell.astype(np.float64) ** 2 + _Z_cell.astype(np.float64) ** 2).astype(np.float32)

        flat_prior_mask = np.zeros((grid_resolution, grid_resolution), dtype=bool)
        if nadir_exclusion_radius > 0:
            flat_prior_mask = _r_cell <= nadir_exclusion_radius
            height_map[flat_prior_mask] = -camera_height_meters
            # Any measured relief here belonged to a height value we just discarded
            # in favour of the flat prior -- it's no longer evidence of anything.
            cell_relief[flat_prior_mask] = 0.0

        # Certainty: sin²(elevation) × smooth nadir ramp. The ramp rises from 0 at
        # nadir_exclusion_radius to full geometric certainty at nadir_exclusion_radius
        # + nadir_ramp_width, avoiding the hard ring artifact a step boundary creates.
        # Flat-prior cells get a fixed low certainty (flat_zone_certainty).
        observed = ~np.isnan(height_map)
        certainty = HeightMapGenerator._build_certainty(
            observed, grid_size_meters, grid_resolution, certainty_falloff_meters,
            nadir_exclusion_radius=nadir_exclusion_radius,
            nadir_ramp_width=nadir_ramp_width,
        )
        if flat_prior_mask.any():
            certainty[flat_prior_mask] = flat_zone_certainty

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

        result = HeightMapGenerator._interpolate(height_map)
        if smooth_sigma > 0:
            result = HeightMapGenerator._smooth_edge_preserving(result, max_sigma=smooth_sigma)

        if debug_dir is not None:
            HeightMapGenerator._save_radial_profile(
                result, certainty, grid_size_meters, debug_dir / "heightmap_radial_profile.json"
            )

        return result, certainty, cell_relief

    @staticmethod
    def _build_certainty(
        observed: np.ndarray,
        grid_size_meters: float,
        grid_resolution: int,
        certainty_falloff_meters: float,
        nadir_exclusion_radius: float = 0.0,
        nadir_ramp_width: float = 5.0,
    ) -> np.ndarray:
        """
        Build a [0, 1] certainty map over the top-down grid.

        Observed cells are scored by distance-decayed certainty (see
        ground_projection_certainty; falloff_m=certainty_falloff_meters, a
        depth-model-trust radius, not physical camera height) multiplied by a
        smooth nadir ramp. The ramp rises from 0 at nadir_exclusion_radius to 1
        at nadir_exclusion_radius + nadir_ramp_width (squared so it has zero
        slope at the start, avoiding a visible ring in the solver output).
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
        return np.where(observed, certainty_field, 0.0).astype(np.float32)

    @staticmethod
    def _flood_fill_ground(
        height_map: np.ndarray,
        camera_height_meters: float,
        max_step: float,
    ) -> np.ndarray:
        """
        Connected-component ground mask, seeded from the grid centre.

        A cell is walkable if it has data and its 3x3 neighbourhood height range
        (max - min) does not exceed max_step -- i.e. no discontinuity passes
        through it. Empty cells (NaN) can never bridge a gap, so they are excluded
        from both the max and min before the range is taken (sky-masked regions
        near the horizon naturally stop the fill, as before). The accepted region
        is then the walkable component connected to the seed.

        Uses 8-connectivity rather than a 4-connected BFS: 4-connectivity measures
        reachability in Manhattan distance, which grows a diamond/star from the
        seed regardless of the actual terrain -- purely a grid-connectivity
        artifact, which is why it showed up in every heightmap_observed_mask.png.
        8-connectivity approximates a circular ball much more closely, and
        scipy.ndimage's compiled filters replace an O(H*W) pure-Python BFS with a
        handful of vectorised passes.
        """
        grid_h, grid_w = height_map.shape
        has_data = ~np.isnan(height_map)

        start_r, start_c = grid_h // 2, grid_w // 2

        # If the centre cell is empty, find the nearest filled cell to it.
        if not has_data[start_r, start_c]:
            ys, xs = np.where(has_data)
            if len(ys) == 0:
                return has_data
            dist = np.hypot(ys - start_r, xs - start_c)
            best = int(np.argmin(dist))
            start_r, start_c = int(ys[best]), int(xs[best])

        # -inf/+inf for missing cells so they never win the max/min at a boundary.
        local_max = maximum_filter(np.where(has_data, height_map, -np.inf), size=3)
        local_min = minimum_filter(np.where(has_data, height_map,  np.inf), size=3)
        walkable = has_data & (local_max - local_min <= max_step)
        walkable[start_r, start_c] = True  # the seed is always accepted

        structure = generate_binary_structure(2, 2)  # 8-connectivity
        labels, _ = label(walkable, structure=structure)
        seed_label = labels[start_r, start_c]
        if seed_label == 0:
            return np.zeros((grid_h, grid_w), dtype=bool)

        return labels == seed_label

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
            pd = panorama_depth.depth
            rm = np.round(region_type_mask).astype(np.int32)
            if rm.shape != pd.shape:
                rm = zoom(rm, (pd.shape[0] / rm.shape[0], pd.shape[1] / rm.shape[1]), order=0)
            valid_region = np.zeros(pd.shape, dtype=bool)
            for idx in _VALID_REGION_INDICES:
                valid_region |= (rm == idx)
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
            & (Y_grid >= ground_y_min) & (Y_grid <= ground_y_max)
            & (r_grid >= nadir_exclusion_radius)
        )
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
    def _interpolate(height_map: np.ndarray, n_octaves: int = 4, noise_seed: int = 0) -> np.ndarray:
        """
        Multi-scale noise inpainting for unknown (NaN) cells.

        Coarse-to-fine: at the coarsest level, known cells are diffused outward
        via 4-neighbor Laplacian diffusion to seed a structurally coherent base.
        At each finer level the previous result is upsampled bicubically, noise
        with cubically-decreasing amplitude is injected into still-unknown cells
        (front-loading large-scale structure), and diffusion propagates boundary
        values inward.  Known cells are hard constraints throughout.
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
                noise = rng.standard_normal((sh, sw)).astype(np.float32) * amplitude
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
