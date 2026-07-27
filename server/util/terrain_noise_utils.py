"""
Terrain noise infilling utilities adapted from pixels2peaks (Jain et al.).

Core ideas:
- diffuse_heightmap: iterative 4-neighbor (Laplacian) heat-equation diffusion.
  Known cells are fixed Dirichlet conditions; unknown cells are driven toward
  the local mean of their neighbours.  Produces smoother, more geologically
  plausible fills than a Gaussian kernel because it exactly solves the discrete
  Laplace equation in the limit.

- inpaint_noise_multiscale: coarse-to-fine noise infilling.  Works at
  progressively finer resolutions, injecting noise with amplitude that scales
  cubically with the downsampling level so large-scale terrain structure (ridges,
  valleys) is established first and fine detail is added last.  Known cells act
  as hard constraints at every scale.

Original repo: https://github.com/aryjain/pixels2peaks
Changes: removed fastflow / hydraulic-erosion dependency, dropped heightmap_max
constraint, converted from PyTorch to NumPy, adapted API to match this project.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom


def diffuse_heightmap(
    Z: np.ndarray,
    mask: np.ndarray,
    n_iters: int = 500,
    z_max: np.ndarray | None = None,
    seed_from: str = 'nearest',
) -> np.ndarray:
    """
    Fill unknown cells via iterative 4-neighbor Laplacian diffusion.

    Parameters
    ----------
    Z:      (H, W) float32 heightmap; unknown cells may hold any initial value.
    mask:   (H, W) bool; True = known (Dirichlet), False = unknown (to fill).
    n_iters: number of diffusion steps.  More gives a smoother fill but costs
             proportionally; 200–1000 is typical depending on gap size.
    z_max:  optional (H, W) per-cell height ceiling; unknown cells are clamped
            after each step.
    seed_from: how to initialise unknown cells before diffusion starts.
        'nearest' — each unknown cell is seeded with the height of its nearest
                    known neighbour.  Prevents cliffs at the visible-terrain
                    boundary: the fill starts at the last-seen height and the
                    diffusion then smooths it, rather than dragging every edge
                    cell up from the global mean.
        'mean'    — original behaviour: seed all unknown cells from the global
                    mean of known cells.
        'keep'    — leave unknown cells untouched; use when the caller has
                    already provided a meaningful initial state (e.g. an
                    upsampled result from a coarser level).

    Returns
    -------
    (H, W) float32 with known cells unchanged and unknown cells filled.
    """
    Z = Z.astype(np.float32, copy=True)
    if not mask.any():
        return Z

    if seed_from == 'nearest':
        from scipy.ndimage import distance_transform_edt
        _, nearest = distance_transform_edt(~mask, return_indices=True)
        Z[~mask] = Z[nearest[0][~mask], nearest[1][~mask]]
    elif seed_from == 'mean':
        Z[~mask] = float(np.mean(Z[mask]))
    # else 'keep': leave Z[~mask] as provided by the caller

    for _ in range(n_iters):
        top    = np.pad(Z, ((1, 0), (0, 0)), mode="edge")[:-1, :]
        bottom = np.pad(Z, ((0, 1), (0, 0)), mode="edge")[1:,  :]
        left   = np.pad(Z, ((0, 0), (1, 0)), mode="edge")[:,  :-1]
        right  = np.pad(Z, ((0, 0), (0, 1)), mode="edge")[:,   1:]
        avg = (top + bottom + left + right) * 0.25
        Z = np.where(mask, Z, avg)
        if z_max is not None:
            Z = np.minimum(Z, z_max)

    return Z


def inpaint_noise_multiscale(
    heightmap: np.ndarray,
    known_mask: np.ndarray,
    n_octaves: int = 6,
    diffuse_iters: int = 8,
    noise_seed: int = 0,
) -> np.ndarray:
    """
    Coarse-to-fine noise infilling for sparse heightmap data.

    At the coarsest level the known values are diffused to seed a low-frequency
    base.  At each finer level the previous result is upsampled bicubically and
    noise is injected into still-unknown cells with amplitude that scales as
    std × level³ × 0.01, so large valleys and ridges form at coarse levels and
    fine surface texture is added last.

    Parameters
    ----------
    heightmap:   (H, W) float32; unknown cells should be 0 or NaN.
    known_mask:  (H, W) bool; True = measured, False = to inpaint.
    n_octaves:   number of coarse-to-fine levels.  Each adds a 2× downsampling
                 step, so the coarsest resolution is H // 2^(n_octaves-1).
    diffuse_iters: Laplacian diffusion steps applied at each scale level.
                   Increase for larger gaps; decrease for speed.
    noise_seed:  RNG seed for reproducibility.

    Returns
    -------
    (H, W) float32; known cells are preserved exactly, unknown cells filled.
    """
    if not known_mask.any():
        return heightmap.astype(np.float32)
    if known_mask.all():
        return heightmap.astype(np.float32)

    heightmap = heightmap.astype(np.float32)
    rng = np.random.default_rng(noise_seed)
    h, w = heightmap.shape
    noise_scale = float(np.std(heightmap[known_mask]))

    def _downsample_masked(factor: int):
        sh = max(1, h // factor)
        sw = max(1, w // factor)
        Z_crop = heightmap[: sh * factor, : sw * factor].copy()
        M_crop = known_mask[: sh * factor, : sw * factor]
        Z_crop[~M_crop] = np.nan
        with np.errstate(all="ignore"):
            z_ds = np.nanmean(
                Z_crop.reshape(sh, factor, sw, factor), axis=(1, 3)
            ).astype(np.float32)
            m_ds = (
                np.nansum(
                    M_crop.reshape(sh, factor, sw, factor).astype(np.float32),
                    axis=(1, 3),
                ) > 0
            )
        return z_ds, m_ds

    ZI: np.ndarray | None = None

    for fexp in range(n_octaves - 1, -1, -1):
        factor = 2 ** fexp
        if factor > 1:
            z_ds, m_ds = _downsample_masked(factor)
        else:
            z_ds = heightmap.copy()
            m_ds = known_mask.copy()

        if ZI is None:
            # Coarsest level: seed from nearest known neighbour so diffusion
            # starts at the boundary height rather than the global mean.
            ZI = diffuse_heightmap(z_ds, m_ds, n_iters=max(diffuse_iters * 10, 200), seed_from='nearest')
        else:
            # Finer level: upsample previous result, inject scale-appropriate noise.
            amplitude = noise_scale * (fexp ** 3) * 1e-2
            ZI_up = zoom(ZI, (z_ds.shape[0] / ZI.shape[0], z_ds.shape[1] / ZI.shape[1]), order=3)
            noise = rng.standard_normal(z_ds.shape).astype(np.float32) * amplitude
            ZI = np.where(m_ds, np.nan_to_num(z_ds, nan=0.0), ZI_up + noise)
            # 'keep': preserve the upsampled+noise values set above rather than
            # overwriting them with a mean seed.
            ZI = diffuse_heightmap(ZI, m_ds, n_iters=max(diffuse_iters, 10), seed_from='keep')

    # Restore known values exactly — diffusion must not drift them.
    return np.where(known_mask, heightmap, ZI).astype(np.float32)


def ridge_chain_jaggedness_map(
    chains: list,
    H: int,
    W: int,
    grid_size_meters: float,
    window_m: float = 20.0,
    spread_sigma_m: float = 15.0,
    ref_low: float = 0.05,
    ref_high: float = 0.20,
) -> np.ndarray:
    """
    Build a (H, W) float32 [0, 1] map of local ridge-chain jaggedness, diffused
    away from each chain's silhouette. 0 = no chain coverage or a smooth profile,
    1 = maximally jagged.

    For distant mountains beyond direct depth range, the ridge silhouette's own
    up/down variation is the only available steepness signal — there's no
    per-pixel ground observation to measure a slope from, unlike terrain close
    to the camera. This is shared between TerrainNoiseRefinementStage's peak
    sharpening and TerrainReconstructionStage's cliff-strength mask, since both
    need the same "how jagged is this stretch of ridgeline" score.

    For each chain:
      1. Arc-length-parameterise the XZ path and resample Y to 1-sample/metre.
      2. Slide a window of width window_m metres along the resampled Y profile;
         at each step score local jaggedness as std(diff(Y_win)) / range(Y_win).
      3. Project each sample point (X, Z) to its heightmap pixel and splat the
         jaggedness score there weighted by 1.

    All splatted values are then Gaussian-spread by spread_sigma_m metres so
    the influence diffuses smoothly away from the ridgeline. Pixels with no
    chain coverage are 0.

    The spread is applied as an explicit distance falloff, NOT as the ratio of
    two Gaussian blurs. Dividing a blurred score by a blurred weight is the
    right way to get the *value* (a splat-density-independent weighted mean of
    nearby chain samples' scores), but the division also exactly cancels the
    blur's own decay -- so the ratio stays at roughly the local ridge score no
    matter how far from any ridge the pixel is, and the only thing that ever
    ended the influence was a `weight_blur > 1e-6` coverage test. That test is
    a numerical noise floor, not a distance: measured on one capture it held
    the score at ~0.7 across the entire half of a 200 m grid nearest the chain,
    still 0.64 at 54 m out, then dropped to 0 in a single cell where the blurred
    weight happened to cross the threshold -- a hard, roughly circular edge with
    no physical meaning. Both consumers read that as real signal: it flipped
    TerrainReconstructionStage's envelope angle from the talus 38 deg to ~69 deg
    across one cell, and TerrainNoiseRefinementStage's sharpness exponent from
    1.0 to ~2.1, each of which puts its own discontinuity into the terrain at
    exactly the same contour.

    So: keep the ratio for the value (smoothed along the ridgeline, which is
    what the blur is genuinely useful for), and multiply it by a Gaussian in
    the true distance to the nearest chain sample, which actually reaches zero.
    """
    half = grid_size_meters / 2.0
    score_acc  = np.zeros((H, W), dtype=np.float64)
    weight_acc = np.zeros((H, W), dtype=np.float64)

    for chain in chains:
        if len(chain) < 4:
            continue
        pts = np.asarray(chain, dtype=np.float64)
        X_c, Y_c, Z_c = pts[:, 0], pts[:, 1], pts[:, 2]

        dxz   = np.sqrt(np.diff(X_c) ** 2 + np.diff(Z_c) ** 2)
        arc   = np.concatenate([[0.0], np.cumsum(dxz)])
        total = arc[-1]
        if total < 1e-6:
            continue

        n_samples = max(4, int(total))
        s_uni     = np.linspace(0.0, total, n_samples)
        Y_uni     = np.interp(s_uni, arc, Y_c)
        X_uni     = np.interp(s_uni, arc, X_c)
        Z_uni     = np.interp(s_uni, arc, Z_c)

        half_w = max(2, int(window_m / 2))
        local_scores = np.empty(n_samples)
        for i in range(n_samples):
            lo_i, hi_i = max(0, i - half_w), min(n_samples, i + half_w + 1)
            Y_win = Y_uni[lo_i:hi_i]
            y_rng = Y_win.max() - Y_win.min()
            local_scores[i] = (
                float(np.std(np.diff(Y_win)) / y_rng)
                if y_rng > 1e-6 and len(Y_win) >= 3
                else 0.0
            )

        t_scores = np.clip((local_scores - ref_low) / (ref_high - ref_low), 0.0, 1.0)

        cols = ((X_uni + half) / grid_size_meters * W).clip(0, W - 1).astype(int)
        rows = ((Z_uni + half) / grid_size_meters * H).clip(0, H - 1).astype(int)
        np.add.at(score_acc,  (rows, cols), t_scores)
        np.add.at(weight_acc, (rows, cols), 1.0)

    splat_mask = weight_acc > 0
    if not splat_mask.any():
        return np.zeros((H, W), dtype=np.float32)

    spread_px   = max(1.0, spread_sigma_m / grid_size_meters * max(H, W))
    score_blur  = gaussian_filter(score_acc,  sigma=spread_px)
    weight_blur = gaussian_filter(weight_acc, sigma=spread_px)

    # Value: splat-density-independent weighted mean of nearby samples' scores.
    # Well-conditioned near the ridgeline where weight_blur is substantial; far
    # out both terms decay into numerical noise, but the falloff below is ~0
    # there anyway, so clamp rather than special-case.
    score = np.divide(
        score_blur, weight_blur,
        out=np.zeros_like(score_blur), where=weight_blur > 1e-12,
    ).clip(0.0, 1.0)

    # Falloff: true Euclidean distance to the nearest chain sample, in metres.
    # exp(-d^2 / 2 sigma^2) matches the Gaussian the spread was always meant to
    # be, and is ~0.011 by 3 sigma instead of surviving indefinitely.
    cell_m  = grid_size_meters / max(H, W)
    dist_m  = distance_transform_edt(~splat_mask) * cell_m
    falloff = np.exp(-0.5 * (dist_m / max(spread_sigma_m, 1e-6)) ** 2)

    return (score * falloff).astype(np.float32)
