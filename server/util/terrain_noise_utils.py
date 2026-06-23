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
from scipy.ndimage import zoom


def diffuse_heightmap(
    Z: np.ndarray,
    mask: np.ndarray,
    n_iters: int = 500,
    z_max: np.ndarray | None = None,
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

    Returns
    -------
    (H, W) float32 with known cells unchanged and unknown cells filled.
    """
    Z = Z.astype(np.float32, copy=True)
    if not mask.any():
        return Z

    # Seed unknown cells with the mean of known cells so diffusion starts close.
    Z[~mask] = float(np.mean(Z[mask]))

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
            # Coarsest level: seed with diffusion from known values.
            ZI = diffuse_heightmap(z_ds, m_ds, n_iters=max(diffuse_iters * 10, 200))
        else:
            # Finer level: upsample previous result, inject scale-appropriate noise.
            amplitude = noise_scale * (fexp ** 3) * 1e-2
            ZI_up = zoom(ZI, (z_ds.shape[0] / ZI.shape[0], z_ds.shape[1] / ZI.shape[1]), order=3)
            noise = rng.standard_normal(z_ds.shape).astype(np.float32) * amplitude
            ZI = np.where(m_ds, np.nan_to_num(z_ds, nan=0.0), ZI_up + noise)
            ZI = diffuse_heightmap(ZI, m_ds, n_iters=max(diffuse_iters, 10))

    # Restore known values exactly — diffusion must not drift them.
    return np.where(known_mask, heightmap, ZI).astype(np.float32)
