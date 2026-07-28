"""
#2 spoke fix — remove radial-streak spokes from a top-down height grid.

Root cause: in the equirectangular->grid projection each panorama COLUMN maps to a
radial line, so per-column depth bias becomes a radial streak ("spoke"). These land
in the observed data and are pinned as Dirichlet BCs, so the harmonic solve can't
smooth them. 89.5% of solve nodes are fixed → the spokes are in the DATA, not the fill.

Fix: subtract only the TANGENTIAL (per-azimuth) high-frequency component. Computed as
a polar round-trip DIFFERENCE (pol - angular_smooth(pol)) so the polar-resample error
(~1.2 m mean, the source of concentric-ring artifacts) CANCELS — only the small spoke
component (~0.19 m mean) survives. Radial detail and large angular features (mountains,
which span tens of degrees) are preserved.

Validated on the reference capture (1024² downsample): removes spokes (mean 0.19 m),
Ymax preserved (53.2 vs 54.3 m). TODO: confirm at 4096², tune sigma, integrate into
TerrainReconstructionStage.run after the restore step, re-check CLIFF/texture.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def despoke(hm, grid_size_m=200.0, sigma_theta_deg=3.0, r_min_m=4.0, feather_m=3.0):
    G = hm.shape[0]
    cx = cy = G / 2.0
    nr, nth = G * 2, 2160                    # oversample polar to keep resample error small
    r = np.linspace(0, G / 2 - 1, nr)
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    R, TH = np.meshgrid(r, th)
    pol = map_coordinates(hm, [cy + R * np.sin(TH), cx + R * np.cos(TH)], order=3, mode="nearest")
    sm = gaussian_filter(pol, (sigma_theta_deg / (360.0 / nth), 0.0), mode="wrap")

    yy, xx = np.mgrid[0:G, 0:G]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    tt = np.arctan2(yy - cy, xx - cx) % (2 * np.pi)
    ri = rr / (G / 2 - 1) * (nr - 1)
    ti = tt / (2 * np.pi) * nth
    spokes = map_coordinates(pol - sm, [ti, ri], order=3, mode="grid-wrap")   # ring-free (difference)

    cell = grid_size_m / G
    w = np.clip((rr * cell - r_min_m) / max(feather_m, 1e-6), 0.0, 1.0)        # skip nadir singularity
    return (hm - spokes * w).astype(hm.dtype)


if __name__ == "__main__":
    import os
    from PIL import Image
    SCR = os.environ.get("IF_SCRATCH", "/tmp/if_scratch")
    hm = np.load(os.path.join(SCR, "recon_base.npy"))     # produced by recon_harness.py

    def shade(z, name, ds=4):
        z = z[::ds, ::ds]; gy, gx = np.gradient(z)
        slope = np.pi/2 - np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
        hs = np.clip(np.sin(np.radians(35))*np.sin(slope) + np.cos(np.radians(35))*np.cos(slope)*np.cos(np.radians(315)-asp), 0, 1)
        Image.fromarray((hs*255).astype(np.uint8)).resize((512,512), Image.NEAREST).save(os.path.join(SCR, name))

    shade(hm, "despoke_before.png")
    for sig in (2.0, 3.0, 4.0):
        out = despoke(hm, sigma_theta_deg=sig)
        shade(out, f"despoke_after_s{sig}.png")
        print(f"sigma={sig}: Ymax {out.max():.1f} (base {hm.max():.1f})  mean|Δ| {np.abs(out-hm).mean():.2f}m  p99|Δ| {np.percentile(np.abs(out-hm),99):.2f}m")
