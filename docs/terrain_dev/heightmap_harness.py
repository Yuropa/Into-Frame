"""
Offline harness for HeightMapGenerator.generate — runs the REAL Height Map stage
math on a saved debug context, no torch/GPU/pipeline needed.

Reproduces the saved Height Map output to max|Δ|=0.000 on the reference capture.

Setup:
    python3 -m venv vtest && ./vtest/bin/python -m pip install "numpy>=2" "scipy>=1.14" pillow
    (the conda `frame`/`stablepoint` scipy is broken on this Mac — dlopen _spropack)

Paths (override via env):
    IF_SERVER   = <repo>/server
    IF_CONTEXT  = the debug context dir (…/Mount.debug 2/context/<uuid>)
    IF_SCRATCH  = a writable scratch dir for sky_mask.npy + outputs
"""
import os, sys, numpy as np, time

SERVER  = os.environ.get("IF_SERVER",  os.path.join(os.path.dirname(__file__), "..", "..", "server"))
D       = os.environ.get("IF_CONTEXT", "/Users/Josh/Desktop/Mount.debug 2/context/0bc34849-eb4d-e372-bc30-97d7d11064f0")
SCR     = os.environ.get("IF_SCRATCH", "/tmp/if_scratch")
os.makedirs(SCR, exist_ok=True)
sys.path.insert(0, os.path.abspath(SERVER))

from util.depth_utils import Depth
from pipeline.heightmap.heightmap_generator import HeightMapGenerator

# sky mask is stored as a giant JSON in the context; cache it as .npy once.
_sky_npy = os.path.join(SCR, "sky_mask.npy")
if not os.path.exists(_sky_npy):
    import json
    np.save(_sky_npy, np.array(json.load(open(f"{D}/Panorama Depth/panorama_sky_mask.json")), dtype=bool))

depth = Depth(np.load(f"{D}/Panorama Depth Calibration/panorama_depth.npy"))
sky   = np.load(_sky_npy)
rt    = np.round(np.load(f"{D}/Panorama Regions (Terrain)/panorama_region_type_map_terrain.npy")).astype(np.float32)
amb   = np.load(f"{D}/Panorama Regions (Terrain)/panorama_region_ambiguous_mask_terrain.npy").astype(bool)

# These are the config.yaml "Height Map" values for the reference capture.
CFG = dict(
    grid_size_meters=200.0, grid_resolution=4096, ground_y_max=-0.5,
    use_equirectangular=True, smooth_sigma=16.0, camera_height_meters=1.0,
    flood_fill=True, flood_fill_max_step=1.5, label_smooth_sigma=1.5,
    nadir_exclusion_radius=1.5, nadir_ramp_width=3.0, far_exclusion_radius=45.0,
    certainty_falloff_meters=20.0, elevation_distortion_power=1.0,
    min_forward_samples=4, fill_boundary_falloff_cells=6.0,
    min_component_area_fraction=0.001, despike_threshold_m=0.3, despike_window=5,
    despike_reference_distance_m=10.0, despike_dense_threshold_scale=2.5,
    despike_dense_min_real_support=0.6, region_closing_iterations=2,
    single_sample_blur_sigma=1.5, reclaimed_certainty_factor=0.5,
)

def run(**overrides):
    cfg = dict(CFG); cfg.update(overrides)
    t = time.time()
    out = HeightMapGenerator.generate(
        depth=depth, intrinsics=None, sky_mask=sky, panorama_depth=None,
        region_type_mask=rt, region_ambiguous_mask=amb, debug_dir=None, **cfg)
    print(f"  ran in {time.time()-t:.1f}s  Y {np.nanmin(out[0]):.1f}..{np.nanmax(out[0]):.1f}")
    return out  # (height, certainty, cell_relief, cell_slope, true_observed, component_id, pano_u, pano_v, real_sample)

def profile(hm, label=""):
    G = hm.shape[0]; cc = (np.arange(G) - G/2 + 0.5) * (200.0/G)
    X, Z = np.meshgrid(cc, cc); r = np.sqrt(X**2 + Z**2)
    print(f"  [{label}] radial p50:", " ".join(
        f"{a}:{np.median(hm[(r>=a)&(r<b)]):.1f}" for a, b in [(0,10),(10,20),(20,30),(30,45),(45,60),(60,80),(80,100)]))

if __name__ == "__main__":
    out = run(); hm = np.nan_to_num(out[0]); profile(hm, "baseline")
    saved = np.nan_to_num(np.load(f"{D}/Height Map/height_map.npy"))
    d = np.abs(hm - saved)
    print(f"  vs saved: max|Δ|={d.max():.3f} match={'YES' if d.max()<0.05 else 'NO'}")
    np.save(os.path.join(SCR, "base_hm.npy"), hm)
