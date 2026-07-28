# Pano → Terrain: work-in-progress handoff

Goal: make the panorama→terrain reconstruction "work well." Three issues raised on the
`Mount` capture (debug context `0bc34849-…`), plus a side quest on object meshing.

## The reference debug context

Everything below is validated against one saved pipeline debug dump:

```
/Users/Josh/Desktop/Mount.debug 2/context/0bc34849-eb4d-e372-bc30-97d7d11064f0
```

On another machine, copy that folder (or regenerate one by running the pipeline with
debug output) and point `IF_CONTEXT` at it. The harnesses read the saved stage outputs
(`.npy`/`.json`) from inside it.

## Environment (why a venv)

The conda `frame`/`stablepoint` envs have a **broken scipy** on this Mac (dlopen of
`scipy/sparse/linalg/_propack/_spropack…so` fails — a zero-fill section offset error,
i.e. old wheel vs new macOS). The reconstruction needs `scipy.sparse.linalg.spsolve`,
which is exactly the broken part. So the offline work uses an isolated venv with modern
wheels — **do not touch the conda envs**:

```bash
python3 -m venv vtest
./vtest/bin/python -m pip install "numpy>=2" "scipy>=1.14" pillow landlab   # landlab only for noise_harness
export IF_SERVER=/path/to/Into-Frame/server
export IF_CONTEXT="/path/to/Mount.debug 2/context/0bc34849-…"
export IF_SCRATCH=/tmp/if_scratch
```

## Harnesses (the enabling asset — in this folder)

Both run the **real** stage code (no re-implementation) on the saved context; only
framework plumbing is stubbed. They reproduce the saved outputs almost exactly.

- `heightmap_harness.py` — drives `HeightMapGenerator.generate`. Reproduces the saved
  Height Map to **max|Δ|=0.000**. `run(**overrides)` returns the 9-tuple; ~75 s/run.
- `recon_harness.py` — drives the real `TerrainReconstructionStage.run` (stubs torch
  type-hints, the `PipelineStage`/`PipelineContext` bases, and `landlab.RasterModelGrid`
  — the last is just a C-order raster with 4-neighbours). Reproduces the saved
  reconstruction to **HEIGHT_MAP max|Δ|=0.03 m, CLIFF_MASK exact**. ~200 s/run.
  `run(hm_override=…, **cfg_override)`; `hm_override` feeds a modified height map.
- `noise_harness.py` — drives the real `TerrainNoiseRefinementStage.run` with **real
  landlab** (`pip install landlab` — installs cleanly). Reproduces the saved final
  terrain's roughness profile; absolute mean|Δ|≈0.34 m is a landlab-RNG floor, so it's
  for statistics/visuals, not bit-exactness. ~54 s/run. `run(hm_override=…, **cfg)`.
- `despoke.py` — the #2 fix as a standalone function + self-test.

Chaining: `heightmap_harness.run()` → `recon_harness.run(hm_override=…)` →
`noise_harness.run(hm_override=…)` gives the full Height-Map→Reconstruction→Noise
path offline. The Terrain **Mesh** stage is not harnessed.

---

## Status of the four issues

### ✅ #3 — texturing "same colour as the mountain top" — DONE (`server/util/panorama_utils.py`)
40–57 % of mid-mountain vertices (r=45–80 m) project **above** the panorama's real
skyline into sky, so `mesh_uvs` squeezed them into a 24 px band just below the ridge →
all sampling the same crest colour. Fix: widened that band to 120 px, capped at the
horizon (never into foreground grass). Validated: overshoot vertices now span ~117 px
of real texture vs 23 px (5× less repetition); a top-down textured render shows less
smear. Honest limit: there is *no* real texture above the skyline, so this is a
best-effort compromise — fully correct would require geometry that doesn't overshoot.

### ✅ #2 — radial spokes — IMPLEMENTED & VALIDATED end-to-end (`server/pipeline/terrain/terrain_reconstruction.py`, `config.yaml`)
**Root cause (measured):** the equirect→grid projection maps each panorama *column*
onto a radial line, so per-column depth bias becomes a radial streak. These land in the
observed data and are pinned as Dirichlet BCs — **~89 % of solve nodes are fixed**, so
the harmonic solve can't remove them. The spokes are in the DATA, not the fill.

**Fix:** new `_despoke` step (runs last in `run()`, after the observed-data restore).
Subtracts only the **tangential high-frequency** component, computed as a polar
round-trip *difference* (`pol − angular_blur(pol)`) so the polar-resample error cancels
— that difference is what avoids the concentric-ring artifact a naive round-trip bakes
in (resample error ~1.2 m; spoke signal ~0.2 m). Config: `despoke_angular_sigma_deg`
(default 3.0, 0 disables), `despoke_min_radius_m`, `despoke_feather_m`.

**Validated end-to-end** through the real reconstruction (recon_harness):
- `despoke_angular_sigma_deg=0` reproduces saved (max|Δ|=0.031, unchanged) → clean no-op.
- `despoke=3` (default): radial profile identical (r45/60/80 = 18.1/30.7/21.8 vs
  18.1/30.7/21.9 saved), **CLIFF_MASK max|Δ|=0.000**, Ymax 53.4 vs 54.4, mean|Δ| 0.19 m
  concentrated in the spoke regions. Hillshade `val_despoked.png`: spokes gone, mountains
  intact, no rings.

**Remaining:** σ=3° is a good default (2–4° all work); confirm the downstream Noise
Refinement / Mesh stages don't reintroduce spokes (they run after this).

### ✅ #1 — "plateau" (broad smoothness) — IMPLEMENTED & VALIDATED; one residual (`terrain_noise_refinement.py`, `config.yaml`)
**Root cause (measured with the noise harness + real landlab):** every synthetic pass
that adds fine detail is later stripped from unobserved cells — Pass 3 diffusion
smooths it, and Pass 5 hydro erosion **downsamples to `hydro_resolution` (256) and back**,
destroying anything finer. Observed cells recover via the final restore; unobserved
terrain (reconstructed slopes, and the nadir most of all) is left as the smooth erosion
output — a synthetic-looking, over-smooth surface (the broad "plateau").

**Fix:** new **Pass 6** in `run()` — re-inject meso-scale fBm *after* every smoothing
pass (so nothing downstream dilutes it), scaled by `(1 - protect)` so it lands only on
unobserved/untrusted cells and fades to nothing into real data. Config
`unobserved_texture_amplitude` (default 0.6 m, 0 disables). This is deliberately NOT the
same as the `noise_amplitude` bump the config notes were reverted: that raised the
global Pass-2 amplitude (before diffusion/erosion) and read as spiky noise under the
camera; Pass 6 is gated, meso-scale (`noise_scale=80` → rolling), and post-erosion.

**Validated** (noise harness, production settings): amp=0 reproduces baseline; amp=0.6
gives natural rolling-hill texture on previously-smooth slopes, **mountains preserved**
(Ymax 51.8 unchanged, r60-80 std 12.6 unchanged), no spiky nadir. Full-terrain hillshade
looks organic vs the smooth baseline.

### ✅ #1b — central PLATEAU / hill (the real one) — FIXED (`terrain_noise_refinement._hydro_erode`)
The big one, found after the review feedback. The reconstruction gives a flat valley
floor at −1.9 m under the camera — correct — but the **Noise Refinement output came back
at +11 m there**: a 13 m hill/plateau. Bisected to the **hydro erosion** pass:
`SinkFillerBarnes` fills closed depressions to route flow *and permanently raises the
terrain*. The camera's valley is a closed basin — a panorama reconstructs a 360° mountain
ring around the viewer, so the basin has no grid-edge outlet — so the filler floods the
whole valley up to its spill saddle (+11 m). That IS the "plateau in the centre."

Fix: erosion may only **incise, never raise** — `return np.minimum(terrain, eroded)`. The
sink-fill still does its job (routing the flow that drives incision), but the raised fill
is discarded. Validated: centre back to −1.7 m (flat valley floor), mountains preserved
(Ymax 52.9), channel incision preserved (Ymin −3.4). Root cause was NOT the nadir prior
or `camera_height` — the near-nadir *is* hallucinated (confirmed: the panorama's bottom
third is invented gravel/snow, the forward-facing hero photo never saw straight down),
but that turned out to be a side issue; the plateau was hydrological.

**Residual (documented, minor):** a *faint* circle at the nadir-exclusion boundary
(r ≈ 4.5 m) remains — the Height-Map flat-prior disc. Small now that the +11 m plateau is
gone. If it still reads in the app, it's a Height-Map change (flat-ground prior + soften
the certainty rim); the near-nadir being hallucinated means a flat-ground-at-camera-height
prior is the honest choice there. `camera_height` is 1.0 in config but the image-derived
ground sits ~1.9 m below the camera — worth deriving it from the observed foreground
rather than hardcoding, which would also inform the "valleys too deep" question.

User review of the current result (2026): "better but not perfect" — two specific asks
still open:
  1. **No plateau/hill in the centre.** The flat-prior nadir disc still reads as a
     coherent bump/crater under the camera. Fix in the Height Map: don't pin the disc as
     a depression relative to its ring, and soften the hard circular certainty rim at
     `prior_radius` so there's no clean-circle signature; or fold the disc into the
     normal interpolation instead of a special flat prior.
  2. **Valleys too deep.** The reconstructed valleys (between the near ground and the
     mountain ring, and the mid-field lows) come back deeper than expected. Candidates:
     the ridge/slope **envelope** pushing the surrounding floor down by contrast, the
     harmonic solve overshooting between the low near-field and the high ridge anchors,
     or hydro-erosion incision (`hydro_erodibility`/`hydro_dt`/`hydro_n_steps`) carving
     too aggressively. Measure valley depth vs the point-cloud in the recon harness
     before/after, and check the envelope + hydro passes specifically.

### ✅ Side quest — too many tiny "inty" meshes — DONE (`panorama_asset_generation/generation.py`)
Meshification was gated on distance only; 7 flower buckets 0.6–1 m away (0.01–0.07 % of
frame, one a conf-0.03 miscrop) each got a bespoke mesh. Added `min_mesh_area_fraction`
(default 0.001) — below it a group stays billboard-only (pool still curated, nothing
dropped). Verified: all 7 → billboards, no legit object lost. **Committed** in
`44b8716 "Terrain Cleanup"`.

---

## Git state at handoff
- `44b8716 "Terrain Cleanup"` (already committed, not by me) contains: the #3 texture
  fix precursor?, Part B (flower size gate), **and** the inert #1 nadir change.
- Working tree: **removes** the inert nadir change, and adds the #3 texture fix +
  the #2 despoke. Review `git diff` and commit. The `docs/terrain_dev/` folder (this
  handoff + harnesses) is untracked — commit or keep local as you prefer.

## The LoRA question (answered, no change)
The Height Map consumes `panorama_depth` = DAP run on `panorama_terrain` (inpainted +
LoRA-corrected). Calibration *fits* its curve against `panorama_object_depth` (the
original, un-LoRA'd panorama — needs genuine photo pixels to match the DA3 hero depth)
and *applies* it to both. So: the depth feeding terrain is the LoRA-corrected plate;
only the calibration anchor is the original. Intentional and correct.
