# Weekly Change Tracking

## Current Baseline

| Field | Value |
|-------|-------|
| Commit | `3164f3ce285ccf4332008b6b6756def2849ac213` |
| Message | Merge branch 'object-clear' |
| Date set | 2026-07-09 |

---

## How to Get a Summary

Ask Claude Code:
> "Summarize all changes since the baseline in WEEKLY_CHANGES.md"

Or manually:
```sh
git log 08ac79ab87b2d85a25c5f482e8a1d5c416b443f2..HEAD --oneline
git diff 08ac79ab87b2d85a25c5f482e8a1d5c416b443f2..HEAD --stat
```

After reviewing, update the baseline to current HEAD:
```sh
git rev-parse HEAD  # copy this hash into the Commit field above
```

---

## Change Log

### Week ending 2026-07-09

*Baseline `676b002` (2026-07-01) → `3164f3c` (2026-07-09). 71 commits, 65 files changed (+4.8k / -819 lines).*

- **Intrinsic images (new)**: new `intrinsic_images` pipeline stage bridging to IntrinsicDiffusion (Luo et al., SIGGRAPH 2024) in its own conda env, predicting per-pixel albedo and surface-normal maps from a single image — merged in via the `IntrinsicDiffusion` branch. New `intrinsic_utils.py` helper.
- **Object removal (new)**: new `inpainting_objectclear` stage wrapping the ObjectClear model (jixin0101/ObjectClear) as a remote conda process; object clearing re-enabled in the pipeline after being toggled off/on across several commits.
- **Panorama depth & seam repair (new)**: new `panorama_depth/calibration.py` stage for calibrating panorama depth, and a new `util/seam_repair.py` module (`heal_seam`) doing inpaint-based seam healing with feathering/band-width controls — several follow-up commits fixing the skybox seam and widening the repair band.
- **Foreground inpainting (new)**: new `panorama_foreground_inpainting` stage combining SAM segmentation, depth-based object filtering, and inpainting to remove foreground objects from panoramas before sky/scene generation.
- **Distribution synthesis (new)**: new `distribution_synthesis` module (~390 lines) for synthesizing/painting learned object distributions onto the top-down region map.
- **Terrain**: large rework of `terrain_texture_generation.py` (+1146/-lines) — texture supersampling, ridge-line and cliff improvements, height-map/relief tuning, color-map updates, point-cloud-based terrain reconstruction, and a second full pass ("terrain generation take 2").
- **Skybox/panorama**: continued LoRA weight iteration (new weights, sweeps, debug, then disabled again), flood-fill and warp fixes, sky-mask adjustments, and a pipeline inpainting cleanup/switch-back.
- **Scene generation**: `scene_generation/generation.py` and new `projection.py` reworked for terrain/object placement; UV-flip fix; Unity `SceneClient`/`SceneObjectManager`/`SceneParamManager` and `TerrainSplat` shader updated to match.
- **Captioning**: switched to Florence captioning with follow-up fixes.
- **Infra**: new setup script and several build fixes, `pattern-synthesis` CMake/CLI updates, Caffe dependency removed, remote/mirror download and seed-value handling improved, API and server cleanup.

### Week ending 2026-07-02

*Baseline `08ac79ab` (2026-06-10) → `676b002` (2026-07-01). 238 commits, 155 files changed (+35.4k / -870 lines). Note: prior baseline pointed to a commit from 2026-06-10, so this entry actually covers ~3 weeks, not one.*

- **Terrain generation**: switched core solver to LandLab; added ridge/mountain extraction, height-map tuning, terrain reconstruction, noise refinement, and texture generation/baking/refinement; reduced terrain spikiness. New Unity `TerrainSplat` shader/material + `TerrainMaterialManager` to render splat textures.
- **Skybox / panorama**: new sky algorithm (multiple passes), panorama segmentation, panorama object classification, region-map generation with certainty maps, prefilled skybox, improved inpainting.
- **Object pipeline**: new object detection (Grounding DINO), object correlation, object distribution/histograms, CLIP-based object typing, object placement with size filtering. Tree generation added then disabled.
- **Unity client**: `SceneClient`, `SceneObjectManager`, `SceneParamManager` added/extended; billboard direction fix; camera height changes; Unity components set up for the scene.
- **Reporting**: new PDF report generator and pipeline-diagram doc.
- **Infra**: new `pattern-synthesis` C++ library (Voronoi PCF, Lloyd relaxation, pybind11 bindings); requirements split into per-model files under `requirements/`; caching/logging/setup-script fixes; merged feature branches `terrain-sketching`, `scene-generation-imporvements`, `object-histograms`, `tree-generator`, `terrian-improvements`.

### Week ending 2026-06-29 (baseline set)

*Baseline established at `08ac79ab` — "Archive the scene" (2026-06-10).*
