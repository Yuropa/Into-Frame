# Weekly Change Tracking

## Current Baseline

| Field | Value |
|-------|-------|
| Commit | `676b00234e31cf048910036c08f768ab6c790eba` |
| Message | skybox updates |
| Date set | 2026-07-02 |

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
