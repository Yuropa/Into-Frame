# Fixes, round 2 (2026-10-13 afternoon)

Follows `run-review-2026-10-13.md`, which diagnosed the first run with the morning's
fixes. Everything below is in the working tree, uncommitted. Nothing has been
through a full pipeline run; "validated" says what was replayed against the cached
Rainier run with the real stage code.

## 1. Lighting

| Change | Where | Validated |
|---|---|---|
| LuxDiT runs on the photograph, not a 90° equirect strip (`estimate_from_photo`) | `lighting/lighting.py` | run on all five samples: Rainier az −165° el 40° (front-lit noon → sun high behind the camera), Shark Fin el 7° behind-right (sunset, cliffs lit from behind the camera), Paris az 70° el 44°, Iceland az 146° el 62° |
| A solar disc found in the panorama is only used when it lies inside the photographed cone (`measured_sun_must_be_photographed`); a generated disc is discarded and the env-map estimate wins | `lighting/lighting.py` | logic only |
| Sun intensity recalibrated for the photo input: scale 15 → 6, floor 0.5 when the HDR map has a peak ≥ 40× its mean, ceiling 3.0 → 2.0 | `scene/lighting.py` | shares measured: Rainier 0.25 → 1.5, Iceland 0.17 → 1.04, Shark Fin 0.08 → 0.46, Paris 0.012 → floor 0.5 |
| One sky: the Unity ambient probe is rendered from the panorama skybox the viewer sees, after it loads, instead of LuxDiT's LDR map (`EnvironmentLighting.useEstimatedEnvironmentMap` switches back) | `EnvironmentLighting.cs`, `PanoramaSkybox.cs` (load generation counter), `SceneParamManager.cs` | not compiled |
| `ambientColor` no longer feeds the directional light's colour | `SceneParamManager.cs` | not compiled |
| The terrain's panorama layer is delit with the same shading-only operation as the objects (`delight_panorama_layer`) | `terrain_texture_generation.py` | logic only |

Skybox Inpainting already places the generated sky's sun from `lighting.sun()`, so
the visible sun, the key light and the ambient probe now come from one direction.

## 2. Object painting

| Change | Where | Validated |
|---|---|---|
| Chroma-preserving delight: only the low-pass shading ratio photo/albedo is divided out; hue and chroma stay the photograph's (`util/delight.py`) | asset generation `_delight_crops`, terrain `_delight_patch`, panorama layer | replayed on crops 149/104/105/135: the red paintbrush stays red where the old blend made it pink-grey (`scratchpad/delight/sheet.png`) |
| Source crops under 384 px are Swin2SR-upscaled (≤ 2 passes) before delight, mesh and card; transparent padding filled with the cutout's mean colour first; cached as `crop_sr_{i}` | asset generation `_source_crop` | SR call not exercised |
| Cards export real normals. Every card shipped with a NORMAL accessor of zeros (both windings cancelled in trimesh's derived normals), so the Lit shader had no diffuse term: cards rendered flat and dark. Now bent normals, up + 0.5 × outward | `grass_cover/cards.py` | exported GLB reloaded: normals (±0.45, 0.89, 0) |
| Card textures and mesh sources must have a plausible box shape for the class (`_MAX_WIDTH_OVER_HEIGHT`): a 215×48 "tree" no longer textures a 5 m card; a group with no plausible crop gets no card | asset generation `_curate`, `_build_card_meshes` | logic only |

## 3. Ground texture

| Change | Where | Validated |
|---|---|---|
| Reference patches are cut only within `reference_max_distance_m` (8 m) of the camera, where a panorama pixel still covers centimetres; falls back to the whole region when a type has nothing that close | `_extract_reference_patches` | logic only |
| Pattern-texture bake off by default (`use_pattern_texture: false`): it was a 2048 px world-space bake of the projected panorama and inherited the smear. The FLUX seamless-tile path is used, at `synthetic_tile_factor` | `config.yaml` | |
| Panorama layer withheld where one pixel would cover more than `panorama_max_texel_m` (0.6 m) of ground, from distance and grazing angle against the height map's surface | `_panorama_visibility_weight` | replayed on the Rainier height map: mean weight 20–35 m 0.67 → 0.30, 35–60 m 0.53 → 0.22, the mountain ring 60–142 m 0.58 → 0.45 (steep faces keep their photograph) |
| Steep-cell → TERRAIN reclassification only on cells with a measured height (`terrain_reclassify_observed_only`) | `region_map_refinement.py` | logic only; was 9.5 M of 16.7 M cells |

## 4. Object fill

| Change | Where | Validated |
|---|---|---|
| `min_exemplars` 8 → 4; when any region group of a type is short, all of that type's groups pool into one distribution painted across the union of their paintable regions (`pool_across_regions`) | `distribution_synthesis.py` | logic only; on the last run tree 6 + 7 + 56 would pool |
| `forest` excluded from scattering (`exclude_types`) | `config.yaml` | |
| New stage **Treeline Backdrop**: upright far VEGETATION runs (depth nearly constant foot-to-crown, unlike a meadow floor) become a ribbon of vertical quads at their measured depth, textured with the panorama strip and the canopy silhouette as alpha; placed as environment mesh `treeline` | `terrain/treeline_backdrop.py`, registry, `config.yaml`, scene generation, `SceneObjectManager.cs` | replayed on Rainier: 140 of 256 azimuth bins, 6.8–48 m, 140 quads, 4096×362 strip; overlay in `scratchpad/treeline/out/treeline_overlay.png` covers the band at the mountain's base. Note the scene's depth puts that band at 10–12 m, which is the depth calibration's doing, not this stage's |

## 5. Runtime LODs

| Change | Where | Validated |
|---|---|---|
| Every placed mesh/card instance carries `lods`: reconstruction (if it passes the shape gates), crossed card, camera-facing billboard, each with a LOCAL scale/offset under the placed object and a screen-height threshold (`lod_*_screen_fraction`: 0.12 / 0.025 / 0.003). Grass exempt | `scene/object.py`, `scene_generation/generation.py`, `config.yaml` | encode/decode round-trips; all 39 stages build from `config.yaml` |
| Unity builds a `LODGroup` with crossfade from the levels, loading each through the asset queue and assembling after the loads; a level whose asset failed is dropped from the group | `SceneObjectManager.cs`, `SceneClient.cs` | not compiled |
| Sway drives every rigged level, not just the first bone chain found | `WindSway.cs` | not compiled |
| Rigging and asset export include LOD meshes/textures | `category_mesh_rigging.py`, `server/server.py` | compiles |
| LOD crossfade enabled in both URP assets | `Settings/Project Configuration/*.asset` | |

## 6. Placement FOV

`PLACEMENT_FOV_SCALE` is now 1.0. Every panorama will be regenerated on the next
run (WorldGen embeds the photo at this width), objects inside the cone come out at
their true angular size, and the cone's resolution gain roughly doubles. Thresholds
tuned under 2.0 should be re-read against the first run: `captured_fov_margin_deg`,
`min_gain` in hi-res cropping, and the counts in `fixes-2026-10-13.md`.

## Verify on the next run

- Lighting log: "estimating from the photograph", the disc decision, the sun's
  az/el and intensity; Skybox Inpainting's "Sun position from the scene's own key
  light" should agree.
- Asset generation: "super-resolved … px" lines; delit crops keep their colour.
- Texture: "reference patches … restricted to … px within 8 m"; the tiles in
  `Terrain Texture Generation/terrain_material.splat` should be flat seamless
  material, not starbursts; `Terrain regions` should list `ground` again.
- Synthesis: "pooled … exemplar(s)" lines and the painted counts for trees/plants.
- Treeline: bins/depth line; the band should sit on the terrain at the mountain's
  base.
- Scene: "Runtime LODs: … instance(s) with 3 level(s)"; in Unity, walk toward a card.
- Unity console: `[EnvironmentLighting] Ambient probe rendered from the panorama
  skybox`.
