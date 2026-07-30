# Weekly Change Tracking

## Current Baseline

| Field | Value |
|-------|-------|
| Commit | `ef9d39cd0fa54594412f711f27b6b993c98dbfe7` |
| Message | Bones not loading |
| Date set | 2026-07-30 |

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

### Week ending 2026-07-30

*Baseline `a6d4f68` (2026-07-17) → `ef9d39c` (2026-07-30). 111 commits, 118 files changed (+12.9k / -1.5k lines).*

*Note: the recorded baseline hash `ecdd6f7` no longer exists — the 2026-07-17 history rewrite (see `project-github-identity`) rewrote it to `a6d4f68`, same "Fix crash" commit and date. Diffed against that.*

- **Scene animation (new, end-to-end)**: four new stages turn the generated video into actual motion in the headset. `ObjectMotionClassificationStage` classifies each extracted object clip as *stationary* (oscillating in place — wind, ripples) or *moving* (a rigid body translating through frame) from its own 2D centroid trajectory; `CategoryMeshRiggingStage` bakes a 3-bone vertical sway skeleton into each shared category mesh; `SceneAnimationStage` annotates already-placed objects with sway params, animated-billboard video, or a rigid-body physics handoff (placement itself unchanged). New `util/gltf_skin.py` hand-injects glTF `skins`/`JOINTS_0`/`WEIGHTS_0` into the GLB after export, since trimesh has no notion of skinning.
- **Unity animation client (new)**: `WindSway.cs` drives the baked bone chain per-instance (procedural, so many differently-phased instances share one rigged asset) with a cantilever amplitude falloff; `PhysicsHandoff.cs` applies a measured initial velocity once at spawn and hands off to Unity physics, keeping gravity on only when the clip's own vertical acceleration looked like free-fall; `AnimatedBillboardVideo.cs` + `AlphaVideoComposite.shader` play the color and grayscale-matte clips and composite them to RGBA per frame into the existing billboard material; `SwayDiagnostics.cs` walks the five ways the sway chain can silently fail and reports which link is broken.
- **Animated skybox (new)**: `PanoramaSkyboxSpin.shader` rotates the equirect sky about an *arbitrary* axis instead of Unity's yaw-only `_Rotation`, so the sky can be spun about the sun direction — the sun stays put and lighting stays consistent with the directional light extracted from that same panorama. Falls back to the built-in `Skybox/Panoramic` (static sky, no magenta dome) when the shader is stripped from the build. Deliberately skips per-frame `DynamicGI.UpdateEnvironment`.
- **Grass cover (new stage)**: `grass_cover/` (`grass_cover.py`, `grass_area.py`, `cards.py`) scatters ground cover from the **original** panorama's region typing, because foreground inpainting erases the near-field meadow — measured on the Mount Rainier capture, the removal mask covered 48.5% of the panorama and ~95% of the vegetation evidence, and the erased meadow came back as bare gravel. Owns the `grass_tuft` class outright, runs after Distribution Synthesis, budgeted at 8k instances / 25 m radius. Far LOD uses crossed-card meshes rather than billboards (a camera-facing quad collapses to a line when viewed from above); near meshes decimated 316k → 4k faces (8.2 MB → 100 KB GLB, p95 error 0.19% of bbox diagonal).
- **Panorama LoRA correction (new stage)**: ObjectClear has no equirectangular awareness, so the content it fabricates reads as a flat photo pasted into equirect space and the panorama-specialized depth model then runs out-of-distribution exactly there. A partial-strength (0.35) FLUX.1 + panorama-LoRA img2img pass re-touches just that region. `panorama_depth_patch/` is the validated-against alternative, kept as an experimental stage: discard the fabricated region's depth entirely and Laplacian-diffuse real surrounding depth into it.
- **Second region-typing pass**: `Panorama Regions (Terrain)` runs the same model again over the object-removed + LoRA-corrected panorama, with its own key set, so a removed tree can't punch a hole in the ridgeline or get anchored as a fake crest. Its WATER calls are cross-checked against the original-photo pass — inpainting may reveal ground, never create water — which fixes a lake surface forming under the viewer, half a metre above its own shoreline.
- **Terrain reconstruction fixes (three validated)**: erosion may now only incise, never raise (`np.minimum`) — `SinkFillerBarnes` was flooding the camera's own closed basin to its spill level, turning a −1.9 m valley floor into a +11 m plateau; a new `_despoke` step removes the radial spokes that per-column depth bias pinned as Dirichlet BCs (~89% of nodes fixed); a new Pass 6 re-injects meso-scale fBm after every smoothing pass, gated by `(1 - protect)` so it lands only on unobserved terrain. Mountain texture reference band widened 24px → 120px (~5× less color repetition).
- **Height map**: far-range depth compression replaces the uniform rescale — scaling off the single farthest non-sky pixel produced a ~7.2× global shrink that put the ground 0.9 m *above* eye level, buried the camera in its own terrain, and turned the baked ground texture into a radial pinwheel. Plus NaN-aware label smoothing for flood-fill connectivity and a `far_exclusion_radius` mirror of the nadir exclusion.
- **Terrain texturing consolidated**: `terrain_texture_bake.py` and `terrain_texture_refinement.py` deleted, their work folded into `terrain_texture_generation.py` / `pattern_texture.py`; new `util/gltf_uv2.py` injects a real standards-named `TEXCOORD_1` (trimesh only writes one UV set) for `TerrainSplat.shader`'s panorama layer.
- **Cross-run GLB fidelity fix** (`util/gltf_attachments.py`): a trimesh round-trip silently dropped the skin and TEXCOORD_1, so on any *resumed* run the client got bone-less meshes (every plant frozen) and terrain sampling a single texel — invisible to every check that inspected the artifact on disk rather than what the client received.
- **Object pipeline**: new `ObjectInstanceRefinementStage` (452 lines) and `object_clustering/` (`clustering.py`, `dinov2_embedder.py`) replacing the old correlation-only path; segmentation `min_area_fraction` raised 0.0003 → 0.0015 after 898 raw crops on one capture turned out to be overwhelmingly individual grass blades; `object_margin_threshold` re-derived (0.9) now that CLIP certainty is temperature-scaled; `util/crop_scoring.py` extracted from asset generation.
- **Distribution synthesis**: failure diagnosis — `synthesize_cli` exits 0 with empty output for all four of its bail-outs, so "painted 0 instances" was indistinguishable from "the data said so"; stderr is now passed through verbatim. Exemplar candidate grid now sized to the tile's own density, plus a per-group instance ceiling (a GameObject budget, not a triangle budget).
- **Perf/infra**: new `util/gpu_task_pool.py` runs GPU tasks one-worker-per-device with an OOM-tolerant serial retry; `util/json_utils.py`, `util/device_utils.py` additions; `scripts/remote-server.sh` and `setup.sh` updated; `config.yaml` +569 lines (heavily annotated with the measurements behind each threshold).
- **Debug tooling**: `docs/terrain_dev/` — a HANDOFF doc plus three offline harnesses (`heightmap_harness.py`, `recon_harness.py`, `noise_harness.py`) that drive the *real* stage code against a saved debug context, reproducing saved outputs to max|Δ| = 0.000 / 0.03 m, plus `despoke.py`. New `scripts/scene-summary.py` (classifies every placed object's animation state), `scripts/check-rigged.py` (which category meshes actually carry a sway skeleton), `scripts/debug-synthesize-tile.py` (replay one synthesis tile in isolation).
- **Known open issue**: rigging `grass_tuft` makes ~6,700 skinned meshes and pins the frame rate to single digits — grass sway belongs in a vertex shader; commenting it out of `rig_categories` is the documented shipping state until then.

### Week ending 2026-07-17

*Baseline `3164f3c` (2026-07-09) → `ecdd6f7` (2026-07-17). 84 commits, 67 files changed (+6.5k / -1.1k lines).*

- **Video generation (new)**: new `video_generation` pipeline stage wrapping LTX-2 (`ltx2_client.py`/`ltx2_client_imp.py`) as a remote process to animate a still scene into a short clip — a fixed motion prompt/negative-prompt pair constrains it to a locked-tripod shot with only ambient motion (leaves, water, light), explicitly suppressing cloud drift and lighting time-lapse artifacts seen in early tests. Resolution is rounded to LTX-2's /64 requirement; fp8-cast quantization added so the ~22B checkpoint fits a 32GB card. Several follow-up commits on frame count, hardlinking output, and generation defaults.
- **Video segmentation (new)**: new `video_object_extraction.py`, `video_segmentation.py`/`_imp.py`/`_result.py` to track and extract objects across the generated video's frames; directory-storage bug fixed shortly after.
- **Terrain texturing**: major rework of `terrain_texture_generation.py` (~900 lines touched) and new `pattern_texture.py` (375 lines, citing Neyret & Cani's "Pattern-Based Texturing Revisited", SIGGRAPH '99) adding Wang-tile/aperiodic pattern-based texture synthesis; texture angle and blend-strength tuning, cliff/interior-peak cleanup.
- **Terrain generation**: `heightmap_generator.py` heavily reworked (665 lines), plus `terrain_generator.py`/`terrain_reconstruction.py`/`terrain_noise_refinement.py` updates — point-cloud-based reconstruction, mesh mountain features, height-map bug fixes.
- **Panorama depth & seams**: `panorama_depth/calibration.py` reworked (247 lines) across several calibration fixes; `util/seam_repair.py` seam-rolling and skybox-seam-boundary fixes; new `util/panorama_projection.py` and `util/panorama_tiling.py` helpers for projection cleanup and segmentation tiling.
- **Foreground/skybox inpainting**: `panorama_foreground_inpainting/generation.py` (292 lines) and `skybox_inpainting.py` continued fixes; region-map generation and certainty-map calculation updated (`region_map.py`, `region_map_generator.py`).
- **Object pipeline**: `object_detection.py`/`grounding_dino_imp.py` and `object_correlation.py` updated; `distribution_synthesis.py` sped up and bug-fixed; object-placement bugs fixed; Unity billboard cropping improved with a new `Billboard.mat` material.
- **New util**: `util/instance_merge.py` (143 lines, new) for merging detected instances.
- **Infra**: `config.yaml` substantially reorganized (344 lines) plus new config files for foreground-inpainting test, pano video generation, and video generation; `scripts/setup.sh` updated for the video stage; `pipeline_context.py`/`context_value.py` reworked (151 lines); remote connection types tweaked; scene-generation config enabled/tuned; general crash fix.

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
