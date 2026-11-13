# Hybrid scene design, end to end

Branch `experiment/3dcodeverse`, 2026-11-12, revised 2026-11-13 after review (phase
order, species alternative, capture path, targets). Design only; nothing here is
implemented. Evidence: the Paris three.js run
(`paris_threejs_1112_1647`, judged 0.767) and the Rainier Blender run
(`mount_rainier_blender_1112_2154`, judged 0.60), both scored against our round-11
renders with `scripts/c3d-compare.py`.

## Implementation status (branch `feature/hybrid-scene-plan`, 2026-11-13)

Built and tested on the cached Paris and Rainier runs:

- **Phase 1, ScenePlan stage** (`server/pipeline/scene_plan/`): `Scene Plan` runs after
  Captioning; Anthropic Messages with the schema offered as a tool (forced tool choice is
  rejected by this model family, so the prompt asks for it and a JSON text reply is the
  fallback) or any OpenAI-compatible endpoint; two-layer cache (`Scene Plan/scene_plan.json`
  in the run, `cache/scene_plan/<sha256>-v1-<model>.json` across runs); `validate_plan`
  enforces evidence labels; spend ledger `cache/scene_plan/spend.jsonl`, `max_usd_per_run`
  cap, `no_api` / `FRAME_NO_API=1`; key from `ANTHROPIC_API_KEY` or the login keyring.
  Readings cost $0.09 (Paris) and $0.10 (Rainier), 35 s each.
- **Phase 2, priors** through `ScenePlan.prior()` (`plan.py`), every decision published as
  `plan_overrides_<stage>.json`: camera height in Panorama Depth Calibration (Rainier
  config 1.0 m -> plan 1.2 m inferred; Paris measured 11.7 m kept against plan 11 m),
  river width in Region Map / `terrain_plan.build_plan` (Paris fit 88 m -> 110 m, the
  15 % edge of the visible 130 m), sun window in Panorama Lighting (Paris: all three
  LuxDiT draws at az 81-86 / el 35 sit outside the plan's az 140 / el 55 window, so the
  sharpest is kept and the disagreement logged), landmark distance in Scene Generation
  (a plan landmark matched by class word and bearing replaces the far-clip fraction).
  Not done: the outpainting prompt (the Panorama stage uses FLUX Fill with no text
  prompt, so there is nothing to feed), bank profiles, plausibility gates, species in
  the detector prompt.
- **Phase 4, grading (partial)**: `scene-check.py` records the plan's readings and the
  priors' decisions per run, and `--judge` grades the photo view with Haiku against the
  acceptance list (observe, issues, acceptance, rubric; score in code, penalty capped at
  0.3, failed must caps at 0.6; ~$0.001 per check). It still grades the pyrender frame:
  the first verdict on Paris round 11 called the flat untextured water severe, which is
  the renderer, not the scene -- the client capture of §9 remains to do.
- **Setup**: `anthropic` and `keyring` in `requirements/requirements.txt`; `setup.sh`
  checks both import and tells the user how to store the key. `scripts/hybrid-run.sh`
  reruns cached scenes from Panorama Lighting with the priors and checks them.
- Not started: 2b import test, phase 3 atmosphere/motion, phase 5 species layer,
  phase 6 selectors, client capture.

## 0. The one-paragraph version

The photograph stays the source of geometry, texture and sky. A vision model reads the
photograph once into a provenance-labelled plan that settles what the photo cannot:
vantage, scale, light, atmosphere, what lies off frame, which species and props are
present, and what must move. The plan steers existing stages through tolerances, never
through coordinates. Near vegetation, the weakest layer in both pipelines, becomes a species
layer: a few rigged variants per species, generated once and cached across photographs,
by procedural code or by image-to-3D depending on a test done before the stage is built. The client gets the two things it lacks for realism, atmosphere and
tone mapping, set from the plan and the photo. A judge grades the frame the user actually
sees against the photo and the plan's acceptance list, writing named defects to the
history and picking between candidates where a stage is stochastic. There is no
code-editing refine loop.

## 1. Where realism comes from, layer by layer

Both runs and both pipelines, honestly assessed:

| layer | 3DCodeVerse | ours today | hybrid takes |
|---|---|---|---|
| vantage, scale | planner reads it (Paris 12 m bridge, right); coder builds it literally | measured; Paris camera 1 m, river 87 m (wrong) | plan as prior, measured geometry |
| terrain, water, far field | invented flat plate or straight channel; filler grid | from depth, segmentation, panorama; tied to the photo | ours, with plan constraints (§5) |
| sky, clouds | gradient plus cloud cards | the outpainted panorama | ours |
| sun, ambient | plan numbers, coder's lamp | LuxDiT HDR + measured sun, unstable run to run | ours inside a plan window |
| atmosphere, aerial perspective | volume fog, Cycles | none in the client; none in pyrender | new, from plan haze + photo horizon |
| tone mapping, exposure | AgX, exposure 0 | tonemapping off, exposure by hand | new, fitted to the photo |
| surfaces, textures | node-tree noise, "placeholder_material" when it fails | delit crops, FLUX plates, PBR maps, terrain splat | ours |
| near vegetation and props | procedural species with variants and sway; dense and animated (Rainier, in Cycles) | SAM3D/TRELLIS meshes at 8 k faces, crossed cards, billboards; sparse, flat | species layer (§6): variants + rig + density, built one of two ways after a test |
| landmarks (tower) | 12-part sub-plan, often fails | labelled, moved, size-capped mesh | ours, with plan distance |
| animation | contract-enforced: sway drivers, travel, wake | sway rig within 8 m, video billboards, physics handoff | ours, extended from the plan's motion list (§8) |
| grading | measured gates + rubric judge with caps | DINO cosine, colour ΔE, a few flags | theirs, as reporter and selector (§9) |

Nothing from their world geometry or lighting is taken.

## 2. Data flow

```
photo ─┬─ Captioning ──────────────────────────────────────────────────────────────┐
       ├─ ScenePlan (new; one vision call, cached by photo hash) ─── plan.json ────┤
       │                                                                           │
       ├─ Depth → Panorama (prompt += plan.horizon, plan.behind) → … → Lighting (sun within plan window)
       ├─ PanoramaDepthCalibration (camera height prior) → HeightMap → RegionMap (water prior, bank profile)
       ├─ Terrain* → Treeline/FarField backdrops (landmark distances) ─────────────┤
       ├─ ObjectLabeling (plan species added to the prompt) → … → Clustering       │
       │     └─ SpeciesAssets (new; variants per species, route A or B, rigged, cached by species) ─┐
       ├─ ObjectDistribution → DistributionSynthesis (plan counts as a density prior)                │
       ├─ PanoramaAssetGeneration (image-to-3D for singular things; species assets replace plant buckets)
       ├─ SceneGeneration (places; atmosphere + exposure params from plan + photo)  ◄────────────────┘
       ├─ VideoGeneration → … → SceneAnimation (wind, flow, drift, travel from plan.motion)
       └─ scene.json ──► Unity client (fog, tonemap, wind) ──► capture ──► scene-check judge + gates
                                                                      └──► history.jsonl, verdict.json
```

Stage names refer to the current `server/config.yaml` order. New stages are ScenePlan and
SpeciesAssets; everything else is an existing stage reading one more input.

## 3. ScenePlan stage

One call, after Captioning, with the photo and the caption.

**Model.** Pluggable chat model with vision and JSON schema output. Start with
`anthropic:claude-opus-5-5` (the reading quality is the point); the same adapter speaks
to any OpenAI-compatible endpoint, so a vision model on the other machine can replace it
later. Key from the login keyring as `scripts/c3d-run.sh` does.

**Cache, two layers.** `ScenePlan/plan.json` in the run folder (normal stage skip), and
`cache/scene_plan/<sha256(photo)>-<schema v>-<model>.json` shared across runs. A photo is
read once. Replays, config changes and re-runs never call the API. Force with a flag.

**Schema v1.** Camera frame, ahead = 0°, clockwise; metres and degrees.

```
place       {name?, confidence, evidence}
camera      {height_m, pitch_deg, on_structure, evidence, because}
sun         {azimuth_deg, elevation_deg, warmth: cool|neutral|golden, sky: clear|hazy|overcast, evidence}
atmosphere  {haze_near_m, haze_far_m, tint_rgb, evidence}
horizon     {skyline: text, evidence}
water       {kind: none|river|lake|sea, level_below_camera_m, width_m, bank_left, bank_right,
             continues: {upstream, downstream}, flow_dir_deg?, evidence}
terrain     {kind: flat|valley|slope|ridge|coast, slope_pct?, ground: text, evidence}
landmarks   [{name, class, bearing_deg, distance_m, height_m, side, evidence, because}]
species     [{name, class: flower|grass|shrub|tree|reed, size_m, colours, form: text,
              density: sparse|patchy|dense, where: text, crop_hint: box?, evidence}]
objects     [{class, count, where, evidence}]
motion      [{target, kind: sway|travel|flow|drift|ripple, amount, evidence}]
behind      {text}                         # always invented; prompt only
acceptance  [{id, text, how: visual|probe, priority}]   # from visible claims only
```

**Provenance.** Every claim carries `evidence: visible|inferred|invented` and a
`because`. The label fixes what the pipeline may do with it (§4). A claim without a
label is dropped. The Paris plan sorted this way: height 12 m inferred (used, logged),
width 140 m visible (bounds the fit), "straight along -Z" invented (ignored), Haussmann
grid invented (prompt text), embankment building invented (discarded).

## 4. Trust rules

| evidence | may do | may not do |
|---|---|---|
| visible | move a measurement inside its tolerance; add a class to the detector prompt; set a species | place anything by coordinate |
| inferred | become a prior with a confidence; win only when our measurement fails sanity or disagrees past tolerance; always logged in the report with both numbers | override a measurement that passed sanity |
| invented | text for outpainting, backdrop and behind-camera prompts; the skyline description | geometry, distances, detections, acceptance items |

One helper, `plan.prior(field, measured, tolerance, sanity)`, applies the rule everywhere
so the policy lives in one place and every override is visible in the run report.

## 5. Terrain and water: physical, measured, constrained

Geometry stays from depth and segmentation. The plan adds constraints the code enforces:

1. Water is a level plane at `camera.height_m` below the camera (we already assume a
   plane; the plan fixes the height, which is the Paris squashed-river fix).
2. Banks have a type. A vertical quay gets a wall profile of the plan's height above
   water; a natural shore shelves. The profile follows the measured shoreline, so it
   bends with the photo.
3. The channel continues off frame at its measured width and the bend of its last
   photographed 40 m; the plan says only whether it continues (river) or closes (lake),
   and which way it flows (for §8).
4. Terrain kind gates the height-map smoothing: valley floors flatten toward the water
   plane, slopes keep the depth gradient, ridges keep their depth silhouette.
5. Plausibility gates, code only: water above the plane, shore land below it, slope past
   the terrain kind's limit, channel narrower than width minus tolerance. A failed gate is
   a scene-check finding, never a silent fix.

Optional, phase 7: for a confidently identified real place, register a public 30 m
elevation tile to the camera through the landmark bearings and use it for the large-scale
terrain beyond the near field. Only route to truly physical terrain for Rainier; needs
network, registration and a fallback.

## 6. Species layer: the realism layer we are missing

The Rainier run's flowers are the strongest thing either pipeline produced: six species
with 3 to 4 variants each, real geometry, species-true colours, dense placement, phase-varied
sway. Ours places 8 k-face image-to-3D meshes near the camera and flat cards beyond, so
meadows read as cut-outs. Two caveats before building anything here:

- Their flowers were seen through Cycles, volume fog and soft light. Whether the same
  geometry reads as realistic in the client (glTFast materials, 10 m shadows, no fog) is
  untested.
- Our cut-out look may be mostly rendering, not geometry: billboards from 60 m, no
  contact shadows, no fog, tonemapping off. §7 fixes those and is cheaper than this stage.

**Test first (one hour).** Export one variant of `src/assets/blue_lupine.py` from the
Rainier run as GLB through `Mesh.save`, drop it into the client beside our current lupine
bucket mesh, screenshot both at 3 m and 15 m. If the procedural one is not clearly better
in the client, route B below wins by default.

**What the layer needs regardless of route:** 3 to 4 variants per species, species-true
colour, the 3-bone sway rig from CategoryMeshRigging, density from the plan as a prior on
DistributionSynthesis, and a cache keyed by species so a species is generated once, ever.

**Route A, procedural code.** A code-writing call (same model family as the plan, or the
local coder) is given the species entry, the plan's size and colours, the delit crop and
the bucket representative, and the authoring contract from their `scene_blender` asset
prompt: a `build_<species>(variant)` bpy module, base on z = 0, seeded, 15 k triangles
max, no cameras, no file I/O, no network; their lint policy table reused as-is. Blender
5.2 headless (`--factory-startup`, temp cwd, import allow-list, 120 s limit) runs N
variants and exports GLBs through `Mesh.save`. Gates, code only: triangle budget, bbox
within 2x of the plan size, one connected component, lowest point at 0, colour histogram
within a ΔE of the crop. One repair call, then fall back. Strength: unlimited variants,
true botanical structure, tiny cache entries. Cost: a sandbox for generated code, a
contract to maintain, and a single-shot call writes simpler plants than their 20-minute
coding sessions did.

**Route B, image-to-3D variants.** TRELLIS.2 already builds flowers from crops. Generate
3 to 4 variants per species from the crop with different seeds and mild prompt variation
(plan `form` and colours), colour-match to the crop, rig, cache. Same gates minus the
code ones. Strength: no new execution path, the asset is tied to the photographed plant,
fits the existing PanoramaAssetGeneration stage. Cost: GPU minutes per species instead of
cents; variants are the same plant re-imagined rather than a species with structure;
meshes stay 8 k-face blobs where the plan calls for grass or reeds.

**Decision rule.** Run the import test and §7 first. Build route A only if the test shows
the procedural mesh clearly better in the client and §7 alone leaves the meadow flat.
Otherwise build B. Do not build both.

**Cache.** Keyed by `sha256(species name + form + size + colours + route + contract
version)`, shared across photographs and runs: lupine generated for Rainier is reused for
any later lupine. This is generation on first sight, not a library.

**Singular things** (a boat, a rock, a building) stay on image-to-3D from the crop,
which is better at a specific photographed object than code is.

## 7. Atmosphere, exposure and tone

The judge's complaint on both runs, and visible on our sheet: the client has no fog and
tonemapping is off, and the pyrender frame scene-check grades has neither.

- **Fog.** URP exponential-height fog, density from `atmosphere.haze_far_m` (1/density
  at the far distance), colour from the panorama's measured horizon band, tinted by the
  plan. Also written as a parameter into scene.json so the capture (§9) and any export
  (§10) use the same numbers.
- **Exposure and tone.** A tonemapper (ACES or AgX) on, exposure fitted in code by
  matching the rendered frame's luminance histogram to the photo's at the photo camera.
  Fit once per scene after SceneGeneration, stored on the Scene, verified by the colour
  ΔE already in scene-check.
- **Sun warmth.** The plan's `warmth` is a check on the LuxDiT sun colour; a golden-hour
  photo with a white sun fails sanity and takes the next lighting sample.
- **Shadows.** Shadow distance is 10 m on the quality tier; raise it to cover the species
  layer and the hero objects, since contact shadows are what grounds vegetation.

## 8. Motion

Our SceneAnimation already writes per-object sway, video billboards and physics. The
plan's `motion` list extends it:

- `sway`: wind direction and strength on WindField from the plan (gusts stay the
  client's); sway range raised from 8 m to the species layer's extent; amplitude per
  species from `size_m`.
- `flow`: water flow direction from `water.flow_dir_deg` into the water shader (it has
  waves but no flow today; a scrolling normal offset is enough for a river).
- `drift`: skybox spin about the sun axis (exists, off) at a slow rate when the plan says
  clouds drift.
- `travel`: a hero object with a velocity (the Paris boat) gets the physics handoff it
  already supports plus a wake decal from the Scene.
- A motion gate in scene-check: two captures 1.5 s apart, pixel-change fraction per
  camera, `no_motion` when nothing moves, reported to the judge as a fact.

## 9. Grading: the frame the user sees

scene-check must grade the client's frame, not pyrender's. Today pyrender has no fog,
no tone mapping, no sway and flat ambient, so a photoreal judge would grade a frame
nobody sees.

- **Capture.** A `capture` message on the existing SceneClient WebSocket: the running
  client moves its camera through the photo camera, the three rig views (left, right,
  back) and the top view, at t = 0 and t = 1.5 s, and writes PNGs to a named folder on
  the quality tier with fog, tonemap and shadows on. scene-check sends the message and
  waits for the files. This replaces a Unity batchmode entry point, which on this box
  means an XR template project, a headless GPU context and async glTFast loading with no
  frame loop; the running client already has all three. pyrender stays for the geometry
  flags (fast, no GPU contention).
- **Measured gates, code.** Content fraction of the photo frame, world edge or horizon
  seam, objects floating or sunk against the height map, scale outliers per class, far
  backdrop intersecting terrain, motion. Each gate caps the judge score, as theirs do.
- **Judge.** A vision model gets the photo, the capture, montages of the rig views and
  the plan's acceptance list. Observe-then-score: summary, issues with `target`, `kind`
  (composition, geometry, material, scale, lighting, atmosphere, missing, extra),
  `severity` and a crop box, then acceptance pass/fail with evidence, then the rubric.
  Score computed in code from weights minus penalties, capped by gates. Haiku-class or a
  local vision model; Opus not needed here.
- **Reporter.** `verdict.json` per check and new columns in `history.jsonl`, plus a
  compare mode that diffs two verdicts and names what changed.
- **Selector, bounded.** Where a stage is stochastic and cheap to sample, generate k
  candidates and let the grader pick on that stage's own output: outpainting seed
  (k = 3, consistency with the photo and the plan horizon), lighting sample (sun within
  the plan window, sharpest peak), species asset variants (closest to the crop), camera
  height when plan and measurement disagree (render both). Never more than k, never a
  retry chain, never an edit.

## 10. Optional offline render path

Their photoreal frames come from Cycles. For hero stills, a scene.json importer in bpy
(terrain, water, backdrops, species GLBs, sun and HDR, fog volume, the plan cameras)
renders with Cycles at 32 samples plus denoise. It would also give the judge a reference
of "our scene, rendered as well as it can be" to separate content faults from client
rendering faults. It does not replace the client and is not on the critical path.

## 11. Caching and cost per photograph

| call | when | model | cost | cached by |
|---|---|---|---|---|
| plan | once per photo | Opus | $0.3 to $0.7 | photo hash |
| species asset, route A | once per species, ever | Opus or local coder | $0.25 each (+1 repair) | species hash |
| species asset, route B | once per species, ever | TRELLIS.2 locally | GPU minutes, no API | species hash |
| judge | per scene-check run | Haiku or local | $0.02 to $0.2 | not cached |
| selectors | per selecting stage | Haiku or local | $0.05 per k | with the stage output |

A replay of a cached run makes no calls. A new photo costs about a dollar plus its new
species. Compare the harness: $14 to $17 per run at list, with a 20 to 50 minute coding
session each time.

**Spend guards, built in phase 1 with the ScenePlan stage:**

- The plan cache misses only on a new photo, an explicit `--refresh-plan`, or a schema
  version bump. A bump invalidates every cached plan, so it is a deliberate change and
  the loader first tries to upgrade an old plan in code before calling the model.
- `--no-api` (default for replays and CI): a cache miss fails the stage loudly instead
  of calling out. No paid call ever happens by accident.
- A spend ledger, `cache/spend.jsonl`, one line per paid call: stage, model, tokens,
  estimated dollars, and the cache key that missed, so every bill has a reason.
- A per-run dollar cap in config (default $2); paid calls past it fail the stage.
- The judge is the only per-check cost. It runs on a Haiku-class or local vision model;
  a local model takes iteration cost to zero.

## 12. Evaluation

Three scenes, same metrics on the same sheet: DINO global and layout, colour ΔE, judge
score, acceptance pass rate, gate error count. Baselines today and from the two runs:

| scene | ours global / layout / ΔE | theirs (best view) | judge |
|---|---|---|---|
| Paris | 0.768 / 0.560 / 14.8 | 0.791 / 0.502 / 16.9 | ours: none; theirs 0.77 |
| Rainier | 0.469 / 0.446 / 16.5 | 0.533 / 0.437 / 17.9 | ours: none; theirs 0.60 |

DINO global is not a realism metric: it already prefers their low-poly Rainier over our
photo-derived scene, because it measures semantic similarity. It stays on the sheet as a
regression guard, not a target. Targets at the end of phase 6: judge ≥ 0.70 on Paris and
Rainier with zero gate errors, colour ΔE down on all three scenes, layout score kept,
Paris camera within 2 m of the plan and river width within 15 % of 130 m, a Rainier
meadow judged "dense and varied" that moves.

## 13. Phases

Sizes are relative; the day counts have not survived contact with Unity or Blender yet.

| phase | work | size | done when |
|---|---|---|---|
| 1 | ScenePlan stage, schema, provenance, two-layer cache, report section | 2 d | three cached plans; a replay makes no call |
| 2 | priors: camera height, water width and bank profile, continuity, plausibility gates, sun window, landmark distances, outpainting prompt | 4 d | Paris camera ≈ 12 m, river ≈ 130 m; sun stable across two runs; no Rainier regressions |
| 2b | the §6 import test: one Rainier lupine GLB in the client beside ours | 1 h | two screenshots and a sentence |
| 3 | atmosphere, exposure fit, shadows, motion from the plan, motion gate (§7, §8) | 3 d | ΔE down on all three; `no_motion` never fires |
| 4 | grading frame: client capture message, measured gates, judge reporter, history columns, compare (§9) | 4 d+ | every scene has a judged row with named issues |
| 5 | species layer, one route chosen by the §6 rule | 3 d (B) / 5 d+ (A) | Rainier meadow built from species variants; judge says dense and varied |
| 6 | selectors: outpainting, lighting, species variant, camera tie | 2 d | history variance down on the selecting stages |
| 7 | optional: Cycles importer; DEM for identified places | 5+ d | only if 2 and 5 leave terrain or stills wanting |

Phases 1 and 2 are the payoff for the known Paris defects and are cheap. 2b and 3 come
before the species layer because they test whether the meadow's problem is rendering;
if §7 alone makes it read, phase 5 shrinks to route B or disappears. Phase 4 before 5 so
the species layer is judged on the real frame from its first version. Phase 6 can
interleave with 5.

## 14. Risks

- **The species bet is unverified.** Their flowers were judged in Cycles; the client is
  not Cycles. The §6 import test and §7 come first so the five-day route is built only
  on evidence, and route B needs no new execution path at all.
- **Generated code runs on our machine (route A only).** Species modules execute in Blender. Bounded
  by the import allow-list, the lint policy table, factory startup, a temp cwd, a
  timeout and no network; the same fence their harness uses on every run.
- **API request timeouts.** Both runs lost long calls to Anthropic timeouts. Set the
  client timeout to 10 minutes and retry once; the species call is small, the plan call
  is medium, neither needs 60 turns.
- **Planner confidently wrong.** Bounded by the trust rules: an inferred value moves a
  measurement only when ours failed sanity, and every override is in the report.
- **Species that look wrong.** Colour-matched to the crop, judged against it, and the
  fallback is today's mesh or card, so the floor is the current pipeline.
- **Capture needs a running client.** The capture message needs the client open and
  connected; scene-check fails loudly when it is not, and pyrender flags still run. A
  batchmode entry point stays an option if an unattended nightly check is ever wanted.
- **DINO global misleads.** It favours clean low-poly scenes. It is on the sheet as a
  regression guard only; the judge, ΔE and gates carry the targets.
- **glTFast materials are not URP Lit.** Normal and roughness maps may not apply in the
  client today; verify in phase 3 with a known PBR mesh, and convert on load if not.
- **Scope creep toward their refine loop.** Not building a code-editing loop is a rule of
  this design, not a phase.

## 15. Not in scope

- Building geometry from plan coordinates.
- A code-editing refine loop.
- Replacing depth, segmentation or the panorama with model output.
- Their three.js or Blender scenes as assets.
