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
- **2b import test, half done**: `scratch/species_test/export_species.py` builds three
  lupine variants, a paintbrush and a subalpine fir procedurally in headless Blender 5.2
  (3-11 k triangles each, CPU, seconds) and re-saves them through `Mesh.save` under
  `scratch/species_test/frame/`. The Unity drop-in and the comparison against our cards
  (every one of Rainier's 1778 flower instances is a card) are still to do.
- Not started: phase 3 atmosphere/motion, phase 5 species layer, phase 6 selectors,
  client capture.

### The plan as scene understanding (2026-11-13, afternoon)

hybrid1 showed the gap the harness does not have: the plan read Paris well (Pont
d'Iena, peniches, apartments behind the quay, an office block) and nothing consumed it,
while our own understanding was one caption line plus RAM++ tags that named no bridge
and no building, so Scene Generation placed 26 boats, 24 trees and one tower. Changes:

- **Schema v2**: every landmark, object and species carries `category` from the
  pipeline's own `OBJECT_CATEGORIES` (or `environment`) and a 2-4 word detector
  `phrase`. Free text normalised badly ("stone arch bridge" token-matched rock). The
  prompt now asks for every visible structure as its own entry with a count. Re-reading
  both photographs cost $0.11 each.
- **Object Labeling reads the plan**: `ScenePlan.vocabulary()` adds each entry's
  category word, phrase and proper name to the Grounding DINO vocabulary
  (`plan_vocabulary: true`). Region-like categories (`forest`) are never prompted: a
  "conifer tree line" box outscored the tree boxes inside it and relabelled 115 of
  Rainier's tree blobs forest.
- **Chunked prompts** (`max_prompt_phrases: 12`): one long prompt dilutes every
  phrase's score. With 29 phrases Paris's hero boat fell 0.63 -> 0.49, the tower
  0.77 -> 0.63 and 13 of 17 boats dropped under the threshold. Chunks are merged by box
  overlap, best score wins. Replayed: Paris keeps 16 of 17 boats and gains the bridge
  (0.47) and the office block; Rainier's blob labels are unchanged (flower 50 vs 49,
  tree 41 vs 45).
- **Built-structure veto** in Object Category Clustering counts the plan's visible
  structure categories as named, beside the RAM++ tags.
- `hybrid-run.sh` pre-seeds the plan whenever the cached record's schema version
  differs from the code's, so a schema bump never cascades into the panorama.
- Not done: the plan's `size_m` / `distance_m` as per-entry scale priors for the size
  veto and the far-object rule; the plan's place/terrain kind in place of the caption
  string match for urban scenes; the representation choice (card / mesh / backdrop)
  from category + angular size instead of the per-class lists.

**hybrid2 results** (both scenes rebuilt from Object Labeling, 13:26-16:11):

| scene | dino_global | dino_layout | colour dE | judge |
|---|---|---|---|---|
| Paris (vs hybrid1) | 0.737 -> 0.722 | 0.569 -> 0.561 | 10.5 -> 14.0 | 0.27 -> 0.19 |
| Rainier (vs hybrid1) | 0.496 -> 0.630 | 0.463 -> 0.455 | 17.8 -> 15.4 | 0.12 -> 0.09 |

- **Detection is solved, representation is not.** The bridge was boxed (0.49) and the
  Pullman block named, but both sit at the depth model's far clamp, so Panorama
  Foreground Inpainting kept them in the terrain plate as backdrop and Object Instance
  Refinement absorbed the bridge into a rock blob (containment 0.92). The one building
  that became geometry was an 18 px crop of distant roofs that SAM3D turned into a
  mesh at 185 m. Nothing the plan knows about a far structure (category, bearing,
  distance, height) is used by the stages that decide what a far detection becomes.
  That is the structures layer: plan-described far structures as placeholder geometry
  at the measured depth, not SAM3D on a sliver.
- **Plan readings drift between calls.** The v2 reading of the same Paris photograph
  moved the river from 130 m visible to 110 m inferred, the sun from az 140 to 110 and
  the camera from 11 m to 10 m; Rainier's camera went 1.2 -> 0.9 m and the sun
  az -85/el 8 -> -75/12. With the width now "inferred" the river prior did not fire,
  the channel fitted 90 m again, and Paris's colour dE went back from 10.5 to 14.0,
  which confirms the width was what hybrid1's dE gain came from. The client now reads
  at temperature 0; the cache already pins a reading once made.
- **Rainier's DINO global of 0.630 is the best so far**, but its labels were unchanged
  by the plan vocabulary, so the gain is the 0.9 m camera and a different LuxDiT /
  texture draw (sun again behind the camera, az 144 el 52), not understanding.
- The judge's failed musts are unchanged in kind: no crowded tour boat with a wake, no
  readable bridge, no dense paintbrush/lupine mass, and on Rainier no trail and no
  lone fir at the left edge (the plan names both; nothing consumes them).

### Species layer, structures layer, sun override (2026-11-13, evening)

Built after hybrid2, from the user's two directions: the plan must improve the
scene (not just four numbers) and the near-field cards are the quality floor, with
the constraint that every generated colour comes from the photograph.

- **Species Assets** (`server/pipeline/species/`, stage after Panorama Asset
  Generation). For each (class, bucket) group of flower/tree detections: the bucket's
  palette is measured from its members' masked hires crops (`palette.py`: bloom /
  foliage / stem split by hue and saturation, Lab k-means, the brighter 60 % of each
  part so the client's sun does not darken the plant twice); the bucket is matched to a
  plan species by bloom-palette dE against the species' crop-hint palette (colour words
  only when there is no hint; a bucket further than 40 dE from every species keeps its
  card); `blender_species.py` builds the species' form (spike, raceme, seedhead,
  conifer, tuft, shrub; one material, a palette texture, no default colour anywhere)
  in Blender 5.2 headless, 1.8 s for nine plants, cached by spec hash in
  `cache/species/`; the mesh replaces `category_mesh_{class}_{bucket}`, the card stays
  as the far LOD. Scene Generation keeps a species mesh to
  `species_mesh_near_distance_m` (60 m) whatever its hull ratio; rigging and sway apply
  as to any vegetation mesh. Rainier replay: five flower buckets matched lupine (dE 3),
  pasqueflower seed heads (9), lousewort (14, 16) and paintbrush (11), four tree
  buckets the subalpine fir. Broadleaf trees have no form yet and keep their cards
  (Paris's plane trees and willows).
- **Structure Placeholders** (`server/pipeline/structures/`, stage after Scene
  Generation). A visible plan landmark of category bridge or building that Scene
  Generation did not place as a mesh near its bearing becomes massing: a bridge is a
  deck on piers with parapets spanning the terrain plan's river channel perpendicular
  to its axis; a building is a block of the plan height facing the camera, walked in
  along its ray until it stands on reconstructed ground. Placed at min(plan distance,
  0.95 x far clip) with the height scaled to keep the angular size, coloured by the
  median of its own detection box in the photograph. Paris replay: the Pont d'Iena
  deck 114 m across the 90 m channel on 5 piers at 550 m, colour (137, 131, 118) from
  the absorbed bridge detection's pixels.
- **Sun override** (`plan_sun_override: true`). When no LuxDiT draw lies inside the
  plan window, the key light is aimed at the plan's direction with the env map's own
  sun colour; the sharpest draw still supplies the sky and ambient. Rainier replay:
  three draws at az -115/-143/169 against the plan's az -75 el 12; key light now
  az -75 el 12 at #ffc16c.
- scene-check reads the latest stage's `scene.json`, so placeholders are graded.

Short test runs (`t1`, Rainier from Species Assets, Paris from Structure
Placeholders): Rainier's judge went 0.09 -> 0.32 with 5/6 acceptance and no failed
must (the paintbrush/lupine mass item passes for the first time), DINO global
0.630 -> 0.528 (the regression guard: the near field no longer resembles the photo's
pixels, which is expected when cut-outs of the photo become geometry), colour dE
unchanged at 15.4. The judge's remaining issues: brown conifers (the measured foliage
was read on the lit side; trees now use 90 % of their pixels), one plant shape per
bucket (now 3 variants per flower bucket, 2 per tree, spread per instance), the
clipped summit and the trail (not addressed). Paris: the bridge was built at 550 m
and painted over by the far-field shell (depth ceiling 531 m); structures now stand
at the far-field start at most (372 m). Full rebuild of both scenes from Panorama
Lighting launched as `hybrid3` at 18:42.

**hybrid3 results** (both scenes rebuilt from Panorama Lighting, 18:42-21:22):

| scene | dino_global | dino_layout | colour dE | judge |
|---|---|---|---|---|
| Paris (vs hybrid2) | 0.722 -> 0.755 | 0.561 -> 0.575 | 14.0 -> 13.8 | 0.19 -> 0.21 |
| Rainier (vs hybrid2) | 0.630 -> 0.373 | 0.455 -> 0.434 | 15.4 -> 24.7 | 0.09 -> 0.12 |

- **Paris** is the best Paris so far on every metric. Both structures were built (the
  Pont d'Iena deck at 372 m, the Pullman block at 226 m), the LuxDiT draw sat inside
  the plan window so the override did not fire, and the species layer was idle (four
  buckets keep their cards: no broadleaf form). The judge still fails the same two
  musts: the near boat is a small yellow-roofed vessel without a crowd or wake, and the
  bridge reads as "a vague grey structure at the horizon on the left" -- a flat deck at
  372 m under the far-field strip is a grey line, not the photo's stone arches.
- **Rainier went dark and blue** and that is the sun override. Render mean RGB
  102/115/116 (t1) -> 44/59/73, capture-view ground 87/91/80 -> 25/33/36, black
  fraction 0.03; the judge saw "dusk or night", 3/6 acceptance, failed must a2 (the
  flowers are unreadable in the dark). On the finished scene with the check renderer:
  sky unchanged, terrain texture unchanged (mean 117/144/151), the loss is shading.
  (a) The horizontal irradiance of the key light fell from 1.95 x sin 37 = 1.17 to
  0.92 x sin 12 = 0.19: this draw gave a smaller sun, the solar model took 45 % for
  the 12-degree path and the cosine took the rest. (b) At 12 degrees the meadow
  geometry around a 0.9 m camera shadows the ground completely (down view 18/27/28 at
  any intensity, 71/70/49 with shadows off). Ground brightness with the irradiance
  preserved: el 12 -> 37/43/41, el 20 -> 58/59/50, el 30 -> 61/61/50; hybrid2's sun
  from behind at el 37 gave 87/91/80. The photograph was exposed for its own light;
  the renderers have no exposure, so moving the sun alone cannot be right.
- **Fix.** When the key light moves to the plan's direction its intensity now keeps
  the env map sun's horizontal irradiance (sin el_env / sin el_plan, and the
  transmittance the map's elevation already paid is not charged twice; cap
  `plan_sun_max_intensity` 4.0), and the elevation is floored at
  `plan_sun_min_elevation_deg` 20 with the azimuth kept. The floor is a threshold
  standing in for the section 7 exposure fit, and is documented as one. Replay:
  key light az -75 el 20, intensity 6.89 -> 4.00. Rainier rerun from Panorama
  Lighting as `hybrid3b` at 21:36.
- **The key light's strength is a draw.** Same photograph, same stage: hybrid2 peak
  969 / share 0.18 / 1.95x; hybrid3 peak 294 / share 0.18 / 0.92x; the replay peak
  1564 / share 0.31. Three LuxDiT draws are not enough to pin the intensity, which is
  another reason the exposure should be fitted to the photograph rather than drawn.

**hybrid3b** (Rainier from Panorama Lighting with the fix, 21:36-23:00): key light
az -75 el 20 at 2.60x (x2.12 on this draw's 1.23x). Judge 0.12 -> 0.23 (4/6, failed
must a2 again: "magenta paintbrush is essentially absent"), colour dE 24.7 -> 17.3,
black 0.03 -> 0.01, DINO global 0.377 (t1 with the sun from behind: 0.528, judge
0.32, dE 15.4). Render mean RGB 44/59/73 -> 70/83/90 against t1's 102/115/116. The
scene is no longer night, but the judge's "flat, desaturated, bluish" is right: lit
from the front-left, the camera sees the shaded side of every plant and tree, and the
only fill is 0.45 of a blue sky, so the paintbrush bucket (bloom 5 % of its pixels
under the cut-out's foliage, built as a spike) reads grey-violet instead of magenta.
Lit from behind the camera, as every run before tonight was, the same meshes read as
flowers. Decision: `plan_sun_override` is **off by default** until the section 7
exposure fit exists (ground bounce in the ambient, exposure fitted to the
photograph's lit surfaces); the override, the irradiance preservation and the
elevation floor stay in the code behind the flag. The Rainier cache holds the
hybrid3b state; a run with the flag off reproduces t1's lighting (modulo the LuxDiT
draw).

### Size-anchored placement and plan bands (2026-11-18)

The user's reading of hybrid3: objects do not feel placed, the scale is off (the
trees above all), the structure has not improved, 3DCodeVerse's scenes had better
structure. Measured in the scenes (mesh extents x placed scale, distance from the
camera):

| scene | what | count | distance | height | plan |
|---|---|---|---|---|---|
| Rainier | "trees" (buckets 0-2) | 1294 | 25-120 m | 1.8-2.8 m | subalpine fir 18 m, forest 20 m |
| Rainier | lone fir | 3 | 31-52 m | 9-10 m | 12 m at 25 m |
| Rainier | flowers | 1944 | 1-28 m | 0.4-0.6 m | 0.35-0.7 m |
| Paris | trees | 23 | 83-160 m | 7-17 m | plane 18 m, willow 12 m |
| Paris | boats | 7 | 69-139 m | 6 m tall | 18 boats, the near one 45 m |

The treeline was thirteen hundred 2 m shrubs over a flat band at 25-120 m: an
object's size is its angular size times the depth model's distance, monocular depth
compresses the far field (Rainier's world is 100 m deep: depth ceiling 100 m, terrain
grid 200 m across, far-field start 73 m, no backdrop shell), so the far firs landed
close and shrank to match, and Distribution Synthesis copied them over the meadow.
Nothing in a single step was wrong; the scene had no composition. The harness had
one because its planner's reading (layout, counts, sizes, distances) was the scene.
Ours holds the same reading and used it for four numbers, the detector vocabulary
and two placeholders; Scene Generation still gave depth the last word on every
position and size.

Built (`e3a540b`.. on the branch):

- **Plan sizes as the height gate** (`ScenePlan.size_prior`, Scene Generation
  `plan_size_anchor`). A class the plan gives a standing height for (species 4 m or
  taller count as tree, objects/landmarks of the category, a forest entry counts for
  tree; strongest evidence, then the median) replaces `CLASS_HEIGHT_RANGE_M` with
  [size/2, size x 2]. `unproject_bbox_equirect` already moved a detection along its
  ray to the nearest edge of the class range; it now does so to the plan's band and
  stops at the furthest reconstructed ground (`usable_far_m`: the far-field start, or
  0.95 x the terrain grid's reach). A painted copy outside the band is dropped. Rainier:
  38 tree detections now stand at 25-73 m, 6.4-9 m tall (the band floor is 9 m; at the
  73 m cap their angle allows 6-9 m), 1915 painted copies dropped. Paris: tree band
  6-24 m, 25 detections untouched at 85-162 m, 2 copies dropped.
- **Plan bands** (`plan_bands.py`, `plan_band_region: {forest: tree}`). A region
  the plan names and the region map shows becomes a band: per contiguous span of
  forest columns, the band's angular height over the plan size gives the distance
  it implies (Rainier spans: 112-450 m), the band stands at min(that, usable far),
  three staggered rows of plan-sized conifers at 0.4 x size spacing, heights from the
  column's angle clamped to the band, every one an ordinary synthetic instance for
  the placement loop (ground snap, overlap test, bucket assets). Rainier: 164
  authored, 141 placed, in 6 spans at 73 m. Paris has no forest entry in its plan, so
  no band (its quays are tree-lined and the region map calls 225 degrees of the
  outpainted ring forest; the plan is the gate that keeps that from becoming a wall).
- **Species meshes further out**: `species_mesh_near_distance_m` 60 -> 120 (the band
  conifers were rendering as their delit crop cards, khaki), and a species mesh keeps
  its geometry to 2 % of the screen height (`species_lod_mesh_screen_fraction`),
  because its card is a photo crop with the meadow in its matte: 1,650 of 2,200
  mid-field flowers were cards at 10-28 m and read as blobs; now 1,983 of 2,182
  instances render as meshes.

Finding that sets the next step: the world is as deep as the depth model's plateau
(Rainier 100 m), while the plan says the summit is 9 km off and the treeline's own
angle says 112-450 m. Inside a 100 m world the treeline can only stand at 73 m, 9 m
tall, angularly right and metrically a model. The structural fix is the depth
ceiling from plan evidence (the size-anchored distance of the far band, the
landmark distances) feeding Panorama Depth Calibration's far range, which is a
2-3 h cascade per scene and a terrain-quality question at a 400-800 m grid; Paris
already runs at 400 m. Not done this round.

**band1 results** (both scenes from Scene Generation, 16:17-16:49):

| scene | dino_global | dino_layout | colour dE | judge |
|---|---|---|---|---|
| Paris (vs hybrid3) | 0.755 -> 0.777 | 0.575 -> 0.594 | 13.8 -> 13.6 | 0.21 -> 0.17 |
| Rainier (vs hybrid3b) | 0.377 -> 0.382 | 0.423 -> 0.374 | 17.3 -> 18.6 | 0.23 -> 0.13 |

Paris is the best Paris on every measured metric (the 7 m saplings on the quays
moved out to the 6 m floor; nothing else changed), and the judge's complaints are the
boats, the bridge and the water as before. Rainier's treeline now stands where a
treeline stands -- 141 firs of 9-10 m at 73 m along the forest spans, in front of the
dome, replacing 1,915 knee-high copies over the meadow -- and the judge marks it
down: "tan spire conifers are uniform copies ... no dark forest band", the meadow
"sparse" (the mid-field flowers that read as blobs were cards with visual mass;
as 0.5 m meshes at 10-28 m they are correct and thin), the lone fir "absent or very
small" (its 10 m stands inside the band now). The structure is right and the
picture is not, which is the honest state of this round: scale and placement are
fixed in the measured sense, and what the eye wants next is density (the plan's
"dense" paintbrush against our sparse mid-field), a dark measured conifer colour
under the client's light rather than the check renderer's, variation (one conifer
bucket with two variants), and the lone fir treated as the landmark the plan says it
is. The Paris boats are their own fix: every hull is forced to its kind's 24.5 m
(0.7-1.4x the kind prior), several are quarter-turned across the river, and the
near tour boat (366 px, score 0.59) is then dropped for lying inside their reach;
the plan's boat sizes (45/35/30/15 m by count) are the prior that should replace
the kind length.

**The trees up the mountain (user, 2026-11-18 evening).** Horizontal rays from the
camera meet Rainier's reconstructed terrain at 25 m on every photographed bearing:
the meadow is 25 m deep in this world and the mountain's flank rises from there,
so a tree pushed to 73 m along its ray stood 20 m up the flank, and the band at a
fixed 73 m was a picket line, not the photo's clumps. Changed: the band walks every
forest column of the region map, casts the band's bottom edge onto the terrain and
stands a tree where it lands (the mask's clumps and gaps are the band's; a column
less than 35 % solid is skipped; a column taller than one tree gets rows behind, up
the slope); a ground-reach measurement (the median horizontal-ray hit, 25 m here)
caps both the band and the size anchor; a tree's height is what its angle allows
where it stands, with the plan's floor only at or beyond half the implied distance.
Rainier now: 74 band trees at 7-25 m, 2.7-3.1 m tall, the 38 detections at 25 m,
2-9 m. That is the faithful diorama and it reads as a model, because it is one: the
treeline's angle says 112-450 m, the plan's summit 9 km, the depth model's meadow 25 m.
No placement rule fixes that. The next build is the world's depth (section 5: a
plan-constrained ground out to the treeline's foot, the mountain as a far shell at
its implied distance), and until then the scale of Rainier is the depth model's.

**Flowers not as colourful (user).** The bucket palettes averaged highlight and
shadow pixels and counted the lousewort's pale yellow (hue 53-65 deg) as foliage,
so its "bloom" was the soil and stems left over, and the paintbrush bucket's three
rows were crimson, brown and pink. A flower bucket now re-reads its bloom around
the species' own colour measured at the plan's crop hint (pixels within 30 deg of
that hue and at least 0.45 of its chroma), each cluster represented by the Lab
median of its more saturated half; foliage starts at 68 deg; a bucket further than
20 dE from every species keeps its card (a shrub/fir-fragment bucket of 3,468
members had matched lousewort at 29, the pasqueflower seed heads paintbrush at
25). Paintbrush bloom [132,53,75]/[108,76,63]/[191,71,90] -> [202,62,86]/[151,43,80];
lousewort [198,154,128] -> [176,177,63]. Every colour is still the bucket's own
pixels. The hybrid3b lighting (plan sun at 2.6x, warm) in the Rainier cache washed
every render since; `band2` reruns Rainier from Panorama Lighting with the override
off (launched 17:50), then Paris from Scene Generation.

**`band2` (19:15) and the forest that is not a forest (user).** Rainier with the
override off: DINO global 0.406 (band1 0.382), layout 0.454 (0.374), ΔE 15.2 (18.6),
judge 0.17 (0.13), no failed musts. Better on every number, and the user is right that
it does not read: "the forest still isn't looking like a forest compared to the photo
and the flowers colors aren't right". Three causes, measured on the cache:

1. *The species reference colour was wrong before the bucket was ever read.* The plan's
   crop hint for the lousewort holds pale-yellow spikes, olive stems and a paintbrush;
   k-means on it returned the olive, and the guided re-read then selected olive pixels
   in every bucket. The plan's own colour words were unused. Now `word_guided_colour`
   reads the hint around each word's anchor (hue ±25°, lightness ±18, lit half,
   chroma-rich median): paintbrush [157,46,85], lousewort [223,217,110], lupine
   [89,88,155] with white [176,187,214], fir [34,55,31]. A flower bucket is then
   matched by *which species' colour is found in its crops* (≥3 % of pixels, lightness
   down-weighted 0.15 in the distance: the hint's heads sit in shade, the crops' in
   sun), and its bloom is the species' photographed colour with one agreeing bucket
   tone behind it. Rainier: lupine dE 12, paintbrush 17, lousewort 6 and its greener
   bucket 19; the stalk bucket (750 members) keeps its card at 29.
2. *Most mid-field flowers were never the species mesh.* 1,699 of the ~2,000 placed
   flowers took the bucket's card (bake-time `min_mesh_angular_px`), and the card was
   the photo crop's cutout: delit, matted with meadow, pale whatever the palette said.
   `species/card.py` rasterises the species mesh itself (painter's fill of its
   palette-coloured faces, 6 ms, no GPU) and the stage publishes it as
   `category_mesh_<class>_<bucket>_card`, so the plant keeps its colours at every
   distance; tree buckets get one too.
3. *The band was one row.* Each column's tree was as tall as the mask's band, so the
   forest was a line of 2.7 m cones at the slope's foot. A forest is several crowns
   deep: trees are now 0.55 of the band's height (floor 0.08 × plan size), rows stand
   behind the first up the slope until their tops reach the column's mask top as seen
   from the camera (a vertical ray gives the terrain height; nothing stands above the
   photographed skyline), the lateral step is 0.3 crown and band trees carry their own
   overlap separation 0.45. Rainier: 585 firs of 1.4–1.5 m, 3.3 rows deep, 412 placed.

Not touched: the firs render khaki in pyrender because a dark-green albedo under the
2.0-intensity warm sun with no exposure control goes olive; that is section 7 and the
judge's "flat, muted" lighting complaint. Replays must run in the `frame` env.

**`band3` (19:48, Rainier from Species Assets).** DINO global 0.464 (band2 0.406),
layout 0.476 (0.454), ΔE 16.4 (15.2), judge 0.31 (0.17; t1's 0.32 is the branch's best).
The sheet now has a dark conifer band with clumps and a lone spire fir at the left,
lupine and yellow spikes in the species' photographed colours, and no khaki: the tree
and flower cards are the species' own. 569 band firs (185 meshes, 238 species cards at
25 m where 1.4 m falls under `min_mesh_angular_px`). The judge's remaining majors are
the smooth clipped summit (world depth, section 5) and a failed must: "the magenta
paintbrush that dominates the photo's foreground is almost absent". That is density:
the paintbrush bucket has 33 members and placed 5 meshes, the lupine bucket's painted
copies placed 920 cards, while the plan says paintbrush *dense, across the whole
width* and lupine *patchy, lower right*. Density from the plan's species density and
`where` is the next build, ahead of the world depth. Paris band2 (from Scene
Generation): 0.776 / 0.594 / 13.6, judge 0.23 (0.17); unchanged geometry.

**`band4` (23:40, Rainier from Scene Generation) -- "shouldn't more of the flowers be
3D objects?" and "we really need to be placing/clustering the trees better" (user).**
Measured first. (1) The most-painted flower bucket (3,470 of 4,280 painted copies) is a
junk bucket: its photographed crops are a trail, grass blobs and mountain strips, and
it had matched the yellow lousewort on 4 % pixel support; the paintbrush bucket had 23
copies. (2) Two-thirds of the species instances wear a variant key (`_v1`, `_v2`) that
`SPECIES_MESHES` did not list, so they went through the reconstruction gate
(surface/hull 0.3-0.9 < 1.4, "card at every distance"); the rest fell under the 120 px
bake gate, a 0.4 m flower beyond 6 m. 1,899 flower cards, 114 meshes. (3) Every band
fir was 1.44 m: the band floor (8 % of the plan size) held every tree in a 25 m world,
the 0.8-1.25 height spread was clipped away under it, and one tree per column every
0.3 crown is a picket fence; the judge: "uniform, evenly spaced spires".
Built: `plan_population.py` -- the painted positions stay and each copy is re-dealt a
species by the plan's density (dense 1, patchy 0.35, sparse 0.12) and `where` (side of
the frame, foreground within 15 m), and takes the species size (paintbrush 2,783,
lousewort 941, lupine 556; pasqueflower and bistort have no mesh). `is_species_key`
treats a variant as its bucket; `species_min_mesh_angular_px` 40 gates species meshes
(a 0.4 m flower to ~15 m, a band fir to 25 m): 369 flower cards, 0 tree cards.
`plan_bands.py` rewritten: the mask's top edge is the skyline, its local maxima
(`find_peaks`, 0.25 deg prominence) are crown tips and each gets a tree exactly as tall
as the tip needs where that column's foot meets the terrain; a mask component narrower
than 2.5x its height is a lone crown and gets no fill; between tips the stands are
filled in rows under the skyline and thinned by a slow random field along the band
(clumps and gaps). Rainier: 63 tips (9 lone), 388 fill, 2.8 rows.
Result: DINO global 0.482 (band3 0.464), layout 0.460 (0.476), ΔE 15.0 (16.4), judge
0.31 (0.31), 6/6 acceptance and no failed must. The paintbrush reads as the dominant
magenta across the meadow; the lousewort spikes are where the photo has them. Still
wrong, and all one cause: the treeline is a scale model. The depth model saturates at
104 m on Rainier (forest at 11-29 m, mountain at 25-104 m), so the "forest" is 1.4-2.8 m
spires at 7-25 m with 0.4 m flowers in front, and the summit is the grid's clipped edge.

**Far field from the plan (`band5`, launched 23:41 from Panorama Depth Calibration).**
Section 5, built: `plan_far_knots` in the calibration stage measures, per photographed
column with forest, the band's angular height against the plan's tree size (18 m fir
-> implied 209 m, measured 24 m) and, for a far landmark (mountain words, plan
distance >= 500 m), the TERRAIN pixels above the horizon within 25 deg of its bearing
(summit: measured 104 m, plan 9000; the icefall 6000). `apply_far_knots` remaps both
depth maps monotonically in log-log through those knots, identity below 0.4x the first
(9 m), the last slope continued. Then the existing machinery does the rest: the
saturation plateau is 6000 m, the far field starts at 4200 m, the grid becomes 400 m
across (depth ceiling 200 m), the summit goes to the far-field shell instead of the
clipped grid edge, and the plan band stands at ~190 m with 9-36 m firs. Risk: the
stretch between 9 and 24 m measured is x3.4 in log slope, so depth noise there becomes
terrain relief; the Treeline Backdrop (min 40 m) may now also stand its ribbon up.

**`band5` (01:11, Rainier from Panorama Depth Calibration).** The remap did what it
said: grid 400 m, far field 4200 m, the treeline ribbon stood up at 166-200 m (Treeline
Backdrop's 40 m floor was never reached before), and the world reads deep on the sheet:
the stands at the mountain's foot at a believable scale, snow patches in the mid-ground,
the lone fir near. DINO global 0.482 -> 0.471, layout 0.460 -> 0.471, ΔE 15.0 -> 14.9,
judge 0.31 -> 0.29, 6/6. Three things wrong, all measured. (1) The plan band's feet were
still at 7-15 m (p50 14.6, only the p95 at 126 m): the foot probe took the first ray that
landed anywhere, and the mask's bottom rows are mid-ground shrubs whose rays land on the
near meadow; `near_fraction` was 0 so the "far ground" test never fired. Now a stand's foot
must land at >= 0.3x the distance its angle implies for the plan tree (a lone crown keeps
its own base). (2) The summit is still the grid's clipped crest: the far field started at
0.7 x the 6000 m plateau = 4200 m, so Terrain Reconstruction anchored the ridge chain in
the grid, the Height Map built the compressed flank as a 19-42 m wall at 173-200 m, and
the shell got 82 faces. When the far field is plan-anchored it now starts at the grid's
edge (200 m): the ridge chain is handed to the shell and pixels deeper than the grid are
excluded from it. (3) "A crowd of tiny figures" in the mid-ground: 0.4 m flowers painted
to 30 m are upright coloured sticks on bare ground at 20-30 m; `class_max_paint_radius_m`
flower 15. `band6` relaunched from Panorama Depth Calibration at 01:13 with all three.
Also measured: the object-removed panorama keeps the meadow's green and pink at 10-30 m,
so the brown mid-ground is the delit terrain albedo under the scene sun with no exposure,
the same cause as the maroon paintbrush and the khaki firs -- section 7 is next.

**`band6` -> `band7` (02:44, Rainier from Terrain Reconstruction).** band6 showed the
ridge chain still anchored with the far field at 200 m: the chain runs round the whole
outpainted 360 along the meadow's horizon (22 m), so its median depth was near and
the whole chain anchored, mountain included. Terrain Reconstruction now hands a near
chain's far points (range under the crest saturated or beyond the grid) to the shell
(1,206 of 4,095 points, bearings -41 to +63) and publishes the anchored chains
(`MOUNTAIN_RIDGE_CHAINS_ANCHORED`); the Far Field Backdrop leaves landform pixels to
the terrain only in anchored columns (8,206 px instead of 107,013; shell 2,275 faces
instead of 82). band6 was stopped and band7 run from Terrain Reconstruction. Result:
judge 0.22, 4/6, failed must a2 again, and three regressions, all understood. (1) The
shell tore the mountain into stair-stepped blocks: its silhouette test compared
depth-map values, and the plan-anchored flank runs 245-6000 m in map units while the
radius mapping folds it into 200-500 m; now judged on the rendered radius. (2) The band
stood at 25-154 m (p50 58) with 1.4-9.8 m firs: the foot fix worked but the forest's
implied 209 m lies beyond the 200 m grid; the forest knot is now capped at 0.75x the
grid's maximum half (150 m). (3) "No dense field of paintbrush": the painter made
1,044 flowers (the junk bucket re-clustered to 811 members, the 15 m radius cut the
rest) and the overlap test kept ~400; the painter's density is learned from the
detections, so it now scales by the plan's species density words (dense 2.5, patchy
1, sparse 0.5). Also: the treeline ribbon behind the band read as "black blob-like
lumps", so the Treeline Backdrop's floor is 250 m while the plan band stands the
photographed forest. `band8` from Panorama Depth Calibration launched 02:47.

**`band8` (04:13, Rainier from Panorama Depth Calibration): judge 0.03, 4/6.** Worse,
and the cause is the split itself: the shell takes pixels deeper than the grid (200 m)
and the grid the rest, but the depth model reads the mountain's foot at the treeline's
own 25-33 m, so the remap put the lower flank at 150-200 m (flat ground at the grid's
edge, a sliver from the camera, sky above it) and the 200 m threshold cut a stair-stepped,
holed boundary through the flank where the log slope amplified the depth noise. The
radius-space tear test changed nothing (2,144 faces vs 2,275). Fix: the far field is not
a depth threshold for landform. `far_landform_mask` -- TERRAIN/SNOW pixels above the
forest's skyline in their column (the horizon where no forest stands) -- is the mountain
whatever its depth, lifted to 1.1x the grid's half in both maps (99,593 of 126,755
landform px; 16,859 lifted), so the Height Map excludes it, the ridge handover and the
shell take it whole. Flowers: 2,259 painted, ~700 placed; `overlap_min_separation_overrides`
flower 0.8 -> 0.4 (a meadow's plants interpenetrate). `band9` launched 04:15 from
Panorama Depth Calibration -- and stopped at 04:43: the lift reached only the compressed map,
the metric map the shell and the ridge handover read had been published before it (shell
2,144 faces again). Metric map republished after the lift; `band10` launched 04:44.

**`band10` (06:09, Rainier from Panorama Depth Calibration): judge 0.08, 4/6.** The lift
reached the shell (ridge handover 1,306 points, shell 2,282 faces) and the mountain is
whole in the far set, but still stair-stepped with holes: the tear test, even on rendered
radius, sees the lifted flank at 220 m (radius 210) beside the summit at 6000 m (radius
500), a 2.4x ratio against the 1.8 threshold, so faces between lifted and unlifted
neighbours are dropped everywhere the depth noise crossed the floor. `max_depth_ratio`
1.8 -> 6 (inside the far set the only silhouette that matters, against the sky, is the
alpha's). The flowers: 1,423 paintbrush placed but p10/50/90 at 4.5/11/14 m, 65 within
3 m -- the painter spreads copies by area, 2.4/m2 flat, where the photograph's foreground
is a carpet. `fill_population`: the plan's density words become plants per m2 (dense 6,
patchy 1.5, sparse 0.5), summed over the class's species inside 15 m across the
photographed cone plus 20 deg either side; the shortfall over the painted copies is
sampled by area on vegetation cells of the top-down region map, 0.25 m apart, each a
species by density and where. Rainier: 988 added to 592 painted in 191 m2 (target 9/m2),
4,010 objects kept. `band11` launched 06:16 from Far Field Backdrop -- and stopped at
06:17: its shell had 2,387 faces against 2,282, so the ratio was not what dropped the
blocks. Measured instead: the shell drops any vertex inside the grid's square (|x|, |z|
< 200 m) and starts at radius 200 m, so at bearings off the axes the lifted flank at
200-283 m was dropped in blocks (the stair steps and holes since band7). The shell now
starts at the grid's corner radius, half x sqrt 2. `band12` launched 06:17 from Far
Field Backdrop and stopped a minute later: the depth threshold had moved with the
radius (far beyond 283 m, the lifted flank at 220 excluded again); decoupled, far in
depth stays the grid's half. `band13` launched 06:20.

**`band13` (07:20, Rainier from Far Field Backdrop): judge 0.15, 5/6, no failed must.**
The foreground is a meadow again: 799 filled + 766 painted flowers in the cone, 4,062
objects, and a2 (dense paintbrush, yellow spikes, lupine) passes -- the lupine is
there (279) but reads dark under the flat light. The mountain still had its vertical
cutouts, with 2,937 faces: not the tear ratio, not the square. Cause found in the
hand-back: the chain's points at the mountain's foot have the forest (150 m, inside
the grid) under their crest, so they anchor, and the backdrop then handed *every*
landform pixel in those columns to the terrain, summit included -- vertical strips cut
wherever the skyline dipped. Now only pixels at or below the anchored crest in that
column are the terrain's. `band14` launched 07:21 from Far Field Backdrop -- and
moved 400 px, so that was not it either; stopped at 07:22. Rasterised the shell from
the capture camera instead (scratch: shell_raster.png, shell_alpha.png): the mesh only
has cells whose own vertex pixel is far, so its boundary is a staircase of stride cells
*inside* the silhouette, and the full-resolution alpha cannot paint what the mesh does
not cover -- the stepped summit and the sky between the treeline and the mountain's
foot, since band7. The vertex grid now keeps any cell with a far neighbour (3x3
maximum filter) so the mesh reaches past the silhouette and the alpha cuts the edge.
`band15` launched 07:22 from Far Field Backdrop.

**`band15` (08:25, Rainier from Far Field Backdrop): judge 0.31, 5/6, no failed must.**
The stair-steps are gone: the summit is the photograph's silhouette on the shell, the
treeline stands at the mountain's foot at a believable scale, the lone fir is at the
left edge, and the foreground is a dense paintbrush, lousewort and lupine meadow (the
lupine reads faint). What remains on the mountain: a strip of sky between the treeline
and the mountain's foot, and the lower flank "smeared and melted". Both are pixels that
stayed in the grid: the strip between a column's own treeline and the lift's smoothed
one, remapped to 150-200 m and laid on flat ground at the grid's edge. The lift now
takes everything that is not sky above the column's own treeline (114,172 px, 8,887
lifted; landform above the horizon where no forest stands). `band16` from Panorama
Depth Calibration follows. Still open after that, in order: exposure (section 7: maroon
paintbrush, faint lupine, grey-brown ground, flat snow), the trail, conifer variation,
haze.

**`band16` (10:00, Rainier from Panorama Depth Calibration): judge 0.26, 5/6, no failed
must.** The mountain's lower flank is now on the shell (5,163 faces), but ragged sky shows
between the spires and the mountain's foot on both sides: the photographed treeline's own
canopy pixels, remapped to 150-200 m, lie flat on the ground at the grid's edge -- the
Treeline Backdrop ribbon used to stand exactly those up, and band7 switched it off. The
lift now takes the far part of the forest too (forest pixels remapped beyond half the
grid), so the shell carries the photograph's own treeline at its radius and the plan
band's trees stand in front of it. `band17` from Panorama Depth Calibration follows.

**`band17` (11:20, Rainier from Panorama Depth Calibration): judge 0.26, 5/6, no failed
must; dino 0.504/0.560, dE 14.0.** The flank's gaps closed, but two new things: a pale blue
strip under the shell's treeline, and flat-topped pale blocks at the horizon left and
right. The strip is a phantom hill: the depth model reads a tree's lower rows nearer than
its top, so under a canopy lifted at 100-180 m the same trees' lower rows (13-93 m) stayed
in the grid as a low mound at the horizon with the skybox between it and the shell. The
lift now takes the whole per-column run of forest rows under a far canopy pixel (315k px,
192k lifted, against 180k/56k). The blocks are the removed near firs: Panorama Foreground
Inpainting left faint grey-blue spires where they stood against the sky, the terrain
regions typed them terrain, and the forest lift stood them up on the shell. The backdrop's
ghost rule (removed-object pixels typed sky or built are left off the shell) now also
drops removed-object pixels whose nearest surviving neighbour is sky, whatever the
inpainted texture was typed (8,068 px). `band18` from Panorama Depth Calibration follows.

**`band18` (12:57, Rainier from Panorama Depth Calibration): judge 0.26, 5/6, no failed
must; DINO 0.482 / 0.580 (band17 0.504 / 0.560), colour dE 12.4 (14.0).** The phantom hill
and the ghost spires are gone (315,204 landform px above the treeline are far, 191,856 of
them lifted; 8,888 removed-object px left to their models; shell 4,067 vertices). What the
sheet still shows, read off the render's depth buffer: a line of sky 0.3 deg tall under the
whole treeline and a wedge of sky 1.3 deg tall at the right edge. The line is the gap
between the two domains: the lifted treeline's foot stands at eye level on the horizon row
of the shell (283 m), the grid's edge is 1 m lower at 200 m, and the ground skirt that was
built for exactly this (Paris) puts its vertices on stride rows, 0.7 deg apart and only
below the horizon, so it never catches a wedge that narrow (0 skirt vertices). The wedge at
the right is a forest on a slope behind a rise in the meadow: its lowest pixel is above the
horizon (150-190 m by the depth map, under the 200 m far start, so not on the shell), no
ray from its band lands on the terrain (151 of the forest band's 401 columns "off the
terrain"), and the region map's blob there is narrower than 2.5x its height, so it counted
as one lone crown: one tree. Two fixes. (1) A foot curtain on the backdrop: per column the
vertex under the shell's lowest one is moved onto the ray to the grid's edge point (the
edge's own ground height along that azimuth) and takes the panorama row that ray sees; the
pixels between the foot and that row stand opaque, less sky, removed objects, water and
anything measured nearer than half the grid's edge. Rainier: 373 vertices, 2,089 px, wedge
0.27 deg median, 2.3 max; a test render against the band18 scene shows the line and the
wedge closed, the photographed forest standing in the wedge. (2) The forest band foots a
column whose rays all miss by the calibrated depth at its lowest pixel (capped at the usable
far distance), and a component counts as a stand when it is wider than two of the plan's
crowns at its measured distance. Rainier, standalone: 99 columns footed by the depth map,
18 trees at bearings 24-30 deg at 190-208 m, 3-13 m tall, where there were none; elsewhere
the band is unchanged (209 vs 196 trees in the cone, median 98 m / 3.4 m). `band19` from Far
Field Backdrop follows, carrying both and the exposure fit of section 7.

**`band19` (14:15, Rainier from Far Field Backdrop): judge 0.31, 5/6, no failed must;
DINO 0.493 / 0.587 (band18 0.482 / 0.580), colour dE 12.3 (12.4).** Back to band15's
score with every horizon artefact gone from the judge's list: no strip, no blocks, no
patch. The forest band placed 982 trees from 424 columns, 95 footed by the depth map (59
still off the terrain); the right edge has its forested hill and its trees. The exposure
fit (section 7) ran on 68,400 near pixels: sun x0.81, ambient x1.23, percentile error
0.044 -> 0.019, sun intensity 2.0 -> 1.62 for the client. The judge's list is now content:
the mountain painterly (major), the flowers low-poly with oversized yellow spikes (major),
the snow a grey sheet without the trail, the conifers uniform narrow cones in a row, and
the light "cool and even" (moderate) -- the fit matched the near field's luminance
percentiles but lowered the sun to do it, and what the judge misses is the warm low sun's
highlights, not the exposure. Next: the sun's colour and direction against the plan's
warmth (section 7), conifer variation, and the flower materials.

### First full runs (`hybrid1`, night of 2026-11-13, both from Panorama Lighting)

| scene | vs | dino_global | dino_layout | colour dE | judge | priors applied |
|---|---|---|---|---|---|---|
| Paris | round11b | 0.769 -> 0.737 | 0.561 -> 0.569 | 14.8 -> 10.5 | 0.27 (3/6 acceptance) | river 89 -> 110 m; tower 169 -> 557 m; camera 11.8 m kept (plan 11) |
| Rainier | round11 | 0.470 -> 0.496 | 0.446 -> 0.463 | 16.5 -> 17.8 | 0.12 (4/6 acceptance) | camera 1.0 -> 1.2 m (plan, inferred); no water |

Paris: 74 min, Rainier: 87 min, both exit 0. Spend for the night: two plan readings
($0.19) plus three judge calls ($0.003).

What it shows:

- The plumbing works end to end: cached plans hit, no cascade from Captioning, every
  prior logged its decision and published it, scene-check carries them into the history.
- The Paris priors did what they were meant to. The river is visibly wider and the
  tower now stands behind the far tree line at a believable size; colour dE improved
  more than in any earlier round. DINO global dropped a little, which §12 already treats
  as a regression guard rather than a target.
- **The sun window never fired.** In both scenes all three LuxDiT draws fell outside
  the plan window, so the measured direction was kept. For Rainier that is the wrong
  call: LuxDiT puts the sun behind the camera and high (az -178, el 42; the stage itself
  logs that this is 150 deg outside the photographed cone, so it is a guess as well),
  while the photo's shadows show low light from the left, which is what the plan says
  (az -85, el 8). Sun direction is never "visible" in the schema's sense unless the disc
  is in frame, so the §4 rule "inferred and outside tolerance -> keep measured" makes
  this prior a no-op. Proposed change: for the sun, let an inferred reading clamp the
  draw into the window when the plan's `because` cites shadow direction and no draw
  lands inside it; log both. Not changed yet, this is a trust-rule decision.
- Rainier rendered darker and bluer than round 11 (the delighting and terrain textures
  reran under the new LuxDiT draw), which is where the dE loss comes from; the plan
  touched nothing in Rainier except the camera height.
- The judge grades the pyrender frame and it shows: for Paris it reports the camera at
  water level and the river flat, both renderer artefacts (the camera is 11.8 m up).
  The absolute scores (0.27, 0.12) are not comparable to the harness's 0.60-0.77 until
  the §9 client capture exists. The failed must items are real though: no stone bridge,
  no crowded boat with a wake, and for Rainier no dense magenta/blue flower mass, which
  is the species layer of phase 5.

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
  *Status (band19):* built as the Exposure Fit stage (`server/pipeline/lighting/exposure_fit.py`,
  after Material Maps). It renders the photo view three ways (sun only, ambient only,
  unlit) at 480 px with `scripts/render-scene.py --light-passes --no-fit`, fits sun and
  ambient scales to the photograph's luminance percentiles over pixels nearer than 60 m
  (the photographed far field carries its own light), and stores them on the scene's
  lighting (`lighting.sun.intensity` scaled, `lighting.ambient_scale`, `lighting.fit`) and
  the plan's atmosphere on `scene.atmosphere`. No tonemapper: it would alter the
  photographed content. Rainier: sun x0.81, ambient x1.23, error 0.044 -> 0.019. The Unity
  client consumes the scaled sun intensity; `ambient_scale` and `atmosphere` are written
  but not yet read by SceneParamManager.
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
