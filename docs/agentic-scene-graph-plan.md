# Agents over a semantic scene graph: a proposal (2026-11-21)

A plan for the round after the hybrid one. The hybrid plan (`hybrid-scene-plan.md`)
gave the pipeline a plan read once from the photograph and a judge at the end. This
one gives it a representation of the scene that every stage reads and writes, an
agent that reasons over that representation and calls the existing methods as tools,
and populations that are fitted to explain the photograph instead of painted by rules.

## 0. Why now

The last round's faults say what is missing. Every one of them was a string standing
in for knowledge:

- The London plane was a conifer because `species_form` falls back on the class word.
- The Paris bank lost its treeline because a distance rule (ribbon beyond 250 m)
  encoded "the plan band stands the forest", which is only true when the species
  has an asset.
- Rainier's meadow is dark because the exposure fit takes luminance percentiles over
  "the near field", and the near field is mostly flowers.
- Shark Fin's cliffs are dark because the plan's 2 deg sun was trusted, though the
  photograph shows cliff faces lit from the right, which a sun behind the fin cannot do.

Each fix was a new rule in config (900-odd keys now) or a regex. The pipeline is 45
stages that each know their own inputs and nothing about the scene; the only thing
that reads the whole scene is the judge, and its findings are text that nobody acts on.
3DCodeVerse has the opposite shape: an agent that reasons about the whole scene and
writes code, with no measured geometry under it. The pipeline needs the agent's loop
without the code, and the agent needs the pipeline's measured tools. The thing between
them is a scene graph.

## 1. The one-paragraph version

Every run builds a **scene graph**: typed nodes (camera, sun and sky, terrain regions,
water bodies, landmarks, structures, populations, singular objects, backdrops) with
attributes that carry their evidence, geometry handles into the existing artifacts,
and relations. The plan is its first layer, measurement its second, fitting its third.
Existing stages become **tools** with declared inputs and outputs in graph terms,
cached and deterministic as they are now; panorama generation stays a fixed tool that
runs once. An **agent** (one orchestrator on a strong vision model, cheap workers for
volume) runs perceive, hypothesise, build, observe, critique, revise, editing only the
graph and never coordinates or code, under a budget. Vegetation and other repeated
things are **populations**: a density model over terrain features plus a clustering
process, fitted so that a draw from it explains the photograph's masks, replacing the
painting rules. The judge's findings land on nodes, so the next round acts on them.

## 2. The scene graph

### Nodes

| node | attributes (examples) | geometry handle | today's source |
|---|---|---|---|
| Camera | height, pitch, fov, on_structure | pose | plan, Depth Calibration |
| Sun / Sky / Atmosphere | azimuth, elevation, warmth, haze near/far, sky kind | HDR, sky mask | plan, Panorama Lighting |
| TerrainRegion (meadow, forest band, scree, beach, cliff, quay) | ground kind, slope, wetness, albedo (delit), extent | mask on the panorama, footprint on the grid | Region Map, plan.terrain |
| WaterBody | kind (river, tarn, sea), level, width, bank profile, motion | region mask, plane | Region Map, plan.water |
| Landmark / Structure | name, distance, height, category | mesh or placeholder | plan.landmarks, Structure Placeholders |
| Population | species mix, density model, clustering, size distribution, orientation | region ref, seed, draw | Object Distribution, Distribution Synthesis, Grass Cover |
| Instance (singular) | class, position, size, motion | mesh, card, video billboard | Object Labeling, Object Completion |
| SpeciesAsset | form, palette, variants, atlas hash | GLBs | Species Assets |
| Backdrop | kind (treeline, far field), depth band | strip mesh | Treeline / Far Field Backdrop |

These are the landscape kinds. §13 gives the structural types underneath them, so the
same graph holds a city, a painting or a fantasy scene, and adds a scene-level Style node.

Every attribute is `(value, unit, evidence, confidence, source)` where evidence is one of
`measured`, `plan-visible`, `plan-inferred`, `fitted`, `generated`, `default`. `generated` marks
what an expansion tool (§11) or a video clip supplied: a hypothesis the critic does not
grade against the photograph. That is the plan's
provenance labelling extended to everything, and it is what the trust rules of the
hybrid plan (§4 there) were trying to express per stage: a `measured` value with high
confidence beats a `plan-inferred` one, a `fitted` value records what it was fitted to.

Relations: `on` (boat on water, flowers on meadow), `in` (region in terrain), `beside`,
`behind` (the plan's off-frame list), `bounded_by` (meadow bounded by forest band),
`lines` (trees line the quay), `moored_to`, `excluded_from` (vegetation excluded from
water). The Distribution Synthesis rules `align_to_water_classes`, `moor_fraction`,
`paint_vegetation_in_water`, `class_paint_regions` are relations written as config
today.

### Statistical nodes

A Population is not a list of positions. It is a generative model and a seed:

```
Population "meadow flowers"
  region: TerrainRegion "subalpine meadow"
  species: {Castilleja 0.45, Pedicularis 0.35, Lupinus 0.20}   (fitted; plan-visible names)
  density: lambda(x) = 4.2/m2 * exp(0.8*f_edge(x) - 1.1*f_slope(x) + 0.5*f_wet(x))  (fitted)
  clustering: Thomas process, parent 0.05/m2, radius 1.8 m                           (fitted)
  size: lognormal(0.32 m, 0.25)                                                      (measured)
  seed: 1130049400
  draw: 61,204 instances within 60 m                                                 (artifact)
```

The instances are a draw from the model, regenerated on demand. The client can draw
more or fewer by distance; the agent can redraw and let the judge pick; a second
photograph of a similar meadow can reuse the fitted model as a prior.

### Storage and provenance

`scene_graph.json` in the run directory, one file, versioned by a schema number like
the plan. Every edit is appended to `scene_graph.log.jsonl`: node, attribute, old,
new, who (stage or agent round), why (one sentence, or the judge issue it answers).
The `PipelineContext` stays as the artifact store; `ContextKey` values become the
geometry handles the graph points at, so nothing is copied. The plan (schema v2) is
already most of the node vocabulary (place, camera, sun, atmosphere, horizon, water,
terrain, landmarks, species, objects, motion, behind, acceptance); phase 0 is mostly a
mapping, not an ontology.

## 3. Existing methods as tools

A tool is a stage with a contract: inputs and outputs named in graph terms, a JSON
schema for its parameters, a declared device and memory need, a duration estimate, a
cache key over its inputs, and determinism given inputs and seed. Nothing inside a
tool calls a language model; only perceive and critique tools do. The pipeline's
cache and `--rerun` logic already give most of this; the contract makes it explicit
enough for an agent to plan with.

| group | tools (from today's stages) | reads | writes | agent-callable |
|---|---|---|---|---|
| perceive | caption; plan read; depth; panorama regions and typing; object labelling (open-vocab); lighting estimate; photo masks per class | photo, panorama | node hypotheses with `measured` / `plan-*` evidence | once each, rarely again |
| world (fixed) | panorama generation and supersampling; skybox; foreground inpainting; panorama depth and calibration | photo, plan.horizon, plan.behind | the panorama and its depth | no: run once; one budgeted regeneration with a changed prompt at most |
| build | height map; region map; terrain mesh and texture; treeline and far-field backdrops; species asset (atlas, mesh, variants); object completion (image-to-3D); structure placeholders; scene assembly; material maps; video generation and animation | graph nodes | geometry handles | yes, with parameters from the graph |
| expand | outpaint a frame-cut or occluded object; novel views of a completed crop; outpaint a panorama region for a second perceive pass; extend a surface sample to a seamless texture (§11) | a node, its crop or region, the Style node | an image and a mask of what was generated; attributes with `generated` evidence | yes, by priority: visible from the camera path first |
| finish | tone and colour fit per node mask; relighting of delit textures by the fitted sun; photo-guided detail near the capture view; client render settings from the Light and Medium nodes (§8) | render and photo masks, graph | material and light parameters with `fitted` evidence | yes, after the content is right |
| observe | check render (capture camera, rig views, orbit); DINO and colour metrics; gates (floating, sunk, oversize, seams); mask comparison per class and depth band; luminance by node mask | scene, photo | measurements attached to nodes | yes, cheap |
| critique | structured VLM read of a render against the photo and the acceptance list, every issue targeted at a node | renders, graph | annotations on nodes | yes, budgeted |
| fit | exposure fit (by class weight); population fit (new, §5); sun fit (new: elevation and azimuth from lit faces of known geometry); camera height fit (two renders, pick) | graph, observations | `fitted` attributes | yes |

The panorama stays fixed because it is the world prior everything else is tied to:
regenerating it moves every measured node. The agent may ask for one regeneration with
a changed prompt when the critique says the world itself is wrong (the horizon, what
lies behind), and that costs its whole budget for the round.

## 4. The agent loop

```
perceive   run the perceive tools once; build the graph: plan layer, measured layer
hypothesise agent reads the graph and the photo; fills gaps; resolves plan-vs-measured
           conflicts with a stated reason; sets each population's model prior
build      the runner derives the tool list from the graph (what changed, what reads it)
           and runs it; unchanged tools hit the cache
observe    check renders and measurements, attached to nodes
critique   structured findings on nodes (kind, severity, evidence crop, suggested edit)
revise     agent edits the graph: attributes, representation choices, fit requests;
           never positions of individual instances, never code
stop       acceptance list satisfied, or no improving edit in two rounds, or budget out
```

What the agent decides is a graph edit, and the set of legal edits is the set of
things a tool can consume: a sun elevation, a camera height, a representation choice
(ribbon or species band for a forest), a population's prior, a species' form, an
exposure weight. What it cannot do is move a flower, write a shader or change config.
The contrast with 3DCodeVerse is exactly there: their refine loop edits code and the
scene drifts with the coder's taste; ours edits a typed model whose every value has a
provenance and a consumer.

Rebuilds are graph-driven: a changed node invalidates the tools that read it, the
same dependency idea the pipeline's cache already uses, but keyed by node rather than
by stage order, so a sun change reruns lighting, exposure and the check render and
nothing else.

One orchestrator, not a swarm. Workers exist only for volume (a critique per view, a
mask QA per crop) and report structured results to the orchestrator; they never edit
the graph.

**A trace the last round would have produced.** Rainier, round 1: critique says "the
mid-ground meadow is a dark flat strip" targeted at `TerrainRegion meadow`. The agent
asks the observe tool for luminance on that node's mask, render against photo: 0.06
against 0.31. It reads the exposure fit's inputs, sees the fit's sample is 80 %
`Population flowers` pixels, and edits `ExposureFit.weights` to weigh ground nodes,
not population nodes. Rebuild: Exposure Fit and the check render. Round 2: the meadow
reads 0.24; the critique moves on to the snow patch, which is a `TerrainRegion snow`
node with `albedo: delit 0.77`, lit by the same fit. The treeline fix of yesterday is a
one-line rule in this frame: a `Backdrop treeline` is built for any forest band whose
species has no `SpeciesAsset`, and the agent never had to find it.

## 5. Populations fitted to the photograph

This is the part that changes how the meadow is made.

**Model.** For a Population over region R, with ground point x on the terrain grid:

- density `lambda(x) = lambda0 * exp(sum_k w_k f_k(x))`, with features `f_k` the terrain
  already knows: elevation band, slope, distance to water, distance to trail or road,
  distance to the forest edge, region type, a wetness proxy from the delit albedo. The
  feature-affinity code in `distribution_synthesis/features.py` computes these today
  and applies them through a KL threshold and a floor; here they are weights that are
  fitted.
- clustering as a Thomas process (parent density, cluster radius, offspring mean), which
  is what "clumps of paintbrush" means; the judge's "evenly spaced, toy-like" is the
  absence of it.
- species mix `pi` over the plan's named species, size as lognormal per species,
  orientation prior (upright, or along a bank line for boats).

Ten to fifteen parameters per population.

**Fitting**, two terms:

1. Where the detector sees individuals (the near cone, flowers within 15 m, trees to
   150 m), the inhomogeneous point process likelihood of the detected positions,
   unprojected onto the grid. For the Poisson part it is closed form; the clustering
   parameters come from the pair-correlation of the detections. This is what
   `bloom_max_boost` and recall scaling approximate by hand.
2. Beyond detection range, agreement between the photograph's class masks and the
   rendered class-ID buffer of a draw: pixel fraction per class per depth band and
   per azimuth bin, plus colour statistics per band (the delit panorama gives the
   photograph's side; the check renderer's card rasteriser gives ours in seconds).
   Simulation-based: CMA-ES or a Bayesian optimiser over the parameters, 30 to 100
   draws, each a few seconds. The fit's residual by band is itself an observation on
   the node.

**What it replaces.** `class_max_paint_radius_m`, `class_falloff_exponent`,
`class_density_scale`, `feature_affinity` thresholds, `bloom_*`, `class_min_full_
density_radius_m`: the density model subsumes them. `photo_veto_classes`,
`paint_vegetation_in_water`, `align_to_water_classes`, `moor_fraction` become
relations on the graph (`excluded_from`, `moored_to`) that the draw respects. Where a
rule encodes a real prior (flowers thin beyond 15 m because the detector cannot see
them, not because they are absent), it becomes the prior's mean, not a cap.

**Beyond flowers.** Trees in a forest band (density against slope and edge distance,
which decides where the band is a band and where it is a ribbon), rocks on scree,
boats moored along a bank line (a one-dimensional point process on the `WaterBody`
edge), grass cover (already a field; it gets the same model and the same fit).

**Validation.** Fit on the left half of the photographed cone, predict the right half's
mask fractions per band; against the hand rules on the three scenes with the existing
sheet (DINO layout, colour, judge), and the count of judge issues that say regular,
repeated, evenly spaced or toy-like.

## 6. What semantics fixes on day one

Before any agent loop, the graph alone retires a class of bug:

- Exposure Fit weighs ground nodes, not population nodes (Rainier's dark meadow).
- A forest band's representation follows its species' asset, not a distance rule
  (Paris's bank).
- The near drop floor, water exclusion and bank alignment read nodes, not plan
  strings and region integers.
- The sun's elevation is checked against lit faces of `measured` geometry before it is
  trusted (Shark Fin's 2 deg sun): a fit tool, not a trust rule.
- Judge issues target nodes, so the history records which node regressed between
  rounds, not which sentence.
- The Unity client reads the graph for class-wise behaviour (no shadows on small
  vegetation, wakes on boats, sway on populations), which it does today by name
  matching.

## 7. Models and where they run

| role | needs | calls per scene | choice |
|---|---|---|---|
| orchestrator (hypothesise, revise) | the best vision reasoning there is; structured output; long context over the graph | 5 to 20, with images | Opus |
| critique per view | consistent structured reading; many calls | 20 to 100 | Haiku or Sonnet; local Qwen3-VL as the volume option |
| mask QA, crop verification during fitting | yes/no on hundreds of crops | 100 to 1000 | local Qwen3-VL, or none (metrics only) |
| judge (final grade) | the same reading every time | 1 to 3 | whichever the bake-off says |
| plan read | once per photograph | 1 | Opus (as now, $0.08 to $0.10) |

Opus is the better planner, and the planner is where quality is decided; the gap
between models on "which node is wrong and what edit fixes it" is the gap that
matters. The second machine is for volume, not for the top-level reasoning: critique
over every view of every round, mask checks inside the population fit, nightly sweeps,
and free iteration while developing the loop. It is a Mac Studio with 256 GB of unified
memory, which sets the server and the model: no CUDA, so vLLM is out, and llama.cpp with
Metal serves GGUF weights behind the same OpenAI-compatible endpoint. 256 GB holds
Qwen3-VL-235B-A22B at 4-bit (about 140 GB of weights, 22 B active per token) with room
for a 32 k context, or the dense Qwen3-VL-32B at 8-bit (35 GB) when a smaller, steadier
reader is wanted. `scripts/local-models/setup-mac.sh` installs llama.cpp, downloads
either model with its vision projector, starts the server and times a vision call;
`scripts/local-models/check-endpoint.sh` on this box verifies the endpoint and prints the
config lines. The plan client's `_openai` provider path already speaks that protocol, so
every role gets a `provider:` line in config and nothing else changes.

Decide with a bake-off, not a belief: the 30-odd verdicts in `scratch/check/*/
*/verdict.json` were written by Haiku; score the same sheets with Sonnet, Opus and the
local model, measure agreement on issue targets and severities against a hand reading,
and give each role the cheapest model whose disagreement does not change the agent's
next edit.

The Mac cannot take the CUDA-only work: Object Completion (40 min per scene), the FLUX
plates and atlases, the image-to-3D generators and video generation stay on this box's
two cards. The split is that this box runs the pipeline and the Mac serves the
vision-language models, and the two never contend for a GPU.

Cost: a round is 5 to 20 orchestrator calls at 10 to 20 k tokens with images (about
$0.3 to $0.6 each) plus cheap critique; $3 to $10 per scene for a three-round loop,
against the hybrid plan's $1 and the harness's $14 to $17. The spend ledger, the
per-run cap and `--no-api` from the hybrid plan apply unchanged.

## 8. Rendering: the photograph as the target for how the scene is drawn

Photorealism is a second fit against the image, run after the content is right, node by
node. Today the only photometric fit is Exposure Fit: two scalars for the whole scene,
capped at x3, weighed by whatever the near field holds. Shark Fin hit the cap and stayed
dark. The graph lets rendering be fitted per node, because every node has a mask in the
photograph and a mask in the render.

**Finish tools** form a new group beside build and fit:

- **Tone and colour per node.** For each node with both masks, a bounded tone curve and
  chroma shift fitted on the mask, applied as material parameters so it survives the
  orbit, not as a post filter on one view. Snow that renders grey and a meadow that
  renders flat are each corrected on their own mask.
- **Light as a fit, not a plan read.** Sun elevation and azimuth from the lit faces of
  known geometry; sky dome colour from the photograph's sky; haze from luminance against
  depth, with the plan's value as the prior. The grazing-sun failure on Shark Fin is this
  fit missing.
- **Relit textures.** The delit albedo we already produce, lit by the fitted sun, with
  shadows following the Light node.
- **Photo-guided detail.** Where the camera path stays near the capture view, the
  photograph's own detail is projected onto near geometry as a view-dependent layer that
  fades with angle from the capture ray. The panorama layer already does this for the
  far field.
- **Client realism from the graph.** Shadows, ambient occlusion, reflections on water
  bodies and volumetric haze are Unity settings driven by the Light, WaterBody and Medium
  nodes. The critique reads the client's frames, not pyrender's, as the hybrid plan asked.

A Cycles path reading the same graph stays an option for hero stills. The gate: luminance
and chroma per node class within 10 % of the photograph on the capture view, measured by
the observe tools we have.

## 9. Style: paintings, concept art and other non-photographs

A painting works because the graph separates what is in the scene from how it is drawn.
Today the two are fused: every stage assumes a photograph and photorealism is the target.
Three things already survive the switch: the panorama layer is image-conditioned, so the
outpainted world carries the brushwork and palette; most texturing is cut from the image,
so cards and baked assets inherit the style; and the judge is relative to the input, not
to realism. Depth estimation degrades gracefully on perspective-consistent paintings and
video models animate paintings well.

**What breaks.** Delighting a painted sky and relighting it with a physical sun destroys
the thing the painting is for. Painted light is often non-physical: several light
directions, no shadows, a glowing horizon with no sun, and the fits would report nonsense
with confidence. Ukiyo-e, Chinese landscape, medieval work, much folk art and most concept
art use overlap and scale rather than perspective, so metric depth has nothing to stand
on. Procedural flowers with photographic textures dropped into a painted field look
pasted on.

**What the graph adds.**

1. **A Style node** at scene level with evidence from the image: palette, stroke scale,
   edge treatment, outline presence, whether perspective and lighting are physical, and a
   confidence. The perceive tools fill it and a vision model reads it well.
2. **A render mode chosen by Style.** Photo: §8. Painted: no delighting, the Light node
   marked non-physical and the client set to painted or no shadows, and a stylisation pass
   in the client fitted to the image (stroke scale, edges, quantisation). The toon-shading
   accident from the earlier rounds becomes a knob.
3. **Style-conditioned textures.** Generated textures take the painting as style
   reference, so a procedural flower is painted the way the field is. Image-to-3D
   generators already return stylised geometry from stylised crops.
4. **A pictorial-depth tool** as the alternative to metric depth: layer order from
   occlusion, scale and ground line, giving a relief world of layered surfaces and
   populations rather than a terrain. The world kind on the graph (§13) selects it.
5. **The critic graded in style.** The question becomes whether the render belongs in
   this painting, not whether it agrees with a photograph. DINO and colour error already
   behave on stylised images.

Everything from Impressionism through concept art is in reach, and concept art matters
most because it is how fantasy scenes arrive. Works with no consistent space, Cubism and
its relatives, have nothing to build.

## 10. Assets: 3D model quality by selection, not by one generator

TRELLIS is a reasonable candidate for compact hero objects and the wrong single answer
for vegetation, which is most of what we build. Quality comes from how assets are
selected and verified more than from which generator is picked. `config.yaml` already
records the experience: SAM 3D Objects is the default because it bakes the photograph
onto one coherent surface (the Paris hulls kept their lettering) but returns foliage as a
relief sheet. TRELLIS.2 is used for flower, plant and bush because it gives real stems
textured all round, but it repaints from its prior (a boat became a stick, a person a
slab), varies wildly across seeds, produces fragment soup and marked the Rainier flowers
metallic 0.74. Trees go to Tree-D. The procedural species forms exist because neither
does thin repeated structure. Three generators chosen by hand per class, and no
asset-level measurement: the judge only sees the whole frame.

**Per class.**

- **Vegetation populations.** Parametric procedural templates fitted to the photograph,
  textured from delit crops, cards as the far LOD. Image-to-3D models are at their worst
  on thin, translucent, repeated geometry, and a population needs a mesh that instances
  cheaply. The template set grows over time, which is generation, not an asset library.
- **Rigid things** (boats, benches, vehicles, rocks). SAM 3D where the photograph sees
  most of the object, a generative model for the unseen side. Multi-view conditioning
  helps more than model choice: complete the crop, generate three or four consistent
  views, then reconstruct. Hunyuan3D 2.1 with multi-view input and TRELLIS.2 both take
  this path and both produce PBR.
- **Landmarks.** Geometry from the panorama and depth plus a generated or structured
  mass, not a single-image object model. The placeholder path stays and its fit improves.
- **Materials.** A generator's metallic and roughness are never trusted. Priors by kind
  (vegetation metallic 0, roughness 0.5 to 0.8), with Material Maps overriding.

**The bake-off.** Asset generation becomes a build tool that draws candidates from the
applicable generators and seeds, then:

1. Rejects on geometry: thickness, connected components, hole area, face budget. This
   catches sheets and soup before anyone looks.
2. Renders each survivor under the fitted sun from the capture view and scores it against
   the cutout with mask IoU, DINO and colour error, all of which `scene-check` has.
3. Shows the critic a turntable sheet next to the crop. Haiku is cheap enough per asset.
4. Keeps the winner as a SpeciesAsset or Thing node with its generator, score and
   evidence. A candidate that beats nothing falls back to a card and is reported as a gap.
5. Caches vetted assets by kind, so a second meadow does not regenerate lupines.

The first experiment is the asset sheet itself, run over the three existing generators
on the Rainier flowers and the Paris boats. Right now there is no number that says which
one is better.

## 11. Expansion: outpainting and completion as one tool

Expansion is one tool with several uses, replacing the single hard-wired Object
Completion stage. Today completion runs once, on a fixed class list, under a FLUX budget,
before anyone knows what matters. An agent that owns the graph knows which expansions the
camera will actually see.

- **Cut by the frame.** A boat half out of frame or a clipped crown. Outpaint the crop,
  then re-measure size and footprint from the completed object.
- **Occluded.** A bench behind a person, a facade behind a tree. The current FLUX Fill
  path, chosen per node rather than per class.
- **Unseen sides.** Novel views of a completed crop feed the reconstructors multi-view
  input, which is where asset quality comes from (§10).
- **Beyond the cone.** The panorama does this for the backdrop. The same tool on panorama
  regions feeds a second perceive pass, so populations and landmarks exist behind the
  camera and the orbit has something to look at.
- **Surface extension.** Seamless ground plates and facade textures from a small sample,
  which the near-ground plate does once today.

Everything the tool produces is a hypothesis and the graph says so: `generated` is an
evidence class beside `measured` and `plan-inferred`. The critic does not penalise
disagreement with the photograph there, and the agent knows that content behind the
camera is cheaper to revise than content in front of it. The Style node conditions every
call, so a painted scene is expanded in paint. The contract is a region on an image, a
prompt built from the node, optional reference views; it returns an image and a mask of
what was generated.

The build agent calls it when a node's evidence is incomplete and the node is visible
from the planned camera path. The fixed budget becomes a priority queue: hero objects and
frame-edge cuts first, far populations never.

## 12. Motion and interactivity: the graph as the client's runtime model

Motion is in the graph today only as a word from the plan. It should be attributes with
evidence like everything else:

- **Populations** carry a sway field (amplitude, frequency, wind direction), fitted from
  the generated video's optical flow on the population's mask.
- **Water bodies** carry a flow field and wave state (direction, speed, amplitude) from
  the same flow.
- **Instances** carry travel as a relation with a speed: along the water body, along the
  road.
- **The Medium** carries cloud drift and haze motion.

Video generation becomes a perceive tool: a clip from the photograph, flow per node mask,
motion attributes with `generated` evidence. Video billboards stay for what cannot be
meshed (crowds, flames, spray). The critic grades the orbit or a short clip rather than a
still, and gains temporal issues: popping, static water, grass that does not move.

**Interactivity** is where the graph pays most. The client receives the graph and the
geometry handles, holds population models rather than instance lists, and draws
instances on demand by distance. That gives LOD for free and lets the scene extend past
the photographed cone from the plan's `behind` nodes. Each node carries affordances from
a short closed set (walkable, collides, floats, climbable, boardable, emits_light,
casts_shadow) that the client maps to physics and behaviour instead of matching class
names. A query over the graph answers "what is under this ray" with a node, so labels,
dialogue and gameplay sit on the same semantics. A runtime edit (moving a boat) is a
graph edit and is logged like the agent's.

## 13. Taxonomy: structural types, open kinds, closed affordances

The node table in §2 is a landscape list, and a closed list is exactly what read a plane
tree as a conifer. The taxonomy is two-level. A small closed set of structural types that
tools are written against; an open vocabulary of kinds supplied by the plan and the
perceive tools; and a closed set of affordances that the client and the tools consume. A
scene also carries a world kind (`terrain`, `relief`, `interior`) that selects the tool
set, and the Style node of §9.

| structural type | what it is | landscape kinds | city kinds | fantasy kinds | tools that consume it |
|---|---|---|---|---|---|
| Surface | ground you stand on | meadow, scree, beach, snow, forest floor | street, plaza, quay, lawn | ash plain, glowing moss | height map, region map, terrain texture, populations |
| Mass | a volume with a footprint | cliff, boulder, hill, the Eiffel Tower as a named Mass | building, tower, pier, monument | floating island, castle, giant trunk | structure placeholders, image-to-3D, pictorial relief |
| Linear | a path or line | river bank, trail, fence, ridge | street, rail, bridge deck, canal, power line | causeway, chain, root | the Linear Structures stage, travel paths |
| Thing | one singular instance | boat, bench, person, sign | car, bus, lamp post, kiosk | airship, dragon, statue | the asset bake-off, expansion |
| Population | many of one kind under one model | flowers, grass, trees, rocks | people, cars, street trees, pigeons | spores, crystals, lanterns | population fit and draw |
| Light | a light source | sun, sky, moon | street lights, lit windows | a second sun, a glowing crystal, an aurora | sun fit, exposure, client lights |
| Medium | what fills the air or water | air, haze, fog, water volume | smog, steam | mist, magic haze | haze fit, water shader, volumetrics |
| Camera, Backdrop, Style | the same in every world | | | | calibration, backdrops, render mode |

**Rules.** The structural type is assigned by the plan with a fallback from the kind's
words; a kind no tool handles falls back to a card or placeholder and is reported as a
gap, never silently rebinned. A city is Masses with facade and floor attributes, Linears
for the street network, and populations of vehicles along roads and people along
sidewalks. Fantasy is the same types with invented kinds: species assets generated from
description rather than photographs, lights that are not the sun, and relations that
break physics on purpose (a Mass with no ground contact). Interiors are the one regime
the pipeline cannot do at all today, because the height map assumes ground; they are a
world kind that waits for its own tool set.

## 13a. What exists (built overnight, 2026-11-22)

The backbone of phases 0 to 5 is in the tree, tested on the cached runs and one real Rainier round. Nothing
is committed.

| piece | where | state |
|---|---|---|
| scene graph | `server/pipeline/scene_graph/graph.py` (nodes, evidence, relations, edit log), `export.py` (graph from a cached run, CPU, ~1 s), `scripts/scene-graph.py export\|show\|html\|validate <run or Scene>` | exports and validates on all three runs; HTML viewers in `scratch/graph/<Scene_>/` |
| edit vocabulary | `server/agent/knobs.yaml`: 44 knobs, each a node attribute mapped to config keys and/or plan fields, with range, rerun stage and cost | the agent may only edit these |
| runner | `server/agent/runner.py`: edits -> derived config.yaml + patched plan (original kept) -> `main.py run --rerun <earliest stage>`; GPU-busy guard | used by the loop |
| critique | `server/agent/critique.py`: issues targeted at graph nodes; Anthropic or any OpenAI-compatible server; `scripts/scene-check.py --judge` now uses it when the run has a graph | Rainier critic1: every issue landed on a node |
| orchestrator + loop | `server/agent/orchestrator.py`, `loop.py`, `scripts/agent-loop.py`: round 0 observes, each round proposes (Opus, ~$0.08 a call), applies, reruns, re-exports, checks; stops on budget, patience or the model | dry run proposes sensibly; the real round `agent1` ran end to end in 24 min: Opus switched the flowers to the fitted model (clusters of 10 at 0.8 m, x1.6, 6 species variants); judge 0.23 -> 0.24, DINO 0.60 -> 0.48, i.e. not an improvement, and the loop's comparison now needs the critic to move by more than its noise (0.05) before it decides |
| Population Fit | `server/pipeline/population_fit/`, config entry before Scene Generation: log-linear Poisson intensity in terrain features with a detection-effort offset, ridge search with a flat fallback on a failed left/right hold-out, Thomas clustering from exemplar spacing; writes `population_models`; the draw replaces painted instances per class when `class_overrides.<class>.enabled` | Rainier flowers: held-out gain +0.38 at ridge 10, 6.5 k drawn within 22 m; Paris boats: water edge -0.69, held-out +1.7; draw off by default |
| local models | `scripts/local-models/setup-mac.sh` (on the Mac Studio), `check-endpoint.sh` (here) | untested on hardware |

Not built: finish tools (§8), the expansion tool (§11), motion attributes and the client
side (§12), the Style node beyond a placeholder (§9), the asset bake-off (§10).

Night of 2026-11-23, Paris on the Mac Studio (Qwen3-VL-235B Q4_K_M, 52 tok/s, a plan read
in 32 s): the plan read, the judge and the orchestrator all run there, and `config.yaml`
now points the Scene Plan stage at it. The Paris rebuild from Panorama Lighting took 74 min
and came out blank (DINO 0.10): the Qwen plan filed the Eiffel Tower under "building", so
Structure Placeholders built a 115 m block at 191 m in front of the landmark card. Fixed
(a placed card of any structure class, or the same name, counts as the structure) and
rebuilt: DINO 0.79, dE 11, judge 0.17 then 0.29 on the same scene (the local judge's
noise). Two agent loops ran, five rounds in all, none an improvement: the local
orchestrator enabled the boat and tree fits expecting passengers and detail (the fit only
moves synthetic instances, and the knob docs now say so), and a 15 m boat height made 56 m
barges. Also fixed on the way: the loop never passed the Mac's address to its judge,
Population Fit ignored classes the agent enabled, the judge scored zeros on an off-shape
plain-JSON reply (strict json_schema fails intermittently on llama-server; replies are now
validated, unwrapped, retried and kept raw), a graph re-export dropped `pop_boat` after
the fit drew 3 boats, and the Mac idle-slept mid-loop (connection timeouts; the serve
script now runs llama-server under `caffeinate -i`, which the Mac has to pick up). The
Qwen plan is thinner than Opus's: water width, level and banks, haze and the place name
come back null, and the species changes between reads (willow 8 m, plane 15 m).

Afternoon of 2026-11-23, the turn from knobs to actions. The knob loop was a config tuner
with the shape of an agent: every action was a stage parameter, so applying one meant
rerunning the pipeline from that stage (10 to 70 min), nothing could place or move an
object, and the objective was a DINO score that barely moved across edits the judge called
major. Decided with the user: the three model calls that matter go to Claude (plan read
Opus, fixer Opus, judge Sonnet, which scored the same Paris render 0.04 twice with an
identical issue list where the Mac's judge gave 0.17 and 0.29); the Mac is a fallback and a
bulk-cheap-calls box. Built: `scripts/scene-snapshot.py` (a built run frozen under
`output/snapshots/<Scene>/<tag>/` with its scores, commit and config, restored in place, so
animation and the agent start from it without a rebuild; `hybrid-run.sh --snapshot`);
`server/agent/editor.py` (actions place, move, scale, rotate, remove, swap_mesh,
set_atmosphere, set_sun on the scene.json the renderers draw; a position is world,
bearing+distance, or a photo box cast against the terrain and water meshes, which
reprojects to the same photo point to three decimals; applied in seconds, rendered with
`render-scene.py --scene`, no pipeline); the `Scene Edits` stage after Exposure Fit, which
replays `<run>/scene_edits.json` so later stages, reruns and the archive keep the agent's
accepted actions; `scripts/scene-edit.py` (list, preview, render, accept, undo);
`server/agent/fixer.py` + `scripts/fix-loop.py`, the issue-driven loop: the judge's issues
(now each with a `photo_box`) are the state, the fixer sees the photograph, the render and
both cropped to the issue's region plus the placed objects and the plan's inventory, and
answers with one to three actions; the loop renders and asks the judge with FOCUS on that
issue (crops of the region, `resolved / evidence / regressions`) and keeps or reverts.
Paris rebuilt with the Opus plan and judged by Sonnet: DINO 0.84, acceptance 2/7, issues
with boxes for the missing tour boat (severe), flat water, the left bank, the missing Pont
d'Iéna, the quays, the tower, the sky and the light. The first live proposal (one Opus
call, $0.05) was the right one: move `thing_19` to the boat's photo box on the water and
scale it to 7 m; it applies on the CPU (36 m, bearing -8). The bridge was missing because
the night's placeholder rule let the Eiffel card stand in for any structure within 20 deg;
narrowed to the same class, the same name or a generic plan category. Still to run on the
GPU: the live loop with renders and focused verdicts. Next: the first-pass gates (an
inventory review after Scene Plan and segmentation, a lighting and layout review before
generation), then content actions (place-from-photo through object completion,
regenerate a surface's texture).

Evening of 2026-11-23, the first gate and two findings. The first-pass gate is built:
`server/pipeline/inventory_review/`, the stage "Inventory Review" right after Object
Labeling. Sonnet sees the photograph with every reviewable detection boxed and numbered,
the clean photograph, each box's class and phrase and the plan's inventory, and decides
keep / drop / relabel per box with a reason; plan items no box covers come back as
`missing` with a photo_box, which the fixer's prompt now carries. A drop demotes the
detection to indeterminate the way the size veto does. On Paris it reviewed 50 boxes for
$0.06: eleven drops (the foreground tour boat was seven boxes, two full-width strips, a
duplicate of the bridge, a hotel box over the trees), one relabel (a "tree" sitting on
the white cruise boat), two missing (the Pullman block, the left houseboat). That is why
the judge called the tour boat missing: the first pass built its fragments. The second
finding is Rainier's black mid-ground, the judge's severe issue since band27: not the
geometry (the render's depth is a smooth ground at 30-100 m) and not the paint (the
terrain texture holds the meadow and snow there) but the light. The exposure fit, solved
on the near field, halved sun and ambient against the bright flower cards, and delit
albedo under a 20-degree sun with half ambient is black. A new editor action `set_light`
(sun_scale / ambient_scale multipliers) is the move for that, in the fixer's vocabulary.
Queued for the night, unattended: Paris rebuilt from Inventory Review (label and
snapshot opus2), the fix loop on it (fix1, 12 rounds, $10 cap), a check and snapshot of
the result, then the same loop on Rainier from its opus1 baseline.

Night of 2026-11-23, the loops run and what they found. Paris rebuilt from Inventory
Review (snapshot opus2): DINO 0.774 against opus1's 0.840, layout 0.580, judge 0.04. The
gate's decisions were right, but the one tour-boat box it kept was rebuilt from its
reference view into a small rusty work boat, so the judge's severe issue stayed "tour
boat missing"; the gate removes fragments, it does not make the kept object better. The
first live fix loop (fix1 on Paris) ran three rounds and kept nothing, for reasons that
were the loop's own: the fixer's listing printed object ids as eight-character prefixes
and the editor matched full ids only, so rounds 2 and 3 were refused outright; round 1
swapped the boat's mesh (the swap kept height, so the length doubled) and asked for a
16 m tall boat because the vocabulary had no length, and the judge saw a slab across
the frame. Fixed (commit 4e37b50): id prefixes resolve; scale and place take length_m;
a swap keeps a long object's length; scaling keeps the footprint centre, which matters
for the structure placeholders, whose vertices carry world coordinates (the Pont d'Iéna
was listed at "0 m, bearing -91"); the listing gives each object's length x height and
the bearing and distance of its world centre; the prompt says to try set_light before
giving up on a black region. The three loops then ran cleanly on the fixed code, 15
attempts in all, $0.90, none kept: Shark Fin (5 attempts, judge 0.0 -> 0.0) placed three
rock meshes for the missing headland, which the judge called dark wedges, gave up on the
cliff and the beach as content, and lifted sun and ambient without taking the fin out of
silhouette; Rainier (6 attempts, 0.2 -> 0.2) tried set_light at 1.15 / 0.8, too timid to
tell, gave up on the meadow, the trail and the snow as content, and placed a fir for the
missing left one from the only tree mesh, which the judge rejected as a pale sparse
spire; Paris (4 attempts, 0.04 -> 0.04) swapped and sized the boat, gave up on it as
content, and moved and scaled the Pont d'Iéna into its photo box, which lifted the
whole-frame judge from 0.04 to 0.2 with no regression and was still thrown away because
the focus verdict only asked "resolved". Two conclusions. The loop's mechanics now hold
(propose, apply, render, judge, keep or revert, about 40 s and $0.07 a round), and the
keep rule was too strict: the judge's focus now also answers `improved`, and a clear
step toward the photograph with no fall in the whole-frame score is kept so the next
round builds on it (commit 8d1c4c2; a Paris rerun, fix3, tests it). Second, the issues
that remain are content, in every scene: the tour boat with its crowd, the stone arches,
the golden cliff, the beach, the meadow with its snow patches and trail, the full dark
fir. No move, scale or swap of the meshes a run already holds can answer them, which is
what the proposal's content actions are for (place-from-photo through object completion,
regenerate a texture, a terrain patch) and the next thing to build. Also for the
morning: Iceland's opus1 render is broken in the first pass (white cut-outs where the
hikers were removed, a large green block at the right, a floating grey line), and the
Exposure Fit's halving of sun and ambient on Rainier is a first-pass decision for the
lighting review gate rather than something to undo in the loop.

How to run in the morning:

```
# the graph of any cached run
scripts/scene-graph.py show "Mount Rainier"            # or export / html / validate
# a dry run: one Opus proposal, no GPU
scripts/agent-loop.py "Mount Rainier" --label dryN --rounds 1 --baseline critic1 --dry-run
# the loop (medium edits only; add --allow-expensive for reruns before Object Completion)
scripts/agent-loop.py "Mount Rainier" --label agentN --rounds 3 --max-usd 5 --baseline critic1
# the fitted flowers instead of the painted ones (one medium rerun, ~20 min)
#   set Population Fit.class_overrides.flower.enabled: true in config.yaml, then
scripts/hybrid-run.sh "Mount Rainier" --from "Population Fit" --label popfit1
# the Mac: run setup-mac.sh there, then here
scripts/local-models/check-endpoint.sh http://<mac>.local:8000/v1 server/samples/Paris.jpg
# freeze / restore a built scene
scripts/scene-snapshot.py save Paris --tag opus1 --check opus1; scripts/scene-snapshot.py list
scripts/scene-snapshot.py restore Paris --tag opus1
# the editor: what is placed, try actions, render, accept
scripts/scene-edit.py Paris --list
scripts/scene-edit.py Paris --actions a.json --label try1 --render [--accept]
# the issue-driven loop (renders, no pipeline rerun)
scripts/fix-loop.py Paris --label fix1 --rounds 12 --max-usd 10 [--baseline opus1] [--dry-run]
scripts/replay-stage.py output/<uuid> "Inventory Review"          # the gate's decisions, boxes.png, review.json (dry; --save to apply)
scripts/hybrid-run.sh Paris --from "Inventory Review" --label opus2 --snapshot opus2
# the fix loop on a built scene (needs a scene-check label as the round-0 baseline), then freeze it
scripts/fix-loop.py Paris --label fixN --rounds 12 --max-usd 10 --baseline opus2
scripts/scene-check.py Paris --label fixN --grass --judge && scripts/scene-snapshot.py save Paris --tag fixN --check fixN
```

Reports land in `scratch/agent/<label>/report.md`; every round's proposal, derived config
and logs sit beside it; the check sheets in `scratch/check/<Scene_>/<label>-r<n>/`.

## 14. Phases

| phase | weeks | builds | done when |
|---|---|---|---|
| 0 graph | 1 | `scene_graph.json` schema; exporter from existing runs (plan layer + measured layer from the context); an HTML viewer; the edit log | the three scenes export; every judge issue in the history maps to a node |
| 1 tools | 1 to 2 | tool registry over the existing stages with contracts; a runner that executes a tool list derived from the graph; graph-keyed invalidation | band27 replays byte-identical through the runner from its graph |
| 2 observe + first consumer | 1 | observe tools (views, orbit, metrics, luminance and mask fractions by node); structured critique targeted at nodes; Exposure Fit by node class; the sun fit | Rainier's meadow luminance within 20 % of the photograph on its mask; Shark Fin's cliffs lit |
| 3 populations | 2 | the density and clustering model, the two-term fit, the draw; flowers and grass on Rainier first | held-out mask fractions beat the hand rules; judge regularity complaints gone |
| 4 the loop | 2 | the orchestrator with budgets and stop rules; graph edits as the only action; the trace in the history | on all three scenes, judge and DINO at or above band27 / final2 / final3 at $10 per scene or less |
| 5 local models | 1 | vLLM on the second machine; `provider:` per role; the bake-off | agreement numbers on the sheet; roles assigned |
| 6 more populations | 2 | trees in bands, rocks, boats; camera height fit | Paris's quays and boats from fitted populations, not vetoes |
| 7 assets | 2 | the asset sheet; the bake-off over SAM 3D, TRELLIS.2, Hunyuan3D 2.1 and the procedural forms; material priors by kind; the vetted cache (§10) | the Rainier flowers and the Paris boats scored per asset; the winner beats today's choice on the sheet |
| 8 expand + finish | 2 | the expansion tool with `generated` evidence and the priority queue; per-node tone and colour fit; relit textures; critique on client frames (§8, §11) | frame-cut boats whole; luminance within 10 % per class on the capture view; the critic reads Unity frames |
| 9 motion + client | 2 | motion attributes from video flow; affordances; the client drawing populations from the model; the graph query (§12) | the orbit graded as a clip; the client walks a Rainier draw it made itself |
| 10 style + worlds | 2 | the Style node; render modes; style-conditioned textures; the pictorial-depth tool; the structural types (§9, §13) | one painting and one city built end to end without a new node type |

Phases 7 to 10 are each worth doing on their own and can run in any order once phase 4
stands; 7 is the second-biggest quality gain after 3. Phases 0 to 2 are worth doing even if the loop never runs: they are the semantics of
§6. Phase 3 is the biggest single quality gain on the table for the meadow. Phase 4 is
where the agent earns its name.

## 15. Risks

- **Agent thrash and cost.** Bounded by budget, by the edit vocabulary (only what a
  tool consumes), by cached tools (a repeated edit costs nothing) and by the stop rule.
  The log makes every round auditable.
- **Nondeterminism.** Seeds live in the graph; a tool given the same graph produces the
  same bytes; the only nondeterministic step is the model call, and its output is a
  logged edit.
- **Judge noise.** Haiku's readings of band23 and band27 differ on the flowers'
  regularity, not on anything measured. Structured critique on nodes, several views,
  DINO and colour kept beside it, and the bake-off of §7.
- **Over-building the graph.** Start from the plan's vocabulary plus measured
  attributes; no ontology work; a node type is added when a tool needs it.
- **The fitting objective.** The photograph's masks are the labels, and the detector
  misses the far meadow; the second term uses colour statistics per band where masks
  run out, and the fit reports its residual rather than hiding it.
- **Generated content taken as measured.** The evidence class is the guard: the critic
  does not grade outpainted ground against the photograph, and the population fit never
  trains on expanded flowers. A Style node that misreads a photograph as painted would
  turn the physical fits off, so the photo path is the default and the Style read
  carries a confidence.
- **GPU contention.** The tool contract carries device and memory; the runner schedules
  one GPU tool at a time per card. Yesterday's loss of a Shark Fin run to two pan
  renders on the same card is the case in point.

## 16. Not in scope

Replacing WorldGen for the panorama; rewriting the Unity client (it consumes the graph,
nothing more); multi-agent swarms; code-writing agents (3DCodeVerse's path, assessed layer by layer in
`hybrid-scene-plan.md` §1); training or fine-tuning models.

Also deferred: interiors, a world kind the height map cannot serve and that needs its own
tool set; and works with no consistent space to build, Cubism and its relatives.
