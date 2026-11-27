# Animation plan

Status 2026-11-27. Scope: make the three test scenes (Paris, Shark Fin, Mount Rainier) move believably —
plants in the wind, boats on the river, water, surf and sky — with the motion taken from a generated
video of the photograph wherever the video can be trusted, and from a motion agent everywhere else.
Every check renders three videos: the orbit spin, the orbit spin with a wireframe overlay, and a 10 s
clip from the capture camera with the scene animating.

## 1. Where things stand

| Piece | State |
|---|---|
| Video Generation (LTX-2) | Conditions on the **equirectangular panorama**. The clip leaves the scene within a second: Paris grows a tripod and a different river, Shark Fin a foreground of grass, Rainier a gravel path. Motion measured from it is mostly noise. |
| Video Object Extraction | SAM2 tracks each detection's mask through that clip → `object_motion_{i}` (centroid track). Inherits the drift. |
| Object Motion Classification | Rule: centroid drift < 0.6 × bbox diagonal → stationary + sway(amplitude, frequency); else moving + velocity/acceleration unprojected at fixed depth. No idea of water, flow or kind of object. |
| Scene Animation | Writes `sway` (vegetation meshes only, distance-gated), `physics.velocity` for movers, video textures for billboards, static colliders. One random wind axis. |
| Renderer (`render-scene.py`) | Static frame. Sway, physics, water and sky motion are ignored. |
| Unity client | WindSway (bone rotation), rigid-body velocity hand-off, billboard video. No water flow, no sky motion, no boat behaviour. |
| What is 3D | Paris: 15 trees, 3 boats, 1 card (Eiffel Tower); 46 photographed occluders (boats, moored barges, bridge, mid-distance trees) stay painted in the terrain plate. Shark Fin: the fin and a reef billboard. Rainier: 3,491 flowers, 769 + 131-card conifers. |

Two conclusions: the motion source has to be rebuilt before anything is measured from it, and anything
that moves has to be 3D — a boat painted into the plate cannot sail.

## 2. Motion model

Every scene-graph node gets a `motion` block, chosen from a small set of motion types. The type says how
the client/renderer moves it; the parameters come from the video when measured, else from the agent's
physical prior.

| Type | For | Parameters | Driven by |
|---|---|---|---|
| `static` | buildings, bridges, rocks, terrain | — | — |
| `wind_sway` | trees, shrubs, flowers, grass | stiffness class (conifer / broadleaf / willow / herb / grass), amplitude, frequency, gust response | shared wind field |
| `floating_moored` | moored barges, houseboats | heave, roll, pitch amplitudes and period; mooring line limits drift | water surface |
| `floating_underway` | tour boats, ferries | speed (m/s), heading = along the flow line ± (upstream/downstream), lane offset, wake | flow field + water surface |
| `flowing_water` | rivers | flow map (direction, speed m/s), ripple scale | flow field |
| `surf` | sea shore | wave period, direction, run-up line, foam band | shoreline + swell direction |
| `spray` | wave impacts on rocks | burst period, height, location | surf phase |
| `cloud_drift` | sky | direction, angular speed, layer split | wind aloft |
| `flag` / `animal` / `vehicle` | later | — | — |

People stay removed.

## 3. The pieces to build

### 3.1 Motion source: learn the motion from the generated video
The video is the richest record we have of how each kind of thing moves. It drifts away from the
photograph, but what we take from it is how things move, not where they are, and that survives drift:
- **Better clips**: generate from the photograph (perspective, the framing LTX-2 is trained on) as
  well as the panorama, 5–7 s, several seeds. Each clip gets a fidelity curve (DINO similarity of every
  frame to frame 0); measurements are weighted by it, and the early part of a drifting clip still counts.
- **Tracks, not positions**: SAM2 keeps each object's identity through the clip (it already does,
  `object_motion_{i}`), so an object is followed even as the scene slides. Dense optical flow (RAFT;
  Farneback on CPU for tests) inside each track's mask, **minus the flow of the ring of background around
  it**, gives the object's own motion with the camera/scene drift cancelled.
- **What is measured, per object**:
  - moves freely or is anchored: net displacement relative to its surroundings vs. oscillation about a
    fixed point (an anchored plant's base stays put while its tip moves; a boat's whole mask translates);
  - for anchored things, the **range of motion**: amplitude as a function of height in the mask (base 0,
    tip max → bend profile and stiffness), the dominant frequency and its spread, whether neighbours move
    together (gust coherence → wind field scale and speed), the direction of the sway (→ wind direction);
  - for free movers: speed, heading, turning, bobbing (vertical oscillation of the mask), and what it is
    on (the region under its track: water, ground, sky).
- **Per region**: water (flow direction and speed field, ripple frequency), sky (cloud drift direction and
  angular speed), surf (wave period, run-up), spray (burst period).
- **Constraints inferred, then generalised**: a free mover over water for the whole track → *stays on the
  water surface, heads along the flow lines, keeps a lane* (follows the curve of the river in 3D even
  where the video never showed that stretch); a free mover over ground → stays on walkable terrain;
  an anchored plant → bends about its base within the measured range. These become the motion blocks'
  constraints, and the measured numbers their parameters; the same species / kind elsewhere in the scene
  (outside the photograph's view, or in the backdrop) gets the measured distribution, not a guess.
- The video is a source of **motion priors only**: distributions and constraints (how far, how fast, how
  often, anchored or free, on what surface), never a copy of its frames or trajectories. The scene's own
  motion is synthesised from those priors, so it loops, varies per instance and covers what the clip
  never showed.

### 3.2 Motion Director (agent)
A new stage after Scene Graph's inputs are known (scene plan + inventory + measurements), one structured
LLM call per scene, cached like the Asset Director:
- input: photograph, the scene graph's nodes (class, size, position, on water / on ground), the video
  measurements and inferred constraints with their confidence, the fidelity curve; its job is to name
  the motion type where the measurement is ambiguous and to carry a measured behaviour to the objects
  the video never showed;
- output per node: motion type, parameters with units, and a reason; per scene: wind direction and
  strength (Beaufort-like), water flow direction (downstream sign), cloud drift, swell direction/period;
- rules it is held to (checked in code, not trusted): speeds within physical ranges per class (tour boat
  1.5–5 m/s, river flow 0.2–2 m/s, cloud angular speed ≤ 0.3°/s), a measured value with good confidence
  wins over a prior, wind / cloud / ripple directions agree unless the agent gives a reason.

### 3.3 Water
- **Flow field**: from the water mesh's outline in world space, a 2D flow map baked over the water sheet:
  along-channel direction from the outline's medial axis (a river), speed highest mid-channel and zero
  at the banks, bent around obstacles (boats, piers, the fin) — the baked obstacle/flow field offered
  earlier. Sign and magnitude from §3.1/§3.2. For the sea (Shark Fin): a swell direction toward the beach
  plus a weak longshore drift.
- **Surface**: two-phase flow-map advection of the photographed water layer and of a ripple normal map
  (the standard flow-map technique: two copies offset half a period, cross-faded, so the texture never
  stretches). Wind ripples on top, aligned with the wind.
- **Surf (Shark Fin)**: wave fronts as a scrolling foam band along the shoreline distance field, period
  from the video or 8–12 s; the spray plume as a procedural periodic burst (particles), its period and
  height taken from the video's plume region.

### 3.3a The river's course
A river runs through the scene as one channel. The water outside the photograph's view comes from the
generated surround, which invents water: Paris's sheet has arms either side of the camera, a bay and
a split behind it. When the water is a river (it enters and leaves the terrain), the water region is
rebuilt as one continuous channel: its centreline from the photographed reach, extended smoothly to the
terrain's edges both ways, its width from the photographed reach; invented arms and bays become bank.
The flow field, the boats' courses and the banks all follow from it.

### 3.4 Boats
- Placed on the water surface; heading aligned to the local flow line (or against it for upstream), the
  hull fitted to the photograph as now.
- `floating_underway`: advance along the flow line at the boat's speed, steer to keep its lane offset,
  wrap at the reach's ends (fade out/in at the backdrop) so a 10 s or 10 min view stays populated.
- All boats bob: heave/roll/pitch from the water's wave spectrum at their size (a 28 m tour boat barely
  rolls; a dinghy does).
- Wake: a V foam decal behind underway boats, length with speed. The photographed wake has to leave the
  water layer first: Paris's foam is painted into the photographed water, so in the first animated test
  it stayed put while the boat sailed away. A boat that moves takes its wake out of the plate with it
  (inpainted like the boat itself) and draws its own.

### 3.5 Plants
- Keep the WindSway model, add a **coherent wind field**: gusts as noise moving downwind across the
  scene, so neighbours move together and a gust is seen crossing the meadow.
- Stiffness classes from the species (conifer: low frequency, tip-weighted; broadleaf: trunk sway +
  leaf flutter; willow: long drapes lag; flowers / grass: high frequency, large angle). The species
  director already knows the class; the Motion Director confirms it.
- In the renderer: a vertex bend in the cutout shader (height-weighted displacement) driven by a time
  uniform and the wind field.

### 3.6 Sky
- Split the sky into a clear-sky gradient (the clear-sky model already built for the sky restore) and a
  cloud layer (the difference, with alpha), drift the cloud layer along the measured / agent direction
  at a slow angular speed, cross-faded two-phase so nothing tears at the wrap. The photographed sky is
  the first frame; it drifts from there.

### 3.7 More 3D, fewer painted objects
- Anything with a non-static motion type is always lifted out of the plate and modelled (boats, moored
  barges, mid-distance trees that the camera can resolve), whatever its size or depth; the round-2
  keep-small rule only keeps static things painted.
- Far detections beyond the terrain (Paris's bridge at 377 m) get a backdrop-standing mesh rather than
  staying in the backdrop's picture when they are landmarks.

### 3.8 Rendering and the three videos
- `render-scene.py --animate 10 --fps 24`: build once, then per frame update boat node poses, the plant
  bend uniform, the water texture phase and the sky layer offset; camera at the capture pose, looking
  forward. Writes `animate.mp4` beside `orbit.mp4` and `orbit_wire.mp4`.
- The queue's `spin` step renders all three for every scene.
- `scene.json` carries the same motion blocks so the Unity client can play them (flow map + water shader
  phase, boat follower, wind field, sky layer); the client work follows the renderer's.

### 3.9 Checks
- Motion statistics of the rendered 10 s clip against the generated video, per region (flow direction
  histogram and speed for water / sky / plants), as numbers in `check.json`.
- A judge pass over a strip of frames and a flow visualisation ("does the water flow one way, do the
  boats sail on it, do the trees move like trees").

## 4. Order of work
1. The river's course (§3.3a) and the background discontinuities (separate from animation, visible in every spin): floating cards in the
   sky, gaps and cut edges in the treeline / backdrop, the step in Shark Fin's horizon and cliff top,
   Rainier's cut mountain edge and the hole in the far hills.
2. Lift every movable object (§3.7) and re-run Paris.
3. Renderer time support + flow field + boats + plant bend + sky drift with agent priors → first 10 s
   clips for all three scenes.
4. Motion learning from the video (§3.1): tracks + relative flow per object and region, free vs.
   anchored, range of motion, constraints; the Motion Director consumes them and fills the gaps.
5. Surf and spray for Shark Fin; wakes.
6. Client (Unity) side of the motion blocks.

## 5. Open questions
- Does the client need to loop forever (VR)? The plan assumes yes: every motion is periodic or wraps.
- The plate's painted far objects (beyond the terrain) stay still; is that acceptable at 300 m+?
