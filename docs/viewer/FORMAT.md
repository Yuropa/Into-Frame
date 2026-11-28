# Web scene bundle, format 1

What `scripts/export-web-scene.py` writes for one run and the website's three.js viewer
(`docs/viewer/`) loads. One directory per scene: `docs/viewer/scenes/<slug>/`.

## Frame

Everything is in the **GL / three.js frame**: the pipeline's server frame (left-handed,
+X right, +Y up, +Z forward) with **X negated** (render-scene.py's `_MIRROR`). +Y is up.
The capture camera's forward is given explicitly (`camera.forward`); it is close to +Z.
Units are metres and seconds.

## manifest.json

```json
{
  "format": 1,
  "name": "Paris", "slug": "paris", "run": "<run id>", "round": "r3",
  "built": "<ISO 8601 time>",
  "camera": {"position": [x, y, z], "forward": [x, y, z], "up": [x, y, z],
             "fovYDeg": 51.2, "aspect": 1.5, "eyeHeightM": 1.6, "walkRadiusM": 3.0},
  "sky": {"image": "sky.jpg", "yawDeg": 0.0, "driftDegS": 0.0},
  "env": "env.glb",
  "water": "water.glb",                       // or null
  "models": {"<key>": "models/<key>.glb"},
  "instances": "instances.json",
  "motion": {
    "wind": {"dir": [x, z], "speedMs": 2.4, "gustiness": 0.4},           // unit vector, GL frame
    "water": {"kind": "river", "flowPeriodS": 2.0, "waveSpeedMs": 3.0, "swellPeriodS": 10.0},  // or null
    "vegetation": {"<kind key>": {"amplitudeRad": 0.05, "frequencyHz": 0.5, "heightM": 9.0}}
  },
  "light": {"sunDir": [x, y, z], "sunColor": [r, g, b], "sunIntensity": 2.3,
            "ambient": [r, g, b], "exposure": 1.0},
  "bytes": 12345678
}
```

`light` (an addition to the first draft of this format): `sunDir` is the unit vector **toward**
the sun in the GL frame; `sunColor` is sRGB 0..1; `sunIntensity` is render-scene's directional
light intensity (pyrender units, the scene's sun intensity times the fitted sun scale);
`ambient` is the linear-RGB ambient render-scene uses (sky mean x fitted ambient scale x tint).
A viewer matching render-scene: diffuse = albedo x (sunColor x sunIntensity x max(n.l, 0) / pi-free
Lambert as pyrender's PBR with roughness 0.9, metallic 0) + albedo x ambient, then sRGB encode.

## sky.jpg

The skybox panorama, equirectangular, 2:1, at most 4096 wide. A view direction `d` in the GL
frame maps to a pixel as render-scene's `render_sky` does in the server frame:

```
lon = -atan2(d.x, d.z) - radians(sky.yawDeg + sky.driftDegS * t)   // server-frame longitude
lat = atan2(d.y, hypot(d.x, d.z))
u   = fract((lon + pi) / (2 pi))        // 0 at the image's left edge
v   = 0.5 - lat / pi                    // 0 at the image's top row
```

The sky is drawn behind everything (infinitely far).

## env.glb

One glTF 2.0 binary with one node per environment piece. `node.extras`:

| key | meaning |
|---|---|
| `role` | `terrain`, `terrainOverlay`, `backdrop`, `treeline`, `formation`, `waterApron`, `wall` |
| `unlit` | true: draw the base colour texture as is (photographed light), no lighting |
| `overlayOrder` | terrainOverlay only: 1, 2, ... draw after the terrain in this order |

`wall`: the terrain's steep faces (quay walls, cliffs), which the terrain mesh leaves out and render-scene
draws as their own mesh, textured from the side with a tiling stone or rock texture (REPEAT wrap).
The terrain overlays still cover those faces' vertices; their weights there are what render-scene draws.

Materials carry `KHR_materials_unlit` when unlit. **Terrain overlays** (the near-ground plate
layers and the panorama layer) share the terrain's POSITION / NORMAL accessors, carry their own
TEXCOORD_0 and a COLOR_0 whose **alpha is the overlay's blend weight**; their material is
`alphaMode: BLEND`. Draw them after the terrain, transparent, with vertex colours, depth test on,
depth write off and a polygon offset (factor -1, units -4 x overlayOrder) so they win over the
terrain they duplicate. Plate overlays sample their texture with REPEAT wrapping (their UVs run
past 1).

The terrain's material is opaque. Unlit pieces (backdrop shells, the water apron) carry no
NORMAL. A material with `extras.foliage: true` is a leaf or petal card: render-scene lights it
from both sides with a wrapped diffuse (`max(n.l * 0.5 + 0.5, 0)`, the normal flipped toward the
viewer on back faces); a viewer may do the same or light it two-sided.

Cutout materials (`alphaMode: MASK`) carry `alphaCutoff: 0.02`, the threshold render-scene's shader
discards at; honour the value given (three.js `alphaTest`), not glTF's usual 0.5: a backdrop shell's soft
lower edge lies between the two and covers the far ground behind it. Their textures (PNG) have the colour
carried out under the discarded texels, so filtering and mipmaps do not draw a dark rim.

## water.glb

The water sheet (one node, `extras: {"role": "water", "kind": "river" | "sea"}`), unlit, its
texture the photographed water. Attributes beyond POSITION / NORMAL / TEXCOORD_0:

- `_CURRENT` (VEC2 float): the current in m/s (x, z of the bundle's frame); at sea the swell's
  direction toward the shore at a slow drift (0.15 of the field's unit speed).
- `_SHORE` (SCALAR float): metres from the vertex to the nearest dry ground.

A viewer animating it runs `docs/viewer/water.glsl`, the same code render-scene's shader runs:
it defines `water_colour(uv, worldPos, cameraPos, current, shore, isSea, t, wind, surf, boatA,
boatB, sunDir, sunColour)` given a host function `vec3 water_tex(vec2 uv, float mipBias)` that
samples the water's texture in linear RGB. `wind` = (dir.x, dir.z, speedMs, 25); `surf` =
(`motion.water.waveSpeedMs`, `swellPeriodS`); `boatA` / `boatB` are mat4s whose column i is
the i-th boat underway (up to four): (x, z, heading x, heading z) and (`lengthM`, `beamM`,
`speedMs`, 1), zero for none. The photograph stays put; the shader lays a moving surface over
it (current-borne ripples, wind waves, the swell and surf at sea, the boats' wakes and bow
waves). The texture has the photographed wakes of the boats underway already painted out.

Both attributes are absent when the scene has no flow field (bundles before round 4 carried
`_FLOW`, the flow in uv per second, instead of `_CURRENT`; a viewer treats a missing `_CURRENT`
as still water).

## models/<key>.glb

Each placed model once, in its own local frame (already mirrored), one mesh. Cards and plants
cut out by alpha use `alphaMode: MASK` (cutoff 0.02, see env.glb), double-sided; their normals are bent
(already in the file). Materials may carry `KHR_materials_unlit`. `baseColorFactor` carries the
exposure fit's per-model colour gain.

## instances.json

```json
{
  "static": {
    "<key>": {"matrices": [[16 floats, column-major, GL frame], ...],
              "sway": "<kind key>" | null, "castShadow": true, "billboard": false}   // billboard: a camera-facing quad (render-scene turns it to the capture camera; the matrices already face it)
  },
  "movers": [
    {"key": "<key>",
     "base": {"position": [x, y, z], "rotationDeg": [x, y, z], "scale": [x, y, z]},
     "motion": {...}}
  ]
}
```

`sway` names the entry of `manifest.motion.vegetation` this model bends by; a viewer reproducing
render-scene bends each vertex in world space (h = world y above the instance's origin):

```
f = clamp(h / heightM, 0, 1.5); s = dot(origin.xz, wind.dir)
gust = 1 - g + g (0.5 + 0.5 sin(2 pi (s - wind.speedMs t) / 25))         // g = wind.gustiness
osc = sin(2 pi frequencyHz t + hashPhase(origin.xz))
bend = amplitudeRad gust (0.45 + 0.55 osc)
off = wind.dir * bend * f^2 * heightM
flutter (leaves, across the wind, per vertex): lp = hashPhase(vertex position)
  off += perp(wind.dir) * amplitudeRad 0.22 gust f^2 heightM
         * sin(2 pi 3.1 frequencyHz t + lp) (0.6 + 0.4 sin(2 pi 0.7 frequencyHz t + 1.7 lp))
p.xz += off;  p.y -= 0.5 |off|^2 / max(h, 0.1)
```

**Movers** are posed each frame from `base` and `motion`. The pose's matrix is
`T(position) * Ry(rotationDeg.y) * Rx(rotationDeg.x) * Rz(rotationDeg.z) * S(scale)` (three.js
Euler order `'YXZ'`), all in the GL frame (the server's yaw and roll are negated). `motion`:

- `{"type": "floating_moored", "bob": B}`: position and rotation stay at `base`, plus the bob.
- `{"type": "floating_underway", "speedMs", "path": [[x, z], ...] (GL frame), "yawOffsetDeg", "bob": B,
  "lengthM", "beamM"}`: the boat is at arc length `min(speedMs t, path length)` along `path`
  (the viewer sails it again after a 20 s pause; render-scene's clips are shorter); its yaw is
  `degrees(atan2(tangent.x, tangent.z)) + yawOffsetDeg` (both in the GL frame), its y `base.y`.
- The bob `B = {"heaveM", "rollDeg", "pitchDeg", "periodS", "phase"}` with `w = 2 pi / periodS`:
  `y += heaveM sin(w t + phase)`, `rotation.z += rollDeg sin(w t + phase + 1.3)`,
  `rotation.x += pitchDeg sin(0.8 w t + phase + 0.4)`.

## What the viewer reads, and asks for

The viewer (`docs/viewer/viewer.js`) follows this format as written. Additions it would use:

- `camera.groundY`: the terrain's height under the capture point. The viewer keeps the eye at the
  capture camera's height above the ground while walking and in VR (`local-floor`: the rig puts
  the headset's eye there); today it ray-casts the terrain for it, which fails where the capture
  point stands over water or a bridge with nothing under it. `eyeHeightM` in Paris's manifest is
  0.0017, not an eye height: the viewer does not rely on it.
- `static.<key>.billboard: true` is not drawn facing the camera yet; say if any bundle uses it.
- `camera.far`: the farthest geometry's distance (the viewer uses 6000 m otherwise).
- A walkable mask or radius per scene if 3 m is wrong for one (a bridge's railing, a cliff edge).

Test hooks (for headless screenshots; harmless otherwise): `?shot=1` holds the page's load event
on `/__hold` until the first frame of the loaded scene is drawn and copies the canvas into an
image; `&t=<seconds>` sets the animation time; `&walk=KeyW,KeyA` walks 3 s with those keys;
`&ui=0` hides the interface. `scratch/r3/viewer/serve.py` is the test server that answers
`/__hold` and `/__release`.
