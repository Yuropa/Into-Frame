// Into Frame scene viewer: loads a scene bundle (docs/viewer/FORMAT.md) and plays its motion.
// Everything is in the bundle's GL frame (+Y up, right-handed); the camera starts where the
// photograph was taken and may walk a few metres from there.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';

const SCENES = { 'paris': 'Paris', 'shark-fin': 'Shark Fin', 'mount-rainier': 'Mount Rainier' };
const params = new URLSearchParams(location.search);
const SHOT = params.get('shot') === '1';
const $ = (id) => document.getElementById(id);
if (params.get('ui') === '0') document.documentElement.classList.add('no-ui');
if (SHOT) {
  // Errors on the page itself, where a headless screenshot can show them.
  const log = (msg) => { const d = document.createElement('pre'); d.textContent = msg; d.style.cssText = 'position:fixed;left:8px;top:60px;z-index:50;color:#f88;font:11px monospace;white-space:pre-wrap;max-width:90vw'; document.body.appendChild(d); };
  window.addEventListener('error', (e) => log('error: ' + e.message));
  window.addEventListener('unhandledrejection', (e) => log('rejection: ' + (e.reason && e.reason.message || e.reason)));
  const ce = console.error.bind(console), cw = console.warn.bind(console);
  console.error = (...a) => { log('console.error: ' + a.join(' ')); ce(...a); };
  console.warn = (...a) => { log('console.warn: ' + a.join(' ')); cw(...a); };
  const cl = console.log.bind(console);
  console.log = (...a) => { log('log: ' + a.join(' ')); cl(...a); };
}

// ── renderer ────────────────────────────────────────────────────────────────
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: SHOT, powerPreference: 'high-performance' });
} catch (e) {
  fail('This viewer needs WebGL, which this browser or device does not offer.', e);
  throw e;
}
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;   // the photographed colours as photographed
renderer.shadowMap.enabled = false;
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local-floor');
$('stage').appendChild(renderer.domElement);

// The camera rides on a rig: identity on the desktop; in VR the headset moves the camera
// inside the rig and the rig carries it to the capture point, its heading and height.
const rig = new THREE.Group();
rig.name = 'rig';

let quality = 'auto';   // auto | high | low
function applySize() {
  const dpr = window.devicePixelRatio || 1;
  renderer.setPixelRatio(quality === 'high' ? dpr : quality === 'low' ? Math.min(dpr, 0.75) : Math.min(dpr, 1.5));
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  if (camera) { camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix(); }
}

// ── shared animation uniforms ─────────────────────────────────────────────────
// The same names and formulas as scripts/render-scene.py --animate, so the web and the
// offline renders move alike.
const U = {
  animTime: { value: 0 },
  wind: { value: new THREE.Vector4(0, 1, 2, 25) },       // dir.x, dir.z, speed m/s, gust wavelength m
  flowPeriod: { value: 2.0 },
  surf: { value: new THREE.Vector2(3.0, 10.0) },          // wave speed m/s, period s
};

let scene, camera, state = null, loadToken = 0;
const clock = new THREE.Clock();
let playing = true, motionOn = true, simTime = 0;

// ── loading ─────────────────────────────────────────────────────────────────
function progress(text, frac) {
  $('loading').hidden = false;
  $('loading-text').textContent = text;
  $('loading-fill').style.width = `${Math.round(100 * Math.max(0, Math.min(1, frac)))}%`;
}

function fail(msg, err) {
  console.error(msg, err || '');
  const box = $('error');
  box.innerHTML = '';
  const s = document.createElement('strong'); s.textContent = 'Could not show this scene. ';
  const p = document.createElement('span'); p.textContent = msg;
  box.append(s, p);
  box.hidden = false;
  $('loading').hidden = true;
  release();
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

const gltfLoader = new GLTFLoader();
function loadGLB(url, onBytes) {
  return new Promise((resolve, reject) => {
    gltfLoader.load(url, (g) => resolve(g), (ev) => onBytes && onBytes(ev.loaded || 0), (e) => reject(new Error(`${url}: ${e?.message || e}`)));
  });
}

function loadTexture(url) {
  return new Promise((resolve, reject) => new THREE.TextureLoader().load(url, resolve, undefined, () => reject(new Error(`${url}: not found`))));
}

function disposeScene(s) {
  if (!s) return;
  s.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    const ms = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
    for (const m of ms) {
      for (const k of ['map', 'emissiveMap', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap', 'alphaMap']) if (m[k]) m[k].dispose();
      m.dispose();
    }
  });
}

async function loadScene(slug) {
  const token = ++loadToken;
  $('error').hidden = true;
  $('info').textContent = '';
  if (!SCENES[slug]) slug = 'paris';
  $('scene-pick').value = slug;
  document.title = `${SCENES[slug]} — Scene Viewer — Into Frame`;
  const base = `scenes/${slug}/`;
  progress(`Loading ${SCENES[slug]}…`, 0.02);
  let manifest;
  try {
    manifest = await fetchJSON(base + 'manifest.json');
  } catch (e) {
    fail(`The ${SCENES[slug]} scene has not been published yet (${e.message}).`, e);
    return;
  }
  if (token !== loadToken) return;
  const total = Math.max(1, Number(manifest.bytes) || 1);
  const done = new Map();
  const bump = (key) => (n) => { done.set(key, n); let s = 0; for (const v of done.values()) s += v; progress(`Loading ${manifest.name || SCENES[slug]}…`, 0.05 + 0.9 * s / total); };

  const s = new THREE.Scene();
  const st = { slug, manifest, base, sky: null, swayMaterials: [], movers: [], terrain: [], start: null };
  try {
    // Sky first: it is what most of the screen shows.
    st.sky = await buildSky(base + (manifest.sky?.image || 'sky.jpg'), manifest.sky || {});
    s.add(st.sky);
    const env = await loadGLB(base + (manifest.env || 'env.glb'), bump('env'));
    prepareEnv(env.scene, st);
    s.add(env.scene);
    if (manifest.water) {
      const water = await loadGLB(base + manifest.water, bump('water'));
      prepareWater(water.scene, st);
      s.add(water.scene);
    }
    const instances = await fetchJSON(base + (manifest.instances || 'instances.json'));
    const keys = new Set([...Object.keys(instances.static || {}), ...(instances.movers || []).map((m) => m.key)]);
    const models = {};
    await Promise.all([...keys].map(async (key) => {
      const rel = manifest.models?.[key];
      if (!rel) { console.warn(`model ${key} missing from the manifest`); return; }
      try { models[key] = (await loadGLB(base + rel, bump(key))).scene; } catch (e) { console.warn(e.message); }
    }));
    if (token !== loadToken) return;
    placeStatic(s, models, instances.static || {}, st);
    placeMovers(s, models, instances.movers || [], st);
    addLights(s, manifest);
  } catch (e) {
    if (token === loadToken) fail(e.message, e);
    return;
  }
  if (token !== loadToken) { disposeScene(s); return; }

  disposeScene(scene);
  scene = s;
  state = st;
  scene.add(rig);
  xr.calibrate = true;
  setupMotion(manifest.motion || {});
  setupCamera(manifest.camera || {});
  simTime = 0;
  $('loading').hidden = true;
  const m = manifest;
  $('info').textContent = [m.name || SCENES[slug], m.round ? `round ${m.round}` : null, m.built ? `built ${String(m.built).slice(0, 10)}` : null].filter(Boolean).join(' · ');
  renderOnce();
  if (SHOT) {
    // Headless screenshots do not always composite a WebGL canvas: show its pixels as an
    // image on top, then let the page's load event fire.
    const w = params.get('walk');     // test: hold these keys for 3 s (e.g. walk=KeyW)
    if (w) { for (const k of w.split(',')) keys.add(k); for (let i = 0; i < 180; i++) walk(1 / 60); keys.clear();
             console.log(`walked to ${camera.position.toArray().map((v) => v.toFixed(2))}, ${Math.hypot(camera.position.x - state.start.pos.x, camera.position.z - state.start.pos.z).toFixed(2)} m out`); }
    for (let i = 0; i < 3; i++) { simTime = Number(params.get('t') ?? 2.5); renderOnce(); }
    const img = new Image();
    img.onload = () => setTimeout(release, 50);
    img.src = renderer.domElement.toDataURL('image/png');
    img.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;z-index:5;pointer-events:none';
    document.body.appendChild(img);
  }
}

// ── sky ─────────────────────────────────────────────────────────────────────
// A large sphere drawn behind everything; the equirect is looked up per pixel the way
// the offline renderer does (render_sky): in the server frame (x = -x_gl), longitude
// atan2(x, z) minus the sky's yaw, column (lon + pi) / 2pi, row (0.5 - lat / pi) from
// the top. The drift turns the yaw.
async function buildSky(url, sky) {
  const tex = await loadTexture(url);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.ShaderMaterial({
    uniforms: { map: { value: tex }, yaw: { value: THREE.MathUtils.degToRad(Number(sky.yawDeg) || 0) },
                drift: { value: THREE.MathUtils.degToRad(Number(sky.driftDegS) || 0) }, animTime: U.animTime },
    vertexShader: `varying vec3 vDir; void main(){ vDir = position; vec4 p = projectionMatrix * modelViewMatrix * vec4(position,1.0); gl_Position = p.xyww; }`,
    fragmentShader: `
      uniform sampler2D map; uniform float yaw; uniform float drift; uniform float animTime; varying vec3 vDir;
      void main(){
        vec3 d = normalize(vDir); d.x = -d.x;
        float lon = atan(d.x, d.z) - yaw - drift * animTime;
        float lat = asin(clamp(d.y, -1.0, 1.0));
        vec2 uv = vec2(fract((lon + 3.14159265) / 6.2831853), 0.5 + lat / 3.14159265);
        gl_FragColor = texture2D(map, uv);
        #include <colorspace_fragment>
      }`,
    side: THREE.BackSide, depthWrite: false, depthTest: false,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), mat);
  mesh.renderOrder = -1000;
  mesh.frustumCulled = false;
  const at = new THREE.Vector3();
  mesh.onBeforeRender = (r, s, cam) => {
    mesh.position.copy(at.setFromMatrixPosition(cam.matrixWorld));
    mesh.scale.setScalar(Math.min(cam.far || 6000, 6000) * 0.5);
    mesh.updateMatrixWorld();
  };
  return mesh;
}

// ── environment ─────────────────────────────────────────────────────────────
function roleOf(o) {
  let n = o;
  while (n) { if (n.userData && n.userData.role) return n.userData; n = n.parent; }
  return {};
}

function unlitFrom(m, extra = {}) {
  const map = m.map || m.emissiveMap || null;
  if (map) map.colorSpace = THREE.SRGBColorSpace;
  return new THREE.MeshBasicMaterial({ map, color: map ? 0xffffff : (m.color || 0x888888), side: THREE.DoubleSide,
    alphaTest: m.alphaTest || 0, transparent: !!m.transparent, ...extra });
}

function prepareEnv(root, st) {
  root.traverse((o) => {
    if (!o.isMesh) return;
    const info = roleOf(o);
    const role = info.role || 'formation';
    const mat = o.material;
    if (role === 'terrainOverlay') {
      const order = Number(info.overlayOrder) || 1;
      o.material = unlitFrom(mat, { vertexColors: !!o.geometry.attributes.color, transparent: true, depthWrite: false,
        polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -4 * order });
      o.renderOrder = order;
    } else if (info.unlit || role === 'backdrop' || role === 'treeline' || role === 'waterApron') {
      // The bundle's alphaCutoff (0.02, render-scene's cut) keeps a backdrop's soft lower edge.
      o.material = unlitFrom(mat, { alphaTest: role === 'waterApron' ? 0 : (mat && mat.alphaTest > 0 ? mat.alphaTest : 0.5) });
    } else if (mat) {
      mat.side = THREE.FrontSide;
      if (mat.map) mat.map.colorSpace = THREE.SRGBColorSpace;
    }
    if (role === 'terrain') st.terrain.push(o);
    o.frustumCulled = role !== 'backdrop';
  });
}

// ── water ───────────────────────────────────────────────────────────────────
// The photographed water advected along its flow (_FLOW, uv per second), two copies half a
// period apart cross-faded; at sea, surf bands run to the shore (_SHORE, metres).
function prepareWater(root, st) {
  root.traverse((o) => {
    if (!o.isMesh) return;
    const geo = o.geometry;
    const kind = roleOf(o).kind || (o.userData && o.userData.kind) || (st.manifest.motion?.water?.kind) || 'river';
    const flow = geo.attributes._flow;
    const shore = geo.attributes._shore;
    const m = unlitFrom(o.material, { side: THREE.DoubleSide });
    if (!flow) { o.material = m; return; }
    if (!shore) geo.setAttribute('_shore', new THREE.BufferAttribute(new Float32Array(geo.attributes.position.count).fill(400), 1));
    m.onBeforeCompile = (sh) => {
      sh.uniforms.animTime = U.animTime; sh.uniforms.flowPeriod = U.flowPeriod; sh.uniforms.surf = U.surf;
      sh.uniforms.isSea = { value: kind === 'sea' ? 1 : 0 };
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\nattribute vec2 _flow; attribute float _shore; varying vec2 vFlow; varying float vShore;')
        .replace('#include <uv_vertex>', '#include <uv_vertex>\nvFlow = _flow; vShore = _shore;');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', '#include <common>\nuniform float animTime; uniform float flowPeriod; uniform vec2 surf; uniform float isSea; varying vec2 vFlow; varying float vShore;')
        .replace('#include <map_fragment>', `
          #ifdef USE_MAP
            float p0 = fract(animTime / flowPeriod);
            float p1 = fract(animTime / flowPeriod + 0.5);
            vec4 wa = texture2D(map, vMapUv - vFlow * (p0 - 0.5) * flowPeriod);
            vec4 wb = texture2D(map, vMapUv - vFlow * (p1 - 0.5) * flowPeriod);
            vec4 sampledDiffuseColor = mix(wb, wa, 1.0 - abs(2.0 * p0 - 1.0));
            if (isSea > 0.5) {
              float wavelength = max(surf.x * surf.y, 1.0);
              float ph = 6.2831853 * (vShore / wavelength + animTime / max(surf.y, 0.5));
              float crest = pow(0.5 + 0.5 * sin(ph), 8.0);
              float zone = smoothstep(70.0, 4.0, vShore);
              float swash = smoothstep(6.0, 0.0, vShore) * (0.5 + 0.5 * sin(ph - 1.2));
              vec3 lit = sampledDiffuseColor.rgb;
              sampledDiffuseColor.rgb = mix(lit, vec3(0.85, 0.87, 0.9) * max(dot(lit, vec3(0.33)) * 2.2, 0.35),
                                            clamp(crest * zone * 0.75 + swash * 0.35, 0.0, 0.85));
            }
            diffuseColor *= sampledDiffuseColor;
          #endif`);
    };
    o.material = m;
  });
}

// ── objects ─────────────────────────────────────────────────────────────────
function fixObjectMaterial(m) {
  if (!m) return m;
  if (m.map) m.map.colorSpace = THREE.SRGBColorSpace;
  if (m.emissiveMap) m.emissiveMap.colorSpace = THREE.SRGBColorSpace;
  const cutout = m.alphaTest > 0 || m.transparent || (m.userData && m.userData.card);
  if (cutout) {
    // Alpha to coverage with a lower cut: a fir's fine sprays average away in the far mips
    // and fell under 0.5 there, leaving the distant trees as bare trunks.
    m.alphaTest = m.alphaTest > 0 ? Math.min(m.alphaTest, 0.3) : 0.3; m.alphaToCoverage = true; m.transparent = false; m.side = THREE.DoubleSide;
    // And no mips: a spray atlas is a few percent opaque, and every mip level below the
    // first is mostly transparent whatever the cut (the renderer without multisampling
    // drew the distant firs as bare poles).
    if (m.map) { m.map.generateMipmaps = false; m.map.minFilter = THREE.LinearFilter; m.map.needsUpdate = true; }
  }
  return m;
}

// Plants bend in the wind: the vertex shader's projection is replaced by one that pushes
// every vertex above the instance's origin downwind, as render-scene.py's mesh.vert.
function swayMaterial(m, sway) {
  const mm = m.clone();
  mm.userData.sway = sway;
  mm.onBeforeCompile = (sh) => {
    sh.uniforms.animTime = U.animTime; sh.uniforms.wind = U.wind;
    sh.uniforms.sway = { value: new THREE.Vector4(sway.amplitudeRad || 0.05, sway.frequencyHz || 0.5, Math.max(sway.heightM || 1, 0.05), sway.gustiness ?? 0.4) };
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nuniform float animTime; uniform vec4 wind; uniform vec4 sway;')
      .replace('#include <project_vertex>', `
        vec4 swLocal = vec4(transformed, 1.0);
        #ifdef USE_INSTANCING
          vec4 swWorld = modelMatrix * instanceMatrix * swLocal;
          vec4 swBase = modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
        #else
          vec4 swWorld = modelMatrix * swLocal;
          vec4 swBase = modelMatrix * vec4(0.0, 0.0, 0.0, 1.0);
        #endif
        {
          float h = max(swWorld.y - swBase.y, 0.0);
          float tall = max(sway.z, 0.05);
          float f = clamp(h / tall, 0.0, 1.5);
          float s = dot(swBase.xz, wind.xy);
          float gust = 1.0 - sway.w + sway.w * (0.5 + 0.5 * sin(6.2831853 * (s - wind.z * animTime) / max(wind.w, 1.0)));
          float ph = fract(sin(dot(swBase.xz, vec2(12.9898, 78.233))) * 43758.5453) * 6.2831853;
          float osc = sin(6.2831853 * sway.y * animTime + ph);
          float bend = sway.x * gust * (0.45 + 0.55 * osc);
          vec2 off = wind.xy * bend * f * f * tall;
          swWorld.xz += off;
          swWorld.y -= 0.5 * dot(off, off) / max(h, 0.1);
        }
        vec4 mvPosition = viewMatrix * swWorld;
        gl_Position = projectionMatrix * mvPosition;`);
  };
  mm.customProgramCacheKey = () => `sway-${sway.amplitudeRad}-${sway.frequencyHz}-${sway.heightM}`;
  return mm;
}

function meshesOf(root) {
  // Each mesh of a model with its transform relative to the model's root.
  root.updateMatrixWorld(true);
  const inv = new THREE.Matrix4().copy(root.matrixWorld).invert();
  const out = [];
  root.traverse((o) => { if (o.isMesh) out.push({ mesh: o, rel: new THREE.Matrix4().multiplyMatrices(inv, o.matrixWorld) }); });
  return out;
}

function placeStatic(s, models, statics, st) {
  const veg = st.manifest.motion?.vegetation || {};
  const gust = st.manifest.motion?.wind?.gustiness ?? 0.4;
  const tmp = new THREE.Matrix4();
  for (const [key, entry] of Object.entries(statics)) {
    const model = models[key];
    const mats = entry.matrices || [];
    if (!model || !mats.length) continue;
    const sway = entry.sway && veg[entry.sway] ? { ...veg[entry.sway], gustiness: gust } : null;
    for (const { mesh, rel } of meshesOf(model)) {
      let material = Array.isArray(mesh.material) ? mesh.material.map(fixObjectMaterial) : fixObjectMaterial(mesh.material);
      if (sway) material = Array.isArray(material) ? material.map((m) => swayMaterial(m, sway)) : swayMaterial(material, sway);
      const im = new THREE.InstancedMesh(mesh.geometry, material, mats.length);
      for (let i = 0; i < mats.length; i++) {
        tmp.fromArray(mats[i]).multiply(rel);
        im.setMatrixAt(i, tmp);
      }
      im.instanceMatrix.needsUpdate = true;
      im.computeBoundingSphere();
      im.frustumCulled = true;
      im.name = key;
      s.add(im);
      if (sway) st.swayMaterials.push(material);
    }
  }
}

function placeMovers(s, models, movers, st) {
  // FORMAT.md, "Movers": a pose from base (position, rotationDeg in three's 'YXZ' order,
  // scale, GL frame) and the motion, set every frame.
  for (const mv of movers) {
    const model = models[mv.key];
    if (!model) continue;
    const obj = model.clone(true);
    obj.traverse((o) => { if (o.isMesh) o.material = Array.isArray(o.material) ? o.material.map(fixObjectMaterial) : fixObjectMaterial(o.material); });
    const holder = new THREE.Group();
    holder.add(obj);
    const b = mv.base || {};
    const base = {
      pos: new THREE.Vector3().fromArray(b.position || [0, 0, 0]),
      rot: (b.rotationDeg || [0, 0, 0]).map((v) => THREE.MathUtils.degToRad(v)),
      scale: new THREE.Vector3().fromArray(b.scale || [1, 1, 1]),
    };
    holder.rotation.order = 'YXZ';
    holder.position.copy(base.pos);
    holder.rotation.set(base.rot[0], base.rot[1], base.rot[2]);
    holder.scale.copy(base.scale);
    s.add(holder);
    const motion = mv.motion || {};
    let path = null, cum = null;
    if (motion.type === 'floating_underway' && Array.isArray(motion.path) && motion.path.length >= 2) {
      path = motion.path.map(([x, z]) => new THREE.Vector2(x, z));
      cum = [0];
      for (let i = 1; i < path.length; i++) cum.push(cum[i - 1] + path[i].distanceTo(path[i - 1]));
    }
    st.movers.push({ holder, base, motion, path, cum });
  }
}

function poseMovers(t) {
  for (const mv of state.movers) {
    const { motion, base, holder } = mv;
    let x = base.pos.x, z = base.pos.z, yaw = base.rot[1];
    if (mv.path) {
      const L = mv.cum[mv.cum.length - 1];
      const d = Math.min((motion.speedMs || 2) * t, L);
      let i = 0;
      while (i < mv.cum.length - 2 && mv.cum[i + 1] < d) i++;
      const seg = Math.max(mv.cum[i + 1] - mv.cum[i], 1e-6);
      const f = (d - mv.cum[i]) / seg;
      const a = mv.path[i], c = mv.path[i + 1];
      x = a.x + f * (c.x - a.x);
      z = a.y + f * (c.y - a.y);
      yaw = Math.atan2(c.x - a.x, c.y - a.y) + THREE.MathUtils.degToRad(motion.yawOffsetDeg || 0);
    }
    const bob = motion.bob || {};
    const w = 2 * Math.PI / Math.max(bob.periodS || 3, 0.5);
    const ph = bob.phase || 0;
    holder.position.set(x, base.pos.y + (bob.heaveM || 0) * Math.sin(w * t + ph), z);
    holder.rotation.set(
      base.rot[0] + THREE.MathUtils.degToRad(bob.pitchDeg || 0) * Math.sin(0.8 * w * t + ph + 0.4),
      yaw,
      base.rot[2] + THREE.MathUtils.degToRad(bob.rollDeg || 0) * Math.sin(w * t + ph + 1.3));
  }
}

function addLights(s, manifest) {
  // As render-scene.py lights it (FORMAT.md, "light"): a directional sun in pyrender's
  // units, which are three's (both divide Lambert by pi), and a flat ambient that pyrender
  // adds undivided -- three's ambient is divided by pi, so it is given pi times over.
  const L = manifest.light || {};
  const sunColor = new THREE.Color().setRGB(...(L.sunColor || [1, 0.97, 0.92]), THREE.SRGBColorSpace);
  const sun = new THREE.DirectionalLight(sunColor, L.sunIntensity ?? 2.2);
  const dir = new THREE.Vector3().fromArray(L.sunDir || [0.4, 0.8, 0.3]).normalize();   // toward the sun
  sun.position.copy(dir.multiplyScalar(100));
  s.add(sun, sun.target);
  const amb = new THREE.Color().fromArray(L.ambient || [0.3, 0.32, 0.36]);   // linear
  s.add(new THREE.AmbientLight(amb, Math.PI));
  renderer.toneMapping = THREE.LinearToneMapping;     // exposure only, no curve
  renderer.toneMappingExposure = Number(L.exposure) || 1.0;
}

// ── motion ──────────────────────────────────────────────────────────────────
function setupMotion(motion) {
  const w = motion.wind || {};
  const dir = new THREE.Vector2().fromArray(w.dir || [0, 1]);
  if (dir.lengthSq() < 1e-9) dir.set(0, 1);
  dir.normalize();
  U.wind.value.set(dir.x, dir.y, Math.max(w.speedMs || 2, 0.5), 25);
  const wa = motion.water || {};
  U.flowPeriod.value = wa.flowPeriodS || 2.0;
  U.surf.value.set(wa.waveSpeedMs || 3.0, wa.swellPeriodS || 10.0);
}

// ── camera and walking ──────────────────────────────────────────────────────
const look = { yaw: 0, pitch: 0 };
const keys = new Set();
const pad = { active: false, x: 0, y: 0, id: null };
const raycaster = new THREE.Raycaster();
const DOWN = new THREE.Vector3(0, -1, 0);

function setupCamera(c) {
  const aspect = window.innerWidth / window.innerHeight;
  if (camera) rig.remove(camera);
  camera = new THREE.PerspectiveCamera(c.fovYDeg || 60, aspect, 0.05, c.far || 6000);
  rig.add(camera);
  const pos = new THREE.Vector3().fromArray(c.position || [0, 1.7, 0]);
  const fwd = new THREE.Vector3().fromArray(c.forward || [0, 0, 1]).normalize();
  state.start = { pos: pos.clone(), fwd: fwd.clone(), radius: c.walkRadiusM ?? 3.0, eye: c.eyeHeightM ?? null };
  state.ground = sampleGround(pos.x, pos.z, state.start.radius);
  // The capture camera's height above the ground under it, kept while walking.
  const g = groundAt(pos.x, pos.z);
  state.start.eyeAbove = g === null ? null : pos.y - g;
  resetView();
  applySize();
}

function resetView() {
  if (!state) return;
  if (renderer.xr.isPresenting) { xr.calibrate = true; return; }
  rig.position.set(0, 0, 0); rig.rotation.set(0, 0, 0); rig.updateMatrixWorld();
  const { pos, fwd } = state.start;
  camera.position.copy(pos);
  look.yaw = Math.atan2(fwd.x, fwd.z);
  look.pitch = Math.asin(THREE.MathUtils.clamp(fwd.y, -1, 1));
  applyLook();
}

function applyLook() {
  const cp = Math.cos(look.pitch);
  const dir = new THREE.Vector3(Math.sin(look.yaw) * cp, Math.sin(look.pitch), Math.cos(look.yaw) * cp);
  camera.up.set(0, 1, 0);
  camera.lookAt(camera.position.clone().add(dir));
}

function castGround(x, z) {
  if (!state || !state.terrain.length) return null;
  raycaster.set(new THREE.Vector3(x, 1e4, z), DOWN);
  raycaster.far = 2e4;
  const hit = raycaster.intersectObjects(state.terrain, false)[0];
  return hit ? hit.point.y : null;
}

// The ground under the walkable disc, sampled once at load on a small grid (a ray against
// the whole terrain mesh per step stalled Rainier's 160k triangles), read bilinearly.
function sampleGround(cx, cz, radius) {
  const n = 11, half = radius + 1.0, step = (2 * half) / (n - 1);
  const h = new Float32Array(n * n);
  let any = false;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const y = castGround(cx - half + i * step, cz - half + j * step);
    h[j * n + i] = y === null ? NaN : y; any = any || y !== null;
  }
  return any ? { cx, cz, half, step, n, h } : null;
}

function groundAt(x, z) {
  const g = state && state.ground;
  if (!g) return castGround(x, z);
  const fx = (x - (g.cx - g.half)) / g.step, fz = (z - (g.cz - g.half)) / g.step;
  if (fx < 0 || fz < 0 || fx > g.n - 1 || fz > g.n - 1) return castGround(x, z);
  const i = Math.min(Math.floor(fx), g.n - 2), j = Math.min(Math.floor(fz), g.n - 2);
  const u = fx - i, v = fz - j;
  const a = g.h[j * g.n + i], b = g.h[j * g.n + i + 1], c = g.h[(j + 1) * g.n + i], d = g.h[(j + 1) * g.n + i + 1];
  const y = (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v;
  return Number.isFinite(y) ? y : castGround(x, z);
}

function walk(dt) {
  if (!state || !camera || renderer.xr.isPresenting) return;
  let fx = 0, fz = 0;
  if (keys.has('KeyW') || keys.has('ArrowUp')) fz += 1;
  if (keys.has('KeyS') || keys.has('ArrowDown')) fz -= 1;
  if (keys.has('KeyA') || keys.has('ArrowLeft')) fx -= 1;
  if (keys.has('KeyD') || keys.has('ArrowRight')) fx += 1;
  if (pad.active) { fx += pad.x; fz += -pad.y; }
  if (!fx && !fz) return;
  const speed = 1.6;   // m/s, a stroll
  const sy = Math.sin(look.yaw), cy = Math.cos(look.yaw);
  // Forward is (sin yaw, cos yaw); right is forward x up = (-cos yaw, sin yaw) in this frame.
  const dx = (sy * fz - cy * fx) * speed * dt;
  const dz = (cy * fz + sy * fx) * speed * dt;
  const { pos, radius } = state.start;
  let nx = camera.position.x + dx, nz = camera.position.z + dz;
  const ox = nx - pos.x, oz = nz - pos.z, r = Math.hypot(ox, oz);
  if (r > radius) { nx = pos.x + ox * radius / r; nz = pos.z + oz * radius / r; }
  camera.position.x = nx; camera.position.z = nz;
  const g = groundAt(nx, nz);
  if (g !== null && state.start.eyeAbove !== null) camera.position.y = g + state.start.eyeAbove;
  applyLook();
}

// Pointer look (mouse and touch on the stage).
const stage = $('stage');
let drag = null;
stage.addEventListener('pointerdown', (e) => { drag = { id: e.pointerId, x: e.clientX, y: e.clientY }; stage.setPointerCapture(e.pointerId); });
stage.addEventListener('pointermove', (e) => {
  if (!drag || drag.id !== e.pointerId || !camera) return;
  const k = THREE.MathUtils.degToRad(camera.fov) / window.innerHeight;
  // Drag the world: moving the pointer right turns the view left.
  look.yaw += (e.clientX - drag.x) * k;
  look.pitch = THREE.MathUtils.clamp(look.pitch + (e.clientY - drag.y) * k, -1.45, 1.45);
  drag.x = e.clientX; drag.y = e.clientY;
  applyLook();
});
const endDrag = (e) => { if (drag && drag.id === e.pointerId) drag = null; };
stage.addEventListener('pointerup', endDrag);
stage.addEventListener('pointercancel', endDrag);

// Touch pad.
const padEl = $('pad'), knob = padEl.querySelector('.vpad__knob');
function padMove(e) {
  const r = padEl.getBoundingClientRect();
  let x = (e.clientX - (r.left + r.width / 2)) / (r.width / 2);
  let y = (e.clientY - (r.top + r.height / 2)) / (r.height / 2);
  const n = Math.hypot(x, y); if (n > 1) { x /= n; y /= n; }
  pad.x = x; pad.y = y;
  knob.style.transform = `translate(${x * 33}px, ${y * 33}px)`;
}
padEl.addEventListener('pointerdown', (e) => { pad.active = true; pad.id = e.pointerId; padEl.setPointerCapture(e.pointerId); padMove(e); e.stopPropagation(); });
padEl.addEventListener('pointermove', (e) => { if (pad.active && pad.id === e.pointerId) padMove(e); });
const padEnd = (e) => { if (pad.id === e.pointerId) { pad.active = false; pad.x = pad.y = 0; knob.style.transform = ''; } };
padEl.addEventListener('pointerup', padEnd);
padEl.addEventListener('pointercancel', padEnd);

// Keys.
window.addEventListener('keydown', (e) => {
  if (e.target && (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT')) return;
  if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) { keys.add(e.code); e.preventDefault(); }
  else if (e.code === 'Space') { togglePlay(); e.preventDefault(); }
  else if (e.code === 'KeyM') toggleMotion();
  else if (e.code === 'KeyR') resetView();
  else if (e.code === 'KeyQ') cycleQuality();
  else if (e.code === 'KeyH') toggleHelp();
});
window.addEventListener('keyup', (e) => keys.delete(e.code));
window.addEventListener('blur', () => keys.clear());

// ── controls ────────────────────────────────────────────────────────────────
function togglePlay() {
  playing = !playing;
  $('btn-play').textContent = playing ? 'Pause' : 'Play';
  $('btn-play').setAttribute('aria-pressed', String(playing));
}
function toggleMotion() {
  motionOn = !motionOn;
  $('btn-motion').textContent = motionOn ? 'Motion on' : 'Motion off';
  $('btn-motion').setAttribute('aria-pressed', String(motionOn));
}
function cycleQuality() {
  quality = quality === 'auto' ? 'high' : quality === 'high' ? 'low' : 'auto';
  $('btn-quality').textContent = `Quality: ${quality}`;
  applySize();
}
function toggleHelp() {
  const h = $('help'); h.hidden = !h.hidden;
  $('btn-help').setAttribute('aria-expanded', String(!h.hidden));
}
$('btn-play').addEventListener('click', togglePlay);
$('btn-motion').addEventListener('click', toggleMotion);
$('btn-reset').addEventListener('click', resetView);
$('btn-quality').addEventListener('click', cycleQuality);
$('btn-help').addEventListener('click', toggleHelp);
$('scene-pick').addEventListener('change', (e) => { location.hash = e.target.value; });
window.addEventListener('hashchange', () => loadScene(location.hash.slice(1)));
window.addEventListener('resize', applySize);

// ── VR (WebXR immersive-vr) ─────────────────────────────────────────────────
// Guarded throughout: without navigator.xr nothing here runs and the button stays hidden.
const xr = { calibrate: true, turnReady: [true, true], controllers: [] };
const _head = new THREE.Vector3(), _fwd = new THREE.Vector3(), _q = new THREE.Quaternion();

function headWorld(out) { camera.updateMatrixWorld(); return out.setFromMatrixPosition(camera.matrixWorld); }

function xrCalibrate() {
  // The headset's position inside the rig (local-floor: y is the wearer's own eye height)
  // is known once frames arrive. Turn the rig so that the headset's forward is the capture
  // camera's, then move it so that the eye stands at the capture point at its height.
  const { pos, fwd } = state.start;
  const yaw = Math.atan2(fwd.x, fwd.z);
  // Headset's own yaw in the rig (its view direction is -Z).
  _fwd.set(0, 0, -1).applyQuaternion(camera.quaternion);
  const headYaw = Math.atan2(_fwd.x, _fwd.z);
  rig.rotation.set(0, yaw - headYaw, 0);
  rig.position.set(0, 0, 0);
  rig.updateMatrixWorld();
  headWorld(_head);
  rig.position.set(pos.x - _head.x, pos.y - _head.y, pos.z - _head.z);
  rig.updateMatrixWorld();
  xr.calibrate = 0;
}

function xrUpdate(dt) {
  if (!state || !camera) return;
  // Calibrate on the second frame of a session: the headset's pose reaches the camera
  // during the first frame's render.
  if (xr.calibrate) { xr.calibrate = xr.calibrate === true ? 1 : xr.calibrate - 1; if (xr.calibrate <= 0) xrCalibrate(); return; }
  const session = renderer.xr.getSession();
  if (!session) return;
  let mx = 0, my = 0, turn = 0;
  for (const src of session.inputSources) {
    const gp = src.gamepad;
    if (!gp || gp.axes.length < 2) continue;
    const ax = gp.axes.length >= 4 ? gp.axes[2] : gp.axes[0];
    const ay = gp.axes.length >= 4 ? gp.axes[3] : gp.axes[1];
    if (src.handedness === 'right') turn = ax;
    else { mx = ax; my = ay; }
  }
  // Snap turn, 30 degrees, about the head.
  const ti = 0;
  if (Math.abs(turn) > 0.7 && xr.turnReady[ti]) {
    xr.turnReady[ti] = false;
    headWorld(_head);
    const a = THREE.MathUtils.degToRad(turn > 0 ? -30 : 30);
    _q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), a);
    rig.position.sub(_head).applyQuaternion(_q).add(_head);
    rig.quaternion.premultiply(_q);
    rig.updateMatrixWorld();
  } else if (Math.abs(turn) < 0.3) xr.turnReady[ti] = true;
  // Walk, head-relative, within the walk radius and on the ground.
  headWorld(_head);
  if (Math.abs(mx) > 0.15 || Math.abs(my) > 0.15) {
    camera.getWorldDirection(_fwd); _fwd.y = 0;
    if (_fwd.lengthSq() > 1e-6) _fwd.normalize();
    const right = new THREE.Vector3(-_fwd.z, 0, _fwd.x);
    const speed = 1.6 * dt;
    const dx = (_fwd.x * -my + right.x * mx) * speed, dz = (_fwd.z * -my + right.z * mx) * speed;
    const { pos, radius } = state.start;
    let nx = _head.x + dx, nz = _head.z + dz;
    const ox = nx - pos.x, oz = nz - pos.z, r = Math.hypot(ox, oz);
    if (r > radius) { nx = pos.x + ox * radius / r; nz = pos.z + oz * radius / r; }
    rig.position.x += nx - _head.x; rig.position.z += nz - _head.z;
    _head.x = nx; _head.z = nz;
  }
  // Keep the eye at the capture camera's height above the ground under it.
  const g = groundAt(_head.x, _head.z);
  if (g !== null && state.start.eyeAbove !== null) {
    const want = g + state.start.eyeAbove;
    rig.position.y += (want - _head.y) * Math.min(1, dt * 6);
  }
  rig.updateMatrixWorld();
}

function setupControllers() {
  const factory = new XRControllerModelFactory();
  const rayGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -1)]);
  for (let i = 0; i < 2; i++) {
    const ray = renderer.xr.getController(i);
    const line = new THREE.Line(rayGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5 }));
    line.scale.z = 2;
    ray.add(line);
    rig.add(ray);
    const grip = renderer.xr.getControllerGrip(i);
    grip.add(factory.createControllerModel(grip));
    rig.add(grip);
  }
}

async function setupVR() {
  const btn = $('btn-vr');
  if (!btn || !('xr' in navigator) || !navigator.xr) return;
  let ok = false;
  try { ok = await navigator.xr.isSessionSupported('immersive-vr'); } catch (e) { ok = false; }
  if (!ok) return;
  btn.hidden = false;
  setupControllers();
  let session = null;
  btn.addEventListener('click', async () => {
    if (session) { session.end(); return; }
    try {
      renderer.xr.setFramebufferScaleFactor(0.9);
      session = await navigator.xr.requestSession('immersive-vr', { optionalFeatures: ['local-floor', 'bounded-floor'] });
      await renderer.xr.setSession(session);
      xr.calibrate = true;
      btn.textContent = 'Exit VR';
      session.addEventListener('end', () => {
        session = null;
        btn.textContent = 'Enter VR';
        rig.position.set(0, 0, 0); rig.rotation.set(0, 0, 0); rig.updateMatrixWorld();
        applySize();
        resetView();
      });
    } catch (e) {
      session = null;
      console.warn('VR session not started:', e && e.message);
    }
  });
}

// ── loop ────────────────────────────────────────────────────────────────────
function renderOnce() {
  if (!scene || !camera) return;
  U.animTime.value = motionOn ? simTime : 0;
  if (state.movers.length) poseMovers(U.animTime.value);
  renderer.render(scene, camera);
}

renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (!scene || !camera) return;
  if (playing && motionOn) simTime += dt;
  walk(dt);
  if (renderer.xr.isPresenting) xrUpdate(dt);
  renderOnce();
});

let released = false;
function release() {
  if (!SHOT || released) return;
  released = true;
  fetch('/__release').catch(() => {});
}

if (!location.hash) {
  // First visit: show the help once.
  try { if (!localStorage.getItem('intoframe.viewer.help')) { toggleHelp(); localStorage.setItem('intoframe.viewer.help', '1'); } } catch (e) { /* storage blocked */ }
}
applySize();
setupVR();
loadScene((location.hash || '#paris').slice(1));
