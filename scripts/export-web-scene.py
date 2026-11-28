#!/usr/bin/env python
"""
Pack a run's final scene into the website viewer's bundle (docs/viewer/FORMAT.md).

    python scripts/export-web-scene.py output/<run-id> --out docs/viewer/scenes/<slug>
           [--scene <scene.json>] [--name "Paris"] [--round r3] [--max-texture 2048] [--budget-mb 40]

The scene is built by render-scene.py's own SceneBuilder at the capture camera -- the LOD
each instance shows there, the terrain's baked texture and its overlay passes, the
photographed water and its flow, the unlit backdrops, the exposure fit's gains, the cards'
bent normals -- and what it built is written out as glTF, so the viewer shows what
render-scene shows. Everything is in the GL frame render-scene draws in (X negated).
"""
from __future__ import annotations

import argparse
import os
import datetime
import importlib.util
import io
import json
import math
import re
import struct
import sys
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))


def _renderer():
    """render-scene.py as a module (RENDER_SCENE names another copy, for a staged one)."""
    path = Path(os.environ.get("RENDER_SCENE") or ROOT / "scripts" / "render-scene.py")
    spec = importlib.util.spec_from_file_location("render_scene", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["render_scene"] = mod
    spec.loader.exec_module(mod)
    return mod


def _flow_field_class():
    try:
        from pipeline.scene_motion.flow_field import FlowField
    except ImportError:   # staged, not installed yet
        import pipeline
        pkg = types.ModuleType("pipeline.scene_motion")
        pkg.__path__ = [str(ROOT / "scratch" / "r3" / "next" / "scene_motion")]
        sys.modules["pipeline.scene_motion"] = pkg
        from pipeline.scene_motion.flow_field import FlowField
    return FlowField


# render-scene's shader discards texels under this alpha on every opaque material (cards,
# billboards, the backdrop shells): what lies between it and glTF's usual 0.5 -- a
# backdrop's soft lower edge, a card's fringe -- is drawn, so the bundle cuts there too.
CUTOFF = 0.02


# ── glTF writing ───────────────────────────────────────────────────────────────

class GLB:
    """A minimal glTF 2.0 binary writer: meshes of triangle primitives, materials with a
    base-colour image (JPEG, or PNG where alpha is needed), extras, custom attributes."""

    def __init__(self, max_texture: int):
        self.g = {"asset": {"version": "2.0", "generator": "Into Frame export-web-scene"},
                  "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "materials": [],
                  "textures": [], "images": [], "samplers": [], "accessors": [], "bufferViews": [], "buffers": []}
        self.bin = bytearray()
        self.max_texture = int(max_texture)
        self._images = {}

    def _view(self, data: bytes, target: int | None = None) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        view = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            view["target"] = target
        self.bin += data
        self.g["bufferViews"].append(view)
        return len(self.g["bufferViews"]) - 1

    def accessor(self, arr: np.ndarray, kind: str, minmax: bool = False, indices: bool = False) -> int:
        arr = np.ascontiguousarray(arr)
        ctype = {np.dtype(np.float32): 5126, np.dtype(np.uint32): 5125, np.dtype(np.uint16): 5123,
                 np.dtype(np.uint8): 5121}[arr.dtype]
        acc = {"bufferView": self._view(arr.tobytes(), 34963 if indices else 34962), "componentType": ctype,
               "count": int(arr.shape[0]), "type": kind}
        if arr.dtype == np.uint8:
            acc["normalized"] = True
        if minmax:
            acc["min"] = [float(v) for v in arr.min(axis=0)]
            acc["max"] = [float(v) for v in arr.max(axis=0)]
        self.g["accessors"].append(acc)
        return len(self.g["accessors"]) - 1

    def image(self, src: np.ndarray, max_size: int | None = None, bleed: bool = True) -> tuple[int, bool]:
        """(image index, has alpha) for an RGB(A) uint8 array, deduplicated."""
        key = (id(src), max_size, bleed)
        if key in self._images:
            return self._images[key]
        a = np.asarray(src)
        if a.dtype != np.uint8:
            a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        alpha = a.ndim == 3 and a.shape[2] == 4 and bool((a[..., 3] < 250).any())
        im = PILImage.fromarray(a if alpha else (a[..., :3] if a.ndim == 3 else a))
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA" if alpha else "RGB")
        m = int(max_size or self.max_texture)
        if max(im.size) > m:
            s = m / max(im.size)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), PILImage.LANCZOS)
        buf = io.BytesIO()
        if alpha:
            # Colour carried out under the transparent texels: a filtered or mipmapped
            # cutout otherwise draws its edge with the black the matte left there.
            from scipy.ndimage import distance_transform_edt
            arr = np.asarray(im.convert("RGBA")).copy()
            solid = arr[..., 3] >= CUTOFF * 255.0
            if bleed and solid.any() and not solid.all():
                _, (ri, ci) = distance_transform_edt(~solid, return_indices=True)
                arr[..., :3] = np.where(solid[..., None], arr[..., :3], arr[ri, ci, :3])
                im = PILImage.fromarray(arr, "RGBA")
            im.save(buf, "PNG", optimize=True)
            mime = "image/png"
        else:
            im.convert("RGB").save(buf, "JPEG", quality=85, optimize=True)
            mime = "image/jpeg"
        self.g["images"].append({"bufferView": self._view(buf.getvalue()), "mimeType": mime})
        out = (len(self.g["images"]) - 1, alpha)
        self._images[key] = out
        return out

    def material(self, m: dict) -> int:
        mat = {"pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": float(m.get("roughness", 0.9)),
                                        "baseColorFactor": [float(v) for v in m.get("factor", (1, 1, 1, 1))]},
               "doubleSided": bool(m.get("doubleSided", True))}
        if m.get("image") is not None:
            img, alpha = self.image(m["image"], m.get("maxSize"), bleed=m.get("bleed", True))
            self.g["samplers"].append({"magFilter": 9729, "minFilter": 9987,
                                       "wrapS": 33071 if m.get("clamp") else 10497,
                                       "wrapT": 33071 if m.get("clamp") else 10497})
            self.g["textures"].append({"source": img, "sampler": len(self.g["samplers"]) - 1})
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": len(self.g["textures"]) - 1}
            if m.get("alphaMode") == "BLEND":
                mat["alphaMode"] = "BLEND"
            elif alpha:
                mat["alphaMode"] = "MASK"
                mat["alphaCutoff"] = float(m.get("alphaCutoff", CUTOFF))
        elif m.get("alphaMode") == "BLEND":
            mat["alphaMode"] = "BLEND"
        if m.get("unlit"):
            mat["extensions"] = {"KHR_materials_unlit": {}}
            self.g.setdefault("extensionsUsed", [])
            if "KHR_materials_unlit" not in self.g["extensionsUsed"]:
                self.g["extensionsUsed"].append("KHR_materials_unlit")
        if m.get("extras"):
            mat["extras"] = m["extras"]
        self.g["materials"].append(mat)
        return len(self.g["materials"]) - 1

    def mesh(self, attributes: dict, indices: np.ndarray, material: int, name: str) -> int:
        prim = {"attributes": attributes, "indices": self.accessor(np.asarray(indices).astype(np.uint32).ravel(), "SCALAR", indices=True),
                "material": material, "mode": 4}
        self.g["meshes"].append({"name": name, "primitives": [prim]})
        return len(self.g["meshes"]) - 1

    def node(self, mesh: int, name: str, matrix: np.ndarray | None = None, extras: dict | None = None) -> int:
        n = {"mesh": mesh, "name": name}
        if matrix is not None and not np.allclose(matrix, np.eye(4)):
            n["matrix"] = [float(v) for v in np.asarray(matrix, dtype=np.float64).T.ravel()]
        if extras:
            n["extras"] = extras
        self.g["nodes"].append(n)
        self.g["scenes"][0]["nodes"].append(len(self.g["nodes"]) - 1)
        return len(self.g["nodes"]) - 1

    def save(self, path: Path) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        self.g["buffers"] = [{"byteLength": len(self.bin)}]
        for k in ("textures", "images", "samplers", "materials", "accessors", "bufferViews"):
            if not self.g[k]:
                del self.g[k]
        js = json.dumps(self.g, separators=(",", ":")).encode()
        js += b" " * ((4 - len(js) % 4) % 4)
        out = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(self.bin))
        out += struct.pack("<I4s", len(js), b"JSON") + js
        out += struct.pack("<I4s", len(self.bin), b"BIN\x00") + bytes(self.bin)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(out)
        return len(out)


def _src(tex) -> np.ndarray | None:
    if tex is None:
        return None
    s = tex.source
    if isinstance(s, PILImage.Image):
        s = np.asarray(s)
    return np.asarray(s)


def _mat_desc(mat, max_size=None, **extra) -> dict:
    """A pyrender material as GLB.material's description. Unlit content (the photograph's
    own light) is drawn by render-scene through the emissive slot with a black base."""
    unlit = mat.emissiveTexture is not None and np.allclose(np.asarray(mat.baseColorFactor)[:3], 0.0)
    tex = mat.emissiveTexture if unlit else mat.baseColorTexture
    clamp = bool(tex is not None and tex.sampler is not None and getattr(tex.sampler, "wrapS", None) == 33071)
    d = {"image": _src(tex), "factor": (1, 1, 1, 1) if unlit else tuple(np.asarray(mat.baseColorFactor, dtype=float)),
         "unlit": unlit, "alphaMode": mat.alphaMode, "doubleSided": bool(mat.doubleSided),
         "roughness": float(mat.roughnessFactor if mat.roughnessFactor is not None else 0.9), "clamp": clamp,
         "maxSize": max_size}
    ef = np.asarray(mat.emissiveFactor if mat.emissiveFactor is not None else (0, 0, 0), dtype=float)
    if not unlit and 0.0005 < ef[2] < 0.002:
        d["extras"] = {"foliage": True}    # render-scene's two-sided wrapped diffuse
    d.update(extra)
    return d


def _attrs(glb: GLB, prim, flip_v=True, extra: dict | None = None, normals: bool = True) -> dict:
    a = {"POSITION": glb.accessor(np.asarray(prim.positions, dtype=np.float32), "VEC3", minmax=True)}
    if normals and prim.normals is not None:
        a["NORMAL"] = glb.accessor(np.asarray(prim.normals, dtype=np.float32), "VEC3")
    if prim.texcoord_0 is not None:
        uv = np.asarray(prim.texcoord_0, dtype=np.float32).copy()
        if flip_v:
            uv[:, 1] = 1.0 - uv[:, 1]     # pyrender's v runs up from the image's bottom row; glTF's down from its top
        a["TEXCOORD_0"] = glb.accessor(uv, "VEC2")
    if extra:
        a.update(extra)
    return a


class _Prim:
    """A primitive's arrays, indexed."""
    def __init__(self, positions, normals, texcoord_0, indices, texcoord_1=None, color_0=None, material=None):
        self.positions, self.normals, self.texcoord_0, self.indices = positions, normals, texcoord_0, indices
        self.texcoord_1, self.color_0, self.material = texcoord_1, color_0, material


def _smooth_normals(pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    fn = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    vn = np.zeros_like(pos)
    for k in range(3):
        np.add.at(vn, tri[:, k], fn)
    return (vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _indexed(prim, keep_normals: bool) -> _Prim:
    """pyrender expands a flat-shaded mesh to three vertices a face; welded again here on
    position and the other attributes (and the normal when `keep_normals`: a model keeps its
    facets; the terrain and the backdrops take smooth normals, a third of the size)."""
    if prim.indices is not None:
        return _Prim(prim.positions, prim.normals, prim.texcoord_0, np.asarray(prim.indices).reshape(-1, 3),
                     prim.texcoord_1, prim.color_0, prim.material)
    cols = [np.asarray(prim.positions, dtype=np.float32)]
    for a in (prim.texcoord_0, prim.texcoord_1, prim.color_0):
        if a is not None:
            cols.append(np.asarray(a, dtype=np.float32))
    if keep_normals and prim.normals is not None:
        cols.append(np.asarray(prim.normals, dtype=np.float32))
    allv = np.concatenate(cols, axis=1)
    uniq, first, inv = np.unique(np.round(allv, 6), axis=0, return_index=True, return_inverse=True)
    tri = inv.reshape(-1, 3).astype(np.uint32)
    pick = lambda a: None if a is None else np.asarray(a, dtype=np.float32)[first]  # noqa: E731
    pos = pick(prim.positions)
    normals = pick(prim.normals) if keep_normals else _smooth_normals(pos, tri)
    return _Prim(pos, normals, pick(prim.texcoord_0), tri, pick(prim.texcoord_1), pick(prim.color_0), prim.material)


def _safe(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scene", type=Path, default=None, help="scene.json to export (default: the run's newest)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--slug", default=None)
    ap.add_argument("--round", default="")
    ap.add_argument("--config", type=Path, default=ROOT / "server" / "config.yaml")
    ap.add_argument("--max-texture", type=int, default=2048, help="largest side of a model's texture")
    ap.add_argument("--env-texture", type=int, default=4096, help="largest side of the panorama overlay and the sky")
    ap.add_argument("--budget-mb", type=float, default=40.0)
    ap.add_argument("--walk-radius", type=float, default=3.0)
    args = ap.parse_args()

    rs = _renderer()
    log = lambda s: print(s, flush=True)  # noqa: E731
    t0 = time.time()
    run_dir = args.run_dir.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("*.glb")) + list((out / "models").glob("*.glb")):
        old.unlink()

    context = rs.load_context(run_dir, args.config)
    from pipeline.pipeline_context import ContextKey
    candidates = sorted(run_dir.glob("*/scene.json"), key=lambda p: p.stat().st_mtime)
    scene_path = args.scene or (candidates[-1] if candidates else run_dir / "Scene Generation" / "scene.json")
    scene = json.loads(scene_path.read_text())
    log(f"scene: {len(scene['objects'])} objects from {scene_path}")

    # The renderer's options as render-scene sets them by default, with its exposure-fit block.
    ns = argparse.Namespace(run_dir=run_dir, scene=scene_path, no_grass=True, only="", exclude="", env_exclude=None,
                            flat_terrain=False, material=None, terrain_texture=2048, pano_uv="baked", sun_scale=2.0,
                            ambient=0.45, exposure=1.0, animate=10.0, sun_az_el=None, no_fit=False,
                            ambient_tint=None, object_gain=1.0, object_gains={})
    fit = (scene.get("lighting") or {}).get("fit")
    fit_path = run_dir / "Exposure Fit" / "exposure_fit.json"
    src = fit if fit else (json.loads(fit_path.read_text()) if fit_path.exists() else {})
    tint = src.get("ambient_tint")
    if isinstance(tint, (list, tuple)) and len(tint) == 3:
        ns.ambient_tint = [float(v) for v in tint]
    ns.object_gain = float(src.get("object_gain") or 1.0)
    gains = src.get("object_gains")
    ns.object_gains = {str(k): [float(x) for x in v] for k, v in gains.items()} if isinstance(gains, dict) else {}
    if fit:
        ns.ambient *= float(fit.get("ambient_scale") or 1.0)
    elif fit_path.exists():
        f = json.loads(fit_path.read_text())
        ns.sun_scale *= float(f.get("sun_scale") or 1.0)
        ns.ambient *= float(f.get("ambient_scale") or 1.0)

    builder = rs.SceneBuilder(context, scene, ns, log)
    flow_path = next((p for p in (scene_path.parent / "water_flow.npz", run_dir / "Scene Motion" / "water_flow.npz") if p.is_file()), None)
    if flow_path is not None:
        builder._flow_field = _flow_field_class().load(flow_path)
        log(f"water flow from {flow_path}")

    intr = context.input_intrinsics(ContextKey.INTRINSICS)
    if intr is not None:
        aspect = intr.width / intr.height
        vfov = 2.0 * math.atan(intr.height / (2.0 * intr.fy))
    else:
        aspect, vfov = 1.5, 2.0 * math.atan(math.tan(math.radians(30.0)) / 1.5)
    R, t = rs.camera_basis(scene)
    Rv, pos = rs.view_pose(R, np.asarray(t, dtype=np.float64).ravel(), "photo")
    pos = np.asarray(pos, dtype=np.float64).ravel()
    ps, stats = builder.build(pos, vfov)
    log(f"built: {stats}")

    kinds = {v: k for k, v in rs.WIRE_COLOURS.items()}
    mover_nodes = {id(m[0]): m for m in builder.mover_nodes}
    env_objects = {str(o.get("name")): o for o in scene["objects"]
                   if str(o.get("name")) in ("terrain", "water", "water_apron", "treeline", "backdrop") or str(o.get("name")).startswith("formation")}
    env_counts = {}
    for name, o in env_objects.items():
        tm = builder.trimesh_for(o.get("mesh")) if o.get("mesh") else None
        if tm is not None:
            env_counts.setdefault(len(tm.vertices), []).append(name)

    env = GLB(args.max_texture)
    water = None
    models: dict[str, str] = {}
    static: dict[str, dict] = {}
    movers = []
    terrain = None
    overlays = []
    motion = scene.get("motion") or {}
    vegetation = motion.get("vegetation") or {}

    for node in ps.mesh_nodes:
        prim = node.mesh.primitives[0]
        pose = ps.get_pose(node)
        kind = kinds.get(builder.wire_nodes.get(node))
        keyed = builder.key_nodes.get(node)
        if id(node) in mover_nodes:
            _, mpos, mrot, mscl, m = mover_nodes[id(node)]
            key = keyed[1]
            if key not in models:
                g = GLB(args.max_texture)
                ip = _indexed(prim, keep_normals=True)
                g.node(g.mesh(_attrs(g, ip), ip.indices, g.material(_mat_desc(prim.material)), key), key)
                g.save(out / "models" / f"{_safe(key)}.glb")
                models[key] = f"models/{_safe(key)}.glb"
            mm = json.loads(json.dumps(m))
            if mm.get("path"):
                mm["path"] = [[-float(x), float(z)] for x, z in mm["path"]]
            if "yawOffsetDeg" in mm:
                mm["yawOffsetDeg"] = -float(mm["yawOffsetDeg"])
            if mm.get("bob"):
                mm["bob"]["rollDeg"] = -float(mm["bob"].get("rollDeg", 0.0))
            mm.pop("why", None)
            # The hull's size for its wake (water.glsl).
            mm["lengthM"] = round(float(mm.pop("_length", 10.0)), 2)
            mm["beamM"] = round(float(mm.pop("_beam", 3.0)), 2)
            movers.append({"key": key, "base": {"position": [-mpos[0], mpos[1], mpos[2]],
                                                "rotationDeg": [mrot[0], -mrot[1], -mrot[2]], "scale": list(mscl)},
                           "motion": mm})
            continue
        if keyed is not None:
            ktype, key = keyed
            if key not in models:
                g = GLB(args.max_texture)
                ip = _indexed(prim, keep_normals=True)
                g.node(g.mesh(_attrs(g, ip), ip.indices, g.material(_mat_desc(prim.material)), key), key)
                g.save(out / "models" / f"{_safe(key)}.glb")
                models[key] = f"models/{_safe(key)}.glb"
            poses = prim.poses if prim.poses is not None else np.eye(4)[None]
            mats = [pose @ p for p in poses]
            sw = rs.kind_key(key) if getattr(prim, "_sway", None) is not None and rs.kind_key(key) in vegetation else None
            entry = static.setdefault(key, {"matrices": [], "sway": sw, "castShadow": not bool(getattr(node.mesh, "no_shadow", False)),
                                            "billboard": ktype == "billboard"})
            entry["matrices"] += [[round(float(v), 5) for v in m.T.ravel()] for m in mats]
            continue
        # Environment.
        if kind == "terrain":
            terrain = (node, prim, pose)
            continue
        if kind is None and prim.color_0 is not None:
            overlays.append((node, prim, pose))      # plate / panorama overlays, written after the terrain
            continue
        if kind == "water" and (prim.texcoord_1 is not None or "water" in env_counts.get(len(prim.positions), [])) and water is None:
            water = GLB(args.max_texture)
            extra = {}
            wk = (motion.get("water") or {}).get("kind") or "river"
            ip = _indexed(prim, keep_normals=False)
            if ip.color_0 is not None and ip.texcoord_1 is not None:
                # The welded vertices' distance to the shore (m) and current (m/s, the
                # bundle's x and z), as render-scene packs them in COLOR_0 for water.glsl.
                c0 = np.asarray(ip.color_0, dtype=np.float32)
                extra = {"_SHORE": water.accessor(c0[:, 0] * 400.0, "SCALAR"),
                         "_CURRENT": water.accessor(np.ascontiguousarray((c0[:, 1:3] - 0.5) * 4.0), "VEC2")}
            water.node(water.mesh(_attrs(water, ip, extra=extra, normals=False), ip.indices,
                                  water.material(_mat_desc(prim.material, max_size=args.env_texture)), "water"),
                       "water", pose, {"role": "water", "kind": wk})
            continue
        ip = _indexed(prim, keep_normals=True)
        names = env_counts.get(len(_indexed(prim, keep_normals=False).positions), [])
        name = names[0] if names else (kind or "env")
        role = ("wall" if kind == "wall" else "waterApron" if name == "water_apron" or (kind == "water") else "treeline" if name == "treeline"
                else "backdrop" if kind == "backdrop" else "formation" if kind == "formation" else name)
        desc = _mat_desc(prim.material, max_size=args.env_texture)
        env.node(env.mesh(_attrs(env, ip, normals=not desc["unlit"]), ip.indices, env.material(desc), name), name, pose,
                 {"role": role, "unlit": bool(desc["unlit"])})

    # The terrain, and its overlays sharing its vertices: render-scene draws each overlay on
    # the terrain's own vertices pulled a hair toward the camera (x0.998, x0.999), indexed
    # as the terrain mesh is; the base it bakes is expanded a face at a time. The base is
    # written on the overlays' indexing (the vertices put back), so all of them share one
    # POSITION / NORMAL pair; the viewer offsets the overlays' depth instead.
    params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}
    half = float(params.get("grid_size_meters", 200.0)) / 2.0
    shared = None
    if terrain is not None:
        tnode, tprim, tpose = terrain
        full = [o for o in overlays if o[1].indices is not None and o[1].color_0 is not None
                and len(o[1].positions) == len(builder.trimesh_for(env_objects["terrain"]["mesh"]).vertices)]
        desc = _mat_desc(tprim.material, bleed=False)
        if full:
            # The overlays index the terrain's own vertices (pulled toward the camera by
            # their factor); the base is drawn a face at a time and, where render-scene
            # draws steep faces as a wall mesh of their own, without them. Its corners are
            # matched to the shared vertices, which keeps its faces and their winding.
            # The steep faces (quay walls, cliffs) are render-scene's side-projected wall
            # mesh, exported as an environment mesh of its own (role "wall"); the base
            # primitive matched here is already without them.
            steep = getattr(builder, "terrain_steep", None)
            from scipy.spatial import cKDTree
            ref = max(full, key=lambda o: len(np.asarray(o[1].indices)))[1]
            tv = np.asarray(builder.trimesh_for(env_objects["terrain"]["mesh"]).vertices)
            k = float(np.median(np.linalg.norm(np.asarray(ref.positions), axis=1) / np.maximum(np.linalg.norm(tv, axis=1), 1e-9)))
            tpos = (np.asarray(ref.positions, dtype=np.float64) / k).astype(np.float32)
            corners = np.asarray(tprim.positions, dtype=np.float32)
            if tprim.indices is not None:
                corners = corners[np.asarray(tprim.indices).ravel()]
            dist, idx = cKDTree(tpos).query(corners)
            if float(dist.max()) > 1e-3:
                raise RuntimeError(f"terrain base does not lie on the overlays' vertices (off by {dist.max():.4f} m)")
            tri = idx.reshape(-1, 3).astype(np.uint32)
            if steep is not None and len(tri) != int((~np.asarray(steep)).sum()):
                raise RuntimeError(f"terrain base has {len(tri)} faces, {int((~np.asarray(steep)).sum())} expected without the steep ones")
            nrm = _smooth_normals(tpos, np.asarray(ref.indices).reshape(-1, 3).astype(np.int64))
            uv = np.stack([(-tpos[:, 0] + half) / (2.0 * half), (tpos[:, 2] + half) / (2.0 * half)], axis=-1).astype(np.float32)
            a = {"POSITION": env.accessor(tpos, "VEC3", minmax=True), "NORMAL": env.accessor(nrm, "VEC3"),
                 "TEXCOORD_0": env.accessor(uv, "VEC2")}
            shared = (a["POSITION"], a["NORMAL"], len(tpos))
            env.node(env.mesh(a, tri, env.material(desc), "terrain"), "terrain", tpose, {"role": "terrain", "unlit": False})
        else:
            ip = _indexed(tprim, keep_normals=False)
            env.node(env.mesh(_attrs(env, ip), ip.indices, env.material(desc), "terrain"), "terrain", tpose,
                     {"role": "terrain", "unlit": False})

    # render-scene draws the plate layers first and the panorama layer over them; the
    # panorama's texture is the 2:1 equirect, the plates' square tiles.
    def _is_pano(o):
        src_ = _src(o[1].material.baseColorTexture)
        return src_ is not None and src_.shape[1] >= 1.8 * src_.shape[0]
    overlays.sort(key=_is_pano)
    for order, (node, prim, pose) in enumerate(overlays, start=1):
        if shared is not None and len(prim.positions) == shared[2]:
            a = {"POSITION": shared[0], "NORMAL": shared[1]}
        else:
            ip0 = _indexed(prim, keep_normals=False)
            a = {"POSITION": env.accessor(np.asarray(ip0.positions, dtype=np.float32), "VEC3", minmax=True),
                 "NORMAL": env.accessor(np.asarray(ip0.normals, dtype=np.float32), "VEC3")}
        ip = _indexed(prim, keep_normals=False) if prim.indices is None else prim
        uv = np.asarray(ip.texcoord_0, dtype=np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        a["TEXCOORD_0"] = env.accessor(uv, "VEC2")
        a["COLOR_0"] = env.accessor(np.clip(np.rint(np.asarray(ip.color_0, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8), "VEC4")
        desc = _mat_desc(prim.material, max_size=args.env_texture)
        desc["alphaMode"] = "BLEND"
        env.node(env.mesh(a, ip.indices, env.material(desc), f"terrain_overlay_{order}"), f"terrain_overlay_{order}", pose,
                 {"role": "terrainOverlay", "unlit": bool(desc["unlit"]), "overlayOrder": order})

    total = env.save(out / "env.glb")
    if water is not None:
        total += water.save(out / "water.glb")
    total += sum((out / p).stat().st_size for p in models.values())

    # Sky.
    sky = context.input_panorama(scene.get("skybox") or "panorama_sky")
    if sky is not None:
        im = sky.image.convert("RGB")
        if im.width > args.env_texture:
            im = im.resize((args.env_texture, args.env_texture // 2), PILImage.LANCZOS)
        im.save(out / "sky.jpg", "JPEG", quality=88, optimize=True)
        total += (out / "sky.jpg").stat().st_size
    sky_m = motion.get("sky") or {}
    right = R[:, 0][[0, 2]]
    sd = math.radians(float(sky_m.get("directionDeg", 0.0)))
    drift = float(sky_m.get("angularSpeedDegS", 0.0)) * float(np.sign(np.dot([math.sin(sd), math.cos(sd)], right / max(np.linalg.norm(right), 1e-9))))

    inst = {"static": static, "movers": movers}
    (out / "instances.json").write_text(json.dumps(inst, separators=(",", ":")))
    total += (out / "instances.json").stat().st_size

    # Light, as render-scene sets it.
    sun = (scene.get("lighting") or {}).get("sun") or {}
    d = np.asarray(sun.get("direction") or [0.3, 0.8, -0.5], dtype=np.float64)
    yaw = math.radians(float(scene.get("skyboxRotation") or 0.0))
    d = np.array([math.cos(yaw) * d[0] + math.sin(yaw) * d[2], d[1], -math.sin(yaw) * d[0] + math.cos(yaw) * d[2]])
    d /= max(np.linalg.norm(d), 1e-6)
    colour = sun.get("color") or "#ffffff"
    rgb = [int(colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5)] if str(colour).startswith("#") else [1.0, 1.0, 1.0]

    wind = motion.get("wind") or {}
    wd = math.radians(float(wind.get("directionDeg", 0.0)))
    wm = motion.get("water")
    mirror = np.array([-1.0, 1.0, 1.0])
    manifest = {
        "format": 1,
        "name": args.name or scene_path.parent.parent.name,
        "slug": args.slug or out.name,
        "run": run_dir.name,
        "round": args.round,
        "built": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "camera": {"position": [round(float(v), 4) for v in pos * mirror],
                   "forward": [round(float(v), 6) for v in Rv[:, 2] * mirror],
                   "up": [round(float(v), 6) for v in Rv[:, 1] * mirror],
                   "fovYDeg": round(math.degrees(vfov), 3), "aspect": round(aspect, 4),
                   # Above the ground under the capture point: the viewer keeps the eye there as it walks.
                   "eyeHeightM": round(float(pos[1]) - float(scene.get("terrainCenterY") or 0.0), 3),
                   "walkRadiusM": float(args.walk_radius)},
        "sky": {"image": "sky.jpg" if sky is not None else None, "yawDeg": float(scene.get("skyboxRotation") or 0.0),
                "driftDegS": round(drift, 5)},
        "env": "env.glb",
        "water": "water.glb" if water is not None else None,
        "models": models,
        "instances": "instances.json",
        "motion": {
            "wind": {"dir": [round(-math.sin(wd), 6), round(math.cos(wd), 6)], "speedMs": max(float(wind.get("speedMs", 2.0)), 0.5),
                     "gustiness": float(wind.get("gustiness", 0.4))},
            "water": ({"kind": wm.get("kind"), "flowPeriodS": 2.0, "waveSpeedMs": float(wm.get("waveSpeedMs") or 3.0),
                       "swellPeriodS": float(wm.get("swellPeriodS") or 10.0)} if wm else None),
            "vegetation": {k: {"amplitudeRad": round(math.radians(float(v.get("amplitudeDeg", 3.0))), 5),
                               "frequencyHz": float(v.get("frequencyHz", 0.5)), "heightM": float(v.get("heightM", 0.0))}
                           for k, v in vegetation.items()},
        },
        "light": {"sunDir": [round(float(v), 6) for v in d * mirror], "sunColor": [round(v, 4) for v in rgb],
                  "sunIntensity": round(float(sun.get("intensity") or 1.0) * float(ns.sun_scale), 4),
                  "ambient": [round(float(v), 5) for v in builder._ambient()], "exposure": 1.0},
    }
    manifest["bytes"] = int(total)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    mb = (total + (out / "manifest.json").stat().st_size) / 2 ** 20
    log(f"bundle: {mb:.1f} MB ({len(models)} models, {sum(len(v['matrices']) for v in static.values())} instances, "
        f"{len(movers)} movers, {len(overlays)} terrain overlays) -> {out} in {time.time() - t0:.0f} s")
    if mb > args.budget_mb:
        log(f"warning: over the {args.budget_mb:.0f} MB budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
