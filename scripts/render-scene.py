#!/usr/bin/env python
"""
Render a run's final scene to images without Unity.

    python scripts/render-scene.py output/<run-id> [--out DIR] [--width 1200]
                                   [--views photo,left,right,back,step,high]
                                   [--terrain-texture 2048] [--no-grass] [--exposure 1.0]

Builds the scene the way the client does -- scene.json's objects with their LOD
level picked for the camera, cards and billboards cut out by alpha, the terrain
painted with the splat material baked to one texture, the water sheet, the
treeline, the panorama skybox behind everything and the scene's sun -- and renders
it with pyrender over EGL (no display needed). The "photo" view is the capture
camera with the photograph's own intrinsics, so `compare.png` puts the input
photo next to the render for a direct check of placement, scale and texture.

Everything is read through the PipelineContext from the cached run directory, in
the server's own left-handed frame (+X right, +Y up, +Z forward); the GL mirror
happens once at the end, so what this shows is what the pipeline placed, and (after
the glTF handedness fix in scene/mesh.py) what Unity shows.

Fidelity caveats: no stochastic tiling or height-biased sharpness beyond the
shader's default, no URP tone mapping, no fog, terrain texels are the baked
texture's (10 cm at 2048), sway/animation is ignored. Lighting is a directional
light plus flat ambient; use --exposure to taste.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

# pyrender 0.1.45 still spells np.infty, which NumPy 2 removed.
if not hasattr(np, "infty"):
    np.infty = np.inf

import trimesh
from PIL import Image as PILImage, ImageDraw
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

import pyrender  # noqa: E402
from pyrender.shader_program import ShaderProgramCache  # noqa: E402

# Mirror between the server frame (left-handed) and GL (right-handed): negate X.
_MIRROR = np.diag([-1.0, 1.0, 1.0, 1.0])


# ── context ─────────────────────────────────────────────────────────────────────

def load_context(run_dir: Path, config_path: Path):
    """The cached run as a PipelineContext positioned after its last stage."""
    from pipeline.pipeline import PipelineConfiguration
    from pipeline.pipeline_context import PipelineContext

    scratch = Path(tempfile.mkdtemp(prefix="render-scene-"))
    config = PipelineConfiguration(output=str(scratch), config_path=config_path, log_mode="plain")
    entries = config.stages_yaml
    names = [e["name"] for e in entries]
    enabled = [e["name"] for e in entries if e.get("enabled", True)]
    context = PipelineContext()
    context.load(run_dir, enabled, names)
    for name in enabled:
        context.push_stage(name)
        context.pop_stage()
    # Resolve every key from the last stage.
    context.push_stage(enabled[-1])
    return context


# ── shaders: pyrender with an alpha cutout ───────────────────────────────────────

def cutout_shader_dir() -> str:
    """A copy of pyrender's shaders whose mesh fragment shader discards texels
    under alpha 0.5, so cards, billboards and grass are cut out in the opaque pass
    with correct depth (pyrender itself only ever blends)."""
    src = Path(pyrender.__file__).parent / "shaders"
    dst = Path(tempfile.mkdtemp(prefix="render-scene-shaders-"))
    shutil.copytree(src, dst, dirs_exist_ok=True)
    frag = dst / "mesh.frag"
    text = frag.read_text()
    marker = "frag_color = clamp(vec4(pow(color.xyz, vec3(1.0/2.2)), color.a * base_color.a), 0.0, 1.0);"
    if marker not in text:
        raise RuntimeError("pyrender's mesh.frag changed; the alpha cutout patch no longer applies")
    # 0.02, not the glTF MASK 0.5: the terrain's panorama overlay carries its blend
    # weight in the vertex alpha and must survive at any weight above nothing.
    frag.write_text(text.replace(marker, marker + "\n    if (frag_color.a < 0.02) discard;"))
    return str(dst)


# ── geometry helpers ─────────────────────────────────────────────────────────────

def to_gl(tm: trimesh.Trimesh) -> trimesh.Trimesh:
    out = tm.copy()
    v = np.array(out.vertices, dtype=np.float64)
    v[:, 0] *= -1.0
    out.vertices = v
    out.faces = np.array(out.faces, dtype=np.int64)[:, ::-1]
    return out


def transform(position, rotation_deg, scale) -> np.ndarray:
    """Unity-style TRS in the server frame: Euler (x, y, z) degrees, applied z, x, y."""
    px, py, pz = position
    rx, ry, rz = (math.radians(a) for a in rotation_deg)
    sx, sy, sz = scale
    cx, sx_ = math.cos(rx), math.sin(rx)
    cy, sy_ = math.cos(ry), math.sin(ry)
    cz, sz_ = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx_], [0, sx_, cx]])
    Ry = np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]])
    Rz = np.array([[cz, -sz_, 0], [sz_, cz, 0], [0, 0, 1]])
    R = Ry @ Rx @ Rz
    M = np.eye(4)
    M[:3, :3] = R @ np.diag([sx, sy, sz])
    M[:3, 3] = [px, py, pz]
    return M


def yaw_to_face(from_pos, to_pos) -> float:
    """Degrees about Y so that an object's -Z face looks from `from_pos` toward
    `to_pos` (Unity's Billboard: LookRotation(-dir), quad front is -Z)."""
    d = np.asarray(to_pos, dtype=np.float64) - np.asarray(from_pos, dtype=np.float64)
    return math.degrees(math.atan2(-d[0], -d[2]))


def vec(v, default=(0.0, 0.0, 0.0)):
    if v is None:
        return default
    return (float(v.get("x", 0.0)), float(v.get("y", 0.0)), float(v.get("z", 0.0)))


def quad() -> trimesh.Trimesh:
    """Unity's Quad: 1x1 in XY, centred, front face -Z."""
    v = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], dtype=np.float64)
    f = np.array([[0, 2, 1], [0, 3, 2]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    return trimesh.Trimesh(vertices=v, faces=f, visual=trimesh.visual.TextureVisuals(uv=uv), process=False)


# ── materials ────────────────────────────────────────────────────────────────────

def material_for(tm: trimesh.Trimesh, image: PILImage.Image | None = None, unlit: bool = False,
                 blend: bool = False, color=None, roughness: float = 0.9, double_sided: bool = True):
    kwargs = dict(metallicFactor=0.0, roughnessFactor=roughness, doubleSided=double_sided,
                  alphaMode="BLEND" if blend else "OPAQUE")
    img = image
    if img is None and hasattr(tm.visual, "material"):
        mat = tm.visual.material
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is not None:
        img = img.convert("RGBA")
        arr = np.asarray(img)
        kwargs["baseColorTexture"] = pyrender.Texture(source=arr, source_channels="RGBA")
        if unlit:
            # The photograph's own light is in the pixels: show them as emission
            # (a second Texture object; one shared between slots loses the cutout).
            kwargs["emissiveTexture"] = pyrender.Texture(source=arr.copy(), source_channels="RGBA")
            kwargs["emissiveFactor"] = [1.0, 1.0, 1.0]
            kwargs["baseColorFactor"] = [0.0, 0.0, 0.0, 1.0]
    elif color is not None:
        kwargs["baseColorFactor"] = list(color)
    else:
        kwargs["baseColorFactor"] = [0.6, 0.6, 0.6, 1.0]
    return pyrender.MetallicRoughnessMaterial(**kwargs)


# ── terrain texture bake ─────────────────────────────────────────────────────────

def bake_terrain_texture(splat, height_map: np.ndarray, half: float, size: int,
                         height_blend_sharpness: float = 4.0,
                         pano_uv_at=None, skip_equirect: bool = False) -> PILImage.Image:
    """TerrainSplat.shader evaluated on a top-down grid: blend maps at the grid UV,
    tiled layers at uv * tile_factor, the equirect layer at the panorama UV the
    mesh carries (TEXCOORD_1, what Unity samples; `pano_uv_at(X, Z) -> (u, v)`
    interpolated from the mesh) or, without it, at the texel's own analytic
    direction from the camera; height-biased weights, macro colour on the tiled
    layers. Row 0 is Z = -half (the server's canvas convention)."""
    us = (np.arange(size, dtype=np.float64) + 0.5) / size
    U, V = np.meshgrid(us, us)  # V: row 0 at z = -half
    X = (U - 0.5) * 2.0 * half
    Z = (V - 0.5) * 2.0 * half
    hm_h, hm_w = height_map.shape
    Y = map_coordinates(height_map, [(V * (hm_h - 1)).ravel(), (U * (hm_w - 1)).ravel()], order=1, mode="nearest").reshape(size, size)

    def sample(img: PILImage.Image, u, v, wrap: bool):
        arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
        h, w = arr.shape[:2]
        if wrap:
            cols = np.mod(u, 1.0) * w - 0.5
            rows = np.mod(v, 1.0) * h - 0.5
            mode = "wrap"
        else:
            cols = u * (w - 1)
            rows = v * (h - 1)
            mode = "nearest"
        out = np.stack([
            map_coordinates(arr[..., c], [rows.ravel(), cols.ravel()], order=1, mode=mode).reshape(u.shape)
            for c in range(4)
        ], axis=-1)
        return out

    layers = list(splat.layers)
    weights = []
    for i in range(len(layers)):
        bm = splat.blend_maps[i // 4] if i // 4 < len(splat.blend_maps) else None
        if bm is None:
            weights.append(np.zeros((size, size), dtype=np.float32))
            continue
        weights.append(sample(bm, U, V, wrap=False)[..., i % 4])

    albedo_eq = np.zeros((size, size, 3), dtype=np.float32)
    albedo_tiled = np.zeros((size, size, 3), dtype=np.float32)
    biased = []
    texs = []
    for i, layer in enumerate(layers):
        if layer.equirect:
            if pano_uv_at is not None:
                u, v = pano_uv_at(X, Z)
            else:
                lon = np.arctan2(X, Z)
                lat = np.arctan2(Y, np.sqrt(X ** 2 + Z ** 2).clip(1e-6))
                u = (lon + np.pi) / (2.0 * np.pi)
                v = 0.5 - lat / np.pi
            tex = sample(layer.tile, u, v, wrap=True)
            tex[..., 3] = 1.0
        else:
            tf = float(layer.tile_factor)
            tex = sample(layer.tile, U * tf, V * tf, wrap=True)
        texs.append(tex)
        biased.append(weights[i] * np.power(tex[..., 3] + 1e-4, height_blend_sharpness))
    total = sum(biased) + 1e-4
    albedo_plain = np.zeros((size, size, 3), dtype=np.float32)  # tiled, no macro (near plate)
    for i, layer in enumerate(layers):
        w = (biased[i] / total)[..., None]
        if layer.equirect:
            albedo_eq += texs[i][..., :3] * w
        elif not getattr(layer, "macro", True):
            albedo_plain += texs[i][..., :3] * w
        else:
            albedo_tiled += texs[i][..., :3] * w

    macro_mod = np.ones((size, size, 3), dtype=np.float32)
    macro = getattr(splat, "macro", None)
    strength = float(getattr(splat, "macro_strength", 0.0) or 0.0)
    if macro is not None and strength > 0.0:
        mean = np.asarray(getattr(splat, "macro_mean", None) or (0.5, 0.5, 0.5), dtype=np.float32)
        m = sample(macro, U, V, wrap=False)[..., :3]
        macro_mod = 1.0 + strength * (m / np.maximum(mean, 0.02) - 1.0)
    if skip_equirect:
        # The tiled layers only, renormalised among themselves: the panorama is
        # alpha-blended over this in a second pass with its own weight, which
        # performs the lerp the shader does.
        # No-macro layers (the near-ground plate) are drawn in their own overlay
        # pass at their own texel size, like the panorama; leave them out here.
        base = [i for i, l in enumerate(layers) if not l.equirect and getattr(l, "macro", True)]
        tiled_total = sum(biased[i] for i in base) + 1e-4
        albedo = np.zeros((size, size, 3), dtype=np.float32)
        for i in base:
            albedo += texs[i][..., :3] * macro_mod * (biased[i] / tiled_total)[..., None]
        return PILImage.fromarray((albedo.clip(0, 1) * 255).astype(np.uint8), "RGB")
    albedo = albedo_eq + albedo_plain + albedo_tiled * macro_mod
    return PILImage.fromarray((albedo.clip(0, 1) * 255).astype(np.uint8), "RGB")


def baked_pano_uv_sampler(vertices: np.ndarray, uv: np.ndarray):
    """Interpolates the mesh's per-vertex panorama UV (glTF V, 0 = image bottom)
    over the top-down grid. U wraps at the longitude seam, so it is interpolated as
    an angle (cos, sin); V is interpolated directly and returned as an image row
    fraction (0 = top), the convention bake_terrain_texture samples with."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    xz = vertices[:, [0, 2]]
    ang = uv[:, 0] * 2.0 * np.pi
    values = np.stack([np.cos(ang), np.sin(ang), 1.0 - uv[:, 1]], axis=-1)
    linear = LinearNDInterpolator(xz, values)
    nearest = NearestNDInterpolator(xz, values)

    def at(X, Z):
        pts = np.stack([X.ravel(), Z.ravel()], axis=-1)
        out = linear(pts)
        missing = np.isnan(out[:, 0])
        if missing.any():
            out[missing] = nearest(pts[missing])
        u = (np.arctan2(out[:, 1], out[:, 0]) / (2.0 * np.pi)) % 1.0
        v = out[:, 2]
        return u.reshape(X.shape), v.reshape(X.shape)
    return at


# ── scene assembly ───────────────────────────────────────────────────────────────

class SceneBuilder:
    def __init__(self, context, scene: dict, args, log):
        self.context = context
        self.scene = scene
        self.args = args
        self.log = log
        self._trimesh_cache: dict[str, trimesh.Trimesh | None] = {}
        self._image_cache: dict[str, PILImage.Image | None] = {}

    def trimesh_for(self, key: str) -> trimesh.Trimesh | None:
        if key not in self._trimesh_cache:
            m = self.context.input_mesh(key)
            self._trimesh_cache[key] = m.mesh if m is not None else None
            if m is None:
                self.log(f"  missing mesh {key}")
        return self._trimesh_cache[key]

    def image_for(self, key: str) -> PILImage.Image | None:
        if key not in self._image_cache:
            im = self.context.input_image(key)
            self._image_cache[key] = im.image if im is not None else None
            if im is None:
                self.log(f"  missing texture {key}")
        return self._image_cache[key]

    def build(self, camera_pos: np.ndarray, vfov_rad: float) -> tuple[pyrender.Scene, dict]:
        from pipeline.pipeline_context import ContextKey

        ps = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=self._ambient())
        stats = {"instances": 0, "lod_mesh": 0, "lod_card": 0, "lod_billboard": 0, "billboards": 0, "env": 0, "skipped": 0}
        instances: dict[tuple, list[np.ndarray]] = {}
        env_names = {"terrain", "water", "treeline", "backdrop"}

        for o in self.scene["objects"]:
            name = str(o.get("name") or "")
            if name in env_names or name.startswith("formation"):
                continue
            if self.args.no_grass and "grass" in name:
                continue
            if self.args.only and self.args.only not in name:
                continue
            if self.args.exclude and self.args.exclude in name:
                continue
            pos = vec(o.get("position"))
            rot = vec(o.get("rotation"))
            scl = vec(o.get("scale"), (1.0, 1.0, 1.0))
            kind = o.get("type", "mesh")
            lods = o.get("lods") or []
            if lods:
                dist = max(float(np.linalg.norm(np.asarray(pos) - camera_pos)), 0.1)
                fraction = scl[1] / (dist * 2.0 * math.tan(vfov_rad / 2.0))
                chosen = lods[-1]
                for level in lods:
                    if fraction >= float(level.get("screenHeight", 0.0)):
                        chosen = level
                        break
                base = transform(pos, rot, scl)
                child = base @ transform(vec(chosen.get("position")), (0, 0, 0), vec(chosen.get("scale"), (1, 1, 1)))
                if chosen.get("mesh"):
                    instances.setdefault(("mesh", chosen["mesh"]), []).append(child)
                    stats["lod_card" if chosen["mesh"].endswith("_card") else "lod_mesh"] += 1
                elif chosen.get("texture"):
                    world = (base @ np.array([*vec(chosen.get("position")), 1.0]))[:3]
                    s = np.asarray(scl) * np.asarray(vec(chosen.get("scale"), (1, 1, 1)))
                    pose = transform(world, (0.0, yaw_to_face(world, camera_pos), 0.0), (s[0], s[1], 1.0))
                    instances.setdefault(("billboard", chosen["texture"]), []).append(pose)
                    stats["lod_billboard"] += 1
                stats["instances"] += 1
                continue
            if kind == "mesh" and o.get("mesh"):
                instances.setdefault(("mesh", o["mesh"]), []).append(transform(pos, rot, scl))
                stats["instances"] += 1
            elif o.get("texture"):
                pose = transform(pos, (0.0, yaw_to_face(pos, camera_pos), 0.0), (scl[0], scl[1], 1.0))
                instances.setdefault(("billboard", o["texture"]), []).append(pose)
                stats["billboards"] += 1
            else:
                stats["skipped"] += 1

        for (kind, key), poses in instances.items():
            if kind == "mesh":
                tm = self.trimesh_for(key)
                if tm is None:
                    stats["skipped"] += len(poses)
                    continue
                mat = material_for(tm)
                geom = to_gl(tm)
                smooth = False
                if key.endswith("_card"):
                    # Crossed cards carry both windings; averaged normals cancel and
                    # flat ones light one side black. The server exports bent normals
                    # (up + 0.5 x outward from the card's axis); rebuild those here.
                    vv = np.asarray(geom.vertices)
                    out = vv[:, [0, 2]] - vv[:, [0, 2]].mean(axis=0)
                    out /= np.linalg.norm(out, axis=1, keepdims=True).clip(1e-6)
                    nrm = np.stack([0.5 * out[:, 0], np.ones(len(vv)), 0.5 * out[:, 1]], axis=-1)
                    geom.vertex_normals = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
                    smooth = True
            else:
                smooth = False
                img = self.image_for(key)
                if img is None:
                    stats["skipped"] += len(poses)
                    continue
                geom = to_gl(quad())
                mat = material_for(geom, image=img)
            gl_poses = np.stack([_MIRROR @ p @ _MIRROR for p in poses])
            ps.add(pyrender.Mesh.from_trimesh(geom, material=mat, poses=gl_poses, smooth=smooth))

        # Environment meshes.
        terrain_pose = None
        for o in self.scene["objects"]:
            name = str(o.get("name") or "")
            if not (name in env_names or name.startswith("formation")):
                continue
            key = o.get("mesh")
            tm = self.trimesh_for(key) if key else None
            if tm is None:
                continue
            pose = transform(vec(o.get("position")), vec(o.get("rotation")), vec(o.get("scale"), (1, 1, 1)))
            if name == "terrain":
                terrain_pose = pose
                tm = tm.copy()
                baked = self._terrain_texture(tm)
                if baked is not None:
                    tex, uv = baked
                    tm.visual = trimesh.visual.TextureVisuals(uv=uv)
                    mat = material_for(tm, image=tex, double_sided=False, roughness=0.95)
                else:
                    mat = material_for(tm, double_sided=False)
                if self._pano_pass is not None:
                    # Second pass for the panorama layer: the same mesh, textured with
                    # the panorama through its own baked UV (TEXCOORD_1, what Unity
                    # samples per fragment), weighted per vertex by the blend map.
                    # A top-down bake cannot paint a near-vertical face; this can.
                    pano_tm, pano_mat = self._pano_pass
                    self._pano_node = (pano_tm, pano_mat, _MIRROR @ pose @ _MIRROR)
                self._plate_nodes = [(t, m, _MIRROR @ pose @ _MIRROR) for t, m in getattr(self, "_plate_passes", [])]
            elif name == "water":
                mat = material_for(tm, blend=True, color=[0.10, 0.30, 0.45, 0.90], roughness=0.1)
            elif name in ("treeline", "backdrop"):
                mat = material_for(tm, unlit=True)
            else:
                mat = material_for(tm)
            ps.add(pyrender.Mesh.from_trimesh(to_gl(tm), material=mat, poses=np.array([_MIRROR @ pose @ _MIRROR]), smooth=False))
            stats["env"] += 1

        for plate_tm, plate_mat, gl_pose in getattr(self, "_plate_nodes", []):
            g = to_gl(plate_tm)
            ps.add(pyrender.Mesh(primitives=[pyrender.Primitive(
                positions=np.asarray(g.vertices, dtype=np.float32),
                normals=np.asarray(g.vertex_normals, dtype=np.float32),
                texcoord_0=np.asarray(plate_tm.visual.uv, dtype=np.float32),
                color_0=plate_tm.metadata["pano_weight"],
                indices=np.asarray(g.faces, dtype=np.uint32),
                material=plate_mat,
            )]), pose=gl_pose)
        self._plate_nodes = []
        if getattr(self, "_pano_node", None) is not None:
            pano_tm, pano_mat, gl_pose = self._pano_node
            g = to_gl(pano_tm)
            prim = pyrender.Primitive(
                positions=np.asarray(g.vertices, dtype=np.float32),
                normals=np.asarray(g.vertex_normals, dtype=np.float32),
                texcoord_0=np.asarray(pano_tm.visual.uv, dtype=np.float32),
                color_0=pano_tm.metadata["pano_weight"],
                indices=np.asarray(g.faces, dtype=np.uint32),
                material=pano_mat,
            )
            ps.add(pyrender.Mesh(primitives=[prim]), pose=gl_pose)
            self._pano_node = None
        self._add_sun(ps)
        return ps, stats

    def _terrain_texture(self, tm: trimesh.Trimesh):
        from pipeline.pipeline_context import ContextKey
        splat = self.context.input_splat_material(ContextKey.TERRAIN_MATERIAL)
        if self.args.material is not None:
            from scene.splat_material import SplatMaterial
            splat = SplatMaterial.load(self.args.material)
            print(f"terrain material overridden from {self.args.material}: {[l.name for l in splat.layers]}")
        hm = self.context.input_depth(ContextKey.HEIGHT_MAP)
        params = self.context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}
        self._pano_pass = None
        self._plate_passes = []
        if splat is None or hm is None or not splat.layers:
            self.log("  terrain: no splat material / height map, using the GLB's own material")
            return None
        half = float(params.get("grid_size_meters", 200.0)) / 2.0
        t = time.time()
        terrain = self.context.input_mesh(ContextKey.TERRAIN_MESH)
        two_pass = self.args.pano_uv == "baked" and terrain is not None and terrain.extra_uv is not None
        if two_pass:
            # Pass 1 bakes only the tiled layers (panorama weight zeroed, so the
            # remainder is what the tiles paint); pass 2 draws the panorama layer
            # through TEXCOORD_1 with its blend weight as a vertex colour.
            tex = bake_terrain_texture(splat, hm.depth.astype(np.float32), half, int(self.args.terrain_texture), skip_equirect=True)
            self._pano_pass = self._panorama_pass_mesh(splat, terrain, half)
            self._plate_passes = self._plate_pass_meshes(splat, terrain, half)
        else:
            tex = bake_terrain_texture(splat, hm.depth.astype(np.float32), half, int(self.args.terrain_texture))
        self.log(f"  terrain: baked {len(splat.layers)} layer(s) to {tex.size[0]} px in {time.time() - t:.1f} s"
                 f" (panorama layer {'drawn through the mesh' + chr(39) + 's baked UV in a second pass' if two_pass else 'from the analytic projection'})")
        v = np.asarray(tm.vertices)
        u = (v[:, 0] + half) / (2.0 * half)
        w = (v[:, 2] + half) / (2.0 * half)  # 0 at z = -half == image row 0 (top)
        uv = np.stack([u, 1.0 - w], axis=-1)  # GL: v = 1 is the image's top row
        return tex, uv

    def _plate_pass_meshes(self, splat, terrain, half: float):
        """One overlay per no-macro tiled layer (the near-ground plate): the
        terrain again, UV = grid UV x tile_factor (what the shader samples),
        textured at the layer's own resolution and blended at its weight. The
        alpha is weight / (1 - panorama weight) because the panorama overlay is
        drawn over this one."""
        out = []
        pano_i = next((i for i, l in enumerate(splat.layers) if l.equirect), None)
        v = np.asarray(terrain.mesh.vertices)
        u = (v[:, 0] + half) / (2.0 * half)
        z = (v[:, 2] + half) / (2.0 * half)

        def weight(i):
            bm = splat.blend_maps[i // 4]
            img = np.asarray(bm.convert("RGBA"), dtype=np.float32)[..., i % 4] / 255.0
            h, w_ = img.shape
            return map_coordinates(img, [(z * (h - 1)).clip(0, h - 1), (u * (w_ - 1)).clip(0, w_ - 1)], order=1, mode="nearest")

        wp = weight(pano_i) if pano_i is not None else np.zeros(len(v))
        for i, layer in enumerate(splat.layers):
            if layer.equirect or getattr(layer, "macro", True):
                continue
            w = (weight(i) / np.maximum(1.0 - wp, 1e-3)).clip(0.0, 1.0)
            keep = np.asarray(terrain.mesh.faces)
            keep = keep[(w[keep] > 1e-3).any(axis=1)]
            if len(keep) == 0:
                continue
            tf = float(layer.tile_factor)
            uv = np.stack([u * tf, -z * tf], axis=-1)  # GL v = 1 is the image's top row
            tm = trimesh.Trimesh(vertices=v * 0.999, faces=keep, process=False)
            tm.visual = trimesh.visual.TextureVisuals(uv=uv)
            tm.metadata["pano_weight"] = np.stack([np.ones_like(w), np.ones_like(w), np.ones_like(w), w], axis=-1).astype(np.float32)
            mat = material_for(tm, image=layer.tile.convert("RGB"), double_sided=False, roughness=0.95, blend=True)
            out.append((tm, mat))
            self.log(f"  terrain: '{layer.name}' drawn as its own overlay ({layer.tile.size[0]} px x{tf:g}, {len(keep)} faces)")
        return out

    def _panorama_pass_mesh(self, splat, terrain, half: float):
        """The terrain again, textured with the equirect layer through TEXCOORD_1 and
        weighted per vertex by that layer's blend-map weight (pyrender multiplies
        COLOR_0 into the texture)."""
        layer_index = next((i for i, l in enumerate(splat.layers) if l.equirect), None)
        if layer_index is None:
            return None
        bm = splat.blend_maps[layer_index // 4]
        w_img = np.asarray(bm.convert("RGBA"), dtype=np.float32)[..., layer_index % 4] / 255.0
        v = np.asarray(terrain.mesh.vertices)
        u = (v[:, 0] + half) / (2.0 * half)
        z = (v[:, 2] + half) / (2.0 * half)
        hgt, wid = w_img.shape
        w = map_coordinates(w_img, [(z * (hgt - 1)).clip(0, hgt - 1), (u * (wid - 1)).clip(0, wid - 1)], order=1, mode="nearest")
        uv = np.asarray(terrain.extra_uv, dtype=np.float64)
        # Pulled 0.2% toward the camera (the origin) so it passes the depth test
        # over the base terrain it duplicates; blended with the weight as alpha.
        pano_tm = trimesh.Trimesh(vertices=v * 0.998, faces=np.asarray(terrain.mesh.faces), process=False)
        pano_tm.visual = trimesh.visual.TextureVisuals(uv=uv)
        # Kept beside the mesh, not in its visual: trimesh's TextureVisuals has no
        # vertex colours, so weights set there were silently dropped and the
        # overlay drew at full opacity everywhere (the near ground then showed
        # the raw panorama smear that Unity, honouring the weight, does not).
        pano_tm.metadata["pano_weight"] = np.stack([np.ones_like(w), np.ones_like(w), np.ones_like(w), w], axis=-1).astype(np.float32)
        mat = material_for(pano_tm, image=splat.layers[layer_index].tile, double_sided=False, roughness=0.95, blend=True)
        return pano_tm, mat

    def _ambient(self):
        sky = self._sky_mean()
        return (sky * float(self.args.ambient)).tolist()

    def _sky_mean(self) -> np.ndarray:
        sky = self.context.input_panorama(self.scene.get("skybox") or "panorama_sky")
        if sky is None:
            return np.array([0.5, 0.55, 0.65])
        arr = np.asarray(sky.image.convert("RGB"), dtype=np.float32) / 255.0
        upper = arr[: arr.shape[0] // 2]
        return (upper.reshape(-1, 3).mean(axis=0) ** 2.2)

    def _add_sun(self, ps: pyrender.Scene):
        sun = (self.scene.get("lighting") or {}).get("sun") or {}
        d = np.asarray(sun.get("direction") or [0.3, 0.8, -0.5], dtype=np.float64)
        yaw = math.radians(float(self.scene.get("skyboxRotation") or 0.0))
        cy, sy = math.cos(yaw), math.sin(yaw)
        d = np.array([cy * d[0] + sy * d[2], d[1], -sy * d[0] + cy * d[2]])
        d /= max(np.linalg.norm(d), 1e-6)
        d_gl = np.array([-d[0], d[1], d[2]])
        # pyrender lights shine down their local -Z: make +Z point toward the sun.
        z = d_gl
        x = np.cross([0.0, 1.0, 0.0], z)
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        pose = np.eye(4)
        pose[:3, 0], pose[:3, 1], pose[:3, 2] = x, y, z
        colour = sun.get("color") or "#ffffff"
        rgb = np.array([int(colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]) if colour.startswith("#") else np.ones(3)
        intensity = float(sun.get("intensity") or 1.0) * float(self.args.sun_scale)
        ps.add(pyrender.DirectionalLight(color=rgb, intensity=intensity), pose=pose)
        self.log(f"  sun: toward {np.round(d, 3)}, intensity {intensity:.2f}")


# ── camera and sky ───────────────────────────────────────────────────────────────

def camera_basis(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    ext = scene.get("extrinsics") or {}
    R = np.array(ext.get("rotation") or np.eye(3).ravel(), dtype=np.float64).reshape(3, 3)
    t = np.array(ext.get("translation") or [0, 0, 0], dtype=np.float64)
    return R, t


def view_pose(R: np.ndarray, t: np.ndarray, view: str) -> tuple[np.ndarray, np.ndarray]:
    """Camera rotation (columns right, up, forward) and position in the server frame."""
    yaw = {"photo": 0.0, "left": 45.0, "right": -45.0, "back": 180.0, "step": 0.0, "high": 0.0, "down": 0.0}.get(view, 0.0)
    pitch = {"high": -20.0, "down": 40.0}.get(view, 0.0)
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rv = R @ Ry @ Rx
    pos = t.copy()
    if view == "step":
        pos = pos + Rv[:, 2] * 2.0
    if view == "high":
        pos = pos + np.array([0.0, 2.5, 0.0]) + Rv[:, 2] * 1.0
    return Rv, pos


def gl_camera_pose(Rv: np.ndarray, pos: np.ndarray) -> np.ndarray:
    right, up, forward = Rv[:, 0], Rv[:, 1], Rv[:, 2]
    m = np.array([-1.0, 1.0, 1.0])
    pose = np.eye(4)
    pose[:3, 0] = right * m
    pose[:3, 1] = up * m
    pose[:3, 2] = -(forward * m)
    pose[:3, 3] = pos * m
    return pose


def render_sky(sky_img: PILImage.Image, yaw_deg: float, Rv: np.ndarray, width: int, height: int,
               fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    arr = np.asarray(sky_img.convert("RGB"), dtype=np.float32)
    H, W = arr.shape[:2]
    j, i = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    d = np.stack([(j - cx) / fx, -(i - cy) / fy, np.ones_like(j)], axis=-1)
    d = d @ Rv.T
    lon = np.arctan2(d[..., 0], d[..., 2]) - math.radians(yaw_deg)
    lat = np.arctan2(d[..., 1], np.hypot(d[..., 0], d[..., 2]))
    cols = np.mod((lon + np.pi) / (2 * np.pi), 1.0) * W - 0.5
    rows = ((0.5 - lat / np.pi) * H - 0.5).clip(0, H - 1)
    out = np.stack([map_coordinates(arr[..., c], [rows.ravel(), cols.ravel()], order=1, mode="wrap").reshape(height, width) for c in range(3)], axis=-1)
    return out.clip(0, 255).astype(np.uint8)


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: scratch/render/<run-id>)")
    ap.add_argument("--config", type=Path, default=ROOT / "server" / "config.yaml")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--views", default="photo,left,right,step", help="comma list of photo,left,right,back,step,high,down")
    ap.add_argument("--terrain-texture", type=int, default=2048, help="baked terrain texture size (px)")
    ap.add_argument("--no-grass", action="store_true")
    ap.add_argument("--material", type=Path, default=None, help="use this saved splat_material directory instead of the run's")
    ap.add_argument("--only", default="", help="render only placed objects whose name contains this (environment meshes always render)")
    ap.add_argument("--exclude", default="", help="skip placed objects whose name contains this")
    ap.add_argument("--sun-scale", type=float, default=2.0, help="directional light intensity per unit of the scene's sun intensity")
    ap.add_argument("--ambient", type=float, default=0.45, help="ambient as a fraction of the sky's mean colour")
    ap.add_argument("--exposure", type=float, default=1.0)
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--pano-uv", choices=["baked", "analytic"], default="baked",
                    help="terrain panorama layer through the mesh's baked TEXCOORD_1 (what Unity samples) or the analytic projection")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    # Not inside the run directory: the context loader treats its subdirectories
    # as stage caches.
    out = (args.out or (ROOT / "scratch" / "render" / run_dir.name)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731

    t0 = time.time()
    context = load_context(run_dir, args.config)
    from pipeline.pipeline_context import ContextKey
    scene_path = run_dir / "Scene Generation" / "scene.json"
    scene = json.loads(scene_path.read_text())
    log(f"scene: {len(scene['objects'])} objects (loaded in {time.time() - t0:.1f} s)")

    intr = context.input_intrinsics(ContextKey.INTRINSICS)
    photo = context.input_image(ContextKey.INPUT)
    if intr is not None:
        aspect = intr.height / intr.width
        fx = intr.fx * args.width / intr.width
        fy = intr.fy * args.width / intr.width
    else:
        aspect = 2.0 / 3.0
        fx = fy = args.width / (2.0 * math.tan(math.radians(30.0)))
    width = int(args.width)
    height = int(round(width * aspect))
    cx, cy = width / 2.0, height / 2.0
    vfov = 2.0 * math.atan(height / (2.0 * fy))

    R, t = camera_basis(scene)
    sky = context.input_panorama(scene.get("skybox") or "panorama_sky")
    yaw = float(scene.get("skyboxRotation") or 0.0)

    renderer = pyrender.OffscreenRenderer(width, height)
    renderer._renderer._program_cache = ShaderProgramCache(shader_dir=cutout_shader_dir())
    flags = pyrender.RenderFlags.NONE if args.no_shadows else pyrender.RenderFlags.SHADOWS_DIRECTIONAL

    builder = SceneBuilder(context, scene, args, log)
    results = {}
    for view in [v.strip() for v in args.views.split(",") if v.strip()]:
        Rv, pos = view_pose(R, t, view)
        t1 = time.time()
        ps, stats = builder.build(pos, vfov)
        cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=float(scene.get("farClipPlane") or 300.0) * 1.5)
        ps.add(cam, pose=gl_camera_pose(Rv, pos))
        colour, depth = renderer.render(ps, flags=flags)
        colour = colour[..., :3].astype(np.float32)
        if args.exposure != 1.0:
            colour = (colour / 255.0 * args.exposure).clip(0, 1) * 255.0
        colour = colour.astype(np.uint8)
        if sky is not None:
            bg = render_sky(sky.image, yaw, Rv, width, height, fx, fy, cx, cy)
            colour = np.where((depth <= 0.0)[..., None], bg, colour)
        img = PILImage.fromarray(colour, "RGB")
        img.save(out / f"{view}.png")
        results[view] = img
        log(f"{view}: {stats} in {time.time() - t1:.1f} s -> {out / f'{view}.png'}")

    if "photo" in results and photo is not None:
        ph = photo.image.convert("RGB").resize((width, height), PILImage.LANCZOS)
        sheet = PILImage.new("RGB", (width * 2 + 10, height + 24), (255, 255, 255))
        sheet.paste(ph, (0, 24))
        sheet.paste(results["photo"], (width + 10, 24))
        d = ImageDraw.Draw(sheet)
        d.text((4, 4), "input photograph", fill=(0, 0, 0))
        d.text((width + 14, 4), "render from the capture camera", fill=(0, 0, 0))
        sheet.save(out / "compare.png")
        log(f"compare: {out / 'compare.png'}")
    renderer.delete()
    log(f"done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
