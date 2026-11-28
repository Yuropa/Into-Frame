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
import re
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
    text = text.replace(marker, marker + "\n    if (frag_color.a < 0.02) discard;")
    # Occlusion on ambient light only, as URP does (UniversalFragmentPBR applies
    # surfaceData.occlusion to global illumination, not to the main light);
    # pyrender multiplies it into everything, which made every mapped asset
    # look far darker here than it will in Unity.
    amb = "    color.xyz += base_color.xyz * ambient_light;"
    ao_block = "    float ao = texture(material.occlusion_texture, uv_0).r;\n    color.xyz *= ao;"
    if amb in text and ao_block in text:
        text = text.replace(ao_block, "    // occlusion applied to ambient above")
        text = text.replace(amb, "#ifdef HAS_OCCLUSION_TEX\n    color.xyz += base_color.xyz * ambient_light * texture(material.occlusion_texture, uv_0).r;\n#else\n" + amb + "\n#endif")
    # Vegetation cards (flagged by material_for with an emissive factor of
    # 0.001 blue): a leaf or a petal is lit from either side and scatters light
    # through itself, so the normal faces the viewer and the diffuse term wraps
    # (half-Lambert) instead of going black side-on to the sun. Rainier band21: a
    # meadow of upright cards under a 20 deg sun from the left rendered black.
    n_line = "    vec3 n = normalize(get_normal());"
    brdf = ("        vec3 res = compute_brdf(n, v, l, roughness, metallic,\n"
            "                                f0, c_diff, base_color.rgb, radiance);")
    if n_line in text and brdf in text:
        text = text.replace(n_line, n_line + "\n    bool veg = material.emissive_factor.b > 0.0005 && material.emissive_factor.b < 0.002;\n"
                            "    if (!gl_FrontFacing) n = -n;")
        text = text.replace(brdf, brdf + "\n        if (veg) { res = c_diff * radiance * clamp(dot(n, l) * 0.5 + 0.5, 0.0, 1.0); }")
    else:
        print("warning: pyrender's mesh.frag changed; the vegetation wrap-lighting patch did not apply", flush=True)
    # Water in motion (--animate): docs/viewer/water.glsl, the same code the web viewer runs.
    # The photographed water stays put and a moving surface is laid over it (current-borne
    # ripples, wind waves, the swell and its surf at sea, the boats' wakes). Flagged by an
    # emissive blue a hair over 1 (material_for's unlit factor is 1): 1.0004 a river or
    # lake, 1.0009 the sea. Its COLOR_0 carries data, not a tint: r the distance to the
    # shore / 400 m, g and b the current in m/s (world x, z) / 4 + 0.5.
    emis = "    emissive *= srgb_to_linear(texture(material.emissive_texture, uv_0)).rgb;"
    if emis in text and "void main()" in text:
        text = text.replace("uniform vec3 ambient_light;", "uniform vec3 ambient_light;\nuniform float anim_time;\nuniform float flow_period;\n"
                            "uniform vec2 surf;\nuniform vec4 wind;\nuniform mat4 boat_a;\nuniform mat4 boat_b;")
        text = text.replace("    color *= color_multiplier;", "    if (!(material.emissive_factor.b > 1.0002 && material.emissive_factor.b < 1.0012)) color *= color_multiplier;")
        glsl = (ROOT / "docs" / "viewer" / "water.glsl").read_text()
        text = text.replace("void main()", "#ifdef HAS_EMISSIVE_TEX\n"
                            "vec3 water_tex(vec2 uv, float bias) { return srgb_to_linear(texture(material.emissive_texture, uv, bias)).rgb; }\n"
                            + glsl + "\n#endif\n\nvoid main()", 1)
        text = text.replace(emis, "#ifdef COLOR_0_LOC\n"
                            "    if (material.emissive_factor.b > 1.0002 && material.emissive_factor.b < 1.0012) {\n"
                            "        vec3 sun_l = vec3(0.0, 1.0, 0.0);\n"
                            "        vec3 sun_c = vec3(1.0);\n"
                            "        if (n_directional_lights > 0) { sun_l = normalize(-directional_lights[0].direction); sun_c = directional_lights[0].color; }\n"
                            "        emissive *= water_colour(uv_0, frag_position, cam_pos, (color_multiplier.gb - 0.5) * 4.0,\n"
                            "                                 color_multiplier.r * 400.0, material.emissive_factor.b > 1.0006 ? 1.0 : 0.0,\n"
                            "                                 anim_time, wind, surf, boat_a, boat_b, sun_l, sun_c);\n"
                            "    } else {\n" + emis + "\n    }\n#else\n" + emis + "\n#endif")
    else:
        print("warning: pyrender's mesh.frag changed; the water animation patch did not apply", flush=True)
    frag.write_text(text)
    # Plants bend in the wind (--animate): every vertex above its instance's origin is
    # pushed downwind by the bend angle times its height (quadratic up the stem, so the
    # base stays put), the angle swinging about a downwind lean at the kind's rate and
    # swelling as gusts run across the scene at the wind's speed; the vertex drops a
    # little so the stem keeps its length; leaves flutter across the wind on their own
    # phases on top. Off unless the primitive sets sway_on.
    vert = dst / "mesh.vert"
    vt = vert.read_text()
    vmark = "    gl_Position = P * V * M * inst_m * vec4(position, 1);\n    frag_position = vec3(M * inst_m * vec4(position, 1.0));"
    if vmark in vt:
        vt = vt.replace("uniform mat4 P;", "uniform mat4 P;\nuniform float anim_time;\nuniform float sway_on;\nuniform vec4 sway;\nuniform vec4 wind;")
        vt = vt.replace(vmark, """    vec4 wp = M * inst_m * vec4(position, 1.0);
    if (sway_on > 0.5) {
        vec4 base = M * inst_m * vec4(0.0, 0.0, 0.0, 1.0);
        float h = max(wp.y - base.y, 0.0);
        float tall = max(sway.z, 0.05);
#ifdef TEXCOORD_0_LOC
#ifdef TEXCOORD_1_LOC
        if (sway_on > 1.5) {
            // A merged field of blades: each vertex carries its blade's root (TEXCOORD_0,
            // x z) and its height above it and the blade's height (TEXCOORD_1).
            base = M * inst_m * vec4(texcoord_0.x, 0.0, texcoord_0.y, 1.0);
            h = texcoord_1.x;
            tall = max(texcoord_1.y, 0.03);
        }
#endif
#endif
        float f = clamp(h / tall, 0.0, 1.5);
        float s = dot(base.xz, wind.xy);
        float gust = 1.0 - sway.w + sway.w * (0.5 + 0.5 * sin(6.2831853 * (s - wind.z * anim_time) / max(wind.w, 1.0)));
        float ph = fract(sin(dot(base.xz, vec2(12.9898, 78.233))) * 43758.5453) * 6.2831853;
        float osc = sin(6.2831853 * sway.y * anim_time + ph);
        float bend = sway.x * gust * (0.45 + 0.55 * osc);
        vec2 off = wind.xy * bend * f * f * tall;
        // Leaves and petals flutter across the wind, each on its own phase, faster than
        // the stem's swing and stronger in the gusts.
        float lp = fract(sin(dot(position.xz + position.yy, vec2(41.3, 289.1))) * 9871.0) * 6.2831853;
        float fl = sin(6.2831853 * sway.y * 3.1 * anim_time + lp) * (0.6 + 0.4 * sin(6.2831853 * sway.y * 0.7 * anim_time + lp * 1.7));
        off += vec2(-wind.y, wind.x) * sway.x * 0.22 * fl * gust * f * f * tall;
        wp.xz += off;
        wp.y -= 0.5 * dot(off, off) / max(h, 0.1);
    }
    gl_Position = P * V * wp;
    frag_position = wp.xyz;""")
        vert.write_text(vt)
    else:
        print("warning: pyrender's mesh.vert changed; the sway patch did not apply", flush=True)
    return str(dst)


# The animation clock and the per-primitive sway the patched shaders read; set by
# main() for --animate (anim_time stays 0 and nothing sways otherwise).
ANIM = {"time": 0.0, "wind": (0.0, 1.0, 2.0, 30.0), "flow_period": 2.0, "surf": (3.0, 10.0),
        "boat_a": np.zeros((4, 4), dtype=np.float32), "boat_b": np.zeros((4, 4), dtype=np.float32)}


def _patch_anim_uniforms() -> None:
    """Set the animation uniforms before each primitive is drawn (pyrender knows only
    its own)."""
    from pyrender.renderer import Renderer
    if getattr(Renderer, "_anim_patched", False):
        return
    original = Renderer._bind_and_draw_primitive

    def bind_and_draw(self, primitive, pose, program, flags):
        sw = getattr(primitive, "_sway", None)
        for name, value in (("anim_time", float(ANIM["time"])), ("flow_period", float(ANIM["flow_period"])),
                            ("surf", np.asarray(ANIM["surf"], dtype=np.float32)),
                            ("sway_on", float(getattr(primitive, "_sway_mode", 1.0)) if sw is not None else 0.0),
                            ("sway", np.asarray(sw if sw is not None else (0.0, 0.0, 1.0, 0.0), dtype=np.float32)),
                            ("wind", np.asarray(ANIM["wind"], dtype=np.float32)),
                            # Four boats' wakes, one per column (set_uniform transposes).
                            ("boat_a", np.ascontiguousarray(np.asarray(ANIM["boat_a"], dtype=np.float32).T)),
                            ("boat_b", np.ascontiguousarray(np.asarray(ANIM["boat_b"], dtype=np.float32).T))):
            try:
                program.set_uniform(name, value)
            except ValueError:
                pass      # not in this program (depth passes, a define that drops it)
        return original(self, primitive, pose, program, flags)

    Renderer._bind_and_draw_primitive = bind_and_draw
    Renderer._anim_patched = True


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


def kind_key(mesh_key: str) -> str:
    """One kind of plant across its variants and LODs (as pipeline.scene_motion.scene_motion.kind_key)."""
    return re.sub(r"(_v\d+|_card)+$", "", str(mesh_key or ""))


def quad() -> trimesh.Trimesh:
    """Unity's Quad: 1x1 in XY, centred, front face -Z."""
    v = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], dtype=np.float64)
    f = np.array([[0, 2, 1], [0, 3, 2]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    return trimesh.Trimesh(vertices=v, faces=f, visual=trimesh.visual.TextureVisuals(uv=uv), process=False)


# ── materials ────────────────────────────────────────────────────────────────────

NO_PBR_MAPS = False  # --no-pbr-maps: base colour only, for before/after comparisons


def material_for(tm: trimesh.Trimesh, image: PILImage.Image | None = None, unlit: bool = False,
                 blend: bool = False, color=None, roughness: float = 0.9, double_sided: bool = True,
                 normal_image: PILImage.Image | None = None, occlusion_image: PILImage.Image | None = None,
                 gain: float = 1.0):
    # gain: a multiplier on the albedo (Exposure Fit's object_gain: generated meshes come out
    # of TRELLIS brighter or darker than the photographed ground the lights were fitted on).
    kwargs = dict(metallicFactor=0.0, roughnessFactor=roughness, doubleSided=double_sided,
                  alphaMode="BLEND" if blend else "OPAQUE")
    img = image
    if img is None and hasattr(tm.visual, "material"):
        mat = tm.visual.material
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is not None:
        img = img.convert("RGBA")
        arr = np.asarray(img)
        src_mat = getattr(tm.visual, "material", None)
        if image is None and str(getattr(src_mat, "alphaMode", "") or "").upper() == "MASK":
            # glTF MASK: Unity cuts at alphaCutoff (0.5 by default). The shader's own
            # discard is at 0.02 for the terrain overlays, so a card's soft matte
            # edge used to draw as a translucent box.
            cut = float(getattr(src_mat, "alphaCutoff", None) or 0.5) * 255.0
            arr = arr.copy()
            arr[..., 3] = np.where(arr[..., 3] >= cut, 255, 0).astype(np.uint8)
            # A cutout is vegetation: flag it for the shader's wrapped, two-sided
            # diffuse (see cutout_shader_dir) through an emissive factor too small
            # to see. Overridden below for unlit content.
            kwargs["emissiveFactor"] = [0.0, 0.0, 0.001]
        # The exporter (scene.mesh.Mesh._clamp_samplers) writes a clamp-to-edge
        # sampler for every mesh whose UVs stay inside the texture; glTFast
        # honours it, so this does too. Under REPEAT the top edge of a card
        # blended its opaque bottom row in: a dark bar over every far tree.
        sampler = None
        uv = getattr(tm.visual, "uv", None)
        if uv is not None and len(uv) and float(np.min(uv)) >= -1e-3 and float(np.max(uv)) <= 1.0 + 1e-3:
            from pyrender.constants import GLTF
            sampler = pyrender.Sampler(wrapS=GLTF.CLAMP_TO_EDGE, wrapT=GLTF.CLAMP_TO_EDGE)
        kwargs["baseColorTexture"] = pyrender.Texture(source=arr, source_channels="RGBA", sampler=sampler)
        rgb = [float(v) for v in gain] if isinstance(gain, (list, tuple)) else [float(gain)] * 3
        if rgb != [1.0, 1.0, 1.0] and not unlit:
            kwargs["baseColorFactor"] = [*rgb, 1.0]
        if unlit:
            # The photograph's own light is in the pixels: show them as emission
            # (a second Texture object; one shared between slots loses the cutout).
            kwargs["emissiveTexture"] = pyrender.Texture(source=arr.copy(), source_channels="RGBA", sampler=sampler)
            kwargs["emissiveFactor"] = [1.0, 1.0, 1.0]
            kwargs["baseColorFactor"] = [0.0, 0.0, 0.0, 1.0]
    elif color is not None:
        kwargs["baseColorFactor"] = list(color)
    else:
        kwargs["baseColorFactor"] = [0.6, 0.6, 0.6, 1.0]
    # The glTF PBR maps Material Maps writes (what glTFast reads in Unity), unless
    # the caller supplied its own image or the surface is drawn unlit.
    src = getattr(tm.visual, "material", None) if image is None and not unlit and not NO_PBR_MAPS else None
    if src is not None:
        def _tex(img, channels):
            return pyrender.Texture(source=np.asarray(img.convert(channels)), source_channels=channels)
        # Normal maps are off by default: pyrender's normal-map path shades a mesh
        # differently even with a perfectly flat map (the Rainier tree came out 15%
        # darker with every texel at 128,128,255), so it cannot judge them; Unity's
        # glTFast follows the glTF spec. RENDER_PBR_MAPS=n,o,mr turns them on.
        _maps = os.environ.get("RENDER_PBR_MAPS", "o,mr").split(",")
        if getattr(src, "normalTexture", None) is not None and "n" in _maps:
            # to_gl mirrors X; glTFast keeps the tangent frame right through the
            # tangent's w sign, pyrender's derivative frame (b = n x t) does not,
            # which inverts the bitangent: flip green to compensate.
            na = np.asarray(src.normalTexture.convert("RGB")).copy()
            na[..., 1] = 255 - na[..., 1]
            kwargs["normalTexture"] = pyrender.Texture(source=na, source_channels="RGB")
        if getattr(src, "occlusionTexture", None) is not None and "o" in _maps:
            kwargs["occlusionTexture"] = _tex(src.occlusionTexture, "RGB")
        if getattr(src, "metallicRoughnessTexture", None) is not None and getattr(src, "normalTexture", None) is not None and "mr" in _maps:
            # Only alongside the other maps: TRELLIS's own metallic-roughness texture
            # alone marks plants as metal, which the client shows but this renderer
            # historically did not; Material Maps replaces it.
            # pyrender reads roughness from R and metallic from G; glTF (and glTFast)
            # store them in G and B. Swizzle, or every asset renders ~80% metallic.
            mr = np.asarray(src.metallicRoughnessTexture.convert("RGB"))
            kwargs["metallicRoughnessTexture"] = pyrender.Texture(
                source=np.stack([mr[..., 1], mr[..., 2], mr[..., 2]], axis=-1).astype(np.uint8), source_channels="RGB")
            kwargs["metallicFactor"] = 1.0
            kwargs["roughnessFactor"] = 1.0
    if normal_image is not None and not NO_PBR_MAPS and "n" in os.environ.get("RENDER_PBR_MAPS", "o,mr").split(","):
        kwargs["normalTexture"] = pyrender.Texture(source=np.asarray(normal_image.convert("RGB")), source_channels="RGB")
    if occlusion_image is not None and not NO_PBR_MAPS:
        kwargs["occlusionTexture"] = pyrender.Texture(source=np.asarray(occlusion_image.convert("RGB")), source_channels="RGB")
    return pyrender.MetallicRoughnessMaterial(**kwargs)


# ── steep terrain faces: quay walls, cliffs ─────────────────────────────────────
# The terrain is textured from above (the splat bake at the grid UV), so a face
# that stands up wears one row of texels stretched down it: vertical streaks.
# Faces steeper than this are drawn as their own mesh, projected from the side
# (box-mapped on the wall's horizontal run and its height) with a wall texture.
WALL_STEEP_NY = 0.5          # |normal.y| below this (steeper than 60 degrees)
WALL_FRINGE_NY = 0.85        # and the faces sharing a vertex with one, up to here: the
                             # grid's diagonals cut the wall's top and foot into teeth
WALL_TILE_M = (4.0, 8.0)     # the wall texture's width and height on the face (m)
WALL_PX_PER_M = 128
# The same test as TerrainTextureGenerationStage's _is_urban_scene (its module
# imports the diffusion stack): a caption naming a city lays its walls in courses.
URBAN_WORDS = frozenset(
    "city cities town urban street streets quay quays quayside embankment promenade "
    "harbour harbor plaza piazza square bridge cathedral church palace castle eiffel "
    "downtown boulevard avenue pier docks dock marina canal skyline buildings village "
    "market station monument temple museum parliament".split()
)


def steep_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    ny = np.abs(n[:, 1])
    hard = ny < WALL_STEEP_NY
    if not hard.any():
        return hard
    on_wall = np.zeros(len(vertices), dtype=bool)
    on_wall[np.asarray(faces)[hard].ravel()] = True
    return hard | ((ny < WALL_FRINGE_NY) & on_wall[np.asarray(faces)].any(axis=1))


def _tileable_noise(rng, shape, cells, order=3) -> np.ndarray:
    """Smooth noise over `shape` that wraps on both axes, `cells` (rows, cols) lattice points."""
    grid = rng.standard_normal(cells)
    rows = (np.arange(shape[0]) + 0.5) / shape[0] * cells[0] - 0.5
    cols = (np.arange(shape[1]) + 0.5) / shape[1] * cells[1] - 0.5
    R, C = np.meshgrid(rows, cols, indexing="ij")
    return map_coordinates(grid, [R, C], order=order, mode="grid-wrap")


def wall_texture(tint, ashlar: bool, wet: bool, seed: int = 7) -> PILImage.Image:
    """A WALL_TILE_M texture that tiles across and up: dressed stone in courses
    (ashlar) or stratified rock, `tint` the albedo; bottom row = the waterline,
    with a darker wet band above it when `wet`."""
    rng = np.random.default_rng(seed)
    W, H = int(WALL_TILE_M[0] * WALL_PX_PER_M), int(WALL_TILE_M[1] * WALL_PX_PER_M)
    m = 1.0 / WALL_PX_PER_M
    fine = sum(_tileable_noise(rng, (H, W), (int(H * m * f), int(W * m * f))) / f for f in (6, 12, 24)) * 0.5
    shade = np.ones((H, W))
    if ashlar:
        course = 0.5
        nc = int(round(WALL_TILE_M[1] / course))
        ys = (np.arange(H) + 0.5) * m              # metres above the waterline, bottom row first
        ci = np.minimum((ys / course).astype(int), nc - 1)
        yin = ys - ci * course
        xs = (np.arange(W) + 0.5) * m
        joint = np.full((H, W), 1e9)
        block = np.zeros((H, W))
        for c in range(nc):
            # Blocks of 0.8 to 1.5 m, the course's joints staggered; the run wraps at the tile's width.
            cuts, x = [], rng.uniform(0.0, WALL_TILE_M[0])
            while len(cuts) == 0 or (x - cuts[0]) < WALL_TILE_M[0] - 0.8:
                cuts.append(x)
                x += rng.uniform(0.8, 1.5)
            cuts = np.mod(np.asarray(cuts), WALL_TILE_M[0])
            rows = ci == c
            d = np.abs(xs[None, :] - cuts[:, None])
            d = np.minimum(d, WALL_TILE_M[0] - d).min(axis=0)
            joint[rows] = np.minimum(d[None, :], np.minimum(yin[rows], course - yin[rows])[:, None])
            idx = np.searchsorted(np.sort(cuts), xs) % len(cuts)
            block[rows] = rng.normal(0.0, 0.06, len(cuts))[idx][None, :]
        shade = (1.0 + block) * np.where(joint < 0.012, 0.6, 1.0) * np.where(joint < 0.035, 0.9 + 0.1 * (joint / 0.035), 1.0)
        shade *= 1.0 + 0.10 * fine
    else:
        strata = _tileable_noise(rng, (H, W), (int(WALL_TILE_M[1] * 3), 3)) * 0.18
        shade = 1.0 + strata + 0.16 * fine
    rgb = np.asarray(tint, dtype=np.float64)[None, None, :] * shade[..., None]
    if wet:
        ys = (np.arange(H) + 0.5) * m
        edge = 0.75 + 0.25 * _tileable_noise(rng, (1, W), (1, int(WALL_TILE_M[0] * 2)))[0]
        t = np.clip((ys[:, None] - edge[None, :] + 0.15) / 0.3, 0.0, 1.0)
        wet_k = 1.0 - (1.0 - t) * 0.45
        rgb = rgb * wet_k[..., None] * np.where(t[..., None] < 1.0, np.array([0.92, 0.97, 0.88]), 1.0)
    img = (rgb.clip(0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)[::-1]   # image row 0 is the top
    return PILImage.fromarray(img, "RGB")


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


def bake_terrain_pbr(splat, height_map: np.ndarray, half: float, size: int,
                     height_blend_sharpness: float = 4.0, plate_layers: bool = False):
    """The layers' packed PBR detail (see SplatLayer.pbr) blended over the same
    top-down grid as bake_terrain_texture: (normal RGB image, occlusion RGB
    image), or None when no layer has any. plate_layers=False leaves the no-macro
    layers out (they are drawn as their own overlays); the panorama layer counts
    as flat. Normals keep the tile's own encoding, which is also the bake's:
    image rows run with the tile's rows."""
    layers = list(splat.layers)
    if not any(getattr(l, "pbr", None) is not None for l in layers):
        return None
    us = (np.arange(size, dtype=np.float64) + 0.5) / size
    U, V = np.meshgrid(us, us)

    def sample(img, u, v, wrap):
        arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
        h, w = arr.shape[:2]
        if wrap:
            cols, rows, mode = np.mod(u, 1.0) * w - 0.5, np.mod(v, 1.0) * h - 0.5, "wrap"
        else:
            cols, rows, mode = u * (w - 1), v * (h - 1), "nearest"
        return np.stack([map_coordinates(arr[..., c], [rows.ravel(), cols.ravel()], order=1, mode=mode).reshape(u.shape)
                         for c in range(4)], axis=-1)

    acc = np.zeros((size, size, 4), dtype=np.float32); wsum = np.zeros((size, size), dtype=np.float32)
    for i, layer in enumerate(layers):
        if not plate_layers and not getattr(layer, "macro", True):
            continue
        bm = splat.blend_maps[i // 4] if i // 4 < len(splat.blend_maps) else None
        if bm is None:
            continue
        w = sample(bm, U, V, wrap=False)[..., i % 4]
        if layer.equirect:
            pbr = np.zeros((size, size, 4), dtype=np.float32); pbr[..., :2] = 0.5; pbr[..., 2] = 1.0
            tex_a = np.ones((size, size), dtype=np.float32)
        else:
            tf = float(layer.tile_factor)
            tex_a = sample(layer.tile, U * tf, V * tf, wrap=True)[..., 3]
            if getattr(layer, "pbr", None) is not None:
                pbr = sample(layer.pbr, U * tf, V * tf, wrap=True)
            else:
                pbr = np.zeros((size, size, 4), dtype=np.float32); pbr[..., :2] = 0.5; pbr[..., 2] = 1.0
        b = w * np.power(tex_a + 1e-4, height_blend_sharpness)
        acc += pbr * b[..., None]; wsum += b
    acc /= np.maximum(wsum, 1e-4)[..., None]
    nxy = acc[..., :2] * 2.0 - 1.0
    nz = np.sqrt(np.clip(1.0 - (nxy ** 2).sum(-1), 0.0, 1.0))
    normal = np.concatenate([nxy, nz[..., None]], axis=-1) * 0.5 + 0.5
    occ = np.repeat(acc[..., 2:3], 3, axis=-1)
    to_img = lambda a: PILImage.fromarray((a.clip(0, 1) * 255.0 + 0.5).astype(np.uint8), "RGB")
    return to_img(normal), to_img(occ)


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

# --wireframe: the triangle edges drawn over the frame, one colour per kind of
# geometry (the grass blades are left out: 60 000 tufts are a solid sheet of lines).
WIRE_COLOURS = {
    "terrain": (255, 255, 255),
    "water": (0, 225, 255),
    "backdrop": (255, 70, 230),
    "formation": (120, 255, 90),
    "object": (255, 225, 0),
    "card": (255, 140, 0),
    "billboard": (255, 60, 60),
    "wall": (190, 190, 255),
}


def render_wire(renderer, ps, node_colours: dict) -> tuple[np.ndarray, np.ndarray]:
    """The scene's triangle edges, each node in its flat colour, and their depth.
    pyrender's segmentation pass gives the flat colours and skips the lights, but
    fills the triangles and culls back faces whatever the flags say, so for this one
    pass its polygon mode is lines and its culling is off."""
    import pyrender.renderer as pr
    from OpenGL import GL
    real_mode, real_enable = pr.glPolygonMode, pr.glEnable
    pr.glPolygonMode = lambda face, mode: real_mode(face, GL.GL_LINE)
    pr.glEnable = lambda cap: GL.glDisable(cap) if cap == GL.GL_CULL_FACE else real_enable(cap)
    try:
        return renderer.render(ps, flags=pyrender.RenderFlags.SEG, seg_node_map=node_colours)
    finally:
        pr.glPolygonMode, pr.glEnable = real_mode, real_enable
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)


def render_coverage(renderer, ps) -> np.ndarray:
    """Every node flat white, multisampled: each pixel's grey is how much of it the
    geometry covers. Culling off, as for the wireframe: the backdrop and the cards are
    double-sided and the segmentation pass culls back faces whatever the flags say."""
    import pyrender.renderer as pr
    from OpenGL import GL
    real_enable = pr.glEnable
    pr.glEnable = lambda cap: GL.glDisable(cap) if cap == GL.GL_CULL_FACE else real_enable(cap)
    try:
        white, _ = renderer.render(ps, flags=pyrender.RenderFlags.SEG,
                                   seg_node_map={n: (255, 255, 255) for n in ps.mesh_nodes})
        return np.asarray(white)
    finally:
        pr.glEnable = real_enable


def wire_overlay(frame: np.ndarray, seg: np.ndarray, wire_depth: np.ndarray, depth: np.ndarray,
                 alpha: float = 0.6, legend: bool = True) -> PILImage.Image:
    """`frame` with the wireframe pass drawn over it. An edge is drawn where it is
    the nearest surface: its depth is the solid pass's depth at that pixel (within
    1%), or there is nothing solid there (the cut-out part of a card or billboard
    quad, which is geometry all the same)."""
    on = (wire_depth > 0.0) & ((depth <= 0.0) | (wire_depth <= depth * 1.01 + 0.02))
    out = frame.astype(np.float32)
    out[on] = out[on] * (1.0 - alpha) + seg[..., :3][on].astype(np.float32) * alpha
    img = PILImage.fromarray(out.clip(0, 255).astype(np.uint8), "RGB")
    if legend:
        d = ImageDraw.Draw(img)
        x, y = 8, img.height - 16
        for name, colour in WIRE_COLOURS.items():
            w = int(d.textlength(name)) + 22
            d.rectangle((x - 3, y - 3, x + w - 6, y + 12), fill=(0, 0, 0))
            d.rectangle((x, y + 1, x + 8, y + 9), fill=colour)
            d.text((x + 12, y - 1), name, fill=(255, 255, 255))
            x += w
    return img


class SceneBuilder:
    def __init__(self, context, scene: dict, args, log):
        self.context = context
        self.scene = scene
        self.args = args
        self.log = log
        self._trimesh_cache: dict[str, trimesh.Trimesh | None] = {}
        self._image_cache: dict[str, PILImage.Image | None] = {}

    def _mesh_height(self, key) -> float:
        """The Y extent of a mesh in its own units (1 when it cannot be read)."""
        cache = self.__dict__.setdefault("_mesh_heights", {})
        if key not in cache:
            tm = self.trimesh_for(key) if key else None
            h = float(tm.extents[1]) if tm is not None and len(tm.vertices) else 1.0
            cache[key] = h if h > 1e-6 else 1.0
        return cache[key]

    def trimesh_for(self, key: str) -> trimesh.Trimesh | None:
        if key not in self._trimesh_cache:
            m = self.context.input_mesh(key)
            self._trimesh_cache[key] = m.mesh if m is not None else None
            if m is None:
                self.log(f"  missing mesh {key}")
        return self._trimesh_cache[key]

    def _water_photo(self, key: str, tm: trimesh.Trimesh, colour, pose: np.ndarray | None = None) -> tuple | None:
        """The water sheet wearing the photographed water (server:
        pipeline.terrain.water_photo): the mesh's TEXCOORD_1 samples the object-free
        panorama. Its alpha is 1 where the photograph shows water; elsewhere its RGB is
        the nearby water's colour carried outward, which is what this draws there (the
        sheet's single colour factor against a photographed sea reads as a stain).
        The photograph carries its own light (the water's colour is mostly the sky it
        mirrors), so it is drawn as emission, like the photographed backdrops, on a
        fully rough surface: the photograph has the sun's glint where the sun put it,
        and a second one from the scene's light was the glitter column down Shark
        Fin's cove. None when the mesh has no second UV set or no texture: the caller
        draws the flat sheet as before."""
        m = self.context.input_mesh(key)
        uv = getattr(m, "extra_uv", None)
        if uv is None or len(uv) != len(tm.vertices):
            return None
        if not hasattr(self, "_water_photo_tex"):
            tex = getattr(m, "preview_image", None) if getattr(m, "preview_overlay", False) else None
            if tex is None:
                tex = self.image_for("water_photo_texture")
            self._water_photo_tex = tex.convert("RGB") if tex is not None else None
            if self._water_photo_tex is not None and pose is not None and getattr(self.args, "animate", 0):
                self._water_photo_tex = self._erase_photographed_wakes(self._water_photo_tex, tm, np.asarray(uv), pose)
        if self._water_photo_tex is None:
            return None
        out = trimesh.Trimesh(vertices=np.asarray(tm.vertices), faces=np.asarray(tm.faces), process=False)
        out.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv, dtype=np.float64))
        return out, material_for(out, image=self._water_photo_tex, unlit=True, roughness=1.0)

    def _erase_photographed_wakes(self, tex: PILImage.Image, tm: trimesh.Trimesh, uv: np.ndarray, pose: np.ndarray) -> PILImage.Image:
        """--animate: a boat underway leaves its photographed wake behind as it moves
        on (the white water behind Paris's tour boat stayed where the photograph has it
        while the boat sailed off). Where each boat underway lay when photographed --
        its hull and a wedge behind it, six lengths long -- the photographed water's
        foam (texels well above the water's own brightness there) is painted over from
        the water around it; water.glsl draws the wake with the boat."""
        under = [m for (_n, _p, _r, _s, m) in self.mover_nodes if m.get("type") == "floating_underway" and m.get("path")]
        if not under:
            return tex
        try:
            import cv2
            from scipy.interpolate import LinearNDInterpolator
        except ImportError as exc:
            self.log(f"  animate: photographed wakes kept ({exc})")
            return tex
        arr = np.asarray(tex.convert("RGB"))
        H, W = arr.shape[:2]
        # Each texel's place on the water: the sheet's ground plan interpolated over its uv,
        # at half resolution over the texels the sheet covers.
        v = (np.c_[np.asarray(tm.vertices), np.ones(len(tm.vertices))] @ pose.T)[:, [0, 2]]
        px = np.stack([uv[:, 0] * W, (1.0 - uv[:, 1]) * H], axis=-1)
        _, first = np.unique(np.round(px, 2), axis=0, return_index=True)
        try:
            to_world = LinearNDInterpolator(px[first], v[first])
        except Exception as exc:
            self.log(f"  animate: photographed wakes kept ({type(exc).__name__}: {exc})")
            return tex
        x0, y0 = np.floor(px.min(axis=0)).astype(int).clip(0)
        x1, y1 = np.ceil(px.max(axis=0)).astype(int)
        x1, y1 = min(x1, W), min(y1, H)
        gx, gy = np.meshgrid(np.arange(x0, x1, 2) + 1.0, np.arange(y0, y1, 2) + 1.0)
        world = to_world(np.stack([gx.ravel(), gy.ravel()], axis=-1))
        inside = np.zeros(len(world), dtype=bool)
        for m in under:
            path = np.asarray(m["path"], dtype=np.float64)
            h = path[1] - path[0]
            h /= max(float(np.linalg.norm(h)), 1e-9)
            L, B = float(m.get("_length", 10.0)), float(m.get("_beam", 3.0))
            q = world - path[0]
            s = -(q @ h)                                   # metres behind the boat's centre
            d = np.abs(q @ np.array([-h[1], h[0]]))
            inside |= np.isfinite(s) & (s > -0.6 * L) & (s < 6.5 * L) & (d < 0.5 * B + 2.5 + 0.36 * np.maximum(s - 0.5 * L, 0.0))
        small = inside.reshape(gx.shape).astype(np.uint8)
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y0:y0 + 2 * small.shape[0]:2, x0:x0 + 2 * small.shape[1]:2] = small
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        if not mask.any():
            return tex
        # Foam: bright, colourless, and well above the water around it.
        lum = arr.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        # The water's own level around each texel (a median over ~160 texels, so a patch of
        # white water does not raise its own reference).
        small_l = cv2.resize(np.clip(lum, 0, 255).astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        around = cv2.resize(cv2.medianBlur(small_l, 41), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        sat = arr.max(axis=-1).astype(np.float32) - arr.min(axis=-1).astype(np.float32)
        # Spray is fine specks: judged on the texels' neighbourhood as well as their own.
        lum_b = cv2.GaussianBlur(lum, (0, 0), 2.0)
        foam = ((mask > 0) & ((lum > around + 14.0) | (lum_b > around + 7.0)) & (sat < 0.4 * lum + 10.0)).astype(np.uint8)
        # White water comes in patches; a ripple's glint is a speck.
        foam = cv2.morphologyEx(foam, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(foam, connectivity=8)
        keep = np.zeros(n, dtype=bool)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 15
        foam = cv2.dilate(keep[lab].astype(np.uint8), np.ones((11, 11), np.uint8))
        if foam.sum() == 0:
            return tex
        # Filled with the water beside it on the same row (the same distance from the
        # camera, so the same scale of ripple), the nearest shift whose texels are clean;
        # what no shift reaches is inpainted. The seams are feathered.
        fill = arr.astype(np.float32).copy()
        todo = foam > 0
        for shift in (90, -90, 180, -180, 270, -270, 360, -360):
            src_bad = np.roll(foam, -shift, axis=1) > 0
            take = todo & ~src_bad
            if take.any():
                fill[take] = np.roll(arr, -shift, axis=1)[take]
                todo &= ~take
            if not todo.any():
                break
        if todo.any():
            fill = np.where(todo[..., None], cv2.inpaint(np.ascontiguousarray(arr), todo.astype(np.uint8), 9, cv2.INPAINT_TELEA), fill)
        wgt = cv2.GaussianBlur(foam.astype(np.float32), (0, 0), 2.0)[..., None]
        wgt = np.maximum(wgt, foam[..., None].astype(np.float32))
        out = np.clip(arr.astype(np.float32) * (1.0 - wgt) + fill * wgt, 0, 255).astype(np.uint8)
        self.log(f"  animate: photographed wake of {len(under)} boat(s) underway painted over ({int(foam.sum())} texels)")
        return PILImage.fromarray(out, "RGB")

    def _water_flow_uv(self, tm: trimesh.Trimesh, pose: np.ndarray) -> np.ndarray | None:
        """The water's flow (Scene Motion's flow field) as a velocity in the photo layer's
        UV per second at each vertex: the flow in world metres per second through the
        local Jacobian of UV over world (x, z), area-weighted over the vertex's faces.
        At sea the field is the swell's direction at unit speed; the surface drifts with
        it at a fraction of that. None without a flow field."""
        wm = (self.scene.get("motion") or {}).get("water")
        ff = getattr(self, "_flow_field", None)
        if not wm or ff is None:
            return None
        v = (np.c_[np.asarray(tm.vertices), np.ones(len(tm.vertices))] @ pose.T)[:, [0, 2]]
        uv = np.asarray(tm.visual.uv, dtype=np.float64)
        f = np.asarray(tm.faces)
        E = np.stack([v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]], axis=-1)      # [F, 2(xz), 2(edges)]
        U = np.stack([uv[f[:, 1]] - uv[f[:, 0]], uv[f[:, 2]] - uv[f[:, 0]]], axis=-1)
        det = E[:, 0, 0] * E[:, 1, 1] - E[:, 0, 1] * E[:, 1, 0]
        ok = np.abs(det) > 1e-9
        inv = np.zeros_like(E)
        inv[ok, 0, 0], inv[ok, 1, 1] = E[ok, 1, 1] / det[ok], E[ok, 0, 0] / det[ok]
        inv[ok, 0, 1], inv[ok, 1, 0] = -E[ok, 0, 1] / det[ok], -E[ok, 1, 0] / det[ok]
        J = U @ inv                                                                     # [F, 2(uv), 2(xz)]
        w = np.where(ok, 0.5 * np.abs(det), 0.0)
        Jv = np.zeros((len(v), 2, 2)); wv = np.zeros(len(v))
        for k in range(3):
            np.add.at(Jv, f[:, k], J * w[:, None, None]); np.add.at(wv, f[:, k], w)
        Jv /= np.maximum(wv, 1e-12)[:, None, None]
        flow = ff.sample(v[:, 0], v[:, 1])
        if wm.get("kind") == "sea":
            flow = flow * 0.15
        return np.einsum("nij,nj->ni", Jv, flow)

    def _water_current(self, tm: trimesh.Trimesh, pose: np.ndarray) -> np.ndarray:
        """The current at each water vertex in m/s (server x, z): the flow field; at sea
        the swell's direction at a slow drift (water.glsl takes its direction for the
        fronts and its speed for the ripples it carries)."""
        ff = getattr(self, "_flow_field", None)
        if ff is None:
            return np.zeros((len(tm.vertices), 2))
        v = (np.c_[np.asarray(tm.vertices), np.ones(len(tm.vertices))] @ pose.T)[:, [0, 2]]
        flow = np.asarray(ff.sample(v[:, 0], v[:, 1]), dtype=np.float64)
        if ((self.scene.get("motion") or {}).get("water") or {}).get("kind") == "sea":
            flow = flow * 0.15
        return np.clip(flow, -1.99, 1.99)

    def _shore_distance(self, tm: trimesh.Trimesh, pose: np.ndarray) -> np.ndarray:
        """Metres from each water vertex to the nearest dry cell of the flow field's grid
        (the surf breaks there); far (400) without a field."""
        ff = getattr(self, "_flow_field", None)
        if ff is None:
            return np.full(len(tm.vertices), 400.0)
        if not hasattr(self, "_shore_grid"):
            from scipy import ndimage
            self._shore_grid = ndimage.distance_transform_edt(ff.water) * ff.cell_m
        from scipy.ndimage import map_coordinates
        v = (np.c_[np.asarray(tm.vertices), np.ones(len(tm.vertices))] @ pose.T)[:, [0, 2]]
        r, c = ff._cell(v[:, 0], v[:, 1])
        return np.clip(map_coordinates(self._shore_grid, [r, c], order=1, mode="nearest"), 0.0, 400.0)

    def pose_movers(self, ps: pyrender.Scene, t: float) -> None:
        """Each moving object's pose at time t: an underway boat at speed along its course
        (arc length on the path, the course's heading plus the angle it was placed at),
        every boat heaving, rolling and pitching on its own period. The boats underway
        also set the wake uniforms (water.glsl): where each is and which way it heads."""
        boat_a = np.zeros((4, 4), dtype=np.float32)
        boat_b = np.zeros((4, 4), dtype=np.float32)
        n_wakes = 0
        for node, pos, rot, scl, m in self.mover_nodes:
            x, y, z = pos
            yaw = rot[1]
            if m.get("type") == "floating_underway" and m.get("path"):
                path = np.asarray(m["path"], dtype=np.float64)
                seg = np.hypot(*np.diff(path, axis=0).T)
                cum = np.r_[0.0, np.cumsum(seg)]
                d = min(float(m.get("speedMs", 2.0)) * t, float(cum[-1]))
                i = int(np.clip(np.searchsorted(cum, d) - 1, 0, len(seg) - 1))
                a = (d - cum[i]) / max(seg[i], 1e-9)
                x, z = path[i] + a * (path[i + 1] - path[i])
                tan = path[i + 1] - path[i]
                yaw = math.degrees(math.atan2(tan[0], tan[1])) + float(m.get("yawOffsetDeg", 0.0))
                if n_wakes < 4 and d < float(cum[-1]):
                    h = tan / max(float(np.linalg.norm(tan)), 1e-9)
                    # The renderer's frame mirrors x.
                    boat_a[n_wakes] = (-x, z, -h[0], h[1])
                    boat_b[n_wakes] = (m.get("_length", 10.0), m.get("_beam", 3.0), float(m.get("speedMs", 2.0)), 1.0)
                    n_wakes += 1
            b = m.get("bob") or {}
            w = 2.0 * math.pi / max(float(b.get("periodS", 3.0)), 0.5)
            ph = float(b.get("phase", 0.0))
            y = y + float(b.get("heaveM", 0.0)) * math.sin(w * t + ph)
            roll = float(b.get("rollDeg", 0.0)) * math.sin(w * t + ph + 1.3)
            pitch = float(b.get("pitchDeg", 0.0)) * math.sin(w * 0.8 * t + ph + 0.4)
            ps.set_pose(node, _MIRROR @ transform((x, y, z), (rot[0] + pitch, yaw, rot[2] + roll), scl) @ _MIRROR)
        ANIM["boat_a"], ANIM["boat_b"] = boat_a, boat_b

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
        self.wire_nodes = {}  # node -> line colour, for --wireframe
        self.key_nodes = {}   # node -> (kind, key) of the placed objects, for the fit's object-id pass
        self.movers, self.mover_nodes = [], []
        self.anim_tags = {}   # node -> "water" | "veg" (the --anim-masks labels; movers from mover_nodes)
        self.terrain_steep = None  # per terrain face: drawn by the wall mesh (see steep_faces)
        stats = {"instances": 0, "lod_mesh": 0, "lod_card": 0, "lod_billboard": 0, "billboards": 0, "env": 0, "skipped": 0}
        instances: dict[tuple, list[np.ndarray]] = {}
        env_names = {"terrain", "water", "water_apron", "treeline", "backdrop"}

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
                # The height on screen is the placed mesh's own: a species mesh is
                # modelled in metres and placed at a scale near 1, and read as
                # "scale.y metres tall" every tree in Paris drew as its card where
                # the client (a LODGroup over the mesh's bounds) draws the mesh.
                fraction = scl[1] * self._mesh_height(lods[0].get("mesh")) / (dist * 2.0 * math.tan(vfov_rad / 2.0))
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
            if kind == "mesh" and o.get("mesh") and getattr(self.args, "animate", 0) and o.get("motion"):
                # Moves on its own (a boat underway, a moored one bobbing): its own node,
                # posed each frame by pose_movers().
                self.movers.append((o["mesh"], pos, rot, scl, o["motion"]))
                stats["instances"] += 1
                continue
            if kind == "mesh" and o.get("mesh"):
                instances.setdefault(("mesh_unlit" if o.get("unlit") else "mesh", o["mesh"]), []).append(transform(pos, rot, scl))
                stats["instances"] += 1
            elif o.get("texture"):
                pose = transform(pos, (0.0, yaw_to_face(pos, camera_pos), 0.0), (scl[0], scl[1], 1.0))
                instances.setdefault(("billboard", o["texture"]), []).append(pose)
                stats["billboards"] += 1
            else:
                stats["skipped"] += 1

        for (kind, key), poses in instances.items():
            if kind in ("mesh", "mesh_unlit"):
                tm = self.trimesh_for(key)
                if tm is None:
                    stats["skipped"] += len(poses)
                    continue
                # One model's own colour gain (Exposure Fit's object_gains) stands in for the scene's.
                own_gain = (getattr(self.args, "object_gains", None) or {}).get(key)
                mat = material_for(tm, unlit=kind == "mesh_unlit",
                                   gain=own_gain if own_gain else float(getattr(self.args, "object_gain", 1.0) or 1.0))
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
                mat = material_for(geom, image=img, gain=float(getattr(self.args, "object_gain", 1.0) or 1.0))
            gl_poses = np.stack([_MIRROR @ p @ _MIRROR for p in poses])
            pm = pyrender.Mesh.from_trimesh(geom, material=mat, poses=gl_poses, smooth=smooth)
            if kind in ("mesh", "mesh_unlit") and casts_no_shadow(key):
                pm.no_shadow = True
            veg = ((self.scene.get("motion") or {}).get("vegetation") or {}).get(kind_key(key)) if getattr(self.args, "animate", 0) else None
            if veg:
                for prim in pm.primitives:
                    prim._sway = (math.radians(float(veg.get("amplitudeDeg", 3.0))), float(veg.get("frequencyHz", 0.5)),
                                  float(veg.get("heightM", 0.0)) or max(self._mesh_height(key) * float(np.median([p[1, 1] for p in poses])), 0.3),
                                  float((self.scene.get("motion") or {}).get("wind", {}).get("gustiness", 0.4)))
            node = ps.add(pm)
            if veg:
                self.anim_tags[node] = "veg"
            wire_kind = "billboard" if kind == "billboard" else ("card" if "_card" in key else "object")
            self.wire_nodes[node] = WIRE_COLOURS[wire_kind]
            self.key_nodes[node] = (kind, key)

        for key, pos, rot, scl, motion in self.movers:
            tm = self.trimesh_for(key)
            if tm is None:
                continue
            own_gain = (getattr(self.args, "object_gains", None) or {}).get(key)
            mat = material_for(tm, gain=own_gain if own_gain else float(getattr(self.args, "object_gain", 1.0) or 1.0))
            node = ps.add(pyrender.Mesh.from_trimesh(to_gl(tm), material=mat, smooth=False),
                          pose=_MIRROR @ transform(pos, rot, scl) @ _MIRROR)
            self.wire_nodes[node] = WIRE_COLOURS["object"]
            self.key_nodes[node] = ("mesh", key)
            ext = np.asarray(tm.extents, dtype=np.float64) * np.abs(np.asarray(scl, dtype=np.float64))
            motion = dict(motion, _length=float(max(ext[0], ext[2])), _beam=float(min(ext[0], ext[2])))
            self.mover_nodes.append((node, pos, rot, scl, motion))

        # Environment meshes.
        terrain_pose = None
        for o in self.scene["objects"]:
            name = str(o.get("name") or "")
            if not (name in env_names or name.startswith("formation")):
                continue
            if self.args.env_exclude and any(sub and sub in name for sub in self.args.env_exclude.split(",")):
                continue
            key = o.get("mesh")
            tm = self.trimesh_for(key) if key else None
            if tm is None:
                continue
            pose = transform(vec(o.get("position")), vec(o.get("rotation")), vec(o.get("scale"), (1, 1, 1)))
            if name == "terrain":
                terrain_pose = pose
                tm = tm.copy()
                baked = None if self.args.flat_terrain else self._terrain_texture(tm)
                if self.args.flat_terrain:
                    # Geometry alone: flat grey, faceted, so steps and folds in the
                    # height map show without the texture hiding them.
                    self._pano_pass = None; self._plate_passes = []; self._terrain_pbr = None
                    tm.visual = trimesh.visual.ColorVisuals(tm, face_colors=[170, 170, 170, 255])
                if baked is not None:
                    tex, uv = baked
                    steep = steep_faces(tm.vertices, tm.faces)
                    if steep.any() and getattr(self, "_wall_tint", None) is not None:
                        wall = self._wall_mesh(tm, steep, pose)
                        node = ps.add(pyrender.Mesh.from_trimesh(to_gl(wall[0]), material=wall[1], smooth=False),
                                      pose=_MIRROR @ pose @ _MIRROR)
                        self.wire_nodes[node] = WIRE_COLOURS["wall"]
                        self.terrain_steep = steep
                        tm = trimesh.Trimesh(vertices=tm.vertices, faces=np.asarray(tm.faces)[~steep], process=False)
                    tm.visual = trimesh.visual.TextureVisuals(uv=uv)
                    pbr_bake = getattr(self, "_terrain_pbr", None)
                    mat = material_for(tm, image=tex, double_sided=False, roughness=0.95,
                                       normal_image=pbr_bake[0] if pbr_bake else None,
                                       occlusion_image=pbr_bake[1] if pbr_bake else None)
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
            elif name in ("water", "water_apron"):
                # The sheet's own colour (the server takes it from the photographed
                # water, TerrainMeshStage._colour_water_from_photo), as Unity does.
                colour = [0.10, 0.30, 0.45, 0.90]
                factor = getattr(getattr(tm.visual, "material", None), "baseColorFactor", None)
                if factor is not None:
                    f = np.asarray(factor, dtype=np.float64).ravel()
                    colour = list(f / 255.0 if f.max() > 1.0 else f)
                    colour[3] = 0.90
                photo = None if os.environ.get("RENDER_WATER_FLAT") == "1" else self._water_photo(key, tm, colour, pose if name == "water" else None)
                if photo is not None:
                    tm, mat = photo
                    flowing = self._water_flow_uv(tm, pose) if getattr(self.args, "animate", 0) and name == "water" else None
                    if flowing is None and getattr(self.args, "animate", 0) and name == "water":
                        # Still water (a lake, or no flow field): the wind's ripples alone.
                        flowing = np.zeros((len(tm.vertices), 2))
                    if flowing is not None:
                        g = to_gl(tm)
                        sea = ((self.scene.get("motion") or {}).get("water") or {}).get("kind") == "sea"
                        mat.emissiveFactor = [1.0, 1.0, 1.0009 if sea else 1.0004]
                        shore = self._shore_distance(tm, pose)
                        cur = self._water_current(tm, pose)
                        node = ps.add(pyrender.Mesh(primitives=[pyrender.Primitive(
                            positions=np.asarray(g.vertices, dtype=np.float32),
                            normals=np.asarray(g.vertex_normals, dtype=np.float32),
                            texcoord_0=np.asarray(tm.visual.uv, dtype=np.float32),
                            texcoord_1=flowing.astype(np.float32),
                            color_0=np.stack([shore / 400.0, -cur[:, 0] / 4.0 + 0.5, cur[:, 1] / 4.0 + 0.5, np.ones_like(shore)], axis=-1).astype(np.float32),
                            indices=np.asarray(g.faces, dtype=np.uint32), material=mat)]),
                            pose=_MIRROR @ pose @ _MIRROR)
                        self.wire_nodes[node] = WIRE_COLOURS["water"]
                        self.anim_tags[node] = "water"
                        stats["env"] += 1
                        continue
                else:
                    mat = material_for(tm, blend=True, color=colour, roughness=0.1)
            elif name in ("treeline", "backdrop"):
                # Lit unless the scene object says it carries its photographed
                # light (SceneObject.unlit); RENDER_UNLIT_ENV=1 for older scenes.
                mat = material_for(tm, unlit=bool(o.get("unlit")) or os.environ.get("RENDER_UNLIT_ENV") == "1")
            else:
                mat = material_for(tm)
            node = ps.add(pyrender.Mesh.from_trimesh(to_gl(tm), material=mat, poses=np.array([_MIRROR @ pose @ _MIRROR]), smooth=False))
            if name in ("water", "water_apron"):
                self.anim_tags[node] = "water"
            wire_kind = ("terrain" if name == "terrain" else "water" if name in ("water", "water_apron")
                         else "backdrop" if name in ("treeline", "backdrop") else "formation")
            self.wire_nodes[node] = WIRE_COLOURS[wire_kind]
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
        if terrain_pose is not None and not self.args.no_grass:
            grass = self._grass_field_mesh()
            if grass is not None:
                g = to_gl(grass)
                blade = grass.metadata["blade"]
                gveg = ((self.scene.get("motion") or {}).get("vegetation") or {}).get("grass_field") if getattr(self.args, "animate", 0) else None
                gm = pyrender.Mesh(primitives=[pyrender.Primitive(
                    positions=np.asarray(g.vertices, dtype=np.float32),
                    normals=np.asarray(g.vertex_normals, dtype=np.float32),
                    color_0=grass.metadata["colour"],
                    # For the sway: the blade's root (the renderer's frame mirrors x) and heights.
                    texcoord_0=np.stack([-blade[:, 0], blade[:, 1]], axis=-1) if gveg else None,
                    texcoord_1=blade[:, 2:4] if gveg else None,
                    indices=np.asarray(g.faces, dtype=np.uint32),
                    material=pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=0.0, roughnessFactor=0.9, doubleSided=True),
                )])
                gm.no_shadow = True
                if gveg:
                    gp = gm.primitives[0]
                    gp._sway = (math.radians(float(gveg.get("amplitudeDeg", 12.0))), float(gveg.get("frequencyHz", 1.4)), 0.3,
                                float((self.scene.get("motion") or {}).get("wind", {}).get("gustiness", 0.4)))
                    gp._sway_mode = 2.0
                gnode = ps.add(gm, pose=_MIRROR @ terrain_pose @ _MIRROR)
                if gveg:
                    self.anim_tags[gnode] = "veg"
                stats["grass_tufts"] = int(grass.metadata["tufts"])
        self._add_sun(ps)
        return ps, stats

    def _grass_field_mesh(self):
        """The scene's grass field (server/pipeline/grass_cover/grass_field.py) grown
        the way GrassField.cs grows it -- jittered lattice, kept by density and
        thinned past near_radius -- as one merged mesh of blade tufts in the terrain's
        own frame, heights from the height map. Cached across views."""
        if hasattr(self, "_grass_cache"):
            return self._grass_cache
        self._grass_cache = None
        field = self.scene.get("grassField") or {}
        if not field.get("density"):
            return None
        from pipeline.pipeline_context import ContextKey
        dens, col = self.image_for(field["density"]), self.image_for(field["colour"])
        hm_d = self.context.input_depth(ContextKey.HEIGHT_MAP)
        params = self.context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}
        if dens is None or col is None or hm_d is None:
            return None
        hm = hm_d.depth
        grid = float(params.get("grid_size_meters", 200.0)); cell = grid / hm.shape[0]
        d = np.asarray(dens.convert("RGB"), dtype=np.float32) / 255.0
        c = np.asarray(col.convert("RGB"), dtype=np.float32) / 255.0
        size = float(field["sizeMeters"]); half = size / 2.0
        spacing = 1.0 / math.sqrt(max(float(field["tuftsPerSquareMeter"]), 0.01))
        n = int(math.ceil(size / spacing))
        rng = np.random.default_rng(12345)
        ix, iz = np.meshgrid(np.arange(n), np.arange(n))
        x = -half + (ix + rng.random(ix.shape)) * spacing
        z = -half + (iz + rng.random(iz.shape)) * spacing
        x, z = x.ravel(), z.ravel()
        r = np.hypot(x, z)
        # Image row 0 is z = -half (grass_field.py writes rows in grid order).
        pr = np.clip(((z + half) / size * d.shape[0]).astype(int), 0, d.shape[0] - 1)
        pc = np.clip(((x + half) / size * d.shape[1]).astype(int), 0, d.shape[1] - 1)
        keep = d[pr, pc, 0] * np.where(r > 10.0, 10.0 / np.maximum(r, 1e-6), 1.0)
        sel = (r <= half) & (rng.random(len(x)) < keep)
        x, z, pr, pc = x[sel], z[sel], pr[sel], pc[sel]
        cap = int(os.environ.get("RENDER_GRASS_MAX", "25000"))
        if len(x) > cap:
            idx = rng.choice(len(x), cap, replace=False); x, z, pr, pc = x[idx], z[idx], pr[idx], pc[idx]
        hr = np.clip(((z + grid / 2) / cell).astype(int), 0, hm.shape[0] - 1)
        hc = np.clip(((x + grid / 2) / cell).astype(int), 0, hm.shape[1] - 1)
        y = hm[hr, hc]
        heights = float(field["maxHeightMeters"]) * d[pr, pc, 1] * (0.8 + 0.4 * rng.random(len(x)))
        blades = int(field.get("bladesPerTuft", 7)); width = float(field.get("bladeWidthMeters", 0.012))
        V, F, C, R = [], [], [], []      # R: each vertex's blade root (x, z), height above it, blade height
        seg = 4
        for k in range(len(x)):
            h = max(float(heights[k]), 0.04)
            base_col = c[pr[k], pc[k]] * (0.85 + 0.3 * rng.random())
            for _ in range(blades if np.hypot(x[k], z[k]) <= 10.0 else min(3, blades)):
                a = rng.random() * 2 * math.pi
                face = np.array([math.cos(a), 0.0, math.sin(a)]); side = np.array([-face[2], 0.0, face[0]])
                root = np.array([x[k], y[k], z[k]]) + face * rng.random() * 0.08 * h
                lean = (0.1 + 0.35 * rng.random()) * h; bh = (0.6 + 0.4 * rng.random()) * h
                w = width * h / 0.4 * (0.7 + 0.6 * rng.random())
                b0 = len(V)
                for sgi in range(seg + 1):
                    t = sgi / seg
                    ctr = root + np.array([0.0, t * bh, 0.0]) + face * lean * t * t
                    hw = 0.5 * w * (1 - 0.85 * t)
                    V.append(ctr - side * hw); V.append(ctr + side * hw)
                    R.append((root[0], root[2], t * bh, bh)); R.append((root[0], root[2], t * bh, bh))
                    col_t = np.clip(base_col * (0.6 + 0.7 * t), 0, 1)  # GrassBlades.shader root 0.6, tip 1.3
                    C.append(col_t); C.append(col_t)
                for sgi in range(seg):
                    i0 = b0 + 2 * sgi
                    F.append((i0, i0 + 2, i0 + 1)); F.append((i0 + 1, i0 + 2, i0 + 3))
        if not V:
            return None
        tm = trimesh.Trimesh(vertices=np.asarray(V), faces=np.asarray(F), process=False)
        up = np.tile([0.0, 1.0, 0.0], (len(V), 1))
        tm.vertex_normals = up
        col4 = np.concatenate([np.asarray(C, dtype=np.float32) ** 2.2, np.ones((len(C), 1), np.float32)], axis=1)
        tm.metadata["colour"] = col4
        tm.metadata["blade"] = np.asarray(R, dtype=np.float32)
        tm.metadata["tufts"] = len(x)
        self.log(f"  grass: {len(x)} tufts from the grass field")
        self._grass_cache = tm
        return tm

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
        self._terrain_pbr = None
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
            self._terrain_pbr = bake_terrain_pbr(splat, hm.depth.astype(np.float32), half, int(self.args.terrain_texture))
            self._pano_pass = self._panorama_pass_mesh(splat, terrain, half)
            self._plate_passes = self._plate_pass_meshes(splat, terrain, half)
            self._wall_style(splat, terrain)
        else:
            tex = bake_terrain_texture(splat, hm.depth.astype(np.float32), half, int(self.args.terrain_texture))
        self.log(f"  terrain: baked {len(splat.layers)} layer(s) to {tex.size[0]} px in {time.time() - t:.1f} s"
                 f" (panorama layer {'drawn through the mesh' + chr(39) + 's baked UV in a second pass' if two_pass else 'from the analytic projection'})")
        v = np.asarray(tm.vertices)
        u = (v[:, 0] + half) / (2.0 * half)
        w = (v[:, 2] + half) / (2.0 * half)  # 0 at z = -half == image row 0 (top)
        uv = np.stack([u, 1.0 - w], axis=-1)  # GL: v = 1 is the image's top row
        return tex, uv

    def _wall_style(self, splat, terrain) -> None:
        """The steep faces' albedo and dressing: the median colour the photograph
        gives the steep faces it paints (the panorama layer through TEXCOORD_1, at
        a weight of 0.6 or more: Paris's photographed quay walls), else the macro
        map's mean; coursed stone when the caption names a city."""
        from pipeline.pipeline_context import ContextKey
        self._wall_tint = None
        v = np.asarray(terrain.mesh.vertices)
        f = np.asarray(terrain.mesh.faces)
        steep = steep_faces(v, f)
        if not steep.any():
            return
        tint = None
        pano_i = next((i for i, l in enumerate(splat.layers) if l.equirect), None)
        w = getattr(self, "_pano_vertex_weight", None)
        if pano_i is not None and w is not None and terrain.extra_uv is not None:
            ids = np.unique(f[steep])
            ids = ids[w[ids] >= 0.6]
            # Above the waterline's reach: the photograph's foot of a quay wall is the
            # water's reflection and the wet band, not the stone.
            wl, tpose = self._water_level(), self._env_pose("terrain")
            if wl is not None and tpose is not None:
                ids = ids[(v[ids] @ tpose[:3, :3].T + tpose[:3, 3])[:, 1] > wl + 1.0]
            if len(ids) >= 30:
                tile = np.asarray(splat.layers[pano_i].tile.convert("RGB"), dtype=np.float32) / 255.0
                uv = np.asarray(terrain.extra_uv, dtype=np.float64)[ids]
                h, wd = tile.shape[:2]
                rows = ((1.0 - np.mod(uv[:, 1], 1.0)) * (h - 1)).round().astype(int)
                cols = (np.mod(uv[:, 0], 1.0) * (wd - 1)).round().astype(int)
                tint = np.median(tile[rows, cols], axis=0)
                src = f"the photographed steep faces ({len(ids)} vertices)"
        if tint is None:
            tint = np.asarray(getattr(splat, "macro_mean", None) or (0.55, 0.53, 0.48), dtype=np.float64)
            src = "the macro map's mean"
        caption = self.context.input_object(ContextKey.INPUT_CAPTION)
        words = set(re.findall(r"[a-z]+", str(caption or "").lower()))
        self._wall_tint = np.asarray(tint, dtype=np.float64)
        self._wall_ashlar = bool(words & URBAN_WORDS)
        self.log(f"  terrain: {int(steep.sum())} steep faces drawn as a side-projected "
                 f"{'coursed stone' if self._wall_ashlar else 'rock'} wall, albedo "
                 f"({tint[0]:.2f}, {tint[1]:.2f}, {tint[2]:.2f}) from {src}")

    def _env_pose(self, name: str) -> np.ndarray | None:
        o = next((o for o in self.scene["objects"] if str(o.get("name")) == name and o.get("mesh")), None)
        return None if o is None else transform(vec(o.get("position")), vec(o.get("rotation")), vec(o.get("scale"), (1, 1, 1)))

    def _water_level(self) -> float | None:
        o = next((o for o in self.scene["objects"] if str(o.get("name")) == "water" and o.get("mesh")), None)
        tm = self.trimesh_for(o["mesh"]) if o is not None else None
        if tm is None or not len(tm.vertices):
            return None
        pose = self._env_pose("water")
        return float(np.median((np.asarray(tm.vertices) @ pose[:3, :3].T + pose[:3, 3])[:, 1]))

    def _wall_mesh(self, tm: trimesh.Trimesh, steep: np.ndarray, pose: np.ndarray):
        """The steep faces on their own vertices, box-mapped: u along the wall's run
        (world X or Z, whichever the wall's smoothed facing is closer to across),
        v = height above the waterline over WALL_TILE_M[1]."""
        from scipy.spatial import cKDTree
        tri = np.asarray(tm.vertices, dtype=np.float64)[np.asarray(tm.faces)[steep]]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        centres = tri.mean(axis=1)
        # Facing smoothed over 3 m of wall, so the mapping axis changes along a line
        # where the wall turns, not face by face where it runs at 45 degrees.
        nb = cKDTree(centres[:, [0, 2]]).query_ball_point(centres[:, [0, 2]], r=3.0)
        sm = np.stack([np.abs(n[idx][:, [0, 2]]).mean(axis=0) for idx in nb])
        along_x = sm[:, 1] >= sm[:, 0]
        world = tri.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
        wl = self._water_level()
        base = wl if wl is not None else float(world[:, 1].min())
        horiz = np.where(np.repeat(along_x, 3), world[:, 0], world[:, 2])
        uv = np.stack([horiz / WALL_TILE_M[0], (world[:, 1] - base) / WALL_TILE_M[1]], axis=-1)
        wall = trimesh.Trimesh(vertices=tri.reshape(-1, 3), faces=np.arange(len(tri) * 3).reshape(-1, 3), process=False)
        wall.visual = trimesh.visual.TextureVisuals(uv=uv)
        tex = wall_texture(self._wall_tint, ashlar=self._wall_ashlar, wet=wl is not None)
        return wall, material_for(wall, image=tex, double_sided=False, roughness=0.9)

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
            keep = keep[(w[keep] > 1e-3).any(axis=1) & ~steep_faces(v, keep)]
            if len(keep) == 0:
                continue
            tf = float(layer.tile_factor)
            uv = np.stack([u * tf, -z * tf], axis=-1)  # GL v = 1 is the image's top row
            tm = trimesh.Trimesh(vertices=v * 0.999, faces=keep, process=False)
            tm.visual = trimesh.visual.TextureVisuals(uv=uv)
            tm.metadata["pano_weight"] = np.stack([np.ones_like(w), np.ones_like(w), np.ones_like(w), w], axis=-1).astype(np.float32)
            pbr = getattr(layer, "pbr", None)
            normal_img = occ_img = None
            if pbr is not None:
                a = np.asarray(pbr.convert("RGBA"), dtype=np.float32) / 255.0
                nxy = a[..., :2] * 2.0 - 1.0
                nz = np.sqrt(np.clip(1.0 - (nxy ** 2).sum(-1), 0.0, 1.0))
                normal_img = PILImage.fromarray(((np.concatenate([nxy, nz[..., None]], -1) * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")
                occ_img = PILImage.fromarray((np.repeat(a[..., 2:3], 3, -1) * 255).astype(np.uint8), "RGB")
            mat = material_for(tm, image=layer.tile.convert("RGB"), double_sided=False, roughness=0.95, blend=True,
                               normal_image=normal_img, occlusion_image=occ_img)
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
        self._pano_vertex_weight = w
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
        tint = np.asarray(getattr(self.args, "ambient_tint", None) or [1.0, 1.0, 1.0], dtype=np.float64)
        return (sky * tint * float(self.args.ambient)).tolist()

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
        if self.args.sun_az_el:
            # Debug override: aim the key light at az,el (deg) in the scene frame.
            az, el = (math.radians(float(v)) for v in self.args.sun_az_el.split(","))
            d = np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
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


# ── shadows: small vegetation casts none ─────────────────────────────────────────
# pyrender's directional shadow map spans the whole scene (the grid and the shell
# to 500 m), so a texel is half a metre: a flower cannot cast a shadow at that
# resolution, only a blot, and a meadow of cards under a 20 deg sun was its own
# solid shadow (Rainier band21: the sun pass showed every plant black while the
# ground around it was lit). Unity does not shadow small vegetation either. A
# mesh flagged `no_shadow` is hidden during the shadow pass and drawn normally.
def _patch_shadow_pass() -> None:
    from pyrender.renderer import Renderer
    if getattr(Renderer, "_no_shadow_patched", False):
        return
    original = Renderer._shadow_mapping_pass

    def _shadow_mapping_pass(self, scene, light_node, flags):
        hidden = []
        for node in scene.mesh_nodes:
            if getattr(node.mesh, "no_shadow", False) and node.mesh.is_visible:
                node.mesh.is_visible = False
                hidden.append(node.mesh)
        try:
            return original(self, scene, light_node, flags)
        finally:
            for mesh in hidden:
                mesh.is_visible = True

    Renderer._shadow_mapping_pass = _shadow_mapping_pass
    Renderer._no_shadow_patched = True


_NO_SHADOW_CLASSES = ("flower", "plant", "grass", "bush", "shrub", "reed", "moss")


def casts_no_shadow(key: str) -> bool:
    """Small vegetation by its asset key (category_mesh_flower_3_v1, ..._card)."""
    m = re.match(r"category_mesh_([a-z_]+?)_\d+", key)
    return bool(m and m.group(1) in _NO_SHADOW_CLASSES)


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
    if view.startswith("orbit:"):
        # A frame of the slow pan: "orbit:<yaw degrees>", the capture camera turned
        # about its own up axis.
        yaw = float(view.split(":", 1)[1])
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


def expected_water_velocity(builder, depth: np.ndarray, water: np.ndarray, Rv: np.ndarray, pos: np.ndarray,
                            fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """--anim-masks: the screen velocity (px/s, [H, W, 2]) the water's flow field gives each
    water pixel: the pixel's world point (from the depth) moved one second along the field,
    projected back. Zero off the water or without a field."""
    H, W = depth.shape
    out = np.zeros((H, W, 2), dtype=np.float32)
    ff = getattr(builder, "_flow_field", None)
    if ff is None or not water.any():
        return out
    i, j = np.nonzero(water)
    z = depth[i, j].astype(np.float64)
    cam = np.stack([(j + 0.5 - cx) / fx * z, -(i + 0.5 - cy) / fy * z, z], axis=-1)   # right, up, forward
    world = pos[None, :] + cam @ Rv.T                                                   # server frame
    vel = ff.sample(world[:, 0], world[:, 2])
    moved = world.copy()
    moved[:, 0] += vel[:, 0]
    moved[:, 2] += vel[:, 1]
    c2 = (moved - pos[None, :]) @ Rv
    ok = c2[:, 2] > 0.1
    u2 = np.where(ok, c2[:, 0] / np.maximum(c2[:, 2], 0.1) * fx + cx, j + 0.5)
    v2 = np.where(ok, -c2[:, 1] / np.maximum(c2[:, 2], 0.1) * fy + cy, i + 0.5)
    out[i, j, 0] = u2 - (j + 0.5)
    out[i, j, 1] = v2 - (i + 0.5)
    return out


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: scratch/render/<run-id>)")
    ap.add_argument("--scene", type=Path, default=None,
                    help="render this scene.json instead of the newest one in run_dir (the exposure fit's lit copy)")
    ap.add_argument("--config", type=Path, default=ROOT / "server" / "config.yaml")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--views", default="photo,left,right,step", help="comma list of photo,left,right,back,step,high,down")
    ap.add_argument("--terrain-texture", type=int, default=2048, help="baked terrain texture size (px)")
    ap.add_argument("--no-grass", action="store_true")
    ap.add_argument("--orbit", type=int, default=0, help="render N frames of a full turn about the capture camera and encode orbit.mp4")
    ap.add_argument("--fps", type=int, default=24, help="frame rate of orbit.mp4 / animate.mp4")
    ap.add_argument("--animate", type=float, default=0.0,
                    help="render this many seconds of the scene's motion (scene.json 'motion': plants in the wind, "
                         "boats on the water, the water's flow, the sky's drift) from the capture camera -> animate.mp4")
    ap.add_argument("--anim-masks", action="store_true",
                    help="--animate: also write anim_masks.npz (per-frame labels: 0 sky, 1 water, 2 vegetation, "
                         "3 other, 10+k mover k; and the water's expected screen velocity) for scratch/r4/anim/metrics.py")
    ap.add_argument("--resume", action="store_true", help="--orbit: keep the frames already in --out and render the rest")
    ap.add_argument("--flat-terrain", action="store_true", help="debug: terrain as flat grey facets, no texture")
    ap.add_argument("--no-pbr-maps", action="store_true", help="ignore normal/occlusion/metallic-roughness textures on assets")
    ap.add_argument("--material", type=Path, default=None, help="use this saved splat_material directory instead of the run's")
    ap.add_argument("--only", default="", help="render only placed objects whose name contains this (environment meshes always render)")
    ap.add_argument("--exclude", default="", help="skip placed objects whose name contains this")
    ap.add_argument("--env-exclude", default=None, help="skip environment meshes (terrain/water/treeline/backdrop/formation_N) whose name contains this")
    ap.add_argument("--sun-scale", type=float, default=2.0, help="directional light intensity per unit of the scene's sun intensity")
    ap.add_argument("--ambient", type=float, default=0.45, help="ambient as a fraction of the sky's mean colour")
    ap.add_argument("--exposure", type=float, default=1.0)
    ap.add_argument("--light-passes", action="store_true",
                    help="for the photo view also write photo_pass_{sun,amb,unlit}.npy: the frame lit by the sun alone, "
                         "the ambient alone and by nothing (what Exposure Fit fits the light scales on)")
    ap.add_argument("--no-fit", action="store_true", help="ignore the run's Exposure Fit (render at the unfitted light scales)")
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--wireframe", action="store_true",
                    help="also write each frame with the triangle edges over it (<view>_wire.png; --orbit: wire_NNNN.png and orbit_wire.mp4)")
    ap.add_argument("--sun-az-el", default=None, help="debug: aim the sun at 'az,el' degrees instead of the scene's direction")
    ap.add_argument("--save-depth", action="store_true", help="also write <view>_depth.npy (metres, 0 where only sky)")
    ap.add_argument("--pano-uv", choices=["baked", "analytic"], default="baked",
                    help="terrain panorama layer through the mesh's baked TEXCOORD_1 (what Unity samples) or the analytic projection")
    args = ap.parse_args()
    global NO_PBR_MAPS
    NO_PBR_MAPS = bool(args.no_pbr_maps)

    run_dir = args.run_dir.resolve()
    # Not inside the run directory: the context loader treats its subdirectories
    # as stage caches.
    out = (args.out or (ROOT / "scratch" / "render" / run_dir.name)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731

    t0 = time.time()
    context = load_context(run_dir, args.config)
    from pipeline.pipeline_context import ContextKey
    # The newest scene.json any stage wrote: Exposure Fit's carries the fitted
    # light and the sun's colour, Scene Generation's the objects alone.
    candidates = sorted(run_dir.glob("*/scene.json"), key=lambda p: p.stat().st_mtime)
    scene_path = args.scene or (candidates[-1] if candidates else run_dir / "Scene Generation" / "scene.json")
    scene = json.loads(scene_path.read_text())
    log(f"scene: {len(scene['objects'])} objects from {scene_path.parent.name} (loaded in {time.time() - t0:.1f} s)")
    # The fitted light scales (pipeline/lighting/exposure_fit.py). A scene.json
    # written after the fit carries them in lighting.fit with the sun's intensity
    # already scaled; Scene Generation's (read above) does not, so the fit file
    # scales both here.
    fit = (scene.get("lighting") or {}).get("fit")
    fit_path = run_dir / "Exposure Fit" / "exposure_fit.json"
    args.ambient_tint, args.object_gain = None, 1.0
    args.object_gains = {}
    if not args.no_fit:
        src = fit if fit else (json.loads(fit_path.read_text()) if fit_path.exists() else {})
        tint = src.get("ambient_tint")
        if isinstance(tint, (list, tuple)) and len(tint) == 3:
            args.ambient_tint = [float(v) for v in tint]
        args.object_gain = float(src.get("object_gain") or 1.0)
        gains = src.get("object_gains")
        args.object_gains = {str(k): [float(x) for x in v] for k, v in gains.items()} if isinstance(gains, dict) else {}
        if args.object_gains:
            log(f"exposure fit: {len(args.object_gains)} model(s) with a colour gain of their own")
        if args.ambient_tint or args.object_gain != 1.0:
            log(f"exposure fit: ambient tint {args.ambient_tint}, object albedo x{args.object_gain:.2f}")
        if fit:
            args.ambient *= float(fit.get("ambient_scale") or 1.0)
            log(f"exposure fit (scene): ambient x{float(fit.get('ambient_scale') or 1.0):.2f}, sun already x{float(fit.get('sun_scale') or 1.0):.2f}")
        elif fit_path.exists():
            fit = json.loads(fit_path.read_text())
            args.sun_scale *= float(fit.get("sun_scale") or 1.0)
            args.ambient *= float(fit.get("ambient_scale") or 1.0)
            log(f"exposure fit (file): sun x{float(fit.get('sun_scale') or 1.0):.2f}, ambient x{float(fit.get('ambient_scale') or 1.0):.2f}")

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

    _patch_shadow_pass()
    renderer = pyrender.OffscreenRenderer(width, height)
    renderer._renderer._program_cache = ShaderProgramCache(shader_dir=cutout_shader_dir())
    flags = pyrender.RenderFlags.NONE if args.no_shadows else pyrender.RenderFlags.SHADOWS_DIRECTIONAL

    builder = SceneBuilder(context, scene, args, log)
    results = {}
    if args.animate > 0:
        _patch_anim_uniforms()
        motion = scene.get("motion") or {}
        wind = motion.get("wind") or {}
        wd = math.radians(float(wind.get("directionDeg", 0.0)))
        # GL mirrors X: the wind's server-frame (sin, cos) becomes (-sin, cos).
        ANIM["wind"] = (-math.sin(wd), math.cos(wd), max(float(wind.get("speedMs", 2.0)), 0.5), 25.0)
        wm = motion.get("water") or {}
        ANIM["surf"] = (float(wm.get("waveSpeedMs") or 3.0), float(wm.get("swellPeriodS") or 10.0))
        flow_path = None
        for cand in ([scene_path.parent / "water_flow.npz"] if scene_path is not None else []) + [run_dir / "Scene Motion" / "water_flow.npz"]:
            if cand.is_file():
                flow_path = cand
                break
        if flow_path is not None:
            sys.path.insert(0, str(ROOT / "server"))
            from pipeline.scene_motion.flow_field import FlowField
            builder._flow_field = FlowField.load(flow_path)
            log(f"animate: water flow from {flow_path}")
        sky_m = motion.get("sky") or {}
        right = R[:, 0][[0, 2]]
        sd = math.radians(float(sky_m.get("directionDeg", 0.0)))
        sky_rate = float(sky_m.get("angularSpeedDegS", 0.0)) * float(np.sign(np.dot([math.sin(sd), math.cos(sd)], right / max(np.linalg.norm(right), 1e-9))))
        frames = int(round(args.animate * args.fps))
        Rv, pos = view_pose(R, t, "photo")
        zfar = float(scene.get("farClipPlane") or 300.0) * 1.5
        ps, stats = builder.build(pos, vfov)
        ps.add(pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=zfar), pose=gl_camera_pose(Rv, pos))
        log(f"animate: {frames} frames, {len(builder.mover_nodes)} moving object(s), sky {sky_rate:+.3f} deg/s; {stats}")
        mask_frames = []
        if args.anim_masks:
            seg_map = {n: (3, 0, 0) for n in ps.mesh_nodes}
            for n, tag in builder.anim_tags.items():
                seg_map[n] = (1, 0, 0) if tag == "water" else (2, 0, 0)
            for k, (n, *_rest) in enumerate(builder.mover_nodes):
                seg_map[n] = (10 + k, 0, 0)
        for i in range(frames):
            tt = i / float(args.fps)
            ANIM["time"] = tt
            builder.pose_movers(ps, tt)
            colour, depth = renderer.render(ps, flags=flags)
            colour = colour[..., :3].astype(np.float32)
            if args.exposure != 1.0:
                colour = (colour / 255.0 * args.exposure).clip(0, 1) * 255.0
            if sky is not None:
                # As for the still views: the sky under each edge pixel's uncovered part.
                open_sky = depth <= 0.0
                coverage = np.ones(depth.shape + (1,), dtype=np.float32)
                if open_sky.any():
                    from scipy.ndimage import binary_dilation
                    edge = binary_dilation(open_sky, iterations=2) & ~open_sky
                    if edge.any():
                        white = render_coverage(renderer, ps)
                        coverage[edge] = np.asarray(white)[..., :1][edge].astype(np.float32) / 255.0
                bg = render_sky(sky.image, yaw + sky_rate * tt, Rv, width, height, fx, fy, cx, cy).astype(np.float32)
                colour = np.where(open_sky[..., None], bg, colour + (1.0 - coverage) * bg)
            colour = colour.clip(0, 255).astype(np.uint8)
            if args.anim_masks:
                seg, _ = renderer.render(ps, flags=pyrender.RenderFlags.SEG, seg_node_map=seg_map)
                lab = np.asarray(seg)[..., 0].astype(np.uint8)
                lab[depth <= 0.0] = 0
                mask_frames.append(lab)
                if i == 0:
                    water_vel = expected_water_velocity(builder, depth, lab == 1, Rv, pos, fx, fy, cx, cy)
            PILImage.fromarray(colour, "RGB").save(out / f"anim_{i:04d}.png")
            if i < 2 or i % 48 == 0:
                log(f"anim_{i:04d}: t={tt:.2f} s")
        sys.path.insert(0, str(ROOT / "scripts"))
        from frames_to_mp4 import encode as encode_mp4
        mp4 = encode_mp4(sorted(out.glob("anim_*.png")), out / "animate.mp4", args.fps)
        if args.anim_masks:
            np.savez_compressed(out / "anim_masks.npz", labels=np.stack(mask_frames), water_vel=water_vel, fps=args.fps,
                                movers=np.array(len(builder.mover_nodes)))
        log(f"animate: {frames} frames at {args.fps} fps -> {mp4}")
        renderer.delete()
        log(f"done in {time.time() - t0:.1f} s")
        return 0
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    if args.orbit > 0:
        views = [f"orbit:{(360.0 * i / args.orbit):.4f}" for i in range(args.orbit)]
    orbit_ps = orbit_cam = orbit_stats = None
    for vi, view in enumerate(views):
        fname = f"orbit_{vi:04d}" if view.startswith("orbit:") else view
        Rv, pos = view_pose(R, t, view)
        t1 = time.time()
        zfar = float(scene.get("farClipPlane") or 300.0) * 1.5
        if view.startswith("orbit:") and args.resume and (out / f"{fname}.png").is_file():
            continue
        if args.light_passes and view == "photo":
            # The frame is linear in the lights: U (nothing lit: the emissive
            # photo content), S (sun only), A (ambient only) let a fit solve
            # clip(U + s (S - U) + a (A - U)) against the photograph.
            base_sun, base_amb = args.sun_scale, args.ambient
            for tag, sun_scale, ambient in (("sun", base_sun, 0.0), ("amb", 0.0, base_amb), ("unlit", 0.0, 0.0)):
                args.sun_scale, args.ambient = sun_scale, ambient
                ps_pass, _ = builder.build(pos, vfov)
                ps_pass.add(pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=zfar), pose=gl_camera_pose(Rv, pos))
                c_pass, _ = renderer.render(ps_pass, flags=flags)
                np.save(out / f"photo_pass_{tag}.npy", (c_pass[..., :3].astype(np.float32) / 255.0).astype(np.float16))
            args.sun_scale, args.ambient = base_sun, base_amb
            log(f"photo: light passes written (sun, ambient, unlit)")
            # Which placed model each pixel shows, for the fit of each model's own
            # colour: the flat-colour pass, one colour per model, everything else
            # black (drawn, so that it hides what stands behind it).
            try:
                keys = sorted({kk for kk in builder.key_nodes.values()})
                index = {kk: i + 1 for i, kk in enumerate(keys)}
                colours = {node: (0, 0, 0) for node in ps_pass.mesh_nodes}
                for node, kk in builder.key_nodes.items():
                    if node in colours:
                        i = index[kk]
                        colours[node] = (i & 255, (i >> 8) & 255, 255)
                seg, _ = renderer.render(ps_pass, flags=pyrender.RenderFlags.SEG, seg_node_map=colours)
                seg = np.asarray(seg)[..., :3].astype(np.int64)
                ids = np.where(seg[..., 2] == 255, seg[..., 0] + (seg[..., 1] << 8), 0)
                ids[ids > len(keys)] = 0
                np.save(out / "photo_object_ids.npy", ids.astype(np.uint16))
                (out / "photo_object_ids.json").write_text(json.dumps([list(kk) for kk in keys]))
                log(f"photo: object-id pass written ({len(keys)} models, {int((ids > 0).sum())} px)")
            except Exception as exc:  # the fit goes on without it
                log(f"photo: no object-id pass ({type(exc).__name__}: {exc})")
        if view.startswith("orbit:") and orbit_ps is not None:
            # The turn keeps the camera where it is: one scene, built and uploaded once, and
            # only the camera's pose changes (a rebuild per frame cost 8-12 s of the frame).
            ps, stats = orbit_ps, orbit_stats
            ps.set_pose(orbit_cam, gl_camera_pose(Rv, pos))
        else:
            ps, stats = builder.build(pos, vfov)
            cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=zfar)
            cam_node = ps.add(cam, pose=gl_camera_pose(Rv, pos))
            if view.startswith("orbit:"):
                orbit_ps, orbit_cam, orbit_stats = ps, cam_node, stats
        colour, depth = renderer.render(ps, flags=flags)
        # Multisampling resolves an edge pixel as the geometry's colour mixed with the
        # clear colour (black) by its coverage: composited on depth alone, every
        # silhouette against the sky carried a dark rim, and the water apron's far
        # edge drew a black line along the whole horizon (Shark Fin). The sky goes in
        # under the uncovered part of each pixel next to it instead, the coverage
        # read from a flat white pass (the colour pass's alpha is a cutout texture's
        # own on cards and the backdrop, and a blend weight on the overlays).
        open_sky = depth <= 0.0
        coverage = np.ones(depth.shape + (1,), dtype=np.float32)
        if sky is not None and open_sky.any():
            from scipy.ndimage import binary_dilation
            edge = binary_dilation(open_sky, iterations=2) & ~open_sky
            if edge.any():
                white = render_coverage(renderer, ps)
                coverage[edge] = np.asarray(white)[..., :1][edge].astype(np.float32) / 255.0
        colour = colour[..., :3].astype(np.float32)
        if args.exposure != 1.0:
            colour = (colour / 255.0 * args.exposure).clip(0, 1) * 255.0
        if sky is not None:
            bg = render_sky(sky.image, yaw, Rv, width, height, fx, fy, cx, cy).astype(np.float32)
            colour = np.where(open_sky[..., None], bg, colour + (1.0 - coverage) * bg)
        colour = colour.clip(0, 255).astype(np.uint8)
        img = PILImage.fromarray(colour, "RGB")
        img.save(out / f"{fname}.png")
        if args.wireframe:
            seg, wire_depth = render_wire(renderer, ps, builder.wire_nodes)
            wname = f"wire_{vi:04d}" if view.startswith("orbit:") else f"{fname}_wire"
            wire_overlay(colour, seg, wire_depth, depth).save(out / f"{wname}.png")
        if args.save_depth:
            np.save(out / f"{fname}_depth.npy", depth.astype(np.float16))
        if not view.startswith("orbit:"):
            results[view] = img
        if not view.startswith("orbit:") or vi < 3 or vi % 20 == 0:
            log(f"{fname}: {stats} in {time.time() - t1:.1f} s -> {out / f'{fname}.png'}")
    if args.orbit > 0:
        sys.path.insert(0, str(ROOT / "scripts"))
        from frames_to_mp4 import encode as encode_mp4
        frames = sorted(out.glob("orbit_*.png"))
        mp4 = encode_mp4(frames, out / "orbit.mp4", args.fps)
        log(f"orbit: {len(frames)} frames at {args.fps} fps -> {mp4}")
        if args.wireframe:
            frames = sorted(out.glob("wire_*.png"))
            mp4 = encode_mp4(frames, out / "orbit_wire.mp4", args.fps)
            log(f"orbit, wireframe: {len(frames)} frames at {args.fps} fps -> {mp4}")

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
