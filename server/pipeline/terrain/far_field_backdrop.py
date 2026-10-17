"""
Far-field backdrop: the part of the scene the terrain grid cannot hold.

The terrain grid is 200 m across and every metric in the near-field chain is
tuned to it, so Panorama Depth Calibration compresses everything beyond 60 m
into the 60-100 m band. That is the right thing for the ground the viewer walks
on and the wrong thing for a mountain that is kilometres away: it was pulled into
the band and modelled as a ring of rock around the meadow, at a hundredth of its
distance and with the parallax to match.

Two domains instead. The grid keeps everything whose metric depth is inside it.
Everything beyond -- the pixels of the (object-removed) panorama whose
uncompressed calibrated depth (PANORAMA_DEPTH_METRIC) is past the grid -- is
stood up here as a relief shell: an equirectangular vertex grid, each vertex
placed along its own ray at a radius that keeps the far field's ORDER but maps
its distance asymptotically into [start_m, far_m],

    r(d) = start + (far - start) * (1 - exp(-(d - start) / scale))

so a ridge at 300 m stands behind one at 150 m and a summit at 3 km still sits
inside the far clip. Faces are dropped across depth discontinuities so the shell
tears cleanly between a foreground ridge and the sky behind it instead of
stretching a skin between them. It is textured with the panorama itself and the
client renders it unlit: it is a photograph of distant scenery with its own light
in it, like the skybox it sits in front of.

Reads
  ContextKey.PANORAMA_TERRAIN          the object-removed panorama
  ContextKey.PANORAMA_DEPTH_METRIC     uncompressed calibrated depth
  ContextKey.PANORAMA_REGION_TYPE_MAP_TERRAIN   sky pixels are never part of the shell
  ContextKey.HEIGHT_MAP_PARAMS         grid size (start_m defaults to its half)
Writes
  ContextKey.FAR_FIELD_BACKDROP        Mesh (absent when nothing is far)
  FAR_FIELD_SUMMARY_KEY                object: what was built (always)
"""
from __future__ import annotations

import math
from logging import Logger
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter, zoom

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.panorama_segmentation.panorama_region_result import RegionType
from scene.mesh import Mesh

FAR_FIELD_SUMMARY_KEY = "far_field_backdrop_summary"


def far_field_radius(
    d: np.ndarray, start_m: float, far_m: float, scale_m: float, start_radius_m: float | None = None,
) -> np.ndarray:
    """Asymptotic mapping of depth beyond start_m into [start_radius_m, far_m).

    start_radius_m is where the shell begins in WORLD radius (the terrain grid's
    edge); start_m is the depth-map value at which content counts as far. They
    differ precisely because the depth models saturate: the far field begins at
    ~25 m in a map whose ground really was measured to 45 m."""
    d = np.asarray(d, dtype=np.float64)
    r0 = start_m if start_radius_m is None else start_radius_m
    over = np.clip(d - start_m, 0.0, None)
    return r0 + (far_m - r0) * (1.0 - np.exp(-over / max(scale_m, 1e-6)))


class FarFieldBackdropConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        start_m: float = 0.0,
        far_m: float = 500.0,
        scale_m: float = 0.0,
        start_radius_m: float = 0.0,
        far_band_fraction: float = 0.9,
        close_px: int = 10,
        stride_px: int = 8,
        max_depth_ratio: float = 1.8,
        feather_px: float = 3.0,
        texture_max_width: int = 4096,
        min_faces: int = 50,
        skirt: bool = True,
        skirt_start_fraction: float = 0.9,
        skirt_sky_below_deg: float = 1.0,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Depth-map value beyond which a pixel belongs to the backdrop. 0 = the
        # far-field start Panorama Depth Calibration measured (FAR_FIELD_START_M:
        # a fraction of the deepest non-sky depth, i.e. where the depth models
        # saturate), falling back to the grid's half-size.
        self.start_m = float(start_m)
        # World radius at which the shell begins (0 = the grid's half-size, so the
        # two domains meet at the grid's edge) and the radius the mapping approaches
        # for the most distant content. scale_m is the mapping's e-folding in depth
        # units; 0 = the saturated band's own width (p99.5 minus start), so the band
        # spreads over ~63% of [start_radius, far] and the depth noise inside it is
        # not amplified into a crumpled shell.
        self.start_radius_m = float(start_radius_m)
        self.far_band_fraction = float(far_band_fraction)
        self.close_px = int(close_px)
        self.far_m = float(far_m)
        self.scale_m = float(scale_m)
        # Vertex spacing in panorama pixels (8 on a 4096-wide panorama is 512x256
        # vertices, ~0.7 degrees).
        self.stride_px = max(2, int(stride_px))
        # Neighbouring vertices whose metric depths differ by more than this ratio
        # are not joined: that edge is a silhouette, not a surface.
        self.max_depth_ratio = float(max_depth_ratio)
        self.feather_px = float(feather_px)
        self.texture_max_width = int(texture_max_width)
        self.min_faces = int(min_faces)
        # Extend the ground from the grid's edge out to the shell (see run).
        self.skirt = bool(skirt)
        self.skirt_start_fraction = float(skirt_start_fraction)
        self.skirt_sky_below_deg = float(skirt_sky_below_deg)


class FarFieldBackdropStage(PipelineStage):
    @classmethod
    def config_class(cls) -> type[FarFieldBackdropConfiguration]:
        return FarFieldBackdropConfiguration

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: FarFieldBackdropConfiguration = self.config
        task = self.create_progress(3, "Far-field backdrop…")

        panorama = context.input_panorama(ContextKey.PANORAMA_TERRAIN)
        metric = context.input_depth(ContextKey.PANORAMA_DEPTH_METRIC)
        type_depth = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP_TERRAIN)
        if panorama is None or metric is None:
            self.log_info("Far-field backdrop: no panorama or no metric depth — skipping")
            context.add_object(FAR_FIELD_SUMMARY_KEY, {"faces": 0, "reason": "missing inputs"})
            self.finish_progress(task)
            return context

        params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}
        grid_half = float(params.get("grid_size_meters", 200.0)) / 2.0
        measured_start = context.input_object(ContextKey.FAR_FIELD_START_M)
        start_radius = cfg.start_radius_m if cfg.start_radius_m > 0 else grid_half

        W, H = panorama.width, panorama.height
        depth = self._fit(metric.depth, H, W, order=1)
        sky = None
        if type_depth is not None:
            sky = np.rint(self._fit(type_depth.depth, H, W, order=0)).astype(np.int32) == int(RegionType.SKY)
        # "Far" is the saturated band of the calibrated depth, not a distance: the
        # map is compressed toward max_depth_m (Panorama Depth Calibration), so the
        # Eiffel Tower, the far bank of the Seine and a 900 m peak all sit within a
        # few metres of its ceiling. The measured saturation start only moves the
        # boundary OUT, never in (a 10.8 m start once put the Rainier forest on the
        # shell), but the floor must be inside the map's own range: the old floor
        # was the grid's half-size, which the ceiling never exceeds, so no capture
        # ever got a backdrop ("nothing beyond 100 m" on every run). The floor is
        # now far_band_fraction of the map's ceiling: on Paris max(30, 0.9 x 70) =
        # 63 m takes the tower and the far bank; on Rainier max(73, 0.9 x 100) = 90 m
        # takes the mountain, whose landform pixels the ridge chains then hand back
        # to the terrain below.
        finite = depth[np.isfinite(depth) & (~sky if sky is not None else True)]
        depth_max = float(finite.max()) if finite.size else 0.0
        band_floor = cfg.far_band_fraction * depth_max
        start_m = cfg.start_m if cfg.start_m > 0 else max(float(measured_start or 0.0), band_floor)
        self.log_info(
            f"Far-field backdrop: depth ceiling {depth_max:.1f} m, measured far start "
            f"{float(measured_start or 0.0):.1f} m, band floor {band_floor:.1f} m -> far beyond {start_m:.1f} m"
        )
        far = np.isfinite(depth) & (depth > start_m)
        if sky is not None:
            far &= ~sky
        # A mountain the ridge chains anchored into the terrain (Region Map /
        # Terrain Reconstruction) must not also stand on this shell behind it, or
        # the viewer sees two summits. When any chain was anchored, the landform
        # types are the terrain's; the shell keeps what the grid cannot hold.
        chains = context.input_object(ContextKey.MOUNTAIN_RIDGE_CHAINS) or []
        if chains and type_depth is not None:
            types_full = np.rint(self._fit(type_depth.depth, H, W, order=0)).astype(np.int32)
            landform = np.isin(types_full, [int(RegionType.TERRAIN), int(RegionType.SNOW)])
            removed = int((far & landform).sum())
            far &= ~landform
            self.log_info(
                f"Far-field backdrop: {len(chains)} ridge chain(s) anchored in the terrain, "
                f"{removed} terrain/snow px left to it"
            )
        # A built structure that reaches the far band goes to the shell whole. The
        # depth model reads a thin lattice against the sky at its surroundings'
        # distance: the Eiffel Tower measured 23-30 m over most of its height and
        # only its top quarter sat in the band, so the shell carried a sliver of
        # tip and the rest, which no terrain stage builds, vanished.
        if type_depth is not None and far.any():
            from scipy.ndimage import label as _label
            types_full = np.rint(self._fit(type_depth.depth, H, W, order=0)).astype(np.int32)
            built = types_full == int(RegionType.BUILT)
            if built.any():
                labels, count = _label(built, structure=np.ones((3, 3), dtype=bool))
                touching = np.unique(labels[built & far])
                touching = touching[touching > 0]
                if touching.size:
                    whole = np.isin(labels, touching) & ~far
                    depth = np.where(whole, depth_max, depth)
                    far |= whole
                    self.log_info(
                        f"Far-field backdrop: {touching.size} built structure(s) reach the far band; "
                        f"{int(whole.sum())} px of them moved to the shell whole"
                    )
        # Bridge the sky-typed gaps inside a far structure. The Eiffel Tower's
        # lattice is typed half SKY, so per-pixel it is a cloud of specks the
        # stride-8 grid and the silhouette rule reduce to nothing; closed over
        # close_px it is one silhouette, and the sky pixels inside it keep the
        # panorama's own sky colour on the strip. Only sky-typed pixels are taken,
        # so real sky between two far things is never bridged.
        if cfg.close_px > 0 and far.any():
            from scipy.ndimage import binary_closing
            closed = binary_closing(far, iterations=cfg.close_px)
            gained = closed & ~far
            if sky is not None:
                gained &= sky
            if gained.any():
                depth = np.where(gained & ~(np.isfinite(depth) & (depth > start_m)), depth_max, depth)
                far |= gained
                self.log_info(f"Far-field backdrop: closed {int(gained.sum())} sky-typed px inside far structures")
        # A structure that became a 3D object (the Eiffel Tower, a far building)
        # stands in the scene as its own mesh; ObjectClear leaves a faint ghost of
        # it in the object-removed panorama, and the shell would carry that ghost
        # as a flat stepped silhouette right behind the model. Sky- and built-typed
        # pixels under the occluder mask are left off the shell.
        occluder = context.input_image(ContextKey.PANORAMA_FOREGROUND_OCCLUDER_MASK)
        if occluder is not None and type_depth is not None and far.any():
            occ = np.asarray(occluder.image.convert("L").resize((W, H), PILImage.NEAREST)) > 127
            types_full = np.rint(self._fit(type_depth.depth, H, W, order=0)).astype(np.int32)
            ghost = far & occ & np.isin(types_full, [int(RegionType.SKY), int(RegionType.BUILT)])
            if ghost.any():
                far &= ~ghost
                self.log_info(f"Far-field backdrop: {int(ghost.sum())} px of removed objects left to their 3D models")
        non_sky = depth[np.isfinite(depth) & (~sky if sky is not None else True)]
        p995 = float(np.percentile(non_sky, 99.5)) if non_sky.size else start_m
        scale_m = cfg.scale_m if cfg.scale_m > 0 else max(p995 - start_m, 1.0)
        self.advance_progress(task)

        # ── vertex grid ───────────────────────────────────────────────────────
        s = cfg.stride_px
        rows = np.arange(0, H, s)
        cols = np.arange(0, W + 1, s)  # one extra column closes the wrap
        cols_w = cols % W
        d_grid = depth[np.ix_(rows, cols_w)]
        far_grid = far[np.ix_(rows, cols_w)]
        lon = (cols / W - 0.5) * 2.0 * np.pi
        lat = (0.5 - rows / H) * np.pi
        LON, LAT = np.meshgrid(lon, lat)
        r = far_field_radius(d_grid, start_m, cfg.far_m, scale_m, start_radius_m=start_radius)
        X = r * np.cos(LAT) * np.sin(LON)
        Y = r * np.sin(LAT)
        Z = r * np.cos(LAT) * np.cos(LON)
        # Vertices that land inside the terrain grid's own square (the diagonals
        # reach 1.41x the half-size) would pierce the near terrain; the grid owns
        # that ground.
        inside_grid = (np.abs(X) < grid_half) & (np.abs(Z) < grid_half)
        shell = far_grid & ~inside_grid

        # ── ground skirt: the ground between the grid's edge and the shell ───
        # From a raised camera (Paris, 12 m up on a bridge) a ray a few degrees
        # below the horizon meets the ground hundreds of metres out, past the
        # grid's edge and short of the far shell, and nothing was there: the
        # skybox showed through as a sky-blue band between the river and the far
        # bank. Below-horizon pixels that are neither sky nor shell get a vertex
        # where their ray meets the ground at the grid edge's own height (per
        # azimuth), no farther out than the shell's foot in that column.
        skirt_full = np.zeros((H, W), dtype=bool)
        X2, Y2, Z2 = X.copy(), Y.copy(), Z.copy()
        skirt = np.zeros_like(shell)
        grounded = np.zeros_like(shell)
        edge_y = self._edge_heights(context, grid_half, cols, W)
        if cfg.skirt and edge_y is not None:
            # Below the horizon nothing is sky: the distant water reflecting it is
            # typed SKY and punched holes in the skirt.
            sky_below = np.radians(cfg.skirt_sky_below_deg)
            not_sky_grid = ~sky[np.ix_(rows, cols_w)] if sky is not None else np.ones_like(shell)
            not_sky_grid = not_sky_grid | (LAT < -sky_below)
            tan_down = np.tan(-LAT)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_hit = np.where((LAT < 0) & (edge_y[None, :] < 0), -edge_y[None, :] / tan_down, np.inf)
            hx, hz = t_hit * np.sin(LON), t_hit * np.cos(LON)
            # Past the grid's inscribed circle rather than its square: the water
            # sheet ends on a circle, and the skirt sits under any terrain that
            # is higher, so overlapping the square's corners costs nothing.
            beyond = np.hypot(hx, hz) >= cfg.skirt_start_fraction * grid_half
            # The shell's foot: the nearest horizontal radius of its vertices per column.
            horiz = r * np.cos(LAT)
            # A far vertex below the horizon whose ray meets the ground first is
            # pulled in to the ground: the built structures moved to the shell
            # whole (a far bank's quays) otherwise hang a wall 20-30 m below the
            # water at 350 m, which the skirt in front of it cannot join.
            grounded = far_grid & np.isfinite(t_hit) & (t_hit < horiz)
            horiz = np.where(grounded, t_hit, horiz)
            shell = shell & ~(grounded & (horiz < cfg.skirt_start_fraction * grid_half))
            rg = horiz / np.cos(LAT)
            X = np.where(grounded, rg * np.cos(LAT) * np.sin(LON), X)
            Y = np.where(grounded, rg * np.sin(LAT), Y)
            Z = np.where(grounded, rg * np.cos(LAT) * np.cos(LON), Z)
            X2, Y2, Z2 = X.copy(), Y.copy(), Z.copy()
            # Far vertices dropped for standing inside the grid count too, or the
            # skirt runs out under them and the quads joining it to the shell
            # fail the silhouette test.
            foot = np.where(far_grid, horiz, np.inf).min(axis=0)
            foot = np.where(np.isfinite(foot), foot, cfg.far_m)
            foot = np.maximum(foot, cfg.skirt_start_fraction * grid_half)
            skirt = (LAT < 0) & np.isfinite(t_hit) & beyond & not_sky_grid & ~shell
            t = np.minimum(t_hit, foot[None, :])
            rr = t / np.cos(LAT)
            X2 = np.where(skirt, rr * np.cos(LAT) * np.sin(LON), X)
            Y2 = np.where(skirt, rr * np.sin(LAT), Y)
            Z2 = np.where(skirt, rr * np.cos(LAT) * np.cos(LON), Z)
            # Full-resolution alpha for the strip: the same test per pixel, with
            # each pixel's column taking its vertex column's edge height.
            if skirt.any():
                prow = np.arange(H)
                plat = (0.5 - prow / H) * np.pi
                pcol = np.minimum(np.arange(W) // s, len(cols) - 1)
                pedge = edge_y[pcol]
                plon = (np.arange(W) / W - 0.5) * 2.0 * np.pi
                with np.errstate(divide="ignore", invalid="ignore"):
                    pt = np.where((plat[:, None] < 0) & (pedge[None, :] < 0), -pedge[None, :] / np.tan(-plat)[:, None], np.inf)
                pbeyond = pt >= cfg.skirt_start_fraction * grid_half
                skirt_full = np.isfinite(pt) & pbeyond & ~far
                if sky is not None:
                    skirt_full &= ~sky | (plat < -sky_below)[:, None]
            self.log_info(
                f"Far-field backdrop: ground skirt of {int(skirt.sum())} vertices "
                f"(edge height {float(np.median(edge_y)):.1f} m) between the grid's edge and the shell"
            )
        valid = shell | skirt
        X, Y, Z = X2, Y2, Z2

        nr, nc = len(rows), len(cols)
        vid = np.full((nr, nc), -1, dtype=np.int64)
        vid[valid] = np.arange(int(valid.sum()))
        vertices = np.stack([X[valid], Y[valid], Z[valid]], axis=-1)
        if vertices.shape[0] == 0:
            self.log_info(f"Far-field backdrop: nothing beyond {start_m:.0f} m — no backdrop")
            context.add_object(FAR_FIELD_SUMMARY_KEY, {"faces": 0, "reason": "nothing far"})
            self.finish_progress(task)
            return context

        # ── faces (skip silhouettes) ──────────────────────────────────────────
        def ratio_ok(a, b):
            return np.maximum(a, b) / np.maximum(np.minimum(a, b), 1e-6) <= cfg.max_depth_ratio

        faces = []
        for i in range(nr - 1):
            v00, v01, v10, v11 = vid[i, :-1], vid[i, 1:], vid[i + 1, :-1], vid[i + 1, 1:]
            moved = skirt | grounded
            sk = moved[i, :-1] | moved[i, 1:] | moved[i + 1, :-1] | moved[i + 1, 1:]
            dg = np.where(sk[None, :], np.stack([
                np.sqrt(X[i] ** 2 + Y[i] ** 2 + Z[i] ** 2)[:-1], np.sqrt(X[i] ** 2 + Y[i] ** 2 + Z[i] ** 2)[1:],
                np.sqrt(X[i + 1] ** 2 + Y[i + 1] ** 2 + Z[i + 1] ** 2)[:-1], np.sqrt(X[i + 1] ** 2 + Y[i + 1] ** 2 + Z[i + 1] ** 2)[1:],
            ]), np.stack([d_grid[i, :-1], d_grid[i, 1:], d_grid[i + 1, :-1], d_grid[i + 1, 1:]]))
            d00, d01, d10, d11 = dg
            t1 = (v00 >= 0) & (v10 >= 0) & (v11 >= 0) & ratio_ok(d00, d10) & ratio_ok(d10, d11) & ratio_ok(d00, d11)
            t2 = (v00 >= 0) & (v11 >= 0) & (v01 >= 0) & ratio_ok(d00, d11) & ratio_ok(d11, d01) & ratio_ok(d00, d01)
            # Winding so the front face looks toward the camera at the origin.
            faces.append(np.stack([v00[t1], v11[t1], v10[t1]], axis=-1))
            faces.append(np.stack([v00[t2], v01[t2], v11[t2]], axis=-1))
        faces = np.concatenate(faces, axis=0) if faces else np.zeros((0, 3), dtype=np.int64)
        if faces.shape[0] < cfg.min_faces:
            self.log_info(f"Far-field backdrop: only {faces.shape[0]} face(s) beyond {start_m:.0f} m — no backdrop")
            context.add_object(FAR_FIELD_SUMMARY_KEY, {"faces": int(faces.shape[0]), "reason": "too few faces"})
            self.finish_progress(task)
            return context
        self.advance_progress(task)

        # ── texture: the panorama rows the shell uses, alpha = far & not sky ──
        used_rows = rows[valid.any(axis=1)]
        row_min, row_max = int(max(0, used_rows.min() - s)), int(min(H - 1, used_rows.max() + s))
        # Colour from the delit panorama when Terrain Texture Generation made one,
        # so the client can light this mesh like everything in front of it.
        colour = context.input_panorama(ContextKey.PANORAMA_TERRAIN_DELIT)
        if colour is None or (colour.width, colour.height) != (panorama.width, panorama.height):
            colour = panorama
        rgb = np.asarray(colour.rgb(), dtype=np.uint8)[row_min:row_max + 1]
        alpha = (far | skirt_full)[row_min:row_max + 1].astype(np.float32)
        if cfg.feather_px > 0:
            alpha = gaussian_filter(alpha, sigma=cfg.feather_px)
        texture = PILImage.fromarray(np.dstack([rgb, (alpha.clip(0, 1) * 255).astype(np.uint8)]), "RGBA")
        if texture.width > cfg.texture_max_width:
            scale = cfg.texture_max_width / texture.width
            texture = texture.resize((cfg.texture_max_width, max(1, int(texture.height * scale))), PILImage.LANCZOS)
        strip_h = row_max - row_min + 1
        ROW, COL = np.meshgrid(rows, cols, indexing="ij")
        uv = np.stack([
            COL[valid] / W,
            1.0 - (ROW[valid] - row_min) / max(strip_h - 1, 1),
        ], axis=-1)
        inward = -vertices / np.linalg.norm(vertices, axis=-1, keepdims=True).clip(1e-6)
        # Lit by the client, so the normals carry the mountain's shape: the shell's
        # own surface normals (turned to face the camera), blended with the inward
        # direction and a little up so depth noise cannot make it blotchy. Purely
        # inward normals made a backlit range one flat dark sheet.
        shape = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).vertex_normals
        shape = np.where((np.sum(shape * inward, axis=-1) < 0)[:, None], -shape, shape)
        normals = 0.5 * shape + 0.35 * inward + 0.15 * np.array([0.0, 1.0, 0.0])
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True).clip(1e-6)

        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=1.0,
            alphaMode="MASK", alphaCutoff=0.5, doubleSided=True,
        )
        tm = trimesh.Trimesh(
            vertices=vertices, faces=faces, vertex_normals=normals,
            visual=trimesh.visual.TextureVisuals(uv=uv, material=material), process=False,
        )
        context.add_mesh(ContextKey.FAR_FIELD_BACKDROP, Mesh(tm))

        d_far = d_grid[shell] if shell.any() else np.array([0.0])
        summary = {
            "faces": int(faces.shape[0]), "vertices": int(vertices.shape[0]),
            "start_m": round(start_m, 1), "start_radius_m": round(start_radius, 1),
            "scale_m": round(scale_m, 1), "far_m": cfg.far_m,
            "metric_depth_p50": round(float(np.median(d_far)), 1),
            "metric_depth_max": round(float(d_far.max()), 1),
            "radius_max": round(float(np.linalg.norm(vertices, axis=-1).max()), 1),
            "strip_rows": [row_min, row_max],
        }
        context.add_object(FAR_FIELD_SUMMARY_KEY, summary)
        self.log_info(
            f"Far-field backdrop: {summary['vertices']} vertices, {summary['faces']} faces for everything "
            f"deeper than {start_m:.1f} m in the depth map (p50 {summary['metric_depth_p50']} m, max "
            f"{summary['metric_depth_max']} m), stood up from {start_radius:.0f} m out to "
            f"{summary['radius_max']} m (scale {scale_m:.1f} m); "
            f"{texture.width}x{texture.height} px strip"
        )
        if self.temp is not None:
            texture.save(self.temp / "far_field_strip.png")
            preview = np.asarray(panorama.rgb(), dtype=np.uint8).copy()
            preview[skirt_full] = (preview[skirt_full] * 0.5 + np.array([0, 128, 255]) * 0.5).astype(np.uint8)
            preview[far] = (preview[far] * 0.5 + np.array([255, 128, 0]) * 0.5).astype(np.uint8)
            PILImage.fromarray(preview).resize((W // 4, H // 4)).save(self.temp / "far_field_overlay.png")
        self.advance_progress(task)
        self.finish_progress(task)
        return context

    def _edge_heights(self, context: PipelineContext, grid_half: float, cols: np.ndarray, W: int) -> "np.ndarray | None":
        """Ground height (camera-relative Y) at the terrain grid's edge along each
        vertex column's azimuth: the highest of the terrain and water surfaces in
        the outer 5% of the grid, binned by azimuth and filled around the circle."""
        pts = []
        for key in (ContextKey.TERRAIN_MESH, ContextKey.WATER_MESH):
            m = context.input_mesh(key)
            if m is not None:
                pts.append(np.asarray(m.mesh.vertices, dtype=np.float64))
        if not pts:
            return None
        v = np.concatenate(pts, axis=0)
        edge = np.maximum(np.abs(v[:, 0]), np.abs(v[:, 2])) >= 0.95 * grid_half
        if edge.sum() < 16:
            return None
        v = v[edge]
        bins = 180
        az = np.arctan2(v[:, 0], v[:, 2])  # lon: X = sin(lon), Z = cos(lon)
        b = ((az + np.pi) / (2 * np.pi) * bins).astype(int) % bins
        heights = np.full(bins, np.nan)
        for k in range(bins):
            sel = v[b == k, 1]
            if sel.size:
                heights[k] = np.percentile(sel, 90)
        if np.isnan(heights).all():
            return None
        idx = np.arange(bins)
        ok = ~np.isnan(heights)
        heights = np.interp(idx, idx[ok], heights[ok], period=bins)
        lon = (cols / W - 0.5) * 2.0 * np.pi
        cb = ((lon + np.pi) / (2 * np.pi) * bins).astype(int) % bins
        return heights[cb]

    @staticmethod
    def _fit(array: np.ndarray, H: int, W: int, order: int) -> np.ndarray:
        if array.shape == (H, W):
            return array
        return zoom(array.astype(np.float32), (H / array.shape[0], W / array.shape[1]), order=order)

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(FAR_FIELD_SUMMARY_KEY) is not None
