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
        stride_px: int = 8,
        max_depth_ratio: float = 1.8,
        feather_px: float = 3.0,
        texture_max_width: int = 4096,
        min_faces: int = 50,
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
        # The grid owns everything inside its half-size, ridge chains included
        # (TerrainReconstructionStage); this shell takes what lies beyond it. The
        # measured saturation start only moves the boundary OUT, never in: on the
        # Rainier capture a 10.8 m start had put the forest and the whole mountain on
        # the shell over a flat plain.
        start_m = cfg.start_m if cfg.start_m > 0 else max(float(measured_start or 0.0), grid_half)
        start_radius = cfg.start_radius_m if cfg.start_radius_m > 0 else grid_half

        W, H = panorama.width, panorama.height
        depth = self._fit(metric.depth, H, W, order=1)
        sky = None
        if type_depth is not None:
            sky = np.rint(self._fit(type_depth.depth, H, W, order=0)).astype(np.int32) == int(RegionType.SKY)
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
        valid = far_grid & ~inside_grid

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
            d00, d01, d10, d11 = d_grid[i, :-1], d_grid[i, 1:], d_grid[i + 1, :-1], d_grid[i + 1, 1:]
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
        rgb = np.asarray(panorama.rgb(), dtype=np.uint8)[row_min:row_max + 1]
        alpha = far[row_min:row_max + 1].astype(np.float32)
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
        normals = -vertices / np.linalg.norm(vertices, axis=-1, keepdims=True).clip(1e-6)

        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=1.0,
            alphaMode="MASK", alphaCutoff=0.5, doubleSided=True,
        )
        tm = trimesh.Trimesh(
            vertices=vertices, faces=faces, vertex_normals=normals,
            visual=trimesh.visual.TextureVisuals(uv=uv, material=material), process=False,
        )
        context.add_mesh(ContextKey.FAR_FIELD_BACKDROP, Mesh(tm))

        d_far = d_grid[valid]
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
            preview[far] = (preview[far] * 0.5 + np.array([255, 128, 0]) * 0.5).astype(np.uint8)
            PILImage.fromarray(preview).resize((W // 4, H // 4)).save(self.temp / "far_field_overlay.png")
        self.advance_progress(task)
        self.finish_progress(task)
        return context

    @staticmethod
    def _fit(array: np.ndarray, H: int, W: int, order: int) -> np.ndarray:
        if array.shape == (H, W):
            return array
        return zoom(array.astype(np.float32), (H / array.shape[0], W / array.shape[1]), order=order)

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(FAR_FIELD_SUMMARY_KEY) is not None
