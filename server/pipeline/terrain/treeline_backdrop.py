"""
Treeline backdrop: the photographed forest band, stood up at its measured depth.

A forest in a landscape photograph is not a set of trees. It is a band of canopy
whose individual crowns were never resolved, and the pipeline has no honest way
to repeat it as instances: Distribution Synthesis learns a scatter pattern from
"forest" exemplars that are themselves whole clumps, and on the 2026-10-13
Rainier run it (correctly) refused to paint them, so the band at the base of the
mountain was represented by nothing but the terrain texture's smear of it.

This stage takes the panorama's own VEGETATION pixels that stand UPRIGHT in the
far field -- a vertical run of canopy whose depth is nearly constant from its
foot to its crown, as opposed to a meadow whose depth increases row by row
toward the horizon -- and builds a ribbon of vertical quads around the camera,
one per azimuth bin, each placed at the band's median depth and textured with
the panorama's own pixels, the canopy's silhouette carried as alpha. It is the
same representation a card is for a single tree, applied to the thing the photo
actually shows.

Reads
  ContextKey.PANORAMA_TERRAIN                 the panorama (objects removed)
  ContextKey.PANORAMA_REGION_TYPE_MAP_TERRAIN its per-pixel RegionType indices
  ContextKey.PANORAMA_DEPTH                   calibrated radial depth, metres
Writes
  ContextKey.TREELINE_BACKDROP                Mesh (absent when no band was found)
  TREELINE_SUMMARY_KEY                        object: what was found (always)
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

TREELINE_SUMMARY_KEY = "treeline_backdrop_summary"


class TreelineBackdropConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        min_distance_m: float = 6.0,
        max_distance_m: float = 400.0,
        min_height_m: float = 1.0,
        max_depth_spread: float = 1.6,
        bin_px: int = 16,
        min_bin_fill: float = 0.5,
        min_bins: int = 4,
        feather_px: float = 2.0,
        texture_max_width: int = 4096,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Canopy closer than this is placed as individual instances by the object
        # pipeline (detected trees, painted trees); the backdrop takes over beyond.
        self.min_distance_m = float(min_distance_m)
        self.max_distance_m = float(max_distance_m)
        # A vertical run of vegetation must stand at least this tall (its angular
        # height times its depth) to count as canopy rather than ground cover.
        self.min_height_m = float(min_height_m)
        # Within one column's run, p90/p10 of depth must stay under this: canopy is a
        # wall (constant depth), a meadow is a floor (depth climbs toward the horizon).
        self.max_depth_spread = float(max_depth_spread)
        # Azimuth bin width in panorama columns; one quad per bin.
        self.bin_px = max(1, int(bin_px))
        # Fraction of a bin's columns that must carry a run for the bin to get a quad.
        self.min_bin_fill = float(min_bin_fill)
        # Fewer qualifying bins than this and the band is noise, not a treeline.
        self.min_bins = int(min_bins)
        # Gaussian blur on the alpha so the silhouette is not a pixel staircase.
        self.feather_px = float(feather_px)
        self.texture_max_width = int(texture_max_width)


class TreelineBackdropStage(PipelineStage):
    @classmethod
    def config_class(cls) -> type[TreelineBackdropConfiguration]:
        return TreelineBackdropConfiguration

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: TreelineBackdropConfiguration = self.config
        task = self.create_progress(3, "Treeline backdrop…")

        panorama = context.input_panorama(ContextKey.PANORAMA_TERRAIN)
        type_depth = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP_TERRAIN)
        depth_depth = context.input_depth(ContextKey.PANORAMA_DEPTH)
        if panorama is None or type_depth is None or depth_depth is None:
            self.log_info("Treeline backdrop: panorama, region types or depth missing — skipping")
            context.add_object(TREELINE_SUMMARY_KEY, {"bins": 0, "reason": "missing inputs"})
            self.finish_progress(task)
            return context

        W, H = panorama.width, panorama.height
        type_map = self._fit(type_depth.depth, H, W, order=0)
        depth = self._fit(depth_depth.depth, H, W, order=1)
        vegetation = np.rint(type_map).astype(np.int32) == int(RegionType.FOREST)

        # Canopy the object removal took away. Far trees are detected as objects
        # and removed from the terrain panorama like near ones, and where
        # ObjectClear repaints them as sky or stone the ribbon loses them: Paris's
        # far bank was 16.8k px of FOREST on the photo and 3.9k on the terrain
        # panorama, and stood as a bare grey quay. A distant treeline is backdrop,
        # not a placed object, so where the removal mask covers canopy the original
        # panorama's typing and colour are used, at the treeline's own depth band.
        restored = np.zeros_like(vegetation)
        orig_type = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP)
        removal = context.input_image(ContextKey.PANORAMA_FOREGROUND_MASK)
        original = context.input_panorama(ContextKey.PANORAMA)
        if orig_type is not None and removal is not None and original is not None:
            orig_forest = np.rint(self._fit(orig_type.depth, H, W, order=0)).astype(np.int32) == int(RegionType.FOREST)
            removed = np.asarray(removal.image.convert("L").resize((W, H), PILImage.NEAREST)) > 127
            restored = orig_forest & removed & ~vegetation
            if restored.any():
                fill = float(cfg.min_distance_m + 0.9 * (min(cfg.max_distance_m, 100.0) - cfg.min_distance_m))
                ok = np.isfinite(depth) & (depth >= cfg.min_distance_m) & (depth <= cfg.max_distance_m)
                depth = np.where(restored & ~ok, fill, depth)
                vegetation = vegetation | restored
                self.log_info(f"Treeline backdrop: {int(restored.sum())} px of canopy restored where objects were removed")
        self.advance_progress(task)

        # ── Per column: the tallest upright run of far vegetation ──────────────
        # Canopy in the far field (beyond where the depth models saturate) is the
        # Far Field Backdrop's; this ribbon is the near treeline only.
        max_distance = cfg.max_distance_m
        params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}
        grid_half = float(params.get("grid_size_meters", 200.0)) / 2.0
        far_start = context.input_object(ContextKey.FAR_FIELD_START_M)
        max_distance = min(max_distance, max(float(far_start or 0.0), grid_half))
        far = vegetation & np.isfinite(depth) & (depth >= cfg.min_distance_m) & (depth <= max_distance)
        row_angle = math.pi / H
        top = np.full(W, -1, dtype=np.int32)
        bottom = np.full(W, -1, dtype=np.int32)
        col_depth = np.full(W, np.nan, dtype=np.float32)
        for c in range(W):
            rows = np.flatnonzero(far[:, c])
            if rows.size < 2:
                continue
            # Split into contiguous runs; keep the tallest one that is a wall.
            breaks = np.flatnonzero(np.diff(rows) > 1)
            starts = np.concatenate([[0], breaks + 1])
            ends = np.concatenate([breaks + 1, [rows.size]])
            best = None
            for s0, e0 in zip(starts, ends):
                run = rows[s0:e0]
                d = depth[run, c]
                d = d[np.isfinite(d) & (d > 0)]
                if d.size < 2:
                    continue
                lo, hi = np.percentile(d, [10, 90])
                if lo <= 0 or hi / lo > cfg.max_depth_spread:
                    continue
                median = float(np.median(d))
                height_m = (run[-1] - run[0] + 1) * row_angle * median
                if height_m < cfg.min_height_m:
                    continue
                if best is None or height_m > best[2]:
                    best = (int(run[0]), int(run[-1]), height_m, median)
            if best is not None:
                top[c], bottom[c], col_depth[c] = best[0], best[1], best[3]
        self.advance_progress(task)

        # ── Per azimuth bin: one quad ──────────────────────────────────────────
        n_bins = max(1, W // cfg.bin_px)
        bins = []
        for b in range(n_bins):
            c0, c1 = b * cfg.bin_px, (W if b == n_bins - 1 else (b + 1) * cfg.bin_px)
            have = top[c0:c1] >= 0
            if have.mean() < cfg.min_bin_fill:
                bins.append(None)
                continue
            bins.append({
                "c0": c0, "c1": c1,
                "top": float(np.percentile(top[c0:c1][have], 25)),
                "bottom": float(np.percentile(bottom[c0:c1][have], 75)),
                "depth": float(np.median(col_depth[c0:c1][have])),
            })
        found = [b for b in bins if b is not None]
        if len(found) < cfg.min_bins:
            self.log_info(
                f"Treeline backdrop: {len(found)} bin(s) with upright far vegetation "
                f"(need {cfg.min_bins}) — no backdrop"
            )
            context.add_object(TREELINE_SUMMARY_KEY, {"bins": len(found), "reason": "too few bins"})
            self.finish_progress(task)
            return context

        row_min = int(max(0, math.floor(min(b["top"] for b in found)) - 4))
        row_max = int(min(H - 1, math.ceil(max(b["bottom"] for b in found)) + 4))
        strip_h = row_max - row_min + 1

        # ── Texture: the panorama strip with the runs' silhouette as alpha ─────
        # Colour from the delit panorama when Terrain Texture Generation made one,
        # so the client can light this mesh like everything in front of it.
        colour = context.input_panorama(ContextKey.PANORAMA_TERRAIN_DELIT)
        if colour is None or (colour.width, colour.height) != (panorama.width, panorama.height):
            colour = panorama
        rgb = np.asarray(colour.rgb(), dtype=np.uint8)
        if restored.any() and original is not None and (original.width, original.height) == (W, H):
            rgb = np.where(restored[..., None], np.asarray(original.rgb(), dtype=np.uint8), rgb)
        rgb = rgb[row_min:row_max + 1]
        alpha = np.zeros((strip_h, W), dtype=np.float32)
        for c in range(W):
            if top[c] >= 0:
                alpha[top[c] - row_min:bottom[c] - row_min + 1, c] = 1.0
        alpha *= vegetation[row_min:row_max + 1].astype(np.float32)
        if cfg.feather_px > 0:
            alpha = gaussian_filter(alpha, sigma=cfg.feather_px)
        texture = PILImage.fromarray(
            np.dstack([rgb, (alpha.clip(0, 1) * 255).astype(np.uint8)]), "RGBA"
        )
        if texture.width > cfg.texture_max_width:
            scale = cfg.texture_max_width / texture.width
            texture = texture.resize(
                (cfg.texture_max_width, max(1, int(texture.height * scale))), PILImage.LANCZOS
            )

        # ── Geometry ───────────────────────────────────────────────────────────
        def unproject(col: float, row: float, d: float) -> np.ndarray:
            lon = (col / W - 0.5) * 2.0 * math.pi
            lat = (0.5 - row / H) * math.pi
            return np.array([
                d * math.cos(lat) * math.sin(lon),
                d * math.sin(lat),
                d * math.cos(lat) * math.cos(lon),
            ])

        vertices, faces, uvs, normals = [], [], [], []
        up = np.array([0.0, 1.0, 0.0])
        for b, entry in enumerate(bins):
            if entry is None:
                continue
            left, right = bins[b - 1] if b > 0 else None, bins[b + 1] if b + 1 < n_bins else None
            # Shared edges take the mean of the neighbours so adjacent quads meet.
            d_l = (entry["depth"] + left["depth"]) / 2.0 if left else entry["depth"]
            d_r = (entry["depth"] + right["depth"]) / 2.0 if right else entry["depth"]
            t_l = min(entry["top"], left["top"]) if left else entry["top"]
            t_r = min(entry["top"], right["top"]) if right else entry["top"]
            b_l = max(entry["bottom"], left["bottom"]) if left else entry["bottom"]
            b_r = max(entry["bottom"], right["bottom"]) if right else entry["bottom"]
            c0, c1 = float(entry["c0"]), float(entry["c1"])
            base = len(vertices)
            corners = [
                (c0, b_l, d_l), (c1, b_r, d_r), (c1, t_r, d_r), (c0, t_l, d_l),
            ]  # bottom-left, bottom-right, top-right, top-left
            for col, row, d in corners:
                p = unproject(col, row, d)
                vertices.append(p)
                # Author V with 0 at the strip's bottom (trimesh flips V on export).
                uvs.append((col / W, 1.0 - (row - row_min) / max(strip_h - 1, 1)))
                inward = -np.array([p[0], 0.0, p[2]])
                inward /= max(float(np.linalg.norm(inward)), 1e-6)
                n = 0.6 * up + 0.4 * inward
                normals.append(n / np.linalg.norm(n))
            faces.append((base + 0, base + 1, base + 2))
            faces.append((base + 0, base + 2, base + 3))
            faces.append((base + 2, base + 1, base + 0))
            faces.append((base + 3, base + 2, base + 0))

        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture,
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="MASK",
            alphaCutoff=0.5,
            doubleSided=True,
        )
        tm = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            vertex_normals=np.asarray(normals, dtype=np.float64),
            visual=trimesh.visual.TextureVisuals(uv=np.asarray(uvs, dtype=np.float64), material=material),
            process=False,
        )
        context.add_mesh(ContextKey.TREELINE_BACKDROP, Mesh(tm))

        depths = [b["depth"] for b in found]
        heights = [(b["bottom"] - b["top"]) * row_angle * b["depth"] for b in found]
        summary = {
            "bins": len(found),
            "of": n_bins,
            "depth_min_m": round(min(depths), 1),
            "depth_max_m": round(max(depths), 1),
            "height_median_m": round(float(np.median(heights)), 1),
            "strip_rows": [row_min, row_max],
        }
        context.add_object(TREELINE_SUMMARY_KEY, summary)
        self.log_info(
            f"Treeline backdrop: {len(found)}/{n_bins} azimuth bins carry upright far canopy, "
            f"{summary['depth_min_m']}-{summary['depth_max_m']} m away, median height "
            f"{summary['height_median_m']} m; {len(faces) // 4} quad(s), "
            f"{texture.width}x{texture.height} px strip"
        )
        if self.temp is not None:
            texture.save(self.temp / "treeline_strip.png")
            preview = PILImage.fromarray(np.asarray(panorama.rgb(), dtype=np.uint8).copy())
            overlay = np.zeros((H, W, 4), dtype=np.uint8)
            overlay[row_min:row_max + 1, :, 1] = 255
            overlay[row_min:row_max + 1, :, 3] = (alpha.clip(0, 1) * 160).astype(np.uint8)
            preview.paste(PILImage.fromarray(overlay, "RGBA"), (0, 0), PILImage.fromarray(overlay, "RGBA"))
            preview.resize((W // 4, H // 4)).save(self.temp / "treeline_overlay.png")
        self.advance_progress(task)
        self.finish_progress(task)
        return context

    @staticmethod
    def _fit(array: np.ndarray, H: int, W: int, order: int) -> np.ndarray:
        if array.shape == (H, W):
            return array
        return zoom(array.astype(np.float32), (H / array.shape[0], W / array.shape[1]), order=order)

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(TREELINE_SUMMARY_KEY) is not None
