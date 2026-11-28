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
from util.image_utils import Image as UtilImage

TREELINE_MASK_KEY = "treeline_backdrop_mask"
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
        min_distance_no_form_m: float = 40.0,
        min_height_m: float = 1.0,
        max_depth_spread: float = 1.6,
        bin_px: int = 16,
        min_bin_fill: float = 0.5,
        min_bins: int = 4,
        min_group_bins: int = 3,
        max_bin_depth_jump: float = 0.12,
        max_base_elevation_deg: float = 8.0,
        feather_px: float = 2.0,
        texture_max_width: int = 8192,
        max_silhouette_slope: float = 1.0,
        silhouette_gap_px: int = 48,
        ground_reach_deg: float = 3.0,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Canopy closer than this is placed as individual instances by the object
        # pipeline (detected trees, painted trees); the backdrop takes over beyond.
        self.min_distance_m = float(min_distance_m)
        self.max_distance_m = float(max_distance_m)
        self.min_distance_no_form_m = float(min_distance_no_form_m)
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
        # Contiguous runs of fewer bins than this are dropped (lone trees).
        self.min_group_bins = int(min_group_bins)
        # Neighbouring bins further apart in depth than this fraction are not one run.
        self.max_bin_depth_jump = float(max_bin_depth_jump)
        # A treeline stands on the ground: a run whose bottom row sits higher
        # above the terrain under it than this floats (Rainier: the near trees'
        # crowns, read at the far depth, hung as black blobs 16 m up in the sky).
        # Measured against the terrain mesh, not the horizon: Rainier's meadow
        # climbs toward the mountain and its whole treeline stands 3-8 deg up.
        self.max_base_elevation_deg = float(max_base_elevation_deg)
        # Gaussian blur on the alpha so the silhouette is not a pixel staircase.
        self.feather_px = float(feather_px)
        self.texture_max_width = int(texture_max_width)
        # The band's top edge may descend at most this many rows per column
        # toward a gap wider than silhouette_gap_px: a band that ends in one
        # column stands as a box with a vertical side (Paris: the photographed
        # bank's canopy stopped dead at the photo's edge, and clumps on the
        # outpainted bank ended in walls). 1.0 is a 45-degree taper in
        # panorama pixels; 0 disables.
        self.max_silhouette_slope = float(max_silhouette_slope)
        self.silhouette_gap_px = int(silhouette_gap_px)
        # A quad whose canopy ends above the ground under it is carried down to
        # that ground, at most ground_reach_deg, wearing the panorama's own pixels
        # there (the bank, the quay under the trees). 0 leaves the base where the
        # canopy ends.
        self.ground_reach_deg = float(ground_reach_deg)


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

        # The plan band (Scene Generation) stands the photographed forest as species
        # geometry only where the plan's tree species have a procedural form. A
        # broadleaf band (Paris's planes and willows) keeps its cards, sparse and
        # flat at 100 m, so the ribbon carries that canopy from where the placed
        # trees thin out, as it did before the plan bands (final: 0 bins, the far
        # bank a row of cut-outs; hybrid3 with the ribbon from 40 m: 223 bins).
        min_distance = cfg.min_distance_m
        if cfg.min_distance_no_form_m > 0:
            from pipeline.scene_plan import ScenePlan
            from pipeline.species.species_assets import species_form
            plan = ScenePlan.from_context(context)
            trees = [sp for sp in (plan.items("species") if plan is not None else [])
                     if str(sp.get("class", "")).lower() == "tree"]
            if not any(species_form(sp) == "conifer" for sp in trees):
                min_distance = cfg.min_distance_no_form_m
                names = ", ".join(str(sp.get("name")) for sp in trees) or "none named"
                self.log_info(f"Treeline backdrop: the plan's tree species ({names}) have no procedural form; "
                              f"the ribbon starts at {min_distance:g} m, not {cfg.min_distance_m:g} m")

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
            # Only canopy that was far on the original panorama: near trees the
            # removal took are placed as objects, and restored at a treeline
            # depth a tree 12 m away stands as a 70 m slab (Rainier: removed
            # canopy median 12 m; Paris's far bank 99 m).
            obj_depth = context.input_depth(ContextKey.PANORAMA_OBJECT_DEPTH)
            if obj_depth is not None:
                od = self._fit(obj_depth.depth, H, W, order=1)
                restored &= np.isfinite(od) & (od >= min_distance)
            if restored.any():
                fill = float(min_distance + 0.9 * (min(cfg.max_distance_m, 100.0) - min_distance))
                ok = np.isfinite(depth) & (depth >= min_distance) & (depth <= cfg.max_distance_m)
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
        far = vegetation & np.isfinite(depth) & (depth >= min_distance) & (depth <= max_distance)
        row_angle = math.pi / H
        top = np.full(W, -1, dtype=np.int32)
        bottom = np.full(W, -1, dtype=np.int32)
        col_depth = np.full(W, np.nan, dtype=np.float32)
        floating = 0
        ground = self._ground_sampler(context)
        max_float = math.radians(cfg.max_base_elevation_deg)
        lift: list[tuple[float, float, float]] = []
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
                if lo > 0 and hi / lo > cfg.max_depth_spread:
                    # A canopy whose depth runs wider than a band at its fringes (sky
                    # between the leaves, the bank under it) is still a canopy: keep
                    # its longest stretch within the spread of its median rather than
                    # dropping the column. Dropped whole, a crown on the spread's edge
                    # (Paris r3b: 96-152 m at the photo's left edge, 1.58 against 1.6)
                    # cut seven bins out of the band, a notched gap at the frame.
                    dr = depth[run, c]
                    med = float(np.median(d))
                    k = math.sqrt(cfg.max_depth_spread)
                    inside = np.isfinite(dr) & (dr >= med / k) & (dr <= med * k)
                    best_seg, cur = (0, -1), None
                    for i, ok in enumerate(inside):
                        if ok:
                            cur = i if cur is None else cur
                            if i - cur > best_seg[1] - best_seg[0]:
                                best_seg = (cur, i)
                        else:
                            cur = None
                    if best_seg[1] - best_seg[0] + 1 < 2:
                        continue
                    run = run[best_seg[0]:best_seg[1] + 1]
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
                base_lat = (H / 2.0 - run[-1]) * row_angle
                ground_lat = 0.0
                if ground is not None:
                    lon = (c / W - 0.5) * 2.0 * math.pi
                    horiz = median * math.cos(base_lat)
                    ground_lat = math.atan2(ground(horiz * math.sin(lon), horiz * math.cos(lon)), horiz)
                lift.append((math.degrees(base_lat), math.degrees(ground_lat), median))
                if base_lat - ground_lat > max_float:
                    floating += 1
                    continue
                if best is None or height_m > best[2]:
                    best = (int(run[0]), int(run[-1]), height_m, median)
            if best is not None:
                top[c], bottom[c], col_depth[c] = best[0], best[1], best[3]
        if lift:
            arr = np.asarray(lift)
            q = lambda a: "/".join(f"{x:.1f}" for x in np.percentile(a, [10, 50, 90]))
            self.log_info(
                f"Treeline backdrop: {len(lift)} candidate run(s): base {q(arr[:, 0])} deg, ground under it "
                f"{q(arr[:, 1])} deg, lift {q(arr[:, 0] - arr[:, 1])} deg (p10/p50/p90), depth {q(arr[:, 2])} m"
            )
        if floating:
            self.log_info(f"Treeline backdrop: {floating} column run(s) floating more than "
                          f"{cfg.max_base_elevation_deg:g} deg above the ground under them dropped")
        if cfg.max_silhouette_slope > 0:
            top, bottom, tapered = self._taper_ends(top, bottom, cfg.max_silhouette_slope, cfg.silhouette_gap_px)
            if tapered:
                self.log_info(f"Treeline backdrop: {tapered} column(s) lowered so the band's ends slope "
                              f"at most {cfg.max_silhouette_slope:g} row(s) per column instead of standing as walls")
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
        # A treeline is a band: a run of one or two bins is a lone tree or a clump,
        # which the tree instances carry. Stood up as quads it folded (Rainier's
        # full rerun: two bins at the photo's centre ramping 62 -> 45 m drew a dark
        # V in front of the forest).
        if cfg.min_group_bins > 1:
            groups = []
            b = 0
            while b < n_bins:
                if bins[b] is None:
                    b += 1
                    continue
                e = b
                # A band's depth changes smoothly; a jump between neighbours is a
                # different thing (Rainier's lone trees zigzagged 49-57-44 m and
                # folded into a dark V), so it ends the run.
                while (e + 1 < n_bins and bins[e + 1] is not None
                       and abs(bins[e + 1]["depth"] - bins[e]["depth"])
                       <= cfg.max_bin_depth_jump * min(bins[e + 1]["depth"], bins[e]["depth"])):
                    e += 1
                groups.append((b, e))
                b = e + 1
            long_bins = {k for g0, g1 in groups if g1 - g0 + 1 >= cfg.min_group_bins for k in range(g0, g1 + 1)}
            bridged = 0
            for g0, g1 in groups:
                if g1 - g0 + 1 >= cfg.min_group_bins:
                    continue
                # A short run with the band on both sides of it is part of the
                # band, its depth off by a jump: dropped, it left a hole the sky
                # showed through (Paris: two full bins at 187-190 m between runs of
                # the far bank's planes, a pale slot in the treeline). It stands at
                # the depths of the band either side, so it cannot fold.
                if g0 - 1 in long_bins and g1 + 1 in long_bins:
                    d0, d1 = bins[g0 - 1]["depth"], bins[g1 + 1]["depth"]
                    for k in range(g0, g1 + 1):
                        t = (k - (g0 - 1)) / float(g1 + 1 - (g0 - 1))
                        bins[k]["depth"] = d0 + (d1 - d0) * t
                    bridged += g1 - g0 + 1
                    continue
                for k in range(g0, g1 + 1):
                    bins[k] = None
            if bridged:
                self.log_info(f"Treeline backdrop: {bridged} bin(s) between runs of the band kept at the band's depth")
        found = [b for b in bins if b is not None]
        if len(found) < cfg.min_bins:
            self.log_info(
                f"Treeline backdrop: {len(found)} bin(s) with upright far vegetation "
                f"(need {cfg.min_bins}) — no backdrop"
            )
            context.add_object(TREELINE_SUMMARY_KEY, {"bins": len(found), "reason": "too few bins"})
            self.finish_progress(task)
            return context

        # The canopy's base stands where the run test ended it, a degree or two
        # over the ground under it at that distance (the lift logged above), and
        # nothing between the two draws: Far Field Backdrop leaves the treeline's
        # pixels to it, and the terrain stops at the grid's edge, so a pale bar of
        # sky ran under the far bank's trees all round Paris. Each quad reaches
        # down to its ground line, carrying the panorama's pixels there.
        ground_row = {}
        if ground is not None and cfg.ground_reach_deg > 0:
            reach_rows = cfg.ground_reach_deg / 180.0 * H
            for b, entry in enumerate(bins):
                if entry is None:
                    continue
                lon = (0.5 * (entry["c0"] + entry["c1"]) / W - 0.5) * 2.0 * math.pi
                base_lat = (H / 2.0 - entry["bottom"]) * row_angle
                horiz = entry["depth"] * math.cos(base_lat)
                g_lat = math.atan2(ground(horiz * math.sin(lon), horiz * math.cos(lon)), horiz)
                g_row = H / 2.0 - g_lat / row_angle
                if entry["bottom"] < g_row <= entry["bottom"] + reach_rows:
                    ground_row[b] = g_row
                    entry["bottom"] = float(g_row)
            if ground_row:
                self.log_info(f"Treeline backdrop: {len(ground_row)} quad(s) carried down to the ground under them")
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
        rgb = self._match_cone_colour(context, rgb, vegetation, W, H)
        canopy_luma = float(rgb[vegetation].reshape(-1, 3).astype(np.float64).mean()) if vegetation.any() else 0.0
        rgb = rgb[row_min:row_max + 1]
        alpha = np.zeros((strip_h, W), dtype=np.float32)
        # Only the columns a quad stands in: the canopy of a dropped bin is not
        # drawn here, and the mask below tells Far Field Backdrop what is, or the
        # shell leaves those pixels off too and the sky shows through both.
        drawn = np.zeros(W, dtype=bool)
        for entry in found:
            drawn[entry["c0"]:entry["c1"]] = True
        for c in range(W):
            if top[c] >= 0 and drawn[c]:
                alpha[top[c] - row_min:bottom[c] - row_min + 1, c] = 1.0
        alpha *= vegetation[row_min:row_max + 1].astype(np.float32)
        alpha = self._drop_fragments(alpha, rgb, canopy_luma)
        for b, g_row in ground_row.items():
            entry = bins[b]
            for c in range(entry["c0"], entry["c1"]):
                col = alpha[:, c]
                lit = np.flatnonzero(col > 0.5)
                if lit.size:
                    col[lit[-1]:min(strip_h, int(math.ceil(g_row)) - row_min + 1)] = 1.0
        if cfg.feather_px > 0:
            alpha = gaussian_filter(alpha, sigma=cfg.feather_px)
        # At the photograph's resolution where it was photographed (see
        # util.captured_cone.sharpen_strip): Paris's far bank was a 127 px strip.
        from util.captured_cone import sharpen_strip
        # The pixels this strip carries, for Far Field Backdrop to leave off its shell.
        full_alpha = np.zeros((H, W), dtype=np.float32)
        full_alpha[row_min:row_max + 1] = alpha
        context.add_image(TREELINE_MASK_KEY, UtilImage(PILImage.fromarray((full_alpha.clip(0, 1) * 255).astype(np.uint8), "L")))
        texture = sharpen_strip(context, panorama, rgb, alpha, row_min, cfg.texture_max_width)
        if texture is None:
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

    def _match_cone_colour(self, context: PipelineContext, rgb: np.ndarray, forest: np.ndarray,
                           W: int, H: int) -> np.ndarray:
        """Canopy outside the photographed cone, coloured like the canopy inside it.

        Outside the cone the panorama is outpainted, and its trees drift: Rainier's
        came out golden-khaki beside the photo's dark firs, a band of autumn
        across half the horizon. A per-channel gain (0.35-1.6) brings the
        outpainted FOREST pixels' mean to the photographed ones'."""
        from util.captured_cone import captured_cone_from_context
        cone = captured_cone_from_context(context, W, H)
        if cone is None:
            return rgb
        u0, u1 = cone.u_range
        v0, v1 = cone.v_range
        cols = np.arange(W)
        in_cols = (cols >= u0) & (cols < u1)
        rows = np.arange(H)
        in_cone = in_cols[None, :] & ((rows >= v0) & (rows < v1))[:, None]
        ref = forest & in_cone
        out = forest & ~in_cols[None, :]
        if ref.sum() < 2000 or out.sum() < 2000:
            return rgb
        target = rgb[ref].reshape(-1, 3).astype(np.float64).mean(0)
        source = rgb[out].reshape(-1, 3).astype(np.float64).mean(0)
        gain = np.clip(target / np.maximum(source, 1.0), 0.35, 1.6)
        self.log_info(
            f"Treeline backdrop: outpainted canopy {source.round().astype(int).tolist()} -> photographed "
            f"{target.round().astype(int).tolist()} (gain {gain.round(2).tolist()})")
        rgb = rgb.copy()
        rgb[out] = np.clip(rgb[out].astype(np.float64) * gain, 0, 255).astype(np.uint8)
        return rgb

    @staticmethod
    def _ground_sampler(context: PipelineContext):
        """Camera-relative ground height at a world (x, z): the nearest terrain
        mesh vertex (the water sheet where there is no terrain). None without a
        terrain mesh, in which case the ground is taken as the horizon."""
        pts = []
        for key in (ContextKey.TERRAIN_MESH, ContextKey.WATER_MESH):
            m = context.input_mesh(key)
            if m is not None:
                pts.append(np.asarray(m.mesh.vertices, dtype=np.float64))
        if not pts:
            return None
        v = np.concatenate(pts, axis=0)
        if v.shape[0] < 16:
            return None
        from scipy.spatial import cKDTree
        tree = cKDTree(v[:, [0, 2]])
        ys = v[:, 1]

        def sample(x: float, z: float) -> float:
            _, i = tree.query((x, z))
            return float(ys[i])

        return sample

    @staticmethod
    def _taper_ends(top: np.ndarray, bottom: np.ndarray, slope: float, gap_px: int):
        """Per-column heights limited to a lower envelope of slope `slope` rows per
        column from every gap wider than gap_px (narrower gaps -- a removed pole,
        a bin the run test missed -- are bridged for the envelope only). Returns
        the adjusted top, bottom and how many columns were lowered. The panorama
        wraps, so the envelope is run on three copies and the middle one kept."""
        W = top.shape[0]
        have = top >= 0
        h = np.where(have, bottom - top + 1, 0).astype(np.float64)
        # Bridge gaps narrower than gap_px with the mean of the neighbours' heights.
        filled = h.copy()
        c = 0
        while c < W:
            if have[c]:
                c += 1
                continue
            e = c
            while e < W and not have[e]:
                e += 1
            if e - c < gap_px and c > 0 and e < W:
                filled[c:e] = 0.5 * (h[c - 1] + h[e])
            c = e
        # The envelope rises from each gap as a rounded shoulder, the outline of a
        # crown seen side-on, not a straight ramp: at a fixed slope every band end
        # was a 45-degree cut through the canopy, read as a wedge (Paris's banks
        # outside the photograph). `slope` sets how far from the gap the shoulder
        # starts: a column d columns in may stand sqrt(d (2 h - d)) rows of its own
        # height h, i.e. a quarter circle of radius h / slope.
        tiled = np.concatenate([filled, filled, filled])
        n = tiled.shape[0]
        gap = tiled <= 0
        dist = np.full(n, np.inf)
        last = -np.inf
        for i in range(n):
            if gap[i]:
                last = i
            dist[i] = i - last
        last = np.inf
        for i in range(n - 1, -1, -1):
            if gap[i]:
                last = i
            dist[i] = min(dist[i], last - i)
        radius = tiled / max(slope, 1e-6)
        d = np.minimum(dist * 1.0, radius)
        env = np.where(dist >= radius, np.inf,
                       slope * np.sqrt(np.clip(d * (2.0 * radius - d), 0.0, None)))
        env = env[W:2 * W]
        new_h = np.minimum(h, env)
        lowered = have & (new_h < h - 0.5)
        new_top = top.copy()
        new_bottom = bottom.copy()
        keep = have & (new_h >= 1.0)
        new_top[keep] = (bottom[keep] - np.rint(new_h[keep]).astype(np.int32) + 1)
        new_top[have & ~keep] = -1
        new_bottom[have & ~keep] = -1
        return new_top, new_bottom, int(lowered.sum())

    @staticmethod
    def _drop_fragments(alpha: np.ndarray, rgb: np.ndarray, canopy_luma: float,
                        min_share: float = 0.03, min_luma_ratio: float = 0.5) -> np.ndarray:
        """Squat or boxy scraps of canopy (a strip of shadowed ground, a trunk cut
        off by the run test) stood as dark slabs on the horizon (Rainier). A
        component under min_share of the canopy goes when it is wider than tall
        or fills over 75% of its box, and any component filling over 80% that is not a long band; a conifer is tall and pointed (fill ~0.5),
        and a broadleaf treeline is one large band. A component darker than
        min_luma_ratio of the panorama's canopy goes too: Rainier's shadowed
        forest came through the delit panorama as a streaked near-black block
        ([14,16,10] against canopy [34,39,27]) over the photographed hills; and
        one bluer than green, which is not canopy at all."""
        from scipy.ndimage import label, find_objects
        labels, n = label(alpha > 0.5)
        if n == 0:
            return alpha
        sizes = np.bincount(labels.ravel(), minlength=n + 1)
        total = float(sizes[1:].sum())
        drop = np.zeros(n + 1, dtype=bool)
        for k, sl in enumerate(find_objects(labels), start=1):
            if sl is None:
                continue
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            fill = sizes[k] / float(max(h * w, 1))
            if sizes[k] < min_share * total and (h < 1.2 * w or fill > 0.75):
                drop[k] = True
            # A filled box is never a canopy silhouette, whatever its share: once
            # near trees stopped being restored, Rainier's shadow block (fill 0.90)
            # was 11% of what was left. A long level band is one, though: Paris's
            # right bank is a row of planes 34 degrees wide and 7 tall, the largest
            # piece of the canopy (19%), fill 0.81, and without it the bank ended
            # in a bare slab under open sky.
            if fill > 0.8 and not (w >= 4 * h and sizes[k] >= min_share * total):
                drop[k] = True
            mask = labels[sl] == k
            mean = rgb[sl][mask].astype(np.float64).mean(axis=0)
            if canopy_luma > 0 and float(mean.mean()) < min_luma_ratio * canopy_luma:
                drop[k] = True
            # Canopy is never bluer than it is green: Rainier kept a navy pillar of
            # shadowed sky-lit something in front of the photographed treeline.
            if mean[2] > 1.05 * mean[1]:
                drop[k] = True
        return np.where(drop[labels], 0.0, alpha).astype(alpha.dtype)

    @staticmethod
    def _fit(array: np.ndarray, H: int, W: int, order: int) -> np.ndarray:
        if array.shape == (H, W):
            return array
        return zoom(array.astype(np.float32), (H / array.shape[0], W / array.shape[1]), order=order)

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(TREELINE_SUMMARY_KEY) is not None
