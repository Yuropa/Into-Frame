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

A canopy standing in front of a far landform (the firs under Rainier's summit)
is a second layer of the same mesh. On one sheet the forest and the mountain
behind it were a skin stretched between two depths with one picture on it: a
step to the side slid the trees and there was no mountain behind them to show.
The canopy keeps its own depth and its photographed pixels on a strip of its
own; the sheet behind it is carried on at the landform's depth and painted with
the landform's colours continued under the trees.

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


def _drop_upright_slivers(vertices: np.ndarray, faces: np.ndarray, min_span_deg: float,
                          min_height_m: float) -> tuple[np.ndarray, int]:
    """`faces` without the connected pieces that are taller than min_height_m and
    cover less than min_span_deg of azimuth, and how many pieces that was."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = vertices.shape[0]
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]]], axis=0)
    graph = coo_matrix((np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    face_comp = comp[faces[:, 0]]
    az = np.degrees(np.arctan2(vertices[:, 0], vertices[:, 2]))
    used = np.zeros(n, dtype=bool)
    used[faces.ravel()] = True
    drop = []
    for c in np.unique(face_comp):
        sel = used & (comp == c)
        y = vertices[sel, 1]
        if float(y.max() - y.min()) <= min_height_m:
            continue
        a = np.sort(az[sel])
        span = 360.0 - float(np.diff(np.concatenate([a, a[:1] + 360.0])).max())
        if span < min_span_deg:
            drop.append(c)
    if not drop:
        return faces, 0
    return faces[~np.isin(face_comp, drop)], len(drop)


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
        start_at_grid_edge: bool = True,
        far_m: float = 500.0,
        scale_m: float = 0.0,
        start_radius_m: float = 0.0,
        far_band_fraction: float = 0.9,
        close_px: int = 10,
        stride_px: int = 8,
        max_depth_ratio: float = 1.8,
        feather_px: float = 3.0,
        texture_max_width: int = 8192,
        min_faces: int = 50,
        min_upright_span_deg: float = 5.0,
        min_upright_height_m: float = 10.0,
        skirt: bool = True,
        skirt_start_fraction: float = 0.9,
        skirt_sky_below_deg: float = 1.0,
        skirt_min_depth_fraction: float = 0.7,
        foot_curtain: bool = True,
        horizon_water_deg: float = 5.0,
        water_apron: bool = True,
        water_apron_min_fraction: float = 0.15,
        water_apron_radial_steps: int = 8,
        water_apron_open_water_m: float = 5000.0,
        # False: textured with the panorama's own (photographed) light and drawn
        # unlit. Relighting the delit far field with the client's sun through
        # normals estimated from km-scale depth made Rainier's summit bright and
        # blotchy where the photo has it in soft evening light.
        lit: bool = False,
        canopy_layer: bool = True,
        canopy_front_ratio: float = 0.7,
        far_crest_m: float = 2000.0,
        crest_flank_fraction: float = 0.33,
        canopy_fill_dilate_px: int = 3,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Depth-map value beyond which a pixel belongs to the backdrop. 0 = the
        # far-field start Panorama Depth Calibration measured (FAR_FIELD_START_M:
        # a fraction of the deepest non-sky depth, i.e. where the depth models
        # saturate), falling back to the grid's half-size.
        self.start_m = float(start_m)
        # True: the shell carries everything the depth map puts beyond the terrain
        # grid, at its own depth (far_field_radius with scale_m = 3 x the grid's
        # half-size is within a few metres of linear out to twice the grid). The
        # saturated-band rule below kept only the depth plateau, so Paris's far
        # bank, 120-270 m by its waterline, had nowhere to stand once it was no
        # longer measured at 500 m.
        self.start_at_grid_edge = bool(start_at_grid_edge)
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
        # A piece of the shell that stands up (taller than min_upright_height_m) on
        # less than this much azimuth is left off: where the near-content and
        # silhouette tests cut a far cliff or a far bank's buildings into single
        # columns, each column is a card standing alone against the sky (Shark Fin's
        # orange pillar, Paris's row of tower blocks). 0 keeps every piece.
        self.min_upright_span_deg = float(min_upright_span_deg)
        self.min_upright_height_m = float(min_upright_height_m)
        # Extend the ground from the grid's edge out to the shell (see run).
        self.skirt = bool(skirt)
        self.skirt_start_fraction = float(skirt_start_fraction)
        self.skirt_sky_below_deg = float(skirt_sky_below_deg)
        # A skirt pixel is drawn where its ray meets the ground plane, 90-650 m
        # out; one the depth map measured much nearer than that is near content
        # (Shark Fin: the surf at a sea stack's foot, 50 m away, copied onto the
        # skirt behind the stack as photo rectangles). Kept only when its depth
        # is at least this fraction of the distance it would be drawn at.
        self.skirt_min_depth_fraction = float(skirt_min_depth_fraction)
        # The shell's foot joined to the grid's edge at full resolution (see the
        # foot curtain in run): the skirt's vertices sit on stride rows, and the
        # wedge between the foot's ray and the ray to the grid's edge is narrower
        # than one.
        self.foot_curtain = bool(foot_curtain)
        # Water the panorama shows at or above the geometric horizon, up to this
        # many degrees, goes to the shell whatever its depth (see run).
        self.horizon_water_deg = float(horizon_water_deg)
        # Water the panorama shows BELOW the horizon is the water sheet's, not the
        # shell's: a pixel of river whose ray meets the water plane short of far_m
        # is left off the shell and the skirt (both carried the Seine as a
        # photograph on a billboard, standing over the real water sheet), and a
        # flat ring at the sheet's own level -- the apron, WATER_APRON_MESH, named
        # "water" for the client's water shader -- runs from the grid's edge out to
        # the shell's foot in every azimuth sector mostly typed water.
        self.water_apron = bool(water_apron)
        self.water_apron_min_fraction = float(water_apron_min_fraction)
        self.water_apron_radial_steps = max(1, int(water_apron_radial_steps))
        # In the azimuths Region Map declared open water (OCEAN_COLUMNS) the apron
        # runs out to this radius rather than to the shell's foot: a sheet ending
        # at far_m (500 m) is still 1.4 deg under the horizon from a 12.5 m
        # clifftop, and where the panorama's sea at the horizon is glare typed as
        # sky nothing stands on the shell there, so Shark Fin's sunward horizon
        # was a bar of skybox between the apron's edge and the sky. At 5 km the
        # edge sits 0.14 deg under the horizon -- a pixel or two in the viewer.
        self.water_apron_open_water_m = float(water_apron_open_water_m)
        self.lit = bool(lit)
        # Canopy (FOREST) under a landform's skyline run and nearer than
        # canopy_front_ratio x what stands behind it becomes a layer of its own,
        # and the sheet behind it takes the landform's depth and colours (see run).
        self.canopy_layer = bool(canopy_layer)
        self.canopy_front_ratio = float(canopy_front_ratio)
        # A landform whose crest is kilometres away is read at its foot at the
        # treeline's own distance (Panorama Depth Calibration lifts those pixels
        # to just past the grid, 220 m on Rainier under a 6000 m summit), so half
        # the mountain stood at the forest's radius and nothing separated the two.
        # Where a column's crest is beyond far_crest_m, the landform's run down
        # from its skyline stands no nearer than crest_flank_fraction of the crest.
        self.far_crest_m = float(far_crest_m)
        self.crest_flank_fraction = float(crest_flank_fraction)
        # The repainted area reaches this far past the canopy's own pixels, so the
        # dark rim the photograph has round every crown is not left on the sheet.
        self.canopy_fill_dilate_px = max(0, int(canopy_fill_dilate_px))


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
        # The shell's innermost radius: the grid's corner, not its half, or every
        # vertex off the grid's axes at the shell's foot lands inside the square
        # (|x|, |z| < half) and is dropped below -- with a plan-anchored far field
        # the mountain's foot stands at the foot radius and lost blocks of flank
        # at bearings 15-45 deg (Rainier band7-11, "stair-stepped, holes").
        start_radius = cfg.start_radius_m if cfg.start_radius_m > 0 else grid_half * math.sqrt(2.0)

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
        if cfg.start_m <= 0 and cfg.start_at_grid_edge:
            # "Far" in depth is beyond the grid's half (what the grid holds); the
            # shell's RADIUS starts at the corner (start_radius) so no vertex lands
            # inside the square -- two different numbers.
            start_m = float(grid_half)
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
        anchored = context.input_object(ContextKey.MOUNTAIN_RIDGE_CHAINS_ANCHORED)
        if chains and type_depth is not None:
            types_full = np.rint(self._fit(type_depth.depth, H, W, order=0)).astype(np.int32)
            landform = np.isin(types_full, [int(RegionType.TERRAIN), int(RegionType.SNOW)])
            columns = ""
            if anchored is not None:
                # Only the columns where a chain point actually anchored: Terrain
                # Reconstruction hands a chain's far points to this shell, so the
                # outpainted horizon anchors round the 360 while the photographed
                # mountain inside it (Rainier: bearings -41 to +63) stands here.
                # ...and only the pixels at or below the anchored crest in that
                # column: a chain point anchored at the treeline's height (its range
                # under the crest is the forest, inside the grid) says nothing about
                # the mountain above it, which stays here (band10-13: vertical
                # cutouts through the summit wherever the skyline dipped).
                from scipy.ndimage import minimum_filter1d
                crest_row = np.full(W, H, dtype=np.int64)
                for ch in anchored:
                    ch = np.asarray(ch, dtype=float)
                    if ch.ndim != 2 or len(ch) == 0:
                        continue
                    lon = np.arctan2(ch[:, 0], ch[:, 2])
                    lat = np.arctan2(ch[:, 1], np.hypot(ch[:, 0], ch[:, 2]).clip(1e-6))
                    cols_a = np.clip(((lon / (2.0 * np.pi) + 0.5) * W).astype(int), 0, W - 1)
                    rows_a = np.clip(((0.5 - lat / np.pi) * H).astype(int), 0, H - 1)
                    np.minimum.at(crest_row, cols_a, rows_a)
                crest_row = minimum_filter1d(crest_row, size=2 * max(1, W // 720) + 1, mode="wrap")
                col_mask = crest_row < H
                landform &= (np.arange(H)[:, None] >= crest_row[None, :])
                columns = f" in {col_mask.mean():.0%} of the columns, below the anchored crest"
            removed = int((far & landform).sum())
            far &= ~landform
            self.log_info(
                f"Far-field backdrop: {len(anchored) if anchored is not None else len(chains)} ridge chain(s) "
                f"anchored in the terrain{columns}, {removed} terrain/snow px left to it"
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
                    # Only a structure that is mostly far already, or whose
                    # not-yet-far pixels stand above the horizon (a tower against
                    # the sky, which no terrain stage builds). A quay on the near
                    # bank whose far end grazes the band otherwise came whole,
                    # trees and all, as a pale rectangle at the shell's foot
                    # (Paris, once the shell started at the grid's edge).
                    above = np.zeros((H, W), dtype=bool)
                    above[: H // 2] = True
                    n_all = np.bincount(labels.ravel(), minlength=count + 1)
                    n_far = np.bincount(labels[far].ravel(), minlength=count + 1)
                    n_out_above = np.bincount(labels[~far & above].ravel(), minlength=count + 1)
                    n_out = n_all - n_far
                    ok = (n_far >= 0.5 * n_all) | (n_out_above >= 0.6 * np.maximum(n_out, 1))
                    kept_near = touching[~ok[touching]]
                    touching = touching[ok[touching]]
                    if kept_near.size:
                        self.log_info(
                            f"Far-field backdrop: {kept_near.size} built structure(s) touching the far band "
                            f"left to the terrain (mostly near and below the horizon)"
                        )
                if touching.size:
                    whole = np.isin(labels, touching) & ~far
                    depth = np.where(whole, depth_max, depth)
                    far |= whole
                    self.log_info(
                        f"Far-field backdrop: {touching.size} built structure(s) reach the far band; "
                        f"{int(whole.sum())} px of them moved to the shell whole"
                    )
        # Water at or above the geometric horizon goes to the shell whatever its
        # depth. A flat water plane never rises above the horizon, so the water
        # mesh cannot show those rows; on Shark Fin the panorama's sea horizon sat
        # 3 deg above the camera's, the saturated top of the sea stood on the
        # shell and the 1.5 deg of sea under it (depth 400 m) fell through to the
        # skybox, so the far sea floated as bars over a strip of sky.
        if type_depth is not None and cfg.horizon_water_deg > 0:
            types_full = np.rint(self._fit(type_depth.depth, H, W, order=0)).astype(np.int32)
            top = max(0, H // 2 - int(round(np.radians(cfg.horizon_water_deg) * H / np.pi)))
            water_up = types_full == int(RegionType.WATER)
            water_up[:top] = False
            water_up[H // 2 + 1:] = False
            gained = water_up & ~far
            if gained.any():
                depth = np.where(gained, depth_max, depth)
                far |= gained
                self.log_info(
                    f"Far-field backdrop: {int(gained.sum())} px of water at or above the horizon "
                    f"moved to the shell (a water plane cannot reach above it)"
                )
        # ── water below the horizon belongs to the water sheet and its apron ──
        y_water = None
        water_covered = np.zeros((H, W), dtype=bool)
        open_water_cols = None
        river_reach = None
        water_mesh = context.input_mesh(ContextKey.WATER_MESH)
        if cfg.water_apron and water_mesh is not None and type_depth is not None:
            y_water = float(np.median(np.asarray(water_mesh.mesh.vertices, dtype=np.float64)[:, 1]))
            if y_water < -0.05:
                types_full = np.rint(self._fit(type_depth.depth, H, W, order=0)).astype(np.int32)
                plat_all = (0.5 - np.arange(H) / H) * np.pi
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_plane_all = np.where(plat_all < 0, -y_water / np.tan(-plat_all), np.inf)
                water_covered = (types_full == int(RegionType.WATER)) & (t_plane_all < cfg.far_m)[:, None]
                # Open water (Region Map's open_water_prior): in the azimuths where
                # the sea reaches the grid's edge, everything the panorama shows
                # below the horizon out to the shell is sea too, whatever the
                # outpainter drew there (Shark Fin: sand). Those pixels leave the
                # shell and the skirt, and the apron carries the sea out to the shell.
                ocean = context.input_object(ContextKey.OCEAN_COLUMNS) or {}
                if ocean.get("columns"):
                    ow = int(ocean.get("width") or W)
                    oc = np.zeros(ow, dtype=bool)
                    oc[np.asarray(ocean["columns"], dtype=np.int64)] = True
                    if ow != W:
                        oc = oc[(np.arange(W) * ow // W)]
                    # A river leaving the grid through its channel (open_water
                    # false) fills its sectors with the plane out to the shell like
                    # the sea does -- the outpainted far river is not shell content
                    # (Paris: pale panels at the horizon) -- but is not carried to
                    # the horizon: a water sector is a wedge between two azimuths,
                    # which stands as a teal box where a river's banks would converge.
                    if ocean.get("open_water", True):
                        open_water_cols = oc
                    sea_rows = (t_plane_all < cfg.far_m) & (plat_all < 0)
                    gained = sea_rows[:, None] & oc[None, :] & ~water_covered
                    if sky is not None:
                        gained &= ~sky
                    if open_water_cols is None:
                        river_reach = self._photographed_river_reach(
                            context, types_full == int(RegionType.WATER), oc, sea_rows, t_plane_all)
                        if river_reach is not None:
                            beyond = t_plane_all[:, None] > river_reach[None, :]
                            self.log_info(
                                f"Far-field backdrop: inside the photograph the river ends where it was photographed to "
                                f"({float(np.median(river_reach[np.isfinite(river_reach)])):.0f} m median over "
                                f"{int(np.isfinite(river_reach).sum())} columns) -- {int((gained & beyond).sum())} px "
                                f"above its far waterline stay with the shell"
                            )
                            gained &= ~beyond
                    water_covered |= gained
                    self.log_info(
                        f"Far-field backdrop: {'open water' if open_water_cols is not None else 'the river'} over "
                        f"{oc.mean():.0%} of the azimuths -- {int(gained.sum())} px below the horizon there "
                        f"treated as {'sea' if open_water_cols is not None else 'river out to the shell'}"
                    )
                dropped = int((far & water_covered).sum())
                far &= ~water_covered
                self.log_info(
                    f"Far-field backdrop: {int(water_covered.sum())} px of water lie on the water plane "
                    f"(y {y_water:.2f} m) short of {cfg.far_m:.0f} m -- {dropped} px left off the shell "
                    f"for the water apron"
                )
            else:
                y_water = None
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
        occ = None
        occluder = context.input_image(ContextKey.PANORAMA_FOREGROUND_OCCLUDER_MASK)
        if occluder is not None and type_depth is not None and far.any():
            occ = np.asarray(occluder.image.convert("L").resize((W, H), PILImage.NEAREST)) > 127
            types_full = np.rint(self._fit(type_depth.depth, H, W, order=0)).astype(np.int32)
            ghost = far & occ & np.isin(types_full, [int(RegionType.SKY), int(RegionType.BUILT)])
            # ...and the pixels of a removed object whose nearest surviving
            # neighbour is sky, whatever the inpainted texture was typed: a near
            # fir removed from against the sky left a faint grey-blue spire the
            # terrain regions typed terrain, which the shell then stood up as a
            # flat-topped pale block at the horizon (Rainier band17).
            if occ.any():
                from scipy.ndimage import distance_transform_edt
                _, (ri, ci) = distance_transform_edt(occ, return_indices=True)
                ghost |= far & occ & (types_full[ri, ci] == int(RegionType.SKY))
            if ghost.any():
                far &= ~ghost
                self.log_info(f"Far-field backdrop: {int(ghost.sum())} px of removed objects left to their 3D models")
        # A named landmark kept in the plate (Panorama Foreground Inpainting's
        # backdrop_kept) is placed by Scene Generation as a card cut from the
        # photograph; on the shell's stride grid it was a column of stepped blocks
        # (the Eiffel Tower's top), so its pixels are left off here.
        landmark = self._kept_landmark_mask(context, W, H)
        if landmark is not None and far.any():
            # Not dilated: the card covers only what its matte does (the Eiffel
            # Tower's from the first platform up), and the shell keeps the rest; a
            # dilated cut left a white band of sky between the two.
            # Sky and built only: the trees in front of its base stay on the shell
            # (cut with them, a white band stood under the Eiffel Tower).
            if type_depth is not None:
                types_full = np.rint(self._fit(type_depth.depth, H, W, order=0)).astype(np.int32)
                landmark &= np.isin(types_full, [int(RegionType.SKY), int(RegionType.BUILT)])
            # The shell drops every face along a cut, so its last rows under the card
            # are kept: it runs on behind the card's lower edge instead of stopping
            # a grid cell short of it.
            rows_hit = np.flatnonzero(landmark.any(axis=1))
            if rows_hit.size:
                landmark[max(rows_hit[-1] - 2 * cfg.stride_px, 0):, :] = False
            if (far & landmark).any():
                self.log_info(f"Far-field backdrop: {int((far & landmark).sum())} px of kept landmarks left to their cards")
                far &= ~landmark
        # The forest band Treeline Backdrop already stands at its measured depth.
        from pipeline.terrain.treeline_backdrop import TREELINE_MASK_KEY
        treeline_mask = context.input_image(TREELINE_MASK_KEY)
        tl = None
        if treeline_mask is not None and far.any():
            tl = np.asarray(treeline_mask.image.convert("L").resize((W, H), PILImage.NEAREST)) > 127
            taken = int((far & tl).sum())
            if taken:
                far &= ~tl
                self.log_info(f"Far-field backdrop: {taken} px left to the treeline backdrop")
        non_sky = depth[np.isfinite(depth) & (~sky if sky is not None else True)]
        p995 = float(np.percentile(non_sky, 99.5)) if non_sky.size else start_m
        scale_m = cfg.scale_m if cfg.scale_m > 0 else max(p995 - start_m, 1.0)
        if cfg.scale_m <= 0 and cfg.start_at_grid_edge:
            scale_m = 3.0 * float(start_radius)
        # ── a canopy in front of a far landform: its own layer ────────────────
        # depth_main is what the sheet stands at: the landform's own depth, floored
        # where its crest is kilometres away, and carried on under the canopy that
        # stands in front of it. The canopy's pixels keep their depth on a second
        # set of vertices (below), so a step aside slides the trees over a mountain
        # that is there behind them. Above the horizon only: below it a far pixel's
        # ray meets the ground first and the shell leaves it to the skirt.
        depth_main = depth
        canopy_split = np.zeros((H, W), dtype=bool)
        depth_canopy = None
        if cfg.canopy_layer and type_depth is not None and far.any():
            from scipy.ndimage import binary_closing, distance_transform_edt
            types_full = np.rint(self._fit(type_depth.depth, H, W, order=0)).astype(np.int32)
            forest = far & (types_full == int(RegionType.FOREST))
            landform = far & np.isin(types_full, [int(RegionType.TERRAIN), int(RegionType.SNOW)])
            if forest.any() and landform.any():
                rows_idx = np.arange(H)[:, None]
                cols_idx = np.arange(W)[None, :]
                d_safe = np.where(np.isfinite(depth), depth, 0.0)
                crest = np.where(landform, d_safe, 0.0).max(axis=0)
                floor_col = np.where(crest >= cfg.far_crest_m, cfg.crest_flank_fraction * crest, 0.0)
                # The landform's run down from its skyline, forest that is as far as
                # the landform itself included (the wooded flank), specks bridged.
                body = landform | (forest & (floor_col[None, :] > 0) & (d_safe >= floor_col[None, :]))
                closed = binary_closing(body, structure=np.ones((9, 1), dtype=bool))
                started = np.maximum.accumulate(closed, axis=0)
                ended = np.maximum.accumulate(started & ~closed, axis=0)
                top_run = closed & ~ended & far
                lifted = top_run & (floor_col[None, :] > 0) & (d_safe < floor_col[None, :])
                depth_main = np.where(lifted, floor_col[None, :], depth).astype(np.float32)
                bottom = np.where(top_run, rows_idx, -1).max(axis=0)
                cand = forest & ~top_run & (rows_idx > bottom[None, :]) & (bottom >= 0)[None, :] & (rows_idx < H // 2)
                # What stands behind a canopy pixel: the nearest far pixel above it
                # in its column that is not canopy.
                src = np.maximum.accumulate(np.where(far & ~cand, rows_idx, -1), axis=0)
                behind = np.where(src >= 0, depth_main[np.clip(src, 0, H - 1), cols_idx], 0.0)
                canopy_split = cand & (src >= 0) & (d_safe < cfg.canopy_front_ratio * behind)
                if canopy_split.any():
                    depth_main = np.where(canopy_split, behind, depth_main).astype(np.float32)
                    _, (ri, ci) = distance_transform_edt(~canopy_split, return_indices=True)
                    depth_canopy = d_safe[ri, ci].astype(np.float32)
                    self.log_info(
                        f"Far-field backdrop: {int(canopy_split.sum())} px of canopy stand in front of a landform "
                        f"(canopy {float(np.median(d_safe[canopy_split])):.0f} m, behind it "
                        f"{float(np.median(behind[canopy_split])):.0f} m) -- a layer of their own; "
                        f"{int(lifted.sum())} landform px under a crest beyond {cfg.far_crest_m:.0f} m stood "
                        f"at {cfg.crest_flank_fraction:.2f} of the crest's depth"
                    )
                else:
                    depth_main = depth
        self.advance_progress(task)

        # ── vertex grid ───────────────────────────────────────────────────────
        s = cfg.stride_px
        rows = np.arange(0, H, s)
        cols = np.arange(0, W + 1, s)  # one extra column closes the wrap
        cols_w = cols % W
        d_grid = depth_main[np.ix_(rows, cols_w)]
        far_grid = far[np.ix_(rows, cols_w)]
        # One cell of margin round the far region: a vertex is kept when any
        # neighbouring cell is far, so the mesh reaches past the silhouette and
        # the full-resolution alpha cuts the edge. With only the far cells the
        # boundary was a staircase of stride cells inside the silhouette --
        # stepped summit, sky between the treeline and the mountain's foot
        # (Rainier band7-14). The extra vertices stand at the shell's foot
        # radius (their depth is under the start) under alpha 0.
        from scipy.ndimage import maximum_filter
        far_grid = maximum_filter(far_grid.astype(np.uint8), size=3, mode="wrap").astype(bool)
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
        foot = None
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
            skirt &= ~water_covered[np.ix_(rows, cols_w)]
            t = np.minimum(t_hit, foot[None, :])
            if tl is not None:
                # The treeline's canopy stands at its measured depth; laid on the
                # skirt as well it was a pale carpet of trees at the shell's foot
                # in front of the treeline (Paris's left bank).
                skirt &= ~tl[np.ix_(rows, cols_w)]
            if cfg.skirt_min_depth_fraction > 0:
                near_content = skirt & np.isfinite(d_grid) & (d_grid < cfg.skirt_min_depth_fraction * t)
                if near_content.any():
                    self.log_info(
                        f"Far-field backdrop: {int(near_content.sum())} skirt vertex(es) measured nearer than "
                        f"{cfg.skirt_min_depth_fraction:.0%} of their drawn distance left off (near content)"
                    )
                    skirt &= ~near_content
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
                skirt_full &= ~water_covered
                if tl is not None:
                    skirt_full &= ~tl
            self.log_info(
                f"Far-field backdrop: ground skirt of {int(skirt.sum())} vertices "
                f"(edge height {float(np.median(edge_y)):.1f} m) between the grid's edge and the shell"
            )
        # ── foot curtain: the shell's foot down to the grid's edge ──────────
        # The skirt's vertices stand on stride rows (0.7 deg at 2048 rows) and
        # only below the horizon, and the wedge between the shell's foot ray and
        # the ray to the grid's edge is narrower than that, or above the horizon
        # altogether: on Rainier (band18) the lifted treeline's foot stood at eye
        # level on the horizon row while the grid's edge, 1 m lower at 200 m, lay
        # 0.3 deg under it -- a line of sky between the shell and the ground
        # under the whole treeline -- and at the right edge a forest whose foot is
        # hidden behind a rise in the meadow left 1.3 deg of sky. Per column, the
        # vertex under the shell's lowest one is moved onto the ray to the grid's
        # edge point (the edge's own ground height along that azimuth), at the
        # grid's edge, and takes the panorama row that ray sees; the pixels
        # between the foot and that row stand opaque on the quads between them.
        # They are drawn behind where they were photographed (the forest's lower
        # crowns at 150-190 m, the ground at 200-280 m), which from the capture
        # camera is exact and from a step aside a parallax error of a few metres.
        curtain = np.zeros_like(shell)
        curtain_full = np.zeros((H, W), dtype=bool)
        row_override = np.full(shell.shape, np.nan)
        if cfg.foot_curtain and edge_y is not None:
            lon_c = lon
            # Horizontal distance to the grid's square edge along each column.
            e_h = grid_half / np.maximum(np.maximum(np.abs(np.sin(lon_c)), np.abs(np.cos(lon_c))), 1e-6)
            lat_edge = np.arctan2(edge_y, e_h)
            row_edge = (0.5 - lat_edge / np.pi) * H
            has_foot = shell.any(axis=0)
            # The lowest shell vertex per column (the last True down the column).
            nr_ = shell.shape[0]
            i_foot = nr_ - 1 - np.argmax(shell[::-1], axis=0)
            i_c = i_foot + 1
            for j in np.where(has_foot)[0]:
                i = int(i_c[j])
                if i >= nr_ or shell[i, j] or skirt[i, j]:
                    continue
                if lat_edge[j] >= LAT[int(i_foot[j]), j]:
                    continue  # the grid's edge stands above the foot: no wedge
                curtain[i, j] = True
                row_override[i, j] = row_edge[j]
                X2[i, j] = e_h[j] * np.sin(lon_c[j])
                Y2[i, j] = edge_y[j]
                Z2[i, j] = e_h[j] * np.cos(lon_c[j])
            if curtain.any():
                # Full-resolution alpha: in each pixel column, from under the
                # lowest far pixel down to the edge ray's row, leaving sky (above
                # the skirt's sky floor), removed objects (their models stand in
                # the scene), water and anything measured nearer than half the
                # grid's edge (a shrub in front of the gap, not the ground in it).
                prow = np.arange(H)
                plat = (0.5 - prow / H) * np.pi
                pcol = np.minimum(np.arange(W) // s, len(cols) - 1)
                foot_full = np.where(far.any(axis=0), H - 1 - np.argmax(far[::-1], axis=0), -1)
                col_has = curtain.any(axis=0)[pcol]
                in_wedge = (prow[:, None] > foot_full[None, :]) & (prow[:, None] <= row_edge[pcol][None, :])
                curtain_full = in_wedge & col_has[None, :] & ~far
                if sky is not None:
                    curtain_full &= ~sky | (plat < -np.radians(cfg.skirt_sky_below_deg))[:, None]
                if occ is not None:
                    curtain_full &= ~occ
                curtain_full &= ~water_covered
                if tl is not None:
                    curtain_full &= ~tl
                curtain_full &= ~(np.isfinite(depth) & (depth < 0.5 * e_h[pcol][None, :]))
                gap_deg = np.degrees(LAT[i_foot[curtain.any(axis=0)], np.where(curtain.any(axis=0))[0]] - lat_edge[curtain.any(axis=0)])
                self.log_info(
                    f"Far-field backdrop: foot curtain of {int(curtain.sum())} vertices joins the shell's foot to the "
                    f"grid's edge over {int(curtain_full.sum())} px (wedge {float(np.median(gap_deg)):.2f} deg median, "
                    f"{float(gap_deg.max()):.2f} max)"
                )
        apron_faces = 0
        if y_water is not None and water_covered.any():
            apron_faces = self._water_apron(
                context, cfg, water_mesh, y_water, water_covered, t_plane_all, grid_half,
                cols, W, foot, open_water_cols, river_reach,
            )
        valid = shell | skirt | curtain
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
        # Silhouettes are judged on the radius the shell stands at, not the depth
        # map's value: a plan-anchored far field (Panorama Depth Calibration) runs
        # 245-6000 m across a mountain flank, which the radius mapping folds into
        # 200-500 m; in map units neighbouring flank pixels differed by 2x and the
        # shell tore into stair-stepped blocks (Rainier band7).
        radius = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        for i in range(nr - 1):
            v00, v01, v10, v11 = vid[i, :-1], vid[i, 1:], vid[i + 1, :-1], vid[i + 1, 1:]
            d00, d01, d10, d11 = radius[i, :-1], radius[i, 1:], radius[i + 1, :-1], radius[i + 1, 1:]
            t1 = (v00 >= 0) & (v10 >= 0) & (v11 >= 0) & ratio_ok(d00, d10) & ratio_ok(d10, d11) & ratio_ok(d00, d11)
            t2 = (v00 >= 0) & (v11 >= 0) & (v01 >= 0) & ratio_ok(d00, d11) & ratio_ok(d11, d01) & ratio_ok(d00, d01)
            # Winding so the front face looks toward the camera at the origin.
            faces.append(np.stack([v00[t1], v11[t1], v10[t1]], axis=-1))
            faces.append(np.stack([v00[t2], v01[t2], v11[t2]], axis=-1))
        faces = np.concatenate(faces, axis=0) if faces else np.zeros((0, 3), dtype=np.int64)
        if cfg.min_upright_span_deg > 0 and faces.shape[0]:
            faces, n_cards = _drop_upright_slivers(vertices, faces, cfg.min_upright_span_deg, cfg.min_upright_height_m)
            if n_cards:
                self.log_info(
                    f"Far-field backdrop: {n_cards} piece(s) standing more than {cfg.min_upright_height_m:.0f} m tall on "
                    f"less than {cfg.min_upright_span_deg:.0f} deg of azimuth left off (cards against the sky)"
                )
        if faces.shape[0] < cfg.min_faces:
            self.log_info(f"Far-field backdrop: only {faces.shape[0]} face(s) beyond {start_m:.0f} m — no backdrop")
            context.add_object(FAR_FIELD_SUMMARY_KEY, {"faces": int(faces.shape[0]), "reason": "too few faces"})
            self.finish_progress(task)
            return context
        self.advance_progress(task)

        # ── the canopy layer: the same grid, at the canopy's own depth ────────
        can_valid = np.zeros_like(valid)
        can_vertices = np.zeros((0, 3))
        can_faces = np.zeros((0, 3), dtype=np.int64)
        if depth_canopy is not None:
            # A cell with any canopy pixel in it, and one cell of margin for the
            # full-resolution alpha to cut the crowns out of. The depth is the
            # canopy's own carried outward, so a spire narrower than a cell does
            # not lean back toward the mountain at its edges.
            cells = maximum_filter(canopy_split.astype(np.uint8), size=(s, s), mode=("nearest", "wrap"))
            can_valid = maximum_filter(cells[np.ix_(rows, cols_w)], size=3, mode="wrap").astype(bool)
            rc = far_field_radius(depth_canopy[np.ix_(rows, cols_w)], start_m, cfg.far_m, scale_m, start_radius_m=start_radius)
            Xc, Yc, Zc = rc * np.cos(LAT) * np.sin(LON), rc * np.sin(LAT), rc * np.cos(LAT) * np.cos(LON)
            can_valid &= ~((np.abs(Xc) < grid_half) & (np.abs(Zc) < grid_half))
            cid = np.full((nr, nc), -1, dtype=np.int64)
            cid[can_valid] = np.arange(int(can_valid.sum())) + vertices.shape[0]
            can_vertices = np.stack([Xc[can_valid], Yc[can_valid], Zc[can_valid]], axis=-1)
            cf = []
            for i in range(nr - 1):
                v00, v01, v10, v11 = cid[i, :-1], cid[i, 1:], cid[i + 1, :-1], cid[i + 1, 1:]
                d00, d01, d10, d11 = rc[i, :-1], rc[i, 1:], rc[i + 1, :-1], rc[i + 1, 1:]
                t1 = (v00 >= 0) & (v10 >= 0) & (v11 >= 0) & ratio_ok(d00, d10) & ratio_ok(d10, d11) & ratio_ok(d00, d11)
                t2 = (v00 >= 0) & (v11 >= 0) & (v01 >= 0) & ratio_ok(d00, d11) & ratio_ok(d11, d01) & ratio_ok(d00, d01)
                cf.append(np.stack([v00[t1], v11[t1], v10[t1]], axis=-1))
                cf.append(np.stack([v00[t2], v01[t2], v11[t2]], axis=-1))
            can_faces = np.concatenate(cf, axis=0) if cf else can_faces
            if can_faces.shape[0] == 0:
                can_valid[:] = False
                can_vertices = np.zeros((0, 3))

        # ── texture: the panorama rows the shell uses, alpha = far & not sky ──
        used_rows = rows[valid.any(axis=1)]
        row_min, row_max = int(max(0, used_rows.min() - s)), int(min(H - 1, used_rows.max() + s))
        # Colour from the delit panorama when Terrain Texture Generation made one,
        # so the client can light this mesh like everything in front of it.
        colour = context.input_panorama(ContextKey.PANORAMA_TERRAIN_DELIT) if cfg.lit else None
        if colour is None or (colour.width, colour.height) != (panorama.width, panorama.height):
            colour = panorama
        rgb_full = np.asarray(colour.rgb(), dtype=np.uint8)
        rgb = rgb_full[row_min:row_max + 1]
        alpha = (far | skirt_full | curtain_full)[row_min:row_max + 1].astype(np.float32)
        if cfg.feather_px > 0:
            alpha = gaussian_filter(alpha, sigma=cfg.feather_px)
        # At the photograph's resolution where it was photographed (see
        # util.captured_cone.sharpen_strip): Rainier's snowfields were blocky.
        from util.captured_cone import sharpen_strip

        def strip_texture(rgb_rows, alpha_rows, first_row, plain=None):
            made = sharpen_strip(context, panorama, rgb_rows, alpha_rows, first_row, cfg.texture_max_width, plain=plain)
            if made is None:
                made = PILImage.fromarray(np.dstack([rgb_rows, (alpha_rows.clip(0, 1) * 255).astype(np.uint8)]), "RGBA")
            return made

        plain = None
        if can_valid.any():
            # Behind the canopy the sheet shows the landform continued: the
            # canopy's pixels (and the rim round them) repainted from what
            # surrounds them, and no photographed detail multiplied back in.
            rgb, plain = self._paint_behind(rgb, canopy_split[row_min:row_max + 1],
                                            (far & ~canopy_split)[row_min:row_max + 1], cfg.canopy_fill_dilate_px)
        texture = strip_texture(rgb, alpha, row_min, plain)
        main_px = texture.height
        gutter_px = 0
        can_row_min = can_row_max = 0
        if can_valid.any():
            can_rows = rows[can_valid.any(axis=1)]
            can_row_min, can_row_max = int(max(row_min, can_rows.min() - s)), int(min(row_max, can_rows.max() + s))
            # The canopy's layer covers everything the sheet behind it had
            # repainted (the rim round the crowns included), so from the camera
            # the two together are the photograph, pixel for pixel; a narrower
            # cut-out left the repainted rim showing as a pale line round the
            # forest. A light feather: a fir's tip is a few pixels wide.
            can_alpha = plain[can_row_min - row_min:can_row_max + 1 - row_min].astype(np.float32)
            if cfg.feather_px > 0:
                can_alpha = np.maximum(can_alpha, gaussian_filter(can_alpha, sigma=0.5 * cfg.feather_px) * 2.0).clip(0, 1)
            can_texture = strip_texture(rgb_full[can_row_min:can_row_max + 1], can_alpha, can_row_min)
            if can_texture.width != texture.width:
                can_texture = can_texture.resize(
                    (texture.width, max(1, round(can_texture.height * texture.width / can_texture.width))), PILImage.LANCZOS)
            # One texture, the canopy's strip under the sheet's, a transparent
            # gutter between them carrying each strip's edge colours.
            gutter_px = 16
            a_main, a_can = np.asarray(texture), np.asarray(can_texture)
            gut = np.concatenate([np.repeat(a_main[-1:], gutter_px // 2, axis=0),
                                  np.repeat(a_can[:1], gutter_px - gutter_px // 2, axis=0)], axis=0).copy()
            gut[..., 3] = 0
            texture = PILImage.fromarray(np.concatenate([a_main, gut, a_can], axis=0), "RGBA")
        can_px = texture.height - main_px - gutter_px
        atlas_px = texture.height
        if texture.width > cfg.texture_max_width:
            scale = cfg.texture_max_width / texture.width
            texture = texture.resize((cfg.texture_max_width, max(1, int(texture.height * scale))), PILImage.LANCZOS)
        strip_h = row_max - row_min + 1
        ROW, COL = np.meshgrid(rows, cols, indexing="ij")
        # A curtain vertex sits on the ray to the grid's edge, between stride rows.
        ROW = np.where(np.isfinite(row_override), row_override, ROW.astype(np.float64))
        ROWG, _ = np.meshgrid(rows, cols, indexing="ij")
        y_main = (ROW[valid] - row_min) / max(strip_h - 1, 1) * (main_px - 1)
        y_can = main_px + gutter_px + (ROWG[can_valid] - can_row_min) / max(can_row_max - can_row_min, 1) * (can_px - 1)
        uv = np.stack([
            np.concatenate([COL[valid], COL[can_valid]]) / W,
            1.0 - np.concatenate([y_main, y_can]) / max(atlas_px - 1, 1),
        ], axis=-1)
        vertices = np.concatenate([vertices, can_vertices], axis=0)
        faces = np.concatenate([faces, can_faces], axis=0)
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
            "lit": cfg.lit,
            "water_apron_faces": int(apron_faces),
            "canopy_faces": int(can_faces.shape[0]),
            "canopy_px": int(canopy_split.sum()),
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
            preview[curtain_full] = (preview[curtain_full] * 0.5 + np.array([255, 0, 200]) * 0.5).astype(np.uint8)
            preview[far] = (preview[far] * 0.5 + np.array([255, 128, 0]) * 0.5).astype(np.uint8)
            PILImage.fromarray(preview).resize((W // 4, H // 4)).save(self.temp / "far_field_overlay.png")
        self.advance_progress(task)
        self.finish_progress(task)
        return context

    def _photographed_river_reach(self, context: PipelineContext, water: np.ndarray, river_cols: np.ndarray,
                                  sea_rows: np.ndarray, t_plane: np.ndarray) -> "np.ndarray | None":
        """Per panorama column, how far out on the water plane the photographed river
        reaches (inf outside the photograph and outside the river's azimuths).

        A river's sectors are filled with the plane out to the shell because the
        outpainted far river is not shell content. Inside the photograph the far
        end of the river is the photograph's: Paris's far bank, its bridge and
        the trees under the tower stand between the waterline (3 degrees under
        the horizon, 200 m on the plane) and the horizon, and the plane carried
        to the shell's foot at 500 m stood over them as a flat-topped blue slab
        wherever the treeline had a gap. The reach is the plane's range at the
        topmost water-typed row of the column, the median over 2 degrees of
        columns (a boat or a bridge pier is not a bank)."""
        from scipy.ndimage import median_filter
        from util.captured_cone import captured_cone_from_context
        H, W = water.shape
        cone = captured_cone_from_context(context, W, H)
        if cone is None:
            return None
        u0, u1 = cone.u_range
        inside = np.zeros(W, dtype=bool)
        inside[max(0, int(np.ceil(u0))): min(W, int(np.floor(u1)))] = True
        inside &= river_cols
        if not inside.any():
            return None
        below = water & sea_rows[:, None]
        has = below.any(axis=0) & inside
        if has.sum() < 8:
            return None
        top = np.where(has, below.argmax(axis=0), 0)
        xs = np.flatnonzero(inside)
        top_in = np.interp(xs, np.flatnonzero(has), top[has])
        k = max(3, int(round(2.0 / 360.0 * W)) | 1)
        top_in = median_filter(top_in, size=k, mode="nearest")
        reach = np.full(W, np.inf, dtype=np.float64)
        reach[xs] = t_plane[np.clip(np.rint(top_in).astype(np.int64), 0, H - 1)]
        return reach

    def _water_apron(
        self, context: PipelineContext, cfg: FarFieldBackdropConfiguration, water_mesh,
        y_water: float, water_covered: np.ndarray, t_plane: np.ndarray, grid_half: float,
        cols: np.ndarray, W: int, foot: "np.ndarray | None",
        open_water_cols: "np.ndarray | None" = None,
        river_reach: "np.ndarray | None" = None,
    ) -> int:
        """A flat ring at the water sheet's level from the grid's edge out to the
        shell's foot, in every vertex-column sector whose panorama pixels between
        those radii are mostly water. Written to WATER_APRON_MESH with the sheet's
        own material, so the client draws it with the same water shader."""
        r_in = float(grid_half) * cfg.skirt_start_fraction
        H = water_covered.shape[0]
        between = (t_plane >= r_in) & (t_plane < cfg.far_m)
        if not between.any():
            return 0
        rows_b = np.flatnonzero(between)
        cols_w = cols % W
        water_frac = water_covered[rows_b][:, cols_w].mean(axis=0)
        is_water = water_frac >= cfg.water_apron_min_fraction
        if is_water.sum() < 2:
            self.log_info("Far-field backdrop: no azimuth sector is mostly water between the grid edge and the shell -- no apron")
            return 0
        lon = (cols / W - 0.5) * 2.0 * np.pi
        K = cfg.water_apron_radial_steps
        r_out = np.full(len(cols), cfg.far_m, dtype=np.float64)
        if foot is not None:
            r_out = np.maximum(np.asarray(foot, dtype=np.float64), r_in * 1.05)
        if river_reach is not None:
            # A photographed river ends at its far waterline, not at the shell's
            # foot behind the far bank (see _photographed_river_reach).
            r_out = np.minimum(r_out, np.maximum(river_reach[cols_w], r_in * 1.05))
        n_open = 0
        if open_water_cols is not None and cfg.water_apron_open_water_m > cfg.far_m:
            # Open water reaches the horizon, whatever the shell does there.
            open_sector = open_water_cols[cols % W]
            r_out = np.where(open_sector, np.maximum(r_out, cfg.water_apron_open_water_m), r_out)
            is_water |= open_sector
            n_open = int(open_sector.sum())
        # Geometric radial spacing: fine at the grid edge, coarse toward the shell.
        radii = r_in * (r_out[None, :] / r_in) ** (np.arange(K + 1)[:, None] / K)  # (K+1, nc)
        y_apron = y_water - 0.03  # just under the sheet, so the two never z-fight where they overlap
        X = radii * np.sin(lon)[None, :]
        Z = radii * np.cos(lon)[None, :]
        Y = np.full_like(X, y_apron)
        vid = np.arange(X.size).reshape(X.shape)
        verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        faces = []
        for j in range(len(cols) - 1):
            if not (is_water[j] and is_water[j + 1]):
                continue
            a, b = vid[:-1, j], vid[1:, j]
            c, d = vid[:-1, j + 1], vid[1:, j + 1]
            faces.append(np.stack([a, b, d], axis=-1))
            faces.append(np.stack([a, d, c], axis=-1))
        if not faces:
            return 0
        faces = np.concatenate(faces, axis=0)
        # Front face up: flip any triangle whose normal points down.
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        flip = n[:, 1] < 0
        faces[flip] = faces[flip][:, [0, 2, 1]]
        used = np.unique(faces)
        remap = np.full(verts.shape[0], -1, dtype=np.int64)
        remap[used] = np.arange(used.size)
        verts, faces = verts[used], remap[faces]
        colour = [0.10, 0.30, 0.45, 0.90]
        try:
            base = water_mesh.mesh.visual.material.baseColorFactor
            if base is not None:
                base = np.asarray(base, dtype=np.float64)
                colour = [float(v) / (255.0 if base.max() > 1.0 else 1.0) for v in base[:4]]
        except Exception:
            pass
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=colour, metallicFactor=0.0, roughnessFactor=0.08, alphaMode="BLEND",
        )
        tm = trimesh.Trimesh(
            vertices=verts, faces=faces,
            vertex_normals=np.tile(np.array([[0.0, 1.0, 0.0]]), (verts.shape[0], 1)),
            visual=trimesh.visual.TextureVisuals(material=material), process=False,
        )
        apron = Mesh(tm)
        # The sheet wears the photographed water (TerrainMeshStage._photograph_water);
        # the apron continues it with the same texture through its own panorama UVs,
        # so the two do not part at the grid's edge.
        photo = context.input_image(ContextKey.WATER_PHOTO_TEXTURE)
        if photo is not None and getattr(water_mesh, "extra_uv", None) is not None:
            from pipeline.terrain.water_photo import attach_water_photo
            apron = attach_water_photo(apron, photo.image)
        context.add_mesh(ContextKey.WATER_APRON_MESH, apron)
        sectors = int(is_water.sum())
        self.log_info(
            f"Far-field backdrop: water apron of {faces.shape[0]} faces at y {y_apron:.2f} m over "
            f"{sectors} of {len(cols)} azimuth sectors, {r_in:.0f} m out to {float(r_out[is_water].max()):.0f} m"
            + (f" ({n_open} open-water sector(s) carried to {cfg.water_apron_open_water_m:.0f} m)" if n_open else "")
        )
        if self.temp is not None:
            apron.save(self.temp / "water_apron.glb")
        return int(faces.shape[0])

    @staticmethod
    def _paint_behind(rgb: np.ndarray, canopy: np.ndarray, source: np.ndarray, dilate_px: int) -> tuple[np.ndarray, np.ndarray]:
        """The strip with the canopy's pixels repainted as the landform continued
        under them, and the mask of what was repainted.

        A fill from the surrounding pixels alone (OpenCV's Telea) is a smear: the
        snow above the firs and the meadow below them drawn out into each other.
        The band of landform just above the canopy's tips is carried down instead,
        turned over and back again under the line of the tips (the upper envelope
        of the repainted area, smoothed along the strip), so what shows between
        two crowns after a step aside is the slope that stands above them at the
        photograph's grain. A band and not the whole flank: turned over whole,
        Rainier's snowfields a hundred rows up came down as white bars behind a
        wooded ridge. The Telea fill carries the first few pixels inside the
        area, where it is the continuation of the edge, and every pixel whose
        source is not landform (`source`)."""
        import cv2
        from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter1d, minimum_filter1d
        mask = binary_dilation(canopy, iterations=dilate_px) if dilate_px > 0 else canopy
        if not mask.any():
            return rgb, mask
        h, w = mask.shape
        r = np.flatnonzero(mask.any(axis=1))
        r0, r1 = max(0, int(r.min()) - 16), min(h, int(r.max()) + 17)
        out = np.ascontiguousarray(rgb).copy()
        out[r0:r1] = cv2.inpaint(np.ascontiguousarray(out[r0:r1]), mask[r0:r1].astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        has = mask.any(axis=0)
        top = np.where(has, np.argmax(mask, axis=0), h).astype(np.float64)
        span = 2 * max(8, w // 170) + 1
        axis = minimum_filter1d(top, size=span, mode="wrap")
        near = axis < h
        if near.any():
            # Columns with no canopy near them take the nearest column's line, so
            # the smoothing below has no cliff to round off.
            idx = distance_transform_edt(~near, return_distances=False, return_indices=True)[0]
            axis = gaussian_filter1d(axis[idx], sigma=span / 6.0, mode="wrap")
            rows = np.arange(h)[:, None]
            cols = np.arange(w)[None, :]
            band = max(12, h // 16)
            below = np.maximum(rows - np.rint(axis[None, :]), 0).astype(np.int64) % (2 * band)
            src = np.rint(axis[None, :]).astype(np.int64) - 1 - np.where(below < band, below, 2 * band - 1 - below)
            ok = mask & (src >= 0) & (src < h)
            ok &= (source & ~mask)[np.clip(src, 0, h - 1), cols]
            weight = np.clip(distance_transform_edt(mask) / 6.0, 0.0, 1.0)
            weight *= np.clip(2.0 * gaussian_filter(ok.astype(np.float32), sigma=2.0) - 1.0, 0.0, 1.0) * ok
            mirrored = np.ascontiguousarray(rgb)[np.clip(src, 0, h - 1), cols].astype(np.float32)
            out = (out * (1.0 - weight[..., None]) + mirrored * weight[..., None]).clip(0, 255).astype(np.uint8)
        return out, mask

    @staticmethod
    def _kept_landmark_mask(context: PipelineContext, W: int, H: int) -> "np.ndarray | None":
        """Panorama pixels of named landmarks left in the terrain plate (the same
        rule as SceneGenerationStage's landmark cards), or None."""
        kept = context.input_object("backdrop_kept") or []
        pano = context.input_panorama(ContextKey.PANORAMA)
        if not kept or pano is None:
            return None
        landmark_classes = {"tower", "monument", "landmark", "statue", "skyscraper", "lighthouse"}
        pw, ph = pano.width, pano.height
        mask = np.zeros((H, W), dtype=bool)
        for idx in kept:
            meta = context.input_object(f"metadata_{idx}") or {}
            crop = context.input_image(f"crop_{idx}")
            if not meta.get("name") or meta.get("class") not in landmark_classes or crop is None or not meta.get("box"):
                continue
            a = np.asarray(crop.image.convert("RGBA").getchannel("A")) > 127
            x, y = int(round(meta["box"][0])), int(round(meta["box"][1]))
            ry, rx = np.nonzero(a)
            rows = np.clip(((y + ry) * H) // ph, 0, H - 1)
            cols = (((x + rx) % pw) * W) // pw
            mask[rows, cols] = True
        return mask if mask.any() else None

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
