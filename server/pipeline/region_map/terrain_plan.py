"""
The terrain plan: a top-down map of the environment built by rule from what the
photograph shows, which the rest of the pipeline then follows.

The region map is a projection of the outpainted panorama's segmentation onto the
ground, and the outpainter does not reason about geography: it gave Shark Fin a
lagoon ringed by sand behind the camera, a sea that was a 30 degree wedge, and a
plateau where the photograph's right edge shows open water; it gave Paris a river
that narrows to a point behind the viewer. Every land type projects as a radial
wedge, so the map is a starburst of slivers.

This module takes the projected map as evidence and rebuilds the water and the
land cover with rules a cartographer would apply:

  * The photograph's water is one of four kinds. SEA: it meets the sky. RIVER: it
    crosses the view with both banks in frame and a steady width. LAKE: it is
    bounded on every visible side. NONE: the photograph shows no water, so the
    scene has none.
  * SEA fills every cell beyond its shore, continues past any land that stands in
    it (a stack, a headland is finite), and continues round each edge of the
    photograph that looks out over open water, along the coast's direction there,
    whatever the outpainter put beyond that edge. An edge that ends on land (a
    headland filling the frame's side) is not extended: the land goes on.
  * A RIVER is a channel of constant width along its fitted axis through the
    whole grid, with continuous banks; water anywhere outside the channel is bank.
  * Water that the photograph did not show exists only as a continuation of water
    it did: any outpainted water body not connected to photographed water is
    land. Small land islets inside a water body are water.
  * Land cover is drawn in regions at least a few metres across, not in radial
    slivers: a mode filter beyond the near field, water fixed.

Everything the plan decides is drawn into terrain_plan.png with the photographed
cone, the shore points and the coast lines it extended, so the map can be read
as a map.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from PIL import Image as PILImage, ImageDraw

from pipeline.panorama_segmentation.panorama_region_result import RegionType, colorize_region_type_map

_GROUND_TYPES = (
    int(RegionType.TERRAIN), int(RegionType.GROUND), int(RegionType.VEGETATION), int(RegionType.BUILT),
    int(RegionType.FOREST), int(RegionType.SNOW), int(RegionType.ROAD), int(RegionType.TRAIL),
)


@dataclass
class TerrainPlan:
    kind: str                                  # "sea" | "river" | "lake" | "none"
    water_y: Optional[float]
    water: np.ndarray                          # bool (res, res): the planned water
    notes: list[str] = field(default_factory=list)
    # debug geometry, world xz metres
    shore_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    coast_lines: list[list[tuple[float, float]]] = field(default_factory=list)   # polylines, world xz
    channel: Optional[dict] = None
    hfov_deg: float = 0.0

    def summary(self) -> dict:
        return {
            "kind": self.kind,
            "water_y": self.water_y,
            "hfov_deg": self.hfov_deg,
            "notes": list(self.notes),
            "coast_lines": [[list(p) for p in line] for line in self.coast_lines],
            "channel": self.channel,
        }


@dataclass
class PlanConfig:
    open_water_min_deg: float = 5.0        # this much of the cone meeting the sky as water makes it a sea
    bank_tolerance_m: float = 1.5          # photographed land this far above the water plane stays land
    edge_deg: float = 10.0                 # the coast's direction at a cone edge is fitted over this much of it
    extend_deg: float = 60.0               # and carried this far round past the edge
    river_margin_m: float = 5.0            # bank strip either side of the channel
    hole_m2: float = 200.0                 # land islets inside water smaller than this are water
    land_smooth_m: float = 4.0             # land-cover regions are at least this wide (0 disables)
    smooth_min_distance_m: float = 20.0    # the near field keeps its photographed detail


def _column_shore_stats(type_idx_map, depth, cone_hfov_deg, half_m, bin_m=2.0, min_px=3):
    """Per panorama column: azimuth, whether it is inside the cone, whether it meets
    the sky as water, the nearest water range and the farthest land range."""
    H, W = type_idx_map.shape
    water_idx, sky_idx = int(RegionType.WATER), int(RegionType.SKY)
    az = ((np.arange(W) + 0.5) / W - 0.5) * 360.0
    in_cone = np.abs(az) <= cone_hfov_deg / 2.0
    rows = np.arange(H)
    lat = (0.5 - (rows + 0.5) / H) * np.pi
    below = lat < 0
    finite = np.isfinite(depth) & (depth > 0)
    r_h = np.where(finite, depth * np.cos(lat)[:, None], np.nan)
    water_px = (type_idx_map == water_idx) & below[:, None] & finite
    land_px = np.isin(type_idx_map, _GROUND_TYPES) & below[:, None] & finite
    # The depth saturates at its far end; land there has no measured range.
    d_max = float(np.nanmax(np.where(finite, depth, np.nan))) if finite.any() else 0.0
    land_px &= depth < 0.97 * d_max
    r_shore = np.full(W, np.nan)
    r_land = np.full(W, np.nan)
    # Range occupancy per column: which bin_m bins of range hold photographed land
    # (a stack at 90-110 m, a headland at 380-450 m); what lies between them in a
    # sea scene is sea.
    nb = int(np.ceil(half_m / bin_m)) + 1
    occ = np.zeros((W, nb), dtype=bool)
    for c in np.flatnonzero(in_cone):
        if water_px[:, c].any():
            r_shore[c] = float(np.nanmin(r_h[water_px[:, c], c]))
        if land_px[:, c].any():
            rl = r_h[land_px[:, c], c]
            r_land[c] = float(np.nanmax(rl))
            rl = rl[rl < half_m]
            if rl.size:
                counts = np.bincount((rl / bin_m).astype(np.int64), minlength=nb)
                occ[c] = counts >= min_px
    occ |= np.roll(occ, 1, axis=1) | np.roll(occ, -1, axis=1)
    # Sea column: the first ground-typed pixel from the top is water.
    non_sky = type_idx_map != sky_idx
    first = np.argmax(non_sky, axis=0)
    has = non_sky.any(axis=0)
    band = max(2, H // 100)
    sea_col = np.zeros(W, dtype=bool)
    for c in np.flatnonzero(has & in_cone):
        r0 = int(first[c])
        seg = type_idx_map[r0:min(H, r0 + band), c]
        ground = seg[np.isin(seg, (water_idx,) + _GROUND_TYPES)]
        if ground.size and int(ground[0]) == water_idx:
            sea_col[c] = True
    return az, in_cone, sea_col, r_shore, r_land, occ, bin_m


def _nearest_land_fill(region_map, cells):
    """Retype `cells` to the nearest cell that is neither water nor sky."""
    from scipy.ndimage import distance_transform_edt
    water_idx, sky_idx = int(RegionType.WATER), int(RegionType.SKY)
    land = (region_map != water_idx) & (region_map != sky_idx)
    land &= ~cells
    if not land.any():
        region_map[cells] = int(RegionType.TERRAIN)
        return region_map
    _, (iy, ix) = distance_transform_edt(~land, return_indices=True)
    region_map[cells] = region_map[iy[cells], ix[cells]]
    return region_map


def _fit_river(water, axis, half_deg, cell_m, grid_m):
    """The photographed river's channel: (x_left, x_right, slope, offset, width,
    spread, n_bands) or None when both banks are not in frame."""
    res = water.shape[0]
    centre_col = res // 2
    edges = []
    # Bands from past the nadir disc (the cells under a viewer on a bridge are
    # typed by the fill, not the photograph, and a band through them read the
    # disc's edge as a bank: Paris came out 87 m wide instead of 130) out to
    # where the photographed banks still lie inside the grid.
    for z0 in np.arange(16.0, min(160.0, grid_m * 0.4), 4.0):
        r0, r1 = int((z0 + grid_m / 2.0) / cell_m), int((z0 + 4.0 + grid_m / 2.0) / cell_m)
        band = water[r0:r1].mean(axis=0) >= 0.5
        cols = np.flatnonzero(band)
        if cols.size == 0:
            continue
        c = cols[np.argmin(np.abs(cols - centre_col))]
        l = c
        while l > 0 and band[l - 1]:
            l -= 1
        r = c
        while r < res - 1 and band[r + 1]:
            r += 1
        if l == 0 or r == res - 1:
            continue
        xl, xr = axis[l], axis[r]
        zc = z0 + 2.0
        if abs(np.degrees(np.arctan2(xl, zc))) > half_deg + 15.0 or abs(np.degrees(np.arctan2(xr, zc))) > half_deg + 15.0:
            continue
        edges.append((zc, xl, xr))
    if len(edges) < 3:
        return None
    e = np.asarray(edges)
    # Far bands where the photographed river is broken up (removed boats, the
    # bridge, the far bank's reflection) read a sliver of it as the whole: keep
    # the bands that are at least most of the widest one.
    widths = e[:, 2] - e[:, 1]
    e = e[widths >= 0.6 * widths.max()]
    if len(e) < 3:
        return None
    zs, xls, xrs = e[:, 0], e[:, 1], e[:, 2]
    width = float(np.median(xrs - xls))
    mid = 0.5 * (xls + xrs)
    slope = float(np.clip(np.polyfit(zs, mid, 1)[0], -0.15, 0.15)) if len(zs) >= 3 else 0.0
    offset = float(np.median(mid - slope * zs))
    spread = float(max(np.std(xls - slope * zs), np.std(xrs - slope * zs)))
    return (offset - width / 2.0, offset + width / 2.0, slope, offset, width, spread, len(edges))


def _mode_filter_land(region_map, water, radius_cells, min_r_cells, R_cells):
    """Land cover in regions at least 2*radius wide: a mode filter at 1/4 resolution,
    water (and the near field) fixed."""
    from scipy.ndimage import uniform_filter
    res = region_map.shape[0]
    s = 4
    small = region_map[::s, ::s]
    k = max(3, int(round(2 * radius_cells / s)) | 1)
    types = [t for t in np.unique(small) if t != int(RegionType.WATER)]
    if len(types) <= 1:
        return region_map
    votes = np.stack([uniform_filter((small == t).astype(np.float32), size=k, mode="nearest") for t in types])
    mode = np.asarray(types, dtype=np.uint8)[np.argmax(votes, axis=0)]
    mode_full = np.repeat(np.repeat(mode, s, axis=0), s, axis=1)[:res, :res]
    out = region_map.copy()
    apply = ~water & (R_cells > min_r_cells)
    out[apply] = mode_full[apply]
    return out


def build_plan(
    region_map: np.ndarray,
    elevation: Optional[np.ndarray],
    type_idx_map: np.ndarray,
    depth: np.ndarray,
    cone_hfov_deg: float,
    camera_height: float,
    grid_m: float,
    cfg: PlanConfig,
    log: Callable[[str], None],
) -> tuple[np.ndarray, Optional[np.ndarray], TerrainPlan]:
    """Rebuild the region map's water and land cover by rule. Returns the new
    region map, the elevation (water cells on their plane) and the plan."""
    from scipy.ndimage import binary_fill_holes, label as _label

    water_idx = int(RegionType.WATER)
    res = region_map.shape[0]
    cell_m = grid_m / res
    half = grid_m / 2.0
    axis = (np.arange(res) + 0.5) * cell_m - half
    X = axis[None, :]
    Z = axis[:, None]
    R = np.hypot(X, Z)
    AZ = np.degrees(np.arctan2(X, Z))
    half_fov = cone_hfov_deg / 2.0
    in_cone = np.abs(AZ) <= half_fov

    H, W = type_idx_map.shape
    if depth.shape != (H, W):
        from scipy.ndimage import zoom as _zoom
        depth = _zoom(depth.astype(np.float32), (H / depth.shape[0], W / depth.shape[1]), order=1)
    az, col_in_cone, sea_col, r_shore, r_land, occ, bin_m = _column_shore_stats(type_idx_map, depth, cone_hfov_deg, half)
    col_of = ((AZ / 360.0 + 0.5) * W).astype(np.int64) % W

    photographed_water = (region_map == water_idx) & in_cone
    out = region_map.copy()
    new_elev = elevation.copy() if elevation is not None else None
    notes: list[str] = []
    plan = TerrainPlan(kind="none", water_y=None, water=np.zeros_like(photographed_water), hfov_deg=float(cone_hfov_deg))

    # Shore points for the drawing: nearest water per column, in world xz.
    sc = np.flatnonzero(col_in_cone & np.isfinite(r_shore))
    if sc.size:
        a = np.radians(az[sc])
        plan.shore_points = np.stack([r_shore[sc] * np.sin(a), r_shore[sc] * np.cos(a)], axis=1)

    # ── classify ──────────────────────────────────────────────────────────────
    min_cols = int(cfg.open_water_min_deg / 360.0 * W)
    river = None
    if sea_col.sum() >= min_cols:
        plan.kind = "sea"
    elif photographed_water.any():
        river = _fit_river(region_map == water_idx, axis, half_fov + 3.0, cell_m, grid_m)
        if river is not None and river[4] >= 5.0 and river[5] <= 0.25 * river[4]:
            plan.kind = "river"
        else:
            if river is not None:
                notes.append(f"banks at x = {river[0]:.0f} / {river[1]:.0f} m vary by {river[5]:.0f} m: not a straight channel")
            river = None
            plan.kind = "lake"
    else:
        plan.kind = "none"

    planned = photographed_water.copy()

    # ── sea ───────────────────────────────────────────────────────────────────
    if plan.kind == "sea":
        y_water = -float(camera_height)
        plan.water_y = y_water
        shore_of = r_shore[col_of]
        # Beyond the shore, in every column that looks out at the sea.
        bin_of = np.minimum((R / bin_m).astype(np.int64), occ.shape[1] - 1)
        occupied = occ[col_of, bin_of]
        # Inside the cone the photograph is the evidence: beyond the shore every
        # range that holds no photographed land is sea, whatever the height map
        # guessed there (it is a guess: the water pixels gave it no samples).
        sea_cells = in_cone & sea_col[col_of] & np.isfinite(shore_of) & (R > shore_of) & ~occupied
        # Beyond the land that blocks the horizon: a stack or headland standing in
        # the sea is finite, and the sea goes on behind it.
        blocked = in_cone & ~sea_col[col_of] & np.isfinite(shore_of)
        behind = blocked & (R > shore_of + 2.0) & ~occupied
        n_behind = int(behind.sum())
        planned |= sea_cells | behind
        notes.append(f"sea over {sea_col.sum() * 360.0 / W:.0f} of the cone's {cone_hfov_deg:.0f} deg")
        if n_behind:
            notes.append(f"{n_behind} cells behind land standing in the sea")
        # Each edge of the photograph that looks out over open water: the coast
        # continues at the direction it has there, the sea on the side it was.
        edge_deg = float(cfg.edge_deg)
        for sign in (-1.0, 1.0):
            edge_az = sign * half_fov
            band_cols = col_in_cone & (np.abs(az - edge_az) <= edge_deg)
            open_frac = float(sea_col[band_cols].mean()) if band_cols.any() else 0.0
            pts_cols = band_cols & np.isfinite(r_shore) & (r_shore < half)
            if open_frac < 0.5 or pts_cols.sum() < 4:
                notes.append(f"{'left' if sign < 0 else 'right'} edge ends on land ({open_frac:.0%} open): not extended")
                continue
            # The shore's direction over a few degrees of arc is too short to fit,
            # so the coast is carried round as a shore distance that recedes
            # linearly to twice its value at the end of the band: a coast of
            # unknown heading, erring towards land near the viewer.
            r_edge = float(np.nanmedian(r_shore[pts_cols]))
            rel = sign * (AZ - edge_az)
            band = (rel > 0.0) & (rel <= float(cfg.extend_deg))
            r_coast = max(5.0, r_edge) * (1.0 + rel / float(cfg.extend_deg))
            ext = band & (R > r_coast)
            planned |= ext
            arc = []
            for rel_d in np.linspace(0.0, float(cfg.extend_deg), 24):
                a = np.radians(edge_az + sign * rel_d)
                rr = max(5.0, r_edge) * (1.0 + rel_d / float(cfg.extend_deg))
                arc.append((float(rr * np.sin(a)), float(rr * np.cos(a))))
            plan.coast_lines.append(arc)
            notes.append(f"{'left' if sign < 0 else 'right'} edge open ({open_frac:.0%}): sea carried "
                         f"{cfg.extend_deg:.0f} deg past it from {r_edge:.0f} m out, {int(ext.sum())} cells")

    # ── river ─────────────────────────────────────────────────────────────────
    elif plan.kind == "river":
        xl, xr, slope, offset, width, spread, nb = river
        lateral = X - (offset + slope * Z)
        channel = np.abs(lateral) <= width / 2.0
        water_cells = region_map == water_idx
        ring = water_cells & (R >= 12.0) & (R < 60.0) & channel
        water_y = None
        bank_y = None
        if elevation is not None:
            wv = elevation[ring]
            wv = wv[np.isfinite(wv)]
            if wv.size >= 50:
                water_y = float(np.median(wv))
            bank_cells = in_cone & ~water_cells & (Z > 0) & ((np.abs(X - xl) <= 10.0) | (np.abs(X - xr) <= 10.0))
            bv = elevation[bank_cells]
            bv = bv[np.isfinite(bv)]
            if bv.size >= 50:
                bank_y = float(np.median(bv))
        plan.water_y = water_y
        planned = channel.copy()
        # Photographed water outside the channel (a basin, a backwater) stays.
        planned |= photographed_water
        plan.channel = {"x_left": xl, "x_right": xr, "slope": slope, "offset": offset, "width": width,
                        "water_y": water_y, "bank_y": bank_y}
        notes.append(f"river {width:.0f} m wide, banks at x = {xl:.0f} / {xr:.0f} m, axis "
                     f"{np.degrees(np.arctan(slope)):+.0f} deg, {nb} range bands within {spread:.0f} m")

    # ── lake / none ───────────────────────────────────────────────────────────
    elif plan.kind == "lake":
        wv = elevation[photographed_water] if elevation is not None else np.zeros(0)
        wv = wv[np.isfinite(wv)]
        plan.water_y = float(np.median(wv)) if wv.size else None
        notes.append("lake: water bounded on every visible side, kept to the photographed extent")
    else:
        notes.append("no water in the photograph: none in the scene")

    # ── water only as a continuation of photographed water ────────────────────
    if planned.any() and cfg.land_smooth_m > 0:
        # Per-column rules leave the water's outline ragged at the cell scale;
        # a body of water has a shoreline. Opened and closed at the land-cover
        # scale (at 1/4 resolution), the photographed water itself kept.
        from scipy.ndimage import binary_closing, binary_opening
        s4 = 4
        rad = max(1, int(round(cfg.land_smooth_m / cell_m / s4)))
        yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
        disc = (xx * xx + yy * yy) <= rad * rad
        small = planned[::s4, ::s4]
        small = binary_opening(binary_closing(small, structure=disc), structure=disc)
        planned = np.repeat(np.repeat(small, s4, axis=0), s4, axis=1)[:res, :res] | photographed_water
    if planned.any():
        planned = binary_fill_holes(planned) if cfg.hole_m2 <= 0 else planned
        if cfg.hole_m2 > 0:
            holes = binary_fill_holes(planned) & ~planned
            if holes.any():
                lab, n = _label(holes)
                counts = np.bincount(lab.ravel())
                small = np.flatnonzero(counts * cell_m * cell_m < cfg.hole_m2)
                small = small[small > 0]
                if small.size:
                    planned |= np.isin(lab, small)
        lab, n = _label(planned)
        seeded = np.unique(lab[photographed_water])
        seeded = seeded[seeded > 0]
        if seeded.size and n > seeded.size:
            planned &= np.isin(lab, seeded)
    dropped = (region_map == water_idx) & ~planned
    added = planned & (region_map != water_idx)
    if dropped.any():
        out = _nearest_land_fill(out, dropped)
    out[planned] = water_idx
    if new_elev is not None and plan.water_y is not None:
        # A body of water is one plane: every planned cell lies on it (the
        # outpainted river behind the viewer was measured 1.5 m lower than the
        # photographed one, and the water mesh took the median of the two).
        new_elev[planned] = plan.water_y
        # And what was water and is now land is a bank, not a flat at water level.
        if dropped.any():
            bank = plan.channel["bank_y"] if (plan.channel and plan.channel.get("bank_y") is not None) else None
            if bank is None:
                bank = plan.water_y + float(cfg.bank_tolerance_m)
            new_elev[dropped] = np.maximum(np.nan_to_num(new_elev[dropped], nan=bank), bank)
    plan.water = planned
    notes.append(f"{int(added.sum())} cells of water added, {int(dropped.sum())} of outpainted water made land")

    # ── land cover in regions, not slivers ────────────────────────────────────
    if cfg.land_smooth_m > 0:
        before = out.copy()
        out = _mode_filter_land(out, planned, cfg.land_smooth_m / cell_m, cfg.smooth_min_distance_m / cell_m, R / cell_m)
        notes.append(f"land cover smoothed to {cfg.land_smooth_m:.0f} m regions beyond {cfg.smooth_min_distance_m:.0f} m: "
                     f"{int((out != before).sum())} cells changed")
    plan.notes = notes
    log(f"Terrain plan ({plan.kind}): " + "; ".join(notes))
    return out, new_elev, plan


def draw_plan(region_map: np.ndarray, plan: TerrainPlan, grid_m: float, size: int = 1024) -> PILImage.Image:
    """The plan as a map: region colours, the photographed cone, shore points and
    the coast lines carried past its edges, the river channel."""
    res = region_map.shape[0]
    s = max(1, res // size)
    rgb = colorize_region_type_map(region_map[::s, ::s])
    img = PILImage.fromarray(rgb).resize((size, size), PILImage.NEAREST)
    d = ImageDraw.Draw(img)
    ppm = size / grid_m

    def to_px(x, z):
        return (size / 2.0 + x * ppm, size / 2.0 + z * ppm)   # raster row = +z

    c = to_px(0.0, 0.0)
    L = grid_m
    for sign in (-1.0, 1.0):
        a = np.radians(sign * plan.hfov_deg / 2.0)
        d.line([c, to_px(L * np.sin(a), L * np.cos(a))], fill=(255, 255, 255), width=2)
    for line in plan.coast_lines:
        d.line([to_px(x, z) for x, z in line], fill=(255, 40, 40), width=2)
    if plan.channel:
        ch = plan.channel
        for off in (ch["x_left"], ch["x_right"]):
            zs = np.array([-grid_m / 2.0, grid_m / 2.0])
            xs = off + ch["slope"] * zs
            d.line([to_px(xs[0], zs[0]), to_px(xs[1], zs[1])], fill=(0, 255, 255), width=2)
    for x, z in plan.shore_points[::max(1, len(plan.shore_points) // 400)]:
        px, pz = to_px(x, z)
        d.ellipse([px - 2, pz - 2, px + 2, pz + 2], fill=(255, 230, 0))
    d.ellipse([c[0] - 4, c[1] - 4, c[0] + 4, c[1] + 4], fill=(255, 255, 255), outline=(0, 0, 0))
    y = 6
    for line in [f"plan: {plan.kind}" + (f", water at {plan.water_y:.1f} m" if plan.water_y is not None else ""),
                 f"grid {grid_m:.0f} m, cone {plan.hfov_deg:.0f} deg"] + plan.notes:
        d.text((8, y + 1), line, fill=(0, 0, 0))
        d.text((7, y), line, fill=(255, 255, 255))
        y += 13
    return img
