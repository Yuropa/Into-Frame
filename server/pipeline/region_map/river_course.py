"""One river through the scene: the water region rebuilt as a continuous channel from
the reach the photograph shows (server/pipeline/region_map/river_course.py).

The outpainted panorama does not reason about geography, and projected onto the
ground its water is a starburst: Paris's Seine grew arms either side of the bridge,
a bay behind the viewer and a split at the back, a river looping back on itself.
Inside the photographed cone the water is evidence; outside it, it is a weak hint
of where the river goes.

The rule, on a top-down grid centred on the camera (rows +z, columns +x):

  * The photographed reach is the water inside the cone. Where the cone (or the
    grid's edge) cuts it, the river leaves the photograph: an exit, with the
    direction the reach's medial axis has there and the width its banks have
    (measured against photographed land only).
  * From each exit the course is marched outward in steps, turning no tighter than
    a minimum radius and steered by the outpainted water ahead (so a generated bend
    is followed if it is gentle; a side arm or a bay is not, since the march only
    looks ahead), until it leaves the grid. The channel is the march swept at the
    photographed width.
  * The new water is the photographed water itself (islands, backwaters and all)
    plus the channel outside the cone. Everything else the outpainter called water
    is land.

It is conservative: a sea (water meeting the sky, or open along a long stretch of
the grid's edge), a lake or pond inside the terrain, water open on one side only,
water the photograph does not show as a channel with both banks, a march that
cannot reach the grid's edge, or a scene plan that says lake/sea -- each leaves the
water as it is and says why.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RiverCourseConfig:
    work_cell_m: float = 1.0          # the rule works at this resolution, applied at the map's
    min_radius_m: float = 120.0       # the course bends no tighter than this outside the photograph
    min_radius_widths: float = 2.0    # ...nor tighter than this many widths
    step_m: float = 2.0
    lookahead_widths: float = 2.0     # the outpainted water this far ahead steers the march
    straight_bias: float = 0.15       # preference for going straight over a slightly wetter turn
    min_width_m: float = 4.0
    min_exit_widths: float = 0.4      # a cut of the reach narrower than this x width is not an exit
    min_bank_fraction: float = 0.2    # of the reach's outline that is photographed land
    max_width_cv: float = 0.6         # spread of the reach's local widths over their median
    max_open_perimeter: float = 0.35  # water open along more of the grid's edge than this is a coast
    min_reach_m2: float = 400.0


@dataclass
class RiverCourse:
    decided: str                       # "river" | "untouched"
    reason: str
    water: np.ndarray                  # the new (or unchanged) water, at the input resolution
    width_m: float = 0.0
    exits: list = field(default_factory=list)      # [(x, z, tangent_x, tangent_z)] world metres
    courses: list = field(default_factory=list)    # polylines [[(x, z), ...]] world metres
    ends: list = field(default_factory=list)       # where each course leaves the grid


def _open_ends(water: np.ndarray, margin: int) -> list[np.ndarray]:
    from scipy import ndimage
    band = np.zeros_like(water)
    band[:margin, :] = band[-margin:, :] = True
    band[:, :margin] = band[:, -margin:] = True
    lab, n = ndimage.label(water & band, structure=np.ones((3, 3)))
    return [np.argwhere(lab == k) for k in range(1, n + 1)]


def _edge_span(cells: np.ndarray, shape) -> float:
    """How much of the grid's perimeter a stretch of edge water covers, in cells."""
    nz, nx = shape
    r, c = cells[:, 0], cells[:, 1]
    span = 0.0
    for on, coord in ((r <= 3, c), (r >= nz - 4, c), (c <= 3, r), (c >= nx - 4, r)):
        if on.any():
            span += float(np.ptp(coord[on]) + 1)
    return span


def _downsample(mask: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return mask.copy()
    n = (mask.shape[0] // f) * f
    m = mask[:n, :n].reshape(n // f, f, n // f, f)
    return m.mean(axis=(1, 3)) >= 0.5


def rebuild_river(water: np.ndarray, in_cone: np.ndarray, cell_m: float, hint_kind: Optional[str] = None,
                  cfg: Optional[RiverCourseConfig] = None, log=None) -> RiverCourse:
    """The water region rebuilt as one river course, or left alone (see the module)."""
    from scipy import ndimage
    from skimage.morphology import skeletonize

    cfg = cfg or RiverCourseConfig()
    log = log or (lambda s: None)
    water = np.asarray(water, dtype=bool)
    in_cone = np.asarray(in_cone, dtype=bool)

    def untouched(reason: str) -> RiverCourse:
        log(f"River course: water left as it is -- {reason}")
        return RiverCourse("untouched", reason, water.copy())

    if hint_kind in ("sea", "ocean", "lake", "pond", "reservoir"):
        return untouched(f"the scene plan reads the water as {hint_kind}")
    if not water.any():
        return untouched("no water")

    f = max(1, int(round(cfg.work_cell_m / cell_m)))
    W = _downsample(water, f)
    C = _downsample(in_cone, f)
    cm = cell_m * f
    n = W.shape[0]
    half = n * cm / 2.0
    to_world = lambda r, c: ((c + 0.5) * cm - half, (r + 0.5) * cm - half)

    # ── is it a river at all ──────────────────────────────────────────────────
    margin = max(2, int(round(6.0 / cm)))
    ends = [e for e in _open_ends(W, margin) if len(e) >= 3]
    open_span = sum(_edge_span(e, W.shape) for e in ends) / (4.0 * n)
    if not ends and hint_kind not in ("river", "stream", "canal"):
        return untouched("the water does not reach the terrain's edge (a lake or a pond)")
    if open_span > cfg.max_open_perimeter:
        return untouched(f"the water is open along {open_span:.0%} of the terrain's edge (a coast, not a river)")

    reach = W & C
    lab, nl = ndimage.label(reach)
    if nl == 0:
        return untouched("the photograph shows no water")
    sizes = ndimage.sum(reach, lab, range(1, nl + 1))
    reach = np.isin(lab, 1 + np.flatnonzero(sizes * cm * cm >= cfg.min_reach_m2))
    if not reach.any():
        return untouched("the photographed water is too small to be a river")

    # Banks: the reach's outline against photographed land. Cuts: its outline
    # against the cone's edge (water continuing outside it) or the grid's edge.
    land_ph = ~W & C
    outline = reach & ~ndimage.binary_erosion(reach, structure=np.ones((3, 3)), border_value=0)
    near_land = ndimage.binary_dilation(land_ph, structure=np.ones((3, 3)))
    bank = outline & near_land
    # The grid's edge counts within the open-end margin: the plan's smoothing rounds a
    # river off a few metres short of it (Paris: 3 m), and that rim is not a bank.
    border = np.zeros_like(W)
    border[:margin, :] = border[-margin:, :] = True; border[:, :margin] = border[:, -margin:] = True
    bank = bank & ~border
    cut = outline & (border | (~near_land & ndimage.binary_dilation(~C, structure=np.ones((3, 3)))))
    bank_frac = float(bank.sum()) / max(int(outline.sum()), 1)
    if bank_frac < cfg.min_bank_fraction:
        return untouched(f"the photograph shows banks on {bank_frac:.0%} of its water's outline (not a channel)")

    # Width against photographed land: the cone's own cut is not a bank. Read on the
    # medial axis where both banks are nearer than the cut (the reach's own centre
    # line there); the truncated wedge by the apex has a medial axis of the cut's.
    # Islands the photograph shows inside the reach are not banks.
    islands = ndimage.binary_fill_holes(reach) & ~reach
    d_land = ndimage.distance_transform_edt(~(land_ph & ~islands)) * cm
    skel = skeletonize(ndimage.binary_fill_holes(reach))   # one axis past an island, not a loop round it
    d_cut = ndimage.distance_transform_edt(~cut) * cm if cut.any() else np.full(W.shape, np.inf)
    sk = skel & (d_land <= d_cut)
    if sk.sum() < 3:
        sk = skel
    # Bank to bank: from every photographed bank cell, the distance to the nearest
    # cell of another stretch of bank (the far bank; the cone's cuts and the grid's
    # edge break the outline into stretches). Islands are filled first: the river's
    # width runs from bank to bank, not to the island.
    filled = ndimage.binary_fill_holes(reach)
    f_outline = filled & ~ndimage.binary_erosion(filled, structure=np.ones((3, 3)), border_value=0)
    f_bank = f_outline & near_land
    blab, nb = ndimage.label(f_bank, structure=np.ones((3, 3)))
    widths = np.zeros(0)
    if nb >= 2:
        sizes_b = ndimage.sum(f_bank, blab, range(1, nb + 1))
        keep_b = 1 + np.flatnonzero(sizes_b * cm >= max(2.0, 0.25 * cfg.min_width_m))
        parts = []
        for k in keep_b:
            other = f_bank & (blab != k)
            if not other.any():
                continue
            d_other = ndimage.distance_transform_edt(~other) * cm
            parts.append(d_other[blab == k])
        if parts:
            widths = np.concatenate(parts)
    if widths.size < 3:
        widths = 2.0 * d_land[sk]
    widths = widths[np.isfinite(widths)]
    if widths.size < 3:
        return untouched("the photographed water has no measurable width")
    width = float(np.median(widths))
    wide_end = max((_edge_span(e, W.shape) * cm for e in ends), default=0.0)
    if wide_end > max(4.0 * width, 100.0):
        return untouched(f"the water leaves the terrain along {wide_end:.0f} m of its edge, {wide_end / width:.0f} times "
                         f"the photographed width ({width:.0f} m): a coast, not a river")
    cv = float(np.std(widths) / max(width, 1e-6))
    if width < cfg.min_width_m:
        return untouched(f"the photographed water is {width:.1f} m wide (a ditch, not a river)")
    if cv > cfg.max_width_cv:
        return untouched(f"the photographed water's width varies by {cv:.0%} (a basin, not a channel)")

    # ── exits ─────────────────────────────────────────────────────────────────
    rlab = ndimage.label(reach)[0]
    out_lab = ndimage.label(W & ~C)[0]
    d_cone = ndimage.distance_transform_edt(~C) * cm
    # A cut cell's own reach piece: the nearest reach cell's label.
    _, (ry, rx) = ndimage.distance_transform_edt(rlab == 0, return_indices=True)
    clab, nc = ndimage.label(ndimage.binary_dilation(cut, structure=np.ones((3, 3))) & reach)
    exits = []
    sk_pts = np.argwhere(skel)
    for k in range(1, nc + 1):
        cells = np.argwhere(clab == k)
        span = float(np.ptp(cells, axis=0).max() + 1) * cm
        if span < cfg.min_exit_widths * width:
            continue
        p = cells.mean(axis=0)
        # The medial axis leading to the cut: skeleton points within reach of it,
        # their principal direction, pointed outward.
        near = np.zeros((0, 2))
        banked = np.argwhere(sk & (rlab == rlab[tuple(np.clip(np.rint(p).astype(int), 0, n - 1))]))
        if len(banked):
            # The centre line where both banks are nearer than the cut: near the cut
            # the medial axis runs off to the corners where the cone meets a bank.
            dist = np.hypot(*(banked - p).T) * cm
            near = banked[dist <= max(1.0 * width, 20.0)]
        if len(near) < 3 and len(sk_pts):
            dist = np.hypot(*(sk_pts - p).T) * cm
            for radius in (max(0.75 * width, 8.0), max(1.5 * width, 20.0)):
                near = sk_pts[dist <= radius]
                if len(near) >= 3:
                    break
        q = near.mean(axis=0) if len(near) else np.array([n / 2.0, n / 2.0])
        if len(near) >= 3:
            t = np.linalg.svd(near - q, full_matrices=False)[2][0]
        else:
            t = p - q
        t = t / max(np.linalg.norm(t), 1e-9)
        # Outward: the way that leaves the photograph.
        probe = lambda v: p + v * (0.5 * width / cm + 3.0)
        def outside(v):
            r_, c_ = np.rint(probe(v)).astype(int)
            return not (0 <= r_ < n and 0 <= c_ < n) or not C[r_, c_]
        if outside(-t) and not outside(t):
            t = -t
        elif outside(t) == outside(-t) and np.dot(t, p - q) < 0:
            t = -t
        # Water that only bulges out of the cone and back (the outside of a bend the
        # cone's edge clips) is not the river leaving the photograph.
        comp = np.unique(out_lab[ndimage.binary_dilation(clab == k, structure=np.ones((3, 3)), iterations=2) & (out_lab > 0)])
        reach_out = max((float(d_cone[out_lab == c].max()) for c in comp), default=0.0)
        if comp.size and reach_out < 0.5 * width and not border[cells[:, 0], cells[:, 1]].any():
            continue
        exits.append((p, t, int(rlab[ry[tuple(np.rint(p).astype(int))], rx[tuple(np.rint(p).astype(int))]])))
    if len(exits) < 2 and not (len(exits) == 1 and any(_edge_span(e, W.shape) > 0 for e in ends)):
        return untouched(f"the photographed water leaves the photograph at {len(exits)} place(s), not as a river does")

    # ── march each exit to the grid's edge ────────────────────────────────────
    r_min = max(cfg.min_radius_m, cfg.min_radius_widths * width)
    step = cfg.step_m / cm
    max_turn = (cfg.step_m / r_min)
    look = cfg.lookahead_widths * width / cm
    halfw = 0.5 * width / cm
    courses, finals = [], []
    # Pieces of the photographed reach that the cone separates (a bend dipping out of
    # it and back) are joined through the water outside it when that way is short:
    # the photograph shows the river on both sides of the gap.
    bridged = set()
    if len({e[2] for e in exits}) > 1:
        from skimage.graph import MCP_Geometric
        cost = np.where(W, 1.0, 1e6)
        limit = max(3.0 * width, 60.0) / cm
        pairs = []
        for i, (pi, _, li) in enumerate(exits):
            mcp = MCP_Geometric(cost)
            acc, _ = mcp.find_costs([tuple(np.rint(pi).astype(int))])
            for j, (pj, _, lj) in enumerate(exits):
                if j <= i or lj == li:
                    continue
                c_ = acc[tuple(np.rint(pj).astype(int))]
                if c_ <= limit:
                    pairs.append((c_, i, j, mcp))
        for c_, i, j, mcp in sorted(pairs, key=lambda x: x[0]):
            if i in bridged or j in bridged:
                continue
            mcp2 = MCP_Geometric(cost)
            mcp2.find_costs([tuple(np.rint(exits[i][0]).astype(int))])
            route = np.asarray(mcp2.traceback(tuple(np.rint(exits[j][0]).astype(int))), dtype=float)
            courses.append(route)
            bridged |= {i, j}
    for k_exit, (p0, t0, _piece) in enumerate(exits):
        if k_exit in bridged:
            continue
        p = p0.astype(float).copy()
        th = math.atan2(t0[1], t0[0])     # angle in (row, col) space
        path = [p.copy()]
        for _ in range(int(4 * n / max(step, 1e-6))):
            if not (1 <= p[0] < n - 1 and 1 <= p[1] < n - 1):
                break
            # Steer for the middle of the water a width ahead, within a width either
            # side of the course (a side arm or a bay beyond that does not pull it),
            # turning no tighter than the minimum radius; straight on over land.
            d = np.array([math.cos(th), math.sin(th)])
            nrm = np.array([-d[1], d[0]])
            ahead = p + max(look * 0.5, halfw, 3.0) * d
            offs = np.arange(-int(2 * halfw) - 1, int(2 * halfw) + 2, dtype=float)
            pts = ahead[None, :] + offs[:, None] * nrm[None, :]
            rr = np.rint(pts[:, 0]).astype(int); cc = np.rint(pts[:, 1]).astype(int)
            inside = (rr >= 0) & (rr < n) & (cc >= 0) & (cc < n)
            wet = np.zeros(len(offs), dtype=bool)
            wet[inside] = W[rr[inside], cc[inside]]
            if wet.any():
                lab1, n1 = ndimage.label(wet)
                centre = len(offs) // 2
                k = lab1[centre] if wet[centre] else lab1[np.flatnonzero(wet)[np.argmin(np.abs(np.flatnonzero(wet) - centre))]]
                run = offs[lab1 == k]
                target = ahead + 0.5 * (run.min() + run.max()) * nrm
                want = math.atan2(target[1] - p[1], target[0] - p[0])
                dth = (want - th + math.pi) % (2 * math.pi) - math.pi
                th += max(-max_turn, min(max_turn, dth))
            p = p + step * np.array([math.cos(th), math.sin(th)])
            path.append(p.copy())
        else:
            return untouched("a course from the photographed reach does not reach the terrain's edge")
        courses.append(np.asarray(path))
        finals.append(np.clip(p, 0, n - 1))

    # The outpainted water must agree that the river leaves the grid where each course
    # does, at separate places: a lake that reaches one edge, or water the cone cuts
    # at its sides, marches out over land.
    agree = set()
    tol = max(2.0 * width, 40.0) / cm
    for q in finals:
        hit = None
        for k, e in enumerate(ends):
            if float(np.min(np.hypot(*(e - q).T))) <= tol:
                hit = k
                break
        if hit is None:
            if hint_kind in ("river", "stream", "canal"):
                log(f"River course: the surround closes the river at ({to_world(*q)[0]:.0f}, {to_world(*q)[1]:.0f}) m; "
                    f"the scene plan reads it as a {hint_kind}, so it runs on")
                continue
            return untouched(f"a course leaves the terrain at ({to_world(*q)[0]:.0f}, {to_world(*q)[1]:.0f}) m "
                             "where the surround shows no water leaving it")
        agree.add(hit)
    need = 1 if hint_kind in ("river", "stream", "canal") else 2
    if len(agree) < need:
        return untouched(f"the courses leave the terrain through {len(agree)} stretch(es) of edge water, not {need}")

    # ── the channel ───────────────────────────────────────────────────────────
    line = np.zeros((n, n), dtype=bool)
    for path in courses:
        for a_, b_ in zip(path[:-1], path[1:]):
            m = max(2, int(np.ceil(np.hypot(*(b_ - a_)) * 2)))
            seg = a_[None, :] + np.linspace(0.0, 1.0, m)[:, None] * (b_ - a_)[None, :]
            r_, c_ = np.rint(seg).astype(int).T
            ok = (r_ >= 0) & (r_ < n) & (c_ >= 0) & (c_ < n)
            line[r_[ok], c_[ok]] = True
    channel = ndimage.distance_transform_edt(~line) <= halfw
    # And the water beside the photographed reach's own centre line: where one of its
    # banks lies just outside the cone, the river there is the photograph's (kept as
    # the surround has it, never widened onto its land).
    centre = sk & (d_land >= 0.35 * width)
    if centre.any():
        channel |= (ndimage.distance_transform_edt(~centre) <= halfw) & W
    # Back to the map's resolution: the photographed water exactly as it is, the
    # channel outside the cone.
    full_n = water.shape[0]
    up = np.repeat(np.repeat(channel, f, axis=0), f, axis=1)
    if up.shape[0] < full_n:
        up = np.pad(up, ((0, full_n - up.shape[0]), (0, full_n - up.shape[1])), mode="edge")
    up = up[:full_n, :full_n]
    new_water = (water & in_cone) | (up & ~in_cone)
    # The photographed water's own continuation just past the cone's edge (the
    # banks run on) where the swept channel is narrower than the reach at the cut.
    new_water |= water & ~in_cone & ndimage.binary_dilation(in_cone & water, iterations=max(1, int(round(2.0 / cell_m))))
    # One body: drop what does not join the photographed water.
    lab2, n2 = ndimage.label(new_water)
    seeds = np.unique(lab2[water & in_cone])
    seeds = seeds[seeds > 0]
    if seeds.size:
        new_water = np.isin(lab2, seeds)

    ex = [(*to_world(*p), float(t[1]), float(t[0])) for p, t, _ in exits]
    cs = [[to_world(r, c) for r, c in path[:: max(1, len(path) // 60)]] + [to_world(*path[-1])] for path in courses]
    fe = [to_world(*p) for p in finals]
    log(f"River course: {len(exits)} exit(s) from the photographed reach, {width:.0f} m wide "
        f"(banks on {bank_frac:.0%} of its outline, widths within {cv:.0%}); courses to the edge at "
        + ", ".join(f"({x:.0f}, {z:.0f})" for x, z in fe)
        + f"; {int((water & ~new_water).sum() * cell_m * cell_m)} m^2 of outpainted water made land, "
          f"{int((new_water & ~water).sum() * cell_m * cell_m)} m^2 of channel added")
    return RiverCourse("river", f"{len(exits)} exits", new_water, width, ex, cs, fe)
