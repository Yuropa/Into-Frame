"""Population Fit: a statistical population model per class, fitted to the photograph
(docs/agentic-scene-graph-plan.md §5), and an optional draw from it that replaces the
hand-painted instances of that class.

Model. For a class (flowers first) the detections inside the photographed cone are a
point pattern; its intensity is log-linear in terrain features at each cell:

    lambda(x) = exp(b0 + sum_k b_k f_k(x))        f_k: forest edge, water edge, slope,
                                                   elevation, greenness, bloom colour

fitted by ridge-penalised Poisson regression over the cells of the observation window
(inside the cone, nearer than ~1.2x the farthest exemplar, on paintable ground), with a
detection-effort offset e(d) = clip((d_ref / d)^gamma, ., 1): a flower of the exemplars'
median size spans fewer pixels farther away, so the detector sees a thinner pattern than
the meadow holds. The draw uses lambda without the offset, so the far meadow is as full as
the near one unless the terrain says otherwise.

Clustering is a Thomas process: parents at lambda / mu, mu offspring each, Gaussian radius
sigma. mu and sigma come from the exemplars' nearest-neighbour spacing against the Poisson
expectation, or from config.

Validation: the window is split left/right of the camera axis; the model fitted on one half
is scored on the other against a flat intensity, and the gain in log-likelihood per
held-out exemplar is reported (positive = the features explain where the flowers are).

Outputs: `population_models` (one record per class, read by the scene-graph exporter) and,
for classes whose override says enabled, the painted instances of that class are dropped
and a draw of `metadata_{i}` entries is appended in Distribution Synthesis's own format
(class, synthetic, world_position [x, 0, z], world_width, world_height, bucket) so Scene
Generation needs no change. Runs on the CPU in seconds.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pipeline.pipeline_context import ContextKey, PipelineContext
from pipeline.pipeline_stage import PipelineStage, PipelineStageConfiguration
from pipeline.panorama_segmentation.panorama_region_result import RegionType

MODELS_KEY = "population_models"
MARKER = "population_fit_complete"
PAINTABLE = (int(RegionType.TERRAIN), int(RegionType.GROUND), int(RegionType.VEGETATION))
_PLAN_WORDS = {"dense": 2.5, "common": 1.5, "patchy": 1.0, "scattered": 0.8, "sparse": 0.5, "rare": 0.3}


class PopulationFitConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        classes: Optional[list[str]] = None,
        class_overrides: Optional[dict[str, dict]] = None,
        min_exemplars: int = 8,
        cell_m: float = 1.0,
        fit_radius_factor: float = 1.2,
        ridge: float = 1.0,
        detect_min_px: float = 20.0,
        detect_gamma: float = 1.5,
        min_effort: float = 0.02,
        features: Optional[list[str]] = None,
        cluster_radius_m: float = 1.5,
        cluster_size: float = 6.0,
        estimate_clustering: bool = True,
        max_distance_m: float = 60.0,
        max_instances: int = 20000,
        size_jitter: float = 0.15,
        plan_density_words: bool = True,
        density_multiplier: float = 1.0,
        holdout: bool = True,
        max_intensity_m2: float = 30.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.classes = list(classes or ["flower"])
        self.class_overrides = {str(k): dict(v or {}) for k, v in (class_overrides or {}).items()}
        self.min_exemplars = int(min_exemplars)
        self.cell_m = float(cell_m)
        self.fit_radius_factor = float(fit_radius_factor)
        self.ridge = float(ridge)
        self.detect_min_px = float(detect_min_px)
        self.detect_gamma = float(detect_gamma)
        self.min_effort = float(min_effort)
        self.features = list(features or ["forest_edge", "water_edge", "slope", "elevation", "greenness", "bloom"])
        self.cluster_radius_m = float(cluster_radius_m)
        self.cluster_size = float(cluster_size)
        self.estimate_clustering = bool(estimate_clustering)
        self.max_distance_m = float(max_distance_m)
        self.max_instances = int(max_instances)
        self.size_jitter = float(size_jitter)
        self.plan_density_words = bool(plan_density_words)
        self.density_multiplier = float(density_multiplier)
        self.holdout = bool(holdout)
        self.max_intensity_m2 = float(max_intensity_m2)

    def for_class(self, cls: str) -> dict:
        o = self.class_overrides.get(cls, {})
        return {
            "enabled": bool(o.get("enabled", False)),
            "density_multiplier": float(o.get("density_multiplier", self.density_multiplier)),
            "cluster_radius_m": float(o["cluster_radius_m"]) if "cluster_radius_m" in o else None,
            "cluster_size": float(o["cluster_size"]) if "cluster_size" in o else None,
            "max_distance_m": float(o["max_distance_m"]) if "max_distance_m" in o else None,
            "seed": int(o["seed"]) if "seed" in o else None,
        }


# ── the fit ───────────────────────────────────────────────────────────────────

def fit_poisson(F: np.ndarray, y: np.ndarray, log_offset: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    """Ridge-penalised Poisson regression: F (n, k) standardised features, y counts,
    log_offset = log(area * effort). Returns (beta with intercept first, log-likelihood)."""
    from scipy.optimize import minimize
    n, k = F.shape
    X = np.hstack([np.ones((n, 1)), F])
    pen = np.r_[0.0, np.full(k, ridge)]

    def f(b):
        eta = X @ b + log_offset
        mu = np.exp(np.clip(eta, -30, 30))
        ll = float((y * eta - mu).sum())
        g = X.T @ (y - mu)
        return -(ll - 0.5 * float((pen * b * b).sum())), -(g - pen * b)

    b0 = np.zeros(k + 1)
    b0[0] = math.log(max(y.sum(), 1.0) / max(np.exp(log_offset).sum(), 1e-9))
    res = minimize(f, b0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    beta = res.x
    eta = X @ beta + log_offset
    ll = float((y * eta - np.exp(np.clip(eta, -30, 30))).sum())
    return beta, ll


def _null_loglik(y: np.ndarray, log_offset: np.ndarray) -> float:
    rate = max(y.sum(), 1e-9) / max(np.exp(log_offset).sum(), 1e-9)
    eta = math.log(rate) + log_offset
    return float((y * eta - np.exp(eta)).sum())


def _nn_spacing(points: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    if len(points) < 3:
        return float("nan")
    d, _ = cKDTree(points).query(points, k=2)
    return float(np.median(d[:, 1]))


class PopulationFitStage(PipelineStage):
    @classmethod
    def config_class(cls):
        return PopulationFitConfiguration

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(MARKER) is not None

    # ── inputs ────────────────────────────────────────────────────────────────
    def _grid(self, context: PipelineContext) -> Optional[dict]:
        cfg: PopulationFitConfiguration = self.config
        region = context.input_depth(ContextKey.REGION_MAP)
        params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}
        height = context.input_depth(ContextKey.HEIGHT_MAP)
        if region is None or height is None or not params:
            return None
        rm = np.asarray(region.depth)
        res = rm.shape[0]
        grid_m = float(params.get("grid_size_meters", 200.0))
        fine = grid_m / res
        f = max(1, int(round(cfg.cell_m / fine)))
        rm_c = rm[::f, ::f].astype(np.int64)
        hm_c = np.asarray(height.depth)[::f, ::f].astype(np.float32)
        n = rm_c.shape[0]
        cell = fine * f
        axis = (np.arange(n) + 0.5) * cell - grid_m / 2.0
        zz, xx = np.meshgrid(axis, axis, indexing="ij")
        out = {"rm": rm_c, "hm": hm_c, "cell": cell, "n": n, "f": f, "grid_m": grid_m, "res": res,
               "xx": xx.astype(np.float32), "zz": zz.astype(np.float32),
               "dist": np.hypot(xx, zz).astype(np.float32)}
        # the photographed cone, geometrically: the photo sits at the panorama's centre
        # (+Z forward), so a cell is in it when its bearing is within the horizontal FOV
        intr = context.input_object(ContextKey.INTRINSICS) or {}
        hfov = float(intr.get("fov", 60.0))
        bearing = np.degrees(np.arctan2(xx, zz))
        out["cone"] = np.abs(bearing) <= hfov / 2.0 + 2.0
        # appearance: the panorama pixel each cell observes (as Distribution Synthesis)
        u = context.input_depth(ContextKey.HEIGHT_MAP_PANO_U)
        v = context.input_depth(ContextKey.HEIGHT_MAP_PANO_V)
        sampled = context.input_depth(ContextKey.HEIGHT_MAP_REAL_SAMPLE_MASK)
        pano = context.input_panorama(ContextKey.PANORAMA)
        out["valid"] = np.zeros((n, n), bool)
        out["in_photo"] = np.zeros((n, n), bool)
        out["green"] = np.zeros((n, n), np.float32)
        out["bloom"] = np.zeros((n, n), bool)
        if u is not None and v is not None and sampled is not None and pano is not None:
            uu = np.asarray(u.depth)[::f, ::f]
            vv = np.asarray(v.depth)[::f, ::f]
            valid = np.asarray(sampled.depth)[::f, ::f].astype(bool)
            if uu.shape == (n, n):
                rgb = np.asarray(pano.rgb(), dtype=np.float32)
                H, W = rgb.shape[:2]
                row = np.zeros((n, n), np.intp)
                col = np.zeros((n, n), np.intp)
                row[valid] = np.clip(((1.0 - vv[valid]) * (H - 1)).astype(np.intp), 0, H - 1)
                col[valid] = np.clip((uu[valid] * (W - 1)).astype(np.intp), 0, W - 1)
                px = rgb[row, col]
                green = (2 * px[..., 1] - px[..., 0] - px[..., 2]) / 255.0
                sat = px.max(axis=-1) - px.min(axis=-1)
                green_dom = (px[..., 1] >= px[..., 0]) & (px[..., 1] >= px[..., 2])
                out["valid"] = valid
                out["green"] = np.where(valid, green, 0.0).astype(np.float32)
                out["bloom"] = valid & (sat >= 60.0) & ~green_dom
                try:
                    from util.captured_cone import captured_cone_from_context
                    cone = captured_cone_from_context(context, W, H)
                except Exception:
                    cone = None
                if cone is not None:
                    u0, u1 = (x / W for x in cone.u_range)
                    v0, v1 = (y / H for y in cone.v_range)
                    out["in_photo"] = valid & (uu >= u0) & (uu <= u1) & (1.0 - vv >= v0) & (1.0 - vv <= v1)
        return out

    def _features(self, g: dict, names: list[str]) -> dict[str, np.ndarray]:
        from pipeline.distribution_synthesis.features import feature_fields
        fields = feature_fields(g["rm"], g["cell"])
        gz, gx = np.gradient(g["hm"], g["cell"])
        slope = np.degrees(np.arctan(np.hypot(gx, gz))).astype(np.float32)
        avail = {
            "forest_edge": np.clip(fields.get("forest_edge"), -50, 50) if "forest_edge" in fields else None,
            "water_edge": np.clip(fields.get("water_edge"), -50, 50) if "water_edge" in fields else None,
            "road": np.clip(fields.get("road"), 0, 80) if "road" in fields else None,
            "trail": np.clip(fields.get("trail"), 0, 80) if "trail" in fields else None,
            "slope": np.clip(slope, 0, 60),
            "elevation": np.clip(g["hm"], -30, 30),
            "greenness": np.clip(g["green"], -0.1, 0.4),
            "bloom": g["bloom"].astype(np.float32),
        }
        return {k: np.asarray(avail[k], np.float32) for k in names if avail.get(k) is not None}

    # ── run ───────────────────────────────────────────────────────────────────
    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: PopulationFitConfiguration = self.config
        dist = context.input_object_distribution(ContextKey.OBJECT_DISTRIBUTION)
        g = self._grid(context)
        if dist is None or g is None:
            self.log_info("Population Fit: no object distribution or terrain grid; nothing to fit")
            context.add_object(MODELS_KEY, {})
            context.add_object(MARKER, True)
            return context
        feats = self._features(g, cfg.features)
        self.log_info(f"Population Fit: grid {g['n']}x{g['n']} at {g['cell']:.2f} m, features {list(feats)}")
        paintable = np.isin(g["rm"], PAINTABLE)
        plan_density = self._plan_density(context)
        px_per_rad = self._px_per_rad(context)

        models: dict[str, Any] = {}
        count = int(context.input_object(ContextKey.OBJECT_COUNT) or 0)
        next_idx = count
        for cls in cfg.classes:
            ex = self._exemplars(dist, cls)
            if ex is None or len(ex["points"]) < cfg.min_exemplars:
                self.log_info(f"  {cls}: {0 if ex is None else len(ex['points'])} exemplars, under {cfg.min_exemplars}; skipped")
                continue
            model = self._fit_class(cls, ex, g, feats, paintable, px_per_rad)
            if model is None:
                continue
            opts = cfg.for_class(cls)
            max_d = opts["max_distance_m"] if opts["max_distance_m"] is not None else \
                float(min(cfg.max_distance_m, max(2.0 * model["fit"]["r_fit_m"], 20.0)))
            model["draw"] = {"enabled": opts["enabled"], "density_multiplier": opts["density_multiplier"],
                             "max_distance_m": max_d,
                             "plan_density_factor": plan_density.get(cls, 1.0) if cfg.plan_density_words else 1.0}
            if opts["cluster_radius_m"] is not None:
                model["clustering"]["radius_m"] = opts["cluster_radius_m"]
                model["clustering"]["source"] = "config"
            if opts["cluster_size"] is not None:
                model["clustering"]["size"] = opts["cluster_size"]
                model["clustering"]["source"] = "config"
            if opts["enabled"]:
                seed = opts["seed"] if opts["seed"] is not None else int(self.seed or 0)
                placed = self._draw(model, ex, g, feats, paintable, seed)
                dropped = self._drop_painted(context, cls, count)
                for x, z, w, h, bucket in placed:
                    context.add_object(f"metadata_{next_idx}", {
                        "class": cls, "synthetic": True, "population_fit": True,
                        "world_position": [float(x), 0.0, float(z)],
                        "world_width": float(w), "world_height": float(h), "bucket": int(bucket)})
                    next_idx += 1
                model["draw"].update({"n": len(placed), "seed": seed, "dropped_painted": dropped})
                self.log_info(f"  {cls}: drew {len(placed)} instances (seed {seed}), dropped {dropped} painted")
                if self.temp is not None:
                    self._debug_image(cls, model, ex, g, placed)
            else:
                self.log_info(f"  {cls}: model fitted, draw off (class_overrides.{cls}.enabled)")
            model["summary"] = self._summary(model)
            models[cls] = model
        if next_idx != count:
            context.add_object(ContextKey.OBJECT_COUNT, next_idx)
        context.add_object(MODELS_KEY, models)
        context.add_object(MARKER, True)
        return context

    # ── pieces ────────────────────────────────────────────────────────────────
    def _exemplars(self, dist, cls: str) -> Optional[dict]:
        pts, sizes, buckets, regions = [], [], [], {}
        distributions = getattr(dist, "distributions", None) or (dist.get("distributions") if isinstance(dist, dict) else {})
        for region, by_type in (distributions or {}).items():
            td = by_type.get(cls) if isinstance(by_type, dict) else None
            if td is None:
                continue
            p = getattr(td, "points", None) if not isinstance(td, dict) else td.get("points")
            s = getattr(td, "sizes", None) if not isinstance(td, dict) else td.get("sizes")
            b = getattr(td, "buckets", None) if not isinstance(td, dict) else td.get("buckets")
            if not p:
                continue
            pts += [list(map(float, q)) for q in p]
            sizes += [list(map(float, q)) for q in (s or [[0.4, 0.4]] * len(p))]
            buckets += [int(q) for q in (b or [0] * len(p))]
            regions[str(region)] = len(p)
        if not pts:
            return None
        return {"points": np.asarray(pts, np.float32), "sizes": np.asarray(sizes, np.float32),
                "buckets": np.asarray(buckets, np.int64), "regions": regions}

    def _plan_density(self, context) -> dict[str, float]:
        out: dict[str, float] = {}
        try:
            from pipeline.scene_plan import ScenePlan
            from pipeline.object_typing.categories import normalize_category
            plan = ScenePlan.from_context(context)
            if plan is not None:
                for sp in plan.items("species", "inferred"):
                    c = normalize_category(sp.get("class"))
                    w = str(sp.get("density") or "").lower()
                    f = next((v for k, v in _PLAN_WORDS.items() if k in w), None)
                    if c and f is not None:
                        out[c] = max(out.get(c, 0.0), f)
        except Exception as exc:
            self.log_info(f"  plan density not read: {exc}")
        return out

    def _px_per_rad(self, context) -> float:
        try:
            intr = context.input_object(ContextKey.INTRINSICS) or {}
            photo = context.input_image(ContextKey.INPUT)
            fx = float(intr.get("fx", 460.0))
            scale = 1.0
            if photo is not None and intr.get("width"):
                scale = float(photo.image.width) / float(intr["width"])
            return fx * scale
        except Exception:
            return 1100.0

    def _window(self, ex: dict, g: dict, paintable: np.ndarray) -> np.ndarray:
        """Cells the detector could have found this class in: paintable ground with a
        panorama sample, inside the photographed cone, nearer than ~1.2x the farthest
        exemplar. A cell that holds an exemplar is in by definition."""
        cfg: PopulationFitConfiguration = self.config
        d = np.hypot(ex["points"][:, 0], ex["points"][:, 1])
        r_fit = max(cfg.fit_radius_factor * float(np.percentile(d, 95)), 5.0)
        row, col = self._cells(ex["points"], g)
        holds = np.zeros(g["rm"].shape, bool)
        holds[row, col] = True
        win = paintable & (g["dist"] <= r_fit) & g["cone"]
        out = {"paintable": int((~paintable[row, col]).sum()), "outside_cone": int((~g["cone"][row, col]).sum()),
               "beyond_r_fit": int((d > r_fit).sum()), "no_uv": int((~g["valid"][row, col]).sum())}
        self.log_info(f"  window radius {r_fit:.0f} m; exemplars failing: {out}")
        self._r_fit = r_fit
        return win | holds

    def _cells(self, points: np.ndarray, g: dict) -> tuple[np.ndarray, np.ndarray]:
        col = ((points[:, 0] + g["grid_m"] / 2.0) / g["cell"]).astype(np.int64)
        row = ((points[:, 1] + g["grid_m"] / 2.0) / g["cell"]).astype(np.int64)
        return np.clip(row, 0, g["n"] - 1), np.clip(col, 0, g["n"] - 1)

    def _effort(self, g: dict, size_m: float, px_per_rad: float) -> np.ndarray:
        cfg: PopulationFitConfiguration = self.config
        d_ref = px_per_rad * max(size_m, 0.05) / cfg.detect_min_px
        e = np.power(np.clip(d_ref / np.maximum(g["dist"], 0.5), 0.0, 1.0), cfg.detect_gamma)
        return np.clip(e, cfg.min_effort, 1.0).astype(np.float32)

    def _fit_class(self, cls, ex, g, feats, paintable, px_per_rad) -> Optional[dict]:
        cfg: PopulationFitConfiguration = self.config
        win = self._window(ex, g, paintable)
        if int(win.sum()) < 50:
            self.log_info(f"  {cls}: observation window has {int(win.sum())} cells; skipped")
            return None
        row, col = self._cells(ex["points"], g)
        counts = np.zeros(g["rm"].shape, np.float32)
        np.add.at(counts, (row, col), 1.0)
        inside = win[row, col]
        size_m = float(np.median(ex["sizes"][:, 1])) if len(ex["sizes"]) else 0.4
        effort = self._effort(g, size_m, px_per_rad)
        names = list(feats)
        F_all = np.stack([feats[k] for k in names], axis=-1) if names else np.zeros(g["rm"].shape + (0,), np.float32)
        mean = F_all[win].mean(axis=0) if names else np.zeros(0)
        std = F_all[win].std(axis=0) + 1e-6 if names else np.ones(0)
        # cells with no panorama sample carry no greenness: hold them at the window mean
        for i, k in enumerate(names):
            if k in ("greenness", "bloom"):
                F_all[..., i] = np.where(g["valid"], F_all[..., i], mean[i])
        Z = (F_all - mean) / std
        area = g["cell"] ** 2
        log_off = np.log(area * effort)

        def fit_on(mask):
            return fit_poisson(Z[mask], counts[mask], log_off[mask], cfg.ridge)

        n_ex_in = int(inside.sum())

        def holdout_gain(ridge):
            gains = []
            for side in (1, -1):
                train = win & ((g["xx"] * side) < 0)
                test = win & ~train
                if counts[train].sum() < cfg.min_exemplars or counts[test].sum() < 2:
                    continue
                b, _ = fit_poisson(Z[train], counts[train], log_off[train], ridge)
                eta_t = b[0] + Z[test] @ b[1:] + log_off[test]
                ll_t = float((counts[test] * eta_t - np.exp(np.clip(eta_t, -30, 30))).sum())
                gains.append((ll_t - _null_loglik(counts[test], log_off[test])) / max(counts[test].sum(), 1.0))
            return float(np.mean(gains)) if gains else None

        # A model that predicts the held-out half worse than a flat rate is overfitted:
        # stiffen the ridge, and at worst fall back to the flat rate (features zeroed).
        ridge = cfg.ridge
        hold = holdout_gain(ridge) if cfg.holdout and n_ex_in >= 2 * cfg.min_exemplars else None
        tried = [(ridge, hold)]
        while hold is not None and hold < 0 and ridge < cfg.ridge * 1000:
            ridge *= 10.0
            hold = holdout_gain(ridge)
            tried.append((ridge, hold))
        flat = hold is not None and hold < 0
        beta, ll = fit_on(win) if not flat else fit_poisson(Z[win] * 0.0, counts[win], log_off[win], ridge)
        if not flat:
            beta, ll = fit_poisson(Z[win], counts[win], log_off[win], ridge)
        ll0 = _null_loglik(counts[win], log_off[win])
        model = {
            "class": cls, "features": names, "intercept": float(beta[0]),
            "coef": {k: float(b) for k, b in zip(names, beta[1:])},
            "standardize": {"mean": mean.tolist(), "std": std.tolist()},
            "effort": {"d_ref_m": float(px_per_rad * max(size_m, 0.05) / cfg.detect_min_px),
                       "gamma": cfg.detect_gamma, "size_m": size_m},
            "exemplars": {"n": int(len(ex["points"])), "in_window": n_ex_in, "regions": ex["regions"],
                          "median_distance_m": float(np.median(np.hypot(ex["points"][:, 0], ex["points"][:, 1]))),
                          "max_distance_m": float(np.hypot(ex["points"][:, 0], ex["points"][:, 1]).max())},
            "fit": {"cells": int(win.sum()), "loglik": ll, "loglik_null": ll0,
                    "gain_per_exemplar": (ll - ll0) / max(n_ex_in, 1), "cell_m": g["cell"],
                    "ridge": ridge, "flat": flat, "holdout_gain_per_exemplar": hold,
                    "ridge_search": tried, "r_fit_m": float(self._r_fit)},
        }
        # mean intensity over the window (per m2, without effort)
        eta = beta[0] + Z[win] @ beta[1:]
        model["fit"]["mean_intensity_m2"] = float(np.exp(np.clip(eta, -30, 30)).mean())
        # clustering from spacing
        pts_in = ex["points"][inside]
        lam = max(model["fit"]["mean_intensity_m2"] * float(effort[win].mean()), 1e-6)
        nn = _nn_spacing(pts_in)
        expect = 0.5 / math.sqrt(lam)
        clustering = {"radius_m": cfg.cluster_radius_m, "size": cfg.cluster_size, "source": "config",
                      "nn_spacing_m": nn, "poisson_nn_m": expect}
        if cfg.estimate_clustering and np.isfinite(nn) and nn > 0 and len(pts_in) >= 12:
            ratio = nn / expect
            if ratio < 0.85:
                clustering.update({"radius_m": float(np.clip(1.5 * nn, 0.4, 10.0)),
                                   "size": float(np.clip((1.0 / ratio) ** 2, 2.0, 12.0)), "source": "exemplars"})
            else:
                clustering.update({"radius_m": float(np.clip(1.5 * nn, 0.3, 10.0)), "size": 1.0, "source": "exemplars"})
        model["clustering"] = clustering
        self.log_info(
            f"  {cls}: {n_ex_in}/{len(ex['points'])} exemplars in a {int(win.sum())}-cell window; "
            f"intensity {model['fit']['mean_intensity_m2']:.2f}/m2; coef "
            + ", ".join(f"{k} {v:+.2f}" for k, v in model["coef"].items())
            + f"; gain {model['fit']['gain_per_exemplar']:.2f}/exemplar"
            + (f", held-out {model['fit']['holdout_gain_per_exemplar']:.2f}" if model["fit"].get("holdout_gain_per_exemplar") is not None else "")
            + f"; clusters {clustering['size']:.1f} x {clustering['radius_m']:.1f} m ({clustering['source']})"
            + (f"; FLAT (ridge search {tried})" if flat else (f"; ridge {ridge:g}" if ridge != cfg.ridge else "")))
        return model

    def _intensity(self, model: dict, g: dict, feats: dict) -> np.ndarray:
        names = model["features"]
        mean = np.asarray(model["standardize"]["mean"], np.float32)
        std = np.asarray(model["standardize"]["std"], np.float32)
        eta = np.full(g["rm"].shape, model["intercept"], np.float32)
        for i, k in enumerate(names):
            f = feats[k]
            if k in ("greenness", "bloom"):
                f = np.where(g["valid"], f, mean[i])
            # outside the fitted range of a feature the model is extrapolating: hold the
            # standardised value at +-2.5 so a cliff or a lake shore cannot explode it
            eta += model["coef"][k] * np.clip((f - mean[i]) / std[i], -2.5, 2.5)
        return np.minimum(np.exp(np.clip(eta, -30, 30)), self.config.max_intensity_m2)

    def _draw(self, model: dict, ex: dict, g: dict, feats: dict, paintable: np.ndarray, seed: int) -> list:
        cfg: PopulationFitConfiguration = self.config
        rng = np.random.default_rng(seed)
        lam = self._intensity(model, g, feats) * model["draw"]["density_multiplier"] * model["draw"]["plan_density_factor"]
        allowed = paintable & (g["dist"] <= model["draw"]["max_distance_m"])
        lam = np.where(allowed, lam, 0.0)
        area = g["cell"] ** 2
        mu = max(float(model["clustering"]["size"]), 1.0)
        sigma = float(model["clustering"]["radius_m"])
        parents = rng.poisson(lam * area / mu)
        rows, cols = np.nonzero(parents)
        reps = parents[rows, cols]
        prow = np.repeat(rows, reps).astype(np.float64) + rng.random(reps.sum())
        pcol = np.repeat(cols, reps).astype(np.float64) + rng.random(reps.sum())
        if mu > 1.0:
            k = rng.poisson(mu, size=len(prow))
            prow = np.repeat(prow, k)
            pcol = np.repeat(pcol, k)
            prow = prow + rng.normal(0.0, sigma / g["cell"], size=len(prow))
            pcol = pcol + rng.normal(0.0, sigma / g["cell"], size=len(pcol))
        ri = np.clip(np.floor(prow).astype(np.int64), 0, g["n"] - 1)
        ci = np.clip(np.floor(pcol).astype(np.int64), 0, g["n"] - 1)
        keep = allowed[ri, ci]
        prow, pcol = prow[keep], pcol[keep]
        if len(prow) > cfg.max_instances:
            sel = rng.choice(len(prow), size=cfg.max_instances, replace=False)
            prow, pcol = prow[sel], pcol[sel]
        x = pcol * g["cell"] - g["grid_m"] / 2.0
        z = prow * g["cell"] - g["grid_m"] / 2.0
        n = len(x)
        pick = rng.integers(0, len(ex["sizes"]), size=n)
        jit = 1.0 + cfg.size_jitter * (2.0 * rng.random(n) - 1.0)
        w = ex["sizes"][pick, 0] * jit
        h = ex["sizes"][pick, 1] * jit
        b = ex["buckets"][pick]
        return list(zip(x.tolist(), z.tolist(), w.tolist(), h.tolist(), b.tolist()))

    def _drop_painted(self, context: PipelineContext, cls: str, count: int) -> int:
        dropped = 0
        for i in range(count):
            m = context.input_object(f"metadata_{i}")
            if isinstance(m, dict) and m.get("synthetic") and str(m.get("class")) == cls \
                    and m.get("source") not in ("plan_band", "plan_population"):
                context.add_object(f"metadata_{i}", None)
                dropped += 1
        return dropped

    def _summary(self, model: dict) -> dict:
        s = {"fit_exemplars": model["exemplars"]["n"], "fit_cells": model["fit"]["cells"],
             "fit_intensity_m2": round(model["fit"]["mean_intensity_m2"], 3),
             "fit_gain_per_exemplar": round(model["fit"]["gain_per_exemplar"], 3),
             "fit_cluster_size": round(model["clustering"]["size"], 2),
             "fit_cluster_radius_m": round(model["clustering"]["radius_m"], 2),
             "fit_effort_d_ref_m": round(model["effort"]["d_ref_m"], 1)}
        if model["fit"].get("holdout_gain_per_exemplar") is not None:
            s["fit_holdout_gain"] = round(model["fit"]["holdout_gain_per_exemplar"], 3)
        for k, v in model["coef"].items():
            s[f"coef_{k}"] = round(v, 3)
        if "n" in model["draw"]:
            s["draw_n"] = model["draw"]["n"]
        return s

    def _debug_image(self, cls: str, model: dict, ex: dict, g: dict, placed: list) -> None:
        try:
            from PIL import Image, ImageDraw
            n = g["n"]
            img = Image.new("RGB", (n, n), (20, 20, 20))
            px = img.load()
            rm = g["rm"]
            for r in range(n):
                for c in range(n):
                    t = int(rm[r, c])
                    px[c, r] = (40, 60, 120) if t == int(RegionType.WATER) else ((50, 90, 50) if t in PAINTABLE else (30, 30, 30))
            d = ImageDraw.Draw(img)
            for x, z, *_ in placed:
                c = int((x + g["grid_m"] / 2) / g["cell"]); r = int((z + g["grid_m"] / 2) / g["cell"])
                if 0 <= c < n and 0 <= r < n:
                    px[c, r] = (230, 200, 60)
            for x, z in ex["points"]:
                c = int((x + g["grid_m"] / 2) / g["cell"]); r = int((z + g["grid_m"] / 2) / g["cell"])
                d.ellipse([c - 2, r - 2, c + 2, r + 2], outline=(255, 60, 60))
            d.ellipse([n // 2 - 3, n // 2 - 3, n // 2 + 3, n // 2 + 3], fill=(255, 255, 255))
            Path(self.temp).mkdir(parents=True, exist_ok=True)
            img.resize((min(1024, n * 2), min(1024, n * 2)), Image.NEAREST).save(Path(self.temp) / f"population_fit_{cls}.png")
            (Path(self.temp) / f"population_fit_{cls}.json").write_text(json.dumps(model, indent=1, default=float))
        except Exception as exc:
            self.log_info(f"  debug image failed: {exc}")
