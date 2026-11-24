"""Exposure Fit: the scene's lights fitted to the photograph.

The renderers (the client, scripts/render-scene.py that scene-check grades) light
the delit terrain and the placed meshes with the estimated sun and an ambient from
the sky, at scales that were never measured against the photograph: on Rainier
band17 the lit near field rendered at 0.87 of the photograph's luminance with its
highlights at 0.75 and its shadows darker still -- a harsh low sun with too little
fill, the paintbrush maroon, the snow flat (docs/hybrid-scene-plan.md section 7).

The frame is linear in the lights, so three passes of the photo view -- the sun
alone (S), the ambient alone (A), nothing lit (U: the emissive photographic
content, cards and the backdrop) -- give the lit frame for any pair of scales,
clip(U + s (S - U) + a (A - U)), without rendering again. Linear in LIGHT: the
passes arrive as the renderer writes them, pow(colour, 1/2.2), and are decoded
before they are combined and encoded again before they are compared (the first
ground fit combined the encoded frames, which predicts x4 of light as x4 of
pixel value where the renderer gives x1.9: Shark Fin's fitted ground was promised
at 0.27 of luminance and rendered at 0.14, and every run was fitted too dark). s and a are searched on
a log grid for the luminance percentiles of the lit pixels nearer than near_m
(the grid and the placed objects; the far shell is the photograph itself) that
best match the photograph's at the same pixels.

That fit judged every near pixel alike, and the placed objects decided it: Rainier's
generated flowers render brighter than the photographed ones and fill the near field,
so both lights went to the floor (x0.5) and the meadow behind them, whose texture IS
the photograph, rendered black; Shark Fin needed more fill than the x3 ceiling allowed
and stayed a silhouette. The ground fit (mode: ground, the default) judges the GROUND
instead -- the pixels where the terrain or the water is what the camera sees, found by
a second depth render without the placed objects. Their albedo is the delit
photograph, so lit correctly they reproduce it: block means of the lit frame against
the photograph's in log luminance (L1), the ambient free to x ambient_max (of light: Shark Fin's
dusk sky gives a sixtieth of the fill its beach received). Then the
ambient's colour per channel (the sky's mean is not the fill the ground received:
Shark Fin's orange sky lit a rose-grey beach), and one gain for the placed objects'
albedo where they cover enough of the frame to measure (object_gain: the generated
assets' brightness against the photograph's at the same pixels, under the fitted
lights).

Input   ContextKey.SCENE (Scene Generation's scene.json on disk, the run's caches),
        ContextKey.INPUT, SCENE_PLAN, the skybox panorama
Output  exposure_fit.json in this stage's folder (render-scene applies it),
        scene.lighting.fit / fit_sun_scale / ambient_scale, scene.atmosphere
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from logging import Logger
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image as PILImage

from pipeline.pipeline_context import ContextKey, PipelineContext
from pipeline.pipeline_stage import PipelineStage, PipelineStageConfiguration

MARKER = "exposure_fit"
ROOT = Path(__file__).resolve().parents[3]
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
GAMMA = 2.2     # scripts/render-scene.py's fragment shader writes pow(colour, 1 / 2.2)


def _lin(a: np.ndarray) -> np.ndarray:
    """A rendered frame (display values) as the light the shader summed."""
    return np.power(np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0), GAMMA)


def _enc(a: np.ndarray) -> np.ndarray:
    """Summed light as the renderer writes it."""
    return np.power(np.clip(a, 0.0, 1.0), 1.0 / GAMMA)


class ExposureFitConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        width: int = 480,
        near_m: float = 60.0,
        percentiles: Optional[list[float]] = None,
        scale_min: float = 0.25,
        scale_max: float = 4.0,
        grid_steps: int = 41,
        min_pixels: int = 2000,
        grass: bool = False,
        timeout_s: float = 900.0,
        atmosphere: bool = True,
        sun_warmth: bool = True,
        plan_sun: bool = True,
        plan_sun_az_tolerance_deg: float = 35.0,
        plan_sun_el_tolerance_deg: float = 20.0,
        plan_sun_min_elevation_deg: float = 20.0,
        mode: str = "ground",
        far_m: float = 400.0,
        ambient_max: float = 100.0,
        block_px: int = 8,
        min_ground_blocks: int = 40,
        tint: bool = True,
        tint_range: Optional[list[float]] = None,
        object_gain: bool = True,
        object_gain_range: Optional[list[float]] = None,
        min_object_blocks: int = 60,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.width = int(width)
        self.near_m = float(near_m)
        self.percentiles = [float(p) for p in (percentiles or [10, 25, 50, 75, 90])]
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.grid_steps = int(grid_steps)
        self.min_pixels = int(min_pixels)
        self.grass = bool(grass)
        self.timeout_s = float(timeout_s)
        self.atmosphere = bool(atmosphere)
        self.sun_warmth = bool(sun_warmth)
        # Aim the key light at the plan's sun when the scene's lies outside the
        # plan's window: Panorama Lighting does this at estimation time, but a
        # cached lighting from before that rule (Rainier: LuxDiT's sun at az 176
        # el 48, behind the camera and high, for a photograph lit low from the
        # left) carries no plan direction, and every run since was lit flat.
        self.plan_sun = bool(plan_sun)
        self.plan_sun_az_tolerance_deg = float(plan_sun_az_tolerance_deg)
        self.plan_sun_el_tolerance_deg = float(plan_sun_el_tolerance_deg)
        self.plan_sun_min_elevation_deg = float(plan_sun_min_elevation_deg)
        # ground: fit on the terrain and water pixels (see the module text);
        # percentiles: the earlier fit on every near pixel's luminance percentiles.
        self.mode = str(mode)
        self.far_m = float(far_m)
        self.ambient_max = float(ambient_max)
        self.block_px = max(2, int(block_px))
        self.min_ground_blocks = int(min_ground_blocks)
        self.tint = bool(tint)
        self.tint_range = [float(v) for v in (tint_range or [0.6, 1.6])]
        self.object_gain = bool(object_gain)
        self.object_gain_range = [float(v) for v in (object_gain_range or [0.7, 1.4])]
        self.min_object_blocks = int(min_object_blocks)


def fit_light_scales(photo: np.ndarray, sun: np.ndarray, amb: np.ndarray, unlit: np.ndarray, near: np.ndarray,
                     percentiles: list[float], scale_min: float = 0.25, scale_max: float = 4.0,
                     steps: int = 41) -> dict:
    """(sun scale, ambient scale) whose lit frame's luminance percentiles over the
    near pixels best match the photograph's, in log space. Arrays are (H, W, 3)
    in [0, 1]; near is the (H, W) mask of pixels the fit judges."""
    pct = np.asarray(percentiles, dtype=np.float64)
    lp = np.percentile(photo[near] @ LUMA, pct)
    u = _lin(unlit[near])
    ds = _lin(sun[near]) - u
    da = _lin(amb[near]) - u

    def render_pct(s: float, a: float) -> np.ndarray:
        c = _enc(u + s * ds + a * da)
        return np.percentile(c @ LUMA, pct)

    def err(pr: np.ndarray) -> float:
        return float(np.sum((np.log(pr + 0.02) - np.log(lp + 0.02)) ** 2))

    grid = np.geomspace(scale_min, scale_max, steps)
    before = render_pct(1.0, 1.0)
    best = (err(before), 1.0, 1.0)
    for s in grid:
        for a in grid:
            e = err(render_pct(float(s), float(a)))
            if e < best[0]:
                best = (e, float(s), float(a))
    e_best, s_best, a_best = best
    after = render_pct(s_best, a_best)
    c_after = _enc(u + s_best * ds + a_best * da)
    ratio = photo[near].mean(axis=0) / np.maximum(c_after.mean(axis=0), 1e-3)
    return {
        "sun_scale": round(s_best, 3), "ambient_scale": round(a_best, 3),
        "percentiles": [float(p) for p in pct],
        "photo": [round(float(v), 4) for v in lp],
        "render_before": [round(float(v), 4) for v in before],
        "render_after": [round(float(v), 4) for v in after],
        "err_before": round(err(before), 4), "err_after": round(e_best, 4),
        "colour_ratio_after": [round(float(v), 3) for v in ratio],
        "n_near": int(near.sum()),
    }


def _blocks(a: np.ndarray, b: int) -> np.ndarray:
    h, w = a.shape[:2]
    h2, w2 = h // b * b, w // b * b
    a = a[:h2, :w2].astype(np.float32)
    return a.reshape(h2 // b, b, w2 // b, b, *a.shape[2:]).mean(axis=(1, 3))


def fit_light_scales_ground(photo: np.ndarray, sun: np.ndarray, amb: np.ndarray, unlit: np.ndarray,
                            lit: np.ndarray, ground: np.ndarray, sun_range: tuple[float, float] = (0.5, 4.0),
                            ambient_range: tuple[float, float] = (0.5, 100.0), steps: int = 41, block_px: int = 8,
                            min_ground_blocks: int = 40, tint_range: Optional[tuple[float, float]] = (0.6, 1.6),
                            object_gain_range: Optional[tuple[float, float]] = (0.7, 1.4),
                            min_object_blocks: int = 60) -> dict:
    """Sun and ambient scales, the ambient's tint and the placed objects' albedo gain.

    photo/sun/amb/unlit are (H, W, 3) in [0, 1]; `lit` marks pixels a light reaches,
    `ground` those of them where the terrain or water is the visible surface. Blocks
    of block_px that are mostly ground are compared with the photograph's in log
    luminance (mean absolute error, with a weak pull to x1 so a direction the data
    does not constrain stays put). With too few ground blocks every lit block is
    judged instead. The passes are decoded to light before they are combined (see
    the module text); the scales are scales of light."""
    b = int(block_px)
    l_unlit = _lin(unlit)
    bp, bu = _blocks(photo, b), _blocks(l_unlit, b)
    bs, ba = _blocks(_lin(sun) - l_unlit, b), _blocks(_lin(amb) - l_unlit, b)
    g_frac, l_frac = _blocks(ground.astype(np.float32), b), _blocks(lit.astype(np.float32), b)
    g_ok = g_frac > 0.6
    judged = "ground"
    if int(g_ok.sum()) < min_ground_blocks:
        g_ok, judged = l_frac > 0.6, "every lit block (too little ground in view)"
    lp = np.log(bp[g_ok] @ LUMA + 0.02)
    u, ds, da = bu[g_ok], bs[g_ok], ba[g_ok]

    def err(sc: float, ac: float) -> float:
        c = _enc(u + sc * ds + ac * da)
        return float(np.abs(np.log(c @ LUMA + 0.02) - lp).mean())

    best = None
    for sc in np.geomspace(sun_range[0], sun_range[1], steps):
        for ac in np.geomspace(ambient_range[0], ambient_range[1], steps):
            e = err(float(sc), float(ac)) + 0.004 * (math.log(sc) ** 2 + math.log(ac) ** 2)
            if best is None or e < best[0]:
                best = (e, float(sc), float(ac))
    _, s_best, a_best = best
    tint = np.ones(3, dtype=np.float64)
    if tint_range is not None:
        for ch in range(3):
            lpc = np.log(bp[g_ok][:, ch] + 0.02)
            cand = None
            for t in np.geomspace(tint_range[0], tint_range[1], 31):
                c = _enc(u[:, ch] + s_best * ds[:, ch] + a_best * t * da[:, ch])
                e = float(np.abs(np.log(c + 0.02) - lpc).mean())
                if cand is None or e < cand[0]:
                    cand = (e, float(t))
            tint[ch] = cand[1]
        tint /= max(float(tint @ LUMA), 1e-6)       # a colour, not a second brightness
    tint32 = tint.astype(np.float32)
    after = _enc(u + s_best * ds + a_best * tint32 * da)
    out = {
        "mode": "ground", "judged": judged, "n_blocks": int(g_ok.sum()), "block_px": b,
        "sun_scale": round(s_best, 3), "ambient_scale": round(a_best, 3),
        "ambient_tint": [round(float(v), 3) for v in tint],
        "err_before": round(err(1.0, 1.0), 4), "err_after": round(float(np.abs(np.log(after @ LUMA + 0.02) - lp).mean()), 4),
        "photo_median": round(float(np.median(bp[g_ok] @ LUMA)), 4),
        "render_median_before": round(float(np.median(_enc(u + ds + da) @ LUMA)), 4),
        "render_median_after": round(float(np.median(after @ LUMA)), 4),
        "colour_ratio_after": [round(float(v), 3) for v in bp[g_ok].mean(axis=0) / np.maximum(after.mean(axis=0), 1e-3)],
        "at_ceiling": bool(a_best >= ambient_range[1] * 0.999 or s_best >= sun_range[1] * 0.999),
        "object_gain": 1.0, "n_object_blocks": 0,
    }
    if object_gain_range is not None:
        o_ok = (l_frac > 0.6) & (g_frac < 0.2)
        out["n_object_blocks"] = int(o_ok.sum())
        if int(o_ok.sum()) >= min_object_blocks:
            lit_o = bs[o_ok] * s_best + ba[o_ok] * a_best * tint32
            # The gain g scales the lit part (albedo), not the emissive part.
            want = np.log(bp[o_ok] @ LUMA + 0.02)
            cand = None
            for g in np.geomspace(object_gain_range[0], object_gain_range[1], 41):
                c = _enc(bu[o_ok] + g * lit_o)
                e = float(np.abs(np.log(c @ LUMA + 0.02) - want).mean()) + 0.004 * math.log(g) ** 2
                if cand is None or e < cand[0]:
                    cand = (e, float(g))
            out["object_gain"] = round(cand[1], 3)
    return out


def horizon_colour(sky: PILImage.Image, above_deg: tuple[float, float] = (0.5, 3.0)) -> str:
    """Median colour of the skybox between above_deg[0] and above_deg[1] above the
    horizon, round the whole panorama: the haze the photograph's own far field
    fades toward."""
    arr = np.asarray(sky.convert("RGB"), dtype=np.float32)
    h = arr.shape[0]
    r0 = int(round(h / 2 - above_deg[1] / 180.0 * h))
    r1 = int(round(h / 2 - above_deg[0] / 180.0 * h))
    band = arr[max(0, r0):max(r0 + 1, r1)].reshape(-1, 3)
    med = np.median(band, axis=0)
    return "#{:02x}{:02x}{:02x}".format(*(int(round(float(v))) for v in np.clip(med, 0, 255)))


class ExposureFitStage(PipelineStage):
    @classmethod
    def config_class(cls):
        return ExposureFitConfiguration

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(MARKER) is not None

    def _render_passes(self, run_dir: Path, out: Path, scene_path: Optional[Path] = None,
                       ground_only: bool = False) -> Optional[str]:
        cfg: ExposureFitConfiguration = self.config
        cmd = [sys.executable, str(ROOT / "scripts" / "render-scene.py"), str(run_dir), "--out", str(out),
               "--views", "photo", "--width", str(cfg.width), "--save-depth", "--no-fit"]
        if ground_only:
            # The terrain and the water alone: where this depth equals the full
            # frame's, the ground is what the camera sees.
            cmd += ["--only", "#no-placed-object#", "--no-grass", "--no-shadows"]
        else:
            cmd.append("--light-passes")
        if scene_path is not None:
            cmd += ["--scene", str(scene_path)]
        if not cfg.grass and not ground_only:
            cmd.append("--no-grass")
        env = dict(os.environ)
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        try:
            res = subprocess.run(cmd, cwd=str(ROOT / "server"), capture_output=True, text=True,
                                 timeout=cfg.timeout_s, env=env)
        except subprocess.TimeoutExpired:
            return f"render timed out after {cfg.timeout_s:.0f} s"
        if res.returncode != 0:
            return "render failed: " + (res.stderr or res.stdout)[-1500:]
        return None

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: ExposureFitConfiguration = self.config
        task = self.create_progress(3)
        scene = context.input_scene(ContextKey.SCENE)
        # The run directory: the one the context was loaded from (also under replay-stage, whose
        # output goes elsewhere), else the parent of this stage's own folder.
        run_dir = getattr(context, "loaded_from", None) or (self.output.parent if self.output is not None else None)
        if scene is None or run_dir is None or not (run_dir / "Scene Generation" / "scene.json").exists():
            self.log_info("Exposure fit: no scene on disk to render; nothing fitted")
            context.add_object(MARKER, {"skipped": "no scene"})
            self.finish_progress(task)
            return context

        # ── atmosphere from the plan and the skybox ─────────────────────────
        if cfg.atmosphere:
            from pipeline.scene_plan import ScenePlan
            plan = ScenePlan.from_context(context)
            sky = context.input_panorama(scene.skybox or ContextKey.PANORAMA_SKY)
            atmosphere = {
                "haze_near_m": plan.get("atmosphere.haze_near_m") if plan else None,
                "haze_far_m": plan.get("atmosphere.haze_far_m") if plan else None,
                "tint_rgb": plan.get("atmosphere.tint_rgb") if plan else None,
                "haze_color": horizon_colour(sky.image) if sky is not None else None,
                "evidence": plan.evidence("atmosphere") if plan else None,
            }
            scene.atmosphere = atmosphere
            self.log_info(f"Atmosphere: haze {atmosphere['haze_near_m']}-{atmosphere['haze_far_m']} m "
                          f"({atmosphere['evidence']}), horizon colour {atmosphere['haze_color']}")
        # ── the key light's direction from the plan ─────────────────────────
        if cfg.plan_sun and plan is not None and scene.lighting is not None:
            paz, pel = plan.get("sun.azimuth_deg"), plan.get("sun.elevation_deg")
            sun = scene.lighting.sun() or {}
            cur = sun.get("direction")
            if (plan.evidence("sun") in ("visible", "inferred") and isinstance(paz, (int, float))
                    and isinstance(pel, (int, float)) and cur and sun.get("direction_source") != "plan"):
                cur_az = math.degrees(math.atan2(float(cur[0]), float(cur[2])))
                cur_el = math.degrees(math.asin(max(-1.0, min(1.0, float(cur[1])))))
                daz = (cur_az - float(paz) + 180.0) % 360.0 - 180.0
                if abs(daz) > cfg.plan_sun_az_tolerance_deg or abs(cur_el - float(pel)) > cfg.plan_sun_el_tolerance_deg:
                    el_used = max(float(pel), cfg.plan_sun_min_elevation_deg)
                    el, az = math.radians(el_used), math.radians(float(paz))
                    direction = [round(math.cos(el) * math.sin(az), 5), round(math.sin(el), 5),
                                 round(math.cos(el) * math.cos(az), 5)]
                    colour = sun.get("color_measured") or sun.get("color") or "#ffffff"
                    scene.lighting.measured_sun = (direction, colour)
                    scene.lighting.direction_source_override = "plan"
                    self.log_info(f"Key light aimed at the plan's sun az {paz:g} el {el_used:g} (plan el {pel:g}, "
                                  f"{plan.evidence('sun')}): the scene's was az {cur_az:.0f} el {cur_el:.0f} "
                                  f"({sun.get('direction_source')}), outside the window; the fit below sets its strength")
        # ── the sun's warmth from the plan ──────────────────────────────────
        # The key light's elevation was raised to light the near field; its colour
        # is the photographed sun's. The plan's elevation sets it, and its warmth
        # word lowers it further when the photograph says golden light.
        if cfg.sun_warmth and plan is not None and scene.lighting is not None:
            pel = plan.get("sun.elevation_deg")
            warmth = str(plan.get("sun.warmth") or "").lower()
            if plan.evidence("sun") in ("visible", "inferred") and isinstance(pel, (int, float)):
                cap = {"golden": 10.0, "amber": 10.0, "orange": 7.0, "warm": 15.0}
                el_colour = float(pel)
                for word, deg in cap.items():
                    if word in warmth:
                        el_colour = min(el_colour, deg)
                before = (scene.lighting.sun() or {}).get("color")
                sun_el = (scene.lighting.sun() or {}).get("elevation_deg")
                if sun_el is None or el_colour < float(sun_el) - 0.5:
                    scene.lighting.sun_colour_elevation_deg = el_colour
                    after = (scene.lighting.sun() or {}).get("color")
                    self.log_info(f"Sun colour for the plan's {warmth or 'unstated'} sun at {pel:g} deg "
                                  f"(colour elevation {el_colour:g}, key light at {sun_el}): {before} -> {after}")
        self.advance_progress(task)

        # ── the three light passes of the photo view ────────────────────────
        out = (self.temp or (run_dir / self.name / "build")) / "render"
        out.mkdir(parents=True, exist_ok=True)
        # The passes are lit by THIS scene's sun (direction, colour), not the one
        # Scene Generation wrote.
        from pipeline.context_value import JSONEncoder
        fit_scene = out.parent / "scene_for_fit.json"
        fit_scene.write_text(json.dumps(scene.encode(), cls=JSONEncoder))
        failure = self._render_passes(run_dir, out, fit_scene)
        if failure is not None:
            self.log_warning(f"Exposure fit: {failure}")
            context.add_scene(ContextKey.SCENE, scene)
            context.add_object(MARKER, {"skipped": failure[:200]})
            self.finish_progress(task)
            return context
        self.advance_progress(task)

        passes = {t: np.load(out / f"photo_pass_{t}.npy").astype(np.float32) for t in ("sun", "amb", "unlit")}
        depth = np.load(out / "photo_depth.npy").astype(np.float32)
        h, w = depth.shape
        photo_img = context.input_image(ContextKey.INPUT)
        if photo_img is None:
            self.log_warning("Exposure fit: no input photograph in the context")
            context.add_scene(ContextKey.SCENE, scene)
            context.add_object(MARKER, {"skipped": "no photograph"})
            self.finish_progress(task)
            return context
        photo = np.asarray(photo_img.image.convert("RGB").resize((w, h), PILImage.LANCZOS), dtype=np.float32) / 255.0
        near = (depth > 0.0) & (depth < cfg.near_m)
        if int(near.sum()) < cfg.min_pixels:
            self.log_info(f"Exposure fit: only {int(near.sum())} lit pixels nearer than {cfg.near_m:.0f} m; nothing fitted")
            context.add_scene(ContextKey.SCENE, scene)
            context.add_object(MARKER, {"skipped": "too few near pixels", "n_near": int(near.sum())})
            self.finish_progress(task)
            return context

        ground_depth = None
        if cfg.mode == "ground":
            gout = out.parent / "render_ground"
            gout.mkdir(parents=True, exist_ok=True)
            failure = self._render_passes(run_dir, gout, fit_scene, ground_only=True)
            if failure is None and (gout / "photo_depth.npy").exists():
                ground_depth = np.load(gout / "photo_depth.npy").astype(np.float32)
            else:
                self.log_warning(f"Exposure fit: no ground-only depth ({failure}); fitting the percentiles of every near pixel")
        if ground_depth is not None and ground_depth.shape == depth.shape:
            lit = (depth > 0.0) & (depth < cfg.far_m) & (
                (np.abs(passes["sun"] - passes["unlit"]) + np.abs(passes["amb"] - passes["unlit"])).sum(axis=-1) > 2e-3)
            ground = lit & (ground_depth > 0.0) & (np.abs(depth - ground_depth) < 0.03 * depth + 0.05)
            fit = fit_light_scales_ground(
                photo, passes["sun"], passes["amb"], passes["unlit"], lit, ground,
                (cfg.scale_min, cfg.scale_max), (cfg.scale_min, cfg.ambient_max), cfg.grid_steps, cfg.block_px,
                cfg.min_ground_blocks, tuple(cfg.tint_range) if cfg.tint else None,
                tuple(cfg.object_gain_range) if cfg.object_gain else None, cfg.min_object_blocks)
            fit.update({"far_m": cfg.far_m, "width": w, "height": h, "n_near": int(ground.sum()), "near_m": cfg.far_m,
                        "fitted_on": "block means of the photo view's ground pixels (terrain, water), log luminance L1"})
            self.log_info(
                f"Exposure fit on {fit['n_blocks']} blocks of {fit['judged']}: sun x{fit['sun_scale']:.2f}, "
                f"ambient x{fit['ambient_scale']:.2f} tint {fit['ambient_tint']}, object gain x{fit['object_gain']:.2f} "
                f"({fit['n_object_blocks']} object blocks); median luminance photo {fit['photo_median']}, render "
                f"{fit['render_median_before']} -> {fit['render_median_after']}; err {fit['err_before']:.3f} -> "
                f"{fit['err_after']:.3f}; colour ratio after {fit['colour_ratio_after']}"
                + ("; AT THE CEILING: the ground is darker than the lights can lift" if fit["at_ceiling"] else ""))
        else:
            fit = fit_light_scales(photo, passes["sun"], passes["amb"], passes["unlit"], near, cfg.percentiles,
                                   cfg.scale_min, cfg.scale_max, cfg.grid_steps)
            fit.update({"near_m": cfg.near_m, "width": w, "height": h,
                        "fitted_on": "the photo view's pixels nearer than near_m, luminance percentiles in log space"})
            pcts = "/".join(f"{p:g}" for p in fit["percentiles"])
            self.log_info(
                f"Exposure fit on {fit['n_near']} px < {cfg.near_m:.0f} m: sun x{fit['sun_scale']:.2f}, ambient x{fit['ambient_scale']:.2f} "
                f"(p{pcts}: photo {fit['photo']}, render {fit['render_before']} -> {fit['render_after']}; "
                f"err {fit['err_before']:.3f} -> {fit['err_after']:.3f}; colour ratio after {fit['colour_ratio_after']})"
            )
        if scene.lighting is not None:
            scene.lighting.fit_sun_scale = float(fit["sun_scale"])
            scene.lighting.ambient_scale = float(fit["ambient_scale"])
            scene.lighting.fit = {k: fit[k] for k in ("sun_scale", "ambient_scale", "near_m", "n_near", "err_before", "err_after",
                                                       "ambient_tint", "object_gain", "mode") if k in fit}
            sun = scene.lighting.sun() or {}
            self.log_info(f"Sun intensity {sun.get('intensity_base')} -> {sun.get('intensity')} for the client")
        else:
            self.log_warning("Exposure fit: the scene has no lighting; the fit is written to exposure_fit.json only")
        if self.output is not None:
            (self.output / "exposure_fit.json").write_text(json.dumps(fit, indent=2))
        # The scene as fitted, beside the passes (what a replay into a scratch folder renders).
        (out.parent / "scene_fitted.json").write_text(json.dumps(scene.encode(), cls=JSONEncoder))
        context.add_scene(ContextKey.SCENE, scene)
        context.add_object(MARKER, fit)
        self.finish_progress(task)
        return context
