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
clip(U + s (S - U) + a (A - U)), without rendering again. s and a are searched on
a log grid for the luminance percentiles of the lit pixels nearer than near_m
(the grid and the placed objects; the far shell is the photograph itself) that
best match the photograph's at the same pixels.

Input   ContextKey.SCENE (Scene Generation's scene.json on disk, the run's caches),
        ContextKey.INPUT, SCENE_PLAN, the skybox panorama
Output  exposure_fit.json in this stage's folder (render-scene applies it),
        scene.lighting.fit / fit_sun_scale / ambient_scale, scene.atmosphere
"""
from __future__ import annotations

import json
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


def fit_light_scales(photo: np.ndarray, sun: np.ndarray, amb: np.ndarray, unlit: np.ndarray, near: np.ndarray,
                     percentiles: list[float], scale_min: float = 0.25, scale_max: float = 4.0,
                     steps: int = 41) -> dict:
    """(sun scale, ambient scale) whose lit frame's luminance percentiles over the
    near pixels best match the photograph's, in log space. Arrays are (H, W, 3)
    in [0, 1]; near is the (H, W) mask of pixels the fit judges."""
    pct = np.asarray(percentiles, dtype=np.float64)
    lp = np.percentile(photo[near] @ LUMA, pct)
    u = unlit[near].astype(np.float32)
    ds = (sun[near] - unlit[near]).astype(np.float32)
    da = (amb[near] - unlit[near]).astype(np.float32)

    def render_pct(s: float, a: float) -> np.ndarray:
        c = np.clip(u + s * ds + a * da, 0.0, 1.0)
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
    c_after = np.clip(u + s_best * ds + a_best * da, 0.0, 1.0)
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

    def _render_passes(self, run_dir: Path, out: Path) -> Optional[str]:
        cfg: ExposureFitConfiguration = self.config
        cmd = [sys.executable, str(ROOT / "scripts" / "render-scene.py"), str(run_dir), "--out", str(out),
               "--views", "photo", "--width", str(cfg.width), "--save-depth", "--light-passes", "--no-fit"]
        if not cfg.grass:
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
        run_dir = self.output.parent if self.output is not None else None
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
        self.advance_progress(task)

        # ── the three light passes of the photo view ────────────────────────
        out = (self.temp or (run_dir / self.name / "build")) / "render"
        out.mkdir(parents=True, exist_ok=True)
        failure = self._render_passes(run_dir, out)
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
            scene.lighting.fit = {k: fit[k] for k in ("sun_scale", "ambient_scale", "near_m", "n_near", "err_before", "err_after")}
            sun = scene.lighting.sun() or {}
            self.log_info(f"Sun intensity {sun.get('intensity_base')} -> {sun.get('intensity')} for the client")
        else:
            self.log_warning("Exposure fit: the scene has no lighting; the fit is written to exposure_fit.json only")
        if self.output is not None:
            (self.output / "exposure_fit.json").write_text(json.dumps(fit, indent=2))
        context.add_scene(ContextKey.SCENE, scene)
        context.add_object(MARKER, fit)
        self.finish_progress(task)
        return context
