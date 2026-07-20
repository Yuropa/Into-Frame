from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.depth_utils import Depth
from util.terrain_noise_utils import diffuse_heightmap
import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import binary_dilation


class PanoramaDepthPatchConfiguration(PipelineStageConfiguration):
    """
    Stage-specific config for PanoramaDepthPatchStage.

    Experimental alternative to PanoramaLoraCorrectionStage for the same
    underlying problem: the second PanoramaDepthStage pass re-estimates depth
    over PANORAMA_TERRAIN, including the region PanoramaForegroundInpaintingStage
    fabricated with ObjectClear -- a model with no equirectangular awareness, so
    its content doesn't obey equirect projection rules and the depth model is
    running out-of-distribution exactly there. Rather than trying to make the
    fabricated pixels more equirect-correct (PanoramaLoraCorrectionStage's
    approach), this stage ignores whatever depth the model produced inside that
    region entirely and replaces it with a pure geometric extrapolation from the
    real, trusted depth immediately surrounding it -- the same technique
    HeightMapGenerator already uses for its own nadir flat-ground prior (see
    diffuse_heightmap / util.terrain_noise_utils), just applied directly to the
    equirectangular depth image instead of the top-down height grid.

    n_iters (int, default 800):
        Laplacian diffusion iterations. More gives a smoother fill; 500-1000 is
        typical (see diffuse_heightmap).

    mask_extra_dilation_px (int, default 8):
        Extra dilation (panorama pixels) applied to PANORAMA_FOREGROUND_MASK
        before treating it as the hole to patch, beyond PanoramaForegroundInpaintingStage's
        own mask_dilation_px. A small positive margin discards a ring of depth
        samples right at ObjectClear's blend boundary too, since those pixels are
        a feathered mix of real and fabricated content and the depth model's
        estimate there is no more trustworthy than inside the mask itself.

    wrap_pad_px (int, default 64):
        Columns wrapped from the opposite edge onto each side before diffusing,
        so the fill is aware of horizontal seam continuity (column 0 and column
        w-1 are the same real-world direction) instead of treating the panorama's
        left/right edges as hard boundaries the way a plain 2D grid would. 0
        disables (matches diffuse_heightmap's native edge-padding behaviour).

    exclude_sky (bool, default True):
        Don't patch pixels PANORAMA_SKY_MASK marks as sky, even if they fall
        inside the (dilated) removal mask -- e.g. the gap left by a removed
        tower's silhouette can genuinely reveal sky, and diffusing ground-plane
        depth into it would be wrong. No-ops (patches everything in the mask)
        if no sky mask is available in context.
    """
    def __init__(
        self,
        *args,
        n_iters: int = 800,
        mask_extra_dilation_px: int = 8,
        wrap_pad_px: int = 64,
        exclude_sky: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_iters = n_iters
        self.mask_extra_dilation_px = mask_extra_dilation_px
        self.wrap_pad_px = wrap_pad_px
        self.exclude_sky = exclude_sky


def _wrap_pad(arr: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return arr
    return np.concatenate([arr[:, -pad:], arr, arr[:, :pad]], axis=1)


def _wrap_crop(arr: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return arr
    return arr[:, pad:-pad]


class PanoramaDepthPatchStage(PipelineStage):
    """
    Replaces depth values inside the foreground-removal region with a Laplacian
    extrapolation from the real surrounding depth, instead of trusting whatever
    the (equirect-unaware-content-fed) depth model produced there. See
    PanoramaDepthPatchConfiguration for the full rationale and
    PanoramaLoraCorrectionStage for the alternative (image-domain) approach this
    is meant to be compared against.

    Input/Output key (SemanticKey.DEPTH) → ContextKey.PANORAMA_DEPTH (in place)

    No-ops (passes depth through unchanged) if PANORAMA_FOREGROUND_MASK isn't in
    context -- i.e. PanoramaForegroundInpaintingStage found nothing to remove.
    """

    @classmethod
    def config_class(cls) -> type[PipelineStageConfiguration]:
        return PanoramaDepthPatchConfiguration

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.DEPTH: ContextKey.PANORAMA_DEPTH,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        (depth_key,) = self._resolved_keys()

        depth = context.input_depth(depth_key)
        if depth is None:
            self.log_warning("No depth in context, skipping")
            return context

        depth_arr = depth.depth.astype(np.float32)
        h, w = depth_arr.shape

        mask_image = context.input_image(ContextKey.PANORAMA_FOREGROUND_MASK)
        if mask_image is None:
            self.log_info("No foreground removal mask in context, nothing to patch")
            return context

        mask_pil = mask_image.image.convert("L")
        if mask_pil.size != (w, h):
            mask_pil = mask_pil.resize((w, h), PILImage.NEAREST)
        hole = np.array(mask_pil) > 127

        if self.config.mask_extra_dilation_px > 0:
            hole = binary_dilation(hole, iterations=self.config.mask_extra_dilation_px)

        if self.config.exclude_sky:
            sky_arr = None
            sky_image = context.input_image(ContextKey.PANORAMA_SKY_MASK)
            if sky_image is not None:
                sky_pil = sky_image.image.convert("L")
                if sky_pil.size != (w, h):
                    sky_pil = sky_pil.resize((w, h), PILImage.NEAREST)
                sky_arr = np.array(sky_pil) > 127
            else:
                sky_obj = context.input_object(ContextKey.PANORAMA_SKY_MASK)
                if sky_obj is not None:
                    sky_arr = np.asarray(sky_obj, dtype=bool)
                    if sky_arr.shape != (h, w):
                        sky_arr = np.array(
                            PILImage.fromarray((sky_arr * 255).astype(np.uint8), mode="L")
                            .resize((w, h), PILImage.NEAREST)
                        ) > 127
            if sky_arr is not None:
                hole &= ~sky_arr

        if not hole.any():
            self.log_info("Nothing left to patch after sky exclusion")
            return context

        self.set_total_tasks(1)
        patch_task = self.create_progress(1, "Patching depth under removed foreground…")

        known = ~hole
        pad = self.config.wrap_pad_px
        depth_padded = _wrap_pad(depth_arr, pad)
        known_padded = _wrap_pad(known, pad)

        diffused_padded = diffuse_heightmap(
            depth_padded, known_padded, n_iters=self.config.n_iters, seed_from='nearest',
        )
        diffused = _wrap_crop(diffused_padded, pad)

        patched = np.where(hole, diffused, depth_arr).astype(np.float32)

        if self.temp is not None:
            PILImage.fromarray((hole * 255).astype(np.uint8), "L").save(
                self.temp / "depth_patch_hole.png"
            )
            Depth(patched.copy()).normalize().save_debug_image(
                self.temp / "panorama_depth_patched.png"
            )

        self.finish_progress(patch_task)

        context.add_depth(depth_key, Depth(patched))
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        # context.depth() would also match ContextKey.PANORAMA_DEPTH written by the
        # earlier "Panorama Depth Calibration" stage (it walks all prior stages),
        # which would make this always look "already done" even before it ever
        # ran, since this stage overwrites the same key in place. has_stage_output()
        # is scoped to this stage's own writes only (see PanoramaDepthCalibrationStage,
        # which has the identical problem for the identical reason).
        (depth_key,) = self._resolved_keys()
        return context.has_stage_output(depth_key)
