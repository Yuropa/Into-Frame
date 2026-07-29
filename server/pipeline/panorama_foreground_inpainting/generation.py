from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.supersampling.image_supersampling import ImageSupersampling
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image
from util.panorama_utils import Panorama
import numpy as np
from PIL import Image as PILImage, ImageFilter
from scipy.ndimage import binary_dilation, binary_closing, label

# Vegetation categories large enough that a mid-ground crop the depth threshold
# doesn't reach (foreground_distance_m) is still a real occluder of the terrain
# behind it. Deliberately narrower than object_typing.categories.VEGETATION_CATEGORIES
# (which also has bush/flower/plant) -- those are exactly the small, individually
# salient blobs foreground_distance_m's own docstring explains SAM fragments into a
# scatter of tiny holes rather than one solid region.
_FOREGROUND_VEGETATION_CATEGORIES = frozenset({"tree", "forest"})

# Built structures with the same property: large, solid, non-terrain vertical
# features (a tower, a distant building, a monument) that can sit well beyond
# foreground_distance_m and still occlude the ground behind them, and whose SAM2
# crop is itself one coherent region rather than the small-blob scatter that
# rules out routing e.g. street furniture through this path. Deliberately
# excludes small OBJECT_CATEGORIES entries (sign, streetlight, fire_hydrant,
# bench, trash_bin, ...) -- those are small enough that if they're really
# occluding the ground they're already caught by foreground_distance_m, and
# unioning them in here would reintroduce the fragmentation problem.
_FOREGROUND_STRUCTURE_CATEGORIES = frozenset({
    "building", "skyscraper", "tower", "monument", "landmark", "statue",
    "lighthouse", "gazebo", "fountain", "wind_turbine", "bridge", "dock",
})


class PanoramaForegroundInpaintingConfiguration(PipelineStageConfiguration):
    """
    Stage-specific config for PanoramaForegroundInpaintingStage.

    foreground_distance_m (float, default 5.0):
        A pixel counts as a removable foreground occluder if its real (metric)
        depth is closer than this many metres. These panoramas are shot from
        ~1-2m off the ground, so this is a direct physical-proximity cutoff:
        "is this standing between the camera and the ground it's trying to
        reconstruct".

        The mask is built straight from the depth map (depth < threshold),
        not from SAM object proposals — a dense per-pixel question like this
        doesn't need an object detector in the loop, and routing it through
        one fragments the result: SAM's automatic mask generator finds masks
        for individually salient blobs (one per flower cluster, one per grass
        tuft) and skips plain texture-less patches between them (bare dirt,
        shadow gaps), leaving a scatter of small holes rather than one solid
        near-field region. Requires PANORAMA_OBJECT_DEPTH (or PANORAMA_DEPTH)
        in context; the stage no-ops without it.

        Separately, crops already classified as one of
        _FOREGROUND_VEGETATION_CATEGORIES ("tree", "forest") or
        _FOREGROUND_STRUCTURE_CATEGORIES ("building", "skyscraper", "tower",
        "monument", "landmark", "statue", "lighthouse", "gazebo", "fountain",
        "wind_turbine", "bridge", "dock") — by the Object Segmentation/
        Classification/Typing stages, which now run before this one
        specifically to feed it — are unioned in regardless of
        foreground_distance_m. A mid-ground tree, tower, or building well
        beyond the near-field cutoff still occludes the terrain behind it, and
        unlike flowers/grass/street-furniture a whole classified crop for one
        of these is large and coherent enough that folding it in doesn't
        reintroduce the small-blob fragmentation problem above.

    min_region_area_px (int, default 200):
        Connected components of the raw depth-threshold mask smaller than this
        are dropped before closing/dilation — denoises stray single-pixel (or
        few-pixel) depth-model glitches that would otherwise show up as tiny
        isolated removal spots.

    closing_radius_px (int, default 12):
        Morphological closing applied to the denoised mask: bridges small real
        gaps (a thin bare patch between two close bushes, sensor noise) into
        one contiguous region before dilation, so the removal mask reads as
        solid rather than pockmarked.

    mask_dilation_px (int, default 15):
        Pixels to dilate the closed foreground mask before removal, giving
        ObjectClear boundary context to blend cleanly.

    guidance_scale (float, default 2.5):
        ObjectClear's removal-strength knob. Higher = stronger removal, lower
        = better background preservation (ObjectClear's own benchmark used 1.0).

    num_inference_steps (int, default 30):
        Diffusion steps for the ObjectClear pass.

    supersample_result (bool, default True):
        ObjectClear runs at a 512px short side. When True, run Swin2SR on its
        output before the final LANCZOS upscale back to panorama resolution —
        same trick PanoramaInpaintingStage uses for Flux — instead of a single
        large LANCZOS stretch, which halves the (typically ~8x) blow-up ratio
        and keeps the removed region close to the surrounding panorama's
        sharpness.
    """
    def __init__(
        self,
        *args,
        foreground_distance_m: float = 5.0,
        min_region_area_px: int = 200,
        closing_radius_px: int = 12,
        mask_dilation_px: int = 15,
        guidance_scale: float = 2.5,
        num_inference_steps: int = 30,
        supersample_result: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.foreground_distance_m = foreground_distance_m
        self.min_region_area_px = min_region_area_px
        self.closing_radius_px = closing_radius_px
        self.mask_dilation_px = mask_dilation_px
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.supersample_result = supersample_result


class PanoramaForegroundInpaintingStage(PipelineStage):
    """
    Removes near-camera foreground occluders from the panorama using metric
    depth thresholding and ObjectClear, producing a clean terrain plate for
    downstream terrain reconstruction.

    Input key  (SemanticKey.PANORAMA) → ContextKey.PANORAMA
    Depth key  (SemanticKey.DEPTH)    → ContextKey.PANORAMA_OBJECT_DEPTH
    Output key (SemanticKey.OUTPUT)   → ContextKey.PANORAMA_TERRAIN

    Unlike PanoramaInpaintingStage, this stage does not classify objects by
    category and does not update the visual ContextKey.PANORAMA — it only
    produces the clean terrain plate consumed by the second PanoramaDepthStage
    → HeightMapStage → terrain reconstruction chain.
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._objectclear = None
        self._samp = None
        self.preferred_device, self.preferred_dtype = preferred_device(DeviceStrategy.MEMORY)

    @classmethod
    def config_class(cls) -> type[PipelineStageConfiguration]:
        return PanoramaForegroundInpaintingConfiguration

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.PANORAMA: ContextKey.PANORAMA,
            SemanticKey.DEPTH: ContextKey.PANORAMA_OBJECT_DEPTH,
            SemanticKey.OUTPUT: ContextKey.PANORAMA_TERRAIN,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        panorama_key, depth_key, output_key = self._resolved_keys()

        panorama = context.input_panorama(panorama_key)
        if panorama is None:
            self.log_warning("No panorama in context, skipping")
            return context

        original_pil = panorama.image.convert("RGB")
        h, w = original_pil.height, original_pil.width

        if self._objectclear is None:
            self._objectclear = InPainting(self.preferred_device, self.preferred_dtype, InPaintingType.OBJECTCLEAR)

        self.set_total_tasks(1)  # remove

        remove_task = self.create_progress(1, "Removing foreground occluders…")

        depth = context.input_depth(depth_key) or context.input_depth(ContextKey.PANORAMA_DEPTH)
        if depth is None:
            self.log_warning("No depth in context, skipping removal")
            terrain_pil = original_pil
        else:
            depth_arr = depth.depth
            if depth_arr.shape != (h, w):
                depth_arr = np.array(
                    PILImage.fromarray(depth_arr, mode="F").resize((w, h), PILImage.NEAREST)
                )

            near_mask = np.isfinite(depth_arr) & (depth_arr < self.config.foreground_distance_m)

            # Denoise the depth-threshold mask alone: drop connected components too
            # small to be a real occluder (stray depth-model glitches). Classified
            # vegetation/structure crops below are unioned in afterward, since
            # they've already been validated by SAM2 + CLIP and don't need this
            # filter.
            if near_mask.any():
                labeled, num_labels = label(near_mask)
                if num_labels > 0:
                    counts = np.bincount(labeled.ravel())
                    small_labels = np.flatnonzero(counts < self.config.min_region_area_px)
                    if small_labels.size:
                        near_mask &= ~np.isin(labeled, small_labels)

            extracted_mask = self._extracted_object_mask(context, h, w)
            if extracted_mask is not None:
                near_mask = near_mask | extracted_mask

            if not near_mask.any():
                self.log_info("Nothing within foreground distance, skipping removal")
                terrain_pil = original_pil
            else:
                # Close small real gaps (a thin bare patch between two close bushes)
                # into one solid region before dilation.
                closed_mask = binary_closing(
                    near_mask,
                    iterations=self.config.closing_radius_px,
                )

                dilated_union = (binary_dilation(
                    closed_mask,
                    iterations=self.config.mask_dilation_px,
                ).astype(np.uint8)) * 255
                mask_pil = PILImage.fromarray(dilated_union, mode="L")

                # Persist the exact dilated removal mask (Image, L mode, 255 = removed/
                # synthetic) to context -- downstream stages can use it to know which
                # pixels are ObjectClear's fabrication rather than real photo content
                # (e.g. a second correction pass, or a depth-domain patch, over just
                # this region; or a confidence/certainty term). Previously this only
                # ever reached disk as a debug dump, never as a context value other
                # stages could read.
                context.add_image(ContextKey.PANORAMA_FOREGROUND_MASK, Image(mask_pil))

                # The occluder-only half of that mask, persisted separately.
                #
                # The full mask above conflates two very different things: real
                # occluders standing between the camera and the ground (the
                # _extracted_object_mask categories -- a tree, a building), and
                # the ground *itself*, swept in wholesale by the
                # foreground_distance_m depth test because in a ground-level
                # capture the ground is within a few metres across the entire
                # lower hemisphere. On the Rainier capture that test covered 48.5%
                # of the panorama and 100% of every row below ~60% height, and the
                # wildflower meadow it erased came back as bare gravel.
                #
                # For geometry that conflation is harmless -- both cases want the
                # occluder gone before height solving. For *colour* it is not:
                # TerrainTextureGenerationStage was left texturing the ground with
                # fabricated gravel when a real photograph of that same ground was
                # sitting in the original panorama the whole time. Splitting the
                # mask lets it take real colour back for the ground while still
                # keeping genuine occluders removed.
                occluder_pil = PILImage.fromarray(
                    (binary_dilation(
                        extracted_mask if extracted_mask is not None else np.zeros_like(near_mask),
                        iterations=self.config.mask_dilation_px,
                    ).astype(np.uint8)) * 255,
                    mode="L",
                )
                context.add_image(ContextKey.PANORAMA_FOREGROUND_OCCLUDER_MASK, Image(occluder_pil))

                if self.temp is not None:
                    mask_pil.save(self.temp / "foreground_mask.png")
                    occluder_pil.save(self.temp / "foreground_occluder_mask.png")

                objectclear_pil = self._objectclear.inpaint(
                    original_pil,
                    mask_pil,
                    temp_path=self.temp,
                    guidance_scale=self.config.guidance_scale,
                    num_inference_steps=self.config.num_inference_steps,
                )

                if self.temp is not None:
                    objectclear_pil.save(self.temp / "panorama_objectclear_raw.png")

                # ObjectClear returns at its native ~512px-short-side inference resolution.
                # Supersample before the final upscale to panorama resolution — halves the
                # LANCZOS stretch ratio (same trick PanoramaInpaintingStage uses for Flux)
                # so the removed region isn't visibly softer than the rest of the panorama.
                if objectclear_pil.size != (w, h):
                    if self.config.supersample_result:
                        if self._samp is None:
                            self._samp = ImageSupersampling(self.preferred_device)
                        objectclear_pil = self._samp.supersample(Image(objectclear_pil), self.temp).image
                    if objectclear_pil.size != (w, h):
                        objectclear_pil = objectclear_pil.resize((w, h), PILImage.LANCZOS)

                if self.temp is not None:
                    objectclear_pil.save(self.temp / "panorama_objectclear_upscaled.png")

                # ObjectClear's own output is a full-panorama resample (it runs inference on
                # the whole image, not just the masked region), so pixels outside the mask
                # drift from the source even though nothing there should change. Feather-
                # composite back onto original_pil, restoring everything but the dilated
                # foreground region — mirrors PanoramaInpaintingStage's compositing.
                feather_radius = max(8, min(w, h) // 100)
                feather_arr = np.array(
                    mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius))
                ).astype(np.float32)[..., np.newaxis] / 255.0
                composited = (
                    np.array(original_pil) * (1.0 - feather_arr)
                    + np.array(objectclear_pil) * feather_arr
                ).astype(np.uint8)
                terrain_pil = PILImage.fromarray(composited)

                if self.temp is not None:
                    terrain_pil.save(self.temp / "panorama_terrain.png")

        self.finish_progress(remove_task)

        context.add_panorama(output_key, Panorama(terrain_pil))
        return context

    def _extracted_object_mask(self, context: PipelineContext, pano_h: int, pano_w: int) -> np.ndarray | None:
        """Union of vegetation + non-terrain structure crop masks (from Object
        Segmentation/Classification/Typing, which run earlier specifically to
        feed this), pasted directly at each crop's own box position in panorama
        pixel space. See _FOREGROUND_VEGETATION_CATEGORIES and
        _FOREGROUND_STRUCTURE_CATEGORIES for the exact class list.

        Object Segmentation/Detection now run directly on the panorama (see
        config.yaml's `keys: input: panorama`), so metadata_{i}["box"] is already
        panorama-pixel-space -- no reprojection needed here any more (an earlier
        version reprojected from the hero photo's perspective space via
        util.panorama_projection.project_mask_to_pano, which needed intrinsics).

        Returns None if there's nothing to add (no objects, or no crop classified
        into one of the unioned categories), so callers can skip the union cheaply.
        """
        object_count = context.input_object(ContextKey.OBJECT_COUNT)
        if not object_count:
            return None

        mask = np.zeros((pano_h, pano_w), dtype=bool)
        found = False
        for idx in range(object_count):
            metadata = context.input_object(f"metadata_{idx}") or {}
            object_class = metadata.get("class")
            if (
                object_class not in _FOREGROUND_VEGETATION_CATEGORIES
                and object_class not in _FOREGROUND_STRUCTURE_CATEGORIES
            ):
                continue

            box = metadata.get("box")
            crop = context.input_image(f"crop_{idx}")
            if not box or crop is None or crop.image.mode != "RGBA":
                continue

            crop_mask = np.array(crop.image.getchannel("A")) > 127
            x, y, w, h = (int(round(v)) for v in box)
            ch, cw = crop_mask.shape
            y0, y1 = max(0, y), min(pano_h, y + ch)
            if y1 <= y0:
                continue
            # Columns wrapped modulo pano_w: a crop whose box straddles the
            # panorama's own left/right seam (rare -- only an object too large
            # for any tile-merge candidate to land on the non-wrapping side,
            # see util/instance_merge.py) still pastes into the right place.
            cols = (x + np.arange(cw)) % pano_w
            rows = np.arange(y0, y1)
            mask[np.ix_(rows, cols)] |= crop_mask[y0 - y: y1 - y, :]
            found = True

        return mask if found else None

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, _, output_key = self._resolved_keys()
        return context.panorama(output_key) is not None

    def model_names(self) -> list[str]:
        names = InPainting.model_names(type=InPaintingType.OBJECTCLEAR)
        if self.config.supersample_result:
            names = names + ImageSupersampling.model_names()
        return names

    def clean_up(self):
        if self._objectclear is not None:
            self._objectclear.close()
            self._objectclear = None
        if self._samp is not None:
            self._samp.close()
            self._samp = None
        super().clean_up()
