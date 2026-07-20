from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey


class PanoramaInpaintingConfiguration(PipelineStageConfiguration):
    """
    Stage-specific config for PanoramaInpaintingStage.

    full_panorama (bool, default False):
        When True, replaces the per-object LaMa+Flux crop loop with a single
        LaMa call and a single Flux call on the full downscaled panorama —
        matching the LayerPano3D research baseline.  Useful for A/B testing
        against the per-object crop approach.

    max_object_area_fraction (float, default 0.15):
        Objects whose mask covers more than this fraction of the total panorama
        pixels are skipped for inpainting — they are too large to fill plausibly.
        Still classified and logged; just not removed.

    supersample_inpaint (bool, default True):
        When True, runs Swin2SR on the Flux output before LANCZOS upscaling it
        back to the panorama resolution. Reduces the LANCZOS stretch ratio from
        4x to 2x, producing sharper fills. Only applies when Flux ran at a
        downscaled resolution (i.e. the panorama is larger than flux_max).

    use_depth_filter (bool, default True):
        Whether to run DepthObjectFilter on SAM masks before classification.
        Requires PANORAMA_OBJECT_DEPTH in context (set by PanoramaDepthStage
        before this stage).

    depth_filter_distance_m (float, default 5.0):
        A mask is kept as a removable foreground occluder if its median real
        (metric) depth is closer than this many metres. These panoramas are
        shot from ~1-2m off the ground, so this is a direct physical-proximity
        cutoff, not a relative/local one — see DepthObjectFilter for why a
        local-relative signal doesn't work here (it favours rough distant
        terrain over flat close terrain, backwards from what "foreground"
        means).
    """
    def __init__(self, *args, full_panorama: bool = False, max_object_area_fraction: float = 0.15, supersample_inpaint: bool = True, use_depth_filter: bool = True, depth_filter_distance_m: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_panorama = full_panorama
        self.max_object_area_fraction = max_object_area_fraction
        self.supersample_inpaint = supersample_inpaint
        self.use_depth_filter = use_depth_filter
        self.depth_filter_distance_m = depth_filter_distance_m
from pipeline.segmentation.image_segmentation import ImageSeg
from pipeline.segmentation.depth_filter import DepthObjectFilter
from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.object_typing.image_clip_classifier import ImageClipClassifier
from pipeline.object_typing.categories import OBJECT_CATEGORIES as _OBJECT_CATEGORIES
from pipeline.captioning.image_captioning import ImageCaptioning
from pipeline.supersampling.image_supersampling import ImageSupersampling
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image
from util.json_utils import write_json
from util.panorama_utils import Panorama
import colorsys
import numpy as np
from PIL import Image as PILImage, ImageFilter
from scipy.ndimage import binary_dilation

# Environment labels that indicate natural terrain. When CLIP's best environment
# match is one of these and its score is competitive with the winning object label,
# the mask is terrain — not a foreground object — and should not be inpainted.
_TERRAIN_ENV_LABELS: frozenset[str] = frozenset({
    "mountain", "cliff", "trail", "ground", "field", "sand", "snow", "ice", "dirt", "mud",
})
# How much the object score may exceed the terrain environment score before we
# stop protecting it (CLIP similarity scores, typically 0.20–0.35).
_TERRAIN_PROTECTION_MARGIN: float = 0.03


def _environment_prompt(scene_caption: str, removed_categories: set[str] | None = None) -> tuple[str, list[str]]:
    """
    Strip object names from the scene caption so Flux doesn't regenerate what
    was just removed. Returns (cleaned_caption, list_of_stripped_words).

    When removed_categories is provided, only those specific category keys are
    stripped (used for the LaMa-output caption, where vegetation/background words
    should be kept because they describe the revealed background). When None, all
    OBJECT_CATEGORIES keys are stripped (used for the original scene caption).
    """
    from pipeline.object_typing.categories import OBJECT_CATEGORIES, VEGETATION_CATEGORIES
    import re

    if removed_categories is not None:
        keys_to_strip = removed_categories
    else:
        # Strip everything except vegetation — vegetation is background, not foreground
        keys_to_strip = set(OBJECT_CATEGORIES.keys()) - VEGETATION_CATEGORIES

    stop_words = set()
    for key in keys_to_strip:
        stop_words.add(key.replace("_", " "))
        stop_words.add(key.replace("_", ""))

    result = scene_caption
    stripped = []
    for word in sorted(stop_words, key=len, reverse=True):
        new = re.sub(rf"\b{re.escape(word)}s?\b", "", result, flags=re.IGNORECASE)
        if new != result:
            stripped.append(word)
        result = new
    result = re.sub(r"\s{2,}", " ", result).strip(" ,;.")
    return result or "outdoor scene", stripped


def _draw_extraction_overlay(
    image: PILImage.Image,
    masks: list[np.ndarray],
    opacity: float = 0.6,
) -> PILImage.Image:
    """Composite each extracted mask over `image` with a distinct hue-spaced color."""
    canvas = np.array(image.convert("RGB"), dtype=np.float32)
    n = len(masks)
    for i, mask in enumerate(masks):
        hue = i / n if n > 1 else 0.5
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
        color = np.array([r * 255, g * 255, b * 255], dtype=np.float32)
        alpha = np.clip(mask, 0.0, 1.0)[..., np.newaxis] * opacity
        canvas = canvas * (1.0 - alpha) + color * alpha
    return PILImage.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))

class PanoramaInpaintingStage(PipelineStage):
    """
    Segments the panorama, extracts and classifies each crop, then inpaints
    the detected regions in two phases (LaMa structural fill → Flux perceptual
    refinement).

    Two panoramas are written:

      ContextKey.PANORAMA         — foreground objects removed; environment pixels
                                    kept at original (for lighting / asset generation)
      ContextKey.PANORAMA_TERRAIN — all detected regions removed (clean ground plane
                                    for PanoramaDepthStage → heightmap → terrain mesh)

    Input key  (SemanticKey.PANORAMA) → ContextKey.PANORAMA  (Panorama)
    Output key (SemanticKey.OUTPUT)   → ContextKey.PANORAMA  (Panorama, objects removed)

    Dynamic context keys per detected crop (index i):
      crop_{i}     → Image
      metadata_{i} → {"box": [...], "score": float, "class": str}

    Also writes ContextKey.OBJECT_COUNT (total crops found).
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._seg = None
        self._classifier = None
        self._captioner = None
        self._samp = None
        self._flux_inpainter = None
        self.preferred_device, self.preferred_dtype = preferred_device(DeviceStrategy.MEMORY)

    @classmethod
    def config_class(cls) -> type[PipelineStageConfiguration]:
        return PanoramaInpaintingConfiguration

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.PANORAMA: ContextKey.PANORAMA,
            SemanticKey.OUTPUT: ContextKey.PANORAMA,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        panorama_key, output_key = self._resolved_keys()

        panorama = context.input_panorama(panorama_key)
        if panorama is None:
            self.log_warning("No panorama in context, skipping")
            return context

        original_pil = panorama.image.convert("RGB")
        original_array = np.array(original_pil)
        h, w = original_pil.height, original_pil.width

        if self._seg is None:
            self._seg = ImageSeg(self.preferred_device)
        if self._classifier is None:
            self._classifier = ImageClipClassifier(self.preferred_device)

        # Dilate the mask so LaMa/Flux have enough boundary material to blend into.
        lama_dilation = 25

        object_masks = []  # foreground-object masks for the visual panorama composite
        all_masks = []     # every detected mask, for the extraction overlay
        removed_object_types: set[str] = set()
        manifest = {"passes": [{"pass": 1, "crops": []}]}
        fraction_advanced = 0.0
        self.set_total_tasks(3)  # seg + classify + inpaint

        seg_task = self.create_progress(2, "Segmenting…")
        result = self._seg.segment(Image(original_pil), self.temp, on_progress=self.make_progress_callback(seg_task))
        self.advance_progress(seg_task)
        self.finish_progress(seg_task)
        fraction_advanced += 1.0 / self.total_tasks

        sam_detected = result.length
        depth_filtered_out = 0
        if self.config.use_depth_filter:
            depth = context.input_depth(ContextKey.PANORAMA_OBJECT_DEPTH) or context.input_depth(ContextKey.PANORAMA_DEPTH)
            if depth is not None:
                result = DepthObjectFilter().filter(
                    result, depth,
                    distance_threshold_m=self.config.depth_filter_distance_m,
                )
                depth_filtered_out = sam_detected - result.length
                if depth_filtered_out:
                    self.log_info(f"Depth filter removed {depth_filtered_out}/{sam_detected} background mask(s)")

        manifest["passes"][0]["sam_detected"] = sam_detected
        manifest["passes"][0]["depth_filtered_out"] = depth_filtered_out

        terrain_pil = original_pil

        if result.length == 0:
            self.log_info("Nothing found, skipping inpainting")
        else:
            self.log_info(f"{result.length} object(s) found")

            # Classify and save crops; track (mask, box) pairs for inpainting
            pass_objects = []
            classify_task = self.create_progress(result.length, "Classifying…")
            for i, crop in enumerate(result.masked_images(Image(original_pil))):
                box = [float(x) for x in crop.box]

                mask_array = np.array(crop.mask).astype(np.float32) / 255.0
                crop_image = crop.image  # masked RGBA crop, equirectangular space

                obj_type, clip_confidence, _, criteria = self._classifier.classify_with_details(
                    crop_image,
                    scene_image=Image(original_pil),
                    box=box,
                )

                # When CLIP is uncertain, fall back to "other" if SAM was confident.
                # Small or distorted panorama crops rarely give CLIP enough signal to
                # beat the 0.1 threshold, but a high SAM score means something real is there.
                inpaint_type = obj_type
                if obj_type == "indeterminate" and crop.score >= 0.7:
                    inpaint_type = "other"

                context.add_image(f"crop_{i}", crop_image)
                context.add_object(f"metadata_{i}", {
                    "box":        box,
                    "score":      float(crop.score),
                    "class":      inpaint_type,
                    "confidence": round(clip_confidence, 4),
                })
                if self.temp is not None:
                    crop_image.image.save(self.temp / f"crop_{i}.png")
                    crop_caption = f"class: {inpaint_type} (clip: {obj_type})\nconfidence: {clip_confidence:.3f}\nscore: {crop.score:.3f}\nbox: {[round(x, 1) for x in box]}\n"
                    (self.temp / f"crop_{i}.txt").write_text(crop_caption)
                if inpaint_type in _OBJECT_CATEGORIES:
                    mask_fraction = float(mask_array.sum()) / (h * w)
                    if mask_fraction > self.config.max_object_area_fraction:
                        self.log_info(f"  crop_{i}: skipping inpaint, mask too large ({mask_fraction:.1%} of panorama)")
                    else:
                        # Protect natural terrain: if the best environment match is a terrain
                        # category and its score is within the margin, the "object" label is
                        # likely a misclassification of a mountain/cliff/ground feature.
                        best_env_label = criteria.get("best_env_label", "")
                        best_env_score = criteria.get("best_env_score", 0.0)
                        best_obj_score = criteria.get("best_obj_score", 0.0)
                        if best_env_label in _TERRAIN_ENV_LABELS and best_env_score >= best_obj_score - _TERRAIN_PROTECTION_MARGIN:
                            self.log_info(f"  crop_{i}: skipping inpaint, terrain env match ({best_env_label!r}, env={best_env_score:.3f} obj={best_obj_score:.3f})")
                        else:
                            object_masks.append(mask_array)
                            pass_objects.append((mask_array, box))
                            removed_object_types.add(inpaint_type)

                manifest["passes"][0]["crops"].append({
                    "index": i,
                    "class": inpaint_type,
                    "clip_class": obj_type,
                    "score": round(float(crop.score), 3),
                    "box":   [round(x, 1) for x in box],
                })

                self.log_info(f"  crop_{i}: {inpaint_type}  (clip={obj_type}, conf={clip_confidence:.2f}, sam={crop.score:.2f})")
                self.advance_progress(classify_task)
            self.finish_progress(classify_task)
            fraction_advanced += 1.0 / self.total_tasks
            all_masks.extend(m for m, _ in pass_objects)

            # Two-phase inpainting — two modes selectable via config.full_panorama:
            #
            #   Per-object (default): accumulating LaMa per crop → Flux per crop.
            #     Each LaMa call benefits from prior removals; Flux stays at native
            #     resolution on a tight crop. More calls, sharper per-object results.
            #
            #   Full-panorama: one LaMa call → one Flux call on the whole image.
            #     Matches the LayerPano3D research baseline. Fewer calls; Flux sees
            #     global context but runs on a downscaled panorama.
            inpaint_task = self.create_progress(len(pass_objects) * 2, "Inpainting…")
            scene_caption = context.input_object(ContextKey.INPUT_CAPTION) or ""
            caption, _ = _environment_prompt(scene_caption)

            flux_max = 1024

            if self.config.full_panorama:
                terrain_pil = self._inpaint_full_panorama(
                    context, original_pil, pass_objects, caption,
                    w, h, flux_max, lama_dilation, inpaint_task,
                    removed_categories=removed_object_types,
                )
            else:
                terrain_pil = self._inpaint_per_object(
                    context, original_pil, pass_objects, caption,
                    w, h, flux_max, lama_dilation, inpaint_task,
                    removed_categories=removed_object_types,
                )

            fraction_advanced += 1.0 / self.total_tasks

        # Snap main task to 1.0 (only needed when nothing was found and
        # classify/inpaint tasks were skipped).
        remaining = 1.0 - fraction_advanced
        if remaining > 1e-9:
            self.progress.advance(self.main_task, remaining)

        crop_count = len(all_masks)
        context.add_object(ContextKey.OBJECT_COUNT, crop_count)

        if self.temp is not None:
            if all_masks:
                overlay = _draw_extraction_overlay(original_pil, all_masks)
                overlay.save(self.temp / "extraction_overlay.png")

            manifest["total_crops"] = crop_count
            manifest["caption"] = context.input_object(ContextKey.INPUT_CAPTION) or ""
            manifest["passes_run"] = 1
            manifest["image_size"] = {"width": w, "height": h}
            with open(self.temp / "manifest.json", "w") as f:
                write_json(manifest, f)

        # Terrain panorama: the result after inpainting (clean ground plane).
        context.add_panorama(ContextKey.PANORAMA_TERRAIN, Panorama(terrain_pil))

        # Visual panorama: original with only foreground-object regions filled from
        # the terrain image; environment features are left at their original pixels.
        if object_masks:
            obj_union = np.zeros((h, w), dtype=np.float32)
            for m in object_masks:
                obj_union = np.maximum(obj_union, m)
            obj_mask_pil = PILImage.fromarray((obj_union * 255).astype(np.uint8), mode="L")
            visual_feather_radius = max(8, min(w, h) // 100)
            obj_feathered = np.array(obj_mask_pil.filter(ImageFilter.GaussianBlur(radius=visual_feather_radius))).astype(np.float32)[..., np.newaxis] / 255.0
            visual_composited = (original_array * (1.0 - obj_feathered) + np.array(terrain_pil) * obj_feathered).astype(np.uint8)
            visual_pil = PILImage.fromarray(visual_composited)
            if self.temp is not None:
                visual_pil.save(self.temp / "panorama_visual.png")
            context.add_panorama(output_key, Panorama(visual_pil))

        return context

    # ── Inpainting helpers ────────────────────────────────────────────────────

    def _supersample_flux(self, pil: PILImage.Image, target_w: int, target_h: int) -> PILImage.Image:
        """Swin2SR 2x the Flux output before LANCZOS upscaling.

        Halves the LANCZOS stretch ratio (e.g. 4x → 2x) for a sharper result.
        Only runs when the image actually needs upscaling and supersample_inpaint
        is enabled in config. Returns the input unchanged if already at target size."""
        if pil.width == target_w and pil.height == target_h:
            return pil
        if self.config.supersample_inpaint:
            if self._samp is None:
                self._samp = ImageSupersampling(self.preferred_device)
            pil = self._samp.supersample(Image(pil), self.temp).image
        if pil.width != target_w or pil.height != target_h:
            pil = pil.resize((target_w, target_h), PILImage.LANCZOS)
        return pil

    def _caption_from_lama(self, lama_arr: np.ndarray, removed_categories: set[str] | None = None) -> tuple[str, dict]:
        if self._captioner is None:
            self._captioner = ImageCaptioning(self.preferred_device)
        # Prompt BLIP to focus on the outdoor background rather than a generic scene description
        raw = self._captioner.caption(Image(PILImage.fromarray(lama_arr)), prompt="the outdoor background shows")
        caption, stripped = _environment_prompt(raw, removed_categories=removed_categories)
        self.log_info(f"Fill caption (raw):       {raw!r}")
        self.log_info(f"Fill caption (stripped):  {stripped}")
        self.log_info(f"Fill caption (final):     {caption!r}")
        debug = {"raw_caption": raw, "stripped_words": stripped, "final_caption": caption}
        return caption, debug

    def _inpaint_full_panorama(
        self,
        context,
        original_pil: PILImage.Image,
        pass_objects: list,
        caption: str,
        w: int,
        h: int,
        flux_max: int,
        lama_dilation: int,
        inpaint_task,
        removed_categories: set[str] | None = None,
    ) -> PILImage.Image:
        """
        Research-baseline mode: one LaMa call on the full panorama with the
        union of all detected masks, then one Flux call on a downscaled version.
        """
        # Union mask of all detected objects
        union_mask = np.zeros((h, w), dtype=np.float32)
        for mask_array, _ in pass_objects:
            union_mask = np.maximum(union_mask, mask_array)

        # Dilate on the full panorama (acceptable cost for a single pass)
        dilated_union = binary_dilation(
            union_mask > 0.5,
            iterations=lama_dilation,
        ).astype(np.float32)

        union_mask_pil    = PILImage.fromarray((dilated_union * 255).astype(np.uint8), mode="L")

        # Phase 1: LaMa — full panorama
        self.log_info(f"  LaMa: full panorama ({w}×{h}px)")
        lama_inpainter = InPainting(self.preferred_device, self.preferred_dtype, InPaintingType.LAMA)
        lama_pil = lama_inpainter.inpaint(original_pil, union_mask_pil, temp_path=self.temp)
        lama_inpainter.close()

        lama_arr = np.array(lama_pil)
        current_arr = np.array(original_pil)
        current_arr[dilated_union > 0.5] = lama_arr[dilated_union > 0.5]

        if self.temp is not None:
            lama_pil.save(self.temp / "inpaint_pass_0_lama.png")
            PILImage.fromarray(current_arr).save(self.temp / "inpaint_pass_1_lama_composite.png")

        # Advance LaMa half of the progress bar
        for _ in pass_objects:
            self.advance_progress(inpaint_task)

        caption, caption_debug = self._caption_from_lama(current_arr, removed_categories=removed_categories)

        # Phase 2: Flux — scale panorama to fit within height×width caps, preserving aspect ratio.
        # We cap height at flux_max and width at 2×flux_max to keep the 2:1 equirectangular ratio
        # at full resolution rather than downscaling based on the longest side (which would halve
        # the height of a standard panorama and give Flux only 512px to work with vertically).
        flux_input = PILImage.fromarray(current_arr)
        flux_mask  = union_mask_pil

        flux_max_h = flux_max        # 1024
        flux_max_w = flux_max * 2    # 2048

        if h > flux_max_h or w > flux_max_w:
            scale   = min(flux_max_h / h if h > flux_max_h else 1.0,
                          flux_max_w / w if w > flux_max_w else 1.0)
            flux_w  = max(16, (int(w * scale) // 16) * 16)
            flux_h  = max(16, (int(h * scale) // 16) * 16)
            flux_input_s = flux_input.resize((flux_w, flux_h), PILImage.LANCZOS)
            flux_mask_s  = flux_mask.resize((flux_w, flux_h),  PILImage.NEAREST)
        else:
            flux_input_s, flux_mask_s = flux_input, flux_mask
            flux_w, flux_h = w, h

        self.log_info(f"  Flux: full panorama input={w}×{h} → flux={flux_w}×{flux_h}px")
        flux_guidance  = 30.0
        flux_steps     = 50
        self._flux_inpainter = InPainting(self.preferred_device, self.preferred_dtype, InPaintingType.FLUX)
        try:
            flux_pil = self._flux_inpainter.inpaint(
                flux_input_s,
                flux_mask_s,
                temp_path=self.temp,
                prompt=caption,
                num_inference_steps=flux_steps,
                guidance_scale=flux_guidance,
            )
        finally:
            self._flux_inpainter.close()
            self._flux_inpainter = None

        if self.temp is not None:
            import json
            debug = {
                "mode": "full_panorama",
                "panorama_resolution": {"w": w, "h": h},
                "flux_resolution": {"w": flux_w, "h": flux_h},
                "flux_params": {"guidance_scale": flux_guidance, "num_inference_steps": flux_steps},
                **caption_debug,
            }
            (self.temp / "inpaint_debug.json").write_text(json.dumps(debug, indent=2))

        if self.temp is not None:
            flux_pil.save(self.temp / "inpaint_pass_2_flux_raw.png")

        flux_pil = self._supersample_flux(flux_pil, w, h)
        flux_arr = np.array(flux_pil)

        # Feathered composite — blend against current_arr (LaMa-filled, object already
        # removed) rather than original_pil (still has the object). The transition zone
        # then blends two "no-object" images instead of showing a ghost of the removed
        # content at the seam. Radius scales with the shorter panorama dimension.
        feather_radius = max(8, min(w, h) // 100)
        feather_pil = PILImage.fromarray((union_mask * 255).astype(np.uint8), mode="L")
        feathered = np.array(feather_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius))).astype(np.float32)[..., np.newaxis] / 255.0
        next_terrain_arr = (current_arr * (1.0 - feathered) + flux_arr * feathered).astype(np.uint8)

        if self.temp is not None:
            flux_pil.save(self.temp / "inpaint_pass_2_flux.png")
            feather_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius)).save(self.temp / "inpaint_pass_2_feather_mask.png")

        for _ in pass_objects:
            self.advance_progress(inpaint_task)

        terrain_pil = PILImage.fromarray(next_terrain_arr)
        if self.temp is not None:
            terrain_pil.save(self.temp / "inpaint_pass_3_terrain.png")
        self.finish_progress(inpaint_task)
        return terrain_pil

    def _inpaint_per_object(
        self,
        context,
        original_pil: PILImage.Image,
        pass_objects: list,
        caption: str,
        w: int,
        h: int,
        flux_max: int,
        lama_dilation: int,
        inpaint_task,
        removed_categories: set[str] | None = None,
    ) -> PILImage.Image:
        """
        Per-object mode: accumulating LaMa per crop, then Flux per crop.
        """
        lama_states = []
        current_arr = np.array(original_pil)

        lama_inpainter = InPainting(self.preferred_device, self.preferred_dtype, InPaintingType.LAMA)
        for lama_idx, (mask_array, box) in enumerate(pass_objects):
            bx, by, bw, bh = box
            left   = max(0, int(bx))
            top    = max(0, int(by))
            right  = min(w, int(bx + bw))
            bottom = min(h, int(by + bh))

            region_mask = mask_array[top:bottom, left:right]
            if region_mask.sum() == 0:
                lama_states.append((mask_array, box, None, None, None, None, None))
                self.advance_progress(inpaint_task)
                continue

            pad     = max(64, int(max(bw, bh) * 1.0))
            crop_y0 = max(0, top    - pad)
            crop_y1 = min(h, bottom + pad)
            crop_x0 = max(0, left   - pad)
            crop_x1 = min(w, right  + pad)

            dilated_crop = binary_dilation(
                mask_array[crop_y0:crop_y1, crop_x0:crop_x1] > 0.5,
                iterations=lama_dilation,
            ).astype(np.float32)

            skip_lama = min(bw, bh) < 80
            if skip_lama:
                self.log_info(f"  LaMa: crop_{lama_idx} skipped ({int(bw)}×{int(bh)}px, too small → Flux only)")
            else:
                lama_crop = PILImage.fromarray(current_arr[crop_y0:crop_y1, crop_x0:crop_x1])
                mask_crop = PILImage.fromarray((dilated_crop * 255).astype(np.uint8), mode="L")

                self.log_info(f"  LaMa: crop_{lama_idx} ({crop_x1 - crop_x0}×{crop_y1 - crop_y0}px)")
                lama_pil = lama_inpainter.inpaint(lama_crop, mask_crop, temp_path=self.temp)
                lama_arr = np.array(lama_pil)

                composited = current_arr[crop_y0:crop_y1, crop_x0:crop_x1].copy()
                composited[dilated_crop > 0.5] = lama_arr[dilated_crop > 0.5]
                current_arr[crop_y0:crop_y1, crop_x0:crop_x1] = composited

                if self.temp is not None:
                    lama_pil.save(self.temp / f"inpaint_{lama_idx}_lama_crop.png")
                    PILImage.fromarray(current_arr).save(self.temp / f"inpaint_{lama_idx}_lama_panorama.png")

            lama_states.append((mask_array, box, dilated_crop, crop_y0, crop_y1, crop_x0, crop_x1))
            self.advance_progress(inpaint_task)

        lama_inpainter.close()

        caption, caption_debug = self._caption_from_lama(current_arr, removed_categories=removed_categories)

        next_terrain_arr = np.array(original_pil)
        valid_states = [s for s in lama_states if s[2] is not None]
        flux_guidance = 30.0
        flux_steps = 50
        crop_debug_entries: list[dict] = []

        if valid_states:
            self._flux_inpainter = InPainting(self.preferred_device, self.preferred_dtype, InPaintingType.FLUX)
            lama_count = len(pass_objects)
            try:
                for flux_idx, (mask_array, box, dilated_crop, crop_y0, crop_y1, crop_x0, crop_x1) in enumerate(valid_states):
                    global_idx = lama_count + flux_idx
                    bx, by, bw, bh = box
                    left   = max(0, int(bx))
                    top    = max(0, int(by))
                    right  = min(w, int(bx + bw))
                    bottom = min(h, int(by + bh))

                    region_mask = mask_array[top:bottom, left:right]
                    if region_mask.sum() == 0:
                        self.advance_progress(inpaint_task)
                        continue

                    crop_h = crop_y1 - crop_y0
                    crop_w = crop_x1 - crop_x0

                    lama_crop = PILImage.fromarray(current_arr[crop_y0:crop_y1, crop_x0:crop_x1])
                    mask_crop = PILImage.fromarray((dilated_crop * 255).astype(np.uint8), mode="L")

                    flux_cw, flux_ch = crop_w, crop_h
                    if crop_w > flux_max or crop_h > flux_max:
                        scale     = flux_max / max(crop_w, crop_h)
                        flux_cw   = max(16, (int(crop_w * scale) // 16) * 16)
                        flux_ch   = max(16, (int(crop_h * scale) // 16) * 16)
                        lama_crop = lama_crop.resize((flux_cw, flux_ch), PILImage.LANCZOS)
                        mask_crop = mask_crop.resize((flux_cw, flux_ch), PILImage.NEAREST)

                    self.log_info(f"  Flux: crop_{flux_idx} input={crop_w}×{crop_h} → flux={flux_cw}×{flux_ch}px")
                    flux_pil = self._flux_inpainter.inpaint(
                        lama_crop,
                        mask_crop,
                        temp_path=self.temp,
                        prompt=caption,
                        num_inference_steps=flux_steps,
                        guidance_scale=flux_guidance,
                    )
                    crop_debug_entries.append({
                        "crop_idx": flux_idx,
                        "crop_resolution": {"w": crop_w, "h": crop_h},
                        "flux_resolution": {"w": flux_cw, "h": flux_ch},
                    })

                    if self.temp is not None:
                        flux_pil.save(self.temp / f"inpaint_{global_idx}_flux_crop_raw.png")

                    flux_pil = self._supersample_flux(flux_pil, crop_w, crop_h)
                    flux_crop_arr = np.array(flux_pil)

                    feather_radius = max(4, min(right - left, bottom - top) // 50)
                    region_mask_pil = PILImage.fromarray((region_mask * 255).astype(np.uint8), mode="L")
                    feathered = np.array(region_mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius))).astype(np.float32)[..., np.newaxis] / 255.0

                    orig_region = current_arr[top:bottom, left:right]
                    flux_region = flux_crop_arr[top - crop_y0 : bottom - crop_y0,
                                                left - crop_x0 : right  - crop_x0]
                    next_terrain_arr[top:bottom, left:right] = (orig_region * (1.0 - feathered) + flux_region * feathered).astype(np.uint8)

                    if self.temp is not None:
                        flux_pil.save(self.temp / f"inpaint_{global_idx}_flux_crop.png")
                        region_mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius)).save(self.temp / f"inpaint_{global_idx}_flux_feather_mask.png")
                        PILImage.fromarray(next_terrain_arr).save(self.temp / f"inpaint_{global_idx}_flux_panorama.png")

                    self.advance_progress(inpaint_task)
            finally:
                self._flux_inpainter.close()
                self._flux_inpainter = None

        # Advance skipped Flux slots (objects with empty masks skipped above)
        for _ in [s for s in lama_states if s[2] is None]:
            self.advance_progress(inpaint_task)

        terrain_pil = PILImage.fromarray(next_terrain_arr)
        if self.temp is not None:
            terrain_pil.save(self.temp / "panorama_terrain.png")
            import json
            debug = {
                "mode": "per_object",
                "panorama_resolution": {"w": w, "h": h},
                "flux_params": {"guidance_scale": flux_guidance, "num_inference_steps": flux_steps},
                "crops": crop_debug_entries if valid_states else [],
                **caption_debug,
            }
            (self.temp / "inpaint_debug.json").write_text(json.dumps(debug, indent=2))
        self.finish_progress(inpaint_task)
        return terrain_pil

    def has_expected_output(self, context: PipelineContext) -> bool:
        count = context.object(ContextKey.OBJECT_COUNT)
        if count is None:
            # No OBJECT_COUNT anywhere upstream means Object Segmentation is
            # disabled (permanent, not "pending") -- nothing to classify and
            # never will be; see the identical reasoning in
            # PanoramaObjectClassificationStage.has_expected_output. Still
            # gated on terrain_ready below, which is unrelated to object count.
            all_classified = True
        else:
            all_classified = all(
                (context.object(f"metadata_{i}") or {}).get("class") is not None
                for i in range(count)
            )
        terrain_ready = context.panorama(ContextKey.PANORAMA_TERRAIN) is not None
        return all_classified and terrain_ready

    def model_names(self) -> list[str]:
        names = (
            ImageSeg.model_names()
            + InPainting.model_names(type=InPaintingType.LAMA)
            + InPainting.model_names(type=InPaintingType.FLUX)
            + ImageClipClassifier.model_names()
        )
        if self.config.supersample_inpaint:
            names = names + ImageSupersampling.model_names()
        return names

    def clean_up(self):
        if self._flux_inpainter is not None:
            self._flux_inpainter.close()
            self._flux_inpainter = None
        if self._seg is not None:
            self._seg.close()
            self._seg = None
        self._classifier = None
        if self._captioner is not None:
            self._captioner.close()
            self._captioner = None
        if self._samp is not None:
            self._samp.close()
            self._samp = None
        super().clean_up()  # calls torch.cuda.empty_cache() after refs are dropped
