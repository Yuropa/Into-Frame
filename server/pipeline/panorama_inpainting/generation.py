from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.segmentation.image_segmentation import ImageSeg
from pipeline.segmentation.depth_filter import DepthObjectFilter
from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.object_typing.image_clip_classifier import ImageClipClassifier
from pipeline.captioning.image_captioning import ImageCaptioning
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image
from util.json_utils import write_json
from util.panorama_utils import Panorama
import colorsys
import numpy as np
from PIL import Image as PILImage, ImageFilter
from scipy.ndimage import binary_dilation


def _environment_prompt(scene_caption: str) -> str:
    """
    Strip object names from the scene caption so Flux doesn't regenerate what
    was just removed. Keeps location/environment words (city names, terrain
    descriptors) and drops specific object category terms.
    """
    from pipeline.object_typing.categories import OBJECT_CATEGORIES
    import re

    stop_words = set()
    for key in OBJECT_CATEGORIES:
        stop_words.add(key.replace("_", " "))
        stop_words.add(key.replace("_", ""))

    # Remove matched words, collapse whitespace
    result = scene_caption
    for word in sorted(stop_words, key=len, reverse=True):
        result = re.sub(rf"\b{re.escape(word)}s?\b", "", result, flags=re.IGNORECASE)
    result = re.sub(r"\s{2,}", " ", result).strip(" ,;.")
    return result or "outdoor scene"


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
    Iteratively segments the panorama, extracts and classifies each crop, inpaints
    the detected regions, then re-segments the result — repeating until nothing is
    found (up to max_passes=10). This ensures objects occluded behind others are
    also captured in later passes.

    Each pass runs one LaMa call with the full panorama as input (for colour/texture
    context), then composites the result back per object using only each object's
    bounding-box region — so the rest of the image is never overwritten.
    Two panoramas are derived from the accumulated result:

      ContextKey.PANORAMA         — foreground objects removed only; environment
                                    features kept at original pixels (for lighting /
                                    asset generation)
      ContextKey.PANORAMA_TERRAIN — all detected regions removed across all passes
                                    (clean ground plane for PanoramaDepthStage →
                                    heightmap → terrain mesh)

    Input key  (SemanticKey.PANORAMA) → ContextKey.PANORAMA  (Panorama)
    Output key (SemanticKey.OUTPUT)   → ContextKey.PANORAMA  (Panorama, objects removed)

    Dynamic context keys per detected crop (index i, accumulated across passes):
      crop_{i}     → Image
      metadata_{i} → {"box": [...], "score": float, "class": str, "type": str}

    Also writes ContextKey.OBJECT_COUNT (total crops across all passes).
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._seg = None
        self._classifier = None
        self._captioner = None
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)

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
        # 50 px matches the research reference; gives clean seams without visible edges.
        lama_dilation = 50

        # Iterative extract-and-inpaint: segment → extract crops → inpaint → repeat
        # until segmentation finds nothing. terrain_pil tracks the progressively
        # cleaned image; each pass removes whatever was found in that pass.
        terrain_pil = original_pil
        all_object_masks = []    # accumulated across passes for the visual panorama
        all_extracted_masks = [] # every detected mask, for the extraction overlay
        global_idx = 0
        manifest = {"passes": []}
        max_passes = 1
        initial_passes = 1
        tasks_per_pass = 3  # seg + classify + inpaint
        passes_allocated = initial_passes
        fraction_advanced = 0.0
        self.set_total_tasks(passes_allocated * tasks_per_pass)

        for pass_num in range(max_passes):
            seg_task = self.create_progress(2, f"Segmenting (pass {pass_num + 1})...")
            result = self._seg.segment(Image(terrain_pil), self.temp, on_progress=self.make_progress_callback(seg_task))
            self.advance_progress(seg_task)
            self.finish_progress(seg_task)
            fraction_advanced += 1.0 / self.total_tasks

            sam_detected = result.length
            depth_filtered_out = 0
            depth = context.input_depth(ContextKey.PANORAMA_DEPTH)
            if depth is not None:
                result = DepthObjectFilter().filter(result, depth)
                depth_filtered_out = sam_detected - result.length
                if depth_filtered_out:
                    self.log_info(f"Pass {pass_num + 1}: depth filter removed {depth_filtered_out}/{sam_detected} background mask(s)")

            pass_entry = {
                "pass": pass_num + 1,
                "sam_detected": sam_detected,
                "depth_filtered_out": depth_filtered_out,
                "crops": [],
            }

            if result.length == 0:
                self.log_info(f"Pass {pass_num + 1}: nothing found, stopping")
                manifest["passes"].append(pass_entry)
                break

            # If this is the last allocated pass and there are still objects, extend
            # the budget before classify/inpaint consume the remaining fraction.
            # new_total_tasks = remaining_tasks / (1 - fraction_advanced) keeps the
            # math correct so future tasks fill exactly the remaining bar fraction.
            if pass_num == passes_allocated - 1 and pass_num + 1 < max_passes:
                new_allocated = min(passes_allocated + initial_passes, max_passes)
                remaining_tasks = new_allocated * tasks_per_pass - (pass_num * tasks_per_pass + 1)
                remaining_fraction = 1.0 - fraction_advanced
                if remaining_tasks > 0 and remaining_fraction > 1e-9:
                    self.set_total_tasks(remaining_tasks / remaining_fraction)
                passes_allocated = new_allocated

            self.log_info(f"Pass {pass_num + 1}: {result.length} object(s) found")

            # Classify and save crops; track (mask, box) pairs for inpainting
            pass_objects = []  # (mask_array, box) for every crop this pass
            classify_task = self.create_progress(result.length, f"Classifying (pass {pass_num + 1})...")
            for i, crop in enumerate(result.masked_images(Image(terrain_pil))):
                obj_type, cls = self._classifier.classify(crop.cropped_image)
                idx = global_idx + i
                box = [float(x) for x in crop.box]

                context.add_image(f"crop_{idx}", crop.image)
                context.add_object(f"metadata_{idx}", {
                    "box":   box,
                    "score": float(crop.score),
                    "class": cls,
                    "type":  obj_type,
                })
                if self.temp is not None:
                    crop.image.save(self.temp / f"crop_{idx}.png")
                    caption = f"type: {obj_type}\nclass: {cls}\nscore: {crop.score:.3f}\nbox: {[round(x, 1) for x in box]}\npass: {pass_num + 1}\n"
                    (self.temp / f"crop_{idx}.txt").write_text(caption)

                mask_array = np.array(crop.mask).astype(np.float32) / 255.0
                pass_objects.append((mask_array, box))
                if cls == "object":
                    all_object_masks.append(mask_array)

                pass_entry["crops"].append({
                    "index": idx,
                    "type":  obj_type,
                    "class": cls,
                    "score": round(float(crop.score), 3),
                    "box":   [round(x, 1) for x in box],
                })

                self.log_info(f"  crop_{idx}: {obj_type} → {cls}")
                self.advance_progress(classify_task)
            self.finish_progress(classify_task)
            fraction_advanced += 1.0 / self.total_tasks
            global_idx += result.length
            all_extracted_masks.extend(m for m, _ in pass_objects)
            manifest["passes"].append(pass_entry)

            # Two-phase inpainting:
            #   Phase 1 (LaMa)  — per-object structural fill, accumulating so each
            #                     call benefits from prior removals. Small masks
            #                     keep LaMa fast and structurally accurate.
            #   Phase 2 (Flux)  — single call over the full-scene union mask on the
            #                     accumulated LaMa result, downsampled to a resolution
            #                     Flux handles well. Fills all regions coherently in
            #                     one pass rather than independently per object.
            # Models are closed between phases to avoid holding both on GPU.
            inpaint_task = self.create_progress(
                len(pass_objects) * 2, f"Inpainting (pass {pass_num + 1})..."
            )
            # Use an environment-only fill prompt rather than the full scene caption.
            # The scene caption names the objects we just removed (e.g. "Eiffel Tower"),
            # and high guidance scale would cause Flux to regenerate them in the holes.
            # A neutral prompt lets Flux fill from surrounding pixel context instead.
            scene_caption = context.input_object(ContextKey.INPUT_CAPTION) or ""
            caption = _environment_prompt(scene_caption)

            # --- Phase 1: LaMa structural fill ---
            # Per-object, accumulating — each call benefits from the previous
            # removals, and the small per-mask size keeps LaMa fast and sharp.
            lama_states = []  # (mask_array, box, dilated_crop, crop_y0, crop_y1, crop_x0, crop_x1)
            current_arr = np.array(terrain_pil)

            lama_inpainter = InPainting(self.device, self.torch_dtype, InPaintingType.LAMA)
            for mask_array, box in pass_objects:
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

                # Compute crop coords before dilation so we dilate only the
                # crop region, not the full panorama. A 101×101 struct applied
                # to a 2048×1024 mask is O(H×W×K²) — very slow per object.
                # iterations=50 on a small crop achieves the same 50px expansion
                # at a fraction of the cost.
                pad     = max(64, int(max(bw, bh) * 0.5))
                crop_y0 = max(0, top    - pad)
                crop_y1 = min(h, bottom + pad)
                crop_x0 = max(0, left   - pad)
                crop_x1 = min(w, right  + pad)

                dilated_crop = binary_dilation(
                    mask_array[crop_y0:crop_y1, crop_x0:crop_x1] > 0.5,
                    iterations=lama_dilation,
                ).astype(np.float32)

                lama_crop = PILImage.fromarray(current_arr[crop_y0:crop_y1, crop_x0:crop_x1])
                mask_crop = PILImage.fromarray((dilated_crop * 255).astype(np.uint8), mode="L")

                lama_pil = lama_inpainter.inpaint(
                    lama_crop,
                    mask_crop,
                    temp_path=self.temp,
                )
                lama_arr = np.array(lama_pil)

                composited = current_arr[crop_y0:crop_y1, crop_x0:crop_x1].copy()
                composited[dilated_crop > 0.5] = lama_arr[dilated_crop > 0.5]
                current_arr[crop_y0:crop_y1, crop_x0:crop_x1] = composited

                lama_states.append((mask_array, box, dilated_crop, crop_y0, crop_y1, crop_x0, crop_x1))
                self.advance_progress(inpaint_task)

            lama_inpainter.close()

            # Re-caption from the LaMa result: at this point the objects are already
            # structurally removed, so the caption no longer mentions them.  This
            # prevents Flux from regenerating the things we just extracted.
            if self._captioner is None:
                self._captioner = ImageCaptioning(self.preferred_device)
            raw_lama_caption = self._captioner.caption(Image(PILImage.fromarray(current_arr)))
            caption = _environment_prompt(raw_lama_caption)
            self.log_info(f"Pass {pass_num + 1} fill caption: {caption!r}")

            # --- Phase 2: Flux perceptual refinement (per-object crop) ---
            # Crop around each object with context padding so Flux always operates
            # on a native-resolution region within its ~1MP training range. This
            # avoids both the full-panorama resolution problem and the upsampling
            # quality loss from downscaling first. Only falls back to scaling when
            # a single object crop itself exceeds flux_max.
            next_terrain_arr = np.array(terrain_pil)
            valid_states = [
                (m, b, d, y0, y1, x0, x1)
                for m, b, d, y0, y1, x0, x1 in lama_states
                if d is not None
            ]

            if valid_states:
                flux_max = 1024
                flux_inpainter = InPainting(self.device, self.torch_dtype, InPaintingType.FLUX)

                for mask_array, box, dilated_crop, crop_y0, crop_y1, crop_x0, crop_x1 in valid_states:
                    bx, by, bw, bh = box
                    left   = max(0, int(bx))
                    top    = max(0, int(by))
                    right  = min(w, int(bx + bw))
                    bottom = min(h, int(by + bh))

                    region_mask = mask_array[top:bottom, left:right]
                    if region_mask.sum() == 0:
                        self.advance_progress(inpaint_task)
                        continue

                    crop_h  = crop_y1 - crop_y0
                    crop_w  = crop_x1 - crop_x0

                    lama_crop = PILImage.fromarray(current_arr[crop_y0:crop_y1, crop_x0:crop_x1])
                    mask_crop = PILImage.fromarray((dilated_crop * 255).astype(np.uint8), mode="L")

                    # Scale only when the crop exceeds flux_max — most objects won't
                    if crop_w > flux_max or crop_h > flux_max:
                        scale     = flux_max / max(crop_w, crop_h)
                        flux_w    = max(16, (int(crop_w * scale) // 16) * 16)
                        flux_h    = max(16, (int(crop_h * scale) // 16) * 16)
                        lama_crop = lama_crop.resize((flux_w, flux_h), PILImage.LANCZOS)
                        mask_crop = mask_crop.resize((flux_w, flux_h), PILImage.NEAREST)

                    flux_pil = flux_inpainter.inpaint(
                        lama_crop,
                        mask_crop,
                        temp_path=self.temp,
                        prompt=caption,
                        num_inference_steps=28,
                        guidance_scale=7.0,
                    )

                    # Flux aligns dims to 16px internally; resize back to crop dims
                    if flux_pil.size != (crop_w, crop_h):
                        flux_pil = flux_pil.resize((crop_w, crop_h), PILImage.LANCZOS)

                    flux_crop_arr = np.array(flux_pil)

                    region_mask_pil = PILImage.fromarray((region_mask * 255).astype(np.uint8), mode="L")
                    feathered = np.array(region_mask_pil.filter(ImageFilter.GaussianBlur(radius=2))).astype(np.float32)[..., np.newaxis] / 255.0

                    orig_region = next_terrain_arr[top:bottom, left:right]
                    flux_region = flux_crop_arr[top - crop_y0 : bottom - crop_y0,
                                                left - crop_x0 : right  - crop_x0]
                    next_terrain_arr[top:bottom, left:right] = (orig_region * (1.0 - feathered) + flux_region * feathered).astype(np.uint8)
                    self.advance_progress(inpaint_task)

                flux_inpainter.close()

            terrain_pil = PILImage.fromarray(next_terrain_arr)

            if self.temp is not None:
                terrain_pil.save(self.temp / f"panorama_pass_{pass_num + 1}.png")

            self.finish_progress(inpaint_task)
            fraction_advanced += 1.0 / self.total_tasks

        # Snap main task to 1.0 for this stage.
        remaining = 1.0 - fraction_advanced
        if remaining > 1e-9:
            self.progress.advance(self.main_task, remaining)

        context.add_object(ContextKey.OBJECT_COUNT, global_idx)

        if self.temp is not None:
            if all_extracted_masks:
                overlay = _draw_extraction_overlay(original_pil, all_extracted_masks)
                overlay.save(self.temp / "extraction_overlay.png")

            manifest["total_crops"] = global_idx
            manifest["caption"] = context.input_object(ContextKey.INPUT_CAPTION) or ""
            manifest["passes_run"] = len(manifest["passes"])
            manifest["image_size"] = {"width": w, "height": h}
            with open(self.temp / "manifest.json", "w") as f:
                write_json(manifest, f)

        # Terrain panorama: the result after all passes (clean ground plane).
        context.add_panorama(ContextKey.PANORAMA_TERRAIN, Panorama(terrain_pil))

        # Visual panorama: original with only foreground-object regions filled from
        # the final terrain image; environment features are left at their original pixels.
        if all_object_masks:
            obj_union = np.zeros((h, w), dtype=np.float32)
            for m in all_object_masks:
                obj_union = np.maximum(obj_union, m)
            obj_mask_pil = PILImage.fromarray((obj_union * 255).astype(np.uint8), mode="L")
            obj_feathered = np.array(obj_mask_pil.filter(ImageFilter.GaussianBlur(radius=4))).astype(np.float32)[..., np.newaxis] / 255.0
            terrain_final_arr = np.array(terrain_pil)
            visual_composited = (original_array * (1.0 - obj_feathered) + terrain_final_arr * obj_feathered).astype(np.uint8)
            context.add_panorama(output_key, Panorama(PILImage.fromarray(visual_composited)))

        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        count = context.input_object(ContextKey.OBJECT_COUNT)
        if count is None:
            return False
        all_classified = all(
            (context.input_object(f"metadata_{i}") or {}).get("class") is not None
            for i in range(count)
        )
        terrain_ready = context.panorama(ContextKey.PANORAMA_TERRAIN) is not None
        return all_classified and terrain_ready

    def model_names(self) -> list[str]:
        return (
            ImageSeg.model_names()
            + InPainting.model_names(type=InPaintingType.LAMA)
            + InPainting.model_names(type=InPaintingType.FLUX)
            + ImageClipClassifier.model_names()
        )

    def clean_up(self):
        if self._seg is not None:
            self._seg.close()
            self._seg = None
        self._classifier = None
        self._captioner = None
        super().clean_up()
