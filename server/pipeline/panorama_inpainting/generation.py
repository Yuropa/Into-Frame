from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.segmentation.image_segmentation import ImageSeg
from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.object_typing.image_clip_classifier import ImageClipClassifier
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image
from util.panorama_utils import Panorama
import numpy as np
from PIL import Image as PILImage, ImageFilter
from scipy.ndimage import binary_dilation

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

        dilation_factor = 20
        struct = np.ones((dilation_factor * 2 + 1, dilation_factor * 2 + 1))

        # Iterative extract-and-inpaint: segment → extract crops → inpaint → repeat
        # until segmentation finds nothing. terrain_pil tracks the progressively
        # cleaned image; each pass removes whatever was found in that pass.
        terrain_pil = original_pil
        all_object_masks = []  # accumulated across passes for the visual panorama
        global_idx = 0
        max_passes = 10
        initial_passes = 3
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

            if result.length == 0:
                self.log_info(f"Pass {pass_num + 1}: nothing found, stopping")
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
                obj_type, cls = self._classifier.classify(crop.image)
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

                self.log_info(f"  crop_{idx}: {obj_type} → {cls}")
                self.advance_progress(classify_task)
            self.finish_progress(classify_task)
            fraction_advanced += 1.0 / self.total_tasks
            global_idx += result.length

            # One LaMa call with the full panorama so it has complete context for
            # colour/texture matching. Write-back is limited per object to its
            # bounding-box region so the rest of the image is never touched.
            inpaint_task = self.create_progress(2, f"Inpainting (pass {pass_num + 1})...")
            terrain_arr = np.array(terrain_pil)

            union = np.zeros((h, w), dtype=np.float32)
            for mask_array, _ in pass_objects:
                union = np.maximum(union, mask_array)
            union_dilated = binary_dilation(union > 0.5, structure=struct).astype(np.float32)

            full_mask_pil = PILImage.fromarray((union_dilated * 255).astype(np.uint8), mode="L")
            masked_pil = PILImage.fromarray((terrain_arr * (1.0 - union_dilated[..., np.newaxis])).astype(np.uint8))

            inpainter = InPainting(self.device, self.torch_dtype, InPaintingType.LAMA)
            result_pil = inpainter.inpaint(
                masked_pil,
                full_mask_pil,
                temp_path=self.temp,
                prompt="no objects, clean background, seamless, empty landscape",
                guidance_scale=2.0,
                strength=1.0,
            )
            inpainter.close()
            self.advance_progress(inpaint_task)

            result_arr = np.array(result_pil)

            # Composite result back per object — only within each object's bbox.
            for mask_array, box in pass_objects:
                bx, by, bw, bh = box
                pad = 32
                left   = max(0, int(bx - pad))
                top    = max(0, int(by - pad))
                right  = min(w, int(bx + bw + pad))
                bottom = min(h, int(by + bh + pad))

                region_mask = mask_array[top:bottom, left:right]
                region_dilated = binary_dilation(region_mask > 0.5, structure=struct).astype(np.float32)
                if region_dilated.sum() == 0:
                    continue

                region_mask_pil = PILImage.fromarray((region_dilated * 255).astype(np.uint8), mode="L")
                feathered = np.array(region_mask_pil.filter(ImageFilter.GaussianBlur(radius=8))).astype(np.float32)[..., np.newaxis] / 255.0

                orig_region   = terrain_arr[top:bottom, left:right]
                result_region = result_arr[top:bottom, left:right]
                terrain_arr[top:bottom, left:right] = (orig_region * (1.0 - feathered) + result_region * feathered).astype(np.uint8)

            terrain_pil = PILImage.fromarray(terrain_arr)

            if self.temp is not None:
                terrain_pil.save(self.temp / f"panorama_pass_{pass_num + 1}.png")

            self.advance_progress(inpaint_task)
            self.finish_progress(inpaint_task)
            fraction_advanced += 1.0 / self.total_tasks

        # Snap main task to 1.0 for this stage.
        remaining = 1.0 - fraction_advanced
        if remaining > 1e-9:
            self.progress.advance(self.main_task, remaining)

        context.add_object(ContextKey.OBJECT_COUNT, global_idx)

        # Terrain panorama: the result after all passes (clean ground plane).
        context.add_panorama(ContextKey.PANORAMA_TERRAIN, Panorama(terrain_pil))

        # Visual panorama: original with only foreground-object regions filled from
        # the final terrain image; environment features are left at their original pixels.
        if all_object_masks:
            obj_union = np.zeros((h, w), dtype=np.float32)
            for m in all_object_masks:
                obj_union = np.maximum(obj_union, m)
            obj_dilated = binary_dilation(obj_union > 0.5, structure=struct).astype(np.float32)
            obj_mask_pil = PILImage.fromarray((obj_dilated * 255).astype(np.uint8), mode="L")
            obj_feathered = np.array(obj_mask_pil.filter(ImageFilter.GaussianBlur(radius=8))).astype(np.float32)[..., np.newaxis] / 255.0
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
            + ImageClipClassifier.model_names()
        )

    def clean_up(self):
        if self._seg is not None:
            self._seg.close()
            self._seg = None
        self._classifier = None
        super().clean_up()
