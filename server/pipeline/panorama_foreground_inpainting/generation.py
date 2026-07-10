from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.segmentation.image_segmentation import ImageSeg
from pipeline.segmentation.depth_filter import DepthObjectFilter
from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.supersampling.image_supersampling import ImageSupersampling
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image
from util.panorama_utils import Panorama
import numpy as np
from PIL import Image as PILImage, ImageFilter
from scipy.ndimage import binary_dilation


class PanoramaForegroundInpaintingConfiguration(PipelineStageConfiguration):
    """
    Stage-specific config for PanoramaForegroundInpaintingStage.

    use_depth_filter (bool, default True):
        Whether to run DepthObjectFilter on SAM masks before removal.
        Requires PANORAMA_OBJECT_DEPTH (or PANORAMA_DEPTH) in context.

    depth_filter_threshold / depth_filter_edge_threshold:
        Same semantics as PanoramaInpaintingConfiguration — see that class.

    mask_dilation_px (int, default 15):
        Pixels to dilate the unioned foreground mask before removal, giving
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
        use_depth_filter: bool = True,
        depth_filter_threshold: float = 0.05,
        depth_filter_edge_threshold: float = 0.005,
        mask_dilation_px: int = 15,
        guidance_scale: float = 2.5,
        num_inference_steps: int = 30,
        supersample_result: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_depth_filter = use_depth_filter
        self.depth_filter_threshold = depth_filter_threshold
        self.depth_filter_edge_threshold = depth_filter_edge_threshold
        self.mask_dilation_px = mask_dilation_px
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.supersample_result = supersample_result


class PanoramaForegroundInpaintingStage(PipelineStage):
    """
    Removes foreground objects from the panorama using SAM2 segmentation
    (optionally depth-filtered) and ObjectClear, producing a clean terrain
    plate for downstream terrain reconstruction.

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
        self._seg = None
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

        if self._seg is None:
            self._seg = ImageSeg(self.preferred_device)
        if self._objectclear is None:
            self._objectclear = InPainting(self.preferred_device, self.preferred_dtype, InPaintingType.OBJECTCLEAR)

        self.set_total_tasks(2)  # segment + remove

        seg_task = self.create_progress(1, "Segmenting…")
        result = self._seg.segment(Image(original_pil), self.temp, on_progress=self.make_progress_callback(seg_task))
        self.finish_progress(seg_task)

        sam_detected = result.length
        if self.config.use_depth_filter:
            depth = context.input_depth(depth_key) or context.input_depth(ContextKey.PANORAMA_DEPTH)
            if depth is not None:
                result = DepthObjectFilter().filter(
                    result, depth,
                    threshold=self.config.depth_filter_threshold,
                    edge_threshold=self.config.depth_filter_edge_threshold,
                )
                depth_filtered_out = sam_detected - result.length
                if depth_filtered_out:
                    self.log_info(f"Depth filter removed {depth_filtered_out}/{sam_detected} background mask(s)")

        remove_task = self.create_progress(1, "Removing foreground objects…")

        if result.is_empty():
            self.log_info("Nothing found, skipping removal")
            terrain_pil = original_pil
        else:
            self.log_info(f"{result.length} object(s) found")

            union_mask = np.zeros((h, w), dtype=np.float32)
            for mask in result.masks:
                mask_arr = np.asarray(mask, dtype=np.float32)
                if mask_arr.shape != (h, w):
                    mask_arr = np.array(
                        PILImage.fromarray((mask_arr * 255).astype(np.uint8), mode="L").resize((w, h), PILImage.NEAREST)
                    ).astype(np.float32) / 255.0
                union_mask = np.maximum(union_mask, mask_arr)

            dilated_union = (binary_dilation(
                union_mask > 0.5,
                iterations=self.config.mask_dilation_px,
            ).astype(np.uint8)) * 255
            mask_pil = PILImage.fromarray(dilated_union, mode="L")

            if self.temp is not None:
                mask_pil.save(self.temp / "foreground_mask.png")

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

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, _, output_key = self._resolved_keys()
        return context.panorama(output_key) is not None

    def model_names(self) -> list[str]:
        names = ImageSeg.model_names() + InPainting.model_names(type=InPaintingType.OBJECTCLEAR)
        if self.config.supersample_result:
            names = names + ImageSupersampling.model_names()
        return names

    def clean_up(self):
        if self._seg is not None:
            self._seg.close()
            self._seg = None
        if self._objectclear is not None:
            self._objectclear.close()
            self._objectclear = None
        if self._samp is not None:
            self._samp.close()
            self._samp = None
        super().clean_up()
