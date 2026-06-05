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
    Segments foreground objects in the panorama, classifies each crop as
    'object' or 'environment' using CLIP, then inpaints only the object regions.

    Input key  (SemanticKey.PANORAMA) → ContextKey.PANORAMA  (Panorama)
    Output key (SemanticKey.OUTPUT)   → ContextKey.PANORAMA  (Panorama, objects removed)

    Dynamic context keys per detected object (index i):
      crop_{i}     → Image
      metadata_{i} → {"box": [...], "score": float, "class": str, "type": str}

    Also writes ContextKey.OBJECT_COUNT so downstream stages can consume
    the same crop/metadata keys without re-running segmentation.
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

        input_image = Image(panorama.image)

        # Segment
        segmenting_task = self.create_progress(2, "Segmenting Panorama...")
        if self._seg is None:
            self._seg = ImageSeg(self.preferred_device)
        self.advance_progress(segmenting_task)

        result = self._seg.segment(input_image, self.temp, on_progress=self.make_progress_callback(segmenting_task))

        # Classify each crop and collect masks for objects only
        if self._classifier is None:
            self._classifier = ImageClipClassifier(self.preferred_device)

        object_masks = []
        classify_task = self.create_progress(result.length, "Classifying objects...")
        for idx, crop in enumerate(result.masked_images(input_image)):
            obj_type, cls = self._classifier.classify(crop.image)

            context.add_image(f"crop_{idx}", crop.image)
            context.add_object(f"metadata_{idx}", {
                "box":   [float(x) for x in crop.box],
                "score": float(crop.score),
                "class": cls,
                "type":  obj_type,
            })
            if self.temp is not None:
                crop.image.save(self.temp / f"crop_{idx}.png")

            if cls == "object":
                # crop.mask is the full-size alpha PIL image — convert to float array
                object_masks.append(
                    np.array(crop.mask).astype(np.float32) / 255.0
                )

            self.log_info(f"  crop_{idx}: {obj_type} → {cls}")
            self.advance_progress(classify_task)

        self.finish_progress(classify_task)
        context.add_object(ContextKey.OBJECT_COUNT, result.length)

        self.advance_progress(segmenting_task)
        self.finish_progress(segmenting_task)

        # Inpaint — union only object masks then fill once
        if object_masks:
            inpainting_task = self.create_progress(2, "Inpainting Panorama...")

            union_mask = np.zeros(
                (input_image.height, input_image.width), dtype=np.float32
            )
            for m in object_masks:
                union_mask = np.maximum(union_mask, m)

            dilation_factor = 20
            struct = np.ones((dilation_factor * 2 + 1, dilation_factor * 2 + 1))
            dilated = binary_dilation(union_mask > 0.5, structure=struct).astype(np.float32)

            mask_pil = PILImage.fromarray((dilated * 255).astype(np.uint8), mode="L")
            img_array = np.array(input_image.rgb())
            masked_array = (img_array * (1.0 - dilated[..., np.newaxis])).astype(np.uint8)
            masked_pil = PILImage.fromarray(masked_array)

            self.advance_progress(inpainting_task)

            inpainter = InPainting(self.device, self.torch_dtype, InPaintingType.LAMA)
            result_pil = inpainter.inpaint(
                masked_pil,
                mask_pil,
                temp_path=self.temp,
                prompt="no objects, clean background, seamless, empty landscape",
                guidance_scale=2.0,
                strength=1.0,
            )
            inpainter.close()

            # Composite: only replace the masked region with LaMa's fill.
            # Feather the boundary so the blend is gradual rather than a hard seam.
            result_array = np.array(result_pil.crop((0, 0, input_image.width, input_image.height)))
            feathered_mask = mask_pil.filter(ImageFilter.GaussianBlur(radius=8))
            mask_3d = np.array(feathered_mask).astype(np.float32)[..., np.newaxis] / 255.0
            composited = (img_array * (1.0 - mask_3d) + result_array * mask_3d).astype(np.uint8)
            context.add_panorama(output_key, Panorama(PILImage.fromarray(composited)))

            self.advance_progress(inpainting_task)
            self.finish_progress(inpainting_task)

        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        count = context.input_object(ContextKey.OBJECT_COUNT)
        if count is None:
            return False
        return all(
            (context.input_object(f"metadata_{i}") or {}).get("class") is not None
            for i in range(count)
        )

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
