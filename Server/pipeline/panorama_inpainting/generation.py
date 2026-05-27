from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.segmentation.image_segmentation import ImageSeg
from pipeline.segmentation.segmentation_result import SegmentationResult
from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.image_utils import Image
from util.panorama_utils import Panorama
import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import binary_dilation

class PanoramaInpaintingStage(PipelineStage):
    """
    Segments foreground objects in the panorama, saves each crop for later
    3D model generation, then inpaints the panorama to remove the objects.

    Input key  (SemanticKey.PANORAMA) → ContextKey.PANORAMA  (Panorama)
    Output key (SemanticKey.OUTPUT)   → ContextKey.PANORAMA  (Panorama, objects removed)

    Dynamic context keys per detected object (index i):
      crop_{i}     → Image  (masked object crop)
      metadata_{i} → object ({"box": [...], "score": float})

    Also writes ContextKey.OBJECT_COUNT so downstream stages (scene generation)
    can consume the same crop/metadata keys without re-running segmentation.
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._seg = None

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
            self._seg = ImageSeg(self.device)
        self.advance_progress(segmenting_task)

        result = self._seg.segment(input_image)

        # Store crops and metadata (same format as SegmentationStage)
        cropping_task = self.create_progress(result.length, "Cropping...")
        for idx, crop in enumerate(result.masked_images(input_image)):
            context.add_image(f"crop_{idx}", crop.image)
            context.add_object(f"metadata_{idx}", {
                "box":   [float(x) for x in crop.box],
                "score": float(crop.score),
            })
            self.advance_progress(cropping_task)
        self.finish_progress(cropping_task)
        context.add_object(ContextKey.OBJECT_COUNT, result.length)

        self.advance_progress(segmenting_task)
        self.finish_progress(segmenting_task)

        # Inpaint — union all masks then fill once
        if result.length > 0:
            inpainting_task = self.create_progress(2, "Inpainting Panorama...")

            union_mask = np.zeros(
                (input_image.height, input_image.width), dtype=np.float32
            )
            for mask in result.masks:
                m = mask.astype(np.float32)
                if m.ndim == 3:
                    m = m[..., 0]
                union_mask = np.maximum(union_mask, m)

            dilation_factor = 20
            struct = np.ones((dilation_factor * 2 + 1, dilation_factor * 2 + 1))
            from scipy.ndimage import binary_dilation as _dilate
            dilated = _dilate(union_mask > 0.5, structure=struct).astype(np.float32)

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

            context.add_panorama(output_key, Panorama(result_pil))

            self.advance_progress(inpainting_task)
            self.finish_progress(inpainting_task)

        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.input_object(ContextKey.OBJECT_COUNT) is not None

    def model_names(self) -> list[str]:
        return ImageSeg.model_names() + InPainting.model_names(type=InPaintingType.LAMA)

    def clean_up(self):
        self._seg = None
        super().clean_up()
