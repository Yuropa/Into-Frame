from pipeline.foreground_inpainting.foreground_inpainting import ForegroundInpaint

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image
import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps

class ForegroundInpainting(PipelineStage):
    """
    Detects and removes foreground objects from the input image, filling the
    removed regions with plausible background content via inpainting.

    Input key  (SemanticKey.INPUT)  → ContextKey.INPUT                   (Image)
    Output key (SemanticKey.OUTPUT) → ContextKey.FOREGROUND_MASKED_IMAGE  (Image, foreground removed)
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._inpaint = None
        self.preferred_device, self.preferred_format = preferred_device(DeviceStrategy.MEMORY)

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.INPUT,
            SemanticKey.OUTPUT: ContextKey.FOREGROUND_MASKED_IMAGE
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        input_key, output_key = self._resolved_keys()

        input_image = context.input_image(input_key).copy()

        max_iter = 20
        inpainting_task = self.create_progress(max_iter + 1, "Inpainting...")

        if self._inpaint is None:
            self.log_info("Loading inpainting models...")
            self._inpaint = ForegroundInpaint(self.preferred_device, self.preferred_format, seed=self.seed)
        self.advance_progress(inpainting_task)

        def on_iteration(idx: int, fill_pct: float):
            self.advance_progress(inpainting_task)

        result = self._inpaint.inpaint(input_image, self.temp, on_iteration=on_iteration)

        self.finish_progress(inpainting_task)

        context.add_image(output_key, result)
        return context
    
    def has_expected_output(self, context: PipelineContext) -> bool:
        _, output_key = self._resolved_keys()
        return context.object(output_key) is not None

    def model_names(self) -> list[str]:
        return ForegroundInpaint.model_names()

    def clean_up(self):
        self._inpaint = None
        super().clean_up()