from pipeline.foreground_inpainting.foreground_inpainting import ForegroundInpaint

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image
import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps

class ForegroundInpainting(PipelineStage):
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

        inpainting_task = self.create_progress(2, "Inpainting...")
        if self._inpaint is None:
            self._inpaint = ForegroundInpaint(self.preferred_device, self.preferred_format)
        self.advance_progress(inpainting_task)

        result = self._inpaint.inpaint(input_image, self.temp)

        self.advance_progress(inpainting_task)
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