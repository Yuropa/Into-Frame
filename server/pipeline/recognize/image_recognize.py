import torch
from pathlib import Path
from PIL import Image as PILImage
from enum import Enum
from pipeline.recognize.image_recognize_ram_plus import ImageRecognizeRAMPlus, RecognizeResult


class RecognizeCaptioningType(Enum):
    RAM_PLUS = 1

    @classmethod
    def default(cls) -> "RecognizeCaptioningType":
        return cls.RAM_PLUS


class ImageRecognize:
    def __init__(
        self,
        device: torch.device,
        type: RecognizeCaptioningType = RecognizeCaptioningType.default(),
    ) -> None:
        match type:
            case RecognizeCaptioningType.RAM_PLUS:
                self.generator = ImageRecognizeRAMPlus(device=device)

    @classmethod
    def model_names(cls, type: RecognizeCaptioningType = RecognizeCaptioningType.default()) -> list[str]:
        match type:
            case RecognizeCaptioningType.RAM_PLUS:
                return ImageRecognizeRAMPlus.model_names()

    def recognize(self, input: PILImage.Image, temp_path: Path, on_progress=None) -> RecognizeResult:
        return self.generator.recognize(input, temp_path, on_progress=on_progress)

    def close(self):
        self.generator.close()
