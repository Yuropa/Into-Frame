import torch
from pathlib import Path
from PIL import Image as PILImage
from pipeline.panorama.image_panorama_cubediff import ImagePanoramaCubeDiff
from pipeline.panorama.image_panorama_dreamcube import ImagePanoramaDreamCube
from pipeline.panorama.panorama_output import PanoramaOutput
from enum import Enum

class PanoramaGeneratorType(Enum):
    CUBEDIFF = 1
    DREAMCUBE = 2

    @classmethod
    def default(cls):
        return cls.CUBEDIFF

class ImagePanorama:
    def __init__(self, device: torch.device, type: PanoramaGeneratorType = PanoramaGeneratorType.default()) -> None:
        match type:
            case PanoramaGeneratorType.CUBEDIFF:
                self.generator = ImagePanoramaCubeDiff(
                    device=device
                )
            case PanoramaGeneratorType.DREAMCUBE:
                self.generator = ImagePanoramaDreamCube(
                    device=device
                )

    @classmethod
    def model_names(cls, type: PanoramaGeneratorType = PanoramaGeneratorType.default()) -> list[str]:
        match type:
            case PanoramaGeneratorType.CUBEDIFF:
                return ImagePanoramaCubeDiff.model_names()
            case PanoramaGeneratorType.DREAMCUBE:
                return ImagePanoramaDreamCube.model_names()

    def pano(self, input: PILImage, temp_path: Path, fov: float = 60.0, caption: str = "") -> PanoramaOutput:
        return self.generator.pano(input, temp_path, fov, caption)

    def close(self):
        self.generator.close()
