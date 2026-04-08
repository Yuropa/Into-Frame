from util.image_utils import Image
from pathlib import Path

class PipelineContext():
    def __init__(self, input: any) -> None:
        self.input_image = Image(input) 
        self._images = {}

    def add_image(self, name: str, input: any):
        self._images[name] = Image(input)

    def image(self, name: str):
        return self._images[name]

    def save(self, path: Path):
        self.input_image.save(path=str(path / "input.jpeg"))

        for name, image in self._images.items(): 
            image.save(path=str(path / (name + ".jpeg")))