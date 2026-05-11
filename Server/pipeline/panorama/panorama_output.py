from util.image_utils import Image
from util.cubemap_utils import CubeMap

class PanoramaOutput:
    image: Image
    cubemap: CubeMap

    def __init__(self, values: dict):
        self.image = Image(values["image"])
        self.cubemap = CubeMap(values["faces"])
