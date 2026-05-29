from util.panorama_utils import Panorama
from util.cubemap_utils import CubeMap


class PanoramaOutput:
    panorama: Panorama
    cubemap: CubeMap

    def __init__(self, values: dict):
        self.panorama = Panorama(values["image"])
        self.cubemap = CubeMap(values["faces"])
