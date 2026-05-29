import io
import base64
from pathlib import Path
from typing import Self, Optional
from PIL import Image as PILImage


class SceneLighting:
    """
    Environment map pair estimated by LuxDiT.

    Holds an LDR environment map and a log-domain environment map, both as
    equirectangular PIL Images. encode() inlines both as base64 PNG so Unity
    receives the full lighting data in the SCENE_INIT message without extra
    HTTP round-trips.
    """

    def __init__(self, ldr: PILImage.Image, log: PILImage.Image):
        self.ldr = ldr
        self.log = log

    @property
    def width(self) -> int:
        return self.ldr.width

    @property
    def height(self) -> int:
        return self.ldr.height

    def encode(self) -> dict:
        def _to_b64(img: PILImage.Image) -> str:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "ldr": _to_b64(self.ldr),
            "log": _to_b64(self.log),
            "width": self.ldr.width,
            "height": self.ldr.height,
        }

    @classmethod
    def decode(cls, data: dict) -> Self:
        def _from_b64(b64: str) -> PILImage.Image:
            return PILImage.open(io.BytesIO(base64.b64decode(b64))).copy()

        return cls(_from_b64(data["ldr"]), _from_b64(data["log"]))

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.ldr.save(path / "ldr.png")
        self.log.save(path / "log.png")

    @classmethod
    def load(cls, path: Path) -> Self:
        ldr = PILImage.open(path / "ldr.png").copy()
        log = PILImage.open(path / "log.png").copy()
        return cls(ldr, log)
