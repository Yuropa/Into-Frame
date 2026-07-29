import io
import base64
from pathlib import Path
from typing import Self, Optional

import numpy as np
from PIL import Image as PILImage

# Fraction of the environment map's pixels treated as "the sun" when locating it:
# the top 0.1% by luminance. Small enough to isolate the solar disc from bright
# sky around it, large enough that the estimate is a centroid over many pixels
# rather than a single argmax that a lone hot pixel could carry.
_SUN_PERCENTILE = 99.9


class SceneLighting:
    """
    Environment map pair estimated by LuxDiT.

    Holds an LDR environment map and a log-domain environment map, both as
    equirectangular PIL Images. encode() inlines both as base64 PNG so Unity
    receives the full lighting data in the SCENE_INIT message without extra
    HTTP round-trips, alongside the key light extracted from them (see sun()).
    """

    def __init__(self, ldr: PILImage.Image, log: PILImage.Image):
        self.ldr = ldr
        self.log = log

    def sun(self) -> Optional[dict]:
        """The dominant directional light, read out of the log environment map.

        Without this the client has no idea where the scene's light comes from:
        SceneLightingData carried only the two images, which EnvironmentLighting
        uses for ambient SH and the reflection probe, so the directional light
        kept whatever direction and intensity its Unity prefab happened to hold.
        Meanwhile every billboard and category mesh is a photo crop whose albedo
        already contains the real sun -- lit again from an unrelated angle, they
        read as a different colour and exposure from the terrain and skybox
        around them.

        Direction and colour are genuinely measured here. Intensity is not: the
        maps are LDR/log-compressed with no photometric calibration, and the sun
        in the LDR map is clipped flat (measured 0.992 luminance on the Rainier
        capture), so there is no absolute scale to recover. What is returned is a
        relative brightness ratio, bounded to a sane window, for the client to
        scale its own nominal sun by -- a heuristic, and labelled as one.

        The log map is used rather than the LDR one precisely because it is the
        less clipped of the two, so the centroid isn't dragged around by a
        saturated plateau.

        Returns {"direction", "color", "intensity"} or None if the map is
        degenerate (uniform, so no direction is identifiable). `direction` is a
        unit vector pointing FROM the scene TOWARD the sun, in the panorama's own
        frame -- the same frame the skybox is in, so the client must apply
        skyboxRotation to it exactly as it does to the skybox, or the light and
        the sky it came from will disagree.
        """
        rgb = np.asarray(self.log.convert("RGB"), dtype=np.float32) / 255.0
        luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        height, width = luminance.shape

        threshold = float(np.percentile(luminance, _SUN_PERCENTILE))
        peak = luminance >= threshold
        if not peak.any() or float(luminance.max()) <= float(luminance.min()):
            return None

        rows, cols = np.nonzero(peak)
        weights = luminance[rows, cols].astype(np.float64)
        weight_total = float(weights.sum())
        if weight_total <= 0.0:
            return None

        # Longitude is circular, so a luminance-weighted mean of raw column
        # indices is wrong for any sun straddling the seam -- average the unit
        # vectors instead, which has no seam to straddle.
        longitude = ((cols + 0.5) / width - 0.5) * 2.0 * np.pi
        latitude = (0.5 - (rows + 0.5) / height) * np.pi
        direction = np.array([
            float((np.cos(latitude) * np.sin(longitude) * weights).sum()),
            float((np.sin(latitude) * weights).sum()),
            float((np.cos(latitude) * np.cos(longitude) * weights).sum()),
        ]) / weight_total
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            # The bright pixels cancelled out -- a uniformly bright sky, or an
            # overcast map with no key light. Ambient alone is the honest answer.
            return None
        direction = direction / norm

        color = (rgb[rows, cols] * weights[:, None]).sum(axis=0) / weight_total
        color_peak = float(color.max())
        if color_peak > 0:
            color = color / color_peak   # hue/tint only; brightness is `intensity`

        # How *concentrated* the lighting is: the sun region's mean luminance
        # against the whole map's. This is a directionality measure, not a
        # photometric one -- it says "harsh key light" vs "diffuse overcast",
        # which is the part that survives LDR compression.
        #
        # Passed through sqrt and halved because the raw ratio is heavily
        # right-skewed and would otherwise sit on whatever ceiling it were
        # clamped at for any clear sky at all (measured: 5.7 on the Rainier
        # capture against ~1.1 for a flat overcast sky, so a clamp at 3.0 was
        # saturated and therefore carried no information). Calibrated so a
        # typical clear sky lands near Unity's own nominal intensity of 1:
        #   overcast 1.1 -> 0.52,  hazy 2.0 -> 0.71,
        #   this capture 5.7 -> 1.20,  harsh desert 14.0 -> 1.60 (clamped)
        mean_luminance = float(luminance.mean())
        ratio = float(weights.mean()) / mean_luminance if mean_luminance > 1e-6 else 1.0

        return {
            "direction": [round(float(v), 5) for v in direction],
            "color": "#{:02x}{:02x}{:02x}".format(
                *(int(round(float(c) * 255.0)) for c in np.clip(color, 0.0, 1.0))
            ),
            "intensity": round(float(np.clip(np.sqrt(ratio) / 2.0, 0.4, 1.6)), 3),
        }

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
            "sun": self.sun(),
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
