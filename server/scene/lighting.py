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

# Maps "share of total irradiance arriving from the sun region" onto Unity's
# directional-light intensity scale. The top 0.1% of a clear sky carries only a
# few percent of total irradiance even when it dominates the *look* of the
# scene -- most of the energy is spread across the whole sky -- so the share is
# scaled up to put a normal sunny day near Unity's nominal intensity of 1.
_SUN_INTENSITY_SCALE = 30.0

# Extra gain applied to `intensity` ONLY when the HDR merge didn't run and the
# estimate came from tonemapped LDR pixels instead (see _radiance).
#
# Tonemapping crushes precisely the range that separates a harsh sun from a bright
# overcast sky, and it crushes it in one direction: the sun's peak is compressed
# toward the sky's, so sun_share -- an energy ratio -- comes out systematically low
# and the client's key light with it. SceneParamManager.ApplySun assigns this
# straight to Unity's directional light, whose nominal daylight value is 1.0.
# Measured across five captures, every one of which fell back to LDR:
#
#     Paris  0.08    Irises 0.10    Rainier 0.20    Shark Fin 0.32    Iceland 0.42
#
# i.e. between 2.4x and 12.2x under nominal, which is the flat, weak lighting this
# corrects. 5.0 is the ratio the one available HDR-vs-LDR comparison showed (the
# LDR map put the sun at ~6x the map mean where the merged HDR put it at ~30x), and
# it lands the Rainier capture -- a clear-sky alpine noon with the solar disc
# visible in frame -- at 1.01.
#
# It is a single constant standing in for a per-scene quantity, and the spread above
# shows why that is a stopgap rather than a fix: the correct gain depends on how
# diffuse the sky actually is, which is exactly what the merger measures and this
# cannot. THE FIX IS TO MAKE THE MERGE RUN (LuxDiTServer.setup logs why it didn't);
# when it does, is_hdr goes True, this is bypassed entirely, and nothing here
# applies. Set to 1.0 to restore the previous under-lit behaviour.
_LDR_FALLBACK_SUN_SCALE = 5.0


class SceneLighting:
    """
    Environment map pair estimated by LuxDiT.

    Holds an LDR environment map and a log-domain environment map, both as
    equirectangular PIL Images. encode() inlines both as base64 PNG so Unity
    receives the full lighting data in the SCENE_INIT message without extra
    HTTP round-trips, alongside the key light extracted from them (see sun()).
    """

    def __init__(self, ldr: PILImage.Image, log: PILImage.Image, hdr=None):
        self.ldr = ldr
        self.log = log
        # (H, W, 3) float32 linear radiance from LuxDiT's own HDR merger, or None.
        # Not serialised to the client -- Unity gets the derived sun plus the LDR
        # map for its probe, which is all it needs, and a float EXR would dwarf
        # the rest of the scene payload.
        self.hdr = hdr

    def _radiance(self) -> "tuple[np.ndarray, bool]":
        """(H, W, 3) linear radiance and whether it is genuinely HDR.

        Falls back to the LDR map linearised by an assumed 2.2 gamma when the
        merger produced nothing. That fallback is explicitly a poor substitute
        and is flagged as such by the second return value: measured on the
        Rainier capture, the LDR map puts the sun at ~6x the map mean where the
        merged HDR puts it around 30x, because the tonemapping has already
        crushed exactly the range that distinguishes a harsh sun from a bright
        overcast sky. Direction and colour survive that compression; relative
        intensity does not.
        """
        if self.hdr is not None:
            return np.asarray(self.hdr, dtype=np.float32), True
        ldr = np.asarray(self.ldr.convert("RGB"), dtype=np.float32) / 255.0
        return np.power(ldr, 2.2, dtype=np.float32), False

    def sun(self) -> Optional[dict]:
        """The dominant directional light, measured from the environment map.

        Without this the client has no idea where the scene's light comes from:
        SceneLightingData carried only the two images, which EnvironmentLighting
        uses for ambient SH and the reflection probe, so the directional light
        kept whatever direction and intensity its Unity prefab happened to hold.
        Meanwhile every billboard and category mesh is a photo crop whose albedo
        already contains the real sun -- lit again from an unrelated angle, they
        read as a different colour and exposure from the terrain and skybox
        around them.

        All three values are measured from linear radiance (see _radiance).
        `intensity` is the sun region's mean radiance relative to the whole map's
        -- a real ratio when the HDR merge succeeded, and a badly compressed one
        when it didn't, which is why `hdr` is reported alongside it.

        Returns {"direction", "color", "intensity", "hdr"} or None if the map is
        degenerate (uniform, so no direction is identifiable). `direction` is a
        unit vector pointing FROM the scene TOWARD the sun, in the panorama's own
        frame -- the same frame the skybox is in, so the client must apply
        skyboxRotation to it exactly as it does to the skybox, or the light and
        the sky it came from will disagree.
        """
        rgb, is_hdr = self._radiance()
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

        # Share of the scene's total irradiance arriving from the sun region,
        # solid-angle weighted (an equirectangular texel subtends
        # (2pi/W)(pi/H)cos(lat), so rows near the poles must not count as much as
        # rows near the horizon). This is the quantity Unity's split actually
        # needs: the reflection probe's ambient SH is low-frequency and cannot
        # represent a sharp solar disc, so the directional light supplies exactly
        # the fraction of the lighting that is concentrated rather than diffuse.
        #
        # Scaled so a fully concentrated source would read 1.0. Overcast maps land
        # near zero and get no directional light worth speaking of, which is
        # correct -- the probe already carries that lighting.
        texel_solid_angle = (
            np.cos((0.5 - (np.arange(height) + 0.5) / height) * np.pi)
            * (2.0 * np.pi / width) * (np.pi / height)
        )[:, None]
        irradiance = luminance * texel_solid_angle
        total = float(irradiance.sum())
        sun_share = float(irradiance[rows, cols].sum()) / total if total > 1e-9 else 0.0

        return {
            "direction": [round(float(v), 5) for v in direction],
            "color": "#{:02x}{:02x}{:02x}".format(
                *(int(round(float(c) * 255.0)) for c in np.clip(color, 0.0, 1.0))
            ),
            "intensity": round(float(np.clip(
                sun_share * _SUN_INTENSITY_SCALE
                * (1.0 if is_hdr else _LDR_FALLBACK_SUN_SCALE),
                0.0, 3.0,
            )), 3),
            # False means the merger didn't run, so `intensity` came from
            # tonemapped pixels and understates a harsh sun. Callers that care
            # (the client logs it) can tell the difference.
            "hdr": bool(is_hdr),
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
