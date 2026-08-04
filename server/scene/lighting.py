import io
import base64
import json
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


def measure_panorama_sun(
    panorama_rgb: np.ndarray,
    sky_mask: "np.ndarray | None" = None,
    *,
    min_peak_contrast: float = 1.5,
) -> "tuple[list, str] | None":
    """Locate the sun in the panorama itself: (unit direction, hex colour) or None.

    The environment map is an ESTIMATE of the lighting; the panorama is the
    photograph (plus a generated sky continuous with it), and on a clear day the
    solar disc is simply visible in it. Measured against the disc's true position
    across four captures, LuxDiT's env-map direction was off by 81-127 degrees:

        Rainier   env el -10.9 az  +99.1   real el +61.5 az -60.9   127 deg
        Shark Fin env el +26.6 az -144.4   real el +30.8 az +53.4   120 deg
        Paris     env el +33.0 az  +78.0   real el +19.4 az -13.8    81 deg
        Iceland   env el +22.9 az  +99.9   real el  +3.9 az +17.3    82 deg

    On Rainier -- a clear alpine noon -- that put the key light 11 degrees BELOW
    the horizon, lighting the scene from underneath. It is not a frame-convention
    bug: no axis flip, 180-degree rotation or axis swap brings all four into
    agreement (best candidate still averages 66 degrees of error). LuxDiT is fed
    only a 90-degree centre crop (PanoramaLightingConfiguration.crop_fov_deg), so
    on two of these the sun is not even inside its input and it is inferring the
    direction from shading cues.

    Direction and colour therefore come from here; INTENSITY does not, and cannot
    -- the panorama is 8-bit and its sun is clipped, which is the whole reason the
    HDR merge exists. See SceneLighting.sun.

    sky_mask          -- boolean, True where the panorama is sky. Strongly
                         preferred: the brightest pixels of a Rainier panorama are
                         sunlit snowfields and of a Paris one the glare off the
                         Seine, and unmasked both outvote the disc. Without it the
                         search is limited to above the horizon, which is weaker
                         but still excludes the ground.
    min_peak_contrast -- the peak region must be at least this many times the
                         median sky luminance to count as a disc at all. An
                         overcast sky has no key light, and its brightest 0.1% is
                         just noise pointing in an arbitrary direction -- returning
                         that would be worse than returning nothing. None means
                         "no identifiable sun", and the caller keeps whatever the
                         environment map said.
    """
    rgb = np.asarray(panorama_rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    rgb = rgb[..., :3] / 255.0 if rgb.max() > 1.5 else rgb[..., :3]
    height, width = rgb.shape[:2]
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    if sky_mask is not None and np.asarray(sky_mask).shape == (height, width):
        candidate = np.asarray(sky_mask, dtype=bool)
    else:
        candidate = np.zeros((height, width), dtype=bool)
        candidate[: height // 2, :] = True
    if not candidate.any():
        return None

    sky_values = luminance[candidate]
    median = float(np.median(sky_values))
    threshold = float(np.percentile(sky_values, _SUN_PERCENTILE))
    peak = candidate & (luminance >= threshold)
    if not peak.any():
        return None

    rows, cols = np.nonzero(peak)
    weights = luminance[rows, cols].astype(np.float64)
    weight_total = float(weights.sum())
    if weight_total <= 0.0:
        return None
    if median > 1e-6 and (weights.mean() / median) < min_peak_contrast:
        return None

    # Same seam-safe averaging as SceneLighting.sun: mean the unit vectors, never
    # the raw column indices, or a sun straddling the wrap lands on the far side.
    longitude = ((cols + 0.5) / width - 0.5) * 2.0 * np.pi
    latitude = (0.5 - (rows + 0.5) / height) * np.pi
    direction = np.array([
        float((np.cos(latitude) * np.sin(longitude) * weights).sum()),
        float((np.sin(latitude) * weights).sum()),
        float((np.cos(latitude) * np.cos(longitude) * weights).sum()),
    ]) / weight_total
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return None
    direction = direction / norm

    color = (rgb[rows, cols] * weights[:, None]).sum(axis=0) / weight_total
    color_peak = float(color.max())
    if color_peak > 0:
        color = color / color_peak

    return (
        [round(float(v), 5) for v in direction],
        "#{:02x}{:02x}{:02x}".format(
            *(int(round(float(c) * 255.0)) for c in np.clip(color, 0.0, 1.0))
        ),
    )


class SceneLighting:
    """
    Environment map pair estimated by LuxDiT.

    Holds an LDR environment map and a log-domain environment map, both as
    equirectangular PIL Images. encode() inlines both as base64 PNG so Unity
    receives the full lighting data in the SCENE_INIT message without extra
    HTTP round-trips, alongside the key light extracted from them (see sun()).
    """

    def __init__(self, ldr: PILImage.Image, log: PILImage.Image, hdr=None, measured_sun=None):
        self.ldr = ldr
        self.log = log
        # (direction, color) measured from the PANORAMA's own sky rather than from
        # the estimated environment map, or None. See sun() and measure_panorama_sun.
        self.measured_sun = measured_sun
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

        # Prefer the sun measured from the panorama itself, when one was found.
        # Direction and colour are observations there and estimates here (see
        # measure_panorama_sun for the 81-127 degree discrepancy); intensity is the
        # reverse -- the panorama's sun is clipped 8-bit, so only the environment
        # map can speak to how much energy it carries. Take each from the source
        # that actually knows.
        measured = self.measured_sun
        return {
            "direction": (
                list(measured[0]) if measured
                else [round(float(v), 5) for v in direction]
            ),
            "color": measured[1] if measured else "#{:02x}{:02x}{:02x}".format(
                *(int(round(float(c) * 255.0)) for c in np.clip(color, 0.0, 1.0))
            ),
            # Where direction/colour came from, so a scene that silently fell back
            # to the estimate is distinguishable from one that didn't.
            "direction_source": "panorama" if measured else "envmap",
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
        # Carry the measured sun back too, for the same reason save/load do -- a
        # decoded SceneLighting that recomputed sun() from the env map alone would
        # report the direction this class exists to override.
        sun = data.get("sun") or {}
        measured_sun = (
            (sun["direction"], sun["color"])
            if sun.get("direction_source") == "panorama" and "direction" in sun and "color" in sun
            else None
        )

        def _from_b64(b64: str) -> PILImage.Image:
            return PILImage.open(io.BytesIO(base64.b64decode(b64))).copy()

        return cls(
            _from_b64(data["ldr"]), _from_b64(data["log"]), measured_sun=measured_sun,
        )

    # The panorama-measured sun has to survive the context cache. PanoramaLightingStage
    # is cached like every other stage, so on any resumed run this pair -- not run() --
    # is what produces the SceneLighting the scene is built from. Persisting only the
    # two PNGs silently dropped `measured_sun` and fell the scene back to the
    # environment map's own direction, which is the 81-127-degrees-wrong one this
    # exists to replace (see measure_panorama_sun). The failure mode is the nasty
    # kind: correct on the run that computed it, wrong on every run after.
    _SUN_FILE = "measured_sun.json"

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.ldr.save(path / "ldr.png")
        self.log.save(path / "log.png")
        sun_path = path / self._SUN_FILE
        if self.measured_sun is not None:
            direction, color = self.measured_sun
            sun_path.write_text(json.dumps({"direction": list(direction), "color": color}))
        elif sun_path.exists():
            # A rerun that found no disc must clear a stale one rather than
            # inheriting the previous run's answer.
            sun_path.unlink()

    @classmethod
    def load(cls, path: Path) -> Self:
        ldr = PILImage.open(path / "ldr.png").copy()
        log = PILImage.open(path / "log.png").copy()
        measured_sun = None
        sun_path = path / cls._SUN_FILE
        if sun_path.exists():
            try:
                data = json.loads(sun_path.read_text())
                measured_sun = (data["direction"], data["color"])
            except (ValueError, KeyError, OSError):
                measured_sun = None
        # hdr is deliberately not persisted (a float EXR would dwarf the cache, and
        # the intensity it feeds is already baked into what the scene shipped), so a
        # loaded SceneLighting reports is_hdr False and takes the LDR fallback path.
        return cls(ldr, log, measured_sun=measured_sun)
