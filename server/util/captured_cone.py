"""
The photographed cone of the panorama, and the mapping between panorama pixels
and the original photo's pixels inside it.

Only one part of the generated 360° panorama was ever observed: the input
photograph, which PanoramaStage (WorldGen) embeds at the panorama's centre and
outpaints around. Everything else is a generative guess, and a guess that tends
to repeat the hero content around the sphere (a sea stack three times, a ridge
twice). Stages that read the panorama as if it were a capture -- object
detection above all -- need to know where the capture ends.

The embedding is WorldGen's `perspective_rays` (server/pipeline/panorama/
pano_utils.py): photo pixel (u, v) in [0, 1] maps to longitude
(u - 0.5) * hfov and latitude (v - 0.5) * vfov, with
vfov = 2 * atan(tan(hfov / 2) * h / w). Note that is linear in angle, not a true
pinhole projection -- the photo is slightly warped on its way into the
panorama, so a reconstruction that wants real pixels should go back to the
photo through this exact mapping rather than through the intrinsics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CapturedCone:
    hfov_deg: float
    vfov_deg: float
    pano_w: int
    pano_h: int
    photo_w: int
    photo_h: int

    @classmethod
    def from_fov(cls, hfov_deg: float, pano_w: int, pano_h: int, photo_w: int, photo_h: int) -> "CapturedCone":
        hfov = math.radians(hfov_deg)
        vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * photo_h / photo_w)
        return cls(hfov_deg, math.degrees(vfov), pano_w, pano_h, photo_w, photo_h)

    # -- panorama-space extent of the capture ------------------------------

    @property
    def u_range(self) -> tuple[float, float]:
        """Panorama columns [left, right) covered by the photo."""
        half = self.hfov_deg / 360.0 * self.pano_w / 2.0
        return self.pano_w / 2.0 - half, self.pano_w / 2.0 + half

    @property
    def v_range(self) -> tuple[float, float]:
        """Panorama rows [top, bottom) covered by the photo."""
        half = self.vfov_deg / 180.0 * self.pano_h / 2.0
        return self.pano_h / 2.0 - half, self.pano_h / 2.0 + half

    def contains_box(self, box, margin_deg: float = 0.0, min_fraction: float = 0.5) -> bool:
        """True when at least `min_fraction` of a panorama-space [x, y, w, h] box
        lies inside the cone widened by `margin_deg` on every side.

        A margin, because a real object straddling the photo's edge is still a
        real object -- the outpainter continues it rather than inventing it --
        and because WorldGen's embedding is not pixel-exact at the border."""
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return False
        mu = margin_deg / 360.0 * self.pano_w
        mv = margin_deg / 180.0 * self.pano_h
        u0, u1 = self.u_range
        v0, v1 = self.v_range
        u0, u1, v0, v1 = u0 - mu, u1 + mu, v0 - mv, v1 + mv
        ix0, ix1 = max(x, u0), min(x + w, u1)
        iy0, iy1 = max(y, v0), min(y + h, v1)
        if ix1 <= ix0 or iy1 <= iy0:
            return False
        return ((ix1 - ix0) * (iy1 - iy0)) / (w * h) >= min_fraction

    # -- panorama pixel <-> photo pixel -------------------------------------

    def pano_to_photo(self, u: float, v: float) -> tuple[float, float]:
        """Panorama pixel -> photo pixel (unclamped; may fall outside the photo)."""
        lon = (u / self.pano_w - 0.5) * 360.0
        lat = (v / self.pano_h - 0.5) * 180.0
        px = (lon / self.hfov_deg + 0.5) * self.photo_w
        py = (lat / self.vfov_deg + 0.5) * self.photo_h
        return px, py

    def photo_to_pano(self, px: float, py: float) -> tuple[float, float]:
        lon = (px / self.photo_w - 0.5) * self.hfov_deg
        lat = (py / self.photo_h - 0.5) * self.vfov_deg
        u = (lon / 360.0 + 0.5) * self.pano_w
        v = (lat / 180.0 + 0.5) * self.pano_h
        return u, v

    def photo_box(self, box, pad_fraction: float = 0.05) -> Optional[list[int]]:
        """Panorama-space [x, y, w, h] -> integer photo-space [x, y, w, h], padded
        by `pad_fraction` of its own size and clamped to the photo. None when the
        box does not reach the photo at all."""
        x, y, w, h = box
        px0, py0 = self.pano_to_photo(x, y)
        px1, py1 = self.pano_to_photo(x + w, y + h)
        pw, ph = px1 - px0, py1 - py0
        px0 -= pw * pad_fraction
        py0 -= ph * pad_fraction
        px1 += pw * pad_fraction
        py1 += ph * pad_fraction
        px0, py0 = max(0, int(math.floor(px0))), max(0, int(math.floor(py0)))
        px1, py1 = min(self.photo_w, int(math.ceil(px1))), min(self.photo_h, int(math.ceil(py1)))
        if px1 - px0 < 2 or py1 - py0 < 2:
            return None
        return [px0, py0, px1 - px0, py1 - py0]

    def scale_gain(self) -> float:
        """How many photo pixels one panorama pixel inside the cone is worth,
        horizontally. > 1 means the photo is the higher-resolution source."""
        u0, u1 = self.u_range
        return self.photo_w / max(u1 - u0, 1.0)


def captured_cone_from_context(context, pano_w: int, pano_h: int) -> Optional[CapturedCone]:
    """Build the cone from the context's camera intrinsics and input image, or
    None when either is missing (no Depth Generation stage, or no photo)."""
    from pipeline.pipeline_context import ContextKey
    from pipeline.panorama.placement import placement_fov_deg

    intrinsics = context.input_intrinsics(ContextKey.INTRINSICS)
    photo = context.input_image(ContextKey.INPUT)
    if intrinsics is None or photo is None:
        return None
    fov = float(getattr(intrinsics, "fov", 0.0) or 0.0)
    if fov <= 0.0:
        return None
    # The width the photo was actually embedded at, not the camera's own FOV --
    # see placement.py; the two differ by PLACEMENT_FOV_SCALE.
    return CapturedCone.from_fov(placement_fov_deg(fov), pano_w, pano_h, photo.width, photo.height)
