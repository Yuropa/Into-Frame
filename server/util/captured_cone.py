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


def inlay_photo(panorama, photo, cone: CapturedCone, keep=None, max_width: int = 8192,
                feather_deg: float = 1.5):
    """The panorama upscaled to carry the photograph at its own resolution inside
    the cone, the photograph's pixels inlaid there.

    The panorama embeds the photo downscaled (Rainier: 1200 photo px across 652
    panorama px), and the terrain, treeline and far backdrop are all textured from
    the panorama, so from the capture camera every one of them showed half the
    photo's detail (blocky snow on the mountain, pixelated canopy on Paris's far
    bank). The output width is the panorama's times cone.scale_gain(), capped at
    max_width; outside the cone it is a plain resize.

    keep: optional bool array (any resolution) of panorama pixels whose content is
    still the photograph's -- pass ~removal_mask for an object-removed panorama,
    or the removed tour boat comes back painted on the water. The inlay fades out
    over feather_deg at the cone's edge and at keep's.

    panorama, photo: PIL images. Returns a PIL RGB image."""
    import numpy as np
    from PIL import Image as PILImage
    from scipy.ndimage import distance_transform_edt, map_coordinates

    w, h = panorama.size
    scale = min(max(1.0, cone.scale_gain()), max_width / float(w))
    W, H = int(round(w * scale)), int(round(h * scale))
    base = np.asarray(panorama.convert("RGB").resize((W, H), PILImage.LANCZOS), dtype=np.float32)
    big = CapturedCone(cone.hfov_deg, cone.vfov_deg, W, H, cone.photo_w, cone.photo_h)
    u0, u1 = big.u_range
    v0, v1 = big.v_range
    c0, c1 = max(int(np.floor(u0)), 0), min(int(np.ceil(u1)), W)
    r0, r1 = max(int(np.floor(v0)), 0), min(int(np.ceil(v1)), H)
    if c1 <= c0 or r1 <= r0:
        return PILImage.fromarray(base.astype(np.uint8))
    cols = np.arange(c0, c1, dtype=np.float64) + 0.5
    rows = np.arange(r0, r1, dtype=np.float64) + 0.5
    lon = (cols / W - 0.5) * 360.0
    lat = (rows / H - 0.5) * 180.0
    px = (lon / cone.hfov_deg + 0.5) * cone.photo_w - 0.5
    py = (lat / cone.vfov_deg + 0.5) * cone.photo_h - 0.5
    PX, PY = np.meshgrid(px, py)
    ph = np.asarray(photo.convert("RGB").resize((cone.photo_w, cone.photo_h)), dtype=np.float32)
    sample = np.stack([map_coordinates(ph[..., c], [PY, PX], order=1, mode="nearest") for c in range(3)], -1)

    # Weight: 1 well inside the photo, fading to 0 over feather_deg at its edge.
    fpx = feather_deg / 360.0 * W
    inside = (PX >= 0) & (PX <= cone.photo_w - 1) & (PY >= 0) & (PY <= cone.photo_h - 1)
    edge = np.minimum.reduce([PX, cone.photo_w - 1 - PX, PY, cone.photo_h - 1 - PY]) * (scale / max(cone.scale_gain(), 1e-6))
    weight = np.clip(edge / max(fpx, 1.0), 0.0, 1.0) * inside
    if keep is not None:
        k = np.asarray(PILImage.fromarray(np.asarray(keep, dtype=np.uint8) * 255).resize((W, H), PILImage.NEAREST))[r0:r1, c0:c1] > 127
        weight *= np.clip(distance_transform_edt(k) / max(fpx, 1.0), 0.0, 1.0)
    region = base[r0:r1, c0:c1]
    base[r0:r1, c0:c1] = region + weight[..., None] * (sample - region)
    return PILImage.fromarray(base.clip(0, 255).astype(np.uint8))


def photo_detail_ratio(panorama, photo, cone: CapturedCone, keep=None, max_width: int = 8192):
    """(ratio, scale): the photograph's fine detail as a multiplicative field at
    `scale` x the panorama's resolution, 1 wherever there is none to add.

    ratio = inlay / panorama-upsampled: the photograph where it was photographed,
    relative to what the panorama made of it. A texture built at panorama resolution --
    the treeline strip with its colour matching and restored canopy, the far
    backdrop -- is upsampled and multiplied by it, which sharpens it inside the
    photographed cone while keeping whatever gain or delighting it carries."""
    import numpy as np
    from PIL import Image as PILImage
    w, h = panorama.size
    hi = inlay_photo(panorama, photo, cone, keep=keep, max_width=max_width)
    W, H = hi.size
    # Against the panorama itself upsampled, not the inlay's own low-pass: the
    # panorama's super-resolution artifacts (Paris's dithered far-bank canopy) are
    # in the texture being multiplied, and only the full photo/panorama ratio
    # replaces them. Outside the inlay the two are the same resize: ratio 1.
    base = panorama.convert("RGB").resize((W, H), PILImage.LANCZOS)
    a = np.asarray(hi, dtype=np.float32)
    b = np.asarray(base, dtype=np.float32)
    ratio = np.clip((a + 4.0) / (b + 4.0), 0.33, 3.0)
    return ratio, W / float(w)


def apply_detail(texture_rgb, ratio, row0: int, scale: float):
    """Upsample a panorama-resolution strip (rows row0.. of the panorama, full
    width) by `scale` and multiply in the matching rows of `ratio`."""
    import numpy as np
    from PIL import Image as PILImage
    th, tw = texture_rgb.shape[:2]
    W = ratio.shape[1]
    Ht = int(round(th * scale))
    up = np.asarray(PILImage.fromarray(texture_rgb).resize((W, Ht), PILImage.BICUBIC), dtype=np.float32)
    r0 = int(round(row0 * scale))
    rr = ratio[r0:r0 + Ht]
    if rr.shape[0] < Ht:
        rr = np.concatenate([rr, np.ones((Ht - rr.shape[0], W, 3), np.float32)], 0)
    return (up * rr).clip(0, 255).astype(np.uint8)


def sharpen_strip(context, panorama, rgb_strip, alpha_strip, row0: int, max_width: int = 8192):
    """An environment strip texture (rows row0.. of `panorama`, full width, at its
    resolution) rebuilt at the photograph's resolution with photo_detail_ratio.
    Returns an RGBA PIL image, or None when there is no photo or cone, or nothing
    to gain. Pixels the foreground removal repainted get no detail."""
    import numpy as np
    from PIL import Image as PILImage
    from pipeline.pipeline_context import ContextKey
    photo = context.input_image(ContextKey.INPUT)
    w, h = panorama.width, panorama.height
    cone = captured_cone_from_context(context, w, h)
    if photo is None or cone is None or cone.scale_gain() <= 1.05 or max_width <= w:
        return None
    keep = None
    removal = context.input_image(ContextKey.PANORAMA_FOREGROUND_MASK)
    if removal is not None:
        keep = np.asarray(removal.image.convert("L").resize((w, h), PILImage.NEAREST)) <= 127
    ratio, scale = photo_detail_ratio(panorama.image.convert("RGB"), photo.image, cone, keep=keep, max_width=max_width)
    rgb = apply_detail(np.ascontiguousarray(rgb_strip), ratio, row0, scale)
    a = PILImage.fromarray((np.clip(alpha_strip, 0, 1) * 255).astype(np.uint8)).resize(
        (rgb.shape[1], rgb.shape[0]), PILImage.BILINEAR)
    return PILImage.fromarray(np.dstack([rgb, np.asarray(a)]), "RGBA")
