from __future__ import annotations
import numpy as np
import PIL.Image
from pathlib import Path
from scipy.spatial import KDTree
from util.image_utils import Image


class Panorama:
    """
    An equirectangular (360°) panorama image.

    Owns all spherical-projection maths so that other pipeline stages don't need
    to re-implement the same trigonometry:

      equirectangular_unproject(depth) — static; depth map → (X, Y, Z) world coords.
      unproject(depth)                 — instance alias of the above.
      uv_for_3d(vertices)              — 3D world points → panorama pixel (u, v, valid).
      sample_3d(vertices)              — 3D world points → bilinear-sampled RGBA colours.
      to_cubemap(face_w)               — equirectangular → CubeMap conversion.

    Coordinate convention (Unity / camera space):
      +Z = forward, +X = right, +Y = up.
      Equirectangular: theta=0 at +Z, phi=0 at the horizon.
    """

    image: PIL.Image.Image

    def __init__(self, obj):
        if isinstance(obj, str):
            self.image = PIL.Image.open(obj).convert("RGB")
        elif isinstance(obj, Panorama):
            self.image = obj.image
        elif isinstance(obj, Image):
            self.image = obj.image
        elif isinstance(obj, PIL.Image.Image):
            self.image = obj.convert("RGB") if obj.mode != "RGB" else obj
        elif isinstance(obj, np.ndarray):
            self.image = PIL.Image.fromarray(obj).convert("RGB")
        elif isinstance(obj, Path):
            self.image = PIL.Image.open(str(obj)).convert("RGB")
        else:
            raise TypeError(f"Panorama: unsupported type {type(obj)}")

        self._rgb = None

    @classmethod
    def load(cls, path: Path) -> Panorama:
        return cls(path)

    def save(self, path):
        self.image.save(path)

    def copy(self) -> Panorama:
        return Panorama(self.image.copy())

    def rgb(self, copy: bool = False) -> PIL.Image.Image:
        if self._rgb is None:
            self._rgb = self.image.convert("RGB")
        return self._rgb.copy() if copy else self._rgb

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    @property
    def size(self) -> tuple[int, int]:
        return (self.image.width, self.image.height)

    # ── Projection ────────────────────────────────────────────────────────────

    @staticmethod
    def equirectangular_unproject(depth) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Unproject an equirectangular depth map into 3D world coordinates.

        depth: Depth whose values are radial (Euclidean) distances in metres.
        Returns (X, Y, Z) float32 arrays with the same shape as depth.depth.

        theta=0 / phi=0 maps to +Z (forward).  Y is Unity-up.
        """
        d = depth.depth.astype(np.float64)
        h, w = d.shape

        cx = np.arange(w, dtype=np.float64)
        cy = np.arange(h, dtype=np.float64)
        cx, cy = np.meshgrid(cx, cy)

        theta   = (cx / w - 0.5) * 2.0 * np.pi   # longitude [-π, π], 0 = +Z
        phi     = (0.5 - cy / h) * np.pi          # latitude  [-π/2, π/2], +ve = up
        cos_phi = np.cos(phi)

        X = (d * cos_phi * np.sin(theta)).astype(np.float32)
        Y = (d * np.sin(phi)).astype(np.float32)
        Z = (d * cos_phi * np.cos(theta)).astype(np.float32)
        return X, Y, Z

    def unproject(self, depth) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Instance convenience wrapper for equirectangular_unproject."""
        return Panorama.equirectangular_unproject(depth)

    def uv_for_3d(
        self,
        vertices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project 3D world-space vertices onto this panorama.

        vertices: (N, 3) float array of (X, Y, Z) positions.
        Returns (pu, pv, valid):
          pu    — float pixel column  in [0, W-1], wraps horizontally.
          pv    — float pixel row     in [0, H-1].
          valid — bool mask; True where the vertex is below the horizon (lat < 0).
        """
        X = vertices[:, 0].astype(np.float64)
        Y = vertices[:, 1].astype(np.float64)
        Z = vertices[:, 2].astype(np.float64)
        W, H = self.width, self.height

        r_xz = np.sqrt(X ** 2 + Z ** 2).clip(1e-6)
        lat  = np.arctan2(Y, r_xz)
        lon  = np.arctan2(X, Z)

        pu = ((lon + np.pi) / (2.0 * np.pi)) * (W - 1)
        pv = (0.5 - lat / np.pi) * (H - 1)

        return pu, pv, lat < 0.0

    def sample_3d(
        self,
        vertices: np.ndarray,
        min_lat_deg: float = -35.0,
    ) -> np.ndarray:
        """
        Sample RGBA colours for 3D world-space vertices from this panorama.

        Bilinear sampling is used; the image wraps horizontally.
        Two classes of vertex are treated as holes and filled via nearest-valid-
        neighbour in the XZ plane:
          • above the horizon  (lat >= 0)       — would sample sky
          • below min_lat_deg  (near-nadir)     — poorly-generated region

        vertices:    (N, 3) float array.
        min_lat_deg: most-negative latitude still considered valid (degrees).
        Returns:     (N, 4) uint8 RGBA array.
        """
        X = vertices[:, 0].astype(np.float64)
        Y = vertices[:, 1].astype(np.float64)
        Z = vertices[:, 2].astype(np.float64)

        pano = np.array(self.rgb(), dtype=np.float32)
        H, W = pano.shape[:2]

        r_xz        = np.sqrt(X ** 2 + Z ** 2).clip(1e-6)
        lat         = np.arctan2(Y, r_xz)
        lon         = np.arctan2(X, Z)
        min_lat_rad = np.radians(min_lat_deg)
        valid       = (lat < 0.0) & (lat >= min_lat_rad)

        pu = ((lon + np.pi) / (2.0 * np.pi)) * (W - 1)
        pv = (0.5 - lat / np.pi) * (H - 1)

        pu0 = np.floor(pu).astype(np.int32) % W
        pu1 = (pu0 + 1) % W
        pv0 = np.clip(np.floor(pv).astype(np.int32), 0, H - 1)
        pv1 = np.clip(pv0 + 1, 0, H - 1)
        fu  = (pu - np.floor(pu))[:, None]
        fv  = (pv - np.floor(pv))[:, None]

        colors_f = (pano[pv0, pu0] * (1 - fu) * (1 - fv) +
                    pano[pv0, pu1] * fu        * (1 - fv) +
                    pano[pv1, pu0] * (1 - fu)  * fv       +
                    pano[pv1, pu1] * fu         * fv)

        colors = np.zeros((len(vertices), 4), dtype=np.uint8)
        colors[:, 3] = 255
        colors[valid, :3] = np.clip(colors_f[valid], 0, 255).astype(np.uint8)

        invalid = ~valid
        if invalid.any() and valid.any():
            _, nn = KDTree(np.stack([X[valid], Z[valid]], axis=-1)).query(
                np.stack([X[invalid], Z[invalid]], axis=-1)
            )
            colors[invalid, :3] = colors[valid][nn, :3]

        return colors

    def to_cubemap(self, face_w: int = 512) -> "CubeMap":
        """Convert to a CubeMap via equirectangular → cubemap projection."""
        import py360convert
        from util.cubemap_utils import CubeMap

        cube_dict = py360convert.e2c(np.array(self.rgb()), face_w=face_w, cube_format="dict")
        return CubeMap({
            k: PIL.Image.fromarray(np.clip(v, 0, 255).astype(np.uint8))
            for k, v in cube_dict.items()
        })
