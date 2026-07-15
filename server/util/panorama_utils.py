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
      project_3d(vertices)             — 3D world points → panorama pixel (u, v), unrestricted.
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

    def _project_vertices(
        self,
        vertices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shared equirectangular projection for project_3d and sample_3d."""
        X = vertices[:, 0].astype(np.float64)
        Y = vertices[:, 1].astype(np.float64)
        Z = vertices[:, 2].astype(np.float64)
        W, H = self.width, self.height

        r_xz = np.sqrt(X ** 2 + Z ** 2).clip(1e-6)
        lat  = np.arctan2(Y, r_xz)
        lon  = np.arctan2(X, Z)

        pu = ((lon + np.pi) / (2.0 * np.pi)) * (W - 1)
        pv = (0.5 - lat / np.pi) * (H - 1)
        return pu, pv, lat

    def project_3d(self, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Project 3D world-space vertices onto this panorama's pixel coordinates,
        with no horizon restriction.

        Unlike sample_3d, whose sky_mask gating exists because it fetches real
        photo colour at the projected pixel (a sky pixel has no ground-truth
        colour to speak of), this is for callers that only need "where would
        this point draw on the panorama," e.g. debug-overlaying an above-horizon
        feature like a mountain ridgeline silhouette. pu, pv are well-defined for
        any vertex not exactly on the vertical poles (X=Z=0), regardless of
        whether it's above or below eye level, so there is nothing to gate here.

        vertices: (N, 3) float array of (X, Y, Z) positions.
        Returns (pu, pv): float pixel column [0, W-1], row [0, H-1].
        """
        pu, pv, _ = self._project_vertices(vertices)
        return pu, pv

    def sample_3d(
        self,
        vertices: np.ndarray,
        sky_mask: np.ndarray | None = None,
        min_lat_deg: float = -35.0,
    ) -> np.ndarray:
        """
        Sample RGBA colours for 3D world-space vertices from this panorama.

        Bilinear sampling is used; the image wraps horizontally.
        Two classes of vertex are treated as holes and filled via nearest-valid-
        neighbour in the XZ plane:
          • sky                                 — determined by sky_mask (see
                                                   below), not by elevation angle.
                                                   A vertex sitting above the
                                                   camera's own height (a rise, a
                                                   hillside, a mountain slope) is
                                                   just as real and just as visible
                                                   in the panorama as one below it;
                                                   only an actual sky pixel has no
                                                   ground-truth colour to sample.
          • below min_lat_deg  (near-nadir)     — poorly-generated region; this is
                                                   the equirectangular pole
                                                   singularity, unrelated to sky.

        vertices:    (N, 3) float array.
        sky_mask:    optional (h, w) bool array in panorama pixel space, True =
                     sky. Nearest-neighbour resampled to this panorama's own
                     size first if its shape differs. When None, no vertex is
                     excluded for being "sky" -- only the near-nadir cutoff below
                     still applies.
        min_lat_deg: most-negative latitude still considered valid (degrees).
        Returns:     (N, 4) uint8 RGBA array.
        """
        X = vertices[:, 0].astype(np.float64)
        Z = vertices[:, 2].astype(np.float64)

        pano = np.array(self.rgb(), dtype=np.float32)
        H, W = pano.shape[:2]

        pu, pv, lat  = self._project_vertices(vertices)
        min_lat_rad  = np.radians(min_lat_deg)
        valid        = lat >= min_lat_rad

        if sky_mask is not None:
            sky_arr = np.asarray(sky_mask, dtype=bool)
            if sky_arr.shape != (H, W):
                sky_img = PIL.Image.fromarray((sky_arr * 255).astype(np.uint8)).resize(
                    (W, H), PIL.Image.NEAREST
                )
                sky_arr = np.asarray(sky_img) > 127
            pu_i = np.clip(np.round(pu).astype(np.int64), 0, W - 1)
            pv_i = np.clip(np.round(pv).astype(np.int64), 0, H - 1)
            valid &= ~sky_arr[pv_i, pu_i]

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

    def perspective_crop(
        self,
        box: list[float],
        mask: "np.ndarray | None" = None,
        fov_scale: float = 1.0,
    ) -> PIL.Image.Image:
        """
        Re-project the region around `box` from equirectangular onto a
        perspective (rectilinear) view, removing panoramic distortion.

        The output has the same pixel dimensions as the bounding box and covers
        exactly its angular extent (modulated by fov_scale).  Objects near the
        poles lose the horizontal stretching that equirectangular imposes; objects
        near the equator are unchanged.

        box:       [bx, by, bw, bh] in panorama pixel coordinates.
        mask:      optional (H, W) float32 array [0, 1]; when provided the
                   returned image is RGBA with the reprojected mask as alpha.
        fov_scale: multiplier on the computed FoV; >1 widens the view margin.
        Returns:   PIL Image (RGB or RGBA).
        """
        W, H = self.width, self.height
        pano = np.array(self.image.convert("RGB"), dtype=np.float32)

        bx, by, bw, bh = box
        out_w = max(2, int(round(bw)))
        out_h = max(2, int(round(bh)))

        # Centre of the box in the panorama's spherical coordinate convention:
        #   lon=0  → +Z (forward), lat=0 → horizon
        lon0 = ((bx + bw * 0.5) / W - 0.5) * 2.0 * np.pi
        lat0 = (0.5 - (by + bh * 0.5) / H) * np.pi

        # Half-FoV angles matching the box's angular extent, clamped so
        # tan() stays well-behaved (89° is safe).
        half_fov_h = min((bw / W) * np.pi * fov_scale, np.radians(89.0))
        half_fov_v = min((bh / (2.0 * H)) * np.pi * fov_scale, np.radians(89.0))

        # Output pixel grid → normalised [-1, 1] → camera-space tangent offsets.
        # Camera convention: +X right, +Y up, +Z forward (into scene).
        uu, vv = np.meshgrid(
            np.linspace(-1.0, 1.0, out_w, dtype=np.float64),
            np.linspace(-1.0, 1.0, out_h, dtype=np.float64),
        )
        x_cam =  uu * np.tan(half_fov_h)
        y_cam = -vv * np.tan(half_fov_v)   # image +v is down; camera +Y is up
        z_cam =  np.ones_like(x_cam)

        norm = np.sqrt(x_cam ** 2 + y_cam ** 2 + z_cam ** 2)
        x_cam /= norm;  y_cam /= norm;  z_cam /= norm

        # Rotation matrix: camera space → world space.
        # Camera axes in world coords (verified: R_x × R_y = R_z):
        #   +X (right)   = ( cos_lon,             0,      -sin_lon           )
        #   +Y (up)      = (-sin_lat * sin_lon,  cos_lat, -sin_lat * cos_lon )
        #   +Z (forward) = ( cos_lat * sin_lon,  sin_lat,  cos_lat * cos_lon )
        cl, sl = np.cos(lat0), np.sin(lat0)
        cn, sn = np.cos(lon0), np.sin(lon0)
        R = np.array([
            [ cn, -sl * sn,  cl * sn],
            [0.0,  cl,       sl     ],
            [-sn, -sl * cn,  cl * cn],
        ])

        dirs  = np.stack([x_cam.ravel(), y_cam.ravel(), z_cam.ravel()])  # (3, N)
        world = R @ dirs                                                   # (3, N)
        X, Y, Z = world[0], world[1], world[2]

        # World direction → equirectangular pixel coordinates (same convention
        # as equirectangular_unproject).
        lon  = np.arctan2(X, Z)
        lat  = np.arctan2(Y, np.sqrt(X ** 2 + Z ** 2))
        src_u = ((lon  / (2.0 * np.pi)) + 0.5) * (W - 1)
        src_v = (0.5 - lat / np.pi) * (H - 1)

        src_u = (src_u % W).reshape(out_h, out_w)
        src_v = np.clip(src_v, 0, H - 1).reshape(out_h, out_w)

        # Bilinear sampling with horizontal wrap.
        u0 = np.floor(src_u).astype(np.int32);  u1 = (u0 + 1) % W
        v0 = np.floor(src_v).astype(np.int32);  v1 = np.clip(v0 + 1, 0, H - 1)
        fu = (src_u - np.floor(src_u))[..., np.newaxis]
        fv = (src_v - np.floor(src_v))[..., np.newaxis]

        color = (pano[v0, u0] * (1 - fu) * (1 - fv)
               + pano[v0, u1] *      fu  * (1 - fv)
               + pano[v1, u0] * (1 - fu) *      fv
               + pano[v1, u1] *      fu  *      fv)
        color = np.clip(color, 0, 255).astype(np.uint8)

        if mask is None:
            return PIL.Image.fromarray(color, "RGB")

        # Project mask with the same sample coordinates.
        fu2, fv2 = fu[..., 0], fv[..., 0]
        mask_s = (mask[v0, u0] * (1 - fu2) * (1 - fv2)
                + mask[v0, u1] *      fu2  * (1 - fv2)
                + mask[v1, u0] * (1 - fu2) *      fv2
                + mask[v1, u1] *      fu2  *      fv2)
        alpha = np.clip(mask_s * 255, 0, 255).astype(np.uint8)
        return PIL.Image.fromarray(np.dstack([color, alpha]), "RGBA")

    def to_cubemap(self, face_w: int = 512) -> "CubeMap":
        """Convert to a CubeMap via equirectangular → cubemap projection."""
        import py360convert
        from util.cubemap_utils import CubeMap

        cube_dict = py360convert.e2c(np.array(self.rgb()), face_w=face_w, cube_format="dict")
        return CubeMap({
            k: PIL.Image.fromarray(np.clip(v, 0, 255).astype(np.uint8))
            for k, v in cube_dict.items()
        })
