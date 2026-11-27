"""A model standing on a level surface, fitted to the box the photograph shows it in.

A detection box unprojected at one depth gives a width and a height on a vertical plane.
For an object seen from above and end-on that plane says little: Paris's tour boat, seen
from the bridge with its stern to the camera, is a 10 x 10 m box whose height is mostly
the boat's length foreshortened and whose width is its beam. Scaled by "the longer side
is the length" the boat came out a fragment's size, was clipped up to the floor of its
kind's length, and stood 28 x 5.5 x 5.7 m, 52 m out, where the photograph has a boat
about 22 x 7 x 4 m, 47 m out: half the photographed width on screen.

The box's four edges are four measurements, and a model on a known level has four things
to settle: how far out it stands, where across the view, how big, and how wide for its
length. `fit_to_box` projects the model's oriented bounding box and solves for those.
The side profile keeps its proportions (one scale for length and height: that is the
side the reference view was drawn from); the beam is free within `beam_limits` of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoxFit:
    x: float
    z: float
    scale: tuple[float, float, float]   # per mesh axis
    length_m: float
    beam_m: float
    height_m: float
    error: float                         # largest edge miss, as a fraction of the box's size

    def describe(self) -> str:
        return (f"{self.length_m:.1f} x {self.beam_m:.1f} x {self.height_m:.1f} m at "
                f"x={self.x:.1f} z={self.z:.1f} (edges within {100 * self.error:.0f}% of the box)")


def box_angles(box, pano_width: float, pano_height: float) -> tuple[float, float, float, float]:
    """(lon_left, lon_right, lat_bottom, lat_top) of a panorama box (x, y, w, h), radians."""
    bx, by, bw, bh = (float(v) for v in box)
    lon0 = (bx / pano_width - 0.5) * 2.0 * np.pi
    lon1 = ((bx + bw) / pano_width - 0.5) * 2.0 * np.pi
    lat_top = (0.5 - by / pano_height) * np.pi
    lat_bottom = (0.5 - (by + bh) / pano_height) * np.pi
    return lon0, lon1, lat_bottom, lat_top


def _corners(bounds, scale, yaw_deg, x, z, y_base):
    """World corners of the scaled, turned box whose underside is at y_base."""
    lo, hi = (np.asarray(b, dtype=float) for b in bounds)
    t = np.radians(yaw_deg)
    # The same convention as the yaw decision in the stage: local +X is (cos, -sin) in (x, z).
    ax = np.array([np.cos(t), -np.sin(t)])
    az = np.array([np.sin(t), np.cos(t)])
    out = []
    for lx in (lo[0], hi[0]):
        for lz in (lo[2], hi[2]):
            px, pz = np.array([x, z]) + lx * scale[0] * ax + lz * scale[2] * az
            for ly in (0.0, (hi[1] - lo[1]) * scale[1]):
                out.append((px, y_base + ly, pz))
    return np.asarray(out, dtype=float)


def _placed(points, lo, scale, yaw_deg, x, z, y_base):
    """World positions of mesh-local `points` on the scaled, turned model whose underside is at y_base."""
    pts = np.asarray(points, dtype=float)
    t = np.radians(yaw_deg)
    ax = np.array([np.cos(t), -np.sin(t)])
    az = np.array([np.sin(t), np.cos(t)])
    xz = np.array([x, z]) + (pts[:, 0:1] * scale[0]) * ax + (pts[:, 2:3] * scale[2]) * az
    return np.column_stack([xz[:, 0], y_base + (pts[:, 1] - float(lo[1])) * scale[1], xz[:, 1]])


def _projected(corners, camera, lon_ref):
    d = corners - np.asarray(camera, dtype=float)
    lon = np.arctan2(d[:, 0], d[:, 2]) - lon_ref
    lon = (lon + np.pi) % (2.0 * np.pi) - np.pi
    lat = np.arctan2(d[:, 1], np.hypot(d[:, 0], d[:, 2]))
    return np.array([lon.min(), lon.max(), lat.min(), lat.max()])


def fit_to_box(
    bounds,
    yaw_deg: float,
    position_xz: tuple[float, float],
    surface_y: float,
    draft_fraction: float,
    camera,
    angles: tuple[float, float, float, float],
    scale0: float,
    beam_limits: tuple[float, float] = (0.5, 2.0),
    scale_limits: tuple[float, float] = (1.0 / 3.0, 3.0),
    points=None,
) -> "BoxFit | None":
    """Fit the model to the box. `bounds` are the mesh's (min, max); `surface_y` the level
    it stands on and `draft_fraction` the part of its height below it; `scale0` the scale
    it would otherwise get. `points`, mesh-local, are projected in place of the bounding
    box's corners (a tapering shape's box has corners where the shape has none: a sea
    stack's box is a quarter wider on screen than the stack). None when the solver does
    not settle."""
    from scipy.optimize import least_squares

    lo, hi = (np.asarray(b, dtype=float) for b in bounds)
    ext = hi - lo
    if not np.all(ext > 1e-9) or scale0 <= 0:
        return None
    long_axis = 0 if ext[0] >= ext[2] else 2
    beam_axis = 2 - long_axis
    lon0, lon1, lat_b, lat_t = angles
    lon_ref = 0.5 * (lon0 + lon1)
    target = np.array([lon0 - lon_ref, lon1 - lon_ref, lat_b, lat_t])
    target[:2] = (target[:2] + np.pi) % (2.0 * np.pi) - np.pi
    size = np.array([target[1] - target[0]] * 2 + [target[3] - target[2]] * 2)
    if not np.all(size > 1e-6):
        return None
    cam = np.asarray(camera, dtype=float)
    x0, z0 = float(position_xz[0]), float(position_xz[1])
    r0 = float(np.hypot(x0 - cam[0], z0 - cam[2]))
    if r0 < 1e-3:
        return None
    along = np.array([x0 - cam[0], z0 - cam[2]]) / r0
    across = np.array([along[1], -along[0]])

    def unpack(p):
        s = scale0 * float(np.exp(p[0]))
        scale = [s, s, s]
        scale[beam_axis] = s * float(np.exp(p[1]))
        x, z = np.array([x0, z0]) + p[2] * r0 * along + p[3] * r0 * across
        return scale, float(x), float(z)

    def residual(p):
        scale, x, z = unpack(p)
        y_base = surface_y - draft_fraction * ext[1] * scale[1]
        if points is not None:
            world = _placed(points, lo, scale, yaw_deg, x, z, y_base)
        else:
            world = _corners((lo, hi), scale, yaw_deg, x, z, y_base)
        return (_projected(world, cam, lon_ref) - target) / size

    bounds_lo = [np.log(scale_limits[0]), np.log(beam_limits[0]), -0.6, -0.3]
    bounds_hi = [np.log(scale_limits[1]), np.log(beam_limits[1]), 0.6, 0.3]
    try:
        sol = least_squares(residual, np.zeros(4), bounds=(bounds_lo, bounds_hi), x_scale=[0.3, 0.3, 0.1, 0.05])
    except Exception:
        return None
    if not sol.success:
        return None
    scale, x, z = unpack(sol.x)
    return BoxFit(
        x=x, z=z, scale=(scale[0], scale[1], scale[2]),
        length_m=float(ext[long_axis] * scale[long_axis]),
        beam_m=float(ext[beam_axis] * scale[beam_axis]),
        height_m=float(ext[1] * scale[1]),
        error=float(np.max(np.abs(sol.fun))),
    )


def silhouette_iou(vertices, faces, bounds, fit: BoxFit, yaw_deg: float, surface_y: float, draft_fraction: float,
                   camera, angles: tuple[float, float, float, float], mask, size: int = 128) -> float:
    """How much of the photographed outline the fitted model fills: intersection over
    union of its silhouette from `camera` with `mask`, the object's cut-out (bool, any
    size) spanning the box `angles`. The box says how big; the outline says which way
    round (a sea stack with a sheer face on one side and a ramp on the other fills its
    box at any yaw)."""
    from PIL import Image, ImageDraw

    lo, hi = (np.asarray(b, dtype=float) for b in bounds)
    y_base = surface_y - draft_fraction * float(hi[1] - lo[1]) * fit.scale[1]
    world = _placed(vertices, lo, fit.scale, yaw_deg, fit.x, fit.z, y_base)
    lon0, lon1, lat_b, lat_t = angles
    lon_ref = 0.5 * (lon0 + lon1)
    d = world - np.asarray(camera, dtype=float)
    lon = (np.arctan2(d[:, 0], d[:, 2]) - lon_ref + np.pi) % (2.0 * np.pi) - np.pi
    lat = np.arctan2(d[:, 1], np.hypot(d[:, 0], d[:, 2]))
    w = ((lon1 - lon0 + np.pi) % (2.0 * np.pi)) - np.pi
    h = lat_t - lat_b
    if w <= 1e-6 or h <= 1e-6:
        return 0.0
    pad = 0.25
    u = (lon + 0.5 * w + pad * w) / (w * (1.0 + 2.0 * pad)) * size
    v = (lat_t + pad * h - lat) / (h * (1.0 + 2.0 * pad)) * size
    uv = np.column_stack([u, v])
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    for tri in uv[np.asarray(faces, dtype=np.int64)]:
        draw.polygon([tuple(p) for p in tri], fill=255)
    model = np.asarray(canvas) > 127
    target = Image.new("L", (size, size), 0)
    inner = (int(round(size / (1.0 + 2.0 * pad))),) * 2
    cut = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).resize(inner, Image.BILINEAR)
    target.paste(cut, (int(round(pad * inner[0])), int(round(pad * inner[1]))))
    photo = np.asarray(target) > 127
    union = float((model | photo).sum())
    return float((model & photo).sum()) / union if union > 0 else 0.0


def surface_samples(mesh, count: int = 60000, seed: int = 0):
    """(points, colours) on the mesh's surface: mesh-local positions and the texture's
    RGB there in [0, 1]; None when the mesh carries no texture to read."""
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    texture = getattr(material, "baseColorTexture", None)
    if texture is None:
        texture = getattr(material, "image", None)
    if uv is None or texture is None or len(uv) != len(mesh.vertices):
        return None
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri = np.asarray(mesh.vertices, dtype=float)[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if not np.isfinite(area).all() or float(area.sum()) <= 0.0:
        return None
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(faces), size=int(count), p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(count)), rng.random(count)
    bary = np.column_stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2])
    points = (tri[pick] * bary[:, :, None]).sum(axis=1)
    st = (np.asarray(uv, dtype=float)[faces[pick]] * bary[:, :, None]).sum(axis=1)
    rgb = np.asarray(texture.convert("RGB"), dtype=np.float32) / 255.0
    th, tw = rgb.shape[:2]
    px = np.clip((st[:, 0] % 1.0) * (tw - 1), 0, tw - 1).astype(np.int64)
    py = np.clip((1.0 - st[:, 1] % 1.0) * (th - 1), 0, th - 1).astype(np.int64)
    return points, rgb[py, px]


def appearance_match(points, colours, bounds, fit: BoxFit, yaw_deg: float, surface_y: float, draft_fraction: float,
                     camera, angles: tuple[float, float, float, float], cutout_rgba, size: int = 96) -> float:
    """How like the photographed cut-out the fitted model looks from `camera`: the
    correlation, over the pixels both cover, of the model's nearest surface colours
    with the cut-out's (luminance, each blurred a little; -1..1). A reconstruction
    wears the photograph on the side it was made from, so the yaw that shows that
    side scores high whatever the outline does (a sea stack's outline fills its box
    about as well from any side)."""
    from PIL import Image
    from scipy import ndimage

    lo, hi = (np.asarray(b, dtype=float) for b in bounds)
    y_base = surface_y - draft_fraction * float(hi[1] - lo[1]) * fit.scale[1]
    world = _placed(points, lo, fit.scale, yaw_deg, fit.x, fit.z, y_base)
    lon0, lon1, lat_b, lat_t = angles
    lon_ref = 0.5 * (lon0 + lon1)
    d = world - np.asarray(camera, dtype=float)
    lon = (np.arctan2(d[:, 0], d[:, 2]) - lon_ref + np.pi) % (2.0 * np.pi) - np.pi
    lat = np.arctan2(d[:, 1], np.hypot(d[:, 0], d[:, 2]))
    w = ((lon1 - lon0 + np.pi) % (2.0 * np.pi)) - np.pi
    h = lat_t - lat_b
    if w <= 1e-6 or h <= 1e-6:
        return 0.0
    u = np.floor((lon + 0.5 * w) / w * size).astype(np.int64)
    v = np.floor((lat_t - lat) / h * size).astype(np.int64)
    ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    if int(ok.sum()) < 100:
        return 0.0
    dist = np.linalg.norm(d, axis=1)[ok]
    cell = (v[ok] * size + u[ok])
    order = np.lexsort((dist, cell))                 # by cell, nearest first
    first = np.ones(len(order), dtype=bool)
    first[1:] = cell[order][1:] != cell[order][:-1]
    luma = (np.asarray(colours, dtype=np.float32)[ok] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    model = np.zeros(size * size, dtype=np.float32)
    seen = np.zeros(size * size, dtype=bool)
    model[cell[order][first]] = luma[order][first]
    seen[cell[order][first]] = True
    model, seen = model.reshape(size, size), seen.reshape(size, size)
    cut = np.asarray(cutout_rgba.convert("RGBA").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    photo = cut[..., :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    both = seen & (cut[..., 3] > 0.5)
    if int(both.sum()) < 0.1 * size * size:
        return 0.0
    m = both.astype(np.float32)
    weight = np.maximum(ndimage.gaussian_filter(m, 1.5), 1e-4)
    a = ndimage.gaussian_filter(model * m, 1.5) / weight
    b = ndimage.gaussian_filter(photo * m, 1.5) / weight
    a, b = a[both] - a[both].mean(), b[both] - b[both].mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 1e-9 else 0.0
