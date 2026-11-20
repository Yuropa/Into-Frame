"""Procedural plant meshes, built inside headless Blender 5.2 and exported as GLB.

    blender-5.2 --background --factory-startup --python blender_species.py -- <spec.json> <out_dir>

Spec: {"species": [{"id", "form", "height_m", "width_m", "bloom_rgb", "foliage_rgb",
"stem_rgb", "density", "variants", "seed"}, ...]}. Forms: spike, raceme, seedhead,
conifer, tuft, shrub. Every colour on the mesh comes from the spec's palettes (bloom /
foliage / stem), sampled per face with small variation inside each palette's spread;
nothing here has a default colour.

One mesh object, one material, one 64x64 palette texture per plant (eight 8 px
swatches a row); each face's UVs sit inside its swatch. Loaders that merge materials
therefore have nothing to merge, and the colour survives any UV-preserving pass.

Built Z-up at the origin with the base on the ground; the glTF exporter writes Y-up.
Output: <out_dir>/<id>_v<k>.glb and <out_dir>/manifest.json.
"""
import json
import math
import os
import random
import struct
import sys
import zlib

import bpy
import numpy as np

SWATCH = 8          # swatches per row and column
TEX = 64            # texture size (SWATCH * 8 px)
# With an atlas (pipeline/species/atlas.py) the palette lives in a 64x64 corner
# of it, and the blooms, leaves and sprays are textured cards cut from its parts.
ATLAS = {"active": False, "size": 1024, "origin": (512, 960), "px": 64}
ROWS = {"bloom": (0, 3), "foliage": (3, 6), "stem": (6, 8)}   # swatch rows per part


# ── small vector helpers (no numpy inside Blender's python by guarantee) ─────────
def v_add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def v_sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def v_mul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def v_len(a): return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
def v_norm(a):
    n = v_len(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-9 else (0.0, 0.0, 1.0)
def v_cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def frame(d):
    """Two unit vectors perpendicular to d."""
    d = v_norm(d)
    helper = (1.0, 0.0, 0.0) if abs(d[0]) < 0.9 else (0.0, 1.0, 0.0)
    s = v_norm(v_cross(d, helper))
    t = v_cross(d, s)
    return s, t


# ── palette ──────────────────────────────────────────────────────────────────────
def _jitter(rgb, rng, amount):
    """A nearby shade: lightness and a touch of hue drift, clamped."""
    k = 1.0 + rng.uniform(-amount, amount)
    out = []
    for i, c in enumerate(rgb):
        c = c * k + rng.uniform(-amount * 40, amount * 40) * (0.5 if i == 1 else 1.0)
        out.append(max(0, min(255, int(round(c)))))
    return tuple(out)


def build_palette(spec, rng):
    """Fill each part's swatch rows from the spec's colours and their jittered shades.
    Returns (pixels rgb list of TEX*TEX, {part: [swatch indices]})."""
    pixels = [(0, 0, 0)] * (TEX * TEX)
    swatches = {}
    for part, (r0, r1) in ROWS.items():
        base = [tuple(int(v) for v in c) for c in (spec.get(f"{part}_rgb") or [])]
        if not base:
            # No colour for this part: borrow the nearest part rather than invent one.
            base = [tuple(int(v) for v in c) for c in (spec.get("foliage_rgb") or spec.get("stem_rgb") or spec.get("bloom_rgb") or [])]
        if not base:
            raise ValueError(f"{spec.get('id')}: no colours at all in the spec")
        slots = (r1 - r0) * SWATCH
        colours = []
        for i in range(slots):
            c = base[i % len(base)]
            colours.append(c if i < len(base) else _jitter(c, rng, 0.10))
        idx = []
        for i, c in enumerate(colours):
            row, col = r0 + i // SWATCH, i % SWATCH
            k = row * SWATCH + col
            idx.append(k)
            for y in range(row * 8, row * 8 + 8):
                for x in range(col * 8, col * 8 + 8):
                    pixels[y * TEX + x] = c
        swatches[part] = idx
    return pixels, swatches


def write_png(path, pixels, size):
    raw = b""
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            row += bytes(pixels[y * size + x])
        raw += bytes(row)
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def swatch_uv(k, corner):
    """UV inside swatch k; corner 0..3 spreads the face's loops over the inner half."""
    row, col = k // SWATCH, k % SWATCH
    cu, cv = (col + 0.5) / SWATCH, (row + 0.5) / SWATCH
    d = 0.2 / SWATCH
    du, dv = ((-d, -d), (d, -d), (d, d), (-d, d))[corner % 4]
    u, v = cu + du, cv + dv
    if ATLAS["active"]:
        ox, oy = ATLAS["origin"]
        u = (ox + u * ATLAS["px"]) / ATLAS["size"]
        v = (oy + v * ATLAS["px"]) / ATLAS["size"]
    # Blender's image origin is bottom-left; the PNG rows are top-down, so flip v.
    return (u, 1.0 - v)


def rect_uv(rect, corner):
    """UV of a card's corner inside an atlas part rect [u0, v0, u1, v1] (pixel rows
    top-down): corners 0, 1 at the card's base (the picture's bottom), 2, 3 at its tip."""
    u0, v0, u1, v1 = rect
    u, v = ((u0, v1), (u1, v1), (u1, v0), (u0, v0))[corner % 4]
    return (u, 1.0 - v)


# ── geometry accumulator ─────────────────────────────────────────────────────────
class Builder:
    def __init__(self, rng, swatches):
        self.rng = rng
        self.swatches = swatches
        self.verts = []
        self.faces = []
        self.face_swatch = []
        self.face_rect = []       # None, or the atlas rect the face's corners map to
        self.parts = {}           # atlas parts: name -> {"uv": [...], "aspect": w/h}
        self.soft = set()         # vertices (leaf tips) free to overshoot the drawn width

    def pick(self, part):
        return self.rng.choice(self.swatches[part])

    def add_face(self, idx, part=None, swatch=None, rect=None):
        self.faces.append(tuple(idx))
        self.face_swatch.append(swatch if swatch is not None else (0 if rect is not None else self.pick(part)))
        self.face_rect.append(rect)

    def has_part(self, name):
        return bool(ATLAS["active"] and name in self.parts)

    def card_part(self, base, up, side, length, name, width=None, mirror=False):
        """A textured quad: the atlas part `name` standing from `base` along `up`
        (its picture's bottom at the base), `width` across `side` (default the
        picture's own aspect). Returns the quad's four vertex indices."""
        info = self.parts[name]
        w = width if width is not None else length * float(info.get("aspect") or 0.5)
        u = v_norm(up); sd = v_norm(side)
        if mirror:
            sd = v_mul(sd, -1.0)
        tip = v_add(base, v_mul(u, length))
        c = [v_add(base, v_mul(sd, -0.5 * w)), v_add(base, v_mul(sd, 0.5 * w)),
             v_add(tip, v_mul(sd, 0.5 * w)), v_add(tip, v_mul(sd, -0.5 * w))]
        b = self.add_verts(c)
        self.add_face((b, b + 1, b + 2, b + 3), rect=info["uv"])
        return b

    def leaf_part(self, base, direction, length, name, min_aspect=0.3):
        """A leaf card lying across its direction (the plane of the leaf holds
        the direction and the horizontal perpendicular), so from a standing
        camera it is seen across its face, not on edge: upright leaf planes at
        random bearings read as black hairlines (Rainier band20)."""
        d = v_norm(direction)
        horiz = v_norm(v_cross(d, (0.0, 0.0, 1.0))) if abs(d[2]) < 0.98 else (1.0, 0.0, 0.0)
        info = self.parts[name]
        width = length * max(float(info.get("aspect") or 0.3), min_aspect)
        v0 = self.card_part(base, d, horiz, length, name, width=width)
        self.soft.update(range(v0, v0 + 4))

    def head_part(self, base, up, length, name, planes=3, roll=0.0, min_aspect=0.5):
        """A flower head as a fan of cards no narrower than min_aspect of its length."""
        info = self.parts[name]
        width = length * max(float(info.get("aspect") or 0.5), min_aspect)
        self.crossed_part(base, up, length, name, width=width, planes=planes, roll=roll)

    def crossed_part(self, base, up, length, name, width=None, planes=2, roll=0.0):
        """`planes` cards of the part fanned about `up` (2: a cross)."""
        u = v_norm(up)
        s0, t0 = frame(u)
        for k in range(planes):
            a = roll + math.pi * k / planes
            side = v_add(v_mul(s0, math.cos(a)), v_mul(t0, math.sin(a)))
            self.card_part(base, u, side, length, name, width=width)

    def add_verts(self, pts):
        base = len(self.verts)
        self.verts.extend(pts)
        return base

    def tube(self, p0, p1, r0, r1, part, segs=6, swatch=None):
        d = v_sub(p1, p0)
        s, t = frame(d)
        ring0, ring1 = [], []
        for i in range(segs):
            a = 2 * math.pi * i / segs
            o = v_add(v_mul(s, math.cos(a)), v_mul(t, math.sin(a)))
            ring0.append(v_add(p0, v_mul(o, r0)))
            ring1.append(v_add(p1, v_mul(o, r1)))
        b = self.add_verts(ring0 + ring1)
        sw = swatch if swatch is not None else self.pick(part)
        for i in range(segs):
            j = (i + 1) % segs
            self.add_face((b + i, b + j, b + segs + j, b + segs + i), swatch=sw)

    def polyline_tube(self, pts, r0, r1, part, segs=6):
        sw = self.pick(part)
        n = len(pts) - 1
        for i in range(n):
            ra = r0 + (r1 - r0) * i / n
            rb = r0 + (r1 - r0) * (i + 1) / n
            self.tube(pts[i], pts[i + 1], ra, rb, part, segs, swatch=sw)

    def strip(self, pts, widths, side, part, swatch=None):
        """A bent card along pts with the given half-widths across `side`."""
        sw = swatch if swatch is not None else self.pick(part)
        left = [v_add(p, v_mul(side, -w)) for p, w in zip(pts, widths)]
        right = [v_add(p, v_mul(side, w)) for p, w in zip(pts, widths)]
        b = self.add_verts(left + right)
        n = len(pts)
        for i in range(n - 1):
            self.add_face((b + i, b + i + 1, b + n + i + 1, b + n + i), swatch=sw)

    def leaf(self, base, direction, length, width, part, bend=0.35, droop=0.25, sections=3, swatch=None):
        """A lance leaf: narrow at the base, widest a third along, pointed tip, drooping."""
        d = v_norm(direction)
        s, _ = frame(d)
        pts, widths = [], []
        for i in range(sections + 1):
            f = i / sections
            p = v_add(base, v_mul(d, length * f))
            p = (p[0], p[1], p[2] - droop * length * f * f)
            pts.append(p)
            widths.append(width * 0.5 * math.sin(math.pi * min(1.0, 0.15 + f * 0.85)) + 0.002 * (i == 0))
        widths[-1] = 0.0005
        self.strip(pts, widths, s, part, swatch)

    def card(self, center, up, side, length, width, part, swatch=None):
        """A flat rectangle (two triangles) centred at `center`."""
        u, s = v_norm(up), v_norm(side)
        hl, hw = length / 2.0, width / 2.0
        pts = [v_add(v_add(center, v_mul(u, -hl)), v_mul(s, -hw)), v_add(v_add(center, v_mul(u, -hl)), v_mul(s, hw)),
               v_add(v_add(center, v_mul(u, hl)), v_mul(s, hw)), v_add(v_add(center, v_mul(u, hl)), v_mul(s, -hw))]
        b = self.add_verts(pts)
        self.add_face((b, b + 1, b + 2, b + 3), part, swatch)

    def blob(self, center, radii, part, segs=6, rings=4, swatch=None):
        """A low-poly ellipsoid (segs x rings) for pea flowers and seed heads."""
        rx, ry, rz = radii
        sw = swatch if swatch is not None else self.pick(part)
        top = self.add_verts([(center[0], center[1], center[2] + rz)])
        ring_base = []
        for r in range(1, rings):
            phi = math.pi * r / rings
            ring_base.append(self.add_verts([
                (center[0] + rx * math.sin(phi) * math.cos(2 * math.pi * i / segs),
                 center[1] + ry * math.sin(phi) * math.sin(2 * math.pi * i / segs),
                 center[2] + rz * math.cos(phi)) for i in range(segs)]))
        bottom = self.add_verts([(center[0], center[1], center[2] - rz)])
        for i in range(segs):
            j = (i + 1) % segs
            self.add_face((top, ring_base[0] + i, ring_base[0] + j), swatch=sw)
            self.add_face((bottom, ring_base[-1] + j, ring_base[-1] + i), swatch=sw)
        for r in range(len(ring_base) - 1):
            a, b = ring_base[r], ring_base[r + 1]
            for i in range(segs):
                j = (i + 1) % segs
                self.add_face((a + i, b + i, b + j, a + j), swatch=sw)

    @property
    def triangles(self):
        return sum(len(f) - 2 for f in self.faces)


# ── forms ────────────────────────────────────────────────────────────────────────
def _density_count(spec, dense, patchy, sparse):
    return {"dense": dense, "patchy": patchy, "sparse": sparse}.get(str(spec.get("density", "patchy")), patchy)


def _stem_path(rng, height, lean=0.08, steps=5):
    """A slightly leaning, slightly curved stem from the ground to `height`."""
    lx, ly = rng.uniform(-lean, lean), rng.uniform(-lean, lean)
    pts = []
    for i in range(steps + 1):
        f = i / steps
        pts.append((lx * height * f * f, ly * height * f * f, height * f))
    return pts


def form_spike(b, spec, rng, h, w):
    stem = _stem_path(rng, h)
    r = max(0.0025, w * 0.02)
    b.polyline_tube(stem, r * 1.3, r * 0.6, "stem")
    # Leaves on the lower two thirds.
    for i in range(rng.randint(16, 24) if b.has_part("leaf") else rng.randint(6, 11)):
        f = rng.uniform(0.08, 0.62)
        base = _along(stem, f)
        a = rng.uniform(0, 2 * math.pi)
        up = rng.uniform(0.35, 0.8)
        d = (math.cos(a), math.sin(a), up)
        if b.has_part("leaf"):
            b.leaf_part(base, d, rng.uniform(0.3, 0.5) * h, "leaf")
        else:
            b.leaf(base, d, rng.uniform(0.25, 0.4) * h, rng.uniform(0.03, 0.06) * w / 0.2, "foliage", droop=0.3)
    if b.has_part("head"):
        # The photographed head as a fan of three cards on the top third of the
        # stem, as wide as the head was painted and never a sliver.
        base = _along(stem, 0.62)
        up = v_sub(stem[-1], base)
        b.head_part(base, up, v_len(up) * rng.uniform(1.05, 1.2), "head", roll=rng.uniform(0, math.pi), min_aspect=0.5)
        return
    # Bloom: a cylindrical head of small bracts on the top third.
    count = _density_count(spec, 150, 110, 75)
    head_r = w * 0.5 * rng.uniform(0.75, 1.0)
    for i in range(count):
        f = 0.64 + 0.36 * (i / count) ** 0.9
        c = _along(stem, f)
        a = rng.uniform(0, 2 * math.pi)
        taper = 1.0 - 0.55 * max(0.0, (f - 0.9) / 0.1)
        out = (math.cos(a), math.sin(a), rng.uniform(0.5, 1.2))
        length = head_r * rng.uniform(0.7, 1.1) * taper
        base = v_add(c, v_mul((math.cos(a), math.sin(a), 0.0), r * 0.8))
        tip = v_add(base, v_mul(v_norm(out), length))
        side, _ = frame(out)
        mid = v_mul(v_add(base, tip), 0.5)
        b.strip([base, mid, tip], [0.004, 0.012 * w / 0.2, 0.002], side, "bloom")
    # A few green bract tips at the very top mix foliage into the bloom.
    for i in range(rng.randint(4, 9)):
        c = _along(stem, rng.uniform(0.9, 1.0))
        a = rng.uniform(0, 2 * math.pi)
        b.leaf(c, (math.cos(a), math.sin(a), 1.5), head_r * 0.9, 0.012 * w / 0.2, "foliage", droop=0.0)


def form_raceme(b, spec, rng, h, w):
    stem = _stem_path(rng, h, lean=0.06)
    r = max(0.003, w * 0.025)
    b.polyline_tube(stem, r * 1.4, r * 0.5, "stem")
    # Palmate leaf whorls on the lower half: leaflets radiating from a petiole tip.
    for wh in range(rng.randint(3, 5)):
        f = rng.uniform(0.05, 0.5)
        base = _along(stem, f)
        a0 = rng.uniform(0, 2 * math.pi)
        pet_len = rng.uniform(0.35, 0.6) * w
        pet_dir = v_norm((math.cos(a0), math.sin(a0), 0.25))
        tip = v_add(base, v_mul(pet_dir, pet_len))
        b.tube(base, tip, r * 0.5, r * 0.35, "stem", segs=4)
        if b.has_part("leaf"):
            # The palmate leaf as one card lying almost flat from the petiole's tip.
            flat = v_norm((pet_dir[0], pet_dir[1], 0.15))
            b.leaf_part(v_add(tip, v_mul(flat, -0.15 * w)), flat, rng.uniform(0.6, 0.8) * w, "leaf", min_aspect=0.7)
            continue
        n_leaf = rng.randint(6, 8)
        for i in range(n_leaf):
            a = a0 + 2 * math.pi * i / n_leaf
            d = (math.cos(a), math.sin(a), rng.uniform(0.0, 0.3))
            b.leaf(tip, d, rng.uniform(0.3, 0.45) * w, rng.uniform(0.04, 0.07) * w / 0.3, "foliage", droop=0.15)
    if b.has_part("head"):
        base = _along(stem, 0.48)
        up = v_sub(stem[-1], base)
        b.head_part(base, up, v_len(up) * rng.uniform(1.02, 1.1), "head", roll=rng.uniform(0, math.pi), min_aspect=0.4)
        return
    # Pea flowers: a tapered spiral column on the top half.
    count = _density_count(spec, 70, 55, 40)
    col_r = w * 0.5 * rng.uniform(0.6, 0.85)
    for i in range(count):
        f = 0.5 + 0.5 * i / count
        c = _along(stem, f)
        a = i * 2.39996 + rng.uniform(-0.3, 0.3)   # golden angle
        taper = 1.0 - 0.7 * max(0.0, (f - 0.85) / 0.15)
        rad = col_r * taper
        pos = v_add(c, (math.cos(a) * rad, math.sin(a) * rad, 0.0))
        size = 0.011 * w / 0.3 * rng.uniform(0.8, 1.2) * (0.6 + 0.4 * taper)
        b.blob(pos, (size, size * 0.8, size * 1.2), "bloom")
        if rng.random() < 0.35:   # the banner petal's pale standard
            b.blob(v_add(pos, (0, 0, size * 0.9)), (size * 0.6, size * 0.5, size * 0.6), "bloom")


def form_seedhead(b, spec, rng, h, w):
    stem = _stem_path(rng, h, lean=0.1)
    r = max(0.0025, w * 0.02)
    b.polyline_tube(stem, r * 1.2, r * 0.7, "stem")
    top = stem[-1]
    head_r = w * 0.5 * rng.uniform(0.5, 0.7)
    if b.has_part("head"):
        base = _along(stem, 0.78)
        up = v_sub(top, base)
        b.head_part(base, up, v_len(up) + head_r * 2.2, "head", roll=rng.uniform(0, math.pi), min_aspect=0.6)
        for i in range(rng.randint(3, 6)):
            lb = _along(stem, rng.uniform(0.0, 0.35))
            a = rng.uniform(0, 2 * math.pi)
            d = (math.cos(a), math.sin(a), 0.5)
            if b.has_part("leaf"):
                b.leaf_part(lb, d, rng.uniform(0.2, 0.35) * h, "leaf")
            else:
                b.leaf(lb, d, rng.uniform(0.2, 0.35) * h, 0.02 * w / 0.2, "foliage", droop=0.3)
        return
    b.blob(v_add(top, (0, 0, head_r * 0.9)), (head_r * 0.8, head_r * 0.8, head_r * 1.1), "bloom", segs=8, rings=5)
    # Fuzzy hairs: thin radial cards.
    count = _density_count(spec, 160, 120, 90)
    centre = v_add(top, (0, 0, head_r * 0.9))
    for i in range(count):
        d = v_norm((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0.3, 1)))
        base = v_add(centre, v_mul(d, head_r * 0.6))
        length = head_r * rng.uniform(0.6, 1.2)
        tip = v_add(base, v_mul(d, length))
        tip = (tip[0], tip[1], tip[2] - 0.3 * length)
        side, _ = frame(d)
        b.strip([base, tip], [0.0015, 0.0008], side, "bloom")
    # A few basal / stem leaves, finely divided: narrow leaflets.
    for i in range(rng.randint(5, 9)):
        base = _along(stem, rng.uniform(0.0, 0.35))
        a = rng.uniform(0, 2 * math.pi)
        b.leaf(base, (math.cos(a), math.sin(a), 0.5), rng.uniform(0.2, 0.35) * h, 0.02 * w / 0.2, "foliage", droop=0.3)


def form_conifer(b, spec, rng, h, w):
    trunk_r = max(0.04, h * 0.012)
    segs = 8
    pts = [(0.0, 0.0, h * i / 10) for i in range(11)]
    b.polyline_tube(pts, trunk_r, trunk_r * 0.12, "stem", segs=segs)
    base_h = h * rng.uniform(0.06, 0.12)
    spacing = max(0.25, h / rng.uniform(24, 32))
    z = base_h
    whorl = 0
    while z < h * 0.985:
        f = (z - base_h) / (h - base_h)
        reach = w * 0.5 * (1.0 - f) ** 0.85 * rng.uniform(0.85, 1.1) + 0.05
        n_br = rng.randint(4, 7) if f < 0.9 else rng.randint(3, 4)
        if b.has_part("spray"):
            # A spray card is one twig; a whorl of a subalpine fir is several.
            n_br = rng.randint(7, 10) if f < 0.9 else rng.randint(4, 5)
        a0 = rng.uniform(0, 2 * math.pi) + whorl * 0.7
        for i in range(n_br):
            a = a0 + 2 * math.pi * i / n_br + rng.uniform(-0.2, 0.2)
            droop = rng.uniform(-0.12, 0.08) - 0.1 * (1 - f)
            d = v_norm((math.cos(a), math.sin(a), droop))
            root = (0.0, 0.0, z)
            tip = v_add(root, v_mul(d, reach))
            tip = (tip[0], tip[1], tip[2] - 0.15 * reach * (1 - f))
            mid = v_mul(v_add(root, tip), 0.5)
            mid = (mid[0], mid[1], mid[2] + 0.02 * reach)
            side, _ = frame(d)
            if b.has_part("spray"):
                # The photographed spray twice: lying flat along the branch (a fir's
                # sprays are flat) and standing on edge, so it reads from every side.
                flat_side = v_norm(v_cross(d, (0.0, 0.0, 1.0)))
                length = reach * rng.uniform(0.95, 1.1)
                # No narrower than half its length: a painted twig is slimmer than
                # the spray it stands for, and the tree read as a bottle brush.
                width = max(length * float(b.parts["spray"].get("aspect") or 0.5), 0.5 * length)
                b.card_part(root, v_norm(v_sub(tip, root)), flat_side, length, "spray", width=width, mirror=rng.random() < 0.5)
                b.card_part(root, v_norm(v_sub(tip, root)), v_norm(v_cross(v_sub(tip, root), flat_side)), length, "spray", width=width)
                continue
            # Needle strips: a flat drooping strip plus a vertical one, both foliage.
            halfw = reach * rng.uniform(0.12, 0.18)
            b.strip([root, mid, tip], [halfw * 0.5, halfw, halfw * 0.25], side, "foliage")
            b.strip([root, mid, tip], [halfw * 0.35, halfw * 0.7, halfw * 0.2], (0.0, 0.0, 1.0), "foliage")
            # Side sprigs along the branch for the lower, longer branches.
            if reach > 0.8:
                for k in range(2):
                    fr = rng.uniform(0.3, 0.8)
                    sp = _lerp(root, tip, fr)
                    sd = v_norm(v_add(d, v_mul(side, rng.choice((-1.0, 1.0)) * 0.9)))
                    st = v_add(sp, v_mul(sd, reach * 0.3))
                    b.strip([sp, st], [halfw * 0.5, halfw * 0.15], v_norm(v_cross(sd, (0, 0, 1))), "foliage")
        z += spacing * rng.uniform(0.85, 1.15)
        whorl += 1
    # Leader.
    if b.has_part("spray"):
        b.crossed_part((0, 0, h * 0.9), (0, 0, 1), h * 0.1, "spray", planes=2, roll=rng.uniform(0, math.pi))
        return
    b.strip([(0, 0, h * 0.93), (0, 0, h)], [0.04 * w / 3.0, 0.005], (1.0, 0.0, 0.0), "foliage")
    b.strip([(0, 0, h * 0.93), (0, 0, h)], [0.04 * w / 3.0, 0.005], (0.0, 1.0, 0.0), "foliage")


def form_tuft(b, spec, rng, h, w):
    if b.has_part("blade"):
        for i in range(rng.randint(2, 4)):
            a = rng.uniform(0, 2 * math.pi)
            base = (math.cos(a) * w * 0.15, math.sin(a) * w * 0.15, 0.0)
            b.crossed_part(base, (rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), 1.0),
                           h * rng.uniform(0.8, 1.05), "blade", planes=3, roll=rng.uniform(0, math.pi))
        return
    count = _density_count(spec, 90, 60, 40)
    for i in range(count):
        a = rng.uniform(0, 2 * math.pi)
        rad = rng.uniform(0, w * 0.35)
        base = (math.cos(a) * rad, math.sin(a) * rad, 0.0)
        lean = rng.uniform(0.15, 0.6)
        d = v_norm((math.cos(a) * lean, math.sin(a) * lean, 1.0))
        length = h * rng.uniform(0.6, 1.05)
        pts, widths = [], []
        for k in range(4):
            f = k / 3
            p = v_add(base, v_mul(d, length * f))
            pts.append((p[0], p[1], p[2] - 0.45 * length * f * f))
            widths.append(0.0035 * w / 0.3 * (1.0 - 0.9 * f) + 0.0004)
        side, _ = frame(d)
        b.strip(pts, widths, side, "foliage")


def form_shrub(b, spec, rng, h, w):
    # A few woody twigs then a cloud of leaf cards on an ellipsoid shell.
    for i in range(rng.randint(4, 7)):
        a = rng.uniform(0, 2 * math.pi)
        tip = (math.cos(a) * w * 0.3, math.sin(a) * w * 0.3, h * rng.uniform(0.5, 0.9))
        b.tube((0, 0, 0), tip, 0.01 * h / 0.5, 0.004, "stem", segs=5)
    count = _density_count(spec, 320, 240, 160)
    for i in range(count):
        u, v = rng.uniform(0, 2 * math.pi), math.acos(rng.uniform(-0.6, 1.0))
        rr = rng.uniform(0.75, 1.0)
        c = (w * 0.5 * rr * math.sin(v) * math.cos(u), w * 0.5 * rr * math.sin(v) * math.sin(u),
             h * 0.15 + h * 0.45 * (1 + rr * math.cos(v)))
        n = v_norm((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0.6, 1)))
        s, t = frame(n)
        if b.has_part("leaf"):
            if i % 6 == 0:
                b.card_part(v_add(c, v_mul(n, -0.1 * h)), n, s, 0.3 * h, "leaf")
            continue
        part = "bloom" if (spec.get("bloom_rgb") and rng.random() < 0.18) else "foliage"
        b.card(c, s, t, 0.05 * h / 0.5, 0.035 * h / 0.5, part)


FORMS = {"spike": form_spike, "raceme": form_raceme, "seedhead": form_seedhead,
         "conifer": form_conifer, "tuft": form_tuft, "shrub": form_shrub}


def _lerp(a, b, f):
    return v_add(a, v_mul(v_sub(b, a), f))


def _along(path, f):
    """Point a fraction f along a polyline (by segment index, fine for stems)."""
    f = max(0.0, min(1.0, f))
    n = len(path) - 1
    i = min(n - 1, int(f * n))
    return _lerp(path[i], path[i + 1], f * n - i)


# ── Blender assembly ─────────────────────────────────────────────────────────────
def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for item in list(coll):
            if item.users == 0 or True:
                try:
                    coll.remove(item)
                except Exception:
                    pass


def make_object(name, b, png_path):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in b.verts], [], b.faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv = uv_layer.data
    for poly, k, rect in zip(mesh.polygons, b.face_swatch, b.face_rect):
        for corner, li in enumerate(poly.loop_indices):
            uv[li].uv = rect_uv(rect, corner) if rect is not None else swatch_uv(k, corner)
    img = bpy.data.images.load(png_path)
    mat = bpy.data.materials.new(name + "_mat")
    mat.use_nodes = True
    mat.use_backface_culling = False
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Closest" if not ATLAS["active"] else "Linear"
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if ATLAS["active"]:
        # The cards' cutouts: alpha from the atlas, clipped (the exporter writes
        # alphaMode MASK for a clipped material; the stage sets it again on load).
        mat.node_tree.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
        try:
            mat.blend_method = "CLIP"
            mat.alpha_threshold = 0.5
        except Exception:
            pass
        try:
            mat.surface_render_method = "DITHERED"
        except Exception:
            pass
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.8
    mesh.materials.append(mat)
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def export(ob, path):
    for o in bpy.data.objects:
        o.select_set(o == ob)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.export_scene.gltf(
        filepath=path, export_format="GLB", use_selection=True, export_apply=True,
        export_materials="EXPORT", export_image_format="AUTO", export_yup=True,
        export_normals=True, export_texcoords=True,
    )


def normalise_size(b, h, w):
    """Scale the built plant to exactly the drawn height, and no wider than the
    drawn width (bracts and leaves overshoot their nominal reach)."""
    zmax = max(v[2] for v in b.verts) or 1.0
    s = h / zmax
    b.verts = [(v[0] * s, v[1] * s, v[2] * s) for v in b.verts]
    # Leaf cards may reach past the drawn width without squeezing the whole
    # plant: measured on them, the paintbrush's head came out 4.5 cm across.
    firm = [v for i, v in enumerate(b.verts) if i not in b.soft] or b.verts
    xs = [v[0] for v in firm]; ys = [v[1] for v in firm]
    extent = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    if extent > w * 1.1:
        k = w * 1.05 / extent
        b.verts = [(v[0] * k, v[1] * k, v[2]) for v in b.verts]


def atlas_with_palette(atlas_path, pixels, sp, out_path, tint=None):
    """The species' atlas with this variant's 64x64 palette written into its
    reserved corner, saved beside the GLB and used as the material's image.
    `tint` (r, g, b multipliers) varies the painting per variant: every copy
    of a plant in one colour read as a stamped sprite (band24)."""
    try:
        img = bpy.data.images.load(atlas_path)
        w, h = img.size
        ATLAS["size"] = w
        ATLAS["origin"] = tuple(sp.get("palette_origin") or (w // 2, h - TEX))
        ATLAS["px"] = int(sp.get("palette_px") or TEX)
        arr = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(arr)
        if tint is not None:
            px = arr.reshape(-1, 4)
            px[:, :3] = np.clip(px[:, :3] * np.asarray(tint, dtype=np.float32), 0.0, 1.0)
        buf = arr.tolist()
        ox, oy = ATLAS["origin"]
        scale = ATLAS["px"] / TEX
        for y in range(ATLAS["px"]):
            for x in range(ATLAS["px"]):
                c = pixels[int(y / scale) * TEX + int(x / scale)]
                # Blender's rows run bottom-up; the origin is in top-down pixel rows.
                row = h - 1 - (oy + y)
                i = (row * w + ox + x) * 4
                buf[i:i + 4] = [c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, 1.0]
        img.pixels.foreach_set(buf)
        img.filepath_raw = out_path
        img.file_format = "PNG"
        img.save()
        return out_path
    except Exception as exc:
        print(f"atlas palette failed ({exc}); flat palette used")
        return None


def bounds_yup(verts):
    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
    # Blender Z-up -> glTF Y-up: (x, z, -y)
    return {"min": [min(xs), min(zs), -max(ys)], "max": [max(xs), max(zs), -min(ys)]}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    spec_path, out_dir = argv[0], argv[1]
    os.makedirs(out_dir, exist_ok=True)
    spec = json.load(open(spec_path))
    manifest = []
    for sp in spec["species"]:
        form = FORMS.get(str(sp.get("form", "tuft")).lower(), form_tuft)
        n_var = int(sp.get("variants", 1))
        for k in range(n_var):
            rng = random.Random(int(sp.get("seed", 0)) * 1000 + k)
            clear_scene()
            pixels, swatches = build_palette(sp, rng)
            png_path = os.path.join(out_dir, f"{sp['id']}_v{k}_palette.png")
            write_png(png_path, pixels, TEX)
            ATLAS["active"] = False
            atlas_path = sp.get("atlas")
            if atlas_path and os.path.exists(atlas_path) and sp.get("parts"):
                # Variant 0 is the painting as made; the others a touch lighter or
                # darker and warmer or cooler.
                lit = rng.uniform(0.88, 1.12) if k else 1.0
                warm = rng.uniform(-0.05, 0.05) if k else 0.0
                tint = (lit * (1.0 + warm), lit, lit * (1.0 - warm))
                png_path = atlas_with_palette(atlas_path, pixels, sp, os.path.join(out_dir, f"{sp['id']}_v{k}_atlas.png"), tint=tint)
                ATLAS["active"] = png_path is not None
            b = Builder(rng, swatches)
            if ATLAS["active"]:
                b.parts = dict(sp["parts"])
            h = float(sp["height_m"]) * rng.uniform(0.88, 1.12)
            w = float(sp["width_m"]) * rng.uniform(0.88, 1.12)
            form(b, sp, rng, h, w)
            normalise_size(b, h, w)
            ob = make_object(f"{sp['id']}_v{k}", b, png_path)
            path = os.path.join(out_dir, f"{sp['id']}_v{k}.glb")
            export(ob, path)
            entry = {"id": sp["id"], "variant": k, "file": os.path.basename(path), "form": sp.get("form"),
                     "triangles": b.triangles, "vertices": len(b.verts), "bounds": bounds_yup(b.verts),
                     "height_m": round(h, 4), "width_m": round(w, 4), "palette": os.path.basename(png_path)}
            manifest.append(entry)
            print(f"BUILT {entry['file']}: {entry['triangles']} tris, {h:.2f} m tall, {w:.2f} m wide")
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
