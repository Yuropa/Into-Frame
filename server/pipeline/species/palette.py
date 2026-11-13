"""Measured colour palettes for the species layer (docs/hybrid-scene-plan.md section 6).

Every colour a generated plant wears comes from the photograph's own pixels: a palette
is a few Lab clusters per part (bloom / foliage / stem) with their shares, read from the
masked crops of the detections in a bucket or from the plan's crop hint on the photo.
Colour words from the plan are used only to MATCH a bucket to a species, never to colour
anything.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from PIL import Image as PILImage
from skimage import color as skcolor

# Anchors for matching only (sRGB 0-255).
COLOUR_WORDS: dict[str, tuple[int, int, int]] = {
    "magenta": (200, 40, 120), "crimson": (170, 25, 60), "pink": (230, 120, 170), "red": (190, 30, 40),
    "scarlet": (210, 40, 40), "orange": (230, 130, 40), "yellow": (235, 210, 60), "pale yellow": (235, 225, 150),
    "cream": (240, 230, 190), "white": (240, 240, 235), "silver": (200, 200, 195), "grey": (150, 150, 150),
    "blue": (60, 90, 200), "violet": (120, 70, 190), "violet blue": (90, 80, 200), "violet-blue": (90, 80, 200),
    "purple": (130, 60, 170), "lavender": (180, 150, 220), "lilac": (190, 160, 220),
    "green": (70, 130, 50), "pale green": (160, 200, 130), "dark green": (30, 70, 35), "olive": (110, 120, 50),
    "yellow-green": (180, 200, 70), "brown": (110, 80, 50), "tan": (190, 160, 110),
}


def _rgb01_from_images(images: Iterable[PILImage.Image], max_px: int = 60000, seed: int = 0) -> np.ndarray:
    """Opaque pixels of every image as float RGB in [0, 1], subsampled."""
    rows = []
    for im in images:
        if im is None:
            continue
        arr = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
        if arr.size == 0:
            continue
        alpha = arr[..., 3] > 0.5
        px = arr[..., :3][alpha]
        if px.size:
            rows.append(px)
    if not rows:
        return np.zeros((0, 3), dtype=np.float32)
    px = np.concatenate(rows, axis=0)
    if len(px) > max_px:
        rng = np.random.default_rng(seed)
        px = px[rng.choice(len(px), max_px, replace=False)]
    return px


def split_parts(rgb01: np.ndarray) -> dict[str, np.ndarray]:
    """Bloom / foliage / stem pixel sets by hue and saturation. Foliage is the green
    band; bloom is saturated non-green or bright near-white; stem is the darker half of
    the foliage."""
    if rgb01.shape[0] == 0:
        return {"bloom": rgb01, "foliage": rgb01, "stem": rgb01}
    hsv = skcolor.rgb2hsv(rgb01.reshape(-1, 1, 3)).reshape(-1, 3)
    h, s, v = hsv[:, 0] * 360.0, hsv[:, 1], hsv[:, 2]
    foliage = (h >= 50.0) & (h <= 170.0) & (s > 0.12) & (v > 0.08)
    bloom = (~foliage) & (v > 0.15) & ((s > 0.25) | ((s < 0.18) & (v > 0.72)))
    fol = rgb01[foliage]
    stem = fol
    if len(fol) > 8:
        lum = fol @ np.array([0.299, 0.587, 0.114])
        stem = fol[lum <= np.median(lum)]
    return {"bloom": rgb01[bloom], "foliage": fol, "stem": stem}


def kmeans_lab(rgb01: np.ndarray, k: int, seed: int = 0, iters: int = 20) -> list[tuple[list[int], float]]:
    """[(rgb 0-255, share), ...] by descending share; k is capped by the pixel count."""
    n = rgb01.shape[0]
    if n == 0:
        return []
    k = int(max(1, min(k, n)))
    lab = skcolor.rgb2lab(rgb01.reshape(-1, 1, 3)).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    centres = lab[rng.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((lab[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        new_labels = d.argmin(1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for j in range(k):
            sel = lab[labels == j]
            if len(sel):
                centres[j] = sel.mean(0)
    out = []
    for j in range(k):
        share = float((labels == j).mean())
        if share <= 0.0:
            continue
        rgb = skcolor.lab2rgb(centres[j].reshape(1, 1, 3)).reshape(3)
        out.append(([int(round(float(c) * 255.0)) for c in np.clip(rgb, 0, 1)], share))
    out.sort(key=lambda t: -t[1])
    return out


def _lit_half(px: np.ndarray, keep: float = 0.6) -> np.ndarray:
    """The brighter `keep` share of a part's pixels. The crops are photographed under
    the scene's own sun and the client lights the plant again, so the shadowed pixels
    would darken it twice; the lit side is the closest thing to albedo without a
    delighting pass."""
    if px.shape[0] < 16:
        return px
    lum = px @ np.array([0.299, 0.587, 0.114])
    cut = np.quantile(lum, 1.0 - keep)
    return px[lum >= cut]


def measure_palette(images: Iterable[PILImage.Image], k_bloom: int = 3, k_foliage: int = 2, seed: int = 0,
                    lit_share: float = 0.6) -> dict:
    """Palette of the given crops: {"bloom": [[r,g,b],...], "bloom_shares": [...],
    "foliage": [...], "stem": [...], "bloom_fraction": f, "pixels": n}."""
    px = _rgb01_from_images(images, seed=seed)
    parts = split_parts(px)
    n = int(px.shape[0])
    bloom = kmeans_lab(_lit_half(parts["bloom"], lit_share), k_bloom, seed)
    foliage = kmeans_lab(_lit_half(parts["foliage"], lit_share), k_foliage, seed)
    stem = kmeans_lab(parts["stem"], 1, seed)
    return {
        "bloom": [c for c, _ in bloom], "bloom_shares": [round(s, 3) for _, s in bloom],
        "foliage": [c for c, _ in foliage], "foliage_shares": [round(s, 3) for _, s in foliage],
        "stem": [c for c, _ in stem],
        "bloom_fraction": round(float(parts["bloom"].shape[0]) / n, 4) if n else 0.0,
        "foliage_fraction": round(float(parts["foliage"].shape[0]) / n, 4) if n else 0.0,
        "pixels": n,
    }


def anchors_from_words(words: Iterable[str]) -> list[list[int]]:
    """Bloom anchors for the plan's colour words (matching only)."""
    out = []
    for w in words:
        key = str(w).strip().lower()
        if key in COLOUR_WORDS:
            out.append(list(COLOUR_WORDS[key]))
            continue
        for tok in key.replace("-", " ").split():
            if tok in COLOUR_WORDS:
                out.append(list(COLOUR_WORDS[tok]))
                break
    return out


def _lab(rgbs: list[list[int]]) -> np.ndarray:
    if not rgbs:
        return np.zeros((0, 3))
    arr = np.asarray(rgbs, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    return skcolor.rgb2lab(arr).reshape(-1, 3)


def palette_distance(a: list[list[int]], b: list[list[int]], a_shares: Optional[list[float]] = None) -> Optional[float]:
    """Share-weighted mean over a's clusters of the nearest b cluster (CIE76 dE)."""
    la, lb = _lab(a), _lab(b)
    if la.shape[0] == 0 or lb.shape[0] == 0:
        return None
    d = np.sqrt(((la[:, None, :] - lb[None, :, :]) ** 2).sum(-1)).min(1)
    w = np.asarray(a_shares if a_shares and len(a_shares) == len(a) else [1.0] * len(a), dtype=np.float64)
    w = w / max(float(w.sum()), 1e-9)
    return float((d * w).sum())


def delta_e_rgb(a: list[int], b: list[int]) -> float:
    la, lb = _lab([a]), _lab([b])
    return float(np.sqrt(((la - lb) ** 2).sum()))
