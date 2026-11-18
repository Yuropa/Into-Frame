"""Measured colour palettes for the species layer (docs/hybrid-scene-plan.md section 6).

Every colour a generated plant wears comes from the photograph's own pixels: a palette
is a few Lab clusters per part (bloom / foliage / stem) with their shares, read from the
masked crops of the detections in a bucket or from the plan's crop hint on the photo.
Colour words from the plan are used only to MATCH a bucket to a species, never to colour
anything.
"""
from __future__ import annotations

import math

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
    # 68 deg, not 50: a pale yellow bloom (lousewort, 53-65 deg) was counted as
    # foliage and its bucket's "bloom" was the soil and stems left over.
    foliage = (h >= 68.0) & (h <= 170.0) & (s > 0.12) & (v > 0.08)
    bloom = (~foliage) & (v > 0.15) & ((s > 0.25) | ((s < 0.18) & (v > 0.72)))
    fol = rgb01[foliage]
    stem = fol
    if len(fol) > 8:
        lum = fol @ np.array([0.299, 0.587, 0.114])
        stem = fol[lum <= np.median(lum)]
    return {"bloom": rgb01[bloom], "foliage": fol, "stem": stem}


def kmeans_lab(rgb01: np.ndarray, k: int, seed: int = 0, iters: int = 20,
               chroma_rich: bool = False) -> list[tuple[list[int], float]]:
    """[(rgb 0-255, share), ...] by descending share; k is capped by the pixel count.
    chroma_rich: each cluster is represented by the Lab median of its more saturated
    half rather than its mean, so a bloom's highlights and shadows do not average
    into pastel."""
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
        centre = centres[j]
        if chroma_rich:
            sel = lab[labels == j]
            chroma = np.hypot(sel[:, 1], sel[:, 2])
            rich = sel[chroma >= np.median(chroma)] if len(sel) >= 4 else sel
            centre = np.median(rich, axis=0)
        rgb = skcolor.lab2rgb(centre.reshape(1, 1, 3)).reshape(3)
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


def guided_bloom(rgb01: np.ndarray, reference: list[int], hue_tol_deg: float = 30.0,
                 chroma_floor: float = 0.45) -> np.ndarray:
    """The pixels that are the bloom the reference colour describes: within
    hue_tol_deg of its hue and at least chroma_floor of its Lab chroma. A masked
    crop of a flower carries stems, soil and snow; the species' own colour, read
    from the photograph at the plan's crop hint, says which pixels to keep."""
    if rgb01.shape[0] == 0:
        return rgb01
    ref = np.asarray(reference, dtype=np.float32).reshape(1, 1, 3) / 255.0
    ref_hsv = skcolor.rgb2hsv(ref).reshape(3)
    ref_lab = skcolor.rgb2lab(ref).reshape(3)
    ref_chroma = float(np.hypot(ref_lab[1], ref_lab[2]))
    hsv = skcolor.rgb2hsv(rgb01.reshape(-1, 1, 3)).reshape(-1, 3)
    lab = skcolor.rgb2lab(rgb01.reshape(-1, 1, 3)).reshape(-1, 3)
    dh = np.abs(((hsv[:, 0] - ref_hsv[0]) * 360.0 + 180.0) % 360.0 - 180.0)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    keep = (dh <= hue_tol_deg) & (chroma >= chroma_floor * ref_chroma) & (hsv[:, 2] > 0.1)
    # Not much darker than the reference (a shaded head is 25-30 L down from a
    # lit one; the hint's pale-yellow lousewort is L 86, a bucket of its smaller
    # spikes L 57), and never the foliage band unless the reference itself is
    # green: a lousewort read around an olive reference came back olive, the
    # stems and leaves inside the hue tolerance.
    keep &= lab[:, 0] >= ref_lab[0] - 32.0
    ref_hue = float(ref_hsv[0] * 360.0)
    if not (85.0 <= ref_hue <= 170.0):
        # 85, not the foliage band's own 68: a lousewort's greener spikes sit at
        # 68-85 deg and are the bloom its pale-yellow reference describes.
        hue = hsv[:, 0] * 360.0
        keep &= ~((hue >= 85.0) & (hue <= 170.0))
    return rgb01[keep]


def hue_distance(a: list[int], b: list[int], l_weight: float = 0.15) -> float:
    """CIE76 distance with the lightness axis down-weighted: the same bloom
    photographed in sun and in shade differs mostly in L (the lousewort's smaller
    spikes are 29 L below the hint's, 12 dE off in a and b)."""
    la, lb = _lab([a])[0], _lab([b])[0]
    return float(math.sqrt(l_weight * (la[0] - lb[0]) ** 2 + (la[1] - lb[1]) ** 2 + (la[2] - lb[2]) ** 2))


def word_guided_colour(rgb01: np.ndarray, anchor: list[int], hue_tol_deg: float = 25.0,
                       l_tol: float = 18.0, chroma_share: float = 0.4, min_px: int = 32) -> Optional[list[int]]:
    """The colour, in these pixels, of the thing the plan's colour word names: the
    pixels within hue_tol_deg of the word's anchor and near its lightness, read as
    the chroma-rich median of their lit half. A near-grey anchor (white, silver)
    selects by low chroma and lightness instead. None when fewer than min_px
    pixels qualify. The plan's crop hint for the lousewort holds pale-yellow
    spikes, olive stems and a paintbrush: 'pale yellow' says which is the bloom."""
    if rgb01.shape[0] == 0:
        return None
    ref = np.asarray(anchor, dtype=np.float32).reshape(1, 1, 3) / 255.0
    ref_hsv = skcolor.rgb2hsv(ref).reshape(3)
    ref_lab = skcolor.rgb2lab(ref).reshape(3)
    ref_chroma = float(np.hypot(ref_lab[1], ref_lab[2]))
    hsv = skcolor.rgb2hsv(rgb01.reshape(-1, 1, 3)).reshape(-1, 3)
    lab = skcolor.rgb2lab(rgb01.reshape(-1, 1, 3)).reshape(-1, 3)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    if ref_chroma < 15.0:
        keep = (chroma < 18.0) & (np.abs(lab[:, 0] - ref_lab[0]) <= l_tol + 7.0)
    else:
        dh = np.abs(((hsv[:, 0] - ref_hsv[0]) * 360.0 + 180.0) % 360.0 - 180.0)
        keep = (dh <= hue_tol_deg) & (np.abs(lab[:, 0] - ref_lab[0]) <= l_tol) & (chroma >= chroma_share * ref_chroma)
    if int(keep.sum()) < min_px:
        return None
    clusters = kmeans_lab(_lit_half(rgb01[keep], 0.6), 1, 0, chroma_rich=True)
    return clusters[0][0] if clusters else None


def chroma_rich_colour(colours: list[list[int]]) -> Optional[list[int]]:
    """The most saturated of a palette's colours (Lab chroma), or None."""
    if not colours:
        return None
    lab = _lab(colours)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    return list(colours[int(np.argmax(chroma))])


def measure_palette(images: Iterable[PILImage.Image], k_bloom: int = 3, k_foliage: int = 2, seed: int = 0,
                    lit_share: float = 0.6, bloom_reference: Optional[list[int]] = None) -> dict:
    """Palette of the given crops: {"bloom": [[r,g,b],...], "bloom_shares": [...],
    "foliage": [...], "stem": [...], "bloom_fraction": f, "pixels": n}.
    bloom_reference: a colour the bloom is known to be (the species' own, read from
    the photograph); the bloom pixels are then the crop's pixels of that hue."""
    px = _rgb01_from_images(images, seed=seed)
    parts = split_parts(px)
    n = int(px.shape[0])
    bloom_px = parts["bloom"]
    guided_fraction = 0.0
    if bloom_reference is not None:
        guided = guided_bloom(px, bloom_reference)
        guided_fraction = float(guided.shape[0]) / n if n else 0.0
        if guided.shape[0] >= 32:
            bloom_px = guided
    bloom = kmeans_lab(_lit_half(bloom_px, lit_share), k_bloom, seed, chroma_rich=True)
    foliage = kmeans_lab(_lit_half(parts["foliage"], lit_share), k_foliage, seed)
    stem = kmeans_lab(parts["stem"], 1, seed)
    return {
        "bloom": [c for c, _ in bloom], "bloom_shares": [round(s, 3) for _, s in bloom],
        "foliage": [c for c, _ in foliage], "foliage_shares": [round(s, 3) for _, s in foliage],
        "stem": [c for c, _ in stem],
        "bloom_fraction": round(float(bloom_px.shape[0]) / n, 4) if n else 0.0,
        "guided_fraction": round(guided_fraction, 4),
        "foliage_fraction": round(float(parts["foliage"].shape[0]) / n, 4) if n else 0.0,
        "pixels": n,
    }


def anchors_from_words(words: Iterable[str]) -> list[list[int]]:
    """Bloom anchors for the plan's colour words."""
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
