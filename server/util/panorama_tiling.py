from dataclasses import dataclass

import numpy as np

# Tile grid geometry shared by every model that has to run on a whole
# equirectangular panorama in pieces (SegFormer region segmentation, SAM2
# instance segmentation, Grounding DINO detection) instead of one squashed
# whole-image pass -- each model's own tile size/overlap are its own
# concern (see its *_imp.py), but the grid math, wraparound, and box
# translation are identical everywhere, so they live here once.


@dataclass(frozen=True)
class Tile:
    x0: int  # may exceed the panorama width for tiles straddling the wrap seam — resolved via modulo on use
    y0: int  # always within [0, height - h]
    w: int
    h: int


def axis_starts(extent: int, tile: int, overlap_frac: float, wrap: bool) -> list[int]:
    """Tile start positions covering `extent`, evenly spaced with the given overlap."""
    if extent <= tile:
        return [0]

    stride = max(1, int(tile * (1.0 - overlap_frac)))
    starts = list(range(0, extent, stride))

    if wrap:
        # Equirectangular panoramas wrap horizontally — column 0 and column
        # extent-1 are the same seam in the final 360° view. Add a tile
        # explicitly centred on that seam so it's never left straddled
        # between two tile edges no matter how the stride above lands.
        starts.append(extent - tile // 2)
    else:
        # No vertical wrap (top/bottom are the zenith/nadir poles, not a
        # seam) — clamp so the last tile's far edge lands exactly on the
        # image border instead of overshooting it.
        starts = [min(s, extent - tile) for s in starts]
        starts.append(extent - tile)

    # De-dup while preserving order.
    seen: set[int] = set()
    deduped = []
    for s in starts:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def build_tiles(width: int, height: int, tile_size: int, overlap_frac: float) -> list[Tile]:
    tw = min(tile_size, width)
    th = min(tile_size, height)
    x_starts = axis_starts(width, tw, overlap_frac, wrap=True)
    y_starts = axis_starts(height, th, overlap_frac, wrap=False)
    return [Tile(x0=x, y0=y, w=tw, h=th) for y in y_starts for x in x_starts]


def extract_tile(img_arr: np.ndarray, tile: Tile) -> np.ndarray:
    width = img_arr.shape[1]
    cols = np.arange(tile.x0, tile.x0 + tile.w) % width
    rows = np.arange(tile.y0, tile.y0 + tile.h)
    return img_arr[rows][:, cols]


def to_panorama_box(box_xywh_local: list[float], tile: Tile, pano_w: int) -> list[float]:
    """Translate a tile-local [x, y, w, h] box into panorama pixel space.

    x is wrapped modulo pano_w (tile.x0 can itself already exceed pano_w for
    a seam-straddling tile — see Tile.x0's own docstring); y is not wrapped
    (no vertical wrap for equirectangular panoramas).
    """
    x, y, w, h = box_xywh_local
    return [(tile.x0 + x) % pano_w, tile.y0 + y, w, h]


def is_edge_truncated(
    box_xywh_local: list[float],
    tile: Tile,
    pano_h: int,
    margin_px: int = 12,
) -> bool:
    """True if `box` (tile-local coordinates) touches a tile edge that is a
    genuine truncation boundary -- i.e. this tile most likely clipped a
    real object that continues into a neighbouring tile.

    Horizontal edges are never truncation boundaries: tiles wrap (see
    build_tiles/axis_starts(wrap=True)), so anything touching a tile's left
    or right edge is genuinely continued in the overlapping neighbour, not
    clipped by the panorama's own border.

    Vertical edges are only real truncation boundaries away from the
    panorama's actual top/bottom (near the zenith/nadir poles, no wrap) --
    a tile whose own y0/y0+h coincides with the panorama's real border
    isn't truncating anything there, it's just the image ending.
    """
    x, y, w, h = box_xywh_local
    touches_top = y <= margin_px
    touches_bottom = (y + h) >= (tile.h - margin_px)
    real_top = tile.y0 == 0
    real_bottom = (tile.y0 + tile.h) >= pano_h
    return (touches_top and not real_top) or (touches_bottom and not real_bottom)
