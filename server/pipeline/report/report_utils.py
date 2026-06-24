"""Shared helpers for building report sections."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from util.depth_utils import Depth


def colorize_depth(depth_obj: "Depth") -> "PILImage.Image":
    """Convert a Depth object to a colourised PIL image (inferno colormap, near=bright)."""
    import matplotlib.cm as cm
    import numpy as np
    from PIL import Image as PILImage

    d = depth_obj.depth.copy().astype(np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    dmin, dmax = d.min(), d.max()
    if dmax > dmin:
        d = 1.0 - (d - dmin) / (dmax - dmin)  # invert so near=bright
    else:
        d = np.zeros_like(d)
    rgba = cm.inferno(d)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return PILImage.fromarray(rgb)
