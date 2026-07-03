"""Shared helpers for building report sections."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from util.depth_utils import Depth


def colorize_depth(depth_obj: "Depth") -> "PILImage.Image":
    """Convert a Depth object to a colourised PIL image (viridis colormap, near=bright)."""
    from PIL import Image as PILImage

    return PILImage.fromarray(depth_obj.color(), mode="RGB")
