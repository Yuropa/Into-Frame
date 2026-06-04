"""
Local override of worldgen.utils.general_utils.map_image_to_pano.

The upstream version fills panorama holes by calling batch_nearest_dot, which
builds a (batch, N_src) similarity matrix on GPU. With a 1024x2048 source image
N_src ≈ 1.5M, so even the default batch=8192 tries to allocate ~48 GB — more
than any single GPU has.

This version replaces that step with scipy distance_transform_edt, which finds
the nearest filled pixel in 2D UV space in O(n) time with negligible memory.
All other logic is identical to the upstream function.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage import measure, draw

from worldgen.utils.general_utils import resize_img_and_rays, pano_unit_rays, fill_mask_from_contour

DEFAULT_HFOV_DEG = 60.0


def perspective_rays(h: int, w: int, hfov_deg: float, device) -> torch.Tensor:
    """Per-pixel unit rays for a perspective camera looking along +Z, Y-up."""
    hfov = math.radians(hfov_deg)
    vfov = 2 * math.atan(math.tan(hfov / 2) * h / w)

    u = (torch.arange(w, device=device).float() + 0.5) / w   # [0, 1]
    v = (torch.arange(h, device=device).float() + 0.5) / h   # [0, 1]
    vv, uu = torch.meshgrid(v, u, indexing="ij")

    phi   = (uu - 0.5) * hfov          # left → right
    theta = (vv - 0.5) * vfov          # image row 0 → negative theta → top of panorama

    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta)
    z = torch.cos(theta) * torch.cos(phi)
    return F.normalize(torch.stack([x, y, z], dim=-1), dim=-1)  # (H, W, 3)


def map_image_to_pano(predictions: dict,
                      crop_center: bool = False,
                      map_h: int = 1024,
                      map_w: int = 2048,
                      nn_batch: int = 8192,   # kept for signature compat, unused
                      device: torch.device = 'cuda'):
    rays_src = predictions["rays"]
    rgb_src  = predictions["rgb"].float()

    rgb_src, rays_src = resize_img_and_rays(rgb_src, rays_src, map_h, map_w)
    img_flat  = rgb_src.reshape(-1, 3)
    rays_flat = rays_src.reshape(-1, 3)

    x, y, z = rays_flat[:, 0], rays_flat[:, 1], rays_flat[:, 2]
    phi   = torch.atan2(x, z)
    theta = torch.asin(torch.clamp(y, -1.0, 1.0))
    u = (phi / torch.pi + 1) / 2
    v = (theta / torch.pi + 0.5)
    u_pix = (u * (map_w - 1)).long()
    v_pix = (v * (map_h - 1)).long()

    pano = torch.zeros((map_h, map_w, 3), dtype=rgb_src.dtype, device=rgb_src.device)
    pano[v_pix, u_pix] = img_flat
    hit_mask = torch.zeros((map_h, map_w), dtype=torch.bool, device=rgb_src.device)
    hit_mask[v_pix, u_pix] = True

    valid_mask = hit_mask

    # Fill holes: for each unfilled pixel copy colour from the nearest filled
    # pixel in UV space using a distance transform — no big matrix multiply.
    hole_mask = ~hit_mask
    if hole_mask.any():
        hit_np = hit_mask.cpu().numpy()                      # (H, W) bool
        _, (row_nn, col_nn) = distance_transform_edt(
            ~hit_np, return_indices=True
        )
        hole_rows, hole_cols = np.where(~hit_np)
        src_rows = row_nn[hole_rows, hole_cols]
        src_cols = col_nn[hole_rows, hole_cols]

        pano_cpu = pano.cpu()
        pano_cpu[hole_rows, hole_cols] = pano_cpu[src_rows, src_cols]
        pano = pano_cpu.to(device)

    if crop_center:
        coords = torch.stack((v_pix, u_pix), dim=-1)
        top_left, bottom_right = coords[0], coords[-1]
        valid_mask = torch.zeros((map_h, map_w), dtype=torch.bool, device=rgb_src.device)
        valid_mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = True
    else:
        valid_mask = F.max_pool2d(valid_mask.unsqueeze(0).float(), kernel_size=3, stride=1, padding=1).squeeze(0)
        valid_mask = fill_mask_from_contour(valid_mask).to(device)

    pano = pano * valid_mask[..., None]

    pano_img = Image.fromarray(pano.clamp(0, 255).cpu().numpy().astype(np.uint8))
    invalid_mask = 1 - valid_mask.float()
    invalid_mask_img = Image.fromarray((invalid_mask.cpu().numpy() * 255).astype(np.uint8))

    return pano_img, invalid_mask_img
