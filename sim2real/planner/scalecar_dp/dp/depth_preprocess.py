"""Depth preprocessing shared by the cache builder, training and the robot runtime.

Stage 1 (``downsample_z16``): real/sim Z16 640x480 (mm, 0 = invalid) -> 120x160 uint16 mm by
4x4 blocks: lower median of the valid pixels; a block with fewer than half valid pixels is invalid (0).
Stage 2 (``network_input``): metres -> 2 channels for the network: inverse depth DEPTH_MIN_M / z for
valid pixels with z clipped to [DEPTH_MIN_M, DEPTH_MAX_M] (in (0.0375, 1]; invalid = 0) and the
validity mask. Augmentations (augment_depth) run between the two stages, in metres.
NumPy and torch versions of stage 1 must agree exactly (tests/test_dp_contract.py).
"""
from __future__ import annotations

import numpy as np

from .contract import DEPTH_BLOCK, DEPTH_H, DEPTH_MAX_M, DEPTH_MIN_M, DEPTH_W

MIN_VALID = DEPTH_BLOCK * DEPTH_BLOCK // 2


def downsample_z16(z16: np.ndarray) -> np.ndarray:
    """(..., 480, 640) uint16 -> (..., 120, 160) uint16 (block lower median of valid pixels)."""
    z = np.asarray(z16)
    if z.shape[-2:] != (DEPTH_H * DEPTH_BLOCK, DEPTH_W * DEPTH_BLOCK) or z.dtype != np.uint16:
        raise ValueError(f'expected uint16 (..., {DEPTH_H * DEPTH_BLOCK}, {DEPTH_W * DEPTH_BLOCK}); got {z.dtype} {z.shape}')
    lead = z.shape[:-2]
    b = z.reshape(*lead, DEPTH_H, DEPTH_BLOCK, DEPTH_W, DEPTH_BLOCK).swapaxes(-3, -2).reshape(*lead, DEPTH_H, DEPTH_W, -1)
    b = np.sort(b, axis=-1)                        # invalid zeros first
    n = (b > 0).sum(-1)
    k = np.minimum(b.shape[-1] - n + (np.maximum(n, 1) - 1) // 2, b.shape[-1] - 1)
    med = np.take_along_axis(b, k[..., None], -1)[..., 0]
    return np.where(n >= MIN_VALID, med, 0).astype(np.uint16)


def downsample_z16_torch(z16):
    """torch version of downsample_z16 for (..., 480, 640) integer tensors (runtime on the robot)."""
    import torch
    z = z16.to(torch.int32)
    lead = z.shape[:-2]
    b = z.reshape(*lead, DEPTH_H, DEPTH_BLOCK, DEPTH_W, DEPTH_BLOCK).transpose(-3, -2).reshape(*lead, DEPTH_H, DEPTH_W, -1)
    b = torch.sort(b, dim=-1).values
    n = (b > 0).sum(-1)
    k = (b.shape[-1] - n + (n.clamp(min=1) - 1) // 2).clamp(max=b.shape[-1] - 1)
    med = torch.gather(b, -1, k[..., None])[..., 0]
    return torch.where(n >= MIN_VALID, med, torch.zeros_like(med)).to(torch.int32)


def network_input(depth_m):
    """torch (..., H, W) metres with 0/NaN = invalid -> (..., 2, H, W) [inverse depth, mask]."""
    import torch
    valid = torch.isfinite(depth_m) & (depth_m > 0)
    z = torch.where(valid, depth_m, torch.ones_like(depth_m)).clamp(DEPTH_MIN_M, DEPTH_MAX_M)
    inv = torch.where(valid, DEPTH_MIN_M / z, torch.zeros_like(z))
    return torch.stack([inv, valid.to(inv.dtype)], -3)
