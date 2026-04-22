"""Deterministic Laplacian-style residual band decomposition utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def decompose_residual_laplacian(
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split residual into (low, mid, high) with deterministic fixed operators."""
    if residual.ndim != 4:
        raise ValueError(f"expected BCHW residual tensor, got shape={tuple(residual.shape)}")
    if residual.shape[-2] < 4 or residual.shape[-1] < 4:
        raise ValueError("residual spatial dimensions must be >= 4 for fixed 3-band decomposition")

    low_base = F.avg_pool2d(residual, kernel_size=4, stride=4)
    mid_base = F.avg_pool2d(residual, kernel_size=2, stride=2)
    low = F.interpolate(low_base, size=residual.shape[-2:], mode="nearest")
    mid_smooth = F.interpolate(mid_base, size=residual.shape[-2:], mode="nearest")
    mid = mid_smooth - low
    high = residual - mid_smooth
    return low, mid, high


def reconstruct_residual_from_bands(
    low: torch.Tensor,
    mid: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    return low + mid + high

