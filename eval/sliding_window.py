"""Sliding-window inference utilities with overlap and blending."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    if window >= length:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def _blend_window(window_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    w1 = torch.hann_window(window_size, periodic=False, device=device, dtype=dtype)
    w2 = torch.outer(w1, w1).unsqueeze(0).unsqueeze(0)
    # Avoid zeros at tile borders so global image edges remain covered.
    return w2.clamp_min(1e-3)


@torch.no_grad()
def predict_model_sliding_window(
    *,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    model_forward: Callable[[torch.nn.Module, dict[str, Any]], dict[str, torch.Tensor]],
    tile_size: int,
    overlap: int,
    amp_enabled: bool = False,
) -> torch.Tensor:
    """Run tiled inference and blend overlapping predictions."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")

    z_lr = batch["z_lr"]
    bsz, _, height, width = z_lr.shape
    tile_h = min(tile_size, height)
    tile_w = min(tile_size, width)
    stride_h = max(tile_h - overlap, 1)
    stride_w = max(tile_w - overlap, 1)
    y_starts = _window_starts(height, tile_h, stride_h)
    x_starts = _window_starts(width, tile_w, stride_w)

    blend = _blend_window(tile_h, z_lr.device, z_lr.dtype)
    if tile_w != tile_h:
        blend_x = torch.hann_window(tile_w, periodic=False, device=z_lr.device, dtype=z_lr.dtype)
        blend = (blend.squeeze(0).squeeze(0)[:, :1] * blend_x.unsqueeze(0)).unsqueeze(0).unsqueeze(0).clamp_min(1e-3)

    out = torch.zeros((bsz, 1, height, width), device=z_lr.device, dtype=z_lr.dtype)
    den = torch.zeros((bsz, 1, height, width), device=z_lr.device, dtype=z_lr.dtype)

    for y0 in y_starts:
        y1 = y0 + tile_h
        for x0 in x_starts:
            x1 = x0 + tile_w
            tile_batch: dict[str, torch.Tensor] = {}
            for key, value in batch.items():
                if not isinstance(value, torch.Tensor):
                    continue
                if value.ndim >= 4:
                    tile_batch[key] = value[:, :, y0:y1, x0:x1]
                else:
                    tile_batch[key] = value
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model_forward(model, tile_batch)
            if "z_hat" not in outputs:
                raise KeyError("model_forward outputs must include 'z_hat'")
            pred_tile = outputs["z_hat"]
            if pred_tile.shape[-2:] != (tile_h, tile_w):
                raise ValueError(
                    f"model tile output shape mismatch: expected {(tile_h, tile_w)}, got {tuple(pred_tile.shape[-2:])}"
                )
            w = blend[..., :tile_h, :tile_w]
            out[:, :, y0:y1, x0:x1] += pred_tile * w
            den[:, :, y0:y1, x0:x1] += w

    return out / den.clamp_min(1e-8)

