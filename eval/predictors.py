"""Prediction-source helpers for experiment evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from eval.sliding_window import predict_model_sliding_window


def predict_model(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    model_forward: Callable[[torch.nn.Module, dict[str, Any]], dict[str, torch.Tensor]],
    *,
    sliding_window_tile_size: int | None = None,
    sliding_window_overlap: int = 0,
    amp_enabled: bool = False,
) -> torch.Tensor:
    if sliding_window_tile_size is not None:
        return predict_model_sliding_window(
            model=model,
            batch=batch,
            model_forward=model_forward,
            tile_size=int(sliding_window_tile_size),
            overlap=int(sliding_window_overlap),
            amp_enabled=amp_enabled,
        )
    outputs = model_forward(model, batch)
    if "z_hat" not in outputs:
        raise KeyError("model_forward outputs must include 'z_hat'")
    return outputs["z_hat"]


def predict_z_lr(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["z_lr"]


def predict_stage_a(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    model_forward: Callable[[torch.nn.Module, dict[str, Any]], dict[str, torch.Tensor]],
) -> torch.Tensor:
    outputs = model_forward(model, batch)
    if "z_stage_a" not in outputs:
        raise KeyError("model_forward outputs must include 'z_stage_a' for prediction-source stage_a")
    return outputs["z_stage_a"]

