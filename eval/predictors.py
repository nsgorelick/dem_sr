"""Prediction-source helpers for experiment evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def predict_model(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    model_forward: Callable[[torch.nn.Module, dict[str, Any]], dict[str, torch.Tensor]],
) -> torch.Tensor:
    outputs = model_forward(model, batch)
    if "z_hat" not in outputs:
        raise KeyError("model_forward outputs must include 'z_hat'")
    return outputs["z_hat"]


def predict_z_lr(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["z_lr"]

