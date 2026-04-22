"""Shared data/model/loss contracts for experiment code."""

from __future__ import annotations

from typing import Mapping

import torch

REQUIRED_BATCH_KEYS: tuple[str, ...] = ("x_dem", "x_ae", "z_lr", "z_gt", "w")
OPTIONAL_BATCH_KEYS: tuple[str, ...] = ("stem", "z_candidate", "z_candidate_valid")


def validate_batch(batch: Mapping[str, object], *, require_ae: bool = True) -> None:
    """Validate basic batch shape/type expectations used by train/eval."""
    required = REQUIRED_BATCH_KEYS if require_ae else ("x_dem", "z_lr", "z_gt", "w")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"batch missing required keys: {missing}")

    for key in required:
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch[{key!r}] must be a torch.Tensor, got {type(value).__name__}")

    z_gt = batch["z_gt"]
    w = batch["w"]
    if isinstance(z_gt, torch.Tensor) and isinstance(w, torch.Tensor) and z_gt.shape != w.shape:
        raise ValueError(f"batch weight shape mismatch: z_gt={tuple(z_gt.shape)} w={tuple(w.shape)}")


def validate_model_outputs(outputs: Mapping[str, object]) -> torch.Tensor:
    """Validate model output contract and return the primary prediction."""
    if "z_hat" not in outputs:
        raise KeyError("model outputs must include 'z_hat'")
    z_hat = outputs["z_hat"]
    if not isinstance(z_hat, torch.Tensor):
        raise TypeError(f"outputs['z_hat'] must be a torch.Tensor, got {type(z_hat).__name__}")
    return z_hat


def validate_loss_outputs(loss_outputs: Mapping[str, object]) -> tuple[torch.Tensor, dict[str, float]]:
    """Validate loss output contract and normalize metrics."""
    loss = loss_outputs.get("loss")
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"loss_outputs['loss'] must be a torch.Tensor, got {type(loss).__name__}")

    metrics_raw = loss_outputs.get("metrics", {})
    if not isinstance(metrics_raw, Mapping):
        raise TypeError("loss_outputs['metrics'] must be a mapping of metric names to scalars")

    metrics: dict[str, float] = {}
    for name, value in metrics_raw.items():
        if isinstance(value, torch.Tensor):
            metrics[str(name)] = float(value.detach().cpu().item())
        else:
            metrics[str(name)] = float(value)
    return loss, metrics

