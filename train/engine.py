"""Shared training loop utilities for experiment entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader

from core.data_schema import validate_batch, validate_model_outputs
from experiments.base import LossBundle


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_forward: Callable[[torch.nn.Module, dict[str, Any]], dict[str, torch.Tensor]],
    loss_fn: Callable[[dict[str, torch.Tensor], dict[str, torch.Tensor]], LossBundle],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    train: bool,
) -> dict[str, float]:
    """Run one train/eval epoch and return aggregated scalar metrics."""
    if train and optimizer is None:
        raise ValueError("optimizer is required for train=True")

    model.train(mode=train)
    total_loss = 0.0
    total_batches = 0
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}

    for batch_raw in loader:
        batch: dict[str, Any] = dict(batch_raw)
        validate_batch(batch, require_ae=True)
        batch_tensors = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model_forward(model, batch_tensors)
                z_hat = validate_model_outputs(outputs)
                if z_hat.shape != batch_tensors["z_gt"].shape:
                    raise ValueError(
                        f"prediction/ground-truth shape mismatch: {tuple(z_hat.shape)} vs {tuple(batch_tensors['z_gt'].shape)}"
                    )
                bundle = loss_fn(outputs, batch_tensors)
                loss = bundle.loss

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        loss_value = float(loss.detach())
        total_loss += loss_value
        total_batches += 1
        for name, value in bundle.metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
            metric_counts[name] = metric_counts.get(name, 0) + 1

    mean_loss = total_loss / max(total_batches, 1)
    out = {"loss": mean_loss, "n_batches": float(total_batches)}
    for name, summed in metric_sums.items():
        out[name] = summed / max(metric_counts[name], 1)
    return out

