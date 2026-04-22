"""Composite loss wrapper for combining multiple components."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch


class LossComponent(Protocol):
    name: str
    weight: float

    def __call__(self, z_hat: torch.Tensor, z_gt: torch.Tensor, w: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class CompositeLossBundle:
    loss: torch.Tensor
    metrics: dict[str, float]


class CompositeLoss:
    """Weighted composition of reusable loss components."""

    def __init__(self, components: Sequence[LossComponent]) -> None:
        self.components = list(components)
        if not self.components:
            raise ValueError("CompositeLoss requires at least one component")

    def __call__(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> CompositeLossBundle:
        z_hat = outputs["z_hat"]
        z_gt = batch["z_gt"]
        w = batch["w"]

        total = torch.zeros((), device=z_hat.device, dtype=z_hat.dtype)
        metrics: dict[str, float] = {}
        for component in self.components:
            value = component(z_hat, z_gt, w)
            weighted = float(component.weight) * value
            total = total + weighted
            metrics[component.name] = float(value.detach())
        return CompositeLossBundle(loss=total, metrics=metrics)

