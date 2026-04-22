"""Base experiment interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LossBundle:
    """Canonical loss return type for experiment losses."""

    loss: torch.Tensor
    metrics: dict[str, float]


class Experiment:
    """Shared experiment contract for v2 entrypoints."""

    name: str = "base"

    def build_model(self, cfg: dict[str, Any]) -> torch.nn.Module:
        raise NotImplementedError

    def model_forward(self, model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def build_loss(self, cfg: dict[str, Any]):
        raise NotImplementedError

