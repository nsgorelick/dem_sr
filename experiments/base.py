"""Base experiment interfaces."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class LossBundle:
    """Canonical loss return type for experiment losses."""

    loss: torch.Tensor
    metrics: dict[str, float]


class Experiment:
    """Shared experiment contract for v2 entrypoints."""

    name: str = "base"

    @classmethod
    def add_train_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register train CLI flags specific to this experiment (optional)."""

    @classmethod
    def add_eval_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register eval CLI flags specific to this experiment (optional)."""

    def build_model(self, cfg: dict[str, Any]) -> torch.nn.Module:
        raise NotImplementedError

    def model_forward(self, model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def build_loss(self, cfg: dict[str, Any]):
        raise NotImplementedError

    def coerce_train_arg_paths(self, args: argparse.Namespace) -> None:
        """Normalize Path-typed fields after argparse (override per plan)."""

    def coerce_eval_arg_paths(self, args: argparse.Namespace) -> None:
        """Normalize Path-typed fields for eval after argparse (override per plan)."""

    def train_epoch_log_suffix(self, metrics: dict[str, float]) -> str:
        """Extra text appended to the standard epoch log line (empty by default)."""
        return ""

    def iter_trainable_parameters(self, model: nn.Module) -> Iterable[nn.Parameter]:
        if hasattr(model, "trainable_parameters"):
            return list(model.trainable_parameters())
        return list(model.parameters())

