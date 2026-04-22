"""Baseline FiLM experiment implementation."""

from __future__ import annotations

from typing import Any

import torch

from dem_film_unet import (
    ARCH_FILM,
    LOSS_PRESET_BASELINE,
)
from experiments.base import Experiment, LossBundle
from losses.composite import build_composite_loss_from_config
from models.wrappers.factory import create_experiment_model


class BaselineFilmExperiment(Experiment):
    name = "baseline"

    def build_model(self, cfg: dict[str, Any]) -> torch.nn.Module:
        arch = str(cfg.get("arch", ARCH_FILM))
        return create_experiment_model(arch)

    def model_forward(self, model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        z_hat = model(batch["x_dem"], batch["x_ae"], batch["z_lr"])
        return {"z_hat": z_hat}

    def build_loss(self, cfg: dict[str, Any]):
        loss_cfg = dict(cfg)
        loss_cfg.setdefault("loss_preset", LOSS_PRESET_BASELINE)
        composite = build_composite_loss_from_config(loss_cfg)

        def _loss_fn(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> LossBundle:
            bundle = composite(outputs, batch)
            return LossBundle(loss=bundle.loss, metrics=bundle.metrics)

        return _loss_fn

