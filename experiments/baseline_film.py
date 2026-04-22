"""Baseline FiLM experiment implementation."""

from __future__ import annotations

from typing import Any

import torch

from dem_film_unet import (
    ARCH_FILM,
    DEFAULT_CONTOUR_INTERVAL_M,
    LOSS_PRESET_BASELINE,
    loss_dem_preset,
)
from experiments.base import Experiment, LossBundle
from losses.components import ElevationL1Loss, SlopeL1Loss
from losses.composite import CompositeLoss
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
        if str(cfg.get("loss_system", "preset")).lower() == "composite":
            return CompositeLoss(
                [
                    ElevationL1Loss(weight=float(cfg.get("lambda_elev", 1.0))),
                    SlopeL1Loss(weight=float(cfg.get("lambda_slope", 0.5))),
                ]
            )

        loss_cfg = {
            "preset": cfg.get("loss_preset", LOSS_PRESET_BASELINE),
            "lambda_slope": float(cfg.get("lambda_slope", 0.5)),
            "lambda_grad": float(cfg.get("lambda_grad", 0.25)),
            "lambda_curv": float(cfg.get("lambda_curv", 0.1)),
            "lambda_ms": float(cfg.get("lambda_ms", 0.5)),
            "lambda_sdf": float(cfg.get("lambda_sdf", 0.5)),
            "lambda_contour": float(cfg.get("lambda_contour", 0.25)),
            "contour_interval": float(cfg.get("contour_interval", DEFAULT_CONTOUR_INTERVAL_M)),
        }

        def _loss_fn(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> LossBundle:
            loss, metrics_raw = loss_dem_preset(
                outputs["z_hat"],
                batch["z_gt"],
                batch["w"],
                **loss_cfg,
            )
            metrics = {name: float(value) for name, value in metrics_raw.items()}
            return LossBundle(loss=loss, metrics=metrics)

        return _loss_fn

