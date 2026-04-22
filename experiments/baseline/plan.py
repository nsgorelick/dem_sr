"""Baseline FiLM experiment implementation."""

from __future__ import annotations

import argparse
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

    @classmethod
    def add_train_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--loss-system",
            choices=("preset", "composite"),
            default="preset",
            help="Loss implementation path for experiment training.",
        )
        parser.add_argument("--lambda-elev", type=float, default=1.0)
        parser.add_argument("--lambda-slope", type=float, default=0.5)
        parser.add_argument("--lambda-grad", type=float, default=0.25)
        parser.add_argument("--lambda-curv", type=float, default=0.1)
        parser.add_argument("--lambda-ms", type=float, default=0.5)
        parser.add_argument("--lambda-sdf", type=float, default=0.5)
        parser.add_argument("--lambda-contour", type=float, default=0.25)
        parser.add_argument("--lambda-hydro-flow", type=float, default=0.01)
        parser.add_argument("--lambda-hydro-pit-spike", type=float, default=0.005)
        parser.add_argument(
            "--enable-hydro-flow",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Enable hydrology flow-direction proxy loss term.",
        )
        parser.add_argument(
            "--enable-hydro-pit-spike",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Enable hydrology pit/spike penalty term.",
        )
        parser.add_argument("--hydro-flow-temperature", type=float, default=8.0)
        parser.add_argument("--hydro-pit-kernel-size", type=int, default=3)
        parser.add_argument("--hydro-water-downweight", type=float, default=0.95)
        parser.add_argument("--hydro-shoreline-downweight", type=float, default=0.7)
        parser.add_argument("--guidance-dropout", type=float, default=0.3)

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
