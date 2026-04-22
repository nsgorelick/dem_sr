"""Composite loss wrapper for combining multiple components."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from dem_film_unet import (
    DEFAULT_CONTOUR_INTERVAL_M,
    LOSS_PRESET_BASELINE,
    LOSS_PRESET_CHOICES,
    LOSS_PRESET_CONTOUR,
    LOSS_PRESET_GEOM,
    LOSS_PRESET_MULTITASK,
)
from losses.components import (
    ContourIndicatorL1Loss,
    ContourSDFL1Loss,
    CurvatureL1Loss,
    ElevationSmoothL1Loss,
    GradientL1Loss,
    MultiScaleElevationSmoothL1Loss,
    SlopeL1Loss,
)


class LossComponent(Protocol):
    name: str

    def __call__(self, z_hat: torch.Tensor, z_gt: torch.Tensor, w: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class CompositeLossBundle:
    loss: torch.Tensor
    metrics: dict[str, float]


class CompositeLoss:
    """Weighted composition of reusable loss components."""

    def __init__(self, components: Sequence[tuple[LossComponent, float, bool]]) -> None:
        self.components = list(components)
        if not self.components:
            raise ValueError("CompositeLoss requires at least one component")

    def __call__(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> CompositeLossBundle:
        z_hat = outputs["z_hat"]
        z_gt = batch["z_gt"]
        w = batch["w"]

        total = torch.zeros((), device=z_hat.device, dtype=z_hat.dtype)
        metrics: dict[str, float] = {}
        for component, weight, enabled in self.components:
            if not enabled:
                continue
            value = component(z_hat, z_gt, w)
            weighted = float(weight) * value
            total = total + weighted
            metrics[component.name] = float(value.detach())
        metrics["total"] = float(total.detach())
        return CompositeLossBundle(loss=total, metrics=metrics)


def _preset_component_switches(preset: str) -> dict[str, bool]:
    if preset not in LOSS_PRESET_CHOICES:
        raise ValueError(f"Unknown preset={preset!r}; expected one of {LOSS_PRESET_CHOICES}")
    if preset == LOSS_PRESET_BASELINE:
        return {"grad": False, "curv": False, "ms_elev": False, "sdf": False, "contour": False}
    if preset == LOSS_PRESET_GEOM:
        return {"grad": True, "curv": True, "ms_elev": True, "sdf": False, "contour": False}
    if preset == LOSS_PRESET_CONTOUR:
        return {"grad": False, "curv": False, "ms_elev": False, "sdf": True, "contour": False}
    if preset == LOSS_PRESET_MULTITASK:
        return {"grad": True, "curv": True, "ms_elev": True, "sdf": True, "contour": True}
    raise AssertionError("unreachable")


def build_composite_loss_from_config(cfg: dict[str, Any]) -> CompositeLoss:
    """Build composable loss matching legacy presets by default."""
    preset = str(cfg.get("loss_preset", LOSS_PRESET_BASELINE))
    switches = _preset_component_switches(preset)
    contour_interval = float(cfg.get("contour_interval", DEFAULT_CONTOUR_INTERVAL_M))
    smooth_l1_beta = float(cfg.get("smooth_l1_beta", 1.0))
    components: list[tuple[LossComponent, float, bool]] = [
        (
            ElevationSmoothL1Loss(smooth_l1_beta=smooth_l1_beta),
            float(cfg.get("lambda_elev", 1.0)),
            bool(cfg.get("enable_elev", True)),
        ),
        (
            SlopeL1Loss(),
            float(cfg.get("lambda_slope", 0.5)),
            bool(cfg.get("enable_slope", True)),
        ),
        (
            GradientL1Loss(),
            float(cfg.get("lambda_grad", 0.25)),
            bool(cfg.get("enable_grad", switches["grad"])),
        ),
        (
            CurvatureL1Loss(),
            float(cfg.get("lambda_curv", 0.1)),
            bool(cfg.get("enable_curv", switches["curv"])),
        ),
        (
            MultiScaleElevationSmoothL1Loss(smooth_l1_beta=smooth_l1_beta),
            float(cfg.get("lambda_ms", 0.5)),
            bool(cfg.get("enable_ms_elev", switches["ms_elev"])),
        ),
        (
            ContourSDFL1Loss(contour_interval=contour_interval),
            float(cfg.get("lambda_sdf", 0.5)),
            bool(cfg.get("enable_sdf", switches["sdf"])),
        ),
        (
            ContourIndicatorL1Loss(contour_interval=contour_interval),
            float(cfg.get("lambda_contour", 0.25)),
            bool(cfg.get("enable_contour", switches["contour"])),
        ),
    ]
    return CompositeLoss(components)

