"""Loss components used by the composite loss system."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dem_film_unet import (
    DEFAULT_CONTOUR_INTERVAL_M,
    contour_sdf,
    contour_soft,
    terrain_grad,
    terrain_laplacian,
    terrain_slope,
)

_EPS = 1e-8


@dataclass(frozen=True)
class ElevationSmoothL1Loss:
    """Weighted SmoothL1 on elevation (matches legacy preset behavior)."""

    smooth_l1_beta: float = 1.0
    name: str = "elev"

    def __call__(self, z_hat: torch.Tensor, z_gt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        elev = F.smooth_l1_loss(z_hat, z_gt, beta=self.smooth_l1_beta, reduction="none")
        return (elev * w).sum() / (w.sum() + _EPS)


@dataclass(frozen=True)
class SlopeL1Loss:
    """Weighted L1 on slope magnitude."""

    name: str = "slope"

    def __call__(self, z_hat: torch.Tensor, z_gt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        s_hat = terrain_slope(z_hat)
        s_gt = terrain_slope(z_gt)
        return (((s_hat - s_gt).abs()) * w).sum() / (w.sum() + _EPS)


@dataclass(frozen=True)
class GradientL1Loss:
    """Weighted L1 on gradient x/y components."""

    name: str = "grad"

    def __call__(self, z_hat: torch.Tensor, z_gt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        gx_h, gy_h = terrain_grad(z_hat)
        gx_g, gy_g = terrain_grad(z_gt)
        return (((gx_h - gx_g).abs() + (gy_h - gy_g).abs()) * w).sum() / (w.sum() + _EPS)


@dataclass(frozen=True)
class CurvatureL1Loss:
    """Weighted L1 on Laplacian curvature proxy."""

    name: str = "curv"

    def __call__(self, z_hat: torch.Tensor, z_gt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        lap_h = terrain_laplacian(z_hat)
        lap_g = terrain_laplacian(z_gt)
        return (((lap_h - lap_g).abs()) * w).sum() / (w.sum() + _EPS)


@dataclass(frozen=True)
class MultiScaleElevationSmoothL1Loss:
    """Weighted SmoothL1 at 2x downsampled scale."""

    smooth_l1_beta: float = 1.0
    name: str = "ms_elev"

    def __call__(self, z_hat: torch.Tensor, z_gt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        z_hat_ds = F.avg_pool2d(z_hat, kernel_size=2, stride=2)
        z_gt_ds = F.avg_pool2d(z_gt, kernel_size=2, stride=2)
        w_ds = F.avg_pool2d(w, kernel_size=2, stride=2)
        elev_ds = F.smooth_l1_loss(z_hat_ds, z_gt_ds, beta=self.smooth_l1_beta, reduction="none")
        return (elev_ds * w_ds).sum() / (w_ds.sum() + _EPS)


@dataclass(frozen=True)
class ContourSDFL1Loss:
    """Weighted L1 on contour signed-distance representation."""

    contour_interval: float = DEFAULT_CONTOUR_INTERVAL_M
    name: str = "sdf"

    def __call__(self, z_hat: torch.Tensor, z_gt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        sdf_h = contour_sdf(z_hat, self.contour_interval)
        sdf_g = contour_sdf(z_gt, self.contour_interval)
        return (((sdf_h - sdf_g).abs()) * w).sum() / (w.sum() + _EPS)


@dataclass(frozen=True)
class ContourIndicatorL1Loss:
    """Weighted L1 on smooth contour-indicator representation."""

    contour_interval: float = DEFAULT_CONTOUR_INTERVAL_M
    name: str = "contour"

    def __call__(self, z_hat: torch.Tensor, z_gt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        c_h = contour_soft(z_hat, self.contour_interval)
        c_g = contour_soft(z_gt, self.contour_interval)
        return (((c_h - c_g).abs()) * w).sum() / (w.sum() + _EPS)

