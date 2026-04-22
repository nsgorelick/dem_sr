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

    def __call__(
        self,
        z_hat: torch.Tensor,
        z_gt: torch.Tensor,
        w: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        elev = F.smooth_l1_loss(z_hat, z_gt, beta=self.smooth_l1_beta, reduction="none")
        return (elev * w).sum() / (w.sum() + _EPS)


@dataclass(frozen=True)
class SlopeL1Loss:
    """Weighted L1 on slope magnitude."""

    name: str = "slope"

    def __call__(
        self,
        z_hat: torch.Tensor,
        z_gt: torch.Tensor,
        w: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        s_hat = terrain_slope(z_hat)
        s_gt = terrain_slope(z_gt)
        return (((s_hat - s_gt).abs()) * w).sum() / (w.sum() + _EPS)


@dataclass(frozen=True)
class GradientL1Loss:
    """Weighted L1 on gradient x/y components."""

    name: str = "grad"

    def __call__(
        self,
        z_hat: torch.Tensor,
        z_gt: torch.Tensor,
        w: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        gx_h, gy_h = terrain_grad(z_hat)
        gx_g, gy_g = terrain_grad(z_gt)
        return (((gx_h - gx_g).abs() + (gy_h - gy_g).abs()) * w).sum() / (w.sum() + _EPS)


@dataclass(frozen=True)
class CurvatureL1Loss:
    """Weighted L1 on Laplacian curvature proxy."""

    name: str = "curv"

    def __call__(
        self,
        z_hat: torch.Tensor,
        z_gt: torch.Tensor,
        w: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        lap_h = terrain_laplacian(z_hat)
        lap_g = terrain_laplacian(z_gt)
        return (((lap_h - lap_g).abs()) * w).sum() / (w.sum() + _EPS)


@dataclass(frozen=True)
class MultiScaleElevationSmoothL1Loss:
    """Weighted SmoothL1 at 2x downsampled scale."""

    smooth_l1_beta: float = 1.0
    name: str = "ms_elev"

    def __call__(
        self,
        z_hat: torch.Tensor,
        z_gt: torch.Tensor,
        w: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
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

    def __call__(
        self,
        z_hat: torch.Tensor,
        z_gt: torch.Tensor,
        w: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        sdf_h = contour_sdf(z_hat, self.contour_interval)
        sdf_g = contour_sdf(z_gt, self.contour_interval)
        return (((sdf_h - sdf_g).abs()) * w).sum() / (w.sum() + _EPS)


@dataclass(frozen=True)
class ContourIndicatorL1Loss:
    """Weighted L1 on smooth contour-indicator representation."""

    contour_interval: float = DEFAULT_CONTOUR_INTERVAL_M
    name: str = "contour"

    def __call__(
        self,
        z_hat: torch.Tensor,
        z_gt: torch.Tensor,
        w: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        c_h = contour_soft(z_hat, self.contour_interval)
        c_g = contour_soft(z_gt, self.contour_interval)
        return (((c_h - c_g).abs()) * w).sum() / (w.sum() + _EPS)


def _hydrology_weight_map(
    w: torch.Tensor,
    batch: dict[str, torch.Tensor] | None,
    *,
    water_downweight: float,
    shoreline_downweight: float,
) -> torch.Tensor:
    # Base trust comes from existing weighting/mask conventions.
    hydro_w = w.clamp(min=0.0)
    if batch is not None and "x_dem" in batch:
        x_dem = batch["x_dem"]
        if x_dem.ndim >= 4 and x_dem.shape[1] >= 5:
            m_wp = x_dem[:, 3:4, :, :].clamp(0.0, 1.0)
            m_ws = x_dem[:, 4:5, :, :].clamp(0.0, 1.0)
            hydro_w = hydro_w * (1.0 - water_downweight * m_wp) * (1.0 - shoreline_downweight * m_ws)
    return hydro_w.clamp(min=0.0)


def _neighbor_drops(z: torch.Tensor) -> torch.Tensor:
    zp = F.pad(z, (1, 1, 1, 1), mode="replicate")
    east = z - zp[:, :, 1:-1, 2:]
    west = z - zp[:, :, 1:-1, :-2]
    south = z - zp[:, :, 2:, 1:-1]
    north = z - zp[:, :, :-2, 1:-1]
    return torch.cat([east, west, south, north], dim=1)


@dataclass(frozen=True)
class FlowDirectionProxyLoss:
    """Soft local downhill-direction consistency (differentiable D8 proxy)."""

    softmax_temperature: float = 8.0
    water_downweight: float = 0.95
    shoreline_downweight: float = 0.7
    name: str = "hydro_flow"

    def __call__(
        self,
        z_hat: torch.Tensor,
        z_gt: torch.Tensor,
        w: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        drops_hat = _neighbor_drops(z_hat)
        drops_gt = _neighbor_drops(z_gt)
        t = float(self.softmax_temperature)
        p_hat = torch.softmax(t * drops_hat, dim=1)
        p_gt = torch.softmax(t * drops_gt, dim=1)
        err = (p_hat - p_gt).abs().sum(dim=1, keepdim=True)
        hydro_w = _hydrology_weight_map(
            w,
            batch,
            water_downweight=float(self.water_downweight),
            shoreline_downweight=float(self.shoreline_downweight),
        )
        return (err * hydro_w).sum() / (hydro_w.sum() + _EPS)


@dataclass(frozen=True)
class PitSpikePenaltyLoss:
    """Penalty on excess local extrema / roughness (pits + spikes)."""

    kernel_size: int = 3
    water_downweight: float = 0.95
    shoreline_downweight: float = 0.7
    name: str = "hydro_pit_spike"

    def __call__(
        self,
        z_hat: torch.Tensor,
        z_gt: torch.Tensor,
        w: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        k = int(self.kernel_size)
        if k < 3 or (k % 2) == 0:
            raise ValueError("kernel_size must be odd and >= 3")
        pad = k // 2
        mean_hat = F.avg_pool2d(z_hat, kernel_size=k, stride=1, padding=pad)
        mean_gt = F.avg_pool2d(z_gt, kernel_size=k, stride=1, padding=pad)
        dev_hat = (z_hat - mean_hat).abs()
        dev_gt = (z_gt - mean_gt).abs()
        # Penalize only roughness in prediction beyond what GT already contains.
        err = torch.relu(dev_hat - dev_gt)
        hydro_w = _hydrology_weight_map(
            w,
            batch,
            water_downweight=float(self.water_downweight),
            shoreline_downweight=float(self.shoreline_downweight),
        )
        return (err * hydro_w).sum() / (hydro_w.sum() + _EPS)

