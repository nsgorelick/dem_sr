"""Loss components used by the composite loss system."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dem_film_unet import terrain_slope


@dataclass(frozen=True)
class ElevationL1Loss:
    """Weighted L1 error on elevation."""

    weight: float = 1.0
    name: str = "elev"

    def __call__(self, z_hat: torch.Tensor, z_gt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        err = (z_hat.float() - z_gt.float()).abs()
        weighted = err * w.float()
        denom = w.float().sum().clamp(min=1e-12)
        return weighted.sum() / denom


@dataclass(frozen=True)
class SlopeL1Loss:
    """Weighted L1 error on terrain slope."""

    weight: float = 0.5
    name: str = "slope"

    def __call__(self, z_hat: torch.Tensor, z_gt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        s_hat = terrain_slope(z_hat.float())
        s_gt = terrain_slope(z_gt.float())
        err = (s_hat - s_gt).abs()
        weighted = err * w.float()
        denom = w.float().sum().clamp(min=1e-12)
        return weighted.sum() / denom

