"""Frequency-domain multi-band residual experiment."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch import nn

from core.frequency import decompose_residual_laplacian, reconstruct_residual_from_bands
from dem_film_unet import ARCH_FILM, GlobalFiLM, ResBlock, Up, down_layer
from experiments.base import Experiment, LossBundle
from losses.composite import build_composite_loss_from_config


class FrequencyDomainResidualModel(nn.Module):
    """Shared backbone + low/mid/high residual heads."""

    dem_ch = (32, 64, 128, 256)
    ae_ch = (16, 32, 64, 128)
    film_alphas = (0.10, 0.15, 0.20)

    def __init__(self, *, r_cap: float = 20.0) -> None:
        super().__init__()
        self.r_cap = float(r_cap)
        d0, d1, d2, d3 = self.dem_ch
        a0, a1, a2, a3 = self.ae_ch
        self.dem_b0 = nn.Sequential(ResBlock(5, d0), ResBlock(d0, d0))
        self.down_d1 = down_layer(d0, d1)
        self.dem_b1 = nn.Sequential(ResBlock(d1, d1), ResBlock(d1, d1))
        self.down_d2 = down_layer(d1, d2)
        self.dem_b2 = nn.Sequential(ResBlock(d2, d2), ResBlock(d2, d2))
        self.down_d3 = down_layer(d2, d3)
        self.dem_b3 = nn.Sequential(ResBlock(d3, d3), ResBlock(d3, d3))

        self.ae_b0 = ResBlock(64, a0)
        self.down_a1 = down_layer(a0, a1)
        self.ae_b1 = ResBlock(a1, a1)
        self.down_a2 = down_layer(a1, a2)
        self.ae_b2 = ResBlock(a2, a2)
        self.down_a3 = down_layer(a2, a3)
        self.ae_b3 = ResBlock(a3, a3)

        self.film1 = GlobalFiLM(a1, d1)
        self.film2 = GlobalFiLM(a2, d2)
        self.film3 = GlobalFiLM(a3, d3)

        self.dec3 = nn.Sequential(ResBlock(d3, d3), ResBlock(d3, d3))
        self.up2 = Up(d3, d2)
        self.dec2 = nn.Sequential(ResBlock(d2 + d2, d2), ResBlock(d2, d2))
        self.up1 = Up(d2, d1)
        self.dec1 = nn.Sequential(ResBlock(d1 + d1, d1), ResBlock(d1, d1))
        self.up0 = Up(d1, d0)
        self.dec0 = nn.Sequential(ResBlock(d0 + d0, d0), ResBlock(d0, d0))

        self.low_head = nn.Conv2d(d0, 1, 1)
        self.mid_head = nn.Conv2d(d0, 1, 1)
        self.high_head = nn.Conv2d(d0, 1, 1)

    def forward(self, x_dem: torch.Tensor, x_ae: torch.Tensor, z_lr: torch.Tensor) -> dict[str, torch.Tensor]:
        xd0 = self.dem_b0(x_dem)
        xd1 = self.dem_b1(self.down_d1(xd0))
        xd2 = self.dem_b2(self.down_d2(xd1))
        xd3 = self.dem_b3(self.down_d3(xd2))
        xa0 = self.ae_b0(x_ae)
        xa1 = self.ae_b1(self.down_a1(xa0))
        xa2 = self.ae_b2(self.down_a2(xa1))
        xa3 = self.ae_b3(self.down_a3(xa2))
        a1, a2, a3 = self.film_alphas
        xd1 = self.film1(xd1, xa1, a1)
        xd2 = self.film2(xd2, xa2, a2)
        xd3 = self.film3(xd3, xa3, a3)
        x = self.dec3(xd3)
        x = self.up2(x)
        x = self.dec2(torch.cat([x, xd2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, xd1], dim=1))
        x = self.up0(x)
        feat = self.dec0(torch.cat([x, xd0], dim=1))

        r_low = self.low_head(feat)
        r_mid = self.mid_head(feat)
        r_high = self.high_head(feat)
        r_hat = reconstruct_residual_from_bands(r_low, r_mid, r_high)
        r_hat = self.r_cap * torch.tanh(r_hat / self.r_cap)
        z_hat = z_lr + r_hat
        return {"z_hat": z_hat, "r_low_hat": r_low, "r_mid_hat": r_mid, "r_high_hat": r_high, "r_hat": r_hat}


class FrequencyDomainExperiment(Experiment):
    name = "frequency_domain"

    @classmethod
    def add_train_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--lambda-band-low", type=float, default=0.5)
        parser.add_argument("--lambda-band-mid", type=float, default=0.75)
        parser.add_argument("--lambda-band-high", type=float, default=1.0)
        parser.add_argument("--lambda-band-high-tv", type=float, default=0.2)
        parser.add_argument("--lambda-band-high-l2", type=float, default=0.05)
        parser.add_argument("--lambda-band-balance", type=float, default=0.05)

    def build_model(self, cfg: dict[str, Any]) -> torch.nn.Module:
        arch = str(cfg.get("arch", ARCH_FILM))
        if arch != ARCH_FILM:
            raise ValueError("frequency_domain currently supports arch=film_unet only")
        return FrequencyDomainResidualModel(r_cap=float(cfg.get("r_cap", 20.0)))

    def model_forward(self, model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return model(batch["x_dem"], batch["x_ae"], batch["z_lr"])

    def build_loss(self, cfg: dict[str, Any]):
        composite = build_composite_loss_from_config(cfg)
        w_low = float(cfg.get("lambda_band_low", 0.5))
        w_mid = float(cfg.get("lambda_band_mid", 0.75))
        w_high = float(cfg.get("lambda_band_high", 1.0))
        w_high_tv = float(cfg.get("lambda_band_high_tv", 0.2))
        w_high_l2 = float(cfg.get("lambda_band_high_l2", 0.05))
        w_balance = float(cfg.get("lambda_band_balance", 0.05))

        def _wmean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return (x * w).sum() / (w.sum() + 1e-8)

        def _loss_fn(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> LossBundle:
            base = composite(outputs, batch)
            z_gt = batch["z_gt"]
            z_lr = batch["z_lr"]
            w = batch["w"]
            r_target = z_gt - z_lr
            r_low_t, r_mid_t, r_high_t = decompose_residual_laplacian(r_target)
            r_low_h = outputs["r_low_hat"]
            r_mid_h = outputs["r_mid_hat"]
            r_high_h = outputs["r_high_hat"]
            r_sum = reconstruct_residual_from_bands(r_low_h, r_mid_h, r_high_h)

            # Scale-normalized per-band errors prevent low/mid collapse.
            s_low = r_low_t.abs().mean().detach().clamp(min=1e-3)
            s_mid = r_mid_t.abs().mean().detach().clamp(min=1e-3)
            s_high = r_high_t.abs().mean().detach().clamp(min=1e-3)
            l_low = _wmean((r_low_h - r_low_t).abs() / s_low, w)
            l_mid = _wmean((r_mid_h - r_mid_t).abs() / s_mid, w)
            l_high = _wmean((r_high_h - r_high_t).abs() / s_high, w)

            # Stronger HF regularization to curb speckle/noise amplification.
            dx = (r_high_h[:, :, :, 1:] - r_high_h[:, :, :, :-1]).abs()
            dy = (r_high_h[:, :, 1:, :] - r_high_h[:, :, :-1, :]).abs()
            tv = dx.mean() + dy.mean()
            l2 = (r_high_h * r_high_h).mean()

            # Reconstruction correctness and band dominance monitoring.
            decomp_recon_err = _wmean((r_target - (r_low_t + r_mid_t + r_high_t)).abs(), w)
            pred_recon_err = _wmean((r_sum - outputs["r_hat"]).abs(), w)
            e_low = r_low_h.abs().mean()
            e_mid = r_mid_h.abs().mean()
            e_high = r_high_h.abs().mean()
            e_total = (e_low + e_mid + e_high).clamp(min=1e-8)
            p_low = e_low / e_total
            p_mid = e_mid / e_total
            p_high = e_high / e_total
            balance_penalty = (torch.max(torch.stack([p_low, p_mid, p_high])) - (1.0 / 3.0)).clamp(min=0.0)

            total = (
                base.loss
                + w_low * l_low
                + w_mid * l_mid
                + w_high * l_high
                + w_high_tv * tv
                + w_high_l2 * l2
                + w_balance * balance_penalty
            )
            metrics = dict(base.metrics)
            metrics.update(
                {
                    "band_low": float(l_low.detach()),
                    "band_mid": float(l_mid.detach()),
                    "band_high": float(l_high.detach()),
                    "band_high_tv": float(tv.detach()),
                    "band_high_l2": float(l2.detach()),
                    "band_share_low": float(p_low.detach()),
                    "band_share_mid": float(p_mid.detach()),
                    "band_share_high": float(p_high.detach()),
                    "band_balance_penalty": float(balance_penalty.detach()),
                    "decomp_recon_err": float(decomp_recon_err.detach()),
                    "pred_recon_err": float(pred_recon_err.detach()),
                    "total": float(total.detach()),
                }
            )
            return LossBundle(loss=total, metrics=metrics)

        return _loss_fn
