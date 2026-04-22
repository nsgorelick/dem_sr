"""Shared-backbone mixture-of-specialists experiment."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dem_film_unet import ARCH_FILM, GlobalFiLM, ResBlock, Up, down_layer
from experiments.base import Experiment, LossBundle
from losses.composite import build_composite_loss_from_config


class SharedBackboneMixtureUNet(nn.Module):
    """FiLM U-Net backbone with soft-routed specialist residual heads."""

    dem_ch = (32, 64, 128, 256)
    ae_ch = (16, 32, 64, 128)
    film_alphas = (0.10, 0.15, 0.20)

    def __init__(self, *, num_experts: int = 3, router_temperature: float = 1.0, r_cap: float = 20.0) -> None:
        super().__init__()
        if num_experts < 2 or num_experts > 3:
            raise ValueError("num_experts must be 2 or 3 for initial MoS path")
        self.num_experts = int(num_experts)
        self.router_temperature = float(router_temperature)
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

        self.specialist_heads = nn.ModuleList([nn.Conv2d(d0, 1, 1) for _ in range(self.num_experts)])
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(d0, self.num_experts))

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

        logits = self.router(feat) / max(self.router_temperature, 1e-6)
        weights = torch.softmax(logits, dim=1)  # B,E
        expert_residuals = torch.stack([head(feat) for head in self.specialist_heads], dim=1)  # B,E,1,H,W
        mix = (expert_residuals * weights[:, :, None, None, None]).sum(dim=1)
        r = self.r_cap * torch.tanh(mix / self.r_cap)
        z_hat = z_lr + r
        return {
            "z_hat": z_hat,
            "specialist_weights": weights,
            "expert_residuals": expert_residuals.squeeze(2),
        }


class MixtureSpecialistsExperiment(Experiment):
    name = "mixture_specialists"

    def build_model(self, cfg: dict[str, Any]) -> torch.nn.Module:
        arch = str(cfg.get("arch", ARCH_FILM))
        if arch != ARCH_FILM:
            raise ValueError("mixture_specialists currently supports arch=film_unet only")
        return SharedBackboneMixtureUNet(
            num_experts=int(cfg.get("mos_num_experts", 3)),
            router_temperature=float(cfg.get("mos_router_temperature", 1.0)),
            r_cap=float(cfg.get("r_cap", 20.0)),
        )

    def model_forward(self, model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return model(batch["x_dem"], batch["x_ae"], batch["z_lr"])

    def build_loss(self, cfg: dict[str, Any]):
        composite = build_composite_loss_from_config(cfg)

        def _loss_fn(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> LossBundle:
            bundle = composite(outputs, batch)
            metrics = dict(bundle.metrics)
            weights = outputs.get("specialist_weights")
            expert_residuals = outputs.get("expert_residuals")
            if isinstance(weights, torch.Tensor):
                mean_w = weights.mean(dim=0)
                entropy = -(weights * torch.log(weights.clamp(min=1e-8))).sum(dim=1).mean()
                for idx in range(mean_w.shape[0]):
                    metrics[f"expert_util_{idx}"] = float(mean_w[idx].detach())
                metrics["expert_entropy"] = float(entropy.detach())
            if isinstance(expert_residuals, torch.Tensor) and expert_residuals.shape[1] > 1:
                std_per_pixel = expert_residuals.std(dim=1)
                metrics["expert_divergence"] = float(std_per_pixel.mean().detach())
            return LossBundle(loss=bundle.loss, metrics=metrics)

        return _loss_fn

