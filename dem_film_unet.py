"""DEM SR model variants and shared losses/utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_FILM = "film_unet"
ARCH_GATED = "gated_unet"
ARCH_CHOICES = (ARCH_FILM, ARCH_GATED)


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1, bias=False) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.silu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = y + self.skip(x)
        return F.silu(y)


def down_layer(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.SiLU(),
    )


class Up(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.silu(self.bn(self.conv(x)))
        return x


class GlobalFiLM(nn.Module):
    def __init__(self, c_ae: int, c_dem: int, hidden: int | None = None) -> None:
        super().__init__()
        h = hidden or max(c_dem, c_ae)
        self.mlp = nn.Sequential(
            nn.Linear(c_ae, h),
            nn.SiLU(),
            nn.Linear(h, 2 * c_dem),
        )

    def forward(self, f_dem: torch.Tensor, f_ae: torch.Tensor, alpha: float) -> torch.Tensor:
        # f_ae: (B, C_ae, H, W) -> GAP -> (B, C_ae)
        v = f_ae.mean(dim=(2, 3))
        h = self.mlp(v)
        gamma, beta = h.chunk(2, dim=1)
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + alpha * gamma) * f_dem + alpha * beta


class SpatialGatedFusion(nn.Module):
    """Fuse AE features into DEM features with per-pixel gating."""

    def __init__(self, c_ae: int, c_dem: int, trust_ch: int = 4) -> None:
        super().__init__()
        self.ae_to_dem = nn.Conv2d(c_ae, c_dem, 1, bias=False)
        self.gate = nn.Conv2d(c_dem + c_dem + trust_ch, c_dem, 1, bias=True)

    def forward(
        self,
        f_dem: torch.Tensor,
        f_ae: torch.Tensor,
        trust: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        a = self.ae_to_dem(f_ae)
        g = torch.sigmoid(self.gate(torch.cat([f_dem, a, trust], dim=1)))
        return f_dem + alpha * g * a


class DemFilmUNet(nn.Module):
    """Predicts residual R; output Z_hat = z_lr + clamped R."""

    dem_ch = (32, 64, 128, 256)
    ae_ch = (16, 32, 64, 128)
    film_alphas = (0.10, 0.15, 0.20)

    def __init__(self, r_cap: float = 20.0) -> None:
        super().__init__()
        self.r_cap = r_cap

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
        self.head = nn.Conv2d(d0, 1, 1)

    def forward(self, x_dem: torch.Tensor, x_ae: torch.Tensor, z_lr: torch.Tensor) -> torch.Tensor:
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
        x = self.dec0(torch.cat([x, xd0], dim=1))
        r = self.head(x)
        r = self.r_cap * torch.tanh(r / self.r_cap)
        return z_lr + r


class DemGatedFusionUNet(nn.Module):
    """Dual-encoder residual U-Net with spatial gated AE fusion."""

    dem_ch = (32, 64, 128, 256)
    ae_ch = (16, 32, 64, 128)
    gate_alphas = (0.10, 0.15, 0.20)

    def __init__(self, r_cap: float = 20.0) -> None:
        super().__init__()
        self.r_cap = r_cap

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

        self.gate1 = SpatialGatedFusion(a1, d1)
        self.gate2 = SpatialGatedFusion(a2, d2)
        self.gate3 = SpatialGatedFusion(a3, d3)

        self.dec3 = nn.Sequential(ResBlock(d3, d3), ResBlock(d3, d3))
        self.up2 = Up(d3, d2)
        self.dec2 = nn.Sequential(ResBlock(d2 + d2, d2), ResBlock(d2, d2))
        self.up1 = Up(d2, d1)
        self.dec1 = nn.Sequential(ResBlock(d1 + d1, d1), ResBlock(d1, d1))
        self.up0 = Up(d1, d0)
        self.dec0 = nn.Sequential(ResBlock(d0 + d0, d0), ResBlock(d0, d0))
        self.head = nn.Conv2d(d0, 1, 1)

    def _trust_pyramid(self, x_dem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_dem channels: [z_lr, u_enc, m_bld, m_wp, m_ws]
        trust = x_dem[:, 1:, :, :]
        t1 = F.avg_pool2d(trust, kernel_size=2, stride=2)
        t2 = F.avg_pool2d(t1, kernel_size=2, stride=2)
        t3 = F.avg_pool2d(t2, kernel_size=2, stride=2)
        return t1, t2, t3

    def forward(self, x_dem: torch.Tensor, x_ae: torch.Tensor, z_lr: torch.Tensor) -> torch.Tensor:
        xd0 = self.dem_b0(x_dem)
        xd1 = self.dem_b1(self.down_d1(xd0))
        xd2 = self.dem_b2(self.down_d2(xd1))
        xd3 = self.dem_b3(self.down_d3(xd2))

        xa0 = self.ae_b0(x_ae)
        xa1 = self.ae_b1(self.down_a1(xa0))
        xa2 = self.ae_b2(self.down_a2(xa1))
        xa3 = self.ae_b3(self.down_a3(xa2))

        t1, t2, t3 = self._trust_pyramid(x_dem)
        a1, a2, a3 = self.gate_alphas
        xd1 = self.gate1(xd1, xa1, t1, a1)
        xd2 = self.gate2(xd2, xa2, t2, a2)
        xd3 = self.gate3(xd3, xa3, t3, a3)

        x = self.dec3(xd3)
        x = self.up2(x)
        x = self.dec2(torch.cat([x, xd2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, xd1], dim=1))
        x = self.up0(x)
        x = self.dec0(torch.cat([x, xd0], dim=1))
        r = self.head(x)
        r = self.r_cap * torch.tanh(r / self.r_cap)
        return z_lr + r


def create_model(arch: str, *, r_cap: float = 20.0) -> nn.Module:
    """Factory for supported DEM SR architectures."""
    if arch == ARCH_FILM:
        return DemFilmUNet(r_cap=r_cap)
    if arch == ARCH_GATED:
        return DemGatedFusionUNet(r_cap=r_cap)
    raise ValueError(f"Unsupported arch={arch!r}; expected one of {ARCH_CHOICES}")


def terrain_slope(z: torch.Tensor, pixel_size_m: float = 10.0) -> torch.Tensor:
    """Slope magnitude as rise/run from centered differences on a 10 m grid."""
    # z: (B, 1, H, W)
    pad = (1, 1, 1, 1)
    zp = F.pad(z, pad, mode="replicate")
    dzdx = (zp[:, :, 1:-1, 2:] - zp[:, :, 1:-1, :-2]) / (2.0 * pixel_size_m)
    dzdy = (zp[:, :, 2:, 1:-1] - zp[:, :, :-2, 1:-1]) / (2.0 * pixel_size_m)
    return torch.sqrt(dzdx * dzdx + dzdy * dzdy + 1e-8)


def slope_to_degrees(slope_rise_run: torch.Tensor) -> torch.Tensor:
    """Convert rise/run slope magnitude to degrees."""
    return torch.atan(slope_rise_run) * (180.0 / torch.pi)


def loss_dem(
    z_hat: torch.Tensor,
    z_gt: torch.Tensor,
    w: torch.Tensor,
    lambda_slope: float = 0.5,
    smooth_l1_beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted SmoothL1 + lambda * weighted |slope_hat - slope_gt|."""
    elev = F.smooth_l1_loss(z_hat, z_gt, beta=smooth_l1_beta, reduction="none")
    l_elev = (elev * w).sum() / (w.sum() + 1e-8)

    s_hat = terrain_slope(z_hat)
    s_gt = terrain_slope(z_gt)
    l_slope = ((s_hat - s_gt).abs() * w).sum() / (w.sum() + 1e-8)

    return l_elev + lambda_slope * l_slope, l_elev.detach(), l_slope.detach()
