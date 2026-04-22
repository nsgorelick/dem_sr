"""DEM SR model variants and shared losses/utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_FILM = "film_unet"
ARCH_GATED = "gated_unet"
ARCH_XATTN = "xattn_unet"
ARCH_HYBRID_TF = "hybrid_tf_unet"
ARCH_RCAN_AE = "rcan_ae_unet"
ARCH_CHOICES = (ARCH_FILM, ARCH_GATED, ARCH_XATTN, ARCH_HYBRID_TF, ARCH_RCAN_AE)

LOSS_PRESET_BASELINE = "baseline"
LOSS_PRESET_GEOM = "geom"
LOSS_PRESET_MULTITASK = "multitask"
LOSS_PRESET_CONTOUR = "contour"
LOSS_PRESET_CHOICES = (
    LOSS_PRESET_BASELINE,
    LOSS_PRESET_GEOM,
    LOSS_PRESET_MULTITASK,
    LOSS_PRESET_CONTOUR,
)

DEFAULT_CONTOUR_INTERVAL_M = 10.0


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


def _pad_hw_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    """Pad BCHW tensor so H,W are multiples of ``multiple``. Returns (padded, pad_h, pad_w)."""
    _, _, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, pad_h, pad_w


def _crop_hw_pad(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_h == 0 and pad_w == 0:
        return x
    _, _, H, W = x.shape
    return x[:, :, : H - pad_h, : W - pad_w]


def window_partition_bchw(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, int, int]:
    """Split BCHW into non-overlapping windows. Returns (windows, nw_h, nw_w).

    ``windows`` has shape ``(B * nw_h * nw_w, window_size * window_size, C)``.
    """
    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0, (H, W, window_size)
    nw_h, nw_w = H // window_size, W // window_size
    x = x.view(B, C, nw_h, window_size, nw_w, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(B * nw_h * nw_w, window_size * window_size, C)
    return x, nw_h, nw_w


def window_reverse_bchw(
    windows: torch.Tensor,
    window_size: int,
    nw_h: int,
    nw_w: int,
    B: int,
    C: int,
) -> torch.Tensor:
    """Inverse of ``window_partition_bchw``."""
    x = windows.view(B, nw_h, nw_w, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, nw_h * window_size, nw_w * window_size)
    return x


class WindowCrossAttnFusion(nn.Module):
    """Windowed cross-attention: DEM queries, AE keys/values; then gated residual (dd.md Plan B2)."""

    def __init__(
        self,
        c_ae: int,
        c_dem: int,
        trust_ch: int = 4,
        *,
        window_size: int = 8,
        d_model: int = 64,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.window_size = window_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.proj_q = nn.Conv2d(c_dem, d_model, 1, bias=False)
        self.proj_k = nn.Conv2d(c_ae, d_model, 1, bias=False)
        self.proj_v = nn.Conv2d(c_ae, d_model, 1, bias=False)
        self.proj_out = nn.Conv2d(d_model, c_dem, 1, bias=False)
        self.gate = nn.Conv2d(c_dem + c_dem + trust_ch, c_dem, 1, bias=True)

    def forward(
        self,
        f_dem: torch.Tensor,
        f_ae: torch.Tensor,
        trust: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        f_dem_in = f_dem
        ws = self.window_size
        f_dem_p, ph, pw = _pad_hw_to_multiple(f_dem_in, ws)
        f_ae_p, _, _ = _pad_hw_to_multiple(f_ae, ws)

        q = self.proj_q(f_dem_p)
        k = self.proj_k(f_ae_p)
        v = self.proj_v(f_ae_p)

        B, _, _, _ = q.shape
        q_w, nw_h, nw_w = window_partition_bchw(q, ws)
        k_w, _, _ = window_partition_bchw(k, ws)
        v_w, _, _ = window_partition_bchw(v, ws)

        nw = nw_h * nw_w
        L = ws * ws
        # (B*nW, L, d) -> (B*nW, nh, L, hd)
        q_w = q_w.view(B * nw, L, self.num_heads, self.head_dim).transpose(1, 2)
        k_w = k_w.view(B * nw, L, self.num_heads, self.head_dim).transpose(1, 2)
        v_w = v_w.view(B * nw, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q_w, k_w, v_w, dropout_p=0.0)
        attn = attn.transpose(1, 2).contiguous().view(B * nw, L, self.d_model)

        merged = window_reverse_bchw(attn, ws, nw_h, nw_w, B, self.d_model)
        merged = _crop_hw_pad(merged, ph, pw)
        a = self.proj_out(merged)

        g = torch.sigmoid(self.gate(torch.cat([f_dem_in, a, trust], dim=1)))
        return f_dem_in + alpha * g * a


def _group_norm(channels: int) -> nn.GroupNorm:
    """Pick a GroupNorm group count that divides ``channels``."""
    for g in (32, 16, 8, 4, 2, 1):
        if channels >= g and channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


class WindowSelfAttn2d(nn.Module):
    """Windowed multi-head self-attention on a single feature map (bottleneck / DSRT-style context)."""

    def __init__(
        self,
        c_in: int,
        *,
        window_size: int = 8,
        d_model: int = 128,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.window_size = window_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.proj_q = nn.Conv2d(c_in, d_model, 1, bias=False)
        self.proj_k = nn.Conv2d(c_in, d_model, 1, bias=False)
        self.proj_v = nn.Conv2d(c_in, d_model, 1, bias=False)
        self.proj_out = nn.Conv2d(d_model, c_in, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        ws = self.window_size
        x_p, ph, pw = _pad_hw_to_multiple(x_in, ws)
        q = self.proj_q(x_p)
        k = self.proj_k(x_p)
        v = self.proj_v(x_p)

        B, _, _, _ = q.shape
        q_w, nw_h, nw_w = window_partition_bchw(q, ws)
        k_w, _, _ = window_partition_bchw(k, ws)
        v_w, _, _ = window_partition_bchw(v, ws)

        nw = nw_h * nw_w
        L = ws * ws
        q_w = q_w.view(B * nw, L, self.num_heads, self.head_dim).transpose(1, 2)
        k_w = k_w.view(B * nw, L, self.num_heads, self.head_dim).transpose(1, 2)
        v_w = v_w.view(B * nw, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q_w, k_w, v_w, dropout_p=0.0)
        attn = attn.transpose(1, 2).contiguous().view(B * nw, L, self.d_model)

        merged = window_reverse_bchw(attn, ws, nw_h, nw_w, B, self.d_model)
        merged = _crop_hw_pad(merged, ph, pw)
        return self.proj_out(merged)


class BottleneckTransformerBlock(nn.Module):
    """Pre-norm + window self-attention + conv FFN (two residual streams)."""

    def __init__(
        self,
        channels: int,
        *,
        window_size: int = 8,
        d_model: int = 128,
        num_heads: int = 8,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.norm1 = _group_norm(channels)
        self.attn = WindowSelfAttn2d(
            channels,
            window_size=window_size,
            d_model=d_model,
            num_heads=num_heads,
        )
        self.norm2 = _group_norm(channels)
        hidden = channels * ffn_mult
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class RCAB(nn.Module):
    """Residual channel-attention block (RCAN-style)."""

    def __init__(self, channels: int, residual_scale: float = 0.1) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = _group_norm(channels)
        self.act = nn.SiLU()
        self.ca = ChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        y = self.ca(y)
        return x + self.residual_scale * y


class ResidualGroup(nn.Module):
    """Stack of RCAB blocks with group-level residual."""

    def __init__(self, channels: int, num_blocks: int, residual_scale: float = 0.1) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.blocks = nn.Sequential(*(RCAB(channels, residual_scale=residual_scale) for _ in range(num_blocks)))
        self.tail = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(x)
        y = self.tail(y)
        return x + self.residual_scale * y


class AEFusionGate(nn.Module):
    """Channel-aware AE conditioning for RCAN trunk."""

    def __init__(self, feat_ch: int, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.ae_proj = nn.Conv2d(64, feat_ch, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(feat_ch, feat_ch, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, f_dem: torch.Tensor, x_ae: torch.Tensor) -> torch.Tensor:
        a = self.ae_proj(x_ae)
        g = self.gate(torch.cat([f_dem, a], dim=1))
        return f_dem + self.alpha * g * a


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


class DemCrossAttnFusionUNet(nn.Module):
    """Dual-encoder U-Net: FiLM at S1; windowed DEM→AE cross-attention at S2 and S3."""

    dem_ch = (32, 64, 128, 256)
    ae_ch = (16, 32, 64, 128)
    film_alpha_s1 = 0.10
    xattn_alphas = (0.15, 0.20)
    xattn_window = 8

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
        self.xattn2 = WindowCrossAttnFusion(a2, d2, trust_ch=4, window_size=self.xattn_window)
        self.xattn3 = WindowCrossAttnFusion(a3, d3, trust_ch=4, window_size=self.xattn_window)

        self.dec3 = nn.Sequential(ResBlock(d3, d3), ResBlock(d3, d3))
        self.up2 = Up(d3, d2)
        self.dec2 = nn.Sequential(ResBlock(d2 + d2, d2), ResBlock(d2, d2))
        self.up1 = Up(d2, d1)
        self.dec1 = nn.Sequential(ResBlock(d1 + d1, d1), ResBlock(d1, d1))
        self.up0 = Up(d1, d0)
        self.dec0 = nn.Sequential(ResBlock(d0 + d0, d0), ResBlock(d0, d0))
        self.head = nn.Conv2d(d0, 1, 1)

    def _trust_pyramid(self, x_dem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        _, t2, t3 = self._trust_pyramid(x_dem)
        ax2, ax3 = self.xattn_alphas
        xd1 = self.film1(xd1, xa1, self.film_alpha_s1)
        xd2 = self.xattn2(xd2, xa2, t2, ax2)
        xd3 = self.xattn3(xd3, xa3, t3, ax3)

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


class DemHybridTransformerUNet(nn.Module):
    """FiLM fusion like baseline; two lightweight transformer bottleneck blocks at S3 before decode."""

    dem_ch = (32, 64, 128, 256)
    ae_ch = (16, 32, 64, 128)
    film_alphas = (0.10, 0.15, 0.20)
    tf_window = 8
    tf_d_model = 128
    tf_heads = 8

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

        self.bottleneck_tf = nn.Sequential(
            BottleneckTransformerBlock(
                d3,
                window_size=self.tf_window,
                d_model=self.tf_d_model,
                num_heads=self.tf_heads,
            ),
            BottleneckTransformerBlock(
                d3,
                window_size=self.tf_window,
                d_model=self.tf_d_model,
                num_heads=self.tf_heads,
            ),
        )

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
        xd3 = self.bottleneck_tf(xd3)

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


class DemRCANAE(nn.Module):
    """RCAN-style DEM SR backbone with AE-conditioned trunk."""

    feat_ch = 48
    num_groups = 4
    blocks_per_group = 3
    group_residual_scale = 0.1
    ae_alpha = 0.1

    def __init__(self, r_cap: float = 20.0) -> None:
        super().__init__()
        self.r_cap = r_cap
        c = self.feat_ch

        self.dem_head = nn.Conv2d(5, c, 3, padding=1, bias=False)
        self.ae_fusion = AEFusionGate(c, alpha=self.ae_alpha)
        self.pre_ca = ChannelAttention(c)
        self.groups = nn.Sequential(
            *(
                ResidualGroup(c, self.blocks_per_group, residual_scale=self.group_residual_scale)
                for _ in range(self.num_groups)
            )
        )
        self.trunk_tail = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.res_head = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(c, 1, 1),
        )

    def forward(self, x_dem: torch.Tensor, x_ae: torch.Tensor, z_lr: torch.Tensor) -> torch.Tensor:
        x = self.dem_head(x_dem)
        x = self.ae_fusion(x, x_ae)
        x = self.pre_ca(x)
        trunk = self.groups(x)
        x = x + self.trunk_tail(trunk)
        r = self.res_head(x)
        r = self.r_cap * torch.tanh(r / self.r_cap)
        return z_lr + r


def create_model(arch: str, *, r_cap: float = 20.0) -> nn.Module:
    """Factory for supported DEM SR architectures."""
    if arch == ARCH_FILM:
        return DemFilmUNet(r_cap=r_cap)
    if arch == ARCH_GATED:
        return DemGatedFusionUNet(r_cap=r_cap)
    if arch == ARCH_XATTN:
        return DemCrossAttnFusionUNet(r_cap=r_cap)
    if arch == ARCH_HYBRID_TF:
        return DemHybridTransformerUNet(r_cap=r_cap)
    if arch == ARCH_RCAN_AE:
        return DemRCANAE(r_cap=r_cap)
    raise ValueError(f"Unsupported arch={arch!r}; expected one of {ARCH_CHOICES}")


def terrain_grad(z: torch.Tensor, pixel_size_m: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Centered-difference gradient components ``(dz/dx, dz/dy)`` in rise/run."""
    zp = F.pad(z, (1, 1, 1, 1), mode="replicate")
    dzdx = (zp[:, :, 1:-1, 2:] - zp[:, :, 1:-1, :-2]) / (2.0 * pixel_size_m)
    dzdy = (zp[:, :, 2:, 1:-1] - zp[:, :, :-2, 1:-1]) / (2.0 * pixel_size_m)
    return dzdx, dzdy


def terrain_slope(z: torch.Tensor, pixel_size_m: float = 10.0) -> torch.Tensor:
    """Slope magnitude as rise/run from centered differences on a 10 m grid."""
    dzdx, dzdy = terrain_grad(z, pixel_size_m=pixel_size_m)
    return torch.sqrt(dzdx * dzdx + dzdy * dzdy + 1e-8)


def slope_to_degrees(slope_rise_run: torch.Tensor) -> torch.Tensor:
    """Convert rise/run slope magnitude to degrees."""
    return torch.atan(slope_rise_run) * (180.0 / torch.pi)


def terrain_laplacian(z: torch.Tensor, pixel_size_m: float = 10.0) -> torch.Tensor:
    """Discrete 5-point Laplacian of ``z``; units ``m / m^2`` (curvature proxy)."""
    zp = F.pad(z, (1, 1, 1, 1), mode="replicate")
    lap = (
        zp[:, :, 1:-1, 2:]
        + zp[:, :, 1:-1, :-2]
        + zp[:, :, 2:, 1:-1]
        + zp[:, :, :-2, 1:-1]
        - 4.0 * z
    ) / (pixel_size_m * pixel_size_m)
    return lap


def contour_sdf(z: torch.Tensor, interval: float = DEFAULT_CONTOUR_INTERVAL_M) -> torch.Tensor:
    """Signed vertical offset to the nearest contour at elevation ``k * interval``.

    Returned values lie in ``[-interval/2, interval/2]``: positive when ``z`` sits
    above the nearest contour, negative when below. Piecewise-linear (derivative
    1 almost everywhere) so it is differentiable w.r.t. ``z``.
    """
    half = 0.5 * interval
    return torch.remainder(z + half, interval) - half


def contour_soft(z: torch.Tensor, interval: float = DEFAULT_CONTOUR_INTERVAL_M) -> torch.Tensor:
    """Smooth contour indicator in ``[0, 1]`` peaking at ``z = k * interval``.

    Uses ``cos^2(pi * z / interval)`` so the peak is at each contour level and
    the derivative is everywhere defined, letting L1 on ``contour_soft(z_hat)``
    vs ``contour_soft(z_gt)`` penalize misaligned contour locations without the
    non-differentiability of a hard binary mask.
    """
    return 0.5 * (1.0 + torch.cos(2.0 * torch.pi * z / interval))


def contour_binary(z: torch.Tensor, interval: float = DEFAULT_CONTOUR_INTERVAL_M) -> torch.Tensor:
    """Hard 0/1 contour-crossing map. Non-differentiable; for reporting only.

    A pixel is marked 1 when ``floor(z / interval)`` differs from any of its
    four neighbors, i.e., a contour line passes through or adjacent to it.
    """
    lvl = torch.floor(z / interval)
    lp = F.pad(lvl, (1, 1, 1, 1), mode="replicate")
    neigh_diff = (
        (lp[:, :, 1:-1, 2:] != lvl).float()
        + (lp[:, :, 1:-1, :-2] != lvl).float()
        + (lp[:, :, 2:, 1:-1] != lvl).float()
        + (lp[:, :, :-2, 1:-1] != lvl).float()
    )
    return (neigh_diff > 0).float()


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


def loss_dem_preset(
    z_hat: torch.Tensor,
    z_gt: torch.Tensor,
    w: torch.Tensor,
    preset: str = LOSS_PRESET_BASELINE,
    *,
    lambda_slope: float = 0.5,
    lambda_grad: float = 0.25,
    lambda_curv: float = 0.1,
    lambda_ms: float = 0.5,
    lambda_sdf: float = 0.5,
    lambda_contour: float = 0.25,
    contour_interval: float = DEFAULT_CONTOUR_INTERVAL_M,
    smooth_l1_beta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Multi-component DEM loss keyed by ``preset``.

    Presets:
      - ``baseline``:  elev SmoothL1 + lambda_slope * |slope diff|.
      - ``geom``:      baseline + grad dx/dy L1 + Laplacian L1 + multi-scale (2x) elev L1.
      - ``multitask``: ``geom`` + contour SDF L1 + soft contour-indicator L1.
      - ``contour``:   baseline + contour SDF L1 only.

    Returns ``(total, components)`` where ``components`` maps each term name
    (``elev``, ``slope``, ``grad``, ``curv``, ``ms_elev``, ``sdf``, ``contour``,
    ``total``) to its detached scalar value. Missing-for-preset terms are omitted.
    """
    if preset not in LOSS_PRESET_CHOICES:
        raise ValueError(f"Unknown preset={preset!r}; expected one of {LOSS_PRESET_CHOICES}")

    w_sum = w.sum() + 1e-8

    def _wmean(x: torch.Tensor) -> torch.Tensor:
        return (x * w).sum() / w_sum

    elev = F.smooth_l1_loss(z_hat, z_gt, beta=smooth_l1_beta, reduction="none")
    l_elev = _wmean(elev)

    s_hat = terrain_slope(z_hat)
    s_gt = terrain_slope(z_gt)
    l_slope = _wmean((s_hat - s_gt).abs())

    total = l_elev + lambda_slope * l_slope
    components: dict[str, torch.Tensor] = {
        "elev": l_elev.detach(),
        "slope": l_slope.detach(),
    }

    if preset in (LOSS_PRESET_GEOM, LOSS_PRESET_MULTITASK):
        gx_h, gy_h = terrain_grad(z_hat)
        gx_g, gy_g = terrain_grad(z_gt)
        l_grad = _wmean((gx_h - gx_g).abs() + (gy_h - gy_g).abs())

        lap_h = terrain_laplacian(z_hat)
        lap_g = terrain_laplacian(z_gt)
        l_curv = _wmean((lap_h - lap_g).abs())

        z_hat_ds = F.avg_pool2d(z_hat, kernel_size=2, stride=2)
        z_gt_ds = F.avg_pool2d(z_gt, kernel_size=2, stride=2)
        w_ds = F.avg_pool2d(w, kernel_size=2, stride=2)
        elev_ds = F.smooth_l1_loss(z_hat_ds, z_gt_ds, beta=smooth_l1_beta, reduction="none")
        l_ms = (elev_ds * w_ds).sum() / (w_ds.sum() + 1e-8)

        total = total + lambda_grad * l_grad + lambda_curv * l_curv + lambda_ms * l_ms
        components["grad"] = l_grad.detach()
        components["curv"] = l_curv.detach()
        components["ms_elev"] = l_ms.detach()

    if preset in (LOSS_PRESET_MULTITASK, LOSS_PRESET_CONTOUR):
        sdf_h = contour_sdf(z_hat, contour_interval)
        sdf_g = contour_sdf(z_gt, contour_interval)
        l_sdf = _wmean((sdf_h - sdf_g).abs())
        total = total + lambda_sdf * l_sdf
        components["sdf"] = l_sdf.detach()

    if preset == LOSS_PRESET_MULTITASK:
        c_h = contour_soft(z_hat, contour_interval)
        c_g = contour_soft(z_gt, contour_interval)
        l_contour = _wmean((c_h - c_g).abs())
        total = total + lambda_contour * l_contour
        components["contour"] = l_contour.detach()

    components["total"] = total.detach()
    return total, components
