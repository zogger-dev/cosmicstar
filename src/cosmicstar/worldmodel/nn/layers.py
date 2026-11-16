"""Shared feed-forward, convolutional, Fourier, and implicit-field layers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def mlp(d_in: int, d: int, hidden: int | None = None) -> nn.Sequential:
    """Two-layer GELU MLP (`Linear -> GELU -> Linear`)."""
    hidden = hidden or d
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d))


def norm_silu_mlp(
    d_in: int, d_out: int, hidden: int, bias: bool = True
) -> nn.Sequential:
    """Three-layer normalized SiLU MLP (`Linear -> LayerNorm -> SiLU -> Linear -> SiLU -> Linear`)."""
    return nn.Sequential(
        nn.Linear(d_in, hidden),
        nn.LayerNorm(hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, d_out, bias=bias),
    )


def small_init(m: nn.Linear, std: float = 1e-3) -> nn.Linear:
    """Initialize a `Linear` layer with small Gaussian weights (`std=1e-3`) and zero bias."""
    nn.init.normal_(m.weight, mean=0.0, std=std)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class FeedForward(nn.Module):
    """Pre-norm SwiGLU block with fused gate+up GEMM and residual skip."""

    def __init__(self, d: int, ff: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(d, ff, bias=False)
        self.w3 = nn.Linear(ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        u1, u2 = F.linear(y, torch.cat([self.w1.weight, self.w2.weight], 0)).chunk(
            2, -1
        )
        return x + self.w3(F.silu(u1) * u2)


class CellNorm(nn.LayerNorm):
    """Per-cell channel `LayerNorm` on `[B, C, H, W]` tensors (padding-independent)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ResBlock(nn.Module):
    """Masked `3x3` residual block on `[B, C, H, W]` zeroing off-map padding `m`."""

    def __init__(self, c: int):
        super().__init__()
        self.c1, self.n1 = nn.Conv2d(c, c, 3, padding=1, bias=False), CellNorm(c)
        self.c2, self.n2 = nn.Conv2d(c, c, 3, padding=1, bias=False), CellNorm(c)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.n1(self.c1(x))) * m
        return F.gelu(x + self.n2(self.c2(h))) * m


class MapFourier(nn.Module):
    """Sin/cos features of map-unit coordinates at octave wavelengths `[shortest, longest]`."""

    def __init__(self, shortest: float = 0.125, longest: float = 512.0):
        super().__init__()
        n = int(round(math.log2(longest / shortest))) + 1
        self.register_buffer(
            "freqs",
            2 * math.pi / (shortest * 2.0 ** torch.arange(n)),
            persistent=False,
        )

    def out_dim(self, d_in: int) -> int:
        return d_in * 2 * len(self.freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x.float()[..., None] * self.freqs
        return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class MapDecoder(nn.Module):
    """Implicit local field decoding `1x1` map-unit cells from latent grid cells + Fourier offsets."""

    def __init__(self, d: int, n_out: int, hidden: int = 256, players: int = 0):
        super().__init__()
        self.player = nn.Embedding(players, d) if players else None
        self.fourier = MapFourier(0.25, 32.0)
        self.mlp = nn.Sequential(
            nn.Linear(d + self.fourier.out_dim(2), hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_out),
        )

    def forward(
        self,
        cells: torch.Tensor,
        cell: torch.Tensor,
        off: torch.Tensor,
        player: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.gather(cells, 1, cell[..., None].expand(-1, -1, cells.shape[-1]))
        if self.player is not None:
            x = x + self.player(player)[:, None].to(x.dtype)
        return self.mlp(torch.cat([x, self.fourier(off).to(x.dtype)], -1)).float()
