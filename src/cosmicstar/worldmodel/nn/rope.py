"""1D temporal (game-loop) and 2D spatial (map-coordinate) rotary position embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn

ROPE_SHORTEST = 1.0 / 32.0
TIME_SHORTEST, TIME_LONGEST = 3.0, 672.0


def cos_sin(ang: torch.Tensor) -> torch.Tensor:
    """Convert rotation angles `[..., dh/2]` to `[..., dh/2, 2]` `(cos, sin)` in compute dtype."""
    dt = (
        torch.get_autocast_dtype("cuda")
        if torch.is_autocast_enabled("cuda")
        else ang.dtype
    )
    return torch.stack([ang.cos(), ang.sin()], -1).to(dt)


def rotate(x: torch.Tensor, cs: torch.Tensor, back: bool = False) -> torch.Tensor:
    """Rotate `x` `[B, H, T, dh]` by `cs` `[B, T, dh/2, 2]` (`back=True` applies inverse)."""
    c, s = cs[..., 0][:, None].to(x.dtype), cs[..., 1][:, None].to(x.dtype)
    s = -s if back else s
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class RoPE1D(nn.Module):
    """Rotary positions over game loops across geometric wavelengths `[lo, hi]`."""

    def __init__(self, dh: int, lo: float = TIME_SHORTEST, hi: float = TIME_LONGEST):
        super().__init__()
        self.register_buffer(
            "omega",
            2 * math.pi / torch.logspace(math.log10(lo), math.log10(hi), dh // 2),
            persistent=False,
        )

    def angles(self, loops: torch.Tensor) -> torch.Tensor:
        """`loops` `[..., F]` -> `[..., F, dh/2]` rotation angles."""
        return loops[..., None].float() * self.omega


class RoPE2D(nn.Module):
    """2D rotary positions over the map plane (`x` and `y` each rotate half of head pairs)."""

    def __init__(self, dh: int, lo: float = 2.0, hi: float = 256.0):
        super().__init__()
        assert dh % 4 == 0, "head dim must be a multiple of 4 for 2D RoPE"
        n = dh // 4
        self.register_buffer(
            "omega",
            2 * math.pi / torch.logspace(math.log10(lo), math.log10(hi), n),
            persistent=False,
        )

    def angles(self, pos: torch.Tensor) -> torch.Tensor:
        """`pos` `[B, T, 2]` in map units -> `[B, T, dh/2]` rotation angles."""
        return torch.cat([pos[..., 0:1] * self.omega, pos[..., 1:2] * self.omega], -1)

    @staticmethod
    def rotate(x: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
        """Rotate `x` `[B, H, T, dh]` directly from angles `ang` `[B, T, dh/2]`."""
        c, s = ang.cos()[:, None].to(x.dtype), ang.sin()[:, None].to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


def apply_rope_2d(
    vec: torch.Tensor, pos_map: torch.Tensor, rope_2d: RoPE2D
) -> torch.Tensor:
    """Apply 2D axial RoPE rotation to `[..., N, D]` using map-unit positions `[..., N, 2]`."""
    orig_shape = vec.shape
    N, D = orig_shape[-2], orig_shape[-1]
    dh = rope_2d.omega.numel() * 4
    heads = D // dh
    v_flat = vec.reshape(-1, N, heads, dh).transpose(1, 2)
    cs = cos_sin(rope_2d.angles(pos_map.reshape(-1, N, 2))).to(vec.dtype)
    v_rot = rotate(v_flat, cs)
    return v_rot.transpose(1, 2).reshape(orig_shape)
