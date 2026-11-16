"""Conditional flow-matching priors over player tactical intents and world outcome shocks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..l1_tactical import squad_pooler as SP
from ..nn.layers import small_init


class _TensorCoreResBlock(nn.Module):
    """High-arithmetic-intensity bf16 Tensor Core residual MLP block."""

    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        small_init(self.fc2, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class ConditionalFlowPrior(nn.Module):
    """Dual-Mode Classifier-Free `Delta A`-Conditioned Strategy Flow Prior
    `v_theta(u_s, s | h_{<= tau}^{(p)}, c_{p, tau}, ret_cond)`.

    Trained with `50%` null-return conditioning dropout so:
      - `mode="forecast"` uses unconditional (`null_ret_emb`) realistic fog-bound sampling,
      - `mode="counterplay"` uses high-`Delta A*` Classifier-Free Guidance on the learned manifold.
    """

    def __init__(
        self,
        d_u: int = SP.innovation_dim(),
        d_model: int = 768,
        d_cap: int = 16,
        hidden: int = 2048,
    ):
        super().__init__()
        self.d_u = d_u
        freqs = 2.0 * math.pi * torch.arange(1, 17, dtype=torch.float32)
        self.register_buffer("s_freqs", freqs, persistent=False)

        self.ret_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )
        self.null_ret_emb = nn.Parameter(torch.zeros(32))

        in_dim = d_u + d_model + d_cap + 32 + 32
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden, bias=False),
            nn.SiLU(),
            _TensorCoreResBlock(hidden),
            _TensorCoreResBlock(hidden),
            _TensorCoreResBlock(hidden),
            nn.Linear(hidden, d_u),
        )
        small_init(self.net[-1], std=2e-2)

        self.tempo_pred = nn.Sequential(
            nn.LayerNorm(d_model + d_cap + d_u),
            nn.Linear(d_model + d_cap + d_u, 256),
            nn.SiLU(),
            nn.Linear(256, SP.D_TEMPO_FEAT),
        )

    def _embed_s(self, s: torch.Tensor) -> torch.Tensor:
        ang = s.unsqueeze(-1) * self.s_freqs
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def _embed_ret(
        self,
        shape_prefix: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        ret_val: torch.Tensor | None = None,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        null_e = self.null_ret_emb.to(dtype).view(*([1] * len(shape_prefix)), 32).expand(*shape_prefix, 32)
        if ret_val is None:
            return null_e
        rv = ret_val.to(dtype)
        r_in = torch.stack([rv, rv.abs()], dim=-1)
        r_e = self.ret_mlp(r_in)
        if cond_mask is None:
            return r_e
        m = cond_mask.to(dtype).unsqueeze(-1)
        return m * r_e + (1.0 - m) * null_e

    def velocity(
        self,
        u_s: torch.Tensor,
        s: torch.Tensor,
        h_p: torch.Tensor,
        c_p: torch.Tensor,
        ret_val: torch.Tensor | None = None,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s_emb = self._embed_s(s).to(h_p.dtype)
        r_emb = self._embed_ret(h_p.shape[:-1], h_p.device, h_p.dtype, ret_val=ret_val, cond_mask=cond_mask)
        return self.net(torch.cat([u_s.to(h_p.dtype), h_p, c_p.to(h_p.dtype), s_emb, r_emb], dim=-1))

    def flow_matching_loss(
        self,
        u_target: torch.Tensor,
        h_p: torch.Tensor,
        c_p: torch.Tensor,
        delta_adv_p: torch.Tensor | None = None,
        cfg_drop_prob: float = 0.50,
    ) -> torch.Tensor:
        """Conditional Flow-Matching MSE loss with `50%` Classifier-Free `Delta A` dropout."""
        u_0 = u_target.detach()
        eps = torch.randn_like(u_0)
        s = torch.rand(u_0.shape[:-1], device=u_0.device, dtype=u_0.dtype)
        u_s = (1.0 - s.unsqueeze(-1)) * u_0 + s.unsqueeze(-1) * eps

        if delta_adv_p is not None:
            ret_val = torch.tanh(delta_adv_p.detach() / 2.5)
            cond_mask = torch.rand(u_0.shape[:-1], device=u_0.device) >= cfg_drop_prob
        else:
            ret_val, cond_mask = None, None

        v_pred = self.velocity(u_s, s, h_p, c_p, ret_val=ret_val, cond_mask=cond_mask)
        return F.mse_loss(v_pred.float(), (u_0 - eps).float())

    def sample(
        self,
        h_p: torch.Tensor,
        c_p: torch.Tensor,
        steps: int = 4,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        counterplay_player: int | None = None,
        cfg_scale: float = 1.25,
        target_ret: float = 0.28,
        init_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Midpoint ODE integration from `s = 1` noise to `s = 0` clean intent, supporting
        Common Random Numbers (`generator` / `init_noise`) and Classifier-Free Counterplay Guidance."""
        if init_noise is not None:
            u = init_noise.to(h_p.dtype) * temperature
        else:
            u = (
                torch.randn(
                    h_p.shape[:-1] + (self.d_u,),
                    generator=generator,
                    device=h_p.device,
                    dtype=torch.float32,
                ).to(h_p.dtype)
                * temperature
            )
        dt = 1.0 / max(1, steps)

        def _eval_v(u_cur: torch.Tensor, s_cur: torch.Tensor) -> torch.Tensor:
            v_unc = self.velocity(u_cur, s_cur, h_p, c_p, ret_val=None)
            if counterplay_player is None or cfg_scale <= 0.0:
                return v_unc
            rv = torch.full(h_p.shape[:-1], target_ret, device=h_p.device, dtype=h_p.dtype)
            cm = torch.zeros(h_p.shape[:-1], dtype=torch.bool, device=h_p.device)
            cm[..., counterplay_player] = True
            v_cond = self.velocity(u_cur, s_cur, h_p, c_p, ret_val=rv, cond_mask=cm)
            w = cm.to(v_unc.dtype).unsqueeze(-1) * cfg_scale
            return v_unc + w * (v_cond - v_unc)

        for i in range(steps):
            s_cur = torch.full(h_p.shape[:-1], 1.0 - i * dt, device=h_p.device, dtype=h_p.dtype)
            v1 = _eval_v(u, s_cur)
            u_mid = u + 0.5 * dt * v1
            s_mid = torch.full(h_p.shape[:-1], 1.0 - (i + 0.5) * dt, device=h_p.device, dtype=h_p.dtype)
            v2 = _eval_v(u_mid, s_mid)
            u = u + dt * v2
        tempo_next = self.tempo_pred(torch.cat([h_p, c_p.to(h_p.dtype), u], dim=-1))
        return u, tempo_next


class WorldOutcomeFlowPrior(nn.Module):
    """Stochastic World Outcome Flow `p_theta(xi_world | S_tau, u_1, u_2, h_obs)` (`32d`)
    modeling multi-modal tactical resolution (snipes, traps, concave engagements)."""

    def __init__(
        self,
        d_xi_world: int = SP.D_XI_WORLD_DEFAULT,
        d_squad: int = SP.D_SQUAD_DEFAULT,
        d_u: int = SP.innovation_dim(),
        d_obs: int = 512,
        hidden: int = 1024,
    ):
        super().__init__()
        self.d_xi_world = d_xi_world
        freqs = 2.0 * math.pi * torch.arange(1, 17, dtype=torch.float32)
        self.register_buffer("s_freqs", freqs, persistent=False)

        cond_dim = 2 * (SP.N_SQUAD_Q + d_squad) + 2 * (SP.N_MACRO_Q + d_squad) + 2 * d_u + d_obs
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.LayerNorm(d_xi_world + hidden + 32),
            nn.Linear(d_xi_world + hidden + 32, hidden, bias=False),
            nn.SiLU(),
            _TensorCoreResBlock(hidden),
            _TensorCoreResBlock(hidden),
            nn.Linear(hidden, d_xi_world),
        )
        small_init(self.net[-1], std=2e-2)

    def _build_cond(
        self,
        q_squad: torch.Tensor,
        z_squad: torch.Tensor,
        gate: torch.Tensor,
        q_macro: torch.Tensor,
        z_macro: torch.Tensor,
        u_flat: torch.Tensor,
        h_obs: torch.Tensor,
    ) -> torch.Tensor:
        g = gate.float().clamp(0.0, 1.0)
        sq = torch.cat([q_squad.float(), z_squad.float()], dim=-1)
        sq_m = (sq * g.unsqueeze(-1)).sum(dim=-2) / g.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mac = torch.cat([q_macro.float(), z_macro.float()], dim=-1)
        inp = torch.cat(
            [sq_m.flatten(-2), mac.flatten(-2), u_flat.float().flatten(-2), h_obs.float()],
            dim=-1,
        )
        return self.cond_proj(inp)

    def velocity(self, xi_s: torch.Tensor, s: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        ang = s.unsqueeze(-1) * self.s_freqs
        s_emb = torch.cat([ang.sin(), ang.cos()], dim=-1).to(cond.dtype)
        return self.net(torch.cat([xi_s.to(cond.dtype), cond, s_emb], dim=-1))

    def flow_matching_loss(
        self,
        xi_target: torch.Tensor,
        q_squad: torch.Tensor,
        z_squad: torch.Tensor,
        gate: torch.Tensor,
        q_macro: torch.Tensor,
        z_macro: torch.Tensor,
        u_flat: torch.Tensor,
        h_obs: torch.Tensor,
    ) -> torch.Tensor:
        xi_0 = xi_target.detach()
        eps = torch.randn_like(xi_0)
        s = torch.rand(xi_0.shape[:-1], device=xi_0.device, dtype=xi_0.dtype)
        xi_s = (1.0 - s.unsqueeze(-1)) * xi_0 + s.unsqueeze(-1) * eps
        cond = self._build_cond(q_squad, z_squad, gate, q_macro, z_macro, u_flat, h_obs)
        v_pred = self.velocity(xi_s, s, cond)
        return F.mse_loss(v_pred.float(), (xi_0 - eps).float())

    def sample(
        self,
        q_squad: torch.Tensor,
        z_squad: torch.Tensor,
        gate: torch.Tensor,
        q_macro: torch.Tensor,
        z_macro: torch.Tensor,
        u_flat: torch.Tensor,
        h_obs: torch.Tensor,
        steps: int = 4,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        init_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = self._build_cond(q_squad, z_squad, gate, q_macro, z_macro, u_flat, h_obs)
        if init_noise is not None:
            xi = init_noise.to(cond.dtype) * temperature
        else:
            xi = (
                torch.randn(
                    cond.shape[:-1] + (self.d_xi_world,),
                    generator=generator,
                    device=cond.device,
                    dtype=torch.float32,
                ).to(cond.dtype)
                * temperature
            )
        dt = 1.0 / max(1, steps)
        for i in range(steps):
            s_cur = torch.full(cond.shape[:-1], 1.0 - i * dt, device=cond.device, dtype=cond.dtype)
            v1 = self.velocity(xi, s_cur, cond)
            s_mid = torch.full(cond.shape[:-1], 1.0 - (i + 0.5) * dt, device=cond.device, dtype=cond.dtype)
            v2 = self.velocity(xi + 0.5 * dt * v1, s_mid, cond)
            xi = xi + dt * v2
        return xi.float().clamp(-2.5, 2.5)
