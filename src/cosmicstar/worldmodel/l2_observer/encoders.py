"""Perspective-isolated and omniscient macro sequence encoders + `OpponentBeliefHead`."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.constants import MAP_SCALE
from ..l1_tactical import squad_pooler as SP
from ..nn.layers import small_init
from ..nn.rope import ROPE_SHORTEST, RoPE2D


class PerspectiveMacroEncoder(nn.Module):
    """Encodes each player's strictly causal, partial-information perspective history
    `h_{<= tau}^{(p)}` (`[B, T, 2, d_model]`) and online Capability Latent `c_{p, tau}`.

    Guarantees exact `0.000000e+00` hidden-enemy leak on observed prefixes via
    `summarize_observed_opp` (reading engine-masked `q_opp_obs, z_opp_obs, gate_opp_obs`)
    while training `summarize_rollout_opp` via self-supervised alignment for `+120s` rollouts
    and fog-lifted Plan-Misalignment evaluation."""

    def __init__(
        self,
        d_squad: int = SP.D_SQUAD_DEFAULT,
        d_opp_z: int = SP.D_OPP_Z_DEFAULT,
        d_model: int = 768,
        d_cap: int = 16,
        layers: int = 6,
        heads: int = 12,
    ):
        super().__init__()
        self.d_squad = d_squad
        self.d_opp_z = d_opp_z
        self.d_model = d_model
        self.d_cap = d_cap
        self.heads = heads
        self.dh = d_model // heads

        sq_dim = d_squad + SP.N_SQUAD_Q + 1  # + gate
        mac_dim = d_squad + SP.N_MACRO_Q
        self.own_sq_proj = nn.Sequential(
            nn.LayerNorm(sq_dim),
            nn.Linear(sq_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Observed-prefix enemy encoder (strictly consumes engine-masked opp_obs -> 0.0 leak!)
        obs_opp_in = SP.N_OPP_Q + d_opp_z + 1
        self.obs_opp_proj = nn.Sequential(
            nn.LayerNorm(obs_opp_in),
            nn.Linear(obs_opp_in, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Open-loop rollout & fog-lifted enemy encoder (self-supervised to match obs_opp_proj!)
        syn_opp_in = SP.N_SQUAD_Q + d_squad + 3  # + (vis_live, shimmer, snap_struct)
        self.syn_opp_proj = nn.Sequential(
            nn.LayerNorm(syn_opp_in),
            nn.Linear(syn_opp_in, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.step_in_proj = nn.Sequential(
            nn.LayerNorm(2 * d_model + mac_dim + SP.D_TEMPO_FEAT),
            nn.Linear(2 * d_model + mac_dim + SP.D_TEMPO_FEAT, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.qkvs = nn.ModuleList([nn.Linear(d_model, 3 * d_model, bias=False) for _ in range(layers)])
        self.projs = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.SiLU(),
                    nn.Linear(4 * d_model, d_model, bias=False),
                )
                for _ in range(layers)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)

        self.cap_in = nn.Sequential(
            nn.LayerNorm(SP.D_TEMPO_FEAT + SP.N_MACRO_Q),
            nn.Linear(SP.D_TEMPO_FEAT + SP.N_MACRO_Q, 128),
            nn.SiLU(),
            nn.Linear(128, d_cap),
        )
        inv_freq = 1.0 / (1000.0 ** (torch.arange(0, self.dh, 2, dtype=torch.float32) / self.dh))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def summarize_observed_opp(
        self,
        q_opp_obs: torch.Tensor,
        z_opp_obs: torch.Tensor,
        gate_opp_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Summarizes strictly leak-free engine `opp_obs` (`[..., 2, M_opp, ...]`) -> `[..., 2, d_model]`."""
        g = gate_opp_obs.float().clamp(0.0, 1.0)
        denom = g.sum(dim=-1, keepdim=True).clamp(min=1.0)
        inp = torch.cat([q_opp_obs.float(), z_opp_obs.float(), g.unsqueeze(-1)], dim=-1) * (g > 1e-5).float().unsqueeze(-1)
        return (self.obs_opp_proj(inp) * g.unsqueeze(-1)).sum(dim=-2) / denom

    def summarize_rollout_opp(
        self,
        q_squad: torch.Tensor,
        z_squad: torch.Tensor,
        gate: torch.Tensor,
        vis_live: torch.Tensor,
        shimmer: torch.Tensor,
        explored: torch.Tensor,
    ) -> torch.Tensor:
        """Summarizes enemy `1 - p` squads under `(vis_live, shimmer, explored)` for open-loop
        rollouts and fog-lifted Plan-Misalignment queries -> `[..., 2, d_model]`."""
        q_opp = q_squad.flip(dims=[-3]).float()
        z_opp = z_squad.flip(dims=[-3]).float()
        g_opp = gate.flip(dims=[-2]).float().clamp(0.0, 1.0)

        # Hard-threshold tiny sigmoid tails (< 0.03) so completely hidden squads contribute 0.0
        vl = torch.where(vis_live > 0.03, vis_live.float(), torch.zeros_like(vis_live.float())) * g_opp
        sh = torch.where(shimmer > 0.03, shimmer.float(), torch.zeros_like(shimmer.float())) * g_opp * (1.0 - vl).clamp(0.0, 1.0)
        is_st = q_opp[..., SP.SQ["struct_frac"]].clamp(0.0, 1.0)
        sn = torch.where(explored > 0.25, explored.float(), torch.zeros_like(explored.float())) * is_st * g_opp * (1.0 - vl - sh).clamp(0.0, 1.0)

        any_w = (vl + sh + sn).clamp(0.0, 1.0)
        id_w = (vl + sn).clamp(0.0, 1.0)  # Shimmer hides unit identity z_opp, HP, shields, and cost!

        q_masked = torch.cat(
            [
                q_opp[..., 0:2] * any_w.unsqueeze(-1),
                q_opp[..., 2:4] * (vl + sh).clamp(0.0, 1.0).unsqueeze(-1),
                q_opp[..., 4:7] * any_w.unsqueeze(-1),
                q_opp[..., 7:14] * vl.unsqueeze(-1),
            ],
            dim=-1,
        )
        z_masked = z_opp * id_w.unsqueeze(-1)
        modes = torch.stack([vl, sh, sn], dim=-1)
        inp = torch.cat([q_masked, z_masked, modes], dim=-1) * (any_w > 1e-5).float().unsqueeze(-1)
        denom = any_w.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return (self.syn_opp_proj(inp) * any_w.unsqueeze(-1)).sum(dim=-2) / denom

    def summarize_perspective_frame(
        self,
        q_squad: torch.Tensor,
        z_squad: torch.Tensor,
        gate: torch.Tensor,
        q_macro: torch.Tensor,
        z_macro: torch.Tensor,
        tempo_feat: torch.Tensor,
        h_opp_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g_own = gate.float().clamp(0.0, 1.0)
        denom_own = g_own.sum(dim=-1, keepdim=True).clamp(min=1.0)
        sq_own_in = torch.cat([q_squad.float(), z_squad.float(), g_own.unsqueeze(-1)], dim=-1)
        h_own = (self.own_sq_proj(sq_own_in) * g_own.unsqueeze(-1)).sum(dim=-2) / denom_own

        mac_both = torch.cat([q_macro.float(), z_macro.float()], dim=-1)
        tok = self.step_in_proj(torch.cat([h_own, h_opp_summary, mac_both, tempo_feat.float()], dim=-1))
        cap_inst = torch.tanh(self.cap_in(torch.cat([tempo_feat.float(), q_macro.float()], dim=-1)))
        return tok, cap_inst

    def forward_sequence(
        self,
        tok_seq: torch.Tensor,
        cap_inst_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, P, D = tok_seq.shape
        x = tok_seq.transpose(1, 2).reshape(B * P, T, D)

        t_pos = torch.arange(T, device=x.device, dtype=torch.float32)
        ang = torch.outer(t_pos, self.inv_freq)[None, None, :, :]
        rcos, rsin = ang.cos(), ang.sin()

        for n1, qkv, proj, n2, ffn in zip(
            self.norms1, self.qkvs, self.projs, self.norms2, self.ffns
        ):
            h = n1(x)
            q, k, v = qkv(h).chunk(3, dim=-1)
            q = q.view(B * P, T, self.heads, self.dh).transpose(1, 2)
            k = k.view(B * P, T, self.heads, self.dh).transpose(1, 2)
            v = v.view(B * P, T, self.heads, self.dh).transpose(1, 2)
            rc, rs = rcos.to(v.dtype), rsin.to(v.dtype)
            q1, q2 = q[..., ::2], q[..., 1::2]
            k1, k2 = k[..., ::2], k[..., 1::2]
            q = torch.stack([q1 * rc - q2 * rs, q1 * rs + q2 * rc], dim=-1).flatten(-2)
            k = torch.stack([k1 * rc - k2 * rs, k1 * rs + k2 * rc], dim=-1).flatten(-2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + proj(a.transpose(1, 2).reshape(B * P, T, D))
            x = x + ffn(n2(x))

        h_out = self.out_norm(x).view(B, P, T, D).transpose(1, 2)

        idx = torch.arange(T, device=x.device, dtype=torch.float32)
        diff = (idx[:, None] - idx[None, :]).clamp(min=0.0)
        decay = 0.85 ** diff
        ema_w = torch.tril(torch.where(idx[None, :] == 0, decay, 0.15 * decay))
        c_flat = cap_inst_seq.permute(0, 2, 1, 3).reshape(B * P, T, self.d_cap)
        c_seq = torch.matmul(ema_w.to(c_flat.dtype), c_flat).view(B, P, T, self.d_cap).transpose(1, 2)
        return h_out, c_seq


class OpponentBeliefHead(nn.Module):
    """Predicts player `p`'s belief `q_phi(S_{opp} | h_{<= tau}^{(p)})` over hidden enemy
    spatial squad distributions (`M_bel = 16` modes), macro economy, and tech composition."""

    def __init__(self, d_model: int = 768, d_squad: int = SP.D_SQUAD_DEFAULT, m_bel: int = 16):
        super().__init__()
        self.m_bel = m_bel
        self.d_squad = d_squad
        # Per mode: x, y, count, cost, hp, shield, gate_logit (7)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512),
            nn.SiLU(),
            nn.Linear(512, m_bel * 7 + 4 + d_squad),
        )
        small_init(self.net[-1], std=2e-2)

    def forward(self, h_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(h_seq)
        m_dim = self.m_bel * 7
        modes = out[..., :m_dim].view(*h_seq.shape[:-1], self.m_bel, 7)
        pos = modes[..., 0:2].clamp(-2.0, 2.0)
        vals = F.softplus(modes[..., 2:6])
        gate = torch.sigmoid(modes[..., 6])
        mac_pred = F.softplus(out[..., m_dim : m_dim + 4])
        z_pred = out[..., m_dim + 4 :]
        return dict(pos=pos, vals=vals, gate=gate, mac=mac_pred, z_mean=z_pred)

    def belief_loss(
        self,
        h_seq: torch.Tensor,
        q_squad: torch.Tensor,
        z_squad: torch.Tensor,
        gate: torch.Tensor,
        q_macro: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-scale continuous spatial field + macro/tech belief loss against detached `S_{1-p}`."""
        pred = self.forward(h_seq)
        # Flip player axis so player 0 predicts player 1 and player 1 predicts player 0
        opp_q = q_squad.flip(dims=[2]).detach().float()
        opp_z = z_squad.flip(dims=[2]).detach().float()
        opp_g = gate.flip(dims=[2]).detach().float().clamp(0.0, 1.0)
        opp_mac = q_macro.flip(dims=[2])[..., 0:4].detach().float()

        # Spatial field kernel loss between [N, 2, M_bel] belief modes and [N, 2, M] true enemy squads
        B, T, _, M, _ = opp_q.shape
        N = B * T * 2
        pos_p = pred["pos"].float().reshape(N, self.m_bel, 2)
        pos_t = opp_q[..., 0:2].reshape(N, M, 2)
        w_ch = pos_p.new_tensor([1.0, 1.0, 0.25, 0.25])
        val_p = (pred["vals"].float() * w_ch * pred["gate"].float().unsqueeze(-1)).reshape(N, self.m_bel, 4)
        val_t = (opp_q[..., 6:10] * w_ch * opp_g.unsqueeze(-1)).reshape(N, M, 4)

        d_pp = (pos_p.unsqueeze(-2) - pos_p.unsqueeze(-3)).square().sum(dim=-1)
        d_tt = (pos_t.unsqueeze(-2) - pos_t.unsqueeze(-3)).square().sum(dim=-1)
        d_pt = (pos_p.unsqueeze(-2) - pos_t.unsqueeze(-3)).square().sum(dim=-1)
        sigmas = (0.12, 0.30, 0.65)
        inv_s = 1.0 / len(sigmas)
        k_pp = sum(torch.exp(-d_pp * (0.5 / (s * s))) for s in sigmas) * inv_s
        k_tt = sum(torch.exp(-d_tt * (0.5 / (s * s))) for s in sigmas) * inv_s
        k_pt = sum(torch.exp(-d_pt * (0.5 / (s * s))) for s in sigmas) * inv_s
        f_dist = 0.1 * (
            (val_p * torch.matmul(k_pp, val_p)).sum(dim=(-2, -1))
            + (val_t * torch.matmul(k_tt, val_t)).sum(dim=(-2, -1))
            - 2.0 * (val_p * torch.matmul(k_pt, val_t)).sum(dim=(-2, -1))
        ).clamp(min=0.0).mean()

        loss_mac = F.smooth_l1_loss(pred["mac"].float(), opp_mac)
        opp_z_mean = (opp_z * opp_g.unsqueeze(-1)).sum(dim=-2) / opp_g.sum(dim=-1, keepdim=True).clamp(min=1.0)
        loss_z = F.smooth_l1_loss(pred["z_mean"].float(), opp_z_mean)
        return f_dist + 0.5 * loss_mac + 0.5 * loss_z


class OmniscientObserverEncoder(nn.Module):
    """Stream 3 (`h_obs`): Causal Omniscient Observer Encoder over `(S_{<= tau}, z_omni, h^(1), h^(2))`."""

    def __init__(
        self,
        d_squad: int = SP.D_SQUAD_DEFAULT,
        d_persp: int = 768,
        d_obs: int = 512,
        layers: int = 3,
        heads: int = 8,
    ):
        super().__init__()
        self.d_obs = d_obs
        self.heads = heads
        self.dh = d_obs // heads

        sq_dim = SP.N_SQUAD_Q + d_squad
        mac_dim = SP.N_MACRO_Q + d_squad
        in_dim = 2 * sq_dim + 2 * mac_dim + d_squad + 2 * d_persp
        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_obs),
            nn.SiLU(),
            nn.Linear(d_obs, d_obs),
        )
        self.norms1 = nn.ModuleList([nn.LayerNorm(d_obs) for _ in range(layers)])
        self.qkvs = nn.ModuleList([nn.Linear(d_obs, 3 * d_obs, bias=False) for _ in range(layers)])
        self.projs = nn.ModuleList([nn.Linear(d_obs, d_obs, bias=False) for _ in range(layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d_obs) for _ in range(layers)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_obs, 3 * d_obs),
                    nn.SiLU(),
                    nn.Linear(3 * d_obs, d_obs, bias=False),
                )
                for _ in range(layers)
            ]
        )
        self.out_norm = nn.LayerNorm(d_obs)

    def summarize_frame(
        self,
        q_squad: torch.Tensor,
        z_squad: torch.Tensor,
        gate: torch.Tensor,
        q_macro: torch.Tensor,
        z_macro: torch.Tensor,
        z_omni_scene: torch.Tensor,
        h_persp: torch.Tensor,
    ) -> torch.Tensor:
        g = gate.float().clamp(0.0, 1.0)
        sq = torch.cat([q_squad.float(), z_squad.float()], dim=-1)
        sq_mean = (sq * g.unsqueeze(-1)).sum(dim=-2) / g.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mac = torch.cat([q_macro.float(), z_macro.float()], dim=-1)
        inp = torch.cat(
            [
                sq_mean.flatten(-2),
                mac.flatten(-2),
                z_omni_scene.float(),
                h_persp.float().flatten(-2),
            ],
            dim=-1,
        )
        return self.in_proj(inp)

    def forward_sequence(self, obs_tok_seq: torch.Tensor) -> torch.Tensor:
        B, T, D = obs_tok_seq.shape
        x = obs_tok_seq
        for n1, qkv, proj, n2, ffn in zip(
            self.norms1, self.qkvs, self.projs, self.norms2, self.ffns
        ):
            q, k, v = qkv(n1(x)).chunk(3, dim=-1)
            q = q.view(B, T, self.heads, self.dh).transpose(1, 2)
            k = k.view(B, T, self.heads, self.dh).transpose(1, 2)
            v = v.view(B, T, self.heads, self.dh).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + proj(a.transpose(1, 2).reshape(B, T, D))
            x = x + ffn(n2(x))
        return self.out_norm(x)
