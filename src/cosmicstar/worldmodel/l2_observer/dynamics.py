"""Squad-level Port-Hamiltonian dynamics and multi-scale spatial field matching loss."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from ..core.constants import MAP_SCALE
from ..l1_tactical import squad_pooler as SP
from ..nn.layers import small_init
from ..nn.rope import ROPE_SHORTEST, RoPE2D
from .perception import StaticTerrainStencil
from .priors import _TensorCoreResBlock


def _soft_cap_frob(mat: torch.Tensor, max_norm: float, dims: tuple[int, ...]) -> torch.Tensor:
    nrm = (mat.square().sum(dim=dims, keepdim=True) + 1e-12).sqrt()
    return mat * (max_norm * torch.tanh(nrm / max_norm) / nrm)


def _port_combine_and_project(
    sq_out: torch.Tensor,
    mac_out: torch.Tensor,
    x_sq_req: torch.Tensor,
    x_mac_req: torch.Tensor,
    gate_in: torch.Tensor,
    g_sq: torch.Tensor,
    g_mac: torch.Tensor,
    b_mask_sq: torch.Tensor,
    b_mask_mac: torch.Tensor,
    advect_gain: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Smooth continuous-gate rank-2 Port-Hamiltonian projection + physical domain clamping."""
    B, _, M, D_s = x_sq_req.shape
    mac_dim = x_mac_req.shape[-1]
    w_dt = sq_out.dtype
    g_f = gate_in.float().clamp(0.0, 1.0)
    g_w = g_f.to(w_dt)
    d2 = D_s * 2

    uvl_raw = sq_out[..., : 3 * d2].view(B, 2, M, 3, D_s, 2)
    c_sq_raw_unscaled = sq_out[..., 3 * d2 : 4 * d2].view(B, 2, M, D_s, 2)
    harv_raw = sq_out[..., 4 * d2 : 4 * d2 + 2].float()
    d_gate_raw = sq_out[..., 4 * d2 + 2].float()
    bu_sq_raw = sq_out[..., 4 * d2 + 3 :].float()

    max_uvl = uvl_raw.new_tensor([0.55, 0.55, 0.75]).view(1, 1, 1, 3, 1, 1)
    nrm_uvl = (uvl_raw.float().square().sum(dim=(-2, -1), keepdim=True) + 1e-8).sqrt()
    scale_uvl = (max_uvl.float() * torch.tanh(nrm_uvl / max_uvl.float()) / nrm_uvl).to(w_dt)
    uvl_cap = uvl_raw * scale_uvl

    g_sq_w = g_sq.to(w_dt)
    proj3 = (uvl_cap * g_sq_w.view(B, 2, M, 1, D_s, 1)).sum(dim=-2)
    U_j, V_j, L_r = uvl_cap.unbind(dim=-3)
    u_g, v_g, lt_g = proj3.unbind(dim=-2)

    j_sq = (U_j * v_g.unsqueeze(-2) - V_j * u_g.unsqueeze(-2)).sum(dim=-1).float()
    r_sq = (L_r * lt_g.unsqueeze(-2)).sum(dim=-1).float()
    diss_sq = (lt_g.float().square().sum(dim=-1) * g_f).sum(dim=-1)

    adv = advect_gain.float().clamp(0.05, 0.50)
    j_kin = torch.cat(
        [adv * x_sq_req[..., 2:4].float(), torch.zeros_like(j_sq[..., 2:])], dim=-1
    )

    mac_f = mac_out.float()
    md2 = mac_dim * 2
    C_mac = _soft_cap_frob(mac_f[..., :md2].view(B, 2, mac_dim, 2), 0.55, (-2, -1))
    L_rm = _soft_cap_frob(mac_f[..., md2 : 2 * md2].view(B, 2, mac_dim, 2), 0.75, (-2, -1))
    bu_mac_raw = mac_f[..., 2 * md2 :]

    # Allow slight non-zero baseline coupling (0.04 + 0.96*g_w) so economy can spawn new squads!
    c_scaled = c_sq_raw_unscaled * (0.25 * (0.04 + 0.96 * g_w)).unsqueeze(-1).unsqueeze(-1)
    c_nrm = (c_scaled.float().square().sum(dim=(-3, -2, -1), keepdim=True) + 1e-8).sqrt()
    c_scale = (0.55 * torch.tanh(c_nrm / 0.55) / c_nrm).to(w_dt)
    C_sq = c_scaled * c_scale

    g_mac_f = g_mac.float()
    port_from_sq = (C_sq * g_sq_w.unsqueeze(-1)).sum(dim=(2, 3)).float()
    port_from_mac = (C_mac * g_mac_f.unsqueeze(-1)).sum(dim=-2)
    j_mac_cpl = (C_mac * port_from_sq.unsqueeze(-2)).sum(dim=-1)
    j_sq_cpl = -(C_sq * port_from_mac.to(w_dt)[:, :, None, None, :]).sum(dim=-1).float()

    lt_gm = (L_rm * g_mac_f.unsqueeze(-1)).sum(dim=-2)
    r_mac = (L_rm * lt_gm.unsqueeze(-2)).sum(dim=-1)

    sq_count = x_sq_req[..., SP.SQ["count"]].float().clamp(min=0.0, max=6.0) * g_f
    harv_rate = F.softplus(harv_raw).clamp(max=0.6)
    harv_inflow = (harv_rate * sq_count.unsqueeze(-1)).sum(dim=2)
    harv_mac = torch.cat([harv_inflow, torch.zeros_like(r_mac[..., 2:])], dim=-1)

    bu_sq = (1.5 * torch.tanh(bu_sq_raw / 1.5)) * b_mask_sq
    bu_mac = (1.5 * torch.tanh(bu_mac_raw / 1.5)) * b_mask_mac

    # Own-squad dynamics weighted by g_f; economy->squad spawn coupling j_sq_cpl allowed to open slots!
    dx_sq = (j_sq + j_kin - r_sq + bu_sq) * g_f.unsqueeze(-1) + j_sq_cpl
    dx_mac = (j_mac_cpl - r_mac) + harv_mac + bu_mac

    x_sq_next_raw = x_sq_req.float() + dx_sq
    x_mac_next_raw = x_mac_req.float() + dx_mac

    q_sq_raw = x_sq_next_raw[..., : SP.N_SQUAD_Q]
    q_sq_n = torch.cat(
        [
            q_sq_raw[..., 0:2].clamp(-2.0, 2.0),
            q_sq_raw[..., 2:4].clamp(-4.0, 4.0),
            q_sq_raw[..., 4:6].clamp(0.01, 1.5),
            q_sq_raw[..., 6:10].clamp(0.0, 15.0),
            q_sq_raw[..., 10:14].clamp(0.0, 1.0),
        ],
        dim=-1,
    )

    # Smooth C^inf continuous gate evolution (ZERO hard boolean step jump!)
    mass_indicator = torch.sigmoid(
        ((q_sq_n[..., SP.SQ["count"]] + q_sq_n[..., SP.SQ["hp"]]) - 4e-4) * 500.0
    )
    gate_next = ((g_f + 0.35 * torch.tanh(d_gate_raw)).clamp(0.0, 1.0) * mass_indicator).clamp(0.0, 1.0)
    survives = gate_next > 0.02

    # Smoothly attenuate empty slots via continuous gate_next rather than discontinuous boolean mask
    att = (gate_next / (gate_next + 1e-3)).unsqueeze(-1)
    q_sq_n = q_sq_n * att
    z_sq_n = (x_sq_next_raw[..., SP.N_SQUAD_Q :].clamp(-4.0, 4.0)) * att

    q_mac_raw = x_mac_next_raw[..., : SP.N_MACRO_Q]
    q_mac_n = torch.cat(
        [
            q_mac_raw[..., 0:4].clamp(0.0, 20.0),
            q_mac_raw[..., 4:6].clamp(-2.0, 2.0),
        ],
        dim=-1,
    )
    z_mac_n = x_mac_next_raw[..., SP.N_MACRO_Q :].clamp(-3.0, 3.0)
    return q_sq_n, z_sq_n, gate_next, survives, q_mac_n, z_mac_n, bu_sq, bu_mac, diss_sq


class SquadPortHamiltonian(nn.Module):
    """Purely Learned Token-Level Port-Hamiltonian World Model over `1` Macro + `M = 64`
    Perceiver Squad Tokens per player + `32d` Stochastic World Outcome `xi_world`.

    ZERO hardcoded `bank + unit_cost + hp` formula; ZERO raw `cost` input into `H` (`w_adv_sq`
    masks out `x, y, vx, vy, cost` so army/tech value is learned 100% from unit embeddings
    `z_squad`, counts, health/shields, terrain stencils, and 2D-RoPE enemy matchup `tug`).
    """

    def __init__(
        self,
        d_squad: int = SP.D_SQUAD_DEFAULT,
        n_squads: int = SP.N_SQUADS_DEFAULT,
        k_intent: int = SP.K_INTENT_DEFAULT,
        d_intent: int = SP.D_INTENT_DEFAULT,
        d_xi: int = SP.D_XI_DEFAULT,
        d_xi_world: int = SP.D_XI_WORLD_DEFAULT,
        hidden: int = 2048,
        d_terr: int = 128,
        d_tug: int = 256,
        d_ctrl: int = 256,
        rank: int = 12,
    ):
        super().__init__()
        self.d_squad = d_squad
        self.n_squads = n_squads
        self.k_intent = k_intent
        self.d_intent = d_intent
        self.d_xi = d_xi
        self.d_xi_world = d_xi_world
        self.hidden = hidden
        self.d_terr = d_terr
        self.d_tug = d_tug
        self.d_ctrl = d_ctrl
        self.tug_heads = d_tug // 64
        self.ctrl_heads = d_ctrl // 64
        self.rank = rank

        self.sq_dim = SP.N_SQUAD_Q + d_squad   # 142
        self.mac_dim = SP.N_MACRO_Q + d_squad  # 134

        # Mask kinematics (0..3) AND raw mineral+gas cost header (col 7) out of H!
        w_adv_sq = torch.ones(self.sq_dim, dtype=torch.float32)
        w_adv_sq[0:4] = 0.0
        w_adv_sq[SP.SQ["cost"]] = 0.0
        w_adv_mac = torch.ones(self.mac_dim, dtype=torch.float32)
        w_adv_mac[4:6] = 0.0  # mask camera (cam_x, cam_y) out of intrinsic economic potential
        self.register_buffer("w_adv_sq", w_adv_sq)
        self.register_buffer("w_adv_mac", w_adv_mac)

        b_mask_sq = torch.ones(self.sq_dim, dtype=torch.float32)
        for col in ("x", "y", "count", "cost", "hp", "shield"):
            b_mask_sq[SP.SQ[col]] = 0.0
        b_mask_mac = torch.ones(self.mac_dim, dtype=torch.float32)
        for col in ("minerals", "vespene", "food_used", "food_cap"):
            b_mask_mac[SP.MQ[col]] = 0.0
        self.register_buffer("b_mask_sq", b_mask_sq)
        self.register_buffer("b_mask_mac", b_mask_mac)

        self.terrain = StaticTerrainStencil(d_out=d_terr)
        self.rope_2d = RoPE2D(64, lo=ROPE_SHORTEST, hi=256.0)

        # 1. Purely Learned Gate-Weighted Squad & Macro Potentials H_1, H_2
        pot_hid = 512
        self.sq_pot_net = nn.Sequential(
            nn.Linear(self.sq_dim + d_terr, pot_hid, bias=False),
            nn.SiLU(),
            nn.Linear(pot_hid, pot_hid, bias=False),
            nn.SiLU(),
            nn.Linear(pot_hid, 1),
        )
        small_init(self.sq_pot_net[-1], std=1e-2)

        self.mac_pot_net = nn.Sequential(
            nn.Linear(self.mac_dim, pot_hid, bias=False),
            nn.SiLU(),
            nn.Linear(pot_hid, pot_hid, bias=False),
            nn.SiLU(),
            nn.Linear(pot_hid, 1),
        )
        small_init(self.mac_pot_net[-1], std=1e-2)

        # Multi-head 2D-RoPE Cross-Player Tactical Matchup Interaction `tug`
        self.tug_qkv = nn.Linear(self.sq_dim + d_terr, 3 * d_tug, bias=False)
        self.tug_sink = nn.Parameter(torch.full((self.tug_heads,), 2.0))
        self.tug_head = nn.Sequential(
            nn.Linear(self.sq_dim + d_terr + d_tug, pot_hid, bias=False),
            nn.SiLU(),
            nn.Linear(pot_hid, pot_hid, bias=False),
            nn.SiLU(),
            nn.Linear(pot_hid, 1, bias=False),
        )
        small_init(self.tug_head[-1], std=5e-3)

        self.kin_net = nn.Sequential(
            nn.Linear(self.sq_dim + d_terr, pot_hid, bias=False),
            nn.SiLU(),
            nn.Linear(pot_hid, 2),
        )
        small_init(self.kin_net[-1], std=1e-2)

        # 2. Multi-head 2D-RoPE Spatial Intent -> Squad routing + Deep 2048-wide Operator Trunk
        self.ctrl_q = nn.Linear(self.sq_dim + d_terr, d_ctrl, bias=False)
        self.ctrl_kv = nn.Linear(d_intent, 2 * d_ctrl, bias=False)
        self.ctrl_sink = nn.Parameter(torch.full((self.ctrl_heads,), 1.5))

        op_sq_in_dim = self.sq_dim + d_terr + d_tug + d_ctrl + d_intent + d_xi + d_xi_world
        op_sq_out_dim = 4 * (self.sq_dim * 2) + 3 + self.sq_dim  # + harv(2) + d_gate(1) + bu_sq
        self.op_sq_norm = nn.LayerNorm(op_sq_in_dim)
        self.op_sq_trunk = nn.Sequential(
            nn.Linear(op_sq_in_dim, hidden, bias=False),
            nn.SiLU(),
            _TensorCoreResBlock(hidden),
            _TensorCoreResBlock(hidden),
            _TensorCoreResBlock(hidden),
            _TensorCoreResBlock(hidden),
            nn.Linear(hidden, op_sq_out_dim),
        )
        small_init(self.op_sq_trunk[-1], std=1e-4)
        with torch.no_grad():
            h_off = 4 * (self.sq_dim * 2)
            self.op_sq_trunk[-1].bias[h_off : h_off + 2].fill_(-4.2)

        op_mac_in_dim = self.mac_dim + d_intent + d_xi + d_xi_world
        op_mac_out_dim = self.mac_dim * 2 + self.mac_dim * 2 + self.mac_dim
        self.op_mac_norm = nn.LayerNorm(op_mac_in_dim)
        self.op_mac_trunk = nn.Sequential(
            nn.Linear(op_mac_in_dim, 768, bias=False),
            nn.SiLU(),
            nn.Linear(768, 768, bias=False),
            nn.SiLU(),
            nn.Linear(768, op_mac_out_dim),
        )
        small_init(self.op_mac_trunk[-1], std=1.5e-4)

        self.advect_gain = nn.Parameter(torch.tensor(0.25))
        self._op_step_fn = self._apply_operators_and_step

    def _cross_squad_context(
        self,
        x_sq: torch.Tensor,
        gate: torch.Tensor,
        terr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Smooth continuous-gate skew-symmetric 2D-RoPE `tug` energy `[B]` + `opp_ctx`."""
        B, _, M, _ = x_sq.shape
        B2 = B * 2
        H = self.tug_heads
        pos_both = x_sq[..., 0:2] * MAP_SCALE
        in_both = torch.cat([x_sq * self.w_adv_sq, terr], dim=-1)

        qkv_both = self.tug_qkv(in_both)
        q_raw, k_raw, v_raw = qkv_both.chunk(3, dim=-1)

        src = in_both.reshape(B2, M, -1)
        pos_s = pos_both.reshape(B2, M, 2)
        pos_d = pos_both.flip(dims=[1]).reshape(B2, M, 2)
        g_s = gate.float().clamp(0.0, 1.0).reshape(B2, M)
        g_d = gate.float().clamp(0.0, 1.0).flip(dims=[1]).reshape(B2, M)

        q_rot = SP._apply_rope_2d(
            F.normalize(q_raw.reshape(B2, M, H, 64), dim=-1, eps=1e-4).reshape(B2, M, self.d_tug),
            pos_s,
            self.rope_2d,
        ).view(B2, M, H, 64).transpose(1, 2)
        k_rot = SP._apply_rope_2d(
            F.normalize(k_raw.flip(dims=[1]).reshape(B2, M, H, 64), dim=-1, eps=1e-4).reshape(B2, M, self.d_tug),
            pos_d,
            self.rope_2d,
        ).view(B2, M, H, 64).transpose(1, 2)
        v_dst = (
            v_raw.flip(dims=[1]).reshape(B2, M, H, 64).transpose(1, 2)
            * g_d[:, None, :, None].to(v_raw.dtype)
        )

        # Continuous log-gate attention bias (zero discontinuous boolean threshold!)
        log_g_d = torch.log(g_d.clamp(min=1e-5))[:, None, None, :].to(q_rot.dtype)
        scores = torch.matmul(q_rot, k_rot.transpose(-1, -2).to(q_rot.dtype)) * 6.0 + log_g_d
        sink = self.tug_sink.view(1, H, 1, 1).expand(B2, H, M, 1).to(scores.dtype)
        w_aug = F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)
        w = w_aug[..., :M]
        ctx = (
            torch.matmul(w.to(v_dst.dtype), v_dst)
            .transpose(1, 2)
            .reshape(B2, M, self.d_tug)
            * g_s.to(v_dst.dtype).unsqueeze(-1)
        )
        tug_raw = self.tug_head(torch.cat([src, ctx], dim=-1)).squeeze(-1).float()
        val_both = (3.5 * torch.tanh(tug_raw / 3.5) * g_s).sum(dim=-1).view(B, 2)
        tug = 0.5 * (val_both[:, 0] - val_both[:, 1])
        return tug, ctx.view(B, 2, M, self.d_tug)

    def evaluate_hamiltonian(
        self,
        x_sq: torch.Tensor,
        x_mac: torch.Tensor,
        gate: torch.Tensor,
        static_map: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluates purely learned `H_1, H_2, A_1 = H_1 - H_2` (ZERO `bank + cost` formula!)."""
        terr = self.terrain(x_sq[..., 0:2], static_map)
        g_f = gate.float().clamp(0.0, 1.0)

        dt_h = torch.bfloat16 if torch.is_autocast_enabled() else x_sq.dtype
        sq_in = torch.cat([x_sq * self.w_adv_sq, terr], dim=-1).to(dt_h)
        sq_pot_raw = self.sq_pot_net(sq_in).squeeze(-1).float()
        sq_pot = (4.0 * torch.tanh(sq_pot_raw / 4.0) * g_f).sum(dim=-1)
        mac_pot = 8.0 * torch.tanh(
            self.mac_pot_net((x_mac * self.w_adv_mac).to(dt_h)).squeeze(-1).float() / 8.0
        )
        own = sq_pot + mac_pot

        k_w = 2.0 * torch.sigmoid(self.kin_net(torch.cat([x_sq.detach(), terr.detach()], dim=-1)).float())
        vel = x_sq[:, :, :, 2:4].float()
        mass_w = (
            x_sq[:, :, :, SP.SQ["hp"]].detach().float()
            + 0.25 * x_sq[:, :, :, SP.SQ["count"]].detach().float()
        ).clamp(min=0.01, max=6.0)
        kin = 0.5 * ((k_w * vel.square()).sum(dim=-1) * mass_w * g_f).sum(dim=-1)
        grad_kin_v = k_w * vel * (mass_w * g_f).unsqueeze(-1)

        tug, opp_ctx = self._cross_squad_context(x_sq, g_f, terr)
        H1 = own[:, 0] + tug
        H2 = own[:, 1] - tug
        A1 = H1 - H2

        return dict(
            H1=H1,
            H2=H2,
            A1=A1,
            own1=own[:, 0],
            own2=own[:, 1],
            tug=tug,
            kin1=kin[:, 0],
            kin2=kin[:, 1],
            grad_kin_v=grad_kin_v,
            terr=terr,
            opp_ctx=opp_ctx,
        )

    def _apply_operators_and_step(
        self,
        x_sq_req: torch.Tensor,
        x_mac_req: torch.Tensor,
        gate: torch.Tensor,
        g_sq: torch.Tensor,
        g_mac: torch.Tensor,
        terr: torch.Tensor,
        opp_ctx: torch.Tensor,
        u_flat: torch.Tensor,
        xi_world: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        u_struct = SP.unpack_innovation(
            u_flat,
            k_intent=self.k_intent,
            d_intent=self.d_intent,
            d_xi=self.d_xi,
        )
        xi = u_struct["xi"]
        B, _, M, _ = x_sq_req.shape
        B2 = B * 2
        Hc = self.ctrl_heads

        sq_pos_map = (x_sq_req[..., 0:2] * MAP_SCALE).reshape(B2, M, 2)
        int_pos_map = (u_struct["intent_pos"] * MAP_SCALE).reshape(B2, self.k_intent, 2)
        cq_raw = self.ctrl_q(torch.cat([x_sq_req, terr], dim=-1)).reshape(B2, M, Hc, 64)
        ck_raw, cv_raw = self.ctrl_kv(u_struct["intent_z"]).chunk(2, dim=-1)
        cq = SP._apply_rope_2d(
            F.normalize(cq_raw, dim=-1, eps=1e-4).reshape(B2, M, self.d_ctrl),
            sq_pos_map,
            self.rope_2d,
        ).view(B2, M, Hc, 64).transpose(1, 2)
        ck = SP._apply_rope_2d(
            F.normalize(ck_raw.reshape(B2, self.k_intent, Hc, 64), dim=-1, eps=1e-4).reshape(B2, self.k_intent, self.d_ctrl),
            int_pos_map,
            self.rope_2d,
        ).view(B2, self.k_intent, Hc, 64).transpose(1, 2)
        cv = (
            (cv_raw * u_struct["intent_gate"].unsqueeze(-1))
            .reshape(B2, self.k_intent, Hc, 64)
            .transpose(1, 2)
        )
        c_scores = torch.matmul(cq, ck.transpose(-1, -2).to(cq.dtype)) * 6.0
        c_sink = self.ctrl_sink.view(1, Hc, 1, 1).expand(B2, Hc, M, 1).to(c_scores.dtype)
        c_attn = F.softmax(torch.cat([c_scores, c_sink], dim=-1), dim=-1)[..., : self.k_intent]
        int_local = (
            torch.matmul(c_attn.to(cv.dtype), cv)
            .transpose(1, 2)
            .reshape(B, 2, M, self.d_ctrl)
        )
        mac_int_exp = u_struct["intent_macro"].unsqueeze(2).expand(B, 2, M, -1)
        xi_exp = xi.unsqueeze(2).expand(B, 2, M, -1)
        # Orient [own_16, opp_16] halves per player so swap(P1 <-> P2) + swap_halves(xi_world) is 100% equivariant!
        xw_halves = xi_world.view(B, 2, self.d_xi_world // 2)
        xw_oriented = torch.stack(
            [
                torch.cat([xw_halves[:, 0], xw_halves[:, 1]], dim=-1),
                torch.cat([xw_halves[:, 1], xw_halves[:, 0]], dim=-1),
            ],
            dim=1,
        )  # [B, 2, d_xi_world]
        xi_w_sq = xw_oriented.unsqueeze(2).expand(B, 2, M, -1).to(x_sq_req.dtype)
        xi_w_mac = xw_oriented.to(x_mac_req.dtype)

        dt_op = torch.bfloat16 if torch.is_autocast_enabled() else x_sq_req.dtype
        op_sq_normed = self.op_sq_norm(
            torch.cat([x_sq_req, terr, opp_ctx, int_local, mac_int_exp, xi_exp, xi_w_sq], dim=-1)
        ).to(dt_op)
        sq_out = self.op_sq_trunk(op_sq_normed)

        mac_out = self.op_mac_trunk(
            self.op_mac_norm(torch.cat([x_mac_req, u_struct["intent_macro"], xi, xi_w_mac], dim=-1)).to(dt_op)
        )

        return _port_combine_and_project(
            sq_out,
            mac_out,
            x_sq_req,
            x_mac_req,
            gate,
            g_sq,
            g_mac,
            self.b_mask_sq,
            self.b_mask_mac,
            self.advect_gain,
        )

    def _apply_operators_ckpt(
        self,
        x_sq_req: torch.Tensor,
        x_mac_req: torch.Tensor,
        gate: torch.Tensor,
        g_sq: torch.Tensor,
        g_mac: torch.Tensor,
        terr: torch.Tensor,
        opp_ctx: torch.Tensor,
        u_flat: torch.Tensor,
        xi_world: torch.Tensor,
    ):
        if torch.is_grad_enabled():
            return grad_checkpoint(
                self._apply_operators_and_step,
                x_sq_req,
                x_mac_req,
                gate,
                g_sq,
                g_mac,
                terr,
                opp_ctx,
                u_flat,
                xi_world,
                use_reentrant=False,
            )
        return self._apply_operators_and_step(
            x_sq_req,
            x_mac_req,
            gate,
            g_sq,
            g_mac,
            terr,
            opp_ctx,
            u_flat,
            xi_world,
        )

    def step_dynamics(
        self,
        state: dict[str, torch.Tensor],
        u_flat: torch.Tensor,
        xi_world: torch.Tensor | None = None,
        static_map: dict[str, torch.Tensor] | None = None,
        create_graph_h: bool | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Advances `(q_squad, z_squad, gate, active, q_macro, z_macro)` by one `5.0s` step."""
        outer_grad = torch.is_grad_enabled()
        q_sq = state["q_squad"].float()
        z_sq = state["z_squad"].float()
        gate = state["gate"].float() if "gate" in state else state["active"].float()
        q_mac = state["q_macro"].float()
        z_mac = state["z_macro"].float()
        B = q_sq.shape[0]
        if xi_world is None:
            xi_world = torch.zeros(B, self.d_xi_world, device=q_sq.device, dtype=torch.float32)

        x_sq_in = torch.cat([q_sq, z_sq], dim=-1)
        x_mac_in = torch.cat([q_mac, z_mac], dim=-1)
        cg = (self.training and outer_grad) if create_graph_h is None else (create_graph_h and outer_grad)

        with torch.enable_grad():
            x_sq_h = (
                x_sq_in
                if (cg and x_sq_in.requires_grad)
                else x_sq_in.detach().requires_grad_(True)
            )
            x_mac_h = (
                x_mac_in
                if (cg and x_mac_in.requires_grad)
                else x_mac_in.detach().requires_grad_(True)
            )
            h_out = self.evaluate_hamiltonian(
                x_sq_h, x_mac_h, gate, static_map=static_map
            )

            total_pot = h_out["own1"].sum() + h_out["own2"].sum()
            g_sq_own, g_mac_own = torch.autograd.grad(
                total_pot, (x_sq_h, x_mac_h), create_graph=cg, retain_graph=True
            )
            (g_sq_tug,) = torch.autograd.grad(
                h_out["tug"].sum(), (x_sq_h,), create_graph=cg, retain_graph=cg
            )

        x_sq_step = x_sq_in if outer_grad else x_sq_in.detach()
        x_mac_step = x_mac_in if outer_grad else x_mac_in.detach()
        grad_kin_v = h_out["grad_kin_v"] if cg else h_out["grad_kin_v"].detach()
        if not cg:
            h_out = {
                k: (v.detach() if isinstance(v, torch.Tensor) else v)
                for k, v in h_out.items()
            }

        sign = torch.tensor([1.0, -1.0], device=q_sq.device, dtype=torch.float32)[None, :, None, None]
        g_sq_adv = g_sq_own.float() + sign * g_sq_tug.float()
        g_mac_adv = g_mac_own.float()
        g_sq_goal = g_sq_own.float() + 2.0 * sign * g_sq_tug.float()
        g_mac_goal = g_mac_own.float()

        g_kin_full = torch.cat(
            [
                torch.zeros_like(grad_kin_v),
                grad_kin_v,
                torch.zeros_like(g_sq_adv[..., 4:]),
            ],
            dim=-1,
        )
        g_sq = _soft_cap_frob((g_sq_adv + g_kin_full) * gate.clamp(0.0, 1.0).unsqueeze(-1), 6.0, (-1,))
        g_mac = _soft_cap_frob(g_mac_adv, 6.0, (-1,))

        if self._op_step_fn == self._apply_operators_and_step and outer_grad:
            q_sq_n, z_sq_n, gate_n, survives, q_mac_n, z_mac_n, bu_sq, bu_mac, diss_sq = grad_checkpoint(
                self._apply_operators_and_step,
                x_sq_step,
                x_mac_step,
                gate,
                g_sq,
                g_mac,
                h_out["terr"],
                h_out["opp_ctx"],
                u_flat,
                xi_world,
                use_reentrant=False,
            )
        else:
            q_sq_n, z_sq_n, gate_n, survives, q_mac_n, z_mac_n, bu_sq, bu_mac, diss_sq = self._op_step_fn(
                x_sq_step,
                x_mac_step,
                gate,
                g_sq,
                g_mac,
                h_out["terr"],
                h_out["opp_ctx"],
                u_flat,
                xi_world,
            )

        next_state = dict(
            q_squad=q_sq_n,
            z_squad=z_sq_n,
            gate=gate_n,
            active=survives,
            q_macro=q_mac_n,
            z_macro=z_mac_n,
        )
        pieces = dict(
            bu_sq=bu_sq,
            bu_mac=bu_mac,
            g_sq_goal=g_sq_goal,
            g_mac_goal=g_mac_goal,
            diss=diss_sq,
            **h_out,
        )
        return next_state, pieces


def _multi_scale_spatial_field_loss(
    pred_s: dict[str, torch.Tensor],
    true_s: dict[str, torch.Tensor],
    sigmas: tuple[float, ...] = (0.08, 0.22, 0.50),
) -> torch.Tensor:
    pos_p = pred_s["q_squad"][..., 0:2].float()
    pos_t = true_s["q_squad"][..., 0:2].float()
    gp = (pred_s["gate"].float() if "gate" in pred_s else pred_s["active"].float()).clamp(0.0, 1.0)
    gt = (true_s["gate"].float() if "gate" in true_s else true_s["active"].float()).clamp(0.0, 1.0)
    w_ch = pos_p.new_tensor([1.0, 1.0, 0.25, 0.25])
    # Note q_squad[..., 6:10] are already partition-of-unity weighted; gate weights soft presence
    val_p = pred_s["q_squad"][..., 6:10].float() * w_ch * (gp > 1e-4).float().unsqueeze(-1)
    val_t = true_s["q_squad"][..., 6:10].float() * w_ch * (gt > 1e-4).float().unsqueeze(-1)

    d_pp = (pos_p.unsqueeze(-2) - pos_p.unsqueeze(-3)).square().sum(dim=-1)
    d_tt = (pos_t.unsqueeze(-2) - pos_t.unsqueeze(-3)).square().sum(dim=-1)
    d_pt = (pos_p.unsqueeze(-2) - pos_t.unsqueeze(-3)).square().sum(dim=-1)

    inv_w = 1.0 / len(sigmas)
    k_pp = sum(torch.exp(-d_pp * (0.5 / (s * s))) for s in sigmas) * inv_w
    k_tt = sum(torch.exp(-d_tt * (0.5 / (s * s))) for s in sigmas) * inv_w
    k_pt = sum(torch.exp(-d_pt * (0.5 / (s * s))) for s in sigmas) * inv_w

    term_pp = (val_p * torch.matmul(k_pp, val_p)).sum(dim=(-2, -1))
    term_tt = (val_t * torch.matmul(k_tt, val_t)).sum(dim=(-2, -1))
    term_pt = (val_p * torch.matmul(k_pt, val_t)).sum(dim=(-2, -1))
    dist2 = 0.1 * (term_pp + term_tt - 2.0 * term_pt).clamp(min=0.0)
    return dist2.mean()


_compiled_field_loss = None

