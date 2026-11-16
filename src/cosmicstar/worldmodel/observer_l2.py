"""Layer 2 (`5.0s` Stride, `120s` History -> `+120s` Rollout) Multi-Scale Stochastic Observer v2.

Core architectural properties:
  1. **Zero Hardcoded Value Anchors (`A = H_1 - H_2`)**:
     - `SquadPortHamiltonian` contains ZERO `bank + unit_cost + hp` formulas, ZERO raw `cost`
       input into `H`, ZERO `phys_adv` baseline, and ZERO winner/Elo labels.
     - `H_1, H_2` are 100% learned continuous gate-weighted functionals grounded by:
       (a) Port-Hamiltonian transition dynamics `dS = (J - R) grad H + B(S)(u, xi_world)`,
       (b) Player control goal alignment (`B u_p` ascending `+grad_p A_p`), and
       (c) Multi-horizon (`10s, 20s, 40s, 80s, 120s`) TD bootstrapping strictly over realized
           combat/structural destruction impulses `r_t = loss_P2 - loss_P1`.
  2. **Airtight 3-Stream Theory-of-Mind (`h_1`, `h_2`, `h_obs`)**:
     - `PerspectiveMacroEncoder` consumes strictly view-faithful `opp_obs` on observed frames
       (`0.000000e+00` hidden-enemy forward & gradient leak!) and transitions seamlessly to
       `SyntheticOppSummaryHead` + cloak/shimmer-aware `SightFootprintHead` during `+120s`
       open-loop rollouts.
     - `OpponentBeliefHead` trains `h^(p)` to infer hidden `S_opp` (`loss_belief`).
     - `OmniscientObserverEncoder` builds `h^(obs)` while shared-policy divergence between
       actual fog `pi(u_p | h^(p))` and fog-lifted `pi(u_p | h^(omni, p))` measures **Plan
       Misalignment** and attributes it directly to unscouted enemy squads.
  3. **Stochastic World Outcome Flow + Dual-Mode Optimal Counterplay + Intrinsic Leverage**:
     - `WorldOutcomeFlowPrior` samples `32d` `xi_world` combat/trap/snipe outcomes without
       denoised-mean blurring, using Common Random Numbers (`seed`) across timeline scrubs.
     - `ConditionalFlowPrior` uses `50%` Classifier-Free `Delta A` dropout + `+grad_u A`
       manifold polish to support both `Forecast Play` and `Optimal Counterplay`.
     - **Intrinsic Learned Leverage** `||grad_{z_m, xy_m} A|| * gate_m` is computed in a single
       differentiable Hamiltonian pass with zero leave-one-out deletion hacks.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from . import energy as E
from . import energy_data as ED
from . import squad_pooler as SP


class StaticTerrainStencil(nn.Module):
    """Samples static map terrain height, pathability, and placement grids at arbitrary
    continuous normalized coordinates `pos_norm` (`[..., N, 2]`, where `1.0 == 100` map
    units relative to map center) using a 5-point cardinal stencil (`0, +-4` map units)
    plus multi-scale 2D Fourier position features."""

    def __init__(self, d_out: int = 128, stencil_map_units: float = 4.0, n_freqs: int = 6):
        super().__init__()
        self.d_out = d_out
        self.register_buffer(
            "omega",
            2.0 * math.pi / torch.logspace(math.log10(4.0), math.log10(256.0), n_freqs),
            persistent=False,
        )
        offsets = torch.tensor(
            [
                [0.0, 0.0],
                [stencil_map_units, 0.0],
                [-stencil_map_units, 0.0],
                [0.0, stencil_map_units],
                [0.0, -stencil_map_units],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("offsets", offsets, persistent=False)
        in_dim = 4 * n_freqs + 5 * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU(),
            nn.Linear(256, d_out),
        )

    def forward(
        self,
        pos_norm: torch.Tensor,
        static_map: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        orig_shape = pos_norm.shape
        N_outer = orig_shape[0]
        dev = pos_norm.device
        n_pts = self.offsets.shape[0]

        if (
            static_map is not None
            and ("static_grid_chw" in static_map or "static_grid" in static_map)
            and "center" in static_map
            and "origin" in static_map
        ):
            if "static_grid_chw" in static_map:
                st = static_map["static_grid_chw"]
            else:
                st = static_map["static_grid"].float().permute(0, 3, 1, 2)
            B_map = st.shape[0]
            pos_flat = pos_norm.reshape(B_map, -1, 2).float() * E.MAP_SCALE
            gh, gw = st.shape[2], st.shape[3]
            off = (static_map["center"].float() - static_map["origin"].float())[:, None, None, :]
            wh = pos_flat.new_tensor([float(gw), float(gh)])
            at = ((pos_flat.unsqueeze(-2) + self.offsets + off) / wh) * 2.0 - 1.0
            with torch.autocast(device_type=dev.type, enabled=False):
                samp = (
                    F.grid_sample(st.float(), at, align_corners=False)
                    .permute(0, 2, 3, 1)
                    .reshape(B_map, pos_flat.shape[1], n_pts * 4)
                )
        else:
            B_map = N_outer
            pos_flat = pos_norm.reshape(B_map, -1, 2).float() * E.MAP_SCALE
            samp = torch.zeros(B_map, pos_flat.shape[1], n_pts * 4, device=dev, dtype=torch.float32)

        ang = pos_flat.unsqueeze(-1) * self.omega
        fourier = torch.cat([ang.sin().flatten(-2), ang.cos().flatten(-2)], dim=-1)
        out = self.net(torch.cat([fourier, samp], dim=-1))
        return out.reshape(*orig_shape[:-1], self.d_out)


def extract_rich_static_map(game) -> dict[str, torch.Tensor]:
    """Extracts 4-channel playable static map tensor `[height_norm, pathable, buildable, playable]`
    (`[gh, gw, 4]` fp16 on CPU) plus `origin`, `center`, `half` directly from `PD.Game`."""
    import numpy as np

    d = game.rest[1]
    H, W = game.H, game.W
    x0, y0, x1, y1 = game.playable
    h_norm = d["map_terrain_height"].reshape(H, W)[y0:y1, x0:x1].astype(np.float32) / 255.0
    if "map_pathing_grid" in d.files:
        path = np.unpackbits(d["map_pathing_grid"].reshape(H, -1), axis=1)[:, :W][y0:y1, x0:x1].astype(np.float32)
    else:
        path = np.ones_like(h_norm)
    if "map_placement_grid" in d.files:
        place = np.unpackbits(d["map_placement_grid"].reshape(H, -1), axis=1)[:, :W][y0:y1, x0:x1].astype(np.float32)
    else:
        place = path
    play = np.ones_like(h_norm)
    grid = np.stack([h_norm, path, place, play], axis=-1)
    return dict(
        static_grid=torch.from_numpy(grid).to(torch.float16),
        origin=torch.tensor([float(x0), float(y0)], dtype=torch.float32),
        center=torch.from_numpy(game.center.copy()).float(),
        half=torch.from_numpy(game.half.copy()).float(),
    )


def collate_static_maps(
    maps_list: list[dict[str, torch.Tensor]],
    device: torch.device | str = "cuda",
) -> dict[str, torch.Tensor]:
    """Pads heterogeneous playable map grids `[gh_i, gw_i, 4]` to `[B, 4, max_gh, max_gw]`
    preserving exact `(origin, center)` world-coordinate alignment."""
    B = len(maps_list)
    if all("static_grid_chw" in m for m in maps_list):
        max_h = max(int(m["static_grid_chw"].shape[1]) for m in maps_list)
        max_w = max(int(m["static_grid_chw"].shape[2]) for m in maps_list)
        st_chw = torch.zeros((B, 4, max_h, max_w), dtype=torch.float32)
        for i, m in enumerate(maps_list):
            g = m["static_grid_chw"]
            _, h, w = g.shape
            st_chw[i, :, :h, :w] = g
        st_chw_dev = st_chw.to(device, non_blocking=True)
        return dict(
            static_grid_chw=st_chw_dev,
            origin=torch.stack([m["origin"].float() for m in maps_list], dim=0).to(device, non_blocking=True),
            center=torch.stack([m["center"].float() for m in maps_list], dim=0).to(device, non_blocking=True),
            half=torch.stack([m["half"].float() for m in maps_list], dim=0).to(device, non_blocking=True),
        )

    max_h = max(int(m["static_grid"].shape[0]) for m in maps_list)
    max_w = max(int(m["static_grid"].shape[1]) for m in maps_list)
    st = torch.zeros((B, max_h, max_w, 4), dtype=torch.float32)
    for i, m in enumerate(maps_list):
        g = m["static_grid"].float()
        if float(g[..., :2].max()) > 2.0:
            g = torch.cat([g[..., :2] / 255.0, g[..., 2:]], dim=-1)
        h, w, _ = g.shape
        st[i, :h, :w] = g
    st_dev = st.to(device, non_blocking=True)
    return dict(
        static_grid=st_dev,
        static_grid_chw=st_dev.permute(0, 3, 1, 2).contiguous(),
        origin=torch.stack([m["origin"].float() for m in maps_list], dim=0).to(device, non_blocking=True),
        center=torch.stack([m["center"].float() for m in maps_list], dim=0).to(device, non_blocking=True),
        half=torch.stack([m["half"].float() for m in maps_list], dim=0).to(device, non_blocking=True),
    )


def extract_gt_vis_live(
    b: dict[str, torch.Tensor],
    f_end: int,
    q_squad: torch.Tensor,
) -> torch.Tensor:
    """Samples ground-truth SC2 engine live visibility (`vis == 2`, grid channel `p*5 + 0`)
    at enemy squad centers `(x_m, y_m)` -> `[B, 2, M]` in `[0, 1]`."""
    dev = q_squad.device
    B, _, M, _ = q_squad.shape
    if "grid" not in b or "center" not in b or "origin" not in b:
        return torch.zeros(B, 2, M, device=dev)
    grid = b["grid"][:, f_end].float()
    gh, gw = grid.shape[1], grid.shape[2]
    off = (b["center"].float() - b["origin"].float())[:, None, :]
    wh = torch.tensor([gw, gh], device=dev, dtype=torch.float32)

    out = []
    for p_i in (0, 1):
        opp = 1 - p_i
        opp_pos_map = q_squad[:, opp, :, 0:2].float() * E.MAP_SCALE
        at = (((opp_pos_map + off) / wh) * 2.0 - 1.0).unsqueeze(2)
        vis_ch = grid[..., p_i * 5 + 0].unsqueeze(1)
        v = F.grid_sample(vis_ch, at, align_corners=False).view(B, M)
        out.append(v.clamp(0.0, 1.0))
    return torch.stack(out, dim=1)


def extract_squad_gt_perception(
    ctx: dict[str, torch.Tensor],
    f_end_ctx: int,
    state: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregates exact per-unit engine `(gt_vis_live, gt_shimmer)` onto enemy Perceiver
    squads via `state["assign_w"]` -> `(gt_vis_live [B, 2, M], gt_shimmer [B, 2, M])`."""
    perc = ctx["perception"]
    vl_u = perc["gt_vis_live"][:, f_end_ctx].float()   # [B, 2, R]
    sh_u = perc["gt_shimmer"][:, f_end_ctx].float()    # [B, 2, R]
    aw = state["assign_w"].float()                     # [B, 2, M, R]
    gate = state["gate"].float()                       # [B, 2, M]

    vl_sq, sh_sq = [], []
    for p_i in (0, 1):
        opp = 1 - p_i
        w_opp = aw[:, opp]                                             # [B, M, R]
        denom = w_opp.sum(dim=-1).clamp(min=1e-6)                      # [B, M]
        vl_m = ((w_opp * vl_u[:, p_i].unsqueeze(1)).sum(dim=-1) / denom) * gate[:, opp]
        sh_m = ((w_opp * sh_u[:, p_i].unsqueeze(1)).sum(dim=-1) / denom) * gate[:, opp]
        vl_sq.append(vl_m.clamp(0.0, 1.0))
        sh_sq.append(sh_m.clamp(0.0, 1.0))
    return torch.stack(vl_sq, dim=1), torch.stack(sh_sq, dim=1)


class SightFootprintHead(nn.Module):
    """Predicts each squad's effective sight & detection radii so open-loop `+120s` rollouts
    dynamically update `(vis_live, shimmer, explored)` including cloak shimmer vs. detection."""

    def __init__(self, d_squad: int = SP.D_SQUAD_DEFAULT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_squad + SP.N_SQUAD_Q),
            nn.Linear(d_squad + SP.N_SQUAD_Q, 64),
            nn.SiLU(),
            nn.Linear(64, 2),  # (sight_delta, det_delta)
        )
        E.small_init(self.net[-1], std=1e-2)

    def radii(self, q_squad: torch.Tensor, z_squad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spread_r = (
            q_squad[..., SP.SQ["spread_x"]].square()
            + q_squad[..., SP.SQ["spread_y"]].square()
            + 1e-6
        ).sqrt()
        raw = self.net(torch.cat([q_squad, z_squad], dim=-1))
        r_sight = (0.10 + spread_r + 0.06 * torch.tanh(raw[..., 0])).clamp(0.05, 0.35)
        det_frac = q_squad[..., SP.SQ["det_frac"]].clamp(0.0, 1.0)
        r_det = (0.11 + spread_r + 0.04 * torch.tanh(raw[..., 1])).clamp(0.06, 0.25) * (det_frac > 0.02).float()
        return r_sight, r_det

    def compute_visibility(
        self,
        q_squad: torch.Tensor,
        z_squad: torch.Tensor,
        gate: torch.Tensor,
        prev_explored: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns `(vis_live, shimmer, explored)` (`[..., 2, M]` in `[0, 1]`), where:
          - `vis_live`: uncloaked or detected enemy fraction inside live sight,
          - `shimmer`: undetected cloaked enemy fraction inside live sight (e.g. Stasis Trap!)."""
        r_sight, r_det = self.radii(q_squad, z_squad)
        pos = q_squad[..., 0:2]
        g_f = gate.float().clamp(0.0, 1.0)
        my_active = g_f > 0.03

        live_list, shim_list = [], []
        for p_i in (0, 1):
            opp = 1 - p_i
            my_pos = pos[..., p_i, :, :]
            opp_pos = pos[..., opp, :, :]
            dist = ((my_pos.unsqueeze(-2) - opp_pos.unsqueeze(-3)).square().sum(dim=-1) + 1e-6).sqrt()

            s_margin = ((r_sight[..., p_i, :].unsqueeze(-1) - dist) / 0.015).masked_fill(
                ~my_active[..., p_i, :].unsqueeze(-1), -25.0
            )
            in_sight = torch.sigmoid(s_margin.max(dim=-2).values)

            has_det = my_active[..., p_i, :] & (q_squad[..., p_i, :, SP.SQ["det_frac"]] > 0.02)
            d_margin = ((r_det[..., p_i, :].unsqueeze(-1) - dist) / 0.015).masked_fill(
                ~has_det.unsqueeze(-1), -25.0
            )
            in_det = torch.sigmoid(d_margin.max(dim=-2).values)

            opp_clk = q_squad[..., opp, :, SP.SQ["cloak_frac"]].clamp(0.0, 1.0)
            opp_g = g_f[..., opp, :]
            v_live = in_sight * ((1.0 - opp_clk) + opp_clk * in_det) * opp_g
            v_shim = in_sight * opp_clk * (1.0 - in_det) * opp_g
            live_list.append(v_live)
            shim_list.append(v_shim)

        vis_live = torch.stack(live_list, dim=-2)
        shimmer = torch.stack(shim_list, dim=-2)
        if prev_explored is None:
            explored = vis_live.detach()
        else:
            explored = torch.maximum(prev_explored, vis_live.detach())
        return vis_live, shimmer, explored


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
        E.small_init(self.net[-1], std=2e-2)

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


class _TensorCoreResBlock(nn.Module):
    """High-arithmetic-intensity bf16 Tensor Core residual MLP block."""

    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        E.small_init(self.fc2, std=0.02)

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
        E.small_init(self.net[-1], std=2e-2)

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
        E.small_init(self.net[-1], std=2e-2)

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
        self.rope_2d = ED.RoPE2D(64, lo=ED.ROPE_SHORTEST, hi=256.0)

        # 1. Purely Learned Gate-Weighted Squad & Macro Potentials H_1, H_2
        pot_hid = 512
        self.sq_pot_net = nn.Sequential(
            nn.Linear(self.sq_dim + d_terr, pot_hid, bias=False),
            nn.SiLU(),
            nn.Linear(pot_hid, pot_hid, bias=False),
            nn.SiLU(),
            nn.Linear(pot_hid, 1),
        )
        E.small_init(self.sq_pot_net[-1], std=1e-2)

        self.mac_pot_net = nn.Sequential(
            nn.Linear(self.mac_dim, pot_hid, bias=False),
            nn.SiLU(),
            nn.Linear(pot_hid, pot_hid, bias=False),
            nn.SiLU(),
            nn.Linear(pot_hid, 1),
        )
        E.small_init(self.mac_pot_net[-1], std=1e-2)

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
        E.small_init(self.tug_head[-1], std=5e-3)

        self.kin_net = nn.Sequential(
            nn.Linear(self.sq_dim + d_terr, pot_hid, bias=False),
            nn.SiLU(),
            nn.Linear(pot_hid, 2),
        )
        E.small_init(self.kin_net[-1], std=1e-2)

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
        E.small_init(self.op_sq_trunk[-1], std=1e-4)
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
        E.small_init(self.op_mac_trunk[-1], std=1.5e-4)

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
        pos_both = x_sq[..., 0:2] * E.MAP_SCALE
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

        sq_pos_map = (x_sq_req[..., 0:2] * E.MAP_SCALE).reshape(B2, M, 2)
        int_pos_map = (u_struct["intent_pos"] * E.MAP_SCALE).reshape(B2, self.k_intent, 2)
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


class ObserverWorldModelL2(nn.Module):
    """Multi-Scale Stochastic Observer v2 (`120s` History -> `+120s` Rollout)."""

    def __init__(
        self,
        d_squad: int = SP.D_SQUAD_DEFAULT,
        n_squads: int = SP.N_SQUADS_DEFAULT,
        d_model: int = 768,
        d_obs: int = 512,
        d_cap: int = 16,
        layers: int = 6,
        heads: int = 12,
        rollout_steps: int = 4,
    ):
        super().__init__()
        self.d_squad = d_squad
        self.n_squads = n_squads
        self.d_model = d_model
        self.d_obs = d_obs
        self.d_cap = d_cap
        self.rollout_steps = rollout_steps

        self.sight = SightFootprintHead(d_squad=d_squad)
        self.persp_enc = PerspectiveMacroEncoder(
            d_squad=d_squad, d_model=d_model, d_cap=d_cap, layers=layers, heads=heads
        )
        self.belief_head = OpponentBeliefHead(d_model=d_model, d_squad=d_squad)
        self.obs_enc = OmniscientObserverEncoder(
            d_squad=d_squad, d_persp=d_model, d_obs=d_obs
        )
        self.flow_prior = ConditionalFlowPrior(
            d_u=SP.innovation_dim(), d_model=d_model, d_cap=d_cap
        )
        self.xi_prior = WorldOutcomeFlowPrior(
            d_xi_world=SP.D_XI_WORLD_DEFAULT,
            d_squad=d_squad,
            d_u=SP.innovation_dim(),
            d_obs=d_obs,
        )
        self.ph = SquadPortHamiltonian(d_squad=d_squad, n_squads=n_squads)

    def compile_parts(self):
        global _compiled_field_loss
        torch._dynamo.config.recompile_limit = 256
        torch._functorch.config.donated_buffer = False
        self.sight.compile(dynamic=True)
        self.persp_enc.compile(dynamic=True)
        self.obs_enc.compile(dynamic=True)
        self.flow_prior.compile(dynamic=True)
        self.xi_prior.compile(dynamic=True)
        if _compiled_field_loss is None:
            _compiled_field_loss = torch.compile(_multi_scale_spatial_field_loss, dynamic=True)

    def _state_transition_losses(
        self,
        pred_s: dict[str, torch.Tensor],
        true_s: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        true_s = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in true_s.items()}
        gp = (pred_s["gate"].float() if "gate" in pred_s else pred_s["active"].float()).clamp(0.0, 1.0)
        gt = (true_s["gate"].float() if "gate" in true_s else true_s["active"].float()).clamp(0.0, 1.0)

        tot_p = pred_s["q_squad"][..., 6:10].sum(dim=-2)
        tot_t = true_s["q_squad"][..., 6:10].sum(dim=-2)
        mac_p = pred_s["q_macro"][..., 0:4]
        mac_t = true_s["q_macro"][..., 0:4]

        mse_tot = (tot_p - tot_t).square().mean(dim=(0, 1))
        mse_mac = (mac_p - mac_t).square().mean(dim=(0, 1))

        w_slot = torch.maximum(gp, gt)
        denom_act = w_slot.sum().clamp(min=1.0)
        sq_diff = (pred_s["q_squad"] - true_s["q_squad"]).square().sum(dim=-1) * w_slot
        loss_slot_q = sq_diff.sum() / denom_act
        loss_gate = F.smooth_l1_loss(gp, gt)

        fl_fn = _compiled_field_loss if _compiled_field_loss is not None else _multi_scale_spatial_field_loss
        loss_field = fl_fn(pred_s, true_s)

        loss_phys = (
            4.0 * mse_tot.sum()
            + 2.0 * mse_mac.sum()
            + 0.5 * loss_slot_q
            + 0.5 * loss_gate
            + 2.0 * loss_field
        )

        lat_sq = F.smooth_l1_loss(
            pred_s["z_squad"] * w_slot.unsqueeze(-1),
            true_s["z_squad"] * w_slot.unsqueeze(-1),
        )
        lat_mac = F.smooth_l1_loss(pred_s["z_macro"], true_s["z_macro"])
        loss_lat = lat_sq + 0.5 * lat_mac

        metrics = dict(
            mse_count=mse_tot[0].detach(),
            mse_cost=mse_tot[1].detach(),
            mse_hp=mse_tot[2].detach(),
            mse_shield=mse_tot[3].detach(),
            mse_minerals=mse_mac[0].detach(),
            mse_vespene=mse_mac[1].detach(),
            loss_field=loss_field.detach(),
        )
        return loss_phys, loss_lat, metrics

    def compute_intrinsic_leverage(
        self,
        state: dict[str, torch.Tensor],
        static_map: dict[str, torch.Tensor] | None = None,
        grid_h: int = 64,
        grid_w: int = 64,
    ) -> dict[str, torch.Tensor]:
        """Computes purely learned intrinsic leverage `||grad_{z_m, xy_m} A||_2 * gate_m`
        in ONE differentiable Hamiltonian evaluation (zero leave-one-out deletion hacks!)."""
        dev = state["q_squad"].device
        q_sq = state["q_squad"][:1].detach().float().requires_grad_(True)
        z_sq = state["z_squad"][:1].detach().float().requires_grad_(True)
        gate = (state["gate"][:1] if "gate" in state else state["active"][:1]).detach().float().clamp(0.0, 1.0)
        x_mac = torch.cat([state["q_macro"][:1], state["z_macro"][:1]], dim=-1).detach().float()

        with torch.enable_grad():
            x_sq = torch.cat([q_sq, z_sq], dim=-1)
            h_out = self.ph.evaluate_hamiltonian(x_sq, x_mac, gate, static_map=static_map)
            a0 = h_out["A1"][0]
            gz, gq = torch.autograd.grad(a0, (z_sq, q_sq), create_graph=False)

        # Combine latent composition/role sensitivity + spatial positioning sensitivity (* 1000 res scale)
        lev_z = gz[0].float().norm(p=2, dim=-1)
        lev_xy = gq[0, ..., 0:2].float().norm(p=2, dim=-1) * 0.25
        squad_lev = (lev_z + lev_xy) * gate[0] * 1000.0  # [2, M]

        if static_map is not None and "half" in static_map:
            hx = float(static_map["half"][0, 0].item()) / E.MAP_SCALE
            hy = float(static_map["half"][0, 1].item()) / E.MAP_SCALE
        else:
            hx, hy = 0.85, 0.85
        xs = torch.linspace(-hx, hx, grid_w, device=dev)
        ys = torch.linspace(-hy, hy, grid_h, device=dev)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid_pts = torch.stack([gx, gy], dim=-1).view(-1, 2)  # [G, 2]

        lev_grids = []
        g_denom = gate[0].clamp(min=0.05)
        pos_0 = q_sq.detach()[0, :, :, 0:2] / g_denom.unsqueeze(-1)  # [2, M, 2] true normalized map coords
        spread_0 = ((q_sq.detach()[0, :, :, 4:6] / g_denom.unsqueeze(-1)).square().mean(dim=-1) + 0.05**2).sqrt()
        for p_i in (0, 1):
            w_p = squad_lev[p_i]  # [M]
            d2 = (grid_pts.unsqueeze(1) - pos_0[p_i].unsqueeze(0)).square().sum(dim=-1)  # [G, M]
            sig2 = (2.0 * spread_0[p_i].clamp(0.05, 0.14).square()).unsqueeze(0)
            g_map = torch.matmul(torch.exp(-d2 / sig2), w_p).view(grid_h, grid_w)
            lev_grids.append(g_map)

        return dict(
            A0=a0.detach() * 1000.0,
            H1=h_out["H1"][0].detach() * 1000.0,
            H2=h_out["H2"][0].detach() * 1000.0,
            squad_leverage=squad_lev.detach(),
            leverage_grid_p1=lev_grids[0].detach(),
            leverage_grid_p2=lev_grids[1].detach(),
        )

    def compute_plan_misalignment(
        self,
        prefix_seq: dict[str, torch.Tensor],
        h_last_fog: torch.Tensor,
        c_last_fog: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Measures shared-policy divergence between actual fog-bound `pi(u_p | h^(p), c_p)`
        and fog-lifted `pi(u_p | h^(omni, p), c_p)`, attributing misalignment directly to
        individual unscouted enemy squads via gradient sensitivity!"""
        q_sq = prefix_seq["q_squad"].float()    # [1, T0, 2, M, 14]
        z_sq = prefix_seq["z_squad"].float()
        gate = (prefix_seq["gate"].float() if "gate" in prefix_seq else prefix_seq["active"].float()).clamp(0.0, 1.0)
        q_mac = prefix_seq["q_macro"].float()
        z_mac = prefix_seq["z_macro"].float()
        tempo = prefix_seq["tempo_feat"].float()
        _, T0, _, M, _ = q_sq.shape
        dev = q_sq.device

        # Prefix steps 0 .. T0-2 remain actual history; at T0-1 lift fog completely!
        if "q_opp_obs" in prefix_seq:
            h_opp_hist = self.persp_enc.summarize_observed_opp(
                prefix_seq["q_opp_obs"], prefix_seq["z_opp_obs"], prefix_seq["gate_opp_obs"]
            )
        else:
            vl_p, sh_p, ex_p = self.sight.compute_visibility(q_sq, z_sq, gate)
            h_opp_hist = self.persp_enc.summarize_rollout_opp(q_sq, z_sq, gate, vl_p, sh_p, ex_p)

        # Differentiable visibility lift on final frame T0 - 1 to attribute unscouted threats
        opp_gate_last = gate[:, -1].flip(dims=[1]).detach()  # [1, 2, M]
        if "gt_vis_live" in prefix_seq:
            actual_vis_last = prefix_seq["gt_vis_live"][:, -1].float().clamp(0.0, 1.0)
        else:
            actual_vis_last, _, _ = self.sight.compute_visibility(q_sq[:, -1], z_sq[:, -1], gate[:, -1])

        unscouted_mask = (1.0 - actual_vis_last) * opp_gate_last  # [1, 2, M]: [p_viewer, m_enemy]

        with torch.enable_grad():
            vis_lift = opp_gate_last.clone().requires_grad_(True)
            z_zero = torch.zeros_like(vis_lift)
            h_opp_omni_last = self.persp_enc.summarize_rollout_opp(
                q_sq[:, -1].detach(),
                z_sq[:, -1].detach(),
                gate[:, -1].detach(),
                vis_lift,
                z_zero,
                vis_lift,
            )  # [1, 2, d_model]
            if T0 > 1:
                h_opp_omni_seq = torch.cat([h_opp_hist[:, :-1].detach(), h_opp_omni_last.unsqueeze(1)], dim=1)
            else:
                h_opp_omni_seq = h_opp_omni_last.unsqueeze(1)

            tok_omni, cap_omni = self.persp_enc.summarize_perspective_frame(
                q_sq.detach(), z_sq.detach(), gate.detach(), q_mac.detach(), z_mac.detach(), tempo.detach(), h_opp_omni_seq
            )
            h_omni_seq, _ = self.persp_enc.forward_sequence(tok_omni, cap_omni)
            h_omni_last = h_omni_seq[:, -1]  # [1, 2, d_model]

            # Evaluate policy velocity divergence across K_probe = 8 Common Random Probes
            K_p = 8
            u_probes = torch.randn(
                (K_p, 2, self.flow_prior.d_u), generator=generator, device=dev, dtype=h_omni_last.dtype
            )
            s_probe = torch.full((K_p, 2), 0.5, device=dev, dtype=h_omni_last.dtype)
            v_fog = self.flow_prior.velocity(
                u_probes, s_probe, h_last_fog.detach().expand(K_p, -1, -1), c_last_fog.detach().expand(K_p, -1, -1)
            )
            v_omni = self.flow_prior.velocity(
                u_probes, s_probe, h_omni_last.expand(K_p, -1, -1), c_last_fog.detach().expand(K_p, -1, -1)
            )
            # Per-player RMS intent velocity shift [2]
            div_per_p = (v_omni.float() - v_fog.float()).square().mean(dim=(0, 2)).sqrt()  # [2]
            (g_lift,) = torch.autograd.grad(div_per_p.sum(), (vis_lift,), create_graph=False)

        # Attribution score on enemy `1 - p`'s squads for viewer `p`: unscouted * |d(div_p)/d(vis_m)|
        unscouted_attr = (g_lift[0].abs() * unscouted_mask[0]).detach()  # [2, M]: viewer p -> enemy 1-p slot m
        return dict(
            plan_misalignment=div_per_p.detach(),       # [2]
            unscouted_attribution=unscouted_attr,       # [2, M]
            unscouted_mask=unscouted_mask[0].detach(),  # [2, M]
        )

    def forward_sequence(
        self,
        seq: dict[str, torch.Tensor],
        static_map: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Trains ObserverWorldModelL2 across `[B, T, ...]` (`T = 48` = `240s` macro sequences)."""
        q_sq = seq["q_squad"].float()
        z_sq = seq["z_squad"].float()
        gate = (seq["gate"].float() if "gate" in seq else seq["active"].float()).clamp(0.0, 1.0)
        active = gate > 0.02
        q_mac = seq["q_macro"].float()
        z_mac = seq["z_macro"].float()
        u_post = seq["u_flat"].float()
        tempo = seq["tempo_feat"].float()
        B, T, _, M, _ = q_sq.shape
        dev = q_sq.device

        xi_w_post = (
            seq["xi_world"].float()
            if "xi_world" in seq
            else torch.zeros(B, T - 1, SP.D_XI_WORLD_DEFAULT, device=dev)
        )
        combat_loss = (
            seq["combat_loss"].float()
            if "combat_loss" in seq
            else torch.zeros(B, T - 1, 2, device=dev)
        )
        z_omni = (
            seq["z_omni_scene"].float()
            if "z_omni_scene" in seq
            else torch.zeros(B, T, self.d_squad, device=dev)
        )

        # 1. Dynamic visibility & shimmer calibration against engine perception
        vis_seq, shim_seq, _ = self.sight.compute_visibility(q_sq, z_sq, gate)
        exp_seq = vis_seq.detach().cummax(dim=1).values

        loss_vis = q_sq.sum() * 0.0
        if "gt_vis_live" in seq:
            with torch.autocast(device_type=dev.type, enabled=False):
                opp_g = gate.flip(dims=[2]).float()
                denom_v = opp_g.sum().clamp(min=1.0)
                bce_v = F.binary_cross_entropy(
                    vis_seq.float().clamp(1e-4, 1.0 - 1e-4),
                    seq["gt_vis_live"].float().clamp(0.0, 1.0),
                    reduction="none",
                )
                loss_vis = (bce_v * opp_g).sum() / denom_v
                if "gt_shimmer" in seq:
                    bce_s = F.binary_cross_entropy(
                        shim_seq.float().clamp(1e-4, 1.0 - 1e-4),
                        seq["gt_shimmer"].float().clamp(0.0, 1.0),
                        reduction="none",
                    )
                    loss_vis = loss_vis + 0.5 * (bce_s * opp_g).sum() / denom_v

        # 2. Strictly leak-free Perspective Streams h^(1), h^(2) + rollout projector alignment
        h_opp_syn = self.persp_enc.summarize_rollout_opp(
            q_sq.detach(), z_sq.detach(), gate.detach(), vis_seq.detach(), shim_seq.detach(), exp_seq.detach()
        )
        if "q_opp_obs" in seq:
            h_opp_obs = self.persp_enc.summarize_observed_opp(
                seq["q_opp_obs"], seq["z_opp_obs"], seq["gate_opp_obs"]
            )
            loss_opp_proj = F.mse_loss(h_opp_syn.float(), h_opp_obs.detach().float())
            h_opp_for_persp = h_opp_obs
        else:
            loss_opp_proj = q_sq.sum() * 0.0
            h_opp_for_persp = h_opp_syn

        tok_seq, cap_inst = self.persp_enc.summarize_perspective_frame(
            q_sq, z_sq, gate, q_mac, z_mac, tempo, h_opp_for_persp
        )
        h_seq, c_seq = self.persp_enc.forward_sequence(tok_seq, cap_inst)

        # 3. Theory-of-Mind Opponent-Belief Head + Omniscient Stream 3 (h_obs)
        loss_belief = self.belief_head.belief_loss(h_seq, q_sq, z_sq, gate, q_mac)
        obs_tok_seq = self.obs_enc.summarize_frame(
            q_sq, z_sq, gate, q_mac, z_mac, z_omni, h_seq.detach()
        )
        h_obs_seq = self.obs_enc.forward_sequence(obs_tok_seq)

        # 4. Sharp 1-step (5s) Port-Hamiltonian Transition on Hindsight (u_post, xi_world)
        # Detach s_t_flat state inputs so transition MSE trains ph + u_post + xi_world without pulling pooler to collapse!
        M1 = B * (T - 1)
        s_t_flat = dict(
            q_squad=q_sq[:, :-1].detach().reshape(M1, 2, M, SP.N_SQUAD_Q),
            z_squad=z_sq[:, :-1].detach().reshape(M1, 2, M, self.d_squad),
            gate=gate[:, :-1].detach().reshape(M1, 2, M),
            active=active[:, :-1].detach().reshape(M1, 2, M),
            q_macro=q_mac[:, :-1].detach().reshape(M1, 2, SP.N_MACRO_Q),
            z_macro=z_mac[:, :-1].detach().reshape(M1, 2, self.d_squad),
        )
        s_n_flat = dict(
            q_squad=q_sq[:, 1:].detach().reshape(M1, 2, M, SP.N_SQUAD_Q),
            z_squad=z_sq[:, 1:].detach().reshape(M1, 2, M, self.d_squad),
            gate=gate[:, 1:].detach().reshape(M1, 2, M),
            active=active[:, 1:].detach().reshape(M1, 2, M),
            q_macro=q_mac[:, 1:].detach().reshape(M1, 2, SP.N_MACRO_Q),
            z_macro=z_mac[:, 1:].detach().reshape(M1, 2, self.d_squad),
        )

        pred_1, pieces_1 = self.ph.step_dynamics(
            s_t_flat,
            u_post.reshape(M1, 2, -1),
            xi_world=xi_w_post.reshape(M1, -1),
            static_map=static_map,
        )
        loss_phys_1, loss_lat_1, m1_logs = self._state_transition_losses(pred_1, s_n_flat)
        _, _, copy_logs = self._state_transition_losses(s_t_flat, s_n_flat)

        # Action Goal Alignment: B(S) u_p ascends +grad_p A_p
        bu_cat = torch.cat([pieces_1["bu_sq"].flatten(2), pieces_1["bu_mac"]], dim=-1).float()
        gg_cat = torch.cat([pieces_1["g_sq_goal"].flatten(2), pieces_1["g_mac_goal"]], dim=-1).detach().float()
        bn = (bu_cat.square().sum(dim=-1) + 1e-8).sqrt()
        gn = (gg_cat.square().sum(dim=-1) + 1e-8).sqrt()
        val_g = (bn > 1e-3) & (gn > 1e-3)
        cos_g = (bu_cat * gg_cat).sum(dim=-1) / (bn * gn).clamp(min=1e-6)
        loss_goal = (F.softplus(-2.0 * cos_g) * val_g.float()).sum() / val_g.float().sum().clamp(min=1.0)
        goal_cos = (cos_g * val_g.float()).sum().detach() / val_g.float().sum().clamp(min=1.0)

        # 5. Purely Learned Advantage A = H_1 - H_2 & Multi-Horizon Combat-Destruction TD!
        # ZERO bank+cost baseline, ZERO phys_adv, ZERO winner labels!
        h_all = self.ph.evaluate_hamiltonian(
            torch.cat([q_sq.detach(), z_sq], dim=-1).reshape(B * T, 2, M, -1),
            torch.cat([q_mac.detach(), z_mac], dim=-1).reshape(B * T, 2, -1),
            gate.detach().reshape(B * T, 2, M),
            static_map=static_map,
        )
        a1_seq = h_all["A1"].float().view(B, T)

        # Step reward r_t = combat_loss_P2 - combat_loss_P1 (in 1,000-mineral+gas units destroyed)
        r_step = (combat_loss[:, :, 1] - combat_loss[:, :, 0]).detach()  # [B, T-1]
        gamma = 0.97
        loss_td = a1_seq.sum() * 0.0
        horizons = [m for m in (2, 4, 8, 16, 24) if m < T]
        for m in horizons:
            # Discounted sum of realized combat/structural destruction impulses over [t .. t + m - 1]
            disc_ret = torch.zeros_like(a1_seq[:, :-m])
            g_pow = 1.0
            for k_s in range(m):
                disc_ret = disc_ret + g_pow * r_step[:, k_s : T - m + k_s]
                g_pow *= gamma
            tgt_a = disc_ret + (gamma ** m) * a1_seq[:, m:].detach()
            loss_td = loss_td + F.smooth_l1_loss(a1_seq[:, :-m], tgt_a) / len(horizons)

        # 6. Dual-Mode CFG Strategy Flow Prior + World Outcome Flow Prior
        h_t = h_seq[:, :-1]
        c_t = c_seq[:, :-1]
        # Player-oriented future advantage change Delta A_p for CFG conditioning (with 50% null dropout)
        m_cfg = min(8, T - 1)
        fut_idx = (torch.arange(T - 1, device=dev) + m_cfg).clamp(max=T - 1)
        da_p1 = (a1_seq[:, fut_idx] - a1_seq[:, :-1]).detach()
        delta_adv_p = torch.stack([da_p1, -da_p1], dim=-1)  # [B, T-1, 2]

        loss_flow = self.flow_prior.flow_matching_loss(
            u_post, h_t, c_t, delta_adv_p=delta_adv_p, cfg_drop_prob=0.50
        )
        loss_xi_flow = self.xi_prior.flow_matching_loss(
            xi_w_post,
            q_sq[:, :-1],
            z_sq[:, :-1],
            gate[:, :-1],
            q_mac[:, :-1],
            z_mac[:, :-1],
            u_post,
            h_obs_seq[:, :-1],
        )

        tempo_pred = self.flow_prior.tempo_pred(torch.cat([h_t, c_t, u_post.detach()], dim=-1)).float()
        loss_tempo = F.smooth_l1_loss(tempo_pred, tempo[:, 1:].detach())

        # 7. Sharp Multi-Step Rollout Stability (k = 2..K on sharp hindsight (u_post, xi_world)!)
        K = min(self.rollout_steps, T - 1)
        loss_phys_multi = loss_phys_1 * 0.0
        loss_lat_multi = loss_lat_1 * 0.0
        if K >= 2 and T > K:
            MK = B * (T - K)
            cur_s = dict(
                q_squad=q_sq[:, : T - K].detach().reshape(MK, 2, M, SP.N_SQUAD_Q),
                z_squad=z_sq[:, : T - K].detach().reshape(MK, 2, M, self.d_squad),
                gate=gate[:, : T - K].detach().reshape(MK, 2, M),
                active=active[:, : T - K].detach().reshape(MK, 2, M),
                q_macro=q_mac[:, : T - K].detach().reshape(MK, 2, SP.N_MACRO_Q),
                z_macro=z_mac[:, : T - K].detach().reshape(MK, 2, self.d_squad),
            )
            for k in range(1, K + 1):
                uk = u_post[:, k - 1 : T - K + k - 1].detach().reshape(MK, 2, -1)
                xk = xi_w_post[:, k - 1 : T - K + k - 1].detach().reshape(MK, -1)
                cur_s, _ = self.ph.step_dynamics(
                    cur_s, uk, xi_world=xk, static_map=static_map, create_graph_h=False
                )
                if k >= 2:
                    true_k = dict(
                        q_squad=q_sq[:, k : T - K + k].reshape(MK, 2, M, SP.N_SQUAD_Q),
                        z_squad=z_sq[:, k : T - K + k].reshape(MK, 2, M, self.d_squad),
                        gate=gate[:, k : T - K + k].reshape(MK, 2, M),
                        active=active[:, k : T - K + k].reshape(MK, 2, M),
                        q_macro=q_mac[:, k : T - K + k].reshape(MK, 2, SP.N_MACRO_Q),
                        z_macro=z_mac[:, k : T - K + k].reshape(MK, 2, self.d_squad),
                    )
                    w_k = 0.5 ** (k - 1)
                    lp_k, ll_k, _ = self._state_transition_losses(cur_s, true_k)
                    loss_phys_multi = loss_phys_multi + w_k * lp_k
                    loss_lat_multi = loss_lat_multi + w_k * ll_k

        loss = (
            loss_phys_1
            + 0.5 * loss_phys_multi
            + 0.15 * (loss_lat_1 + 0.5 * loss_lat_multi)
            + 1.0 * loss_flow.float()
            + 0.5 * loss_xi_flow.float()
            + 0.25 * loss_belief.float()
            + 0.15 * loss_opp_proj.float()
            + 0.10 * loss_tempo
            + 0.10 * loss_vis
            + 0.05 * loss_goal
            + 0.50 * loss_td
        )

        logs = dict(
            loss=loss.detach(),
            loss_phys=loss_phys_1.detach(),
            loss_phys_multi=loss_phys_multi.detach(),
            loss_flow=loss_flow.detach().float(),
            loss_xi_flow=loss_xi_flow.detach().float(),
            loss_belief=loss_belief.detach().float(),
            loss_opp_proj=loss_opp_proj.detach().float(),
            loss_td=loss_td.detach(),
            loss_vis=loss_vis.detach(),
            loss_goal=loss_goal.detach(),
            goal_cos=goal_cos,
            A1_k_mean=a1_seq.mean().detach(),
            A1_k_abs=a1_seq.abs().mean().detach(),
            rmse_cost=m1_logs["mse_cost"].sqrt(),
            copy_rmse_cost=copy_logs["mse_cost"].sqrt(),
            rmse_hp=m1_logs["mse_hp"].sqrt(),
            copy_rmse_hp=copy_logs["mse_hp"].sqrt(),
            rmse_count=m1_logs["mse_count"].sqrt(),
            copy_rmse_count=copy_logs["mse_count"].sqrt(),
            rmse_minerals=m1_logs["mse_minerals"].sqrt(),
            copy_rmse_minerals=copy_logs["mse_minerals"].sqrt(),
            loss_field=m1_logs["loss_field"],
            copy_loss_field=copy_logs["loss_field"],
        )
        return loss, logs

    @torch.no_grad()
    def rollout_observer(
        self,
        prefix_seq: dict[str, torch.Tensor],
        static_map: dict[str, torch.Tensor] | None = None,
        n_branches: int = 32,
        horizon_steps: int = 24,
        flow_steps: int = 4,
        temperature: float = 1.0,
        grid_h: int = 64,
        grid_w: int = 64,
        seed: int = 12345,
        mode: str = "forecast",
        cfg_scale: float = 0.85,
        grad_step: float = 0.0,
        n_cp_candidates: int = 3,
    ) -> dict[str, torch.Tensor]:
        """Rolls out `N = n_branches` stochastic `5.0s` futures over `horizon_steps` (`24 -> +120s`)
        using **Common Random Numbers (`seed`)**, returning:
          - Purely learned omniscient positional evaluation (`A0`, `prob_lead_now_p1`, `A_traj`),
          - Intrinsic learned leverage (`squad_leverage`, `leverage_grid_p1/p2`),
          - Theory-of-Mind Plan Misalignment & unscouted threat attribution,
          - Representative step-0 `(u_step0, xi_world_step0, s_step1)` for `L1ChunkFlowDecoder`!
        """
        dev = prefix_seq["q_squad"].device
        q_sq = prefix_seq["q_squad"].float()
        z_sq = prefix_seq["z_squad"].float()
        gate = (prefix_seq["gate"].float() if "gate" in prefix_seq else prefix_seq["active"].float()).clamp(0.0, 1.0)
        active = gate > 0.02
        q_mac = prefix_seq["q_macro"].float()
        z_mac = prefix_seq["z_macro"].float()
        tempo = prefix_seq["tempo_feat"].float()
        _, T0, _, M, _ = q_sq.shape
        z_omni = (
            prefix_seq["z_omni_scene"].float()
            if "z_omni_scene" in prefix_seq
            else torch.zeros(1, T0, self.d_squad, device=dev)
        )

        # Strictly separated Common Random Number streams so Forecast & Counterplay share
        # identical opponent policy noise and world-outcome combat noise across branches & scrubs!
        gen_mis = torch.Generator(device=dev)
        gen_mis.manual_seed(int(seed))
        gen_u_p1 = torch.Generator(device=dev)
        gen_u_p1.manual_seed(int(seed) + 101)
        gen_u_p2 = torch.Generator(device=dev)
        gen_u_p2.manual_seed(int(seed) + 202)
        gen_xi = torch.Generator(device=dev)
        gen_xi.manual_seed(int(seed) + 303)
        gen_cp = torch.Generator(device=dev)
        gen_cp.manual_seed(int(seed) + 7919)

        counterplay_p = None
        if mode == "counterplay_p1":
            counterplay_p = 0
        elif mode == "counterplay_p2":
            counterplay_p = 1

        # 1. Intrinsic Learned Leverage + initial A0 in ONE differentiable pass
        lev_info = self.compute_intrinsic_leverage(
            dict(
                q_squad=q_sq[:, -1],
                z_squad=z_sq[:, -1],
                gate=gate[:, -1],
                q_macro=q_mac[:, -1],
                z_macro=z_mac[:, -1],
            ),
            static_map=static_map,
            grid_h=grid_h,
            grid_w=grid_w,
        )
        a0_res = lev_info["A0"]  # scalar in resource-equivalents

        # 2. Observed prefix encoding (0.000000e+00 leak on observed frames!)
        vis_pref, shim_pref, _ = self.sight.compute_visibility(q_sq, z_sq, gate)
        exp_pref = vis_pref.detach().cummax(dim=1).values
        exp_cur = exp_pref[:, -1]

        if "q_opp_obs" in prefix_seq:
            h_opp_pref = self.persp_enc.summarize_observed_opp(
                prefix_seq["q_opp_obs"], prefix_seq["z_opp_obs"], prefix_seq["gate_opp_obs"]
            )
        else:
            h_opp_pref = self.persp_enc.summarize_rollout_opp(
                q_sq, z_sq, gate, vis_pref, shim_pref, exp_pref
            )

        tok_pref, cap_pref = self.persp_enc.summarize_perspective_frame(
            q_sq, z_sq, gate, q_mac, z_mac, tempo, h_opp_pref
        )
        h_pref_seq, c_pref_seq = self.persp_enc.forward_sequence(tok_pref, cap_pref)
        obs_tok_pref = self.obs_enc.summarize_frame(
            q_sq, z_sq, gate, q_mac, z_mac, z_omni, h_pref_seq
        )

        # 3. Theory-of-Mind Plan Misalignment & Unscouted Threat Attribution at t = T0 - 1
        mis_info = self.compute_plan_misalignment(
            prefix_seq, h_pref_seq[:, -1], c_pref_seq[:, -1], generator=gen_mis
        )

        # In Counterplay mode, grant ONLY `counterplay_p` full-attention awareness at t = T0 - 1
        # (while opponent `1 - counterplay_p` remains strictly fog-bound!)
        tok_start = tok_pref
        cap_start = cap_pref
        exp_start = exp_cur.clone()
        if counterplay_p is not None:
            opp_gate_last = gate[:, -1].flip(dims=[1]).detach()
            h_opp_omni_last = self.persp_enc.summarize_rollout_opp(
                q_sq[:, -1], z_sq[:, -1], gate[:, -1], opp_gate_last, torch.zeros_like(opp_gate_last), opp_gate_last
            )
            h_opp_cp = h_opp_pref.clone()
            h_opp_cp[:, -1, counterplay_p] = h_opp_omni_last[:, counterplay_p]
            tok_start, cap_start = self.persp_enc.summarize_perspective_frame(
                q_sq, z_sq, gate, q_mac, z_mac, tempo, h_opp_cp
            )
            exp_start[:, counterplay_p] = torch.maximum(exp_start[:, counterplay_p], opp_gate_last[:, counterplay_p])

        # 4. Batched Full-Horizon (+120s) Rollout with On-Manifold Trajectory Shooting in Counterplay
        N = n_branches
        K_shoot = max(1, int(n_cp_candidates)) if counterplay_p is not None else 1
        B_roll = K_shoot * N

        tok_hist = tok_start.expand(B_roll, -1, -1, -1).clone()
        cap_hist = cap_start.expand(B_roll, -1, -1, -1).clone()
        obs_hist = obs_tok_pref.expand(B_roll, -1, -1).clone()
        exp_n = exp_start.expand(B_roll, -1, -1).clone()

        cur_s = dict(
            q_squad=q_sq[:, -1].expand(B_roll, -1, -1, -1).clone(),
            z_squad=z_sq[:, -1].expand(B_roll, -1, -1, -1).clone(),
            gate=gate[:, -1].expand(B_roll, -1, -1).clone(),
            active=active[:, -1].expand(B_roll, -1, -1).clone(),
            q_macro=q_mac[:, -1].expand(B_roll, -1, -1).clone(),
            z_macro=z_mac[:, -1].expand(B_roll, -1, -1).clone(),
        )

        a_steps = [(a0_res / 1000.0).expand(B_roll)]
        step0_u_all = None
        step0_xi_all = None
        step0_s1_all = None
        d_u = self.flow_prior.d_u
        d_xi_w = self.xi_prior.d_xi_world

        for step_k in range(horizon_steps):
            h_seq, c_seq = self.persp_enc.forward_sequence(tok_hist, cap_hist)
            h_last = h_seq[:, -1]
            c_last = c_seq[:, -1]
            h_obs_last = self.obs_enc.forward_sequence(obs_hist)[:, -1]

            eps_p1 = torch.randn((N, d_u), generator=gen_u_p1, device=dev, dtype=torch.float32)
            eps_p2 = torch.randn((N, d_u), generator=gen_u_p2, device=dev, dtype=torch.float32)
            eps_xi = torch.randn((N, d_xi_w), generator=gen_xi, device=dev, dtype=torch.float32)

            if K_shoot == 1:
                u_noise = torch.stack([eps_p1, eps_p2], dim=1)       # [N, 2, d_u]
                xi_noise = eps_xi                                    # [N, d_xi_w]
            else:
                u_base = torch.stack([eps_p1, eps_p2], dim=1)        # [N, 2, d_u]
                u_kn = u_base.unsqueeze(0).repeat(K_shoot, 1, 1, 1)  # [K_shoot, N, 2, d_u]
                # Candidates 1..K_shoot-1 explore diverse on-manifold trajectories for counterplay_p
                # against the EXACT SAME opponent policy & world combat noise!
                eps_alt = torch.randn(
                    (K_shoot - 1, N, d_u), generator=gen_cp, device=dev, dtype=torch.float32
                ) * 1.08
                u_kn[1:, :, counterplay_p, :] = eps_alt
                u_noise = u_kn.reshape(B_roll, 2, d_u)
                xi_noise = eps_xi.repeat(K_shoot, 1)

            u_samp, tempo_next = self.flow_prior.sample(
                h_last,
                c_last,
                steps=flow_steps,
                temperature=temperature,
                counterplay_player=counterplay_p,
                cfg_scale=cfg_scale,
                init_noise=u_noise,
            )

            xi_w_samp = self.xi_prior.sample(
                cur_s["q_squad"],
                cur_s["z_squad"],
                cur_s["gate"],
                cur_s["q_macro"],
                cur_s["z_macro"],
                u_samp,
                h_obs_last,
                steps=flow_steps,
                temperature=temperature,
                init_noise=xi_noise,
            )

            cur_s, _ = self.ph.step_dynamics(
                cur_s, u_samp.float(), xi_world=xi_w_samp, static_map=static_map
            )

            if step_k == 0:
                step0_u_all = u_samp.float().clone()
                step0_xi_all = xi_w_samp.float().clone()
                step0_s1_all = {k: v.clone() for k, v in cur_s.items()}

            hk = self.ph.evaluate_hamiltonian(
                torch.cat([cur_s["q_squad"], cur_s["z_squad"]], dim=-1),
                torch.cat([cur_s["q_macro"], cur_s["z_macro"]], dim=-1),
                cur_s["gate"],
                static_map=static_map,
            )
            a_steps.append(hk["A1"].float())

            if step_k + 1 < horizon_steps:
                vl_next, sh_next, exp_n = self.sight.compute_visibility(
                    cur_s["q_squad"], cur_s["z_squad"], cur_s["gate"], prev_explored=exp_n
                )
                h_opp_next = self.persp_enc.summarize_rollout_opp(
                    cur_s["q_squad"], cur_s["z_squad"], cur_s["gate"], vl_next, sh_next, exp_n
                )
                tok_next, cap_next = self.persp_enc.summarize_perspective_frame(
                    cur_s["q_squad"],
                    cur_s["z_squad"],
                    cur_s["gate"],
                    cur_s["q_macro"],
                    cur_s["z_macro"],
                    tempo_next,
                    h_opp_next,
                )
                tok_hist = torch.cat([tok_hist[:, -47:], tok_next.unsqueeze(1)], dim=1)
                cap_hist = torch.cat([cap_hist[:, -47:], cap_next.unsqueeze(1)], dim=1)

                h_next_last = self.persp_enc.forward_sequence(tok_hist[:, -8:], cap_hist[:, -8:])[0][:, -1]
                z_omni_zero = torch.zeros(B_roll, self.d_squad, device=dev)
                obs_next = self.obs_enc.summarize_frame(
                    cur_s["q_squad"],
                    cur_s["z_squad"],
                    cur_s["gate"],
                    cur_s["q_macro"],
                    cur_s["z_macro"],
                    z_omni_zero,
                    h_next_last,
                )
                obs_hist = torch.cat([obs_hist[:, -47:], obs_next.unsqueeze(1)], dim=1)

        a_traj_all = torch.stack(a_steps, dim=1) * 1000.0  # [B_roll, H + 1]
        if K_shoot > 1:
            a_kn = a_traj_all.view(K_shoot, N, horizon_steps + 1)
            h_short = min(4, horizon_steps)
            h_mid = min(12, horizon_steps)
            j_score = 0.15 * a_kn[:, :, h_short] + 0.35 * a_kn[:, :, h_mid] + 0.50 * a_kn[:, :, -1]
            if counterplay_p == 1:
                j_score = -j_score
            best_k = j_score.argmax(dim=0)  # [N]
            sel_flat = best_k * N + torch.arange(N, device=dev)
            a_traj = a_traj_all[sel_flat]
            cur_s = {k: v[sel_flat] for k, v in cur_s.items()}
            if step0_u_all is not None:
                step0_u_all = step0_u_all[sel_flat]
                step0_xi_all = step0_xi_all[sel_flat]
                step0_s1_all = {k: v[sel_flat] for k, v in step0_s1_all.items()}
        else:
            a_traj = a_traj_all

        a_end = a_traj[:, -1]
        drift = a_end - a0_res

        # Select representative branch for L1ChunkFlowDecoder playback:
        # median outcome branch in Forecast mode, or 85th-percentile multi-horizon return in Counterplay mode!
        h_short = min(4, horizon_steps)
        h_mid = min(12, horizon_steps)
        rep_score = 0.15 * a_traj[:, h_short] + 0.35 * a_traj[:, h_mid] + 0.50 * a_end
        if counterplay_p == 0:
            rep_idx = int(torch.argsort(rep_score)[int(0.85 * (N - 1))].item())
        elif counterplay_p == 1:
            rep_idx = int(torch.argsort(-rep_score)[int(0.85 * (N - 1))].item())
        else:
            rep_idx = int(torch.argsort(a_end)[N // 2].item())

        s1_rep = (
            {k: v[rep_idx : rep_idx + 1] for k, v in step0_s1_all.items()}
            if step0_s1_all is not None
            else None
        )
        u0_rep = step0_u_all[rep_idx : rep_idx + 1] if step0_u_all is not None else None
        xi0_rep = step0_xi_all[rep_idx : rep_idx + 1] if step0_xi_all is not None else None

        #Omniscient positional lead probability sigmoid(A0 / 1500) + terminal stronger-hand fraction
        prob_lead_now_p1 = torch.sigmoid(a0_res / 1500.0)
        prob_improve_p1 = (drift > 0.0).float().mean()
        prob_stronger_p1 = (a_end > 0.0).float().mean()
        expected_drift = drift.mean()
        std_drift = drift.std(unbiased=False)
        k_cvar = max(1, N // 10)
        cvar10_p1 = torch.topk(drift, k=k_cvar, largest=False).values.mean()

        end_total_cost = cur_s["q_squad"][:, :, :, SP.SQ["cost"]].sum(dim=-1).mean(dim=0)
        end_minerals = cur_s["q_macro"][:, :, 0].mean(dim=0)

        return dict(
            A0=a0_res,
            H1_0=lev_info["H1"],
            H2_0=lev_info["H2"],
            prob_lead_now_p1=prob_lead_now_p1,
            A_traj=a_traj,
            expected_drift=expected_drift,
            prob_improve_p1=prob_improve_p1,
            prob_stronger_hand_p1=prob_stronger_p1,
            std_drift=std_drift,
            cvar10_p1=cvar10_p1,
            capability_p1=c_pref_seq[0, -1, 0].float(),
            capability_p2=c_pref_seq[0, -1, 1].float(),
            end_total_cost=end_total_cost,
            end_minerals=end_minerals,
            # Intrinsic learned leverage (backwards-aliased to squad_risk / threat_grid_* too)
            squad_leverage=lev_info["squad_leverage"],
            squad_risk=lev_info["squad_leverage"],
            squad_pos=q_sq[0, -1, :, :, 0:2],
            squad_gate=gate[0, -1],
            squad_active=active[0, -1],
            leverage_grid_p1=lev_info["leverage_grid_p1"],
            leverage_grid_p2=lev_info["leverage_grid_p2"],
            threat_grid_p1=lev_info["leverage_grid_p1"],
            threat_grid_p2=lev_info["leverage_grid_p2"],
            # Theory-of-Mind Plan Misalignment & Unscouted Threats
            plan_misalignment=mis_info["plan_misalignment"],
            unscouted_attribution=mis_info["unscouted_attribution"],
            unscouted_mask=mis_info["unscouted_mask"],
            # Step-0 representative L2 transition for L1ChunkFlowDecoder 16-frame micro playback
            rep_branch_idx=rep_idx,
            step0_u=u0_rep,
            step0_xi_world=xi0_rep,
            step0_s1=s1_rep,
        )
