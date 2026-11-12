"""Layer 2 (5.0s Stride) Multi-Horizon Stochastic Observer World Model.

Zero Gram matrices. Operates directly over Layer 1's `1` Macro Token + `M = 64` Squad
Tokens per player:
  1. `StaticTerrainStencil`: Evaluates static terrain height, pathability, and
     buildability stencils continuously at moving Squad & Intent `(x, y)` coordinates
     during multi-step rollouts so ground squads respect cliffs/chokes while air squads
     cross dead airspace.
  2. `PerspectiveMacroEncoder`: Builds two strictly causal, fog-of-war-masked player
     perspective streams `h_{<= tau}^{(1)}` and `h_{<= tau}^{(2)}` (fixing the omniscient
     cross-player leak in `CausalMacroBlock`) plus an online causal Capability Latent
     `c_{p, tau}` inferred from each player's camera/hotkey/command tempo.
  3. `ConditionalFlowPrior`: Generates multi-modal 5-second player innovations
     `U_{tau, p} = (Z_{intent}, xi)` via Conditional Flow Matching trained with
     per-step Diffusion Forcing noise schedules.
  4. `SquadPortHamiltonian`: Evaluates player energies `(H_1, H_2)` and advantage
     `A_1 = H_1 - H_2` in resource-equivalent units (1.0 == 1,000 minerals+gas) via
     2D-RoPE cross-attention, and steps Squad + Macro states forward through structured
     `(J - R) grad H + B(S) U` dynamics with dynamic visibility updates.
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
        # 4 static channels x 5 stencil points + 4 * n_freqs Fourier features
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
        """`pos_norm`: `[N_outer, ..., N, 2]` -> `[N_outer, ..., N, d_out]`.
        Automatically broadcasts across `N_outer = B_map * mult` without materializing
        redundant copies of `static_grid` or triggering host-device syncs."""
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
                st = static_map["static_grid_chw"]  # [B_map, 4, gh, gw]
            else:
                st = static_map["static_grid"].float().permute(0, 3, 1, 2)
            B_map = st.shape[0]
            pos_flat = pos_norm.reshape(B_map, -1, 2).float() * E.MAP_SCALE  # [B_map, K_all, 2]
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

        ang = pos_flat.unsqueeze(-1) * self.omega  # [B_map, K_all, 2, n_freqs]
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
    if "static_grid_chw" in maps_list[0]:
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
    grid = b["grid"][:, f_end].float()  # [B, gh, gw, C_LIVE]
    gh, gw = grid.shape[1], grid.shape[2]
    off = (b["center"].float() - b["origin"].float())[:, None, :]  # [B, 1, 2]
    wh = torch.tensor([gw, gh], device=dev, dtype=torch.float32)

    out = []
    for p_i in (0, 1):
        opp = 1 - p_i
        opp_pos_map = q_squad[:, opp, :, 0:2].float() * E.MAP_SCALE  # [B, M, 2]
        at = (((opp_pos_map + off) / wh) * 2.0 - 1.0).unsqueeze(2)   # [B, M, 1, 2]
        vis_ch = grid[..., p_i * 5 + 0].unsqueeze(1)                 # [B, 1, gh, gw]
        v = F.grid_sample(vis_ch, at, align_corners=False).view(B, M)
        out.append(v.clamp(0.0, 1.0))
    return torch.stack(out, dim=1)


class SightFootprintHead(nn.Module):
    """Predicts each active squad's effective sight radius (in normalized `/ 100.0` map
    units) so `L2` can dynamically update fog-of-war visibility along multi-step rollouts
    without engine raster grids."""

    def __init__(self, d_squad: int = SP.D_SQUAD_DEFAULT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_squad + SP.N_SQUAD_Q),
            nn.Linear(d_squad + SP.N_SQUAD_Q, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        E.small_init(self.net[-1], std=1e-2)

    def sight_radius(self, q_squad: torch.Tensor, z_squad: torch.Tensor) -> torch.Tensor:
        """Returns sight radius in normalized map units (`0.09 == 9` map units baseline +
        squad spatial spread + learned unit-type sight bonus)."""
        spread_r = (
            q_squad[..., SP.SQ["spread_x"]].square()
            + q_squad[..., SP.SQ["spread_y"]].square()
            + 1e-6
        ).sqrt()
        delta = 0.06 * torch.tanh(self.net(torch.cat([q_squad, z_squad], dim=-1)).squeeze(-1))
        # Typical SC2 sight range is 8-14 map units (0.08 - 0.14 normalized) + cluster spread
        return (0.10 + spread_r + delta).clamp(0.05, 0.35)

    def compute_visibility(
        self,
        q_squad: torch.Tensor,
        z_squad: torch.Tensor,
        active: torch.Tensor,
        prev_explored: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes pairwise soft live visibility `vis_live` (`[..., 2, M]` in `[0, 1]`,
        where `vis_live[..., p, m]` is whether player `p` currently sees enemy `1-p`'s
        squad `m`) and monotonic `explored` (`[..., 2, M]`)."""
        r_sight = self.sight_radius(q_squad, z_squad)  # [..., 2, M]
        pos = q_squad[..., 0:2]                        # [..., 2, M, 2]

        live_list = []
        for p_i in (0, 1):
            opp = 1 - p_i
            my_pos = pos[..., p_i, :, :]          # [..., M_my, 2]
            opp_pos = pos[..., opp, :, :]         # [..., M_opp, 2]
            my_r = r_sight[..., p_i, :]           # [..., M_my]
            my_act = active[..., p_i, :]          # [..., M_my]

            # Smooth epsilon-regularized distance matrix [..., M_my, M_opp] (avoids sqrt(0) NaN grads)
            dist = ((my_pos.unsqueeze(-2) - opp_pos.unsqueeze(-3)).square().sum(dim=-1) + 1e-6).sqrt()
            # Soft visibility margin (sharpness 0.015 == 1.5 map units)
            margin = (my_r.unsqueeze(-1) - dist) / 0.015
            margin = margin.masked_fill(~my_act.unsqueeze(-1), -20.0)
            max_margin = margin.max(dim=-2).values  # [..., M_opp]
            v_live = torch.sigmoid(max_margin) * active[..., opp, :].float()
            live_list.append(v_live)

        vis_live = torch.stack(live_list, dim=-2)  # [..., 2, M]
        if prev_explored is None:
            explored = vis_live.detach()
        else:
            explored = torch.maximum(prev_explored, vis_live.detach())
        return vis_live, explored


def mask_enemy_for_perspective(
    q_opp: torch.Tensor,
    z_opp: torch.Tensor,
    act_opp: torch.Tensor,
    vis_live_p: torch.Tensor,
    explored_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applies StarCraft II fog-of-war rules to enemy (`1 - p`) Squad Tokens for player `p`'s
    Strategy Prior:
      - `vis_live_p > 0.5`: full live squad (`q_opp`, `z_opp`) is visible.
      - `vis_live_p <= 0.5` & `explored_p > 0.5` & `struct_frac > 0.5`: structure snapshot
        remains visible at `(x, y)` with velocity zeroed and live `z_opp` attenuated.
      - Mobile units in fog (`vis_live_p <= 0.5` & `struct_frac <= 0.5`): completely hidden!
    """
    is_struct = q_opp[..., SP.SQ["struct_frac"]] > 0.5
    snap_w = ((explored_p > 0.3).float() * is_struct.float()) * (1.0 - vis_live_p)
    vis_w = (vis_live_p + snap_w).clamp(0.0, 1.0)
    vis_act = act_opp & (vis_w > 0.15)

    # Zero enemy velocities when only fog-snapshot visible (strictly out-of-place for autograd)
    q_masked = torch.cat(
        [
            q_opp[..., 0:2],
            q_opp[..., 2:4] * vis_live_p.unsqueeze(-1),
            q_opp[..., 4:],
        ],
        dim=-1,
    ) * vis_w.unsqueeze(-1)
    z_masked = z_opp * vis_live_p.unsqueeze(-1)
    return q_masked, z_masked, vis_act


class PerspectiveMacroEncoder(nn.Module):
    """Encodes each player's strictly causal, partial-information perspective history
    `h_{<= tau}^{(p)}` (`[B, T, 2, d]`) and online Capability Latent `c_{p, tau}` (`[B, T, 2, d_cap]`),
    completely eliminating the omniscient cross-player leak in `CausalMacroBlock`."""

    def __init__(
        self,
        d_squad: int = SP.D_SQUAD_DEFAULT,
        d_model: int = 768,
        d_cap: int = 16,
        layers: int = 6,
        heads: int = 12,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_cap = d_cap
        self.heads = heads
        self.dh = d_model // heads

        sq_dim = d_squad + SP.N_SQUAD_Q
        mac_dim = d_squad + SP.N_MACRO_Q
        self.own_sq_proj = nn.Linear(sq_dim, d_model)
        self.opp_sq_proj = nn.Linear(sq_dim + 2, d_model)  # + (vis_live, explored)
        self.step_in_proj = nn.Sequential(
            nn.LayerNorm(2 * d_model + mac_dim + SP.D_TEMPO_FEAT),
            nn.Linear(2 * d_model + mac_dim + SP.D_TEMPO_FEAT, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Causal temporal Transformer blocks (per-player isolated stream: NO omniscient cross-MLP!)
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

        # Online causal Capability Latent c_{p, tau} from observed tempo/camera/hotkey stream
        self.cap_in = nn.Sequential(
            nn.LayerNorm(SP.D_TEMPO_FEAT + SP.N_MACRO_Q),
            nn.Linear(SP.D_TEMPO_FEAT + SP.N_MACRO_Q, 128),
            nn.SiLU(),
            nn.Linear(128, d_cap),
        )
        inv_freq = 1.0 / (1000.0 ** (torch.arange(0, self.dh, 2, dtype=torch.float32) / self.dh))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def summarize_perspective_frame(
        self,
        q_squad: torch.Tensor,
        z_squad: torch.Tensor,
        active: torch.Tensor,
        q_macro: torch.Tensor,
        z_macro: torch.Tensor,
        tempo_feat: torch.Tensor,
        vis_live: torch.Tensor,
        explored: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Summarizes one or more frames `[..., 2, ...]` into per-player perspective token
        `tok_p` `[..., 2, d_model]` and instantaneous capability observation `cap_inst` `[..., 2, d_cap]`
        in a single vectorized pass across both players."""
        act_own = active.float()
        denom_own = act_own.sum(dim=-1, keepdim=True).clamp(min=1.0)
        sq_own_in = torch.cat([q_squad, z_squad], dim=-1)
        h_own_both = (self.own_sq_proj(sq_own_in) * act_own.unsqueeze(-1)).sum(dim=-2) / denom_own

        q_m, z_m, act_m = mask_enemy_for_perspective(
            q_squad.flip(dims=[-3]),
            z_squad.flip(dims=[-3]),
            active.flip(dims=[-2]),
            vis_live,
            explored,
        )
        act_opp_f = act_m.float()
        denom_opp = act_opp_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        sq_opp_in = torch.cat(
            [q_m, z_m, vis_live.unsqueeze(-1), explored.unsqueeze(-1)],
            dim=-1,
        )
        h_opp_both = (self.opp_sq_proj(sq_opp_in) * act_opp_f.unsqueeze(-1)).sum(dim=-2) / denom_opp

        mac_both = torch.cat([q_macro, z_macro], dim=-1)
        tok = self.step_in_proj(torch.cat([h_own_both, h_opp_both, mac_both, tempo_feat], dim=-1))
        cap_inst = torch.tanh(self.cap_in(torch.cat([tempo_feat, q_macro], dim=-1)))
        return tok, cap_inst

    def forward_sequence(
        self,
        tok_seq: torch.Tensor,
        cap_inst_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs causal attention along `T` (`tok_seq`: `[B, T, 2, d_model]`) with strictly
        isolated player streams (no omniscient cross-player leak) and causal EMA pooling
        for capability `c_p` (`[B, T, 2, d_cap]`)."""
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

        h_out = self.out_norm(x).view(B, P, T, D).transpose(1, 2)  # [B, T, 2, D]

        # Vectorized causal EMA along T (alpha = 0.15) via lower-triangular Toeplitz matmul (zero Python loop!)
        idx = torch.arange(T, device=x.device, dtype=torch.float32)
        diff = (idx[:, None] - idx[None, :]).clamp(min=0.0)
        decay = 0.85 ** diff
        ema_w = torch.tril(torch.where(idx[None, :] == 0, decay, 0.15 * decay))  # [T, T]
        c_flat = cap_inst_seq.permute(0, 2, 1, 3).reshape(B * P, T, self.d_cap)
        c_seq = torch.matmul(ema_w.to(c_flat.dtype), c_flat).view(B, P, T, self.d_cap).transpose(1, 2)
        return h_out, c_seq


class _TensorCoreResBlock(nn.Module):
    """High-arithmetic-intensity bf16 Tensor Core residual MLP block (zero inner LayerNorm)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        E.small_init(self.fc2, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class ConditionalFlowPrior(nn.Module):
    """Conditional Flow-Matching Strategy Prior `v_theta(u_s, s | h_{<= tau}^{(p)}, c_{p, tau})`
    over player `p`'s 5-second innovation `U_{tau, p} = (Z_{intent}, xi)`.

    Linear optimal-transport path from noise `s = 1` (`eps ~ N(0, I)`) to clean Posterior
    intent `s = 0` (`u_0`):
        u_s = (1 - s) * u_0 + s * eps
        du_s / ds = eps - u_0   (denoising velocity toward u_0 is `v = u_0 - eps`).
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

        in_dim = d_u + d_model + d_cap + 32
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

        # Predicts next-step expected tempo_feat during open-loop observer rollouts
        self.tempo_pred = nn.Sequential(
            nn.LayerNorm(d_model + d_cap + d_u),
            nn.Linear(d_model + d_cap + d_u, 256),
            nn.SiLU(),
            nn.Linear(256, SP.D_TEMPO_FEAT),
        )

    def _embed_s(self, s: torch.Tensor) -> torch.Tensor:
        ang = s.unsqueeze(-1) * self.s_freqs
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def velocity(
        self,
        u_s: torch.Tensor,
        s: torch.Tensor,
        h_p: torch.Tensor,
        c_p: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts flow velocity `u_0 - eps` pointing from `s=1` noise toward `s=0` clean intent."""
        s_emb = self._embed_s(s)
        return self.net(torch.cat([u_s, h_p, c_p, s_emb], dim=-1))

    def flow_matching_loss(
        self,
        u_target: torch.Tensor,
        h_p: torch.Tensor,
        c_p: torch.Tensor,
        s: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes Conditional Flow-Matching MSE loss and returns `(loss, u_noisy, s)`
        so Diffusion-Forcing rollouts can also step physics from partially denoised intents."""
        u_0 = u_target.detach()
        eps = torch.randn_like(u_0)
        if s is None:
            s = torch.rand(u_0.shape[:-1], device=u_0.device, dtype=u_0.dtype)
        u_s = (1.0 - s.unsqueeze(-1)) * u_0 + s.unsqueeze(-1) * eps
        v_pred = self.velocity(u_s, s, h_p, c_p)
        v_true = u_0 - eps
        loss = F.mse_loss(v_pred, v_true)
        # 1-step denoised estimate u_hat_0 = u_s + s * v_pred (exact when v_pred == u_0 - eps)
        u_denoised = u_s + s.unsqueeze(-1) * v_pred
        return loss, u_denoised, s

    def sample(
        self,
        h_p: torch.Tensor,
        c_p: torch.Tensor,
        steps: int = 4,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrates from `s = 1.0` Gaussian noise down to `s = 0.0` in `steps` midpoint
        steps, returning `(u_sampled, pred_tempo_feat)`."""
        u = torch.randn(*h_p.shape[:-1], self.d_u, device=h_p.device, dtype=h_p.dtype) * temperature
        dt = 1.0 / max(1, steps)
        for i in range(steps):
            s_cur = torch.full(h_p.shape[:-1], 1.0 - i * dt, device=h_p.device, dtype=h_p.dtype)
            v1 = self.velocity(u, s_cur, h_p, c_p)
            u_mid = u + 0.5 * dt * v1
            s_mid = torch.full(h_p.shape[:-1], 1.0 - (i + 0.5) * dt, device=h_p.device, dtype=h_p.dtype)
            v2 = self.velocity(u_mid, s_mid, h_p, c_p)
            u = u + dt * v2
        tempo_next = self.tempo_pred(torch.cat([h_p, c_p, u], dim=-1))
        return u, tempo_next


def _soft_cap_frob(mat: torch.Tensor, max_norm: float, dims: tuple[int, ...]) -> torch.Tensor:
    """Smoothly bounds Frobenius norm over `dims` strictly below `max_norm` while
    preserving exact identity slope (`scale -> 1.0`) near zero."""
    nrm = (mat.square().sum(dim=dims, keepdim=True) + 1e-12).sqrt()
    return mat * (max_norm * torch.tanh(nrm / max_norm) / nrm)


def _port_combine_and_project(
    sq_out: torch.Tensor,
    mac_out: torch.Tensor,
    x_sq_req: torch.Tensor,
    x_mac_req: torch.Tensor,
    active: torch.Tensor,
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
]:
    """Low-bandwidth rank-2 Port-Hamiltonian projection + physical domain clamping.
    Performs the Frobenius norm caps and rank-2 operator contractions in `sq_out.dtype`
    (bf16 under autocast -> 4x lower HBM traffic than rank-4 fp32) and casts state
    increments to fp32 for exact Euler integration."""
    B, _, M, D_s = x_sq_req.shape
    mac_dim = x_mac_req.shape[-1]
    w_dt = sq_out.dtype
    act_f = active.float()
    act_w = active.to(w_dt)
    d2 = D_s * 2

    uvl_raw = sq_out[..., : 3 * d2].view(B, 2, M, 3, D_s, 2)
    c_sq_raw_unscaled = sq_out[..., 3 * d2 : 4 * d2].view(B, 2, M, D_s, 2)
    harv_raw = sq_out[..., 4 * d2 : 4 * d2 + 2].float()
    bu_sq_raw = sq_out[..., 4 * d2 + 2 :].float()

    max_uvl = uvl_raw.new_tensor([0.55, 0.55, 0.75]).view(1, 1, 1, 3, 1, 1)
    nrm_uvl = (uvl_raw.float().square().sum(dim=(-2, -1), keepdim=True) + 1e-8).sqrt()
    scale_uvl = (max_uvl.float() * torch.tanh(nrm_uvl / max_uvl.float()) / nrm_uvl).to(w_dt)
    uvl_cap = uvl_raw * scale_uvl

    g_sq_w = g_sq.to(w_dt)
    proj3 = (uvl_cap * g_sq_w.view(B, 2, M, 1, D_s, 1)).sum(dim=-2)  # [B, 2, M, 3, 2]
    U_j, V_j, L_r = uvl_cap.unbind(dim=-3)
    u_g, v_g, lt_g = proj3.unbind(dim=-2)

    j_sq = (U_j * v_g.unsqueeze(-2) - V_j * u_g.unsqueeze(-2)).sum(dim=-1).float()
    r_sq = (L_r * lt_g.unsqueeze(-2)).sum(dim=-1).float()
    diss_sq = (lt_g.float().square().sum(dim=-1) * act_f).sum(dim=-1)

    adv = advect_gain.float().clamp(0.05, 0.50)
    j_kin = torch.cat(
        [adv * x_sq_req[..., 2:4].float(), torch.zeros_like(j_sq[..., 2:])], dim=-1
    )

    mac_f = mac_out.float()
    md2 = mac_dim * 2
    C_mac = _soft_cap_frob(mac_f[..., :md2].view(B, 2, mac_dim, 2), 0.55, (-2, -1))
    L_rm = _soft_cap_frob(mac_f[..., md2 : 2 * md2].view(B, 2, mac_dim, 2), 0.75, (-2, -1))
    bu_mac_raw = mac_f[..., 2 * md2 :]

    c_scaled = c_sq_raw_unscaled * (0.25 * act_w).unsqueeze(-1).unsqueeze(-1)
    c_nrm = (c_scaled.float().square().sum(dim=(-3, -2, -1), keepdim=True) + 1e-8).sqrt()
    c_scale = (0.55 * torch.tanh(c_nrm / 0.55) / c_nrm).to(w_dt)
    C_sq = c_scaled * c_scale

    g_mac_f = g_mac.float()
    port_from_sq = (C_sq * g_sq_w.unsqueeze(-1)).sum(dim=(2, 3)).float()  # [B, 2, 2]
    port_from_mac = (C_mac * g_mac_f.unsqueeze(-1)).sum(dim=-2)           # [B, 2, 2]
    j_mac_cpl = (C_mac * port_from_sq.unsqueeze(-2)).sum(dim=-1)
    j_sq_cpl = -(C_sq * port_from_mac.to(w_dt)[:, :, None, None, :]).sum(dim=-1).float()

    lt_gm = (L_rm * g_mac_f.unsqueeze(-1)).sum(dim=-2)
    r_mac = (L_rm * lt_gm.unsqueeze(-2)).sum(dim=-1)

    sq_count = x_sq_req[..., SP.SQ["count"]].float().clamp(min=0.0, max=6.0) * act_f
    harv_rate = F.softplus(harv_raw).clamp(max=0.6)
    harv_inflow = (harv_rate * sq_count.unsqueeze(-1)).sum(dim=2)
    harv_mac = torch.cat([harv_inflow, torch.zeros_like(r_mac[..., 2:])], dim=-1)

    bu_sq = (1.5 * torch.tanh(bu_sq_raw / 1.5)) * b_mask_sq
    bu_mac = (1.5 * torch.tanh(bu_mac_raw / 1.5)) * b_mask_mac

    act_f_col = act_f.unsqueeze(-1)
    dx_sq = ((j_sq + j_kin + j_sq_cpl - r_sq) + bu_sq) * act_f_col
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
            q_sq_raw[..., 10:12].clamp(0.0, 1.0),
        ],
        dim=-1,
    )

    survives = active & (q_sq_n[..., SP.SQ["count"]] > 5e-4) & (
        (q_sq_n[..., SP.SQ["hp"]] + q_sq_n[..., SP.SQ["cost"]]) > 5e-4
    )
    surv_f = survives.float().unsqueeze(-1)
    q_sq_n = q_sq_n * surv_f
    z_sq_n = (x_sq_next_raw[..., SP.N_SQUAD_Q :].clamp(-4.0, 4.0)) * surv_f

    q_mac_raw = x_mac_next_raw[..., : SP.N_MACRO_Q]
    q_mac_n = torch.cat(
        [
            q_mac_raw[..., 0:4].clamp(0.0, 20.0),
            q_mac_raw[..., 4:6].clamp(-2.0, 2.0),
        ],
        dim=-1,
    )
    z_mac_n = x_mac_next_raw[..., SP.N_MACRO_Q :].clamp(-3.0, 3.0)
    return q_sq_n, z_sq_n, survives, q_mac_n, z_mac_n, bu_sq, bu_mac, diss_sq


class SquadPortHamiltonian(nn.Module):
    """Token-level Port-Hamiltonian World Model over `1` Macro + `M = 64` Squad Tokens per player.

    Energies `H_1, H_2` and advantage `A_1 = H_1 - H_2` are anchored in resource-equivalent
    units (`1.0 == 1,000` minerals+vespene, so `A_1 * 1000` is directly interpretable in
    mineral+gas value equivalents without arbitrary calibration constants).
    """

    def __init__(
        self,
        d_squad: int = SP.D_SQUAD_DEFAULT,
        n_squads: int = SP.N_SQUADS_DEFAULT,
        k_intent: int = SP.K_INTENT_DEFAULT,
        d_intent: int = SP.D_INTENT_DEFAULT,
        d_xi: int = SP.D_XI_DEFAULT,
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
        self.hidden = hidden
        self.d_terr = d_terr
        self.d_tug = d_tug
        self.d_ctrl = d_ctrl
        self.tug_heads = d_tug // 64
        self.ctrl_heads = d_ctrl // 64
        self.rank = rank

        self.sq_dim = SP.N_SQUAD_Q + d_squad   # 140
        self.mac_dim = SP.N_MACRO_Q + d_squad  # 134

        # Mask kinematic coordinates (x, y, vx, vy) out of intrinsic stored potential `own_p`
        w_adv_sq = torch.ones(self.sq_dim, dtype=torch.float32)
        w_adv_sq[0:4] = 0.0
        w_adv_mac = torch.ones(self.mac_dim, dtype=torch.float32)
        w_adv_mac[4:6] = 0.0  # mask camera (cam_x, cam_y) out of intrinsic economic potential
        self.register_buffer("w_adv_sq", w_adv_sq)
        self.register_buffer("w_kin_sq", 1.0 - w_adv_sq)
        self.register_buffer("w_adv_mac", w_adv_mac)

        # Control port mask `b_mask`: blocks teleportation of (x, y, count, cost, hp, shield)
        # and (minerals, vespene, food_used, food_cap)! All movement, economy, and combat
        # damage MUST flow through `(J - R) grad H`.
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

        # 1. Squad & Macro Hamiltonian H(S_1, S_2) (compact pot_hid=512 keeps step-1 create_graph=True lean)
        pot_hid = 512
        self.sq_pot_net = nn.Sequential(
            nn.Linear(self.sq_dim, pot_hid, bias=False),
            nn.SiLU(),
            nn.Linear(pot_hid, pot_hid, bias=False),
            nn.SiLU(),
            nn.Linear(pot_hid, 1),
        )
        E.small_init(self.sq_pot_net[-1], std=1e-2)
        self.mac_pot_net = nn.Sequential(
            nn.Linear(self.mac_dim, pot_hid, bias=False),
            nn.SiLU(),
            nn.Linear(pot_hid, 1),
        )
        E.small_init(self.mac_pot_net[-1], std=1e-2)

        # Multi-head 2D-RoPE Cross-Player Tug-of-War Interaction (fused QKV GEMM)
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

        # Squad Kinetic Energy T_p >= 0
        self.kin_net = nn.Sequential(
            nn.Linear(self.sq_dim + d_terr, pot_hid, bias=False),
            nn.SiLU(),
            nn.Linear(pot_hid, 2),
        )
        E.small_init(self.kin_net[-1], std=1e-2)

        # 2. Multi-head 2D-RoPE Spatial Intent -> Squad routing + Deep 2048-wide Tensor-Core Operator Trunk
        self.ctrl_q = nn.Linear(self.sq_dim + d_terr, d_ctrl, bias=False)
        self.ctrl_kv = nn.Linear(d_intent, 2 * d_ctrl, bias=False)
        self.ctrl_sink = nn.Parameter(torch.full((self.ctrl_heads,), 1.5))

        op_sq_in_dim = self.sq_dim + d_terr + d_tug + d_ctrl + d_intent + d_xi
        op_sq_out_dim = 4 * (self.sq_dim * 2) + 2 + self.sq_dim
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

        op_mac_in_dim = self.mac_dim + d_intent + d_xi
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

        # Learnable advection rate converting 5s velocity gradient into Δ(x, y)
        self.advect_gain = nn.Parameter(torch.tensor(0.25))  # 5.0s * 5 m/s / 100 map_scale == 0.25!
        self._op_step_fn = self._apply_operators_and_step

    def _cross_squad_context(
        self,
        x_sq: torch.Tensor,
        active: torch.Tensor,
        terr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes skew-symmetric `tug` energy `[B]` and multi-head 2D-RoPE enemy interaction
        contexts `opp_ctx` `[B, 2, M, d_tug]` in a single fused `[B * 2, M, ...]` pass."""
        B, _, M, _ = x_sq.shape
        B2 = B * 2
        H = self.tug_heads
        pos_both = x_sq[..., 0:2] * E.MAP_SCALE
        in_both = torch.cat([x_sq * self.w_adv_sq, terr], dim=-1)

        qkv_both = self.tug_qkv(in_both)  # [B, 2, M, 3 * d_tug]
        q_raw, k_raw, v_raw = qkv_both.chunk(3, dim=-1)

        src = in_both.reshape(B2, M, -1)
        pos_s = pos_both.reshape(B2, M, 2)
        pos_d = pos_both.flip(dims=[1]).reshape(B2, M, 2)
        act_s = active.reshape(B2, M)
        act_d = active.flip(dims=[1]).reshape(B2, M)

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
            * act_d[:, None, :, None].to(v_raw.dtype)
        )

        scores = torch.matmul(q_rot, k_rot.transpose(-1, -2).to(q_rot.dtype)) * 6.0  # [B2, H, M, M]
        scores = scores.masked_fill(~act_d[:, None, None, :], -20.0)
        sink = self.tug_sink.view(1, H, 1, 1).expand(B2, H, M, 1).to(scores.dtype)
        w_aug = F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)
        has_dst = act_d.any(dim=-1)[:, None, None, None].to(w_aug.dtype)
        w = w_aug[..., :M] * has_dst
        ctx = (
            torch.matmul(w.to(v_dst.dtype), v_dst)
            .transpose(1, 2)
            .reshape(B2, M, self.d_tug)
            * act_s.to(v_dst.dtype).unsqueeze(-1)
        )
        tug_raw = self.tug_head(torch.cat([src, ctx], dim=-1)).squeeze(-1).float()
        val_both = (2.5 * torch.tanh(tug_raw / 2.5) * act_s.float()).sum(dim=-1).view(B, 2)
        tug = 0.5 * (val_both[:, 0] - val_both[:, 1])
        return tug, ctx.view(B, 2, M, self.d_tug)

    def evaluate_hamiltonian(
        self,
        x_sq: torch.Tensor,
        x_mac: torch.Tensor,
        active: torch.Tensor,
        static_map: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluates `H_1, H_2, A_1` in resource-equivalent (`1,000` mineral+gas) units."""
        terr = self.terrain(x_sq[..., 0:2], static_map)  # [B, 2, M, d_terr]
        act_f = active.float()                           # [B, 2, M]

        # Batched physical value baseline + composition/tech potential across both players [B, 2]
        bank_val = x_mac[:, :, SP.MQ["minerals"]].float() + x_mac[:, :, SP.MQ["vespene"]].float()
        army_econ_val = (
            (
                x_sq[:, :, :, SP.SQ["cost"]].float()
                + 0.15 * (x_sq[:, :, :, SP.SQ["hp"]].float() + x_sq[:, :, :, SP.SQ["shield"]].float())
            )
            * act_f
        ).sum(dim=-1)

        sq_pot_raw = self.sq_pot_net(x_sq * self.w_adv_sq).squeeze(-1).float()
        sq_pot = (3.0 * torch.tanh(sq_pot_raw / 3.0) * act_f).sum(dim=-1)
        mac_pot = 4.0 * torch.tanh(self.mac_pot_net(x_mac * self.w_adv_mac).squeeze(-1).float() / 4.0)
        own = bank_val + army_econ_val + sq_pot + mac_pot  # [B, 2]

        # Batched O(v^2) kinetic energy + exact closed-form d(kin)/dv across both players [B, 2]
        k_w = 2.0 * torch.sigmoid(self.kin_net(torch.cat([x_sq.detach(), terr.detach()], dim=-1)).float())
        vel = x_sq[:, :, :, 2:4].float()
        mass_w = (
            x_sq[:, :, :, SP.SQ["cost"]].detach().float()
            + 0.1 * x_sq[:, :, :, SP.SQ["count"]].detach().float()
        ).clamp(min=0.01, max=6.0)
        kin = 0.5 * ((k_w * vel.square()).sum(dim=-1) * mass_w * act_f).sum(dim=-1)  # [B, 2]
        grad_kin_v = k_w * vel * (mass_w * act_f).unsqueeze(-1)                      # [B, 2, M, 2]

        tug, opp_ctx = self._cross_squad_context(x_sq, active, terr)
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
        active: torch.Tensor,
        g_sq: torch.Tensor,
        g_mac: torch.Tensor,
        terr: torch.Tensor,
        opp_ctx: torch.Tensor,
        u_flat: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Fused Tensor-Core Port-Hamiltonian operator evaluation + Triton-compiled domain projection."""
        u_struct = SP.unpack_innovation(
            u_flat,
            k_intent=self.k_intent,
            d_intent=self.d_intent,
            d_xi=self.d_xi,
        )
        xi = u_struct["xi"]  # [B, 2, d_xi]
        B, _, M, _ = x_sq_req.shape
        B2 = B * 2
        Hc = self.ctrl_heads

        # 1. Multi-head 2D-RoPE Spatial Intent -> Squad Context (with null-intent sink)
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
        c_scores = torch.matmul(cq, ck.transpose(-1, -2).to(cq.dtype)) * 6.0  # [B2, Hc, M, K]
        c_sink = self.ctrl_sink.view(1, Hc, 1, 1).expand(B2, Hc, M, 1).to(c_scores.dtype)
        c_attn = F.softmax(torch.cat([c_scores, c_sink], dim=-1), dim=-1)[..., : self.k_intent]
        int_local = (
            torch.matmul(c_attn.to(cv.dtype), cv)
            .transpose(1, 2)
            .reshape(B, 2, M, self.d_ctrl)
        )
        mac_int_exp = u_struct["intent_macro"].unsqueeze(2).expand(B, 2, M, -1)
        xi_exp = xi.unsqueeze(2).expand(B, 2, M, -1)

        # 2. Deep 1536-wide bf16 Tensor-Core operator trunk + Macro trunk
        op_sq_normed = self.op_sq_norm(
            torch.cat([x_sq_req, terr, opp_ctx, int_local, mac_int_exp, xi_exp], dim=-1)
        )
        sq_out = self.op_sq_trunk(op_sq_normed)

        mac_out = self.op_mac_trunk(
            self.op_mac_norm(torch.cat([x_mac_req, u_struct["intent_macro"], xi], dim=-1))
        )

        # 3. Low-bandwidth rank-2 port combination + physical domain projection
        return _port_combine_and_project(
            sq_out,
            mac_out,
            x_sq_req,
            x_mac_req,
            active,
            g_sq,
            g_mac,
            self.b_mask_sq,
            self.b_mask_mac,
            self.advect_gain,
        )

    def step_dynamics(
        self,
        state: dict[str, torch.Tensor],
        u_flat: torch.Tensor,
        static_map: dict[str, torch.Tensor] | None = None,
        create_graph_h: bool | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Advances `(q_squad, z_squad, active, q_macro, z_macro)` by one 5.0-second macro step
        under `dS = (J - R) grad H + B(S) U`."""
        outer_grad = torch.is_grad_enabled()
        q_sq = state["q_squad"].float()
        z_sq = state["z_squad"].float()
        active = state["active"]
        q_mac = state["q_macro"].float()
        z_mac = state["z_macro"].float()

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
                x_sq_h, x_mac_h, active, static_map=static_map
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

        # Player-oriented advantage gradients (+grad_1 A_1 for P1, +grad_2 (-A_1) for P2)
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
        g_sq = (g_sq_adv + g_kin_full) * active.float().unsqueeze(-1)
        g_mac = g_mac_adv

        if outer_grad:
            q_sq_n, z_sq_n, survives, q_mac_n, z_mac_n, bu_sq, bu_mac, diss_sq = grad_checkpoint(
                self._op_step_fn,
                x_sq_step,
                x_mac_step,
                active,
                g_sq,
                g_mac,
                h_out["terr"],
                h_out["opp_ctx"],
                u_flat,
                use_reentrant=False,
            )
        else:
            q_sq_n, z_sq_n, survives, q_mac_n, z_mac_n, bu_sq, bu_mac, diss_sq = self._op_step_fn(
                x_sq_step,
                x_mac_step,
                active,
                g_sq,
                g_mac,
                h_out["terr"],
                h_out["opp_ctx"],
                u_flat,
            )

        next_state = dict(
            q_squad=q_sq_n,
            z_squad=z_sq_n,
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
    """Permutation- and slot-split-invariant 2D continuous field kernel distance
    `||F_pred - F_true||_K^2` over squad `(cost, hp, shield, count)` distributions at
    8, 22, and 50 map-unit spatial scales (fused multi-scale trace in 3 batched matmuls)."""
    pos_p = pred_s["q_squad"][..., 0:2].float()  # [N, 2, M, 2]
    pos_t = true_s["q_squad"][..., 0:2].float()
    w_ch = pos_p.new_tensor([1.0, 1.0, 0.25, 0.25])
    val_p = pred_s["q_squad"][..., 6:10].float() * w_ch * pred_s["active"].float().unsqueeze(-1)
    val_t = true_s["q_squad"][..., 6:10].float() * w_ch * true_s["active"].float().unsqueeze(-1)

    d_pp = (pos_p.unsqueeze(-2) - pos_p.unsqueeze(-3)).square().sum(dim=-1)  # [N, 2, M, M]
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
    """Complete Multi-Horizon Stochastic Observer combining:
      - `SightFootprintHead` (dynamic fog-of-war visibility)
      - `PerspectiveMacroEncoder` (causal partial-info player streams + capability `c_p`)
      - `ConditionalFlowPrior` (Diffusion-Forcing Flow Matching over intents + micro residual)
      - `SquadPortHamiltonian` (token-level Port-Hamiltonian dynamics + resource-grounded `A_1`)
    """

    def __init__(
        self,
        d_squad: int = SP.D_SQUAD_DEFAULT,
        n_squads: int = SP.N_SQUADS_DEFAULT,
        d_model: int = 768,
        d_cap: int = 16,
        layers: int = 6,
        heads: int = 12,
        rollout_steps: int = 4,
    ):
        super().__init__()
        self.d_squad = d_squad
        self.n_squads = n_squads
        self.d_model = d_model
        self.d_cap = d_cap
        self.rollout_steps = rollout_steps

        self.sight = SightFootprintHead(d_squad=d_squad)
        self.persp_enc = PerspectiveMacroEncoder(
            d_squad=d_squad, d_model=d_model, d_cap=d_cap, layers=layers, heads=heads
        )
        self.flow_prior = ConditionalFlowPrior(
            d_u=SP.innovation_dim(), d_model=d_model, d_cap=d_cap
        )
        self.ph = SquadPortHamiltonian(d_squad=d_squad, n_squads=n_squads)

    def compile_parts(self):
        """Compiles `SightFootprintHead`, `PerspectiveMacroEncoder`, `ConditionalFlowPrior`,
        and multi-scale field kernel into fused Triton + bf16 Tensor Core graphs while letting
        `[B*T*128, D]` Port-Hamiltonian MLPs stream directly onto cuBLASLt bf16 Tensor Cores
        inside `grad_checkpoint` without AOTAutograd multi-step buffer blowup."""
        global _compiled_field_loss
        torch._dynamo.config.recompile_limit = 256
        torch._functorch.config.donated_buffer = False
        self.sight.compile(dynamic=True)
        self.persp_enc.compile(dynamic=True)
        self.flow_prior.compile(dynamic=True)
        if _compiled_field_loss is None:
            _compiled_field_loss = torch.compile(_multi_scale_spatial_field_loss, dynamic=True)

    def _state_transition_losses(
        self,
        pred_s: dict[str, torch.Tensor],
        true_s: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Computes grounded physical loss (conserved totals + aligned squad headers +
        multi-scale spatial field kernel) and latent context loss."""
        # Always stop-grad target states so SquadPooler routing at tau+k is never pulled toward bad predictions!
        true_s = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in true_s.items()}

        # 1. Player-level conserved totals: [count, cost, hp, shield] + macro [minerals, vespene, food_used, food_cap]
        tot_p = (pred_s["q_squad"][..., 6:10] * pred_s["active"].float().unsqueeze(-1)).sum(dim=-2)
        tot_t = (true_s["q_squad"][..., 6:10] * true_s["active"].float().unsqueeze(-1)).sum(dim=-2)
        mac_p = pred_s["q_macro"][..., 0:4]
        mac_t = true_s["q_macro"][..., 0:4]

        mse_tot = (tot_p - tot_t).square().mean(dim=(0, 1))  # [4]: count, cost, hp, shield
        mse_mac = (mac_p - mac_t).square().mean(dim=(0, 1))  # [4]: min, vesp, food_u, food_c

        # 2. Aligned slot-level header loss on slots active at tau or tau+1
        both_act = (pred_s["active"] | true_s["active"]).float()
        denom_act = both_act.sum().clamp(min=1.0)
        sq_diff = (pred_s["q_squad"] - true_s["q_squad"]).square().sum(dim=-1) * both_act
        loss_slot_q = sq_diff.sum() / denom_act

        # 3. Multi-scale 2D continuous spatial field loss (permutation & split invariant)
        fl_fn = _compiled_field_loss if _compiled_field_loss is not None else _multi_scale_spatial_field_loss
        loss_field = fl_fn(pred_s, true_s)

        loss_phys = (
            4.0 * mse_tot.sum()
            + 2.0 * mse_mac.sum()
            + 0.5 * loss_slot_q
            + 2.0 * loss_field
        )

        # 4. Latent context consistency (stop-grad target to prevent collapse)
        lat_sq = F.smooth_l1_loss(
            pred_s["z_squad"] * both_act.unsqueeze(-1),
            (true_s["z_squad"].detach()) * both_act.unsqueeze(-1),
        )
        lat_mac = F.smooth_l1_loss(pred_s["z_macro"], true_s["z_macro"].detach())
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

    def forward_sequence(
        self,
        seq: dict[str, torch.Tensor],
        static_map: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Trains Observer on a `[B, T, ...]` sequence of 5.0-second Squad summaries:
          - Posterior -> Omniscient Physics 1-step transition
          - Fog-of-War Strategy Prior Conditional Flow Matching
          - Diffusion-Forcing multi-step (`k = 1..K`) rollout stability using noisy/denoised Prior intents
          - Multi-horizon telescopic Hamiltonian value consistency (`10s, 20s, 40s, 80s`)
        """
        q_sq = seq["q_squad"].float()        # [B, T, 2, M, 12]
        z_sq = seq["z_squad"].float()        # [B, T, 2, M, d_squad]
        active = seq["active"].bool()        # [B, T, 2, M]
        q_mac = seq["q_macro"].float()       # [B, T, 2, 6]
        z_mac = seq["z_macro"].float()       # [B, T, 2, d_squad]
        u_post = seq["u_flat"].float()       # [B, T-1, 2, D_u]
        tempo = seq["tempo_feat"].float()    # [B, T, 2, 16]
        B, T, _, M, _ = q_sq.shape

        # 1. Vectorized dynamic visibility & monotonic explored masks across all [B, T, 2, M] in ONE pass!
        vis_seq, _ = self.sight.compute_visibility(q_sq, z_sq, active)  # [B, T, 2, M]
        exp_seq = vis_seq.detach().cummax(dim=1).values                 # [B, T, 2, M]

        # Calibrate sight head against ground-truth engine visibility if provided (exact fp32 BCE)
        loss_vis = q_sq.sum() * 0.0
        if "gt_vis_live" in seq:
            with torch.autocast(device_type=q_sq.device.type, enabled=False):
                gt_v = seq["gt_vis_live"].float()  # [B, T, 2, M]
                opp_act = active.flip(dims=[2]).float()
                loss_vis = (
                    F.binary_cross_entropy(vis_seq.float().clamp(1e-4, 1.0 - 1e-4), gt_v, reduction="none")
                    * opp_act
                ).sum() / opp_act.sum().clamp(min=1.0)

        # 2. Perspective-masked causal encoder -> h_{<= tau}^{(p)} and capability c_{p, tau}
        tok_seq, cap_inst = self.persp_enc.summarize_perspective_frame(
            q_sq, z_sq, active, q_mac, z_mac, tempo, vis_seq, exp_seq
        )
        h_seq, c_seq = self.persp_enc.forward_sequence(tok_seq, cap_inst)

        # 3. Conditional Flow-Matching Prior on steps 0 .. T - 2
        h_t = h_seq[:, :-1]  # [B, T-1, 2, d_model]
        c_t = c_seq[:, :-1]  # [B, T-1, 2, d_cap]
        loss_flow, u_denoised, _ = self.flow_prior.flow_matching_loss(u_post, h_t, c_t)

        # Tempo prediction auxiliary loss for open-loop rollouts
        tempo_pred = self.flow_prior.tempo_pred(torch.cat([h_t, c_t, u_post.detach()], dim=-1)).float()
        loss_tempo = F.smooth_l1_loss(tempo_pred, tempo[:, 1:].detach())

        # 4. 1-step (5s) Omniscient Physics trained on Posterior Intents
        # StaticTerrainStencil broadcasts B maps across B * (T-1) automatically with zero copies!
        M1 = B * (T - 1)
        s_t_flat = dict(
            q_squad=q_sq[:, :-1].reshape(M1, 2, M, SP.N_SQUAD_Q),
            z_squad=z_sq[:, :-1].reshape(M1, 2, M, self.d_squad),
            active=active[:, :-1].reshape(M1, 2, M),
            q_macro=q_mac[:, :-1].reshape(M1, 2, SP.N_MACRO_Q),
            z_macro=z_mac[:, :-1].reshape(M1, 2, self.d_squad),
        )
        s_n_flat = dict(
            q_squad=q_sq[:, 1:].reshape(M1, 2, M, SP.N_SQUAD_Q),
            z_squad=z_sq[:, 1:].reshape(M1, 2, M, self.d_squad),
            active=active[:, 1:].reshape(M1, 2, M),
            q_macro=q_mac[:, 1:].reshape(M1, 2, SP.N_MACRO_Q),
            z_macro=z_mac[:, 1:].reshape(M1, 2, self.d_squad),
        )

        pred_1, pieces_1 = self.ph.step_dynamics(
            s_t_flat, u_post.reshape(M1, 2, -1), static_map=static_map
        )
        loss_phys_1, loss_lat_1, m1_logs = self._state_transition_losses(pred_1, s_n_flat)
        _, _, copy_logs = self._state_transition_losses(s_t_flat, s_n_flat)

        # Goal alignment: active player control pushes along +grad A_p
        bu_cat = torch.cat([pieces_1["bu_sq"].flatten(2), pieces_1["bu_mac"]], dim=-1).float()
        gg_cat = torch.cat([pieces_1["g_sq_goal"].flatten(2), pieces_1["g_mac_goal"]], dim=-1).detach().float()
        bn = (bu_cat.square().sum(dim=-1) + 1e-8).sqrt()
        gn = (gg_cat.square().sum(dim=-1) + 1e-8).sqrt()
        val_g = (bn > 1e-3) & (gn > 1e-3)
        cos_g = (bu_cat * gg_cat).sum(dim=-1) / (bn * gn).clamp(min=1e-6)
        loss_goal = (F.softplus(-2.0 * cos_g) * val_g.float()).sum() / val_g.float().sum().clamp(min=1.0)
        goal_cos = (cos_g * val_g.float()).sum().detach() / val_g.float().sum().clamp(min=1.0)

        # 5. Diffusion-Forcing Multi-Step Rollout (k = 2..K, +10s..+20s)
        K = min(self.rollout_steps, T - 1)
        loss_phys_multi = loss_phys_1 * 0.0
        loss_lat_multi = loss_lat_1 * 0.0
        if K >= 2 and T > K:
            MK = B * (T - K)
            cur_s = dict(
                q_squad=q_sq[:, : T - K].reshape(MK, 2, M, SP.N_SQUAD_Q),
                z_squad=z_sq[:, : T - K].reshape(MK, 2, M, self.d_squad),
                active=active[:, : T - K].reshape(MK, 2, M),
                q_macro=q_mac[:, : T - K].reshape(MK, 2, SP.N_MACRO_Q),
                z_macro=z_mac[:, : T - K].reshape(MK, 2, self.d_squad),
            )
            u_df = (0.5 * u_post + 0.5 * u_denoised.detach().float())
            for k in range(1, K + 1):
                uk = u_df[:, k - 1 : T - K + k - 1].reshape(MK, 2, -1)
                cur_s, _ = self.ph.step_dynamics(
                    cur_s, uk, static_map=static_map, create_graph_h=False
                )
                if k >= 2:
                    true_k = dict(
                        q_squad=q_sq[:, k : T - K + k].reshape(MK, 2, M, SP.N_SQUAD_Q),
                        z_squad=z_sq[:, k : T - K + k].reshape(MK, 2, M, self.d_squad),
                        active=active[:, k : T - K + k].reshape(MK, 2, M),
                        q_macro=q_mac[:, k : T - K + k].reshape(MK, 2, SP.N_MACRO_Q),
                        z_macro=z_mac[:, k : T - K + k].reshape(MK, 2, self.d_squad),
                    )
                    w_k = 0.5 ** (k - 1)
                    lp_k, ll_k, _ = self._state_transition_losses(cur_s, true_k)
                    loss_phys_multi = loss_phys_multi + w_k * lp_k
                    loss_lat_multi = loss_lat_multi + w_k * ll_k

        # 6. Multi-horizon physical value bootstrapping on H_1, H_2, A_1 (zero winner label / Elo leak!)
        # Reuse `pieces_1["A1"]` on [0 .. T-2] and evaluate `evaluate_hamiltonian` ONLY on final frame T-1!
        h_last = self.ph.evaluate_hamiltonian(
            torch.cat([q_sq[:, -1], z_sq[:, -1]], dim=-1),
            torch.cat([q_mac[:, -1], z_mac[:, -1]], dim=-1),
            active[:, -1],
            static_map=static_map,
        )
        a1_seq = torch.cat([pieces_1["A1"].float().view(B, T - 1), h_last["A1"].float().view(B, 1)], dim=1)
        phys_val = (
            q_mac[..., 0] + q_mac[..., 1] + (q_sq[..., SP.SQ["cost"]] * active.float()).sum(dim=-1)
        ).detach()
        phys_adv = phys_val[:, :, 0] - phys_val[:, :, 1]  # [B, T]
        learned_adv = a1_seq - phys_adv

        loss_td = a1_seq.sum() * 0.0
        horizons = [m for m in (2, 4, 8, 12) if m < T]
        for m in horizons:
            disc = 0.94**m
            tgt_learned = (phys_adv[:, m:] - phys_adv[:, :-m]) + disc * learned_adv[:, m:].detach()
            loss_td = loss_td + F.smooth_l1_loss(learned_adv[:, :-m], tgt_learned) / len(horizons)

        loss = (
            loss_phys_1
            + 0.5 * loss_phys_multi
            + 0.15 * (loss_lat_1 + 0.5 * loss_lat_multi)
            + 1.0 * loss_flow.float()
            + 0.1 * loss_tempo
            + 0.1 * loss_vis
            + 0.05 * loss_goal
            + 0.25 * loss_td
        )

        logs = dict(
            loss=loss.detach(),
            loss_phys=loss_phys_1.detach(),
            loss_phys_multi=loss_phys_multi.detach(),
            loss_flow=loss_flow.detach().float(),
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
        horizon_steps: int = 8,
        flow_steps: int = 4,
        temperature: float = 1.0,
        grid_h: int = 64,
        grid_w: int = 64,
    ) -> dict[str, torch.Tensor]:
        """Rolls out `N = n_branches` stochastic 5.0-second futures from the end of
        `prefix_seq` (`[1, T_obs, ...]`) over `horizon_steps` (e.g., `8 x 5s = 40s`),
        conditioning each player's Flow Prior strictly on their own fog-of-war view
        and inferred online capability `c_p`.

        Returns:
          - `A0`: scalar static advantage at `tau_0` (in resource units, `* 1000` == minerals+gas)
          - `A_traj`: `[N, H + 1]` advantage trajectories across branches (`0 .. H`)
          - `expected_drift`: `E[A(tau_0 + H) - A(tau_0)]`
          - `prob_improve_p1`: fraction of branches where `Delta A > 0`
          - `prob_stronger_hand_p1`: fraction of branches where `A(tau_0 + H) > 0`
          - `std_drift`: `Std[Delta A]` (tactical volatility)
          - `cvar10_p1`: worst-10% branch `Delta A` for P1 (tail risk)
          - `squad_risk`: `[2, M]` expected resource+HP loss per current squad across branches
          - `threat_grid_p1`, `threat_grid_p2`: `[grid_h, grid_w]` emergent 2D spatial threat
            heatmaps (expected value/HP loss projected back onto the battlefield!)
        """
        dev = prefix_seq["q_squad"].device
        q_sq = prefix_seq["q_squad"].float()    # [1, T0, 2, M, 12]
        z_sq = prefix_seq["z_squad"].float()
        active = prefix_seq["active"].bool()
        q_mac = prefix_seq["q_macro"].float()
        z_mac = prefix_seq["z_macro"].float()
        tempo = prefix_seq["tempo_feat"].float()
        _, T0, _, M, _ = q_sq.shape

        # Encode observed prefix [0 .. T0 - 1] once in a single vectorized pass
        vis_pref, _ = self.sight.compute_visibility(q_sq, z_sq, active)
        exp_pref = vis_pref.detach().cummax(dim=1).values
        exp_cur = exp_pref[:, -1]

        tok_pref, cap_pref = self.persp_enc.summarize_perspective_frame(
            q_sq, z_sq, active, q_mac, z_mac, tempo, vis_pref, exp_pref
        )

        # Tile prefix state across N parallel rollout branches
        N = n_branches
        tok_hist = tok_pref.expand(N, -1, -1, -1).clone()
        cap_hist = cap_pref.expand(N, -1, -1, -1).clone()
        exp_n = exp_cur.expand(N, -1, -1).clone()

        cur_s = dict(
            q_squad=q_sq[:, -1].expand(N, -1, -1, -1).clone(),
            z_squad=z_sq[:, -1].expand(N, -1, -1, -1).clone(),
            active=active[:, -1].expand(N, -1, -1).clone(),
            q_macro=q_mac[:, -1].expand(N, -1, -1).clone(),
            z_macro=z_mac[:, -1].expand(N, -1, -1).clone(),
        )
        # Evaluate initial advantage A(tau_0) (StaticTerrainStencil broadcasts B_map=1 across N!)
        h0 = self.ph.evaluate_hamiltonian(
            torch.cat([cur_s["q_squad"][:1], cur_s["z_squad"][:1]], dim=-1),
            torch.cat([cur_s["q_macro"][:1], cur_s["z_macro"][:1]], dim=-1),
            cur_s["active"][:1],
            static_map=static_map,
        )
        a0 = h0["A1"].float()[0]  # scalar in 1,000-resource units
        init_sq_val = (
            cur_s["q_squad"][0, :, :, SP.SQ["cost"]]
            + 0.5 * (cur_s["q_squad"][0, :, :, SP.SQ["hp"]] + cur_s["q_squad"][0, :, :, SP.SQ["shield"]])
        ) * cur_s["active"][0].float()  # [2, M]
        init_pos = cur_s["q_squad"][0, :, :, 0:2]      # [2, M, 2]
        init_spread = cur_s["q_squad"][0, :, :, 4:6]   # [2, M, 2]
        init_act = cur_s["active"][0]                  # [2, M]

        a_steps = [a0.expand(N)]

        for step_k in range(horizon_steps):
            h_seq, c_seq = self.persp_enc.forward_sequence(tok_hist, cap_hist)
            h_last = h_seq[:, -1]  # [N, 2, d_model]
            c_last = c_seq[:, -1]  # [N, 2, d_cap]

            u_samp, tempo_next = self.flow_prior.sample(
                h_last, c_last, steps=flow_steps, temperature=temperature
            )
            cur_s, pieces_k = self.ph.step_dynamics(cur_s, u_samp.float(), static_map=static_map)

            # Evaluate new state's Hamiltonian & update fog-of-war visibility for next step!
            hk = self.ph.evaluate_hamiltonian(
                torch.cat([cur_s["q_squad"], cur_s["z_squad"]], dim=-1),
                torch.cat([cur_s["q_macro"], cur_s["z_macro"]], dim=-1),
                cur_s["active"],
                static_map=static_map,
            )
            a_steps.append(hk["A1"].float())

            if step_k + 1 < horizon_steps:
                vl_next, exp_n = self.sight.compute_visibility(
                    cur_s["q_squad"], cur_s["z_squad"], cur_s["active"], prev_explored=exp_n
                )
                tok_next, cap_next = self.persp_enc.summarize_perspective_frame(
                    cur_s["q_squad"],
                    cur_s["z_squad"],
                    cur_s["active"],
                    cur_s["q_macro"],
                    cur_s["z_macro"],
                    tempo_next,
                    vl_next,
                    exp_n,
                )
                # Keep sliding context window <= 64 macro steps (~5.3 min) for constant speed
                tok_hist = torch.cat([tok_hist[:, -63:], tok_next.unsqueeze(1)], dim=1)
                cap_hist = torch.cat([cap_hist[:, -63:], cap_next.unsqueeze(1)], dim=1)

        a_traj = torch.stack(a_steps, dim=1) * 1000.0  # [N, H + 1] in mineral+gas equivalent units!
        a0_res = a_traj[0, 0]
        a_end = a_traj[:, -1]
        drift = a_end - a0_res  # [N]

        prob_improve_p1 = (drift > 0.0).float().mean()
        prob_stronger_p1 = (a_end > 0.0).float().mean()
        expected_drift = drift.mean()
        std_drift = drift.std(unbiased=False)
        k_cvar = max(1, N // 10)
        cvar10_p1 = torch.topk(drift, k=k_cvar, largest=False).values.mean()

        # Per-squad expected attrition/risk over the horizon (in resource-equivalent units)
        end_sq_val = (
            cur_s["q_squad"][:, :, :, SP.SQ["cost"]]
            + 0.5 * (cur_s["q_squad"][:, :, :, SP.SQ["hp"]] + cur_s["q_squad"][:, :, :, SP.SQ["shield"]])
        ) * cur_s["active"].float()  # [N, 2, M]
        sq_losses = (init_sq_val.unsqueeze(0) - end_sq_val).clamp(min=0.0) * 1000.0  # [N, 2, M]
        # Combine mean expected damage + 90th-percentile tail damage (captures 40%-branch Pool snipes!)
        sq_p90 = torch.quantile(sq_losses, 0.90, dim=0)
        squad_risk = (0.5 * sq_losses.mean(dim=0) + 0.5 * sq_p90) * init_act.float()  # [2, M]

        # Project emergent squad risk onto 2D playable map grid [grid_h, grid_w]
        if static_map is not None and "half" in static_map:
            hx = float(static_map["half"][0, 0].item()) / E.MAP_SCALE
            hy = float(static_map["half"][0, 1].item()) / E.MAP_SCALE
        else:
            hx, hy = 0.85, 0.85
        xs = torch.linspace(-hx, hx, grid_w, device=dev)
        ys = torch.linspace(-hy, hy, grid_h, device=dev)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # [H_g, W_g]

        threat_grids = []
        for p_i in (0, 1):
            # Threat TO player p_i's exposed structures & forces across rollout futures
            sig_x = init_spread[p_i, :, 0].clamp(0.05, 0.20)
            sig_y = init_spread[p_i, :, 1].clamp(0.05, 0.20)
            dx2 = (gx.unsqueeze(-1) - init_pos[p_i, :, 0]) ** 2 / (2.0 * sig_x**2)
            dy2 = (gy.unsqueeze(-1) - init_pos[p_i, :, 1]) ** 2 / (2.0 * sig_y**2)
            kern = torch.exp(-(dx2 + dy2)) * init_act[p_i].float()  # [H_g, W_g, M]
            t_grid = torch.matmul(kern, squad_risk[p_i])            # [H_g, W_g]
            threat_grids.append(t_grid)

        end_total_cost = (
            cur_s["q_squad"][:, :, :, SP.SQ["cost"]] * cur_s["active"].float()
        ).sum(dim=-1).mean(dim=0)  # [2]
        end_minerals = cur_s["q_macro"][:, :, 0].mean(dim=0)  # [2]

        return dict(
            A0=a0_res,
            A_traj=a_traj,
            expected_drift=expected_drift,
            prob_improve_p1=prob_improve_p1,
            prob_stronger_hand_p1=prob_stronger_p1,
            std_drift=std_drift,
            cvar10_p1=cvar10_p1,
            capability_p1=c_last.float().mean(dim=0)[0],
            capability_p2=c_last.float().mean(dim=0)[1],
            end_total_cost=end_total_cost,
            end_minerals=end_minerals,
            squad_risk=squad_risk,
            squad_pos=init_pos,
            squad_active=init_act,
            threat_grid_p1=threat_grids[0],
            threat_grid_p2=threat_grids[1],
        )
