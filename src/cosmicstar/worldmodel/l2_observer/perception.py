"""Static map terrain stencils, squad visibility extraction, and `SightFootprintHead`."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.constants import MAP_SCALE
from ..l1_tactical import squad_pooler as SP
from ..nn.layers import small_init

if TYPE_CHECKING:
    from ..data.game_store import Game


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
            pos_flat = pos_norm.reshape(B_map, -1, 2).float() * MAP_SCALE
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
            pos_flat = pos_norm.reshape(B_map, -1, 2).float() * MAP_SCALE
            samp = torch.zeros(B_map, pos_flat.shape[1], n_pts * 4, device=dev, dtype=torch.float32)

        ang = pos_flat.unsqueeze(-1) * self.omega
        fourier = torch.cat([ang.sin().flatten(-2), ang.cos().flatten(-2)], dim=-1)
        out = self.net(torch.cat([fourier, samp], dim=-1))
        return out.reshape(*orig_shape[:-1], self.d_out)


def extract_rich_static_map(game) -> dict[str, torch.Tensor]:
    """Extracts 4-channel playable static map tensor `[height_norm, pathable, buildable, playable]`
    (`[gh, gw, 4]` fp16 on CPU) plus `origin`, `center`, `half` directly from `Game`."""
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
        small_init(self.net[-1], std=1e-2)

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
