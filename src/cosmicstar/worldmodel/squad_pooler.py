"""Layer 1 -> Layer 2 Squad-Token Cross-Attention Pooler & Hindsight Intent Posterior.

Replaces algebraic Gram matrices (G_qq, G_qc, G_cc) with:
  1. `SquadPooler`: Learned 2D-RoPE cross-attention pooling from Layer 1's contextualized
     entities into `1` Macro Token + `M = 64` Squad Tokens per player using Top-2 per-unit
     routing. Guarantees 100% conservation of unit count, resource value, HP, and shields
     (including cargo inside transports and workers inside gas geysers) while keeping
     unused slots strictly inactive (`mass == 0`) in early-game states.
  2. `HindsightIntentPosterior`: Compresses a 5-second (`8 x 0.625s` = 112-loop) window
     of `(S_tau, binned commands + camera shifts + control-group recalls/binds, S_{tau+1})`
     into `K = 4` spatial-semantic intent slots + `1` macro intent + a `16d` micro-execution
     residual `xi`, grounded to map coordinates via an auxiliary 2D-RoPE squad-order pointer
     objective.
"""

from __future__ import annotations

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import energy as E
from . import energy_data as ED

N_SQUADS_DEFAULT = 64
D_SQUAD_DEFAULT = 128
K_INTENT_DEFAULT = 4
D_INTENT_DEFAULT = 32
D_XI_DEFAULT = 16
D_TEMPO_FEAT = 16

# 12 grounded physical observables per Squad Token (fp32, zero Gram matrices)
SQUAD_Q_COLS = (
    "x",           # 0: center-of-mass x relative to map center (/ 100.0)
    "y",           # 1: center-of-mass y relative to map center (/ 100.0)
    "vx",          # 2: velocity vx (/ SPEED_SCALE, 1.0 == 5.0 map units/s)
    "vy",          # 3: velocity vy (/ SPEED_SCALE, 1.0 == 5.0 map units/s)
    "spread_x",    # 4: spatial std-dev in x (/ 100.0, >= 0.01)
    "spread_y",    # 5: spatial std-dev in y (/ 100.0, >= 0.01)
    "count",       # 6: conserved unit count (/ N0 = 100.0)
    "cost",        # 7: conserved mineral + vespene value (/ 1000.0, matches BANK_SCALE)
    "hp",          # 8: conserved hitpoints (/ 1000.0)
    "shield",      # 9: conserved shields (/ 1000.0)
    "struct_frac", # 10: structure mass fraction in [0, 1] (for fog vis==1 snapshot rules)
    "air_frac",    # 11: flying unit mass fraction in [0, 1] (for cliff/dead-space physics)
)
SQ = {name: i for i, name in enumerate(SQUAD_Q_COLS)}
N_SQUAD_Q = len(SQUAD_Q_COLS)  # 12

# 6 grounded physical observables per Player Macro Token
MACRO_Q_COLS = (
    "minerals",    # 0: bank minerals (/ 1000.0)
    "vespene",     # 1: bank vespene (/ 1000.0)
    "food_used",   # 2: supply used (/ 100.0)
    "food_cap",    # 3: supply cap (/ 100.0)
    "cam_x",       # 4: player camera x relative to map center (/ 100.0)
    "cam_y",       # 5: player camera y relative to map center (/ 100.0)
)
MQ = {name: i for i, name in enumerate(MACRO_Q_COLS)}
N_MACRO_Q = len(MACRO_Q_COLS)  # 6


def build_unit_meta_tensors(
    unit_data_path: str = "data/worldmodel/unit_data.json",
    n_raw_types: int = ED.N_RAW_TYPES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns `(cost_1k, is_struct, is_worker)` indexed by raw SC2 API unit type id."""
    cost_1k = torch.zeros(n_raw_types, dtype=torch.float32)
    is_struct = torch.zeros(n_raw_types, dtype=torch.float32)
    is_worker = torch.zeros(n_raw_types, dtype=torch.float32)
    raw = json.load(open(unit_data_path))
    for k, u in raw.items():
        tid = int(k)
        if 0 <= tid < n_raw_types:
            c = float(u.get("minerals", 0) + u.get("vespene", 0))
            cost_1k[tid] = max(c, 0.0) / E.BANK_SCALE
            is_struct[tid] = 1.0 if u.get("structure", False) else 0.0
            is_worker[tid] = 1.0 if u.get("worker", False) else 0.0
    return cost_1k, is_struct, is_worker


def _apply_rope_2d(vec: torch.Tensor, pos_map: torch.Tensor, rope_2d: ED.RoPE2D) -> torch.Tensor:
    """Applies 2D axial RoPE rotation to `[..., N, D]` using map-unit positions `[..., N, 2]`."""
    orig_shape = vec.shape
    N, D = orig_shape[-2], orig_shape[-1]
    dh = rope_2d.omega.numel() * 4
    heads = D // dh
    v_flat = vec.reshape(-1, N, heads, dh).transpose(1, 2)  # [M_b, H, N, dh]
    cs = ED.cos_sin(rope_2d.angles(pos_map.reshape(-1, N, 2))).to(vec.dtype)
    v_rot = ED.rotate(v_flat, cs)
    return v_rot.transpose(1, 2).reshape(orig_shape)


class SquadPooler(nn.Module):
    """Pools variable-count L1 entities over a 5-second (`8 x 0.625s`) window into
    `1` Macro Token + `M = 64` Squad Tokens per player via Top-2 exact-conservation
    2D-RoPE cross-attention."""

    def __init__(
        self,
        tables: ED.Tables,
        d_l1: int = 256,
        d_squad: int = D_SQUAD_DEFAULT,
        n_squads: int = N_SQUADS_DEFAULT,
        unit_data_path: str = "data/worldmodel/unit_data.json",
    ):
        super().__init__()
        self.d_l1 = d_l1
        self.d_squad = d_squad
        self.n_squads = n_squads

        cost_1k, is_struct, is_worker = build_unit_meta_tensors(unit_data_path)
        self.register_buffer("cost_1k", cost_1k, persistent=False)
        self.register_buffer("is_struct", is_struct, persistent=False)
        self.register_buffer("is_worker", is_worker, persistent=False)
        self.register_buffer("is_flying", tables.k[:, 5].float(), persistent=False)

        # 2D RoPE matching L1's spatial wavelengths (dh = 64 -> 4 heads for 256d)
        self.rope_2d = ED.RoPE2D(64, lo=ED.ROPE_SHORTEST, hi=256.0)

        # Learned diverse base slot queries + fallback relative map offsets (in map units)
        self.slot_queries = nn.Parameter(torch.randn(n_squads, d_l1) * 0.05)
        nn.init.orthogonal_(self.slot_queries)
        # Multi-ring spatial offsets around player's unit centroid for unseeded / waking slots
        angles = torch.linspace(0, 2.0 * math.pi * (1.0 - 1.0 / n_squads), n_squads)
        radii = torch.linspace(0.0, 48.0, n_squads)
        init_offsets = torch.stack([radii * angles.cos(), radii * angles.sin()], dim=-1)
        self.register_buffer("slot_init_offsets", init_offsets, persistent=False)

        # Temporal seeding from previous 5s step's squad latents
        self.prev_to_query = nn.Sequential(
            nn.LayerNorm(d_squad + N_SQUAD_Q),
            nn.Linear(d_squad + N_SQUAD_Q, d_l1),
            nn.SiLU(),
            nn.Linear(d_l1, d_l1),
        )
        E.small_init(self.prev_to_query[-1], std=2e-2)

        # Entity key/value projections incorporating trajectory delta + control groups + role flags
        # Input: c_end (d_l1) + c_delta (d_l1) + groups_onscreen (12) + role_flags (4: struct, worker, air, cost)
        ent_in_dim = 2 * d_l1 + 12 + 4
        self.ent_k = nn.Sequential(
            nn.LayerNorm(ent_in_dim),
            nn.Linear(ent_in_dim, d_l1, bias=False),
        )
        self.ent_v = nn.Sequential(
            nn.LayerNorm(ent_in_dim),
            nn.Linear(ent_in_dim, d_squad),
            nn.SiLU(),
            nn.Linear(d_squad, d_squad),
        )
        self.query_norm = nn.LayerNorm(d_l1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(8.0)))

        # Cargo / gas-worker fallback context encoder for units with inside > 0
        self.cargo_type_emb = nn.Embedding(ED.N_RAW_TYPES, d_squad)
        nn.init.normal_(self.cargo_type_emb.weight, std=0.02)

        # Squad post-pooling refinement MLP
        self.squad_norm = nn.LayerNorm(d_squad)
        self.squad_mlp = nn.Sequential(
            nn.Linear(d_squad + N_SQUAD_Q, d_squad),
            nn.SiLU(),
            nn.Linear(d_squad, d_squad),
        )
        E.small_init(self.squad_mlp[-1], std=2e-2)

        # Macro token projection from [ply_end, cls_end, ply_delta] -> d_squad
        self.macro_proj = nn.Sequential(
            nn.LayerNorm(3 * d_l1 + N_MACRO_Q),
            nn.Linear(3 * d_l1 + N_MACRO_Q, d_squad),
            nn.SiLU(),
            nn.Linear(d_squad, d_squad),
        )

    def _extract_macro_q(self, b: dict[str, torch.Tensor], f_end: int) -> torch.Tensor:
        """Extracts `[B, 2, 6]` grounded player macro observables at frame `f_end`."""
        with torch.autocast(device_type=b["ply_num"].device.type, enabled=False):
            pl = b["ply_num"][:, f_end].float()  # [B, 2, C_PLY]
            n_pc = len(ED.PC_KEYS)
            v = torch.expm1(pl[..., :n_pc] * 8.0)
            mins = v[..., ED.PC["minerals"]] / E.BANK_SCALE
            vesp = v[..., ED.PC["vespene"]] / E.BANK_SCALE
            food_used = v[..., ED.PC["food_used"]] / E.N0
            food_cap = v[..., ED.PC["food_cap"]] / E.N0

            # Recover camera relative to map center (/ 100.0)
            if "half" in b:
                hx = b["half"][:, 0:1].float()
                hy = b["half"][:, 1:2].float()
            else:
                hx = torch.full_like(mins[:, :1], 100.0)
                hy = torch.full_like(mins[:, :1], 100.0)
            cam_ok = pl[..., n_pc + 2] > 0.5
            cam_x = torch.where(cam_ok, (pl[..., n_pc + 0] * hx) / E.MAP_SCALE, 0.0)
            cam_y = torch.where(cam_ok, (pl[..., n_pc + 1] * hy) / E.MAP_SCALE, 0.0)
            return torch.stack([mins, vesp, food_used, food_cap, cam_x, cam_y], dim=-1)

    def forward(
        self,
        b: dict[str, torch.Tensor],
        ctx: dict[str, torch.Tensor],
        f_start: int = 0,
        f_end: int = -1,
        prev_state: dict[str, torch.Tensor] | None = None,
        batch_frame_end: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Pools L1 window context `ctx` (from `Frames.encode_window_context`) into
        player-symmetric Squad & Macro tokens at frame `f_end` (summarizing `[f_start .. f_end]`).

        Args:
            b: Raw collated batch dict on device.
            ctx: Output dict of `Frames.encode_window_context(b, f_idx)`.
            f_start: Start index inside `ctx` tensors (default `0`).
            f_end: End index inside `ctx` tensors (default `-1` or last frame of window).
            prev_state: Optional dict from step `tau - 1` (`q_squad`, `z_squad`, `active`)
                to seed temporally persistent squad queries.
            batch_frame_end: Corresponding frame index inside `b` (defaults to `f_end`).
        """
        c_ent_seq = ctx["c_ent"]  # [B, nf, R, d_l1]
        B, nf, R, _ = c_ent_seq.shape
        if f_end < 0:
            f_end = nf + f_end
        if batch_frame_end is None:
            batch_frame_end = f_end

        dev = c_ent_seq.device
        c_end = c_ent_seq[:, f_end].float()
        c_start = c_ent_seq[:, f_start].float()
        c_delta = c_end - c_start

        q_unit = ctx["q"][:, f_end].float()  # [B, R, 8]
        on_map = ctx["mask"][:, f_end]       # [B, R] bool (inside == 0 & alive)
        owner_raw = b["owner"]               # [B, R] (0=P1, 1=P2, NEUTRAL)
        alive_all = b["mask"][:, batch_frame_end] & b["alive"][:, batch_frame_end]
        inside = b["inside"][:, batch_frame_end]  # 0=map, 1=cargo, 2=gas
        parent_row = (
            b["parent_row"][:, batch_frame_end]
            if "parent_row" in b
            else torch.full((B, R), -1, device=dev, dtype=torch.long)
        )

        rtype = b["rtype"][:, batch_frame_end].clamp(0, ED.N_RAW_TYPES - 1)
        num = b["num"][:, batch_frame_end].float()

        if "center" in b:
            center = b["center"][:, None, :].float()  # [B, 1, 2]
        else:
            center = torch.full((B, 1, 2), 100.0, device=dev)
        pos_map = torch.stack(
            [
                b["x"][:, batch_frame_end].float(),
                b["y"][:, batch_frame_end].float(),
            ],
            dim=-1,
        ) - center  # [B, R, 2] in map units

        # Control groups + F2_ONSCREEN
        onscreen = ((num[..., ED.I["flags2"]].long() & 4) != 0).float().unsqueeze(-1)
        if "groups" in b:
            grp_in = torch.cat([b["groups"][:, batch_frame_end].float(), onscreen], dim=-1)
        else:
            grp_in = torch.cat([torch.zeros(B, R, 11, device=dev), onscreen], dim=-1)

        u_cost = self.cost_1k[rtype]      # [B, R]
        u_struct = self.is_struct[rtype]  # [B, R]
        u_worker = self.is_worker[rtype]  # [B, R]
        u_air = self.is_flying[rtype]     # [B, R]
        role_flags = torch.stack([u_struct, u_worker, u_air, torch.log1p(u_cost * 10.0)], dim=-1)

        ent_in = torch.cat([c_end, c_delta, grp_in, role_flags], dim=-1)
        k_all = F.normalize(self.ent_k(ent_in), dim=-1, eps=1e-4)  # [B, R, d_l1]
        v_onmap = self.ent_v(ent_in)                     # [B, R, d_squad]
        v_cargo = self.cargo_type_emb(rtype)             # [B, R, d_squad]
        v_all = torch.where((inside == 0).unsqueeze(-1), v_onmap, v_cargo)

        # Per-unit physical quantities (in fp32 for exact conservation)
        with torch.autocast(device_type=dev.type, enabled=False):
            x_norm = ((pos_map[..., 0]) / E.MAP_SCALE).clamp(-2.0, 2.0)
            y_norm = ((pos_map[..., 1]) / E.MAP_SCALE).clamp(-2.0, 2.0)
            # On-map units get velocity from q_unit; cargo inherits parent velocity via pos_map/0
            vx_norm = torch.where(inside == 0, q_unit[..., E.Q["vx"]], torch.zeros_like(x_norm))
            vy_norm = torch.where(inside == 0, q_unit[..., E.Q["vy"]], torch.zeros_like(y_norm))
            hp_1k = num[..., ED.I["hp"]].clamp(min=0.0) / E.BANK_SCALE
            sh_1k = num[..., ED.I["shield"]].clamp(min=0.0) / E.BANK_SCALE

            # Rotate keys once with 2D RoPE at exact entity map coordinates
            k_rot = _apply_rope_2d(k_all, pos_map, self.rope_2d)
            scale = self.logit_scale.exp().clamp(min=1.0, max=50.0)

            q_squad_list = []
            z_squad_list = []
            active_list = []
            weights_list = []

            M = self.n_squads
            for p_idx in (0, 1):
                p_onmap = on_map & (owner_raw == p_idx)         # [B, R]
                p_alive = alive_all & (owner_raw == p_idx)      # [B, R] (includes cargo & gas)
                p_cargo = p_alive & (~p_onmap)                  # [B, R]

                # Compute player's unit centroid in map units for fallback slot offsets
                cnt_onmap = p_onmap.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
                centroid = (pos_map * p_onmap.float().unsqueeze(-1)).sum(dim=1, keepdim=True) / cnt_onmap.unsqueeze(-1)
                q_pos = centroid + self.slot_init_offsets[None, :, :]  # [B, M, 2]
                q_vec = self.slot_queries[None, :, :].expand(B, M, -1)

                if prev_state is not None:
                    prev_act = prev_state["active"][:, p_idx]          # [B, M] bool
                    prev_q = prev_state["q_squad"][:, p_idx].float()   # [B, M, N_SQUAD_Q]
                    prev_z = prev_state["z_squad"][:, p_idx].float()   # [B, M, d_squad]
                    # Advect previous squad center by 5.0s (~112 loops * SPEED_SCALE = 25.0 map units at v=1.0)
                    adv_pos = (
                        prev_q[..., 0:2] * E.MAP_SCALE
                        + prev_q[..., 2:4] * (E.SPEED_SCALE * 112.0)
                    )
                    q_pos = torch.where(prev_act.unsqueeze(-1), adv_pos, q_pos)
                    seeded_q = q_vec + self.prev_to_query(torch.cat([prev_z, prev_q], dim=-1))
                    q_vec = torch.where(prev_act.unsqueeze(-1), seeded_q, q_vec)

                q_vec = F.normalize(self.query_norm(q_vec), dim=-1, eps=1e-4)
                q_rot = _apply_rope_2d(q_vec, q_pos, self.rope_2d)  # [B, M, d_l1]

                # Cross-attention scores [B, M, R]
                logits = torch.matmul(q_rot, k_rot.transpose(1, 2)) * scale

                # Top-2 exact-conservation routing across the M=64 slots per living unit
                top_vals, top_idx = torch.topk(logits, k=2, dim=1)  # [B, 2, R]
                top_w = F.softmax(top_vals, dim=1) * p_onmap.float().unsqueeze(1)  # [B, 2, R]

                w_onmap = torch.zeros(B, M, R, device=dev, dtype=torch.float32)
                w_onmap.scatter_add_(1, top_idx, top_w)  # sum_m w_onmap[b, m, u] == 1.0 for all p_onmap!

                # Attach cargo / gas-geyser units directly to their parent transport/extractor's slots
                if p_cargo.any():
                    valid_par = (parent_row >= 0) & (parent_row < R)
                    safe_par = parent_row.clamp(0, max(0, R - 1))  # [B, R]
                    par_idx_exp = safe_par.unsqueeze(1).expand(B, M, R)
                    w_from_par = torch.gather(w_onmap, 2, par_idx_exp)  # [B, M, R]
                    par_claimed = valid_par & (w_from_par.sum(dim=1) > 0.5)

                    # Fallback if parent wasn't routed (e.g. neutral gas geyser): use unit's own top-2 at geyser pos
                    top_w_fb = F.softmax(top_vals, dim=1)
                    w_fb = torch.zeros(B, M, R, device=dev, dtype=torch.float32)
                    w_fb.scatter_add_(1, top_idx, top_w_fb)

                    w_c = torch.where(par_claimed.unsqueeze(1), w_from_par, w_fb)
                    w = w_onmap + w_c * p_cargo.float().unsqueeze(1)
                else:
                    w = w_onmap

                mass = w.sum(dim=-1)  # [B, M] exact unit count in slot
                active = mass > 1e-4  # [B, M] bool: unused slots have exact mass == 0.0!
                denom = mass.clamp(min=1e-6)

                # Grounded physical header q_squad [B, M, 12]
                xm = (w * x_norm.unsqueeze(1)).sum(dim=-1) / denom
                ym = (w * y_norm.unsqueeze(1)).sum(dim=-1) / denom
                vxm = (w * vx_norm.unsqueeze(1)).sum(dim=-1) / denom
                vym = (w * vy_norm.unsqueeze(1)).sum(dim=-1) / denom

                var_x = (w * (x_norm.unsqueeze(1) - xm.unsqueeze(-1)).square()).sum(dim=-1) / denom
                var_y = (w * (y_norm.unsqueeze(1) - ym.unsqueeze(-1)).square()).sum(dim=-1) / denom
                spx = (var_x + 1e-4).sqrt() * active.float()
                spy = (var_y + 1e-4).sqrt() * active.float()

                cnt_sum = mass / E.N0
                cost_sum = (w * u_cost.unsqueeze(1)).sum(dim=-1)
                hp_sum = (w * hp_1k.unsqueeze(1)).sum(dim=-1)
                sh_sum = (w * sh_1k.unsqueeze(1)).sum(dim=-1)
                st_frac = ((w * u_struct.unsqueeze(1)).sum(dim=-1) / denom) * active.float()
                air_frac = ((w * u_air.unsqueeze(1)).sum(dim=-1) / denom) * active.float()

                act_f = active.float()
                q_sq = torch.stack(
                    [
                        xm * act_f,
                        ym * act_f,
                        vxm * act_f,
                        vym * act_f,
                        spx,
                        spy,
                        cnt_sum,
                        cost_sum,
                        hp_sum,
                        sh_sum,
                        st_frac,
                        air_frac,
                    ],
                    dim=-1,
                )  # [B, M, 12]

                # Pooled contextualized squad latent z_squad [B, M, d_squad]
                z_raw = torch.matmul(w, v_all) / denom.unsqueeze(-1)
                z_normed = self.squad_norm(z_raw) * act_f.unsqueeze(-1)
                z_sq = (
                    z_normed + self.squad_mlp(torch.cat([z_normed, q_sq], dim=-1))
                ) * act_f.unsqueeze(-1)

                q_squad_list.append(q_sq)
                z_squad_list.append(z_sq)
                active_list.append(active)
                weights_list.append(w)

            q_squad = torch.stack(q_squad_list, dim=1)        # [B, 2, M, 12]
            z_squad = torch.stack(z_squad_list, dim=1)        # [B, 2, M, d_squad]
            active_squad = torch.stack(active_list, dim=1)    # [B, 2, M]
            assign_w = torch.stack(weights_list, dim=1)       # [B, 2, M, R]

            # Player Macro Token [B, 2, 6] + [B, 2, d_squad]
            q_macro = self._extract_macro_q(b, batch_frame_end)
            ply_end = ctx["ply_out"][:, f_end].float()        # [B, 2, d_l1]
            ply_start = ctx["ply_out"][:, f_start].float()
            cls_end = ctx["cls_out"][:, f_end].float().unsqueeze(1).expand(B, 2, -1)
            macro_in = torch.cat([ply_end, cls_end, ply_end - ply_start, q_macro], dim=-1)
            z_macro = torch.tanh(self.macro_proj(macro_in)) * 2.0  # bounded (-2, 2)

            return dict(
                q_squad=q_squad,
                z_squad=z_squad,
                active=active_squad,
                q_macro=q_macro,
                z_macro=z_macro,
                assign_w=assign_w,
            )


class HindsightIntentPosterior(nn.Module):
    """Compresses a 5-second (`8 x 0.625s` = 112-loop) window `(S_tau, actions/cam/hotkeys, S_{tau+1})`
    into `K = 4` spatial-semantic intents + `1` macro intent + `16d` micro residual `xi`,
    plus purely causal 5s player tempo statistics for online capability inference `c_p`."""

    def __init__(
        self,
        d_squad: int = D_SQUAD_DEFAULT,
        k_intent: int = K_INTENT_DEFAULT,
        d_intent: int = D_INTENT_DEFAULT,
        d_xi: int = D_XI_DEFAULT,
        hidden: int = 256,
    ):
        super().__init__()
        self.d_squad = d_squad
        self.k_intent = k_intent
        self.d_intent = d_intent
        self.d_xi = d_xi

        self.rope_2d = ED.RoPE2D(64, lo=ED.ROPE_SHORTEST, hi=256.0)

        # Raw action event encoder (commands, camera pans, selections, control-group binds/recalls)
        self.act_kind_emb = nn.Embedding(16, 64, padding_idx=0)
        self.act_grp_emb = nn.Embedding(16, 32)
        self.act_gop_emb = nn.Embedding(8, 16)
        self.act_feat_proj = nn.Sequential(
            nn.Linear(64 + 32 + 16 + 12 + 3, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # Squad state + hindsight delta encoder
        sq_in_dim = 2 * (d_squad + N_SQUAD_Q) + 1
        self.sq_enc = nn.Sequential(
            nn.LayerNorm(sq_in_dim),
            nn.Linear(sq_in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # Learned K=4 spatial intent slot queries
        self.intent_queries = nn.Parameter(torch.randn(k_intent, hidden) * 0.05)
        nn.init.orthogonal_(self.intent_queries)
        self.intent_attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.intent_norm = nn.LayerNorm(hidden)

        self.intent_gate_head = nn.Linear(hidden, 1)
        self.intent_z_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_intent),
        )
        self.intent_pos_offset = nn.Linear(hidden, 2)
        E.small_init(self.intent_pos_offset, std=1e-2)

        # Macro intent & micro-execution residual heads
        macro_in_dim = 2 * (d_squad + N_MACRO_Q) + hidden + D_TEMPO_FEAT
        self.macro_intent_head = nn.Sequential(
            nn.LayerNorm(macro_in_dim),
            nn.Linear(macro_in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_intent),
        )
        self.xi_head = nn.Sequential(
            nn.LayerNorm(macro_in_dim),
            nn.Linear(macro_in_dim, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, d_xi),
        )
        E.small_init(self.xi_head[-1], std=2e-2)

        # Auxiliary 2D-RoPE pointer projections to ground intent (x_k, y_k) onto ordered squads
        self.ptr_q = nn.Linear(d_intent, 64, bias=False)
        self.ptr_k = nn.Linear(d_squad + N_SQUAD_Q, 64, bias=False)
        self.ptr_rope = ED.RoPE2D(64, lo=ED.ROPE_SHORTEST, hi=256.0)
        self.ptr_scale = nn.Parameter(torch.tensor(math.log(6.0)))

    def extract_window_tempo_and_tokens(
        self,
        b: dict[str, torch.Tensor],
        f_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts:
        - `tempo_feat`: `[B, 2, 16]` purely causal 5s behavioral tempo stats (camera jumps,
          hotkey binds/recalls, command dispersion, APM) for capability latent `c_p`.
        - `act_tok`: `[B, 2, N_tok, hidden]` spatialized action + camera tokens.
        - `act_pos`: `[B, 2, N_tok, 2]` normalized coordinates (`/ 100.0`).
        - `act_valid`: `[B, 2, N_tok]` bool mask.
        """
        dev = b["ply_num"].device
        B = b["ply_num"].shape[0]
        f_idx = torch.as_tensor(f_indices, device=dev, dtype=torch.long)
        nf = len(f_indices)

        # 1. Camera trajectory across the nf frames [B, nf, 2, 2]
        pl = b["ply_num"][:, f_idx].float()
        n_pc = len(ED.PC_KEYS)
        hx = b["half"][:, 0].float()[:, None, None] if "half" in b else torch.full((B, 1, 1), 100.0, device=dev)
        hy = b["half"][:, 1].float()[:, None, None] if "half" in b else torch.full((B, 1, 1), 100.0, device=dev)
        cam_ok = pl[..., n_pc + 2] > 0.5
        cam_x = torch.where(cam_ok, (pl[..., n_pc + 0] * hx) / E.MAP_SCALE, 0.0)
        cam_y = torch.where(cam_ok, (pl[..., n_pc + 1] * hy) / E.MAP_SCALE, 0.0)
        cam_xy = torch.stack([cam_x, cam_y], dim=-1).transpose(1, 2)  # [B, 2, nf, 2]

        cam_step = (cam_xy[:, :, 1:] - cam_xy[:, :, :-1]).norm(dim=-1)  # [B, 2, nf-1]
        cam_jumps = (cam_step > 0.12).float().sum(dim=-1)               # >12 map-unit jumps
        cam_dist = cam_step.sum(dim=-1)

        # 2. Action tokens across the nf strided frames
        has_acts = "act_mask" in b and "act_kind" in b
        if has_acts:
            amask = b["act_mask"][:, f_idx]          # [B, nf, A]
            aply = b["act_player"][:, f_idx]         # [B, nf, A]
            akind = b["act_kind"][:, f_idx].clamp(0, 15)
            agrp = b["act_group"][:, f_idx].clamp(0, 15) if "act_group" in b else torch.zeros_like(akind)
            agop = b["act_gop"][:, f_idx].clamp(0, 7) if "act_gop" in b else torch.zeros_like(akind)
            anum = b["act_num"][:, f_idx].float() if "act_num" in b else torch.zeros(*akind.shape, 12, device=dev)
            axy = (b["act_xy"][:, f_idx].float() / E.MAP_SCALE) if "act_xy" in b else torch.zeros(*akind.shape, 2, device=dev)
            aplaced = b["act_placed"][:, f_idx] if "act_placed" in b else (axy.abs().sum(dim=-1) > 1e-5)

            # Fallback unplaced actions (e.g. hotkey recalls / macro production) to current camera xy!
            cam_per_f = cam_xy.transpose(1, 2)  # [B, nf, 2, 2]

            tok_p, pos_p, val_p, tempo_p = [], [], [], []
            for p_i in (0, 1):
                pm = amask & (aply == p_i)  # [B, nf, A]
                cam_p_f = cam_per_f[:, :, p_i, :].unsqueeze(2).expand(*axy.shape)
                pos_ev = torch.where(aplaced.unsqueeze(-1), axy, cam_p_f)  # [B, nf, A, 2]

                # Action category counts for online capability c_p
                is_cmd = pm & ((akind == 1) | (akind == 2) | (akind == 11))
                is_sel = pm & ((akind >= 3) & (akind <= 6))
                is_grp = pm & ((akind >= 7) & (akind <= 9))
                is_cam = pm & (akind == 10)
                n_total = pm.float().sum(dim=(1, 2))
                n_cmd = is_cmd.float().sum(dim=(1, 2))
                n_sel = is_sel.float().sum(dim=(1, 2))
                n_grp = is_grp.float().sum(dim=(1, 2))
                n_cam = is_cam.float().sum(dim=(1, 2))

                # Spatial dispersion of commands across the map in this 5s window
                cmd_w = is_cmd.float()
                cmd_cnt = cmd_w.sum(dim=(1, 2)).clamp(min=1.0)
                cmd_mean = (pos_ev * cmd_w.unsqueeze(-1)).sum(dim=(1, 2)) / cmd_cnt.unsqueeze(-1)
                cmd_disp = (
                    (pos_ev - cmd_mean[:, None, None, :]).square().sum(dim=-1) * cmd_w
                ).sum(dim=(1, 2)) / cmd_cnt

                # Distinct control groups touched
                grp_onehot = F.one_hot(agrp.clamp(0, 10), num_classes=11).float() * is_grp.float().unsqueeze(-1)
                unique_grps = (grp_onehot.sum(dim=(1, 2)) > 0).float().sum(dim=-1)

                t_vec = torch.stack(
                    [
                        torch.log1p(n_total) / 4.0,
                        torch.log1p(n_cmd) / 4.0,
                        torch.log1p(n_sel) / 4.0,
                        torch.log1p(n_grp) / 3.0,
                        torch.log1p(n_cam) / 3.0,
                        torch.log1p(cam_jumps[:, p_i]) / 2.5,
                        cam_dist[:, p_i].clamp(max=5.0) / 5.0,
                        (cmd_disp + 1e-6).sqrt().clamp(max=2.0),
                        unique_grps / 10.0,
                        (n_cmd > 0).float(),
                        (n_grp > 0).float(),
                        (cam_jumps[:, p_i] > 1).float(),
                        cam_xy[:, p_i, -1, 0],
                        cam_xy[:, p_i, -1, 1],
                        (cam_xy[:, p_i, -1, 0] - cam_xy[:, p_i, 0, 0]),
                        (cam_xy[:, p_i, -1, 1] - cam_xy[:, p_i, 0, 1]),
                    ],
                    dim=-1,
                )  # [B, 16]
                tempo_p.append(t_vec)

                # Flatten top-N_act events (cap to 48 per 5s window for fast constant memory)
                B_sz, nf_sz, A_sz = pm.shape
                flat_m = pm.reshape(B_sz, nf_sz * A_sz)
                K_keep = min(48, nf_sz * A_sz)
                # Prioritize physical commands & group recalls first, then camera/selection
                prio = pm.float() * (1.0 + 2.0 * is_cmd.float() + 1.5 * is_grp.float())
                _, keep_idx = torch.topk(prio.reshape(B_sz, nf_sz * A_sz), k=K_keep, dim=1)

                gather_1d = lambda t: torch.gather(t.reshape(B_sz, nf_sz * A_sz), 1, keep_idx)
                gather_nd = lambda t: torch.gather(
                    t.reshape(B_sz, nf_sz * A_sz, -1),
                    1,
                    keep_idx.unsqueeze(-1).expand(B_sz, K_keep, t.shape[-1]),
                )

                v_keep = gather_1d(flat_m)
                k_keep = gather_1d(akind)
                g_keep = gather_1d(agrp)
                op_keep = gather_1d(agop)
                num_keep = gather_nd(anum)
                pos_keep = gather_nd(pos_ev)
                pl_keep = gather_1d(aplaced).float().unsqueeze(-1)

                emb_ev = self.act_feat_proj(
                    torch.cat(
                        [
                            self.act_kind_emb(k_keep),
                            self.act_grp_emb(g_keep),
                            self.act_gop_emb(op_keep),
                            num_keep,
                            pos_keep,
                            pl_keep,
                        ],
                        dim=-1,
                    )
                )  # [B, K_keep, hidden]
                tok_p.append(emb_ev)
                pos_p.append(pos_keep)
                val_p.append(v_keep)

            tempo_feat = torch.stack(tempo_p, dim=1)  # [B, 2, 16]
            act_tok = torch.stack(tok_p, dim=1)       # [B, 2, K_keep, hidden]
            act_pos = torch.stack(pos_p, dim=1)       # [B, 2, K_keep, 2]
            act_val = torch.stack(val_p, dim=1)       # [B, 2, K_keep]
        else:
            tempo_feat = torch.zeros(B, 2, D_TEMPO_FEAT, device=dev)
            act_tok = torch.zeros(B, 2, 1, self.intent_queries.shape[-1], device=dev)
            act_pos = cam_xy[:, :, -1:, :]
            act_val = torch.ones(B, 2, 1, dtype=torch.bool, device=dev)

        return tempo_feat, act_tok, act_pos, act_val

    def forward(
        self,
        s_tau: dict[str, torch.Tensor],
        s_next: dict[str, torch.Tensor],
        b: dict[str, torch.Tensor],
        ctx: dict[str, torch.Tensor],
        f_indices: list[int],
        ctx_indices: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Computes Hindsight Posterior `(intent_pos, intent_gate, intent_z, intent_macro, xi)`,
        packed flat innovation vector `u_flat` for Flow Matching, causal `tempo_feat`, and
        auxiliary 2D-RoPE pointer grounding loss `aux_loss`."""
        with torch.autocast(device_type=s_tau["q_squad"].device.type, enabled=False):
            B, _, M, _ = s_tau["q_squad"].shape
            dev = s_tau["q_squad"].device

            tempo_feat, act_tok, act_pos, act_val = self.extract_window_tempo_and_tokens(
                b, f_indices
            )

            # Detach SquadPooler states entering HindsightIntentPosterior so u_post does not
            # create a target-to-input shortcut from s_next back into SquadPooler routing.
            q_sq_t = s_tau["q_squad"].detach().float()
            z_sq_t = s_tau["z_squad"].detach().float()
            act_t = s_tau["active"].detach()
            assign_w = s_tau["assign_w"].detach().float()      # [B, 2, M, R]
            q_mac_t = s_tau["q_macro"].detach().float()
            z_mac_t = s_tau["z_macro"].detach().float()

            q_sq_n = s_next["q_squad"].detach().float()
            z_sq_n = s_next["z_squad"].detach().float()
            q_mac_n = s_next["q_macro"].detach().float()
            z_mac_n = s_next["z_macro"].detach().float()

            # Compute which squads in s_tau received newly issued orders during ctx_indices
            if ctx_indices is None:
                q_chg_seq = ctx["q_changed"][:, : len(f_indices)]  # [B, nf, R] bool
            else:
                c_idx = torch.as_tensor(ctx_indices, device=dev, dtype=torch.long)
                q_chg_seq = ctx["q_changed"][:, c_idx]
            unit_ordered = q_chg_seq.any(dim=1).float()        # [B, R]
            mass = assign_w.sum(dim=-1).clamp(min=1e-6)        # [B, 2, M]
            squad_ordered_frac = (
                torch.matmul(assign_w, unit_ordered.unsqueeze(1).unsqueeze(-1)).squeeze(-1) / mass
            ) * act_t.float()  # [B, 2, M] in [0, 1]

            # Encode squad start + end state tokens [B, 2, M, hidden]
            sq_t = torch.cat([q_sq_t, z_sq_t], dim=-1)
            sq_n = torch.cat([q_sq_n, z_sq_n], dim=-1)
            sq_tok = self.sq_enc(
                torch.cat([sq_t, sq_n - sq_t, squad_ordered_frac.unsqueeze(-1)], dim=-1)
            )
            sq_pos = q_sq_t[..., 0:2]  # [B, 2, M, 2]
            sq_val = act_t             # [B, 2, M]

            # Combine squad tokens + spatialized action/camera tokens as KV memory per player
            kv_tok = torch.cat([sq_tok, act_tok.float()], dim=2).reshape(B * 2, M + act_tok.shape[2], -1)
            kv_pos = torch.cat([sq_pos, act_pos.float()], dim=2).reshape(B * 2, M + act_pos.shape[2], 2)
            kv_val = torch.cat([sq_val, act_val], dim=2).reshape(B * 2, M + act_val.shape[2])

            # Ensure at least column 0 is valid (branchless, zero host-device sync)
            safe_mask = ~kv_val
            all_masked = safe_mask.all(dim=-1, keepdim=True)
            safe_mask = torch.cat([safe_mask[:, :1] & (~all_masked), safe_mask[:, 1:]], dim=-1)

            q_in = self.intent_queries[None, :, :].expand(B * 2, self.k_intent, -1)
            attn_out, attn_w = self.intent_attn(
                q_in, kv_tok, kv_tok, key_padding_mask=safe_mask, need_weights=True
            )  # attn_w: [B*2, K, M + K_act]
            h_int = self.intent_norm(q_in + attn_out)  # [B*2, K, hidden]

            # Attention-anchored map coordinate + bounded learned refinement
            base_pos = torch.matmul(attn_w, kv_pos)  # [B*2, K, 2]
            intent_pos = (base_pos + 0.25 * torch.tanh(self.intent_pos_offset(h_int))).clamp(-2.0, 2.0)
            intent_gate = torch.sigmoid(self.intent_gate_head(h_int).squeeze(-1))  # [B*2, K]
            intent_z = torch.tanh(self.intent_z_head(h_int)) * 2.0                # [B*2, K, d_intent]

            intent_pos = intent_pos.view(B, 2, self.k_intent, 2)
            intent_gate = intent_gate.view(B, 2, self.k_intent)
            intent_z = intent_z.view(B, 2, self.k_intent, self.d_intent)

            # Macro intent & micro-execution residual xi
            mac_t = torch.cat([q_mac_t, z_mac_t], dim=-1)
            mac_n = torch.cat([q_mac_n, z_mac_n], dim=-1)
            h_pool = h_int.mean(dim=1).view(B, 2, -1)
            mac_in = torch.cat([mac_t, mac_n - mac_t, h_pool, tempo_feat.float()], dim=-1)

            intent_macro = torch.tanh(self.macro_intent_head(mac_in)) * 2.0  # [B, 2, d_intent]
            xi = torch.tanh(self.xi_head(mac_in)) * 1.5                      # [B, 2, d_xi]

            # Auxiliary 2D-RoPE pointer grounding loss: intents -> ordered squads
            pq = F.normalize(self.ptr_q(intent_z), dim=-1, eps=1e-4).reshape(B * 2, self.k_intent, 64)
            pk = F.normalize(self.ptr_k(sq_t), dim=-1, eps=1e-4).reshape(B * 2, M, 64)
            pq_rot = _apply_rope_2d(pq, intent_pos.reshape(B * 2, self.k_intent, 2) * E.MAP_SCALE, self.ptr_rope)
            pk_rot = _apply_rope_2d(pk, sq_pos.reshape(B * 2, M, 2) * E.MAP_SCALE, self.ptr_rope)

            # Max-pooled intent match per squad modulated by intent_gate
            ptr_scores = torch.matmul(pq_rot, pk_rot.transpose(1, 2)) * self.ptr_scale.exp().clamp(1.0, 30.0)
            gated_scores = ptr_scores + torch.log(intent_gate.reshape(B * 2, self.k_intent, 1).clamp(min=1e-4))
            squad_pred_logit = torch.logsumexp(gated_scores, dim=1).view(B, 2, M)

            target_ord = (squad_ordered_frac > 0.05).float()
            valid_sq = act_t.float()
            bce = F.binary_cross_entropy_with_logits(squad_pred_logit, target_ord, reduction="none")
            aux_ptr_loss = (bce * valid_sq).sum() / valid_sq.sum().clamp(min=1.0)

            # Mild L2 prior regularization + gate sparsity bias so unused intent slots stay quiet
            u_flat = pack_innovation(intent_pos, intent_gate, intent_z, intent_macro, xi)
            reg_loss = 1e-3 * u_flat.square().mean() + 2e-2 * intent_gate.mean()

            return dict(
                intent_pos=intent_pos,
                intent_gate=intent_gate,
                intent_z=intent_z,
                intent_macro=intent_macro,
                xi=xi,
                u_flat=u_flat,
                tempo_feat=tempo_feat.float(),
                squad_ordered_frac=squad_ordered_frac,
                aux_loss=aux_ptr_loss + reg_loss,
                aux_ptr_loss=aux_ptr_loss.detach(),
            )


def innovation_dim(
    k_intent: int = K_INTENT_DEFAULT,
    d_intent: int = D_INTENT_DEFAULT,
    d_xi: int = D_XI_DEFAULT,
) -> int:
    """Total flat dimension of one player's 5s innovation vector `U_{tau, p}`:
    `K * (2 + 1 + d_intent) + d_intent + d_xi` (180 dims for defaults K=4, d_intent=32, d_xi=16)."""
    return k_intent * (3 + d_intent) + d_intent + d_xi


def pack_innovation(
    intent_pos: torch.Tensor,
    intent_gate: torch.Tensor,
    intent_z: torch.Tensor,
    intent_macro: torch.Tensor,
    xi: torch.Tensor,
) -> torch.Tensor:
    """Packs structured intent & micro-residual tensors into `[..., D_u]` in unconstrained
    Euclidean space suitable for Conditional Flow Matching (`gate` mapped via logit)."""
    eps = 1e-4
    g_clamped = intent_gate.clamp(eps, 1.0 - eps)
    gate_logit = (torch.log(g_clamped) - torch.log1p(-g_clamped)) * 0.25  # O(1) scale
    slots_flat = torch.cat(
        [intent_pos, gate_logit.unsqueeze(-1), intent_z], dim=-1
    ).flatten(-2, -1)
    return torch.cat([slots_flat, intent_macro, xi], dim=-1)


def unpack_innovation(
    u_flat: torch.Tensor,
    k_intent: int = K_INTENT_DEFAULT,
    d_intent: int = D_INTENT_DEFAULT,
    d_xi: int = D_XI_DEFAULT,
) -> dict[str, torch.Tensor]:
    """Unpacks a flat `[..., D_u]` vector (from Posterior or Flow-Matching Prior) back into
    structured `(intent_pos, intent_gate, intent_z, intent_macro, xi)`."""
    slot_w = 3 + d_intent
    n_slots = k_intent * slot_w
    slots = u_flat[..., :n_slots].view(*u_flat.shape[:-1], k_intent, slot_w)
    intent_pos = slots[..., 0:2].clamp(-2.0, 2.0)
    intent_gate = torch.sigmoid(slots[..., 2] * 4.0)
    intent_z = slots[..., 3:].clamp(-2.5, 2.5)
    intent_macro = u_flat[..., n_slots : n_slots + d_intent].clamp(-2.5, 2.5)
    xi = u_flat[..., n_slots + d_intent : n_slots + d_intent + d_xi].clamp(-2.0, 2.0)
    return dict(
        intent_pos=intent_pos,
        intent_gate=intent_gate,
        intent_z=intent_z,
        intent_macro=intent_macro,
        xi=xi,
    )
