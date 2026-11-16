"""Layer 1 -> Layer 2 Smooth Perceiver Cross-Attention Compressor & Hindsight Outcome Posterior.

Key properties (Multi-Scale Stochastic Observer v2):
  1. `PerceiverSquadCompressor` (aliased as `SquadPooler`):
     - Compresses a `16`-frame (`16 x 7 = 112` loops = `5.0s`) slice of Layer-1 tokens into
       `1` Macro Token + `M = 64` smooth Perceiver Tokens per player using multi-head 2D-RoPE
       cross-attention over `[L_slice * R + 1]` tokens with a learnable Null-Sink (ZERO `top-k`!).
     - Uses continuous `gate` (`1 - p_sink`) for `C^inf` smooth temporal seeding and exact
       `dt_loops`-scaled velocity advection (eliminating 1-second scrubbing spikes).
     - Preserves exact conservation of `count, cost, hp, shield` via smooth partition-of-unity
       normalization across active queries while allowing specialized queries to lock onto
       high-leverage singletons (`WarpPrismPhasing`, `OracleStasisTrap`) co-located with armies.
     - Separately pools `M_opp = 32` strictly leak-free Perceived-Enemy Tokens (`q_opp_obs`,
       `z_opp_obs`, `gate_opp_obs`) solely from public per-unit observables inside engine
       `gt_vis_live`, `gt_shimmer` (position/spread/shimmer only, `z=0`), and `gt_seen_struct`
       (`0.000000e+00` hidden-enemy or opponent-hotkey/camera leak!).
  2. `HindsightOutcomePosterior` (aliased as `HindsightIntentPosterior`):
     - Infers perspective-faithful player intents `(u_1, u_2)` strictly from each player's own
       information set and commands (`0.000e+00` hidden-enemy leak), PLUS an omniscient `32d`
       world-outcome innovation `xi_world` capturing bimodal combat/snipe/trap resolution and
       realized combat/structural destruction impulses (`combat_loss`).
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
N_OPP_OBS_DEFAULT = 32
D_SQUAD_DEFAULT = 128
D_OPP_Z_DEFAULT = 64
K_INTENT_DEFAULT = 4
D_INTENT_DEFAULT = 32
D_XI_DEFAULT = 16
D_XI_WORLD_DEFAULT = 32
D_TEMPO_FEAT = 16

# 14 grounded physical observables per Squad Token (fp32)
SQUAD_Q_COLS = (
    "x",           # 0: center-of-mass x relative to map center (/ 100.0)
    "y",           # 1: center-of-mass y relative to map center (/ 100.0)
    "vx",          # 2: velocity vx (/ SPEED_SCALE, 1.0 == 5.0 map units/s)
    "vy",          # 3: velocity vy (/ SPEED_SCALE, 1.0 == 5.0 map units/s)
    "spread_x",    # 4: spatial std-dev in x (/ 100.0, >= 0.01)
    "spread_y",    # 5: spatial std-dev in y (/ 100.0, >= 0.01)
    "count",       # 6: conserved unit count (/ N0 = 100.0)
    "cost",        # 7: conserved mineral + vespene header (/ 1000.0, NOT added to H!)
    "hp",          # 8: conserved hitpoints (/ 1000.0)
    "shield",      # 9: conserved shields (/ 1000.0)
    "struct_frac", # 10: structure fraction in [0, 1]
    "air_frac",    # 11: flying unit fraction in [0, 1]
    "cloak_frac",  # 12: cloaked fraction in [0, 1] (engine cloak in {1, 2, 4})
    "det_frac",    # 13: detector fraction in [0, 1] (engine detect_range > 0)
)
SQ = {name: i for i, name in enumerate(SQUAD_Q_COLS)}
N_SQUAD_Q = len(SQUAD_Q_COLS)  # 14

# 12 public observables per Perceived-Enemy Token (strictly view-faithful)
OPP_OBS_Q_COLS = (
    "x",               # 0
    "y",               # 1
    "vx",              # 2 (zeroed unless live-visible & detected)
    "vy",              # 3 (zeroed unless live-visible & detected)
    "spread_x",        # 4
    "spread_y",        # 5
    "count",           # 6 (/ 100.0)
    "hp",              # 7 (/ 1000.0, zeroed on shimmer/fog-snapshot)
    "shield",          # 8 (/ 1000.0, zeroed on shimmer/fog-snapshot)
    "vis_live_frac",   # 9: fully visible & detected
    "shimmer_frac",    # 10: translucent cloak shimmer only (display=3, cloak=1)
    "snap_struct_frac",# 11: explored structure snapshot in fog
)
OQ = {name: i for i, name in enumerate(OPP_OBS_Q_COLS)}
N_OPP_Q = len(OPP_OBS_Q_COLS)  # 12

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

MULE_RTYPE = 268


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


def extract_combat_losses(
    b: dict[str, torch.Tensor],
    b_f_start: int,
    b_f_end: int,
    cost_1k: torch.Tensor,
    is_struct: torch.Tensor,
) -> torch.Tensor:
    """Computes `[B, 2]` realized combat/structural destruction impulses (`1.0 == 1,000`
    mineral+vespene of genuinely destroyed units/buildings) over `(b_f_start, b_f_end]`,
    excluding hallucinations, timed-life MULEs, and uncompleted building self-cancels."""
    dev = b["mask"].device
    B, F_tot, R = b["mask"].shape
    c_buf = cost_1k.to(dev)
    st_buf = is_struct.to(dev)

    if "death_frame" in b:
        df = b["death_frame"]  # [B, R]
        died = (df > b_f_start) & (df <= b_f_end)
        f_pre = (df - 1).clamp(min=0, max=F_tot - 1)  # [B, R]
    else:
        died = b["mask"][:, b_f_start] & b["alive"][:, b_f_start] & (~b["alive"][:, b_f_end])
        f_pre = torch.full((B, R), max(0, b_f_end - 1), device=dev, dtype=torch.long)

    b_idx = torch.arange(B, device=dev)[:, None].expand(B, R)
    r_idx = torch.arange(R, device=dev)[None, :].expand(B, R)
    rtype_pre = b["rtype"][b_idx, f_pre, r_idx].clamp(0, c_buf.shape[0] - 1)
    num_pre = b["num"][b_idx, f_pre, r_idx].float()

    flags = num_pre[..., ED.I["flags"]].round().long()
    is_halluc = (flags & 8) != 0
    is_mule = rtype_pre == MULE_RTYPE
    build_prog = num_pre[..., ED.I["build"]]
    hp = num_pre[..., ED.I["hp"]]
    hp_max = num_pre[..., ED.I["hp_max"]].clamp(min=1.0)
    is_st = st_buf[rtype_pre] > 0.5
    self_cancel = is_st & (build_prog < 0.98) & (hp > 0.65 * hp_max)

    valid_kill = died & (~is_halluc) & (~is_mule) & (~self_cancel)
    unit_val = c_buf[rtype_pre] * valid_kill.float()  # [B, R]

    owner = b["owner"]  # [B, R]: 0=P1, 1=P2
    loss_p1 = (unit_val * (owner == 0).float()).sum(dim=-1)
    loss_p2 = (unit_val * (owner == 1).float()).sum(dim=-1)
    return torch.stack([loss_p1, loss_p2], dim=-1)  # [B, 2]


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


class PerceiverSquadCompressor(nn.Module):
    """Smooth Perceiver Cross-Attention `L1 -> L2` Compressor (`M = 64` queries/player +
    learnable Null-Sink over `[L_slice x R]` tokens, ZERO `top-k`!) plus strictly view-faithful
    Perceived-Enemy Perceiver (`M_opp = 32` queries/player)."""

    def __init__(
        self,
        tables: ED.Tables,
        d_l1: int = 256,
        d_squad: int = D_SQUAD_DEFAULT,
        n_squads: int = N_SQUADS_DEFAULT,
        n_opp_obs: int = N_OPP_OBS_DEFAULT,
        d_opp_z: int = D_OPP_Z_DEFAULT,
        heads: int = 4,
        unit_data_path: str = "data/worldmodel/unit_data.json",
    ):
        super().__init__()
        self.d_l1 = d_l1
        self.d_squad = d_squad
        self.n_squads = n_squads
        self.n_opp_obs = n_opp_obs
        self.d_opp_z = d_opp_z
        self.heads = heads
        self.dh = d_l1 // heads  # 64

        cost_1k, is_struct, is_worker = build_unit_meta_tensors(unit_data_path)
        self.register_buffer("cost_1k", cost_1k, persistent=False)
        self.register_buffer("is_struct", is_struct, persistent=False)
        self.register_buffer("is_worker", is_worker, persistent=False)
        self.register_buffer("is_flying", tables.k[:, 5].float(), persistent=False)
        self.register_buffer("unit_feat_tab", tables.unit_feat.float(), persistent=False)

        self.rope_2d = ED.RoPE2D(self.dh, lo=ED.ROPE_SHORTEST, hi=256.0)

        # 1. Own-Squad Spatio-Role Competitive Perceiver Queries (M = 64, ZERO top-k!)
        # 5 orthogonal physical role channels: [ground_combat, flying, cloak_trap, worker, structure]
        self.slot_queries = nn.Parameter(torch.randn(n_squads, d_l1) * 0.05)
        nn.init.orthogonal_(self.slot_queries)

        def _vogel_disk(n_pts: int, max_r: float, phase: float = 0.0) -> torch.Tensor:
            idx = torch.arange(n_pts, dtype=torch.float32)
            r = max_r * torch.sqrt((idx + 0.5) / max(1, n_pts))
            theta = idx * 2.39996323 + phase
            return torch.stack([r * theta.cos(), r * theta.sin()], dim=-1)

        # Allocate M=64 slots across roles (scaled if n_squads != 64 for unit tests)
        if n_squads == 64:
            role_counts = [28, 10, 6, 8, 12]
        else:
            c0 = max(1, round(n_squads * 28 / 64))
            c1 = max(1, round(n_squads * 10 / 64))
            c2 = max(1, round(n_squads * 6 / 64))
            c3 = max(1, round(n_squads * 8 / 64))
            c4 = max(1, n_squads - (c0 + c1 + c2 + c3))
            role_counts = [c0, c1, c2, c3, c4]
            role_counts[0] += n_squads - sum(role_counts)

        offsets_parts, role_init_parts, sigma_init_parts = [], [], []
        role_radii = [72.0, 70.0, 68.0, 68.0, 70.0]
        role_sigmas = [5.5, 5.8, 5.8, 6.0, 6.2]
        for r_idx, cnt in enumerate(role_counts):
            if cnt <= 0:
                continue
            offsets_parts.append(_vogel_disk(cnt, role_radii[r_idx], phase=r_idx * 0.7))
            rl = torch.full((cnt, 5), -2.5)
            rl[:, r_idx] = 3.0
            role_init_parts.append(rl)
            sigma_init_parts.append(torch.full((cnt,), math.log(role_sigmas[r_idx])))

        init_offsets = torch.cat(offsets_parts, dim=0)[:n_squads]
        role_logits_init = torch.cat(role_init_parts, dim=0)[:n_squads]
        log_sigma_init = torch.cat(sigma_init_parts, dim=0)[:n_squads]

        self.register_buffer("slot_init_offsets", init_offsets, persistent=False)
        self.slot_role_logits = nn.Parameter(role_logits_init)
        self.slot_log_sigma = nn.Parameter(log_sigma_init)
        self.slot_log_gamma = nn.Parameter(torch.full((n_squads,), math.log(5.5)))
        self.slot_prior_bias = nn.Parameter(torch.linspace(0.25, -0.25, n_squads))
        self.time_recency_temp = nn.Parameter(torch.tensor(2.2))

        self.prev_to_query = nn.Sequential(
            nn.LayerNorm(d_squad + N_SQUAD_Q),
            nn.Linear(d_squad + N_SQUAD_Q, d_l1),
            nn.SiLU(),
            nn.Linear(d_l1, d_l1),
        )
        E.small_init(self.prev_to_query[-1], std=2e-2)

        self.slot_refine_mlp = nn.Sequential(
            nn.LayerNorm(d_l1),
            nn.Linear(d_l1, d_l1),
            nn.SiLU(),
            nn.Linear(d_l1, d_l1),
        )
        E.small_init(self.slot_refine_mlp[-1], std=2e-2)

        # Per-(f, r) entity key/value input: c_ent(d_l1) + groups_onscreen(12) + role_flags(6) + rel_time(2)
        ent_in_dim = d_l1 + 12 + 6 + 2
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
        self.cargo_type_emb = nn.Embedding(ED.N_RAW_TYPES, d_squad)
        nn.init.normal_(self.cargo_type_emb.weight, std=0.02)
        self.cargo_k_emb = nn.Embedding(ED.N_RAW_TYPES, d_l1)
        nn.init.normal_(self.cargo_k_emb.weight, std=0.02)

        self.query_norm = nn.LayerNorm(d_l1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(4.0)))

        self.squad_norm = nn.LayerNorm(d_squad)
        self.squad_mlp = nn.Sequential(
            nn.Linear(d_squad + N_SQUAD_Q + 1, d_squad),
            nn.SiLU(),
            nn.Linear(d_squad, d_squad),
        )
        E.small_init(self.squad_mlp[-1], std=2e-2)

        # 2. Strictly View-Faithful Perceived-Enemy Perceiver (M_opp = 32, ZERO c_ent / hidden leak!)
        self.opp_queries = nn.Parameter(torch.randn(n_opp_obs, d_l1) * 0.05)
        nn.init.orthogonal_(self.opp_queries)
        opp_off = _vogel_disk(n_opp_obs, 72.0, phase=0.35)
        self.register_buffer("opp_init_offsets", opp_off, persistent=False)
        self.opp_log_sigma = nn.Parameter(torch.full((n_opp_obs,), math.log(8.5)))
        self.opp_prior_bias = nn.Parameter(torch.linspace(0.20, -0.20, n_opp_obs))

        self.pub_unit_emb = nn.Embedding(ED.N_RAW_TYPES, 64)
        nn.init.normal_(self.pub_unit_emb.weight, std=0.02)
        self.pub_uf_proj = nn.Linear(tables.unit_feat_dim, 64)
        # Public unit features: pub_emb(64) + pub_uf(64) + pub_kin_upg(11) + perc_mode(3) = 142
        pub_in_dim = 64 + 64 + 11 + 3
        self.opp_k = nn.Sequential(
            nn.LayerNorm(pub_in_dim),
            nn.Linear(pub_in_dim, d_l1, bias=False),
        )
        self.opp_v = nn.Sequential(
            nn.LayerNorm(pub_in_dim),
            nn.Linear(pub_in_dim, d_opp_z),
            nn.SiLU(),
            nn.Linear(d_opp_z, d_opp_z),
        )
        self.opp_q_norm = nn.LayerNorm(d_l1)
        self.opp_refine_mlp = nn.Sequential(
            nn.LayerNorm(d_l1),
            nn.Linear(d_l1, d_l1),
            nn.SiLU(),
            nn.Linear(d_l1, d_l1),
        )
        E.small_init(self.opp_refine_mlp[-1], std=2e-2)

        # 3. Perspective-Isolated Macro Token projection (NO omniscient cls_out!)
        self.macro_proj = nn.Sequential(
            nn.LayerNorm(3 * d_l1 + N_MACRO_Q),
            nn.Linear(3 * d_l1 + N_MACRO_Q, d_squad),
            nn.SiLU(),
            nn.Linear(d_squad, d_squad),
        )
        # Separate Omniscient Scene Token projection from cls_out for Observer-only h_obs / xi_world
        self.omni_scene_proj = nn.Sequential(
            nn.LayerNorm(2 * d_l1),
            nn.Linear(2 * d_l1, d_squad),
            nn.SiLU(),
            nn.Linear(d_squad, d_squad),
        )

    def _extract_macro_q(self, b: dict[str, torch.Tensor], f_end: int) -> torch.Tensor:
        """Extracts `[B, 2, 6]` grounded player macro observables at frame `f_end`."""
        with torch.autocast(device_type=b["ply_num"].device.type, enabled=False):
            pl = b["ply_num"][:, f_end].float()
            n_pc = len(ED.PC_KEYS)
            v = torch.expm1(pl[..., :n_pc] * 8.0)
            mins = v[..., ED.PC["minerals"]] / E.BANK_SCALE
            vesp = v[..., ED.PC["vespene"]] / E.BANK_SCALE
            food_used = v[..., ED.PC["food_used"]] / E.N0
            food_cap = v[..., ED.PC["food_cap"]] / E.N0

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

    def _pool_perceived_enemies(
        self,
        b: dict[str, torch.Tensor],
        ctx: dict[str, torch.Tensor],
        f_end: int,
        batch_frame_end: int,
        pos_map_end: torch.Tensor,
        center: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pools strictly view-faithful enemy observations `(q_opp_obs, z_opp_obs, gate_opp_obs)`
        for each player `p` over enemy `1 - p`'s end-frame entities where `gt_vis_live`,
        `gt_shimmer`, or `gt_seen_struct` holds (`0.000000e+00` hidden-enemy leak!)."""
        dev = pos_map_end.device
        B, R, _ = pos_map_end.shape
        H, dh, M_o = self.heads, self.dh, self.n_opp_obs

        perc = ctx["perception"]
        vis_live_all = perc["gt_vis_live"][:, f_end]      # [B, 2, R]
        shimmer_all = perc["gt_shimmer"][:, f_end]        # [B, 2, R]
        snap_all = perc["gt_seen_struct"][:, f_end]       # [B, 2, R]

        rtype = b["rtype"][:, batch_frame_end].clamp(0, ED.N_RAW_TYPES - 1)
        num = b["num"][:, batch_frame_end].float()
        q_end = ctx["q"][:, f_end].float()
        owner_raw = b["owner"]  # 0=P1, 1=P2

        x_n = (pos_map_end[..., 0] / E.MAP_SCALE).clamp(-2.0, 2.0)
        y_n = (pos_map_end[..., 1] / E.MAP_SCALE).clamp(-2.0, 2.0)
        vx_n = q_end[..., E.Q["vx"]]
        vy_n = q_end[..., E.Q["vy"]]
        hp_1k = num[..., ED.I["hp"]].clamp(min=0.0) / E.BANK_SCALE
        sh_1k = num[..., ED.I["shield"]].clamp(min=0.0) / E.BANK_SCALE
        en_f = q_end[..., E.Q["energy"]]
        facing = num[..., ED.I["facing"]]
        upg = torch.stack(
            [
                num[..., ED.I["atk_up"]] / 3.0,
                num[..., ED.I["arm_up"]] / 3.0,
                num[..., ED.I["shd_up"]] / 3.0,
            ],
            dim=-1,
        )  # [B, R, 3]

        pub_type_128 = torch.cat(
            [self.pub_unit_emb(rtype), self.pub_uf_proj(self.unit_feat_tab[rtype])], dim=-1
        )  # [B, R, 128]

        scale = self.logit_scale.exp().clamp(1.5, 20.0)
        sigma_o = self.opp_log_sigma.exp().clamp(4.0, 18.0)       # [M_o]
        sigma_o0 = (2.6 * sigma_o).clamp(15.0, 34.0)              # [M_o] coarse capture radius
        upper_o = torch.triu(torch.ones(M_o, M_o, device=dev, dtype=torch.float32), diagonal=1)

        q_opp_list, z_opp_list, g_opp_list = [], [], []

        for p_i in (0, 1):
            opp = 1 - p_i
            is_opp = owner_raw == opp
            vl = vis_live_all[:, p_i] & is_opp
            sh = shimmer_all[:, p_i] & is_opp & (~vl)
            sn = snap_all[:, p_i] & is_opp & (~vl) & (~sh)
            any_perc = vl | sh | sn  # [B, R]

            vl_f = vl.float()
            sh_f = sh.float()
            sn_f = sn.float()
            id_known_f = (vl | sn).float()  # unit type known on live sight or structure snapshot, ZERO on shimmer!

            # Zero out EVERY unperceived unit feature BEFORE linear projections so masked units have 0.0 grad!
            any_f = any_perc.float()
            pos_perc = pos_map_end * any_f.unsqueeze(-1)
            type_in = pub_type_128 * id_known_f.unsqueeze(-1)
            kin_upg_in = torch.cat(
                [
                    (x_n * any_f).unsqueeze(-1),
                    (y_n * any_f).unsqueeze(-1),
                    (vx_n * (vl_f + sh_f)).unsqueeze(-1),
                    (vy_n * (vl_f + sh_f)).unsqueeze(-1),
                    (hp_1k * vl_f).unsqueeze(-1),
                    (sh_1k * vl_f).unsqueeze(-1),
                    (en_f * vl_f).unsqueeze(-1),
                    (facing.sin() * vl_f).unsqueeze(-1),
                    (facing.cos() * vl_f).unsqueeze(-1),
                    upg[..., 0:2] * vl_f.unsqueeze(-1),
                ],
                dim=-1,
            )  # [B, R, 11]
            mode_in = torch.stack([vl_f, sh_f, sn_f], dim=-1)  # [B, R, 3]
            pub_in = torch.cat([type_in, kin_upg_in, mode_in], dim=-1) * any_f.unsqueeze(-1)

            ko = F.normalize(self.opp_k(pub_in), dim=-1, eps=1e-4) * any_f.unsqueeze(-1)
            vo = self.opp_v(pub_in) * id_known_f.unsqueeze(-1)  # Shimmer gives z_opp == 0.0!

            with torch.autocast(device_type=dev.type, enabled=False):
                pos_p32 = pos_perc.float()
                ko_rot = _apply_rope_2d(ko.float(), pos_p32, self.rope_2d)  # [B, R, D]

                # Anchor around map center (0, 0) in map-centered coords (0.0 hidden-enemy dependence!)
                qo_pos_0 = self.opp_init_offsets[None, :, :].expand(B, M_o, 2).float()
                qo_vec_0 = F.normalize(self.opp_q_norm(self.opp_queries[None].expand(B, M_o, -1)), dim=-1, eps=1e-4).float()
                qo_rot_0 = _apply_rope_2d(qo_vec_0, qo_pos_0, self.rope_2d)

                dist2_o0 = (qo_pos_0.unsqueeze(2) - pos_p32.unsqueeze(1)).square().sum(dim=-1)
                sem_o0 = 2.0 * torch.tanh(torch.matmul(qo_rot_0, ko_rot.transpose(-1, -2)) * (scale * 0.25))
                logits_o0 = (
                    -dist2_o0 / (2.0 * sigma_o0[None, :, None].square())
                    + sem_o0
                    + self.opp_prior_bias[None, :, None].float()
                )
                # Competitive softmax across M_o slots per perceived entity (independent across r -> 0.0 hidden leak!)
                part_o0 = F.softmax(logits_o0, dim=1) * any_f.unsqueeze(1)  # [B, M_o, R]
                mass_o0 = part_o0.sum(dim=-1)                               # [B, M_o]
                pull_o0 = mass_o0 / (mass_o0 + 0.15)
                cent_o0 = (part_o0.unsqueeze(-1) * pos_p32.unsqueeze(1)).sum(dim=2) / mass_o0.unsqueeze(-1).clamp(min=1e-4)
                qo_pos_1 = pull_o0.unsqueeze(-1) * cent_o0 + (1.0 - pull_o0.unsqueeze(-1)) * qo_pos_0

                k_pool_o0 = torch.matmul(part_o0, ko.float()) / mass_o0.unsqueeze(-1).clamp(min=1e-4)
                qo_vec_1 = F.normalize(
                    self.opp_q_norm(qo_vec_0 + pull_o0.unsqueeze(-1) * self.opp_refine_mlp(k_pool_o0).float()),
                    dim=-1,
                    eps=1e-4,
                )
                qo_rot_1 = _apply_rope_2d(qo_vec_1, qo_pos_1, self.rope_2d)

                # Smooth upper-triangular duplicate suppression for co-located perceived enemy slots
                slot_d2_o = (qo_pos_1.unsqueeze(2) - qo_pos_1.unsqueeze(1)).square().sum(dim=-1)
                dup_pen_o = (pull_o0.unsqueeze(2) * torch.exp(-slot_d2_o / 18.0) * upper_o.unsqueeze(0)).sum(dim=1)

                dist2_o1 = (qo_pos_1.unsqueeze(2) - pos_p32.unsqueeze(1)).square().sum(dim=-1)
                sem_o1 = 2.5 * torch.tanh(torch.matmul(qo_rot_1, ko_rot.transpose(-1, -2)) * (scale * 0.25))
                logits_o1 = (
                    -dist2_o1 / (2.0 * sigma_o[None, :, None].square())
                    + sem_o1
                    + self.opp_prior_bias[None, :, None].float()
                    - 5.0 * dup_pen_o.unsqueeze(-1)
                )

                part_o = F.softmax(logits_o1, dim=1) * any_f.unsqueeze(1)   # [B, M_o, R]: exact 0.0 on hidden!
                mass_o = part_o.sum(dim=-1)                                 # [B, M_o]
                gate_o = (1.0 - torch.exp(-2.5 * mass_o)).clamp(0.0, 1.0)
                w_denom = mass_o.clamp(min=1e-5)

                xm = (part_o * (x_n * any_f).unsqueeze(1)).sum(dim=-1) / w_denom
                ym = (part_o * (y_n * any_f).unsqueeze(1)).sum(dim=-1) / w_denom
                vxm = (part_o * (vx_n * (vl_f + sh_f)).unsqueeze(1)).sum(dim=-1) / w_denom
                vym = (part_o * (vy_n * (vl_f + sh_f)).unsqueeze(1)).sum(dim=-1) / w_denom
                var_x = (part_o * ((x_n * any_f).unsqueeze(1) - xm.unsqueeze(-1)).square()).sum(dim=-1) / w_denom
                var_y = (part_o * ((y_n * any_f).unsqueeze(1) - ym.unsqueeze(-1)).square()).sum(dim=-1) / w_denom
                spx = (var_x + 1e-4).sqrt() * gate_o
                spy = (var_y + 1e-4).sqrt() * gate_o

                cnt_o = mass_o / E.N0
                hp_o = (part_o * (hp_1k * vl_f).unsqueeze(1)).sum(dim=-1)
                sh_o = (part_o * (sh_1k * vl_f).unsqueeze(1)).sum(dim=-1)
                vl_frac = (part_o * vl_f.unsqueeze(1)).sum(dim=-1) / w_denom
                sh_frac = (part_o * sh_f.unsqueeze(1)).sum(dim=-1) / w_denom
                sn_frac = (part_o * sn_f.unsqueeze(1)).sum(dim=-1) / w_denom

                q_o = torch.stack(
                    [
                        xm * gate_o,
                        ym * gate_o,
                        vxm * gate_o,
                        vym * gate_o,
                        spx,
                        spy,
                        cnt_o,
                        hp_o,
                        sh_o,
                        vl_frac * gate_o,
                        sh_frac * gate_o,
                        sn_frac * gate_o,
                    ],
                    dim=-1,
                )  # [B, M_o, 12]
                z_o = (torch.matmul(part_o, vo.float()) / w_denom.unsqueeze(-1)) * gate_o.unsqueeze(-1)
            q_opp_list.append(q_o)
            z_opp_list.append(z_o)
            g_opp_list.append(gate_o)

        return (
            torch.stack(q_opp_list, dim=1),
            torch.stack(z_opp_list, dim=1),
            torch.stack(g_opp_list, dim=1),
        )

    def forward(
        self,
        b: dict[str, torch.Tensor],
        ctx: dict[str, torch.Tensor],
        f_start: int = 0,
        f_end: int = -1,
        prev_state: dict[str, torch.Tensor] | None = None,
        batch_frame_end: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compresses `L1` slice `[f_start .. f_end]` (`<= 16` frames = `5.0s` at `step=7`)
        into smooth competitive Spatio-Role Perceiver Squad & Macro tokens + leak-free
        Perceived-Enemy tokens (`C^infty` smooth, ZERO `top-k`!)."""
        c_ent_seq = ctx["c_ent"]  # [B, nf, R, d_l1]
        B, nf, R, _ = c_ent_seq.shape
        if f_end < 0:
            f_end = nf + f_end
        f_start = max(0, min(f_start, f_end))
        if batch_frame_end is None:
            batch_frame_end = f_end
        L_sl = f_end - f_start + 1
        batch_frame_start = batch_frame_end - (L_sl - 1)

        dev = c_ent_seq.device
        M = self.n_squads

        # Slice [f_start .. f_end] across L_sl <= 16 frames
        c_sl = c_ent_seq[:, f_start : f_end + 1].float()  # [B, L_sl, R, d_l1]
        b_f_indices = list(range(max(0, batch_frame_start), batch_frame_end + 1))
        if len(b_f_indices) < L_sl:
            b_f_indices = [b_f_indices[0]] * (L_sl - len(b_f_indices)) + b_f_indices
        bf_t = torch.as_tensor(b_f_indices, device=dev, dtype=torch.long)

        owner_raw = b["owner"]  # [B, R]: 0=P1, 1=P2, NEUTRAL
        alive_sl = b["mask"][:, bf_t] & b["alive"][:, bf_t]  # [B, L_sl, R]
        inside_sl = b["inside"][:, bf_t]                     # [B, L_sl, R]
        rtype_sl = b["rtype"][:, bf_t].clamp(0, ED.N_RAW_TYPES - 1)
        num_sl = b["num"][:, bf_t].float()

        if "center" in b:
            center = b["center"][:, None, None, :].float()  # [B, 1, 1, 2]
        else:
            center = torch.full((B, 1, 1, 2), 100.0, device=dev)
        pos_sl = (
            torch.stack([b["x"][:, bf_t].float(), b["y"][:, bf_t].float()], dim=-1) - center
        )  # [B, L_sl, R, 2]

        onscreen_sl = ((num_sl[..., ED.I["flags2"]].round().long() & 4) != 0).float().unsqueeze(-1)
        if "groups" in b:
            grp_sl = torch.cat([b["groups"][:, bf_t].float(), onscreen_sl], dim=-1)
        else:
            grp_sl = torch.cat([torch.zeros(B, L_sl, R, 11, device=dev), onscreen_sl], dim=-1)

        u_cost_sl = self.cost_1k[rtype_sl]
        u_struct_sl = self.is_struct[rtype_sl]
        u_worker_sl = self.is_worker[rtype_sl]
        u_air_sl = self.is_flying[rtype_sl]
        clk_raw_sl = num_sl[..., ED.I["cloak"]].round().long()
        u_cloak_sl = ((clk_raw_sl == 1) | (clk_raw_sl == 2) | (clk_raw_sl == 4)).float()
        u_det_sl = (num_sl[..., ED.I["detect_range"]] > 0.5).float()
        role_sl = torch.stack(
            [
                u_struct_sl,
                u_worker_sl,
                u_air_sl,
                torch.log1p(u_cost_sl * 10.0),
                u_cloak_sl,
                u_det_sl,
            ],
            dim=-1,
        )  # [B, L_sl, R, 6]

        rel_idx = torch.linspace(-1.0, 0.0, L_sl, device=dev, dtype=torch.float32)
        recency = torch.exp(rel_idx * 2.0)  # [L_sl], peaks at 1.0 on f_end
        rel_feat = torch.stack([rel_idx, recency], dim=-1)[None, :, None, :].expand(B, L_sl, R, 2)

        ent_in = torch.cat([c_sl, grp_sl, role_sl, rel_feat], dim=-1)  # [B, L_sl, R, ent_in_dim]
        k_onmap = F.normalize(self.ent_k(ent_in), dim=-1, eps=1e-4)
        v_onmap = self.ent_v(ent_in)
        k_cargo = F.normalize(self.cargo_k_emb(rtype_sl), dim=-1, eps=1e-4)
        v_cargo = self.cargo_type_emb(rtype_sl)
        is_map_sl = (inside_sl == 0).unsqueeze(-1)
        k_tok = torch.where(is_map_sl, k_onmap, k_cargo)  # [B, L_sl, R, d_l1]
        v_tok = torch.where(is_map_sl, v_onmap, v_cargo)  # [B, L_sl, R, d_squad]

        # End-of-slice physical observables at f_end (fp32)
        pos_end = pos_sl[:, -1]       # [B, R, 2]
        q_unit_end = ctx["q"][:, f_end].float()
        inside_end = inside_sl[:, -1]
        alive_end = alive_sl[:, -1]
        num_end = num_sl[:, -1]
        u_cost_end = u_cost_sl[:, -1]
        u_struct_end = u_struct_sl[:, -1]
        u_worker_end = u_worker_sl[:, -1]
        u_air_end = u_air_sl[:, -1]
        u_cloak_end = u_cloak_sl[:, -1]
        u_det_end = u_det_sl[:, -1]
        flags_end = num_end[..., ED.I["flags"]].round().long()
        burrow_unit_end = (((flags_end & 2) != 0) & (u_struct_end < 0.5)).float()

        cur_loop = (
            b["loops"][:, batch_frame_end].float()
            if "loops" in b
            else torch.full((B,), float(batch_frame_end * 7), device=dev)
        )

        with torch.autocast(device_type=dev.type, enabled=False):
            # 1. Smooth temporal recency pooling per entity r across L_sl <= 16 frames -> [B, R, D]
            t_temp = self.time_recency_temp.float().clamp(0.5, 6.0)
            t_logits = (rel_idx * t_temp)[None, :, None].expand(B, L_sl, R).masked_fill(~alive_sl, -1e9)
            t_w = F.softmax(t_logits, dim=1) * alive_sl.float()
            t_w = t_w / t_w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            k_ent = F.normalize((k_tok.float() * t_w.unsqueeze(-1)).sum(dim=1), dim=-1, eps=1e-4)  # [B, R, d_l1]
            v_ent = (v_tok.float() * t_w.unsqueeze(-1)).sum(dim=1)                                 # [B, R, d_squad]
            k_rot = _apply_rope_2d(k_ent, pos_end.float(), self.rope_2d)                           # [B, R, d_l1]

            # 2. Orthogonal 5D physical role vector per entity: [ground_combat, flying, cloak_trap, worker, structure]
            r_clk = ((u_cloak_end > 0.5) | (burrow_unit_end > 0.5)).float()
            r_wk = ((u_worker_end > 0.5) & (r_clk < 0.5)).float()
            r_fly = ((u_air_end > 0.5) & (r_clk < 0.5) & (r_wk < 0.5)).float()
            r_st = ((u_struct_end > 0.5) & (r_clk < 0.5) & (r_wk < 0.5) & (r_fly < 0.5)).float()
            r_gnd = (1.0 - (r_clk + r_wk + r_fly + r_st)).clamp(0.0, 1.0)
            role_5d = torch.stack([r_gnd, r_fly, r_clk, r_wk, r_st], dim=-1)  # [B, R, 5]

            q_role = F.softmax(self.slot_role_logits.float(), dim=-1)         # [M, 5]
            role_d2 = (q_role[None, :, None, :] - role_5d[:, None, :, :]).square().sum(dim=-1)  # [B, M, R]
            role_sim = (q_role.unsqueeze(1) * q_role.unsqueeze(0)).sum(dim=-1)                  # [M, M]
            upper_m = torch.triu(torch.ones(M, M, device=dev, dtype=torch.float32), diagonal=1)

            sigma = self.slot_log_sigma.float().exp().clamp(3.5, 14.0)        # [M] sharp local radius
            sigma_m = (1.65 * sigma).clamp(7.5, 20.0)                         # [M] medium capture radius
            sigma_0 = (3.2 * sigma).clamp(16.0, 34.0)                         # [M] coarse capture radius
            gamma = self.slot_log_gamma.float().exp().clamp(3.0, 12.0)        # [M]
            prior_b = self.slot_prior_bias.float()[None, :, None]             # [1, M, 1]
            scale = self.logit_scale.float().exp().clamp(min=1.5, max=20.0)

            x_norm = (pos_end[..., 0] / E.MAP_SCALE).clamp(-2.0, 2.0)
            y_norm = (pos_end[..., 1] / E.MAP_SCALE).clamp(-2.0, 2.0)
            vx_norm = torch.where(inside_end == 0, q_unit_end[..., E.Q["vx"]], torch.zeros_like(x_norm))
            vy_norm = torch.where(inside_end == 0, q_unit_end[..., E.Q["vy"]], torch.zeros_like(y_norm))
            hp_1k = num_end[..., ED.I["hp"]].clamp(min=0.0) / E.BANK_SCALE
            sh_1k = num_end[..., ED.I["shield"]].clamp(min=0.0) / E.BANK_SCALE

            q_squad_list, z_squad_list, gate_list, active_list, weights_list = [], [], [], [], []

            for p_idx in (0, 1):
                p_own = owner_raw == p_idx                         # [B, R]
                p_alive_end = alive_end & p_own                    # [B, R]
                p_alive_f = p_alive_end.float()
                pos_f = pos_end.float()

                # Map-wide Vogel disk anchors around map center (0, 0) (invariant to remote unit births/deaths!)
                base_pos = self.slot_init_offsets[None, :, :].expand(B, M, 2).float()
                base_vec = self.slot_queries[None, :, :].expand(B, M, -1).float()

                if prev_state is not None:
                    if "gate" in prev_state:
                        prev_gate = prev_state["gate"][:, p_idx].float().clamp(0.0, 1.0)
                    else:
                        prev_gate = prev_state["active"][:, p_idx].float()
                    w_seed = prev_gate.unsqueeze(-1)
                    g_safe = prev_gate.unsqueeze(-1).clamp(min=0.05)
                    prev_q = prev_state["q_squad"][:, p_idx].float()
                    prev_z = prev_state["z_squad"][:, p_idx].float()
                    if "loop_end" in prev_state and "loops" in b:
                        dt_loops = (cur_loop - prev_state["loop_end"].float()).clamp(0.0, 224.0)[:, None, None]
                    else:
                        dt_loops = 112.0
                    # Divide stored (xm * gate, vxm * gate) by prev_gate before unscaling!
                    prev_xy = (prev_q[..., 0:2] / g_safe) * E.MAP_SCALE
                    prev_vel = (prev_q[..., 2:4] / g_safe) * E.SPEED_SCALE
                    adv_pos = (prev_xy + prev_vel * dt_loops).clamp(-120.0, 120.0)
                    q_pos_0 = w_seed * adv_pos + (1.0 - w_seed) * base_pos
                    q_vec_0 = base_vec + w_seed * self.prev_to_query(torch.cat([prev_z, prev_q], dim=-1)).float()
                else:
                    q_pos_0 = base_pos
                    q_vec_0 = base_vec

                q_vec_0 = F.normalize(self.query_norm(q_vec_0).float(), dim=-1, eps=1e-4)
                q_rot_0 = _apply_rope_2d(q_vec_0, q_pos_0, self.rope_2d)

                # Pass 0: Coarse Spatio-Role Capture across M slots per living entity
                dist2_0 = (q_pos_0.unsqueeze(2) - pos_f.unsqueeze(1)).square().sum(dim=-1)
                sem_0 = 1.5 * torch.tanh(torch.matmul(q_rot_0, k_rot.transpose(-1, -2)) * (scale * 0.25))
                logits_0 = (
                    -dist2_0 / (2.0 * sigma_0[None, :, None].square())
                    - gamma[None, :, None] * role_d2
                    + sem_0
                    + prior_b
                )
                part_0 = F.softmax(logits_0, dim=1) * p_alive_f.unsqueeze(1)  # [B, M, R]
                mass_0 = part_0.sum(dim=-1)
                pull_0 = mass_0 / (mass_0 + 0.03)
                cent_0 = (part_0.unsqueeze(-1) * pos_f.unsqueeze(1)).sum(dim=2) / mass_0.unsqueeze(-1).clamp(min=1e-4)
                q_pos_m = pull_0.unsqueeze(-1) * cent_0 + (1.0 - pull_0.unsqueeze(-1)) * q_pos_0

                # Pass 1: Medium-Radius Snap + Initial Coalescing
                slot_d2_m = (q_pos_m.unsqueeze(2) - q_pos_m.unsqueeze(1)).square().sum(dim=-1)
                dup_pen_m = (pull_0.unsqueeze(2) * role_sim.unsqueeze(0) * torch.exp(-slot_d2_m / 36.0) * upper_m.unsqueeze(0)).sum(dim=1)
                dist2_m = (q_pos_m.unsqueeze(2) - pos_f.unsqueeze(1)).square().sum(dim=-1)
                logits_m = (
                    -dist2_m / (2.0 * sigma_m[None, :, None].square())
                    - gamma[None, :, None] * role_d2
                    + sem_0
                    + prior_b
                    - 4.5 * dup_pen_m.unsqueeze(-1)
                )
                part_m = F.softmax(logits_m, dim=1) * p_alive_f.unsqueeze(1)
                mass_m = part_m.sum(dim=-1)
                pull_m = mass_m / (mass_m + 0.03)
                cent_m = (part_m.unsqueeze(-1) * pos_f.unsqueeze(1)).sum(dim=2) / mass_m.unsqueeze(-1).clamp(min=1e-4)
                q_pos_1 = pull_m.unsqueeze(-1) * cent_m + (1.0 - pull_m.unsqueeze(-1)) * q_pos_m

                k_pool_1 = torch.matmul(part_m, k_ent) / mass_m.unsqueeze(-1).clamp(min=1e-4)
                q_vec_1 = F.normalize(
                    self.query_norm(q_vec_0 + pull_m.unsqueeze(-1) * self.slot_refine_mlp(k_pool_1).float()).float(),
                    dim=-1,
                    eps=1e-4,
                )
                q_rot_1 = _apply_rope_2d(q_vec_1, q_pos_1, self.rope_2d)

                # Pass 2: Sharp Local Spatio-Role Partition-of-Unity + C^infty Coalescing (ZERO top-k!)
                slot_d2 = (q_pos_1.unsqueeze(2) - q_pos_1.unsqueeze(1)).square().sum(dim=-1)  # [B, M, M]
                dup_overlap = pull_m.unsqueeze(2) * role_sim.unsqueeze(0) * torch.exp(-slot_d2 / 32.0)
                dup_pen = (dup_overlap * upper_m.unsqueeze(0)).sum(dim=1)                     # [B, M]

                dist2_1 = (q_pos_1.unsqueeze(2) - pos_f.unsqueeze(1)).square().sum(dim=-1)
                sem_1 = 2.5 * torch.tanh(torch.matmul(q_rot_1, k_rot.transpose(-1, -2)) * (scale * 0.25))
                logits_1 = (
                    -dist2_1 / (2.0 * sigma[None, :, None].square())
                    - gamma[None, :, None] * role_d2
                    + sem_1
                    + prior_b
                    - 7.0 * dup_pen.unsqueeze(-1)
                )

                part_w = F.softmax(logits_1, dim=1) * p_alive_f.unsqueeze(1)  # [B, M, R]
                mass = part_w.sum(dim=-1)                                     # [B, M] conserved unit count in slot
                gate = (1.0 - torch.exp(-2.5 * mass)).clamp(0.0, 1.0)         # [B, M] smooth in [0, 1]
                active = (gate > 0.05) & (mass > 0.02)
                w_denom = mass.clamp(min=1e-5)

                z_raw = torch.matmul(part_w, v_ent) / w_denom.unsqueeze(-1)

                xm = (part_w * x_norm.unsqueeze(1)).sum(dim=-1) / w_denom
                ym = (part_w * y_norm.unsqueeze(1)).sum(dim=-1) / w_denom
                vxm = (part_w * vx_norm.unsqueeze(1)).sum(dim=-1) / w_denom
                vym = (part_w * vy_norm.unsqueeze(1)).sum(dim=-1) / w_denom

                var_x = (part_w * (x_norm.unsqueeze(1) - xm.unsqueeze(-1)).square()).sum(dim=-1) / w_denom
                var_y = (part_w * (y_norm.unsqueeze(1) - ym.unsqueeze(-1)).square()).sum(dim=-1) / w_denom
                spx = (var_x + 1e-4).sqrt() * gate
                spy = (var_y + 1e-4).sqrt() * gate

                cnt_sum = mass / E.N0
                cost_sum = (part_w * u_cost_end.unsqueeze(1)).sum(dim=-1)
                hp_sum = (part_w * hp_1k.unsqueeze(1)).sum(dim=-1)
                sh_sum = (part_w * sh_1k.unsqueeze(1)).sum(dim=-1)
                st_frac = ((part_w * u_struct_end.unsqueeze(1)).sum(dim=-1) / w_denom) * gate
                air_frac = ((part_w * u_air_end.unsqueeze(1)).sum(dim=-1) / w_denom) * gate
                clk_frac = ((part_w * u_cloak_end.unsqueeze(1)).sum(dim=-1) / w_denom) * gate
                det_frac = ((part_w * u_det_end.unsqueeze(1)).sum(dim=-1) / w_denom) * gate

                q_sq = torch.stack(
                    [
                        xm * gate,
                        ym * gate,
                        vxm * gate,
                        vym * gate,
                        spx,
                        spy,
                        cnt_sum,
                        cost_sum,
                        hp_sum,
                        sh_sum,
                        st_frac,
                        air_frac,
                        clk_frac,
                        det_frac,
                    ],
                    dim=-1,
                )  # [B, M, 14]

                z_normed = self.squad_norm(z_raw).float() * gate.unsqueeze(-1)
                z_sq = (
                    z_normed
                    + self.squad_mlp(torch.cat([z_normed, q_sq, gate.unsqueeze(-1)], dim=-1)).float()
                ) * gate.unsqueeze(-1)

                q_squad_list.append(q_sq)
                z_squad_list.append(z_sq)
                gate_list.append(gate)
                active_list.append(active)
                weights_list.append(part_w)

            q_squad = torch.stack(q_squad_list, dim=1)        # [B, 2, M, 14]
            z_squad = torch.stack(z_squad_list, dim=1)        # [B, 2, M, d_squad]
            gate_squad = torch.stack(gate_list, dim=1)        # [B, 2, M]
            active_squad = torch.stack(active_list, dim=1)    # [B, 2, M]
            assign_w = torch.stack(weights_list, dim=1)       # [B, 2, M, R]

            # Strictly leak-free Perceived-Enemy Perceiver tokens [B, 2, M_opp, ...]
            q_opp_obs, z_opp_obs, gate_opp_obs = self._pool_perceived_enemies(
                b, ctx, f_end, batch_frame_end, pos_end, center.squeeze(2)
            )

            # Perspective-isolated Macro Token [B, 2, 6] + [B, 2, d_squad] (ZERO cls_out!)
            q_macro = self._extract_macro_q(b, batch_frame_end)
            ply_sl = ctx["ply_out"][:, f_start : f_end + 1].float()  # [B, L_sl, 2, d_l1]
            ply_end = ply_sl[:, -1]
            ply_start = ply_sl[:, 0]
            ply_mean = ply_sl.mean(dim=1)
            macro_in = torch.cat([ply_end, ply_mean, ply_end - ply_start, q_macro], dim=-1)
            z_macro = torch.tanh(self.macro_proj(macro_in).float()) * 2.0

            # Separate Omniscient Scene Token (Observer-only, NEVER passed into h^(1) or h^(2)!)
            cls_sl = ctx["cls_out"][:, f_start : f_end + 1].float()  # [B, L_sl, d_l1]
            z_omni_scene = torch.tanh(
                self.omni_scene_proj(torch.cat([cls_sl[:, -1], cls_sl.mean(dim=1)], dim=-1)).float()
            ) * 2.0  # [B, d_squad]

            return dict(
                q_squad=q_squad,
                z_squad=z_squad,
                gate=gate_squad,
                active=active_squad,
                q_macro=q_macro,
                z_macro=z_macro,
                q_opp_obs=q_opp_obs,
                z_opp_obs=z_opp_obs,
                gate_opp_obs=gate_opp_obs,
                z_omni_scene=z_omni_scene,
                assign_w=assign_w,
                loop_end=cur_loop,
            )


class HindsightOutcomePosterior(nn.Module):
    """Infers:
      1. Perspective-faithful 5.0s player intents `u_flat` (`[B, 2, D_u]`): strictly from each
         player's own squads, perceived enemy `opp_obs`, and own commands/camera/hotkeys (`0.0`
         hidden-enemy leak!).
      2. Omniscient `32d` world-outcome innovation `xi_world` (`[B, D_xi_world]`): captures
         stochastic multi-unit combat/trap/snipe outcomes across `S_tau -> S_{tau+1}`.
      3. Realized `5.0s` combat/structural destruction impulses `combat_loss` (`[B, 2]`)."""

    def __init__(
        self,
        d_squad: int = D_SQUAD_DEFAULT,
        d_opp_z: int = D_OPP_Z_DEFAULT,
        k_intent: int = K_INTENT_DEFAULT,
        d_intent: int = D_INTENT_DEFAULT,
        d_xi: int = D_XI_DEFAULT,
        d_xi_world: int = D_XI_WORLD_DEFAULT,
        hidden: int = 256,
        unit_data_path: str = "data/worldmodel/unit_data.json",
    ):
        super().__init__()
        self.d_squad = d_squad
        self.d_opp_z = d_opp_z
        self.k_intent = k_intent
        self.d_intent = d_intent
        self.d_xi = d_xi
        self.d_xi_world = d_xi_world

        cost_1k, is_struct, _ = build_unit_meta_tensors(unit_data_path)
        self.register_buffer("cost_1k", cost_1k, persistent=False)
        self.register_buffer("is_struct", is_struct, persistent=False)

        self.act_kind_emb = nn.Embedding(16, 64, padding_idx=0)
        self.act_grp_emb = nn.Embedding(16, 32)
        self.act_gop_emb = nn.Embedding(8, 16)
        self.act_feat_proj = nn.Sequential(
            nn.Linear(64 + 32 + 16 + 12 + 3, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # Own-squad encoder (strictly causal own state + whether player issued new orders to it!)
        sq_in_dim = d_squad + N_SQUAD_Q + 2  # + (gate, squad_ordered_frac)
        self.sq_enc = nn.Sequential(
            nn.LayerNorm(sq_in_dim),
            nn.Linear(sq_in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        opp_in_dim = d_opp_z + N_OPP_Q + 1
        self.opp_enc = nn.Sequential(
            nn.LayerNorm(opp_in_dim),
            nn.Linear(opp_in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

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

        # Own macro intent + micro-execution tempo residual (strictly own-perspective!)
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

        # Player-equivariant Omniscient World Outcome Shock Posterior -> [B, 2, d_xi_world // 2] -> [B, 32]
        u_dim = innovation_dim(k_intent, d_intent, d_xi)
        world_in_dim = (
            2 * 2 * (N_SQUAD_Q + d_squad)
            + 2 * 2 * (N_MACRO_Q + d_squad)
            + 2 * d_squad
            + 2 * u_dim
            + 2
        )
        self.world_xi_net = nn.Sequential(
            nn.LayerNorm(world_in_dim),
            nn.Linear(world_in_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, d_xi_world // 2),
        )
        E.small_init(self.world_xi_net[-1], std=2e-2)

        # Auxiliary 2D-RoPE pointer projections grounding intent (x_k, y_k) onto ordered squads
        self.ptr_q = nn.Linear(d_intent, 64, bias=False)
        self.ptr_k = nn.Linear(d_squad + N_SQUAD_Q, 64, bias=False)
        self.ptr_rope = ED.RoPE2D(64, lo=ED.ROPE_SHORTEST, hi=256.0)
        self.ptr_scale = nn.Parameter(torch.tensor(math.log(6.0)))

    def extract_window_tempo_and_tokens(
        self,
        b: dict[str, torch.Tensor],
        f_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts purely causal 5s behavioral tempo stats (`[B, 2, 16]`) and spatialized
        action/camera event tokens (`[B, 2, K_keep, hidden]`) across `f_indices` (`16` frames)."""
        dev = b["ply_num"].device
        B = b["ply_num"].shape[0]
        f_idx = torch.as_tensor(f_indices, device=dev, dtype=torch.long)
        nf = len(f_indices)

        pl = b["ply_num"][:, f_idx].float()
        n_pc = len(ED.PC_KEYS)
        hx = b["half"][:, 0].float()[:, None, None] if "half" in b else torch.full((B, 1, 1), 100.0, device=dev)
        hy = b["half"][:, 1].float()[:, None, None] if "half" in b else torch.full((B, 1, 1), 100.0, device=dev)
        cam_ok = pl[..., n_pc + 2] > 0.5
        cam_x = torch.where(cam_ok, (pl[..., n_pc + 0] * hx) / E.MAP_SCALE, 0.0)
        cam_y = torch.where(cam_ok, (pl[..., n_pc + 1] * hy) / E.MAP_SCALE, 0.0)
        cam_xy = torch.stack([cam_x, cam_y], dim=-1).transpose(1, 2)  # [B, 2, nf, 2]

        cam_step = (cam_xy[:, :, 1:] - cam_xy[:, :, :-1]).norm(dim=-1)
        cam_jumps = (cam_step > 0.12).float().sum(dim=-1)
        cam_dist = cam_step.sum(dim=-1)

        has_acts = "act_mask" in b and "act_kind" in b
        if has_acts:
            amask = b["act_mask"][:, f_idx]
            aply = b["act_player"][:, f_idx]
            akind = b["act_kind"][:, f_idx].clamp(0, 15)
            agrp = b["act_group"][:, f_idx].clamp(0, 15) if "act_group" in b else torch.zeros_like(akind)
            agop = b["act_gop"][:, f_idx].clamp(0, 7) if "act_gop" in b else torch.zeros_like(akind)
            anum = b["act_num"][:, f_idx].float() if "act_num" in b else torch.zeros(*akind.shape, 12, device=dev)
            axy = (b["act_xy"][:, f_idx].float() / E.MAP_SCALE) if "act_xy" in b else torch.zeros(*akind.shape, 2, device=dev)
            aplaced = b["act_placed"][:, f_idx] if "act_placed" in b else (axy.abs().sum(dim=-1) > 1e-5)

            cam_per_f = cam_xy.transpose(1, 2)
            tok_p, pos_p, val_p, tempo_p = [], [], [], []
            for p_i in (0, 1):
                pm = amask & (aply == p_i)
                cam_p_f = cam_per_f[:, :, p_i, :].unsqueeze(2).expand(*axy.shape)
                pos_ev = torch.where(aplaced.unsqueeze(-1), axy, cam_p_f)

                is_cmd = pm & ((akind == 1) | (akind == 2) | (akind == 11))
                is_sel = pm & ((akind >= 3) & (akind <= 6))
                is_grp = pm & ((akind >= 7) & (akind <= 9))
                is_cam = pm & (akind == 10)
                n_total = pm.float().sum(dim=(1, 2))
                n_cmd = is_cmd.float().sum(dim=(1, 2))
                n_sel = is_sel.float().sum(dim=(1, 2))
                n_grp = is_grp.float().sum(dim=(1, 2))
                n_cam = is_cam.float().sum(dim=(1, 2))

                cmd_w = is_cmd.float()
                cmd_cnt = cmd_w.sum(dim=(1, 2)).clamp(min=1.0)
                cmd_mean = (pos_ev * cmd_w.unsqueeze(-1)).sum(dim=(1, 2)) / cmd_cnt.unsqueeze(-1)
                cmd_disp = (
                    (pos_ev - cmd_mean[:, None, None, :]).square().sum(dim=-1) * cmd_w
                ).sum(dim=(1, 2)) / cmd_cnt

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
                )
                tempo_p.append(t_vec)

                B_sz, nf_sz, A_sz = pm.shape
                flat_m = pm.reshape(B_sz, nf_sz * A_sz)
                K_keep = min(64, nf_sz * A_sz)
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
                )
                tok_p.append(emb_ev)
                pos_p.append(pos_keep)
                val_p.append(v_keep)

            tempo_feat = torch.stack(tempo_p, dim=1)
            act_tok = torch.stack(tok_p, dim=1)
            act_pos = torch.stack(pos_p, dim=1)
            act_val = torch.stack(val_p, dim=1)
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
        """Infers view-faithful player intents `u_flat` (`[B, 2, D_u]`), omniscient world
        outcome shock `xi_world` (`[B, 32]`), causal `tempo_feat`, and `combat_loss` (`[B, 2]`)."""
        with torch.autocast(device_type=s_tau["q_squad"].device.type, enabled=False):
            B, _, M, _ = s_tau["q_squad"].shape
            dev = s_tau["q_squad"].device

            tempo_feat, act_tok, act_pos, act_val = self.extract_window_tempo_and_tokens(
                b, f_indices
            )

            q_sq_t = s_tau["q_squad"].detach().float()
            z_sq_t = s_tau["z_squad"].detach().float()
            gate_t = (
                s_tau["gate"].detach().float()
                if "gate" in s_tau
                else s_tau["active"].detach().float()
            )
            act_t = s_tau["active"].detach()
            assign_w = s_tau["assign_w"].detach().float()
            q_mac_t = s_tau["q_macro"].detach().float()
            z_mac_t = s_tau["z_macro"].detach().float()

            q_opp_t = s_tau["q_opp_obs"].detach().float()
            z_opp_t = s_tau["z_opp_obs"].detach().float()
            g_opp_t = s_tau["gate_opp_obs"].detach().float()
            act_opp_t = g_opp_t > 0.02

            q_sq_n = s_next["q_squad"].detach().float()
            z_sq_n = s_next["z_squad"].detach().float()
            gate_n = (
                s_next["gate"].detach().float()
                if "gate" in s_next
                else s_next["active"].detach().float()
            )
            q_mac_n = s_next["q_macro"].detach().float()
            z_mac_n = s_next["z_macro"].detach().float()

            if ctx_indices is None:
                q_chg_seq = ctx["q_changed"][:, : len(f_indices)]
            else:
                c_idx = torch.as_tensor(ctx_indices, device=dev, dtype=torch.long)
                q_chg_seq = ctx["q_changed"][:, c_idx]

            # Mask unit_ordered strictly by unit owner inside matmul so P1's squad_ordered_frac has 0.0 P2 dependence!
            unit_ordered = q_chg_seq.any(dim=1).float()  # [B, R]
            mass = assign_w.sum(dim=-1).clamp(min=1e-6)
            squad_ordered_frac = (
                torch.matmul(assign_w, unit_ordered.unsqueeze(1).unsqueeze(-1)).squeeze(-1) / mass
            ) * gate_t

            # Perspective-faithful KV memory: own squads + perceived enemy opp_obs + own commands!
            sq_t = torch.cat([q_sq_t, z_sq_t], dim=-1)
            sq_tok = self.sq_enc(
                torch.cat([sq_t, gate_t.unsqueeze(-1), squad_ordered_frac.unsqueeze(-1)], dim=-1)
            )
            opp_tok = self.opp_enc(
                torch.cat([q_opp_t, z_opp_t, g_opp_t.unsqueeze(-1)], dim=-1)
            )

            M_o = q_opp_t.shape[2]
            K_a = act_tok.shape[2]
            N_mem = M + M_o + K_a
            kv_tok = torch.cat([sq_tok, opp_tok, act_tok.float()], dim=2).reshape(B * 2, N_mem, -1)
            kv_pos = torch.cat([q_sq_t[..., 0:2], q_opp_t[..., 0:2], act_pos.float()], dim=2).reshape(B * 2, N_mem, 2)
            kv_val = torch.cat([act_t, act_opp_t, act_val], dim=2).reshape(B * 2, N_mem)

            safe_mask = ~kv_val
            all_masked = safe_mask.all(dim=-1, keepdim=True)
            safe_mask = torch.cat([safe_mask[:, :1] & (~all_masked), safe_mask[:, 1:]], dim=-1)

            q_in = self.intent_queries[None, :, :].expand(B * 2, self.k_intent, -1)
            attn_out, attn_w = self.intent_attn(
                q_in, kv_tok, kv_tok, key_padding_mask=safe_mask, need_weights=True
            )
            h_int = self.intent_norm(q_in + attn_out)

            base_pos = torch.matmul(attn_w, kv_pos)
            intent_pos = (base_pos + 0.25 * torch.tanh(self.intent_pos_offset(h_int))).clamp(-2.0, 2.0)
            intent_gate = torch.sigmoid(self.intent_gate_head(h_int).squeeze(-1))
            intent_z = torch.tanh(self.intent_z_head(h_int)) * 2.0

            intent_pos = intent_pos.view(B, 2, self.k_intent, 2)
            intent_gate = intent_gate.view(B, 2, self.k_intent)
            intent_z = intent_z.view(B, 2, self.k_intent, self.d_intent)

            # Own macro intent + micro-execution residual (strictly own macro + own tempo!)
            mac_t = torch.cat([q_mac_t, z_mac_t], dim=-1)
            mac_n = torch.cat([q_mac_n, z_mac_n], dim=-1)
            h_pool = h_int.mean(dim=1).view(B, 2, -1)
            mac_in = torch.cat([mac_t, mac_n - mac_t, h_pool, tempo_feat.float()], dim=-1)

            intent_macro = torch.tanh(self.macro_intent_head(mac_in)) * 2.0
            xi = torch.tanh(self.xi_head(mac_in)) * 1.5
            u_flat = pack_innovation(intent_pos, intent_gate, intent_z, intent_macro, xi)

            # Realized combat/structural destruction impulses in (f_indices[0]-1 .. f_indices[-1]]
            b_f_start = max(0, int(f_indices[0]) - 1)
            b_f_end = int(f_indices[-1])
            combat_loss = extract_combat_losses(
                b, b_f_start, b_f_end, self.cost_1k, self.is_struct
            )  # [B, 2]

            # Player-equivariant Omniscient World Outcome Shock xi_world [B, 2, 16] -> [B, 32]
            sq_n = torch.cat([q_sq_n, z_sq_n], dim=-1)
            sum_t = (sq_t * gate_t.unsqueeze(-1)).sum(dim=2) / gate_t.sum(dim=2, keepdim=True).clamp(min=1.0)
            sum_n = (sq_n * gate_n.unsqueeze(-1)).sum(dim=2) / gate_n.sum(dim=2, keepdim=True).clamp(min=1.0)
            dsum = sum_n - sum_t
            dmac = mac_n - mac_t
            z_om_t = s_tau.get("z_omni_scene", torch.zeros(B, self.d_squad, device=dev)).detach().float()
            z_om_n = s_next.get("z_omni_scene", torch.zeros(B, self.d_squad, device=dev)).detach().float()
            u_det = u_flat.detach()
            w_p_list = []
            for p_i in (0, 1):
                opp = 1 - p_i
                w_in_p = torch.cat(
                    [
                        sum_t[:, p_i], sum_t[:, opp],
                        dsum[:, p_i], dsum[:, opp],
                        mac_t[:, p_i], mac_t[:, opp],
                        dmac[:, p_i], dmac[:, opp],
                        z_om_t, z_om_n - z_om_t,
                        u_det[:, p_i], u_det[:, opp],
                        combat_loss[:, p_i : p_i + 1], combat_loss[:, opp : opp + 1],
                    ],
                    dim=-1,
                )
                w_p_list.append(torch.tanh(self.world_xi_net(w_in_p)) * 2.0)
            xi_world = torch.cat(w_p_list, dim=-1)  # [B, 32] (exact P1|P2 16d halves!)

            # Auxiliary 2D-RoPE pointer grounding loss: intents -> ordered own squads
            pq = F.normalize(self.ptr_q(intent_z), dim=-1, eps=1e-4).reshape(B * 2, self.k_intent, 64)
            pk = F.normalize(self.ptr_k(sq_t), dim=-1, eps=1e-4).reshape(B * 2, M, 64)
            pq_rot = _apply_rope_2d(pq, intent_pos.reshape(B * 2, self.k_intent, 2) * E.MAP_SCALE, self.ptr_rope)
            pk_rot = _apply_rope_2d(pk, q_sq_t[..., 0:2].reshape(B * 2, M, 2) * E.MAP_SCALE, self.ptr_rope)

            ptr_scores = torch.matmul(pq_rot, pk_rot.transpose(1, 2)) * self.ptr_scale.exp().clamp(1.0, 30.0)
            gated_scores = ptr_scores + torch.log(intent_gate.reshape(B * 2, self.k_intent, 1).clamp(min=1e-4))
            squad_pred_logit = torch.logsumexp(gated_scores, dim=1).view(B, 2, M)

            target_ord = (squad_ordered_frac > 0.05).float()
            valid_sq = gate_t
            bce = F.binary_cross_entropy_with_logits(squad_pred_logit, target_ord, reduction="none")
            aux_ptr_loss = (bce * valid_sq).sum() / valid_sq.sum().clamp(min=1.0)

            reg_loss = (
                1e-3 * u_flat.square().mean()
                + 1e-3 * xi_world.square().mean()
                + 2e-2 * intent_gate.mean()
            )

            return dict(
                intent_pos=intent_pos,
                intent_gate=intent_gate,
                intent_z=intent_z,
                intent_macro=intent_macro,
                xi=xi,
                u_flat=u_flat,
                xi_world=xi_world,
                combat_loss=combat_loss,
                tempo_feat=tempo_feat.float(),
                squad_ordered_frac=squad_ordered_frac,
                aux_loss=aux_ptr_loss + reg_loss,
                aux_ptr_loss=aux_ptr_loss.detach(),
            )


# Backwards-compatible aliases
SquadPooler = PerceiverSquadCompressor
HindsightIntentPosterior = HindsightOutcomePosterior


def innovation_dim(
    k_intent: int = K_INTENT_DEFAULT,
    d_intent: int = D_INTENT_DEFAULT,
    d_xi: int = D_XI_DEFAULT,
) -> int:
    """Total flat dimension of one player's 5s innovation vector `U_{tau, p}` (`180` dims)."""
    return k_intent * (3 + d_intent) + d_intent + d_xi


def pack_innovation(
    intent_pos: torch.Tensor,
    intent_gate: torch.Tensor,
    intent_z: torch.Tensor,
    intent_macro: torch.Tensor,
    xi: torch.Tensor,
) -> torch.Tensor:
    """Packs structured intent & micro-residual tensors into `[..., D_u]`."""
    eps = 1e-4
    g_clamped = intent_gate.clamp(eps, 1.0 - eps)
    gate_logit = (torch.log(g_clamped) - torch.log1p(-g_clamped)) * 0.25
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
    """Unpacks `[..., D_u]` back into structured `(intent_pos, intent_gate, intent_z, intent_macro, xi)`."""
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
