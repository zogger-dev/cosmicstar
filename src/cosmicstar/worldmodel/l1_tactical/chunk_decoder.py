"""Parallel 16-Frame (`step = 7` loops = `0.3125s`, `5.0s` total) Layer-1 Tactical Chunk Flow Decoder.

Bridges Layer-2 strategic/tactical transitions `(S_0 -> S_1, u_1, u_2, xi_world)` back down to
individual StarCraft II unit micro-trajectories without `O((16 * R)^2)` attention blowup:
  - Jointly denoises `[R]` pre-existing player entities + `[2 * B_max]` (`B_max = 16`/player)
    newly spawned/warped-in unit slots in `4` parallel Conditional Flow-Matching steps.
  - Per living entity (`16 x 10 = 160` dims): `0.3125s`-spaced `(dx, dy)` displacements,
    `(vx, vy)` velocities, `(hp_frac, shield_frac)`, survival `alive`, and unit order targets
    `(ord_active, ord_dx, ord_dy)`.
  - Per spawn slot (`4 + 16 x 6 = 100` dims + unit-type head): `spawn_active`, `birth_time`,
    world `spawn_xy`, `16`-frame post-spawn kinematics/HP/survival, and `rtype` logits
    (capturing frontline Warp Prism Stalker warp-ins, Zergling/Hydra hatches, and Stasis Traps).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.constants import K_Q, MAP_SCALE, N_RAW_TYPES, SPEED_SCALE
from ..nn.layers import small_init
from ..nn.rope import ROPE_SHORTEST, RoPE2D
from . import squad_pooler as SP
from .frames import I

K_CHUNK_DEFAULT = 16       # 16 frames x 7 loops = 112 loops = 5.0s (0.3125s / step)
B_MAX_SPAWN_DEFAULT = 16   # max newly spawned/warped-in units per player per 5.0s chunk
POS_DELTA_SCALE = 25.0     # 1.0 == 25.0 map units displacement over 5.0s
D_ENT_STEP = 10            # (dx, dy, vx, vy, hp, sh, alive, ord_act, ord_dx, ord_dy)
D_SPAWN_STEP = 6           # (dx, dy, vx, vy, hp, alive)


class _AdaLN(nn.Module):
    """Adaptive LayerNorm modulation conditioned on `(flow_time, xi_world, macro_u)`."""

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 3 * d_model),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.proj(cond).unsqueeze(1).chunk(3, dim=-1)
        h = self.norm(x) * (1.0 + scale) + shift
        return h, 1.0 + gate


class _ChunkFlowBlock(nn.Module):
    """2D-RoPE Self-Attention across `[R + 2*B_max]` entity/spawn tokens + 2D-RoPE
    Cross-Attention to evolved L2 `(Squad, Intent, WorldOutcome)` tokens + SwiGLU."""

    def __init__(self, d_model: int, d_cond: int, heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.dh = d_model // heads

        self.adaln_self = _AdaLN(d_model, d_cond)
        self.qkv_self = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_self = nn.Linear(d_model, d_model, bias=False)
        small_init(self.out_self, std=2e-2)

        self.norm_cross_q = nn.LayerNorm(d_model)
        self.q_cross = nn.Linear(d_model, d_model, bias=False)
        self.kv_cross = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_cross = nn.Linear(d_model, d_model, bias=False)
        small_init(self.out_cross, std=2e-2)

        self.adaln_ff = _AdaLN(d_model, d_cond)
        ff_dim = int(d_model * 2.5)
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)
        small_init(self.w2, std=2e-2)

    def forward(
        self,
        x: torch.Tensor,
        tok_pos_map: torch.Tensor,
        tok_valid: torch.Tensor,
        cond_kv: torch.Tensor,
        cond_pos_map: torch.Tensor,
        cond_vec: torch.Tensor,
        rope_2d: RoPE2D,
        cond_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        H, dh = self.heads, self.dh

        # 1. 2D-RoPE Self-Attention over active living entities + spawn slots
        h_s, gate_s = self.adaln_self(x, cond_vec)
        qkv = self.qkv_self(h_s).view(B, N, 3, H, dh)
        q_s = SP._apply_rope_2d(qkv[:, :, 0].reshape(B, N, D), tok_pos_map, rope_2d).view(B, N, H, dh).transpose(1, 2)
        k_s = SP._apply_rope_2d(qkv[:, :, 1].reshape(B, N, D), tok_pos_map, rope_2d).view(B, N, H, dh).transpose(1, 2)
        v_s = qkv[:, :, 2].transpose(1, 2)

        # Prevent all-False rows from producing NaN in SDPA; diagonal self-attn always safe
        eye_N = torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
        allow_self = (tok_valid[:, None, :] & tok_valid[:, :, None]) | eye_N
        attn_s = F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=allow_self.unsqueeze(1))
        x = x + gate_s * self.out_self(attn_s.transpose(1, 2).reshape(B, N, D))

        # 2. 2D-RoPE Cross-Attention to active L2 (S_0 -> S_1, u_1, u_2, xi_world) memory
        N_c = cond_kv.shape[1]
        q_c = SP._apply_rope_2d(self.q_cross(self.norm_cross_q(x)), tok_pos_map, rope_2d).view(B, N, H, dh).transpose(1, 2)
        kv_c = self.kv_cross(cond_kv).view(B, N_c, 2, H, dh)
        k_c = SP._apply_rope_2d(kv_c[:, :, 0].reshape(B, N_c, D), cond_pos_map, rope_2d).view(B, N_c, H, dh).transpose(1, 2)
        v_c = kv_c[:, :, 1].transpose(1, 2)
        c_mask = cond_valid[:, None, None, :] if cond_valid is not None else None
        attn_c = F.scaled_dot_product_attention(q_c, k_c, v_c, attn_mask=c_mask)
        x = x + self.out_cross(attn_c.transpose(1, 2).reshape(B, N, D))

        # 3. AdaLN SwiGLU FFN
        h_f, gate_f = self.adaln_ff(x, cond_vec)
        ff_out = self.w2(F.silu(self.w1(h_f)) * self.w3(h_f))
        x = (x + gate_f * ff_out) * tok_valid.unsqueeze(-1).to(x.dtype)
        return x


class L1ChunkFlowDecoder(nn.Module):
    """Parallel `16`-Frame (`0.3125s` step = `5.0s`) Conditional Flow-Matching Decoder."""

    def __init__(
        self,
        d_l1: int = 256,
        d_squad: int = SP.D_SQUAD_DEFAULT,
        d_model: int = 256,
        k_chunk: int = K_CHUNK_DEFAULT,
        b_max_spawn: int = B_MAX_SPAWN_DEFAULT,
        k_intent: int = SP.K_INTENT_DEFAULT,
        d_intent: int = SP.D_INTENT_DEFAULT,
        d_xi: int = SP.D_XI_DEFAULT,
        d_xi_world: int = SP.D_XI_WORLD_DEFAULT,
        n_layers: int = 3,
        heads: int = 4,
    ):
        super().__init__()
        self.d_l1 = d_l1
        self.d_squad = d_squad
        self.d_model = d_model
        self.k_chunk = k_chunk
        self.b_max = b_max_spawn
        self.k_intent = k_intent
        self.d_intent = d_intent
        self.d_xi = d_xi
        self.d_xi_world = d_xi_world
        self.d_ent_traj = k_chunk * D_ENT_STEP          # 160
        self.d_spawn_traj = 4 + k_chunk * D_SPAWN_STEP  # 100

        dh = d_model // heads
        self.rope_2d = RoPE2D(dh, lo=ROPE_SHORTEST, hi=256.0)

        # Terrain height + pathability + placement local 5-point sampler
        offsets = torch.tensor(
            [[0.0, 0.0], [4.0, 0.0], [-4.0, 0.0], [0.0, 4.0], [0.0, -4.0]],
            dtype=torch.float32,
        )
        self.register_buffer("terr_offsets", offsets, persistent=False)
        self.terr_proj = nn.Sequential(
            nn.Linear(5 * 4, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )

        # Flow-time sinusoidal embedding + global (u_macro, xi, xi_world, q_macro) conditioning
        self.register_buffer(
            "t_freqs",
            torch.exp(-math.log(1000.0) * torch.arange(0, 32, dtype=torch.float32) / 32.0) * math.pi,
            persistent=False,
        )
        glob_in_dim = 64 + 2 * (2 * SP.N_MACRO_Q + d_intent + d_xi) + d_xi_world
        self.d_cond = d_model
        self.glob_cond_net = nn.Sequential(
            nn.LayerNorm(glob_in_dim),
            nn.Linear(glob_in_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # L2 Conditioning Token Encoders: evolved Squads (2*M) + Intents (2*K) + Global(1)
        sq_cond_in = 2 * (SP.N_SQUAD_Q + d_squad + 1) + 2 + 64  # s0, delta_s, player_onehot, terrain
        self.sq_cond_proj = nn.Sequential(
            nn.LayerNorm(sq_cond_in),
            nn.Linear(sq_cond_in, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        int_cond_in = 2 + 1 + d_intent + 2 + 64
        self.int_cond_proj = nn.Sequential(
            nn.LayerNorm(int_cond_in),
            nn.Linear(int_cond_in, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Entity token input: c_ent_0(d_l1) + q_0(8) + owner_onehot(2) + rtype_emb(64) + terr(64) + noisy_traj(160)
        self.rtype_emb = nn.Embedding(N_RAW_TYPES, 64)
        nn.init.normal_(self.rtype_emb.weight, std=0.02)
        ent_in_dim = d_l1 + K_Q + 2 + 64 + 64 + self.d_ent_traj
        self.ent_in_proj = nn.Sequential(
            nn.LayerNorm(ent_in_dim),
            nn.Linear(ent_in_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Spawn slot token input: player_emb(2) + slot_emb(d_model//2) + terr(64) + noisy_spawn(100)
        self.spawn_slot_emb = nn.Embedding(2 * b_max_spawn, 64)
        nn.init.normal_(self.spawn_slot_emb.weight, std=0.02)
        spawn_in_dim = 2 + 64 + 64 + self.d_spawn_traj
        self.spawn_in_proj = nn.Sequential(
            nn.LayerNorm(spawn_in_dim),
            nn.Linear(spawn_in_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList(
            [_ChunkFlowBlock(d_model, self.d_cond, heads=heads) for _ in range(n_layers)]
        )

        # Output velocity heads (zero-initialized residual heads for stable CFM start)
        self.ent_vel_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, self.d_ent_traj),
        )
        small_init(self.ent_vel_head[-1], std=1e-2)

        self.spawn_vel_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, self.d_spawn_traj),
        )
        small_init(self.spawn_vel_head[-1], std=1e-2)

        # Unit-type classifier for active spawned units
        self.spawn_rtype_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, N_RAW_TYPES),
        )
        small_init(self.spawn_rtype_head[-1], std=1e-2)

    def _sample_terrain(
        self,
        pos_map: torch.Tensor,
        static_map: dict[str, torch.Tensor] | None = None,
        b: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Samples 5-point static terrain features at `pos_map` (`[B, N, 2]` map units
        relative to map center) -> `[B, N, 64]`."""
        B, N, _ = pos_map.shape
        dev = pos_map.device
        src = static_map if static_map is not None else b
        if (
            src is not None
            and ("static_grid_chw" in src or "static_grid" in src)
            and "center" in src
            and "origin" in src
        ):
            if "static_grid_chw" in src:
                st = src["static_grid_chw"].float()
            else:
                g = src["static_grid"].float()
                if g.shape[-1] == 4 and bool((g[..., :2].max() > 2.0).item()):
                    g = torch.cat([g[..., :2] / 255.0, g[..., 2:]], dim=-1)
                st = g.permute(0, 3, 1, 2).contiguous()
            B_map = st.shape[0]
            if B_map != B:
                st = st.expand(B, *st.shape[1:])
            gh, gw = st.shape[2], st.shape[3]
            off = (src["center"].float() - src["origin"].float())
            if off.shape[0] != B:
                off = off.expand(B, -1)
            off = off[:, None, None, :]
            wh = pos_map.new_tensor([float(gw), float(gh)])
            at = ((pos_map.float().unsqueeze(-2) + self.terr_offsets + off) / wh) * 2.0 - 1.0
            with torch.autocast(device_type=dev.type, enabled=False):
                samp = (
                    F.grid_sample(st, at, align_corners=False)
                    .permute(0, 2, 3, 1)
                    .reshape(B, N, 20)
                )
        else:
            samp = torch.zeros(B, N, 20, device=dev, dtype=torch.float32)
        return self.terr_proj(samp)

    def extract_chunk_targets(
        self,
        b: dict[str, torch.Tensor],
        f_0: int,
    ) -> dict[str, torch.Tensor]:
        """Extracts ground-truth `16`-frame entity trajectories `[B, R, 160]` and canonical
        `B_max = 16` spawn slot trajectories `[B, 2, B_max, 100]` over `(f_0 .. f_0 + K]`."""
        dev = b["mask"].device
        B, F_tot, R = b["mask"].shape
        K = self.k_chunk
        assert f_0 + K < F_tot, f"Need f_0 + {K} < F_tot={F_tot}, got f_0={f_0}"

        f_seq = torch.arange(f_0 + 1, f_0 + K + 1, device=dev, dtype=torch.long)  # [K]
        f_prev = f_seq - 1

        center = b["center"].float() if "center" in b else torch.full((B, 2), 100.0, device=dev)
        half = b["half"].float() if "half" in b else torch.full((B, 2), 100.0, device=dev)
        owner = b["owner"]  # [B, R]: 0=P1, 1=P2, 16=NEUTRAL

        on_map_0 = b["mask"][:, f_0] & b["alive"][:, f_0] & (b["inside"][:, f_0] == 0)
        ent_valid = on_map_0 & (owner < 2)  # [B, R]

        x_0 = b["x"][:, f_0].float()  # [B, R]
        y_0 = b["y"][:, f_0].float()  # [B, R]

        x_seq = b["x"][:, f_seq].float()    # [B, K, R]
        y_seq = b["y"][:, f_seq].float()    # [B, K, R]
        xp_seq = b["x"][:, f_prev].float()  # [B, K, R]
        yp_seq = b["y"][:, f_prev].float()  # [B, K, R]

        alive_seq = b["mask"][:, f_seq] & b["alive"][:, f_seq] & (b["inside"][:, f_seq] == 0)
        both_seq = alive_seq & b["mask"][:, f_prev] & b["alive"][:, f_prev] & (b["inside"][:, f_prev] == 0)

        if "loops" in b:
            dt_loops = (b["loops"][:, f_seq] - b["loops"][:, f_prev]).clamp(min=1).float().unsqueeze(-1)
        else:
            dt_loops = torch.full((B, K, 1), 7.0, device=dev)

        # Carry last valid position once a unit dies so dead units sit cleanly at death spot
        dx_raw = ((x_seq - x_0.unsqueeze(1)) / POS_DELTA_SCALE).clamp(-3.0, 3.0)
        dy_raw = ((y_seq - y_0.unsqueeze(1)) / POS_DELTA_SCALE).clamp(-3.0, 3.0)
        dx_clean = torch.zeros_like(dx_raw)
        dy_clean = torch.zeros_like(dy_raw)
        last_dx = torch.zeros(B, R, device=dev)
        last_dy = torch.zeros(B, R, device=dev)
        for k_i in range(K):
            al_k = alive_seq[:, k_i]
            last_dx = torch.where(al_k, dx_raw[:, k_i], last_dx)
            last_dy = torch.where(al_k, dy_raw[:, k_i], last_dy)
            dx_clean[:, k_i] = last_dx
            dy_clean[:, k_i] = last_dy

        vx_seq = (((x_seq - xp_seq) * both_seq.float()) / (SPEED_SCALE * dt_loops)).clamp(-3.0, 3.0)
        vy_seq = (((y_seq - yp_seq) * both_seq.float()) / (SPEED_SCALE * dt_loops)).clamp(-3.0, 3.0)

        num_seq = b["num"][:, f_seq].float()
        hp_max = num_seq[..., I["hp_max"]]
        sh_max = num_seq[..., I["shield_max"]]
        hp_seq = ((num_seq[..., I["hp"]] / hp_max.clamp(min=1.0)) * (hp_max > 0) * alive_seq.float()).clamp(0.0, 1.0)
        sh_seq = ((num_seq[..., I["shield"]] / sh_max.clamp(min=1.0)) * (sh_max > 0) * alive_seq.float()).clamp(0.0, 1.0)
        alive_pm = torch.where(alive_seq, 1.0, -1.0)

        abil_seq = b["slot_abil"][:, f_seq, :, 0]
        reg_seq = b["slot_reg"][:, f_seq, :, 0, :].float()
        ord_act = (abil_seq > 0) & alive_seq
        ord_act_pm = torch.where(ord_act, 1.0, -1.0)
        hx = half[:, 0][:, None, None]
        hy = half[:, 1][:, None, None]
        ord_dx = ((reg_seq[..., 0] * hx) / POS_DELTA_SCALE).clamp(-3.0, 3.0) * ord_act.float()
        ord_dy = ((reg_seq[..., 1] * hy) / POS_DELTA_SCALE).clamp(-3.0, 3.0) * ord_act.float()

        # [B, K, R, 10] -> [B, R, K * 10]
        ent_traj = torch.stack(
            [dx_clean, dy_clean, vx_seq, vy_seq, hp_seq, sh_seq, alive_pm, ord_act_pm, ord_dx, ord_dy],
            dim=-1,
        ).transpose(1, 2).reshape(B, R, K * D_ENT_STEP)
        ent_traj = ent_traj * ent_valid.unsqueeze(-1).float()

        # Canonical newly spawned / unloaded player units in (f_0 .. f_0 + K]
        k_grid = torch.arange(K, device=dev)[None, :, None]  # [1, K, 1]
        first_k = torch.where(alive_seq, k_grid, K + 1).amin(dim=1)  # [B, R]
        is_new_unit = (~on_map_0) & (first_k < K) & (owner < 2)      # [B, R]
        fk_clamped = first_k.clamp(0, K - 1)                         # [B, R]

        b_idx = torch.arange(B, device=dev)[:, None].expand(B, R)
        r_idx = torch.arange(R, device=dev)[None, :].expand(B, R)
        x_birth = x_seq[b_idx, fk_clamped, r_idx]                    # [B, R]
        y_birth = y_seq[b_idx, fk_clamped, r_idx]                    # [B, R]
        rtype_birth = b["rtype"][:, f_seq][b_idx, fk_clamped, r_idx].clamp(0, N_RAW_TYPES - 1)

        xb_norm = ((x_birth - center[:, 0:1]) / MAP_SCALE).clamp(-2.0, 2.0)
        yb_norm = ((y_birth - center[:, 1:2]) / MAP_SCALE).clamp(-2.0, 2.0)

        spawn_traj_list, spawn_rtype_list, spawn_active_list = [], [], []
        for p_i in (0, 1):
            cand = is_new_unit & (owner == p_i)  # [B, R]
            # Deterministic lexicographic sort key: birth frame then x then y (1e6 for non-candidates)
            sort_key = torch.where(
                cand,
                fk_clamped.float() * 10.0 + xb_norm + 0.05 * yb_norm,
                torch.full_like(xb_norm, 1e6),
            )
            B_m = self.b_max
            if R >= B_m:
                _, sel_r = torch.topk(sort_key, k=B_m, dim=1, largest=False)  # [B, B_m]
                sp_act = torch.gather(cand, 1, sel_r)                         # [B, B_m]
            else:
                pad_n = B_m - R
                sel_r = F.pad(torch.argsort(sort_key, dim=1), (0, pad_n), value=0)
                sp_act = F.pad(torch.gather(cand, 1, sel_r[:, :R]), (0, pad_n), value=False)

            fk_s = torch.gather(fk_clamped, 1, sel_r)                         # [B, B_m]
            xb_s = torch.gather(x_birth, 1, sel_r)
            yb_s = torch.gather(y_birth, 1, sel_r)
            xbn_s = torch.gather(xb_norm, 1, sel_r) * sp_act.float()
            ybn_s = torch.gather(yb_norm, 1, sel_r) * sp_act.float()
            rt_s = torch.gather(rtype_birth, 1, sel_r) * sp_act.long()

            # Gather [B, K, B_m] post-spawn kinematics
            sel_kr = sel_r.unsqueeze(1).expand(B, K, B_m)
            xs_k = torch.gather(x_seq, 2, sel_kr)
            ys_k = torch.gather(y_seq, 2, sel_kr)
            vxs_k = torch.gather(vx_seq, 2, sel_kr)
            vys_k = torch.gather(vy_seq, 2, sel_kr)
            hps_k = torch.gather(hp_seq, 2, sel_kr)
            als_k = torch.gather(alive_seq, 2, sel_kr) & sp_act.unsqueeze(1)

            dxs_raw = ((xs_k - xb_s.unsqueeze(1)) / POS_DELTA_SCALE).clamp(-3.0, 3.0)
            dys_raw = ((ys_k - yb_s.unsqueeze(1)) / POS_DELTA_SCALE).clamp(-3.0, 3.0)
            born_yet = torch.arange(K, device=dev)[None, :, None] >= fk_s.unsqueeze(1)
            dxs_k = torch.where(als_k, dxs_raw, torch.zeros_like(dxs_raw)) * born_yet.float()
            dys_k = torch.where(als_k, dys_raw, torch.zeros_like(dys_raw)) * born_yet.float()
            als_pm = torch.where(als_k, 1.0, -1.0)

            sp_step = torch.stack(
                [dxs_k, dys_k, vxs_k * als_k.float(), vys_k * als_k.float(), hps_k, als_pm],
                dim=-1,
            ).transpose(1, 2).reshape(B, B_m, K * D_SPAWN_STEP)

            act_pm = torch.where(sp_act, 1.0, -1.0)
            bt_norm = ((fk_s.float() / float(max(1, K - 1))) * 2.0 - 1.0) * sp_act.float()
            sp_hdr = torch.stack([act_pm, bt_norm, xbn_s, ybn_s], dim=-1)
            sp_full = torch.cat([sp_hdr, sp_step], dim=-1)

            spawn_traj_list.append(sp_full)
            spawn_rtype_list.append(rt_s)
            spawn_active_list.append(sp_act)

        return dict(
            ent_traj=ent_traj,                                  # [B, R, 160]
            ent_valid=ent_valid,                                # [B, R]
            pos_0_map=torch.stack([x_0, y_0], dim=-1) - center[:, None, :],  # [B, R, 2]
            spawn_traj=torch.stack(spawn_traj_list, dim=1),     # [B, 2, B_max, 100]
            spawn_rtype=torch.stack(spawn_rtype_list, dim=1),   # [B, 2, B_max]
            spawn_active=torch.stack(spawn_active_list, dim=1), # [B, 2, B_max]
        )

    def _build_conditioning(
        self,
        s_0: dict[str, torch.Tensor],
        s_1: dict[str, torch.Tensor],
        u_flat: torch.Tensor,
        xi_world: torch.Tensor,
        flow_t: torch.Tensor,
        static_map: dict[str, torch.Tensor] | None = None,
        b: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Builds `(cond_kv [B, N_c, D], cond_pos_map [B, N_c, 2], cond_vec [B, D])`."""
        B, _, M, _ = s_0["q_squad"].shape
        dev = u_flat.device

        # 1. Global conditioning vector cond_vec [B, D]
        t_ang = flow_t.view(B, 1).float() * self.t_freqs[None, :]
        t_emb = torch.cat([t_ang.sin(), t_ang.cos()], dim=-1)  # [B, 64]

        unpacked = SP.unpack_innovation(
            u_flat.float(), self.k_intent, self.d_intent, self.d_xi
        )
        qm0 = s_0["q_macro"].float()
        qm1 = s_1["q_macro"].float()
        p_glob = torch.cat([qm0, qm1 - qm0, unpacked["intent_macro"], unpacked["xi"]], dim=-1).flatten(1)
        cond_vec = self.glob_cond_net(torch.cat([t_emb, p_glob, xi_world.float()], dim=-1))

        # 2. Evolved Squad conditioning tokens [B, 2*M, D]
        q0, z0 = s_0["q_squad"].float(), s_0["z_squad"].float()
        g0 = (s_0["gate"].float() if "gate" in s_0 else s_0["active"].float()).clamp(0.0, 1.0)
        q1, z1 = s_1["q_squad"].float(), s_1["z_squad"].float()
        g1 = (s_1["gate"].float() if "gate" in s_1 else s_1["active"].float()).clamp(0.0, 1.0)

        # Unscale gate-weighted centroids so partial-gate squads sit at exact world coords
        pos0_norm = q0[..., 0:2] / g0.unsqueeze(-1).clamp(min=0.05)
        pos1_norm = q1[..., 0:2] / g1.unsqueeze(-1).clamp(min=0.05)
        w0 = g0.unsqueeze(-1)
        w1 = g1.unsqueeze(-1)
        sq_mid_pos = ((w0 * pos0_norm + w1 * pos1_norm) / (w0 + w1).clamp(min=0.05)) * MAP_SCALE
        sq_mid_flat = sq_mid_pos.reshape(B, 2 * M, 2)
        sq_terr = self._sample_terrain(sq_mid_flat, static_map=static_map, b=b).view(B, 2, M, 64)

        p_oh_sq = F.one_hot(torch.tensor([0, 1], device=dev), num_classes=2).float()[None, :, None, :].expand(B, 2, M, 2)
        sq_in = torch.cat(
            [q0, z0, g0.unsqueeze(-1), q1 - q0, z1 - z0, (g1 - g0).unsqueeze(-1), p_oh_sq, sq_terr],
            dim=-1,
        ).reshape(B, 2 * M, -1)
        sq_kv = self.sq_cond_proj(sq_in)
        sq_valid = ((g0 > 0.03) | (g1 > 0.03)).reshape(B, 2 * M)

        # 3. Spatial Intent conditioning tokens [B, 2*K_intent, D]
        K_i = self.k_intent
        int_pos_map = unpacked["intent_pos"].reshape(B, 2 * K_i, 2) * MAP_SCALE
        int_terr = self._sample_terrain(int_pos_map, static_map=static_map, b=b).view(B, 2, K_i, 64)
        p_oh_i = F.one_hot(torch.tensor([0, 1], device=dev), num_classes=2).float()[None, :, None, :].expand(B, 2, K_i, 2)
        int_in = torch.cat(
            [
                unpacked["intent_pos"],
                unpacked["intent_gate"].unsqueeze(-1),
                unpacked["intent_z"],
                p_oh_i,
                int_terr,
            ],
            dim=-1,
        ).reshape(B, 2 * K_i, -1)
        int_kv = self.int_cond_proj(int_in)
        int_valid = (unpacked["intent_gate"] > 0.05).reshape(B, 2 * K_i)

        # 4. Global world-outcome token at map origin (0, 0) (always valid -> prevents NaN SDPA)
        glob_kv = cond_vec.unsqueeze(1)
        glob_pos = torch.zeros(B, 1, 2, device=dev, dtype=torch.float32)
        glob_valid = torch.ones(B, 1, dtype=torch.bool, device=dev)

        cond_kv = torch.cat([sq_kv, int_kv, glob_kv], dim=1)
        cond_pos_map = torch.cat([sq_mid_flat, int_pos_map, glob_pos], dim=1)
        cond_valid = torch.cat([sq_valid, int_valid, glob_valid], dim=1)
        return cond_kv, cond_pos_map, cond_vec, cond_valid

    def predict_velocity(
        self,
        ent_y: torch.Tensor,
        spawn_y: torch.Tensor,
        flow_t: torch.Tensor,
        c_ent_0: torch.Tensor,
        q_0: torch.Tensor,
        pos_0_map: torch.Tensor,
        rtype_0: torch.Tensor,
        owner_raw: torch.Tensor,
        ent_valid: torch.Tensor,
        s_0: dict[str, torch.Tensor],
        s_1: dict[str, torch.Tensor],
        u_flat: torch.Tensor,
        xi_world: torch.Tensor,
        static_map: dict[str, torch.Tensor] | None = None,
        b: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts joint CFM velocities `(v_ent [B, R, 160], v_spawn [B, 2, B_max, 100])`
        plus spawn unit-type logits `[B, 2, B_max, N_RAW_TYPES]`."""
        B, R, _ = ent_y.shape
        dev = ent_y.device
        B_m = self.b_max
        N_sp = 2 * B_m

        cond_kv, cond_pos_map, cond_vec, cond_valid = self._build_conditioning(
            s_0, s_1, u_flat, xi_world, flow_t, static_map=static_map, b=b
        )

        # Anchor each entity token at midpoint of its current denoised trajectory for accurate 2D-RoPE!
        ent_steps = ent_y.view(B, R, self.k_chunk, D_ENT_STEP)
        mid_disp = ent_steps[:, :, self.k_chunk // 2, 0:2].clamp(-3.0, 3.0) * POS_DELTA_SCALE
        ent_pos_cur = (pos_0_map + 0.5 * mid_disp) * ent_valid.unsqueeze(-1).float()
        ent_terr = self._sample_terrain(ent_pos_cur, static_map=static_map, b=b)

        own_clamped = owner_raw.clamp(0, 1)
        own_oh = F.one_hot(own_clamped, num_classes=2).float() * (owner_raw < 2).unsqueeze(-1).float()
        rt_emb = self.rtype_emb(rtype_0.clamp(0, N_RAW_TYPES - 1))

        ent_in = torch.cat([c_ent_0.float(), q_0.float(), own_oh, rt_emb, ent_terr, ent_y.float()], dim=-1)
        h_ent = self.ent_in_proj(ent_in) * ent_valid.unsqueeze(-1).float()

        # Spawn slot tokens [B, 2*B_max, D] anchored at their current denoised spawn_xy!
        sp_flat = spawn_y.reshape(B, N_sp, self.d_spawn_traj).float()
        sp_pos_cur = sp_flat[:, :, 2:4].clamp(-2.0, 2.0) * MAP_SCALE
        sp_terr = self._sample_terrain(sp_pos_cur, static_map=static_map, b=b)

        slot_ids = torch.arange(N_sp, device=dev)
        slot_emb = self.spawn_slot_emb(slot_ids)[None, :, :].expand(B, N_sp, -1)
        sp_p_oh = F.one_hot(slot_ids // B_m, num_classes=2).float()[None, :, :].expand(B, N_sp, 2)
        sp_in = torch.cat([sp_p_oh, slot_emb, sp_terr, sp_flat], dim=-1)
        h_sp = self.spawn_in_proj(sp_in)

        x_tok = torch.cat([h_ent, h_sp], dim=1)                                   # [B, R + N_sp, D]
        tok_pos = torch.cat([ent_pos_cur, sp_pos_cur], dim=1)                     # [B, R + N_sp, 2]
        sp_valid = torch.ones(B, N_sp, dtype=torch.bool, device=dev)
        tok_valid = torch.cat([ent_valid, sp_valid], dim=1)                       # [B, R + N_sp]

        for blk in self.blocks:
            x_tok = blk(
                x_tok, tok_pos, tok_valid, cond_kv, cond_pos_map, cond_vec, self.rope_2d, cond_valid=cond_valid
            )

        h_ent_out = x_tok[:, :R]
        h_sp_out = x_tok[:, R:].view(B, 2, B_m, self.d_model)

        v_ent = self.ent_vel_head(h_ent_out) * ent_valid.unsqueeze(-1).float()
        v_spawn = self.spawn_vel_head(h_sp_out)
        spawn_rtype_logits = self.spawn_rtype_head(h_sp_out)
        return v_ent, v_spawn, spawn_rtype_logits

    def flow_matching_loss(
        self,
        b: dict[str, torch.Tensor],
        ctx: dict[str, torch.Tensor],
        f_0: int,
        c_0: int,
        s_0: dict[str, torch.Tensor],
        s_1: dict[str, torch.Tensor],
        u_flat: torch.Tensor,
        xi_world: torch.Tensor,
        static_map: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Single-pass Conditional Flow-Matching loss for live `L1 + L2` co-training."""
        with torch.autocast(device_type=u_flat.device.type, enabled=False):
            tgt = self.extract_chunk_targets(b, f_0)
            ent_1 = tgt["ent_traj"]            # [B, R, 160]
            ent_valid = tgt["ent_valid"]       # [B, R]
            pos_0_map = tgt["pos_0_map"]       # [B, R, 2]
            sp_1 = tgt["spawn_traj"]           # [B, 2, B_max, 100]
            sp_rtype = tgt["spawn_rtype"]      # [B, 2, B_max]
            sp_act = tgt["spawn_active"]       # [B, 2, B_max]

            B = ent_1.shape[0]
            dev = ent_1.device
            flow_t = torch.rand(B, device=dev, dtype=torch.float32)
            t_ent = flow_t[:, None, None]
            t_sp = flow_t[:, None, None, None]

            ent_0 = torch.randn_like(ent_1) * ent_valid.unsqueeze(-1).float()
            sp_0 = torch.randn_like(sp_1)

            ent_y = (1.0 - t_ent) * ent_0 + t_ent * ent_1
            sp_y = (1.0 - t_sp) * sp_0 + t_sp * sp_1

            # 30% stochastic inertia dropout on (c_ent_0, q_0[..., 2:4]) during training so
            # mid-chunk maneuver synthesis binds tightly to L2 (S_0 -> S_1, u_flat, xi_world)!
            c_ent_in = ctx["c_ent"][:, c_0]
            q_0_in = ctx["q"][:, c_0]
            if self.training:
                keep_ctx = (torch.rand((B, 1, 1), device=dev) >= 0.30).to(c_ent_in.dtype)
                c_ent_in = c_ent_in * keep_ctx
                q_0_in = q_0_in.clone()
                q_0_in[..., 2:4] = q_0_in[..., 2:4] * keep_ctx.to(q_0_in.dtype)

            v_ent_pred, v_sp_pred, sp_rt_logits = self.predict_velocity(
                ent_y=ent_y,
                spawn_y=sp_y,
                flow_t=flow_t,
                c_ent_0=c_ent_in,
                q_0=q_0_in,
                pos_0_map=pos_0_map,
                rtype_0=b["rtype"][:, f_0],
                owner_raw=b["owner"],
                ent_valid=ent_valid,
                s_0=s_0,
                s_1=s_1,
                u_flat=u_flat,
                xi_world=xi_world,
                static_map=static_map,
                b=b,
            )

            v_ent_tgt = ent_1 - ent_0
            ent_sq = (v_ent_pred - v_ent_tgt).square().mean(dim=-1)  # [B, R]
            loss_ent = (ent_sq * ent_valid.float()).sum() / ent_valid.float().sum().clamp(min=1.0)

            v_sp_tgt = sp_1 - sp_0
            # Weight active spawn slots 4x higher than empty slots so births sharpen quickly
            sp_w = torch.where(sp_act, 4.0, 0.5)
            sp_sq = (v_sp_pred - v_sp_tgt).square().mean(dim=-1)     # [B, 2, B_max]
            loss_spawn_flow = (sp_sq * sp_w).sum() / sp_w.sum().clamp(min=1.0)

            if bool(sp_act.any().item()):
                ce_all = F.cross_entropy(
                    sp_rt_logits.reshape(-1, N_RAW_TYPES),
                    sp_rtype.reshape(-1),
                    reduction="none",
                ).view_as(sp_act)
                # Weight unit-type CE by flow_t (strongest near t -> 1 when slot identity is resolved)
                ce_w = sp_act.float() * (0.25 + 0.75 * flow_t[:, None, None])
                loss_spawn_ce = (ce_all * ce_w).sum() / ce_w.sum().clamp(min=1.0)
            else:
                loss_spawn_ce = sp_rt_logits.sum() * 0.0

            total = loss_ent + 0.5 * loss_spawn_flow + 0.2 * loss_spawn_ce
            return dict(
                loss=total,
                loss_ent=loss_ent.detach(),
                loss_spawn_flow=loss_spawn_flow.detach(),
                loss_spawn_ce=loss_spawn_ce.detach(),
            )

    @torch.no_grad()
    def sample_chunk(
        self,
        b: dict[str, torch.Tensor],
        ctx: dict[str, torch.Tensor],
        f_0: int,
        c_0: int,
        s_0: dict[str, torch.Tensor],
        s_1: dict[str, torch.Tensor],
        u_flat: torch.Tensor,
        xi_world: torch.Tensor,
        static_map: dict[str, torch.Tensor] | None = None,
        steps: int = 6,
        temperature: float = 0.08,
        seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Integrates the `steps`-step ODE `t: 0 -> 1` in parallel to synthesize `16`-frame
        (`0.3125s`-spaced) unit positions, survival, orders, and newly spawned units."""
        with torch.autocast(device_type=u_flat.device.type, enabled=False):
            dev = u_flat.device
            B, _, R = b["mask"].shape
            K = self.k_chunk
            B_m = self.b_max

            center = b["center"].float() if "center" in b else torch.full((B, 2), 100.0, device=dev)
            owner_raw = b["owner"]
            on_map_0 = b["mask"][:, f_0] & b["alive"][:, f_0] & (b["inside"][:, f_0] == 0)
            ent_valid = on_map_0 & (owner_raw < 2)

            x_0 = b["x"][:, f_0].float()
            y_0 = b["y"][:, f_0].float()
            pos_0_world = torch.stack([x_0, y_0], dim=-1)      # [B, R, 2]
            pos_0_map = pos_0_world - center[:, None, :]       # [B, R, 2]

            gen = None
            if seed is not None:
                gen = torch.Generator(device=dev)
                gen.manual_seed(int(seed))

            ent_y = (
                torch.randn((B, R, self.d_ent_traj), generator=gen, device=dev)
                * float(temperature)
                * ent_valid.unsqueeze(-1).float()
            )
            sp_y = torch.randn((B, 2, B_m, self.d_spawn_traj), generator=gen, device=dev)

            dt = 1.0 / float(max(1, steps))
            sp_rt_logits = None
            for step_i in range(max(1, steps)):
                t_cur = torch.full((B,), (step_i + 0.5) * dt, device=dev, dtype=torch.float32)
                v_ent, v_sp, sp_rt_logits = self.predict_velocity(
                    ent_y=ent_y,
                    spawn_y=sp_y,
                    flow_t=t_cur,
                    c_ent_0=ctx["c_ent"][:, c_0],
                    q_0=ctx["q"][:, c_0],
                    pos_0_map=pos_0_map,
                    rtype_0=b["rtype"][:, f_0],
                    owner_raw=owner_raw,
                    ent_valid=ent_valid,
                    s_0=s_0,
                    s_1=s_1,
                    u_flat=u_flat,
                    xi_world=xi_world,
                    static_map=static_map,
                    b=b,
                )
                ent_y = (ent_y + dt * v_ent) * ent_valid.unsqueeze(-1).float()
                sp_y = sp_y + dt * v_sp

            # Unpack world-space trajectories for Inspector & tactical analysis
            ent_steps = ent_y.view(B, R, K, D_ENT_STEP)
            ent_xy = pos_0_world.unsqueeze(2) + ent_steps[..., 0:2].clamp(-3.0, 3.0) * POS_DELTA_SCALE
            ent_vel = ent_steps[..., 2:4].clamp(-3.0, 3.0) * SPEED_SCALE
            ent_hp = ent_steps[..., 4].clamp(0.0, 1.0)
            ent_sh = ent_steps[..., 5].clamp(0.0, 1.0)
            ent_alive_prob = torch.sigmoid(ent_steps[..., 6] * 3.0) * ent_valid.unsqueeze(-1).float()
            ent_ord_prob = torch.sigmoid(ent_steps[..., 7] * 3.0) * ent_valid.unsqueeze(-1).float()
            ent_ord_target_xy = ent_xy + ent_steps[..., 8:10].clamp(-3.0, 3.0) * POS_DELTA_SCALE

            sp_act_prob = torch.sigmoid(sp_y[..., 0] * 3.0)                                   # [B, 2, B_m]
            sp_birth_frame = (((sp_y[..., 1].clamp(-1.0, 1.0) + 1.0) * 0.5) * (K - 1)).round().long()
            sp_origin_xy = center[:, None, None, :] + sp_y[..., 2:4].clamp(-2.0, 2.0) * MAP_SCALE
            sp_steps = sp_y[..., 4:].view(B, 2, B_m, K, D_SPAWN_STEP)
            sp_xy = sp_origin_xy.unsqueeze(3) + sp_steps[..., 0:2].clamp(-3.0, 3.0) * POS_DELTA_SCALE
            sp_hp = sp_steps[..., 4].clamp(0.0, 1.0)
            sp_alive_prob = torch.sigmoid(sp_steps[..., 5] * 3.0) * sp_act_prob.unsqueeze(-1)
            # Zero out rule-table pad 0 when picking most likely unit type
            sp_rt_logits_clean = sp_rt_logits.clone()
            sp_rt_logits_clean[..., 0] = -1e9
            sp_rtype_pred = sp_rt_logits_clean.argmax(dim=-1)

            return dict(
                ent_valid=ent_valid,                 # [B, R]
                ent_owner=owner_raw,                 # [B, R]
                ent_rtype=b["rtype"][:, f_0],        # [B, R]
                ent_pos_0=pos_0_world,               # [B, R, 2]
                ent_xy=ent_xy,                       # [B, R, 16, 2] world coords
                ent_vel=ent_vel,                     # [B, R, 16, 2]
                ent_hp=ent_hp,                       # [B, R, 16]
                ent_sh=ent_sh,                       # [B, R, 16]
                ent_alive_prob=ent_alive_prob,       # [B, R, 16]
                ent_ord_prob=ent_ord_prob,           # [B, R, 16]
                ent_ord_target_xy=ent_ord_target_xy, # [B, R, 16, 2] world coords
                spawn_prob=sp_act_prob,              # [B, 2, B_max]
                spawn_birth_k=sp_birth_frame,        # [B, 2, B_max] in 0..15
                spawn_origin_xy=sp_origin_xy,        # [B, 2, B_max, 2]
                spawn_xy=sp_xy,                      # [B, 2, B_max, 16, 2]
                spawn_hp=sp_hp,                      # [B, 2, B_max, 16]
                spawn_alive_prob=sp_alive_prob,      # [B, 2, B_max, 16]
                spawn_rtype=sp_rtype_pred,           # [B, 2, B_max]
            )
