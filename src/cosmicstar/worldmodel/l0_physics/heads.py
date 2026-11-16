"""Grounded next-tubelet prediction heads for units, queues, births, AoE effects, and recipients."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..core.constants import C_GROUPS, N_BUFF_IDS, N_EFFECT_IDS, N_RAW_ABILITY_IDS
from ..data.game_store import NEUTRAL, PROD, QUEUE
from ..nn.attention import Heads, PairAttend
from ..nn.layers import mlp
from ..nn.rope import ROPE_SHORTEST, RoPE2D, cos_sin
from .encoders import N_FLAGS, I
from .trunk import GRID, N_CODES, pointer_pairs

# Active SC2 unit flag bits out of N_FLAGS = 16:
#   u_flags  (0..6): FLYING, BURROWED, IN_CARGO, POWERED, HALLUCINATION, DETECTOR, RADAR
#   u_flags2 (8..10): ACTIVE, BLIP, ONSCREEN (bit 10 kept so the model tracks camera
#                     movement and player visual attention)
# NOTE: Unused engine padding bits (7, 11, 12, 13, 14, 15) are dropped from loss
# scoring for now; RowHead.flags retains width N_FLAGS = 16 for checkpoint stability.
ACTIVE_FLAG_BITS = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10)
REG_ALWAYS_VALID = ("contents", "facing", "z")  # valid even on fog-of-war SNAPSHOT rows

MAX_BIRTHS, MARGIN = 64, 2  # birth count classes; extra queries per frame
MAX_EFF_SPAWNS, EFF_MARGIN = 24, 2  # spawned AoE count classes; extra queries per step
N_EFF_SPAWN_TYPES = 16  # raw SC2 EffectData ids 0..12
MAX_EFF_DOTS = 12  # 1 for circle AoEs, up to 11 for LurkerSpines/ThermalLances rays
MAX_EFF_AGE = 8  # sub-tubelet spawn age classes (0 .. k - 1)
EFF_RADII = (
    0.0,
    1.5,   # 1: PsiStorm
    4.0,   # 2: GuardianShield
    4.0,   # 3: TemporalFieldGrowing
    4.0,   # 4: TemporalField
    0.15,  # 5: ThermalLances
    13.0,  # 6: ScannerSweep
    0.5,   # 7: NukeDot
    5.0,   # 8: LiberatorZoneSetup
    5.0,   # 9: LiberatorZone
    2.0,   # 10: BlindingCloud
    0.5,   # 11: CorrosiveBile
    0.5,   # 12: LurkerSpines
    0.0,
    0.0,
    0.0,
)
POS_SCALE = 0.29  # a fast unit's per-loop move, map units
POS_BOOST = 4.0
DISCOUNT = 0.05  # critters' whole loss and larvae's position
GRID = 16  # the grid's cells per side
N_CODES = 5  # pair codes of the action pointers, see pointer_pairs
N_UNIT_EDGES = 6  # eng->tgt, tgt->eng, child->par, par->child, ord0->tgt, tgt->ord0
N_PTRS = 2 + QUEUE  # engaged, parent, ord[0..QUEUE-1]


class RowHead(nn.Module):
    """A row's token -> its raw fields at some frame. One head, shared by rows and birth
    queries, with zero-initialized signed f_src copy/flip residual gates on unit
    transitions."""

    def __init__(self, d, n_units, n_abilities):
        super().__init__()
        self.unit, self.inside = nn.Linear(d, n_units), nn.Linear(d, 3)
        self.alive = nn.Linear(d, 1)
        nn.init.constant_(self.alive.bias, -5.0)
        self.reg = nn.Linear(d, 2 + len(REG))
        self.flags, self.display, self.cloak = (
            nn.Linear(d, N_FLAGS),
            nn.Linear(d, 4),
            nn.Linear(d, 4),
        )
        self.slot = nn.Embedding(QUEUE + PROD, d)
        self.anc_abil_emb = nn.Embedding(n_abilities, d)
        self.anc_reg_proj = mlp(4, d)
        self.slot_heads = max(1, min(4, d // 32))
        self.slot_dh = d // self.slot_heads
        self.slot_n1 = nn.LayerNorm(d)
        self.slot_qkv = nn.Linear(d, 3 * d)
        self.slot_o = nn.Linear(d, d)
        self.slot_n2 = nn.LayerNorm(d)
        self.slot_mix = mlp(d, d, 2 * d)
        self.slot_abil = nn.Linear(d, n_abilities)
        self.abil_shift = nn.Linear(d, (QUEUE + PROD) * (QUEUE + PROD))
        nn.init.zeros_(self.abil_shift.weight)
        nn.init.zeros_(self.abil_shift.bias)
        self.reg_route = nn.Linear(d, QUEUE + PROD + 1)
        self.reg_route_diag = nn.Linear(d, QUEUE + PROD)
        nn.init.zeros_(self.reg_route_diag.weight)
        nn.init.zeros_(self.reg_route_diag.bias)
        self.slot_reg = nn.Linear(
            d, 4
        )  # order target dx, dy, whether it has one; progress
        self.buffs, self.groups = nn.Linear(d, N_BUFF_IDS), nn.Linear(d, C_GROUPS)
        self.bits = nn.Linear(d, 3)  # has an add-on, an engaged target, a rally point
        self.rally = nn.Linear(d, 2)
        self.abilities = nn.Linear(d, N_RAW_ABILITY_IDS)
        # Signed zero-init f_src residual gates (>0 reinforces stay, <0 flips/suppresses)
        self.stay_type = nn.Linear(d, 1)
        self.stay_inside = nn.Linear(d, 1)
        self.stay_display = nn.Linear(d, 1)
        self.stay_cloak = nn.Linear(d, 1)
        self.stay_flags = nn.Linear(d, N_FLAGS)
        self.stay_groups = nn.Linear(d, C_GROUPS)
        self.stay_bits = nn.Linear(d, 3)
        self.stay_buffs = nn.Linear(d, N_BUFF_IDS)
        self.stay_abilities = nn.Linear(d, 1)
        self.rally_keep = nn.Linear(d, 1)
        self.rally_delta = nn.Linear(d, 2)
        for m_s in (
            self.stay_type,
            self.stay_inside,
            self.stay_display,
            self.stay_cloak,
            self.stay_flags,
            self.stay_groups,
            self.stay_bits,
            self.stay_buffs,
            self.stay_abilities,
            self.rally_keep,
            self.rally_delta,
        ):
            nn.init.zeros_(m_s.weight)
            nn.init.zeros_(m_s.bias)
        self.ptr_q = nn.Embedding(
            N_PTRS, d
        )  # engaged target, container, each order's target
        ptr_dh = max(4, (d // 4) * 4)
        self.ptr_mix = mlp(d, d)
        self.ptr_qh = nn.Linear(d, ptr_dh, bias=False)
        self.ptr_kh = nn.Linear(d, ptr_dh, bias=False)
        self.ptr_none = nn.Linear(d, 1)  # column R: "no target (-1)"
        self.ptr_rope = RoPE2D(ptr_dh, lo=ROPE_SHORTEST, hi=256.0)
        self.ptr_shift = nn.Linear(d, N_PTRS * N_PTRS)
        nn.init.zeros_(self.ptr_shift.weight)
        nn.init.zeros_(self.ptr_shift.bias)

    def forward(self, x):
        return dict(
            type=self.unit(x),
            inside=self.inside(x),
            alive=self.alive(x).squeeze(-1),
            reg=self.reg(x),
            flags=self.flags(x),
            display=self.display(x),
            cloak=self.cloak(x),
            buffs=self.buffs(x),
            groups=self.groups(x),
            bits=self.bits(x),
            rally=self.rally(x),
        )

    def predict_next(
        self,
        x,
        typ_fp,
        ins_fp,
        disp_fp,
        cloak_fp,
        flags_fp,
        grp_fp,
        bits_fp,
        buffs_fp,
        rf_fp,
    ):
        """Decode next-tubelet unit state with signed f_src copy/flip residual gates."""
        n_u = self.unit.out_features
        typ_l = self.unit(x).float().scatter_add(
            1, typ_fp.long().clamp(0, n_u - 1)[:, None], self.stay_type(x).float()
        )
        ins_l = self.inside(x).float().scatter_add(
            1, ins_fp.long().clamp(0, 2)[:, None], self.stay_inside(x).float()
        )
        disp_l = self.display(x).float().scatter_add(
            1, disp_fp.long().clamp(0, 3)[:, None], self.stay_display(x).float()
        )
        cloak_l = self.cloak(x).float().scatter_add(
            1, cloak_fp.long().clamp(0, 3)[:, None], self.stay_cloak(x).float()
        )
        flags_l = self.flags(x).float() + self.stay_flags(x).float() * (
            2.0 * flags_fp.float() - 1.0
        )
        grp_l = self.groups(x).float() + self.stay_groups(x).float() * (
            2.0 * grp_fp.float() - 1.0
        )
        bits_l = self.bits(x).float() + self.stay_bits(x).float() * (
            2.0 * bits_fp.float() - 1.0
        )
        buffs_l = self.buffs(x).float() + self.stay_buffs(x).float() * (
            2.0 * buffs_fp.float() - 1.0
        )
        rally_fresh = self.rally(x).float()
        keep_r = torch.sigmoid(self.rally_keep(x).float())
        rally_cont = (
            keep_r * rf_fp[:, 1:3].float()
            + (1.0 - keep_r) * rally_fresh
            + self.rally_delta(x).float()
        )
        rally_p = torch.where((rf_fp[:, 0:1] > 0), rally_cont, rally_fresh)
        return dict(
            type=typ_l,
            inside=ins_l,
            alive=self.alive(x).squeeze(-1).float(),
            reg=self.reg(x).float(),
            flags=flags_l,
            display=disp_l,
            cloak=cloak_l,
            buffs=buffs_l,
            groups=grp_l,
            bits=bits_l,
            rally=rally_p,
        )

    def predict_abilities(self, x, src_ab=None):
        logits = self.abilities(x).float()
        if src_ab is not None:
            logits = logits + self.stay_abilities(x).float() * (
                2.0 * src_ab.float() - 1.0
            )
        return logits

    def slots(self, x, anc_abil=None, anc_reg=None):
        N, S_q, d = len(x), QUEUE + PROD, x.shape[-1]
        h = x[:, None] + self.slot.weight[None].to(x.dtype)
        if anc_abil is not None and anc_reg is not None and N > 0:
            anc_in = torch.cat(
                [anc_reg[..., :3], 1.0 - anc_reg[..., 3:4]], dim=-1
            ).to(x.dtype)
            h = h + self.anc_abil_emb(anc_abil) + self.anc_reg_proj(anc_in)
            sq, sk, sv = (
                self.slot_qkv(self.slot_n1(h))
                .view(N, S_q, 3, self.slot_heads, self.slot_dh)
                .permute(2, 0, 3, 1, 4)
            )
            sa = F.scaled_dot_product_attention(sq, sk, sv)
            h = h + self.slot_o(sa.transpose(1, 2).reshape(N, S_q, d))
        y = h + self.slot_mix(self.slot_n2(h))
        logits = self.slot_abil(y).float()
        if anc_abil is not None and N > 0:
            shift = self.abil_shift(x).float().view(N, S_q, S_q)
            idx = anc_abil[:, None, :].expand(-1, S_q, -1)
            logits = logits.scatter_add(2, idx, shift)
        delta_reg = self.slot_reg(y).float()
        if anc_reg is not None and N > 0:
            rt_logits = self.reg_route(y).float()
            diag_idx = torch.arange(S_q, device=x.device)[None, :, None].expand(
                N, S_q, 1
            )
            rt_logits = rt_logits.scatter_add(
                2, diag_idx, self.reg_route_diag(x).float()[..., None]
            )
            w = F.softmax(rt_logits, dim=-1)
            routed = torch.einsum("nsm,nmc->nsc", w[..., :S_q], anc_reg.float())
            reg_out = routed + delta_reg
        else:
            reg_out = delta_reg
        return logits, reg_out

    def pointers(self, x, rows, xy_q=None, xy_k=None, now_ptr=None):
        """x [N, d] tokens, rows [N, R, d] their tubelets' rows -> [N, N_PTRS, R + 1]
        scores of each pointer over unit rows 0..R-1 plus column R ("no target (-1)"),
        combining RoPE2D spatial target selection and gated routing from the anchor
        frame's existing pointers (keep / queue-shift / aggro / clear)."""
        R = rows.shape[1]
        q_in = self.ptr_mix(x[:, None] + self.ptr_q.weight[None])
        none_logit = self.ptr_none(q_in).squeeze(-1).float()[..., None]
        if R > 0:
            q = self.ptr_qh(q_in)
            k = self.ptr_kh(rows)
            if xy_q is not None and xy_k is not None:
                q = self.ptr_rope.rotate(
                    q[:, None], self.ptr_rope.angles(xy_q[:, None])
                ).squeeze(1)
                k = self.ptr_rope.rotate(
                    k[:, None], self.ptr_rope.angles(xy_k)
                ).squeeze(1)
            row_logits = torch.einsum("npd,nrd->npr", q, k).float() / math.sqrt(
                q.shape[-1]
            )
            logits = torch.cat([row_logits, none_logit], dim=-1)
        else:
            logits = none_logit
        if now_ptr is not None and len(x) > 0:
            shift = self.ptr_shift(x).float().view(len(x), N_PTRS, N_PTRS)
            idx = torch.where((now_ptr >= 0) & (now_ptr < R), now_ptr, R)[
                :, None, :
            ].expand(-1, N_PTRS, -1)
            logits = logits.scatter_add(2, idx, shift)
        return logits


# regressed as change, scaled: (name, divisor); static unit-table constants excluded
REG_LEGACY_NAMES = (
    "hp",
    "shield",
    "energy",
    "build",
    "contents",
    "harvesters",
    "ideal_harvesters",
    "hp_max",
    "shield_max",
    "weapon_cooldown",
    "cargo_used",
    "buff_remain",
    "facing",
    "radius",
    "cargo_max",
    "buff_max",
    "detect_range",
    "radar_range",
    "atk_up",
    "arm_up",
    "shd_up",
    "z",
)
REG = (
    ("hp", 100.0),
    ("shield", 100.0),
    ("energy", 50.0),
    ("build", 0.1),
    ("contents", 100.0),
    ("harvesters", 4.0),
    ("hp_max", 100.0),
    ("shield_max", 100.0),
    ("weapon_cooldown", 8.0),
    ("cargo_used", 4.0),
    ("buff_remain", 50.0),
    ("facing", 1.0),
    ("buff_max", 50.0),
    ("atk_up", 1.0),
    ("arm_up", 1.0),
    ("shd_up", 1.0),
    ("z", 1.0),
)


class _CountAttnPool(nn.Module):
    """Multi-query cross-attention pooler from global context g_ctx [B, M, d] over
    [self-sink, rows, acts, optional effs], zero-initialized at output for exact
    warm-start compatibility."""

    def __init__(self, d, heads=4, n_queries=4):
        super().__init__()
        self.H = heads
        self.Q = n_queries
        self.dh = max(8, d // heads)
        inner = self.H * self.dh
        self.q_emb = nn.Embedding(n_queries, d)
        self.kind = nn.Embedding(4, d)  # 0: sink, 1: row, 2: act, 3: eff
        self.qn = nn.LayerNorm(d)
        self.kn = nn.LayerNorm(d)
        self.q = nn.Linear(d, inner, bias=False)
        self.kv = nn.Linear(d, 2 * inner, bias=False)
        self.out = mlp(n_queries * inner, d, 2 * d)
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(
        self,
        g_ctx,
        r_mem=None,
        rmask=None,
        a_mem=None,
        amask=None,
        e_mem=None,
        emask=None,
    ):
        B, M, d = g_ctx.shape
        BM = B * M
        dt = (
            torch.get_autocast_dtype("cuda")
            if g_ctx.is_cuda and torch.is_autocast_enabled("cuda")
            else g_ctx.dtype
        )
        g_f = g_ctx.reshape(BM, d).to(dt)
        q_in = g_f[:, None] + self.q_emb.weight[None].to(dt)
        sink = g_f[:, None] + self.kind.weight[0].to(dt)
        mem_parts = [sink]
        mask_parts = [torch.ones(BM, 1, dtype=torch.bool, device=g_ctx.device)]
        if r_mem is not None and rmask is not None and r_mem.shape[2] > 0:
            mem_parts.append(
                r_mem.reshape(BM, -1, d).to(dt) + self.kind.weight[1].to(dt)
            )
            mask_parts.append(rmask.reshape(BM, -1))
        if a_mem is not None and amask is not None and a_mem.shape[2] > 0:
            mem_parts.append(
                a_mem.reshape(BM, -1, d).to(dt) + self.kind.weight[2].to(dt)
            )
            mask_parts.append(amask.reshape(BM, -1))
        if e_mem is not None and emask is not None and e_mem.shape[2] > 0:
            mem_parts.append(
                e_mem.reshape(BM, -1, d).to(dt) + self.kind.weight[3].to(dt)
            )
            mask_parts.append(emask.reshape(BM, -1))
        mem = torch.cat(mem_parts, dim=1)
        mmask = torch.cat(mask_parts, dim=1)
        q = self.q(self.qn(q_in)).view(BM, self.Q, self.H, self.dh).transpose(1, 2)
        k, v = (
            self.kv(self.kn(mem))
            .view(BM, -1, 2, self.H, self.dh)
            .permute(2, 0, 3, 1, 4)
        )
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mmask[:, None, None])
        pooled = attn.transpose(1, 2).reshape(BM, self.Q * self.H * self.dh)
        return self.out(pooled).view(B, M, d).to(g_ctx.dtype)


class BirthHead(nn.Module):
    """Per tubelet, how many player units appear in (f_src, f_tgt], then that many
    queries plus a margin in a single symmetric pass: causal query self-attention +
    final-step cross-attention over unit rows, map cells, player globals, and the
    transition actionlet, with parent-, cell-, and placed-action-anchored spawns."""

    def __init__(self, d, heads, head, n_glob=3):
        super().__init__()
        self.glob_proj = mlp(n_glob * d, d, 2 * d)
        self.row_cnt_gate = nn.Linear(d, 1)
        self.row_cnt_feat = nn.Linear(d, d)
        nn.init.constant_(self.row_cnt_gate.bias, -3.0)
        nn.init.zeros_(self.row_cnt_feat.weight)
        nn.init.zeros_(self.row_cnt_feat.bias)
        self.act_cnt_gate = nn.Linear(d, 1)
        self.act_cnt_feat = nn.Linear(d, d)
        nn.init.constant_(self.act_cnt_gate.bias, -3.0)
        nn.init.zeros_(self.act_cnt_feat.weight)
        nn.init.zeros_(self.act_cnt_feat.bias)
        self.cnt_pool = _CountAttnPool(d, heads=4, n_queries=4)
        self.count = mlp(d, MAX_BIRTHS)
        self.query = nn.Embedding(MAX_BIRTHS + MARGIN, d)
        self.row_xy = mlp(2, d)
        self.cell_xy = mlp(2, d)
        self.act_xy = mlp(2, d)
        self.kind_tag = nn.Embedding(4, d)  # 0: row, 1: cell, 2: glob, 3: act
        self.dh = min(32, d)
        self.H = max(1, d // self.dh)
        inner = self.H * self.dh
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    dict(
                        sn=nn.LayerNorm(d),
                        sqkv=nn.Linear(d, 3 * inner),
                        so=nn.Linear(inner, d),
                        cn=nn.LayerNorm(d),
                        kn=nn.LayerNorm(d),
                        cq=nn.Linear(d, inner),
                        ckv=nn.Linear(d, 2 * inner),
                        co=nn.Linear(inner, d),
                        fn=nn.LayerNorm(d),
                        ff=mlp(d, d, 4 * d),
                    )
                )
                for _ in range(2)
            ]
        )
        self.out_norm = nn.LayerNorm(d)
        self.real = nn.Linear(d, 1)
        self.owner = nn.Linear(d, 2)
        self.init_stat = nn.Linear(d, 4)
        self.parent_q, self.parent_k, self.no_parent = (
            nn.Linear(d, d),
            nn.Linear(d, d),
            nn.Linear(d, 1),
        )
        self.cell_q, self.cell_k = nn.Linear(d, d), nn.Linear(d, d)
        self.act_q, self.act_k = nn.Linear(d, d), nn.Linear(d, d)
        self.pos_rel = nn.Linear(d, 2)
        self.pos_cell_rel = nn.Linear(d, 2)
        nn.init.zeros_(self.pos_rel.weight)
        nn.init.zeros_(self.pos_rel.bias)
        nn.init.zeros_(self.pos_cell_rel.weight)
        nn.init.zeros_(self.pos_cell_rel.bias)
        self.head = [head]

    def predict_count(
        self,
        glob,
        rows,
        rmask,
        acts=None,
        amask=None,
        anc_txy=None,
        aplaced=None,
        act_txy=None,
    ):
        """glob [B, M, 3, d], rows [B, M, R, d], rmask [B, M, R], optional transition
        acts [B, M, A_m, d] / amask [B, M, A_m] -> [B, M, MAX_BIRTHS]."""
        g_ctx = self.glob_proj(glob.flatten(2))
        r_mem = rows
        if rows.shape[2] > 0:
            gate = torch.sigmoid(self.row_cnt_gate(rows)) * rmask[..., None].to(
                rows.dtype
            )
            g_ctx = g_ctx + (gate * self.row_cnt_feat(rows)).sum(2)
            if anc_txy is not None:
                r_mem = rows + self.row_xy(anc_txy.to(rows.dtype)) * rmask[
                    ..., None
                ].to(rows.dtype)
        a_mem = acts
        if acts is not None and amask is not None and acts.shape[2] > 0:
            agate = torch.sigmoid(self.act_cnt_gate(acts)) * amask[..., None].to(
                acts.dtype
            )
            g_ctx = g_ctx + (agate * self.act_cnt_feat(acts)).sum(2)
            if act_txy is not None and aplaced is not None:
                a_mem = acts + self.act_xy(act_txy.to(acts.dtype)) * aplaced[
                    ..., None
                ].to(acts.dtype)
        g_ctx = g_ctx + self.cnt_pool(
            g_ctx, r_mem=r_mem, rmask=rmask, a_mem=a_mem, amask=amask
        )
        return self.count(g_ctx)

    def _run_queries(
        self,
        glob,
        rows,
        rmask,
        cells,
        anc_txy,
        cell_txy,
        acts,
        amask,
        aplaced,
        act_txy,
        n,
    ):
        K = rows.shape[0]
        dt = (
            torch.get_autocast_dtype("cuda")
            if rows.is_cuda and torch.is_autocast_enabled("cuda")
            else rows.dtype
        )
        rows_d, cells_d, glob_d = rows.to(dt), cells.to(dt), glob.to(dt)
        g_ctx = self.glob_proj(glob_d.flatten(1))
        x = g_ctx[:, None] + self.query.weight[:n][None].to(dt)
        r_mem = (
            rows_d
            + self.row_xy(anc_txy.to(dt)) * rmask[..., None].to(dt)
            + self.kind_tag.weight[0].to(dt)
        )
        c_mem = (
            cells_d + self.cell_xy(cell_txy.to(dt)) + self.kind_tag.weight[1].to(dt)
        )
        g_mem = glob_d + self.kind_tag.weight[2].to(dt)
        mem_parts = [r_mem, c_mem, g_mem]
        mask_parts = [
            rmask,
            torch.ones(
                K,
                cells.shape[1] + glob.shape[1],
                dtype=torch.bool,
                device=rows.device,
            ),
        ]
        a_mem = None
        if acts is not None and acts.shape[1] > 0:
            acts_d = acts.to(dt)
            a_mem = (
                acts_d
                + self.act_xy(act_txy.to(dt)) * aplaced[..., None].to(dt)
                + self.kind_tag.weight[3].to(dt)
            )
            mem_parts.append(a_mem)
            mask_parts.append(amask)
        mem = torch.cat(mem_parts, dim=1)
        mmask = torch.cat(mask_parts, dim=1)
        for lyr in self.layers:
            sq, sk, sv = (
                lyr["sqkv"](lyr["sn"](x))
                .view(K, n, 3, self.H, self.dh)
                .permute(2, 0, 3, 1, 4)
            )
            sa = F.scaled_dot_product_attention(sq, sk, sv, is_causal=True)
            x = x + lyr["so"](sa.transpose(1, 2).reshape(K, n, self.H * self.dh))
            cq = lyr["cq"](lyr["cn"](x)).view(K, n, self.H, self.dh).transpose(1, 2)
            ck, cv = (
                lyr["ckv"](lyr["kn"](mem))
                .view(K, -1, 2, self.H, self.dh)
                .permute(2, 0, 3, 1, 4)
            )
            ca = F.scaled_dot_product_attention(
                cq, ck, cv, attn_mask=mmask[:, None, None]
            )
            x = x + lyr["co"](ca.transpose(1, 2).reshape(K, n, self.H * self.dh))
            x = x + lyr["ff"](lyr["fn"](x))
        x = self.out_norm(x)
        return x, r_mem, c_mem, a_mem

    def forward(
        self,
        glob,
        rows,
        rmask,
        cells,
        anc_txy,
        cell_txy,
        acts,
        amask,
        aplaced,
        act_txy,
        n,
        need_fields=True,
        grad_ckpt=False,
    ):
        """glob [K, 3, d], rows [K, R, d], rmask [K, R], cells [K, S, d],
        anc_txy [K, R, 2], cell_txy [K, S, 2], acts [K, A_m, d], amask [K, A_m],
        aplaced [K, A_m], act_txy [K, A_m, 2], n queries -> outputs per (step, query)."""
        K, R, d = rows.shape
        if not need_fields:
            from torch.utils.checkpoint import checkpoint

            use_ckpt = (
                grad_ckpt
                and torch.is_grad_enabled()
                and any(p.requires_grad for p in self.parameters())
            )
            chunk = 256
            reals = []
            for s in range(0, K, chunk):
                e = min(K, s + chunk)
                fn = lambda g_, r_, rm_, c_, at_, ct_, ac_, am_, ap_, axy_: self.real(
                    self._run_queries(
                        g_, r_, rm_, c_, at_, ct_, ac_, am_, ap_, axy_, n
                    )[0]
                ).squeeze(-1)
                args = (
                    glob[s:e],
                    rows[s:e],
                    rmask[s:e],
                    cells[s:e],
                    anc_txy[s:e],
                    cell_txy[s:e],
                    acts[s:e],
                    amask[s:e],
                    aplaced[s:e],
                    act_txy[s:e],
                )
                if use_ckpt:
                    reals.append(checkpoint(fn, *args, use_reentrant=False))
                else:
                    reals.append(fn(*args))
            return None, torch.cat(reals, dim=0), None, None, None, None
        x, r_mem, c_mem, a_mem = self._run_queries(
            glob, rows, rmask, cells, anc_txy, cell_txy, acts, amask, aplaced, act_txy, n
        )
        real = self.real(x).squeeze(-1)
        owner = self.owner(x)
        init_stat = self.init_stat(x)
        parent = (
            torch.einsum("kqd,krd->kqr", self.parent_q(x), self.parent_k(r_mem)).float()
            / math.sqrt(d)
        )
        parent = parent.masked_fill(~rmask[:, None], -1e4)
        parent = torch.cat([parent, self.no_parent(x).float()], -1)
        p_par = F.softmax(parent, dim=-1)
        w_par, w_nopar = p_par[..., :R], p_par[..., R:]
        par_xy = (
            torch.einsum("kqr,krp->kqp", w_par, anc_txy.float())
            if R > 0
            else x.new_zeros((K, n, 2), dtype=torch.float32)
        )
        c_logits = (
            torch.einsum("kqd,ksd->kqs", self.cell_q(x), self.cell_k(c_mem)).float()
            / math.sqrt(d)
        )
        if a_mem is not None and a_mem.shape[1] > 0:
            a_logits = (
                torch.einsum("kqd,kad->kqa", self.act_q(x), self.act_k(a_mem)).float()
                / math.sqrt(d)
            )
            a_logits = a_logits.masked_fill(~aplaced[:, None], -1e4)
            w_ca = F.softmax(torch.cat([c_logits, a_logits], dim=-1), dim=-1)
            ca_txy = torch.cat([cell_txy.float(), act_txy.float()], dim=1)
            nopar_xy = torch.einsum("kqm,kmp->kqp", w_ca, ca_txy)
        else:
            w_cell = F.softmax(c_logits, dim=-1)
            nopar_xy = torch.einsum("kqs,ksp->kqp", w_cell, cell_txy.float())
        pos = (
            par_xy
            + (1.0 - w_nopar) * self.pos_rel(x).float()
            + w_nopar * (nopar_xy + self.pos_cell_rel(x).float())
        )
        return x, real, pos, parent, owner, init_stat


def _eff_age_features(age, omega):
    """Exact loop-age features [..., 18]: log1p(age)/6, age/300, and 8-band sin/cos."""
    ag = age.float().clamp(min=0.0)
    ang = ag[..., None] * omega
    return torch.cat(
        [
            torch.log1p(ag)[..., None] / 6.0,
            (ag / 300.0)[..., None],
            ang.sin(),
            ang.cos(),
        ],
        dim=-1,
    )


class EffPersistHead(nn.Module):
    """Per active effect token at f_src: predicts whether it persists to f_tgt and its
    normalized 2D displacement (dx, dy) / (POS_SCALE * k) for moving effects, with a
    direct (eff_type, eff_age) skip path so exact engine durations never blur."""

    def __init__(self, d):
        super().__init__()
        self.register_buffer(
            "age_omega",
            2.0
            * math.pi
            / torch.tensor([4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]),
            persistent=False,
        )
        self.type_emb = nn.Embedding(N_EFF_SPAWN_TYPES, d)
        self.age_mlp = mlp(18, d, d)
        self.ctx_out = nn.Linear(d, d)
        nn.init.zeros_(self.ctx_out.weight)
        nn.init.zeros_(self.ctx_out.bias)
        self.mix = mlp(d, d, 2 * d)
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)
        self.persist = nn.Linear(d, 1)
        nn.init.zeros_(self.persist.weight)
        nn.init.constant_(self.persist.bias, 1.5)
        self.dxy = nn.Linear(d, 2)
        nn.init.zeros_(self.dxy.weight)
        nn.init.zeros_(self.dxy.bias)

    def forward(self, e, ty_s=None, age_s=None):
        e_in = e
        if ty_s is not None and age_s is not None and len(e) > 0:
            af = _eff_age_features(age_s, self.age_omega).to(e.dtype)
            ctx = F.gelu(
                self.type_emb(ty_s.long().clamp(0, N_EFF_SPAWN_TYPES - 1)).to(e.dtype)
                + self.age_mlp(af)
            )
            e_in = e + self.ctx_out(ctx)
        h = e_in + self.mix(e_in)
        return self.persist(h).squeeze(-1), self.dxy(h)


class EffSpawnHead(nn.Module):
    """Per tubelet, predicts how many new AoE effects (circles or collinear Lurker/
    Colossus rays) appear in (f_src, f_tgt], then decodes queries via causal self-
    attention + final-step cross-attention over unit rows, map cells, globals, the
    transition actionlet, and active f_src precursor effects (e.g. LiberatorZoneSetup ->
    LiberatorZone, TemporalFieldGrowing -> TemporalField)."""

    def __init__(self, d, heads, n_glob=3):
        super().__init__()
        self.register_buffer(
            "age_omega",
            2.0
            * math.pi
            / torch.tensor([4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]),
            persistent=False,
        )
        self.glob_proj = mlp(n_glob * d, d, 2 * d)
        self.row_cnt_gate = nn.Linear(d, 1)
        self.row_cnt_feat = nn.Linear(d, d)
        nn.init.constant_(self.row_cnt_gate.bias, -3.0)
        nn.init.zeros_(self.row_cnt_feat.weight)
        nn.init.zeros_(self.row_cnt_feat.bias)
        self.act_cnt_gate = nn.Linear(d, 1)
        self.act_cnt_feat = nn.Linear(d, d)
        nn.init.constant_(self.act_cnt_gate.bias, -3.0)
        nn.init.zeros_(self.act_cnt_feat.weight)
        nn.init.zeros_(self.act_cnt_feat.bias)
        self.cnt_pool = _CountAttnPool(d, heads=4, n_queries=4)
        self.count = mlp(d, MAX_EFF_SPAWNS)
        with torch.no_grad():
            self.count[-1].bias[0] = 2.5
        self.query = nn.Embedding(MAX_EFF_SPAWNS + EFF_MARGIN, d)
        self.row_xy = mlp(2, d)
        self.cell_xy = mlp(2, d)
        self.act_xy = mlp(2, d)
        self.eff_xy = mlp(2, d)
        self.eff_type_in = nn.Embedding(N_EFF_SPAWN_TYPES, d)
        self.eff_age_in = mlp(18, d, d)
        self.kind_tag = nn.Embedding(4, d)  # 0: row, 1: cell, 2: glob, 3: act
        self.eff_kind_tag = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.eff_kind_tag, std=0.02)
        self.dh = min(32, d)
        self.H = max(1, d // self.dh)
        inner = self.H * self.dh
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    dict(
                        sn=nn.LayerNorm(d),
                        sqkv=nn.Linear(d, 3 * inner),
                        so=nn.Linear(inner, d),
                        cn=nn.LayerNorm(d),
                        kn=nn.LayerNorm(d),
                        cq=nn.Linear(d, inner),
                        ckv=nn.Linear(d, 2 * inner),
                        co=nn.Linear(inner, d),
                        fn=nn.LayerNorm(d),
                        ff=mlp(d, d, 4 * d),
                    )
                )
                for _ in range(2)
            ]
        )
        self.out_norm = nn.LayerNorm(d)
        self.real = nn.Linear(d, 1)
        nn.init.constant_(self.real.bias, -1.5)
        self.eff_type = nn.Linear(d, N_EFF_SPAWN_TYPES)
        self.owner = nn.Linear(d, 2)
        self.views = nn.Linear(d, 2)
        self.n_dots = nn.Linear(d, MAX_EFF_DOTS)
        self.age = nn.Linear(d, MAX_EFF_AGE)
        self.row_q, self.row_k = nn.Linear(d, d), nn.Linear(d, d)
        self.cell_q, self.cell_k = nn.Linear(d, d), nn.Linear(d, d)
        self.act_q, self.act_k = nn.Linear(d, d), nn.Linear(d, d)
        self.eff_q, self.eff_k = nn.Linear(d, d), nn.Linear(d, d)
        nn.init.zeros_(self.eff_q.weight)
        nn.init.zeros_(self.eff_q.bias)
        self.eff_ptr_bias = nn.Parameter(torch.tensor(-4.0))
        self.pos_rel = nn.Linear(d, 2)
        self.ray_dxy = nn.Linear(d, 2)
        nn.init.zeros_(self.pos_rel.weight)
        nn.init.zeros_(self.pos_rel.bias)
        nn.init.zeros_(self.ray_dxy.weight)
        nn.init.zeros_(self.ray_dxy.bias)

    def _encode_eff_mem(self, effs, emask, eff_txy, eff_ty, eff_age, dt, add_tag=True):
        e_d = effs.to(dt)
        af = _eff_age_features(eff_age, self.age_omega).to(dt)
        em_d = emask[..., None].to(dt)
        e_mem = (
            e_d
            + (
                self.eff_xy(eff_txy.to(dt))
                + self.eff_type_in(eff_ty.long().clamp(0, N_EFF_SPAWN_TYPES - 1)).to(dt)
                + self.eff_age_in(af)
            )
            * em_d
        )
        if add_tag:
            e_mem = e_mem + self.eff_kind_tag.to(dt)
        return e_mem

    def predict_count(
        self,
        glob,
        rows,
        rmask,
        acts=None,
        amask=None,
        anc_txy=None,
        aplaced=None,
        act_txy=None,
        effs=None,
        emask=None,
        eff_txy=None,
        eff_ty=None,
        eff_age=None,
    ):
        g_ctx = self.glob_proj(glob.flatten(2))
        r_mem = rows
        if rows.shape[2] > 0:
            gate = torch.sigmoid(self.row_cnt_gate(rows)) * rmask[..., None].to(
                rows.dtype
            )
            g_ctx = g_ctx + (gate * self.row_cnt_feat(rows)).sum(2)
            if anc_txy is not None:
                r_mem = rows + self.row_xy(anc_txy.to(rows.dtype)) * rmask[
                    ..., None
                ].to(rows.dtype)
        a_mem = acts
        if acts is not None and amask is not None and acts.shape[2] > 0:
            agate = torch.sigmoid(self.act_cnt_gate(acts)) * amask[..., None].to(
                acts.dtype
            )
            g_ctx = g_ctx + (agate * self.act_cnt_feat(acts)).sum(2)
            if act_txy is not None and aplaced is not None:
                a_mem = acts + self.act_xy(act_txy.to(acts.dtype)) * aplaced[
                    ..., None
                ].to(acts.dtype)
        e_mem = None
        if (
            effs is not None
            and emask is not None
            and eff_txy is not None
            and eff_ty is not None
            and eff_age is not None
            and effs.shape[2] > 0
        ):
            dt = (
                torch.get_autocast_dtype("cuda")
                if rows.is_cuda and torch.is_autocast_enabled("cuda")
                else rows.dtype
            )
            e_mem = self._encode_eff_mem(
                effs, emask, eff_txy, eff_ty, eff_age, dt, add_tag=False
            )
        g_ctx = g_ctx + self.cnt_pool(
            g_ctx,
            r_mem=r_mem,
            rmask=rmask,
            a_mem=a_mem,
            amask=amask,
            e_mem=e_mem,
            emask=emask,
        )
        return self.count(g_ctx)

    def _run_queries(
        self,
        glob,
        rows,
        rmask,
        cells,
        anc_txy,
        cell_txy,
        acts,
        amask,
        aplaced,
        act_txy,
        n,
        effs=None,
        emask=None,
        eff_txy=None,
        eff_ty=None,
        eff_age=None,
    ):
        K = rows.shape[0]
        dt = (
            torch.get_autocast_dtype("cuda")
            if rows.is_cuda and torch.is_autocast_enabled("cuda")
            else rows.dtype
        )
        rows_d, cells_d, glob_d = rows.to(dt), cells.to(dt), glob.to(dt)
        g_ctx = self.glob_proj(glob_d.flatten(1))
        x = g_ctx[:, None] + self.query.weight[:n][None].to(dt)
        r_mem = (
            rows_d
            + self.row_xy(anc_txy.to(dt)) * rmask[..., None].to(dt)
            + self.kind_tag.weight[0].to(dt)
        )
        c_mem = (
            cells_d + self.cell_xy(cell_txy.to(dt)) + self.kind_tag.weight[1].to(dt)
        )
        g_mem = glob_d + self.kind_tag.weight[2].to(dt)
        mem_parts = [r_mem, c_mem, g_mem]
        mask_parts = [
            rmask,
            torch.ones(
                K,
                cells.shape[1] + glob.shape[1],
                dtype=torch.bool,
                device=rows.device,
            ),
        ]
        a_mem = None
        if acts is not None and acts.shape[1] > 0:
            acts_d = acts.to(dt)
            a_mem = (
                acts_d
                + self.act_xy(act_txy.to(dt)) * aplaced[..., None].to(dt)
                + self.kind_tag.weight[3].to(dt)
            )
            mem_parts.append(a_mem)
            mask_parts.append(amask)
        e_mem = None
        if (
            effs is not None
            and emask is not None
            and eff_txy is not None
            and eff_ty is not None
            and eff_age is not None
            and effs.shape[1] > 0
        ):
            e_mem = self._encode_eff_mem(
                effs, emask, eff_txy, eff_ty, eff_age, dt, add_tag=True
            )
            mem_parts.append(e_mem)
            mask_parts.append(emask)
        mem = torch.cat(mem_parts, dim=1)
        mmask = torch.cat(mask_parts, dim=1)
        for lyr in self.layers:
            sq, sk, sv = (
                lyr["sqkv"](lyr["sn"](x))
                .view(K, n, 3, self.H, self.dh)
                .permute(2, 0, 3, 1, 4)
            )
            sa = F.scaled_dot_product_attention(sq, sk, sv, is_causal=True)
            x = x + lyr["so"](sa.transpose(1, 2).reshape(K, n, self.H * self.dh))
            cq = lyr["cq"](lyr["cn"](x)).view(K, n, self.H, self.dh).transpose(1, 2)
            ck, cv = (
                lyr["ckv"](lyr["kn"](mem))
                .view(K, -1, 2, self.H, self.dh)
                .permute(2, 0, 3, 1, 4)
            )
            ca = F.scaled_dot_product_attention(
                cq, ck, cv, attn_mask=mmask[:, None, None]
            )
            x = x + lyr["co"](ca.transpose(1, 2).reshape(K, n, self.H * self.dh))
            x = x + lyr["ff"](lyr["fn"](x))
        x = self.out_norm(x)
        return x, r_mem, c_mem, a_mem, e_mem

    def forward(
        self,
        glob,
        rows,
        rmask,
        cells,
        anc_txy,
        cell_txy,
        acts,
        amask,
        aplaced,
        act_txy,
        n,
        effs=None,
        emask=None,
        eff_txy=None,
        eff_ty=None,
        eff_age=None,
        need_fields=True,
        grad_ckpt=False,
    ):
        K, R, d = rows.shape
        has_eff = (
            effs is not None
            and emask is not None
            and eff_txy is not None
            and eff_ty is not None
            and eff_age is not None
            and effs.shape[1] > 0
        )
        if not need_fields:
            from torch.utils.checkpoint import checkpoint

            use_ckpt = (
                grad_ckpt
                and torch.is_grad_enabled()
                and any(p.requires_grad for p in self.parameters())
            )
            chunk = 256
            reals = []
            for s in range(0, K, chunk):
                e = min(K, s + chunk)
                if has_eff:
                    fn = (
                        lambda g_, r_, rm_, c_, at_, ct_, ac_, am_, ap_, axy_, ef_, em_, etxy_, ety_, eag_: self.real(
                            self._run_queries(
                                g_,
                                r_,
                                rm_,
                                c_,
                                at_,
                                ct_,
                                ac_,
                                am_,
                                ap_,
                                axy_,
                                n,
                                effs=ef_,
                                emask=em_,
                                eff_txy=etxy_,
                                eff_ty=ety_,
                                eff_age=eag_,
                            )[0]
                        ).squeeze(
                            -1
                        )
                    )
                    args = (
                        glob[s:e],
                        rows[s:e],
                        rmask[s:e],
                        cells[s:e],
                        anc_txy[s:e],
                        cell_txy[s:e],
                        acts[s:e],
                        amask[s:e],
                        aplaced[s:e],
                        act_txy[s:e],
                        effs[s:e],
                        emask[s:e],
                        eff_txy[s:e],
                        eff_ty[s:e],
                        eff_age[s:e],
                    )
                else:
                    fn = lambda g_, r_, rm_, c_, at_, ct_, ac_, am_, ap_, axy_: self.real(
                        self._run_queries(
                            g_, r_, rm_, c_, at_, ct_, ac_, am_, ap_, axy_, n
                        )[0]
                    ).squeeze(-1)
                    args = (
                        glob[s:e],
                        rows[s:e],
                        rmask[s:e],
                        cells[s:e],
                        anc_txy[s:e],
                        cell_txy[s:e],
                        acts[s:e],
                        amask[s:e],
                        aplaced[s:e],
                        act_txy[s:e],
                    )
                if use_ckpt:
                    reals.append(checkpoint(fn, *args, use_reentrant=False))
                else:
                    reals.append(fn(*args))
            return None, torch.cat(reals, dim=0), None, None, None, None, None, None, None

        x, r_mem, c_mem, a_mem, e_mem = self._run_queries(
            glob,
            rows,
            rmask,
            cells,
            anc_txy,
            cell_txy,
            acts,
            amask,
            aplaced,
            act_txy,
            n,
            effs=effs,
            emask=emask,
            eff_txy=eff_txy,
            eff_ty=eff_ty,
            eff_age=eff_age,
        )
        real = self.real(x).squeeze(-1)
        eff_type = self.eff_type(x)
        owner = self.owner(x)
        views = self.views(x)
        n_dots = self.n_dots(x)
        age = self.age(x)
        ray_dxy = self.ray_dxy(x).float()

        logit_parts = []
        txy_parts = []
        if R > 0:
            r_logits = (
                torch.einsum("kqd,krd->kqr", self.row_q(x), self.row_k(r_mem)).float()
                / math.sqrt(d)
            ).masked_fill(~rmask[:, None], -1e4)
            logit_parts.append(r_logits)
            txy_parts.append(anc_txy.float())
        c_logits = (
            torch.einsum("kqd,ksd->kqs", self.cell_q(x), self.cell_k(c_mem)).float()
            / math.sqrt(d)
        )
        logit_parts.append(c_logits)
        txy_parts.append(cell_txy.float())
        if a_mem is not None and a_mem.shape[1] > 0:
            a_logits = (
                torch.einsum("kqd,kad->kqa", self.act_q(x), self.act_k(a_mem)).float()
                / math.sqrt(d)
            ).masked_fill(~aplaced[:, None], -1e4)
            logit_parts.append(a_logits)
            txy_parts.append(act_txy.float())
        if e_mem is not None and e_mem.shape[1] > 0:
            e_logits = (
                torch.einsum("kqd,ked->kqe", self.eff_q(x), self.eff_k(e_mem)).float()
                / math.sqrt(d)
                + self.eff_ptr_bias.float()
            ).masked_fill(~emask[:, None], -1e4)
            logit_parts.append(e_logits)
            txy_parts.append(eff_txy.float())
        w_rca = F.softmax(torch.cat(logit_parts, dim=-1), dim=-1)
        rca_txy = torch.cat(txy_parts, dim=1)
        start_pos = torch.einsum("kqm,kmp->kqp", w_rca, rca_txy) + self.pos_rel(x).float()
        return x, real, start_pos, ray_dxy, eff_type, owner, views, n_dots, age


class RecipientHead(nn.Module):
    """Scores each live owned unit row at anchor frame f_anc as a recipient of action
    (ub, uf, sl), combining content matching, placed RoPE2D spatial proximity, and
    action-gated routing over f_anc's control/selection groups plus causal intra-tubelet
    box/click selections."""

    def __init__(self, d):
        super().__init__()
        dh_s = max(4, ((d // 2) // 4) * 4)
        dh_c = d - dh_s
        self.dh_c, self.dh_s = dh_c, dh_s
        self.q_mix = mlp(d, d)
        self.k_mix = mlp(d, d)
        self.qc = nn.Linear(d, dh_c, bias=False)
        self.kc = nn.Linear(d, dh_c, bias=False)
        self.qs = nn.Linear(d, dh_s, bias=False)
        self.ks = nn.Linear(d, dh_s, bias=False)
        self.rope = RoPE2D(dh_s, lo=ROPE_SHORTEST, hi=256.0)
        self.sel_gate = mlp(d, C_GROUPS + 2)
        self.row_bias = nn.Linear(d, 1)
        self.cmd_bias = nn.Linear(d, 1)
        nn.init.zeros_(self.cmd_bias.weight)
        nn.init.constant_(self.cmd_bias.bias, -2.0)

    def forward(self, q_tok, k_tok, xy_a, placed, xy_r, grp_feat):
        qm = self.q_mix(q_tok)
        km = self.k_mix(k_tok)
        qc, kc = self.qc(qm), self.kc(km)
        score_c = torch.einsum("ud,urd->ur", qc, kc).float() / math.sqrt(self.dh_c)
        qs = self.qs(qm)[:, None, None]
        ks = self.ks(km)[:, None]
        qs = self.rope.rotate(qs, self.rope.angles(xy_a[:, None])).squeeze(2).squeeze(1)
        ks = self.rope.rotate(ks, self.rope.angles(xy_r)).squeeze(1)
        score_s = (
            torch.einsum("ud,urd->ur", qs, ks).float() / math.sqrt(self.dh_s)
        ) * placed[:, None]
        gate = self.sel_gate(qm).float()
        score_g = torch.einsum("uc,urc->ur", gate, grp_feat.float())
        return (
            score_c
            + score_s
            + score_g
            + self.row_bias(km).squeeze(-1).float()
            + self.cmd_bias(qm).float()
        )
