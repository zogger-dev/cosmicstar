"""The L0 physics model: a single causal pass over a sequence predicts the next engine
state at every frame, as a language model does.

    per frame        ResNet over the live + static map at 16 x 16 anchors, entities (with local 5x5 1-cell terrain), globals, effects and action tokens
    each block       spatial, feed-forward, temporal, feed-forward, then this frame's actions
      spatial        every token reads all tokens of the frame in one softmax, with directed edge biases on unit pointers (engages, inside, order target)
      temporal       rows, cells and globals, each reading earlier frames' action-updated tokens; each player's actions: causal, with their own weights
      actions        actions read the state (both players' tokens), not each other; the state reads the actions, carried on as a residual delta
    outputs          s + g(s) decoded directly by grounded next-frame heads (units, queues, pointers, births, map fields, player deltas, command recipients)
"""

import collections
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

from .dense import ResBlock
from .direct import RoPE2D
from .encoder import mlp
from .grid_latent import MapDecoder
from .physics_data import (
    C_DYNAMIC,
    C_LIVE,
    C_STATIC,
    NEUTRAL,
    PATCH,
    PROD,
    QUEUE,
    TOP,
)
from .tokenize import C_GROUPS, N_BUFF_IDS, N_EFFECT_IDS, N_RAW_ABILITY_IDS, PC_KEYS

from .entity_encoder import N_FLAGS, PLY_PROBE_IDX, ActionEncoder, EntityEncoder, I

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
ROPE_SHORTEST = 1 / 32  # map units: a cell is one map unit
# game loops: 3 tells gaps of 1, 2 and 3 apart (1 and 2 alias); 672 is 30 s at 22.4
# loops a second
TIME_SHORTEST, TIME_LONGEST = (
    3.0,
    672.0,
)


def cos_sin(ang):
    """Rotation angles [..., dh / 2] -> cosines and sines [..., dh / 2, 2], once per
    pass and shared by every layer that rotates by them, in the dtype the layers compute
    in (bfloat16 under autocast): computed per call they cost about 5% of an update.
    """
    dt = (
        torch.get_autocast_dtype("cuda")
        if torch.is_autocast_enabled("cuda")
        else ang.dtype
    )
    return torch.stack([ang.cos(), ang.sin()], -1).to(dt)


def rotate(x, cs, back=False):
    """x [B, H, T, dh], cs [B, T, dh / 2, 2] from cos_sin. back: the inverse
    rotation."""
    c, s = cs[..., 0][:, None].to(x.dtype), cs[..., 1][:, None].to(x.dtype)
    s = -s if back else s
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class RoPE1D(nn.Module):
    """Rotary positions over time, in game loops: each head's pairs rotate at
    wavelengths from lo to hi (geometric), so a query and a key score by how many loops
    apart they are, whatever the step between frames."""

    def __init__(self, dh, lo=TIME_SHORTEST, hi=TIME_LONGEST):
        super().__init__()
        self.register_buffer(
            "omega",
            2 * math.pi / torch.logspace(math.log10(lo), math.log10(hi), dh // 2),
            persistent=False,
        )

    def angles(self, loops):
        """loops [..., F] -> [..., F, dh / 2] rotation angles."""
        return loops[..., None].float() * self.omega


def attend_pairs(q, k, v, mask, n_u=0, edges=None, bias=None):
    """One softmax per query over the keys mask [B, K] allows. Active frames are
    pre-partitioned into [:n_u] unbiased and [n_u:] biased subsets with precomputed
    sparse pointer edge tuples, so every active frame runs SDPA once with zero syncs."""
    gqa = q.shape[1] != k.shape[1]
    if edges is None:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[:, None, None], enable_gqa=gqa
        )
    q_b, k_b, v_b, m_b = q[n_u:], k[n_u:], v[n_u:], mask[n_u:]
    pi, qi, ki, ci = edges
    am = torch.zeros(
        q_b.shape[0], q.shape[2], k.shape[2], q.shape[1], dtype=q.dtype, device=q.device
    )
    am = am.index_put((pi, qi, ki), bias.to(q.dtype)[ci])
    am = am.permute(0, 3, 1, 2).masked_fill(~m_b[:, None, None], -float("inf"))
    r_b = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=am, enable_gqa=gqa)
    if n_u == 0:
        return r_b
    r_u = F.scaled_dot_product_attention(
        q[:n_u], k[:n_u], v[:n_u], attn_mask=mask[:n_u, None, None], enable_gqa=gqa
    )
    return torch.cat([r_u, r_b], dim=0)


def _flex_spatial_raw(q, k, v, mask, eng_tgt, par_tgt, ord_tgt, bd):
    """FlexAttention with detached [6, H] edge biases -> (out, lse), zero atomicAdds."""
    b0, b1, b2, b3, b4, b5 = bd.unbind(0)

    def score_mod(score, b, hh, qi, ki):
        s = torch.where(mask[b, ki], score, -float("inf"))
        add = (
            torch.where(eng_tgt[b, qi] == ki, b0[hh], 0.0)
            + torch.where(eng_tgt[b, ki] == qi, b1[hh], 0.0)
            + torch.where(par_tgt[b, qi] == ki, b2[hh], 0.0)
            + torch.where(par_tgt[b, ki] == qi, b3[hh], 0.0)
            + torch.where(ord_tgt[b, qi] == ki, b4[hh], 0.0)
            + torch.where(ord_tgt[b, ki] == qi, b5[hh], 0.0)
        )
        return s + add.to(s.dtype)

    return flex_attention(
        q,
        k,
        v,
        score_mod=score_mod,
        return_lse=True,
        enable_gqa=(q.shape[1] != k.shape[1]),
    )


class _EdgeBiasGrad(torch.autograd.Function):
    """Exact analytical gradient dL/dbias on the O(R) active pointer edges from LSE,
    avoiding O(B * T^2) global atomicAdd contention inside FlexAttention backward."""

    @staticmethod
    def forward(ctx, out, lse, q, k, v, mask, eng, par, ord0, bias):
        ctx.save_for_backward(
            out.detach(), lse, q, k, v, mask, eng, par, ord0, bias.detach()
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        out, lse, q, k, v, mask, eng, par, ord0, bd = ctx.saved_tensors
        D = (grad_out.float() * out.float()).sum(-1)
        scale = 1.0 / math.sqrt(q.shape[-1])
        rep = q.shape[1] // k.shape[1]
        gb = torch.zeros_like(bd, dtype=torch.float32)
        ptrs = ((0, eng), (2, par), (4, ord0))
        for k_idx, ptr in ptrs:
            bi0, idx_a0 = torch.nonzero(ptr >= 0, as_tuple=True)
            if not len(bi0):
                continue
            idx_b0 = ptr[bi0, idx_a0].long()
            for rev, slot in ((False, k_idx), (True, k_idx + 1)):
                qi, ki = (idx_b0, idx_a0) if rev else (idx_a0, idx_b0)
                ok = mask[bi0, ki]
                bi, q_i, k_i = bi0[ok], qi[ok], ki[ok]
                if not len(bi):
                    continue
                ks = k[bi, :, k_i].float()
                vs = v[bi, :, k_i].float()
                if rep > 1:
                    ks = ks.repeat_interleave(rep, 1)
                    vs = vs.repeat_interleave(rep, 1)
                s = (q[bi, :, q_i].float() * ks).sum(-1) * scale
                for kk, pp in ptrs:
                    s = s + (pp[bi, q_i] == k_i)[:, None].float() * bd[kk].float()[None]
                    s = (
                        s
                        + (pp[bi, k_i] == q_i)[:, None].float()
                        * bd[kk + 1].float()[None]
                    )
                p = torch.exp(s - lse[bi, :, q_i].float())
                dov = (grad_out[bi, :, q_i].float() * vs).sum(-1)
                gb[slot] = (p * (dov - D[bi, :, q_i])).sum(0)
        return (
            grad_out,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            gb.to(bd.dtype),
        )


class FlexSpatial(nn.Module):
    """Compiled inner FlexAttention module (keeps _EdgeBiasGrad outside Dynamo)."""

    def __init__(self):
        super().__init__()
        self.is_compiled = False

    def forward(self, q, k, v, mask, eng_tgt, par_tgt, ord_tgt, bd):
        return _flex_spatial_raw(q, k, v, mask, eng_tgt, par_tgt, ord_tgt, bd)


def attend_spatial(q, k, v, mask, eng_tgt, par_tgt, ord_tgt, bias, flex_mod=None):
    """Spatial self-attention with directed unit-pointer edge biases [N_UNIT_EDGES, H]
    evaluated directly from 1D target-token indices [B, T] (-1: none). FlexAttention on
    CUDA (zero [T, T] allocation, analytical LSE edge-bias grad); scatter-added SDPA on
    CPU or uncompiled no-grad."""
    compiled_flex = flex_mod is not None and getattr(flex_mod, "is_compiled", False)
    if not q.is_cuda or (not compiled_flex and not torch.is_grad_enabled()):
        B, H, T, _ = q.shape
        am = torch.zeros(B, H, T, T, dtype=q.dtype, device=q.device)
        sb = bias.to(q.dtype)
        for k_idx, ptr in ((0, eng_tgt), (2, par_tgt), (4, ord_tgt)):
            has = ptr >= 0
            idx_f = ptr.clamp(min=0).long()[:, None, :, None].expand(B, H, T, 1)
            w_f = has[:, None, :, None].to(q.dtype) * sb[k_idx][None, :, None, None]
            am = am.scatter_add(3, idx_f, w_f)
            idx_r = ptr.clamp(min=0).long()[:, None, None, :].expand(B, H, 1, T)
            w_r = has[:, None, None, :].to(q.dtype) * sb[k_idx + 1][None, :, None, None]
            am = am.scatter_add(2, idx_r, w_r)
        am = am.masked_fill(~mask[:, None, None], -float("inf"))
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=am, enable_gqa=(q.shape[1] != k.shape[1])
        )

    fn = flex_mod if flex_mod is not None else _flex_spatial_raw
    out, lse = fn(q, k, v, mask, eng_tgt, par_tgt, ord_tgt, bias.detach())
    if torch.is_grad_enabled() and bias.requires_grad:
        out = _EdgeBiasGrad.apply(
            out,
            lse.detach(),
            q.detach(),
            k.detach(),
            v.detach(),
            mask,
            eng_tgt,
            par_tgt,
            ord_tgt,
            bias,
        )
    return out


class Attend(nn.Module):
    """Spatial softmax over allowed keys with unit-pointer edge biases."""

    def __init__(self):
        super().__init__()
        self.flex = FlexSpatial()

    def forward(self, q, k, v, mask, eng_tgt, par_tgt, ord_tgt, bias):
        return attend_spatial(
            q, k, v, mask, eng_tgt, par_tgt, ord_tgt, bias, flex_mod=self.flex
        )


class PairAttend(nn.Module):
    """The same, with the action pointers' bias on a pair's score."""

    def forward(self, q, k, v, mask, n_u, edges, bias):
        return attend_pairs(q, k, v, mask, n_u=n_u, edges=edges, bias=bias)


class Heads(nn.Module):
    """Projections to H query heads and KV key/value heads of explicit width dh, each KV
    head shared by H // KV query heads. Queries and keys are normalized, with a learned
    scale."""

    def __init__(self, d, heads, kv_heads, dh=32):
        super().__init__()
        self.H, self.KV, self.dh = heads, kv_heads, dh
        self.q, self.k, self.v = (
            nn.Linear(d, heads * dh, bias=False),
            nn.Linear(d, kv_heads * dh, bias=False),
            nn.Linear(d, kv_heads * dh, bias=False),
        )
        self.o = nn.Linear(heads * dh, d, bias=False)
        self.scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def _kv(self, x, layer, cs=None):
        """Rotated by cs, if given, at the KV key-value heads (native GQA in attention)."""
        B, T = x.shape[:2]
        z = layer(x).view(B, T, self.KV, self.dh).transpose(1, 2)
        return rotate(z, cs) if cs is not None else z

    def qkv(self, x, cs=None):
        """Fused [Q, K, V] projection and single RoPE pass when Q, K, V share input x."""
        B, T = x.shape[:2]
        w = torch.cat([self.q.weight, self.k.weight, self.v.weight], 0)
        z = F.linear(x, w).view(B, T, self.H + 2 * self.KV, self.dh).transpose(1, 2)
        if cs is not None:
            z = rotate(z, cs)
        q, k, v = z.split([self.H, self.KV, self.KV], 1)
        q = F.normalize(q, dim=-1) * self.scale.exp().to(q.dtype) * math.sqrt(self.dh)
        return q, F.normalize(k, dim=-1), v

    def kv(self, x, cs=None):
        """Fused [K, V] projection and single RoPE pass when K, V share clean input x."""
        B, T = x.shape[:2]
        w = torch.cat([self.k.weight, self.v.weight], 0)
        z = F.linear(x, w).view(B, T, 2 * self.KV, self.dh).transpose(1, 2)
        if cs is not None:
            z = rotate(z, cs)
        k, v = z.split([self.KV, self.KV], 1)
        return F.normalize(k, dim=-1), v

    def query(self, x, cs=None):
        B, T = x.shape[:2]
        q = self.q(x).view(B, T, self.H, self.dh).transpose(1, 2)
        q = rotate(q, cs) if cs is not None else q
        return (
            F.normalize(q, dim=-1) * self.scale.exp().to(q.dtype) * math.sqrt(self.dh)
        )

    def key(self, x, cs=None):
        return F.normalize(self._kv(x, self.k, cs), dim=-1)

    def value(self, x, cs=None):
        """cs: each value rotated by its token's place, so that with out(a, the query's
        cs) a token reads its keys' contents turned by their offset from it."""
        return self._kv(x, self.v, cs)

    def out(self, a, cs=None):
        """[B, H, T, dh] -> [B, T, d]; cs: rotated back by the query's place."""
        a = rotate(a, cs, back=True) if cs is not None else a
        return self.o(a.transpose(1, 2).flatten(2))


class FeedForward(nn.Module):
    """Pre-norm SwiGLU with fused gate+up GEMM, added to its input."""

    def __init__(self, d, ff):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.w1, self.w2, self.w3 = (
            nn.Linear(d, ff, bias=False),
            nn.Linear(d, ff, bias=False),
            nn.Linear(ff, d, bias=False),
        )

    def forward(self, x):
        y = self.norm(x)
        u1, u2 = F.linear(y, torch.cat([self.w1.weight, self.w2.weight], 0)).chunk(
            2, -1
        )
        return x + self.w3(F.silu(u1) * u2)


class Block(nn.Module):
    """Spatial attention and a feed-forward, temporal attention and a feed-forward, then
    this step's actions as a change kept apart from the state. The action timeline has
    weights of its own: action tokens and entity tokens are different kinds of sequence.
    """

    def __init__(
        self,
        d,
        heads,
        kv_heads,
        ff,
        dh=32,
        spatial_every=1,
        offset=0,
    ):
        super().__init__()
        # spatial attention runs on every spatial_every-th frame from this block's
        # offset, and on the window's first frame; the other frames keep the pair
        # context their tokens carry forward in time. Blocks take different offsets, so
        # the stack refreshes the configuration more often than any one block pays for
        self.every, self.offset = spatial_every, offset % max(1, spatial_every)
        self.ns = nn.LayerNorm(d)
        self.space = Heads(d, heads, kv_heads, dh=dh)
        self.ff_space = FeedForward(d, ff)
        self.ent = Attend()
        # unit-to-unit directed pointer edge biases in spatial attention
        self.space_bias = nn.Parameter(torch.zeros(N_UNIT_EDGES, heads))
        self.nt, self.na, self.nat = (
            nn.LayerNorm(d),
            nn.LayerNorm(d),
            nn.LayerNorm(d),
        )
        self.time, self.act, self.act_time = (
            Heads(d, heads, kv_heads, dh=dh) for _ in range(3)
        )
        self.ff_time, self.ff_act = FeedForward(d, ff), FeedForward(d, ff)
        self.pair = PairAttend()
        # action-to-entity pair bias by code: none, action -> target, target -> action,
        # selection -> added unit, added unit -> selection
        self.bias = nn.Parameter(torch.zeros(N_CODES, heads))
        nn.init.zeros_(
            self.act.o.weight
        )  # starts as the identity: the updated tokens start as the state

    def _spatial_qkv(self, x, ang):
        return self.space.qkv(self.ns(x), ang)

    def _spatial_post(self, x, a, ang):
        return self.ff_space(x + self.space.out(a, ang))

    def spatial(self, x, c):
        """All tokens of a frame read each other, entities and map cells in one
        softmax, with directed unit-pointer edge biases."""
        q, k, v = self._spatial_qkv(x, c["ang"])
        a = self.ent(
            q,
            k,
            v,
            c["valid"],
            c["eng_tgt"],
            c["par_tgt"],
            c["ord_tgt"],
            self.space_bias.to(q.dtype),
        )
        return x + self.space.out(a, c["ang"])

    def _spatial_and_ff(self, x, c):
        q, k, v = self._spatial_qkv(x, c["ang"])
        a = self.ent(
            q,
            k,
            v,
            c["valid"],
            c["eng_tgt"],
            c["par_tgt"],
            c["ord_tgt"],
            self.space_bias.to(q.dtype),
        )
        return self._spatial_post(x, a, c["ang"])

    def temporal(self, x, key, mask, ang, heads, norm, place=None):
        """x, key [N, T, d], mask [N, T], ang [N, T, dh / 2, 2] the cosines and sines of
        each frame's rotation by its game loop: each token reads the keys at or before
        its frame. place: (the keys' places, the queries' places) [N, T, dh / 2, 2], so
        a token reads each earlier frame turned by how far it has moved since.
        A token with nothing to read keeps its own frame's key: an empty softmax is NaN.
        """
        N, T, d = x.shape
        y = norm(x)
        if key is x and place is None:
            qk, kk, vk = heads.qkv(y, ang)
        else:
            ky = norm(key)
            qk = heads.query(y, ang)
            kk = heads.key(ky, ang)
            vk = heads.value(ky, place[0] if place else None)
        tri = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        allow = (tri[None] & mask[:, None]) | torch.eye(
            T, dtype=torch.bool, device=x.device
        )[None]
        a = F.scaled_dot_product_attention(
            qk,
            kk,
            vk,
            attn_mask=allow[:, None],
            enable_gqa=(qk.shape[1] != kk.shape[1]),
        )
        return torch.where(
            mask[..., None], x + heads.out(a, place[1] if place else None), x
        )

    def _temporal_step(self, x, dl, mask, t_ang, p_k, p_t, B, F_):
        T, d = x.shape[1], x.shape[-1]
        xr = x.view(B, F_, T, d).transpose(1, 2).reshape(B * T, F_, d)
        up = (x + dl).view(B, F_, T, d).transpose(1, 2).reshape(B * T, F_, d)
        key = torch.cat(
            [xr[:, :1], torch.where(mask[:, :-1, None], up[:, :-1], xr[:, 1:])], 1
        )
        y = (
            self.temporal(xr, key, mask, t_ang, self.time, self.nt, (p_k, p_t))
            .view(B, T, F_, d)
            .transpose(1, 2)
            .reshape(B * F_, T, d)
            .to(x.dtype)
        )
        return self.ff_time(y)

    def _act_qkv(self, x_fa, dl_fa, a_fa, ang_fa, ang_a_fa):
        # State -> Action queries x_fa + dl_fa so multi-step intra-tubelet chains compose,
        # while Action -> State reads clean x_fa so actions never leak backward or across
        # players inside the tubelet via dl_fa.
        q = self.act.query(self.na(x_fa + dl_fa), ang_fa)
        k, v = self.act.kv(self.na(x_fa), ang_fa)
        qa, ka, va = self.act.qkv(self.na(a_fa), ang_a_fa)
        bias = (self.bias * (torch.arange(N_CODES, device=x_fa.device) > 0)[:, None]).to(
            q.dtype
        )  # code 0: no pair, no bias
        return q, k, v, qa, ka, va, bias

    def actions(self, x, dl, a, c):
        """Frames with an action: actions read the state, not each other (a player's own
        earlier actions reach it through its causal timeline, and the two players'
        actions of one frame are simultaneous), then the state reads the actions; x +
        the returned dl is the updated state. dl [BF, T, d], a [BF, A, d]."""
        fa = c["fa"]
        ang_fa, ang_a_fa = c["ang"][fa], c["ang_a"][fa]
        q, k, v, qa, ka, va, bias = self._act_qkv(
            x[fa], dl[fa], a[fa], ang_fa, ang_a_fa
        )
        am = c["act_mask"][fa]
        n_u = c["fa_nu"]
        ra = self.pair(qa, k, v, c["valid"][fa], n_u, c["edges_ae"], bias)
        r = self.pair(q, ka, va, am, n_u, c["edges_ea"], bias)
        a, dl = self._act_post(a, dl, fa, ra, r, am, ang_a_fa, ang_fa)
        return dl, a

    def _act_post(self, a, dl, fa, ra, r, am, ang_a_fa, ang_fa):
        a_fa, dl_fa = a[fa], dl[fa]
        a_up = torch.where(
            am[..., None], a_fa + self.act.out(ra, ang_a_fa), a_fa
        ).to(a.dtype)
        dl_up = (dl_fa + self.act.out(r, ang_fa)).to(dl.dtype)
        return a.index_put((fa,), a_up), dl.index_put((fa,), dl_up)

    def _pack_act_chunk(self, a, B, F_, F_act, k, K):
        d = a.shape[-1]
        a_4d = a.view(B, F_act, 2, K, d)
        a_pad = F.pad(a_4d[:, k - 1 :], (0, 0, 0, 0, 0, 0, 0, k - 1))
        return (
            a_pad.view(B, F_, k, 2, K, d)
            .transpose(2, 3)
            .reshape(B * F_, 2 * k * K, d)
        )

    def _unpack_act_chunk(self, a, a_chunk, B, F_, F_act, k, K):
        d = a.shape[-1]
        a_4d = a.view(B, F_act, 2, K, d)
        a_up = (
            a_chunk.view(B, F_, 2, k, K, d)
            .transpose(2, 3)
            .reshape(B, F_ * k, 2, K, d)[:, : F_act - (k - 1)]
        )
        return torch.cat([a_4d[:, : k - 1], a_up], dim=1).reshape(B * F_act, 2 * K, d)

    def frames(self, F_, device):
        """The frames this block's spatial attention runs on."""
        f = torch.arange(F_, device=device)
        return torch.nonzero((f % self.every == self.offset) | (f == 0), as_tuple=True)[
            0
        ]

    def spatial_frames(self, x, c):
        """Spatial attention and its feed-forward on this block's frames only; the other
        frames pass through unchanged. The pair structure (who is in range of whom)
        changes slowly, a unit's own clocks every frame, so the frames between read
        the last spatial frame's pair context through their own timelines."""
        if self.every <= 1:
            return self._spatial_and_ff(x, c)
        B, F_ = c["B"], c["F"]
        fs = self.frames(F_, x.device)
        idx = (torch.arange(B, device=x.device)[:, None] * F_ + fs[None]).reshape(-1)
        sub = {
            k: c[k][idx]
            for k in ("valid", "ang", "eng_tgt", "par_tgt", "ord_tgt")
        }
        y = self._spatial_and_ff(x[idx], sub)
        return x.index_copy(0, idx, y.to(x.dtype))

    def forward(self, x, dl, a, c, last):
        B, F_ = c["B"], c["F"]
        x = self.spatial_frames(x, c)
        # timelines: earlier steps' tokens as the last block's actions left them.
        # Effects have none: masked, they come out as they went in
        p_k, p_t = c["t_place"]
        x = self._temporal_step(x, dl, c["t_mask"], c["t_ang"], p_k, p_t, B, F_)
        if last:
            return x, dl, a
        k = c.get("tubelet", 1)
        if k <= 1:
            a = self._act_timeline(a, c)
            dl, a = self.actions(x, dl, a, c)
            return x, dl, a
        F_act, K = c["F_act"], c["K"]
        ac = c.get("act_compact")
        if K > 0 and ac is not None:
            a = self._act_timeline_body(a, ac[0], ac[1], ac[2], B, F_act)
        a_chunk = self._pack_act_chunk(a, B, F_, F_act, k, K)
        dl, a_chunk = self.actions(x, dl, a_chunk, c)
        a = self._unpack_act_chunk(a, a_chunk, B, F_, F_act, k, K)
        return x, dl, a

    def _act_timeline_body(self, a, idx, m_c, ang_c, B, F_):
        d = a.shape[-1]
        K = a.shape[1] // 2
        seq = a.view(B, F_, 2, K, d).transpose(1, 2).reshape(B * 2, F_ * K, d)
        idx_e = idx[..., None].expand(-1, -1, d)
        sub = seq.gather(1, idx_e)
        y_sub = torch.where(
            m_c[..., None],
            self.ff_act(
                self.temporal(sub, sub, m_c, ang_c, self.act_time, self.nat)
            ),
            sub,
        )
        out = seq.scatter(1, idx_e, y_sub.to(seq.dtype))
        return (
            out.view(B, 2, F_, K, d).transpose(1, 2).reshape(B * F_, 2 * K, d).to(a.dtype)
        )

    def _act_timeline(self, a, c):
        """Each player's action tokens over the window, causal, in time order: its own
        slots of each frame, frame after frame, so a timeline is a reshape. a [BF, 2K,
        d]: player 0's K slots, then player 1's."""
        ac = c.get("act_compact")
        if a.shape[1] == 0 or ac is None:
            return a
        return self._act_timeline_body(a, ac[0], ac[1], ac[2], c["B"], c["F"])


class Tower(nn.Module):
    """A plain ResNet over the live + static map at its own size: 3 stride-2 stages to an
    eighth of the resolution."""

    def __init__(self, c_in, d, blocks=2):
        super().__init__()
        c = [max(32, d // 2), d, d]
        self.down = nn.ModuleList(
            nn.Conv2d(a, b, 3, stride=2, padding=1) for a, b in zip([c_in] + c[:-1], c)
        )
        self.res = nn.ModuleList(
            nn.ModuleList(ResBlock(k) for _ in range(blocks)) for k in c
        )

    def forward(self, x):
        one = x.new_ones(1, 1, 1, 1)
        for down, res in zip(self.down, self.res):
            x = down(x)
            for b in res:
                x = b(x, one)
        return x


def _unit_ptr(row_idx, mask, g0, T):
    """[B, F, R] unit target row index -> [BF, T] token index (-1 where absent or invalid)."""
    B, F_, R = mask.shape
    in_range = (row_idx >= 0) & (row_idx < R)
    tgt_ok = mask.gather(2, row_idx.clamp(0, max(0, R - 1)).long()) if R else mask
    ok = mask & in_range & tgt_ok
    full = torch.full((B, F_, T), -1, dtype=torch.int32, device=mask.device)
    full[:, :, g0 : g0 + R] = torch.where(ok, row_idx.to(torch.int32) + g0, -1)
    return full.reshape(B * F_, T)


class PhysicsEncoder(nn.Module):
    """The whole sequence through the terrain tower and spatio-temporal blocks. When
    tubelet > 1, reads persistent state (units, globals, map) once per tubelet at its
    anchor frame (4m + k - 1), packs loop-rate effects over [4m .. 4m + k - 1] and
    transition actions [4m + k - 1 .. 4m + 2k - 2] into a pure d_tube trunk, and emits
    tubelet-rate clean and action-updated state tokens directly at width d_tube."""

    def __init__(
        self,
        vocab,
        avocab,
        d=128,
        blocks=6,
        heads=8,
        kv_heads=4,
        dh=64,
        ff=1024,
        spatial_every=1,
        tubelet=4,
        d_tube=256,
        **_unused,
    ):
        super().__init__()
        self.d, self.dh, self.tubelet = d, dh, tubelet
        self.d_tube = d_tube if tubelet > 1 else d
        self.spatial_every = spatial_every
        self.acts = ActionEncoder(avocab, d)
        self.entities = EntityEncoder(vocab, d, self.acts)
        self.tower = Tower(C_LIVE + C_STATIC, d)
        self.rope = RoPE2D(dh, lo=ROPE_SHORTEST, hi=256.0)
        self.rope_t = RoPE1D(dh)
        if tubelet > 1:
            self.entities.use_eff_view = True
            self.eff_absent_emb = nn.Parameter(torch.zeros(d))
            nn.init.normal_(self.eff_absent_emb, std=0.02)
            self.pers_pack = nn.Linear(d, d_tube)
            self.cell_pack = nn.Linear(d, d_tube)
            self.eff_pack = nn.Linear(tubelet * d + tubelet, d_tube)
            self.act_pack = nn.Linear(d, d_tube)
            self.act_slot = nn.Embedding(tubelet, d_tube)
            b_dim = d_tube
        else:
            b_dim = d
        # the blocks' spatial frames are spread over the stride, so with 6 blocks and
        # a stride of 8 the stack refreshes the pair structure at offsets 0 1 3 4 5 7
        self.blocks = nn.ModuleList(
            Block(
                b_dim,
                heads,
                kv_heads,
                ff,
                dh=dh,
                spatial_every=spatial_every,
                offset=(i * spatial_every + blocks // 2) // blocks,
            )
            for i in range(blocks)
        )
        self.norm_e, self.norm_c, self.norm_g, self.norm_a, self.norm_eff = (
            nn.LayerNorm(b_dim),
            nn.LayerNorm(b_dim),
            nn.LayerNorm(b_dim),
            nn.LayerNorm(b_dim),
            nn.LayerNorm(b_dim),
        )
        self.grad_ckpt = False

    def compile_parts(self):
        """Compile FlexSpatial, terrain Tower, entity/action encoders, and block stages."""
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.accumulated_recompile_limit = 512
        kw = dict(dynamic=True, recompile_limit=64)
        self.tower.compile(**kw)
        self.acts.compile(**kw)
        self.entities.units._dense = torch.compile(self.entities.units._dense, **kw)
        for m in self.modules():
            if type(m).__name__ == "FlexSpatial":
                m.is_compiled = True
                m.compile(**kw)
        blk_names = (
            "_spatial_qkv",
            "_spatial_post",
            "_temporal_step",
            "_act_timeline_body",
            "_act_qkv",
            "_act_post",
        )
        if self.tubelet > 1:
            blk_names = blk_names + ("_pack_act_chunk", "_unpack_act_chunk")
        for blk in self.blocks:
            for name in blk_names:
                setattr(blk, name, torch.compile(getattr(blk, name), **kw))

    @staticmethod
    @torch.no_grad()
    def _pack_act_timeline(a_mask, a_ang):
        """Compact [B * 2, L_act] action timelines to only the active slots per player
        (padded to a multiple of 16), preserving exact chronological order."""
        if not a_mask.numel():
            return None
        counts = a_mask.sum(1)
        n_max = int(counts.max().item())
        if n_max == 0:
            return None
        n_pad = min(a_mask.shape[1], max(16, ((n_max + 15) // 16) * 16))
        idx = (~a_mask).to(torch.uint8).argsort(dim=1, stable=True)[:, :n_pad]
        m_c = torch.arange(n_pad, device=a_mask.device)[None] < counts[:, None]
        ang_c = a_ang.gather(
            1, idx[:, :, None, None].expand(-1, -1, a_ang.shape[2], 2)
        )
        return idx, m_c, ang_c

    def anchors(self, b):
        """[B, S, 2] the grid cells' centers, map units from the center of the map."""
        c = (
            torch.arange(GRID, device=b["mask"].device, dtype=torch.float32) + 0.5
        ) - GRID / 2
        return (
            torch.stack(torch.broadcast_tensors(c[None, :], c[:, None]), -1).view(
                1, GRID * GRID, 2
            )
            * b["side"][:, None, None]
            / GRID
        )

    @staticmethod
    def _slice_state_frames(b, kf, k, M, F_use):
        """Slice persistent state (units, globals, map) at each tubelet's anchor frame
        kf = 4m + (k - 1), remapping sparse per-unit bag indices to [B, M, R]."""
        B, F_, R = b["mask"].shape
        out = {
            key: b[key]
            for key in (
                "owner",
                "static_grid",
                "origin",
                "center",
                "half",
                "side",
                "view",
                "players",
            )
            if key in b
        }
        for key in (
            "mask",
            "alive",
            "inside",
            "parent_row",
            "engaged_row",
            "type",
            "x",
            "y",
            "num",
            "groups",
            "cargo_slot",
            "ply_num",
            "grid",
        ):
            if key in b:
                out[key] = b[key][:, kf]
        bag_cols = dict(
            ord=("ord_abil", "ord_feat", "ord_slot", "ord_in_q"),
            prod=("prod_abil", "prod_feat", "prod_slot", "prod_in_q"),
            buff=("buff_id",),
            rally=("rally_feat",),
            ab=("ab_id",),
            auto=("auto_fam", "auto_but", "auto_on"),
        )
        for bag, cols in bag_cols.items():
            at = b[f"{bag}_at"]
            if R > 0 and len(at):
                ri = at % R
                bf = at // R
                fi = bf % F_
                bi = bf // F_
                keep = (fi < F_use) & ((fi % k) == (k - 1))
                out[f"{bag}_at"] = (bi[keep] * M + (fi[keep] // k)) * R + ri[keep]
                for col in cols:
                    if col in b:
                        out[col] = b[col][keep]
            else:
                out[f"{bag}_at"] = at[:0]
                for col in cols:
                    if col in b:
                        out[col] = b[col][:0]
        return out

    def embed_map(self, b):
        """Map tokens [B * F_m, S, d] over the frames in b['grid'] (M anchor frames when
        sliced per tubelet, or all F frames when k == 1)."""
        B, F_m = b["grid"].shape[:2]
        H, W = b["grid"].shape[2:4]
        dt = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else self.tower.down[0].weight.dtype
        )
        st = b["static_grid"].to(dt)
        st = torch.cat([st[..., :2] / 255, st[..., 2:]], -1)
        st_chw = (
            st.permute(0, 3, 1, 2)[:, None]
            .expand(B, F_m, C_STATIC, H, W)
            .reshape(B * F_m, C_STATIC, H, W)
        )
        g_raw = b["grid"]
        if g_raw.ndim == 4:
            shifts = torch.arange(C_LIVE, device=g_raw.device, dtype=torch.int32)
            g_unp = ((g_raw.to(torch.int32)[..., None] >> shifts) & 1).to(dt)
        else:
            g_unp = g_raw.to(dt)
        live = g_unp.permute(0, 1, 4, 2, 3).reshape(B * F_m, C_LIVE, H, W)
        raw = torch.cat([live, st_chw], 1)
        top = self.tower(raw)
        at = (self.anchors(b) + (b["center"] - b["origin"])[:, None]) / torch.tensor(
            [W, H], device=raw.device, dtype=torch.float32
        ) * 2 - 1
        at = at.repeat_interleave(F_m, 0).view(B * F_m, GRID, GRID, 2)
        return (
            F.grid_sample(top, at.to(top.dtype), align_corners=False)
            .flatten(2)
            .transpose(1, 2)
        )

    @staticmethod
    @torch.no_grad()
    def _eff_tubelet_indices(eff_mask, eff_xy, eff_type, eff_owner, eff_view, B, M, k):
        """Group active per-frame effects inside each (b, m) tubelet by exact
        (view, owner, type, x, y, within_frame_occ) with zero tracking heuristics."""
        X = eff_mask.shape[2]
        if X == 0:
            return None
        bm_i, s_i, x_i = eff_mask.view(B * M, k, X).nonzero(as_tuple=True)
        n_act = bm_i.numel()
        if n_act == 0:
            return None
        dev = eff_mask.device
        f_flat = bm_i * k + s_i
        xy = eff_xy.view(B * M * k, X, 2)[f_flat, x_i].float()
        xy = torch.where(xy == 0.0, 0.0, xy)
        xb = xy[:, 0].view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        yb = xy[:, 1].view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        k_xy = (xb << 32) | yb

        ty = eff_type.view(B * M * k, X)[f_flat, x_i].to(torch.int64).clamp(0, 255)
        ow = eff_owner.view(B * M * k, X)[f_flat, x_i].to(torch.int64).clamp(0, 7)
        if eff_view is not None:
            vw = eff_view.view(B * M * k, X)[f_flat, x_i].to(torch.int64).clamp(0, 3)
        else:
            vw = torch.zeros_like(ty)

        # Within-frame multiplicity occ_in_frame so identical same-frame effects stay distinct
        k_hi_f = (((((bm_i << 4) | s_i) << 2) | vw) << 3 | ow) << 8 | ty
        o1 = k_xy.argsort(stable=True)
        o2 = k_hi_f[o1].argsort(stable=True)
        of = o1[o2]
        kh_f, kl_f = k_hi_f[of], k_xy[of]
        same_f = torch.zeros(n_act, dtype=torch.bool, device=dev)
        same_f[1:] = (kh_f[1:] == kh_f[:-1]) & (kl_f[1:] == kl_f[:-1])
        idx_n = torch.arange(n_act, device=dev)
        occ_f = idx_n - torch.where(~same_f, idx_n, 0).cummax(0).values

        # Group across the k sub-frames of each bm tubelet by exact (bm, vw, ow, ty, occ, xy)
        bm_f, s_f = bm_i[of], s_i[of]
        k_hi_t = (
            (((((bm_f << 2) | vw[of]) << 3 | ow[of]) << 8 | ty[of]) << 12)
            | occ_f.clamp(max=4095)
        )
        t0 = s_f.argsort(stable=True)
        t1 = kl_f[t0].argsort(stable=True)
        t2 = k_hi_t[t0[t1]].argsort(stable=True)
        ot_rel = t0[t1[t2]]
        ot = of[ot_rel]

        kh, kl = k_hi_t[ot_rel], kl_f[ot_rel]
        is_new = torch.ones(n_act, dtype=torch.bool, device=dev)
        is_new[1:] = (kh[1:] != kh[:-1]) | (kl[1:] != kl[:-1])
        u_global = is_new.to(torch.int64).cumsum(0) - 1

        bm_s, s_s, f_s, x_s = bm_i[ot], s_i[ot], f_flat[ot], x_i[ot]
        bm_u = bm_s[is_new]
        counts = torch.bincount(bm_u, minlength=B * M)
        x_tube = int(counts.max().item())

        u_idx = torch.arange(bm_u.numel(), device=dev)
        bm_new = torch.ones_like(bm_u, dtype=torch.bool)
        bm_new[1:] = bm_u[1:] != bm_u[:-1]
        u_slot_unique = u_idx - torch.where(bm_new, u_idx, 0).cummax(0).values
        u_slot = u_slot_unique[u_global]

        pos_out = torch.zeros(B * M, x_tube, 2, dtype=torch.float32, device=dev)
        pos_out[bm_u, u_slot_unique] = xy[ot][is_new]
        return x_tube, bm_s, u_slot, s_s, f_s, x_s, pos_out.view(B, M, x_tube, 2)

    def _pack_eff_tubelets(self, eff, b, F_use, M):
        """Collapse exact-(view, owner, type, x, y) effects across each k-frame tubelet
        into [B, M, X_tube, d_tube], passing moving effects through as 1-slot tubelets."""
        B, _, X, d = eff.shape
        k, dt_dim = self.tubelet, self.d_tube
        idx = self._eff_tubelet_indices(
            b["eff_mask"][:, :F_use],
            b["eff_xy"][:, :F_use],
            b["eff_type"][:, :F_use],
            b["eff_owner"][:, :F_use],
            b["eff_view"][:, :F_use] if "eff_view" in b else None,
            B,
            M,
            k,
        )
        if idx is None:
            z_e = eff.new_zeros(B, M, 0, dt_dim)
            z_v = torch.zeros(B, M, 0, dtype=torch.bool, device=eff.device)
            z_p = torch.zeros(B, M, 0, 2, dtype=torch.float32, device=eff.device)
            z_s = torch.full((B, M, 0), -1, dtype=torch.long, device=eff.device)
            return z_e, z_v, z_p, z_s
        x_tube, bm_s, u_slot, s_s, f_s, x_s, eff_pos_t = idx
        eff_grid = (
            self.eff_absent_emb.to(eff.dtype)
            .view(1, 1, 1, d)
            .expand(B * M, x_tube, k, d)
            .clone()
        )
        eff_grid[bm_s, u_slot, s_s] = eff.view(B * M * k, X, d)[f_s, x_s]
        eff_pres = torch.zeros(B * M, x_tube, k, dtype=torch.bool, device=eff.device)
        eff_pres[bm_s, u_slot, s_s] = True
        src_slot = torch.full((B * M, x_tube), -1, dtype=torch.long, device=eff.device)
        on_anc = s_s == (k - 1)
        if on_anc.any():
            src_slot[bm_s[on_anc], u_slot[on_anc]] = x_s[on_anc]
        valid_eff = eff_pres.any(-1)
        raw_eff = torch.cat([eff_grid.flatten(2), eff_pres.to(eff.dtype)], dim=-1)
        tube_eff = (self.eff_pack(raw_eff) * valid_eff[..., None].to(eff.dtype)).view(
            B, M, x_tube, dt_dim
        )
        return (
            tube_eff,
            valid_eff.view(B, M, x_tube),
            eff_pos_t,
            src_slot.view(B, M, x_tube),
        )

    def _build_ctx(
        self,
        B,
        F_,
        T,
        g0,
        valid,
        pos,
        t_mask,
        loops,
        eng_row,
        par_row,
        ord_row,
        mask,
        act_mask_2k,
        act_xy,
        act_target,
        act_sel,
    ):
        """Precompute spatial/temporal RoPE cosines+sines, unit pointers and action pairs
        for either frame space (F_, T) or tubelet space (M, T_tube)."""
        K = act_mask_2k.shape[-1]
        A = 2 * K
        t_ang = self.rope_t.angles(loops)
        h2 = t_ang.shape[-1]
        t_cs = cos_sin(t_ang)
        act_mask = act_mask_2k.reshape(B * F_, A)
        fa = torch.nonzero(act_mask.any(1), as_tuple=True)[0]
        ang = cos_sin(self.rope.angles(pos))
        p_t = ang.view(B, F_, T, h2, 2).transpose(1, 2).reshape(B * T, F_, h2, 2)
        p_k = torch.cat(
            [
                p_t[:, :1],
                torch.where(t_mask[:, :-1, None, None], p_t[:, :-1], p_t[:, 1:]),
            ],
            1,
        )
        c = dict(
            B=B,
            F=F_,
            valid=valid,
            ang=ang,
            eng_tgt=_unit_ptr(eng_row, mask, g0, T),
            par_tgt=_unit_ptr(par_row, mask, g0, T),
            ord_tgt=_unit_ptr(ord_row, mask, g0, T),
            t_place=(p_k, p_t),
            ang_a=cos_sin(self.rope.angles(act_xy.reshape(B * F_, A, 2))),
            act_mask=act_mask,
            t_mask=t_mask,
            t_ang=t_cs[:, None].expand(B, T, F_, h2, 2).reshape(B * T, F_, h2, 2),
            a_mask=act_mask_2k.transpose(1, 2).reshape(B * 2, F_ * K),
        )
        if len(fa):
            fa_ord, n_u, edges_ae, edges_ea = pointer_pairs(
                act_target.reshape(B * F_, A),
                fa,
                A,
                T,
                g0,
                act_sel,
            )
            c["fa"] = fa_ord
            c["fa_nu"] = n_u
            c["edges_ae"] = edges_ae
            c["edges_ea"] = edges_ea
            fa = fa_ord
        return c, t_cs, fa

    def forward(self, b, last_action=True):
        """last_action False skips the final action step, when only the clean state is
        wanted: the updated tokens returned are then a stage old."""
        from torch.utils.checkpoint import checkpoint

        B, F_, R = b["mask"].shape
        dev = b["mask"].device
        d = self.d
        k = max(1, self.tubelet)
        M = F_ // k
        F_use = M * k
        S = GRID * GRID
        A, X = b["act_mask"].shape[2], b["eff_mask"].shape[2]
        K = A // 2
        g0 = 3
        ckpt = (
            self.grad_ckpt
            and torch.is_grad_enabled()
            and any(p.requires_grad for p in self.parameters())
        )
        kf = torch.arange(M, device=dev) * k + (k - 1) if k > 1 else None
        b_st = self._slice_state_frames(b, kf, k, M, F_use) if k > 1 else b
        ply = self.entities.globals.player if self.entities.use_eff_view else None
        if ckpt:
            seed = torch.zeros(1, device=dev, requires_grad=True)
            cells = checkpoint(
                lambda _: self.embed_map(b_st), seed, use_reentrant=True
            )
            G, rows = checkpoint(
                lambda _: (self.entities.globals(b_st), self.entities.units(b_st)),
                seed,
                use_reentrant=True,
            )
            eff = self.entities.effects(b, ply)
        else:
            cells = self.embed_map(b_st)
            G, rows = self.entities.globals(b_st), self.entities.units(b_st)
            eff = self.entities.effects(b, ply)
        dt = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else rows.dtype
        )
        G = G.to(dt)
        rows = (rows * b_st["mask"][..., None]).to(dt)
        eff = (eff[:, :F_use] * b["eff_mask"][:, :F_use, :, None]).to(dt)
        cells = cells.to(dt)
        acts = self.acts(b)[:, :F_use].to(dt)
        cen = self.anchors(b)
        rel = torch.where(
            b_st["mask"][..., None],
            torch.stack([b_st["x"], b_st["y"]], -1) - b["center"][:, None, None],
            0.0,
        )
        loops = (
            b["loops"][:, :F_use] - b["loops"][:, :1]
            if "loops" in b
            else torch.arange(F_use, device=dev)[None].expand(B, -1)
        )

        if k <= 1:
            x = torch.cat(
                [
                    G.reshape(B * F_, g0, d),
                    rows.reshape(B * F_, R, d),
                    eff.reshape(B * F_, X, d),
                    cells,
                ],
                1,
            )
            n = g0 + R + X
            T = n + S
            acts_f = acts.reshape(B * F_, A, d)
            valid = torch.cat(
                [
                    torch.ones(B, F_, g0, dtype=torch.bool, device=dev),
                    b["mask"],
                    b["eff_mask"],
                    torch.ones(B, F_, S, dtype=torch.bool, device=dev),
                ],
                2,
            ).reshape(B * F_, T)
            eff_xy_clean = torch.where(b["eff_mask"][..., None], b["eff_xy"], 0.0)
            pos = torch.cat(
                [
                    torch.zeros(B, F_, g0, 2, device=dev),
                    rel,
                    eff_xy_clean,
                    cen[:, None].expand(-1, F_, -1, -1),
                ],
                2,
            ).reshape(B * F_, T, 2)
            t_mask = (
                torch.cat(
                    [
                        torch.ones(B, F_, g0, dtype=torch.bool, device=dev),
                        b["mask"],
                        torch.zeros_like(b["eff_mask"]),
                        torch.ones(B, F_, S, dtype=torch.bool, device=dev),
                    ],
                    2,
                )
                .transpose(1, 2)
                .reshape(B * T, F_)
            )
            c, t_cs, fa = self._build_ctx(
                B,
                F_,
                T,
                g0,
                valid,
                pos,
                t_mask,
                loops,
                b["engaged_row"],
                b["parent_row"],
                b["ord_row"][..., 0],
                b["mask"],
                b["act_mask"].view(B, F_, 2, K),
                b["act_xy"],
                b["act_target"],
                b.get("act_sel"),
            )
            h2 = t_cs.shape[-2]
            c["a_ang"] = (
                t_cs.repeat_interleave(K, 1)[:, None]
                .expand(B, 2, F_ * K, h2, 2)
                .reshape(B * 2, F_ * K, h2, 2)
            )
            c["act_compact"] = self._pack_act_timeline(c["a_mask"], c["a_ang"])
            dl = torch.zeros_like(x)
            for i, blk in enumerate(self.blocks):
                last = (not last_action and i == len(self.blocks) - 1) or not len(fa)
                if ckpt:
                    x, dl, acts_f = checkpoint(
                        lambda x_, dl_, a_, _b=blk, _l=last: _b(x_, dl_, a_, c, _l),
                        x,
                        dl,
                        acts_f,
                        use_reentrant=True,
                    )
                else:
                    x, dl, acts_f = blk(x, dl, acts_f, c, last)
            out = lambda z: dict(
                ent=self.norm_e(z[:, g0 : g0 + R]).view(B, F_, R, d),
                eff=self.norm_eff(z[:, g0 + R : n]).view(B, F_, X, d),
                cells=self.norm_c(z[:, n:]).view(B, F_, S, d),
                glob=self.norm_g(z[:, :g0]).view(B, F_, g0, d),
            )
            upd = out(x + dl)
            upd["act"] = self.norm_a(acts_f.view(B, F_, A, d))
            upd["eff_src_slot"] = torch.where(
                b["eff_mask"],
                torch.arange(X, device=dev, dtype=torch.long)[None, None, :],
                -1,
            )
            return out(x), upd

        # Pure k-frame tubelet trunk: persistent state at kf = 4m + (k - 1),
        # loop-rate effects over [4m .. 4m + k - 1], and transition actions [4m + k - 1 .. 4m + 2k - 2]
        dt_dim = self.d_tube
        m_r = b_st["mask"]
        valid_p = torch.cat(
            [torch.ones(B, M, g0, dtype=torch.bool, device=dev), m_r],
            2,
        )
        pos_p = torch.cat(
            [torch.zeros(B, M, g0, 2, device=dev), rel],
            2,
        )
        tube_p = self.pers_pack(torch.cat([G, rows], dim=2)) * valid_p[..., None].to(dt)
        valid_r = m_r
        tube_cells = self.cell_pack(cells.view(B, M, S, d))
        valid_c = torch.ones(B, M, S, dtype=torch.bool, device=dev)
        pos_c = cen[:, None].expand(B, M, S, 2)

        tube_eff, valid_eff, eff_pos_t, eff_src_slot = self._pack_eff_tubelets(
            eff, b, F_use, M
        )
        X_tube = tube_eff.shape[2]
        if X_tube == 0:
            tube_p = tube_p + (eff.sum() + self.eff_absent_emb.sum()) * 0.0

        n_t = g0 + R + X_tube
        T_t = n_t + S
        x_tube = torch.cat(
            [tube_p, tube_eff.to(tube_p.dtype), tube_cells.to(tube_p.dtype)],
            2,
        ).reshape(B * M, T_t, dt_dim)
        valid_t = torch.cat(
            [
                valid_p,
                valid_eff,
                valid_c,
            ],
            2,
        ).reshape(B * M, T_t)
        pos_t = torch.cat(
            [pos_p, eff_pos_t, pos_c], 2
        ).reshape(B * M, T_t, 2)
        t_mask_t = (
            torch.cat(
                [
                    valid_p,
                    torch.zeros(B, M, X_tube, dtype=torch.bool, device=dev),
                    valid_c,
                ],
                2,
            )
            .transpose(1, 2)
            .reshape(B * T_t, M)
        )

        acts_4d = acts.view(B, F_use, 2, K, d)
        s_idx = (torch.arange(F_use, device=dev) - (k - 1)).remainder(k)
        a_all = (
            self.act_pack(acts_4d)
            + self.act_slot(s_idx)[None, :, None, None, :].to(acts_4d.dtype)
        ).reshape(B * F_use, A, dt_dim).to(x_tube.dtype)

        am_4d = b["act_mask"][:, :F_use].view(B, F_use, 2, K)
        axy_4d = b["act_xy"][:, :F_use].view(B, F_use, 2, K, 2)
        atgt_4d = b["act_target"][:, :F_use].view(B, F_use, 2, K)
        am_pad = F.pad(am_4d[:, k - 1 :], (0, 0, 0, 0, 0, k - 1), value=False)
        axy_pad = F.pad(axy_4d[:, k - 1 :], (0, 0, 0, 0, 0, 0, 0, k - 1))
        atgt_pad = F.pad(atgt_4d[:, k - 1 :], (0, 0, 0, 0, 0, k - 1), value=-1)

        am_t_2k = am_pad.view(B, M, k, 2, K).transpose(2, 3).reshape(B, M, 2, k * K)
        axy_t = axy_pad.view(B, M, k, 2, K, 2).transpose(2, 3).reshape(B, M, k * A, 2)
        atgt_t = atgt_pad.view(B, M, k, 2, K).transpose(2, 3).reshape(B, M, k * A)
        if R > 0 and K > 0:
            in_r = (atgt_t >= 0) & (atgt_t < R)
            r_ok = valid_r.gather(2, atgt_t.clamp(0, R - 1).long())
            atgt_t = torch.where(in_r & r_ok, atgt_t, -1)
        else:
            atgt_t = torch.full_like(atgt_t, -1)

        sel = b.get("act_sel")
        sel_t = None
        if sel is not None and len(sel) and K > 0 and R > 0:
            sel_d = sel.to(dev)
            b_i, f_i = sel_d[:, 0] // F_, sel_d[:, 0] % F_
            keep = (
                (f_i >= k - 1)
                & (f_i < F_use)
                & (sel_d[:, 2] >= 0)
                & (sel_d[:, 2] < R)
            )
            if keep.any():
                b_i, f_i, sl_i, r_i = (
                    b_i[keep],
                    f_i[keep],
                    sel_d[keep, 1],
                    sel_d[keep, 2],
                )
                m_i = (f_i - (k - 1)) // k
                s_i = (f_i - (k - 1)) % k
                rok = valid_r[b_i, m_i, r_i]
                if rok.any():
                    b_i, m_i, s_i, sl_i, r_i = (
                        b_i[rok],
                        m_i[rok],
                        s_i[rok],
                        sl_i[rok],
                        r_i[rok],
                    )
                    bm_t = b_i * M + m_i
                    sl_t = ((sl_i // K) * k + s_i) * K + (sl_i % K)
                    sel_t = torch.stack([bm_t, sl_t, r_i], 1)

        c_t, _, fa_t = self._build_ctx(
            B,
            M,
            T_t,
            g0,
            valid_t,
            pos_t,
            t_mask_t,
            loops[:, kf],
            b["engaged_row"][:, kf],
            b["parent_row"][:, kf],
            b["ord_row"][:, kf, :, 0],
            b["mask"][:, kf],
            am_t_2k,
            axy_t,
            atgt_t,
            sel_t,
        )
        t_cs_act = cos_sin(self.rope_t.angles(loops))
        h2_t = t_cs_act.shape[-2]
        c_t["tubelet"] = k
        c_t["F_act"] = F_use
        c_t["K"] = K
        c_t["a_mask"] = am_4d.transpose(1, 2).reshape(B * 2, F_use * K)
        c_t["a_ang"] = (
            t_cs_act.repeat_interleave(K, 1)[:, None]
            .expand(B, 2, F_use * K, h2_t, 2)
            .reshape(B * 2, F_use * K, h2_t, 2)
        )
        c_t["act_compact"] = self._pack_act_timeline(c_t["a_mask"], c_t["a_ang"])

        dl_t = torch.zeros_like(x_tube)
        for i, blk in enumerate(self.blocks):
            last_t = (not last_action and i == len(self.blocks) - 1) or not len(fa_t)
            if ckpt:
                x_tube, dl_t, a_all = checkpoint(
                    lambda x_, dl_, a_, _b=blk, _l=last_t: _b(
                        x_, dl_, a_, c_t, _l
                    ),
                    x_tube,
                    dl_t,
                    a_all,
                    use_reentrant=True,
                )
            else:
                x_tube, dl_t, a_all = blk(x_tube, dl_t, a_all, c_t, last_t)

        xt = x_tube.view(B, M, T_t, dt_dim)
        ut = (x_tube + dl_t).view(B, M, T_t, dt_dim)
        out_fn = lambda z: dict(
            ent=self.norm_e(z[:, :, g0 : g0 + R]),
            eff=self.norm_eff(z[:, :, g0 + R : n_t]),
            cells=self.norm_c(z[:, :, n_t:]),
            glob=self.norm_g(z[:, :, :g0]),
        )
        upd = out_fn(ut)
        upd["act"] = self.norm_a(a_all.view(B, F_use, A, dt_dim))
        upd["eff_src_slot"] = eff_src_slot
        return out_fn(xt), upd


@torch.no_grad()
def pointer_pairs(target, fa, A, n, g0, sel=None):
    """Partition active frames fa into [:n_u] unbiased and [n_u:] biased subsets, and
    precompute sparse (pi, qi, ki, code_idx) edge tuples once per pass."""
    t = target[fa]
    fi, ai = torch.nonzero(t >= 0, as_tuple=True)
    row, code = t[fi, ai] + g0, torch.ones_like(fi)
    if sel is not None and len(sel):
        sel = sel.to(t.device)
        si, pi = torch.nonzero(sel[:, :1] == fa[None], as_tuple=True)
        fi = torch.cat([fi, pi])
        ai = torch.cat([ai, sel[si, 1]])
        row = torch.cat([row, sel[si, 2] + g0])
        code = torch.cat([code, torch.full_like(pi, 3)])
    if not len(fi):
        return fa, int(len(fa)), None, None
    has_b = torch.zeros(len(fa), dtype=torch.bool, device=t.device)
    has_b[fi] = True
    u_idx = torch.nonzero(~has_b, as_tuple=True)[0]
    b_idx = torch.nonzero(has_b, as_tuple=True)[0]
    fa_ord = torch.cat([fa[u_idx], fa[b_idx]])
    n_u, n_b = int(len(u_idx)), int(len(b_idx))
    remap = torch.full((len(fa),), -1, dtype=torch.long, device=t.device)
    remap[b_idx] = torch.arange(n_b, device=t.device)
    inv = remap[fi]
    ae = torch.zeros(n_b, A, n, dtype=torch.uint8, device=t.device)
    ea = torch.zeros(n_b, n, A, dtype=torch.uint8, device=t.device)
    ae[inv, ai, row] = code.to(torch.uint8)
    ea[inv, row, ai] = (code + 1).to(torch.uint8)
    pa, qa, ka = torch.nonzero(ae > 0, as_tuple=True)
    pe, qe, ke = torch.nonzero(ea > 0, as_tuple=True)
    edges_ae = (pa, qa, ka, ae[pa, qa, ka].long()) if len(pa) else None
    edges_ea = (pe, qe, ke, ea[pe, qe, ke].long()) if len(pe) else None
    return fa_ord, n_u, edges_ae, edges_ea


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


class Physics(nn.Module):
    """Causal L0 engine simulator: encoder, residual next-tubelet head, grounded unit/
    queue/pointer heads, single-pass symmetric births with final-step actionlet cross-
    attention, map fields, un-logged player economy deltas, command recipients, and
    gradient-detached diagnostic probes. forward(batch) -> (loss, logs)."""

    def __init__(
        self,
        vocab,
        avocab,
        d=128,
        moving_neutrals=(),
        larva=151,
        **kw,
    ):
        super().__init__()
        self.enc = PhysicsEncoder(vocab, avocab, d, **kw)
        h_dim = self.enc.d_tube
        self.g = mlp(h_dim, h_dim, 2 * h_dim)
        nn.init.zeros_(self.g[-1].weight)
        nn.init.zeros_(self.g[-1].bias)
        self.head = RowHead(h_dim, vocab.n_units, vocab.n_abilities)
        self.births = BirthHead(h_dim, kw.get("heads", 8), self.head)
        self.eff_persist = EffPersistHead(h_dim)
        self.eff_spawns = EffSpawnHead(h_dim, kw.get("heads", 8))
        self.map_dec = MapDecoder(h_dim, C_DYNAMIC)
        self.map_src_mlp = mlp(
            h_dim + self.map_dec.fourier.out_dim(2) + C_DYNAMIC, C_DYNAMIC, 128
        )
        nn.init.zeros_(self.map_src_mlp[-1].weight)
        nn.init.zeros_(self.map_src_mlp[-1].bias)
        self.map_stay = nn.Linear(h_dim, C_DYNAMIC)
        nn.init.zeros_(self.map_stay.weight)
        nn.init.constant_(self.map_stay.bias, 2.5)
        self.player_next = nn.Linear(h_dim, 2)  # raw (minerals, vespene) delta / 50
        self.recip = RecipientHead(h_dim)
        # Auxiliary diagnostic probes: strictly detached from trunk representations
        self.ply_probe = mlp(h_dim, len(PLY_PROBE_IDX), h_dim)
        self.pb_probe = MapDecoder(h_dim, 4)
        self.register_buffer(
            "critter",
            torch.tensor(sorted(moving_neutrals) or [-1], dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "active_flag_bits",
            torch.tensor(ACTIVE_FLAG_BITS, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "reg_need_vis",
            torch.tensor(
                [n not in REG_ALWAYS_VALID for n, _ in REG], dtype=torch.bool
            ),
            persistent=False,
        )
        self.larva = larva
        self.record_pos = False
        self.eval_cnts = None

    def predict_map(self, cells, cell, off, src_map):
        """Decode [BM, K, C_DYNAMIC] at sub-cell offsets `off` with f_src residual."""
        base_o = self.map_dec(cells, cell, off).float()
        x_c = torch.gather(cells, 1, cell[..., None].expand(-1, -1, cells.shape[-1]))
        f_off = self.map_dec.fourier(off).to(x_c.dtype)
        res_o = self.map_src_mlp(
            torch.cat([x_c, f_off, src_map.to(x_c.dtype)], dim=-1)
        ).float()
        stay_m = self.map_stay(x_c).float()
        return base_o + res_o + stay_m * (2.0 * src_map.float() - 1.0)

    @staticmethod
    def _unpack_dyn_map(m_raw, BM, dev):
        if m_raw.ndim == 4:
            flat = m_raw.reshape(BM, TOP * TOP).to(torch.int32)
            shifts = torch.arange(C_DYNAMIC, device=dev, dtype=torch.int32)
            return ((flat[..., None] >> shifts) & 1).float()
        m = m_raw.reshape(BM, TOP * TOP, C_DYNAMIC * PATCH * PATCH)
        centre = (PATCH * PATCH) // 2
        return m[..., centre :: PATCH * PATCH][:, :, :C_DYNAMIC].float()

    def compile_parts(self):
        """Compile encoder submodules plus the fused dense per-unit loss head."""
        self.enc.compile_parts()
        self._dense_raw_losses = torch.compile(
            self._dense_raw_losses, dynamic=True, recompile_limit=64
        )

    def forward(self, b):
        clean, out = self.enc(b)
        B, F_, R = b["mask"].shape
        k = max(1, self.enc.tubelet)
        dev = b["mask"].device
        if k > 1:
            M = F_ // k
            m_start = 1 if M > 2 else 0
            m_end = M - 1
            m_idx = torch.arange(m_start, m_end, device=dev)
            f_src = m_idx * k + (k - 1)
            f_tgt = f_src + k
        else:
            m_start = 0
            m_end = F_ - 1
            m_idx = torch.arange(m_start, m_end, device=dev)
            f_src = m_idx
            f_tgt = f_src + 1
        logs = {}
        self.eval_cnts = collections.Counter() if self.record_pos else None
        # who is scored: alive at source anchor frame f_src, with a target frame f_tgt
        sc = b["mask"][:, f_src] & b["alive"][:, f_src] & b["mask"][:, f_tgt]
        nxt = {k_: v[:, m_start:m_end] for k_, v in out.items() if k_ != "act"}
        self._rows = nxt["ent"]  # what pointers point into
        s = nxt["ent"]
        p = s + self.g(s)
        bi, ui, ri = torch.nonzero(sc, as_tuple=True)
        raw, raw_logs = self.raw_losses(p, s, b, sc, bi, ui, ri, f_src, f_tgt)
        logs.update(raw_logs)
        brth, blogs = self.birth_losses(nxt, out["act"], b, f_src, f_tgt)
        logs.update(blogs)
        eff_l, elogs = self.effect_losses(nxt, out["act"], b, f_src, f_tgt)
        logs.update(elogs)
        mp, probe_pb, mlogs = self.map_loss(
            nxt["cells"], clean["cells"][:, m_start:m_end], b, f_src, f_tgt
        )
        logs.update(mlogs)
        pn, probe_ply, plogs = self.player_losses(
            nxt["glob"][:, :, 1:3],
            clean["glob"][:, m_start:m_end, 1:3],
            b,
            f_src,
            f_tgt,
        )
        logs.update(plogs)
        rc, rlogs = self.recipient_loss(clean, out["act"], b)
        logs.update(rlogs)
        main_loss = raw + brth + eff_l + mp + pn + rc
        loss = main_loss + 0.05 * (probe_ply + probe_pb)
        logs["loss"] = main_loss
        if self.record_pos and self.eval_cnts is not None:
            self.eff_eval = dict(self.eval_cnts)
        self.eval_cnts = None
        self._rows = None  # it would hold this pass's graph on the model until the next
        return loss, logs

    def player_losses(self, nxt_ply, clean_ply, b, f_src, f_tgt):
        """Predict raw un-logged (minerals, vespene) change over the tubelet on nxt_ply,
        and run a detached macro-count probe on clean_ply.detach() (zero trunk gradient)."""
        raw_src = torch.expm1(b["ply_num"][:, f_src, :, :2].float() * 8.0)
        raw_tgt = torch.expm1(b["ply_num"][:, f_tgt, :, :2].float() * 8.0)
        econ_tgt = (raw_tgt - raw_src) / 50.0
        econ_pred = self.player_next(nxt_ply).float()
        econ_w = torch.where(econ_tgt.abs() > 0.01, 1.0, 0.25)
        w_sum = econ_w.sum().clamp(min=1.0)
        pn = (
            F.smooth_l1_loss(econ_pred, econ_tgt, beta=0.5, reduction="none") * econ_w
        ).sum() / w_sum
        with torch.no_grad():
            pn_copy = (
                F.smooth_l1_loss(
                    torch.zeros_like(econ_tgt), econ_tgt, beta=0.5, reduction="none"
                )
                * econ_w
            ).sum() / w_sum

        # Detached probe: 7 macroscopic counts at f_src (no gradient into encoder/trunk)
        probe_in = clean_ply.detach()
        probe_pred = self.ply_probe(probe_in).float()
        probe_tgt = b["ply_num"][:, f_src][..., PLY_PROBE_IDX].float() * 8.0
        probe_ply = F.smooth_l1_loss(probe_pred, probe_tgt, beta=0.5)
        with torch.no_grad():
            cnt_pred = torch.expm1(probe_pred.detach().clamp(min=0.0, max=8.0))
            cnt_tgt = torch.expm1(probe_tgt)
            probe_mae = (cnt_pred - cnt_tgt).abs().mean()
        logs = dict(
            player=pn,
            player_copy=pn_copy,
            probe_ply=probe_ply.detach(),
            probe_ply_mae=probe_mae,
        )
        return pn, probe_ply, logs

    def recipient_loss(self, clean, act, b):
        """Which units carry out each command: scores each command token against every
        live owned row at the causal anchor frame f_anc with F1-calibrated BCE."""
        pr = b.get("act_recip")
        zero = act.sum() * 0
        if pr is None or not len(pr):
            return zero, {}
        B, F_act, A, d = act.shape
        F_b, R = b["mask"].shape[1], b["mask"].shape[2]
        if R == 0:
            return zero, {}
        k = max(1, self.enc.tubelet)
        pr = pr.to(act.device)
        bi_p, fi_p = pr[:, 0] // F_b, pr[:, 0] % F_b
        keep = (fi_p >= k - 1) & (fi_p < F_act) & (pr[:, 2] >= 0) & (pr[:, 2] < R)
        if not keep.any():
            return zero, {}
        bi_p, fi_p, sl_p, r_p = bi_p[keep], fi_p[keep], pr[keep, 1], pr[keep, 2]
        uq, inv = torch.unique((bi_p * F_act + fi_p) * A + sl_p, return_inverse=True)
        fb, sl = uq // A, uq % A
        ub, uf = fb // F_act, fb % F_act
        um = (uf - (k - 1)) // k
        f_anc = um * k + (k - 1)
        U = len(uq)

        who = b["act_player"][ub, uf, sl]
        own = b["owner"][ub] == who[:, None]
        valid = (
            b["mask"][ub, uf]
            & b["alive"][ub, uf]
            & b["mask"][ub, f_anc]
            & b["alive"][ub, f_anc]
        )
        valid_own = valid & own

        sel_add = torch.zeros((U, R), dtype=torch.float32, device=act.device)
        any_sel_add = torch.zeros((U, 1), dtype=torch.float32, device=act.device)
        act_sel = b.get("act_sel")
        if act_sel is not None and len(act_sel):
            asd = act_sel.to(act.device)
            sb, sf = asd[:, 0] // F_b, asd[:, 0] % F_b
            ssl, sr = asd[:, 1], asd[:, 2]
            s_ok = (sf < F_act) & (sr >= 0) & (sr < R)
            if s_ok.any():
                sb, sf, ssl, sr = sb[s_ok], sf[s_ok], ssl[s_ok], sr[s_ok]
                swho = b["act_player"][sb, sf, ssl]
                prior = (
                    (ub[:, None] == sb[None])
                    & (swho[None] == who[:, None])
                    & (sf[None] >= f_anc[:, None])
                    & (
                        (sf[None] < uf[:, None])
                        | ((sf[None] == uf[:, None]) & (ssl[None] < sl[:, None]))
                    )
                )
                ui_m, si_m = torch.nonzero(prior, as_tuple=True)
                if len(ui_m):
                    sr_m = sr[si_m]
                    rok = b["mask"][ub[ui_m], f_anc[ui_m], sr_m]
                    ui_m, sr_m = ui_m[rok], sr_m[rok]
                    if len(ui_m):
                        sel_add[ui_m, sr_m] = 1.0
                        any_sel_add[ui_m, 0] = 1.0

        grp_anc = (b["groups"][ub, f_anc] > 0).float() * b["mask"][
            ub, f_anc, :, None
        ].float()
        grp_feat = torch.cat(
            [grp_anc, sel_add[..., None], any_sel_add[:, :, None].expand(-1, R, 1)],
            dim=-1,
        )
        xy_a = b["act_xy"][ub, uf, sl].float()
        placed = b["act_num"][ub, uf, sl, 11].float()
        xy_r = (
            torch.stack([b["x"][ub, f_anc], b["y"][ub, f_anc]], -1).float()
            - b["center"][ub, None].float()
        ) * b["mask"][ub, f_anc, :, None].float()

        logit = self.recip(
            act[ub, uf, sl], clean["ent"][ub, um], xy_a, placed, xy_r, grp_feat
        )
        tgt = torch.zeros_like(logit)
        tgt[inv, r_p] = 1.0
        tgt = tgt * valid_own.float()

        pw = torch.tensor(1.25, device=logit.device)
        bce = (
            F.binary_cross_entropy_with_logits(
                logit, tgt, pos_weight=pw, reduction="none"
            )
            * valid_own.float()
        )
        loss = (0.25 * bce.sum(1) / tgt.sum(1).clamp(min=1)).mean()
        with torch.no_grad():
            f1 = lambda p_: 2 * (p_ * tgt).sum() / (p_.sum() + tgt.sum()).clamp(min=1)
            sel = (b["groups"][ub, uf, :, 10] > 0) & own
            logs = dict(
                recip=loss,
                recip_f1=f1(((logit > 0) & valid_own).float()),
                recip_sel_f1=f1((sel & valid).float()),
            )
        return loss, logs

    def _dense_raw_losses(
        self,
        x_sc,
        crit,
        larva,
        x_ft,
        x_fp,
        y_ft,
        y_fp,
        n_ft,
        n_fp,
        typ_ft,
        typ_fp,
        ins_ft,
        ins_fp,
        alive_ft,
        buffs_ft_u8,
        buffs_fp_u8,
        grp_ft,
        grp_fp,
        eng_ft,
        eng_fp,
        rf_ft,
        rf_fp,
        pos_scale,
    ):
        whole = torch.where(crit, DISCOUNT, 1.0).float()
        wt = whole[:, None]
        pos_w = torch.where(larva, whole * DISCOUNT, whole)
        pw_sum = pos_w.sum().clamp(min=1e-6)

        disp_fp = n_fp[:, I["display"]].long().clamp(0, 3)
        disp_ft = n_ft[:, I["display"]].long().clamp(0, 3)
        cloak_fp = n_fp[:, I["cloak"]].long().clamp(0, 3)
        cloak_ft = n_ft[:, I["cloak"]].long().clamp(0, 3)
        f_shifts = torch.arange(N_FLAGS, device=x_sc.device)
        flags_p_raw = n_fp[:, I["flags"]].long() | (n_fp[:, I["flags2"]].long() << 8)
        flags_t_raw = n_ft[:, I["flags"]].long() | (n_ft[:, I["flags2"]].long() << 8)
        flags_fp = ((flags_p_raw[:, None] >> f_shifts) & 1).float()
        flags_ft = ((flags_t_raw[:, None] >> f_shifts) & 1).float()
        bits_fp = torch.stack(
            [n_fp[:, I["add_on"]] >= 0, eng_fp >= 0, rf_fp[:, 0] > 0], -1
        ).float()
        bits_ft = torch.stack(
            [n_ft[:, I["add_on"]] >= 0, eng_ft >= 0, rf_ft[:, 0] > 0], -1
        ).float()

        o = self.head.predict_next(
            x_sc,
            typ_fp,
            ins_fp,
            disp_fp,
            cloak_fp,
            flags_fp,
            grp_fp,
            bits_fp,
            buffs_fp_u8,
            rf_fp,
        )

        tpos = torch.stack([(x_ft - x_fp) / pos_scale, (y_ft - y_fp) / pos_scale], -1)
        lp = F.smooth_l1_loss(
            o["reg"][:, :2].float(), tpos, beta=0.04, reduction="none"
        ).sum(-1)
        l_pos = (lp * pos_w).sum() / pw_sum
        l_pos_copy = (
            F.smooth_l1_loss(
                torch.zeros_like(tpos), tpos, beta=0.04, reduction="none"
            ).sum(-1)
            * pos_w
        ).sum() / pw_sum

        # Pillar 1: mask out fog-of-war SNAPSHOT <-> VISIBLE fake stat jumps (display == 2)
        # on non-geometry/non-contents REG channels, and normalize per channel by active
        # delta events with beta = 0.20 so combat/energy/mining deltas never shrink to 0.
        reg_t = torch.stack(
            [(n_ft[:, I[k_]] - n_fp[:, I[k_]]) / dv for k_, dv in REG], -1
        ).float()
        j = [k_ for k_, _ in REG].index("facing")
        reg_t[:, j] = torch.remainder(reg_t[:, j] + math.pi, 2 * math.pi) - math.pi
        is_snap = (disp_fp == 2) | (disp_ft == 2)
        reg_valid = (~(is_snap[:, None] & self.reg_need_vis[None, :])).float()
        chg_r = (reg_t.abs() > 0.005) & (reg_valid > 0.5)
        w_reg = wt * reg_valid * torch.where(chg_r, 1.5, 1.0)
        min_chg_r = (0.002 * (wt * reg_valid).sum(0)).clamp(min=16.0)
        n_chg_r = torch.maximum((wt * chg_r.float()).sum(0), min_chg_r)
        err_r = F.smooth_l1_loss(
            o["reg"][:, 2:].float(), reg_t, beta=0.20, reduction="none"
        )
        l_reg = 0.25 * ((err_r * w_reg).sum(0) / n_chg_r).mean()

        # Pillar 2: F1-calibrated event-normalized transition losses on f_src-residual heads
        typ_chg = typ_ft.long() != typ_fp.long()
        ce_typ = F.cross_entropy(o["type"], typ_ft.long(), reduction="none")
        l_type = (
            0.01
            * (ce_typ * whole * torch.where(typ_chg, 2.0, 1.0)).sum()
            / (whole * typ_chg.float()).sum().clamp(min=8.0)
        )

        ins_chg = ins_ft.long() != ins_fp.long()
        cargo_chg = ins_chg & ((ins_fp.long() == 1) | (ins_ft.long() == 1))
        w_ins = whole * torch.where(cargo_chg, 2.5, torch.where(ins_chg, 1.5, 1.0))
        ce_ins = F.cross_entropy(o["inside"], ins_ft.long(), reduction="none")
        l_inside = (
            0.03 * (ce_ins * w_ins).sum() / (whole * ins_chg.float()).sum().clamp(min=8.0)
        )

        act_fb = self.active_flag_bits
        fl_p_a, fl_t_a = flags_fp[:, act_fb], flags_ft[:, act_fb]
        fl_chg = (fl_t_a - fl_p_a).abs() > 0.5
        bce_fl = F.binary_cross_entropy_with_logits(
            o["flags"][:, act_fb], fl_t_a, reduction="none"
        )
        l_flags = (
            0.025
            * (bce_fl * wt * torch.where(fl_chg, 2.0, 1.0)).sum()
            / (wt * fl_chg.float()).sum().clamp(min=16.0)
        )

        disp_chg = disp_ft != disp_fp
        ce_disp = F.cross_entropy(o["display"], disp_ft, reduction="none")
        l_display = (
            0.02
            * (ce_disp * whole * torch.where(disp_chg, 1.5, 1.0)).sum()
            / (whole * disp_chg.float()).sum().clamp(min=8.0)
        )

        clk_chg = cloak_ft != cloak_fp
        ce_clk = F.cross_entropy(o["cloak"], cloak_ft, reduction="none")
        l_cloak = (
            0.02
            * (ce_clk * whole * torch.where(clk_chg, 2.0, 1.0)).sum()
            / (whole * clk_chg.float()).sum().clamp(min=8.0)
        )

        # F1-calibrated tubelet death BCE: pos_weight = 2.0, normalized by deaths
        died = (~alive_ft).float()
        pw_alive = torch.tensor(2.0, device=x_sc.device)
        bce_alive = (
            F.binary_cross_entropy_with_logits(
                o["alive"],
                died,
                pos_weight=pw_alive,
                reduction="none",
            )
            * whole
        )
        l_alive = 0.1 * bce_alive.sum() / (died * whole).sum().clamp(min=4.0)
        pred_d = (o["alive"].detach() > 0).float() * (~crit).float()
        tgt_d = died * (~crit).float()
        death_f1 = 2.0 * (pred_d * tgt_d).sum() / (pred_d.sum() + tgt_d.sum()).clamp(
            min=1.0
        )

        bf_t, bf_p = buffs_ft_u8.float(), buffs_fp_u8.float()
        bf_chg = (bf_t - bf_p).abs() > 0.5
        bce_bf = F.binary_cross_entropy_with_logits(o["buffs"], bf_t, reduction="none")
        l_buffs = (
            0.025
            * (bce_bf * wt * torch.where(bf_chg, 2.0, 1.0)).sum()
            / (wt * bf_chg.float()).sum().clamp(min=16.0)
        )

        gr_t, gr_p = grp_ft.float(), grp_fp.float()
        gr_chg = (gr_t - gr_p).abs() > 0.5
        bce_gr = F.binary_cross_entropy_with_logits(o["groups"], gr_t, reduction="none")
        l_groups = (
            0.025
            * (bce_gr * wt * torch.where(gr_chg, 2.0, 1.0)).sum()
            / (wt * gr_chg.float()).sum().clamp(min=16.0)
        )

        bt_chg = (bits_ft - bits_fp).abs() > 0.5
        bce_bt = F.binary_cross_entropy_with_logits(
            o["bits"], bits_ft, reduction="none"
        )
        l_bits = (
            0.02
            * (bce_bt * wt * torch.where(bt_chg, 2.0, 1.0)).sum()
            / (wt * bt_chg.float()).sum().clamp(min=8.0)
        )

        rm = rf_ft[:, 0] > 0
        r_chg = rm & (
            (rf_fp[:, 0] <= 0)
            | ((rf_ft[:, 1:3].float() - rf_fp[:, 1:3].float()).abs().amax(-1) > 0.005)
        )
        rl_elem = F.smooth_l1_loss(
            o["rally"], rf_ft[:, 1:3].float(), beta=0.02, reduction="none"
        ).sum(-1)
        w_rl = whole * rm.float() * torch.where(r_chg, 2.0, 1.0)
        l_rally = (
            0.05 * (rl_elem * w_rl).sum() / (whole * r_chg.float()).sum().clamp(min=4.0)
        )
        return (
            l_pos,
            l_pos_copy,
            l_reg,
            l_type,
            l_inside,
            l_flags,
            l_display,
            l_cloak,
            l_alive,
            death_f1,
            l_buffs,
            l_groups,
            l_bits,
            l_rally,
        )

    @staticmethod
    def _gather_bag_u8(at_flat, bag_at, bag_id, n_rows, n_cols, total_bfr, dev):
        out = torch.zeros(n_rows, n_cols, device=dev, dtype=torch.uint8)
        if n_rows == 0 or len(bag_at) == 0:
            return out
        uq, inv = torch.unique(at_flat, return_inverse=True)
        u_pos = torch.full((total_bfr,), -1, device=dev, dtype=torch.long)
        u_pos[uq] = torch.arange(len(uq), device=dev)
        hit = u_pos[bag_at]
        ok = hit >= 0
        if ok.any():
            u_out = torch.zeros(len(uq), n_cols, device=dev, dtype=torch.uint8)
            u_out[hit[ok], bag_id[ok]] = 1
            return u_out[inv]
        return out

    def raw_losses(self, p, s, b, sc, bi, ui, ri, f_src, f_tgt):
        if not len(bi):
            z = p.sum() * 0
            logs = {
                k_: z
                for k_ in (
                    "pos",
                    "pos_copy",
                    "reg",
                    "type",
                    "inside",
                    "flags",
                    "display",
                    "cloak",
                    "alive",
                    "death_f1",
                    "buffs",
                    "groups",
                    "bits",
                    "rally",
                    "queues",
                    "abilities",
                    "pointers",
                )
            }
            if self.record_pos:
                empty = np.zeros(0, dtype=np.float32)
                self.pos_eval = {
                    k_: (empty, empty)
                    for k_ in (
                        "player moving",
                        "player still",
                        "player all",
                        "larva",
                        "critter",
                        "other neutral",
                    )
                }
            return z, logs
        B, F_, R = b["mask"].shape
        k = max(1, self.enc.tubelet)
        pos_scale = POS_SCALE * k
        fp = f_src[ui]
        ft = f_tgt[ui]
        own = b["owner"][bi, ri]
        rt = b["rtype"][bi, ft, ri]
        crit = (own == NEUTRAL) & torch.isin(rt, self.critter)
        larva = rt == self.larva
        x_sc = p[bi, ui, ri]
        x_ft, x_fp = b["x"][bi, ft, ri], b["x"][bi, fp, ri]
        y_ft, y_fp = b["y"][bi, ft, ri], b["y"][bi, fp, ri]
        n_ft, n_fp = b["num"][bi, ft, ri], b["num"][bi, fp, ri]
        alive_ft = b["alive"][bi, ft, ri]
        typ_ft, typ_fp = b["type"][bi, ft, ri], b["type"][bi, fp, ri]
        ins_ft, ins_fp = b["inside"][bi, ft, ri], b["inside"][bi, fp, ri]
        grp_ft, grp_fp = b["groups"][bi, ft, ri], b["groups"][bi, fp, ri]
        eng_ft, eng_fp = b["engaged_row"][bi, ft, ri], b["engaged_row"][bi, fp, ri]
        rf_ft, rf_fp = b["rally_first"][bi, ft, ri], b["rally_first"][bi, fp, ri]

        total_bfr = B * F_ * R
        key_ft = (bi * F_ + ft) * R + ri
        key_fp = (bi * F_ + fp) * R + ri
        buffs_ft_u8 = self._gather_bag_u8(
            key_ft, b["buff_at"], b["buff_id"], len(bi), N_BUFF_IDS, total_bfr, p.device
        )
        buffs_fp_u8 = self._gather_bag_u8(
            key_fp, b["buff_at"], b["buff_id"], len(bi), N_BUFF_IDS, total_bfr, p.device
        )

        (
            l_pos,
            l_pos_copy,
            l_reg,
            l_type,
            l_inside,
            l_flags,
            l_display,
            l_cloak,
            l_alive,
            death_f1,
            l_buffs,
            l_groups,
            l_bits,
            l_rally,
        ) = self._dense_raw_losses(
            x_sc,
            crit,
            larva,
            x_ft,
            x_fp,
            y_ft,
            y_fp,
            n_ft,
            n_fp,
            typ_ft,
            typ_fp,
            ins_ft,
            ins_fp,
            alive_ft,
            buffs_ft_u8,
            buffs_fp_u8,
            grp_ft,
            grp_fp,
            eng_ft,
            eng_fp,
            rf_ft,
            rf_fp,
            pos_scale,
        )
        logs = dict(
            pos=l_pos,
            pos_copy=l_pos_copy,
            reg=l_reg,
            type=l_type,
            inside=l_inside,
            flags=l_flags,
            display=l_display,
            cloak=l_cloak,
            alive=l_alive,
            death_f1=death_f1,
            buffs=l_buffs,
            groups=l_groups,
            bits=l_bits,
            rally=l_rally,
        )
        if self.record_pos:
            with torch.no_grad():
                pred_pos = self.head.reg(x_sc)[:, :2].float()
                tpos = torch.stack(
                    [(x_ft - x_fp) / pos_scale, (y_ft - y_fp) / pos_scale], -1
                )
                self.pos_eval = self._position_errors(
                    pred_pos,
                    tpos,
                    own,
                    crit,
                    larva,
                    alive_ft & (ins_fp == 0) & (ins_ft == 0),
                    pos_scale,
                    k,
                )
                if self.eval_cnts is not None:
                    ply_m = own != NEUTRAL
                    # Morph accuracy on player units
                    m_chg = ply_m & (typ_ft.long() != typ_fp.long())
                    if m_chg.any():
                        n_u = self.head.unit.out_features
                        pr_typ = (
                            self.head.unit(x_sc[m_chg])
                            .float()
                            .scatter_add(
                                1,
                                typ_fp[m_chg].long().clamp(0, n_u - 1)[:, None],
                                self.head.stay_type(x_sc[m_chg]).float(),
                            )
                            .argmax(-1)
                        )
                        self.eval_cnts["type_chg_ok"] += int(
                            (pr_typ == typ_ft[m_chg].long()).sum().item()
                        )
                        self.eval_cnts["type_chg_tot"] += int(m_chg.sum().item())
                    # Active-flag flip F1 and group flip F1 on player units
                    if ply_m.any():
                        xp = x_sc[ply_m]
                        f_sh = self.active_flag_bits
                        fp_r = n_fp[ply_m, I["flags"]].long() | (
                            n_fp[ply_m, I["flags2"]].long() << 8
                        )
                        ft_r = n_ft[ply_m, I["flags"]].long() | (
                            n_ft[ply_m, I["flags2"]].long() << 8
                        )
                        fl_p = ((fp_r[:, None] >> f_sh) & 1).float()
                        fl_t = ((ft_r[:, None] >> f_sh) & 1).float()
                        fl_pr = (
                            self.head.flags(xp)[:, f_sh].float()
                            + self.head.stay_flags(xp)[:, f_sh].float()
                            * (2.0 * fl_p - 1.0)
                        ) > 0.0
                        pr_f_chg = fl_pr.float() != fl_p
                        gt_f_chg = fl_t != fl_p
                        self.eval_cnts["flags_chg_tp"] += int(
                            (pr_f_chg & gt_f_chg & (fl_pr.float() == fl_t)).sum().item()
                        )
                        self.eval_cnts["flags_chg_pred"] += int(pr_f_chg.sum().item())
                        self.eval_cnts["flags_chg_gt"] += int(gt_f_chg.sum().item())

                        gp_p, gp_t = grp_fp[ply_m].float(), grp_ft[ply_m].float()
                        gp_pr = (
                            self.head.groups(xp).float()
                            + self.head.stay_groups(xp).float() * (2.0 * gp_p - 1.0)
                        ) > 0.0
                        pr_g_chg = gp_pr.float() != gp_p
                        gt_g_chg = gp_t != gp_p
                        self.eval_cnts["groups_chg_tp"] += int(
                            (pr_g_chg & gt_g_chg & (gp_pr.float() == gp_t)).sum().item()
                        )
                        self.eval_cnts["groups_chg_pred"] += int(pr_g_chg.sum().item())
                        self.eval_cnts["groups_chg_gt"] += int(gt_g_chg.sum().item())
        self._ptr_stats = None
        logs["queues"], logs["abilities"], logs["pointers"] = self._sampled(
            p, b, bi, ui, ri, f_src, f_tgt
        )
        if self._ptr_stats is not None:
            logs["ptr_acc"], logs["ptr_copy"] = self._ptr_stats
        total = (
            POS_BOOST * logs["pos"]
            + logs["reg"]
            + logs["type"]
            + logs["inside"]
            + logs["flags"]
            + logs["display"]
            + logs["cloak"]
            + logs["alive"]
            + logs["queues"]
            + logs["buffs"]
            + logs["groups"]
            + logs["bits"]
            + logs["rally"]
            + logs["abilities"]
            + logs["pointers"]
        )
        return total, logs

    def _position_errors(self, pred, tpos, own, crit, larva, keep, pos_scale, k):
        """For evaluation: {kind of row: (model errors, copy errors)}, one per row, map
        units across the predicted k-loop tubelet."""
        step = tpos.norm(dim=-1) * pos_scale
        err = (pred - tpos).norm(dim=-1) * pos_scale
        neutral = own == NEUTRAL
        player = ~neutral & ~larva
        moving = step > 0.003 * k
        kinds = {
            "player moving": player & moving,
            "player still": player & ~moving,
            "player all": player,
            "larva": larva & ~neutral,
            "critter": crit,
            "other neutral": neutral & ~crit,
        }
        return {
            k_: (err[v & keep].cpu().numpy(), step[v & keep].cpu().numpy())
            for k_, v in kinds.items()
        }

    @staticmethod
    def _gather_slots(b, at, total_bfr, dev):
        """Gather [N, QUEUE + PROD] slot abilities and [N, QUEUE + PROD, 4] slot
        regressions for flat indices `at` (handling duplicate indices cleanly)."""
        if "slot_abil" in b and "slot_reg" in b:
            return (
                b["slot_abil"].reshape(-1, QUEUE + PROD)[at],
                b["slot_reg"].reshape(-1, QUEUE + PROD, 4)[at],
            )
        uq, inv = torch.unique(at, return_inverse=True)
        u_pos = torch.full((total_bfr,), -1, device=dev)
        u_pos[uq] = torch.arange(len(uq), device=dev)
        u_abil = torch.zeros(
            (len(uq), QUEUE + PROD), dtype=torch.long, device=dev
        )
        u_reg = torch.zeros(
            (len(uq), QUEUE + PROD, 4), dtype=torch.float32, device=dev
        )
        if len(b["ord_at"]):
            oq = b["ord_in_q"] if "ord_in_q" in b else (b["ord_slot"] < QUEUE)
            o_hit = u_pos[b["ord_at"][oq]]
            o_ok = o_hit >= 0
            if o_ok.any():
                o_sl = b["ord_slot"][oq][o_ok]
                u_abil[o_hit[o_ok], o_sl] = b["ord_abil"][oq][o_ok]
                u_reg[o_hit[o_ok], o_sl] = b["ord_feat"][oq][o_ok]
        if len(b["prod_at"]):
            pq = b["prod_in_q"] if "prod_in_q" in b else (b["prod_slot"] < PROD)
            p_hit = u_pos[b["prod_at"][pq]]
            p_ok = p_hit >= 0
            if p_ok.any():
                p_sl = QUEUE + b["prod_slot"][pq][p_ok]
                u_abil[p_hit[p_ok], p_sl] = b["prod_abil"][pq][p_ok]
                u_reg[p_hit[p_ok], p_sl, 3] = b["prod_feat"][pq][p_ok, 0]
        return u_abil[inv], u_reg[inv]

    def _sampled(self, p, b, bi, ui, ri, f_src, f_tgt, cap=4096, ptr_cap=1024):
        """The large outputs, on a sample of the scored tokens, rows whose queue,
        engaged target, or available-ability set changes across (f_src, f_tgt] first:
        order and production slots, available abilities, and pointers over rows + -1."""
        B, F_, R = b["mask"].shape
        k = max(1, self.enc.tubelet)
        af_idx = f_src[:, None] + torch.arange(k, device=p.device)[None, :]
        rc_src = b["row_changed"] if "row_changed" in b else b["queue_changed"]
        ch = rc_src[:, af_idx].any(dim=2)[bi, ui, ri]
        sel_ch = torch.nonzero(ch, as_tuple=True)[0]
        rest = torch.nonzero(~ch, as_tuple=True)[0]
        ch_cap = (cap * 3) // 4
        if len(sel_ch) > ch_cap:
            sel_ch = sel_ch[torch.randperm(len(sel_ch), device=p.device)[:ch_cap]]
        take = rest[
            torch.randperm(len(rest), device=p.device)[: max(0, cap - len(sel_ch))]
        ]
        sel = torch.cat([sel_ch, take])
        if not len(sel):
            z = p.sum() * 0
            return z, z, z
        x = p[bi[sel], ui[sel], ri[sel]]
        ft_sel = f_tgt[ui[sel]]
        fanc_sel = f_src[ui[sel]]
        at = (bi[sel] * F_ + ft_sel) * R + ri[sel]
        at_q_anc = (bi[sel] * F_ + fanc_sel) * R + ri[sel]
        anc_abil, anc_reg = self._gather_slots(b, at_q_anc, B * F_ * R, p.device)
        t_abil, t_reg = self._gather_slots(b, at, B * F_ * R, p.device)
        abil, reg = self.head.slots(x, anc_abil=anc_abil, anc_reg=anc_reg)
        la = F.cross_entropy(abil.float().flatten(0, 1), t_abil.flatten())
        m = (t_abil > 0)[..., None].float()
        queues = la + 0.25 * (
            F.smooth_l1_loss(reg.float(), t_reg, beta=0.02, reduction="none") * m
        ).sum() / m.sum().clamp(min=1)
        # available abilities: f_src residual + F1-calibrated event-normalized BCE
        total_bfr = B * F_ * R
        t_ab = self._gather_bag_u8(
            at, b["ab_at"], b["ab_id"], len(sel), N_RAW_ABILITY_IDS, total_bfr, p.device
        ).float()
        src_ab = self._gather_bag_u8(
            at_q_anc,
            b["ab_at"],
            b["ab_id"],
            len(sel),
            N_RAW_ABILITY_IDS,
            total_bfr,
            p.device,
        ).float()
        ab_logits = self.head.predict_abilities(x, src_ab=src_ab)
        ab_chg = (t_ab - src_ab).abs() > 0.5
        bce_ab = F.binary_cross_entropy_with_logits(ab_logits, t_ab, reduction="none")
        abilities = (
            0.025
            * (bce_ab * torch.where(ab_chg, 2.0, 1.0)).sum()
            / ab_chg.float().sum().clamp(min=32.0)
        )
        # pointers: scored over unit rows 0..R-1 plus column R ("no target (-1)"),
        # conditioned strictly on the source tubelet's anchor frame fanc
        s2 = sel[torch.randperm(len(sel), device=p.device)[:ptr_cap]]
        bs, us, rs = bi[s2], ui[s2], ri[s2]
        fts = f_tgt[us]
        fanc = f_src[us]
        rows = self._rows[bs, us]
        at2 = (bs * F_ + fts) * R + rs
        at_anc = (bs * F_ + fanc) * R + rs
        now = torch.cat(
            [
                b["engaged_row"].reshape(-1)[at_anc][:, None],
                b["parent_row"].reshape(-1)[at_anc][:, None],
                b["ord_row"].reshape(-1, QUEUE)[at_anc],
            ],
            1,
        ).long()
        tgt = torch.cat(
            [
                b["engaged_row"].reshape(-1)[at2][:, None],
                b["parent_row"].reshape(-1)[at2][:, None],
                b["ord_row"].reshape(-1, QUEUE)[at2],
            ],
            1,
        ).long()
        xy_q = torch.stack([b["x"][bs, fanc, rs], b["y"][bs, fanc, rs]], -1)
        xy_k = torch.stack([b["x"][bs, fanc], b["y"][bs, fanc]], -1)
        logits = self.head.pointers(
            p[bs, us, rs], rows, xy_q=xy_q, xy_k=xy_k, now_ptr=now
        )
        # a target row not live in fanc can't be pointed at; column R (-1) is always valid
        if R > 0:
            live = b["mask"][bs, fanc]
            logits = torch.cat(
                [logits[..., :R].masked_fill(~live[:, None], -1e4), logits[..., R:]],
                dim=-1,
            ).float()
            valid_ptr = (tgt < 0) | (
                (tgt < R) & live.gather(1, tgt.clamp(0, R - 1))
            )
        else:
            logits = logits.float()
            valid_ptr = tgt < 0
        tgt_cls = torch.where(tgt >= 0, tgt, R)
        now_cls = torch.where((now >= 0) & (now < R), now, R)
        active_ptr = (now >= 0) | (tgt >= 0)
        if valid_ptr.any():
            ce_p = F.cross_entropy(
                logits[valid_ptr], tgt_cls[valid_ptr], reduction="none"
            )
            w_p = torch.where(active_ptr[valid_ptr], 1.0, 0.05)
            pointers = (ce_p * w_p).sum() / w_p.sum().clamp(min=1.0)
        else:
            pointers = p.sum() * 0
        # accuracy vs copy baseline across all active/acquiring/clearing pointer slots
        eval_ptr = valid_ptr & active_ptr
        with torch.no_grad():
            self._ptr_stats = (
                (
                    (logits.argmax(-1) == tgt_cls)[eval_ptr].float().mean(),
                    (now_cls == tgt_cls)[eval_ptr].float().mean(),
                )
                if eval_ptr.any()
                else None
            )
        return queues, abilities, pointers

    def birth_losses(self, out, act, b, f_src, f_tgt):
        B, F_, R = b["mask"].shape
        M_pred = out["ent"].shape[1]
        k = max(1, self.enc.tubelet)
        dev = out["ent"].device
        if M_pred == 0:
            z = out["ent"].sum() * 0
            return z, dict(birth_count=z, birth_cnt_f1=z, birth_real=z, birth_fields=z)
        af_idx = f_src[:, None] + torch.arange(k, device=dev)[None, :]
        n_b = b["birth_count"][:, af_idx].sum(-1)  # [B, M_pred]: births in (f_src, f_tgt]
        glob, rows, cells = out["glob"], out["ent"], out["cells"]
        rmask = b["mask"][:, f_src]
        anc_xy = torch.stack([b["x"][:, f_src], b["y"][:, f_src]], -1).float()
        anc_txy_all = (
            (anc_xy - b["center"][:, None, None].float())
            / b["half"][:, None, None].float()
        ) * rmask[..., None].float()
        cell_txy_all = self.enc.anchors(b).float() / b["half"][:, None].float()
        act_win = act[:, af_idx].flatten(2, 3)
        amask_win = b["act_mask"][:, af_idx].flatten(2, 3)
        aplaced_win = (
            b["act_placed"][:, af_idx] & b["act_mask"][:, af_idx]
        ).flatten(2, 3)
        act_txy_win = (
            b["act_xy"][:, af_idx].float().flatten(2, 3)
            / b["half"][:, None, None].float()
        ) * aplaced_win[..., None].float()

        nb_np = n_b.cpu().numpy()
        fb_np, ff_np = np.nonzero(nb_np >= 0)
        f0_val = int(f_src[0].item())
        flast_val = int(f_tgt[-1].item())
        has_unpar_np = np.zeros((B, M_pred), dtype=bool)
        if int(nb_np.max()) > 0 and len(b["birth_b"]):
            bb_np = b["birth_b"].cpu().numpy()
            bf_np = b["birth_f"].cpu().numpy()
            br_np = b["birth_r"].cpu().numpy()
            bp_np = b["birth_p"].cpu().numpy()
            birth_map = {}
            for bi_v, fi_v, ri_v, pi_v in zip(bb_np, bf_np, br_np, bp_np):
                fi_int = int(fi_v)
                if f0_val < fi_int <= flast_val:
                    b_int = int(bi_v)
                    j_step = (fi_int - 1 - f0_val) // k
                    p_int = int(pi_v)
                    birth_map.setdefault((b_int, j_step), []).append(
                        (int(ri_v), p_int, fi_int)
                    )
                    if p_int < 0:
                        has_unpar_np[b_int, j_step] = True
        else:
            birth_map = {}

        has_unpar_bm = torch.from_numpy(has_unpar_np).to(dev)
        cnt_logits = self.births.predict_count(
            glob,
            rows,
            rmask,
            acts=act_win,
            amask=amask_win,
            anc_txy=anc_txy_all,
            aplaced=aplaced_win,
            act_txy=act_txy_win,
        ).float()
        cnt_tgt = n_b.clamp(max=MAX_BIRTHS - 1)
        b_tube_w = torch.where(
            cnt_tgt == 0, 0.25, torch.where(has_unpar_bm, 2.0, 1.0)
        )
        ce_bc = F.cross_entropy(
            cnt_logits.flatten(0, 1), cnt_tgt.flatten(), reduction="none"
        )
        logs = dict(
            birth_count=(ce_bc * b_tube_w.flatten()).sum()
            / b_tube_w.sum().clamp(min=1.0)
        )
        with torch.no_grad():
            pr_cnt = cnt_logits.detach().argmax(-1)
            has_b_p = (pr_cnt > 0).float()
            has_b_t = (cnt_tgt > 0).float()
            logs["birth_cnt_f1"] = (
                2.0
                * (has_b_p * has_b_t).sum()
                / (has_b_p.sum() + has_b_t.sum()).clamp(min=1.0)
            )
            if self.record_pos and self.eval_cnts is not None and has_unpar_bm.any():
                self.eval_cnts["birth_unpar_ok"] += int(
                    ((pr_cnt > 0) & has_unpar_bm).sum().item()
                )
                self.eval_cnts["birth_unpar_tot"] += int(has_unpar_bm.sum().item())
        if not len(fb_np):
            z = out["ent"].sum() * 0
            logs["birth_real"], logs["birth_fields"] = z, z
            return logs["birth_count"], logs
        fb = torch.from_numpy(fb_np).to(dev)
        ff = torch.from_numpy(ff_np).to(dev)
        groups = {}
        for idx_i, c in enumerate(nb_np[fb_np, ff_np].tolist()):
            groups.setdefault(min(c, MAX_BIRTHS - 1), []).append(idx_i)

        total_real, total_fields, frames = 0.0, 0.0, 0
        for c, idx in groups.items():
            idx_t = torch.tensor(idx, device=dev)
            sb, sf = fb[idx_t], ff[idx_t]
            x, real, pos, parent, owner_logits, init_stat = self.births(
                glob[sb, sf],
                rows[sb, sf],
                rmask[sb, sf],
                cells[sb, sf],
                anc_txy_all[sb, sf],
                cell_txy_all[sb],
                act_win[sb, sf],
                amask_win[sb, sf],
                aplaced_win[sb, sf],
                act_txy_win[sb, sf],
                c + MARGIN,
                need_fields=bool(c),
                grad_ckpt=self.enc.grad_ckpt,
            )
            target = torch.zeros_like(real)
            if c:
                head_type = self.head.unit(x)
                vi_l, m_lens, rr_pad, par_pad, fi_pad = [], [], [], [], []
                for i, g_idx in enumerate(idx):
                    entries = birth_map.get(
                        (int(fb_np[g_idx]), int(ff_np[g_idx])), ()
                    )[:c]
                    if not entries:
                        continue
                    m_i = len(entries)
                    vi_l.append(i)
                    m_lens.append(m_i)
                    rr_row = [e[0] for e in entries] + [entries[-1][0]] * (c - m_i)
                    par_row = [e[1] for e in entries] + [entries[-1][1]] * (c - m_i)
                    fi_row = [e[2] for e in entries] + [entries[-1][2]] * (c - m_i)
                    rr_pad.append(rr_row)
                    par_pad.append(par_row)
                    fi_pad.append(fi_row)
                if vi_l:
                    vi = torch.tensor(vi_l, device=dev, dtype=torch.long)
                    vb, vf = sb[vi], sf[vi]
                    vf_tgt = f_tgt[vf]
                    rr_t = torch.tensor(rr_pad, device=dev, dtype=torch.long)
                    par_t = torch.tensor(par_pad, device=dev, dtype=torch.long)
                    fi_b_t = torch.tensor(fi_pad, device=dev, dtype=torch.long)
                    vf_obs = torch.where(
                        b["mask"][vb[:, None], vf_tgt[:, None], rr_t],
                        vf_tgt[:, None],
                        fi_b_t,
                    )
                    tt = b["type"][vb[:, None], vf_obs, rr_t].long()
                    own_t = b["owner"][vb[:, None], rr_t].clamp(0, 1).long()
                    txy = (
                        torch.stack(
                            [
                                b["x"][vb[:, None], vf_obs, rr_t],
                                b["y"][vb[:, None], vf_obs, rr_t],
                            ],
                            -1,
                        )
                        - b["center"][vb].view(-1, 1, 2)
                    ) / b["half"][vb].view(-1, 1, 2)
                    if R > 0:
                        par_valid = (
                            (par_t >= 0)
                            & (par_t < R)
                            & rmask[vb[:, None], vf[:, None], par_t.clamp(0, R - 1)]
                        )
                    else:
                        par_valid = par_t < 0
                    par_all = torch.where(par_valid, par_t, parent.shape[-1] - 1)
                    lt = F.log_softmax(head_type[vi].float(), -1)
                    lt_tt = lt.gather(
                        2, tt[:, None, :].expand(-1, c + MARGIN, -1)
                    )
                    lp = F.log_softmax(parent[vi].float(), -1)
                    lp_par = lp.gather(
                        2, par_all[:, None, :].expand(-1, c + MARGIN, -1)
                    )
                    lo = F.log_softmax(owner_logits[vi].float(), -1)
                    lo_own = lo.gather(
                        2, own_t[:, None, :].expand(-1, c + MARGIN, -1)
                    )
                    cost_np = (
                        (
                            torch.cdist(pos[vi].float(), txy, p=1)
                            - lt_tt
                            - lp_par
                            - lo_own
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    fi_l, qi_l, ti_l, inv_m_l = [], [], [], []
                    for j, m_i in enumerate(m_lens):
                        qj, tj = linear_sum_assignment(cost_np[j, :, :m_i])
                        fi_l.extend([j] * m_i)
                        qi_l.extend(qj.tolist())
                        ti_l.extend(tj.tolist())
                        inv_m_l.extend([1.0 / m_i] * m_i)
                    fi_t = torch.tensor(fi_l, device=dev, dtype=torch.long)
                    qi_t = torch.tensor(qi_l, device=dev, dtype=torch.long)
                    ti_t = torch.tensor(ti_l, device=dev, dtype=torch.long)
                    inv_m_t = torch.tensor(inv_m_l, device=dev, dtype=torch.float32)
                    orig_i = vi[fi_t]
                    target[orig_i, qi_t] = 1.0
                    rr_m = rr_t[fi_t, ti_t]
                    vf_obs_m = vf_obs[fi_t, ti_t]
                    w = torch.where(
                        b["rtype"][vb[fi_t], vf_obs_m, rr_m] == self.larva,
                        DISCOUNT,
                        1.0,
                    )  # a larva's position takes the discount, its birth doesn't
                    lpos_p = (
                        F.smooth_l1_loss(
                            pos[orig_i, qi_t].float(), txy[fi_t, ti_t], reduction="none"
                        ).sum(-1)
                        * w
                    )
                    ce_type_p = F.cross_entropy(
                        head_type[orig_i, qi_t].float(),
                        tt[fi_t, ti_t],
                        reduction="none",
                    )
                    par = par_all[fi_t, ti_t]
                    ce_par_p = F.cross_entropy(
                        parent[orig_i, qi_t].float(), par, reduction="none"
                    )
                    ce_own_p = F.cross_entropy(
                        owner_logits[orig_i, qi_t].float(),
                        own_t[fi_t, ti_t],
                        reduction="none",
                    )
                    # Light confirmation on newborn initial stats and active unit flags
                    x_m = x[orig_i, qi_t]
                    n_nb = b["num"][vb[fi_t], vf_obs_m, rr_m].float()
                    hpm_nb = n_nb[:, I["hp_max"]]
                    shm_nb = n_nb[:, I["shield_max"]]
                    stat_tgt = torch.stack(
                        [
                            (n_nb[:, I["hp"]] / hpm_nb.clamp(min=1.0)) * (hpm_nb > 0),
                            (n_nb[:, I["shield"]] / shm_nb.clamp(min=1.0))
                            * (shm_nb > 0),
                            n_nb[:, I["energy"]] / 200.0,
                            n_nb[:, I["build"]],
                        ],
                        dim=-1,
                    )
                    l_stat_p = F.smooth_l1_loss(
                        init_stat[orig_i, qi_t].float(), stat_tgt, reduction="none"
                    ).mean(-1)
                    flags_nb = n_nb[:, I["flags"]].long() | (
                        n_nb[:, I["flags2"]].long() << 8
                    )
                    act_fb = self.active_flag_bits
                    fbits_nb = ((flags_nb[:, None] >> act_fb) & 1).float()
                    l_flags_nb = F.binary_cross_entropy_with_logits(
                        self.head.flags(x_m)[:, act_fb].float(),
                        fbits_nb,
                        reduction="none",
                    ).mean(-1)
                    l_init_p = 0.25 * (l_stat_p + l_flags_nb)
                    w_bf = torch.where(~par_valid[fi_t, ti_t], 1.75, 1.0)
                    total_fields = (
                        total_fields
                        + (
                            (lpos_p + ce_type_p + ce_par_p + ce_own_p + l_init_p)
                            * w_bf
                            * inv_m_t
                        ).sum()
                    )
                    frames += len(vi_l)
            w_rl_bm = torch.where(has_unpar_bm[sb, sf], 1.5, 1.0)[:, None]
            total_real = total_real + (
                F.binary_cross_entropy_with_logits(
                    real.float(), target, reduction="none"
                )
                * w_rl_bm
            ).sum()
        logs["birth_real"] = total_real / max(1, len(fb_np)) / (1 + MARGIN)
        logs["birth_fields"] = (
            total_fields / max(1, frames) if frames else out["ent"].sum() * 0
        )
        return logs["birth_count"] + logs["birth_real"] + logs["birth_fields"], logs

    @staticmethod
    def _group_eff_spawns(ty_arr, ow_arr, vw_arr, xy_arr, age_arr):
        """Group newly spawned per-view effect entries (0 <= age < k) at one (b, m)
        target frame into canonical AoE objects (collapsing contiguous collinear
        LurkerSpines / ThermalLances ray dots and merging across views 0 & 1)."""
        by_v = {0: [], 1: []}
        for ty, ow, vw, (x, y), ag in zip(ty_arr, ow_arr, vw_arr, xy_arr, age_arr):
            by_v[1 if vw == 1 else 0].append(
                (int(ty), int(ow), float(x), float(y), int(round(ag)))
            )
        per_v_objs = []
        for vw, rows in by_v.items():
            i, n = 0, len(rows)
            while i < n:
                ty, ow, x0, y0, ag = rows[i]
                if ty in (5, 12):
                    exp_step = 0.25 if ty == 5 else 1.0
                    j = i + 1
                    dx0, dy0 = 0.0, 0.0
                    while (
                        j < n
                        and (j - i) < MAX_EFF_DOTS
                        and rows[j][0] == ty
                        and rows[j][1] == ow
                        and abs(rows[j][4] - ag) <= 1
                    ):
                        dx = rows[j][2] - rows[j - 1][2]
                        dy = rows[j][3] - rows[j - 1][3]
                        step = math.hypot(dx, dy)
                        if abs(step - exp_step) > 0.05:
                            break
                        if j == i + 1:
                            dx0, dy0 = dx, dy
                        elif abs(dx - dx0) > 0.03 or abs(dy - dy0) > 0.03:
                            break
                        j += 1
                    x1, y1 = rows[j - 1][2], rows[j - 1][3]
                    per_v_objs.append((ty, ow, j - i, x0, y0, x1, y1, ag, vw))
                    i = j
                else:
                    per_v_objs.append((ty, ow, 1, x0, y0, x0, y0, ag, vw))
                    i += 1
        merged = {}
        for ty, ow, nd, x0, y0, x1, y1, ag, vw in per_v_objs:
            key = (ty, ow, nd, round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2))
            rec = merged.get(key)
            if rec is None:
                merged[key] = [ty, ow, nd, x0, y0, x1, y1, ag, [0.0, 0.0]]
                merged[key][8][vw] = 1.0
            else:
                rec[8][vw] = 1.0
                if ag < rec[7]:
                    rec[7] = ag
        return list(merged.values())

    def effect_losses(self, out, act, b, f_src, f_tgt):
        """Discrete AoE effect transitions across (f_src, f_tgt]:
        1) EffPersistHead on active f_src effect tokens (persistence/expiry + moving dxy).
        2) EffSpawnHead on newly spawned AoE circles & rays (0 <= age_tgt < k), attending
           over active f_src precursor effects alongside rows, cells, globals & actions."""
        B, F_, R = b["mask"].shape
        M_pred = out["ent"].shape[1]
        k = max(1, self.enc.tubelet)
        dev = out["ent"].device
        if M_pred == 0:
            z = out["ent"].sum() * 0
            return z, dict(
                eff_persist=z,
                eff_persist_f1=z,
                eff_expire_f1=z,
                eff_spawn_cnt=z,
                eff_spawn_cnt_f1=z,
                eff_spawn_real=z,
                eff_spawn_fields=z,
            )

        # 1) Continuing / expiring active effect tokens from f_src -> f_tgt
        src_slot = out["eff_src_slot"]
        bi_s, mi_s, ui_s = torch.nonzero(src_slot >= 0, as_tuple=True)
        n_act = int(bi_s.numel())
        pos_scale = POS_SCALE * k
        pers_tp = pers_pred = pers_gt = exp_tp = exp_pred = exp_gt = 0

        if n_act > 0:
            e_act = out["eff"][bi_s, mi_s, ui_s]
            xs_slot = src_slot[bi_s, mi_s, ui_s]
            fs_act = f_src[mi_s]
            ty_s_t = b["eff_type"][bi_s, fs_act, xs_slot]
            ag_s_t = b["eff_age"][bi_s, fs_act, xs_slot].float()
            p_logit, p_dxy = self.eff_persist(e_act, ty_s=ty_s_t, age_s=ag_s_t)

            bm_s_np = (bi_s * M_pred + mi_s).cpu().numpy()
            ty_s_np = ty_s_t.cpu().numpy()
            ow_s_np = b["eff_owner"][bi_s, fs_act, xs_slot].cpu().numpy()
            vw_s_np = (
                b["eff_view"][bi_s, fs_act, xs_slot].cpu().numpy()
                if "eff_view" in b
                else np.zeros(n_act, dtype=np.int64)
            )
            ag_s_np = np.rint(ag_s_t.cpu().numpy()).astype(np.int64)
            xy_s_t = b["eff_xy"][bi_s, fs_act, xs_slot].float()
            xy_s_np = xy_s_t.cpu().numpy()

            buckets = {}
            for idx_s in range(n_act):
                bkey = (
                    int(bm_s_np[idx_s]),
                    int(vw_s_np[idx_s]),
                    int(ow_s_np[idx_s]),
                    int(ty_s_np[idx_s]),
                    int(ag_s_np[idx_s]),
                )
                buckets.setdefault(bkey, []).append(idx_s)

            tgt_persist_np = np.zeros(n_act, dtype=np.float32)
            tgt_xy_np = xy_s_np.copy()

            em_tgt = b["eff_mask"][:, f_tgt]
            ea_tgt = b["eff_age"][:, f_tgt]
            bi_c, mi_c, xi_c = torch.nonzero(em_tgt & (ea_tgt >= k), as_tuple=True)
            if len(bi_c):
                ft_c = f_tgt[mi_c]
                bm_c_np = (bi_c * M_pred + mi_c).cpu().numpy()
                ty_c_np = b["eff_type"][bi_c, ft_c, xi_c].cpu().numpy()
                ow_c_np = b["eff_owner"][bi_c, ft_c, xi_c].cpu().numpy()
                vw_c_np = (
                    b["eff_view"][bi_c, ft_c, xi_c].cpu().numpy()
                    if "eff_view" in b
                    else np.zeros(len(bi_c), dtype=np.int64)
                )
                ag_c_np = (
                    np.rint(b["eff_age"][bi_c, ft_c, xi_c].cpu().numpy()).astype(np.int64)
                    - k
                )
                xy_c_np = b["eff_xy"][bi_c, ft_c, xi_c].float().cpu().numpy()

                for jc in range(len(bm_c_np)):
                    bkey = (
                        int(bm_c_np[jc]),
                        int(vw_c_np[jc]),
                        int(ow_c_np[jc]),
                        int(ty_c_np[jc]),
                        int(ag_c_np[jc]),
                    )
                    cand = buckets.get(bkey)
                    if not cand:
                        continue
                    xt, yt = float(xy_c_np[jc, 0]), float(xy_c_np[jc, 1])
                    best_pos, best_d = -1, 6.0
                    for pos_c, idx_s in enumerate(cand):
                        d_m = abs(float(xy_s_np[idx_s, 0]) - xt) + abs(
                            float(xy_s_np[idx_s, 1]) - yt
                        )
                        if d_m <= best_d:
                            best_d, best_pos = d_m, pos_c
                    if best_pos >= 0:
                        matched_s = cand.pop(best_pos)
                        tgt_persist_np[matched_s] = 1.0
                        tgt_xy_np[matched_s] = xy_c_np[jc]

            tgt_persist = torch.from_numpy(tgt_persist_np).to(dev)
            tgt_xy = torch.from_numpy(tgt_xy_np).to(dev)
            is_exp = tgt_persist < 0.5
            bce_elem = F.binary_cross_entropy_with_logits(
                p_logit.float(), tgt_persist, reduction="none"
            )
            denom_exp = is_exp.float().sum().clamp(min=max(4.0, 0.10 * n_act))
            bce_p = 0.15 * (bce_elem * torch.where(is_exp, 2.0, 1.0)).sum() / denom_exp
            m_pers = tgt_persist > 0.5
            if m_pers.any():
                tgt_dxy = (tgt_xy[m_pers] - xy_s_t[m_pers]) / pos_scale
                l_edxy = F.smooth_l1_loss(
                    p_dxy[m_pers].float(), tgt_dxy, beta=0.02, reduction="none"
                ).sum(-1).sum() / max(4, int(m_pers.sum().item()))
            else:
                l_edxy = p_dxy.sum() * 0.0
            l_eff_persist = bce_p + 2.0 * l_edxy

            with torch.no_grad():
                pr_p = (p_logit.detach() > 0.0).float()
                pers_tp = int((pr_p * tgt_persist).sum().item())
                pers_pred = int(pr_p.sum().item())
                pers_gt = int(tgt_persist.sum().item())
                pr_exp = 1.0 - pr_p
                tg_exp = 1.0 - tgt_persist
                exp_tp = int((pr_exp * tg_exp).sum().item())
                exp_pred = int(pr_exp.sum().item())
                exp_gt = int(tg_exp.sum().item())
                eff_persist_f1 = torch.tensor(
                    2.0 * pers_tp / max(1, pers_pred + pers_gt), device=dev
                )
                eff_expire_f1 = torch.tensor(
                    2.0 * exp_tp / max(1, exp_pred + exp_gt), device=dev
                )
        else:
            l_eff_persist = (
                out["eff"].sum()
                + sum(p_.sum() for p_ in self.eff_persist.parameters())
            ) * 0.0
            eff_persist_f1 = torch.zeros((), device=dev)
            eff_expire_f1 = torch.zeros((), device=dev)

        # 2) Newly spawned AoE circles & collinear rays (0 <= age_tgt < k at f_tgt)
        em_tgt = b["eff_mask"][:, f_tgt]
        ea_tgt = b["eff_age"][:, f_tgt]
        bi_n, mi_n, xi_n = torch.nonzero(em_tgt & (ea_tgt < k), as_tuple=True)
        n_sp_np = np.zeros((B, M_pred), dtype=np.int64)
        has_spell_np = np.zeros((B, M_pred), dtype=bool)
        spawn_map = {}
        if len(bi_n):
            ft_n = f_tgt[mi_n]
            bi_n_np = bi_n.cpu().numpy()
            mi_n_np = mi_n.cpu().numpy()
            ty_n_np = b["eff_type"][bi_n, ft_n, xi_n].cpu().numpy()
            ow_n_np = b["eff_owner"][bi_n, ft_n, xi_n].cpu().numpy()
            vw_n_np = (
                b["eff_view"][bi_n, ft_n, xi_n].cpu().numpy()
                if "eff_view" in b
                else np.zeros(len(bi_n), dtype=np.int64)
            )
            xy_n_np = b["eff_xy"][bi_n, ft_n, xi_n].float().cpu().numpy()
            ag_n_np = ea_tgt[bi_n, mi_n, xi_n].float().cpu().numpy()

            raw_bm = collections.defaultdict(lambda: ([], [], [], [], []))
            for idx_e in range(len(bi_n_np)):
                g_l = raw_bm[(int(bi_n_np[idx_e]), int(mi_n_np[idx_e]))]
                g_l[0].append(ty_n_np[idx_e])
                g_l[1].append(ow_n_np[idx_e])
                g_l[2].append(vw_n_np[idx_e])
                g_l[3].append(xy_n_np[idx_e])
                g_l[4].append(ag_n_np[idx_e])
            for (b_v, m_v), (tys, ows, vws, xys, ags) in raw_bm.items():
                objs = self._group_eff_spawns(tys, ows, vws, xys, ags)
                if objs:
                    spawn_map[(b_v, m_v)] = objs
                    n_sp_np[b_v, m_v] = len(objs)
                    if any(o_[0] not in (5, 12) for o_ in objs):
                        has_spell_np[b_v, m_v] = True

        n_sp = torch.from_numpy(n_sp_np).to(dev)
        has_spell_bm = torch.from_numpy(has_spell_np).to(dev)
        af_idx = f_src[:, None] + torch.arange(k, device=dev)[None, :]
        glob, rows, cells = out["glob"], out["ent"], out["cells"]
        rmask = b["mask"][:, f_src]
        anc_xy = torch.stack([b["x"][:, f_src], b["y"][:, f_src]], -1).float()
        anc_txy_all = (
            (anc_xy - b["center"][:, None, None].float())
            / b["half"][:, None, None].float()
        ) * rmask[..., None].float()
        cell_txy_all = self.enc.anchors(b).float() / b["half"][:, None].float()
        act_win = act[:, af_idx].flatten(2, 3)
        amask_win = b["act_mask"][:, af_idx].flatten(2, 3)
        aplaced_win = (
            b["act_placed"][:, af_idx] & b["act_mask"][:, af_idx]
        ).flatten(2, 3)
        act_txy_win = (
            b["act_xy"][:, af_idx].float().flatten(2, 3)
            / b["half"][:, None, None].float()
        ) * aplaced_win[..., None].float()

        effs_all = out["eff"]
        emask_all = src_slot >= 0
        X_e = src_slot.shape[2]
        if X_e > 0 and b["eff_mask"].shape[2] > 0:
            sl_c = src_slot.clamp(min=0)
            bi_e = torch.arange(B, device=dev)[:, None, None].expand(B, M_pred, X_e)
            fs_e = f_src[None, :, None].expand(B, M_pred, X_e)
            eff_xy_raw = b["eff_xy"][bi_e, fs_e, sl_c].float() * emask_all[
                ..., None
            ].float()
            eff_txy_all = eff_xy_raw / b["half"][:, None, None].float()
            eff_ty_all = (
                b["eff_type"][bi_e, fs_e, sl_c]
                .long()
                .clamp(0, N_EFF_SPAWN_TYPES - 1)
                * emask_all.long()
            )
            eff_age_all = b["eff_age"][bi_e, fs_e, sl_c].float() * emask_all.float()
        else:
            eff_txy_all = torch.zeros(B, M_pred, X_e, 2, device=dev)
            eff_ty_all = torch.zeros(B, M_pred, X_e, dtype=torch.long, device=dev)
            eff_age_all = torch.zeros(B, M_pred, X_e, device=dev)

        cnt_logits = self.eff_spawns.predict_count(
            glob,
            rows,
            rmask,
            acts=act_win,
            amask=amask_win,
            anc_txy=anc_txy_all,
            aplaced=aplaced_win,
            act_txy=act_txy_win,
            effs=effs_all,
            emask=emask_all,
            eff_txy=eff_txy_all,
            eff_ty=eff_ty_all,
            eff_age=eff_age_all,
        ).float()
        cnt_tgt = n_sp.clamp(max=MAX_EFF_SPAWNS - 1)
        sp_tube_w = torch.where(
            cnt_tgt == 0, 0.20, torch.where(has_spell_bm, 1.80, 0.50)
        )
        ce_sc = F.cross_entropy(
            cnt_logits.flatten(0, 1), cnt_tgt.flatten(), reduction="none"
        )
        l_sp_cnt = (ce_sc * sp_tube_w.flatten()).sum() / sp_tube_w.sum().clamp(min=1.0)
        with torch.no_grad():
            pr_sp_cnt = cnt_logits.detach().argmax(-1)
            has_sp_p = (pr_sp_cnt > 0).float()
            has_sp_t = (cnt_tgt > 0).float()
            sp_cnt_tp = int((has_sp_p * has_sp_t).sum().item())
            sp_cnt_pred = int(has_sp_p.sum().item())
            sp_cnt_gt = int(has_sp_t.sum().item())
            eff_spawn_cnt_f1 = torch.tensor(
                2.0 * sp_cnt_tp / max(1, sp_cnt_pred + sp_cnt_gt), device=dev
            )
            sp_spell_tube_ok = int(((pr_sp_cnt > 0) & has_spell_bm).sum().item())
            sp_spell_tube_tot = int(has_spell_bm.sum().item())

        fb_np, ff_np = np.nonzero(n_sp_np >= 0)
        fb = torch.from_numpy(fb_np).to(dev)
        ff = torch.from_numpy(ff_np).to(dev)
        groups = {}
        for idx_i, c in enumerate(n_sp_np[fb_np, ff_np].tolist()):
            groups.setdefault(min(c, MAX_EFF_SPAWNS - 1), []).append(idx_i)

        total_real, total_fields, frames = 0.0, 0.0, 0
        sp_ty_ok, sp_ty_tot = 0, 0
        sp_spell_ty_ok, sp_spell_ty_tot = 0, 0
        for c, idx in groups.items():
            idx_t = torch.tensor(idx, device=dev)
            sb, sf = fb[idx_t], ff[idx_t]
            (
                _,
                real,
                start_pos,
                ray_dxy,
                eff_type_logits,
                owner_logits,
                view_logits,
                ndots_logits,
                age_logits,
            ) = self.eff_spawns(
                glob[sb, sf],
                rows[sb, sf],
                rmask[sb, sf],
                cells[sb, sf],
                anc_txy_all[sb, sf],
                cell_txy_all[sb],
                act_win[sb, sf],
                amask_win[sb, sf],
                aplaced_win[sb, sf],
                act_txy_win[sb, sf],
                c + EFF_MARGIN,
                effs=effs_all[sb, sf],
                emask=emask_all[sb, sf],
                eff_txy=eff_txy_all[sb, sf],
                eff_ty=eff_ty_all[sb, sf],
                eff_age=eff_age_all[sb, sf],
                need_fields=bool(c),
                grad_ckpt=self.enc.grad_ckpt,
            )
            target = torch.zeros_like(real)
            if c:
                vi_l, m_lens = [], []
                ty_pad, ow_pad, vw_pad, nd_pad, ag_pad, st_pad, ry_pad = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for i_local, g_idx in enumerate(idx):
                    entries = spawn_map.get(
                        (int(fb_np[g_idx]), int(ff_np[g_idx])), ()
                    )[:c]
                    if not entries:
                        continue
                    m_i = len(entries)
                    vi_l.append(i_local)
                    m_lens.append(m_i)
                    padded = list(entries) + [entries[-1]] * (c - m_i)
                    ty_pad.append(
                        [min(max(e[0], 0), N_EFF_SPAWN_TYPES - 1) for e in padded]
                    )
                    ow_pad.append([min(max(e[1], 0), 1) for e in padded])
                    nd_pad.append(
                        [min(max(e[2] - 1, 0), MAX_EFF_DOTS - 1) for e in padded]
                    )
                    st_pad.append([[e[3], e[4]] for e in padded])
                    ry_pad.append(
                        [[(e[5] - e[3]) / 10.0, (e[6] - e[4]) / 10.0] for e in padded]
                    )
                    ag_pad.append([min(max(e[7], 0), MAX_EFF_AGE - 1) for e in padded])
                    vw_pad.append([e[8] for e in padded])
                if vi_l:
                    vi = torch.tensor(vi_l, device=dev, dtype=torch.long)
                    vb = sb[vi]
                    ty_t = torch.tensor(ty_pad, device=dev, dtype=torch.long)
                    ow_t = torch.tensor(ow_pad, device=dev, dtype=torch.long)
                    nd_t = torch.tensor(nd_pad, device=dev, dtype=torch.long)
                    ag_t = torch.tensor(ag_pad, device=dev, dtype=torch.long)
                    vw_t = torch.tensor(vw_pad, device=dev, dtype=torch.float32)
                    st_txy = (
                        torch.tensor(st_pad, device=dev, dtype=torch.float32)
                        / b["half"][vb].view(-1, 1, 2).float()
                    )
                    ry_t = torch.tensor(ry_pad, device=dev, dtype=torch.float32)

                    lt = F.log_softmax(eff_type_logits[vi].float(), -1).gather(
                        2, ty_t[:, None, :].expand(-1, c + EFF_MARGIN, -1)
                    )
                    lo = F.log_softmax(owner_logits[vi].float(), -1).gather(
                        2, ow_t[:, None, :].expand(-1, c + EFF_MARGIN, -1)
                    )
                    cost_np = (
                        (
                            torch.cdist(start_pos[vi].float(), st_txy, p=1)
                            - lt
                            - lo
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    fi_l, qi_l, ti_l, inv_m_l = [], [], [], []
                    for j_m, m_i in enumerate(m_lens):
                        qj, tj = linear_sum_assignment(cost_np[j_m, :, :m_i])
                        fi_l.extend([j_m] * m_i)
                        qi_l.extend(qj.tolist())
                        ti_l.extend(tj.tolist())
                        inv_m_l.extend([1.0 / m_i] * m_i)
                    fi_t = torch.tensor(fi_l, device=dev, dtype=torch.long)
                    qi_t = torch.tensor(qi_l, device=dev, dtype=torch.long)
                    ti_t = torch.tensor(ti_l, device=dev, dtype=torch.long)
                    inv_m_t = torch.tensor(inv_m_l, device=dev, dtype=torch.float32)
                    orig_i = vi[fi_t]
                    target[orig_i, qi_t] = 1.0

                    ty_m = ty_t[fi_t, ti_t]
                    nd_m = nd_t[fi_t, ti_t]
                    is_ray = (ty_m == 5) | (ty_m == 12)
                    w_sp = torch.where(is_ray, 0.35, 2.0)
                    l_start = (
                        4.0
                        * F.smooth_l1_loss(
                            start_pos[orig_i, qi_t].float(),
                            st_txy[fi_t, ti_t],
                            beta=0.02,
                            reduction="none",
                        ).sum(-1)
                    )
                    w_ray = torch.where(nd_m > 0, 1.0, 0.1)
                    l_ray = w_ray * F.smooth_l1_loss(
                        ray_dxy[orig_i, qi_t].float(),
                        ry_t[fi_t, ti_t],
                        beta=0.02,
                        reduction="none",
                    ).sum(-1)
                    ce_ty = F.cross_entropy(
                        eff_type_logits[orig_i, qi_t].float(), ty_m, reduction="none"
                    )
                    ce_ow = F.cross_entropy(
                        owner_logits[orig_i, qi_t].float(),
                        ow_t[fi_t, ti_t],
                        reduction="none",
                    )
                    bce_vw = F.binary_cross_entropy_with_logits(
                        view_logits[orig_i, qi_t].float(),
                        vw_t[fi_t, ti_t],
                        reduction="none",
                    ).mean(-1)
                    ce_nd = F.cross_entropy(
                        ndots_logits[orig_i, qi_t].float(), nd_m, reduction="none"
                    )
                    ce_ag = 0.25 * F.cross_entropy(
                        age_logits[orig_i, qi_t].float(),
                        ag_t[fi_t, ti_t],
                        reduction="none",
                    )
                    total_fields = (
                        total_fields
                        + (
                            (
                                l_start
                                + l_ray
                                + ce_ty
                                + ce_ow
                                + bce_vw
                                + ce_nd
                                + ce_ag
                            )
                            * w_sp
                            * inv_m_t
                        ).sum()
                    )
                    frames += len(vi_l)
                    with torch.no_grad():
                        pr_ty = eff_type_logits[orig_i, qi_t].detach().argmax(-1)
                        sp_ty_ok += int((pr_ty == ty_m).sum().item())
                        sp_ty_tot += int(ty_m.numel())
                        sp_m = ~is_ray
                        if sp_m.any():
                            sp_spell_ty_ok += int(
                                (pr_ty[sp_m] == ty_m[sp_m]).sum().item()
                            )
                            sp_spell_ty_tot += int(sp_m.sum().item())
            w_rl_sp = torch.where(has_spell_bm[sb, sf], 2.0, 1.0)[:, None]
            total_real = total_real + (
                F.binary_cross_entropy_with_logits(
                    real.float(), target, reduction="none"
                )
                * w_rl_sp
            ).sum()

        l_sp_real = total_real / max(1, len(fb_np)) / (1 + EFF_MARGIN)
        l_sp_fields = (
            total_fields / max(1, frames)
            if frames
            else (
                out["ent"].sum()
                + sum(p_.sum() for p_ in self.eff_spawns.parameters())
            )
            * 0.0
        )
        if self.record_pos and self.eval_cnts is not None:
            self.eval_cnts.update(
                dict(
                    pers_tp=pers_tp,
                    pers_pred=pers_pred,
                    pers_gt=pers_gt,
                    exp_tp=exp_tp,
                    exp_pred=exp_pred,
                    exp_gt=exp_gt,
                    sp_cnt_tp=sp_cnt_tp,
                    sp_cnt_pred=sp_cnt_pred,
                    sp_cnt_gt=sp_cnt_gt,
                    sp_ty_ok=sp_ty_ok,
                    sp_ty_tot=sp_ty_tot,
                    sp_spell_tube_ok=sp_spell_tube_ok,
                    sp_spell_tube_tot=sp_spell_tube_tot,
                    sp_spell_ty_ok=sp_spell_ty_ok,
                    sp_spell_ty_tot=sp_spell_ty_tot,
                )
            )
        logs = dict(
            eff_persist=l_eff_persist,
            eff_persist_f1=eff_persist_f1,
            eff_expire_f1=eff_expire_f1,
            eff_spawn_cnt=l_sp_cnt,
            eff_spawn_cnt_f1=eff_spawn_cnt_f1,
            eff_spawn_real=l_sp_real,
            eff_spawn_fields=l_sp_fields,
        )
        total_eff = l_eff_persist + l_sp_cnt + l_sp_real + 0.15 * l_sp_fields
        return total_eff, logs

    def map_loss(self, nxt_cells, clean_cells, b, f_src, f_tgt, K=256):
        """Next-tubelet visibility, seen and creep at f_tgt on nxt_cells with f_src
        residual + unbiased Horvitz-Thompson boundary/static sampling, plus a detached
        (pathable, buildable) probe at f_src on clean_cells.detach()."""
        B, F_ = b["mask"].shape[:2]
        M_pred, S = nxt_cells.shape[1:3]
        dev = nxt_cells.device
        if M_pred == 0:
            z = nxt_cells.sum() * 0
            return z, z, dict(map=z, map_chg_f1=z, probe_pb=z, probe_pb_f1=z)
        BM = B * M_pred
        side = int(round(math.sqrt(S)))
        r = TOP // side
        cells = nxt_cells.reshape(BM, S, -1)
        side_bm = b["side"][:, None].expand(B, M_pred).reshape(-1, 1).float()
        size = side_bm / TOP

        src_all = self._unpack_dyn_map(b["map"][:, f_src], BM, dev)
        tgt_all = self._unpack_dyn_map(b["map"][:, f_tgt], BM, dev)
        chg_cell = (tgt_all != src_all).any(-1)  # [BM, 4096]
        n_chg_cell = chg_cell.float().sum(1, keepdim=True)  # [BM, 1]
        n_stay_cell = float(TOP * TOP) - n_chg_cell

        k_chg, k_stay = 128, 256
        idx_chg = (
            torch.where(chg_cell, 2.0, 0.0) + torch.rand((BM, TOP * TOP), device=dev)
        ).topk(k_chg, dim=1).indices
        idx_stay = (
            torch.where(~chg_cell, 2.0, -2.0) + torch.rand((BM, TOP * TOP), device=dev)
        ).topk(k_stay, dim=1).indices
        is_c = chg_cell.gather(1, idx_chg)
        is_s = ~chg_cell.gather(1, idx_stay)
        w_c = is_c.float() * (n_chg_cell / float(k_chg)).clamp(min=1.0)
        w_s = is_s.float() * (n_stay_cell / float(k_stay)).clamp(min=1.0)

        idx_ms = torch.cat([idx_chg, idx_stay], dim=1)
        w_cell = torch.cat([w_c, w_s], dim=1)
        iy_m, ix_m = idx_ms // TOP, idx_ms % TOP
        cell_m = (iy_m // r) * side + (ix_m // r)
        off_m = torch.stack(
            [(ix_m % r + 0.5 - r / 2) * size, (iy_m % r + 0.5 - r / 2) * size], -1
        )
        sg = torch.gather(src_all, 1, idx_ms[..., None].expand(-1, -1, C_DYNAMIC))
        tg = torch.gather(tgt_all, 1, idx_ms[..., None].expand(-1, -1, C_DYNAMIC))
        o = self.predict_map(cells, cell_m, off_m, sg)

        chg_m = (tg - sg).abs() > 0.5
        bce_m = F.binary_cross_entropy_with_logits(o, tg, reduction="none")
        w_elem = w_cell[..., None] * torch.where(chg_m, 3.0, 1.0)
        min_chg_m = max(32.0, 0.002 * float(BM * TOP * TOP))
        n_chg_c = (w_cell[..., None] * chg_m.float()).sum((0, 1)).clamp(min=min_chg_m)
        mp = 0.06 * ((bce_m * w_elem).sum((0, 1)) / n_chg_c).mean()
        with torch.no_grad():
            pr_b = (o.detach() > 0.0).float()
            pr_fl = (pr_b != sg).float()
            gt_fl = chg_m.float()
            tp_fl = (
                w_cell[..., None] * pr_fl * gt_fl * (pr_b == tg).float()
            ).sum()
            pd_fl = (w_cell[..., None] * pr_fl).sum()
            gd_fl = (w_cell[..., None] * gt_fl).sum()
            map_chg_f1 = 2.0 * tp_fl / (pd_fl + gd_fl).clamp(min=1.0)
            if self.record_pos and self.eval_cnts is not None:
                self.eval_cnts["map_chg_tp"] += int(round(float(tp_fl.item())))
                self.eval_cnts["map_chg_pred"] += int(round(float(pd_fl.item())))
                self.eval_cnts["map_chg_gt"] += int(round(float(gd_fl.item())))

        # Detached probe: [p0_path, p0_build, p1_path, p1_build] at f_src on uniform K
        idx = torch.randint(0, TOP * TOP, (BM, K), device=dev)
        iy, ix = idx // TOP, idx % TOP
        cell = (iy // r) * side + (ix // r)
        off = torch.stack(
            [(ix % r + 0.5 - r / 2) * size, (iy % r + 0.5 - r / 2) * size], -1
        )
        clean_det = clean_cells.detach().reshape(BM, S, -1)
        pb_logits = self.pb_probe(clean_det, cell, off).float()
        ctr_bm = b["center"][:, None].expand(B, M_pred, 2).reshape(-1, 2).float()
        org_bm = b["origin"][:, None].expand(B, M_pred, 2).reshape(-1, 2).float()
        wx = torch.floor(
            ctr_bm[:, 0:1] + (ix.float() + 0.5) * size - 0.5 * side_bm
        ).long()
        wy = torch.floor(
            ctr_bm[:, 1:2] + (iy.float() + 0.5) * size - 0.5 * side_bm
        ).long()
        gx = wx - torch.round(org_bm[:, 0:1]).long()
        gy = wy - torch.round(org_bm[:, 1:2]).long()
        gh, gw = b["static_grid"].shape[1:3]
        in_b = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
        gx_c, gy_c = gx.clamp(0, gw - 1), gy.clamp(0, gh - 1)
        bi_bm = torch.arange(B, device=dev)[:, None].expand(B, M_pred).reshape(-1, 1)
        fs_bm = f_src[None, :].expand(B, M_pred).reshape(-1, 1)
        playable = in_b & (b["static_grid"][bi_bm, gy_c, gx_c, 2] > 0)
        if b["grid"].ndim == 4:
            g_val = b["grid"][bi_bm, fs_bm, gy_c, gx_c].to(torch.int32)
            pb_shifts = torch.tensor([3, 4, 8, 9], device=dev, dtype=torch.int32)
            pb_tgt = ((g_val[..., None] >> pb_shifts) & 1).float()
        else:
            pb_tgt = b["grid"][bi_bm, fs_bm, gy_c, gx_c][..., (3, 4, 8, 9)].float()
        pb_elem = F.binary_cross_entropy_with_logits(
            pb_logits, pb_tgt, reduction="none"
        ).mean(-1)
        probe_pb = (pb_elem * playable.float()).sum() / playable.float().sum().clamp(
            min=1.0
        )
        with torch.no_grad():
            pb_p = (pb_logits.detach() > 0).float() * playable[..., None].float()
            pb_t = pb_tgt * playable[..., None].float()
            probe_pb_f1 = (
                2.0 * (pb_p * pb_t).sum() / (pb_p.sum() + pb_t.sum()).clamp(min=1.0)
            )
        logs = dict(
            map=mp,
            map_chg_f1=map_chg_f1,
            probe_pb=probe_pb.detach(),
            probe_pb_f1=probe_pb_f1,
        )
        return mp, probe_pb, logs
