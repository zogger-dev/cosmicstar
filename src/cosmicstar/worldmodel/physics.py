"""The physics model (DESIGN_physics.md): a single pass over a whole sequence predicts the next state at every frame, as a language
model does. No horizon input, no motion input.

    per frame        entities, action tokens, globals and 64 x 64 map tokens, each map token reading its PATCH x PATCH real cells
    each level       64, 32, 16: conv blocks on the map, then repeated (timeline, state attention, action attention)
      timeline       entities along their rows, cells too at 32 and 16, globals, each reading earlier frames' action-updated tokens; each player's actions: causal
      state          4 x 4 windows of cells and entities, reading the globals; globals read everything. No action of this frame: this is the clean state
      action         the same with this frame's actions: targeted ones in the windows, the others sharing a window. Carried on as a delta on the state
    outputs          a residual per token, s + g(s), trained against an EMA teacher's next token with Barlow Twins, plus the raw heads

At inference, frame by frame: state at t -> action predictor -> actions at t -> updated tokens -> t + 1. Training runs the same dependencies in one pass with the
real actions: a clean state at t holds no action from t, and the target is the teacher's clean state at t + 1.

An action points at its target through a learned bias on the attention score of that pair, one per direction. Nothing is copied
from the target, so an entity knows nothing about being targeted until attention tells it."""

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

from .action_encoder import LoopTime
from .action_tokens import C_ACTION, N_ACT_KINDS, N_GROUP_OPS, N_GROUPS, N_REMOVE, N_SUBGROUPS
from .dense import Down, ResBlock, WIN
from .direct import RoPE2D
from .encoder import mlp
from .grid_latent import MapDecoder
from .physics_data import C_DYNAMIC, C_STATIC, NEUTRAL, PATCH, PROD, QUEUE, RAW_NUMS, TOP
from .pyramid import _rot
from . import quad_latent
from .quad_latent import SLIDE, window_of, window_sets
from .tokenize import C_GROUPS, N_BUFF_IDS, N_EFFECT_IDS, N_RAW_ABILITY_IDS, PC_KEYS

quad_latent.MAX_SCORES = MAX_SCORES = 1.9e7   # attention scores per head in one FlexAttention call: larger ones fail in the driver (DESIGN_physics.md)
I = {n: i for i, n in enumerate(RAW_NUMS)}
C_PLY = 13                       # player numbers: PC_KEYS and the camera's x, y, known
ROW_NUMS = len(RAW_NUMS) + 8     # embedded numbers per row, see row_numbers
N_FLAGS = 16                     # flags, flags2 bits
MAX_BIRTHS, MARGIN = 64, 2       # birth count classes; extra queries per frame
POS_SCALE = 0.29                 # a fast unit's per-loop move, map units
POS_BOOST = 4.0
DISCOUNT = 0.05                  # critters' whole loss and larvae's position


def row_numbers(b):
    """[B, F, R, ROW_NUMS] scaled numbers of the rows."""
    n = b["num"]
    g = lambda k: n[..., I[k]]
    hpm, shm, bm = g("hp_max"), g("shield_max"), g("buff_max")
    log = lambda x, s=8.0: torch.log1p(x.clamp(min=0)) / s
    out = [(g("hp") / hpm.clamp(min=1)) * (hpm > 0), (g("shield") / shm.clamp(min=1)) * (shm > 0), g("energy") / 200, log(hpm), log(shm), g("build"),
           log(g("contents")), g("harvesters") / 16, g("ideal_harvesters") / 16, g("facing").sin(), g("facing").cos(), g("radius") / 4,
           g("weapon_cooldown").clamp(min=0) / 32, g("cargo_used") / 8, g("cargo_max") / 8, (g("buff_remain") / bm.clamp(min=1)) * (bm > 0), log(bm),
           g("detect_range") / 12, g("radar_range") / 32, g("atk_up") / 3, g("arm_up") / 3, g("shd_up") / 3, g("add_on") >= 0, g("engaged") >= 0, g("z") / 16]
    out += [(g("display") == k) for k in (1, 2, 3)] + [(g("cloak") == k) for k in (0, 1, 2, 3)]
    out += [b["inside"] == 1, b["inside"] == 2, ~b["alive"], b["parent_row"] >= 0]
    return torch.stack([x.to(n.dtype) for x in out], -1)


def attend_pairs(q, k, v, mask, causal=False, code=None, rowmap=None, bias=None):
    """One softmax per query over the keys mask [B, K] allows. causal: only keys at or before the query. code [P, Q, K] and rowmap [B] add bias [3, H] to a
    pair's score. FlexAttention when training; without gradient SDPA, since FlexAttention's no-gradient path fails to compile (quad_latent.attend)."""
    if not torch.is_grad_enabled():
        Q, K = q.shape[2], k.shape[2]
        allow = mask[:, None, None]
        if causal:
            allow = allow & torch.tril(torch.ones(Q, K, dtype=torch.bool, device=mask.device))
        r = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
        if code is not None:
            rows = torch.nonzero(rowmap >= 0, as_tuple=True)[0]
            add = bias[code[rowmap[rows]].long()].permute(0, 3, 1, 2).to(q.dtype)      # [P, H, Q, K]
            am = add.masked_fill(~allow[rows], -float("inf"))
            r[rows] = F.scaled_dot_product_attention(q[rows], k[rows], v[rows], attn_mask=am)
        return r

    def score_mod(score, b, hh, qi, ki):
        s = torch.where(mask[b, ki], score, -float("inf"))
        if causal:
            s = torch.where(ki <= qi, s, -float("inf"))
        if code is not None:
            r = rowmap[b]
            s = s + torch.where(r >= 0, bias[code[r.clamp(min=0), qi, ki].long(), hh], 0.0)
        return s
    return flex_attention(q, k, v, score_mod=score_mod)


class TimeAttend(nn.Module):
    """One chunk of a timeline's causal attention: the part that compiles."""

    def forward(self, q, k, v, mask):
        return attend_pairs(q, k, v, mask, causal=True)


class Timeline(nn.Module):
    """Causal attention along time on [N, T, d], in chunks: one row per thing, its tokens over the frames. A learned frame position, since
    frames are always one game loop apart."""

    def __init__(self, d, heads, mlp_ratio=4.0):
        super().__init__()
        self.H, self.dh = heads, d // heads
        self.norm, self.qkv, self.o = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2, self.ff = nn.LayerNorm(d), mlp(d, d, int(d * mlp_ratio))
        self.time = LoopTime()
        self.time_in = nn.Linear(self.time.out_dim(), d)
        self.attend = TimeAttend()

    def forward(self, x, mask, past=None):
        """x [N, T, d], mask [N, T]: real tokens. Rows without a real token are skipped. past [N, T, d]: tokens to read instead of x's: frame t reads past's frames
        before t, and x itself only where past has nothing to give, its first frame in the row."""
        N, T, d = x.shape
        rows = torch.nonzero(mask.any(1), as_tuple=True)[0]
        if not len(rows):
            return x
        t = self.time_in(self.time(torch.arange(T, device=x.device)).to(x.dtype))
        key = x
        if past is not None:
            key = torch.cat([x[:, :1], torch.where(mask[:, :-1, None], past[:, :-1], x[:, 1:])], 1)
        res = []
        for ch in rows.split(max(2, int(MAX_SCORES / T / T))):
            n = len(ch)
            ch = ch if n > 1 else ch.repeat(2)      # at least 2 rows: see FrameBlock
            h = self.qkv(self.norm(x[ch]) + t)
            q, k, v = h.view(len(ch), T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
            if past is not None:
                _, k, v = self.qkv(self.norm(key[ch]) + t).view(len(ch), T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
            res.append(self.attend(q, k, v, mask[ch]).transpose(1, 2).reshape(len(ch), T, d)[:n])
        a = torch.cat(res).to(x.dtype)
        y = x[rows] + self.o(a)
        y = y + self.ff(self.n2(y))
        out = x.clone()
        out[rows] = torch.where(mask[rows][..., None], y, x[rows])
        return out


class FramePre(nn.Module):
    """The tokens' queries, keys and values, rotated by their positions."""

    def __init__(self, d, heads):
        super().__init__()
        self.H, self.dh = heads, d // heads
        self.norm_e, self.norm_m = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)

    def forward(self, xe, xm, ang):
        h = torch.cat([self.norm_e(xe), self.norm_m(xm)], 1)
        B, T, _ = h.shape
        q, k, v = self.qkv(h).view(B, T, 3, self.H, self.dh).unbind(2)
        return _rot(q, ang), _rot(k, ang), v


class FrameWin(nn.Module):
    """One size group of windows: a joint softmax per token over its window and the globals."""

    def forward(self, qf, kf, vf, gk, gv, gm, tok, dest, frame, m, code=None, rowmap=None, bias=None):
        rows, cap = m.shape
        H, dh = qf.shape[1:]
        pad = lambda t_: qf.new_zeros(rows * cap, H, dh).index_copy(0, dest, t_.index_select(0, tok)).view(rows, cap, H, dh)
        r = attend_pairs(pad(qf).transpose(1, 2), torch.cat([pad(kf), gk[frame]], 1).transpose(1, 2), torch.cat([pad(vf), gv[frame]], 1).transpose(1, 2),
                         torch.cat([m, gm[frame]], 1), code=code, rowmap=rowmap, bias=bias)
        return r.transpose(1, 2).reshape(rows * cap, H, dh).index_select(0, dest)


class FrameGlob(nn.Module):
    """Each global over all of the level's tokens."""

    def forward(self, q, k, v, valid):
        return attend_pairs(q, k, v, valid)


class FramePost(nn.Module):
    def __init__(self, d, mlp_ratio):
        super().__init__()
        self.o = nn.Linear(d, d)
        self.ne, self.me = nn.LayerNorm(d), mlp(d, d, int(d * mlp_ratio))
        self.nm, self.mm = nn.LayerNorm(d), mlp(d, d, int(d * mlp_ratio))

    def forward(self, xe, xm, a):
        o = self.o(a)
        n0 = xe.shape[1]
        xe, xm = xe + o[:, :n0], xm + o[:, n0:]
        return torch.cat([xe + self.me(self.ne(xe)), xm + self.mm(self.nm(xm))], 1)


class FrameBlock(nn.Module):
    """Current frame: per window a joint softmax over its tokens and the globals, with the action pointers' bias; per global, over everything. Its parts
    compile; the loop over size groups stays Python."""

    def __init__(self, d, heads, mlp_ratio=4.0):
        super().__init__()
        self.pre, self.win, self.glob, self.post = FramePre(d, heads), FrameWin(), FrameGlob(), FramePost(d, mlp_ratio)
        self.bias = nn.Parameter(torch.zeros(3, heads))

    def forward(self, x, g0, n0, ang, valid, sets):
        """x [B, T, d]: [globals (g0), rows, actions and effects, map tokens (from n0)]; sets: window_sets with pairs."""
        B, T, d = x.shape
        xe, xm = x[:, :n0], x[:, n0:]
        q, k, v = self.pre(xe, xm, ang)
        H, dh = q.shape[2:]
        qf, kf, vf = q.reshape(B * T, H, dh), k.reshape(B * T, H, dh), v.reshape(B * T, H, dh)
        gk, gv, gm = k[:, :g0], v[:, :g0], valid[:, :g0]
        bias = (self.bias * torch.tensor([0.0, 1.0, 1.0], device=x.device)[:, None]).to(qf.dtype)    # code 0: no pair, no bias
        res = []
        for tok, dest, frame, m, code, rowmap in sets:
            # one shape of call only: a new specialization can't compile during a checkpoint's recomputation. So at least 2 rows, and always a code table
            if m.shape[0] == 1:
                m, frame = torch.cat([m, m.new_zeros(m.shape)]), frame.repeat(2)
            if code is None:
                code = torch.zeros(1, m.shape[1], m.shape[1] + g0, dtype=torch.uint8, device=m.device)
                rowmap = torch.full((m.shape[0],), -1, device=m.device)
            elif len(rowmap) < m.shape[0]:
                rowmap = torch.cat([rowmap, rowmap.new_full((1,), -1)])
            res.append(self.win(qf, kf, vf, gk, gv, gm, tok, dest, frame, m, code, rowmap, bias))
        a = qf.new_zeros(B * T, H, dh).index_copy(0, torch.cat([s_[0] for s_ in sets]), torch.cat(res))
        ch = max(2, int(MAX_SCORES / g0 / T))
        ga = torch.cat([self.glob(q[i:i + ch, :g0].transpose(1, 2), k[i:i + ch].transpose(1, 2), v[i:i + ch].transpose(1, 2), valid[i:i + ch]) for i in range(0, B, ch)]).transpose(1, 2)
        a = torch.cat([ga, a.view(B, T, H, dh)[:, g0:]], 1).reshape(B, T, d)
        return self.post(xe, xm, a)


@torch.no_grad()
def with_pairs(sets, n_tok, a_tok, t_tok, n_glob=3):
    """Adds to each window set the pointer pairs in it: code [P, cap + g, cap + g] (1 action -> target, 2 target -> action) and rowmap [rows]. a_tok,
    t_tok: flat token positions of each action and its target. A pair split across windows has no bias there."""
    out = []
    for tok, dest, frame, m in sets:
        rows, cap = m.shape
        code, rowmap = None, None
        if len(a_tok):
            place = torch.full((n_tok,), -1, device=tok.device)
            place[tok] = dest
            pa, pt = place[a_tok], place[t_tok]
            ok = (pa >= 0) & (pt >= 0) & (pa // cap == pt // cap)
            if ok.any():
                pa, pt = pa[ok], pt[ok]
                r = pa // cap
                uq, inv = torch.unique(r, return_inverse=True)
                rowmap = torch.full((rows,), -1, device=tok.device)
                rowmap[uq] = torch.arange(len(uq), device=tok.device)
                code = torch.zeros(len(uq), cap, cap + n_glob, dtype=torch.uint8, device=tok.device)
                code[inv, pa % cap, pt % cap] = 1
                code[inv, pt % cap, pa % cap] = 2
        out.append((tok, dest, frame, m, code, rowmap))
    return out


class PhysicsEncoder(nn.Module):
    """The whole sequence through the pyramid. Returns the last level's tokens, rows, cells and globals, twice: the clean state and the action-updated tokens."""

    def __init__(self, vocab, avocab, d=128, conv_blocks=2, blocks=(2, 2, 2), heads=4, mlp_ratio=4.0, cell_timeline_from=1):
        super().__init__()
        self.d = d
        self.unit = nn.Embedding(vocab.n_units, d)
        self.ability = nn.Embedding(vocab.n_abilities, d)
        self.owner = nn.Embedding(3, d)
        self.num_in = mlp(ROW_NUMS, d)
        self.groups = nn.Linear(C_GROUPS, d)
        self.buff = nn.Embedding(N_BUFF_IDS, d)
        self.bag_in = nn.ModuleDict(dict(ord=mlp(4, d), prod=mlp(2, d), rally=mlp(3, d)))
        self.raw_ability = nn.Embedding(N_RAW_ABILITY_IDS, d)
        self.auto_on = nn.Embedding(2, d)
        self.map_in = nn.Linear((C_DYNAMIC + C_STATIC) * PATCH * PATCH + 2, d)
        self.race = nn.Embedding(4, d)
        self.ply_in = mlp(C_PLY, d)
        self.player = nn.Embedding(2, d)
        self.cls = nn.Parameter(torch.zeros(1, d))
        self.view_in = nn.Linear(2, d)        # which players a sample's state is seen by: [1, 1] for physics
        self.map_wh = nn.Linear(2, d)
        # actions
        self.a_kind, self.a_family = nn.Embedding(N_ACT_KINDS, d), nn.Embedding(avocab.n_families, d, padding_idx=0)
        self.a_button, self.a_tkind = nn.Embedding(avocab.n_buttons, d, padding_idx=0), nn.Embedding(4, d)
        self.a_group, self.a_gop = nn.Embedding(N_GROUPS, d, padding_idx=0), nn.Embedding(N_GROUP_OPS, d, padding_idx=0)
        self.a_remove = nn.Embedding(N_REMOVE, d, padding_idx=0)
        self.a_subgroup = nn.Embedding(N_SUBGROUPS, d, padding_idx=0)
        self.a_num = mlp(C_ACTION, d)
        self.a_mix = mlp(d, d)
        # effects: no ids, so tokens of the frame they're in, without a timeline
        self.effect = nn.Embedding(N_EFFECT_IDS, d)
        self.eff_num = nn.Linear(1, d)
        # levels
        self.rope = RoPE2D(d // heads, lo=1.0, hi=256.0)
        self.grids = [TOP >> i for i in range(3)]
        self.convs = nn.ModuleList([nn.ModuleList([ResBlock(d) for _ in range(conv_blocks)]) for _ in self.grids])
        self.down = nn.ModuleList([Down(d) for _ in self.grids[:-1]])
        mk = lambda c, n: nn.ModuleList([nn.ModuleList([c(d, heads, mlp_ratio) for _ in range(k)]) for k in n])
        self.t_ent, self.t_act, self.t_glob, self.frame = mk(Timeline, blocks), mk(Timeline, blocks), mk(Timeline, blocks), mk(FrameBlock, blocks)
        self.a_frame = mk(FrameBlock, blocks)      # starts as the identity: the updated tokens start as the state
        for level in self.a_frame:
            for blk in level:
                for m in (blk.post.o, blk.post.me[-1], blk.post.mm[-1]):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
        self.t_cell = nn.ModuleList([nn.ModuleList([Timeline(d, heads, mlp_ratio) for _ in range(k)]) if li >= cell_timeline_from else None for li, k in enumerate(blocks)])
        self.norm_e, self.norm_c, self.norm_g = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        nn.init.normal_(self.cls, std=0.02)
        self.grad_ckpt = False

    def compile_parts(self):
        """The attention. Compiling the convs and MLPs too fails at 40 frames and up (Inductor's CantSplit), and gains little. One specialization per window
        size group: past the default limit of 8 a part falls back to eager, and that crashes inside a checkpoint's recomputation."""
        torch._dynamo.config.recompile_limit = 256
        for m in self.modules():
            if type(m).__name__ in ("FrameWin", "FrameGlob", "TimeAttend"):
                m.compile(dynamic=True)

    def _bag(self, name, b, emb):
        """Mean of a list-valued state per token: [B * F * R] flat index, embedded items."""
        where, B, F_, R = b[f"{name}_at"], *b["mask"].shape
        out = emb.new_zeros(B * F_ * R, emb.shape[-1]).index_add(0, where, emb)
        cnt = torch.bincount(where, minlength=B * F_ * R).clamp(min=1)[:, None].to(emb.dtype)
        return (out / cnt).view(B, F_, R, -1)

    def rows(self, b):
        ids = self.unit(b["type"].long()) + self.owner(b["owner"])[:, None] + self.num_in(row_numbers(b)) + self.groups(b["groups"].to(b["num"].dtype))
        ids = ids + self._bag("ord", b, self.ability(b["ord_abil"]) + self.bag_in["ord"](b["ord_feat"]))
        ids = ids + self._bag("prod", b, self.ability(b["prod_abil"]) + self.bag_in["prod"](b["prod_feat"]))
        ids = ids + self._bag("buff", b, self.buff(b["buff_id"]))
        ids = ids + self._bag("rally", b, self.bag_in["rally"](b["rally_feat"]))
        ids = ids + self._bag("ab", b, self.raw_ability(b["ab_id"]))
        ids = ids + self._bag("auto", b, self.a_family(b["auto_fam"]) + self.a_button(b["auto_but"]) + self.auto_on(b["auto_on"]))
        return ids * b["mask"][..., None]

    def globals(self, b):
        B, F_ = b["mask"].shape[:2]
        p = self.ply_in(b["ply_num"]) + self.race(b["players"])[:, None] + self.player.weight[None, None]
        c = (self.cls + self.view_in(b["view"]) + self.map_wh(b["half"] / 100))[:, None].expand(B, F_, -1)
        return torch.cat([c[:, :, None], p], 2)                 # [B, F, 3, d]

    def actions(self, b):
        n = self.a_kind(b["act_kind"]) + self.a_family(b["act_family"]) + self.a_button(b["act_button"]) + self.a_tkind(b["act_tkind"]) \
            + self.a_group(b["act_group"]) + self.a_gop(b["act_gop"]) + self.a_remove(b["act_remove"]) + self.a_subgroup(b["act_subgroup"]) + self.a_num(b["act_num"]) \
            + self.player(b["act_player"])
        return self.a_mix(n)

    def map_tokens(self, b):
        B, F_ = b["mask"].shape[:2]
        c = (torch.arange(TOP, device=b["map"].device, dtype=torch.float32) + 0.5) / TOP * 2 - 1
        rel = torch.stack(torch.broadcast_tensors(c[None, :], c[:, None]), -1)
        st = b["static"][:, None].expand(B, F_, -1, -1, -1).float().clone()
        st[..., 2 * PATCH * PATCH:3 * PATCH * PATCH] /= 255
        x = torch.cat([b["map"].float(), st, rel[None, None].expand(B, F_, -1, -1, -1)], -1)
        return self.map_in(x.to(self.cls.dtype)).permute(0, 1, 4, 2, 3)      # [B, F, d, TOP, TOP]

    def forward(self, b, last_action=True):
        """last_action False skips the last block's action attention, when only the clean state is wanted: the updated tokens returned are then a block old."""
        B, F_, R = b["mask"].shape
        dev = b["mask"].device
        d = self.d
        A = b["act_mask"].shape[2]
        G = self.globals(b).reshape(B * F_, 3, d)
        g0 = 3
        X = b["eff_mask"].shape[2]
        eff = self.effect(b["eff_type"]) + self.owner(b["eff_owner"]) + self.eff_num(b["eff_radius"][..., None])
        acts = self.actions(b)
        E = torch.cat([self.rows(b), torch.zeros_like(acts), eff], 2).reshape(B * F_, R + A + X, d)      # the state's action slots stay out of its attention
        ne = R + A + X
        n0 = g0 + ne
        emask = torch.cat([b["mask"], b["act_mask"], b["eff_mask"]], 2).reshape(B * F_, ne)
        epos = torch.cat([torch.stack([b["x"], b["y"]], -1) - b["center"][:, None, None], b["act_xy"], b["eff_xy"]], 2).reshape(B * F_, ne, 2)
        member_e = torch.cat([b["mask"], b["act_mask"] & b["act_placed"], b["eff_mask"]], 2).reshape(B * F_, ne)
        no_act = torch.cat([torch.zeros_like(b["mask"]), b["act_mask"], torch.zeros_like(b["eff_mask"])], 2).reshape(B * F_, ne)
        half = b["side"][:, None].expand(B, F_).reshape(-1) / 2
        N = self.map_tokens(b).reshape(B * F_, d, TOP, TOP)
        gmask = torch.ones(B * F_, g0, dtype=torch.bool, device=dev)
        fidx = torch.arange(B * F_, device=dev)
        # pointers: action -> target row, flat token positions per level
        ai = torch.nonzero(b["act_target"].reshape(-1) >= 0, as_tuple=True)[0]
        aframe = ai // A
        aslot = ai % A
        arow = b["act_target"].reshape(-1)[ai]
        one = N.new_ones(1, 1, 1, 1)
        from torch.utils.checkpoint import checkpoint
        ckpt = self.grad_ckpt and any(p.requires_grad for p in self.parameters())
        run = (lambda f, *a: checkpoint(f, *a, use_reentrant=False)) if ckpt else (lambda f, *a: f(*a))
        de, dn = E.new_zeros(B * F_, g0 + ne, d), N.new_zeros(B * F_, d, TOP, TOP)      # what the actions have changed so far: tokens, cells
        for li, S in enumerate(self.grids):
            for blk in self.convs[li]:
                N = run(blk, N, one)
            size = 2 * half / S
            c = (torch.arange(S, device=dev, dtype=torch.float32) + 0.5) - S / 2
            cen = torch.stack(torch.broadcast_tensors(c[None, :], c[:, None]), -1).view(1, S * S, 2) * size[:, None, None]
            ij = (epos / size[:, None, None] + S / 2).floor().long().clamp(0, S - 1)
            cj = torch.arange(S, device=dev)
            nij = torch.stack(torch.broadcast_tensors(cj[None, :], cj[:, None]), -1).view(1, S * S, 2).expand(B * F_, -1, -1)
            tij = torch.cat([torch.zeros(B * F_, g0, 2, dtype=torch.long, device=dev), ij, nij], 1)
            valid = torch.cat([gmask, emask, torch.ones(B * F_, S * S, dtype=torch.bool, device=dev)], 1)
            member = torch.cat([torch.zeros_like(gmask), member_e, torch.ones(B * F_, S * S, dtype=torch.bool, device=dev)], 1)
            alone = torch.cat([torch.zeros_like(gmask), emask & ~member_e, torch.zeros(B * F_, S * S, dtype=torch.bool, device=dev)], 1)   # no place: one window per frame
            state_valid = valid & ~torch.cat([torch.zeros_like(gmask), no_act, torch.zeros(B * F_, S * S, dtype=torch.bool, device=dev)], 1)
            state_member = member & state_valid
            T = valid.shape[1]
            a_tok = aframe * T + g0 + R + aslot
            t_tok = aframe * T + g0 + arow
            sets, state_sets = [], []
            for sl in (0, SLIDE):
                w, nw = window_of(tij, S, sl)
                sets.append(with_pairs(window_sets(w.masked_fill(alone, nw), member | alone, nw + 1), B * F_ * T, a_tok, t_tok, g0))
                state_sets.append([(*s_, None, None) for s_ in window_sets(w, state_member, nw)])
            pos = torch.cat([torch.zeros(B * F_, g0, 2, device=dev), epos, cen.expand(B * F_, -1, -1)], 1)
            ang = self.rope.angles(pos)
            x = torch.cat([G, E, N.permute(0, 2, 3, 1).reshape(B * F_, S * S, d)], 1)
            for j, blk in enumerate(self.frame[li]):
                # timeline: earlier frames' tokens as the last block's actions left them
                up = x + torch.cat([de, dn.permute(0, 2, 3, 1).reshape(B * F_, S * S, d)], 1)
                by_row = lambda z, n_: z.view(B, F_, n_, d).permute(0, 2, 1, 3).reshape(B * n_, F_, d)
                e = x[:, g0:n0].view(B, F_, ne, d)
                rows_ = run(self.t_ent[li][j], by_row(x[:, g0:g0 + R], R), b["mask"].permute(0, 2, 1).reshape(B * R, F_), by_row(up[:, g0:g0 + R], R))
                acts = self._act_timeline(acts, b, self.t_act[li][j], run)
                e = torch.cat([rows_.view(B, R, F_, d).permute(0, 2, 1, 3), e[:, :, R:]], 2)
                gl = run(self.t_glob[li][j], by_row(x[:, :g0], g0), torch.ones(B * g0, F_, dtype=torch.bool, device=dev), by_row(up[:, :g0], g0))
                x = torch.cat([gl.view(B, g0, F_, d).permute(0, 2, 1, 3).reshape(B * F_, g0, d), e.reshape(B * F_, ne, d), x[:, n0:]], 1)
                if self.t_cell[li] is not None:
                    cells = run(self.t_cell[li][j], by_row(x[:, n0:], S * S), torch.ones(B * S * S, F_, dtype=torch.bool, device=dev), by_row(up[:, n0:], S * S))
                    x = torch.cat([x[:, :n0], cells.view(B, S * S, F_, d).permute(0, 2, 1, 3).reshape(B * F_, S * S, d)], 1)
                x = run(blk, x, g0, n0, ang, state_valid, state_sets[j % 2])    # the clean state
                if not last_action and li == len(self.grids) - 1 and j == len(self.frame[li]) - 1:
                    break
                # the same tokens with this frame's actions, from where the last block left them
                y = x + torch.cat([de, dn.permute(0, 2, 3, 1).reshape(B * F_, S * S, d)], 1)
                a0 = g0 + R
                y = torch.cat([y[:, :a0], acts.view(B * F_, A, d), y[:, a0 + A:]], 1)
                y = run(self.a_frame[li][j], y, g0, n0, ang, valid, sets[j % 2])
                acts = y[:, a0:a0 + A].view(B, F_, A, d)
                delta = y - x
                de = torch.cat([delta[:, :a0], delta[:, a0:a0 + A] * 0, delta[:, a0 + A:n0]], 1)
                dn = delta[:, n0:].view(B * F_, S, S, d).permute(0, 3, 1, 2)
            G, E = x[:, :g0], x[:, g0:n0]
            N = x[:, n0:].view(B * F_, S, S, d).permute(0, 3, 1, 2)
            if li < len(self.down):
                N, dn = self.down[li](N), F.avg_pool2d(dn, 2)
        out = lambda g_, e_, n_: dict(ent=self.norm_e(e_[:, :R]).view(B, F_, R, d), cells=self.norm_c(n_.permute(0, 2, 3, 1).reshape(B * F_, S * S, d)).view(B, F_, S * S, d),
                                      glob=self.norm_g(g_).view(B, F_, g0, d))
        return out(G, E, N), out(G + de[:, :g0], E + de[:, g0:n0], N + dn)

    def _act_timeline(self, acts, b, tl, run):
        """Each player's action tokens over the window, causal, in time order. Tokens live per frame; a timeline needs one row per player."""
        B, F_, A, d = acts.shape
        order = b["act_order"]                                    # [B, 2, N] flat (frame * A + slot) of each player's tokens, -1 pad
        if not (order >= 0).any():
            return acts
        if order.shape[2] == 1:
            order = F.pad(order, (0, 1), value=-1)
        Nn = order.shape[2]
        flat = acts.reshape(B, F_ * A, d)
        m = order >= 0
        seq = torch.gather(flat[:, None].expand(-1, 2, -1, -1), 2, order.clamp(min=0)[..., None].expand(-1, -1, -1, d))
        y = run(tl, seq.reshape(B * 2, Nn, d), m.reshape(B * 2, Nn)).view(B, 2, Nn, d)
        out = flat.clone()
        bi, pi, ni = torch.nonzero(m, as_tuple=True)
        out[bi, order[bi, pi, ni]] = y[bi, pi, ni].to(out.dtype)
        return out.view(B, F_, A, d)


class RowHead(nn.Module):
    """A row's token -> its raw fields at some frame. One head, shared by rows and birth queries."""

    def __init__(self, d, n_units, n_abilities):
        super().__init__()
        self.unit, self.inside = nn.Linear(d, n_units), nn.Linear(d, 3)
        self.alive = nn.Linear(d, 1)
        self.reg = nn.Linear(d, 2 + len(REG))
        self.flags, self.display, self.cloak = nn.Linear(d, N_FLAGS), nn.Linear(d, 4), nn.Linear(d, 4)
        self.slot = nn.Embedding(QUEUE + PROD, d)
        self.slot_mix = mlp(d, d)
        self.slot_abil = nn.Linear(d, n_abilities)
        self.slot_reg = nn.Linear(d, 4)  # order target dx, dy, whether it has one; progress
        self.buffs, self.groups = nn.Linear(d, N_BUFF_IDS), nn.Linear(d, C_GROUPS)
        self.bits = nn.Linear(d, 3)      # has an add-on, an engaged target, a rally point
        self.rally = nn.Linear(d, 2)
        self.abilities = nn.Linear(d, N_RAW_ABILITY_IDS)
        self.ptr_q = nn.Embedding(2 + QUEUE, d)      # engaged target, container, each order's target
        self.ptr_mix, self.ptr_k = mlp(d, d), nn.Linear(d, d)

    def forward(self, x):
        return dict(type=self.unit(x), inside=self.inside(x), alive=self.alive(x).squeeze(-1), reg=self.reg(x), flags=self.flags(x),
                    display=self.display(x), cloak=self.cloak(x), buffs=self.buffs(x), groups=self.groups(x), bits=self.bits(x), rally=self.rally(x))

    def slots(self, x):
        y = self.slot_mix(x[:, None] + self.slot.weight[None])
        return self.slot_abil(y), self.slot_reg(y)

    def pointers(self, x, rows):
        """x [N, d] tokens, rows [N, R, d] their frames' rows -> [N, 2 + QUEUE, R] scores of each pointer over the rows."""
        q = self.ptr_mix(x[:, None] + self.ptr_q.weight[None])
        return torch.einsum("npd,nrd->npr", q, self.ptr_k(rows)) / math.sqrt(x.shape[-1])


# regressed as change, scaled: (name, divisor)
REG = (("hp", 100.0), ("shield", 100.0), ("energy", 50.0), ("build", 0.1), ("contents", 100.0), ("harvesters", 4.0), ("ideal_harvesters", 4.0),
       ("hp_max", 100.0), ("shield_max", 100.0), ("weapon_cooldown", 8.0), ("cargo_used", 4.0), ("buff_remain", 50.0), ("facing", 1.0),
       ("radius", 0.5), ("cargo_max", 4.0), ("buff_max", 50.0), ("detect_range", 4.0), ("radar_range", 8.0), ("atk_up", 1.0), ("arm_up", 1.0), ("shd_up", 1.0), ("z", 1.0))


class BirthHead(nn.Module):
    """Per frame, how many units appear next, then that many queries plus a margin, in parallel: each reads the frame's rows and says real or
    empty, with a new row's fields and a pointer to its parent."""

    def __init__(self, d, heads, head):
        super().__init__()
        self.count = mlp(d, MAX_BIRTHS)
        self.query = nn.Embedding(MAX_BIRTHS + MARGIN, d)
        self.H, self.dh = heads, d // heads
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.n1, self.n2, self.ff = nn.LayerNorm(d), nn.LayerNorm(d), mlp(d, d, 4 * d)
        self.real, self.pos = nn.Linear(d, 1), nn.Linear(d, 2)
        self.parent_q, self.parent_k, self.no_parent = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, 1)
        self.head = [head]

    def forward(self, cls, rows, rmask, n):
        """cls [K, d], rows [K, R, d], rmask [K, R], n queries -> outputs per (frame, query)."""
        K, R, d = rows.shape
        x = cls[:, None] + self.query.weight[:n][None]
        k, v = self.kv(rows).view(K, R, 2, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q = self.q(self.n1(x)).view(K, n, self.H, self.dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=rmask[:, None, None])
        x = x + self.o(a.transpose(1, 2).reshape(K, n, d))
        x = x + self.ff(self.n2(x))
        parent = torch.einsum("kqd,krd->kqr", self.parent_q(x), self.parent_k(rows)) / math.sqrt(d)
        parent = parent.masked_fill(~rmask[:, None], -1e4)
        parent = torch.cat([parent, self.no_parent(x)], -1)
        return x, self.real(x).squeeze(-1), self.pos(x), parent


class Physics(nn.Module):
    """Encoder, EMA teacher, residual head, raw heads, births. forward(batch) -> (loss, logs)."""

    def __init__(self, vocab, avocab, d=128, ema=0.996, demean=True, proj=512, moving_neutrals=(), larva=151, **kw):
        super().__init__()
        self.enc = PhysicsEncoder(vocab, avocab, d, **kw)
        self.teacher = copy.deepcopy(self.enc)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.g = mlp(d, d, 2 * d)
        nn.init.zeros_(self.g[-1].weight)
        nn.init.zeros_(self.g[-1].bias)
        self.proj = nn.Sequential(nn.Linear(d, proj), nn.GELU(), nn.Linear(proj, proj))
        self.head = RowHead(d, vocab.n_units, vocab.n_abilities)
        self.births = BirthHead(d, kw.get("heads", 4), self.head)
        self.map_dec = MapDecoder(d, C_DYNAMIC + N_EFFECT_IDS)
        self.player_next = nn.Linear(d, len(PC_KEYS))
        self.ema, self.demean = ema, demean
        self.register_buffer("critter", torch.tensor(sorted(moving_neutrals) or [-1], dtype=torch.int64), persistent=False)
        self.larva = larva
        self.aux_weight = 1.0
        self.record_pos = False

    @torch.no_grad()
    def update_teacher(self):
        for p, t in zip(self.enc.parameters(), self.teacher.parameters()):
            t.lerp_(p, 1 - self.ema)
        for p, t in zip(self.enc.buffers(), self.teacher.buffers()):
            t.copy_(p)

    def barlow(self, p, t, rows_t, lam=0.005):
        """Barlow Twins between predicted and teacher tokens at (sample, row) pairs; correlations after removing each row's mean over time if demean."""
        zp, zt = self.proj(p).float(), self.proj(t).float()
        if self.demean:
            n = int(rows_t.max()) + 1
            cnt = torch.bincount(rows_t, minlength=n).clamp(min=1)[:, None].to(zp.dtype)
            sub = lambda z: z - (torch.zeros(n, z.shape[1], device=z.device, dtype=z.dtype).index_add(0, rows_t, z) / cnt)[rows_t]
            zp, zt = sub(zp), sub(zt)
        zp = (zp - zp.mean(0)) / (zp.std(0) + 1e-4)
        zt = (zt - zt.mean(0)) / (zt.std(0) + 1e-4)
        c = zp.T @ zt / len(zp)
        on = (torch.diagonal(c) - 1).pow(2).sum()
        off = (c.pow(2).sum() - torch.diagonal(c).pow(2).sum())
        return (on + lam * off) / zp.shape[1], c.detach()

    def forward(self, b):
        _, out = self.enc(b)
        with torch.no_grad():
            tgt, _ = self.teacher(b, last_action=False)
        B, F_, R = b["mask"].shape
        logs = {}
        # who is scored: alive with a next frame; dead rows score nothing
        sc = b["mask"][:, :-1] & b["alive"][:, :-1] & b["mask"][:, 1:]
        nxt = {k: v[:, :-1] for k, v in out.items()}   # frame t's action-updated tokens: everything next-state reads this
        self._rows = nxt["ent"]                        # what pointers point into
        s = nxt["ent"]
        p = s + self.g(s)
        bi, fi, ri = torch.nonzero(sc, as_tuple=True)
        pp, tt = p[bi, fi, ri], tgt["ent"][:, 1:][bi, fi, ri]
        row_key = bi * R + ri
        uq, inv = torch.unique(row_key, return_inverse=True)
        lat, corr = self.barlow(pp, tt, inv)
        logs["barlow"] = lat
        with torch.no_grad():
            logs["latent_change"] = (tgt["ent"][:, 1:][bi, fi, ri] - tgt["ent"][:, :-1][bi, fi, ri]).pow(2).sum(-1).mean()
        raw, raw_logs = self.raw_losses(p, s, b, sc, bi, fi, ri)
        logs.update(raw_logs)
        brth, blogs = self.birth_losses(nxt, b)
        logs.update(blogs)
        mp = self.map_loss(nxt, b)
        logs["map"] = mp
        pn = F.smooth_l1_loss(self.player_next(nxt["glob"][:, :, 1:3]).float(), (b["ply_num"][:, 1:] - b["ply_num"][:, :-1])[..., :len(PC_KEYS)] * 100)
        logs["player"] = pn
        loss = lat + self.aux_weight * (raw + brth + mp + pn)
        logs["loss"] = loss
        return loss, logs

    def raw_losses(self, p, s, b, sc, bi, fi, ri):
        B, F_, R = b["mask"].shape
        n = b["num"]
        typ, rtyp = b["type"].long(), b["rtype"]
        own = b["owner"][bi, ri]
        t_type = typ[bi, fi + 1, ri]
        rt = rtyp[bi, fi + 1, ri]
        crit = (own == NEUTRAL) & torch.isin(rt, self.critter)
        larva = rt == self.larva
        whole = torch.where(crit, DISCOUNT, 1.0).to(p.dtype)
        pos_w = torch.where(larva, whole * DISCOUNT, whole)
        o = self.head(p[bi, fi, ri])
        logs = {}
        dx = (b["x"][bi, fi + 1, ri] - b["x"][bi, fi, ri]) / POS_SCALE
        dy = (b["y"][bi, fi + 1, ri] - b["y"][bi, fi, ri]) / POS_SCALE
        tpos = torch.stack([dx, dy], -1)
        lp = F.smooth_l1_loss(o["reg"][:, :2].float(), tpos, reduction="none").sum(-1)
        logs["pos"] = (lp * pos_w).sum() / pos_w.sum()
        logs["pos_copy"] = (F.smooth_l1_loss(torch.zeros_like(tpos), tpos, reduction="none").sum(-1) * pos_w).sum() / pos_w.sum()
        if self.record_pos:
            self.pos_eval = self._position_errors(o["reg"][:, :2].float(), tpos, own, crit, larva, b["alive"][bi, fi + 1, ri] & (b["inside"][bi, fi, ri] == 0) & (b["inside"][bi, fi + 1, ri] == 0))
        reg_t = torch.stack([(n[bi, fi + 1, ri, I[k]] - n[bi, fi, ri, I[k]]) / dv for k, dv in REG], -1)
        lr = F.smooth_l1_loss(o["reg"][:, 2:].float(), reg_t, reduction="none").mean(-1)
        logs["reg"] = (lr * whole).sum() / whole.sum()
        logs["type"] = (F.cross_entropy(o["type"].float(), t_type, reduction="none") * whole).sum() / whole.sum()
        logs["inside"] = (F.cross_entropy(o["inside"].float(), b["inside"][bi, fi + 1, ri].long(), reduction="none") * whole).sum() / whole.sum()
        flags_t = (n[bi, fi + 1, ri, I["flags"]].long() | (n[bi, fi + 1, ri, I["flags2"]].long() << 8))
        bits = ((flags_t[:, None] >> torch.arange(N_FLAGS, device=p.device)) & 1).float()
        logs["flags"] = (F.binary_cross_entropy_with_logits(o["flags"].float(), bits, reduction="none").mean(-1) * whole).sum() / whole.sum()
        logs["display"] = (F.cross_entropy(o["display"].float(), n[bi, fi + 1, ri, I["display"]].long().clamp(0, 3), reduction="none") * whole).sum() / whole.sum()
        logs["cloak"] = (F.cross_entropy(o["cloak"].float(), n[bi, fi + 1, ri, I["cloak"]].long().clamp(0, 3), reduction="none") * whole).sum() / whole.sum()
        # the alive field: scored on rows alive now, flips balanced against the many that stay
        died = (~b["alive"][bi, fi + 1, ri]).float()
        pw = (len(died) - died.sum()).clamp(min=1) / died.sum().clamp(min=1)
        logs["alive"] = F.binary_cross_entropy_with_logits(o["alive"].float(), died, pos_weight=pw.clamp(max=100.0), reduction="none").mul(whole).sum() / whole.sum()
        logs["queues"], logs["abilities"], logs["pointers"] = self._sampled(p, b, bi, fi, ri)
        # buffs, control groups and small presence bits: every scored token. Rare positives are the point, so report them too
        key = (bi * F_ + fi + 1) * R + ri
        buffs_t = torch.zeros(len(bi), N_BUFF_IDS, device=p.device, dtype=o["buffs"].float().dtype)
        pos = torch.full((B * F_ * R,), -1, device=p.device)
        pos[key] = torch.arange(len(key), device=p.device)
        hit = pos[b["buff_at"]]
        ok = hit >= 0
        buffs_t[hit[ok], b["buff_id"][ok]] = 1
        wt = whole[:, None]
        logs["buffs"] = (F.binary_cross_entropy_with_logits(o["buffs"].float(), buffs_t, reduction="none", pos_weight=torch.full((N_BUFF_IDS,), 20.0, device=p.device)) * wt).sum() / wt.sum() / 32
        logs["groups"] = (F.binary_cross_entropy_with_logits(o["groups"].float(), b["groups"][bi, fi + 1, ri].float(), reduction="none").mean(-1) * whole).sum() / whole.sum()
        bits_t = torch.stack([n[bi, fi + 1, ri, I["add_on"]] >= 0, b["engaged_row"][bi, fi + 1, ri] >= 0, b["rally_first"][bi, fi + 1, ri, 0] > 0], -1).float()
        logs["bits"] = (F.binary_cross_entropy_with_logits(o["bits"].float(), bits_t, reduction="none").mean(-1) * whole).sum() / whole.sum()
        rm = b["rally_first"][bi, fi + 1, ri, 0] > 0
        logs["rally"] = F.smooth_l1_loss(o["rally"][rm].float(), b["rally_first"][bi, fi + 1, ri, 1:3][rm]) if rm.any() else p.sum() * 0
        total = POS_BOOST * logs["pos"] + logs["reg"] + logs["type"] + logs["inside"] + logs["flags"] + logs["display"] + logs["cloak"] + logs["alive"] + logs["queues"] \
            + logs["buffs"] + logs["groups"] + logs["bits"] + logs["rally"] + logs["abilities"] + logs["pointers"]
        return total, logs

    def _position_errors(self, pred, tpos, own, crit, larva, keep):
        """For evaluation: {kind of row: (rows, sum of model error, sum of copy error)}, map units per loop, the rows and kinds of scripts/physics_baselines.py."""
        step = tpos.norm(dim=-1) * POS_SCALE
        err = (pred - tpos).norm(dim=-1) * POS_SCALE
        neutral = own == NEUTRAL
        player = ~neutral & ~larva
        moving = step > 0.003
        kinds = {"player moving": player & moving, "player still": player & ~moving, "player all": player, "larva": larva & ~neutral, "critter": crit, "other neutral": neutral & ~crit}
        return {k: np.array([float((v & keep).sum()), float(err[v & keep].sum()), float(step[v & keep].sum())]) for k, v in kinds.items()}

    def _sampled(self, p, b, bi, fi, ri, cap=4096, ptr_cap=1024):
        """The large outputs, on a sample of the scored tokens, rows whose queue changes first: order and production slots, available abilities, and pointers to
        the engaged target, the container and each order's target."""
        B, F_, R = b["mask"].shape
        key = (bi * (F_ - 1) + fi) * R + ri
        ch = b["queue_changed"].reshape(-1)[key]
        sel = torch.nonzero(ch, as_tuple=True)[0]
        rest = torch.nonzero(~ch, as_tuple=True)[0]
        take = rest[torch.randperm(len(rest), device=p.device)[:max(0, cap - len(sel))]]
        sel = torch.cat([sel[:cap], take])
        if not len(sel):
            z = p.sum() * 0
            return z, z, z
        x = p[bi[sel], fi[sel], ri[sel]]
        abil, reg = self.head.slots(x)
        at = (bi[sel] * F_ + fi[sel] + 1) * R + ri[sel]
        t_abil = b["slot_abil"].reshape(-1, QUEUE + PROD)[at]
        t_reg = b["slot_reg"].reshape(-1, QUEUE + PROD, 4)[at]
        la = F.cross_entropy(abil.float().flatten(0, 1), t_abil.flatten())
        m = (t_abil > 0)[..., None].float()
        queues = la + (F.smooth_l1_loss(reg.float(), t_reg, reduction="none") * m).sum() / m.sum().clamp(min=1)
        # available abilities: the cooldowns the state otherwise hides
        pos = torch.full((B * F_ * R,), -1, device=p.device)
        pos[at] = torch.arange(len(at), device=p.device)
        hit = pos[b["ab_at"]]
        ok = hit >= 0
        t_ab = torch.zeros(len(sel), N_RAW_ABILITY_IDS, device=p.device)
        t_ab[hit[ok], b["ab_id"][ok]] = 1
        abilities = F.binary_cross_entropy_with_logits(self.head.abilities(x).float(), t_ab, pos_weight=torch.full((N_RAW_ABILITY_IDS,), 20.0, device=p.device)) * 8
        # pointers: scored where the next frame has a target row
        s2 = sel[torch.randperm(len(sel), device=p.device)[:ptr_cap]]
        rows = self._rows[bi[s2], fi[s2]]
        at2 = (bi[s2] * F_ + fi[s2] + 1) * R + ri[s2]
        tgt = torch.cat([b["engaged_row"].reshape(-1)[at2][:, None], b["parent_row"].reshape(-1)[at2][:, None], b["ord_row"].reshape(-1, QUEUE)[at2]], 1).long()
        logits = self.head.pointers(p[bi[s2], fi[s2], ri[s2]], rows)
        live = b["mask"][bi[s2], fi[s2]]                        # a target not live in this frame can't be pointed at: it would score the mask's -1e4
        logits = logits.masked_fill(~live[:, None], -1e4).float()
        has = (tgt >= 0) & live.gather(1, tgt.clamp(min=0))
        pointers = F.cross_entropy(logits[has], tgt[has]) if has.any() else p.sum() * 0
        return queues, abilities, pointers

    def birth_losses(self, out, b):
        B, F_, R = b["mask"].shape
        d = out["ent"].shape[-1]
        n_b = b["birth_count"]                                        # [B, F - 1]: births first seen in frame f + 1
        cls = out["glob"][:, :, 0]
        logs = dict(birth_count=F.cross_entropy(self.births.count(cls).float().flatten(0, 1), n_b.flatten().clamp(max=MAX_BIRTHS - 1)))
        rows, rmask = out["ent"], b["mask"][:, :-1]
        fb, ff = torch.nonzero(n_b >= 0, as_tuple=True)
        total_real, total_fields, frames = 0.0, 0.0, 0
        groups = {}
        for k, c in zip(range(len(fb)), n_b[fb, ff].tolist()):
            groups.setdefault(min(c, MAX_BIRTHS - 1), []).append(k)
        for c, idx in groups.items():
            idx = torch.tensor(idx, device=fb.device)
            sb, sf = fb[idx], ff[idx]
            x, real, pos, parent = self.births(cls[sb, sf], rows[sb, sf], rmask[sb, sf], c + MARGIN)
            target = torch.zeros_like(real)
            if c:
                head = self.head(x)
                loss_f = []
                for i in range(len(idx)):
                    tb = (b["birth_b"] == sb[i]) & (b["birth_f"] == sf[i] + 1)
                    rr = b["birth_r"][tb][:c]
                    if not len(rr):
                        continue
                    tt = b["type"][sb[i], sf[i] + 1, rr].long()
                    txy = (torch.stack([b["x"][sb[i], sf[i] + 1, rr], b["y"][sb[i], sf[i] + 1, rr]], -1) - b["center"][sb[i]]) / b["half"][sb[i]]
                    lt = F.log_softmax(head["type"][i].float(), -1)
                    cost = torch.cdist(pos[i].float(), txy, p=1) - lt[:, tt]
                    qi, ti = linear_sum_assignment(cost.detach().cpu().numpy())
                    qi, ti = torch.as_tensor(qi, device=x.device), torch.as_tensor(ti, device=x.device)
                    target[i, qi] = 1
                    w = torch.where(b["rtype"][sb[i], sf[i] + 1, rr][ti] == self.larva, DISCOUNT, 1.0)      # a larva's position takes the discount, its birth doesn't
                    lpos = (F.smooth_l1_loss(pos[i, qi].float(), txy[ti], reduction="none").sum(-1) * w).mean()
                    par = b["birth_p"][tb][:c][ti]
                    par = torch.where(par >= 0, par, parent.shape[-1] - 1)
                    loss_f.append(lpos + F.cross_entropy(head["type"][i, qi].float(), tt[ti]) + F.cross_entropy(parent[i, qi].float(), par))
                if loss_f:
                    total_fields = total_fields + torch.stack(loss_f).sum()
                    frames += len(loss_f)
            total_real = total_real + F.binary_cross_entropy_with_logits(real.float(), target, reduction="sum")
        logs["birth_real"] = total_real / max(1, len(fb)) / (1 + MARGIN)
        logs["birth_fields"] = total_fields / max(1, frames) if frames else out["ent"].sum() * 0
        return logs["birth_count"] + logs["birth_real"] + logs["birth_fields"], logs

    def map_loss(self, out, b, K=256):
        """Next-frame visibility, seen, creep and each effect type at sampled map tokens, read from the coarsest cells. Effects are rare, so the cells they cover
        are always among the sample (eff_force)."""
        B, F_ = b["mask"].shape[:2]
        S = out["cells"].shape[2]
        side = int(round(math.sqrt(S)))
        cells = out["cells"].reshape(B * (F_ - 1), S, -1)
        idx = torch.randint(0, TOP * TOP, (B * (F_ - 1), K), device=cells.device)
        force = b["eff_force"].reshape(B * (F_ - 1), -1)
        idx = torch.cat([idx, torch.where(force >= 0, force, idx[:, :force.shape[1]])], 1)
        iy, ix = idx // TOP, idx % TOP
        r = TOP // side
        cell = (iy // r) * side + (ix // r)
        size = b["side"][:, None].expand(B, F_ - 1).reshape(-1, 1) / TOP
        off = torch.stack([(ix % r + 0.5 - r / 2) * size, (iy % r + 0.5 - r / 2) * size], -1)
        o = self.map_dec(cells, cell, off)
        m = b["map"][:, 1:].reshape(B * (F_ - 1), TOP * TOP, C_DYNAMIC * PATCH * PATCH)
        centre = (PATCH * PATCH) // 2
        tgt = m[..., centre::PATCH * PATCH][:, :, :C_DYNAMIC].float()
        tg = torch.gather(tgt, 1, idx[..., None].expand(-1, -1, C_DYNAMIC))
        te = torch.zeros(*idx.shape, N_EFFECT_IDS, device=idx.device, dtype=tg.dtype)
        if len(b["eff_cover"]):
            fr, ce, ty = b["eff_cover"].unbind(1)                      # next frames' covered tokens: (flat frame, token, effect type)
            hit = idx[fr] == ce[:, None]
            pi, ki = torch.nonzero(hit, as_tuple=True)
            te[fr[pi], ki, ty[pi]] = 1
        return F.binary_cross_entropy_with_logits(o[..., :C_DYNAMIC], tg) + F.binary_cross_entropy_with_logits(o[..., C_DYNAMIC:], te) * 4
