"""The physics model (DESIGN_physics.md): a single pass over a whole sequence predicts
the next state at every frame, as a language model does. No horizon input, no motion
input.

    per frame        256 learned tokens per map plus a ResNet over the live map at the anchors of a 16 x 16 grid, entities, globals, effects and action tokens
    each block       spatial, feed-forward, temporal, feed-forward, then this frame's actions
      spatial        every token reads all tokens of the frame in one softmax. No action of this frame: this is the clean state
      temporal       rows, cells and globals, each reading earlier frames' action-updated tokens; each player's actions: causal, with their own
                     attention and feed-forward weights
      actions        actions read the state (both players' tokens), not each other: a player's own earlier actions come through its timeline;
                     the state reads the actions. Carried on as a change kept apart from the state
    outputs          a residual per token, s + g(s), trained against an EMA teacher's next token with Barlow Twins, plus the raw heads

At inference, frame by frame: state at t -> action predictor -> actions at t -> updated
tokens -> t + 1. Training runs the same dependencies in one pass with the real actions:
a clean state at t holds no action from t, and the target is the teacher's clean state
at t + 1.

An action points at its target through a learned bias on the attention score of that
pair, one per direction. Nothing is copied from the target, so an entity knows nothing
about being targeted until attention tells it.

A state's own pointers (engaged target, container, each order's target) are inputs by
identity: each row has a random code, fresh per window, and a row carries the codes of
the rows it points at. A code means nothing but which row, so the network reads a
pointer by matching codes, and the pointer head can keep a target by matching it. Orders
and production carry their place in the queue.
"""

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

from .action_tokens import (
    C_ACTION,
    N_ACT_KINDS,
    N_GROUP_OPS,
    N_GROUPS,
    N_REMOVE,
    N_SUBGROUPS,
)
from .dense import ResBlock
from .direct import RoPE2D
from .encoder import mlp
from .grid_latent import MapDecoder
from .physics_data import (
    C_DYNAMIC,
    C_LIVE,
    C_STATIC,
    CODE,
    NEUTRAL,
    PATCH,
    PROD,
    QUEUE,
    RAW_NUMS,
    TOP,
)
from .tokenize import C_GROUPS, N_BUFF_IDS, N_EFFECT_IDS, N_RAW_ABILITY_IDS, PC_KEYS

I = {n: i for i, n in enumerate(RAW_NUMS)}
C_PLY = 13  # player numbers: PC_KEYS and the camera's x, y, known
ROW_NUMS = len(RAW_NUMS) + 8  # embedded numbers per row, see row_numbers
N_FLAGS = 16  # flags, flags2 bits
MAX_BIRTHS, MARGIN = 64, 2  # birth count classes; extra queries per frame
POS_SCALE = 0.29  # a fast unit's per-loop move, map units
POS_BOOST = 4.0
DISCOUNT = 0.05  # critters' whole loss and larvae's position
GRID = 16  # the grid's cells per side
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


def rotate(x, cs):
    """x [B, H, T, dh], cs [B, T, dh / 2, 2] from cos_sin."""
    c, s = cs[..., 0][:, None].to(x.dtype), cs[..., 1][:, None].to(x.dtype)
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


def row_numbers(b):
    """[B, F, R, ROW_NUMS] scaled numbers of the rows."""
    n = b["num"]
    g = lambda k: n[..., I[k]]
    hpm, shm, bm = g("hp_max"), g("shield_max"), g("buff_max")
    log = lambda x, s=8.0: torch.log1p(x.clamp(min=0)) / s
    out = [
        (g("hp") / hpm.clamp(min=1)) * (hpm > 0),
        (g("shield") / shm.clamp(min=1)) * (shm > 0),
        g("energy") / 200,
        log(hpm),
        log(shm),
        g("build"),
        log(g("contents")),
        g("harvesters") / 16,
        g("ideal_harvesters") / 16,
        g("facing").sin(),
        g("facing").cos(),
        g("radius") / 4,
        g("weapon_cooldown").clamp(min=0) / 32,
        g("cargo_used") / 8,
        g("cargo_max") / 8,
        (g("buff_remain") / bm.clamp(min=1)) * (bm > 0),
        log(bm),
        g("detect_range") / 12,
        g("radar_range") / 32,
        g("atk_up") / 3,
        g("arm_up") / 3,
        g("shd_up") / 3,
        g("add_on") >= 0,
        g("engaged") >= 0,
        g("z") / 16,
    ]
    out += [(g("display") == k) for k in (1, 2, 3)] + [
        (g("cloak") == k) for k in (0, 1, 2, 3)
    ]
    out += [b["inside"] == 1, b["inside"] == 2, ~b["alive"], b["parent_row"] >= 0]
    return torch.stack([x.to(n.dtype) for x in out], -1)


def attend_pairs(q, k, v, mask, causal=False, code=None, rowmap=None, bias=None):
    """One softmax per query over the keys mask [B, K] allows. causal: only keys at or
    before the query. code [P, Q, K] and rowmap [B] add bias [3, H] to a pair's score.
    FlexAttention when training; without gradient SDPA, since FlexAttention's
    no-gradient path fails to compile (quad_latent.attend).
    """
    if not torch.is_grad_enabled():
        Q, K = q.shape[2], k.shape[2]
        allow = mask[:, None, None]
        if causal:
            allow = allow & torch.tril(
                torch.ones(Q, K, dtype=torch.bool, device=mask.device)
            )
        r = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
        if code is not None:
            rows = torch.nonzero(rowmap >= 0, as_tuple=True)[0]
            add = (
                bias[code[rowmap[rows]].long()].permute(0, 3, 1, 2).to(q.dtype)
            )  # [P, H, Q, K]
            am = add.masked_fill(~allow[rows], -float("inf"))
            r[rows] = F.scaled_dot_product_attention(
                q[rows], k[rows], v[rows], attn_mask=am
            )
        return r

    def score_mod(score, b, hh, qi, ki):
        s = torch.where(mask[b, ki], score, -float("inf"))
        if causal:
            s = torch.where(ki <= qi, s, -float("inf"))
        if code is not None:
            r = rowmap[b]
            s = s + torch.where(
                r >= 0, bias[code[r.clamp(min=0), qi, ki].long(), hh], 0.0
            )
        return s

    return flex_attention(q, k, v, score_mod=score_mod)


class Attend(nn.Module):
    """One softmax per query over the keys a mask allows: the part that compiles."""

    def forward(self, q, k, v, mask):
        return attend_pairs(q, k, v, mask)


class PairAttend(nn.Module):
    """The same, with the action pointers' bias on a pair's score."""

    def forward(self, q, k, v, mask, code, rowmap, bias):
        return attend_pairs(q, k, v, mask, code=code, rowmap=rowmap, bias=bias)


class Heads(nn.Module):
    """Projections to 8 query heads and 4 key and value heads, each shared by 2 query
    heads. Queries and keys are normalized, with a learned scale."""

    def __init__(self, d, heads, kv_heads):
        super().__init__()
        self.H, self.KV, self.dh = heads, kv_heads, d // heads
        self.q, self.k, self.v = (
            nn.Linear(d, d, bias=False),
            nn.Linear(d, kv_heads * self.dh, bias=False),
            nn.Linear(d, kv_heads * self.dh, bias=False),
        )
        self.o = nn.Linear(d, d, bias=False)
        self.scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def _kv(self, x, layer):
        B, T = x.shape[:2]
        return (
            layer(x)
            .view(B, T, self.KV, self.dh)
            .repeat_interleave(self.H // self.KV, 2)
            .transpose(1, 2)
        )

    def query(self, x, cs=None):
        B, T = x.shape[:2]
        q = self.q(x).view(B, T, self.H, self.dh).transpose(1, 2)
        q = rotate(q, cs) if cs is not None else q
        return (
            F.normalize(q, dim=-1) * self.scale.exp().to(q.dtype) * math.sqrt(self.dh)
        )

    def key(self, x, cs=None):
        k = self._kv(x, self.k)
        k = rotate(k, cs) if cs is not None else k
        return F.normalize(k, dim=-1)

    def value(self, x):
        return self._kv(x, self.v)

    def out(self, a):
        """[B, H, T, dh] -> [B, T, d]."""
        return self.o(a.transpose(1, 2).flatten(2))


class FeedForward(nn.Module):
    """Pre-norm SwiGLU, added to its input."""

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
        return x + self.w3(F.silu(self.w1(y)) * self.w2(y))


class Block(nn.Module):
    """Spatial attention and a feed-forward, temporal attention and a feed-forward, then
    this frame's actions as a change kept apart from the state. The action timeline has
    weights of its own: action tokens and entity tokens are different kinds of sequence.
    """

    def __init__(self, d, heads, kv_heads, ff):
        super().__init__()
        self.ns, self.nt, self.na, self.nat = (
            nn.LayerNorm(d),
            nn.LayerNorm(d),
            nn.LayerNorm(d),
            nn.LayerNorm(d),
        )
        self.space, self.time, self.act, self.act_time = (
            Heads(d, heads, kv_heads) for _ in range(4)
        )
        self.ff_space, self.ff_time, self.ff_act = (
            FeedForward(d, ff),
            FeedForward(d, ff),
            FeedForward(d, ff),
        )
        self.ent, self.pair = Attend(), PairAttend()
        self.bias = nn.Parameter(torch.zeros(3, heads))
        nn.init.zeros_(
            self.act.o.weight
        )  # starts as the identity: the updated tokens start as the state

    def spatial(self, x, c):
        """All tokens of a frame read each other, entities and map cells in one
        softmax."""
        y = self.ns(x)
        a = self.ent(
            self.space.query(y, c["ang"]),
            self.space.key(y, c["ang"]),
            self.space.value(y),
            c["valid"],
        )
        return x + self.space.out(a)

    def temporal(self, x, key, mask, ang, heads, norm):
        """x, key [N, T, d], mask [N, T], ang [N, T, dh / 2, 2] the cosines and sines of
        each frame's rotation by its game loop: each token reads the keys at or before
        its frame.
        A token with nothing to read keeps its own frame's key: an empty softmax is NaN.
        """
        N, T, d = x.shape
        y, ky = norm(x), norm(key)
        tri = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        allow = (tri[None] & mask[:, None]) | torch.eye(
            T, dtype=torch.bool, device=x.device
        )[None]
        a = F.scaled_dot_product_attention(
            heads.query(y, ang),
            heads.key(ky, ang),
            heads.value(ky),
            attn_mask=allow[:, None],
        )
        return torch.where(mask[..., None], x + heads.out(a), x)

    def actions(self, x, dl, a, c):
        """Frames with an action: actions read the state, not each other (a player's own
        earlier actions reach it through its causal timeline, and the two players'
        actions of one frame are simultaneous), then the state reads the actions; x +
        the returned dl is the updated state. dl [BF, T, d], a [BF, A, d]."""
        fa = c["fa"]
        y, ya = self.na(x[fa] + dl[fa]), self.na(a[fa])
        q, k, v = (
            self.act.query(y, c["ang"][fa]),
            self.act.key(y, c["ang"][fa]),
            self.act.value(y),
        )
        qa, ka, va = (
            self.act.query(ya, c["ang_a"][fa]),
            self.act.key(ya, c["ang_a"][fa]),
            self.act.value(ya),
        )
        bias = (self.bias * torch.tensor([0.0, 1.0, 1.0], device=x.device)[:, None]).to(
            q.dtype
        )  # code 0: no pair, no bias
        am = c["act_mask"][fa]
        ra = self.pair(qa, k, v, c["valid"][fa], c["code_ae"], c["rowmap"], bias)
        r = self.pair(q, ka, va, am, c["code_ea"], c["rowmap"], bias)
        a, dl = a.clone(), dl.clone()
        a[fa] = torch.where(am[..., None], a[fa] + self.act.out(ra), a[fa]).to(a.dtype)
        dl[fa] = (dl[fa] + self.act.out(r)).to(dl.dtype)
        return dl, a

    def forward(self, x, dl, a, c, last):
        B, F_, d = c["B"], c["F"], x.shape[-1]
        x = self.ff_space(self.spatial(x, c))
        # timelines: earlier frames' tokens as the last block's actions left them.
        # Effects have none: masked, they come out as they went in
        T = x.shape[1]
        by_token = lambda z: z.view(B, F_, T, d).transpose(1, 2).reshape(B * T, F_, d)
        xr, up, mask = by_token(x), by_token(x + dl), c["t_mask"]
        key = torch.cat(
            [xr[:, :1], torch.where(mask[:, :-1, None], up[:, :-1], xr[:, 1:])], 1
        )
        x = (
            self.temporal(xr, key, mask, c["t_ang"], self.time, self.nt)
            .view(B, T, F_, d)
            .transpose(1, 2)
            .reshape(B * F_, T, d)
            .to(x.dtype)
        )
        x = self.ff_time(x)
        if last:
            return x, dl, a
        a = self._act_timeline(a, c)
        dl, a = self.actions(x, dl, a, c)
        return x, dl, a

    def _act_timeline(self, a, c):
        """Each player's action tokens over the window, causal, in time order: its own
        slots of each frame, frame after frame, so a timeline is a reshape. a [BF, 2K,
        d]: player 0's K slots, then player 1's."""
        B, F_, d = c["B"], c["F"], a.shape[-1]
        K, m = a.shape[1] // 2, c["a_mask"]
        if not m.any():
            return a
        seq = a.view(B, F_, 2, K, d).transpose(1, 2).reshape(B * 2, F_ * K, d)
        y = torch.where(
            m[..., None],
            self.ff_act(
                self.temporal(seq, seq, m, c["a_ang"], self.act_time, self.nat)
            ),
            seq,
        )
        return (
            y.view(B, 2, F_, K, d).transpose(1, 2).reshape(B * F_, 2 * K, d).to(a.dtype)
        )


class Tower(nn.Module):
    """A plain ResNet over the live map at its own size: 3 strides, doubling to the
    model's width at an eighth of the resolution."""

    def __init__(self, c_in, d, blocks=1):
        super().__init__()
        c = [d // 4, d // 2, d]
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


class PhysicsEncoder(nn.Module):
    """The whole sequence through the tower and the blocks. Returns rows, cells and
    globals twice, the clean state and the action-updated tokens."""

    def __init__(
        self, vocab, avocab, d=512, blocks=6, heads=8, kv_heads=4, ff=1365, n_maps=1
    ):
        super().__init__()
        self.d = d
        self.unit = nn.Embedding(vocab.n_units, d)
        self.ability = nn.Embedding(vocab.n_abilities, d)
        self.owner = nn.Embedding(3, d)
        self.num_in = mlp(ROW_NUMS, d)
        self.groups = nn.Linear(C_GROUPS, d)
        self.buff = nn.Embedding(N_BUFF_IDS, d)
        self.bag_in = nn.ModuleDict(
            dict(ord=mlp(4, d), prod=mlp(2, d), rally=mlp(3, d))
        )
        self.bag_slot = nn.ModuleDict(
            dict(ord=nn.Embedding(QUEUE, d), prod=nn.Embedding(PROD, d))
        )  # which place in the queue: 0 is now
        # pointers as inputs: each row's random identity code, and the codes of the rows
        # it points at, each kind through its own map
        self.code_in = nn.ModuleDict(
            {
                k: nn.Linear(CODE, d, bias=False)
                for k in ("self", "engaged", "container", "order")
            }
        )
        self.raw_ability = nn.Embedding(N_RAW_ABILITY_IDS, d)
        self.auto_on = nn.Embedding(2, d)
        self.race = nn.Embedding(4, d)
        self.ply_in = mlp(C_PLY, d)
        self.player = nn.Embedding(2, d)
        self.cls = nn.Parameter(torch.zeros(1, d))
        self.view_in = nn.Linear(
            2, d
        )  # which players a sample's state is seen by: [1, 1] for physics
        self.map_wh = nn.Linear(2, d)
        # actions
        self.a_kind, self.a_family = nn.Embedding(N_ACT_KINDS, d), nn.Embedding(
            avocab.n_families, d, padding_idx=0
        )
        self.a_button, self.a_tkind = nn.Embedding(
            avocab.n_buttons, d, padding_idx=0
        ), nn.Embedding(4, d)
        self.a_group, self.a_gop = nn.Embedding(
            N_GROUPS, d, padding_idx=0
        ), nn.Embedding(N_GROUP_OPS, d, padding_idx=0)
        self.a_remove = nn.Embedding(N_REMOVE, d, padding_idx=0)
        self.a_subgroup = nn.Embedding(N_SUBGROUPS, d, padding_idx=0)
        self.a_num = mlp(C_ACTION, d)
        self.a_mix = mlp(d, d)
        # effects: no ids, so tokens of the frame they're in, without a timeline
        self.effect = nn.Embedding(N_EFFECT_IDS, d)
        self.eff_num = nn.Linear(1, d)
        # map
        self.tower = Tower(C_LIVE + C_STATIC, d)
        self.map_tokens = nn.Parameter(
            torch.zeros(n_maps, GRID * GRID, d)
        )  # what each place of each map is, whatever is on it now
        self.rope = RoPE2D(d // heads, lo=ROPE_SHORTEST, hi=256.0)
        self.rope_t = RoPE1D(d // heads)
        self.blocks = nn.ModuleList(
            Block(d, heads, kv_heads, ff) for _ in range(blocks)
        )
        self.norm_e, self.norm_c, self.norm_g = (
            nn.LayerNorm(d),
            nn.LayerNorm(d),
            nn.LayerNorm(d),
        )
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.map_tokens, std=0.02)
        self.grad_ckpt = False

    def compile_parts(self):
        """The masked attention. Each part sees several sizes, so it needs more than the
        default limit of 8 compiled versions."""
        torch._dynamo.config.recompile_limit = 256
        for m in self.modules():
            if type(m).__name__ in ("Attend", "PairAttend"):
                m.compile(dynamic=True)

    def _bag(self, name, b, emb):
        """Mean of a list-valued state per token: [B * F * R] flat index, embedded
        items."""
        where, B, F_, R = b[f"{name}_at"], *b["mask"].shape
        out = emb.new_zeros(B * F_ * R, emb.shape[-1]).index_add(0, where, emb)
        cnt = (
            torch.bincount(where, minlength=B * F_ * R)
            .clamp(min=1)[:, None]
            .to(emb.dtype)
        )
        return (out / cnt).view(B, F_, R, -1)

    def rows(self, b):
        B, F_, R = b["mask"].shape
        code = b["row_code"]  # [B, R, CODE]
        dt = b["num"].dtype

        def pointed(bi, row):
            """The code of the row pointed at, zeros where row < 0 (no target)."""
            return torch.where((row >= 0)[..., None], code[bi, row.clamp(min=0)], 0).to(
                dt
            )

        bb = torch.arange(B, device=code.device)[:, None, None]
        ids = (
            self.unit(b["type"].long())
            + self.owner(b["owner"])[:, None]
            + self.num_in(row_numbers(b))
            + self.groups(b["groups"].to(dt))
        )
        ids = (
            ids
            + self.code_in["self"](code.to(dt))[:, None]
            + self.code_in["engaged"](pointed(bb, b["engaged_row"].long()))
            + self.code_in["container"](pointed(bb, b["parent_row"].long()))
        )
        o_b = b["ord_at"] // (F_ * R)
        ids = ids + self._bag(
            "ord",
            b,
            self.ability(b["ord_abil"])
            + self.bag_in["ord"](b["ord_feat"])
            + self.bag_slot["ord"](b["ord_slot"])
            + self.code_in["order"](pointed(o_b, b["ord_trow"])),
        )
        ids = ids + self._bag(
            "prod",
            b,
            self.ability(b["prod_abil"])
            + self.bag_in["prod"](b["prod_feat"])
            + self.bag_slot["prod"](b["prod_slot"]),
        )
        ids = ids + self._bag("buff", b, self.buff(b["buff_id"]))
        ids = ids + self._bag("rally", b, self.bag_in["rally"](b["rally_feat"]))
        ids = ids + self._bag("ab", b, self.raw_ability(b["ab_id"]))
        ids = ids + self._bag(
            "auto",
            b,
            self.a_family(b["auto_fam"])
            + self.a_button(b["auto_but"])
            + self.auto_on(b["auto_on"]),
        )
        return ids

    def globals(self, b):
        B, F_ = b["mask"].shape[:2]
        p = (
            self.ply_in(b["ply_num"])
            + self.race(b["players"])[:, None]
            + self.player.weight[None, None]
        )
        c = (self.cls + self.view_in(b["view"]) + self.map_wh(b["half"] / 100))[
            :, None
        ].expand(B, F_, -1)
        return torch.cat([c[:, :, None], p], 2)  # [B, F, 3, d]

    def actions(self, b):
        n = (
            self.a_kind(b["act_kind"])
            + self.a_family(b["act_family"])
            + self.a_button(b["act_button"])
            + self.a_tkind(b["act_tkind"])
            + self.a_group(b["act_group"])
            + self.a_gop(b["act_gop"])
            + self.a_remove(b["act_remove"])
            + self.a_subgroup(b["act_subgroup"])
            + self.a_num(b["act_num"])
            + self.player(b["act_player"])
        )
        return self.a_mix(n)

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

    def embed_map(self, b):
        """Each frame's map tokens [BF, S, d]: the map's learned tokens + the tower over
        the live map, read at their anchors."""
        B, F_ = b["mask"].shape[:2]
        dt = self.cls.dtype
        st = b["static_grid"].to(dt)
        st = torch.cat([st[..., :2] / 255, st[..., 2:]], -1)
        raw = (
            torch.cat([b["grid"].to(dt), st[:, None].expand(B, F_, -1, -1, -1)], -1)
            .reshape(B * F_, *b["grid"].shape[2:4], C_LIVE + C_STATIC)
            .permute(0, 3, 1, 2)
        )
        top = self.tower(raw)
        H, W = raw.shape[2:]
        at = (self.anchors(b) + (b["center"] - b["origin"])[:, None]) / torch.tensor(
            [W, H], device=raw.device, dtype=raw.dtype
        ) * 2 - 1
        at = at.repeat_interleave(F_, 0).view(B * F_, GRID, GRID, 2)
        live = (
            F.grid_sample(top, at.to(top.dtype), align_corners=False)
            .flatten(2)
            .transpose(1, 2)
        )
        return self.map_tokens[b["map_id"]].repeat_interleave(F_, 0) + live

    def forward(self, b, last_action=True):
        """last_action False skips the last block's action step, when only the clean
        state is wanted: the updated tokens returned are then a block old."""
        from torch.utils.checkpoint import checkpoint

        B, F_, R = b["mask"].shape
        dev = b["mask"].device
        d = self.d
        A, X = b["act_mask"].shape[2], b["eff_mask"].shape[2]
        g0 = 3
        ckpt = self.grad_ckpt and any(p.requires_grad for p in self.parameters())
        run = (
            (lambda f, *a: checkpoint(f, *a, use_reentrant=False))
            if ckpt
            else (lambda f, *a: f(*a))
        )
        cells = run(self.embed_map, b)
        G = self.globals(b).reshape(B * F_, g0, d)
        rows = (self.rows(b) * b["mask"][..., None]).reshape(B * F_, R, d)
        eff = (
            self.effect(b["eff_type"])
            + self.owner(b["eff_owner"])
            + self.eff_num(b["eff_radius"][..., None])
        ).reshape(B * F_, X, d)
        x = torch.cat([G, rows, eff, cells], 1)
        n = g0 + R + X
        T = n + GRID * GRID
        acts = self.actions(b).reshape(B * F_, A, d)
        valid = torch.cat(
            [
                torch.ones(B, F_, g0, dtype=torch.bool, device=dev),
                b["mask"],
                b["eff_mask"],
                torch.ones(B, F_, GRID * GRID, dtype=torch.bool, device=dev),
            ],
            2,
        ).reshape(B * F_, T)
        cen = self.anchors(b)
        rel = torch.stack([b["x"], b["y"]], -1) - b["center"][:, None, None]
        pos = torch.cat(
            [
                torch.zeros(B, F_, g0, 2, device=dev),
                rel,
                b["eff_xy"],
                cen[:, None].expand(-1, F_, -1, -1),
            ],
            2,
        ).reshape(B * F_, T, 2)
        # tokens without a place sit at the origin, unrotated
        t_mask = (
            torch.cat(
                [
                    torch.ones(B, F_, g0, dtype=torch.bool, device=dev),
                    b["mask"],
                    torch.zeros_like(b["eff_mask"]),
                    torch.ones(B, F_, GRID * GRID, dtype=torch.bool, device=dev),
                ],
                2,
            )
            .transpose(1, 2)
            .reshape(B * T, F_)
        )
        # time: each frame's game loop from the window's first; without loops (older
        # batches), a loop a frame
        loops = (
            b["loops"] - b["loops"][:, :1]
            if "loops" in b
            else torch.arange(F_, device=dev)[None].expand(B, -1)
        )
        t_ang = self.rope_t.angles(loops)  # [B, F, dh / 2]
        K, h2 = A // 2, t_ang.shape[-1]
        t_cs = cos_sin(t_ang)  # [B, F, dh / 2, 2]
        act_mask = b["act_mask"].reshape(B * F_, A)
        fa = torch.nonzero(act_mask.any(1), as_tuple=True)[0]
        # every rotation as cosines and sines, once: 6 blocks of spatial, temporal and
        # action layers all rotate by these
        c = dict(
            B=B,
            F=F_,
            valid=valid,
            ang=cos_sin(self.rope.angles(pos)),
            ang_a=cos_sin(self.rope.angles(b["act_xy"].reshape(B * F_, A, 2))),
            act_mask=act_mask,
            t_mask=t_mask,
            t_ang=t_cs[:, None].expand(B, T, F_, h2, 2).reshape(B * T, F_, h2, 2),
            a_mask=b["act_mask"]
            .view(B, F_, 2, K)
            .transpose(1, 2)
            .reshape(B * 2, F_ * K),
            a_ang=t_cs.repeat_interleave(K, 1)[:, None]
            .expand(B, 2, F_ * K, h2, 2)
            .reshape(B * 2, F_ * K, h2, 2),
        )
        if len(fa):
            fa = fa if len(fa) > 1 else fa.repeat(2)
            c["fa"] = fa
            c["code_ae"], c["code_ea"], c["rowmap"] = pointer_pairs(
                b["act_target"].reshape(B * F_, A), fa, A, T, g0
            )
        dl = torch.zeros_like(x)
        for i, blk in enumerate(self.blocks):
            last = (not last_action and i == len(self.blocks) - 1) or not len(fa)
            x, dl, acts = run(blk, x, dl, acts, c, last)
        out = lambda z: dict(
            ent=self.norm_e(z[:, g0 : g0 + R]).view(B, F_, R, d),
            cells=self.norm_c(z[:, n:]).view(B, F_, GRID * GRID, d),
            glob=self.norm_g(z[:, :g0]).view(B, F_, g0, d),
        )
        return out(x), out(x + dl)


@torch.no_grad()
def pointer_pairs(target, fa, A, n, g0):
    """For the frames fa with actions, n state tokens: code tables of action -> target
    row (1) over the state, [P, A, n], and target row -> action (2), [P, n, A], and
    rowmap [len(fa)] -> P or -1. A table always exists, so a compiled part sees one
    shape of call.
    """
    t = target[fa]
    fi, ai = torch.nonzero(t >= 0, as_tuple=True)
    if not len(fi):
        return (
            torch.zeros(1, A, n, dtype=torch.uint8, device=t.device),
            torch.zeros(1, n, A, dtype=torch.uint8, device=t.device),
            torch.full((len(fa),), -1, device=t.device),
        )
    uq, inv = torch.unique(fi, return_inverse=True)
    rowmap = torch.full((len(fa),), -1, device=t.device)
    rowmap[uq] = torch.arange(len(uq), device=t.device)
    ae, ea = torch.zeros(
        len(uq), A, n, dtype=torch.uint8, device=t.device
    ), torch.zeros(len(uq), n, A, dtype=torch.uint8, device=t.device)
    row = t[fi, ai] + g0
    ae[inv, ai, row] = 1
    ea[inv, row, ai] = 2
    return ae, ea, rowmap


class RowHead(nn.Module):
    """A row's token -> its raw fields at some frame. One head, shared by rows and birth
    queries."""

    def __init__(self, d, n_units, n_abilities):
        super().__init__()
        self.unit, self.inside = nn.Linear(d, n_units), nn.Linear(d, 3)
        self.alive = nn.Linear(d, 1)
        self.reg = nn.Linear(d, 2 + len(REG))
        self.flags, self.display, self.cloak = (
            nn.Linear(d, N_FLAGS),
            nn.Linear(d, 4),
            nn.Linear(d, 4),
        )
        self.slot = nn.Embedding(QUEUE + PROD, d)
        self.slot_mix = mlp(d, d)
        self.slot_abil = nn.Linear(d, n_abilities)
        self.slot_reg = nn.Linear(
            d, 4
        )  # order target dx, dy, whether it has one; progress
        self.buffs, self.groups = nn.Linear(d, N_BUFF_IDS), nn.Linear(d, C_GROUPS)
        self.bits = nn.Linear(d, 3)  # has an add-on, an engaged target, a rally point
        self.rally = nn.Linear(d, 2)
        self.abilities = nn.Linear(d, N_RAW_ABILITY_IDS)
        self.ptr_q = nn.Embedding(
            2 + QUEUE, d
        )  # engaged target, container, each order's target
        self.ptr_mix, self.ptr_k = mlp(d, d), nn.Linear(d, d)

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

    def slots(self, x):
        y = self.slot_mix(x[:, None] + self.slot.weight[None])
        return self.slot_abil(y), self.slot_reg(y)

    def pointers(self, x, rows):
        """x [N, d] tokens, rows [N, R, d] their frames' rows -> [N, 2 + QUEUE, R]
        scores of each pointer over the rows."""
        q = self.ptr_mix(x[:, None] + self.ptr_q.weight[None])
        return torch.einsum("npd,nrd->npr", q, self.ptr_k(rows)) / math.sqrt(
            x.shape[-1]
        )


# regressed as change, scaled: (name, divisor)
REG = (
    ("hp", 100.0),
    ("shield", 100.0),
    ("energy", 50.0),
    ("build", 0.1),
    ("contents", 100.0),
    ("harvesters", 4.0),
    ("ideal_harvesters", 4.0),
    ("hp_max", 100.0),
    ("shield_max", 100.0),
    ("weapon_cooldown", 8.0),
    ("cargo_used", 4.0),
    ("buff_remain", 50.0),
    ("facing", 1.0),
    ("radius", 0.5),
    ("cargo_max", 4.0),
    ("buff_max", 50.0),
    ("detect_range", 4.0),
    ("radar_range", 8.0),
    ("atk_up", 1.0),
    ("arm_up", 1.0),
    ("shd_up", 1.0),
    ("z", 1.0),
)


class BirthHead(nn.Module):
    """Per frame, how many units appear next, then that many queries plus a margin, in
    parallel: each reads the frame's rows and says real or empty, with a new row's
    fields and a pointer to its parent."""

    def __init__(self, d, heads, head):
        super().__init__()
        self.count = mlp(d, MAX_BIRTHS)
        self.query = nn.Embedding(MAX_BIRTHS + MARGIN, d)
        self.H, self.dh = heads, d // heads
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.n1, self.n2, self.ff = nn.LayerNorm(d), nn.LayerNorm(d), mlp(d, d, 4 * d)
        self.real, self.pos = nn.Linear(d, 1), nn.Linear(d, 2)
        self.parent_q, self.parent_k, self.no_parent = (
            nn.Linear(d, d),
            nn.Linear(d, d),
            nn.Linear(d, 1),
        )
        self.head = [head]

    def forward(self, cls, rows, rmask, n):
        """cls [K, d], rows [K, R, d], rmask [K, R], n queries -> outputs per (frame,
        query)."""
        K, R, d = rows.shape
        x = cls[:, None] + self.query.weight[:n][None]
        k, v = self.kv(rows).view(K, R, 2, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q = self.q(self.n1(x)).view(K, n, self.H, self.dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=rmask[:, None, None])
        x = x + self.o(a.transpose(1, 2).reshape(K, n, d))
        x = x + self.ff(self.n2(x))
        parent = torch.einsum(
            "kqd,krd->kqr", self.parent_q(x), self.parent_k(rows)
        ) / math.sqrt(d)
        parent = parent.masked_fill(~rmask[:, None], -1e4)
        parent = torch.cat([parent, self.no_parent(x)], -1)
        return x, self.real(x).squeeze(-1), self.pos(x), parent


class Physics(nn.Module):
    """Encoder, EMA teacher, residual head, raw heads, births. forward(batch) -> (loss,
    logs)."""

    def __init__(
        self,
        vocab,
        avocab,
        d=512,
        ema=0.996,
        demean=True,
        proj=1024,
        moving_neutrals=(),
        larva=151,
        **kw,
    ):
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
        self.births = BirthHead(d, kw.get("heads", 8), self.head)
        self.map_dec = MapDecoder(d, C_DYNAMIC + N_EFFECT_IDS)
        self.player_next = nn.Linear(d, len(PC_KEYS))
        self.ema, self.demean = ema, demean
        self.register_buffer(
            "critter",
            torch.tensor(sorted(moving_neutrals) or [-1], dtype=torch.int64),
            persistent=False,
        )
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
        """Barlow Twins between predicted and teacher tokens at (sample, row) pairs;
        correlations after removing each row's mean over time if demean."""
        zp, zt = self.proj(p).float(), self.proj(t).float()
        if self.demean:
            n = int(rows_t.max()) + 1
            cnt = torch.bincount(rows_t, minlength=n).clamp(min=1)[:, None].to(zp.dtype)
            sub = (
                lambda z: z
                - (
                    torch.zeros(
                        n, z.shape[1], device=z.device, dtype=z.dtype
                    ).index_add(0, rows_t, z)
                    / cnt
                )[rows_t]
            )
            zp, zt = sub(zp), sub(zt)
        zp = (zp - zp.mean(0)) / (zp.std(0) + 1e-4)
        zt = (zt - zt.mean(0)) / (zt.std(0) + 1e-4)
        c = zp.T @ zt / len(zp)
        on = (torch.diagonal(c) - 1).pow(2).sum()
        off = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()
        return (on + lam * off) / zp.shape[1], c.detach()

    def forward(self, b):
        _, out = self.enc(b)
        with torch.no_grad():
            tgt, _ = self.teacher(b, last_action=False)
        B, F_, R = b["mask"].shape
        logs = {}
        # who is scored: alive with a next frame; dead rows score nothing
        sc = b["mask"][:, :-1] & b["alive"][:, :-1] & b["mask"][:, 1:]
        nxt = {
            k: v[:, :-1] for k, v in out.items()
        }  # frame t's action-updated tokens: everything next-state reads this
        self._rows = nxt["ent"]  # what pointers point into
        s = nxt["ent"]
        p = s + self.g(s)
        bi, fi, ri = torch.nonzero(sc, as_tuple=True)
        pp, tt = p[bi, fi, ri], tgt["ent"][:, 1:][bi, fi, ri]
        row_key = bi * R + ri
        uq, inv = torch.unique(row_key, return_inverse=True)
        lat, corr = self.barlow(pp, tt, inv)
        logs["barlow"] = lat
        with torch.no_grad():
            logs["latent_change"] = (
                (tgt["ent"][:, 1:][bi, fi, ri] - tgt["ent"][:, :-1][bi, fi, ri])
                .pow(2)
                .sum(-1)
                .mean()
            )
        raw, raw_logs = self.raw_losses(p, s, b, sc, bi, fi, ri)
        logs.update(raw_logs)
        brth, blogs = self.birth_losses(nxt, b)
        logs.update(blogs)
        mp = self.map_loss(nxt, b)
        logs["map"] = mp
        pn = F.smooth_l1_loss(
            self.player_next(nxt["glob"][:, :, 1:3]).float(),
            (b["ply_num"][:, 1:] - b["ply_num"][:, :-1])[..., : len(PC_KEYS)] * 100,
        )
        logs["player"] = pn
        loss = lat + self.aux_weight * (raw + brth + mp + pn)
        logs["loss"] = loss
        self._rows = None  # it would hold this pass's graph on the model until the next
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
        logs["pos_copy"] = (
            F.smooth_l1_loss(torch.zeros_like(tpos), tpos, reduction="none").sum(-1)
            * pos_w
        ).sum() / pos_w.sum()
        if self.record_pos:
            self.pos_eval = self._position_errors(
                o["reg"][:, :2].float(),
                tpos,
                own,
                crit,
                larva,
                b["alive"][bi, fi + 1, ri]
                & (b["inside"][bi, fi, ri] == 0)
                & (b["inside"][bi, fi + 1, ri] == 0),
            )
        reg_t = torch.stack(
            [(n[bi, fi + 1, ri, I[k]] - n[bi, fi, ri, I[k]]) / dv for k, dv in REG], -1
        )
        lr = F.smooth_l1_loss(o["reg"][:, 2:].float(), reg_t, reduction="none").mean(-1)
        logs["reg"] = (lr * whole).sum() / whole.sum()
        logs["type"] = (
            F.cross_entropy(o["type"].float(), t_type, reduction="none") * whole
        ).sum() / whole.sum()
        logs["inside"] = (
            F.cross_entropy(
                o["inside"].float(),
                b["inside"][bi, fi + 1, ri].long(),
                reduction="none",
            )
            * whole
        ).sum() / whole.sum()
        flags_t = n[bi, fi + 1, ri, I["flags"]].long() | (
            n[bi, fi + 1, ri, I["flags2"]].long() << 8
        )
        bits = (
            (flags_t[:, None] >> torch.arange(N_FLAGS, device=p.device)) & 1
        ).float()
        logs["flags"] = (
            F.binary_cross_entropy_with_logits(
                o["flags"].float(), bits, reduction="none"
            ).mean(-1)
            * whole
        ).sum() / whole.sum()
        logs["display"] = (
            F.cross_entropy(
                o["display"].float(),
                n[bi, fi + 1, ri, I["display"]].long().clamp(0, 3),
                reduction="none",
            )
            * whole
        ).sum() / whole.sum()
        logs["cloak"] = (
            F.cross_entropy(
                o["cloak"].float(),
                n[bi, fi + 1, ri, I["cloak"]].long().clamp(0, 3),
                reduction="none",
            )
            * whole
        ).sum() / whole.sum()
        # the alive field: scored on rows alive now, flips balanced against the many
        # that stay
        died = (~b["alive"][bi, fi + 1, ri]).float()
        pw = (len(died) - died.sum()).clamp(min=1) / died.sum().clamp(min=1)
        logs["alive"] = (
            F.binary_cross_entropy_with_logits(
                o["alive"].float(),
                died,
                pos_weight=pw.clamp(max=100.0),
                reduction="none",
            )
            .mul(whole)
            .sum()
            / whole.sum()
        )
        self._ptr_stats = None
        logs["queues"], logs["abilities"], logs["pointers"] = self._sampled(
            p, b, bi, fi, ri
        )
        if self._ptr_stats is not None:
            logs["ptr_acc"], logs["ptr_copy"] = self._ptr_stats
        # buffs, control groups and small presence bits: every scored token. Rare
        # positives are the point, so report them too
        key = (bi * F_ + fi + 1) * R + ri
        buffs_t = torch.zeros(
            len(bi), N_BUFF_IDS, device=p.device, dtype=o["buffs"].float().dtype
        )
        pos = torch.full((B * F_ * R,), -1, device=p.device)
        pos[key] = torch.arange(len(key), device=p.device)
        hit = pos[b["buff_at"]]
        ok = hit >= 0
        buffs_t[hit[ok], b["buff_id"][ok]] = 1
        wt = whole[:, None]
        logs["buffs"] = (
            (
                F.binary_cross_entropy_with_logits(
                    o["buffs"].float(),
                    buffs_t,
                    reduction="none",
                    pos_weight=torch.full((N_BUFF_IDS,), 20.0, device=p.device),
                )
                * wt
            ).sum()
            / wt.sum()
            / 32
        )
        logs["groups"] = (
            F.binary_cross_entropy_with_logits(
                o["groups"].float(),
                b["groups"][bi, fi + 1, ri].float(),
                reduction="none",
            ).mean(-1)
            * whole
        ).sum() / whole.sum()
        bits_t = torch.stack(
            [
                n[bi, fi + 1, ri, I["add_on"]] >= 0,
                b["engaged_row"][bi, fi + 1, ri] >= 0,
                b["rally_first"][bi, fi + 1, ri, 0] > 0,
            ],
            -1,
        ).float()
        logs["bits"] = (
            F.binary_cross_entropy_with_logits(
                o["bits"].float(), bits_t, reduction="none"
            ).mean(-1)
            * whole
        ).sum() / whole.sum()
        rm = b["rally_first"][bi, fi + 1, ri, 0] > 0
        logs["rally"] = (
            F.smooth_l1_loss(
                o["rally"][rm].float(), b["rally_first"][bi, fi + 1, ri, 1:3][rm]
            )
            if rm.any()
            else p.sum() * 0
        )
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

    def _position_errors(self, pred, tpos, own, crit, larva, keep):
        """For evaluation: {kind of row: (rows, sum of model error, sum of copy error)},
        map units per loop, the rows and kinds of scripts/physics_baselines.py."""
        step = tpos.norm(dim=-1) * POS_SCALE
        err = (pred - tpos).norm(dim=-1) * POS_SCALE
        neutral = own == NEUTRAL
        player = ~neutral & ~larva
        moving = step > 0.003
        kinds = {
            "player moving": player & moving,
            "player still": player & ~moving,
            "player all": player,
            "larva": larva & ~neutral,
            "critter": crit,
            "other neutral": neutral & ~crit,
        }
        return {
            k: np.array(
                [
                    float((v & keep).sum()),
                    float(err[v & keep].sum()),
                    float(step[v & keep].sum()),
                ]
            )
            for k, v in kinds.items()
        }

    def _sampled(self, p, b, bi, fi, ri, cap=4096, ptr_cap=1024):
        """The large outputs, on a sample of the scored tokens, rows whose queue changes
        first: order and production slots, available abilities, and pointers to the
        engaged target, the container and each order's target."""
        B, F_, R = b["mask"].shape
        key = (bi * (F_ - 1) + fi) * R + ri
        ch = b["queue_changed"].reshape(-1)[key]
        sel = torch.nonzero(ch, as_tuple=True)[0]
        rest = torch.nonzero(~ch, as_tuple=True)[0]
        take = rest[
            torch.randperm(len(rest), device=p.device)[: max(0, cap - len(sel))]
        ]
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
        queues = la + (
            F.smooth_l1_loss(reg.float(), t_reg, reduction="none") * m
        ).sum() / m.sum().clamp(min=1)
        # available abilities: the cooldowns the state otherwise hides
        pos = torch.full((B * F_ * R,), -1, device=p.device)
        pos[at] = torch.arange(len(at), device=p.device)
        hit = pos[b["ab_at"]]
        ok = hit >= 0
        t_ab = torch.zeros(len(sel), N_RAW_ABILITY_IDS, device=p.device)
        t_ab[hit[ok], b["ab_id"][ok]] = 1
        abilities = (
            F.binary_cross_entropy_with_logits(
                self.head.abilities(x).float(),
                t_ab,
                pos_weight=torch.full((N_RAW_ABILITY_IDS,), 20.0, device=p.device),
            )
            * 8
        )
        # pointers: scored where the next frame has a target row
        s2 = sel[torch.randperm(len(sel), device=p.device)[:ptr_cap]]
        rows = self._rows[bi[s2], fi[s2]]
        at2 = (bi[s2] * F_ + fi[s2] + 1) * R + ri[s2]
        tgt = torch.cat(
            [
                b["engaged_row"].reshape(-1)[at2][:, None],
                b["parent_row"].reshape(-1)[at2][:, None],
                b["ord_row"].reshape(-1, QUEUE)[at2],
            ],
            1,
        ).long()
        logits = self.head.pointers(p[bi[s2], fi[s2], ri[s2]], rows)
        # a target not live in this frame can't be pointed at: it would score the mask's
        # -1e4
        live = b["mask"][bi[s2], fi[s2]]
        logits = logits.masked_fill(~live[:, None], -1e4).float()
        has = (tgt >= 0) & live.gather(1, tgt.clamp(min=0))
        pointers = F.cross_entropy(logits[has], tgt[has]) if has.any() else p.sum() * 0
        # accuracy, and the baseline of keeping this frame's pointer: the same kind and
        # slot, a frame earlier
        with torch.no_grad():
            now = torch.cat(
                [
                    b["engaged_row"].reshape(-1)[at2 - R][:, None],
                    b["parent_row"].reshape(-1)[at2 - R][:, None],
                    b["ord_row"].reshape(-1, QUEUE)[at2 - R],
                ],
                1,
            ).long()
            self._ptr_stats = (
                (
                    (logits.argmax(-1) == tgt)[has].float().mean(),
                    (now == tgt)[has].float().mean(),
                )
                if has.any()
                else None
            )
        return queues, abilities, pointers

    def birth_losses(self, out, b):
        B, F_, R = b["mask"].shape
        d = out["ent"].shape[-1]
        n_b = b["birth_count"]  # [B, F - 1]: births first seen in frame f + 1
        cls = out["glob"][:, :, 0]
        logs = dict(
            birth_count=F.cross_entropy(
                self.births.count(cls).float().flatten(0, 1),
                n_b.flatten().clamp(max=MAX_BIRTHS - 1),
            )
        )
        rows, rmask = out["ent"], b["mask"][:, :-1]
        fb, ff = torch.nonzero(n_b >= 0, as_tuple=True)
        total_real, total_fields, frames = 0.0, 0.0, 0
        groups = {}
        for k, c in zip(range(len(fb)), n_b[fb, ff].tolist()):
            groups.setdefault(min(c, MAX_BIRTHS - 1), []).append(k)
        for c, idx in groups.items():
            idx = torch.tensor(idx, device=fb.device)
            sb, sf = fb[idx], ff[idx]
            x, real, pos, parent = self.births(
                cls[sb, sf], rows[sb, sf], rmask[sb, sf], c + MARGIN
            )
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
                    txy = (
                        torch.stack(
                            [
                                b["x"][sb[i], sf[i] + 1, rr],
                                b["y"][sb[i], sf[i] + 1, rr],
                            ],
                            -1,
                        )
                        - b["center"][sb[i]]
                    ) / b["half"][sb[i]]
                    lt = F.log_softmax(head["type"][i].float(), -1)
                    cost = torch.cdist(pos[i].float(), txy, p=1) - lt[:, tt]
                    qi, ti = linear_sum_assignment(cost.detach().cpu().numpy())
                    qi, ti = torch.as_tensor(qi, device=x.device), torch.as_tensor(
                        ti, device=x.device
                    )
                    target[i, qi] = 1
                    w = torch.where(
                        b["rtype"][sb[i], sf[i] + 1, rr][ti] == self.larva,
                        DISCOUNT,
                        1.0,
                    )  # a larva's position takes the discount, its birth doesn't
                    lpos = (
                        F.smooth_l1_loss(
                            pos[i, qi].float(), txy[ti], reduction="none"
                        ).sum(-1)
                        * w
                    ).mean()
                    par = b["birth_p"][tb][:c][ti]
                    par = torch.where(par >= 0, par, parent.shape[-1] - 1)
                    loss_f.append(
                        lpos
                        + F.cross_entropy(head["type"][i, qi].float(), tt[ti])
                        + F.cross_entropy(parent[i, qi].float(), par)
                    )
                if loss_f:
                    total_fields = total_fields + torch.stack(loss_f).sum()
                    frames += len(loss_f)
            total_real = total_real + F.binary_cross_entropy_with_logits(
                real.float(), target, reduction="sum"
            )
        logs["birth_real"] = total_real / max(1, len(fb)) / (1 + MARGIN)
        logs["birth_fields"] = (
            total_fields / max(1, frames) if frames else out["ent"].sum() * 0
        )
        return logs["birth_count"] + logs["birth_real"] + logs["birth_fields"], logs

    def map_loss(self, out, b, K=256):
        """Next-frame visibility, seen, creep and each effect type at sampled map
        tokens, read from the coarsest cells. Effects are rare, so the cells they cover
        are always among the sample (eff_force)."""
        B, F_ = b["mask"].shape[:2]
        S = out["cells"].shape[2]
        side = int(round(math.sqrt(S)))
        cells = out["cells"].reshape(B * (F_ - 1), S, -1)
        idx = torch.randint(0, TOP * TOP, (B * (F_ - 1), K), device=cells.device)
        force = b["eff_force"].reshape(B * (F_ - 1), -1)
        idx = torch.cat(
            [idx, torch.where(force >= 0, force, idx[:, : force.shape[1]])], 1
        )
        iy, ix = idx // TOP, idx % TOP
        r = TOP // side
        cell = (iy // r) * side + (ix // r)
        size = b["side"][:, None].expand(B, F_ - 1).reshape(-1, 1) / TOP
        off = torch.stack(
            [(ix % r + 0.5 - r / 2) * size, (iy % r + 0.5 - r / 2) * size], -1
        )
        o = self.map_dec(cells, cell, off)
        m = b["map"][:, 1:].reshape(B * (F_ - 1), TOP * TOP, C_DYNAMIC * PATCH * PATCH)
        centre = (PATCH * PATCH) // 2
        tgt = m[..., centre :: PATCH * PATCH][:, :, :C_DYNAMIC].float()
        tg = torch.gather(tgt, 1, idx[..., None].expand(-1, -1, C_DYNAMIC))
        te = torch.zeros(*idx.shape, N_EFFECT_IDS, device=idx.device, dtype=tg.dtype)
        if len(b["eff_cover"]):
            fr, ce, ty = b["eff_cover"].unbind(
                1
            )  # next frames' covered tokens: (flat frame, token, effect type)
            hit = idx[fr] == ce[:, None]
            pi, ki = torch.nonzero(hit, as_tuple=True)
            te[fr[pi], ki, ty[pi]] = 1
        return (
            F.binary_cross_entropy_with_logits(o[..., :C_DYNAMIC], tg)
            + F.binary_cross_entropy_with_logits(o[..., C_DYNAMIC:], te) * 4
        )
