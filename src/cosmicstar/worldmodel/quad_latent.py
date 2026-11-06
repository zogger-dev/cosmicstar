"""Fixed map pyramid (2026-11-01, DESIGN_grid_latent.md "Fixed pyramid"): one grid from the map down to
the latent, by halving the map: 128 x 128, 64, 32, 16 norm tokens over a square of the map's longest
playable side. The 16 x 16 level is the latent; there is no separate step onto it.

Three kinds of token share a level's attention:

    norm     the fixed grid; every frame has all of them
    lattice  the real map cells, in map units, over the same square: real cells at 1 unit, merged 2 x 2
             to 2 and 4 units on the way down; cells of the square outside the playable area are still
             tokens, their playable layer false (zeros where the cache has no terrain)
    entity   entities and effects, kept individually at every level

128, 64, 32: 3 ResNet blocks on the norm tokens, then 3 attention blocks. Each 4 x 4 window of norm
tokens attends jointly with the lattice and entity tokens whose centers fall inside it, and reads the
globals; the middle block's windows slide by 2 cells. Each global attends over all of that level's tokens,
so globals read and write at every layer. A 2 x 2 merge leads to the next level.

16: 3 ResNet blocks, then full attention over norm tokens, entities and globals (no lattice: too many
tokens). The latent is [CLS, players, 16 x 16 norm tokens], taken out after the last block.

Window sizes vary with clumps, so windows are padded by size group, never to the fullest window.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

from .dense import ResBlock, Down, WIN
from .direct import Block
from .encoder import mlp
from .grid_latent import GridEncoder, cell_of, geometry
from .pyramid import _rot

TOP = 128  # norm tokens per side at the finest level
SLIDE = WIN // 2
CAPS = (
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    4096,
)  # window tokens, by size group
MAX_SCORES = 1.5e8  # attention scores per head in one call: larger groups are split


def attend(q, k, v, mask):
    """One softmax per query over the keys mask [B, K] allows. FlexAttention when training (its backward is
    far faster: a third of the step went to SDPA's); without gradient SDPA, the same numbers (5e-7), since
    FlexAttention's no-gradient path fails to compile inside these parts."""
    if not torch.is_grad_enabled():
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None])

    def score_mod(score, b, hh, qi, ki):
        return torch.where(mask[b, ki], score, -float("inf"))

    return flex_attention(q, k, v, score_mod=score_mod)


def window_of(ij, S, slide):
    """ij [..., 2] (norm token x, y) -> window id; n ids per side (a slid grid has cut-short windows at the edges)."""
    n = (S + slide + WIN - 1) // WIN
    w = (ij + slide) // WIN
    return w[..., 1] * n + w[..., 0], n * n


@torch.no_grad()
def window_sets(win, member, n_win):
    """win [B, T] window ids, member [B, T] -> the windows grouped by size, as a list of (tok, dest, frame, m):
    flat token positions, their places in a rows x cap padded tensor, each row's frame, which places are real.
    """
    B, T = win.shape
    idx = torch.nonzero(member.reshape(-1), as_tuple=True)[0]
    w = (win + torch.arange(B, device=win.device)[:, None] * n_win).reshape(-1)[idx]
    uq, inv, cnt = torch.unique(w, return_inverse=True, return_counts=True)
    order = torch.argsort(inv, stable=True)
    tok, win_s = idx[order], inv[order]
    starts = torch.cat([cnt.new_zeros(1), cnt.cumsum(0)[:-1]])
    rank = torch.arange(len(tok), device=win.device) - starts[win_s]
    caps = torch.tensor(CAPS, device=win.device)
    assert int(cnt.max()) <= CAPS[-1], f"a window of {int(cnt.max())} tokens"
    group = torch.searchsorted(caps, cnt)
    out = []
    for g in torch.unique(group).tolist():
        ids = torch.nonzero(group == g, as_tuple=True)[0]
        cap = CAPS[g]
        for chunk in ids.split(max(1, int(MAX_SCORES / cap / (cap + 3)))):
            row = torch.full((len(uq),), -1, device=win.device)
            row[chunk] = torch.arange(len(chunk), device=win.device)
            sel = row[win_s] >= 0
            dest = row[win_s[sel]] * cap + rank[sel]
            m = (
                torch.zeros(len(chunk) * cap, dtype=torch.bool, device=win.device)
                .index_fill(0, dest, True)
                .view(len(chunk), cap)
            )
            out.append((tok[sel], dest, uq[chunk] // n_win, m))
    return out


class QuadPre(nn.Module):
    """The tokens' queries, keys and values, rotated by their positions."""

    def __init__(self, d, heads):
        super().__init__()
        self.H, self.dh = heads, d // heads
        self.nc, self.ne = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)

    def forward(self, xe, xm, ang):
        """xe [B, n0, d] (globals, entities), xm [B, M, d] (map tokens), ang [B, T, dh / 2] -> q, k, v [B, T, H, dh]."""
        h = torch.cat([self.ne(xe), self.nc(xm)], 1)
        B, T, _ = h.shape
        q, k, v = self.qkv(h).view(B, T, 3, self.H, self.dh).unbind(2)
        return _rot(q, ang), _rot(k, ang), v


class WinAttend(nn.Module):
    """One size group of windows: each row's tokens and the globals, one joint softmax per token."""

    def forward(self, qf, kf, vf, gk, gv, gm, tok, dest, frame, m):
        """qf, kf, vf [B T, H, dh]; gk, gv [B, g0, H, dh], gm [B, g0]; tok, dest (see window_sets); frame [rows];
        m [rows, cap] -> the windows' tokens' results, in tok's order."""
        rows, cap = m.shape
        H, dh = qf.shape[1:]
        pad = (
            lambda t_: qf.new_zeros(rows * cap, H, dh)
            .index_copy(0, dest, t_.index_select(0, tok))
            .view(rows, cap, H, dh)
        )
        r = attend(
            pad(qf).transpose(1, 2),
            torch.cat([pad(kf), gk[frame]], 1).transpose(1, 2),
            torch.cat([pad(vf), gv[frame]], 1).transpose(1, 2),
            torch.cat([m, gm[frame]], 1),
        )
        return r.transpose(1, 2).reshape(rows * cap, H, dh).index_select(0, dest)


class GlobalAttend(nn.Module):
    """Each global over all of the level's tokens."""

    def forward(self, q, k, v, valid):
        """q [B, H, g0, dh], k, v [B, H, T, dh], valid [B, T] -> [B, H, g0, dh]."""
        return attend(q, k, v, valid)


class QuadPost(nn.Module):
    """The attention's result into the tokens, then separate MLPs for map tokens and for entities / globals."""

    def __init__(self, d, mlp_ratio):
        super().__init__()
        self.o = nn.Linear(d, d)
        self.mc, self.mcn = mlp(d, d, int(d * mlp_ratio)), nn.LayerNorm(d)
        self.me, self.men = mlp(d, d, int(d * mlp_ratio)), nn.LayerNorm(d)

    def forward(self, xe, xm, a):
        o = self.o(a)
        n0 = xe.shape[1]
        xe, xm = xe + o[:, :n0], xm + o[:, n0:]
        return torch.cat([xe + self.me(self.men(xe)), xm + self.mc(self.mcn(xm))], 1)


class QuadBlock(nn.Module):
    """One attention block of a windowed level: per window, a joint softmax over its tokens and the
    globals; per global, one over everything. Its parts compile; this loop over the size groups stays
    Python, since the number of groups changes with every batch."""

    def __init__(self, d, heads, mlp_ratio=4.0):
        super().__init__()
        self.pre, self.win, self.glob = QuadPre(d, heads), WinAttend(), GlobalAttend()
        self.post = QuadPost(d, mlp_ratio)

    def forward(self, x, g0, n0, ang, valid, sets):
        """x [B, T, d]: [globals (g0), entities, map tokens (from n0)]; ang [B, T, dh / 2]; valid [B, T]."""
        B, T, d = x.shape
        xe, xm = x[:, :n0], x[:, n0:]
        q, k, v = self.pre(xe, xm, ang)
        H, dh = q.shape[2:]
        qf, kf, vf = (
            q.reshape(B * T, H, dh),
            k.reshape(B * T, H, dh),
            v.reshape(B * T, H, dh),
        )
        gk, gv, gm = k[:, :g0], v[:, :g0], valid[:, :g0]
        res = [
            self.win(qf, kf, vf, gk, gv, gm, tok, dest, frame, m)
            for tok, dest, frame, m in sets
        ]
        a = qf.new_zeros(B * T, H, dh).index_copy(
            0, torch.cat([s_[0] for s_ in sets]), torch.cat(res)
        )
        ga = self.glob(
            q[:, :g0].transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), valid
        ).transpose(1, 2)
        a = torch.cat([ga, a.view(B, T, H, dh)[:, g0:]], 1).reshape(B, T, d)
        return self.post(xe, xm, a)


class QuadEncoder(GridEncoder):
    """See the module docstring. Shares the token parts, the player index, the viewer code and the map
    view embedding with GridEncoder. Reads the square lattice from grid_latent.map_views.
    """

    def __init__(
        self,
        vocab,
        n_families,
        d=128,
        S=16,
        conv_blocks=3,
        attn_blocks=3,
        full_blocks=3,
        heads=4,
        mlp_ratio=4.0,
    ):
        super().__init__(
            vocab,
            n_families,
            d,
            S=S,
            pool_level=0,
            mix_depth=0,
            heads=heads,
            mlp_ratio=mlp_ratio,
            levels=1,
        )
        for name in (
            "off_mu",
            "pool",
            "mix",
            "convs",
            "attn",
            "down",
            "_masks",
            "team",
            "norm",
            "vis_in",
        ):
            delattr(self, name)
        self.grids = [TOP >> i for i in range(4)]
        assert self.grids[-1] == S
        self.seed = nn.Linear(d, d)
        self.convs = nn.ModuleList(
            [
                nn.ModuleList([ResBlock(d) for _ in range(conv_blocks)])
                for _ in self.grids
            ]
        )
        self.attn = nn.ModuleList(
            [
                nn.ModuleList(
                    [QuadBlock(d, heads, mlp_ratio) for _ in range(attn_blocks)]
                )
                for _ in self.grids[:-1]
            ]
        )
        self.down = nn.ModuleList([Down(d) for _ in self.grids[:-1]])
        self.lat_down = nn.ModuleList([Down(d) for _ in self.grids[:-2]])
        self.full = nn.ModuleList(
            [Block(d, heads, self.rope, mlp_ratio) for _ in range(full_blocks)]
        )

    def compile_parts(self):
        """Everything that compiles. Not the window grouping (shapes that depend on the data) nor the loops over levels."""
        for m in self.modules():
            if type(m).__name__ in (
                "ResBlock",
                "Down",
                "Block",
                "QuadPre",
                "QuadPost",
                "WinAttend",
                "GlobalAttend",
            ):
                m.compile(dynamic=True)
        self.lattice = torch.compile(self.lattice, dynamic=True)
        self.seed_norm = torch.compile(self.seed_norm, dynamic=True)

    def seed_norm(self, L, lp, lv, half):
        """The norm tokens at the top level: their place and the mean of the lattice cells inside them."""
        B, D = L.shape[0], L.shape[1]
        size, centers = geometry(half, TOP)
        seg = (
            cell_of(lp, size, TOP)
            + torch.arange(B, device=L.device)[:, None] * TOP * TOP
        ).reshape(-1)
        w = lv.reshape(-1).float()
        tot = torch.zeros(B * TOP * TOP, D, device=L.device).index_add_(
            0, seg, L.permute(0, 2, 3, 1).reshape(-1, D).float() * w[:, None]
        )
        den = (
            torch.zeros(B * TOP * TOP, device=L.device)
            .index_add_(0, seg, w)
            .clamp(min=1)
        )
        N0 = (
            self.pos_mu(centers)
            + self.cell_q
            + self.seed((tot / den[:, None]).view(B, TOP * TOP, D).to(centers.dtype))
        )
        return N0.view(B, TOP, TOP, D).permute(0, 3, 1, 2)

    def lattice(self, b):
        """The real map cells over the square: [B, d, Q, Q], after the minimap and the stem."""
        x = self.grid_static(b["lat_grid"])
        h = self.grid_mix(
            F.gelu(
                self.grid_view(b["lat_view"])
                + self.grid_code(b["view_player"])[:, :, None, None]
            )
        )
        return F.gelu(x.index_add(0, b["view_frame"], h.to(x.dtype)))

    def forward(self, b):
        cls, ply, ent, eff, effmask, effpos = self.embed_parts(b)
        B, P, D = ply.shape
        dev = ply.device
        G = torch.cat([cls.expand(B, -1, -1).to(ply.dtype), ply], 1)
        gmask = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=dev), b["ply_mask"]], 1
        )
        E = torch.cat([ent.to(ply.dtype), eff.to(ply.dtype)], 1)
        emask = torch.cat([b["ent_mask"], effmask], 1)
        epos = torch.cat([b["ent_pos"], effpos.to(b["ent_pos"].dtype)], 1).float()
        g0, ne = G.shape[1], E.shape[1]
        n0 = g0 + ne
        half = b["half"].float()
        side = 2 * half.max(-1).values  # the square's side, map units
        ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled()
        from torch.utils.checkpoint import checkpoint

        run = (
            (lambda f, *a: checkpoint(f, *a, use_reentrant=False))
            if ckpt
            else (lambda f, *a: f(*a))
        )

        def lat_geom(s, Q):
            """A lattice level of s map units per cell: cell centers (square-centered) and which are in the square."""
            c = (torch.arange(Q, device=dev, dtype=torch.float32) + 0.5) * s
            xy = torch.stack(torch.broadcast_tensors(c[None, :], c[:, None]), -1).view(
                1, Q * Q, 2
            )
            pos = xy - side[:, None, None] / 2
            return pos, (pos.abs() - s / 2 < side[:, None, None] / 2).all(-1)

        L = self.lattice(
            {k: b[k] for k in ("lat_grid", "lat_view", "view_player", "view_frame")}
        )
        lp, lv = lat_geom(1, L.shape[-1])
        lm = lv.view(B, 1, L.shape[-2], L.shape[-1]).to(L.dtype)
        L = (L * lm).contiguous(memory_format=torch.channels_last)
        for blk in self.stem:
            L = run(blk, L, lm)
        N = (
            self.seed_norm(L, lp, lv, half)
            .to(G.dtype)
            .contiguous(memory_format=torch.channels_last)
        )
        one = N.new_ones(B, 1, 1, 1)
        s = 1
        gpos = torch.zeros(B, g0, 2, device=dev)
        for li, S in enumerate(self.grids[:-1]):
            for blk in self.convs[li]:
                N = run(blk, N, one)
            Q = L.shape[-1]
            lp, lv = lat_geom(s, Q)
            size, centers = geometry(half, S)
            c = torch.arange(S, device=dev)
            nij = (
                torch.stack(torch.broadcast_tensors(c[None, :], c[:, None]), -1)
                .view(1, S * S, 2)
                .expand(B, -1, -1)
            )
            ij = lambda pos: torch.stack(
                [
                    (pos[..., k] / size[:, None] + S / 2).floor().long().clamp(0, S - 1)
                    for k in (0, 1)
                ],
                -1,
            )
            tij = torch.cat([ij(epos), nij, ij(lp)], 1)
            valid = torch.cat(
                [gmask, emask, torch.ones(B, S * S, dtype=torch.bool, device=dev), lv],
                1,
            )
            member = valid.clone()
            member[:, :g0] = False
            tij = torch.cat(
                [torch.zeros(B, g0, 2, dtype=torch.long, device=dev), tij], 1
            )
            sets = []
            for slide in (0, SLIDE):
                w, nw = window_of(tij, S, slide)
                sets.append(window_sets(w, member, nw))
            pos = torch.cat([gpos, epos, centers.expand(B, -1, -1), lp], 1)
            ang = self.attn_rope.angles(pos)
            x = torch.cat(
                [
                    G,
                    E,
                    N.permute(0, 2, 3, 1).reshape(B, S * S, D).to(G.dtype),
                    L.permute(0, 2, 3, 1).reshape(B, Q * Q, D).to(G.dtype),
                ],
                1,
            )
            for j, blk in enumerate(self.attn[li]):
                x = run(blk, x, g0, n0, ang, valid, sets[j % 2])
            G, E = x[:, :g0], x[:, g0:n0]
            N = (
                x[:, n0 : n0 + S * S]
                .view(B, S, S, D)
                .permute(0, 3, 1, 2)
                .contiguous(memory_format=torch.channels_last)
            )
            L = x[:, n0 + S * S :].view(B, Q, Q, D).permute(0, 3, 1, 2)
            L = (L * lv.view(B, 1, Q, Q).to(L.dtype)).contiguous(
                memory_format=torch.channels_last
            )
            N = self.down[li](N)
            if li < len(self.lat_down):
                L = self.lat_down[li](L)
                s *= 2
        # the latent's level: full attention, no lattice
        S = self.grids[-1]
        for blk in self.convs[-1]:
            N = run(blk, N, one)
        size, centers = geometry(half, S)
        y = torch.cat([G, E, N.permute(0, 2, 3, 1).reshape(B, S * S, D).to(G.dtype)], 1)
        mask = torch.cat(
            [gmask, emask, torch.ones(B, S * S, dtype=torch.bool, device=dev)], 1
        )
        ang = self.rope.angles(torch.cat([gpos, epos, centers.expand(B, -1, -1)], 1))
        for blk in self.full:
            y = run(blk, y, mask, ang)
        keep = torch.cat([y[:, :g0], y[:, n0:]], 1)
        lat = self.lat_norm(keep)
        lmask = torch.cat(
            [gmask, torch.ones(B, S * S, dtype=torch.bool, device=dev)], 1
        )
        lang = self.rope.angles(torch.cat([gpos, centers.expand(B, -1, -1)], 1))
        return dict(
            tokens=lat,
            mask=lmask,
            ang=lang,
            cls=lat[:, 0],
            players=lat[:, 1:g0],
            P=P,
            cell_slice=(g0, g0 + S * S),
            cell_hw=(S, S),
            cell_size=size,
            centers=centers,
        )
