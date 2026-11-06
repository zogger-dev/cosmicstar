"""Dense windowed map encoder (2026-10-31): the pyramid rebuilt from dense,
tensor-core-friendly parts. The sparse local attention of pyramid.py (neighbor shifts,
pair searches, segment softmax) kept the GPU memory-bound (~300 W of 600, 5.3 s/step;
torch.compile only reached 4.1 s).

Map: the grid is cropped to the playable area (tokenize.GameFrames.grid_origin) and
embedded per cell into one minimap (PyramidEncoder.minimap: static layers + each
player's visible / seen / creep with their player code). Then

    stem: GELU, 2 ResNet blocks                                         (1 map unit per cell)
    for s in 1, 2, 4, 8, 16 map units per cell:
        3 ResNet blocks (dense 3 x 3 convs, per-cell LayerNorm, identity skips; cells off the
        frame's own map zeroed after every map step, so batch-mates' padding never leaks in)
        3 attention blocks, one FlexAttention softmax per query over [globals, entities, cells]:
            a cell sees the 16 cells of its fixed 4 x 4 window, every entity and the globals;
            entities and globals see everything
        2 x 2 downsample (a learned 2 x 2 stride-2 conv plus a projection shortcut of the
        2 x 2 average), except after the last level
    output: [CLS, players, units, effects, the 16-unit cells]; every level's map is kept in
    "maps" (for a later full-resolution decoder)

No pair biases (2026-10-31): a learned geometric bias inside the kernel (from the gap
between bodies, and between a body and a cell's square) cost ~40% of the full-resolution
block on CUDA cores, even in its cheapest form (fixed per-head scales, learned per-query
strengths). Rotary angles down to 1/8 map unit with learned q / k already give a learned
function of the offset, and the radius is in each entity's features; the tight-passage
evaluation checks what's lost.

Positions: rotary angles at wavelengths 0.125-512 map units, entity position embeddings
from map units at octaves 0.125-512 (encoder.MapFourier), so the scale is the same on
every map and resolves 1/8 of a cell. Widths stay d everywhere (cells, entities,
globals)."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from .direct import RoPE2D
from .encoder import MapFourier, mlp
from .pyramid import ON_MAP, PyramidEncoder, _rot

WIN = 4  # attention windows: WIN x WIN cells
BLOCK = 128  # FlexAttention block: the dense part and the cells are padded to it
ROPE_LO, ROPE_HI = 0.125, 512.0  # wavelengths, map units
_flex = torch.compile(flex_attention, dynamic=True)


class CellNorm(nn.LayerNorm):
    """LayerNorm over each cell's channels on [B, C, H, W] (a view for channels-last
    maps). Each cell is normalized by itself: GroupNorm's statistics spanned the whole
    canvas, padding included, so a frame's output depended on the largest map in its
    batch."""

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ResBlock(nn.Module):
    """ResNet basic block on [B, C, H, W]: conv 3 x 3, norm, GELU, conv 3 x 3, norm, +
    x, GELU. m [B, 1, H, W]: the frame's own cells; padding is zeroed between the convs
    too (a norm turns a zero cell into its bias, which the second conv would read at the
    map's edge)."""

    def __init__(self, c):
        super().__init__()
        self.c1, self.n1 = nn.Conv2d(c, c, 3, padding=1, bias=False), CellNorm(c)
        self.c2, self.n2 = nn.Conv2d(c, c, 3, padding=1, bias=False), CellNorm(c)

    def forward(self, x, m):
        h = F.gelu(self.n1(self.c1(x))) * m
        return F.gelu(x + self.n2(self.c2(h))) * m


class Down(nn.Module):
    """2 x 2 downsample with a projection shortcut: conv 2 x 2 stride 2 (a learned merge
    of each 2 x 2 block) + linear(average of the block), as ResNet bridges a resolution
    change.
    """

    def __init__(self, c):
        super().__init__()
        self.merge, self.norm = nn.Conv2d(c, c, 2, stride=2, bias=False), CellNorm(c)
        self.short = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return self.norm(self.merge(x)) + self.short(F.avg_pool2d(x, 2))


class WindowBlock(nn.Module):
    """One attention block over [globals, entities, cells (window-major)]: shared q / k
    / v, a joint softmax per query (FlexAttention: block mask for the windows, score_mod
    for validity), then separate MLPs for cells and for entities / globals."""

    def __init__(self, d, heads, mlp_ratio=4.0):
        super().__init__()
        self.H, self.dh = heads, d // heads
        self.nc, self.ne = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mc, self.mcn = mlp(d, d, int(d * mlp_ratio)), nn.LayerNorm(d)
        self.me, self.men = mlp(d, d, int(d * mlp_ratio)), nn.LayerNorm(d)

    def forward(self, x, ctx):
        """x [B, L, d]; ctx: n0 (first cell index), ang [B, L, dh / 2], valid [B, L]
        (keys), block_mask."""
        B, L, d = x.shape
        n0 = ctx["n0"]
        h = torch.cat([self.ne(x[:, :n0]), self.nc(x[:, n0:])], 1)
        q, k, v = self.qkv(h).view(B, L, 3, self.H, self.dh).unbind(2)
        q, k = _rot(q, ctx["ang"]), _rot(k, ctx["ang"])
        valid = ctx["valid"]

        def score_mod(score, b, hh, qi, ki):
            return torch.where(valid[b, ki], score, -float("inf"))

        a = (
            _flex(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                score_mod=score_mod,
                block_mask=ctx["block_mask"],
            )
            .transpose(1, 2)
            .reshape(B, L, d)
        )
        x = x + self.o(a)
        return torch.cat(
            [
                x[:, :n0] + self.me(self.men(x[:, :n0])),
                x[:, n0:] + self.mc(self.mcn(x[:, n0:])),
            ],
            1,
        )


class DenseEncoder(PyramidEncoder):
    """See the module docstring. Shares the token parts (entities with their lists,
    players, effects, the minimap embedding) with PyramidEncoder; replaces the map
    pyramid."""

    def __init__(
        self,
        vocab,
        n_families,
        d=128,
        levels=5,
        stem_blocks=2,
        conv_blocks=3,
        attn_blocks=3,
        heads=4,
        mlp_ratio=4.0,
    ):
        super().__init__(
            vocab,
            n_families,
            d,
            widths=(d,),
            conv_blocks=1,
            attn_blocks=1,
            full_levels=1,
            heads=heads,
            mlp_ratio=mlp_ratio,
        )
        for name in (
            "stem",
            "local_conv",
            "local",
            "merge",
            "full_rope",
            "full_conv",
            "full",
            "null_tok",
        ):
            delattr(self, name)
        self.pos_mu = nn.Sequential(
            MapFourier(ROPE_LO, ROPE_HI),
            nn.Linear(MapFourier(ROPE_LO, ROPE_HI).out_dim(2), d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.stem = nn.ModuleList([ResBlock(d) for _ in range(stem_blocks)])
        self.convs = nn.ModuleList(
            [
                nn.ModuleList([ResBlock(d) for _ in range(conv_blocks)])
                for _ in range(levels)
            ]
        )
        self.attn = nn.ModuleList(
            [
                nn.ModuleList(
                    [WindowBlock(d, heads, mlp_ratio) for _ in range(attn_blocks)]
                )
                for _ in range(levels)
            ]
        )
        self.down = nn.ModuleList([Down(d) for _ in range(levels - 1)])
        self.attn_rope = RoPE2D(d // heads, lo=ROPE_LO, hi=ROPE_HI)
        self.rope = RoPE2D(d // heads, lo=ROPE_LO, hi=ROPE_HI)  # the predictor's angles
        self._masks = {}

    def place(self, b, g):
        """Entity / effect / pointer positions from map units (the same scale on every
        map)."""
        return self.pos_mu(b[f"{g}_pos"]) if f"{g}_pos" in b else self.pos(b[f"{g}_xy"])

    def _block_mask(self, n0, L, dev):
        """[globals + entities (n0, a multiple of BLOCK), cells (window-major, L - n0, a
        multiple of BLOCK)]: the dense rows and columns are full blocks; a cell block
        sees the dense blocks fully and itself partly (each cell its WIN x WIN window).
        Built from the structure (the generic builder evaluates every pair: 18 s at
        level 1). Cached per shape."""
        key = (n0, L, str(dev))
        if key not in self._masks:
            if len(self._masks) > 64:
                self._masks.clear()
            nd, nb = n0 // BLOCK, L // BLOCK
            i32 = dict(dtype=torch.int32, device=dev)
            full_num = torch.full((nb,), nd, **i32)
            full_num[:nd] = nb
            full_idx = torch.arange(nb, **i32)[None].expand(nb, nb).contiguous()
            part_num = torch.zeros(nb, **i32)
            part_num[nd:] = 1
            part_idx = torch.zeros(nb, nb, **i32)
            part_idx[nd:, 0] = torch.arange(nd, nb, **i32)

            def mask_mod(
                b, h, qi, ki
            ):  # inside a cell block (n0 is a multiple of the window size)
                return qi // (WIN * WIN) == ki // (WIN * WIN)

            self._masks[key] = BlockMask.from_kv_blocks(
                part_num[None, None],
                part_idx[None, None],
                full_num[None, None],
                full_idx[None, None],
                BLOCK_SIZE=BLOCK,
                mask_mod=mask_mod,
                seq_lengths=(L, L),
            )
        return self._masks[key]

    @staticmethod
    def _to_windows(C, valid):
        """[B, d, H, W] -> window-major tokens [B, n, d] (H, W padded to multiples of
        WIN) and their validity; returns the padded size for the way back."""
        B, d, H, W = C.shape
        Hp, Wp = -(-H // WIN) * WIN, -(-W // WIN) * WIN
        C = F.pad(C, (0, Wp - W, 0, Hp - H))
        valid = F.pad(valid, (0, Wp - W, 0, Hp - H))
        t = (
            C.view(B, d, Hp // WIN, WIN, Wp // WIN, WIN)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(B, Hp * Wp, d)
        )
        vm = (
            valid.view(B, Hp // WIN, WIN, Wp // WIN, WIN)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, Hp * Wp)
        )
        return t, vm, (Hp, Wp)

    @staticmethod
    def _from_windows(t, H, W, Hp, Wp):
        B, n, d = t.shape
        C = (
            t.view(B, Hp // WIN, Wp // WIN, WIN, WIN, d)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(B, d, Hp, Wp)
        )
        return C[:, :, :H, :W]

    def forward(self, b):
        cls, ply, ent, eff, effmask, effpos = self.embed_parts(b)
        B, P, D = ply.shape
        Nu = ent.shape[1]
        G = torch.cat([cls.expand(B, -1, -1).to(ply.dtype), ply], 1)
        gmask = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=ply.device), b["ply_mask"]], 1
        )
        E = torch.cat([ent.to(ply.dtype), eff.to(ply.dtype)], 1)
        emask = torch.cat([b["ent_mask"], effmask], 1)
        epos = torch.cat([b["ent_pos"], effpos.to(b["ent_pos"].dtype)], 1).float()
        g0, ne = G.shape[1], E.shape[1]
        n0 = (
            -(-(g0 + ne) // BLOCK) * BLOCK
        )  # the dense part padded to a block (invalid slots)
        pad_e = n0 - g0 - ne
        origin = (
            b["grid_origin"] - b["center"]
        )  # world (0, 0) of the canvas, in playable-centered units
        grid = b["grid"]
        on_map = (b["on_map"] if "on_map" in b else grid[:, ON_MAP : ON_MAP + 1]) > 0
        # cells outside the frame's own (cropped) map are zeroed after every map step,
        # so what a frame computes never depends on how much padding its batch-mates
        # added
        own = lambda s_: (
            (F.max_pool2d(on_map.float(), s_, s_) if s_ > 1 else on_map.float()) > 0
        ).to(grid.dtype)
        keep = lambda C_, s_: C_ * own(s_).to(C_.dtype)
        C = keep(F.gelu(self.minimap(grid, b)), 1).contiguous(
            memory_format=torch.channels_last
        )
        ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled()
        from torch.utils.checkpoint import checkpoint

        run = (
            (lambda f, *a: checkpoint(f, *a, use_reentrant=False))
            if ckpt
            else (lambda f, *a: f(*a))
        )
        m = own(1).to(C.dtype)
        for blk in self.stem:
            C = run(blk, C, m)
        gpos = G.new_zeros(B, g0, 2).float()
        maps = []
        s = 1
        for li in range(len(self.convs)):
            m = own(s).to(C.dtype)
            for blk in self.convs[li]:
                C = run(blk, C, m)
            H, W = C.shape[2:]
            cvalid = (
                (F.max_pool2d(on_map.float(), s, s)[:, 0] > 0)
                if s > 1
                else on_map[:, 0]
            )
            ct, cv, (Hp, Wp) = self._to_windows(C, cvalid)
            nc = ct.shape[1]
            ncp = -(-nc // BLOCK) * BLOCK  # the cells padded to a block
            # cell centers, playable-centered map units, in the same window-major order
            ys = (torch.arange(Hp, device=C.device, dtype=torch.float32) + 0.5) * s
            xs = (torch.arange(Wp, device=C.device, dtype=torch.float32) + 0.5) * s
            cxy = torch.stack(
                torch.broadcast_tensors(xs[None, :], ys[:, None]), -1
            )  # [Hp, Wp, 2]
            cxy = (
                cxy.view(Hp // WIN, WIN, Wp // WIN, WIN, 2)
                .permute(0, 2, 1, 3, 4)
                .reshape(1, Hp * Wp, 2)
            )
            cxy = cxy + origin[:, None, :]
            pos = torch.cat(
                [
                    gpos,
                    epos,
                    epos.new_zeros(B, pad_e, 2),
                    cxy.expand(B, -1, -1),
                    cxy.new_zeros(B, ncp - nc, 2),
                ],
                1,
            )
            fz = lambda n: torch.zeros(B, n, dtype=torch.bool, device=C.device)
            ctx = dict(
                n0=n0,
                ang=self.attn_rope.angles(pos),
                valid=torch.cat([gmask, emask, fz(pad_e), cv, fz(ncp - nc)], 1),
                block_mask=self._block_mask(n0, n0 + ncp, C.device),
            )
            x = torch.cat(
                [
                    G,
                    E,
                    G.new_zeros(B, pad_e, D),
                    ct.to(G.dtype),
                    G.new_zeros(B, ncp - nc, D),
                ],
                1,
            )
            for blk in self.attn[li]:
                x = run(blk, x, ctx)
            G, E = x[:, :g0], x[:, g0 : g0 + ne]
            C = keep(
                self._from_windows(x[:, n0 : n0 + nc], H, W, Hp, Wp), s
            ).contiguous(memory_format=torch.channels_last)
            maps.append(C)
            if li < len(self.down):
                s *= 2
                C = keep(self.down[li](C), s)
        # output: [CLS, players, units, effects, the last level's cells]
        H, W = C.shape[2:]
        cvalid = F.max_pool2d(on_map.float(), s, s)[:, 0] > 0
        ys = (torch.arange(H, device=C.device, dtype=torch.float32) + 0.5) * s
        xs = (torch.arange(W, device=C.device, dtype=torch.float32) + 0.5) * s
        cxy = (
            torch.stack(torch.broadcast_tensors(xs[None, :], ys[:, None]), -1).view(
                1, H * W, 2
            )
            + origin[:, None, :]
        )
        cells = C.permute(0, 2, 3, 1).reshape(B, H * W, D)
        x = self.norm(torch.cat([G, E, cells.to(G.dtype)], 1))
        mask = torch.cat([gmask, emask, cvalid.reshape(B, H * W)], 1)
        pos = torch.cat([gpos, epos, cxy.expand(B, -1, -1)], 1)
        return dict(
            tokens=x,
            mask=mask,
            ang=self.rope.angles(pos),
            cls=x[:, 0],
            players=x[:, 1 : 1 + P],
            units=x[:, 1 + P : 1 + P + Nu],
            P=P,
            unit_slice=(1 + P, 1 + P + Nu),
            cell_slice=(g0 + ne, x.shape[1]),
            cell_hw=(H, W),
            maps=maps,
        )
