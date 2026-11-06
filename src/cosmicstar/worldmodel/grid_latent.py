"""Grid latent (DESIGN_grid_latent.md, 2026-11-01): a fixed-size latent for every map,
and an entity-list decoder from it.

Encoder: the dense encoder (dense.py) runs as before; then its entity / effect tokens
and the cells of one map level (real cells, each at its true center in map units) are
scattered onto a fixed S x S grid over the playable area (centered, scaled by the
longest side, so cells are square in map units and the shorter side is padded evenly).
Each latent cell pools what fell into it with one attention softmax (query: the cell's
center; keys / values: the token plus its offset from the cell center, so the real
lattice and its scale stay visible), then a few transformer blocks mix [globals, S x S
cells]. The latent is those 1 + P + S^2 tokens.

Decoder (a probe: low gradient into the latent): per latent cell, a Poisson count for
every (owner, type), owner in {neutral, player 0, player 1} (the frame's player order);
one query per entity, built from its cell's vector, its (owner, type) and its slot
within that (cell, owner, type) group; the queries attend to each other and to the
latent and read out the entity's offset within its cell, its numbers
(tokenize.entity_features) and its order. Training places the queries from the true
counts. The loss ignores the order of entities within a group: predictions and targets
are each sorted by y offset within the group and paired in that order (exact for 1-D, on
the GPU, no assignment solver; evaluation compares it with a Hungarian assignment).
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from .dense import ROPE_HI, DenseEncoder
from .direct import Block, RoPE2D
from .encoder import MapFourier, mlp
from .pyramid import ON_MAP
from .tokenize import CODE_NEUTRAL, ENTITY_NUMS, GRID_CHANNELS, N_GRID_PLAYERS

N_OWNERS = 3  # neutral, the frame's player 0, player 1
MAX_SLOT = 128  # slot embeddings (a group larger than this shares the last)
VIEW_LAYERS = ("visible", "seen", "creep")  # one player's map view
STATIC_LAYERS = GRID_CHANNELS[
    len(VIEW_LAYERS) * N_GRID_PLAYERS :
]  # the map's own layers


def grad_scale(x, w):
    """x's value, w times its gradient."""
    return x if w == 1 else x.detach() + w * (x - x.detach())


def map_views(b):
    """A collated batch -> the same batch with its map split into what it is: b["grid"]
    the frame's static layers (STATIC_LAYERS); b["view_grid"] [M, 3, H, W] one row per
    player view the frames hold (a player's frame: the viewer's; a combined frame:
    both), b["view_frame"] and b["view_player"] [M] saying whose. A view has no slot for
    anyone else."""
    g = b["grid"]
    B, _, H, W = g.shape
    n = len(VIEW_LAYERS) * N_GRID_PLAYERS
    b["view_frame"], b["view_player"] = torch.nonzero(
        b["ply_gridok"][:, :N_GRID_PLAYERS], as_tuple=True
    )
    b["view_grid"] = g[:, :n].view(B, N_GRID_PLAYERS, len(VIEW_LAYERS), H, W)[
        b["view_frame"], b["view_player"]
    ]
    b["grid"] = g[:, n:]
    b["on_map"] = b["grid"][
        :, STATIC_LAYERS.index("on_map") : STATIC_LAYERS.index("on_map") + 1
    ]
    # the lattice for QuadEncoder: the same layers on a square of each frame's longest
    # playable side, centered on the playable center (padded where the cache has no
    # terrain; real cells past the square are dropped)
    side = (2 * b["half"].max(-1).values).long()
    Q = -(-int(side.max()) // 4) * 4
    origin = (b["grid_origin"] - b["center"]).round().long()
    lat, view = g.new_zeros(B, len(STATIC_LAYERS), Q, Q), g.new_zeros(
        len(b["view_frame"]), len(VIEW_LAYERS), Q, Q
    )
    for i in range(B):
        ox, oy, L = (
            int(origin[i, 0]) + int(side[i]) // 2,
            int(origin[i, 1]) + int(side[i]) // 2,
            int(side[i]),
        )
        sy, sx = max(-oy, 0), max(-ox, 0)
        h, w = min(H - sy, L - max(oy, 0)), min(W - sx, L - max(ox, 0))
        if h <= 0 or w <= 0:
            continue
        dy, dx = max(oy, 0), max(ox, 0)
        lat[i, :, dy : dy + h, dx : dx + w] = b["grid"][i, :, sy : sy + h, sx : sx + w]
        for r in torch.nonzero(b["view_frame"] == i, as_tuple=True)[0]:
            view[r, :, dy : dy + h, dx : dx + w] = b["view_grid"][
                r, :, sy : sy + h, sx : sx + w
            ]
    b["lat_grid"], b["lat_view"] = lat, view
    return b


def geometry(half, S):
    """Latent cells over the playable area: half [B, 2] (playable half-extents, map
    units) -> cell size [B] and cell centers [B, S * S, 2] (playable-centered map units,
    row-major in y).
    """
    size = 2 * half.max(-1).values / S
    c = torch.arange(S, device=half.device, dtype=torch.float32) + 0.5 - S / 2
    cy, cx = torch.meshgrid(c, c, indexing="ij")
    return size, torch.stack([cx, cy], -1).view(1, S * S, 2) * size[:, None, None]


def cell_of(pos, size, S):
    """pos [B, N, 2] (playable-centered map units) -> latent cell index [B, N]."""
    ij = torch.floor(pos / size[:, None, None] + S / 2).long().clamp(0, S - 1)
    return ij[..., 1] * S + ij[..., 0]


class ScatterPool(nn.Module):
    """Each latent cell attends over the tokens that fell into it (one softmax per cell
    and head, by scatter: no padding to the fullest cell). Empty cells get nothing
    (their query alone).
    """

    def __init__(self, d, heads, mlp_ratio=4.0):
        super().__init__()
        self.H, self.dh = heads, d // heads
        self.nx, self.nq = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.n2, self.ff = nn.LayerNorm(d), mlp(d, d, int(d * mlp_ratio))

    def forward(self, x, off, seg, q0):
        """x, off [B, N, d] (tokens, their offset embeddings); seg [B, N] in 0..C (C =
        dump: invalid tokens); q0 [B, C, d] cell queries -> [B, C, d]."""
        B, N, d = x.shape
        C = q0.shape[1]
        H, dh = self.H, self.dh
        segf = (seg + torch.arange(B, device=x.device)[:, None] * (C + 1)).reshape(-1)
        q = torch.cat([self.q(self.nq(q0)), q0.new_zeros(B, 1, d)], 1).reshape(
            B * (C + 1), H, dh
        )
        k, v = self.kv(self.nx(x) + off).reshape(B * N, 2, H, dh).unbind(1)
        score = (q[segf].float() * k.float()).sum(-1) / dh**0.5  # [B N, H]
        mx = torch.full((B * (C + 1), H), -1e30, device=x.device).scatter_reduce(
            0, segf[:, None].expand(-1, H), score, "amax", include_self=True
        )
        w = torch.exp(score - mx[segf])
        den = torch.zeros(B * (C + 1), H, device=x.device).index_add_(0, segf, w)
        num = torch.zeros(B * (C + 1), H, dh, device=x.device).index_add_(
            0, segf, w[..., None] * v.float()
        )
        a = (num / den.clamp(min=1e-6)[..., None]).view(B, C + 1, d)[:, :C].to(q0.dtype)
        y = q0 + self.o(a)
        return y + self.ff(self.n2(y))


class GridEncoder(DenseEncoder):
    """The dense encoder, then the scatter onto the fixed grid and the latent mixer
    (module docstring). Returns the latent as tokens [CLS, players, S x S cells] (the
    same keys the dynamics read), the dense output under "dense" (the posterior and the
    decoder's targets read its entity tokens), the cell size and centers."""

    def __init__(
        self,
        vocab,
        n_families,
        d=128,
        S=16,
        pool_level=1,
        mix_depth=4,
        heads=4,
        mlp_ratio=4.0,
        **kw,
    ):
        super().__init__(vocab, n_families, d, heads=heads, mlp_ratio=mlp_ratio, **kw)
        self.S, self.pool_level = S, pool_level
        self.off_mu = nn.Sequential(
            MapFourier(0.125, 64.0),
            nn.Linear(MapFourier(0.125, 64.0).out_dim(2), d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.cell_q = nn.Parameter(torch.zeros(d))
        self.pool = ScatterPool(d, heads, mlp_ratio)
        self.mix = nn.ModuleList(
            [Block(d, heads, self.rope, mlp_ratio) for _ in range(mix_depth)]
        )
        self.lat_norm = nn.LayerNorm(d)
        self.viewer = nn.Linear(
            N_GRID_PLAYERS, d, bias=False
        )  # multi-hot: [1, 0] and [0, 1] the players, [1, 1] the observer

    def owner(self, b, g):
        """Whose (the player's index, neutral its own) and their relation to the
        viewer."""
        return self.pcode(b[f"{g}_pcode"]) + self.rel(b[f"{g}_rel"])

    def embed_parts(self, b):
        cls, *rest = super().embed_parts(b)
        return (
            cls
            + self.viewer(b["ply_gridok"][:, :N_GRID_PLAYERS].to(cls.dtype))[:, None],
            *rest,
        )

    def minimap(self, grid, b):
        """[B, C, H, W] static layers -> [B, w0, H, W]: static(the map's layers) plus,
        per player view the frame holds, mix(GELU(view(its layers) + code(its player))).
        Batches without map_views: as before.
        """
        if "view_grid" not in b:
            return super().minimap(grid, b)
        x = self.grid_static(grid)
        h = self.grid_mix(
            F.gelu(
                self.grid_view(b["view_grid"])
                + self.grid_code(b["view_player"])[:, :, None, None]
            )
        )
        return x.index_add(0, b["view_frame"], h.to(x.dtype))

    def forward(self, b):
        e = super().forward(b)
        x, m = e["tokens"], e["mask"]
        B, _, d = x.shape
        P = e["P"]
        g0, (c0, _) = 1 + P, e["cell_slice"]
        S = self.S
        # tokens to scatter: entities and effects (their dense outputs), and the real
        # cells of one map level
        epos = [b["ent_pos"].float()] + (
            [b["eff_pos"].float()] if "eff_type" in b else []
        )
        epos = torch.cat(epos, 1)
        C = e["maps"][self.pool_level]
        s = 2**self.pool_level
        H, W = C.shape[2:]
        origin = b["grid_origin"] - b["center"]
        ys = (torch.arange(H, device=x.device, dtype=torch.float32) + 0.5) * s
        xs = (torch.arange(W, device=x.device, dtype=torch.float32) + 0.5) * s
        cxy = (
            torch.stack(torch.broadcast_tensors(xs[None, :], ys[:, None]), -1).view(
                1, H * W, 2
            )
            + origin[:, None, :]
        )
        on_map = (
            b["on_map"] if "on_map" in b else b["grid"][:, ON_MAP : ON_MAP + 1]
        ).float()
        cvalid = (F.max_pool2d(on_map, s, s) if s > 1 else on_map)[:, 0].reshape(
            B, H * W
        ) > 0
        tok = torch.cat(
            [x[:, g0:c0], C.permute(0, 2, 3, 1).reshape(B, H * W, d).to(x.dtype)], 1
        )
        pos = torch.cat([epos, cxy.expand(B, -1, -1)], 1)
        valid = torch.cat([m[:, g0:c0], cvalid], 1)
        size, centers = geometry(b["half"].float(), S)
        cell = cell_of(pos, size, S)
        off = pos - torch.gather(centers, 1, cell[..., None].expand(-1, -1, 2))
        seg = torch.where(valid, cell, S * S)
        q0 = (self.pos_mu(centers) + self.cell_q).to(x.dtype)
        lat = self.pool(tok, self.off_mu(off).to(x.dtype), seg, q0)
        y = torch.cat([x[:, :g0], lat], 1)
        mask = torch.cat(
            [m[:, :g0], torch.ones(B, S * S, dtype=torch.bool, device=x.device)], 1
        )
        ang = self.rope.angles(torch.cat([centers.new_zeros(B, g0, 2), centers], 1))
        for blk in self.mix:
            y = blk(y, mask, ang)
        y = self.lat_norm(y)
        return dict(
            tokens=y,
            mask=mask,
            ang=ang,
            cls=y[:, 0],
            players=y[:, 1:g0],
            P=P,
            cell_slice=(g0, g0 + S * S),
            cell_hw=(S, S),
            dense=e,
            cell_size=size,
            centers=centers,
        )


# ------------------------------------------------------------------ the entity decoder


def targets(fb, S, n_units):
    """What the decoder must produce for a collated frame batch fb (any batch of
    frames): per entity (units, not effects) its latent cell, (owner, type) class, slot
    within its (cell, owner, type) group (by y offset), offset within the cell (cell
    units, -0.5..0.5); per cell, the count of each (owner, type). All on the GPU, fixed
    shapes."""
    pos, msk = fb["ent_pos"].float(), fb["ent_mask"]
    B, N = msk.shape
    size, centers = geometry(fb["half"].float(), S)
    cell = cell_of(pos, size, S)
    ctr = torch.gather(centers, 1, cell[..., None].expand(-1, -1, 2))
    off = (pos - ctr) / size[:, None, None]
    pc = fb["ent_pcode"]
    pi = (
        (pc[..., None] == fb["ply_pcode"][:, None, :])
        .float()
        .argmax(-1)
        .clamp(max=N_OWNERS - 2)
    )
    owner = torch.where(pc == CODE_NEUTRAL, 0, 1 + pi)
    n_oc = N_OWNERS * n_units
    oc = owner * n_units + fb["ent_type"].clamp(0, n_units - 1)
    gkey = cell * n_oc + oc  # per sample
    counts = (
        torch.zeros(B, S * S * n_oc, device=pos.device)
        .scatter_add_(1, gkey, msk.float())
        .view(B, S * S, n_oc)
    )
    # slot = rank within the group, by y offset (padding sorts last)
    gg = gkey + torch.arange(B, device=pos.device)[:, None] * (
        S * S * n_oc
    )  # global group id
    big = float(B * S * S * n_oc + 1)
    comp = torch.where(
        msk, gg.double() * 4 + (off[..., 1].double().clamp(-1.4, 1.4) + 1.5), big * 4
    ).reshape(-1)
    order = torch.argsort(comp)
    sg = torch.where(msk.reshape(-1), gg.reshape(-1), int(big))[order]
    first = torch.searchsorted(sg, sg)
    rank = torch.arange(B * N, device=pos.device) - first
    slot = torch.empty_like(rank).scatter_(0, order, rank).view(B, N)
    return dict(
        cell=cell,
        oc=oc,
        gg=gg,
        slot=slot,
        off=off,
        mask=msk,
        counts=counts,
        size=size,
        ctr=ctr,
        num=fb["ent_num"].float(),
        order=fb["ent_order"],
    )


class EntityDecoder(nn.Module):
    """Latent -> (log count rates per cell and (owner, type), per-entity readouts for
    the given queries). Queries come from targets() (training: the true counts)."""

    def __init__(self, d, n_units, n_abilities, n_num, depth=2, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.n_units = n_units
        n_oc = N_OWNERS * n_units
        self.count = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_oc))
        nn.init.zeros_(self.count[-1].weight)
        nn.init.constant_(
            self.count[-1].bias, -7.0
        )  # ~1 entity per 3 cells over ~1000 classes
        self.oc = nn.Embedding(n_oc, d)
        self.slot = nn.Embedding(MAX_SLOT, d)
        self.rope = RoPE2D(d // heads, lo=1.0, hi=ROPE_HI)
        self.layers = nn.ModuleList(
            [Block(d, heads, self.rope, mlp_ratio, cross=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d)
        self.off, self.num, self.order = (
            nn.Linear(d, 2),
            nn.Linear(d, n_num),
            nn.Linear(d, n_abilities),
        )

    def rates(self, lat, cell_slice):
        c0, c1 = cell_slice
        return self.count(lat[:, c0:c1]).float()  # [B, S^2, n_oc] log rates

    def forward(self, lat, lat_mask, cell_slice, tg):
        c0, c1 = cell_slice
        cells = lat[:, c0:c1]
        d = lat.shape[-1]
        q = (
            torch.gather(cells, 1, tg["cell"][..., None].expand(-1, -1, d))
            + self.oc(tg["oc"]).to(lat.dtype)
            + self.slot(tg["slot"].clamp(max=MAX_SLOT - 1)).to(lat.dtype)
        )
        qm = tg["mask"].clone()
        qm[:, 0] = (
            True  # a frame without entities: one dummy key, so no row is all-masked
        )
        ang = self.rope.angles(tg["ctr"])
        for blk in self.layers:
            q = blk(q, qm, ang, lat, lat_mask)
        h = self.norm(q)
        return dict(
            h=h,
            off=self.off(h).float(),
            num=self.num(h).float(),
            order=self.order(h).float(),
        )


def match(out, tg):
    """Pairs predictions with targets within each (cell, owner, type) group, both sorted
    by y offset: returns (prediction index, target index, valid), flat over the
    batch."""
    msk = tg["mask"].reshape(-1)
    big = 1e12  # above every group key x 4 (no sync to find the maximum)
    gg = tg["gg"].reshape(-1).double()
    yp = out["off"][..., 1].reshape(-1).double().clamp(-1.4, 1.4) + 1.5
    yt = tg["off"][..., 1].reshape(-1).double().clamp(-1.4, 1.4) + 1.5
    ip = torch.argsort(torch.where(msk, gg * 4 + yp, big))
    it = torch.argsort(torch.where(msk, gg * 4 + yt, big))
    return ip, it, msk[it]


def decode_losses(dec, lat, lat_mask, cell_slice, tg):
    """Count (Poisson, per cell, summed over classes) and matched entity losses; returns
    (losses, decoder output, the pairing)."""
    lr = dec.rates(lat, cell_slice)
    cnt = (
        F.poisson_nll_loss(
            lr, tg["counts"], log_input=True, full=False, reduction="none"
        )
        .sum(-1)
        .mean()
    )
    out = dec(lat, lat_mask, cell_slice, tg)
    ip, it, ok = match(out, tg)
    w = ok.float()
    n = w.sum().clamp(min=1)
    fl = lambda x: x.reshape(-1, *x.shape[2:])
    off = (
        F.smooth_l1_loss(
            fl(out["off"])[ip], fl(tg["off"])[it], beta=0.1, reduction="none"
        ).sum(-1)
        * w
    ).sum() / n
    num = (
        F.mse_loss(fl(out["num"])[ip], fl(tg["num"])[it], reduction="none").mean(-1) * w
    ).sum() / n
    ordl = (
        F.cross_entropy(
            fl(out["order"])[ip], fl(tg["order"])[it].clamp(min=0), reduction="none"
        )
        * w
    ).sum() / n
    return dict(cnt=cnt, off=off, num=num, ord=ordl), out, lr, (ip, it, ok)


def gram(x, w=None):
    """[B, N, d] -> [B, d, d], second moments over the (weighted) rows."""
    x = x.float()
    if w is None:
        return x.transpose(1, 2) @ x / x.shape[1]
    w = w.float()
    return (x * w[..., None]).transpose(1, 2) @ x / w.sum(1).clamp(min=1)[:, None, None]


def cross_gram(a, b, w):
    w = w.float()
    return (
        (a.float() * w[..., None]).transpose(1, 2)
        @ b.float()
        / w.sum(1).clamp(min=1)[:, None, None]
    )


# -------------------------------------------------------- the map decoder (pretraining)


def canvas_cells(fb, S):
    """Every canvas cell of a collated frame batch (1 map unit each, from grid_origin):
    its latent cell [B, H W] and its offset from that cell's center in map units [B, H
    W, 2]."""
    B, _, H, W = fb["grid"].shape
    origin = (fb["grid_origin"] - fb["center"]).float()
    ys = torch.arange(H, device=origin.device, dtype=torch.float32) + 0.5
    xs = torch.arange(W, device=origin.device, dtype=torch.float32) + 0.5
    xy = (
        torch.stack(torch.broadcast_tensors(xs[None, :], ys[:, None]), -1).view(
            1, H * W, 2
        )
        + origin[:, None, :]
    )
    size, centers = geometry(fb["half"].float(), S)
    cell = cell_of(xy, size, S)
    return cell, xy - torch.gather(centers, 1, cell[..., None].expand(-1, -1, 2))


class MapDecoder(nn.Module):
    """Latent cells -> the map layers at 1 map unit: a real cell reads its latent cell's
    vector and its offset from that cell's center (Fourier features), through an MLP (an
    implicit field per latent cell). With players: one player's view of those cells, the
    query saying whose."""

    def __init__(self, d, n_out, hidden=256, players=0):
        super().__init__()
        self.player = nn.Embedding(players, d) if players else None
        self.fourier = MapFourier(0.25, 32.0)
        self.mlp = nn.Sequential(
            nn.Linear(d + self.fourier.out_dim(2), hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, cells, cell, off, player=None):
        """cells [B, S^2, d]; cell [B, K], off [B, K, 2] (the real cells to decode);
        player [B] -> [B, K, n_out]."""
        x = torch.gather(cells, 1, cell[..., None].expand(-1, -1, cells.shape[-1]))
        if self.player is not None:
            x = x + self.player(player)[:, None].to(x.dtype)
        return self.mlp(torch.cat([x, self.fourier(off).to(x.dtype)], -1)).float()


# -------------------------------------------------- the state decoder (pretraining, v2)
# Entities decoded from cell + slot queries alone: type and owner are outputs, not
# inputs; the entity numbers by kind; the order queue. No counts (training places one
# query per true entity).

NUM_BIN = (
    "flying",
    "burrowed",
    "cloaked",
    "halluc",
    "snapshot",
    "powered",
    "selected",
    "engaged",
    "active",
    "blip",
    "onscreen",
    "abilities_known",
    "has_add_on",
)
NUM_CAT = dict(
    display=("display_visible", "display_snapshot", "display_hidden"),
    cloak=("cloak_unknown", "cloak_cloaked", "cloak_detected", "cloak_none"),
    inside=("passenger", "in_gas"),
)  # one-hot groups; a further class for "none set"
NUM_CONT = tuple(
    n
    for n in ENTITY_NUMS
    if n not in NUM_BIN and not any(n in g for g in NUM_CAT.values())
)
IDX = {n: i for i, n in enumerate(ENTITY_NUMS)}
I_CONT = [IDX[n] for n in NUM_CONT]
I_BIN = [IDX[n] for n in NUM_BIN]
# order queue slots decoded in full (the current order + 7 queued); the length beyond
QUEUE = 8
PROD = 8  # production queue slots (a reactored facility's full queue)
MAX_COUNT = 128  # per-cell entity count classes (0 .. 127)


def cell_targets(fb, S, n_units, n_abilities):
    """Decoder targets for a collated frame batch, grouped by latent cell only: per
    entity its cell, slot within the cell (by y offset), offset (cell units), type,
    owner (neutral / player 0 / 1), numbers by kind, and its first QUEUE orders
    (ability, target offset, targets a unit, progress).
    """
    tg = targets(fb, S, n_units)
    msk = tg["mask"]
    B, N = msk.shape
    pos = fb["ent_pos"].float()
    gg = tg["cell"] + torch.arange(B, device=pos.device)[:, None] * (S * S)
    big = float(B * S * S + 1)
    comp = torch.where(
        msk,
        gg.double() * 4 + (tg["off"][..., 1].double().clamp(-1.4, 1.4) + 1.5),
        big * 4,
    ).reshape(-1)
    order = torch.argsort(comp)
    sg = torch.where(msk.reshape(-1), gg.reshape(-1), int(big))[order]
    rank = torch.arange(B * N, device=pos.device) - torch.searchsorted(sg, sg)
    tg["slot"] = torch.empty_like(rank).scatter_(0, order, rank).view(B, N)
    tg["gg"] = gg
    tg["type"] = fb["ent_type"].clamp(0, n_units - 1)
    tg["owner"] = tg["oc"] // n_units
    num = tg["num"]
    tg["cont"], tg["bin"] = num[..., I_CONT], num[..., I_BIN]
    for name, cols in NUM_CAT.items():
        oh = num[..., [IDX[c] for c in cols]]
        tg[f"cat_{name}"] = torch.where(oh.sum(-1) > 0, oh.argmax(-1), len(cols))
    # the order queue: bag_ord rows (entity b * N + row, ability, [dx, dy, targets unit,
    # progress, position / 5])
    ent, ab, ft = fb["bag_ord_ent"], fb["bag_ord_id"], fb["bag_ord_feat"].float()
    qp = torch.round(ft[:, 4] * 5).long()
    slot = torch.where(
        qp < QUEUE, ent * QUEUE + qp, B * N * QUEUE
    )  # past the queue: a dump row
    put = (
        lambda src, fill, shape: torch.full(
            (B * N * QUEUE + 1, *shape), fill, dtype=src.dtype, device=pos.device
        )
        .index_copy_(0, slot, src)[:-1]
        .view(B, N, QUEUE, *shape)
    )
    tg["q_abil"] = put(ab, n_abilities, ())  # n_abilities = empty slot
    tg["q_tgt"] = put(ft[:, :2], 0.0, (2,))
    tg["q_unit"] = put(ft[:, 2], 0.0, ())
    tg["q_prog"] = put(ft[:, 3], 0.0, ())
    tg["q_has"] = tg["q_abil"] < n_abilities
    # the queue's full length (orders past the decoded slots included), log-scaled
    qn = torch.zeros(B * N, device=pos.device).index_add_(
        0, ent, torch.ones_like(ent, dtype=torch.float32)
    )
    tg["q_len"] = torch.log1p(qn).view(B, N)
    # the production queue: bag_prod rows (entity, ability, [progress, position / 5])
    pent, pab, pft = fb["bag_prod_ent"], fb["bag_prod_id"], fb["bag_prod_feat"].float()
    pp = torch.round(pft[:, 1] * 5).long()
    pslot = torch.where(pp < PROD, pent * PROD + pp, B * N * PROD)
    pput = (
        lambda src, fill: torch.full(
            (B * N * PROD + 1,), fill, dtype=src.dtype, device=pos.device
        )
        .index_copy_(0, pslot, src)[:-1]
        .view(B, N, PROD)
    )
    tg["p_abil"] = pput(pab, n_abilities)
    tg["p_prog"] = pput(pft[:, 0], 0.0)
    tg["p_has"] = tg["p_abil"] < n_abilities
    # the container (passengers: their transport / bunker / command center; workers in
    # gas; add-ons: their building): its row in the frame, N for none
    par = fb["ent_parent"]
    tg["parent"] = torch.where((par >= 0) & msk, par, N)
    # entities per latent cell (the count that places the queries at inference)
    tg["cell_count"] = (
        torch.zeros(B, S * S, device=pos.device)
        .scatter_add_(1, tg["cell"], msk.float())
        .long()
        .clamp(max=MAX_COUNT - 1)
    )
    return tg


class StateEntityDecoder(nn.Module):
    """Latent -> per-entity readouts for queries of (cell, slot): offset, type, owner,
    the numbers by kind (continuous, binary, categorical groups) and the order queue."""

    def __init__(self, d, n_units, n_abilities, depth=2, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.n_abilities = n_abilities
        self.slot = nn.Embedding(MAX_SLOT, d)
        self.rope = RoPE2D(d // heads, lo=1.0, hi=ROPE_HI)
        self.layers = nn.ModuleList(
            [Block(d, heads, self.rope, mlp_ratio, cross=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d)
        self.off, self.type_head, self.owner = (
            nn.Linear(d, 2),
            nn.Linear(d, n_units),
            nn.Linear(d, N_OWNERS),
        )
        self.cont, self.bin = nn.Linear(d, len(I_CONT)), nn.Linear(d, len(I_BIN))
        self.cat = nn.ModuleDict(
            {k: nn.Linear(d, len(v) + 1) for k, v in NUM_CAT.items()}
        )
        self.q_abil = nn.Linear(d, QUEUE * (n_abilities + 1))
        self.q_tgt, self.q_unit, self.q_prog = (
            nn.Linear(d, QUEUE * 2),
            nn.Linear(d, QUEUE),
            nn.Linear(d, QUEUE),
        )
        self.q_len = nn.Linear(d, 1)
        self.p_abil, self.p_prog = nn.Linear(d, PROD * (n_abilities + 1)), nn.Linear(
            d, PROD
        )
        self.ptr_q, self.ptr_k = nn.Linear(d, d), nn.Linear(
            d, d
        )  # container pointer: query . key over the frame
        self.ptr_none = nn.Linear(d, 1)
        self.count = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, MAX_COUNT)
        )  # per latent cell, from the latent

    def forward(self, lat, lat_mask, cell_slice, tg):
        c0, c1 = cell_slice
        cells = lat[:, c0:c1]
        d = lat.shape[-1]
        q = torch.gather(cells, 1, tg["cell"][..., None].expand(-1, -1, d)) + self.slot(
            tg["slot"].clamp(max=MAX_SLOT - 1)
        ).to(lat.dtype)
        qm = tg["mask"].clone()
        qm[:, 0] = True
        ang = self.rope.angles(tg["ctr"])
        for blk in self.layers:
            q = blk(q, qm, ang, lat, lat_mask)
        h = self.norm(q)
        B, N, _ = h.shape
        f = lambda m: m(h).float()
        # container pointer: each query's scores over the frame's queries (masked) and
        # "none"
        sc = (self.ptr_q(h).float() @ self.ptr_k(h).float().transpose(1, 2)) / h.shape[
            -1
        ] ** 0.5
        sc = sc.masked_fill(~tg["mask"][:, None, :], -1e4)
        ptr = torch.cat([sc, self.ptr_none(h).float()], -1)  # [B, N, N + 1]
        return dict(
            h=h,
            off=f(self.off),
            type=f(self.type_head),
            owner=f(self.owner),
            cont=f(self.cont),
            bin=f(self.bin),
            **{f"cat_{k}": f(m) for k, m in self.cat.items()},
            q_abil=f(self.q_abil).view(B, N, QUEUE, -1),
            q_tgt=f(self.q_tgt).view(B, N, QUEUE, 2),
            q_unit=f(self.q_unit),
            q_prog=f(self.q_prog),
            q_len=f(self.q_len)[..., 0],
            p_abil=f(self.p_abil).view(B, N, PROD, -1),
            p_prog=f(self.p_prog),
            ptr=ptr,
            count=self.count(cells).float(),
        )


# class weights n_c ** -power: 1 = fully balanced (rare states over-predicted: v6 at
# step 6,000 had
BALANCE_POWER = 0.5
# recall 0.95-1.00, precision 0.08-0.37 on them), 0.5 = square root, 0 = the plain mean


def _class_weighted(l, cls, w, K):
    """Mean of losses l with each sample weighted by its class's count to the power
    -BALANCE_POWER (counts over the batch's weighted samples); cls [n] in 0..K-1."""
    cnt = torch.zeros(K, device=l.device).index_add_(0, cls, w)
    cw = cnt.clamp(min=1) ** -BALANCE_POWER
    sw = cw[cls] * w
    return (l * sw).sum() / sw.sum().clamp(min=1e-8)


def balanced(loss, pos, w):
    """Class-weighted mean over positive cases (an occupied queue slot, a unit with a
    container) and the rest: each weighted by its class count ** -BALANCE_POWER, so the
    rare class isn't drowned by "empty" / "none" without being over-predicted."""
    l, p_, w_ = loss.reshape(-1), pos.reshape(-1), w.expand_as(loss).reshape(-1)
    return _class_weighted(l, p_.long(), w_, 2)


def balanced_ce(logits, target, w):
    """Cross entropy with each class weighted by its count ** -BALANCE_POWER (a state
    seen once in a thousand rows, cloaked, detected, inside a transport, weighs more
    than its count, less than the common state)."""
    return _class_weighted(
        F.cross_entropy(logits, target, reduction="none"), target, w, logits.shape[-1]
    )


def balanced_bce(logits, target, w):
    """Per flag, the BCE with positives and negatives weighted by their counts **
    -BALANCE_POWER, then averaged over flags."""
    l = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return torch.stack(
        [
            _class_weighted(l[:, j], (target[:, j] > 0.5).long(), w, 2)
            for j in range(l.shape[1])
        ]
    ).mean()


@torch.no_grad()
def hungarian(out, tg):
    """The cheapest pairing of predictions and entities over each whole frame; returns
    (prediction index, target index, valid), flat over the batch. Cost: position error
    in map units, plus the cross entropy of type and owner, plus the mean error of the
    numbers, so entities at the very same place (passengers and workers in gas sit
    exactly at their container) still pair by what they are. A latent cell's edge is a
    line we drew: a prediction landing just across it pairs with its entity at the small
    real cost.
    """
    B, N = tg["mask"].shape
    size = tg["size"][:, None, None]
    xp, xt = tg["ctr"] + out["off"] * size, tg["ctr"] + tg["off"] * size
    lt, lo = F.log_softmax(out["type"], -1), F.log_softmax(out["owner"], -1)
    ipv, itv = [], []
    for b in range(B):
        ids = torch.nonzero(tg["mask"][b], as_tuple=True)[0]
        if not len(ids):
            continue
        cost = (
            torch.cdist(xp[b, ids], xt[b, ids], p=1)
            + torch.cdist(out["cont"][b, ids], tg["cont"][b, ids], p=1)
            / tg["cont"].shape[-1]
        )
        cost = (
            cost
            - lt[b, ids][:, tg["type"][b, ids]]
            - lo[b, ids][:, tg["owner"][b, ids]]
        )
        r, t_ = linear_sum_assignment(cost.float().cpu().numpy())
        offset = b * N
        ipv.append(ids[torch.from_numpy(r).to(ids.device)] + offset)
        itv.append(ids[torch.from_numpy(t_).to(ids.device)] + offset)
    ipv, itv = torch.cat(ipv), torch.cat(itv)
    pad = torch.nonzero(~tg["mask"].reshape(-1), as_tuple=True)[0]
    return (
        torch.cat([ipv, pad]),
        torch.cat([itv, pad]),
        torch.cat(
            [
                torch.ones_like(ipv, dtype=torch.bool),
                torch.zeros_like(pad, dtype=torch.bool),
            ]
        ),
    )


def state_losses(dec, lat, lat_mask, cell_slice, tg):
    """Hungarian-matched (over each frame) losses of every readout; returns (losses,
    output, pairing)."""
    out = dec(lat, lat_mask, cell_slice, tg)
    ip, it, ok = hungarian(out, tg)
    w = ok.float()
    n = w.sum().clamp(min=1)
    fl = lambda x: x.reshape(-1, *x.shape[2:])
    P_, T_ = (lambda k: fl(out[k])[ip]), (lambda k: fl(tg[k])[it])
    avg = lambda x: (x * w).sum() / n
    # the entity's offset from the center of the cell that queried it, which may be a
    # neighbor
    off_t = (
        T_("off")
        + (T_("ctr") - fl(tg["ctr"])[ip])
        / tg["size"][it // tg["mask"].shape[1]][:, None]
    )
    out["_off_t"] = off_t
    L = dict(
        off=avg(F.smooth_l1_loss(P_("off"), off_t, beta=0.1, reduction="none").sum(-1)),
        type=avg(F.cross_entropy(P_("type"), T_("type"), reduction="none")),
        owner=avg(F.cross_entropy(P_("owner"), T_("owner"), reduction="none")),
        cont=avg(
            F.smooth_l1_loss(P_("cont"), T_("cont"), beta=0.05, reduction="none").mean(
                -1
            )
        ),
        bin=balanced_bce(P_("bin"), T_("bin"), w),
        cat=sum(balanced_ce(P_(f"cat_{k}"), T_(f"cat_{k}"), w) for k in NUM_CAT)
        / len(NUM_CAT),
    )
    qa, qt = P_("q_abil"), T_("q_abil")
    qh = T_("q_has").float()
    L["q_abil"] = balanced(
        F.cross_entropy(qa.flatten(0, 1), qt.flatten(), reduction="none").view(
            -1, QUEUE
        ),
        qh,
        w[:, None],
    )
    wq = qh * w[:, None]
    nq = wq.sum().clamp(min=1)
    L["q_tgt"] = (
        F.smooth_l1_loss(P_("q_tgt"), T_("q_tgt"), beta=0.05, reduction="none").sum(-1)
        * wq
    ).sum() / nq
    L["q_unit"] = (
        F.binary_cross_entropy_with_logits(P_("q_unit"), T_("q_unit"), reduction="none")
        * wq
    ).sum() / nq
    L["q_prog"] = (
        F.smooth_l1_loss(P_("q_prog"), T_("q_prog"), beta=0.05, reduction="none") * wq
    ).sum() / nq
    L["q_len"] = avg(
        F.smooth_l1_loss(P_("q_len"), T_("q_len"), beta=0.05, reduction="none")
    )
    pa, pt = P_("p_abil"), T_("p_abil")
    L["p_abil"] = balanced(
        F.cross_entropy(pa.flatten(0, 1), pt.flatten(), reduction="none").view(
            -1, PROD
        ),
        T_("p_has").float(),
        w[:, None],
    )
    ph = T_("p_has").float() * w[:, None]
    L["p_prog"] = (
        F.smooth_l1_loss(P_("p_prog"), T_("p_prog"), beta=0.05, reduction="none") * ph
    ).sum() / ph.sum().clamp(min=1)
    # container pointer: the target is the query matched to the parent's row (queries
    # and rows differ within a cell)
    B, N = tg["mask"].shape
    q_of_row = torch.full((B * N,), -1, dtype=torch.long, device=ip.device).scatter_(
        0, it, ip
    )
    pr = T_("parent")  # parent row in the frame, N = none
    frame = it // N
    pq = q_of_row[(frame * N + pr.clamp(max=N - 1))] - frame * N
    ptr_t = torch.where((pr < N) & (pq >= 0), pq, N)
    L["parent"] = balanced(
        F.cross_entropy(P_("ptr"), ptr_t, reduction="none"), (ptr_t < N).float(), w
    )
    out["_ptr_t"] = ptr_t
    # entities per latent cell
    L["count"] = F.cross_entropy(out["count"].flatten(0, 1), tg["cell_count"].flatten())
    return L, out, (ip, it, ok)


class DecompHead(nn.Module):
    """The decomposition check: from the combined view's latent, each player's own view
    latent at the same moment. The combined latent plus a learned per-player embedding,
    self-attention blocks (2D RoPE at the latent cells), a linear readout per token."""

    def __init__(self, d, depth=2, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.player = nn.Embedding(2, d)
        self.rope = RoPE2D(d // heads)
        self.layers = nn.ModuleList(
            [Block(d, heads, self.rope, mlp_ratio) for _ in range(depth)]
        )
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, x, mask, ang, player):
        """x [B, T, d] combined latents, player [B] (0 / 1) -> [B, T, d] that player's
        latent."""
        x = x + self.player(player)[:, None].to(x.dtype)
        for blk in self.layers:
            x = blk(x, mask, ang)
        return self.out(x)
