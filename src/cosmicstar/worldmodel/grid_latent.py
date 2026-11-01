"""Grid latent (DESIGN_grid_latent.md, 2026-11-01): a fixed-size latent for every map, and an
entity-list decoder from it.

Encoder: the dense encoder (dense.py) runs as before; then its entity / effect tokens and the
cells of one map level (real cells, each at its true center in map units) are scattered onto a
fixed S x S grid over the playable area (centered, scaled by the longest side, so cells are
square in map units and the shorter side is padded evenly). Each latent cell pools what fell
into it with one attention softmax (query: the cell's center; keys / values: the token plus its
offset from the cell center, so the real lattice and its scale stay visible), then a few
transformer blocks mix [globals, S x S cells]. The latent is those 1 + P + S^2 tokens.

Decoder (a probe: low gradient into the latent): per latent cell, a Poisson count for every
(owner, type), owner in {neutral, player 0, player 1} (the frame's player order); one query per
entity, built from its cell's vector, its (owner, type) and its slot within that (cell, owner,
type) group; the queries attend to each other and to the latent and read out the entity's offset
within its cell, its numbers (tokenize.entity_features) and its order. Training places the queries
from the true counts. The loss ignores the order of entities within a group: predictions and
targets are each sorted by y offset within the group and paired in that order (exact for 1-D,
on the GPU, no assignment solver; evaluation compares it with a Hungarian assignment)."""

import torch
import torch.nn.functional as F
from torch import nn

from .dense import ROPE_HI, DenseEncoder
from .direct import Block, RoPE2D
from .encoder import MapFourier, mlp
from .pyramid import ON_MAP
from .tokenize import CODE_NEUTRAL

N_OWNERS = 3                       # neutral, the frame's player 0, player 1
MAX_SLOT = 128                     # slot embeddings (a group larger than this shares the last)


def grad_scale(x, w):
    """x's value, w times its gradient."""
    return x if w == 1 else x.detach() + w * (x - x.detach())


def geometry(half, S):
    """Latent cells over the playable area: half [B, 2] (playable half-extents, map units) ->
    cell size [B] and cell centers [B, S * S, 2] (playable-centered map units, row-major in y)."""
    size = 2 * half.max(-1).values / S
    c = (torch.arange(S, device=half.device, dtype=torch.float32) + 0.5 - S / 2)
    cy, cx = torch.meshgrid(c, c, indexing="ij")
    return size, torch.stack([cx, cy], -1).view(1, S * S, 2) * size[:, None, None]


def cell_of(pos, size, S):
    """pos [B, N, 2] (playable-centered map units) -> latent cell index [B, N]."""
    ij = torch.floor(pos / size[:, None, None] + S / 2).long().clamp(0, S - 1)
    return ij[..., 1] * S + ij[..., 0]


class ScatterPool(nn.Module):
    """Each latent cell attends over the tokens that fell into it (one softmax per cell and head,
    by scatter: no padding to the fullest cell). Empty cells get nothing (their query alone)."""

    def __init__(self, d, heads, mlp_ratio=4.0):
        super().__init__()
        self.H, self.dh = heads, d // heads
        self.nx, self.nq = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.n2, self.ff = nn.LayerNorm(d), mlp(d, d, int(d * mlp_ratio))

    def forward(self, x, off, seg, q0):
        """x, off [B, N, d] (tokens, their offset embeddings); seg [B, N] in 0..C (C = dump: invalid
        tokens); q0 [B, C, d] cell queries -> [B, C, d]."""
        B, N, d = x.shape
        C = q0.shape[1]
        H, dh = self.H, self.dh
        segf = (seg + torch.arange(B, device=x.device)[:, None] * (C + 1)).reshape(-1)
        q = torch.cat([self.q(self.nq(q0)), q0.new_zeros(B, 1, d)], 1).reshape(B * (C + 1), H, dh)
        k, v = self.kv(self.nx(x) + off).reshape(B * N, 2, H, dh).unbind(1)
        score = (q[segf].float() * k.float()).sum(-1) / dh ** 0.5                       # [B N, H]
        mx = torch.full((B * (C + 1), H), -1e30, device=x.device).scatter_reduce(
            0, segf[:, None].expand(-1, H), score, "amax", include_self=True)
        w = torch.exp(score - mx[segf])
        den = torch.zeros(B * (C + 1), H, device=x.device).index_add_(0, segf, w)
        num = torch.zeros(B * (C + 1), H, dh, device=x.device).index_add_(0, segf, w[..., None] * v.float())
        a = (num / den.clamp(min=1e-6)[..., None]).view(B, C + 1, d)[:, :C].to(q0.dtype)
        y = q0 + self.o(a)
        return y + self.ff(self.n2(y))


class GridEncoder(DenseEncoder):
    """The dense encoder, then the scatter onto the fixed grid and the latent mixer (module
    docstring). Returns the latent as tokens [CLS, players, S x S cells] (the same keys the
    dynamics read), the dense output under "dense" (the posterior and the decoder's targets read
    its entity tokens), the cell size and centers."""

    def __init__(self, vocab, n_families, d=128, S=16, pool_level=1, mix_depth=4, heads=4, mlp_ratio=4.0, **kw):
        super().__init__(vocab, n_families, d, heads=heads, mlp_ratio=mlp_ratio, **kw)
        self.S, self.pool_level = S, pool_level
        self.off_mu = nn.Sequential(MapFourier(0.125, 64.0), nn.Linear(MapFourier(0.125, 64.0).out_dim(2), d), nn.GELU(),
                                    nn.Linear(d, d))
        self.cell_q = nn.Parameter(torch.zeros(d))
        self.pool = ScatterPool(d, heads, mlp_ratio)
        self.mix = nn.ModuleList([Block(d, heads, self.rope, mlp_ratio) for _ in range(mix_depth)])
        self.lat_norm = nn.LayerNorm(d)

    def forward(self, b):
        e = super().forward(b)
        x, m = e["tokens"], e["mask"]
        B, _, d = x.shape
        P = e["P"]
        g0, (c0, _) = 1 + P, e["cell_slice"]
        S = self.S
        # tokens to scatter: entities and effects (their dense outputs), and the real cells of one map level
        epos = [b["ent_pos"].float()] + ([b["eff_pos"].float()] if "eff_type" in b else [])
        epos = torch.cat(epos, 1)
        C = e["maps"][self.pool_level]
        s = 2 ** self.pool_level
        H, W = C.shape[2:]
        origin = b["grid_origin"] - b["center"]
        ys = (torch.arange(H, device=x.device, dtype=torch.float32) + 0.5) * s
        xs = (torch.arange(W, device=x.device, dtype=torch.float32) + 0.5) * s
        cxy = torch.stack(torch.broadcast_tensors(xs[None, :], ys[:, None]), -1).view(1, H * W, 2) + origin[:, None, :]
        on_map = b["grid"][:, ON_MAP:ON_MAP + 1].float()
        cvalid = (F.max_pool2d(on_map, s, s) if s > 1 else on_map)[:, 0].reshape(B, H * W) > 0
        tok = torch.cat([x[:, g0:c0], C.permute(0, 2, 3, 1).reshape(B, H * W, d).to(x.dtype)], 1)
        pos = torch.cat([epos, cxy.expand(B, -1, -1)], 1)
        valid = torch.cat([m[:, g0:c0], cvalid], 1)
        size, centers = geometry(b["half"].float(), S)
        cell = cell_of(pos, size, S)
        off = pos - torch.gather(centers, 1, cell[..., None].expand(-1, -1, 2))
        seg = torch.where(valid, cell, S * S)
        q0 = (self.pos_mu(centers) + self.cell_q).to(x.dtype)
        lat = self.pool(tok, self.off_mu(off).to(x.dtype), seg, q0)
        y = torch.cat([x[:, :g0], lat], 1)
        mask = torch.cat([m[:, :g0], torch.ones(B, S * S, dtype=torch.bool, device=x.device)], 1)
        ang = self.rope.angles(torch.cat([centers.new_zeros(B, g0, 2), centers], 1))
        for blk in self.mix:
            y = blk(y, mask, ang)
        y = self.lat_norm(y)
        return dict(tokens=y, mask=mask, ang=ang, cls=y[:, 0], players=y[:, 1:g0], P=P, cell_slice=(g0, g0 + S * S),
                    cell_hw=(S, S), dense=e, cell_size=size, centers=centers)


# ------------------------------------------------------------------ the entity decoder

def targets(fb, S, n_units):
    """What the decoder must produce for a collated frame batch fb (any batch of frames): per
    entity (units, not effects) its latent cell, (owner, type) class, slot within its (cell, owner,
    type) group (by y offset), offset within the cell (cell units, -0.5..0.5); per cell, the count
    of each (owner, type). All on the GPU, fixed shapes."""
    pos, msk = fb["ent_pos"].float(), fb["ent_mask"]
    B, N = msk.shape
    size, centers = geometry(fb["half"].float(), S)
    cell = cell_of(pos, size, S)
    ctr = torch.gather(centers, 1, cell[..., None].expand(-1, -1, 2))
    off = (pos - ctr) / size[:, None, None]
    pc = fb["ent_pcode"]
    pi = (pc[..., None] == fb["ply_pcode"][:, None, :]).float().argmax(-1).clamp(max=N_OWNERS - 2)
    owner = torch.where(pc == CODE_NEUTRAL, 0, 1 + pi)
    n_oc = N_OWNERS * n_units
    oc = owner * n_units + fb["ent_type"].clamp(0, n_units - 1)
    gkey = cell * n_oc + oc                                                     # per sample
    counts = torch.zeros(B, S * S * n_oc, device=pos.device).scatter_add_(1, gkey, msk.float()).view(B, S * S, n_oc)
    # slot = rank within the group, by y offset (padding sorts last)
    gg = gkey + torch.arange(B, device=pos.device)[:, None] * (S * S * n_oc)    # global group id
    big = float(B * S * S * n_oc + 1)
    comp = torch.where(msk, gg.double() * 4 + (off[..., 1].double().clamp(-1.4, 1.4) + 1.5), big * 4).reshape(-1)
    order = torch.argsort(comp)
    sg = torch.where(msk.reshape(-1), gg.reshape(-1), int(big))[order]
    first = torch.searchsorted(sg, sg)
    rank = torch.arange(B * N, device=pos.device) - first
    slot = torch.empty_like(rank).scatter_(0, order, rank).view(B, N)
    return dict(cell=cell, oc=oc, gg=gg, slot=slot, off=off, mask=msk, counts=counts, size=size, ctr=ctr,
                num=fb["ent_num"].float(), order=fb["ent_order"])


class EntityDecoder(nn.Module):
    """Latent -> (log count rates per cell and (owner, type), per-entity readouts for the given
    queries). Queries come from targets() (training: the true counts)."""

    def __init__(self, d, n_units, n_abilities, n_num, depth=2, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.n_units = n_units
        n_oc = N_OWNERS * n_units
        self.count = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_oc))
        nn.init.zeros_(self.count[-1].weight)
        nn.init.constant_(self.count[-1].bias, -7.0)  # ~1 entity per 3 cells over ~1000 classes
        self.oc = nn.Embedding(n_oc, d)
        self.slot = nn.Embedding(MAX_SLOT, d)
        self.rope = RoPE2D(d // heads, lo=1.0, hi=ROPE_HI)
        self.layers = nn.ModuleList([Block(d, heads, self.rope, mlp_ratio, cross=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.off, self.num, self.order = nn.Linear(d, 2), nn.Linear(d, n_num), nn.Linear(d, n_abilities)

    def rates(self, lat, cell_slice):
        c0, c1 = cell_slice
        return self.count(lat[:, c0:c1]).float()                                # [B, S^2, n_oc] log rates

    def forward(self, lat, lat_mask, cell_slice, tg):
        c0, c1 = cell_slice
        cells = lat[:, c0:c1]
        d = lat.shape[-1]
        q = torch.gather(cells, 1, tg["cell"][..., None].expand(-1, -1, d)) + self.oc(tg["oc"]).to(lat.dtype) \
            + self.slot(tg["slot"].clamp(max=MAX_SLOT - 1)).to(lat.dtype)
        qm = tg["mask"].clone()
        qm[:, 0] = True  # a frame without entities: one dummy key, so no row is all-masked
        ang = self.rope.angles(tg["ctr"])
        for blk in self.layers:
            q = blk(q, qm, ang, lat, lat_mask)
        h = self.norm(q)
        return dict(h=h, off=self.off(h).float(), num=self.num(h).float(), order=self.order(h).float())


def match(out, tg):
    """Pairs predictions with targets within each (cell, owner, type) group, both sorted by y offset:
    returns (prediction index, target index, valid), flat over the batch."""
    msk = tg["mask"].reshape(-1)
    big = 1e12  # above every group key x 4 (no sync to find the maximum)
    gg = tg["gg"].reshape(-1).double()
    yp = out["off"][..., 1].reshape(-1).double().clamp(-1.4, 1.4) + 1.5
    yt = tg["off"][..., 1].reshape(-1).double().clamp(-1.4, 1.4) + 1.5
    ip = torch.argsort(torch.where(msk, gg * 4 + yp, big))
    it = torch.argsort(torch.where(msk, gg * 4 + yt, big))
    return ip, it, msk[it]


def decode_losses(dec, lat, lat_mask, cell_slice, tg):
    """Count (Poisson, per cell, summed over classes) and matched entity losses; returns (losses,
    decoder output, the pairing)."""
    lr = dec.rates(lat, cell_slice)
    cnt = F.poisson_nll_loss(lr, tg["counts"], log_input=True, full=False, reduction="none").sum(-1).mean()
    out = dec(lat, lat_mask, cell_slice, tg)
    ip, it, ok = match(out, tg)
    w = ok.float()
    n = w.sum().clamp(min=1)
    fl = lambda x: x.reshape(-1, *x.shape[2:])
    off = (F.smooth_l1_loss(fl(out["off"])[ip], fl(tg["off"])[it], beta=0.1, reduction="none").sum(-1) * w).sum() / n
    num = (F.mse_loss(fl(out["num"])[ip], fl(tg["num"])[it], reduction="none").mean(-1) * w).sum() / n
    ordl = (F.cross_entropy(fl(out["order"])[ip], fl(tg["order"])[it].clamp(min=0), reduction="none") * w).sum() / n
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
    return (a.float() * w[..., None]).transpose(1, 2) @ b.float() / w.sum(1).clamp(min=1)[:, None, None]
