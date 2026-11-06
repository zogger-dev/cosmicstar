"""Pyramid state encoder: entities and the map grid, joined level by level.

Levels (cell size in map units; the canvas is the map at one cell per map unit, padded to a
multiple of 16):

    Every level runs 3 conv blocks on the map (depthwise 3 x 3 + MLP), then 3 attention blocks.

    s = 1, 2, 4   local levels. Each attention block is one round of local joint attention:
                  every cell attends to the cells of its 3 x 3 neighborhood (sliding, no tiling),
                  the units and effects standing in them, the global tokens (CLS, players) and a
                  learned null key; every unit attends to the 3 x 3 cells around its own cell, the
                  units in those cells, the globals and the null key; the globals attend to every
                  cell and every unit. Map tokens never see past their neighborhood here. Then the
                  map is pooled 2 x 2 (patch merge) and the units are reassigned to the coarser grid.
    s = 8, 16     full levels: CLS, players, every map cell, every unit and effect and a null token
                  in one sequence under full self-attention (after the level's conv blocks on the map).

All attention uses 2D RoPE in map units (offsets from the playable area's center): units at their
exact positions, cells at their centers, CLS / players / null at the origin. The null key gives
softmax attention a way to say "little here", so twenty zerglings don't read like one.

Entity tokens are the flat encoder's (encoder.StateEncoder.entity_tokens) plus layout 2: the
list-valued state pooled into the token (orders, production, buffs, available abilities, rally
points: per item a table embedding + an MLP of its numbers, averaged, then a linear map of the mean
and the log count per list)
and a pointer to the unit's container or building (its token through a linear map). Units keep
width d through every level, and so do the map cells (decision, user, 2026-10-31: one width
throughout, so units and cells exchange information at full width with no projection; a narrower
map at the fine levels is a later option, `widths`, which then projects units into it).

Output: the token sequence [CLS, players, units, effects, the s = 16 map cells] with its mask and
the predictor's rotation angles; units sit at [1 + P, 1 + P + N_units).
"""

import torch
import torch.nn.functional as F
from torch import nn

from .direct import Block, RoPE2D
from .encoder import StateEncoder, mlp
from .tokenize import (
    BAGS,
    C_GRID,
    GRID_CHANNELS,
    N_BUFF_IDS,
    N_CODE_EMB,
    N_EFFECT_IDS,
    N_GRID_PLAYERS,
    N_RAW_ABILITY_IDS,
)

ON_MAP = GRID_CHANNELS.index("on_map")
OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def _rot(x, ang):
    """x [..., H, dh], ang [..., dh/2] (broadcast over heads)."""
    c, s = ang.cos()[..., None, :].to(x.dtype), ang.sin()[..., None, :].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


def _pairs_within(cell_key, valid, offsets_key, n_per):
    """Pairs (query i, key j) with key j's cell in query i's 3 x 3 neighborhood.

    cell_key [M]: flat (sample, cell) key of each item; valid [M]; offsets_key [M, 9]: the 9
    neighbor keys of each query (-1 = off the canvas). Returns query and key indices (flat).
    """
    idx = torch.nonzero(valid, as_tuple=True)[0]
    keys = cell_key[idx]
    skeys, order = torch.sort(keys)
    sidx = idx[order]
    q = offsets_key.reshape(-1)
    qi = torch.arange(offsets_key.shape[0], device=q.device).repeat_interleave(
        offsets_key.shape[1]
    )
    ok = (q >= 0) & valid[qi]
    q, qi = q[ok], qi[ok]
    lo = torch.searchsorted(skeys, q)
    hi = torch.searchsorted(skeys, q, right=True)
    cnt = hi - lo
    qi = qi.repeat_interleave(cnt)
    start = lo.repeat_interleave(cnt)
    within = torch.arange(int(cnt.sum()), device=q.device) - torch.repeat_interleave(
        torch.cumsum(cnt, 0) - cnt, cnt
    )
    kj = sidx[start + within]
    return qi, kj


def _segment_softmax_merge(dense_logits, dense_vals, pair_logits, pair_q, pair_v, n_q):
    """One softmax per query over dense keys and sparse (pair) keys.

    dense_logits [Q, H, K] with -inf for missing keys, dense_vals: a function (weights [Q, H, K])
    -> [Q, H, dh] weighted sum; pair_logits [Pp, H], pair_q [Pp] query index, pair_v [Pp, H, dh].
    Returns [Q, H, dh]."""
    H = dense_logits.shape[1]
    with torch.no_grad():
        m = dense_logits.amax(-1)
        if pair_logits.numel():
            pm = torch.full((n_q, H), float("-inf"), device=m.device, dtype=m.dtype)
            pm.scatter_reduce_(
                0, pair_q[:, None].expand(-1, H), pair_logits.detach(), "amax"
            )
            m = torch.maximum(m, pm)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    w = torch.exp(dense_logits - m[..., None])
    den = w.sum(-1)
    num = dense_vals(w)
    if pair_logits.numel():
        pw = torch.exp(pair_logits - m[pair_q])
        den = den.index_add(0, pair_q, pw)
        num = num.index_add(0, pair_q, pw[..., None] * pair_v)
    return num / den.clamp(min=1e-20)[..., None]


class ConvBlock(nn.Module):
    """Map conv block, ConvNeXt style: depthwise 3 x 3 (the spatial mixing), LayerNorm, pointwise
    MLP, residual. [B, h, w, d] -> same."""

    def __init__(self, d, mlp_ratio=4.0):
        super().__init__()
        self.dw = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.norm, self.mlp = nn.LayerNorm(d), mlp(d, d, int(d * mlp_ratio))

    def forward(self, C):
        y = self.dw(C.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return C + self.mlp(self.norm(y))


class LocalLevel(nn.Module):
    """One local attention block: 3 x 3 neighborhood joint attention of map cells, units /
    effects and globals (CLS, players), with a null key, then MLPs (cells at the level's width,
    units and globals at d)."""

    def __init__(self, dl, D, mlp_ratio=4.0, dh=32):
        super().__init__()
        self.dl, self.D = dl, D
        self.dh = min(dh, dl)
        self.H = dl // self.dh
        self.rope = RoPE2D(self.dh, lo=1.0, hi=256.0)
        # joint attention
        self.nc, self.ne = nn.LayerNorm(dl), nn.LayerNorm(D)
        # units and globals join at the level's width: the same width as the map (no projection) by default
        self.down = nn.Identity() if dl == D else nn.Linear(D, dl)
        self.qkv = nn.Linear(dl, 3 * dl)
        self.o = nn.Linear(dl, dl)
        self.up = nn.Identity() if dl == D else nn.Linear(dl, D)
        self.k_null = nn.Parameter(torch.zeros(self.H, self.dh))
        self.v_null = nn.Parameter(torch.zeros(self.H, self.dh))
        nn.init.normal_(self.k_null, std=0.02)
        nn.init.normal_(self.v_null, std=0.02)
        self.nm, self.mmlp = nn.LayerNorm(dl), mlp(dl, dl, int(dl * mlp_ratio))
        self.nE, self.Emlp = nn.LayerNorm(D), mlp(D, D, int(D * mlp_ratio))

    def forward(self, C, cmask, s, E, emask, epos, G, gmask, center):
        """C [B, h, w, dl] map; cmask [B, h, w] cells on the map; s cell size (map units);
        E [B, N, D] units + effects, emask [B, N], epos [B, N, 2] centered map units;
        G [B, Gn, D] globals, gmask [B, Gn]; center [B, 2] playable center (map units).
        """
        B, h, w, dl = C.shape
        N, Gn = E.shape[1], G.shape[1]
        oc, oe, og = self.attend(C, cmask, s, E, emask, epos, G, gmask, center)
        # ---- outputs
        dt = C.dtype
        C = C + self.o(oc.view(B, h, w, dl)).to(dt)
        eo = torch.cat([oe.view(B, N, dl), og.reshape(B, Gn, dl).to(oe.dtype)], 1)
        eo = self.up(self.o(eo))
        EG = torch.cat([E, G], 1) + eo.to(E.dtype)
        C = C + self.mmlp(self.nm(C))
        EG = EG + self.Emlp(self.nE(EG))
        return C, EG[:, :N], EG[:, N:]

    debug = False

    def attend(self, C, cmask, s, E, emask, epos, G, gmask, center):
        """The joint attention itself: (cell outputs [B, h, w, H, dh] flat as [Q, H, dh], unit outputs
        [B * N, H, dh], global outputs [B, Gn, H, dh]), before the output projection."""
        B, h, w, dl = C.shape
        N, Gn, H, dh = E.shape[1], G.shape[1], self.H, self.dh
        dev = C.device
        c = self.qkv(self.nc(C)).view(B, h, w, 3, H, dh)
        eg = self.qkv(self.down(self.ne(torch.cat([E, G], 1)))).view(
            B, N + Gn, 3, H, dh
        )
        qc, kc, vc = c.unbind(3)
        qe, ke, ve = eg[:, :N].unbind(2)
        qg, kg, vg = eg[:, N:].unbind(2)
        # RoPE: cells at their centers, units at their positions, globals unrotated (the origin)
        ys = (torch.arange(h, device=dev, dtype=torch.float32) + 0.5) * s
        xs = (torch.arange(w, device=dev, dtype=torch.float32) + 0.5) * s
        cpos = torch.stack(
            torch.broadcast_tensors(xs[None, None, :], ys[None, :, None]), -1
        )  # [1, h, w, 2] raw
        cpos = cpos - center[:, None, None, :]  # [B, h, w, 2]
        cang, eang = self.rope.angles(cpos.view(B, h * w, 2)).view(
            B, h, w, -1
        ), self.rope.angles(epos.float())
        qc, kc = _rot(qc, cang), _rot(kc, cang)
        qe, ke = _rot(qe, eang), _rot(ke, eang)
        scale = dh**-0.5
        f32 = torch.float32
        # unit cells at this level
        raw = epos.float() + center[:, None, :]
        ex = (raw[..., 0] / s).floor().long().clamp(0, w - 1)
        ey = (raw[..., 1] / s).floor().long().clamp(0, h - 1)
        ecell = ey * w + ex  # [B, N]
        bN = torch.arange(B, device=dev)[:, None] * (h * w)
        # neighbor cell keys of each cell / unit (-1 off the canvas or off the map)
        cflat_valid = cmask.reshape(B * h * w)

        def nbr_keys(yy, xx, base):
            out = []
            for dy, dx in OFFSETS:
                ny, nx = yy + dy, xx + dx
                ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
                k = base + ny.clamp(0, h - 1) * w + nx.clamp(0, w - 1)
                ok = ok & cflat_valid[k]
                out.append(torch.where(ok, k, torch.full_like(k, -1)))
            return torch.stack(out, -1)

        unit_nbrs = nbr_keys(ey, ex, bN)  # [B, N, 9]

        kc_flat, vc_flat = kc.reshape(B * h * w, H, dh), vc.reshape(B * h * w, H, dh)
        qc_flat = qc.reshape(B * h * w, H, dh)
        Q = B * h * w
        knull, vnull = self.k_null.to(qc.dtype), self.v_null.to(qc.dtype)
        gvalid = gmask

        # ======== cell queries: neighbors through shifted views of the padded map (no copies)
        kp = F.pad(kc, (0, 0, 0, 0, 1, 1, 1, 1))
        vp = F.pad(vc, (0, 0, 0, 0, 1, 1, 1, 1))
        mp = F.pad(cmask.float(), (1, 1, 1, 1)) > 0
        l_nb = []
        for dy, dx in OFFSETS:
            ks = kp[:, 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            l = torch.einsum("bxyhd,bxyhd->bxyh", qc, ks).float() * scale
            l_nb.append(
                l.masked_fill(
                    ~mp[:, 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w, None],
                    float("-inf"),
                )
            )
        l_nb = torch.stack(l_nb, -1)  # [B, h, w, H, 9]
        l_g = torch.einsum("bxyhd,bghd->bxyhg", qc, kg).float() * scale
        l_g = l_g.masked_fill(~gvalid[:, None, None, None, :], float("-inf"))
        l_n = torch.einsum("bxyhd,hd->bxyh", qc, knull).float()[..., None] * scale
        dense = torch.cat([l_nb, l_g, l_n], -1).view(Q, H, 9 + Gn + 1)

        def cell_vals(wt):
            wt = wt.reshape(B, h, w, H, -1)
            a = torch.einsum("bxyhg,bghd->bxyhd", wt[..., 9 : 9 + Gn], vg.float())
            a = a + wt[..., -1:] * vnull.float()
            for o, (dy, dx) in enumerate(OFFSETS):
                a = (
                    a
                    + wt[..., o : o + 1]
                    * vp[:, 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w].float()
                )
            return a.reshape(Q, H, dh)

        # unit keys for cells: each valid unit is a key for the 9 cells around it
        ue = unit_nbrs.reshape(B * N, 9)
        evalid = emask.reshape(B * N)
        pi = torch.arange(B * N, device=dev).repeat_interleave(9)
        pq = ue.reshape(-1)
        okp = (pq >= 0) & evalid[pi]
        pi, pq = pi[okp], pq[okp]
        ke_f, ve_f = ke.reshape(B * N, H, dh), ve.reshape(B * N, H, dh)
        l_p = (qc_flat[pq].float() * ke_f[pi].float()).sum(-1) * scale
        oc = _segment_softmax_merge(dense, cell_vals, l_p, pq, ve_f[pi].float(), Q)

        # ======== unit queries
        qe_f = qe.reshape(B * N, H, dh)
        un = ue
        l_unb = (
            torch.einsum(
                "qhd,qkhd->qhk", qe_f.to(f32), kc_flat[un.clamp(min=0)].to(f32)
            )
            * scale
        )
        l_unb = l_unb.masked_fill((un < 0)[:, None, :], float("-inf"))
        ub = torch.arange(B, device=dev).repeat_interleave(N)
        l_ug = torch.einsum("qhd,qghd->qhg", qe_f.to(f32), kg[ub].to(f32)) * scale
        l_ug = l_ug.masked_fill(~gvalid[ub][:, None, :], float("-inf"))
        l_un = (qe_f.to(f32) * knull.to(f32)).sum(-1, keepdim=True) * scale
        udense = torch.cat([l_unb, l_ug, l_un], -1)
        vg_u = vg[ub]

        def unit_vals(wt):
            a = torch.einsum(
                "qhk,qkhd->qhd", wt[..., :9], vc_flat[un.clamp(min=0)].to(f32)
            )
            a = a + torch.einsum("qhg,qghd->qhd", wt[..., 9 : 9 + Gn], vg_u.to(f32))
            return a + wt[..., -1:] * vnull.to(f32)

        # unit-unit pairs: keys in the query's 3 x 3 cell neighborhood (itself included)
        ckey = (bN + ecell).reshape(B * N)
        qi, kj = _pairs_within(ckey, evalid, ue, 9)
        l_uu = (qe_f[qi].to(f32) * ke_f[kj].to(f32)).sum(-1) * scale
        oe = _segment_softmax_merge(
            udense, unit_vals, l_uu, qi, ve_f[kj].to(f32), B * N
        )

        # ======== global queries: every cell, every unit, the globals, the null key
        kall = torch.cat(
            [kg, knull.expand(B, 1, H, dh), kc.reshape(B, h * w, H, dh), ke], 1
        )
        vall = torch.cat(
            [vg, vnull.expand(B, 1, H, dh), vc.reshape(B, h * w, H, dh), ve], 1
        )
        amask = torch.cat(
            [
                gmask,
                torch.ones(B, 1, dtype=torch.bool, device=dev),
                cmask.reshape(B, h * w),
                emask,
            ],
            1,
        )
        og = F.scaled_dot_product_attention(
            qg.transpose(1, 2),
            kall.transpose(1, 2).to(qg.dtype),
            vall.transpose(1, 2).to(qg.dtype),
            attn_mask=amask[:, None, None, :],
        ).transpose(
            1, 2
        )  # [B, Gn, H, dh]
        if self.debug:
            self._dbg = dict(
                qc=qc,
                kc=kc,
                vc=vc,
                qe=qe,
                ke=ke,
                ve=ve,
                qg=qg,
                kg=kg,
                vg=vg,
                ex=ex,
                ey=ey,
                knull=knull,
                vnull=vnull,
            )
        return oc, oe, og


class PatchMerge(nn.Module):
    """[B, h, w, din] -> [B, h/2, w/2, dout]: each 2 x 2 block concatenated, normed, projected."""

    def __init__(self, din, dout):
        super().__init__()
        self.norm, self.proj = nn.LayerNorm(4 * din), nn.Linear(4 * din, dout)

    def forward(self, x):
        B, h, w, d = x.shape
        x = (
            x.view(B, h // 2, 2, w // 2, 2, d)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(B, h // 2, w // 2, 4 * d)
        )
        return self.proj(self.norm(x))


class PyramidEncoder(StateEncoder):
    """See the module docstring."""

    def __init__(
        self,
        vocab,
        n_families,
        d=128,
        widths=(128, 128, 128),
        conv_blocks=3,
        attn_blocks=3,
        full_levels=2,
        heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__(vocab, d, depth=1, heads=heads, mlp_ratio=mlp_ratio)
        del self.blocks, self.windows
        self.ruleset = None
        # layout-2 entity parts
        self.buff = nn.Embedding(N_BUFF_IDS, d)
        self.raw_ability = nn.Embedding(N_RAW_ABILITY_IDS, d)
        self.bag_in = nn.ModuleDict({k: mlp(c, d) for k, c in BAGS.items()})
        self.bag_out = nn.ModuleDict(
            {k: nn.Linear(d + 1, d) for k in BAGS}
        )  # [mean item, log1p(count)] -> d
        self.ptr = nn.Linear(d, d)
        self.src = nn.Embedding(
            N_CODE_EMB, d
        )  # which player's view a token came from (their player code)
        # effects
        self.effect = nn.Embedding(N_EFFECT_IDS, d)
        self.eff_in = mlp(1, d)
        self.eff_marker = nn.Parameter(torch.zeros(d))
        # player extras: camera, saved cameras, last command
        self.cam_save_in = nn.Linear(d, d)
        self.lcfam = nn.Embedding(max(n_families, 1), d)
        # map
        # one minimap from every player's layers: each player's visible / seen / creep with their
        # player code, embedded per cell and summed, plus the static layers; then two 3x3 convs
        self.grid_static = nn.Conv2d(C_GRID - 3 * N_GRID_PLAYERS, widths[0], 1)
        self.grid_view = nn.Conv2d(3, widths[0], 1)
        self.grid_code = nn.Embedding(N_CODE_EMB, widths[0])
        self.grid_mix = nn.Conv2d(widths[0], widths[0], 1)
        self.stem = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(widths[0], widths[0], 3, padding=1),
            nn.GELU(),
            nn.Conv2d(widths[0], widths[0], 3, padding=1),
        )
        # every level: conv_blocks conv blocks on the map, then attn_blocks attention blocks
        self.local_conv = nn.ModuleList(
            [
                nn.ModuleList([ConvBlock(wd, mlp_ratio) for _ in range(conv_blocks)])
                for wd in widths
            ]
        )
        self.local = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        LocalLevel(wd, d, mlp_ratio, dh=d // heads)
                        for _ in range(attn_blocks)
                    ]
                )
                for wd in widths
            ]
        )
        outs = list(widths[1:]) + [d]
        self.merge = nn.ModuleList(
            [PatchMerge(a, b) for a, b in zip(widths, outs)]
            + [PatchMerge(d, d) for _ in range(full_levels - 1)]
        )
        self.full_rope = RoPE2D(d // heads, lo=1.0, hi=256.0)
        self.full_conv = nn.ModuleList(
            [
                nn.ModuleList([ConvBlock(d, mlp_ratio) for _ in range(conv_blocks)])
                for _ in range(full_levels)
            ]
        )
        self.full = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Block(d, heads, self.full_rope, mlp_ratio)
                        for _ in range(attn_blocks)
                    ]
                )
                for _ in range(full_levels)
            ]
        )
        self.null_tok = nn.Parameter(torch.zeros(1, 1, d))
        self.rope = RoPE2D(d // heads)  # the predictor's angles (same head width)
        for p in (self.eff_marker, self.null_tok):
            nn.init.normal_(p, std=0.02)

    # ------------------------------------------------------------ tokens
    def entity_tokens(self, b, g="ent"):
        tok = super().entity_tokens(b, g)
        if g != "ent" or "bag_ord_ent" not in b:
            return tok
        if "ent_src" in b:
            tok = tok + self.src(b["ent_src"])
        B, N, D = tok.shape
        tables = dict(
            ord=self.ability,
            prod=self.ability,
            buff=self.buff,
            abil=self.raw_ability,
            rally=None,
        )
        add = torch.zeros(B * N, D, device=tok.device, dtype=torch.float32)
        for k in BAGS:
            ent = b[f"bag_{k}_ent"]
            if not ent.numel():
                continue
            item = self.bag_in[k](b[f"bag_{k}_feat"])
            if tables[k] is not None:
                item = item + tables[k](b[f"bag_{k}_id"])
            pooled = torch.zeros(
                B * N, D, device=tok.device, dtype=torch.float32
            ).index_add_(0, ent, item.float())
            cnt = torch.zeros(B * N, device=tok.device, dtype=torch.float32).index_add_(
                0, ent, torch.ones_like(ent, dtype=torch.float32)
            )
            mean = (
                pooled / cnt.clamp(min=1)[:, None]
            )  # a list's size is the log count, not the magnitude
            add = (
                add
                + self.bag_out[k](
                    torch.cat([mean, torch.log1p(cnt)[:, None]], -1).to(item.dtype)
                ).float()
            )
        tok = tok + add.view(B, N, D).to(tok.dtype)
        par = b["ent_parent"]
        ok = (par >= 0) & b["ent_mask"]
        pt = torch.gather(tok, 1, par.clamp(min=0)[..., None].expand(-1, -1, D))
        return tok + self.ptr(pt) * ok[..., None].to(tok.dtype)

    def minimap(self, grid, b):
        """[B, C_GRID, H, W] -> [B, w0, H, W]: sum over the players whose layers the grid holds of
        mix(GELU(view(their layers) + code(their player code))), plus static(the map's layers).
        """
        x = self.grid_static(grid[:, 3 * N_GRID_PLAYERS :])
        ok = b["ply_gridok"] & b["ply_mask"]
        for i in range(min(N_GRID_PLAYERS, ok.shape[1])):
            h = (
                self.grid_view(grid[:, 3 * i : 3 * i + 3])
                + self.grid_code(b["ply_pcode"][:, i])[:, :, None, None]
            )
            x = x + self.grid_mix(F.gelu(h)) * ok[:, i, None, None, None].to(x.dtype)
        return x

    def embed_parts(self, b):
        ent = self.entity_tokens(b)
        ups = b["ply_ups"] @ self.upgrade.weight
        ply = (
            self.player_marker
            + self.race(b["ply_race"])
            + self.owner(b, "ply")
            + self.ply_in(b["ply_num"])
            + ups
        )
        if "ply_cam" in b:
            ply = ply + self.pos(b["ply_cam"]) * b["ply_cam_ok"][..., None].to(
                ply.dtype
            )
            ok = b["ply_camsave_ok"].to(ply.dtype)  # [B, P, 8]
            cs = (self.pos(b["ply_camsave"]) * ok[..., None]).sum(2) / ok.sum(
                2, keepdim=True
            ).clamp(min=1)
            ply = ply + self.cam_save_in(cs) + self.lcfam(b["ply_lcfam"])
        cls = (
            self.cls
            + self.cls_in(torch.cat([self.time_ff(b["time"]), b["format"]], -1))[
                :, None
            ]
        )
        if "eff_type" in b:
            eff = (
                self.effect(b["eff_type"])
                + self.owner(b, "eff")
                + self.place(b, "eff")
                + self.eff_in(b["eff_num"])
                + self.eff_marker
            )
            if "eff_src" in b:
                eff = eff + self.src(b["eff_src"])
            emask, epos = b["eff_mask"], b["eff_pos"]
        else:
            B = ent.shape[0]
            eff = ent.new_zeros(B, 0, ent.shape[-1])
            emask = torch.zeros(B, 0, dtype=torch.bool, device=ent.device)
            epos = ent.new_zeros(B, 0, 2)
        return cls, ply, ent, eff, emask, epos

    # ------------------------------------------------------------ forward
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
        epos = torch.cat([b["ent_pos"], effpos.to(b["ent_pos"].dtype)], 1)
        center = (
            b["center"] - b["grid_origin"] if "grid_origin" in b else b["center"]
        )  # the playable center in canvas units
        grid = b["grid"]
        on_map = grid[:, ON_MAP : ON_MAP + 1] > 0
        C = self.stem(self.minimap(grid, b)).permute(0, 2, 3, 1)  # [B, Hc, Wc, w0]
        s = 1
        ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled()
        from torch.utils.checkpoint import checkpoint

        run = (
            (lambda f, *a: checkpoint(f, *a, use_reentrant=False))
            if ckpt
            else (lambda f, *a: f(*a))
        )
        for li, blocks in enumerate(self.local):
            cmask = (
                F.max_pool2d(on_map.float(), s, s)[:, 0] > 0 if s > 1 else on_map[:, 0]
            )
            for cb in self.local_conv[li]:
                C = run(cb, C)
            for blk in blocks:
                C, E, G = run(blk, C, cmask, s, E, emask, epos, G, gmask, center)
            C = self.merge[li](C)
            s *= 2
        # full levels (s = 8, then 16)
        for fi, blocks in enumerate(self.full):
            for cb in self.full_conv[fi]:
                C = run(cb, C)
            Bc, h, w, _ = C.shape
            cmask = F.max_pool2d(on_map.float(), s, s)[:, 0] > 0
            ys = (torch.arange(h, device=C.device, dtype=torch.float32) + 0.5) * s
            xs = (torch.arange(w, device=C.device, dtype=torch.float32) + 0.5) * s
            cpos = (
                torch.stack(torch.broadcast_tensors(xs[None, :], ys[:, None]), -1).view(
                    1, h * w, 2
                )
                - center[:, None, :]
            )
            null = self.null_tok.expand(B, -1, -1).to(G.dtype)
            x = torch.cat(
                [G, null, C.reshape(B, h * w, D).to(G.dtype), E.to(G.dtype)], 1
            )
            mask = torch.cat(
                [
                    gmask,
                    torch.ones(B, 1, dtype=torch.bool, device=x.device),
                    cmask.reshape(B, h * w),
                    emask,
                ],
                1,
            )
            pos = torch.cat(
                [x.new_zeros(B, G.shape[1] + 1, 2).float(), cpos.float(), epos.float()],
                1,
            )
            ang = self.full_rope.angles(pos)
            for blk in blocks:
                x = run(blk, x, mask, ang)
            g0 = G.shape[1]
            G, C, E = (
                x[:, :g0],
                x[:, g0 + 1 : g0 + 1 + h * w].reshape(B, h, w, D),
                x[:, g0 + 1 + h * w :],
            )
            if fi + 1 < len(self.full):
                C = self.merge[len(self.local) + fi](C)
                s *= 2
        # output: [CLS, players, units, effects, map cells (last level)]
        Bc, h, w, _ = C.shape
        cmask = F.max_pool2d(on_map.float(), s, s)[:, 0] > 0
        ys = (torch.arange(h, device=C.device, dtype=torch.float32) + 0.5) * s
        xs = (torch.arange(w, device=C.device, dtype=torch.float32) + 0.5) * s
        cpos = (
            torch.stack(torch.broadcast_tensors(xs[None, :], ys[:, None]), -1).view(
                1, h * w, 2
            )
            - center[:, None, :]
        )
        x = self.norm(torch.cat([G, E, C.reshape(B, h * w, D).to(G.dtype)], 1))
        mask = torch.cat([gmask, emask, cmask.reshape(B, h * w)], 1)
        pos = torch.cat(
            [x.new_zeros(B, G.shape[1], 2).float(), epos.float(), cpos.float()], 1
        )
        ang = self.rope.angles(pos)
        return dict(
            tokens=x,
            mask=mask,
            ang=ang,
            cls=x[:, 0],
            players=x[:, 1 : 1 + P],
            units=x[:, 1 + P : 1 + P + Nu],
            P=P,
            unit_slice=(1 + P, 1 + P + Nu),
        )
