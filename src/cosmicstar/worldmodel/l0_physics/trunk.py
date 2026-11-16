"""Causal tubelet spatiotemporal trunk (`Tower`, `Block`, `PhysicsEncoder`)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..data.game_store import C_LIVE, C_STATIC
from ..nn.attention import Attend, Heads, PairAttend
from ..nn.layers import FeedForward, ResBlock, mlp
from ..nn.rope import ROPE_SHORTEST, RoPE1D, RoPE2D, cos_sin
from .encoders import ActionEncoder, EntityEncoder, I

GRID = 16  # map anchor cells per side
N_CODES = 5  # action-to-entity pair bias codes (see pointer_pairs)
N_UNIT_EDGES = 6  # eng->tgt, tgt->eng, child->par, par->child, ord0->tgt, tgt->ord0


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
