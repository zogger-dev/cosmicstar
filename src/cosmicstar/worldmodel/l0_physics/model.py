"""Top-level causal L0 `Physics` world model."""

from __future__ import annotations

import collections

import torch
from torch import nn

from ..data.game_store import C_DYNAMIC, PATCH, TOP
from ..nn.layers import MapDecoder, mlp
from .encoders import PLY_PROBE_IDX
from .heads import (
    ACTIVE_FLAG_BITS,
    BirthHead,
    EffPersistHead,
    EffSpawnHead,
    REG,
    REG_ALWAYS_VALID,
    RecipientHead,
    RowHead,
)
from .losses import PhysicsLossesMixin
from .trunk import PhysicsEncoder


class Physics(PhysicsLossesMixin, nn.Module):
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

