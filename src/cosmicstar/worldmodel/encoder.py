"""State encoder: one frame -> a fixed set of slot vectors.

Input tokens (see tokenize.py), each a sum of parts in one d-dimensional space:
    entity  = Unit[type] + Owner + Position(xy) + MLP_entity(numbers) (+ Ability[producing])
              + Ability[current order] + MLP_order(order target offset, progress, queue)
    player  = PlayerMarker + Race[race] + Owner + MLP_player(stats) + sum(Upgrade[done])
    CLS     = learned + MLP_cls(game time, game format)
    Owner   = TeamCode + PlayerCode + Relation
Unit, Ability, Upgrade and Race are separate embedding tables.

Self-attention over [CLS, players, entities] (pre-norm transformer), then readouts:

    cls       the whole game                              [B, d]
    players   one per player                              [B, P, d]   (+ player_mask)
    grid      a fixed G x G grid over the playable area   [B, G*G, d]   (+ MLP_vis(what the viewer sees there))
    windows   any queried (center, size) windows          [B, Q, d]   (if win_center/win_size given)

Entity states are the encoder's working state ("units", [B, N, d]); they're returned
for the readouts but aren't part of the representation. The same module serves the
observer view and the player view (only the owner codes / relations differ); the EMA
target encoder is a copy of it.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .tokenize import C_ENTITY, C_FORMAT, C_ORDER, C_PLAYER, C_VIS, N_CODE_EMB, N_REL


class Fourier(nn.Module):
    """sin/cos features of coordinates in [-1, 1] at octave frequencies."""

    def __init__(self, n_freq):
        super().__init__()
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(n_freq), persistent=False)

    def out_dim(self, d_in):
        return d_in * 2 * len(self.freqs)

    def forward(self, x):
        a = x[..., None] * self.freqs
        return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class TimeFeatures(nn.Module):
    """Game time (seconds) -> sin/cos at periods from 1 min to ~2.3 h (geometric) plus log
    time, so any game length is unambiguous. No periods under a minute: absolute game time
    has no meaningful rhythm at the scale of seconds, and fast-cycling features make the
    encoding of frames a few seconds apart jump around."""

    def __init__(self, n_periods=8, shortest=60.0, longest=8192.0):
        super().__init__()
        periods = torch.logspace(math.log10(shortest), math.log10(longest), n_periods)
        self.register_buffer("omega", 2 * math.pi / periods, persistent=False)

    def out_dim(self):
        return 2 * len(self.omega) + 1

    def forward(self, seconds):
        a = seconds[..., None] * self.omega
        return torch.cat([a.sin(), a.cos(), torch.log1p(seconds[..., None] / 60.0)], -1)


def mlp(d_in, d, hidden=None):
    hidden = hidden or d
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d))


class PositionEncoder(nn.Module):
    """Map position (normalized to [-1, 1]) -> d. Shared by entities and readout queries."""

    def __init__(self, d, n_freq=8):
        super().__init__()
        self.ff = Fourier(n_freq)  # finest wavelength ~ 1/128 of the map
        self.mlp = mlp(self.ff.out_dim(2), d)

    def forward(self, xy):
        return self.mlp(self.ff(xy))


class WindowReadout(nn.Module):
    """Reads a vector out of the entity states for a map window (center, size).

    The window's query is built from its position (shared position encoder) and its
    physical size in map cells, then cross-attends over the entities: those inside the
    window get full weight, those outside are softly pushed down (no hard edge), and a
    learned "empty" token is always there so an empty window reads cleanly.
    """

    def __init__(self, d, heads, pos, depth=2, mlp_ratio=4.0, falloff=4.0):
        super().__init__()
        self.heads, self.falloff, self.pos = heads, falloff, pos
        self.size_in = mlp(2, d)
        self.empty = nn.Parameter(torch.zeros(1, 1, d))
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict(dict(
                nq=nn.LayerNorm(d), nkv=nn.LayerNorm(d), q=nn.Linear(d, d), kv=nn.Linear(d, 2 * d),
                o=nn.Linear(d, d), nf=nn.LayerNorm(d), ff=mlp(d, d, int(d * mlp_ratio)))))
        self.norm = nn.LayerNorm(d)
        nn.init.normal_(self.empty, std=0.02)

    def bias(self, ent_xy, ent_mask, center, size):
        """[B, Q, K+1] additive attention bias: 0 inside the window, -(k*(r-1))^2 outside."""
        rel = (ent_xy[:, None] - center[:, :, None]) / (size[:, :, None] / 2)  # [B, Q, K, 2]
        r = rel.abs().amax(-1)  # box distance in half-window units (<= 1 inside)
        b = -(self.falloff * F.relu(r - 1)) ** 2
        b = b.masked_fill(~ent_mask[:, None], float("-inf"))
        return torch.cat([torch.zeros_like(b[..., :1]), b], -1)  # the empty token: always 0

    def forward(self, ent_h, ent_xy, ent_mask, center, size, cells):
        """center, size: [B, Q, 2] normalized; cells: [B, 2] map cells per normalized unit."""
        B, Q = center.shape[:2]
        size_cells = size * cells[:, None]  # physical size, so a cell means the same area on any map
        x = self.pos(center) + self.size_in(torch.log(size_cells / 16.0))
        kv_src = torch.cat([self.empty.expand(B, 1, -1), ent_h], 1)
        bias = self.bias(ent_xy, ent_mask, center, size).to(x.dtype)[:, None]  # [B, 1, Q, K+1]
        H = self.heads
        for L in self.layers:
            q = L["q"](L["nq"](x)).view(B, Q, H, -1).transpose(1, 2)
            k, v = L["kv"](L["nkv"](kv_src)).view(B, -1, 2, H, q.shape[-1]).unbind(2)
            a = F.scaled_dot_product_attention(q, k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias)
            x = x + L["o"](a.transpose(1, 2).reshape(B, Q, -1))
            x = x + L["ff"](L["nf"](x))
        return self.norm(x)


class StateEncoder(nn.Module):
    def __init__(self, vocab, d=256, depth=6, heads=8, mlp_ratio=4.0, dropout=0.0, grid=16):
        super().__init__()
        self.d, self.grid = d, grid
        self.unit = nn.Embedding(vocab.n_units, d, padding_idx=0)
        self.ability = nn.Embedding(vocab.n_abilities, d, padding_idx=0)
        self.upgrade = nn.Embedding(max(vocab.n_upgrades, 1), d)
        self.race = nn.Embedding(vocab.n_races, d)
        self.player_marker = nn.Parameter(torch.zeros(d))
        self.team = nn.Embedding(N_CODE_EMB, d)
        self.pcode = nn.Embedding(N_CODE_EMB, d)
        self.rel = nn.Embedding(N_REL, d)
        self.pos = PositionEncoder(d)
        self.ent_in = mlp(C_ENTITY, d)
        self.ord_in = mlp(C_ORDER, d)
        self.ply_in = mlp(C_PLAYER, d)
        self.time_ff = TimeFeatures()
        self.cls_in = mlp(self.time_ff.out_dim() + C_FORMAT, d)
        self.vis_in = mlp(C_VIS, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, int(d * mlp_ratio), dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.windows = WindowReadout(d, heads, self.pos)
        c = (torch.arange(grid) * 2 + 1) / grid - 1  # cell centers in [-1, 1]
        gy, gx = torch.meshgrid(c, c, indexing="ij")
        self.register_buffer("grid_center", torch.stack([gx, gy], -1).view(-1, 2), persistent=False)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.player_marker, std=0.02)

    def owner(self, b, g):
        return self.team(b[f"{g}_team"]) + self.pcode(b[f"{g}_pcode"]) + self.rel(b[f"{g}_rel"])

    def entity_tokens(self, b, g="ent"):
        """Input embedding of entities from their features (tokenize.entity_features). Also
        used by the action encoder for the units an action points at."""
        tok = (self.unit(b[f"{g}_type"]) + self.owner(b, g) + self.pos(b[f"{g}_xy"])
               + self.ent_in(b[f"{g}_num"]) + self.ability(b[f"{g}_produce"]))
        if self.use_orders:  # off for checkpoints trained before orders were extracted
            tok = tok + self.ability(b[f"{g}_order"]) + self.ord_in(b[f"{g}_ord"])
        return tok

    def embed(self, b):
        """Token embeddings [B, T, d] and validity mask [B, T] (True = real token)."""
        B = b["time"].shape[0]
        ent = self.entity_tokens(b)
        ups = b["ply_ups"] @ self.upgrade.weight  # sum of the done upgrades' embeddings
        ply = (self.player_marker + self.race(b["ply_race"]) + self.owner(b, "ply")
               + self.ply_in(b["ply_num"]) + ups)
        cls = self.cls + self.cls_in(torch.cat([self.time_ff(b["time"]), b["format"]], -1))[:, None]
        tokens = torch.cat([cls, ply, ent], 1)
        mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=ent.device), b["ply_mask"], b["ent_mask"]], 1)
        return tokens, mask

    grad_ckpt = False  # recompute layer activations in backward (much less memory for many frames)
    use_orders = True  # entity tokens include the unit's current order (zeros in older extractions)

    def forward(self, b):
        x, mask = self.embed(b)
        if self.grad_ckpt and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            for layer in self.blocks.layers:
                x = checkpoint(layer, x, src_key_padding_mask=~mask, use_reentrant=False)
            x = self.norm(x)
        else:
            x = self.norm(self.blocks(x, src_key_padding_mask=~mask))
        P = b["ply_mask"].shape[1]
        out = dict(cls=x[:, 0], players=x[:, 1:1 + P], player_mask=b["ply_mask"],
                   units=x[:, 1 + P:], unit_mask=b["ent_mask"])
        B = x.shape[0]
        center = self.grid_center.expand(B, -1, -1)
        size = torch.full_like(center, 2.0 / self.grid)
        out["grid"] = self.read_windows(out, b, center, size)
        if "vis" in b:
            out["grid"] = out["grid"] + self.vis_in(b["vis"])
        if "win_center" in b:
            out["windows"] = self.read_windows(out, b, b["win_center"], b["win_size"])
        return out

    def read_windows(self, out, b, center, size):
        """Window slots for any (center, size) queries, reusing an encoded frame."""
        return self.windows(out["units"], b["ent_xy"], out["unit_mask"], center, size, b["cells"])
