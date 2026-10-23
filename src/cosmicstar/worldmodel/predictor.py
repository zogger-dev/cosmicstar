"""Predictor: from a few encoded context frames, predict the target encoder's slots at t + delta.

Context tokens are every slot of every context frame (CLS, players, grid cells), each
tagged with what kind of slot it is, where it is (grid cell / player index) and when it
is (seconds relative to t). Query tokens - one per target slot - start from that same
slot in the latest context frame, plus the kind and place tags, the horizon delta and a
learned "to predict" vector: the predictor starts as "nothing changes" and learns the
change (a residual prediction; unlike I-JEPA's masked patches, our target slot is always
in the context, one step earlier). A transformer runs
over [context, queries]; context tokens attend only to context, queries to everything.
The outputs at the query positions are the predicted target slots.

Actions (optional): the action encoder's summaries, one per (segment, player), join the
context as a fourth kind of token, tagged with the player and the segment's start time.
Segments before the present are what was done; the segment from the present to the target
is the conditioning ("given what the players do next, what does the game look like?").
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import Fourier, TimeFeatures, mlp

SLOT_CLS, SLOT_PLAYER, SLOT_GRID, SLOT_ACTION = 0, 1, 2, 3
MAX_PLAYERS = 8


class SlotTags(nn.Module):
    """kind + place embedding shared by context and query tokens."""

    def __init__(self, d, grid):
        super().__init__()
        self.kind = nn.Embedding(4, d)
        self.player = nn.Embedding(MAX_PLAYERS, d)
        self.cell_ff = Fourier(5)
        self.cell = mlp(self.cell_ff.out_dim(2), d)
        c = (torch.arange(grid) * 2 + 1) / grid - 1
        gy, gx = torch.meshgrid(c, c, indexing="ij")
        self.register_buffer("cells", torch.stack([gx, gy], -1).view(-1, 2), persistent=False)

    def forward(self, n_players, device):
        """[1 + P + G, d] tags in slot order (CLS, players, grid)."""
        cls = self.kind.weight[SLOT_CLS][None]
        ply = self.kind.weight[SLOT_PLAYER][None] + self.player.weight[:n_players]
        grd = self.kind.weight[SLOT_GRID][None] + self.cell(self.cell_ff(self.cells))
        return torch.cat([cls, ply, grd], 0)


def slots_of(out):
    """Encoder output -> [B, 1 + P + G, d] slot vectors and their mask."""
    B = out["cls"].shape[0]
    x = torch.cat([out["cls"][:, None], out["players"], out["grid"]], 1)
    ones = torch.ones(B, 1, dtype=torch.bool, device=x.device)
    m = torch.cat([ones, out["player_mask"], torch.ones(out["grid"].shape[:2], dtype=torch.bool, device=x.device)], 1)
    return x, m


class Predictor(nn.Module):
    def __init__(self, d=256, depth=4, heads=8, mlp_ratio=4.0, grid=16):
        super().__init__()
        self.tags = SlotTags(d, grid)
        self.time = TimeFeatures(n_periods=8, shortest=1.0, longest=256.0)
        self.time_in = mlp(self.time.out_dim() + 1, d)  # + sign (past / future)
        self.inp = nn.Linear(d, d)
        self.act_in = nn.Linear(d, d)
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, int(d * mlp_ratio), 0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, d)
        nn.init.normal_(self.query, std=0.02)

    def time_tag(self, seconds):
        s = seconds.float()
        return self.time_in(torch.cat([self.time(s.abs()), torch.sign(s)[..., None]], -1))

    def forward(self, ctx, ctx_mask, ctx_dt, delta, act=None, act_mask=None, act_dt=None):
        """ctx: [B, K, S, d] slots of K context frames; ctx_mask: [B, K, S]; ctx_dt: [B, K]
        seconds relative to t (<= 0); delta: [B] seconds ahead.
        act (optional): [B, G, P, d] action summaries per (segment, player); act_mask [B, G, P];
        act_dt [B, G] segment start times relative to t. Returns [B, S, d]."""
        B, K, S, d = ctx.shape
        P = S - 1 - self.tags.cells.shape[0]
        tags = self.tags(P, ctx.device)  # [S, d]
        c = self.inp(ctx) + tags[None, None] + self.time_tag(ctx_dt)[:, :, None]
        c = c.reshape(B, K * S, d)
        c_pad = ~ctx_mask.reshape(B, K * S)
        if act is not None:
            G, PA = act.shape[1], act.shape[2]
            a = (self.act_in(act) + self.tags.kind.weight[SLOT_ACTION] + self.tags.player.weight[:PA][None, None]
                 + self.time_tag(act_dt)[:, :, None])
            c = torch.cat([c, a.reshape(B, G * PA, d)], 1)
            c_pad = torch.cat([c_pad, ~act_mask.reshape(B, G * PA)], 1)
        q = self.inp(ctx[:, -1]) + self.query + tags[None] + self.time_tag(delta)[:, None]
        x = torch.cat([c, q], 1)
        T = c.shape[1]
        # context attends to context only; queries attend to everything (incl. other queries)
        attn = torch.zeros(T + S, T + S, dtype=torch.bool, device=x.device)
        attn[:T, T:] = True  # True = blocked
        pad = torch.cat([c_pad, ~ctx_mask[:, -1]], 1)  # queries mirror the last frame's slots
        x = self.blocks(x, mask=attn, src_key_padding_mask=pad)
        return self.out(self.norm(x[:, T:]))
