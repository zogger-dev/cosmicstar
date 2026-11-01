"""Inverse dynamics: what did each player do between two frames?

Learned queries, one per player, cross-attend over the slots of a frame and of the frame one
step later (tagged "before" / "after") and emit a small vector per player: the latent action.
The window model trained it to agree with the command encoder's summary of the same interval
(InfoNCE) and to drive the predictor, so it has to encode the part of the change that the
player's commands explain, and nothing the commands don't say (it can't just copy the next
frame: the commands don't contain it).
"""

import torch
from torch import nn

from .action_encoder import LoopTime
from .encoder import mlp


class InverseModel(nn.Module):
    def __init__(self, d=256, d_z=64, heads=8, depth=2, mlp_ratio=4.0, max_players=8):
        super().__init__()
        self.before = nn.Parameter(torch.zeros(1, 1, d))
        self.after = nn.Parameter(torch.zeros(1, 1, d))
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        self.player = nn.Embedding(max_players, d)
        self.time = LoopTime()
        self.hor = mlp(self.time.out_dim(), d)  # how far apart the two frames are (variable horizons)
        self.layers = nn.ModuleList([nn.ModuleDict(dict(
            nq=nn.LayerNorm(d), nkv=nn.LayerNorm(d), attn=nn.MultiheadAttention(d, heads, batch_first=True),
            nf=nn.LayerNorm(d), ff=mlp(d, d, int(d * mlp_ratio)))) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_z))
        for p in (self.before, self.after, self.query):
            nn.init.normal_(p, std=0.02)

    def forward(self, slots_a, slots_b, mask_a, mask_b, P, horizon_loops=None):
        """slots_a / slots_b [N, S, d]: a frame and the frame one step later; masks [N, S]
        (True = real slot); horizon_loops [N]: game loops between the frames, when it varies.
        Returns the latent action of each of the P players, [N, P, d_z]."""
        N = slots_a.shape[0]
        kv = torch.cat([slots_a + self.before.to(slots_a.dtype), slots_b + self.after.to(slots_b.dtype)], 1)
        pad = ~torch.cat([mask_a, mask_b], 1)
        q = (self.query + self.player.weight[:P][None]).expand(N, -1, -1).to(kv.dtype)
        if horizon_loops is not None:
            q = q + self.hor(self.time(horizon_loops))[:, None].to(kv.dtype)
        for layer in self.layers:
            k = layer["nkv"](kv)
            a, _ = layer["attn"](layer["nq"](q), k, k, key_padding_mask=pad, need_weights=False)
            q = q + a
            q = q + layer["ff"](layer["nf"](q))
        return self.out(q)


C_INTENT_ORD = 5  # per order: target offset (x, y), targets a unit, progress, queue position (tokenize.BAGS["ord"])
C_INTENT_CMD = 8  # per command: no target / point / unit, target offset (x, y), queued, time into the window (2)


class EnemyIntent(nn.Module):
    """What the posterior knows about the enemy in a player view and the prior doesn't
    (2026-10-31): for each enemy unit the viewer sees, its true order queue and the enemy's
    commands in the window that selected it (stim, blink, a move), read from the enemy's own
    view. Each list is pooled per unit (mean + log count, like the encoder's lists) and the
    result is added to that unit's present token, for the posterior only. The last layer starts
    at zero, so a fresh model's posterior is unchanged."""

    def __init__(self, n_abilities, d):
        super().__init__()
        self.ability = nn.Embedding(n_abilities, d)
        self.ord_in = nn.Linear(C_INTENT_ORD, d)
        self.cmd_in = nn.Linear(C_INTENT_CMD, d)
        self.out = mlp(2 * d + 2, d)
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, priv, n_tokens, d_dtype):
        """priv: {ord,cmd}_{idx, ab, feat} flat over the batch (idx = sample * N + unit row);
        n_tokens = B * N. Returns [B * N, d] to add to the present's unit tokens."""
        pooled = []
        for k, lin in (("ord", self.ord_in), ("cmd", self.cmd_in)):
            idx = priv[f"{k}_idx"]
            x = self.ability(priv[f"{k}_ab"]) + lin(priv[f"{k}_feat"])
            s = x.new_zeros(n_tokens, x.shape[-1]).index_add_(0, idx, x)
            c = x.new_zeros(n_tokens).index_add_(0, idx, torch.ones_like(idx, dtype=x.dtype))
            pooled += [s / c.clamp(min=1)[:, None], torch.log1p(c)[:, None]]
        return self.out(torch.cat(pooled, -1)).to(d_dtype)
