"""Inverse dynamics: what did each player do between two frames?

Learned queries, one per player, cross-attend over the slots of a frame and of the frame one
step later (tagged "before" / "after") and emit a small vector per player: the latent action.
train_window.py trains it to agree with the command encoder's summary of the same interval
(InfoNCE) and to drive the predictor, so it has to encode the part of the change that the
player's commands explain, and nothing the commands don't say (it can't just copy the next
frame: the commands don't contain it).
"""

import torch
from torch import nn

from .action_encoder import RelTime
from .encoder import mlp


class InverseModel(nn.Module):
    def __init__(self, d=256, d_z=64, heads=8, depth=2, mlp_ratio=4.0, max_players=8):
        super().__init__()
        self.before = nn.Parameter(torch.zeros(1, 1, d))
        self.after = nn.Parameter(torch.zeros(1, 1, d))
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        self.player = nn.Embedding(max_players, d)
        self.time = RelTime()
        self.hor = mlp(self.time.out_dim(), d)  # how far apart the two frames are (variable horizons)
        self.layers = nn.ModuleList([nn.ModuleDict(dict(
            nq=nn.LayerNorm(d), nkv=nn.LayerNorm(d), attn=nn.MultiheadAttention(d, heads, batch_first=True),
            nf=nn.LayerNorm(d), ff=mlp(d, d, int(d * mlp_ratio)))) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_z))
        for p in (self.before, self.after, self.query):
            nn.init.normal_(p, std=0.02)

    def forward(self, slots_a, slots_b, mask_a, mask_b, P, horizon_s=None):
        """slots_a / slots_b [N, S, d]: a frame and the frame one step later; masks [N, S]
        (True = real slot); horizon_s [N]: seconds between the frames, when it varies.
        Returns the latent action of each of the P players, [N, P, d_z]."""
        N = slots_a.shape[0]
        kv = torch.cat([slots_a + self.before.to(slots_a.dtype), slots_b + self.after.to(slots_b.dtype)], 1)
        pad = ~torch.cat([mask_a, mask_b], 1)
        q = (self.query + self.player.weight[:P][None]).expand(N, -1, -1).to(kv.dtype)
        if horizon_s is not None:
            q = q + self.hor(self.time(horizon_s))[:, None].to(kv.dtype)
        for layer in self.layers:
            k = layer["nkv"](kv)
            a, _ = layer["attn"](layer["nq"](q), k, k, key_padding_mask=pad, need_weights=False)
            q = q + a
            q = q + layer["ff"](layer["nf"](q))
        return self.out(q)
