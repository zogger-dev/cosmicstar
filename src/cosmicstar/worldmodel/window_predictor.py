"""Block-causal world model over a window of consecutive frames (LLM-style "next step").

The window is one interleaved sequence, a block per frame:

    [S0 A0] [S1 A1] [S2 A2] ... [S(T-1) A(T-1)]

    Sk  the context encoder's slots of frame k (CLS, players, grid cells)
    Ak  the action encoder's summaries of what each player did between frame k and k+1

Attention is block-causal: a token sees its own block and every earlier block, never a
later one. So in one pass every prefix of the window is a training example (teacher
forcing, like an LLM learning every next token of a document at once).

Each state slot's output predicts the same slot h frames later, for each horizon h in
`horizons` (one small head per horizon). It has seen the frames up to k and the actions up
to and including Ak, so the one-step prediction is action-conditioned; longer horizons
predict without the actions in between (rollouts feed predictions back step by step).
Starting from its own slot, "nothing changes" is the natural default for each slot.

(A policy head predicting Ak from Sk will need Sk not to see Ak; that's a mask refinement
for later, see the module discussion in README.)
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .encoder import mlp
from .predictor import SLOT_ACTION, SlotTags
from .action_encoder import RelTime


def block_causal_attention(q, k, v, W, pad=None, chunk=16):
    """Attention where token i may attend token j iff block(j) <= block(i), blocks being runs of
    W consecutive tokens (one frame). Queries go in chunks of `chunk` blocks; a chunk only
    reads keys up to its own last block, so the future is skipped instead of computed and the
    mask is [chunk*W, keys] rather than [N, N] (N^2 bytes: ~2 GB for a 30 s window).
    q, k, v [B, H, N, dh]; pad [B, N] True = padded (never attended)."""
    N = q.shape[2]
    T = N // W
    block = torch.arange(T, device=q.device).repeat_interleave(W)
    outs = []
    for b0 in range(0, T, chunk):
        b1 = min(T, b0 + chunk)
        qs, qe = b0 * W, b1 * W
        m = block[None, :qe] <= block[qs:qe, None]                      # [qc, keys]
        if pad is not None:
            m = m[None, None] & ~pad[:, None, None, :qe]                # [B, 1, qc, keys]
        outs.append(F.scaled_dot_product_attention(q[:, :, qs:qe], k[:, :, :qe], v[:, :, :qe], attn_mask=m))
    return torch.cat(outs, 2)


class Block(nn.Module):
    """Pre-norm transformer layer with block-causal attention (block_causal_attention)."""

    def __init__(self, d, heads, mlp_ratio):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = mlp(d, d, int(d * mlp_ratio))

    def forward(self, x, W, pad):
        B, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = block_causal_attention(q, k, v, W, pad)
        x = x + self.o(a.transpose(1, 2).reshape(B, N, d))
        return x + self.ff(self.n2(x))


class WindowPredictor(nn.Module):
    def __init__(self, d=256, depth=4, heads=8, mlp_ratio=4.0, grid=16, horizons=(1, 4, 16), grad_ckpt=False,
                 residual=False, horizon_cond=False):
        super().__init__()
        self.horizons, self.heads, self.grad_ckpt, self.residual = tuple(horizons), heads, grad_ckpt, residual
        self.tags = SlotTags(d, grid)
        self.time = RelTime(n_periods=8, shortest=0.25, longest=128.0)
        self.time_in = mlp(self.time.out_dim(), d)
        # variable horizon: one head (horizons=("var",)) told how far ahead to predict (forward's horizon_s)
        self.hor_in = mlp(self.time.out_dim(), d) if horizon_cond else None
        self.state_in = nn.Linear(d, d)
        self.act_in = nn.Linear(d, d)
        self.layers = nn.ModuleList([Block(d, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.ModuleDict({str(h): mlp(d, d) for h in self.horizons})
        if residual:  # prediction = the slot's own (normalized) present + a change that starts at 0
            for head in self.out.values():
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def forward(self, slots, slot_mask, times, act=None, act_mask=None, base=None, return_state=False, horizon_s=None):
        """slots [B, T, S, d] (S = 1 + P + G), slot_mask [B, T, S], times [B, T] seconds from the
        window start; act [B, T, P, d] summaries for each frame interval, act_mask [B, T, P].
        base [B, T, S, d] (with residual=True): the present in target space, i.e. the target
        encoder's normalized slots of frame k, so a zero change is exactly "nothing changes".
        Returns {h: [B, T, S, d]}: position k's prediction of frame k + h (valid for k < T - h)."""
        B, T, S, d = slots.shape
        P = S - 1 - self.tags.cells.shape[0]
        tt = self.time_in(self.time(times))                                    # [B, T, d]
        x = self.state_in(slots) + self.tags(P, slots.device)[None, None] + tt[:, :, None]
        if horizon_s is not None:  # [B] seconds ahead, when the horizon varies between batches
            x = x + self.hor_in(self.time(horizon_s))[:, None, None]
        blocks, pad = [x], [~slot_mask]
        if act is not None:
            PA = act.shape[2]
            a = (self.act_in(act) + self.tags.kind.weight[SLOT_ACTION]
                 + self.tags.player.weight[:PA][None, None] + tt[:, :, None])
            blocks.append(a)
            pad.append(~act_mask)
        x = torch.cat(blocks, 2)                                              # [B, T, S(+P), d]
        W = x.shape[2]
        x = x.reshape(B, T * W, d)
        pad = torch.cat(pad, 2).reshape(B, T * W)
        # padded keys are never attended (every query keeps at least block 0's CLS, so no row
        # is fully masked); without padding (e.g. all games the same format) skip the pad mask
        pad = pad if bool(pad.any()) else None
        for layer in self.layers:
            if self.grad_ckpt and self.training:
                x = checkpoint(layer, x, W, pad, use_reentrant=False)
            else:
                x = layer(x, W, pad)
        x = self.norm(x).view(B, T, W, d)[:, :, :S]
        if not self.residual:
            preds = {h: self.out[str(h)](x) for h in self.horizons}
        else:
            assert base is not None, "residual prediction needs the present in target space"
            preds = {h: base + self.out[str(h)](x).float() for h in self.horizons}
        # x: the model's state at each frame after attending over everything so far (frames
        # and actions up to k): what it understands about the game at that moment
        return (preds, x) if return_state else preds
