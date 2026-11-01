"""Latent dynamics (DESIGN_latent_dynamics.md, 2026-11-01): the world model predicts the encoder's
own representation of the future, as a residual

    ŝ_{t+k} = s_t + h(s_t, k) + g(s_t + h(s_t, k), A, z)

h, passive dynamics: self-attention over the state tokens, conditioned on the horizon k: what
happens with no new commands. g, the action delta: the state tokens attend to the action tokens
(the replay commands) and the players' z tokens; it never sees the state through the actions.
Both end in a zero-initialized projection, so a fresh model predicts persistence (ŝ = s_t).

Targets come from a slow-moving copy of the encoder (exponential moving average, stop-gradient).
Anti-collapse and identifiability terms (all masked, fixed shapes):

- vicreg: per-dimension standard deviation of s_t above 1 over the batch's tokens, and small
  off-diagonal covariance (VICReg), so the latent can't shrink to a constant;
- cross_corr: the batch cross-correlation between two token streams (h's delta vs g's delta, g's
  delta vs s_t), pushed to zero, so g carries what h and the state don't;
- floor_std: g's per-dimension spread over samples with actions above a floor (g = 0 would be
  trivially uncorrelated);
- barlow: Barlow Twins between the posterior's z and the players' command summaries, where commands
  exist (the inverse side: z must mean "what was done")."""

import torch
import torch.nn.functional as F
from torch import nn

from .action_encoder import LoopTime
from .direct import Block, RoPE2D
from .encoder import mlp


class PassiveDynamics(nn.Module):
    """h: [B, T, d] state tokens + horizon -> [B, T, d] delta (self-attention with 2D RoPE)."""

    def __init__(self, d, depth, heads, mlp_ratio=4.0):
        super().__init__()
        self.rope = RoPE2D(d // heads)  # rotate() only: the angles come from the encoder
        self.time = LoopTime()
        self.hor_in = mlp(self.time.out_dim(), d)
        self.layers = nn.ModuleList([Block(d, heads, self.rope, mlp_ratio) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, s, mask, ang, hor_loops):
        x = s + self.hor_in(self.time(hor_loops))[:, None].to(s.dtype)
        for layer in self.layers:
            x = layer(x, mask, ang)
        return self.out(x)


class ActionBlock(nn.Module):
    """Queries only: the state tokens attend to the memory (actions, z), then an MLP."""

    def __init__(self, d, heads, mlp_ratio=4.0):
        super().__init__()
        self.heads = heads
        self.nq, self.nkv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.n2, self.ff = nn.LayerNorm(d), mlp(d, d, int(d * mlp_ratio))

    def forward(self, x, mem, mem_mask):
        B, T, d = x.shape
        H = self.heads
        q = self.q(self.nq(x)).view(B, T, H, d // H).transpose(1, 2)
        k, v = self.kv(self.nkv(mem)).view(B, -1, 2, H, d // H).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mem_mask[:, None, None, :])
        x = x + self.o(a.transpose(1, 2).reshape(B, T, d))
        return x + self.ff(self.n2(x))


class ActionDynamics(nn.Module):
    """g: state tokens (after h) attending to the action and z tokens -> [B, T, d] delta. The
    memory always holds each player's z token, so no row is empty. rope given (--g_self): each
    block also self-attends over the state tokens first (2D RoPE), so a command's effect can move
    between tokens (units leaving one cell arrive in another)."""

    def __init__(self, d, depth, heads, mlp_ratio=4.0, rope=None):
        super().__init__()
        self.time = LoopTime()
        self.hor_in = mlp(self.time.out_dim(), d)
        self.self_attn = rope is not None
        self.rope = rope
        self.layers = nn.ModuleList([Block(d, heads, rope, mlp_ratio, cross=True) if rope is not None
                                     else ActionBlock(d, heads, mlp_ratio) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, hor_loops, mem, mem_mask, mask=None, ang=None):
        x = x + self.hor_in(self.time(hor_loops))[:, None].to(x.dtype)
        for layer in self.layers:
            x = layer(x, mask, ang, mem, mem_mask) if self.self_attn else layer(x, mem, mem_mask)
        return self.out(x)


# ------------------------------------------------------------------ masked statistics

def _moments(x, w):
    """x [..., d], w [...] (0/1 weights) -> (mean [d], centered x, n)."""
    x, w = x.float(), w.float()
    n = w.sum().clamp(min=2)
    mean = (x * w[..., None]).sum(tuple(range(x.dim() - 1))) / n
    return mean, (x - mean) * w[..., None], n


def vicreg(x, w, target_std=1.0, eps=1e-4):
    """(variance term, covariance term) over the weighted tokens of x [..., d]."""
    _, xc, n = _moments(x, w)
    xc = xc.reshape(-1, x.shape[-1])
    var = (xc * xc).sum(0) / (n - 1)
    v = F.relu(target_std - torch.sqrt(var + eps)).mean()
    cov = xc.T @ xc / (n - 1)
    off = cov - torch.diag(torch.diag(cov))
    return v, (off * off).sum() / x.shape[-1]


def cross_corr(a, b, w, eps=1e-4):
    """Mean squared entry of the cross-correlation between a and b [..., d] over weighted tokens."""
    _, ac, n = _moments(a, w)
    _, bc, _ = _moments(b, w)
    ac, bc = ac.reshape(-1, a.shape[-1]), bc.reshape(-1, b.shape[-1])
    sa = torch.sqrt((ac * ac).sum(0) / (n - 1) + eps)
    sb = torch.sqrt((bc * bc).sum(0) / (n - 1) + eps)
    c = (ac / sa).T @ (bc / sb) / (n - 1)
    return (c * c).mean()


def floor_std(x, w, floor, eps=1e-4):
    """Hinge on the per-dimension spread of x [..., d] over weighted tokens."""
    _, xc, n = _moments(x, w)
    xc = xc.reshape(-1, x.shape[-1])
    return F.relu(floor - torch.sqrt((xc * xc).sum(0) / (n - 1) + eps)).mean()


def barlow(a, b, w, off_weight, eps=1e-4):
    """Barlow Twins over weighted pairs: the cross-correlation of standardized a, b [..., d] toward
    the identity (diagonal to 1, off-diagonal to 0, the latter weighted)."""
    _, ac, n = _moments(a, w)
    _, bc, _ = _moments(b, w)
    ac, bc = ac.reshape(-1, a.shape[-1]), bc.reshape(-1, b.shape[-1])
    ac = ac / torch.sqrt((ac * ac).sum(0) / (n - 1) + eps)
    bc = bc / torch.sqrt((bc * bc).sum(0) / (n - 1) + eps)
    c = ac.T @ bc / (n - 1)
    on = ((1 - torch.diagonal(c)) ** 2).sum()
    off = (c * c).sum() - (torch.diagonal(c) ** 2).sum()
    return on + off_weight * off


def vicreg_pair(a, b, w, w_cov=0.04, queue=None):
    """VICReg between paired a, b [..., d] over weighted pairs: the mean squared distance of each pair,
    plus each side's variance hinge and off-diagonal covariance (the 25 : 25 : 1 weighting of VICReg,
    divided by 25). A direct gradient per pair, so it works with tens of pairs where Barlow Twins'
    batch cross-correlation is mostly noise. queue: recent (a, b, w) rows, detached, added to the
    variance / covariance statistics only (a memory bank, as in MoCo)."""
    wf = w.float()
    mse = (((a.float() - b.float()) ** 2).mean(-1) * wf).sum() / wf.sum().clamp(min=1)
    a2, b2, w2 = a.reshape(-1, a.shape[-1]).float(), b.reshape(-1, b.shape[-1]).float(), w.reshape(-1)
    if queue:
        a2 = torch.cat([q[0].float() for q in queue] + [a2])
        b2 = torch.cat([q[1].float() for q in queue] + [b2])
        w2 = torch.cat([q[2] for q in queue] + [w2])
    va, ca = vicreg(a2, w2)
    vb, cb = vicreg(b2, w2)
    return mse + va + vb + w_cov * (ca + cb)


@torch.no_grad()
def ema_update(target, online, tau):
    """target <- tau * target + (1 - tau) * online (parameters); buffers copied."""
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.lerp_(po.detach(), 1 - tau)
    for bt, bo in zip(target.buffers(), online.buffers()):
        bt.copy_(bo)


N_RAW_DELTA = 2 + 49  # position (fraction of the playable area, ent_xy) + the entity numbers (tokenize.ENTITY_NUMS)


class DeltaInverse(nn.Module):
    """What happened to each unit, read only from its change Δ = s_{t+k} − s_t and the horizon
    (never s_t itself): moved, touched by a command, order changed, the new order; and a Barlow
    Twins alignment between the latent change and the unit's raw feature change (position, HP,
    shields, energy, build progress, cooldowns, ...: everything that changed, without decoding
    it; the flags alone let Δ say "this unit changed" without saying how). With
    gradient through both encodings, a latent that doesn't change over time (Δ ≈ 0) can't
    answer, which is the pressure against collapse in time; static units are correctly
    "nothing happened", so the slowness of most of the state is respected, and a clock stamped
    into every token is the same for all units, so it can't say which of them moved."""

    def __init__(self, d, n_orders):
        super().__init__()
        self.time = LoopTime()
        self.body = mlp(d + self.time.out_dim(), d)
        self.flags = nn.Linear(d, 3)        # moved, touched, order changed
        self.order = nn.Linear(d, n_orders)
        self.proj = nn.Linear(d, 64)                 # the latent change, for the Barlow Twins pairing
        self.raw_in = mlp(N_RAW_DELTA, 64)           # the raw feature change, its other side


    def forward(self, delta, hor_loops):
        """delta [B, N, d], hor_loops [B] -> (flag logits [B, N, 3], order logits [B, N, n_orders],
        the latent change projected for Barlow Twins [B, N, 64])."""
        t = self.time(hor_loops)[:, None].expand(-1, delta.shape[1], -1).to(delta.dtype)
        h = F.gelu(self.body(torch.cat([delta, t], -1)))
        return self.flags(h), self.order(h), self.proj(delta)
