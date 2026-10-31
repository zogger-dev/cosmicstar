"""Direct state-space prediction: the state at t (+ the commands in [t, t+k)) -> each entity's
state at t+k, read by probes, for every horizon k up to a few seconds.

Why not JEPA at this level: SC2 state is already an abstraction (a list of entities with exact
attributes), so regressing it wastes no capacity on unpredictable detail, needs no target
encoder and cannot collapse. Given everyone's commands the engine is deterministic, so the
only uncertainty in the observer view is the commands themselves; a latent z per (sample,
player) absorbs them when they are withheld (action dropout) and should carry nothing when
they are given. That gap is the first thing to measure.

Pieces (scripts/train_direct.py wires them):

    RoPEEncoder       the state encoder's token embedding (entities, players, CLS) under
                      self-attention with 2D rotary positions in map units, so a score depends
                      on the offset between two things, with sign (north of me != south of me)
    DirectPredictor   the encoded present + a horizon embedding + z, through blocks of
                      self-attention (RoPE) and cross-attention to the window's action tokens
                      (the action encoder's causal transformer), so the entities read what was
                      done to them
    Probes            per-entity heads at t+k: exit (present / died / gone), position offset
                      (two-hot per axis, map units), change in hp / shield / energy fraction and
                      build progress (two-hot), current order's ability (softmax over the vocabulary)
    PriorHead         p(z | present, commands or "withheld") as a mixture of Gaussians, in the
                      format of latent.py so its KL / sampling helpers apply

Two-hot targets (MuZero style) make a unit that may go left or right a bimodal prediction
instead of a blur between; the expected value of the distribution is the point estimate.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .action_encoder import RelTime
from .encoder import StateEncoder, mlp
from .latent import LV_MIN

EXIT_PRESENT, EXIT_DIED, EXIT_GONE = 0, 1, 2  # gone: left the observation alive (loaded, morphed to a new tag, ...)
N_EXIT = 3


class RoPE2D(nn.Module):
    """Rotary positions over the plane: half of each head's pairs rotate with x, half with y,
    at wavelengths from `lo` to `hi` map units (geometric). Tokens without a place (CLS,
    players) sit at the origin, as RoPE-ViT does with its class token."""

    def __init__(self, dh, lo=2.0, hi=256.0):
        super().__init__()
        assert dh % 4 == 0, "head dim must be a multiple of 4 for 2D RoPE"
        n = dh // 4
        self.register_buffer("omega", 2 * math.pi / torch.logspace(math.log10(lo), math.log10(hi), n), persistent=False)

    def angles(self, pos):
        """pos [B, T, 2] in map units -> [B, T, dh / 2] rotation angles."""
        return torch.cat([pos[..., 0:1] * self.omega, pos[..., 1:2] * self.omega], -1)

    @staticmethod
    def rotate(x, ang):
        """x [B, H, T, dh], ang [B, T, dh / 2]."""
        c, s = ang.cos()[:, None].to(x.dtype), ang.sin()[:, None].to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Block(nn.Module):
    """Pre-norm transformer block: self-attention with 2D RoPE, optional cross-attention to a
    memory (no positions), MLP."""

    def __init__(self, d, heads, rope, mlp_ratio=4.0, cross=False):
        super().__init__()
        self.heads, self.rope = heads, rope
        self.n1, self.qkv, self.o = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.cross = cross
        if cross:
            self.nq, self.nkv = nn.LayerNorm(d), nn.LayerNorm(d)
            self.q, self.kv, self.co = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.n2, self.ff = nn.LayerNorm(d), mlp(d, d, int(d * mlp_ratio))

    def forward(self, x, mask, ang, mem=None, mem_mask=None):
        B, T, d = x.shape
        H = self.heads
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, H, d // H).permute(2, 0, 3, 1, 4)
        q, k = self.rope.rotate(q, ang), self.rope.rotate(k, ang)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        x = x + self.o(a.transpose(1, 2).reshape(B, T, d))
        if self.cross and mem is not None:
            q = self.q(self.nq(x)).view(B, T, H, -1).transpose(1, 2)
            k, v = self.kv(self.nkv(mem)).view(B, -1, 2, H, d // H).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mem_mask[:, None, None, :])
            x = x + self.co(a.transpose(1, 2).reshape(B, T, d))
        return x + self.ff(self.n2(x))


def positions(b):
    """Map positions in map units of the token sequence [CLS, players, entities]: zeros for
    CLS and players, the entities' normalized xy times the playable half-extent."""
    B, P = b["ply_mask"].shape
    ent = b["ent_xy"] * b["cells"][:, None]
    return torch.cat([ent.new_zeros(B, 1 + P, 2), ent], 1)


class RoPEEncoder(StateEncoder):
    """The state encoder's embedding (unit / owner / position / numbers / production / order
    tables) under RoPE self-attention. Returns the full token sequence; no grid or window
    readouts (this model predicts state, not slots)."""

    def __init__(self, vocab, d=256, depth=6, heads=8, mlp_ratio=4.0, ruleset=None):
        super().__init__(vocab, d, depth=1, heads=heads, mlp_ratio=mlp_ratio)
        del self.blocks, self.windows
        self.rope = RoPE2D(d // heads)
        self.layers = nn.ModuleList([Block(d, heads, self.rope, mlp_ratio) for _ in range(depth)])
        # types from the game's balance data (worldmodel/ruleset.py) instead of the learned tables
        self.ruleset = ruleset
        if ruleset is not None:
            del self.unit, self.ability, self.upgrade
            self._rs_sel = None  # (rulesets, upgrade names) of the current batch, set by embed()

    def _rs_batch(self, b, g):
        """The rulesets of the samples the group g belongs to: the frame's own for entities,
        the window's for action pointers (collate_actions carries them as act_ruleset)."""
        return b["ruleset"] if g == "ent" else b["act_ruleset"]

    def entity_tokens(self, b, g="ent"):
        if self.ruleset is None:
            return super().entity_tokens(b, g)
        rs = self._rs_batch(b, g)
        if b[f"{g}_rtype"].dim() == 1:  # flat pointers (selections): one row per unit, with its sample index
            rows = self.ruleset.lookup([rs[i] for i in b[f"{g}_sample"].tolist()] if f"{g}_sample" in b else rs,
                                       unit_api=b[f"{g}_rtype"][:, None], abil_api=torch.stack([b[f"{g}_rproduce"], b[f"{g}_rorder"]], 1))
            unit, abil = rows["unit"][:, 0], rows["ability"]
        else:
            rows = self.ruleset.lookup(rs, unit_api=b[f"{g}_rtype"], abil_api=torch.stack([b[f"{g}_rproduce"], b[f"{g}_rorder"]], -1))
            unit, abil = rows["unit"], rows["ability"]
        tok = unit + self.owner(b, g) + self.pos(b[f"{g}_xy"]) + self.ent_in(b[f"{g}_num"]) + abil[..., 0, :] + self.grp_in(b[f"{g}_groups"])
        if self.use_orders:
            tok = tok + abil[..., 1, :] + self.ord_in(b[f"{g}_ord"])
        return tok

    def snapshot_tokens(self, b):
        if self.ruleset is None:
            return super().snapshot_tokens(b)
        return self.ruleset.lookup(b["act_ruleset"], unit_api=b["act_rsnap"])["unit"]

    def embed(self, b):
        if self.ruleset is None:
            return super().embed(b)
        B = b["time"].shape[0]
        ent = self.entity_tokens(b)
        P = b["ply_mask"].shape[1]
        names = [b["ply_upnames"][i][p] if p < len(b["ply_upnames"][i]) else [] for i in range(B) for p in range(P)]
        ups = self.ruleset.lookup([b["ruleset"][i] for i in range(B) for _ in range(P)], upg_names=names)["upgrade"].view(B, P, -1)
        ply = (self.player_marker + self.race(b["ply_race"]) + self.owner(b, "ply") + self.ply_in(b["ply_num"]) + ups.to(ent.dtype))
        cls = self.cls + self.cls_in(torch.cat([self.time_ff(b["time"]), b["format"]], -1))[:, None]
        tokens = torch.cat([cls, ply, ent], 1)
        mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=ent.device), b["ply_mask"], b["ent_mask"]], 1)
        return tokens, mask

    def forward(self, b):
        x, mask = self.embed(b)
        ang = self.rope.angles(positions(b))
        for layer in self.layers:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(layer, x, mask, ang, use_reentrant=False)
            else:
                x = layer(x, mask, ang)
        x = self.norm(x)
        P = b["ply_mask"].shape[1]
        return dict(tokens=x, mask=mask, ang=ang, cls=x[:, 0], players=x[:, 1:1 + P], units=x[:, 1 + P:], P=P)


class DirectPredictor(nn.Module):
    """Encoded present (+ z added to the tokens by the caller) -> per-token state at t + k."""

    def __init__(self, d=256, depth=4, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.rope = RoPE2D(d // heads)
        self.time = RelTime()
        self.hor_in = mlp(self.time.out_dim(), d)
        self.null = nn.Parameter(torch.zeros(1, 1, d))  # always-present memory token: "no commands"
        self.layers = nn.ModuleList([Block(d, heads, self.rope, mlp_ratio, cross=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        nn.init.normal_(self.null, std=0.02)

    def forward(self, x, mask, ang, hor_s, mem, mem_mask):
        """x [B, T, d], mask [B, T], ang [B, T, dh/2], hor_s [B] seconds, mem [B, N, d] action
        tokens with mem_mask [B, N] (False everywhere when commands are withheld)."""
        B = x.shape[0]
        x = x + self.hor_in(self.time(hor_s))[:, None].to(x.dtype)
        mem = torch.cat([self.null.expand(B, -1, -1).to(mem.dtype), mem], 1)
        mem_mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), mem_mask], 1)
        for layer in self.layers:
            x = layer(x, mask, ang, mem, mem_mask)
        return self.norm(x)


# ------------------------------------------------------------------ two-hot targets

def two_hot(v, lo, hi, n):
    """Values [...] -> soft targets [..., n] over n bins spanning [lo, hi] (clamped)."""
    idx = ((v.float() - lo) / (hi - lo) * (n - 1)).clamp(0, n - 1)
    i0 = idx.floor().long().clamp(max=n - 2)
    w = (idx - i0.float())[..., None]
    out = torch.zeros(v.shape + (n,), device=v.device)
    out.scatter_(-1, i0[..., None], 1 - w)
    out.scatter_add_(-1, (i0 + 1)[..., None], w)
    return out


def centers(lo, hi, n, device):
    return torch.linspace(lo, hi, n, device=device)


def expect(logits, lo, hi):
    """Expected value of a two-hot distribution's logits [..., n]."""
    return F.softmax(logits.float(), -1) @ centers(lo, hi, logits.shape[-1], logits.device)


def soft_ce(logits, target):
    """Cross entropy against soft targets, per element [...]."""
    return -(target * F.log_softmax(logits.float(), -1)).sum(-1)


class Probes(nn.Module):
    """Per-entity readouts at t + k, as changes from t where the quantity is continuous (so
    "nothing happens" is the same target for every unit): position offsets in map units over
    [-pos_range, pos_range], hp / shield / energy / build-progress fraction changes over [-1, 1]."""

    POS_BINS, FRAC_BINS = 81, 65

    def __init__(self, d, n_abilities, pos_range=20.0):
        super().__init__()
        self.pos_range = pos_range
        sizes = dict(exit=N_EXIT, dx=self.POS_BINS, dy=self.POS_BINS, hp=self.FRAC_BINS, sh=self.FRAC_BINS,
                     en=self.FRAC_BINS, build=self.FRAC_BINS, order=n_abilities)
        self.heads = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n))
                                    for k, n in sizes.items()})

    FRACS = ("hp", "sh", "en", "build")

    def forward(self, h):
        return {k: head(h).float() for k, head in self.heads.items()}

    def losses(self, out, tgt, mask):
        """out: the heads' logits [B, N, *]; tgt: dict of targets [B, N] (exit class, dx, dy,
        fractions, order index); mask [B, N] real entities. Returns per-head mean losses, the
        attribute heads over entities present at t + k."""
        m = mask.float()
        present = m * (tgt["exit"] == EXIT_PRESENT).float()
        avg = lambda x, w: (x * w).sum() / w.sum().clamp(min=1)
        R = self.pos_range
        L = dict(exit=avg(F.cross_entropy(out["exit"].transpose(1, 2), tgt["exit"], reduction="none"), m),
                 dx=avg(soft_ce(out["dx"], two_hot(tgt["dx"], -R, R, self.POS_BINS)), present),
                 dy=avg(soft_ce(out["dy"], two_hot(tgt["dy"], -R, R, self.POS_BINS)), present),
                 order=avg(F.cross_entropy(out["order"].transpose(1, 2), tgt["order"], reduction="none"), present))
        for k in self.FRACS:
            L[k] = avg(soft_ce(out[k], two_hot(tgt[k], -1.0, 1.0, self.FRAC_BINS)), present)
        return L

    def point(self, out):
        """Point estimates from the heads: exit class, dx / dy / fractions as expectations,
        order as argmax."""
        R = self.pos_range
        est = dict(exit=out["exit"].argmax(-1), dx=expect(out["dx"], -R, R), dy=expect(out["dy"], -R, R),
                   order=out["order"].argmax(-1))
        for k in self.FRACS:
            est[k] = expect(out[k], -1.0, 1.0)
        return est


class PriorHead(nn.Module):
    """p(z | input) as a mixture of K diagonal Gaussians, in latent.py's dict format
    (logits [B, P, K], mu / lv [B, P, K, d_z]). The input is whatever the caller assembles per
    (sample, player): here the present's CLS and player slot, the player's command summary (or
    zeros) and a flag embedding saying whether commands were given."""

    def __init__(self, d_in, d_z, K=1, hidden=None):
        super().__init__()
        hidden = hidden or 2 * d_in
        self.K, self.d_z = K, d_z
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, K + 2 * K * d_z))
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.5)

    def forward(self, x):
        B, P, _ = x.shape
        o = self.net(x).float()
        logits, rest = o[..., :self.K], o[..., self.K:].view(B, P, self.K, 2, self.d_z)
        return dict(logits=logits, mu=rest[..., 0, :], lv=rest[..., 1, :].clamp(LV_MIN, 4.0))
