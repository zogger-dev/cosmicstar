"""Evaluation instruments for the world model: probes and a collapse alarm.

Probes are small MLPs that read ground-truth quantities (tokenize.probe_targets) off slot
vectors. They train on the side, on detached representations: no gradient reaches the
encoder or the predictor, so they measure what JEPA learned instead of teaching it.

They're trained on the target encoder's (layer-normed) slots of real frames, then applied to
three readings of the same future frame t + delta:

    now       the encoding of frame t        ("nothing changes", in real units)
    pred      the predictor's output for t + delta
    ceiling   the target encoder's encoding of t + delta (what the representation can hold)

and scored by the share of the now -> ceiling gap the prediction closes. A model that
learned nothing about dynamics scores ~0 %; a perfect forecaster scores 100 %.

The collapse alarm: the share of the representation's variance that comes from changes over
time within a game (vs. differences between games). A representation that keeps only what's
constant within a game (map, matchup) makes every future look like the present and the
JEPA loss trivially small; this share falling far below an untrained encoder's flags it.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .encoder import mlp
from .tokenize import PROBE_CLS, PROBE_GRID, PROBE_PLY, collate, probe_targets

KINDS = ("cls", "ply", "grid")
N_OUT = dict(cls=len(PROBE_CLS), ply=len(PROBE_PLY), grid=len(PROBE_GRID))


def readout(d, n_out, hidden, layers):
    """Linear (layers=1) or an MLP with `layers` linear layers of width `hidden`.
    hidden="legacy": the shape the first runs actually had, d -> n_out -> GELU -> n_out (a
    bottleneck as narrow as the output; kept only to load those checkpoints)."""
    if hidden == "legacy":
        return nn.Sequential(nn.Linear(d, n_out), nn.GELU(), nn.Linear(n_out, n_out))
    if layers == 1:
        return nn.Sequential(nn.Linear(d, n_out))
    mods = [nn.Linear(d, hidden), nn.GELU()]
    for _ in range(layers - 2):
        mods += [nn.Linear(hidden, hidden), nn.GELU()]
    return nn.Sequential(*mods, nn.Linear(hidden, n_out))


class Probes(nn.Module):
    def __init__(self, d=256, hidden=512, layers=2):
        super().__init__()
        self.config = dict(d=d, hidden=hidden, layers=layers)
        self.heads = nn.ModuleDict({k: readout(d, N_OUT[k], hidden, layers) for k in KINDS})

    def forward(self, slots, n_players):
        """slots [B, 1 + P + G, d] (slot order: CLS, players, grid) -> {kind: predictions}."""
        x = slots.detach().float()
        return dict(cls=self.heads["cls"](x[:, 0]), ply=self.heads["ply"](x[:, 1:1 + n_players]),
                    grid=self.heads["grid"](x[:, 1 + n_players:]))


def collate_targets(targets, n_players, dev):
    """List of probe_targets dicts -> {kind: tensor}, ply padded to n_players (+ mask)."""
    B = len(targets)
    ply = np.zeros((B, n_players, N_OUT["ply"]), np.float32)
    pm = np.zeros((B, n_players), bool)
    for i, t in enumerate(targets):
        ply[i, :len(t["ply"])] = t["ply"]
        pm[i, :len(t["ply"])] = True
    return dict(cls=torch.from_numpy(np.stack([t["cls"] for t in targets])).to(dev),
                grid=torch.from_numpy(np.stack([t["grid"] for t in targets])).to(dev),
                ply=torch.from_numpy(ply).to(dev), ply_mask=torch.from_numpy(pm).to(dev))


def probe_errors(out, tgt):
    """Mean squared error per kind (players masked)."""
    e = {k: F.mse_loss(out[k], tgt[k], reduction="none").mean(-1) for k in KINDS}
    m = tgt["ply_mask"].float()
    return dict(cls=e["cls"].mean(), grid=e["grid"].mean(), ply=(e["ply"] * m).sum() / m.sum().clamp(min=1))


def skill(now, pred, ceiling, min_room=0.05):
    """Probe errors -> (skill, best possible skill, share of the gap closed).

    skill      1 - pred / now: how much better than "nothing changes" the prediction reads
               (0 = no better than copying, < 0 = worse)
    best       1 - ceiling / now: the same for the true future encoding (the most there is to gain)
    closed     skill / best, only when there's room (best >= min_room); at very short horizons
               the future barely differs from the present and the ratio is meaningless (NaN)"""
    s = 1 - pred / now if now > 0 else float("nan")
    best = 1 - ceiling / now if now > 0 else float("nan")
    return s, best, (s / best if best >= min_room else float("nan"))


@torch.no_grad()
def retrieval_among(pred, cands, cand_mask, correct, present):
    """Retrieval when the true future may lie outside the window: pred [K, S, d] predictions,
    cands [C, S, d] every target frame of the sample (window frames and the extra frames
    beyond it), cand_mask [C, S], correct / present [K] candidate indices of each prediction's
    true future and of its present. Returns (top-1 accuracy, share that picked the present,
    share closer to the true future than to the present)."""
    p, t, m = pred.float(), cands.float(), cand_mask.float()
    dist = (p.pow(2).sum(-1)[:, None] + t.pow(2).sum(-1)[None] - 2 * torch.einsum("ksd,csd->kcs", p, t))  # [K, C, S]
    w = m[present][:, None] * m[None]
    dist = (dist * w).sum(-1) / w.sum(-1).clamp(min=1)                                  # [K, C]
    pick = dist.argmin(-1)
    k = torch.arange(len(correct), device=dist.device)
    beats = (dist[k, correct] < dist[k, present]).float()
    return (pick == correct).float().mean().item(), (pick == present).float().mean().item(), beats.mean().item()


@torch.no_grad()
def retrieval(pred, target, mask, h):
    """Question-free check: does position k's prediction of frame k + h pick out the true frame
    k + h among all frames of its window (same game, so map and matchup don't help)?

    pred, target [B, T, S, d] (target layer-normed), mask [B, T, S]. Distance = mean squared
    slot distance over slots valid in both. Returns (top-1 accuracy, share that picked frame k
    itself ("nothing changes"), share closer to the true future than to the present). Chance
    is 1 / T for top-1 and 50 % for the pairwise check; neighbouring frames are near-identical
    at short horizons, so the pairwise check is the fair one there."""
    B, T, S, d = target.shape
    p, t = pred[:, :T - h].float(), target.float()
    m = mask.float()
    dot = torch.einsum("bksd,bjsd->bkjs", p, t)
    dist = (p.pow(2).sum(-1)[:, :, None] + t.pow(2).sum(-1)[:, None] - 2 * dot)      # [B, K, T, S]
    w = m[:, :T - h, None] * m[:, None]
    dist = (dist * w).sum(-1) / w.sum(-1).clamp(min=1)                                  # [B, K, T]
    pick = dist.argmin(-1)                                                              # [B, K]
    k = torch.arange(T - h, device=pick.device)[None]
    kk = k.expand_as(pick)
    d_future = dist.gather(-1, (kk + h)[..., None])[..., 0]
    d_present = dist.gather(-1, kk[..., None])[..., 0]
    beats = (d_future < d_present).float().mean().item()  # closer to the true future than to the present
    return (pick == k + h).float().mean().item(), (pick == k).float().mean().item(), beats


@torch.no_grad()
def temporal_share(encoder, games, dev, n_times=24, max_games=8):
    """Share of the encoder's variance (layer-normed CLS and mean grid slot) that is within-game
    over time rather than between games. games: GameFrames."""
    feats = {"cls": [], "grid": []}
    for g in games[:max_games]:
        om = g.owner_map("observer")
        fs = np.linspace(len(g) * 0.1, len(g) * 0.9, n_times).astype(int)
        b = {k: x.to(dev) for k, x in collate([g.frame(int(f), om) for f in fs]).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = encoder(b)
        d = out["cls"].shape[-1]
        feats["cls"].append(F.layer_norm(out["cls"].float(), (d,)).cpu().numpy())
        feats["grid"].append(F.layer_norm(out["grid"].float(), (d,)).mean(1).cpu().numpy())
    share = {}
    for k, arrs in feats.items():
        X = np.stack(arrs)                        # [games, times, d]
        within, between = X.var(1).mean(), X.mean(1).var(0).mean()
        share[k] = float(within / (within + between))
    return share
