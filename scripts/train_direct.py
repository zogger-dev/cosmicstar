"""Direct state-space simulator (worldmodel/direct.py): the state at t, plus everyone's
commands in [t, t+k) or nothing, -> every entity's state at t+k, for k up to --max_loops game loops.

A batch is --frames present frames, each paired with --hor horizons k drawn uniformly from the
multiples of the extraction's frame step up to --max_loops (a step of 4 gives 4, 8, ..., 56; a
step of 1 gives every loop 1..56). Horizons, like every time the model reads, are game loops,
never frame numbers. The present is
encoded once per frame; each (frame, k) sample gets its own command window [t, t+k), with
--act_drop of them emptied (the model is told commands were withheld). Targets are the same
units' attributes at t+k, read from the extraction, no target encoder.

The latent z per (sample, player): the posterior (worldmodel/inverse_model.py) reads the
present and the future as the encoder sees them; the prior (direct.PriorHead) reads the
present and the player's command summary, or "withheld". KL(posterior || prior) above
--free_nats is charged with --beta. With commands given the future is determined and KL
should go to ~0; with them withheld z must carry what the players did. That difference,
printed per horizon in evaluation, is the test of the design. Evaluation also prints each
probe's error against copying the present (persistence), with z from the posterior (the
oracle), a prior sample and the prior's mean, and the births per window the model ignores.

Latent mode (--model latent, 2026-11-01; worldmodel/latent_dynamics.py, DESIGN_latent_dynamics.md):
the model predicts the encoder's own representation of the future, ŝ_{t+k} = s_t + h(s_t, k) +
g(·, actions, z), against targets from the encoder on the future frame (stop-gradient; --target
online, or a moving-average copy with --target ema). Losses: latent prediction, VICReg on s_t,
decorrelation of g from h and from s_t, a floor on g's spread where commands exist, Barlow Twins
between z and the command summaries, the KL. The probes decode ŝ_{t+k} on detached features
(evaluation only; --probe_weight lets a share of their gradient shape the latent), so the evaluation tables still read
positions, orders and deaths against persistence.

Views (2026-10-31): a game is its two player views and their combined view (tokenize.
CombinedFrames: both views joined, both players' commands; there is no observer extraction).
Each present frame is drawn from one of the three (--obs_share of them combined). In a player
view the posterior also reads what the viewer can't know about the enemy units they see: their
true orders and the enemy's commands in the window that selected them (inverse_model.EnemyIntent);
the prior never does, so the KL charges for the enemy intent the prior can't predict. KL is
charged for every player (the viewer's and the enemy's z). Horizons are drawn log-uniformly
over the multiples of the frame step up to --max_loops (224 loops = 10 s by default).

    python scripts/train_direct.py --steps 3000
    python scripts/train_direct.py --resume data/worldmodel/runs/<run> --eval_only
"""
import argparse
import collections
import json
import math
import os
import re
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# inductor's mixed-order reduction fusion fails to compile the backward of the per-cell LayerNorm
# + padding mask in the dense encoder's ResNet blocks (CantSplit, torch 2.10)
os.environ.setdefault("TORCHINDUCTOR_MIX_ORDER_REDUCTION", "0")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# the dense encoder's blocks see float32 and bfloat16 inputs (per-cell LayerNorm outputs float32 under autocast),
# with and without gradient (the moving-average target), at several shapes: past the default 8 variants
# dynamo silently ran the rest uncompiled
torch._dynamo.config.recompile_limit = 64
# an inductor bug (tiling_utils AssertionError) on rare shape variants: run that graph uncompiled, don't crash
torch._dynamo.config.suppress_errors = True

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cosmicstar.worldmodel.data import LPS, GamePool, Prefetch, to  # noqa: E402
from cosmicstar.worldmodel import corpus, statefile  # noqa: E402
from cosmicstar.worldmodel import vocab as vocab_mod  # noqa: E402
from cosmicstar.worldmodel import latent as lat  # noqa: E402
from cosmicstar.worldmodel import direct as dm  # noqa: E402
from cosmicstar.worldmodel.action_encoder import ActionEncoder, LoopTime  # noqa: E402
from cosmicstar.worldmodel.encoder import mlp  # noqa: E402
from cosmicstar.worldmodel.action_tokens import ActionVocab, GameActions, collate_actions, self_family  # noqa: E402
from cosmicstar.worldmodel.dense import DenseEncoder  # noqa: E402
from cosmicstar.worldmodel import latent_dynamics as ld  # noqa: E402
from cosmicstar.worldmodel import grid_latent as gl  # noqa: E402
from cosmicstar.worldmodel.pyramid import PyramidEncoder  # noqa: E402
from cosmicstar.worldmodel.inverse_model import C_INTENT_CMD, C_INTENT_ORD, EnemyIntent, InverseModel  # noqa: E402
from cosmicstar.worldmodel.ruleset import RulesetEncoder, RulesetTables  # noqa: E402
from cosmicstar.worldmodel.tokenize import C_ENTITY, CombinedFrames, GameFrames, collate  # noqa: E402
from cosmicstar.worldmodel.vocab import CAT_ARMY, CAT_STRUCTURE  # noqa: E402

TARGETS = ("exit", "dx", "dy", "hp", "sh", "en", "build", "order")
RF = torch.autograd.profiler.record_function


def actions_path(path, actions_dir=None):
    """A view's action table: <id>.acts.npz next to it in the games tree; for the legacy flat
    layout (<game>_p<slot>.npz in one folder) <game>.npz in actions_dir."""
    p = corpus.acts_path(path)
    if p is not None:
        return p
    return os.path.join(actions_dir or os.path.dirname(path), re.sub(r"_p\d+$", "", os.path.splitext(os.path.basename(path))[0]) + ".npz")


class Game:
    """A replay's frames (one player's view, or the combined view), its action tokens and
    per-frame deaths. enemy: the other player's view (player views of a 1v1), for the posterior."""

    def __init__(self, path, actions_dir, vocab, avocab, frames=None, dead=None):
        self.g = frames if frames is not None else GameFrames(path, vocab)
        self.ga = GameActions(actions_path(path, actions_dir), self.g, avocab)
        links = {int(k): v for k, v in avocab.ruleset_table(self.g.meta)["links"].items()}
        self.g.set_command_families(lambda link: self_family(avocab, links, link))  # players' last commands
        self.viewer = self.g.viewer  # 0 = combined view, else the player whose view this is (their commands only)
        if dead is None:
            with statefile.load(path) as z:
                dead = z["dead_id"], z["dead_frame_start"]
        self.dead_id, self.dead_start = dead
        self.slot_index = {s: i for i, s in enumerate(self.g.slots)}
        self.enemy, self.v2e = None, None


class GameSet:
    """One game: its player views (by slot) and, for a 1v1 with both views, the combined view."""

    def __init__(self, path, actions_dir, vocab, avocab, combined=True):
        m = corpus.VIEW_RE.match(os.path.basename(path))
        d = os.path.dirname(path)
        paths = sorted((q for q in os.listdir(d) if (mm := corpus.VIEW_RE.match(q)) and mm.group("id") == m.group("id")),
                       key=lambda q: int(corpus.VIEW_RE.match(q).group("slot"))) if m else [os.path.basename(path)]
        self.views = [Game(os.path.join(d, q), actions_dir, vocab, avocab) for q in paths]
        self.obs = None
        if len(self.views) == 2:
            a, b = self.views
            for me, en in ((a, b), (b, a)):  # viewer's unit id -> the same unit's id in the enemy's view (by tag)
                order = np.argsort(en.g.unit_tag)
                st = en.g.unit_tag[order]
                j = np.clip(np.searchsorted(st, me.g.unit_tag), 0, len(st) - 1)
                me.enemy, me.v2e = en, np.where(st[j] == me.g.unit_tag, order[j], -1).astype(np.int64)
            if combined:
                c = CombinedFrames(a.g, b.g)
                self.obs = Game(os.path.join(d, paths[0]), actions_dir, vocab, avocab, frames=c, dead=self._deaths(c))

    def _deaths(self, c):
        """Deaths of the combined view: both views' deaths, in combined ids, once each."""
        F = len(c.loops)
        fr, ids = [], []
        for i, g in enumerate(self.views):
            dfr = np.clip(np.searchsorted(g.dead_start, np.arange(len(g.dead_id)), side="right") - 1, 0, F - 1)
            cid = c.cid(i, g.dead_id)
            fr.append(dfr[cid >= 0])
            ids.append(cid[cid >= 0])
        key = np.unique((np.concatenate(fr).astype(np.int64) << 32) | np.concatenate(ids).astype(np.int64))
        dfr, did = key >> 32, (key & 0xFFFFFFFF).astype(np.int32)
        return did, np.concatenate([[0], np.cumsum(np.bincount(dfr, minlength=F))]).astype(np.int64)

    def pick(self, rng, obs_share, view=None):
        """A view for one present frame: view = "player" / "obs", or drawn (obs_share combined)."""
        if view == "obs" or (view is None and self.obs is not None and rng.rand() < obs_share):
            return self.obs if self.obs is not None else self.views[rng.randint(len(self.views))]
        return self.views[rng.randint(len(self.views))]


def enemy_info(game, t, lo, hi):
    """The posterior's privileged inputs in a player view (inverse_model.EnemyIntent): for each
    enemy unit visible at t (not a snapshot, not inside anything), its order queue at t and the
    enemy's commands in [lo, hi) that selected it, from the enemy's own view. Rows are the unit's
    row within frame t (its token)."""
    g, en = game.g, game.enemy
    ge = en.g
    u, a, b = g.u, g.start[t], g.start[t + 1]
    vis = (u["owner"][a:b] == en.viewer) & (u["display"][a:b] == 1) & (u["inside"][a:b] == 0)
    loc = np.nonzero(vis)[0]
    vid = u["id"][a + loc].astype(np.int64)
    eid = np.where(vid < len(game.v2e), game.v2e[np.clip(vid, 0, len(game.v2e) - 1)], -1)
    erow = ge.rows_of(t, eid)
    ok = erow >= 0
    loc, eid, erow = loc[ok], eid[ok], erow[ok]
    ri, ab, feat = ge.bags(erow)["ord"]
    out = dict(ord_row=loc[ri], ord_ab=ab.astype(np.int64), ord_feat=feat.astype(np.float32).reshape(-1, C_INTENT_ORD))
    c = ge.cmd
    rows = np.zeros(0, np.int64)
    cab, cfeat = np.zeros(0, np.int64), np.zeros((0, C_INTENT_CMD), np.float32)
    if c is not None and len(loc):
        i0, i1 = np.searchsorted(c["loop"], lo), np.searchsorted(c["loop"], hi)
        s0, s1 = c["start"][i0:i1], c["start"][i0 + 1:i1 + 1]
        ev = np.repeat(np.arange(i0, i1), s1 - s0)
        rec = c["unit"][s0[0]:s1[-1]].astype(np.int64) if i1 > i0 else np.zeros(0, np.int64)
        order = np.argsort(eid)
        se = eid[order]
        j = np.clip(np.searchsorted(se, rec), 0, max(len(se) - 1, 0))
        hit = (se[j] == rec) if len(se) else np.zeros(len(rec), bool)
        ev, li = ev[hit], order[j[hit]]
        rows = loc[li]
        ux, uy = u["x"][a + rows], u["y"][a + rows]
        tk = c["target"][ev].astype(np.int64)
        tx, ty = c["tx"][ev].astype(np.float32), c["ty"][ev].astype(np.float32)
        on_unit = tk == 2
        if on_unit.any():  # a unit target: where that unit is at t in the enemy's view
            tr = ge.rows_of(t, c["target_id"][ev[on_unit]])
            tx[on_unit] = np.where(tr >= 0, ge.u["x"][np.maximum(tr, 0)], np.nan)
            ty[on_unit] = np.where(tr >= 0, ge.u["y"][np.maximum(tr, 0)], np.nan)
        hasp = np.isfinite(tx) & np.isfinite(ty)
        dt = (c["loop"][ev] - lo).astype(np.float32)
        cfeat = np.stack([tk == 0, tk == 1, on_unit,
                          np.where(hasp, np.clip((np.nan_to_num(tx) - ux) / g.half[0], -2, 2), 0),
                          np.where(hasp, np.clip((np.nan_to_num(ty) - uy) / g.half[1], -2, 2), 0),
                          c["queued"][ev], dt / 224.0, np.log1p(dt) / 6.0], 1).astype(np.float32).reshape(-1, C_INTENT_CMD)
        cab = g.vocab.ability_index(c["abil"][ev]).astype(np.int64)
    out.update(cmd_row=rows.astype(np.int64), cmd_ab=cab, cmd_feat=cfeat)
    return out


TIGHT_GAP = 1.0        # map units: a passage counts as tight within this much of a wall or a structure's edge
SMALL_R, LARGE_R = 0.5, 0.75  # unit radius: small (zergling, marine, worker) / large (tank, archon, ultralisk)


def tight_units(g, t):
    """Fine-scale passage (evaluation): ground units at frame t whose body is within TIGHT_GAP of
    unpathable terrain or of a structure's edge, where a fraction of a cell decides who fits.
    Returns a bool per row of frame t."""
    u, a, b = g.u, g.start[t], g.start[t + 1]
    x, y = u["x"][a:b].astype(np.float32), u["y"][a:b].astype(np.float32)
    r = u["radius"][a:b].astype(np.float32) if "radius" in u else np.full(b - a, 0.5, np.float32)
    ground = (u["flags"][a:b] & 1) == 0
    unit = (u["cat"][a:b] == CAT_ARMY) & ground & (u["inside"][a:b] == 0)  # army: workers hug their base all game
    out = np.zeros(b - a, bool)
    idx = np.nonzero(unit)[0]
    if not len(idx):
        return out
    # terrain: distance from the unit's center to the nearest unpathable cell (a distance map per game)
    if getattr(g, "_wall_dist", None) is None:
        from scipy.ndimage import distance_transform_edt
        g._wall_dist = distance_transform_edt(g.static[0] > 0).astype(np.float32)
    Hm, Wm = g._wall_dist.shape
    wd = g._wall_dist[np.clip(y[idx].astype(np.int64), 0, Hm - 1), np.clip(x[idx].astype(np.int64), 0, Wm - 1)]
    out[idx] = wd - r[idx] < TIGHT_GAP + 0.5   # + half a cell: the distance runs from cell centers
    # structures: the gap between the unit's body and the structure's edge
    st = np.nonzero(u["cat"][a:b] == CAT_STRUCTURE)[0]
    if len(st):
        gap = np.hypot(x[idx, None] - x[None, st], y[idx, None] - y[None, st]) - r[idx, None] - r[None, st]
        out[idx] |= (gap < TIGHT_GAP).any(1)
    return out


def make_sample(game, t, k, rng, with_actions, om, frame):
    """The frame at t (built once per present frame and shared by its horizons), the frame k
    frames later (for the posterior), the commands in between (or none) and per-entity targets
    there for the entities of t, in their row order. The sample's horizon is recorded in game
    loops (the loop difference between the two frames)."""
    g, u = game.g, game.g.u
    a, b = g.start[t], g.start[t + 1]
    ids = u["id"][a:b]
    rows = g.rows_of(t + k, ids)
    present = rows >= 0
    dead = game.dead_id[game.dead_start[t + 1]:game.dead_start[t + k + 1]]
    died = np.isin(ids, dead) & ~present
    r = np.where(present, rows, a)  # a valid row anywhere for absent units; masked in the loss
    frac = lambda x, mx: np.where(mx > 0, x / np.maximum(mx, 1e-6), 0).astype(np.float32)
    # positions and fractions as changes from t (persistence = 0); the order as the ability at t+k
    tgt = dict(exit=np.where(present, dm.EXIT_PRESENT, np.where(died, dm.EXIT_DIED, dm.EXIT_GONE)).astype(np.int64),
               dx=(u["x"][r] - u["x"][a:b]).astype(np.float32), dy=(u["y"][r] - u["y"][a:b]).astype(np.float32),
               hp=frac(u["hp"][r], u["hp_max"][r]) - frac(u["hp"][a:b], u["hp_max"][a:b]),
               sh=frac(u["shield"][r], u["shield_max"][r]) - frac(u["shield"][a:b], u["shield_max"][a:b]),
               en=((u["energy"][r] - u["energy"][a:b]) / 200.0).astype(np.float32),
               build=(u["build"][r].astype(np.float32) - u["build"][a:b].astype(np.float32)),
               order=u["order"][r].astype(np.int64))
    now = {k: np.zeros(len(ids), np.float32) for k in ("dx", "dy", "hp", "sh", "en", "build")}
    now["order"] = u["order"][a:b].astype(np.int64)
    lo, hi = int(g.loops[t]), int(g.loops[t + k])
    # the full window always: withholding is a mask (has_act) on what g and the prior read; the inverse's targets
    # (the commands that produced the change) exist either way
    act = game.ga.window(lo, hi, np.array([lo]), lo, om, static_pointers=True, only=game.viewer or None)
    if game.enemy is not None:  # player views: the enemy's commands, from their own view (a training target only)
        act_en = game.enemy.ga.window(lo, hi, np.array([lo]), lo, om, static_pointers=True, only=game.enemy.viewer)
        en_pidx = game.slot_index.get(game.enemy.viewer, -1)
    else:
        act_en = game.ga.window(lo, lo, np.array([lo]), lo, om, static_pointers=True, only=game.viewer or None)
        en_pidx = -1
    sum_pidx = [game.slot_index[s] for s in ([game.viewer] if game.viewer else g.slots)]  # summary -> player index
    pidx = np.array([game.slot_index.get(int(o), -1) for o in u["owner"][a:b]], np.int64)
    touched = np.isin(ids, act["touched_uid"][act["touched_uid"] >= 0])  # selected or targeted by a command in the window
    births = int(g.start[t + k + 1] - g.start[t + k]) - int(present.sum())
    priv = enemy_info(game, t, lo, hi) if game.enemy is not None else None
    enemy = (u["owner"][a:b] == game.enemy.viewer) if game.enemy is not None else np.zeros(len(ids), bool)
    rr = u["radius"][a:b].astype(np.float32) if "radius" in u else np.full(len(ids), 0.5, np.float32)
    tight = tight_units(g, t)
    fut_row = np.where(present, rows - g.start[t + k], -1).astype(np.int64)  # the unit's row in frame t+k (-1: gone)
    return dict(frame=frame, future=g.frame(t + k, om), act=act, act_en=act_en, en_pidx=en_pidx, tgt=tgt, now=now, pidx=pidx, priv=priv, enemy=enemy, fut_row=fut_row,
                tight_sm=tight & (rr < SMALL_R), tight_lg=tight & (rr >= LARGE_R),
                has_act=with_actions, hor_loops=hi - lo, births=births, touched=touched, sum_pidx=sum_pidx, _game=game, _lo=lo, _om=om)


def make_seq(game, t0, d, n, rng, with_actions, om, frame0, base=False):
    """A rollout sequence (--seq_len > 0): frames t0, t0 + d, ..., t0 + n d (d frames apart), each
    step's commands (the full window; has_act masks what g and the prior read), the enemy's commands
    (player views; inverse targets) and the posterior's enemy intent per step. base: also the
    make_sample of (t0, n d), for evaluation's metrics at the end of the rollout."""
    g = game.g
    frames = [frame0] + [g.frame(t0 + i * d, om) for i in range(1, n + 1)]
    acts, acts_en, privs, hors = [], [], [], []
    for i in range(n):
        lo, hi = int(g.loops[t0 + i * d]), int(g.loops[t0 + (i + 1) * d])
        acts.append(game.ga.window(lo, hi, np.array([lo]), lo, om, static_pointers=True, only=game.viewer or None))
        if game.enemy is not None:
            acts_en.append(game.enemy.ga.window(lo, hi, np.array([lo]), lo, om, static_pointers=True, only=game.enemy.viewer))
        else:
            acts_en.append(game.ga.window(lo, lo, np.array([lo]), lo, om, static_pointers=True, only=game.viewer or None))
        privs.append(enemy_info(game, t0 + i * d, lo, hi) if game.enemy is not None else None)
        hors.append(hi - lo)
    out = dict(frames=frames, acts=acts, acts_en=acts_en, privs=privs, hors=hors, has_act=with_actions,
               en_pidx=game.slot_index.get(game.enemy.viewer, -1) if game.enemy is not None else -1,
               sum_pidx=[game.slot_index[s_] for s_ in ([game.viewer] if game.viewer else g.slots)])
    if base:
        out["base"] = make_sample(game, t0, n * d, rng, with_actions, om, frame0)
    return out


TARGET_KEYS = ("ent_pos", "ent_mask", "half", "ent_pcode", "ply_pcode", "ent_type", "ent_num", "ent_order")


def collate_seq(seqs, dev, has_act=None):
    """S sequences of n steps: all S (n + 1) frames in one frame batch (each encoded once: a step's
    future is the next step's present); per step (S n of them, sequence-major) the commands, the
    enemy's commands, the horizon, the enemy intent. fi / fut: each step's present / future frame.
    tb holds the decoder's target fields of the futures; tb_last those of each sequence's last frame."""
    S, n = len(seqs), len(seqs[0]["acts"])
    fb = to(collate([f for q in seqs for f in q["frames"]]), dev)
    N = fb["ent_mask"].shape[1]
    pres = [si * (n + 1) + i for si in range(S) for i in range(n)]
    futs = [x + 1 for x in pres]
    last = [si * (n + 1) + n for si in range(S)]
    ha = [bool(q["has_act"]) if has_act is None else has_act for q in seqs for _ in range(n)]
    b = dict(fb=fb, fi=torch.tensor(pres, device=dev), fut=torch.tensor(futs, device=dev), S=S, n=n,
             tb={k: fb[k][torch.tensor(futs, device=dev)] for k in TARGET_KEYS},
             tb_last={k: fb[k][torch.tensor(last, device=dev)] for k in TARGET_KEYS},
             ab=to(collate_actions([a for q in seqs for a in q["acts"]]), dev),
             ab_en=to(collate_actions([a for q in seqs for a in q["acts_en"]]), dev),
             en_pidx=torch.tensor([q["en_pidx"] for q in seqs for _ in range(n)], device=dev),
             hor=torch.tensor([h for q in seqs for h in q["hors"]], dtype=torch.float32, device=dev),
             has_act=torch.tensor(ha, device=dev),
             sum_pidx=(pad([np.array(q["sum_pidx"]) + 1 for q in seqs for _ in range(n)],
                           max(len(q["sum_pidx"]) for q in seqs), np.int64) - 1).to(dev))
    priv = {}
    steps = [(si * n + i, q["privs"][i]) for si, q in enumerate(seqs) for i in range(n) if q["privs"][i] is not None]
    for k, c in (("ord", C_INTENT_ORD), ("cmd", C_INTENT_CMD)):  # flat: step * N + the unit's row in the step's present
        priv[f"{k}_idx"] = torch.from_numpy(np.concatenate([p[f"{k}_row"] + j * N for j, p in steps] or [np.zeros(0, np.int64)])).to(dev)
        priv[f"{k}_ab"] = torch.from_numpy(np.concatenate([p[f"{k}_ab"] for _, p in steps] or [np.zeros(0, np.int64)])).to(dev)
        priv[f"{k}_feat"] = torch.from_numpy(np.concatenate([p[f"{k}_feat"] for _, p in steps] or [np.zeros((0, c), np.float32)])).to(dev)
    b["priv"] = priv
    return b


def sel_enc(o, idx):
    """An encoder output (nested dicts / lists of [F, ...] tensors) at frames idx."""
    if torch.is_tensor(o):
        return o[idx] if o.dim() > 0 else o
    if isinstance(o, dict):
        return {k: sel_enc(v, idx) for k, v in o.items()}
    if isinstance(o, list):
        return [sel_enc(v, idx) for v in o]
    return o


def withhold(samples):
    """The same samples with their commands withheld (an empty window)."""
    out = []
    for s in samples:
        game, lo, om = s["_game"], s["_lo"], s["_om"]
        out.append(dict(s, act=game.ga.window(lo, lo, np.array([lo]), lo, om, static_pointers=True, only=game.viewer or None), has_act=False))
    return out


def pad(arrs, n, dtype=None):
    buf = np.zeros((len(arrs), n), dtype or arrs[0].dtype)
    for i, a in enumerate(arrs):
        buf[i, :len(a)] = a
    return torch.from_numpy(buf)


def collate_samples(samples, dev):
    """frames: the unique present frames (one per --frames), fi: sample -> frame index."""
    frames, fi = [], []
    for s in samples:
        if not frames or frames[-1] is not s["frame"]:
            frames.append(s["frame"])
        fi.append(len(frames) - 1)
    fb = to(collate(frames), dev)
    tb = to(collate([s["future"] for s in samples]), dev)
    ab = to(collate_actions([s["act"] for s in samples]), dev)
    ab_en = to(collate_actions([s["act_en"] for s in samples]), dev)
    N = fb["ent_mask"].shape[1]
    b = dict(fb=fb, tb=tb, ab=ab, ab_en=ab_en, en_pidx=torch.tensor([s["en_pidx"] for s in samples], device=dev),
             fi=torch.tensor(fi, device=dev),
             hor=torch.tensor([s["hor_loops"] for s in samples], dtype=torch.float32, device=dev),  # game loops
             has_act=torch.tensor([s["has_act"] for s in samples], device=dev),
             pidx=pad([s["pidx"] for s in samples], N).to(dev),
             touched=pad([s["touched"] for s in samples], N).to(dev),
             enemy=pad([s["enemy"] for s in samples], N).to(dev),
             tight_sm=pad([s["tight_sm"] for s in samples], N).to(dev),
             tight_lg=pad([s["tight_lg"] for s in samples], N).to(dev),
             fut_row=(pad([s["fut_row"] + 1 for s in samples], N) - 1).to(dev),
             sum_pidx=(pad([np.array(s["sum_pidx"]) + 1 for s in samples], max(len(s["sum_pidx"]) for s in samples), np.int64) - 1).to(dev),
             births=float(np.mean([s["births"] for s in samples])))
    priv = {}
    for k, c in (("ord", C_INTENT_ORD), ("cmd", C_INTENT_CMD)):  # flat over the batch: sample * N + unit row
        ps = [(i, s["priv"]) for i, s in enumerate(samples) if s["priv"] is not None]
        priv[f"{k}_idx"] = torch.from_numpy(np.concatenate([p[f"{k}_row"] + i * N for i, p in ps] or [np.zeros(0, np.int64)])).to(dev)
        priv[f"{k}_ab"] = torch.from_numpy(np.concatenate([p[f"{k}_ab"] for _, p in ps] or [np.zeros(0, np.int64)])).to(dev)
        priv[f"{k}_feat"] = torch.from_numpy(np.concatenate([p[f"{k}_feat"] for _, p in ps] or [np.zeros((0, c), np.float32)])).to(dev)
    b["priv"] = priv
    b["tgt"] = {k: pad([s["tgt"][k] for s in samples], N).to(dev) for k in TARGETS}
    b["now"] = {k: pad([s["now"][k] for s in samples], N).to(dev) for k in TARGETS if k != "exit"}
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=corpus.GAMES, help="the games tree (every player view of a split's game ids), or a "
                    "legacy flat folder of <game>_p<slot>.npz files named in the split")
    ap.add_argument("--actions_dir", default=None, help="legacy flat layout only: the folder of <game>.npz action tables")
    ap.add_argument("--splits", default=os.path.join(corpus.GAMES, "splits", "default.json"))
    ap.add_argument("--max_loops", type=int, default=224, help="longest horizon in game loops (224 = 10 s)")
    ap.add_argument("--hor_dist", default="log", choices=("log", "uniform"),
                    help="how horizons are drawn: log-uniform (short ones as often per doubling as long ones) or uniform")
    ap.add_argument("--eval_loops", default="4,16,56,112,224", help="horizons evaluated, in game loops (multiples of the frame step)")
    ap.add_argument("--obs_share", type=float, default=1 / 3, help="share of present frames from the combined view "
                    "(the rest from the player views)")
    ap.add_argument("--frames", type=int, default=16, help="present frames per batch")
    ap.add_argument("--hor", type=int, default=2, help="horizons per present frame")
    ap.add_argument("--act_drop", type=float, default=0.5, help="share of samples with commands withheld")
    ap.add_argument("--types", default="table", help="unit / ability / upgrade embeddings: table (learned per id, the default; "
                    "decided 2026-10-30: learn them from the replays first, use the game data for probes and distillation later) "
                    "or ruleset (computed from the balance / game data, worldmodel/ruleset.py)")
    ap.add_argument("--rules_tables", default="data/worldmodel/rulesets.json")
    ap.add_argument("--rs_depth", type=int, default=3, help="attention layers over the ruleset's nodes")
    ap.add_argument("--no_type_residual", action="store_true", help="no learned per-name residual on the ruleset embeddings")
    ap.add_argument("--encoder", default="dense", choices=("dense", "pyramid", "flat"),
                    help="dense: dense ResNet convs + 4x4-window joint attention with all entities at 1-16 map units per cell "
                    "(worldmodel/dense.py); pyramid: units + map grid, 3x3 local joint attention at 1/2/4 map units per cell, full joint "
                    "attention at 8/16 (worldmodel/pyramid.py); flat: units only, full attention (the October pilot)")
    ap.add_argument("--widths", default="128,128,128", help="pyramid: map cell dimensions of the local levels (256, 128, 64 "
                    "grids); equal to --d means units and cells attend at full width with no projection")
    ap.add_argument("--conv_blocks", type=int, default=3, help="pyramid: conv blocks on the map at every level")
    ap.add_argument("--attn_blocks", type=int, default=3, help="pyramid: attention blocks at every level")
    ap.add_argument("--d", type=int, default=128, help="token width everywhere (units, map cells, players, actions, predictor)")
    ap.add_argument("--head_dim", type=int, default=32, help="attention head width; heads = d / head_dim (4 at d 128) in the "
                    "encoder's local and full attention and the predictor")
    ap.add_argument("--enc_depth", type=int, default=6)
    ap.add_argument("--pred_depth", type=int, default=4)
    ap.add_argument("--d_z", type=int, default=32, help="Gaussian z (--z gauss): its dimension")
    ap.add_argument("--z", default="categorical", choices=("categorical", "gauss"),
                    help="z per (sample, player): categorical = --z_groups one-hot variables of --z_classes classes each "
                    "(DreamerV3: straight-through samples, unimix, balanced KL with free bits); gauss = the earlier diagonal Gaussian")
    ap.add_argument("--z_groups", type=int, default=32)
    ap.add_argument("--z_classes", type=int, default=32)
    ap.add_argument("--unimix", type=float, default=0.01, help="categorical z: share of uniform mixed into both distributions")
    ap.add_argument("--kl_dyn", type=float, default=0.5, help="categorical z: weight of KL(sg(post) || prior), trains the prior")
    ap.add_argument("--kl_rep", type=float, default=0.1, help="categorical z: weight of KL(post || sg(prior)), regularizes the posterior")
    ap.add_argument("--es_k", type=int, default=0, help="latent + categorical z: K prior samples scored with the energy score "
                    "(a proper scoring rule: pulls samples to the truth, rewards their spread; averaging futures is penalized)")
    ap.add_argument("--w_es", type=float, default=1.0)
    ap.add_argument("--free_bits", type=float, default=1.0, help="categorical z: nats free on each KL side, per (sample, player)")
    ap.add_argument("--prior_k", type=int, default=1, help="mixture components of the prior")
    ap.add_argument("--beta", type=float, default=0.05, help="weight per nat of KL above --free_nats")
    ap.add_argument("--beta_warmup", type=int, default=300)
    ap.add_argument("--free_nats", type=float, default=0.5, help="KL per (sample, player) below this is free")
    ap.add_argument("--steps", type=int, default=3000, help="stop after this many steps (a resumed run can raise it)")
    ap.add_argument("--sched_steps", type=int, default=0, help="length of the cosine learning-rate schedule (default "
                    "--steps): longer than --steps keeps the rate up for a run meant to be resumed")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pool", type=int, default=12, help="games in memory (each: both player views and the combined view)")
    ap.add_argument("--rotate_every", type=int, default=50)
    ap.add_argument("--rotate", type=int, default=2)
    ap.add_argument("--val_games", type=int, default=2)
    ap.add_argument("--val_states", type=int, default=48, help="states per (set, horizon) in evaluation")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--val_every", type=int, default=50, help="train vs validation loss and KL (held-out batches) every this many steps")
    ap.add_argument("--val_batches", type=int, default=2, help="fixed held-out batches for --val_every")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--max_games", type=int, default=0)
    ap.add_argument("--no_ckpt", action="store_true", help="no activation checkpointing in the encoder")
    ap.add_argument("--model", default="direct", choices=("direct", "latent", "grid"),
                    help="direct: the predictor outputs the future values; latent: residual dynamics in the encoder's space")
    ap.add_argument("--grid", type=int, default=16, help="grid: the latent's cells per side (every map)")
    ap.add_argument("--pool_level", type=int, default=1, help="grid: the dense encoder's map level scattered onto the latent "
                    "(0 = 1 map unit per cell, 1 = 2, ...)")
    ap.add_argument("--mix_depth", type=int, default=4, help="grid: transformer blocks over [globals, latent cells]")
    ap.add_argument("--dec_depth", type=int, default=2, help="grid: entity decoder blocks")
    ap.add_argument("--dec_grad", type=float, default=0.1, help="grid: share of the entity decoder's gradient reaching the latent")
    ap.add_argument("--w_ent", type=float, default=1.0, help="grid: entity decoder losses (counts, offsets, numbers, orders)")
    ap.add_argument("--w_gram", type=float, default=0.1, help="grid: second-moment losses (latent cells, decoded entities, "
                    "entities x cells), predicted vs target")
    ap.add_argument("--w_halone", type=float, default=1.0, help="grid: latent loss of s + h alone on samples with commands withheld")
    ap.add_argument("--seq_len", type=int, default=0, help="grid: rollout training: sequences of this many steps (0: the "
                    "direct multi-horizon samples)")
    ap.add_argument("--seqs", type=int, default=5, help="grid, --seq_len: sequences per batch")
    ap.add_argument("--deltas", default="1,2,3,4", help="grid, --seq_len: step sizes in frames, one drawn per sequence")
    ap.add_argument("--w_ol", type=float, default=1.0, help="grid, --seq_len: open-loop rollout latent loss (fed its own predictions)")
    ap.add_argument("--ol_ramp", type=int, default=2000, help="grid, --seq_len: steps over which the open-loop weight ramps up")
    ap.add_argument("--seq_eval_states", type=int, default=16, help="grid, --seq_len: sequences per (set, horizon) in evaluation")
    ap.add_argument("--g_self", action="store_true", help="latent: g's blocks also self-attend over the state tokens "
                    "(an action can move things between cells)")
    ap.add_argument("--target", default="online", choices=("online", "ema"), help="latent: the future encoder (stop-gradient)")
    ap.add_argument("--ema_tau", type=float, default=0.996)
    ap.add_argument("--h_depth", type=int, default=4, help="latent: passive dynamics depth")
    ap.add_argument("--g_depth", type=int, default=2, help="latent: action dynamics depth")
    ap.add_argument("--w_lat", type=float, default=1.0)
    ap.add_argument("--w_var", type=float, default=1.0)
    ap.add_argument("--w_cov", type=float, default=0.04)
    ap.add_argument("--w_dec", type=float, default=1.0)
    ap.add_argument("--w_gvar", type=float, default=1.0)
    ap.add_argument("--g_floor", type=float, default=0.05, help="latent: g's minimum spread per dimension where commands exist")
    ap.add_argument("--w_inv", type=float, default=0.05, help="latent: Barlow Twins between z and the command summaries")
    ap.add_argument("--inv_pair", default="barlow", choices=("barlow", "vicreg"), help="latent: how z and the command summaries "
                    "are matched: Barlow Twins (batch cross-correlation; needs many pairs) or VICReg (per-pair MSE + variance and "
                    "covariance on each side)")
    ap.add_argument("--inv_queue", type=int, default=8, help="latent, --inv_pair vicreg: batches of past projections in the "
                    "variance / covariance statistics")
    ap.add_argument("--inv_dim", type=int, default=64, help="latent: projection width for the z / command pairing")
    ap.add_argument("--inv_off", type=float, default=0.02, help="latent: Barlow Twins off-diagonal weight")
    ap.add_argument("--w_dinv", type=float, default=0.0, help="latent: the per-unit inverse head on Δ = s_{t+k} − s_t "
                    "(moved / touched / order changed / new order), with gradient through the future encoding (>0 makes "
                    "the futures' encoding train: the cost of the direct model)")
    ap.add_argument("--w_dbar", type=float, default=0.05, help="latent: weight of the latent-change / raw-change Barlow Twins "
                    "inside the Δ inverse term (it sums over 64 dimensions, ~60 at the start)")
    ap.add_argument("--dbar_off", type=float, default=0.02, help="latent: Barlow Twins off-diagonal weight, latent change vs raw change")
    ap.add_argument("--probe_weight", type=float, default=0.0, help="latent: share of the probes' gradient that reaches the "
                    "latent (0: detached, evaluation only; a small value keeps the latent from dropping what the probes read, "
                    "e.g. motion; the probes themselves always train at full strength)")
    ap.add_argument("--profile", type=int, default=0, help="profile the step this many steps in (prints a table)")
    ap.add_argument("--no_compile", action="store_true", help="no torch.compile (by default the encoder's blocks, the "
                    "predictor, posterior, prior, probes and the enemy-intent pooling are compiled: fused kernels; the "
                    "first steps of each new shape compile)")
    ap.add_argument("--out", default="data/worldmodel/runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--refit_probes", type=int, default=0, help="with --resume --eval_only: re-initialize the probes and train "
                    "only them for this many steps on the frozen model before evaluating (removes the lag of probes trained "
                    "online against a changing representation)")
    ap.add_argument("--eval_sets", default="", help="comma-separated evaluation sets to run (default: all), e.g. unseen-maps")
    args = ap.parse_args()

    ck = None
    if args.resume:
        if os.path.isdir(args.resume):
            args.resume = os.path.join(args.resume, "model_latest.pt")
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        # the checkpoint's settings, except where the command line says otherwise for these
        given = {a.dest for a in ap._actions if any(o in sys.argv for o in a.option_strings)}
        over = {k: getattr(args, k) for k in ("steps", "eval_every", "save_every", "val_states", "refit_probes", "eval_sets")
                if k in given}
        args = argparse.Namespace(**{**vars(ap.parse_args([])), **ck["args"], "resume": args.resume, "eval_only": args.eval_only, **over})
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    dev = "cuda"
    split = json.load(open(args.splits))

    views = corpus.views_by_id(args.games)  # games tree: id -> its extracted player views

    def available(names):
        """Every extracted view of the split's games (ids in the tree; legacy: file names)."""
        paths = []
        for n in sorted(names):
            paths += views.get(n, []) if n in views else [os.path.join(args.games, n)]
        return [p for p in paths if os.path.exists(p) and os.path.exists(actions_path(p, args.actions_dir))]

    train_paths = available(split["train"])
    tables = RulesetTables(args.rules_tables) if args.types == "ruleset" else None
    if tables is not None:  # only games whose ruleset the balance data covers
        def covered(paths):
            keep = []
            for q in paths:
                if not os.path.exists(q):
                    continue
                with statefile.load(q) as z:
                    if json.loads(str(z["meta"])).get("ruleset") in tables.rulesets:
                        keep.append(q)
            return keep
        n0 = len(train_paths)
        train_paths = covered(train_paths)
        for k in ("train", "held_out_games", "held_out_map_games"):  # the game lists; held_out_maps are map names
            split[k] = sorted({corpus.VIEW_RE.match(os.path.basename(q)).group("id") if corpus.VIEW_RE.match(os.path.basename(q))
                               else os.path.basename(q) for q in covered(available(split.get(k, [])))})
        print(f"ruleset tables: {len(tables.rulesets)} rulesets; {len(train_paths)} of {n0} train games covered", flush=True)
    def one_per_game(paths):  # a game set loads its sibling views
        seen, out = set(), []
        for q in paths:
            m = corpus.VIEW_RE.match(os.path.basename(q))
            key = m.group("id") if m else q
            if key not in seen:
                seen.add(key)
                out.append(q)
        return out
    train_paths = one_per_game(train_paths)
    if args.max_games:
        train_paths = train_paths[:args.max_games]
    eval_paths = {"seen-games": train_paths[:args.val_games],
                  "unseen-games": one_per_game(available(split["held_out_games"]))[:args.val_games],
                  "unseen-maps": one_per_game(available(split["held_out_map_games"]))[:args.val_games]}
    run = os.path.dirname(os.path.abspath(args.resume)) if args.resume else \
        os.path.join(args.out, time.strftime("direct_%Y%m%d_%H%M%S"))
    voc = json.load(open(os.path.join(run, "vocab.json"))) if args.resume else vocab_mod.build(train_paths[:60])
    v = vocab_mod.Vocab(voc)
    av = ActionVocab()
    load = lambda p: GameSet(p, args.actions_dir, v, av)
    t0 = time.time()
    eval_games = {name: [load(p) for p in ps] for name, ps in eval_paths.items() if ps}
    pool = GamePool(train_paths, min(args.pool, len(train_paths)), load, rng, background=0 if args.eval_only else args.rotate,
                    names=ck.get("pool") if ck is not None else None)
    loaded = [x for _, gs in pool.games for x in gs.views] + [x for gss in eval_games.values() for gs in gss for x in gs.views]
    step_loops = collections.Counter(x.g.step for x in loaded).most_common(1)[0][0]
    assert all(x.g.step == step_loops for x in loaded), "mixed frame rates"
    # horizons in game loops: the multiples of the frame step up to --max_loops (frames are step_loops apart)
    hor_loops = list(range(step_loops, args.max_loops + 1, step_loops))
    hor_p = 1.0 / np.array(hor_loops, np.float64) if args.hor_dist == "log" else np.ones(len(hor_loops))
    hor_p /= hor_p.sum()
    assert hor_loops, f"--max_loops {args.max_loops} is shorter than the frame step ({step_loops} loops)"
    k_max = hor_loops[-1] // step_loops  # in frames, for bounds only
    eval_k = [int(x) for x in args.eval_loops.split(",")]
    deltas = [int(x) for x in args.deltas.split(",")]
    if args.seq_len:
        assert args.model == "grid", "rollout training (--seq_len) is built for the grid model"
        k_max = max(k_max, max(deltas) * args.seq_len)
    bad = [k for k in eval_k if k % step_loops or not step_loops <= k <= args.max_loops]
    assert not bad, f"--eval_loops {bad}: not multiples of the frame step ({step_loops} loops) within --max_loops"
    print(f"loaded {len(pool.games) + sum(len(x) for x in eval_games.values())} games ({len(loaded)} player views) in {time.time() - t0:.0f}s", flush=True)

    os.makedirs(run, exist_ok=True)
    if not args.resume:
        json.dump(dict(vars(args), step_loops=step_loops, hor_loops=hor_loops, n_train=len(train_paths),
                       eval={k: [os.path.basename(p) for p in ps] for k, ps in eval_paths.items()}),
                  open(os.path.join(run, "config.json"), "w"), indent=1)
        json.dump(voc, open(os.path.join(run, "vocab.json"), "w"))
    print(f"{len(train_paths)} train games (pool {len(pool.games)}) | eval: "
          + ", ".join(f"{k} {len(v_)} games" for k, v_ in eval_games.items())
          + f" | frame step {step_loops} loops, horizons {hor_loops[0]}..{hor_loops[-1]} loops by {step_loops} (eval {eval_k})"
          f" | batch {args.frames} frames x {args.hor} horizons"
          f" | commands withheld on {args.act_drop:.0%} | z: d_z {args.d_z}, prior x{args.prior_k}, beta {args.beta}"
          f" (warmup {args.beta_warmup}), free {args.free_nats} nats | {run}", flush=True)

    rs = RulesetEncoder(tables, args.d, args.rs_depth, residual=not args.no_type_residual) if tables is not None else None
    heads = args.d // args.head_dim  # the predictor reads the encoder's rotation angles: same head width in both
    grid = args.model == "grid"

    def make_dense():
        if grid:
            return gl.GridEncoder(v, av.n_families, args.d, S=args.grid, pool_level=args.pool_level, mix_depth=args.mix_depth,
                                  conv_blocks=args.conv_blocks, attn_blocks=args.attn_blocks, heads=heads).to(dev)
        return DenseEncoder(v, av.n_families, args.d, conv_blocks=args.conv_blocks, attn_blocks=args.attn_blocks, heads=heads).to(dev)
    if grid:
        assert args.encoder == "dense" and args.w_dinv == 0, "grid: the dense encoder underneath; no per-unit Δ head (no unit tokens)"
    if args.encoder == "dense":
        assert rs is None, "the dense encoder uses the learned type tables (--types table)"
        enc = make_dense()
    elif args.encoder == "pyramid":
        assert rs is None, "the pyramid encoder uses the learned type tables (--types table)"
        enc = PyramidEncoder(v, av.n_families, args.d, widths=tuple(int(x) for x in args.widths.split(",")),
                             conv_blocks=args.conv_blocks, attn_blocks=args.attn_blocks, heads=heads).to(dev)
    else:
        enc = dm.RoPEEncoder(v, args.d, args.enc_depth, heads=heads, ruleset=rs).to(dev)
    enc.grad_ckpt = not args.no_ckpt
    COMPILED = ("ResBlock", "Down", "WindowBlock", "ConvBlock", "LocalLevel", "PatchMerge", "Block", "ScatterPool")
    if not args.no_compile:  # the encoder's blocks (the pyramid's pair search breaks its graph; dynamo handles that)
        for m in enc.modules():
            if type(m).__name__ in COMPILED:
                m.compile(dynamic=True)
    aenc = ActionEncoder(enc, av, d=args.d).to(dev)
    cat_z = args.z == "categorical"
    z_dim = args.z_groups * args.z_classes if cat_z else args.d_z   # what the rest of the model sees per player
    latent = args.model in ("latent", "grid")
    pred = None if latent else dm.DirectPredictor(args.d, args.pred_depth, heads=heads).to(dev)
    if latent:
        dyn_h = ld.PassiveDynamics(args.d, args.h_depth, heads).to(dev)
        dyn_g = ld.ActionDynamics(args.d, args.g_depth, heads, rope=dm.RoPE2D(args.d // heads) if args.g_self else None).to(dev)
        proj_z, proj_a = nn.Linear(z_dim, args.inv_dim).to(dev), nn.Linear(args.d, args.inv_dim).to(dev)  # pairing heads
        dinv = ld.DeltaInverse(args.d, v.n_abilities).to(dev)
        probes_true = dm.Probes(args.d, v.n_abilities).to(dev)  # diagnostic: the same readout from the true future latent
        # the probes read [future token, present token]: their targets are changes from t (dx, dy, ...), which a
        # future token alone can't give (it doesn't say where the unit is now)
        probe_in, probe_true_in = nn.Linear(2 * args.d, args.d).to(dev), nn.Linear(2 * args.d, args.d).to(dev)
        # ... and the horizon: how far apart the two tokens are (at 4 loops the latent's non-motion changes are as
        # large as the motion; without k the probes can't tell)
        probe_time = LoopTime().to(dev)
        probe_hor, probe_true_hor = mlp(probe_time.out_dim(), args.d).to(dev), mlp(probe_time.out_dim(), args.d).to(dev)
        enc_ema = None
        if args.target == "ema":
            assert args.encoder == "dense", "--target ema builds a second dense encoder"
            enc_ema = make_dense()
            enc_ema.load_state_dict(enc.state_dict())
            enc_ema.requires_grad_(False)
            if not args.no_compile:
                for m in enc_ema.modules():
                    if type(m).__name__ in COMPILED:
                        m.compile(dynamic=True)
    probes = dm.Probes(args.d, v.n_abilities).to(dev)
    edec = gl.EntityDecoder(args.d, v.n_units, v.n_abilities, C_ENTITY, args.dec_depth, heads).to(dev) if grid else None
    post = InverseModel(args.d, z_dim if cat_z else 2 * args.d_z).to(dev)
    prior = (lat.CategoricalPrior(3 * args.d, args.z_groups, args.z_classes) if cat_z
             else dm.PriorHead(3 * args.d, args.d_z, args.prior_k)).to(dev)
    player_emb = nn.Embedding(8, args.d).to(dev)
    withheld = nn.Embedding(2, args.d).to(dev)  # commands given / withheld, for the prior
    zup = nn.Linear(z_dim, args.d).to(dev)
    intent = EnemyIntent(v.n_abilities, args.d).to(dev)  # the posterior's view of the enemy's orders and commands
    if not args.no_compile:
        for m in ((dyn_h, dyn_g) if latent else (pred,)) + (post, prior, probes, intent) + ((edec,) if grid else ()):
            m.compile(dynamic=True)
    modules = dict(encoder=enc, actions=aenc, probes=probes, posterior=post, prior=prior,
                   player_emb=player_emb, withheld=withheld, zup=zup, intent=intent)  # the ruleset encoder is inside the state encoder
    if latent:
        modules.update(dyn_h=dyn_h, dyn_g=dyn_g, proj_z=proj_z, proj_a=proj_a, dinv=dinv, probes_true=probes_true,
                       probe_in=probe_in, probe_true_in=probe_true_in, probe_hor=probe_hor, probe_true_hor=probe_true_hor)
    else:
        modules["predictor"] = pred
    if grid:
        modules["entity_dec"] = edec
    params = [p for m in modules.values() for p in m.parameters()]
    if latent and enc_ema is not None:
        modules["encoder_ema"] = enc_ema  # saved and restored, not optimized
    n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    print(f"encoder {n(enc):.2f}M" + (f" (ruleset encoder {n(rs):.2f}M)" if rs is not None else "") + f", action encoder {n(aenc):.2f}M, "
          + (f"dynamics h {n(dyn_h):.2f}M g {n(dyn_g):.2f}M" if latent else f"predictor {n(pred):.2f}M")
          + f", probes {n(probes):.2f}M, posterior {n(post):.2f}M, prior {n(prior):.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    warm = 200
    sched_len = args.sched_steps or args.steps
    lr_at = lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / sched_len)))
    start = 0
    if ck is not None:
        fresh = [k for k in modules if k not in ck]
        for k, m in modules.items():
            if k in ck:
                m.load_state_dict(ck[k])
        if fresh:
            print(f"modules not in the checkpoint (fresh): {', '.join(fresh)}", flush=True)
        if not fresh and not args.refit_probes:
            opt.load_state_dict(ck["opt"])
        start = ck["step"]
        rng.set_state(ck["np_rng"])
        print(f"resuming {run} at step {start}", flush=True)
    for g_ in opt.param_groups:
        g_.setdefault("initial_lr", args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at, last_epoch=start - 1)

    # TensorBoard: `tensorboard --logdir data/worldmodel/runs` overlays every run. Evaluation charts
    # put the model's errors and persistence's on one chart per (set, metric, horizon); purge_step
    # drops points a resumed run logged after its checkpoint.
    tb = SummaryWriter(os.path.join(run, "tb_eval" if args.eval_only else "tb"), purge_step=None if args.eval_only else start or None)
    if not start:
        tb.add_text("config", "```\n" + json.dumps(vars(args), indent=1) + "\n```")
    tag = lambda s_: re.sub(r"[^A-Za-z0-9_.-]+", "_", s_)
    EVAL_KEYS = ("exit acc", "died recall", "pos err", "pos err moving", "pos err touched", "pos err enemy mv",
                 "pos err tight sm", "pos err tight lg", "hp mae", "shield mae", "order acc")
    if latent:  # the same readout from the true future latent: is motion in the encoder at all?
        EVAL_KEYS = EVAL_KEYS + ("pos mv truelat",)
    if grid:
        EVAL_KEYS = ("pos err", "pos err moving", "count L1", "hp mae", "order acc", "latent ratio", "hungarian gap")
    eval_names = [f"{name}{sfx}" for name in eval_games for sfx in ("", " obs")]  # player views / the combined view
    tb.add_custom_scalars({"train vs validation": {"loss (probes, no KL)": ["Multiline", ["^loss/"]], "KL": ["Multiline", ["^kl/"]]},
                           **{f"{name} {key}": {f"k={k} loops": ["Multiline", [f"^{tag(name)}/{tag(key)}/k{k:02d}/"]] for k in eval_k}
                              for name in eval_names for key in EVAL_KEYS + ("KL",)}})

    def draw_batch(games, n_frames, n_hor, rng, ks=None, with_actions=None, view=None):
        """Samples grouped by present frame (collate_samples relies on the grouping). games: game
        sets; view: "player" / "obs" (default: drawn, --obs_share combined). ks: horizons in game
        loops (default: drawn from hor_loops, --hor_dist)."""
        out = []
        for gi in rng.randint(len(games), size=n_frames):
            game = games[gi].pick(rng, args.obs_share, view)
            hi = len(game.g) - k_max - 1
            if hi < 1:
                continue
            t = rng.randint(0, hi + 1)
            om = game.g.owner_map("player" if game.viewer else "observer", me_slot=game.viewer or None, augment=True, rng=rng)
            frame = game.g.frame(t, om)  # one frame object per present frame: collate_samples groups by identity
            for j in range(n_hor):
                k_loops = int(ks[j]) if ks is not None else int(rng.choice(hor_loops, p=hor_p))
                k = k_loops // step_loops  # frames ahead
                wa = bool(with_actions) if with_actions is not None else bool(rng.rand() >= args.act_drop)
                out.append(make_sample(game, t, k, rng, wa, om, frame))
        return out

    def forward(b, z_from="post", cache=None):
        """One pass. z_from: post (a posterior sample in training, its mean otherwise), prior
        (a sample) or prior_mean. Returns the probes' logits, the targets' mask and the latent terms."""
        fb, tb, ab = b["fb"], b["tb"], b["ab"]
        fi = b["fi"]
        if rs is not None:
            rs.clear()  # one ruleset pass per batch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with RF("encode present"):
                e = cache["e"] if cache is not None else enc(fb)
            tokens, mask, ang = e["tokens"][fi], e["mask"][fi], e["ang"][fi]
            P = e["P"]
            B, T, d = tokens.shape
            with RF("encode commands"):
                summaries, s_mask0, h = aenc(ab, horizon_loops=b["hor"])
            # one summary per summarized player (all players in the observer view, the viewer alone
            # in a player view): scatter them to the frame's player order; the rest stay zero / masked
            S = summaries.shape[2]
            pidx = b["sum_pidx"][:, :S]
            valid = (pidx >= 0) & s_mask0[:, 0, :S]
            # fixed-shape scatter (masked indexing would make the CPU wait to learn its size)
            at = pidx.clamp(min=0)
            summ = torch.zeros(B, P, d, device=tokens.device, dtype=summaries.dtype).scatter_add_(
                1, at[..., None].expand(-1, -1, d), summaries[:, 0] * valid[..., None].to(summaries.dtype))
            s_mask = torch.zeros(B, P, device=tokens.device).scatter_add_(1, at, valid.float()) > 0
            s_mask = s_mask & fb["ply_mask"][fi]
            summ_inv, smask_inv = summ, s_mask       # the inverse's targets: every player whose commands are known
            if grid and enc.training:
                se, sem, _ = aenc(b["ab_en"], horizon_loops=b["hor"])
                ep = b["en_pidx"]
                ve = (ep >= 0) & sem[:, 0, 0] & torch.gather(fb["ply_mask"][fi], 1, ep.clamp(min=0)[:, None])[:, 0]
                oh = F.one_hot(ep.clamp(min=0), P).bool() & ve[:, None]
                summ_inv = summ_inv + oh[..., None].to(summ.dtype) * se[:, 0, :1].to(summ.dtype)
                smask_inv = smask_inv | oh
            summ = summ * b["has_act"][:, None, None].to(summ.dtype)
            # prior: present + commands (or "withheld"); posterior: present + future
            ply = e["players"][fi] + player_emb.weight[:P][None].to(tokens.dtype)
            # commands given / withheld per player: a player view never gives the enemy's
            wh = withheld((b["has_act"][:, None] & s_mask).long()).to(tokens.dtype)
            pin = torch.cat([e["cls"][fi][:, None].expand(-1, P, -1), ply, summ.to(tokens.dtype) + wh], -1)
            pri = prior(pin)
            kl = tokens.new_zeros(B, P).float()
            kl_ch = kl                            # what's charged, per (sample, player)
            if cat_z:
                p_pri = lat.cat_probs(pri, args.unimix)                                  # [B, P, G, C]
            u0, u1 = e.get("unit_slice", (1 + P, tokens.shape[1]))                  # the units' rows (probes, z)
            post_mu, ef_live, hd_pre = None, None, None
            if z_from == "post" or enc.training or latent:
                with RF("encode future"):
                    if cache is not None:  # sequences: the futures are the next steps' presents, already encoded
                        ef_live, ef = cache["ef_live"], cache["ef"]
                    elif grid:  # the online future (with gradient: the inverse reads its change) + the targets' copy
                        ef_live = enc(tb)
                        if enc_ema is not None:
                            with torch.no_grad():
                                ef = enc_ema(tb)
                        else:
                            ef = ef_live
                    elif latent and args.w_dinv > 0 and enc.training:  # futures with gradient, for the Δ inverse head only
                        ef_live = enc(tb)
                        if enc_ema is not None:  # the targets come from the moving-average copy: a second encoding
                            with torch.no_grad():
                                ef = enc_ema(tb)
                        else:
                            ef = ef_live
                    elif latent:  # targets: stop-gradient (the online encoder, or its moving-average copy)
                        with torch.no_grad():
                            ef = (enc_ema or enc)(tb)
                    else:
                        ef = enc(tb)
                if grid:
                    # the posterior is the inverse: from the action residual δ_a = E(s_{t+k}) − E(s_t) − sg(h(E(s_t), k))
                    # only (never the state itself), the players' action embeddings z
                    with RF("dynamics h"):
                        hd_pre = dyn_h(tokens, mask, ang, b["hor"])
                    delta_a = ef_live["tokens"] - tokens - hd_pre.detach()
                    if len(b["priv"]["ord_idx"]) or len(b["priv"]["cmd_idx"]):
                        # player views: the enemy's true orders and commands on the units the viewer sees, pooled into
                        # those units' latent cells (action information, not state)
                        N_ = fb["ent_mask"].shape[1]
                        iv = intent(b["priv"], B * N_, delta_a.dtype).view(B, N_, -1)
                        cell = gl.cell_of(fb["ent_pos"][fi].float(), e["cell_size"][fi], args.grid)
                        c0_ = e["cell_slice"][0]
                        add = torch.zeros(B, args.grid ** 2, iv.shape[-1], device=iv.device, dtype=iv.dtype).scatter_add_(
                            1, cell[..., None].expand(-1, -1, iv.shape[-1]), iv * fb["ent_mask"][fi][..., None].to(iv.dtype))
                        delta_a = torch.cat([delta_a[:, :c0_], delta_a[:, c0_:] + add], 1)
                    post_out = post(delta_a, delta_a[:, :0], mask, mask[:, :0], P, horizon_loops=b["hor"])
                else:
                    pre = e["tokens"][fi]
                    if len(b["priv"]["ord_idx"]) or len(b["priv"]["cmd_idx"]):  # player views: the enemy's intent
                        pre = pre.clone()
                        Nu = u1 - u0
                        pre[:, u0:u1] = pre[:, u0:u1] + intent(b["priv"], B * Nu, pre.dtype).view(B, Nu, -1)
                    fut_tok = ef["tokens"].detach() if latent else ef["tokens"]
                    post_out = post(pre, fut_tok, mask, ef["mask"], P, horizon_loops=b["hor"])
                if cat_z:
                    q_post = lat.cat_probs(post_out.view(B, P, args.z_groups, args.z_classes), args.unimix)
                    kl_ch, kl = lat.cat_kl_balanced(q_post, p_pri, args.kl_dyn, args.kl_rep, args.free_bits)
                    z = lat.cat_sample(q_post) if enc.training else lat.cat_mode(q_post)
                    post_mu = q_post.reshape(B, P, -1)                                   # for the Barlow Twins pairing
                else:
                    post_mu, post_lv = lat.split_gauss(post_out)
                    zs = lat.rsample(post_mu, post_lv)
                    kl = lat.kl_mc(zs, post_mu, post_lv, pri)
                    kl_ch = kl.clamp(min=args.free_nats)
                    z = zs if enc.training else post_mu
                if not (z_from == "post" or enc.training):
                    post_mu = post_mu.detach()
            if z_from == "prior":
                z = lat.cat_sample(p_pri).detach() if cat_z else lat.mix_sample(pri)
            elif z_from == "prior_mean":
                z = p_pri if cat_z else lat.mix_mean(pri)
            if cat_z:
                z = z.reshape(B, P, -1)
            zt = zup(z.float()).to(tokens.dtype)                                      # [B, P, d]
            if latent:
                return latent_step(b, e, ef, tokens, mask, ang, h, zt, post_mu, summ, s_mask, kl, kl_ch, u0, u1, P,
                                   p_pri if cat_z else None, ef_live, hd_pre, (summ_inv, smask_inv))
            ztok = torch.zeros_like(tokens)
            ztok[:, 1:1 + P] = zt
            pidx = b["pidx"]
            owned = pidx >= 0
            zu = torch.gather(zt, 1, pidx.clamp(min=0)[..., None].expand(-1, -1, zt.shape[-1])) * owned[..., None].to(zt.dtype)
            ztok = torch.cat([ztok[:, :u0], zu, ztok[:, u1:]], 1)
            x = tokens + ztok
            with RF("predictor"):
                hh = pred(x, mask, ang, b["hor"], h, ab["act_mask"] & b["has_act"][:, None])
            out = probes(hh[:, u0:u1])
        real = fb["ply_mask"][fi]  # every player has a z: the viewer's and, in a player view, the enemy's
        m = real.float()
        return out, fb["ent_mask"][fi], dict(kl=(kl * m).sum() / m.sum().clamp(min=1),
                                            kl_charged=(kl_ch * m).sum() / m.sum().clamp(min=1),
                                            kl_bp=kl.detach(), kl_own=s_mask & real, kl_enemy=~s_mask & real)

    pair_q = collections.deque(maxlen=args.inv_queue)

    def latent_step(b, e, ef, s, mask, ang, act_tok, zt, post_mu, summ, s_mask, kl, kl_ch, u0, u1, P, p_pri=None, ef_live=None,
                    hd_pre=None, inv_tgt=None):
        """ŝ_{t+k} = s_t + h(s_t, k) + g(·, actions, z) and the latent losses (inside forward's autocast)."""
        fb, ab, fi = b["fb"], b["ab"], b["fi"]
        B, T, d = s.shape
        has_act = b["has_act"]
        with RF("dynamics"):
            hd = hd_pre if hd_pre is not None else dyn_h(s, mask, ang, b["hor"])
            mem = torch.cat([act_tok, zt + player_emb.weight[:P][None].to(zt.dtype)], 1)
            mem_mask = torch.cat([ab["act_mask"] & has_act[:, None], fb["ply_mask"][fi]], 1)
            gd = dyn_g(s + hd, b["hor"], mem, mem_mask, mask, ang)
            s_hat = s + hd + gd
        # targets: each present token's counterpart in the future encoding (globals by position, units
        # by id, cells by grid position; effects have none)
        fr = b.get("fut_row")  # (rollout sequences have none: the grid needs no per-unit alignment)
        if grid:  # a fixed-size latent: every token's target is the same token of the future latent
            tgt = ef["tokens"].detach().float()
            fut_live = tgt
            ok = mask & ef["mask"]
        else:
            f0, f1 = ef["unit_slice"]
            c0, c1 = e["cell_slice"]
            fc0, fc1 = ef["cell_slice"]
            tidx = torch.full((B, T), -1, dtype=torch.long, device=s.device)
            tidx[:, :1 + P] = torch.arange(1 + P, device=s.device)
            tidx[:, u0:u1] = torch.where(fr >= 0, f0 + fr, -1)
            if c1 - c0 == fc1 - fc0:
                tidx[:, c0:c1] = torch.arange(fc0, fc1, device=s.device)
            gat = lambda t_: torch.gather(t_, 1, tidx.clamp(min=0)[..., None].expand(-1, -1, d))
            tgt = gat(ef["tokens"]).detach().float()          # targets: the online encoder or its moving-average copy
            fut_live = gat(ef_live["tokens"]) if ef_live is not None else tgt  # the online encoding with gradient (Δ head)
            ok = (tidx >= 0) & mask & torch.gather(ef["mask"], 1, tidx.clamp(min=0))
        w = ok.float()
        lat_loss = (((s_hat.float() - tgt) ** 2).mean(-1) * w).sum() / w.sum().clamp(min=1)
        # how much the latent changes (persistence's error in latent space); lat / chg < 1: better than persistence
        chg = (((s.float() - tgt) ** 2).mean(-1) * w).sum() / w.sum().clamp(min=1)
        # h alone on the samples with commands withheld (s + h, no g): h is the best prediction from the state alone, so
        # the action residual δ_a = target − s − h is what the state can't explain (g still reads z there)
        wh = w * (~has_act).float()[:, None]
        hal = (((s.float() + hd.float() - tgt) ** 2).mean(-1) * wh).sum() / wh.sum().clamp(min=1)
        es_loss = s.new_zeros(()).float()
        if args.es_k > 1 and p_pri is not None and enc.training:
            # energy score over K futures from prior samples (straight-through, so the prior learns): mean distance to
            # the truth minus half the mean distance between samples, per token (distances / sqrt(d))
            mem_mask_k = torch.cat([ab["act_mask"] & has_act[:, None], fb["ply_mask"][fi]], 1)
            preds = []
            for _ in range(args.es_k):
                zk = zup(lat.cat_sample(p_pri).reshape(B, P, -1)).to(s.dtype)
                memk = torch.cat([act_tok, zk + player_emb.weight[:P][None].to(zk.dtype)], 1)
                preds.append((s + hd + dyn_g(s + hd, b["hor"], memk, mem_mask_k, mask, ang)).float())
            sc = d ** 0.5
            to_truth = torch.stack([((pk - tgt) ** 2).sum(-1).add(1e-8).sqrt() / sc for pk in preds]).mean(0)
            spread = torch.stack([((preds[i] - preds[j]) ** 2).sum(-1).add(1e-8).sqrt() / sc
                                  for i in range(len(preds)) for j in range(i + 1, len(preds))]).mean(0)
            es_loss = ((to_truth - 0.5 * spread) * w).sum() / w.sum().clamp(min=1)
        var_loss, cov_loss = ld.vicreg(s, mask)
        dec_loss = ld.cross_corr(hd, gd, mask) + ld.cross_corr(gd, s, mask)
        gvar_loss = ld.floor_std(gd, mask & has_act[:, None], args.g_floor)
        inv_loss = s.new_zeros(()).float()
        npairs = s.new_zeros(()).float()
        if post_mu is not None:
            pairs = s_mask & has_act[:, None]
            npairs = pairs.float().sum()
            if grid and inv_tgt is not None and enc.training:  # every known player's commands, withheld or not
                summ_t, pairs = inv_tgt[0], inv_tgt[1]
                npairs = pairs.float().sum()
            else:
                summ_t = summ
            pz, pa = proj_z(post_mu.float()), proj_a(summ_t.float())
            if args.inv_pair == "vicreg":
                inv_loss = ld.vicreg_pair(pz, pa, pairs, queue=pair_q if enc.training else None)
                if enc.training:  # recent projections, detached: the variance / covariance statistics span ~inv_queue batches
                    pair_q.append((pz.detach().reshape(-1, pz.shape[-1]), pa.detach().reshape(-1, pa.shape[-1]),
                                   pairs.reshape(-1)))
            else:
                inv_loss = ld.barlow(pz, pa, pairs, args.inv_off)
        dinv_loss = s.new_zeros(()).float()
        if args.w_dinv > 0 and fut_live.requires_grad:
            # per unit, from its change only: moved, touched by a command, order changed, the new order
            here = (fr >= 0) & fb["ent_mask"][fi]
            delta = (fut_live[:, u0:u1] - s[:, u0:u1]).float()
            flags, ord_logits, dproj = dinv(delta, b["hor"])
            tg = b["tgt"]
            moved = ((tg["dx"] ** 2 + tg["dy"] ** 2).sqrt() > 0.5).float()
            changed = tg["order"] != b["now"]["order"]
            want = torch.stack([moved, b["touched"].float(), changed.float()], -1)
            hw = here.float()
            bce = F.binary_cross_entropy_with_logits(flags.float(), want, reduction="none").mean(-1)
            ce = F.cross_entropy(ord_logits.float().flatten(0, 1), tg["order"].flatten().clamp(min=0), reduction="none").view_as(hw)
            cw = hw * changed.float()
            # the raw feature change of each unit, on comparable scales at the source: position as a fraction of the
            # playable area (ent_xy) and the entity numbers (fractions, logs, /max by construction); no batch scaling
            tb_ = b["tb"]
            frc = fr.clamp(min=0)
            take = lambda x: torch.gather(x, 1, frc[..., None].expand(-1, -1, x.shape[-1]))
            raw = torch.cat([take(tb_["ent_xy"]) - fb["ent_xy"][fi], take(tb_["ent_num"]) - fb["ent_num"][fi]], -1).float()
            dbar = ld.barlow(dproj.float(), dinv.raw_in(raw), here, args.dbar_off)
            dinv_loss = (bce * hw).sum() / hw.sum().clamp(min=1) + (ce * cw).sum() / cw.sum().clamp(min=1) + args.w_dbar * dbar
        if grid:
            return grid_decode(b, e, ef, s, s_hat, tgt, mask, dict(
                kl=(kl * fb["ply_mask"][fi].float()).sum() / fb["ply_mask"][fi].float().sum().clamp(min=1),
                kl_charged=(kl_ch * fb["ply_mask"][fi].float()).sum() / fb["ply_mask"][fi].float().sum().clamp(min=1),
                kl_bp=kl.detach(), kl_own=s_mask & fb["ply_mask"][fi], kl_enemy=~s_mask & fb["ply_mask"][fi],
                lat=lat_loss, var=var_loss, cov=cov_loss, dec=dec_loss, gvar=gvar_loss, inv=inv_loss, dinv=dinv_loss,
                chg=chg.detach(), ratio=(lat_loss / chg.clamp(min=1e-8)).detach(), es=es_loss, hal=hal, pairs=npairs.detach(),
                _ol=dict(s=s, mem=mem, mem_mask=mem_mask, tgt=tgt, mask=mask, ang=ang, summ=summ, s_mask=s_mask)))
        # diagnostic probes on the true future latent (detached): does the encoder keep motion at all?
        here_t = (fr >= 0) & fb["ent_mask"][fi]
        cur = s[:, u0:u1].detach()
        kt = probe_time(b["hor"])
        out_true = probes_true(probe_true_in(torch.cat([tgt[:, u0:u1].to(s.dtype), cur], -1)) + probe_true_hor(kt)[:, None].to(s.dtype))
        units = s_hat[:, u0:u1]
        w_p = args.probe_weight  # same values, gradient into the latent scaled by w_p
        out = probes(probe_in(torch.cat([units.detach() if w_p <= 0 else w_p * units + (1 - w_p) * units.detach(), cur], -1))
                     + probe_hor(kt)[:, None].to(s.dtype))
        real = fb["ply_mask"][fi]
        m = real.float()
        return out, fb["ent_mask"][fi], dict(
            kl=(kl * m).sum() / m.sum().clamp(min=1), kl_charged=(kl_ch * m).sum() / m.sum().clamp(min=1),
            kl_bp=kl.detach(), kl_own=s_mask & real, kl_enemy=~s_mask & real,
            lat=lat_loss, var=var_loss, cov=cov_loss, dec=dec_loss, gvar=gvar_loss, inv=inv_loss, dinv=dinv_loss,
            chg=chg.detach(), ratio=(lat_loss / chg.clamp(min=1e-8)).detach(), out_true=out_true, here_true=here_t, es=es_loss)

    def forward_seq(b, z_from="post"):
        """Sequences: every frame encoded once online (with gradient) and once by the targets' copy; the
        one-step pass (teacher-forced: each step from the real present) is forward() on those encodings;
        then the open-loop rollout from each sequence's first frame (its own predictions fed back)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with RF("encode frames"):
                e_all = enc(b["fb"])
                if enc_ema is not None:
                    with torch.no_grad():
                        em_all = enc_ema(b["fb"])
                else:
                    em_all = sel_enc(e_all, torch.arange(e_all["tokens"].shape[0], device=dev))
        cache = dict(e=e_all, ef_live=sel_enc(e_all, b["fut"]), ef=sel_enc(em_all, b["fut"]))
        out, mask, lt = forward(b, z_from, cache)
        ol, x_last, _ = open_loop(b, lt, "post" if z_from == "post" else "prior")
        lt["ol"] = ol
        lt["_x_last"] = x_last
        if enc.training:  # the decoder on the rollout's last prediction (truth at the last frame)
            o = lt["_ol"]
            S, n = b["S"], b["n"]
            last_mask = o["mask"].view(S, n, -1)[:, -1]
            tl = gl.targets(b["tb_last"], args.grid, v.n_units)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                Lol, _, _, _ = gl.decode_losses(edec, gl.grad_scale(x_last, args.dec_grad), last_mask, e_all["cell_slice"], tl)
            out = {**out, **{f"{k_}_ol": x for k_, x in Lol.items()}}
        return out, mask, lt

    def open_loop(b, lt, z_mode="post"):
        """Roll the dynamics from each sequence's first real latent, feeding predictions back, with each
        step's commands (masked when withheld) and z: the posterior's (from the real step's action
        residual) or a sample of the prior read from the predicted latent. Returns the mean latent loss
        over steps against the targets' copy, the last prediction and the per-step losses."""
        o = lt["_ol"]
        S, n = b["S"], b["n"]
        rs = lambda x: x.view(S, n, *x.shape[1:])
        s_, mem, mm, tgt, mask, ang = (rs(o[k_]) for k_ in ("s", "mem", "mem_mask", "tgt", "mask", "ang"))
        hor, has_act = rs(b["hor"]), rs(b["has_act"])
        P = o["summ"].shape[1]
        if z_mode == "prior":
            summ, smask = rs(o["summ"]), rs(o["s_mask"])
        x = s_[:, 0]
        per = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(n):
                m_i = mem[:, i]
                if z_mode == "prior":  # z from the prior on the predicted latent (imagination)
                    pe = player_emb.weight[:P][None].to(x.dtype)
                    wh = withheld((has_act[:, i, None] & smask[:, i]).long()).to(x.dtype)
                    pin = torch.cat([x[:, :1].expand(-1, P, -1), x[:, 1:1 + P] + pe, summ[:, i].to(x.dtype) + wh], -1)
                    zk = lat.cat_sample(lat.cat_probs(prior(pin), args.unimix)).reshape(S, P, -1)
                    m_i = torch.cat([m_i[:, :-P], zup(zk.float()).to(x.dtype) + pe], 1)
                hd = dyn_h(x, mask[:, i], ang[:, i], hor[:, i])
                x = x + hd + dyn_g(x + hd, hor[:, i], m_i, mm[:, i], mask[:, i], ang[:, i])
                w = mask[:, i].float()
                per.append((((x.float() - tgt[:, i]) ** 2).mean(-1) * w).sum() / w.sum().clamp(min=1))
        return torch.stack(per).mean(), x, per

    def draw_seq(games, n_seq, n, rng, d=None, with_actions=None, view=None, base=False):
        """n_seq sequences of n steps; the step (frames) drawn from --deltas per sequence unless given."""
        out = []
        while len(out) < n_seq:
            game = games[rng.randint(len(games))].pick(rng, args.obs_share, view)
            dd = int(d) if d is not None else int(rng.choice(deltas))
            hi = len(game.g) - n * dd - 1
            if hi < 1:
                continue
            t0 = rng.randint(0, hi + 1)
            om = game.g.owner_map("player" if game.viewer else "observer", me_slot=game.viewer or None, augment=True, rng=rng)
            wa = bool(with_actions) if with_actions is not None else bool(rng.rand() >= args.act_drop)
            out.append(make_seq(game, t0, dd, n, rng, wa, om, game.g.frame(t0, om), base))
        return out

    def seq_eval_runner():
        """evaluate_grid's hooks for rollouts: a horizon K is reached by the smallest step d in --deltas
        with K / d steps <= --seq_len; the metrics compare the rollout's last prediction with the truth."""
        def plan(k):
            for d_ in sorted(deltas):
                if k % (d_ * step_loops) == 0 and k // (d_ * step_loops) <= args.seq_len:
                    return d_, k // (d_ * step_loops)
            raise ValueError(f"horizon {k} loops: no step in --deltas reaches it within --seq_len steps")

        def make_batches(games, view, k, erng):
            d_, n_ = plan(k)
            per = max(1, min(args.seqs * 2, args.seq_eval_states))
            return [draw_seq(games, per, n_, erng, d=d_, with_actions=True, view=view, base=True)
                    for _ in range(max(1, args.seq_eval_states // per))]

        def run(seqs, wa, zf):
            b = collate_seq(seqs, dev, has_act=wa)
            _, _, lt = forward_seq(b, z_from=zf)
            bb = collate_samples([q["base"] for q in seqs], dev)
            o = lt["_ol"]
            S, n = b["S"], b["n"]
            last = lambda x: x.view(S, n, *x.shape[1:])[:, -1]
            tgt_k, mask_k, s0 = last(o["tgt"]), last(o["mask"]), o["s"].view(S, n, *o["s"].shape[1:])[:, 0]
            x = lt["_x_last"]
            cs = (1 + o["summ"].shape[1], x.shape[1])
            tp, tf_ = gl.targets(bb["fb"], args.grid, v.n_units), gl.targets(bb["tb"], args.grid, v.n_units)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, outf, lrf, pair = gl.decode_losses(edec, x, mask_k, cs, tf_)
                gm = grid_metrics(bb, tp, tf_, outf, lrf, pair, tgt_k.to(x.dtype), mask_k, cs)
            w = mask_k.float()
            err = lambda a: float((((a.float() - tgt_k) ** 2).mean(-1) * w).sum() / w.sum().clamp(min=1))
            return gm, err(x) / max(err(s0), 1e-8), float(lt["kl"])
        return make_batches, run

    def grid_decode(b, e, ef, s, s_hat, tgt, mask, lt):
        """Grid mode: the entity decoder on the present latent (truth at t) and on the prediction
        ŝ_{t+k} (truth at t+k), each with --dec_grad of its gradient into the latent, and the
        second-moment losses (latent cells; decoded entities vs the target encoder's entity tokens;
        entities x their cells), predicted vs target."""
        fb, tb = b["fb"], b["tb"]
        cs = e["cell_slice"]
        c0, c1 = cs
        d = s.shape[-1]
        tp = gl.targets(fb, args.grid, v.n_units)
        Lp, _, _, _ = gl.decode_losses(edec, gl.grad_scale(e["tokens"], args.dec_grad), e["mask"], cs, tp)
        tf_ = gl.targets(tb, args.grid, v.n_units)
        Lf, outf, lrf, pair = gl.decode_losses(edec, gl.grad_scale(s_hat, args.dec_grad), mask, cs, tf_)
        tm = tf_["mask"]
        du = ef["dense"]["units"].detach()                 # the target encoder's entity tokens at t+k (same rows)
        cells_at = lambda x: torch.gather(x[:, c0:c1], 1, tf_["cell"][..., None].expand(-1, -1, d))
        g_m = ((gl.gram(s_hat[:, c0:c1]) - gl.gram(tgt[:, c0:c1])) ** 2).mean()
        g_e = ((gl.gram(outf["h"], tm) - gl.gram(du, tm)) ** 2).mean()
        g_x = ((gl.cross_gram(outf["h"], cells_at(s_hat), tm) - gl.cross_gram(du, cells_at(tgt), tm)) ** 2).mean()
        lt["gram"] = g_m + g_e + g_x
        out = {**{f"{k_}_now": x for k_, x in Lp.items()}, **{f"{k_}_fut": x for k_, x in Lf.items()}}
        if not enc.training and "tgt" in b:  # (rollout evaluation computes its metrics at the rollout's end)
            lt["gm"] = grid_metrics(b, tp, tf_, outf, lrf, pair, tgt.to(s_hat.dtype), mask, cs)
        return out, tm, lt

    @torch.no_grad()
    def grid_metrics(b, tp, tf_, outf, lrf, pair, tgt, mask, cs):
        """Evaluation: the decoded future against the truth at t+k, from ŝ (model), from the target
        latent (truelat: what the latent holds) and, where defined, persistence (the units' own
        displacement; the present's counts)."""
        fi = b["fi"]
        B, N = tf_["mask"].shape
        fl = lambda x: x.reshape(-1, *x.shape[2:])
        tm = tf_["mask"]

        def per_row(out, pr):
            ip, it, okm = pr
            sz = tf_["size"][it // N]
            err = (fl(out["off"])[ip] - fl(tf_["off"])[it]).norm(dim=-1) * sz
            hp = (fl(out["num"])[ip][:, 0] - fl(tf_["num"])[it][:, 0]).abs()
            oa = (fl(out["order"])[ip].argmax(-1) == fl(tf_["order"])[it]).float()
            put = lambda x: torch.zeros(B * N, device=x.device).scatter_(0, it, torch.where(okm, x, 0)).view(B, N)
            return put(err), put(hp), put(oa)

        tg, fr = b["tgt"], b["fut_row"]
        disp = (tg["dx"] ** 2 + tg["dy"] ** 2).sqrt()
        mv = (fr >= 0) & b["fb"]["ent_mask"][fi] & (disp > 0.5)
        idx = torch.where(mv, fr, N)                        # the moving units' rows at t+k (N: nowhere)
        mv_row = torch.zeros(B, N + 1, device=disp.device).scatter_(1, idx, 1.0)[:, :N] > 0
        mv_row = mv_row & tm
        persist_mv = torch.zeros(B, N + 1, device=disp.device).scatter_(1, idx, disp * mv)[:, :N]
        avg = lambda x, w: float((x * w).sum() / w.float().sum().clamp(min=1))
        l1 = lambda lr: float((torch.round(lr.exp()) - tf_["counts"]).abs().sum((1, 2)).mean())
        _, outt, lrt, pt = gl.decode_losses(edec, tgt, mask, cs, tf_)
        r = {}
        for nm, (o_, l_, p_) in (("model", (outf, lrf, pair)), ("truelat", (outt, lrt, pt))):
            err, hp, oa = per_row(o_, p_)
            r[f"pos err/{nm}"] = avg(err, tm)
            r[f"pos err moving/{nm}"] = avg(err, mv_row)
            r[f"count L1/{nm}"] = l1(l_)
            r[f"hp mae/{nm}"] = avg(hp, tm)
            r[f"order acc/{nm}"] = avg(oa, tm)
            if nm == "model":
                ctr = tf_["ctr"]
                sz = tf_["size"][:, None, None]
                r["_hung"] = (ctr + outf["off"] * sz, ctr + tf_["off"] * sz, tf_["oc"], tm, err)
        r["pos err moving/persist"] = avg(persist_mv, mv_row)
        r["count L1/persist"] = float((tp["counts"][fi] - tf_["counts"]).abs().sum((1, 2)).mean())
        return r

    def hungarian_gap(h):
        """Mean position error of the decoded future under the in-cell pairing (sorted by y) and under a
        global Hungarian assignment within each (owner, type), ignoring cells: the gap is how much the
        cell-local pairing mismatches (units decoded across a cell border, pairs crossed in x)."""
        from scipy.optimize import linear_sum_assignment
        pa, ta, oc, tm, err = (x.float().cpu().numpy() if x.dtype != torch.bool else x.cpu().numpy() for x in h)
        oc = oc.astype(np.int64)
        tot_c = tot_h = n = 0.0
        for i in range(len(tm)):
            m = tm[i]
            o, P_, T_ = oc[i][m], pa[i][m], ta[i][m]
            tot_c += err[i][m].sum()
            n += m.sum()
            for g_ in np.unique(o):
                sel = o == g_
                C = np.linalg.norm(P_[sel][:, None] - T_[sel][None], axis=-1)
                r_, c_ = linear_sum_assignment(C)
                tot_h += C[r_, c_].sum()
        return tot_c / max(n, 1), tot_h / max(n, 1)

    def direct_eval_runner():
        def make_batches(games, view, k, erng):
            return [draw_batch(games, args.frames, 1, erng, ks=[k], with_actions=True, view=view)
                    for _ in range(max(1, args.val_states // args.frames))]

        def run(samples, wa, zf):
            b = collate_samples(samples if wa else withhold(samples), dev)
            _, _, lt = forward(b, z_from=zf)
            return lt["gm"], float(lt["ratio"]), float(lt["kl"])
        return make_batches, run

    @torch.no_grad()
    def evaluate_grid(step):
        make_batches, run = seq_eval_runner() if args.seq_len else direct_eval_runner()
        for m in modules.values():
            m.eval()
        wanted = {x for x in args.eval_sets.split(",") if x} if args.eval_sets else set(eval_games)
        cols = (("model", "post"), ("model", "prior"), ("truelat", "post"), ("persist", "post"))
        for (name0, games), view in ((x, vw) for x in eval_games.items() if x[0] in wanted for vw in ("player", "obs")):
            name = name0 + ("" if view == "player" else " obs")
            for k in eval_k:
                erng = np.random.RandomState(1234 + k)
                batches = make_batches(games, view, k, erng)
                rows, gaps = {}, []
                for wa in (True, False):
                    acc, kls = collections.defaultdict(list), []
                    for bi, samples in enumerate(batches):
                        for zf in ("post", "prior"):
                            gm, ratio, kl_ = run(samples, wa, zf)
                            for key, val in gm.items():
                                if not key.startswith("_"):
                                    acc[(key, zf)].append(val)
                            acc[("latent ratio/model", zf)].append(ratio)
                            if zf == "post":
                                kls.append(kl_)
                                if wa and bi == 0:
                                    gaps.append(hungarian_gap(gm["_hung"]))
                    rows[wa] = (acc, float(np.mean(kls)))
                keys = ("pos err", "pos err moving", "count L1", "hp mae", "order acc", "latent ratio")
                hdr = "  ".join(f"{(c if c != 'model' else zf)[:7]:>7s}" for c, zf in cols)
                print(f"[{name} step {step}] k={k} loops ({k / LPS:.2f}s) | KL with commands {rows[True][1]:.2f}, withheld "
                      f"{rows[False][1]:.2f} nats | position error, in-cell pairing {gaps[0][0]:.3f} vs global Hungarian "
                      f"{gaps[0][1]:.3f} map units\n   {'':16s} with commands: {hdr}  | withheld: {hdr}", flush=True)
                for key in keys:
                    cells = []
                    for wa in (True, False):
                        acc = rows[wa][0]
                        cells.append("  ".join(f"{np.mean(acc[(f'{key}/{c}', zf)]):7.3f}" if acc.get((f"{key}/{c}", zf)) else f"{'':7s}"
                                               for c, zf in cols))
                        cmd = "cmds" if wa else "withheld"
                        for c, zf in cols:
                            if acc.get((f"{key}/{c}", zf)):
                                tb.add_scalar(f"{tag(name)}/{tag(key)}/k{k:02d}/{cmd}_{c if c != 'model' else zf}",
                                              np.mean(acc[(f"{key}/{c}", zf)]), step)
                    print(f"   {key:16s} " + "  | ".join(cells), flush=True)
                for wa in (True, False):
                    tb.add_scalar(f"{tag(name)}/KL/k{k:02d}/{'cmds' if wa else 'withheld'}_all", rows[wa][1], step)
                tb.add_scalar(f"{tag(name)}/hungarian gap/k{k:02d}/cell", gaps[0][0], step)
                tb.add_scalar(f"{tag(name)}/hungarian gap/k{k:02d}/global", gaps[0][1], step)
        tb.flush()
        for m in modules.values():
            m.train()

    def metrics(out, b, mask):
        return metrics_with(probes, out, b, mask)

    def metrics_with(probe_set, out, b, mask):
        """Per-probe errors of the model's point estimates and of persistence, over real
        entities (exit) / entities present at t+k (the rest)."""
        est = probe_set.point(out)
        tgt, now = b["tgt"], b["now"]
        m = mask.float()
        present = m * (tgt["exit"] == dm.EXIT_PRESENT).float()
        moved = present * ((tgt["dx"] ** 2 + tgt["dy"] ** 2).sqrt() > 0.5).float()
        avg = lambda x, w: float((x * w).sum() / w.sum().clamp(min=1))
        dist = lambda dx, dy: ((dx - tgt["dx"]) ** 2 + (dy - tgt["dy"]) ** 2).sqrt()
        r = collections.OrderedDict()
        r["exit acc"] = (avg((est["exit"] == tgt["exit"]).float(), m), avg((tgt["exit"] == dm.EXIT_PRESENT).float(), m))
        r["died recall"] = (avg((est["exit"] == dm.EXIT_DIED).float(), m * (tgt["exit"] == dm.EXIT_DIED).float()), 0.0)
        r["pos err"] = (avg(dist(est["dx"], est["dy"]), present), avg(dist(now["dx"], now["dy"]), present))
        r["pos err moving"] = (avg(dist(est["dx"], est["dy"]), moved), avg(dist(now["dx"], now["dy"]), moved))
        touched = present * b["touched"].float()  # units a command in the window selected or targeted
        enemy = moved * b["enemy"].float()          # moving enemy units (player views: what the enemy's z is for)
        r["pos err enemy mv"] = (avg(dist(est["dx"], est["dy"]), enemy), avg(dist(now["dx"], now["dy"]), enemy))
        for key, flag in (("pos err tight sm", "tight_sm"), ("pos err tight lg", "tight_lg")):  # moving, near a tight passage
            w_ = moved * b[flag].float()
            r[key] = (avg(dist(est["dx"], est["dy"]), w_), avg(dist(now["dx"], now["dy"]), w_))
        r["pos err touched"] = (avg(dist(est["dx"], est["dy"]), touched), avg(dist(now["dx"], now["dy"]), touched))
        r["touched share"] = (float(touched.sum() / present.sum().clamp(min=1)), 0.0)
        r["hp mae"] = (avg((est["hp"] - tgt["hp"]).abs(), present), avg((now["hp"] - tgt["hp"]).abs(), present))
        r["shield mae"] = (avg((est["sh"] - tgt["sh"]).abs(), present), avg((now["sh"] - tgt["sh"]).abs(), present))
        r["order acc"] = (avg((est["order"] == tgt["order"]).float(), present), avg((now["order"] == tgt["order"]).float(), present))
        r["moving share"] = (float(moved.sum() / present.sum().clamp(min=1)), 0.0)
        return r

    kl_str = lambda d_: " ".join(f"{p_} {d_[p_]:5.2f}" for p_ in ("all", "own", "enemy") if p_ in d_) + " nats"

    @torch.no_grad()
    def evaluate(step):
        if grid:
            return evaluate_grid(step)
        for m in modules.values():
            m.eval()
        wanted = {x for x in args.eval_sets.split(",") if x} if args.eval_sets else set(eval_games)
        for (name0, games), view in ((x, vw) for x in eval_games.items() if x[0] in wanted for vw in ("player", "obs")):
            name = name0 + ("" if view == "player" else " obs")
            for k in eval_k:
                erng = np.random.RandomState(1234 + k)
                rows = {}
                births = []
                batches = [draw_batch(games, args.frames, 1, erng, ks=[k], with_actions=True, view=view)
                           for _ in range(max(1, args.val_states // args.frames))]
                for wa in (True, False):  # the same states, with their commands and without
                    acc = collections.defaultdict(lambda: collections.defaultdict(list))
                    kls = collections.defaultdict(list)
                    for samples in batches:
                        b = collate_samples(samples if wa else withhold(samples), dev)
                        if wa:
                            births.append(b["births"])
                        for zf in ("post", "prior", "prior_mean"):
                            out, mask, lt = forward(b, z_from=zf)
                            if zf == "post":
                                kls["all"].append(float(lt["kl"]))
                                for part in ("own", "enemy"):  # own: summarized players (the viewer / both in obs)
                                    mm = lt[f"kl_{part}"]
                                    if mm.any():
                                        kls[part].append(float(lt["kl_bp"][mm].mean()))
                            for key, (val, base) in metrics(out, b, mask).items():
                                acc[key][zf].append(val)
                                acc[key]["persist"].append(base)
                            if latent:
                                val, base = metrics_with(probes_true, lt["out_true"], b, mask)["pos err moving"]
                                acc["pos mv truelat"][zf].append(val)
                                acc["pos mv truelat"]["persist"].append(base)
                    rows[wa] = (acc, {p_: float(np.mean(v_)) for p_, v_ in kls.items()})
                hs = k / LPS
                keys = EVAL_KEYS
                cols = ("post", "prior", "prior_mean", "persist")
                moving = np.mean(rows[True][0]["moving share"]["post"])
                touched = np.mean(rows[True][0]["touched share"]["post"])
                print(f"[{name} step {step}] k={k} loops ({hs:.2f}s) births/window {np.mean(births):.1f} | moving {moving:.0%} | touched by a command {touched:.0%}"
                      + (" | type residual norms " + " ".join(f"{k_[:4]} {v_:.2f}" for k_, v_ in rs.residual_norms().items()) if rs is not None and rs.residual else "") + "\n"
                      f"   {'':16s} with commands: KL {kl_str(rows[True][1])} | commands withheld: KL {kl_str(rows[False][1])}\n"
                      f"   {'':16s} " + "  ".join(f"{c[:6]:>6s}" for c in cols) + "  | " + "  ".join(f"{c[:6]:>6s}" for c in cols), flush=True)
                for key in keys:
                    cells = []
                    for wa in (True, False):
                        acc = rows[wa][0]
                        cells.append("  ".join(f"{np.mean(acc[key][c]):6.3f}" if not (key == "died recall" and c == "persist")
                                               else f"{'':6s}" for c in cols))
                    print(f"   {key:16s} " + "  | ".join(cells), flush=True)
                for wa in (True, False):
                    cmd = "cmds" if wa else "withheld"
                    for part, val in rows[wa][1].items():
                        tb.add_scalar(f"{tag(name)}/KL/k{k:02d}/{cmd}_{part}", val, step)
                    for key in keys:
                        for c in cols:
                            if not (key == "died recall" and c == "persist"):
                                tb.add_scalar(f"{tag(name)}/{tag(key)}/k{k:02d}/{cmd}_{c}", np.mean(rows[wa][0][key][c]), step)
        tb.flush()
        for m in modules.values():
            m.train()

    def save(step, tag):
        torch.save(dict(step=step, args=vars(args), opt=opt.state_dict(), pool=pool.names(), np_rng=rng.get_state(),
                        **{k: m.state_dict() for k, m in modules.items()}), os.path.join(run, f"model_{tag}.pt"))

    if args.eval_only and args.refit_probes and latent and not grid:
        # fresh probes on the frozen model: train only them, then evaluate
        names = ("probes", "probe_in", "probe_hor", "probes_true", "probe_true_in", "probe_true_hor")
        for k in names:
            for mm in modules[k].modules():
                if hasattr(mm, "reset_parameters"):
                    mm.reset_parameters()
        for k, m in modules.items():
            m.requires_grad_(k in names)
            m.eval()
        pparams = [p for k in names for p in modules[k].parameters()]
        popt = torch.optim.AdamW(pparams, lr=1e-3, weight_decay=0.01)
        prng_ = np.random.RandomState(args.seed + 777)
        t0 = time.time()
        for i in range(args.refit_probes):
            rb = collate_samples(draw_batch([x for _, x in pool.games], args.frames, args.hor, prng_), dev)
            out, mask, lt = forward(rb)
            pl = sum(probes.losses(out, rb["tgt"], mask).values()) + sum(probes_true.losses(lt["out_true"], rb["tgt"], lt["here_true"]).values())
            popt.zero_grad(set_to_none=True)
            pl.backward()
            popt.step()
            if (i + 1) % 100 == 0:
                print(f"refit {i + 1}/{args.refit_probes}: probe loss {float(pl):.3f} ({time.time() - t0:.0f}s)", flush=True)
            if (i + 1) % 250 == 0:
                pool.rotate(args.rotate)
    if args.eval_only:
        evaluate(start)
        return

    for m in modules.values():
        m.train()
    hist = collections.defaultdict(list)
    t_last = time.time()
    # batches are built on a background thread (its own random stream), 2 ahead; the pool
    # rotates by batches built, swapping in games a loader thread prepared
    prng = np.random.RandomState(args.seed * 1000003 + start)
    built = [start]

    def make_batch():
        if built[0] and built[0] % args.rotate_every == 0:
            pool.rotate(args.rotate)
        built[0] += 1
        t0 = time.time()
        games_ = [x for _, x in pool.games]
        b_ = (collate_seq(draw_seq(games_, args.seqs, args.seq_len, prng), dev) if args.seq_len
              else collate_samples(draw_batch(games_, args.frames, args.hor, prng), dev))
        return b_, time.time() - t0

    # validation every --val_every steps: the training objective on fixed batches from the held-out
    # games and maps (drawn once, kept on the GPU), next to the training average
    vrng = np.random.RandomState(4321)
    held = [gs for name in ("unseen-games", "unseen-maps") for gs in eval_games.get(name, [])]
    val_batches = ([collate_seq(draw_seq(held, args.seqs, args.seq_len, vrng), dev) if args.seq_len else
                    collate_samples(draw_batch(held, args.frames, args.hor, vrng), dev) for _ in range(args.val_batches)]
                   if held else [])

    @torch.no_grad()
    def validate():
        for m in modules.values():
            m.eval()
        rec, kl = [], []
        for vb in val_batches:
            out, mask, lt = forward_seq(vb) if args.seq_len else forward(vb)
            rec.append(lt["lat"] if latent else sum(probes.losses(out, vb["tgt"], mask).values()))
            kl.append(lt["kl"])
        for m in modules.values():
            m.train()
        return float(torch.stack(rec).mean()), float(torch.stack(kl).mean())

    batches = Prefetch(make_batch, depth=2)
    for step in range(start, args.steps):
        t_wait = time.time()
        b, t_build = batches.get()
        t_data = time.time() - t_wait  # time the GPU loop waited for data (building happens alongside)
        prof = None
        if args.profile and step == start + args.profile:  # one profiled step: the stages' GPU time
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
        out, mask, lt = forward_seq(b) if args.seq_len else forward(b)
        L = out if grid else probes.losses(out, b["tgt"], mask)
        beta_t = args.beta * min(1.0, step / max(1, args.beta_warmup))
        kl_coef = min(1.0, step / max(1, args.beta_warmup)) if cat_z else beta_t  # categorical: weights are inside kl_charged
        loss = (args.w_ent if grid else 1.0) * sum(L.values()) + kl_coef * lt["kl_charged"]
        if grid:
            loss = loss + args.w_gram * lt["gram"] + args.w_halone * lt["hal"]
            if args.seq_len:  # the open-loop rollout, ramped in
                loss = loss + args.w_ol * min(1.0, step / max(1, args.ol_ramp)) * lt["ol"]
        elif latent:
            loss = loss + sum(probes_true.losses(lt["out_true"], b["tgt"], lt["here_true"]).values())  # trains only probes_true
            loss = loss + (args.w_lat * lt["lat"] + args.w_var * lt["var"] + args.w_cov * lt["cov"] + args.w_dec * lt["dec"]
                           + args.w_gvar * lt["gvar"] + args.w_inv * lt["inv"] + args.w_dinv * lt["dinv"] + args.w_es * lt["es"])
        opt.zero_grad(set_to_none=True)
        with RF("backward"):
            loss.backward()
        if prof is not None:
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30), flush=True)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if latent and enc_ema is not None:
            ld.ema_update(enc_ema, enc, args.ema_tau)
        # logged values stay on the GPU until the log line: reading one waits for the GPU to finish
        # everything queued, and a dozen such waits per step left it idle at every step boundary
        hist["loss"].append(loss.detach())
        hist["recon"].append((lt["lat"] if latent else sum(L.values())).detach())  # the train-vs-validation loss
        if latent:
            for k_ in ("lat", "var", "cov", "dec", "gvar", "inv", "dinv", "chg", "ratio", "es") + (("gram", "hal", "pairs") if grid else ()) + (("ol",) if args.seq_len else ()):
                hist[k_].append(lt[k_].detach().float())
        hist["kl"].append(lt["kl"].detach())
        hist["gn"].append(gn.detach())
        hist["data_s"].append(t_data)
        hist["build_s"].append(t_build)
        for k_, v_ in L.items():
            hist[k_].append(v_.detach())
        if (step + 1) % 25 == 0:
            avg = {k_: float(torch.stack(v_[-25:]).float().mean()) if torch.is_tensor(v_[-1]) else float(np.mean(v_[-25:]))
                   for k_, v_ in hist.items()}
            for v_ in hist.values():  # keep only what the next average needs
                del v_[:-25]
            dt = time.time() - t_last
            t_last = time.time()
            print(f"step {step + 1} loss {avg['loss']:.3f} | "
                  + (" ".join(f"{k_} {avg[k_]:.3f}" for k_ in L) + (f" gram {avg['gram']:.4f} h-alone {avg['hal']:.4f} pairs {avg['pairs']:.0f} " if grid else "")
                     + (f"open-loop {avg['ol']:.4f} " if args.seq_len else "") if grid else
                     f"exit {avg['exit']:.3f} dx {avg['dx']:.3f} dy {avg['dy']:.3f} "
                     f"hp {avg['hp']:.3f} sh {avg['sh']:.3f} en {avg['en']:.3f} build {avg['build']:.3f} order {avg['order']:.3f} ")
                  + (f"| latent {avg['lat']:.4f} var {avg['var']:.3f} cov {avg['cov']:.3f} dec {avg['dec']:.3f} gvar {avg['gvar']:.3f} "
                     f"inv {avg['inv']:.2f} dinv {avg['dinv']:.3f} chg {avg['chg']:.4f} ratio {avg['ratio']:.3f} es {avg['es']:.3f} " if latent else "")
                  + f"| KL {avg['kl']:.2f} (beta {beta_t:.3f}) | grad {avg['gn']:.2f} | {dt / 25:.2f}s/step (waited {avg['data_s']:.2f}s, "
                  f"built {avg['build_s']:.2f}s alongside) "
                  f"| mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
            for k_, v_ in avg.items():
                tb.add_scalar(f"train/{k_}", v_, step + 1)
            tb.add_scalar("train/lr", sched.get_last_lr()[0], step + 1)
            tb.add_scalar("train/beta", beta_t, step + 1)
            tb.add_scalar("perf/s_per_step", dt / 25, step + 1)
            tb.add_scalar("perf/mem_gb", torch.cuda.max_memory_allocated() / 2**30, step + 1)
            with open(os.path.join(run, "log.jsonl"), "a") as f:
                f.write(json.dumps(dict(step=step + 1, **{k_: float(v_) for k_, v_ in avg.items()})) + "\n")
        if val_batches and (step + 1) % args.val_every == 0:
            v_rec, v_kl = validate()
            t_rec = float(torch.stack(hist["recon"][-25:]).float().mean())
            t_kl = float(torch.stack(hist["kl"][-25:]).float().mean())
            tb.add_scalar("loss/train", t_rec, step + 1)
            tb.add_scalar("loss/val", v_rec, step + 1)
            tb.add_scalar("kl/train", t_kl, step + 1)
            tb.add_scalar("kl/val", v_kl, step + 1)
            print(f"[val step {step + 1}] loss train {t_rec:.3f} val {v_rec:.3f} | KL train {t_kl:.3f} val {v_kl:.3f}", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            evaluate(step + 1)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            save(step + 1, "latest")
    tb.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
