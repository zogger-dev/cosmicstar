"""Window world model: block-causal next-step prediction over consecutive frames.

A sample is a window of T consecutive frames of one game (--window_s seconds; every
--frame_stride-th extracted frame) with the actions in it. The context encoder encodes every
window frame into slots (CLS, players, grid), the action encoder summarizes each frame
interval per player, and the block-causal predictor reads the interleaved sequence
[S0 A0][S1 A1]... and predicts, at every frame k, the target encoder's slots of the frame
h seconds later, for each horizon in --horizons_s. All prefixes train in one pass.

Horizons may reach beyond the window: the context is the window, but a prediction's target
can be any later frame of the game. Frames past the window are encoded by the target encoder
only (no gradient, not fed to the predictor), so a 10 s window can still be asked about 30 s
ahead at little cost. Beyond-window targets are taken every --long_sub positions.

Batches draw each window from a different game, from a rotating pool of games held in memory
(the dataset doesn't fit: ~180 MB per game); gradient accumulation adds more games per update.
Held-out sets come from the fixed split (scripts/make_splits.py): unseen games, unseen maps.

Player views (--pov_dir): the context is one player's view of the game (fog, snapshots,
allies' private details hidden, only their own actions, not even their teammate's), and the
targets are the target encoder's slots of the *observer's* frames (everything), with roles
(me / ally / enemy) relative to that player. Horizon 0 then asks for the full present: what
the player can't see. The probes' "now" reading is the player's own view encoded, i.e.
copying what they see.

    python scripts/train_window.py --pov_dir data/worldmodel/pov --splits data/worldmodel/pov/sample.json --residual --window_s 30 --horizons_s 0,0.71,2.86,10,30

Latent-action mode (--latent_action_s D, e.g. 5): the world model is asked a different
question. At every frame the predictor gets a latent action per player for the D seconds
AHEAD of it and predicts the frame D seconds later (one step, every position). The latent
action comes from two sources that must agree (InfoNCE across the batch): the command encoder's
summary of that interval's commands, and an inverse model that reads the frame and the frame
D later (the state change). Either source can drive the predictor in training (--z_source);
evaluation reports the agreement, and whether the predictor uses the action: real commands vs
another window's commands (counterfactual recall@1) vs none, and the inverse model as the
oracle. Rationale: with actions given only up to the present, their effects are already in
the frames and the predictor learns to ignore them (scripts/action_check.py).

Anti-collapse: each window's mean CLS must vary across games; the spread is estimated over
the current windows plus a rolling memory of recent ones (detached), so a batch that happens
to hold similar-looking windows doesn't yank the encoder.

Instruments (evaluation only, nothing here trains the model; see worldmodel/probes.py):
    probes      ground-truth quantities read off the slots: skill vs copying the present,
                and the best skill possible (reading the true future encoding)
    retrieval   is the prediction closer to the true future than to the present / top-1 among
                the window's frames
    collapse    share of the target encoder's variance that changes over time within a game

    python scripts/train_window.py --residual --window_s 30 --horizons_s 0.18,0.71,2.86,10,30

Checkpoints: model_latest.pt every --save_minutes (default 20, so at most that much work is
lost), milestones model_step<N>.pt every --save_every steps. To pause without losing anything,
create an empty file PAUSE in the run folder: the run saves and exits after the current step.
Resume (same run folder, same settings, logs appended):

    python scripts/train_window.py --resume data/worldmodel/runs/<run>

Checkpoints hold the full training state (weights, optimizers, schedule, random generators,
anti-collapse memory, which games are in the pool). Checkpoints from before resume existed
hold weights only; those resume warm: weights restored, schedules jumped to the step,
optimizer statistics rebuilt from scratch.
"""

import argparse
import collections
import copy
import csv
import json
import math
import os
import time

# a growing, fragmented CUDA cache pushed a run past the card's memory, and on Windows that
# means paging GPU memory to system RAM (steps ~40% slower); expandable segments avoid that
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cosmicstar.worldmodel import vocab as vocab_mod
from cosmicstar.worldmodel.action_encoder import ActionEncoder
from cosmicstar.worldmodel.action_tokens import ActionVocab, GameActions, collate_actions
from cosmicstar.worldmodel.encoder import StateEncoder
from cosmicstar.worldmodel.inverse_model import InverseModel
from cosmicstar.worldmodel.predictor import slots_of
from cosmicstar.worldmodel.probes import (KINDS, Probes, collate_targets, probe_errors, retrieval_among, skill,
                                          temporal_share)
from cosmicstar.worldmodel.tokenize import GameFrames, collate, probe_targets
from cosmicstar.worldmodel.window_predictor import WindowPredictor

LPS = 22.4


class GamePool:
    """Holds `size` games in memory; rotate(n) swaps n of them for games not loaded."""

    def __init__(self, paths, size, load, rng, names=None):
        self.paths, self.load, self.rng = list(paths), load, rng
        if names:  # resuming: the same games as before
            where = {os.path.basename(q): i for i, q in enumerate(self.paths)}
            idx = [where[n] for n in names if n in where]
        else:
            idx = list(rng.permutation(len(self.paths))[:size])
        self.games = [(i, load(self.paths[i])) for i in idx]

    def names(self):
        return [os.path.basename(self.paths[i]) for i, _ in self.games]

    def rotate(self, n):
        loaded = {i for i, _ in self.games}
        free = [i for i in range(len(self.paths)) if i not in loaded]
        for _ in range(min(n, len(free))):
            slot = self.rng.randint(len(self.games))
            new = free.pop(self.rng.randint(len(free)))
            self.games[slot] = (new, self.load(self.paths[new]))

    def draw(self, n):
        return [self.games[i][1] for i in self.rng.choice(len(self.games), n, replace=False)]


class WindowSpec:
    """Window length, frame stride and horizons, in frames of the extracted data."""

    def __init__(self, window_s, stride, horizons_s, long_sub, step_loops, ahead_s=0.0):
        self.stride, self.long_sub = stride, long_sub
        self.frame_s = step_loops * stride / LPS                       # seconds between window frames
        self.T = int(round(window_s / self.frame_s)) + 1
        self.horizons = sorted({0 if h == 0 else max(1, int(round(h / self.frame_s))) for h in horizons_s})  # in window frames
        self.ahead = int(round(ahead_s / self.frame_s)) if ahead_s else 0  # latent-action step, in frames
        assert self.ahead < self.T, "the latent-action step must be shorter than the window"

    def seconds(self, h):
        return round(h * self.frame_s, 2)


def make_window(game, spec, rng, augment=True):
    """One sample: T window frames (context), the extra frames past the window that long
    horizons need (targets only), and for each horizon the target index of every position
    (-1 = no target: past the end of the game, or a beyond-window position not sampled)."""
    g, ga, views = game
    T, st = spec.T, spec.stride
    span = st * (T - 1)
    viewer = None
    if views:  # a player's view: theirs is the context, the observer's frames are the targets
        viewer = list(views)[rng.randint(len(views))]
        gv, ga = views[viewer]
    n = len(g) if viewer is None else min(len(g), len(gv))
    if n <= span:
        return None
    s = rng.randint(0, n - span)
    om = g.owner_map("observer" if viewer is None else "player", me_slot=viewer, augment=augment, rng=rng)
    win = [s + st * k for k in range(T)]
    extra, tgt = {}, {}
    for h in spec.horizons:
        idx = np.full(T, -1, np.int64)
        for k in range(T):
            if k + h < T:
                idx[k] = k + h
            elif k % spec.long_sub == 0:
                f = s + st * (k + h)
                if f < n:
                    idx[k] = T + extra.setdefault(f, len(extra))
        tgt[h] = idx
    tframes = win + sorted(extra, key=extra.get)
    loops = g.loops[win]
    # latent-action mode: each frame's actions are the interval after it, so the action window's
    # segment bounds run `ahead` frames past the window (evenly spaced like the frames)
    bounds = loops[0] + (g.step * st) * np.arange(T + spec.ahead) if spec.ahead else loops
    return dict(frames=[(g if viewer is None else gv).frame(f, om) for f in win],
                # observer mode: the window frames double as targets; player views: the observer's
                wframes=None if viewer is None else [g.frame(f, om) for f in win],
                tframes=[g.frame(f, om) for f in tframes[T:]],
                probe=[probe_targets(g, f) for f in tframes],
                tgt=tgt, times=((loops - loops[0]) / LPS).astype(np.float32), viewer=viewer,
                act=None if ga is None else ga.window(loops[0], int(bounds[-1]) + g.step * st, bounds, loops[0], om, only=viewer))


def to(b, dev):
    return {k: x.to(dev, non_blocking=True) for k, x in b.items()}


def kind_slices(P):
    return dict(cls=slice(0, 1), ply=slice(1, 1 + P), grid=slice(1 + P, None))


def slot_loss(pred, target, mask, P):
    """Smooth L1 per slot, averaged within each kind (CLS / players / grid), then across kinds.
    pred, target [..., S, d]; mask [..., S]."""
    err = F.smooth_l1_loss(pred.float(), target, reduction="none").mean(-1)
    m = mask.float()
    parts = [(err[..., s] * m[..., s]).sum() / m[..., s].sum().clamp(min=1) for s in kind_slices(P).values()]
    return sum(parts) / len(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/worldmodel/obs")
    ap.add_argument("--actions_dir", default="data/worldmodel/actions")
    ap.add_argument("--pov_dir", default="", help="player views (<game>_p<slot>.npz): context = a player's view, targets = observer")
    ap.add_argument("--no_actions", action="store_true", help="ablation: no action stream")
    ap.add_argument("--latent_action_s", type=float, default=0.0,
                    help="latent-action mode: predict this many seconds ahead, given the interval's actions as a latent "
                         "action per player (see the module docstring); 0 = the multi-horizon mode")
    ap.add_argument("--d_z", type=int, default=64, help="latent action size")
    ap.add_argument("--nce_weight", type=float, default=1.0, help="weight of the agreement (InfoNCE) loss")
    ap.add_argument("--nce_temp", type=float, default=0.1)
    ap.add_argument("--z_source", default="mix", help="what drives the predictor in training: cmd, inv, or mix (coin per pass)")
    ap.add_argument("--window_s", type=float, default=10.0, help="context window length")
    ap.add_argument("--frame_stride", type=int, default=1, help="use every n-th extracted frame")
    ap.add_argument("--horizons_s", default="0.18,0.71,2.86,10,30",
                    help="seconds ahead predicted at every position (may exceed the window)")
    ap.add_argument("--long_sub", type=int, default=8, help="beyond-window targets every n positions")
    ap.add_argument("--windows", type=int, default=4, help="windows per forward pass (one game each)")
    ap.add_argument("--accum", type=int, default=2, help="forward passes per optimizer step")
    ap.add_argument("--pool", type=int, default=60, help="games held in memory")
    ap.add_argument("--rotate_every", type=int, default=50, help="steps between swapping games in the pool")
    ap.add_argument("--rotate", type=int, default=4, help="games swapped each time")
    ap.add_argument("--splits", default="data/worldmodel/splits.json", help="fixed split (scripts/make_splits.py)")
    ap.add_argument("--val_games", type=int, default=6, help="held-out games per evaluation set")
    ap.add_argument("--val_windows", type=int, default=8, help="windows per evaluation set")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--var_weight", type=float, default=0.1)
    ap.add_argument("--var_memory", type=int, default=256, help="recent windows kept for the CLS spread estimate")
    ap.add_argument("--depth", type=int, default=4, help="predictor layers")
    ap.add_argument("--residual", action="store_true", help="predict a change on top of the present (copy by default)")
    ap.add_argument("--no_ckpt", action="store_true", help="no gradient checkpointing (more memory, faster)")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--save_every", type=int, default=1000, help="milestone checkpoints every n steps (kept)")
    ap.add_argument("--save_minutes", type=float, default=20, help="overwrite model_latest.pt this often (bounds work lost)")
    ap.add_argument("--max_games", type=int, default=0)
    ap.add_argument("--out", default="data/worldmodel/runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="", help="checkpoint to continue from (same run folder and settings)")
    args = ap.parse_args()

    ck = None
    if args.resume:
        if os.path.isdir(args.resume):  # a run folder: its latest checkpoint
            found = [os.path.join(args.resume, "model_latest.pt")] if os.path.exists(
                os.path.join(args.resume, "model_latest.pt")) else sorted(
                (q for q in (os.path.join(args.resume, f) for f in os.listdir(args.resume))
                 if os.path.basename(q).startswith("model_step")),
                key=lambda q: int(os.path.basename(q)[len("model_step"):-3]))[-1:]
            if not found:
                raise SystemExit(f"no checkpoint in {args.resume}")
            args.resume = found[0]
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        args = argparse.Namespace(**{**vars(ap.parse_args([])), **ck["args"], "resume": args.resume})
    la = args.latent_action_s > 0
    if la:  # one step ahead, at every position, given that interval's actions
        assert not args.no_actions, "the latent-action mode needs the action stream"
        args.horizons_s, args.long_sub = str(args.latent_action_s), 1
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    dev = "cuda"
    split = json.load(open(args.splits))

    def views_of(f):  # slot -> player-view file
        if not args.pov_dir:
            return {}
        out = {}
        for q in os.listdir(args.pov_dir):
            if q.startswith(f[:-4] + "_p") and q.endswith(".npz") and not q.endswith(".tmp.npz"):
                out[int(q[len(f) - 2:-4])] = os.path.join(args.pov_dir, q)
        return out

    def available(names):  # extracted, with parsed actions (and player views, if used)
        return [os.path.join(args.data, f) for f in sorted(names)
                if os.path.exists(os.path.join(args.data, f)) and os.path.exists(os.path.join(args.actions_dir, f))
                and (not args.pov_dir or views_of(f))]

    train_paths = available(split["train"])
    if args.max_games:
        train_paths = train_paths[:args.max_games]
    if ck is not None and ck.get("train_files"):  # resuming: the original training set, not today's
        keep = set(ck["train_files"])
        train_paths = [q for q in train_paths if os.path.basename(q) in keep]
    # seen-games (trained on) vs unseen: a big gap means memorizing (more data helps); the same
    # plateau on both means the model or objective is the limit, not the data
    eval_paths = {"seen-games": train_paths[:args.val_games],
                  "unseen-games": available(split["held_out_games"])[:args.val_games],
                  "unseen-maps": available(split["held_out_map_games"])[:args.val_games]}
    run = os.path.dirname(os.path.abspath(args.resume)) if args.resume else \
        os.path.join(args.out, time.strftime("window_%Y%m%d_%H%M%S"))
    if args.resume:  # the run's own vocabulary (rebuilding could differ once more games exist)
        voc = json.load(open(os.path.join(run, "vocab.json")))
    else:
        voc = vocab_mod.build(train_paths[:60])  # unit / ability / upgrade vocab from training games only
    v = vocab_mod.Vocab(voc)
    av = None if args.no_actions else ActionVocab()

    def load(p):
        """(observer frames, their actions, None) or, with player views, (observer frames, None,
        {slot: (that player's view, actions resolved against it)})."""
        g = GameFrames(p, v)
        acts = os.path.join(args.actions_dir, os.path.basename(p))
        if not args.pov_dir:
            return g, (None if av is None else GameActions(acts, g, av)), None
        views = {}
        for slot, q in sorted(views_of(os.path.basename(p)).items()):
            gv = GameFrames(q, v)
            views[slot] = (gv, None if av is None else GameActions(acts, gv, av))
        return g, None, views

    eval_games = {name: [load(p) for p in ps] for name, ps in eval_paths.items() if ps}
    pool = GamePool(train_paths, min(args.pool, len(train_paths)), load, rng,
                    names=ck.get("pool") if ck is not None else None)
    loaded = [x for _, x in pool.games] + [x for gs in eval_games.values() for x in gs]
    step_loops = collections.Counter(x[0].step for x in loaded).most_common(1)[0][0]
    assert all(x[0].step == step_loops for x in loaded), "mixed frame rates"
    spec = WindowSpec(args.window_s, args.frame_stride, [float(x) for x in args.horizons_s.split(",")],
                      args.long_sub, step_loops, ahead_s=args.latent_action_s)
    T, horizons = spec.T, spec.horizons

    os.makedirs(run, exist_ok=True)
    if not args.resume:
        json.dump(dict(vars(args), T=T, step_loops=step_loops, horizons_frames=horizons, n_train=len(train_paths),
                       eval={k: [os.path.basename(p) for p in ps] for k, ps in eval_paths.items()}),
                  open(os.path.join(run, "config.json"), "w"), indent=1)
        json.dump(voc, open(os.path.join(run, "vocab.json"), "w"))
    print(f"{len(train_paths)} train games (pool {len(pool.games)}) | eval: "
          + ", ".join(f"{k} {len(v_)} games" for k, v_ in eval_games.items())
          + f" | window {args.window_s}s = {T} frames, {spec.frame_s:.2f}s apart | horizons "
          + ", ".join(f"{spec.seconds(h)}s" for h in horizons)
          + f" | actions {'off' if av is None else 'on'}"
          + (f" | latent action: {args.latent_action_s}s steps ({spec.ahead} frames), d_z {args.d_z}, driven by {args.z_source}" if la else "")
          + (f" | player views ({sum(len(x[2]) for x in loaded)} loaded) -> observer targets" if args.pov_dir else "")
          + f" | {run}", flush=True)

    enc = StateEncoder(v).to(dev)
    enc.grad_ckpt = not args.no_ckpt
    tenc = copy.deepcopy(enc).requires_grad_(False)
    pred = WindowPredictor(depth=args.depth, horizons=horizons, grad_ckpt=not args.no_ckpt, residual=args.residual).to(dev)
    aenc = None if av is None else ActionEncoder(enc, av).to(dev)
    inv = InverseModel(enc.d, args.d_z).to(dev) if la else None       # latent action from the state change
    zproj = nn.Linear(enc.d, args.d_z).to(dev) if la else None          # ... and from the commands' summary
    zup = nn.Linear(args.d_z, enc.d).to(dev) if la else None            # latent action -> predictor input
    probes = Probes().to(dev)
    params = list(enc.parameters()) + list(pred.parameters()) + (list(aenc.parameters()) if aenc else [])
    if la:
        params += list(inv.parameters()) + list(zproj.parameters()) + list(zup.parameters())
    n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    print(f"encoder {n(enc):.2f}M, predictor {n(pred):.2f}M" + (f", action encoder {n(aenc):.2f}M" if aenc else ""), flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    popt = torch.optim.AdamW(probes.parameters(), lr=1e-3)
    warm = 200
    lr_at = lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))
    start = 0
    if ck is not None:
        enc.load_state_dict(ck["encoder"], strict=False)  # older checkpoints lack the order path
        tenc.load_state_dict(ck["target"], strict=False)
        enc.use_orders = tenc.use_orders = any(k.startswith("ord_in") for k in ck["encoder"])
        pred.load_state_dict(ck["predictor"])
        if aenc is not None:
            aenc.load_state_dict(ck["actions"])
        if la and ck.get("inverse"):
            inv.load_state_dict(ck["inverse"])
            zproj.load_state_dict(ck["zproj"])
            zup.load_state_dict(ck["zup"])
        try:
            probes.load_state_dict(ck["probes"])
        except RuntimeError:
            print("(saved probes read different quantities; probes restart)", flush=True)
        start = ck.get("step") or int(os.path.basename(args.resume).split("step")[-1].split(".")[0])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
            popt.load_state_dict(ck["popt"])
        else:
            print(f"warm resume from step {start}: weights restored, optimizer statistics rebuilt", flush=True)
    for g_ in opt.param_groups:
        g_.setdefault("initial_lr", args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at, last_epoch=start - 1)

    @torch.no_grad()
    def encode_targets(frames, S, chunk=64):
        """Target encoder over extra frames (no gradient), in chunks so evaluation doesn't spike
        memory: normalized slots [N, S, d] and mask [N, S], player slots padded to S."""
        outs, masks = [], []
        G = pred.tags.cells.shape[0]
        for i in range(0, len(frames), chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s, m = slots_of(tenc(to(collate(frames[i:i + chunk]), dev)))
            s = F.layer_norm(s.float(), s.shape[-1:])
            pad = S - s.shape[1]  # this chunk may have fewer player slots than the batch
            if pad:
                P_e = s.shape[1] - 1 - G
                s = torch.cat([s[:, :1 + P_e], s.new_zeros(len(s), pad, s.shape[2]), s[:, 1 + P_e:]], 1)
                m = torch.cat([m[:, :1 + P_e], m.new_zeros(len(m), pad), m[:, 1 + P_e:]], 1)
            outs.append(s)
            masks.append(m)
        return torch.cat(outs), torch.cat(masks)

    def encode(wins, encoder):
        """Everything about a batch of windows that doesn't depend on which actions drive the
        predictor: context slots [B, T, S, d] with mask, the present (target space), the target
        table per horizon with validity and flat index, and in latent-action mode the latent
        action of every (frame, player) from the commands (z_cmd [B, T, P, d_z]) and from the
        state change (z_inv [B, T - ahead, P, d_z])."""
        B = len(wins)
        fb = to(collate([f for w in wins for f in w["frames"]]), dev)
        times = torch.from_numpy(np.stack([w["times"] for w in wins])).to(dev)
        st = dict(B=B, times=times)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = encoder(fb)
            slots, mask = slots_of(out)
            S = slots.shape[1]
            slots, mask = slots.view(B, T, S, -1), mask.view(B, T, S)
            with torch.no_grad():
                tw, _ = slots_of(tenc(fb))  # the window frames as the target encoder sees them
            present = F.layer_norm(tw.float(), tw.shape[-1:]).view(B, T, S, -1)
            if aenc is not None:
                ab = to(collate_actions([w["act"] for w in wins]), dev)
                st["summaries"], st["s_mask"], _ = aenc(ab, ahead=spec.ahead or None)
                if la:
                    A = spec.ahead
                    st["z_cmd"] = zproj(st["summaries"])
                    Pz = st["z_cmd"].shape[2]
                    st["z_inv"] = inv(slots[:, :T - A].flatten(0, 1), slots[:, A:].flatten(0, 1),
                                      mask[:, :T - A].flatten(0, 1), mask[:, A:].flatten(0, 1), Pz).view(B, T - A, Pz, -1)
        # all target frames in one flat table: window frames, then each window's extra frames
        extra = [f for w in wins for f in w["tframes"]]
        if wins[0]["wframes"] is None:  # observer: the window frames are their own targets
            flat, fmask = present.flatten(0, 1), mask.flatten(0, 1)
        else:  # player views: targets are the observer's frames
            flat, fmask = encode_targets([f for w in wins for f in w["wframes"]], S)
        if extra:
            eo, em = encode_targets(extra, S)
            flat, fmask = torch.cat([flat, eo]), torch.cat([fmask, em])
        offsets, o = [], B * T
        for b, w in enumerate(wins):  # window frame k of window b -> b*T + k; extra j -> offset_b + j
            offsets.append(o)
            o += len(w["tframes"])
        targets, valid, index = {}, {}, {}
        for h in horizons:
            idx = np.stack([np.where(w["tgt"][h] < 0, -1,
                                     np.where(w["tgt"][h] < T, b * T + w["tgt"][h], offsets[b] + w["tgt"][h] - T))
                            for b, w in enumerate(wins)])
            it = torch.from_numpy(idx).to(dev)
            ok = it >= 0
            targets[h] = flat[it.clamp(min=0)]
            valid[h] = fmask[it.clamp(min=0)] & ok[..., None] & mask
            index[h] = idx
        st.update(slots=slots, mask=mask, present=present, out=out, P=out["players"].shape[1],
                  flat=flat, fmask=fmask, targets=targets, valid=valid, index=index, offsets=offsets)
        return st

    def run_pred(st, use_inv=False, borrow=0, none=False):
        """The predictor over encoded windows. Latent-action mode: driven by the commands' latent
        actions (default), by the state change's (use_inv: the oracle), by another window's
        commands (borrow = roll along the batch; same seats) or by nothing (none)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            act = {}
            if aenc is not None:
                s_mask = torch.zeros_like(st["s_mask"]) if none else st["s_mask"]
                if la:
                    z = st["z_cmd"]
                    if borrow:
                        z = torch.roll(z, borrow, 0)
                    if use_inv:
                        z = torch.cat([st["z_inv"], z[:, T - spec.ahead:]], 1)
                    act = dict(act=zup(z), act_mask=s_mask)
                else:
                    act = dict(act=st["summaries"], act_mask=s_mask)
            return pred(st["slots"], st["mask"], st["times"], base=st["present"], **act)  # base is only used with --residual

    def forward(wins, encoder, use_inv=False):
        """Predictions {h: [B, T, S, d]} and the encoded windows (see encode)."""
        st = encode(wins, encoder)
        return run_pred(st, use_inv=use_inv), st

    def window_loss(preds, targets, valid, P):
        losses = {h: slot_loss(preds[h], targets[h], valid[h], P) for h in horizons}
        return sum(losses.values()) / len(losses), losses

    def position_loss(pred_h, target, valid, P):
        """The training loss at every position, [B, T]."""
        err = F.smooth_l1_loss(pred_h.float(), target, reduction="none").mean(-1)
        m = valid.float()
        parts = [(err[..., s] * m[..., s]).sum(-1) / m[..., s].sum(-1).clamp(min=1) for s in kind_slices(P).values()]
        return sum(parts) / len(parts)

    def agreement_loss(st):
        """InfoNCE between each (frame, player)'s latent action from the state change and from
        the commands, over the batch; the same window's overlapping intervals aren't negatives.
        Returns the loss and the top-1 agreement."""
        zi = st["z_inv"]
        Bz, Tz, Pz, _ = zi.shape
        zc, m = st["z_cmd"][:, :Tz], st["s_mask"][:, :Tz]
        sel = m.reshape(-1)
        zi = F.normalize(zi.float().reshape(-1, zi.shape[-1])[sel], dim=-1)
        zc = F.normalize(zc.float().reshape(-1, zc.shape[-1])[sel], dim=-1)
        win = torch.arange(Bz, device=dev).repeat_interleave(Tz * Pz)[sel]
        frame = torch.arange(Tz, device=dev).repeat_interleave(Pz).repeat(Bz)[sel]
        sim = zi @ zc.t() / args.nce_temp
        near = (win[:, None] == win[None, :]) & ((frame[:, None] - frame[None, :]).abs() < spec.ahead)
        near.fill_diagonal_(False)
        sim = sim.masked_fill(near, float("-inf"))
        tgt = torch.arange(len(sim), device=dev)
        loss = (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.t(), tgt)) / 2
        return loss, (sim.argmax(1) == tgt).float().mean()

    # fixed evaluation windows per held-out set
    vrng = np.random.RandomState(123)
    eval_wins = {}
    for name, gs in eval_games.items():
        ws, i = [], 0
        while len(ws) < args.val_windows and i < 50 * args.val_windows:
            w = make_window(gs[i % len(gs)], spec, vrng, augment=False)
            i += 1
            if w is not None:
                ws.append(w)
        eval_wins[name] = ws

    @torch.no_grad()
    def evaluate(step, name):
        enc.eval(); pred.eval(); probes.eval()
        if aenc is not None:
            aenc.eval()
        acc = collections.defaultdict(list)
        val_wins = eval_wins[name]
        for i in range(0, len(val_wins), args.windows):
            wins = val_wins[i:i + args.windows]
            preds, st = forward(wins, enc)
            targets, valid, present, mask, P = st["targets"], st["valid"], st["present"], st["mask"], st["P"]
            flat, fmask, index, offsets = st["flat"], st["fmask"], st["index"], st["offsets"]
            B = len(wins)
            if la:  # the latent action: agreement, and whether the predictor uses it
                _, agree = agreement_loss(st)
                acc[("la", "agree")].append(agree.item())
                h = spec.ahead
                ok = torch.from_numpy(index[h] >= 0).to(dev)
                m_now = valid[h] & ok[..., None]
                real = position_loss(preds[h], targets[h], m_now, P)
                none = position_loss(run_pred(st, none=True)[h], targets[h], m_now, P)
                oracle = position_loss(run_pred(st, use_inv=True)[h], targets[h], m_now, P)
                for k_, x in (("real", real), ("none", none), ("oracle", oracle)):
                    acc[("la", k_)].append(x[ok].mean().item())
                if B > 1:
                    fakes = torch.stack([position_loss(run_pred(st, borrow=s)[h], targets[h], m_now, P) for s in range(1, B)])
                    rank = 1 + (fakes <= real[None]).sum(0).float()  # ties count against the real action
                    acc[("la", "recall")].append((rank[ok] == 1).float().mean().item())
                    acc[("la", "mrr")].append((1 / rank[ok]).mean().item())
                    acc[("la", "borrowed")].append(fakes.mean(0)[ok].mean().item())
                    acc[("la", "chance")].append(1 / B)
            truth_all = collate_targets([t for w in wins for t in w["probe"][:T]] +
                                        [t for w in wins for t in w["probe"][T:]], P, dev)
            for h in horizons:
                ok = torch.from_numpy(index[h] >= 0).to(dev)             # [B, T] positions with a target
                if not ok.any():
                    continue
                m_now = valid[h] & ok[..., None]
                acc[("latent", h)].append(slot_loss(preds[h][ok], targets[h][ok], m_now[ok], P).item())
                acc[("nochange", h)].append(slot_loss(present[ok], targets[h][ok], m_now[ok], P).item())
                it = torch.from_numpy(index[h]).to(dev)[ok]
                fut = {k: x[it] for k, x in truth_all.items()}
                for reading, src in (("now", present[ok]), ("pred", preds[h][ok]), ("ceil", targets[h][ok])):
                    e = probe_errors(probes(src, P), fut)
                    for k in KINDS:
                        acc[(reading, h, k)].append(e[k].item())
                for b in range(B):  # retrieval among this window's target frames
                    kk = np.where(index[h][b] >= 0)[0]
                    if not len(kk):
                        continue
                    lo, hi = b * T, (b + 1) * T
                    ex = list(range(offsets[b], offsets[b] + len(wins[b]["tframes"])))
                    cand = torch.cat([flat[lo:hi], flat[ex]]) if ex else flat[lo:hi]
                    cmask = torch.cat([fmask[lo:hi], fmask[ex]]) if ex else fmask[lo:hi]
                    tg = index[h][b][kk]
                    correct = np.where(tg < offsets[0], tg - lo, T + tg - offsets[b])
                    r, c, beats = retrieval_among(preds[h][b, kk], cand, cmask,
                                                  torch.from_numpy(correct).to(dev), torch.from_numpy(kk).to(dev))
                    acc[("retrieval", h)].append(r)
                    acc[("copy", h)].append(c)
                    acc[("beats", h)].append(beats)
        share = temporal_share(tenc, [x[0] for x in eval_games[name]], dev)
        enc.train(); pred.train(); probes.train()
        if aenc is not None:
            aenc.train()
        mean = {k: float(np.mean(v)) for k, v in acc.items()}
        rows = []
        for h in horizons:
            if ("latent", h) not in mean:
                continue
            sk = {k: skill(mean[("now", h, k)], mean[("pred", h, k)], mean[("ceil", h, k)]) for k in KINDS}
            rows.append(dict(set=name, step=step, horizon_s=spec.seconds(h), latent=mean[("latent", h)],
                             nochange=mean[("nochange", h)], retrieval=mean.get(("retrieval", h), float("nan")),
                             copy=mean.get(("copy", h), float("nan")), beats_present=mean.get(("beats", h), float("nan")),
                             **{f"skill_{k}": sk[k][0] for k in KINDS}, **{f"best_{k}": sk[k][1] for k in KINDS},
                             **{f"closed_{k}": sk[k][2] for k in KINDS},
                             **{f"{nm}_{k}": mean[(nm, h, k)] for nm in ("now", "pred", "ceil") for k in KINDS},
                             share_cls=share["cls"], share_grid=share["grid"]))
            r = rows[-1]
            print(f"[{name} {step}] +{r['horizon_s']:.2f}s latent {r['latent']:.4f} (no-change {r['nochange']:.4f}) | "
                  "probe skill vs copying (best possible) " + " ".join(f"{k} {sk[k][0]:+.0%}({sk[k][1]:+.0%})" for k in KINDS)
                  + f" | closer to future than present {r['beats_present']:.0%} | top-1 {r['retrieval']:.0%} "
                  f"(picked present {r['copy']:.0%})", flush=True)
        print(f"[{name} {step}] temporal share of variance: cls {share['cls']:.0%}, grid {share['grid']:.0%}", flush=True)
        if la and ("la", "agree") in acc:
            m_la = {k[1]: float(np.mean(v)) for k, v in acc.items() if k[0] == "la"}
            g_ = lambda k: m_la.get(k, float("nan"))
            print(f"[{name} {step}] latent action: agreement top-1 {g_('agree'):.0%} | predictor given real vs another "
                  f"window's commands: recall@1 {g_('recall'):.0%} (chance {g_('chance'):.0%}) MRR {g_('mrr'):.2f} | "
                  f"loss real {g_('real'):.4f} borrowed {g_('borrowed'):.4f} none {g_('none'):.4f} oracle {g_('oracle'):.4f}", flush=True)
            for r in rows:
                r.update({f"la_{k}": v_ for k, v_ in m_la.items()})
        return rows, share

    def save(tag, step):
        a = {k: v_ for k, v_ in vars(args).items() if k != "resume"}
        torch.save(dict(encoder=enc.state_dict(), target=tenc.state_dict(), predictor=pred.state_dict(),
                        actions=aenc.state_dict() if aenc else None, probes=probes.state_dict(), args=a,
                        inverse=inv.state_dict() if la else None, zproj=zproj.state_dict() if la else None,
                        zup=zup.state_dict() if la else None,
                        T=T, horizons=horizons, step_loops=step_loops,
                        # training state, for --resume
                        step=step, opt=opt.state_dict(), popt=popt.state_dict(), np_rng=rng.get_state(),
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        cls_memory=cls_memory.cpu(), pool=pool.names(),
                        train_files=[os.path.basename(q) for q in train_paths]),
                   os.path.join(run, f"model_{tag}.tmp"))
        os.replace(os.path.join(run, f"model_{tag}.tmp"), os.path.join(run, f"model_{tag}.pt"))  # never half-written

    log_rows, share0 = [], {}
    cls_memory = torch.zeros(0, enc.d, device=dev)  # recent windows' mean CLS (detached)
    if ck is not None:
        if os.path.exists(os.path.join(run, "eval.csv")):  # the evaluation history so far
            with open(os.path.join(run, "eval.csv"), newline="") as fh:
                log_rows = [r for r in csv.DictReader(fh) if int(r["step"]) <= start]
            for r in log_rows:
                if int(r["step"]) == 0:
                    share0[r["set"]] = dict(cls=float(r["share_cls"]), grid=float(r["share_grid"]))
        if "np_rng" in ck:
            rng.set_state(ck["np_rng"])
            torch.set_rng_state(ck["torch_rng"])
            torch.cuda.set_rng_state(ck["cuda_rng"])
            cls_memory = ck["cls_memory"].to(dev)
        print(f"resuming {run} at step {start} of {args.steps}", flush=True)
    for name in eval_wins:
        if name not in share0:
            rows, share0[name] = evaluate(0, name)
            log_rows += rows
    t0 = last_save = time.time()
    pause_file = os.path.join(run, "PAUSE")
    for step in range(start + 1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        tot, var_tot, nce_tot, agree_tot, by_h = 0.0, 0.0, 0.0, 0.0, collections.defaultdict(float)
        for _ in range(args.accum):
            wins = []
            while len(wins) < args.windows:
                wins += [w for w in (make_window(gm, spec, rng) for gm in pool.draw(args.windows - len(wins))) if w is not None]
            use_inv = la and (args.z_source == "inv" or (args.z_source == "mix" and rng.rand() < 0.5))
            preds, st = forward(wins, enc, use_inv=use_inv)
            P, out = st["P"], st["out"]
            loss, losses = window_loss(preds, st["targets"], st["valid"], P)
            nce, agree = agreement_loss(st) if la else (torch.zeros((), device=dev), torch.zeros((), device=dev))
            # anti-collapse: each window's mean CLS must vary across games. The spread is taken
            # over these windows plus the recent ones in memory (gradient through these only).
            cls = out["cls"].float().view(len(wins), T, -1).mean(1)
            spread = torch.cat([cls, cls_memory])
            var = F.relu(1.0 - spread.std(0)).mean() if len(spread) > 1 else cls.sum() * 0  # one window: no spread yet
            cls_memory = torch.cat([cls.detach(), cls_memory])[:args.var_memory]
            ((loss + args.var_weight * var + args.nce_weight * nce) / args.accum).backward()
            tot += loss.item() / args.accum
            var_tot += var.item() / args.accum
            nce_tot += nce.item() / args.accum
            agree_tot += agree.item() / args.accum
            for h, l in losses.items():
                by_h[h] += l.item() / args.accum
            # probes: train on the target encoder's slots of real window frames (detached)
            truth = collate_targets([t for w in wins for t in w["probe"][:T]], P, dev)
            pl = probe_errors(probes(st["flat"][:len(wins) * T], P), truth)  # the target encoder's window frames
            popt.zero_grad(set_to_none=True)
            sum(pl.values()).backward()
            popt.step()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        m = 1 - (1 - args.ema) * 0.5 * (1 + math.cos(math.pi * step / args.steps))  # follows the step, so resumes continue it
        with torch.no_grad():
            for pt, pc in zip(tenc.parameters(), enc.parameters()):
                pt.mul_(m).add_(pc.detach(), alpha=1 - m)
        if step % args.rotate_every == 0:
            pool.rotate(args.rotate)
        if step % 25 == 0:
            sps = (time.time() - t0) / (step - start)
            print(f"step {step:5d} loss {tot:.4f} [" + " ".join(f"{spec.seconds(h)}s:{l:.4f}" for h, l in sorted(by_h.items()))
                  + f"] var {var_tot:.3f}" + (f" agree {nce_tot:.3f}/{agree_tot:.0%}" if la else "")
                  + f" lr {opt.param_groups[0]['lr']:.2e} {sps:.2f}s/step "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            for name in eval_wins:
                rows, share = evaluate(step, name)
                log_rows += rows
                for k in ("cls", "grid"):
                    if share[k] < 0.5 * share0[name][k]:
                        print(f"!! collapse alarm ({name}): {k} temporal share {share[k]:.0%} vs {share0[name][k]:.0%} at init",
                              flush=True)
            with open(os.path.join(run, "eval.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
                w.writeheader()
                w.writerows(log_rows)
        if step % args.save_every == 0:
            save(f"step{step}", step)
        if time.time() - last_save >= args.save_minutes * 60:
            save("latest", step)
            last_save = time.time()
        if os.path.exists(pause_file):  # touch <run>/PAUSE to stop cleanly; resume with --resume <run>
            save("latest", step)
            os.remove(pause_file)
            print(f"paused at step {step}; resume with --resume {run}", flush=True)
            return
    save("final", args.steps)
    print(f"saved {run}/model_final.pt", flush=True)


if __name__ == "__main__":
    main()
