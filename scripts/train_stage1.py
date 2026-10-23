"""Stage-1 world model, end to end: encoder + predictor + EMA target, on extracted games.

A sample is (game, time t, context frames t - (K-1)*stride .. t, target frame t + delta).
The context encoder encodes each context frame into slots (CLS, players, grid); the EMA
target encoder encodes the target frame; the predictor predicts the target slots from
the context slots. Loss: smooth L1 against layer-normalized targets, averaged per slot
kind (CLS, players, grid) so the 256 grid cells don't drown out the rest.

Actions (unless --no_actions): the replay's actions from the first context frame up to the
target, encoded by the action encoder into one summary per (segment, player), where the
segments are [context frame k, context frame k+1) and [t, t + delta); the last one is the
conditioning (what the players do between now and the target). --no_actions is the ablation.

Validation is on held-out games, against the "nothing changes" baseline (the target
encoder's encoding of frame t, i.e. predicting that t + delta looks like t).

    python scripts/train_stage1.py --data data/worldmodel/obs --steps 3000
"""

import argparse
import copy
import csv
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from cosmicstar.worldmodel import vocab as vocab_mod
from cosmicstar.worldmodel.action_encoder import ActionEncoder
from cosmicstar.worldmodel.action_tokens import ActionVocab, GameActions, collate_actions
from cosmicstar.worldmodel.encoder import StateEncoder
from cosmicstar.worldmodel.predictor import Predictor, slots_of
from cosmicstar.worldmodel.tokenize import GameFrames, collate

LPS = 22.4  # game loops per second (Faster)


def sample_batch(games, B, K, stride_s, deltas_s, rng):
    """B samples: context frames [B*K] (time-major within a sample), target frames [B], context
    times [B, K] and horizons [B] in seconds relative to t, and action windows (or None).
    games: list of (GameFrames, GameActions or None). Stride and horizons are in seconds, so
    games extracted at different frame rates mix."""
    ctx, tgt, dts, dls, wins = [], [], [], [], []
    while len(tgt) < B:
        g, ga = games[rng.randint(len(games))]
        stride = max(1, round(stride_s * LPS / g.step))
        dl = max(1, round(float(rng.choice(deltas_s)) * LPS / g.step))
        lo, hi = (K - 1) * stride, len(g) - 1 - dl
        if hi <= lo:
            continue
        t = rng.randint(lo, hi + 1)
        om = g.owner_map("observer", augment=True, rng=rng)  # one labeling for the whole sample
        idx = [t - (K - 1 - k) * stride for k in range(K)]
        ctx += [g.frame(f, om) for f in idx]
        tgt.append(g.frame(t + dl, om))
        dts.append([(g.loops[f] - g.loops[t]) / LPS for f in idx])
        dls.append((g.loops[t + dl] - g.loops[t]) / LPS)
        if ga is not None:
            # segments: [context k, context k+1) for the past, then [t, t + delta) = what's done next
            bounds = [g.loops[f] for f in idx]
            wins.append(ga.window(g.loops[idx[0]], g.loops[t + dl], bounds, g.loops[t], om))
    return ctx, tgt, torch.tensor(dts), torch.tensor(dls, dtype=torch.float32), (wins or None)


def to(b, dev):
    return {k: x.to(dev, non_blocking=True) for k, x in b.items()}


def norm_targets(x):
    return F.layer_norm(x.float(), x.shape[-1:])


def slot_loss(pred, target, mask, n_players):
    """Smooth L1 per slot, averaged within each kind (CLS, players, grid), then across kinds."""
    err = F.smooth_l1_loss(pred.float(), target, reduction="none").mean(-1)  # [B, S]
    m = mask.float()
    kinds = (slice(0, 1), slice(1, 1 + n_players), slice(1 + n_players, None))
    parts = [(err[:, k] * m[:, k]).sum() / m[:, k].sum().clamp(min=1) for k in kinds]
    return sum(parts) / 3, [p.item() for p in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/worldmodel/sample12")
    ap.add_argument("--val_games", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--context", type=int, default=4, help="context frames")
    ap.add_argument("--stride_s", type=float, default=2.9, help="seconds between context frames")
    ap.add_argument("--deltas_s", default="2.9,10,30", help="prediction horizons in seconds")
    ap.add_argument("--actions_dir", default="data/worldmodel/actions")
    ap.add_argument("--no_actions", action="store_true", help="ablation: predict without the action stream")
    ap.add_argument("--max_games", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ema", type=float, default=0.996, help="start momentum (-> 1.0, cosine)")
    ap.add_argument("--var_weight", type=float, default=0.1, help="variance hinge on context CLS (anti-collapse)")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--out", default="data/worldmodel/runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    dev = "cuda"
    # only games whose actions were parsed, so both arms of the ablation see the same games
    paths = [p for p in sorted(glob.glob(os.path.join(args.data, "*.npz")))
             if os.path.exists(os.path.join(args.actions_dir, os.path.basename(p)))]
    if args.max_games:
        paths = paths[:args.max_games]
    voc = vocab_mod.build(paths)
    v = vocab_mod.Vocab(voc)
    av = None if args.no_actions else ActionVocab()
    games = []
    for p in paths:
        g = GameFrames(p, v)
        games.append((g, None if av is None else GameActions(os.path.join(args.actions_dir, os.path.basename(p)), g, av)))
    order = rng.permutation(len(games))
    val = [games[i] for i in order[:args.val_games]]
    train = [games[i] for i in order[args.val_games:]]
    deltas = [float(x) for x in args.deltas_s.split(",")]
    run = os.path.join(args.out, time.strftime("stage1_%Y%m%d_%H%M%S"))
    os.makedirs(run, exist_ok=True)
    json.dump(dict(vars(args), train=[g.meta["map"] for g, _ in train], val=[g.meta["map"] for g, _ in val]),
              open(os.path.join(run, "config.json"), "w"), indent=1)
    json.dump(voc, open(os.path.join(run, "vocab.json"), "w"))
    print(f"{len(train)} train games ({sum(len(g) for g, _ in train)} frames), {len(val)} val games | "
          f"actions {'off' if args.no_actions else 'on'} | run dir {run}", flush=True)

    enc = StateEncoder(v).to(dev)
    tenc = copy.deepcopy(enc).requires_grad_(False)  # EMA target encoder
    pred = Predictor().to(dev)
    aenc = None if av is None else ActionEncoder(enc, av).to(dev)
    params = list(enc.parameters()) + list(pred.parameters()) + (list(aenc.parameters()) if aenc else [])
    print(f"encoder {sum(p.numel() for p in enc.parameters()) / 1e6:.2f}M, predictor {sum(p.numel() for p in pred.parameters()) / 1e6:.2f}M"
          + (f", action encoder {sum(p.numel() for p in aenc.parameters()) / 1e6:.2f}M" if aenc else ""), flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    # fixed validation samples
    vrng = np.random.RandomState(123)
    val_batches = [sample_batch(val, args.batch, args.context, args.stride_s, [d], vrng) for d in deltas for _ in range(4)]

    def step_tensors(ctx, tgt, dts, dls, wins, encoder):
        B, K = dts.shape
        cb, tb = to(collate(ctx), dev), to(collate(tgt), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c_out = encoder(cb)
            c_slots, c_mask = slots_of(c_out)
            S = c_slots.shape[1]
            c_slots, c_mask = c_slots.view(B, K, S, -1), c_mask.view(B, K, S)
            with torch.no_grad():
                t_slots, t_mask = slots_of(tenc(tb))
                last = {k: x.view(B, K, *x.shape[1:])[:, -1] for k, x in cb.items() if x.dim() >= 1 and x.shape[0] == B * K}
                base_slots, _ = slots_of(tenc(last))  # "nothing changes": encoding of frame t
            act = {}
            if aenc is not None and wins is not None:
                ab = to(collate_actions(wins), dev)
                summaries, s_mask, _ = aenc(ab)
                act = dict(act=summaries, act_mask=s_mask, act_dt=ab["seg_t"])
            p = pred(c_slots, c_mask, dts.to(dev), dls.to(dev), **act)
        P = c_out["players"].shape[1]
        target = norm_targets(t_slots)
        return p, target, norm_targets(base_slots), t_mask & c_mask[:, -1], P, c_out

    log = csv.writer(open(os.path.join(run, "log.csv"), "w", newline=""))
    log.writerow(["step", "split", "delta_s", "loss", "baseline", "cls", "players", "grid", "target_cls_std", "lr", "sec_per_step"])
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            enc.eval(); pred.eval()
            if aenc is not None:
                aenc.eval()
            by_d = {}
            with torch.no_grad():
                for vb in val_batches:
                    p, target, base, mask, P, _ = step_tensors(*vb, enc)
                    l, parts = slot_loss(p, target, mask, P)
                    bl, _ = slot_loss(base, target, mask, P)
                    d = round(float(vb[3].mean()), 1)
                    by_d.setdefault(d, []).append((l.item(), bl.item(), *parts))
            msg = []
            for d, rows in sorted(by_d.items()):
                r = np.mean(rows, 0)
                log.writerow([step, "val", d, *np.round(r, 4), "", "", ""])
                msg.append(f"{d:4.1f}s: {r[0]:.4f} vs no-change {r[1]:.4f}")
            print(f"[val step {step}] " + " | ".join(msg), flush=True)
            enc.train(); pred.train()
            if aenc is not None:
                aenc.train()
            if step == args.steps:
                break

        ctx, tgt, dts, dls, wins = sample_batch(train, args.batch, args.context, args.stride_s, deltas, rng)
        p, target, base, mask, P, c_out = step_tensors(ctx, tgt, dts, dls, wins, enc)
        loss, parts = slot_loss(p, target, mask, P)
        cls = c_out["cls"].float()
        var = F.relu(1.0 - cls.std(0)).mean()  # keep context CLS dims alive across the batch
        total = loss + args.var_weight * var
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        m = 1 - (1 - args.ema) * 0.5 * (1 + math.cos(math.pi * step / args.steps))  # momentum -> 1
        with torch.no_grad():
            for pt, pc in zip(tenc.parameters(), enc.parameters()):
                pt.mul_(m).add_(pc.detach(), alpha=1 - m)
        if step % 50 == 0:
            with torch.no_grad():
                bl, _ = slot_loss(base, target, mask, P)
                tstd = slots_of(tenc(to(collate(tgt), dev)))[0][:, 0].float().std(0).mean().item()
            sps = (time.time() - t0) / max(step, 1)
            log.writerow([step, "train", "mix", round(loss.item(), 4), round(bl.item(), 4), *np.round(parts, 4),
                          round(tstd, 4), opt.param_groups[0]["lr"], round(sps, 3)])
            print(f"step {step:5d} loss {loss.item():.4f} (no-change {bl.item():.4f}) cls/ply/grid {parts[0]:.3f}/{parts[1]:.3f}/{parts[2]:.3f} "
                  f"var {var.item():.3f} target-CLS std {tstd:.3f} lr {opt.param_groups[0]['lr']:.2e} {sps:.2f}s/step", flush=True)
    torch.save(dict(encoder=enc.state_dict(), target=tenc.state_dict(), predictor=pred.state_dict(),
                    actions=aenc.state_dict() if aenc else None, args=vars(args)),
               os.path.join(run, "model.pt"))
    print(f"saved {run}/model.pt", flush=True)


if __name__ == "__main__":
    main()
