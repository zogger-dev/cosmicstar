"""Pair-based simulator: the state at t plus everyone's commands in [t, t+D) -> the state at t+D.

A sample is a pair of moments of one game, D seconds apart, with a short context before t
(--context frames, --context_stride frames apart: enough to read motion, since a frame holds
no velocities or orders). D is drawn log-uniformly per batch between --min_s and --max_s, so
one model learns the dynamics of a second and of a quarter minute, and the latent action is
conditioned on D. The observer view with all players' actions: state and actions determine
the next state, so no history beyond the context is needed (the belief model, under fog, is
the sequence model of train_window.py).

Per (pair, player) a latent action with two sources that must agree (InfoNCE over the batch,
every item a different moment): the command encoder's summary of that player's commands in
the interval, which drives the predictor, and an inverse model reading the frames at t and
t+D (the change; the oracle in evaluation). Command pointers keep only what the player did
(static_pointers: no future unit state). The prediction loss is divided by the loss of copying
the present, so every horizon weighs the same.

Two things make the action channel learnable, both learned the hard way (the first version
reached 90% in-sample agreement by matching on game identity and 2% held out, with the
predictor ignoring the actions):
  - hard negatives: a batch is --games_per_batch games x several moments each, so the same
    game and player at other moments are negatives and identity doesn't discriminate;
  - a counterfactual loss on the predictor: for each pair the prediction with the real latent
    action must score best against predictions with borrowed ones (mostly from the same game),
    as a softmax over candidates on the relative loss (--cf_weight, --cf_temp).

An explicit spatial target (--cnt_weight): a small head on the predictor's grid outputs
predicts each cell's unit / army / structure counts at t+D (log1p, from probe_targets). The
embedding target alone never beat "nothing changes" on the grid (the residual added noise on
static cells and couldn't improve on the present where things moved); counts make "nothing
changes here" exact and a command's consequence concrete, and they evaluate in game units
with no readers: RMSE on the cells that change, real commands vs adjacent intervals' vs none
vs copying the present.

Evaluation at fixed horizons (--eval_s), per held-out set: the model's loss vs copying the
present, agreement top-1, and whether the predictor uses the actions (real commands vs other
pairs' commands: recall@1 vs chance, MRR; and no actions), plus the oracle.

    python scripts/train_pairs.py --min_s 1 --max_s 15 --batch 32
    python scripts/train_pairs.py --resume data/worldmodel/runs/<run>
"""

import argparse
import collections
import copy
import csv
import json
import math
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train_window import LPS, GamePool, kind_slices, slot_loss, to  # noqa: E402
from cosmicstar.worldmodel.tokenize import N_CODES  # noqa: E402
from cosmicstar.worldmodel import vocab as vocab_mod  # noqa: E402
from cosmicstar.worldmodel.action_encoder import ActionEncoder  # noqa: E402
from cosmicstar.worldmodel.action_tokens import ActionVocab, GameActions, collate_actions  # noqa: E402
from cosmicstar.worldmodel.encoder import StateEncoder  # noqa: E402
from cosmicstar.worldmodel.inverse_model import InverseModel  # noqa: E402
from cosmicstar.worldmodel.predictor import slots_of  # noqa: E402
from cosmicstar.worldmodel.probes import temporal_share  # noqa: E402
from cosmicstar.worldmodel.tokenize import GameFrames, collate, probe_targets  # noqa: E402
from cosmicstar.worldmodel.window_predictor import WindowPredictor  # noqa: E402


NEAR = (-3, -2, -1, 1, 2, 3)  # evaluation: borrowed chunks from the same game this many intervals away


def make_pair(game, d_frames, ctx, stride, rng, augment=True, codes=None, near=(), t=None):
    """One sample: `ctx` context frames ending at t, the frame at t + d_frames, and every
    player's commands in between (one segment, pointers static). codes: the batch's owner-code
    draw, shared by all its pairs (a pair's own random codes would let the in-batch losses
    tell pairs apart by their codes alone, which is what the first version learned)."""
    g, ga = game
    lo, hi = stride * (ctx - 1), len(g) - d_frames - 1
    if hi < lo:
        return None
    if t is None or not (lo <= t <= hi):  # a requested moment outside the game: a random one instead
        t = rng.randint(lo, hi + 1)
    om = g.owner_map("observer", augment=augment, rng=rng, codes=codes)
    idx = [t - stride * (ctx - 1 - i) for i in range(ctx)]
    loops = g.loops

    def chunk(t0):  # the commands in [t0, t0 + D), as a chunk starting at t0
        return ga.window(int(loops[t0]), int(loops[t0 + d_frames]), np.array([loops[t0]]), int(loops[t0]), om,
                         static_pointers=True)

    near_acts = {off: chunk(t + off * d_frames) for off in near
                 if 0 <= t + off * d_frames and t + off * d_frames + d_frames < len(g)}
    return dict(t=t, frames=[g.frame(f, om) for f in idx], target=g.frame(t + d_frames, om),
                times=((loops[idx] - loops[t]) / LPS).astype(np.float32), act=chunk(t), near=near_acts,
                cnt_now=probe_targets(g, t)["grid"][:, :3], cnt_fut=probe_targets(g, t + d_frames)["grid"][:, :3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/worldmodel/obs")
    ap.add_argument("--actions_dir", default="data/worldmodel/actions")
    ap.add_argument("--splits", default="data/worldmodel/splits.json")
    ap.add_argument("--min_s", type=float, default=1.0, help="shortest horizon (log-uniform per batch)")
    ap.add_argument("--max_s", type=float, default=15.0, help="longest horizon")
    ap.add_argument("--eval_s", default="1,5,15", help="fixed horizons for evaluation")
    ap.add_argument("--context", type=int, default=3, help="context frames ending at t")
    ap.add_argument("--no_orders", action="store_true", help="ablation: ignore unit orders in the state (data that has them)")
    ap.add_argument("--target_no_orders", action="store_true",
                    help="orders in the input state only: the target encoder ignores them, so the target is the physical "
                         "state (orders at t+D are partly the players' future commands, i.e. policy, not dynamics)")
    ap.add_argument("--context_stride", type=int, default=2, help="frames between context frames")
    ap.add_argument("--batch", type=int, default=32, help="pairs per pass")
    ap.add_argument("--games_per_batch", type=int, default=8, help="the batch's pairs come from this many games (hard negatives)")
    ap.add_argument("--adjacent", type=int, default=2,
                    help="per game, this many of its moments are consecutive intervals (t, t+D, ...): the hardest negatives, "
                         "since only the specific commands tell them apart; the rest are random moments")
    ap.add_argument("--cf_weight", type=float, default=1.0, help="counterfactual loss: the real latent action must predict best")
    ap.add_argument("--cf_temp", type=float, default=0.05, help="its temperature, on the loss relative to copying the present")
    ap.add_argument("--cf_fakes", type=int, default=5, help="borrowed latent actions per pair in training (same game first)")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--pool", type=int, default=60)
    ap.add_argument("--rotate_every", type=int, default=50)
    ap.add_argument("--rotate", type=int, default=4)
    ap.add_argument("--val_games", type=int, default=3)
    ap.add_argument("--val_pairs", type=int, default=64, help="pairs per (set, horizon)")
    ap.add_argument("--fakes", type=int, default=7, help="other pairs' commands tried per pair in the counterfactual check")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--var_weight", type=float, default=0.1)
    ap.add_argument("--var_memory", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--d_z", type=int, default=64)
    ap.add_argument("--nce_weight", type=float, default=0.1)
    ap.add_argument("--nce_temp", type=float, default=0.1)
    ap.add_argument("--cnt_weight", type=float, default=1.0, help="per-cell count target on the grid outputs (0 = off)")
    ap.add_argument("--z_source", default="cmd", help="what drives the predictor in training: cmd, inv, or mix")
    ap.add_argument("--no_ckpt", action="store_true")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--save_minutes", type=float, default=20)
    ap.add_argument("--max_games", type=int, default=0)
    ap.add_argument("--out", default="data/worldmodel/runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--eval_only", action="store_true", help="with --resume: evaluate the checkpoint and exit")
    args = ap.parse_args()

    ck = None
    if args.resume:
        if os.path.isdir(args.resume):
            args.resume = os.path.join(args.resume, "model_latest.pt")
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        args = argparse.Namespace(**{**vars(ap.parse_args([])), **ck["args"], "resume": args.resume, "eval_only": args.eval_only})
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    dev = "cuda"
    split = json.load(open(args.splits))

    def available(names):
        return [os.path.join(args.data, f) for f in sorted(names)
                if os.path.exists(os.path.join(args.data, f)) and os.path.exists(os.path.join(args.actions_dir, f))]

    train_paths = available(split["train"])
    if args.max_games:
        train_paths = train_paths[:args.max_games]
    if ck is not None and ck.get("train_files"):
        keep = set(ck["train_files"])
        train_paths = [q for q in train_paths if os.path.basename(q) in keep]
    eval_paths = {"seen-games": train_paths[:args.val_games],
                  "unseen-games": available(split["held_out_games"])[:args.val_games],
                  "unseen-maps": available(split["held_out_map_games"])[:args.val_games]}
    run = os.path.dirname(os.path.abspath(args.resume)) if args.resume else \
        os.path.join(args.out, time.strftime("pairs_%Y%m%d_%H%M%S"))
    voc = json.load(open(os.path.join(run, "vocab.json"))) if args.resume else vocab_mod.build(train_paths[:60])
    v = vocab_mod.Vocab(voc)
    av = ActionVocab()

    def load(p):
        g = GameFrames(p, v)
        return g, GameActions(os.path.join(args.actions_dir, os.path.basename(p)), g, av)

    eval_games = {name: [load(p) for p in ps] for name, ps in eval_paths.items() if ps}
    pool = GamePool(train_paths, min(args.pool, len(train_paths)), load, rng,
                    names=ck.get("pool") if ck is not None else None)
    loaded = [x for _, x in pool.games] + [x for gs in eval_games.values() for x in gs]
    step_loops = collections.Counter(g.step for g, _ in loaded).most_common(1)[0][0]
    assert all(g.step == step_loops for g, _ in loaded), "mixed frame rates"
    frame_s = step_loops / LPS
    eval_s = [float(x) for x in args.eval_s.split(",")]

    os.makedirs(run, exist_ok=True)
    if not args.resume:
        json.dump(dict(vars(args), step_loops=step_loops, n_train=len(train_paths), horizons_s=args.eval_s,
                       window_s=args.context * args.context_stride * frame_s, frame_stride=1, long_sub=1, no_actions=False,
                       residual=True, eval={k: [os.path.basename(p) for p in ps] for k, ps in eval_paths.items()}),
                  open(os.path.join(run, "config.json"), "w"), indent=1)
        json.dump(voc, open(os.path.join(run, "vocab.json"), "w"))
    print(f"{len(train_paths)} train games (pool {len(pool.games)}) | eval: "
          + ", ".join(f"{k} {len(v_)} games" for k, v_ in eval_games.items())
          + f" | orders {'off' if args.no_orders else 'in the input only' if args.target_no_orders else 'on'}"
          + f" | horizons {args.min_s}-{args.max_s}s (eval {args.eval_s}) | context {args.context} frames "
          f"{args.context_stride * frame_s:.2f}s apart | batch {args.batch} x {args.accum} | predictor driven by {args.z_source} | {run}",
          flush=True)

    enc = StateEncoder(v).to(dev)
    enc.grad_ckpt = not args.no_ckpt
    tenc = copy.deepcopy(enc).requires_grad_(False)
    pred = WindowPredictor(depth=args.depth, horizons=("var",), grad_ckpt=not args.no_ckpt, residual=True, horizon_cond=True).to(dev)
    aenc = ActionEncoder(enc, av).to(dev)
    inv = InverseModel(enc.d, args.d_z).to(dev)
    zproj = nn.Linear(enc.d, args.d_z).to(dev)
    zup = nn.Linear(args.d_z, enc.d).to(dev)
    if args.no_orders:
        enc.use_orders = False
    tenc.use_orders = enc.use_orders and not args.target_no_orders
    chead = nn.Sequential(nn.LayerNorm(enc.d), nn.Linear(enc.d, enc.d), nn.GELU(), nn.Linear(enc.d, 3)).to(dev)  # counts per cell
    modules = dict(encoder=enc, predictor=pred, actions=aenc, inverse=inv, zproj=zproj, zup=zup, chead=chead)
    params = [p for m in modules.values() for p in m.parameters()]
    n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    print(f"encoder {n(enc):.2f}M, predictor {n(pred):.2f}M, action encoder {n(aenc):.2f}M, inverse {n(inv):.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    warm = 200
    lr_at = lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))
    start = 0
    cls_memory = torch.zeros(0, enc.d, device=dev)
    if ck is not None:
        for k, m in modules.items():
            if k in ck:  # older checkpoints lack the count head; the encoder may lack the order path
                m.load_state_dict(ck[k], strict=k != "encoder")
        tenc.load_state_dict(ck["target"], strict=False)
        enc.use_orders = any(k.startswith("ord_in") for k in ck["encoder"]) and not args.no_orders
        tenc.use_orders = enc.use_orders and not args.target_no_orders
        try:
            opt.load_state_dict(ck["opt"])
        except ValueError:  # the model gained parameters since (e.g. the order path): statistics rebuilt
            print("(optimizer statistics rebuilt: the checkpoint's parameter groups differ)", flush=True)
        start = ck["step"]
        rng.set_state(ck["np_rng"])
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"])
        cls_memory = ck["cls_memory"].to(dev)
        print(f"resuming {run} at step {start}", flush=True)
    for g_ in opt.param_groups:
        g_.setdefault("initial_lr", args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at, last_epoch=start - 1)

    def encode(pairs, d_s):
        """Everything about a batch of pairs that doesn't depend on which actions drive the
        predictor: context slots, the present and the target (target space), the latent
        actions from the commands (z_cmd) and from the state change (z_inv), [B, P, d_z]."""
        B, C = len(pairs), args.context
        fb = to(collate([f for p in pairs for f in p["frames"]]), dev)
        tb = to(collate([p["target"] for p in pairs]), dev)
        hor = torch.full((B,), float(d_s), device=dev)
        st = dict(B=B, hor=hor, times=torch.from_numpy(np.stack([p["times"] for p in pairs])).to(dev),
                  gid=torch.tensor([p.get("gid", i) for i, p in enumerate(pairs)], device=dev),
                  t=torch.tensor([p["t"] for p in pairs], device=dev), d_frames=int(round(d_s / frame_s)),
                  cnt_now=torch.from_numpy(np.stack([p["cnt_now"] for p in pairs])).to(dev),
                  cnt_fut=torch.from_numpy(np.stack([p["cnt_fut"] for p in pairs])).to(dev))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = enc(fb)
            slots, mask = slots_of(out)
            S = slots.shape[1]
            slots, mask = slots.view(B, C, S, -1), mask.view(B, C, S)
            after, amask = slots_of(enc(tb))  # the future frame as the context encoder sees it, for the inverse model
            with torch.no_grad():
                pw, _ = slots_of(tenc(fb))
                present = F.layer_norm(pw.float(), pw.shape[-1:]).view(B, C, S, -1)
                tg, tmask = slots_of(tenc(tb))
                target = F.layer_norm(tg.float(), tg.shape[-1:])
            ab = to(collate_actions([p["act"] for p in pairs]), dev)
            summaries, s_mask, _ = aenc(ab, horizon_s=hor)
            z_cmd, s_mask = zproj(summaries[:, 0]), s_mask[:, 0]
            z_inv = inv(slots[:, -1], after, mask[:, -1], amask, z_cmd.shape[1], horizon_s=hor)
        st.update(slots=slots, mask=mask, present=present, target=target, valid=tmask & mask[:, -1], P=out["players"].shape[1],
                  cls=out["cls"].view(B, C, -1)[:, -1], z_cmd=z_cmd, z_inv=z_inv, s_mask=s_mask)
        return st

    def run_pred(st, use_inv=False, borrow=0, none=False, z=None):
        """The prediction for t + D, [B, S, d], driven by the commands' latent action (default),
        the state change's (use_inv: the oracle), another pair's commands (borrow = roll along
        the batch, or z given explicitly) or nothing (none)."""
        B, C = st["B"], args.context
        if z is None:
            z = st["z_inv"] if use_inv else st["z_cmd"]
        if borrow:
            z = torch.roll(z, borrow, 0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = zup(z)
            act = torch.zeros(B, C, a.shape[1], a.shape[2], device=dev, dtype=a.dtype)
            act[:, -1] = a  # the latent action sits with the present frame
            am = torch.zeros(B, C, a.shape[1], dtype=torch.bool, device=dev)
            if not none:
                am[:, -1] = st["s_mask"]
            preds = pred(st["slots"], st["mask"], st["times"], act=act, act_mask=am, base=st["present"], horizon_s=st["hor"])
        return preds["var"][:, -1]

    def counts(pr, st):
        """The count head on a prediction's grid slots: log1p units / army / structures per cell, [B, G*G, 3]."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return chead(pr[:, 1 + st["P"]:]).float()

    def pair_loss(pr, st):
        """The training loss per pair, [B]: smooth L1 per slot, averaged within each kind
        (CLS / players / grid) and then across kinds."""
        err = F.smooth_l1_loss(pr.float(), st["target"], reduction="none").mean(-1)
        m = st["valid"].float()
        parts = [(err[:, s] * m[:, s]).sum(1) / m[:, s].sum(1).clamp(min=1) for s in kind_slices(st["P"]).values()]
        return sum(parts) / len(parts)

    def agreement(st):
        """InfoNCE between each (pair, player)'s latent action from the state change and from
        the commands, over the batch: the same game and player at other moments are negatives
        (hard), overlapping moments of the same game are not. Returns the loss and top-1."""
        m = st["s_mask"].reshape(-1)
        Pz = st["s_mask"].shape[1]
        zi = F.normalize(st["z_inv"].float().reshape(-1, args.d_z)[m], dim=-1)
        zc = F.normalize(st["z_cmd"].float().reshape(-1, args.d_z)[m], dim=-1)
        gid, t = st["gid"].repeat_interleave(Pz)[m], st["t"].repeat_interleave(Pz)[m]
        sim = zi @ zc.t() / args.nce_temp
        near = (gid[:, None] == gid[None, :]) & ((t[:, None] - t[None, :]).abs() < st["d_frames"])
        near.fill_diagonal_(False)
        sim = sim.masked_fill(near, float("-inf"))
        tgt = torch.arange(len(sim), device=dev)
        return (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.t(), tgt)) / 2, (sim.argmax(1) == tgt).float().mean()

    def borrow_index(st, K):
        """For each pair, K other pairs whose latent actions to try: the same game's other
        moments first (hard), then other games. [K, B] indices."""
        gid = st["gid"].cpu().numpy()
        B = len(gid)
        out = []
        for i in range(B):
            same = [j for j in range(B) if j != i and gid[j] == gid[i]]
            other = [j for j in range(B) if gid[j] != gid[i]]
            cand = same + other
            out.append(cand[:K] if len(cand) >= K else (cand * K)[:K])
        return torch.tensor(np.array(out).T, device=dev)  # [K, B]

    def counterfactual(st, real_rel, nochange):
        """The real latent action must give the best prediction: softmax over the real and K
        borrowed candidates on the loss relative to copying the present. Returns the loss
        and the in-batch share of pairs where the real action wins."""
        idx = borrow_index(st, args.cf_fakes)
        fakes = torch.stack([pair_loss(run_pred(st, z=st["z_cmd"][idx[k]]), st) / (nochange + 1e-3) for k in range(len(idx))])
        logits = -torch.cat([real_rel[None], fakes]) / args.cf_temp              # [1+K, B]
        loss = F.cross_entropy(logits.t(), torch.zeros(len(real_rel), dtype=torch.long, device=dev))
        return loss, (fakes > real_rel[None]).all(0).float().mean()

    # fixed evaluation pairs per set and horizon
    vrng = np.random.RandomState(123)
    eval_pairs = {name: [] for name in eval_games}
    for name, gs in eval_games.items():
        for d_s in eval_s:
            d_frames = max(1, int(round(d_s / frame_s)))
            ps, i = [], 0
            while len(ps) < args.val_pairs and i < 20 * args.val_pairs:
                p = make_pair(gs[i % len(gs)], d_frames, args.context, args.context_stride, vrng, augment=False, near=NEAR)
                i += 1
                if p is not None:
                    p["gid"] = (i - 1) % len(gs)
                    ps.append(p)
            eval_pairs[name].append((round(d_frames * frame_s, 2), ps))

    @torch.no_grad()
    def evaluate(step, name):
        for m in modules.values():
            m.eval()
        share = temporal_share(tenc, [g for g, _ in eval_games[name]], dev)
        rows = []
        for d_s, pairs in eval_pairs[name]:
            acc = collections.defaultdict(list)
            for i in range(0, len(pairs), args.batch):
                ps = pairs[i:i + args.batch]
                st = encode(ps, d_s)
                pr = run_pred(st)
                real = pair_loss(pr, st)
                cnt_dec = {"real": counts(pr, st), "none": counts(run_pred(st, none=True), st)}
                K = min(args.fakes, len(ps) - 1)
                fakes = torch.stack([pair_loss(run_pred(st, borrow=s), st) for s in range(1, K + 1)])  # [K, B]
                rank = 1 + (fakes <= real[None]).sum(0).float()  # ties count against the real actions
                # the honest half of the check: borrowed actions from the SAME game (other moments)
                ar = torch.arange(len(ps), device=dev)
                same = torch.stack([st["gid"][(ar - s_) % len(ps)] == st["gid"] for s_ in range(1, K + 1)])  # [K, B]
                worse = fakes <= real[None]
                has_same, has_other = same.any(0), (~same).any(0)
                acc["win_same"] += ((~worse) | ~same).all(0)[has_same].float().tolist()
                acc["win_other"] += ((~worse) | same).all(0)[has_other].float().tolist()
                acc["borrowed_same"] += ((fakes * same).sum(0) / same.sum(0).clamp(min=1))[has_same].tolist()
                acc["borrowed_other"] += ((fakes * ~same).sum(0) / (~same).sum(0).clamp(min=1))[has_other].tolist()
                # the hardest check: the same game's commands a few intervals before or after
                near_loss = {}
                for off in NEAR:
                    idx = [i for i, p in enumerate(ps) if off in p["near"]]
                    if not idx:
                        continue
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        nb = to(collate_actions([ps[i]["near"][off] for i in idx]), dev)
                        summ, _, _ = aenc(nb, horizon_s=st["hor"][idx])
                        z_full = st["z_cmd"].clone()
                        z_full[idx] = zproj(summ[:, 0]).to(z_full.dtype)
                    pr_off = run_pred(st, z=z_full)
                    near_loss[off] = (idx, pair_loss(pr_off, st)[idx])
                    if abs(off) == 1:
                        cnt_dec[off] = (idx, counts(pr_off, st))
                for i in range(len(ps)):
                    got = {off: l[[j for j, k in enumerate(ix)].index(ix.index(i))] for off, (ix, l) in near_loss.items() if i in ix}
                    if not got:
                        continue
                    acc["near_all"].append(float(all(v > real[i] for v in got.values())))
                    adj = [v for off, v in got.items() if abs(off) == 1]
                    if adj:
                        acc["near_adj"].append(float(all(v > real[i] for v in adj)))
                        acc["near_adj_loss"] += [v.item() for v in adj]
                    acc["near_far_loss"] += [v.item() for off, v in got.items() if abs(off) > 1]
                # counts per cell, game units: the cells whose unit count changes between t and t+D
                now_u, fut = torch.expm1(st["cnt_now"]), torch.expm1(st["cnt_fut"])
                chg = (fut[..., 0] - now_u[..., 0]).abs() >= 1.0                               # [B, G*G]
                has = chg.any(1)
                for qi, qn in ((0, "units"), (1, "army")):
                    err = {"copy": (now_u[..., qi] - fut[..., qi]) ** 2}
                    for k in ("real", "none"):
                        err[k] = (torch.expm1(cnt_dec[k][..., qi]) - fut[..., qi]) ** 2
                    for off in (-1, 1):
                        if off in cnt_dec:
                            idx, c = cnt_dec[off]
                            e = torch.full_like(err["real"], float("nan"))
                            e[idx] = ((torch.expm1(c[..., qi]) - fut[..., qi]) ** 2)[idx]
                            err[off] = e
                    per_pair = {k: torch.stack([v[n][chg[n]].mean() for n in torch.where(has)[0]]) if has.any() else v.new_zeros(0)
                                for k, v in err.items()}
                    for k, v in err.items():
                        acc[(f"cnt_{qn}", k, "chg")] += v[chg].tolist()
                        acc[(f"cnt_{qn}", k, "all")] += v.flatten().tolist()
                        if k != "real" and len(per_pair[k]):
                            ok_ = ~torch.isnan(per_pair[k])
                            acc[(f"cnt_{qn}", k, "win")] += (per_pair["real"][ok_] < per_pair[k][ok_]).float().tolist()
                _, agree = agreement(st)
                acc["real"] += real.tolist()
                acc["copy"] += pair_loss(st["present"][:, -1], st).tolist()
                acc["none"] += pair_loss(run_pred(st, none=True), st).tolist()
                acc["oracle"] += pair_loss(run_pred(st, use_inv=True), st).tolist()
                acc["borrowed"] += fakes.mean(0).tolist()
                acc["recall"] += (rank == 1).float().tolist()
                acc["mrr"] += (1 / rank).tolist()
                acc["agree"].append(agree.item())
                acc["chance"].append(1 / (K + 1))
            m_ = {k: float(np.nanmean(x)) if len(x) else float("nan") for k, x in acc.items()}
            cnt_rows = {}
            for qn in ("units", "army"):
                r_ = lambda k, w: float(np.sqrt(np.nanmean(acc[(f"cnt_{qn}", k, w)]))) if acc.get((f"cnt_{qn}", k, w)) else float("nan")
                w_ = lambda k: float(np.mean(acc[(f"cnt_{qn}", k, "win")])) if acc.get((f"cnt_{qn}", k, "win")) else float("nan")
                cnt_rows[qn] = dict(real=r_("real", "chg"), adj_m=r_(-1, "chg"), adj_p=r_(1, "chg"), none=r_("none", "chg"),
                                    copy=r_("copy", "chg"), real_all=r_("real", "all"), copy_all=r_("copy", "all"),
                                    win_m=w_(-1), win_p=w_(1), win_none=w_("none"), win_copy=w_("copy"))
            ratio = m_["real"] / max(m_["copy"], 1e-9)
            rows.append(dict(set=name, step=step, horizon_s=d_s, loss=m_["real"], copy=m_["copy"], ratio=ratio,
                             agree=m_["agree"], recall=m_["recall"], chance=m_["chance"], mrr=m_["mrr"],
                             borrowed=m_["borrowed"], none=m_["none"], oracle=m_["oracle"],
                             share_cls=share["cls"], share_grid=share["grid"]))
            for qn, c in cnt_rows.items():
                print(f"[{name} {step}] D={d_s:g}s counts/cell ({qn}, game units) changing cells RMSE: real {c['real']:.2f} "
                      f"adj-1 {c['adj_m']:.2f} adj+1 {c['adj_p']:.2f} none {c['none']:.2f} copy {c['copy']:.2f} | all cells real "
                      f"{c['real_all']:.2f} copy {c['copy_all']:.2f} | real beats, per pair: adj-1 {c['win_m']:.0%} adj+1 {c['win_p']:.0%} "
                      f"none {c['win_none']:.0%} copy {c['win_copy']:.0%}", flush=True)
                rows_extra = {f"cnt_{qn}_{k}": v for k, v in c.items()}
            print(f"[{name} {step}] D={d_s:g}s: loss {m_['real']:.4f} vs copy {m_['copy']:.4f} ({ratio:.0%}) | agreement top-1 "
                  f"{m_['agree']:.0%} | actions: recall@1 {m_['recall']:.0%} (chance {m_['chance']:.0%}) MRR {m_['mrr']:.2f} | "
                  f"borrowed {m_['borrowed']:.4f} none {m_['none']:.4f} oracle {m_['oracle']:.4f} | real beats all same-game "
                  f"borrows {m_.get('win_same', float('nan')):.0%} (their loss {m_.get('borrowed_same', float('nan')):.4f}), "
                  f"all other-game {m_.get('win_other', float('nan')):.0%} ({m_.get('borrowed_other', float('nan')):.4f}) | "
                  f"same game NEAR: beats the adjacent chunks {m_.get('near_adj', float('nan')):.0%} (their loss "
                  f"{m_.get('near_adj_loss', float('nan')):.4f}), beats all within 3 {m_.get('near_all', float('nan')):.0%} "
                  f"(2-3 away: {m_.get('near_far_loss', float('nan')):.4f})", flush=True)
        print(f"[{name} {step}] temporal share of variance: cls {share['cls']:.0%}, grid {share['grid']:.0%}", flush=True)
        for m in modules.values():
            m.train()
        return rows, share

    def save(tag, step):
        a = {k: v_ for k, v_ in vars(args).items() if k != "resume"}
        torch.save(dict({k: m.state_dict() for k, m in modules.items()}, target=tenc.state_dict(), args=a,
                        step_loops=step_loops, horizons=("var",), T=args.context, step=step, opt=opt.state_dict(),
                        np_rng=rng.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        cls_memory=cls_memory.cpu(), pool=pool.names(),
                        train_files=[os.path.basename(q) for q in train_paths]),
                   os.path.join(run, f"model_{tag}.tmp"))
        os.replace(os.path.join(run, f"model_{tag}.tmp"), os.path.join(run, f"model_{tag}.pt"))

    if args.eval_only:
        for name in eval_pairs:
            evaluate(start, name)
        return
    log_rows, share0 = [], {}
    if ck is not None and os.path.exists(os.path.join(run, "eval.csv")):
        with open(os.path.join(run, "eval.csv"), newline="") as fh:
            log_rows = [r for r in csv.DictReader(fh) if int(r["step"]) <= start]
        for r in log_rows:
            if int(r["step"]) == 0:
                share0[r["set"]] = dict(cls=float(r["share_cls"]), grid=float(r["share_grid"]))
    for name in eval_pairs:
        if name not in share0:
            rows, share0[name] = evaluate(0, name)
            log_rows += rows
    t0 = last_save = time.time()
    pause_file = os.path.join(run, "PAUSE")
    for step in range(start + 1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        d_s = math.exp(rng.uniform(math.log(args.min_s), math.log(args.max_s)))
        d_frames = max(1, int(round(d_s / frame_s)))
        d_s = d_frames * frame_s
        tot = collections.defaultdict(float)
        for _ in range(args.accum):
            pairs = []
            G = min(args.games_per_batch, len(pool.games))
            picks = rng.choice(len(pool.games), G, replace=False)
            codes = (rng.permutation(N_CODES), rng.permutation(N_CODES))  # one owner-code draw for the whole batch
            first_t = {}
            while len(pairs) < args.batch:  # a few moments per game: hard negatives
                gi = int(picks[len(pairs) % G])
                n_g = sum(1 for q in pairs if q["gid"] == gi)
                t_req = first_t[gi] + n_g * d_frames if 1 <= n_g < args.adjacent and gi in first_t else None
                p = make_pair(pool.games[gi][1], d_frames, args.context, args.context_stride, rng, codes=codes, t=t_req)
                if p is not None:
                    p["gid"] = gi
                    pairs.append(p)
                    first_t.setdefault(gi, p["t"])
            st = encode(pairs, d_s)
            use_inv = args.z_source == "inv" or (args.z_source == "mix" and rng.rand() < 0.5)
            pr = run_pred(st, use_inv=use_inv)
            with torch.no_grad():
                nochange = pair_loss(st["present"][:, -1], st)                 # [B]
            real_rel = pair_loss(pr, st) / (nochange + 1e-3)
            loss, rel = slot_loss(pr, st["target"], st["valid"], st["P"]), real_rel.mean()
            nce, agree = agreement(st)
            cf, cf_win = counterfactual(st, real_rel, nochange) if args.cf_weight > 0 else (torch.zeros((), device=dev),) * 2
            cnt = F.mse_loss(counts(pr, st), st["cnt_fut"]) if args.cnt_weight > 0 else torch.zeros((), device=dev)
            with torch.no_grad():
                cnt_copy = F.mse_loss(st["cnt_now"], st["cnt_fut"])  # the same target, copying the present's counts
            # anti-collapse: the present's CLS must vary across moments (these plus recent ones)
            cls = st["cls"].float()
            spread = torch.cat([cls, cls_memory])
            var = F.relu(1.0 - spread.std(0)).mean() if len(spread) > 1 else cls.sum() * 0
            cls_memory = torch.cat([cls.detach(), cls_memory])[:args.var_memory]
            ((rel + args.var_weight * var + args.nce_weight * nce + args.cf_weight * cf + args.cnt_weight * cnt) / args.accum).backward()
            for k, x in (("loss", loss), ("copy", nochange.mean()), ("rel", rel), ("nce", nce), ("agree", agree), ("var", var),
                         ("cf", cf), ("cf_win", cf_win), ("cnt", cnt), ("cnt_copy", cnt_copy)):
                tot[k] += x.item() / args.accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        m_ema = 1 - (1 - args.ema) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        with torch.no_grad():
            for pt, pc in zip(tenc.parameters(), enc.parameters()):
                pt.mul_(m_ema).add_(pc.detach(), alpha=1 - m_ema)
        if step % args.rotate_every == 0:
            pool.rotate(args.rotate)
        if step % 25 == 0:
            sps = (time.time() - t0) / (step - start)
            print(f"step {step:5d} D={d_s:4.1f}s loss {tot['loss']:.4f} vs copy {tot['copy']:.4f} ({tot['rel']:.0%}) "
                  f"agree {tot['nce']:.3f}/{tot['agree']:.0%} cf {tot['cf']:.3f}/{tot['cf_win']:.0%} "
                  f"cnt {tot['cnt']:.4f} (copy {tot['cnt_copy']:.4f}) var {tot['var']:.3f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{sps:.2f}s/step mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            for name in eval_pairs:
                rows, share = evaluate(step, name)
                log_rows += rows
                for k in ("cls", "grid"):
                    if share[k] < 0.5 * share0[name][k]:
                        print(f"!! collapse alarm ({name}): {k} temporal share {share[k]:.0%} vs {share0[name][k]:.0%} at init", flush=True)
            with open(os.path.join(run, "eval.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
                w.writeheader()
                w.writerows(log_rows)
        if step % args.save_every == 0:
            save(f"step{step}", step)
        if time.time() - last_save >= args.save_minutes * 60:
            save("latest", step)
            last_save = time.time()
        if os.path.exists(pause_file):
            save("latest", step)
            os.remove(pause_file)
            print(f"paused at step {step}; resume with --resume {run}", flush=True)
            return
    save("final", args.steps)
    print(f"saved {run}/model_final.pt", flush=True)


if __name__ == "__main__":
    main()
