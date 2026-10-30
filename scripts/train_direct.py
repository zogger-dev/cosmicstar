"""Direct state-space simulator (worldmodel/direct.py): the state at t, plus everyone's
commands in [t, t+k) or nothing, -> every entity's state at t+k, for k up to --max_s seconds.

A batch is --frames present frames, each paired with --hor horizons k drawn uniformly in
frames (the observer extraction is 4 loops per frame, so 2.5 s is 14 frames). The present is
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

    python scripts/train_direct.py --steps 3000
    python scripts/train_direct.py --resume data/worldmodel/runs/<run> --eval_only
"""
import argparse
import collections
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

from train_window import LPS, GamePool, to  # noqa: E402
from cosmicstar.worldmodel import vocab as vocab_mod  # noqa: E402
from cosmicstar.worldmodel import latent as lat  # noqa: E402
from cosmicstar.worldmodel import direct as dm  # noqa: E402
from cosmicstar.worldmodel.action_encoder import ActionEncoder  # noqa: E402
from cosmicstar.worldmodel.action_tokens import ActionVocab, GameActions, collate_actions  # noqa: E402
from cosmicstar.worldmodel.inverse_model import InverseModel  # noqa: E402
from cosmicstar.worldmodel.tokenize import GameFrames, collate  # noqa: E402

TARGETS = ("exit", "dx", "dy", "hp", "sh", "en", "build", "order")


class Game:
    """A replay's frames, its action tokens and its per-frame death lists."""

    def __init__(self, path, actions_dir, vocab, avocab):
        self.g = GameFrames(path, vocab)
        self.ga = GameActions(os.path.join(actions_dir, os.path.basename(path)), self.g, avocab)
        with np.load(path) as z:
            self.dead_id, self.dead_start = z["dead_id"], z["dead_frame_start"]
        self.slot_index = {s: i for i, s in enumerate(self.g.slots)}


def make_sample(game, t, k, rng, with_actions, om, frame):
    """The frame at t (built once per present frame and shared by its horizons), the frame at
    t+k (for the posterior), the commands in between (or none) and per-entity targets at t+k
    for the entities of t, in their row order."""
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
    act = game.ga.window(lo, hi if with_actions else lo, np.array([lo]), lo, om, static_pointers=True)
    pidx = np.array([game.slot_index.get(int(o), -1) for o in u["owner"][a:b]], np.int64)
    touched = np.isin(ids, act["touched_uid"][act["touched_uid"] >= 0])  # selected or targeted by a command in the window
    births = int(g.start[t + k + 1] - g.start[t + k]) - int(present.sum())
    return dict(frame=frame, future=g.frame(t + k, om), act=act, tgt=tgt, now=now, pidx=pidx,
                has_act=with_actions, hor_s=k * g.step / LPS, births=births, touched=touched, _game=game, _lo=lo, _om=om)


def withhold(samples):
    """The same samples with their commands withheld (an empty window)."""
    out = []
    for s in samples:
        game, lo, om = s["_game"], s["_lo"], s["_om"]
        out.append(dict(s, act=game.ga.window(lo, lo, np.array([lo]), lo, om, static_pointers=True), has_act=False))
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
    N = fb["ent_mask"].shape[1]
    b = dict(fb=fb, tb=tb, ab=ab, fi=torch.tensor(fi, device=dev),
             hor=torch.tensor([s["hor_s"] for s in samples], dtype=torch.float32, device=dev),
             has_act=torch.tensor([s["has_act"] for s in samples], device=dev),
             pidx=pad([s["pidx"] for s in samples], N).to(dev),
             touched=pad([s["touched"] for s in samples], N).to(dev),
             births=float(np.mean([s["births"] for s in samples])))
    b["tgt"] = {k: pad([s["tgt"][k] for s in samples], N).to(dev) for k in TARGETS}
    b["now"] = {k: pad([s["now"][k] for s in samples], N).to(dev) for k in TARGETS if k != "exit"}
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/worldmodel/obs")
    ap.add_argument("--actions_dir", default="data/worldmodel/actions")
    ap.add_argument("--splits", default="data/worldmodel/splits.json")
    ap.add_argument("--max_s", type=float, default=2.5, help="longest horizon in seconds")
    ap.add_argument("--eval_frames", default="1,4,8,14", help="horizons (in frames) evaluated")
    ap.add_argument("--frames", type=int, default=16, help="present frames per batch")
    ap.add_argument("--hor", type=int, default=2, help="horizons per present frame")
    ap.add_argument("--act_drop", type=float, default=0.5, help="share of samples with commands withheld")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--enc_depth", type=int, default=6)
    ap.add_argument("--pred_depth", type=int, default=4)
    ap.add_argument("--d_z", type=int, default=32)
    ap.add_argument("--prior_k", type=int, default=1, help="mixture components of the prior")
    ap.add_argument("--beta", type=float, default=0.05, help="weight per nat of KL above --free_nats")
    ap.add_argument("--beta_warmup", type=int, default=300)
    ap.add_argument("--free_nats", type=float, default=0.5, help="KL per (sample, player) below this is free")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--rotate_every", type=int, default=50)
    ap.add_argument("--rotate", type=int, default=2)
    ap.add_argument("--val_games", type=int, default=3)
    ap.add_argument("--val_states", type=int, default=48, help="states per (set, horizon) in evaluation")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--max_games", type=int, default=0)
    ap.add_argument("--no_ckpt", action="store_true", help="no activation checkpointing in the encoder")
    ap.add_argument("--out", default="data/worldmodel/runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--eval_only", action="store_true")
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
    eval_paths = {"seen-games": train_paths[:args.val_games],
                  "unseen-games": available(split["held_out_games"])[:args.val_games],
                  "unseen-maps": available(split["held_out_map_games"])[:args.val_games]}
    run = os.path.dirname(os.path.abspath(args.resume)) if args.resume else \
        os.path.join(args.out, time.strftime("direct_%Y%m%d_%H%M%S"))
    voc = json.load(open(os.path.join(run, "vocab.json"))) if args.resume else vocab_mod.build(train_paths[:60])
    v = vocab_mod.Vocab(voc)
    av = ActionVocab()
    load = lambda p: Game(p, args.actions_dir, v, av)
    t0 = time.time()
    eval_games = {name: [load(p) for p in ps] for name, ps in eval_paths.items() if ps}
    pool = GamePool(train_paths, min(args.pool, len(train_paths)), load, rng,
                    names=ck.get("pool") if ck is not None else None)
    loaded = [x for _, x in pool.games] + [x for gs in eval_games.values() for x in gs]
    step_loops = collections.Counter(x.g.step for x in loaded).most_common(1)[0][0]
    assert all(x.g.step == step_loops for x in loaded), "mixed frame rates"
    frame_s = step_loops / LPS
    k_max = max(1, int(round(args.max_s / frame_s)))
    eval_k = [int(x) for x in args.eval_frames.split(",")]
    print(f"loaded {len(loaded)} games in {time.time() - t0:.0f}s", flush=True)

    os.makedirs(run, exist_ok=True)
    if not args.resume:
        json.dump(dict(vars(args), step_loops=step_loops, k_max=k_max, n_train=len(train_paths),
                       eval={k: [os.path.basename(p) for p in ps] for k, ps in eval_paths.items()}),
                  open(os.path.join(run, "config.json"), "w"), indent=1)
        json.dump(voc, open(os.path.join(run, "vocab.json"), "w"))
    print(f"{len(train_paths)} train games (pool {len(pool.games)}) | eval: "
          + ", ".join(f"{k} {len(v_)} games" for k, v_ in eval_games.items())
          + f" | frame {frame_s:.3f}s, horizons 1..{k_max} frames (eval {eval_k}) | batch {args.frames} frames x {args.hor} horizons"
          f" | commands withheld on {args.act_drop:.0%} | z: d_z {args.d_z}, prior x{args.prior_k}, beta {args.beta}"
          f" (warmup {args.beta_warmup}), free {args.free_nats} nats | {run}", flush=True)

    enc = dm.RoPEEncoder(v, args.d, args.enc_depth).to(dev)
    enc.grad_ckpt = not args.no_ckpt
    aenc = ActionEncoder(enc, av, d=args.d).to(dev)
    pred = dm.DirectPredictor(args.d, args.pred_depth).to(dev)
    probes = dm.Probes(args.d, v.n_abilities).to(dev)
    post = InverseModel(args.d, 2 * args.d_z).to(dev)
    prior = dm.PriorHead(3 * args.d, args.d_z, args.prior_k).to(dev)
    player_emb = nn.Embedding(8, args.d).to(dev)
    withheld = nn.Embedding(2, args.d).to(dev)  # commands given / withheld, for the prior
    zup = nn.Linear(args.d_z, args.d).to(dev)
    modules = dict(encoder=enc, actions=aenc, predictor=pred, probes=probes, posterior=post, prior=prior,
                   player_emb=player_emb, withheld=withheld, zup=zup)
    params = [p for m in modules.values() for p in m.parameters()]
    n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    print(f"encoder {n(enc):.2f}M, action encoder {n(aenc):.2f}M, predictor {n(pred):.2f}M, probes {n(probes):.2f}M, "
          f"posterior {n(post):.2f}M, prior {n(prior):.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    warm = 200
    lr_at = lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))
    start = 0
    if ck is not None:
        for k, m in modules.items():
            m.load_state_dict(ck[k])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        rng.set_state(ck["np_rng"])
        print(f"resuming {run} at step {start}", flush=True)
    for g_ in opt.param_groups:
        g_.setdefault("initial_lr", args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at, last_epoch=start - 1)

    def draw_batch(games, n_frames, n_hor, rng, ks=None, with_actions=None):
        """Samples grouped by present frame (collate_samples relies on the grouping)."""
        out = []
        for gi in rng.randint(len(games), size=n_frames):
            game = games[gi]
            hi = len(game.g) - k_max - 1
            if hi < 1:
                continue
            t = rng.randint(0, hi + 1)
            om = game.g.owner_map("observer", augment=True, rng=rng)
            frame = game.g.frame(t, om)  # one frame object per present frame: collate_samples groups by identity
            for j in range(n_hor):
                k = int(ks[j]) if ks is not None else rng.randint(1, k_max + 1)
                wa = bool(with_actions) if with_actions is not None else bool(rng.rand() >= args.act_drop)
                out.append(make_sample(game, t, k, rng, wa, om, frame))
        return out

    def forward(b, z_from="post"):
        """One pass. z_from: post (a posterior sample in training, its mean otherwise), prior
        (a sample) or prior_mean. Returns the probes' logits, the targets' mask and the latent terms."""
        fb, tb, ab = b["fb"], b["tb"], b["ab"]
        fi = b["fi"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = enc(fb)
            tokens, mask, ang = e["tokens"][fi], e["mask"][fi], e["ang"][fi]
            P = e["P"]
            B, T, d = tokens.shape
            summaries, s_mask, h = aenc(ab, horizon_s=b["hor"])
            s_mask = s_mask[:, 0, :P] & fb["ply_mask"][fi]
            summ = summaries[:, 0, :P] * b["has_act"][:, None, None].to(summaries.dtype)
            # prior: present + commands (or "withheld"); posterior: present + future
            ply = e["players"][fi] + player_emb.weight[:P][None].to(tokens.dtype)
            wh = withheld(b["has_act"].long())[:, None].to(tokens.dtype)
            pin = torch.cat([e["cls"][fi][:, None].expand(-1, P, -1), ply, summ.to(tokens.dtype) + wh], -1)
            pri = prior(pin)
            kl = tokens.new_zeros(B, P).float()
            if z_from == "post" or enc.training:
                ef = enc(tb)
                post_mu, post_lv = lat.split_gauss(post(e["tokens"][fi], ef["tokens"], mask, ef["mask"], P, horizon_s=b["hor"]))
                zs = lat.rsample(post_mu, post_lv)
                kl = lat.kl_mc(zs, post_mu, post_lv, pri)
                z = zs if enc.training else post_mu
            if z_from == "prior":
                z = lat.mix_sample(pri)
            elif z_from == "prior_mean":
                z = lat.mix_mean(pri)
            zt = zup(z.float()).to(tokens.dtype)                                      # [B, P, d]
            ztok = torch.zeros_like(tokens)
            ztok[:, 1:1 + P] = zt
            pidx = b["pidx"]
            owned = pidx >= 0
            ztok[:, 1 + P:][owned] = zt[torch.nonzero(owned, as_tuple=True)[0], pidx[owned]]
            x = tokens + ztok
            hh = pred(x, mask, ang, b["hor"], h, ab["act_mask"] & b["has_act"][:, None])
            out = probes(hh[:, 1 + P:])
        m = s_mask.float()
        return out, fb["ent_mask"][fi], dict(kl=(kl * m).sum() / m.sum().clamp(min=1),
                                            kl_charged=(kl.clamp(min=args.free_nats) * m).sum() / m.sum().clamp(min=1))

    def metrics(out, b, mask):
        """Per-probe errors of the model's point estimates and of persistence, over real
        entities (exit) / entities present at t+k (the rest)."""
        est = probes.point(out)
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
        r["pos err touched"] = (avg(dist(est["dx"], est["dy"]), touched), avg(dist(now["dx"], now["dy"]), touched))
        r["touched share"] = (float(touched.sum() / present.sum().clamp(min=1)), 0.0)
        r["hp mae"] = (avg((est["hp"] - tgt["hp"]).abs(), present), avg((now["hp"] - tgt["hp"]).abs(), present))
        r["shield mae"] = (avg((est["sh"] - tgt["sh"]).abs(), present), avg((now["sh"] - tgt["sh"]).abs(), present))
        r["order acc"] = (avg((est["order"] == tgt["order"]).float(), present), avg((now["order"] == tgt["order"]).float(), present))
        r["moving share"] = (float(moved.sum() / present.sum().clamp(min=1)), 0.0)
        return r

    @torch.no_grad()
    def evaluate(step):
        for m in modules.values():
            m.eval()
        for name, games in eval_games.items():
            for k in eval_k:
                erng = np.random.RandomState(1234 + k)
                rows = {}
                births = []
                batches = [draw_batch(games, args.frames, 1, erng, ks=[k], with_actions=True)
                           for _ in range(max(1, args.val_states // args.frames))]
                for wa in (True, False):  # the same states, with their commands and without
                    acc = collections.defaultdict(lambda: collections.defaultdict(list))
                    kls = []
                    for samples in batches:
                        b = collate_samples(samples if wa else withhold(samples), dev)
                        if wa:
                            births.append(b["births"])
                        for zf in ("post", "prior", "prior_mean"):
                            out, mask, lt = forward(b, z_from=zf)
                            if zf == "post":
                                kls.append(float(lt["kl"]))
                            for key, (val, base) in metrics(out, b, mask).items():
                                acc[key][zf].append(val)
                                acc[key]["persist"].append(base)
                    rows[wa] = (acc, float(np.mean(kls)))
                hs = k * frame_s
                keys = ("exit acc", "died recall", "pos err", "pos err moving", "pos err touched", "hp mae", "shield mae", "order acc")
                cols = ("post", "prior", "prior_mean", "persist")
                moving = np.mean(rows[True][0]["moving share"]["post"])
                touched = np.mean(rows[True][0]["touched share"]["post"])
                print(f"[{name} step {step}] k={k} ({hs:.2f}s) births/window {np.mean(births):.1f} | moving {moving:.0%} | touched by a command {touched:.0%}\n"
                      f"   {'':16s} with commands: KL {rows[True][1]:5.2f} nats          | commands withheld: KL {rows[False][1]:5.2f} nats\n"
                      f"   {'':16s} " + "  ".join(f"{c[:6]:>6s}" for c in cols) + "  | " + "  ".join(f"{c[:6]:>6s}" for c in cols), flush=True)
                for key in keys:
                    cells = []
                    for wa in (True, False):
                        acc = rows[wa][0]
                        cells.append("  ".join(f"{np.mean(acc[key][c]):6.3f}" if not (key == "died recall" and c == "persist")
                                               else f"{'':6s}" for c in cols))
                    print(f"   {key:16s} " + "  | ".join(cells), flush=True)
        for m in modules.values():
            m.train()

    def save(step, tag):
        torch.save(dict(step=step, args=vars(args), opt=opt.state_dict(), pool=pool.names(), np_rng=rng.get_state(),
                        **{k: m.state_dict() for k, m in modules.items()}), os.path.join(run, f"model_{tag}.pt"))

    if args.eval_only:
        evaluate(start)
        return

    for m in modules.values():
        m.train()
    hist = collections.defaultdict(list)
    t_last = time.time()
    for step in range(start, args.steps):
        if step and step % args.rotate_every == 0:
            pool.rotate(args.rotate)
        games = [x for _, x in pool.games]
        t_data = time.time()
        b = collate_samples(draw_batch(games, args.frames, args.hor, rng), dev)
        t_data = time.time() - t_data
        out, mask, lt = forward(b)
        L = probes.losses(out, b["tgt"], mask)
        beta_t = args.beta * min(1.0, step / max(1, args.beta_warmup))
        loss = sum(L.values()) + beta_t * lt["kl_charged"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        hist["loss"].append(loss.item())
        hist["kl"].append(float(lt["kl"]))
        hist["gn"].append(float(gn))
        hist["data_s"].append(t_data)
        for k_, v_ in L.items():
            hist[k_].append(float(v_))
        if (step + 1) % 25 == 0:
            avg = {k_: np.mean(v_[-25:]) for k_, v_ in hist.items()}
            dt = time.time() - t_last
            t_last = time.time()
            print(f"step {step + 1} loss {avg['loss']:.3f} | exit {avg['exit']:.3f} dx {avg['dx']:.3f} dy {avg['dy']:.3f} "
                  f"hp {avg['hp']:.3f} sh {avg['sh']:.3f} en {avg['en']:.3f} build {avg['build']:.3f} order {avg['order']:.3f} "
                  f"| KL {avg['kl']:.2f} (beta {beta_t:.3f}) | grad {avg['gn']:.2f} | {dt / 25:.2f}s/step (data {avg['data_s']:.2f}s) "
                  f"| mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
            with open(os.path.join(run, "log.jsonl"), "a") as f:
                f.write(json.dumps(dict(step=step + 1, **{k_: float(v_) for k_, v_ in avg.items()})) + "\n")
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            evaluate(step + 1)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            save(step + 1, "latest")
    print("done", flush=True)


if __name__ == "__main__":
    main()
