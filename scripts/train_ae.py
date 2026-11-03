"""Pretraining the grid latent as an autoencoder (2026-11-01, DESIGN_grid_latent.md "Pretraining"):
one game frame -> the grid latent (worldmodel/quad_latent.QuadEncoder: 16 x 16 cells + CLS and
players, the same on every map) -> the whole frame back, gradients fully on. No dynamics, no z,
no actions. If it trains, a latent exists that holds the game state in a decodable form.

Data (2026-11-01): moments come from the frame cache (worldmodel/frame_cache.py, built by
scripts/build_frame_cache.py): ready frames on disk from every extracted game, each moment in both
player views and the combined view, read at random (~10 ms per batch). The earlier in-memory pool of
12 whole games grew to ~47 GB and was OOM-killed.

Decoded:
- entities: Poisson counts per (cell, owner, type); one query per entity (true counts in
  training) reading its offset in the cell, its numbers (tokenize.entity_features) and its order;
  paired with the truth by a Hungarian assignment over each frame (order-free; see grid_latent.hungarian);
- the map at 1 map unit, each real cell from its latent cell's vector and its offset (MapDecoder; a
  random sample of --map_cells on-map cells per frame in training, all of them in evaluation): the
  static layers (pathable, placeable, height) and one view (visible, seen, creep) per player the
  frame holds: the viewer's in a player's frame, both in a combined frame;
- the players' numbers (stats, extras) and upgrades from their latent tokens.

Evaluation (player views and the combined view of seen / unseen games and unseen maps): position
error of the matched entities, counts (precision / recall of the predicted counts, the error per
frame), HP, numbers, orders; map-layer accuracy and IoU; player numbers and upgrades.

    python scripts/train_ae.py --steps 20000
"""
import argparse
import collections
import json
import pickle
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import train_direct as td  # noqa: E402  (environment, compile settings, the game sets)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from cosmicstar.worldmodel import corpus, statefile  # noqa: E402
from cosmicstar.worldmodel import grid_latent as gl  # noqa: E402
from cosmicstar.worldmodel import vocab as vocab_mod  # noqa: E402
from cosmicstar.worldmodel.action_tokens import ActionVocab  # noqa: E402
from cosmicstar.worldmodel import frame_cache as fc  # noqa: E402
from cosmicstar.worldmodel.quad_latent import QuadEncoder  # noqa: E402
from cosmicstar.worldmodel import rare_eval as re_  # noqa: E402
from cosmicstar.worldmodel.data import Prefetch, to  # noqa: E402
from cosmicstar.worldmodel.encoder import mlp  # noqa: E402
from cosmicstar.worldmodel.tokenize import collate as collate_frames  # noqa: E402

ON_MAP = gl.STATIC_LAYERS.index("on_map")
# decoded static layers (on_map and playable are structure: on_map is the mask itself, playable ~1 in the crop)
STATIC_BIN = ("pathable", "placeable")
STATIC_OUT = [gl.STATIC_LAYERS.index(c) for c in (*STATIC_BIN, "height")]
HEIGHT = len(STATIC_BIN)                 # its output


def collate(frames):
    return gl.map_views(collate_frames(frames))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/worldmodel/frame_cache", help="the frame cache (scripts/build_frame_cache.py)")
    ap.add_argument("--frames", type=int, default=48, help="frames per batch (~0.43 GB of GPU memory each)")
    ap.add_argument("--obs_share", type=float, default=1 / 3, help="share of frames from the combined view")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--head_dim", type=int, default=32)
    ap.add_argument("--grid", type=int, default=16, help="latent cells per side")
    ap.add_argument("--full_blocks", type=int, default=3, help="full attention blocks at the latent's level")
    ap.add_argument("--dec_depth", type=int, default=2)
    ap.add_argument("--conv_blocks", type=int, default=3)
    ap.add_argument("--attn_blocks", type=int, default=3)
    ap.add_argument("--decomp_grad", type=float, default=0.0, help="decomposition check: share of its gradient reaching "
                    "the encoder (0: a probe; the targets are always stop-gradient)")
    ap.add_argument("--map_cells", type=int, default=4096, help="real map cells decoded per frame in training")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_frames", type=int, default=64, help="frames per (set, view) in evaluation")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--val_every", type=int, default=100, help="the training objective on fixed held-out batches, next to "
                    "the training average")
    ap.add_argument("--val_batches", type=int, default=2, help="held-out batches per validation set")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--no_ckpt", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--out", default="data/worldmodel/runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rare_eval", default="data/worldmodel/rare_eval.pkl", help="the rare-state evaluation frames "
                    "(scripts/build_rare_eval.py); skipped if missing")
    ap.add_argument("--balance_power", type=float, default=0.5, help="class weights count ** -power in the balanced "
                    "losses (1: fully balanced, 0.5: square root, 0: plain mean)")
    ap.add_argument("--resume", default="", help="a run directory: continue from its model_latest.pt (same run, same logs)")
    ap.add_argument("--eval_ckpt", default="", help="evaluate this checkpoint (a model_latest.pt) once and exit")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    gl.BALANCE_POWER = args.balance_power
    rng = np.random.RandomState(args.seed)
    dev = "cuda"

    # moments come from the frame cache (scripts/build_frame_cache.py): ready frames on disk, all games
    train = fc.FrameCache(args.cache, ["train"])
    held = {"unseen-games": fc.FrameCache(args.cache, ["held_out_games"]),
            "unseen-maps": fc.FrameCache(args.cache, ["held_out_map_games"])}
    v = fc.load_vocab(args.cache)
    av = ActionVocab()
    if args.resume:
        run = os.path.abspath(args.resume)
    elif args.eval_ckpt:  # results go to the checkpoint's run (tb_eval)
        run = os.path.dirname(os.path.abspath(args.eval_ckpt))
    else:
        run = os.path.join(args.out, time.strftime("ae_%Y%m%d_%H%M%S"))
    os.makedirs(run, exist_ok=True)
    print(f"frame cache {args.cache}: {len(train)} train moments from {len(train.games)} games | held out: "
          + ", ".join(f"{k} {len(c)} moments / {len(c.games)} games" for k, c in held.items()) + f" | {run}", flush=True)
    if not (args.resume or args.eval_ckpt):
        json.dump(dict(vars(args), n_train_games=len(train.games), n_train_moments=len(train)),
                  open(os.path.join(run, "config.json"), "w"), indent=1)
        json.dump(json.load(open(os.path.join(args.cache, "vocab.json"))), open(os.path.join(run, "vocab.json"), "w"))

    heads = args.d // args.head_dim
    enc = QuadEncoder(v, av.n_families, args.d, S=args.grid, conv_blocks=args.conv_blocks, attn_blocks=args.attn_blocks,
                      full_blocks=args.full_blocks, heads=heads).to(dev)
    enc.grad_ckpt = not args.no_ckpt
    edec = gl.StateEntityDecoder(args.d, v.n_units, v.n_abilities, args.dec_depth, heads).to(dev)
    mdec = gl.MapDecoder(args.d, n_out=len(STATIC_OUT)).to(dev)
    vdec = gl.MapDecoder(args.d, n_out=len(gl.VIEW_LAYERS), players=2).to(dev)
    probe = collate(train.read(0)[0][:1])  # the players' number and upgrade widths, from one frame
    n_pnum, n_ups = probe["ply_num"].shape[-1], probe["ply_ups"].shape[-1]
    pnum, pups = mlp(args.d, n_pnum).to(dev), mlp(args.d, n_ups).to(dev)
    if not args.no_compile:
        enc.compile_parts()
        for m in (edec, mdec, vdec):
            m.compile(dynamic=True)
    # learned uncertainty weights (Kendall et al.): loss = sum exp(-s_i) L_i + s_i over the loss groups
    log_var = nn.Parameter(torch.zeros(19, device=dev))
    uw = nn.ParameterDict(dict(log_var=log_var))
    dhead = gl.DecompHead(args.d, heads=heads).to(dev)
    modules = dict(encoder=enc, entity_dec=edec, map_dec=mdec, view_dec=vdec, player_num=pnum, player_ups=pups, loss_weights=uw, decomp=dhead)
    params = [p for m in modules.values() for p in m.parameters()]
    n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    print(f"encoder {n(enc):.2f}M, entity decoder {n(edec):.2f}M, map decoders {n(mdec) + n(vdec):.2f}M | latent "
          f"{args.grid}x{args.grid}x{args.d} + {1 + 2} globals", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    warm = 500
    start = 0
    if args.resume:  # the model, optimizer and step of the run's last checkpoint
        ck = torch.load(os.path.join(run, "model_latest.pt"), map_location=dev, weights_only=False)
        for k_, m_ in modules.items():
            m_.load_state_dict(ck[k_])
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"])
        print(f"resuming {run} at step {start}", flush=True)
    for g_ in opt.param_groups:
        g_.setdefault("initial_lr", args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))), last_epoch=start - 1)
    # a resumed run drops the points its previous process logged past the checkpoint
    tb = SummaryWriter(os.path.join(run, "tb_eval" if args.eval_ckpt else "tb"), purge_step=start or None)
    if not start:
        tb.add_text("config", "```\n" + json.dumps(vars(args), indent=1) + "\n```")
    tag = lambda s_: re.sub(r"[^A-Za-z0-9_.-]+", "_", s_)

    def draw_triples(cache, n_moments, rng):
        """Moments from the cache, each in both player views and the combined view, [player 0's view,
        player 1's view, combined] per moment; returns the frames and, per moment, the frame
        indices of the combined view and of player 0's / player 1's own view."""
        frames, trip = [], []
        for m in cache.sample(rng, n_moments):
            base = len(frames)
            frames += m
            trip.append((base + 2, base, base + 1))
        return frames, trip

    def draw(cache, n_frames, rng, view):
        """One frame per moment: a player's view (either) or the combined view."""
        return [m[2] if view == "obs" else m[rng.randint(2)] for m in cache.sample(rng, n_frames)]

    LOSS_GROUPS = ("off", "type", "owner", "cont", "bin", "cat", "q_abil", "q_tgt", "q_unit", "q_prog", "q_len", "p_abil",
                   "p_prog", "parent", "count", "map", "height", "pnum", "pups")

    def decomp(e, trip):
        """The decomposition check: each player's latent from the combined latent of the same moment.
        Returns the loss and, per player, the fraction of variance unexplained (and that of copying the
        combined latent, the baseline)."""
        o, p0, p1 = trip[:, 0], trip[:, 1], trip[:, 2]
        lat = e["tokens"]
        x = gl.grad_scale(lat[o], args.decomp_grad)
        M = o.shape[0]
        xx = torch.cat([x, x])
        pl = torch.cat([torch.zeros(M, dtype=torch.long, device=dev), torch.ones(M, dtype=torch.long, device=dev)])
        pred = dhead(xx, torch.cat([e["mask"][o]] * 2), torch.cat([e["ang"][o]] * 2), pl).float()
        tgt = torch.cat([lat[p0], lat[p1]]).detach().float()
        w = torch.cat([e["mask"][o] & e["mask"][p0], e["mask"][o] & e["mask"][p1]]).float()
        sq = lambda a: (((a - tgt) ** 2).mean(-1) * w).sum() / w.sum().clamp(min=1)
        loss = sq(pred)
        var = sq((tgt * w[..., None]).sum((0, 1)) / w.sum().clamp(min=1))   # around the mean latent token
        return loss, dict(fvu=float(loss / var.clamp(min=1e-8)), fvu_copy=float(sq(torch.cat([lat[o]] * 2).float()) / var.clamp(min=1e-8)))

    def step_fn(fb, train=True, trip=None):
        """Encode, decode, the raw losses; in evaluation also the metrics."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = enc(fb)
            lat, lmask, cs = e["tokens"], e["mask"], e["cell_slice"]
            tg = gl.cell_targets(fb, args.grid, v.n_units, v.n_abilities)
            L, out, pair = gl.state_losses(edec, lat, lmask, cs, tg)
            # the map: a sample of cells per frame (training), all of them (evaluation)
            cell, off = gl.canvas_cells(fb, args.grid)
            grid = fb["grid"]
            B, C, H, W = grid.shape
            flat = grid.view(B, C, H * W)
            idx = torch.randint(0, H * W, (B, args.map_cells), device=dev) if train else \
                torch.arange(H * W, device=dev)[None].expand(B, -1)
            cells = lat[:, cs[0]:cs[1]]
            cell_k, off_k = torch.gather(cell, 1, idx), torch.gather(off, 1, idx[..., None].expand(-1, -1, 2))
            mo = mdec(cells, cell_k, off_k)
            tgt_map = torch.gather(flat, 2, idx[:, None].expand(-1, C, -1)).transpose(1, 2).float()   # [B, K, C]
            onm = tgt_map[..., ON_MAP] > 0
            w = onm.float()
            tgt_map = tgt_map[..., STATIC_OUT]
            # one player's view per row: a player's frame has one, a combined frame two
            vf, nv = fb["view_frame"], len(gl.VIEW_LAYERS)
            vo = vdec(cells[vf], cell_k[vf], off_k[vf], fb["view_player"])                       # [M, K, 3]
            tgt_view = torch.gather(fb["view_grid"].view(vf.shape[0], nv, H * W), 2,
                                    idx[vf][:, None].expand(-1, nv, -1)).transpose(1, 2).float()
            bce = lambda p_, t_: F.binary_cross_entropy_with_logits(p_, t_, reduction="none").sum(-1)
            wv = w[vf]
            L["map"] = ((bce(mo[..., :HEIGHT], tgt_map[..., :HEIGHT]) * w).sum() / w.sum().clamp(min=1)
                        + (bce(vo, tgt_view) * wv).sum() / wv.sum().clamp(min=1)) / (HEIGHT + nv)
            L["height"] = (((mo[..., HEIGHT] - tgt_map[..., HEIGHT]) ** 2) * w).sum() / w.sum().clamp(min=1)
            P = e["P"]
            pt = lat[:, 1:1 + P]
            pm = fb["ply_mask"].float()
            pn, ups_logit = pnum(pt).float(), pups(pt).float()
            L["pnum"] = (F.smooth_l1_loss(pn, fb["ply_num"].float(), reduction="none").mean(-1) * pm).sum() / pm.sum().clamp(min=1)
            L["pups"] = (F.binary_cross_entropy_with_logits(ups_logit, fb["ply_ups"].float(), reduction="none").mean(-1) * pm).sum() / pm.sum().clamp(min=1)
            dm_ = None
            if trip is not None:
                L["decomp"], dm_ = decomp(e, trip)
        if train:
            return L, None
        m = metrics(fb, tg, out, pair, mo, tgt_map, onm, vo, tgt_view, pn, ups_logit, pm)
        if dm_ is not None:
            m["decomp fvu"], m["decomp fvu copy"] = dm_["fvu"], dm_["fvu_copy"]
        return L, m

    @torch.no_grad()
    def metrics(fb, tg, out, pair, mo, tgt_map, onm, vo, tgt_view, pn, ups_logit, pm):
        """Per attribute: position error, type / owner accuracy, each continuous number's error, each
        flag's F1, each categorical's accuracy, the order queue, each map layer's IoU, the players."""
        ip, it, ok = pair
        B, N = tg["mask"].shape
        fl = lambda x: x.reshape(-1, *x.shape[2:])
        P_, T_ = (lambda k: fl(out[k])[ip]), (lambda k: fl(tg[k])[it])
        okf = ok.float()
        nok = okf.sum().clamp(min=1)
        avg = lambda x: float((x.float() * okf).sum() / nok)
        m = {}
        err = (P_("off") - out["_off_t"]).norm(dim=-1) * tg["size"][it // N]
        m["pos err"] = avg(err)
        m["pos err p50"] = float(err[ok].median()) if ok.any() else 0.0
        m["pos within 0.5"] = avg(err < 0.5)
        m["type acc"] = avg(P_("type").argmax(-1) == T_("type"))
        m["owner acc"] = avg(P_("owner").argmax(-1) == T_("owner"))
        ce, ct = P_("cont"), T_("cont")
        for j, name in enumerate(gl.NUM_CONT):
            m[f"num {name} mae"] = avg((ce[:, j] - ct[:, j]).abs())
        be, bt = P_("bin") > 0, T_("bin") > 0.5
        for j, name in enumerate(gl.NUM_BIN):
            tp = float((be[:, j] & bt[:, j] & ok).sum())
            fp = float((be[:, j] & ~bt[:, j] & ok).sum())
            fn = float((~be[:, j] & bt[:, j] & ok).sum())
            m[f"flag {name} f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
        for k, cols in gl.NUM_CAT.items():  # accuracy, and the recall of each state (rare ones would hide in the accuracy)
            ph_, th_ = P_(f"cat_{k}").argmax(-1), T_(f"cat_{k}")
            m[f"cat {k} acc"] = avg(ph_ == th_)
            for c, cname in enumerate(cols + ("none",)):
                sel = (th_ == c) & ok
                m[f"cat {k}:{cname} recall"] = float(((ph_ == c) & sel).float().sum() / sel.float().sum().clamp(min=1))
                m[f"cat {k}:{cname} share"] = float(sel.float().sum() / nok)
        # type: balanced accuracy (the mean recall over the types present), next to the plain accuracy
        tp_, tt_ = P_("type").argmax(-1), T_("type")
        K_ = v.n_units
        hit_ = torch.zeros(K_, device=dev).index_add_(0, tt_, ((tp_ == tt_) & ok).float())
        cnt_ = torch.zeros(K_, device=dev).index_add_(0, tt_, okf)
        m["type balanced acc"] = float((hit_ / cnt_.clamp(min=1))[cnt_ > 0].mean()) if (cnt_ > 0).any() else 0.0
        qa, qt, qh = P_("q_abil").argmax(-1), T_("q_abil"), T_("q_has")
        m["queue order acc"] = float(((qa == qt) & qh & ok[:, None]).float().sum() / (qh & ok[:, None]).float().sum().clamp(min=1))
        m["queue length acc"] = avg((qa < v.n_abilities).sum(-1) == qh.sum(-1))
        m["queue len mae"] = avg((P_("q_len") - T_("q_len")).abs())
        pa, pt_, ph = P_("p_abil").argmax(-1), T_("p_abil"), T_("p_has")
        m["production acc"] = float(((pa == pt_) & ph & ok[:, None]).float().sum() / (ph & ok[:, None]).float().sum().clamp(min=1))
        m["production length acc"] = avg((pa < v.n_abilities).sum(-1) == ph.sum(-1))
        pr_hat, pr_t = P_("ptr").argmax(-1), out["_ptr_t"]
        has_par = (pr_t < tg["mask"].shape[1]) & ok
        m["container acc"] = float(((pr_hat == pr_t) & has_par).float().sum() / has_par.float().sum().clamp(min=1))
        m["no-container acc"] = float(((pr_hat == pr_t) & ~has_par & ok).float().sum() / (~has_par & ok).float().sum().clamp(min=1))
        ch, ct = out["count"].argmax(-1), tg["cell_count"]
        m["cell count acc"] = float((ch == ct).float().mean())
        m["cell count acc nonempty"] = float(((ch == ct) & (ct > 0)).float().sum() / (ct > 0).float().sum().clamp(min=1))
        m["count L1/frame"] = float((ch - ct).abs().sum(-1).float().mean())
        m["queue target mae"] = float(((P_("q_tgt") - T_("q_tgt")).abs().sum(-1) * (qh & ok[:, None])).sum()
                                      / (qh & ok[:, None]).float().sum().clamp(min=1))
        onv = onm[fb["view_frame"]]
        for names, po, to_, on in ((gl.VIEW_LAYERS, vo, tgt_view, onv), (STATIC_BIN, mo, tgt_map, onm)):
            for j, c in enumerate(names):
                p_, t_ = (po[..., j] > 0) & on, (to_[..., j] > 0.5) & on
                m[f"map {c} iou"] = float((p_ & t_).float().sum() / (p_ | t_).float().sum().clamp(min=1))
        m["map height mae"] = float(((mo[..., HEIGHT] - tgt_map[..., HEIGHT]).abs() * onm.float()).sum() / onm.float().sum().clamp(min=1))
        m["player num mae"] = float(((pn - fb["ply_num"].float()).abs().mean(-1) * pm).sum() / pm.sum().clamp(min=1))
        m["upgrades acc"] = float((((ups_logit > 0) == (fb["ply_ups"] > 0)).float().mean(-1) * pm).sum() / pm.sum().clamp(min=1))
        return m

    erng = np.random.RandomState(1234)
    rare = pickle.load(open(args.rare_eval, "rb")) if args.rare_eval and os.path.exists(args.rare_eval) else None
    eval_sets = {"seen-games": train, **{k: c for k, c in held.items() if len(c)}}
    eval_trips = {name: draw_triples(c, args.eval_frames // 3, erng) for name, c in eval_sets.items()}
    eval_frames = {(name, view): draw(c, args.eval_frames, erng, view) for name, c in eval_sets.items() for view in ("player", "obs")}

    @torch.no_grad()
    def evaluate(step):
        for m_ in modules.values():
            m_.eval()
        for (name, view), frames in eval_frames.items():
            acc = collections.defaultdict(list)
            for i in range(0, len(frames), 16):
                _, mm = step_fn(to(collate(frames[i:i + 16]), dev), train=False)
                for k_, x in mm.items():
                    acc[k_].append(x)
            label = name + ("" if view == "player" else " obs")
            avg = {k_: float(np.mean(x)) for k_, x in acc.items()}
            keys = ("pos err", "pos err p50", "pos within 0.5", "cell count acc nonempty", "count L1/frame", "type acc", "type balanced acc",
                    "owner acc", "queue order acc", "queue length acc", "queue len mae", "queue target mae", "production acc",
                    "production length acc", "container acc", "no-container acc", "map height mae", "player num mae",
                    "upgrades acc")
            print(f"[{label} step {step}] " + " | ".join(f"{k_} {avg[k_]:.3f}" for k_ in keys), flush=True)
            mp = [k_ for k_ in avg if k_.startswith("map ") and k_.endswith(" iou")]
            print(f"[{label} step {step}] map IoU " + " ".join(f"{k_[4:-4]} {avg[k_]:.3f}" for k_ in mp), flush=True)
            if (name, view) == ("unseen-maps", "player"):  # every attribute, once per evaluation
                for grp in ("num ", "flag ", "cat "):
                    ks = [k_ for k_ in avg if k_.startswith(grp)]
                    print(f"[{label} step {step}] " + " ".join(f"{k_[len(grp):]} {avg[k_]:.3f}" for k_ in ks), flush=True)
            for k_, x in avg.items():
                tb.add_scalar(f"{tag(label)}/{tag(k_)}", x, step)
        if rare is not None:  # rare entity states on frames chosen to contain them
            rc = re_.RareCounts()
            for i in range(0, len(rare["frames"]), 16):
                fb = to(collate(rare["frames"][i:i + 16]), dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    e = enc(fb)
                    tg = gl.cell_targets(fb, args.grid, v.n_units, v.n_abilities)
                    _, out, pair = gl.state_losses(edec, e["tokens"], e["mask"], e["cell_slice"], tg)
                rc.add(out, tg, pair)
            sm = rc.summary()
            print(f"[rare states step {step}] recall / precision / F1 (units): " + " | ".join(
                f"{k_} {r:.2f}/{p_:.2f}/{f1:.2f} ({n_})" for k_, (r, p_, f1, n_) in sm.items()), flush=True)
            for k_, (r, p_, f1, n_) in sm.items():
                tb.add_scalar(f"rare/{tag(k_)}_recall", r, step)
                tb.add_scalar(f"rare/{tag(k_)}_f1", f1, step)
        for name, (frames, trip) in eval_trips.items():  # the decomposition check, per split
            fv, fc = [], []
            for i in range(0, len(trip), 5):
                chunk = trip[i:i + 5]
                lo = chunk[0][1]
                fb = to(collate(frames[lo:chunk[-1][0] + 1]), dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, dm_ = decomp(enc(fb), torch.tensor(chunk, device=dev) - lo)
                fv.append(dm_["fvu"])
                fc.append(dm_["fvu_copy"])
            print(f"[{name} step {step}] decomposition: player latents from the combined latent, unexplained variance "
                  f"{np.mean(fv):.3f} (copying the combined latent: {np.mean(fc):.3f})", flush=True)
            tb.add_scalar(f"{tag(name)}/decomp_fvu", float(np.mean(fv)), step)
            tb.add_scalar(f"{tag(name)}/decomp_fvu_copy", float(np.mean(fc)), step)
        tb.flush()
        for m_ in modules.values():
            m_.train()

    # validation: the reconstruction losses (unweighted sum over the groups) on fixed batches from the held-out
    # games (seen maps) and the held-out maps, every --val_every steps, next to the training average
    vrng = np.random.RandomState(4321)
    val_sets = {name: [draw_triples(c, args.frames // 3, vrng) for _ in range(args.val_batches)] for name, c in held.items() if len(c)}

    @torch.no_grad()
    def validate():
        for m_ in modules.values():
            m_.eval()
        out = {}
        for name, bs in val_sets.items():
            tot = []
            for frames, trip in bs:
                L, _ = step_fn(to(collate(frames), dev), train=True)
                tot.append(float(sum(L[k_] for k_ in LOSS_GROUPS)))
            out[name] = float(np.mean(tot))
        for m_ in modules.values():
            m_.train()
        return out

    def save(step):
        torch.save(dict(step=step, args=vars(args), opt=opt.state_dict(), **{k_: m_.state_dict() for k_, m_ in modules.items()}),
                   os.path.join(run, "model_latest.pt"))

    if args.eval_ckpt:  # one evaluation of a saved model, no training
        ck = torch.load(args.eval_ckpt, map_location=dev, weights_only=False)
        for k_, m_ in modules.items():
            if k_ in ck:
                m_.load_state_dict(ck[k_])
        evaluate(int(ck.get("step", 0)))
        return

    prng = np.random.RandomState(args.seed * 1000003 + start)

    def make_batch():
        t_ = time.time()
        frames, trip = draw_triples(train, args.frames // 3, prng)
        return to(collate(frames), dev), torch.tensor(trip, device=dev), time.time() - t_

    batches = Prefetch(make_batch, depth=2)
    recon_last = []
    hist = collections.defaultdict(list)
    t_last = time.time()
    for m_ in modules.values():
        m_.train()
    for step in range(start, args.steps):
        tw = time.time()
        fb, trip, t_build = batches.get()
        t_wait = time.time() - tw
        L, _ = step_fn(fb, trip=trip)
        lv = uw["log_var"]
        loss = sum(torch.exp(-lv[i]) * L[k_] + lv[i] for i, k_ in enumerate(LOSS_GROUPS)) + L["decomp"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        hist["loss"].append(loss.detach())
        hist["recon"].append(sum(L[k_] for k_ in LOSS_GROUPS).detach())  # the train-vs-validation loss
        hist["grad"].append(gn.detach())
        for k_, x in L.items():
            hist[k_].append(x.detach())
        for i, k_ in enumerate(LOSS_GROUPS):
            hist[f"w_{k_}"].append(torch.exp(-uw["log_var"][i]).detach())
        hist["wait"].append(t_wait)
        recon_last.append(hist["recon"][-1])
        del recon_last[:-25]
        if (step + 1) % 25 == 0:
            avg = {k_: float(torch.stack(x).float().mean()) if torch.is_tensor(x[0]) else float(np.mean(x)) for k_, x in hist.items()}
            hist.clear()
            dt = (time.time() - t_last) / 25
            t_last = time.time()
            print(f"step {step + 1} " + " ".join(f"{k_} {x:.4f}" for k_, x in avg.items() if k_ != "wait" and not k_.startswith("w_"))
                  + f" | {dt:.2f}s/step (waited {avg['wait']:.2f}s) | mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
            for k_, x in avg.items():
                tb.add_scalar(f"train/{k_}", x, step + 1)
            tb.add_scalar("perf/s_per_step", dt, step + 1)
            tb.add_scalar("train/lr", sched.get_last_lr()[0], step + 1)
        if val_sets and (step + 1) % args.val_every == 0:
            vl = validate()
            t_rec = float(np.mean([float(x) for x in recon_last])) if recon_last else float("nan")
            tb.add_scalar("loss/train", t_rec, step + 1)
            for name, x in vl.items():
                tb.add_scalar(f"loss/val {name}", x, step + 1)
            print(f"[val step {step + 1}] loss train {t_rec:.3f} | " + " | ".join(f"{k_} {x:.3f}" for k_, x in vl.items()), flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            evaluate(step + 1)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            save(step + 1)
    tb.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
