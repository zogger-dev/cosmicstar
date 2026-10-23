"""Per-cell spatial check: does the real interval's chunk beat the adjacent chunks at predicting
WHERE things will be?

The embedding-level loss averages over ~260 slots while an interval's commands change a few
cells, so a real effect can hide in the average. Here frozen readers (per-cell unit / army /
structure counts, trained on the run's target encoder as in probe_frozen.py) decode the
predictor's output for held-out pairs driven by the real commands, by the adjacent intervals'
commands (same game, one interval before and one after) and by no commands, next to copying
the present and the ceiling (the reader on the true future's encoding). Errors are in game
units, over all cells and over the cells that change between t and t+D, where an interval's
specific commands should show; and per pair, how often the real commands beat each alternative
on the changing cells.

    python scripts/spatial_check.py data/worldmodel/runs/pairs_20261021_154208 --d_s 5,15
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train_window import to  # noqa: E402
from train_pairs import make_pair  # noqa: E402
from probe_frozen import VARIANTS, encode as encode_frames, fit  # noqa: E402
from cosmicstar.worldmodel import vocab as vocab_mod  # noqa: E402
from cosmicstar.worldmodel.action_encoder import ActionEncoder  # noqa: E402
from cosmicstar.worldmodel.action_tokens import ActionVocab, GameActions, collate_actions  # noqa: E402
from cosmicstar.worldmodel.encoder import StateEncoder  # noqa: E402
from cosmicstar.worldmodel.predictor import slots_of  # noqa: E402
from cosmicstar.worldmodel.probes import Probes  # noqa: E402
from cosmicstar.worldmodel.tokenize import GameFrames, collate, probe_targets  # noqa: E402
from cosmicstar.worldmodel.window_predictor import WindowPredictor  # noqa: E402

LPS = 22.4
QUANT = (("units per cell", 0), ("army units per cell", 1), ("structures per cell", 2))
SOURCES = ("real", -1, 1, "none")


def load_models(run, ckpt, dev):
    ck = torch.load(os.path.join(run, ckpt), map_location=dev, weights_only=False)
    a = ck["args"]
    v = vocab_mod.Vocab(json.load(open(os.path.join(run, "vocab.json"))))
    enc = StateEncoder(v).to(dev).eval()
    enc.load_state_dict(ck["encoder"], strict=False)
    enc.use_orders = any(k.startswith("ord_in") for k in ck["encoder"]) and not a.get("no_orders")
    tenc = copy.deepcopy(enc)
    tenc.load_state_dict(ck["target"], strict=False)
    tenc.use_orders = enc.use_orders and not a.get("target_no_orders")
    pred = WindowPredictor(depth=a["depth"], horizons=("var",), residual=True, horizon_cond=True).to(dev).eval()
    pred.load_state_dict(ck["predictor"])
    av = ActionVocab()
    aenc = ActionEncoder(enc, av).to(dev).eval()
    aenc.load_state_dict(ck["actions"])
    zproj = nn.Linear(enc.d, a["d_z"]).to(dev).eval()
    zproj.load_state_dict(ck["zproj"])
    zup = nn.Linear(a["d_z"], enc.d).to(dev).eval()
    zup.load_state_dict(ck["zup"])
    return dict(a=a, v=v, av=av, enc=enc, tenc=tenc, pred=pred, aenc=aenc, zproj=zproj, zup=zup, step_loops=ck["step_loops"])


@torch.no_grad()
def predict(M, pairs, d_s, source, dev):
    """The predictor's output for t + D, [B, S, d], driven by the real commands ("real"), an
    adjacent interval's commands (-1 / +1) or nothing ("none"); with the present and the true
    future in target space, and P."""
    a = M["a"]
    C, B = a["context"], len(pairs)
    fb = to(collate([f for p in pairs for f in p["frames"]]), dev)
    tb = to(collate([p["target"] for p in pairs]), dev)
    hor = torch.full((B,), float(d_s), device=dev)
    times = torch.from_numpy(np.stack([p["times"] for p in pairs])).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        slots, mask = slots_of(M["enc"](fb))
        S = slots.shape[1]
        slots, mask = slots.view(B, C, S, -1), mask.view(B, C, S)
        pw, _ = slots_of(M["tenc"](fb))
        present = F.layer_norm(pw.float(), pw.shape[-1:]).view(B, C, S, -1)
        tg, _ = slots_of(M["tenc"](tb))
        future = F.layer_norm(tg.float(), tg.shape[-1:])
        acts = [p["near"][source] if source in (-1, 1) else p["act"] for p in pairs]
        ab = to(collate_actions(acts), dev)
        summ, s_mask, _ = M["aenc"](ab, horizon_s=hor)
        act = M["zup"](M["zproj"](summ[:, 0]))
        A = torch.zeros(B, C, act.shape[1], act.shape[2], device=dev, dtype=act.dtype)
        A[:, -1] = act
        am = torch.zeros(B, C, act.shape[1], dtype=torch.bool, device=dev)
        if source != "none":
            am[:, -1] = s_mask[:, 0]
        out = M["pred"](slots, mask, times, act=A, act_mask=am, base=present, horizon_s=hor)["var"][:, -1]
    return out.float(), present[:, -1], future, fb["ply_mask"].shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="model_final.pt")
    ap.add_argument("--d_s", default="5,15")
    ap.add_argument("--pairs", type=int, default=96, help="pairs per (set, horizon)")
    ap.add_argument("--train_games", type=int, default=40, help="games the readers are trained on")
    ap.add_argument("--per_game", type=int, default=200)
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    dev = "cuda"
    rng = np.random.RandomState(0)
    M = load_models(args.run, args.ckpt, dev)
    a = M["a"]
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    frame_s = M["step_loops"] / LPS
    print(f"{args.run}: orders {'off' if not M['enc'].use_orders else 'input only' if not M['tenc'].use_orders else 'on'}, "
          f"context {a['context']}", flush=True)

    # readers: per-cell counts from the target encoder's grid slots
    split = json.load(open(a["splits"]))
    seen = set(cfg["eval"].get("seen-games", []))
    train_files = [os.path.join(a["data"], f) for f in sorted(split["train"])
                   if os.path.exists(os.path.join(a["data"], f)) and f not in seen][:args.train_games]
    print(f"readers: encoding {len(train_files)} training games x {args.per_game} frames ...", flush=True)
    data = encode_frames(M["tenc"], [GameFrames(p, M["v"]) for p in train_files], args.per_game, 32, rng, dev)
    torch.manual_seed(0)
    reader = fit(Probes(**VARIANTS["mlp1024x3"]).to(dev), data, steps=args.steps).eval()
    del data

    def load(p):
        g = GameFrames(p, M["v"])
        return g, GameActions(os.path.join(a["actions_dir"], os.path.basename(p)), g, M["av"])

    eval_games = {name: [load(os.path.join(a["data"], f)) for f in files]
                  for name, files in cfg["eval"].items() if name != "seen-games" and files}
    for d_s in [float(x) for x in args.d_s.split(",")]:
        d_frames = max(1, int(round(d_s / frame_s)))
        d_eff = d_frames * frame_s
        for name, gs in eval_games.items():
            vr = np.random.RandomState(123)
            pairs, meta, i = [], [], 0
            while len(pairs) < args.pairs and i < 30 * args.pairs:
                gi = i % len(gs)
                i += 1
                p = make_pair(gs[gi], d_frames, a["context"], a["context_stride"], vr, augment=False, near=(-1, 1))
                if p is not None and len(p["near"]) == 2:  # both adjacent intervals exist
                    pairs.append(p)
                    meta.append((gi, p["t"]))
            t_now = np.stack([probe_targets(gs[gi][0], t)["grid"] for gi, t in meta])             # [N, G*G, 4] log1p
            t_fut = np.stack([probe_targets(gs[gi][0], t + d_frames)["grid"] for gi, t in meta])
            dec, emb = {}, {}  # readers' decoding (clipped: they extrapolate on predicted vectors) and per-cell embedding error
            lo, hi = 0.0, float(np.log1p(300))
            with torch.no_grad():
                for src in SOURCES:
                    outs, ceil, errs, cerr = [], [], [], []
                    for b0 in range(0, len(pairs), 32):
                        out, present, future, P = predict(M, pairs[b0:b0 + 32], d_eff, src, dev)
                        outs.append(reader.heads["grid"](out[:, 1 + P:]).clamp(lo, hi).cpu().numpy())
                        errs.append((out[:, 1 + P:] - future[:, 1 + P:]).pow(2).mean(-1).sqrt().cpu().numpy())  # [B, G*G]
                        if src == "real":
                            ceil.append(reader.heads["grid"](future[:, 1 + P:]).clamp(lo, hi).cpu().numpy())
                            cerr.append((present[:, 1 + P:] - future[:, 1 + P:]).pow(2).mean(-1).sqrt().cpu().numpy())
                    dec[src], emb[src] = np.concatenate(outs), np.concatenate(errs)
                    if src == "real":
                        dec["ceiling"], emb["copy"] = np.concatenate(ceil), np.concatenate(cerr)
            print(f"\n=== {name}, D = {d_eff:.1f}s, {len(pairs)} pairs", flush=True)
            # the model's own space, per cell: distance of each source's predicted grid slot to the
            # true future's slot, over all cells and over the cells whose true unit count changes
            truth_u, now_u = np.expm1(t_fut[..., 0]), np.expm1(t_now[..., 0])
            chg = np.abs(truth_u - now_u) >= 1.0
            has = chg.any(1)
            names_e = {"real": "real", -1: "adj-1", 1: "adj+1", "none": "none", "copy": "copy"}
            order_e = ("real", -1, 1, "none", "copy")
            print(f"  embedding error per grid cell (changing cells {chg.mean():.1%}; copy = the present's own slot)")
            print("    all cells:      " + "  ".join(f"{names_e[k]} {emb[k].mean():.3f}" for k in order_e))
            print("    changing cells: " + "  ".join(f"{names_e[k]} {emb[k][chg].mean():.3f}" for k in order_e))
            e_real = np.array([emb["real"][n][chg[n]].mean() for n in np.where(has)[0]])
            print("    real beats, on changing cells, per pair: " + "  ".join(
                f"{names_e[k]} {np.mean(e_real < np.array([emb[k][n][chg[n]].mean() for n in np.where(has)[0]])):.0%}"
                for k in (-1, 1, "none", "copy")), flush=True)
            for qname, q in QUANT:
                truth, now = np.expm1(t_fut[..., q]), np.expm1(t_now[..., q])
                changing = np.abs(truth - now) >= 1.0
                share = changing.mean()
                preds = {k: np.expm1(v[..., q]) for k, v in dec.items()}
                preds["copy"] = now
                order = ("real", -1, 1, "none", "copy", "ceiling")
                names = {"real": "real", -1: "adj-1", 1: "adj+1", "none": "none", "copy": "copy", "ceiling": "ceiling"}
                rmse_all = {k: np.sqrt(np.mean((preds[k] - truth) ** 2)) for k in order}
                rmse_chg = {k: np.sqrt(np.mean(((preds[k] - truth) ** 2)[changing])) if changing.any() else float("nan") for k in order}
                # per pair, on its changing cells: does the real chunk beat the alternative?
                has = changing.any(1)
                wins = {}
                for k in (-1, 1, "none", "copy"):
                    e_real = np.array([np.mean(((preds["real"][n] - truth[n]) ** 2)[changing[n]]) for n in np.where(has)[0]])
                    e_alt = np.array([np.mean(((preds[k][n] - truth[n]) ** 2)[changing[n]]) for n in np.where(has)[0]])
                    wins[k] = float(np.mean(e_real < e_alt)) if len(e_real) else float("nan")
                print(f"  {qname}, via readers (clipped; rough on predicted vectors) (spread {truth.std():.2f}; changing cells {share:.1%})")
                print("    RMSE all cells:      " + "  ".join(f"{names[k]} {rmse_all[k]:.3f}" for k in order))
                print("    RMSE changing cells: " + "  ".join(f"{names[k]} {rmse_chg[k]:.3f}" for k in order))
                print("    real beats, on changing cells, per pair: " + "  ".join(f"{names[k]} {wins[k]:.0%}" for k in (-1, 1, "none", "copy")), flush=True)
    print("\nchance for 'real beats X' is 50%; ceiling = the readers on the true future's encoding; copy = the present's true counts")


if __name__ == "__main__":
    main()
