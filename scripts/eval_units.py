"""Plain-units evaluation of a window world model checkpoint.

For each held-out set, the average error (mean absolute error, in game units) of what the
probes read:

    representation   probe on the real frame's encoding vs that frame's truth
                     ("can the encoding even tell how many workers there are?")
    forecast +h      probe on the predicted encoding of t + h vs the truth at t + h,
                     next to copying the present (probe on frame t's encoding vs truth at t + h)

    python scripts/eval_units.py data/worldmodel/runs/window_20261019_201707
"""

import argparse
import collections
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train_window as tw  # noqa: E402  (same windows and target construction as training)
from cosmicstar.worldmodel import vocab as vocab_mod  # noqa: E402
from cosmicstar.worldmodel.action_encoder import ActionEncoder  # noqa: E402
from cosmicstar.worldmodel.action_tokens import ActionVocab, GameActions, collate_actions  # noqa: E402
from cosmicstar.worldmodel.encoder import StateEncoder  # noqa: E402
from cosmicstar.worldmodel.predictor import slots_of  # noqa: E402
from cosmicstar.worldmodel.probes import Probes, collate_targets  # noqa: E402
from cosmicstar.worldmodel.tokenize import GameFrames, collate  # noqa: E402
from cosmicstar.worldmodel.window_predictor import WindowPredictor  # noqa: E402

# probe outputs -> game units (inverse of tokenize.probe_targets / player stats scaling)
PLY = [("workers", lambda y: np.expm1(y)), ("army units", lambda y: np.expm1(y)),
       ("structures", lambda y: np.expm1(y)), ("army hp+shields", lambda y: np.expm1(y) * 100),
       ("mineral income/min", lambda y: np.expm1(y * 10)), ("gas income/min", lambda y: np.expm1(y * 10)),
       ("supply used", lambda y: y * 200), ("army value", lambda y: np.expm1(y * 10))]
GRID = [("units per cell", lambda y: np.expm1(y)), ("army units per cell", lambda y: np.expm1(y)),
        ("structures per cell", lambda y: np.expm1(y))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="model_final.pt")
    ap.add_argument("--horizons_s", default="10,30")
    ap.add_argument("--examples", type=int, default=0, help="also print this many example rows per set")
    ap.add_argument("--sets", default="", help="comma-separated eval sets (default: all)")
    ap.add_argument("--probe_file", default="", help="probe_<name>.pt from probe_frozen.py (default: the run's own probe)")
    args = ap.parse_args()
    dev = "cuda"
    ck = torch.load(os.path.join(args.run, args.ckpt), map_location=dev, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    v = vocab_mod.Vocab(json.load(open(os.path.join(args.run, "vocab.json"))))
    av = None if a.no_actions else ActionVocab()
    enc = StateEncoder(v).to(dev).eval()
    enc.load_state_dict(ck["encoder"])
    tenc = copy.deepcopy(enc)
    tenc.load_state_dict(ck["target"])
    pred = WindowPredictor(depth=a.depth, horizons=ck["horizons"], residual=a.residual).to(dev).eval()
    pred.load_state_dict(ck["predictor"])
    aenc = None
    if av is not None:
        aenc = ActionEncoder(enc, av).to(dev).eval()
        aenc.load_state_dict(ck["actions"])
    if args.probe_file:  # a readout trained on frozen encodings (probe_frozen.py)
        pf = torch.load(os.path.join(args.run, args.probe_file), map_location=dev, weights_only=False)
        probes = Probes(**pf["config"]).to(dev).eval()
        probes.load_state_dict(pf["state"])
    else:  # the probe the run trained alongside the model
        probes = Probes(hidden="legacy" if ck["probes"]["heads.ply.0.weight"].shape[0] == 7 else 512).to(dev).eval()
        probes.load_state_dict(ck["probes"])
    spec = tw.WindowSpec(a.window_s, a.frame_stride, [float(x) for x in a.horizons_s.split(",")], a.long_sub, ck["step_loops"])
    T = spec.T
    want = {max(1, int(round(float(h) / spec.frame_s))) for h in args.horizons_s.split(",")} & set(spec.horizons)

    def load(p):
        g = GameFrames(p, v)
        return g, (None if av is None else GameActions(os.path.join(a.actions_dir, os.path.basename(p)), g, av)), None

    @torch.no_grad()
    def run_batch(wins):
        B = len(wins)
        fb = tw.to(collate([f for w in wins for f in w["frames"]]), dev)
        times = torch.from_numpy(np.stack([w["times"] for w in wins])).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            slots, mask = slots_of(enc(fb))
            S = slots.shape[1]
            slots, mask = slots.view(B, T, S, -1), mask.view(B, T, S)
            tw_slots, _ = slots_of(tenc(fb))
            present = F.layer_norm(tw_slots.float(), tw_slots.shape[-1:]).view(B, T, S, -1)
            act = {}
            if aenc is not None:
                summaries, s_mask, _ = aenc(tw.to(collate_actions([w["act"] for w in wins]), dev))
                act = dict(act=summaries, act_mask=s_mask)
            preds = pred(slots, mask, times, base=present, **act)
            extra = [f for w in wins for f in w["tframes"]]
            flat = present.flatten(0, 1)
            if extra:
                es, _ = slots_of(tenc(tw.to(collate(extra), dev)))
                flat = torch.cat([flat, F.layer_norm(es.float(), es.shape[-1:])])
        P = S - 1 - pred.tags.cells.shape[0]
        return preds, present, flat, P

    @torch.no_grad()
    def read(slots, P):
        o = probes(slots, P)
        return o["ply"].float().cpu().numpy(), o["grid"].float().cpu().numpy()

    for name, files in cfg["eval"].items():
        if args.sets and name not in args.sets.split(","):
            continue
        games = [load(os.path.join(a.data, f)) for f in files]
        if not games:
            continue
        vrng = np.random.RandomState(123)
        wins, i = [], 0
        while len(wins) < a.val_windows and i < 50 * a.val_windows:
            w = tw.make_window(games[i % len(games)], spec, vrng, augment=False)
            i += 1
            if w is not None:
                wins.append(w)
        err = collections.defaultdict(list)  # (what, quantity) -> absolute errors
        examples = []  # per (window, position, player): real/read now, real/forecast at each horizon
        for j in range(0, len(wins), a.windows):
            batch = wins[j:j + a.windows]
            preds, present, flat, P = run_batch(batch)
            B = len(batch)
            truth = collate_targets([t for w in batch for t in w["probe"][:T]] + [t for w in batch for t in w["probe"][T:]], P, dev)
            tp, tg = truth["ply"].cpu().numpy(), truth["grid"].cpu().numpy()
            pm = truth["ply_mask"].cpu().numpy()
            offsets, o = [], B * T
            for w in batch:
                offsets.append(o)
                o += len(w["tframes"])

            def add(what, rp, rg, tidx, real_ply_mask):
                for q, (qn, f) in enumerate(PLY):
                    e = np.abs(f(rp[..., q]) - f(tp[tidx][..., q]))[real_ply_mask]
                    err[(what, qn)].append(e)
                for q, (qn, f) in enumerate(GRID):
                    err[(what, qn)].append(np.abs(f(rg[..., q]) - f(tg[tidx][..., q])).ravel())

            # representation: real window frames, read back
            rp, rg = read(present.flatten(0, 1), P)
            idx = np.arange(B * T)
            add("representation", rp, rg, idx, pm[idx])
            if args.examples:
                hs = sorted(want)
                fc = {h: read(preds[h].flatten(0, 1), P)[0] for h in hs}  # forecasts at every position
                for b, w in enumerate(batch):
                    for k in range(T):
                        tgi = [w["tgt"][h][k] for h in hs]
                        if min(tgi) < 0:
                            continue  # need a real future frame for every horizon
                        flat_t = [b * T + t if t < T else offsets[b] + t - T for t in tgi]
                        for p in range(P):
                            if not pm[b * T + k, p]:
                                continue
                            row = dict(window=j + b, t=round(float(w["times"][k]), 1), player=p)
                            for q, (qn, f) in enumerate(PLY):
                                if qn not in ("workers", "army units", "supply used", "army value"):
                                    continue
                                row[(qn, "real now")] = f(tp[b * T + k, p, q])
                                row[(qn, "read now")] = f(rp[b * T + k, p, q])
                                for h, ft in zip(hs, flat_t):
                                    row[(qn, f"real +{spec.seconds(h):g}s")] = f(tp[ft, p, q])
                                    row[(qn, f"forecast +{spec.seconds(h):g}s")] = f(fc[h][b * T + k, p, q])
                            examples.append(row)
            for h in sorted(want):
                tgt = np.concatenate([np.where(w["tgt"][h] < 0, -1, np.where(w["tgt"][h] < T, b * T + w["tgt"][h],
                                                                              offsets[b] + w["tgt"][h] - T))
                                      for b, w in enumerate(batch)])
                ok = tgt >= 0
                rp, rg = read(preds[h].flatten(0, 1)[torch.from_numpy(ok).to(dev)], P)
                add(f"forecast +{spec.seconds(h)}s", rp, rg, tgt[ok], pm[tgt[ok]])
                rp, rg = read(present.flatten(0, 1)[torch.from_numpy(ok).to(dev)], P)
                add(f"copy +{spec.seconds(h)}s", rp, rg, tgt[ok], pm[tgt[ok]])
        print(f"\n=== {name} ({len(games)} games, {len(wins)} windows): mean absolute error, game units")
        quantities = [q for q, _ in PLY] + [q for q, _ in GRID]
        cols = ["representation"] + [f"{k} +{spec.seconds(h)}s" for h in sorted(want) for k in ("copy", "forecast")]
        print(f"{'':22s}" + "".join(f"{c:>20s}" for c in cols))
        for q in quantities:
            print(f"{q:22s}" + "".join(f"{np.concatenate(err[(c, q)]).mean():20.2f}" for c in cols))
        if args.examples and examples:
            ex_rng = np.random.RandomState(0)
            pick = ex_rng.choice(len(examples), min(args.examples, len(examples)), replace=False)
            for qn in ("workers", "army units", "supply used", "army value"):
                keys = [k for k in examples[0] if isinstance(k, tuple) and k[0] == qn]
                print(f"\n  {qn}: examples (window, game second in window, player)")
                print("  " + f"{'':14s}" + "".join(f"{k[1]:>14s}" for k in keys))
                for i in sorted(pick, key=lambda i: (examples[i]["window"], examples[i]["t"], examples[i]["player"])):
                    r = examples[i]
                    print("  " + f"w{r['window']} {r['t']:5.1f}s p{r['player']}".ljust(14) + "".join(f"{r[k]:14.1f}" for k in keys))
                vals = {k: np.array([r[k] for r in examples]) for k in keys}
                real_now = vals[(qn, "real now")]
                print("  " + f"{'spread p10-p90':14s}" + "".join(f"{np.percentile(vals[k], 10):6.0f}-{np.percentile(vals[k], 90):<7.0f}" for k in keys))
                for k in keys:
                    if k[1].startswith("forecast"):
                        real = vals[(qn, k[1].replace("forecast", "real"))]
                        e = vals[k] - real
                        copy_e = real_now - real
                        print(f"    {k[1]:16s} error vs real: median {np.median(e):+6.1f}, p10..p90 {np.percentile(e, 10):+6.1f}..{np.percentile(e, 90):+6.1f}"
                              f"   | copying now: p10..p90 {np.percentile(copy_e, 10):+6.1f}..{np.percentile(copy_e, 90):+6.1f}")
                e = vals[(qn, "read now")] - real_now
                print(f"    {'read now':16s} error vs real: median {np.median(e):+6.1f}, p10..p90 {np.percentile(e, 10):+6.1f}..{np.percentile(e, 90):+6.1f}")
                # MSE (and RMSE, same units as the quantity) of each estimate against the real value
                print(f"    {'MSE (RMSE)':16s} vs the real value at the same time")
                mse = lambda x, y: float(np.mean((x - y) ** 2))
                read_now = vals[(qn, "read now")]
                print(f"      {'read now':34s} {mse(read_now, real_now):9.1f} ({mse(read_now, real_now) ** 0.5:6.1f})")
                for k in keys:
                    if k[1].startswith("forecast"):
                        hz = k[1].split()[-1]
                        real = vals[(qn, f"real {hz}")]
                        for label, x in ((f"forecast {hz}", vals[k]), (f"read now, used as the {hz} forecast", read_now),
                                         (f"real now, used as the {hz} forecast", real_now)):
                            print(f"      {label:34s} {mse(x, real):9.1f} ({mse(x, real) ** 0.5:6.1f})")


if __name__ == "__main__":
    main()
