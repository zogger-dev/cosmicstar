"""Train probes from scratch on a checkpoint's frozen encodings, and compare readouts.

The probes trained during a run chase a moving target encoder, and the first runs' probes
were also a 7-unit bottleneck (probes.readout "legacy"). This trains fresh readouts on fixed
encodings of training-game frames (target encoder, layer-normed, as in training) and reports
how well each reads the real frames of held-out games, in game units (RMSE):

    linear       one linear layer
    mlp512       256 -> 512 -> out
    mlp1024x3    256 -> 1024 -> 1024 -> out
    run probe    the probe the run trained alongside the model (for reference)

Each trained readout is saved to <run>/probe_<name>.pt (eval_units.py --probe_file).

    python scripts/probe_frozen.py data/worldmodel/runs/window_20261019_201707

--context compares, at the same moments and against the same truth, a single frame's encoding
with the predictor's state after attending over the window so far. For a run on player views
(--pov_dir) the truth is the observer's (everything), per player row split by relation to the
viewer (me / ally / enemy), and the readings are:

    own view     the player's own frame, encoded (what they see right now)
    context      the predictor's state at that frame (their view + their actions so far)
    pred now     the predictor's horizon-0 output (its guess of the full present, target space)
    observer     the observer's frame, encoded (everything: the ceiling)
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

import train_window as tw  # noqa: E402  (same windows as training)
from eval_units import GRID, PLY  # noqa: E402  (game-unit conversions)
from cosmicstar.worldmodel import vocab as vocab_mod  # noqa: E402
from cosmicstar.worldmodel.action_encoder import ActionEncoder  # noqa: E402
from cosmicstar.worldmodel.action_tokens import ActionVocab, GameActions, collate_actions  # noqa: E402
from cosmicstar.worldmodel.encoder import StateEncoder  # noqa: E402
from cosmicstar.worldmodel.window_predictor import WindowPredictor  # noqa: E402
from cosmicstar.worldmodel.predictor import slots_of  # noqa: E402
from cosmicstar.worldmodel.probes import KINDS, Probes  # noqa: E402
from cosmicstar.worldmodel.tokenize import GameFrames, collate, probe_targets  # noqa: E402

VARIANTS = dict(linear=dict(hidden=512, layers=1), mlp512=dict(hidden=512, layers=2),
                mlp1024x3=dict(hidden=1024, layers=3))


@torch.no_grad()
def encode(tenc, games, per_game, cells, rng, dev, batch=192):
    """Frozen encodings + truth for frames sampled uniformly from each game. Grid: `cells`
    random cells per frame. Returns tensors on dev."""
    out = dict(cls=[], ply=[], ply_mask=[], grid=[], t_cls=[], t_ply=[], t_grid=[])
    for g in games:
        om = g.owner_map("observer")
        fs = np.sort(rng.choice(len(g), min(per_game, len(g)), replace=False))
        for i in range(0, len(fs), batch):
            chunk = fs[i:i + batch]
            frames = [g.frame(int(f), om) for f in chunk]
            b = {k: x.to(dev) for k, x in collate(frames).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s, m = slots_of(tenc(b))
            s = F.layer_norm(s.float(), s.shape[-1:])
            P = b["ply_mask"].shape[1]
            truth = [probe_targets(g, int(f)) for f in chunk]
            cell = torch.from_numpy(rng.randint(0, s.shape[1] - 1 - P, size=(len(chunk), cells))).to(dev)
            out["cls"].append(s[:, 0].half())
            out["ply"].append(s[:, 1:1 + P].half())
            out["ply_mask"].append(b["ply_mask"])
            grid = s[:, 1 + P:]
            out["grid"].append(torch.gather(grid, 1, cell[..., None].expand(-1, -1, grid.shape[-1])).half())
            out["t_cls"].append(torch.from_numpy(np.stack([t["cls"] for t in truth])).to(dev))
            out["t_ply"].append(torch.from_numpy(np.stack([t["ply"] for t in truth])).to(dev))
            tg = torch.from_numpy(np.stack([t["grid"] for t in truth])).to(dev)
            out["t_grid"].append(torch.gather(tg, 1, cell[..., None].expand(-1, -1, tg.shape[-1])))
    return {k: torch.cat(v) for k, v in out.items()}  # all games here have the same player count


def fit(probe, data, steps=4000, bs=1024, lr=1e-3, dev="cuda"):
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    N = len(data["cls"])
    for _ in range(steps):
        i = torch.randint(0, N, (bs,), device=dev)
        loss = 0
        loss = loss + F.mse_loss(probe.heads["cls"](data["cls"][i].float()), data["t_cls"][i])
        m = data["ply_mask"][i]
        e = F.mse_loss(probe.heads["ply"](data["ply"][i].float()), data["t_ply"][i], reduction="none").mean(-1)
        loss = loss + (e * m).sum() / m.sum().clamp(min=1)
        loss = loss + F.mse_loss(probe.heads["grid"](data["grid"][i].float()), data["t_grid"][i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return probe


@torch.no_grad()
def rmse_units(probe, data):
    """RMSE in game units per player quantity and grid quantity."""
    m = data["ply_mask"].cpu().numpy()
    rp = probe.heads["ply"](data["ply"].float()).cpu().numpy()
    tp = data["t_ply"].cpu().numpy()
    rg = probe.heads["grid"](data["grid"].float()).cpu().numpy()
    tg = data["t_grid"].cpu().numpy()
    out = {}
    for q, (qn, f) in enumerate(PLY):
        out[qn] = float(np.sqrt(np.mean((f(rp[..., q]) - f(tp[..., q]))[m] ** 2)))
    for q, (qn, f) in enumerate(GRID):
        out[qn] = float(np.sqrt(np.mean((f(rg[..., q]) - f(tg[..., q])) ** 2)))
    return out


@torch.no_grad()
def rmse_rows(probe, data, rows):
    """RMSE in game units of the per-player quantities over the player rows in `rows` [N, P]."""
    rp = probe.heads["ply"](data["ply"].float()).cpu().numpy()
    tp = data["t_ply"].cpu().numpy()
    return {qn: float(np.sqrt(np.mean((f(rp[..., q]) - f(tp[..., q]))[rows] ** 2))) if rows.any() else float("nan")
            for q, (qn, f) in enumerate(PLY)}


@torch.no_grad()
def encode_windows(models, games, spec, n_windows, cells, rng, dev, every=4, batch=4):
    """For moments in the second half of windows (>= half a window of history behind them):
    the single-frame encoding ("frame": target encoder, no context) and the predictor's state
    after attending over the window so far ("context"), with the same truth for both."""
    enc, tenc, pred, aenc = models
    T = spec.T
    ks = list(range(T // 2, T, every))
    names = ("frame", "context")
    if games and games[0][2]:  # player views: also the horizon-0 guess and the observer's frame
        names = ("frame", "context", "pred now", "observer")
    reps = {r: collections.defaultdict(list) for r in names}
    wins = []
    for game in games:
        made = tries = 0
        while made < n_windows and tries < 5 * n_windows:
            tries += 1
            w = tw.make_window(game, spec, rng, augment=True)
            if w is not None:
                wins.append(w)
                made += 1
    for i in range(0, len(wins), batch):
        ws = wins[i:i + batch]
        B = len(ws)
        fb = tw.to(collate([f for w in ws for f in w["frames"]]), dev)
        times = torch.from_numpy(np.stack([w["times"] for w in ws])).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            slots, mask = slots_of(enc(fb))
            S = slots.shape[1]
            slots, mask = slots.view(B, T, S, -1), mask.view(B, T, S)
            tws, _ = slots_of(tenc(fb))
            present = F.layer_norm(tws.float(), tws.shape[-1:]).view(B, T, S, -1)
            act = {}
            if aenc is not None:
                summaries, s_mask, _ = aenc(tw.to(collate_actions([w["act"] for w in ws]), dev))
                act = dict(act=summaries, act_mask=s_mask)
            preds, state = pred(slots, mask, times, base=present, return_state=True, **act)
        state = F.layer_norm(state.float(), state.shape[-1:])
        P = S - 1 - pred.tags.cells.shape[0]
        truth = [w["probe"][k] for w in ws for k in ks]
        pmask = fb["ply_mask"].view(B, T, -1)[:, ks].flatten(0, 1)
        prel = fb["ply_rel"].view(B, T, -1)[:, ks].flatten(0, 1)
        cell = torch.from_numpy(rng.randint(0, S - 1 - P, size=(B * len(ks), cells))).to(dev)
        t_grid = torch.from_numpy(np.stack([t["grid"] for t in truth])).to(dev)
        readings = [("frame", present), ("context", state)]
        if "observer" in reps:
            obs, _ = slots_of(tenc(tw.to(collate([f for w in ws for f in w["wframes"]]), dev)))
            readings += [("pred now", preds[0].float()), ("observer", F.layer_norm(obs.float(), obs.shape[-1:]).view(B, T, S, -1))]
        for name, X in readings:
            X = X[:, ks].flatten(0, 1)                                                  # [B*K, S, d]
            r = reps[name]
            r["cls"].append(X[:, 0].half())
            r["ply"].append(X[:, 1:1 + P].half())
            r["ply_mask"].append(pmask)
            r["ply_rel"].append(prel)
            grid = X[:, 1 + P:]
            r["grid"].append(torch.gather(grid, 1, cell[..., None].expand(-1, -1, grid.shape[-1])).half())
            r["t_cls"].append(torch.from_numpy(np.stack([t["cls"] for t in truth])).to(dev))
            r["t_ply"].append(torch.from_numpy(np.stack([t["ply"] for t in truth])).to(dev))
            r["t_grid"].append(torch.gather(t_grid, 1, cell[..., None].expand(-1, -1, t_grid.shape[-1])))
    return {name: {k: torch.cat(v) for k, v in r.items()} for name, r in reps.items()}


def load_models(ck, v, dev):
    a = argparse.Namespace(**ck["args"])
    av = None if a.no_actions else ActionVocab()
    enc = StateEncoder(v).to(dev).eval()
    enc.load_state_dict(ck["encoder"], strict=False)  # older checkpoints lack the visibility / order paths
    enc.use_orders = any(k.startswith("ord_in") for k in ck["encoder"])
    tenc = copy.deepcopy(enc)
    tenc.load_state_dict(ck["target"], strict=False)
    pred = WindowPredictor(depth=a.depth, horizons=ck["horizons"], residual=a.residual).to(dev).eval()
    pred.load_state_dict(ck["predictor"])
    aenc = None
    if av is not None:
        aenc = ActionEncoder(enc, av).to(dev).eval()
        aenc.load_state_dict(ck["actions"])
    return (enc, tenc, pred, aenc), av


def context_mode(args, ck, cfg, v, dev, rng):
    """Single-frame encoding vs the predictor's state with context, same moments, same truth."""
    a = ck["args"]
    models, av = load_models(ck, v, dev)
    spec = tw.WindowSpec(a["window_s"], a["frame_stride"], [float(x) for x in a["horizons_s"].split(",")],
                         a["long_sub"], ck["step_loops"])

    def load(p):
        g = GameFrames(p, v)
        acts = os.path.join(a["actions_dir"], os.path.basename(p))
        if not a.get("pov_dir"):
            return g, (None if av is None else GameActions(acts, g, av)), None
        views = {}
        stem = os.path.basename(p)[:-4] + "_p"
        for q in sorted(os.listdir(a["pov_dir"])):
            if q.startswith(stem) and q.endswith(".npz") and not q.endswith(".tmp.npz"):
                gv = GameFrames(os.path.join(a["pov_dir"], q), v)
                views[int(q[len(stem):-4])] = (gv, None if av is None else GameActions(acts, gv, av))
        return g, None, views

    split = json.load(open(a["splits"]))
    pov = a.get("pov_dir")
    have = lambda names: [os.path.join(a["data"], f) for f in sorted(names) if os.path.exists(os.path.join(a["data"], f))
                          and (not pov or any(q.startswith(f[:-4] + "_p") for q in os.listdir(pov)))]
    train_files = [p for p in have(split["train"]) if os.path.basename(p) not in set(cfg["eval"].get("seen-games", []))]
    print(f"windows from {len(train_files)} training games x {args.windows_per_game} ...", flush=True)
    train = encode_windows(models, [load(p) for p in train_files], spec, args.windows_per_game, args.cells, rng, dev)
    held = {name: encode_windows(models, [load(os.path.join(a["data"], f)) for f in files], spec,
                                 args.windows_per_game, args.cells, rng, dev)
            for name, files in cfg["eval"].items() if name != "seen-games" and files}
    print(f"{len(train['frame']['cls'])} training moments (each with >= {spec.T // 2 * spec.frame_s:.0f}s of history)", flush=True)
    if a.get("pov_dir"):
        return pov_report(args, train, held, dev)
    readers = {}
    for rep in ("frame", "context"):
        for name in ("mlp512", "mlp1024x3"):
            torch.manual_seed(0)
            p = fit(Probes(**VARIANTS[name]).to(dev), train[rep], steps=args.steps)
            readers[(rep, name)] = p.eval()
    quantities = ["workers", "army units", "supply used", "army value", "mineral income/min", "gas income/min",
                  "army hp+shields", "units per cell", "army units per cell"]
    for set_name, data in held.items():
        res = {k: rmse_units(p, data[k[0]]) for k, p in readers.items()}
        tp, m = data["frame"]["t_ply"].cpu().numpy(), data["frame"]["ply_mask"].cpu().numpy()
        tg = data["frame"]["t_grid"].cpu().numpy()
        spread = {qn: float(np.std(f(tp[..., q])[m])) for q, (qn, f) in enumerate(PLY)}
        spread.update({qn: float(np.std(f(tg[..., q]))) for q, (qn, f) in enumerate(GRID)})
        print(f"\n=== {set_name}: RMSE reading the same moments, game units (lower is better)")
        print(f"{'':20s}" + "".join(f"{r + ' ' + n:>20s}" for r, n in readers) + f"{'(spread)':>11s}")
        for qn in quantities:
            print(f"{qn:20s}" + "".join(f"{res[k][qn]:20.2f}" for k in readers) + f"{spread[qn]:11.2f}")
    print("\nframe = one frame's encoding, no context; context = the predictor's state at that frame after "
          "attending over the window so far (frames + actions)")


def pov_report(args, train, held, dev):
    """Player-view runs: the observer's truth read from each reading, per relation to the viewer."""
    readers = {}
    for rep in train:
        torch.manual_seed(0)
        readers[rep] = fit(Probes(**VARIANTS["mlp1024x3"]).to(dev), train[rep], steps=args.steps).eval()
    names = {"frame": "own view", "context": "context", "pred now": "pred now", "observer": "observer"}
    for set_name, data in held.items():
        rel = data["frame"]["ply_rel"].cpu().numpy()
        m = data["frame"]["ply_mask"].cpu().numpy()
        tp = data["frame"]["t_ply"].cpu().numpy()
        print(f"\n=== {set_name}: RMSE of the true (observer) value, game units (lower is better); "
              f"{len(rel)} moments, each a player's view with >= half a window of history")
        for r, rn in ((1, "me"), (2, "ally"), (3, "enemy")):
            rows = m & (rel == r)
            if not rows.any():
                continue
            res = {k: rmse_rows(p, data[k], rows) for k, p in readers.items()}
            print(f"  -- {rn} ({rows.sum()} player rows)")
            print(f"  {'':20s}" + "".join(f"{names[k]:>12s}" for k in readers) + f"{'(spread)':>11s}")
            for q, (qn, f) in enumerate(PLY):
                spread = float(np.std(f(tp[..., q])[rows]))
                print(f"  {qn:20s}" + "".join(f"{res[k][qn]:12.2f}" for k in readers) + f"{spread:11.2f}")
        tg = data["frame"]["t_grid"].cpu().numpy()
        print("  -- map grid (cells)")
        res = {k: rmse_units(p, data[k]) for k, p in readers.items()}
        for q, (qn, f) in enumerate(GRID):
            print(f"  {qn:20s}" + "".join(f"{res[k][qn]:12.2f}" for k in readers) + f"{float(np.std(f(tg[..., q]))):11.2f}")
    print("\nown view = the player's frame alone; context = the predictor's state after the window so far (their "
          "frames + their actions); pred now = its horizon-0 guess of the full present; observer = the full frame "
          "(ceiling). spread = std of the true value (always guessing the average scores about this)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="model_final.pt")
    ap.add_argument("--per_game", type=int, default=400, help="frames sampled per game")
    ap.add_argument("--cells", type=int, default=32, help="grid cells sampled per frame")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--context", action="store_true", help="compare single-frame encodings with the predictor's context state")
    ap.add_argument("--windows_per_game", type=int, default=30, help="--context: windows sampled per game")
    args = ap.parse_args()
    if args.context:
        dev = "cuda"
        ck = torch.load(os.path.join(args.run, args.ckpt), map_location=dev, weights_only=False)
        cfg = json.load(open(os.path.join(args.run, "config.json")))
        v = vocab_mod.Vocab(json.load(open(os.path.join(args.run, "vocab.json"))))
        context_mode(args, ck, cfg, v, dev, np.random.RandomState(0))
        return
    dev = "cuda"
    rng = np.random.RandomState(0)
    ck = torch.load(os.path.join(args.run, args.ckpt), map_location=dev, weights_only=False)
    a = ck["args"]
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    v = vocab_mod.Vocab(json.load(open(os.path.join(args.run, "vocab.json"))))
    tenc = StateEncoder(v).to(dev).eval()
    tenc.load_state_dict(ck["target"], strict=False)  # older checkpoints lack the visibility / order paths
    tenc.use_orders = any(k.startswith("ord_in") for k in ck["target"])
    split = json.load(open(a["splits"]))
    have = lambda names: [os.path.join(a["data"], f) for f in sorted(names) if os.path.exists(os.path.join(a["data"], f))]
    train_files = [p for p in have(split["train"]) if os.path.basename(p) not in set(cfg["eval"].get("seen-games", []))]
    print(f"encoding {len(train_files)} training games x {args.per_game} frames ...", flush=True)
    train = encode(tenc, [GameFrames(p, v) for p in train_files], args.per_game, args.cells, rng, dev)
    held = {name: encode(tenc, [GameFrames(os.path.join(a["data"], f), v) for f in files], args.per_game, args.cells, rng, dev)
            for name, files in cfg["eval"].items() if name != "seen-games" and files}
    print(f"{len(train['cls'])} training frames; held-out: " + ", ".join(f"{k} {len(d['cls'])}" for k, d in held.items()), flush=True)

    readers = {}
    try:  # the run's own probe, if it reads the same quantities as today's targets
        legacy = ck["probes"]["heads.ply.0.weight"].shape[0] == ck["probes"]["heads.ply.2.weight"].shape[1]
        run_probe = Probes(hidden="legacy" if legacy else 512).to(dev).eval()
        run_probe.load_state_dict(ck["probes"])
        readers["run probe"] = run_probe
    except (RuntimeError, KeyError, TypeError):
        print("(the run's own probe was trained on different quantities, or the run had none; left out)", flush=True)
    for name, conf in VARIANTS.items():
        torch.manual_seed(0)
        p = fit(Probes(**conf).to(dev), train, steps=args.steps)
        p.eval()
        readers[name] = p
        torch.save(dict(config=conf, state=p.state_dict()), os.path.join(args.run, f"probe_{name}.pt"))

    quantities = ["workers", "army units", "supply used", "army value", "mineral income/min", "gas income/min", "army hp+shields", "units per cell", "army units per cell"]
    for set_name, data in held.items():
        spread = {}
        tp = data["t_ply"].cpu().numpy()
        m = data["ply_mask"].cpu().numpy()
        for q, (qn, f) in enumerate(PLY):
            spread[qn] = float(np.std(f(tp[..., q])[m]))
        tg = data["t_grid"].cpu().numpy()
        for q, (qn, f) in enumerate(GRID):
            spread[qn] = float(np.std(f(tg[..., q])))
        print(f"\n=== {set_name}: RMSE reading real frames, game units (lower is better)")
        print(f"{'':20s}" + "".join(f"{r:>12s}" for r in readers) + f"{'(spread)':>12s}")
        res = {r: rmse_units(p, data) for r, p in readers.items()}
        for qn in quantities:
            print(f"{qn:20s}" + "".join(f"{res[r][qn]:12.2f}" for r in readers) + f"{spread[qn]:12.2f}")
    print("\n(spread = standard deviation of the real value: a reader that always guesses the average scores about this)")


if __name__ == "__main__":
    main()
