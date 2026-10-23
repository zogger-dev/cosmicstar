"""Counterfactual action check: does the predictor use the action stream?

For held-out windows, the prediction made with the real actions is compared with predictions
made from K action chunks borrowed from other games (same viewer, so player codes line up) and
with no actions at all. If the predictor only learned how the game drifts on its own, the real
and borrowed actions score the same. Per horizon and evaluation set:

    recall@1     share of positions where the real actions give the closest prediction to the
                 true target among K + 1 candidates (chance = 1 / (K + 1))
    MRR          mean reciprocal rank of the real actions
    loss         the training loss (smooth L1 per slot) with real / borrowed / no actions
    beats none   share of positions where real actions beat having no actions

    python scripts/action_check.py data/worldmodel/runs/<run> --ckpt model_step2000.pt
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

import train_window as tw  # noqa: E402  (same windows and targets as training)
from cosmicstar.worldmodel import vocab as vocab_mod  # noqa: E402
from cosmicstar.worldmodel.action_encoder import ActionEncoder  # noqa: E402
from cosmicstar.worldmodel.action_tokens import ActionVocab, GameActions, collate_actions  # noqa: E402
from cosmicstar.worldmodel.encoder import StateEncoder  # noqa: E402
from cosmicstar.worldmodel.predictor import slots_of  # noqa: E402
from cosmicstar.worldmodel.tokenize import GameFrames, collate  # noqa: E402
from cosmicstar.worldmodel.window_predictor import WindowPredictor  # noqa: E402


def load_models(ck, v, dev):
    """Like probe_frozen.load_models, but tolerant of checkpoints from before the encoder
    got its visibility input (those are run with the visibility features stripped)."""
    a = argparse.Namespace(**ck["args"])
    enc = StateEncoder(v).to(dev).eval()
    has_vis = any(k.startswith("vis_in") for k in ck["encoder"])
    enc.load_state_dict(ck["encoder"], strict=False)  # older checkpoints lack the visibility / order paths
    enc.use_orders = any(k.startswith("ord_in") for k in ck["encoder"])
    tenc = copy.deepcopy(enc)
    tenc.load_state_dict(ck["target"], strict=False)
    pred = WindowPredictor(depth=a.depth, horizons=ck["horizons"], residual=a.residual).to(dev).eval()
    pred.load_state_dict(ck["predictor"])
    av = ActionVocab()
    aenc = ActionEncoder(enc, av).to(dev).eval()
    aenc.load_state_dict(ck["actions"])
    return (enc, tenc, pred, aenc), av, not has_vis


@torch.no_grad()
def window_state(models, w, dev, strip_vis):
    """Everything about a window that doesn't depend on the actions: context slots, the
    present, and the target table per horizon."""
    enc, tenc, _, _ = models
    T = len(w["frames"])

    def batch(frames):
        b = tw.to(collate(frames), dev)
        if strip_vis:
            b.pop("vis", None)
        return b

    def target_slots(frames):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s, m = slots_of(tenc(batch(frames)))
        return F.layer_norm(s.float(), s.shape[-1:]), m

    fb = batch(w["frames"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        slots, mask = slots_of(enc(fb))
    S = slots.shape[1]
    slots, mask = slots.view(1, T, S, -1), mask.view(1, T, S)
    present, _ = target_slots(w["frames"])
    present = present.view(1, T, S, -1)
    if w["wframes"] is None:  # observer: the window frames are their own targets
        flat, fmask = present[0], mask[0]
    else:  # player view: the observer's frames
        flat, fmask = target_slots(w["wframes"])
    if w["tframes"]:
        eo, em = target_slots(w["tframes"])
        flat, fmask = torch.cat([flat, eo]), torch.cat([fmask, em])
    targets, valid = {}, {}
    for h, idx in w["tgt"].items():
        it = torch.from_numpy(idx).to(dev)
        ok = it >= 0
        targets[h] = flat[it.clamp(min=0)]                                # [T, S, d]
        valid[h] = fmask[it.clamp(min=0)] & ok[:, None] & mask[0]        # [T, S]
    return dict(slots=slots, mask=mask, present=present, targets=targets, valid=valid,
                times=torch.from_numpy(w["times"][None]).to(dev), P=fb["ply_mask"].shape[1])


@torch.no_grad()
def predict(models, st, act, dev, none=False):
    _, _, pred, aenc = models
    ab = tw.to(collate_actions([act]), dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        summaries, s_mask, _ = aenc(ab)
        if none:
            s_mask = torch.zeros_like(s_mask)  # no action token can be attended
        return pred(st["slots"], st["mask"], st["times"], base=st["present"], act=summaries, act_mask=s_mask)


def position_loss(pred_h, target, valid, P):
    """The training loss at each position: smooth L1 per slot, averaged within each kind
    (CLS / players / grid) and then across kinds. [T]"""
    err = F.smooth_l1_loss(pred_h[0].float(), target, reduction="none").mean(-1)
    m = valid.float()
    parts = [(err[:, s] * m[:, s]).sum(1) / m[:, s].sum(1).clamp(min=1) for s in tw.kind_slices(P).values()]
    return sum(parts) / len(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="model_latest.pt")
    ap.add_argument("--games", type=int, default=3, help="games per evaluation set")
    ap.add_argument("--windows", type=int, default=2, help="windows per game (and per viewer, for player views)")
    ap.add_argument("--fakes", type=int, default=8, help="borrowed action chunks per window")
    args = ap.parse_args()
    dev = "cuda"
    rng = np.random.RandomState(0)
    ck = torch.load(os.path.join(args.run, args.ckpt), map_location=dev, weights_only=False)
    a = ck["args"]
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    v = vocab_mod.Vocab(json.load(open(os.path.join(args.run, "vocab.json"))))
    models, av, strip_vis = load_models(ck, v, dev)
    spec = tw.WindowSpec(a["window_s"], a["frame_stride"], [float(x) for x in a["horizons_s"].split(",")],
                         a["long_sub"], ck["step_loops"])
    pov = a.get("pov_dir")

    def load(p):
        g = GameFrames(p, v)
        acts = os.path.join(a["actions_dir"], os.path.basename(p))
        if not pov:
            return g, GameActions(acts, g, av), None
        views = {}
        stem = os.path.basename(p)[:-4] + "_p"
        for q in sorted(os.listdir(pov)):
            if q.startswith(stem) and q.endswith(".npz") and not q.endswith(".tmp.npz"):
                gv = GameFrames(os.path.join(pov, q), v)
                views[int(q[len(stem):-4])] = (gv, GameActions(acts, gv, av))
        return g, None, views

    print(f"{args.run} {args.ckpt} | window {a['window_s']}s | horizons " + ", ".join(f"{spec.seconds(h)}s" for h in spec.horizons)
          + (" | player views" if pov else " | observer") + (" | visibility features stripped (older encoder)" if strip_vis else ""), flush=True)
    # windows, grouped by viewer so borrowed actions come from the same seat (same codes)
    wins = collections.defaultdict(list)  # (set, viewer) -> [(game index, window)]
    for set_name, files in cfg["eval"].items():
        if set_name == "seen-games" or not files:
            continue
        for gi, f in enumerate(files[:args.games]):
            g, ga, views = load(os.path.join(a["data"], f))
            seats = [(slot, (g, None, {slot: views[slot]})) for slot in views] if views else [(None, (g, ga, None))]
            for viewer, game in seats:
                made = tries = 0
                while made < args.windows and tries < 20 * args.windows:
                    tries += 1
                    w = tw.make_window(game, spec, rng, augment=False)
                    if w is not None:
                        wins[(set_name, viewer)].append((gi, w))
                        made += 1
    n_wins = sum(len(x) for x in wins.values())
    print(f"{n_wins} windows in {len(wins)} (set, viewer) groups", flush=True)

    acc = collections.defaultdict(list)
    for (set_name, viewer), group in wins.items():
        for gi, w in group:
            others = [o for oi, o in group if oi != gi]  # borrowed actions: other games, same seat
            K = min(args.fakes, len(others))
            if K == 0:
                continue
            st = window_state(models, w, dev, strip_vis)
            fakes = [others[i]["act"] for i in rng.choice(len(others), K, replace=False)]
            cands = [predict(models, st, w["act"], dev)] + [predict(models, st, f, dev) for f in fakes]
            none = predict(models, st, w["act"], dev, none=True)
            for h in spec.horizons:
                ok = st["valid"][h].any(1)
                if not ok.any():
                    continue
                L = torch.stack([position_loss(c[h], st["targets"][h], st["valid"][h], st["P"]) for c in cands])[:, ok]  # [K+1, n]
                Ln = position_loss(none[h], st["targets"][h], st["valid"][h], st["P"])[ok]
                rank = 1 + (L[1:] < L[0]).sum(0).float()
                acc[(set_name, h, "recall")].append((rank == 1).float().mean().item())
                acc[(set_name, h, "mrr")].append((1 / rank).mean().item())
                acc[(set_name, h, "true")].append(L[0].mean().item())
                acc[(set_name, h, "fake")].append(L[1:].mean().item())
                acc[(set_name, h, "none")].append(Ln.mean().item())
                acc[(set_name, h, "beats_none")].append((L[0] < Ln).float().mean().item())
                acc[(set_name, h, "K")].append(K)
    for set_name in sorted({k[0] for k in acc}):
        print(f"\n=== {set_name}")
        print(f"{'horizon':>8} {'recall@1':>9} {'chance':>7} {'MRR':>6} | {'loss real':>9} {'borrowed':>9} {'none':>9} | {'beats none':>10}")
        for h in spec.horizons:
            if (set_name, h, "recall") not in acc:
                continue
            m = lambda k: float(np.mean(acc[(set_name, h, k)]))
            K = m("K")
            print(f"{spec.seconds(h):>7.2f}s {m('recall'):>9.0%} {1 / (K + 1):>7.0%} {m('mrr'):>6.2f} | "
                  f"{m('true'):>9.4f} {m('fake'):>9.4f} {m('none'):>9.4f} | {m('beats_none'):>10.0%}")
    print("\nrecall@1 at chance = the predictor ignores which actions it is given; loss real < borrowed < none = actions are used and help")


if __name__ == "__main__":
    main()
