"""Trains the Layer 2 (60 x 5s = 5-minute horizon) Port-Hamiltonian model
(src/cosmicstar/worldmodel/energy_l2.py) on pre-extracted Layer 1 full-rank 5s summaries.

    .venv/bin/python scripts/train_energy_l2.py --steps 3000 --window 60 --batch 8
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cosmicstar.worldmodel import energy_l2


def load_game_summaries(summary_dir, subset):
    files = sorted(glob.glob(os.path.join(summary_dir, subset, "*.pt")))
    games = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        if len(d["loops"]) >= 16:
            games.append(d)
    return games


def slice_and_swap(g, t0, L, swap=False):
    """Slices [t0 : t0 + L] (padding if shorter) and optionally swaps Player 1 <-> Player 2."""
    T_g = len(g["loops"])
    if T_g >= L:
        s = slice(t0, t0 + L)
        pad = 0
    else:
        s = slice(0, T_g)
        pad = L - T_g

    def _get(k):
        v = g[k][s]
        if pad > 0:
            # Repeat final frame to pad short games to L
            last = v[-1:].expand(pad, *v.shape[1:])
            v = torch.cat([v, last], dim=0)
        return v

    cls_out = _get("cls_out")
    ply_out = _get("ply_out")
    g_qq = _get("G_qq")
    g_qc = _get("G_qc")
    g_cc = _get("G_cc_triu")
    h1 = _get("H1_l1")
    h2 = _get("H2_l1")
    a1 = _get("A1_l1")
    u_win = _get("u_win")
    act_feat = _get("act_feat")
    u_active = _get("u_active")

    if swap:
        ply_out = ply_out.flip(dims=[1])
        g_qq = g_qq.flip(dims=[1])
        g_qc = g_qc.flip(dims=[1])
        g_cc = g_cc.flip(dims=[1])
        h1, h2 = h2, h1
        a1 = -a1
        u_win = u_win.flip(dims=[1])
        act_feat = act_feat.flip(dims=[1])
        u_active = u_active.flip(dims=[1])

    return dict(
        cls_out=cls_out,
        ply_out=ply_out,
        G_qq=g_qq,
        G_qc=g_qc,
        G_cc_triu=g_cc,
        H1_l1=h1,
        H2_l1=h2,
        A1_l1=a1,
        u_win=u_win,
        act_feat=act_feat,
        u_active=u_active,
    )


def sample_batch(games, batch_size, L, rng, device="cuda"):
    items = []
    for _ in range(batch_size):
        g = games[int(rng.integers(len(games)))]
        T_g = len(g["loops"])
        t0 = int(rng.integers(0, max(1, T_g - L + 1))) if T_g > L else 0
        swap = bool(rng.integers(0, 2))
        items.append(slice_and_swap(g, t0, L, swap=swap))

    batch = {
        k: torch.stack([it[k] for it in items], dim=0).to(device, non_blocking=True)
        for k in items[0]
    }
    return batch


@torch.no_grad()
def evaluate_unseen_trajectories(model, val_games, device="cuda"):
    """Runs full-game causal inference on all unseen games and computes zero-shot
    winner accuracy at 25%, 50%, 70%, 85%, 95%, and 100% of game duration."""
    model.eval()
    stage_pcts = [0.25, 0.50, 0.70, 0.85, 0.95, 1.00]
    acc_l1 = {p: 0 for p in stage_pcts}
    acc_l2 = {p: 0 for p in stage_pcts}
    acc_tot = {p: 0 for p in stage_pcts}
    n_valid = 0

    for g in val_games:
        win = g.get("winner", 0)
        if win not in (1, 2):
            continue
        n_valid += 1
        T_g = len(g["loops"])
        item = slice_and_swap(g, 0, T_g, swap=False)
        batch = {k: v.unsqueeze(0).to(device) for k, v in item.items()}

        x_obs, x_lat, _, _, _ = model.seq_enc(batch)
        x = torch.cat([x_obs, x_lat], dim=-1)[0]  # [T_g, 2, d_p]
        hout_l2 = model.H_l2(x[:, 0], x[:, 1])

        # Oriented strategic advantage (> 0 for P1 win) in shared 1e-3 Joule units:
        # Both Layer 1 and Layer 2 follow the same dissipative sign (-A1_l1 > 0, -A1_l2 > 0)
        a_l1 = -g["A1_l1"].numpy()
        a_l2 = -hout_l2["A1"].cpu().numpy()
        a_tot = a_l1 + a_l2

        for pct in stage_pcts:
            idx = min(T_g - 1, max(0, int(round(pct * (T_g - 1)))))
            v1 = float(a_l1[idx])
            v2 = float(a_l2[idx])
            vt = float(a_tot[idx])
            acc_l1[pct] += int((v1 > 0) == (win == 1))
            acc_l2[pct] += int((v2 > 0) == (win == 1))
            acc_tot[pct] += int((vt > 0) == (win == 1))

    out = {}
    for pct in stage_pcts:
        tag = f"acc_{int(pct*100)}"
        out[f"{tag}_l1"] = round(acc_l1[pct] / max(1, n_valid), 4)
        out[f"{tag}_l2"] = round(acc_l2[pct] / max(1, n_valid), 4)
        out[f"{tag}_tot"] = round(acc_tot[pct] / max(1, n_valid), 4)
    return out


def evaluate_windows(model, val_games, L, device="cuda"):
    model.eval()
    items = []
    for g in val_games:
        T_g = len(g["loops"])
        t0 = max(0, T_g - L)
        items.append(slice_and_swap(g, t0, min(L, T_g), swap=False))
    # Collate Fixed-L windows for games with T_g >= L
    fixed = [slice_and_swap(g, max(0, len(g["loops"]) - L), L, swap=False) for g in val_games]
    batch = {k: torch.stack([it[k] for it in fixed], dim=0).to(device) for k in fixed[0]}
    with torch.enable_grad():
        loss, logs = model(batch)
    out = {k: round(float(v), 5) for k, v in logs.items()}
    out["loss"] = round(float(loss.detach()), 5)
    c = max(1e-8, out["copy_loss_obs"])
    out["r2_obs_5s"] = round(1.0 - out["loss_obs"] / c, 4)
    out.update(evaluate_unseen_trajectories(model, val_games, device=device))
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_dir", default="data/worldmodel/l2_summaries_5014")
    ap.add_argument("--run", default="data/worldmodel/runs/energy_l2_60x5s")
    ap.add_argument("--window", type=int, default=60, help="60 x 5s = 5 minutes")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--r_gram", type=int, default=16)
    ap.add_argument("--r_macro", type=int, default=32)
    ap.add_argument("--u_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--rollout", type=int, default=4, help="multi-step rollout (4 x 5s = 20s)")
    ap.add_argument("--wd", type=float, default=0.03, help="AdamW weight decay")
    ap.add_argument("--resume", default="", help="checkpoint to resume from")
    ap.add_argument("--val_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.run, exist_ok=True)

    t0 = time.time()
    train_games = load_game_summaries(args.summary_dir, "train")
    val_games = load_game_summaries(args.summary_dir, "unseen_games")
    print(
        f"Loaded {len(train_games)} train games and {len(val_games)} unseen games in {time.time() - t0:.1f}s",
        flush=True,
    )

    model = energy_l2.EnergyModelL2(
        d=args.d,
        layers=args.layers,
        heads=args.heads,
        r_gram=args.r_gram,
        r_macro=args.r_macro,
        u_dim=args.u_dim,
        hidden=args.hidden,
        rank=args.rank,
        rollout_steps=args.rollout,
    ).cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Layer 2 EnergyModelL2: {sum(p.numel() for p in params)/1e6:.2f}M parameters", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_step = int(ck.get("step", 0))
        # Advance RNG to match step
        for _ in range(start_step * args.batch * 3):
            rng.integers(0, 10)
        print(f"Resumed from {args.resume} at step {start_step}", flush=True)
    else:
        # Initial step 0 evaluation
        val0 = evaluate_windows(model, val_games, args.window)
        print(json.dumps({"step": 0, "eval": "unseen_games", **val0}), flush=True)

    t_train = time.time()
    for step in range(start_step + 1, args.steps + 1):
        batch = sample_batch(train_games, args.batch, args.window, rng, device="cuda")
        loss, logs = model(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if step <= args.warmup:
            lr_scale = step / max(1, args.warmup)
        else:
            prog = (step - args.warmup) / max(1, args.steps - args.warmup)
            lr_scale = 0.1 + 0.45 * (1.0 + np.cos(np.pi * prog))
        for g in opt.param_groups:
            g["lr"] = args.lr * float(lr_scale)
        opt.step()

        if step % 25 == 0:
            c = max(1e-8, float(logs["copy_loss_obs"]))
            l_obs = float(logs["loss_obs"])
            print(
                json.dumps(
                    dict(
                        step=step,
                        s=round(time.time() - t_train, 1),
                        loss=round(float(loss.detach()), 5),
                        loss_obs=round(l_obs, 5),
                        copy_loss_obs=round(c, 5),
                        r2_obs_5s=round(1.0 - l_obs / c, 4),
                        loss_obs_multi=round(float(logs["loss_obs_multi"]), 5),
                        loss_td=round(float(logs["loss_td"]), 5),
                        loss_lat=round(float(logs["loss_lat"]), 5),
                        goal_cos=round(float(logs["goal_cos"]), 4),
                        z_macro_std=round(float(logs["z_macro_std"]), 4),
                    )
                ),
                flush=True,
            )

        if step % args.val_every == 0 or step == args.steps:
            val_res = evaluate_windows(model, val_games, args.window)
            rec = {"step": step, "eval": "unseen_games", **val_res}
            print(json.dumps(rec), flush=True)
            with open(os.path.join(args.run, "eval.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            torch.save(
                {"step": step, "model": model.state_dict(), "opt": opt.state_dict(), "args": vars(args)},
                os.path.join(args.run, f"model_l2_step{step}.pt"),
            )


if __name__ == "__main__":
    main()
