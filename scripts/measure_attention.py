"""The player as a power source: how much attention each player spends per second, how
much game energy it sets in motion, and how much of the capacity goes unused.

    python scripts/measure_attention.py --passes 150

Samples passes of windows through train_physics's reader (the physics batches) and
measures, per player per window:

    actions        action tokens by kind (commands, reissues, cancels; selections and
                   groups; camera) per second
    moved          rows of the player whose order queue changed, and their value in
                   resource units (unit_data: minerals + vespene), per second: the game
                   energy the attention set in motion
    spent          resources that left the bank per second (purchases: the bank's
                   negative steps); income the positive steps
    idle           seconds of the window with no action at all

Reports distributions over player-windows, the ceiling (the top percentiles of
actions a second), the value set in motion per command, and how value per second
scales with actions per second. No model involved; CPU only.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools

import train_physics as tp

from cosmicstar.worldmodel import energy_data, physics_data
from cosmicstar.worldmodel.action_tokens import (
    A_CAMERA,
    A_CAMERA_SAVE,
    A_CANCEL,
    A_CMD,
    A_GROUP,
    A_REISSUE,
    A_SELECT,
)
from cosmicstar.worldmodel.tokenize import PC_KEYS

LOOPS_PER_SECOND = 22.4
COMMANDS = (A_CMD, A_REISSUE, A_CANCEL)
SELECTS = (A_SELECT, A_GROUP)
CAMERA = (A_CAMERA, A_CAMERA_SAVE)
PC = {n: i for i, n in enumerate(PC_KEYS)}


def per_window(b, value_of):
    """-> dict of [B, 2] arrays: per window and player, the counts over the window."""
    B, F_, R = b["mask"].shape
    kind, player, am = b["act_kind"], b["act_player"], b["act_mask"]
    out = {}
    for name, kinds in (
        ("commands", COMMANDS),
        ("selects", SELECTS),
        ("camera", CAMERA),
    ):
        is_kind = torch.zeros_like(kind, dtype=torch.bool)
        for k in kinds:
            is_kind |= kind == k
        out[name] = torch.stack(
            [(is_kind & am & (player == p)).sum((1, 2)) for p in (0, 1)], 1
        ).float()
    any_act = torch.stack(
        [(am & (player == p)).any(2) for p in (0, 1)], 2
    )  # [B, F, 2] a frame with an action by p
    out["idle_frames"] = (~any_act).sum(1).float()
    out["actions"] = out["commands"] + out["selects"] + out["camera"]
    # rows that received a command (act_recip: frame of the batch, slot, row), and
    # their value, by owner: what the attention set in motion. Order changes without
    # a command (a worker's gather / return flip, an order finishing) are counted
    # apart as auto_changed
    recip = torch.zeros(B * F_, R, dtype=torch.bool)
    pr = b["act_recip"]
    if len(pr):
        recip[pr[:, 0], pr[:, 2]] = True
    recip = recip.view(B, F_, R) & b["mask"] & b["alive"]
    rtype = b["rtype"].clamp(0, energy_data.N_RAW_TYPES - 1)
    owner = b["owner"][:, None, :].expand_as(recip)
    out["moved_rows"] = torch.stack(
        [(recip & (owner == p)).sum((1, 2)) for p in (0, 1)], 1
    ).float()
    out["moved_value"] = torch.stack(
        [(value_of[rtype] * (recip & (owner == p))).sum((1, 2)) for p in (0, 1)], 1
    )
    changed = b["queue_changed"] & b["mask"][:, 1:] & b["alive"][:, 1:] & ~recip[:, 1:]
    out["auto_changed"] = torch.stack(
        [(changed & (owner[:, 1:] == p)).sum((1, 2)) for p in (0, 1)], 1
    ).float()
    # the bank's steps: purchases down, income up
    v = torch.expm1(b["ply_num"][..., : len(PC_KEYS)] * 8)
    bank = v[..., PC["minerals"]] + v[..., PC["vespene"]]  # [B, F, 2]
    d = bank[:, 1:] - bank[:, :-1]
    out["spent"] = (-d).clamp(min=0).sum(1)
    out["income"] = d.clamp(min=0).sum(1)
    out["army_value"] = torch.stack(
        [
            (
                value_of[b["rtype"][:, 0].clamp(0, energy_data.N_RAW_TYPES - 1)]
                * (b["mask"][:, 0] & b["alive"][:, 0] & (b["owner"] == p))
            ).sum(1)
            for p in (0, 1)
        ],
        1,
    )
    out["seconds"] = torch.full((B, 2), F_ / LOOPS_PER_SECOND)
    return {k: t.reshape(-1).numpy() for k, t in out.items()}


def pct(x, qs=(5, 25, 50, 75, 95, 99)):
    return {f"p{q}": round(float(np.percentile(x, q)), 3) for q in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data/games_v3")
    ap.add_argument("--vocab", default="data/worldmodel/frame_cache_seq/vocab.json")
    ap.add_argument("--patch", default="5.0.14")
    ap.add_argument("--base_build", type=int, default=94137)
    ap.add_argument("--split", default="data/games/splits/default.json")
    ap.add_argument("--passes", type=int, default=150)
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--tokens", type=int, default=6000)
    ap.add_argument("--stride", type=int, default=14)
    ap.add_argument("--bucket", type=int, default=32)
    ap.add_argument("--min_bucket_games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    split = json.load(open(args.split))
    train_ids = set(split["train"]) - set(tp.EXCLUDED)
    bases = tp.of_patch(physics_data.base_paths(args.games, train_ids), args.patch)
    map_ids = {
        m: i for i, m in enumerate(sorted({tp.map_of(b) for b in bases.values()}))
    }
    print(f"{len(bases)} games of patch {args.patch}", flush=True)
    value_of = energy_data.Tables(args.base_build).k[:, 0]
    plan_kw = dict(
        seed=args.seed,
        stride=args.stride,
        width=args.bucket,
        min_games=args.min_bucket_games,
        tokens=args.tokens,
        per_window=0,
        accum=1,
    )
    data = tp.batches(
        bases, args.vocab, map_ids, args.seqs, args.frames, args.workers, plan_kw, 0
    )
    acc = {}
    for i in range(args.passes):
        b = next(data)
        for k, v in per_window(b, value_of).items():
            acc.setdefault(k, []).append(v)
        if (i + 1) % 25 == 0:
            print(f"pass {i + 1}/{args.passes}", flush=True)
    a = {k: np.concatenate(v) for k, v in acc.items()}
    sec = a["seconds"]
    aps, cps = a["actions"] / sec, a["commands"] / sec
    moved_ps, spent_ps = a["moved_value"] / sec, a["spent"] / sec
    energy_ps = moved_ps + spent_ps
    busy = a["commands"] > 0
    report = dict(
        player_windows=len(sec),
        seconds_each=round(float(sec[0]), 2),
        actions_per_s=pct(aps),
        commands_per_s=pct(cps),
        selects_per_s=pct(a["selects"] / sec),
        camera_per_s=pct(a["camera"] / sec),
        share_windows_no_action=round(float((a["actions"] == 0).mean()), 3),
        share_windows_no_command=round(float((a["commands"] == 0).mean()), 3),
        idle_share_of_frames=round(
            float(a["idle_frames"].sum() / (len(sec) * args.frames)), 3
        ),
        moved_rows_per_s=pct(a["moved_rows"] / sec),
        auto_changed_rows_per_s=pct(a["auto_changed"] / sec),
        moved_value_per_s=pct(moved_ps),
        spent_per_s=pct(spent_ps),
        income_per_s=pct(a["income"] / sec),
        army_value=pct(a["army_value"]),
        # the energy set in motion per command, where there were commands
        value_per_command=pct(a["moved_value"][busy] / a["commands"][busy]),
        rows_per_command=pct(a["moved_rows"][busy] / a["commands"][busy]),
        # does more attention move more energy? value/s by quartile of commands/s
        energy_per_s_by_commands_quartile={},
        corr_commands_vs_moved_value=round(float(np.corrcoef(cps, moved_ps)[0, 1]), 3),
        corr_actions_vs_energy=round(float(np.corrcoef(aps, energy_ps)[0, 1]), 3),
    )
    edges = np.percentile(cps, [0, 25, 50, 75, 100])
    for lo, hi in itertools.pairwise(edges):
        m = (cps >= lo) & (cps <= hi)
        report["energy_per_s_by_commands_quartile"][f"{lo:.1f}-{hi:.1f}/s"] = dict(
            n=int(m.sum()),
            moved_value_per_s=round(float(np.median(moved_ps[m])), 1),
            spent_per_s=round(float(np.median(spent_ps[m])), 1),
            value_per_command=round(
                float(np.median(a["moved_value"][m & busy] / a["commands"][m & busy])),
                1,
            )
            if (m & busy).any()
            else None,
        )
    print(json.dumps(report, indent=1))
    if args.out:
        np.savez(args.out, **a)


if __name__ == "__main__":
    main()
