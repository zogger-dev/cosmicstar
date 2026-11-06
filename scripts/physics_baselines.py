"""What the physics model has to beat, on the same store and windows (DESIGN_physics.md "Evaluation"): copy the last frame; follow the current order at
the unit type's usual speed. Position by split, moving and still rows apart, neutrals and critters apart, plus rare changes against copying.

    python scripts/physics_baselines.py --windows 300
"""

import argparse
import collections
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cosmicstar.worldmodel import physics_data
from cosmicstar.worldmodel.vocab import Vocab

MOVING_NEUTRALS = (321, 322, 335, 336, 337, 612, 659, 660, 661)
LARVA = 151
EXCLUDED = (
    "982fc2368b80adb3",
    "1ff02cd0dd2751fa",
    "c50626c6c77288c6",
    "5bf221dbd7045ab1",
    "d2de5d4aef655894",
)
I = {n: i for i, n in enumerate(physics_data.RAW_NUMS)}


def first_order_target(w):
    """[F, R, 2] each row's first order target position (NaN: none): a point, or the targeted row's position."""
    out = np.full(w["mask"].shape + (2,), np.nan, np.float32)
    if not w["ord_where"]:
        return out
    wh, v = np.concatenate(w["ord_where"]), np.concatenate(w["ord_vals"])
    q = wh[:, 2] == 0
    wh, v = wh[q], v[q]
    out[wh[:, 0], wh[:, 1], 0], out[wh[:, 0], wh[:, 1], 1] = v[:, 1], v[:, 2]
    tr = v[:, 3].astype(np.int64)
    has = tr >= 0
    out[wh[has, 0], wh[has, 1], 0] = w["x"][wh[has, 0], tr[has]]
    out[wh[has, 0], wh[has, 1], 1] = w["y"][wh[has, 0], tr[has]]
    return out


def measure(w, speed=None, acc=None):
    m, al = w["mask"], w["alive"]
    t = w["type"].astype(np.int64)
    sc = (
        m[:-1]
        & al[:-1]
        & m[1:]
        & al[1:]
        & (w["inside"][:-1] == 0)
        & (w["inside"][1:] == 0)
    )
    fi, ri = np.nonzero(sc)
    dx, dy = w["x"][fi + 1, ri] - w["x"][fi, ri], w["y"][fi + 1, ri] - w["y"][fi, ri]
    step = np.hypot(dx, dy)
    ty = t[fi, ri]
    if speed is not None:
        for k, s in zip(ty[step > 0.003].tolist(), step[step > 0.003].tolist()):
            speed[k].append(s)
    if acc is None:
        return
    tg = first_order_target(w)[fi, ri]
    px, py = w["x"][fi, ri], w["y"][fi, ri]
    to = np.hypot(tg[:, 0] - px, tg[:, 1] - py)
    usual = acc["speed"]
    sp = np.array([usual.get(int(k), 0.0) for k in ty.tolist()], np.float32)
    go = np.minimum(sp, np.nan_to_num(to))
    ok = np.isfinite(to) & (to > 1e-3)
    ex = np.where(ok, np.nan_to_num((tg[:, 0] - px) / np.maximum(to, 1e-3)) * go, 0)
    ey = np.where(ok, np.nan_to_num((tg[:, 1] - py) / np.maximum(to, 1e-3)) * go, 0)
    copy_err = step
    follow_err = np.hypot(dx - ex, dy - ey)
    own = w["owner"][ri]
    neutral = own == physics_data.NEUTRAL
    critter = neutral & np.isin(ty, MOVING_NEUTRALS)
    larva = ty == LARVA
    player = ~neutral & ~larva
    moving = step > 0.003
    for name, sel in (
        ("player moving", player & moving),
        ("player still", player & ~moving),
        ("player all", player),
        ("larva", larva),
        ("critter", critter),
        ("other neutral", neutral & ~critter),
    ):
        if sel.any():
            acc["copy"][name].append(copy_err[sel])
            acc["follow"][name].append(follow_err[sel])
    # rare changes: how often the copy baseline is wrong, per token and frame
    n = w["num"]
    fr = sc
    acc["count"]["tokens"] += int(fr.sum())
    for k in (
        "hp",
        "shield",
        "energy",
        "contents",
        "weapon_cooldown",
        "flags",
        "buff_remain",
    ):
        acc["count"][f"change {k}"] += int(
            (n[:-1, :, I[k]][fr] != n[1:, :, I[k]][fr]).sum()
        )
    acc["count"]["frames"] += w["mask"].shape[0] - 1
    acc["count"]["births"] += len(w["birth_row"])
    acc["count"]["frames_with_birth"] += len(np.unique(w["birth_frame"]))
    d = w["death_frame"]
    acc["count"]["deaths"] += int((d >= 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data/games_arrow")
    ap.add_argument("--vocab", default="data/worldmodel/frame_cache_seq/vocab.json")
    ap.add_argument("--split", default="data/games/splits/default.json")
    ap.add_argument("--windows", type=int, default=300)
    ap.add_argument("--frames", type=int, default=56)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    vocab = Vocab(json.load(open(args.vocab)))
    split = json.load(open(args.split))
    rng = np.random.default_rng(0)
    sets = {
        "train": split["train"],
        "unseen games": split["held_out_games"],
        "unseen maps": split["held_out_map_games"],
    }

    def read(gid_base):
        g = physics_data.Game(gid_base, vocab)
        g.acts = None
        g._actions = lambda *a, **k: None
        return g.window(
            int(np.random.randint(100, max(101, g.F - args.frames - 1))), args.frames
        )

    # usual speed per type, on training games only
    bases = physics_data.base_paths(args.games, set(split["train"]) - set(EXCLUDED))
    ids = sorted(bases)
    speed = collections.defaultdict(list)
    with ThreadPoolExecutor(args.workers) as pool:
        for w in pool.map(
            read, [bases[ids[i]] for i in rng.integers(len(ids), size=args.windows)]
        ):
            measure(w, speed=speed)
    usual = {k: float(np.median(v)) for k, v in speed.items() if len(v) >= 50}
    print(f"usual speed for {len(usual)} types", flush=True)
    res = {}
    for name, game_ids in sets.items():
        bases = physics_data.base_paths(args.games, set(game_ids) - set(EXCLUDED))
        ids = sorted(bases)
        acc = dict(
            speed=usual,
            copy=collections.defaultdict(list),
            follow=collections.defaultdict(list),
            count=collections.Counter(),
        )
        with ThreadPoolExecutor(args.workers) as pool:
            for w in pool.map(
                read, [bases[ids[i]] for i in rng.integers(len(ids), size=args.windows)]
            ):
                measure(w, acc=acc)
        print(f"\n== {name}: {len(ids)} games, {args.windows} windows of {args.frames}")
        print(
            f"{'':16s} {'tokens':>10s} | {'copy: mean':>11s} {'median':>7s} {'p99':>7s} | {'follow: mean':>12s} {'median':>7s} {'p99':>7s}"
        )
        for k in (
            "player moving",
            "player still",
            "player all",
            "larva",
            "critter",
            "other neutral",
        ):
            if k not in acc["copy"]:
                continue
            c, f = np.concatenate(acc["copy"][k]), np.concatenate(acc["follow"][k])
            print(
                f"{k:16s} {len(c):10d} | {c.mean():11.4f} {np.median(c):7.3f} {np.percentile(c, 99):7.3f} | {f.mean():12.4f} {np.median(f):7.3f} {np.percentile(f, 99):7.3f}"
            )
        cnt = acc["count"]
        print(
            "changes per token:",
            {
                k[7:]: round(v / cnt["tokens"], 5)
                for k, v in cnt.items()
                if k.startswith("change ")
            },
        )
        print(
            f"births per loop {cnt['births'] / cnt['frames']:.4f}, frames with a birth {cnt['frames_with_birth'] / cnt['frames']:.4f}, deaths per loop {cnt['deaths'] / cnt['frames']:.4f}",
            flush=True,
        )
        res[name] = {k: float(np.concatenate(v).mean()) for k, v in acc["copy"].items()}
    json.dump(res, open("data/worldmodel/physics_baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
