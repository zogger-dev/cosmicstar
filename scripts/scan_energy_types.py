"""Per raw unit type (API id): the energy pool and whether it flies, scanned from
sampled windows of the training games -> data/worldmodel/energy_types.json, which
energy_data.Tables reads.

    python scripts/scan_energy_types.py --passes 80

The API exports a unit's energy but not its maximum, so the pool is the largest energy
seen, rounded up to 100 or 200 (this patch's two pool sizes). Flying is the raw flags'
bit 0, kept per type so the table can say it for a row whose frame does not.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_physics as tp

from cosmicstar.worldmodel import physics_data
from cosmicstar.worldmodel.energy_data import FLYING_BIT, N_RAW_TYPES
from cosmicstar.worldmodel.physics_data import RAW_NUMS

I = {n: i for i, n in enumerate(RAW_NUMS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data/games_v3")
    ap.add_argument("--vocab", default="data/worldmodel/frame_cache_seq/vocab.json")
    ap.add_argument("--patch", default="5.0.14")
    ap.add_argument("--split", default="data/games/splits/default.json")
    ap.add_argument("--passes", type=int, default=80)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="data/worldmodel/energy_types.json")
    args = ap.parse_args()
    split = json.load(open(args.split))
    ids = set(split["train"]) - set(tp.EXCLUDED)
    bases = tp.of_patch(physics_data.base_paths(args.games, ids), args.patch)
    map_ids = {
        m: i for i, m in enumerate(sorted({tp.map_of(b) for b in bases.values()}))
    }
    plan_kw = dict(
        seed=1, stride=14, width=32, min_games=40, tokens=6000, per_window=0, accum=1
    )
    data = tp.batches(bases, args.vocab, map_ids, 16, 24, args.workers, plan_kw, 0)
    emax, fly, seen = (torch.zeros(N_RAW_TYPES) for _ in range(3))
    for _ in range(args.passes):
        b = next(data)
        m = b["mask"]
        rt = b["rtype"].clamp(0, N_RAW_TYPES - 1)[m]
        emax.scatter_reduce_(0, rt, b["num"][..., I["energy"]][m], "amax")
        fly.scatter_reduce_(
            0, rt, (b["num"][..., I["flags"]][m].long() & FLYING_BIT).float(), "amax"
        )
        seen.scatter_add_(0, rt, torch.ones(len(rt)))
    name = {
        int(k): v["name"]
        for k, v in json.load(open("data/worldmodel/unit_data.json")).items()
    }
    out = {"energy_max": {}, "flying": []}
    for t in torch.nonzero(seen > 0).flatten().tolist():
        if emax[t] > 0:
            out["energy_max"][str(t)] = 100 if emax[t] <= 100 else 200
        if fly[t] > 0:
            out["flying"].append(t)
    print(
        f"types seen {int((seen > 0).sum())}, with energy {len(out['energy_max'])},"
        f" flying {len(out['flying'])}"
    )
    print("energy:", {name.get(int(t), t): v for t, v in out["energy_max"].items()})
    print("flying:", [name.get(t, t) for t in out["flying"]])
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
