"""Parses every replay's game events into an action table (see worldmodel/replay_actions.py).

Output: <out>/<same name as the game's state file>.npz. No game engine needed.

    python scripts/extract_actions.py --out data/worldmodel/actions
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import extract_states as ex  # noqa: E402  (same replay selection and file naming as the state)
from cosmicstar.worldmodel import replay_actions  # noqa: E402


def one(job):
    path, out = job
    if os.path.exists(out):
        return "skip", path
    try:
        d = replay_actions.parse(path)
        tmp = out + ".tmp.npz"
        np.savez_compressed(tmp, **d)
        os.replace(tmp, out)
        return "ok", path
    except Exception as e:
        return f"fail: {repr(e)[:120]}", path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default=r"C:\Users\prade\Documents\StarCraft II\Accounts\52583725\1-S2-1-1544307\Replays\Multiplayer")
    ap.add_argument("--out", default="data/worldmodel/actions")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs, skipped = ex.build_manifest(args.replays, None, args.out)
    print(f"{len(jobs)} replays; skipped: {skipped}", flush=True)
    with Pool(args.workers) as pool:
        res = pool.map(one, [(j["path"], j["out"]) for j in jobs])
    status = {}
    for s, p in res:
        status[s.split(":")[0]] = status.get(s.split(":")[0], 0) + 1
        if s.startswith("fail"):
            print(os.path.basename(p), s, flush=True)
    print(status)


if __name__ == "__main__":
    main()
