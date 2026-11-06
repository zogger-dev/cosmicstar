"""Builds each game's <base>.index.npz for the physics model's training store (worldmodel/physics_data.py): rows' first and last
frames, death frames and parents. Skips games that already have one and games missing a view.

    python scripts/build_physics_index.py --games data/games_v3 --workers 12
"""

import argparse
import glob
import os
from multiprocessing import Pool

from cosmicstar.worldmodel import physics_data


def one(base):
    replay = base.replace(ARGS.games, ARGS.replays, 1) + ".SC2Replay"
    try:
        return base, physics_data.build_index(base, replay), None
    except Exception as ex:
        return base, 0, repr(ex)[:200]


ap = argparse.ArgumentParser()
ap.add_argument("--games", default="data/games_v3")
ap.add_argument("--replays", default="data/replays")
ap.add_argument("--workers", type=int, default=12)
ARGS = ap.parse_args()

if __name__ == "__main__":
    bases = [
        u[: -len(".p1.units.arrow")]
        for u in sorted(
            glob.glob(os.path.join(ARGS.games, "1v1", "*", "*", "*.p1.units.arrow"))
        )
    ]
    bases = [
        b
        for b in bases
        if not os.path.exists(b + ".index.npz")
        and all(os.path.exists(f"{b}.p{s}.units.arrow") for s in physics_data.VIEWS)
    ]
    print(len(bases), "games to index", flush=True)
    done = failed = 0
    with Pool(ARGS.workers) as pool:
        for base, frames, err in pool.imap_unordered(one, bases):
            done += 1
            if err:
                failed += 1
                print("fail", base, err, flush=True)
    print("done", done, "failed", failed, flush=True)
