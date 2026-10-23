"""Fixed train / held-out split for the world model, written once and reused by every run.

    held-out maps    every game on these maps is left out: does the model analyse maps it has
                     never seen, or memorize the ones it has?
    held-out games   a seeded random share of each remaining map's games: does it generalize to
                     matches it hasn't seen?
    train            everything else

Files are the extracted state file names (the same names as the action files), so a game's
split doesn't depend on whether it's extracted yet.

    python scripts/make_splits.py --maps "Arctic Flowers LE,Emerald City CE"
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import extract_states as ex  # noqa: E402  (same replay selection and file names as the extraction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default=r"C:\Users\prade\Documents\StarCraft II\Accounts\52583725\1-S2-1-1544307\Replays\Multiplayer")
    ap.add_argument("--maps", default="Arctic Flowers LE,Emerald City CE", help="maps held out entirely")
    ap.add_argument("--game_share", type=float, default=0.05, help="share of each remaining map's games held out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/worldmodel/splits.json")
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; splits are fixed once made (delete it deliberately to remake)")
    held_maps = [m.strip() for m in args.maps.split(",") if m.strip()]
    jobs, _ = ex.build_manifest(args.replays, None, "data/worldmodel/obs")
    by_map = collections.defaultdict(list)
    for j in jobs:
        by_map[ex.replay_info(j["path"])["map"]].append(os.path.basename(j["out"]))
    unknown = set(held_maps) - set(by_map)
    if unknown:
        sys.exit(f"no games on {unknown}; maps: {sorted(by_map)}")
    rng = np.random.RandomState(args.seed)
    split = dict(held_out_maps=held_maps, held_out_map_games=[], held_out_games=[], train=[],
                 seed=args.seed, game_share=args.game_share)
    for m in sorted(by_map):
        files = sorted(by_map[m])
        if m in held_maps:
            split["held_out_map_games"] += files
            continue
        k = int(round(args.game_share * len(files)))
        pick = set(rng.choice(len(files), k, replace=False).tolist()) if k else set()
        split["held_out_games"] += [f for i, f in enumerate(files) if i in pick]
        split["train"] += [f for i, f in enumerate(files) if i not in pick]
    json.dump(split, open(args.out, "w"), indent=1)
    print(f"train {len(split['train'])} | held-out games {len(split['held_out_games'])} | "
          f"held-out maps {held_maps}: {len(split['held_out_map_games'])} games -> {args.out}")


if __name__ == "__main__":
    main()
