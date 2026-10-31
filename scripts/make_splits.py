"""Fixed train / held-out split for the world model, written once and reused by every run.

    held-out maps    every game on these maps is left out: does the model analyse maps it has
                     never seen, or memorize the ones it has?
    held-out games   a seeded random share of each remaining map's games: does it generalize to
                     matches it hasn't seen?
    train            everything else

Games are ids from the replay tree (worldmodel/corpus.py), so a game's split doesn't depend
on whether it's extracted yet, and one split serves every view of a game.

    python scripts/make_splits.py --format 1v1 --maps "Ley Lines LE,Pylon LE" --out data/games/splits/default.json
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from cosmicstar.worldmodel import corpus  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default=corpus.REPLAYS, help="the replay tree")
    ap.add_argument("--format", default="1v1", help="games of this format (empty = all)")
    ap.add_argument("--maps", default="", help="comma-separated maps held out entirely: names, slugs or hash prefixes "
                    "(a name takes every version of that map)")
    ap.add_argument("--game_share", type=float, default=0.05, help="share of each remaining map's games held out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(corpus.GAMES, "splits", "default.json"))
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; splits are fixed once made (delete it deliberately to remake)")
    index = corpus.load_index(args.replays, rebuild=True)
    names = [m.strip() for m in args.maps.split(",") if m.strip()]
    held_maps = corpus.map_dirs_named(index, names)  # map directories (hashes): every version of a named map
    by_map, map_name = collections.defaultdict(list), {}
    for gid, d in index.items():
        if not args.format or d["format"] == args.format:
            by_map[corpus.map_dir(d["map_hash"])].append(gid)
            map_name[corpus.map_dir(d["map_hash"])] = d["map"]
    if names and not held_maps:
        sys.exit(f"no games on {names}; maps: {sorted(set(map_name.values()))}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.RandomState(args.seed)
    split = dict(held_out_maps={m: map_name.get(m) for m in sorted(held_maps)}, held_out_map_games=[], held_out_games=[],
                 train=[], seed=args.seed, game_share=args.game_share, format=args.format)
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
          f"held-out maps {split['held_out_maps']}: {len(split['held_out_map_games'])} games -> {args.out}")


if __name__ == "__main__":
    main()
