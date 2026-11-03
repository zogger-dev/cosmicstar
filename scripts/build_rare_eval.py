"""Rare-state evaluation set (2026-11-01): fixed frames from the held-out caches (held-out games and
held-out maps), chosen to contain each rare entity state, so its recall / F1 is measured on enough
units. Random evaluation frames almost never contain them: a blip appears in ~0.6% of view-frames,
cloaked / detected units in fewer.

States: the binary flags blip, cloaked, burrowed, halluc, engaged, has_add_on; the categorical
states cloak:cloaked, cloak:detected, inside:passenger, inside:in_gas, display:hidden; units inside a
container (the container pointer). Up to --per_state frames each (player views and the combined
view), with fixed player codes. Written as one pickle: {"frames": [...], "tags": [[state, ...]],
"counts": {state: units}}.

    python scripts/build_rare_eval.py --cache data/worldmodel/frame_cache --out data/worldmodel/rare_eval.pkl
"""
import argparse
import collections
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from cosmicstar.worldmodel import frame_cache as fc  # noqa: E402
from cosmicstar.worldmodel.rare_eval import STATES, states_of  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/worldmodel/frame_cache")
    ap.add_argument("--out", default="data/worldmodel/rare_eval.pkl")
    ap.add_argument("--per_state", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    c = fc.FrameCache(args.cache, ["held_out_games", "held_out_map_games"])
    rng = np.random.RandomState(args.seed)
    want = STATES
    got = collections.defaultdict(int)
    frames, tags, units = [], [], collections.Counter()
    for i in rng.permutation(len(c)):
        if all(got[s] >= args.per_state for s in want):
            break
        w = c.read(int(i))
        for fr in w[rng.randint(len(w))]:  # one frame of the entry (a window's frames are near-duplicates), its 3 views
            st = states_of(fr)
            need = [s for s in want if st[s] > 0 and got[s] < args.per_state]
            if not need:
                continue
            frames.append(fr)
            tags.append([s for s in want if st[s] > 0])
            for s in want:
                if st[s] > 0:
                    got[s] += 1
                    units[s] += st[s]
    pickle.dump(dict(frames=frames, tags=tags, counts=dict(units), frames_per_state=dict(got)), open(args.out, "wb"), protocol=5)
    print(f"{len(frames)} frames -> {args.out}")
    for s in want:
        print(f"  {s:18s} frames {got[s]:3d}  units {units[s]:6d}")


if __name__ == "__main__":
    main()
