"""Runs every build from every spawn on a map with scripts/zerg_bench.py.

The spawn is decided by the game's random seed, so for each (build, spawn) it
tries seeds until one gives that spawn (zerg_bench exits with code 3 early
otherwise). Seeds found are cached in <out>/seeds.json.

Usage:
    python scripts/zerg_bench_matrix.py --map Ladder2019Season3/AcropolisLE.SC2Map
Then: python scripts/zerg_bench_compare.py data/zerg_bench/AcropolisLE
"""

import argparse
import json
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--map", default="Ladder2019Season3/AcropolisLE.SC2Map")
parser.add_argument("--builds", default="13ov,15ov")
parser.add_argument("--spawns", type=int, default=2)
parser.add_argument("--loops", type=int, default=int(4 * 60 * 22.4))
parser.add_argument("--out", default=None)
args = parser.parse_args()

map_name = os.path.splitext(os.path.basename(args.map))[0]
out = args.out or os.path.join("data", "zerg_bench", map_name)
os.makedirs(out, exist_ok=True)
seeds_path = os.path.join(out, "seeds.json")
seeds = json.load(open(seeds_path)) if os.path.exists(seeds_path) else {}
env = dict(os.environ, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
bench = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zerg_bench.py")

for build in args.builds.split(","):
    for spawn in range(args.spawns):
        run_dir = os.path.join(out, f"{build}_spawn{spawn}")
        seed = seeds.get(str(spawn), 1)
        while True:
            print(f"== {build} spawn {spawn}: seed {seed}", flush=True)
            r = subprocess.run(
                [sys.executable, bench, "--map", args.map, "--build", build, "--seed", str(seed),
                 "--spawn", str(spawn), "--loops", str(args.loops), "--out_dir", run_dir],
                env=env, stderr=subprocess.DEVNULL,
            )
            if r.returncode == 0:
                seeds[str(spawn)] = seed
                json.dump(seeds, open(seeds_path, "w"), indent=2)
                break
            if r.returncode != 3:
                sys.exit(f"zerg_bench failed with code {r.returncode}")
            seed += 1
