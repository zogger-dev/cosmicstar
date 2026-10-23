"""Compares zerg_bench runs: build B minus build A, per spawn and averaged.

Usage: python scripts/zerg_bench_compare.py data/zerg_bench/AcropolisLE [--a 13ov --b 15ov]
"""

import argparse
import csv
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")  # Δ in the tables on Windows consoles
from collections import defaultdict

import numpy as np

LPS = 22.4

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("--a", default="13ov")
parser.add_argument("--b", default="15ov")
args = parser.parse_args()


def load(run_dir):
    with open(os.path.join(run_dir, "timeseries.csv")) as f:
        rows = list(csv.DictReader(f))
    ts = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    ts["all_drones"] = ts["drones"] + ts["drones_training"]
    events = defaultdict(list)
    with open(os.path.join(run_dir, "events.csv")) as f:
        for r in csv.DictReader(f):
            events[r["event"]].append((int(r["loop"]), r["detail"]))
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    return ts, events, meta


runs = defaultdict(dict)  # spawn -> build -> run
for d in sorted(glob.glob(os.path.join(args.dir, "*_spawn*"))):
    build, spawn = os.path.basename(d).rsplit("_spawn", 1)
    if os.path.exists(os.path.join(d, "timeseries.csv")):
        runs[int(spawn)][build] = load(d)
spawns = [s for s in sorted(runs) if args.a in runs[s] and args.b in runs[s]]
if not spawns:
    raise SystemExit("no spawn has both builds")


def at(ts, key, loop):
    i = np.searchsorted(ts["loop"], loop)
    return ts[key][min(i, len(ts[key]) - 1)]


def income(ts, window=int(30 * LPS)):
    c = ts["collected_minerals"]
    r = np.full_like(c, np.nan)
    r[window:] = (c[window:] - c[:-window]) / window * LPS * 60
    return r


# ---------- event timings ----------
KEY_EVENTS = ["overlord", "hatch natural started", "extractor main started", "pool started", "hatch natural done",
              "pool done", "queen", "lings", "pull gas", "ling speed", "hatch third started", "natural saturated",
              "hatch third done", "supply blocked"]
print(f"Event times (s), {args.a} vs {args.b}")
for s in spawns:
    print(f"  spawn {s}:")
    for e in KEY_EVENTS:
        cols = []
        for b in (args.a, args.b):
            ev = runs[s][b][1].get(e, [])
            cols.append(", ".join(
                f"{l / LPS:.1f}" + (f" ({d.split('s from')[0]}s)" if e == "supply blocked" else "") for l, d in ev[:6]) or "-")
        print(f"    {e:<22} {cols[0]:<44} {cols[1]}")

# ---------- differences at checkpoints ----------
print(f"\n{args.b} minus {args.a}")
print(f"{'time':>5} {'spawn':>5} {'Δmined':>8} {'Δdrones':>8} {'Δdrone-s':>9} {'Δbank':>7} {'Δblocked-s':>10} {'income A':>9} {'income B':>9}")
for t in (60, 90, 120, 150, 180, 210, 240):
    loop = t * LPS
    for s in spawns + ["mean"]:
        ss = spawns if s == "mean" else [s]
        vals = []
        for sp in ss:
            A, B = runs[sp][args.a][0], runs[sp][args.b][0]
            if loop > A["loop"][-1] or loop > B["loop"][-1]:
                break
            vals.append([
                at(B, "collected_minerals", loop) - at(A, "collected_minerals", loop),
                at(B, "all_drones", loop) - at(A, "all_drones", loop),
                at(B, "drone_seconds", loop) - at(A, "drone_seconds", loop),
                at(B, "minerals", loop) - at(A, "minerals", loop),
                (at(B, "blocked_loops", loop) - at(A, "blocked_loops", loop)) / LPS,
            ])
        if len(vals) != len(ss):
            continue
        v = np.mean(vals, axis=0)
        ia = np.nanmean([income(runs[sp][args.a][0])[int(np.searchsorted(runs[sp][args.a][0]["loop"], loop))] for sp in ss])
        ib = np.nanmean([income(runs[sp][args.b][0])[int(np.searchsorted(runs[sp][args.b][0]["loop"], loop))] for sp in ss])
        print(f"{t // 60}:{t % 60:02d} {str(s):>5} {v[0]:8.1f} {v[1]:8.1f} {v[2]:9.1f} {v[3]:7.1f} {v[4]:10.1f} {ia:9.0f} {ib:9.0f}")
print("(Δmined = total minerals mined; Δdrones = drones + drone eggs; Δdrone-s = cumulative drone-seconds (drones alive x time);")
print(" Δblocked-s = supply-blocked time with idle larva; income = minerals/min over the last 30 s)")

# ---------- plots ----------
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("Install matplotlib for plots.")

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for k, s in enumerate(spawns):
    A, B = runs[s][args.a][0], runs[s][args.b][0]
    n = min(len(A["loop"]), len(B["loop"]))
    t = A["loop"][:n] / LPS
    c = colors[k]
    ax[0, 0].plot(t, B["collected_minerals"][:n] - A["collected_minerals"][:n], c=c, label=f"spawn {s}")
    ax[0, 1].plot(t, B["all_drones"][:n] - A["all_drones"][:n], c=c, label=f"spawn {s}")
    ax[1, 0].plot(t, income(A)[:n], c=c, ls="--", label=f"{args.a} spawn {s}")
    ax[1, 0].plot(t, income(B)[:n], c=c, label=f"{args.b} spawn {s}")
    ax[1, 1].plot(t, B["drone_seconds"][:n] - A["drone_seconds"][:n], c=c, label=f"spawn {s}")
for a in ax.flat:
    a.set_xlabel("game time (s)")
    a.legend()
    a.grid(alpha=0.3)
ax[0, 0].axhline(0, c="k", lw=0.6)
ax[0, 1].axhline(0, c="k", lw=0.6)
ax[0, 0].set(title=f"Total minerals mined: {args.b} − {args.a}", ylabel="minerals")
ax[0, 1].set(title=f"Drones (incl. eggs): {args.b} − {args.a}", ylabel="drones")
ax[1, 0].set(title="Mineral income (30 s window)", ylabel="minerals / min")
ax[1, 1].set(title=f"Drone-seconds: {args.b} − {args.a}", ylabel="drone-seconds")
ax[1, 1].axhline(0, c="k", lw=0.6)
fig.suptitle(f"{os.path.basename(os.path.normpath(args.dir))}: {args.a} vs {args.b}")
fig.tight_layout()
png = os.path.join(args.dir, f"compare_{args.a}_vs_{args.b}.png")
fig.savefig(png, dpi=120)
print(f"\nSaved {png}")
