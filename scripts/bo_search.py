"""Grid-searches a build-order template with scripts/zerg_bench.py, in parallel.

The template is a .bo file with {placeholders}; the grid is a JSON object
{placeholder: [values, ...]} (or a list of such objects, each a separate grid).
Every combination is rendered, played on every (map, seed) given, and scored:

    roaches_done   loop the last of the scripted roaches popped (if any)
    drones_240     drones + drone eggs at 4:00 (the objective)
    mined_240      minerals mined by 4:00 (tie-break)

A variant is feasible if every run has its roaches out by --roach_deadline.
Results go to <out>/results.csv; the best feasible variants are printed.

Usage:
    python scripts/bo_search.py builds/2v2_roach.bo.tmpl grid.json --maps data/maps/ReclamationLE.SC2Map:1 ...
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

LPS = 22.4
HERE = os.path.dirname(os.path.abspath(__file__))


def variants(grid):
    grids = grid if isinstance(grid, list) else [grid]
    for g in grids:
        keys = list(g)
        for combo in itertools.product(*(g[k] for k in keys)):
            yield dict(zip(keys, combo))


def name_of(v):
    """Short, stable directory name (Windows paths max out at 260 chars)."""
    import hashlib
    return "v" + hashlib.sha1(json.dumps(v, sort_keys=True).encode()).hexdigest()[:10]


def metrics(run_dir, roach_n):
    ts = list(csv.DictReader(open(os.path.join(run_dir, "timeseries.csv"))))
    ev = list(csv.DictReader(open(os.path.join(run_dir, "events.csv"))))
    i = min(int(240 * LPS), len(ts)) - 1
    m = dict(
        drones_240=float(ts[i]["drones"]) + float(ts[i]["drones_training"]),
        mined_240=float(ts[i]["collected_minerals"]),
        gas_240=float(ts[i]["collected_vespene"]),
        blocked_s=float(ts[i]["blocked_loops"]) / LPS,
        capped_s=float(ts[i].get("capped_loops", "nan")) / LPS,
        capped_post_s=(float(ts[i].get("capped_loops", "nan")) - float(ts[int(70 * LPS)].get("capped_loops", "nan"))) / LPS,
        hold_capped_s=float(ts[i].get("hold_capped_loops", "nan")) / LPS,
    )
    pops = [int(e["loop"]) for e in ev if e["event"] == "roaches pop" and int(e["detail"].split()[0]) >= roach_n]
    m["roaches_done"] = pops[0] / LPS if pops else float("nan")
    for key, name in (("roaches started", "roach_start"), ("hatch third started", "hatch3"),
                      ("roach warren started", "warren"), ("roach warren done", "warren_done")):
        hits = [int(e["loop"]) / LPS for e in ev if e["event"] == key]
        m[name] = hits[0] if hits else float("nan")
    # Larva held for the roaches: game time between the hold starting and the
    # roaches starting during which larva actually sat there.
    start = [e for e in ev if e["event"] == "roaches started"]
    m["larva_hold_s"] = float("nan")
    if start:
        held_from = float(start[0]["detail"].split("held larva from ")[1].rstrip("s"))
        a, b = int(held_from * LPS), int(start[0]["loop"])
        m["larva_hold_s"] = sum(1 for r in ts[max(a - 1, 0):b] if int(r["larva"]) > 0) / LPS
    exts = [int(e["loop"]) / LPS for e in ev if e["event"].startswith("extractor") and e["event"].endswith("started")]
    m["extractors"] = " ".join(f"{x:.0f}" for x in exts)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("template")
    ap.add_argument("grid", help="JSON file or inline JSON")
    ap.add_argument("--maps", nargs="+", required=True, help="map_path:seed ...")
    ap.add_argument("--team_size", type=int, default=2)
    ap.add_argument("--loops", type=int, default=int(240 * LPS) + 10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--roach_n", type=int, default=6)
    ap.add_argument("--roach_deadline", type=float, default=235.0)
    ap.add_argument("--max_larva_hold", type=float, default=5.0, help="max seconds larva may wait for the roaches")
    ap.add_argument("--max_capped_post", type=float, default=None,
                    help="instead of --max_larva_hold: max hatchery-seconds at 3 larva after 1:10")
    ap.add_argument("--out", default="data/bo_search/run")
    ap.add_argument("--score_only", action="store_true", help="don't run anything; re-score finished runs")
    args = ap.parse_args()

    grid = json.load(open(args.grid)) if os.path.exists(args.grid) else json.loads(args.grid)
    template = open(args.template, encoding="utf-8").read()
    os.makedirs(args.out, exist_ok=True)
    env = dict(os.environ, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
    jobs = []
    for v in variants(grid):
        vdir = os.path.join(args.out, name_of(v))
        os.makedirs(vdir, exist_ok=True)
        json.dump(v, open(os.path.join(vdir, "params.json"), "w"), indent=1)
        bo = os.path.join(vdir, "build.bo")
        with open(bo, "w", encoding="utf-8") as f:
            f.write(template.format(**v))
        for ms in args.maps:
            mp, seed = ms.rsplit(":", 1)
            jobs.append((v, bo, mp, seed, os.path.join(vdir, f"{os.path.splitext(os.path.basename(mp))[0][:12]}_{seed}")))

    def run(job):
        v, bo, mp, seed, rdir = job
        if os.path.exists(os.path.join(rdir, "events.csv")):
            return job, 0  # already done (resumable)
        r = subprocess.run([sys.executable, os.path.join(HERE, "zerg_bench.py"), "--build", bo, "--map", mp,
                            "--team_size", str(args.team_size), "--seed", seed, "--loops", str(args.loops),
                            "--out_dir", rdir], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return job, r.returncode

    if args.score_only:
        jobs = []
    print(f"{len(jobs)} runs ({len(jobs) // len(args.maps)} variants x {len(args.maps)} maps/seeds), {args.workers} at a time", flush=True)
    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for job, rc in pool.map(run, jobs):
            done += 1
            if rc:
                print(f"  run failed ({rc}): {job[4]}", flush=True)
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)

    rows = []
    for v in variants(grid):
        vdir = os.path.join(args.out, name_of(v))
        ms = []
        for m in args.maps:
            mp, seed = m.rsplit(":", 1)
            rdir = os.path.join(vdir, f"{os.path.splitext(os.path.basename(mp))[0][:12]}_{seed}")
            try:
                ms.append(metrics(rdir, args.roach_n))
            except FileNotFoundError:
                pass
        if not ms:
            continue
        row = dict(v)
        for k in ("drones_240", "mined_240", "gas_240", "blocked_s", "roaches_done", "roach_start", "warren", "warren_done",
                  "hatch3", "larva_hold_s", "capped_s", "hold_capped_s", "capped_post_s"):
            vals = [m[k] for m in ms]
            row[k] = round(float(np.mean(vals)), 1)
            if k == "roaches_done":
                row["roaches_latest"] = round(float(np.max(vals)), 1)
        row["extractors"] = " | ".join(m["extractors"] for m in ms)
        row["larva_hold_max"] = round(float(np.max([m["larva_hold_s"] for m in ms])), 1)
        if args.max_capped_post is not None:
            row["feasible"] = len(ms) == len(args.maps) and all(
                m["roaches_done"] <= args.roach_deadline and m["capped_post_s"] <= args.max_capped_post for m in ms)
        else:
            row["feasible"] = len(ms) == len(args.maps) and all(
                m["roaches_done"] <= args.roach_deadline and m["larva_hold_s"] <= args.max_larva_hold
                and not m["hold_capped_s"] > 0.5 for m in ms)
        rows.append(row)

    if not rows:
        raise SystemExit("no finished runs")
    rows.sort(key=lambda r: (not r["feasible"], -r["drones_240"], r["capped_s"] if r["capped_s"] == r["capped_s"] else 0,
                             -r["mined_240"]))
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{sum(r['feasible'] for r in rows)}/{len(rows)} variants feasible (roaches out by {args.roach_deadline}s and larva held"
          f" <= {args.max_larva_hold}s on every map)")
    keys = list(grid[0] if isinstance(grid, list) else grid)
    for r in rows[:15]:
        print(("OK  " if r["feasible"] else "--  ") + "  ".join(f"{k}={r[k]}" for k in keys)
              + f" | drones@4:00 {r['drones_240']}  mined {r['mined_240']:.0f}  roaches out {r['roaches_done']}s"
              f" (latest {r['roaches_latest']})  larva held {r['larva_hold_max']}s (capped {r['hold_capped_s']}s)"
              f"  capped total {r['capped_s']}s  warren {r['warren']}  hatch3 {r['hatch3']}  blocked {r['blocked_s']}s")


if __name__ == "__main__":
    main()
