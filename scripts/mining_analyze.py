"""Turns a mining_probe.py trace into mining-cycle timings and speed curves.

For each SCV it segments every trip (dropoff to dropoff) into:
    dwell   dropoff -> starts moving away from the CC
    out     starts moving -> stops at the patch
    mine    stops at the patch -> carry buff appears
    wait    carry buff appears -> starts moving back
    back    starts moving back -> carry buff disappears (dropoff)
and measures the path actually walked each way (sum of per-loop displacement),
which is a better x-axis than straight-line distance: minerals are 2x1
rectangles and workers approach at an angle.

Workers bounce: a worker sent to a patch may find it occupied (or reserved)
and wander to another, especially while the line settles. So a trip is
attributed to the patch it was actually mined from (the gather order target
at pickup), a trip is "bounced" if it saw more than one gather target or moved
between arriving and picking up, and only trips after a worker has OWNED a
patch (OWN_TRIPS consecutive clean trips on the same patch) are used.

For stacked patches (several workers on one patch, e.g. the pro split) it
also reports the pickup-to-pickup intervals on the patch, which show when the
next worker can start mining, and each patch's total income. If minerals.csv is
present it checks when the patch's contents drop relative to pickups.

Then reports per-patch averages, fits cycle_loops = overhead + path / speed,
estimates top speed and acceleration, and prints speed curves aligned on
leaving the CC, arriving at the patch and leaving the patch. Saves plots if
matplotlib is installed.

Usage: python scripts/mining_analyze.py [data/mining]
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

LOOPS_PER_SECOND = 22.4  # "Faster" game speed; matches ladder/real time
STOPPED = 1e-3  # per-loop displacement below this counts as standing still
ALIGN_WINDOW = 16  # loops to show either side of an aligned event
OWN_TRIPS = 3  # consecutive clean trips on one patch before a worker owns it
AT_PATCH = 1.0  # edge distance to a patch (circle approx.) that counts as "at" it
PHASES = ("dwell", "out", "mine", "wait", "back", "cycle", "out_len", "back_len")
EVENTS = ("leave CC", "arrive patch", "leave patch")


def load(out_dir):
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    rows = defaultdict(list)
    with open(os.path.join(out_dir, "trace.csv")) as f:
        for r in csv.DictReader(f):
            rows[int(r["tag"])].append(
                (int(r["loop"]), float(r["x"]), float(r["y"]), int(r["carrying"]), int(r["order_target"]))
            )
    traces = {}
    for tag, rs in rows.items():
        rs.sort()
        traces[tag] = dict(
            loop=np.array([r[0] for r in rs]),
            x=np.array([r[1] for r in rs]),
            y=np.array([r[2] for r in rs]),
            carry=np.array([r[3] for r in rs]),
            target=np.array([r[4] for r in rs], dtype=np.uint64),
        )
    return meta, traces


def edge_dist(x, y, obj, scv_radius):
    return np.hypot(x - obj["x"], y - obj["y"]) - obj["radius"] - scv_radius


def segment(t, minerals, scv_radius):
    """Returns (trips, speed). Each trip has phase durations in loops, path
    lengths, the patch actually mined, whether it bounced, and event indices."""
    speed = np.zeros_like(t["x"])
    speed[1:] = np.hypot(np.diff(t["x"]), np.diff(t["y"])) / np.diff(t["loop"])
    # Edge distance to the nearest patch; only used to recognise "at a patch".
    # Patch identity comes from the gather order target, because the circle
    # approximation misattributes back-row patches to the front row.
    d_min = np.stack([edge_dist(t["x"], t["y"], m, scv_radius) for m in minerals.values()]).min(axis=0)
    mineral_tags = set(np.array(list(minerals), dtype=np.uint64))

    dc = np.diff(t["carry"])
    pickups = np.where(dc == 1)[0] + 1
    dropoffs = np.where(dc == -1)[0] + 1

    trips = []
    for d0, d1 in zip(dropoffs[:-1], dropoffs[1:]):
        p = pickups[(pickups > d0) & (pickups < d1)]
        if len(p) != 1:
            continue
        p = p[0]
        moving = np.where(speed[d0:p] > STOPPED)[0]
        if len(moving) == 0:
            continue
        depart = d0 + moving[0]
        still = np.where((speed[depart:p] <= STOPPED) & (d_min[depart:p] < AT_PATCH))[0]
        if len(still) == 0:
            continue
        arrive = depart + still[0]
        moving_back = np.where(speed[p:d1 + 1] > STOPPED)[0]
        if len(moving_back) == 0:
            continue
        leave = p + moving_back[0]

        gather_targets = set(np.unique(t["target"][d0:p])) & mineral_tags
        mined = t["target"][p - 1]
        bounced = (
            len(gather_targets) != 1  # re-targeted mid-trip (or no target seen)
            or mined not in mineral_tags
            or np.any(speed[arrive:p] > STOPPED)  # walked off after arriving
        )
        trips.append(
            dict(
                patch=int(mined),
                bounced=bool(bounced),
                dwell=depart - d0,
                out=arrive - depart,
                mine=p - arrive,
                wait=leave - p,
                back=d1 - leave,
                cycle=d1 - d0,
                out_len=speed[depart:arrive].sum(),
                back_len=speed[leave:d1 + 1].sum(),
                events=(depart, arrive, leave),
                pickup=p,
            )
        )
    return trips, speed


def aligned(speed, idxs):
    w = ALIGN_WINDOW
    rows = [speed[i - w : i + w] for i in idxs if i - w >= 0 and i + w <= len(speed)]
    return np.mean(rows, axis=0) if rows else None


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/mining"
    meta, traces = load(out_dir)
    r = meta["scv_radius"]
    minerals = {m["tag"]: m for m in meta["minerals"]}

    results = []
    curves = {e: [] for e in EVENTS}
    owners = defaultdict(set)
    pickups = defaultdict(list)  # patch -> [(loop, worker)] from owned trips
    print(f"{'worker':>12} {'assigned':>12} {'trips':>5} {'bounced':>7} {'owned':>5}  first owned loop")
    for tag_str, assigned in meta["assignment"].items():
        tag = int(tag_str)
        t = traces[tag]
        trips, speed = segment(t, minerals, r)

        # Ownership: drop bounced trips, and the first OWN_TRIPS clean trips of
        # every run on one patch (the line is still settling during those).
        streak, prev = 0, None
        for tr in trips:
            if tr["bounced"]:
                streak, prev = 0, None
            else:
                streak = streak + 1 if tr["patch"] == prev else 1
                prev = tr["patch"]
            tr["owned"] = not tr["bounced"] and streak > OWN_TRIPS
        owned = [tr for tr in trips if tr["owned"]]
        first = t["loop"][owned[0]["events"][0]] if owned else "-"
        print(
            f"{tag:12d} {assigned:12d} {len(trips):5d} "
            f"{sum(tr['bounced'] for tr in trips):7d} {len(owned):5d}  {first}"
            + ("" if not owned or owned[0]["patch"] == assigned else f"  (mined {owned[0]['patch']})")
        )

        by_patch = defaultdict(list)
        for tr in owned:
            by_patch[tr["patch"]].append(tr)
        for patch, ptrips in by_patch.items():
            owners[patch].add(tag)
            pickups[patch] += [(int(t["loop"][tr["pickup"]]), tag) for tr in ptrips]
            avg = {k: np.mean([tr[k] for tr in ptrips]) for k in PHASES}
            results.append(dict(tag=tag, patch=patch, n=len(ptrips), speed=speed, **avg))
            for i, e in enumerate(EVENTS):
                curves[e].append(aligned(speed, [tr["events"][i] for tr in ptrips]))

    assigned_count = Counter(meta["assignment"].values())
    for patch, tags in owners.items():
        if len(tags) > assigned_count[patch]:
            print(f"NOTE: patch {patch} mined by {len(tags)} workers but {assigned_count[patch]} assigned (a worker moved here)")
    if not results:
        print("No owned trips; record longer or lower OWN_TRIPS.")
        return

    print()
    results.sort(key=lambda d: d["out_len"] + d["back_len"])
    print(
        f"{'patch':>12} {'trips':>5} {'path':>6} {'dwell':>6} {'out':>6} {'mine':>6} "
        f"{'wait':>6} {'back':>6} {'cycle':>6} {'min/min':>8}"
    )
    for d in results:
        rate = 5 / d["cycle"] * LOOPS_PER_SECOND * 60
        path = (d["out_len"] + d["back_len"]) / 2
        print(
            f"{d['patch']:12d} {d['n']:5d} {path:6.2f} {d['dwell']:6.1f} {d['out']:6.1f} {d['mine']:6.1f} "
            f"{d['wait']:6.1f} {d['back']:6.1f} {d['cycle']:6.1f} {rate:8.1f}"
        )
    print("(times in game loops; path = one-way distance walked; rate = minerals per real minute)")

    paths = np.array([d["out_len"] + d["back_len"] for d in results])  # round trip
    cycles = np.array([d["cycle"] for d in results])
    fit = None
    if np.ptp(paths) > 0.1:
        fit = np.polyfit(paths, cycles, 1)
        k, overhead = fit
        print(
            f"\ncycle_loops ~= {overhead:.1f} + {k:.2f} * round_trip_path"
            f"   (effective speed {1 / k:.4f} units/loop incl. accel/decel)"
        )

    per_patch = defaultdict(list)
    for d in results:
        per_patch[d["patch"]].append(d)
    print(f"\n{'patch':>12} {'workers':>7} {'path':>6} {'total min/min':>13}  pickup-to-pickup intervals (loops: count)")
    for patch, ds in sorted(per_patch.items(), key=lambda kv: kv[1][0]["out_len"]):
        loops = sorted(l for l, _ in pickups[patch])
        intervals = Counter(np.diff(loops).tolist()).most_common(3)
        total = sum(5 / d["cycle"] * LOOPS_PER_SECOND * 60 for d in ds)
        path = np.mean([(d["out_len"] + d["back_len"]) / 2 for d in ds])
        print(f"{patch:12d} {len(ds):7d} {path:6.2f} {total:13.1f}  {intervals}")
    print("(on a stacked patch, short interval - 45 = loops the next worker waits after the previous pickup)")

    contents_path = os.path.join(out_dir, "minerals.csv")
    if os.path.exists(contents_path):
        changes = defaultdict(list)
        with open(contents_path) as f:
            for row in csv.DictReader(f):
                changes[int(row["tag"])].append(int(row["loop"]))
        offsets = Counter(
            min((c - l for c in changes[patch]), key=abs)
            for patch, pk in pickups.items() if changes[patch]
            for l, _ in pk
        )
        print(f"patch contents drop at (loops after pickup): {offsets.most_common(3)}")

    all_speed = np.concatenate([d["speed"] for d in results])
    vmax = all_speed.max()
    print(f"top speed observed: {vmax:.4f} units/loop = {vmax * LOOPS_PER_SECOND:.3f} units/s")

    means = {}
    for e in EVENTS:
        cs = [c for c in curves[e] if c is not None]
        if not cs:
            continue
        means[e] = np.mean(cs, axis=0)
        print(f"\nSpeed (units/loop) aligned on {e}, loops {-ALIGN_WINDOW}..{ALIGN_WINDOW - 1}:")
        print(" ".join(f"{v:.3f}" for v in means[e]))
    if "leave patch" in means:
        ramp = np.diff(means["leave patch"][ALIGN_WINDOW : ALIGN_WINDOW + 8])
        print(f"\nacceleration from standstill ~= {np.median(ramp):.4f} units/loop^2")
    if "arrive patch" in means:
        ramp = -np.diff(means["arrive patch"][ALIGN_WINDOW - 10 : ALIGN_WINDOW - 1])
        print(f"deceleration into patch     ~= {np.median(ramp):.4f} units/loop^2")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nInstall matplotlib for plots.")
        return

    cc = meta["command_center"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for d in (results[0], results[-1]):
        t = traces[d["tag"]]
        dist = edge_dist(t["x"], t["y"], cc, r)
        sl = slice(600, 900)
        label = f"path {(d['out_len'] + d['back_len']) / 2:.2f}"
        ax[0, 0].plot(t["loop"][sl], dist[sl], label=label)
        ax[1, 0].plot(t["loop"][sl], d["speed"][sl], label=label)
    ax[0, 0].set(title="Distance from CC edge", xlabel="game loop", ylabel="units")
    ax[1, 0].set(title="Speed", xlabel="game loop", ylabel="units / loop")
    ax[0, 0].legend(); ax[1, 0].legend()

    xs = np.arange(-ALIGN_WINDOW, ALIGN_WINDOW)
    for e, mean in means.items():
        ax[0, 1].plot(xs, mean, marker=".", label=e)
    ax[0, 1].axvline(0, color="gray", lw=0.5)
    ax[0, 1].set(title="Mean speed aligned on events", xlabel="loops from event", ylabel="units / loop")
    ax[0, 1].legend()

    ax[1, 1].scatter(paths, cycles)
    if fit is not None:
        g = np.linspace(paths.min(), paths.max(), 10)
        ax[1, 1].plot(g, np.polyval(fit, g), "--")
    ax[1, 1].set(title="Cycle time vs round-trip path", xlabel="round-trip path walked (units)", ylabel="loops / trip")

    fig.tight_layout()
    png = os.path.join(out_dir, "mining.png")
    fig.savefig(png, dpi=120)
    print(f"\nSaved {png}")


if __name__ == "__main__":
    main()
