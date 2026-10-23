"""Plots velocity and acceleration, loop by loop, for one mining cycle on the
nearest and farthest patch in a mining_probe.py trace.

Velocity is signed along the dropoff -> patch axis (+ = toward the patch,
- = toward the CC), so turnarounds show up as sign changes.

Usage: python scripts/mining_trip_plot.py [data/mining] [trip_index]
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import mining_analyze as ma

PHASE_COLORS = {"out": "#dbe9f6", "mine": "#f6e3c8", "wait": "#eeeeee", "back": "#dcefdc"}


def cycle(t, tr, pad=4):
    """Index range and signed velocity for one dropoff -> dropoff cycle."""
    dep, arr, leave = tr["events"]
    d0 = dep - tr["dwell"]
    d1 = leave + tr["back"]
    axis = np.array([t["x"][arr] - t["x"][d0], t["y"][arr] - t["y"][d0]])
    axis /= np.linalg.norm(axis)
    idx = np.arange(d0 - pad, d1 + pad + 1)
    disp = np.stack([np.diff(t["x"][idx[0] - 1 : idx[-1] + 1]), np.diff(t["y"][idx[0] - 1 : idx[-1] + 1])])
    vel = axis @ disp
    phases = dict(out=(dep, arr), mine=(arr, arr + tr["mine"]), wait=(leave - tr["wait"], leave), back=(leave, d1))
    return idx, vel, phases, d0, d1


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/mining"
    trip_index = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    meta, traces = ma.load(out_dir)
    minerals = {m["tag"]: m for m in meta["minerals"]}

    per_worker = []
    for tag_str in meta["assignment"]:
        t = traces[int(tag_str)]
        trips, _ = ma.segment(t, minerals, meta["scv_radius"])
        clean = [tr for tr in trips if not tr["bounced"]]
        if len(clean) > trip_index:
            per_worker.append((clean[trip_index]["out_len"], t, clean[trip_index]))
    per_worker.sort(key=lambda w: w[0])
    picks = [("near", *per_worker[0]), ("far", *per_worker[-1])]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")
    for col, (name, path, t, tr) in enumerate(picks):
        idx, vel, phases, d0, d1 = cycle(t, tr)
        loops = t["loop"][idx] - t["loop"][d0]
        acc = np.diff(vel, prepend=vel[0])
        for row, (y, label) in enumerate(((vel, "velocity (units/loop)"), (acc, "acceleration (units/loop²)"))):
            ax = axes[row, col]
            for ph, (a, b) in phases.items():
                ax.axvspan(t["loop"][a] - t["loop"][d0], t["loop"][b] - t["loop"][d0], color=PHASE_COLORS[ph], lw=0)
            if row == 0:
                ax.plot(loops, y, marker=".", ms=3)
                ax.axhline(0.1758, ls=":", c="gray"); ax.axhline(-0.1758, ls=":", c="gray")
            else:
                ax.bar(loops, y, width=0.8)
            ax.axhline(0, c="black", lw=0.5)
            for x in (0, t["loop"][d1] - t["loop"][d0]):
                ax.axvline(x, c="red", lw=0.8)
            ax.set_ylabel(label)
        axes[0, col].set_title(f"{name} patch, one-way path {path:.2f} — red = dropoff")
        axes[1, col].set_xlabel("game loops since dropoff")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PHASE_COLORS.values()]
    fig.legend(handles, ["outbound", "mining", "pause after mining", "return"], loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(out_dir, "trip_kinematics.png")
    fig.savefig(png, dpi=120)
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
