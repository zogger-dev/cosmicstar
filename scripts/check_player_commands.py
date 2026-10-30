"""Reports, for player-view state files, how well the engine's commands join to the replay's
command stream and how well the selection tracker reproduces the engine's control groups,
selection and command recipients (worldmodel/command_join.py, worldmodel/selection_tracker.py).

    python scripts/check_player_commands.py data/worldmodel/pov/*_p[123].npz --actions data/worldmodel/actions
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import extract_states as ex  # noqa: E402
from cosmicstar.worldmodel import selection_tracker  # noqa: E402


def rate(st, k, total):
    return f"{st.get(k, 0) / st[total]:.4f}" if st.get(total) else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--actions", default="data/worldmodel/actions")
    ap.add_argument("--replays", help="replay directory: take death loops from the replays' own events instead of the "
                    "state files' per-frame dead lists (which are up to a frame late)")
    ap.add_argument("--verbose", action="store_true", help="print the first mismatches of each kind")
    args = ap.parse_args()
    total = {}
    for f in args.files:
        name, slot = os.path.basename(f)[:-4].rsplit("_p", 1)
        slot = int(slot)
        pov = dict(np.load(f))
        if "cmd_loop" not in pov:  # an observer file or a composed view: no engine commands to check against
            print(f"{name} p{slot}: no cmd_* arrays (composed or observer view), skipped", flush=True)
            continue
        acts = dict(np.load(os.path.join(args.actions, name + ".npz")))
        if "addsub_prio" not in acts:
            print(f"{name}: the action table predates the group masks / priorities; re-run extract_actions.py", flush=True)
            continue
        died = None
        if args.replays:
            path = [q for q in glob.glob(os.path.join(args.replays, "**", "*.SC2Replay"), recursive=True) if ex.out_name(q)[:-4] == name]
            if path:
                died = selection_tracker.deaths_by_replay_tag(ex.tracker_lifecycle(path[0], 10 ** 9)[1])
        st = selection_tracker.validate(pov, acts, slot, died, verbose=args.verbose)
        for k, v in st.items():
            if k != "mismatches":
                total[k] = total.get(k, 0) + int(v)
        print(f"{name} p{slot}: join {rate(st, 'join_engine_matched', 'join_engine')} of {st['join_engine']} engine commands, "
              f"{rate(st, 'join_cmd_rows_matched', 'join_cmd_rows')} of {st['join_cmd_rows']} CMD rows | groups: count "
              f"{rate(st, 'grp_count', 'cells')}, leader {rate(st, 'grp_leader', 'grp_nonempty')} of {st['grp_nonempty']} "
              f"non-empty | selection {rate(st, 'sel', 'frames')} | recipients exact {rate(st, 'cmd_exact', 'cmd_matched')}, "
              f"subset {rate(st, 'cmd_subset', 'cmd_matched')}, order {rate(st, 'cmd_order', 'cmd_matched')}, none "
              f"{rate(st, 'cmd_none', 'cmd_matched')} | engine selection: events {rate(st, 'ev_eq', 'ev')} of {st.get('ev', 0)} "
              f"(group ops {rate(st, 'ev_group_eq', 'ev_group')} of {st.get('ev_group', 0)}), recipients = selection "
              f"{rate(st, 'cmd_sel_eq', 'cmd_sel')}, subset {rate(st, 'cmd_sel_subset', 'cmd_sel')} | tracker errors "
              f"{st['tracker_errors']}", flush=True)
    if len(args.files) > 1 and total:
        print("all:", json.dumps({k: v for k, v in total.items()}))
        st = total
        print(f"all: join {rate(st, 'join_engine_matched', 'join_engine')} engine, {rate(st, 'join_cmd_rows_matched', 'join_cmd_rows')} "
              f"CMD rows | groups count {rate(st, 'grp_count', 'cells')}, leader {rate(st, 'grp_leader', 'grp_nonempty')} | "
              f"selection {rate(st, 'sel', 'frames')} | recipients exact {rate(st, 'cmd_exact', 'cmd_matched')}, "
              f"subset {rate(st, 'cmd_subset', 'cmd_matched')} | engine selection: events {rate(st, 'ev_eq', 'ev')}, "
              f"group ops {rate(st, 'ev_group_eq', 'ev_group')}, recipients = selection {rate(st, 'cmd_sel_eq', 'cmd_sel')}")


if __name__ == "__main__":
    main()
