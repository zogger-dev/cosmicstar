"""Joins the engine's executed commands (a player-view state file's cmd_* arrays) to the
replay's command events (replay_actions.py rows), so that observer-view training can use
the recipients: each row of the action table is the same for every view.

The engine reports each command with the exact loop it was issued at (Action.game_loop),
its API ability id and its target; the replay has the loop, its own (link, cmd) ability
numbering and the same target. A CMD row with a target pairs with an engine command at the
same loop and target (points within 0.5, unit tags exact, no target: no target); a REPEAT
row with the engine command of the same API ability as the last paired one; a CMD_POINT /
CMD_UNIT row with the engine command at its new target (the REPEAT that follows it then
finds its command used). Within a loop both sides are in issue order and each engine
command is used once, the way scripts/build_ability_table.py matches for the ability table.
"""

import collections

import numpy as np

from . import replay_actions as ra

REPORTED = (ra.CMD, ra.CMD_POINT, ra.CMD_UNIT, ra.REPEAT)


def same_target(acts, i, target, tx, ty, target_tag):
    t = acts["target"][i]
    if t == ra.TARGET_POINT:
        return target == 1 and abs(tx - acts["tx"][i]) < 0.5 and abs(ty - acts["ty"][i]) < 0.5
    if t == ra.TARGET_UNIT:
        return target == 2 and target_tag == int(acts["target_tag"][i])
    if t == ra.TARGET_NONE:
        return target == 0
    return True  # snapshot (a fogged thing): no tag to compare


def match(acts, slot, pov):
    """pov: a player-view state file's arrays (cmd_*, unit_tag). Returns (cmd_event[C]: the
    action-table row each engine command executes, -1 if none; stats)."""
    loops, abil, target = pov["cmd_loop"], pov["cmd_abil"], pov["cmd_target"]
    tx, ty, tid, tags = pov["cmd_tx"], pov["cmd_ty"], pov["cmd_target_id"], pov["unit_tag"]
    by_loop = collections.defaultdict(list)
    for c in range(len(loops)):
        by_loop[int(loops[c])].append(c)
    used = np.zeros(len(loops), bool)
    cmd_event = np.full(len(loops), -1, np.int64)
    rows = np.where((acts["player"] == slot) & np.isin(acts["kind"], REPORTED))[0]
    st = collections.Counter(engine=len(loops), rows=len(rows), cmd_rows=0, cmd_rows_matched=0)
    last_api = None
    for i in rows:
        kind = acts["kind"][i]
        is_cmd = kind == ra.CMD and acts["abil_link"][i] >= 0
        st["cmd_rows"] += is_cmd
        hit = None
        for c in by_loop.get(int(acts["loop"][i]), ()):
            if used[c]:
                continue
            if kind == ra.REPEAT:
                ok = last_api is not None and abil[c] == last_api
            else:
                ttag = int(tags[tid[c]]) - (1 << 32) if target[c] == 2 else 0
                ok = same_target(acts, i, int(target[c]), float(tx[c]), float(ty[c]), ttag)
            if ok:
                hit = c
                break
        if hit is None:
            continue
        used[hit] = True
        cmd_event[hit] = i
        if kind == ra.CMD:
            last_api = int(abil[hit])
        st["cmd_rows_matched"] += is_cmd
    st["engine_matched"] = int(used.sum())
    return cmd_event, dict(st)
