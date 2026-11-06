"""A player's control groups and active selection, replayed from the replay's own events.

The API does not expose control-group contents (ui_data.groups is index + leader type +
count per group; the selection panel lists types without tags), but the replay logs every
change to them: SSelectionDeltaEvent and SControlGroupUpdateEvent, parsed by
replay_actions.py into SELECT / GROUP rows. This module replays those rows into eleven lists
(groups 0-9 and the active selection, 10) and snapshots them per state frame. The semantics
below were checked against the engine's own summaries and recipients (worldmodel/README.md,
"Command recipients and control groups"); sc2reader's SelectionTracker is the reference for
the event structure but sorts by tag as an approximation and ignores deaths.

  - Each list is in ascending unit-tag order, the game's own order: removal masks index it,
    the engine's recipients come in it, and the selection panel lists units in it. (Subgroup
    priorities and links in the deltas only describe the units; they do not order the list.)
  - SELECT on list g: drop the masked positions, add the new units, keep the order. The game
    logs these for the player's selection changes and, on any list, when a member changes
    type (morphs, burrow, an egg hatching: the egg's tag is removed and the hatched unit's
    added) or when it cleans dead members out of a selection the player just changed.
  - GROUP ops: set (0) / set+steal (4): group := selection; add (1) / add+steal (5):
    group := (group minus mask) + selection; recall (2): selection := group minus mask;
    clear (3): group := []. The "steal" variants need nothing more: the game logs the
    removal from every other group itself, as a burst of recall / mask / set rows on each
    affected group followed by the selection's restoration, all in the same loop.
  - Dead units are NOT removed from a list when they die: they stay as stale entries that
    the masks of later recalls and deltas account for (a baneling that exploded at loop 5076
    was still position 1 of its group's recall mask at 5233). Pruning at the death loop
    shifts every later mask. The engine's group counts and recipients exclude them, so the
    per-frame output does too (a member whose death loop is <= the frame loop is left out).

A frame at loop L shows the lists after every event at loops < L (an action at loop t is
reported by the engine in the first observation after t); the per-command snapshots are
taken right before the command's row.
"""

import collections

import numpy as np

from . import replay_actions as ra

N_LISTS = 11  # control groups 0-9, the active selection is 10
SELECTION = 10
ENGINE_TAG_OFFSET = 1 << 32


def ragged(acts, start_key, flat_key, i):
    """Row i's slice of a ragged action-table column. The *_start arrays hold one start per
    row and no trailing sentinel, so the last row's slice ends at the flat array's end (indexing
    start[i + 1] there raised IndexError on views whose frames reach the replay's last row).
    """
    start = acts[start_key]
    end = start[i + 1] if i + 1 < len(start) else len(acts[flat_key])
    return acts[flat_key][start[i] : end].tolist()


class Tracker:
    def __init__(self):
        self.lists = [[] for _ in range(N_LISTS)]  # replay tags, ascending
        self.errors = collections.Counter()
        self.error_rows = (
            []
        )  # rows of the action table whose mask reached past the end of the list

    def remove(self, lst, kind, positions, row=-1):
        n = len(lst)
        if kind == ra.REMOVE_NONE:
            return list(lst)
        if any(p >= n for p in positions):
            self.errors["mask past the end"] += 1
            self.error_rows.append(row)
        if kind == ra.REMOVE_ZERO:  # positions to keep
            return [lst[p] for p in positions if p < n]
        drop = set(positions)  # MASK / ONE: positions to remove
        return [u for i, u in enumerate(lst) if i not in drop]

    @staticmethod
    def merge(lst, tags):
        return sorted(set(lst) | set(tags))

    def apply(self, acts, i):
        """Applies row i of the action table (a SELECT or GROUP row)."""
        g = int(acts["group"][i])
        kind = int(acts["select_remove"][i])
        positions = ragged(acts, "rem_start", "rem_pos", i)
        if acts["kind"][i] == ra.SELECT:
            added = ragged(acts, "add_start", "add_tag", i)
            self.lists[g] = self.merge(
                self.remove(self.lists[g], kind, positions, i), added
            )
            return
        op = int(acts["group_op"][i])
        sel = self.lists[SELECTION]
        if op in (0, 4):
            self.lists[g] = list(sel)
        elif op in (1, 5):
            self.lists[g] = self.merge(
                self.remove(self.lists[g], kind, positions, i), sel
            )
        elif op == 2:
            self.lists[SELECTION] = self.remove(self.lists[g], kind, positions, i)
        elif op == 3:
            self.lists[g] = []
        else:
            self.errors[f"group op {op}"] += 1

    def snapshot(self):
        return [tuple(lst) for lst in self.lists]


def player_rows(acts, slot):
    return np.where(acts["player"] == slot)[0]


def replay(acts, slot, frame_loops=(), command_rows=(), after_rows=None):
    """Runs the tracker over one player's rows. Returns (per-frame snapshots: 11 tuples of
    replay tags per frame loop, including stale dead members; snapshots of the selection
    right before each row in command_rows: {row: tuple}; the tracker). after_rows, a dict,
    receives the selection after each SELECT / GROUP row: {row: tuple}."""
    tr = Tracker()
    rows = player_rows(acts, slot)
    loops = acts["loop"][rows]
    wanted = set(int(r) for r in command_rows)
    frames, at_rows = [], {}
    fi, n = 0, len(frame_loops)
    for r, loop in zip(rows, loops):
        while fi < n and frame_loops[fi] <= loop:  # frame at L: events at loops < L
            frames.append(tr.snapshot())
            fi += 1
        r = int(r)
        if r in wanted:
            at_rows[r] = tr.snapshot()[SELECTION]
        if acts["kind"][r] in (ra.SELECT, ra.GROUP):
            tr.apply(acts, r)
            if after_rows is not None:
                after_rows[r] = tuple(tr.lists[SELECTION])
    while fi < n:
        frames.append(tr.snapshot())
        fi += 1
    return frames, at_rows, tr


def deaths_by_replay_tag(died_engine):
    """Engine tag -> loop (extract_states.tracker_lifecycle) as replay tag -> loop."""
    return {int(t) - ENGINE_TAG_OFFSET: int(l) for t, l in died_engine.items()}


def deaths_from_frames(arrays):
    """Replay tag -> loop of the frame that reported the death (dead_id lists of a state
    file; at most a frame late against the replay's own death loops)."""
    unit_tag, frame_loop, ds = (
        arrays["unit_tag"],
        arrays["frame_loop"],
        arrays["dead_frame_start"],
    )
    died = {}
    for f in range(len(frame_loop)):
        for i in arrays["dead_id"][ds[f] : ds[f + 1]]:
            died.setdefault(int(unit_tag[i]) - ENGINE_TAG_OFFSET, int(frame_loop[f]))
    return died


def frame_groups(acts, slots, frame_loop, unit_tag, died):
    """The grp_* arrays of a state file, for the players in slots (a player view: its one slot;
    an observer file: every player): group g of player index k (grp_player[k] is its slot) at
    frame f is grp_id[grp_start[(f*P+k)*11+g]:grp_start[(f*P+k)*11+g+1]], P = len(slots),
    state unit ids in the game's order (ascending tag). died: replay tag -> death loop; dead
    members and members the engine never reported (no state id) are left out."""
    ids = {int(t) - ENGINE_TAG_OFFSET: i for i, t in enumerate(unit_tag.tolist())}
    per_player = [replay(acts, slot, frame_loop)[0] for slot in slots]
    start, flat = [0], []
    for f in range(len(frame_loop)):
        loop = int(frame_loop[f])
        for frames in per_player:
            for lst in frames[f]:
                flat.extend(
                    ids[t] for t in lst if t in ids and died.get(t, loop + 1) > loop
                )
                start.append(len(flat))
    return dict(
        grp_player=np.asarray(list(slots), np.int8),
        grp_start=np.asarray(start, np.int64),
        grp_id=np.asarray(flat, np.int32),
    )


N_CAMERA_SAVES = 8  # camera hotkey slots (F5-F8 and the shifted set; slots 0-7 all seen in pro replays)
FLAG_SET_AUTOCAST, FLAG_AUTOCAST_ON = (
    0x40,
    0x80,
)  # SCmdEvent m_cmdFlags (3 toggles in 25 pro replays, all 0xC0: on)


def frame_memory(acts, slots, frame_loop, unit_tag, died):
    """The rest of what a player's interface remembers, per frame, from the replay's events
    (the same frame rule as frame_groups: a frame at loop L shows the state after every event
    at loops < L). For the players in slots (grp_player order), F frames, P players:

        cam_save[F, P, 8, 2]   camera hotkey positions (slot 0-7), NaN until saved
        subgroup[F, P]         the active subgroup (tab) of the selection: the last SELECT row's
                               index; a control group recall resets it to 0 (assumed: the game
                               logs no index for recalls)
        lc_*[F, P]             the player's last command, what a REPEAT re-fires and a CMD_POINT /
                               CMD_UNIT retargets: lc_link, lc_cmd (the replay's ability
                               numbering), lc_flags, lc_target (replay_actions.TARGET_*), lc_x,
                               lc_y, lc_unit (state id, -1 none), lc_loop (-1: no command yet)
        ac_*                   autocast toggles, an event list: ac_loop, ac_player (slot),
                               ac_link, ac_cmd, ac_on, and the units it applied to (the tracked
                               alive selection at that row) ac_unit[ac_start[i]:ac_start[i+1]].
                               The observation never shows autocast state; the tokenizer
                               accumulates these per unit (rare: a few per game)
    """
    ids = {int(t) - ENGINE_TAG_OFFSET: i for i, t in enumerate(unit_tag.tolist())}
    F, P = len(frame_loop), len(slots)
    cam = np.full((F, P, N_CAMERA_SAVES, 2), np.nan, np.float32)
    sub = np.zeros((F, P), np.int16)
    lc = dict(
        link=np.full((F, P), -1, np.int16),
        cmd=np.full((F, P), -1, np.int16),
        flags=np.zeros((F, P), np.int64),
        target=np.zeros((F, P), np.int8),
        x=np.full((F, P), np.nan, np.float32),
        y=np.full((F, P), np.nan, np.float32),
        unit=np.full((F, P), -1, np.int32),
        loop=np.full((F, P), -1, np.int32),
    )
    ac = dict(loop=[], player=[], link=[], cmd=[], on=[], start=[0], unit=[])
    for k, slot in enumerate(slots):
        rows = player_rows(acts, slot)
        toggles = [
            int(r)
            for r in rows
            if acts["kind"][r] == ra.CMD and int(acts["flags"][r]) & FLAG_SET_AUTOCAST
        ]
        _, at_rows, _ = replay(acts, slot, (), toggles)
        cams = np.full((N_CAMERA_SAVES, 2), np.nan, np.float32)
        subgroup = 0
        last = dict(
            link=-1, cmd=-1, flags=0, target=0, x=np.nan, y=np.nan, unit=-1, loop=-1
        )
        fi = 0
        for r in rows:
            r, loop = int(r), int(acts["loop"][r])
            while fi < F and frame_loop[fi] <= loop:
                cam[fi, k], sub[fi, k] = cams, subgroup
                for key, v in last.items():
                    lc[key][fi, k] = v
                fi += 1
            kind = acts["kind"][r]
            if kind == ra.CAMERA_SAVE:
                g = int(acts["group"][r])
                if 0 <= g < N_CAMERA_SAVES:
                    cams[g] = (acts["cam_x"][r], acts["cam_y"][r])
            elif kind == ra.SELECT:
                subgroup = int(acts["subgroup"][r])
            elif kind == ra.GROUP and int(acts["group_op"][r]) == 2:
                subgroup = 0
            elif kind == ra.CMD:
                tag = int(acts["target_tag"][r])
                last = dict(
                    link=int(acts["abil_link"][r]),
                    cmd=int(acts["abil_cmd"][r]),
                    flags=int(acts["flags"][r]),
                    target=int(acts["target"][r]),
                    x=float(acts["tx"][r]),
                    y=float(acts["ty"][r]),
                    unit=ids.get(tag, -1) if tag else -1,
                    loop=loop,
                )
                if r in at_rows:
                    ac["loop"].append(loop)
                    ac["player"].append(slot)
                    ac["link"].append(last["link"])
                    ac["cmd"].append(last["cmd"])
                    ac["on"].append(bool(int(acts["flags"][r]) & FLAG_AUTOCAST_ON))
                    ac["unit"].extend(
                        ids[t]
                        for t in at_rows[r]
                        if t in ids and died.get(t, loop + 1) >= loop
                    )
                    ac["start"].append(len(ac["unit"]))
            elif kind in (ra.CMD_POINT, ra.CMD_UNIT):
                tag = int(acts["target_tag"][r])
                last = dict(
                    last,
                    target=int(acts["target"][r]),
                    x=float(acts["tx"][r]),
                    y=float(acts["ty"][r]),
                    unit=ids.get(tag, -1) if tag else -1,
                )
        while fi < F:
            cam[fi, k], sub[fi, k] = cams, subgroup
            for key, v in last.items():
                lc[key][fi, k] = v
            fi += 1
    out = dict(cam_save=cam, subgroup=sub)
    out.update({f"lc_{key}": v for key, v in lc.items()})
    out.update(
        ac_loop=np.asarray(ac["loop"], np.int32),
        ac_player=np.asarray(ac["player"], np.int8),
        ac_link=np.asarray(ac["link"], np.int16),
        ac_cmd=np.asarray(ac["cmd"], np.int16),
        ac_on=np.asarray(ac["on"], np.bool_),
        ac_start=np.asarray(ac["start"], np.int64),
        ac_unit=np.asarray(ac["unit"], np.int32),
    )
    return out


def selected_at(arrays, loop):
    """The engine's is_selected set (replay tags) as of `loop`, from a player view's per-loop
    snapshots (sel_loop / sel_start / sel_id: stored at every loop the set changed). None when
    the file has no snapshots or `loop` precedes the first one."""
    if "sel_loop" not in arrays:
        return None
    sl = arrays["sel_loop"]
    k = int(np.searchsorted(sl, loop, side="right")) - 1
    if k < 0:
        return None
    ss, unit_tag = arrays["sel_start"], arrays["unit_tag"]
    return {
        int(unit_tag[i]) - ENGINE_TAG_OFFSET
        for i in arrays["sel_id"][ss[k] : ss[k + 1]]
    }


def validate(arrays, acts, slot, died=None, verbose=False):
    """Checks the tracker against the engine on a player-view file with cmd_* arrays (and, from
    an extraction with --ui, the ui_* arrays; without them only the command checks run).
    Counts (rates are fractions of the totals):

        cells, grp_count   (frame, group 0-9) cells; those where the tracked alive count
                           equals ui_grp_count
        grp_nonempty, grp_leader
                           cells with units on either side; those with equal count and the first
                           tracked unit's type equal to ui_grp_leader (the remaining ones are
                           mixed groups with a unit mid-morph, where the game shows the other
                           type; ordering by the deltas' subgroup priorities did worse)
        frames, sel        frames; those where the tracked alive selection's unit types
                           equal the selection panel's (the panel is the engine's; it keeps
                           showing a unit that died while selected until the next change)
        cmd, cmd_matched   engine commands; those joined to a replay row (command_join)
        cmd_exact          ... whose recipients are exactly the tracked alive selection at that
                           row; cmd_subset: a proper subset of it (one larva of the selected
                           larvae trains, one zergling morphs; a morph that only part of a mixed
                           selection can do); cmd_order: exact and in the same order
        cmd_none           ... with no recipient in the tracked selection at all
        tracker_errors     masks reaching past the end of a list (an inconsistent replay)

    With the engine's per-loop is_selected snapshots (sel_*; player views), also:

        ev, ev_eq          loops with a SELECT / GROUP row of the player; those where the tracked
                           alive selection after the loop's rows equals the engine's selection
                           at the next loop (the ones that differ are returned in "mismatches":
                           loop, rows, tracked, engine, up to 20)
        ev_group_eq        ... of the loops with a GROUP row (set / add / recall / steal)
        cmd_sel_eq, cmd_sel_subset
                           engine commands whose recipients equal / are a proper subset of the
                           engine's own selection at the loop after the command
    """
    from . import command_join

    frame_loop, unit_tag = arrays["frame_loop"], arrays["unit_tag"]
    F = len(frame_loop)
    if died is None:
        died = deaths_from_frames(arrays)
    cmd_event, join = command_join.match(acts, slot, arrays)
    cmd_rows = [int(r) for r in cmd_event if r >= 0]
    after = {}
    frames, at_rows, tr = replay(acts, slot, frame_loop, cmd_rows, after)
    ids = {int(t) - ENGINE_TAG_OFFSET: i for i, t in enumerate(unit_tag.tolist())}
    fs, u_id, u_type = arrays["frame_start"], arrays["u_id"], arrays["u_type"]
    out = collections.Counter(
        cells=F * 10,
        grp_count=0,
        grp_nonempty=0,
        grp_leader=0,
        frames=F,
        sel=0,
        cmd=len(cmd_event),
        cmd_matched=len(cmd_rows),
        cmd_exact=0,
        cmd_subset=0,
        cmd_order=0,
        cmd_none=0,
    )
    out.update({f"join_{k}": v for k, v in join.items()})
    shown = collections.Counter()
    has_ui = "ui_grp_count" in arrays
    if has_ui:
        ui_count, ui_leader = arrays["ui_grp_count"], arrays["ui_grp_leader"]
        ui_type, ui_start = arrays["ui_sel_type"], arrays["ui_sel_start"]
    else:
        out["cells"] = out["frames"] = 0
    for f in range(F if has_ui else 0):
        loop = int(frame_loop[f])
        types = dict(
            zip(u_id[fs[f] : fs[f + 1]].tolist(), u_type[fs[f] : fs[f + 1]].tolist())
        )
        snap = [[t for t in lst if died.get(t, loop + 1) > loop] for lst in frames[f]]

        def type_of(tag):
            return types.get(ids.get(tag, -1), -1)

        for g in range(10):
            n = len(snap[g])
            if n == ui_count[f, g]:
                out["grp_count"] += 1
            if n or ui_count[f, g]:
                out["grp_nonempty"] += 1
                if n == ui_count[f, g] and n and type_of(snap[g][0]) == ui_leader[f, g]:
                    out["grp_leader"] += 1
                elif verbose and shown["grp"] < 10 and n != ui_count[f, g]:
                    shown["grp"] += 1
                    print(
                        f"  group {g} frame {f} loop {loop}: tracked {n} ui {int(ui_count[f, g])}"
                    )
        sel = sorted(type_of(t) for t in snap[SELECTION])
        panel = sorted(ui_type[ui_start[f] : ui_start[f + 1]].tolist())
        if sel == panel:
            out["sel"] += 1
        elif verbose and shown["sel"] < 10:
            shown["sel"] += 1
            print(f"  selection frame {f} loop {loop}: tracked {sel} ui {panel}")
    cs, cu = arrays["cmd_start"], arrays["cmd_unit"]
    rtag = {i: int(t) - ENGINE_TAG_OFFSET for i, t in enumerate(unit_tag.tolist())}
    for c, r in enumerate(cmd_event):
        if r < 0:
            continue
        loop = int(arrays["cmd_loop"][c])
        recip = [rtag[int(i)] for i in cu[cs[c] : cs[c + 1]]]
        sel = [
            t for t in at_rows[int(r)] if died.get(t, loop + 1) >= loop
        ]  # a morph is logged as a death at its loop
        if set(recip) == set(sel):
            out["cmd_exact"] += 1
            out["cmd_order"] += tuple(recip) == tuple(sel)
        elif set(recip) <= set(sel):
            out["cmd_subset"] += 1
        else:
            out["cmd_none"] += not (set(recip) & set(sel))
            if verbose and shown["cmd"] < 10:
                shown["cmd"] += 1
                print(
                    f"  command {c} loop {loop} ability {int(arrays['cmd_abil'][c])} row {int(r)}: recipients {recip} tracked {sel}"
                )
    # the engine's own selection, per loop (player views extracted with the snapshots)
    mismatches = []
    if "sel_loop" in arrays:
        rows = [
            int(r)
            for r in player_rows(acts, slot)
            if acts["kind"][r] in (ra.SELECT, ra.GROUP)
        ]
        by_loop = collections.OrderedDict()
        for r in rows:
            by_loop.setdefault(int(acts["loop"][r]), []).append(r)
        last_snap = int(arrays["sel_loop"][-1]) if len(arrays["sel_loop"]) else -1
        for loop, rs in by_loop.items():
            engine = selected_at(arrays, loop + 1)
            if (
                engine is None
                or loop + 1 > last_snap + 1
                or loop + 1 > int(frame_loop[-1])
            ):
                continue
            tracked = {t for t in after[rs[-1]] if died.get(t, loop + 2) > loop + 1}
            out["ev"] += 1
            has_group = any(acts["kind"][r] == ra.GROUP for r in rs)
            out["ev_group"] += has_group
            if tracked == engine:
                out["ev_eq"] += 1
                out["ev_group_eq"] += has_group
            elif len(mismatches) < 20:
                desc = []
                for r in rs:
                    kind = ra.KINDS[acts["kind"][r]]
                    mask = ragged(acts, "rem_start", "rem_pos", r)
                    added = ragged(acts, "add_start", "add_tag", r)
                    desc.append(
                        f"{kind} g{int(acts['group'][r])}"
                        + (f" op{int(acts['group_op'][r])}" if kind == "GROUP" else "")
                        + f" rm{int(acts['select_remove'][r])}{mask}"
                        + (f" add{added}" if added else "")
                    )
                mismatches.append(
                    dict(
                        loop=loop,
                        rows=desc,
                        tracked_only=sorted(tracked - engine),
                        engine_only=sorted(engine - tracked),
                    )
                )
        for c in range(len(cmd_event)):
            loop = int(arrays["cmd_loop"][c])
            engine = selected_at(arrays, loop + 1)
            if engine is None:
                continue
            recip = set(rtag[int(i)] for i in cu[cs[c] : cs[c + 1]])
            out["cmd_sel"] += 1
            out["cmd_sel_eq"] += recip == engine
            out["cmd_sel_subset"] += recip < engine
    out["tracker_errors"] = sum(tr.errors.values())
    out = dict(out)
    if verbose:
        for m in mismatches:
            print("  event mismatch", m)
    out["mismatches"] = mismatches
    return out
