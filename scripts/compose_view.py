"""Composes a player's view from their teammate's view and the observer view.

The engine's player view for slot 4 crashes at the first research order of every game (all
builds; worldmodel/README.md). Teammates share vision, and comparing two teammates' engine
views showed them identical except for:

    - each other's units: a few private fields hidden (ALLY_HIDDEN below)
    - snapshots (remembered structures in fog): the same content, different tags per viewer

and a player's own units are identical to the observer's rows for them. So a player's view =
their teammate's view, with this player's own units replaced by the observer's rows and the
teammate's units shown as an ally sees them.

Not composed: the teammate's commands with recipients (cmd_*), selection snapshots (sel_*),
tracked groups (grp_*) and ui summaries (ui_*) are theirs, so they are left out, and the
selected bit of u_flags is cleared (an ally does not see the selection). This player's groups
are in the observer file (grp_* there covers every player).

scripts/validate_compose.py checks this against real engine views (a full teammate view, and
slot 4 up to the crash).

    python scripts/compose_view.py --mate pov/<name>_p3.npz --observer obs/<name>.npz --slot 4 --out pov/<name>_p4.npz
"""

import argparse
import json

import numpy as np

# what an ally's view shows of a unit that isn't theirs, relative to the owner's own view
# (measured: tests in worldmodel/README.md). value None = drop the ragged entries.
ALLY_HIDDEN = dict(harvesters=0, weapon_cooldown=0.0, engaged=0, add_on=0, cargo_used=0, prod=None, pass_=None, ord=None)
# engaged / add_on: no unit (tag 0); prod / pass_ / ord: production queue, passengers and orders emptied

U_FIELDS = ("type", "owner", "x", "y", "z", "facing", "radius", "hp", "hp_max", "shield", "shield_max", "energy",
            "build", "flags", "contents", "harvesters", "weapon_cooldown", "cargo_used", "display", "cloak")
ID_FIELDS = ("add_on", "engaged")  # unit ids (per file) -> remapped through tags
RAGGED = dict(prod=("u_prod_start", ("prod_abil", "prod_progress")), buff=("u_buff_start", ("b_id",)),
              pass_=("u_pass_start", ("pass_id", "pass_type", "pass_hp")),
              ord=("u_ord_start", ("ord_abil", "ord_tx", "ord_ty", "ord_target", "ord_progress")))
RAG_IDS = ("pass_id", "ord_target")  # ragged columns that are unit ids (per file) -> remapped through tags


def _ragged_take(starts, counts):
    """Concatenated indices of runs [starts[i], starts[i] + counts[i])."""
    if not len(counts) or counts.sum() == 0:
        return np.zeros(0, np.int64)
    ends = np.cumsum(counts)
    offs = np.repeat(starts - (ends - counts), counts)
    return np.arange(ends[-1]) + offs


def _counts(z, start_key, first_ragged):
    return np.diff(np.append(z[start_key], len(z[first_ragged])))


def filled(z, fld, rows):
    """display / cloak of a player's own units, for observer files extracted without them: own
    units are always visible (display 1); cloak is 3 (not cloaked) or 4 (cloaked, allied)."""
    if fld == "display":
        return np.ones(len(rows), np.int8)
    if fld == "cloak":
        return np.where((z["u_flags"][rows] >> 2) & 1, 4, 3).astype(np.int8)
    raise KeyError(f"u_{fld}")


def compose(mate, obs, slot, mate_slot, hidden=ALLY_HIDDEN):
    """mate, obs: dicts of arrays (np.load of a player view / the observer view). Returns the
    arrays of a player-view file for `slot`."""
    if not np.array_equal(mate["frame_loop"], obs["frame_loop"][:len(mate["frame_loop"])]):
        raise ValueError("frame loops differ between the teammate view and the observer view")
    srcs = dict(m=mate, o=obs)
    ragged = {k: v for k, v in RAGGED.items() if v[0] in mate}  # older files have no orders
    counts = {s: {k: _counts(z, st, cols[0]) for k, (st, cols) in ragged.items()} for s, z in srcs.items()}
    tag_of = {s: z["unit_tag"] for s, z in srcs.items()}
    out = {f: [] for f in U_FIELDS + ID_FIELDS}
    out_tag, out_starts = [], {k: [] for k in ragged}
    rag = {c: [] for _, cols in ragged.values() for c in cols}
    rag_len = {k: 0 for k in ragged}
    frame_start = [0]
    for f in range(len(mate["frame_loop"])):
        ma, mb = mate["frame_start"][f], mate["frame_start"][f + 1]
        oa, ob = obs["frame_start"][f], obs["frame_start"][f + 1]
        m_rows = np.arange(ma, mb)
        m_rows = m_rows[mate["u_owner"][ma:mb] != slot]          # the teammate's view, minus this player's units
        o_rows = np.arange(oa, ob)
        o_rows = o_rows[obs["u_owner"][oa:ob] == slot]            # this player's units: the observer's rows
        for s, rows in (("m", m_rows), ("o", o_rows)):
            z = srcs[s]
            mate_owned = (z["u_owner"][rows] == mate_slot) if s == "m" else np.zeros(len(rows), bool)
            for fld in U_FIELDS:
                if f"u_{fld}" not in z:  # observer files from before display/cloak were extracted
                    v = filled(z, fld, rows)
                    out[fld].append(v)
                    continue
                v = z[f"u_{fld}"][rows].copy()
                if fld in hidden and hidden[fld] is not None:
                    v[mate_owned] = hidden[fld]
                out[fld].append(v)
            for fld in ID_FIELDS:
                ids = z[f"u_{fld}"][rows]
                t = np.where(ids >= 0, tag_of[s][np.clip(ids, 0, None)], 0).astype(np.uint64)
                if fld in hidden and hidden[fld] is not None:
                    t[mate_owned] = hidden[fld]
                out[fld].append(t)
            out_tag.append(tag_of[s][z["u_id"][rows]])
            for k, (st, cols) in ragged.items():
                c = counts[s][k][rows].copy()
                if k in hidden and hidden[k] is None:
                    c[mate_owned] = 0
                idx = _ragged_take(z[st][rows], c)
                out_starts[k].append(rag_len[k] + np.concatenate([[0], np.cumsum(c)[:-1]]) if len(c) else np.zeros(0, np.int64))
                rag_len[k] += int(c.sum())
                for col in cols:
                    v = z[col][idx]
                    if col in RAG_IDS:  # unit ids -> tags (0 = none)
                        v = np.where(v >= 0, tag_of[s][np.clip(v, 0, None)], 0).astype(np.uint64)
                    rag[col].append(v)
        frame_start.append(frame_start[-1] + len(m_rows) + len(o_rows))
    tags = np.concatenate(out_tag).astype(np.uint64)
    refs = np.concatenate([np.concatenate(out[f]) for f in ID_FIELDS]
                          + [np.asarray(x, np.uint64) for c in RAG_IDS if c in rag for x in rag[c]]).astype(np.uint64)
    uniq = np.unique(np.concatenate([tags, refs[refs > 0]]))  # referenced units can be out of view that frame
    uid = np.searchsorted(uniq, tags)
    lookup = lambda t: np.where(t > 0, np.searchsorted(uniq, t), -1)

    arrays = {f"u_{f}": np.concatenate(out[f]).astype(mate[f"u_{f}"].dtype) for f in U_FIELDS}
    for f in ID_FIELDS:  # referenced units that aren't in this view get -1
        t = np.concatenate(out[f])
        i = np.clip(np.searchsorted(uniq, t), 0, len(uniq) - 1)
        arrays[f"u_{f}"] = np.where((t > 0) & (uniq[i] == t), i, -1).astype(mate[f"u_{f}"].dtype)
    arrays["u_id"] = uid.astype(mate["u_id"].dtype)
    for k, (st, cols) in ragged.items():
        arrays[st] = np.concatenate(out_starts[k]).astype(np.int64)
        for col in cols:
            v = np.concatenate(rag[col]) if rag[col] else np.zeros(0, mate[col].dtype)
            if col in RAG_IDS:
                i = np.clip(np.searchsorted(uniq, v), 0, max(len(uniq) - 1, 0))
                v = np.where(len(uniq) and (uniq[i] == v) & (v > 0), i, -1)
            arrays[col] = v.astype(mate[col].dtype)
    arrays["frame_start"] = np.asarray(frame_start, np.int64)
    arrays["unit_tag"] = uniq
    for k, v in mate.items():  # everything else is the team's: frames, effects, maps, visibility, stats
        if k not in arrays and not k.startswith(("u_", "cmd_", "sel_", "grp_", "ui_")) and k not in ("meta",):
            arrays[k] = v
    arrays["u_flags"] = (arrays["u_flags"] & ~np.uint8(64)).astype(mate["u_flags"].dtype)  # F_SELECTED is the mate's
    meta = json.loads(str(mate["meta"]))
    meta.update(perspective=slot, composed_from=dict(mate=mate_slot, rule=sorted(hidden)))
    arrays["meta"] = np.asarray(json.dumps(meta))
    # dead units: the teammate's list (ids remapped through tags)
    d = mate["dead_id"]
    dt = mate["unit_tag"][d] if len(d) else np.zeros(0, np.uint64)
    arrays["dead_id"] = lookup(dt).astype(mate["dead_id"].dtype)
    return arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mate", required=True, help="the teammate's player-view file")
    ap.add_argument("--observer", required=True)
    ap.add_argument("--slot", type=int, required=True, help="slot to compose")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    mate = dict(np.load(args.mate))
    mate_slot = json.loads(str(mate["meta"]))["perspective"]
    arrays = compose(mate, dict(np.load(args.observer)), args.slot, mate_slot)
    tmp = args.out + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    import os
    os.replace(tmp, args.out)
    print(f"composed slot {args.slot} from slot {mate_slot}: {len(arrays['frame_loop'])} frames, {len(arrays['u_id'])} rows -> {args.out}")


if __name__ == "__main__":
    main()
