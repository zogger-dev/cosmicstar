"""Replay actions -> action tokens, aligned with the state frames.

One token per thing a player did (raw events: replay_actions.py):

    CMD        a new command: ability (family + button; right-click = Smart), flags, target
    REISSUE    the player's last command executed again:
                 - a retarget: CMD_POINT / CMD_UNIT changes the last command's target
                   and the REPEAT that comes with it fires it (one token, the REPEAT is
                   folded in)
                 - a plain REPEAT: same command, same target
    SELECT     selection change: the added units (pointers), how many, how units were removed
    GROUP      control group operation + number
    CAMERA, CAMERA_SAVE, PING, TRADE, REQUEST, FULFILL, CANCEL, LEAVE

A held or mashed hotkey (a CMD / REISSUE followed by plain REPEATs within BURST loops,
with nothing else from that player in between) stays one token with a press count:
"Drone x7". Each press is a separate event on disk; the folding happens only here, so
the raw stream stays lossless for a future control decoder.

Units an action refers to (command targets, pinged units, the units added to a
selection) are pointers: the unit's own state features at the latest frame at or before
the action (GameFrames.entity_features), embedded by the state encoder like any entity.
Fogged snapshots carry no tag; they get their unit type and position instead.

Abilities are named per ruleset from the Editor exports
(data/worldmodel/ability_links.json): (family, button), e.g. ("BarracksTrain", 0) =
Marine. The family and button vocabularies are shared across builds, so link numbers
that move between patches don't matter.
"""

import json
import os

import numpy as np

from . import replay_actions as ra
from ..core.paths import ABILITY_LINKS_PATH as LINKS
from ..core.paths import RULES_DIR as RULES

ENGINE_TAG_OFFSET = 1 << 32

(
    A_PAD,
    A_CMD,
    A_REISSUE,
    A_SELECT,
    A_GROUP,
    A_CAMERA,
    A_CAMERA_SAVE,
    A_PING,
    A_TRADE,
    A_REQUEST,
    A_FULFILL,
    A_CANCEL,
    A_LEAVE,
) = range(13)
N_ACT_KINDS = 13
FAM_PAD, FAM_UNK, FAM_SMART, FAM_FIRST = 0, 1, 2, 3
N_BUTTONS = 32  # button index + 1 (0 = no ability)
TK_NONE, TK_POINT, TK_UNIT, TK_SNAPSHOT = (
    ra.TARGET_NONE,
    ra.TARGET_POINT,
    ra.TARGET_UNIT,
    ra.TARGET_SNAPSHOT,
)
N_GROUPS, N_GROUP_OPS, N_REMOVE, N_SUBGROUPS = 12, 7, 5, 33  # +1 each: 0 = n/a
BURST = 6  # loops: plain REPEATs this close to the last press are the same burst
MAX_SELECTED = 32  # pointers kept per SELECT (the count keeps the full number)
C_ACTION = 12  # numeric features per token (see GameActions.window)

TOKEN_COLS = dict(
    loop=np.int32,
    player=np.int8,
    kind=np.int8,
    family=np.int32,
    button=np.int8,
    queued=np.bool_,
    smart=np.bool_,
    minimap=np.bool_,
    presses=np.int16,
    tkind=np.int8,
    x=np.float32,
    y=np.float32,
    uid=np.int32,
    snap=np.int64,
    snap_raw=np.int64,
    group=np.int8,
    gop=np.int8,
    remove=np.int8,
    subgroup=np.int8,
    n_add=np.int16,
    n_rem=np.int16,
    recipient=np.int8,
    minerals=np.int32,
    gas=np.int32,
    sel_start=np.int64,
)


class ActionVocab:
    """Ability families and buttons, plus each ruleset's link tables."""

    def __init__(self, links_path=LINKS, rules_dir=RULES):
        self.table = json.load(open(links_path))
        fams = sorted({n for t in self.table.values() for n in t["links"].values()})
        self.family = {n: FAM_FIRST + i for i, n in enumerate(fams)}
        self.family_names = ["<pad>", "<unk>", "<smart>"] + fams
        self.n_families = FAM_FIRST + len(fams)
        self.n_buttons = N_BUTTONS + 1
        self.rules_dir = rules_dir
        self._unit_ids = {}

    def ruleset_table(self, meta):
        """The link tables for a game: its own ruleset, or another ruleset of the same
        build."""
        t = self.table.get(meta.get("ruleset"))
        if t is None:
            t = next(
                (
                    v
                    for v in self.table.values()
                    if v["base_build"] == meta.get("base_build")
                ),
                None,
            )
        if t is None:
            raise KeyError(
                f"no ability table for {meta.get('ruleset')} (build"
                f" {meta.get('base_build')})"
            )
        return t

    def unit_ids(self, meta):
        """Unit name -> API unit type id, from the ruleset's rules dump."""
        key = meta.get("ruleset")
        if key not in self._unit_ids:
            path = os.path.join(self.rules_dir, f"{key}.json")
            units = json.load(open(path))["units"] if os.path.exists(path) else []
            self._unit_ids[key] = {u["name"]: u["unit_id"] for u in units}
        return self._unit_ids[key]


class GameActions:
    """One game's action tokens (all players, time order), built once from the raw
    events."""

    def __init__(self, path, frames, avocab):
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}  # NpzFile re-reads an array on every access
        self.fr = frames
        table = avocab.ruleset_table(frames.meta)
        links = {int(k): v for k, v in table["links"].items()}
        unit_links = {int(k): v for k, v in table["units"].items()}
        name_to_api = avocab.unit_ids(frames.meta)
        tag_to_uid = (
            {int(t): i for i, t in enumerate(frames.unit_tag)}
            if frames.unit_tag is not None
            else {}
        )

        def uid_of(replay_tag):
            return (
                tag_to_uid.get(int(replay_tag) + ENGINE_TAG_OFFSET, -1)
                if replay_tag
                else -1
            )

        def snap_type(link):
            """(vocab index, raw API id) of a fogged target's unit type."""
            api = name_to_api.get(unit_links.get(int(link), ""), 0)
            return (int(frames.vocab.unit_index([api])[0]) if api else 0), int(api or 0)

        cols = {k: [] for k in TOKEN_COLS}
        sel = []
        last_cmd, last_tok, last_raw, press_loop, awaiting = {}, {}, {}, {}, {}

        def add(**kw):
            for k, dt in TOKEN_COLS.items():
                default = (
                    np.nan
                    if k in ("x", "y")
                    else (
                        False
                        if dt is np.bool_
                        else (
                            1
                            if k == "presses"
                            else (
                                0
                                if k in ("family", "button", "snap", "snap_raw")
                                else -1
                            )
                        )
                    )
                )
                cols[k].append(kw.get(k, default))
            cols["sel_start"][-1] = len(sel)
            return len(cols["kind"]) - 1

        def target(i):
            t = int(d["target"][i])
            out = dict(tkind=t)
            if t in (TK_POINT, TK_UNIT, TK_SNAPSHOT):
                out.update(x=float(d["tx"][i]), y=float(d["ty"][i]))
            if t == TK_UNIT:
                out["uid"] = uid_of(d["target_tag"][i])
            if t == TK_SNAPSHOT:
                out["snap"], out["snap_raw"] = snap_type(d["snap_link"][i])
            return out

        kind, loop, player = d["kind"], d["loop"], d["player"]
        # the token each raw command row went into (a held hotkey's presses share one),
        # to join the engine's executed commands and their recipients to tokens
        tok_of_raw = np.full(len(kind), -1, np.int64)
        add_start = np.append(d["add_start"], len(d["add_tag"]))
        rem_start = np.append(d["rem_start"], len(d["rem_pos"]))
        for i in range(len(kind)):
            k, L, p = int(kind[i]), int(loop[i]), int(player[i])
            prev = last_raw.get(p)
            if k == ra.CMD:
                link, cmd, fl = (
                    int(d["abil_link"][i]),
                    int(d["abil_cmd"][i]),
                    int(d["flags"][i]),
                )
                ability = dict(
                    family=FAM_SMART if link < 0 else self_family(avocab, links, link),
                    button=0 if link < 0 else min(cmd, N_BUTTONS - 1) + 1,
                    queued=bool(fl & ra.FLAG_QUEUED),
                    smart=bool(fl & ra.FLAG_SMART),
                    minimap=bool(fl & ra.FLAG_MINIMAP),
                )
                last_cmd[p] = dict(ability, **target(i))
                last_tok[p] = add(loop=L, player=p, kind=A_CMD, **last_cmd[p])
                press_loop[p] = L
            elif k in (ra.CMD_POINT, ra.CMD_UNIT):
                base = last_cmd.get(p, dict(family=FAM_UNK, button=0))
                last_cmd[p] = dict(
                    {
                        kk: v
                        for kk, v in base.items()
                        if kk not in ("tkind", "x", "y", "uid", "snap")
                    },
                    **target(i),
                    minimap=bool(int(d["flags"][i]) & ra.FLAG_MINIMAP)
                    or base.get("minimap", False),
                )
                last_tok[p] = add(loop=L, player=p, kind=A_REISSUE, **last_cmd[p])
                press_loop[p], awaiting[p] = L, L
            elif k == ra.REPEAT:
                if awaiting.get(p) == L and prev in (ra.CMD_POINT, ra.CMD_UNIT):
                    awaiting[p] = (
                        None  # the REPEAT that fires a retarget: already one token
                    )
                elif (
                    p in last_tok
                    and prev in (ra.CMD, ra.REPEAT, ra.CMD_POINT, ra.CMD_UNIT)
                    and L - press_loop.get(p, -BURST - 1) <= BURST
                ):
                    cols["presses"][last_tok[p]] += 1  # held / mashed hotkey
                else:
                    last_tok[p] = add(
                        loop=L,
                        player=p,
                        kind=A_REISSUE,
                        **last_cmd.get(p, dict(family=FAM_UNK)),
                    )
                press_loop[p] = L
            if k in (ra.CMD, ra.CMD_POINT, ra.CMD_UNIT, ra.REPEAT):
                tok_of_raw[i] = last_tok.get(p, -1)
            if k == ra.SELECT:
                tags = d["add_tag"][add_start[i] : add_start[i + 1]]
                n_rem = rem_start[i + 1] - rem_start[i]
                add(
                    loop=L,
                    player=p,
                    kind=A_SELECT,
                    group=int(d["group"][i]) + 1,
                    remove=int(d["select_remove"][i]) + 1,
                    subgroup=min(int(d["subgroup"][i]), N_SUBGROUPS - 2) + 1,
                    n_add=len(tags),
                    n_rem=n_rem,
                )
                sel.extend(uid_of(t) for t in tags[:MAX_SELECTED])
            elif k == ra.GROUP:
                add(
                    loop=L,
                    player=p,
                    kind=A_GROUP,
                    group=int(d["group"][i]) + 1,
                    gop=int(d["group_op"][i]) + 1,
                )
            elif k == ra.CAMERA:
                add(
                    loop=L,
                    player=p,
                    kind=A_CAMERA,
                    x=float(d["cam_x"][i]),
                    y=float(d["cam_y"][i]),
                )
            elif k == ra.CAMERA_SAVE:
                add(
                    loop=L,
                    player=p,
                    kind=A_CAMERA_SAVE,
                    group=int(d["group"][i]) + 1,
                    x=float(d["cam_x"][i]),
                    y=float(d["cam_y"][i]),
                )
            elif k == ra.PING:
                t = target(i)
                t.update(
                    x=float(d["tx"][i]), y=float(d["ty"][i])
                )  # a ping always has a point
                if t["tkind"] == TK_UNIT and t.get("uid", -1) < 0:
                    t["snap"], t["snap_raw"] = snap_type(d["snap_link"][i])
                elif t["tkind"] != TK_UNIT:
                    t["tkind"] = TK_POINT
                add(
                    loop=L,
                    player=p,
                    kind=A_PING,
                    minimap=bool(int(d["flags"][i]) & ra.FLAG_MINIMAP),
                    **t,
                )
            elif k == ra.TRADE:
                add(
                    loop=L,
                    player=p,
                    kind=A_TRADE,
                    recipient=int(d["recipient"][i]),
                    minerals=int(d["minerals"][i]),
                    gas=int(d["gas"][i]),
                )
            elif k == ra.REQUEST:
                add(
                    loop=L,
                    player=p,
                    kind=A_REQUEST,
                    minerals=int(d["minerals"][i]),
                    gas=int(d["gas"][i]),
                )
            elif k == ra.REQUEST_FULFILL:
                add(loop=L, player=p, kind=A_FULFILL)
            elif k == ra.REQUEST_CANCEL:
                add(loop=L, player=p, kind=A_CANCEL)
            elif k == ra.LEAVE:
                add(loop=L, player=p, kind=A_LEAVE)
            last_raw[p] = k
        self.tok = {k: np.asarray(v, dtype=TOKEN_COLS[k]) for k, v in cols.items()}
        self.sel_uid = np.asarray(sel, np.int64)
        self.sel_end = np.append(self.tok["sel_start"][1:], len(self.sel_uid)).astype(
            np.int64
        )  # token i owns [start, end)
        self.n_raw = len(kind)
        self.tok_of_raw = tok_of_raw
        self.raw = {
            k: d[k]
            for k in (
                "kind",
                "loop",
                "player",
                "target",
                "tx",
                "ty",
                "target_tag",
                "abil_link",
            )
        }  # what command_join.match reads

    def __len__(self):
        return len(self.tok["kind"])


def self_family(avocab, links, link):
    name = links.get(link)
    return avocab.family.get(name, FAM_UNK) if name else FAM_UNK
