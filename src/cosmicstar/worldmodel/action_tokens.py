"""Replay actions -> action tokens, aligned with the state frames.

One token per thing a player did (raw events: replay_actions.py):

    CMD        a new command: ability (family + button; right-click = Smart), flags, target
    REISSUE    the player's last command executed again:
                 - a retarget: CMD_POINT / CMD_UNIT changes the last command's target and the
                   REPEAT that comes with it fires it (one token, the REPEAT is folded in)
                 - a plain REPEAT: same command, same target
    SELECT     selection change: the added units (pointers), how many, how units were removed
    GROUP      control group operation + number
    CAMERA, CAMERA_SAVE, PING, TRADE, REQUEST, FULFILL, CANCEL, LEAVE

A held or mashed hotkey (a CMD / REISSUE followed by plain REPEATs within BURST loops, with
nothing else from that player in between) stays one token with a press count: "Drone x7".
Each press is a separate event on disk; the folding happens only here, so the raw stream
stays lossless for a future control decoder.

Units an action refers to (command targets, pinged units, the units added to a selection)
are pointers: the unit's own state features at the latest frame at or before the action
(GameFrames.entity_features), embedded by the state encoder like any entity. Fogged snapshots
carry no tag; they get their unit type and position instead.

Abilities are named per ruleset from the Editor exports (data/worldmodel/ability_links.json):
(family, button), e.g. ("BarracksTrain", 0) = Marine. The family and button vocabularies are
shared across builds, so link numbers that move between patches don't matter.
"""

import json
import os

import numpy as np

from . import replay_actions as ra
from .tokenize import CODE_NEUTRAL, _log

LINKS = "data/worldmodel/ability_links.json"
RULES = "data/worldmodel/rules"
ENGINE_TAG_OFFSET = 1 << 32

(A_PAD, A_CMD, A_REISSUE, A_SELECT, A_GROUP, A_CAMERA, A_CAMERA_SAVE, A_PING, A_TRADE,
 A_REQUEST, A_FULFILL, A_CANCEL, A_LEAVE) = range(13)
N_ACT_KINDS = 13
FAM_PAD, FAM_UNK, FAM_SMART, FAM_FIRST = 0, 1, 2, 3
N_BUTTONS = 32                   # button index + 1 (0 = no ability)
TK_NONE, TK_POINT, TK_UNIT, TK_SNAPSHOT = ra.TARGET_NONE, ra.TARGET_POINT, ra.TARGET_UNIT, ra.TARGET_SNAPSHOT
N_GROUPS, N_GROUP_OPS, N_REMOVE = 12, 7, 5  # +1 each: 0 = n/a
BURST = 6                        # loops: plain REPEATs this close to the last press are the same burst
MAX_SELECTED = 32                # pointers kept per SELECT (the count keeps the full number)
C_ACTION = 12                    # numeric features per token (see GameActions.window)

TOKEN_COLS = dict(loop=np.int32, player=np.int8, kind=np.int8, family=np.int32, button=np.int8,
                  queued=np.bool_, smart=np.bool_, minimap=np.bool_, presses=np.int16,
                  tkind=np.int8, x=np.float32, y=np.float32, uid=np.int32, snap=np.int64,
                  group=np.int8, gop=np.int8, remove=np.int8, n_add=np.int16, n_rem=np.int16,
                  recipient=np.int8, minerals=np.int32, gas=np.int32, sel_start=np.int64)


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
        """The link tables for a game: its own ruleset, or another ruleset of the same build."""
        t = self.table.get(meta.get("ruleset"))
        if t is None:
            t = next((v for v in self.table.values() if v["base_build"] == meta.get("base_build")), None)
        if t is None:
            raise KeyError(f"no ability table for {meta.get('ruleset')} (build {meta.get('base_build')})")
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
    """One game's action tokens (all players, time order), built once from the raw events."""

    def __init__(self, path, frames, avocab):
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}  # NpzFile re-reads an array on every access
        self.fr = frames
        table = avocab.ruleset_table(frames.meta)
        links = {int(k): v for k, v in table["links"].items()}
        unit_links = {int(k): v for k, v in table["units"].items()}
        name_to_api = avocab.unit_ids(frames.meta)
        tag_to_uid = {int(t): i for i, t in enumerate(frames.unit_tag)} if frames.unit_tag is not None else {}

        def uid_of(replay_tag):
            return tag_to_uid.get(int(replay_tag) + ENGINE_TAG_OFFSET, -1) if replay_tag else -1

        def snap_type(link):
            api = name_to_api.get(unit_links.get(int(link), ""), 0)
            return int(frames.vocab.unit_index([api])[0]) if api else 0

        cols = {k: [] for k in TOKEN_COLS}
        sel = []
        last_cmd, last_tok, last_raw, press_loop, awaiting = {}, {}, {}, {}, {}

        def add(**kw):
            for k, dt in TOKEN_COLS.items():
                default = np.nan if k in ("x", "y") else (False if dt is np.bool_ else 1 if k == "presses"
                                                          else 0 if k in ("family", "button", "snap") else -1)
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
                out["snap"] = snap_type(d["snap_link"][i])
            return out

        kind, loop, player = d["kind"], d["loop"], d["player"]
        add_start = np.append(d["add_start"], len(d["add_tag"]))
        rem_start = np.append(d["rem_start"], len(d["rem_pos"]))
        for i in range(len(kind)):
            k, L, p = int(kind[i]), int(loop[i]), int(player[i])
            prev = last_raw.get(p)
            if k == ra.CMD:
                link, cmd, fl = int(d["abil_link"][i]), int(d["abil_cmd"][i]), int(d["flags"][i])
                ability = dict(family=FAM_SMART if link < 0 else self_family(avocab, links, link),
                               button=0 if link < 0 else min(cmd, N_BUTTONS - 1) + 1,
                               queued=bool(fl & ra.FLAG_QUEUED), smart=bool(fl & ra.FLAG_SMART),
                               minimap=bool(fl & ra.FLAG_MINIMAP))
                last_cmd[p] = dict(ability, **target(i))
                last_tok[p] = add(loop=L, player=p, kind=A_CMD, **last_cmd[p])
                press_loop[p] = L
            elif k in (ra.CMD_POINT, ra.CMD_UNIT):
                base = last_cmd.get(p, dict(family=FAM_UNK, button=0))
                last_cmd[p] = dict({kk: v for kk, v in base.items() if kk not in ("tkind", "x", "y", "uid", "snap")},
                                   **target(i), minimap=bool(int(d["flags"][i]) & ra.FLAG_MINIMAP) or base.get("minimap", False))
                last_tok[p] = add(loop=L, player=p, kind=A_REISSUE, **last_cmd[p])
                press_loop[p], awaiting[p] = L, L
            elif k == ra.REPEAT:
                if awaiting.get(p) == L and prev in (ra.CMD_POINT, ra.CMD_UNIT):
                    awaiting[p] = None  # the REPEAT that fires a retarget: already one token
                elif p in last_tok and prev in (ra.CMD, ra.REPEAT, ra.CMD_POINT, ra.CMD_UNIT) \
                        and L - press_loop.get(p, -BURST - 1) <= BURST:
                    cols["presses"][last_tok[p]] += 1  # held / mashed hotkey
                else:
                    last_tok[p] = add(loop=L, player=p, kind=A_REISSUE, **last_cmd.get(p, dict(family=FAM_UNK)))
                press_loop[p] = L
            elif k == ra.SELECT:
                tags = d["add_tag"][add_start[i]:add_start[i + 1]]
                n_rem = rem_start[i + 1] - rem_start[i]
                add(loop=L, player=p, kind=A_SELECT, group=int(d["group"][i]) + 1,
                    remove=int(d["select_remove"][i]) + 1, n_add=len(tags), n_rem=n_rem)
                sel.extend(uid_of(t) for t in tags[:MAX_SELECTED])
            elif k == ra.GROUP:
                add(loop=L, player=p, kind=A_GROUP, group=int(d["group"][i]) + 1, gop=int(d["group_op"][i]) + 1)
            elif k == ra.CAMERA:
                add(loop=L, player=p, kind=A_CAMERA, x=float(d["cam_x"][i]), y=float(d["cam_y"][i]))
            elif k == ra.CAMERA_SAVE:
                add(loop=L, player=p, kind=A_CAMERA_SAVE, group=int(d["group"][i]) + 1,
                    x=float(d["cam_x"][i]), y=float(d["cam_y"][i]))
            elif k == ra.PING:
                t = target(i)
                t.update(x=float(d["tx"][i]), y=float(d["ty"][i]))  # a ping always has a point
                if t["tkind"] == TK_UNIT and t.get("uid", -1) < 0:
                    t["snap"] = snap_type(d["snap_link"][i])
                elif t["tkind"] != TK_UNIT:
                    t["tkind"] = TK_POINT
                add(loop=L, player=p, kind=A_PING, minimap=bool(int(d["flags"][i]) & ra.FLAG_MINIMAP), **t)
            elif k == ra.TRADE:
                add(loop=L, player=p, kind=A_TRADE, recipient=int(d["recipient"][i]),
                    minerals=int(d["minerals"][i]), gas=int(d["gas"][i]))
            elif k == ra.REQUEST:
                add(loop=L, player=p, kind=A_REQUEST, minerals=int(d["minerals"][i]), gas=int(d["gas"][i]))
            elif k == ra.REQUEST_FULFILL:
                add(loop=L, player=p, kind=A_FULFILL)
            elif k == ra.REQUEST_CANCEL:
                add(loop=L, player=p, kind=A_CANCEL)
            elif k == ra.LEAVE:
                add(loop=L, player=p, kind=A_LEAVE)
            last_raw[p] = k
        self.tok = {k: np.asarray(v, dtype=TOKEN_COLS[k]) for k, v in cols.items()}
        self.sel_uid = np.asarray(sel, np.int64)
        self.sel_end = np.append(self.tok["sel_start"][1:], len(self.sel_uid)).astype(np.int64)  # token i owns [start, end)
        self.n_raw = len(kind)

    def __len__(self):
        return len(self.tok["kind"])

    def _pointer_rows(self, uids, loops):
        """Unit rows for pointers: the latest frame at or before each action (or the next
        frame, for units created in between); -1 where the unit isn't found."""
        fr = self.fr
        f = np.clip(np.searchsorted(fr.loops, loops, side="right") - 1, 0, len(fr) - 1)
        rows = np.full(len(uids), -1, np.int64)
        for ff in np.unique(f):
            m = (f == ff) & (uids >= 0)
            if m.any():
                rows[m] = fr.rows_of(int(ff), uids[m])
                miss = m & (rows < 0)
                if miss.any() and ff + 1 < len(fr):
                    rows[miss] = fr.rows_of(int(ff) + 1, uids[miss])
        return rows

    def window(self, lo, hi, bounds, t_loop, owners, only=None, static_pointers=False):
        """Tokens with lo <= loop < hi, as encoder inputs for one sample.

        only: a slot whose actions are the only ones kept (a player's view: they don't see
        anyone else's, not even their teammate's); then there's one summary per segment, theirs.
        static_pointers: the units a command points at (targets, selections) keep only what the
        player did (which unit: type and owner, and where it was clicked), not the unit's state
        at that moment (hp, energy, progress). For actions in the interval being predicted,
        that state is the future.

        bounds: segment start loops (ascending); a token's segment is the last bound <= its
        loop. t_loop: the sample's present (times are in seconds relative to it).
        owners: the sample's owner map (same codes as the state tokens)."""
        tk, fr = self.tok, self.fr
        idx = np.where((tk["loop"] >= lo) & (tk["loop"] < hi) & ((tk["player"] == only) if only else True))[0]
        n = len(idx)
        neutral = (CODE_NEUTRAL, CODE_NEUTRAL, 0)
        own = np.array([owners.get(int(s), neutral) for s in tk["player"][idx]], np.int64).reshape(-1, 3)
        rec = np.array([owners.get(int(s), neutral) if s >= 0 else neutral for s in tk["recipient"][idx]],
                       np.int64).reshape(-1, 3)
        xy = np.stack([tk["x"][idx], tk["y"][idx]], 1)
        has_xy = np.isfinite(xy).all(1)
        loops = tk["loop"][idx]

        # pointers: command / ping targets
        trow = self._pointer_rows(tk["uid"][idx].astype(np.int64), loops)
        tok_ok = trow >= 0
        tgt = fr.entity_features(np.where(tok_ok, trow, 0), owners, prefix="tgt")
        # pointers: units added to a selection
        s0, s1 = tk["sel_start"][idx], self.sel_end[idx]
        s_ev = np.repeat(np.arange(n), s1 - s0)
        s_uid = self.sel_uid[np.concatenate([np.arange(a, b) for a, b in zip(s0, s1)])] if n else np.zeros(0, np.int64)
        srow = self._pointer_rows(s_uid, loops[s_ev]) if len(s_uid) else np.zeros(0, np.int64)
        keep = srow >= 0
        sel = fr.entity_features(srow[keep], owners, prefix="sel")
        sel["sel_event"] = s_ev[keep]
        if static_pointers:
            for d_, pre in ((tgt, "tgt"), (sel, "sel")):
                d_[f"{pre}_num"][:] = 0
                d_[f"{pre}_produce"][:] = 0
                d_[f"{pre}_order"][:] = 0
                d_[f"{pre}_ord"][:] = 0

        summarized = [only] if only else fr.slots
        slot_index = {s: i for i, s in enumerate(summarized)}
        num = np.stack([
            has_xy, tk["queued"][idx], tk["smart"][idx], tk["minimap"][idx],
            _log(tk["presses"][idx] - 1.0, 2.0),
            _log(tk["n_add"][idx].clip(0).astype(np.float32), 3.0),
            _log(tk["n_rem"][idx].clip(0).astype(np.float32), 3.0),
            _log(tk["minerals"][idx].clip(0).astype(np.float32), 8.0),
            _log(tk["gas"][idx].clip(0).astype(np.float32), 8.0),
            tk["recipient"][idx] >= 0, tok_ok, (tk["uid"][idx] >= 0) & ~tok_ok,  # pointer missing
        ], 1).astype(np.float32).reshape(-1, C_ACTION)
        out = dict(
            act_kind=tk["kind"][idx].astype(np.int64),
            act_family=tk["family"][idx].astype(np.int64),
            act_button=tk["button"][idx].astype(np.int64),
            act_tkind=tk["tkind"][idx].clip(0).astype(np.int64),
            act_group=tk["group"][idx].clip(0).astype(np.int64),
            act_gop=tk["gop"][idx].clip(0).astype(np.int64),
            act_remove=tk["remove"][idx].clip(0).astype(np.int64),
            act_team=own[:, 0], act_pcode=own[:, 1],
            act_rteam=rec[:, 0], act_rpcode=rec[:, 1],
            act_snap=tk["snap"][idx].astype(np.int64),
            act_xy=np.where(has_xy[:, None], fr.normalize_xy(np.nan_to_num(xy)), 0).astype(np.float32),
            act_num=num,
            act_t=((loops - t_loop) / 22.4).astype(np.float32),
            act_seg=(np.searchsorted(bounds, loops, side="right") - 1).astype(np.int64),
            act_player=np.array([slot_index.get(int(s), 0) for s in tk["player"][idx]], np.int64),
            act_tgt_ok=tok_ok,
            # one summary per (segment, player): when each segment starts, who the players are
            seg_t=((np.asarray(bounds) - t_loop) / 22.4).astype(np.float32),
            sum_team=np.array([owners[s][0] for s in summarized], np.int64),
            sum_pcode=np.array([owners[s][1] for s in summarized], np.int64),
            # state unit ids the window's commands touched (selected or targeted), for diagnostics
            touched_uid=np.unique(np.concatenate([s_uid, tk["uid"][idx].astype(np.int64)])) if n else np.zeros(0, np.int64),
        )
        out.update(tgt)
        out.update(sel)
        return out


def self_family(avocab, links, link):
    name = links.get(link)
    return avocab.family.get(name, FAM_UNK) if name else FAM_UNK


ACT_KEYS = ("kind", "family", "button", "tkind", "group", "gop", "remove", "team", "pcode", "rteam",
            "rpcode", "snap", "xy", "num", "t", "seg", "player", "tgt_ok")
ENT_KEYS = ("type", "team", "pcode", "rel", "produce", "order", "ord", "xy", "num")


def collate_actions(windows):
    """Pads action windows into a batch.

    act_*: [B, N] padded (act_mask True for real tokens); tgt_*: [B, N] target pointers;
    sel_*: [M] flat, sel_event = flat token index (b * N + i) each selected unit belongs to."""
    import torch
    B = len(windows)
    N = max(max((len(w["act_kind"]) for w in windows), default=0), 1)
    out = {}
    mask = np.zeros((B, N), bool)
    for b, w in enumerate(windows):
        mask[b, :len(w["act_kind"])] = True
    out["act_mask"] = torch.from_numpy(mask)
    for g, keys in (("act", ACT_KEYS), ("tgt", ENT_KEYS)):
        for k in keys:
            arrs = [w[f"{g}_{k}"] for w in windows]
            buf = np.zeros((B, N) + arrs[0].shape[1:], arrs[0].dtype)
            for b, a in enumerate(arrs):
                buf[b, :len(a)] = a
            out[f"{g}_{k}"] = torch.from_numpy(buf)
    for k in ENT_KEYS:
        out[f"sel_{k}"] = torch.from_numpy(np.concatenate([w[f"sel_{k}"] for w in windows]))
    out["sel_event"] = torch.from_numpy(np.concatenate([w["sel_event"] + b * N for b, w in enumerate(windows)]))
    P = max(len(w["sum_team"]) for w in windows)
    pm = np.zeros((B, P), bool)
    team, pcode = np.zeros((B, P), np.int64), np.zeros((B, P), np.int64)
    for b, w in enumerate(windows):
        n = len(w["sum_team"])
        pm[b, :n], team[b, :n], pcode[b, :n] = True, w["sum_team"], w["sum_pcode"]
    out["sum_mask"], out["sum_team"], out["sum_pcode"] = torch.from_numpy(pm), torch.from_numpy(team), torch.from_numpy(pcode)
    out["seg_t"] = torch.from_numpy(np.stack([w["seg_t"] for w in windows]))  # same segment count per sample
    return out
