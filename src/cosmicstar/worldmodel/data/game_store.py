"""The physics model's training store (DESIGN_physics.md): any window of a game read
straight from its Arrow views.

A game is two views, p1 and p2, each with that player's own units and the neutrals as
that player knows them. Players are indexed 0, 1 (neutral 2). An own row comes from its
player's view. A neutral row joins both views: it keeps one row by its type and place,
since a neutral out of sight is a snapshot under another tag; critters go by tag and
exist only while seen. <game>.index.npz (build_index, once per game) holds what a window
can't work out on its own: each row's first and last frame, its death frame and its
parent, and each neutral's contents as last seen, since a snapshot reports none.
Contents change only in a player's view, so before they are first seen they are what
they are when they are.

A sequence is a grid, frames x rows, with a unit on the same row throughout:

    mask       the row has a token in this frame. False only before birth, or while a unit is out of sight unexplained
    alive      false from the frame after its death: the row keeps its last values, hit points and shields 0
    inside     0 on the map, 1 passenger, 2 in a gas building (bridged as tokenize._gas_gaps does), with parent_row

Targets are the same columns one frame later. Births are rows whose first frame in the
whole game is inside the window.
"""

import glob
import json
import os
import threading
import types

import numpy as np

from . import arrow_state, command_join, statefile
from .action_tokens import GameActions
from ..core.constants import (
    C_GROUPS,
    GAS_GAP_LOOPS,
    GAS_TYPES as _GAS,
    NEUTRAL_PLAYER,
    N_EFFECT_IDS,
    N_RAW_ABILITY_IDS,
    PC_KEYS,
    WORKER_TYPES,
)
from ..core.paths import RULESETS_PATH

GAS_TYPES = (*_GAS, 1955, 1956, 1960)  # the rich gas buildings of these patches too

VIEWS = (1, 2)  # the views' file suffixes: player slots
SOURCES = (
    1,
    2,
    0,
)  # where rows come from: each player's own, then the neutrals of both views
MOVING_NEUTRALS = (
    321,
    322,
    335,
    336,
    337,
    612,
    659,
    660,
    661,
)  # neutral types seen moving in 30 games: critters; data, not names
NEUTRAL = 2  # owner index of neutral rows
BATCH = 64  # frames per Arrow batch (extract_states.ARROW_FRAMES)
MARGIN = (
    GAS_GAP_LOOPS + 1
)  # frames read either side of a window, to bridge gaps across its edges
PATCH = 3  # real cells per side each 64-level map token reads
TOP = 64  # map tokens per side
C_DYNAMIC = 6  # map layers decoded: (visible, seen, creep) per player
# map layers read: (visible, seen, creep, pathable, buildable) per player; height per
# player, playable, known
C_LIVE, C_STATIC = (
    10,
    4,
)
MAX_ROWS = 4096
SNAPSHOT = 2  # u_display
_RULESET_IDS: dict[str, int] | None = None


def ruleset_id_of(ruleset_name: str, rulesets_path: str = RULESETS_PATH) -> int:
    """Canonical integer index of `ruleset_name` inside sorted `rulesets.json` (`-1` if absent)."""
    global _RULESET_IDS
    if _RULESET_IDS is None:
        if os.path.exists(rulesets_path):
            rs = json.load(open(rulesets_path))["rulesets"]
            _RULESET_IDS = {k: i for i, k in enumerate(sorted(rs.keys()))}
        else:
            _RULESET_IDS = {}
    if ruleset_name in _RULESET_IDS:
        return _RULESET_IDS[ruleset_name]
    prefix = str(ruleset_name).split("_")[0]
    for k, idx in _RULESET_IDS.items():
        if k.split("_")[0] == prefix:
            return idx
    return -1


def base_paths(root, split_ids=None):
    """Game id -> its base path (no view suffix), for ids with both views."""
    out = {}
    for u in glob.glob(os.path.join(root, "1v1", "*", "*", "*.p1.units.arrow")):
        base = u[: -len(".p1.units.arrow")]
        gid = os.path.basename(base)
        if (split_ids is None or gid in split_ids) and all(
            os.path.exists(f"{base}.p{s}.units.arrow") for s in VIEWS
        ):
            out[gid] = base
    return out


def build_index(base, replay_path=None):
    """One game's <base>.index.npz: for each view, row ids with first frame, last frame,
    death frame (-1: never) and parent id (-1: none), and frames of the game. Parents
    come from the replay's creator tags, so own-view ids only.
    """
    out = {}
    for s in VIEWS:
        r, fr = arrow_state.open_view(f"{base}.p{s}")
        first, last = {}, {}
        for i in range(r.num_record_batches):
            b = r.get_batch(i)
            ids, frame = b.column("u_id").to_numpy(zero_copy_only=False), b.column(
                "frame"
            ).to_numpy(zero_copy_only=False)
            own = _own_mask(b.column("u_owner").to_numpy(zero_copy_only=False), s)
            io, fo = ids[own], frame[own]
            uq, idx = np.unique(io, return_index=True)
            for k, j in zip(uq.tolist(), fo[idx].tolist()):
                first.setdefault(k, j)
            uq, idx = np.unique(io[::-1], return_index=True)
            last.update(zip(uq.tolist(), fo[::-1][idx].tolist()))
        d = statefile.load(f"{base}.p{s}.cst")
        ids = np.array(sorted(first), np.int32)
        pos = {k: i for i, k in enumerate(ids.tolist())}
        death = np.full(len(ids), -1, np.int32)
        ds, di = d["dead_frame_start"], d["dead_id"]
        for f in np.nonzero(np.diff(ds))[0].tolist():
            for k in di[ds[f] : ds[f + 1]].tolist():
                if k in pos:
                    death[pos[k]] = f
        parent = np.full(len(ids), -1, np.int32)
        if s and replay_path:
            parent = _parents(replay_path, d["unit_tag"], ids)
        out[f"id{s}"], out[f"first{s}"] = ids, np.array(
            [first[k] for k in ids.tolist()], np.int32
        )
        out[f"last{s}"], out[f"death{s}"], out[f"parent{s}"] = (
            np.array([last[k] for k in ids.tolist()], np.int32),
            death,
            parent,
        )
        out["F"] = np.int32(
            sum(fr.get_batch(i).num_rows for i in range(fr.num_record_batches))
        )
    out.update(_neutral_index(base))
    tmp = f"{base}.index.tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, f"{base}.index.npz")
    return int(out["F"])


def _neutral_index(base):
    """The neutrals' rows over both views: each view's unit id -> a neutral id
    (nid<view>, -1: not neutral), each neutral's first, last and death frame, and its
    contents whenever they change while seen, per neutral (nlog_*).
    """
    keys, out, rows, deaths = {}, {}, [], []
    for s in VIEWS:
        r, _ = arrow_state.open_view(f"{base}.p{s}")
        d = statefile.load(f"{base}.p{s}.cst")
        tags = d["unit_tag"]
        nid = np.full(len(tags), -1, np.int32)
        for i in range(r.num_record_batches):
            u, _, fr = arrow_state.read_units(r, i)
            k = _own_mask(u["u_owner"], 0)
            ids, fo = u["u_id"][k], fr[k]
            uq, at = np.unique(ids, return_index=True)
            for v, j in zip(uq.tolist(), at.tolist()):
                if nid[v] < 0:
                    ty = int(u["u_type"][k][j])
                    key = (
                        ("tag", int(tags[v]))
                        if ty in MOVING_NEUTRALS
                        else (
                            ty,
                            round(float(u["u_x"][k][j]), 1),
                            round(float(u["u_y"][k][j]), 1),
                        )
                    )
                    nid[v] = keys.setdefault(key, len(keys))
            rows.append(
                (nid[ids], fo, u["u_display"][k] != SNAPSHOT, u["u_contents"][k])
            )
        ds, di = d["dead_frame_start"], d["dead_id"]
        for f in np.nonzero(np.diff(ds))[0].tolist():
            v = di[ds[f] : ds[f + 1]]
            v = nid[v[(v >= 0) & (v < len(nid))]]
            deaths += [(int(x), f) for x in v[v >= 0].tolist()]
        out[f"nid{s}"] = nid
    N = len(keys)
    n, fo, seen, cont = (np.concatenate([x[i] for x in rows]) for i in range(4))
    first, last, death = (
        np.full(N, 2**30, np.int32),
        np.full(N, -1, np.int32),
        np.full(N, -1, np.int32),
    )
    np.minimum.at(first, n, fo)
    np.maximum.at(last, n, fo)
    for k, f in sorted(deaths, key=lambda x: -x[1]):
        death[k] = f
    last = np.where(death >= 0, np.minimum(last, death), last)
    n, fo, cont = n[seen], fo[seen], cont[seen]
    o = np.lexsort((fo, n))
    n, fo, cont = n[o], fo[o], cont[o]
    keep = np.ones(len(n), bool)
    keep[1:] = (n[1:] != n[:-1]) | (cont[1:] != cont[:-1])
    n, fo, cont = n[keep], fo[keep], cont[keep]
    out.update(
        id0=np.arange(N, dtype=np.int32),
        first0=first,
        last0=last,
        death0=death,
        parent0=np.full(N, -1, np.int32),
        nlog_start=np.searchsorted(n, np.arange(N + 1)).astype(np.int64),
        nlog_frame=fo.astype(np.int32),
        nlog_contents=cont.astype(np.float32),
    )
    return out


def _parents(replay_path, unit_tag, ids):
    """A view's row ids -> their creator's id, from the replay's SUnitBornEvent (-1: no
    creator, or not this view's)."""
    import mpyq
    from . import replay_actions

    a = mpyq.MPQArchive(replay_path)
    proto, _ = replay_actions._protocol(a)
    off = 1 << 32
    tag_id = {int(t): i for i, t in enumerate(unit_tag.tolist())}
    out = np.full(len(ids), -1, np.int32)
    pos = {int(k): i for i, k in enumerate(ids)}
    for e in proto.decode_replay_tracker_events(a.read_file("replay.tracker.events")):
        if (
            not e["_event"].endswith("SUnitBornEvent")
            or e.get("m_creatorUnitTagIndex") is None
        ):
            continue
        me = tag_id.get(off + (e["m_unitTagIndex"] << 18) + e["m_unitTagRecycle"])
        par = tag_id.get(
            off + (e["m_creatorUnitTagIndex"] << 18) + e["m_creatorUnitTagRecycle"]
        )
        if me in pos and par is not None:
            out[pos[me]] = par
    return out


def _own_mask(owner, s):
    return owner == (s if s else NEUTRAL_PLAYER)


class Game:
    """One game's two views, opened lazily: batches are memory mapped, so windows of
    many games can be read without holding any."""

    def __init__(self, base, vocab):
        self.base, self.vocab = base, vocab
        self._window_lock = threading.Lock()
        self.views = {s: arrow_state.open_view(f"{base}.p{s}") for s in VIEWS}
        self.map_id = (
            0  # the map's learned tokens, set by whoever knows the training maps
        )
        ix = np.load(f"{base}.index.npz")
        self.F = int(ix["F"])
        self.index = {
            s: {k: ix[f"{k}{s}"] for k in ("id", "first", "last", "death", "parent")}
            for s in SOURCES
        }
        self.nid = {s: ix[f"nid{s}"] for s in VIEWS}
        self.nlog = ix["nlog_start"], ix["nlog_frame"], ix["nlog_contents"]
        self.rest = {s: statefile.load(f"{base}.p{s}.cst") for s in (1, 2)}
        self._eff_age = {}  # view -> effect_ages, made when first read
        d = self.rest[1]
        self.meta = json.loads(str(d["meta"]))
        W, H = self.W, self.H = self.meta["map_size"]
        x0, y0, x1, y1 = self.playable = self.meta["playable"]
        self.center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
        self.half = np.array([(x1 - x0) / 2, (y1 - y0) / 2], np.float32)
        self.side = float(max(x1 - x0, y1 - y0))
        self.cells = self._cell_index()
        h_ = d["map_terrain_height"].reshape(H, W)[y0:y1, x0:x1]
        # [h, w, C_STATIC]: height per player, then playable and known: zero only where
        # a batch pads
        self.static_grid = np.stack([h_, h_, np.ones_like(h_), np.ones_like(h_)], -1)
        self.grp = {
            s: (
                (self.rest[s]["grp_start"], self.rest[s]["grp_id"])
                if "grp_start" in self.rest[s].files
                else None
            )
            for s in (1, 2)
        }
        self.tag_row = {
            s: {int(t): i for i, t in enumerate(self.rest[s]["unit_tag"].tolist())}
            for s in (1, 2)
        }
        self.acts = None
        # any view's unit id -> a flat index over all rows: tags are the same in every
        # view, so a pointer reaches a row in another view, and a neutral's tags all
        # reach its one row
        tags = {s: self.rest[s]["unit_tag"] for s in VIEWS}
        self.flat_off, n, tag_flat = {}, 0, {}
        for s in VIEWS:
            self.flat_off[s] = n
            for k, i in enumerate(self.index[s]["id"].tolist()):
                tag_flat[int(tags[s][i])] = n + k
            n += len(self.index[s]["id"])
        self.flat_off[0] = n
        for s in VIEWS:
            for i in np.nonzero(self.nid[s] >= 0)[0].tolist():
                tag_flat[int(tags[s][i])] = n + int(self.nid[s][i])
        n += len(self.index[0]["id"])
        self.n_flat = n
        self.to_flat = {
            s: np.array([tag_flat.get(int(t), -1) for t in tags[s].tolist()], np.int64)
            for s in VIEWS
        }

    def _cell_index(self):
        """(y, x) real cell of each of the PATCH x PATCH samples of each map token:
        [TOP, TOP, PATCH * PATCH]. Tokens cover a square of the playable area's longest
        side, around its center; samples outside the map clip to its edge, where
        playable is false.
        """
        s = self.side / TOP
        c = (np.arange(TOP) + 0.5) * s - self.side / 2
        off = np.arange(PATCH) - (PATCH - 1) / 2
        xs = np.floor(self.center[0] + c[None, :, None] + off[None, None, :]).astype(
            np.int64
        )
        ys = np.floor(self.center[1] + c[:, None, None] + off[None, None, :]).astype(
            np.int64
        )
        x = np.broadcast_to(xs[:, :, None, :], (TOP, TOP, PATCH, PATCH)).reshape(
            TOP, TOP, -1
        )
        y = np.broadcast_to(ys[:, :, :, None], (TOP, TOP, PATCH, PATCH)).reshape(
            TOP, TOP, -1
        )
        return np.clip(y, 0, self.H - 1), np.clip(x, 0, self.W - 1)

    def _read(self, s, f0, f1, neutral=False):
        """View s, frames [f0, f1): its own rows', or the neutrals', columns and list
        columns, oldest first, already decoded and filtered."""
        r, _ = self.views[s]
        out_u, out_l = [], []
        for i in range(
            max(0, f0 // BATCH), min(r.num_record_batches, (f1 - 1) // BATCH + 1)
        ):
            if (s, i) not in self._batches:
                self._batches[(s, i)] = arrow_state.read_units(r, i)
            u, l, fr = self._batches[(s, i)]
            keep = _own_mask(u["u_owner"], 0 if neutral else s) & (fr >= f0) & (fr < f1)
            # a player's own snapshots are not units (see the desync check): tags
            # outside the replay's log
            if not neutral:
                keep &= u["u_display"] != SNAPSHOT
            out_u.append({k: v[keep] for k, v in u.items()} | {"frame": fr[keep]})
            out_l.append((keep, l))
        return out_u, out_l

    def window(self, t0, L, swap=False, step=1, need_map=True, compact=False):
        """L frames from frame t0 spaced by `step` game loops, as one dict of numpy
        arrays: see the module docstring. One window at a time per game: a window keeps
        its row map and batches on the game, and two threads reading the same game would
        map one window's targets with the other's rows.
        """
        with self._window_lock:
            return self._window(
                t0, L, swap, step=step, need_map=need_map, compact=compact
            )

    def _window(self, t0, L, swap, step=1, need_map=True, compact=False):
        step = max(1, int(step))
        t_end = t0 + (L - 1) * step
        rows, row_of = [], {}
        self._batches = {}  # own rows and neutrals share a view's batches
        self._uid_row = {
            s: np.full(len(self.rest[s]["unit_tag"]), -1, np.int64) for s in VIEWS
        }
        for s in SOURCES:
            ix = self.index[s]
            live = (ix["first"] <= t_end) & (ix["last"] >= t0)
            for k in np.nonzero(live)[0].tolist():
                r_idx = len(rows)
                uid = int(ix["id"][k])
                row_of[(s, uid)] = r_idx
                if s in self._uid_row and 0 <= uid < len(self._uid_row[s]):
                    self._uid_row[s][uid] = r_idx
                rows.append((s, k))
        R = len(rows)
        assert R <= MAX_ROWS, f"{R} rows"
        out = _blank(L, R)
        self.flat_row = np.full(
            self.n_flat + 1, -1, np.int32
        )  # the last entry answers id -1
        for r_, (s, k) in enumerate(rows):
            self.flat_row[self.flat_off[s] + k] = r_
        for s in SOURCES:
            self._fill(out, s, t0, L, row_of, step=step)
        self._dead_rows(out, rows, t0, L, step=step)
        self._batches = {}
        self._uid_row = None
        sw = (lambda o: 1 - o if o < 2 else o) if swap else (lambda o: o)
        out["owner"] = np.array(
            [sw({1: 0, 2: 1, 0: NEUTRAL}[s]) for s, _ in rows], np.int64
        ).reshape(-1)
        births, parents = [], []
        for r_, (s, k) in enumerate(rows):
            if s == 0:
                continue
            ix = self.index[s]
            raw_first = int(ix["first"][k])
            if t0 < raw_first <= t_end:
                f = (raw_first - t0 + step - 1) // step
                if 0 < f < L:
                    births.append((f, r_))
                    p = (
                        row_of.get((s, int(ix["parent"][k])), -1)
                        if ix["parent"][k] >= 0
                        else -1
                    )
                    parents.append(p)
        out["birth_frame"] = np.array([b[0] for b in births], np.int64)
        out["birth_row"] = np.array([b[1] for b in births], np.int64)
        out["birth_parent"] = np.array(parents, np.int64)
        out["death_frame"] = np.array(
            [self._death(s, k, t0, L, step=step) for s, k in rows], np.int64
        )
        out["map"], out["grid"] = self._map_frames(
            t0, L, swap, step=step, need_map=need_map, compact=compact
        )
        out["static_grid"], out["origin"], out["map_id"] = (
            self.static_grid,
            np.array(self.playable[:2], np.float32),
            np.int64(self.map_id),
        )
        out["center"], out["half"], out["side"] = (
            self.center,
            self.half,
            np.float32(self.side),
        )
        out["loops"] = t0 + 1 + np.arange(L) * step
        out["players"] = np.array(
            [self.vocab.race.get(p["race"], 3) for p in self.meta["players"]], np.int64
        )[:: -1 if swap else 1]
        out["ply_num"] = self._players(t0, L, swap, step=step)
        out["actions"] = self._actions(t0, L, row_of, swap, step=step)
        out["effects"] = self._effects(t0, L, swap, step=step)
        out["autocast"] = self._autocast(t0, L, step=step)
        out["ruleset_id"] = np.int64(ruleset_id_of(self.meta.get("ruleset", "")))
        return out

    def _autocast(self, t0, L, step=1):
        """Each row's remembered autocast toggles per frame: [(frame, row, family,
        button, on)]. A toggle at loop l first shows in the frame after l's, like any
        other action's effect. A unit never toggled has no entry: its default isn't
        known.
        """
        t_end = t0 + (L - 1) * step
        out = []
        for s, events in getattr(self, "autocast", {}).items():
            state = {}
            for loop, on, fam, but, units in events:
                if loop >= t_end + 2:
                    break
                f0 = max(0, (loop - t0 + step - 1) // step)
                rows = self._row_of_id(s, units)
                for r_ in rows[rows >= 0].tolist():
                    key = (r_, fam, but)
                    if key in state:
                        out.append((state[key][0], f0, r_, fam, but, state[key][1]))
                    state[key] = (f0, on)
            for (r_, fam, but), (f0, on) in state.items():
                out.append((f0, L, r_, fam, but, on))
        rows = []
        for a, b, r_, fam, but, on in out:
            rows += [(f, r_, fam, but, int(on)) for f in range(a, min(b, L))]
        return np.array(rows, np.int64).reshape(-1, 5)

    def _effects(self, t0, L, swap, step=1):
        """Both player views' effects, side by side with no merging, since each view has
        only some: (frame, type, owner index, x, y, radius, age in loops, view index)."""
        rows = []
        for s in (1, 2):
            d = self.rest[s]
            st = d["e_frame_start"]
            if step == 1:
                a, b = st[t0], st[t0 + L]
                if a == b:
                    continue
                fr = np.searchsorted(st, np.arange(a, b), side="right") - 1 - t0
                sel_idx = slice(a, b)
            else:
                chunks, fr_chunks = [], []
                for f in range(L):
                    rf = t0 + f * step
                    if rf + 1 < len(st) and st[rf + 1] > st[rf]:
                        c_idx = np.arange(st[rf], st[rf + 1])
                        chunks.append(c_idx)
                        fr_chunks.append(np.full(len(c_idx), f, dtype=np.int64))
                if not chunks:
                    continue
                sel_idx = np.concatenate(chunks)
                fr = np.concatenate(fr_chunks)
            if s not in self._eff_age:
                self._eff_age[s] = effect_ages(d)
            age = self._eff_age[s][sel_idx]
            own = d["e_owner"][sel_idx].astype(np.int64)
            own = np.where(
                own == 1,
                1 if swap else 0,
                np.where(own == 2, 0 if swap else 1, NEUTRAL),
            )
            vw = np.full(len(fr), (2 - s) if swap else (s - 1), dtype=np.int64)
            rows.append(
                np.stack(
                    [
                        fr,
                        np.clip(d["e_type"][sel_idx], 0, N_EFFECT_IDS - 1),
                        own,
                        d["e_x"][sel_idx] - self.center[0],
                        d["e_y"][sel_idx] - self.center[1],
                        d["e_radius"][sel_idx],
                        age,
                        vw,
                    ],
                    1,
                )
            )
        return (
            np.concatenate(rows).astype(np.float32)
            if rows
            else np.zeros((0, 8), np.float32)
        )

    def _players(self, t0, L, swap, step=1):
        """[L, 2, C_PLY]: each player's PC_KEYS numbers, camera, and supply left (cap -
        used, / 16, below 0 when over the cap): bfloat16 resolves it finely near 0, where
        being blocked is, and the log of supply used resolves only 1.5 at 150. From
        their own view."""
        out = np.zeros((L, 2, len(PC_KEYS) + 4), np.float32)
        t_end = t0 + (L - 1) * step
        for pi, s in enumerate((1, 2)):
            at = 1 - pi if swap else pi
            _, fr = self.views[s]
            if step == 1:
                for i in range(t0 // BATCH, (t0 + L - 1) // BATCH + 1):
                    b = fr.get_batch(i)
                    base = i * BATCH
                    lo, hi = max(t0, base), min(t0 + L, base + b.num_rows)
                    sl = np.arange(lo - base, hi - base)
                    for j, k in enumerate(PC_KEYS):
                        v = (
                            b.column(f"pc_{k}")
                            .to_numpy(zero_copy_only=False)[sl]
                            .astype(np.float32)
                        )
                        out[lo - t0 : hi - t0, at, j] = np.log1p(np.maximum(v, 0)) / 8
                        if k == "food_cap":
                            cap = v
                        if k == "food_used":
                            out[lo - t0 : hi - t0, at, len(PC_KEYS) + 3] = (cap - v) / 16
                    for j, k in enumerate(("cam_x", "cam_y")):
                        v = (
                            b.column(k)
                            .to_numpy(zero_copy_only=False)[sl]
                            .astype(np.float32)
                        )
                        ok = np.isfinite(v)
                        out[lo - t0 : hi - t0, at, len(PC_KEYS) + j] = np.where(
                            ok, (np.nan_to_num(v) - self.center[j]) / self.half[j], 0
                        )
                        out[lo - t0 : hi - t0, at, len(PC_KEYS) + 2] = ok
            else:
                raw_frames = t0 + np.arange(L) * step
                for i in range(t0 // BATCH, t_end // BATCH + 1):
                    b = fr.get_batch(i)
                    base = i * BATCH
                    in_b = (raw_frames >= base) & (raw_frames < base + b.num_rows)
                    if not in_b.any():
                        continue
                    f_idx = np.nonzero(in_b)[0]
                    sl = raw_frames[f_idx] - base
                    for j, k in enumerate(PC_KEYS):
                        v = (
                            b.column(f"pc_{k}")
                            .to_numpy(zero_copy_only=False)[sl]
                            .astype(np.float32)
                        )
                        out[f_idx, at, j] = np.log1p(np.maximum(v, 0)) / 8
                        if k == "food_cap":
                            cap = v
                        if k == "food_used":
                            out[f_idx, at, len(PC_KEYS) + 3] = (cap - v) / 16
                    for j, k in enumerate(("cam_x", "cam_y")):
                        v = (
                            b.column(k)
                            .to_numpy(zero_copy_only=False)[sl]
                            .astype(np.float32)
                        )
                        ok = np.isfinite(v)
                        out[f_idx, at, len(PC_KEYS) + j] = np.where(
                            ok, (np.nan_to_num(v) - self.center[j]) / self.half[j], 0
                        )
                        out[f_idx, at, len(PC_KEYS) + 2] = ok
        return out

    def _load_actions(self, avocab):
        """The game's action tokens, once. The tags of both views' rows stand in for one
        id space; a neutral's tags all stand for its one row."""
        tags, key = [], []
        for s in VIEWS:
            ut = self.rest[s]["unit_tag"]
            for i in self.index[s]["id"].tolist():
                tags.append(int(ut[i]))
                key.append((s, i))
        for s in VIEWS:
            ut = self.rest[s]["unit_tag"]
            for i in np.nonzero(self.nid[s] >= 0)[0].tolist():
                tags.append(int(ut[i]))
                key.append((0, int(self.nid[s][i])))
        stub = types.SimpleNamespace(meta=self.meta, unit_tag=tags, vocab=self.vocab)
        self.acts = GameActions(self.base + ".acts.npz", stub, avocab)
        self.act_key = key
        # the engine's executed commands, joined to tokens: token -> the units that
        # carried it out, as row keys (view, unit id). Each player's from its own view
        self.recip = {}
        for s in (1, 2):
            pov = self.rest[s]
            if "cmd_loop" not in pov.files:
                continue
            ev, _ = command_join.match(self.acts.raw, s, pov)
            cs, cu = pov["cmd_start"], pov["cmd_unit"]
            for c in np.nonzero(ev >= 0)[0].tolist():
                t = int(self.acts.tok_of_raw[ev[c]])
                if t >= 0 and cs[c + 1] > cs[c]:
                    self.recip.setdefault(t, set()).update(
                        (s, int(u)) for u in cu[cs[c] : cs[c + 1]].tolist()
                    )
        # autocast: the observation never shows it, so each toggle's state is remembered
        # per unit: (loop, on, family, button, unit ids) per view
        table = avocab.ruleset_table(self.meta)
        links = {int(k): v for k, v in table["links"].items()}
        self.autocast = {}
        for s in (1, 2):
            d = self.rest[s]
            ev = []
            if "ac_loop" in d.files:
                st = np.append(d["ac_start"], len(d["ac_unit"]))
                for i in range(len(d["ac_loop"])):
                    fam = avocab.family.get(links.get(int(d["ac_link"][i])), 1)
                    ev.append(
                        (
                            int(d["ac_loop"][i]),
                            bool(d["ac_on"][i]),
                            fam,
                            int(d["ac_cmd"][i]) + 1,
                            d["ac_unit"][st[i] : st[i + 1]].astype(np.int64),
                        )
                    )
            self.autocast[s] = ev

    def _actions(self, t0, L, row_of, swap, step=1):
        """Tokens in frames [t0, t0 + L * step): an action at loop l is binned into the
        strided frame `(l - (t0 + 1)) // step` so it reads the state at `t0 + f * step`
        and drives the transition to `t0 + (f + 1) * step`."""
        tk = self.acts.tok
        lo, hi = t0 + 1, t0 + 1 + L * step
        idx = np.nonzero((tk["loop"] >= lo) & (tk["loop"] < hi) & (tk["kind"] > 0))[0]
        n = len(idx)
        xy = np.stack([tk["x"][idx], tk["y"][idx]], 1)
        placed = np.isfinite(xy).all(1)
        target = np.full(n, -1, np.int64)
        uid = tk["uid"][idx]
        xy = np.nan_to_num(xy)
        for i in np.nonzero(uid >= 0)[0].tolist():
            target[i] = row_of.get(self.act_key[int(uid[i])], -1)
        # a selection's added units, and a command's recipients: (token, row) pairs
        sel, recip = [], []
        a0, a1 = tk["sel_start"][idx], self.acts.sel_end[idx]
        for i in np.nonzero(a1 > a0)[0].tolist():
            for u in self.acts.sel_uid[a0[i] : a1[i]].tolist():
                r_ = row_of.get(self.act_key[u], -1) if u >= 0 else -1
                if r_ >= 0:
                    sel.append((i, r_))
        for i, t in enumerate(idx.tolist()):
            for key in self.recip.get(t, ()):
                r_ = row_of.get(key, -1)
                if r_ >= 0:
                    recip.append((i, r_))
        flip = lambda p: (2 - p if swap else p - 1)
        num = np.stack(
            [
                tk["queued"][idx],
                tk["smart"][idx],
                tk["minimap"][idx],
                np.log1p(np.maximum(tk["presses"][idx] - 1.0, 0)) / 2,
                np.log1p(tk["n_add"][idx].clip(0)) / 3,
                np.log1p(tk["n_rem"][idx].clip(0)) / 3,
                np.log1p(tk["minerals"][idx].clip(0)) / 8,
                np.log1p(tk["gas"][idx].clip(0)) / 8,
                tk["recipient"][idx] >= 0,
                target >= 0,
                (uid >= 0) & (target < 0),
                placed,
            ],
            1,
        ).astype(np.float32)
        return dict(
            frame=np.clip((tk["loop"][idx] - lo) // step, 0, L - 1).astype(np.int64),
            player=flip(tk["player"][idx].astype(np.int64)).clip(0, 1),
            kind=tk["kind"][idx].astype(np.int64),
            family=tk["family"][idx].astype(np.int64),
            button=tk["button"][idx].astype(np.int64),
            tkind=tk["tkind"][idx].clip(0).astype(np.int64),
            group=tk["group"][idx].clip(0).astype(np.int64),
            gop=tk["gop"][idx].clip(0).astype(np.int64),
            remove=tk["remove"][idx].clip(0).astype(np.int64),
            subgroup=tk["subgroup"][idx].clip(0).astype(np.int64),
            num=num,
            xy=np.where(placed[:, None], xy - self.center, 0).astype(np.float32),
            placed=placed,
            target=target,
            sel=np.array(sel, np.int64).reshape(-1, 2),
            recip=np.array(recip, np.int64).reshape(-1, 2),
        )  # no place: the center, like the globals

    def _death(self, s, k, t0, L, step=1):
        d = int(self.index[s]["death"][k])
        t_end = t0 + (L - 1) * step
        if step == 1:
            return d - t0 if t0 <= d < t0 + L else -1
        return (d - t0 + step - 1) // step if t0 <= d <= t_end else -1

    def _fill(self, out, s, t0, L, row_of, step=1):
        """Writes view s's rows of the window into the grid, then bridges passengers and
        gas workers."""
        if s == 0:
            return self._fill_neutrals(out, t0, L, row_of, step=step)
        t_end = t0 + (L - 1) * step
        lo, hi = max(0, t0 - MARGIN), min(self.F, t_end + 1 + MARGIN)
        U, Ls = self._read(s, lo, hi)
        if not U:
            return
        u = {k: np.concatenate([x[k] for x in U]) for k in U[0]}
        frame, ids = u["frame"], u["u_id"]
        inw = (frame >= t0) & (frame <= t_end)
        if step > 1:
            inw = inw & (((frame - t0) % step) == 0)
        fo = (frame[inw] - t0) // step
        u_map = self._uid_row.get(s) if self._uid_row is not None else None
        if u_map is not None:
            raw_ids = ids[inw]
            ok_id = (raw_ids >= 0) & (raw_ids < len(u_map))
            ro = np.full(len(raw_ids), -1, np.int64)
            ro[ok_id] = u_map[raw_ids[ok_id]]
        else:
            ro = np.array(
                [row_of.get((s, int(i)), -1) for i in ids[inw].tolist()], np.int64
            )
        ok = ro >= 0
        fo, ro = fo[ok], ro[ok]
        src = np.nonzero(inw)[0][ok]
        _set(out, u, src, fo, ro, self)
        out["engaged_row"][fo, ro] = self._row_of_id(s, u["u_engaged"][src])
        off, bags = 0, []
        for (keep, l), x in zip(Ls, U):
            idx = np.nonzero(keep)[0]
            bags.append((off, idx, l))
            off += len(idx)
        self._bags(out, bags, src, fo, ro, s)
        if s:
            self._groups(out, s, t0, L, ids, frame, row_of, step=step)
            self._contained(out, s, t0, L, row_of, u, bags, step=step)

    def _fill_neutrals(self, out, t0, L, row_of, step=1):
        """Neutrals from both views: a snapshot first, then what a player sees over it,
        once. Contents as last seen."""
        t_end = t0 + (L - 1) * step
        ix = self.index[0]
        nid_row = np.full(len(ix["id"]), -1, np.int64)
        for (s, k), r_ in row_of.items():
            if s == 0:
                nid_row[k] = r_
        seen = np.zeros(out["mask"].shape, bool)
        neut_cache = {}
        for s in VIEWS:
            U, Ls = self._read(s, t0, t_end + 1, neutral=True)
            if U:
                neut_cache[s] = (
                    U,
                    Ls,
                    {k: np.concatenate([x[k] for x in U]) for k in U[0]},
                )
        for visible in (False, True):
            for s in VIEWS:
                if s not in neut_cache:
                    continue
                U, Ls, u = neut_cache[s]
                n = self.nid[s][u["u_id"]]
                on_stride = (((u["frame"] - t0) % step) == 0) if step > 1 else True
                fo = np.clip((u["frame"] - t0) // step, 0, L - 1)
                ro = nid_row[n]
                d = ix["death"][n]
                ok = (
                    on_stride
                    & (ro >= 0)
                    & ((d < 0) | (u["frame"] <= d))
                    & ((u["u_display"] != SNAPSHOT) == visible)
                    & ~seen[fo, np.maximum(ro, 0)]
                )
                src = np.nonzero(ok)[0]
                if not len(src):
                    continue
                _set(out, u, src, fo[src], ro[src], self)
                if not visible:
                    continue
                seen[fo[src], ro[src]] = True
                out["engaged_row"][fo[src], ro[src]] = self._row_of_id(
                    s, u["u_engaged"][src]
                )
                off, bags = 0, []
                for (keep, l), x in zip(Ls, U):
                    idx = np.nonzero(keep)[0]
                    bags.append((off, idx, l))
                    off += len(idx)
                self._bags(out, bags, src, fo[src], ro[src], s)
        start, log_f, log_c = self.nlog
        at = RAW_NUMS.index("contents")
        frames = t0 + np.arange(L) * step
        for k in np.nonzero(nid_row >= 0)[0].tolist():
            a, b = int(start[k]), int(start[k + 1])
            if a == b:
                continue
            j = np.searchsorted(log_f[a:b], frames, side="right") - 1
            r_ = nid_row[k]
            m = out["mask"][:, r_]
            out["num"][m, r_, at] = log_c[a:b][np.maximum(j, 0)][m]

    def _row_of_id(self, s, ids):
        """View s's unit ids (-1: none) -> rows of this window (-1: no row)."""
        ids = np.asarray(ids, np.int64)
        tf = self.to_flat[s]
        ok = (ids >= 0) & (ids < len(tf))
        flat = np.full(len(ids), self.n_flat, np.int64)
        flat[ok] = tf[ids[ok]]
        flat[flat < 0] = self.n_flat
        return self.flat_row[flat]

    def _groups(self, out, s, t0, L, ids, frame, row_of, step=1):
        g = self.grp.get(s)
        if g is None:
            return
        start, gid = g
        P = len(self.rest[s]["grp_player"])
        u_map = self._uid_row.get(s) if self._uid_row is not None else None
        for f in range(L):
            rf = t0 + f * step
            base = (rf * P) * C_GROUPS
            if base + C_GROUPS >= len(start):
                break
            if start[base] == start[base + C_GROUPS]:
                continue
            for k in range(C_GROUPS):
                a, b = start[base + k], start[base + k + 1]
                if a == b:
                    continue
                members = gid[a:b]
                if u_map is not None:
                    valid = (members >= 0) & (members < len(u_map))
                    ro = u_map[members[valid]]
                    ro = ro[ro >= 0]
                    if len(ro):
                        out["groups"][f, ro, k] = 1
                else:
                    for m in members.tolist():
                        r_ = row_of.get((s, m), -1)
                        if r_ >= 0:
                            out["groups"][f, r_, k] = 1

    def _bags(self, out, bags, src, fo, ro, s):
        """Orders, production, buffs and rally points of the window's rows, as flat
        lists: (frame, row, ...)."""
        pos = np.full(sum(len(b[1]) for b in bags), -1, np.int64)
        pos[src] = np.arange(len(src))
        for off, idx, l in bags:
            sel = np.nonzero(pos[off : off + len(idx)] >= 0)[0]
            if not len(sel):
                continue
            which = pos[off : off + len(idx)][sel]
            orig = idx[sel]
            for name, arrs in (
                ("ord", ("ord_abil", "ord_tx", "ord_ty", "ord_target", "ord_progress")),
                ("prod", ("prod_abil", "prod_progress")),
                ("buff", ("b_id",)),
                ("rally", ("rally_x", "rally_y", "rally_target")),
                ("ab", ("ab_id",)),
            ):
                offsets = np.asarray(l[arrs[0]].offsets)
                vals = [l[a].values.to_numpy(zero_copy_only=False) for a in arrs]
                n = offsets[orig + 1] - offsets[orig]
                if not n.sum():
                    continue
                it = np.concatenate(
                    [np.arange(a, b) for a, b in zip(offsets[orig], offsets[orig + 1])]
                )
                rep = np.repeat(which, n)
                seq = np.concatenate([np.arange(c) for c in n])
                st = [v[it].astype(np.float32) for v in vals]
                if name == "ord":  # order targets as rows
                    st[3] = self._row_of_id(s, vals[3][it]).astype(np.float32)
                elif name == "rally":
                    st[2] = self._row_of_id(s, vals[2][it]).astype(np.float32)
                out[f"{name}_where"].append(np.stack([fo[rep], ro[rep], seq], 1))
                out[f"{name}_vals"].append(np.stack(st, 1))

    def _contained(self, out, s, t0, L, row_of, u, bags, step=1):
        """Rows missing from their frame: passengers from their transport's list, gas
        workers from the order they vanished under."""
        order = np.lexsort((u["frame"], u["u_id"]))
        ids, fr = u["u_id"][order], u["frame"][order]
        cut = np.nonzero(np.diff(ids) == 0)[0]
        gaps = cut[(fr[cut + 1] - fr[cut]) > 1]
        for j in gaps.tolist():
            uid, a, b = int(ids[j]), int(fr[j]), int(fr[j + 1])
            r_ = row_of.get((s, uid), -1)
            if (
                r_ < 0
                or b - a > GAS_GAP_LOOPS
                or int(u["u_type"][order[j]]) not in WORKER_TYPES
            ):
                continue
            tgt = self._order_target(bags, order[j])
            if tgt is None:
                continue
            g = row_of.get((s, tgt), -1)
            if g < 0:
                continue
            if step == 1:
                ff_iter = [f - t0 for f in range(max(a + 1, t0), min(b, t0 + L))]
            else:
                ff_iter = [
                    ff for ff in range(L) if a < (t0 + ff * step) < b
                ]
            for ff in ff_iter:
                if out["mask"][ff, g] and int(out["type"][ff, g]) in GAS_TYPES:
                    (
                        out["mask"][ff, r_],
                        out["inside"][ff, r_],
                        out["parent_row"][ff, r_],
                    ) = (True, 2, g)
                    out["x"][ff, r_], out["y"][ff, r_] = (
                        out["x"][ff, g],
                        out["y"][ff, g],
                    )
                    out["type"][ff, r_] = u["u_type"][order[j]]
                    out["num"][ff, r_] = _carry(u, order[j])
        for off, idx, l in bags:
            offsets = np.asarray(l["pass_id"].offsets)
            if offsets[-1] == offsets[0]:
                continue
            vid, vty, vhp = (
                l[k].values.to_numpy(zero_copy_only=False)
                for k in ("pass_id", "pass_type", "pass_hp")
            )
            for local, i in enumerate(idx.tolist()):
                raw_f = int(u["frame"][off + local])
                if raw_f < t0 or (step > 1 and ((raw_f - t0) % step) != 0):
                    continue
                f = (raw_f - t0) // step
                if not 0 <= f < L:
                    continue
                for it in range(offsets[i], offsets[i + 1]):
                    r_, c = row_of.get((s, int(vid[it])), -1), row_of.get(
                        (s, int(u["u_id"][off + local])), -1
                    )
                    if r_ < 0 or c < 0 or out["mask"][f, r_]:
                        continue
                    (
                        out["mask"][f, r_],
                        out["inside"][f, r_],
                        out["parent_row"][f, r_],
                    ) = (True, 1, c)
                    out["x"][f, r_], out["y"][f, r_], out["type"][f, r_] = (
                        u["u_x"][off + local],
                        u["u_y"][off + local],
                        vty[it],
                    )
                    out["num"][f, r_, 0] = float(vhp[it])

    def _order_target(self, bags, row):
        for off, idx, l in bags:
            if off <= row < off + len(idx):
                i = idx[row - off]
                offsets = np.asarray(l["ord_target"].offsets)
                if offsets[i + 1] > offsets[i]:
                    t = int(
                        l["ord_target"].values.to_numpy(zero_copy_only=False)[
                            offsets[i]
                        ]
                    )
                    return t if t >= 0 else None
                return None
        return None

    def _dead_rows(self, out, rows, t0, L, step=1):
        """From its death frame, the first it is gone in, a row keeps its last values with
        hit points and shields 0, and isn't alive. A row still present at its death
        frame is dead from the next. Frames it was out of sight in before stay absent.
        """
        for r_, (s, k) in enumerate(rows):
            d = self._death(s, k, t0, L, step=step)
            died_before = 0 <= self.index[s]["death"][k] < t0
            if died_before or d < 0:
                continue
            live = np.nonzero(out["mask"][: d + 1, r_])[0]
            if not len(live):
                continue
            f0 = int(live[-1])
            d0 = max(d, f0 + 1)
            if d0 >= L:
                continue
            for key in (
                "x",
                "y",
                "type",
                "inside",
                "parent_row",
                "num",
                "groups",
                "id",
                "engaged_row",
            ):
                if key in out:
                    out[key][d0:, r_] = out[key][f0, r_]
            out["num"][d0:, r_, 0] = 0
            out["num"][d0:, r_, 1] = 0
            out["mask"][d0:, r_] = True
            out["alive"][d0:, r_] = False

    @staticmethod
    def _fb_rows(batch, name, j_sel, H):
        """Zero-copy slice (or strided gather) of a FixedSizeBinaryArray -> [n, H, rb]."""
        c = batch.column(name)
        bw = c.type.byte_width
        rb = bw // H
        if isinstance(j_sel, slice):
            j0, j1 = j_sel.start, j_sel.stop
            n = j1 - j0
            return np.frombuffer(
                c.buffers()[1], np.uint8, count=n * bw, offset=(c.offset + j0) * bw
            ).reshape(n, H, rb)
        all_rows = np.frombuffer(
            c.buffers()[1], np.uint8, count=len(c) * bw, offset=c.offset * bw
        ).reshape(len(c), H, rb)
        return all_rows[j_sel]

    def _map_frames(self, t0, L, swap, step=1, need_map=True, compact=False):
        """[L, TOP, TOP, C_DYNAMIC * PATCH * PATCH] uint8 (or bit-packed center [L, TOP, TOP]
        uint8 when compact=True): each player's visible, seen and creep, the next-frame
        targets; and [L, h, w, C_LIVE] uint8 (or bit-packed [L, h, w] int16 when
        compact=True), the playable area at its own size with each player's pathable and
        buildable too, the input."""
        x0, y0, x1, y1 = self.playable
        gh, gw = y1 - y0, x1 - x0
        bx0, bx1 = x0 // 8, (x1 + 7) // 8
        dx0 = x0 - bx0 * 8
        if compact:
            out = np.zeros((L, TOP, TOP), np.uint8) if need_map else None
            grid = np.zeros((L, gh, gw), np.int16)
            centre = (PATCH * PATCH) // 2
            yc, xc = self.cells[0][:, :, centre], self.cells[1][:, :, centre]
            xc_byte, xc_shift = xc >> 3, (7 - (xc & 7)).astype(np.uint8)
        else:
            out = (
                np.zeros((L, TOP, TOP, C_DYNAMIC * PATCH * PATCH), np.uint8)
                if need_map
                else None
            )
            grid = np.zeros((L, gh, gw, C_LIVE), np.uint8)
            y, x = self.cells
        if L <= 0:
            return out, grid
        t_end = t0 + (L - 1) * step
        raw_frames = None if step == 1 else (t0 + np.arange(L, dtype=np.int64) * step)
        for pi, s in enumerate((1, 2)):
            p_out = 1 - pi if swap else pi
            at = p_out * 3 * PATCH * PATCH
            g = p_out * 5
            _, fr = self.views[s]
            for bi in range(t0 // BATCH, t_end // BATCH + 1):
                b = fr.get_batch(bi)
                base = bi * BATCH
                if step == 1:
                    lo, hi = max(t0, base), min(t0 + L, base + b.num_rows)
                    if lo >= hi:
                        continue
                    f_sl = slice(lo - t0, hi - t0)
                    j_sel = slice(lo - base, hi - base)
                else:
                    in_b = (raw_frames >= base) & (raw_frames < base + b.num_rows)
                    if not in_b.any():
                        continue
                    f_sl = np.nonzero(in_b)[0]
                    j_sel = raw_frames[f_sl] - base
                vis = self._fb_rows(b, "vis", j_sel, self.H)
                creep_raw = self._fb_rows(b, "creep", j_sel, self.H)
                path_raw = self._fb_rows(b, "pathable", j_sel, self.H)
                build_raw = self._fb_rows(b, "buildable", j_sel, self.H)
                vis_crop = vis[:, y0:y1, x0:x1]
                creep_crop = np.unpackbits(
                    creep_raw[:, y0:y1, bx0:bx1], axis=2
                )[:, :, dx0 : dx0 + gw]
                path_crop = np.unpackbits(
                    path_raw[:, y0:y1, bx0:bx1], axis=2
                )[:, :, dx0 : dx0 + gw]
                build_crop = np.unpackbits(
                    build_raw[:, y0:y1, bx0:bx1], axis=2
                )[:, :, dx0 : dx0 + gw]
                if compact:
                    bits_g = (
                        (vis_crop == 2).astype(np.int16)
                        | ((vis_crop > 0).astype(np.int16) << 1)
                        | (creep_crop.astype(np.int16) << 2)
                        | (path_crop.astype(np.int16) << 3)
                        | (build_crop.astype(np.int16) << 4)
                    )
                    grid[f_sl] |= bits_g << g
                    if need_map:
                        vc = vis[:, yc, xc]
                        cc = (creep_raw[:, yc, xc_byte] >> xc_shift) & 1
                        bits_m = (
                            (vc == 2).astype(np.uint8)
                            | ((vc > 0).astype(np.uint8) << 1)
                            | (cc << 2)
                        )
                        out[f_sl] |= bits_m << (p_out * 3)
                else:
                    grid[f_sl, :, :, g] = vis_crop == 2
                    grid[f_sl, :, :, g + 1] = vis_crop > 0
                    grid[f_sl, :, :, g + 2] = creep_crop
                    grid[f_sl, :, :, g + 3] = path_crop
                    grid[f_sl, :, :, g + 4] = build_crop
                    if need_map:
                        creep_full = np.unpackbits(creep_raw, axis=2)[:, :, : self.W]
                        vs = vis[:, y, x]
                        out[f_sl, :, :, at : at + PATCH * PATCH] = vs == 2
                        out[
                            f_sl, :, :, at + PATCH * PATCH : at + 2 * PATCH * PATCH
                        ] = vs > 0
                        out[
                            f_sl, :, :, at + 2 * PATCH * PATCH : at + 3 * PATCH * PATCH
                        ] = creep_full[:, y, x]
        return out, grid


def _blank(L, R):
    out = dict(
        mask=np.zeros((L, R), bool),
        alive=np.ones((L, R), bool),
        inside=np.zeros((L, R), np.int8),
        parent_row=np.full((L, R), -1, np.int32),
        type=np.zeros((L, R), np.int16),
        x=np.zeros((L, R), np.float32),
        y=np.zeros((L, R), np.float32),
        id=np.full((L, R), -1, np.int32),
        num=np.zeros((L, R, len(RAW_NUMS)), np.float32),
        groups=np.zeros((L, R, C_GROUPS), np.int8),
    )
    out["engaged_row"] = np.full((L, R), -1, np.int32)
    for k in ("ord", "prod", "buff", "rally", "ab"):
        out[f"{k}_where"], out[f"{k}_vals"] = [], []
    return out


# the raw numbers each row carries, unscaled: the model embeds them, the raw heads
# predict them
RAW_NUMS = (
    "hp",
    "shield",
    "energy",
    "hp_max",
    "shield_max",
    "build",
    "contents",
    "harvesters",
    "ideal_harvesters",
    "facing",
    "radius",
    "weapon_cooldown",
    "cargo_used",
    "cargo_max",
    "buff_remain",
    "buff_max",
    "detect_range",
    "radar_range",
    "atk_up",
    "arm_up",
    "shd_up",
    "flags",
    "flags2",
    "display",
    "cloak",
    "add_on",
    "engaged",
    "z",
)


def _raw(u, i):
    return np.stack([u["u_" + k][i].astype(np.float32) for k in RAW_NUMS], -1)


def _carry(u, i):
    return _raw(u, i)


def _set(out, u, src, fo, ro, game):
    out["mask"][fo, ro] = True
    out["x"][fo, ro], out["y"][fo, ro] = u["u_x"][src], u["u_y"][src]
    out["type"][fo, ro] = u["u_type"][src]
    out["id"][fo, ro] = u["u_id"][src]
    out["num"][fo, ro] = _raw(u, src)


QUEUE, PROD = 8, 8  # order and production slots with targets (physics.RowHead)
N_CARGO = 16  # 1..N_CARGO: deterministic passenger slot inside a container, 0: outside


def effect_ages(d, near=1.5):
    """Each effect entry of a view's .cst -> loops since that effect began, counted over
    the whole game, so one already running when a window opens keeps its age. Effects
    have no id: an entry continues the previous frame's entry of the same type, owner
    and place, or else, for ones that move (guardian shield, the colossus beam), the
    nearest unmatched one of the same type and owner within `near` map units. It starts
    when seen first: an effect cast out of this view's sight is as old as its sighting.
    """
    st, T, O = d["e_frame_start"], d["e_type"], d["e_owner"]
    X, Y = d["e_x"].astype(np.float64), d["e_y"].astype(np.float64)
    age = np.zeros(len(T), np.int32)
    prev = []  # (type, owner, x, y, age) of the last frame
    for f in range(len(st) - 1):
        a, b = int(st[f]), int(st[f + 1])
        if a == b:
            prev = []
            continue
        exact = {}
        for j, p in enumerate(prev):
            exact.setdefault(p[:4], []).append(j)
        used, cur, loose = set(), [], []
        for i in range(a, b):
            k = (int(T[i]), int(O[i]), X[i], Y[i])
            js = exact.get(k)
            if js:
                j = js.pop()
                used.add(j)
                age[i] = prev[j][4] + 1
            else:
                loose.append(i)
        for i in loose:
            best, bd = -1, near
            for j, p in enumerate(prev):
                if j in used or p[0] != T[i] or p[1] != O[i]:
                    continue
                dd = abs(p[2] - X[i]) + abs(p[3] - Y[i])
                if dd <= bd:
                    best, bd = j, dd
            if best >= 0:
                used.add(best)
                age[i] = prev[best][4] + 1
        cur = [(int(T[i]), int(O[i]), X[i], Y[i], int(age[i])) for i in range(a, b)]
        prev = cur
    return age


FORCED = 16  # map tokens under effects always among each frame's decoded sample


def collate(windows, vocab, rng=None, drop_dense_slots=False):
    """Windows of the same length -> one batch of tensors: grids [B, F, R, ...], lists
    flat by (B * F + f) * R + r."""
    import torch

    B, F_ = len(windows), windows[0]["mask"].shape[0]
    R = max(w["mask"].shape[1] for w in windows)
    out = {}
    for k in (
        "mask",
        "alive",
        "inside",
        "parent_row",
        "engaged_row",
        "type",
        "x",
        "y",
        "num",
        "groups",
    ):
        a0 = windows[0][k]
        fill = True if k == "alive" else -1 if k in ("parent_row", "engaged_row") else 0
        buf = np.full((B, F_, R) + a0.shape[2:], fill, a0.dtype)
        for b, w in enumerate(windows):
            buf[b, :, : w[k].shape[1]] = w[k]
        out[k] = torch.from_numpy(buf)
    par = out["parent_row"].numpy()
    msk = out["mask"].numpy()
    cslot = np.zeros((B, F_, R), np.int64)
    pb, pf, pr = np.nonzero(msk & (par >= 0))
    if len(pb):
        keys = (pb * F_ + pf) * R + par[pb, pf, pr]
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        run_start = np.maximum.accumulate(
            np.where(np.r_[True, sk[1:] != sk[:-1]], np.arange(len(sk)), 0)
        )
        cslot[pb[order], pf[order], pr[order]] = np.minimum(
            np.arange(len(sk)) - run_start + 1, N_CARGO
        )
    out["cargo_slot"] = torch.from_numpy(cslot)
    out["rtype"] = out["type"].long()
    out["type"] = torch.from_numpy(
        vocab.unit_index(out["rtype"].numpy()).astype(np.int64)
    )
    own = np.full((B, R), NEUTRAL, np.int64)
    dframe = np.full((B, R), -1, np.int64)
    for b, w in enumerate(windows):
        own[b, : len(w["owner"])] = w["owner"]
        if "death_frame" in w:
            dframe[b, : len(w["death_frame"])] = w["death_frame"]
    out["owner"] = torch.from_numpy(own)
    out["death_frame"] = torch.from_numpy(dframe)
    if "ruleset_id" in windows[0]:
        out["ruleset_id"] = torch.from_numpy(
            np.stack([np.int64(w["ruleset_id"]) for w in windows])
        )
    for k in (
        "center",
        "half",
        "side",
        "players",
        "ply_num",
        "map",
        "origin",
        "map_id",
    ):
        if k in windows[0] and windows[0][k] is not None:
            out[k] = torch.from_numpy(np.stack([w[k] for w in windows]))
    # the map at its own size, zero past its edge: the known layer tells. Sizes are
    # multiples of 8, the tower's reduction
    gh, gw = (
        int(np.ceil(max(w["grid"].shape[i] for w in windows) / 8)) * 8 for i in (1, 2)
    )
    packed_grid = windows[0]["grid"].ndim == 3
    grid = (
        np.zeros((B, F_, gh, gw), np.int16)
        if packed_grid
        else np.zeros((B, F_, gh, gw, C_LIVE), np.uint8)
    )
    sgrid = np.zeros((B, gh, gw, C_STATIC), np.uint8)
    for b, w in enumerate(windows):
        h_, w_ = w["grid"].shape[1:3]
        grid[b, :, :h_, :w_], sgrid[b, :h_, :w_] = w["grid"], w["static_grid"]
    out["grid"], out["static_grid"] = torch.from_numpy(grid), torch.from_numpy(sgrid)
    out["view"] = torch.ones(B, 2)
    out["loops"] = torch.from_numpy(
        np.stack([np.asarray(w["loops"], np.int64) for w in windows])
    )  # [B, F]: each frame's game loop
    flat = lambda b, wh: (b * F_ + wh[:, 0]) * R + wh[:, 1]
    bags = {k: ([], []) for k in ("ord", "prod", "buff", "rally", "ab")}
    ord_row = np.full((B, F_, R, QUEUE), -1, np.int32)
    rally = np.zeros((B, F_, R, 4), np.float32)
    slot_abil = np.zeros(
        (B, F_, R, QUEUE + PROD), np.int32 if drop_dense_slots else np.int64
    )
    slot_reg = (
        None if drop_dense_slots else np.zeros((B, F_, R, QUEUE + PROD, 4), np.float32)
    )
    for b, w in enumerate(windows):
        for k in bags:
            if w[f"{k}_where"]:
                wh, v = np.concatenate(w[f"{k}_where"]), np.concatenate(w[f"{k}_vals"])
                bags[k][0].append((flat(b, wh), wh, v))
    cat = lambda k: (
        [np.concatenate(z) for z in zip(*bags[k][0])]
        if bags[k][0]
        else (
            np.zeros(0, np.int64),
            np.zeros((0, 3), np.int64),
            np.zeros((0, 5), np.float32),
        )
    )
    bh = lambda at: at // (F_ * R)
    at, wh, v = cat("ord")
    ab = (
        vocab.ability_index(v[:, 0].astype(np.int64)).astype(np.int64)
        if len(v)
        else np.zeros(0, np.int64)
    )
    if len(at):
        bb = bh(at)
        ux, uy = (
            out["x"].numpy()[bb, wh[:, 0], wh[:, 1]],
            out["y"].numpy()[bb, wh[:, 0], wh[:, 1]],
        )
        hx, hy = out["half"].numpy()[bb, 0], out["half"].numpy()[bb, 1]
        dx, dy = (
            np.where(np.isfinite(v[:, 1]), np.nan_to_num(v[:, 1]) - ux, 0) / hx,
            np.where(np.isfinite(v[:, 2]), np.nan_to_num(v[:, 2]) - uy, 0) / hy,
        )
        feat = np.stack(
            [np.clip(dx, -2, 2), np.clip(dy, -2, 2), v[:, 3] >= 0, v[:, 4]], 1
        ).astype(
            np.float32
        )  # has a target: as a row of this window
        q = wh[:, 2] < QUEUE
        slot_abil[bb[q], wh[q, 0], wh[q, 1], wh[q, 2]] = ab[q]
        if slot_reg is not None:
            slot_reg[bb[q], wh[q, 0], wh[q, 1], wh[q, 2]] = feat[q]
        ord_row[bb[q], wh[q, 0], wh[q, 1], wh[q, 2]] = v[q, 3]
    else:
        feat = np.zeros((0, 4), np.float32)
        q = np.zeros(0, bool)
    out["ord_at"], out["ord_abil"], out["ord_feat"] = (
        torch.from_numpy(at.astype(np.int64)),
        torch.from_numpy(ab),
        torch.from_numpy(feat),
    )
    out["ord_in_q"] = torch.from_numpy(q)
    out["ord_slot"] = torch.from_numpy(
        np.minimum(wh[:, 2], QUEUE - 1).astype(np.int64)
    )  # 0: what the unit is doing now
    out["ord_trow"] = torch.from_numpy(
        v[:, 3].astype(np.int64) if len(v) else np.zeros(0, np.int64)
    )  # the target's row in this window, -1: none
    at, wh, v = cat("prod")
    ab = (
        vocab.ability_index(v[:, 0].astype(np.int64)).astype(np.int64)
        if len(v)
        else np.zeros(0, np.int64)
    )
    feat = (
        np.stack([v[:, 1], wh[:, 2] / 5], 1).astype(np.float32)
        if len(v)
        else np.zeros((0, 2), np.float32)
    )
    if len(at):
        bb = bh(at)
        q = wh[:, 2] < PROD
        slot_abil[bb[q], wh[q, 0], wh[q, 1], QUEUE + wh[q, 2]] = ab[q]
        if slot_reg is not None:
            slot_reg[bb[q], wh[q, 0], wh[q, 1], QUEUE + wh[q, 2], 3] = v[q, 1]
    else:
        q = np.zeros(0, bool)
    out["prod_at"], out["prod_abil"], out["prod_feat"] = (
        torch.from_numpy(at.astype(np.int64)),
        torch.from_numpy(ab),
        torch.from_numpy(feat),
    )
    out["prod_in_q"] = torch.from_numpy(q)
    out["prod_slot"] = torch.from_numpy(np.minimum(wh[:, 2], PROD - 1).astype(np.int64))
    at, wh, v = cat("buff")
    out["buff_at"], out["buff_id"] = torch.from_numpy(
        at.astype(np.int64)
    ), torch.from_numpy(
        np.clip(v[:, 0], 0, 511).astype(np.int64) if len(v) else np.zeros(0, np.int64)
    )
    at, wh, v = cat("rally")
    if len(at):
        bb = bh(at)
        ux, uy = (
            out["x"].numpy()[bb, wh[:, 0], wh[:, 1]],
            out["y"].numpy()[bb, wh[:, 0], wh[:, 1]],
        )
        hx, hy = out["half"].numpy()[bb, 0], out["half"].numpy()[bb, 1]
        feat = np.stack(
            [
                np.clip((v[:, 0] - ux) / hx, -2, 2),
                np.clip((v[:, 1] - uy) / hy, -2, 2),
                v[:, 2] >= 0,
            ],
            1,
        ).astype(np.float32)
        first = wh[:, 2] == 0
        rally[bb[first], wh[first, 0], wh[first, 1]] = np.concatenate(
            [np.ones((first.sum(), 1), np.float32), feat[first]], 1
        )
    else:
        feat = np.zeros((0, 3), np.float32)
    out["rally_at"], out["rally_feat"] = torch.from_numpy(
        at.astype(np.int64)
    ), torch.from_numpy(feat)
    out["rally_first"], out["ord_row"] = torch.from_numpy(rally), torch.from_numpy(
        ord_row
    )
    at, wh, v = cat("ab")
    out["ab_at"], out["ab_id"] = torch.from_numpy(
        at.astype(np.int64)
    ), torch.from_numpy(
        np.clip(v[:, 0], 0, N_RAW_ABILITY_IDS - 1).astype(np.int64)
        if len(v)
        else np.zeros(0, np.int64)
    )
    if not drop_dense_slots:
        out["slot_abil"], out["slot_reg"] = torch.from_numpy(
            slot_abil
        ), torch.from_numpy(slot_reg)
    q_ch = (slot_abil[:, 1:] != slot_abil[:, :-1]).any(-1)
    eng_np = out["engaged_row"].numpy()
    eng_ch = eng_np[:, 1:] != eng_np[:, :-1]
    ab_hash = np.zeros(B * F_ * R, np.uint64)
    if len(at):
        ab_vals = np.clip(v[:, 0], 0, N_RAW_ABILITY_IDS - 1).astype(np.uint64) + 1
        ab_mix = (ab_vals * np.uint64(0x9E3779B97F4A7C15)) ^ (
            ab_vals * np.uint64(0xBF58476D1CE4E5B9)
        )
        np.add.at(ab_hash, at, ab_mix)
    ab_hash = ab_hash.reshape(B, F_, R)
    ab_ch = ab_hash[:, 1:] != ab_hash[:, :-1]
    out["queue_changed"] = torch.from_numpy(q_ch)
    out["row_changed"] = torch.from_numpy(q_ch | eng_ch | ab_ch)
    # births: first seen in frame f + 1 means the count of frame f
    bb, ff, rr, pp = [], [], [], []
    count = np.zeros((B, F_ - 1), np.int64)
    for b, w in enumerate(windows):
        bb += [b] * len(w["birth_row"])
        ff += w["birth_frame"].tolist()
        rr += w["birth_row"].tolist()
        pp += w["birth_parent"].tolist()
        np.add.at(count[b], w["birth_frame"] - 1, 1)
    out["birth_count"] = torch.from_numpy(count)
    out["birth_b"], out["birth_f"], out["birth_r"], out["birth_p"] = (
        torch.tensor(x, dtype=torch.int64) for x in (bb, ff, rr, pp)
    )
    # actions: each frame has K slots per player, player 0's then player 1's, in the
    # order given; a player's timeline is its slots frame after frame
    K = 1
    for w in windows:
        if len(w["actions"]["frame"]):
            K = max(
                K,
                int(
                    np.bincount(
                        w["actions"]["frame"] * 2
                        + w["actions"]["player"].astype(np.int64)
                    ).max()
                ),
            )
    A = 2 * K
    shape = (B, F_, A)
    amask = np.zeros(shape, bool)
    keys = dict(
        kind=np.int64,
        family=np.int64,
        button=np.int64,
        tkind=np.int64,
        group=np.int64,
        gop=np.int64,
        remove=np.int64,
        subgroup=np.int64,
        player=np.int64,
    )
    ac = {k: np.zeros(shape, t) for k, t in keys.items()}
    anum, axy = np.zeros(
        shape + (windows[0]["actions"]["num"].shape[1],), np.float32
    ), np.zeros(shape + (2,), np.float32)
    placed, target = np.zeros(shape, bool), np.full(shape, -1, np.int64)
    sel_pairs, recip_pairs = [], []
    for b, w in enumerate(windows):
        a = w["actions"]
        seen = np.zeros((F_, 2), np.int64)
        slot = np.zeros(len(a["frame"]), np.int64)
        for i, f in enumerate(a["frame"].tolist()):
            p = int(a["player"][i])
            s = p * K + seen[f, p]
            slot[i] = s
            seen[f, p] += 1
            amask[b, f, s] = True
            for k in keys:
                ac[k][b, f, s] = a[k][i]
            anum[b, f, s], axy[b, f, s], placed[b, f, s] = (
                a["num"][i],
                a["xy"][i],
                a["placed"][i],
            )
            target[b, f, s] = a["target"][i]
        # (frame of the batch, slot, row)
        fr = a["frame"].astype(np.int64)
        for k_, dst in (("sel", sel_pairs), ("recip", recip_pairs)):
            pr = a.get(k_, np.zeros((0, 2), np.int64))
            if len(pr):
                i_ = pr[:, 0]
                dst.append(np.stack([b * F_ + fr[i_], slot[i_], pr[:, 1]], 1))
    for k in keys:
        out[f"act_{k}"] = torch.from_numpy(ac[k])
    (
        out["act_mask"],
        out["act_num"],
        out["act_xy"],
        out["act_placed"],
        out["act_target"],
    ) = (torch.from_numpy(x) for x in (amask, anum, axy, placed, target))
    for k_, pr in (("act_sel", sel_pairs), ("act_recip", recip_pairs)):
        out[k_] = torch.from_numpy(
            np.concatenate(pr) if pr else np.zeros((0, 3), np.int64)
        )
    au = [
        (b * F_ + w["autocast"][:, 0]) * R + w["autocast"][:, 1]
        for b, w in enumerate(windows)
    ]
    av = np.concatenate([w["autocast"] for w in windows])
    out["auto_at"], out["auto_fam"], out["auto_but"], out["auto_on"] = torch.from_numpy(
        np.concatenate(au).astype(np.int64)
    ), *(torch.from_numpy(av[:, k]) for k in (2, 3, 4))
    # effects, per frame: tokens, and the map tokens each covers in the next frame
    X = max(
        1,
        max(
            (
                np.bincount(w["effects"][:, 0].astype(np.int64), minlength=1).max()
                if len(w["effects"])
                else 0
            )
            for w in windows
        ),
    )
    emask, etype, eown = (
        np.zeros((B, F_, X), bool),
        np.zeros((B, F_, X), np.int64),
        np.full((B, F_, X), NEUTRAL, np.int64),
    )
    eview = np.zeros((B, F_, X), np.int64)
    exy, erad = np.zeros((B, F_, X, 2), np.float32), np.zeros((B, F_, X), np.float32)
    eage = np.zeros((B, F_, X), np.float32)
    for b, w in enumerate(windows):
        eff_arr = w["effects"]
        if not len(eff_arr):
            continue
        seen = np.zeros(F_, np.int64)
        for row in eff_arr.tolist():
            f, ty, o, x, y, r, age = row[:7]
            vw = int(row[7]) if len(row) > 7 else 0
            f, ty, o = int(f), int(ty), int(o)
            i = seen[f]
            seen[f] += 1
            (
                emask[b, f, i],
                etype[b, f, i],
                eown[b, f, i],
                eview[b, f, i],
                exy[b, f, i],
                erad[b, f, i],
                eage[b, f, i],
            ) = (True, ty, o, vw, (x, y), r / 8, age)
    (
        out["eff_mask"],
        out["eff_type"],
        out["eff_owner"],
        out["eff_view"],
        out["eff_xy"],
        out["eff_radius"],
        out["eff_age"],
    ) = (torch.from_numpy(x) for x in (emask, etype, eown, eview, exy, erad, eage))
    return out


BATCH_KEYS = (
    "mask",
    "alive",
    "inside",
    "x",
    "y",
    "num",
    "groups",
    "rtype",
    "owner",
    "players",
    "view",
    "ply_num",
    "slot_abil",
    "slot_reg",
    "queue_changed",
    "center",
    "half",
    "side",
    "origin",
    "grid",
    "static_grid",
    "loops",
    "eff_mask",
    "eff_type",
    "eff_owner",
    "eff_xy",
    "eff_radius",
    "eff_age",
    "act_mask",
    "act_player",
    "act_kind",
    "act_group",
    "act_gop",
    "act_num",
    "act_xy",
    "act_placed",
    "act_target",
    "death_frame",
    "parent_row",
    "ruleset_id",
    "birth_count",
    "birth_b",
    "birth_f",
    "birth_r",
    "birth_p",
)


def trim_batch(b: dict) -> dict:
    """Keep only `BATCH_KEYS` and slice `slot_reg`/`slot_abil` to active L1/L2 columns."""
    out = {k: b[k] for k in BATCH_KEYS if k in b}
    if "slot_reg" in out:
        out["slot_reg"] = out["slot_reg"][..., :1, :].contiguous()
    if "slot_abil" in out:
        out["slot_abil"] = out["slot_abil"][..., : QUEUE + 1].contiguous()
    return out


def to_device(b: dict, device) -> dict:
    """Transfer `BATCH_KEYS` tensors in `b` to `device` (`non_blocking=True`)."""
    return {k: b[k].to(device, non_blocking=True) for k in BATCH_KEYS if k in b}
