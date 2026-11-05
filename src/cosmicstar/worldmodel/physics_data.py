"""The physics model's training store (DESIGN_physics.md): any window of a game read straight from its Arrow views.

A game is two views, p1 and p2, each with that player's own units and the neutrals as that player knows them. Players are indexed 0,
1 (neutral 2). An own row comes from its player's view. A neutral row joins both views: it keeps one row by its type and place, since a
neutral out of sight is a snapshot under another tag; critters go by tag and exist only while seen. <game>.index.npz (build_index, once
per game) holds what a window can't work out on its own: each row's first and last frame, its death frame and its parent, and each
neutral's contents as last seen, since a snapshot reports none. Contents change only in a player's view, so before they are first seen
they are what they are when they are.

A sequence is a grid, frames x rows, with a unit on the same row throughout:

    mask       the row has a token in this frame. False only before birth, or while a unit is out of sight unexplained
    alive      false from the frame after its death: the row keeps its last values, hit points and shields 0
    inside     0 on the map, 1 passenger, 2 in a gas building (bridged as tokenize._gas_gaps does), with parent_row

Targets are the same columns one frame later. Births are rows whose first frame in the whole game is inside the window."""

import glob
import json
import os
import types

import numpy as np

from . import arrow_state, statefile
from .action_tokens import GameActions
from .tokenize import N_EFFECT_IDS, N_RAW_ABILITY_IDS, PC_KEYS, C_GROUPS, GAS_GAP_LOOPS, GAS_TYPES as _GAS, NEUTRAL_PLAYER, WORKER_TYPES

GAS_TYPES = (*_GAS, 1955, 1956, 1960)     # the rich gas buildings of these patches too

VIEWS = (1, 2)                   # the views' file suffixes: player slots
SOURCES = (1, 2, 0)              # where rows come from: each player's own, then the neutrals of both views
MOVING_NEUTRALS = (321, 322, 335, 336, 337, 612, 659, 660, 661)   # neutral types seen moving in 30 games: critters; data, not names
NEUTRAL = 2                      # owner index of neutral rows
BATCH = 64                       # frames per Arrow batch (extract_states.ARROW_FRAMES)
MARGIN = GAS_GAP_LOOPS + 1       # frames read either side of a window, to bridge gaps across its edges
PATCH = 3                        # real cells per side each 64-level map token reads
TOP = 64                         # map tokens per side
C_DYNAMIC = 6                    # map layers decoded: (visible, seen, creep) per player
C_LIVE, C_STATIC = 10, 4         # map layers read: (visible, seen, creep, pathable, buildable) per player; height per player, playable, known
MAX_ROWS = 4096
SNAPSHOT = 2                     # u_display


def base_paths(root, split_ids):
    """Game id -> its base path (no view suffix), for ids with both views."""
    out = {}
    for u in glob.glob(os.path.join(root, "1v1", "*", "*", "*.p1.units.arrow")):
        base = u[:-len(".p1.units.arrow")]
        gid = os.path.basename(base)
        if gid in split_ids and all(os.path.exists(f"{base}.p{s}.units.arrow") for s in VIEWS):
            out[gid] = base
    return out


def build_index(base, replay_path=None):
    """One game's <base>.index.npz: for each view, row ids with first frame, last frame, death frame (-1: never) and parent id
    (-1: none), and frames of the game. Parents come from the replay's creator tags, so own-view ids only."""
    out = {}
    for s in VIEWS:
        r, fr = arrow_state.open_view(f"{base}.p{s}")
        first, last = {}, {}
        for i in range(r.num_record_batches):
            b = r.get_batch(i)
            ids, frame = b.column("u_id").to_numpy(zero_copy_only=False), b.column("frame").to_numpy(zero_copy_only=False)
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
            for k in di[ds[f]:ds[f + 1]].tolist():
                if k in pos:
                    death[pos[k]] = f
        parent = np.full(len(ids), -1, np.int32)
        if s and replay_path:
            parent = _parents(replay_path, d["unit_tag"], ids)
        out[f"id{s}"], out[f"first{s}"] = ids, np.array([first[k] for k in ids.tolist()], np.int32)
        out[f"last{s}"], out[f"death{s}"], out[f"parent{s}"] = np.array([last[k] for k in ids.tolist()], np.int32), death, parent
        out["F"] = np.int32(sum(fr.get_batch(i).num_rows for i in range(fr.num_record_batches)))
    out.update(_neutral_index(base))
    tmp = f"{base}.index.tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, f"{base}.index.npz")
    return int(out["F"])


def _neutral_index(base):
    """The neutrals' rows over both views: each view's unit id -> a neutral id (nid<view>, -1: not neutral), each neutral's first, last and
    death frame, and its contents whenever they change while seen, per neutral (nlog_*)."""
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
                    key = ("tag", int(tags[v])) if ty in MOVING_NEUTRALS else (ty, round(float(u["u_x"][k][j]), 1), round(float(u["u_y"][k][j]), 1))
                    nid[v] = keys.setdefault(key, len(keys))
            rows.append((nid[ids], fo, u["u_display"][k] != SNAPSHOT, u["u_contents"][k]))
        ds, di = d["dead_frame_start"], d["dead_id"]
        for f in np.nonzero(np.diff(ds))[0].tolist():
            v = di[ds[f]:ds[f + 1]]
            v = nid[v[(v >= 0) & (v < len(nid))]]
            deaths += [(int(x), f) for x in v[v >= 0].tolist()]
        out[f"nid{s}"] = nid
    N = len(keys)
    n, fo, seen, cont = (np.concatenate([x[i] for x in rows]) for i in range(4))
    first, last, death = np.full(N, 2**30, np.int32), np.full(N, -1, np.int32), np.full(N, -1, np.int32)
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
    out.update(id0=np.arange(N, dtype=np.int32), first0=first, last0=last, death0=death, parent0=np.full(N, -1, np.int32),
               nlog_start=np.searchsorted(n, np.arange(N + 1)).astype(np.int64), nlog_frame=fo.astype(np.int32), nlog_contents=cont.astype(np.float32))
    return out


def _parents(replay_path, unit_tag, ids):
    """A view's row ids -> their creator's id, from the replay's SUnitBornEvent (-1: no creator, or not this view's)."""
    import mpyq
    from . import replay_actions
    a = mpyq.MPQArchive(replay_path)
    proto, _ = replay_actions._protocol(a)
    off = 1 << 32
    tag_id = {int(t): i for i, t in enumerate(unit_tag.tolist())}
    out = np.full(len(ids), -1, np.int32)
    pos = {int(k): i for i, k in enumerate(ids)}
    for e in proto.decode_replay_tracker_events(a.read_file("replay.tracker.events")):
        if not e["_event"].endswith("SUnitBornEvent") or e.get("m_creatorUnitTagIndex") is None:
            continue
        me = tag_id.get(off + (e["m_unitTagIndex"] << 18) + e["m_unitTagRecycle"])
        par = tag_id.get(off + (e["m_creatorUnitTagIndex"] << 18) + e["m_creatorUnitTagRecycle"])
        if me in pos and par is not None:
            out[pos[me]] = par
    return out


def _own_mask(owner, s):
    return owner == (s if s else NEUTRAL_PLAYER)


class Game:
    """One game's two views, opened lazily: batches are memory mapped, so windows of many games can be read without holding any."""

    def __init__(self, base, vocab):
        self.base, self.vocab = base, vocab
        self.views = {s: arrow_state.open_view(f"{base}.p{s}") for s in VIEWS}
        self.map_id = 0                         # the map's learned tokens, set by whoever knows the training maps
        ix = np.load(f"{base}.index.npz")
        self.F = int(ix["F"])
        self.index = {s: {k: ix[f"{k}{s}"] for k in ("id", "first", "last", "death", "parent")} for s in SOURCES}
        self.nid = {s: ix[f"nid{s}"] for s in VIEWS}
        self.nlog = ix["nlog_start"], ix["nlog_frame"], ix["nlog_contents"]
        self.rest = {s: statefile.load(f"{base}.p{s}.cst") for s in (1, 2)}
        d = self.rest[1]
        self.meta = json.loads(str(d["meta"]))
        W, H = self.W, self.H = self.meta["map_size"]
        x0, y0, x1, y1 = self.playable = self.meta["playable"]
        self.center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
        self.half = np.array([(x1 - x0) / 2, (y1 - y0) / 2], np.float32)
        self.side = float(max(x1 - x0, y1 - y0))
        self.cells = self._cell_index()
        h_ = d["map_terrain_height"].reshape(H, W)[y0:y1, x0:x1]
        self.static_grid = np.stack([h_, h_, np.ones_like(h_), np.ones_like(h_)], -1)      # [h, w, C_STATIC]: height per player, then playable and known: zero only where a batch pads
        self.grp = {s: (self.rest[s]["grp_start"], self.rest[s]["grp_id"]) if "grp_start" in self.rest[s].files else None for s in (1, 2)}
        self.tag_row = {s: {int(t): i for i, t in enumerate(self.rest[s]["unit_tag"].tolist())} for s in (1, 2)}
        self.acts = None
        # any view's unit id -> a flat index over all rows: tags are the same in every view, so a pointer reaches a row in another view, and a neutral's tags all reach its one row
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
        self.to_flat = {s: np.array([tag_flat.get(int(t), -1) for t in tags[s].tolist()], np.int64) for s in VIEWS}

    def _cell_index(self):
        """(y, x) real cell of each of the PATCH x PATCH samples of each map token: [TOP, TOP, PATCH * PATCH]. Tokens cover a square of the
        playable area's longest side, around its center; samples outside the map clip to its edge, where playable is false."""
        s = self.side / TOP
        c = (np.arange(TOP) + 0.5) * s - self.side / 2
        off = np.arange(PATCH) - (PATCH - 1) / 2
        xs = np.floor(self.center[0] + c[None, :, None] + off[None, None, :]).astype(np.int64)
        ys = np.floor(self.center[1] + c[:, None, None] + off[None, None, :]).astype(np.int64)
        x = np.broadcast_to(xs[:, :, None, :], (TOP, TOP, PATCH, PATCH)).reshape(TOP, TOP, -1)
        y = np.broadcast_to(ys[:, :, :, None], (TOP, TOP, PATCH, PATCH)).reshape(TOP, TOP, -1)
        return np.clip(y, 0, self.H - 1), np.clip(x, 0, self.W - 1)

    def _read(self, s, f0, f1, neutral=False):
        """View s, frames [f0, f1): its own rows', or the neutrals', columns and list columns, oldest first, already decoded and filtered."""
        r, _ = self.views[s]
        out_u, out_l = [], []
        for i in range(max(0, f0 // BATCH), min(r.num_record_batches, (f1 - 1) // BATCH + 1)):
            if (s, i) not in self._batches:
                self._batches[(s, i)] = arrow_state.read_units(r, i)
            u, l, fr = self._batches[(s, i)]
            keep = _own_mask(u["u_owner"], 0 if neutral else s) & (fr >= f0) & (fr < f1)
            if not neutral:  # a player's own snapshots are not units (see the desync check): tags outside the replay's log
                keep &= u["u_display"] != SNAPSHOT
            out_u.append({k: v[keep] for k, v in u.items()} | {"frame": fr[keep]})
            out_l.append((keep, l))
        return out_u, out_l

    def window(self, t0, L, swap=False):
        """L consecutive frames from frame t0, as one dict of numpy arrays: see the module docstring."""
        rows, row_of = [], {}
        self._batches = {}                     # own rows and neutrals share a view's batches
        for s in SOURCES:
            ix = self.index[s]
            live = (ix["first"] < t0 + L) & (ix["last"] >= t0)
            for k in np.nonzero(live)[0].tolist():
                row_of[(s, int(ix["id"][k]))] = len(rows)
                rows.append((s, k))
        R = len(rows)
        assert R <= MAX_ROWS, f"{R} rows"
        out = _blank(L, R)
        self.flat_row = np.full(self.n_flat + 1, -1, np.int32)        # the last entry answers id -1
        for r_, (s, k) in enumerate(rows):
            self.flat_row[self.flat_off[s] + k] = r_
        for s in SOURCES:
            self._fill(out, s, t0, L, row_of)
        self._dead_rows(out, rows, t0, L)
        self._batches = {}
        sw = (lambda o: 1 - o if o < 2 else o) if swap else (lambda o: o)
        out["owner"] = np.array([sw({1: 0, 2: 1, 0: NEUTRAL}[s]) for s, _ in rows], np.int64).reshape(-1)
        births, parents = [], []
        for r_, (s, k) in enumerate(rows):
            ix = self.index[s]
            f = int(ix["first"][k]) - t0
            if 0 < f < L:
                births.append((f, r_))
                p = row_of.get((s, int(ix["parent"][k])), -1) if ix["parent"][k] >= 0 else -1
                parents.append(p)
        out["birth_frame"] = np.array([b[0] for b in births], np.int64)
        out["birth_row"] = np.array([b[1] for b in births], np.int64)
        out["birth_parent"] = np.array(parents, np.int64)
        out["death_frame"] = np.array([self._death(s, k, t0, L) for s, k in rows], np.int64)
        out["map"], out["grid"] = self._map_frames(t0, L, swap)
        out["static_grid"], out["origin"], out["map_id"] = self.static_grid, np.array(self.playable[:2], np.float32), np.int64(self.map_id)
        out["center"], out["half"], out["side"] = self.center, self.half, np.float32(self.side)
        out["loops"] = t0 + 1 + np.arange(L)
        out["players"] = np.array([self.vocab.race.get(p["race"], 3) for p in self.meta["players"]], np.int64)[::-1 if swap else 1]
        out["ply_num"] = self._players(t0, L, swap)
        out["actions"] = self._actions(t0, L, row_of, swap)
        out["effects"] = self._effects(t0, L, swap)
        out["autocast"] = self._autocast(t0, L)
        return out

    def _autocast(self, t0, L):
        """Each row's remembered autocast toggles per frame: [(frame, row, family, button, on)]. A toggle at loop l first shows in the frame after l's, like
        any other action's effect. A unit never toggled has no entry: its default isn't known."""
        out = []
        for s, events in getattr(self, "autocast", {}).items():
            state = {}
            for loop, on, fam, but, units in events:
                if loop >= t0 + 1 + L:
                    break
                f0 = max(0, loop - t0)              # frame index of the loop after the toggle's
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

    def _effects(self, t0, L, swap):
        """Both player views' effects, side by side with no merging, since each view has only some: (frame, type, owner index, x, y, radius)."""
        rows = []
        for s in (1, 2):
            d = self.rest[s]
            st = d["e_frame_start"]
            a, b = st[t0], st[t0 + L]
            if a == b:
                continue
            fr = np.searchsorted(st, np.arange(a, b), side="right") - 1 - t0
            own = d["e_owner"][a:b].astype(np.int64)
            own = np.where(own == 1, 1 if swap else 0, np.where(own == 2, 0 if swap else 1, NEUTRAL))
            rows.append(np.stack([fr, np.clip(d["e_type"][a:b], 0, N_EFFECT_IDS - 1), own, d["e_x"][a:b] - self.center[0], d["e_y"][a:b] - self.center[1], d["e_radius"][a:b]], 1))
        return np.concatenate(rows).astype(np.float32) if rows else np.zeros((0, 6), np.float32)

    def _players(self, t0, L, swap):
        """[L, 2, C_PLY]: each player's PC_KEYS numbers and camera, from their own view."""
        out = np.zeros((L, 2, len(PC_KEYS) + 3), np.float32)
        for pi, s in enumerate((1, 2)):
            at = 1 - pi if swap else pi
            _, fr = self.views[s]
            for i in range(t0 // BATCH, (t0 + L - 1) // BATCH + 1):
                b = fr.get_batch(i)
                base = i * BATCH
                lo, hi = max(t0, base), min(t0 + L, base + b.num_rows)
                sl = np.arange(lo - base, hi - base)
                for j, k in enumerate(PC_KEYS):
                    v = b.column(f"pc_{k}").to_numpy(zero_copy_only=False)[sl].astype(np.float32)
                    out[lo - t0:hi - t0, at, j] = np.log1p(np.maximum(v, 0)) / 8
                for j, k in enumerate(("cam_x", "cam_y")):
                    v = b.column(k).to_numpy(zero_copy_only=False)[sl].astype(np.float32)
                    ok = np.isfinite(v)
                    out[lo - t0:hi - t0, at, len(PC_KEYS) + j] = np.where(ok, (np.nan_to_num(v) - self.center[j]) / self.half[j], 0)
                    out[lo - t0:hi - t0, at, len(PC_KEYS) + 2] = ok
        return out

    def _load_actions(self, avocab):
        """The game's action tokens, once. The tags of both views' rows stand in for one id space; a neutral's tags all stand for its one row."""
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
        # autocast: the observation never shows it, so each toggle's state is remembered per unit: (loop, on, family, button, unit ids) per view
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
                    ev.append((int(d["ac_loop"][i]), bool(d["ac_on"][i]), fam, int(d["ac_cmd"][i]) + 1, d["ac_unit"][st[i]:st[i + 1]].astype(np.int64)))
            self.autocast[s] = ev

    def _actions(self, t0, L, row_of, swap):
        """Tokens in frames [t0, t0 + L): an action at loop l is in the frame of that loop, so it reads the state it was given on and affects the next frame."""
        tk = self.acts.tok
        lo, hi = t0 + 1, t0 + 1 + L
        idx = np.nonzero((tk["loop"] >= lo) & (tk["loop"] < hi) & (tk["kind"] > 0))[0]
        n = len(idx)
        xy = np.stack([tk["x"][idx], tk["y"][idx]], 1)
        placed = np.isfinite(xy).all(1)
        target = np.full(n, -1, np.int64)
        uid = tk["uid"][idx]
        xy = np.nan_to_num(xy)
        for i in np.nonzero(uid >= 0)[0].tolist():
            target[i] = row_of.get(self.act_key[int(uid[i])], -1)
        flip = lambda p: (2 - p if swap else p - 1)
        num = np.stack([tk["queued"][idx], tk["smart"][idx], tk["minimap"][idx], np.log1p(np.maximum(tk["presses"][idx] - 1.0, 0)) / 2,
                        np.log1p(tk["n_add"][idx].clip(0)) / 3, np.log1p(tk["n_rem"][idx].clip(0)) / 3, np.log1p(tk["minerals"][idx].clip(0)) / 8,
                        np.log1p(tk["gas"][idx].clip(0)) / 8, tk["recipient"][idx] >= 0, target >= 0, (uid >= 0) & (target < 0), placed], 1).astype(np.float32)
        return dict(frame=(tk["loop"][idx] - lo).astype(np.int64), player=flip(tk["player"][idx].astype(np.int64)).clip(0, 1), kind=tk["kind"][idx].astype(np.int64),
                    family=tk["family"][idx].astype(np.int64), button=tk["button"][idx].astype(np.int64), tkind=tk["tkind"][idx].clip(0).astype(np.int64),
                    group=tk["group"][idx].clip(0).astype(np.int64), gop=tk["gop"][idx].clip(0).astype(np.int64), remove=tk["remove"][idx].clip(0).astype(np.int64), subgroup=tk["subgroup"][idx].clip(0).astype(np.int64),
                    num=num, xy=np.where(placed[:, None], xy - self.center, 0).astype(np.float32), placed=placed, target=target)   # no place: the center, like the globals

    def _death(self, s, k, t0, L):
        d = int(self.index[s]["death"][k])
        return d - t0 if t0 <= d < t0 + L else -1

    def _fill(self, out, s, t0, L, row_of):
        """Writes view s's rows of the window into the grid, then bridges passengers and gas workers."""
        if s == 0:
            return self._fill_neutrals(out, t0, L, row_of)
        lo, hi = max(0, t0 - MARGIN), min(self.F, t0 + L + MARGIN)
        U, Ls = self._read(s, lo, hi)
        if not U:
            return
        u = {k: np.concatenate([x[k] for x in U]) for k in U[0]}
        frame, ids = u["frame"], u["u_id"]
        inw = (frame >= t0) & (frame < t0 + L)
        fo = frame[inw] - t0
        ro = np.array([row_of.get((s, int(i)), -1) for i in ids[inw].tolist()], np.int64)
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
            self._groups(out, s, t0, L, ids, frame, row_of)
            self._contained(out, s, t0, L, row_of, u, bags)

    def _fill_neutrals(self, out, t0, L, row_of):
        """Neutrals from both views: a snapshot first, then what a player sees over it, once. Contents as last seen."""
        ix = self.index[0]
        nid_row = np.full(len(ix["id"]), -1, np.int64)
        for (s, k), r_ in row_of.items():
            if s == 0:
                nid_row[k] = r_
        seen = np.zeros(out["mask"].shape, bool)
        for visible in (False, True):
            for s in VIEWS:
                U, Ls = self._read(s, t0, t0 + L, neutral=True)
                u = {k: np.concatenate([x[k] for x in U]) for k in U[0]}
                n = self.nid[s][u["u_id"]]
                fo, ro = u["frame"] - t0, nid_row[n]
                d = ix["death"][n]
                ok = (ro >= 0) & ((d < 0) | (u["frame"] <= d)) & ((u["u_display"] != SNAPSHOT) == visible) & ~seen[fo, ro]
                src = np.nonzero(ok)[0]
                _set(out, u, src, fo[src], ro[src], self)
                if not visible:
                    continue
                seen[fo[src], ro[src]] = True
                out["engaged_row"][fo[src], ro[src]] = self._row_of_id(s, u["u_engaged"][src])
                off, bags = 0, []
                for (keep, l), x in zip(Ls, U):
                    idx = np.nonzero(keep)[0]
                    bags.append((off, idx, l))
                    off += len(idx)
                self._bags(out, bags, src, fo[src], ro[src], s)
        start, log_f, log_c = self.nlog
        at = RAW_NUMS.index("contents")
        frames = t0 + np.arange(L)
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

    def _groups(self, out, s, t0, L, ids, frame, row_of):
        g = self.grp.get(s)
        if g is None:
            return
        start, gid = g
        P = len(self.rest[s]["grp_player"])
        for f in range(L):
            base = ((t0 + f) * P) * C_GROUPS
            for k in range(C_GROUPS):
                members = gid[start[base + k]:start[base + k + 1]]
                for m in members.tolist():
                    r_ = row_of.get((s, m), -1)
                    if r_ >= 0:
                        out["groups"][f, r_, k] = 1

    def _bags(self, out, bags, src, fo, ro, s):
        """Orders, production, buffs and rally points of the window's rows, as flat lists: (frame, row, ...)."""
        pos = np.full(sum(len(b[1]) for b in bags), -1, np.int64)
        pos[src] = np.arange(len(src))
        for off, idx, l in bags:
            sel = np.nonzero(pos[off:off + len(idx)] >= 0)[0]
            if not len(sel):
                continue
            which = pos[off:off + len(idx)][sel]
            orig = idx[sel]
            for name, arrs in (("ord", ("ord_abil", "ord_tx", "ord_ty", "ord_target", "ord_progress")), ("prod", ("prod_abil", "prod_progress")),
                               ("buff", ("b_id",)), ("rally", ("rally_x", "rally_y", "rally_target")), ("ab", ("ab_id",))):
                offsets = np.asarray(l[arrs[0]].offsets)
                vals = [l[a].values.to_numpy(zero_copy_only=False) for a in arrs]
                n = offsets[orig + 1] - offsets[orig]
                if not n.sum():
                    continue
                it = np.concatenate([np.arange(a, b) for a, b in zip(offsets[orig], offsets[orig + 1])])
                rep = np.repeat(which, n)
                seq = np.concatenate([np.arange(c) for c in n])
                st = [v[it].astype(np.float32) for v in vals]
                if name == "ord":                 # order targets as rows
                    st[3] = self._row_of_id(s, vals[3][it]).astype(np.float32)
                elif name == "rally":
                    st[2] = self._row_of_id(s, vals[2][it]).astype(np.float32)
                out[f"{name}_where"].append(np.stack([fo[rep], ro[rep], seq], 1))
                out[f"{name}_vals"].append(np.stack(st, 1))

    def _contained(self, out, s, t0, L, row_of, u, bags):
        """Rows missing from their frame: passengers from their transport's list, gas workers from the order they vanished under."""
        order = np.lexsort((u["frame"], u["u_id"]))
        ids, fr = u["u_id"][order], u["frame"][order]
        cut = np.nonzero(np.diff(ids) == 0)[0]
        gaps = cut[(fr[cut + 1] - fr[cut]) > 1]
        for j in gaps.tolist():
            uid, a, b = int(ids[j]), int(fr[j]), int(fr[j + 1])
            r_ = row_of.get((s, uid), -1)
            if r_ < 0 or b - a > GAS_GAP_LOOPS or int(u["u_type"][order[j]]) not in WORKER_TYPES:
                continue
            tgt = self._order_target(bags, order[j])
            if tgt is None:
                continue
            g = row_of.get((s, tgt), -1)
            if g < 0:
                continue
            for f in range(max(a + 1, t0), min(b, t0 + L)):
                ff = f - t0
                if out["mask"][ff, g] and int(out["type"][ff, g]) in GAS_TYPES:
                    out["mask"][ff, r_], out["inside"][ff, r_], out["parent_row"][ff, r_] = True, 2, g
                    out["x"][ff, r_], out["y"][ff, r_] = out["x"][ff, g], out["y"][ff, g]
                    out["type"][ff, r_] = u["u_type"][order[j]]
                    out["num"][ff, r_] = _carry(u, order[j])
        for off, idx, l in bags:
            offsets = np.asarray(l["pass_id"].offsets)
            if offsets[-1] == offsets[0]:
                continue
            vid, vty, vhp = (l[k].values.to_numpy(zero_copy_only=False) for k in ("pass_id", "pass_type", "pass_hp"))
            for local, i in enumerate(idx.tolist()):
                for it in range(offsets[i], offsets[i + 1]):
                    f = int(u["frame"][off + local]) - t0
                    if not 0 <= f < L:
                        continue
                    r_, c = row_of.get((s, int(vid[it])), -1), row_of.get((s, int(u["u_id"][off + local])), -1)
                    if r_ < 0 or c < 0 or out["mask"][f, r_]:
                        continue
                    out["mask"][f, r_], out["inside"][f, r_], out["parent_row"][f, r_] = True, 1, c
                    out["x"][f, r_], out["y"][f, r_], out["type"][f, r_] = u["u_x"][off + local], u["u_y"][off + local], vty[it]
                    out["num"][f, r_, 0] = float(vhp[it])

    def _order_target(self, bags, row):
        for off, idx, l in bags:
            if off <= row < off + len(idx):
                i = idx[row - off]
                offsets = np.asarray(l["ord_target"].offsets)
                if offsets[i + 1] > offsets[i]:
                    t = int(l["ord_target"].values.to_numpy(zero_copy_only=False)[offsets[i]])
                    return t if t >= 0 else None
                return None
        return None

    def _dead_rows(self, out, rows, t0, L):
        """After a death a row keeps its last values with hit points and shields 0, and isn't alive."""
        for r_, (s, k) in enumerate(rows):
            d = self._death(s, k, t0, L)
            died_before = 0 <= self.index[s]["death"][k] < t0
            if died_before:
                continue
            if d < 0 or d + 1 >= L:
                continue
            live = np.nonzero(out["mask"][:d + 1, r_])[0]
            if not len(live):
                continue
            f0 = int(live[-1])
            for key in ("x", "y", "type", "inside", "parent_row", "num", "groups", "id", "engaged_row"):
                if key in out:
                    out[key][d + 1:, r_] = out[key][f0, r_]
            out["num"][d + 1:, r_, 0] = 0
            out["num"][d + 1:, r_, 1] = 0
            out["mask"][d + 1:, r_] = True
            out["alive"][d + 1:, r_] = False

    def _map_frames(self, t0, L, swap):
        """[L, TOP, TOP, C_DYNAMIC * PATCH * PATCH] uint8: each player's visible, seen and creep, the next-frame targets; and [L, h, w, C_LIVE], the playable area at its own
        size with each player's pathable and buildable too, the input."""
        out = np.zeros((L, TOP, TOP, C_DYNAMIC * PATCH * PATCH), np.uint8)
        x0, y0, x1, y1 = self.playable
        grid = np.zeros((L, y1 - y0, x1 - x0, C_LIVE), np.uint8)
        bits = lambda col, j: np.unpackbits(np.frombuffer(col[j].as_py(), np.uint8).reshape(self.H, -1), axis=1)[:, :self.W]
        y, x = self.cells
        for pi, s in enumerate((1, 2)):
            at = (1 - pi if swap else pi) * 3 * PATCH * PATCH
            _, fr = self.views[s]
            for f in range(L):
                b = fr.get_batch((t0 + f) // BATCH)
                j = (t0 + f) % BATCH
                vis = np.frombuffer(b.column("vis")[j].as_py(), np.uint8).reshape(self.H, self.W)
                creep = np.unpackbits(np.frombuffer(b.column("creep")[j].as_py(), np.uint8).reshape(self.H, -1), axis=1)[:, :self.W]
                out[f, :, :, at:at + PATCH * PATCH] = (vis == 2)[y, x]
                out[f, :, :, at + PATCH * PATCH:at + 2 * PATCH * PATCH] = (vis > 0)[y, x]
                out[f, :, :, at + 2 * PATCH * PATCH:at + 3 * PATCH * PATCH] = creep[y, x]
                g = (1 - pi if swap else pi) * 5
                crop = lambda a: a[y0:y1, x0:x1]
                grid[f, :, :, g], grid[f, :, :, g + 1], grid[f, :, :, g + 2] = crop(vis == 2), crop(vis > 0), crop(creep)
                grid[f, :, :, g + 3], grid[f, :, :, g + 4] = crop(bits(b.column("pathable"), j)), crop(bits(b.column("buildable"), j))
        return out, grid


def _blank(L, R):
    out = dict(mask=np.zeros((L, R), bool), alive=np.ones((L, R), bool), inside=np.zeros((L, R), np.int8),
               parent_row=np.full((L, R), -1, np.int32), type=np.zeros((L, R), np.int16), x=np.zeros((L, R), np.float32),
               y=np.zeros((L, R), np.float32), id=np.full((L, R), -1, np.int32), num=np.zeros((L, R, len(RAW_NUMS)), np.float32),
               groups=np.zeros((L, R, C_GROUPS), np.int8))
    out["engaged_row"] = np.full((L, R), -1, np.int32)
    for k in ("ord", "prod", "buff", "rally", "ab"):
        out[f"{k}_where"], out[f"{k}_vals"] = [], []
    return out


# the raw numbers each row carries, unscaled: the model embeds them, the raw heads predict them
RAW_NUMS = ("hp", "shield", "energy", "hp_max", "shield_max", "build", "contents", "harvesters", "ideal_harvesters", "facing", "radius",
            "weapon_cooldown", "cargo_used", "cargo_max", "buff_remain", "buff_max", "detect_range", "radar_range", "atk_up", "arm_up",
            "shd_up", "flags", "flags2", "display", "cloak", "add_on", "engaged", "z")


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


QUEUE, PROD = 8, 8               # order and production slots with targets (physics.RowHead)
FORCED = 16                      # map tokens under effects always among each frame's decoded sample


def collate(windows, vocab):
    """Windows of the same length -> one batch of tensors: grids [B, F, R, ...], lists flat by (B * F + f) * R + r."""
    import torch
    B, F_ = len(windows), windows[0]["mask"].shape[0]
    R = max(w["mask"].shape[1] for w in windows)
    out = {}
    for k in ("mask", "alive", "inside", "parent_row", "engaged_row", "type", "x", "y", "num", "groups"):
        a0 = windows[0][k]
        fill = True if k == "alive" else -1 if k in ("parent_row", "engaged_row") else 0
        buf = np.full((B, F_, R) + a0.shape[2:], fill, a0.dtype)
        for b, w in enumerate(windows):
            buf[b, :, :w[k].shape[1]] = w[k]
        out[k] = torch.from_numpy(buf)
    out["rtype"] = out["type"].long()
    out["type"] = torch.from_numpy(vocab.unit_index(out["rtype"].numpy()).astype(np.int64))
    own = np.full((B, R), NEUTRAL, np.int64)
    for b, w in enumerate(windows):
        own[b, :len(w["owner"])] = w["owner"]
    out["owner"] = torch.from_numpy(own)
    for k in ("center", "half", "side", "players", "ply_num", "map", "origin", "map_id"):
        out[k] = torch.from_numpy(np.stack([w[k] for w in windows]))
    # the map at its own size, zero past its edge: the known layer tells. Sizes are multiples of 8, the tower's reduction
    gh, gw = (int(np.ceil(max(w["grid"].shape[i] for w in windows) / 8)) * 8 for i in (1, 2))
    grid, sgrid = np.zeros((B, F_, gh, gw, C_LIVE), np.uint8), np.zeros((B, gh, gw, C_STATIC), np.uint8)
    for b, w in enumerate(windows):
        h_, w_ = w["grid"].shape[1:3]
        grid[b, :, :h_, :w_], sgrid[b, :h_, :w_] = w["grid"], w["static_grid"]
    out["grid"], out["static_grid"] = torch.from_numpy(grid), torch.from_numpy(sgrid)
    out["view"] = torch.ones(B, 2)
    flat = lambda b, wh: (b * F_ + wh[:, 0]) * R + wh[:, 1]
    bags = {k: ([], []) for k in ("ord", "prod", "buff", "rally", "ab")}
    ord_row = np.full((B, F_, R, QUEUE), -1, np.int32)
    rally = np.zeros((B, F_, R, 4), np.float32)
    slot_abil = np.zeros((B, F_, R, QUEUE + PROD), np.int64)
    slot_reg = np.zeros((B, F_, R, QUEUE + PROD, 4), np.float32)
    for b, w in enumerate(windows):
        for k in bags:
            if w[f"{k}_where"]:
                wh, v = np.concatenate(w[f"{k}_where"]), np.concatenate(w[f"{k}_vals"])
                bags[k][0].append((flat(b, wh), wh, v))
    cat = lambda k: [np.concatenate(z) for z in zip(*bags[k][0])] if bags[k][0] else (np.zeros(0, np.int64), np.zeros((0, 3), np.int64), np.zeros((0, 5), np.float32))
    bh = lambda at: at // (F_ * R)
    at, wh, v = cat("ord")
    ab = vocab.ability_index(v[:, 0].astype(np.int64)).astype(np.int64) if len(v) else np.zeros(0, np.int64)
    if len(at):
        bb = bh(at)
        ux, uy = out["x"].numpy()[bb, wh[:, 0], wh[:, 1]], out["y"].numpy()[bb, wh[:, 0], wh[:, 1]]
        hx, hy = out["half"].numpy()[bb, 0], out["half"].numpy()[bb, 1]
        dx, dy = np.where(np.isfinite(v[:, 1]), np.nan_to_num(v[:, 1]) - ux, 0) / hx, np.where(np.isfinite(v[:, 2]), np.nan_to_num(v[:, 2]) - uy, 0) / hy
        feat = np.stack([np.clip(dx, -2, 2), np.clip(dy, -2, 2), v[:, 3] >= 0, v[:, 4]], 1).astype(np.float32)   # has a target: as a row of this window
        q = wh[:, 2] < QUEUE
        slot_abil[bb[q], wh[q, 0], wh[q, 1], wh[q, 2]] = ab[q]
        slot_reg[bb[q], wh[q, 0], wh[q, 1], wh[q, 2]] = np.stack([feat[q, 0], feat[q, 1], feat[q, 2], feat[q, 3]], 1)
        ord_row[bb[q], wh[q, 0], wh[q, 1], wh[q, 2]] = v[q, 3]
    else:
        feat = np.zeros((0, 4), np.float32)
    out["ord_at"], out["ord_abil"], out["ord_feat"] = torch.from_numpy(at.astype(np.int64)), torch.from_numpy(ab), torch.from_numpy(feat)
    at, wh, v = cat("prod")
    ab = vocab.ability_index(v[:, 0].astype(np.int64)).astype(np.int64) if len(v) else np.zeros(0, np.int64)
    feat = np.stack([v[:, 1], wh[:, 2] / 5], 1).astype(np.float32) if len(v) else np.zeros((0, 2), np.float32)
    if len(at):
        bb = bh(at)
        q = wh[:, 2] < PROD
        slot_abil[bb[q], wh[q, 0], wh[q, 1], QUEUE + wh[q, 2]] = ab[q]
        slot_reg[bb[q], wh[q, 0], wh[q, 1], QUEUE + wh[q, 2], 3] = v[q, 1]
    out["prod_at"], out["prod_abil"], out["prod_feat"] = torch.from_numpy(at.astype(np.int64)), torch.from_numpy(ab), torch.from_numpy(feat)
    at, wh, v = cat("buff")
    out["buff_at"], out["buff_id"] = torch.from_numpy(at.astype(np.int64)), torch.from_numpy(np.clip(v[:, 0], 0, 511).astype(np.int64) if len(v) else np.zeros(0, np.int64))
    at, wh, v = cat("rally")
    if len(at):
        bb = bh(at)
        ux, uy = out["x"].numpy()[bb, wh[:, 0], wh[:, 1]], out["y"].numpy()[bb, wh[:, 0], wh[:, 1]]
        hx, hy = out["half"].numpy()[bb, 0], out["half"].numpy()[bb, 1]
        feat = np.stack([np.clip((v[:, 0] - ux) / hx, -2, 2), np.clip((v[:, 1] - uy) / hy, -2, 2), v[:, 2] >= 0], 1).astype(np.float32)
        first = wh[:, 2] == 0
        rally[bb[first], wh[first, 0], wh[first, 1]] = np.concatenate([np.ones((first.sum(), 1), np.float32), feat[first]], 1)
    else:
        feat = np.zeros((0, 3), np.float32)
    out["rally_at"], out["rally_feat"] = torch.from_numpy(at.astype(np.int64)), torch.from_numpy(feat)
    out["rally_first"], out["ord_row"] = torch.from_numpy(rally), torch.from_numpy(ord_row)
    at, wh, v = cat("ab")
    out["ab_at"], out["ab_id"] = torch.from_numpy(at.astype(np.int64)), torch.from_numpy(np.clip(v[:, 0], 0, N_RAW_ABILITY_IDS - 1).astype(np.int64) if len(v) else np.zeros(0, np.int64))
    out["slot_abil"], out["slot_reg"] = torch.from_numpy(slot_abil), torch.from_numpy(slot_reg)
    changed = np.zeros((B, F_, R), bool)
    changed[:, :-1] = (slot_abil[:, 1:] != slot_abil[:, :-1]).any(-1)
    out["queue_changed"] = torch.from_numpy(changed[:, :-1])
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
    out["birth_b"], out["birth_f"], out["birth_r"], out["birth_p"] = (torch.tensor(x, dtype=torch.int64) for x in (bb, ff, rr, pp))
    # actions, per frame
    A = max(2, max((np.bincount(w["actions"]["frame"], minlength=1).max() if len(w["actions"]["frame"]) else 0) for w in windows))
    shape = (B, F_, A)
    amask = np.zeros(shape, bool)
    keys = dict(kind=np.int64, family=np.int64, button=np.int64, tkind=np.int64, group=np.int64, gop=np.int64, remove=np.int64, subgroup=np.int64, player=np.int64)
    ac = {k: np.zeros(shape, t) for k, t in keys.items()}
    anum, axy = np.zeros(shape + (windows[0]["actions"]["num"].shape[1],), np.float32), np.zeros(shape + (2,), np.float32)
    placed, target = np.zeros(shape, bool), np.full(shape, -1, np.int64)
    orders = [[[], []] for _ in windows]
    for b, w in enumerate(windows):
        a = w["actions"]
        seen = np.zeros(F_, np.int64)
        for i, f in enumerate(a["frame"].tolist()):
            s = seen[f]
            seen[f] += 1
            amask[b, f, s] = True
            for k in keys:
                ac[k][b, f, s] = a[k][i]
            anum[b, f, s], axy[b, f, s], placed[b, f, s] = a["num"][i], a["xy"][i], a["placed"][i]
            target[b, f, s] = a["target"][i]
            orders[b][int(a["player"][i])].append(f * A + s)
    N = max(2, max(max(len(o[0]), len(o[1])) for o in orders))
    order = np.full((B, 2, N), -1, np.int64)
    for b, o in enumerate(orders):
        for p in (0, 1):
            order[b, p, :len(o[p])] = o[p]
    for k in keys:
        out[f"act_{k}"] = torch.from_numpy(ac[k])
    out["act_mask"], out["act_num"], out["act_xy"], out["act_placed"], out["act_target"], out["act_order"] = (
        torch.from_numpy(x) for x in (amask, anum, axy, placed, target, order))
    au = [(b * F_ + w["autocast"][:, 0]) * R + w["autocast"][:, 1] for b, w in enumerate(windows)]
    av = np.concatenate([w["autocast"] for w in windows])
    out["auto_at"], out["auto_fam"], out["auto_but"], out["auto_on"] = torch.from_numpy(np.concatenate(au).astype(np.int64)), *(torch.from_numpy(av[:, k]) for k in (2, 3, 4))
    # effects, per frame: tokens, and the map tokens each covers in the next frame
    X = max(1, max((np.bincount(w["effects"][:, 0].astype(np.int64), minlength=1).max() if len(w["effects"]) else 0) for w in windows))
    emask, etype, eown = np.zeros((B, F_, X), bool), np.zeros((B, F_, X), np.int64), np.full((B, F_, X), NEUTRAL, np.int64)
    exy, erad = np.zeros((B, F_, X, 2), np.float32), np.zeros((B, F_, X), np.float32)
    force = np.full((B, F_ - 1, FORCED), -1, np.int64)
    cover = []
    for b, w in enumerate(windows):
        seen = np.zeros(F_, np.int64)
        size = float(w["side"]) / TOP
        centers = (np.arange(TOP) + 0.5) * size - float(w["side"]) / 2
        n_forced = np.zeros(F_, np.int64)
        for f, ty, o, x, y, r in w["effects"].tolist():
            f, ty, o = int(f), int(ty), int(o)
            i = seen[f]
            seen[f] += 1
            emask[b, f, i], etype[b, f, i], eown[b, f, i], exy[b, f, i], erad[b, f, i] = True, ty, o, (x, y), r / 8
            if f == 0:
                continue
            reach = max(r, size / 2)
            jx = np.nonzero(np.abs(centers - x) <= reach)[0]
            jy = np.nonzero(np.abs(centers - y) <= reach)[0]
            for iy in jy.tolist():
                for ix in jx.tolist():
                    if (centers[ix] - x) ** 2 + (centers[iy] - y) ** 2 <= reach * reach + size * size / 2:
                        cell = iy * TOP + ix
                        cover.append(((b * (F_ - 1) + f - 1), cell, ty))
                        if n_forced[f] < FORCED:
                            force[b, f - 1, n_forced[f]] = cell
                            n_forced[f] += 1
    out["eff_mask"], out["eff_type"], out["eff_owner"], out["eff_xy"], out["eff_radius"] = (torch.from_numpy(x) for x in (emask, etype, eown, exy, erad))
    out["eff_force"] = torch.from_numpy(force)
    out["eff_cover"] = torch.tensor(cover, dtype=torch.int64).reshape(-1, 3)
    return out
