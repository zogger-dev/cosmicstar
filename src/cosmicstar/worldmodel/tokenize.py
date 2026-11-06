"""Turns an extracted frame into encoder inputs.

A frame is what physically exists at one instant (no orders, no velocity: motion
and intent are for the predictor to infer across frames and from the action stream).
It becomes one vector per thing, of two kinds:

    entities  every unit, structure and neutral thing (mineral fields and geysers
              with what's left in them, rocks, towers)
    players   one per player: bank, income, workers, supply, army value, race, upgrades

plus frame-level inputs (game time, game format) that go into the encoder's CLS, and per
grid cell what the viewer can see (share of the cell visible now / ever seen; all 1 for
the observer).

A player's view (extract_states.py --views players) shows what that player knows: fog,
snapshots of remembered structures, allies' units without their private details. Tracker
stats and upgrades are kept for the viewer only (other players' economy isn't on
screen).

Layout-2 additions (2026-10-31, the pyramid encoder): each entity also carries its
facing, radius, weapon cooldown, engaged target, upgrade levels, buffs, detection and
radar range, cargo, display / cloak state and its list-valued state pooled into its
token (the full order queue, the production queue, buffs, available abilities, rally
points). Units inside other units are rows of their own with a pointer to their
container: passengers of transports, bunkers and command centers (from the extraction's
passenger lists) and workers inside gas buildings (synthesized while they're out of
sight harvesting). Add-ons point at their building. Effects (storm, bile, ...) are
tokens of their own. Players carry player_common, the score details, units lost, the
camera and its saved positions, the subgroup and the last command. The map grids
(visibility, creep, pathing, placement, height) go to the encoder at one cell per map
unit.

The combined view (CombinedFrames, 2026-10-31): there is no observer extraction; the
full-information view is the two player views side by side, not merged: every entity and
effect token of both views (a unit both players see is two tokens, a snapshot one player
keeps of minerals the other saw destroyed stays a token), each marked with the view it
came from (ent_src / eff_src: that viewer's player code) and its relation as that viewer
sees it. Attention sorts out what's true. Players carry both players' private parts. The
map grid carries visibility, seen and creep layers per player (in player-slot order,
like the player tokens; ply_gridok says whose are filled: the viewer's in a player view,
both in the combined view); the encoder embeds each player's layers with that player's
code and sums them into one minimap, so "seen by this player, not by that one" is a
feature it can learn.

Ownership is a role, never an identity, and works for any game format:
    team code    random per sample from a pool of 8 (same code = same team)
    player code  random per sample from a pool of 8 (same code = same player)
    relation     "player" view only: me / ally / enemy (observer view: none)
Neutral things get their own team and player code.
"""

import json

import numpy as np

from .vocab import PAD

N_CODES = 8  # team / player codes per game (max 8 players)
CODE_NEUTRAL = N_CODES  # team and player code for neutral things
N_CODE_EMB = N_CODES + 1
REL_NONE, REL_ME, REL_ALLY, REL_ENEMY = 0, 1, 2, 3
N_REL = 4
NEUTRAL_PLAYER = 16  # engine owner id of neutral units

# entity numbers (entity_features): the 18 of layout 1, then layout 2's
ENTITY_NUMS = (
    "hp",
    "shield",
    "energy",
    "hp_max",
    "shield_max",
    "build",
    "flying",
    "burrowed",
    "cloaked",
    "halluc",
    "snapshot",
    "powered",
    "selected",
    "progress",
    "progress2",
    "queue",
    "contents",
    "harvesters",
    "facing_sin",
    "facing_cos",
    "radius",
    "weapon_cooldown",
    "engaged",
    "engaged_dx",
    "engaged_dy",
    "atk_up",
    "arm_up",
    "shd_up",
    "buff_frac",
    "buff_max",
    "ideal_harvesters",
    "cargo_used",
    "cargo_max",
    "detect_range",
    "radar_range",
    "display_visible",
    "display_snapshot",
    "display_hidden",
    "cloak_unknown",
    "cloak_cloaked",
    "cloak_detected",
    "cloak_none",
    "active",
    "blip",
    "onscreen",
    "passenger",
    "in_gas",
    "abilities_known",
    "has_add_on",
)
SC_KEYS = (
    "idle_production_time",
    "idle_worker_time",
    "total_value_units",
    "total_value_structures",
    "killed_value_units",
    "killed_value_structures",
    "collected_minerals",
    "collected_vespene",
    "collection_rate_minerals",
    "collection_rate_vespene",
    "spent_minerals",
    "spent_vespene",
    *[f"food_used_{c}" for c in ("none", "army", "economy", "technology", "upgrade")],
    *[
        f"{a}_{r}_{c}"
        for a in ("killed", "lost", "friendly_fire", "used", "total_used")
        for r in ("minerals", "vespene")
        for c in ("none", "army", "economy", "technology", "upgrade")
    ],
    *[
        f"total_{a}_{c}"
        for a in ("damage_dealt", "damage_taken", "healed")
        for c in ("life", "shields", "energy")
    ],
    "current_apm",
    "current_effective_apm",
)
PC_KEYS = (
    "minerals",
    "vespene",
    "food_cap",
    "food_used",
    "food_army",
    "food_workers",
    "idle_worker_count",
    "army_count",
    "warp_gate_count",
    "larva_count",
)
# player numbers: tracker stats (9), player_common (10), score details (78), then units
# lost (log), whether the private parts are known, camera xy (normalized), subgroup,
# last command: age (log loops), has a point, x, y, targets a unit
N_STATS = 9
C_PLAYER = N_STATS + len(PC_KEYS) + len(SC_KEYS) + 2 + 2 + 1 + 5
C_ENTITY, C_FORMAT, C_VIS = len(ENTITY_NUMS), 4, 2
N_CAM_SAVES = 8
# map grid channels at one cell per map unit (collate builds the canvas)
N_GRID_PLAYERS = 2  # per-view layers for the first two players (1v1)
GRID_CHANNELS = (
    *[f"{c}_{i}" for i in range(N_GRID_PLAYERS) for c in ("visible", "seen", "creep")],
    "pathable",
    "placeable",
    "height",
    "on_map",
    "playable",
)
C_GRID = len(GRID_CHANNELS)
GAS_TYPES = (20, 61, 88, 1943, 1980, 1981)  # refinery, assimilator, extractor (+ rich)
WORKER_TYPES = (45, 84, 104)  # SCV, probe, drone
GAS_GAP_LOOPS = 64  # longest out-of-sight harvest gap bridged (~2.9 s)
# list-valued entity state pooled into the token: name -> number of item features
BAGS = dict(ord=5, prod=2, buff=1, abil=1, rally=3)
N_BUFF_IDS, N_RAW_ABILITY_IDS, N_EFFECT_IDS = 512, 4096, 64
# the unit's membership in its owner's control groups 0-9 and active selection (10),
# from the
C_GROUPS = 11
# extraction's selection tracker (grp_*): the memory a 2.5 s window cannot rebuild from
# events
# the entity's current order: has one, target offset (x, y), targets a unit, progress,
# queue length
C_ORDER = 6
GRID = 16  # the encoder's grid (G x G over the playable area)
STAT_KEYS = (
    "minerals",
    "vespene",
    "mineral_rate",
    "vespene_rate",
    "workers",
    "food_used",
    "food_made",
    "army_minerals",
    "army_vespene",
)


def _log(x, scale=10.0):
    return np.log1p(np.maximum(x, 0)) / scale


class GameFrames:
    """One extracted replay, held in memory, served frame by frame."""

    def __init__(self, path, vocab):
        from . import statefile

        d = statefile.load(path)
        self.path = path
        self.vocab = vocab
        self.meta = json.loads(str(d["meta"]))
        self.loops = d["frame_loop"]
        self.start = d["frame_start"]
        self.step = (
            int(np.median(np.diff(self.loops))) if len(self.loops) > 1 else 16
        )  # loops per frame
        self.unit_tag = (
            d["unit_tag"] if "unit_tag" in d.files else None
        )  # state unit id -> engine tag
        self.grp = (
            (d["grp_start"], d["grp_id"], len(d["grp_player"]))
            if "grp_start" in d.files
            else None
        )
        self._rows_cache = {}
        u = self.u = {k[2:]: d[k] for k in d.files if k.startswith("u_")}
        self.F = len(self.loops)
        u["fidx"] = np.repeat(np.arange(self.F), np.diff(self.start)).astype(np.int32)
        self._lists(d)
        x0, y0, x1, y1 = self.meta["playable"]
        self.center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
        self.half = np.array([(x1 - x0) / 2, (y1 - y0) / 2], np.float32)
        self.players = self.meta["players"]
        self.slots = [p["slot"] for p in self.players]
        self.viewer = int(
            self.meta.get("perspective", 0)
        )  # 0 = observer, else the player's slot
        self.known = (
            {self.viewer} if self.viewer else set()
        )  # players whose private state is in the view
        # the engine's executed commands of the viewer, with recipients (state unit ids,
        # ragged by start)
        self.cmd = (
            {
                k: d[f"cmd_{k}"]
                for k in (
                    "loop",
                    "frame",
                    "abil",
                    "target",
                    "tx",
                    "ty",
                    "target_id",
                    "queued",
                    "start",
                    "unit",
                )
            }
            if "cmd_loop" in d.files
            else None
        )
        self._production(d)
        self._orders(d)
        self._contained(d)
        u = self.u
        u["unit_idx"] = vocab.unit_index(u["type"])
        u["cat"] = vocab.category(u["type"])
        u["resource"] = (u["owner"] == NEUTRAL_PLAYER) & (u["contents"] > 0)
        self._links()
        self._slim()
        self._effects(d)
        self._grids(d)
        self._player_extras(d)
        self._stats(d)
        self._upgrades(d)
        sizes = sorted(
            (
                sum(p["team"] == t for p in self.players)
                for t in {p["team"] for p in self.players}
            ),
            reverse=True,
        )
        self.format = np.array(
            [len(sizes) / 4, sizes[0] / 4, sizes[-1] / 4, len(self.players) / 8],
            np.float32,
        )
        self._visibility(d)
        # [F, P] ability family of each player's last command (set_command_families)
        self.lc_family = None

    def __len__(self):
        return len(self.loops)

    # ------------------------------------------------------------ precomputation

    LISTS = dict(
        ord="ord_abil",
        prod="prod_abil",
        buff="b_id",
        rally="rally_x",
        ab="ab_id",
        pass_="pass_id",
    )

    def _lists(self, d):
        """Each row's ragged lists as (start, count) columns, so rows can be added and
        reordered; the list arrays themselves are kept for the bags (orders, production,
        buffs, abilities, rally points) and the passengers."""
        u = self.u
        self.lists = {}
        for name, arr in self.LISTS.items():
            key = name.rstrip("_") + "_start"
            n_items = len(d[arr]) if arr in d.files else 0
            if key not in u:
                u[f"{name}_n"] = np.zeros(len(u["id"]), np.int32)
                u[f"{name}_s"] = np.zeros(len(u["id"]), np.int64)
                continue
            start = u.pop(key).astype(np.int64)
            u[f"{name}_s"] = start
            u[f"{name}_n"] = np.diff(np.append(start, n_items)).astype(np.int32)
        L = self.lists
        if "ord_abil" in d.files:
            L["ord"] = dict(
                abil=d["ord_abil"],
                tx=d["ord_tx"],
                ty=d["ord_ty"],
                target=d["ord_target"],
                progress=d["ord_progress"].astype(np.float32),
            )
        if "prod_abil" in d.files:
            L["prod"] = dict(
                abil=d["prod_abil"], progress=d["prod_progress"].astype(np.float32)
            )
        if "b_id" in d.files:
            L["buff"] = dict(id=d["b_id"])
        if "ab_id" in d.files:
            L["ab"] = dict(id=d["ab_id"], point=d["ab_point"])
        if "rally_x" in d.files:
            L["rally"] = dict(x=d["rally_x"], y=d["rally_y"], target=d["rally_target"])
        if "pass_id" in d.files:
            L["pass_"] = dict(id=d["pass_id"], type=d["pass_type"], hp=d["pass_hp"])

    def _row_lookup(self):
        """(frame, unit id) -> row, for resolving the links between units of one
        frame."""
        u = self.u
        key = (u["fidx"].astype(np.int64) << 32) | u["id"].astype(np.int64)
        order = np.argsort(key, kind="stable")
        return key[order], order

    def _find(self, look, fidx, ids):
        keys, order = look
        q = (np.asarray(fidx, np.int64) << 32) | np.asarray(ids, np.int64)
        if not len(keys):
            return np.full(len(q), -1, np.int64)
        i = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
        return np.where((keys[i] == q) & (np.asarray(ids) >= 0), order[i], -1)

    def _contained(self, d):
        """Rows for units inside other units, each with a pointer (parent id) to its
        container: passengers (the extraction's passenger lists: transports, bunkers,
        command centers, ...) sit at the container's position with the passenger flag;
        workers inside gas buildings, which the observation drops while they harvest,
        are bridged between their last and next sighting when their order was the gas
        building, with the in-gas flag. Add-ons get their building as parent. Rows are
        then reordered by frame."""
        u, n = self.u, len(self.u["id"])
        u["parent"] = np.full(n, -1, np.int32)
        u["inside"] = np.zeros(n, np.int8)  # 1 passenger, 2 in a gas building
        look = self._row_lookup()
        # add-ons: the building's add_on column names the add-on's id
        has = (u["add_on"] >= 0) if "add_on" in u else np.zeros(n, bool)
        if has.any():
            r = self._find(look, u["fidx"][has], u["add_on"][has])
            ok = r >= 0
            u["parent"][r[ok]] = u["id"][has][ok]
        new = []  # dicts of row columns
        # passengers
        P = self.lists.get("pass_")
        if P is not None and len(P["id"]):
            tr = np.nonzero(u["pass__n"] > 0)[0]
            cnt = u["pass__n"][tr]
            rows = np.repeat(tr, cnt)
            items = np.concatenate(
                [np.arange(s, s + c) for s, c in zip(u["pass__s"][tr], cnt)]
            )
            pid = P["id"][items].astype(np.int32)
            onmap = self._find(look, u["fidx"][rows], pid) >= 0
            rows, items, pid = rows[~onmap], items[~onmap], pid[~onmap]
            # max hp of a passenger: its own value when it was on the map, else the
            # largest seen for its type
            hpmax = self._max_hp_by_id(pid, P["type"][items])
            new.append(
                dict(
                    src=rows,
                    id=pid,
                    type=P["type"][items].astype(u["type"].dtype),
                    hp=P["hp"][items].astype(np.float32),
                    hp_max=hpmax,
                    inside=1,
                    parent=u["id"][rows],
                    **self.PASSENGER_RESET,
                )
            )
        # workers inside gas buildings
        g = self._gas_gaps(look)
        if g is not None:
            new.append(g)
        if new:
            self._append_rows(new)

    # a passenger row starts as a copy of its transport's: these columns are the
    # transport's own state
    PASSENGER_RESET = dict(
        shield=0.0,
        shield_max=0.0,
        energy=0.0,
        build=1.0,
        flags=0,
        contents=0,
        harvesters=0,
        add_on=-1,
        weapon_cooldown=0.0,
        engaged=-1,
        cargo_used=0,
        cargo_max=0,
        display=3,
        cloak=0,
        flags2=0,
        buff_remain=0,
        buff_max=0,
        ideal_harvesters=0,
        detect_range=0.0,
        radar_range=0.0,
        produce=0,
        produce_raw=0,
        progress=0.0,
        progress2=0.0,
        queue=0,
        order=0,
        order_raw=0,
        ord=0.0,
        radius=0.0,
        facing=0.0,
    )

    def _max_hp_by_id(self, ids, types):
        u = self.u
        out = np.zeros(len(ids), np.float32)
        order = np.argsort(u["id"], kind="stable")
        sid = u["id"][order]
        i = np.clip(np.searchsorted(sid, ids), 0, max(len(sid) - 1, 0))
        hit = len(sid) > 0
        if hit:
            ok = sid[i] == ids
            out[ok] = u["hp_max"][order[i[ok]]]
        miss = out <= 0
        if miss.any():
            by_type = {}
            for t, h in zip(u["type"], u["hp_max"]):
                if h > by_type.get(int(t), 0):
                    by_type[int(t)] = float(h)
            out[miss] = [by_type.get(int(t), 0.0) for t in types[miss]]
        return out

    def _gas_gaps(self, look):
        """Workers out of sight for at most GAS_GAP_LOOPS whose first order before the
        gap targeted a gas building that exists through the gap: one row per missing
        frame at the building's position."""
        u = self.u
        if "ord" not in self.lists:
            return None
        L = self.lists["ord"]
        wk = np.nonzero(np.isin(u["type"], WORKER_TYPES) & (u["ord_n"] > 0))[0]
        if not len(wk):
            return None
        tgt = L["target"][u["ord_s"][wk]]
        isgas_t = tgt >= 0
        wk, tgt = wk[isgas_t], tgt[isgas_t]
        trow = self._find(look, u["fidx"][wk], tgt)
        ok = trow >= 0
        ok[ok] = np.isin(u["type"][trow[ok]], GAS_TYPES)
        wk, tgt = wk[ok], tgt[ok]
        if not len(wk):
            return None
        # the worker's next sighting after each candidate row
        allw = np.nonzero(np.isin(u["type"], WORKER_TYPES))[0]
        key = (u["id"][allw].astype(np.int64) << 32) | u["fidx"][allw].astype(np.int64)
        order = np.argsort(key)
        skey = key[order]
        mykey = (u["id"][wk].astype(np.int64) << 32) | u["fidx"][wk].astype(np.int64)
        j = np.searchsorted(skey, mykey) + 1
        valid = j < len(skey)
        nxt = np.where(valid, skey[np.minimum(j, len(skey) - 1)], 0)
        same = valid & ((nxt >> 32) == (mykey >> 32))
        f0 = u["fidx"][wk].astype(np.int64)
        f1 = np.where(same, nxt & 0xFFFFFFFF, f0 + 1)
        gap_loops = self.loops[np.minimum(f1, self.F - 1)] - self.loops[f0]
        sel = same & (f1 > f0 + 1) & (gap_loops <= GAS_GAP_LOOPS)
        wk, tgt, f0, f1 = wk[sel], tgt[sel], f0[sel], f1[sel]
        if not len(wk):
            return None
        reps = (f1 - f0 - 1).astype(np.int64)
        src = np.repeat(wk, reps)
        fr = np.concatenate([np.arange(a + 1, b) for a, b in zip(f0, f1)])
        gid = np.repeat(tgt, reps)
        grow = self._find(look, fr, gid)
        keep = grow >= 0
        return dict(
            src=src[keep],
            fidx=fr[keep].astype(np.int32),
            x=u["x"][grow[keep]],
            y=u["y"][grow[keep]],
            inside=2,
            parent=gid[keep].astype(np.int32),
            clear_lists=True,
        )

    def _append_rows(self, new):
        """Adds rows copied from their source rows with the given columns overridden
        (lists emptied: a contained unit's orders and the like aren't observed), then
        reorders by frame.
        """
        u = self.u
        add = {k: [] for k in u}
        for spec in new:
            src = spec["src"]
            for k in u:
                col = u[k][src].copy()
                if k in spec:
                    v = spec[k]
                    col[...] = (
                        v
                        if np.isscalar(v)
                        else np.asarray(v).astype(col.dtype).reshape(col.shape)
                    )
                elif k.endswith("_n"):
                    col[:] = 0
                add[k].append(col)
        for k in u:
            u[k] = np.concatenate([u[k]] + add[k])
        order = np.argsort(u["fidx"], kind="stable")
        for k in u:
            u[k] = u[k][order]
        self.start = np.concatenate(
            [[0], np.cumsum(np.bincount(u["fidx"], minlength=self.F))]
        ).astype(np.int64)

    def _links(self):
        """Pointers resolved to rows: parent_row (container / building) and the engaged
        target's offset, per row (rows of the same frame)."""
        u = self.u
        look = self._row_lookup()
        u["parent_row"] = self._find(look, u["fidx"], u["parent"]).astype(np.int64)
        eng = (
            self._find(look, u["fidx"], u["engaged"])
            if "engaged" in u
            else np.full(len(u["id"]), -1)
        )
        ok = eng >= 0
        u["eng_dx"] = np.where(ok, u["x"][np.maximum(eng, 0)] - u["x"], 0).astype(
            np.float32
        )
        u["eng_dy"] = np.where(ok, u["y"][np.maximum(eng, 0)] - u["y"], 0).astype(
            np.float32
        )
        u["eng_ok"] = ok

    def _effects(self, d):
        if "e_frame_start" not in d.files:
            self.eff = None
            return
        self.eff = dict(
            start=d["e_frame_start"],
            type=d["e_type"].astype(np.int64),
            owner=d["e_owner"],
            x=d["e_x"],
            y=d["e_y"],
            radius=d["e_radius"].astype(np.float32),
        )

    def _grids(self, d):
        """Map layers at one cell per map unit, rows = y (verified against unit
        positions): per frame visibility (0 hidden, 1 fogged, 2 visible) and creep bits;
        static pathing, placement and terrain height."""
        W, H = self.meta["map_size"]
        self.map_wh = (int(W), int(H))
        self.vis_raw = d["vis"] if "vis" in d.files else None
        self.creep_raw = d["creep"] if "creep" in d.files else None
        unpack = lambda k: (
            np.unpackbits(d[k]).reshape(H, -1)[:, :W]
            if k in d.files
            else np.zeros((H, W), np.uint8)
        )
        static = np.zeros((4, H, W), np.uint8)
        static[0] = unpack("map_pathing_grid")
        static[1] = unpack("map_placement_grid")
        if "map_terrain_height" in d.files:
            static[2] = d["map_terrain_height"].reshape(H, W)
        static[3] = 1
        x0, y0, x1, y1 = self.meta["playable"]
        self.playable_mask = np.zeros((H, W), np.uint8)
        self.playable_mask[y0:y1, x0:x1] = 1
        self.static = static

    @property
    def grid_origin(self):
        """World (x, y) of the grid's cell (0, 0): the playable area's corner rounded
        down to a multiple of 16 (the grid is cropped to the playable area; cell
        boundaries stay on the world's 16-unit lines)."""
        x0, y0 = self.meta["playable"][:2]
        return np.array([x0 // 16 * 16, y0 // 16 * 16], np.int64)

    def grid(self, f):
        """[C_GRID, h, w] uint8 layers of frame f (height raw 0-255, the rest 0/1),
        cropped to the playable area from grid_origin. Visible, seen and creep are the
        viewer's, in the viewer's player-slot layers (the rest zero)."""
        ox, oy = self.grid_origin
        x1, y1 = self.meta["playable"][2:]
        return self._full_grid(f)[:, oy:y1, ox:x1]

    def _full_grid(self, f):
        W, H = self.map_wh
        out = np.zeros((C_GRID, H, W), np.uint8)
        vi = self.slots.index(self.viewer) if self.viewer in self.slots else None
        for i in (
            [vi] if vi is not None else range(min(len(self.slots), N_GRID_PLAYERS))
        ):
            if i is None or i >= N_GRID_PLAYERS:
                continue
            o = 3 * i
            if self.vis_raw is not None:
                v = self.vis_raw[f]
                out[o], out[o + 1] = v == 2, v > 0
            else:
                out[o], out[o + 1] = 1, 1
            if self.creep_raw is not None:
                out[o + 2] = np.unpackbits(self.creep_raw[f], axis=1)[:, :W]
        s0 = 3 * N_GRID_PLAYERS
        out[s0], out[s0 + 1], out[s0 + 2], out[s0 + 3] = self.static
        out[s0 + 4] = self.playable_mask
        return out

    def _player_extras(self, d):
        """[F, P, ...] player_common, score details (viewer only: private), units lost
        per player (from the deaths and the owners' last rows), the camera, its saved
        positions, the subgroup and the last command."""
        F, P = self.F, len(self.slots)
        self.pextra = np.zeros((F, P, C_PLAYER - N_STATS), np.float32)
        self.cam = np.zeros((F, P, 2), np.float32)
        self.cam_save = np.zeros((F, P, N_CAM_SAVES, 2), np.float32)
        self.cam_save_ok = np.zeros((F, P, N_CAM_SAVES), bool)
        self.lc_link = np.full((F, P), -1, np.int64)
        known = np.zeros(P, bool)
        if self.viewer and self.viewer in self.slots:
            vi = self.slots.index(self.viewer)
            known[vi] = True
            o = 0
            for k in PC_KEYS:
                if f"pc_{k}" in d.files:
                    self.pextra[:, vi, o] = _log(d[f"pc_{k}"])
                o += 1
            for k in SC_KEYS:
                if f"sc_{k}" in d.files:
                    self.pextra[:, vi, o] = _log(d[f"sc_{k}"])
                o += 1
            if "cam" in d.files:
                self.cam[:, vi] = d["cam"]
        base = len(PC_KEYS) + len(SC_KEYS)
        self.pextra[:, :, base + 2 : base + 4] = np.where(
            known[None, :, None], (self.cam - self.center) / self.half, 0
        )
        # units lost per player: deaths in the view, owner from the unit's last row
        # before it died
        if "dead_id" in d.files and len(d["dead_id"]):
            ds = d["dead_frame_start"]
            # frame of each death (a death after the last frame's offset counts in the
            # last frame)
            dfr = np.clip(
                np.searchsorted(ds, np.arange(len(d["dead_id"])), side="right") - 1,
                0,
                F - 1,
            )
            u = self.u
            order = np.argsort(u["id"], kind="stable")
            sid = u["id"][order]
            i = np.clip(
                np.searchsorted(sid, d["dead_id"], side="right") - 1, 0, len(sid) - 1
            )
            own = np.where(sid[i] == d["dead_id"], u["owner"][order[i]], -1)
            for pi, slot in enumerate(self.slots):
                c = np.cumsum(np.bincount(dfr[own == slot], minlength=F))
                self.pextra[:, pi, base] = _log(c.astype(np.float32), 8.0)
        self.pextra[:, :, base + 1] = known[None]
        # interface memory per player (the extraction keeps the viewer's: player axis is
        # the tracked players)
        if "cam_save" in d.files and "grp_player" in d.files:
            tracked = [int(s) for s in d["grp_player"]]
            for ti, slot in enumerate(tracked):
                if slot not in self.slots or ti >= d["cam_save"].shape[1]:
                    continue
                pi = self.slots.index(slot)
                cs = d["cam_save"][:, ti]
                self.cam_save_ok[:, pi] = np.isfinite(cs).all(-1)
                self.cam_save[:, pi] = np.nan_to_num(cs)
                if "subgroup" in d.files:
                    self.pextra[:, pi, base + 4] = d["subgroup"][:, ti] / 10.0
                if "lc_link" in d.files:
                    self.lc_link[:, pi] = d["lc_link"][:, ti]
                    age = np.maximum(self.loops - d["lc_loop"][:, ti], 0)
                    has = d["lc_link"][:, ti] >= 0
                    lx, ly = d["lc_x"][:, ti], d["lc_y"][:, ti]
                    pt = has & np.isfinite(lx)
                    self.pextra[:, pi, base + 5] = np.where(
                        has, _log(age.astype(np.float32), 8.0), 0
                    )
                    self.pextra[:, pi, base + 6] = pt
                    self.pextra[:, pi, base + 7] = np.where(
                        pt, (np.nan_to_num(lx) - self.center[0]) / self.half[0], 0
                    )
                    self.pextra[:, pi, base + 8] = np.where(
                        pt, (np.nan_to_num(ly) - self.center[1]) / self.half[1], 0
                    )
                    self.pextra[:, pi, base + 9] = (
                        has & (d["lc_target"][:, ti] >= 2)
                        if "lc_target" in d.files
                        else 0
                    )

    def set_command_families(self, family_of_link):
        """Maps the last-command ability links to action families (needs the ruleset's
        link table, which the action side holds)."""
        self.lc_family = (
            np.vectorize(
                lambda l: family_of_link(int(l)) if l >= 0 else 0, otypes=[np.int64]
            )(self.lc_link)
            if self.lc_link.size
            else np.zeros_like(self.lc_link)
        )

    def _production(self, d):
        """What each producer (structure, egg, cocoon) is producing: the first queue
        entry, its progress, the second entry's progress (non-zero only when two run at
        once, e.g. a reactor) and the queue length. Production is state (resources
        committed, a timer running); other units' orders are intent, which the state
        leaves out."""
        u, n = self.u, len(self.u["id"])
        if "prod" in self.lists:  # current extractor: production queues only
            start, count = u["prod_s"], u["prod_n"]
            abil, prog = d["prod_abil"], d["prod_progress"].astype(np.float32)
            has1, has2 = count > 0, count > 1
            first = np.where(has1, start, 0)
            second = np.where(has2, start + 1, 0)
            u["produce_raw"] = np.where(
                has1, abil[first] if len(abil) else 0, 0
            ).astype(np.int32)
            u["produce"] = np.where(
                has1, self.vocab.ability_index(abil[first]) if len(abil) else PAD, PAD
            )
            u["progress"] = np.where(has1, prog[first] if len(prog) else 0, 0).astype(
                np.float32
            )
            u["progress2"] = np.where(has2, prog[second] if len(prog) else 0, 0).astype(
                np.float32
            )
            u["queue"] = count
        else:  # older extractions: first order of every unit, producers picked here
            producer = np.isin(u["type"], list(self.vocab.producers))
            u["produce_raw"] = np.where(producer, u["order_abil"], 0).astype(np.int32)
            u["produce"] = np.where(
                producer, self.vocab.ability_index(u["order_abil"]), PAD
            )
            u["progress"] = np.where(
                producer, u.get("order_progress", np.zeros(n, np.float32)), 0
            ).astype(np.float32)
            u["progress2"] = np.zeros(n, np.float32)
            u["queue"] = np.where(producer, u["n_orders"], 0)

    def _orders(self, d):
        """The unit's first order, its current intent: the ability (vocab index) and
        C_ORDER numbers (has an order, target offset from the unit in normalized map
        units, targets a unit, progress, log queue length). Zeros for extractions
        without orders."""
        u, n = self.u, len(self.u["id"])
        u["order"] = np.zeros(n, np.int64)
        u["order_raw"] = np.zeros(n, np.int32)
        u["ord"] = np.zeros((n, C_ORDER), np.float32)
        if "ord" not in self.lists or not len(d["ord_abil"]):
            return
        start, count = u["ord_s"], u["ord_n"]
        has = count > 0
        first = np.where(has, start, 0)
        u["order"] = np.where(has, self.vocab.ability_index(d["ord_abil"][first]), PAD)
        u["order_raw"] = np.where(has, d["ord_abil"][first], 0).astype(np.int32)
        tx, ty = d["ord_tx"][first], d["ord_ty"][first]
        has_pos = has & np.isfinite(tx)
        u["ord"][:, 0] = has
        u["ord"][:, 1] = np.where(
            has_pos, (np.nan_to_num(tx) - u["x"]) / self.half[0], 0
        )
        u["ord"][:, 2] = np.where(
            has_pos, (np.nan_to_num(ty) - u["y"]) / self.half[1], 0
        )
        u["ord"][:, 3] = has & (d["ord_target"][first] >= 0)
        u["ord"][:, 4] = np.where(has, d["ord_progress"][first], 0)
        u["ord"][:, 5] = np.log1p(count) / 2

    # unit columns the model and probes read, in the narrowest type that holds them; the
    # rest (height, facing, cooldowns, ragged-list indices, ...) is dropped at load:
    # memory bounds how many games (and player views) fit in the pool
    KEEP = dict(
        id=np.int32,
        type=np.int16,
        owner=np.int8,
        x=np.float32,
        y=np.float32,
        hp=np.float32,
        hp_max=np.float32,
        shield=np.float32,
        shield_max=np.float32,
        energy=np.float32,
        build=np.float16,
        flags=np.uint8,
        contents=np.int16,
        harvesters=np.int16,
        unit_idx=np.int16,
        cat=np.int8,
        resource=np.bool_,
        produce=np.int16,
        progress=np.float32,
        progress2=np.float32,
        queue=np.int8,
        order=np.int16,
        ord=np.float32,
        produce_raw=np.int32,
        order_raw=np.int32,
        # layout 2
        fidx=np.int32,
        facing=np.float16,
        radius=np.float16,
        weapon_cooldown=np.float16,
        atk_up=np.int8,
        arm_up=np.int8,
        shd_up=np.int8,
        buff_remain=np.int16,
        buff_max=np.int16,
        ideal_harvesters=np.int16,
        cargo_used=np.int16,
        cargo_max=np.int16,
        detect_range=np.float16,
        radar_range=np.float16,
        display=np.int8,
        cloak=np.int8,
        flags2=np.uint8,
        add_on=np.int32,
        inside=np.int8,
        parent_row=np.int64,
        eng_dx=np.float32,
        eng_dy=np.float32,
        eng_ok=np.bool_,
        ord_s=np.int64,
        ord_n=np.int16,
        prod_s=np.int64,
        prod_n=np.int16,
        buff_s=np.int64,
        buff_n=np.int16,
        ab_s=np.int64,
        ab_n=np.int32,
        rally_s=np.int64,
        rally_n=np.int16,
    )

    def _slim(self):
        self.u = {
            k: self.u[k].astype(t, copy=False)
            for k, t in self.KEEP.items()
            if k in self.u
        }

    def _stats(self, d):
        """stats[f, p] = latest tracker PlayerStats row at or before frame f."""
        F, P = len(self.loops), len(self.slots)
        self.stats = np.zeros((F, P, N_STATS), np.float32)
        for pi, slot in enumerate(self.slots):
            rows = np.where(d["p_player"] == slot)[0]
            if not len(rows):
                continue
            idx = np.searchsorted(d["p_loop"][rows], self.loops, side="right") - 1
            ok = idx >= 0
            vals = np.stack([d[f"p_{k}"][rows] for k in STAT_KEYS], 1)
            feat = np.concatenate(
                [_log(vals[:, :5]), vals[:, 5:7] / 200.0, _log(vals[:, 7:])], 1
            )
            self.stats[ok, pi] = feat[idx[ok]]

    def _visibility(self, d, grid=GRID):
        """vis[f, cell] = (share of the cell visible now, share ever seen), cells
        row-major over the playable area like the encoder's grid. The observer sees
        everything."""
        F = len(self.loops)
        if "vis" not in d.files:
            self.vis = None
            return
        x0, y0, x1, y1 = self.meta["playable"]
        v = d["vis"][:, y0:y1, x0:x1]  # map_state rows are y
        gy = np.minimum((np.arange(y1 - y0) * grid) // (y1 - y0), grid - 1)
        gx = np.minimum((np.arange(x1 - x0) * grid) // (x1 - x0), grid - 1)
        cell = (gy[:, None] * grid + gx[None, :]).ravel()
        size = np.bincount(cell, minlength=grid * grid).astype(np.float32)
        self.vis = np.zeros((F, grid * grid, C_VIS), np.float32)
        for f in range(F):
            vf = v[f].ravel()
            self.vis[f, :, 0] = np.bincount(cell, vf == 2, grid * grid) / size
            self.vis[f, :, 1] = np.bincount(cell, vf > 0, grid * grid) / size

    def _upgrades(self, d):
        """ups[f, p, k] = 1 once player p has completed upgrade k."""
        F, P = len(self.loops), len(self.slots)
        self.ups = np.zeros((F, P, max(self.vocab.n_upgrades, 1)), np.float32)
        for loop, slot, name in zip(d["up_loop"], d["up_player"], d["up_name"]):
            k = self.vocab.upgrade_slot.get(str(name))
            if slot in self.slots and k is not None:
                self.ups[
                    np.searchsorted(self.loops, loop) :, self.slots.index(slot), k
                ] = 1.0

    # ------------------------------------------------------------ ownership

    def owner_map(
        self, view="observer", me_slot=None, augment=False, rng=None, codes=None
    ):
        """slot -> (team code, player code, relation) for one sample.

        Without augment, codes follow team order and slot order; with augment they're
        drawn at random from the pool, so a code never means anything beyond "same
        team" / "same player" within this sample. codes = (team permutation, player
        permutation) uses given draws: samples that are compared with each other (a
        batch with in-batch negatives) must share them, or the codes alone tell the
        samples apart.
        """
        rng = rng or np.random
        teams = sorted({p["team"] for p in self.players})
        if codes is not None:
            team_codes, player_codes = (
                codes[0][: len(teams)],
                codes[1][: len(self.slots)],
            )
        else:
            team_codes = (
                rng.permutation(N_CODES)[: len(teams)]
                if augment
                else np.arange(len(teams))
            )
            player_codes = (
                rng.permutation(N_CODES)[: len(self.slots)]
                if augment
                else np.arange(len(self.slots))
            )
        my_team = next((p["team"] for p in self.players if p["slot"] == me_slot), None)
        out = {}
        for i, p in enumerate(self.players):
            rel = REL_NONE
            if view == "player":
                rel = (
                    REL_ME
                    if p["slot"] == me_slot
                    else REL_ALLY if p["team"] == my_team else REL_ENEMY
                )
            out[p["slot"]] = (
                int(team_codes[teams.index(p["team"])]),
                int(player_codes[i]),
                rel,
            )
        return out

    # ------------------------------------------------------------ one frame

    def entity_features(self, rows, owners, prefix="ent"):
        """Encoder inputs for the given unit rows (any rows, any frames): what an entity
        token is made of. Shared by frames and by actions that point at units."""
        u = self.u
        hp, hpm = u["hp"][rows], u["hp_max"][rows]
        sh, shm = u["shield"][rows], u["shield_max"][rows]
        flags = u["flags"][rows]
        num = (
            np.stack(
                [
                    np.where(hpm > 0, hp / np.maximum(hpm, 1), 0),
                    np.where(shm > 0, sh / np.maximum(shm, 1), 0),
                    u["energy"][rows] / 200.0,
                    _log(hpm, 8.0),
                    _log(shm, 8.0),
                    u["build"][rows].astype(np.float32),
                    *[
                        ((flags >> k) & 1).astype(np.float32) for k in range(7)
                    ],  # flying burrowed cloaked halluc snapshot powered selected
                    u["progress"][rows],
                    u["progress2"][rows],
                    u["queue"][rows] / 5.0,
                    _log(u["contents"][rows].astype(np.float32), 8.0),
                    u["harvesters"][rows] / 16.0,
                    *self._layout2_numbers(rows),
                ],
                1,
            )
            .astype(np.float32)
            .reshape(-1, C_ENTITY)
        )
        neutral = (CODE_NEUTRAL, CODE_NEUTRAL, REL_NONE)
        own = np.array(
            [owners.get(int(s), neutral) for s in u["owner"][rows]], np.int64
        ).reshape(-1, 3)
        xy = np.stack([u["x"][rows], u["y"][rows]], 1).reshape(-1, 2)
        return {
            f"{prefix}_type": u["unit_idx"][rows].astype(np.int64),
            # raw API ids, for the ruleset encoder (worldmodel/ruleset.py)
            f"{prefix}_rtype": u["type"][rows].astype(np.int64),
            f"{prefix}_rproduce": u["produce_raw"][rows].astype(np.int64),
            f"{prefix}_rorder": u["order_raw"][rows].astype(np.int64),
            f"{prefix}_team": own[:, 0],
            f"{prefix}_pcode": own[:, 1],
            f"{prefix}_rel": own[:, 2],
            f"{prefix}_produce": u["produce"][rows].astype(np.int64),
            f"{prefix}_order": u["order"][rows].astype(np.int64),
            f"{prefix}_ord": u["ord"][rows].reshape(-1, C_ORDER),
            f"{prefix}_xy": self.normalize_xy(xy),
            f"{prefix}_pos": (xy - self.center).astype(
                np.float32
            ),  # map units from the playable center
            f"{prefix}_num": num,
            f"{prefix}_groups": np.zeros((len(rows), C_GROUPS), np.float32),
        }  # frame() fills the real bits

    def _layout2_numbers(self, rows):
        u = self.u
        g = lambda k, default=0: (
            u[k][rows].astype(np.float32)
            if k in u
            else np.full(len(rows), default, np.float32)
        )
        facing = g("facing")
        disp, cloak, f2 = (
            g("display", 1),
            g("cloak"),
            u["flags2"][rows] if "flags2" in u else np.zeros(len(rows), np.uint8),
        )
        bmax = g("buff_max")
        inside = g("inside")
        src = (
            u["src"][rows] if "src" in u else np.full(len(rows), self.viewer)
        )  # the view a row came from
        viewer_own = (
            (u["owner"][rows] == src)
            & (src > 0)
            & bool(self.meta.get("abilities_queried", False))
        )
        return [
            np.sin(facing),
            np.cos(facing),
            g("radius") / 4.0,
            np.clip(g("weapon_cooldown"), 0, None) / 32.0,
            g("eng_ok"),
            np.clip(g("eng_dx") / 16.0, -2, 2),
            np.clip(g("eng_dy") / 16.0, -2, 2),
            g("atk_up") / 3.0,
            g("arm_up") / 3.0,
            g("shd_up") / 3.0,
            np.where(bmax > 0, g("buff_remain") / np.maximum(bmax, 1), 0),
            _log(bmax, 8.0),
            g("ideal_harvesters") / 16.0,
            g("cargo_used") / 8.0,
            g("cargo_max") / 8.0,
            g("detect_range") / 12.0,
            g("radar_range") / 32.0,
            disp == 1,
            disp == 2,
            disp == 3,
            cloak == 0,
            cloak == 1,
            cloak == 2,
            cloak == 3,
            (f2 & 1) > 0,
            (f2 & 2) > 0,
            (f2 & 4) > 0,
            inside == 1,
            inside == 2,
            viewer_own,
            g("add_on", -1) >= 0,
        ]

    def bags(self, rows):
        """List-valued state of the given rows, flattened: {name: (row index into rows,
        item id, item features)}. Item ids: ability vocab index (orders, production),
        raw buff id, raw ability id (available abilities), 0 (rally points)."""
        u, L, out = self.u, self.lists, {}
        rows = np.asarray(rows)

        def items(name):
            n = u[f"{name}_n"][rows].astype(np.int64)
            s = u[f"{name}_s"][rows]
            ri = np.repeat(np.arange(len(rows)), n)
            if not len(ri):
                return ri, ri, ri
            off = np.arange(len(ri)) - np.repeat(np.cumsum(n) - n, n)
            return ri, np.repeat(s, n) + off, off

        if "ord" in L:
            ri, it, pos = items("ord")
            o = L["ord"]
            tx, ty = o["tx"][it] if len(it) else np.zeros(0), (
                o["ty"][it] if len(it) else np.zeros(0)
            )
            hasp = np.isfinite(tx)
            ux, uy = u["x"][rows[ri]], u["y"][rows[ri]]
            feat = (
                np.stack(
                    [
                        np.where(
                            hasp,
                            np.clip((np.nan_to_num(tx) - ux) / self.half[0], -2, 2),
                            0,
                        ),
                        np.where(
                            hasp,
                            np.clip((np.nan_to_num(ty) - uy) / self.half[1], -2, 2),
                            0,
                        ),
                        (o["target"][it] >= 0) if len(it) else np.zeros(0),
                        o["progress"][it] if len(it) else np.zeros(0),
                        pos / 5.0,
                    ],
                    1,
                )
                .astype(np.float32)
                .reshape(-1, BAGS["ord"])
            )
            out["ord"] = (
                ri,
                (
                    self.vocab.ability_index(o["abil"][it])
                    if len(it)
                    else np.zeros(0, np.int64)
                ),
                feat,
            )
        if "prod" in L:
            ri, it, pos = items("prod")
            o = L["prod"]
            feat = (
                np.stack([o["progress"][it] if len(it) else np.zeros(0), pos / 5.0], 1)
                .astype(np.float32)
                .reshape(-1, BAGS["prod"])
            )
            out["prod"] = (
                ri,
                (
                    self.vocab.ability_index(o["abil"][it])
                    if len(it)
                    else np.zeros(0, np.int64)
                ),
                feat,
            )
        if "buff" in L:
            ri, it, pos = items("buff")
            out["buff"] = (
                ri,
                np.clip(L["buff"]["id"][it], 0, N_BUFF_IDS - 1).astype(np.int64),
                np.ones((len(ri), 1), np.float32),
            )
        if "ab" in L:
            ri, it, pos = items("ab")
            o = L["ab"]
            out["abil"] = (
                ri,
                np.clip(o["id"][it], 0, N_RAW_ABILITY_IDS - 1).astype(np.int64),
                o["point"][it].astype(np.float32).reshape(-1, 1),
            )
        if "rally" in L:
            ri, it, pos = items("rally")
            o = L["rally"]
            ux, uy = u["x"][rows[ri]], u["y"][rows[ri]]
            feat = (
                np.stack(
                    [
                        (
                            np.clip((o["x"][it] - ux) / self.half[0], -2, 2)
                            if len(it)
                            else np.zeros(0)
                        ),
                        (
                            np.clip((o["y"][it] - uy) / self.half[1], -2, 2)
                            if len(it)
                            else np.zeros(0)
                        ),
                        (o["target"][it] >= 0) if len(it) else np.zeros(0),
                    ],
                    1,
                )
                .astype(np.float32)
                .reshape(-1, BAGS["rally"])
            )
            out["rally"] = (ri, np.zeros(len(ri), np.int64), feat)
        for k in BAGS:
            out.setdefault(
                k,
                (
                    np.zeros(0, np.int64),
                    np.zeros(0, np.int64),
                    np.zeros((0, BAGS[k]), np.float32),
                ),
            )
        return out

    def normalize_xy(self, xy):
        """Map coordinates -> [-1, 1] over the playable area."""
        return ((np.asarray(xy, np.float32) - self.center) / self.half).astype(
            np.float32
        )

    def rows_of(self, f, ids):
        """Unit rows in frame f for state unit ids; -1 where the unit isn't in that
        frame."""
        if f not in self._rows_cache:
            if len(self._rows_cache) > 512:
                self._rows_cache.clear()
            a, b = self.start[f], self.start[f + 1]
            fid = self.u["id"][a:b]
            order = np.argsort(fid, kind="stable")
            self._rows_cache[f] = (fid[order], order + a)
        sid, srow = self._rows_cache[f]
        ids = np.asarray(ids, np.int64)
        if not len(sid):
            return np.full(len(ids), -1, np.int64)
        i = np.clip(np.searchsorted(sid, ids), 0, len(sid) - 1)
        return np.where(sid[i] == ids, srow[i], -1)

    def _sources(self, rows, owners, ent, eff, f):
        """Each token's source view as that viewer's player code (CODE_NEUTRAL: no
        single viewer); in the combined view, relations as the source viewer sees
        them."""
        code = lambda slot: owners[slot][1] if slot in owners else CODE_NEUTRAL
        if "src" not in self.u or self.eff is None:
            n_e = len(eff["eff_type"])
            return np.full(len(rows), code(self.viewer), np.int64), np.full(
                n_e, code(self.viewer), np.int64
            )
        src = self.u["src"][rows].astype(np.int64)
        ent["ent_rel"] = self._relation(self.u["owner"][rows], src)
        esrc = self.eff["src"][self.eff["start"][f] : self.eff["start"][f + 1]].astype(
            np.int64
        )
        eff["eff_rel"] = self._relation(
            self.eff["owner"][self.eff["start"][f] : self.eff["start"][f + 1]], esrc
        )
        return np.array([code(int(x)) for x in src], np.int64), np.array(
            [code(int(x)) for x in esrc], np.int64
        )

    def _relation(self, owner, src):
        team = {p["slot"]: p["team"] for p in self.players}
        out = np.full(len(owner), REL_NONE, np.int64)
        for i, (o, sv) in enumerate(zip(owner.tolist(), src.tolist())):
            if o in team and sv in team:
                out[i] = (
                    REL_ME
                    if o == sv
                    else REL_ALLY if team[o] == team[sv] else REL_ENEMY
                )
        return out

    def groups(self, f, rows):
        """[len(rows), C_GROUPS] membership bits of the frame's units in control groups
        0-9 and the active selection (10), every player's (a unit is only in its
        owner's), zeros without data.
        """
        out = np.zeros((len(rows), C_GROUPS), np.float32)
        if self.grp is None:
            return out
        start, ids, P = self.grp
        a = self.start[f]
        for k in range(P):
            base = (f * P + k) * C_GROUPS
            for g in range(C_GROUPS):
                members = ids[start[base + g] : start[base + g + 1]]
                if len(members):
                    r = self.rows_of(f, members) - a
                    out[r[r >= 0], g] = 1.0
        return out

    def frame(self, f, owners):
        u, a, b = self.u, self.start[f], self.start[f + 1]
        rows = np.arange(a, b)  # hook for views that hide things (fog): drop rows here

        p_idx = [i for i, s in enumerate(self.slots) if s in owners]
        p_own = np.array([owners[self.slots[i]] for i in p_idx], np.int64).reshape(
            -1, 3
        )
        ply_num = np.concatenate([self.stats[f, p_idx], self.pextra[f, p_idx]], -1)
        ply_ups = self.ups[f, p_idx]
        # a player sees only their own economy and upgrades (the combined view:
        # everyone's)
        if self.viewer:
            hide = np.array([self.slots[i] != self.viewer for i in p_idx])
            ply_num[:, :N_STATS] = np.where(hide[:, None], 0, ply_num[:, :N_STATS])
            ply_ups = np.where(hide[:, None], 0, ply_ups)
        parent = u["parent_row"][rows] - a
        parent = np.where((parent >= 0) & (parent < len(rows)), parent, -1)
        if self.eff is not None:
            e0, e1 = self.eff["start"][f], self.eff["start"][f + 1]
            neutral = (CODE_NEUTRAL, CODE_NEUTRAL, REL_NONE)
            eown = np.array(
                [owners.get(int(s), neutral) for s in self.eff["owner"][e0:e1]],
                np.int64,
            ).reshape(-1, 3)
            exy = np.stack([self.eff["x"][e0:e1], self.eff["y"][e0:e1]], 1).reshape(
                -1, 2
            )
            eff = dict(
                eff_type=np.clip(self.eff["type"][e0:e1], 0, N_EFFECT_IDS - 1),
                eff_team=eown[:, 0],
                eff_pcode=eown[:, 1],
                eff_rel=eown[:, 2],
                eff_xy=self.normalize_xy(exy),
                eff_pos=(exy - self.center).astype(np.float32),
                eff_num=(self.eff["radius"][e0:e1] / 8.0)
                .astype(np.float32)
                .reshape(-1, 1),
            )
        else:
            eff = dict(
                eff_type=np.zeros(0, np.int64),
                eff_team=np.zeros(0, np.int64),
                eff_pcode=np.zeros(0, np.int64),
                eff_rel=np.zeros(0, np.int64),
                eff_xy=np.zeros((0, 2), np.float32),
                eff_pos=np.zeros((0, 2), np.float32),
                eff_num=np.zeros((0, 1), np.float32),
            )
        fam = (
            self.lc_family[f, p_idx]
            if self.lc_family is not None
            else np.zeros(len(p_idx), np.int64)
        )
        cam = self.cam[f, p_idx]
        cam_ok = np.array([self.slots[i] in self.known for i in p_idx], bool)
        vis = (
            self.vis[f]
            if self.vis is not None
            else np.ones((GRID * GRID, C_VIS), np.float32)
        )
        ent = {**self.entity_features(rows, owners), "ent_groups": self.groups(f, rows)}
        ent["ent_src"], eff["eff_src"] = self._sources(rows, owners, ent, eff, f)
        return dict(
            **ent,
            ent_cat=u["cat"][rows],
            ent_resource=u["resource"][rows],
            ply_race=np.array(
                [
                    self.vocab.race.get(
                        self.players[i]["race"], self.vocab.race["Random"]
                    )
                    for i in p_idx
                ],
                np.int64,
            ),
            ply_team=p_own[:, 0],
            ply_pcode=p_own[:, 1],
            ply_rel=p_own[:, 2],
            ply_num=ply_num,
            ply_ups=ply_ups,
            ply_cam=np.where(
                cam_ok[:, None], (cam - self.center) / self.half, 0
            ).astype(np.float32),
            ply_cam_ok=cam_ok,
            ply_camsave=((self.cam_save[f, p_idx] - self.center) / self.half).astype(
                np.float32
            ),
            ply_camsave_ok=self.cam_save_ok[f, p_idx],
            ply_lcfam=fam.astype(np.int64),
            ply_gridok=np.array(
                [
                    i < N_GRID_PLAYERS
                    and (
                        self.slots[i] in self.known
                        or not self.viewer
                        and "src" not in u
                    )
                    for i in p_idx
                ],
                bool,
            ),  # whose map layers the grid holds
            ent_parent=parent.astype(np.int64),
            grid_origin=self.grid_origin.astype(np.float32),
            **eff,
            bags=self.bags(rows),
            grid=self.grid(f),
            center=self.center,
            # playable half-extents, map units (the grid latent's geometry)
            half=self.half,
            vis=vis,
            ruleset=self.meta.get("ruleset", ""),
            ply_upnames=[
                [self.vocab.upgrade_names[k] for k in np.nonzero(ply_ups[i])[0]]
                for i in range(len(p_idx))
            ],
            time=np.float32(self.loops[f] / 22.4),
            format=self.format,
            cells=self.half,  # map cells per normalized unit (x, y)
        )


GROUPS = (
    (
        "ent",
        (
            "type",
            "rtype",
            "rproduce",
            "rorder",
            "team",
            "pcode",
            "rel",
            "produce",
            "order",
            "ord",
            "xy",
            "num",
            "groups",
            "parent",
            "pos",
            "src",
        ),
    ),
    (
        "ply",
        (
            "race",
            "team",
            "pcode",
            "rel",
            "num",
            "ups",
            "cam",
            "cam_ok",
            "camsave",
            "camsave_ok",
            "lcfam",
            "gridok",
        ),
    ),
    ("eff", ("type", "team", "pcode", "rel", "xy", "pos", "num", "src")),
)
PAD_VALUES = {"ent_parent": -1}


def collate(frames):
    """Pads a list of frame dicts into a batch; <group>_mask is True for real tokens."""
    import torch

    out = {}
    for group, keys in GROUPS:
        lens = [len(f[f"{group}_{keys[0]}"]) for f in frames]
        n = max(max(lens), 1)
        mask = np.zeros((len(frames), n), bool)
        for i, L in enumerate(lens):
            mask[i, :L] = True
        out[f"{group}_mask"] = torch.from_numpy(mask)
        for k in keys:
            if f"{group}_{k}" not in frames[0]:
                continue
            arrs = [f[f"{group}_{k}"] for f in frames]
            buf = np.full(
                (len(frames), n) + arrs[0].shape[1:],
                PAD_VALUES.get(f"{group}_{k}", 0),
                arrs[0].dtype,
            )
            for i, a in enumerate(arrs):
                buf[i, : len(a)] = a
            out[f"{group}_{k}"] = torch.from_numpy(buf)
    if "bags" in frames[0]:  # list-valued entity state, flat: bag_<k>_ent = b * N + row
        N = out["ent_mask"].shape[1]
        for k, c in BAGS.items():
            ents, ids, feats = [], [], []
            for i, f in enumerate(frames):
                ri, iid, ft = f["bags"][k]
                ents.append(i * N + ri)
                ids.append(iid)
                feats.append(ft)
            out[f"bag_{k}_ent"] = torch.from_numpy(
                np.concatenate(ents).astype(np.int64)
            )
            out[f"bag_{k}_id"] = torch.from_numpy(np.concatenate(ids).astype(np.int64))
            out[f"bag_{k}_feat"] = torch.from_numpy(
                np.concatenate(feats).astype(np.float32).reshape(-1, c)
            )
    if (
        "grid" in frames[0]
    ):  # map layers on one canvas: one cell per map unit, padded to a multiple of 16
        Hc = max(16, int(np.ceil(max(f["grid"].shape[1] for f in frames) / 16) * 16))
        Wc = max(16, int(np.ceil(max(f["grid"].shape[2] for f in frames) / 16) * 16))
        canvas = np.zeros((len(frames), C_GRID, Hc, Wc), np.float32)
        for i, f in enumerate(frames):
            g = f["grid"]
            canvas[i, :, : g.shape[1], : g.shape[2]] = g
        hi = GRID_CHANNELS.index("height")
        canvas[:, hi] /= 255.0
        out["grid"] = torch.from_numpy(canvas)
        out["grid_origin"] = torch.from_numpy(
            np.stack([f["grid_origin"] for f in frames]).astype(np.float32)
        )
        out["center"] = torch.from_numpy(
            np.stack([f["center"] for f in frames]).astype(np.float32)
        )
        out["half"] = torch.from_numpy(
            np.stack([f["half"] for f in frames]).astype(np.float32)
        )
    out["time"] = torch.tensor([f["time"] for f in frames], dtype=torch.float32)
    out["format"] = torch.from_numpy(np.stack([f["format"] for f in frames]))
    out["cells"] = torch.from_numpy(np.stack([f["cells"] for f in frames]))
    out["vis"] = torch.from_numpy(np.stack([f["vis"] for f in frames]))
    out["ruleset"] = [
        f["ruleset"] for f in frames
    ]  # per sample (strings; left alone by data.to)
    out["ply_upnames"] = [
        f["ply_upnames"] for f in frames
    ]  # per sample, per player: completed upgrade names
    if "win_center" in frames[0]:  # same number of windows per frame
        out["win_center"] = torch.from_numpy(
            np.stack([f["win_center"] for f in frames])
        )
        out["win_size"] = torch.from_numpy(np.stack([f["win_size"] for f in frames]))
    return out


class CombinedFrames(GameFrames):
    """The full-information view of a 1v1 game: its two player views side by side (no
    observer extraction, no merging). Every row and effect of both views is kept, with
    u["src"] / eff["src"] = the slot whose view it came from: a unit both players see is
    two rows, each as that player sees it. Unit ids stay per view (the second view's are
    offset), so each row's future is its own view's. A replay pointer resolves to the
    unit's row in either view (alias: the same engine tag in the other view). Players:
    both players' private parts from their own views. Grid: each view's layers in its
    player's slots.

    a, b: the two GameFrames (same frames)."""

    def __init__(self, a, b):
        assert np.array_equal(a.loops, b.loops), "the two views' frames differ"
        self.path, self.vocab = a.path, a.vocab
        self.meta = dict(a.meta, perspective=0)
        self.loops, self.step, self.F = a.loops, a.step, a.F
        self.players, self.slots, self.viewer = a.players, a.slots, 0
        self.known = {a.viewer, b.viewer}
        self.center, self.half, self.format = a.center, a.half, a.format
        self.map_wh, self.static, self.playable_mask = (
            a.map_wh,
            a.static,
            a.playable_mask,
        )
        self.parts = (a, b)
        self.cmd = None
        self._rows_cache = {}
        self.id_off = int(
            max(len(a.unit_tag), a.u["id"].max() + 1 if len(a.u["id"]) else 0)
        )
        # rows: both views', frame by frame; the second view's ids, list starts and
        # parent rows offset
        u = {}
        na = len(a.u["id"])
        for k in a.u:
            pb = b.u[k]
            if k == "id":
                pb = pb.astype(np.int64) + self.id_off
            elif k.endswith("_s") and k[:-2] in a.lists:
                pb = pb + len(next(iter(a.lists[k[:-2]].values())))
            elif k == "parent_row":
                pb = np.where(pb >= 0, pb + na, -1)
            u[k] = np.concatenate([a.u[k], pb.astype(a.u[k].dtype, copy=False)])
        u["src"] = np.concatenate(
            [np.full(na, a.viewer, np.int8), np.full(len(b.u["id"]), b.viewer, np.int8)]
        )
        order = np.argsort(u["fidx"], kind="stable")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        u = {k: v[order] for k, v in u.items()}
        u["parent_row"] = np.where(
            u["parent_row"] >= 0, inv[np.maximum(u["parent_row"], 0)], -1
        )
        self.u = u
        self.start = np.concatenate(
            [[0], np.cumsum(np.bincount(u["fidx"], minlength=self.F))]
        ).astype(np.int64)
        self.lists = {
            name: {
                k: np.concatenate([a.lists[name][k], b.lists[name][k]])
                for k in a.lists[name]
            }
            for name in a.lists
            if name in b.lists
        }
        # engine tags: id -> tag over both id ranges; alias: a unit's id in the other
        # view
        tag = np.zeros(self.id_off + len(b.unit_tag), np.uint64)
        tag[: len(a.unit_tag)] = a.unit_tag
        tag[self.id_off :] = b.unit_tag
        self.unit_tag = tag
        order_b = np.argsort(b.unit_tag)
        sb = b.unit_tag[order_b]
        j = np.clip(np.searchsorted(sb, a.unit_tag), 0, max(len(sb) - 1, 0))
        a2b = (
            np.where(
                (sb[j] == a.unit_tag) & (a.unit_tag > 0), order_b[j] + self.id_off, -1
            )
            if len(sb)
            else np.full(len(a.unit_tag), -1)
        )
        self.alias = np.full(len(tag), -1, np.int64)
        self.alias[: len(a.unit_tag)] = a2b
        ok = a2b >= 0
        self.alias[a2b[ok]] = np.nonzero(ok)[0]
        # effects: both views', marked with their source
        if a.eff is not None and b.eff is not None:
            ef = np.concatenate(
                [np.repeat(np.arange(self.F), np.diff(g.eff["start"])) for g in (a, b)]
            )
            o = np.argsort(ef, kind="stable")
            self.eff = {
                k: np.concatenate([a.eff[k], b.eff[k]])[o]
                for k in a.eff
                if k != "start"
            }
            self.eff["src"] = np.concatenate(
                [
                    np.full(len(a.eff["type"]), a.viewer, np.int8),
                    np.full(len(b.eff["type"]), b.viewer, np.int8),
                ]
            )[o]
            self.eff["start"] = np.concatenate(
                [[0], np.cumsum(np.bincount(ef, minlength=self.F))]
            ).astype(np.int64)
        else:
            self.eff = None
        # players: everyone's tracker stats and upgrades; private parts from the owner's
        # view
        self.stats, self.ups = a.stats, np.maximum(a.ups, b.ups)
        self.pextra, self.cam = a.pextra.copy(), a.cam.copy()
        self.cam_save, self.cam_save_ok, self.lc_link = (
            a.cam_save.copy(),
            a.cam_save_ok.copy(),
            a.lc_link.copy(),
        )
        bi = self.slots.index(b.viewer)
        for arr, src in (
            (self.pextra, b.pextra),
            (self.cam, b.cam),
            (self.cam_save, b.cam_save),
            (self.cam_save_ok, b.cam_save_ok),
            (self.lc_link, b.lc_link),
        ):
            arr[:, bi] = src[:, bi]
        self.lc_family = None
        self.vis = (
            np.maximum(a.vis, b.vis)
            if a.vis is not None and b.vis is not None
            else None
        )
        # control groups and selections: both players' (the tracker axis is the views'
        # tracked players)
        self.grp = self._join_groups(a, b)

    def _join_groups(self, a, b):
        if a.grp is None or b.grp is None:
            return None
        G = C_GROUPS
        seg_ids, starts = [], [0]
        for f in range(self.F):
            for i, g in enumerate((a, b)):
                start, ids, P = g.grp
                for k in range(P):
                    base = (f * P + k) * G
                    seg = ids[start[base] : start[base + G]].astype(np.int64) + (
                        self.id_off if i else 0
                    )
                    seg_ids.append(seg)
                    starts.extend(
                        (
                            start[base + 1 : base + G + 1] - start[base] + starts[-1]
                        ).tolist()
                    )
        P = a.grp[2] + b.grp[2]
        return (
            np.asarray(starts, np.int64),
            (np.concatenate(seg_ids) if seg_ids else np.zeros(0, np.int64)),
            P,
        )

    def _full_grid(self, f):
        out = self.parts[0]._full_grid(f)
        bi = self.slots.index(self.parts[1].viewer)
        if bi < N_GRID_PLAYERS:
            out[3 * bi : 3 * bi + 3] = self.parts[1]._full_grid(f)[3 * bi : 3 * bi + 3]
        return out

    def cid(self, i, ids):
        """View i's unit ids -> combined ids (-1 stays -1)."""
        ids = np.asarray(ids, np.int64)
        return np.where(ids >= 0, ids + (self.id_off if i else 0), -1)
