"""Turns an extracted frame into encoder inputs.

A frame is what physically exists at one instant (no orders, no velocity: motion
and intent are for the predictor to infer across frames and from the action stream).
It becomes one vector per thing, of two kinds:

    entities  every unit, structure and neutral thing (mineral fields and geysers
              with what's left in them, rocks, towers)
    players   one per player: bank, income, workers, supply, army value, race, upgrades

plus frame-level inputs (game time, game format) that go into the encoder's CLS, and per
grid cell what the viewer can see (share of the cell visible now / ever seen; all 1 for the
observer).

A player's view (extract_states.py --views players) shows what that player knows: fog,
snapshots of remembered structures, allies' units without their private details. Tracker
stats and upgrades are kept for the viewer only (other players' economy isn't on screen).

Ownership is a role, never an identity, and works for any game format:
    team code    random per sample from a pool of 8 (same code = same team)
    player code  random per sample from a pool of 8 (same code = same player)
    relation     "player" view only: me / ally / enemy (observer view: none)
Neutral things get their own team and player code.
"""

import json

import numpy as np

from .vocab import PAD

N_CODES = 8                      # team / player codes per game (max 8 players)
CODE_NEUTRAL = N_CODES           # team and player code for neutral things
N_CODE_EMB = N_CODES + 1
REL_NONE, REL_ME, REL_ALLY, REL_ENEMY = 0, 1, 2, 3
N_REL = 4
NEUTRAL_PLAYER = 16              # engine owner id of neutral units

C_ENTITY, C_PLAYER, C_FORMAT, C_VIS = 17, 9, 4, 2
C_ORDER = 6  # the entity's current order: has one, target offset (x, y), targets a unit, progress, queue length
GRID = 16                        # the encoder's grid (G x G over the playable area)
STAT_KEYS = ("minerals", "vespene", "mineral_rate", "vespene_rate", "workers", "food_used", "food_made",
             "army_minerals", "army_vespene")


def _log(x, scale=10.0):
    return np.log1p(np.maximum(x, 0)) / scale


class GameFrames:
    """One extracted replay, held in memory, served frame by frame."""

    def __init__(self, path, vocab):
        d = np.load(path)
        self.path = path
        self.vocab = vocab
        self.meta = json.loads(str(d["meta"]))
        self.loops = d["frame_loop"]
        self.start = d["frame_start"]
        self.step = int(np.median(np.diff(self.loops))) if len(self.loops) > 1 else 16  # loops per frame
        self.unit_tag = d["unit_tag"] if "unit_tag" in d.files else None  # state unit id -> engine tag
        self._rows_cache = {}
        u = self.u = {k[2:]: d[k] for k in d.files if k.startswith("u_")}
        u["unit_idx"] = vocab.unit_index(u["type"])
        u["cat"] = vocab.category(u["type"])
        u["resource"] = (u["owner"] == NEUTRAL_PLAYER) & (u["contents"] > 0)
        x0, y0, x1, y1 = self.meta["playable"]
        self.center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
        self.half = np.array([(x1 - x0) / 2, (y1 - y0) / 2], np.float32)
        self._production(d)
        self._orders(d)
        self._slim()
        self.players = self.meta["players"]
        self.slots = [p["slot"] for p in self.players]
        self._stats(d)
        self._upgrades(d)
        sizes = sorted((sum(p["team"] == t for p in self.players) for t in {p["team"] for p in self.players}),
                       reverse=True)
        self.format = np.array([len(sizes) / 4, sizes[0] / 4, sizes[-1] / 4, len(self.players) / 8], np.float32)
        self.viewer = int(self.meta.get("perspective", 0))  # 0 = observer, else the player's slot
        self._visibility(d)

    def __len__(self):
        return len(self.loops)

    # ------------------------------------------------------------ precomputation

    def _production(self, d):
        """What each producer (structure, egg, cocoon) is producing: the first queue entry,
        its progress, the second entry's progress (non-zero only when two run at once, e.g.
        a reactor) and the queue length. Production is state (resources committed, a timer
        running); other units' orders are intent, which the state leaves out."""
        u, n = self.u, len(self.u["id"])
        if "prod_start" in u:  # current extractor: production queues only
            start = u["prod_start"]
            count = np.diff(np.append(start, len(d["prod_abil"])))
            abil, prog = d["prod_abil"], d["prod_progress"].astype(np.float32)
            has1, has2 = count > 0, count > 1
            first = np.where(has1, start, 0)
            second = np.where(has2, start + 1, 0)
            u["produce_raw"] = np.where(has1, abil[first] if len(abil) else 0, 0).astype(np.int32)
            u["produce"] = np.where(has1, self.vocab.ability_index(abil[first]) if len(abil) else PAD, PAD)
            u["progress"] = np.where(has1, prog[first] if len(prog) else 0, 0).astype(np.float32)
            u["progress2"] = np.where(has2, prog[second] if len(prog) else 0, 0).astype(np.float32)
            u["queue"] = count
        else:  # older extractions: first order of every unit, producers picked here
            producer = np.isin(u["type"], list(self.vocab.producers))
            u["produce_raw"] = np.where(producer, u["order_abil"], 0).astype(np.int32)
            u["produce"] = np.where(producer, self.vocab.ability_index(u["order_abil"]), PAD)
            u["progress"] = np.where(producer, u.get("order_progress", np.zeros(n, np.float32)), 0).astype(np.float32)
            u["progress2"] = np.zeros(n, np.float32)
            u["queue"] = np.where(producer, u["n_orders"], 0)

    def _orders(self, d):
        """The unit's first order, its current intent: the ability (vocab index) and C_ORDER
        numbers (has an order, target offset from the unit in normalized map units, targets a
        unit, progress, log queue length). Zeros for extractions without orders."""
        u, n = self.u, len(self.u["id"])
        u["order"] = np.zeros(n, np.int64)
        u["order_raw"] = np.zeros(n, np.int32)
        u["ord"] = np.zeros((n, C_ORDER), np.float32)
        if "ord_start" not in u or "ord_abil" not in d.files or not len(d["ord_abil"]):
            return
        start = u["ord_start"]
        count = np.diff(np.append(start, len(d["ord_abil"])))
        has = count > 0
        first = np.where(has, start, 0)
        u["order"] = np.where(has, self.vocab.ability_index(d["ord_abil"][first]), PAD)
        u["order_raw"] = np.where(has, d["ord_abil"][first], 0).astype(np.int32)
        tx, ty = d["ord_tx"][first], d["ord_ty"][first]
        has_pos = has & np.isfinite(tx)
        u["ord"][:, 0] = has
        u["ord"][:, 1] = np.where(has_pos, (np.nan_to_num(tx) - u["x"]) / self.half[0], 0)
        u["ord"][:, 2] = np.where(has_pos, (np.nan_to_num(ty) - u["y"]) / self.half[1], 0)
        u["ord"][:, 3] = has & (d["ord_target"][first] >= 0)
        u["ord"][:, 4] = np.where(has, d["ord_progress"][first], 0)
        u["ord"][:, 5] = np.log1p(count) / 2

    # unit columns the model and probes read, in the narrowest type that holds them; the rest
    # (height, facing, cooldowns, ragged-list indices, ...) is dropped at load: memory bounds
    # how many games (and player views) fit in the pool
    KEEP = dict(id=np.int32, type=np.int16, owner=np.int8, x=np.float32, y=np.float32, hp=np.float32,
                hp_max=np.float32, shield=np.float32, shield_max=np.float32, energy=np.float32, build=np.float16,
                flags=np.uint8, contents=np.int16, harvesters=np.int16, unit_idx=np.int16, cat=np.int8,
                resource=np.bool_, produce=np.int16, progress=np.float32, progress2=np.float32, queue=np.int8,
                order=np.int16, ord=np.float32, produce_raw=np.int32, order_raw=np.int32)

    def _slim(self):
        self.u = {k: self.u[k].astype(t, copy=False) for k, t in self.KEEP.items() if k in self.u}

    def _stats(self, d):
        """stats[f, p] = latest tracker PlayerStats row at or before frame f."""
        F, P = len(self.loops), len(self.slots)
        self.stats = np.zeros((F, P, C_PLAYER), np.float32)
        for pi, slot in enumerate(self.slots):
            rows = np.where(d["p_player"] == slot)[0]
            if not len(rows):
                continue
            idx = np.searchsorted(d["p_loop"][rows], self.loops, side="right") - 1
            ok = idx >= 0
            vals = np.stack([d[f"p_{k}"][rows] for k in STAT_KEYS], 1)
            feat = np.concatenate([_log(vals[:, :5]), vals[:, 5:7] / 200.0, _log(vals[:, 7:])], 1)
            self.stats[ok, pi] = feat[idx[ok]]

    def _visibility(self, d, grid=GRID):
        """vis[f, cell] = (share of the cell visible now, share ever seen), cells row-major over
        the playable area like the encoder's grid. The observer sees everything."""
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
                self.ups[np.searchsorted(self.loops, loop):, self.slots.index(slot), k] = 1.0

    # ------------------------------------------------------------ ownership

    def owner_map(self, view="observer", me_slot=None, augment=False, rng=None, codes=None):
        """slot -> (team code, player code, relation) for one sample.

        Without augment, codes follow team order and slot order; with augment they're
        drawn at random from the pool, so a code never means anything beyond "same
        team" / "same player" within this sample. codes = (team permutation, player
        permutation) uses given draws: samples that are compared with each other (a batch
        with in-batch negatives) must share them, or the codes alone tell the samples apart.
        """
        rng = rng or np.random
        teams = sorted({p["team"] for p in self.players})
        if codes is not None:
            team_codes, player_codes = codes[0][:len(teams)], codes[1][:len(self.slots)]
        else:
            team_codes = rng.permutation(N_CODES)[:len(teams)] if augment else np.arange(len(teams))
            player_codes = rng.permutation(N_CODES)[:len(self.slots)] if augment else np.arange(len(self.slots))
        my_team = next((p["team"] for p in self.players if p["slot"] == me_slot), None)
        out = {}
        for i, p in enumerate(self.players):
            rel = REL_NONE
            if view == "player":
                rel = REL_ME if p["slot"] == me_slot else REL_ALLY if p["team"] == my_team else REL_ENEMY
            out[p["slot"]] = (int(team_codes[teams.index(p["team"])]), int(player_codes[i]), rel)
        return out

    # ------------------------------------------------------------ one frame

    def entity_features(self, rows, owners, prefix="ent"):
        """Encoder inputs for the given unit rows (any rows, any frames): what an entity token
        is made of. Shared by frames and by actions that point at units."""
        u = self.u
        hp, hpm = u["hp"][rows], u["hp_max"][rows]
        sh, shm = u["shield"][rows], u["shield_max"][rows]
        flags = u["flags"][rows]
        num = np.stack([
            np.where(hpm > 0, hp / np.maximum(hpm, 1), 0),
            np.where(shm > 0, sh / np.maximum(shm, 1), 0),
            u["energy"][rows] / 200.0,
            _log(hpm, 8.0), _log(shm, 8.0),
            u["build"][rows].astype(np.float32),
            *[((flags >> k) & 1).astype(np.float32) for k in range(6)],  # flying burrowed cloaked halluc snapshot powered
            u["progress"][rows],
            u["progress2"][rows],
            u["queue"][rows] / 5.0,
            _log(u["contents"][rows].astype(np.float32), 8.0),
            u["harvesters"][rows] / 16.0,
        ], 1).astype(np.float32).reshape(-1, C_ENTITY)
        neutral = (CODE_NEUTRAL, CODE_NEUTRAL, REL_NONE)
        own = np.array([owners.get(int(s), neutral) for s in u["owner"][rows]], np.int64).reshape(-1, 3)
        xy = np.stack([u["x"][rows], u["y"][rows]], 1).reshape(-1, 2)
        return {f"{prefix}_type": u["unit_idx"][rows].astype(np.int64),
                # raw API ids, for the ruleset encoder (worldmodel/ruleset.py)
                f"{prefix}_rtype": u["type"][rows].astype(np.int64),
                f"{prefix}_rproduce": u["produce_raw"][rows].astype(np.int64),
                f"{prefix}_rorder": u["order_raw"][rows].astype(np.int64),
                f"{prefix}_team": own[:, 0], f"{prefix}_pcode": own[:, 1], f"{prefix}_rel": own[:, 2],
                f"{prefix}_produce": u["produce"][rows].astype(np.int64),
                f"{prefix}_order": u["order"][rows].astype(np.int64),
                f"{prefix}_ord": u["ord"][rows].reshape(-1, C_ORDER),
                f"{prefix}_xy": self.normalize_xy(xy),
                f"{prefix}_num": num}

    def normalize_xy(self, xy):
        """Map coordinates -> [-1, 1] over the playable area."""
        return ((np.asarray(xy, np.float32) - self.center) / self.half).astype(np.float32)

    def rows_of(self, f, ids):
        """Unit rows in frame f for state unit ids; -1 where the unit isn't in that frame."""
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

    def frame(self, f, owners):
        u, a, b = self.u, self.start[f], self.start[f + 1]
        rows = np.arange(a, b)  # hook for views that hide things (fog): drop rows here

        p_idx = [i for i, s in enumerate(self.slots) if s in owners]
        p_own = np.array([owners[self.slots[i]] for i in p_idx], np.int64).reshape(-1, 3)
        ply_num, ply_ups = self.stats[f, p_idx], self.ups[f, p_idx]
        if self.viewer:  # a player sees only their own economy and upgrades
            hide = np.array([self.slots[i] != self.viewer for i in p_idx])
            ply_num, ply_ups = np.where(hide[:, None], 0, ply_num), np.where(hide[:, None], 0, ply_ups)
        vis = self.vis[f] if self.vis is not None else np.ones((GRID * GRID, C_VIS), np.float32)
        return dict(
            **self.entity_features(rows, owners),
            ent_cat=u["cat"][rows],
            ent_resource=u["resource"][rows],
            ply_race=np.array([self.vocab.race.get(self.players[i]["race"], self.vocab.race["Random"])
                               for i in p_idx], np.int64),
            ply_team=p_own[:, 0], ply_pcode=p_own[:, 1], ply_rel=p_own[:, 2],
            ply_num=ply_num,
            ply_ups=ply_ups,
            vis=vis,
            ruleset=self.meta.get("ruleset", ""),
            ply_upnames=[[self.vocab.upgrade_names[k] for k in np.nonzero(ply_ups[i])[0]] for i in range(len(p_idx))],
            time=np.float32(self.loops[f] / 22.4),
            format=self.format,
            cells=self.half,  # map cells per normalized unit (x, y)
        )


GROUPS = (("ent", ("type", "rtype", "rproduce", "rorder", "team", "pcode", "rel", "produce", "order", "ord", "xy", "num")),
          ("ply", ("race", "team", "pcode", "rel", "num", "ups")))


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
            arrs = [f[f"{group}_{k}"] for f in frames]
            buf = np.zeros((len(frames), n) + arrs[0].shape[1:], arrs[0].dtype)
            for i, a in enumerate(arrs):
                buf[i, :len(a)] = a
            out[f"{group}_{k}"] = torch.from_numpy(buf)
    out["time"] = torch.tensor([f["time"] for f in frames], dtype=torch.float32)
    out["format"] = torch.from_numpy(np.stack([f["format"] for f in frames]))
    out["cells"] = torch.from_numpy(np.stack([f["cells"] for f in frames]))
    out["vis"] = torch.from_numpy(np.stack([f["vis"] for f in frames]))
    out["ruleset"] = [f["ruleset"] for f in frames]          # per sample (strings; left alone by data.to)
    out["ply_upnames"] = [f["ply_upnames"] for f in frames]  # per sample, per player: completed upgrade names
    if "win_center" in frames[0]:  # same number of windows per frame
        out["win_center"] = torch.from_numpy(np.stack([f["win_center"] for f in frames]))
        out["win_size"] = torch.from_numpy(np.stack([f["win_size"] for f in frames]))
    return out
