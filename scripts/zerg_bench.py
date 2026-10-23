"""Plays a Zerg build order (a builds/*.bo file) in a real game and logs the economy.

One run = one build from one spawn on one map. The opponent is a very-easy
built-in Zerg whose units are removed with debug commands every few loops
(its hatchery is kept so the game doesn't end), so there are no interactions.
Run every build from every spawn (scripts/zerg_bench_matrix.py) and compare
with scripts/zerg_bench_compare.py. The build-order language is described in
builds/README.md.

Standing behaviour (not scripted): pro split at the start; main-base drones
are held on their patch (gather is re-sent if they bounce); natural/third
drones are rallied to one patch and left to spread out; injects on cooldown,
one queen per hatchery; costs and build times come from the game's data.

Outputs (in --out_dir):
    timeseries.csv  one row per game loop: bank, mined, supply, drones, larva, per-base drones,
                    supply-blocked-with-idle-larva loops (cumulative), drone-seconds (cumulative)
    events.csv      build-order events (loop, event, detail)
    meta.json       map, build, seed, spawn, base locations, game data
"""

import csv
import json
import math
import os
import re
import sys
from collections import Counter, deque

import numpy as np
from absl import app, flags
from pysc2 import run_configs
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import debug_pb2 as sc_debug
from s2clientprotocol import query_pb2 as sc_query
from s2clientprotocol import raw_pb2 as sc_raw
from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "Ladder2019Season3/AcropolisLE.SC2Map", "Map path, relative to the SC2 Maps dir.")
flags.DEFINE_string("build", "13ov", "Build-order file, or a name in builds/ (without .bo).")
flags.DEFINE_integer("seed", 1, "Game random seed (decides the spawn).")
flags.DEFINE_integer("spawn", -1, "Required spawn index (sorted start locations); exit 3 if the seed gives another. -1 = any.")
flags.DEFINE_integer("loops", int(4 * 60 * 22.4), "Game loops to play (22.4 loops = 1 s).")
flags.DEFINE_string("out_dir", "data/zerg_bench/run", "Output directory.")
flags.DEFINE_integer("team_size", 1, "1 = 1v1; 2 = 2v2 (a very-easy computer ally, two removed enemies).")

BUILDS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "builds")

# Unit types
DRONE, LARVA, HATCHERY, OVERLORD, EXTRACTOR, POOL, QUEEN, ZERGLING = 104, 151, 86, 106, 88, 89, 126, 105
ROACH_WARREN, ROACH = 97, 110
LAIR, HIVE = 100, 101
MORPH_LAIR = 1216
TOWNHALL_TYPES = {86, 100, 101, 18, 132, 130, 59}  # hatch/lair/hive, CC/orbital/PF, nexus

# Abilities (raw)
TRAIN_DRONE, TRAIN_ZERGLING, TRAIN_OVERLORD, TRAIN_QUEEN, TRAIN_ROACH = 1342, 1343, 1344, 1632, 1351
BUILD_HATCHERY, BUILD_EXTRACTOR, BUILD_POOL, BUILD_ROACH_WARREN = 1152, 1154, 1155, 1165
CANCEL_BUILD = 314
INJECT = 251
LING_SPEED = 1253
GATHER, GATHER_DRONE, RETURN_DRONE, RETURN_ANY = 3666, 1183, 1184, 3667
MOVE = 16

CARRY_MINERALS = {271, 272}
CARRY_GAS = {273, 274, 275}
ALLIANCE_SELF, ALLIANCE_NEUTRAL, ALLIANCE_ENEMY = 1, 3, 4

DRONE_SPEED = 3.94 / 22.4  # units per loop
GAS_PER_TRIP = 4
SATURATION = 16  # mineral drones per base
OVERSATURATION = 20  # then every base up to this (3rd drones on far patches) before going further
BASES = ("main", "natural", "third")

# Costs and build times: defaults, overwritten from the game's data at startup
# (load_game_data), since balance patches change them.
COSTS = dict(drone=50, overlord=100, hatch=275, extractor=25, pool=200, queen=175, ling_pair=50,
             ling_speed=100, ling_speed_gas=100, roach_warren=150, roach=75, roach_gas=25, lair=150, lair_gas=100)
HATCH_BUILD_LOOPS = 1600
BUILD_LOOPS = dict(roach_warren=880, roach=432, extractor=480)
DRONE_EGG_LOOPS = 272
OVERLORD_LOOPS = 400
POOL_LOOPS = 1040
QUEEN_LOOPS = 800
LARVA_SPAWN_LOOPS = 240  # a hatchery below 3 larva spawns one every 10.7 s (measured from replays)
INJECT_LOOPS = 29 * 22.4
INJECT_BUFF = 11  # QueenSpawnLarvaTimer


def load_game_data(controller):
    """Sets COSTS / build times from the running game version."""
    global HATCH_BUILD_LOOPS, DRONE_EGG_LOOPS, OVERLORD_LOOPS, POOL_LOOPS, QUEEN_LOOPS
    data = controller.data_raw()
    u = {d.unit_id: d for d in data.units}
    speed = next(x for x in data.upgrades if x.name == "zerglingmovementspeed")
    drone = u[DRONE].mineral_cost
    # A Zerg structure's unit cost includes the drone that becomes it.
    COSTS.update(
        drone=drone, overlord=u[OVERLORD].mineral_cost, queen=u[QUEEN].mineral_cost,
        hatch=u[HATCHERY].mineral_cost - drone, extractor=u[EXTRACTOR].mineral_cost - drone,
        pool=u[POOL].mineral_cost - drone, ling_pair=2 * u[ZERGLING].mineral_cost,
        ling_speed=speed.mineral_cost, ling_speed_gas=speed.vespene_cost,
        roach_warren=u[ROACH_WARREN].mineral_cost - drone, roach=u[ROACH].mineral_cost, roach_gas=u[ROACH].vespene_cost,
    )
    BUILD_LOOPS.update(roach_warren=u[ROACH_WARREN].build_time, roach=u[ROACH].build_time, extractor=u[EXTRACTOR].build_time)
    HATCH_BUILD_LOOPS = u[HATCHERY].build_time
    DRONE_EGG_LOOPS = int(u[DRONE].build_time)
    OVERLORD_LOOPS = u[OVERLORD].build_time
    POOL_LOOPS, QUEEN_LOOPS = u[POOL].build_time, u[QUEEN].build_time
    return dict(COSTS, hatch_build_loops=HATCH_BUILD_LOOPS, drone_egg_loops=DRONE_EGG_LOOPS)


# ---------------------------------------------------------------- build orders

ACTIONS = {
    # action: (min args, max args)
    "overlord": (0, 1), "hatch": (1, 1), "extractor": (0, 1), "pool": (0, 0), "extractor_trick": (0, 0),
    "queen": (1, 1), "lings": (1, 1), "ling_speed": (0, 0), "rally": (1, 2), "default": (1, 1),
    "pull_gas": (1, 1), "auto_overlords": (0, 0), "roach_warren": (0, 0), "roaches": (1, 2), "transfer": (3, 3), "sync_injects": (0, 0), "queens": (2, 3), "gas_pull": (2, 2), "lair": (0, 0), "cycle_overlords": (0, 0),
}
EVENTS = {"pool", "pool_soon", "warren", "warren_soon", "roaches", "main", "natural", "third", "natural_soon", "third_soon",
          "queen:main", "queen:natural", "queen:third"}
RALLY_ROLES = {"main", "natural", "third", "gas", "gas:main", "gas:natural", "gas:third", "extractor", "pool",
               "roach_warren"}


class Step:
    def __init__(self, trigger, action, args, background, text):
        self.trigger, self.action, self.args, self.background, self.text = trigger, action, args, background, text
        self.fired = False  # trigger has been met (sticky)
        self.state = {}

    def __repr__(self):
        return self.text


def parse_build(path, _seen=None):
    """Parses a .bo file (see builds/README.md) into a list of Steps."""
    _seen = _seen or set()
    path = os.path.abspath(path)
    if path in _seen:
        raise ValueError(f"include cycle at {path}")
    _seen = _seen | {path}
    steps = []
    with open(path) as f:
        for n, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            where = f"{os.path.basename(path)}:{n}: {line}"
            toks = line.split()
            if toks[0] == "include":
                steps += parse_build(os.path.join(os.path.dirname(path), toks[1]), _seen)
                continue
            background = toks[0] == "&"
            if background:
                toks = toks[1:]
            if re.fullmatch(r"\d+", toks[0]):
                trigger, toks = ("supply", int(toks[0])), toks[1:]
            elif re.fullmatch(r"@extractors:\d+", toks[0]):
                trigger, toks = ("event", toks[0][1:]), toks[1:]
            elif re.fullmatch(r"@\d+:\d\d", toks[0]):
                mm, ss = toks[0][1:].split(":")
                trigger, toks = ("time", round((int(mm) * 60 + int(ss)) * 22.4)), toks[1:]
            elif toks[0].startswith("@"):
                trigger, toks = ("event", toks[0][1:]), toks[1:]
                if trigger[1] not in EVENTS:
                    raise ValueError(f"{where}: unknown event @{trigger[1]} (known: {sorted(EVENTS)})")
            else:
                trigger = ("now",)
                if toks[0] == "-":
                    toks = toks[1:]
            if not toks or toks[0] not in ACTIONS:
                raise ValueError(f"{where}: unknown action (known: {sorted(ACTIONS)})")
            action, args = toks[0], toks[1:]
            lo, hi = ACTIONS[action]
            if not lo <= len(args) <= hi:
                raise ValueError(f"{where}: {action} takes {lo}-{hi} arguments")
            if action == "rally":
                if args[0] not in RALLY_ROLES:
                    raise ValueError(f"{where}: unknown rally role (known: {sorted(RALLY_ROLES)})")
                if len(args) == 2 and not re.fullmatch(r"x\d+", args[1]):
                    raise ValueError(f"{where}: rally count must look like x3")
            if action in ("overlord", "hatch", "extractor", "queen") and args and args[0] not in BASES:
                raise ValueError(f"{where}: base must be one of {BASES}")
            if action == "default" and args[0] not in BASES + ("fill",):
                raise ValueError(f"{where}: default must be one of {BASES + ('fill',)}")
            steps.append(Step(trigger, action, args, background, where))
    return steps


def resolve_build(name):
    if os.path.exists(name):
        return name
    p = os.path.join(BUILDS_DIR, name + ".bo")
    if os.path.exists(p):
        return p
    raise SystemExit(f"no build file {name!r} (looked in {BUILDS_DIR})")


# ---------------------------------------------------------------- helpers

def _p2(x, y):
    return sc_common.Point2D(x=x, y=y)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _xy(u):
    return (u.pos.x, u.pos.y)


def _cmd(ability_id, tags, target_tag=None, point=None, queue=False):
    c = sc_raw.ActionRawUnitCommand(ability_id=ability_id, unit_tags=list(tags), queue_command=queue)
    if target_tag is not None:
        c.target_unit_tag = target_tag
    if point is not None:
        c.target_world_space_pos.x, c.target_world_space_pos.y = point
    return sc_pb.Action(action_raw=sc_raw.ActionRaw(unit_command=c))


class MapInfo:
    """Base locations, from resource clusters and the static placement grid."""

    def __init__(self, controller, units, main_xy):
        # Resources under fog are snapshots with contents 0, so identify them
        # by type name rather than by mineral_contents / vespene_contents.
        names = {d.unit_id: d.name for d in controller.data_raw().units}
        self.is_mineral = is_mineral = lambda u: "MineralField" in names.get(u.unit_type, "")
        self.is_geyser = is_geyser = lambda u: "Geyser" in names.get(u.unit_type, "")
        info = controller.game_info()
        grid = info.start_raw.placement_grid
        self.placeable = np.unpackbits(np.frombuffer(grid.data, dtype=np.uint8))[: grid.size.x * grid.size.y].reshape(
            grid.size.y, grid.size.x
        )
        self.main = main_xy
        others = [(p.x, p.y) for p in info.start_raw.start_locations]
        self.spawns = sorted([main_xy] + others)
        self.spawn_index = self.spawns.index(main_xy)

        resources = [u for u in units if u.alliance == ALLIANCE_NEUTRAL and (is_mineral(u) or is_geyser(u))]
        self.bases = []
        for c in self._cluster(resources):
            minerals = [u for u in c if is_mineral(u)]
            geysers = [u for u in c if is_geyser(u)]
            if len(minerals) < 4 or not geysers:
                continue  # mineral walls etc.
            spot = main_xy if min(_dist(_xy(u), main_xy) for u in c) < 12 else self._townhall_spot(minerals, geysers)
            if spot:
                self.bases.append(dict(spot=spot, minerals=[u.tag for u in minerals], geysers=[u.tag for u in geysers]))

        main = min(self.bases, key=lambda b: _dist(b["spot"], main_xy))
        free = [b for b in self.bases if b is not main and min(_dist(b["spot"], s) for s in self.spawns) > 5]
        natural, self.natural_walk = self._closest_walk(controller, main["spot"], free)
        # In team games every spawn has a natural; the third must not be one of them.
        taken = [natural] + [self._closest_walk(controller, sp, free)[0] for sp in self.spawns if sp != main_xy]
        third, self.third_walk = self._closest_walk(controller, natural["spot"], [b for b in free if b not in taken])
        self.named = {"main": main, "natural": natural, "third": third}

    def index(self, name):
        return self.bases.index(self.named[name])

    def name_of(self, bi):
        return next((n for n, b in self.named.items() if b is self.bases[bi]), f"base{bi}")

    @staticmethod
    def _closest_walk(controller, start, bases):
        cands = sorted(bases, key=lambda b: _dist(b["spot"], start))[:5]
        res = controller.query(sc_query.RequestQuery(
            pathing=[sc_query.RequestQueryPathing(start_pos=_p2(*start), end_pos=_p2(*b["spot"])) for b in cands]))
        walks = [r.distance if r.distance > 0 else 1e9 for r in res.pathing]
        i = int(np.argmin(walks))
        return cands[i], walks[i]

    def refresh(self, units):
        """Re-read each base's resource tags: fogged snapshots get new tags
        once the real units come into view."""
        for b in self.bases:
            near = [u for u in units if u.alliance == ALLIANCE_NEUTRAL and _dist(_xy(u), b["spot"]) < 11]
            minerals = [u.tag for u in near if self.is_mineral(u)]
            geysers = [u.tag for u in near if self.is_geyser(u)]
            if minerals:
                b["minerals"] = minerals
            if geysers:
                b["geysers"] = geysers

    @staticmethod
    def _cluster(units, link=8.5):
        clusters = []
        for u in units:
            near = [c for c in clusters if any(_dist(_xy(u), _xy(v)) < link for v in c)]
            merged = [u]
            for c in near:
                merged += c
                clusters.remove(c)
            clusters.append(merged)
        return clusters

    def _footprint_ok(self, x, y, half=2.5):
        x0, x1, y0, y1 = int(x - half), int(x + half), int(y - half), int(y + half)
        if x0 < 0 or y0 < 0 or x1 > self.placeable.shape[1] or y1 > self.placeable.shape[0]:
            return False
        return bool(self.placeable[y0:y1, x0:x1].all())

    def _townhall_spot(self, minerals, geysers):
        cx = np.mean([u.pos.x for u in minerals + geysers])
        cy = np.mean([u.pos.y for u in minerals + geysers])
        best = None
        for dx in range(-12, 13):
            for dy in range(-12, 13):
                x, y = math.floor(cx) + dx + 0.5, math.floor(cy) + dy + 0.5
                if any(_dist((x, y), _xy(m)) < 6 for m in minerals) or any(_dist((x, y), _xy(g)) < 7 for g in geysers):
                    continue
                if not self._footprint_ok(x, y):
                    continue
                score = sum(_dist((x, y), _xy(u)) for u in minerals + geysers)
                if best is None or score < best[0]:
                    best = (score, (x, y))
        return best[1] if best else None


# ---------------------------------------------------------------- the bot

class Bench:
    def __init__(self, controller, steps):
        self.c = controller
        self.steps = steps
        self.i = 0  # next main-sequence step
        self.background = []
        self.loop = 0
        self.events = []
        self.assign = {}  # drone tag -> ("min", patch) | ("rally", patch) | ("gas", extractor) | ("build", job)
        self.patch_base = {}
        self.jobs = []
        self.trick_extractors = set()
        self.roaches_seen = 0
        self.rally = deque()  # roles for the next drones made
        self.default = "main"
        self.pull_gas_at = None
        self.gas_pulled = False
        self.auto_overlords = False
        self.queen_home = {}
        self.inject_pops = {}  # hatch tag -> loop its inject pops
        self.cycle_overlords = False
        self.cycle_owed = 0  # overlords still to make this macro cycle
        self.cycle_until = 0
        self.sync_injects = False
        self.injected = set()  # queens that have injected at least once
        self.collected_hist = []
        self.pending_cmds = []
        # Roles for drones in eggs, in hatch order. Larva, egg and drone have
        # different tags, but drone eggs all take DRONE_EGG_LOOPS, so drones
        # hatch in the order they were made.
        self.role_queue = deque()  # (due loop, role)
        self.seen_drones = set()
        self.structures_seen = {}
        self.done_seen = set()
        self.saturated = set()
        self.blocked_loops = 0
        self.capped_loops = 0  # hatchery-loops spent at 3 larva (natural spawning stopped)
        self.hold_capped_loops = 0  # ... while a roach batch was holding larva
        self.drone_loops = 0

    # ---------- helpers ----------
    def event(self, name, detail=""):
        self.events.append((self.loop, name, detail))
        print(f"{self.loop:5d} {self.loop / 22.4:6.1f}s  {name} {detail}")

    def act(self, a):
        self.pending_cmds.append(a)

    def income_per_loop(self):
        h = self.collected_hist
        if len(h) < 50:
            return 0.9
        span = min(224, len(h) - 1)
        return max((h[-1] - h[-1 - span]) / span, 0.1)

    def walk_loops(self, a, b):
        res = self.c.query(sc_query.RequestQuery(pathing=[sc_query.RequestQueryPathing(start_pos=_p2(*a), end_pos=_p2(*b))]))
        return (res.pathing[0].distance or _dist(a, b)) / DRONE_SPEED

    def bi(self, name):
        return self.map.index(name)

    def hatch_at(self, name):
        bi = self.bi(name)
        return next((h for h in self.hatches if self.base_of(_xy(h)) == bi), None)

    def base_done(self, name):
        h = self.hatch_at(name)
        return h is not None and h.build_progress >= 1

    def base_of(self, xy):
        return min(range(len(self.map.bases)), key=lambda i: _dist(self.map.bases[i]["spot"], xy))

    def patch_base_of(self, patch):
        if patch not in self.patch_base:
            for i, b in enumerate(self.map.bases):
                if patch in b["minerals"]:
                    self.patch_base[patch] = i
        return self.patch_base.get(patch)

    def mineral_drones(self, name):
        bi = self.bi(name)
        return sum(1 for a in self.assign.values() if a[0] in ("min", "rally") and self.patch_base_of(a[1]) == bi)

    # ---------- per-loop observation ----------
    def observe(self, obs):
        o = obs.observation
        self.loop = o.game_loop
        self.pc = o.player_common
        self.score = o.score.score_details
        self.collected_hist.append(self.score.collected_minerals)
        self.units = {u.tag: u for u in o.raw_data.units}
        if hasattr(self, "map"):
            self.map.refresh(o.raw_data.units)
            self.patch_base = {}
            self.remap_patches(o.raw_data.units)
        mine = [u for u in o.raw_data.units if u.alliance == ALLIANCE_SELF]
        self.mine = mine
        self.drones = {u.tag: u for u in mine if u.unit_type == DRONE}
        self.larva = [u for u in mine if u.unit_type == LARVA]
        self.hatches = [u for u in mine if u.unit_type in (HATCHERY, LAIR, HIVE)]
        self.done_hatches = [h for h in self.hatches if h.build_progress >= 1]
        self.extractors = [u for u in mine if u.unit_type == EXTRACTOR]
        self.pool = next((u for u in mine if u.unit_type == POOL), None)
        self.warren = next((u for u in mine if u.unit_type == ROACH_WARREN), None)
        n_roach = sum(1 for u in mine if u.unit_type == ROACH)
        if n_roach > self.roaches_seen and hasattr(self, "map"):
            self.event("roaches pop", f"{n_roach} total")
        self.roaches_seen = n_roach
        self.queens = [u for u in mine if u.unit_type == QUEEN]
        self.training = Counter(
            u.orders[0].ability_id for u in mine if u.orders and u.orders[0].ability_id in (TRAIN_DRONE, TRAIN_OVERLORD, TRAIN_ZERGLING)
        )
        # Drones we lost track of are inside an extractor (or became buildings).
        for tag, a in list(self.assign.items()):
            if tag not in self.drones and a[0] != "gas":
                del self.assign[tag]
        if hasattr(self, "map"):
            self.log_structures()

    def remap_patches(self, units):
        """A fogged mineral patch is a snapshot whose tag changes when the real
        patch comes into view; move assignments to the patch at the same spot."""
        if not hasattr(self, "patch_xy"):
            self.patch_xy = {}
        minerals = [u for u in units if u.alliance == ALLIANCE_NEUTRAL and self.map.is_mineral(u)]
        for u in minerals:
            self.patch_xy[u.tag] = _xy(u)
        for tag, a in list(self.assign.items()):
            if a[0] in ("min", "rally") and a[1] not in self.units and a[1] in self.patch_xy:
                old = self.patch_xy[a[1]]
                new = min(minerals, key=lambda u: _dist(_xy(u), old), default=None)
                if new is not None and _dist(_xy(new), old) < 1.0:
                    self.assign[tag] = (a[0], new.tag)

    def log_structures(self):
        names = {HATCHERY: "hatch", EXTRACTOR: "extractor", POOL: "pool", ROACH_WARREN: "roach warren"}
        for u in self.mine:
            if u.unit_type not in names:
                continue
            where = self.map.name_of(self.base_of(_xy(u)))
            name = f"{names[u.unit_type]} {where}" if u.unit_type not in (POOL, ROACH_WARREN) else names[u.unit_type]
            if u.tag not in self.structures_seen:
                self.structures_seen[u.tag] = self.loop
                if self.loop > 0:
                    self.event(f"{name} started", f"bank={self.pc.minerals}")
            if u.build_progress >= 1 and u.tag not in self.done_seen:
                self.done_seen.add(u.tag)
                if self.loop > 0:
                    self.event(f"{name} done")
        for name in ("natural", "third"):
            if name not in self.saturated and self.mineral_drones(name) >= SATURATION:
                self.saturated.add(name)
                self.event(f"{name} saturated", f"{SATURATION} mineral drones")

    def suppress_enemy(self, obs):
        kill = [u.tag for u in obs.observation.raw_data.units if u.alliance == ALLIANCE_ENEMY and u.unit_type not in TOWNHALL_TYPES]
        if kill:
            self.c.debug(sc_debug.DebugCommand(kill_unit=sc_debug.DebugKillUnit(tag=kill)))

    # ---------- where drones go ----------
    def hold_on_patch(self, drone, bi):
        """Main-style: 2 per patch (3 if full), held there (no bouncing)."""
        counts = Counter(a[1] for a in self.assign.values() if a[0] == "min")
        spot = self.map.bases[bi]["spot"]
        for per_patch in (2, 3):
            free = [p for p in self.map.bases[bi]["minerals"] if p in self.units and counts[p] < per_patch]
            if free:
                # Up to 2: closest patches first. A 3rd drone goes on the farthest
                # patches, where the longer trip leaves room for a third miner.
                sign = 1 if per_patch == 2 else -1
                patch = min(free, key=lambda p: (counts[p], sign * _dist(_xy(self.units[p]), spot)))
                self.assign[drone.tag] = ("min", patch)
                self.act(_cmd(GATHER, [drone.tag], target_tag=patch))
                return

    def rally_to(self, drone, bi):
        """Pro-style rally: one patch, drones spread out by themselves."""
        spot = self.map.bases[bi]["spot"]
        patch = min((p for p in self.map.bases[bi]["minerals"] if p in self.units), key=lambda p: _dist(_xy(self.units[p]), spot))
        self.assign[drone.tag] = ("rally", patch)
        self.act(_cmd(GATHER, [drone.tag], target_tag=patch))

    def default_base(self, near=None):
        """Where a drone without a role goes. With `fill`: a base below
        saturation - the one nearest `near` (where the drone hatched), so a
        drone never walks main<->natural when its own base has room."""
        if self.default != "fill":
            return self.default
        for level in (SATURATION, OVERSATURATION):
            room = [n for n in BASES if self.hatch_at(n) is not None and self.mineral_drones(n) < level]
            if room:
                if near is None:
                    return room[0]
                return min(room, key=lambda n: _dist(self.map.named[n]["spot"], near))
        return max((n for n in BASES if self.hatch_at(n) is not None), key=BASES.index)

    def send_to(self, drone, name):
        if name == "main":
            self.hold_on_patch(drone, self.bi("main"))
        else:
            self.rally_to(drone, self.bi(name))

    def apply_role(self, d, role):
        if role in (None, "default"):
            self.send_to(d, self.default_base(near=_xy(d)))
        elif role in BASES:
            self.send_to(d, role)
        elif role in ("extractor", "pool", "roach_warren"):
            self.start_job(role, drone=d)
        elif role.startswith("gas"):
            ext = self.extractor_needing_workers(role.split(":")[1] if ":" in role else None)
            if ext is None or self.gas_pulled:
                self.send_to(d, self.default_base())
                self.event("gas drone", f"no extractor for {role}: sent to {self.default_base()}")
                return
            self.assign[d.tag] = ("gas", ext.tag)
            self.act(_cmd(GATHER, [d.tag], target_tag=ext.tag) if ext.build_progress >= 1 else _cmd(MOVE, [d.tag], point=_xy(ext)))
            self.event("gas drone", f"{self.map.name_of(self.base_of(_xy(ext)))} extractor")

    def take_drone_from(self, bi):
        """Tag of the best mining drone to take off base bi: from the patch with
        the most drones (farthest first), recounting every time, so the line
        never ends up 3+1."""
        counts = Counter(a[1] for a in self.assign.values() if a[0] in ("min", "rally"))
        cands = [(t, a[1]) for t, a in self.assign.items()
                 if a[0] in ("min", "rally") and t in self.drones and self.patch_base_of(a[1]) == bi]
        if not cands:
            return None
        spot = self.map.bases[bi]["spot"]
        t, _ = min(cands, key=lambda c: (-counts[c[1]], -(_dist(_xy(self.units[c[1]]), spot) if c[1] in self.units else 0),
                                         any(b in CARRY_MINERALS for b in self.drones[c[0]].buff_ids)))
        return t

    def move_drone(self, t, order):
        """Reassign a mining drone; one carrying minerals drops them off first."""
        d = self.drones[t]
        carrying = any(b in CARRY_MINERALS for b in d.buff_ids)
        n = len(self.pending_cmds)
        order(d)
        if carrying:
            new = self.pending_cmds[n:]
            del self.pending_cmds[n:]
            self.act(_cmd(RETURN_ANY, [t]))
            for c in new:
                c.action_raw.unit_command.queue_command = True
                self.pending_cmds.append(c)

    def rebalance(self):
        """Never 3 drones on one patch while another patch of the same base has
        1: move one across (held, main-style drones only)."""
        by_base = {}
        counts = Counter(a[1] for a in self.assign.values() if a[0] == "min")
        for t, a in self.assign.items():
            if a[0] == "min":
                by_base.setdefault(self.patch_base_of(a[1]), []).append((t, a[1]))
        for bi, lst in by_base.items():
            if bi is None:
                continue
            patches = [p for p in self.map.bases[bi]["minerals"] if p in self.units]
            if not patches:
                continue
            lo = min(patches, key=lambda p: counts[p])
            hi = max(patches, key=lambda p: counts[p])
            if counts[hi] - counts[lo] < 2:
                continue
            movers = [t for t, p in lst if p == hi and t in self.drones
                      and not any(b in CARRY_MINERALS for b in self.drones[t].buff_ids)]
            if movers:
                self.assign[movers[0]] = ("min", lo)
                self.act(_cmd(GATHER, [movers[0]], target_tag=lo))
                self.event("rebalance", f"{counts[hi]}->{counts[lo]} at {self.map.name_of(bi)}")

    def extractor_needing_workers(self, base=None):
        """The (real, not trick) extractor with the fewest drones, below 3."""
        counts = Counter(a[1] for a in self.assign.values() if a[0] == "gas")
        exts = [e for e in self.extractors if e.tag not in self.trick_extractors and counts[e.tag] < 3
                and (base is None or self.base_of(_xy(e)) == self.bi(base))]
        return min(exts, key=lambda e: counts[e.tag], default=None)

    def pop_role(self, tag):
        """Role for a drone we haven't seen before, if an egg is due now."""
        if tag in self.seen_drones:
            return None  # e.g. a builder coming back from a cancelled extractor
        if self.role_queue and self.role_queue[0][0] <= self.loop + 6:
            return self.role_queue.popleft()[1]
        return None

    def initial_split(self):
        main = self.map.named["main"]
        patches = sorted(main["minerals"], key=lambda p: _dist(_xy(self.units[p]), main["spot"]))
        slots = patches + patches[: len(self.drones) - len(patches)]
        free = list(self.drones.values())
        for p in slots:
            d = min(free, key=lambda u: _dist(_xy(u), _xy(self.units[p])))
            free.remove(d)
            self.assign[d.tag] = ("min", p)
            self.act(_cmd(GATHER, [d.tag], target_tag=p))

    def manage_workers(self):
        for tag, d in self.drones.items():
            a = self.assign.get(tag)
            role = self.pop_role(tag) if a is None else None
            self.seen_drones.add(tag)
            if a is None:
                self.apply_role(d, role)
                continue
            carrying_min = any(b in CARRY_MINERALS for b in d.buff_ids)
            carrying_gas = any(b in CARRY_GAS for b in d.buff_ids)
            if a[0] == "gas" and self.gas_pulled and not carrying_gas:
                del self.assign[tag]
                self.send_to(d, self.default_base())
                continue
            if a[0] in ("build", "rally") or carrying_min or carrying_gas:
                continue
            target = a[1]
            if a[0] == "gas":
                ext = self.units.get(target)
                if ext is None or ext.build_progress < 1:
                    continue
            order = d.orders[0] if d.orders else None
            if order is None or (order.ability_id == GATHER_DRONE and order.target_unit_tag != target) or (
                order.ability_id not in (GATHER_DRONE, RETURN_DRONE)
            ):
                self.act(_cmd(GATHER, [tag], target_tag=target))

    def manage_gas(self):
        if self.pull_gas_at is None or self.gas_pulled:
            return
        gas_workers = [t for t, a in self.assign.items() if a[0] == "gas"]
        in_flight = sum(1 for t in gas_workers if t not in self.drones or any(b in CARRY_GAS for b in self.drones[t].buff_ids))
        if gas_workers and self.score.collected_vespene + GAS_PER_TRIP * in_flight >= self.pull_gas_at:
            self.gas_pulled = True
            self.event("pull gas", f"collected={self.score.collected_vespene} in_flight={in_flight}")

    # ---------- production ----------
    def larva_at(self, name):
        """Larva in the order to spend them: a hatchery at 3 (not spawning)
        first, then the named base's, then the hatchery with the most."""
        bi = self.bi(name) if name else None
        per = Counter(self.base_of(_xy(l)) for l in self.larva)
        def key(l):
            b = self.base_of(_xy(l))
            return (per[b] < 3, not (bi is not None and b == bi), -per[b])
        return sorted(self.larva, key=key)

    def larva_wanted(self):
        """Larva being kept for a roach batch that is about to start."""
        for st in self.background + self.steps[self.i:self.i + 1]:
            if st.action == "roaches" and self.warren is not None and st.fired and not st.state.get("done_at"):
                return int(st.args[0]) - st.state.get("made", 0)
        return 0

    def train(self, ability, cost, base="main"):
        """Returns True if ordered."""
        if self.pc.minerals < cost or not self.larva:
            return False
        lv = self.larva_at(base)[0]
        self.act(_cmd(ability, [lv.tag]))
        self.larva.remove(lv)
        self.pc.minerals -= cost
        return True

    def make_drone(self):
        if self.pc.food_used >= self.pc.food_cap or self.pc.minerals < COSTS["drone"] or not self.larva:
            return False
        role = self.rally.popleft() if self.rally else "default"
        dest = self.default_base() if role == "default" else role.split(":")[-1] if role.startswith("gas:") else role
        if role == "default" and self.default == "fill":
            # Use larva at a base that has room, so the drone mines where it hatches.
            for level in (SATURATION, OVERSATURATION):
                room = [n for n in BASES if self.hatch_at(n) is not None and self.mineral_drones(n) < level
                        and any(self.base_of(_xy(l)) == self.bi(n) for l in self.larva)]
                if room:
                    dest = room[0]
                    break
        # Hatch drones where they're going (main larva for builders and gas).
        base = dest if dest in BASES and self.base_done(dest) else "main"
        self.train(TRAIN_DRONE, COSTS["drone"], base)
        self.role_queue.append((self.loop + DRONE_EGG_LOOPS, role))
        self.pc.food_used += 1
        if role != "default":
            self.event("drone", f"rally {role}")
        return True

    def overlords_to_start(self):
        """How many overlords to start now so larva never sits idle for lack
        of supply, starting each as late as possible.

        Looks one overlord build time ahead: the supply we'll want to spend in
        that window (larva we'll have -- waiting now, natural spawns, injects
        popping -- capped by what we can afford) against the supply we'll have
        by the end of it (cap + overlords in eggs + hatcheries that finish in
        time)."""
        window = OVERLORD_LOOPS
        if len(self.larva) <= 1 and self.done_hatches:
            # Larva is scarce: spending this one on a drone means the next
            # chance to start an overlord is a larva spawn away.
            window += LARVA_SPAWN_LOOPS / len(self.done_hatches)
        larva = len(self.larva)
        for h in self.done_hatches:
            here = sum(1 for l in self.larva if self.base_of(_xy(l)) == self.base_of(_xy(h)))
            larva += min(max(0, 3 - here), int(round(window / LARVA_SPAWN_LOOPS)))
            pop = self.inject_pops.get(h.tag)
            if pop is not None and pop <= self.loop + window:
                larva += 3
        affordable = (self.pc.minerals + self.income_per_loop() * window) / COSTS["drone"]
        want = min(larva, affordable) + 2 * self.queens_due(window) + self.roach_extra_supply(window)
        soon = sum(1 for h in self.hatches if h.build_progress < 1 and (1 - h.build_progress) * HATCH_BUILD_LOOPS <= window)
        have = self.pc.food_cap + 8 * self.training[TRAIN_OVERLORD] + 6 * soon - self.pc.food_used
        n = 0
        # Each overlord also uses a larva (and its cost) that would have been a drone.
        while have + 8 * n < want - n and self.pc.food_cap + 8 * (self.training[TRAIN_OVERLORD] + n) < 200:
            n += 1
        return n

    def queens_due(self, window):
        """Scripted queens likely to start within `window` loops: their
        hatchery is (nearly) done and, for @pool / @queen triggers, so is that."""
        n = 0
        for st in self.steps[self.i:] + self.background:
            if st.action not in ("queen", "queens") or st.state.get("queued"):
                continue
            h = self.hatch_at(st.args[0])
            if h is None or (1 - h.build_progress) * HATCH_BUILD_LOOPS > window:
                continue
            ev = st.trigger[1] if st.trigger[0] == "event" else None
            if ev == "pool" and (self.pool is None or (1 - self.pool.build_progress) * POOL_LOOPS > window):
                continue
            if ev and ev.startswith("queen:") and not st.fired:
                hb = self.hatch_at(ev.split(":")[1])
                busy = hb is not None and any(o.ability_id == TRAIN_QUEEN and (1 - o.progress) * QUEEN_LOOPS <= window for o in hb.orders)
                if not busy:
                    continue
            n += len(st.args) if st.action == "queens" else 1
        return n

    def roach_extra_supply(self, window):
        """A scripted roach batch due within `window` needs 1 supply per larva
        more than the drones the forecast assumes."""
        for st in self.steps[self.i:] + self.background:
            if st.action == "roaches" and self.warren is not None and \
                    (1 - self.warren.build_progress) * BUILD_LOOPS["roach_warren"] <= window:
                return int(st.args[0]) - st.state.get("made", 0)
        return 0

    def cycle_overlords_needed(self):
        """Overlords to make now so supply lasts until the next macro cycle's
        overlords pop: one inject cycle (~32 s) plus an overlord's build time."""
        window = 32 * 22.4 + OVERLORD_LOOPS
        n_h = len(self.done_hatches)
        larva = len(self.larva) + n_h * window / LARVA_SPAWN_LOOPS + 3 * min(n_h, len(self.injected))
        affordable = (self.pc.minerals + self.income_per_loop() * window) / COSTS["drone"]
        want = min(larva, affordable) + 2 * self.queens_due(window) + self.roach_extra_supply(window)
        soon = sum(1 for h in self.hatches if h.build_progress < 1 and (1 - h.build_progress) * HATCH_BUILD_LOOPS <= window)
        have = self.pc.food_cap + 8 * self.training[TRAIN_OVERLORD] + 6 * soon - self.pc.food_used
        n = 0
        while have + 8 * n < want - n and self.pc.food_cap + 8 * (self.training[TRAIN_OVERLORD] + n) < 200 and n < 3:
            n += 1
        return n

    def track_injects(self):
        """Remember when each hatchery's inject will pop (+3 larva)."""
        for h in self.done_hatches:
            injected = INJECT_BUFF in h.buff_ids
            if injected and h.tag not in self.inject_pops:
                self.inject_pops[h.tag] = self.loop + INJECT_LOOPS
            elif not injected and h.tag in self.inject_pops:
                del self.inject_pops[h.tag]
                self.event("inject pops", self.map.name_of(self.base_of(_xy(h))))
                if self.cycle_overlords and self.loop > self.cycle_until:
                    # Macro cycle: decide this cycle's overlords once, as the injects pop.
                    self.cycle_owed = self.cycle_overlords_needed()
                    self.cycle_until = self.loop + int(3 * 22.4)

    # ---------- builders ----------
    def target_for(self, kind, base="main"):
        if kind == "hatch":
            return dict(ability=BUILD_HATCHERY, target_point=self.map.named[base]["spot"])
        if kind == "pool":
            return dict(ability=BUILD_POOL, target_point=self.pool_spot)
        if kind == "roach_warren":
            return dict(ability=BUILD_ROACH_WARREN, target_point=self.warren_spot)
        b = self.map.named[base]
        geysers = [g for g in b["geysers"] if g in self.units]
        taken = {_xy(e) for e in self.extractors}
        g = min((g for g in geysers if _xy(self.units[g]) not in taken), key=lambda g: _dist(_xy(self.units[g]), b["spot"]))
        return dict(ability=BUILD_EXTRACTOR, target_tag=g)

    def start_job(self, kind, drone=None, base="main", step=None):
        t = self.target_for("extractor" if kind == "trick_extractor" else kind, base)
        dest = t.get("target_point") or _xy(self.units[t["target_tag"]])
        d = drone
        if d is None:
            d = self.pick_builder(dest)
        cost = COSTS["extractor" if kind == "trick_extractor" else kind]
        job = dict(kind=kind, base=base, ability=t["ability"], cost=cost, point=t.get("target_point"),
                   tag=t.get("target_tag"), drone=d.tag, state="moving", dest=dest, step=step)
        self.assign[d.tag] = ("build", job)
        self.jobs.append(job)
        self.act(_cmd(MOVE, [d.tag], point=dest))
        self.event("send builder", f"{kind} {base} bank={self.pc.minerals}")
        return job

    def pick_builder(self, dest):
        """A mining drone from the building's own base (else the nearest base):
        empty-handed first, then off a patch with 3 drones, then the farthest
        patches (they hold about half the minerals), then closest to the spot."""
        counts = Counter(a[1] for a in self.assign.values() if a[0] == "min")
        cands = [(d, self.assign[t][1]) for t, d in self.drones.items() if self.assign.get(t, ("",))[0] in ("min", "rally")]
        # Builders come from the building's own base; for a new base, from the
        # nearest base that has drones (walking time is what matters).
        has = {self.patch_base_of(c[1]) for c in cands} - {None}
        here = min(has, key=lambda b: _dist(self.map.bases[b]["spot"], dest)) if has else None
        cands = [c for c in cands if self.patch_base_of(c[1]) == here] or cands

        def key(c):
            d, p = c
            b = self.patch_base_of(p)
            far = _dist(_xy(self.units[p]), self.map.bases[b]["spot"]) if p in self.units and b is not None else 0
            return (any(x in CARRY_MINERALS for x in d.buff_ids), -counts[p], -far, _dist(_xy(d), dest))
        return min(cands, key=key)[0]

    def job_for(self, kind, base=None, step=None):
        for j in self.jobs:
            if j["kind"] == kind and (base is None or j["base"] == base) and (step is None or j["step"] in (None, step)):
                return j
        return None

    def advance_jobs(self):
        for job in self.jobs:
            d = self.drones.get(job["drone"])
            if d is not None and not job.get("arrived") and _dist(_xy(d), job["dest"]) < 2.5:
                job["arrived"] = self.loop
                self.event("builder arrived", f"{job['kind']} {job['base']} bank={self.pc.minerals}")
            if job["state"] == "moving":
                if d is None:
                    job["state"] = "placed"
                    continue
                # Place only once its step has fired and is being executed.
                if job["step"] is None or not job["step"].state.get("active"):
                    continue
                if self.pc.minerals >= job["cost"]:
                    self.act(_cmd(job["ability"], [d.tag], point=job["point"]) if job["point"]
                             else _cmd(job["ability"], [d.tag], target_tag=job["tag"]))
                    self.pc.minerals -= job["cost"]
                    job["state"] = "ordered"
                    job["ordered_at"] = self.loop
            elif job["state"] == "ordered":
                if d is None:
                    job["state"] = "placed"
                elif self.loop - job["ordered_at"] > 120 and not d.orders:
                    job["state"] = "moving"  # placement failed; retry
                    self.event("retry", job["kind"])

    def presend_builders(self):
        """For upcoming structure steps with no rallied builder coming, pull the
        closest mining drone so it arrives just as the bank reaches the cost
        (counting drones still to make before a supply trigger), not earlier:
        until then it keeps mining."""
        for st in self.background + self.steps[self.i : self.i + 3]:
            a = st.action
            if a not in ("hatch", "pool", "roach_warren", "extractor") or st.state.get("done"):
                continue
            base = st.args[0] if st.args else "main"
            if self.job_for(a, base, step=st) or a in self.rally or any(r == a for _, r in self.role_queue):
                continue
            t = self.target_for(a, base)
            dest = t.get("target_point") or _xy(self.units[t["target_tag"]])
            cands = [d for tg, d in self.drones.items() if self.assign.get(tg, ("",))[0] in ("min", "rally")]
            if not cands:
                continue
            builder = self.pick_builder(dest)
            walk = self.walk_loops(_xy(builder), dest) if a == "hatch" else _dist(_xy(builder), dest) / DRONE_SPEED
            if st.trigger[0] == "time" and self.loop + walk < st.trigger[1]:
                continue  # don't arrive before its time
            if st.trigger[0] == "event" and not st.fired:
                continue
            if st.trigger[0] == "now" and not st.background and st is not self.steps[self.i]:
                continue  # no target to time against until it's actually next in the build order
            supply = st.trigger[1] if st.trigger[0] == "supply" else self.pc.food_used
            need = COSTS[a] + COSTS["drone"] * max(0, supply - self.pc.food_used) - self.pc.minerals
            if need / self.income_per_loop() <= walk:
                self.start_job(a, drone=builder, base=base, step=st)

    # ---------- build-order steps ----------
    def triggered(self, st):
        if st.fired:
            return True
        kind = st.trigger[0]
        if kind == "now":
            ok = True
        elif kind == "supply":
            ok = self.pc.food_used >= st.trigger[1]
        elif kind == "time":
            ok = self.loop >= st.trigger[1]
        else:
            ev = st.trigger[1]
            if ev == "pool":
                ok = self.pool is not None and self.pool.build_progress >= 1
            elif ev == "warren":
                ok = self.warren is not None and self.warren.build_progress >= 1
            elif ev in ("pool_soon", "warren_soon"):
                # About to finish: hold larva/money so it's spent the moment it's done.
                b, loops, lead = (self.pool, POOL_LOOPS, 2.0) if ev == "pool_soon" else (self.warren, BUILD_LOOPS["roach_warren"], 1.5)
                ok = b is not None and (1 - b.build_progress) * loops <= lead * 22.4
            elif ev.startswith("extractors:"):
                n = int(ev.split(":")[1])
                ok = sum(1 for e in self.extractors if e.tag not in self.trick_extractors and e.build_progress >= 1) >= n
            elif ev == "roaches":  # a scripted roach batch has started
                ok = any(e[1] == "roaches started" for e in self.events)
            elif ev.endswith("_soon"):
                # The hatchery will be done by the time a drone walks there from the main.
                h = self.hatch_at(ev[:-5])
                walk = (self.map.natural_walk if ev == "natural_soon" else self.map.natural_walk + self.map.third_walk) / DRONE_SPEED
                ok = h is not None and (1 - h.build_progress) * HATCH_BUILD_LOOPS <= walk
            elif ev.startswith("queen:"):
                bi = self.bi(ev.split(":")[1])
                ok = any(self.base_of(_xy(q)) == bi or self.queen_home.get(q.tag) == bi for q in self.queens)
            else:
                ok = self.base_done(ev)
        if ok:
            st.fired = True
            self.event("step", st.text.split(": ", 1)[1])
        return ok

    def run_step(self, st):
        """'done', 'reserve' (fired; save minerals for it) or 'wait'."""
        if not self.triggered(st):
            return "wait"
        st.state["active"] = True
        a, args = st.action, st.args
        if a == "overlord":
            if not self.train(TRAIN_OVERLORD, COSTS["overlord"], args[0] if args else "main"):
                return "reserve"
            self.event("overlord", f"at {self.pc.food_used}/{self.pc.food_cap}")
            return "done"
        if a in ("hatch", "extractor", "pool", "roach_warren"):
            base = args[0] if args else "main"
            job = self.job_for(a, base, step=st)
            if job is None and a != "hatch" and a in self.rally:
                return "reserve"  # its rallied builder hasn't been made yet
            if job is None and a != "hatch" and any(r == a for _, r in self.role_queue):
                return "reserve"  # its rallied builder is still an egg
            if job is None:
                return "reserve"  # presend_builders pulls a mining drone on its timer
            job["step"] = st
            if job["state"] in ("ordered", "placed"):
                st.state["done"] = True
                return "done"
            return "reserve"
        if a == "extractor_trick":
            return "done" if self.do_trick(st) else "reserve"
        if a == "queens":
            # Several queens started together (wait until all can be afforded),
            # so they pop - and inject - in sync. If the pool is about to finish,
            # already hold money (queens before drones).
            if self.pool is None:
                return "wait"
            if self.pool.build_progress < 1:
                return "reserve" if (1 - self.pool.build_progress) * POOL_LOOPS <= 2.0 * 22.4 else "wait"
            hs = [self.hatch_at(b) for b in args]
            if any(h is None or h.build_progress < 1 for h in hs):
                return "wait"
            if any(any(o.ability_id == TRAIN_QUEEN for o in h.orders) for h in hs):
                return "reserve"
            n = len(hs)
            if self.pc.minerals < n * COSTS["queen"] or self.pc.food_used + 2 * n > self.pc.food_cap:
                return "reserve"
            for h, b in zip(hs, args):
                self.act(_cmd(TRAIN_QUEEN, [h.tag]))
                self.event("queen", b)
            self.pc.minerals -= n * COSTS["queen"]
            self.pc.food_used += 2 * n
            st.state["queued"] = True
            return "done"
        if a == "queen":
            h = self.hatch_at(args[0])
            if h is None or h.build_progress < 1 or self.pool is None or self.pool.build_progress < 1:
                return "wait"
            if any(o.ability_id == TRAIN_QUEEN for o in h.orders):
                return "reserve"  # one queen at a time per hatchery
            if self.pc.minerals >= COSTS["queen"] and self.pc.food_used + 2 <= self.pc.food_cap:
                self.act(_cmd(TRAIN_QUEEN, [h.tag]))
                self.pc.minerals -= COSTS["queen"]
                self.pc.food_used += 2
                self.event("queen", args[0])
                st.state["queued"] = True
                return "done"
            return "reserve"
        if a == "roaches":
            # All N at once: from the moment this fires, hold larva and money
            # (no drones) until the warren is done and all N can start together.
            if self.warren is None:
                return "wait"
            n = int(args[0])
            if len(args) > 1 and args[1] == "asap":
                # Start each roach as soon as larva/money/gas/supply allow. If the
                # warren is about to finish (@warren_soon), hold larva for it.
                if self.warren.build_progress < 1:
                    st.state.setdefault("holding_since", self.loop)
                    return "reserve"
                made = st.state.setdefault("made", 0)
                while made < n:
                    if (self.pc.food_used + 2 > self.pc.food_cap or self.pc.vespene < COSTS["roach_gas"]
                            or not self.train(TRAIN_ROACH, COSTS["roach"])):
                        st.state["made"] = made
                        return "reserve"
                    self.pc.vespene -= COSTS["roach_gas"]
                    self.pc.food_used += 2
                    made += 1
                    if made == 1:
                        st.state.setdefault("holding_since", self.loop)
                    self.event("roach", f"{made} started")
                st.state["done_at"] = self.loop
                self.event("roaches started", f"{n}, held larva from {st.state['holding_since'] / 22.4:.1f}s")
                return "done"
            ready = (self.warren.build_progress >= 1 and len(self.larva) >= n
                     and self.pc.minerals >= n * COSTS["roach"] and self.pc.vespene >= n * COSTS["roach_gas"]
                     and self.pc.food_used + 2 * n <= self.pc.food_cap)
            if not ready:
                st.state.setdefault("holding_since", self.loop)
                return "reserve"
            for _ in range(n):
                self.train(TRAIN_ROACH, COSTS["roach"])
            self.pc.vespene -= n * COSTS["roach_gas"]
            self.pc.food_used += 2 * n
            st.state["done_at"] = self.loop
            self.event("roaches started", f"{n}, held larva from {st.state['holding_since'] / 22.4:.1f}s")
            return "done"
        if a == "lings":
            made = st.state.setdefault("made", 0)
            while made < int(args[0]):
                if self.pc.food_used + 1 > self.pc.food_cap or not self.train(TRAIN_ZERGLING, COSTS["ling_pair"]):
                    st.state["made"] = made
                    return "reserve"
                made += 1
                self.pc.food_used += 1
                self.event("lings", f"pair {made}")
            return "done"
        if a == "lair":
            h = self.hatch_at("main")
            if self.pool is None or self.pool.build_progress < 1 or h is None or h.unit_type != HATCHERY:
                return "wait"
            if self.pc.vespene < COSTS["lair_gas"]:
                return "wait"
            if h.orders or self.pc.minerals < COSTS["lair"]:
                return "reserve"
            self.act(_cmd(MORPH_LAIR, [h.tag]))
            self.pc.minerals -= COSTS["lair"]
            self.pc.vespene -= COSTS["lair_gas"]
            self.event("lair started", f"bank {self.pc.minerals}/{self.pc.vespene}")
            return "done"
        if a == "ling_speed":
            if self.pool is None or self.pool.build_progress < 1 or self.pc.vespene < COSTS["ling_speed_gas"]:
                return "wait"
            if self.pc.minerals < COSTS["ling_speed"]:
                return "reserve"
            self.act(_cmd(LING_SPEED, [self.pool.tag]))
            self.pc.minerals -= COSTS["ling_speed"]
            self.event("ling speed", f"bank gas {self.pc.vespene}")
            return "done"
        if a == "transfer":
            n, src, dst = int(args[0]), self.bi(args[1]), args[2]
            moved = 0
            for _ in range(n):
                t = self.take_drone_from(src)
                if t is None:
                    break
                self.move_drone(t, lambda d: self.send_to(d, dst))
                moved += 1
            self.event("transfer", f"{moved} drones {args[1]} -> {dst}")
            return "done"
        if a == "gas_pull":
            # Pull drones off this base's mineral line into its extractor now.
            n, base = int(args[0]), args[1]
            moved = 0
            for _ in range(n):
                ext = self.extractor_needing_workers(base)
                if ext is None:
                    break
                t = self.take_drone_from(self.bi(base))
                if t is None:
                    break
                def to_gas(d, ext=ext):
                    self.assign[d.tag] = ("gas", ext.tag)
                    self.act(_cmd(GATHER, [d.tag], target_tag=ext.tag) if ext.build_progress >= 1 else _cmd(MOVE, [d.tag], point=_xy(ext)))
                self.move_drone(t, to_gas)
                moved += 1
            self.event("gas pull", f"{moved} drones {base}")
            return "done"
        if a == "rally":
            n = int(args[1][1:]) if len(args) == 2 else 1
            self.rally.extend([args[0]] * n)
            return "done"
        if a == "default":
            self.default = args[0]
            return "done"
        if a == "pull_gas":
            self.pull_gas_at = int(args[0])
            return "done"
        if a == "sync_injects":
            self.sync_injects = True
            return "done"
        if a == "cycle_overlords":
            self.cycle_overlords = True
            self.auto_overlords = False
            return "done"
        if a == "auto_overlords":
            self.auto_overlords = True
            return "done"
        raise ValueError(a)

    def do_trick(self, st):
        t = st.state.get("trick")
        if t is None:
            t = st.state["trick"] = dict(state="build", seen=set(e.tag for e in self.extractors))
            self.start_job("trick_extractor", step=st)
        if t["state"] == "build":
            ext = next((e for e in self.extractors if e.build_progress < 1 and e.tag not in t["seen"]), None)
            if ext:
                t.update(state="drone", extractor=ext.tag)
                self.trick_extractors.add(ext.tag)
            return False
        if t["state"] == "drone":
            if self.make_drone():
                t["state"] = "cancel"
            return False
        if t["state"] == "cancel":
            self.act(_cmd(CANCEL_BUILD, [t["extractor"]]))
            t["state"] = "wait"
            self.event("trick cancel")
            return False
        if t["state"] == "wait":
            if t["extractor"] not in self.units:
                t["state"] = "done"
                self.event("trick done", f"supply {self.pc.food_used}/{self.pc.food_cap}")
                return True
            return False
        return True

    def drone_cap(self):
        """Never drone past the next supply-triggered step that hasn't run."""
        for st in self.steps[self.i:]:
            if st.trigger[0] == "supply" and not st.background:
                return st.trigger[1]
        return 200

    def injects(self):
        injectors = {}
        for q in sorted(self.queens, key=lambda q: q.tag):
            home = self.queen_home.get(q.tag)
            if home is None:
                home = self.base_of(_xy(q))
                self.queen_home[q.tag] = home
            injectors.setdefault(home, q)
        for q in self.queens:
            home = self.queen_home[q.tag]
            if injectors.get(home) is not q:
                # A spare queen takes a finished hatchery without an injector.
                spare = [self.base_of(_xy(h)) for h in self.done_hatches if self.base_of(_xy(h)) not in injectors]
                if not spare:
                    continue
                self.queen_home[q.tag] = home = spare[0]
                injectors[home] = q
            if q.energy < 25 or (q.orders and q.orders[0].ability_id == INJECT):
                continue
            if self.sync_injects:
                # Every round waits until each finished hatchery's queen is
                # ready, so their injects pop together (queens drift apart otherwise).
                done = {self.base_of(_xy(h)) for h in self.done_hatches}
                if not done <= set(injectors) or any(injectors[b].energy < 25 for b in done):
                    continue
            h = next((h for h in self.done_hatches if self.base_of(_xy(h)) == home), None)
            if h is not None:
                self.act(_cmd(INJECT, [q.tag], target_tag=h.tag))
                self.injected.add(q.tag)

    def tick(self):
        reserved = False
        # Overlords first: the forecast only asks for one when larva would
        # otherwise sit idle, so it outranks the scripted steps.
        self.track_injects()
        if self.cycle_overlords:
            while self.cycle_owed > 0 and self.loop <= self.cycle_until and self.train(TRAIN_OVERLORD, COSTS["overlord"]):
                self.cycle_owed -= 1
                self.training[TRAIN_OVERLORD] += 1
                self.event("overlord", f"cycle at {self.pc.food_used}/{self.pc.food_cap}")
            if self.cycle_owed > 0 and self.loop <= self.cycle_until:
                reserved = True  # this cycle's overlords come first
            # Safety net: actually blocked with larva waiting and nothing coming.
            if self.larva and self.pc.food_used >= self.pc.food_cap and self.training[TRAIN_OVERLORD] == 0:
                if self.train(TRAIN_OVERLORD, COSTS["overlord"]):
                    self.training[TRAIN_OVERLORD] += 1
                    self.event("overlord", f"emergency at {self.pc.food_used}/{self.pc.food_cap}")
        if self.auto_overlords:
            for _ in range(self.overlords_to_start()):
                if not self.train(TRAIN_OVERLORD, COSTS["overlord"]):
                    reserved = True
                    break
                self.training[TRAIN_OVERLORD] += 1
                self.event("overlord", f"auto at {self.pc.food_used}/{self.pc.food_cap}")
        for st in list(self.background):
            r = self.run_step(st)
            if r == "done":
                self.background.remove(st)
            elif r == "reserve":
                reserved = True
        while self.i < len(self.steps):
            st = self.steps[self.i]
            if st.background:
                self.background.append(st)
                self.i += 1
                continue
            r = self.run_step(st)
            if r != "done":
                reserved = reserved or r == "reserve"
                break
            self.i += 1
        self.presend_builders()
        self.advance_jobs()
        if not reserved:
            cap = self.drone_cap()
            while self.pc.food_used < cap and self.make_drone():
                pass
        # A hatchery at 3 larva stops spawning: spend one there even while
        # saving for something, unless the larva is kept for a roach batch.
        per = Counter(self.base_of(_xy(l)) for l in self.larva)
        if (any(v >= 3 for v in per.values()) and len(self.larva) > self.larva_wanted()
                and self.pc.food_used < self.pc.food_cap and self.pc.minerals >= COSTS["drone"]):
            self.make_drone()
        self.manage_gas()
        if self.loop % 22 == 0:
            self.rebalance()
        self.manage_workers()
        self.injects()
        # Metrics: supply blocked only counts while larva sits idle.
        blocked = bool(self.larva) and self.pc.food_used >= self.pc.food_cap
        if blocked:
            self.blocked_loops += 1
            self.blocked_since = getattr(self, "blocked_since", None) or self.loop
        elif getattr(self, "blocked_since", None):
            self.event("supply blocked", f"{(self.loop - self.blocked_since) / 22.4:.1f}s from {self.blocked_since / 22.4:.1f}s")
            self.blocked_since = None
        self.drone_loops += self.n_drones()
        per_hatch = Counter(self.base_of(_xy(l)) for l in self.larva)
        capped = sum(1 for h in self.done_hatches if per_hatch[self.base_of(_xy(h))] >= 3)
        if capped:
            self.capped_loops += capped
            if any(st.action == "roaches" and "holding_since" in st.state and not st.state.get("done_at")
                   for st in self.background + self.steps[self.i:self.i + 1]):
                self.hold_capped_loops += capped

    def n_drones(self):
        """Drones that exist (incl. ones inside an extractor), not eggs."""
        return len(self.drones) + sum(1 for t, a in self.assign.items() if a[0] == "gas" and t not in self.drones)


def main(argv):
    del argv
    build_path = resolve_build(FLAGS.build)
    steps = parse_build(build_path)
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    run_config = run_configs.get()
    with run_config.start(want_rgb=False) as controller:
        if os.path.exists(FLAGS.map):
            # A map file anywhere (e.g. copied out of the Battle.net cache): pass
            # its absolute path. Sending map_data instead makes SC2 write it to a
            # temp file shared by every instance, which races in parallel runs.
            local_map = sc_pb.LocalMap(map_path=os.path.abspath(FLAGS.map))
        else:
            local_map = sc_pb.LocalMap(map_path=FLAGS.map, map_data=run_config.map_data(FLAGS.map))
        create = sc_pb.RequestCreateGame(
            local_map=local_map,
            realtime=False,
            random_seed=FLAGS.seed,
        )
        # Slots 1..team_size are one team on standard melee maps.
        create.player_setup.add(type=sc_pb.Participant)
        for _ in range(FLAGS.team_size - 1):
            create.player_setup.add(type=sc_pb.Computer, race=sc_common.Zerg, difficulty=sc_pb.VeryEasy)
        for _ in range(FLAGS.team_size):
            create.player_setup.add(type=sc_pb.Computer, race=sc_common.Zerg, difficulty=sc_pb.VeryEasy)
        controller.create_game(create)
        controller.join_game(sc_pb.RequestJoinGame(race=sc_common.Zerg, options=sc_pb.InterfaceOptions(raw=True, score=True)))

        game_data = load_game_data(controller)
        bench = Bench(controller, steps)
        obs = controller.observe()
        bench.observe(obs)
        bench.map = MapInfo(controller, obs.observation.raw_data.units, _xy(bench.hatches[0]))
        m = bench.map
        print(f"spawn {m.spawn_index} at {m.main}; natural {m.named['natural']['spot']} (walk {m.natural_walk:.1f}), "
              f"third {m.named['third']['spot']} (walk {m.third_walk:.1f} from natural)")
        if FLAGS.spawn >= 0 and m.spawn_index != FLAGS.spawn:
            print(f"seed {FLAGS.seed} gives spawn {m.spawn_index}, wanted {FLAGS.spawn}")
            sys.exit(3)
        bench.pool_spot = find_building_spot(controller, bench, BUILD_POOL)
        bench.warren_spot = find_building_spot(controller, bench, BUILD_ROACH_WARREN, avoid=[bench.pool_spot])
        for u in bench.mine:  # the starting hatchery isn't an event
            bench.structures_seen[u.tag] = 0
            bench.done_seen.add(u.tag)
        bench.suppress_enemy(obs)
        bench.initial_split()

        meta = dict(
            map=FLAGS.map, build=os.path.basename(build_path), build_file=open(build_path).read(), seed=FLAGS.seed,
            spawn=m.spawn_index, main=m.main, natural=m.named["natural"]["spot"], third=m.named["third"]["spot"],
            natural_walk=m.natural_walk, third_walk=m.third_walk, pool_spot=bench.pool_spot, game_data=game_data,
        )
        with open(os.path.join(FLAGS.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        ts = open(os.path.join(FLAGS.out_dir, "timeseries.csv"), "w", newline="")
        tw = csv.writer(ts)
        tw.writerow([
            "loop", "minerals", "vespene", "collected_minerals", "collected_vespene", "spent_minerals",
            "food_used", "food_cap", "drones", "drones_training", "larva", "gas_drones",
            "main_drones", "natural_drones", "third_drones", "blocked_loops", "drone_seconds", "roaches",
            "capped_loops", "hold_capped_loops",
        ])
        for _ in range(FLAGS.loops):
            if bench.pending_cmds:
                controller.actions(sc_pb.RequestAction(actions=bench.pending_cmds))
                bench.pending_cmds = []
            controller.step(1)
            obs = controller.observe()
            if obs.player_result:
                print("Game ended early.")
                break
            bench.observe(obs)
            if bench.loop % 16 == 0:
                bench.suppress_enemy(obs)
            bench.tick()
            s = bench.score
            tw.writerow([
                bench.loop, bench.pc.minerals, bench.pc.vespene, s.collected_minerals, s.collected_vespene, s.spent_minerals,
                bench.pc.food_used, bench.pc.food_cap, bench.n_drones(), bench.training[TRAIN_DRONE], len(bench.larva),
                sum(1 for a in bench.assign.values() if a[0] == "gas"),
                bench.mineral_drones("main"), bench.mineral_drones("natural"), bench.mineral_drones("third"),
                bench.blocked_loops, round(bench.drone_loops / 22.4, 1), bench.roaches_seen,
                bench.capped_loops, bench.hold_capped_loops,
            ])
        ts.close()
        with open(os.path.join(FLAGS.out_dir, "events.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["loop", "event", "detail"])
            w.writerows(bench.events)
        print(f"collected {bench.score.collected_minerals} minerals, {bench.score.collected_vespene} gas; "
              f"supply {bench.pc.food_used}/{bench.pc.food_cap}; blocked with idle larva {bench.blocked_loops / 22.4:.1f}s")


def find_building_spot(controller, bench, ability, avoid=()):
    """A 3x3 spot close to the main hatchery, on the side away from the
    minerals and clear of the hatchery->geyser lanes (gas drones' paths)."""
    main = bench.map.main
    mins = bench.map.named["main"]["minerals"]
    mx = np.mean([bench.units[m].pos.x for m in mins])
    my = np.mean([bench.units[m].pos.y for m in mins])
    away = np.array([main[0] - mx, main[1] - my])
    away /= np.linalg.norm(away)
    geysers = [_xy(bench.units[g]) for g in bench.map.named["main"]["geysers"] if g in bench.units]

    def in_lane(p, width=3.0):
        for g in geysers:
            a, b, q = np.array(main), np.array(g), np.array(p)
            t = np.clip(np.dot(q - a, b - a) / np.dot(b - a, b - a), 0, 1)
            if np.linalg.norm(q - (a + t * (b - a))) < width:
                return True
        return False
    cands = []
    for dx in range(-9, 10):
        for dy in range(-9, 10):
            p = (math.floor(main[0]) + dx + 0.5, math.floor(main[1]) + dy + 0.5)
            if 5.5 <= _dist(p, main) <= 9 and all(_dist(p, a) >= 3.5 for a in avoid) and not in_lane(p):
                cands.append(p)
    target = (main[0] + 6 * away[0], main[1] + 6 * away[1])
    cands.sort(key=lambda p: _dist(p, target))
    res = controller.query(sc_query.RequestQuery(
        placements=[sc_query.RequestQueryBuildingPlacement(ability_id=ability, target_pos=_p2(*p)) for p in cands]))
    for p, r in zip(cands, res.placements):
        if r.result == 1:
            return p
    raise RuntimeError(f"no spot for ability {ability}")


if __name__ == "__main__":
    app.run(main)
