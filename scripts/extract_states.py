"""Extracts full-information game states from replays for the world model (stage 1).

Each replay is played back in the engine as an observer (no player perspective) with
fog disabled, start to end, and the whole state is dumped every --step game loops.
Observing as a player instead hides private details of everyone else's units even with
fog off (production and its progress, add-ons, cargo; undetected cloaked units come
back with owner 0 and no HP), so observer mode is the only complete view. Commands
aren't extracted here: they come from the replay's own command events (see
unit_tag below for linking them to these units). Unit orders (move, attack, gather, train,
...) and their targets are kept (ord_*): they are the part of the coming change that is
already decided, so a single frame with them is close to a Markov state. Output is one
packed state file (.cst, worldmodel/statefile.py: per-unit deltas + zstd, read with
statefile.load) per replay in --out:

    frame_loop[F]            game loop of each frame
    frame_start[F+1]         unit rows of frame f are u_*[frame_start[f]:frame_start[f+1]]
    u_*[N]                   one row per (frame, unit), see U_DTYPES
    prod_abil, prod_progress production queue of producers (structures, eggs, cocoons, larva:
                             from the game's own unit data): what each entry produces and its
                             progress. Entries in progress come first (2 with a reactor), the
                             rest are queued (progress 0). Some morphs (orbital, lair, hive)
                             report progress 0 throughout.
    b_id / pass_*            buffs, passengers
    grp_*                    every player's control groups 0-9 and active selection (10) per
                             frame, from the replay's selection events (worldmodel/
                             selection_tracker.py): grp_player[P] the slots; group g of player
                             k at frame f is grp_id[grp_start[(f*P+k)*11+g]:grp_start[(f*P+k)*11+g+1]],
                             state unit ids in the game's order (ascending tag), dead members
                             left out. Player-view files have the same with P = 1 (their player)
    ord_*                    the unit's orders in queue order: ability, target point (NaN if
                             none), target unit id (-1 if none), progress
                             all ragged per unit row: row i owns [u_<x>_start[i], u_<x>_start[i+1])
    e_*, e_frame_start[F+1]  effects (storms, biles, scans, ...), one row per position
    dead_id, dead_frame_start[F+1]
                             units (already seen) that died since the previous frame (player
                             views before 2026-10-30 miss the deaths reported in the per-loop
                             observations --substep discards, about 3 in 4)
    map_pathing_grid, map_placement_grid, map_terrain_height
                             the map's static grids (formats in meta)
    p_*                      per-player stats from the tracker (all players, every 160 loops)
    up_loop, up_player, up_name
                             upgrades completed (tracker)
    unit_tag[U]              engine tag of each per-replay unit id. The replay's own events
                             use replay tags = (index << 18) + recycle, and
                             engine tag = 2**32 + replay tag (verified exact on 5.0.14 and
                             5.0.17). Commands at fogged things carry tag 0 plus a snapshot
                             (type, position) instead.
    meta (json string)       map, playable area, start locations, players (race, team, result,
                             MMR), build, ruleset key, grid formats, layout

Not here: creep (an observer's creep map is always empty; creep's sources - hatcheries,
tumors, overlords - are in the state) and commands (from the replay's own events).

Every player's view of the replay is extracted (observed_player_id = slot, fog on), exactly as
the engine shows it to that player, to the games tree (worldmodel/corpus.py):
data/games/<format>/<build>/<map>/<id>.p<slot>.cst, next to the game's action table
(<id>.acts.npz) and its extraction sidecar (<id>.meta.json: per view the layout, step,
substep, stats, extractor commit; a failed view leaves <id>.p<slot>.fail.json). The observer
view (fog off) is extracted only with --observer, for its neutrals; it reports no orders, production,
passengers or effects, so a player's own units come from that player's view. Each view has the layout above plus:

    vis[F, H, W]             the player's visibility map per frame (0 never seen, 1 fogged,
                             2 visible; the engine's own map_state)
    creep[F, H, W]           creep as that player knows it
    pathable[F, H, W / 8]    the player's pathing per frame, bit packed like map_pathing_grid: changes with buildings and
                             rocks, as that player last saw them (the minimap feature layer, flipped to map_state's rows)
    buildable[F, H, W / 8]   the same for placement
    u_display, u_cloak       the engine's display_type (1 visible, 2 snapshot, 3 hidden) and
                             cloak state per unit, as reported (also in observer files)
    cmd_*                    every command the engine executed for this player, with its
                             recipients (worldmodel/README.md, "Command recipients"):
                             cmd_loop (the loop it was issued, exact), cmd_frame (the frame
                             whose observation window reported it), cmd_abil (API ability),
                             cmd_target (0 none, 1 point, 2 unit), cmd_tx/cmd_ty, cmd_target_id,
                             cmd_queued; recipients of command c are the state unit ids
                             cmd_unit[cmd_start[c]:cmd_start[c+1]], in the engine's order
    sel_loop[S], sel_start[S+1], sel_id
                             the engine's is_selected set (state unit ids) at every loop it
                             changed: the player's selection as of loop L is the last snapshot
                             with sel_loop <= L (the per-loop observations are read for this)
    ui_grp_count, ui_grp_leader [F, 10], ui_sel_panel[F], ui_sel_type / ui_sel_start[F+1]
                             with --ui: the engine's own summary of the same (ui_data: count and
                             leader type per group, the selection panel's unit types), for
                             validating grp_* (needs the feature layer on)

Layout 2 (meta["layout"] == 2, 2026-10-30): everything else the raw API reports, nothing left
on the table for the model to need later:

    u_flags2                 F2_ACTIVE (a structure that is doing something), F2_BLIP (radar
                             blip), F2_ONSCREEN (on the observed player's screen)
    u_atk_up, u_arm_up, u_shd_up
                             the unit's attack / armor / shield upgrade levels
    u_buff_remain, u_buff_max
                             the engine's buff timer (loops remaining and total, for the one
                             buff it chooses to show; 0 without)
    u_ideal_harvesters, u_cargo_max, u_detect_range, u_radar_range
    rally_x, rally_y, rally_target
                             rally points, ragged per unit row (u_rally_start): point (NaN if
                             none) and target unit id (-1 if none); a town hall has two (units,
                             workers). Player views only: like power sources, the observer view
                             gets none (private details, as with production and cargo)
    ab_id, ab_point          the abilities the unit can use right now (u_ab_start; costs ignored):
                             the API's only cooldown signal, a used Blink / Stim / bile is absent
                             until its cooldown ends. One query per frame for the observed
                             player's visible units (meta["abilities_queried"]; the observer
                             view gets no answer for anyone and skips it)
    pc_*[F]                  the observed player's player_common per frame: minerals, vespene,
                             food_cap / used / army / workers, idle_worker_count, army_count,
                             warp_gate_count, larva_count (exact, every frame; the tracker's
                             p_* are every 160 loops)
    sc_*[F]                  the engine's score details per frame: collected / spent / killed /
                             lost / friendly-fire minerals and vespene per category (army,
                             economy, technology, upgrade, none), total_value_units /
                             structures, killed_value_*, damage dealt / taken / healed per vital
                             (life, shields, energy), idle production / worker time, apm.
                             Zeros in the observer view (no observed player)
    cam[F, 2]                the observed player's camera center
    pw_*, pw_frame_start     power sources (pylons, warp prisms): position, radius, unit id
    rd_*, rd_frame_start     radar rings (sensor towers)
    pu_id, pu_frame_start    the raw player block's upgrade ids per frame (the engine's own
                             list; up_* from the tracker names them)
    cam_save[F, P, 8, 2], subgroup[F, P], lc_*[F, P], ac_*
                             the rest of the players' interface memory, from the replay's
                             events like grp_* (selection_tracker.frame_memory): camera hotkey
                             positions, the selection's active subgroup (tab), the last command
                             (what a repeat re-fires), autocast toggles with their recipients
    p_*                      now every PlayerStats field of the tracker (the nine classic
                             names plus minerals_killed_army, vespene_lost_economy,
                             minerals_used_in_progress_technology, ...; see stat_name)

Nothing is derived or masked here: other players' private details, snapshots of structures in
fog and undetected cloaked units are whatever the engine reports for that player.

The engine's client keeps a unit that left the player's vision in the observation for 47
observation *requests*, not game loops (measured: 2.1 s at one request per loop, 8.4 s at one
per 4 loops, longer still at 8), so a player view stepped several loops per observation leaks
enemy movement through the fog. --substep 1 (the default for player views) requests an
observation every loop and discards the intermediate ones unparsed, which reproduces the
loop-by-loop observation exactly (verified on 200 s of one game, all three views: visible
units identical at every loop) for ~15% more time. Observer views have no fog and skip it.
Snapshot units (display 2) carry client-assigned tags that differ between runs.

Unit tags are mapped to small per-replay ids (u_id); anything that points at a unit
(add-ons, engaged targets, passengers, deaths) uses the same ids (-1 = none).

Parsing: observations are protobuf; the pure-Python parser was 44% of the frame time. Run
scripts/regen_s2proto.py once per environment to rewrite the s2clientprotocol bindings in the
modern style, and the workers use the C++ (upb) parser automatically (protobuf_impl).

Request costs (measured, 200 s of a game, upb): step calls are nearly free whatever their
size (4 x step(1) costs the same as step(4)); each observation request costs ~2 ms of engine
time (it serializes the raw observation) and parsing it ~0.1 ms; a RequestPing is cheap but
does NOT advance the client's fog timers, only observation requests do. So for a player view
the per-loop observation is required anyway (see --substep), and keeping and parsing every one
of them (--step 1) costs about the same as discarding three of four: ~16 s per 200 s of game
plus our own per-frame processing. The Windows client has no headless mode; its window is
minimized after launch (minimize_client_window), which made stepping ~20% faster.

Usage:
    python scripts/extract_states.py --format 1v1 --step 4 --workers 12     # every 1v1 game not yet done
"""

import argparse
import glob
import collections
import faulthandler
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

LPS = 22.4
EXCLUDED_BUILDS = {97364, 97425, 97563}  # patch 5.0.16, reverted
# Importing the Wine run config registers it with pysc2: on Linux the Windows client runs under
# Wine/Proton from a Lutris prefix (cosmicstar/envs/wine_run_config.py; SC2PATH overrides).
from cosmicstar.envs.wine_run_config import battlenet_cache_dirs, sc2_install_dir  # noqa: E402
from cosmicstar.envs import depot  # noqa: E402  maps and mods by hash (the client won't fetch them)
from cosmicstar.worldmodel import corpus  # noqa: E402  the replay and games trees

SC2_DIR = sc2_install_dir() or r"C:\Program Files (x86)\StarCraft II"
BALANCE_DIR = "data/worldmodel/balance"  # Editor exports per ruleset (export_balance.py)
CACHE_DIRS = battlenet_cache_dirs(SC2_DIR)
FETCH_MISSING = True  # fetch a replay's missing map/mods from the depot into CACHE_DIRS[0] (--no_fetch off)
HERE = os.path.dirname(os.path.abspath(__file__))

U_DTYPES = dict(  # one row per (frame, unit)
    id=np.int32, type=np.int16, owner=np.int8, x=np.float32, y=np.float32, z=np.float16, facing=np.float16,
    radius=np.float16, hp=np.float32, hp_max=np.float32, shield=np.float32, shield_max=np.float32,
    energy=np.float32, build=np.float16, flags=np.uint8, contents=np.int16, harvesters=np.int16,
    add_on=np.int32, weapon_cooldown=np.float16, engaged=np.int32, cargo_used=np.int16,
    display=np.int8, cloak=np.int8,
    # layout 2 (2026-10-30): the rest of the raw Unit
    flags2=np.uint8, atk_up=np.int8, arm_up=np.int8, shd_up=np.int8, buff_remain=np.int16, buff_max=np.int16,
    ideal_harvesters=np.int16, cargo_max=np.int16, detect_range=np.float16, radar_range=np.float16,
    prod_start=np.int64, buff_start=np.int64, pass_start=np.int64, ord_start=np.int64, rally_start=np.int64, ab_start=np.int64)
R_DTYPES = dict(  # ragged: unit row i owns [<x>_start[i], <x>_start[i+1]) (the last row: to the end)
    prod_abil=np.int16, prod_progress=np.float16, b_id=np.int16, pass_id=np.int32, pass_type=np.int16, pass_hp=np.float32,
    ord_abil=np.int16, ord_tx=np.float32, ord_ty=np.float32, ord_target=np.int32, ord_progress=np.float16,
    rally_x=np.float32, rally_y=np.float32, rally_target=np.int32, ab_id=np.int32, ab_point=np.bool_)
E_DTYPES = dict(type=np.int16, owner=np.int8, x=np.float32, y=np.float32, radius=np.float16)
# per-frame ragged tables from the raw observation's player block (row i of frame f: [<x>_frame_start[f], ...[f+1]))
PW_DTYPES = dict(x=np.float32, y=np.float32, radius=np.float16, id=np.int32)  # power sources (pylons, warp prisms)
RD_DTYPES = dict(x=np.float32, y=np.float32, radius=np.float16)  # radar rings (sensor towers)
CMD_DTYPES = dict(  # player views: the engine's executed commands (recipients: cmd_unit, ragged by cmd_start)
    loop=np.int32, frame=np.int32, abil=np.int32, target=np.int8, tx=np.float32, ty=np.float32,
    target_id=np.int32, queued=np.bool_, start=np.int64)

# Unit flags packed into u_flags (F_SELECTED: the observed player's selection, player views only).
F_FLYING, F_BURROWED, F_CLOAKED, F_HALLUC, F_SNAPSHOT, F_POWERED, F_SELECTED = 1, 2, 4, 8, 16, 32, 64
F2_ACTIVE, F2_BLIP, F2_ONSCREEN = 1, 2, 4  # u_flags2: is_active (a structure doing something), radar blip, on the player's screen
LAYOUT = 3  # meta["layout"]: 2 = everything the raw API reports (rally, score, player_common, abilities, ...); 3 = and per-frame pathable, buildable

PLAYER_STATS = {  # tracker PlayerStats field -> output name (food is fixed point /4096)
    "m_scoreValueMineralsCurrent": "minerals",
    "m_scoreValueVespeneCurrent": "vespene",
    "m_scoreValueMineralsCollectionRate": "mineral_rate",
    "m_scoreValueVespeneCollectionRate": "vespene_rate",
    "m_scoreValueWorkersActiveCount": "workers",
    "m_scoreValueFoodUsed": "food_used",
    "m_scoreValueFoodMade": "food_made",
    "m_scoreValueMineralsUsedActiveForces": "army_minerals",
    "m_scoreValueVespeneUsedActiveForces": "army_vespene",
}


# ---------------------------------------------------------------- replay metadata

def stat_name(key):
    """m_scoreValueMineralsLostArmy -> minerals_lost_army (the tracker's other stats: killed,
    lost, friendly fire, used current / in progress, per army / economy / technology)."""
    key = key[len("m_scoreValue"):] if key.startswith("m_scoreValue") else key.lstrip("m_")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()


from cosmicstar.worldmodel.corpus import _protocol, _s, replay_info, ruleset_key  # noqa: E402,F401  (the replay readers)


def run_config_for(info):
    """(run_config, current): the current client plays its own replays as is; older base
    builds need their binary and data version, plus the map passed in."""
    from pysc2 import run_configs
    from pysc2.run_configs import lib as rcl
    current = info["base_build"] == max(installed_builds())
    if current:
        return run_configs.get(), True
    return run_configs.get(version=rcl.Version(info["game_version"], info["base_build"], info["data_version"],
                                                "SC2_x64.exe")), False


def map_path(map_hash, handles=None):
    """The cached map archive; with the replay's handles and FETCH_MISSING, fetches the map and
    any missing mod from the depot first."""
    p = depot.find_cached(CACHE_DIRS, map_hash, "s2ma")
    if handles and FETCH_MISSING and (p is None or any(depot.find_cached(CACHE_DIRS, h, ext) is None
                                                        for ext, h in handles)):
        # a cached map with an uncached mod (a tournament's balance mod) fails to launch too
        depot.ensure_cached(handles, CACHE_DIRS, log=lambda m: print(m, flush=True))
        p = depot.find_cached(CACHE_DIRS, map_hash, "s2ma")
    return p


def tracker_tables(path, until):
    """Per-player stats and upgrades for every player, from the tracker events."""
    import mpyq
    a = mpyq.MPQArchive(path)
    proto, _ = _protocol(a)
    stats = collections.defaultdict(list)  # every m_scoreValue* field (PLAYER_STATS names the classic ones)
    ups = {"loop": [], "player": [], "name": []}
    for e in proto.decode_replay_tracker_events(a.read_file("replay.tracker.events")):
        if e["_gameloop"] > until:
            break
        t = e["_event"].split(".")[-1]
        if t == "SPlayerStatsEvent":
            st = e["m_stats"]
            stats["loop"].append(e["_gameloop"])
            stats["player"].append(e["m_playerId"])
            for k, v in st.items():
                name = PLAYER_STATS.get(k) or stat_name(k)
                stats[name].append(v / 4096 if name.startswith("food") else v)
        elif t == "SUpgradeEvent" and e["_gameloop"] > 0 and not _s(e["m_upgradeTypeName"]).startswith(COSMETIC):
            ups["loop"].append(e["_gameloop"])
            ups["player"].append(e["m_playerId"])
            ups["name"].append(_s(e["m_upgradeTypeName"]))
    return stats, ups


# ---------------------------------------------------------------- engine playback

ENGINE_TAG_OFFSET = 1 << 32  # engine tag = 2**32 + replay tag
COSMETIC = ("Spray", "Reward")  # cosmetic "upgrades": unlocks at loop 0, sprays used in game
STRUCTURE = 8  # Attribute.Structure
# Production abilities by the game data's own naming: train, research, add-ons, morphs
# (orbital, lair, hive, baneling, overseer, warp gate...). Excludes structures' other
# orders: lift/land, root/uproot, depot lower, tumor spread, attack, move, recharge.
PRODUCTION = re.compile(r"Train|Research|AddOns|UpgradeTo|MorphTo|MorphBuilding", re.I)
# `kill -USR1 <worker pid>` prints every thread's Python stack to the worker's stderr (the
# workers' stderr goes to <out>/workers.log): the tool for a worker that sits on a stuck engine
if hasattr(faulthandler, "register") and hasattr(__import__("signal"), "SIGUSR1"):
    faulthandler.register(__import__("signal").SIGUSR1, all_threads=True)
LAUNCH_LOCK = os.path.join(tempfile.gettempdir(), "cosmicstar_sc2_launch.lock")
ENGINE_TIMEOUT = 90   # seconds a worker waits for the engine to launch and start the replay (map load included)
STEP_TIMEOUT = 30     # seconds for any request once the replay runs (a step or observation normally takes milliseconds)
LAUNCH_SETTLE = 8     # seconds the launch lock is held after the replay started, for the engine's shared temp files


class launch_lock:
    """Serializes replay launches across worker processes. Passing map_data makes SC2
    write a shared temp map file; concurrent launches can load each other's map, which
    shows up as InvalidReplayData or a desync."""

    def __enter__(self):
        # one lock per Wine prefix: engines in different pool prefixes share nothing
        pfx = os.environ.get("WINEPREFIX", "")
        path = LAUNCH_LOCK if not pfx else LAUNCH_LOCK[:-5] + "_" + hashlib.sha1(pfx.encode()).hexdigest()[:8] + ".lock"
        self.f = open(path, "a+")
        if sys.platform == "win32":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
                    return self
                except OSError:
                    time.sleep(0.2)
        import fcntl
        fcntl.flock(self.f.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if sys.platform == "win32":
            import msvcrt
            self.f.seek(0)
            msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
        self.f.close()


def tracker_lifecycle(path, until):
    """Engine tag -> loop of each unit's birth/init and death, from the replay's own log."""
    import mpyq
    a = mpyq.MPQArchive(path)
    proto, _ = _protocol(a)
    born, died = {}, {}
    for e in proto.decode_replay_tracker_events(a.read_file("replay.tracker.events")):
        if e["_gameloop"] > until:
            break
        t = e["_event"].split(".")[-1]
        if t in ("SUnitBornEvent", "SUnitInitEvent", "SUnitDiedEvent"):
            tag = ENGINE_TAG_OFFSET + (e["m_unitTagIndex"] << 18) + e["m_unitTagRecycle"]
            (died if t == "SUnitDiedEvent" else born).setdefault(tag, e["_gameloop"])
    return born, died


class DesyncCheck:
    """Checks the engine's simulation against the replay's own unit log, exactly:

        undead   a unit the replay says died is still in the engine `grace` loops later
        phantom  the engine shows a unit the replay never recorded being born

    A diverged simulation trips one of these within seconds. Unlike counting units it
    isn't fooled by units the engine doesn't show (inside gas buildings, transports) or
    by type changes (burrowed drones are a different type)."""

    def __init__(self, born, died, grace=45):
        self.born, self.died, self.grace = born, died, grace

    def check(self, loop, tags):
        for t in tags:
            d = self.died.get(t)
            if d is not None and loop > d + self.grace:
                raise RuntimeError(f"desync at {loop / LPS:.0f}s: unit {t} died at {d / LPS:.0f}s in the replay but is alive")
            b = self.born.get(t)
            if b is None or b > loop + self.grace:
                raise RuntimeError(f"desync at {loop / LPS:.0f}s: unit {t} isn't in the replay's unit log")


class TypedList:
    """A growing column held as typed arrays instead of Python objects: append, extend and len as a list, np.asarray
    for the whole. Python lists of a long game's available abilities alone reached about 7 GB at one frame per loop."""

    def __init__(self, dtype, flush=1 << 16):
        self.dtype, self.flush_at = np.dtype(dtype), flush
        self.chunks, self.pending, self.n = [], [], 0

    def _flush(self):
        if self.pending:
            self.chunks.append(np.asarray(self.pending, dtype=self.dtype))
            self.pending = []

    def append(self, v):
        self.pending.append(v)
        self.n += 1
        if len(self.pending) >= self.flush_at:
            self._flush()

    def extend(self, vs):
        before = len(self.pending)
        self.pending.extend(vs)
        self.n += len(self.pending) - before
        if len(self.pending) >= self.flush_at:
            self._flush()

    def __len__(self):
        return self.n

    def take(self):
        """What's held, as one array, and let it go; the count keeps running (ragged offsets stay absolute)."""
        out = np.asarray(self)
        self.chunks = []
        return out

    def __array__(self, dtype=None, copy=None):
        self._flush()
        if not self.chunks:
            out = np.zeros(0, self.dtype)
        elif len(self.chunks) == 1:
            out = self.chunks[0]
        else:
            out = np.concatenate(self.chunks)
            self.chunks = [out]  # one copy from here on, not the pieces and the whole
        return out if dtype is None or np.dtype(dtype) == out.dtype else out.astype(dtype)


RAGGED = dict(prod=("prod_abil", "prod_progress"), buff=("b_id",), **{"pass": ("pass_id", "pass_type", "pass_hp")},
              ord=("ord_abil", "ord_tx", "ord_ty", "ord_target", "ord_progress"), rally=("rally_x", "rally_y", "rally_target"),
              ab=("ab_id", "ab_point"))
NEUTRAL = 16  # the API's owner of minerals, gas, rocks and critters
ARROW_FRAMES = 64  # frames per record batch: a window reads and decompresses only its batches


class ArrowView:
    """Streams a view to <view>.units.arrow (one row per unit, ragged lists as list columns) and <view>.frames.arrow (one
    row per frame: loop, grids, the player block), a record batch per ARROW_FRAMES frames, zstd per buffer. Unit columns are
    deltas from the batch's first frame on (worldmodel/arrow_state.py)."""

    def __init__(self, out_path):
        import pyarrow as pa
        self.pa, self.base = pa, out_path[:-len(".cst")]
        self.options = pa.ipc.IpcWriteOptions(compression=pa.Codec("zstd", compression_level=3))
        self.sinks, self.writers = {}, {}

    def _write(self, name, batch, metadata=None):
        if name not in self.writers:
            self.sinks[name] = self.pa.OSFile(f"{self.base}.{name}.arrow.tmp", "wb")
            self.writers[name] = self.pa.ipc.new_file(self.sinks[name], batch.schema, options=self.options)
        self.writers[name].write_batch(batch, custom_metadata=metadata)

    def write_units(self, first_frame, counts, chunks, rag):
        pa = self.pa
        cols = {k: chunks[k].take() for k in U_DTYPES}
        from cosmicstar.worldmodel import arrow_state
        stored, how = arrow_state.encode_units({f"u_{k}": cols[k] for k in U_DTYPES if not k.endswith("_start")}, counts)
        names, arrs = list(stored), [pa.array(v) for v in stored.values()]
        names.append("frame")
        arrs.append(pa.array(np.repeat(np.arange(first_frame, first_frame + len(counts), dtype=np.int32), counts)))
        for g, members in RAGGED.items():
            st = cols[f"{g}_start"]
            total = len(rag[members[0]])
            off = np.append(st, total).astype(np.int64)
            off -= off[0]
            for c in members:
                names.append(c)
                arrs.append(pa.LargeListArray.from_arrays(pa.array(off), pa.array(rag[c].take())))
        self._write("units", pa.record_batch(arrs, names=names), {"how": json.dumps(how)})

    def write_frames(self, columns, grids):
        pa = self.pa
        names, arrs = list(columns), [pa.array(v) for v in columns.values()]
        for k, frames in grids.items():
            names.append(k)
            arrs.append(pa.FixedSizeBinaryArray.from_buffers(pa.binary(frames[0].size), len(frames),
                                                             [None, pa.py_buffer(b"".join(f.tobytes() for f in frames))]))
        self._write("frames", pa.record_batch(arrs, names=names))

    def close(self):
        for k, w in self.writers.items():
            w.close()
            self.sinks[k].close()
            os.replace(f"{self.base}.{k}.arrow.tmp", f"{self.base}.{k}.arrow")

    def read_columns(self, *names):
        """Whole columns of the units file, for the checks after the playback."""
        r = self.pa.ipc.open_file(self.pa.memory_map(f"{self.base}.units.arrow"))
        from cosmicstar.worldmodel import arrow_state
        got = [arrow_state.read_units(r, i)[0] for i in range(r.num_record_batches)]
        return {k: np.concatenate([g[k] for g in got]) for k in names}


def extract(path, out_path, step, until, perspective=0, substep=0, ui=False, abilities=True, acts_path=None, arrow=False):
    """Plays one replay and saves every frame (see the module doc). perspective 0: observer,
    fog off; a player slot: as that player sees it (fog on); ui: also the engine's control group
    and selection panel summaries (ui_*), which need the feature layer on; abilities: query the
    abilities each unit can use right now at every frame (ab_*: the only cooldown signal)."""
    from pysc2.lib import protocol
    from s2clientprotocol import query_pb2, sc2api_pb2 as sc_pb

    info = replay_info(path)
    info["layout"] = LAYOUT
    mp = map_path(info["map_hash"], info["handles"])
    if mp is None:
        raise FileNotFoundError("map not in the Battle.net cache: " + info["map"])
    until = info["duration_loops"] if until <= 0 else min(until, info["duration_loops"] or until)
    data = open(path, "rb").read()
    stats, ups = tracker_tables(path, until)
    desync = DesyncCheck(*tracker_lifecycle(path, until + 10 * LPS))
    run_config, current = run_config_for(info)
    info["perspective"] = perspective

    ids = {}  # unit tag -> per-replay id

    def uid(tag):
        if tag not in ids:
            ids[tag] = len(ids)
        return ids[tag]

    # Player views: the engine's executed commands with their recipients (every observation,
    # the per-loop ones included) and its per-frame control group / selection panel summary.
    cmd = {k: [] for k in CMD_DTYPES}
    cmd_tags = []  # recipients' engine tags, flat; mapped to ids at the end
    cmd["start"].append(0)
    ui_count, ui_leader, ui_panel, ui_sel, ui_sel_start = [], [], [], [], [0]
    sel_loop, sel_start, sel_ids, last_sel = [], [0], [], None  # is_selected snapshots, at every loop the set changed

    def take_selection(units, loop):
        nonlocal last_sel
        cur = [u.tag for u in units if u.is_selected]
        if cur != last_sel:
            last_sel = cur
            sel_loop.append(loop)
            sel_ids.extend(uid(t) for t in cur)
            sel_start.append(len(sel_ids))

    def take_actions(actions, frame):
        for a in actions:
            if not a.HasField("action_raw") or not a.action_raw.HasField("unit_command"):
                continue
            uc = a.action_raw.unit_command
            cmd["loop"].append(a.game_loop)
            cmd["frame"].append(frame)
            cmd["abil"].append(uc.ability_id)
            has_pos = uc.HasField("target_world_space_pos")
            cmd["target"].append(1 if has_pos else 2 if uc.HasField("target_unit_tag") else 0)
            cmd["tx"].append(uc.target_world_space_pos.x if has_pos else np.nan)
            cmd["ty"].append(uc.target_world_space_pos.y if has_pos else np.nan)
            cmd["target_id"].append(uid(uc.target_unit_tag) if uc.HasField("target_unit_tag") else -1)
            cmd["queued"].append(uc.queue_command)
            cmd_tags.extend(uc.unit_tags)
            cmd["start"].append(len(cmd_tags))

    cols = {k: [] for k in U_DTYPES}      # the current frame's unit rows
    chunks = {k: TypedList(U_DTYPES[k]) for k in U_DTYPES}    # ... packed into typed arrays (Python lists for a whole game ran to GBs)
    n_rows = 0
    rag = {k: TypedList(R_DTYPES[k]) for k in R_DTYPES}  # ragged per-unit lists (production, buffs, passengers, orders)
    eff = {k: TypedList(E_DTYPES[k]) for k in E_DTYPES}  # per-frame effects, one row per effect position
    dead = TypedList(np.int32)
    frame_loop, frame_start, eff_start, dead_start = [], [0], [0], [0]
    vis, creep, pathable, buildable = [], [], [], []  # player views only
    # layout 2: the observed player's block per frame (player_common, score details, camera; the
    # observer's are the engine's placeholders), and the raw player block's ragged tables
    pc = collections.defaultdict(lambda: TypedList(np.float32))   # pc_<field>[F]
    sc = collections.defaultdict(lambda: TypedList(np.float32))   # sc_<field>[_<category|vital>][F]
    cam = []                             # cam_x, cam_y [F]
    pw = {k: [] for k in PW_DTYPES}      # power sources
    rd = {k: [] for k in RD_DTYPES}      # radar rings
    pu = []                              # the raw player block's upgrade ids (nothing else in extract() may bind `pu`)
    pw_start, rd_start, pu_start = [0], [0], [0]
    sink = ArrowView(out_path) if arrow else None
    flushed = [0, 0]                     # frames and rows already streamed to Arrow
    ab_failed = [0]                      # available-abilities queries refused (switched off after the first)

    def flush_arrow():
        f0, f1 = flushed[0], len(frame_loop)
        if f1 == f0:
            return
        sink.write_units(f0, np.diff(np.asarray(frame_start[f0:f1 + 1], np.int64)), chunks, rag)
        columns = dict(loop=np.asarray(frame_loop[f0:f1], np.int32))
        columns.update({f"pc_{k}": v.take() for k, v in pc.items()}, **{f"sc_{k}": v.take() for k, v in sc.items()})
        xy = np.asarray(cam, np.float32).reshape(-1, 2)
        columns.update(cam_x=xy[:, 0], cam_y=xy[:, 1])
        cam.clear()
        sink.write_frames(columns, dict(vis=vis, creep=creep, pathable=pathable, buildable=buildable) if vis else {})
        vis.clear()
        creep.clear()
        pathable.clear()
        buildable.clear()
        flushed[0] = f1

    def take_player(obs, raw):
        # every declared field (unset ones read as 0), so the per-frame columns stay aligned
        for fd in obs.player_common.DESCRIPTOR.fields:
            if fd.name != "player_id":
                pc[fd.name].append(getattr(obs.player_common, fd.name))
        det = obs.score.score_details
        for fd in det.DESCRIPTOR.fields:
            v = getattr(det, fd.name)
            if fd.type == fd.TYPE_MESSAGE:  # per category (army, economy, ...) or per vital (life, shields, energy)
                for sub in fd.message_type.fields:
                    sc[f"{fd.name}_{sub.name}"].append(getattr(v, sub.name))
            else:
                sc[fd.name].append(v)
        cam.append((raw.player.camera.x, raw.player.camera.y))
        for ps in raw.player.power_sources:
            pw["x"].append(ps.pos.x); pw["y"].append(ps.pos.y); pw["radius"].append(ps.radius)
            pw["id"].append(uid(ps.tag) if ps.tag else -1)
        for rr in raw.radar:
            rd["x"].append(rr.pos.x); rd["y"].append(rr.pos.y); rd["radius"].append(rr.radius)
        pu.extend(raw.player.upgrade_ids)
        pw_start.append(len(pw["x"])); rd_start.append(len(rd["x"])); pu_start.append(len(pu))

    def available_abilities(units):
        """unit tag -> the abilities it can use right now (resource costs ignored): a used Blink,
        Stim or bile is missing until its cooldown ends. One query per frame for the observed
        player's visible units; the engine answers nothing for units the observer doesn't control
        (tested: 0 rows for all three owners), so the observer view skips the query."""
        if not abilities or ab_failed[0] or perspective == 0:
            return {}
        tags = [u.tag for u in units if u.display_type == 1 and u.alliance == 1]
        if not tags:
            return {}
        try:
            resp = c.query(query_pb2.RequestQuery(abilities=[query_pb2.RequestQueryAvailableAbilities(unit_tag=t) for t in tags],
                                                  ignore_resource_requirements=True))
        except (protocol.ProtocolError, ValueError) as ex:
            ab_failed[0] = 1
            print(f"available-abilities query refused, ab_* left empty: {str(ex)[:200]}", file=sys.stderr, flush=True)
            return {}
        return {r.unit_tag: r.abilities for r in resp.abilities}

    if perspective == 0:
        substep = 0  # no fog in the observer view: the per-loop observation requests are pointless there
    # pysc2's default controller timeout is 6 hours: a hung engine (seen at startup when the
    # prefix's wineserver was rebooting) would park the worker for the day. ENGINE_TIMEOUT makes
    # it a "timed out" error, which run_worker retries.
    # One launch at a time, through the replay start: eight Proton launches at once each run
    # Proton's prefix setup, which restarts the prefix's wineserver under the others (the losers
    # hang at the umu.exe stage), and the map load below must not overlap either. Once running,
    # engines are concurrent; after the first wave they rarely collide.
    lock = launch_lock()
    lock.__enter__()
    try:
        proc = run_config.start(want_rgb=False, timeout_seconds=ENGINE_TIMEOUT)
    except BaseException:
        lock.__exit__(None, None, None)
        raise
    with proc as c:
        try:
            opts = sc_pb.InterfaceOptions(raw=True, score=True, show_cloaked=True, show_burrowed_shadows=True,
                                          show_placeholders=False, raw_affects_selection=False)
            if perspective:
                # the minimap at the map's own size: pathable and buildable per frame, as this player knows them. The
                # smallest screen, since no screen layer is kept. ui_data (control groups, selection panel) needs the layer on too.
                opts.feature_layer.width = 24
                opts.feature_layer.resolution.x = opts.feature_layer.resolution.y = 16
                opts.feature_layer.minimap_resolution.x, opts.feature_layer.minimap_resolution.y = info["map_cells"]
            req = sc_pb.RequestStartReplay(
                # observer: fog off, everything; a player: that player's view, fog on (module doc)
                replay_data=data, observed_player_id=perspective, disable_fog=perspective == 0, options=opts)
            if not current:
                # the map passed in, not looked up by handle: by handle the older engine sometimes
                # tries to validate / download the dependency through Battle.net, fails without a
                # session and the replay is over the moment it starts (7 of 10 games in one batch,
                # 2026-10-30, the same map both working and failing). The launch lock is held through
                # this because with map_data SC2 writes a shared temp map file that concurrent
                # starts can pick up from each other (InvalidReplayData or a desync).
                req.map_data = open(mp, "rb").read()
            c.start_replay(req)
            c._client._sock.settimeout(STEP_TIMEOUT)  # the replay is running: a request that takes longer is a hung engine
            data = c.data_raw()  # this game's own rules decide what counts as production
            if c.status == protocol.Status.ended:
                # the game is over the moment it starts, intermittently per start (the same game
                # passes on another try; 2026-10-30): the start response still says in_replay,
                # the ended status arrives with the next responses. Retried by run_worker.
                raise RuntimeError("replay ended at start")
            gi = c.game_info()
            # The engine keeps reading its launch map from the shared temp folder (TempLaunchMap~N,
            # TempKeyFrameP*) after start_replay returns; a start queued right behind it collided
            # and was refused with InvalidReplayData (bursts, 2026-10-30). The lock is held through
            # the first requests and a settle time so the next start begins on a quiet folder.
            if not os.environ.get("WINEPREFIX"):  # shared base prefix only; pool prefixes share no temp files
                time.sleep(LAUNCH_SETTLE)
        finally:
            lock.__exit__(None, None, None)
        minimize_client_window(proc.pid)
        producers = {u.unit_id for u in data.units  # structures, eggs, cocoons, larva
                     if STRUCTURE in u.attributes or any(k in u.name for k in ("Egg", "Cocoon", "Larva"))}
        production = {a.ability_id for a in data.abilities if PRODUCTION.search(a.link_name)}
        sr = gi.start_raw
        info["playable"] = [sr.playable_area.p0.x, sr.playable_area.p0.y, sr.playable_area.p1.x, sr.playable_area.p1.y]
        info["map_size"] = [sr.map_size.x, sr.map_size.y]
        if perspective and info["map_cells"] != info["map_size"]:
            raise RuntimeError(f"the replay says the map is {info['map_cells']}, the engine {info['map_size']}")
        info["start_locations"] = [[pt.x, pt.y] for pt in sr.start_locations]  # the possible enemy bases
        grids = {}
        for name in ("pathing_grid", "placement_grid", "terrain_height"):
            img = getattr(sr, name)
            grids[f"map_{name}"] = np.frombuffer(img.data, np.uint8)
            info[f"{name}_format"] = dict(bits=img.bits_per_pixel, size=[img.size.x, img.size.y])
        teams = {}
        for p in info["players"]:
            teams.setdefault(p["team"], set()).add(p["slot"])
        done_players = set()
        loop, stalled = 0, 0
        while loop < until:
            try:
                if substep and substep < step:
                    # The client keeps a unit that left the player's vision in the observation for 47
                    # observation *requests* (measured: 2.1 s at 1 loop per request, 8.4 s at 4, longer
                    # at 8), so stepping several loops per observation leaks enemy movement through the
                    # fog. An observation is requested every substep loops and, except for the commands
                    # and deaths it reports, discarded; only every --step loops is one parsed in full and kept.
                    for _ in range(step // substep - 1):
                        c.step(substep)
                        # pysc2's observe, not a raw websocket write / recv: the raw path hung mid-game
                        # (worker blocked in recv, engine idle; reproduced on 93333 within a minute,
                        # 2026-10-30). The response is parsed either way (~0.1 ms with upb).
                        ob = c.observe()
                        if perspective:  # the commands and deaths this observation reports (both are
                            take_actions(ob.actions, len(frame_loop))  # "since the last one"), and the selection
                            dead.extend(ids[t] for t in ob.observation.raw_data.event.dead_units if t in ids)
                            take_selection(ob.observation.raw_data.units, ob.observation.game_loop)
                    c.step(substep)
                else:
                    c.step(step)
                o = c.observe()
            except protocol.ProtocolError as ex:
                # The replay is over (a player's view ends when that player leaves the game; the
                # engine's status turns ended and the next step is refused). pysc2 words it
                # "`step` called while in state: Status.ended", older texts said "already ended".
                if "already ended" in str(ex) or "Status.ended" in str(ex):
                    if frame_loop:  # frames were extracted: the game ended, keep them
                        break
                    raise RuntimeError("replay ended at start")  # nothing extracted: retried by run_worker
                raise
            prev, loop = loop, o.observation.game_loop
            stalled = stalled + 1 if loop <= prev else 0
            if stalled >= 5:
                if loop >= until - 4 * step:  # the replay simply ended
                    break
                raise RuntimeError(f"replay stalled at {loop / LPS:.0f}s")
            raw = o.observation.raw_data
            avail = available_abilities(raw.units)
            take_player(o.observation, raw)
            for u in raw.units:
                if arrow and not perspective and u.owner != NEUTRAL:  # the observer pass is only for the neutrals; the players' views have the rest
                    continue
                cols["id"].append(uid(u.tag))
                cols["type"].append(u.unit_type)
                cols["owner"].append(u.owner)
                cols["x"].append(u.pos.x)
                cols["y"].append(u.pos.y)
                cols["z"].append(u.pos.z)
                cols["facing"].append(u.facing)
                cols["radius"].append(u.radius)
                cols["hp"].append(u.health)
                cols["hp_max"].append(u.health_max)
                cols["shield"].append(u.shield)
                cols["shield_max"].append(u.shield_max)
                cols["energy"].append(u.energy)
                cols["build"].append(u.build_progress)
                cols["flags"].append(F_FLYING * u.is_flying | F_BURROWED * u.is_burrowed | F_CLOAKED * (u.cloak in (1, 2, 4))
                                     | F_HALLUC * u.is_hallucination | F_SNAPSHOT * (u.display_type == 2)
                                     | F_POWERED * u.is_powered | F_SELECTED * u.is_selected)
                cols["contents"].append(u.mineral_contents or u.vespene_contents)
                cols["harvesters"].append(u.assigned_harvesters)
                cols["add_on"].append(uid(u.add_on_tag) if u.add_on_tag else -1)
                cols["weapon_cooldown"].append(u.weapon_cooldown)
                cols["engaged"].append(uid(u.engaged_target_tag) if u.engaged_target_tag else -1)
                cols["cargo_used"].append(u.cargo_space_taken)
                cols["display"].append(u.display_type)
                cols["cloak"].append(u.cloak)
                cols["flags2"].append(F2_ACTIVE * u.is_active | F2_BLIP * u.is_blip | F2_ONSCREEN * u.is_on_screen)
                cols["atk_up"].append(u.attack_upgrade_level)
                cols["arm_up"].append(u.armor_upgrade_level)
                cols["shd_up"].append(u.shield_upgrade_level)
                cols["buff_remain"].append(min(u.buff_duration_remain, 32767))
                cols["buff_max"].append(min(u.buff_duration_max, 32767))
                cols["ideal_harvesters"].append(u.ideal_harvesters)
                cols["cargo_max"].append(u.cargo_space_max)
                cols["detect_range"].append(u.detect_range)
                cols["radar_range"].append(u.radar_range)
                # Rally points (structures): where produced units go; town halls have two (units, workers)
                cols["rally_start"].append(len(rag["rally_x"]))
                for r in u.rally_targets:
                    rag["rally_x"].append(r.point.x if r.HasField("point") else np.nan)
                    rag["rally_y"].append(r.point.y if r.HasField("point") else np.nan)
                    rag["rally_target"].append(uid(r.tag) if r.tag else -1)
                # Abilities usable right now (cooldowns): see available_abilities
                cols["ab_start"].append(len(rag["ab_id"]))
                for ab in avail.get(u.tag, ()):
                    rag["ab_id"].append(ab.ability_id)
                    rag["ab_point"].append(ab.requires_point)
                # Production queue (producers only); other units' orders are intent, not state.
                cols["prod_start"].append(len(rag["prod_abil"]))
                if u.unit_type in producers:
                    for od in u.orders:
                        if od.ability_id in production:
                            rag["prod_abil"].append(od.ability_id)
                            rag["prod_progress"].append(od.progress)
                cols["buff_start"].append(len(rag["b_id"]))
                rag["b_id"].extend(u.buff_ids)
                cols["pass_start"].append(len(rag["pass_id"]))
                for pas in u.passengers:
                    rag["pass_id"].append(uid(pas.tag))
                    rag["pass_type"].append(pas.unit_type)
                    rag["pass_hp"].append(pas.health)
                # Orders in queue order (production included): what the unit was told to do.
                cols["ord_start"].append(len(rag["ord_abil"]))
                for od in u.orders:
                    rag["ord_abil"].append(od.ability_id)
                    has_pos = od.HasField("target_world_space_pos")
                    rag["ord_tx"].append(od.target_world_space_pos.x if has_pos else np.nan)
                    rag["ord_ty"].append(od.target_world_space_pos.y if has_pos else np.nan)
                    rag["ord_target"].append(uid(od.target_unit_tag) if od.target_unit_tag else -1)
                    rag["ord_progress"].append(od.progress)
            for e in raw.effects:
                for pt in e.pos:
                    eff["type"].append(e.effect_id)
                    eff["owner"].append(e.owner)
                    eff["x"].append(pt.x)
                    eff["y"].append(pt.y)
                    eff["radius"].append(e.radius)
            # Only units that have been seen: the engine also reports deaths of short-lived
            # internal units (spawned and destroyed between frames, absent from the
            # replay's own unit log), which aren't game units.
            dead.extend(ids[t] for t in raw.event.dead_units if t in ids)
            frame_loop.append(loop)
            n_rows += len(cols["id"])
            for k, v in cols.items():
                chunks[k].extend(v)
                v.clear()
            frame_start.append(n_rows)
            eff_start.append(len(eff["type"]))
            dead_start.append(len(dead))
            if perspective:
                vis.append(np.frombuffer(raw.map_state.visibility.data, np.uint8).copy())
                creep.append(np.frombuffer(raw.map_state.creep.data, np.uint8).copy())
                mm = o.observation.feature_layer_data.minimap_renders
                pathable.append(np.frombuffer(mm.pathable.data, np.uint8).reshape(sr.map_size.y, -1)[::-1].copy())    # the minimap's rows run the other way
                buildable.append(np.frombuffer(mm.buildable.data, np.uint8).reshape(sr.map_size.y, -1)[::-1].copy())
                take_actions(o.actions, len(frame_loop) - 1)
                take_selection(raw.units, loop)
            if arrow and len(frame_loop) - flushed[0] >= ARROW_FRAMES:
                flush_arrow()
            if perspective and ui:
                ud = o.observation.ui_data
                cnt, lead = np.zeros(10, np.int16), np.zeros(10, np.int32)
                for g in ud.groups:
                    cnt[g.control_group_index], lead[g.control_group_index] = g.count, g.leader_unit_type
                ui_count.append(cnt)
                ui_leader.append(lead)
                # the selection panel: whichever panel the game shows for the selection
                if ud.HasField("single"):
                    panel, types = 1, [ud.single.unit.unit_type]
                elif ud.HasField("multi"):
                    panel, types = 2, [u.unit_type for u in ud.multi.units]
                elif ud.HasField("cargo"):
                    panel, types = 3, [ud.cargo.unit.unit_type]
                elif ud.HasField("production"):
                    panel, types = 4, [ud.production.unit.unit_type]
                else:
                    panel, types = 0, []
                ui_panel.append(panel)
                ui_sel.extend(types)
                ui_sel_start.append(len(ui_sel))
            # a player still sees snapshots of enemy structures that died in the fog, so in a
            # player view only the player's own units are checked against the replay's log; the observer
            # keeps showing neutrals the replay has recorded as dead (mined out, destroyed), so not those
            desync.check(loop, [u.tag for u in raw.units if u.owner == perspective or (not perspective and u.owner != NEUTRAL)])
            # The game is decided once every player of some team has a result (left or lost);
            # one player leaving a team game doesn't end it (their ally plays on).
            done_players |= {r.player_id for r in o.player_result}
            if any(members <= done_players for members in teams.values()):
                info["end"] = dict(loop=loop, results={str(r.player_id): r.result for r in o.player_result})
                break

    if arrow:
        vis_format = dict(bits=raw.map_state.visibility.bits_per_pixel, size=[raw.map_state.visibility.size.x, raw.map_state.visibility.size.y]) if perspective else None
        creep_format = dict(bits=raw.map_state.creep.bits_per_pixel, size=[raw.map_state.creep.size.x, raw.map_state.creep.size.y]) if perspective else None
        flush_arrow()
        sink.close()
        arrays = {}
    else:
        arrays = {f"u_{k}": np.asarray(chunks[k], dtype=U_DTYPES[k]) for k in U_DTYPES}
        arrays.update({k: np.asarray(v, dtype=R_DTYPES[k]) for k, v in rag.items()})
    arrays.update({f"e_{k}": np.asarray(v, dtype=E_DTYPES[k]) for k, v in eff.items()})
    arrays.update(frame_loop=np.asarray(frame_loop, np.int32), frame_start=np.asarray(frame_start, np.int64),
                  e_frame_start=np.asarray(eff_start, np.int64), dead_frame_start=np.asarray(dead_start, np.int64),
                  dead_id=np.asarray(dead, np.int32), **grids)
    arrays.update({f"p_{k}": np.asarray(v, np.float32 if k not in ("loop", "player") else np.int32) for k, v in stats.items()})
    if not arrow:
        arrays.update({f"pc_{k}": np.asarray(v, np.float32) for k, v in pc.items()})
        arrays.update({f"sc_{k}": np.asarray(v, np.float32) for k, v in sc.items()})
    arrays.update({f"pw_{k}": np.asarray(v, dtype=PW_DTYPES[k]) for k, v in pw.items()}, pw_frame_start=np.asarray(pw_start, np.int64))
    arrays.update({f"rd_{k}": np.asarray(v, dtype=RD_DTYPES[k]) for k, v in rd.items()}, rd_frame_start=np.asarray(rd_start, np.int64))
    arrays.update(pu_id=np.asarray(pu, np.int32), pu_frame_start=np.asarray(pu_start, np.int64))
    if not arrow:
        arrays.update(cam=np.asarray(cam, np.float32).reshape(-1, 2))
    info["abilities_queried"] = bool(abilities) and not ab_failed[0] and perspective != 0
    arrays.update(up_loop=np.asarray(ups["loop"], np.int32), up_player=np.asarray(ups["player"], np.int8),
                  up_name=np.asarray(ups["name"]))
    if arrow and perspective:
        info.update(vis_format=vis_format, creep_format=creep_format)
    if perspective and vis and not arrow:
        ms = raw.map_state
        h, w = ms.visibility.size.y, ms.visibility.size.x
        arrays["vis"] = np.stack(vis).reshape(len(vis), h, -1)
        arrays["creep"] = np.stack(creep).reshape(len(creep), h, -1)
        vis, creep = [], []
        info["vis_format"] = dict(bits=ms.visibility.bits_per_pixel, size=[w, h])
        info["creep_format"] = dict(bits=ms.creep.bits_per_pixel, size=[ms.creep.size.x, ms.creep.size.y])
    stats = dict(frames=len(frame_loop), rows=n_rows)
    if perspective:
        cmd_ids = np.asarray([uid(t) for t in cmd_tags], np.int32)
        arrays.update({f"cmd_{k}": np.asarray(v, dtype=CMD_DTYPES[k]) for k, v in cmd.items()}, cmd_unit=cmd_ids,
                      sel_loop=np.asarray(sel_loop, np.int32), sel_start=np.asarray(sel_start, np.int64),
                      sel_id=np.asarray(sel_ids, np.int32))
        if ui:
            arrays.update(ui_grp_count=np.asarray(ui_count, np.int16).reshape(-1, 10),
                          ui_grp_leader=np.asarray(ui_leader, np.int32).reshape(-1, 10),
                          ui_sel_panel=np.asarray(ui_panel, np.int8), ui_sel_type=np.asarray(ui_sel, np.int32),
                          ui_sel_start=np.asarray(ui_sel_start, np.int64))
    tags = np.zeros(len(ids), np.uint64)
    for tag, i in ids.items():
        tags[i] = tag
    arrays.update(unit_tag=tags, meta=np.asarray(json.dumps(info)))
    # Control groups and selection per frame from the replay's own events (every player in the
    # observer view, the observed one in a player view), the latter checked against the engine's
    # summaries, selection and recipients (worldmodel/selection_tracker.py)
    from cosmicstar.worldmodel import replay_actions, selection_tracker
    acts = replay_actions.parse(path)
    if acts_path and not os.path.exists(acts_path):  # the game's action table, once, next to its views
        np.savez_compressed(acts_path + ".tmp.npz", **acts)
        os.replace(acts_path + ".tmp.npz", acts_path)
    died = selection_tracker.deaths_by_replay_tag(desync.died)  # exact death loops, from the replay
    slots = [perspective] if perspective else [p["slot"] for p in info["players"]]
    arrays.update(selection_tracker.frame_groups(acts, slots, arrays["frame_loop"], tags, died))
    arrays.update(selection_tracker.frame_memory(acts, slots, arrays["frame_loop"], tags, died))
    if perspective:
        checks = selection_tracker.validate({**arrays, **sink.read_columns("u_id", "u_type")} if arrow else arrays, acts, perspective, died)
        stats["groups"] = {k: int(v) for k, v in checks.items() if k != "mismatches"}
    stats["units"] = len(ids)
    from cosmicstar.worldmodel import statefile
    statefile.save(out_path, arrays)  # .cst: per-unit deltas + zstd (worldmodel/statefile.py), written atomically
    return stats


# ---------------------------------------------------------------- job list and workers

def installed_builds():
    """Base builds we have an engine for, in the install or the side install with the older
    builds (wine_run_config.installed_builds; the run config picks the install per build)."""
    from cosmicstar.envs import wine_run_config
    return wine_run_config.installed_builds()


def build_manifest(replays_root, games_root, fmt, step, substep, builds=(), maps=(), ids=(), any_ruleset=False,
                   redo=False, retry_failed=False, observer=False):
    """One job per (game, player view) still to extract, from the replay tree's index
    (corpus.py): the format's games on installed base builds, long enough, with their map
    cached (fetched from the depot unless --no_fetch) and a ruleset the Editor could export
    (any_ruleset: those too; their abilities can't be named, which only matters for the action
    vocabulary). A view already extracted with this layout, step and substep is done (redo: do it
    again); a view with a fail file is left alone (retry_failed: try again). Returns (jobs,
    {reason: count of skipped games or views})."""
    have = installed_builds()
    exported = os.path.join(BALANCE_DIR, "status.json")
    exported = json.load(open(exported)) if os.path.exists(exported) and not any_ruleset else None
    index = corpus.load_index(replays_root)
    maps = corpus.map_dirs_named(index, maps) if maps else set()  # map directories (hashes); a name takes every version
    jobs, skipped = [], collections.Counter()
    for gid, d in sorted(index.items()):
        if (fmt and d["format"] != fmt) or (builds and d["base_build"] not in builds) \
                or (maps and corpus.map_dir(d["map_hash"]) not in maps) or (ids and gid not in ids):
            continue
        path = os.path.join(replays_root, d["path"])
        why = None
        if not os.path.exists(path):
            why = "replay file missing"
        elif d["build"] in EXCLUDED_BUILDS:
            why = "patch 5.0.16"
        elif d["base_build"] not in have:
            why = f"base build {d['base_build']} not installed"
        elif d["duration_loops"] < 90 * LPS:
            why = "under 90 s (remake/leave)"
        elif map_path(d["map_hash"], d.get("handles")) is None:
            why = "map not cached (depot fetch failed)" if FETCH_MISSING else "map not cached"
        elif exported is not None and not exported.get(d["ruleset"], {}).get("ok"):
            why = "ruleset not exported" if d["ruleset"] not in exported else "ruleset not exportable"
        if why:
            skipped[why] += 1
            continue
        files = corpus.GameFiles(games_root, corpus.rel_dir(d["format"], d["base_build"], d["map_hash"]), gid)
        meta = corpus.read_meta(files.meta()).get("views", {})
        for slot in ([0] if observer else []) + [p["slot"] for p in d["players"]]:
            if slot in CRASHING_VIEWS:
                continue
            out, rec = files.view(slot), meta.get(str(slot), {})
            if os.path.exists(out) and rec.get("layout") == LAYOUT and rec.get("step") == step and rec.get("substep") == substep and not redo:
                skipped["done"] += 1
                continue
            if os.path.exists(files.fail(slot)) and not retry_failed:
                skipped["failed earlier (--retry_failed)"] += 1
                continue
            jobs.append(dict(id=gid, path=path, out=out, acts=files.acts(), meta=files.meta(), fail=files.fail(slot),
                             map=d["map"], base_build=d["base_build"], ruleset=d["ruleset"], perspective=slot,
                             slots=[p["slot"] for p in d["players"]], teams={p["slot"]: p["team"] for p in d["players"]},
                             redo=os.path.exists(out)))  # redo: the output exists (stale layout / step, or --redo)
    return jobs, dict(skipped)


# The engine's view of slot 4 crashes at the game's first research order (worldmodel/README.md);
# that view is skipped (the 2v2 composition from the teammate's view was dropped with the 1v1 focus).
CRASHING_VIEWS = {4}


def claim(job):
    """Atomically claims a job for this worker (any free worker takes the next one)."""
    try:
        os.makedirs(os.path.dirname(job["out"]), exist_ok=True)
        fd = os.open(job["out"] + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())  # who holds it, for the memory governor
        os.close(fd)
        return True
    except FileExistsError:
        return False


def available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return float("inf")


def kill_prefix_processes(prefix):
    """SIGKILL everything running in a Wine prefix (the engine and Proton's helpers), found by their environment."""
    root = prefix.rstrip("/")
    root = root[:-len("/pfx")] if root.endswith("/pfx") else root
    want = f"STEAM_COMPAT_DATA_PATH={root}".encode()  # Proton points WINEPREFIX at <root>/pfx/ underneath
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/environ", "rb") as f:
                env = f.read()
            if want in env.split(b"\0"):
                os.kill(int(pid), 9)
        except OSError:
            pass


def newest_claim(job_list, pids):
    """(job, pid) of the most recently started game among those workers are still playing."""
    best = None
    for j in job_list:
        c = j["out"] + ".claim"
        try:
            pid = int(open(c).read() or 0)
            t = os.path.getmtime(c)
        except (OSError, ValueError):
            continue
        if pid in pids and not os.path.exists(j["out"]) and (best is None or t > best[0]):
            best = (t, j, pid)
    return best[1:] if best else None


def minimize_client_window(pid, tries=20):
    """Windows only: minimizes the client's window once it exists. The Windows client has no
    headless mode (-displayMode 0 only means windowed) and keeps rendering; minimized, the
    engine stepped ~20% faster in a side-by-side test (5.8 vs 7.1 ms per 4-loop frame)."""
    if sys.platform != "win32":
        return 0
    import ctypes
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def cb(hwnd, _):
        wpid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
        if wpid.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    for _ in range(tries):
        user32.EnumWindows(cb, 0)
        if found:
            break
        time.sleep(0.5)
    for h in found:
        user32.ShowWindow(h, 6)  # SW_MINIMIZE
    return len(found)


def protobuf_impl():
    """The protobuf implementation for the workers: upb (the C++ parser, ~150x faster per
    observation, which made parsing 44% of the frame time in pure Python) when the installed
    s2clientprotocol bindings are in the modern style written by scripts/regen_s2proto.py,
    else pure Python (the wheel's old-style modules can't load under upb). The environment
    variable PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION overrides. Output is identical either way
    (checked on a game: every array equal, NaN-aware)."""
    if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"):
        return os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]
    import s2clientprotocol
    bindings = os.path.join(os.path.dirname(s2clientprotocol.__file__), "sc2api_pb2.py")
    return "upb" if "AddSerializedFile" in open(bindings, encoding="utf-8").read() else "python"


def run_worker(jobs, step, until, substep=0, ui=False, abilities=True):
    from absl import flags
    flags.FLAGS(sys.argv[:1])
    commit = os.environ.get("COSMICSTAR_COMMIT", "")
    for j in jobs:
        # done earlier in this run (another worker) or before it: skip, unless the job says redo
        if (os.path.exists(j["out"]) and not j.get("redo")) or not claim(j):
            continue
        os.makedirs(os.path.dirname(j["out"]), exist_ok=True)
        t0 = time.time()
        slot = j.get("perspective", 0)
        for attempt in range(3):
            try:
                st = extract(j["path"], j["out"], step, until, slot, substep, ui, abilities, j.get("acts"), j.get("arrow", False))
                rec = dict(layout=LAYOUT, step=step, substep=substep, until=until, ui=ui, abilities=abilities, stats=st,
                           mb=round(os.path.getsize(j["out"]) / 1e6, 1), s=round(time.time() - t0), tries=attempt + 1,
                           extractor=commit, when=time.strftime("%Y-%m-%dT%H:%M:%S"))
                if j.get("meta"):
                    corpus.update_meta(j["meta"], slot, rec)
                if j.get("fail") and os.path.exists(j["fail"]):
                    os.remove(j["fail"])
                print(json.dumps(dict(ok=j.get("id", os.path.basename(j["path"])), view=slot, **st, mb=rec["mb"], s=rec["s"],
                                      tries=attempt + 1)), flush=True)
                break
            except Exception as ex:
                # replay starts fail in bursts while another instance fetches data for a map it
                # hasn't seen (InvalidReplayData); they pass once it's cached, so wait and retry.
                # The client also stalls past pysc2's 120 s websocket timeout now and then
                # (seen once in 12 games under Wine; the game passed alone), so retry those too,
                # and under Proton with several instances launching, the websocket of one is lost
                # or never comes up (ConnectError / ConnectionError / WebSocketConnectionClosed:
                # 6 of 12 player views with 4 workers, 2026-10-30); those passed alone as well.
                if any(k in repr(ex) for k in ("InvalidReplayData", "timed out", "Connect", "ended at start", "Status.ended")) and attempt < 2:
                    print(json.dumps(dict(retry=j.get("id", os.path.basename(j["path"])), view=slot, error=repr(ex)[:200],
                                          attempt=attempt + 1, s=round(time.time() - t0))), flush=True)
                    time.sleep(45 * (attempt + 1))
                    continue
                import traceback
                where = " | ".join(l.strip() for l in traceback.format_exc().splitlines() if l.strip().startswith("File"))[-300:]
                rec = dict(error=repr(ex)[:300], where=where, tries=attempt + 1, s=round(time.time() - t0), step=step,
                           substep=substep, layout=LAYOUT, extractor=commit, when=time.strftime("%Y-%m-%dT%H:%M:%S"))
                if j.get("fail"):
                    json.dump(rec, open(j["fail"], "w"), indent=1)
                print(json.dumps(dict(fail=j.get("id", os.path.basename(j["path"])), view=slot, error=rec["error"][:200],
                                      tries=attempt + 1, s=rec["s"], where=where)), flush=True)
                break
        # the claim stays until the run ends (main removes them): a view that finished or failed
        # must not be picked up again by the next worker walking the same job list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default=corpus.REPLAYS, help="the replay tree (scripts/ingest_replays.py)")
    ap.add_argument("--games", default=corpus.GAMES, help="the games tree the views go to")
    ap.add_argument("--format", default="1v1", help="games of this format (1v1, 2v2, ...; empty = all)")
    ap.add_argument("--builds", default="", help="comma-separated base builds (default: every installed one)")
    ap.add_argument("--maps", default="", help="comma-separated map names, slugs or hash prefixes (default: all)")
    ap.add_argument("--ids", default="", help="comma-separated game ids, or @file with a JSON list of ids")
    ap.add_argument("--redo", action="store_true", help="extract again views already done with this layout / step / substep")
    ap.add_argument("--retry_failed", action="store_true", help="try views with a fail file again")
    ap.add_argument("--step", type=int, default=16, help="game loops between frames (16 = 0.71 s)")
    ap.add_argument("--until", type=float, default=0, help="seconds of game time to extract (0 = whole game)")
    ap.add_argument("--substep", type=int, default=1, help="request an observation every this many loops (discarded "
                    "unparsed) and keep one every --step loops; without it a player view leaks enemy movement through "
                    "the fog (module doc)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N games of the job list")
    ap.add_argument("--no_fetch", action="store_true", help="don't fetch missing maps/mods from the depot")
    ap.add_argument("--slot", type=int, default=-1, help="(worker) its index; with --prefix_pool, worker k runs its engines in "
                    "Wine prefix k of wine_run_config.POOL_DIR, made on first use")
    ap.add_argument("--no_prefix_pool", action="store_true", help="all workers in the one base prefix (serialized launches, "
                    "keeper, shared temp slots) instead of a prefix per worker")
    ap.add_argument("--no_abilities", action="store_true", help="skip the per-frame available-abilities query (ab_*, the "
                    "cooldown signal)")
    ap.add_argument("--ui", action="store_true", help="also keep the engine's control group / selection panel summaries "
                    "(ui_*, for validating the tracked groups); turns the feature layer on")
    ap.add_argument("--any_ruleset", action="store_true", help="also replays whose ruleset has no Editor balance export "
                    "(their abilities can't be named)")
    ap.add_argument("--observer", action="store_true", help="also the observer view (<id>.p0.cst: fog off, every neutral; no orders, "
                    "production, passengers or effects, so it complements the players' views)")
    ap.add_argument("--arrow", action="store_true", help="stream each view's units and frames to <view>.units.arrow and "
                    "<view>.frames.arrow as it plays, the observer view's neutrals only; the rest (groups, effects, deaths, "
                    "...) stays in the .cst")
    ap.add_argument("--mem_low", type=float, default=4.0, help="GB available below which the most recently started game is "
                    "stopped and put back in the queue; the rest play on")
    ap.add_argument("--mem_high", type=float, default=9.0, help="GB available above which a stopped worker starts again")
    ap.add_argument("--worker", help=argparse.SUPPRESS)  # internal: job list for one worker process
    args = ap.parse_args()

    global FETCH_MISSING
    FETCH_MISSING = not args.no_fetch
    if args.worker and sys.platform != "win32" and not args.no_prefix_pool and args.slot >= 0:
        from cosmicstar.envs import wine_run_config as _wrc
        os.environ["WINEPREFIX"] = _wrc.ensure_pool_prefix(args.slot)
        _wrc.clear_shared_temp(os.environ["WINEPREFIX"], force=True)  # its own temp slots; nothing else runs in this prefix
    if args.worker:
        run_worker(json.load(open(args.worker)), args.step, int(args.until * LPS), args.substep, args.ui, not args.no_abilities)
        return
    logs = os.path.join(args.games, "logs")
    os.makedirs(logs, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    # resuming: claims left by an interrupted run (no output, no partial file) are free again
    for c in glob.glob(os.path.join(args.games, "**", "*.claim"), recursive=True):
        out_path = c[:-len(".claim")]
        if os.path.exists(out_path + ".tmp"):
            os.remove(out_path + ".tmp")
        os.remove(c)

    builds = {int(b) for b in args.builds.split(",") if b}
    maps = [m.strip() for m in args.maps.split(",") if m.strip()]
    ids = set(json.load(open(args.ids[1:]))) if args.ids.startswith("@") else {i for i in args.ids.split(",") if i}
    jobs, skipped = build_manifest(args.replays, args.games, args.format, args.step, args.substep, builds, maps, ids,
                                   args.any_ruleset, args.redo, args.retry_failed, args.observer)
    for j in jobs:
        j["arrow"] = args.arrow
    if args.limit:
        keep = sorted({j["id"] for j in jobs})[:args.limit]
        jobs = [j for j in jobs if j["id"] in keep]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    except OSError:
        commit = ""
    settings = {k: v for k, v in vars(args).items() if k not in ("worker", "slot")}
    json.dump(dict(run=run_id, extractor=commit, layout=LAYOUT, settings=settings, skipped=skipped, jobs=jobs),
              open(os.path.join(logs, run_id + ".json"), "w"), indent=1)
    print(f"run {run_id}: {len(jobs)} views of {len({j['id'] for j in jobs})} games to extract ({args.format or 'all formats'}; "
          f"layout {LAYOUT}, step {args.step}, substep {args.substep}); skipped: {skipped}", flush=True)
    env = dict(os.environ)
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = protobuf_impl()
    env["COSMICSTAR_COMMIT"] = commit
    print("protobuf parser:", env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"], flush=True)

    keeper = None
    if sys.platform != "win32" and args.no_prefix_pool:
        from cosmicstar.envs import wine_run_config as _wrc
        # the engine's shared temp folder (launch maps, key frames of every start): stale files
        # from killed engines pile up there; nothing runs in the prefix yet, so clear it
        _wrc.clear_shared_temp(log=lambda m: print(json.dumps(dict(note=m)), flush=True))
        keeper = _wrc.keep_prefix_alive()  # one wineserver for the whole run; returns once it is up
        print(json.dumps(dict(note="prefix keeper up", pid=keeper.pid)), flush=True)

    if sys.platform != "win32":
        from cosmicstar.envs import wine_run_config as _wrc
        _wrc.keep_engine_windows_minimized()

    worker_log = open(os.path.join(logs, run_id + ".log"), "a")  # the workers' stderr (pysc2 / engine noise, stack dumps)

    def run(job_list, n):
        """n worker processes sharing job_list (each claims the next free job). Workers
        share this process's stdout; piping them through here would block all but one."""
        path = os.path.join(logs, run_id + ".jobs.json")
        json.dump(job_list, open(path, "w"))
        def start(k):
            return subprocess.Popen([sys.executable, os.path.abspath(__file__), "--step", str(args.step), "--until", str(args.until),
                                     "--substep", str(args.substep), "--worker", path]
                                    + (["--no_fetch"] if args.no_fetch else []) + (["--ui"] if args.ui else [])
                                    + (["--no_abilities"] if args.no_abilities else [])
                                    + (["--no_prefix_pool"] if args.no_prefix_pool else []) + ["--slot", str(k)],
                                    env=env, stderr=worker_log)

        def stop_newest(procs, stopped):
            """The memory governor: the most recent game goes back in the queue; its worker waits for memory."""
            found = newest_claim(job_list, {p.pid: k for k, p in procs.items()})
            if found is None:
                return
            j_, pid = found
            k = next(k for k, p in procs.items() if p.pid == pid)
            procs.pop(k).kill()
            if not args.no_prefix_pool:
                from cosmicstar.envs import wine_run_config as _wrc
                kill_prefix_processes(_wrc.ensure_pool_prefix(k))
            for f in glob.glob(glob.escape(j_["out"][:-len(".cst")]) + "*.tmp") + [j_["out"] + ".claim", j_["out"] + ".tmp"]:
                if os.path.exists(f):
                    os.remove(f)
            stopped.append(k)
            print(json.dumps(dict(memory_stop=j_["id"], view=j_.get("perspective", 0), slot=k, available_gb=round(available_gb(), 1))), flush=True)

        procs = {k: start(k) for k in range(n)}
        stopped, last = [], 0.0
        while True:
            for k in [k for k, p in procs.items() if p.poll() is not None]:
                del procs[k]
            todo = any(not os.path.exists(j_["out"]) and not os.path.exists(j_["out"] + ".claim") for j_ in job_list)
            if not procs and not (stopped and todo):
                break
            mem, now = available_gb(), time.time()
            if mem < args.mem_low and len(procs) > 1 and now - last > 20:
                stop_newest(procs, stopped)
                last = now
            elif stopped and todo and now - last > 30 and (mem > args.mem_high or not procs):
                k = stopped.pop()
                procs[k] = start(k)
                last = now
                print(json.dumps(dict(memory_resume=k, available_gb=round(mem, 1))), flush=True)
            time.sleep(5)

    # One base build at a time. Older builds' game data is fetched on demand into the
    # shared data store; instances fetching at the same time (worse: for different
    # builds) fail to load replays (InvalidReplayData) or crash. So each build starts
    # with a single game alone, which pulls its data in, then runs in parallel.
    for build in sorted({j["base_build"] for j in jobs}):
        left = [j for j in jobs if j["base_build"] == build and (j.get("redo") or not os.path.exists(j["out"]))]
        if not left:
            continue
        print(json.dumps(dict(build=build, todo=len(left))), flush=True)
        for j in left[:3]:  # warm-up, alone, until one succeeds (a fresh output, not a redo job's old one)
            t_warm = time.time()
            run([j], 1)
            if os.path.exists(j["out"]) and os.path.getmtime(j["out"]) >= t_warm - 1:
                break
        run(left, min(args.workers, len(left)))
    for c in glob.glob(os.path.join(args.games, "**", "*.claim"), recursive=True):
        os.remove(c)  # the run is over; every claim has served
    if keeper is not None:
        keeper.terminate()


if __name__ == "__main__":
    main()
