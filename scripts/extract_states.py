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
compressed .npz per replay in --out:

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
    meta (json string)       map, playable area, players (race, team, result, MMR), build,
                             ruleset key, grid formats

Not here: creep (an observer's creep map is always empty; creep's sources - hatcheries,
tumors, overlords - are in the state) and commands (from the replay's own events).

Player views (--views players): the same replay played as each player (observed_player_id =
slot, fog on), exactly as the engine shows it to that player, written to --pov_out as
<name>_p<slot>.npz with the same layout plus:

    vis[F, H, W]             the player's visibility map per frame (0 never seen, 1 fogged,
                             2 visible; the engine's own map_state)
    creep[F, H, W]           creep as that player knows it
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
    python scripts/extract_states.py --replays <dir> --out data/worldmodel/obs --step 4 --workers 8
    python scripts/extract_states.py --views observer,players --pov_out data/worldmodel/pov ...
"""

import argparse
import glob
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
    prod_start=np.int64, buff_start=np.int64, pass_start=np.int64, ord_start=np.int64)
R_DTYPES = dict(  # ragged: unit row i owns [<x>_start[i], <x>_start[i+1]) (the last row: to the end)
    prod_abil=np.int16, prod_progress=np.float16, b_id=np.int16, pass_id=np.int32, pass_type=np.int16, pass_hp=np.float32,
    ord_abil=np.int16, ord_tx=np.float32, ord_ty=np.float32, ord_target=np.int32, ord_progress=np.float16)
E_DTYPES = dict(type=np.int16, owner=np.int8, x=np.float32, y=np.float32, radius=np.float16)
CMD_DTYPES = dict(  # player views: the engine's executed commands (recipients: cmd_unit, ragged by cmd_start)
    loop=np.int32, frame=np.int32, abil=np.int32, target=np.int8, tx=np.float32, ty=np.float32,
    target_id=np.int32, queued=np.bool_, start=np.int64)

# Unit flags packed into u_flags (F_SELECTED: the observed player's selection, player views only).
F_FLYING, F_BURROWED, F_CLOAKED, F_HALLUC, F_SNAPSHOT, F_POWERED, F_SELECTED = 1, 2, 4, 8, 16, 32, 64

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

def _protocol(archive):
    from s2protocol import versions
    header = versions.latest().decode_replay_header(archive.header["user_data_header"]["content"])
    build = header["m_version"]["m_build"]
    try:
        return versions.build(build), build
    except ImportError:
        return versions.latest(), build


def _s(b):
    return b.decode("utf-8", "replace") if isinstance(b, bytes) else b


def replay_info(path):
    """Cheap metadata from the replay file (no engine)."""
    import mpyq
    a = mpyq.MPQArchive(path)
    proto, build = _protocol(a)
    header = proto.decode_replay_header(a.header["user_data_header"]["content"])
    det = proto.decode_replay_details(a.read_file("replay.details"))
    meta = json.loads(a.read_file("replay.gamemetadata.json"))
    handles = det["m_cacheHandles"]
    h = handles[-1]  # the map; the others are mods (core data, balance mods, ...)
    mods = [x[8:].hex() for x in handles[:-1]]
    players = [dict(slot=i + 1, name=_s(p["m_name"]), race=_s(p["m_race"]), team=p["m_teamId"], result=p["m_result"],
                    mmr=None) for i, p in enumerate(det["m_playerList"])]
    try:  # MMR lives with the lobby user; players and users meet through the working-set slot
        init = proto.decode_replay_initdata(a.read_file("replay.initData"))["m_syncLobbyState"]
        users = init["m_userInitialData"]
        slot_user = {sl["m_workingSetSlotId"]: sl["m_userId"] for sl in init["m_lobbyState"]["m_slots"]
                     if sl.get("m_userId") is not None}
        for pl, p in zip(players, det["m_playerList"]):
            u = slot_user.get(p.get("m_workingSetSlotId"))
            if u is not None and users[u].get("m_scaledRating") is not None:
                pl["mmr"] = users[u]["m_scaledRating"]
    except Exception:
        pass
    base_build, data_version = int(meta["BaseBuild"][4:]), meta.get("DataVersion")
    return dict(build=build, base_build=base_build, game_version=".".join(meta["GameVersion"].split(".")[:-1]),
                data_version=data_version, map=_s(det["m_title"]), map_hash=h[8:].hex(), mods=mods,
                handles=[depot.parse_handle(x) for x in handles],
                ruleset=ruleset_key(base_build, data_version, mods),
                # The header's exact loop count (metadata "Duration" is in normal-speed seconds
                # of 16 loops, not real seconds)
                duration_loops=int(header["m_elapsedGameLoops"]), players=players)


def ruleset_key(base_build, data_version, mods):
    """The rules a game ran under: build + data version + its mods (balance mods change
    between games on the same build)."""
    import hashlib
    raw = json.dumps([base_build, data_version, mods])
    return f"{base_build}_{hashlib.sha1(raw.encode()).hexdigest()[:8]}"


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
    if p is None and handles and FETCH_MISSING:
        depot.ensure_cached(handles, CACHE_DIRS, log=lambda m: print(m, flush=True))
        p = depot.find_cached(CACHE_DIRS, map_hash, "s2ma")
    return p


def tracker_tables(path, until):
    """Per-player stats and upgrades for every player, from the tracker events."""
    import mpyq
    a = mpyq.MPQArchive(path)
    proto, _ = _protocol(a)
    stats = {k: [] for k in ("loop", "player", *PLAYER_STATS.values())}
    ups = {"loop": [], "player": [], "name": []}
    for e in proto.decode_replay_tracker_events(a.read_file("replay.tracker.events")):
        if e["_gameloop"] > until:
            break
        t = e["_event"].split(".")[-1]
        if t == "SPlayerStatsEvent":
            st = e["m_stats"]
            stats["loop"].append(e["_gameloop"])
            stats["player"].append(e["m_playerId"])
            for k, name in PLAYER_STATS.items():
                v = st.get(k, 0)
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
LAUNCH_LOCK = os.path.join(tempfile.gettempdir(), "cosmicstar_sc2_launch.lock")


class launch_lock:
    """Serializes replay launches across worker processes. Passing map_data makes SC2
    write a shared temp map file; concurrent launches can load each other's map, which
    shows up as InvalidReplayData or a desync."""

    def __enter__(self):
        self.f = open(LAUNCH_LOCK, "a+")
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


def extract(path, out_path, step, until, perspective=0, substep=0, ui=False):
    """Plays one replay and saves every frame (see the module doc). perspective 0: observer,
    fog off; a player slot: as that player sees it (fog on); ui: also the engine's control group
    and selection panel summaries (ui_*), which need the feature layer on."""
    from pysc2.lib import protocol
    from s2clientprotocol import sc2api_pb2 as sc_pb

    info = replay_info(path)
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
    chunks = {k: [] for k in U_DTYPES}    # ... packed into typed arrays per frame (Python lists for a whole game ran to GBs)
    n_rows = 0
    rag = {k: [] for k in R_DTYPES}  # ragged per-unit lists (production, buffs, passengers, orders)
    eff = {k: [] for k in E_DTYPES}  # per-frame effects, one row per effect position
    dead = []
    frame_loop, frame_start, eff_start, dead_start = [], [0], [0], [0]
    vis, creep = [], []  # player views only

    if perspective == 0:
        substep = 0  # no fog in the observer view: the per-loop observation requests are pointless there
    proc = run_config.start(want_rgb=False)
    with proc as c:
        opts = sc_pb.InterfaceOptions(raw=True, score=False, show_cloaked=True, show_burrowed_shadows=True,
                                      show_placeholders=False, raw_affects_selection=False)
        if perspective and ui:
            # ui_data (control groups, selection panel) is only filled with the feature layer
            # on; the smallest one costs nothing measurable (3000 loops: 4.4 s vs 4.1 s raw-only)
            opts.feature_layer.width = 24
            opts.feature_layer.resolution.x = opts.feature_layer.resolution.y = 16
            opts.feature_layer.minimap_resolution.x = opts.feature_layer.minimap_resolution.y = 16
        req = sc_pb.RequestStartReplay(
            # observer: fog off, everything; a player: that player's view, fog on (module doc)
            replay_data=data, observed_player_id=perspective, disable_fog=perspective == 0, options=opts)
        if current:
            c.start_replay(req)
        else:
            req.map_data = open(mp, "rb").read()
            with launch_lock():
                c.start_replay(req)
        minimize_client_window(proc.pid)
        data = c.data_raw()  # this game's own rules decide what counts as production
        producers = {u.unit_id for u in data.units  # structures, eggs, cocoons, larva
                     if STRUCTURE in u.attributes or any(k in u.name for k in ("Egg", "Cocoon", "Larva"))}
        production = {a.ability_id for a in data.abilities if PRODUCTION.search(a.link_name)}
        gi = c.game_info()
        sr = gi.start_raw
        info["playable"] = [sr.playable_area.p0.x, sr.playable_area.p0.y, sr.playable_area.p1.x, sr.playable_area.p1.y]
        info["map_size"] = [sr.map_size.x, sr.map_size.y]
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
        ping = sc_pb.Request(observation=sc_pb.RequestObservation())
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
                        c._client.write(ping)
                        r = c._client._sock.recv()
                        if perspective:  # ~0.1 ms to parse with upb: the commands and deaths this observation
                            ob = sc_pb.Response.FromString(r).observation  # reports (both are "since the last one"),
                            take_actions(ob.actions, len(frame_loop))      # and the player's selection
                            dead.extend(ids[t] for t in ob.observation.raw_data.event.dead_units if t in ids)
                            take_selection(ob.observation.raw_data.units, ob.observation.game_loop)
                    c.step(substep)
                else:
                    c.step(step)
                o = c.observe()
            except protocol.ProtocolError as ex:  # a player's view ends when that player leaves the game
                if "already ended" in str(ex) and perspective:
                    break
                raise
            prev, loop = loop, o.observation.game_loop
            stalled = stalled + 1 if loop <= prev else 0
            if stalled >= 5:
                if loop >= until - 4 * step:  # the replay simply ended
                    break
                raise RuntimeError(f"replay stalled at {loop / LPS:.0f}s")
            raw = o.observation.raw_data
            for u in raw.units:
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
                for pu in u.passengers:
                    rag["pass_id"].append(uid(pu.tag))
                    rag["pass_type"].append(pu.unit_type)
                    rag["pass_hp"].append(pu.health)
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
                chunks[k].append(np.asarray(v, dtype=U_DTYPES[k]))
                v.clear()
            frame_start.append(n_rows)
            eff_start.append(len(eff["type"]))
            dead_start.append(len(dead))
            if perspective:
                vis.append(np.frombuffer(raw.map_state.visibility.data, np.uint8).copy())
                creep.append(np.frombuffer(raw.map_state.creep.data, np.uint8).copy())
                take_actions(o.actions, len(frame_loop) - 1)
                take_selection(raw.units, loop)
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
            # player view only the player's own units are checked against the replay's log
            desync.check(loop, [u.tag for u in raw.units if not perspective or u.owner == perspective])
            # The game is decided once every player of some team has a result (left or lost);
            # one player leaving a team game doesn't end it (their ally plays on).
            done_players |= {r.player_id for r in o.player_result}
            if any(members <= done_players for members in teams.values()):
                info["end"] = dict(loop=loop, results={str(r.player_id): r.result for r in o.player_result})
                break

    arrays = {f"u_{k}": (np.concatenate(chunks[k]) if chunks[k] else np.zeros(0, U_DTYPES[k])) for k in U_DTYPES}
    arrays.update({k: np.asarray(v, dtype=R_DTYPES[k]) for k, v in rag.items()})
    arrays.update({f"e_{k}": np.asarray(v, dtype=E_DTYPES[k]) for k, v in eff.items()})
    arrays.update(frame_loop=np.asarray(frame_loop, np.int32), frame_start=np.asarray(frame_start, np.int64),
                  e_frame_start=np.asarray(eff_start, np.int64), dead_frame_start=np.asarray(dead_start, np.int64),
                  dead_id=np.asarray(dead, np.int32), **grids)
    arrays.update({f"p_{k}": np.asarray(v, np.float32 if k not in ("loop", "player") else np.int32) for k, v in stats.items()})
    arrays.update(up_loop=np.asarray(ups["loop"], np.int32), up_player=np.asarray(ups["player"], np.int8),
                  up_name=np.asarray(ups["name"]))
    if perspective and vis:
        ms = raw.map_state
        h, w = ms.visibility.size.y, ms.visibility.size.x
        arrays["vis"] = np.stack(vis).reshape(len(vis), h, -1)
        arrays["creep"] = np.stack(creep).reshape(len(creep), h, -1)
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
    died = selection_tracker.deaths_by_replay_tag(desync.died)  # exact death loops, from the replay
    slots = [perspective] if perspective else [p["slot"] for p in info["players"]]
    arrays.update(selection_tracker.frame_groups(acts, slots, arrays["frame_loop"], tags, died))
    if perspective:
        checks = selection_tracker.validate(arrays, acts, perspective, died)
        stats["groups"] = {k: int(v) for k, v in checks.items() if k != "mismatches"}
    stats["units"] = len(ids)
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out_path)
    return stats


# ---------------------------------------------------------------- job list and workers

def installed_builds():
    """Base builds we have an engine for, in the install or the side install with the older
    builds (wine_run_config.installed_builds; the run config picks the install per build)."""
    from cosmicstar.envs import wine_run_config
    return wine_run_config.installed_builds()


def out_name(path):
    import hashlib
    return hashlib.sha1(os.path.basename(path).encode()).hexdigest()[:12] + ".npz"


def build_manifest(replay_dir, player, out_dir, require_player=False, any_ruleset=False):
    """Every replay we can extract: right patch, binary installed, map cached (and the
    player present, if required). any_ruleset: also rulesets without an Editor balance export
    (their abilities can't be named, which only matters for the action vocabulary)."""
    have = installed_builds()
    # rulesets the Editor could export (export_balance.py): without that we can't name actions
    exported = os.path.join(BALANCE_DIR, "status.json")
    exported = json.load(open(exported)) if os.path.exists(exported) and not any_ruleset else None
    jobs, skipped = [], {}
    for path in sorted(glob.glob(os.path.join(replay_dir, "**", "*.SC2Replay"), recursive=True), key=os.path.getmtime):
        try:
            info = replay_info(path)
        except Exception:
            skipped["unreadable"] = skipped.get("unreadable", 0) + 1
            continue
        why = None
        if info["build"] in EXCLUDED_BUILDS:
            why = "patch 5.0.16"
        elif info["base_build"] not in have:
            why = f"base build {info['base_build']} not installed"
        elif info["duration_loops"] < 90 * LPS:
            why = "under 90 s (remake/leave)"
        elif map_path(info["map_hash"], info["handles"]) is None:
            why = "map not cached (depot fetch failed)" if FETCH_MISSING else "map not cached"
        elif exported is not None and not exported.get(info["ruleset"], {}).get("ok"):
            why = "ruleset not exported" if info["ruleset"] not in exported else "ruleset not exportable"
        if why is None and require_player and not any(player and player.lower() in p["name"].lower()
                                                       for p in info["players"]):
            why = "player not in game"
        if why:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        jobs.append(dict(path=path, out=os.path.join(out_dir, out_name(path)), map=info["map"],
                         base_build=info["base_build"], ruleset=info["ruleset"],
                         slots=[p["slot"] for p in info["players"]], teams={p["slot"]: p["team"] for p in info["players"]}))
    return jobs, skipped


# The engine's view of slot 4 crashes at the game's first research order (worldmodel/README.md);
# that view is skipped (the 2v2 composition from the teammate's view was dropped with the 1v1 focus).
CRASHING_VIEWS = {4}


def pov_name(pov_out, job, slot):
    return os.path.join(pov_out, out_name(job["path"])[:-4] + f"_p{slot}.npz")


def claim(job):
    """Atomically claims a job for this worker (any free worker takes the next one)."""
    try:
        os.close(os.open(job["out"] + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


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


def run_worker(jobs, step, until, substep=0, ui=False):
    from absl import flags
    flags.FLAGS(sys.argv[:1])
    for j in jobs:
        if os.path.exists(j["out"]) or not claim(j):
            continue
        t0 = time.time()
        for attempt in range(3):
            try:
                st = extract(j["path"], j["out"], step, until, j.get("perspective", 0), substep, ui)
                print(json.dumps(dict(ok=os.path.basename(j["path"]), view=j.get("perspective", 0), **st,
                                      mb=round(os.path.getsize(j["out"]) / 1e6, 1), s=round(time.time() - t0), tries=attempt + 1)), flush=True)
                break
            except Exception as ex:
                # replay starts fail in bursts while another instance fetches data for a map it
                # hasn't seen (InvalidReplayData); they pass once it's cached, so wait and retry.
                # The client also stalls past pysc2's 120 s websocket timeout now and then
                # (seen once in 12 games under Wine; the game passed alone), so retry those too,
                # and under Proton with several instances launching, the websocket of one is lost
                # or never comes up (ConnectError / ConnectionError / WebSocketConnectionClosed:
                # 6 of 12 player views with 4 workers, 2026-10-30); those passed alone as well.
                if ("InvalidReplayData" in repr(ex) or "timed out" in repr(ex) or "Connect" in repr(ex)) and attempt < 2:
                    time.sleep(45 * (attempt + 1))
                    continue
                print(json.dumps(dict(fail=os.path.basename(j["path"]), error=repr(ex)[:200], tries=attempt + 1)), flush=True)
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default=r"C:\Users\prade\Documents\StarCraft II\Accounts\52583725\1-S2-1-1544307\Replays\Multiplayer")
    ap.add_argument("--player", default="zogger", help="with --require_player: only games with this player")
    ap.add_argument("--require_player", action="store_true", help="skip games without --player")
    ap.add_argument("--out", default="data/worldmodel/full")
    ap.add_argument("--step", type=int, default=16, help="game loops between frames (16 = 0.71 s)")
    ap.add_argument("--until", type=float, default=0, help="seconds of game time to extract (0 = whole game)")
    ap.add_argument("--substep", type=int, default=None, help="request an observation every this many loops (discarded "
                    "unparsed) and keep one every --step loops; default 1 for player views, 0 (none) for observer-only "
                    "runs. Without it a player view leaks enemy movement through the fog (module doc)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N replays (newest last)")
    ap.add_argument("--views", default="observer", help="observer, players, or observer,players")
    ap.add_argument("--pov_out", default="data/worldmodel/pov", help="player views go here")
    ap.add_argument("--games", help="json list of output names (<hash>.npz): only these games")
    ap.add_argument("--no_fetch", action="store_true", help="don't fetch missing maps/mods from the depot")
    ap.add_argument("--ui", action="store_true", help="player views: also keep the engine's control group / selection "
                    "panel summaries (ui_*, for validating the tracked groups); turns the feature layer on")
    ap.add_argument("--any_ruleset", action="store_true", help="also replays whose ruleset has no Editor balance export "
                    "(their abilities can't be named)")
    ap.add_argument("--worker", help=argparse.SUPPRESS)  # internal: job list for one worker process
    args = ap.parse_args()

    global FETCH_MISSING
    FETCH_MISSING = not args.no_fetch
    os.makedirs(args.out, exist_ok=True)
    manifest = os.path.join(args.out, "manifest.json")
    if args.worker:
        run_worker(json.load(open(args.worker)), args.step, int(args.until * LPS), args.substep, args.ui)
        return
    os.makedirs(args.pov_out, exist_ok=True)
    # resuming: claims left by an interrupted run (no output, no partial file) are free again
    for c in glob.glob(os.path.join(args.out, "*.claim")) + glob.glob(os.path.join(args.pov_out, "*.claim")):
        out_path = c[:-len(".claim")]
        if not os.path.exists(out_path):
            if os.path.exists(out_path + ".tmp.npz"):
                os.remove(out_path + ".tmp.npz")
            os.remove(c)

    jobs, skipped = build_manifest(args.replays, args.player, args.out, args.require_player, args.any_ruleset)
    if args.games:
        keep = set(json.load(open(args.games)))
        jobs = [j for j in jobs if os.path.basename(j["out"]) in keep]
    if args.limit:
        jobs = jobs[-args.limit:]
    views = set(args.views.split(","))
    if args.substep is None:
        args.substep = 1 if "players" in views else 0
    expanded = []  # one job per (replay, view)
    for j in jobs:
        if "observer" in views:
            expanded.append(dict(j, perspective=0))
        if "players" in views:
            for slot in j["slots"]:
                if slot in CRASHING_VIEWS:  # the engine crashes in this view
                    continue
                expanded.append(dict(j, perspective=slot, out=pov_name(args.pov_out, j, slot)))
    jobs = expanded
    json.dump(jobs, open(manifest, "w"), indent=1)
    for c in glob.glob(os.path.join(args.out, "*.claim")) + glob.glob(os.path.join(args.pov_out, "*.claim")):
        os.remove(c)  # claims from a previous run
    todo = sum(not os.path.exists(j["out"]) for j in jobs)
    print(f"{len(jobs)} extractions ({args.views}; step {args.step}, substep {args.substep}; {todo} not done yet); "
          f"skipped replays: {skipped}", flush=True)
    env = dict(os.environ)
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = protobuf_impl()
    print("protobuf parser:", env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"], flush=True)

    def run(job_list, n):
        """n worker processes sharing job_list (each claims the next free job). Workers
        share this process's stdout; piping them through here would block all but one."""
        path = os.path.join(args.out, "jobs_now.json")
        json.dump(job_list, open(path, "w"))
        procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--out", args.out, "--step",
                                   str(args.step), "--until", str(args.until), "--substep", str(args.substep), "--worker", path]
                                  + (["--no_fetch"] if args.no_fetch else []) + (["--ui"] if args.ui else []),
                                  env=env, stderr=subprocess.DEVNULL) for _ in range(n)]
        for p in procs:
            p.wait()

    # One base build at a time. Older builds' game data is fetched on demand into the
    # shared data store; instances fetching at the same time (worse: for different
    # builds) fail to load replays (InvalidReplayData) or crash. So each build starts
    # with a single game alone, which pulls its data in, then runs in parallel.
    for build in sorted({j["base_build"] for j in jobs}):
        left = [j for j in jobs if j["base_build"] == build and not os.path.exists(j["out"])]
        if not left:
            continue
        print(json.dumps(dict(build=build, todo=len(left))), flush=True)
        for j in left[:3]:  # warm-up, alone, until one succeeds
            run([j], 1)
            if os.path.exists(j["out"]):
                break
        run(left, min(args.workers, len(left)))


if __name__ == "__main__":
    main()
