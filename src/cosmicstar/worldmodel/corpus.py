"""The replay corpus and the games extracted from it: one tree each, mirrored.

    data/replays/<format>/<build>/<map>/<id>.SC2Replay     the replay (<map>: the map's hash, see below)
                                        <id>.json          its sidecar (replay_sidecar): players (name,
                                                           race, MMR, result), teams, duration, date,
                                                           map, builds, ruleset, where it came from
    data/replays/index.json                                every sidecar in one file, rebuilt by
                                                           build_index (for queries; the sidecars
                                                           are the source of truth)
    data/replays/maps.json                                 {map dir: name, games, builds}, rebuilt
                                                           with the index: the human's map list
    data/games/<format>/<build>/<map>/<id>.p<slot>.cst     a player's view (scripts/extract_states.py)
                                        <id>.acts.npz      the replay's action table (replay_actions)
                                        <id>.meta.json     extraction sidecar: per view the layout,
                                                           step, substep, stats, extractor commit
                                        <id>.p<slot>.fail.json   a failed view (error, tries); gone
                                                           once the view succeeds
                                        <id>.p<slot>.claim       transient: a worker holds the view
    data/games/splits/<name>.json                          game ids (scripts/make_splits.py)
    data/games/logs/<run>.log, <run>.json                  each extraction run's worker output and
                                                           its settings / job list

<id> is the first 16 hex digits of the sha1 of the replay file: the same game from two packs
is one game (both packs listed in its sidecar), a renamed file is not a new game. <format> is
1v1, 2v2, ..., ffa, from the replay's teams; <build> the base build; <map> the first MAP_HEX
digits of the map file's hash (the replay's cache handle, what the engine actually loaded): a
tournament's own version of a ladder map, or a map's next revision, is a different directory
even under the same name, and the same map under two names is one (the sidecar has the name;
maps.json lists them). Pruning: a format, patch or map is one
directory in both trees; anything else is a query over the sidecars (or index.json) that
yields ids, then those ids' files. scripts/ingest_replays.py sorts packs into the replay tree.
"""
import collections
import datetime
import hashlib
import json
import os
import re

REPLAYS = "data/replays"
GAMES = "data/games"
INDEX = "index.json"
EXT = ".SC2Replay"
ID_HEX = 16


def game_id(path_or_bytes):
    """sha1 of the replay's bytes, first ID_HEX hex digits."""
    data = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes, "rb").read()
    return hashlib.sha1(data).hexdigest()[:ID_HEX]


def map_slug(name):
    """'Ley Lines LE' -> 'ley_lines_le'; anything not alphanumeric becomes one underscore."""
    s = re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")
    return s or "unknown"


def game_format(players):
    """'1v1', '2v2', ..., 'ffa' (3+ teams of one), else the team sizes joined ('1v2')."""
    sizes = sorted(collections.Counter(p["team"] for p in players).values())
    if not sizes:
        return "empty"
    if len(sizes) > 2 and all(s == 1 for s in sizes):
        return "ffa"
    return "v".join(str(s) for s in sizes)


MAP_HEX = 16


def map_dir(map_hash):
    return map_hash[:MAP_HEX]


def rel_dir(fmt, base_build, map_hash):
    return os.path.join(fmt, str(base_build), map_dir(map_hash))


# ---------------------------------------------------------------- replay metadata (no engine)

def _s(b):
    return b.decode("utf-8", "replace") if isinstance(b, bytes) else b


def _protocol(archive):
    from s2protocol import versions
    header = versions.latest().decode_replay_header(archive.header["user_data_header"]["content"])
    build = header["m_version"]["m_build"]
    try:
        return versions.build(build), build
    except ImportError:
        return versions.latest(), build


def ruleset_key(base_build, data_version, mods):
    """The rules a game ran under: build + data version + its mods (balance mods change
    between games on the same build)."""
    raw = json.dumps([base_build, data_version, mods])
    return f"{base_build}_{hashlib.sha1(raw.encode()).hexdigest()[:8]}"


def filetime_iso(ft, offset=0):
    """A Windows FILETIME (100 ns since 1601, as the replay's m_timeUTC) -> ISO 8601 UTC."""
    try:
        t = datetime.datetime(1601, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(microseconds=ft / 10)
        return t.replace(microsecond=0).isoformat()
    except (OverflowError, ValueError, TypeError):
        return None


def replay_info(path):
    """Cheap metadata from the replay file (no engine): builds, map and its depot handles,
    ruleset, duration, players with MMR, date."""
    import mpyq
    from cosmicstar.envs import depot
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
    map_cells = None
    try:  # MMR lives with the lobby user; players and users meet through the working-set slot
        init = proto.decode_replay_initdata(a.read_file("replay.initData"))["m_syncLobbyState"]
        gd = init["m_gameDescription"]
        map_cells = [gd["m_mapSizeX"], gd["m_mapSizeY"]]
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
                duration_loops=int(header["m_elapsedGameLoops"]), players=players,
                time_utc=filetime_iso(det.get("m_timeUTC")), format=game_format(players), map_cells=map_cells)


def replay_sidecar(info, gid, rel_path, source):
    """The <id>.json next to a replay. source: {"pack": ..., "file": original name} (merged
    into a list when the same game arrives again)."""
    keep = ("format", "build", "base_build", "game_version", "data_version", "map", "map_hash", "mods", "handles",
            "ruleset", "duration_loops", "players", "time_utc")  # handles: (ext, hash) of the map and mods, for the depot
    d = {k: info[k] for k in keep}
    d.update(id=gid, path=rel_path, duration_s=round(info["duration_loops"] / 22.4, 1),
             sources=[dict(source, ingested=datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat())])
    return d


# ---------------------------------------------------------------- the trees

def iter_sidecars(root=REPLAYS, fmt=None):
    """(path, sidecar dict) for every replay sidecar under root (format filtered if given)."""
    top = os.path.join(root, fmt) if fmt else root
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(".json") and name != INDEX and len(name) == ID_HEX + 5:
                p = os.path.join(dirpath, name)
                try:
                    yield p, json.load(open(p))
                except (OSError, ValueError):
                    continue


MAPS = "maps.json"


def build_index(root=REPLAYS):
    """Rewrites root/index.json ({id: sidecar}) and root/maps.json ({map dir: {name, names,
    games, builds, hash}}). Returns the index."""
    index = {d["id"]: d for _, d in iter_sidecars(root)}
    maps = {}
    for d in index.values():
        m = maps.setdefault(map_dir(d["map_hash"]), dict(name=d["map"], names=[], games=0, builds=[], hash=d["map_hash"]))
        m["games"] += 1
        if d["map"] not in m["names"]:
            m["names"].append(d["map"])
        if d["base_build"] not in m["builds"]:
            m["builds"].append(d["base_build"])
    for name, obj in ((INDEX, index), (MAPS, dict(sorted(maps.items(), key=lambda kv: (kv[1]["name"], kv[0]))))):
        tmp = os.path.join(root, name + ".tmp")
        json.dump(obj, open(tmp, "w"), indent=0 if name == INDEX else 1)
        os.replace(tmp, os.path.join(root, name))
    return index


def map_dirs_named(index, names):
    """The map directories whose name or hash prefix matches any of names (names: real map
    names, slugs, or hash prefixes)."""
    want = {map_slug(n) for n in names} | {n.lower() for n in names}
    out = set()
    for d in index.values():
        md = map_dir(d["map_hash"])
        if map_slug(d["map"]) in want or any(md.startswith(n) for n in want if n and all(c in "0123456789abcdef" for c in n)):
            out.add(md)
    return out


def load_index(root=REPLAYS, rebuild=False):
    """root/index.json, built if missing (or rebuild)."""
    p = os.path.join(root, INDEX)
    if rebuild or not os.path.exists(p):
        return build_index(root)
    return json.load(open(p))


class GameFiles:
    """Paths of one game's files in the games tree."""

    def __init__(self, games_root, rel, gid):
        self.dir = os.path.join(games_root, rel)
        self.id = gid

    def view(self, slot):
        return os.path.join(self.dir, f"{self.id}.p{slot}.cst")

    def acts(self):
        return os.path.join(self.dir, f"{self.id}.acts.npz")

    def meta(self):
        return os.path.join(self.dir, f"{self.id}.meta.json")

    def fail(self, slot):
        return os.path.join(self.dir, f"{self.id}.p{slot}.fail.json")


VIEW_RE = re.compile(r"^(?P<id>[0-9a-f]{%d})\.p(?P<slot>\d+)\.cst$" % ID_HEX)


def acts_path(view_path):
    """The action table next to a view: <dir>/<id>.acts.npz."""
    m = VIEW_RE.match(os.path.basename(view_path))
    return os.path.join(os.path.dirname(view_path), m.group("id") + ".acts.npz") if m else None


def iter_views(games_root=GAMES, fmt=None):
    """(id, slot, path) of every extracted view under the games tree, sorted."""
    top = os.path.join(games_root, fmt) if fmt else games_root
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames.sort()
        for name in sorted(filenames):
            m = VIEW_RE.match(name)
            if m:
                yield m.group("id"), int(m.group("slot")), os.path.join(dirpath, name)


def views_by_id(games_root=GAMES, fmt=None):
    """{id: [view paths]} for the games tree."""
    out = collections.defaultdict(list)
    for gid, _, p in iter_views(games_root, fmt):
        out[gid].append(p)
    return out


def read_meta(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def update_meta(path, slot, record):
    """Merges one view's record into <id>.meta.json under a file lock (two workers finish a
    game's two views independently)."""
    import fcntl
    lock = path + ".lock"
    with open(lock, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        meta = read_meta(path)
        meta.setdefault("views", {})[str(slot)] = record
        tmp = path + ".tmp"
        json.dump(meta, open(tmp, "w"), indent=1)
        os.replace(tmp, path)
    try:
        os.remove(lock)
    except OSError:
        pass
