"""Reconstructs a Zerg player's build order from a replay's tracker events.

No game engine needed. The tracker logs larva -> egg when a morph starts, and
when an egg hatches the egg dies on the same loop (and at the same spot) as its
unit is born. So each morph's start time and product are exact, and supply
(consumed when the egg starts), supply cap and larva count can be rebuilt at
every moment.

Usage: python scripts/replay_build_order.py <replay> [<replay> ...] [--until 270]
"""

import argparse
import os

import mpyq
from s2protocol import versions

LPS = 22.4
FOOD = {"Drone": 1, "Zergling": 0.5, "Queen": 2, "Roach": 2, "Baneling": 0.5, "Overlord": 0}
QUEEN_LOOPS = 800  # queens aren't larva morphs: start = birth - build time
STRUCTURE_FROM_DRONE = {"Hatchery", "Extractor", "ExtractorRich", "SpawningPool", "EvolutionChamber",
                        "RoachWarren", "BanelingNest", "SpineCrawler", "SporeCrawler"}


def protocol_for(archive):
    header = versions.latest().decode_replay_header(archive.header["user_data_header"]["content"])
    build = header["m_version"]["m_build"]
    try:
        return versions.build(build), build
    except ImportError:
        return versions.latest(), build  # newer than s2protocol knows; the format rarely changes


def s(b):
    return b.decode("utf-8", "replace") if isinstance(b, bytes) else b


def key(e):
    return (e["m_unitTagIndex"], e["m_unitTagRecycle"])


def egg_products(events, pid):
    """(egg tag, start loop) -> unit type it hatched into."""
    born_at = {}  # loop -> [(type, (x, y))] for this player's larva-morph units
    for e in events:
        if e["_event"].endswith("SUnitBornEvent") and e["m_controlPlayerId"] == pid and e["_gameloop"] > 0:
            typ = s(e["m_unitTypeName"])
            if typ in FOOD and typ != "Queen":
                born_at.setdefault(e["_gameloop"], []).append((typ, (e["m_x"], e["m_y"])))
    tag_type, tag_owner, open_eggs, products = {}, {}, {}, {}
    for e in events:
        t = e["_event"].split(".")[-1]
        if t in ("SUnitBornEvent", "SUnitInitEvent"):
            tag_type[key(e)], tag_owner[key(e)] = s(e["m_unitTypeName"]), e["m_controlPlayerId"]
        elif t == "SUnitTypeChangeEvent":
            k, new = key(e), s(e["m_unitTypeName"])
            if tag_owner.get(k) == pid and new == "Egg" and tag_type.get(k) == "Larva":
                open_eggs[k] = e["_gameloop"]
            tag_type[k] = new
        elif t == "SUnitDiedEvent" and key(e) in open_eggs:
            k = key(e)
            cands = born_at.get(e["_gameloop"], [])
            if cands:
                pos = (e["m_x"], e["m_y"])
                i = min(range(len(cands)), key=lambda i: abs(cands[i][1][0] - pos[0]) + abs(cands[i][1][1] - pos[1]))
                products[(k, open_eggs[k])] = cands.pop(i)[0]
            del open_eggs[k]
    return products


def timeline(path, until, player=None):
    archive = mpyq.MPQArchive(path)
    proto, build = protocol_for(archive)
    details = proto.decode_replay_details(archive.read_file("replay.details"))
    players = [(i + 1, s(p["m_name"]), s(p["m_race"])) for i, p in enumerate(details["m_playerList"])]
    zergs = [p for p in players if p[2] == "Zerg" and (player is None or player.lower() in p[1].lower())]
    if not zergs:
        return None
    pid, name, _ = zergs[0]
    events = list(proto.decode_replay_tracker_events(archive.read_file("replay.tracker.events")))
    products = egg_products(events, pid)

    queen_starts = sorted(e["_gameloop"] - QUEEN_LOOPS for e in events
                          if e["_event"].endswith("SUnitBornEvent") and e["m_controlPlayerId"] == pid
                          and s(e["m_unitTypeName"]) == "Queen")
    # The drone that becomes a building is logged as dying when the building
    # finishes; supply already dropped when it started, so skip those deaths.
    done_loops = [e["_gameloop"] for e in events if e["_event"].endswith("SUnitDoneEvent")]
    tag_type, tag_owner = {}, {}
    larva = set()
    done = set()
    supply, cap = 12.0, 14
    lines, marks = [], {}

    def line(loop, what):
        lines.append(f"{loop / LPS:6.1f}s  {supply:4g}/{cap:<3d} larva {len(larva)}  {what}")

    for e in events:
        loop = e["_gameloop"]
        if loop > until * LPS:
            break
        while queen_starts and queen_starts[0] <= loop:
            q = queen_starts.pop(0)
            supply += 2
            marks.setdefault("Queen", []).append((q, supply, cap, len(larva)))
            lines.append(f"{q / LPS:6.1f}s  {supply:4g}/{cap:<3d} larva {len(larva)}  queen started")
        t = e["_event"].split(".")[-1]
        k = key(e) if "m_unitTagIndex" in e else None
        if t == "SUnitBornEvent":
            typ = s(e["m_unitTypeName"])
            tag_type[k], tag_owner[k] = typ, e["m_controlPlayerId"]
            if e["m_controlPlayerId"] != pid or loop == 0:
                if typ == "Larva" and e["m_controlPlayerId"] == pid:
                    larva.add(k)
                continue
            if typ == "Larva":
                larva.add(k)
            elif typ == "Overlord":
                cap = min(200, cap + 8)
                line(loop, "overlord pops")
            elif typ == "Queen":
                line(loop, "queen pops")
        elif t == "SUnitTypeChangeEvent":
            new = s(e["m_unitTypeName"])
            if tag_owner.get(k) == pid and new == "Egg" and tag_type.get(k) == "Larva":
                larva.discard(k)
                typ = products.get((k, loop), "?")
                supply += FOOD.get(typ, 0) * (2 if typ == "Zergling" else 1)
                marks.setdefault(typ, []).append((loop, supply, cap, len(larva)))
                line(loop, f"{typ.lower()} started" + (" (pair)" if typ == "Zergling" else ""))
            tag_type[k] = new
        elif t == "SUnitInitEvent":
            typ = s(e["m_unitTypeName"])
            tag_type[k], tag_owner[k] = typ, e["m_controlPlayerId"]
            if e["m_controlPlayerId"] == pid and typ in STRUCTURE_FROM_DRONE:
                supply -= 1
                marks.setdefault(f"{typ} started", []).append((loop, supply, cap, len(larva)))
                line(loop, f"{typ} started")
        elif t == "SUnitDoneEvent" and tag_owner.get(k) == pid:
            typ = tag_type.get(k)
            done.add(k)
            if typ == "Hatchery":
                cap = min(200, cap + 6)
            marks.setdefault(f"{typ} done", []).append((loop, supply, cap, len(larva)))
            line(loop, f"{typ} done")
        elif t == "SUnitDiedEvent" and tag_owner.get(k) == pid:
            typ = tag_type.get(k)
            larva.discard(k)
            if typ == "Drone" and any(abs(loop - d) <= 2 for d in done_loops):
                continue  # the builder of a structure that just finished
            if typ in ("Drone", "Zergling", "Queen"):
                supply -= FOOD[typ]
                line(loop, f"{typ} died")
            elif typ in STRUCTURE_FROM_DRONE and k not in done and e.get("m_killerPlayerId") in (None, pid):
                supply += 1  # cancelled: the drone comes back
                line(loop, f"{typ} cancelled")
        elif t == "SPlayerStatsEvent" and e["m_playerId"] == pid and loop % int(LPS * 30) < 160:
            st = e["m_stats"]
            lines.append(f"{loop / LPS:6.1f}s        [stats: supply {st['m_scoreValueFoodUsed'] / 4096:g}/"
                         f"{st['m_scoreValueFoodMade'] / 4096:g}, workers {st['m_scoreValueWorkersActiveCount']},"
                         f" bank {st['m_scoreValueMineralsCurrent']}, income {st['m_scoreValueMineralsCollectionRate']}/min]")
    opp = ", ".join(f"{n} ({r})" for i, n, r in players if i != pid)
    return dict(name=name, opp=opp, map=s(details["m_title"]), build=build, lines=lines, marks=marks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replays", nargs="+")
    ap.add_argument("--until", type=float, default=270, help="seconds of game time to show")
    ap.add_argument("--player", default=None, help="Zerg player name (substring); default: the first Zerg")
    args = ap.parse_args()
    for path in args.replays:
        r = timeline(path, args.until, args.player)
        if r is None:
            print(f"\n=== {os.path.basename(path)}: no Zerg player")
            continue
        print(f"\n=== {os.path.basename(path)}  ({r['map']}, build {r['build']})  {r['name']} (Zerg) vs {r['opp']}")
        for l in r["lines"]:
            print(l)


if __name__ == "__main__":
    main()
