"""Dumps the game's static data for every ruleset our replays were played under.

A ruleset is build + data version + mods (balance mods change between games on the
same build), see extract_states.ruleset_key. For each ruleset one replay is started
and the engine is asked for its data while that replay is loaded, so the dump is
exactly the rules those games ran under: units (costs, supply, HP, armor, speed,
sight, attributes, weapons), abilities, upgrades, buffs and effects.

Output: <out>/<ruleset>.json (the engine's ResponseData, field names as in the API)
plus <out>/index.json mapping ruleset -> build, data version, mods, example replay.
Rulesets run one at a time (older builds fetch game data on first use, and parallel
fetches fail).

    python scripts/dump_rules.py --replays <dir> --out data/worldmodel/rules
"""

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import extract_states as ex  # noqa: E402


def dump(path, info):
    from google.protobuf import json_format
    from s2clientprotocol import sc2api_pb2 as sc_pb

    run_config, current = ex.run_config_for(info)
    with run_config.start(want_rgb=False) as c:
        req = sc_pb.RequestStartReplay(
            replay_data=open(path, "rb").read(),
            observed_player_id=1,
            options=sc_pb.InterfaceOptions(raw=True),
        )
        if not current:
            req.map_data = open(
                ex.map_path(info["map_hash"], info["handles"]), "rb"
            ).read()
        c.start_replay(req)
        data = c.data_raw()
    return json_format.MessageToDict(data, preserving_proto_field_name=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--replays",
        default=r"C:\Users\prade\Documents\StarCraft II\Accounts\52583725\1-S2-1-1544307\Replays\Multiplayer",
    )
    ap.add_argument("--out", default="data/worldmodel/rules")
    ap.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="replays to try per ruleset before giving up",
    )
    args = ap.parse_args()
    from absl import flags

    flags.FLAGS(sys.argv[:1])
    os.makedirs(args.out, exist_ok=True)

    have = ex.installed_builds()
    groups = {}
    for path in sorted(
        glob.glob(os.path.join(args.replays, "*.SC2Replay")), key=os.path.getmtime
    ):
        try:
            info = ex.replay_info(path)
        except Exception:
            continue
        if (
            info["build"] in ex.EXCLUDED_BUILDS
            or info["base_build"] not in have
            or not ex.map_path(info["map_hash"], info["handles"])
        ):
            continue
        groups.setdefault(info["ruleset"], []).append((path, info))
    index_path = os.path.join(args.out, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {}
    print(
        f"{len(groups)} rulesets across {sum(map(len, groups.values()))} replays",
        flush=True,
    )
    for key, games in sorted(groups.items()):
        out = os.path.join(args.out, key + ".json")
        if os.path.exists(out):
            continue
        for path, info in games[
            -args.attempts :
        ]:  # newest games first fail least often
            try:
                data = dump(path, info)
            except Exception as e:
                print(
                    f"  {key}: {os.path.basename(path)} failed: {repr(e)[:100]}",
                    flush=True,
                )
                continue
            json.dump(data, open(out, "w"))
            index[key] = dict(
                base_build=info["base_build"],
                data_version=info["data_version"],
                mods=info["mods"],
                game_version=info["game_version"],
                games=len(games),
                example=os.path.basename(path),
            )
            json.dump(index, open(index_path, "w"), indent=1)
            print(
                f"{key}: {len(data.get('units', []))} units,"
                f" {len(data.get('abilities', []))} abilities,"
                f" {len(data.get('upgrades', []))} upgrades,"
                f" {len(data.get('buffs', []))} buffs, {len(data.get('effects', []))}"
                f" effects  ({len(games)} games)",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
