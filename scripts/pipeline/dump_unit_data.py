"""Dumps static unit data (name, attributes, costs, supply) from the running game
version.

python scripts/dump_unit_data.py --out data/worldmodel/unit_data.json
"""

import argparse
import json
import os
import sys

from absl import flags
from pysc2 import run_configs
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb

STRUCTURE = 8  # Attribute.Structure
WORKERS = {45, 104, 84}  # SCV, Drone, Probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="data/maps/ReclamationLE.SC2Map")
    ap.add_argument("--out", default="data/worldmodel/unit_data.json")
    args = ap.parse_args()
    flags.FLAGS(sys.argv[:1])
    rc = run_configs.get()
    with rc.start(want_rgb=False) as c:
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(map_path=os.path.abspath(args.map)), realtime=False
        )
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(
            type=sc_pb.Computer, race=sc_common.Terran, difficulty=sc_pb.VeryEasy
        )
        c.create_game(create)
        c.join_game(
            sc_pb.RequestJoinGame(
                race=sc_common.Zerg, options=sc_pb.InterfaceOptions(raw=True)
            )
        )
        data = c.data_raw()
    units = {}
    for u in data.units:
        if not u.name:
            continue
        attrs = list(u.attributes)
        units[u.unit_id] = dict(
            name=u.name,
            race=u.race,
            attributes=attrs,
            structure=STRUCTURE in attrs,
            worker=u.unit_id in WORKERS,
            food=u.food_required,
            minerals=u.mineral_cost,
            vespene=u.vespene_cost,
            build_time=u.build_time,
            available=u.available,
        )
    json.dump(units, open(args.out, "w"), indent=1)
    print(f"{len(units)} unit types -> {args.out}")


if __name__ == "__main__":
    main()
