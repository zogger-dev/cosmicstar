"""Records per-game-loop SCV trajectories while mining.

Starts a 1v1 against a very-easy built-in AI, splits the starting SCVs onto
the main base's mineral line, then steps the game ONE loop at a time and logs
every SCV's position, orders and carry state, plus each patch's remaining
minerals.

--split pro     (default) the standard opening split: one SCV per patch, and
                the spare SCVs (4 of the starting 12) doubled up on the
                closest patches.
--split single  exactly one SCV per patch (no patch contention); spare SCVs
                are parked behind the command center.

--hold_patch    (default on) no bouncing: every loop, any assigned SCV that is
                not carrying minerals and is gathering from a different patch
                (or has gone idle) is re-sent to its own patch, the way a
                player spams gather on the patch they want a worker to stay on.
                Every re-send is logged to holds.csv.

Outputs (in --out_dir):
    trace.csv     one row per (game_loop, scv)
    minerals.csv  one row per (game_loop, patch) whenever a patch's contents change
    holds.csv     one row per re-sent gather command (--hold_patch)
    meta.json     command center / mineral geometry and the scv -> patch assignment
    probe.SC2Replay

Analyze with scripts/mining_analyze.py.
"""

import csv
import json
import math
import os

from absl import app, flags
from pysc2 import run_configs
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import raw_pb2 as sc_raw
from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "Melee/Simple64.SC2Map", "Map path, relative to the SC2 Maps dir.")
flags.DEFINE_integer("loops", 22 * 60 * 3, "Game loops to record (22.4 loops = 1s at Faster).")
flags.DEFINE_integer("seed", 1, "Game random seed.")
flags.DEFINE_string("out_dir", "data/mining", "Output directory.")
flags.DEFINE_enum("split", "pro", ["pro", "single"], "How to put the starting SCVs on minerals.")
flags.DEFINE_bool("hold_patch", True, "Re-send gather to stop SCVs bouncing off their patch.")

SCV = 45
COMMAND_CENTER = 18
ALLIANCE_SELF = 1
ALLIANCE_NEUTRAL = 3

MOVE = 16
HARVEST_GATHER = 3666  # generic; resolves to Harvest_Gather_SCV (295)
HARVEST_GATHER_SCV = 295  # how the gather order shows up in unit.orders
CARRY_MINERAL_BUFFS = {271, 272}  # CarryMineralFieldMinerals, ...HighYield...


def _dist(a, b):
    return math.hypot(a.pos.x - b.pos.x, a.pos.y - b.pos.y)


def _unit_command(ability_id, unit_tags, target_tag=None, point=None):
    cmd = sc_raw.ActionRawUnitCommand(ability_id=ability_id, unit_tags=unit_tags)
    if target_tag is not None:
        cmd.target_unit_tag = target_tag
    if point is not None:
        cmd.target_world_space_pos.x, cmd.target_world_space_pos.y = point
    return sc_pb.Action(action_raw=sc_raw.ActionRaw(unit_command=cmd))


def _start_game(run_config, controller):
    create = sc_pb.RequestCreateGame(
        local_map=sc_pb.LocalMap(
            map_path=FLAGS.map, map_data=run_config.map_data(FLAGS.map)
        ),
        realtime=False,
        random_seed=FLAGS.seed,
    )
    create.player_setup.add(type=sc_pb.Participant)
    create.player_setup.add(
        type=sc_pb.Computer, race=sc_common.Terran, difficulty=sc_pb.VeryEasy
    )
    controller.create_game(create)
    controller.join_game(
        sc_pb.RequestJoinGame(
            race=sc_common.Terran, options=sc_pb.InterfaceOptions(raw=True, score=True)
        )
    )


def _assign_workers(units, split):
    """Returns (cc, minerals, assignment, parked_tags, park_point)."""
    cc = next(
        u for u in units if u.alliance == ALLIANCE_SELF and u.unit_type == COMMAND_CENTER
    )
    scvs = sorted(
        (u for u in units if u.alliance == ALLIANCE_SELF and u.unit_type == SCV),
        key=lambda u: _dist(u, cc),
    )
    # Any neutral unit with minerals near our CC; avoids hardcoding the many
    # mineral field unit-type ids.
    minerals = sorted(
        (
            u
            for u in units
            if u.alliance == ALLIANCE_NEUTRAL
            and u.mineral_contents > 0
            and _dist(u, cc) < 12
        ),
        key=lambda u: _dist(u, cc),
    )
    if len(scvs) < len(minerals):
        raise RuntimeError(f"{len(scvs)} SCVs for {len(minerals)} patches")

    # One slot per patch; with the pro split, spare SCVs take a second slot on
    # the closest patches (12 SCVs / 8 patches -> the 4 closest get 2).
    # Center distance ranks patches by walking distance well; a 5x5 footprint
    # gap does not, because the CC's collision shape has cut corners.
    slots = list(minerals)
    if split == "pro":
        slots += minerals[: min(len(scvs) - len(minerals), len(minerals))]

    # Greedy: fill each slot (closest patches first) with the nearest free SCV,
    # roughly what a player's box-split does.
    free = list(scvs)
    assignment = {}
    for m in slots:
        scv = min(free, key=lambda u: _dist(u, m))
        free.remove(scv)
        assignment[scv.tag] = m.tag
    parked = [scv.tag for scv in free]

    # Park spare SCVs on the far side of the CC from the mineral line.
    mx = sum(m.pos.x for m in minerals) / len(minerals)
    my = sum(m.pos.y for m in minerals) / len(minerals)
    dx, dy = cc.pos.x - mx, cc.pos.y - my
    norm = math.hypot(dx, dy) or 1.0
    park_point = (cc.pos.x + 6 * dx / norm, cc.pos.y + 6 * dy / norm)
    return cc, minerals, assignment, parked, park_point


def main(argv):
    del argv
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    run_config = run_configs.get()

    with run_config.start(want_rgb=False) as controller:
        _start_game(run_config, controller)
        obs = controller.observe()
        units = obs.observation.raw_data.units
        cc, minerals, assignment, parked, park_point = _assign_workers(units, FLAGS.split)

        actions = [
            _unit_command(HARVEST_GATHER, [scv_tag], target_tag=mineral_tag)
            for scv_tag, mineral_tag in assignment.items()
        ]
        if parked:
            actions.append(_unit_command(MOVE, parked, point=park_point))
        controller.actions(sc_pb.RequestAction(actions=actions))

        def geom(u):
            return {"tag": u.tag, "x": u.pos.x, "y": u.pos.y, "radius": u.radius}

        mineral_tags = {m.tag for m in minerals}
        last_contents = {m.tag: m.mineral_contents for m in minerals}

        scv_radius = next(u.radius for u in units if u.unit_type == SCV)
        meta = {
            "map": FLAGS.map,
            "start_loop": obs.observation.game_loop,
            "command_center": geom(cc),
            "minerals": [
                dict(geom(m), contents=m.mineral_contents)
                for m in minerals
            ],
            "split": FLAGS.split,
            "scv_radius": scv_radius,
            "assignment": {str(k): v for k, v in assignment.items()},
            "parked": parked,
        }
        with open(os.path.join(FLAGS.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        trace_path = os.path.join(FLAGS.out_dir, "trace.csv")
        minerals_path = os.path.join(FLAGS.out_dir, "minerals.csv")
        holds_path = os.path.join(FLAGS.out_dir, "holds.csv")
        n_holds = 0
        with open(trace_path, "w", newline="") as f, open(minerals_path, "w", newline="") as mf,                 open(holds_path, "w", newline="") as hf:
            w = csv.writer(f)
            mw = csv.writer(mf)
            mw.writerow(["loop", "tag", "contents"])
            hw = csv.writer(hf)
            hw.writerow(["loop", "tag", "reason", "order_target", "assigned"])
            w.writerow(
                [
                    "loop", "tag", "x", "y", "facing", "carrying",
                    "order_ability", "order_target", "collected_minerals",
                ]
            )
            for i in range(FLAGS.loops):
                controller.step(1)
                obs = controller.observe()
                if obs.player_result:
                    print("Game ended early.")
                    break
                o = obs.observation
                collected = o.score.score_details.collected_minerals
                holds = []
                for u in o.raw_data.units:
                    if u.tag in mineral_tags and u.mineral_contents != last_contents[u.tag]:
                        last_contents[u.tag] = u.mineral_contents
                        mw.writerow([o.game_loop, u.tag, u.mineral_contents])
                    if u.alliance != ALLIANCE_SELF or u.unit_type != SCV:
                        continue
                    order = u.orders[0] if u.orders else None
                    carrying = any(b in CARRY_MINERAL_BUFFS for b in u.buff_ids)
                    if FLAGS.hold_patch and u.tag in assignment and not carrying:
                        # Only while empty-handed: a gather command while
                        # carrying would interrupt the return trip.
                        want = assignment[u.tag]
                        if order is None:
                            reason, target = "idle", 0
                        elif order.ability_id == HARVEST_GATHER_SCV and order.target_unit_tag != want:
                            reason, target = "bounce", order.target_unit_tag
                        else:
                            reason = None
                        if reason:
                            holds.append(_unit_command(HARVEST_GATHER, [u.tag], target_tag=want))
                            hw.writerow([o.game_loop, u.tag, reason, target, want])
                    w.writerow(
                        [
                            o.game_loop,
                            u.tag,
                            f"{u.pos.x:.5f}",
                            f"{u.pos.y:.5f}",
                            f"{u.facing:.5f}",
                            int(carrying),
                            order.ability_id if order else 0,
                            order.target_unit_tag if order else 0,
                            collected,
                        ]
                    )
                if holds:
                    controller.actions(sc_pb.RequestAction(actions=holds))
                    n_holds += len(holds)
                if i % 500 == 0:
                    print(f"loop {o.game_loop}  collected={collected}  holds={n_holds}")

        with open(os.path.join(FLAGS.out_dir, "probe.SC2Replay"), "wb") as f:
            f.write(controller.save_replay())
        print(f"Wrote {trace_path}")


if __name__ == "__main__":
    app.run(main)
