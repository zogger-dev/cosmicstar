"""Builds the per-build ability table: replay (link, cmd) -> API ability id.

The replay names abilities by (link, cmd): the ability's position in that build's catalog
and the button within it. The engine reports a player's commands in API ids. Playing a
replay while observing a player gives both views of the same commands; lining them up
gives the table. Only (game, player) pairs in --picks are played (chosen to cover every
(build, link, cmd) used in our games).

Votes accumulate in --votes (resumable); the table (majority vote per key, with its share)
is written to --table.

    python scripts/build_ability_table.py --picks data/worldmodel/actions/translation_playbacks.json
"""

import argparse
import collections
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import extract_states as ex  # noqa: E402
from cosmicstar.worldmodel import replay_actions as ra  # noqa: E402

STEP = 8
ENGINE_DIR = "data/worldmodel/engine_actions"  # the engine's raw commands per playback, for offline re-matching
REPORTED = (ra.CMD, ra.CMD_POINT, ra.CMD_UNIT, ra.REPEAT)  # replay events the engine reports as commands


def engine_commands(path, slot):
    """[(loop, ability_id, x, y, engine_target_tag)] the engine reports for one player."""
    from absl import flags
    from s2clientprotocol import sc2api_pb2 as sc_pb
    flags.FLAGS(sys.argv[:1])
    info = ex.replay_info(path)
    run_config, current = ex.run_config_for(info)
    out = []
    with run_config.start(want_rgb=False) as c:
        req = sc_pb.RequestStartReplay(replay_data=open(path, "rb").read(), observed_player_id=slot,
                                       options=sc_pb.InterfaceOptions(raw=True))
        if current:
            c.start_replay(req)
        else:
            req.map_data = open(ex.map_path(info["map_hash"]), "rb").read()
            with ex.launch_lock():
                c.start_replay(req)
        loop, stalled = 0, 0
        from pysc2.lib import protocol
        while loop < info["duration_loops"]:
            if c.status == protocol.Status.ended:  # the replay is over (e.g. the player left)
                break
            c.step(STEP)
            o = c.observe()
            prev, loop = loop, o.observation.game_loop
            stalled = stalled + 1 if loop <= prev else 0
            if stalled >= 5 or o.player_result and loop >= info["duration_loops"] - 4 * STEP:
                break
            for a in o.actions:
                if not a.HasField("action_raw") or not a.action_raw.HasField("unit_command"):
                    continue
                uc = a.action_raw.unit_command
                pt = uc.target_world_space_pos if uc.HasField("target_world_space_pos") else None
                out.append((loop, uc.ability_id, pt.x if pt else np.nan, pt.y if pt else np.nan,
                            uc.target_unit_tag if uc.HasField("target_unit_tag") else 0))
    return out


def same_target(d, i, e):
    t = d["target"][i]
    if t == ra.TARGET_POINT:
        return not np.isnan(e[2]) and abs(e[2] - d["tx"][i]) < 0.5 and abs(e[3] - d["ty"][i]) < 0.5
    if t == ra.TARGET_UNIT:
        return e[4] == (1 << 32) + int(d["target_tag"][i])
    if t == ra.TARGET_NONE:
        return np.isnan(e[2]) and e[4] == 0
    return True  # snapshot: no tag to compare


def match(d, slot, eng):
    """Pairs the replay's reported commands with the engine's, in order, each engine action
    used once. Within one observation both lists are in issue order, so a target-less
    command (train, research) pairs with the next unused engine action rather than the
    first one that happens to have no target either."""
    by_obs = collections.defaultdict(list)
    for e in eng:
        by_obs[int(e[0])].append(list(e) + [False])  # last field: used
    votes, matched, cmds = collections.Counter(), 0, 0
    last_api = None
    rows = np.where((d["player"] == slot) & np.isin(d["kind"], REPORTED))[0]
    for i in rows:
        obs = (int(d["loop"][i]) // STEP + 1) * STEP
        cands = by_obs.get(obs, []) + by_obs.get(obs + STEP, [])
        kind = d["kind"][i]
        is_cmd = kind == ra.CMD and d["abil_link"][i] >= 0
        cmds += is_cmd
        hit = None
        for e in cands:
            if e[5]:
                continue
            if kind == ra.REPEAT:
                ok = last_api is not None and e[1] == last_api
            else:
                ok = same_target(d, i, e)
            if ok:
                hit = e
                break
        if hit is None:
            continue
        hit[5] = True
        if kind == ra.CMD:
            last_api = hit[1]
        if is_cmd and hit[1] != 1:  # 1 = Smart: never a translated ability
            votes[(int(d["abil_link"][i]), int(d["abil_cmd"][i]), int(hit[1]))] += 1
            matched += 1
    return votes, matched, cmds


def votes_for(job):
    name, slot, path, actions_file, build = job
    raw = os.path.join(ENGINE_DIR, f"{os.path.splitext(name)[0]}_p{slot}.npz")
    try:
        if os.path.exists(raw):
            eng = [tuple(r) for r in np.load(raw)["eng"]]
        else:
            eng = engine_commands(path, slot)
            np.savez_compressed(raw, eng=np.asarray(eng, np.float64).reshape(-1, 5))
    except Exception as e:
        return name, slot, build, None, repr(e)[:150]
    votes, matched, cmds = match(np.load(actions_file), slot, eng)
    return name, slot, build, dict((f"{k[0]}:{k[1]}:{k[2]}", v) for k, v in votes.items()), f"{matched}/{cmds}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", default="data/worldmodel/actions/translation_playbacks.json")
    ap.add_argument("--manifest", default="data/worldmodel/actions/manifest.json")
    ap.add_argument("--votes", default="data/worldmodel/ability_votes.json")
    ap.add_argument("--table", default="data/worldmodel/ability_table.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(ENGINE_DIR, exist_ok=True)
    man = {os.path.basename(j["out"]): j for j in json.load(open(args.manifest))}
    store = json.load(open(args.votes)) if os.path.exists(args.votes) else {"done": [], "votes": {}}
    done = {tuple(x) for x in store["done"]}
    jobs = collections.defaultdict(list)
    for name, slot in json.load(open(args.picks)):
        if (name, slot) in done or name not in man:
            continue
        j = man[name]
        jobs[j["base_build"]].append((name, slot, j["path"], j["out"], j["base_build"]))
    print(f"{sum(map(len, jobs.values()))} playbacks to run", flush=True)

    def merge(res):
        for name, slot, build, votes, status in res:
            if votes is None:
                print(f"  fail {name} p{slot}: {status}", flush=True)
                continue
            for k, v in votes.items():
                key = f"{build}:{k}"
                store["votes"][key] = store["votes"].get(key, 0) + v
            store["done"].append([name, slot])
            print(f"  {build} {name} p{slot}: matched {status}", flush=True)
        json.dump(store, open(args.votes, "w"))

    for build in sorted(jobs):  # one build at a time; a solo warm-up first
        js = jobs[build]
        print(f"build {build}: {len(js)} playbacks", flush=True)
        with Pool(1) as pool:
            merge(pool.map(votes_for, js[:1]))
        if len(js) > 1:
            with Pool(min(args.workers, len(js) - 1)) as pool:
                merge(pool.map(votes_for, js[1:]))

    # table: majority API id per (build, link, cmd), with its share of the votes
    per_key = collections.defaultdict(collections.Counter)
    for key, v in store["votes"].items():
        b, l, c, api = key.split(":")
        per_key[f"{b}:{l}:{c}"][int(api)] += v
    table = {k: dict(api=cnt.most_common(1)[0][0], share=round(cnt.most_common(1)[0][1] / sum(cnt.values()), 3),
                     votes=sum(cnt.values())) for k, cnt in per_key.items()}
    json.dump(table, open(args.table, "w"), indent=0)
    print(f"table: {len(table)} keys -> {args.table}")


if __name__ == "__main__":
    main()
