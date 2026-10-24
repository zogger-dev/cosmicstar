"""Exports each ruleset's balance data with the StarCraft II Editor (see worldmodel/README.md).

For every ruleset in <rules>/index.json, one of its replays is exported (trying up to --tries
replays, since a replay can fail on its own map) with

    SC2Editor_x64.exe -exportreplay <replay> -exportreplaydir <out>/<ruleset>

one at a time (the Editor fetches the replay's dependencies itself). On Linux the Editor runs
under Wine/Proton like the game (cosmicstar/envs/wine_run_config.py; the output matched a
Windows export file for file, apart from the locales the install lacks). The outcome, including
the Editor's error when it can't export, goes to <out>/status.json; replays of rulesets that
failed are left out of the dataset (extract_states.build_manifest).

    python scripts/export_balance.py
"""

import argparse
import collections
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_states as ex  # noqa: E402  (replay_info: a replay's ruleset)

from cosmicstar.envs import wine_run_config as wrc  # noqa: E402

WINDOWS = sys.platform == "win32"
SC2_DIR = wrc.sc2_install_dir() or r"C:\Program Files (x86)\StarCraft II"
EDITOR = os.path.join(SC2_DIR, "Support64", "SC2Editor_x64.exe")
LOGS = os.path.join(wrc.documents_dir(), "StarCraft II", "EditorLogs")
REPLAYS = (r"C:\Users\prade\Documents\StarCraft II\Accounts\52583725\1-S2-1-1544307\Replays\Multiplayer"
           if WINDOWS else "data/replays/my-recent-games")


def editor_command(replay, out):
    """argv and env for one export; under Wine the Editor gets Windows paths."""
    if WINDOWS:
        return [EDITOR, "-exportreplay", replay, "-exportreplaydir", out], None
    prefix = wrc.wine_prefix(SC2_DIR)
    return wrc.wine_command(EDITOR, ["-exportreplay", wrc.windows_path(replay, prefix),
                                     "-exportreplaydir", wrc.windows_path(out, prefix)], prefix)


def export(replay, out, timeout=1200):
    """Runs one export; returns (ok, error message)."""
    replay, out = os.path.abspath(replay), os.path.abspath(out)  # the Editor runs from its own directory
    os.makedirs(out, exist_ok=True)
    started = time.time()
    cmd, env = editor_command(replay, out)
    try:
        subprocess.run(cmd, env=env, cwd=os.path.dirname(EDITOR), timeout=timeout,
                       stdout=subprocess.DEVNULL if not WINDOWS else None, stderr=subprocess.STDOUT if not WINDOWS else None)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    if glob.glob(os.path.join(out, "*.xml")):
        return True, ""
    logs = [p for p in glob.glob(os.path.join(LOGS, "*ExportReplayErrors.txt")) if os.path.getmtime(p) >= started - 1]
    msg = open(max(logs, key=os.path.getmtime), encoding="utf-8-sig").read().strip() if logs else "no output"
    os.rmdir(out) if not os.listdir(out) else None
    return False, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default="data/worldmodel/rules")
    ap.add_argument("--replays", default=REPLAYS)
    ap.add_argument("--out", default="data/worldmodel/balance")
    ap.add_argument("--redo", action="store_true", help="re-export rulesets already exported")
    ap.add_argument("--tries", type=int, default=5, help="replays to try per ruleset")
    args = ap.parse_args()
    index = json.load(open(os.path.join(args.rules, "index.json")))
    status_file = os.path.join(args.out, "status.json")
    status = json.load(open(status_file)) if os.path.exists(status_file) else {}
    todo = [k for k in sorted(index) if args.redo or not (status.get(k, {}).get("ok")
                                                          and glob.glob(os.path.join(args.out, k, "*.xml")))]
    # a replay can fail on its own map, so try replays of several maps per ruleset
    replays = collections.defaultdict(list)
    if todo:
        for path in sorted(glob.glob(os.path.join(args.replays, "*.SC2Replay"))):
            try:
                info = ex.replay_info(path)
            except Exception:
                continue
            key = info["ruleset"]
            if key in todo and info["map_hash"] not in {m for m, _ in replays[key]}:
                replays[key].append((info["map_hash"], path))  # one replay per map
    for key in todo:
        v = index[key]
        out = os.path.join(args.out, key)
        tried = []
        ok = False
        for _, replay in replays[key][:args.tries]:
            t = time.time()
            ok, err = export(replay, out)
            tried.append(dict(replay=os.path.basename(replay), error=err))
            print(f"{key} ({v['games']} games) {os.path.basename(replay)}: {'ok' if ok else 'FAILED: ' + err} "
                  f"[{time.time() - t:.0f}s]", flush=True)
            if ok:
                break
        status[key] = dict(ok=ok, tried=tried, base_build=v["base_build"], games=v["games"])
        json.dump(status, open(status_file, "w"), indent=1)


if __name__ == "__main__":
    main()
