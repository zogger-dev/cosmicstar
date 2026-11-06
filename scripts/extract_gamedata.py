"""Extracts the game's data XMLs (GameData/*.xml of every mod) from the installed CASC
storage, per build, through CascLib (envs/casc.py).

The Editor's balance export leaves out what upgrades do (a Marauder's +1 damage and +1
vs Armored per weapons level, adrenal glands' attack period, ...); those live in
UpgradeData.xml and the catalogs it references (WeaponData, EffectData, UnitData,
BehaviorData), inside the mods of the game: Core, Liberty, Swarm, Void and the
multiplayer balance mods layered on top. This writes them to
<out>/<build>/<mod>/GameData/<file>.xml, mirroring the storage paths.

    python scripts/extract_gamedata.py                 # every installed build
    python scripts/extract_gamedata.py --builds 98310 --list   # just list what matches
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from cosmicstar.envs import tact, wine_run_config  # noqa: E402
from cosmicstar.envs.casc import Storage  # noqa: E402


def build_key(install, build):
    """BuildConfig hash of a base build in an install's .build.info (None: the active
    row)."""
    _, rows = tact.read_build_info(install)
    for r in rows:
        if r.get("Version", "").endswith(f".{build}"):
            return r["Build Key"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--builds",
        default="",
        help="comma-separated base builds (default: every installed one)",
    )
    ap.add_argument(
        "--mask",
        default=r"mods\*gamedata\*.xml",
        help="CascLib name mask (names are lower-case, backslashes)",
    )
    ap.add_argument(
        "--mods",
        default="core,liberty,swarm,void,voidmulti",
        help=(
            "comma-separated mod names to keep (the ladder dependency chain, in load"
            " order); empty = all"
        ),
    )
    ap.add_argument("--out", default="data/worldmodel/gamedata")
    ap.add_argument("--list", action="store_true", help="list matching files and exit")
    args = ap.parse_args()
    builds = [int(b) for b in args.builds.split(",") if b] or sorted(
        wine_run_config.installed_builds()
    )
    for build in builds:
        install = wine_run_config.install_for_build(build)
        key = build_key(install, build)
        if key is None:
            print(
                f"build {build}: not in {install}/.build.info, skipped (install it with"
                " cosmicstar.envs.tact first)",
                flush=True,
            )
            continue
        print(f"build {build}: {install} (build key {key})", flush=True)
        with Storage(install, build_key=key) as st:
            names = st.find(args.mask)
            if args.mods:
                keep = {
                    m.strip().lower() + ".sc2mod"
                    for m in args.mods.split(",")
                    if m.strip()
                }
                names = [n for n in names if n.split("\\")[1] in keep]
            print(
                f"  {len(names)} files match {args.mask}"
                + (f" in mods {args.mods}" if args.mods else ""),
                flush=True,
            )
            for n in names[:5] + (["..."] if len(names) > 5 else []):
                print("   ", n)
            if args.list:
                continue
            n_bytes = 0
            for name in names:
                rel = name.replace("\\", "/").lstrip("/")
                dst = os.path.join(args.out, str(build), rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                data = st.read(name)
                with open(dst, "wb") as f:
                    f.write(data)
                n_bytes += len(data)
            print(
                f"  wrote {len(names)} files, {n_bytes / 1e6:.1f} MB ->"
                f" {os.path.join(args.out, str(build))}",
                flush=True,
            )


if __name__ == "__main__":
    main()
