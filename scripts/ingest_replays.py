"""Sorts replays into the corpus tree (worldmodel/corpus.py):

    data/replays/<format>/<build>/<map>/<id>.SC2Replay + <id>.json

from loose files, directories (walked) and archives (.zip / .rar / .7z, unpacked with 7z
into a temporary folder; the archive itself stays where it is). <id> is the content
hash, so a game that arrives twice is one game with both sources in its sidecar. Loose
files are moved (--copy keeps them). Unreadable replays stay where they are and are
listed at the end.

    python scripts/ingest_replays.py data/replays/pro-1v1
    python scripts/ingest_replays.py "data/replays/TSL 9 Replay Pack.zip" --source "TSL 9"
"""

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from cosmicstar.worldmodel import corpus  # noqa: E402

ARCHIVES = (".zip", ".rar", ".7z")


class Ingest:
    def __init__(self, root, copy=False):
        self.root, self.copy = root, copy
        self.counts = collections.Counter()
        self.by_kind = collections.Counter()
        self.unreadable = []

    def file(self, path, pack, original):
        try:
            data = open(path, "rb").read()
            gid = corpus.game_id(data)
            info = corpus.replay_info(path)
        except Exception as ex:
            self.unreadable.append((path, repr(ex)[:120]))
            self.counts["unreadable"] += 1
            return
        rel = corpus.rel_dir(info["format"], info["base_build"], info["map_hash"])
        dest = os.path.join(self.root, rel, gid + corpus.EXT)
        side = os.path.join(self.root, rel, gid + ".json")
        source = dict(pack=pack, file=original)
        if os.path.exists(side):
            d = json.load(open(side))
            if not any(
                s.get("pack") == pack and s.get("file") == original
                for s in d["sources"]
            ):
                d["sources"].append(
                    dict(
                        source,
                        ingested=corpus.replay_sidecar(info, gid, "", source)[
                            "sources"
                        ][0]["ingested"],
                    )
                )
                json.dump(d, open(side, "w"), indent=1)
            if not self.copy and os.path.abspath(path) != os.path.abspath(dest):
                os.remove(path)  # the duplicate
            self.counts["duplicate"] += 1
            return
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if self.copy:
            shutil.copy2(path, dest + ".tmp")
        else:
            shutil.move(path, dest + ".tmp")
        os.replace(dest + ".tmp", dest)
        json.dump(
            corpus.replay_sidecar(
                info, gid, os.path.join(rel, gid + corpus.EXT), source
            ),
            open(side, "w"),
            indent=1,
        )
        self.counts["new"] += 1
        self.by_kind[(info["format"], info["base_build"])] += 1

    def directory(self, top, pack):
        for dirpath, dirnames, filenames in os.walk(top):
            dirnames.sort()
            for name in sorted(filenames):
                p = os.path.join(dirpath, name)
                if name.lower().endswith(corpus.EXT.lower()):
                    self.file(p, pack, os.path.relpath(p, top))
                elif name.lower().endswith(ARCHIVES):
                    self.archive(p, pack + "/" + name)

    def archive(self, path, pack):
        tmp = tempfile.mkdtemp(
            prefix="ingest_",
            dir=(
                os.path.join(self.root, ".ingest")
                if os.path.isdir(os.path.join(self.root, ".ingest"))
                else None
            ),
        )
        try:
            r = subprocess.run(
                ["7z", "x", "-y", "-o" + tmp, path], capture_output=True, text=True
            )
            if r.returncode != 0:
                self.unreadable.append((path, "7z: " + r.stderr.strip()[-120:]))
                self.counts["unreadable archives"] += 1
                return
            copy, self.copy = (
                self.copy,
                False,
            )  # the unpacked copies are moved into place
            self.directory(tmp, pack)
            self.copy = copy
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def path(self, p, pack=None):
        if os.path.isdir(p):
            self.directory(p, pack or os.path.basename(os.path.normpath(p)))
        elif p.lower().endswith(ARCHIVES):
            self.archive(p, pack or os.path.basename(p))
        elif p.lower().endswith(corpus.EXT.lower()):
            self.file(
                p,
                pack or os.path.basename(os.path.dirname(os.path.abspath(p))),
                os.path.basename(p),
            )
        else:
            print(f"skipped (not a replay, directory or archive): {p}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="replay files, directories or archives")
    ap.add_argument("--root", default=corpus.REPLAYS)
    ap.add_argument(
        "--source",
        help="pack label for the sidecars (default: the directory's or archive's name)",
    )
    ap.add_argument(
        "--copy", action="store_true", help="copy loose files instead of moving them"
    )
    ap.add_argument(
        "--no_index", action="store_true", help="don't rebuild index.json afterwards"
    )
    args = ap.parse_args()
    os.makedirs(os.path.join(args.root, ".ingest"), exist_ok=True)
    ing = Ingest(args.root, args.copy)
    for p in args.inputs:
        ing.path(p, args.source)
    shutil.rmtree(os.path.join(args.root, ".ingest"), ignore_errors=True)
    print(json.dumps(dict(ing.counts)))
    for (fmt, build), n in sorted(ing.by_kind.items()):
        print(f"  {fmt} {build}: {n}")
    for p, why in ing.unreadable[:20]:
        print(f"  unreadable: {p}: {why}", file=sys.stderr)
    if not args.no_index:
        index = corpus.build_index(args.root)
        print(
            f"index: {len(index)} games -> {os.path.join(args.root, corpus.INDEX)},"
            f" maps -> {os.path.join(args.root, corpus.MAPS)}"
        )


if __name__ == "__main__":
    main()
