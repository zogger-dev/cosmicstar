"""Build the frame cache (worldmodel/frame_cache.py): every extracted 1v1 game with both
player views, --per_game moments each, in all three views, written per split (train,
held_out_games, held_out_map_games) as <out>/<split>/<game id>.fcache, with the
vocabulary in <out>/vocab.json.

Each worker holds one game at a time plus its compressed windows (measure before raising
--workers), so --workers bounds the memory. Games already written are skipped
(rerunnable).

    python scripts/build_frame_cache.py --out data/worldmodel/frame_cache --per_game 80 --workers 4
"""

import argparse
import ctypes
import gc
import json
import multiprocessing as mp
import os
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from cosmicstar.worldmodel import corpus  # noqa: E402
from cosmicstar.worldmodel import frame_cache as fc  # noqa: E402
from cosmicstar.worldmodel import vocab as vocab_mod  # noqa: E402

_V = None


def _init(vocab_path):
    global _V
    _V = vocab_mod.Vocab(json.load(open(vocab_path)))


def _release():
    """Free the finished game now: a GameSet's views reference each other, so without
    this it lives until the cycle collector happens to run and the next game loads on
    top of it; then hand freed arenas back.
    """
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def build_one(job):
    """One game -> its .fcache; returns (game id, moments written, seconds) or (game id,
    error)."""
    try:
        return _build_one(job)
    finally:
        _release()


def _build_one(job):
    import train_direct as td
    from cosmicstar.worldmodel.action_tokens import ActionVocab

    path, out_path, per_game, length, seed = job
    gid = os.path.basename(out_path)[: -len(".fcache")]
    t0 = time.time()
    try:
        gs = td.GameSet(path, None, _V, ActionVocab())
        if gs.obs is None or len(gs.views) != 2:
            return gid, "not a 1v1 with both views"
        n = min(len(gs.obs.g), *(len(x.g) for x in gs.views))
        rng = np.random.RandomState(seed)
        oms = [x.g.owner_map("player", me_slot=x.viewer) for x in gs.views] + [
            gs.obs.g.owner_map("observer")
        ]
        games = list(gs.views) + [gs.obs]
        starts = fc.choose_starts(n, per_game, length, rng)

        def windows():  # one at a time: write_game compresses each and drops its frames
            for t0 in starts:
                ts = range(int(t0), int(t0) + length)
                yield int(t0), [int(gs.obs.g.loops[t]) for t in ts], [
                    [g.g.frame(t, om) for g, om in zip(games, oms)] for t in ts
                ]

        written = fc.write_game(
            out_path,
            windows(),
            dict(
                game=gid,
                frames=int(n),
                step=int(gs.obs.g.step),
                map=gs.obs.g.meta.get("map"),
                slots=[x.viewer for x in gs.views],
            ),
        )
        return gid, written, time.time() - t0
    except Exception as e:  # report and go on: one bad game shouldn't stop the cache
        return gid, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=corpus.GAMES)
    ap.add_argument(
        "--splits", default=os.path.join(corpus.GAMES, "splits", "default.json")
    )
    ap.add_argument("--out", default="data/worldmodel/frame_cache")
    ap.add_argument("--per_game", type=int, default=300, help="windows per game")
    ap.add_argument(
        "--window",
        type=int,
        default=15,
        help=(
            "consecutive frames per window, each in 3 views (15: a moment and the 14"
            " frames after it, 2.5 s at the 4-loop frame step; 1: single moments)"
        ),
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--vocab",
        default="",
        help="an existing vocab.json (default: built from 60 training games)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    split = json.load(open(args.splits))
    views = corpus.views_by_id(args.games)
    os.makedirs(args.out, exist_ok=True)
    vpath = os.path.join(args.out, "vocab.json")
    if not os.path.exists(vpath):
        if args.vocab:
            voc = json.load(open(args.vocab))
        else:
            first = [views[g][0] for g in sorted(split["train"]) if g in views][:60]
            voc = vocab_mod.build(first)
        json.dump(voc, open(vpath, "w"))
    jobs = []
    for s in ("train", "held_out_games", "held_out_map_games"):
        os.makedirs(os.path.join(args.out, s), exist_ok=True)
        for i, g in enumerate(sorted(split.get(s, []))):
            if g not in views:
                continue
            out = os.path.join(args.out, s, f"{g}.fcache")
            if not os.path.exists(out):
                jobs.append(
                    (
                        views[g][0],
                        out,
                        args.per_game,
                        args.window,
                        args.seed * 100003 + zlib.crc32(g.encode()) % 100003,
                    )
                )
    print(
        f"{len(jobs)} games to build, {args.workers} workers, {args.per_game} windows"
        f" of {args.window} frames each -> {args.out}",
        flush=True,
    )
    t0, done, moments = time.time(), 0, 0
    with mp.get_context("spawn").Pool(
        args.workers, initializer=_init, initargs=(vpath,), maxtasksperchild=8
    ) as pool:
        for r in pool.imap_unordered(build_one, jobs):
            done += 1
            if isinstance(r[1], int):
                moments += r[1]
            else:
                print(f"skipped {r[0]}: {r[1]}", flush=True)
            if done % 20 == 0 or done == len(jobs):
                el = time.time() - t0
                print(
                    f"{done}/{len(jobs)} games, {moments} windows"
                    f" ({3 * args.window * moments} view-frames), {el:.0f}s,"
                    f" ~{el / done * (len(jobs) - done):.0f}s left",
                    flush=True,
                )
    print("done", flush=True)


if __name__ == "__main__":
    main()
