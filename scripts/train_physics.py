"""Trains the physics model (worldmodel/physics.py, DESIGN_physics.md) on the Arrow
store.

    python scripts/train_physics.py --seqs 6 --frames 56 --accum 4 --steps 40000
    python scripts/train_physics.py --measure      # peak memory at the largest windows, a few steps, nothing saved
    python scripts/train_physics.py --resume RUN/model_step2000.pt --run RUN              # carries on from a checkpoint
    python scripts/train_physics.py --resume RUN/model_step2000.pt --run RUN --eval_only  # the full evaluation of a checkpoint, then exits

Quick validation every --val_every updates goes to the log. The full evaluation pauses
training after each checkpoint, and goes to the log and RUN/eval.jsonl.
"""

import argparse
import collections
import gc
import glob
import json
import os
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# the cap below needs segments that grow; PyTorch 2.15's pool of compile workers crashes
# on this machine, and one thread avoids it
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import numpy as np
import torch
import torch.multiprocessing as mp

from cosmicstar.worldmodel import epochs, physics, physics_data, statefile
from cosmicstar.worldmodel.action_tokens import ActionVocab
from cosmicstar.worldmodel.vocab import Vocab

TOKENS_PER_WINDOW = 259  # beyond its rows: 3 globals and 256 cells
EXCLUDED = (
    "982fc2368b80adb3",
    "1ff02cd0dd2751fa",
    "c50626c6c77288c6",
    "5bf221dbd7045ab1",
    "d2de5d4aef655894",
)


def map_of(base):
    """A game's map: its folder in the games tree, the map's hash."""
    return os.path.basename(os.path.dirname(base))


def patch_of(base):
    """A game's patch, e.g. 5.0.14, from its replay's metadata."""
    return json.loads(str(statefile.load(f"{base}.p1.cst")["meta"])).get("game_version")


def of_patch(bases, patch):
    """The games of one patch (patch empty: all). Rules change between patches and the
    model has no rules input yet, so training mixing two patches would learn a blend of
    both: Ghost life, Spire cost and the like."""
    return {g: b for g, b in bases.items() if not patch or patch_of(b) == patch}


class Games:
    """Open games, least recently used closed. Reads windows on worker threads."""

    def __init__(self, bases, vocab, avocab, keep=48, map_ids=None):
        self.bases, self.vocab, self.avocab, self.keep, self.map_ids = (
            bases,
            vocab,
            avocab,
            keep,
            map_ids or {},
        )
        self.open, self.lock = collections.OrderedDict(), threading.Lock()

    def get(self, gid):
        with self.lock:
            g = self.open.get(gid)
            if g is not None:
                self.open.move_to_end(gid)
                return g
        g = physics_data.Game(self.bases[gid], self.vocab)
        g._load_actions(self.avocab)
        g.map_id = self.map_ids.get(map_of(self.bases[gid]), 0)
        with self.lock:
            self.open[gid] = g
            while len(self.open) > self.keep:
                self.open.popitem(last=False)
        return g

    def read(self, gid, t0, L, swap):
        g = self.get(gid)
        t0 = min(t0, g.F - L - 1) if t0 >= 0 else np.random.randint(0, max(1, g.F - L))
        return g.window(t0, L, swap=swap)


def warm_compile(model, games, ids, L):
    """Compile on two windows. Compiling holds several GB until the next collection, and
    a full pass has no room for that. No optimizer step: this changes no weights."""
    rng = np.random.default_rng()
    b = physics_data.collate(
        [games.read(ids[rng.integers(len(ids))], -1, L, False) for _ in range(2)],
        games.vocab,
    )
    b = {k: v.cuda() for k, v in b.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = model(b)
    loss.backward()
    model.zero_grad(set_to_none=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(b)
    del b, loss
    gc.collect()
    # what stays, compiled code above all, is left out of every later collection: 0.3 s
    # a pass otherwise
    gc.freeze()


def _read_passes(
    q, bases, vocab_path, map_ids, seqs, L, workers, chosen, plan_kw, start
):
    """The reader process: the passes of the epoch plan (epochs.Plan) into q, from pass
    `start` counted from the first epoch, so a resumed run carries on where it stopped.
    chosen: a fixed list of (game, frame) instead, cycled seqs at a time.
    """
    vocab = Vocab(json.load(open(vocab_path)))
    games = Games(bases, vocab, ActionVocab(), map_ids=map_ids)
    pool = ThreadPoolExecutor(workers)
    if chosen:
        i, rng = 0, np.random.default_rng()
        while True:
            pick = [chosen[(i + k) % len(chosen)] for k in range(seqs)]
            i += seqs
            q.put(
                physics_data.collate(
                    list(
                        pool.map(
                            lambda p: games.read(p[0], p[1], L, bool(rng.integers(2))),
                            pick,
                        )
                    ),
                    vocab,
                )
            )
    plan = epochs.Plan(
        bases, {g: map_of(b) for g, b in bases.items()}, L, seqs=seqs, **plan_kw
    )
    epoch, k = plan.locate(start)
    while True:
        passes = plan.passes(epoch)
        print(
            json.dumps(
                dict(
                    epoch=epoch,
                    passes=len(passes),
                    windows=sum(map(len, passes)),
                    from_pass=k,
                    buckets=len(plan.top),
                )
            ),
            flush=True,
        )
        for wins in passes[k:]:
            q.put(
                physics_data.collate(
                    list(pool.map(lambda p: games.read(p[0], p[1], L, p[2]), wins)),
                    vocab,
                )
            )
        epoch, k = epoch + 1, 0


def batches(
    bases,
    vocab_path,
    map_ids,
    seqs,
    L,
    workers,
    plan_kw=None,
    start=0,
    prefetch=4,
    chosen=None,
):
    """Passes of windows from a process of their own: reading on threads here took the
    interpreter from the training thread, about 1.1 s of idle GPU each time 16 windows
    were read. A thread here holds passes ready, since the reader hands one over late
    while its own threads are reading. chosen: a fixed list of (game, frame) to cycle
    instead of random ones.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue(2)
    reader = ctx.Process(
        target=_read_passes,
        args=(
            q,
            bases,
            vocab_path,
            map_ids,
            seqs,
            L,
            workers,
            chosen,
            plan_kw or {},
            start,
        ),
        daemon=True,
    )
    reader.start()
    ready = queue.Queue(prefetch)

    def pull():
        while True:
            try:
                ready.put(q.get(timeout=5))
            except queue.Empty:
                if not reader.is_alive():
                    ready.put(
                        RuntimeError(
                            f"the reader process died, exit code {reader.exitcode}"
                        )
                    )
                    return

    threading.Thread(target=pull, daemon=True).start()
    while True:
        b = ready.get()
        if isinstance(b, Exception):
            raise b
        yield b


def largest_windows(bases, L, n):
    """(game, frame) of the windows with the most rows alive across all games: the
    memory worst case."""
    best = []
    for gid, base in bases.items():
        ix = np.load(base + ".index.npz")
        F_ = int(ix["F"])
        d = np.zeros(F_ + 2, np.int64)
        for s in physics_data.SOURCES:
            np.add.at(d, ix[f"first{s}"], 1)
            np.add.at(d, ix[f"last{s}"] + 1, -1)
        c = np.cumsum(d)[:F_]
        t = int(np.argmax(c))
        best.append((int(c[t]), gid, max(0, min(t - L // 2, F_ - L - 1))))
    best.sort(reverse=True)
    return [(g, t) for _, g, t in best[:n]], best[:n]


def fixed_windows(bases, n, L, seed=0):
    """n (game, frame, swap), the same every call, so numbers compare across updates.
    Games taken in turn from a shuffled list."""
    rng = np.random.default_rng(seed)
    ids = [sorted(bases)[i] for i in rng.permutation(len(bases))]
    out = []
    for i in range(n):
        gid = ids[i % len(ids)]
        F_ = int(np.load(bases[gid] + ".index.npz")["F"])
        out.append((gid, int(rng.integers(0, max(1, F_ - L - 1))), bool(i % 2)))
    return out


@torch.no_grad()
def evaluate(model, games, windows, seqs, L, workers):
    """Every loss over fixed windows, and position error by kind of row. Sampling inside
    the losses is seeded too, and training's random state is left as it was."""
    grad_ckpt, model.enc.grad_ckpt = model.enc.grad_ckpt, False
    model.record_pos = True
    logs, pos, n = (
        collections.Counter(),
        collections.defaultdict(list),
        0,
    )
    with ThreadPoolExecutor(workers) as pool, torch.random.fork_rng(devices=[0]):
        torch.manual_seed(0)
        for i in range(0, len(windows), seqs):
            ws = list(
                pool.map(
                    lambda p: games.read(p[0], p[1], L, p[2]), windows[i : i + seqs]
                )
            )
            b = {
                k: v.cuda(non_blocking=True)
                for k, v in physics_data.collate(ws, games.vocab).items()
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, lg = model(b)
            for k, v in lg.items():
                logs[k] += float(v)
            for k, v in model.pos_eval.items():
                pos[k].append(v)
            n += 1
    model.enc.grad_ckpt, model.record_pos = grad_ckpt, False
    out = {k: round(v / n, 4) for k, v in logs.items()}
    out["position"] = {k: position_stats(v) for k, v in pos.items()}
    return out


def position_stats(parts):
    """Mean, root mean square and 95th and 99th percentiles of the model's and copy's
    per-row errors, map units per loop."""
    stats = {"rows": int(sum(len(e) for e, _ in parts))}
    for name, i in (("model", 0), ("copy", 1)):
        e = np.concatenate([p[i] for p in parts]).astype(np.float64)
        if not len(e):
            e = np.zeros(1)
        p95, p99 = np.percentile(e, [95, 99])
        stats[name] = round(float(e.mean()), 5)
        stats[name + "_rms"] = round(float(np.sqrt((e**2).mean())), 5)
        stats[name + "_p95"] = round(float(p95), 5)
        stats[name + "_p99"] = round(float(p99), 5)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data/games_v3")
    ap.add_argument("--vocab", default="data/worldmodel/frame_cache_seq/vocab.json")
    ap.add_argument(
        "--patch",
        default="5.0.14",
        help="train and validate on one patch's games (empty: every patch)",
    )
    ap.add_argument("--split", default="data/games/splits/default.json")
    ap.add_argument("--seqs", type=int, default=16, help="most windows in a pass")
    ap.add_argument(
        "--tokens",
        type=int,
        default=3900,
        help="budget of a pass: windows x (most rows + 259)",
    )
    ap.add_argument(
        "--bucket",
        type=int,
        default=32,
        help=(
            "a pass holds windows whose row counts fall in one bucket of this width, so"
            " it pads little"
        ),
    )
    ap.add_argument(
        "--min_bucket_games",
        type=int,
        default=40,
        help=(
            "the thin ends of the size range join their neighbours until a bucket has"
            " windows from this many games"
        ),
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=14,
        help=(
            "an epoch starts a window every this many frames of each game; this many"
            " epochs start one at every frame"
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the epoch plan's seed: offsets, order and swaps",
    )
    ap.add_argument("--eval_seqs", type=int, default=4)
    ap.add_argument("--frames", type=int, default=56)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument(
        "--hours",
        type=float,
        default=0.0,
        help="stop when this long has passed since launch, with a checkpoint and the"
        " full evaluation; 0: run all --steps",
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--warmup", type=int, default=500, help="updates of a linear rise to --lr"
    )
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument(
        "--no_demean",
        action="store_true",
        help="Barlow Twins correlations without removing each row's mean over time",
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--measure_steps", type=int, default=4)
    ap.add_argument(
        "--typical",
        action="store_true",
        help="--measure on random windows instead of the largest",
    )
    ap.add_argument(
        "--min_map_games",
        type=int,
        default=5,
        help=(
            "a map with fewer training games is left out: too little to learn its"
            " tokens"
        ),
    )
    ap.add_argument("--blocks", type=int, default=6, help="spatial-temporal blocks")
    ap.add_argument("--run", default="")
    ap.add_argument(
        "--gpu_gb",
        type=float,
        default=29.0,
        help=(
            "most GPU memory PyTorch may hold, cache included, so the desktop keeps the"
            " rest"
        ),
    )
    ap.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="checkpoint and full evaluation, in updates",
    )
    ap.add_argument(
        "--val_every", type=int, default=25, help="quick validation, in updates"
    )
    ap.add_argument(
        "--val_windows",
        type=int,
        default=6,
        help="windows per set in a quick validation",
    )
    ap.add_argument(
        "--eval_windows",
        type=int,
        default=144,
        help="windows per set in a full evaluation",
    )
    ap.add_argument("--resume", default="", help="checkpoint to carry on from")
    ap.add_argument(
        "--eval_only",
        action="store_true",
        help="the full evaluation of --resume, then exit",
    )
    args = ap.parse_args()
    t_start = time.time()  # --hours counts from here

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(
            min(
                1.0,
                args.gpu_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory,
            )
        )
    vocab, avocab = Vocab(json.load(open(args.vocab))), ActionVocab()
    split = json.load(open(args.split))
    train_ids = set(split["train"]) - set(EXCLUDED)
    bases = of_patch(physics_data.base_paths(args.games, train_ids), args.patch)
    # a learned table per map: only maps with games enough to learn one
    per_map = collections.Counter(map_of(b) for b in bases.values())
    map_ids = {
        m: i
        for i, m in enumerate(
            sorted(m for m, n in per_map.items() if n >= args.min_map_games)
        )
    }
    bases = {g: b for g, b in bases.items() if map_of(b) in map_ids}
    ids = sorted(bases)
    print(
        f"{len(ids)} training games of patch {args.patch or 'any'} on {len(map_ids)}"
        f" maps, left out {sum(n for m, n in per_map.items() if m not in map_ids)} on"
        f" {len(per_map) - len(map_ids)}",
        flush=True,
    )
    model = physics.Physics(
        vocab,
        avocab,
        args.d,
        demean=not args.no_demean,
        moving_neutrals=physics_data.MOVING_NEUTRALS,
        blocks=args.blocks,
        n_maps=len(map_ids),
    ).cuda()
    model.enc.grad_ckpt = args.ckpt
    if not args.no_compile:
        model.enc.compile_parts()
        model.teacher.compile_parts()
    print(
        f"{sum(p.numel() for n, p in model.named_parameters() if not n.startswith('teacher')) / 1e6:.1f}M"
        " parameters",
        flush=True,
    )
    games = Games(bases, vocab, avocab, map_ids=map_ids)
    val_bases = {
        "unseen_games": {
            g: b
            for g, b in of_patch(
                physics_data.base_paths(
                    args.games, set(split["held_out_games"]) - set(EXCLUDED)
                ),
                args.patch,
            ).items()
            if map_of(b) in map_ids
        }
    }
    quick = {
        k: fixed_windows(v, args.val_windows, args.frames) for k, v in val_bases.items()
    }
    full = {
        k: fixed_windows(v, args.eval_windows, args.frames)
        for k, v in {"train": bases, **val_bases}.items()
    }
    all_bases = {k: v for d in (bases, *val_bases.values()) for k, v in d.items()}
    val_games = Games(
        all_bases, vocab, avocab, keep=2 * args.val_windows, map_ids=map_ids
    )
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()  # which code these weights go with
    run = (
        args.run
        or f"data/worldmodel/runs/physics_{commit}_{time.strftime('%Y%m%d_%H%M%S')}"
    )

    def full_eval(at):
        t = time.time()
        for k, w in full.items():
            r = dict(
                eval=k,
                step=at,
                windows=len(w),
                **evaluate(
                    model,
                    Games(all_bases, vocab, avocab, keep=8, map_ids=map_ids),
                    w,
                    args.eval_seqs,
                    args.frames,
                    args.workers,
                ),
            )
            print(json.dumps(r), flush=True)
            with open(f"{run}/eval.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
        print(
            json.dumps(
                dict(
                    eval_s=round(time.time() - t, 1),
                    step=at,
                    gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                )
            ),
            flush=True,
        )

    start, ck = 0, None
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        start = ck["step"]
    if args.eval_only:
        os.makedirs(run, exist_ok=True)
        full_eval(start)
        return
    chosen = None
    if args.measure and not args.typical:
        chosen, info = largest_windows(bases, args.frames, args.seqs)
        print("largest", [(i[0], i[1], i[2]) for i in info], flush=True)
    plan_kw = dict(
        seed=args.seed,
        stride=args.stride,
        width=args.bucket,
        min_games=args.min_bucket_games,
        tokens=args.tokens,
        per_window=TOKENS_PER_WINDOW,
        accum=args.accum,
    )
    data = batches(
        bases,
        args.vocab,
        map_ids,
        args.seqs,
        args.frames,
        args.workers,
        plan_kw,
        start * args.accum,
        chosen=chosen,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    if ck:
        opt.load_state_dict(ck["opt"])
        del ck
    if not args.measure:
        os.makedirs(run, exist_ok=True)
        json.dump(
            dict(vars(args), commit=commit, map_ids=map_ids),
            open(
                f"{run}/config_from{start}.json" if start else f"{run}/config.json", "w"
            ),
        )
    steps = args.measure_steps if args.measure else args.steps
    t0 = time.time()
    if not args.no_compile:
        warm_compile(model, games, ids, args.frames)
    for step in range(start, steps):
        logs = collections.Counter()
        data_s = 0.0
        # an update's passes, weighted by their windows: each loss term is a mean over
        # its pass, so without weights a pass of 3 large windows would count as much as
        # one of 8 small ones; weighted, every window counts the same whichever pass it
        # is in. All windows have the same frames, so counting each frame's prediction
        # as a sample gives the same weights
        t = time.time()
        passes = [next(data) for _ in range(args.accum if not args.measure else 1)]
        data_s += time.time() - t
        sizes = [b["mask"].shape[0] for b in passes]
        for b, size in zip(passes, sizes):
            w = size / sum(sizes)
            b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, lg = model(b)
            (loss * w).backward()
            del loss
            # a pass's checkpoints sit in a reference cycle: uncollected, they take GBs
            # from the next pass
            gc.collect()
            for k, v in lg.items():
                logs[k] += float(v) * w
        del passes
        n = 1  # logs are already the update's weighted means
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)
        opt.step()
        opt.zero_grad(set_to_none=True)
        model.update_teacher()
        if args.measure or step % 20 == 0:
            print(
                json.dumps(
                    dict(
                        step=step,
                        s=round(time.time() - t0, 1),
                        data_s=round(data_s, 2),
                        gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        windows=b["mask"].shape[0],
                        rows=b["mask"].shape[2],
                        actions=b["act_mask"].shape[2],
                        **{k: round(v / n, 4) for k, v in logs.items()},
                    )
                ),
                flush=True,
            )
        if args.measure:
            continue
        if (step + 1) % args.val_every == 0:
            t = time.time()
            for k, w in quick.items():
                print(
                    json.dumps(
                        dict(
                            val=k,
                            step=step + 1,
                            **evaluate(
                                model,
                                val_games,
                                w,
                                args.eval_seqs,
                                args.frames,
                                args.workers,
                            ),
                        )
                    ),
                    flush=True,
                )
            print(
                json.dumps(dict(val_s=round(time.time() - t, 1), step=step + 1)),
                flush=True,
            )
        out_of_time = bool(args.hours) and (
            time.time() - t_start > args.hours * 3600 - 300
        )  # 5 minutes kept for the last checkpoint and its evaluation
        if (step + 1) % args.save_every == 0 or out_of_time:
            torch.save(
                dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1),
                f"{run}/model_step{step + 1}.pt",
            )
            full_eval(step + 1)
        if out_of_time:
            print(json.dumps(dict(stopped=step + 1, hours=args.hours)), flush=True)
            break


if __name__ == "__main__":
    main()
