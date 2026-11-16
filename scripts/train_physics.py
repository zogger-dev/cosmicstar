"""Trains the physics model (worldmodel/physics.py, DESIGN_physics.md) on the Arrow
store.

    python scripts/train_physics.py --seqs 6 --frames 56 --accum 4 --steps 40000
    python scripts/train_physics.py --measure      # peak memory at the largest windows, a few steps, nothing saved
    python scripts/train_physics.py --resume RUN/model_step2000.pt --run RUN              # carries on from a checkpoint
    python scripts/train_physics.py --resume RUN/model_step2000.pt --run RUN --eval_only  # the full evaluation of a checkpoint, then exits

Quick validation every --val_every updates goes to the log. The full evaluation pauses
training after each checkpoint, and goes to the log and RUN/eval.jsonl.
"""

import _thread
import argparse
import collections
import gc
import glob
import json
import math
import os
import queue
import signal
import subprocess
import sys
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

from cosmicstar.worldmodel import (
    entity_encoder,
    epochs,
    physics,
    physics_data,
    statefile,
)
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

    def __init__(self, bases, vocab, avocab, keep=48, map_ids=None, compact=True):
        self.bases, self.vocab, self.avocab, self.keep, self.map_ids, self.compact = (
            bases,
            vocab,
            avocab,
            keep,
            map_ids or {},
            compact,
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
        return g.window(t0, L, swap=swap, compact=self.compact)


def save_checkpoint(run, model, opt, step_num):
    """Atomically write a model + optimizer checkpoint via temporary file + rename."""
    os.makedirs(run, exist_ok=True)
    dst = f"{run}/model_step{step_num}.pt"
    tmp = f"{run}/.model_step{step_num}.pt.tmp"
    torch.save(
        dict(model=model.state_dict(), opt=opt.state_dict(), step=step_num),
        tmp,
    )
    os.replace(tmp, dst)
    return dst


def warm_compile(model, games, ids, L):
    """Compile on two windows. Compiling holds several GB until the next collection, and
    a full pass has no room for that. No optimizer step: this changes no weights."""
    rng = np.random.default_rng()
    b = physics_data.collate(
        [games.read(ids[rng.integers(len(ids))], -1, L, False) for _ in range(2)],
        games.vocab,
        drop_dense_slots=True,
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


def host_mem_info():
    """Return (host_used_gb, swap_free_mb) from /proc/meminfo (includes /dev/shm)."""
    vals = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                vals[parts[0].rstrip(":")] = int(parts[1])
                if "MemAvailable" in vals and "SwapFree" in vals:
                    break
    used_gb = (vals["MemTotal"] - vals["MemAvailable"]) / (1024.0 * 1024.0)
    swap_free_mb = vals["SwapFree"] / 1024.0
    return used_gb, swap_free_mb


def start_mem_watchdog(max_ram_gb, procs_ref, stop_event):
    """Poll /proc/meminfo every 0.5 s; kill workers and abort cleanly well before OOM."""

    def watch():
        while not stop_event.is_set():
            used_gb, swap_mb = host_mem_info()
            if used_gb >= max_ram_gb or (
                used_gb >= max(44.0, max_ram_gb - 2.0) and swap_mb < 32.0
            ):
                stop_event.set()
                for p in list(procs_ref):
                    if p.is_alive():
                        p.kill()
                print(
                    json.dumps(
                        dict(
                            abort="memory_watchdog",
                            ram_used_gb=round(used_gb, 2),
                            max_ram_gb=max_ram_gb,
                            swap_free_mb=round(swap_mb, 1),
                        )
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                _thread.interrupt_main()
                time.sleep(2.0)
                os._exit(2)
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()


def _worker_loop(in_q, out_q, bases, vocab_path, map_ids, L):
    """Worker process: reads and collates assigned passes with a small game cache and
    returns compact NumPy dicts over pipe (zero /dev/shm allocation). Ignores group
    SIGINT/SIGTERM so the main process can finish its current step and checkpoint."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    vocab = Vocab(json.load(open(vocab_path)))
    games = Games(bases, vocab, ActionVocab(), keep=4, map_ids=map_ids, compact=True)
    pool = ThreadPoolExecutor(2)
    while True:
        item = in_q.get()
        if item is None:
            return
        idx, wins = item
        try:
            ws = list(pool.map(lambda p: games.read(p[0], p[1], L, p[2]), wins))
            batch = physics_data.collate(ws, vocab, drop_dense_slots=True)
            del ws
            raw = {k: v.numpy() for k, v in batch.items()}
            del batch
            gc.collect(0)
            out_q.put((idx, raw))
            del raw
        except Exception as e:
            out_q.put((idx, e))


def batches(
    bases,
    vocab_path,
    map_ids,
    seqs,
    L,
    workers,
    plan_kw=None,
    start=0,
    prefetch=3,
    chosen=None,
    ram_soft_gb=44.0,
    procs_ref=None,
):
    """Passes of windows collated across bounded OS worker processes with graduated
    host-RAM backpressure, reassembled strictly in deterministic epoch-plan order."""
    n_procs = min(max(1, workers), 8)
    ctx = mp.get_context("spawn")
    cap = prefetch + n_procs
    slots = threading.Semaphore(cap)
    in_flight_lock = threading.Lock()
    in_flight = [0]
    in_q = ctx.Queue(cap)
    out_q = ctx.Queue(cap + 2)
    procs = [
        ctx.Process(
            target=_worker_loop,
            args=(in_q, out_q, bases, vocab_path, map_ids, L),
            daemon=True,
        )
        for _ in range(n_procs)
    ]
    if procs_ref is not None:
        procs_ref.extend(procs)
    for p in procs:
        p.start()
    ready = queue.Queue(prefetch)
    stop = threading.Event()

    def throttle_and_acquire():
        slots.acquire()
        while not stop.is_set():
            with in_flight_lock:
                cur = in_flight[0]
            used_gb = host_mem_info()[0]
            limit = (
                cap
                if used_gb < ram_soft_gb
                else max(2, n_procs // 2)
                if used_gb < ram_soft_gb + 3.0
                else 0
            )
            if cur <= limit:
                break
            time.sleep(0.05)
        with in_flight_lock:
            in_flight[0] += 1

    def feed():
        try:
            idx = 0
            if chosen:
                i, rng = 0, np.random.default_rng()
                while not stop.is_set():
                    throttle_and_acquire()
                    if stop.is_set():
                        return
                    pick = [
                        (
                            chosen[(i + k) % len(chosen)][0],
                            chosen[(i + k) % len(chosen)][1],
                            bool(rng.integers(2)),
                        )
                        for k in range(seqs)
                    ]
                    i += seqs
                    in_q.put((idx, pick))
                    idx += 1
            plan = epochs.Plan(
                bases,
                {g: map_of(b) for g, b in bases.items()},
                L,
                seqs=seqs,
                **(plan_kw or {}),
            )
            epoch, k = plan.locate(start)
            while not stop.is_set():
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
                    throttle_and_acquire()
                    if stop.is_set():
                        return
                    in_q.put((idx, wins))
                    idx += 1
                epoch, k = epoch + 1, 0
        except Exception as e:
            ready.put(e)

    def _to_pinned(v, use_pin):
        t = torch.from_numpy(v)
        return t.pin_memory() if use_pin and t.numel() * t.element_size() >= 65536 else t

    def pull():
        use_pin = torch.cuda.is_available()
        if use_pin:
            torch.cuda.set_device(0)
        pending, nxt = {}, 0
        while not stop.is_set():
            try:
                idx, raw = out_q.get(timeout=5)
                b = (
                    raw
                    if isinstance(raw, Exception)
                    else {k: _to_pinned(v, use_pin) for k, v in raw.items()}
                )
                pending[idx] = b
                while nxt in pending:
                    ready.put(pending.pop(nxt))
                    nxt += 1
            except queue.Empty:
                if not stop.is_set() and not any(p.is_alive() for p in procs):
                    ready.put(RuntimeError("all data reader processes died"))
                    return

    threading.Thread(target=feed, daemon=True).start()
    threading.Thread(target=pull, daemon=True).start()
    try:
        while True:
            b = ready.get()
            with in_flight_lock:
                in_flight[0] = max(0, in_flight[0] - 1)
            slots.release()
            if isinstance(b, Exception):
                raise b
            yield b
    finally:
        stop.set()
        for p in procs:
            if p.is_alive():
                p.kill()


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


def load_init_weights(model, path):
    """Warm-start compatible weights from `path` (handling _orig_mod, map_dec channel
    shrinkage, and REG pruning), returning newly initialized parameter names."""
    raw_sd = entity_encoder.upgrade_state(torch.load(path, map_location="cpu")["model"])
    clean_raw = {
        k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v
        for k, v in raw_sd.items()
    }
    cur_sd = model.state_dict()
    load_sd = {}
    reg_keep = [0, 1] + [2 + physics.REG_LEGACY_NAMES.index(n) for n, _ in physics.REG]
    pre_pillar2 = "head.stay_buffs.weight" not in clean_raw
    pw20_shift = math.log(20.0)
    for k, cur_v in cur_sd.items():
        ck_key = k.replace("._orig_mod.", ".").removeprefix("_orig_mod.")
        if ck_key not in clean_raw:
            continue
        src_v = clean_raw[ck_key]
        if src_v.shape == cur_v.shape:
            if pre_pillar2 and ck_key in ("head.buffs.bias", "head.abilities.bias"):
                load_sd[k] = src_v - pw20_shift
            else:
                load_sd[k] = src_v
        elif (
            ck_key in ("map_dec.mlp.4.weight", "map_dec.mlp.4.bias")
            and src_v.shape[0] > cur_v.shape[0]
        ):
            load_sd[k] = src_v[: cur_v.shape[0]]
        elif ck_key in ("head.reg.weight", "head.reg.bias") and src_v.shape[0] == (
            2 + len(physics.REG_LEGACY_NAMES)
        ):
            load_sd[k] = src_v[reg_keep]
    missing, _ = model.load_state_dict(load_sd, strict=False)
    print(
        json.dumps(
            dict(
                init_from=path,
                loaded_tensors=len(load_sd),
                new_tensors=len(missing),
                new_keys=missing,
            )
        ),
        flush=True,
    )
    return set(missing)


@torch.no_grad()
def evaluate(model, games, windows, seqs, L, workers):
    """Every loss over fixed windows, and position error by kind of row. Sampling inside
    the losses is seeded too, and training's random state is left as it was."""
    grad_ckpt, model.enc.grad_ckpt = model.enc.grad_ckpt, False
    model.record_pos = True
    logs, pos, eff_cnts, n = (
        collections.Counter(),
        collections.defaultdict(list),
        collections.Counter(),
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
                for k, v in physics_data.collate(
                    ws, games.vocab, drop_dense_slots=True
                ).items()
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, lg = model(b)
            for k, v in lg.items():
                logs[k] += float(v)
            for k, v in model.pos_eval.items():
                pos[k].append(v)
            if getattr(model, "eff_eval", None) is not None:
                for k, v in model.eff_eval.items():
                    eff_cnts[k] += int(v)
            n += 1
    model.enc.grad_ckpt, model.record_pos = grad_ckpt, False
    out = {k: round(v / n, 4) for k, v in logs.items()}
    if eff_cnts:
        out["eff_persist_f1"] = round(
            2.0
            * eff_cnts["pers_tp"]
            / max(1, eff_cnts["pers_pred"] + eff_cnts["pers_gt"]),
            4,
        )
        out["eff_expire_f1"] = round(
            2.0
            * eff_cnts["exp_tp"]
            / max(1, eff_cnts["exp_pred"] + eff_cnts["exp_gt"]),
            4,
        )
        out["eff_spawn_cnt_f1"] = round(
            2.0
            * eff_cnts["sp_cnt_tp"]
            / max(1, eff_cnts["sp_cnt_pred"] + eff_cnts["sp_cnt_gt"]),
            4,
        )
        out["eff_spawn_type_acc"] = round(
            eff_cnts["sp_ty_ok"] / max(1, eff_cnts["sp_ty_tot"]), 4
        )
        if eff_cnts["sp_spell_tube_tot"] > 0:
            out["eff_spawn_spell_rec"] = round(
                eff_cnts["sp_spell_tube_ok"] / max(1, eff_cnts["sp_spell_tube_tot"]), 4
            )
        if eff_cnts["sp_spell_ty_tot"] > 0:
            out["eff_spawn_spell_ty_acc"] = round(
                eff_cnts["sp_spell_ty_ok"] / max(1, eff_cnts["sp_spell_ty_tot"]), 4
            )
        if eff_cnts["type_chg_tot"] > 0:
            out["type_morph_acc"] = round(
                eff_cnts["type_chg_ok"] / max(1, eff_cnts["type_chg_tot"]), 4
            )
        if eff_cnts["flags_chg_pred"] + eff_cnts["flags_chg_gt"] > 0:
            out["flags_flip_f1"] = round(
                2.0
                * eff_cnts["flags_chg_tp"]
                / max(1, eff_cnts["flags_chg_pred"] + eff_cnts["flags_chg_gt"]),
                4,
            )
        if eff_cnts["groups_chg_pred"] + eff_cnts["groups_chg_gt"] > 0:
            out["groups_flip_f1"] = round(
                2.0
                * eff_cnts["groups_chg_tp"]
                / max(1, eff_cnts["groups_chg_pred"] + eff_cnts["groups_chg_gt"]),
                4,
            )
        if eff_cnts["map_chg_pred"] + eff_cnts["map_chg_gt"] > 0:
            out["map_flip_f1"] = round(
                2.0
                * eff_cnts["map_chg_tp"]
                / max(1, eff_cnts["map_chg_pred"] + eff_cnts["map_chg_gt"]),
                4,
            )
        if eff_cnts["birth_unpar_tot"] > 0:
            out["birth_unpar_cnt_rec"] = round(
                eff_cnts["birth_unpar_ok"] / max(1, eff_cnts["birth_unpar_tot"]), 4
            )
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
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument(
        "--tubelet",
        type=int,
        default=4,
        help="frames per causal tubelet (1: frame-by-frame)",
    )
    ap.add_argument("--d_tube", type=int, default=256, help="tubelet trunk width")
    ap.add_argument("--dh", type=int, default=64)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv_heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--measure_steps", type=int, default=4)
    ap.add_argument(
        "--typical",
        action="store_true",
        help="--measure on random windows instead of the largest",
    )
    ap.add_argument("--blocks", type=int, default=6, help="spatial-temporal blocks")
    ap.add_argument(
        "--spatial_every",
        type=int,
        default=1,
        help="spatial attention on every n-th frame of a block (offsets spread over the"
        " blocks); the window's first frame always",
    )
    ap.add_argument("--run", default="")
    ap.add_argument(
        "--gpu_gb",
        type=float,
        default=28.0,
        help=(
            "most GPU memory PyTorch may hold, cache included, so the desktop keeps the"
            " rest"
        ),
    )
    ap.add_argument(
        "--ram_gb",
        type=float,
        default=50.0,
        help="hard host RAM ceiling (GiB); watchdog aborts before swap/OOM",
    )
    ap.add_argument(
        "--ram_soft_gb",
        type=float,
        default=44.0,
        help="soft host RAM threshold (GiB) that pauses prefetching",
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
        "--init_from",
        default="",
        help="checkpoint to warm-start model weights from at step 0 (fresh optimizer)",
    )
    ap.add_argument(
        "--new_lr_mult",
        type=float,
        default=4.0,
        help="LR multiplier on newly initialized heads when --init_from is set",
    )
    ap.add_argument(
        "--eval_only",
        action="store_true",
        help="the full evaluation of --resume, then exit",
    )
    args = ap.parse_args()
    t_start = time.time()  # --hours counts from here
    watchdog_procs, watchdog_stop = [], threading.Event()
    start_mem_watchdog(args.ram_gb, watchdog_procs, watchdog_stop)

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(
            min(
                1.0,
                args.gpu_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory,
            )
        )
    vocab, avocab = Vocab(json.load(open(args.vocab))), ActionVocab()
    split = json.load(open(args.split))
    usable = {
        g: b
        for g, b in physics_data.base_paths(args.games).items()
        if g not in EXCLUDED
    }
    all_patch = of_patch(usable, args.patch)
    train_ids = set(split["train"]) - set(EXCLUDED)
    bases = {g: all_patch[g] for g in sorted(train_ids) if g in all_patch}
    map_ids = {
        m: i for i, m in enumerate(sorted({map_of(b) for b in bases.values()}))
    }
    ids = sorted(bases)
    print(
        f"{len(ids)} training games of patch {args.patch or 'any'} across {len(map_ids)} maps",
        flush=True,
    )
    model = physics.Physics(
        vocab,
        avocab,
        args.d,
        moving_neutrals=physics_data.MOVING_NEUTRALS,
        blocks=args.blocks,
        heads=args.heads,
        kv_heads=args.kv_heads,
        dh=args.dh,
        ff=args.ff,
        spatial_every=args.spatial_every,
        tubelet=args.tubelet,
        d_tube=args.d_tube,
    ).cuda()
    model.enc.grad_ckpt = args.ckpt
    if not args.no_compile:
        model.compile_parts()
    print(
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M parameters",
        flush=True,
    )
    games = Games(bases, vocab, avocab, keep=4, map_ids=map_ids)
    held_game_ids = set(split["held_out_games"]) - set(EXCLUDED)
    held_maps = set(split.get("held_out_maps", {}).keys())
    val_bases = {
        "unseen_games": {
            g: all_patch[g] for g in sorted(held_game_ids) if g in all_patch
        },
        "unseen_maps": {
            g: b for g, b in sorted(all_patch.items()) if map_of(b) in held_maps
        },
    }
    val_bases = {k: v for k, v in val_bases.items() if v}
    quick = {
        k: fixed_windows(v, args.val_windows, args.frames) for k, v in val_bases.items()
    }
    full = {
        k: fixed_windows(v, args.eval_windows, args.frames)
        for k, v in {"train": bases, **val_bases}.items()
    }
    all_bases = {k: v for d in (bases, *val_bases.values()) for k, v in d.items()}
    val_games = Games(
        all_bases, vocab, avocab, keep=4, map_ids=map_ids
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
        ev_games = Games(all_bases, vocab, avocab, keep=4, map_ids=map_ids)
        for k, w in full.items():
            r = dict(
                eval=k,
                step=at,
                windows=len(w),
                **evaluate(
                    model,
                    ev_games,
                    w,
                    args.eval_seqs,
                    args.frames,
                    args.workers,
                ),
            )
            print(json.dumps(r), flush=True)
            with open(f"{run}/eval.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
        ev_games.open.clear()
        del ev_games
        gc.collect()
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
    new_param_names = set()
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        old_sd = ck["model"]
        ck["model"] = entity_encoder.upgrade_state(old_sd)
        if ck["model"].keys() != old_sd.keys():  # saved before the entity encoders
            buffers = {n for n, _ in model.named_buffers()}
            ck["opt"] = entity_encoder.upgrade_optimizer(
                ck["opt"],
                [
                    k
                    for k in old_sd
                    if not k.startswith("teacher.")
                    and entity_encoder.upgrade_state({k: 0}).keys() - buffers
                ],
                model,
            )
        del old_sd
        model.load_state_dict(ck["model"])
        start = ck["step"]
    elif args.init_from:
        new_param_names = load_init_weights(model, args.init_from)
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
        eff_mult=max(1, args.tubelet),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    if new_param_names:
        base_p, new_p = [], []
        for n_p, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n_p in new_param_names:
                new_p.append(p)
            else:
                base_p.append(p)
        param_groups = [
            dict(params=base_p, lr=args.lr, lr_mult=1.0),
            dict(
                params=new_p,
                lr=args.lr * args.new_lr_mult,
                lr_mult=args.new_lr_mult,
            ),
        ]
        opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=0.05, fused=True)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05, fused=True)
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
    stop_sig = [None]
    last_saved = [start]

    def _on_signal(signum, _frame):
        if watchdog_stop.is_set():
            raise KeyboardInterrupt("memory_watchdog")
        name = signal.Signals(signum).name
        if stop_sig[0] is None:
            stop_sig[0] = name
            print(
                json.dumps(dict(signal=name, stopping="after_current_step")),
                flush=True,
            )
        else:
            raise KeyboardInterrupt(f"second signal {name}")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    steps = args.measure_steps if args.measure else args.steps
    t0 = time.time()
    if not args.no_compile:
        warm_compile(model, games, ids, args.frames)
        games.open.clear()
        gc.collect()
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
        ram_soft_gb=args.ram_soft_gb,
        procs_ref=watchdog_procs,
    )
    completed_step = start
    try:
        for step in range(start, steps):
            logs = collections.Counter()
            data_s = 0.0
            # an update's passes, weighted by their windows: each loss term is a mean over
            # its pass, so without weights a pass of 3 large windows would count as much as
            # one of 8 small ones; weighted, every window counts the same whichever pass it
            # is in. All windows have the same frames, so counting each frame's prediction
            # as a sample gives the same weights
            # Stream accum micro-batches one at a time so CPU RAM never holds all passes at
            # once; weight backward by window count and normalize once before clip_grad_norm_.
            n_acc = 1 if args.measure else args.accum
            tot_w = 0
            last_shape = None
            for _ in range(n_acc):
                t = time.time()
                b_cpu = next(data)
                data_s += time.time() - t
                size = int(b_cpu["mask"].shape[0])
                tot_w += size
                last_shape = (
                    size,
                    b_cpu["mask"].shape[2],
                    b_cpu["act_mask"].shape[2],
                )
                b = {k: v.cuda(non_blocking=True) for k, v in b_cpu.items()}
                del b_cpu
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, lg = model(b)
                (loss * size).backward()
                del loss, b
                # young pass objects only (full gen-2 gc.collect() scans Dynamo/Inductor caches)
                gc.collect(0)
                for k, v in lg.items():
                    logs[k] += (
                        float(v.detach()) if torch.is_tensor(v) else float(v)
                    ) * size
                del lg
            grads = [p.grad for p in params if p.grad is not None]
            if grads:
                torch._foreach_mul_(grads, 1.0 / tot_w)
            n = tot_w
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            if step + 1 <= args.warmup or args.measure or steps <= args.warmup:
                lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
            else:
                prog = min(1.0, (step + 1 - args.warmup) / max(1, steps - args.warmup))
                lr = args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
            for g in opt.param_groups:
                g["lr"] = lr * g.get("lr_mult", 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            completed_step = step + 1
            if args.measure or step % 20 == 0 or stop_sig[0] is not None:
                used_ram_gb, _ = host_mem_info()
                rec = dict(
                    step=step,
                    s=round(time.time() - t0, 1),
                    data_s=round(data_s, 2),
                    lr=round(lr, 7),
                    gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    gpu_rsv_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                    ram_gb=round(used_ram_gb, 2),
                    windows=last_shape[0],
                    rows=last_shape[1],
                    actions=last_shape[2],
                    **{k: round(v / n, 4) for k, v in logs.items()},
                )
                print(json.dumps(rec), flush=True)
                if not args.measure:
                    with open(f"{run}/train.jsonl", "a") as f:
                        f.write(json.dumps(rec) + "\n")
            if args.measure:
                if stop_sig[0] is not None:
                    break
                continue
            if stop_sig[0] is not None:
                ckpt_path = save_checkpoint(run, model, opt, completed_step)
                last_saved[0] = completed_step
                print(
                    json.dumps(
                        dict(
                            stopped=completed_step,
                            signal=stop_sig[0],
                            checkpoint=ckpt_path,
                        )
                    ),
                    flush=True,
                )
                break
            if (step + 1) % args.val_every == 0:
                t = time.time()
                for k, w in quick.items():
                    if stop_sig[0] is not None:
                        break
                    vrec = dict(
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
                    print(json.dumps(vrec), flush=True)
                    with open(f"{run}/val.jsonl", "a") as f:
                        f.write(json.dumps(vrec) + "\n")
                print(
                    json.dumps(dict(val_s=round(time.time() - t, 1), step=step + 1)),
                    flush=True,
                )
                if stop_sig[0] is not None:
                    ckpt_path = save_checkpoint(run, model, opt, completed_step)
                    last_saved[0] = completed_step
                    print(
                        json.dumps(
                            dict(
                                stopped=completed_step,
                                signal=stop_sig[0],
                                checkpoint=ckpt_path,
                            )
                        ),
                        flush=True,
                    )
                    break
            out_of_time = bool(args.hours) and (
                time.time() - t_start > args.hours * 3600 - 300
            )  # 5 minutes kept for the last checkpoint and its evaluation
            if (step + 1) % args.save_every == 0 or (step + 1) == steps or out_of_time:
                save_checkpoint(run, model, opt, step + 1)
                last_saved[0] = step + 1
                if stop_sig[0] is None:
                    full_eval(step + 1)
            if stop_sig[0] is not None:
                if last_saved[0] != completed_step:
                    ckpt_path = save_checkpoint(run, model, opt, completed_step)
                    last_saved[0] = completed_step
                else:
                    ckpt_path = f"{run}/model_step{completed_step}.pt"
                print(
                    json.dumps(
                        dict(
                            stopped=completed_step,
                            signal=stop_sig[0],
                            checkpoint=ckpt_path,
                        )
                    ),
                    flush=True,
                )
                break
            if out_of_time:
                print(json.dumps(dict(stopped=step + 1, hours=args.hours)), flush=True)
                break
    except KeyboardInterrupt:
        if (
            not args.measure
            and not watchdog_stop.is_set()
            and completed_step > last_saved[0]
        ):
            opt.zero_grad(set_to_none=True)
            ckpt_path = save_checkpoint(run, model, opt, completed_step)
            last_saved[0] = completed_step
            print(
                json.dumps(
                    dict(
                        stopped=completed_step,
                        signal=stop_sig[0] or "SIGINT",
                        checkpoint=ckpt_path,
                    )
                ),
                flush=True,
            )
        else:
            raise
    finally:
        data.close()
        watchdog_stop.set()


if __name__ == "__main__":
    main()
