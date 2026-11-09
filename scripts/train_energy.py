"""Trains the energy model (worldmodel/energy.py, DESIGN_energy.md) on the Arrow store,
reading frames through worldmodel/energy_data.py.

    python scripts/train_energy.py --patch 5.0.14 --hours 8 --gpu_gb 29
    python scripts/train_energy.py --measure     # peak memory at the largest windows, a few steps, nothing saved

Every frame of a window is a sample: the step from it to the next frame. The pair
terms are O(rows²) a frame, so frames go through the model in chunks of at most
--pairs row pairs. Quick validation every --val_every updates goes to the log; the
full evaluation pauses training after each checkpoint and goes to the log and
RUN/eval.jsonl. The reader, splits and fixed windows are train_physics's.

Unit types and order abilities enter through the ruleset encoder (worldmodel/ruleset.py)
over the patch's balance data, run once a pass and trained with the model; it lives
inside the frame encoder, so its weights are in the optimizer and the checkpoint.
"""

import argparse
import collections
import gc
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# the cap below needs segments that grow (train_physics sets the same before torch)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.multiprocessing as mp
import train_physics as tp

from cosmicstar.worldmodel import energy, energy_data, epochs, physics_data
from cosmicstar.worldmodel.action_tokens import ActionVocab
from cosmicstar.worldmodel.vocab import Vocab


def _trim_batch(b):
    """Keeps only the keys and slot slices read by energy_data.Frames, cutting IPC and
    host-to-device transfer size by ~40%."""
    out = {k: b[k] for k in energy_data.BATCH_KEYS if k in b}
    if "slot_reg" in out:
        out["slot_reg"] = out["slot_reg"][..., :1, :].contiguous()
    if "slot_abil" in out:
        out["slot_abil"] = out["slot_abil"][..., : physics_data.QUEUE + 1].contiguous()
    return out


def _reader_worker(work_q, out_q, bases, vocab_path, map_ids, L, threads, keep):
    """Independent OS process that pulls (idx, wins) specifications from work_q, reads
    and collates windows without GIL contention across workers, and pushes trimmed
    batches to out_q."""
    vocab = Vocab(json.load(open(vocab_path)))
    games = tp.Games(bases, vocab, ActionVocab(), keep=keep, map_ids=map_ids)
    pool = ThreadPoolExecutor(threads)
    n = 0
    while True:
        item = work_q.get()
        if item is None:
            return
        idx, wins = item
        ws = list(pool.map(lambda p: games.read(p[0], p[1], L, p[2]), wins))
        out_q.put((idx, _trim_batch(physics_data.collate(ws, vocab))))
        del ws
        n += 1
        if n % 10 == 0:
            gc.collect()


def energy_batches(
    bases,
    vocab_path,
    map_ids,
    seqs,
    L,
    readers=2,
    threads=3,
    plan_kw=None,
    start=0,
    prefetch=2,
    chosen=None,
):
    """Multi-process prefetched batch iterator that keeps a queue of collated passes
    ready in order."""
    ctx = mp.get_context("spawn")
    max_in_flight = prefetch + readers
    work_q = ctx.Queue(max_in_flight)
    out_q = ctx.Queue(max_in_flight)
    sem = threading.Semaphore(max_in_flight)
    keep = 10

    procs = []
    for _ in range(readers):
        p = ctx.Process(
            target=_reader_worker,
            args=(work_q, out_q, bases, vocab_path, map_ids, L, threads, keep),
            daemon=True,
        )
        p.start()
        procs.append(p)

    def feeder():
        idx = 0
        if chosen:
            i, rng = 0, np.random.default_rng(0)
            while True:
                sem.acquire()
                pick = [
                    (
                        chosen[(i + k) % len(chosen)][0],
                        chosen[(i + k) % len(chosen)][1],
                        bool(rng.integers(2)),
                    )
                    for k in range(seqs)
                ]
                i += seqs
                work_q.put((idx, pick))
                idx += 1
        plan = epochs.Plan(
            bases,
            {g: tp.map_of(b) for g, b in bases.items()},
            L,
            seqs=seqs,
            **(plan_kw or {}),
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
                sem.acquire()
                work_q.put((idx, wins))
                idx += 1
            epoch, k = epoch + 1, 0

    threading.Thread(target=feeder, daemon=True).start()
    ready = queue.Queue(prefetch)

    def puller():
        while True:
            try:
                _, b = out_q.get(timeout=5)
                b_pinned = {
                    k: (
                        v.pin_memory()
                        if isinstance(v, torch.Tensor) and v.numel() > 4096
                        else v
                    )
                    for k, v in b.items()
                }
                ready.put(b_pinned)
            except queue.Empty:
                dead = [p for p in procs if not p.is_alive()]
                if dead:
                    ready.put(
                        RuntimeError(
                            f"reader process died, exit code {dead[0].exitcode}"
                        )
                    )
                    return
            except (ConnectionResetError, EOFError, OSError):
                return

    threading.Thread(target=puller, daemon=True).start()
    while True:
        b = ready.get()
        if isinstance(b, Exception):
            raise b
        sem.release()
        yield b


def report(logs, digits=5):
    """Logs for printing: the mean squared errors averaged over chunks and passes
    become their roots (rmse_*, copy_rmse_*). Converts GPU scalars to float only at
    logging time so the training loop never stalls on cudaStreamSynchronize."""
    out = {}
    for k, v in logs.items():
        v = float(v)
        if k.startswith(("mse_", "copy_mse_")):
            k, v = k.replace("mse_", "rmse_"), max(v, 0.0) ** 0.5
        out[k] = round(v, digits)
    return out


def run_batch(model, enc, b, pairs=None, backward=True, weight=1.0):
    """One pass's frames (every window, frames 1 to F − 2) through the encoder (bf16
    attention) and the fixed-dimension Gram Port-Hamiltonian model (fp32)."""
    B, F_, R = b["mask"].shape
    X = b["eff_mask"].shape[2] if "eff_mask" in b else 0
    T = R + 67 + X
    if backward and B > 1 and B * T > 6200:
        B_keep = max(1, 6200 // T)
        b = {k: v[:B_keep] for k, v in b.items()}
    with torch.set_grad_enabled(backward), torch.autocast("cuda", dtype=torch.bfloat16):
        s_t, s_next = enc.frames(b, list(range(1, F_ - 1)))
    # Run the fixed-dimension Port-Hamiltonian step in full fp32 precision
    with torch.enable_grad(), torch.autocast("cuda", enabled=False):
        loss, lg = model(s_t, s_next)
    if backward:
        (loss * weight).backward()
    logs = {k: v.detach() for k, v in lg.items()}
    logs["loss"] = loss.detach()
    del loss, s_t, s_next
    return logs


_EVAL_CACHE = {}


def evaluate(model, enc, games, windows, seqs, L, workers, pairs):
    """The losses and logs over fixed windows. The force needs autograd, so this runs
    under enable_grad in eval mode (no double-backward graph) and never steps."""
    model.eval(), enc.eval()
    logs, n = {}, 0
    key = (id(windows), seqs, L)
    if key in _EVAL_CACHE:
        batches = _EVAL_CACHE[key]
    elif len(windows) <= 16:
        batches = []
        with ThreadPoolExecutor(workers) as pool:
            for i in range(0, len(windows), seqs):
                ws = list(
                    pool.map(
                        lambda p: games.read(p[0], p[1], L, p[2]),
                        windows[i : i + seqs],
                    )
                )
                batches.append(_trim_batch(physics_data.collate(ws, games.vocab)))
        _EVAL_CACHE[key] = batches
    else:
        with ThreadPoolExecutor(workers) as pool, torch.enable_grad():
            for i in range(0, len(windows), seqs):
                ws = list(
                    pool.map(
                        lambda p: games.read(p[0], p[1], L, p[2]),
                        windows[i : i + seqs],
                    )
                )
                b_cpu = _trim_batch(physics_data.collate(ws, games.vocab))
                del ws
                b = energy_data.to_device(b_cpu, "cuda")
                del b_cpu
                lg = run_batch(model, enc, b, pairs, backward=False)
                for k, v in lg.items():
                    logs[k] = logs[k] + v if k in logs else v
                n += 1
                del b
        model.train(), enc.train()
        return report({k: v / n for k, v in logs.items()})
    with torch.enable_grad():
        for b_cpu in batches:
            b = energy_data.to_device(b_cpu, "cuda")
            lg = run_batch(model, enc, b, pairs, backward=False)
            for k, v in lg.items():
                logs[k] = logs[k] + v if k in logs else v
            n += 1
            del b
    model.train(), enc.train()
    return report({k: v / n for k, v in logs.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data/games_v3")
    ap.add_argument("--vocab", default="data/worldmodel/frame_cache_seq/vocab.json")
    ap.add_argument("--patch", default="5.0.14")
    ap.add_argument("--base_build", type=int, default=94137, help="the patch's ruleset")
    ap.add_argument("--split", default="data/games/splits/default.json")
    ap.add_argument("--seqs", type=int, default=12, help="most windows in a pass")
    ap.add_argument("--frames", type=int, default=112)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--dh", type=int, default=64)
    ap.add_argument("--ff", type=int, default=768)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--u_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--rules_tables", default="data/worldmodel/rulesets.json")
    ap.add_argument("--rules_d", type=int, default=256, help="ruleset encoder width")
    ap.add_argument(
        "--rules_depth", type=int, default=3, help="attention layers over the ruleset"
    )
    ap.add_argument("--rules_heads", type=int, default=8)
    ap.add_argument(
        "--rules_no_residual",
        action="store_true",
        help="no learned per-name residual on the ruleset nodes",
    )
    ap.add_argument(
        "--pairs", type=int, default=2_000_000, help="row pairs a chunk of frames"
    )
    ap.add_argument(
        "--tokens", type=int, default=6600, help="rows a pass, for the epoch plan"
    )
    ap.add_argument("--stride", type=int, default=14)
    ap.add_argument("--bucket", type=int, default=32)
    ap.add_argument("--min_bucket_games", type=int, default=40)
    ap.add_argument("--readers", type=int, default=6, help="reader worker processes")
    ap.add_argument("--workers", type=int, default=3, help="threads per reader process")
    ap.add_argument("--prefetch", type=int, default=2, help="prefetched passes in RAM")
    ap.add_argument("--gpu_gb", type=float, default=29.2)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--measure_steps", type=int, default=4)
    ap.add_argument("--run", default="")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--val_every", type=int, default=25)
    ap.add_argument("--val_windows", type=int, default=8)
    ap.add_argument("--eval_windows", type=int, default=32)
    ap.add_argument("--eval_seqs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, args.gpu_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    )
    vocab, avocab = Vocab(json.load(open(args.vocab))), ActionVocab()
    split = json.load(open(args.split))
    train_ids = set(split["train"]) - set(tp.EXCLUDED)
    bases = tp.of_patch(physics_data.base_paths(args.games, train_ids), args.patch)
    map_ids = {
        m: i for i, m in enumerate(sorted({tp.map_of(b) for b in bases.values()}))
    }
    ids = sorted(bases)
    print(
        f"{len(ids)} training games of patch {args.patch} on {len(map_ids)} maps",
        flush=True,
    )
    tables = energy_data.Tables(args.base_build, rulesets=args.rules_tables)
    print(f"ruleset {tables.ruleset}", flush=True)
    model = energy.EnergyModel(args.d, args.u_dim, args.hidden).cuda()
    enc = energy_data.Frames(
        vocab,
        tables,
        args.d,
        args.u_dim,
        attn_layers=args.blocks,
        dh=args.dh,
        ff=args.ff,
    ).cuda()
    if not args.no_compile:
        enc.compile_parts()
        model.compile_parts()
    params = [p for m in (model, enc) for p in m.parameters() if p.requires_grad]
    print(
        f"{sum(p.numel() for p in params) / 1e6:.2f}M parameters (model"
        f" {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, enc"
        f" {sum(p.numel() for p in enc.parameters()) / 1e6:.2f}M)",
        flush=True,
    )
    val_bases = {
        "unseen_games": tp.of_patch(
            physics_data.base_paths(
                args.games, set(split["held_out_games"]) - set(tp.EXCLUDED)
            ),
            args.patch,
        )
    }
    quick = {
        k: tp.fixed_windows(v, args.val_windows, args.frames)
        for k, v in val_bases.items()
    }
    full = {
        k: tp.fixed_windows(v, args.eval_windows, args.frames)
        for k, v in {"train": bases, **val_bases}.items()
    }
    all_bases = {k: v for d in (bases, *val_bases.values()) for k, v in d.items()}
    val_games = tp.Games(
        all_bases, vocab, avocab, keep=2 * args.val_windows, map_ids=map_ids
    )
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    run = (
        args.run
        or f"data/worldmodel/runs/energy_{commit}_{time.strftime('%Y%m%d_%H%M%S')}"
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
                    enc,
                    tp.Games(all_bases, vocab, avocab, keep=4, map_ids=map_ids),
                    w,
                    args.eval_seqs,
                    args.frames,
                    args.workers,
                    args.pairs,
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
        enc.load_state_dict(ck["enc"])
        start = ck["step"]
    chosen = None
    eff_seqs = args.seqs
    if args.measure:
        chosen, info = tp.largest_windows(bases, args.frames, args.seqs)
        eff_seqs = max(1, min(args.seqs, args.tokens // (info[0][0] + 67)))
        print(
            "largest",
            [(i[0], i[1], i[2]) for i in info[:eff_seqs]],
            f"(eff_seqs={eff_seqs})",
            flush=True,
        )
    plan_kw = dict(
        seed=args.seed,
        stride=args.stride,
        width=args.bucket,
        min_games=args.min_bucket_games,
        tokens=args.tokens,
        per_window=67,
        accum=args.accum,
    )
    data = energy_batches(
        bases,
        args.vocab,
        map_ids,
        eff_seqs,
        args.frames,
        readers=args.readers,
        threads=args.workers,
        plan_kw=plan_kw,
        start=start * args.accum,
        prefetch=args.prefetch,
        chosen=chosen,
    )
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    if ck:
        opt.load_state_dict(ck["opt"])
        del ck
    if not args.measure:
        os.makedirs(run, exist_ok=True)
        json.dump(
            dict(vars(args), commit=commit),
            open(
                f"{run}/config_from{start}.json" if start else f"{run}/config.json", "w"
            ),
        )
    steps = args.measure_steps if args.measure else args.steps
    t0 = time.time()
    for step in range(start, steps):
        logs = {}
        t = time.time()
        passes = [next(data) for _ in range(args.accum if not args.measure else 1)]
        data_s = time.time() - t
        sizes = [b["mask"].shape[0] for b in passes]
        rows = max(b["mask"].shape[2] for b in passes)
        for b, size in zip(passes, sizes):
            w = size / sum(sizes)
            b = energy_data.to_device(b, "cuda")
            lg = run_batch(model, enc, b, args.pairs, weight=w)
            for k, v in lg.items():
                logs[k] = logs[k] + v * w if k in logs else v * w
            del b
        del passes
        if step == start + 1:
            gc.collect()
            gc.freeze()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if args.measure or step % 20 == 0:
            with torch.no_grad():
                emb_norms = enc.emb_norms()
            print(
                json.dumps(
                    dict(
                        step=step,
                        s=round(time.time() - t0, 1),
                        data_s=round(data_s, 2),
                        gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        windows=sizes[-1],
                        rows=rows,
                        **report(logs),
                        **{
                            f"emb_norm_{k}": round(v, 4)
                            for k, v in emb_norms.items()
                        },
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
                                enc,
                                val_games,
                                w,
                                args.eval_seqs,
                                args.frames,
                                args.workers,
                                args.pairs,
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
        )
        if (step + 1) % args.save_every == 0 or out_of_time:
            torch.save(
                dict(
                    model=model.state_dict(),
                    enc=enc.state_dict(),
                    opt=opt.state_dict(),
                    step=step + 1,
                ),
                f"{run}/model_step{step + 1}.pt",
            )
            full_eval(step + 1)
        if out_of_time:
            print(json.dumps(dict(stopped=step + 1, hours=args.hours)), flush=True)
            break


if __name__ == "__main__":
    main()
