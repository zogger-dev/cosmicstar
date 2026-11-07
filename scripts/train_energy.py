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
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# the cap below needs segments that grow (train_physics sets the same before torch)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import train_physics as tp

from cosmicstar.worldmodel import energy, energy_data, physics_data
from cosmicstar.worldmodel.action_tokens import ActionVocab
from cosmicstar.worldmodel.ruleset import RulesetEncoder, RulesetTables
from cosmicstar.worldmodel.vocab import Vocab


def sub(s, sl):
    """The frames sl of a State."""
    return energy.State(
        s.q[sl],
        s.g[sl],
        s.k[sl],
        s.c[sl],
        s.owner[sl],
        s.mask[sl],
        s.p[sl],
        s.cp[sl],
        s.u[sl],
    )


def run_batch(model, enc, b, pairs, backward=True, weight=1.0):
    """One pass's frames (every window, frames 1 to F − 2) through the model in chunks
    of at most `pairs` row pairs, the loss backward per chunk weighted by its share of
    the frames -> the pass's mean logs. The encoder runs once for all frames; only the
    pair terms are chunk-sized. The graph is cut at the encoder's outputs (c, cp, u,
    the only parts of the State that carry gradient): each chunk's backward stops
    there, and one backward through the encoder and the ruleset encoder at the end of
    the pass carries the summed gradients, instead of one per chunk."""
    B, F_, R = b["mask"].shape
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s, q_next, p_next, scored = enc.frames(b, list(range(1, F_ - 1)))
    heads = (s.c, s.cp, s.u)
    # fp32 leaves: their gradient sums over the chunks (bf16 would round each add)
    cut = [
        t.detach().float().requires_grad_(backward and t.requires_grad) for t in heads
    ]
    s = energy.State(
        s.q, s.g, s.k, cut[0], s.owner, s.mask, s.p, cut[1], cut[2]
    )  # c, cp, u
    M = s.mask.shape[0]
    step = max(1, pairs // max(1, R * R))
    logs = collections.Counter()
    for i in range(0, M, step):
        sl = slice(i, i + step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, lg = model(sub(s, sl), q_next[sl], p_next[sl], scored[sl])
        w = min(step, M - i) / M
        if backward:
            (loss * w * weight).backward()
        for k, v in lg.items():
            logs[k] += float(v) * w
        logs["loss"] += float(loss.detach()) * w
        del loss
    if backward:
        pairs_ = [
            (h, c.grad.to(h.dtype)) for h, c in zip(heads, cut) if c.grad is not None
        ]
        if pairs_:
            torch.autograd.backward(*zip(*pairs_))
    del s, cut, heads
    return logs


def evaluate(model, enc, games, windows, seqs, L, workers, pairs):
    """The losses and logs over fixed windows. The force needs autograd, so this runs
    under enable_grad in eval mode (no double-backward graph) and never steps."""
    model.eval()
    logs, n = collections.Counter(), 0
    with ThreadPoolExecutor(workers) as pool, torch.enable_grad():
        for i in range(0, len(windows), seqs):
            ws = list(
                pool.map(
                    lambda p: games.read(p[0], p[1], L, p[2]), windows[i : i + seqs]
                )
            )
            b = energy_data.to_device(physics_data.collate(ws, games.vocab), "cuda")
            lg = run_batch(model, enc, b, pairs, backward=False)
            for k, v in lg.items():
                logs[k] += v
            n += 1
            del b
    model.train()
    return {k: round(v / n, 5) for k, v in logs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data/games_v3")
    ap.add_argument("--vocab", default="data/worldmodel/frame_cache_seq/vocab.json")
    ap.add_argument("--patch", default="5.0.14")
    ap.add_argument("--base_build", type=int, default=94137, help="the patch's ruleset")
    ap.add_argument("--split", default="data/games/splits/default.json")
    ap.add_argument("--seqs", type=int, default=16, help="most windows in a pass")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--u_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
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
        "--tokens", type=int, default=6000, help="rows a pass, for the epoch plan"
    )
    ap.add_argument("--stride", type=int, default=14)
    ap.add_argument("--bucket", type=int, default=32)
    ap.add_argument("--min_bucket_games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gpu_gb", type=float, default=29.0)
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
    tables = energy_data.Tables(args.base_build)
    print(f"ruleset {tables.ruleset}", flush=True)
    model = energy.EnergyModel(args.d, args.u_dim, args.hidden).cuda()
    rules = RulesetEncoder(
        RulesetTables(args.rules_tables),
        d=args.rules_d,
        depth=args.rules_depth,
        heads=args.rules_heads,
        residual=not args.rules_no_residual,
    )
    enc = energy_data.Frames(vocab, tables, args.d, args.u_dim, rules).cuda()
    params = [p for m in (model, enc) for p in m.parameters() if p.requires_grad]
    print(
        f"{sum(p.numel() for p in params) / 1e6:.2f}M parameters (ruleset encoder"
        f" {sum(p.numel() for p in rules.parameters()) / 1e6:.2f}M)",
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
                    tp.Games(all_bases, vocab, avocab, keep=8, map_ids=map_ids),
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
    if args.measure:
        chosen, info = tp.largest_windows(bases, args.frames, args.seqs)
        print("largest", [(i[0], i[1], i[2]) for i in info], flush=True)
    plan_kw = dict(
        seed=args.seed,
        stride=args.stride,
        width=args.bucket,
        min_games=args.min_bucket_games,
        tokens=args.tokens,
        per_window=0,
        accum=args.accum,
    )
    data = tp.batches(
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
        logs = collections.Counter()
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
                logs[k] += v * w
            del b
        del passes
        gc.collect()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if args.measure or step % 20 == 0:
            with torch.no_grad():
                res_norms = rules.residual_norms()
            print(
                json.dumps(
                    dict(
                        step=step,
                        s=round(time.time() - t0, 1),
                        data_s=round(data_s, 2),
                        gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        windows=sizes[-1],
                        rows=rows,
                        **{k: round(v, 5) for k, v in logs.items()},
                        **{
                            f"rules_res_norm_{k}": round(v, 4)
                            for k, v in res_norms.items()
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
