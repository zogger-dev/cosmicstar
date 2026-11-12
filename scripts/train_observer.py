"""Sparse Live-Window Co-Training for the Multi-Horizon Stochastic Observer (`L1` 0.625s -> `L2` 5.0s).

Jointly trains:
  - `Frames` (`L1`, warm-started from `model_step44285.pt`, adapted at `step=14` with human
    control-group & camera/onscreen features)
  - `SquadPooler` (`1` Macro + `M = 64` Squad Tokens/player via Top-2 exact-conservation routing)
  - `HindsightIntentPosterior` (`K = 4` spatial intents + macro intent + `16d` micro residual `xi`)
  - `ObserverWorldModelL2` (fog-of-war `PerspectiveMacroEncoder` + Capability `c_p` +
    Diffusion-Forcing `ConditionalFlowPrior` + token-level `SquadPortHamiltonian`)

Usage:
    PYTHONPATH=src:scripts .venv/bin/python scripts/train_observer.py --steps 2500 --batch 12 --live_batch 4
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import torch
import torch.multiprocessing as mp
import train_physics as tp

from cosmicstar.worldmodel import energy_data as ED
from cosmicstar.worldmodel import observer_l2 as OL2
from cosmicstar.worldmodel import physics_data as PD
from cosmicstar.worldmodel import squad_pooler as SP
from cosmicstar.worldmodel.action_tokens import ActionVocab
from cosmicstar.worldmodel.vocab import Vocab


def _trim_batch(b: dict) -> dict:
    """Trims collated batch dict to keys required by `Frames`, `SquadPooler`, and `HindsightIntentPosterior`."""
    out = {k: b[k] for k in ED.BATCH_KEYS if k in b}
    if "slot_reg" in out:
        out["slot_reg"] = out["slot_reg"][..., :1, :].contiguous()
    if "slot_abil" in out:
        out["slot_abil"] = out["slot_abil"][..., : PD.QUEUE + 1].contiguous()
    return out


def _to_numpy_dict(d: dict) -> dict:
    """Converts Tensor values to NumPy arrays so multiprocessing Queues never allocate `/dev/shm` segments."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        elif isinstance(v, dict):
            out[k] = _to_numpy_dict(v)
        elif isinstance(v, list):
            out[k] = [_to_numpy_dict(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def _from_numpy_dict(d: dict) -> dict:
    """Zero-copy conversion of NumPy arrays back to CPU `torch.Tensor`s in the consumer process."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v)
        elif isinstance(v, dict):
            out[k] = _from_numpy_dict(v)
        elif isinstance(v, list):
            out[k] = [_from_numpy_dict(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def _extract_one_game_np(gid: str, base: str, vocab: Vocab, avocab: ActionVocab, chunk_size: int) -> dict | None:
    """Opens a single game, extracts `L=10` (`step=14`) chunks as NumPy arrays, and frees the game immediately."""
    g = PD.Game(base, vocab)
    g._load_actions(avocab)

    t_marks = list(range(112, g.F - 113, 112))
    if len(t_marks) < 8:
        del g
        return None

    static_map = _to_numpy_dict(OL2.extract_rich_static_map(g))
    res = g.meta.get("end", {}).get("results", {})
    winner = 1 if res.get("1") == 1 else (2 if res.get("2") == 1 else 0)

    batches = []
    for i in range(0, len(t_marks), chunk_size):
        marks_c = t_marks[i : i + chunk_size]
        ws = [g.window(lp - 14, 10, swap=False, step=14) for lp in marks_c]
        batches.append(_to_numpy_dict(_trim_batch(PD.collate(ws, vocab))))
        del ws
    del g

    return dict(
        gid=gid,
        winner=winner,
        loops=np.array(t_marks, dtype=np.int32),
        static_map=static_map,
        batches=batches,
    )


def _bounded_extract_worker(in_q, out_q, vocab_path: str):
    """Bounded CPU extraction worker: blocks when `out_q` (maxsize=2) is full so RAM never grows."""
    vocab = Vocab(json.load(open(vocab_path)))
    avocab = ActionVocab()
    while True:
        item = in_q.get()
        if item is None:
            return
        gid, base, chunk_size = item
        payload = _extract_one_game_np(gid, base, vocab, avocab, chunk_size)
        out_q.put((gid, payload))
        del payload
        gc.collect()


@torch.no_grad()
def gpu_encode_game_batches(
    enc: ED.Frames,
    pooler: SP.SquadPooler,
    posterior: SP.HindsightIntentPosterior,
    cpu_payload: dict,
    device: torch.device,
) -> dict:
    """Runs `Frames` (batched `bf16` on GPU) + sequential `SquadPooler` (preserving temporal
    squad slot tracking across the entire game) + batched `HindsightIntentPosterior`."""
    enc.eval(), pooler.eval(), posterior.eval()

    q_sq_all, z_sq_all, act_all = [], [], []
    q_mac_all, z_mac_all = [], []
    u_all, tempo_all, vis_all = [], [], []

    prev_s = None
    for b_cpu in cpu_payload["batches"]:
        b = ED.to_device(b_cpu, device)
        B_c = b["mask"].shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = enc.encode_window_context(b, list(range(1, 10)))

        # Sequential SquadPooler along macro steps tau so slot queries track squads smoothly
        s_cur_list = []
        s_nxt_list = []
        for j in range(B_c):
            bj = {k: (v[j : j + 1] if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            cj = {k: v[j : j + 1] for k, v in ctx.items()}
            s_c = pooler(bj, cj, f_start=0, f_end=0, prev_state=prev_s, batch_frame_end=1)
            s_n = pooler(bj, cj, f_start=0, f_end=8, prev_state=s_c, batch_frame_end=9)
            s_cur_list.append(s_c)
            s_nxt_list.append(s_n)
            prev_s = s_n  # exact 5.0s evolved state seeds next 5.0s step!

        def _cat_s(lst, key):
            return torch.cat([x[key] for x in lst], dim=0)

        s_cur_b = {k: _cat_s(s_cur_list, k) for k in ("q_squad", "z_squad", "active", "q_macro", "z_macro", "assign_w")}
        s_nxt_b = {k: _cat_s(s_nxt_list, k) for k in ("q_squad", "z_squad", "active", "q_macro", "z_macro")}

        post_b = posterior(
            s_cur_b,
            s_nxt_b,
            b,
            ctx,
            f_indices=list(range(2, 10)),
            ctx_indices=list(range(0, 8)),
        )
        gt_vis = OL2.extract_gt_vis_live(b, f_end=1, q_squad=s_cur_b["q_squad"])

        q_sq_all.append(s_cur_b["q_squad"].cpu().float())
        z_sq_all.append(s_cur_b["z_squad"].cpu().to(torch.bfloat16))
        act_all.append(s_cur_b["active"].cpu().bool())
        q_mac_all.append(s_cur_b["q_macro"].cpu().float())
        z_mac_all.append(s_cur_b["z_macro"].cpu().to(torch.bfloat16))
        u_all.append(post_b["u_flat"].cpu().float())
        tempo_all.append(post_b["tempo_feat"].cpu().float())
        vis_all.append(gt_vis.cpu().to(torch.float16))

        # Also append the final terminal state s_T at the end of the last chunk!
        last_s_nxt = s_nxt_list[-1]
        last_b = {k: (v[-1:] if isinstance(v, torch.Tensor) else v) for k, v in b.items()}

    # Append terminal step T so T states have T-1 transitions u_flat[0..T-2]
    q_sq_all.append(last_s_nxt["q_squad"].cpu().float())
    z_sq_all.append(last_s_nxt["z_squad"].cpu().to(torch.bfloat16))
    act_all.append(last_s_nxt["active"].cpu().bool())
    q_mac_all.append(last_s_nxt["q_macro"].cpu().float())
    z_mac_all.append(last_s_nxt["z_macro"].cpu().to(torch.bfloat16))
    tempo_all.append(tempo_all[-1][-1:].clone())
    vis_last = OL2.extract_gt_vis_live(last_b, f_end=9, q_squad=last_s_nxt["q_squad"])
    vis_all.append(vis_last.cpu().to(torch.float16))

    loops_ext = torch.cat(
        [cpu_payload["loops"], cpu_payload["loops"][-1:] + 112], dim=0
    )

    return dict(
        gid=cpu_payload["gid"],
        winner=cpu_payload["winner"],
        loops=loops_ext,
        static_map=cpu_payload["static_map"],
        q_squad=torch.cat(q_sq_all, dim=0),      # [T, 2, M, 12] fp32
        z_squad=torch.cat(z_sq_all, dim=0),      # [T, 2, M, 128] bf16
        active=torch.cat(act_all, dim=0),        # [T, 2, M] bool
        q_macro=torch.cat(q_mac_all, dim=0),     # [T, 2, 6] fp32
        z_macro=torch.cat(z_mac_all, dim=0),     # [T, 2, 128] bf16
        u_flat=torch.cat(u_all, dim=0),          # [T-1, 2, 180] fp32
        tempo_feat=torch.cat(tempo_all, dim=0),  # [T, 2, 16] fp32
        gt_vis_live=torch.cat(vis_all, dim=0),   # [T, 2, M] fp16
    )


def _iter_bounded_extractions(todo: list[tuple[str, str, int]], vocab_path: str, workers: int = 6):
    """Yields `(gid, cpu_payload_tensors)` with strict backpressure (`out_q` maxsize=3) and zero `/dev/shm` tensors."""
    if not todo:
        return
    n_w = max(1, min(workers, len(todo)))
    ctx = mp.get_context("spawn")
    in_q = ctx.Queue()
    out_q = ctx.Queue(maxsize=3)
    for item in todo:
        in_q.put(item)
    for _ in range(n_w):
        in_q.put(None)

    procs = []
    for _ in range(n_w):
        p = ctx.Process(target=_bounded_extract_worker, args=(in_q, out_q, vocab_path), daemon=True)
        p.start()
        procs.append(p)

    try:
        for _ in range(len(todo)):
            gid, payload_np = out_q.get()
            if payload_np is None:
                yield gid, None
            else:
                yield gid, _from_numpy_dict(payload_np)
                del payload_np
    finally:
        for p in procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()


def build_or_load_cache(
    subset: str,
    bases: dict[str, str],
    cache_dir: str,
    vocab_path: str,
    enc: ED.Frames,
    pooler: SP.SquadPooler,
    posterior: SP.HindsightIntentPosterior,
    device: torch.device,
    workers: int = 4,
    force_rebuild: bool = False,
) -> dict[str, dict]:
    """Loads cached `bf16` macro game summaries from disk or extracts missing ones with bounded RAM."""
    sub_dir = os.path.join(cache_dir, subset)
    os.makedirs(sub_dir, exist_ok=True)

    loaded: dict[str, dict] = {}
    todo = []
    for gid, base in sorted(bases.items()):
        pt_path = os.path.join(sub_dir, f"{gid}.pt")
        if not force_rebuild and os.path.exists(pt_path):
            d = torch.load(pt_path, map_location="cpu", weights_only=False)
            if d is not None and len(d["loops"]) >= 10:
                loaded[gid] = d
        else:
            todo.append((gid, base, 24))

    if not todo:
        print(f"[{subset}] Loaded {len(loaded)} cached games from {sub_dir}", flush=True)
        return loaded

    print(
        f"[{subset}] Extracting {len(todo)} games ({len(loaded)} already cached) using {workers} bounded CPU workers...",
        flush=True,
    )
    t0 = time.perf_counter()
    done = 0
    for gid, cpu_payload in _iter_bounded_extractions(todo, vocab_path, workers=workers):
        done += 1
        if cpu_payload is not None:
            entry = gpu_encode_game_batches(enc, pooler, posterior, cpu_payload, device)
            pt_path = os.path.join(sub_dir, f"{gid}.pt")
            tmp_path = pt_path + ".tmp"
            torch.save(entry, tmp_path)
            os.replace(tmp_path, pt_path)
            loaded[gid] = entry
            del cpu_payload
        if done % 20 == 0 or done == len(todo):
            el = time.perf_counter() - t0
            rate = done / max(el, 1e-6)
            eta = (len(todo) - done) / max(rate, 1e-6)
            print(
                f"  [{subset} {done:3d}/{len(todo)}] {el:.1f}s ({rate:.2f} games/s, ETA {eta:.0f}s)",
                flush=True,
            )
        if done % 10 == 0:
            gc.collect()
    return loaded


def slice_cached_game(entry: dict, t0: int, L: int, swap: bool = False) -> dict[str, torch.Tensor]:
    """Slices `[t0 : t0 + L]` (`[t0 : t0 + L - 1]` for `u_flat`), padding short games by
    repeating the final frame, and applies exact player-slot swap (`P1 <-> P2`) if `swap`."""
    T_g = len(entry["loops"])
    if T_g >= L:
        s_state = slice(t0, t0 + L)
        s_u = slice(t0, t0 + L - 1)
        pad = 0
    else:
        s_state = slice(0, T_g)
        s_u = slice(0, T_g - 1)
        pad = L - T_g

    def _sl_state(t: torch.Tensor) -> torch.Tensor:
        v = t[s_state]
        if pad > 0:
            v = torch.cat([v, v[-1:].expand(pad, *v.shape[1:])], dim=0)
        return v.flip(dims=[1]) if swap else v

    u_v = entry["u_flat"][s_u]
    if pad > 0:
        u_pad = torch.zeros((pad, *u_v.shape[1:]), dtype=u_v.dtype)
        u_v = torch.cat([u_v, u_pad], dim=0)
    if swap:
        u_v = u_v.flip(dims=[1])

    return dict(
        q_squad=_sl_state(entry["q_squad"]).float(),
        z_squad=_sl_state(entry["z_squad"]),
        active=_sl_state(entry["active"]).bool(),
        q_macro=_sl_state(entry["q_macro"]).float(),
        z_macro=_sl_state(entry["z_macro"]),
        tempo_feat=_sl_state(entry["tempo_feat"]).float(),
        gt_vis_live=_sl_state(entry["gt_vis_live"]).float(),
        u_flat=u_v.float(),
    )


def _ensure_static_map_chw(sm: dict[str, torch.Tensor]) -> None:
    """Pre-computes normalized `[4, gh, gw]` `static_grid_chw` once on CPU so per-step
    collation never scans `.max()` or `.permute()`."""
    if "static_grid_chw" in sm:
        return
    g = sm["static_grid"].float()
    if float(g[..., :2].max()) > 2.0:
        g = torch.cat([g[..., :2] / 255.0, g[..., 2:]], dim=-1)
    sm["static_grid_chw"] = g.permute(2, 0, 1).contiguous()


def _live_worker_loop(req_q, out_q, bases: dict[str, str], vocab_path: str):
    """Clean worker loop receiving `(full_specs, live_reqs)` and returning `(full_specs, metas, b_pinned)`."""
    vocab = Vocab(json.load(open(vocab_path)))
    avocab = ActionVocab()
    game_lru: dict[str, PD.Game] = {}
    lru_order: list[str] = []

    def _get_game(gid: str) -> PD.Game:
        if gid in game_lru:
            lru_order.remove(gid)
            lru_order.append(gid)
            return game_lru[gid]
        g = PD.Game(bases[gid], vocab)
        g._load_actions(avocab)
        game_lru[gid] = g
        lru_order.append(gid)
        while len(lru_order) > 4:
            old = lru_order.pop(0)
            game_lru.pop(old, None)
        return g

    n = 0
    while True:
        item = req_q.get()
        if item is None:
            return
        full_specs, live_reqs = item
        if not live_reqs:
            out_q.put((full_specs, [], None))
            continue
        ws, metas = [], []
        for gid, lp_start, swap, meta in live_reqs:
            g = _get_game(gid)
            ws.append(g.window(max(14, lp_start - 14), 18, swap=swap, step=14))
            metas.append(meta)
        b_coll_np = _to_numpy_dict(_trim_batch(PD.collate(ws, vocab)))
        out_q.put((full_specs, metas, b_coll_np))
        del ws, b_coll_np
        n += 1
        if n % 10 == 0:
            gc.collect()


class CoTrainBatchIterator:
    """Asynchronous multi-process iterator yielding `(full_specs, live_metas, b_live_pinned)`."""

    def __init__(
        self,
        train_cache: dict[str, dict],
        bases: dict[str, str],
        vocab_path: str,
        window_t: int,
        batch_size: int,
        live_batch: int,
        seed: int = 0,
        n_workers: int = 2,
        prefetch: int = 2,
    ):
        self.train_cache = train_cache
        for e in train_cache.values():
            _ensure_static_map_chw(e["static_map"])
        self.train_gids = sorted(train_cache.keys())
        self.train_lengths = {gid: len(train_cache[gid]["loops"]) for gid in self.train_gids}
        self.train_loops = {gid: train_cache[gid]["loops"].tolist() for gid in self.train_gids}
        self.window_t = window_t
        self.batch_size = batch_size
        self.live_batch = min(live_batch, batch_size)

        ctx = mp.get_context("spawn")
        self.req_q = ctx.Queue(prefetch + n_workers)
        self.out_q = ctx.Queue(prefetch + n_workers)
        self.ready = queue.Queue(min(2, prefetch))
        self.sem = threading.Semaphore(prefetch + n_workers)
        self.rng = np.random.default_rng(seed)
        self._stop = False

        self.procs = []
        for _ in range(n_workers):
            p = ctx.Process(
                target=_live_worker_loop,
                args=(self.req_q, self.out_q, bases, vocab_path),
                daemon=True,
            )
            p.start()
            self.procs.append(p)

        threading.Thread(target=self._feeder, daemon=True).start()
        threading.Thread(target=self._puller, daemon=True).start()

    def _feeder(self):
        L = self.window_t
        while not self._stop:
            self.sem.acquire()
            if self._stop:
                return
            full_specs = []
            live_reqs = []
            for b_i in range(self.batch_size):
                gid = self.train_gids[int(self.rng.integers(len(self.train_gids)))]
                T_g = self.train_lengths[gid]
                t0 = int(self.rng.integers(0, max(1, T_g - L + 1))) if T_g > L else 0
                swap = bool(self.rng.integers(0, 2))
                valid_len = min(L, T_g - t0)
                if b_i < self.live_batch and valid_len >= 5:
                    tau_rel = int(self.rng.integers(1, valid_len - 2))
                    lp_start = int(self.train_loops[gid][t0 + tau_rel])
                    live_reqs.append((gid, lp_start, swap, (b_i, gid, t0, tau_rel, swap)))
                full_specs.append((gid, t0, swap))
            self.req_q.put((full_specs, live_reqs))

    def _puller(self):
        while not self._stop:
            try:
                full_specs, metas, b_np = self.out_q.get(timeout=5.0)
                b = None
                if b_np is not None:
                    b_raw = _from_numpy_dict(b_np)
                    b = {
                        k: (v.pin_memory() if isinstance(v, torch.Tensor) and v.numel() > 2048 else v)
                        for k, v in b_raw.items()
                    }
                    del b_np

                items = []
                maps = []
                for gid, t0, swap in full_specs:
                    e = self.train_cache[gid]
                    items.append(slice_cached_game(e, t0, self.window_t, swap=swap))
                    maps.append(e["static_map"])
                seq_pinned = {
                    k: torch.stack([it_i[k] for it_i in items], dim=0).pin_memory()
                    for k in items[0]
                }
                max_h = max(int(m["static_grid_chw"].shape[1]) for m in maps)
                max_w = max(int(m["static_grid_chw"].shape[2]) for m in maps)
                st_chw = torch.zeros((len(maps), 4, max_h, max_w), dtype=torch.float32)
                for i, m in enumerate(maps):
                    g = m["static_grid_chw"]
                    _, h, w = g.shape
                    st_chw[i, :, :h, :w] = g
                st_pinned = dict(
                    static_grid_chw=st_chw.pin_memory(),
                    origin=torch.stack([m["origin"].float() for m in maps], dim=0).pin_memory(),
                    center=torch.stack([m["center"].float() for m in maps], dim=0).pin_memory(),
                    half=torch.stack([m["half"].float() for m in maps], dim=0).pin_memory(),
                )
                self.ready.put((full_specs, metas, b, seq_pinned, st_pinned))
            except queue.Empty:
                dead = [p for p in self.procs if not p.is_alive()]
                if dead and not self._stop:
                    self.ready.put(RuntimeError(f"Live reader died: exit {dead[0].exitcode}"))
                    return
            except (EOFError, OSError, ConnectionResetError):
                return

    def next_batch(self):
        item = self.ready.get()
        if isinstance(item, Exception):
            raise item
        self.sem.release()
        return item

    def close(self):
        self._stop = True
        for _ in self.procs:
            try:
                self.req_q.put_nowait(None)
            except Exception:
                pass
        for p in self.procs:
            p.terminate()


@torch.no_grad()
def evaluate_observer(
    obs_l2: OL2.ObserverWorldModelL2,
    val_cache: dict[str, dict],
    window_t: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluates:
      1. Full-sequence `forward_sequence` physical 1-step & multi-step losses vs. copy baseline.
      2. Open-loop multi-horizon Flow-Prior stochastic rollout (`+5s, +10s, +20s, +40s`) RMSE
         vs. copy baseline AND zero-shot "Stronger Hand" `P(A_{tau+40s} > 0)` alignment at
         mid/late game (`50%` and `80%` game progress).
    """
    obs_l2.eval()
    gids = sorted(val_cache.keys())
    if not gids:
        return {}

    agg_logs: dict[str, float] = {}
    n_batches = 0
    bs = 16
    for i in range(0, len(gids), bs):
        sub_gids = gids[i : i + bs]
        items = []
        maps = []
        for gid in sub_gids:
            e = val_cache[gid]
            T_g = len(e["loops"])
            t0 = max(0, (T_g - window_t) // 2)
            items.append(slice_cached_game(e, t0, window_t, swap=False))
            maps.append(e["static_map"])
        seq_b = {
            k: torch.stack([it[k] for it in items], dim=0).to(device, non_blocking=True)
            for k in items[0]
        }
        st_b = OL2.collate_static_maps(maps, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, logs = obs_l2.forward_sequence(seq_b, static_map=st_b)
        for k, v in logs.items():
            agg_logs[k] = agg_logs.get(k, 0.0) + float(v)
        n_batches += 1

    out = {k: round(v / max(1, n_batches), 5) for k, v in agg_logs.items()}

    # Open-loop 40s (H=8) Flow-Prior rollout evaluation on up to 24 unseen games
    eval_gids = gids[:24]
    se_model_40s_cost, se_copy_40s_cost = [], []
    se_model_40s_min, se_copy_40s_min = [], []
    stronger_50_ok, stronger_80_ok, n_win_games = 0, 0, 0

    for gid in eval_gids:
        e = val_cache[gid]
        T_g = len(e["loops"])
        if T_g < 18:
            continue
        st_1 = OL2.collate_static_maps([e["static_map"]], device=device)

        # Evaluate 40s physical rollout accuracy & Stronger Hand at 50% of game in ONE pass
        t_mid = max(4, min(T_g - 9, T_g // 2))
        pref_len = min(32, t_mid + 1)
        sl_full = slice_cached_game(e, t_mid + 1 - pref_len, pref_len + 8, swap=False)
        pref_seq = {
            k: sl_full[k][:pref_len].unsqueeze(0).to(device)
            for k in ("q_squad", "z_squad", "active", "q_macro", "z_macro", "tempo_feat")
        }

        with torch.autocast("cuda", dtype=torch.bfloat16):
            roll = obs_l2.rollout_observer(
                pref_seq, static_map=st_1, n_branches=16, horizon_steps=8, flow_steps=4
            )

        pred_cost_40 = roll["end_total_cost"].float().cpu()
        init_cost_0 = (sl_full["q_squad"][pref_len - 1, :, :, SP.SQ["cost"]] * sl_full["active"][pref_len - 1].float()).sum(dim=-1)
        true_cost_40 = (sl_full["q_squad"][pref_len + 7, :, :, SP.SQ["cost"]] * sl_full["active"][pref_len + 7].float()).sum(dim=-1)
        se_model_40s_cost.append((pred_cost_40 - true_cost_40).square().mean().item())
        se_copy_40s_cost.append((init_cost_0 - true_cost_40).square().mean().item())

        pred_min_40 = roll["end_minerals"].float().cpu()
        init_min_0 = sl_full["q_macro"][pref_len - 1, :, 0]
        true_min_40 = sl_full["q_macro"][pref_len + 7, :, 0]
        se_model_40s_min.append((pred_min_40 - true_min_40).square().mean().item())
        se_copy_40s_min.append((init_min_0 - true_min_40).square().mean().item())

        if e["winner"] in (1, 2):
            n_win_games += 1
            p1_win = e["winner"] == 1
            stronger_50_ok += int((float(roll["prob_stronger_hand_p1"]) >= 0.5) == p1_win)

            t_80 = max(4, int(round(0.80 * (T_g - 1))))
            p80_len = min(32, t_80 + 1)
            sl_80 = slice_cached_game(e, t_80 + 1 - p80_len, p80_len, swap=False)
            pref_80 = {
                k: sl_80[k].unsqueeze(0).to(device)
                for k in ("q_squad", "z_squad", "active", "q_macro", "z_macro", "tempo_feat")
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                roll_80 = obs_l2.rollout_observer(
                    pref_80, static_map=st_1, n_branches=16, horizon_steps=8, flow_steps=4
                )
            stronger_80_ok += int((float(roll_80["prob_stronger_hand_p1"]) >= 0.5) == p1_win)

    if se_model_40s_cost:
        out["roll40s_rmse_cost"] = round(float(np.mean(se_model_40s_cost)) ** 0.5, 4)
        out["copy40s_rmse_cost"] = round(float(np.mean(se_copy_40s_cost)) ** 0.5, 4)
        out["roll40s_rmse_min"] = round(float(np.mean(se_model_40s_min)) ** 0.5, 4)
        out["copy40s_rmse_min"] = round(float(np.mean(se_copy_40s_min)) ** 0.5, 4)
    if n_win_games > 0:
        out["stronger_hand_acc_50pct"] = round(stronger_50_ok / n_win_games, 4)
        out["stronger_hand_acc_80pct"] = round(stronger_80_ok / n_win_games, 4)
    return out


def refresh_cache_subset(
    gids_to_refresh: list[str],
    bases: dict[str, str],
    cache_dict: dict[str, dict],
    vocab_path: str,
    enc: ED.Frames,
    pooler: SP.SquadPooler,
    posterior: SP.HindsightIntentPosterior,
    device: torch.device,
    workers: int = 3,
):
    """Fast rolling `bf16` re-encode of a subset of games with bounded RAM."""
    todo = [(gid, bases[gid], 24) for gid in gids_to_refresh if gid in bases]
    if not todo:
        return
    for gid, cpu_payload in _iter_bounded_extractions(todo, vocab_path, workers=workers):
        if cpu_payload is not None:
            entry = gpu_encode_game_batches(enc, pooler, posterior, cpu_payload, device)
            _ensure_static_map_chw(entry["static_map"])
            cache_dict[gid] = entry
            del cpu_payload
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data/games_v3")
    ap.add_argument("--vocab", default="data/worldmodel/frame_cache_seq/vocab.json")
    ap.add_argument("--split", default="data/games/splits/default.json")
    ap.add_argument("--rules_tables", default="data/worldmodel/rulesets.json")
    ap.add_argument("--base_build", type=int, default=94137)
    ap.add_argument("--patch", default="5.0.14")
    ap.add_argument(
        "--l1_ckpt",
        default="data/worldmodel/runs/energy_r16_zerogram/model_step44285.pt",
    )
    ap.add_argument("--cache_dir", default="data/worldmodel/observer_cache_5014")
    ap.add_argument("--run", default="data/worldmodel/runs/observer_v1")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--window", type=int, default=48, help="48 x 5s = 4.0 min macro sequences")
    ap.add_argument("--batch", type=int, default=22)
    ap.add_argument("--live_batch", type=int, default=8, help="Live L1 windows per step for L2->L1 backprop")
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--lr_l1", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--val_every", type=int, default=300)
    ap.add_argument("--refresh_every", type=int, default=600)
    ap.add_argument("--cache_workers", type=int, default=6)
    ap.add_argument("--live_workers", type=int, default=6)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--no_grad_ckpt_l1", action="store_true")
    ap.add_argument("--max_train_games", type=int, default=0, help="0 = all 5.0.14 train games")
    ap.add_argument("--max_map_val_games", type=int, default=36)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.run, exist_ok=True)
    dev = torch.device("cuda")

    vocab = Vocab(json.load(open(args.vocab)))
    tables = ED.Tables(args.base_build, rulesets=args.rules_tables)
    split = json.load(open(args.split))

    train_ids = set(split["train"]) - set(tp.EXCLUDED)
    val_ids = set(split["held_out_games"]) - set(tp.EXCLUDED)
    map_val_ids = set(split.get("held_out_map_games", [])) - set(tp.EXCLUDED)

    train_bases = tp.of_patch(PD.base_paths(args.games, train_ids), args.patch)
    val_bases = tp.of_patch(PD.base_paths(args.games, val_ids), args.patch)
    map_val_bases_all = tp.of_patch(PD.base_paths(args.games, map_val_ids), args.patch)

    if args.max_train_games > 0:
        train_bases = dict(sorted(train_bases.items())[: args.max_train_games])
    map_val_bases = dict(sorted(map_val_bases_all.items())[: args.max_map_val_games])

    print(
        f"Splits ({args.patch}): train={len(train_bases)}, "
        f"held_out_games={len(val_bases)}, held_out_map_games={len(map_val_bases)}",
        flush=True,
    )

    # 1. Initialize L1 Frames + SquadPooler + HindsightIntentPosterior + ObserverWorldModelL2
    enc = ED.Frames(vocab, tables, 256, 32, attn_layers=6, dh=64, ff=768).to(dev)
    enc.grad_ckpt = not bool(args.no_grad_ckpt_l1)
    l1_ck = torch.load(args.l1_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = enc.load_state_dict(l1_ck["enc"], strict=False)
    print(f"Warm-started L1 Frames from {args.l1_ckpt}: missing={missing}, unexpected={unexpected}", flush=True)
    del l1_ck

    pooler = SP.SquadPooler(tables, d_l1=256).to(dev)
    posterior = SP.HindsightIntentPosterior().to(dev)
    obs_l2 = OL2.ObserverWorldModelL2(rollout_steps=4).to(dev)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        enc.load_state_dict(ck["enc"])
        pooler.load_state_dict(ck["pooler"])
        posterior.load_state_dict(ck["posterior"])
        obs_l2.load_state_dict(ck["obs_l2"])
        start_step = int(ck.get("step", 0))
        print(f"Resumed full Observer checkpoint {args.resume} at step {start_step}", flush=True)

    if not args.no_compile:
        enc.compile_parts()
        obs_l2.compile_parts()
        pooler_fn = torch.compile(pooler, dynamic=True)
        posterior_fn = torch.compile(posterior, dynamic=True)
        print("Enabled torch.compile + bf16 Tensor Cores + TF32 across L1, Pooler, Posterior, and L2", flush=True)
    else:
        pooler_fn = pooler
        posterior_fn = posterior

    # 2. Build or load initial bf16 Macro Summary Cache
    val_cache = build_or_load_cache(
        "held_out_games", val_bases, args.cache_dir, args.vocab, enc, pooler, posterior, dev, workers=args.cache_workers
    )
    map_val_cache = build_or_load_cache(
        "held_out_map_games", map_val_bases, args.cache_dir, args.vocab, enc, pooler, posterior, dev, workers=args.cache_workers
    )
    train_cache = build_or_load_cache(
        "train", train_bases, args.cache_dir, args.vocab, enc, pooler, posterior, dev, workers=args.cache_workers
    )

    # 3. Optimizer with differential LR (gentle adaptation on pretrained Frames body, full LR on new modules)
    l1_backbone_params = [
        p for n, p in enc.named_parameters() if p.requires_grad and not n.startswith("grp_c")
    ]
    pooler_post_params = (
        [p for n, p in enc.named_parameters() if p.requires_grad and n.startswith("grp_c")]
        + list(pooler.parameters())
        + list(posterior.parameters())
    )
    l2_params = list(obs_l2.parameters())
    new_params = pooler_post_params + l2_params
    opt = torch.optim.AdamW(
        [
            {"params": l1_backbone_params, "lr": args.lr_l1, "base_lr": args.lr_l1},
            {"params": new_params, "lr": args.lr, "base_lr": args.lr},
        ],
        weight_decay=args.wd,
        fused=True,
    )
    print(
        f"Trainable params: L1_backbone={sum(p.numel() for p in l1_backbone_params)/1e6:.2f}M, "
        f"New(grp_c+Pooler+Post+L2)={sum(p.numel() for p in new_params)/1e6:.2f}M",
        flush=True,
    )

    log_path = os.path.join(args.run, "train.jsonl")
    eval_path = os.path.join(args.run, "eval.jsonl")

    # Initial step-0 baseline evaluation
    if start_step == 0:
        ev_games = evaluate_observer(obs_l2, val_cache, args.window, dev)
        ev_maps = evaluate_observer(obs_l2, map_val_cache, args.window, dev)
        print(json.dumps({"step": 0, "split": "held_out_games", **ev_games}), flush=True)
        print(json.dumps({"step": 0, "split": "held_out_map_games", **ev_maps}), flush=True)
        with open(eval_path, "a") as f:
            f.write(json.dumps({"step": 0, "held_out_games": ev_games, "held_out_map_games": ev_maps}) + "\n")

    # 4. Launch background live-window prefetcher for Sparse Live-Window Co-Training
    it = CoTrainBatchIterator(
        train_cache=train_cache,
        bases=train_bases,
        vocab_path=args.vocab,
        window_t=args.window,
        batch_size=args.batch,
        live_batch=args.live_batch,
        seed=args.seed + start_step,
        n_workers=args.live_workers,
        prefetch=4,
    )

    refresh_cursor = 0
    sorted_train_gids = sorted(train_cache.keys())
    t_loop_start = time.perf_counter()
    best_val_loss = float("inf")

    try:
        for step in range(start_step + 1, args.steps + 1):
            enc.train(), pooler.train(), posterior.train(), obs_l2.train()
            full_specs, live_metas, b_live_cpu, seq_cpu, st_cpu = it.next_batch()
            seq_b = {k: v.to(dev, non_blocking=True) for k, v in seq_cpu.items()}
            st_b = {k: v.to(dev, non_blocking=True) for k, v in st_cpu.items()}

            aux_post_loss = seq_b["q_squad"].sum() * 0.0
            pending_wb = None
            if b_live_cpu is not None and len(live_metas) > 0:
                b_live = ED.to_device(b_live_cpu, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ctx_live = enc.encode_window_context(b_live, list(range(1, 18)))
                    prev_state_live = {
                        k: torch.stack(
                            [seq_b[k][m[0], m[3] - 1].detach() for m in live_metas], dim=0
                        )
                        for k in ("q_squad", "z_squad", "active")
                    }
                    s_live_0 = pooler_fn(
                        b_live, ctx_live, f_start=0, f_end=0, prev_state=prev_state_live, batch_frame_end=1
                    )
                    s_live_1 = pooler_fn(
                        b_live, ctx_live, f_start=0, f_end=8, prev_state=s_live_0, batch_frame_end=9
                    )
                    s_live_2 = pooler_fn(
                        b_live, ctx_live, f_start=8, f_end=16, prev_state=s_live_1, batch_frame_end=17
                    )
                    post_0 = posterior_fn(
                        s_live_0, s_live_1, b_live, ctx_live, f_indices=list(range(2, 10)), ctx_indices=list(range(0, 8))
                    )
                    post_1 = posterior_fn(
                        s_live_1, s_live_2, b_live, ctx_live, f_indices=list(range(10, 18)), ctx_indices=list(range(8, 16))
                    )
                aux_post_loss = post_0["aux_loss"].float() + post_1["aux_loss"].float()

                # Splice live states & intents out-of-place on GPU (ZERO mid-forward .cpu() stalls!)
                q_sq_rows = list(seq_b["q_squad"].unbind(0))
                z_sq_rows = list(seq_b["z_squad"].unbind(0))
                act_rows = list(seq_b["active"].unbind(0))
                q_mac_rows = list(seq_b["q_macro"].unbind(0))
                z_mac_rows = list(seq_b["z_macro"].unbind(0))
                u_rows = list(seq_b["u_flat"].unbind(0))
                tmp_rows = list(seq_b["tempo_feat"].unbind(0))

                for li, (b_idx, gid, t0, tau_rel, swap) in enumerate(live_metas):
                    def _sp2(row_t, v0, v1):
                        pair = torch.stack([v0, v1], dim=0).to(row_t.dtype)
                        return torch.cat([row_t[:tau_rel], pair, row_t[tau_rel + 2 :]], dim=0)

                    q_sq_rows[b_idx] = _sp2(q_sq_rows[b_idx], s_live_0["q_squad"][li], s_live_1["q_squad"][li])
                    z_sq_rows[b_idx] = _sp2(z_sq_rows[b_idx], s_live_0["z_squad"][li], s_live_1["z_squad"][li])
                    act_rows[b_idx] = _sp2(act_rows[b_idx], s_live_0["active"][li], s_live_1["active"][li])
                    q_mac_rows[b_idx] = _sp2(q_mac_rows[b_idx], s_live_0["q_macro"][li], s_live_1["q_macro"][li])
                    z_mac_rows[b_idx] = _sp2(z_mac_rows[b_idx], s_live_0["z_macro"][li], s_live_1["z_macro"][li])
                    tmp_rows[b_idx] = _sp2(tmp_rows[b_idx], post_0["tempo_feat"][li], post_1["tempo_feat"][li])
                    if tau_rel + 2 <= u_rows[b_idx].shape[0]:
                        u_rows[b_idx] = _sp2(u_rows[b_idx], post_0["u_flat"][li], post_1["u_flat"][li])

                seq_b["q_squad"] = torch.stack(q_sq_rows, dim=0)
                seq_b["z_squad"] = torch.stack(z_sq_rows, dim=0)
                seq_b["active"] = torch.stack(act_rows, dim=0)
                seq_b["q_macro"] = torch.stack(q_mac_rows, dim=0)
                seq_b["z_macro"] = torch.stack(z_mac_rows, dim=0)
                seq_b["u_flat"] = torch.stack(u_rows, dim=0)
                seq_b["tempo_feat"] = torch.stack(tmp_rows, dim=0)

                pending_wb = (
                    live_metas,
                    torch.stack([s_live_0["q_squad"].detach(), s_live_1["q_squad"].detach()], dim=1).float(),
                    torch.stack([s_live_0["z_squad"].detach(), s_live_1["z_squad"].detach()], dim=1).to(torch.bfloat16),
                    torch.stack([s_live_0["active"].detach(), s_live_1["active"].detach()], dim=1).bool(),
                    torch.stack([s_live_0["q_macro"].detach(), s_live_1["q_macro"].detach()], dim=1).float(),
                    torch.stack([s_live_0["z_macro"].detach(), s_live_1["z_macro"].detach()], dim=1).to(torch.bfloat16),
                    torch.stack([post_0["u_flat"].detach(), post_1["u_flat"].detach()], dim=1).float(),
                    torch.stack([post_0["tempo_feat"].detach(), post_1["tempo_feat"].detach()], dim=1).float(),
                )

            with torch.autocast("cuda", dtype=torch.bfloat16):
                l2_loss, logs = obs_l2.forward_sequence(seq_b, static_map=st_b)
            loss = l2_loss.float() + 0.15 * aux_post_loss
            if not torch.isfinite(loss):
                print(f"[step {step}] WARNING: non-finite loss={loss.item()}, skipping step", flush=True)
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn_l1 = float(torch.nn.utils.clip_grad_norm_(l1_backbone_params, 1.0))
            gn_pp = float(torch.nn.utils.clip_grad_norm_(pooler_post_params, 1.5))
            gn_l2 = float(torch.nn.utils.clip_grad_norm_(l2_params, 1.5))
            grad_norm = (gn_l1 * gn_l1 + gn_pp * gn_pp + gn_l2 * gn_l2) ** 0.5
            if not math.isfinite(grad_norm):
                print(
                    f"[step {step}] WARNING: non-finite grad_norm (l1={gn_l1}, pp={gn_pp}, l2={gn_l2}), skipping step",
                    flush=True,
                )
                opt.zero_grad(set_to_none=True)
                continue

            if step <= args.warmup:
                lr_scale = step / max(1, args.warmup)
            else:
                prog = (step - args.warmup) / max(1, args.steps - args.warmup)
                lr_scale = 0.10 + 0.45 * (1.0 + math.cos(math.pi * prog))
            for pg in opt.param_groups:
                pg["lr"] = pg["base_lr"] * lr_scale
            opt.step()

            # Single batched post-step CPU write-back (after backward + fused AdamW!)
            if pending_wb is not None:
                wb_metas, wb_qsq, wb_zsq, wb_act, wb_qmac, wb_zmac, wb_u, wb_tmp = (
                    pending_wb[0],
                    *(t.cpu() for t in pending_wb[1:]),
                )
                for li, (_, gid, t0, tau_rel, swap) in enumerate(wb_metas):
                    g_entry = train_cache[gid]
                    abs_t = t0 + tau_rel
                    if abs_t + 2 <= g_entry["u_flat"].shape[0]:
                        fl = (lambda x: x.flip(dims=[1])) if swap else (lambda x: x)
                        g_entry["q_squad"][abs_t : abs_t + 2] = fl(wb_qsq[li])
                        g_entry["z_squad"][abs_t : abs_t + 2] = fl(wb_zsq[li])
                        g_entry["active"][abs_t : abs_t + 2] = fl(wb_act[li])
                        g_entry["q_macro"][abs_t : abs_t + 2] = fl(wb_qmac[li])
                        g_entry["z_macro"][abs_t : abs_t + 2] = fl(wb_zmac[li])
                        g_entry["u_flat"][abs_t : abs_t + 2] = fl(wb_u[li])
                        g_entry["tempo_feat"][abs_t : abs_t + 2] = fl(wb_tmp[li])

            if step % 25 == 0 or step == 1:
                el = time.perf_counter() - t_loop_start
                steps_done = step - start_step
                rec = {
                    "step": step,
                    "sec_per_step": round(el / max(1, steps_done), 3),
                    "lr": round(args.lr * lr_scale, 6),
                    "grad_norm": round(grad_norm, 3),
                    "aux_post": round(float(aux_post_loss.detach()), 4),
                    **{k: round(float(v), 4) for k, v in logs.items()},
                }
                print(json.dumps(rec), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

            del seq_b, st_b, loss, l2_loss

            # Rolling full-game cache refresh so long prefixes track evolved L1 + SquadPooler
            if args.refresh_every > 0 and step % args.refresh_every == 0 and step < args.steps:
                torch.cuda.empty_cache()
                t_ref = time.perf_counter()
                n_ref = min(32, len(sorted_train_gids))
                batch_gids = [
                    sorted_train_gids[(refresh_cursor + k) % len(sorted_train_gids)]
                    for k in range(n_ref)
                ]
                refresh_cursor = (refresh_cursor + n_ref) % len(sorted_train_gids)
                refresh_cache_subset(
                    batch_gids, train_bases, train_cache, args.vocab, enc, pooler, posterior, dev, workers=args.cache_workers
                )
                refresh_cache_subset(
                    sorted(val_cache.keys()), val_bases, val_cache, args.vocab, enc, pooler, posterior, dev, workers=args.cache_workers
                )
                refresh_cache_subset(
                    sorted(map_val_cache.keys()), map_val_bases, map_val_cache, args.vocab, enc, pooler, posterior, dev, workers=args.cache_workers
                )
                print(
                    f"[step {step}] Refreshed {n_ref} train + all held-out games in {time.perf_counter() - t_ref:.1f}s",
                    flush=True,
                )

            if step % args.val_every == 0 or step == args.steps:
                torch.cuda.empty_cache()
                if step == args.steps and args.refresh_every > 0:
                    refresh_cache_subset(
                        sorted(val_cache.keys()), val_bases, val_cache, args.vocab, enc, pooler, posterior, dev, workers=args.cache_workers
                    )
                    refresh_cache_subset(
                        sorted(map_val_cache.keys()), map_val_bases, map_val_cache, args.vocab, enc, pooler, posterior, dev, workers=args.cache_workers
                    )
                ev_games = evaluate_observer(obs_l2, val_cache, args.window, dev)
                ev_maps = evaluate_observer(obs_l2, map_val_cache, args.window, dev)
                print(json.dumps({"step": step, "split": "held_out_games", **ev_games}), flush=True)
                print(json.dumps({"step": step, "split": "held_out_map_games", **ev_maps}), flush=True)
                with open(eval_path, "a") as f:
                    f.write(
                        json.dumps({"step": step, "held_out_games": ev_games, "held_out_map_games": ev_maps})
                        + "\n"
                    )

                ckpt_payload = dict(
                    step=step,
                    enc=enc.state_dict(),
                    pooler=pooler.state_dict(),
                    posterior=posterior.state_dict(),
                    obs_l2=obs_l2.state_dict(),
                    opt=opt.state_dict(),
                    args=vars(args),
                )
                latest_pt = os.path.join(args.run, "model_latest.pt")
                torch.save(ckpt_payload, latest_pt + ".tmp")
                os.replace(latest_pt + ".tmp", latest_pt)

                if ev_games.get("loss", float("inf")) <= best_val_loss:
                    best_val_loss = ev_games["loss"]
                    best_pt = os.path.join(args.run, "model_best.pt")
                    torch.save(ckpt_payload, best_pt + ".tmp")
                    os.replace(best_pt + ".tmp", best_pt)

                if step == args.steps:
                    final_pt = os.path.join(args.run, f"model_step{step:05d}.pt")
                    torch.save(ckpt_payload, final_pt)
    finally:
        it.close()


if __name__ == "__main__":
    main()
