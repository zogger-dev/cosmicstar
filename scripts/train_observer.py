"""All-932-Game Multi-Scale Co-Training for Stochastic Observer v2 (`L1` 7-Loop Chunk Flow + `L2` 120s Perceiver World Model).

Jointly trains across all 5 StarCraft II builds (`93333, 94137, 95299, 95841, 96883`, `932` games,
`1,352` symmetric player perspectives):
  - `Frames` (`L1`, `48`-frame `step=7` = `0.3125s` causal encoder with perspective-isolated
    spatial attention masks, warm-started from `model_step44285.pt`)
  - `PerceiverSquadCompressor` (`M = 64` smooth Perceiver queries/player + Null-Sink + `M_opp = 32`
    strictly leak-free Perceived-Enemy tokens, ZERO `top-k`)
  - `HindsightOutcomePosterior` (`u_1, u_2` perspective-faithful intents + `32d` player-equivariant
    `xi_world` outcome shock + realized `combat_loss` destruction impulses)
  - `L1ChunkFlowDecoder` (`4`-step parallel `16`-frame `0.3125s` trajectory, order & spawn decoder)
  - `ObserverWorldModelL2` (purely learned gate-weighted `A = H_1 - H_2`, `WorldOutcomeFlowPrior`,
    3-stream Theory-of-Mind `h_1, h_2, h_obs` + `OpponentBeliefHead`, Dual-Mode CFG `Delta A` Prior)

Usage:
    PYTHONPATH=src:scripts .venv/bin/python scripts/train_observer.py \
        --cache_dir data/worldmodel/observer_cache_v2 \
        --run data/worldmodel/runs/observer_v2 \
        --steps 4000 --batch 24 --live_batch 6
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

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pyarrow as pa
import torch
import torch.multiprocessing as mp
import train_physics as tp

from cosmicstar.worldmodel import energy_data as ED
from cosmicstar.worldmodel import l1_chunk_decoder as LCD
from cosmicstar.worldmodel import observer_l2 as OL2
from cosmicstar.worldmodel import physics_data as PD
from cosmicstar.worldmodel import squad_pooler as SP
from cosmicstar.worldmodel.action_tokens import ActionVocab
from cosmicstar.worldmodel.vocab import Vocab

CACHE_SCHEMA_VERSION = 3


def _trim_batch(b: dict) -> dict:
    """Trims collated batch dict to keys required by `Frames`, `Perceiver`, `Posterior`, and `L1ChunkFlowDecoder`."""
    out = {k: b[k] for k in ED.BATCH_KEYS if k in b}
    if "slot_reg" in out:
        out["slot_reg"] = out["slot_reg"][..., :1, :].contiguous()
    if "slot_abil" in out:
        out["slot_abil"] = out["slot_abil"][..., : PD.QUEUE + 1].contiguous()
    return out


def _to_numpy_dict(d: dict) -> dict:
    """Converts Tensor values to NumPy arrays (losslessly bit-packing binary `grid`) so IPC Queues stay tiny."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
            if k == "grid" and arr.dtype == np.uint8:
                out[k] = ("__packbits__", int(arr.shape[-1]), np.packbits(arr, axis=-1))
            else:
                out[k] = arr
        elif isinstance(v, dict):
            out[k] = _to_numpy_dict(v)
        elif isinstance(v, list):
            out[k] = [_to_numpy_dict(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def _from_numpy_dict(d: dict) -> dict:
    """Zero-copy conversion of NumPy arrays (and bit-packed `grid`) back to CPU `torch.Tensor`s."""
    out = {}
    for k, v in d.items():
        if isinstance(v, tuple) and len(v) == 3 and v[0] == "__packbits__":
            _, c_dim, packed = v
            unpacked = np.unpackbits(packed, axis=-1)[..., :c_dim]
            out[k] = torch.from_numpy(np.ascontiguousarray(unpacked))
        elif isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v)
        elif isinstance(v, dict):
            out[k] = _from_numpy_dict(v)
        elif isinstance(v, list):
            out[k] = [_from_numpy_dict(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def _slice_ctx(ctx: dict, sl: slice) -> dict:
    """Slices batch dimension `0` across both flat Tensors and nested `perception` dict."""
    out = {}
    for k, v in ctx.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[sl]
        elif isinstance(v, dict):
            out[k] = {pk: (pv[sl] if isinstance(pv, torch.Tensor) else pv) for pk, pv in v.items()}
        else:
            out[k] = v
    return out


def _extract_one_game_np(gid: str, base: str, vocab: Vocab, avocab: ActionVocab, chunk_size: int) -> dict | None:
    """Extracts contiguous `49`-frame (`step = 7` loops = `0.3125s`, `15.0s` causal context)
    windows advancing by `2` macro `5.0s` (`112`-loop) steps per window."""
    g = PD.Game(base, vocab)
    g._load_actions(avocab)

    t_marks = list(range(112, g.F - 113, 112))
    M_marks = len(t_marks)
    if M_marks < 8:
        del g
        return None

    static_map = _to_numpy_dict(OL2.extract_rich_static_map(g))
    res = g.meta.get("end", {}).get("results", {})
    winner = 1 if res.get("1") == 1 else (2 if res.get("2") == 1 else 0)

    # Build window specs: each 49-frame window covers 2 transitions (or 1 on final even tail)
    # cursor = 0 -> t0 = 0 (covers marks 0, 1, 2)
    # cursor = 2 -> t0 = t_marks[2] - 112 = 224 (covers marks 2, 3, 4)
    win_specs = []
    cursor = 0
    while True:
        t0 = 0 if cursor == 0 else (t_marks[cursor] - 112)
        rem = M_marks - cursor
        if rem <= 3:
            # Final window! rem == 3 covers [cursor, cursor+1, cursor+2] (2 transitions);
            # rem == 2 covers [cursor, cursor+1] (1 transition).
            n_trans = rem - 1
            win_specs.append((t0, n_trans, True))
            break
        else:
            win_specs.append((t0, 2, False))
            cursor += 2

    batches = []
    for i in range(0, len(win_specs), chunk_size):
        sp_c = win_specs[i : i + chunk_size]
        ws = [g.window(t0, 49, swap=False, step=7, need_map=False) for (t0, _, _) in sp_c]
        meta_c = np.array([[n_tr, int(is_last)] for (_, n_tr, is_last) in sp_c], dtype=np.int32)
        b_coll = _trim_batch(PD.collate(ws, vocab))
        del ws
        b_np = _to_numpy_dict(b_coll)
        del b_coll
        batches.append(dict(batch=b_np, win_meta=meta_c))
    del g

    return dict(
        gid=gid,
        winner=winner,
        loops=np.array(t_marks, dtype=np.int32),
        static_map=static_map,
        batches=batches,
    )


def _bounded_extract_worker(in_q, out_q, vocab_path: str):
    torch.set_num_threads(1)
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
        pa.default_memory_pool().release_unused()


@torch.no_grad()
def gpu_encode_game_batches(
    enc: ED.Frames,
    pooler: SP.PerceiverSquadCompressor,
    posterior: SP.HindsightOutcomePosterior,
    cpu_payload: dict,
    device: torch.device,
) -> dict:
    """Encodes `49`-frame (`step=7`) GPU chunks into full v2 `bf16`/`fp32` macro sequences."""
    enc.eval(), pooler.eval(), posterior.eval()

    q_sq_all, z_sq_all, gate_all, act_all = [], [], [], []
    q_mac_all, z_mac_all = [], []
    q_opp_all, z_opp_all, g_opp_all = [], [], []
    z_omni_all, vis_all, shim_all = [], [], []
    u_all, xi_w_all, loss_all, tempo_all = [], [], [], []

    prev_seed = None  # state at mark (cursor - 1) so s_c at cursor has exact dt_loops = 112!

    def _append_state(st: dict, vl: torch.Tensor, sh: torch.Tensor):
        q_sq_all.append(st["q_squad"].detach().float())
        z_sq_all.append(st["z_squad"].detach().to(torch.bfloat16))
        gate_all.append(st["gate"].detach().to(torch.float16))
        act_all.append(st["active"].detach().bool())
        q_mac_all.append(st["q_macro"].detach().float())
        z_mac_all.append(st["z_macro"].detach().to(torch.bfloat16))
        q_opp_all.append(st["q_opp_obs"].detach().to(torch.float16))
        z_opp_all.append(st["z_opp_obs"].detach().to(torch.bfloat16))
        g_opp_all.append(st["gate_opp_obs"].detach().to(torch.float16))
        z_omni_all.append(st["z_omni_scene"].detach().to(torch.bfloat16))
        vis_all.append(vl.detach().to(torch.float16))
        shim_all.append(sh.detach().to(torch.float16))

    while cpu_payload["batches"]:
        chunk = cpu_payload["batches"].pop(0)
        b = ED.to_device(chunk["batch"], device)
        win_meta = chunk["win_meta"]
        del chunk
        B_c = b["mask"].shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = enc.encode_window_context(b, list(range(1, 49)))

            for j in range(B_c):
                n_tr = int(win_meta[j, 0])
                is_last = bool(win_meta[j, 1])
                bj = {
                    k: (v[j : j + 1] if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == B_c else v)
                    for k, v in b.items()
                }
                cj = _slice_ctx(ctx, slice(j, j + 1))

                # s_c at b-frame 16 (ctx 0..15), s_n1 at b-frame 32 (ctx 16..31)
                s_c = pooler(bj, cj, f_start=0, f_end=15, prev_state=prev_seed, batch_frame_end=16)
                s_n1 = pooler(bj, cj, f_start=16, f_end=31, prev_state=s_c, batch_frame_end=32)
                vl_c, sh_c = OL2.extract_squad_gt_perception(cj, 15, s_c)
                vl_n1, sh_n1 = OL2.extract_squad_gt_perception(cj, 31, s_n1)

                post_0 = posterior(
                    s_c, s_n1, bj, cj, f_indices=list(range(17, 33)), ctx_indices=list(range(16, 32))
                )
                _append_state(s_c, vl_c, sh_c)
                u_all.append(post_0["u_flat"].detach().float())
                xi_w_all.append(post_0["xi_world"].detach().float())
                loss_all.append(post_0["combat_loss"].detach().float())
                tempo_all.append(post_0["tempo_feat"].detach().float())

                if n_tr == 2:
                    s_n2 = pooler(bj, cj, f_start=32, f_end=47, prev_state=s_n1, batch_frame_end=48)
                    post_1 = posterior(
                        s_n1, s_n2, bj, cj, f_indices=list(range(33, 49)), ctx_indices=list(range(32, 48))
                    )
                    _append_state(s_n1, vl_n1, sh_n1)
                    u_all.append(post_1["u_flat"].detach().float())
                    xi_w_all.append(post_1["xi_world"].detach().float())
                    loss_all.append(post_1["combat_loss"].detach().float())
                    tempo_all.append(post_1["tempo_feat"].detach().float())
                    prev_seed = s_n1  # loop_end = cursor + 112 -> exact dt_loops = 112 into next window's s_c!
                    if is_last:
                        vl_n2, sh_n2 = OL2.extract_squad_gt_perception(cj, 47, s_n2)
                        _append_state(s_n2, vl_n2, sh_n2)
                        tempo_all.append(post_1["tempo_feat"].detach().float().clone())
                else:
                    # Final 1-transition window!
                    _append_state(s_n1, vl_n1, sh_n1)
                    tempo_all.append(post_0["tempo_feat"].detach().float().clone())
                    prev_seed = s_c

    loops_t = cpu_payload["loops"].long()
    T_tot = len(q_sq_all)
    assert T_tot == len(loops_t), f"State count {T_tot} != loops {len(loops_t)}"
    assert len(u_all) == T_tot - 1, f"Transition count {len(u_all)} != {T_tot - 1}"

    return dict(
        schema_version=CACHE_SCHEMA_VERSION,
        gid=cpu_payload["gid"],
        winner=cpu_payload["winner"],
        loops=loops_t,
        static_map=cpu_payload["static_map"],
        q_squad=torch.cat(q_sq_all, dim=0).cpu(),        # [T, 2, 64, 14] fp32
        z_squad=torch.cat(z_sq_all, dim=0).cpu(),        # [T, 2, 64, 128] bf16
        gate=torch.cat(gate_all, dim=0).cpu(),           # [T, 2, 64] fp16
        active=torch.cat(act_all, dim=0).cpu(),          # [T, 2, 64] bool
        q_macro=torch.cat(q_mac_all, dim=0).cpu(),       # [T, 2, 6] fp32
        z_macro=torch.cat(z_mac_all, dim=0).cpu(),       # [T, 2, 128] bf16
        q_opp_obs=torch.cat(q_opp_all, dim=0).cpu(),     # [T, 2, 32, 12] fp16
        z_opp_obs=torch.cat(z_opp_all, dim=0).cpu(),     # [T, 2, 32, 64] bf16
        gate_opp_obs=torch.cat(g_opp_all, dim=0).cpu(),  # [T, 2, 32] fp16
        z_omni_scene=torch.cat(z_omni_all, dim=0).cpu(), # [T, 128] bf16
        gt_vis_live=torch.cat(vis_all, dim=0).cpu(),     # [T, 2, 64] fp16
        gt_shimmer=torch.cat(shim_all, dim=0).cpu(),     # [T, 2, 64] fp16
        tempo_feat=torch.cat(tempo_all, dim=0).cpu(),    # [T, 2, 16] fp32
        u_flat=torch.cat(u_all, dim=0).cpu(),            # [T-1, 2, 180] fp32
        xi_world=torch.cat(xi_w_all, dim=0).cpu(),       # [T-1, 32] fp32
        combat_loss=torch.cat(loss_all, dim=0).cpu(),    # [T-1, 2] fp32
    )


def _iter_bounded_extractions(todo: list[tuple[str, str, int]], vocab_path: str, workers: int = 8):
    if not todo:
        return
    n_w = max(1, min(workers, len(todo)))
    ctx = mp.get_context("spawn")
    in_q = ctx.Queue()
    out_q = ctx.Queue(maxsize=2)
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
    pooler: SP.PerceiverSquadCompressor,
    posterior: SP.HindsightOutcomePosterior,
    device: torch.device,
    workers: int = 8,
    chunk_size: int = 12,
    force_rebuild: bool = False,
) -> dict[str, dict]:
    sub_dir = os.path.join(cache_dir, subset)
    os.makedirs(sub_dir, exist_ok=True)

    loaded: dict[str, dict] = {}
    todo = []
    for gid, base in sorted(bases.items()):
        pt_path = os.path.join(sub_dir, f"{gid}.pt")
        if not force_rebuild and os.path.exists(pt_path):
            try:
                d = torch.load(pt_path, map_location="cpu", weights_only=False)
                if (
                    d is not None
                    and d.get("schema_version") == CACHE_SCHEMA_VERSION
                    and len(d["loops"]) >= 8
                    and d["q_squad"].shape[-1] == SP.N_SQUAD_Q
                ):
                    _ensure_static_map_chw(d["static_map"])
                    loaded[gid] = d
                    continue
            except Exception:
                pass
        todo.append((gid, base, chunk_size))

    if not todo:
        print(f"[{subset}] Loaded {len(loaded)} cached v2 games from {sub_dir}", flush=True)
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
            _ensure_static_map_chw(entry["static_map"])
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
    """Slices `[t0 : t0 + L]` (`[t0 : t0 + L - 1]` for transitions), padding short games cleanly,
    and applies exact equivariant player-slot swap (`P1 <-> P2`) when `swap=True`."""
    T_g = len(entry["loops"])
    if T_g >= L:
        s_state = slice(t0, t0 + L)
        s_u = slice(t0, t0 + L - 1)
        pad = 0
    else:
        s_state = slice(0, T_g)
        s_u = slice(0, T_g - 1)
        pad = L - T_g

    def _sl_player_state(t: torch.Tensor) -> torch.Tensor:
        v = t[s_state]
        if pad > 0:
            v = torch.cat([v, v[-1:].expand(pad, *v.shape[1:])], dim=0)
        return v.flip(dims=[1]) if swap else v

    def _sl_scene_state(t: torch.Tensor) -> torch.Tensor:
        v = t[s_state]
        if pad > 0:
            v = torch.cat([v, v[-1:].expand(pad, *v.shape[1:])], dim=0)
        return v

    def _sl_trans(t: torch.Tensor, flip_p: bool = False, flip_halves: bool = False) -> torch.Tensor:
        v = t[s_u]
        if pad > 0:
            v = torch.cat([v, torch.zeros((pad, *v.shape[1:]), dtype=v.dtype)], dim=0)
        if swap:
            if flip_p:
                v = v.flip(dims=[1])
            elif flip_halves:
                v = v.view(v.shape[0], 2, -1).flip(dims=[1]).reshape(v.shape[0], -1)
        return v

    return dict(
        q_squad=_sl_player_state(entry["q_squad"]).float(),
        z_squad=_sl_player_state(entry["z_squad"]),
        gate=_sl_player_state(entry["gate"]).float(),
        active=_sl_player_state(entry["active"]).bool(),
        q_macro=_sl_player_state(entry["q_macro"]).float(),
        z_macro=_sl_player_state(entry["z_macro"]),
        q_opp_obs=_sl_player_state(entry["q_opp_obs"]).float(),
        z_opp_obs=_sl_player_state(entry["z_opp_obs"]),
        gate_opp_obs=_sl_player_state(entry["gate_opp_obs"]).float(),
        z_omni_scene=_sl_scene_state(entry["z_omni_scene"]),
        tempo_feat=_sl_player_state(entry["tempo_feat"]).float(),
        gt_vis_live=_sl_player_state(entry["gt_vis_live"]).float(),
        gt_shimmer=_sl_player_state(entry["gt_shimmer"]).float(),
        u_flat=_sl_trans(entry["u_flat"], flip_p=True).float(),
        xi_world=_sl_trans(entry["xi_world"], flip_halves=True).float(),
        combat_loss=_sl_trans(entry["combat_loss"], flip_p=True).float(),
    )


def _ensure_static_map_chw(sm: dict[str, torch.Tensor]) -> None:
    if "static_grid_chw" in sm:
        return
    g = sm["static_grid"].float()
    if float(g[..., :2].max()) > 2.0:
        g = torch.cat([g[..., :2] / 255.0, g[..., 2:]], dim=-1)
    sm["static_grid_chw"] = g.permute(2, 0, 1).contiguous()


def _live_worker_loop(req_q, out_q, bases: dict[str, str], vocab_path: str):
    """Extracts `49`-frame (`step = 7` loops) live windows for joint `L1 + Perceiver + L1ChunkFlowDecoder + L2` co-training."""
    torch.set_num_threads(1)
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
        while len(lru_order) > 2:
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
        for gid, t0_win, swap, meta in live_reqs:
            g = _get_game(gid)
            ws.append(g.window(max(0, t0_win), 49, swap=swap, step=7, need_map=False))
            metas.append(meta)
        b_coll = _trim_batch(PD.collate(ws, vocab))
        del ws
        b_coll_np = _to_numpy_dict(b_coll)
        del b_coll
        out_q.put((full_specs, metas, b_coll_np))
        del b_coll_np
        n += 1
        if n % 8 == 0:
            gc.collect()
            pa.default_memory_pool().release_unused()


class CoTrainBatchIterator:
    """Asynchronous multi-process iterator yielding pinned `(full_specs, live_metas, b_live, seq_b, st_b)`."""

    def __init__(
        self,
        train_cache: dict[str, dict],
        bases: dict[str, str],
        vocab_path: str,
        window_t: int,
        batch_size: int,
        live_batch: int,
        seed: int = 0,
        n_workers: int = 4,
        prefetch: int = 3,
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
                if b_i < self.live_batch and valid_len >= 6:
                    # Pick tau_rel in [2 .. valid_len - 3] so:
                    # - tau_rel - 2 seeds s_prev (at b-frame 16),
                    # - tau_rel - 1 is s_prev,
                    # - tau_rel is s_live_0 (at b-frame 32),
                    # - tau_rel + 1 is s_live_1 (at b-frame 48),
                    # and L1ChunkFlowDecoder trains cleanly on [b-frame 32 .. 48]!
                    tau_rel = int(self.rng.integers(2, valid_len - 2))
                    lp_prev = int(self.train_loops[gid][t0 + tau_rel - 1])
                    t0_win = max(0, lp_prev - 112)
                    live_reqs.append((gid, t0_win, swap, (b_i, gid, t0, tau_rel, swap)))
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
                max_h = max(224, *(int(m["static_grid_chw"].shape[1]) for m in maps))
                max_w = max(224, *(int(m["static_grid_chw"].shape[2]) for m in maps))
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
    """Evaluates held-out full-sequence losses + open-loop `+40s` & `+120s` rollout RMSE
    and zero-shot Positional Lead alignment (winner used strictly for diagnostic accuracy)."""
    obs_l2.eval()
    gids = sorted(val_cache.keys())
    if not gids:
        return {}

    agg_logs: dict[str, float] = {}
    n_batches = 0
    bs = 16
    for i in range(0, len(gids), bs):
        sub_gids = gids[i : i + bs]
        items, maps = [], []
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
            _, logs = obs_l2.forward_sequence(seq_b, static_map=st_b)
        for k, v in logs.items():
            agg_logs[k] = agg_logs.get(k, 0.0) + float(v)
        n_batches += 1

    out = {k: round(v / max(1, n_batches), 5) for k, v in agg_logs.items()}

    pref_keys = (
        "q_squad", "z_squad", "gate", "active", "q_macro", "z_macro",
        "q_opp_obs", "z_opp_obs", "gate_opp_obs", "z_omni_scene",
        "tempo_feat", "gt_vis_live", "gt_shimmer",
    )
    eval_gids = gids[:24]
    se_model_40s_cost, se_copy_40s_cost = [], []
    se_model_40s_min, se_copy_40s_min = [], []
    lead_50_ok, lead_80_ok, stronger_80_ok, n_win_games = 0, 0, 0, 0

    for gid in eval_gids:
        e = val_cache[gid]
        T_g = len(e["loops"])
        if T_g < 18:
            continue
        st_1 = OL2.collate_static_maps([e["static_map"]], device=device)

        t_mid = max(4, min(T_g - 9, T_g // 2))
        pref_len = min(24, t_mid + 1)
        sl_full = slice_cached_game(e, t_mid + 1 - pref_len, pref_len + 8, swap=False)
        pref_seq = {k: sl_full[k][:pref_len].unsqueeze(0).to(device) for k in pref_keys}

        with torch.autocast("cuda", dtype=torch.bfloat16):
            roll = obs_l2.rollout_observer(
                pref_seq, static_map=st_1, n_branches=16, horizon_steps=8, flow_steps=4, seed=42
            )

        pred_cost_40 = roll["end_total_cost"].float().cpu()
        init_cost_0 = sl_full["q_squad"][pref_len - 1, :, :, SP.SQ["cost"]].sum(dim=-1)
        true_cost_40 = sl_full["q_squad"][pref_len + 7, :, :, SP.SQ["cost"]].sum(dim=-1)
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
            lead_50_ok += int((float(roll["A0"]) >= 0.0) == p1_win)

            t_80 = max(4, int(round(0.80 * (T_g - 1))))
            p80_len = min(24, t_80 + 1)
            sl_80 = slice_cached_game(e, t_80 + 1 - p80_len, p80_len, swap=False)
            pref_80 = {k: sl_80[k].unsqueeze(0).to(device) for k in pref_keys}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                roll_80 = obs_l2.rollout_observer(
                    pref_80, static_map=st_1, n_branches=16, horizon_steps=16, flow_steps=4, seed=42
                )
            lead_80_ok += int((float(roll_80["A0"]) >= 0.0) == p1_win)
            stronger_80_ok += int((float(roll_80["prob_stronger_hand_p1"]) >= 0.5) == p1_win)

    if se_model_40s_cost:
        out["roll40s_rmse_cost"] = round(float(np.mean(se_model_40s_cost)) ** 0.5, 4)
        out["copy40s_rmse_cost"] = round(float(np.mean(se_copy_40s_cost)) ** 0.5, 4)
        out["roll40s_rmse_min"] = round(float(np.mean(se_model_40s_min)) ** 0.5, 4)
        out["copy40s_rmse_min"] = round(float(np.mean(se_copy_40s_min)) ** 0.5, 4)
    if n_win_games > 0:
        out["lead_now_acc_50pct"] = round(lead_50_ok / n_win_games, 4)
        out["lead_now_acc_80pct"] = round(lead_80_ok / n_win_games, 4)
        out["stronger_hand_acc_80pct"] = round(stronger_80_ok / n_win_games, 4)
    return out


def refresh_cache_subset(
    gids_to_refresh: list[str],
    bases: dict[str, str],
    cache_dict: dict[str, dict],
    vocab_path: str,
    enc: ED.Frames,
    pooler: SP.PerceiverSquadCompressor,
    posterior: SP.HindsightOutcomePosterior,
    device: torch.device,
    workers: int = 6,
):
    todo = [(gid, bases[gid], 12) for gid in gids_to_refresh if gid in bases]
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
    ap.add_argument("--patch", default="", help="Empty = all 932 games across all 5 builds!")
    ap.add_argument(
        "--l1_ckpt",
        default="data/worldmodel/runs/energy_r16_zerogram/model_step44285.pt",
    )
    ap.add_argument("--cache_dir", default="data/worldmodel/observer_cache_v2")
    ap.add_argument("--run", default="data/worldmodel/runs/observer_v2")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--window", type=int, default=48, help="48 x 5s = 240s (120s history + 120s rollout)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--live_batch", type=int, default=6, help="Live 48-frame step=7 L1+L1Chunk+L2 windows/step")
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--lr_l1", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--refresh_every", type=int, default=1000)
    ap.add_argument("--cache_workers", type=int, default=8)
    ap.add_argument("--live_workers", type=int, default=4)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--no_grad_ckpt_l1", action="store_true")
    ap.add_argument("--max_train_games", type=int, default=0, help="0 = all 676 train games (1,352 swapped)")
    ap.add_argument("--max_val_games", type=int, default=0, help="0 = all 36 held_out_games")
    ap.add_argument("--max_map_val_games", type=int, default=48)
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

    train_bases = PD.base_paths(args.games, train_ids)
    val_bases = PD.base_paths(args.games, val_ids)
    map_val_bases_all = PD.base_paths(args.games, map_val_ids)
    if args.patch:
        train_bases = tp.of_patch(train_bases, args.patch)
        val_bases = tp.of_patch(val_bases, args.patch)
        map_val_bases_all = tp.of_patch(map_val_bases_all, args.patch)

    if args.max_train_games > 0:
        train_bases = dict(sorted(train_bases.items())[: args.max_train_games])
    if args.max_val_games > 0:
        val_bases = dict(sorted(val_bases.items())[: args.max_val_games])
    if args.max_map_val_games > 0:
        map_val_bases = dict(sorted(map_val_bases_all.items())[: args.max_map_val_games])
    else:
        map_val_bases = dict(sorted(map_val_bases_all.items()))

    print(
        f"Splits (patch={args.patch or 'ALL_5_BUILDS'}): train={len(train_bases)} ({2*len(train_bases)} sym perspectives), "
        f"held_out_games={len(val_bases)}, held_out_map_games={len(map_val_bases)} (of {len(map_val_bases_all)} total)",
        flush=True,
    )

    # 1. Initialize L1 Frames + PerceiverSquadCompressor + HindsightOutcomePosterior + L1ChunkFlowDecoder + ObserverWorldModelL2
    enc = ED.Frames(vocab, tables, 256, 32, attn_layers=6, dh=64, ff=768).to(dev)
    enc.grad_ckpt = not bool(args.no_grad_ckpt_l1)
    l1_ck = torch.load(args.l1_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = enc.load_state_dict(l1_ck["enc"], strict=False)
    print(f"Warm-started L1 Frames from {args.l1_ckpt}: missing={missing}, unexpected={unexpected}", flush=True)
    del l1_ck

    pooler = SP.PerceiverSquadCompressor(tables, d_l1=256).to(dev)
    posterior = SP.HindsightOutcomePosterior().to(dev)
    decoder = LCD.L1ChunkFlowDecoder(d_l1=256, d_squad=SP.D_SQUAD_DEFAULT).to(dev)
    obs_l2 = OL2.ObserverWorldModelL2(rollout_steps=4).to(dev)

    start_step = 0
    _clean_sd = lambda sd: {
        k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v for k, v in sd.items()
    }
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        enc.load_state_dict(_clean_sd(ck["enc"]))
        pooler.load_state_dict(_clean_sd(ck["pooler"]))
        posterior.load_state_dict(_clean_sd(ck["posterior"]))
        if "decoder" in ck:
            decoder.load_state_dict(_clean_sd(ck["decoder"]))
        obs_l2.load_state_dict(_clean_sd(ck["obs_l2"]))
        start_step = int(ck.get("step", 0))
        print(f"Resumed full Observer v2 checkpoint {args.resume} at step {start_step}", flush=True)
        del ck

    if not args.no_compile:
        enc.compile_parts()
        obs_l2.compile_parts()
        print("Enabled torch.compile + bf16 Tensor Cores + TF32 across L1 & L2", flush=True)
    pooler_fn = pooler
    posterior_fn = posterior

    # 2. Build or load initial bf16 v2 Macro Summary Cache
    val_cache = build_or_load_cache(
        "held_out_games", val_bases, args.cache_dir, args.vocab, enc, pooler, posterior, dev, workers=args.cache_workers
    )
    map_val_cache = build_or_load_cache(
        "held_out_map_games", map_val_bases, args.cache_dir, args.vocab, enc, pooler, posterior, dev, workers=args.cache_workers
    )
    train_cache = build_or_load_cache(
        "train", train_bases, args.cache_dir, args.vocab, enc, pooler, posterior, dev, workers=args.cache_workers
    )

    # 3. Optimizer with differential LR
    l1_backbone_params = [
        p for n, p in enc.named_parameters() if p.requires_grad and not n.startswith("grp_c")
    ]
    pooler_post_dec_params = (
        [p for n, p in enc.named_parameters() if p.requires_grad and n.startswith("grp_c")]
        + list(pooler.parameters())
        + list(posterior.parameters())
        + list(decoder.parameters())
    )
    l2_params = list(obs_l2.parameters())
    new_params = pooler_post_dec_params + l2_params
    opt = torch.optim.AdamW(
        [
            {"params": l1_backbone_params, "lr": args.lr_l1, "base_lr": args.lr_l1},
            {"params": new_params, "lr": args.lr, "base_lr": args.lr},
        ],
        weight_decay=args.wd,
        fused=True,
    )
    if args.resume and os.path.exists(args.resume):
        ck_opt = torch.load(args.resume, map_location=dev, weights_only=False)
        if "opt" in ck_opt:
            try:
                opt.load_state_dict(ck_opt["opt"])
                print("Restored AdamW optimizer state cleanly", flush=True)
            except Exception as exc:
                print(f"Note: fresh optimizer state ({exc})", flush=True)
        del ck_opt

    print(
        f"Trainable params: L1_backbone={sum(p.numel() for p in l1_backbone_params)/1e6:.2f}M, "
        f"Perceiver+Post+L1Chunk+L2={sum(p.numel() for p in new_params)/1e6:.2f}M",
        flush=True,
    )

    log_path = os.path.join(args.run, "train.jsonl")
    eval_path = os.path.join(args.run, "eval.jsonl")

    if start_step == 0:
        ev_games = evaluate_observer(obs_l2, val_cache, args.window, dev)
        ev_maps = evaluate_observer(obs_l2, map_val_cache, args.window, dev)
        print(json.dumps({"step": 0, "split": "held_out_games", **ev_games}), flush=True)
        print(json.dumps({"step": 0, "split": "held_out_map_games", **ev_maps}), flush=True)
        with open(eval_path, "a") as f:
            f.write(json.dumps({"step": 0, "held_out_games": ev_games, "held_out_map_games": ev_maps}) + "\n")

    it = CoTrainBatchIterator(
        train_cache=train_cache,
        bases=train_bases,
        vocab_path=args.vocab,
        window_t=args.window,
        batch_size=args.batch,
        live_batch=args.live_batch,
        seed=args.seed + start_step,
        n_workers=args.live_workers,
        prefetch=2,
    )

    refresh_cursor = 0
    map_refresh_cursor = 0
    sorted_train_gids = sorted(train_cache.keys())
    sorted_map_val_gids = sorted(map_val_cache.keys())
    t_loop_start = time.perf_counter()
    best_val_loss = float("inf")
    if start_step > 0 and os.path.exists(eval_path):
        for ln in open(eval_path):
            if ln.strip():
                rec_ev = json.loads(ln)
                if rec_ev.get("step", 0) > 0 and "held_out_games" in rec_ev:
                    best_val_loss = min(best_val_loss, float(rec_ev["held_out_games"].get("loss", float("inf"))))

    try:
        for step in range(start_step + 1, args.steps + 1):
            enc.train(), pooler.train(), posterior.train(), decoder.train(), obs_l2.train()
            full_specs, live_metas, b_live_cpu, seq_cpu, st_cpu = it.next_batch()
            try:
                seq_b = {k: v.to(dev, non_blocking=True) for k, v in seq_cpu.items()}
                st_b = {k: v.to(dev, non_blocking=True) for k, v in st_cpu.items()}

                aux_post_loss = seq_b["q_squad"].sum() * 0.0
                chunk_dec_loss = seq_b["q_squad"].sum() * 0.0
                pending_wb = None

                if b_live_cpu is not None and len(live_metas) > 0:
                    b_live = ED.to_device(b_live_cpu, dev)
                    live_b_idx = [m[0] for m in live_metas]
                    st_live_map = dict(
                        static_grid_chw=st_b["static_grid_chw"][live_b_idx],
                        origin=st_b["origin"][live_b_idx],
                        center=st_b["center"][live_b_idx],
                        half=st_b["half"][live_b_idx],
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        ctx_live = enc.encode_window_context(b_live, list(range(1, 49)))
                        seed_live = {
                            k: torch.stack(
                                [seq_b[k][m[0], m[3] - 2].detach() for m in live_metas], dim=0
                            )
                            for k in ("q_squad", "z_squad", "gate", "active")
                        }
                        seed_live["loop_end"] = b_live["loops"][:, 0].float()
                        s_prev = pooler_fn(
                            b_live, ctx_live, f_start=0, f_end=15, prev_state=seed_live, batch_frame_end=16
                        )
                        s_live_0 = pooler_fn(
                            b_live, ctx_live, f_start=16, f_end=31, prev_state=s_prev, batch_frame_end=32
                        )
                        s_live_1 = pooler_fn(
                            b_live, ctx_live, f_start=32, f_end=47, prev_state=s_live_0, batch_frame_end=48
                        )
                        post_prev_0 = posterior_fn(
                            s_prev, s_live_0, b_live, ctx_live,
                            f_indices=list(range(17, 33)), ctx_indices=list(range(16, 32)),
                        )
                        post_0_1 = posterior_fn(
                            s_live_0, s_live_1, b_live, ctx_live,
                            f_indices=list(range(33, 49)), ctx_indices=list(range(32, 48)),
                        )
                        dec_out = decoder.flow_matching_loss(
                            b=b_live,
                            ctx=ctx_live,
                            f_0=32,
                            c_0=31,
                            s_0=s_live_0,
                            s_1=s_live_1,
                            u_flat=post_0_1["u_flat"],
                            xi_world=post_0_1["xi_world"],
                            static_map=st_live_map,
                        )

                        vl_live_0, sh_live_0 = OL2.extract_squad_gt_perception(ctx_live, 31, s_live_0)
                        vl_live_1, sh_live_1 = OL2.extract_squad_gt_perception(ctx_live, 47, s_live_1)

                    aux_post_loss = 0.5 * (post_prev_0["aux_loss"].float() + post_0_1["aux_loss"].float())
                    chunk_dec_loss = dec_out["loss"].float()
                    del b_live, ctx_live, s_prev, dec_out

                    # Splice live states [tau_rel, tau_rel + 1] & transitions [tau_rel - 1, tau_rel] on GPU!
                    # Detach spatial/gate keys into L2 so L2 never pulls Perceiver slots toward spatial collapse!
                    state_keys = (
                        "q_squad", "z_squad", "gate", "active", "q_macro", "z_macro",
                        "q_opp_obs", "z_opp_obs", "gate_opp_obs", "z_omni_scene",
                    )
                    detach_spatial_keys = {"q_squad", "gate", "active", "q_macro", "q_opp_obs", "gate_opp_obs"}
                    row_bufs = {k: list(seq_b[k].unbind(0)) for k in state_keys}
                    vl_rows = list(seq_b["gt_vis_live"].unbind(0))
                    sh_rows = list(seq_b["gt_shimmer"].unbind(0))
                    tmp_rows = list(seq_b["tempo_feat"].unbind(0))
                    u_rows = list(seq_b["u_flat"].unbind(0))
                    xi_rows = list(seq_b["xi_world"].unbind(0))
                    cl_rows = list(seq_b["combat_loss"].unbind(0))

                    for li, (b_idx, gid, t0, tau_rel, swap) in enumerate(live_metas):
                        def _sp2(row_t, v0, v1, off):
                            pair = torch.stack([v0, v1], dim=0).to(row_t.dtype)
                            return torch.cat([row_t[:off], pair, row_t[off + 2 :]], dim=0)

                        for sk in state_keys:
                            v0 = s_live_0[sk][li].detach() if sk in detach_spatial_keys else s_live_0[sk][li]
                            v1 = s_live_1[sk][li].detach() if sk in detach_spatial_keys else s_live_1[sk][li]
                            row_bufs[sk][b_idx] = _sp2(row_bufs[sk][b_idx], v0, v1, tau_rel)
                        vl_rows[b_idx] = _sp2(vl_rows[b_idx], vl_live_0[li].detach(), vl_live_1[li].detach(), tau_rel)
                        sh_rows[b_idx] = _sp2(sh_rows[b_idx], sh_live_0[li].detach(), sh_live_1[li].detach(), tau_rel)
                        tmp_rows[b_idx] = _sp2(
                            tmp_rows[b_idx], post_prev_0["tempo_feat"][li], post_0_1["tempo_feat"][li], tau_rel
                        )
                        if tau_rel + 1 <= u_rows[b_idx].shape[0]:
                            u_rows[b_idx] = _sp2(u_rows[b_idx], post_prev_0["u_flat"][li], post_0_1["u_flat"][li], tau_rel - 1)
                            xi_rows[b_idx] = _sp2(xi_rows[b_idx], post_prev_0["xi_world"][li], post_0_1["xi_world"][li], tau_rel - 1)
                            cl_rows[b_idx] = _sp2(cl_rows[b_idx], post_prev_0["combat_loss"][li], post_0_1["combat_loss"][li], tau_rel - 1)

                    for sk in state_keys:
                        seq_b[sk] = torch.stack(row_bufs[sk], dim=0)
                    seq_b["gt_vis_live"] = torch.stack(vl_rows, dim=0)
                    seq_b["gt_shimmer"] = torch.stack(sh_rows, dim=0)
                    seq_b["tempo_feat"] = torch.stack(tmp_rows, dim=0)
                    seq_b["u_flat"] = torch.stack(u_rows, dim=0)
                    seq_b["xi_world"] = torch.stack(xi_rows, dim=0)
                    seq_b["combat_loss"] = torch.stack(cl_rows, dim=0)

                    pending_wb = (
                        live_metas,
                        torch.stack([s_live_0["q_squad"].detach(), s_live_1["q_squad"].detach()], dim=1).float(),
                        torch.stack([s_live_0["z_squad"].detach(), s_live_1["z_squad"].detach()], dim=1).to(torch.bfloat16),
                        torch.stack([s_live_0["gate"].detach(), s_live_1["gate"].detach()], dim=1).to(torch.float16),
                        torch.stack([s_live_0["active"].detach(), s_live_1["active"].detach()], dim=1).bool(),
                        torch.stack([s_live_0["q_macro"].detach(), s_live_1["q_macro"].detach()], dim=1).float(),
                        torch.stack([s_live_0["z_macro"].detach(), s_live_1["z_macro"].detach()], dim=1).to(torch.bfloat16),
                        torch.stack([s_live_0["q_opp_obs"].detach(), s_live_1["q_opp_obs"].detach()], dim=1).to(torch.float16),
                        torch.stack([s_live_0["z_opp_obs"].detach(), s_live_1["z_opp_obs"].detach()], dim=1).to(torch.bfloat16),
                        torch.stack([s_live_0["gate_opp_obs"].detach(), s_live_1["gate_opp_obs"].detach()], dim=1).to(torch.float16),
                        torch.stack([s_live_0["z_omni_scene"].detach(), s_live_1["z_omni_scene"].detach()], dim=1).to(torch.bfloat16),
                        torch.stack([vl_live_0.detach(), vl_live_1.detach()], dim=1).to(torch.float16),
                        torch.stack([sh_live_0.detach(), sh_live_1.detach()], dim=1).to(torch.float16),
                        torch.stack([post_prev_0["u_flat"].detach(), post_0_1["u_flat"].detach()], dim=1).float(),
                        torch.stack([post_prev_0["xi_world"].detach(), post_0_1["xi_world"].detach()], dim=1).float(),
                        torch.stack([post_prev_0["tempo_feat"].detach(), post_0_1["tempo_feat"].detach()], dim=1).float(),
                    )
                    del s_live_0, s_live_1, vl_live_0, vl_live_1, sh_live_0, sh_live_1, post_prev_0, post_0_1, row_bufs, vl_rows, sh_rows, tmp_rows, u_rows, xi_rows, cl_rows

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l2_loss, logs = obs_l2.forward_sequence(seq_b, static_map=st_b)
                loss = l2_loss.float() + 0.20 * aux_post_loss + 1.00 * chunk_dec_loss
                if not torch.isfinite(loss):
                    print(f"[step {step}] WARNING: non-finite loss={loss.item()}, skipping step", flush=True)
                    opt.zero_grad(set_to_none=True)
                    continue

                opt.zero_grad(set_to_none=True)
                loss.backward()
            except torch.OutOfMemoryError as oom_exc:
                print(f"[step {step}] WARNING: caught transient CUDA OOM ({oom_exc}), clearing cache & continuing", flush=True)
                opt.zero_grad(set_to_none=True)
                for _vn in (
                    "b_live", "ctx_live", "s_prev", "s_live_0", "s_live_1",
                    "post_prev_0", "post_0_1", "dec_out", "seq_b", "st_b",
                    "loss", "l2_loss", "aux_post_loss", "chunk_dec_loss", "logs", "pending_wb",
                ):
                    locals().pop(_vn, None)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            gn_l1 = float(torch.nn.utils.clip_grad_norm_(l1_backbone_params, 1.0))
            gn_pp = float(torch.nn.utils.clip_grad_norm_(pooler_post_dec_params, 1.5))
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

            # Asynchronous post-step CPU cache write-back (with exact swap handling!)
            if pending_wb is not None:
                (
                    wb_metas,
                    wb_qsq, wb_zsq, wb_gate, wb_act, wb_qmac, wb_zmac,
                    wb_qopp, wb_zopp, wb_gopp, wb_zomni,
                    wb_vl, wb_sh,
                    wb_u, wb_xi, wb_tmp,
                ) = (pending_wb[0], *(t.cpu() for t in pending_wb[1:]))
                for li, (_, gid, t0, tau_rel, swap) in enumerate(wb_metas):
                    g_entry = train_cache[gid]
                    abs_t = t0 + tau_rel
                    if abs_t + 2 <= g_entry["q_squad"].shape[0] and abs_t + 1 <= g_entry["u_flat"].shape[0]:
                        fl = (lambda x: x.flip(dims=[1])) if swap else (lambda x: x)
                        fl_h = (lambda x: x.view(2, 2, -1).flip(dims=[1]).reshape(2, -1)) if swap else (lambda x: x)
                        g_entry["q_squad"][abs_t : abs_t + 2] = fl(wb_qsq[li])
                        g_entry["z_squad"][abs_t : abs_t + 2] = fl(wb_zsq[li])
                        g_entry["gate"][abs_t : abs_t + 2] = fl(wb_gate[li])
                        g_entry["active"][abs_t : abs_t + 2] = fl(wb_act[li])
                        g_entry["q_macro"][abs_t : abs_t + 2] = fl(wb_qmac[li])
                        g_entry["z_macro"][abs_t : abs_t + 2] = fl(wb_zmac[li])
                        g_entry["q_opp_obs"][abs_t : abs_t + 2] = fl(wb_qopp[li])
                        g_entry["z_opp_obs"][abs_t : abs_t + 2] = fl(wb_zopp[li])
                        g_entry["gate_opp_obs"][abs_t : abs_t + 2] = fl(wb_gopp[li])
                        g_entry["z_omni_scene"][abs_t : abs_t + 2] = wb_zomni[li]
                        g_entry["gt_vis_live"][abs_t : abs_t + 2] = fl(wb_vl[li])
                        g_entry["gt_shimmer"][abs_t : abs_t + 2] = fl(wb_sh[li])
                        g_entry["tempo_feat"][abs_t : abs_t + 2] = fl(wb_tmp[li])
                        g_entry["u_flat"][abs_t - 1 : abs_t + 1] = fl(wb_u[li])
                        g_entry["xi_world"][abs_t - 1 : abs_t + 1] = fl_h(wb_xi[li])

            if step % 25 == 0 or step == 1:
                el = time.perf_counter() - t_loop_start
                steps_done = step - start_step
                rec = {
                    "step": step,
                    "sec_per_step": round(el / max(1, steps_done), 3),
                    "lr": round(args.lr * lr_scale, 6),
                    "grad_norm": round(grad_norm, 3),
                    "chunk_dec": round(float(chunk_dec_loss.detach()), 4),
                    "aux_post": round(float(aux_post_loss.detach()), 4),
                    **{k: round(float(v), 4) for k, v in logs.items()},
                }
                print(json.dumps(rec), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

            del seq_b, st_b, loss, l2_loss, aux_post_loss, chunk_dec_loss, logs, pending_wb

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
                if sorted_map_val_gids:
                    n_mref = min(36, len(sorted_map_val_gids))
                    map_batch_gids = [
                        sorted_map_val_gids[(map_refresh_cursor + k) % len(sorted_map_val_gids)]
                        for k in range(n_mref)
                    ]
                    map_refresh_cursor = (map_refresh_cursor + n_mref) % len(sorted_map_val_gids)
                    refresh_cache_subset(
                        map_batch_gids, map_val_bases, map_val_cache, args.vocab, enc, pooler, posterior, dev, workers=args.cache_workers
                    )
                print(
                    f"[step {step}] Refreshed {n_ref} train + held-out games in {time.perf_counter() - t_ref:.1f}s",
                    flush=True,
                )

            if step % args.val_every == 0 or step == args.steps:
                torch.cuda.empty_cache()
                ckpt_payload = dict(
                    schema_version=CACHE_SCHEMA_VERSION,
                    step=step,
                    enc=_clean_sd(enc.state_dict()),
                    pooler=_clean_sd(pooler.state_dict()),
                    posterior=_clean_sd(posterior.state_dict()),
                    decoder=_clean_sd(decoder.state_dict()),
                    obs_l2=_clean_sd(obs_l2.state_dict()),
                    opt=opt.state_dict(),
                    args=vars(args),
                )
                latest_pt = os.path.join(args.run, "model_latest.pt")
                torch.save(ckpt_payload, latest_pt + ".tmp")
                os.replace(latest_pt + ".tmp", latest_pt)
                if step % 1000 == 0 or step == args.steps:
                    step_pt = os.path.join(args.run, f"model_step{step:05d}.pt")
                    torch.save(ckpt_payload, step_pt + ".tmp")
                    os.replace(step_pt + ".tmp", step_pt)

                if step == args.steps and args.refresh_every > 0:
                    refresh_cache_subset(
                        sorted(val_cache.keys()), val_bases, val_cache, args.vocab, enc, pooler, posterior, dev, workers=args.cache_workers
                    )
                    if sorted_map_val_gids:
                        refresh_cache_subset(
                            sorted_map_val_gids[:48], map_val_bases, map_val_cache, args.vocab, enc, pooler, posterior, dev, workers=args.cache_workers
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

                if ev_games.get("loss", float("inf")) <= best_val_loss:
                    best_val_loss = ev_games["loss"]
                    best_pt = os.path.join(args.run, "model_best.pt")
                    torch.save(ckpt_payload, best_pt + ".tmp")
                    os.replace(best_pt + ".tmp", best_pt)
                del ckpt_payload
    finally:
        it.close()


if __name__ == "__main__":
    main()
