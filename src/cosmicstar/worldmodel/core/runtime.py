"""Shared runtime, checkpoint, `torch.compile`, and host-RAM watchdog utilities."""

from __future__ import annotations

import _thread
import json
import os
import sys
import threading
import time
from typing import Any

import torch


def configure_torch_env() -> None:
    """Configure CUDA allocator, Inductor compile threads, and TF32 matmul defaults."""
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def clean_state_dict(sd: dict[str, Any]) -> dict[str, Any]:
    """Strip `torch.compile` `_orig_mod.` prefixes from checkpoint `state_dict` keys."""
    return {
        k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v
        for k, v in sd.items()
    }


def atomic_save_checkpoint(dst: str, payload: dict[str, Any]) -> str:
    """Atomically write `payload` to `dst` via a sibling temporary file + `os.replace`."""
    parent = os.path.dirname(os.path.abspath(dst))
    os.makedirs(parent, exist_ok=True)
    tmp = os.path.join(parent, f".{os.path.basename(dst)}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, dst)
    return dst


def host_mem_info() -> tuple[float, float]:
    """Return `(host_used_gb, swap_free_mb)` from `/proc/meminfo` (includes `/dev/shm`)."""
    vals: dict[str, int] = {}
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


def start_mem_watchdog(
    max_ram_gb: float,
    procs_ref: list[Any],
    stop_event: threading.Event,
) -> None:
    """Poll `/proc/meminfo` every `0.5 s`; terminate workers and abort cleanly before OOM."""

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
