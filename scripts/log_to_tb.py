"""Import a train_direct.py text log into TensorBoard, for runs started before
train_direct wrote TensorBoard events itself. Writes the same tags and chart layout
train_direct does, into <run>/tb, where <run> is the run folder named in the log's
header.

    python scripts/log_to_tb.py data/worldmodel/runs/pyramid_d128_20261031_1328.log
    python scripts/log_to_tb.py <log> --force   # refresh: replace an earlier import

A run still in progress can be imported again with --force as its log grows.
"""

import argparse
import collections
import json
import math
import os
import re
import shutil

from torch.utils.tensorboard import SummaryWriter

STEP = re.compile(
    r"^step (\d+) loss ([\d.]+) \| (.*?) \| KL ([-\d.]+) \(beta ([\d.]+)\) \| grad"
    r" ([\d.]+) \| ([\d.]+)s/step "
    r"\(data ([\d.]+)s\) \| mem ([\d.]+)G"
)
EVAL = re.compile(r"^\[(\S+) step (\d+)\] k=(\d+) loops")
KL = re.compile(
    r"with commands: KL\s+([-\d.]+) nats\s+\| commands withheld: KL\s+([-\d.]+) nats"
)
COLS = ("post", "prior", "prior_mean", "persist")
EVAL_KEYS = (
    "exit acc",
    "died recall",
    "pos err",
    "pos err moving",
    "pos err touched",
    "hp mae",
    "shield mae",
    "order acc",
)
tag = lambda s_: re.sub(r"[^A-Za-z0-9_.-]+", "_", s_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="the text log train_direct.py printed")
    ap.add_argument(
        "--run", default="", help="the run folder (default: read from the log's header)"
    )
    ap.add_argument(
        "--force", action="store_true", help="replace <run>/tb if it exists"
    )
    args = ap.parse_args()
    lines = open(args.log).read().splitlines()
    run = args.run or next(
        ln.rsplit("|", 1)[1].strip() for ln in lines if " train games (pool " in ln
    )
    out = os.path.join(run, "tb")
    if os.path.exists(out):
        if not args.force:
            raise SystemExit(f"{out} exists (pass --force to replace it)")
        shutil.rmtree(out)
    cfg = json.load(open(os.path.join(run, "config.json")))
    lr_at = (
        lambda s: cfg["lr"]
        * min(1, (s + 1) / 200)
        * 0.5
        * (1 + math.cos(math.pi * min(1.0, s / cfg["steps"])))
    )

    train, evals = [], collections.defaultdict(
        dict
    )  # evals[(name, step, k)] = {tag suffix: value}
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = STEP.match(ln)
        if m:
            s = int(m.group(1))
            vals = {
                "loss": float(m.group(2)),
                "kl": float(m.group(4)),
                "gn": float(m.group(6)),
                "data_s": float(m.group(8)),
            }
            vals.update(
                {k_: float(v_) for k_, v_ in re.findall(r"(\w+) ([\d.]+)", m.group(3))}
            )
            train.append(
                (s, vals, float(m.group(5)), float(m.group(7)), float(m.group(9)))
            )
        m = EVAL.match(ln)
        if m and i + 3 + len(EVAL_KEYS) <= len(lines):
            name, s, k = m.group(1), int(m.group(2)), int(m.group(3))
            kl = KL.search(lines[i + 1])
            rows = lines[i + 3 : i + 3 + len(EVAL_KEYS)]
            if kl and all(r.strip().startswith(key) for r, key in zip(rows, EVAL_KEYS)):
                d = evals[(name, s, k)]
                d["KL/cmds"], d["KL/withheld"] = float(kl.group(1)), float(kl.group(2))
                for key, r in zip(EVAL_KEYS, rows):
                    for cmd, half in zip(
                        ("cmds", "withheld"), r.strip()[len(key) :].split("|")
                    ):
                        nums = [float(x) for x in half.split()]
                        for c, v_ in zip(
                            COLS, nums
                        ):  # died recall has no persistence column
                            d[f"{tag(key)}/{cmd}_{c}"] = v_
                i += 3 + len(EVAL_KEYS)
                continue
        i += 1

    tb = SummaryWriter(out)
    tb.add_text("config", "```\n" + json.dumps(cfg, indent=1) + "\n```")
    names = sorted({n for n, _, _ in evals})
    ks = sorted({k for _, _, k in evals})
    tb.add_custom_scalars(
        {
            f"{name} {key}": {
                f"k={k} loops": ["Multiline", [f"^{tag(name)}/{tag(key)}/k{k:02d}/"]]
                for k in ks
            }
            for name in names
            for key in EVAL_KEYS + ("KL",)
        }
    )
    for s, vals, beta, sps, mem in train:
        for k_, v_ in vals.items():
            tb.add_scalar(f"train/{k_}", v_, s)
        tb.add_scalar("train/lr", lr_at(s), s)
        tb.add_scalar("train/beta", beta, s)
        tb.add_scalar("perf/s_per_step", sps, s)
        tb.add_scalar("perf/mem_gb", mem, s)
    for (name, s, k), d in evals.items():
        for suffix, v_ in d.items():
            group, rest = suffix.split("/", 1)
            tb.add_scalar(f"{tag(name)}/{group}/k{k:02d}/{rest}", v_, s)
    tb.close()
    print(
        f"{out}: {len(train)} training points (last step"
        f" {train[-1][0] if train else 0}), {len(evals)} evaluations"
        f" ({', '.join(names)}; k {ks})"
    )


if __name__ == "__main__":
    main()
