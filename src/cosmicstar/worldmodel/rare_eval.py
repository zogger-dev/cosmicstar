"""Rare entity states (2026-11-01): which states the rare-state evaluation set covers, how to find
them in a frame, and their metrics on the decoder's matched output. Random evaluation frames rarely
contain a blip, a cloaked or detected unit, a hallucination or a passenger, so their F1 / recall was
computed on a handful of units; scripts/build_rare_eval.py picks fixed held-out frames that contain
each state instead."""

import collections

import numpy as np
import torch

from .grid_latent import IDX, NUM_BIN, NUM_CAT

FLAGS = ("blip", "cloaked", "burrowed", "halluc", "engaged", "has_add_on")
CATS = {"cloak:cloaked": ("cloak", "cloak_cloaked"), "cloak:detected": ("cloak", "cloak_detected"),
        "inside:passenger": ("inside", "passenger"), "inside:in_gas": ("inside", "in_gas"),
        "display:hidden": ("display", "display_hidden")}
STATES = list(FLAGS) + list(CATS) + ["contained"]


def states_of(fr):
    """state -> number of entities in that state, for one (uncollated) frame."""
    num, m = fr["ent_num"], {}
    for s in FLAGS:
        m[s] = int((num[:, IDX[s]] > 0.5).sum())
    for s, (_, col) in CATS.items():
        m[s] = int((num[:, IDX[col]] > 0.5).sum())
    m["contained"] = int((fr["ent_parent"] >= 0).sum())
    return m


class RareCounts:
    """Accumulates, per state, true positives / false positives / false negatives over matched entities."""

    def __init__(self):
        self.c = collections.defaultdict(lambda: np.zeros(3))

    @torch.no_grad()
    def add(self, out, tg, pair):
        ip, it, ok = pair
        N = tg["mask"].shape[1]
        fl = lambda x: x.reshape(-1, *x.shape[2:])
        P_, T_ = (lambda k: fl(out[k])[ip]), (lambda k: fl(tg[k])[it])
        okb = ok.bool()

        def put(state, pred, truth):
            self.c[state] += [float((pred & truth & okb).sum()), float((pred & ~truth & okb).sum()),
                              float((~pred & truth & okb).sum())]
        pb, tb = P_("bin") > 0, T_("bin") > 0.5
        for s in FLAGS:
            j = NUM_BIN.index(s)
            put(s, pb[:, j], tb[:, j])
        for s, (grp, col) in CATS.items():
            k = NUM_CAT[grp].index(col)
            put(s, P_(f"cat_{grp}").argmax(-1) == k, T_(f"cat_{grp}") == k)
        pr_t = out["_ptr_t"]
        has = pr_t < N
        hit = P_("ptr").argmax(-1) == pr_t
        self.c["contained"] += [float((hit & has & okb).sum()), float((~hit & has & okb).sum()), float((~hit & has & okb).sum())]

    def summary(self):
        """state -> (recall, precision, F1, units)."""
        r = {}
        for s in STATES:
            tp, fp, fn = self.c[s]
            rec, prec = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
            r[s] = (rec, prec, 2 * tp / max(2 * tp + fp + fn, 1), int(tp + fn))
        return r
