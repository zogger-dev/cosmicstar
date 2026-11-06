"""Epochs of training windows: every pass planned ahead, reproducible from a seed.

An epoch tiles each game with windows every `stride` frames, from an offset of its own; the offsets go through a permutation per game, so `stride`
epochs start a window at every frame of every game exactly once, and no window repeats before that. The model sees no absolute time, so the
stride is invisible to it, and offsets differ by game, so no alignment is special.

A window's size is its rows: the units alive in any of its frames, worked out from each game's index (first and last frame per row) without reading
the game. Windows of similar size share a pass, so a pass pads little: buckets of `width` rows, the thin ends merged until each bucket has windows
from `min_games` games. A pass holds as many windows as fit the token budget at its bucket's largest window, at most `seqs`.

Spread evenly through an epoch:
- buckets: each bucket's passes at evenly spaced places, and the passes of one update from different buckets
- games: each game's windows of a bucket at evenly spaced passes of that bucket; never twice in a pass, and not again within `recent` passes where
  another game can stand in
- maps: a pass takes distinct maps where it can
- time in game (thirds) and effects present: from a game, the window whose kind was seen longest ago
"""

import zlib

import numpy as np

from . import statefile
from .physics_data import SOURCES, VIEWS

THIRDS = 3


def game_windows(base, L):
    """For each start frame t0 in [0, F - L): the window's rows, and whether it has effects in either view."""
    ix = np.load(f"{base}.index.npz")
    F = int(ix["F"])
    t0 = np.arange(max(1, F - L))
    rows = np.zeros(len(t0), np.int32)
    for s in SOURCES:
        first, last = np.sort(ix[f"first{s}"]), np.sort(ix[f"last{s}"])
        rows += (
            np.searchsorted(first, t0 + L, "left") - np.searchsorted(last, t0, "left")
        ).astype(np.int32)
    eff = np.zeros(len(t0), bool)
    for s in VIEWS:
        d = statefile.load(f"{base}.p{s}.cst")
        if "e_frame_start" in d.files:
            st = d["e_frame_start"]
            hi = np.minimum(t0 + L, len(st) - 1)
            eff |= st[hi] > st[np.minimum(t0, len(st) - 1)]
    return F, rows, eff


class Plan:
    """All windows of all games, bucketed. passes(epoch) -> the epoch's passes, each [(game, t0, swap), ...]."""

    def __init__(
        self,
        bases,
        maps,
        L,
        seed=0,
        stride=14,
        width=32,
        min_games=40,
        tokens=3900,
        seqs=16,
        per_window=259,
        accum=2,
        recent=8,
    ):
        self.ids = sorted(bases)
        self.L, self.seed, self.stride, self.accum, self.recent = (
            L,
            seed,
            stride,
            accum,
            recent,
        )
        self.F, self.rows, self.eff = {}, {}, {}
        for g in self.ids:
            self.F[g], self.rows[g], self.eff[g] = game_windows(bases[g], L)
        self.map = {g: maps[g] for g in self.ids}
        # buckets: the raw ones with windows from min_games games or more; anything below or above joins the end bucket
        raw = {}
        for g in self.ids:
            for b in np.unique(self.rows[g] // width).tolist():
                raw.setdefault(b, set()).add(g)
        full = [b for b in sorted(raw) if len(raw[b]) >= min_games]
        self.lo, self.hi, self.width = full[0], full[-1], width
        n = self.hi - self.lo + 1
        top = np.zeros(n, np.int64)
        for g in self.ids:
            np.maximum.at(top, self.bucket(self.rows[g]), self.rows[g])
        self.top = top
        self.cap = np.clip(tokens // (top + per_window), 1, seqs)

    def bucket(self, rows):
        return np.clip(np.asarray(rows) // self.width, self.lo, self.hi) - self.lo

    def offset(self, g, epoch):
        """The game's offset this epoch: a permutation of 0..stride-1 per game, a new one each cycle of stride epochs."""
        perm = np.random.default_rng(
            [self.seed, zlib.crc32(g.encode()), epoch // self.stride]
        ).permutation(self.stride)
        return int(perm[epoch % self.stride])

    def windows(self, epoch):
        """The epoch's windows: arrays of game index, t0, bucket, kind (third * 2 + effects)."""
        gi, t0s = [], []
        for i, g in enumerate(self.ids):
            t = np.arange(self.offset(g, epoch), len(self.rows[g]), self.stride)
            gi.append(np.full(len(t), i)), t0s.append(t)
        gi, t0 = np.concatenate(gi), np.concatenate(t0s)
        rows = np.concatenate([self.rows[g][t0s[i]] for i, g in enumerate(self.ids)])
        eff = np.concatenate([self.eff[g][t0s[i]] for i, g in enumerate(self.ids)])
        F = np.array([self.F[g] for g in self.ids])[gi]
        third = np.minimum(t0 * THIRDS // np.maximum(F - self.L, 1), THIRDS - 1)
        return gi, t0, self.bucket(rows), third * 2 + eff

    def sizes(self, epoch):
        """Windows per bucket and passes per bucket: as few passes as the caps allow, windows shared out evenly among them."""
        _, _, b, _ = self.windows(epoch)
        n = np.bincount(b, minlength=len(self.top))
        return n, -(-n // self.cap)

    def n_passes(self, epoch):
        return int(self.sizes(epoch)[1].sum())

    def locate(self, k):
        """Pass k counted from the first epoch -> (epoch, pass within it)."""
        e = 0
        while k >= (n := self.n_passes(e)):
            k -= n
            e += 1
        return e, k

    def passes(self, epoch):
        rng = np.random.default_rng([self.seed, epoch, 1])
        gi, t0, b, kind = self.windows(epoch)
        n, P = self.sizes(epoch)
        # the order of buckets: each bucket's passes evenly spaced, then the passes of one update from different buckets
        pos = np.concatenate([(np.arange(p) + rng.random()) / p for p in P])
        which = np.repeat(np.arange(len(P)), P)
        order = which[np.argsort(pos, kind="stable")].tolist()
        for u in range(0, len(order), self.accum):
            for j in range(u + 1, min(u + self.accum, len(order))):
                if order[j] in order[u:j]:
                    k = next(
                        (
                            k
                            for k in range(j + 1, len(order))
                            if order[k] not in order[u:j]
                        ),
                        None,
                    )
                    if k is not None:
                        order[j], order[k] = order[k], order[j]
        # each bucket's windows by game, in random order; each game's next turn evenly spaced through the bucket's passes
        G = len(self.ids)
        pool = {}
        for bk in range(len(P)):
            sel = np.nonzero(b == bk)[0]
            sel = sel[rng.permutation(len(sel))]
            per = [[] for _ in range(G)]
            for w in sel.tolist():
                per[gi[w]].append(w)
            cnt = np.array([len(x) for x in per], np.float64)
            due = np.where(cnt > 0, rng.random(G) * P[bk] / np.maximum(cnt, 1), np.inf)
            pool[bk] = dict(
                per=per,
                cnt=cnt,
                gap=P[bk] / np.maximum(cnt, 1),
                due=due,
                left=n[bk],
                passes=P[bk],
            )
        maps = np.array([self.map[g] for g in self.ids])
        seen_kind = np.full(THIRDS * 2, -1.0)
        last = np.full(G, -(10**9))  # the pass each game was last in, any bucket
        out, swaps = [], rng.random(len(gi)) < 0.5
        for p, bk in enumerate(order):
            s = pool[bk]
            size = -(
                -s["left"] // s["passes"]
            )  # windows shared evenly over the passes left
            s["left"] -= size
            s["passes"] -= 1
            cand = np.argsort(s["due"], kind="stable")[: max(6 * size, 32)]
            cand = cand[np.isfinite(s["due"][cand])].tolist()
            # a game with more windows left than the bucket has passes after this one must be in this one, or it would repeat in a pass later
            must = np.nonzero(s["cnt"] > s["passes"])[0]
            took = must[np.argsort(-s["cnt"][must], kind="stable")][:size].tolist()
            took_maps = {maps[c] for c in took}
            fresh = lambda c: p - last[c] > self.recent
            for want_map, want_fresh in (
                (True, True),
                (False, True),
                (True, False),
                (False, False),
            ):  # in order of due: distinct maps, games not just seen
                for c in cand:
                    if (
                        len(took) < size
                        and c not in took
                        and (not want_fresh or fresh(c))
                        and (not want_map or maps[c] not in took_maps)
                    ):
                        took.append(c), took_maps.add(maps[c])
            if (
                len(took) < size
            ):  # fewer games left than windows: a game again, the ones with most left
                left = {
                    c: len(s["per"][c]) - took.count(c) for c in range(G) if s["per"][c]
                }
                while len(took) < size:
                    c = max(left, key=left.get)
                    took.append(c)
                    left[c] -= 1
            wins = []
            for c in took:
                lst = s["per"][c]
                k = min(
                    range(len(lst)), key=lambda i: seen_kind[kind[lst[i]]]
                )  # the kind of window seen longest ago
                w = lst.pop(k)
                seen_kind[kind[w]] = p
                last[c] = p
                s["cnt"][c] -= 1
                wins.append((self.ids[gi[w]], int(t0[w]), bool(swaps[w])))
                s["due"][c] = s["due"][c] + s["gap"][c] if lst else np.inf
            out.append(wins)
        return out
