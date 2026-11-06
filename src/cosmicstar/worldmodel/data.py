"""Shared pieces of the training scripts: game-loop rate, the in-memory game pool, a
batch prefetcher and a device helper."""

import os
import queue
import threading

import numpy as np

LPS = 22.4  # game loops per second at "faster" speed


class GamePool:
    """Holds `size` games in memory; rotate(n) swaps n of them for games not loaded.

    background=k: a loader thread keeps k replacement games loaded ahead of time, so a
    rotation is an instant swap (loading a game with its three views takes ~5 s);
    rotate() swaps in whatever is ready and never waits. games is replaced as a whole
    list, so readers in other threads always see a consistent pool."""

    def __init__(self, paths, size, load, rng, names=None, background=0):
        self.paths, self.load, self.rng = list(paths), load, rng
        if names:  # resuming: the same games as before
            where = {os.path.basename(q): i for i, q in enumerate(self.paths)}
            idx = [where[n] for n in names if n in where]
        else:
            idx = list(rng.permutation(len(self.paths))[:size])
        self.games = [(i, load(self.paths[i])) for i in idx]
        self.ready = queue.Queue()
        self._lock = threading.Lock()
        self._pending = set()  # indices being loaded or ready
        self._want = background
        if background:
            self._lrng = np.random.RandomState(int(rng.randint(1 << 31)))
            self._wake = threading.Event()
            threading.Thread(target=self._loader, daemon=True).start()
            self._wake.set()

    def _loader(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            while True:
                with self._lock:
                    if self.ready.qsize() >= self._want:
                        break
                    taken = {i for i, _ in self.games} | self._pending
                    free = [i for i in range(len(self.paths)) if i not in taken]
                    if not free:
                        break
                    i = free[self._lrng.randint(len(free))]
                    self._pending.add(i)
                self.ready.put((i, self.load(self.paths[i])))

    def names(self):
        return [os.path.basename(self.paths[i]) for i, _ in self.games]

    def rotate(self, n):
        if self._want:  # swap in games the loader has ready; never wait for one
            games = list(self.games)
            for _ in range(n):
                try:
                    i, g = self.ready.get_nowait()
                except queue.Empty:
                    break
                slot = self.rng.randint(len(games))
                with self._lock:
                    self._pending.discard(i)
                games[slot] = (i, g)
            self.games = games
            self._wake.set()
            return
        loaded = {i for i, _ in self.games}
        free = [i for i in range(len(self.paths)) if i not in loaded]
        for _ in range(min(n, len(free))):
            slot = self.rng.randint(len(self.games))
            new = free.pop(self.rng.randint(len(free)))
            self.games[slot] = (new, self.load(self.paths[new]))

    def draw(self, n):
        return [
            self.games[i][1] for i in self.rng.choice(len(self.games), n, replace=False)
        ]


class Prefetch:
    """Builds batches on a background thread, `depth` ahead: make() -> batch. The GPU
    works on one batch while the next ones are drawn, collated and copied. Exceptions
    surface in get().
    """

    def __init__(self, make, depth=2):
        self.make, self.q, self.stop = make, queue.Queue(depth), False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self.stop:
            try:
                item = self.make()
            except BaseException as e:  # noqa: BLE001 - handed to the consumer
                item = e
            self.q.put(item)
            if isinstance(item, BaseException):
                return

    def get(self):
        item = self.q.get()
        if isinstance(item, BaseException):
            raise item
        return item


def to(b, dev):
    return {
        k: (x.to(dev, non_blocking=True) if hasattr(x, "to") else x)
        for k, x in b.items()
    }
