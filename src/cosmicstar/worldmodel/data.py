"""Shared pieces of the training scripts: game-loop rate, the in-memory game pool and a
device helper."""
import os

LPS = 22.4  # game loops per second at "faster" speed


class GamePool:
    """Holds `size` games in memory; rotate(n) swaps n of them for games not loaded."""

    def __init__(self, paths, size, load, rng, names=None):
        self.paths, self.load, self.rng = list(paths), load, rng
        if names:  # resuming: the same games as before
            where = {os.path.basename(q): i for i, q in enumerate(self.paths)}
            idx = [where[n] for n in names if n in where]
        else:
            idx = list(rng.permutation(len(self.paths))[:size])
        self.games = [(i, load(self.paths[i])) for i in idx]

    def names(self):
        return [os.path.basename(self.paths[i]) for i, _ in self.games]

    def rotate(self, n):
        loaded = {i for i, _ in self.games}
        free = [i for i in range(len(self.paths)) if i not in loaded]
        for _ in range(min(n, len(free))):
            slot = self.rng.randint(len(self.games))
            new = free.pop(self.rng.randint(len(free)))
            self.games[slot] = (new, self.load(self.paths[new]))

    def draw(self, n):
        return [self.games[i][1] for i in self.rng.choice(len(self.games), n, replace=False)]


def to(b, dev):
    return {k: (x.to(dev, non_blocking=True) if hasattr(x, "to") else x) for k, x in b.items()}
