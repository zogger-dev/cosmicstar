"""Frame cache (2026-11-01): ready-made frames on disk, so training samples moments from
every game instead of holding a dozen whole games in RAM (the in-memory pool reached ~47
GB and was OOM-killed).

A game's state file can't be read frame by frame (statefile.py: units are delta-coded
against the previous frame, maps XOR'd), and part of a frame comes from whole-game
preprocessing (bridged gas workers, containers, control groups, latest stats). So the
cache is built once: load a game with its three views, choose windows while it's
decoded, build each frame in all three views (p1, p2, the combined view) and write them
out. Training then reads single windows.

Layout: <root>/vocab.json (the vocabulary the frames were built with: frames carry
vocabulary indices) and <root>/<split>/<game id>.fcache per game:

    MAGIC, an 8-byte header length, a JSON header, then one zstd-compressed pickle per entry.

Two entry kinds:
- windows (the header's "windows": [{t0, loops, offset, size}], "length" L): L
  consecutive frames (L = 15: a moment and the 14 frames after it, 56 game loops = 2.5 s
  at the 4-loop frame step), stored as {"frames": [[p1, p2, combined] per frame]},
  compressed together (consecutive frames are nearly identical). For dynamics
  (sequences) and, one frame at a time, for the encoder.
- moments (the first cache, header "moments": [{t, loop, offset, size}]): {"views": [p1,
  p2, combined]}, a window of one.

Frames are built with the fixed player codes: the code is the player's index, 0 or 1
(neutral has its own), the same in all three views. The reader swaps the two players at
random per sample, every frame and view of the sample alike (swap_players), so an index
identifies a player within a sample and says nothing else about them.

Choice: --per_game entries per game, one per equal slice of the game (stratified over
game time, jittered), so openings, mid games and late games are all represented."""

import json
import os
import pickle
import struct

import numpy as np

from .tokenize import C_GRID

MAGIC = b"FCACHE1\n"
TEAM_KEYS = ("ent_team", "eff_team", "ply_team")
PLAYER_KEYS = ("ent_pcode", "eff_pcode", "ply_pcode", "ent_src", "eff_src")
LAYERS_SWAPPED = np.r_[
    3:6, 0:3, 6:C_GRID
]  # visible / seen / creep are per player, first


def _zstd():
    import zstandard

    return zstandard


def choose_starts(n_frames, per_game, length, rng, margin=8):
    """Stratified window starts: per_game equal slices of [margin, n_frames - margin -
    length], one jittered start each (length 1: single moments)."""
    lo, hi = margin, n_frames - margin - length + 1
    if hi - lo < per_game:
        return np.arange(lo, max(lo, hi))
    edges = np.linspace(lo, hi, per_game + 1).astype(np.int64)
    return np.array(
        [rng.randint(a, max(a + 1, b)) for a, b in zip(edges[:-1], edges[1:])]
    )


choose_moments = lambda n_frames, per_game, rng, margin=8: choose_starts(
    n_frames, per_game, 1, rng, margin
)


def write_game(path, windows, meta):
    """windows: an iterable of (t0, [loop per frame], [[p1, p2, combined] per frame]).
    Each window is compressed as soon as it's built and its frames dropped:
    uncompressed, a game's 300 windows are 13,500 frames (tens of GB across workers;
    holding them crashed the machine, 2026-11-01).
    """
    c = _zstd().ZstdCompressor(level=6)
    blobs, index, off, L = [], [], 0, 0
    for t0, loops, frames in windows:
        L = len(frames)
        b = c.compress(pickle.dumps({"frames": frames}, protocol=5))
        index.append(
            dict(t0=int(t0), loops=[int(x) for x in loops], offset=off, size=len(b))
        )
        blobs.append(b)
        off += len(b)
        del frames
    head = json.dumps(dict(meta, length=L, windows=index)).encode()
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)
    return len(index)


def read_header(path):
    with open(path, "rb") as f:
        assert f.read(len(MAGIC)) == MAGIC, path
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        return head, len(MAGIC) + 8 + n


def _swap_frame(fr):
    """One frame with players 0 and 1 exchanged: their codes, their tokens' order, their
    map layers."""
    fr = dict(fr)
    for k in TEAM_KEYS + PLAYER_KEYS:
        if k in fr:
            x = np.asarray(fr[k])
            fr[k] = np.where(x < 2, 1 - x, x).astype(x.dtype)
    for k in fr:
        if k.startswith("ply_"):
            fr[k] = fr[k][::-1].copy()
    fr["grid"] = fr["grid"][LAYERS_SWAPPED]
    return fr


def swap_players(views):
    """[p1, p2, combined] of one frame with the players exchanged: still [player 0's
    view, player 1's view, combined]."""
    a, b, c = [_swap_frame(fr) for fr in views]
    return [b, a, c]


class FrameCache:
    """The entries (windows, or single moments) of one or more splits: index them from
    the headers; read any entry; sample moments (a random frame of a random window) or
    whole windows.
    """

    def __init__(self, root, splits):
        self.root = root
        self.items = []  # (path, absolute offset, size, game index)
        self.games = []  # game ids
        self.length = 1
        for split in splits:
            d = os.path.join(root, split)
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                if not name.endswith(".fcache"):
                    continue
                p = os.path.join(d, name)
                head, base = read_header(p)
                gi = len(self.games)
                self.games.append(name[: -len(".fcache")])
                entries = head.get("windows", head.get("moments", []))
                self.length = head.get("length", 1)
                self.items += [(p, base + m["offset"], m["size"], gi) for m in entries]
        self._d = None

    def __len__(self):
        return len(self.items)

    def read(self, i):
        """Entry i with fixed codes: [[p1, p2, combined] per frame] (a moment cache: one
        frame)."""
        if self._d is None:
            self._d = _zstd().ZstdDecompressor()
        p, off, size, _ = self.items[i]
        with open(p, "rb") as f:
            f.seek(off)
            e = pickle.loads(self._d.decompress(f.read(size)))
        return e["frames"] if "frames" in e else [e["views"]]

    def sample(self, rng, n):
        """n moments, each [player 0's view, player 1's view, combined], players swapped
        at random per moment: a random frame of a random entry."""
        out = []
        for i in rng.randint(len(self.items), size=n):
            w = self.read(int(i))
            m = w[rng.randint(len(w))]
            out.append(swap_players(m) if rng.randint(2) else m)
        return out

    def sample_windows(self, rng, n):
        """n whole windows, players swapped at random per window (all its frames and
        views alike)."""
        out = []
        for i in rng.randint(len(self.items), size=n):
            w = self.read(int(i))
            out.append([swap_players(views) for views in w] if rng.randint(2) else w)
        return out


def load_vocab(root):
    from . import vocab as vocab_mod

    return vocab_mod.Vocab(json.load(open(os.path.join(root, "vocab.json"))))
