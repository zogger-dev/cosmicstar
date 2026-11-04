"""The state file format (.cst): the extractor's arrays, transformed so that zstd sees mostly
zeros, in one file with a JSON header and one compressed blob per array.

A state file is a dict of numpy arrays (scripts/extract_states.py lists them). Stored as
plain per-array zlib (np.savez_compressed) a median pro player view is 8.6 MB; the same arrays
are 3.2 MB here, bit-exact, in 0.4 s either way (zlib: 1.6 s to write; 2026-10-31,
worldmodel/README.md "Packed state files"). Three lossless transforms do the work:

    delta_id    unit columns (one value per unit row): each value minus the same unit's value
                in the previous frame, matched by u_id, so a unit that stands still is zeros.
                float32 columns are exact multiples of 1/4096 (the engine's fixed point; checked
                per column at save time, else the raw bits are delta'd), float16 columns are
                delta'd as their bit patterns, integers wrap. Decoding needs u_id and
                frame_start, so those are never delta'd by id.
    xor_frames  [F, H, W] maps (visibility, creep): each frame XOR the previous one.
    diff        monotonic start / index arrays: consecutive differences (the counts).

Everything else is stored as is. Use:

    from cosmicstar.worldmodel import statefile
    statefile.save(path, arrays)          # atomic: writes path + ".tmp", then renames
    with statefile.load(path) as d:       # a StateFile: d.files, d[name], name in d, dict(d)
        x = d["u_x"]                      # arrays are decoded on first access and cached

load() falls back to np.load for the older .npz files, which have the same interface.
"""
import json
import os

import numpy as np

MAGIC = b"CSTATE1\n"
VERSION = 1
FIXED = 4096  # the engine's fixed-point scale for positions, hp, shield, energy, ...
LEVEL = 9     # zstd level: 3.2 MB / 0.4 s on the median view; 19 gives ~12% less for 15x the time
XOR_KEYS = ("vis", "creep")
# unit columns stored without delta_id: u_id is needed to decode the others, and raw is its best
# form anyway (the engine's row order is only half stable frame to frame, so row-aligned or
# consecutive deltas are bigger; 632 KB of the 3.2 MB)
NO_DELTA = ("u_id",)


def _zstd():
    import zstandard
    return zstandard


# ------------------------------------------------------------------ the per-unit delta

def prev_rows(u_id, frame_start):
    """For every unit row, the row of the same unit in the previous frame, -1 if none."""
    n_units = int(u_id.max()) + 1 if len(u_id) else 0
    prev = np.full(len(u_id), -1, np.int64)
    pos = np.full(n_units, -1, np.int64)
    for f in range(len(frame_start) - 1):
        a, b = int(frame_start[f]), int(frame_start[f + 1])
        ids = u_id[a:b]
        if f:
            prev[a:b] = pos[ids]
        pos[:] = -1
        pos[ids] = np.arange(a, b)
    return prev


def _to_ints(v):
    """A unit column as an integer array of the same width (fixed point for exact float32,
    bit patterns otherwise). Returns (ints, how) with how in fixed | bits | int."""
    if v.dtype == np.float32:
        fixed = np.round(v.astype(np.float64) * FIXED)
        if np.all(np.isfinite(v)) and np.abs(fixed).max(initial=0) < 2 ** 31 and np.array_equal((fixed / FIXED).astype(np.float32), v):
            return fixed.astype(np.int32), "fixed"
        return v.view(np.int32), "bits"
    if v.dtype == np.float16:
        return v.view(np.int16), "bits"
    if v.dtype == np.float64:
        return v.view(np.int64), "bits"
    if v.dtype == np.bool_:
        return v.view(np.uint8), "bits"
    return v, "int"


def _from_ints(ints, how, dtype):
    dtype = np.dtype(dtype)
    if how == "fixed":
        return (ints.astype(np.float64) / FIXED).astype(np.float32)
    if how == "bits":
        return ints.view(dtype)
    return ints.astype(dtype, copy=False)


def encode_delta(cols, prev):
    """{name: int column} -> the same minus the previous frame's row (wrapping), batched."""
    out = {}
    has = prev >= 0
    src = prev[has]
    for k, v in cols.items():
        d = v.copy()
        d[has] = (v[has] - v[src]).astype(v.dtype, copy=False)  # numpy integer arithmetic wraps
        out[k] = d
    return out


def decode_delta(cols, prev, frame_start):
    """The inverse of encode_delta: frame by frame, every column at once."""
    names = list(cols)
    outs = {k: cols[k].copy() for k in names}
    for f in range(1, len(frame_start) - 1):
        a, b = int(frame_start[f]), int(frame_start[f + 1])
        p = prev[a:b]
        has = p >= 0
        if not has.any():
            continue
        src = p[has]
        for k in names:
            o = outs[k]
            o[a:b][has] = (o[a:b][has] + o[src]).astype(o.dtype, copy=False)
    return outs


# ------------------------------------------------------------------ save

def save(path, arrays, level=LEVEL):
    """Writes the arrays to path (.cst), atomically. Returns the file size in bytes."""
    zstd = _zstd()
    arrays = {k: (np.asarray(v) if not isinstance(v, np.ndarray) else v) for k, v in arrays.items()}
    n_rows = len(arrays["u_id"]) if "u_id" in arrays else -1
    unit_cols = [k for k, v in arrays.items() if k.startswith("u_") and v.ndim == 1 and len(v) == n_rows
                 and k not in NO_DELTA and not k.endswith("_start")]
    prev = prev_rows(arrays["u_id"], arrays["frame_start"]) if unit_cols else None
    unit_cols = set(unit_cols)
    header, blobs, offset = {}, [], 0
    comp = zstd.ZstdCompressor(level=level)
    for k, v in arrays.items():
        entry = dict(dtype=v.dtype.str, shape=list(v.shape), transform="raw")
        if k in unit_cols:  # one column at a time: a long game's columns together are GBs
            ints, how = _to_ints(v)
            stored = encode_delta({k: ints}, prev)[k]
            entry.update(transform="delta_id", how=how, stored_dtype=stored.dtype.str)
        elif k in XOR_KEYS and v.ndim == 3 and v.dtype == np.uint8 and len(v) > 1:
            stored = v.copy()
            stored[1:] ^= v[:-1]
            entry.update(transform="xor_frames", stored_dtype=stored.dtype.str)
        elif (k.endswith("_start") and v.ndim == 1 and np.issubdtype(v.dtype, np.integer) and len(v)
              and np.all(np.diff(v) >= 0)):
            d = np.diff(v, prepend=v[:1] * 0)
            d[0] = v[0]
            stored = d.astype(np.int32) if d.max(initial=0) < 2 ** 31 else d
            entry.update(transform="diff", stored_dtype=stored.dtype.str)
        else:
            stored = v
            entry["stored_dtype"] = v.dtype.str
        if stored.dtype.kind in "US":  # strings (meta, up_name): keep numpy's own encoding
            raw = stored.tobytes()
            entry["stored_dtype"] = stored.dtype.str
        else:
            raw = np.ascontiguousarray(stored).tobytes()
        blob = comp.compress(raw)
        entry.update(offset=offset, size=len(blob), raw=len(raw))
        offset += len(blob)
        header[k] = entry
        blobs.append(blob)
    head = json.dumps(dict(version=VERSION, codec="zstd", level=level, arrays=header)).encode()
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(MAGIC)
        f.write(len(head).to_bytes(8, "little"))
        f.write(head)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)
    return os.path.getsize(path)


# ------------------------------------------------------------------ load

class StateFile:
    """A .cst file: arrays decoded on first access and cached. Mirrors NpzFile's interface
    (files, [], in, keys, iteration, context manager, close)."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            if f.read(len(MAGIC)) != MAGIC:
                raise ValueError(f"{path}: not a state file")
            n = int.from_bytes(f.read(8), "little")
            head = json.loads(f.read(n))
            self._base = len(MAGIC) + 8 + n
            self._blobs = f.read()  # the whole file: a median view is ~3 MB
        self.header = head
        self.arrays = head["arrays"]
        self.files = list(self.arrays)
        self._cache = {}
        self._prev = None
        self._dec = _zstd().ZstdDecompressor()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        self._blobs = b""

    def keys(self):
        return self.files

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __contains__(self, k):
        return k in self.arrays

    def get(self, k, default=None):
        return self[k] if k in self.arrays else default

    def _stored(self, k):
        e = self.arrays[k]
        raw = self._dec.decompress(self._blobs[e["offset"]:e["offset"] + e["size"]], max_output_size=e["raw"])
        return np.frombuffer(raw, dtype=np.dtype(e["stored_dtype"])).reshape(e["shape"]).copy()

    def __getitem__(self, k):
        if k in self._cache:
            return self._cache[k]
        e = self.arrays[k]
        t = e["transform"]
        if t == "raw":
            v = self._stored(k)
        elif t == "diff":
            d = self._stored(k)
            v = np.cumsum(d, dtype=np.int64).astype(np.dtype(e["dtype"]))
        elif t == "xor_frames":
            v = np.bitwise_xor.accumulate(self._stored(k), axis=0)
        elif t == "delta_id":
            self._decode_deltas()
            return self._cache[k]
        else:
            raise ValueError(f"{self.path}: unknown transform {t} for {k}")
        self._cache[k] = v
        return v

    def _decode_deltas(self):
        """All delta_id columns in one pass over the frames (one pass costs the same for 1 or 27)."""
        names = [k for k, e in self.arrays.items() if e["transform"] == "delta_id" and k not in self._cache]
        if not names:
            return
        if self._prev is None:
            self._prev = prev_rows(self["u_id"], self["frame_start"])
        stored = {k: self._stored(k) for k in names}
        ints = decode_delta(stored, self._prev, self["frame_start"])
        for k in names:
            e = self.arrays[k]
            self._cache[k] = _from_ints(ints[k], e["how"], e["dtype"])


def load(path):
    """A StateFile for a .cst file, np.load's NpzFile for an .npz (same interface)."""
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
    if magic == MAGIC:
        return StateFile(path)
    return np.load(path)


def is_state_file(name):
    return name.endswith(".cst") or name.endswith(".npz")
