"""A view on disk as Arrow files (scripts/extract_states.py --arrow), seek friendly: <view>.units.arrow, one row per unit
with ragged lists as list columns, and <view>.frames.arrow, one row per frame, in record batches of a fixed number of frames.
In the units file a batch's first frame is a keyframe and every other unit column is the change from that unit's row in the
previous frame of the batch (statefile's integer deltas), so reading a moment touches only its batch. Each batch's custom
metadata says how to undo that."""

import json

import numpy as np
import pyarrow as pa

from . import statefile


def encode_units(cols, counts):
    """{name: column} (u_id whole) and rows per frame -> ({name: stored column}, {name: [how, dtype]})."""
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    prev = statefile.prev_rows(cols["u_id"], starts)
    stored, how = {}, {}
    for k, v in cols.items():
        if k == "u_id":
            stored[k] = v
            continue
        ints, h = statefile._to_ints(v)
        stored[k] = statefile.encode_delta({k: ints}, prev)[k]
        how[k] = [h, v.dtype.str]
    return stored, how


def open_view(base):
    """(units reader, frames reader) of <base>.units.arrow, <base>.frames.arrow, memory mapped."""
    return pa.ipc.open_file(pa.memory_map(f"{base}.units.arrow")), pa.ipc.open_file(
        pa.memory_map(f"{base}.frames.arrow")
    )


def read_units(reader, i):
    """Batch i of a units file: ({name: unit column}, {name: list column}, frame per row)."""
    b, meta = reader.get_batch_with_custom_metadata(i)
    how = json.loads(meta[b"how"])
    ids = b.column("u_id").to_numpy(zero_copy_only=False)
    frame = b.column("frame").to_numpy(zero_copy_only=False)
    f0 = int(frame[0]) if len(frame) else 0
    n_frames = int(frame[-1]) - f0 + 1 if len(frame) else 0
    starts = np.searchsorted(frame, np.arange(f0, f0 + n_frames + 1)).astype(np.int64)
    prev = statefile.prev_rows(ids, starts)
    ints = {k: b.column(k).to_numpy(zero_copy_only=False).copy() for k in how}
    decoded = statefile.decode_delta(ints, prev, starts)
    units = {k: statefile._from_ints(decoded[k], *how[k]) for k in how}
    units["u_id"] = ids
    lists = {
        f.name: b.column(f.name) for f in b.schema if pa.types.is_large_list(f.type)
    }
    return units, lists, frame
