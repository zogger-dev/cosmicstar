"""Vocabularies for StarCraft II concepts (units, abilities, upgrades, races).

Within `unit` and `ability`, index `0` is padding (`PAD`) and `1` is unknown/rare (`UNK`);
real entries start at `2`. Races use `0..3` without padding.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np

from .paths import UNIT_DATA_PATH, VOCAB_PATH, resolve_vocab_path

PAD, UNK = 0, 1
RACES = ("Terran", "Zerg", "Protoss", "Random")
CAT_OTHER, CAT_WORKER, CAT_STRUCTURE, CAT_ARMY = 0, 1, 2, 3


class Vocab:
    """Discrete concept vocabulary over unit types, abilities, upgrades, and races."""

    def __init__(self, d: dict):
        kinds = collections.defaultdict(list)
        for kind, key in d["concepts"]:
            kinds[kind].append(key)
        self._unit = {int(k): 2 + i for i, k in enumerate(kinds["unit"])}
        self._abil = {int(k): 2 + i for i, k in enumerate(kinds["ability"])}
        self.n_units = 2 + len(self._unit)
        self.n_abilities = 2 + len(self._abil)
        self.upgrade_names = list(kinds["upgrade"])
        self.upgrade_slot = {k: i for i, k in enumerate(self.upgrade_names)}
        self.race = {k: i for i, k in enumerate(RACES)}
        self.n_races = len(RACES)
        self.categories = {int(k): v for k, v in d.get("categories", {}).items()}
        self.producers = set(d.get("producers", []))

    @classmethod
    def load(cls, path: str | None = None) -> "Vocab":
        resolved = resolve_vocab_path(path)
        with open(resolved) as f:
            return cls(json.load(f))

    @property
    def n_upgrades(self) -> int:
        return len(self.upgrade_names)

    def unit_index(self, raw_types):
        return _lookup(self._unit, raw_types)

    def ability_index(self, raw_abilities):
        return _lookup(self._abil, raw_abilities)

    def category(self, raw_types):
        """Category of each raw unit type id (`CAT_OTHER` if unknown)."""
        raw = np.asarray(raw_types, np.int64)
        if not hasattr(self, "_cat_lut"):
            self._cat_lut = np.full(
                max(self.categories, default=0) + 1, CAT_OTHER, np.int64
            )
            for k, v in self.categories.items():
                self._cat_lut[k] = v
        ok = (raw >= 0) & (raw < len(self._cat_lut))
        return np.where(
            ok, self._cat_lut[np.clip(raw, 0, len(self._cat_lut) - 1)], CAT_OTHER
        )


def _lookup(table: dict[int, int], arr):
    """Maps raw ids through `table`; ids not in `table` -> `UNK`, ids `<= 0` -> `PAD`."""
    arr = np.asarray(arr, np.int64)
    if not len(arr):
        return np.zeros(0, np.int64)
    lut = np.full(max(max(table, default=0), int(arr.max())) + 1, UNK, np.int64)
    for k, v in table.items():
        lut[k] = v
    out = lut[np.clip(arr, 0, len(lut) - 1)]
    out[arr <= 0] = PAD
    return out


def unit_info(unit_data_path: str = UNIT_DATA_PATH):
    """Categories and unit types that produce units (structures, eggs, cocoons, larva)."""
    if not os.path.exists(unit_data_path):
        return {}, []
    cats, producers = {}, []
    with open(unit_data_path) as f:
        raw_ud = json.load(f)
    for k, u in raw_ud.items():
        cats[int(k)] = (
            CAT_WORKER
            if u["worker"]
            else (
                CAT_STRUCTURE
                if u["structure"]
                else CAT_ARMY if u["food"] > 0 else CAT_OTHER
            )
        )
        if (
            u["structure"]
            or "Egg" in u["name"]
            or "Cocoon" in u["name"]
            or u["name"] == "Larva"
        ):
            producers.append(int(k))
    return cats, sorted(producers)


def build(paths: list[str], min_count: int = 50, unit_data: str = UNIT_DATA_PATH) -> dict:
    from ..data import statefile

    types = collections.Counter()
    abils = collections.Counter()
    ups = collections.Counter()
    for p in paths:
        d = statefile.load(p)
        t, c = np.unique(d["u_type"], return_counts=True)
        types.update(dict(zip(t.tolist(), c.tolist())))
        for key in ("prod_abil", "ord_abil", "u_order_abil", "a_abil"):
            if key in d.files:
                a, c = np.unique(d[key], return_counts=True)
                abils.update({k: v for k, v in zip(a.tolist(), c.tolist()) if k > 0})
        ups.update(d["up_name"].tolist())
    concepts = (
        [("unit", k) for k in sorted(k for k, v in types.items() if v >= min_count)]
        + [("ability", k) for k in sorted(k for k, v in abils.items() if v >= min_count)]
        + [("upgrade", k) for k in sorted(k for k, v in ups.items() if v >= 3)]
    )
    cats, producers = unit_info(unit_data)
    return dict(
        concepts=concepts,
        categories=cats,
        producers=producers,
        counts=dict(
            games=len(paths), unit_types_seen=len(types), abilities_seen=len(abils)
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", default=VOCAB_PATH)
    ap.add_argument("--min_count", type=int, default=50)
    args = ap.parse_args()
    paths = sorted(
        glob.glob(os.path.join(args.data_dir, "*.cst"))
        + glob.glob(os.path.join(args.data_dir, "*.npz"))
    )
    v = build(paths, args.min_count)
    with open(args.out, "w") as f:
        json.dump(v, f, indent=1)
    kinds = collections.Counter(k for k, _ in v["concepts"])
    print(f"{len(paths)} games: {len(v['concepts'])} concepts {dict(kinds)} -> {args.out}")


if __name__ == "__main__":
    main()
