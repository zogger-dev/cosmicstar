"""Vocabularies for the game's concepts, one index space per kind:

    units      unit types (zergling, hatchery, mineral field, ...)
    abilities  abilities (what a structure / egg is producing; later: commands)
    upgrades   upgrades (completed ones go on the player token)
    races      Terran / Zerg / Protoss / Random

Each kind gets its own embedding table in the encoder. Within a kind, index 0 is
padding and 1 is "unknown/rare"; real values start at 2 (races: 0..3, no padding).

    python -m cosmicstar.worldmodel.vocab data/worldmodel/full --out data/worldmodel/vocab.json
"""

import argparse
import collections
import glob
import json
import os

import numpy as np

from . import statefile

PAD, UNK = 0, 1
RACES = ("Terran", "Zerg", "Protoss", "Random")
# Unit categories (from scripts/dump_unit_data.py), used to sample readout windows.
CAT_OTHER, CAT_WORKER, CAT_STRUCTURE, CAT_ARMY = 0, 1, 2, 3


class Vocab:
    def __init__(self, d):
        kinds = collections.defaultdict(list)
        for kind, key in d["concepts"]:
            kinds[kind].append(key)
        self._unit = {int(k): 2 + i for i, k in enumerate(kinds["unit"])}
        self._abil = {int(k): 2 + i for i, k in enumerate(kinds["ability"])}
        self.n_units, self.n_abilities = 2 + len(self._unit), 2 + len(self._abil)
        self.upgrade_names = list(kinds["upgrade"])
        self.upgrade_slot = {k: i for i, k in enumerate(self.upgrade_names)}  # multi-hot position = row
        self.race = {k: i for i, k in enumerate(RACES)}
        self.n_races = len(RACES)
        self.categories = {int(k): v for k, v in d.get("categories", {}).items()}
        self.producers = set(d.get("producers", []))  # raw unit types whose order is "producing"

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path)))

    @property
    def n_upgrades(self):
        return len(self.upgrade_names)

    def unit_index(self, raw_types):
        return _lookup(self._unit, raw_types)

    def ability_index(self, raw_abilities):
        return _lookup(self._abil, raw_abilities)

    def category(self, raw_types):
        """Category of each raw unit type id (CAT_OTHER if unknown)."""
        raw = np.asarray(raw_types, np.int64)
        if not hasattr(self, "_cat_lut"):
            self._cat_lut = np.full(max(self.categories, default=0) + 1, CAT_OTHER, np.int64)
            for k, v in self.categories.items():
                self._cat_lut[k] = v
        ok = (raw >= 0) & (raw < len(self._cat_lut))
        return np.where(ok, self._cat_lut[np.clip(raw, 0, len(self._cat_lut) - 1)], CAT_OTHER)


def _lookup(table, arr):
    """Maps raw ids through table; ids not in it -> UNK, ids <= 0 -> PAD."""
    arr = np.asarray(arr, np.int64)
    if not len(arr):
        return np.zeros(0, np.int64)
    lut = np.full(max(max(table, default=0), int(arr.max())) + 1, UNK, np.int64)
    for k, v in table.items():
        lut[k] = v
    out = lut[np.clip(arr, 0, len(lut) - 1)]
    out[arr <= 0] = PAD
    return out


def unit_info(unit_data_path):
    """Categories, and the unit types that produce things (structures, eggs, cocoons)."""
    if not os.path.exists(unit_data_path):
        return {}, []
    cats, producers = {}, []
    for k, u in json.load(open(unit_data_path)).items():
        cats[int(k)] = (CAT_WORKER if u["worker"] else CAT_STRUCTURE if u["structure"]
                        else CAT_ARMY if u["food"] > 0 else CAT_OTHER)
        if u["structure"] or "Egg" in u["name"] or "Cocoon" in u["name"] or u["name"] == "Larva":
            producers.append(int(k))
    return cats, sorted(producers)


def build(paths, min_count=50, unit_data="data/worldmodel/unit_data.json"):
    types, abils, ups = collections.Counter(), collections.Counter(), collections.Counter()
    for p in paths:
        d = statefile.load(p)
        t, c = np.unique(d["u_type"], return_counts=True)
        types.update(dict(zip(t.tolist(), c.tolist())))
        for key in ("prod_abil", "ord_abil", "u_order_abil", "a_abil"):  # production and unit orders
            if key in d.files:
                a, c = np.unique(d[key], return_counts=True)
                abils.update({k: v for k, v in zip(a.tolist(), c.tolist()) if k > 0})
        ups.update(d["up_name"].tolist())
    concepts = ([("unit", k) for k in sorted(k for k, v in types.items() if v >= min_count)]
                + [("ability", k) for k in sorted(k for k, v in abils.items() if v >= min_count)]
                + [("upgrade", k) for k in sorted(k for k, v in ups.items() if v >= 3)])
    cats, producers = unit_info(unit_data)
    return dict(concepts=concepts, categories=cats, producers=producers,
                counts=dict(games=len(paths), unit_types_seen=len(types), abilities_seen=len(abils)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="data/worldmodel/vocab.json")
    ap.add_argument("--min_count", type=int, default=50)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.cst")) + glob.glob(os.path.join(args.data_dir, "*.npz")))
    v = build(paths, args.min_count)
    json.dump(v, open(args.out, "w"), indent=1)
    kinds = collections.Counter(k for k, _ in v["concepts"])
    print(f"{len(paths)} games: {len(v['concepts'])} concepts {dict(kinds)} -> {args.out}")


if __name__ == "__main__":
    main()
