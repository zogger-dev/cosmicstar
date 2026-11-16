"""Unified multi-patch SC2 ruleset tables and unit/ability metadata catalogs.

Consolidates ruleset feature parsing (`Tables`) and unit cost/casualty metadata into a
single canonical module shared across `L1`, `L2`, and `Inspector`.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch

from .constants import (
    ACCEL_AT,
    BANK_SCALE,
    CLIFF_WALKERS,
    INSTANT_ACCEL,
    LOOPS_PER_SECOND_EXPORT,
    N_RAW_TYPES,
    SPEED_AT,
)
from .paths import ENERGY_TYPES_PATH, RULESETS_PATH, UNIT_DATA_PATH
from .vocab import Vocab

# Temporary / summoned / auxiliary units excluded from combat casualty tallies
EXCLUDE_FROM_CASUALTIES: frozenset[str] = frozenset(
    {
        "larva",
        "egg",
        "adeptphaseshift",
        "kd8charge",
        "interceptorsucceed",
        "interceptor",
        "pointdefensedrone",
        "autoturret",
        "broodling",
        "broodlingescort",
        "locustmp",
        "locustmpflying",
        "disruptorphased",
        "parasiticbombdummy",
        "mule",
        "creeptumor",
        "creeptumorburrowed",
        "creeptumorqueen",
        "changeling",
        "changelingzealot",
        "changelingmarine",
        "changelingmarineshield",
        "changelingzergling",
        "changelingzerglingwings",
    }
)

# Base Zerg structure build costs excluding the morphing Drone (for 25% cancellation loss)
ZERG_BUILD_COSTS: dict[str, tuple[int, int]] = {
    "extractor": (25, 0),
    "spawningpool": (200, 0),
    "hatchery": (300, 0),
    "evolutionchamber": (75, 0),
    "roachwarren": (150, 0),
    "banelingnest": (100, 50),
    "hydraliskden": (100, 100),
    "spire": (200, 200),
    "greaterspire": (100, 150),
    "ultraliskcavern": (150, 200),
    "infestationpit": (100, 100),
    "spinecrawler": (100, 0),
    "sporecrawler": (75, 0),
    "nydusnetwork": (150, 150),
    "nyduscanal": (0, 0),
    "lurkerdenmp": (100, 100),
}


def load_unit_metadata(unit_data_path: str = UNIT_DATA_PATH) -> dict[int, dict[str, Any]]:
    """Loads rich per-unit-id metadata (`name`, `race`, `minerals`, `vespene`, `category`, ...)."""
    out: dict[int, dict[str, Any]] = {}
    with open(unit_data_path) as f:
        raw_ud = json.load(f)
    for k, u in raw_ud.items():
        uid = int(k)
        cat = "other"
        if u.get("worker"):
            cat = "worker"
        elif u.get("structure"):
            cat = "structure"
        elif u.get("food", 0) > 0:
            cat = "army"
        out[uid] = {
            "name": u.get("name", f"Unit_{uid}"),
            "race": u.get("race", 0),
            "minerals": u.get("minerals", 0),
            "vespene": u.get("vespene", 0),
            "food": u.get("food", 0.0),
            "structure": bool(u.get("structure", False)),
            "worker": bool(u.get("worker", False)),
            "flying": bool(u.get("flying", False)),
            "radius": float(u.get("radius", 0.5) or 0.5),
            "category": cat,
        }
    return out


def build_unit_meta_tensors(
    unit_data_path: str = UNIT_DATA_PATH,
    n_raw_types: int = N_RAW_TYPES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns `(cost_1k, is_struct, is_worker)` float32 tensors indexed by raw SC2 API unit id."""
    cost_1k = torch.zeros(n_raw_types, dtype=torch.float32)
    is_struct = torch.zeros(n_raw_types, dtype=torch.float32)
    is_worker = torch.zeros(n_raw_types, dtype=torch.float32)
    with open(unit_data_path) as f:
        raw = json.load(f)
    for k, u in raw.items():
        tid = int(k)
        if 0 <= tid < n_raw_types:
            c = float(u.get("minerals", 0) + u.get("vespene", 0))
            cost_1k[tid] = max(c, 0.0) / BANK_SCALE
            is_struct[tid] = 1.0 if u.get("structure", False) else 0.0
            is_worker[tid] = 1.0 if u.get("worker", False) else 0.0
    return cost_1k, is_struct, is_worker


def load_ability_name_maps(
    vocab: Vocab,
    ruleset_key: str,
    rulesets_path: str = RULESETS_PATH,
) -> tuple[dict[int, str], dict[int, str]]:
    """Returns `(raw_abil_name, vocab_abil_name)` lookup dictionaries for `ruleset_key`."""
    with open(rulesets_path) as f:
        rs_doc = json.load(f)
    g_rs = rs_doc["rulesets"][ruleset_key]
    raw_abil_name: dict[int, str] = {0: "None"}
    for ab in g_rs["abilities"]:
        nm = ab.get("name", "")
        for aid in ab.get("api_ids", []):
            if aid is not None and aid >= 0:
                raw_abil_name[int(aid)] = nm
    vocab_abil_name: dict[int, str] = {0: "None", 1: "Unknown"}
    for raw_a, v_idx in vocab._abil.items():
        vocab_abil_name[int(v_idx)] = raw_abil_name.get(int(raw_a), f"Abil_{raw_a}")
    return raw_abil_name, vocab_abil_name


def _parse_single_ruleset(g: dict, rels: list[str], widths: dict[str, int]):
    w_u = widths["unit"]
    w_w = widths["weapon"]
    w_g = widths["upgrade"]
    unit_feat_dim = w_u + w_w
    speed = np.zeros(N_RAW_TYPES, np.float32)
    mass = np.zeros(N_RAW_TYPES, np.float32)
    unit_feat = np.zeros((N_RAW_TYPES, unit_feat_dim), np.float32)
    for u in g["units"]:
        sp = u["scal"][SPEED_AT] * 5 / LOOPS_PER_SECOND_EXPORT
        ac = float(np.expm1(u["scal"][ACCEL_AT] * 8))
        m = (
            0.0
            if ac >= INSTANT_ACCEL or ac <= 0
            else LOOPS_PER_SECOND_EXPORT**2 / ac
        )
        scal = np.asarray(u["scal"], dtype=np.float32)
        wpn = (
            np.sum(u["weapons"], axis=0, dtype=np.float32)
            if u["weapons"]
            else np.zeros(w_w, np.float32)
        )
        uf = np.concatenate([scal, wpn])
        for a in u["api_ids"]:
            if 0 <= a < N_RAW_TYPES:
                speed[a], mass[a] = sp, m
                unit_feat[a] = uf

    rel_u = rels.index("a_makes_u")
    rel_g = rels.index("a_makes_g")
    makes_u = {s: d_ for r, s, d_ in g["edges"] if r == rel_u}
    makes_g = {s: d_ for r, s, d_ in g["edges"] if r == rel_g}
    abil_by_raw: dict[int, np.ndarray] = {}
    for i, ab in enumerate(g["abilities"]):
        af = np.asarray(ab["feat"], dtype=np.float32)
        u_sc = (
            np.asarray(g["units"][makes_u[i]]["scal"], dtype=np.float32)
            if i in makes_u
            else np.zeros(w_u, np.float32)
        )
        g_ft = (
            np.asarray(g["upgrades"][makes_g[i]]["feat"], dtype=np.float32)
            if i in makes_g
            else np.zeros(w_g, np.float32)
        )
        vec = np.concatenate([af, u_sc, g_ft])
        for a_id in ab["api_ids"]:
            if a_id is not None and a_id >= 0:
                abil_by_raw[int(a_id)] = vec
    return speed, mass, unit_feat, abil_by_raw


class Tables:
    """Multi-patch SC2 static unit & ability feature tables indexed by raw API id.

    Provides `(value, mass, speed, build_time, energy_max, flying, cliff)` and full
    ruleset graph node features (`unit_feat`, `abil_feat`) across both a default
    `base_build` and every ruleset in `rulesets.json`.
    """

    def __init__(
        self,
        base_build: int = 94137,
        unit_data: str = UNIT_DATA_PATH,
        rulesets: str = RULESETS_PATH,
        energy_types: str = ENERGY_TYPES_PATH,
    ):
        value, build, race = (np.zeros(N_RAW_TYPES, np.float32) for _ in range(3))
        is_struct = np.zeros(N_RAW_TYPES, np.float32)
        is_worker = np.zeros(N_RAW_TYPES, np.float32)
        with open(unit_data) as f:
            raw_ud = json.load(f)
        for t, u in raw_ud.items():
            t = int(t)
            if 0 <= t < N_RAW_TYPES:
                value[t] = u["minerals"] + u["vespene"]
                build[t] = u["build_time"]
                race[t] = u["race"]
                is_struct[t] = 1.0 if u.get("structure", False) else 0.0
                is_worker[t] = 1.0 if u.get("worker", False) else 0.0

        with open(rulesets) as f:
            rs_doc = json.load(f)
        widths = rs_doc["widths"]
        rels = list(rs_doc["relations"])
        rs = rs_doc["rulesets"]
        self.ruleset_names = sorted(rs.keys())
        self.ruleset_to_id = {k: i for i, k in enumerate(self.ruleset_names)}
        keys = [k for k in self.ruleset_names if k.split("_")[0] == str(base_build)]
        self.ruleset = keys[0] if keys else self.ruleset_names[-1]

        w_u = widths["unit"]
        w_w = widths["weapon"]
        w_a = widths["ability"]
        w_g = widths["upgrade"]
        self.unit_feat_dim = w_u + w_w
        self.abil_feat_dim = w_a + w_u + w_g

        energy_max, flying = (np.zeros(N_RAW_TYPES, np.float32) for _ in range(2))
        if os.path.exists(energy_types):
            with open(energy_types) as f:
                et = json.load(f)
            for t, e in et["energy_max"].items():
                energy_max[int(t)] = e
            for t in et["flying"]:
                flying[t] = 1.0
        cliff = np.zeros(N_RAW_TYPES, np.float32)
        cliff[list(CLIFF_WALKERS)] = 1.0

        all_k_list = []
        all_uf_list = []
        self._all_abil_by_raw = []
        for rk in self.ruleset_names:
            sp_r, m_r, uf_r, ab_r = _parse_single_ruleset(rs[rk], rels, widths)
            all_k_list.append(
                np.stack([value, m_r, sp_r, build, energy_max, flying, cliff], 1)
            )
            all_uf_list.append(uf_r)
            self._all_abil_by_raw.append(ab_r)

        def_idx = self.ruleset_to_id[self.ruleset]
        self.abil_by_raw = self._all_abil_by_raw[def_idx]
        self.race = torch.from_numpy(race.astype(np.int64))
        self.is_struct = torch.from_numpy(is_struct)
        self.is_worker = torch.from_numpy(is_worker)
        self.k = torch.from_numpy(all_k_list[def_idx])
        self.unit_feat = torch.from_numpy(all_uf_list[def_idx])
        self.all_k = torch.from_numpy(np.stack(all_k_list, axis=0))
        self.all_unit_feat = torch.from_numpy(np.stack(all_uf_list, axis=0))

    def abil_feat(self, vocab: Vocab) -> torch.Tensor:
        """`[vocab.n_abilities, abil_feat_dim]` static feature tensor for default ruleset."""
        out = np.zeros((vocab.n_abilities, self.abil_feat_dim), np.float32)
        for raw_a, idx in vocab._abil.items():
            if raw_a in self.abil_by_raw and 0 <= idx < vocab.n_abilities:
                out[idx] = self.abil_by_raw[raw_a]
        return torch.from_numpy(out)

    def all_abil_feat(self, vocab: Vocab) -> torch.Tensor:
        """`[N_RS, vocab.n_abilities, abil_feat_dim]` static ability features across all rulesets."""
        out = np.zeros(
            (len(self.ruleset_names), vocab.n_abilities, self.abil_feat_dim), np.float32
        )
        for ri, ab_map in enumerate(self._all_abil_by_raw):
            for raw_a, idx in vocab._abil.items():
                if raw_a in ab_map and 0 <= idx < vocab.n_abilities:
                    out[ri, idx] = ab_map[raw_a]
        return torch.from_numpy(out)
