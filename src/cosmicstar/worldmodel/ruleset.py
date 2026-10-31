"""The ruleset encoder: the game's balance data -> embeddings for unit types, ability commands
and upgrades, computed from the data rather than looked up in a learned table.

Nodes (scripts/build_ruleset_tables.py): every unit type, ability command and upgrade level of
a ruleset, each with a fixed-width feature vector. A unit also owns a set of weapons, each a
vector (range, period, damage, bonus damage per attribute tag, ...); the set is summed, so any
number of weapons works and a unit without one adds nothing. Every node gets a learned
residual keyed by its stable name (zero-initialized; L2 keeps the data carrying the load), so
a unit the data describes well needs none, and one in a future patch starts from its numbers.

Attention over all nodes of the ruleset, with an additive bias per edge relation (has ability,
produced by, requires, ...), lets a type's vector hold what is true in every game: a weapon's
bonus tag meets the attribute tags of the types it applies to, a target class meets movement
types, a train command meets its product. What depends on the situation stays with the
entity-level attention over the frame.

The tables are a function of the ruleset alone, so the encoder runs once per ruleset per batch
(about 1600 tokens) and the entity tokens gather their rows (RulesetEncoder.lookup).
"""
import json

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import mlp

KIND_UNIT, KIND_ABILITY, KIND_UPGRADE = 0, 1, 2
MAX_WEAPONS = 4


class RulesetTables:
    """The builder's JSON as tensors per ruleset, plus API id -> row lookups."""

    def __init__(self, path="data/worldmodel/rulesets.json"):
        d = json.load(open(path))
        self.widths = d["widths"]
        self.relations = list(d["relations"])
        self.n_relations = len(self.relations)
        self.names = {k: {n: i for i, n in enumerate(v)} for k, v in d["names"].items()}
        self.n_names = {k: len(v) for k, v in self.names.items()}
        self.rulesets = {}
        for key, g in d["rulesets"].items():
            self.rulesets[key] = self._tensors(g)

    def _tensors(self, g):
        U, A, G = len(g["units"]), len(g["abilities"]), len(g["upgrades"])
        t = dict(U=U, A=A, G=G)
        t["unit_scal"] = torch.tensor([u["scal"] for u in g["units"]], dtype=torch.float32)
        wpn = torch.zeros(U, MAX_WEAPONS, self.widths["weapon"])
        wmask = torch.zeros(U, MAX_WEAPONS, dtype=torch.bool)
        for i, u in enumerate(g["units"]):
            for j, w in enumerate(u["weapons"][:MAX_WEAPONS]):
                wpn[i, j], wmask[i, j] = torch.tensor(w), True
        t["unit_wpn"], t["wpn_mask"] = wpn, wmask
        t["abil_feat"] = torch.tensor([a["feat"] for a in g["abilities"]], dtype=torch.float32).view(A, -1)
        t["upg_feat"] = torch.tensor([u["feat"] for u in g["upgrades"]], dtype=torch.float32).view(G, -1)
        t["unit_name"] = torch.tensor([self.names["units"][u["name"]] for u in g["units"]])
        t["abil_name"] = torch.tensor([self.names["abilities"][a["name"]] for a in g["abilities"]])
        t["upg_name"] = torch.tensor([self.names["upgrades"][u["name"]] for u in g["upgrades"]])
        # edges in the joint node space [units | abilities | upgrades]
        off = {KIND_UNIT: 0, KIND_ABILITY: U, KIND_UPGRADE: U + A}
        rel, src, dst = [], [], []
        for r, s, d_ in g["edges"]:
            rel.append(r)
            src.append(s + self._kind_offset(self.relations[r], True, off))
            dst.append(d_ + self._kind_offset(self.relations[r], False, off))
        t["edge_rel"], t["edge_src"], t["edge_dst"] = torch.tensor(rel), torch.tensor(src), torch.tensor(dst)
        # API id -> row (row U / A / G = unknown)
        def lut(items, key, n):
            pairs = [(a, i) for i, it in enumerate(items) for a in (it[key] if isinstance(it[key], list) else [it[key]])
                     if a is not None and a >= 0]
            out = torch.full((max((a for a, _ in pairs), default=0) + 2,), n, dtype=torch.long)
            for a, i in pairs:
                out[a] = i
            return out
        t["unit_lut"] = lut(g["units"], "api_ids", U)
        t["abil_lut"] = lut(g["abilities"], "api_ids", A)
        t["upg_lut"] = lut(g["upgrades"], "api_id", G)
        t["upg_by_name"] = {u["name"]: i for i, u in enumerate(g["upgrades"])}
        t["upg_by_lower"] = {u["name"].lower(): i for i, u in enumerate(g["upgrades"])}
        return t

    @staticmethod
    def _kind_offset(name, is_src, off):
        """Relation names encode the node kinds: "<src>_..._<dst>" with u / a / g."""
        k = name[0] if is_src else name[-1]
        return off[{"u": KIND_UNIT, "a": KIND_ABILITY, "g": KIND_UPGRADE}[k]]



class Block(nn.Module):
    def __init__(self, d, heads, mlp_ratio=4.0):
        super().__init__()
        self.heads = heads
        self.n1, self.qkv, self.o = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2, self.ff = nn.LayerNorm(d), mlp(d, d, int(d * mlp_ratio))

    def forward(self, x, bias):
        N, d = x.shape
        H = self.heads
        q, k, v = self.qkv(self.n1(x)).view(N, 3, H, d // H).permute(1, 2, 0, 3)  # [3, H, N, dh]
        a = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=bias[None])[0]  # [H, N, dh]
        x = x + self.o(a.transpose(0, 1).reshape(N, d))
        return x + self.ff(self.n2(x))


class RulesetEncoder(nn.Module):
    """Tables of a ruleset -> embeddings of its units [U+1, d] (last row: unknown type),
    ability commands [A+1, d] and upgrades [G+1, d]."""

    def __init__(self, tables, d=256, depth=3, heads=8, mlp_ratio=4.0, residual=True):
        super().__init__()
        self.tables, self.d, self.heads, self.residual = tables, d, heads, residual
        w = tables.widths
        self.unit_in, self.wpn_in = mlp(w["unit"], d), mlp(w["weapon"], d)
        self.abil_in, self.upg_in = mlp(w["ability"], d), mlp(w["upgrade"], d)
        self.kind = nn.Embedding(3, d)
        self.unknown = nn.Parameter(torch.zeros(3, d))
        self.res = nn.ModuleDict({k: nn.Embedding(tables.n_names[k], d) for k in ("units", "abilities", "upgrades")})
        for e in self.res.values():
            nn.init.zeros_(e.weight)
        self.rel_bias = nn.Parameter(torch.zeros(tables.n_relations, heads))
        self.blocks = nn.ModuleList([Block(d, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        nn.init.normal_(self.unknown, std=0.02)
        self._dev_tables = {}
        self._cache = {}  # ruleset -> this batch's embeddings; clear() at every step

    def clear(self):
        self._cache = {}

    def _on(self, key, dev):
        """A ruleset's tensors on the device (cached)."""
        if (key, dev) not in self._dev_tables:
            t = self.tables.rulesets[key]
            self._dev_tables[(key, dev)] = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items()}
        return self._dev_tables[(key, dev)]

    def forward(self, key):
        if key in self._cache:
            return self._cache[key]
        self._cache[key] = out = self._encode(key)
        return out

    def _encode(self, key):
        t = self._on(key, self.kind.weight.device)
        U, A, G = t["U"], t["A"], t["G"]
        units = self.unit_in(t["unit_scal"]) + (self.wpn_in(t["unit_wpn"]) * t["wpn_mask"][..., None]).sum(1)
        x = torch.cat([units + self.kind.weight[KIND_UNIT], self.abil_in(t["abil_feat"]) + self.kind.weight[KIND_ABILITY],
                       self.upg_in(t["upg_feat"]) + self.kind.weight[KIND_UPGRADE]], 0)
        if self.residual:
            x = x + torch.cat([self.res["units"](t["unit_name"]), self.res["abilities"](t["abil_name"]), self.res["upgrades"](t["upg_name"])], 0)
        N = x.shape[0]
        bias = x.new_zeros(self.heads, N, N)
        bias[:, t["edge_src"], t["edge_dst"]] = self.rel_bias[t["edge_rel"]].t().to(bias.dtype)
        for blk in self.blocks:
            x = blk(x, bias)
        x = self.norm(x)
        unk = self.unknown.to(x.dtype)
        return dict(unit=torch.cat([x[:U], unk[0:1]]), ability=torch.cat([x[U:U + A], unk[1:2]]), upgrade=torch.cat([x[U + A:], unk[2:3]]))

    def residual_norms(self):
        return {k: float(e.weight.norm(dim=-1).mean()) for k, e in self.res.items()}

    def lookup(self, rulesets, unit_api=None, abil_api=None, upg_names=None):
        """Per-sample rows. rulesets: list of B keys; unit_api / abil_api: [B, ...] raw API ids
        (0 / negative = none -> the unknown row for units, nothing for abilities); upg_names:
        per sample a list of completed upgrade names -> the sum of their rows, [B, d]. Returns a
        dict of the requested embeddings; samples of the same ruleset share one encoder pass and
        one gather. Works for B = 0."""
        dev = self.kind.weight.device
        keys = sorted(set(rulesets))
        emb = {key: self(key) for key in keys}
        groups = {key: torch.tensor([i for i, k in enumerate(rulesets) if k == key], device=dev, dtype=torch.long) for key in keys}
        dtype = next(iter(emb.values()))["unit"].dtype if emb else torch.float32

        def gather(api, which, lut):
            out = torch.zeros(api.shape + (self.d,), device=dev, dtype=dtype)
            for key, sel in groups.items():
                t, e = self._on(key, dev), emb[key][which]
                ids = api[sel].clamp(min=0, max=len(t[lut]) - 1)
                out[sel] = e[t[lut][ids]]
            return out

        out = {}
        if unit_api is not None:
            out["unit"] = gather(unit_api, "unit", "unit_lut")
        if abil_api is not None:
            r = gather(abil_api, "ability", "abil_lut")
            out["ability"] = torch.where((abil_api > 0)[..., None], r, torch.zeros_like(r))
        if upg_names is not None:
            rows = []
            for b, key in enumerate(rulesets):
                t, e = self._on(key, dev), emb[key]["upgrade"]
                idx = [t["upg_by_name"].get(n, t["upg_by_lower"].get(n.lower(), -1)) for n in upg_names[b]]
                idx = [i for i in idx if i >= 0]
                rows.append(e[idx].sum(0) if idx else e.new_zeros(e.shape[-1]))
            out["upgrade"] = torch.stack(rows) if rows else torch.zeros(0, self.d, device=dev, dtype=dtype)
        return out
