"""The energy model's State from the physics batches (physics_data.collate): frames a
loop apart, rows on fixed indices, each player's bank and supply.

Tables     a row type's constants: value and build time from unit_data.json, speed and
           acceleration from the patch's ruleset (rulesets.json), the energy pool and
           whether it flies from energy_types.json (scanned from the data; absent, no
           pools and nothing flies by type, and the flag in the frame still says)
Frames     the encoder: a batch and a list of frames -> State, the next frame's
           coordinates, and which rows are scored. The semantic context c is the row
           type's node from the ruleset encoder (ruleset.py: the patch's balance data,
           encoded once a pass, trained end to end with the model), a small MLP over
           the raw numbers the coordinates leave out, and the current order's ability
           node (its cost, product and range are in the node's numbers); the input u
           is the ability node again with the order's target offset, has a target,
           progress, and whether the queue changed this step; the player context cp is
           the race. Velocities and the ledger's rates
           are one-frame differences from the previous frame (instantaneous: a trip
           landing, a purchase), so a window's first frame is never stepped from.

Rows: real when present, alive and on the map (passengers and units in gas buildings
stand where their parent is, so they are left out). A row is scored when it has an
observed next frame on the map; a death is scored (hp 0), a row out of sight is not.
"""

import json
import os

import numpy as np
import torch
from torch import nn

from . import energy as E
from .physics_data import NEUTRAL, RAW_NUMS
from .tokenize import PC_KEYS

I = {n: i for i, n in enumerate(RAW_NUMS)}
PC = {n: i for i, n in enumerate(PC_KEYS)}
N_RAW_TYPES = 4096
LOOPS_PER_SECOND_EXPORT = 16.0  # the ruleset's speeds are per this many loops
INSTANT_ACCEL = 100.0  # export acceleration at or above which a type is instant
SPEED_AT, ACCEL_AT = 17, 18  # ruleset unit scalars: speed / 5, log1p(accel) / 8
CLIFF_WALKERS = (49, 4)  # API ids: Reaper, Colossus
FLYING_BIT = 1  # bit 0 of the raw flags
# the raw numbers the context reads: what the coordinates, givens and constants leave
# out
EXTRAS = (
    "harvesters",
    "ideal_harvesters",
    "cargo_used",
    "cargo_max",
    "buff_remain",
    "buff_max",
    "detect_range",
    "radar_range",
    "atk_up",
    "arm_up",
    "shd_up",
    "flags",
    "flags2",
    "display",
    "cloak",
    "add_on",
)
# not read: "engaged" (the API's current target) is the pair structure in a unary
# feature, and would let the own term absorb damage
N_RACES = 5
UNKNOWN_ABILITY = 2**31 - 1  # a raw id past every lookup table: the unknown row


class Tables:
    """Per raw unit type (API id): value, build_time (loops), race, speed (map units a
    loop), mass (loops² a map unit: the inverse of acceleration; 0 when instant),
    energy_max (0: no pool), flying, cliff."""

    def __init__(
        self,
        base_build,
        unit_data="data/worldmodel/unit_data.json",
        rulesets="data/worldmodel/rulesets.json",
        energy_types="data/worldmodel/energy_types.json",
    ):
        value, build, race = (np.zeros(N_RAW_TYPES, np.float32) for _ in range(3))
        for t, u in json.load(open(unit_data)).items():
            t = int(t)
            if 0 <= t < N_RAW_TYPES:
                value[t] = u["minerals"] + u["vespene"]
                build[t] = u["build_time"]
                race[t] = u["race"]
        rs = json.load(open(rulesets))["rulesets"]
        keys = [k for k in rs if k.split("_")[0] == str(base_build)]
        self.ruleset = keys[0] if keys else sorted(rs)[-1]
        speed, mass = (
            np.zeros(N_RAW_TYPES, np.float32),
            np.zeros(N_RAW_TYPES, np.float32),
        )
        for u in rs[self.ruleset]["units"]:
            sp = u["scal"][SPEED_AT] * 5 / LOOPS_PER_SECOND_EXPORT
            ac = float(np.expm1(u["scal"][ACCEL_AT] * 8))
            m = (
                0.0
                if ac >= INSTANT_ACCEL or ac <= 0
                else LOOPS_PER_SECOND_EXPORT**2 / ac
            )
            for a in u["api_ids"]:
                if 0 <= a < N_RAW_TYPES:
                    speed[a], mass[a] = sp, m
        energy_max, flying = (np.zeros(N_RAW_TYPES, np.float32) for _ in range(2))
        if os.path.exists(energy_types):
            et = json.load(open(energy_types))
            for t, e in et["energy_max"].items():
                energy_max[int(t)] = e
            for t in et["flying"]:
                flying[t] = 1.0
        cliff = np.zeros(N_RAW_TYPES, np.float32)
        cliff[list(CLIFF_WALKERS)] = 1.0
        self.race = torch.from_numpy(race.astype(np.int64))
        # value, mass, speed, build_time, energy_max, flying, cliff
        self.k = torch.from_numpy(
            np.stack([value, mass, speed, build, energy_max, flying, cliff], 1)
        )


class Frames(nn.Module):
    """A batch's frames as State. frames(b, ff) -> (State of M = B · len(ff) frames,
    q_next, p_next, scored). ff: frame indices in [1, F − 2]."""

    def __init__(self, vocab, tables, d, u_dim, rules):
        super().__init__()
        self.tables, self.rules = tables, rules
        if tables.ruleset not in rules.tables.rulesets:
            raise KeyError(f"ruleset {tables.ruleset} is not in the ruleset tables")
        self.register_buffer("k_type", tables.k)
        self.register_buffer("race", tables.race)
        # the batch's order slots hold vocab ability indices; the ruleset looks
        # abilities up by raw API id. 0: no order (a zero row), 1: an ability the vocab
        # does not know (the ruleset's unknown row)
        raw = torch.zeros(vocab.n_abilities, dtype=torch.long)
        raw[1] = UNKNOWN_ABILITY
        for a, i in vocab._abil.items():
            raw[i] = a
        self.register_buffer("abil_raw", raw)
        self.unit_c = nn.Linear(rules.d, d)
        self.abil_c = nn.Linear(rules.d, d)
        self.abil_u = nn.Linear(rules.d, u_dim)
        self.extras = E.mlp(len(EXTRAS), d, d)
        self.race_emb = nn.Embedding(N_RACES, d)
        self.reg_u = nn.Linear(5, u_dim)
        self.register_buffer("extras_at", torch.tensor([I[n] for n in EXTRAS]))

    def nodes(self, rtype, abil):
        """The ruleset encoder's rows for raw unit types and vocab ability indices:
        the unit node (the unknown row for a type the ruleset lacks) and the ability
        node (zero where there is no order). The ruleset is encoded once a call."""
        self.rules.clear()  # a pass's graph is freed by its backward: encode anew
        key = self.tables.ruleset
        emb, t = self.rules(key), self.rules._on(key, rtype.device)
        ulut, alut = t["unit_lut"], t["abil_lut"]
        unit = emb["unit"][ulut[rtype.clamp(0, len(ulut) - 1)]]
        raw = self.abil_raw[abil]
        ability = emb["ability"][alut[raw.clamp(0, len(alut) - 1)]]
        return unit, ability * (abil > 0)[..., None].to(ability.dtype)

    def coords(self, b, ff):
        """[B, nf, R, len(COORDS)] the rows' integrated coordinates at frames ff,
        velocities from the frame before (0 where either frame lacks the row)."""
        x, y, num = b["x"][:, ff], b["y"][:, ff], b["num"][:, ff]
        px, py = b["x"][:, ff - 1], b["y"][:, ff - 1]
        both = (b["mask"][:, ff] & b["mask"][:, ff - 1]).to(x.dtype)
        hp_max, sh_max = num[..., I["hp_max"]], num[..., I["shield_max"]]
        cols = [
            x,
            y,
            (x - px) * both,
            (y - py) * both,
            num[..., I["hp"]] / hp_max.clamp(min=1),
            num[..., I["shield"]] / sh_max.clamp(min=1),
            num[..., I["energy"]] / E.ENERGY_UNIT,
            num[..., I["contents"]],
        ]
        return torch.stack(cols, -1)

    def givens(self, b, ff):
        """[B, nf, R, len(GIVEN)] height, heading as cosine and sine, weapon cooldown,
        build progress."""
        num = b["num"][:, ff]
        facing = num[..., I["facing"]]
        return torch.stack(
            [
                num[..., I["z"]],
                facing.cos(),
                facing.sin(),
                num[..., I["weapon_cooldown"]],
                num[..., I["build"]],
            ],
            -1,
        )

    def ledger(self, b, ff):
        """[B, nf, P, len(PLAYER)] bank and supply (the batch stores log1p / 8) and
        their one-frame rates (ff ≥ 1)."""
        pl = b["ply_num"]

        def raw(f):
            v = torch.expm1(pl[:, f, :, : len(PC_KEYS)] * 8)
            used, cap = v[..., PC["food_used"]], v[..., PC["food_cap"]]
            return torch.stack(
                [v[..., PC["minerals"]], v[..., PC["vespene"]], used, cap, cap - used],
                -1,
            )

        now, before = raw(ff), raw(ff - 1)
        return torch.cat([now, now - before], -1)

    def frames(self, b, ff):
        ff = torch.as_tensor(ff, device=b["mask"].device)
        B, _, R = b["mask"].shape
        nf = len(ff)
        q, q_next = self.coords(b, ff), self.coords(b, ff + 1)
        g = self.givens(b, ff)
        p, p_next = self.ledger(b, ff), self.ledger(b, ff + 1)
        rtype = b["rtype"][:, ff].clamp(0, N_RAW_TYPES - 1)
        owner = b["owner"][:, None].expand(B, nf, R)
        owner = torch.where(owner == NEUTRAL, E.GROUND, owner + 1)
        on_map = lambda f: b["mask"][:, f] & (b["inside"][:, f] == 0)
        mask = on_map(ff) & b["alive"][:, ff]
        scored = mask & on_map(ff + 1)
        num = b["num"][:, ff]
        kt = self.k_type[
            rtype
        ]  # value, mass, speed, build_time, energy_max, flying, cliff
        value = torch.where(owner == E.GROUND, q[..., E.Q["contents"]], kt[..., 0])
        # a pool the table missed shows as energy in the frame: a full pool then
        energy_max = torch.where(
            (kt[..., 4] <= 0) & (num[..., I["energy"]] > 0), E.ENERGY_UNIT, kt[..., 4]
        )
        flying = torch.maximum(
            kt[..., 5], (num[..., I["flags"]].long() & FLYING_BIT).to(kt.dtype)
        )
        k = torch.stack(
            [
                value,
                kt[..., 1],
                kt[..., 2],
                num[..., I["hp_max"]],
                num[..., I["shield_max"]],
                energy_max,
                kt[..., 3],
                num[..., I["radius"]],
                flying,
                kt[..., 6],
            ],
            -1,
        )
        abil = b["slot_abil"][:, ff, :, 0]
        unit_node, abil_node = self.nodes(rtype, abil)
        extras = num[..., self.extras_at]
        extras = torch.sign(extras) * torch.log1p(extras.abs())
        c = self.unit_c(unit_node) + self.extras(extras) + self.abil_c(abil_node)
        reg = torch.cat(
            [
                b["slot_reg"][:, ff, :, 0, :],
                b["queue_changed"][:, ff][..., None].float(),
            ],
            -1,
        )
        u = self.abil_u(abil_node) + self.reg_u(reg)
        race = self.race[rtype] * mask
        cp = torch.stack(
            [
                torch.where(owner == pl_, race, torch.zeros_like(race)).amax(-1)
                for pl_ in (E.P1, E.P2)
            ],
            -1,
        ).clamp(max=N_RACES - 1)
        cp = self.race_emb(cp)
        flat = lambda t: t.reshape(B * nf, *t.shape[2:])
        s = E.State(
            flat(q),
            flat(g),
            flat(k),
            flat(c),
            flat(owner),
            flat(mask),
            flat(p),
            flat(cp),
            flat(u),
        )
        return s, flat(q_next), flat(p_next), flat(scored)


BATCH_KEYS = (
    "mask",
    "alive",
    "inside",
    "x",
    "y",
    "num",
    "rtype",
    "owner",
    "ply_num",
    "slot_abil",
    "slot_reg",
    "queue_changed",
)


def to_device(b, device):
    """The batch keys the energy model reads, on the device."""
    return {k: b[k].to(device, non_blocking=True) for k in BATCH_KEYS}
