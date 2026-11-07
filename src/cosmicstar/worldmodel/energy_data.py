"""The energy model's State from the physics batches (physics_data.collate): frames a
loop apart, rows on fixed indices, each player's bank and supply.

Tables     a row type's constants: value and build time from unit_data.json, speed and
           acceleration from the patch's ruleset (rulesets.json)
Frames     the encoder: a batch and a list of frames -> State, the next frame's
           coordinates, and which rows are scored. The semantic context c is a type
           embedding, a small MLP over the raw numbers the coordinates leave out, and
           the current order's ability; the input u is the current order (ability,
           target offset, has a target, progress) and whether the queue changed this
           step; the player context cp is the race. Velocities are differences from the
           previous frame, so a window's first frame is never stepped from; the ledger's
           rates are backward differences over up to RATE_LOOPS loops, since the bank
           moves in a worker trip's steps and a one-loop difference is spikes.

Rows: real when present, alive and on the map (passengers and units in gas buildings
stand where their parent is, so they are left out). A row is scored when it has an
observed next frame on the map; a death is scored (hp 0), a row out of sight is not.
"""

import json

import numpy as np
import torch
from torch import nn

from . import energy as E
from .physics_data import NEUTRAL, RAW_NUMS
from .tokenize import PC_KEYS

I = {n: i for i, n in enumerate(RAW_NUMS)}
PC = {n: i for i, n in enumerate(PC_KEYS)}
RATE_LOOPS = 22  # a second: the ledger's rates average over this many loops
ENERGY_MAX = 200.0
N_RAW_TYPES = 4096
LOOPS_PER_SECOND_EXPORT = 16.0  # the ruleset's speeds are per this many loops
INSTANT_ACCEL = 100.0  # export acceleration at or above which a type is instant
SPEED_AT, ACCEL_AT = 17, 18  # ruleset unit scalars: speed / 5, log1p(accel) / 8
# the raw numbers the context reads: what the coordinates leave out
EXTRAS = (
    "harvesters",
    "ideal_harvesters",
    "radius",
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
    "engaged",
)
N_RACES = 5


class Tables:
    """Per raw unit type (API id): value, build_time (loops), race, speed (map units a
    loop), mass (loops² a map unit: the inverse of acceleration; 0 when instant)."""

    def __init__(
        self,
        base_build,
        unit_data="data/worldmodel/unit_data.json",
        rulesets="data/worldmodel/rulesets.json",
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
        self.race = torch.from_numpy(race.astype(np.int64))
        self.k = torch.from_numpy(np.stack([value, mass, speed, build], 1))


def wrap(a):
    """Angles to (−π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class Frames(nn.Module):
    """A batch's frames as State. frames(b, ff) -> (State of M = B · len(ff) frames,
    q_next, p_next, scored). ff: frame indices in [1, F − 2]."""

    def __init__(self, vocab, tables, d, u_dim):
        super().__init__()
        self.tables = tables
        self.register_buffer("k_type", tables.k)
        self.register_buffer("race", tables.race)
        self.type_emb = nn.Embedding(vocab.n_units, d)
        self.abil = nn.Embedding(vocab.n_abilities, d)
        self.extras = E.mlp(len(EXTRAS), d, d)
        self.race_emb = nn.Embedding(N_RACES, d)
        self.abil_u = nn.Embedding(vocab.n_abilities, u_dim)
        self.reg_u = nn.Linear(5, u_dim)
        self.register_buffer("extras_at", torch.tensor([I[n] for n in EXTRAS]))

    def coords(self, b, ff):
        """[B, nf, R, len(COORDS)] the rows' coordinates at frames ff, velocities from
        the frame before (0 where either frame lacks the row)."""
        x, y, num = b["x"][:, ff], b["y"][:, ff], b["num"][:, ff]
        px, py, pnum = b["x"][:, ff - 1], b["y"][:, ff - 1], b["num"][:, ff - 1]
        both = (b["mask"][:, ff] & b["mask"][:, ff - 1]).to(x.dtype)
        z, pz = num[..., I["z"]], pnum[..., I["z"]]
        facing, pfacing = num[..., I["facing"]], pnum[..., I["facing"]]
        hp_max, sh_max = num[..., I["hp_max"]], num[..., I["shield_max"]]
        cols = [
            x,
            y,
            z,
            (x - px) * both,
            (y - py) * both,
            (z - pz) * both,
            facing,
            wrap(facing - pfacing) * both,
            num[..., I["hp"]] / hp_max.clamp(min=1),
            num[..., I["shield"]] / sh_max.clamp(min=1),
            num[..., I["energy"]] / ENERGY_MAX,
            num[..., I["weapon_cooldown"]],
            num[..., I["build"]],
            num[..., I["contents"]],
        ]
        return torch.stack(cols, -1)

    def ledger(self, b, ff):
        """[B, nf, P, len(PLAYER)] bank and supply (the batch stores log1p / 8) and
        their rates over up to RATE_LOOPS loops back."""
        pl = b["ply_num"]

        def raw(f):
            v = torch.expm1(pl[:, f, :, : len(PC_KEYS)] * 8)
            used, cap = v[..., PC["food_used"]], v[..., PC["food_cap"]]
            return torch.stack(
                [v[..., PC["minerals"]], v[..., PC["vespene"]], used, cap, cap - used],
                -1,
            )

        out = []
        for f in ff.tolist():
            kk = min(f, RATE_LOOPS)
            now = raw(f)
            rate = (now - raw(f - kk)) / kk if kk else torch.zeros_like(now)
            out.append(torch.cat([now, rate], -1))
        return torch.stack(out, 1)

    def frames(self, b, ff):
        ff = torch.as_tensor(ff, device=b["mask"].device)
        B, _, R = b["mask"].shape
        nf = len(ff)
        q, q_next = self.coords(b, ff), self.coords(b, ff + 1)
        p, p_next = self.ledger(b, ff), self.ledger(b, ff + 1)
        rtype = b["rtype"][:, ff].clamp(0, N_RAW_TYPES - 1)
        owner = b["owner"][:, None].expand(B, nf, R)
        owner = torch.where(owner == NEUTRAL, E.GROUND, owner + 1)
        on_map = lambda f: b["mask"][:, f] & (b["inside"][:, f] == 0)
        mask = on_map(ff) & b["alive"][:, ff]
        scored = mask & on_map(ff + 1)
        num = b["num"][:, ff]
        kt = self.k_type[rtype]  # value, mass, speed, build_time
        value = torch.where(owner == E.GROUND, q[..., E.Q["contents"]], kt[..., 0])
        k = torch.stack(
            [
                value,
                kt[..., 1],
                kt[..., 2],
                num[..., I["hp_max"]],
                num[..., I["shield_max"]],
                torch.full_like(value, ENERGY_MAX),
                kt[..., 3],
            ],
            -1,
        )
        abil = b["slot_abil"][:, ff, :, 0]
        extras = num[..., self.extras_at]
        extras = torch.sign(extras) * torch.log1p(extras.abs())
        c = self.type_emb(b["type"][:, ff]) + self.extras(extras) + self.abil(abil)
        reg = torch.cat(
            [
                b["slot_reg"][:, ff, :, 0, :],
                b["queue_changed"][:, ff][..., None].float(),
            ],
            -1,
        )
        u = self.abil_u(abil) + self.reg_u(reg)
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
    "type",
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
