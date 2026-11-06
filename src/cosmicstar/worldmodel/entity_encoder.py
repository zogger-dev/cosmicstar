"""The physics model's input tokens: one per unit row, per effect, per player and the
window's global token, and one per action. Each kind has its own encoder; a few
embeddings are shared between kinds (see EntityEncoder).

Every encoder takes the collated batch and returns tokens [B, F, n, d]; masking and the
layout of the sequence are the physics encoder's."""

import math

import torch
from torch import nn

from .action_tokens import (
    C_ACTION,
    N_ACT_KINDS,
    N_GROUP_OPS,
    N_GROUPS,
    N_REMOVE,
    N_SUBGROUPS,
)
from .encoder import mlp
from .physics_data import CODE, PROD, QUEUE, RAW_NUMS
from .tokenize import C_GROUPS, N_BUFF_IDS, N_EFFECT_IDS, N_RAW_ABILITY_IDS

I = {n: i for i, n in enumerate(RAW_NUMS)}
C_PLY = 14  # player numbers: PC_KEYS, the camera's x, y, known, and supply left
# a row's place: sin and cos of x and y at wavelengths from 4 to 256 map units
MAP_WAVES = 6
N_FLAGS = 16  # flags, flags2 bits
EFF_WAVES = 6  # an effect's age: sin and cos at wavelengths from 4 to 1024 loops
# embedded numbers per row, see row_numbers
ROW_NUMS = len(RAW_NUMS) + 8 + 4 * MAP_WAVES + N_FLAGS


def left_to_go(feat, j):
    """feat with its progress column j as 1 - progress: bfloat16 resolves values near 0
    finely and near 1 to 0.004, and completion is the moment that matters."""
    return torch.cat([feat[:, :j], 1 - feat[:, j : j + 1], feat[:, j + 1 :]], 1)


def effect_numbers(b):
    """[B, F, X, 2 + 2 * EFF_WAVES]: radius, and age in loops as log and as sin and cos
    at wavelengths from 4 to 1024 loops, so a delay such as a storm's ticks or a bile's
    landing reads exactly at any age."""
    age = b["eff_age"].float() if "eff_age" in b else torch.zeros_like(b["eff_radius"])
    w = (
        2
        * math.pi
        / torch.logspace(
            math.log10(4.0), math.log10(1024.0), EFF_WAVES, device=age.device
        )
    )
    a = age[..., None] * w
    return torch.cat(
        [b["eff_radius"][..., None], torch.log1p(age)[..., None] / 8, a.sin(), a.cos()],
        -1,
    ).to(b["eff_radius"].dtype)


def row_numbers(b):
    """[B, F, R, ROW_NUMS] scaled numbers of the rows."""
    n = b["num"]
    g = lambda k: n[..., I[k]]
    hpm, shm, bm = g("hp_max"), g("shield_max"), g("buff_max")
    log = lambda x, s=8.0: torch.log1p(x.clamp(min=0)) / s
    out = [
        (g("hp") / hpm.clamp(min=1)) * (hpm > 0),
        (g("shield") / shm.clamp(min=1)) * (shm > 0),
        g("energy") / 200,
        log(hpm),
        log(shm),
        1 - g("build"),  # left to go: bfloat16 is fine near 0, where completion is
        log(g("contents")),
        g("harvesters") / 16,
        g("ideal_harvesters") / 16,
        g("facing").sin(),
        g("facing").cos(),
        g("radius") / 4,
        g("weapon_cooldown").clamp(min=0) / 32,
        g("cargo_used") / 8,
        g("cargo_max") / 8,
        (g("buff_remain") / bm.clamp(min=1)) * (bm > 0),
        log(bm),
        g("detect_range") / 12,
        g("radar_range") / 32,
        g("atk_up") / 3,
        g("arm_up") / 3,
        g("shd_up") / 3,
        g("add_on") >= 0,
        g("engaged") >= 0,
        g("z") / 16,
    ]
    out += [(g("display") == k) for k in (1, 2, 3)] + [
        (g("cloak") == k) for k in (0, 1, 2, 3)
    ]
    out += [b["inside"] == 1, b["inside"] == 2, ~b["alive"], b["parent_row"] >= 0]
    # flags, flags2: hallucinated, powered, burrowed, selected and the rest, as bits
    bits = g("flags").long() | (g("flags2").long() << 8)
    out += list(
        ((bits[..., None] >> torch.arange(N_FLAGS, device=n.device)) & 1).unbind(-1)
    )
    # where on the map, in map units: map knowledge (ramps, chokes) is tied to place
    w = (
        2
        * math.pi
        / torch.logspace(math.log10(4.0), math.log10(256.0), MAP_WAVES, device=n.device)
    )
    for k in ("x", "y"):
        a = b[k][..., None].float() * w
        out += list(a.sin().unbind(-1)) + list(a.cos().unbind(-1))
    return torch.stack([x.to(n.dtype) for x in out], -1)


class ActionEncoder(nn.Module):
    """[B, F, A, d]: one token per action, the sum of its parts' embeddings through an
    MLP. Owns the family, button and player embeddings the entity encoders share."""

    def __init__(self, avocab, d):
        super().__init__()
        self.kind = nn.Embedding(N_ACT_KINDS, d)
        self.family = nn.Embedding(avocab.n_families, d, padding_idx=0)
        self.button = nn.Embedding(avocab.n_buttons, d, padding_idx=0)
        self.tkind = nn.Embedding(4, d)
        self.group = nn.Embedding(N_GROUPS, d, padding_idx=0)
        self.gop = nn.Embedding(N_GROUP_OPS, d, padding_idx=0)
        self.remove = nn.Embedding(N_REMOVE, d, padding_idx=0)
        self.subgroup = nn.Embedding(N_SUBGROUPS, d, padding_idx=0)
        self.num = mlp(C_ACTION, d)
        self.mix = mlp(d, d)
        self.player = nn.Embedding(2, d)

    def forward(self, b):
        n = (
            self.kind(b["act_kind"])
            + self.family(b["act_family"])
            + self.button(b["act_button"])
            + self.tkind(b["act_tkind"])
            + self.group(b["act_group"])
            + self.gop(b["act_gop"])
            + self.remove(b["act_remove"])
            + self.subgroup(b["act_subgroup"])
            + self.num(b["act_num"])
            + self.player(b["act_player"])
        )
        return self.mix(n)


class UnitEncoder(nn.Module):
    """[B, F, R, d]: one token per unit row. Its type, owner, numbers and control
    groups; pointer codes for itself and the rows it engages, sits in and orders target;
    and the mean of each list it carries: orders, production, buffs, rallies, available
    abilities and autocast settings. Autocast reads the action encoder's family and
    button embeddings."""

    def __init__(self, vocab, d, family, button):
        super().__init__()
        self.unit = nn.Embedding(vocab.n_units, d)
        self.ability = nn.Embedding(vocab.n_abilities, d)
        self.owner = nn.Embedding(3, d)
        self.num = mlp(ROW_NUMS, d)
        self.groups = nn.Linear(C_GROUPS, d)
        self.buff = nn.Embedding(N_BUFF_IDS, d)
        self.bag_in = nn.ModuleDict(
            dict(ord=mlp(4, d), prod=mlp(2, d), rally=mlp(3, d))
        )
        self.bag_slot = nn.ModuleDict(
            dict(ord=nn.Embedding(QUEUE, d), prod=nn.Embedding(PROD, d))
        )  # which place in the queue: 0 is now
        # pointers as inputs: each row's random identity code, and the codes of the rows
        # it points at, each kind through its own map
        self.code_in = nn.ModuleDict(
            {
                k: nn.Linear(CODE, d, bias=False)
                for k in ("self", "engaged", "container", "order")
            }
        )
        self.raw_ability = nn.Embedding(N_RAW_ABILITY_IDS, d)
        self.auto_on = nn.Embedding(2, d)
        self.family, self.button = family, button  # shared with ActionEncoder

    def _bag(self, name, b, emb):
        """Mean of a list-valued state per token: [B * F * R] flat index, embedded
        items."""
        where, B, F_, R = b[f"{name}_at"], *b["mask"].shape
        out = emb.new_zeros(B * F_ * R, emb.shape[-1]).index_add(0, where, emb)
        cnt = (
            torch.bincount(where, minlength=B * F_ * R)
            .clamp(min=1)[:, None]
            .to(emb.dtype)
        )
        return (out / cnt).view(B, F_, R, -1)

    def forward(self, b):
        B, F_, R = b["mask"].shape
        code = b["row_code"]  # [B, R, CODE]
        dt = b["num"].dtype

        def pointed(bi, row):
            """The code of the row pointed at, zeros where row < 0 (no target)."""
            return torch.where((row >= 0)[..., None], code[bi, row.clamp(min=0)], 0).to(
                dt
            )

        bb = torch.arange(B, device=code.device)[:, None, None]
        ids = (
            self.unit(b["type"].long())
            + self.owner(b["owner"])[:, None]
            + self.num(row_numbers(b))
            + self.groups(b["groups"].to(dt))
        )
        ids = (
            ids
            + self.code_in["self"](code.to(dt))[:, None]
            + self.code_in["engaged"](pointed(bb, b["engaged_row"].long()))
            + self.code_in["container"](pointed(bb, b["parent_row"].long()))
        )
        o_b = b["ord_at"] // (F_ * R)
        ids = ids + self._bag(
            "ord",
            b,
            self.ability(b["ord_abil"])
            + self.bag_in["ord"](left_to_go(b["ord_feat"], 3))
            + self.bag_slot["ord"](b["ord_slot"])
            + self.code_in["order"](pointed(o_b, b["ord_trow"])),
        )
        ids = ids + self._bag(
            "prod",
            b,
            self.ability(b["prod_abil"])
            + self.bag_in["prod"](left_to_go(b["prod_feat"], 0))
            + self.bag_slot["prod"](b["prod_slot"]),
        )
        ids = ids + self._bag("buff", b, self.buff(b["buff_id"]))
        ids = ids + self._bag("rally", b, self.bag_in["rally"](b["rally_feat"]))
        ids = ids + self._bag("ab", b, self.raw_ability(b["ab_id"]))
        ids = ids + self._bag(
            "auto",
            b,
            self.family(b["auto_fam"])
            + self.button(b["auto_but"])
            + self.auto_on(b["auto_on"]),
        )
        return ids


class EffectEncoder(nn.Module):
    """[B, F, X, d]: one token per effect of a frame: its type, owner (the unit
    encoder's owner embedding), radius and age. Effects have no ids, so no timeline."""

    def __init__(self, d, owner):
        super().__init__()
        self.effect = nn.Embedding(N_EFFECT_IDS, d)
        self.num = nn.Linear(2 + 2 * EFF_WAVES, d)
        self.owner = owner  # shared with UnitEncoder

    def forward(self, b):
        return (
            self.effect(b["eff_type"])
            + self.owner(b["eff_owner"])
            + self.num(effect_numbers(b))
        )


class GlobalEncoder(nn.Module):
    """[B, F, 3, d]: the window's global token (who sees the state, the map's size) and
    one token per player (race, player numbers, which player: the action encoder's
    player embedding)."""

    def __init__(self, d, player):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, d))
        self.view = nn.Linear(2, d)  # which players see the state: [1, 1] for physics
        self.map_wh = nn.Linear(2, d)
        self.race = nn.Embedding(4, d)
        self.ply_num = mlp(C_PLY, d)
        self.player = player  # shared with ActionEncoder
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, b):
        B, F_ = b["mask"].shape[:2]
        p = (
            self.ply_num(b["ply_num"])
            + self.race(b["players"])[:, None]
            + self.player.weight[None, None]
        )
        c = (self.cls + self.view(b["view"]) + self.map_wh(b["half"] / 100))[
            :, None
        ].expand(B, F_, -1)
        return torch.cat([c[:, :, None], p], 2)


class EntityEncoder(nn.Module):
    """Every entity token of a window: (globals [B, F, 3, d], units [B, F, R, d],
    effects [B, F, X, d]). Units and effects share the owner embedding; units read the
    action encoder's family and button embeddings for autocast, and players its player
    embedding."""

    def __init__(self, vocab, d, actions):
        super().__init__()
        self.globals = GlobalEncoder(d, actions.player)
        self.units = UnitEncoder(vocab, d, actions.family, actions.button)
        self.effects = EffectEncoder(d, self.units.owner)

    def forward(self, b):
        return self.globals(b), self.units(b), self.effects(b)


# old checkpoints: PhysicsEncoder attributes -> their places now (several where shared)
_MOVED = {
    "unit": ["entities.units.unit"],
    "ability": ["entities.units.ability"],
    "owner": ["entities.units.owner", "entities.effects.owner"],
    "num_in": ["entities.units.num"],
    "groups": ["entities.units.groups"],
    "buff": ["entities.units.buff"],
    "bag_in": ["entities.units.bag_in"],
    "bag_slot": ["entities.units.bag_slot"],
    "code_in": ["entities.units.code_in"],
    "raw_ability": ["entities.units.raw_ability"],
    "auto_on": ["entities.units.auto_on"],
    "effect": ["entities.effects.effect"],
    "eff_num": ["entities.effects.num"],
    "cls": ["entities.globals.cls"],
    "view_in": ["entities.globals.view"],
    "map_wh": ["entities.globals.map_wh"],
    "race": ["entities.globals.race"],
    "ply_in": ["entities.globals.ply_num"],
    "player": ["acts.player", "entities.globals.player"],
    "a_kind": ["acts.kind"],
    "a_family": ["acts.family", "entities.units.family"],
    "a_button": ["acts.button", "entities.units.button"],
    "a_tkind": ["acts.tkind"],
    "a_group": ["acts.group"],
    "a_gop": ["acts.gop"],
    "a_remove": ["acts.remove"],
    "a_subgroup": ["acts.subgroup"],
    "a_num": ["acts.num"],
    "a_mix": ["acts.mix"],
}


def upgrade_state(sd, encoders=("enc", "teacher")):
    """A state dict from before the entity encoders were split out, with its encoder
    keys renamed to their places now; a newer one comes back as it is."""
    out = {}
    for k, v in sd.items():
        head, _, rest = k.partition(".")
        attr, _, tail = rest.partition(".")
        if head in encoders and attr in _MOVED:
            for new in _MOVED[attr]:
                out[f"{head}.{new}.{tail}" if tail else f"{head}.{new}"] = v
        else:
            out[k] = v
    return out


def old_name(name):
    """A parameter's name now -> its name before the split (the inverse of _MOVED,
    taking the first place of a shared one)."""
    head, _, rest = name.partition(".")
    for attr, news in _MOVED.items():
        new = news[0]
        if rest == new or rest.startswith(new + "."):
            return f"{head}.{attr}{rest[len(new):]}"
    return name


def upgrade_optimizer(opt_sd, old_keys, model):
    """An optimizer state saved before the split, for model's trainable parameters now.
    The optimizer holds parameters by position; the old positions are the old state
    dict's order of trainable parameters (old_keys, buffers and the frozen teacher left
    out by the caller), the new ones model.parameters()' order."""
    old_at = {k: i for i, k in enumerate(old_keys)}
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    idx = [old_at[old_name(n)] for n in names]
    (group,) = opt_sd["param_groups"]
    assert len(group["params"]) == len(old_keys) == len(names)
    state = {i: opt_sd["state"][j] for i, j in enumerate(idx) if j in opt_sd["state"]}
    return dict(state=state, param_groups=[dict(group, params=list(range(len(names))))])
