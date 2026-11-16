"""The game's own data catalogs (GameData/*.xml inside the mods, extracted by
scripts/extract_gamedata.py), layered the way the engine loads them, with inheritance.

What the balance export leaves out lives here, above all what an upgrade *does*: a
CUpgrade's EffectArray lists (catalog, id, field) references with a value, applied once
per level, e.g.
    Effect,PunisherGrenadesU,Amount                 +1   (Marauder base damage per level)
    Effect,PunisherGrenadesU,AttributeBonus[Armored] +1   (and +1 vs Armored per level)
    Weapon,Claws,RateMultiplier                     +0.2  (adrenal glands; two mods add 0.2 each)
    Unit,Marine,LifeArmor                           +1

Layering (the ladder dependency chain): core, liberty, swarm, void, voidmulti. A later
mod's entry with the same id modifies the earlier one: scalar fields are set, array
entries with `index=` patch that position, entries without an index append,
`removed="1"` drops an entry. `parent=` inherits the resolved parent's fields and arrays
(a child's indexed patches refer to the inherited array). Entries with default="1" are
templates and never used directly.
"""

import copy
import os
import xml.etree.ElementTree as ET

MODS = ("core", "liberty", "swarm", "void", "voidmulti")  # load order
GAMEDATA = "data/worldmodel/gamedata"
# EffectArray fields that carry mechanics (the rest are icons, tooltips, actors)
MECHANICS = {
    "Amount",
    "AttributeBonus",
    "LifeArmor",
    "ShieldArmor",
    "LifeMax",
    "ShieldsMax",
    "EnergyMax",
    "Speed",
    "Range",
    "RateMultiplier",
    "Period",
    "Acceleration",
    "LifeRegenRate",
    "ShieldRegenRate",
    "EnergyRegenRate",
    "Sight",
    "DisplayAttackCount",
    "Radius",
    "Duration",
    "Chance",
    "MaxStackCount",
}


def mod_file(build, mod, catalog, root=GAMEDATA):
    return os.path.join(
        root,
        str(build),
        "mods",
        f"{mod}.sc2mod",
        "base.sc2data",
        "gamedata",
        f"{catalog.lower()}.xml",
    )


class Entry:
    """One catalog entry's accumulated operations (across mods), resolved against its
    parent."""

    def __init__(self, cid, tag):
        self.id, self.tag = cid, tag
        self.parent = None
        self.default = False
        # (kind, field, index, attrib) in mod order; kind: set | append | patch | remove
        self.ops = []

    def apply(self, el):
        if el.get("parent") and self.parent is None:
            self.parent = el.get("parent")
        if el.get("default") == "1":
            self.default = True
        for ch in el:
            if ch.tag is ET.Comment:
                continue
            idx = ch.get("index")
            attrs = {
                k: v for k, v in ch.attrib.items() if k not in ("index", "removed")
            }
            if ch.get("removed") == "1":
                self.ops.append(("remove", ch.tag, idx, attrs))
            elif ch.tag.endswith("Array"):
                self.ops.append(
                    ("patch" if idx is not None else "append", ch.tag, idx, attrs)
                )
            else:
                self.ops.append(("set", ch.tag, idx, attrs))


class Catalog:
    """All entries of one catalog (e.g. 'UpgradeData') for a build, layered over
    MODS."""

    def __init__(self, build, catalog, mods=MODS, root=GAMEDATA):
        self.entries = {}
        self.mods_seen = []
        for mod in mods:
            path = mod_file(build, mod, catalog, root)
            if not os.path.exists(path):
                continue
            self.mods_seen.append(mod)
            for el in ET.parse(path).getroot():
                if el.tag is ET.Comment or el.get("id") is None:
                    continue
                e = self.entries.setdefault(el.get("id"), Entry(el.get("id"), el.tag))
                e.apply(el)
        self._resolved = {}

    def resolve(self, cid):
        """{field: value or [array entries]} with inheritance and patches applied."""
        if cid in self._resolved:
            return self._resolved[cid]
        e = self.entries.get(cid)
        if e is None:
            return {}
        data = (
            copy.deepcopy(self.resolve(e.parent))
            if e.parent and e.parent != cid
            else {}
        )
        for kind, field, idx, attrs in e.ops:
            if kind == "set":
                data[field] = attrs
            elif kind == "append":
                data.setdefault(field, [])
                if not isinstance(data[field], list):
                    data[field] = []
                data[field].append(attrs)
            elif kind == "patch":
                arr = data.setdefault(field, [])
                if not isinstance(arr, list):
                    arr = data[field] = []
                i = int(idx) if idx.isdigit() else None
                # named index (e.g. AttributeBonus[Armored]-style arrays keyed by name)
                if i is None:
                    arr.append(dict(attrs, index=idx))
                else:
                    while len(arr) <= i:
                        arr.append({})
                    arr[i] = {**arr[i], **attrs}
            elif kind == "remove":
                if (
                    field in data
                    and isinstance(data[field], list)
                    and idx is not None
                    and idx.isdigit()
                ):
                    i = int(idx)
                    if i < len(data[field]):
                        data[field][i] = {"removed": "1"}
                else:
                    data.pop(field, None)
        self._resolved[cid] = data
        return data

    def ids(self, include_defaults=False):
        return [i for i, e in self.entries.items() if include_defaults or not e.default]


def parse_reference(ref):
    """'Effect,PunisherGrenadesU,AttributeBonus[Armored]' -> ('Effect',
    'PunisherGrenadesU', 'AttributeBonus', 'Armored')."""
    parts = ref.split(",")
    if len(parts) < 3:
        return None
    cat, cid, field = parts[0], parts[1], ",".join(parts[2:])
    key = None
    if "[" in field and field.endswith("]"):
        field, key = field[: field.index("[")], field[field.index("[") + 1 : -1]
    return cat, cid, field, key


def upgrade_effects(build, root=GAMEDATA):
    """{upgrade id: [(catalog, id, field, key, operation, value)]} for every
    non-template CUpgrade of the build, mechanics fields only (icons and tooltips
    dropped), plus {upgrade id: [affected unit ids]}."""
    cat = Catalog(build, "UpgradeData", root=root)
    effects, affected = {}, {}
    for uid in cat.ids():
        data = cat.resolve(uid)
        out = []
        for ent in data.get("EffectArray", []):
            if ent.get("removed") == "1" or "Reference" not in ent:
                continue
            ref = parse_reference(ent["Reference"])
            if ref is None or ref[2] not in MECHANICS:
                continue
            try:
                val = float(ent.get("Value", "0"))
            except ValueError:
                continue
            out.append(
                (ref[0], ref[1], ref[2], ref[3], ent.get("Operation", "Add"), val)
            )
        effects[uid] = out
        affected[uid] = [
            a["value"]
            for a in data.get("AffectedUnitArray", [])
            if isinstance(a, dict) and "value" in a
        ]
    return effects, affected, cat.mods_seen


if __name__ == "__main__":
    import sys

    build = int(sys.argv[1]) if len(sys.argv) > 1 else 98310
    eff, aff, mods = upgrade_effects(build)
    print(
        f"build {build}: mods {mods}, {len(eff)} upgrades,"
        f" {sum(len(v) for v in eff.values())} mechanic effects"
    )
    for uid in sys.argv[2:] or (
        "TerranInfantryWeaponsLevel1",
        "TerranInfantryWeaponsLevel3",
        "TerranInfantryArmorsLevel1",
        "zerglingattackspeed",
        "Stimpack",
    ):
        print(f"-- {uid}: affected {aff.get(uid)}")
        for e in eff.get(uid, []):
            print("    ", e)
