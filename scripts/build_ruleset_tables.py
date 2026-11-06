"""Turns each ruleset's balance export (the Editor's XML per unit,
scripts/export_balance.py) and engine rules dump (scripts/dump_rules.py) into the tables
the ruleset encoder reads (worldmodel/ruleset.py): one graph per ruleset of unit,
ability-command and upgrade-level nodes, each with a fixed-width feature vector, plus
typed edges between them.

Every optional XML element contributes its numbers and a presence bit (zeros when
absent), so "has no energy" is stated once. Weapon bonuses are a vector over the
attribute tags (Light, Armored, ...), one slot per tag, so a unit with no bonus is all
zeros and nesting disappears. Node names (unit id, "ability/command", upgrade level id)
are the stable keys across patches; the API ids of each ruleset map onto them through
the rules dump (units by name, ability commands by link name + command index, upgrades
by level name).

    python scripts/build_ruleset_tables.py   (-> data/worldmodel/rulesets.json)
"""

import argparse
import glob
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)
from cosmicstar.worldmodel import gamedata  # noqa: E402

ATTRIBUTES = (
    "Light",
    "Armored",
    "Biological",
    "Mechanical",
    "Psionic",
    "Massive",
    "Structure",
    "Heroic",
    "Hover",
    "Robotic",
    "Summoned",
)
RACES = ("Terr", "Zerg", "Prot", "Neut")
MOVEMENT = ("Ground", "Fly", "CliffJumper", "Colossus", "Burrowed", "Creep")
TARGETS = ("any", "ground", "air")
WEAPON_KINDS = ("Ranged", "Melee", "Splash", "Spell")
COMMAND_TYPES = (
    "command",
    "passive",
    "train",
    "build",
    "research",
)  # the last three: production commands, from
# the producer's <trains> / <builds> / <researches> lists (the export's <abilities>
# leave them out)
ABILITY_TARGETS = ("None", "Point", "Unit", "PointOrUnit")
# edge relations between nodes (u = unit, a = ability command, g = upgrade level)
RELATIONS = (
    "u_has_a",
    "a_of_u",
    "u_has_g",
    "g_of_u",
    "a_makes_u",
    "u_made_by_a",
    "a_makes_g",
    "g_made_by_a",
    "a_needs_u",
    "u_needed_by_a",
    "a_needs_g",
    "g_needed_by_a",
    "g_needs_g",
    "g_needed_by_g",
    "u_produced_by_u",
    "u_produces_u",
    "g_affects_u",
    "u_affected_by_g",
)
# per-level upgrade increments on a weapon, from the game data
# (gamedata.upgrade_effects): the weapon vector carries them after the export's fields
WEAPON_UPGRADE = (
    "has_gamedata",
    "amount_per_level",
    "rate_mult_per_level",
    "range_per_level",
    "period_per_level",
)  # + bonus per attribute
UPGRADE_FEATURES = (
    "has_gamedata",
    "n_amount",
    "amount",
    "n_bonus",
    "bonus",
    "life_armor",
    "shield_armor",
    "life_max",
    "shields_max",
    "speed",
    "rate_mult",
    "range",
    "n_affected",
)


def lg(x, s=1.0):
    return math.log1p(max(float(x or 0), 0.0)) / s


def num(el, attr, default=0.0):
    if el is None or el.get(attr) in (None, ""):
        return default
    try:
        return float(el.get(attr))
    except ValueError:
        return default


def onehot(value, choices):
    return [1.0 if value == c else 0.0 for c in choices]


def unit_scalars(root):
    """Fixed-width numbers of one unit XML (UNIT_SCALARS names them)."""
    life, shields, energy = root.find("life"), root.find("shields"), root.find("energy")
    armor, sharm, cost, mov, misc, meta = (
        root.find(k)
        for k in ("armor", "shieldArmor", "cost", "movement", "misc", "meta")
    )
    attrs = {a.get("type") for a in root.findall("attributes/attribute")}
    return [
        lg(num(life, "max"), 6),
        lg(num(life, "regenRate")),
        float(shields is not None),
        lg(num(shields, "max"), 6),
        lg(num(shields, "regenRate")),
        float(energy is not None),
        lg(num(energy, "max"), 6),
        lg(num(energy, "regenRate")),
        num(armor, "start") / 5,
        num(armor, "max") / 5,
        num(sharm, "start") / 5,
        float(cost is not None),
        lg(num(cost, "minerals"), 7),
        lg(num(cost, "vespene"), 7),
        num(cost, "supply") / 8,
        lg(num(cost, "time"), 5),
        float(mov is not None),
        num(mov, "speed") / 5,
        lg(num(mov, "acceleration"), 8),
        lg(num(mov, "turnRate"), 8),
        *onehot(mov.get("type") if mov is not None else None, MOVEMENT),
        num(misc, "radius"),
        num(misc, "sightRadius") / 12,
        num(misc, "cargoSize") / 8,
        lg(num(misc, "cargoCapacity"), 3),
        num(misc, "supply") / 8,  # supply provided (negative = consumed)
        *onehot(meta.get("race") if meta is not None else None, RACES),
        *[float(a in attrs) for a in ATTRIBUTES],
    ]


def weapon_vector(w, inc=None):
    """inc: per-level upgrade increments for this weapon from the game data (None: no
    data)."""
    misc, eff = w.find("misc"), w.find("effect")
    wid, eid = w.get("id"), eff.get("id") if eff is not None else None
    up = [0.0] * (len(WEAPON_UPGRADE) + len(ATTRIBUTES))
    if inc is not None:
        up[0] = 1.0
        up[1] = inc.get(("Effect", eid, "Amount", None), 0.0) / 10
        up[2] = inc.get(("Weapon", wid, "RateMultiplier", None), 0.0)
        up[3] = inc.get(("Weapon", wid, "Range", None), 0.0) / 10
        up[4] = inc.get(("Weapon", wid, "Period", None), 0.0)
        for i, a in enumerate(ATTRIBUTES):
            up[len(WEAPON_UPGRADE) + i] = (
                inc.get(("Effect", eid, "AttributeBonus", a), 0.0) / 10
            )
    radius = num(eff, "radius", -1.0)
    bonus, bonus_max = [0.0] * len(ATTRIBUTES), [0.0] * len(ATTRIBUTES)
    for b in w.findall("effect/bonus"):
        if b.get("type") in ATTRIBUTES:
            i = ATTRIBUTES.index(b.get("type"))
            bonus[i], bonus_max[i] = num(b, "damage") / 50, num(b, "max") / 50
    return [
        num(misc, "range") / 10,
        num(misc, "speed"),
        num(misc, "count", 1.0),
        num(eff, "damage") / 50,
        num(eff, "max") / 50,
        max(radius, 0.0),
        float(radius > 0),
        *onehot(misc.get("targets") if misc is not None else None, TARGETS),
        *onehot(eff.get("kind") if eff is not None else None, WEAPON_KINDS),
        *bonus,
        *bonus_max,
        *up,
    ]


def command_vector(cmd, target, ctype=None):
    meta, misc, cost = cmd.find("meta"), cmd.find("misc"), cmd.find("cost")
    return [
        *onehot(
            ctype or (meta.get("type") if meta is not None else None), COMMAND_TYPES
        ),
        float(misc is not None and misc.get("range") is not None),
        num(misc, "range") / 10,
        float(cost is not None),
        lg(num(cost, "minerals"), 7),
        lg(num(cost, "vespene"), 7),
        lg(num(cost, "time"), 5),
        lg(num(cost, "energy"), 5),
        lg(num(cost, "life"), 5),
        lg(num(cost, "cooldown"), 5),
        *onehot(target, ABILITY_TARGETS),
    ]


def level_vector(level, k, effects=None, affected=None):
    """Export fields (level index, costs) + what the level does, from the game data."""
    cost = level.find("cost")
    f = [0.0] * len(UPGRADE_FEATURES)
    if effects is not None:
        f[0] = 1.0
        am = [v for c, _, fld, key, op, v in effects if fld == "Amount"]
        bo = [v for c, _, fld, key, op, v in effects if fld == "AttributeBonus"]
        pick = lambda fld: [v for c, _, f_, key, op, v in effects if f_ == fld]
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        f[1], f[2] = lg(len(am)), mean(am) / 10
        f[3], f[4] = lg(len(bo)), mean(bo) / 10
        f[5], f[6] = mean(pick("LifeArmor")), mean(pick("ShieldArmor"))
        f[7], f[8] = mean(pick("LifeMax")) / 50, mean(pick("ShieldsMax")) / 50
        f[9], f[10], f[11] = (
            mean(pick("Speed")),
            mean(pick("RateMultiplier")),
            mean(pick("Range")) / 10,
        )
        f[12] = lg(len(affected or []))
    return [
        float(k) / 3,
        lg(num(cost, "minerals"), 7),
        lg(num(cost, "vespene"), 7),
        lg(num(cost, "time"), 5),
        *f,
    ]


def weapon_increments(effects, families):
    """{(catalog, id, field, key): per-level increment} summed over upgrade families.
    Each level of a family lists the same increments, so one representative level per
    family counts (all of its entries: adrenal glands adds 0.2 twice in one upgrade =
    0.4)."""
    rep = {}
    for uid in sorted(effects):
        rep.setdefault(families.get(uid, uid), uid)
    out = {}
    for uid in rep.values():
        for cat, cid, fld, key, op, val in effects[uid]:
            if op == "Add":
                out[(cat, cid, fld, key)] = out.get((cat, cid, fld, key), 0.0) + val
    return out


def build_ruleset(xml_dir, rules, build, links):
    """One ruleset's graph: nodes with features and API ids, and typed edges. build: the
    base build whose game data (data/worldmodel/gamedata/<build>) supplies the upgrade
    effects; links: {catalog link index (str): link name} from
    data/worldmodel/ability_links.json, which names the production abilities the export
    only refers to by index."""
    effects, affected, families, mods = {}, {}, {}, []
    if os.path.isdir(os.path.join(gamedata.GAMEDATA, str(build))):
        effects, affected, mods = gamedata.upgrade_effects(build)
        cat = gamedata.Catalog(build, "UpgradeData")
        families = {
            uid: (
                e.parent
                if e.parent
                and cat.entries.get(e.parent)
                and cat.entries[e.parent].default
                else uid
            )
            for uid, e in cat.entries.items()
        }
    inc = weapon_increments(effects, families) if effects else None
    by_name = {u["name"]: u["unit_id"] for u in rules["units"] if u["name"]}
    # ability commands by (link name, command index) -> API ids and target kind
    cmd_api, cmd_target = {}, {}
    for a in rules["abilities"]:
        key = (a["link_name"], int(a["link_index"]))
        cmd_api.setdefault(key, []).append(int(a["ability_id"]))
        cmd_target.setdefault(key, a.get("target", "None"))
    upg_api = {u["name"]: int(u["upgrade_id"]) for u in rules["upgrades"] if u["name"]}

    units, abilities, upgrades, edges = [], {}, {}, []
    uidx, by_catalog = (
        {},
        {},
    )  # by_catalog: (ability catalog index, command index) -> ability node idx

    def ability_node(ab_id, cmd, ctype=None, cmd_name=None):
        key = f"{ab_id}/{cmd_name or cmd.get('id')}"
        if key not in abilities:
            k = (ab_id, int(cmd.get("index") or 0))
            abilities[key] = dict(
                name=key,
                idx=len(abilities),
                api_ids=cmd_api.get(k, []),
                feat=command_vector(cmd, cmd_target.get(k, "None"), ctype),
                req_units=[],
                req_upgrades=[],
                makes_unit=[],
                makes_upgrade=[],
            )
            for r in cmd.findall("requires/unit"):
                abilities[key]["req_units"].append(r.get("id"))
            for r in cmd.findall("requires/upgrade"):
                abilities[key]["req_upgrades"].append(r.get("id"))
        return abilities[key]

    def upgrade_node(level, k):
        name = level.get("id")
        if name is None:
            return None
        if name not in upgrades:
            upgrades[name] = dict(
                name=name,
                idx=len(upgrades),
                api_id=upg_api.get(name, -1),
                feat=level_vector(
                    level, k, effects.get(name) if effects else None, affected.get(name)
                ),
                affected=list(affected.get(name, []))
                + [
                    cid
                    for c, cid, fld, key, op, v in effects.get(name, [])
                    if c == "Unit"
                ],
                req_units=[r.get("id") for r in level.findall("requires/unit")],
                req_upgrades=[r.get("id") for r in level.findall("requires/upgrade")],
            )
        return upgrades[name]

    for path in sorted(glob.glob(os.path.join(xml_dir, "*.xml"))):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        if root.tag != "unit":
            continue
        name = root.get("id")
        u = dict(
            name=name,
            idx=len(units),
            api_ids=[by_name[name]] if name in by_name else [],
            scal=unit_scalars(root),
            weapons=[weapon_vector(w, inc) for w in root.findall("weapons/weapon")],
            weapon_ids=[
                (
                    w.get("id"),
                    (
                        w.find("effect").get("id")
                        if w.find("effect") is not None
                        else None
                    ),
                )
                for w in root.findall("weapons/weapon")
            ],
            abilities=[],
            upgrades=[],
            producer=(
                root.find("producer").get("id")
                if root.find("producer") is not None
                else None
            ),
            products=[
                (t.get("id"), t.get("ability"), t.get("index"))
                for t in root.findall("trains/unit") + root.findall("builds/unit")
            ],
            researches=[
                (t.get("id"), t.get("ability"), t.get("index"))
                for t in root.findall("researches/upgrade")
            ],
        )
        for ab in root.findall("abilities/ability"):
            for cmd in ab.findall("command"):
                node = ability_node(ab.get("id"), cmd)
                u["abilities"].append(node["idx"])
                if ab.get("index") is not None:
                    by_catalog[(int(ab.get("index")), int(cmd.get("index") or 0))] = (
                        node["idx"]
                    )
        # production commands: one node per (producer's ability, command), named by the
        # ability's link name (stable across patches) and the product
        for section, ctype in (
            ("trains/unit", "train"),
            ("builds/unit", "build"),
            ("researches/upgrade", "research"),
        ):
            for el in root.findall(section):
                link = links.get(str(el.get("ability")))
                if link is None or el.get("id") is None:
                    continue
                node = ability_node(link, el, ctype, cmd_name=el.get("id"))
                u["abilities"].append(node["idx"])
                by_catalog[(int(el.get("ability")), int(el.get("index") or 0))] = node[
                    "idx"
                ]
        for up in root.findall("upgrades/upgrade"):
            for k, level in enumerate(up.findall("level")):
                node = upgrade_node(level, k + 1)
                if node is not None:
                    u["upgrades"].append(node["idx"])
        for el in root.findall(
            "researches/upgrade"
        ):  # single-level upgrades exist only here (Stimpack, ...)
            upgrade_node(el, 1)
        uidx[name] = u["idx"]
        units.append(u)

    def edge(rel, src, dst):
        edges.append((RELATIONS.index(rel), src, dst))

    for u in units:
        for a in u["abilities"]:
            edge("u_has_a", u["idx"], a)
            edge("a_of_u", a, u["idx"])
        for g in u["upgrades"]:
            edge("u_has_g", u["idx"], g)
            edge("g_of_u", g, u["idx"])
        if u["producer"] in uidx:
            edge("u_produced_by_u", u["idx"], uidx[u["producer"]])
            edge("u_produces_u", uidx[u["producer"]], u["idx"])
    # what a command makes: the producer's trains / builds / researches name the product
    # and the (ability catalog index, command index) of the command that makes it
    for u in units:
        for name, ab, ci in u["products"]:
            a = by_catalog.get((int(ab), int(ci or 0))) if ab is not None else None
            if a is not None and name in uidx:
                edge("a_makes_u", a, uidx[name])
                edge("u_made_by_a", uidx[name], a)
        for name, ab, ci in u["researches"]:
            a = by_catalog.get((int(ab), int(ci or 0))) if ab is not None else None
            if name is None:
                continue
            targets = (
                [upgrades[name]["idx"]]
                if name in upgrades
                else [g["idx"] for n, g in upgrades.items() if n.startswith(name)]
            )
            for g in targets if a is not None else []:
                edge("a_makes_g", a, g)
                edge("g_made_by_a", g, a)
    for a in abilities.values():
        for n in a["req_units"]:
            if n in uidx:
                edge("a_needs_u", a["idx"], uidx[n])
                edge("u_needed_by_a", uidx[n], a["idx"])
        for n in a["req_upgrades"]:
            if n in upgrades:
                edge("a_needs_g", a["idx"], upgrades[n]["idx"])
                edge("g_needed_by_a", upgrades[n]["idx"], a["idx"])
    for g in upgrades.values():
        for n in g["req_upgrades"]:
            if n in upgrades:
                edge("g_needs_g", g["idx"], upgrades[n]["idx"])
                edge("g_needed_by_g", upgrades[n]["idx"], g["idx"])
    if effects:
        weapon_owner = {}
        for u in units:
            for wid, eid in u["weapon_ids"]:
                weapon_owner.setdefault(("Weapon", wid), set()).add(u["idx"])
                weapon_owner.setdefault(("Effect", eid), set()).add(u["idx"])
        for name, g in upgrades.items():
            targets = {uidx[n] for n in g.get("affected", []) if n in uidx}
            for c, cid, fld, key, op, v in effects.get(name, []):
                targets |= weapon_owner.get((c, cid), set())
            for t in targets:
                edge("g_affects_u", g["idx"], t)
                edge("u_affected_by_g", t, g["idx"])
    abil = [
        dict(name=a["name"], api_ids=a["api_ids"], feat=a["feat"])
        for a in sorted(abilities.values(), key=lambda x: x["idx"])
    ]
    upgs = [
        dict(name=g["name"], api_id=g["api_id"], feat=g["feat"])
        for g in sorted(upgrades.values(), key=lambda x: x["idx"])
    ]
    units_out = [
        dict(name=u["name"], api_ids=u["api_ids"], scal=u["scal"], weapons=u["weapons"])
        for u in units
    ]
    return dict(
        units=units_out,
        abilities=abil,
        upgrades=upgs,
        edges=sorted(set(edges)),
        gamedata_mods=mods,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", default="data/worldmodel/balance")
    ap.add_argument("--rules", default="data/worldmodel/rules")
    ap.add_argument(
        "--links",
        default="data/worldmodel/ability_links.json",
        help="scripts/build_link_table.py's output",
    )
    ap.add_argument("--out", default="data/worldmodel/rulesets.json")
    args = ap.parse_args()
    all_links = json.load(open(args.links)) if os.path.exists(args.links) else {}
    out = dict(attributes=ATTRIBUTES, relations=RELATIONS, rulesets={})
    names = dict(units=set(), abilities=set(), upgrades=set())
    for d in sorted(glob.glob(os.path.join(args.balance, "*/"))):
        key = os.path.basename(d.rstrip("/"))
        rules_path = os.path.join(args.rules, key + ".json")
        if not os.path.exists(rules_path) or not glob.glob(os.path.join(d, "*.xml")):
            print(f"{key}: no rules dump or no XMLs, skipped")
            continue
        g = build_ruleset(
            d,
            json.load(open(rules_path)),
            int(key.split("_")[0]),
            all_links.get(key, {}).get("links", {}),
        )
        for k in names:
            names[k].update(n["name"] for n in g[k])
        mapped = (
            sum(bool(u["api_ids"]) for u in g["units"]),
            sum(bool(a["api_ids"]) for a in g["abilities"]),
            sum(g_["api_id"] >= 0 for g_ in g["upgrades"]),
        )
        print(
            f"{key}: {len(g['units'])} units ({mapped[0]} with API ids),"
            f" {len(g['abilities'])} ability commands ({mapped[1]} with API ids), "
            + (
                f"game data from {len(g['gamedata_mods'])} mods, "
                if g["gamedata_mods"]
                else "NO game data (upgrade effects missing), "
            )
            + f"{len(g['upgrades'])} upgrade levels ({mapped[2]} with API ids),"
            f" {len(g['edges'])} edges, {sum(len(u['weapons']) for u in g['units'])}"
            f" weapons (max {max(len(u['weapons']) for u in g['units'])} per unit)"
        )
        out["rulesets"][key] = g
    out["names"] = {k: sorted(v) for k, v in names.items()}
    out["widths"] = dict(
        unit=(
            len(out["rulesets"][next(iter(out["rulesets"]))]["units"][0]["scal"])
            if out["rulesets"]
            else 0
        ),
        weapon=7
        + len(TARGETS)
        + len(WEAPON_KINDS)
        + 3 * len(ATTRIBUTES)
        + len(WEAPON_UPGRADE),
        ability=len(COMMAND_TYPES) + 9 + len(ABILITY_TARGETS),
        upgrade=4 + len(UPGRADE_FEATURES),
    )
    json.dump(out, open(args.out, "w"))
    print(
        f"{len(out['rulesets'])} rulesets, names: "
        + ", ".join(f"{k} {len(v)}" for k, v in out["names"].items())
        + f" -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
