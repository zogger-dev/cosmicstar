"""Builds the per-build ability table from the Editor's balance exports.

The replay names an ability by (link, cmd): the ability's link in that build's catalog and the
button within it. The Editor's export of a replay's rules gives the link numbers:

    python scripts/export_balance.py   (-> data/worldmodel/balance/<ruleset>)

Each exported unit lists its abilities by catalog name and link (<ability id= index=>) and the
units/upgrades it builds, trains or researches by link only (<unit id= ability= index=>); those
links are named through the API, which says which ability makes each unit and upgrade. The
API's AbilityData then expands every link to all its buttons: link_index == the replay's cmd.

Output (--out): {ruleset: {"base_build", "links": {link: link_name}, "units": {unit link: name},
                         "commands": {"link:cmd": {"name", "index", "api", "button"}}}}
The replay-vote table (build_ability_table.py), where present, is checked against it.

    python scripts/build_link_table.py
"""

import argparse
import collections
import glob
import json
import os
import re

ABILITY = re.compile(r'<ability id="([^"]+)" index="(\d+)"')
PRODUCES = re.compile(r'<(unit|upgrade) id="([^"]+)" ability="(\d+)" index="(\d+)"')
UNIT_LINK = re.compile(r'<unit id="([^"]+)">\s*<meta [^>]*\bindex="(\d+)"')  # the file's own unit


def export_units(d):
    """{unit link: unit name} from one balance export (fogged snapshots and pings name units
    by link)."""
    out = {}
    for f in glob.glob(os.path.join(d, "*.xml")):
        m = UNIT_LINK.search(open(f, encoding="utf-8").read())
        if m:
            out[int(m.group(2))] = m.group(1)
    return out


def export_links(d):
    """{link: {link_name, or (producer, "unit"|"upgrade", name, cmd)}} from one balance export.
    The producer is the unit whose file lists the build/train/research entry."""
    out = collections.defaultdict(set)
    for f in glob.glob(os.path.join(d, "*.xml")):
        x = open(f, encoding="utf-8").read()
        root = re.search(r'<unit id="([^"]+)"', x)
        producer = root.group(1) if root else ""
        for name, link in ABILITY.findall(x):
            out[int(link)].add(name)
        for kind, name, link, cmd in PRODUCES.findall(x):
            out[int(link)].add((producer, kind, name, int(cmd)))
    return out


def name_produced(items, family, makes, abil, loose=False, taken=()):
    """Name a link known only by what it produces. Several abilities can make the same unit
    (GatewayTrain and WarpGateTrain both make Zealots), so the API's "made by" alone is not
    enough: score each candidate by how many items sit at the same button (by button name or by
    the API's "made by"), and prefer abilities named after the producing unit.
    loose: also accept a button name that merely contains the item's name (the Nuke's button is
    NukeArm); only used for links nothing else names, since it confuses e.g. CreepTumorBuild."""
    producers = {p for p, _, _, _ in items}
    cands = {n for n in family if any(n.startswith(p) for p in producers if p)}

    def same(button, name):
        return button and (button == name or (loose and name in button))

    for _, kind, name, cmd in items:
        a = abil.get(makes.get((kind, name)))
        if a and a.get("link_name"):
            cands.add(a["link_name"])
        if loose:  # melee abilities only: campaign ones (SpectreNuke) are marked unavailable
            cands.update(n for n, f in family.items()
                         if same(f.get(cmd, {}).get("button_name"), name) and f[cmd].get("available"))

    def score(n):
        hit = 0
        for _, kind, name, cmd in items:
            b = family[n].get(cmd)
            if b and (same(b.get("button_name"), name) or b["ability_id"] == makes.get((kind, name))):
                hit += 1
        own = hit > 0 and any(n.startswith(p) for p in producers if p)  # the producer's own ability
        return own, hit

    best = sorted(cands - set(taken), key=lambda n: (score(n), n), reverse=True)
    return best[0] if best and score(best[0])[1] >= 1 else None


def build_table(export, rules):
    abil = {a["ability_id"]: a for a in rules["abilities"]}
    makes = {("unit", u["name"]): u.get("ability_id") for u in rules["units"]}
    makes.update({("upgrade", u["name"]): u.get("ability_id") for u in rules["upgrades"]})
    family = collections.defaultdict(dict)  # link_name -> {link_index: AbilityData}
    for a in abil.values():
        if a.get("link_name"):
            family[a["link_name"]][a["link_index"]] = a
    links, problems, unnamed = {}, [], []
    for link, refs in sorted(export.items()):
        direct = {r for r in refs if isinstance(r, str)}
        produced = [r for r in refs if not isinstance(r, str)]
        if len(direct) > 1:
            problems.append(f"link {link}: several names {sorted(direct)}")
        name = min(direct) if direct else name_produced(produced, family, makes, abil)
        if name is None:
            if produced:
                unnamed.append((link, produced))
            continue  # otherwise only campaign / co-op things the API doesn't know
        if direct and produced and name_produced(produced, family, makes, abil) not in (None, name):
            problems.append(f"link {link}: named {name} but produces like {name_produced(produced, family, makes, abil)}")
        links[link] = name
    for link, produced in unnamed:  # last resort, among the names no other link has
        name = name_produced(produced, family, makes, abil, loose=True, taken=links.values())
        if name:
            links[link] = name
    taken = collections.defaultdict(list)
    for link, name in links.items():
        taken[name].append(link)
    for name, ls in taken.items():
        if len(ls) > 1:
            problems.append(f"{name} named by several links {ls}")
    commands = {}
    for link, name in links.items():
        if name not in family:
            problems.append(f"link {link}: {name} has no API ability")
            continue
        for idx, a in family[name].items():
            commands[f"{link}:{idx}"] = dict(name=name, index=idx, api=a["ability_id"], button=a.get("button_name", ""))
    return links, commands, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", default="data/worldmodel/balance")
    ap.add_argument("--rules", default="data/worldmodel/rules")
    ap.add_argument("--votes", default="data/worldmodel/ability_table.json")
    ap.add_argument("--out", default="data/worldmodel/ability_links.json")
    args = ap.parse_args()
    index = json.load(open(os.path.join(args.rules, "index.json")))
    votes = json.load(open(args.votes)) if os.path.exists(args.votes) else {}
    table = {}
    for d in sorted(glob.glob(os.path.join(args.balance, "*"))):
        key = os.path.basename(d)
        if not os.path.isdir(d) or key not in index:
            continue
        build = index[key]["base_build"]
        rules = json.load(open(os.path.join(args.rules, key + ".json")))
        links, commands, problems = build_table(export_links(d), rules)
        table[key] = dict(base_build=build, links=links, commands=commands, units=export_units(d))
        # cross-check against the replay votes for this build
        checked = disagree = 0
        api_name = {a["ability_id"]: a["link_name"] for a in rules["abilities"]}
        for vkey, t in votes.items():
            b, link, cmd = map(int, vkey.split(":"))
            if b != build:
                continue
            checked += 1
            c = commands.get(f"{link}:{cmd}")
            if c is None or c["api"] != t["api"]:
                disagree += 1
                problems.append(f"vote {vkey}: replays say {api_name.get(t['api'])} ({t['votes']} votes), "
                                f"export says {c and c['name']}")
        print(f"{key}: {len(links)} links, {len(commands)} commands; votes checked {checked}, disagree {disagree}")
        for p in problems:
            print("   ", p)
    json.dump(table, open(args.out, "w"), indent=0)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
