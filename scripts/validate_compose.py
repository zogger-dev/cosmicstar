"""Checks a composed player view (compose_view.py) against the engine's real view of that player.

Real units are matched by engine tag and compared field by field (unit references and ragged
lists - production, buffs, passengers - through tags). Snapshots (remembered structures in fog)
get different tags in each player's view, so they're compared by content (type, position,
owner, hp). Frames are compared up to the shorter file (a real slot-4 view stops before the
engine crash).

    python scripts/validate_compose.py --real pov/<name>_p2.npz --composed tmp/<name>_p2_composed.npz
"""

import argparse
import collections
import json

import numpy as np

FIELDS = ("type", "owner", "x", "y", "z", "facing", "radius", "hp", "hp_max", "shield", "shield_max", "energy",
          "build", "flags", "contents", "harvesters", "weapon_cooldown", "cargo_used", "display", "cloak")
RAGGED = dict(production=("u_prod_start", ("prod_abil", "prod_progress")), buffs=("u_buff_start", ("b_id",)),
              passengers=("u_pass_start", ("pass_id", "pass_type", "pass_hp")),
              orders=("u_ord_start", ("ord_abil", "ord_tx", "ord_ty", "ord_target", "ord_progress")))


def ragged_of(z, row, key):
    st, cols = RAGGED[key]
    a = z[st][row]
    b = z[st][row + 1] if row + 1 < len(z[st]) else len(z[cols[0]])
    out = []
    for c in cols:
        v = z[c][a:b]
        if c in ("pass_id", "ord_target"):
            v = np.where(v >= 0, z["unit_tag"][np.clip(v, 0, None)], 0)
        out.append(tuple(np.round(v.astype(np.float64), 3).tolist()))
    return tuple(out)


def ref_tag(z, row, fld):
    i = z[f"u_{fld}"][row]
    return int(z["unit_tag"][i]) if i >= 0 else 0


def validate(real, comp, every=1, slot=None):
    n = min(len(real["frame_loop"]), len(comp["frame_loop"]))
    assert np.array_equal(real["frame_loop"][:n], comp["frame_loop"][:n]), "frame loops differ"
    team = {p["slot"]: p["team"] for p in json.loads(str(real["meta"]))["players"]}
    slot = slot or json.loads(str(real["meta"]))["perspective"]
    stats = collections.defaultdict(lambda: [0, 0])
    examples = collections.defaultdict(list)
    missing = collections.Counter()
    snap = [0, 0]
    for f in range(0, n, every):
        ra, rb = real["frame_start"][f], real["frame_start"][f + 1]
        ca, cb = comp["frame_start"][f], comp["frame_start"][f + 1]
        r_snap = real["u_display"][ra:rb] == 2
        c_snap = comp["u_display"][ca:cb] == 2

        def content(z, a, b, m):
            return collections.Counter(zip(z["u_type"][a:b][m].tolist(), np.round(z["u_x"][a:b][m], 2).tolist(),
                                           np.round(z["u_y"][a:b][m], 2).tolist(), z["u_owner"][a:b][m].tolist(),
                                           np.round(z["u_hp"][a:b][m], 1).tolist()))
        rs, cs = content(real, ra, rb, r_snap), content(comp, ca, cb, c_snap)
        snap[0] += sum((rs & cs).values())
        snap[1] += sum(((rs - cs) + (cs - rs)).values())
        rt = {int(real["unit_tag"][real["u_id"][r]]): r for r in np.arange(ra, rb)[~r_snap]}
        ct = {int(comp["unit_tag"][comp["u_id"][r]]): r for r in np.arange(ca, cb)[~c_snap]}
        for t in set(rt) ^ set(ct):
            missing["only in real" if t in rt else "only in composed"] += 1
        for t in set(rt) & set(ct):
            i, j = rt[t], ct[t]
            o = int(real["u_owner"][i])
            who = "own" if o == slot else "ally" if team.get(o) == team.get(slot) else "enemy" if o in team else "neutral"
            checks = [(fld, real[f"u_{fld}"][i].item(), comp[f"u_{fld}"][j].item()) for fld in FIELDS]
            checks += [(fld, ref_tag(real, i, fld), ref_tag(comp, j, fld)) for fld in ("add_on", "engaged")]
            checks += [(k, ragged_of(real, i, k), ragged_of(comp, j, k)) for k in RAGGED
                       if RAGGED[k][0] in real and RAGGED[k][0] in comp]
            for fld, a, b in checks:
                eq = a == b or (isinstance(a, float) and np.isnan(a) and np.isnan(b))
                stats[(who, fld)][0] += eq
                stats[(who, fld)][1] += 1
                if not eq and len(examples[(who, fld)]) < 2:
                    examples[(who, fld)].append(dict(loop=int(real["frame_loop"][f]), type=int(real["u_type"][i]), real=a, composed=b))
    vis = {k: bool(np.array_equal(real[k][:n], comp[k][:n])) for k in ("vis", "creep") if k in real and k in comp}
    return dict(frames=n, every=every, missing=dict(missing), snapshots_same=snap[0], snapshots_diff=snap[1],
                maps_identical=vis, stats=stats, examples=examples)


def report(res):
    print(f"{res['frames']} frames (every {res['every']}); units in only one file: {res['missing'] or 'none'}; "
          f"snapshots: {res['snapshots_same']} same, {res['snapshots_diff']} different; maps identical: {res['maps_identical']}")
    total = [0, 0]
    for who in ("own", "ally", "enemy", "neutral"):
        rows = {fld: v for (w, fld), v in res["stats"].items() if w == who}
        if not rows:
            continue
        n = rows["type"][1]
        diffs = {fld: v for fld, v in rows.items() if v[0] < v[1]}
        for v in rows.values():
            total[0] += v[0]
            total[1] += v[1]
        print(f"  {who:8s} n={n:7d}: " + ("every field identical" if not diffs else
              "; ".join(f"{fld} {e / t:.2%}" for fld, (e, t) in sorted(diffs.items(), key=lambda x: x[1][0] / x[1][1]))))
        for fld in diffs:
            print(f"      {fld}: {res['examples'][(who, fld)]}")
    print(f"  overall: {total[0]}/{total[1]} field values identical ({total[0] / max(total[1], 1):.4%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--composed", required=True)
    ap.add_argument("--every", type=int, default=1, help="compare every n-th frame")
    args = ap.parse_args()
    report(validate(dict(np.load(args.real)), dict(np.load(args.composed)), args.every))


if __name__ == "__main__":
    main()
