"""The one-pass model's dependencies on a real window, exactly: a frame's clean state holds no action of that frame and nothing later."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from train_physics import EXCLUDED, MOVING_NEUTRALS, Games, largest_windows  # noqa: E402

from cosmicstar.worldmodel import physics, physics_data  # noqa: E402
from cosmicstar.worldmodel.action_tokens import ActionVocab  # noqa: E402
from cosmicstar.worldmodel.vocab import Vocab  # noqa: E402

L = 12


def load():
    vocab, avocab = Vocab(json.load(open("data/worldmodel/frame_cache_seq/vocab.json"))), ActionVocab()
    split = json.load(open("data/games/splits/default.json"))
    bases = physics_data.base_paths("data/games_arrow", set(split["train"]) - set(EXCLUDED))
    games = Games(bases, vocab, avocab)
    b = None
    for gid in sorted(bases)[:40]:
        g = games.get(gid)
        for t0 in range(600, g.F - L - 1, 40):
            w = games.read(gid, t0, L, False)
            c = physics_data.collate([w], vocab)
            f = c["act_mask"][0].any(1).nonzero().flatten()
            f = f[(f >= 2) & (f < L - 2)]
            if len(f) and (c["act_target"][0, f[0]] >= 0).any():
                b = c, int(f[0])
                break
        if b:
            break
    return physics.Physics(vocab, avocab, 128, moving_neutrals=MOVING_NEUTRALS, blocks=3).eval(), *b


def diff(a, c):
    """Largest change per frame across rows, cells and globals."""
    return torch.stack([(a[k] - c[k]).flatten(2).abs().amax(2) for k in ("ent", "cells", "glob")]).amax(0)[0]


@torch.no_grad()
def main():
    torch.manual_seed(0)
    model, b, t = load()
    enc = model.enc
    for blk in enc.blocks:
        torch.nn.init.normal_(blk.act.o.weight, std=0.02)     # it starts at zero, the identity: actions would show no effect at all
        torch.nn.init.normal_(blk.bias, std=0.5)
    clean, upd = enc(b)
    print("frames", L, "action frame", t, "rows", b["mask"].shape[2], "grid", b["grid"].shape[2:4], "finite", all(torch.isfinite(v).all() for v in (*clean.values(), *upd.values())), flush=True)
    tol = 1e-5

    # an action at frame t, its numbers and its target: not the clean state at t or before, not updated before t; updated at t; clean afterwards
    c = {k: v.clone() for k, v in b.items()}
    m = c["act_mask"][0, t]
    c["act_num"][0, t][m] += 1.0
    c["act_target"][0, t][m & (c["act_target"][0, t] >= 0)] = 0
    a, u = enc(c)
    dc, du = diff(clean, a), diff(upd, u)
    print("change an action   clean", [f"{x:.0e}" for x in dc.tolist()], "\n                 updated", [f"{x:.0e}" for x in du.tolist()], flush=True)
    assert dc[:t + 1].max() < tol and du[:t].max() < tol and du[t] > tol and dc[t + 1] > tol

    # a row's state, the map and a global at t: nothing earlier
    for name, change in (("row", lambda c: c["num"][0, t].__iadd__(1.0)), ("map", lambda c: c["grid"][0, t].__ixor__(torch.tensor(1, dtype=torch.uint8))), ("player", lambda c: c["ply_num"][0, t].__iadd__(1.0))):
        c = {k: v.clone() for k, v in b.items()}
        change(c)
        a, u = enc(c)
        dc, du = diff(clean, a), diff(upd, u)
        print(f"change {name:6s} clean", [f"{x:.0e}" for x in dc.tolist()], flush=True)
        assert dc[:t].max() < tol and du[:t].max() < tol and dc[t] > tol

    # a padded row changes nothing
    if (~b["mask"][0, t]).any():
        c = {k: v.clone() for k, v in b.items()}
        r = (~b["mask"][0, t]).nonzero()[0, 0]
        c["x"][0, t, r] += 5
        c["num"][0, t, r] += 5
        a, u = enc(c)
        real = {k: (clean[k], a[k]) for k in ("cells", "glob")}
        change = max((p - n).abs().max() for p, n in real.values())
        change = max(change, ((clean["ent"] - a["ent"]).abs().amax(-1) * b["mask"]).max())   # its own output is unused
        print("change an absent row, others", f"{change:.0e}", flush=True)
        assert change < tol
    print("no leaks")


if __name__ == "__main__":
    main()
