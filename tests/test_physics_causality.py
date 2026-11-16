"""The one-pass model's dependencies on a real window, exactly: a frame's clean state
holds no action of that frame and nothing later."""

import glob
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from train_physics import EXCLUDED, Games  # noqa: E402

from cosmicstar.worldmodel import physics, physics_data  # noqa: E402
from cosmicstar.worldmodel.action_tokens import ActionVocab  # noqa: E402
from cosmicstar.worldmodel.vocab import Vocab  # noqa: E402

L = 16


def load(tubelet=4):
    vocab, avocab = (
        Vocab(json.load(open("data/worldmodel/frame_cache_seq/vocab.json"))),
        ActionVocab(),
    )
    ids = {
        os.path.basename(p)[: -len(".index.npz")]
        for p in glob.glob("data/games_v3/1v1/*/*/*.index.npz")
    }  # any game will do: nothing is learned here
    bases = physics_data.base_paths("data/games_v3", ids - set(EXCLUDED))
    games = Games(bases, vocab, avocab)
    b = None
    lo = 2 * tubelet - 1 if tubelet > 1 else 2
    hi = L - tubelet - 1 if tubelet > 1 else L - 2
    for gid in sorted(bases)[:40]:
        g = games.get(gid)
        for t0 in range(600, g.F - L - 1, 40):
            w = games.read(gid, t0, L, False)
            c = physics_data.collate([w], vocab)
            f = c["act_mask"][0].any(1).nonzero().flatten()
            f = f[(f >= lo) & (f <= hi)]
            if len(f) and (c["act_target"][0, f[0]] >= 0).any():
                b = c, int(f[0])
                break
        if b:
            break
    dh = 64 if tubelet > 1 else 32
    ff = 1024 if tubelet > 1 else 512
    return (
        physics.Physics(
            vocab,
            avocab,
            128,
            moving_neutrals=physics_data.MOVING_NEUTRALS,
            blocks=3,
            tubelet=tubelet,
            dh=dh,
            ff=ff,
        ).eval(),
        *b,
    )


def diff(a, c):
    """Largest change per step/frame across rows, cells and globals."""
    return torch.stack(
        [(a[k] - c[k]).flatten(2).abs().amax(2) for k in ("ent", "cells", "glob")]
    ).amax(0)[0]


@torch.no_grad()
def check_mode(tubelet):
    torch.manual_seed(0)
    model, b, t = load(tubelet=tubelet)
    enc = model.enc
    k = max(1, tubelet)
    for blk in enc.modules():
        if isinstance(blk, physics.Block):
            torch.nn.init.normal_(
                blk.act.o.weight, std=0.02
            )  # it starts at zero, the identity: actions would show no effect at all
            torch.nn.init.normal_(blk.bias, std=0.5)
            torch.nn.init.normal_(blk.space_bias, std=0.5)
    clean, upd = enc(b)
    print(
        f"[tubelet={k}] frames",
        L,
        "action frame",
        t,
        "rows",
        b["mask"].shape[2],
        "grid",
        b["grid"].shape[2:4],
        "finite",
        all(torch.isfinite(v).all() for v in (*clean.values(), *upd.values())),
        flush=True,
    )
    tol = 1e-5

    # Action at frame t belongs to transition chunk m_a = (t - (k - 1)) // k:
    # clean[0 .. m_a] unchanged, upd[< m_a] unchanged, upd[m_a] changed,
    # and clean[m_a + 1] changed.
    m_a = (t - (k - 1)) // k
    c = {k_: v.clone() for k_, v in b.items()}
    m = c["act_mask"][0, t]
    c["act_num"][0, t][m] += 1.0
    c["act_target"][0, t][m & (c["act_target"][0, t] >= 0)] = 0
    a, u = enc(c)
    dc, du = diff(clean, a), diff(upd, u)
    print(
        "change an action   clean",
        [f"{x:.0e}" for x in dc.tolist()],
        "\n                 updated",
        [f"{x:.0e}" for x in du.tolist()],
        flush=True,
    )
    assert (
        dc[: m_a + 1].max() < tol
        and du[:m_a].max() < tol
        and du[m_a].item() > tol
        and dc[m_a + 1].item() > tol
    )

    # A selection at t pointing at a unit it adds: same causal boundary m_a
    c = {k_: v.clone() for k_, v in b.items()}
    slot = int(c["act_mask"][0, t].nonzero()[0, 0])
    f_anc = m_a * k + (k - 1)
    row = int(c["mask"][0, f_anc].nonzero()[-1, 0])
    c["act_sel"] = torch.tensor([[t, slot, row]])
    a, u = enc(c)
    dc, du = diff(clean, a), diff(upd, u)
    print(
        "change a selection clean",
        [f"{x:.0e}" for x in dc.tolist()],
        "\n                 updated",
        [f"{x:.0e}" for x in du.tolist()],
        flush=True,
    )
    assert (
        dc[: m_a + 1].max() < tol
        and du[:m_a].max() < tol
        and du[m_a].item() > tol
    )

    # Intra-tubelet action token causality & cross-player action isolation:
    # 1) A late action at f_late = f_anc + (k - 1) inside tubelet m_a must NOT leak
    #    backward into earlier loops' action tokens upd["act"][:, :f_late].
    # 2) Player 1's action inside tubelet m_a must NOT leak into Player 0's action
    #    tokens upd["act"][:, f_anc : f_anc + k, :K] of the same tubelet.
    K_slots = b["act_mask"].shape[2] // 2
    if K_slots > 0:
        f_late = f_anc + (k - 1)
        c_base = {k_: v.clone() for k_, v in b.items()}
        c_base["act_mask"][0, f_anc, 0] = True
        c_base["act_player"][0, f_anc, 0] = 0
        c_base["act_kind"][0, f_anc, 0] = 1
        c_base["act_mask"][0, f_late, 0] = True
        c_base["act_player"][0, f_late, 0] = 0
        c_base["act_kind"][0, f_late, 0] = 1
        _, u_base = enc(c_base)

        if k > 1:
            c_late = {k_: v.clone() for k_, v in c_base.items()}
            c_late["act_num"][0, f_late, 0] += 5.0
            c_late["act_target"][0, f_late, 0] = row
            _, u_late = enc(c_late)
            intra_leak = (
                (u_late["act"][:, :f_late] - u_base["act"][:, :f_late])
                .abs()
                .max()
                .item()
            )
            print(f"intra-tubelet reverse act leak: {intra_leak:.0e}", flush=True)
            assert intra_leak < tol
            assert (
                (u_late["act"][:, f_late] - u_base["act"][:, f_late])
                .abs()
                .max()
                .item()
                > tol
            )

        c_p1 = {k_: v.clone() for k_, v in c_base.items()}
        c_p1["act_mask"][0, f_anc, K_slots] = True
        c_p1["act_player"][0, f_anc, K_slots] = 1
        c_p1["act_kind"][0, f_anc, K_slots] = 1
        c_p1["act_num"][0, f_anc, K_slots] += 5.0
        c_p1["act_target"][0, f_anc, K_slots] = row
        _, u_p1 = enc(c_p1)
        opp_leak = (
            (
                u_p1["act"][:, f_anc : f_anc + k, :K_slots]
                - u_base["act"][:, f_anc : f_anc + k, :K_slots]
            )
            .abs()
            .max()
            .item()
        )
        print(f"cross-player simultaneous act leak: {opp_leak:.0e}", flush=True)
        assert opp_leak < tol

    # Macro-count isolation: perturbing food_cap..larva_count (incl. warp_gate_count)
    # and supply_left across all frames must change NOTHING in GlobalEncoder / trunk.
    c_macro = {k_: v.clone() for k_, v in b.items()}
    c_macro["ply_num"][:, :, :, (2, 3, 4, 5, 6, 7, 8, 9, 13)] += 100.0
    am, um = enc(c_macro)
    macro_leak = max(diff(clean, am).max().item(), diff(upd, um).max().item())
    print(f"macro-count input leak: {macro_leak:.0e}", flush=True)
    assert macro_leak == 0.0

    # Anchor-frame persistent state perturbations inside tubelet m_s:
    # nothing in earlier tubelets (< m_s) changes; tubelet m_s changes.
    t_anc = (t // k) * k + (k - 1)
    m_s = t_anc // k
    live_rows = b["mask"][0, t_anc].nonzero().flatten()
    r0, r1 = int(live_rows[0]), int(live_rows[-1])
    all_changes = dict(
        row=lambda c: c["num"][0, t_anc].__iadd__(1.0),
        ptr=lambda c: c["engaged_row"][0, t_anc, r0].fill_(r1),
        map=lambda c: c["grid"][0, t_anc].__ixor__(
            torch.tensor(1, dtype=c["grid"].dtype)
        ),
        player=lambda c: c["ply_num"][0, t_anc, :, :2].__iadd__(1.0),
    )
    for name in ("row", "ptr", "map", "player"):
        c = {k_: v.clone() for k_, v in b.items()}
        all_changes[name](c)
        a, u = enc(c)
        dc, du = diff(clean, a), diff(upd, u)
        print(
            f"change {name:6s} clean",
            [f"{x:.0e}" for x in dc.tolist()],
            flush=True,
        )
        assert (
            (m_s == 0 or (dc[:m_s].max() < tol and du[:m_s].max() < tol))
            and dc[m_s].item() > tol
        )

    # A padded/absent row slot at t_anc changes nothing real
    if (~b["mask"][0, t_anc]).any():
        c = {k_: v.clone() for k_, v in b.items()}
        r = (~b["mask"][0, t_anc]).nonzero()[0, 0]
        c["x"][0, t_anc, r] += 5
        c["num"][0, t_anc, r] += 5
        c["engaged_row"][0, t_anc, r] = r0
        c["parent_row"][0, t_anc, r] = r0
        c["ord_row"][0, t_anc, r, 0] = r0
        a, u = enc(c)
        real = {k_: (clean[k_], a[k_]) for k_ in ("cells", "glob")}
        change = max((p - n).abs().max() for p, n in real.values())
        kf_idx = (
            torch.arange(L // k) * k + (k - 1) if k > 1 else torch.arange(L)
        )
        valid_m = b["mask"][:, kf_idx]
        change = max(
            change, ((clean["ent"] - a["ent"]).abs().amax(-1) * valid_m).max()
        )
        print("change an absent row, others", f"{change:.0e}", flush=True)
        assert change < tol

    # Non-anchor sub-frame invariance & effect tubelet causality when k > 1
    if k > 1:
        t_sub = (t // k) * k + 1  # non-anchor sub-frame (anchor is + 3)
        c_sub = {k_: v.clone() for k_, v in b.items()}
        c_sub["x"][0, t_sub] += 25.0
        c_sub["num"][0, t_sub] += 50.0
        c_sub["engaged_row"][0, t_sub, r0] = r1
        c_sub["grid"][0, t_sub] ^= torch.tensor(1, dtype=c_sub["grid"].dtype)
        c_sub["ply_num"][0, t_sub] += 10.0
        cs1, us1 = enc(c_sub)
        sub_leak = max(diff(clean, cs1).max().item(), diff(upd, us1).max().item())
        print(f"non-anchor state sub-frame leak: {sub_leak:.0e}", flush=True)
        assert sub_leak == 0.0

        # Effect 4->1 tubelet causality and masked-slot invariance
        m_e = t_anc // k
        f_e0 = m_e * k
        ce = {k_: v.clone() for k_, v in b.items()}
        X_e = max(2, ce["eff_mask"].shape[2] + 2)
        pad_e = lambda t_, val=0: (
            torch.nn.functional.pad(t_, (0, X_e - t_.shape[2]), value=val)
            if t_.ndim == 3
            else torch.nn.functional.pad(t_, (0, 0, 0, X_e - t_.shape[2]), value=val)
        )
        for ek, ev in (
            ("eff_mask", False),
            ("eff_type", 0),
            ("eff_owner", 2),
            ("eff_view", 0),
            ("eff_xy", 0.0),
            ("eff_radius", 0.0),
            ("eff_age", 0.0),
        ):
            if ek in ce:
                ce[ek] = pad_e(ce[ek], ev)
        ce["eff_mask"][0, f_e0 : f_e0 + k, X_e - 2] = True
        ce["eff_type"][0, f_e0 : f_e0 + k, X_e - 2] = 1
        ce["eff_owner"][0, f_e0 : f_e0 + k, X_e - 2] = 0
        ce["eff_view"][0, f_e0 : f_e0 + k, X_e - 2] = 0
        ce["eff_xy"][0, f_e0 : f_e0 + k, X_e - 2] = torch.tensor([3.5, -2.25])
        ce["eff_radius"][0, f_e0 : f_e0 + k, X_e - 2] = 1.5 / 8.0
        ce["eff_age"][0, f_e0 : f_e0 + k, X_e - 2] = torch.arange(k, dtype=torch.float32)
        ce["eff_mask"][0, f_e0 : f_e0 + k, X_e - 1] = False

        ec1, eu1 = enc(ce)
        dc_e, du_e = diff(clean, ec1), diff(upd, eu1)
        assert m_e == 0 or (dc_e[:m_e].max() < tol and du_e[:m_e].max() < tol)
        assert dc_e[m_e].item() > tol

        # Perturbing masked-out effect slot X_e-1 must change nothing anywhere
        ce_m = {k_: v.clone() for k_, v in ce.items()}
        ce_m["eff_xy"][0, f_e0, X_e - 1] = torch.tensor([99.0, -77.0])
        ce_m["eff_age"][0, f_e0, X_e - 1] = 123.0
        ec2, eu2 = enc(ce_m)
        eff_mask_leak = max(diff(ec1, ec2).max().item(), diff(eu1, eu2).max().item())
        if eu1["eff"].shape[2] > 0:
            eff_mask_leak = max(
                eff_mask_leak,
                (eu1["eff"] - eu2["eff"]).abs().max().item(),
            )
        print(f"masked effect slot leak: {eff_mask_leak:.0e}", flush=True)
        assert eff_mask_leak < tol

        # Perturbing active effect age/radius at m_e must not leak to earlier tubelets [:m_e]
        ce_p = {k_: v.clone() for k_, v in ce.items()}
        ce_p["eff_age"][0, f_e0 : f_e0 + k, X_e - 2] += 10.0
        ec3, eu3 = enc(ce_p)
        if m_e > 0 and eu1["eff"].shape[2] == eu3["eff"].shape[2] and eu1["eff"].shape[2] > 0:
            eff_tok_leak = (eu1["eff"][:, :m_e] - eu3["eff"][:, :m_e]).abs().max().item()
            print(f"eff token backward leak: {eff_tok_leak:.0e}", flush=True)
            assert eff_tok_leak < tol

    # Verify neutral critters never appear in births
    if len(b["birth_r"]):
        born_owners = b["owner"][b["birth_b"], b["birth_r"]]
        assert (born_owners != physics_data.NEUTRAL).all(), "neutral birth leaked!"

    with torch.enable_grad():
        # Verify detached probes flow ZERO gradient into encoder / trunk / main heads
        model.zero_grad(set_to_none=True)
        cl_g, up_g = model.enc(b)
        m_start = 1 if (k > 1 and (L // k) > 2) else 0
        m_end = (L // k - 1) if k > 1 else (L - 1)
        f_src = (
            torch.arange(m_start, m_end) * k + (k - 1)
            if k > 1
            else torch.arange(m_start, m_end)
        )
        f_tgt = f_src + k
        _, pr_ply, _ = model.player_losses(
            up_g["glob"][:, m_start:m_end, 1:3],
            cl_g["glob"][:, m_start:m_end, 1:3],
            b,
            f_src,
            f_tgt,
        )
        _, pr_pb, _ = model.map_loss(
            up_g["cells"][:, m_start:m_end],
            cl_g["cells"][:, m_start:m_end],
            b,
            f_src,
            f_tgt,
        )
        (pr_ply + pr_pb).backward()
        trunk_modules = (
            model.enc,
            model.g,
            model.head,
            model.births,
            model.eff_persist,
            model.eff_spawns,
            model.map_dec,
            model.map_src_mlp,
            model.map_stay,
            model.player_next,
            model.recip,
        )
        for mod in trunk_modules:
            for p_ in mod.parameters():
                assert p_.grad is None, "detached probe leaked gradient into trunk!"
        assert any(
            p_.grad is not None and p_.grad.abs().max() > 0
            for p_ in model.ply_probe.parameters()
        )
        assert any(
            p_.grad is not None and p_.grad.abs().max() > 0
            for p_ in model.pb_probe.parameters()
        )
        model.zero_grad(set_to_none=True)

        # Full forward + backward including a player birth, a persisting GuardianShield,
        # a spawned PsiStorm circle, and a spawned 9-dot LurkerSpines collinear ray
        b_grad = {k_: v.clone() for k_, v in b.items()}
        f_s0 = int(f_src[0].item())
        f_b = int(f_tgt[0].item())
        p_rows = (
            (b_grad["owner"][0] != physics_data.NEUTRAL) & b_grad["mask"][0, f_b]
        ).nonzero().flatten()
        if len(p_rows):
            r_b = int(p_rows[0])
            b_grad["birth_count"][0, f_b - 1] += 1
            b_grad["birth_b"] = torch.cat([b_grad["birth_b"], torch.tensor([0])])
            b_grad["birth_f"] = torch.cat([b_grad["birth_f"], torch.tensor([f_b])])
            b_grad["birth_r"] = torch.cat([b_grad["birth_r"], torch.tensor([r_b])])
            b_grad["birth_p"] = torch.cat([b_grad["birth_p"], torch.tensor([-1])])

        X_add = 12
        pad_eg = lambda t_, val=0: (
            torch.nn.functional.pad(t_, (0, X_add), value=val)
            if t_.ndim == 3
            else torch.nn.functional.pad(t_, (0, 0, 0, X_add), value=val)
        )
        x0_i = b_grad["eff_mask"].shape[2]
        for ek, ev in (
            ("eff_mask", False),
            ("eff_type", 0),
            ("eff_owner", 2),
            ("eff_view", 0),
            ("eff_xy", 0.0),
            ("eff_radius", 0.0),
            ("eff_age", 0.0),
        ):
            if ek in b_grad:
                b_grad[ek] = pad_eg(b_grad[ek], ev)
        # Persisting GuardianShield (id=2) from f_s0 (age=5) -> f_b (age=5+k)
        b_grad["eff_mask"][0, f_s0, x0_i] = True
        b_grad["eff_type"][0, f_s0, x0_i] = 2
        b_grad["eff_owner"][0, f_s0, x0_i] = 0
        b_grad["eff_xy"][0, f_s0, x0_i] = torch.tensor([2.0, 3.0])
        b_grad["eff_radius"][0, f_s0, x0_i] = 4.0 / 8.0
        b_grad["eff_age"][0, f_s0, x0_i] = 5.0

        b_grad["eff_mask"][0, f_b, x0_i] = True
        b_grad["eff_type"][0, f_b, x0_i] = 2
        b_grad["eff_owner"][0, f_b, x0_i] = 0
        b_grad["eff_xy"][0, f_b, x0_i] = torch.tensor([2.3, 3.1])
        b_grad["eff_radius"][0, f_b, x0_i] = 4.0 / 8.0
        b_grad["eff_age"][0, f_b, x0_i] = float(5 + k)

        # Spawned PsiStorm circle (id=1, age=0) at f_b
        b_grad["eff_mask"][0, f_b, x0_i + 1] = True
        b_grad["eff_type"][0, f_b, x0_i + 1] = 1
        b_grad["eff_owner"][0, f_b, x0_i + 1] = 1
        b_grad["eff_xy"][0, f_b, x0_i + 1] = torch.tensor([-4.0, 1.5])
        b_grad["eff_radius"][0, f_b, x0_i + 1] = 1.5 / 8.0
        b_grad["eff_age"][0, f_b, x0_i + 1] = 0.0

        # Spawned 9-dot LurkerSpines ray (id=12, step=1.0, age=0) at f_b
        for d_i in range(9):
            sl_e = x0_i + 2 + d_i
            b_grad["eff_mask"][0, f_b, sl_e] = True
            b_grad["eff_type"][0, f_b, sl_e] = 12
            b_grad["eff_owner"][0, f_b, sl_e] = 0
            b_grad["eff_xy"][0, f_b, sl_e] = torch.tensor([float(d_i), 0.0])
            b_grad["eff_radius"][0, f_b, sl_e] = 0.5 / 8.0
            b_grad["eff_age"][0, f_b, sl_e] = 0.0

        loss, logs = model(b_grad)
        assert torch.isfinite(loss), f"non-finite loss: {logs}"
        loss.backward()
        for name_m, mod_m in (
            ("eff_persist", model.eff_persist),
            ("eff_spawns", model.eff_spawns),
            ("map_src_mlp", model.map_src_mlp),
            ("map_stay", model.map_stay),
            ("stay_type", model.head.stay_type),
            ("births.cnt_pool", model.births.cnt_pool),
            ("eff_spawns.cnt_pool", model.eff_spawns.cnt_pool),
            ("eff_spawns.eff_q", model.eff_spawns.eff_q),
            ("eff_spawns.eff_type_in", model.eff_spawns.eff_type_in),
        ):
            assert any(
                p_.grad is not None and p_.grad.abs().max() > 0
                for p_ in mod_m.parameters()
            ), f"{name_m} got zero gradient!"
        model.zero_grad(set_to_none=True)


def main():
    check_mode(tubelet=4)
    check_mode(tubelet=1)
    print("no leaks")


if __name__ == "__main__":
    main()
