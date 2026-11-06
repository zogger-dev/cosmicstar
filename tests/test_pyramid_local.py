"""LocalLevel.attend against a dense reference: one softmax over all tokens with the
neighborhood rules written as an explicit mask."""

import os
import sys

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)
from cosmicstar.worldmodel.pyramid import LocalLevel  # noqa: E402


def reference(dbg, cmask, emask, gmask, h, w):
    B = cmask.shape[0]
    H, dh = dbg["qc"].shape[-2:]
    outs_c, outs_e, outs_g = [], [], []
    for b in range(B):
        qc, kc, vc = (
            dbg[k][b].reshape(h * w, H, dh).float() for k in ("qc", "kc", "vc")
        )
        qe, ke, ve = (dbg[k][b].float() for k in ("qe", "ke", "ve"))
        qg, kg, vg = (dbg[k][b].float() for k in ("qg", "kg", "vg"))
        N, Gn = qe.shape[0], qg.shape[0]
        cy, cx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        cy, cx = cy.reshape(-1), cx.reshape(-1)
        ey, ex = dbg["ey"][b].cpu(), dbg["ex"][b].cpu()
        ty = torch.cat(
            [
                cy,
                ey,
                torch.zeros(Gn, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            ]
        )
        tx = torch.cat(
            [
                cx,
                ex,
                torch.zeros(Gn, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            ]
        )
        kind = torch.cat(
            [torch.zeros(h * w), torch.ones(N), 2 * torch.ones(Gn), 3 * torch.ones(1)]
        ).long()  # cell unit global null
        valid = torch.cat(
            [
                cmask[b].reshape(-1).cpu(),
                emask[b].cpu(),
                gmask[b].cpu(),
                torch.ones(1, dtype=torch.bool),
            ]
        )
        K = torch.cat([kc, ke, kg, dbg["knull"].float()[None]], 0)
        V = torch.cat([vc, ve, vg, dbg["vnull"].float()[None]], 0)
        Qs = torch.cat([qc, qe, qg], 0)
        L = Qs.shape[0]
        near = ((ty[:L, None] - ty[None]).abs() <= 1) & (
            (tx[:L, None] - tx[None]).abs() <= 1
        )
        qk, kk = kind[:L, None], kind[None]
        m = torch.zeros(L, K.shape[0], dtype=torch.bool)
        local = (
            (qk < 2) & (kk < 2) & near
        )  # cells / units: neighborhood cells and units
        m |= local
        m |= (qk < 2) & (kk >= 2)  # everyone local sees globals and null
        m |= qk == 2  # globals see everything
        m &= valid[None]
        logits = torch.einsum("qhd,khd->hqk", Qs, K) * dh**-0.5
        logits = logits.masked_fill(~m[None].to(logits.device), float("-inf"))
        o = torch.einsum("hqk,khd->qhd", logits.softmax(-1), V)
        outs_c.append(o[: h * w])
        outs_e.append(o[h * w : h * w + N])
        outs_g.append(o[h * w + N :])
    return torch.cat(outs_c), torch.cat(outs_e), torch.stack(outs_g)


def test_local_attention_matches_dense():
    check(dl=64, D=96)  # a narrower map: units projected into it
    check(dl=128, D=128)  # one width throughout (the default): no projection


def check(dl, D):
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B, h, w, N, Gn, s = 2, 8, 12, 23, 3, 2
    lvl = LocalLevel(dl, D).to(dev)
    lvl.debug = True
    C = torch.randn(B, h, w, dl, device=dev)
    cmask = torch.ones(B, h, w, dtype=torch.bool, device=dev)
    cmask[1, :, -3:] = False  # off the map
    E = torch.randn(B, N, D, device=dev)
    emask = torch.ones(B, N, dtype=torch.bool, device=dev)
    emask[0, -5:] = False  # padding
    center = torch.tensor(
        [[w * s / 2, h * s / 2], [w * s / 2 - 3, h * s / 2 + 1]], device=dev
    )
    raw = torch.rand(B, N, 2, device=dev) * torch.tensor(
        [w * s * 0.75, h * s], device=dev
    )
    raw[0, :6] = raw[0, :1]  # a clump in one cell
    epos = raw - center[:, None]
    G = torch.randn(B, Gn, D, device=dev)
    gmask = torch.ones(B, Gn, dtype=torch.bool, device=dev)
    gmask[1, -1] = False
    oc, oe, og = lvl.attend(C, cmask, s, E, emask, epos, G, gmask, center)
    rc, re_, rg = reference(lvl._dbg, cmask, emask, gmask, h, w)
    ec = (oc.cpu() - rc.cpu()).abs().max().item()
    ee = (
        (
            (oe.cpu() - re_.cpu()).abs().max(-1).values.max(-1).values
            * emask.reshape(-1).cpu()
        )
        .max()
        .item()
    )
    eg = (
        ((og.float().cpu() - rg.cpu()).abs() * gmask.cpu()[..., None, None])
        .max()
        .item()
    )
    print(
        f"dl {dl} D {D}: max abs error: cells {ec:.2e} units {ee:.2e} globals {eg:.2e}"
    )
    assert ec < 1e-4 and ee < 1e-4 and eg < 1e-4
    # gradients flow to every input
    (oc.sum() + oe.sum() + og.sum()).backward()
    assert all(
        p.grad is not None
        for n, p in lvl.named_parameters()
        if n.startswith(("qkv", "k_null", "v_null"))
    )


if __name__ == "__main__":
    test_local_attention_matches_dense()
    print("ok")
