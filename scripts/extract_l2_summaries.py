"""Extracts 5-second (112-loop) summary frames from a trained Layer 1 EnergyModel +
Frames encoder across all patch 5.0.14 games (train + held_out_games) for Layer 2
(60 x 5s = 5-minute horizon) training and evaluation.

For each 5s step tau (at loop 112 * tau), saves strictly the un-bottlenecked latents,
grounded unit coordinates Q (no manual macro ledgers), Layer 1 base energies, and
5-second window action summaries:
  - cls_out:   [T, d] bf16          (un-bottlenecked CLS token before z_ply_proj)
  - ply_out:   [T, 2, d] bf16       (un-bottlenecked Player 1 & 2 tokens before z_ply_proj)
  - G_qq:      [T, 2, 36] fp32      (grounded 8x8 upper-tri unit Gram Q_p^T Q_p / N0)
  - G_qc:      [T, 2, 8, d] bf16    (full-rank 8xd Cross-Gram Q_p^T C_p / N0)
  - u_c:       [N_live, d] bf16     (live unit d=256 normalized latents C_p so exact
                                     full-rank 256x256 G_cc = C_p^T C_p / N0 is
                                     materialized on GPU in O(0.1ms) without 5GB disk I/O)
  - u_owner:   [N_live] uint8       (1 = P1, 2 = P2)
  - frame_ptr: [T + 1] int32        (CSR offsets into u_c / u_owner per 5s step tau)
  - H1_l1, H2_l1, A1_l1: [T] fp32   (Layer 1 base Hamiltonian outputs for additive EBM)
  - u_win:     [T, 2, u_dim] fp32   (Layer 1 control impulses pooled over the 5s window)
  - act_feat:  [T, 2, 24] fp32      (5s window physical command summary from acts.tok)
  - u_active:  [T, 2] bool          (whether player p issued commands in the 5s window)
  - loops:     [T] int32            (game loop of each 5s summary frame)
"""

import argparse
import gc
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import train_energy as te
import train_physics as tp
from cosmicstar.worldmodel import arrow_state, energy, energy_data, physics_data
from cosmicstar.worldmodel.action_tokens import ActionVocab
from cosmicstar.worldmodel.vocab import Vocab


def _window_action_summary(g, t_start, stride=112):
    """Aggregates physical player commands in [t_start, t_start + stride) from g.acts.tok
    into a [2, 24] float32 feature vector and [2] active boolean per 5s window."""
    tk = g.acts.tok
    lo, hi = t_start + 1, t_start + 1 + stride
    idx = np.nonzero((tk["loop"] >= lo) & (tk["loop"] < hi))[0]
    feat = np.zeros((2, 24), dtype=np.float32)
    active = np.zeros(2, dtype=bool)
    if len(idx) == 0:
        return feat, active

    kind = tk["kind"][idx].astype(np.int64)
    is_cmd = (kind == 1) | (kind == 2) | (kind == 11)
    idx = idx[is_cmd]
    if len(idx) == 0:
        return feat, active

    ply = (tk["player"][idx].astype(np.int64) - 1).clip(0, 1)
    kind = tk["kind"][idx].astype(np.int64)
    fam = tk["family"][idx].clip(0, 15).astype(np.int64)
    xy = np.nan_to_num(np.stack([tk["x"][idx], tk["y"][idx]], 1) - g.center) / 100.0
    mins = np.log1p(tk["minerals"][idx].clip(0)) / 8.0
    gas = np.log1p(tk["gas"][idx].clip(0)) / 8.0
    queued = tk["queued"][idx].astype(np.float32)
    smart = tk["smart"][idx].astype(np.float32)
    minimap = tk["minimap"][idx].astype(np.float32)

    for p in (0, 1):
        m = ply == p
        if not np.any(m):
            continue
        active[p] = True
        cnt = float(np.sum(m))
        feat[p, 0] = np.log1p(cnt) / 4.0
        feat[p, 1] = float(np.sum(kind[m] == 1)) / max(1.0, cnt)
        feat[p, 2] = float(np.sum(kind[m] == 2)) / max(1.0, cnt)
        feat[p, 3] = float(np.sum(kind[m] == 11)) / max(1.0, cnt)
        feat[p, 4] = float(np.sum(mins[m])) / 4.0
        feat[p, 5] = float(np.sum(gas[m])) / 4.0
        feat[p, 6] = float(np.mean(queued[m]))
        feat[p, 7] = float(np.mean(smart[m]))
        feat[p, 8] = float(np.mean(minimap[m]))
        feat[p, 9:11] = np.mean(xy[m], axis=0)
        feat[p, 11:13] = np.std(xy[m], axis=0)
        # Histogram over command families (0..10)
        for f_id in range(11):
            feat[p, 13 + f_id] = np.log1p(float(np.sum(fam[m] == f_id))) / 3.0

    return feat, active


def encode_bursts_with_full_latents(enc, model, b, f_idx):
    """Runs Layer 1 Frames._encode_sequence on b at f_idx, returning both the standard
    r=8 state (for Layer 1 H^(1)) AND the un-bottlenecked d=256 cls_out, ply_out, and
    normalized unit latents c_norm before z_proj."""
    f_idx_t = torch.as_tensor(f_idx, device=b["mask"].device)
    B, F_total, R = b["mask"].shape
    nf = len(f_idx)

    q = enc.coords(b, f_idx_t)  # [B, nf, R, K_Q]
    p = enc.ledger(b, f_idx_t)  # [B, nf, 2, 2]

    rtype = b["rtype"][:, f_idx_t].clamp(0, energy_data.N_RAW_TYPES - 1)
    owner = b["owner"][:, None].expand(B, nf, R)
    owner = torch.where(owner == energy_data.NEUTRAL, energy.GROUND, owner + 1)
    on_map = b["mask"][:, f_idx_t] & (b["inside"][:, f_idx_t] == 0)
    mask = on_map & b["alive"][:, f_idx_t]

    num = b["num"][:, f_idx_t]
    kt = enc.k_type[rtype]
    kt_comp = torch.sign(kt) * torch.log1p(kt.abs()) / 8.0

    abil = b["slot_abil"][:, f_idx_t, :, 0]
    prod_abil = b["slot_abil"][:, f_idx_t, :, physics_data.QUEUE]

    extras = num[..., enc.extras_at]
    extras = torch.sign(extras) * torch.log1p(extras.abs()) / 8.0
    ext_in = torch.cat([extras, kt_comp], dim=-1)

    xy_raw = torch.stack([b["x"][:, f_idx_t], b["y"][:, f_idx_t]], dim=-1).float()
    xy_ang = xy_raw.unsqueeze(-1) * enc.map_omega
    xy_fourier = torch.cat(
        [xy_ang.sin().flatten(-2), xy_ang.cos().flatten(-2)], dim=-1
    ).to(q.dtype)

    facing = num[..., energy_data.I["facing"]]
    ord_reg = b["slot_reg"][:, f_idx_t, :, 0, :]
    kin_in = torch.cat(
        [q, xy_fourier, facing.sin().unsqueeze(-1), facing.cos().unsqueeze(-1), ord_reg],
        dim=-1,
    )

    dev = b["mask"].device
    center = (
        b["center"][:, None, None, :].float()
        if "center" in b
        else torch.full((B, 1, 1, 2), 100.0, device=dev)
    )
    pos_ent = xy_raw - center
    S = enc.field.grid_size * enc.field.grid_size
    pos_field = enc.field.anchors(b)[:, None, :, :].expand(B, nf, S, 2)
    pos_ef = torch.cat([pos_ent, pos_field], dim=2)

    eff_tok, eff_mask, eff_xy = enc._embed_effects(b, f_idx_t)
    g0, X = 3, eff_tok.shape[2]
    T_t = g0 + R + S
    T = T_t + X

    unit_c, ord_c, prod_c, _ = enc.nodes(rtype, abil, prod_abil)
    field_all = enc.field(b, f_idx_t, pos_ef)
    dt = energy_data._compute_dtype(enc.cls_tok)
    field_ent = field_all[:, :, :R]
    field_tok = field_all[:, :, R:] + enc.field.field_emb[None, None].to(dt)
    c_ent = (
        unit_c
        + ord_c
        + prod_c
        + enc.owner_emb(owner).to(dt)
        + enc.kin_c(kin_in)
        + enc.extras(ext_in)
        + field_ent
    ).to(dt)
    g_tok = enc._embed_globals(b, f_idx_t).to(dt)
    x_tok = torch.cat([g_tok, c_ent, field_tok.to(dt), eff_tok.to(dt)], dim=2).reshape(
        B * nf, T, -1
    )

    valid_g = torch.ones(B, nf, g0, dtype=torch.bool, device=dev)
    valid_field = torch.ones(B, nf, S, dtype=torch.bool, device=dev)
    valid_all = torch.cat([valid_g, mask, valid_field, eff_mask], dim=2)
    valid_flat = valid_all.reshape(B * nf, 1, 1, T)

    t_valid = torch.cat([valid_g, mask, valid_field], dim=2)
    t_mask = t_valid.transpose(1, 2).reshape(B * T_t, nf)
    tri = torch.tril(torch.ones(nf, nf, dtype=torch.bool, device=dev))
    eye = torch.eye(nf, dtype=torch.bool, device=dev)
    allow_t = ((tri[None] & t_mask[:, None, :]) | eye[None])[:, None]
    t_mask_3d = t_mask.unsqueeze(-1)

    pos_g = torch.zeros(B, nf, g0, 2, device=dev, dtype=torch.float32)
    pos_all = torch.cat([pos_g, pos_ent, pos_field, eff_xy], dim=2)

    cs_2d = energy_data.cos_sin(enc.rope_2d.angles(pos_all.reshape(B * nf, T, 2)))
    h2 = cs_2d.shape[-2]
    p_t = (
        cs_2d[:, :T_t]
        .view(B, nf, T_t, h2, 2)
        .transpose(1, 2)
        .reshape(B * T_t, nf, h2, 2)
    )
    if "loops" in b:
        loops = b["loops"][:, f_idx_t] - b["loops"][:, f_idx_t[:1]]
    else:
        loops = torch.arange(nf, device=dev)[None].expand(B, -1)
    t_cs = energy_data.cos_sin(enc.rope_1d.angles(loops))
    cs_1d = (
        t_cs[:, None, :, :, :]
        .expand(B, T_t, nf, h2, 2)
        .reshape(B * T_t, nf, h2, 2)
    )

    for blk in enc.blocks:
        x_tok = blk(x_tok, valid_flat, cs_2d, allow_t, t_mask_3d, cs_1d, p_t)

    # Un-bottlenecked d=256 representations (before z_ply_proj and z_proj!)
    cls_raw = x_tok[:, 0:1]  # [B * nf, 1, d]
    ply_raw = x_tok[:, 1:3]  # [B * nf, 2, d]
    cls_normed = enc.z_ply_norm(cls_raw).squeeze(1).view(B, nf, -1)  # [B, nf, d]
    ply_normed = enc.z_ply_norm(ply_raw + cls_raw).view(B, nf, 2, -1)  # [B, nf, 2, d]

    z_ply_flat = torch.tanh(enc.z_ply_proj(ply_normed.view(B * nf, 2, -1))).view(
        B, nf, 2, enc.r
    )

    c_flat = x_tok[:, g0 : g0 + R]
    # Pre-projection normalized d=256 unit embeddings, scaled by 1/sqrt(d) so inner products are O(1)
    d_dim = c_flat.shape[-1]
    c_norm = (
        enc.z_norm(c_flat)
        * mask.reshape(B * nf, R, 1).to(x_tok.dtype)
        / (d_dim**0.5)
    ).view(B, nf, R, d_dim)
    z = (
        torch.tanh(enc.z_proj(enc.z_norm(c_flat) * mask.reshape(B * nf, R, 1).to(x_tok.dtype)))
        .view(B, nf, R, enc.r)
    )

    # Control vector u on the burst frames
    f_clamped = f_idx_t.clamp(max=b["queue_changed"].shape[1] - 1)
    q_changed = (
        b["queue_changed"][:, f_clamped]
        & (f_idx_t < b["queue_changed"].shape[1])[None, :, None]
        & mask
    )
    dt_u = energy_data._compute_dtype(enc.cls_tok)
    a_tab_u = enc.abil_emb.weight.to(dt_u) + enc.abil_mlp(enc.abil_feat.to(dt_u))
    z_u = torch.zeros(1, enc.u_dim, device=a_tab_u.device, dtype=dt_u)
    a_u = torch.cat([z_u, enc.abil_u(a_tab_u[1:])], dim=0)
    ord_u = torch.nn.functional.embedding(abil.clamp(0, len(a_u) - 1), a_u, padding_idx=0)
    row_u = ord_u + enc.reg_u(ord_reg)
    u_list = []
    for pl in (energy.P1, energy.P2):
        pl_chg = q_changed & (owner == pl)
        u_pl = (row_u * pl_chg.unsqueeze(-1).to(row_u.dtype)).sum(dim=2) / 10.0
        u_list.append(u_pl)
    u_burst = torch.stack(u_list, dim=2)  # [B, nf, 2, u_dim]

    return (
        q,
        c_norm,
        z,
        cls_normed,
        ply_normed,
        z_ply_flat,
        owner,
        mask,
        p,
        u_burst,
    )


def encode_continuous_with_full_latents(
    enc, model, b, f_idx, idx_out=None, f_chunk=16, t_chunk=4
):
    """Runs Layer 1 Frames._encode_sequence continuously across the full game prefix
    (e.g. t = 14, 28, ..., T) at f_idx, using chunked token embedding, chunked spatial
    SDPA, and chunked causal temporal SDPA so even 56-minute games (~5,400 stride-14
    steps, R ~ 5,000) run in <16 GiB VRAM with zero numerical approximation."""
    import torch.nn.functional as F

    f_idx_t = torch.as_tensor(f_idx, device=b["mask"].device)
    B, F_total, R = b["mask"].shape
    nf = len(f_idx)
    dev = b["mask"].device
    dt = energy_data._compute_dtype(enc.cls_tok)

    if "px" in b:
        x, y, num = b["x"][:, f_idx_t], b["y"][:, f_idx_t], b["num"][:, f_idx_t]
        px, py = b["px"][:, f_idx_t], b["py"][:, f_idx_t]
        both = (b["mask"][:, f_idx_t] & b["pmask"][:, f_idx_t]).to(x.dtype)
        cx = b["center"][:, 0, None, None].to(x.dtype)
        cy = b["center"][:, 1, None, None].to(x.dtype)
        hx = b["half"][:, 0, None, None].clamp(min=1.0).to(x.dtype)
        hy = b["half"][:, 1, None, None].clamp(min=1.0).to(x.dtype)
        x_norm = ((x - cx) / hx).clamp(-2.0, 2.0)
        y_norm = ((y - cy) / hy).clamp(-2.0, 2.0)
        vx_norm = (((x - px) * both) / energy.SPEED_SCALE).clamp(-4.0, 4.0)
        vy_norm = (((y - py) * both) / energy.SPEED_SCALE).clamp(-4.0, 4.0)
        hp_max = num[..., energy_data.I["hp_max"]]
        sh_max = num[..., energy_data.I["shield_max"]]
        hp_frac = (num[..., energy_data.I["hp"]] / hp_max.clamp(min=1.0)) * (hp_max > 0)
        sh_frac = (num[..., energy_data.I["shield"]] / sh_max.clamp(min=1.0)) * (
            sh_max > 0
        )
        en_frac = (num[..., energy_data.I["energy"]] / energy.ENERGY_UNIT).clamp(
            0.0, 1.5
        )
        q = torch.stack(
            [
                torch.ones_like(x_norm),
                x_norm,
                y_norm,
                vx_norm,
                vy_norm,
                hp_frac,
                sh_frac,
                en_frac,
            ],
            dim=-1,
        )
        del x, y, num, px, py, both, hp_max, sh_max, hp_frac, sh_frac, en_frac, x_norm, y_norm, vx_norm, vy_norm
    else:
        q = enc.coords(b, f_idx_t)  # [B, nf, R, K_Q]
    p = enc.ledger(b, f_idx_t)  # [B, nf, 2, 2]

    owner_row = torch.where(
        b["owner"] == energy_data.NEUTRAL, energy.GROUND, b["owner"] + 1
    )  # [B, R]
    owner = owner_row[:, None].expand(B, nf, R)
    on_map = b["mask"][:, f_idx_t] & (b["inside"][:, f_idx_t] == 0)
    mask = on_map & b["alive"][:, f_idx_t]

    center = (
        b["center"][:, None, None, :].float()
        if "center" in b
        else torch.full((B, 1, 1, 2), 100.0, device=dev)
    )
    S = enc.field.grid_size * enc.field.grid_size
    anchors_b = enc.field.anchors(b)[:, None, :, :]  # [B, 1, S, 2]

    eff_tok, eff_mask, eff_xy = enc._embed_effects(b, f_idx_t)
    g0, X = 3, eff_tok.shape[2]
    T_t = g0 + R + S
    T = T_t + X
    d_model = enc.cls_tok.shape[-1]

    g_tok = enc._embed_globals(b, f_idx_t).to(dt)
    x_tok = torch.empty((B, nf, T, d_model), dtype=dt, device=dev)
    pos_all = torch.zeros((B, nf, T, 2), dtype=torch.float32, device=dev)

    # Precompute ability control table once for u_burst
    dt_u = dt
    a_tab_u = enc.abil_emb.weight.to(dt_u) + enc.abil_mlp(enc.abil_feat.to(dt_u))
    z_u = torch.zeros(1, enc.u_dim, device=a_tab_u.device, dtype=dt_u)
    a_u = torch.cat([z_u, enc.abil_u(a_tab_u[1:])], dim=0)
    u_burst = torch.empty((B, nf, 2, enc.u_dim), dtype=dt_u, device=dev)

    # Chunked per-frame token embedding + continuous potential field
    owner_emb_r = enc.owner_emb(owner_row).to(dt)[:, None]  # [B, 1, R, d]
    field_emb_s = enc.field.field_emb[None, None].to(dt)    # [1, 1, S, d]
    for c0 in range(0, nf, f_chunk):
        c1 = min(nf, c0 + f_chunk)
        fc_t = f_idx_t[c0:c1]
        nc = c1 - c0

        q_c = q[:, c0:c1]
        rtype_c = b["rtype"][:, fc_t].clamp(0, energy_data.N_RAW_TYPES - 1)
        num_c = b["num"][:, fc_t]
        kt_c = enc.k_type[rtype_c]
        kt_comp_c = torch.sign(kt_c) * torch.log1p(kt_c.abs()) / 8.0

        abil_c = b["slot_abil"][:, fc_t, :, 0]
        prod_abil_c = b["slot_abil"][:, fc_t, :, -1]

        extras_c = num_c[..., enc.extras_at]
        extras_c = torch.sign(extras_c) * torch.log1p(extras_c.abs()) / 8.0
        ext_in_c = torch.cat([extras_c, kt_comp_c], dim=-1)

        xy_raw_c = torch.stack([b["x"][:, fc_t], b["y"][:, fc_t]], dim=-1).float()
        xy_ang_c = xy_raw_c.unsqueeze(-1) * enc.map_omega
        xy_fourier_c = torch.cat(
            [xy_ang_c.sin().flatten(-2), xy_ang_c.cos().flatten(-2)], dim=-1
        ).to(q.dtype)

        facing_c = num_c[..., energy_data.I["facing"]]
        ord_reg_c = b["slot_reg"][:, fc_t, :, 0, :]
        kin_in_c = torch.cat(
            [
                q_c,
                xy_fourier_c,
                facing_c.sin().unsqueeze(-1),
                facing_c.cos().unsqueeze(-1),
                ord_reg_c,
            ],
            dim=-1,
        )

        pos_ent_c = xy_raw_c - center
        pos_field_c = anchors_b.expand(B, nc, S, 2)
        pos_ef_c = torch.cat([pos_ent_c, pos_field_c], dim=2)

        unit_c, ord_c, prod_c, _ = enc.nodes(rtype_c, abil_c, prod_abil_c)
        field_c = enc.field(b, fc_t, pos_ef_c)
        field_ent_c = field_c[:, :, :R]
        field_tok_c = field_c[:, :, R:] + field_emb_s

        c_ent_c = (
            unit_c
            + ord_c
            + prod_c
            + owner_emb_r
            + enc.kin_c(kin_in_c)
            + enc.extras(ext_in_c)
            + field_ent_c
        ).to(dt)

        x_tok[:, c0:c1, :g0] = g_tok[:, c0:c1]
        x_tok[:, c0:c1, g0 : g0 + R] = c_ent_c
        x_tok[:, c0:c1, g0 + R : T_t] = field_tok_c.to(dt)
        x_tok[:, c0:c1, T_t:] = eff_tok[:, c0:c1].to(dt)

        pos_all[:, c0:c1, g0 : g0 + R] = pos_ent_c
        pos_all[:, c0:c1, g0 + R : T_t] = pos_field_c
        pos_all[:, c0:c1, T_t:] = eff_xy[:, c0:c1]

        # Control vector u on chunk frames
        if "px" in b:
            q_changed_c = b["queue_changed"][:, fc_t] & mask[:, c0:c1]
        else:
            f_prev_c = (fc_t - 1).clamp(min=0, max=b["queue_changed"].shape[1] - 1)
            q_changed_c = b["queue_changed"][:, f_prev_c] & mask[:, c0:c1]
        ord_u_c = torch.nn.functional.embedding(
            abil_c.clamp(0, len(a_u) - 1), a_u, padding_idx=0
        )
        row_u_c = ord_u_c + enc.reg_u(ord_reg_c)
        for pl_i, pl in enumerate((energy.P1, energy.P2)):
            pl_chg = q_changed_c & (owner[:, c0:c1] == pl)
            u_burst[:, c0:c1, pl_i] = (
                row_u_c * pl_chg.unsqueeze(-1).to(row_u_c.dtype)
            ).sum(dim=2) / 10.0
        del q_c, rtype_c, num_c, kt_c, kt_comp_c, abil_c, prod_abil_c, extras_c, ext_in_c
        del xy_raw_c, xy_ang_c, xy_fourier_c, facing_c, ord_reg_c, kin_in_c
        del pos_ent_c, pos_field_c, pos_ef_c, unit_c, ord_c, prod_c, field_c, field_ent_c, field_tok_c, c_ent_c

    del g_tok, eff_tok, eff_xy
    if idx_out is not None:
        idx_t = torch.as_tensor(idx_out, dtype=torch.long, device=dev)
        q = q[:, idx_t].contiguous()
        p = p[:, idx_t].contiguous()
    for k_del in list(b.keys()):
        if k_del != "loops":
            b.pop(k_del, None)
    x_tok = x_tok.reshape(B * nf, T, d_model)
    pos_flat = pos_all.reshape(B * nf, T, 2)
    pos_t = (
        pos_all[:, :, :T_t]
        .transpose(1, 2)
        .reshape(B * T_t, nf, 2)
        .contiguous()
    )
    del pos_all

    valid_g = torch.ones(B, nf, g0, dtype=torch.bool, device=dev)
    valid_field = torch.ones(B, nf, S, dtype=torch.bool, device=dev)
    valid_all = torch.cat([valid_g, mask, valid_field, eff_mask], dim=2)
    valid_flat = valid_all.reshape(B * nf, 1, 1, T)

    t_valid = torch.cat([valid_g, mask, valid_field], dim=2)
    t_mask = t_valid.transpose(1, 2).reshape(B * T_t, nf)
    t_mask_3d = t_mask.unsqueeze(-1)
    idx_i = torch.arange(nf, device=dev)[:, None]
    idx_j = torch.arange(nf, device=dev)[None, :]
    tri = idx_j <= idx_i
    eye = idx_j == idx_i

    if "loops" in b:
        loops = b["loops"][:, f_idx_t] - b["loops"][:, f_idx_t[:1]]
    else:
        loops = torch.arange(nf, device=dev)[None].expand(B, -1)
    t_cs = energy_data.cos_sin(enc.rope_1d.angles(loops))  # [B, nf, h2, 2]

    for blk in enc.blocks:
        sp, ffs = blk.space, blk.ff_space
        # 1. In-place chunked 2D-RoPE Spatial self-attention within each frame
        for c0 in range(0, B * nf, f_chunk):
            c1 = min(B * nf, c0 + f_chunk)
            xc = x_tok[c0:c1]
            cs_c = energy_data.cos_sin(enc.rope_2d.angles(pos_flat[c0:c1]))
            qs, ks, vs = energy_data._rotary_qkv(
                xc,
                blk.ns.weight,
                blk.ns.bias,
                sp.q.weight,
                sp.k.weight,
                sp.v.weight,
                cs_c,
                cs_c,
                cs_c,
                sp.scale,
                sp.H,
                sp.KV,
                sp.dh,
            )
            a_s = F.scaled_dot_product_attention(
                qs.to(dt), ks.to(dt), vs.to(dt), attn_mask=valid_flat[c0:c1]
            )
            x_tok[c0:c1] = energy_data._compiled_out_ff(
                xc,
                a_s,
                cs_c,
                sp.o.weight,
                ffs.norm.weight,
                ffs.norm.bias,
                ffs.w1.weight,
                ffs.w2.weight,
                ffs.w3.weight,
            )

        # 2. In-place chunked 1D-RoPE Causal Temporal self-attention on persistent tokens [:T_t]
        tm, fft = blk.time, blk.ff_time
        for k0 in range(0, T_t, t_chunk):
            k1 = min(T_t, k0 + t_chunk)
            xt_c = x_tok[:, k0:k1, :].transpose(0, 1).contiguous()
            tm_c = t_mask[k0:k1]
            tm3_c = t_mask_3d[k0:k1]
            cs1_c = t_cs[0:1].expand(k1 - k0, -1, -1, -1)
            pt_c = energy_data.cos_sin(enc.rope_2d.angles(pos_t[k0:k1]))
            allow_c = ((tri[None] & tm_c[:, None, :]) | eye[None])[:, None]
            qt, kt_, vt = energy_data._rotary_qkv(
                xt_c,
                blk.nt.weight,
                blk.nt.bias,
                tm.q.weight,
                tm.k.weight,
                tm.v.weight,
                cs1_c,
                cs1_c,
                pt_c,
                tm.scale,
                tm.H,
                tm.KV,
                tm.dh,
            )
            a_t = F.scaled_dot_product_attention(
                qt.to(dt), kt_.to(dt), vt.to(dt), attn_mask=allow_c
            )
            out_t = energy_data._compiled_out_ff_masked(
                xt_c,
                a_t,
                pt_c,
                tm3_c,
                tm.o.weight,
                fft.norm.weight,
                fft.norm.bias,
                fft.w1.weight,
                fft.w2.weight,
                fft.w3.weight,
            )
            x_tok[:, k0:k1, :] = out_t.transpose(0, 1)

    del pos_flat, pos_t, valid_flat, t_mask, t_mask_3d

    # Slice to readout steps idx_out (e.g. every 8th step = 112 loops) before final norms
    x_tok_4d = x_tok.view(B, nf, T, -1)
    if idx_out is not None:
        idx_t = torch.as_tensor(idx_out, dtype=torch.long, device=dev)
        x_tok_4d = x_tok_4d[:, idx_t].contiguous()
        owner = owner[:, idx_t].contiguous()
        mask = mask[:, idx_t].contiguous()
        nf_out = len(idx_out)
    else:
        nf_out = nf
    del x_tok

    cls_raw = x_tok_4d[:, :, 0:1].reshape(B * nf_out, 1, -1)
    ply_raw = x_tok_4d[:, :, 1:3].reshape(B * nf_out, 2, -1)
    cls_normed = enc.z_ply_norm(cls_raw).squeeze(1).view(B, nf_out, -1)
    ply_normed = enc.z_ply_norm(ply_raw + cls_raw).view(B, nf_out, 2, -1)

    z_ply_flat = torch.tanh(enc.z_ply_proj(ply_normed.view(B * nf_out, 2, -1))).view(
        B, nf_out, 2, enc.r
    )

    c_flat = x_tok_4d[:, :, g0 : g0 + R].reshape(B * nf_out, R, -1)
    del x_tok_4d
    d_dim = c_flat.shape[-1]
    c_normed_raw = enc.z_norm(c_flat) * mask.reshape(B * nf_out, R, 1).to(c_flat.dtype)
    del c_flat
    c_norm = (c_normed_raw / (d_dim**0.5)).view(B, nf_out, R, d_dim)
    z = torch.tanh(enc.z_proj(c_normed_raw)).view(B, nf_out, R, enc.r)
    del c_normed_raw

    return (
        q,
        c_norm,
        z,
        cls_normed,
        ply_normed,
        z_ply_flat,
        owner,
        mask,
        p,
        u_burst,
    )


def _fill_batch_slots_direct(
    l,
    orig,
    fo,
    ro,
    is_curr,
    s,
    g,
    vocab,
    ux_arr,
    uy_arr,
    hx,
    hy,
    slot_abil_curr,
    slot_abil_prev,
    slot_reg_0,
):
    """Fills slot_abil (at t or t - 1) and slot_reg_0 (at t) directly from Arrow list
    columns without allocating intermediate list bags or 16-slot float32 arrays."""
    if not len(orig):
        return
    which = np.arange(len(orig))

    # 1. Orders (QUEUE = 8 slots)
    offsets = np.asarray(l["ord_abil"].offsets)
    n = offsets[orig + 1] - offsets[orig]
    if n.sum():
        it = np.concatenate(
            [np.arange(a, b) for a, b in zip(offsets[orig], offsets[orig + 1])]
        )
        rep = np.repeat(which, n)
        seq = np.concatenate([np.arange(c) for c in n])
        q_m = seq < physics_data.QUEUE
        if np.any(q_m):
            it_q, rep_q, seq_q = it[q_m], rep[q_m], seq[q_m]
            raw_ab = (
                l["ord_abil"]
                .values.to_numpy(zero_copy_only=False)[it_q]
                .astype(np.int64)
            )
            ab = vocab.ability_index(raw_ab).astype(np.int64)
            f_q, r_q = fo[rep_q], ro[rep_q]
            if is_curr:
                slot_abil_curr[f_q, r_q, seq_q] = ab
                # Slot 0 regression features at frame t
                s0 = seq_q == 0
                if np.any(s0):
                    it_0, f_0, r_0 = it_q[s0], f_q[s0], r_q[s0]
                    tx = (
                        l["ord_tx"]
                        .values.to_numpy(zero_copy_only=False)[it_0]
                        .astype(np.float32)
                    )
                    ty = (
                        l["ord_ty"]
                        .values.to_numpy(zero_copy_only=False)[it_0]
                        .astype(np.float32)
                    )
                    tt = l["ord_target"].values.to_numpy(zero_copy_only=False)[it_0]
                    tp_ = (
                        l["ord_progress"]
                        .values.to_numpy(zero_copy_only=False)[it_0]
                        .astype(np.float32)
                    )
                    trow = g._row_of_id(s, tt)
                    ux, uy = ux_arr[f_0, r_0], uy_arr[f_0, r_0]
                    dx = np.where(np.isfinite(tx), np.nan_to_num(tx) - ux, 0.0) / hx
                    dy = np.where(np.isfinite(ty), np.nan_to_num(ty) - uy, 0.0) / hy
                    slot_reg_0[f_0, r_0, 0, 0] = np.clip(dx, -2.0, 2.0)
                    slot_reg_0[f_0, r_0, 0, 1] = np.clip(dy, -2.0, 2.0)
                    slot_reg_0[f_0, r_0, 0, 2] = (trow >= 0).astype(np.float32)
                    slot_reg_0[f_0, r_0, 0, 3] = tp_
            else:
                slot_abil_prev[f_q, r_q, seq_q] = ab

    # 2. Production (PROD = 8 slots)
    offsets_p = np.asarray(l["prod_abil"].offsets)
    n_p = offsets_p[orig + 1] - offsets_p[orig]
    if n_p.sum():
        it_p = np.concatenate(
            [np.arange(a, b) for a, b in zip(offsets_p[orig], offsets_p[orig + 1])]
        )
        rep_p = np.repeat(which, n_p)
        seq_p = np.concatenate([np.arange(c) for c in n_p])
        q_p = seq_p < physics_data.PROD
        if np.any(q_p):
            it_pq, rep_pq, seq_pq = it_p[q_p], rep_p[q_p], seq_p[q_p]
            raw_ab_p = (
                l["prod_abil"]
                .values.to_numpy(zero_copy_only=False)[it_pq]
                .astype(np.int64)
            )
            ab_p = vocab.ability_index(raw_ab_p).astype(np.int64)
            f_pq, r_pq = fo[rep_pq], ro[rep_pq]
            if is_curr:
                slot_abil_curr[f_pq, r_pq, physics_data.QUEUE + seq_pq] = ab_p
            else:
                slot_abil_prev[f_pq, r_pq, physics_data.QUEUE + seq_pq] = ab_p


def read_game_continuous(
    gid, base, vocab, avocab, map_id, step_stride=14, out_stride=112
):
    """Streams the full game prefix t = 0 .. T at step_stride (14 loops), allocating
    strictly nf = len(t_14) frames (never g.F or 2*nf dense arrays, and never calling
    physics_data.collate's 16-slot float32 slot_reg). Keeps px, py, pmask and
    slot_abil_prev from t - 1 for exact 1-loop velocities and queue_changed.
    Peak CPU RAM per game is <400 MiB even on 45-minute games."""
    physics_data.MAX_ROWS = 16384
    g = physics_data.Game(base, vocab)
    g._load_actions(avocab)
    g.map_id = map_id

    sub_k = out_stride // step_stride  # 8 steps of 14 loops = 112 loops
    t_14 = list(range(step_stride, g.F - 1, step_stride))
    idx_112 = list(range(sub_k - 1, len(t_14), sub_k))
    t_marks = [t_14[i] for i in idx_112]
    if len(t_marks) < 4:
        return gid, g.meta, t_marks, None, None, None, None

    t_14_arr = np.array(t_14, dtype=np.int64)
    t_prev_arr = t_14_arr - 1
    nf = len(t_14_arr)

    curr_map = np.full(g.F + 64, -1, dtype=np.int64)
    curr_map[t_14_arr] = np.arange(nf, dtype=np.int64)
    prev_map = np.full(g.F + 64, -1, dtype=np.int64)
    prev_map[t_prev_arr] = np.arange(nf, dtype=np.int64)

    rows, row_of = [], {}
    for s in physics_data.SOURCES:
        ix = g.index[s]
        live = (ix["first"] < g.F) & (ix["last"] >= 0)
        for k in np.nonzero(live)[0].tolist():
            row_of[(s, int(ix["id"][k]))] = len(rows)
            rows.append((s, k))
    R = len(rows)

    mask = np.zeros((nf, R), bool)
    pmask = np.zeros((nf, R), bool)
    alive = np.ones((nf, R), bool)
    inside = np.zeros((nf, R), np.int8)
    rtype = np.zeros((nf, R), np.int16)
    x_arr = np.zeros((nf, R), np.float32)
    y_arr = np.zeros((nf, R), np.float32)
    px_arr = np.zeros((nf, R), np.float32)
    py_arr = np.zeros((nf, R), np.float32)
    num_arr = np.zeros((nf, R, len(physics_data.RAW_NUMS)), np.float32)

    n_slots_tot = physics_data.QUEUE + physics_data.PROD
    slot_abil_curr = np.zeros((nf, R, n_slots_tot), np.int32)
    slot_abil_prev = np.zeros((nf, R, n_slots_tot), np.int32)
    slot_reg_0 = np.zeros((nf, R, 1, 4), np.float32)

    g.flat_row = np.full(g.n_flat + 1, -1, np.int32)
    for r_, (s, k) in enumerate(rows):
        g.flat_row[g.flat_off[s] + k] = r_

    row_lut = {}
    for s in (1, 2):
        max_id = max(int(g.index[s]["id"].max(initial=0)), 0) + 1
        lut = np.full(max_id + 1024, -1, np.int64)
        for (src_s, uid), r_ in row_of.items():
            if src_s == s and 0 <= uid < len(lut):
                lut[uid] = r_
        row_lut[s] = lut

    ix0 = g.index[0]
    nid_row = np.full(len(ix0["id"]), -1, np.int64)
    for (src_s, k), r_ in row_of.items():
        if src_s == 0:
            nid_row[k] = r_
    seen_neutral_curr = np.zeros((nf, R), bool)
    seen_neutral_prev = np.zeros((nf, R), bool)
    hx, hy = float(g.half[0]), float(g.half[1])

    for s in (1, 2):
        r, _ = g.views[s]
        lut_s = row_lut[s]
        nid_s = g.nid[s]
        for i in range(r.num_record_batches):
            u, l, fr = arrow_state.read_units(r, i)
            in_valid = (fr >= 0) & (fr < g.F)
            f_c = np.where(in_valid, curr_map[np.clip(fr, 0, g.F)], -1)
            f_p = np.where(in_valid, prev_map[np.clip(fr, 0, g.F)], -1)
            if not (np.any(f_c >= 0) or np.any(f_p >= 0)):
                continue

            own_s = physics_data._own_mask(u["u_owner"], s) & (
                u["u_display"] != physics_data.SNAPSHOT
            )
            own_n = physics_data._own_mask(u["u_owner"], 0)

            # A. Current frames t_14
            for is_curr, f_map_b in ((True, f_c), (False, f_p)):
                in_b = f_map_b >= 0
                if not np.any(in_b):
                    continue

                # 1. Player s units
                idx_s = np.nonzero(in_b & own_s)[0]
                if len(idx_s):
                    uids = u["u_id"][idx_s]
                    valid_u = (uids >= 0) & (uids < len(lut_s))
                    ro = np.where(valid_u, lut_s[np.clip(uids, 0, len(lut_s) - 1)], -1)
                    ok = ro >= 0
                    if np.any(ok):
                        idx_s = idx_s[ok]
                        fo = f_map_b[idx_s]
                        ro = ro[ok]
                        if is_curr:
                            mask[fo, ro] = True
                            x_arr[fo, ro] = u["u_x"][idx_s]
                            y_arr[fo, ro] = u["u_y"][idx_s]
                            rtype[fo, ro] = u["u_type"][idx_s]
                            num_arr[fo, ro] = physics_data._raw(u, idx_s)
                        else:
                            pmask[fo, ro] = True
                            px_arr[fo, ro] = u["u_x"][idx_s]
                            py_arr[fo, ro] = u["u_y"][idx_s]
                        _fill_batch_slots_direct(
                            l,
                            idx_s,
                            fo,
                            ro,
                            is_curr,
                            s,
                            g,
                            vocab,
                            x_arr,
                            y_arr,
                            hx,
                            hy,
                            slot_abil_curr,
                            slot_abil_prev,
                            slot_reg_0,
                        )

                # 2. Neutral units
                idx_n = np.nonzero(in_b & own_n)[0]
                if len(idx_n):
                    n = nid_s[u["u_id"][idx_n]]
                    valid_n = n >= 0
                    if np.any(valid_n):
                        idx_n = idx_n[valid_n]
                        n = n[valid_n]
                        ro_n = nid_row[n]
                        d_n = ix0["death"][n]
                        fr_n = fr[idx_n]
                        ok_n = (ro_n >= 0) & ((d_n < 0) | (fr_n <= d_n))
                        if np.any(ok_n):
                            idx_n = idx_n[ok_n]
                            ro_n = ro_n[ok_n]
                            fo_n = f_map_b[idx_n]
                            vis_n = u["u_display"][idx_n] != physics_data.SNAPSHOT
                            m_arr = mask if is_curr else pmask
                            x_t = x_arr if is_curr else px_arr
                            y_t = y_arr if is_curr else py_arr
                            seen_n = (
                                seen_neutral_curr if is_curr else seen_neutral_prev
                            )

                            snap_m = ~vis_n
                            if np.any(snap_m):
                                idx_sn = idx_n[snap_m]
                                fo_sn, ro_sn = fo_n[snap_m], ro_n[snap_m]
                                unwritten = ~m_arr[fo_sn, ro_sn]
                                if np.any(unwritten):
                                    ii = idx_sn[unwritten]
                                    ff, rr = fo_sn[unwritten], ro_sn[unwritten]
                                    m_arr[ff, rr] = True
                                    x_t[ff, rr] = u["u_x"][ii]
                                    y_t[ff, rr] = u["u_y"][ii]
                                    if is_curr:
                                        rtype[ff, rr] = u["u_type"][ii]
                                        num_arr[ff, rr] = physics_data._raw(u, ii)

                            if np.any(vis_n):
                                idx_vn = idx_n[vis_n]
                                fo_vn, ro_vn = fo_n[vis_n], ro_n[vis_n]
                                unseen = ~seen_n[fo_vn, ro_vn]
                                if np.any(unseen):
                                    ii = idx_vn[unseen]
                                    ff, rr = fo_vn[unseen], ro_vn[unseen]
                                    m_arr[ff, rr] = True
                                    seen_n[ff, rr] = True
                                    x_t[ff, rr] = u["u_x"][ii]
                                    y_t[ff, rr] = u["u_y"][ii]
                                    if is_curr:
                                        rtype[ff, rr] = u["u_type"][ii]
                                        num_arr[ff, rr] = physics_data._raw(u, ii)
                                    _fill_batch_slots_direct(
                                        l,
                                        ii,
                                        ff,
                                        rr,
                                        is_curr,
                                        s,
                                        g,
                                        vocab,
                                        x_arr,
                                        y_arr,
                                        hx,
                                        hy,
                                        slot_abil_curr,
                                        slot_abil_prev,
                                        slot_reg_0,
                                    )

    # Neutral contents as last seen at t_14_arr
    start, log_f, log_c = g.nlog
    at_cont = physics_data.RAW_NUMS.index("contents")
    for k in np.nonzero(nid_row >= 0)[0].tolist():
        a, b_ = int(start[k]), int(start[k + 1])
        if a == b_:
            continue
        j = np.searchsorted(log_f[a:b_], t_14_arr, side="right") - 1
        r_ = nid_row[k]
        m = mask[:, r_]
        num_arr[m, r_, at_cont] = log_c[a:b_][np.maximum(j, 0)][m]

    queue_changed = (slot_abil_curr != slot_abil_prev).any(-1)
    del slot_abil_prev
    # Keep only order slot 0 and prod slot 0 (at index -1) in slot_abil
    slot_abil_2 = np.stack(
        [slot_abil_curr[:, :, 0], slot_abil_curr[:, :, physics_data.QUEUE]], axis=-1
    ).astype(np.int64)
    del slot_abil_curr

    # Map grids at t_14_arr only (padded to multiple of 8 matching physics_data.collate)
    x0, y0, x1, y1 = g.playable
    h_raw, w_raw = y1 - y0, x1 - x0
    gh = int(np.ceil(h_raw / 8.0)) * 8
    gw = int(np.ceil(w_raw / 8.0)) * 8
    grid = np.zeros((1, nf, gh, gw, physics_data.C_LIVE), np.uint8)
    sgrid = np.zeros((1, gh, gw, physics_data.C_STATIC), np.uint8)
    sgrid[0, :h_raw, :w_raw] = g.static_grid

    bits = lambda col, j: np.unpackbits(
        np.frombuffer(col[j].as_py(), np.uint8).reshape(g.H, -1), axis=1
    )[:, : g.W]
    crop = lambda a: a[y0:y1, x0:x1]
    for pi, s in enumerate((1, 2)):
        _, fr = g.views[s]
        g_off = pi * 5
        for idx_k, f in enumerate(t_14):
            b_rec = fr.get_batch(f // physics_data.BATCH)
            j = f % physics_data.BATCH
            vis = np.frombuffer(b_rec.column("vis")[j].as_py(), np.uint8).reshape(
                g.H, g.W
            )
            creep = np.unpackbits(
                np.frombuffer(b_rec.column("creep")[j].as_py(), np.uint8).reshape(
                    g.H, -1
                ),
                axis=1,
            )[:, : g.W]
            grid[0, idx_k, :h_raw, :w_raw, g_off] = crop(vis == 2)
            grid[0, idx_k, :h_raw, :w_raw, g_off + 1] = crop(vis > 0)
            grid[0, idx_k, :h_raw, :w_raw, g_off + 2] = crop(creep)
            grid[0, idx_k, :h_raw, :w_raw, g_off + 3] = crop(
                bits(b_rec.column("pathable"), j)
            )
            grid[0, idx_k, :h_raw, :w_raw, g_off + 4] = crop(
                bits(b_rec.column("buildable"), j)
            )

    # Effects at t_14_arr
    ef = g._effects(0, g.F, False)
    if len(ef):
        nf_idx = curr_map[np.clip(ef[:, 0].astype(np.int64), 0, g.F)]
        ok = nf_idx >= 0
        ef = ef[ok].copy()
        ef[:, 0] = nf_idx[ok].astype(np.float32)
    X_eff = max(
        1,
        (
            int(np.bincount(ef[:, 0].astype(np.int64), minlength=1).max())
            if len(ef)
            else 0
        ),
    )
    emask = np.zeros((1, nf, X_eff), bool)
    etype = np.zeros((1, nf, X_eff), np.int64)
    eown = np.full((1, nf, X_eff), physics_data.NEUTRAL, np.int64)
    exy = np.zeros((1, nf, X_eff, 2), np.float32)
    erad = np.zeros((1, nf, X_eff), np.float32)
    eage = np.zeros((1, nf, X_eff), np.float32)
    seen_ef = np.zeros(nf, np.int64)
    for f_e, ty, o, x_e, y_e, r_e, age_e in ef.tolist():
        f_i, ty_i, o_i = int(f_e), int(ty), int(o)
        slot_i = seen_ef[f_i]
        seen_ef[f_i] += 1
        emask[0, f_i, slot_i] = True
        etype[0, f_i, slot_i] = ty_i
        eown[0, f_i, slot_i] = o_i
        exy[0, f_i, slot_i] = (x_e, y_e)
        erad[0, f_i, slot_i] = r_e
        eage[0, f_i, slot_i] = age_e

    owner_arr = np.array(
        [{1: 0, 2: 1, 0: physics_data.NEUTRAL}[s] for s, _ in rows], np.int64
    ).reshape(1, R)
    players_arr = np.array(
        [g.vocab.race.get(p["race"], 3) for p in g.meta["players"]], np.int64
    ).reshape(1, 2)
    ply_num_arr = g._players(0, g.F, False)[t_14_arr][None]

    b_cpu = dict(
        mask=torch.from_numpy(mask[None]),
        pmask=torch.from_numpy(pmask[None]),
        alive=torch.from_numpy(alive[None]),
        inside=torch.from_numpy(inside[None]),
        x=torch.from_numpy(x_arr[None]),
        y=torch.from_numpy(y_arr[None]),
        px=torch.from_numpy(px_arr[None]),
        py=torch.from_numpy(py_arr[None]),
        num=torch.from_numpy(num_arr[None]),
        rtype=torch.from_numpy(rtype[None].astype(np.int64)),
        owner=torch.from_numpy(owner_arr),
        players=torch.from_numpy(players_arr),
        view=torch.ones(1, 2),
        ply_num=torch.from_numpy(ply_num_arr),
        slot_abil=torch.from_numpy(slot_abil_2[None]),
        slot_reg=torch.from_numpy(slot_reg_0[None]),
        queue_changed=torch.from_numpy(queue_changed[None]),
        center=torch.from_numpy(g.center[None]),
        half=torch.from_numpy(g.half[None]),
        side=torch.tensor([float(g.side)], dtype=torch.float32),
        origin=torch.from_numpy(np.array(g.playable[:2], np.float32)[None]),
        grid=torch.from_numpy(grid),
        static_grid=torch.from_numpy(sgrid),
        loops=torch.from_numpy((t_14_arr + 1)[None]),
        eff_mask=torch.from_numpy(emask),
        eff_type=torch.from_numpy(etype),
        eff_owner=torch.from_numpy(eown),
        eff_xy=torch.from_numpy(exy),
        eff_radius=torch.from_numpy(erad),
        eff_age=torch.from_numpy(eage),
    )

    act_feats = []
    act_actives = []
    for t in t_marks:
        af, ac = _window_action_summary(g, t, stride=out_stride)
        act_feats.append(af)
        act_actives.append(ac)

    return (
        gid,
        g.meta,
        t_marks,
        b_cpu,
        idx_112,
        np.stack(act_feats, 0),
        np.stack(act_actives, 0),
    )


def extract_game_tensors_continuous(enc, model, vocab, game_data, step_stride=14, out_stride=112):
    gid, meta, t_marks, b_cpu, idx_112, act_feats, act_actives = game_data
    sub_k = out_stride // step_stride
    ii_q, jj_q = energy._triu_indices(energy.K_Q)
    ii_c, jj_c = energy._triu_indices(256)

    extra_keys = ("px", "py", "pmask")
    b = energy_data.to_device(b_cpu, "cuda")
    for k in extra_keys:
        if k in b_cpu:
            b[k] = b_cpu[k].to("cuda", non_blocking=True)
    nf = b["mask"].shape[1]
    f_eval = list(range(nf))

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        q, c_norm, z, cls_n, ply_n, zp, owner, mask, p, u_b = (
            encode_continuous_with_full_latents(
                enc, model, b, f_eval, idx_out=idx_112
            )
        )
    del b

    T_steps = len(idx_112)

    q_f = q[0].float()            # [T_steps, R, 8]
    c_f = c_norm[0].float()       # [T_steps, R, 256]
    z_f = z[0].float()            # [T_steps, R, 8]
    cls_f = cls_n[0].bfloat16()   # [T_steps, 256]
    ply_f = ply_n[0].bfloat16()   # [T_steps, 2, 256]
    zp_f = zp[0].float()          # [T_steps, 2, 8]
    own_f = owner[0]              # [T_steps, R]
    msk_f = mask[0]               # [T_steps, R]
    p_f = p[0].float()            # [T_steps, 2, 2]

    # Pool control vector u_b over each 8-step (112-loop) window ending at idx_112
    u_b_win = u_b[0, : T_steps * sub_k].float().view(T_steps, sub_k, 2, -1).mean(dim=1)
    del q, c_norm, z, cls_n, ply_n, zp, owner, mask, p, u_b

    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        x_obs, x_lat = energy.build_gram_state(
            q_f, z_f, own_f, msk_f, p_f, zp_f, r=enc.r
        )
        x_l1 = torch.cat([x_obs, x_lat], dim=-1)
        hout = model.H(x_l1[:, 0], x_l1[:, 1])

        qq_pls, qc_pls, cc_pls = [], [], []
        for pl in (energy.P1, energy.P2):
            m_pl = (msk_f & (own_f == pl)).unsqueeze(-1).float()
            qp = q_f * m_pl
            cp = c_f * m_pl
            g_qq = torch.matmul(qp.transpose(-1, -2), qp) / energy.N0
            g_qc = torch.matmul(qp.transpose(-1, -2), cp) / energy.N0
            g_cc = torch.matmul(cp.transpose(-1, -2), cp) / energy.N0
            qq_pls.append(g_qq[:, ii_q, jj_q])
            qc_pls.append(g_qc.bfloat16())
            cc_pls.append(g_cc[:, ii_c, jj_c].bfloat16())
        g_qq_both = torch.stack(qq_pls, dim=1)  # [T_steps, 2, 36] fp32
        g_qc_both = torch.stack(qc_pls, dim=1)  # [T_steps, 2, 8, 256] bf16
        g_cc_both = torch.stack(cc_pls, dim=1)  # [T_steps, 2, 32896] bf16

    res = meta.get("end", {}).get("results", {})
    winner = 1 if res.get("1") == 1 else (2 if res.get("2") == 1 else 0)

    out_dict = dict(
        gid=gid,
        winner=winner,
        loops=torch.tensor(t_marks, dtype=torch.int32),
        cls_out=cls_f.cpu(),
        ply_out=ply_f.cpu(),
        G_qq=g_qq_both.cpu(),
        G_qc=g_qc_both.cpu(),
        G_cc_triu=g_cc_both.cpu(),
        H1_l1=hout["H1"].cpu(),
        H2_l1=hout["H2"].cpu(),
        A1_l1=hout["A1"].cpu(),
        u_win=u_b_win.cpu(),
        act_feat=torch.from_numpy(act_feats),
        u_active=torch.from_numpy(act_actives),
    )
    del q_f, c_f, z_f, cls_f, ply_f, zp_f, own_f, msk_f, p_f, u_b_win
    del g_qq_both, g_qc_both, g_cc_both, hout, x_obs, x_lat, x_l1
    return out_dict


def read_game_bursts(gid, base, vocab, avocab, map_id, stride=112):
    """Reads all 5s summary bursts (4 frames each) and 5s window actions for one game,
    keeping the game's Arrow batch cache open across windows so each batch is decoded once."""
    g = physics_data.Game(base, vocab)
    g._load_actions(avocab)
    g.map_id = map_id

    # Cache arrow_state.read_units locally for this game's views
    local_cache = {}
    orig_read = g._read

    def cached_read(s, f0, f1, neutral=False):
        r, _ = g.views[s]
        out_u, out_l = [], []
        for i in range(
            max(0, f0 // physics_data.BATCH),
            min(r.num_record_batches, (f1 - 1) // physics_data.BATCH + 1),
        ):
            if (s, i) not in local_cache:
                while len(local_cache) >= 4:
                    local_cache.pop(next(iter(local_cache)))
                local_cache[(s, i)] = arrow_state.read_units(r, i)
            u, l, fr = local_cache[(s, i)]
            keep = (
                physics_data._own_mask(u["u_owner"], 0 if neutral else s)
                & (fr >= f0)
                & (fr < f1)
            )
            if not neutral:
                keep &= u["u_display"] != physics_data.SNAPSHOT
            out_u.append({k: v[keep] for k, v in u.items()} | {"frame": fr[keep]})
            out_l.append((keep, l))
        return out_u, out_l

    g._read = cached_read

    t_marks = list(range(stride, g.F - 1, stride))
    ws = []
    act_feats = []
    act_actives = []
    for t in t_marks:
        ws.append(g.window(t - 3, 4, False))
        # Actions in [t, t + stride) drive the transition from summary frame t -> t + stride
        af, ac = _window_action_summary(g, t, stride=stride)
        act_feats.append(af)
        act_actives.append(ac)

    local_cache.clear()
    return gid, g.meta, t_marks, ws, np.stack(act_feats, 0), np.stack(act_actives, 0)


def extract_game_tensors(enc, model, vocab, game_data, batch_size=32):
    gid, meta, t_marks, ws, act_feats, act_actives = game_data
    T_steps = len(ws)
    ii_q, jj_q = energy._triu_indices(energy.K_Q)
    ii_c, jj_c = energy._triu_indices(256)

    cls_list, ply_list = [], []
    g_qq_list, g_qc_list, g_cc_list = [], [], []
    h1_list, h2_list, a1_list = [], [], []
    u_win_list = []

    for i in range(0, T_steps, batch_size):
        chunk = ws[i : i + batch_size]
        b_cpu = te._trim_batch(physics_data.collate(chunk, vocab))
        b = energy_data.to_device(b_cpu, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            q, c_norm, z, cls_n, ply_n, zp, owner, mask, p, u_b = (
                encode_bursts_with_full_latents(enc, model, b, [1, 2, 3])
            )

        # Take the final frame of the 4-frame burst (index -1, which corresponds to frame t_mark)
        q_f = q[:, -1].float()          # [B_c, R, 8]
        c_f = c_norm[:, -1].float()     # [B_c, R, 256]
        z_f = z[:, -1].float()          # [B_c, R, 8]
        cls_f = cls_n[:, -1].bfloat16() # [B_c, 256]
        ply_f = ply_n[:, -1].bfloat16() # [B_c, 2, 256]
        zp_f = zp[:, -1].float()        # [B_c, 2, 8]
        own_f = owner[:, -1]            # [B_c, R]
        msk_f = mask[:, -1]             # [B_c, R]
        p_f = p[:, -1].float()          # [B_c, 2, 2]
        u_mean = u_b.float().mean(dim=1) # [B_c, 2, u_dim]

        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            x_obs, x_lat = energy.build_gram_state(
                q_f, z_f, own_f, msk_f, p_f, zp_f, r=enc.r
            )
            x_l1 = torch.cat([x_obs, x_lat], dim=-1)
            hout = model.H(x_l1[:, 0], x_l1[:, 1])

            # Compute grounded G_qq [B_c, 2, 36], full-rank Cross-Gram G_qc [B_c, 2, 8, 256],
            # and full-rank 256x256 unit-unit Gram G_cc_triu [B_c, 2, 32896]
            qq_pls, qc_pls, cc_pls = [], [], []
            for pl in (energy.P1, energy.P2):
                m_pl = (msk_f & (own_f == pl)).unsqueeze(-1).float()
                qp = q_f * m_pl
                cp = c_f * m_pl
                g_qq = torch.matmul(qp.transpose(-1, -2), qp) / energy.N0
                g_qc = torch.matmul(qp.transpose(-1, -2), cp) / energy.N0
                g_cc = torch.matmul(cp.transpose(-1, -2), cp) / energy.N0
                qq_pls.append(g_qq[:, ii_q, jj_q])
                qc_pls.append(g_qc.bfloat16())
                cc_pls.append(g_cc[:, ii_c, jj_c].bfloat16())
            g_qq_both = torch.stack(qq_pls, dim=1) # [B_c, 2, 36] fp32
            g_qc_both = torch.stack(qc_pls, dim=1) # [B_c, 2, 8, 256] bf16
            g_cc_both = torch.stack(cc_pls, dim=1) # [B_c, 2, 32896] bf16

        cls_list.append(cls_f.cpu())
        ply_list.append(ply_f.cpu())
        g_qq_list.append(g_qq_both.cpu())
        g_qc_list.append(g_qc_both.cpu())
        g_cc_list.append(g_cc_both.cpu())
        h1_list.append(hout["H1"].cpu())
        h2_list.append(hout["H2"].cpu())
        a1_list.append(hout["A1"].cpu())
        u_win_list.append(u_mean.cpu())

    res = meta.get("end", {}).get("results", {})
    winner = 1 if res.get("1") == 1 else (2 if res.get("2") == 1 else 0)

    return dict(
        gid=gid,
        winner=winner,
        loops=torch.tensor(t_marks, dtype=torch.int32),
        cls_out=torch.cat(cls_list, dim=0),
        ply_out=torch.cat(ply_list, dim=0),
        G_qq=torch.cat(g_qq_list, dim=0),
        G_qc=torch.cat(g_qc_list, dim=0),
        G_cc_triu=torch.cat(g_cc_list, dim=0),
        H1_l1=torch.cat(h1_list, dim=0),
        H2_l1=torch.cat(h2_list, dim=0),
        A1_l1=torch.cat(a1_list, dim=0),
        u_win=torch.cat(u_win_list, dim=0),
        act_feat=torch.from_numpy(act_feats),
        u_active=torch.from_numpy(act_actives),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data/games_v3")
    ap.add_argument("--vocab", default="data/worldmodel/frame_cache_seq/vocab.json")
    ap.add_argument("--split", default="data/games/splits/default.json")
    ap.add_argument("--rules_tables", default="data/worldmodel/rulesets.json")
    ap.add_argument("--base_build", type=int, default=94137)
    ap.add_argument("--patch", default="5.0.14")
    ap.add_argument(
        "--ckpt",
        default="data/worldmodel/runs/energy_3c7cbe7_20261107_213712/model_step30000.pt",
    )
    ap.add_argument("--out_dir", default="data/worldmodel/l2_summaries_5014")
    ap.add_argument("--stride", type=int, default=112)
    ap.add_argument("--step_stride", type=int, default=14)
    ap.add_argument("--continuous", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "unseen_games"), exist_ok=True)

    vocab, avocab = Vocab(json.load(open(args.vocab))), ActionVocab()
    split = json.load(open(args.split))
    train_ids = set(split["train"]) - set(tp.EXCLUDED)
    val_ids = set(split["held_out_games"]) - set(tp.EXCLUDED)
    train_bases = tp.of_patch(physics_data.base_paths(args.games, train_ids), args.patch)
    val_bases = tp.of_patch(physics_data.base_paths(args.games, val_ids), args.patch)
    map_ids = {
        m: i for i, m in enumerate(sorted({tp.map_of(b) for b in train_bases.values()}))
    }

    tables = energy_data.Tables(args.base_build, rulesets=args.rules_tables)
    model = energy.EnergyModel(256, 32, 256).cuda()
    enc = energy_data.Frames(
        vocab, tables, 256, 32, attn_layers=6, dh=64, ff=768
    ).cuda()
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"])
    enc.load_state_dict(ck["enc"])
    model.eval()
    enc.eval()
    del ck

    tasks = []
    for subset, bases_dict in [("unseen_games", val_bases), ("train", train_bases)]:
        for gid, base in sorted(bases_dict.items()):
            out_file = os.path.join(args.out_dir, subset, f"{gid}.pt")
            if not os.path.exists(out_file):
                tasks.append((subset, gid, base, out_file))

    mode_str = f"continuous (step_stride={args.step_stride}, out_stride={args.stride})" if args.continuous else f"bursts (stride={args.stride})"
    print(
        f"Extracting {len(tasks)} games [{mode_str}] (unseen={len(val_bases)}, train={len(train_bases)}) to {args.out_dir}...",
        flush=True,
    )
    t0 = time.time()
    total_bytes = 0

    def _submit_read(pool, gid, base):
        mid = map_ids.get(tp.map_of(base), 0)
        if args.continuous:
            return pool.submit(
                read_game_continuous,
                gid,
                base,
                vocab,
                avocab,
                mid,
                args.step_stride,
                args.stride,
            )
        return pool.submit(
            read_game_bursts,
            gid,
            base,
            vocab,
            avocab,
            mid,
            args.stride,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Submit in rolling window of args.workers so RAM stays tightly bounded
        pending = {}
        it = iter(tasks)
        for _ in range(min(len(tasks), args.workers)):
            sub, gid, base, out_f = next(it)
            fut = _submit_read(pool, gid, base)
            pending[fut] = (sub, gid, out_f)

        done_cnt = 0
        while pending:
            done_fut = next(iter(f for f in pending if f.done()), None)
            if done_fut is None:
                # Wait for the oldest submitted future
                done_fut = next(iter(pending))
            sub, gid, out_f = pending.pop(done_fut)
            game_data = done_fut.result()

            try:
                nxt = next(it)
                f_new = _submit_read(pool, nxt[1], nxt[2])
                pending[f_new] = (nxt[0], nxt[1], nxt[3])
            except StopIteration:
                pass

            valid_game = (
                (game_data[3] is not None and len(game_data[2]) >= 4)
                if args.continuous
                else (len(game_data[3]) >= 4)
            )
            if valid_game:
                if args.continuous:
                    tensors = extract_game_tensors_continuous(
                        enc, model, vocab, game_data, args.step_stride, args.stride
                    )
                else:
                    tensors = extract_game_tensors(enc, model, vocab, game_data)
                tmp_f = out_f + ".tmp"
                torch.save(tensors, tmp_f)
                os.replace(tmp_f, out_f)
                sz = os.path.getsize(out_f)
                total_bytes += sz
                del tensors
            del game_data
            gc.collect()
            torch.cuda.empty_cache()
            done_cnt += 1
            if done_cnt % 10 == 0 or done_cnt == len(tasks):
                el = time.time() - t0
                rate = done_cnt / max(1e-6, el)
                eta = (len(tasks) - done_cnt) / max(1e-6, rate)
                print(
                    f"[{done_cnt:3d}/{len(tasks)}] ({sub}/{gid}) | {el:.1f}s elapsed ({rate:.2f} games/s, ETA {eta:.0f}s) | {total_bytes / 2**20:.1f} MiB saved",
                    flush=True,
                )


if __name__ == "__main__":
    main()
