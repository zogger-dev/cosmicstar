"""Per-frame `L1` transformer forward pass, counterfactual sandbox, and 2D map payload."""

from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..core.constants import (
    BANK_SCALE,
    ENERGY_UNIT,
    GROUND,
    MAP_SCALE,
    N0,
    N_RAW_TYPES,
    P1,
    P2,
    SPEED_SCALE,
)
from ..data.game_store import NEUTRAL, QUEUE, collate, to_device, trim_batch
from ..l1_tactical import energy as E
from ..l1_tactical import frames as ED
from ..l1_tactical.frames import (
    EFF_WAVES,
    I,
    MAP_WAVES,
    _compute_dtype,
    _rotary_qkv,
    extract_entity_perception,
)
from ..nn.rope import cos_sin, rotate


class L1FrameMixin:
    """Single-frame `L1` autograd force-field, counterfactual, and map field methods."""

    def inspect_frame(
        self,
        gid: str,
        loop_t: int,
        window_len: int = 8,
        counterfactual: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Runs a deep hooked forward + autograd pass of Layer 1 (`Frames` + `EnergyModel`)
        at game loop `loop_t` (with causal window `window_len` frames ending at `loop_t`,
        plus `loop_t + 1` for ground-truth 1-step transition comparison)."""
        with self.lock:
            t_start = time.time()
            g = self._get_game(gid)
            meta_info = self.catalog_by_id.get(gid, {"gid": gid, "duration_loops": g.F})

            # Clamp loop_t into valid range [window_len + 1, g.F - 3]
            nf = max(4, min(24, int(window_len)))
            loop_t = max(nf + 1, min(g.F - 3, int(loop_t)))
            # We read L = nf + 2 frames so ff = [1 .. nf] encodes transitions [1..nf] -> [2..nf+1],
            # where index nf (i.e. frame index -2 of the nf+1 encoded frames) is loop_t!
            L_read = nf + 2
            t0 = loop_t - nf

            w = g.window(t0, L_read, swap=False)
            b_cpu = trim_batch(collate([w], self.vocab))
            b = to_device(b_cpu, self.device)

            # Target frame index in `b` is `f_curr = nf` (which corresponds to loop_t),
            # and `f_next = nf + 1` (which corresponds to loop_t + 1).
            all_f = list(range(1, nf + 2))  # length nf_enc = nf + 1; index -2 is f_curr=nf, -1 is f_next=nf+1
            target_idx = nf - 1  # index inside all_f corresponding to b[:, nf] (loop_t)

            # Save baseline un-edited evaluation if counterfactual is requested
            base_scalars = None
            if counterfactual:
                with torch.enable_grad():
                    base_res = self._forward_and_analyze(b, all_f, target_idx, full_analysis=False)
                    base_scalars = base_res["scalars"]
                b = self._apply_counterfactual(b, nf, counterfactual)

            with torch.enable_grad():
                analysis = self._forward_and_analyze(b, all_f, target_idx, full_analysis=True)

            # Extract map grid, .s2ma overlays, and painted wave-packet fields at f_curr = nf
            map_payload = self._extract_map_payload(g, b, nf)

            # Extract Match HUD (resources, supply, rates, upgrades), Production queue, and Forces
            hud_payload = self._extract_hud_and_production(g, w, b, nf, int(loop_t), analysis.get("units", []), meta_info)

            elapsed_ms = round((time.time() - t_start) * 1000.0, 1)
            out = {
                "gid": gid,
                "loop_t": int(loop_t),
                "time_s": round(loop_t / 22.4, 2),
                "window_len": nf,
                "elapsed_ms": elapsed_ms,
                "active_l1_ckpt": os.path.basename(self.active_l1_ckpt),
                "active_l2_ckpt": os.path.basename(self.active_observer_ckpt),
                "meta": meta_info,
                "map": map_payload,
                "hud": hud_payload["hud"],
                "production": hud_payload["production"],
                "forces": hud_payload["forces"],
                "units_lost": hud_payload.get("units_lost", {"p1": [], "p2": []}),
                "base_scalars": base_scalars,
                **analysis,
            }
            return out

    def _apply_counterfactual(
        self, b: dict[str, torch.Tensor], f_curr: int, cf: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Clones batch `b` and applies counterfactual edits (delete units, move units,
        scale HP/shield, or adjust player resources) at `f_curr`."""
        b = {k: v.clone() for k, v in b.items()}
        for row_idx in cf.get("deleted_rows", []):
            r = int(row_idx)
            if 0 <= r < b["mask"].shape[2]:
                b["alive"][0, f_curr:, r] = False
                b["mask"][0, f_curr:, r] = False

        for row_str, xy in cf.get("moved_rows", {}).items():
            r = int(row_str)
            if 0 <= r < b["x"].shape[2] and len(xy) == 2:
                dx = float(xy[0]) - float(b["x"][0, f_curr, r].item())
                dy = float(xy[1]) - float(b["y"][0, f_curr, r].item())
                b["x"][0, f_curr, r] = float(xy[0])
                b["y"][0, f_curr, r] = float(xy[1])
                # Also shift previous frame so velocity vx/vy doesn't spike artificially unless desired
                if not cf.get("keep_velocity_spike", False) and f_curr > 0:
                    b["x"][0, f_curr - 1, r] = b["x"][0, f_curr - 1, r] + dx
                    b["y"][0, f_curr - 1, r] = b["y"][0, f_curr - 1, r] + dy

        for row_str, hp_mult in cf.get("hp_scale", {}).items():
            r = int(row_str)
            if 0 <= r < b["num"].shape[2]:
                mult = max(0.0, min(1.0, float(hp_mult)))
                b["num"][0, f_curr, r, I["hp"]] *= mult
                b["num"][0, f_curr, r, I["shield"]] *= mult

        return b

    def _forward_and_analyze(
        self,
        b: dict[str, torch.Tensor],
        all_f: list[int],
        target_idx: int,
        full_analysis: bool = True,
    ) -> dict[str, Any]:
        """Executes `Frames` + `EnergyModel` while capturing all intermediate transformer
        layer states, exact 2D-RoPE spatial and 1D-RoPE temporal attention matrices,
        additive embedding norms, Gram matrices, SVD spectra, and end-to-end autograd
        marginal unit values and spatial force fields."""
        enc = self.enc_l1
        model = self.model_l1
        dev = self.device

        # Make (x, y) leaf tensors with requires_grad=True for end-to-end spatial force field
        b_x = b["x"].float().detach().requires_grad_(True)
        b_y = b["y"].float().detach().requires_grad_(True)
        b_mod = {**b, "x": b_x, "y": b_y}

        f_idx = torch.as_tensor(all_f, device=dev)
        B, F_total, R = b_mod["mask"].shape
        nf = len(all_f)

        q = enc.coords(b_mod, f_idx)  # [B, nf, R, 8]
        p = enc.ledger(b_mod, f_idx)  # [B, nf, 2, 2]

        rtype = b_mod["rtype"][:, f_idx].clamp(0, N_RAW_TYPES - 1)
        owner = b_mod["owner"][:, None].expand(B, nf, R)
        owner = torch.where(owner == NEUTRAL, E.GROUND, owner + 1)
        on_map = b_mod["mask"][:, f_idx] & (b_mod["inside"][:, f_idx] == 0)
        mask = on_map & b_mod["alive"][:, f_idx]

        num = b_mod["num"][:, f_idx]
        kt = enc.k_type[rtype]
        kt_comp = torch.sign(kt) * torch.log1p(kt.abs()) / 8.0

        abil = b_mod["slot_abil"][:, f_idx, :, 0]
        prod_abil = b_mod["slot_abil"][:, f_idx, :, QUEUE]

        extras = num[..., enc.extras_at]
        extras = torch.sign(extras) * torch.log1p(extras.abs()) / 8.0
        ext_in = torch.cat([extras, kt_comp], dim=-1)

        xy_raw = torch.stack([b_mod["x"][:, f_idx], b_mod["y"][:, f_idx]], dim=-1).float()
        xy_ang = xy_raw.unsqueeze(-1) * enc.map_omega
        xy_fourier = torch.cat(
            [xy_ang.sin().flatten(-2), xy_ang.cos().flatten(-2)], dim=-1
        ).to(q.dtype)

        facing = num[..., I["facing"]]
        ord_reg = b_mod["slot_reg"][:, f_idx, :, 0, :]
        kin_in = torch.cat(
            [q, xy_fourier, facing.sin().unsqueeze(-1), facing.cos().unsqueeze(-1), ord_reg],
            dim=-1,
        )

        center = (
            b_mod["center"][:, None, None, :].float()
            if "center" in b_mod
            else torch.full((B, 1, 1, 2), 100.0, device=dev)
        )
        pos_ent = xy_raw - center
        S = 0

        eff_tok, eff_mask, eff_xy = enc._embed_effects(b_mod, f_idx)
        g0, X = 3, eff_tok.shape[2]
        T_t = g0 + R
        T = T_t + X

        unit_c, ord_c, prod_c, _ = enc.nodes(rtype, abil, prod_abil)
        field_ent = enc.field(b_mod, f_idx, pos_ent)
        dt = _compute_dtype(enc.cls_tok)

        own_c = enc.owner_emb(owner).to(dt)
        kin_c = enc.kin_c(kin_in).to(dt)
        ext_c = enc.extras(ext_in).to(dt)

        c_ent = (unit_c + ord_c + prod_c + own_c + kin_c + ext_c + field_ent).to(dt)
        g_tok = enc._embed_globals(b_mod, f_idx).to(dt)
        x_tok = torch.cat([g_tok, c_ent, eff_tok.to(dt)], dim=2).reshape(
            B * nf, T, -1
        )

        valid_g = torch.ones(B, nf, g0, dtype=torch.bool, device=dev)
        valid_all = torch.cat([valid_g, mask, eff_mask], dim=2)
        valid_flat = valid_all.reshape(B * nf, 1, 1, T)

        t_valid = torch.cat([valid_g, mask], dim=2)
        t_mask = t_valid.transpose(1, 2).reshape(B * T_t, nf)
        tri = torch.tril(torch.ones(nf, nf, dtype=torch.bool, device=dev))
        eye = torch.eye(nf, dtype=torch.bool, device=dev)
        allow_t = ((tri[None] & t_mask[:, None, :]) | eye[None])[:, None]
        t_mask_3d = t_mask.unsqueeze(-1)

        pos_g = torch.zeros(B, nf, g0, 2, device=dev, dtype=torch.float32)
        pos_all = torch.cat([pos_g, pos_ent, eff_xy], dim=2)

        cs_2d = cos_sin(enc.rope_2d.angles(pos_all.reshape(B * nf, T, 2)))
        h2 = cs_2d.shape[-2]
        p_t = (
            cs_2d[:, :T_t]
            .view(B, nf, T_t, h2, 2)
            .transpose(1, 2)
            .reshape(B * T_t, nf, h2, 2)
        )
        loops = b_mod["loops"][:, f_idx] - b_mod["loops"][:, f_idx[:1]]
        t_cs = cos_sin(enc.rope_1d.angles(loops))
        cs_1d = (
            t_cs[:, None, :, :, :]
            .expand(B, T_t, nf, h2, 2)
            .reshape(B * T_t, nf, h2, 2)
        )

        # Capture layer trajectories & exact spatial/temporal attention at `target_idx`
        layer_states = [x_tok[target_idx].detach().float()]
        spatial_attns = []   # list of 6 tensors [H=4, T, T] at target_idx
        temporal_attns = []  # list of 6 tensors [T_t, H=4, nf] at target_idx
        spatial_deltas = []  # [6, T] L2 norm of spatial sub-block update
        temporal_deltas = [] # [6, T] L2 norm of temporal sub-block update

        for blk in enc.blocks:
            d_dim = x_tok.shape[2]
            sp, ffs = blk.space, blk.ff_space
            qs, ks, vs = _rotary_qkv(
                x_tok,
                blk.ns.weight,
                blk.ns.bias,
                sp.q.weight,
                sp.k.weight,
                sp.v.weight,
                cs_2d,
                cs_2d,
                cs_2d,
                sp.scale,
                sp.H,
                sp.KV,
                sp.dh,
            )
            if full_analysis:
                with torch.no_grad():
                    qt_s = qs[target_idx].float()
                    kt_s = ks[target_idx].float()
                    logits_s = torch.matmul(qt_s, kt_s.transpose(-1, -2)) / (sp.dh**0.5)
                    logits_s = logits_s.masked_fill(~valid_flat[target_idx], float("-inf"))
                    attn_s = torch.softmax(logits_s, dim=-1)  # [H, T, T]
                    spatial_attns.append(attn_s.detach())

            a_s = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=valid_flat)
            x_after_space = ED._compiled_out_ff(
                x_tok,
                a_s,
                cs_2d,
                sp.o.weight,
                ffs.norm.weight,
                ffs.norm.bias,
                ffs.w1.weight,
                ffs.w2.weight,
                ffs.w3.weight,
            )
            if full_analysis:
                with torch.no_grad():
                    spatial_deltas.append(
                        (x_after_space[target_idx].float() - x_tok[target_idx].float())
                        .norm(dim=-1)
                        .detach()
                    )

            x_p = x_after_space[:, :T_t]
            x_t = x_p.view(B, nf, T_t, d_dim).transpose(1, 2).reshape(B * T_t, nf, d_dim)
            tm, fft = blk.time, blk.ff_time
            qt, kt, vt = _rotary_qkv(
                x_t,
                blk.nt.weight,
                blk.nt.bias,
                tm.q.weight,
                tm.k.weight,
                tm.v.weight,
                cs_1d,
                cs_1d,
                p_t,
                tm.scale,
                tm.H,
                tm.KV,
                tm.dh,
            )
            if full_analysis:
                with torch.no_grad():
                    # Target frame query is index `target_idx` along nf
                    qt_t = qt[:, :, target_idx : target_idx + 1, :].float()
                    kt_t = kt.float()
                    logits_t = torch.matmul(qt_t, kt_t.transpose(-1, -2)) / (tm.dh**0.5)
                    logits_t = logits_t.masked_fill(
                        ~allow_t[:, :, target_idx : target_idx + 1, :], float("-inf")
                    )
                    attn_t = torch.softmax(logits_t, dim=-1).squeeze(2)  # [T_t, H, nf]
                    temporal_attns.append(attn_t.detach())

            a_t = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=allow_t)
            x_t_out = ED._compiled_out_ff_masked(
                x_t,
                a_t,
                p_t,
                t_mask_3d,
                tm.o.weight,
                fft.norm.weight,
                fft.norm.bias,
                fft.w1.weight,
                fft.w2.weight,
                fft.w3.weight,
            )
            x_p_out = x_t_out.view(B, T_t, nf, d_dim).transpose(1, 2).reshape(B * nf, T_t, d_dim)
            x_tok = torch.cat([x_p_out, x_after_space[:, T_t:]], dim=1)

            if full_analysis:
                with torch.no_grad():
                    temporal_deltas.append(
                        (x_tok[target_idx].float() - x_after_space[target_idx].float())
                        .norm(dim=-1)
                        .detach()
                    )
                    layer_states.append(x_tok[target_idx].detach().float())

        cls_out = x_tok[:, 0:1]
        ply_out = x_tok[:, 1:3]
        z_ply_flat = F.softsign(enc.z_ply_proj(enc.z_ply_norm(ply_out + cls_out)))

        z_flat, c_normed = enc.project_entities(x_tok[:, g0 : g0 + R], rtype, mask)

        # Control port u (post-command abilities at f_idx + 1)
        u_all, u_act_all = enc._encode_commands(
            b_mod, f_idx, abil, prod_abil, ord_reg, owner, mask
        )

        # Extract frame `t` (target_idx) and frame `t+1` (target_idx + 1)
        q_t = q[:, target_idx].float()              # [1, R, 8]
        z_t = z_flat[target_idx : target_idx + 1].float()  # [1, R, 8]
        zp_t = z_ply_flat[target_idx : target_idx + 1].float()  # [1, 2, 8]
        own_t = owner[:, target_idx]                # [1, R]
        msk_t = mask[:, target_idx]                 # [1, R]
        p_t_led = p[:, target_idx].float()          # [1, 2, 2]
        u_t = u_all[:, target_idx].float()          # [1, 2, u_dim]
        act_t = u_act_all[:, target_idx]            # [1, 2]

        # Introducing per-unit existence weight `unit_w = 1.0` (with requires_grad=True)
        # so d(-A_1)/d(unit_w_i) gives the exact marginal advantage of unit i!
        unit_w = torch.ones((1, R, 1), dtype=torch.float32, device=dev, requires_grad=True)
        q_weighted = q_t * unit_w
        z_weighted = z_t * unit_w

        x_obs_t, x_lat_t = E.build_gram_state(
            q_weighted, z_weighted, own_t, msk_t, p_t_led, zp_t, r=enc.r
        )
        x_t_full = torch.cat([x_obs_t, x_lat_t], dim=-1)  # [1, 2, 138]
        x1_t, x2_t = x_t_full[:, 0], x_t_full[:, 1]

        # Run Port-Hamiltonian dynamics on detached x1_t, x2_t so its internal
        # autograd.grad(..., retain_graph=False) does not free the encoder graph!
        x1_next_pred, x2_next_pred, pieces = model.dynamics(
            x1_t.detach(), x2_t.detach(), u_t[:, 0], u_t[:, 1], model.H, dt=1.0
        )

        # Next frame ground-truth state x_{t+1}
        with torch.no_grad():
            n_idx = min(nf - 1, target_idx + 1)
            x_obs_n, x_lat_n = E.build_gram_state(
                q[:, n_idx].float(),
                z_flat[n_idx : n_idx + 1].float(),
                owner[:, n_idx],
                mask[:, n_idx],
                p[:, n_idx].float(),
                z_ply_flat[n_idx : n_idx + 1].float(),
                r=enc.r,
            )
            x_next_true = torch.cat([x_obs_n, x_lat_n], dim=-1)

        # Goal cosine alignment at frame t (including symplectic pull from q to p)
        _, cos1 = model._goal_loss(pieces["g1_goal_eff"], pieces["bu1"], act_t[:, 0])
        _, cos2 = model._goal_loss(pieces["g2_goal_eff"], pieces["bu2"], act_t[:, 1])

        H1_val = float(pieces["H1"][0].item() * 1000.0)
        H2_val = float(pieces["H2"][0].item() * 1000.0)
        V1_val = float(pieces["V1"][0].item() * 1000.0)
        V2_val = float(pieces["V2"][0].item() * 1000.0)
        T1_val = float(pieces["T1"][0].item() * 1000.0)
        T2_val = float(pieces["T2"][0].item() * 1000.0)
        A1_val = float(pieces["A1"][0].item() * 1000.0)
        own1_val = float(pieces["own1"][0].item() * 1000.0)
        own2_val = float(pieces["own2"][0].item() * 1000.0)
        tug_val = float(pieces["tug"][0].item() * 1000.0)

        scalars = {
            "H1": round(H1_val, 4),
            "H2": round(H2_val, 4),
            "V1": round(V1_val, 4),
            "V2": round(V2_val, 4),
            "T1": round(T1_val, 4),
            "T2": round(T2_val, 4),
            "A1": round(A1_val, 4),
            "raw_A1": round(A1_val, 4),
            "own1": round(own1_val, 4),
            "own2": round(own2_val, 4),
            "tug": round(tug_val, 4),
            "kin1": round(float(pieces["kin1"][0].item() * 1000.0), 4),
            "kin2": round(float(pieces["kin2"][0].item() * 1000.0), 4),
            "diss1": round(float(pieces["diss1"][0].item() * 1000.0), 4),
            "diss2": round(float(pieces["diss2"][0].item() * 1000.0), 4),
            "goal_cos1": round(float(cos1.item()), 4),
            "goal_cos2": round(float(cos2.item()), 4),
            "u1_active": bool(act_t[0, 0].item()),
            "u2_active": bool(act_t[0, 1].item()),
            "u1_norm": round(float(u_t[0, 0].norm().item()), 4),
            "u2_norm": round(float(u_t[0, 1].norm().item()), 4),
        }

        if not full_analysis:
            return {"scalars": scalars}

        # Evaluate model.H(x1_t, x2_t) connected to the encoder graph (unit_w, b_x, b_y)
        h_graph = model.H(x1_t, x2_t)
        neg_A1 = -h_graph["A1"][0]
        t_tot = h_graph["T1"][0] + h_graph["T2"][0]

        # Fast kinetic gradient wrt q_t
        grad_q_kin = torch.autograd.grad(t_tot, q_t, retain_graph=True, allow_unused=True)[0]
        if grad_q_kin is not None:
            force_kin_x = (grad_q_kin[0, :, 1] / E.MAP_SCALE * 1000.0).detach().cpu().numpy()
            force_kin_y = (grad_q_kin[0, :, 2] / E.MAP_SCALE * 1000.0).detach().cpu().numpy()
        else:
            force_kin_x = np.zeros(R, dtype=np.float32)
            force_kin_y = np.zeros(R, dtype=np.float32)

        # Single end-to-end backward pass through all 6 transformer blocks (retain_graph=False)
        grad_w, grad_bx, grad_by = torch.autograd.grad(
            neg_A1,
            (unit_w, b_x, b_y),
            retain_graph=False,
        )
        # Note: neg_A1 = -(H1 - H2) = (H2 - H1), which is Player 2's advantage.
        # Therefore:
        #   d(A2)/dw_i = +d(neg_A1)/dw_i (impact on Player 2's advantage)
        #   d(A1)/dw_i = -d(neg_A1)/dw_i (impact on Player 1's advantage)
        raw_gw = (0.5 * grad_w[0, :, 0] * 1000.0).detach().cpu().numpy()
        raw_fx = (grad_bx[0, all_f[target_idx]] * 1000.0).detach().cpu().numpy()
        raw_fy = (grad_by[0, all_f[target_idx]] * 1000.0).detach().cpu().numpy()

        own_arr = own_t[0].detach().cpu().numpy()
        # Owner-relative marginal advantage: positive always indicates benefit to the unit's owner!
        # For P1 (owner=1): d(A1)/dw_i = -raw_gw
        # For P2 (owner=2): d(A2)/dw_i = +raw_gw
        # For Neutral: 0.0
        marginal_adv_own = np.where(own_arr == 1, -raw_gw, np.where(own_arr == 2, raw_gw, 0.0))
        marginal_adv_p1 = -raw_gw

        # Spatial force fields: point in direction of increasing unit owner's advantage!
        force_adv_x = np.where(own_arr == 1, -raw_fx, raw_fx)
        force_adv_y = np.where(own_arr == 1, -raw_fy, raw_fy)

        # Build detailed unit, token, Field/Mode, and Port-Hamiltonian flow payloads
        with torch.no_grad():
            gram_payload = self._build_gram_payload(
                q_t[0],
                z_t[0],
                c_normed[target_idx].float() / (256.0**0.5),
                own_t[0],
                msk_t[0],
                p_t_led[0],
                zp_t[0],
                x_obs_t[0],
                x_lat_t[0],
                pieces["g1_goal"][0],
                pieces["g2_goal"][0],
            )

            phd_payload = self._build_phd_payload(
                x1_t[0],
                x2_t[0],
                x_next_true[0, 0],
                x_next_true[0, 1],
                pieces,
            )

            units_payload, tokens_payload, attn_payload = self._build_tokens_and_units(
                b_mod,
                all_f[target_idx],
                target_idx,
                q_t[0],
                z_t[0],
                c_normed[target_idx].float() / (256.0**0.5),
                own_t[0],
                msk_t[0],
                eff_mask[0, target_idx],
                pos_all[0, target_idx],
                center[0, 0, 0],
                marginal_adv_own,
                force_adv_x,
                force_adv_y,
                force_kin_x,
                force_kin_y,
                (
                    unit_c[0, target_idx],
                    ord_c[0, target_idx],
                    prod_c[0, target_idx],
                    own_c[0, target_idx],
                    kin_c[0, target_idx],
                    ext_c[0, target_idx],
                    field_ent[0, target_idx],
                ),
                layer_states,
                spatial_attns,
                temporal_attns,
                spatial_deltas,
                temporal_deltas,
                loops[0].cpu().numpy().tolist(),
                marginal_adv_p1=marginal_adv_p1,
            )

        return {
            "scalars": scalars,
            "gram": gram_payload,
            "phd": phd_payload,
            "units": units_payload,
            "tokens": tokens_payload,
            "attention": attn_payload,
        }

    def _extract_map_payload(
        self, g: Game, b: dict[str, torch.Tensor], f_curr: int
    ) -> dict[str, Any]:
        """Extracts terrain height, pathability, buildability, visibility, creep, .s2ma
        NoFlyZones / HealingShrines, and evaluates `ContinuousPotentialField` on a dense
        64x64 grid over the playable map."""
        x0, y0, x1, y1 = [int(v) for v in g.playable]
        pw, ph = x1 - x0, y1 - y0
        cx, cy = float(g.center[0]), float(g.center[1])
        hx, hy = float(g.half[0]), float(g.half[1])

        # Static terrain height [ph, pw]
        h_grid = g.static_grid[:, :, 0].astype(np.uint8)
        # Live grid at f_curr [ph, pw, 10]: (vis, seen, creep, pathable, buildable) x 2 players
        live = b["grid"][0, f_curr].detach().cpu().numpy()

        # Downsample large grids by 2x if > 128 for crisp & fast JSON transport, or keep exact
        # Actually ph, pw is typically ~140x140 = 19,600 ints; let's downsample to 80x80 for
        # compact JSON (<40 KB) while preserving exact playable bounds!
        G_vis = 80
        ys = np.linspace(0, ph - 1, G_vis).astype(int)
        xs = np.linspace(0, pw - 1, G_vis).astype(int)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")

        height_ds = h_grid[yy, xx].tolist()
        path_ds = live[yy, xx, 3].astype(int).tolist()
        build_ds = live[yy, xx, 4].astype(int).tolist()
        creep_ds = np.maximum(live[yy, xx, 2], live[yy, xx, 7]).astype(int).tolist()
        p1_vis_ds = live[yy, xx, 0].astype(int).tolist()
        p2_vis_ds = live[yy, xx, 5].astype(int).tolist()

        # Evaluate ContinuousPotentialField AND painted 2D wave-packet fields on a 44x44 grid
        G_pot = 44
        r_modes = self.enc_l1.r
        sae_features = []
        sae_layer_r2 = []
        with torch.no_grad():
            px_lin = torch.linspace(-hx, hx, G_pot, device=self.device)
            py_lin = torch.linspace(-hy, hy, G_pot, device=self.device)
            gy, gx = torch.meshgrid(py_lin, px_lin, indexing="ij")
            pos_grid = torch.stack([gx, gy], dim=-1).reshape(1, 1, G_pot * G_pot, 2)
            f_idx_t = torch.tensor([f_curr], device=self.device)
            v_loc = self.enc_l1.field(b, f_idx_t, pos_grid)[0, 0].float()  # [G_pot*G_pot, 256]
            v_norm = v_loc.norm(dim=-1).reshape(G_pot, G_pot).cpu().numpy().round(2).tolist()
            # Project V_loc through z_norm + z_proj (sigmoid) to see the r tactical field modes!
            v_z = torch.sigmoid(self.enc_l1.z_proj(self.enc_l1.z_norm(v_loc.to(self.enc_l1.cls_tok.dtype))).float())
            v_z_modes = v_z.reshape(G_pot, G_pot, r_modes).permute(2, 0, 1).cpu().numpy().round(3).tolist()

            # Evaluate painted entity wave-packet fields F_1(x,y), F_2(x,y), F_env(x,y), Pi_p(x,y)
            cache = self._last_frame_cache
            if cache and "q_tensor" in cache:
                sf = E.evaluate_spatial_fields_grid(
                    cache["q_tensor"],
                    cache["z_tensor"],
                    cache["owner_tensor"],
                    cache["mask_tensor"],
                    gx,
                    gy,
                )
                f_diff_norm = sf["F_diff_norm"].cpu().numpy().round(4).tolist()
                f_contest = sf["F_contest"].cpu().numpy().round(4).tolist()
                t_density = sf["T_density"].cpu().numpy().round(4).tolist()
                f_p1_modes = sf["F_p1"].cpu().numpy().round(4).tolist()
                f_p2_modes = sf["F_p2"].cpu().numpy().round(4).tolist()
                f_env_modes = sf["F_env"].cpu().numpy().round(4).tolist()
                pi_p1_modes = sf["Pi_p1"].cpu().numpy().round(4).tolist()
                pi_p2_modes = sf["Pi_p2"].cpu().numpy().round(4).tolist()
                flux_x = sf["J_x"].cpu().numpy().round(4).tolist()
                flux_y = sf["J_y"].cpu().numpy().round(4).tolist()
                flux_p1_x = sf["J_p1_x"].cpu().numpy().round(4).tolist()
                flux_p1_y = sf["J_p1_y"].cpu().numpy().round(4).tolist()
                flux_p2_x = sf["J_p2_x"].cpu().numpy().round(4).tolist()
                flux_p2_y = sf["J_p2_y"].cpu().numpy().round(4).tolist()

                # Evaluate 7-Layer Post-Hoc Top-K SAE Stack if available
                self._maybe_load_sae()
                if self.sae_stack is not None and "layer_states_gpu" in cache:
                    sae_res = self.sae_stack.analyze_frame_entities(
                        cache["layer_states_gpu"],
                        cache["live_rows"],
                        3,
                        cache["q_tensor"],
                        cache["z_tensor"],
                        cache["owner_tensor"],
                        cache["mask_tensor"],
                        gx,
                        gy,
                        top_n_map_features=16,
                    )
                    sae_features = sae_res["active_map_features"]
                    sae_layer_r2 = sae_res["layer_r2"]
                    by_row = sae_res["unit_sae_by_row"]
                    for u_dict in cache["units_out"]:
                        u_dict["sae_layers"] = by_row.get(int(u_dict["row"]), [])
            else:
                z_grid = np.zeros((G_pot, G_pot)).tolist()
                f_diff_norm = z_grid
                f_contest = z_grid
                t_density = z_grid
                f_p1_modes = [z_grid] * r_modes
                f_p2_modes = [z_grid] * r_modes
                f_env_modes = [z_grid] * r_modes
                pi_p1_modes = [z_grid] * r_modes
                pi_p2_modes = [z_grid] * r_modes
                flux_x = z_grid
                flux_y = z_grid
                flux_p1_x = z_grid
                flux_p1_y = z_grid
                flux_p2_x = z_grid
                flux_p2_y = z_grid

        map_hash = g.meta.get("map_hash", "")
        map_name = g.meta.get("map", "")
        s2ma_info = self.s2ma_cache.get_map_objects(map_hash, map_name)

        return {
            "map_name": map_name,
            "map_size": [int(g.W), int(g.H)],
            "playable": [x0, y0, x1, y1],
            "center": [round(cx, 2), round(cy, 2)],
            "half": [round(hx, 2), round(hy, 2)],
            "grid_res": G_vis,
            "height": height_ds,
            "pathable": path_ds,
            "buildable": build_ds,
            "creep": creep_ds,
            "p1_vis": p1_vis_ds,
            "p2_vis": p2_vis_ds,
            "pot_res": G_pot,
            "pot_norm": v_norm,
            "pot_z_modes": v_z_modes,       # [r, G_pot, G_pot]
            "field_diff_norm": f_diff_norm, # [G_pot, G_pot] ||F_1|| - ||F_2||
            "field_contest": f_contest,     # [G_pot, G_pot] F_1 . F_2
            "field_t_density": t_density,   # [G_pot, G_pot] ||Pi_1||^2 + ||Pi_2||^2
            "field_p1_modes": f_p1_modes,   # [r, G_pot, G_pot]
            "field_p2_modes": f_p2_modes,   # [r, G_pot, G_pot]
            "field_env_modes": f_env_modes, # [r, G_pot, G_pot]
            "field_pi1_modes": pi_p1_modes, # [r, G_pot, G_pot]
            "field_pi2_modes": pi_p2_modes, # [r, G_pot, G_pot]
            "flux_x": flux_x,               # [G_pot, G_pot] 2D momentum vector flux X
            "flux_y": flux_y,               # [G_pot, G_pot] 2D momentum vector flux Y
            "flux_p1_x": flux_p1_x,
            "flux_p1_y": flux_p1_y,
            "flux_p2_x": flux_p2_x,
            "flux_p2_y": flux_p2_y,
            "sae_features": sae_features,   # Top active L6 SAE features painted on 2D map
            "sae_layer_r2": sae_layer_r2,   # 7-layer SAE explained variance R^2
            "s2ma": s2ma_info,
        }
