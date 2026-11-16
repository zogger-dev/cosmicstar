"""Gram phase-space matrices, Port-Hamiltonian flow decomposition, and token drilldowns."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..data.game_store import QUEUE
from ..l1_tactical import energy as E
from ..l1_tactical.frames import I
from .map_s2ma import _pca_rgb


class L1GramSaeMixin:
    """16x16 / 256x256 Gram spectra, PHD flow decomposition, and per-token attention queries."""

    def _build_gram_payload(
        self,
        q: torch.Tensor,         # [R, 8]
        z: torch.Tensor,         # [R, 8]
        c_norm: torch.Tensor,    # [R, 256]
        owner: torch.Tensor,     # [R]
        mask: torch.Tensor,      # [R]
        p_led: torch.Tensor,     # [2, 2]
        z_ply: torch.Tensor,     # [2, 8]
        x_obs: torch.Tensor,     # [2, 10]
        x_lat: torch.Tensor,     # [2, 140]
        g1_goal: torch.Tensor,   # [150]
        g2_goal: torch.Tensor,   # [150]
    ) -> dict[str, Any]:
        """Constructs 8x8 Continuous Field Overlap matrices (F_self, F_cross) and their
        Hamiltonian sensitivities, per-mode field/flux vectors (F_env, T_field, A_self,
        A_cross, z_ply), 10 grounded physical totals, and 256x256 SVD spectra."""
        r = z.shape[-1]
        labels_8 = [f"z{i}" for i in range(r)]
        labels_16 = list(E.Q_COLS) + labels_8
        yz = torch.cat([q, z], dim=-1)  # [R, 16]

        layout = E.gram_layout(r)
        ii_z, jj_z = E._triu_indices(r)
        n_self = layout["n_self"]    # 36
        n_cross = layout["n_cross"]  # 64
        n_env = layout["n_env"]      # 8
        n_ply = layout["n_ply"]      # 8
        n_q_lat = layout["n_q_lat"]  # 116

        def _unpack_self_8x8(vec36: np.ndarray, is_grad: bool = False) -> np.ndarray:
            mat = np.zeros((r, r), dtype=np.float32)
            for k in range(n_self):
                i, j = int(ii_z[k]), int(jj_z[k])
                val = float(vec36[k]) * (1.0 if (not is_grad or i == j) else 0.5)
                mat[i, j] = val
                mat[j, i] = val
            return np.round(mat, 4 if is_grad else 5)

        def _unpack_player_field(pl_idx: int, g_vec: torch.Tensor) -> dict[str, Any]:
            # Note: -A1 = H2 - H1, so d(-A1)/dx_p = -g_goal_p
            g_np = (-g_vec.detach().cpu().numpy()) * 1000.0
            lat_np = x_lat[pl_idx].detach().cpu().numpy()
            obs_np = x_obs[pl_idx].detach().cpu().numpy()
            g_obs_np = g_np[: E.N_OBS_P]
            g_lat_np = g_np[E.N_OBS_P :]

            f_self_8 = _unpack_self_8x8(lat_np[:n_self], is_grad=False)
            grad_self_8 = _unpack_self_8x8(g_lat_np[:n_self], is_grad=True)

            off_c = n_self
            f_cross_8 = np.round(lat_np[off_c : off_c + n_cross].reshape(r, r), 5)
            grad_cross_8 = np.round(g_lat_np[off_c : off_c + n_cross].reshape(r, r), 4)

            off_env = off_c + n_cross
            f_env_v = np.round(lat_np[off_env : off_env + n_env], 5).tolist()
            grad_env_v = np.round(g_lat_np[off_env : off_env + n_env], 4).tolist()

            off_ply = off_env + n_env
            z_ply_v = np.round(lat_np[off_ply : off_ply + n_ply], 4).tolist()
            grad_ply_v = np.round(g_lat_np[off_ply : off_ply + n_ply], 4).tolist()

            off_t = n_q_lat
            t_field_v = np.round(lat_np[off_t : off_t + r], 5).tolist()
            grad_t_v = np.round(g_lat_np[off_t : off_t + r], 4).tolist()

            off_as = off_t + r
            a_self_v = np.round(lat_np[off_as : off_as + r], 5).tolist()
            grad_as_v = np.round(g_lat_np[off_as : off_as + r], 4).tolist()

            off_ac = off_as + r
            a_cross_v = np.round(lat_np[off_ac : off_ac + r], 5).tolist()
            grad_ac_v = np.round(g_lat_np[off_ac : off_ac + r], 4).tolist()

            # Embed into (8+r)x(8+r) combined view (grounded totals on top-left diagonal, F_self on bottom-right)
            g_16 = np.zeros((8 + r, 8 + r), dtype=np.float32)
            grad_16 = np.zeros((8 + r, 8 + r), dtype=np.float32)
            q_to_obs = {0: 2, 3: 6, 4: 7, 5: 3, 6: 4, 7: 5}
            for q_i, obs_i in q_to_obs.items():
                g_16[q_i, q_i] = float(obs_np[obs_i])
                grad_16[q_i, q_i] = float(g_obs_np[obs_i])
            g_16[8:, 8:] = f_self_8
            grad_16[8:, 8:] = grad_self_8

            return {
                "F_self_8x8": f_self_8.tolist(),
                "grad_F_self_8x8": grad_self_8.tolist(),
                "F_cross_8x8": f_cross_8.tolist(),
                "grad_F_cross_8x8": grad_cross_8.tolist(),
                "G_16x16": np.round(g_16, 5).tolist(),
                "grad_G_16x16": np.round(grad_16, 4).tolist(),
                "obs_totals": np.round(obs_np, 5).tolist(),
                "grad_obs_totals": np.round(g_obs_np, 4).tolist(),
                "mode_vectors": {
                    "F_env": f_env_v,
                    "grad_F_env": grad_env_v,
                    "T_field": t_field_v,
                    "grad_T_field": grad_t_v,
                    "A_self": a_self_v,
                    "grad_A_self": grad_as_v,
                    "A_cross": a_cross_v,
                    "grad_A_cross": grad_ac_v,
                    "z_ply": z_ply_v,
                    "grad_z_ply": grad_ply_v,
                },
            }

        pf_p1 = _unpack_player_field(0, g1_goal)
        pf_p2 = _unpack_player_field(1, g2_goal)

        # Layer 2 learned strategic basis W_c [256, 16]
        w_c = F.normalize(self.enc_l1.z_proj.weight.detach().t(), dim=0) * (256.0 / 16.0) ** 0.25

        owners_out = {}
        for name, pl_id in (("p1", E.P1), ("p2", E.P2), ("neutral", E.GROUND)):
            m = mask & (owner == pl_id)
            idx_live = torch.nonzero(m).squeeze(-1)
            yz_p = yz[idx_live]  # [N_p, 16]

            # Full-rank 256x256 G_cc SVD via thin SVD of unit-normalized centered C_p / sqrt(N0)
            cp = c_norm[idx_live] / (E.N0**0.5)  # [N_p, 256]
            if len(idx_live) > 0:
                try:
                    cp_unit = F.normalize(c_norm[idx_live], dim=-1)
                    cp_cent = (cp_unit - cp_unit.mean(dim=0, keepdim=True)) if len(idx_live) > 1 else cp_unit
                    u_svd, s_svd, vt_svd = torch.linalg.svd(cp_cent, full_matrices=False)
                    eig_vals = s_svd.square() / E.N0
                    top_k = min(16, len(eig_vals))
                    sv_list = eig_vals[:top_k].cpu().numpy().round(5).tolist()
                    p_prob = eig_vals / eig_vals.sum().clamp(min=1e-12)
                    ent = -(p_prob * (p_prob.clamp(min=1e-12).log())).sum().item()
                    eff_rank = round(math.exp(ent), 2)
                    k8 = min(8, len(s_svd))
                    loadings = ((u_svd[:, :k8] * s_svd[:k8]) / 100.0).t().cpu().numpy().round(6).tolist()
                except Exception:
                    sv_list, eff_rank, loadings = [], 0.0, []
            else:
                sv_list, eff_rank, loadings = [], 0.0, []

            if len(idx_live) > 0:
                cp_w = cp @ w_c  # [N_p, 16]
                l2_cc_16 = torch.tanh(cp_w.t() @ cp_w).cpu().numpy().round(4).tolist()
                qp_s = q[idx_live] / (E.N0**0.5)
                l2_qc_8x16 = torch.tanh(qp_s.t() @ cp_w).cpu().numpy().round(4).tolist()
            else:
                l2_cc_16 = np.zeros((16, 16)).tolist()
                l2_qc_8x16 = np.zeros((8, 16)).tolist()

            if name == "p1":
                pf = pf_p1
            elif name == "p2":
                pf = pf_p2
            else:
                g_16_n = ((yz_p.t() @ yz_p) / E.N0).cpu().numpy().round(5) if len(idx_live) > 0 else np.zeros((8 + r, 8 + r))
                pf = {
                    "F_self_8x8": g_16_n[8:, 8:].tolist(),
                    "grad_F_self_8x8": np.zeros((r, r)).tolist(),
                    "F_cross_8x8": np.zeros((r, r)).tolist(),
                    "grad_F_cross_8x8": np.zeros((r, r)).tolist(),
                    "G_16x16": g_16_n.tolist(),
                    "grad_G_16x16": np.zeros((8 + r, 8 + r)).tolist(),
                    "obs_totals": [0.0] * E.N_OBS_P,
                    "grad_obs_totals": [0.0] * E.N_OBS_P,
                    "mode_vectors": {
                        "F_env": [0.0] * r,
                        "grad_F_env": [0.0] * r,
                        "T_field": [0.0] * r,
                        "grad_T_field": [0.0] * r,
                        "A_self": [0.0] * r,
                        "grad_A_self": [0.0] * r,
                        "A_cross": [0.0] * r,
                        "grad_A_cross": [0.0] * r,
                        "z_ply": [0.0] * r,
                        "grad_z_ply": [0.0] * r,
                    },
                }

            owners_out[name] = {
                "n_units": int(len(idx_live)),
                "unit_rows": idx_live.cpu().numpy().astype(int).tolist(),
                **pf,
                "svd_eigenvalues": sv_list,
                "eff_rank": eff_rank,
                "eigenmode_unit_loadings": loadings,
                "l2_cc_16x16": l2_cc_16,
                "l2_qc_8x16": l2_qc_8x16,
                "ledger": (
                    (p_led[pl_id - 1].cpu().numpy() * E.BANK_SCALE).round(1).tolist()
                    if pl_id in (E.P1, E.P2)
                    else [0.0, 0.0]
                ),
                "z_ply": (
                    z_ply[pl_id - 1].cpu().numpy().round(4).tolist()
                    if pl_id in (E.P1, E.P2)
                    else [0.0] * r
                ),
            }

        return {
            "labels_8": labels_8,
            "labels_16": labels_16,
            "obs_cols": list(E.OBS_COLS),
            "q_cols": list(E.Q_COLS),
            "owners": owners_out,
        }

    def _build_phd_payload(
        self,
        x1_t: torch.Tensor,
        x2_t: torch.Tensor,
        x1_next_true: torch.Tensor,
        x2_next_true: torch.Tensor,
        pieces: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Extracts Port-Hamiltonian flow decomposition:
            dx = J(x) grad H (conservative) - R(x) grad H (dissipative) + B(x) u (control)
        vs ground-truth dx = x_{t+1} - x_t across all 10 Grounded Physical Totals and
        key Continuous Spatial Field coordinates."""
        layout = E.gram_layout(self.model_l1.r)

        axes = [
            ("minerals (q)", E.OBS["minerals"]),
            ("vespene (q)", E.OBS["vespene"]),
            ("count (q)", E.OBS["count"]),
            ("hp (q)", E.OBS["hp"]),
            ("shield (q)", E.OBS["shield"]),
            ("energy (q)", E.OBS["energy"]),
            ("vel_x (p_lin)", E.OBS["vx"]),
            ("vel_y (p_lin)", E.OBS["vy"]),
            ("vx^2 (p_quad)", E.OBS["vx2"]),
            ("vy^2 (p_quad)", E.OBS["vy2"]),
            ("F_self[0,0] (q_lat)", layout["f_self_slice"][0]),
            ("F_cross[0,0] (q_lat)", layout["f_cross_slice"][0]),
            ("F_env[0] (q_lat)", layout["f_env_slice"][0]),
            ("T_field[0] (p_quad)", layout["t_field_slice"][0]),
            ("A_self[0] (p_lin)", layout["a_self_slice"][0]),
            ("A_cross[0] (p_lin)", layout["a_cross_slice"][0]),
        ]

        def _extract_player(
            xt: torch.Tensor,
            xn: torch.Tensor,
            j_vec: torch.Tensor,
            r_vec: torch.Tensor,
            bu_vec: torch.Tensor,
            dx_vec: torch.Tensor,
            g_adv: torch.Tensor,
        ):
            true_dx = (xn - xt).detach().cpu().numpy()
            j_np = j_vec.detach().cpu().numpy()
            neg_r_np = (-r_vec).detach().cpu().numpy()
            bu_np = bu_vec.detach().cpu().numpy()
            dx_np = dx_vec.detach().cpu().numpy()
            ga_np = (-g_adv).detach().cpu().numpy()

            rows = []
            for label, idx in axes:
                w = float(layout["w_obs"][idx].item()) if idx < E.N_OBS_P else 1.0
                rows.append(
                    {
                        "label": label,
                        "idx": idx,
                        "val": round(float(xt[idx].item()), 5),
                        "cons": round(float(j_np[idx] * w), 5),
                        "diss": round(float(neg_r_np[idx] * w), 5),
                        "ctrl": round(float(bu_np[idx] * w), 5),
                        "pred_dx": round(float(dx_np[idx] * w), 5),
                        "true_dx": round(float(true_dx[idx] * w), 5),
                        "grad_adv": round(float(ga_np[idx] * 1000.0), 4),
                    }
                )
            return rows

        return {
            "p1": _extract_player(
                x1_t,
                x1_next_true,
                pieces["j1"][0],
                pieces["r1"][0],
                pieces["bu1"][0],
                pieces["dx1"][0],
                pieces["g1_goal"][0],
            ),
            "p2": _extract_player(
                x2_t,
                x2_next_true,
                pieces["j2"][0],
                pieces["r2"][0],
                pieces["bu2"][0],
                pieces["dx2"][0],
                pieces["g2_goal"][0],
            ),
        }

    def _build_tokens_and_units(
        self,
        b: dict[str, torch.Tensor],
        f_curr: int,
        target_idx: int,
        q: torch.Tensor,             # [R, 8]
        z: torch.Tensor,             # [R, 8]
        c_norm: torch.Tensor,        # [R, 256]
        owner: torch.Tensor,         # [R]
        mask: torch.Tensor,          # [R]
        eff_mask: torch.Tensor,      # [X]
        pos_rel: torch.Tensor,       # [T, 2] relative to center
        center_xy: torch.Tensor,     # [2]
        marginal_adv: np.ndarray,    # [R]
        force_adv_x: np.ndarray,     # [R]
        force_adv_y: np.ndarray,     # [R]
        force_kin_x: np.ndarray,     # [R]
        force_kin_y: np.ndarray,     # [R]
        emb_parts_tuple: tuple[torch.Tensor, ...],
        layer_states: list[torch.Tensor],   # 7 tensors [T, 256]
        spatial_attns: list[torch.Tensor],  # 6 tensors [H=4, T, T]
        temporal_attns: list[torch.Tensor], # 6 tensors [T_t, H=4, nf]
        spatial_deltas: list[torch.Tensor], # 6 tensors [T]
        temporal_deltas: list[torch.Tensor],# 6 tensors [T]
        window_loops: list[int],
        marginal_adv_p1: np.ndarray | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """Builds the active unit list, active transformer token catalog, per-layer
        representation trajectories, and compact spatial/temporal attention structures."""
        R = mask.shape[0]
        S = 0
        g0 = 3
        T_t = g0 + R
        cx, cy = float(center_xy[0].item()), float(center_xy[1].item())

        # Compute 3D PCA RGB across live entities for all 7 layers in one GPU batch
        live_rows = torch.nonzero(mask).squeeze(-1).cpu().numpy().astype(int).tolist()
        live_tok_indices = [g0 + r for r in live_rows]
        live_tok_t = torch.as_tensor(live_tok_indices, dtype=torch.long, device=self.device)

        st_stack = torch.stack(layer_states, dim=0)  # [7, T, 256]
        layer_rgb_by_row: dict[int, list[list[int]]] = {r: [] for r in live_rows}
        if len(live_rows) >= 3:
            v_live = st_stack[:, live_tok_t, :]  # [7, N_live, 256]
            v_cent = v_live - v_live.mean(dim=1, keepdim=True)
            try:
                _, _, vt = torch.linalg.svd(v_cent, full_matrices=False)  # vt: [7, K, 256]
                proj = torch.matmul(v_cent, vt[:, :3, :].transpose(-1, -2)).cpu().numpy()  # [7, N_live, 3]
                lo = np.percentile(proj, 5, axis=1, keepdims=True)
                hi = np.percentile(proj, 95, axis=1, keepdims=True)
                rgb_all = (np.clip((proj - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0) * 215.0 + 40.0).astype(int)
                for idx_i, r in enumerate(live_rows):
                    layer_rgb_by_row[r] = rgb_all[:, idx_i, :].tolist()
            except Exception:
                for r in live_rows:
                    layer_rgb_by_row[r] = [[160, 160, 160]] * 7
        else:
            for r in live_rows:
                layer_rgb_by_row[r] = [[160, 160, 160]] * 7

        # Additive input embedding component norms for live rows
        part_names = ("unit", "order", "prod", "owner", "kin", "extras", "field")
        part_norms = [p.detach().float().norm(dim=-1).cpu().numpy() for p in emb_parts_tuple]

        # Layer 2 learned basis W_c [256, 16] projections per unit
        w_c = F.normalize(self.enc_l1.z_proj.weight.detach().t(), dim=0)
        l2_proj_all = (c_norm @ w_c).detach().cpu().numpy()  # [R, 16]

        rtype_np = b["rtype"][0, f_curr].detach().cpu().numpy().astype(int)
        x_np = b["x"][0, f_curr].detach().cpu().numpy()
        y_np = b["y"][0, f_curr].detach().cpu().numpy()
        num_np = b["num"][0, f_curr].detach().cpu().numpy()
        abil_np = b["slot_abil"][0, f_curr, :, 0].detach().cpu().numpy().astype(int)
        prod_np = b["slot_abil"][0, f_curr, :, QUEUE].detach().cpu().numpy().astype(int)
        q_np = q.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        own_np = owner.detach().cpu().numpy().astype(int)

        # Stack layer norms [7, T] and cosine similarity to final layer l=6 [7, T]
        st_norms = st_stack.norm(dim=-1).cpu().numpy()  # [7, T]
        final_st = st_stack[-1:]  # [1, T, 256]
        cos_to_final = F.cosine_similarity(st_stack, final_st, dim=-1).cpu().numpy()  # [7, T]
        sp_del_np = torch.stack(spatial_deltas, dim=0).cpu().numpy()   # [6, T]
        tm_del_np = torch.stack(temporal_deltas, dim=0).cpu().numpy()  # [6, T]

        units_out = []
        for r in live_rows:
            tok_i = g0 + r
            utype = int(rtype_np[r])
            um = self.unit_meta.get(
                utype,
                {
                    "name": f"Unit_{utype}",
                    "category": "other",
                    "structure": False,
                    "worker": False,
                    "minerals": 0,
                    "vespene": 0,
                    "food": 0.0,
                },
            )
            a_id = int(abil_np[r])
            p_id = int(prod_np[r])
            flying = bool(self.tables.k[utype, 5].item() > 0.5) if utype < len(self.tables.k) else False

            units_out.append(
                {
                    "row": r,
                    "tok_idx": tok_i,
                    "owner": int(own_np[r]),  # 1=P1, 2=P2, 0=Neutral
                    "rtype": utype,
                    "name": um["name"],
                    "category": um["category"],
                    "structure": um["structure"],
                    "worker": um["worker"],
                    "flying": flying,
                    "value": int(um["minerals"] + um["vespene"]),
                    "x": round(float(x_np[r]), 2),
                    "y": round(float(y_np[r]), 2),
                    "vx": round(float(q_np[r, 3] * E.SPEED_SCALE), 3),
                    "vy": round(float(q_np[r, 4] * E.SPEED_SCALE), 3),
                    "mass": round(float(np.linalg.norm(z_np[r])), 3),
                    "hp": round(float(num_np[r, I["hp"]]), 1),
                    "hp_max": round(float(num_np[r, I["hp_max"]]), 1),
                    "shield": round(float(num_np[r, I["shield"]]), 1),
                    "shield_max": round(float(num_np[r, I["shield_max"]]), 1),
                    "energy": round(float(num_np[r, I["energy"]]), 1),
                    "contents": round(float(num_np[r, I["contents"]]), 0),
                    "radius": round(max(0.35, float(num_np[r, I["radius"]])), 2),
                    "facing": round(float(num_np[r, I["facing"]]), 2),
                    "weapon_cd": round(float(num_np[r, I["weapon_cooldown"]]), 1),
                    "order_abil": self.vocab_abil_name.get(a_id, f"Abil_{a_id}"),
                    "prod_abil": self.vocab_abil_name.get(p_id, f"Abil_{p_id}"),
                    "q": q_np[r].round(4).tolist(),
                    "z": z_np[r].round(4).tolist(),
                    "l2_w_proj": l2_proj_all[r].round(4).tolist(),
                    "marginal_adv": round(float(marginal_adv[r]), 4),
                    "marginal_adv_own": round(float(marginal_adv[r]), 4),
                    "marginal_adv_p1": round(float(marginal_adv_p1[r]), 4) if marginal_adv_p1 is not None else round(float(marginal_adv[r]), 4),
                    "force_adv": [round(float(force_adv_x[r]), 4), round(float(force_adv_y[r]), 4)],
                    "force_kin": [round(float(force_kin_x[r]), 4), round(float(force_kin_y[r]), 4)],
                    "emb_parts": {
                        k: round(float(part_norms[idx][r]), 3) for idx, k in enumerate(part_names)
                    },
                    "layer_norms": st_norms[:, tok_i].round(3).tolist(),
                    "layer_cos_final": cos_to_final[:, tok_i].round(4).tolist(),
                    "spatial_deltas": sp_del_np[:, tok_i].round(3).tolist(),
                    "temporal_deltas": tm_del_np[:, tok_i].round(3).tolist(),
                    "layer_pca_rgb": layer_rgb_by_row[r],
                }
            )

        # Build compact active token catalog:
        # [CLS (0), P1 (1), P2 (2)] + live units + 64 field anchors + active effects
        tokens_out = []
        active_tok_ids = [0, 1, 2]
        tokens_out.append({"tok_idx": 0, "kind": "global", "label": "[CLS]", "x": round(cx, 1), "y": round(cy, 1), "owner": 0})
        tokens_out.append({"tok_idx": 1, "kind": "global", "label": "[P1]", "x": round(cx, 1), "y": round(cy, 1), "owner": 1})
        tokens_out.append({"tok_idx": 2, "kind": "global", "label": "[P2]", "x": round(cx, 1), "y": round(cy, 1), "owner": 2})

        for u in units_out:
            active_tok_ids.append(u["tok_idx"])
            tokens_out.append(
                {
                    "tok_idx": u["tok_idx"],
                    "kind": "unit",
                    "row": u["row"],
                    "label": f"{u['name']} #{u['row']}",
                    "x": u["x"],
                    "y": u["y"],
                    "owner": u["owner"],
                }
            )

        pos_np = pos_rel.detach().cpu().numpy()
        for s_i in range(S):
            tok_i = g0 + R + s_i
            active_tok_ids.append(tok_i)
            tokens_out.append(
                {
                    "tok_idx": tok_i,
                    "kind": "field",
                    "anchor_idx": s_i,
                    "label": f"FieldAnchor ({s_i // 8},{s_i % 8})",
                    "x": round(cx + float(pos_np[tok_i, 0]), 2),
                    "y": round(cy + float(pos_np[tok_i, 1]), 2),
                    "owner": 0,
                }
            )

        eff_live = torch.nonzero(eff_mask).squeeze(-1).cpu().numpy().astype(int).tolist()
        for e_i in eff_live:
            tok_i = T_t + e_i
            active_tok_ids.append(tok_i)
            tokens_out.append(
                {
                    "tok_idx": tok_i,
                    "kind": "effect",
                    "label": f"Effect #{e_i}",
                    "x": round(cx + float(pos_np[tok_i, 0]), 2),
                    "y": round(cy + float(pos_np[tok_i, 1]), 2),
                    "owner": 0,
                }
            )

        # Cache full active attention sub-matrices in RAM for <1ms interactive queries
        act_t_tensor = torch.as_tensor(active_tok_ids, dtype=torch.long, device=self.device)
        tok_id_to_compact = {tid: idx for idx, tid in enumerate(active_tok_ids)}

        n_act = len(active_tok_ids)
        sp_cache = np.zeros((6, 5, n_act, n_act), dtype=np.float32)

        for l_idx in range(6):
            att_l = spatial_attns[l_idx][:, act_t_tensor][:, :, act_t_tensor]  # [4, n_act, n_act]
            att_mean = att_l.mean(dim=0, keepdim=True)  # [1, n_act, n_act]
            att_5 = torch.cat([att_mean, att_l], dim=0).cpu().numpy()  # [5, n_act, n_act]
            sp_cache[l_idx] = att_5

        # Temporal attention for active persistent tokens (< T_t)
        persist_compact = [tid for tid in active_tok_ids if tid < T_t]
        persist_t = torch.as_tensor(persist_compact, dtype=torch.long, device=self.device)
        # [6, len(persist), 4, nf]
        tm_cache = np.stack(
            [temporal_attns[l_idx][persist_t].cpu().numpy() for l_idx in range(6)],
            axis=0,
        )

        # Cache in memory for instant (<1ms) token/layer/head/Gram-256 drilldowns
        self._last_frame_cache = {
            "active_tok_ids": active_tok_ids,
            "tok_id_to_compact": tok_id_to_compact,
            "sp_cache": sp_cache,  # [6, 5, n_act, n_act] (0=mean, 1..4=H0..H3)
            "persist_tok_ids": persist_compact,
            "persist_to_idx": {tid: i for i, tid in enumerate(persist_compact)},
            "tm_cache": tm_cache,  # [6, n_persist, 4, nf]
            "window_loops": window_loops,
            "layer_states": [st[act_t_tensor].cpu().numpy() for st in layer_states],  # 7 x [n_act, 256]
            "layer_states_gpu": [st.detach() for st in layer_states],                 # 7 x [T, 256]
            "c_norm_np": c_norm.detach().cpu().numpy(),  # [R, 256]
            "q_np": q_np,                                # [R, 8]
            "z_np": z_np,                                # [R, r]
            "q_tensor": q.detach(),                      # [R, 8]
            "z_tensor": z.detach(),                      # [R, r]
            "owner_tensor": owner.detach(),              # [R]
            "mask_tensor": mask.detach(),                # [R]
            "own_np": own_np,                            # [R]
            "live_rows": live_rows,
            "units_out": units_out,
        }

        attn_summary = {
            "window_loops": window_loops,
            "global_temporal": {
                "cls": tm_cache[:, 0, :, :].mean(axis=1).round(4).tolist(),  # [6, nf]
                "p1": tm_cache[:, 1, :, :].mean(axis=1).round(4).tolist(),
                "p2": tm_cache[:, 2, :, :].mean(axis=1).round(4).tolist(),
            },
        }
        return units_out, tokens_out, attn_summary

    def query_token_details(
        self, tok_idx: int, layer: int = 5, head: int = 0
    ) -> dict[str, Any]:
        """Instant (<1ms) drilldown for any clicked token `tok_idx` from the cached frame:
        returns its full spatial attention distribution across all active tokens for the
        requested layer (0..5) and head (0=Mean, 1..4=H0..H3), its full causal temporal
        attention across all 6 layers & 4 heads, and its cosine similarity to all other
        active tokens at layer `layer`."""
        with self.lock:
            cache = self._last_frame_cache
            if not cache or tok_idx not in cache["tok_id_to_compact"]:
                return {"error": f"Token {tok_idx} not in active cache"}

            c_idx = cache["tok_id_to_compact"][tok_idx]
            active_ids = cache["active_tok_ids"]
            l_clamped = max(0, min(5, int(layer)))
            h_clamped = max(0, min(4, int(head)))

            # 1. Outgoing spatial attention weights (what `tok_idx` attends to)
            out_weights = cache["sp_cache"][l_clamped, h_clamped, c_idx]  # [n_act]
            # 2. Incoming spatial attention weights (who attends to `tok_idx`)
            in_weights = cache["sp_cache"][l_clamped, h_clamped, :, c_idx]  # [n_act]

            # Per-head breakdown for top 15 outgoing targets
            top_idx = np.argsort(-out_weights)[:25]
            outgoing = [
                {
                    "tok_idx": int(active_ids[i]),
                    "weight": round(float(out_weights[i]), 4),
                    "heads": cache["sp_cache"][l_clamped, 1:, c_idx, i].round(4).tolist(),
                }
                for i in top_idx
                if out_weights[i] > 0.002
            ]

            top_in_idx = np.argsort(-in_weights)[:25]
            incoming = [
                {
                    "tok_idx": int(active_ids[i]),
                    "weight": round(float(in_weights[i]), 4),
                }
                for i in top_in_idx
                if in_weights[i] > 0.002
            ]

            # 3. Cosine similarity between `tok_idx` and all other active tokens at `layer + 1`
            st_l = cache["layer_states"][l_clamped + 1]  # [n_act, 256]
            v_self = st_l[c_idx]
            norms = np.linalg.norm(st_l, axis=-1) * max(1e-8, float(np.linalg.norm(v_self)))
            cos_sim = (st_l @ v_self) / np.maximum(norms, 1e-8)
            cos_by_tok = {
                int(active_ids[i]): round(float(cos_sim[i]), 4) for i in range(len(active_ids))
            }

            # 4. Temporal attention [6, 4, nf]
            if tok_idx in cache["persist_to_idx"]:
                p_idx = cache["persist_to_idx"][tok_idx]
                temp_attn = cache["tm_cache"][:, p_idx, :, :].round(4).tolist()  # [6, 4, nf]
            else:
                temp_attn = []

            return {
                "tok_idx": tok_idx,
                "layer": l_clamped,
                "head": h_clamped,
                "outgoing": outgoing,
                "incoming": incoming,
                "cosine_similarity": cos_by_tok,
                "temporal_attn": temp_attn,
                "window_loops": cache["window_loops"],
            }

    def query_gram_256(
        self,
        owner: str = "p1",
        mat_type: str = "G_cc_256",
        cell_i: int = 0,
        cell_j: int = 0,
    ) -> dict[str, Any]:
        """Instant (<3ms) computation of full-rank 256x256 G_cc, Centered Covariance Cov(C_p),
        Clustered Covariance, 8x256 Cross-Gram G_qc, or N_live x N_live unit-to-unit
        Gram matrix K_uu = C C^T, plus interpretable feature correlations and signed dipole
        per-unit attribution for cell (cell_i, cell_j)."""
        from collections import Counter

        with self.lock:
            cache = self._last_frame_cache
            if not cache or "c_norm_np" not in cache:
                return {"error": "No cached frame available"}

            c_norm = cache["c_norm_np"]          # [R, 256] (unit L2 norm ~ 1)
            c_ln = c_norm * 16.0                 # [R, 256] (LayerNorm O(1) per channel)
            q_np = cache["q_np"]                 # [R, 8]
            own_np = cache["own_np"]             # [R]
            live_rows = cache["live_rows"]
            units_out = cache["units_out"]
            u_by_row = {u["row"]: u for u in units_out}

            def _rows_for(pl_code: int) -> list[int]:
                return [r for r in live_rows if int(own_np[r]) == pl_code]

            def _channel_correlations(vec: np.ndarray, rows_sub: list[int]) -> list[list[Any]]:
                if len(rows_sub) < 3 or float(np.std(vec)) < 1e-7:
                    return []
                us = [u_by_row[r] for r in rows_sub]
                cand: dict[str, np.ndarray] = {
                    "is_structure": np.array([1.0 if u["structure"] else 0.0 for u in us]),
                    "is_worker": np.array([1.0 if u["worker"] else 0.0 for u in us]),
                    "is_army": np.array([1.0 if u["category"] == "army" else 0.0 for u in us]),
                    "is_flying": np.array([1.0 if u["flying"] else 0.0 for u in us]),
                    "hp": np.array([float(u["hp"]) for u in us]),
                    "shield": np.array([float(u["shield"]) for u in us]),
                    "value": np.array([float(u["value"]) for u in us]),
                    "speed": np.array([float(np.hypot(u["vx"], u["vy"])) for u in us]),
                    "d(-A1)": np.array([float(u["marginal_adv"]) for u in us]),
                }
                for k in range(len(us[0]["z"])):
                    cand[f"z{k}"] = np.array([float(u["z"][k]) for u in us])
                for nm in set(u["name"] for u in us):
                    cand[f"type:{nm}"] = np.array([1.0 if u["name"] == nm else 0.0 for u in us])
                for ab in set(u["order_abil"].split("/")[-1] for u in us):
                    cand[f"ord:{ab}"] = np.array(
                        [1.0 if u["order_abil"].split("/")[-1] == ab else 0.0 for u in us]
                    )
                v_c = vec - vec.mean()
                v_norm = float(np.linalg.norm(v_c))
                if v_norm < 1e-7:
                    return []
                out_c = []
                for k_name, f_arr in cand.items():
                    f_c = f_arr - f_arr.mean()
                    f_norm = float(np.linalg.norm(f_c))
                    if f_norm < 1e-7:
                        continue
                    r_val = float(np.dot(v_c, f_c) / (v_norm * f_norm))
                    out_c.append([k_name, round(r_val, 3)])
                out_c.sort(key=lambda x: -abs(x[1]))
                return out_c[:5]

            if mat_type in ("K_uu", "K_spatial"):
                own_order = {1: 0, 2: 1, 0: 2}
                sorted_u = sorted(
                    units_out,
                    key=lambda u: (own_order.get(u["owner"], 3), u["category"], u["rtype"], u["row"]),
                )
                s_rows = [u["row"] for u in sorted_u]
                s_labels = [f"P{u['owner']} {u['name']} #{u['row']}" for u in sorted_u]
                s_owners = [u["owner"] for u in sorted_u]
                if len(s_rows) > 0:
                    if mat_type == "K_spatial":
                        with torch.no_grad():
                            k_vecs, w_vecs = E._get_wavevectors(torch.device("cpu"), torch.float32)
                            sqrt_w = w_vecs.sqrt()[None, :]
                            pos_s = torch.from_numpy(q_np[s_rows, 1:3]).float()
                            th = pos_s @ k_vecs.t()
                            psi_s = torch.cat([torch.cos(th) * sqrt_w, torch.sin(th) * sqrt_w], dim=-1)
                            k_uu = (psi_s @ psi_s.t()).numpy().round(4)
                    else:
                        c_mat = c_norm[s_rows]  # [N, 256]
                        c_unit = c_mat / np.maximum(np.linalg.norm(c_mat, axis=-1, keepdims=True), 1e-8)
                        k_uu = (c_unit @ c_unit.T).round(4)  # [N, N] true cosine in [-1, +1]
                    ci = max(0, min(len(s_rows) - 1, int(cell_i)))
                    cj = max(0, min(len(s_rows) - 1, int(cell_j)))
                    unit_scores = {
                        int(s_rows[idx]): round(float(k_uu[ci, idx]) / 100.0, 6)
                        for idx in range(len(s_rows))
                    }
                    mat_list = k_uu.tolist()
                    u_i = sorted_u[ci]
                    u_j = sorted_u[cj]
                    dist_ij = round(float(np.hypot(u_i["x"] - u_j["x"], u_i["y"] - u_j["y"])), 1)
                    cell_corrs = {
                        "row_label": s_labels[ci],
                        "col_label": s_labels[cj],
                        "row_corrs": [[f"Mass:{u_i.get('mass', 1.0):.2f}", 1.0], [f"Dist:{dist_ij}u", 1.0]],
                        "col_corrs": [[f"Mass:{u_j.get('mass', 1.0):.2f}", 1.0], [f"Cat:{u_j['category']}", 1.0]],
                        "pos_pole": f"Spatial/Latent Neighbor of {s_labels[ci]}",
                        "neg_pole": f"Distant / Opposite to {s_labels[ci]}",
                    }
                else:
                    mat_list, unit_scores, ci, cj, cell_corrs = [], {}, 0, 0, {}
                return {
                    "mat_type": mat_type,
                    "owner": owner,
                    "shape": [len(s_rows), len(s_rows)],
                    "matrix": mat_list,
                    "unit_labels": s_labels,
                    "unit_rows": s_rows,
                    "unit_owners": s_owners,
                    "cell": [ci, cj],
                    "unit_scores": unit_scores,
                    "cell_corrs": cell_corrs,
                }

            pl_code = 1 if owner == "p1" else (2 if owner == "p2" else 0)
            rp = live_rows if owner == "diff" else _rows_for(pl_code)

            if mat_type == "G_qc_8x256":
                ci = max(0, min(7, int(cell_i)))
                cj = max(0, min(255, int(cell_j)))
                if owner == "diff":
                    r1, r2 = _rows_for(1), _rows_for(2)
                    m1 = (q_np[r1].T @ c_ln[r1]) / E.N0 if r1 else np.zeros((8, 256))
                    m2 = (q_np[r2].T @ c_ln[r2]) / E.N0 if r2 else np.zeros((8, 256))
                    mat = (m1 - m2).round(4)
                else:
                    mat = ((q_np[rp].T @ c_ln[rp]) / E.N0).round(4) if rp else np.zeros((8, 256))
                unit_scores = {
                    int(r): round(float(q_np[r, ci] * c_ln[r, cj] / E.N0), 6)
                    for r in live_rows
                }
                q_name = E.Q_COLS[ci]
                cell_corrs = {
                    "row_label": f"Q[{q_name}]",
                    "col_label": f"c_{cj}",
                    "row_corrs": [[q_name, 1.0]],
                    "col_corrs": _channel_correlations(c_ln[rp, cj], rp) if rp else [],
                    "pos_pole": f"+{q_name} & +c_{cj}",
                    "neg_pole": f"+{q_name} & -c_{cj}",
                }
                return {
                    "mat_type": "G_qc_8x256",
                    "owner": owner,
                    "shape": [8, 256],
                    "matrix": mat.tolist(),
                    "cell": [ci, cj],
                    "unit_scores": unit_scores,
                    "cell_corrs": cell_corrs,
                }

            # 256x256 Modes: G_cc_active (W_z 8D Rowspace), G_cc_256 (Raw), G_cc_cov (Centered), G_cc_clustered
            ci = max(0, min(255, int(cell_i)))
            cj = max(0, min(255, int(cell_j)))
            centered = mat_type in ("G_cc_active", "G_cc_cov", "G_cc_clustered")

            # Unit-L2-normalized embeddings (scaled by 16 so channel std ~ 1) prevent
            # LayerNorm outlier-denominator imbalance where Structures have 5x the norm of Mobile units
            if mat_type == "G_cc_256":
                c_base = c_ln.copy()
            else:
                norms_r = np.maximum(np.linalg.norm(c_ln, axis=-1, keepdims=True), 1e-8)
                c_base = (c_ln / norms_r) * 16.0

            if mat_type == "G_cc_active":
                # Project out the 248D nullspace ker(W_z), keeping the 8D active subspace rowspace(W_z)
                w_z = self.enc_l1.z_proj.weight.detach().float().cpu().numpy()  # [8, 256]
                _, _, vt_z = np.linalg.svd(w_z, full_matrices=False)
                p_row = vt_z.T @ vt_z  # [256, 256]
                c_base = (c_base @ p_row) * (256.0 / 8.0) ** 0.5

            def _player_c(rows_p: list[int]) -> np.ndarray:
                if not rows_p:
                    return np.zeros((0, 256), dtype=np.float32)
                arr = c_base[rows_p]
                if centered and len(rows_p) > 1:
                    arr = arr - arr.mean(axis=0, keepdims=True)
                return arr

            if owner == "diff":
                r1, r2 = _rows_for(1), _rows_for(2)
                c1, c2 = _player_c(r1), _player_c(r2)
                m1 = (c1.T @ c1) / E.N0 if len(r1) else np.zeros((256, 256))
                m2 = (c2.T @ c2) / E.N0 if len(r2) else np.zeros((256, 256))
                mat_full = m1 - m2
            else:
                cp = _player_c(rp)
                mat_full = (cp.T @ cp) / E.N0 if len(rp) else np.zeros((256, 256))

            ch_order = list(range(256))
            if mat_type == "G_cc_clustered":
                diag = np.maximum(np.abs(np.diag(mat_full)), 1e-12)
                norms = np.sqrt(diag)
                corr = mat_full / (norms[:, None] * norms[None, :])
                visited = np.zeros(256, dtype=bool)
                order_idx = []
                curr = int(np.argmax(diag))
                for _ in range(256):
                    order_idx.append(curr)
                    visited[curr] = True
                    if len(order_idx) == 256:
                        break
                    sc = np.where(visited, -1e9, corr[curr] * (norms**0.25))
                    curr = int(np.argmax(sc))
                ch_order = order_idx
                mat_full = mat_full[np.ix_(ch_order, ch_order)]

            real_i = ch_order[ci]
            real_j = ch_order[cj]
            cell_val = float(mat_full[ci, cj])
            cell_sign = 1.0 if cell_val >= 0 else -1.0

            mu_by_own: dict[int, np.ndarray] = {}
            for pl_k in (1, 2, 0):
                rk = _rows_for(pl_k)
                if centered and len(rk) > 1:
                    mu_by_own[pl_k] = c_base[rk].mean(axis=0)
                else:
                    mu_by_own[pl_k] = np.zeros(256, dtype=np.float32)

            unit_scores = {}
            pos_names, neg_names = [], []
            for r in live_rows:
                pl_k = int(own_np[r])
                mu_k = mu_by_own.get(pl_k, np.zeros(256, dtype=np.float32))
                vi = float(c_base[r, real_i] - mu_k[real_i])
                vj = float(c_base[r, real_j] - mu_k[real_j])
                prod = vi * vj
                # Only units whose product (vi * vj) has the SAME sign as `cell_val` are driving this cell!
                if prod * cell_sign <= 0:
                    unit_scores[int(r)] = 0.0
                    continue
                mag = abs(prod) / E.N0
                sgn = 1.0 if vi >= 0 else -1.0
                unit_scores[int(r)] = round(sgn * mag, 6)
                if r in rp and abs(prod) > 0.25:
                    tag = f"{u_by_row[r]['name']}:{u_by_row[r]['order_abil'].split('/')[-1]}"
                    if sgn > 0:
                        pos_names.append(tag)
                    else:
                        neg_names.append(tag)

            pos_top = ", ".join(f"{k}({v})" for k, v in Counter(pos_names).most_common(3)) or "—"
            neg_top = ", ".join(f"{k}({v})" for k, v in Counter(neg_names).most_common(3)) or "—"
            vec_i = np.array([c_base[r, real_i] - mu_by_own[int(own_np[r])][real_i] for r in rp]) if rp else np.zeros(0)
            vec_j = np.array([c_base[r, real_j] - mu_by_own[int(own_np[r])][real_j] for r in rp]) if rp else np.zeros(0)

            cell_corrs = {
                "row_label": f"c_{real_i}" + (f" (pos #{ci})" if mat_type == "G_cc_clustered" else ""),
                "col_label": f"c_{real_j}" + (f" (pos #{cj})" if mat_type == "G_cc_clustered" else ""),
                "row_corrs": _channel_correlations(vec_i, rp),
                "col_corrs": _channel_correlations(vec_j, rp),
                "pos_pole": pos_top,
                "neg_pole": neg_top,
            }

            return {
                "mat_type": mat_type,
                "owner": owner,
                "shape": [256, 256],
                "matrix": mat_full.round(4).tolist(),
                "channel_order": ch_order,
                "cell": [ci, cj],
                "real_cell": [real_i, real_j],
                "unit_scores": unit_scores,
                "cell_corrs": cell_corrs,
            }
