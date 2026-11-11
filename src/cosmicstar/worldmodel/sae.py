"""Multi-Layer Post-Hoc Top-K Sparse Autoencoder (SAE) Stack for Entity Latents.

Decomposes the 256D residual stream across all 7 representation stages of `Frames`
(Layer 0 additive input embedding + Transformer Blocks 1..6) into sparse, non-negative,
monosemantic dictionary features:

    h_u = ReLU(W_enc (x_u - b_pre) + b_enc)          in R^{M_dict}_{>= 0}
    f_u = TopK(h_u, k=K_active)                      in R^{M_dict}_{>= 0}  (||f_u||_0 <= K)
    x_hat_u = W_dec f_u + b_pre                      (||W_dec[:, j]||_2 = 1)

At the final entity layer (l = 6), each sparse feature j with activation f_{u, j} >= 0
and unit mass m_u > 0 also defines a localized 2D Polar Gauss-Rayleigh spatial field
on the map:

    F_p^{(SAE j)}(x, y) = (1 / sqrt(N_0)) sum_{u in P} psi(x - x_u, y - y_u) m_u f_{u, j}
"""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmicstar.worldmodel import energy as E


class TopKLayerSAE(nn.Module):
    """Top-K Sparse Autoencoder for a single transformer layer's 256D entity states."""

    def __init__(self, d_in: int = 256, d_sae: int = 128, k_active: int = 4, aux_k: int = 16):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k_active = k_active
        self.aux_k = aux_k

        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.w_enc = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.w_dec = nn.Parameter(torch.empty(d_in, d_sae))

        # Initialize encoder/decoder as transposes with unit-norm columns
        nn.init.kaiming_uniform_(self.w_dec)
        with torch.no_grad():
            self.w_dec.data = F.normalize(self.w_dec.data, dim=0)
            self.w_enc.data.copy_(self.w_dec.data.t())

        self.register_buffer("steps_since_active", torch.zeros(d_sae, dtype=torch.long))

    @torch.no_grad()
    def normalize_decoder_(self):
        """Projects decoder columns onto the unit sphere and removes parallel gradient components."""
        norms = self.w_dec.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.w_dec.data.div_(norms)
        if self.w_dec.grad is not None:
            proj = (self.w_dec.grad * self.w_dec.data).sum(dim=0, keepdim=True)
            self.w_dec.grad.sub_(proj * self.w_dec.data)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes normalized entity states x [N, d_in] -> (f_sparse [N, d_sae], top_vals [N, k], top_idx [N, k])."""
        x_cent = x - self.b_pre
        pre_act = F.relu(F.linear(x_cent, self.w_enc, self.b_enc))
        top_vals, top_idx = torch.topk(pre_act, k=min(self.k_active, self.d_sae), dim=-1)
        f_sparse = torch.zeros_like(pre_act)
        f_sparse.scatter_(-1, top_idx, top_vals)
        return f_sparse, top_vals, top_idx

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Computes Top-K reconstruction loss, explained variance R^2, and dead-feature AuxK loss."""
        x_cent = x - self.b_pre
        pre_act = F.relu(F.linear(x_cent, self.w_enc, self.b_enc))
        top_vals, top_idx = torch.topk(pre_act, k=min(self.k_active, self.d_sae), dim=-1)
        f_sparse = torch.zeros_like(pre_act)
        f_sparse.scatter_(-1, top_idx, top_vals)

        x_hat = F.linear(f_sparse, self.w_dec, self.b_pre)
        err = x - x_hat
        mse = err.square().mean()
        total_var = (x - x.mean(dim=0, keepdim=True)).square().mean().clamp(min=1e-8)
        r2 = 1.0 - (mse / total_var)

        # Track feature activity & compute AuxK revival loss on dead features
        if self.training:
            with torch.no_grad():
                fired = (f_sparse > 0).any(dim=0)
                self.steps_since_active += 1
                self.steps_since_active[fired] = 0

        dead_mask = self.steps_since_active > 25
        if self.training and dead_mask.any():
            n_dead = int(dead_mask.sum().item())
            k_aux = min(self.aux_k, n_dead)
            aux_pre = pre_act.masked_fill(~dead_mask[None, :], -1e9)
            aux_vals, aux_idx = torch.topk(aux_pre, k=k_aux, dim=-1)
            aux_vals = F.relu(aux_vals)
            f_aux = torch.zeros_like(pre_act)
            f_aux.scatter_(-1, aux_idx, aux_vals)
            err_hat = F.linear(f_aux, self.w_dec)
            aux_loss = (err.detach() - err_hat).square().mean() / total_var.detach()
        else:
            aux_loss = mse * 0.0

        loss = (mse / total_var.detach()) + 0.05 * aux_loss
        return {
            "loss": loss,
            "mse": mse.detach(),
            "r2": r2.detach(),
            "f_sparse": f_sparse,
            "x_hat": x_hat,
        }


class MultiLayerSAEStack(nn.Module):
    """Stack of 7 Top-K SAEs covering Layer 0 (Input Embedding) and Transformer Blocks 1..6."""

    def __init__(
        self,
        n_layers: int = 7,
        d_in: int = 256,
        d_sae: int = 128,
        k_active: int = 4,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_in = d_in
        self.d_sae = d_sae
        self.k_active = k_active
        self.layers = nn.ModuleList(
            [TopKLayerSAE(d_in=d_in, d_sae=d_sae, k_active=k_active) for _ in range(n_layers)]
        )
        # Populated by train_sae.py after training: list of 7 lists of 128 feature metadata dicts
        self.feature_meta: list[list[dict[str, Any]]] = [
            [
                {
                    "id": j,
                    "label": f"L{l}·F{j:03d}",
                    "top_units": "—",
                    "top_orders": "—",
                    "freq": 0.0,
                    "mean_act": 0.0,
                }
                for j in range(d_sae)
            ]
            for l in range(n_layers)
        ]

    @staticmethod
    def preprocess_states(x: torch.Tensor) -> torch.Tensor:
        """L2-normalizes 256D entity vectors (scaled by sqrt(d_in) = 16.0) so mobile units
        and structures have balanced O(1) norms at every layer."""
        return F.normalize(x.float(), dim=-1) * (x.shape[-1] ** 0.5)

    @torch.no_grad()
    def normalize_decoders_(self):
        for sae in self.layers:
            sae.normalize_decoder_()

    @torch.no_grad()
    def analyze_frame_entities(
        self,
        layer_states: list[torch.Tensor],  # 7 tensors of shape [T, 256]
        live_rows: list[int],
        g0: int,
        q: torch.Tensor,       # [R, 8]
        z: torch.Tensor,       # [R, r]
        owner: torch.Tensor,   # [R]
        mask: torch.Tensor,    # [R]
        gx: torch.Tensor,      # [G, G]
        gy: torch.Tensor,      # [G, G]
        top_n_map_features: int = 16,
    ) -> dict[str, Any]:
        """Evaluates all 7 layer SAEs on the live entities of the current frame and
        paints the top active final-layer (l=6) SAE features onto the 2D map grid."""
        if not live_rows:
            return {
                "layer_r2": [0.0] * self.n_layers,
                "unit_sae_by_row": {},
                "active_map_features": [],
            }

        dev = layer_states[0].device
        live_tok_t = torch.as_tensor([g0 + r for r in live_rows], dtype=torch.long, device=dev)
        N_live = len(live_rows)

        layer_r2 = []
        # per_unit_layers[u_idx][l] = list of {"id": int, "label": str, "act": float}
        per_unit_layers: list[list[list[dict[str, Any]]]] = [
            [[] for _ in range(self.n_layers)] for _ in range(N_live)
        ]

        final_f_sparse = None
        for l_idx in range(self.n_layers):
            sae = self.layers[l_idx]
            x_raw = layer_states[l_idx][live_tok_t]  # [N_live, 256]
            x_in = self.preprocess_states(x_raw)
            f_sparse, top_vals, top_idx = sae.encode(x_in)
            x_hat = F.linear(f_sparse, sae.w_dec, sae.b_pre)
            mse = (x_in - x_hat).square().mean()
            var = (x_in - x_in.mean(dim=0, keepdim=True)).square().mean().clamp(min=1e-8)
            r2 = float((1.0 - mse / var).clamp(0.0, 1.0).item())
            layer_r2.append(round(r2, 4))

            tv_np = top_vals.cpu().numpy()
            ti_np = top_idx.cpu().numpy()
            meta_l = self.feature_meta[l_idx] if l_idx < len(self.feature_meta) else []

            for u_i in range(N_live):
                feats_u = []
                for k_i in range(tv_np.shape[1]):
                    val = float(tv_np[u_i, k_i])
                    if val <= 1e-4:
                        continue
                    fid = int(ti_np[u_i, k_i])
                    lbl = meta_l[fid]["label"] if fid < len(meta_l) else f"L{l_idx}·F{fid:03d}"
                    feats_u.append({"fid": fid, "id": fid, "label": lbl, "act": round(val, 3)})
                per_unit_layers[u_i][l_idx] = feats_u

            if l_idx == self.n_layers - 1:
                final_f_sparse = f_sparse  # [N_live, d_sae]

        unit_sae_by_row = {
            int(live_rows[u_i]): per_unit_layers[u_i] for u_i in range(N_live)
        }

        # Paint the top active final-layer (l=6) SAE features onto the 2D map grid!
        active_map_features = []
        if final_f_sparse is not None:
            R = mask.shape[0]
            f_full = torch.zeros((R, self.d_sae), dtype=torch.float32, device=dev)
            live_rows_t = torch.as_tensor(live_rows, dtype=torch.long, device=dev)
            f_full[live_rows_t] = final_f_sparse.float()

            # Weight each unit's SAE feature activation by its learned mass ||z_u||_2,
            # and prioritize Player 1 & Player 2 owned units when ranking map features
            mass_u = z.float().norm(dim=-1, keepdim=True).clamp(min=0.2)  # [R, 1]
            f_weighted = f_full * mass_u  # [R, d_sae]
            ply_w = torch.where((owner == 1) | (owner == 2), 1.0, 0.08).unsqueeze(-1)
            rank_score = (f_weighted * ply_w).sum(dim=0)  # [d_sae]
            tot_act = f_weighted.sum(dim=0)               # [d_sae]

            n_pick = min(top_n_map_features, int((rank_score > 1e-3).sum().item()))
            if n_pick > 0:
                _, top_fids = torch.topk(rank_score, k=n_pick)
                f_sub = f_weighted[:, top_fids]  # [R, n_pick]

                # Evaluate 2D Polar Gauss-Rayleigh wave-packet fields for these n_pick SAE features
                sf_sae = E.evaluate_spatial_fields_grid(q, f_sub, owner, mask, gx, gy)
                f_p1_sae = sf_sae["F_p1"].cpu().numpy().round(4)    # [n_pick, G, G]
                f_p2_sae = sf_sae["F_p2"].cpu().numpy().round(4)    # [n_pick, G, G]
                f_env_sae = sf_sae["F_env"].cpu().numpy().round(4)  # [n_pick, G, G]

                meta_final = self.feature_meta[-1]
                f_full_np = f_full.cpu().numpy()
                f_weighted_np = f_weighted.cpu().numpy()
                for idx_k, fid_t in enumerate(top_fids):
                    fid = int(fid_t.item())
                    m_info = meta_final[fid] if fid < len(meta_final) else {"id": fid, "label": f"F{fid:03d}"}
                    u_acts = {
                        int(r): round(float(f_full_np[r, fid]), 3)
                        for r in live_rows
                        if f_full_np[r, fid] > 1e-3
                    }
                    u_weights = {
                        int(r): round(float(f_weighted_np[r, fid]), 3)
                        for r in live_rows
                        if f_full_np[r, fid] > 1e-3
                    }
                    active_map_features.append(
                        {
                            "fid": fid,
                            "id": fid,
                            "label": m_info.get("label", f"F{fid:03d}"),
                            "top_units": m_info.get("top_units", "—"),
                            "top_orders": m_info.get("top_orders", "—"),
                            "total_act": round(float(tot_act[fid].item()), 2),
                            "max_act": round(max(u_acts.values(), default=0.0), 3),
                            "n_units": len(u_acts),
                            "n_active_units": len(u_acts),
                            "unit_acts": u_acts,
                            "unit_weights": u_weights,
                            "grid_p1": f_p1_sae[idx_k].tolist(),
                            "grid_p2": f_p2_sae[idx_k].tolist(),
                            "grid_env": f_env_sae[idx_k].tolist(),
                        }
                    )

        return {
            "layer_r2": layer_r2,
            "unit_sae_by_row": unit_sae_by_row,
            "active_map_features": active_map_features,
        }
