"""Top-level fog-of-war `ObserverWorldModelL2` training & multi-branch rollout engine."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.constants import MAP_SCALE
from ..l1_tactical import squad_pooler as SP
from ..nn.layers import small_init
from .dynamics import (
    SquadPortHamiltonian,
    _compiled_field_loss,
    _multi_scale_spatial_field_loss,
)
from .encoders import (
    OmniscientObserverEncoder,
    OpponentBeliefHead,
    PerspectiveMacroEncoder,
)
from .perception import SightFootprintHead
from .priors import ConditionalFlowPrior, WorldOutcomeFlowPrior


class ObserverWorldModelL2(nn.Module):
    """Multi-Scale Stochastic Observer v2 (`120s` History -> `+120s` Rollout)."""

    def __init__(
        self,
        d_squad: int = SP.D_SQUAD_DEFAULT,
        n_squads: int = SP.N_SQUADS_DEFAULT,
        d_model: int = 768,
        d_obs: int = 512,
        d_cap: int = 16,
        layers: int = 6,
        heads: int = 12,
        rollout_steps: int = 4,
    ):
        super().__init__()
        self.d_squad = d_squad
        self.n_squads = n_squads
        self.d_model = d_model
        self.d_obs = d_obs
        self.d_cap = d_cap
        self.rollout_steps = rollout_steps

        self.sight = SightFootprintHead(d_squad=d_squad)
        self.persp_enc = PerspectiveMacroEncoder(
            d_squad=d_squad, d_model=d_model, d_cap=d_cap, layers=layers, heads=heads
        )
        self.belief_head = OpponentBeliefHead(d_model=d_model, d_squad=d_squad)
        self.obs_enc = OmniscientObserverEncoder(
            d_squad=d_squad, d_persp=d_model, d_obs=d_obs
        )
        self.flow_prior = ConditionalFlowPrior(
            d_u=SP.innovation_dim(), d_model=d_model, d_cap=d_cap
        )
        self.xi_prior = WorldOutcomeFlowPrior(
            d_xi_world=SP.D_XI_WORLD_DEFAULT,
            d_squad=d_squad,
            d_u=SP.innovation_dim(),
            d_obs=d_obs,
        )
        self.ph = SquadPortHamiltonian(d_squad=d_squad, n_squads=n_squads)

    def compile_parts(self):
        global _compiled_field_loss
        torch._dynamo.config.recompile_limit = 256
        torch._functorch.config.donated_buffer = False
        self.sight.compile(dynamic=True)
        self.persp_enc.compile(dynamic=True)
        self.obs_enc.compile(dynamic=True)
        self.flow_prior.compile(dynamic=True)
        self.xi_prior.compile(dynamic=True)
        if _compiled_field_loss is None:
            _compiled_field_loss = torch.compile(_multi_scale_spatial_field_loss, dynamic=True)

    def _state_transition_losses(
        self,
        pred_s: dict[str, torch.Tensor],
        true_s: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        true_s = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in true_s.items()}
        gp = (pred_s["gate"].float() if "gate" in pred_s else pred_s["active"].float()).clamp(0.0, 1.0)
        gt = (true_s["gate"].float() if "gate" in true_s else true_s["active"].float()).clamp(0.0, 1.0)

        tot_p = pred_s["q_squad"][..., 6:10].sum(dim=-2)
        tot_t = true_s["q_squad"][..., 6:10].sum(dim=-2)
        mac_p = pred_s["q_macro"][..., 0:4]
        mac_t = true_s["q_macro"][..., 0:4]

        mse_tot = (tot_p - tot_t).square().mean(dim=(0, 1))
        mse_mac = (mac_p - mac_t).square().mean(dim=(0, 1))

        w_slot = torch.maximum(gp, gt)
        denom_act = w_slot.sum().clamp(min=1.0)
        sq_diff = (pred_s["q_squad"] - true_s["q_squad"]).square().sum(dim=-1) * w_slot
        loss_slot_q = sq_diff.sum() / denom_act
        loss_gate = F.smooth_l1_loss(gp, gt)

        fl_fn = _compiled_field_loss if _compiled_field_loss is not None else _multi_scale_spatial_field_loss
        loss_field = fl_fn(pred_s, true_s)

        loss_phys = (
            4.0 * mse_tot.sum()
            + 2.0 * mse_mac.sum()
            + 0.5 * loss_slot_q
            + 0.5 * loss_gate
            + 2.0 * loss_field
        )

        lat_sq = F.smooth_l1_loss(
            pred_s["z_squad"] * w_slot.unsqueeze(-1),
            true_s["z_squad"] * w_slot.unsqueeze(-1),
        )
        lat_mac = F.smooth_l1_loss(pred_s["z_macro"], true_s["z_macro"])
        loss_lat = lat_sq + 0.5 * lat_mac

        metrics = dict(
            mse_count=mse_tot[0].detach(),
            mse_cost=mse_tot[1].detach(),
            mse_hp=mse_tot[2].detach(),
            mse_shield=mse_tot[3].detach(),
            mse_minerals=mse_mac[0].detach(),
            mse_vespene=mse_mac[1].detach(),
            loss_field=loss_field.detach(),
        )
        return loss_phys, loss_lat, metrics

    def compute_intrinsic_leverage(
        self,
        state: dict[str, torch.Tensor],
        static_map: dict[str, torch.Tensor] | None = None,
        grid_h: int = 64,
        grid_w: int = 64,
    ) -> dict[str, torch.Tensor]:
        """Computes purely learned intrinsic leverage `||grad_{z_m, xy_m} A||_2 * gate_m`
        in ONE differentiable Hamiltonian evaluation (zero leave-one-out deletion hacks!)."""
        dev = state["q_squad"].device
        q_sq = state["q_squad"][:1].detach().float().requires_grad_(True)
        z_sq = state["z_squad"][:1].detach().float().requires_grad_(True)
        gate = (state["gate"][:1] if "gate" in state else state["active"][:1]).detach().float().clamp(0.0, 1.0)
        x_mac = torch.cat([state["q_macro"][:1], state["z_macro"][:1]], dim=-1).detach().float()

        with torch.enable_grad():
            x_sq = torch.cat([q_sq, z_sq], dim=-1)
            h_out = self.ph.evaluate_hamiltonian(x_sq, x_mac, gate, static_map=static_map)
            a0 = h_out["A1"][0]
            gz, gq = torch.autograd.grad(a0, (z_sq, q_sq), create_graph=False)

        # Combine latent composition/role sensitivity + spatial positioning sensitivity (* 1000 res scale)
        lev_z = gz[0].float().norm(p=2, dim=-1)
        lev_xy = gq[0, ..., 0:2].float().norm(p=2, dim=-1) * 0.25
        squad_lev = (lev_z + lev_xy) * gate[0] * 1000.0  # [2, M]

        if static_map is not None and "half" in static_map:
            hx = float(static_map["half"][0, 0].item()) / MAP_SCALE
            hy = float(static_map["half"][0, 1].item()) / MAP_SCALE
        else:
            hx, hy = 0.85, 0.85
        xs = torch.linspace(-hx, hx, grid_w, device=dev)
        ys = torch.linspace(-hy, hy, grid_h, device=dev)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid_pts = torch.stack([gx, gy], dim=-1).view(-1, 2)  # [G, 2]

        lev_grids = []
        g_denom = gate[0].clamp(min=0.05)
        pos_0 = q_sq.detach()[0, :, :, 0:2] / g_denom.unsqueeze(-1)  # [2, M, 2] true normalized map coords
        spread_0 = ((q_sq.detach()[0, :, :, 4:6] / g_denom.unsqueeze(-1)).square().mean(dim=-1) + 0.05**2).sqrt()
        for p_i in (0, 1):
            w_p = squad_lev[p_i]  # [M]
            d2 = (grid_pts.unsqueeze(1) - pos_0[p_i].unsqueeze(0)).square().sum(dim=-1)  # [G, M]
            sig2 = (2.0 * spread_0[p_i].clamp(0.05, 0.14).square()).unsqueeze(0)
            g_map = torch.matmul(torch.exp(-d2 / sig2), w_p).view(grid_h, grid_w)
            lev_grids.append(g_map)

        return dict(
            A0=a0.detach() * 1000.0,
            H1=h_out["H1"][0].detach() * 1000.0,
            H2=h_out["H2"][0].detach() * 1000.0,
            squad_leverage=squad_lev.detach(),
            leverage_grid_p1=lev_grids[0].detach(),
            leverage_grid_p2=lev_grids[1].detach(),
        )

    def compute_plan_misalignment(
        self,
        prefix_seq: dict[str, torch.Tensor],
        h_last_fog: torch.Tensor,
        c_last_fog: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Measures shared-policy divergence between actual fog-bound `pi(u_p | h^(p), c_p)`
        and fog-lifted `pi(u_p | h^(omni, p), c_p)`, attributing misalignment directly to
        individual unscouted enemy squads via gradient sensitivity!"""
        q_sq = prefix_seq["q_squad"].float()    # [1, T0, 2, M, 14]
        z_sq = prefix_seq["z_squad"].float()
        gate = (prefix_seq["gate"].float() if "gate" in prefix_seq else prefix_seq["active"].float()).clamp(0.0, 1.0)
        q_mac = prefix_seq["q_macro"].float()
        z_mac = prefix_seq["z_macro"].float()
        tempo = prefix_seq["tempo_feat"].float()
        _, T0, _, M, _ = q_sq.shape
        dev = q_sq.device

        # Prefix steps 0 .. T0-2 remain actual history; at T0-1 lift fog completely!
        if "q_opp_obs" in prefix_seq:
            h_opp_hist = self.persp_enc.summarize_observed_opp(
                prefix_seq["q_opp_obs"], prefix_seq["z_opp_obs"], prefix_seq["gate_opp_obs"]
            )
        else:
            vl_p, sh_p, ex_p = self.sight.compute_visibility(q_sq, z_sq, gate)
            h_opp_hist = self.persp_enc.summarize_rollout_opp(q_sq, z_sq, gate, vl_p, sh_p, ex_p)

        # Differentiable visibility lift on final frame T0 - 1 to attribute unscouted threats
        opp_gate_last = gate[:, -1].flip(dims=[1]).detach()  # [1, 2, M]
        if "gt_vis_live" in prefix_seq:
            actual_vis_last = prefix_seq["gt_vis_live"][:, -1].float().clamp(0.0, 1.0)
        else:
            actual_vis_last, _, _ = self.sight.compute_visibility(q_sq[:, -1], z_sq[:, -1], gate[:, -1])

        unscouted_mask = (1.0 - actual_vis_last) * opp_gate_last  # [1, 2, M]: [p_viewer, m_enemy]

        with torch.enable_grad():
            vis_lift = opp_gate_last.clone().requires_grad_(True)
            z_zero = torch.zeros_like(vis_lift)
            h_opp_omni_last = self.persp_enc.summarize_rollout_opp(
                q_sq[:, -1].detach(),
                z_sq[:, -1].detach(),
                gate[:, -1].detach(),
                vis_lift,
                z_zero,
                vis_lift,
            )  # [1, 2, d_model]
            if T0 > 1:
                h_opp_omni_seq = torch.cat([h_opp_hist[:, :-1].detach(), h_opp_omni_last.unsqueeze(1)], dim=1)
            else:
                h_opp_omni_seq = h_opp_omni_last.unsqueeze(1)

            tok_omni, cap_omni = self.persp_enc.summarize_perspective_frame(
                q_sq.detach(), z_sq.detach(), gate.detach(), q_mac.detach(), z_mac.detach(), tempo.detach(), h_opp_omni_seq
            )
            h_omni_seq, _ = self.persp_enc.forward_sequence(tok_omni, cap_omni)
            h_omni_last = h_omni_seq[:, -1]  # [1, 2, d_model]

            # Evaluate policy velocity divergence across K_probe = 8 Common Random Probes
            K_p = 8
            u_probes = torch.randn(
                (K_p, 2, self.flow_prior.d_u), generator=generator, device=dev, dtype=h_omni_last.dtype
            )
            s_probe = torch.full((K_p, 2), 0.5, device=dev, dtype=h_omni_last.dtype)
            v_fog = self.flow_prior.velocity(
                u_probes, s_probe, h_last_fog.detach().expand(K_p, -1, -1), c_last_fog.detach().expand(K_p, -1, -1)
            )
            v_omni = self.flow_prior.velocity(
                u_probes, s_probe, h_omni_last.expand(K_p, -1, -1), c_last_fog.detach().expand(K_p, -1, -1)
            )
            # Per-player RMS intent velocity shift [2]
            div_per_p = (v_omni.float() - v_fog.float()).square().mean(dim=(0, 2)).sqrt()  # [2]
            (g_lift,) = torch.autograd.grad(div_per_p.sum(), (vis_lift,), create_graph=False)

        # Attribution score on enemy `1 - p`'s squads for viewer `p`: unscouted * |d(div_p)/d(vis_m)|
        unscouted_attr = (g_lift[0].abs() * unscouted_mask[0]).detach()  # [2, M]: viewer p -> enemy 1-p slot m
        return dict(
            plan_misalignment=div_per_p.detach(),       # [2]
            unscouted_attribution=unscouted_attr,       # [2, M]
            unscouted_mask=unscouted_mask[0].detach(),  # [2, M]
        )

    def forward_sequence(
        self,
        seq: dict[str, torch.Tensor],
        static_map: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Trains ObserverWorldModelL2 across `[B, T, ...]` (`T = 48` = `240s` macro sequences)."""
        q_sq = seq["q_squad"].float()
        z_sq = seq["z_squad"].float()
        gate = (seq["gate"].float() if "gate" in seq else seq["active"].float()).clamp(0.0, 1.0)
        active = gate > 0.02
        q_mac = seq["q_macro"].float()
        z_mac = seq["z_macro"].float()
        u_post = seq["u_flat"].float()
        tempo = seq["tempo_feat"].float()
        B, T, _, M, _ = q_sq.shape
        dev = q_sq.device

        xi_w_post = (
            seq["xi_world"].float()
            if "xi_world" in seq
            else torch.zeros(B, T - 1, SP.D_XI_WORLD_DEFAULT, device=dev)
        )
        combat_loss = (
            seq["combat_loss"].float()
            if "combat_loss" in seq
            else torch.zeros(B, T - 1, 2, device=dev)
        )
        z_omni = (
            seq["z_omni_scene"].float()
            if "z_omni_scene" in seq
            else torch.zeros(B, T, self.d_squad, device=dev)
        )

        # 1. Dynamic visibility & shimmer calibration against engine perception
        vis_seq, shim_seq, _ = self.sight.compute_visibility(q_sq, z_sq, gate)
        exp_seq = vis_seq.detach().cummax(dim=1).values

        loss_vis = q_sq.sum() * 0.0
        if "gt_vis_live" in seq:
            with torch.autocast(device_type=dev.type, enabled=False):
                opp_g = gate.flip(dims=[2]).float()
                denom_v = opp_g.sum().clamp(min=1.0)
                bce_v = F.binary_cross_entropy(
                    vis_seq.float().clamp(1e-4, 1.0 - 1e-4),
                    seq["gt_vis_live"].float().clamp(0.0, 1.0),
                    reduction="none",
                )
                loss_vis = (bce_v * opp_g).sum() / denom_v
                if "gt_shimmer" in seq:
                    bce_s = F.binary_cross_entropy(
                        shim_seq.float().clamp(1e-4, 1.0 - 1e-4),
                        seq["gt_shimmer"].float().clamp(0.0, 1.0),
                        reduction="none",
                    )
                    loss_vis = loss_vis + 0.5 * (bce_s * opp_g).sum() / denom_v

        # 2. Strictly leak-free Perspective Streams h^(1), h^(2) + rollout projector alignment
        h_opp_syn = self.persp_enc.summarize_rollout_opp(
            q_sq.detach(), z_sq.detach(), gate.detach(), vis_seq.detach(), shim_seq.detach(), exp_seq.detach()
        )
        if "q_opp_obs" in seq:
            h_opp_obs = self.persp_enc.summarize_observed_opp(
                seq["q_opp_obs"], seq["z_opp_obs"], seq["gate_opp_obs"]
            )
            loss_opp_proj = F.mse_loss(h_opp_syn.float(), h_opp_obs.detach().float())
            h_opp_for_persp = h_opp_obs
        else:
            loss_opp_proj = q_sq.sum() * 0.0
            h_opp_for_persp = h_opp_syn

        tok_seq, cap_inst = self.persp_enc.summarize_perspective_frame(
            q_sq, z_sq, gate, q_mac, z_mac, tempo, h_opp_for_persp
        )
        h_seq, c_seq = self.persp_enc.forward_sequence(tok_seq, cap_inst)

        # 3. Theory-of-Mind Opponent-Belief Head + Omniscient Stream 3 (h_obs)
        loss_belief = self.belief_head.belief_loss(h_seq, q_sq, z_sq, gate, q_mac)
        obs_tok_seq = self.obs_enc.summarize_frame(
            q_sq, z_sq, gate, q_mac, z_mac, z_omni, h_seq.detach()
        )
        h_obs_seq = self.obs_enc.forward_sequence(obs_tok_seq)

        # 4. Sharp 1-step (5s) Port-Hamiltonian Transition on Hindsight (u_post, xi_world)
        # Detach s_t_flat state inputs so transition MSE trains ph + u_post + xi_world without pulling pooler to collapse!
        M1 = B * (T - 1)
        s_t_flat = dict(
            q_squad=q_sq[:, :-1].detach().reshape(M1, 2, M, SP.N_SQUAD_Q),
            z_squad=z_sq[:, :-1].detach().reshape(M1, 2, M, self.d_squad),
            gate=gate[:, :-1].detach().reshape(M1, 2, M),
            active=active[:, :-1].detach().reshape(M1, 2, M),
            q_macro=q_mac[:, :-1].detach().reshape(M1, 2, SP.N_MACRO_Q),
            z_macro=z_mac[:, :-1].detach().reshape(M1, 2, self.d_squad),
        )
        s_n_flat = dict(
            q_squad=q_sq[:, 1:].detach().reshape(M1, 2, M, SP.N_SQUAD_Q),
            z_squad=z_sq[:, 1:].detach().reshape(M1, 2, M, self.d_squad),
            gate=gate[:, 1:].detach().reshape(M1, 2, M),
            active=active[:, 1:].detach().reshape(M1, 2, M),
            q_macro=q_mac[:, 1:].detach().reshape(M1, 2, SP.N_MACRO_Q),
            z_macro=z_mac[:, 1:].detach().reshape(M1, 2, self.d_squad),
        )

        pred_1, pieces_1 = self.ph.step_dynamics(
            s_t_flat,
            u_post.reshape(M1, 2, -1),
            xi_world=xi_w_post.reshape(M1, -1),
            static_map=static_map,
        )
        loss_phys_1, loss_lat_1, m1_logs = self._state_transition_losses(pred_1, s_n_flat)
        _, _, copy_logs = self._state_transition_losses(s_t_flat, s_n_flat)

        # Action Goal Alignment: B(S) u_p ascends +grad_p A_p
        bu_cat = torch.cat([pieces_1["bu_sq"].flatten(2), pieces_1["bu_mac"]], dim=-1).float()
        gg_cat = torch.cat([pieces_1["g_sq_goal"].flatten(2), pieces_1["g_mac_goal"]], dim=-1).detach().float()
        bn = (bu_cat.square().sum(dim=-1) + 1e-8).sqrt()
        gn = (gg_cat.square().sum(dim=-1) + 1e-8).sqrt()
        val_g = (bn > 1e-3) & (gn > 1e-3)
        cos_g = (bu_cat * gg_cat).sum(dim=-1) / (bn * gn).clamp(min=1e-6)
        loss_goal = (F.softplus(-2.0 * cos_g) * val_g.float()).sum() / val_g.float().sum().clamp(min=1.0)
        goal_cos = (cos_g * val_g.float()).sum().detach() / val_g.float().sum().clamp(min=1.0)

        # 5. Purely Learned Advantage A = H_1 - H_2 & Multi-Horizon Combat-Destruction TD!
        # ZERO bank+cost baseline, ZERO phys_adv, ZERO winner labels!
        h_all = self.ph.evaluate_hamiltonian(
            torch.cat([q_sq.detach(), z_sq], dim=-1).reshape(B * T, 2, M, -1),
            torch.cat([q_mac.detach(), z_mac], dim=-1).reshape(B * T, 2, -1),
            gate.detach().reshape(B * T, 2, M),
            static_map=static_map,
        )
        a1_seq = h_all["A1"].float().view(B, T)

        # Step reward r_t = combat_loss_P2 - combat_loss_P1 (in 1,000-mineral+gas units destroyed)
        r_step = (combat_loss[:, :, 1] - combat_loss[:, :, 0]).detach()  # [B, T-1]
        gamma = 0.97
        loss_td = a1_seq.sum() * 0.0
        horizons = [m for m in (2, 4, 8, 16, 24) if m < T]
        for m in horizons:
            # Discounted sum of realized combat/structural destruction impulses over [t .. t + m - 1]
            disc_ret = torch.zeros_like(a1_seq[:, :-m])
            g_pow = 1.0
            for k_s in range(m):
                disc_ret = disc_ret + g_pow * r_step[:, k_s : T - m + k_s]
                g_pow *= gamma
            tgt_a = disc_ret + (gamma ** m) * a1_seq[:, m:].detach()
            loss_td = loss_td + F.smooth_l1_loss(a1_seq[:, :-m], tgt_a) / len(horizons)

        # 6. Dual-Mode CFG Strategy Flow Prior + World Outcome Flow Prior
        h_t = h_seq[:, :-1]
        c_t = c_seq[:, :-1]
        # Player-oriented future advantage change Delta A_p for CFG conditioning (with 50% null dropout)
        m_cfg = min(8, T - 1)
        fut_idx = (torch.arange(T - 1, device=dev) + m_cfg).clamp(max=T - 1)
        da_p1 = (a1_seq[:, fut_idx] - a1_seq[:, :-1]).detach()
        delta_adv_p = torch.stack([da_p1, -da_p1], dim=-1)  # [B, T-1, 2]

        loss_flow = self.flow_prior.flow_matching_loss(
            u_post, h_t, c_t, delta_adv_p=delta_adv_p, cfg_drop_prob=0.50
        )
        loss_xi_flow = self.xi_prior.flow_matching_loss(
            xi_w_post,
            q_sq[:, :-1],
            z_sq[:, :-1],
            gate[:, :-1],
            q_mac[:, :-1],
            z_mac[:, :-1],
            u_post,
            h_obs_seq[:, :-1],
        )

        tempo_pred = self.flow_prior.tempo_pred(torch.cat([h_t, c_t, u_post.detach()], dim=-1)).float()
        loss_tempo = F.smooth_l1_loss(tempo_pred, tempo[:, 1:].detach())

        # 7. Sharp Multi-Step Rollout Stability (k = 2..K on sharp hindsight (u_post, xi_world)!)
        K = min(self.rollout_steps, T - 1)
        loss_phys_multi = loss_phys_1 * 0.0
        loss_lat_multi = loss_lat_1 * 0.0
        if K >= 2 and T > K:
            MK = B * (T - K)
            cur_s = dict(
                q_squad=q_sq[:, : T - K].detach().reshape(MK, 2, M, SP.N_SQUAD_Q),
                z_squad=z_sq[:, : T - K].detach().reshape(MK, 2, M, self.d_squad),
                gate=gate[:, : T - K].detach().reshape(MK, 2, M),
                active=active[:, : T - K].detach().reshape(MK, 2, M),
                q_macro=q_mac[:, : T - K].detach().reshape(MK, 2, SP.N_MACRO_Q),
                z_macro=z_mac[:, : T - K].detach().reshape(MK, 2, self.d_squad),
            )
            for k in range(1, K + 1):
                uk = u_post[:, k - 1 : T - K + k - 1].detach().reshape(MK, 2, -1)
                xk = xi_w_post[:, k - 1 : T - K + k - 1].detach().reshape(MK, -1)
                cur_s, _ = self.ph.step_dynamics(
                    cur_s, uk, xi_world=xk, static_map=static_map, create_graph_h=False
                )
                if k >= 2:
                    true_k = dict(
                        q_squad=q_sq[:, k : T - K + k].reshape(MK, 2, M, SP.N_SQUAD_Q),
                        z_squad=z_sq[:, k : T - K + k].reshape(MK, 2, M, self.d_squad),
                        gate=gate[:, k : T - K + k].reshape(MK, 2, M),
                        active=active[:, k : T - K + k].reshape(MK, 2, M),
                        q_macro=q_mac[:, k : T - K + k].reshape(MK, 2, SP.N_MACRO_Q),
                        z_macro=z_mac[:, k : T - K + k].reshape(MK, 2, self.d_squad),
                    )
                    w_k = 0.5 ** (k - 1)
                    lp_k, ll_k, _ = self._state_transition_losses(cur_s, true_k)
                    loss_phys_multi = loss_phys_multi + w_k * lp_k
                    loss_lat_multi = loss_lat_multi + w_k * ll_k

        loss = (
            loss_phys_1
            + 0.5 * loss_phys_multi
            + 0.15 * (loss_lat_1 + 0.5 * loss_lat_multi)
            + 1.0 * loss_flow.float()
            + 0.5 * loss_xi_flow.float()
            + 0.25 * loss_belief.float()
            + 0.15 * loss_opp_proj.float()
            + 0.10 * loss_tempo
            + 0.10 * loss_vis
            + 0.05 * loss_goal
            + 0.50 * loss_td
        )

        logs = dict(
            loss=loss.detach(),
            loss_phys=loss_phys_1.detach(),
            loss_phys_multi=loss_phys_multi.detach(),
            loss_flow=loss_flow.detach().float(),
            loss_xi_flow=loss_xi_flow.detach().float(),
            loss_belief=loss_belief.detach().float(),
            loss_opp_proj=loss_opp_proj.detach().float(),
            loss_td=loss_td.detach(),
            loss_vis=loss_vis.detach(),
            loss_goal=loss_goal.detach(),
            goal_cos=goal_cos,
            A1_k_mean=a1_seq.mean().detach(),
            A1_k_abs=a1_seq.abs().mean().detach(),
            rmse_cost=m1_logs["mse_cost"].sqrt(),
            copy_rmse_cost=copy_logs["mse_cost"].sqrt(),
            rmse_hp=m1_logs["mse_hp"].sqrt(),
            copy_rmse_hp=copy_logs["mse_hp"].sqrt(),
            rmse_count=m1_logs["mse_count"].sqrt(),
            copy_rmse_count=copy_logs["mse_count"].sqrt(),
            rmse_minerals=m1_logs["mse_minerals"].sqrt(),
            copy_rmse_minerals=copy_logs["mse_minerals"].sqrt(),
            loss_field=m1_logs["loss_field"],
            copy_loss_field=copy_logs["loss_field"],
        )
        return loss, logs

    @torch.no_grad()
    def rollout_observer(
        self,
        prefix_seq: dict[str, torch.Tensor],
        static_map: dict[str, torch.Tensor] | None = None,
        n_branches: int = 32,
        horizon_steps: int = 24,
        flow_steps: int = 4,
        temperature: float = 1.0,
        grid_h: int = 64,
        grid_w: int = 64,
        seed: int = 12345,
        mode: str = "forecast",
        cfg_scale: float = 0.85,
        grad_step: float = 0.0,
        n_cp_candidates: int = 3,
    ) -> dict[str, torch.Tensor]:
        """Rolls out `N = n_branches` stochastic `5.0s` futures over `horizon_steps` (`24 -> +120s`)
        using **Common Random Numbers (`seed`)**, returning:
          - Purely learned omniscient positional evaluation (`A0`, `prob_lead_now_p1`, `A_traj`),
          - Intrinsic learned leverage (`squad_leverage`, `leverage_grid_p1/p2`),
          - Theory-of-Mind Plan Misalignment & unscouted threat attribution,
          - Representative step-0 `(u_step0, xi_world_step0, s_step1)` for `L1ChunkFlowDecoder`!
        """
        dev = prefix_seq["q_squad"].device
        q_sq = prefix_seq["q_squad"].float()
        z_sq = prefix_seq["z_squad"].float()
        gate = (prefix_seq["gate"].float() if "gate" in prefix_seq else prefix_seq["active"].float()).clamp(0.0, 1.0)
        active = gate > 0.02
        q_mac = prefix_seq["q_macro"].float()
        z_mac = prefix_seq["z_macro"].float()
        tempo = prefix_seq["tempo_feat"].float()
        _, T0, _, M, _ = q_sq.shape
        z_omni = (
            prefix_seq["z_omni_scene"].float()
            if "z_omni_scene" in prefix_seq
            else torch.zeros(1, T0, self.d_squad, device=dev)
        )

        # Strictly separated Common Random Number streams so Forecast & Counterplay share
        # identical opponent policy noise and world-outcome combat noise across branches & scrubs!
        gen_mis = torch.Generator(device=dev)
        gen_mis.manual_seed(int(seed))
        gen_u_p1 = torch.Generator(device=dev)
        gen_u_p1.manual_seed(int(seed) + 101)
        gen_u_p2 = torch.Generator(device=dev)
        gen_u_p2.manual_seed(int(seed) + 202)
        gen_xi = torch.Generator(device=dev)
        gen_xi.manual_seed(int(seed) + 303)
        gen_cp = torch.Generator(device=dev)
        gen_cp.manual_seed(int(seed) + 7919)

        counterplay_p = None
        if mode == "counterplay_p1":
            counterplay_p = 0
        elif mode == "counterplay_p2":
            counterplay_p = 1

        # 1. Intrinsic Learned Leverage + initial A0 in ONE differentiable pass
        lev_info = self.compute_intrinsic_leverage(
            dict(
                q_squad=q_sq[:, -1],
                z_squad=z_sq[:, -1],
                gate=gate[:, -1],
                q_macro=q_mac[:, -1],
                z_macro=z_mac[:, -1],
            ),
            static_map=static_map,
            grid_h=grid_h,
            grid_w=grid_w,
        )
        a0_res = lev_info["A0"]  # scalar in resource-equivalents

        # 2. Observed prefix encoding (0.000000e+00 leak on observed frames!)
        vis_pref, shim_pref, _ = self.sight.compute_visibility(q_sq, z_sq, gate)
        exp_pref = vis_pref.detach().cummax(dim=1).values
        exp_cur = exp_pref[:, -1]

        if "q_opp_obs" in prefix_seq:
            h_opp_pref = self.persp_enc.summarize_observed_opp(
                prefix_seq["q_opp_obs"], prefix_seq["z_opp_obs"], prefix_seq["gate_opp_obs"]
            )
        else:
            h_opp_pref = self.persp_enc.summarize_rollout_opp(
                q_sq, z_sq, gate, vis_pref, shim_pref, exp_pref
            )

        tok_pref, cap_pref = self.persp_enc.summarize_perspective_frame(
            q_sq, z_sq, gate, q_mac, z_mac, tempo, h_opp_pref
        )
        h_pref_seq, c_pref_seq = self.persp_enc.forward_sequence(tok_pref, cap_pref)
        obs_tok_pref = self.obs_enc.summarize_frame(
            q_sq, z_sq, gate, q_mac, z_mac, z_omni, h_pref_seq
        )

        # 3. Theory-of-Mind Plan Misalignment & Unscouted Threat Attribution at t = T0 - 1
        mis_info = self.compute_plan_misalignment(
            prefix_seq, h_pref_seq[:, -1], c_pref_seq[:, -1], generator=gen_mis
        )

        # In Counterplay mode, grant ONLY `counterplay_p` full-attention awareness at t = T0 - 1
        # (while opponent `1 - counterplay_p` remains strictly fog-bound!)
        tok_start = tok_pref
        cap_start = cap_pref
        exp_start = exp_cur.clone()
        if counterplay_p is not None:
            opp_gate_last = gate[:, -1].flip(dims=[1]).detach()
            h_opp_omni_last = self.persp_enc.summarize_rollout_opp(
                q_sq[:, -1], z_sq[:, -1], gate[:, -1], opp_gate_last, torch.zeros_like(opp_gate_last), opp_gate_last
            )
            h_opp_cp = h_opp_pref.clone()
            h_opp_cp[:, -1, counterplay_p] = h_opp_omni_last[:, counterplay_p]
            tok_start, cap_start = self.persp_enc.summarize_perspective_frame(
                q_sq, z_sq, gate, q_mac, z_mac, tempo, h_opp_cp
            )
            exp_start[:, counterplay_p] = torch.maximum(exp_start[:, counterplay_p], opp_gate_last[:, counterplay_p])

        # 4. Batched Full-Horizon (+120s) Rollout with On-Manifold Trajectory Shooting in Counterplay
        N = n_branches
        K_shoot = max(1, int(n_cp_candidates)) if counterplay_p is not None else 1
        B_roll = K_shoot * N

        tok_hist = tok_start.expand(B_roll, -1, -1, -1).clone()
        cap_hist = cap_start.expand(B_roll, -1, -1, -1).clone()
        obs_hist = obs_tok_pref.expand(B_roll, -1, -1).clone()
        exp_n = exp_start.expand(B_roll, -1, -1).clone()

        cur_s = dict(
            q_squad=q_sq[:, -1].expand(B_roll, -1, -1, -1).clone(),
            z_squad=z_sq[:, -1].expand(B_roll, -1, -1, -1).clone(),
            gate=gate[:, -1].expand(B_roll, -1, -1).clone(),
            active=active[:, -1].expand(B_roll, -1, -1).clone(),
            q_macro=q_mac[:, -1].expand(B_roll, -1, -1).clone(),
            z_macro=z_mac[:, -1].expand(B_roll, -1, -1).clone(),
        )

        a_steps = [(a0_res / 1000.0).expand(B_roll)]
        step0_u_all = None
        step0_xi_all = None
        step0_s1_all = None
        d_u = self.flow_prior.d_u
        d_xi_w = self.xi_prior.d_xi_world

        for step_k in range(horizon_steps):
            h_seq, c_seq = self.persp_enc.forward_sequence(tok_hist, cap_hist)
            h_last = h_seq[:, -1]
            c_last = c_seq[:, -1]
            h_obs_last = self.obs_enc.forward_sequence(obs_hist)[:, -1]

            eps_p1 = torch.randn((N, d_u), generator=gen_u_p1, device=dev, dtype=torch.float32)
            eps_p2 = torch.randn((N, d_u), generator=gen_u_p2, device=dev, dtype=torch.float32)
            eps_xi = torch.randn((N, d_xi_w), generator=gen_xi, device=dev, dtype=torch.float32)

            if K_shoot == 1:
                u_noise = torch.stack([eps_p1, eps_p2], dim=1)       # [N, 2, d_u]
                xi_noise = eps_xi                                    # [N, d_xi_w]
            else:
                u_base = torch.stack([eps_p1, eps_p2], dim=1)        # [N, 2, d_u]
                u_kn = u_base.unsqueeze(0).repeat(K_shoot, 1, 1, 1)  # [K_shoot, N, 2, d_u]
                # Candidates 1..K_shoot-1 explore diverse on-manifold trajectories for counterplay_p
                # against the EXACT SAME opponent policy & world combat noise!
                eps_alt = torch.randn(
                    (K_shoot - 1, N, d_u), generator=gen_cp, device=dev, dtype=torch.float32
                ) * 1.08
                u_kn[1:, :, counterplay_p, :] = eps_alt
                u_noise = u_kn.reshape(B_roll, 2, d_u)
                xi_noise = eps_xi.repeat(K_shoot, 1)

            u_samp, tempo_next = self.flow_prior.sample(
                h_last,
                c_last,
                steps=flow_steps,
                temperature=temperature,
                counterplay_player=counterplay_p,
                cfg_scale=cfg_scale,
                init_noise=u_noise,
            )

            xi_w_samp = self.xi_prior.sample(
                cur_s["q_squad"],
                cur_s["z_squad"],
                cur_s["gate"],
                cur_s["q_macro"],
                cur_s["z_macro"],
                u_samp,
                h_obs_last,
                steps=flow_steps,
                temperature=temperature,
                init_noise=xi_noise,
            )

            cur_s, _ = self.ph.step_dynamics(
                cur_s, u_samp.float(), xi_world=xi_w_samp, static_map=static_map
            )

            if step_k == 0:
                step0_u_all = u_samp.float().clone()
                step0_xi_all = xi_w_samp.float().clone()
                step0_s1_all = {k: v.clone() for k, v in cur_s.items()}

            hk = self.ph.evaluate_hamiltonian(
                torch.cat([cur_s["q_squad"], cur_s["z_squad"]], dim=-1),
                torch.cat([cur_s["q_macro"], cur_s["z_macro"]], dim=-1),
                cur_s["gate"],
                static_map=static_map,
            )
            a_steps.append(hk["A1"].float())

            if step_k + 1 < horizon_steps:
                vl_next, sh_next, exp_n = self.sight.compute_visibility(
                    cur_s["q_squad"], cur_s["z_squad"], cur_s["gate"], prev_explored=exp_n
                )
                h_opp_next = self.persp_enc.summarize_rollout_opp(
                    cur_s["q_squad"], cur_s["z_squad"], cur_s["gate"], vl_next, sh_next, exp_n
                )
                tok_next, cap_next = self.persp_enc.summarize_perspective_frame(
                    cur_s["q_squad"],
                    cur_s["z_squad"],
                    cur_s["gate"],
                    cur_s["q_macro"],
                    cur_s["z_macro"],
                    tempo_next,
                    h_opp_next,
                )
                tok_hist = torch.cat([tok_hist[:, -47:], tok_next.unsqueeze(1)], dim=1)
                cap_hist = torch.cat([cap_hist[:, -47:], cap_next.unsqueeze(1)], dim=1)

                h_next_last = self.persp_enc.forward_sequence(tok_hist[:, -8:], cap_hist[:, -8:])[0][:, -1]
                z_omni_zero = torch.zeros(B_roll, self.d_squad, device=dev)
                obs_next = self.obs_enc.summarize_frame(
                    cur_s["q_squad"],
                    cur_s["z_squad"],
                    cur_s["gate"],
                    cur_s["q_macro"],
                    cur_s["z_macro"],
                    z_omni_zero,
                    h_next_last,
                )
                obs_hist = torch.cat([obs_hist[:, -47:], obs_next.unsqueeze(1)], dim=1)

        a_traj_all = torch.stack(a_steps, dim=1) * 1000.0  # [B_roll, H + 1]
        if K_shoot > 1:
            a_kn = a_traj_all.view(K_shoot, N, horizon_steps + 1)
            h_short = min(4, horizon_steps)
            h_mid = min(12, horizon_steps)
            j_score = 0.15 * a_kn[:, :, h_short] + 0.35 * a_kn[:, :, h_mid] + 0.50 * a_kn[:, :, -1]
            if counterplay_p == 1:
                j_score = -j_score
            best_k = j_score.argmax(dim=0)  # [N]
            sel_flat = best_k * N + torch.arange(N, device=dev)
            a_traj = a_traj_all[sel_flat]
            cur_s = {k: v[sel_flat] for k, v in cur_s.items()}
            if step0_u_all is not None:
                step0_u_all = step0_u_all[sel_flat]
                step0_xi_all = step0_xi_all[sel_flat]
                step0_s1_all = {k: v[sel_flat] for k, v in step0_s1_all.items()}
        else:
            a_traj = a_traj_all

        a_end = a_traj[:, -1]
        drift = a_end - a0_res

        # Select representative branch for L1ChunkFlowDecoder playback:
        # median outcome branch in Forecast mode, or 85th-percentile multi-horizon return in Counterplay mode!
        h_short = min(4, horizon_steps)
        h_mid = min(12, horizon_steps)
        rep_score = 0.15 * a_traj[:, h_short] + 0.35 * a_traj[:, h_mid] + 0.50 * a_end
        if counterplay_p == 0:
            rep_idx = int(torch.argsort(rep_score)[int(0.85 * (N - 1))].item())
        elif counterplay_p == 1:
            rep_idx = int(torch.argsort(-rep_score)[int(0.85 * (N - 1))].item())
        else:
            rep_idx = int(torch.argsort(a_end)[N // 2].item())

        s1_rep = (
            {k: v[rep_idx : rep_idx + 1] for k, v in step0_s1_all.items()}
            if step0_s1_all is not None
            else None
        )
        u0_rep = step0_u_all[rep_idx : rep_idx + 1] if step0_u_all is not None else None
        xi0_rep = step0_xi_all[rep_idx : rep_idx + 1] if step0_xi_all is not None else None

        #Omniscient positional lead probability sigmoid(A0 / 1500) + terminal stronger-hand fraction
        prob_lead_now_p1 = torch.sigmoid(a0_res / 1500.0)
        prob_improve_p1 = (drift > 0.0).float().mean()
        prob_stronger_p1 = (a_end > 0.0).float().mean()
        expected_drift = drift.mean()
        std_drift = drift.std(unbiased=False)
        k_cvar = max(1, N // 10)
        cvar10_p1 = torch.topk(drift, k=k_cvar, largest=False).values.mean()

        end_total_cost = cur_s["q_squad"][:, :, :, SP.SQ["cost"]].sum(dim=-1).mean(dim=0)
        end_minerals = cur_s["q_macro"][:, :, 0].mean(dim=0)

        return dict(
            A0=a0_res,
            H1_0=lev_info["H1"],
            H2_0=lev_info["H2"],
            prob_lead_now_p1=prob_lead_now_p1,
            A_traj=a_traj,
            expected_drift=expected_drift,
            prob_improve_p1=prob_improve_p1,
            prob_stronger_hand_p1=prob_stronger_p1,
            std_drift=std_drift,
            cvar10_p1=cvar10_p1,
            capability_p1=c_pref_seq[0, -1, 0].float(),
            capability_p2=c_pref_seq[0, -1, 1].float(),
            end_total_cost=end_total_cost,
            end_minerals=end_minerals,
            # Intrinsic learned leverage (backwards-aliased to squad_risk / threat_grid_* too)
            squad_leverage=lev_info["squad_leverage"],
            squad_risk=lev_info["squad_leverage"],
            squad_pos=q_sq[0, -1, :, :, 0:2],
            squad_gate=gate[0, -1],
            squad_active=active[0, -1],
            leverage_grid_p1=lev_info["leverage_grid_p1"],
            leverage_grid_p2=lev_info["leverage_grid_p2"],
            threat_grid_p1=lev_info["leverage_grid_p1"],
            threat_grid_p2=lev_info["leverage_grid_p2"],
            # Theory-of-Mind Plan Misalignment & Unscouted Threats
            plan_misalignment=mis_info["plan_misalignment"],
            unscouted_attribution=mis_info["unscouted_attribution"],
            unscouted_mask=mis_info["unscouted_mask"],
            # Step-0 representative L2 transition for L1ChunkFlowDecoder 16-frame micro playback
            rep_branch_idx=rep_idx,
            step0_u=u0_rep,
            step0_xi_world=xi0_rep,
            step0_s1=s1_rep,
        )
