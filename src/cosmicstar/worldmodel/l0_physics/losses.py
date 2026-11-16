"""Multi-task training & evaluation losses and Hungarian assignment for L0 `Physics`."""

from __future__ import annotations

import collections
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..core.constants import N_BUFF_IDS, N_RAW_ABILITY_IDS, PC_KEYS
from ..data.game_store import C_DYNAMIC, NEUTRAL, PATCH, PROD, QUEUE, TOP
from .encoders import I, N_FLAGS, PLY_PROBE_IDX
from .heads import (
    ACTIVE_FLAG_BITS,
    DISCOUNT,
    EFF_MARGIN,
    EFF_RADII,
    MARGIN,
    MAX_BIRTHS,
    MAX_EFF_AGE,
    MAX_EFF_DOTS,
    MAX_EFF_SPAWNS,
    N_EFF_SPAWN_TYPES,
    N_PTRS,
    POS_BOOST,
    POS_SCALE,
    REG,
    REG_ALWAYS_VALID,
)
from .trunk import GRID


class PhysicsLossesMixin:
    """Loss computation and evaluation metrics mixed into `Physics`."""

    def player_losses(self, nxt_ply, clean_ply, b, f_src, f_tgt):
        """Predict raw un-logged (minerals, vespene) change over the tubelet on nxt_ply,
        and run a detached macro-count probe on clean_ply.detach() (zero trunk gradient)."""
        raw_src = torch.expm1(b["ply_num"][:, f_src, :, :2].float() * 8.0)
        raw_tgt = torch.expm1(b["ply_num"][:, f_tgt, :, :2].float() * 8.0)
        econ_tgt = (raw_tgt - raw_src) / 50.0
        econ_pred = self.player_next(nxt_ply).float()
        econ_w = torch.where(econ_tgt.abs() > 0.01, 1.0, 0.25)
        w_sum = econ_w.sum().clamp(min=1.0)
        pn = (
            F.smooth_l1_loss(econ_pred, econ_tgt, beta=0.5, reduction="none") * econ_w
        ).sum() / w_sum
        with torch.no_grad():
            pn_copy = (
                F.smooth_l1_loss(
                    torch.zeros_like(econ_tgt), econ_tgt, beta=0.5, reduction="none"
                )
                * econ_w
            ).sum() / w_sum

        # Detached probe: 7 macroscopic counts at f_src (no gradient into encoder/trunk)
        probe_in = clean_ply.detach()
        probe_pred = self.ply_probe(probe_in).float()
        probe_tgt = b["ply_num"][:, f_src][..., PLY_PROBE_IDX].float() * 8.0
        probe_ply = F.smooth_l1_loss(probe_pred, probe_tgt, beta=0.5)
        with torch.no_grad():
            cnt_pred = torch.expm1(probe_pred.detach().clamp(min=0.0, max=8.0))
            cnt_tgt = torch.expm1(probe_tgt)
            probe_mae = (cnt_pred - cnt_tgt).abs().mean()
        logs = dict(
            player=pn,
            player_copy=pn_copy,
            probe_ply=probe_ply.detach(),
            probe_ply_mae=probe_mae,
        )
        return pn, probe_ply, logs

    def recipient_loss(self, clean, act, b):
        """Which units carry out each command: scores each command token against every
        live owned row at the causal anchor frame f_anc with F1-calibrated BCE."""
        pr = b.get("act_recip")
        zero = act.sum() * 0
        if pr is None or not len(pr):
            return zero, {}
        B, F_act, A, d = act.shape
        F_b, R = b["mask"].shape[1], b["mask"].shape[2]
        if R == 0:
            return zero, {}
        k = max(1, self.enc.tubelet)
        pr = pr.to(act.device)
        bi_p, fi_p = pr[:, 0] // F_b, pr[:, 0] % F_b
        keep = (fi_p >= k - 1) & (fi_p < F_act) & (pr[:, 2] >= 0) & (pr[:, 2] < R)
        if not keep.any():
            return zero, {}
        bi_p, fi_p, sl_p, r_p = bi_p[keep], fi_p[keep], pr[keep, 1], pr[keep, 2]
        uq, inv = torch.unique((bi_p * F_act + fi_p) * A + sl_p, return_inverse=True)
        fb, sl = uq // A, uq % A
        ub, uf = fb // F_act, fb % F_act
        um = (uf - (k - 1)) // k
        f_anc = um * k + (k - 1)
        U = len(uq)

        who = b["act_player"][ub, uf, sl]
        own = b["owner"][ub] == who[:, None]
        valid = (
            b["mask"][ub, uf]
            & b["alive"][ub, uf]
            & b["mask"][ub, f_anc]
            & b["alive"][ub, f_anc]
        )
        valid_own = valid & own

        sel_add = torch.zeros((U, R), dtype=torch.float32, device=act.device)
        any_sel_add = torch.zeros((U, 1), dtype=torch.float32, device=act.device)
        act_sel = b.get("act_sel")
        if act_sel is not None and len(act_sel):
            asd = act_sel.to(act.device)
            sb, sf = asd[:, 0] // F_b, asd[:, 0] % F_b
            ssl, sr = asd[:, 1], asd[:, 2]
            s_ok = (sf < F_act) & (sr >= 0) & (sr < R)
            if s_ok.any():
                sb, sf, ssl, sr = sb[s_ok], sf[s_ok], ssl[s_ok], sr[s_ok]
                swho = b["act_player"][sb, sf, ssl]
                prior = (
                    (ub[:, None] == sb[None])
                    & (swho[None] == who[:, None])
                    & (sf[None] >= f_anc[:, None])
                    & (
                        (sf[None] < uf[:, None])
                        | ((sf[None] == uf[:, None]) & (ssl[None] < sl[:, None]))
                    )
                )
                ui_m, si_m = torch.nonzero(prior, as_tuple=True)
                if len(ui_m):
                    sr_m = sr[si_m]
                    rok = b["mask"][ub[ui_m], f_anc[ui_m], sr_m]
                    ui_m, sr_m = ui_m[rok], sr_m[rok]
                    if len(ui_m):
                        sel_add[ui_m, sr_m] = 1.0
                        any_sel_add[ui_m, 0] = 1.0

        grp_anc = (b["groups"][ub, f_anc] > 0).float() * b["mask"][
            ub, f_anc, :, None
        ].float()
        grp_feat = torch.cat(
            [grp_anc, sel_add[..., None], any_sel_add[:, :, None].expand(-1, R, 1)],
            dim=-1,
        )
        xy_a = b["act_xy"][ub, uf, sl].float()
        placed = b["act_num"][ub, uf, sl, 11].float()
        xy_r = (
            torch.stack([b["x"][ub, f_anc], b["y"][ub, f_anc]], -1).float()
            - b["center"][ub, None].float()
        ) * b["mask"][ub, f_anc, :, None].float()

        logit = self.recip(
            act[ub, uf, sl], clean["ent"][ub, um], xy_a, placed, xy_r, grp_feat
        )
        tgt = torch.zeros_like(logit)
        tgt[inv, r_p] = 1.0
        tgt = tgt * valid_own.float()

        pw = torch.tensor(1.25, device=logit.device)
        bce = (
            F.binary_cross_entropy_with_logits(
                logit, tgt, pos_weight=pw, reduction="none"
            )
            * valid_own.float()
        )
        loss = (0.25 * bce.sum(1) / tgt.sum(1).clamp(min=1)).mean()
        with torch.no_grad():
            f1 = lambda p_: 2 * (p_ * tgt).sum() / (p_.sum() + tgt.sum()).clamp(min=1)
            sel = (b["groups"][ub, uf, :, 10] > 0) & own
            logs = dict(
                recip=loss,
                recip_f1=f1(((logit > 0) & valid_own).float()),
                recip_sel_f1=f1((sel & valid).float()),
            )
        return loss, logs

    def _dense_raw_losses(
        self,
        x_sc,
        crit,
        larva,
        x_ft,
        x_fp,
        y_ft,
        y_fp,
        n_ft,
        n_fp,
        typ_ft,
        typ_fp,
        ins_ft,
        ins_fp,
        alive_ft,
        buffs_ft_u8,
        buffs_fp_u8,
        grp_ft,
        grp_fp,
        eng_ft,
        eng_fp,
        rf_ft,
        rf_fp,
        pos_scale,
    ):
        whole = torch.where(crit, DISCOUNT, 1.0).float()
        wt = whole[:, None]
        pos_w = torch.where(larva, whole * DISCOUNT, whole)
        pw_sum = pos_w.sum().clamp(min=1e-6)

        disp_fp = n_fp[:, I["display"]].long().clamp(0, 3)
        disp_ft = n_ft[:, I["display"]].long().clamp(0, 3)
        cloak_fp = n_fp[:, I["cloak"]].long().clamp(0, 3)
        cloak_ft = n_ft[:, I["cloak"]].long().clamp(0, 3)
        f_shifts = torch.arange(N_FLAGS, device=x_sc.device)
        flags_p_raw = n_fp[:, I["flags"]].long() | (n_fp[:, I["flags2"]].long() << 8)
        flags_t_raw = n_ft[:, I["flags"]].long() | (n_ft[:, I["flags2"]].long() << 8)
        flags_fp = ((flags_p_raw[:, None] >> f_shifts) & 1).float()
        flags_ft = ((flags_t_raw[:, None] >> f_shifts) & 1).float()
        bits_fp = torch.stack(
            [n_fp[:, I["add_on"]] >= 0, eng_fp >= 0, rf_fp[:, 0] > 0], -1
        ).float()
        bits_ft = torch.stack(
            [n_ft[:, I["add_on"]] >= 0, eng_ft >= 0, rf_ft[:, 0] > 0], -1
        ).float()

        o = self.head.predict_next(
            x_sc,
            typ_fp,
            ins_fp,
            disp_fp,
            cloak_fp,
            flags_fp,
            grp_fp,
            bits_fp,
            buffs_fp_u8,
            rf_fp,
        )

        tpos = torch.stack([(x_ft - x_fp) / pos_scale, (y_ft - y_fp) / pos_scale], -1)
        lp = F.smooth_l1_loss(
            o["reg"][:, :2].float(), tpos, beta=0.04, reduction="none"
        ).sum(-1)
        l_pos = (lp * pos_w).sum() / pw_sum
        l_pos_copy = (
            F.smooth_l1_loss(
                torch.zeros_like(tpos), tpos, beta=0.04, reduction="none"
            ).sum(-1)
            * pos_w
        ).sum() / pw_sum

        # Pillar 1: mask out fog-of-war SNAPSHOT <-> VISIBLE fake stat jumps (display == 2)
        # on non-geometry/non-contents REG channels, and normalize per channel by active
        # delta events with beta = 0.20 so combat/energy/mining deltas never shrink to 0.
        reg_t = torch.stack(
            [(n_ft[:, I[k_]] - n_fp[:, I[k_]]) / dv for k_, dv in REG], -1
        ).float()
        j = [k_ for k_, _ in REG].index("facing")
        reg_t[:, j] = torch.remainder(reg_t[:, j] + math.pi, 2 * math.pi) - math.pi
        is_snap = (disp_fp == 2) | (disp_ft == 2)
        reg_valid = (~(is_snap[:, None] & self.reg_need_vis[None, :])).float()
        chg_r = (reg_t.abs() > 0.005) & (reg_valid > 0.5)
        w_reg = wt * reg_valid * torch.where(chg_r, 1.5, 1.0)
        min_chg_r = (0.002 * (wt * reg_valid).sum(0)).clamp(min=16.0)
        n_chg_r = torch.maximum((wt * chg_r.float()).sum(0), min_chg_r)
        err_r = F.smooth_l1_loss(
            o["reg"][:, 2:].float(), reg_t, beta=0.20, reduction="none"
        )
        l_reg = 0.25 * ((err_r * w_reg).sum(0) / n_chg_r).mean()

        # Pillar 2: F1-calibrated event-normalized transition losses on f_src-residual heads
        typ_chg = typ_ft.long() != typ_fp.long()
        ce_typ = F.cross_entropy(o["type"], typ_ft.long(), reduction="none")
        l_type = (
            0.01
            * (ce_typ * whole * torch.where(typ_chg, 2.0, 1.0)).sum()
            / (whole * typ_chg.float()).sum().clamp(min=8.0)
        )

        ins_chg = ins_ft.long() != ins_fp.long()
        cargo_chg = ins_chg & ((ins_fp.long() == 1) | (ins_ft.long() == 1))
        w_ins = whole * torch.where(cargo_chg, 2.5, torch.where(ins_chg, 1.5, 1.0))
        ce_ins = F.cross_entropy(o["inside"], ins_ft.long(), reduction="none")
        l_inside = (
            0.03 * (ce_ins * w_ins).sum() / (whole * ins_chg.float()).sum().clamp(min=8.0)
        )

        act_fb = self.active_flag_bits
        fl_p_a, fl_t_a = flags_fp[:, act_fb], flags_ft[:, act_fb]
        fl_chg = (fl_t_a - fl_p_a).abs() > 0.5
        bce_fl = F.binary_cross_entropy_with_logits(
            o["flags"][:, act_fb], fl_t_a, reduction="none"
        )
        l_flags = (
            0.025
            * (bce_fl * wt * torch.where(fl_chg, 2.0, 1.0)).sum()
            / (wt * fl_chg.float()).sum().clamp(min=16.0)
        )

        disp_chg = disp_ft != disp_fp
        ce_disp = F.cross_entropy(o["display"], disp_ft, reduction="none")
        l_display = (
            0.02
            * (ce_disp * whole * torch.where(disp_chg, 1.5, 1.0)).sum()
            / (whole * disp_chg.float()).sum().clamp(min=8.0)
        )

        clk_chg = cloak_ft != cloak_fp
        ce_clk = F.cross_entropy(o["cloak"], cloak_ft, reduction="none")
        l_cloak = (
            0.02
            * (ce_clk * whole * torch.where(clk_chg, 2.0, 1.0)).sum()
            / (whole * clk_chg.float()).sum().clamp(min=8.0)
        )

        # F1-calibrated tubelet death BCE: pos_weight = 2.0, normalized by deaths
        died = (~alive_ft).float()
        pw_alive = torch.tensor(2.0, device=x_sc.device)
        bce_alive = (
            F.binary_cross_entropy_with_logits(
                o["alive"],
                died,
                pos_weight=pw_alive,
                reduction="none",
            )
            * whole
        )
        l_alive = 0.1 * bce_alive.sum() / (died * whole).sum().clamp(min=4.0)
        pred_d = (o["alive"].detach() > 0).float() * (~crit).float()
        tgt_d = died * (~crit).float()
        death_f1 = 2.0 * (pred_d * tgt_d).sum() / (pred_d.sum() + tgt_d.sum()).clamp(
            min=1.0
        )

        bf_t, bf_p = buffs_ft_u8.float(), buffs_fp_u8.float()
        bf_chg = (bf_t - bf_p).abs() > 0.5
        bce_bf = F.binary_cross_entropy_with_logits(o["buffs"], bf_t, reduction="none")
        l_buffs = (
            0.025
            * (bce_bf * wt * torch.where(bf_chg, 2.0, 1.0)).sum()
            / (wt * bf_chg.float()).sum().clamp(min=16.0)
        )

        gr_t, gr_p = grp_ft.float(), grp_fp.float()
        gr_chg = (gr_t - gr_p).abs() > 0.5
        bce_gr = F.binary_cross_entropy_with_logits(o["groups"], gr_t, reduction="none")
        l_groups = (
            0.025
            * (bce_gr * wt * torch.where(gr_chg, 2.0, 1.0)).sum()
            / (wt * gr_chg.float()).sum().clamp(min=16.0)
        )

        bt_chg = (bits_ft - bits_fp).abs() > 0.5
        bce_bt = F.binary_cross_entropy_with_logits(
            o["bits"], bits_ft, reduction="none"
        )
        l_bits = (
            0.02
            * (bce_bt * wt * torch.where(bt_chg, 2.0, 1.0)).sum()
            / (wt * bt_chg.float()).sum().clamp(min=8.0)
        )

        rm = rf_ft[:, 0] > 0
        r_chg = rm & (
            (rf_fp[:, 0] <= 0)
            | ((rf_ft[:, 1:3].float() - rf_fp[:, 1:3].float()).abs().amax(-1) > 0.005)
        )
        rl_elem = F.smooth_l1_loss(
            o["rally"], rf_ft[:, 1:3].float(), beta=0.02, reduction="none"
        ).sum(-1)
        w_rl = whole * rm.float() * torch.where(r_chg, 2.0, 1.0)
        l_rally = (
            0.05 * (rl_elem * w_rl).sum() / (whole * r_chg.float()).sum().clamp(min=4.0)
        )
        return (
            l_pos,
            l_pos_copy,
            l_reg,
            l_type,
            l_inside,
            l_flags,
            l_display,
            l_cloak,
            l_alive,
            death_f1,
            l_buffs,
            l_groups,
            l_bits,
            l_rally,
        )

    @staticmethod
    def _gather_bag_u8(at_flat, bag_at, bag_id, n_rows, n_cols, total_bfr, dev):
        out = torch.zeros(n_rows, n_cols, device=dev, dtype=torch.uint8)
        if n_rows == 0 or len(bag_at) == 0:
            return out
        uq, inv = torch.unique(at_flat, return_inverse=True)
        u_pos = torch.full((total_bfr,), -1, device=dev, dtype=torch.long)
        u_pos[uq] = torch.arange(len(uq), device=dev)
        hit = u_pos[bag_at]
        ok = hit >= 0
        if ok.any():
            u_out = torch.zeros(len(uq), n_cols, device=dev, dtype=torch.uint8)
            u_out[hit[ok], bag_id[ok]] = 1
            return u_out[inv]
        return out

    def raw_losses(self, p, s, b, sc, bi, ui, ri, f_src, f_tgt):
        if not len(bi):
            z = p.sum() * 0
            logs = {
                k_: z
                for k_ in (
                    "pos",
                    "pos_copy",
                    "reg",
                    "type",
                    "inside",
                    "flags",
                    "display",
                    "cloak",
                    "alive",
                    "death_f1",
                    "buffs",
                    "groups",
                    "bits",
                    "rally",
                    "queues",
                    "abilities",
                    "pointers",
                )
            }
            if self.record_pos:
                empty = np.zeros(0, dtype=np.float32)
                self.pos_eval = {
                    k_: (empty, empty)
                    for k_ in (
                        "player moving",
                        "player still",
                        "player all",
                        "larva",
                        "critter",
                        "other neutral",
                    )
                }
            return z, logs
        B, F_, R = b["mask"].shape
        k = max(1, self.enc.tubelet)
        pos_scale = POS_SCALE * k
        fp = f_src[ui]
        ft = f_tgt[ui]
        own = b["owner"][bi, ri]
        rt = b["rtype"][bi, ft, ri]
        crit = (own == NEUTRAL) & torch.isin(rt, self.critter)
        larva = rt == self.larva
        x_sc = p[bi, ui, ri]
        x_ft, x_fp = b["x"][bi, ft, ri], b["x"][bi, fp, ri]
        y_ft, y_fp = b["y"][bi, ft, ri], b["y"][bi, fp, ri]
        n_ft, n_fp = b["num"][bi, ft, ri], b["num"][bi, fp, ri]
        alive_ft = b["alive"][bi, ft, ri]
        typ_ft, typ_fp = b["type"][bi, ft, ri], b["type"][bi, fp, ri]
        ins_ft, ins_fp = b["inside"][bi, ft, ri], b["inside"][bi, fp, ri]
        grp_ft, grp_fp = b["groups"][bi, ft, ri], b["groups"][bi, fp, ri]
        eng_ft, eng_fp = b["engaged_row"][bi, ft, ri], b["engaged_row"][bi, fp, ri]
        rf_ft, rf_fp = b["rally_first"][bi, ft, ri], b["rally_first"][bi, fp, ri]

        total_bfr = B * F_ * R
        key_ft = (bi * F_ + ft) * R + ri
        key_fp = (bi * F_ + fp) * R + ri
        buffs_ft_u8 = self._gather_bag_u8(
            key_ft, b["buff_at"], b["buff_id"], len(bi), N_BUFF_IDS, total_bfr, p.device
        )
        buffs_fp_u8 = self._gather_bag_u8(
            key_fp, b["buff_at"], b["buff_id"], len(bi), N_BUFF_IDS, total_bfr, p.device
        )

        (
            l_pos,
            l_pos_copy,
            l_reg,
            l_type,
            l_inside,
            l_flags,
            l_display,
            l_cloak,
            l_alive,
            death_f1,
            l_buffs,
            l_groups,
            l_bits,
            l_rally,
        ) = self._dense_raw_losses(
            x_sc,
            crit,
            larva,
            x_ft,
            x_fp,
            y_ft,
            y_fp,
            n_ft,
            n_fp,
            typ_ft,
            typ_fp,
            ins_ft,
            ins_fp,
            alive_ft,
            buffs_ft_u8,
            buffs_fp_u8,
            grp_ft,
            grp_fp,
            eng_ft,
            eng_fp,
            rf_ft,
            rf_fp,
            pos_scale,
        )
        logs = dict(
            pos=l_pos,
            pos_copy=l_pos_copy,
            reg=l_reg,
            type=l_type,
            inside=l_inside,
            flags=l_flags,
            display=l_display,
            cloak=l_cloak,
            alive=l_alive,
            death_f1=death_f1,
            buffs=l_buffs,
            groups=l_groups,
            bits=l_bits,
            rally=l_rally,
        )
        if self.record_pos:
            with torch.no_grad():
                pred_pos = self.head.reg(x_sc)[:, :2].float()
                tpos = torch.stack(
                    [(x_ft - x_fp) / pos_scale, (y_ft - y_fp) / pos_scale], -1
                )
                self.pos_eval = self._position_errors(
                    pred_pos,
                    tpos,
                    own,
                    crit,
                    larva,
                    alive_ft & (ins_fp == 0) & (ins_ft == 0),
                    pos_scale,
                    k,
                )
                if self.eval_cnts is not None:
                    ply_m = own != NEUTRAL
                    # Morph accuracy on player units
                    m_chg = ply_m & (typ_ft.long() != typ_fp.long())
                    if m_chg.any():
                        n_u = self.head.unit.out_features
                        pr_typ = (
                            self.head.unit(x_sc[m_chg])
                            .float()
                            .scatter_add(
                                1,
                                typ_fp[m_chg].long().clamp(0, n_u - 1)[:, None],
                                self.head.stay_type(x_sc[m_chg]).float(),
                            )
                            .argmax(-1)
                        )
                        self.eval_cnts["type_chg_ok"] += int(
                            (pr_typ == typ_ft[m_chg].long()).sum().item()
                        )
                        self.eval_cnts["type_chg_tot"] += int(m_chg.sum().item())
                    # Active-flag flip F1 and group flip F1 on player units
                    if ply_m.any():
                        xp = x_sc[ply_m]
                        f_sh = self.active_flag_bits
                        fp_r = n_fp[ply_m, I["flags"]].long() | (
                            n_fp[ply_m, I["flags2"]].long() << 8
                        )
                        ft_r = n_ft[ply_m, I["flags"]].long() | (
                            n_ft[ply_m, I["flags2"]].long() << 8
                        )
                        fl_p = ((fp_r[:, None] >> f_sh) & 1).float()
                        fl_t = ((ft_r[:, None] >> f_sh) & 1).float()
                        fl_pr = (
                            self.head.flags(xp)[:, f_sh].float()
                            + self.head.stay_flags(xp)[:, f_sh].float()
                            * (2.0 * fl_p - 1.0)
                        ) > 0.0
                        pr_f_chg = fl_pr.float() != fl_p
                        gt_f_chg = fl_t != fl_p
                        self.eval_cnts["flags_chg_tp"] += int(
                            (pr_f_chg & gt_f_chg & (fl_pr.float() == fl_t)).sum().item()
                        )
                        self.eval_cnts["flags_chg_pred"] += int(pr_f_chg.sum().item())
                        self.eval_cnts["flags_chg_gt"] += int(gt_f_chg.sum().item())

                        gp_p, gp_t = grp_fp[ply_m].float(), grp_ft[ply_m].float()
                        gp_pr = (
                            self.head.groups(xp).float()
                            + self.head.stay_groups(xp).float() * (2.0 * gp_p - 1.0)
                        ) > 0.0
                        pr_g_chg = gp_pr.float() != gp_p
                        gt_g_chg = gp_t != gp_p
                        self.eval_cnts["groups_chg_tp"] += int(
                            (pr_g_chg & gt_g_chg & (gp_pr.float() == gp_t)).sum().item()
                        )
                        self.eval_cnts["groups_chg_pred"] += int(pr_g_chg.sum().item())
                        self.eval_cnts["groups_chg_gt"] += int(gt_g_chg.sum().item())
        self._ptr_stats = None
        logs["queues"], logs["abilities"], logs["pointers"] = self._sampled(
            p, b, bi, ui, ri, f_src, f_tgt
        )
        if self._ptr_stats is not None:
            logs["ptr_acc"], logs["ptr_copy"] = self._ptr_stats
        total = (
            POS_BOOST * logs["pos"]
            + logs["reg"]
            + logs["type"]
            + logs["inside"]
            + logs["flags"]
            + logs["display"]
            + logs["cloak"]
            + logs["alive"]
            + logs["queues"]
            + logs["buffs"]
            + logs["groups"]
            + logs["bits"]
            + logs["rally"]
            + logs["abilities"]
            + logs["pointers"]
        )
        return total, logs

    def _position_errors(self, pred, tpos, own, crit, larva, keep, pos_scale, k):
        """For evaluation: {kind of row: (model errors, copy errors)}, one per row, map
        units across the predicted k-loop tubelet."""
        step = tpos.norm(dim=-1) * pos_scale
        err = (pred - tpos).norm(dim=-1) * pos_scale
        neutral = own == NEUTRAL
        player = ~neutral & ~larva
        moving = step > 0.003 * k
        kinds = {
            "player moving": player & moving,
            "player still": player & ~moving,
            "player all": player,
            "larva": larva & ~neutral,
            "critter": crit,
            "other neutral": neutral & ~crit,
        }
        return {
            k_: (err[v & keep].cpu().numpy(), step[v & keep].cpu().numpy())
            for k_, v in kinds.items()
        }

    @staticmethod
    def _gather_slots(b, at, total_bfr, dev):
        """Gather [N, QUEUE + PROD] slot abilities and [N, QUEUE + PROD, 4] slot
        regressions for flat indices `at` (handling duplicate indices cleanly)."""
        if "slot_abil" in b and "slot_reg" in b:
            return (
                b["slot_abil"].reshape(-1, QUEUE + PROD)[at],
                b["slot_reg"].reshape(-1, QUEUE + PROD, 4)[at],
            )
        uq, inv = torch.unique(at, return_inverse=True)
        u_pos = torch.full((total_bfr,), -1, device=dev)
        u_pos[uq] = torch.arange(len(uq), device=dev)
        u_abil = torch.zeros(
            (len(uq), QUEUE + PROD), dtype=torch.long, device=dev
        )
        u_reg = torch.zeros(
            (len(uq), QUEUE + PROD, 4), dtype=torch.float32, device=dev
        )
        if len(b["ord_at"]):
            oq = b["ord_in_q"] if "ord_in_q" in b else (b["ord_slot"] < QUEUE)
            o_hit = u_pos[b["ord_at"][oq]]
            o_ok = o_hit >= 0
            if o_ok.any():
                o_sl = b["ord_slot"][oq][o_ok]
                u_abil[o_hit[o_ok], o_sl] = b["ord_abil"][oq][o_ok]
                u_reg[o_hit[o_ok], o_sl] = b["ord_feat"][oq][o_ok]
        if len(b["prod_at"]):
            pq = b["prod_in_q"] if "prod_in_q" in b else (b["prod_slot"] < PROD)
            p_hit = u_pos[b["prod_at"][pq]]
            p_ok = p_hit >= 0
            if p_ok.any():
                p_sl = QUEUE + b["prod_slot"][pq][p_ok]
                u_abil[p_hit[p_ok], p_sl] = b["prod_abil"][pq][p_ok]
                u_reg[p_hit[p_ok], p_sl, 3] = b["prod_feat"][pq][p_ok, 0]
        return u_abil[inv], u_reg[inv]

    def _sampled(self, p, b, bi, ui, ri, f_src, f_tgt, cap=4096, ptr_cap=1024):
        """The large outputs, on a sample of the scored tokens, rows whose queue,
        engaged target, or available-ability set changes across (f_src, f_tgt] first:
        order and production slots, available abilities, and pointers over rows + -1."""
        B, F_, R = b["mask"].shape
        k = max(1, self.enc.tubelet)
        af_idx = f_src[:, None] + torch.arange(k, device=p.device)[None, :]
        rc_src = b["row_changed"] if "row_changed" in b else b["queue_changed"]
        ch = rc_src[:, af_idx].any(dim=2)[bi, ui, ri]
        sel_ch = torch.nonzero(ch, as_tuple=True)[0]
        rest = torch.nonzero(~ch, as_tuple=True)[0]
        ch_cap = (cap * 3) // 4
        if len(sel_ch) > ch_cap:
            sel_ch = sel_ch[torch.randperm(len(sel_ch), device=p.device)[:ch_cap]]
        take = rest[
            torch.randperm(len(rest), device=p.device)[: max(0, cap - len(sel_ch))]
        ]
        sel = torch.cat([sel_ch, take])
        if not len(sel):
            z = p.sum() * 0
            return z, z, z
        x = p[bi[sel], ui[sel], ri[sel]]
        ft_sel = f_tgt[ui[sel]]
        fanc_sel = f_src[ui[sel]]
        at = (bi[sel] * F_ + ft_sel) * R + ri[sel]
        at_q_anc = (bi[sel] * F_ + fanc_sel) * R + ri[sel]
        anc_abil, anc_reg = self._gather_slots(b, at_q_anc, B * F_ * R, p.device)
        t_abil, t_reg = self._gather_slots(b, at, B * F_ * R, p.device)
        abil, reg = self.head.slots(x, anc_abil=anc_abil, anc_reg=anc_reg)
        la = F.cross_entropy(abil.float().flatten(0, 1), t_abil.flatten())
        m = (t_abil > 0)[..., None].float()
        queues = la + 0.25 * (
            F.smooth_l1_loss(reg.float(), t_reg, beta=0.02, reduction="none") * m
        ).sum() / m.sum().clamp(min=1)
        # available abilities: f_src residual + F1-calibrated event-normalized BCE
        total_bfr = B * F_ * R
        t_ab = self._gather_bag_u8(
            at, b["ab_at"], b["ab_id"], len(sel), N_RAW_ABILITY_IDS, total_bfr, p.device
        ).float()
        src_ab = self._gather_bag_u8(
            at_q_anc,
            b["ab_at"],
            b["ab_id"],
            len(sel),
            N_RAW_ABILITY_IDS,
            total_bfr,
            p.device,
        ).float()
        ab_logits = self.head.predict_abilities(x, src_ab=src_ab)
        ab_chg = (t_ab - src_ab).abs() > 0.5
        bce_ab = F.binary_cross_entropy_with_logits(ab_logits, t_ab, reduction="none")
        abilities = (
            0.025
            * (bce_ab * torch.where(ab_chg, 2.0, 1.0)).sum()
            / ab_chg.float().sum().clamp(min=32.0)
        )
        # pointers: scored over unit rows 0..R-1 plus column R ("no target (-1)"),
        # conditioned strictly on the source tubelet's anchor frame fanc
        s2 = sel[torch.randperm(len(sel), device=p.device)[:ptr_cap]]
        bs, us, rs = bi[s2], ui[s2], ri[s2]
        fts = f_tgt[us]
        fanc = f_src[us]
        rows = self._rows[bs, us]
        at2 = (bs * F_ + fts) * R + rs
        at_anc = (bs * F_ + fanc) * R + rs
        now = torch.cat(
            [
                b["engaged_row"].reshape(-1)[at_anc][:, None],
                b["parent_row"].reshape(-1)[at_anc][:, None],
                b["ord_row"].reshape(-1, QUEUE)[at_anc],
            ],
            1,
        ).long()
        tgt = torch.cat(
            [
                b["engaged_row"].reshape(-1)[at2][:, None],
                b["parent_row"].reshape(-1)[at2][:, None],
                b["ord_row"].reshape(-1, QUEUE)[at2],
            ],
            1,
        ).long()
        xy_q = torch.stack([b["x"][bs, fanc, rs], b["y"][bs, fanc, rs]], -1)
        xy_k = torch.stack([b["x"][bs, fanc], b["y"][bs, fanc]], -1)
        logits = self.head.pointers(
            p[bs, us, rs], rows, xy_q=xy_q, xy_k=xy_k, now_ptr=now
        )
        # a target row not live in fanc can't be pointed at; column R (-1) is always valid
        if R > 0:
            live = b["mask"][bs, fanc]
            logits = torch.cat(
                [logits[..., :R].masked_fill(~live[:, None], -1e4), logits[..., R:]],
                dim=-1,
            ).float()
            valid_ptr = (tgt < 0) | (
                (tgt < R) & live.gather(1, tgt.clamp(0, R - 1))
            )
        else:
            logits = logits.float()
            valid_ptr = tgt < 0
        tgt_cls = torch.where(tgt >= 0, tgt, R)
        now_cls = torch.where((now >= 0) & (now < R), now, R)
        active_ptr = (now >= 0) | (tgt >= 0)
        if valid_ptr.any():
            ce_p = F.cross_entropy(
                logits[valid_ptr], tgt_cls[valid_ptr], reduction="none"
            )
            w_p = torch.where(active_ptr[valid_ptr], 1.0, 0.05)
            pointers = (ce_p * w_p).sum() / w_p.sum().clamp(min=1.0)
        else:
            pointers = p.sum() * 0
        # accuracy vs copy baseline across all active/acquiring/clearing pointer slots
        eval_ptr = valid_ptr & active_ptr
        with torch.no_grad():
            self._ptr_stats = (
                (
                    (logits.argmax(-1) == tgt_cls)[eval_ptr].float().mean(),
                    (now_cls == tgt_cls)[eval_ptr].float().mean(),
                )
                if eval_ptr.any()
                else None
            )
        return queues, abilities, pointers

    def birth_losses(self, out, act, b, f_src, f_tgt):
        B, F_, R = b["mask"].shape
        M_pred = out["ent"].shape[1]
        k = max(1, self.enc.tubelet)
        dev = out["ent"].device
        if M_pred == 0:
            z = out["ent"].sum() * 0
            return z, dict(birth_count=z, birth_cnt_f1=z, birth_real=z, birth_fields=z)
        af_idx = f_src[:, None] + torch.arange(k, device=dev)[None, :]
        n_b = b["birth_count"][:, af_idx].sum(-1)  # [B, M_pred]: births in (f_src, f_tgt]
        glob, rows, cells = out["glob"], out["ent"], out["cells"]
        rmask = b["mask"][:, f_src]
        anc_xy = torch.stack([b["x"][:, f_src], b["y"][:, f_src]], -1).float()
        anc_txy_all = (
            (anc_xy - b["center"][:, None, None].float())
            / b["half"][:, None, None].float()
        ) * rmask[..., None].float()
        cell_txy_all = self.enc.anchors(b).float() / b["half"][:, None].float()
        act_win = act[:, af_idx].flatten(2, 3)
        amask_win = b["act_mask"][:, af_idx].flatten(2, 3)
        aplaced_win = (
            b["act_placed"][:, af_idx] & b["act_mask"][:, af_idx]
        ).flatten(2, 3)
        act_txy_win = (
            b["act_xy"][:, af_idx].float().flatten(2, 3)
            / b["half"][:, None, None].float()
        ) * aplaced_win[..., None].float()

        nb_np = n_b.cpu().numpy()
        fb_np, ff_np = np.nonzero(nb_np >= 0)
        f0_val = int(f_src[0].item())
        flast_val = int(f_tgt[-1].item())
        has_unpar_np = np.zeros((B, M_pred), dtype=bool)
        if int(nb_np.max()) > 0 and len(b["birth_b"]):
            bb_np = b["birth_b"].cpu().numpy()
            bf_np = b["birth_f"].cpu().numpy()
            br_np = b["birth_r"].cpu().numpy()
            bp_np = b["birth_p"].cpu().numpy()
            birth_map = {}
            for bi_v, fi_v, ri_v, pi_v in zip(bb_np, bf_np, br_np, bp_np):
                fi_int = int(fi_v)
                if f0_val < fi_int <= flast_val:
                    b_int = int(bi_v)
                    j_step = (fi_int - 1 - f0_val) // k
                    p_int = int(pi_v)
                    birth_map.setdefault((b_int, j_step), []).append(
                        (int(ri_v), p_int, fi_int)
                    )
                    if p_int < 0:
                        has_unpar_np[b_int, j_step] = True
        else:
            birth_map = {}

        has_unpar_bm = torch.from_numpy(has_unpar_np).to(dev)
        cnt_logits = self.births.predict_count(
            glob,
            rows,
            rmask,
            acts=act_win,
            amask=amask_win,
            anc_txy=anc_txy_all,
            aplaced=aplaced_win,
            act_txy=act_txy_win,
        ).float()
        cnt_tgt = n_b.clamp(max=MAX_BIRTHS - 1)
        b_tube_w = torch.where(
            cnt_tgt == 0, 0.25, torch.where(has_unpar_bm, 2.0, 1.0)
        )
        ce_bc = F.cross_entropy(
            cnt_logits.flatten(0, 1), cnt_tgt.flatten(), reduction="none"
        )
        logs = dict(
            birth_count=(ce_bc * b_tube_w.flatten()).sum()
            / b_tube_w.sum().clamp(min=1.0)
        )
        with torch.no_grad():
            pr_cnt = cnt_logits.detach().argmax(-1)
            has_b_p = (pr_cnt > 0).float()
            has_b_t = (cnt_tgt > 0).float()
            logs["birth_cnt_f1"] = (
                2.0
                * (has_b_p * has_b_t).sum()
                / (has_b_p.sum() + has_b_t.sum()).clamp(min=1.0)
            )
            if self.record_pos and self.eval_cnts is not None and has_unpar_bm.any():
                self.eval_cnts["birth_unpar_ok"] += int(
                    ((pr_cnt > 0) & has_unpar_bm).sum().item()
                )
                self.eval_cnts["birth_unpar_tot"] += int(has_unpar_bm.sum().item())
        if not len(fb_np):
            z = out["ent"].sum() * 0
            logs["birth_real"], logs["birth_fields"] = z, z
            return logs["birth_count"], logs
        fb = torch.from_numpy(fb_np).to(dev)
        ff = torch.from_numpy(ff_np).to(dev)
        groups = {}
        for idx_i, c in enumerate(nb_np[fb_np, ff_np].tolist()):
            groups.setdefault(min(c, MAX_BIRTHS - 1), []).append(idx_i)

        total_real, total_fields, frames = 0.0, 0.0, 0
        for c, idx in groups.items():
            idx_t = torch.tensor(idx, device=dev)
            sb, sf = fb[idx_t], ff[idx_t]
            x, real, pos, parent, owner_logits, init_stat = self.births(
                glob[sb, sf],
                rows[sb, sf],
                rmask[sb, sf],
                cells[sb, sf],
                anc_txy_all[sb, sf],
                cell_txy_all[sb],
                act_win[sb, sf],
                amask_win[sb, sf],
                aplaced_win[sb, sf],
                act_txy_win[sb, sf],
                c + MARGIN,
                need_fields=bool(c),
                grad_ckpt=self.enc.grad_ckpt,
            )
            target = torch.zeros_like(real)
            if c:
                head_type = self.head.unit(x)
                vi_l, m_lens, rr_pad, par_pad, fi_pad = [], [], [], [], []
                for i, g_idx in enumerate(idx):
                    entries = birth_map.get(
                        (int(fb_np[g_idx]), int(ff_np[g_idx])), ()
                    )[:c]
                    if not entries:
                        continue
                    m_i = len(entries)
                    vi_l.append(i)
                    m_lens.append(m_i)
                    rr_row = [e[0] for e in entries] + [entries[-1][0]] * (c - m_i)
                    par_row = [e[1] for e in entries] + [entries[-1][1]] * (c - m_i)
                    fi_row = [e[2] for e in entries] + [entries[-1][2]] * (c - m_i)
                    rr_pad.append(rr_row)
                    par_pad.append(par_row)
                    fi_pad.append(fi_row)
                if vi_l:
                    vi = torch.tensor(vi_l, device=dev, dtype=torch.long)
                    vb, vf = sb[vi], sf[vi]
                    vf_tgt = f_tgt[vf]
                    rr_t = torch.tensor(rr_pad, device=dev, dtype=torch.long)
                    par_t = torch.tensor(par_pad, device=dev, dtype=torch.long)
                    fi_b_t = torch.tensor(fi_pad, device=dev, dtype=torch.long)
                    vf_obs = torch.where(
                        b["mask"][vb[:, None], vf_tgt[:, None], rr_t],
                        vf_tgt[:, None],
                        fi_b_t,
                    )
                    tt = b["type"][vb[:, None], vf_obs, rr_t].long()
                    own_t = b["owner"][vb[:, None], rr_t].clamp(0, 1).long()
                    txy = (
                        torch.stack(
                            [
                                b["x"][vb[:, None], vf_obs, rr_t],
                                b["y"][vb[:, None], vf_obs, rr_t],
                            ],
                            -1,
                        )
                        - b["center"][vb].view(-1, 1, 2)
                    ) / b["half"][vb].view(-1, 1, 2)
                    if R > 0:
                        par_valid = (
                            (par_t >= 0)
                            & (par_t < R)
                            & rmask[vb[:, None], vf[:, None], par_t.clamp(0, R - 1)]
                        )
                    else:
                        par_valid = par_t < 0
                    par_all = torch.where(par_valid, par_t, parent.shape[-1] - 1)
                    lt = F.log_softmax(head_type[vi].float(), -1)
                    lt_tt = lt.gather(
                        2, tt[:, None, :].expand(-1, c + MARGIN, -1)
                    )
                    lp = F.log_softmax(parent[vi].float(), -1)
                    lp_par = lp.gather(
                        2, par_all[:, None, :].expand(-1, c + MARGIN, -1)
                    )
                    lo = F.log_softmax(owner_logits[vi].float(), -1)
                    lo_own = lo.gather(
                        2, own_t[:, None, :].expand(-1, c + MARGIN, -1)
                    )
                    cost_np = (
                        (
                            torch.cdist(pos[vi].float(), txy, p=1)
                            - lt_tt
                            - lp_par
                            - lo_own
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    fi_l, qi_l, ti_l, inv_m_l = [], [], [], []
                    for j, m_i in enumerate(m_lens):
                        qj, tj = linear_sum_assignment(cost_np[j, :, :m_i])
                        fi_l.extend([j] * m_i)
                        qi_l.extend(qj.tolist())
                        ti_l.extend(tj.tolist())
                        inv_m_l.extend([1.0 / m_i] * m_i)
                    fi_t = torch.tensor(fi_l, device=dev, dtype=torch.long)
                    qi_t = torch.tensor(qi_l, device=dev, dtype=torch.long)
                    ti_t = torch.tensor(ti_l, device=dev, dtype=torch.long)
                    inv_m_t = torch.tensor(inv_m_l, device=dev, dtype=torch.float32)
                    orig_i = vi[fi_t]
                    target[orig_i, qi_t] = 1.0
                    rr_m = rr_t[fi_t, ti_t]
                    vf_obs_m = vf_obs[fi_t, ti_t]
                    w = torch.where(
                        b["rtype"][vb[fi_t], vf_obs_m, rr_m] == self.larva,
                        DISCOUNT,
                        1.0,
                    )  # a larva's position takes the discount, its birth doesn't
                    lpos_p = (
                        F.smooth_l1_loss(
                            pos[orig_i, qi_t].float(), txy[fi_t, ti_t], reduction="none"
                        ).sum(-1)
                        * w
                    )
                    ce_type_p = F.cross_entropy(
                        head_type[orig_i, qi_t].float(),
                        tt[fi_t, ti_t],
                        reduction="none",
                    )
                    par = par_all[fi_t, ti_t]
                    ce_par_p = F.cross_entropy(
                        parent[orig_i, qi_t].float(), par, reduction="none"
                    )
                    ce_own_p = F.cross_entropy(
                        owner_logits[orig_i, qi_t].float(),
                        own_t[fi_t, ti_t],
                        reduction="none",
                    )
                    # Light confirmation on newborn initial stats and active unit flags
                    x_m = x[orig_i, qi_t]
                    n_nb = b["num"][vb[fi_t], vf_obs_m, rr_m].float()
                    hpm_nb = n_nb[:, I["hp_max"]]
                    shm_nb = n_nb[:, I["shield_max"]]
                    stat_tgt = torch.stack(
                        [
                            (n_nb[:, I["hp"]] / hpm_nb.clamp(min=1.0)) * (hpm_nb > 0),
                            (n_nb[:, I["shield"]] / shm_nb.clamp(min=1.0))
                            * (shm_nb > 0),
                            n_nb[:, I["energy"]] / 200.0,
                            n_nb[:, I["build"]],
                        ],
                        dim=-1,
                    )
                    l_stat_p = F.smooth_l1_loss(
                        init_stat[orig_i, qi_t].float(), stat_tgt, reduction="none"
                    ).mean(-1)
                    flags_nb = n_nb[:, I["flags"]].long() | (
                        n_nb[:, I["flags2"]].long() << 8
                    )
                    act_fb = self.active_flag_bits
                    fbits_nb = ((flags_nb[:, None] >> act_fb) & 1).float()
                    l_flags_nb = F.binary_cross_entropy_with_logits(
                        self.head.flags(x_m)[:, act_fb].float(),
                        fbits_nb,
                        reduction="none",
                    ).mean(-1)
                    l_init_p = 0.25 * (l_stat_p + l_flags_nb)
                    w_bf = torch.where(~par_valid[fi_t, ti_t], 1.75, 1.0)
                    total_fields = (
                        total_fields
                        + (
                            (lpos_p + ce_type_p + ce_par_p + ce_own_p + l_init_p)
                            * w_bf
                            * inv_m_t
                        ).sum()
                    )
                    frames += len(vi_l)
            w_rl_bm = torch.where(has_unpar_bm[sb, sf], 1.5, 1.0)[:, None]
            total_real = total_real + (
                F.binary_cross_entropy_with_logits(
                    real.float(), target, reduction="none"
                )
                * w_rl_bm
            ).sum()
        logs["birth_real"] = total_real / max(1, len(fb_np)) / (1 + MARGIN)
        logs["birth_fields"] = (
            total_fields / max(1, frames) if frames else out["ent"].sum() * 0
        )
        return logs["birth_count"] + logs["birth_real"] + logs["birth_fields"], logs

    @staticmethod
    def _group_eff_spawns(ty_arr, ow_arr, vw_arr, xy_arr, age_arr):
        """Group newly spawned per-view effect entries (0 <= age < k) at one (b, m)
        target frame into canonical AoE objects (collapsing contiguous collinear
        LurkerSpines / ThermalLances ray dots and merging across views 0 & 1)."""
        by_v = {0: [], 1: []}
        for ty, ow, vw, (x, y), ag in zip(ty_arr, ow_arr, vw_arr, xy_arr, age_arr):
            by_v[1 if vw == 1 else 0].append(
                (int(ty), int(ow), float(x), float(y), int(round(ag)))
            )
        per_v_objs = []
        for vw, rows in by_v.items():
            i, n = 0, len(rows)
            while i < n:
                ty, ow, x0, y0, ag = rows[i]
                if ty in (5, 12):
                    exp_step = 0.25 if ty == 5 else 1.0
                    j = i + 1
                    dx0, dy0 = 0.0, 0.0
                    while (
                        j < n
                        and (j - i) < MAX_EFF_DOTS
                        and rows[j][0] == ty
                        and rows[j][1] == ow
                        and abs(rows[j][4] - ag) <= 1
                    ):
                        dx = rows[j][2] - rows[j - 1][2]
                        dy = rows[j][3] - rows[j - 1][3]
                        step = math.hypot(dx, dy)
                        if abs(step - exp_step) > 0.05:
                            break
                        if j == i + 1:
                            dx0, dy0 = dx, dy
                        elif abs(dx - dx0) > 0.03 or abs(dy - dy0) > 0.03:
                            break
                        j += 1
                    x1, y1 = rows[j - 1][2], rows[j - 1][3]
                    per_v_objs.append((ty, ow, j - i, x0, y0, x1, y1, ag, vw))
                    i = j
                else:
                    per_v_objs.append((ty, ow, 1, x0, y0, x0, y0, ag, vw))
                    i += 1
        merged = {}
        for ty, ow, nd, x0, y0, x1, y1, ag, vw in per_v_objs:
            key = (ty, ow, nd, round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2))
            rec = merged.get(key)
            if rec is None:
                merged[key] = [ty, ow, nd, x0, y0, x1, y1, ag, [0.0, 0.0]]
                merged[key][8][vw] = 1.0
            else:
                rec[8][vw] = 1.0
                if ag < rec[7]:
                    rec[7] = ag
        return list(merged.values())

    def effect_losses(self, out, act, b, f_src, f_tgt):
        """Discrete AoE effect transitions across (f_src, f_tgt]:
        1) EffPersistHead on active f_src effect tokens (persistence/expiry + moving dxy).
        2) EffSpawnHead on newly spawned AoE circles & rays (0 <= age_tgt < k), attending
           over active f_src precursor effects alongside rows, cells, globals & actions."""
        B, F_, R = b["mask"].shape
        M_pred = out["ent"].shape[1]
        k = max(1, self.enc.tubelet)
        dev = out["ent"].device
        if M_pred == 0:
            z = out["ent"].sum() * 0
            return z, dict(
                eff_persist=z,
                eff_persist_f1=z,
                eff_expire_f1=z,
                eff_spawn_cnt=z,
                eff_spawn_cnt_f1=z,
                eff_spawn_real=z,
                eff_spawn_fields=z,
            )

        # 1) Continuing / expiring active effect tokens from f_src -> f_tgt
        src_slot = out["eff_src_slot"]
        bi_s, mi_s, ui_s = torch.nonzero(src_slot >= 0, as_tuple=True)
        n_act = int(bi_s.numel())
        pos_scale = POS_SCALE * k
        pers_tp = pers_pred = pers_gt = exp_tp = exp_pred = exp_gt = 0

        if n_act > 0:
            e_act = out["eff"][bi_s, mi_s, ui_s]
            xs_slot = src_slot[bi_s, mi_s, ui_s]
            fs_act = f_src[mi_s]
            ty_s_t = b["eff_type"][bi_s, fs_act, xs_slot]
            ag_s_t = b["eff_age"][bi_s, fs_act, xs_slot].float()
            p_logit, p_dxy = self.eff_persist(e_act, ty_s=ty_s_t, age_s=ag_s_t)

            bm_s_np = (bi_s * M_pred + mi_s).cpu().numpy()
            ty_s_np = ty_s_t.cpu().numpy()
            ow_s_np = b["eff_owner"][bi_s, fs_act, xs_slot].cpu().numpy()
            vw_s_np = (
                b["eff_view"][bi_s, fs_act, xs_slot].cpu().numpy()
                if "eff_view" in b
                else np.zeros(n_act, dtype=np.int64)
            )
            ag_s_np = np.rint(ag_s_t.cpu().numpy()).astype(np.int64)
            xy_s_t = b["eff_xy"][bi_s, fs_act, xs_slot].float()
            xy_s_np = xy_s_t.cpu().numpy()

            buckets = {}
            for idx_s in range(n_act):
                bkey = (
                    int(bm_s_np[idx_s]),
                    int(vw_s_np[idx_s]),
                    int(ow_s_np[idx_s]),
                    int(ty_s_np[idx_s]),
                    int(ag_s_np[idx_s]),
                )
                buckets.setdefault(bkey, []).append(idx_s)

            tgt_persist_np = np.zeros(n_act, dtype=np.float32)
            tgt_xy_np = xy_s_np.copy()

            em_tgt = b["eff_mask"][:, f_tgt]
            ea_tgt = b["eff_age"][:, f_tgt]
            bi_c, mi_c, xi_c = torch.nonzero(em_tgt & (ea_tgt >= k), as_tuple=True)
            if len(bi_c):
                ft_c = f_tgt[mi_c]
                bm_c_np = (bi_c * M_pred + mi_c).cpu().numpy()
                ty_c_np = b["eff_type"][bi_c, ft_c, xi_c].cpu().numpy()
                ow_c_np = b["eff_owner"][bi_c, ft_c, xi_c].cpu().numpy()
                vw_c_np = (
                    b["eff_view"][bi_c, ft_c, xi_c].cpu().numpy()
                    if "eff_view" in b
                    else np.zeros(len(bi_c), dtype=np.int64)
                )
                ag_c_np = (
                    np.rint(b["eff_age"][bi_c, ft_c, xi_c].cpu().numpy()).astype(np.int64)
                    - k
                )
                xy_c_np = b["eff_xy"][bi_c, ft_c, xi_c].float().cpu().numpy()

                for jc in range(len(bm_c_np)):
                    bkey = (
                        int(bm_c_np[jc]),
                        int(vw_c_np[jc]),
                        int(ow_c_np[jc]),
                        int(ty_c_np[jc]),
                        int(ag_c_np[jc]),
                    )
                    cand = buckets.get(bkey)
                    if not cand:
                        continue
                    xt, yt = float(xy_c_np[jc, 0]), float(xy_c_np[jc, 1])
                    best_pos, best_d = -1, 6.0
                    for pos_c, idx_s in enumerate(cand):
                        d_m = abs(float(xy_s_np[idx_s, 0]) - xt) + abs(
                            float(xy_s_np[idx_s, 1]) - yt
                        )
                        if d_m <= best_d:
                            best_d, best_pos = d_m, pos_c
                    if best_pos >= 0:
                        matched_s = cand.pop(best_pos)
                        tgt_persist_np[matched_s] = 1.0
                        tgt_xy_np[matched_s] = xy_c_np[jc]

            tgt_persist = torch.from_numpy(tgt_persist_np).to(dev)
            tgt_xy = torch.from_numpy(tgt_xy_np).to(dev)
            is_exp = tgt_persist < 0.5
            bce_elem = F.binary_cross_entropy_with_logits(
                p_logit.float(), tgt_persist, reduction="none"
            )
            denom_exp = is_exp.float().sum().clamp(min=max(4.0, 0.10 * n_act))
            bce_p = 0.15 * (bce_elem * torch.where(is_exp, 2.0, 1.0)).sum() / denom_exp
            m_pers = tgt_persist > 0.5
            if m_pers.any():
                tgt_dxy = (tgt_xy[m_pers] - xy_s_t[m_pers]) / pos_scale
                l_edxy = F.smooth_l1_loss(
                    p_dxy[m_pers].float(), tgt_dxy, beta=0.02, reduction="none"
                ).sum(-1).sum() / max(4, int(m_pers.sum().item()))
            else:
                l_edxy = p_dxy.sum() * 0.0
            l_eff_persist = bce_p + 2.0 * l_edxy

            with torch.no_grad():
                pr_p = (p_logit.detach() > 0.0).float()
                pers_tp = int((pr_p * tgt_persist).sum().item())
                pers_pred = int(pr_p.sum().item())
                pers_gt = int(tgt_persist.sum().item())
                pr_exp = 1.0 - pr_p
                tg_exp = 1.0 - tgt_persist
                exp_tp = int((pr_exp * tg_exp).sum().item())
                exp_pred = int(pr_exp.sum().item())
                exp_gt = int(tg_exp.sum().item())
                eff_persist_f1 = torch.tensor(
                    2.0 * pers_tp / max(1, pers_pred + pers_gt), device=dev
                )
                eff_expire_f1 = torch.tensor(
                    2.0 * exp_tp / max(1, exp_pred + exp_gt), device=dev
                )
        else:
            l_eff_persist = (
                out["eff"].sum()
                + sum(p_.sum() for p_ in self.eff_persist.parameters())
            ) * 0.0
            eff_persist_f1 = torch.zeros((), device=dev)
            eff_expire_f1 = torch.zeros((), device=dev)

        # 2) Newly spawned AoE circles & collinear rays (0 <= age_tgt < k at f_tgt)
        em_tgt = b["eff_mask"][:, f_tgt]
        ea_tgt = b["eff_age"][:, f_tgt]
        bi_n, mi_n, xi_n = torch.nonzero(em_tgt & (ea_tgt < k), as_tuple=True)
        n_sp_np = np.zeros((B, M_pred), dtype=np.int64)
        has_spell_np = np.zeros((B, M_pred), dtype=bool)
        spawn_map = {}
        if len(bi_n):
            ft_n = f_tgt[mi_n]
            bi_n_np = bi_n.cpu().numpy()
            mi_n_np = mi_n.cpu().numpy()
            ty_n_np = b["eff_type"][bi_n, ft_n, xi_n].cpu().numpy()
            ow_n_np = b["eff_owner"][bi_n, ft_n, xi_n].cpu().numpy()
            vw_n_np = (
                b["eff_view"][bi_n, ft_n, xi_n].cpu().numpy()
                if "eff_view" in b
                else np.zeros(len(bi_n), dtype=np.int64)
            )
            xy_n_np = b["eff_xy"][bi_n, ft_n, xi_n].float().cpu().numpy()
            ag_n_np = ea_tgt[bi_n, mi_n, xi_n].float().cpu().numpy()

            raw_bm = collections.defaultdict(lambda: ([], [], [], [], []))
            for idx_e in range(len(bi_n_np)):
                g_l = raw_bm[(int(bi_n_np[idx_e]), int(mi_n_np[idx_e]))]
                g_l[0].append(ty_n_np[idx_e])
                g_l[1].append(ow_n_np[idx_e])
                g_l[2].append(vw_n_np[idx_e])
                g_l[3].append(xy_n_np[idx_e])
                g_l[4].append(ag_n_np[idx_e])
            for (b_v, m_v), (tys, ows, vws, xys, ags) in raw_bm.items():
                objs = self._group_eff_spawns(tys, ows, vws, xys, ags)
                if objs:
                    spawn_map[(b_v, m_v)] = objs
                    n_sp_np[b_v, m_v] = len(objs)
                    if any(o_[0] not in (5, 12) for o_ in objs):
                        has_spell_np[b_v, m_v] = True

        n_sp = torch.from_numpy(n_sp_np).to(dev)
        has_spell_bm = torch.from_numpy(has_spell_np).to(dev)
        af_idx = f_src[:, None] + torch.arange(k, device=dev)[None, :]
        glob, rows, cells = out["glob"], out["ent"], out["cells"]
        rmask = b["mask"][:, f_src]
        anc_xy = torch.stack([b["x"][:, f_src], b["y"][:, f_src]], -1).float()
        anc_txy_all = (
            (anc_xy - b["center"][:, None, None].float())
            / b["half"][:, None, None].float()
        ) * rmask[..., None].float()
        cell_txy_all = self.enc.anchors(b).float() / b["half"][:, None].float()
        act_win = act[:, af_idx].flatten(2, 3)
        amask_win = b["act_mask"][:, af_idx].flatten(2, 3)
        aplaced_win = (
            b["act_placed"][:, af_idx] & b["act_mask"][:, af_idx]
        ).flatten(2, 3)
        act_txy_win = (
            b["act_xy"][:, af_idx].float().flatten(2, 3)
            / b["half"][:, None, None].float()
        ) * aplaced_win[..., None].float()

        effs_all = out["eff"]
        emask_all = src_slot >= 0
        X_e = src_slot.shape[2]
        if X_e > 0 and b["eff_mask"].shape[2] > 0:
            sl_c = src_slot.clamp(min=0)
            bi_e = torch.arange(B, device=dev)[:, None, None].expand(B, M_pred, X_e)
            fs_e = f_src[None, :, None].expand(B, M_pred, X_e)
            eff_xy_raw = b["eff_xy"][bi_e, fs_e, sl_c].float() * emask_all[
                ..., None
            ].float()
            eff_txy_all = eff_xy_raw / b["half"][:, None, None].float()
            eff_ty_all = (
                b["eff_type"][bi_e, fs_e, sl_c]
                .long()
                .clamp(0, N_EFF_SPAWN_TYPES - 1)
                * emask_all.long()
            )
            eff_age_all = b["eff_age"][bi_e, fs_e, sl_c].float() * emask_all.float()
        else:
            eff_txy_all = torch.zeros(B, M_pred, X_e, 2, device=dev)
            eff_ty_all = torch.zeros(B, M_pred, X_e, dtype=torch.long, device=dev)
            eff_age_all = torch.zeros(B, M_pred, X_e, device=dev)

        cnt_logits = self.eff_spawns.predict_count(
            glob,
            rows,
            rmask,
            acts=act_win,
            amask=amask_win,
            anc_txy=anc_txy_all,
            aplaced=aplaced_win,
            act_txy=act_txy_win,
            effs=effs_all,
            emask=emask_all,
            eff_txy=eff_txy_all,
            eff_ty=eff_ty_all,
            eff_age=eff_age_all,
        ).float()
        cnt_tgt = n_sp.clamp(max=MAX_EFF_SPAWNS - 1)
        sp_tube_w = torch.where(
            cnt_tgt == 0, 0.20, torch.where(has_spell_bm, 1.80, 0.50)
        )
        ce_sc = F.cross_entropy(
            cnt_logits.flatten(0, 1), cnt_tgt.flatten(), reduction="none"
        )
        l_sp_cnt = (ce_sc * sp_tube_w.flatten()).sum() / sp_tube_w.sum().clamp(min=1.0)
        with torch.no_grad():
            pr_sp_cnt = cnt_logits.detach().argmax(-1)
            has_sp_p = (pr_sp_cnt > 0).float()
            has_sp_t = (cnt_tgt > 0).float()
            sp_cnt_tp = int((has_sp_p * has_sp_t).sum().item())
            sp_cnt_pred = int(has_sp_p.sum().item())
            sp_cnt_gt = int(has_sp_t.sum().item())
            eff_spawn_cnt_f1 = torch.tensor(
                2.0 * sp_cnt_tp / max(1, sp_cnt_pred + sp_cnt_gt), device=dev
            )
            sp_spell_tube_ok = int(((pr_sp_cnt > 0) & has_spell_bm).sum().item())
            sp_spell_tube_tot = int(has_spell_bm.sum().item())

        fb_np, ff_np = np.nonzero(n_sp_np >= 0)
        fb = torch.from_numpy(fb_np).to(dev)
        ff = torch.from_numpy(ff_np).to(dev)
        groups = {}
        for idx_i, c in enumerate(n_sp_np[fb_np, ff_np].tolist()):
            groups.setdefault(min(c, MAX_EFF_SPAWNS - 1), []).append(idx_i)

        total_real, total_fields, frames = 0.0, 0.0, 0
        sp_ty_ok, sp_ty_tot = 0, 0
        sp_spell_ty_ok, sp_spell_ty_tot = 0, 0
        for c, idx in groups.items():
            idx_t = torch.tensor(idx, device=dev)
            sb, sf = fb[idx_t], ff[idx_t]
            (
                _,
                real,
                start_pos,
                ray_dxy,
                eff_type_logits,
                owner_logits,
                view_logits,
                ndots_logits,
                age_logits,
            ) = self.eff_spawns(
                glob[sb, sf],
                rows[sb, sf],
                rmask[sb, sf],
                cells[sb, sf],
                anc_txy_all[sb, sf],
                cell_txy_all[sb],
                act_win[sb, sf],
                amask_win[sb, sf],
                aplaced_win[sb, sf],
                act_txy_win[sb, sf],
                c + EFF_MARGIN,
                effs=effs_all[sb, sf],
                emask=emask_all[sb, sf],
                eff_txy=eff_txy_all[sb, sf],
                eff_ty=eff_ty_all[sb, sf],
                eff_age=eff_age_all[sb, sf],
                need_fields=bool(c),
                grad_ckpt=self.enc.grad_ckpt,
            )
            target = torch.zeros_like(real)
            if c:
                vi_l, m_lens = [], []
                ty_pad, ow_pad, vw_pad, nd_pad, ag_pad, st_pad, ry_pad = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for i_local, g_idx in enumerate(idx):
                    entries = spawn_map.get(
                        (int(fb_np[g_idx]), int(ff_np[g_idx])), ()
                    )[:c]
                    if not entries:
                        continue
                    m_i = len(entries)
                    vi_l.append(i_local)
                    m_lens.append(m_i)
                    padded = list(entries) + [entries[-1]] * (c - m_i)
                    ty_pad.append(
                        [min(max(e[0], 0), N_EFF_SPAWN_TYPES - 1) for e in padded]
                    )
                    ow_pad.append([min(max(e[1], 0), 1) for e in padded])
                    nd_pad.append(
                        [min(max(e[2] - 1, 0), MAX_EFF_DOTS - 1) for e in padded]
                    )
                    st_pad.append([[e[3], e[4]] for e in padded])
                    ry_pad.append(
                        [[(e[5] - e[3]) / 10.0, (e[6] - e[4]) / 10.0] for e in padded]
                    )
                    ag_pad.append([min(max(e[7], 0), MAX_EFF_AGE - 1) for e in padded])
                    vw_pad.append([e[8] for e in padded])
                if vi_l:
                    vi = torch.tensor(vi_l, device=dev, dtype=torch.long)
                    vb = sb[vi]
                    ty_t = torch.tensor(ty_pad, device=dev, dtype=torch.long)
                    ow_t = torch.tensor(ow_pad, device=dev, dtype=torch.long)
                    nd_t = torch.tensor(nd_pad, device=dev, dtype=torch.long)
                    ag_t = torch.tensor(ag_pad, device=dev, dtype=torch.long)
                    vw_t = torch.tensor(vw_pad, device=dev, dtype=torch.float32)
                    st_txy = (
                        torch.tensor(st_pad, device=dev, dtype=torch.float32)
                        / b["half"][vb].view(-1, 1, 2).float()
                    )
                    ry_t = torch.tensor(ry_pad, device=dev, dtype=torch.float32)

                    lt = F.log_softmax(eff_type_logits[vi].float(), -1).gather(
                        2, ty_t[:, None, :].expand(-1, c + EFF_MARGIN, -1)
                    )
                    lo = F.log_softmax(owner_logits[vi].float(), -1).gather(
                        2, ow_t[:, None, :].expand(-1, c + EFF_MARGIN, -1)
                    )
                    cost_np = (
                        (
                            torch.cdist(start_pos[vi].float(), st_txy, p=1)
                            - lt
                            - lo
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    fi_l, qi_l, ti_l, inv_m_l = [], [], [], []
                    for j_m, m_i in enumerate(m_lens):
                        qj, tj = linear_sum_assignment(cost_np[j_m, :, :m_i])
                        fi_l.extend([j_m] * m_i)
                        qi_l.extend(qj.tolist())
                        ti_l.extend(tj.tolist())
                        inv_m_l.extend([1.0 / m_i] * m_i)
                    fi_t = torch.tensor(fi_l, device=dev, dtype=torch.long)
                    qi_t = torch.tensor(qi_l, device=dev, dtype=torch.long)
                    ti_t = torch.tensor(ti_l, device=dev, dtype=torch.long)
                    inv_m_t = torch.tensor(inv_m_l, device=dev, dtype=torch.float32)
                    orig_i = vi[fi_t]
                    target[orig_i, qi_t] = 1.0

                    ty_m = ty_t[fi_t, ti_t]
                    nd_m = nd_t[fi_t, ti_t]
                    is_ray = (ty_m == 5) | (ty_m == 12)
                    w_sp = torch.where(is_ray, 0.35, 2.0)
                    l_start = (
                        4.0
                        * F.smooth_l1_loss(
                            start_pos[orig_i, qi_t].float(),
                            st_txy[fi_t, ti_t],
                            beta=0.02,
                            reduction="none",
                        ).sum(-1)
                    )
                    w_ray = torch.where(nd_m > 0, 1.0, 0.1)
                    l_ray = w_ray * F.smooth_l1_loss(
                        ray_dxy[orig_i, qi_t].float(),
                        ry_t[fi_t, ti_t],
                        beta=0.02,
                        reduction="none",
                    ).sum(-1)
                    ce_ty = F.cross_entropy(
                        eff_type_logits[orig_i, qi_t].float(), ty_m, reduction="none"
                    )
                    ce_ow = F.cross_entropy(
                        owner_logits[orig_i, qi_t].float(),
                        ow_t[fi_t, ti_t],
                        reduction="none",
                    )
                    bce_vw = F.binary_cross_entropy_with_logits(
                        view_logits[orig_i, qi_t].float(),
                        vw_t[fi_t, ti_t],
                        reduction="none",
                    ).mean(-1)
                    ce_nd = F.cross_entropy(
                        ndots_logits[orig_i, qi_t].float(), nd_m, reduction="none"
                    )
                    ce_ag = 0.25 * F.cross_entropy(
                        age_logits[orig_i, qi_t].float(),
                        ag_t[fi_t, ti_t],
                        reduction="none",
                    )
                    total_fields = (
                        total_fields
                        + (
                            (
                                l_start
                                + l_ray
                                + ce_ty
                                + ce_ow
                                + bce_vw
                                + ce_nd
                                + ce_ag
                            )
                            * w_sp
                            * inv_m_t
                        ).sum()
                    )
                    frames += len(vi_l)
                    with torch.no_grad():
                        pr_ty = eff_type_logits[orig_i, qi_t].detach().argmax(-1)
                        sp_ty_ok += int((pr_ty == ty_m).sum().item())
                        sp_ty_tot += int(ty_m.numel())
                        sp_m = ~is_ray
                        if sp_m.any():
                            sp_spell_ty_ok += int(
                                (pr_ty[sp_m] == ty_m[sp_m]).sum().item()
                            )
                            sp_spell_ty_tot += int(sp_m.sum().item())
            w_rl_sp = torch.where(has_spell_bm[sb, sf], 2.0, 1.0)[:, None]
            total_real = total_real + (
                F.binary_cross_entropy_with_logits(
                    real.float(), target, reduction="none"
                )
                * w_rl_sp
            ).sum()

        l_sp_real = total_real / max(1, len(fb_np)) / (1 + EFF_MARGIN)
        l_sp_fields = (
            total_fields / max(1, frames)
            if frames
            else (
                out["ent"].sum()
                + sum(p_.sum() for p_ in self.eff_spawns.parameters())
            )
            * 0.0
        )
        if self.record_pos and self.eval_cnts is not None:
            self.eval_cnts.update(
                dict(
                    pers_tp=pers_tp,
                    pers_pred=pers_pred,
                    pers_gt=pers_gt,
                    exp_tp=exp_tp,
                    exp_pred=exp_pred,
                    exp_gt=exp_gt,
                    sp_cnt_tp=sp_cnt_tp,
                    sp_cnt_pred=sp_cnt_pred,
                    sp_cnt_gt=sp_cnt_gt,
                    sp_ty_ok=sp_ty_ok,
                    sp_ty_tot=sp_ty_tot,
                    sp_spell_tube_ok=sp_spell_tube_ok,
                    sp_spell_tube_tot=sp_spell_tube_tot,
                    sp_spell_ty_ok=sp_spell_ty_ok,
                    sp_spell_ty_tot=sp_spell_ty_tot,
                )
            )
        logs = dict(
            eff_persist=l_eff_persist,
            eff_persist_f1=eff_persist_f1,
            eff_expire_f1=eff_expire_f1,
            eff_spawn_cnt=l_sp_cnt,
            eff_spawn_cnt_f1=eff_spawn_cnt_f1,
            eff_spawn_real=l_sp_real,
            eff_spawn_fields=l_sp_fields,
        )
        total_eff = l_eff_persist + l_sp_cnt + l_sp_real + 0.15 * l_sp_fields
        return total_eff, logs

    def map_loss(self, nxt_cells, clean_cells, b, f_src, f_tgt, K=256):
        """Next-tubelet visibility, seen and creep at f_tgt on nxt_cells with f_src
        residual + unbiased Horvitz-Thompson boundary/static sampling, plus a detached
        (pathable, buildable) probe at f_src on clean_cells.detach()."""
        B, F_ = b["mask"].shape[:2]
        M_pred, S = nxt_cells.shape[1:3]
        dev = nxt_cells.device
        if M_pred == 0:
            z = nxt_cells.sum() * 0
            return z, z, dict(map=z, map_chg_f1=z, probe_pb=z, probe_pb_f1=z)
        BM = B * M_pred
        side = int(round(math.sqrt(S)))
        r = TOP // side
        cells = nxt_cells.reshape(BM, S, -1)
        side_bm = b["side"][:, None].expand(B, M_pred).reshape(-1, 1).float()
        size = side_bm / TOP

        src_all = self._unpack_dyn_map(b["map"][:, f_src], BM, dev)
        tgt_all = self._unpack_dyn_map(b["map"][:, f_tgt], BM, dev)
        chg_cell = (tgt_all != src_all).any(-1)  # [BM, 4096]
        n_chg_cell = chg_cell.float().sum(1, keepdim=True)  # [BM, 1]
        n_stay_cell = float(TOP * TOP) - n_chg_cell

        k_chg, k_stay = 128, 256
        idx_chg = (
            torch.where(chg_cell, 2.0, 0.0) + torch.rand((BM, TOP * TOP), device=dev)
        ).topk(k_chg, dim=1).indices
        idx_stay = (
            torch.where(~chg_cell, 2.0, -2.0) + torch.rand((BM, TOP * TOP), device=dev)
        ).topk(k_stay, dim=1).indices
        is_c = chg_cell.gather(1, idx_chg)
        is_s = ~chg_cell.gather(1, idx_stay)
        w_c = is_c.float() * (n_chg_cell / float(k_chg)).clamp(min=1.0)
        w_s = is_s.float() * (n_stay_cell / float(k_stay)).clamp(min=1.0)

        idx_ms = torch.cat([idx_chg, idx_stay], dim=1)
        w_cell = torch.cat([w_c, w_s], dim=1)
        iy_m, ix_m = idx_ms // TOP, idx_ms % TOP
        cell_m = (iy_m // r) * side + (ix_m // r)
        off_m = torch.stack(
            [(ix_m % r + 0.5 - r / 2) * size, (iy_m % r + 0.5 - r / 2) * size], -1
        )
        sg = torch.gather(src_all, 1, idx_ms[..., None].expand(-1, -1, C_DYNAMIC))
        tg = torch.gather(tgt_all, 1, idx_ms[..., None].expand(-1, -1, C_DYNAMIC))
        o = self.predict_map(cells, cell_m, off_m, sg)

        chg_m = (tg - sg).abs() > 0.5
        bce_m = F.binary_cross_entropy_with_logits(o, tg, reduction="none")
        w_elem = w_cell[..., None] * torch.where(chg_m, 3.0, 1.0)
        min_chg_m = max(32.0, 0.002 * float(BM * TOP * TOP))
        n_chg_c = (w_cell[..., None] * chg_m.float()).sum((0, 1)).clamp(min=min_chg_m)
        mp = 0.06 * ((bce_m * w_elem).sum((0, 1)) / n_chg_c).mean()
        with torch.no_grad():
            pr_b = (o.detach() > 0.0).float()
            pr_fl = (pr_b != sg).float()
            gt_fl = chg_m.float()
            tp_fl = (
                w_cell[..., None] * pr_fl * gt_fl * (pr_b == tg).float()
            ).sum()
            pd_fl = (w_cell[..., None] * pr_fl).sum()
            gd_fl = (w_cell[..., None] * gt_fl).sum()
            map_chg_f1 = 2.0 * tp_fl / (pd_fl + gd_fl).clamp(min=1.0)
            if self.record_pos and self.eval_cnts is not None:
                self.eval_cnts["map_chg_tp"] += int(round(float(tp_fl.item())))
                self.eval_cnts["map_chg_pred"] += int(round(float(pd_fl.item())))
                self.eval_cnts["map_chg_gt"] += int(round(float(gd_fl.item())))

        # Detached probe: [p0_path, p0_build, p1_path, p1_build] at f_src on uniform K
        idx = torch.randint(0, TOP * TOP, (BM, K), device=dev)
        iy, ix = idx // TOP, idx % TOP
        cell = (iy // r) * side + (ix // r)
        off = torch.stack(
            [(ix % r + 0.5 - r / 2) * size, (iy % r + 0.5 - r / 2) * size], -1
        )
        clean_det = clean_cells.detach().reshape(BM, S, -1)
        pb_logits = self.pb_probe(clean_det, cell, off).float()
        ctr_bm = b["center"][:, None].expand(B, M_pred, 2).reshape(-1, 2).float()
        org_bm = b["origin"][:, None].expand(B, M_pred, 2).reshape(-1, 2).float()
        wx = torch.floor(
            ctr_bm[:, 0:1] + (ix.float() + 0.5) * size - 0.5 * side_bm
        ).long()
        wy = torch.floor(
            ctr_bm[:, 1:2] + (iy.float() + 0.5) * size - 0.5 * side_bm
        ).long()
        gx = wx - torch.round(org_bm[:, 0:1]).long()
        gy = wy - torch.round(org_bm[:, 1:2]).long()
        gh, gw = b["static_grid"].shape[1:3]
        in_b = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
        gx_c, gy_c = gx.clamp(0, gw - 1), gy.clamp(0, gh - 1)
        bi_bm = torch.arange(B, device=dev)[:, None].expand(B, M_pred).reshape(-1, 1)
        fs_bm = f_src[None, :].expand(B, M_pred).reshape(-1, 1)
        playable = in_b & (b["static_grid"][bi_bm, gy_c, gx_c, 2] > 0)
        if b["grid"].ndim == 4:
            g_val = b["grid"][bi_bm, fs_bm, gy_c, gx_c].to(torch.int32)
            pb_shifts = torch.tensor([3, 4, 8, 9], device=dev, dtype=torch.int32)
            pb_tgt = ((g_val[..., None] >> pb_shifts) & 1).float()
        else:
            pb_tgt = b["grid"][bi_bm, fs_bm, gy_c, gx_c][..., (3, 4, 8, 9)].float()
        pb_elem = F.binary_cross_entropy_with_logits(
            pb_logits, pb_tgt, reduction="none"
        ).mean(-1)
        probe_pb = (pb_elem * playable.float()).sum() / playable.float().sum().clamp(
            min=1.0
        )
        with torch.no_grad():
            pb_p = (pb_logits.detach() > 0).float() * playable[..., None].float()
            pb_t = pb_tgt * playable[..., None].float()
            probe_pb_f1 = (
                2.0 * (pb_p * pb_t).sum() / (pb_p.sum() + pb_t.sum()).clamp(min=1.0)
            )
        logs = dict(
            map=mp,
            map_chg_f1=map_chg_f1,
            probe_pb=probe_pb.detach(),
            probe_pb_f1=probe_pb_f1,
        )
        return mp, probe_pb, logs
