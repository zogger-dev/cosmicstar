"""Multi-scale timeline and fog-of-war Stochastic Observer (`L2`) rollout inspection."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from ..core.constants import BANK_SCALE, MAP_SCALE, N0, SPEED_SCALE
from ..data.game_store import collate, to_device, trim_batch
from ..l1_tactical import squad_pooler as SP
from ..l2_observer import (
    collate_static_maps,
    extract_rich_static_map,
    extract_squad_gt_perception,
)


def _describe_squad_units_from_weights(
    assign_w_pm: np.ndarray,
    own_mask: np.ndarray,
    alive_mask: np.ndarray,
    rtypes: np.ndarray,
    unit_meta: dict[int, dict[str, Any]],
    max_types: int = 3,
) -> str:
    counts: dict[str, float] = {}
    for r in range(len(assign_w_pm)):
        w = float(assign_w_pm[r])
        if not (own_mask[r] and alive_mask[r]) or w < 0.06:
            continue
        uid = int(rtypes[r])
        nm = unit_meta.get(uid, {}).get("name", f"Unit_{uid}")
        counts[nm] = counts.get(nm, 0.0) + w
    if not counts:
        return ""
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:max_types]
    parts = []
    for nm, cnt in top:
        c_int = max(1, int(round(cnt)))
        parts.append(f"{c_int}×{nm}" if c_int > 1 or cnt >= 0.6 else nm)
    return ", ".join(parts)


class ObserverViewMixin:
    """Full-game 5s timeline and multi-branch fog-of-war observer rollout methods."""

    def get_timeline(self, gid: str, mode: str = "cont") -> dict[str, Any]:
        """Return full-game 5.0s timeline via `ObserverWorldModelL2` across all 5 SC2 patches."""
        g_game = self._get_game(gid)
        meta_info = self.catalog_by_id.get(gid, {"gid": gid, "duration_loops": g_game.F})
        with self.lock, torch.no_grad():
            self._maybe_reload_observer()
            g_sum = self._get_obs_game_summary(gid, up_to_loop=g_game.F)
            loops = g_sum["loops"].astype(int).tolist()
            T = len(loops)
            t_sec = [round(l / 22.4, 2) for l in loops]
            q_sq = g_sum["q_squad"].to(self.device).float()
            z_sq = g_sum["z_squad"].to(self.device).float()
            gate = g_sum["gate"].to(self.device).float()
            q_mac = g_sum["q_macro"].to(self.device).float()
            z_mac = g_sum["z_macro"].to(self.device).float()
            st_1 = collate_static_maps([g_sum["static_map"]], device=self.device)
            st_t = {
                k: (v.expand(T, *v.shape[1:]) if isinstance(v, torch.Tensor) and v.shape[0] == 1 else v)
                for k, v in st_1.items()
            }
            hout = self.obs_l2.ph.evaluate_hamiltonian(
                torch.cat([q_sq, z_sq], dim=-1),
                torch.cat([q_mac, z_mac], dim=-1),
                gate,
                static_map=st_t,
            )
            h1_v = (hout["H1"].detach().cpu().numpy() * 1000.0).round(2).tolist()
            h2_v = (hout["H2"].detach().cpu().numpy() * 1000.0).round(2).tolist()
            a1_v = (hout["A1"].detach().cpu().numpy() * 1000.0).round(2).tolist()
            zeros = [0.0] * T
            q_np = q_sq.detach().cpu().numpy()
            p1_u = (q_np[:, 0, :, 6].sum(axis=-1) * N0).round(1).tolist()
            p2_u = (q_np[:, 1, :, 6].sum(axis=-1) * N0).round(1).tolist()
            p1_hp = (q_np[:, 0, :, 8].sum(axis=-1) + q_np[:, 0, :, 9].sum(axis=-1)).round(2).tolist()
            p2_hp = (q_np[:, 1, :, 8].sum(axis=-1) + q_np[:, 1, :, 9].sum(axis=-1)).round(2).tolist()
            z_mac_32 = z_mac[:, :, :32].detach().cpu().numpy().round(3).tolist()
            return {
                "gid": gid,
                "mode": mode or "observer_v2",
                "meta": meta_info,
                "T": T,
                "loops": loops,
                "t_sec": t_sec,
                "H1_l1": zeros,
                "H2_l1": zeros,
                "A1_l1": zeros,
                "H1_l2": h1_v,
                "H2_l2": h2_v,
                "A1_l2": a1_v,
                "A1_tot": a1_v,
                "diss1_l2": zeros,
                "diss2_l2": zeros,
                "p1_units": p1_u,
                "p2_units": p2_u,
                "p1_hp_eq": p1_hp,
                "p2_hp_eq": p2_hp,
                "z_macro": z_mac_32,
            }

    @torch.no_grad()
    def _get_obs_game_summary(self, gid: str, up_to_loop: int | None = None) -> dict[str, Any]:
        """Progressively encodes and caches 5.0s (`112`-loop) v2 Perceiver Squad + Macro +
        Perceived-Enemy + Tempo prefix summaries up to `up_to_loop` (`step=7`, `need_map=False`)."""
        g = self._get_game(gid)
        all_marks = list(range(112, max(225, g.F - 113), 112))
        if not all_marks:
            all_marks = [min(112, max(28, g.F // 2))]
        cutoff = g.F if up_to_loop is None else max(224, int(up_to_loop))
        needed_marks = [m for m in all_marks if m <= cutoff] or [all_marks[0]]
        target_n = len(needed_marks)
        if target_n % 2 == 1 and target_n < len(all_marks):
            target_n += 1
            needed_marks = all_marks[:target_n]

        if gid not in self._obs_game_cache:
            if len(self._obs_game_cache) >= 6:
                oldest = next(iter(self._obs_game_cache))
                del self._obs_game_cache[oldest]
            self._obs_game_cache[gid] = dict(
                gid=gid,
                loops=np.zeros((0,), dtype=np.int32),
                static_map=extract_rich_static_map(g),
                q_squad=None,
                z_squad=None,
                gate=None,
                active=None,
                q_macro=None,
                z_macro=None,
                q_opp_obs=None,
                z_opp_obs=None,
                gate_opp_obs=None,
                z_omni_scene=None,
                gt_vis_live=None,
                gt_shimmer=None,
                tempo_feat=None,
                prev_s=None,
            )

        entry = self._obs_game_cache[gid]
        done_n = len(entry["loops"])
        if done_n >= len(needed_marks):
            return entry

        new_marks = needed_marks[done_n:]
        q_sq_all, z_sq_all, gate_all, act_all = [], [], [], []
        q_mac_all, z_mac_all = [], []
        q_opp_all, z_opp_all, g_opp_all = [], [], []
        z_omni_all, vis_all, shim_all, tempo_all = [], [], [], []
        prev_s = entry["prev_s"]

        def _push_state(st: dict[str, torch.Tensor], vl: torch.Tensor, sh: torch.Tensor, tmp: torch.Tensor):
            q_sq_all.append(st["q_squad"].detach().cpu().float())
            z_sq_all.append(st["z_squad"].detach().cpu().to(torch.bfloat16))
            gate_all.append(st["gate"].detach().cpu().to(torch.float16))
            act_all.append(st["active"].detach().cpu().bool())
            q_mac_all.append(st["q_macro"].detach().cpu().float())
            z_mac_all.append(st["z_macro"].detach().cpu().to(torch.bfloat16))
            q_opp_all.append(st["q_opp_obs"].detach().cpu().to(torch.float16))
            z_opp_all.append(st["z_opp_obs"].detach().cpu().to(torch.bfloat16))
            g_opp_all.append(st["gate_opp_obs"].detach().cpu().to(torch.float16))
            z_omni_all.append(st["z_omni_scene"].detach().cpu().to(torch.bfloat16))
            vis_all.append(vl.detach().cpu().to(torch.float16))
            shim_all.append(sh.detach().cpu().to(torch.float16))
            tempo_all.append(tmp.detach().cpu().float())

        def _slice_ctx_one(ctx_d: dict[str, Any], j: int) -> dict[str, Any]:
            return {
                k: ({pk: pv[j : j + 1] for pk, pv in v.items()} if isinstance(v, dict) else v[j : j + 1])
                for k, v in ctx_d.items()
            }

        pair_specs = []
        idx_m = 0
        while idx_m + 1 < len(new_marks):
            m0 = new_marks[idx_m]
            t0_w = 0 if (done_n == 0 and idx_m == 0) else max(0, m0 - 112)
            pair_specs.append(t0_w)
            idx_m += 2
        odd_mark = new_marks[idx_m] if idx_m < len(new_marks) else None

        chunk = 16
        for i in range(0, len(pair_specs), chunk):
            sp_c = pair_specs[i : i + chunk]
            ws = [g.window(t0_w, 33, swap=False, step=7, need_map=False) for t0_w in sp_c]
            b = to_device(trim_batch(collate(ws, self.vocab)), self.device)
            B_c = b["mask"].shape[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctx = self.obs_enc.encode_window_context(b, list(range(1, 33)))
                tmp_0_b = self.obs_posterior.extract_window_tempo_and_tokens(b, list(range(1, 17)))[0]
                tmp_1_b = self.obs_posterior.extract_window_tempo_and_tokens(b, list(range(17, 33)))[0]
                for j in range(B_c):
                    bj = {
                        k: (v[j : j + 1] if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == B_c else v)
                        for k, v in b.items()
                    }
                    cj = _slice_ctx_one(ctx, j)
                    s_0 = self.obs_pooler(bj, cj, f_start=0, f_end=15, prev_state=prev_s, batch_frame_end=16)
                    s_1 = self.obs_pooler(bj, cj, f_start=16, f_end=31, prev_state=s_0, batch_frame_end=32)
                    vl_0, sh_0 = extract_squad_gt_perception(cj, 15, s_0)
                    vl_1, sh_1 = extract_squad_gt_perception(cj, 31, s_1)
                    _push_state(s_0, vl_0, sh_0, tmp_0_b[j : j + 1])
                    _push_state(s_1, vl_1, sh_1, tmp_1_b[j : j + 1])
                    prev_s = {
                        k: s_1[k].detach()
                        for k in ("q_squad", "z_squad", "gate", "active", "loop_end")
                    }

        if odd_mark is not None:
            w_odd = g.window(max(0, odd_mark - 112), 17, swap=False, step=7, need_map=False)
            b_odd = to_device(trim_batch(collate([w_odd], self.vocab)), self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctx_odd = self.obs_enc.encode_window_context(b_odd, list(range(1, 17)))
                tmp_odd = self.obs_posterior.extract_window_tempo_and_tokens(b_odd, list(range(1, 17)))[0]
                s_odd = self.obs_pooler(
                    b_odd, ctx_odd, f_start=0, f_end=15, prev_state=prev_s, batch_frame_end=16
                )
                vl_o, sh_o = extract_squad_gt_perception(ctx_odd, 15, s_odd)
                _push_state(s_odd, vl_o, sh_o, tmp_odd)
                prev_s = {
                    k: s_odd[k].detach()
                    for k in ("q_squad", "z_squad", "gate", "active", "loop_end")
                }

        def _append_cat(old_t, new_list):
            nt = torch.cat(new_list, dim=0)
            return nt if old_t is None else torch.cat([old_t, nt], dim=0)

        entry["loops"] = np.array(needed_marks, dtype=np.int32)
        entry["q_squad"] = _append_cat(entry["q_squad"], q_sq_all)
        entry["z_squad"] = _append_cat(entry["z_squad"], z_sq_all)
        entry["gate"] = _append_cat(entry["gate"], gate_all)
        entry["active"] = _append_cat(entry["active"], act_all)
        entry["q_macro"] = _append_cat(entry["q_macro"], q_mac_all)
        entry["z_macro"] = _append_cat(entry["z_macro"], z_mac_all)
        entry["q_opp_obs"] = _append_cat(entry["q_opp_obs"], q_opp_all)
        entry["z_opp_obs"] = _append_cat(entry["z_opp_obs"], z_opp_all)
        entry["gate_opp_obs"] = _append_cat(entry["gate_opp_obs"], g_opp_all)
        entry["z_omni_scene"] = _append_cat(entry["z_omni_scene"], z_omni_all)
        entry["gt_vis_live"] = _append_cat(entry["gt_vis_live"], vis_all)
        entry["gt_shimmer"] = _append_cat(entry["gt_shimmer"], shim_all)
        entry["tempo_feat"] = _append_cat(entry["tempo_feat"], tempo_all)
        entry["prev_s"] = prev_s
        return entry

    def inspect_observer_rollout(
        self,
        gid: str,
        loop_t: int,
        n_branches: int = 32,
        horizon_steps: int = 24,
        mode: str = "forecast",
        grid_res: int = 44,
    ) -> dict[str, Any]:
        """Runs Multi-Scale Stochastic Observer v2 (`ObserverWorldModelL2` + `L1ChunkFlowDecoder`)
        at exact `loop_t`, rolling out `n_branches` `5.0s` futures over `horizon_steps` (`+120s`)
        with Common Random Numbers, 3-Stream Theory-of-Mind Plan Misalignment, Intrinsic Leverage,
        and `+5.0s` (`16 x 0.3125s`) tactical entity trajectories."""
        self._maybe_reload_observer()
        with self.lock, torch.no_grad():
            t0_wall = time.perf_counter()
            g = self._get_game(gid)
            loop_t = max(112, min(g.F - 4, int(loop_t)))
            n_branches = max(4, min(64, int(n_branches)))
            horizon_steps = max(2, min(32, int(horizon_steps)))
            if mode not in ("forecast", "counterplay_p1", "counterplay_p2"):
                mode = "forecast"

            g_sum = self._get_obs_game_summary(gid, up_to_loop=loop_t)
            marks = g_sum["loops"]
            valid_idx = np.where(marks <= loop_t - 224)[0]
            if len(valid_idx) > 46:
                valid_idx = valid_idx[-46:]

            prev_s = None
            if len(valid_idx) > 0:
                last_i = int(valid_idx[-1])
                prev_s = {
                    "q_squad": g_sum["q_squad"][last_i : last_i + 1].to(self.device).float(),
                    "z_squad": g_sum["z_squad"][last_i : last_i + 1].to(self.device).float(),
                    "gate": g_sum["gate"][last_i : last_i + 1].to(self.device).float(),
                    "active": g_sum["active"][last_i : last_i + 1].to(self.device).bool(),
                    "loop_end": torch.tensor([float(marks[last_i])], device=self.device, dtype=torch.float32),
                }

            # Live-encode exact step=7 window ending right at loop_t
            if loop_t >= 224:
                w_live = g.window(loop_t - 224, 33, swap=False, step=7, need_map=False)
                b_live = to_device(trim_batch(collate([w_live], self.vocab)), self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ctx_live = self.obs_enc.encode_window_context(b_live, list(range(1, 33)))
                    tmp_pre = self.obs_posterior.extract_window_tempo_and_tokens(b_live, list(range(1, 17)))[0]
                    tmp_now = self.obs_posterior.extract_window_tempo_and_tokens(b_live, list(range(17, 33)))[0]
                    s_pre = self.obs_pooler(
                        b_live, ctx_live, f_start=0, f_end=15, prev_state=prev_s, batch_frame_end=16
                    )
                    s_now = self.obs_pooler(
                        b_live, ctx_live, f_start=16, f_end=31, prev_state=s_pre, batch_frame_end=32
                    )
                    vl_pre, sh_pre = extract_squad_gt_perception(ctx_live, 15, s_pre)
                    vl_now, sh_now = extract_squad_gt_perception(ctx_live, 31, s_now)
                s_pre["gt_vis_live"] = vl_pre
                s_pre["gt_shimmer"] = sh_pre
                s_now["gt_vis_live"] = vl_now
                s_now["gt_shimmer"] = sh_now
                live_states = [s_pre, s_now]
                live_tempos = [tmp_pre, tmp_now]
                f_end_b, f_end_c = 32, 31
            else:
                w_live = g.window(max(0, loop_t - 112), 17, swap=False, step=7, need_map=False)
                b_live = to_device(trim_batch(collate([w_live], self.vocab)), self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ctx_live = self.obs_enc.encode_window_context(b_live, list(range(1, 17)))
                    tmp_now = self.obs_posterior.extract_window_tempo_and_tokens(b_live, list(range(1, 17)))[0]
                    s_now = self.obs_pooler(
                        b_live, ctx_live, f_start=0, f_end=15, prev_state=prev_s, batch_frame_end=16
                    )
                    vl_now, sh_now = extract_squad_gt_perception(ctx_live, 15, s_now)
                s_now["gt_vis_live"] = vl_now
                s_now["gt_shimmer"] = sh_now
                live_states = [s_now]
                live_tempos = [tmp_now]
                f_end_b, f_end_c = 16, 15

            def _cat_pref(key_cache: str, live_list: list[torch.Tensor]) -> torch.Tensor:
                lv = torch.cat(
                    [(x.float() if x.is_floating_point() else x) for x in live_list], dim=0
                ).unsqueeze(0)
                if len(valid_idx) == 0:
                    return lv
                idx_t = torch.as_tensor(valid_idx, dtype=torch.long)
                hist = g_sum[key_cache][idx_t].to(self.device)
                if hist.is_floating_point():
                    hist = hist.float()
                return torch.cat([hist.unsqueeze(0), lv], dim=1)

            state_keys = (
                "q_squad",
                "z_squad",
                "gate",
                "active",
                "q_macro",
                "z_macro",
                "q_opp_obs",
                "z_opp_obs",
                "gate_opp_obs",
                "z_omni_scene",
                "gt_vis_live",
                "gt_shimmer",
            )
            prefix_seq = {sk: _cat_pref(sk, [st[sk] for st in live_states]) for sk in state_keys}
            prefix_seq["tempo_feat"] = _cat_pref("tempo_feat", live_tempos)
            st_1 = collate_static_maps([g_sum["static_map"]], device=self.device)

            # Common Random Numbers: seed depends ONLY on gid (NEVER loop_t!) so 1s scrubs are smooth!
            seed_val = ((sum(ord(c) for c in gid) * 1315423911) + 42) & 0x7FFFFFFF
            with torch.autocast("cuda", dtype=torch.bfloat16):
                roll = self.obs_l2.rollout_observer(
                    prefix_seq,
                    static_map=st_1,
                    n_branches=n_branches,
                    horizon_steps=horizon_steps,
                    flow_steps=4,
                    temperature=1.0,
                    grid_h=grid_res,
                    grid_w=grid_res,
                    seed=seed_val,
                    mode=mode,
                )

            # Synthesize +5.0s (16 frames @ 0.3125s) tactical entity trajectories via L1ChunkFlowDecoder
            micro_chunk = None
            if (
                roll.get("step0_s1") is not None
                and roll.get("step0_u") is not None
                and roll.get("step0_xi_world") is not None
            ):
                ch = self.obs_decoder.sample_chunk(
                    b=b_live,
                    ctx=ctx_live,
                    f_0=f_end_b,
                    c_0=f_end_c,
                    s_0=s_now,
                    s_1=roll["step0_s1"],
                    u_flat=roll["step0_u"],
                    xi_world=roll["step0_xi_world"],
                    static_map=st_1,
                    steps=4,
                    seed=seed_val,
                )
                ent_valid_np = ch["ent_valid"][0].detach().cpu().numpy()
                ent_owner_np = ch["ent_owner"][0].detach().cpu().numpy()
                ent_rtype_np = ch["ent_rtype"][0].detach().cpu().numpy()
                ent_pos0_np = ch["ent_pos_0"][0].detach().cpu().numpy()
                ent_xy_np = ch["ent_xy"][0].detach().cpu().numpy()
                ent_hp_np = ch["ent_hp"][0].detach().cpu().numpy()
                ent_al_np = ch["ent_alive_prob"][0].detach().cpu().numpy()
                ent_op_np = ch["ent_ord_prob"][0].detach().cpu().numpy()
                ent_ot_np = ch["ent_ord_target_xy"][0].detach().cpu().numpy()

                micro_units = []
                for r in np.where(ent_valid_np)[0]:
                    uid = int(ent_rtype_np[r])
                    um = self.unit_meta.get(uid, {})
                    is_st = bool(um.get("structure", False))
                    # Include all mobile units + any structure predicted to take damage / die
                    if is_st and float(ent_al_np[r].min()) > 0.90 and float(ent_hp_np[r].max() - ent_hp_np[r].min()) < 0.04:
                        continue
                    micro_units.append(
                        {
                            "owner": int(ent_owner_np[r]) + 1,
                            "rtype": uid,
                            "name": um.get("name", f"Unit_{uid}"),
                            "structure": is_st,
                            "flying": bool(um.get("flying", False)),
                            "radius": round(float(um.get("radius", 0.5)), 2),
                            "x0": round(float(ent_pos0_np[r, 0]), 2),
                            "y0": round(float(ent_pos0_np[r, 1]), 2),
                            "x": np.round(ent_xy_np[r, :, 0], 2).tolist(),
                            "y": np.round(ent_xy_np[r, :, 1], 2).tolist(),
                            "hp": np.round(ent_hp_np[r], 2).tolist(),
                            "alive_prob": np.round(ent_al_np[r], 2).tolist(),
                            "ord_prob": np.round(ent_op_np[r], 2).tolist(),
                            "ord_tx": np.round(ent_ot_np[r, :, 0], 2).tolist(),
                            "ord_ty": np.round(ent_ot_np[r, :, 1], 2).tolist(),
                        }
                    )

                sp_prob_np = ch["spawn_prob"][0].detach().cpu().numpy()
                sp_bk_np = ch["spawn_birth_k"][0].detach().cpu().numpy()
                sp_orig_np = ch["spawn_origin_xy"][0].detach().cpu().numpy()
                sp_xy_np = ch["spawn_xy"][0].detach().cpu().numpy()
                sp_al_np = ch["spawn_alive_prob"][0].detach().cpu().numpy()
                sp_rt_np = ch["spawn_rtype"][0].detach().cpu().numpy()

                micro_spawns = []
                for p_i in (0, 1):
                    for s_i in range(sp_prob_np.shape[1]):
                        pr = float(sp_prob_np[p_i, s_i])
                        if pr < 0.65:
                            continue
                        uid = int(sp_rt_np[p_i, s_i])
                        um = self.unit_meta.get(uid, {})
                        sp_nm = um.get("name", f"Unit_{uid}")
                        if sp_nm.lower() in ("larva", "egg", "creeptumor", "creeptumorburrowed"):
                            continue
                        micro_spawns.append(
                            {
                                "owner": p_i + 1,
                                "prob": round(pr, 2),
                                "birth_k": int(sp_bk_np[p_i, s_i]),
                                "rtype": uid,
                                "name": sp_nm,
                                "flying": bool(um.get("flying", False)),
                                "radius": round(float(um.get("radius", 0.5)), 2),
                                "origin_x": round(float(sp_orig_np[p_i, s_i, 0]), 2),
                                "origin_y": round(float(sp_orig_np[p_i, s_i, 1]), 2),
                                "x": np.round(sp_xy_np[p_i, s_i, :, 0], 2).tolist(),
                                "y": np.round(sp_xy_np[p_i, s_i, :, 1], 2).tolist(),
                                "alive_prob": np.round(sp_al_np[p_i, s_i], 2).tolist(),
                            }
                        )

                micro_chunk = {
                    "dt_sec": 0.3125,
                    "n_steps": 16,
                    "step_secs": [round((k + 1) * 0.3125, 2) for k in range(16)],
                    "rep_branch_idx": int(roll.get("rep_branch_idx", 0)),
                    "units": micro_units,
                    "spawns": micro_spawns,
                }

            a_traj_np = roll["A_traj"].detach().float().cpu().numpy()  # [N, H + 1]
            q10, q25, q50, q75, q90 = np.percentile(a_traj_np, [10, 25, 50, 75, 90], axis=0)
            q_mean = a_traj_np.mean(axis=0)
            drift_np = a_traj_np[:, -1] - a_traj_np[:, 0]
            k_cv = max(1, n_branches // 10)
            cvar10_p2 = float(-np.sort(drift_np)[-k_cv:].mean())

            cx, cy = float(g.center[0]), float(g.center[1])
            q_sq_0 = s_now["q_squad"][0].detach().float().cpu().numpy()   # [2, M, 14]
            gate_0 = s_now["gate"][0].detach().float().cpu().numpy()      # [2, M]
            act_0 = s_now["active"][0].detach().cpu().numpy()             # [2, M]
            assign_0 = s_now["assign_w"][0].detach().float().cpu().numpy()  # [2, M, R]
            lev_0 = roll["squad_leverage"].detach().float().cpu().numpy()   # [2, M]
            vl_0_np = s_now["gt_vis_live"][0].detach().float().cpu().numpy()  # [2, M]: viewer p -> enemy 1-p slot m
            sh_0_np = s_now["gt_shimmer"][0].detach().float().cpu().numpy()   # [2, M]

            b_own_np = b_live["owner"][0].detach().cpu().numpy()
            b_alive_np = (b_live["mask"][0, f_end_b] & b_live["alive"][0, f_end_b]).detach().cpu().numpy()
            b_rtype_np = b_live["rtype"][0, f_end_b].detach().cpu().numpy()

            squads_out = []
            squad_desc_map: dict[tuple[int, int], str] = {}
            for p_i in (0, 1):
                opp = 1 - p_i
                own_m = b_own_np == p_i
                for m in range(q_sq_0.shape[1]):
                    if not act_0[p_i, m]:
                        continue
                    qm = q_sq_0[p_i, m]
                    gm = max(1e-4, float(gate_0[p_i, m]))
                    # q_squad[0:2] is gate-weighted (xm * gate), divide by gm for true world centroid!
                    wx = cx + (float(qm[0]) / gm) * MAP_SCALE
                    wy = cy + (float(qm[1]) / gm) * MAP_SCALE
                    u_desc = _describe_squad_units_from_weights(
                        assign_0[p_i, m], own_m, b_alive_np, b_rtype_np, self.unit_meta
                    )
                    squad_desc_map[(p_i, m)] = u_desc
                    lev_val = round(float(lev_0[p_i, m]), 1)
                    squads_out.append(
                        {
                            "owner": p_i + 1,
                            "slot": int(m),
                            "x": round(wx, 2),
                            "y": round(wy, 2),
                            "vx": round((float(qm[2]) / gm) * 5.0, 2),
                            "vy": round((float(qm[3]) / gm) * 5.0, 2),
                            "spread_x": round((float(qm[4]) / gm) * MAP_SCALE, 2),
                            "spread_y": round((float(qm[5]) / gm) * MAP_SCALE, 2),
                            "gate": round(float(gate_0[p_i, m]), 3),
                            "count": round(float(qm[6]) * N0, 1),
                            "cost": round(float(qm[7]) * BANK_SCALE, 0),
                            "hp": round(float(qm[8]) * 1000.0, 0),
                            "shield": round(float(qm[9]) * 1000.0, 0),
                            "struct_frac": round(float(qm[10]) / gm, 2),
                            "air_frac": round(float(qm[11]) / gm, 2),
                            "cloak_frac": round(float(qm[12]) / gm, 2),
                            "det_frac": round(float(qm[13]) / gm, 2),
                            "leverage": lev_val,
                            "risk": lev_val,
                            "vis_by_opp": round(min(1.0, float(vl_0_np[opp, m]) / gm), 2),
                            "shimmer_by_opp": round(min(1.0, float(sh_0_np[opp, m]) / gm), 2),
                            "units": u_desc,
                        }
                    )
            squads_out.sort(key=lambda s: (-s["leverage"], -s["cost"]))

            # Unscouted Enemy Attribution driving each player's Theory-of-Mind Plan Misalignment
            unsc_attr_np = roll["unscouted_attribution"].detach().float().cpu().numpy()  # [2, M]
            unsc_mask_np = roll["unscouted_mask"].detach().float().cpu().numpy()         # [2, M]
            unscouted_threats: dict[str, list[dict[str, Any]]] = {"p1": [], "p2": []}
            for p_viewer, p_key in ((0, "p1"), (1, "p2")):
                opp = 1 - p_viewer
                order = np.argsort(-unsc_attr_np[p_viewer])
                for m in order[:5]:
                    m_i = int(m)
                    attr_v = float(unsc_attr_np[p_viewer, m_i])
                    if attr_v <= 1e-4 or not act_0[opp, m_i]:
                        continue
                    gm = max(1e-4, float(gate_0[opp, m_i]))
                    qm = q_sq_0[opp, m_i]
                    wx = cx + (float(qm[0]) / gm) * MAP_SCALE
                    wy = cy + (float(qm[1]) / gm) * MAP_SCALE
                    u_desc = squad_desc_map.get((opp, m_i), "")
                    if not u_desc:
                        continue
                    unscouted_threats[p_key].append(
                        {
                            "enemy_owner": opp + 1,
                            "slot": m_i,
                            "x": round(wx, 1),
                            "y": round(wy, 1),
                            "attr": round(attr_v, 4),
                            "unscouted_frac": round(float(unsc_mask_np[p_viewer, m_i]) / gm, 2),
                            "shimmer": round(float(sh_0_np[p_viewer, m_i]) / gm, 2),
                            "cost": int(round(float(qm[7]) * BANK_SCALE)),
                            "units": u_desc,
                        }
                    )

            lg_p1 = roll["leverage_grid_p1"].detach().float().cpu().numpy()
            lg_p2 = roll["leverage_grid_p2"].detach().float().cpu().numpy()
            # Net Intrinsic Leverage: +ve (Red) = P1 high-leverage positional control; -ve (Blue) = P2 high-leverage control
            lg_net = lg_p1 - lg_p2

            mis_np = roll["plan_misalignment"].detach().float().cpu().numpy()
            del b_live, ctx_live, prefix_seq
            torch.cuda.empty_cache()
            elapsed_ms = (time.perf_counter() - t0_wall) * 1000.0
            return {
                "available": True,
                "schema_version": 2,
                "mode": mode,
                "active_observer_ckpt": self.active_observer_ckpt,
                "gid": gid,
                "loop_t": loop_t,
                "t_sec": round(loop_t / 22.4, 1),
                "n_branches": n_branches,
                "horizon_steps": horizon_steps,
                "horizon_sec": round(horizon_steps * 5.0, 1),
                "step_secs": [round(k * 5.0, 1) for k in range(horizon_steps + 1)],
                "A0": round(float(roll["A0"].item()), 1),
                "H1_0": round(float(roll["H1_0"].item()), 1),
                "H2_0": round(float(roll["H2_0"].item()), 1),
                "prob_lead_now_p1": round(float(roll["prob_lead_now_p1"].item()), 4),
                "expected_A_end": round(float(q_mean[-1]), 1),
                "expected_drift": round(float(roll["expected_drift"].item()), 1),
                "prob_improve_p1": round(float(roll["prob_improve_p1"].item()), 4),
                "prob_stronger_hand_p1": round(float(roll["prob_stronger_hand_p1"].item()), 4),
                "std_drift": round(float(roll["std_drift"].item()), 1),
                "cvar10_p1": round(float(roll["cvar10_p1"].item()), 1),
                "cvar10_p2": round(cvar10_p2, 1),
                "plan_misalignment": {
                    "p1": round(float(mis_np[0]), 4),
                    "p2": round(float(mis_np[1]), 4),
                },
                "unscouted_threats": unscouted_threats,
                "capability_norm": {
                    "p1": round(float(roll["capability_p1"].norm().item()), 3),
                    "p2": round(float(roll["capability_p2"].norm().item()), 3),
                },
                "quantiles": {
                    "p10": np.round(q10, 1).tolist(),
                    "p25": np.round(q25, 1).tolist(),
                    "p50": np.round(q50, 1).tolist(),
                    "p75": np.round(q75, 1).tolist(),
                    "p90": np.round(q90, 1).tolist(),
                    "mean": np.round(q_mean, 1).tolist(),
                },
                "branches": np.round(a_traj_np[: min(16, n_branches)], 1).tolist(),
                "squads": squads_out,
                "micro_chunk": micro_chunk,
                "grid_res": grid_res,
                "leverage_grid_p1": np.round(lg_p1, 2).tolist(),
                "leverage_grid_p2": np.round(lg_p2, 2).tolist(),
                "leverage_grid_net": np.round(lg_net, 2).tolist(),
                "threat_grid_p1": np.round(lg_p1, 2).tolist(),
                "threat_grid_p2": np.round(lg_p2, 2).tolist(),
                "threat_grid_net": np.round(lg_net, 2).tolist(),
                "elapsed_ms": round(elapsed_ms, 1),
            }
