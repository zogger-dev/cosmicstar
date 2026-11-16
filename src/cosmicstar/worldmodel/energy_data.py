"""The energy model's State from the physics batches (physics_data.collate).

Tables     a row type's constants: value and build time from unit_data.json, speed and
           acceleration from the patch's ruleset (rulesets.json), the energy pool and
           whether it flies from energy_types.json
Frames     the spatiotemporal encoder: a batch and frame indices -> (State at t, State at t+1).
           1. Grounded per-unit columns Q = (1, x, y, vx, vy, hp, shield, energy)
              scaled to [-1, 1] / [0, 1].
           2. Continuous localized potential field sampled at each entity's position and
              on a 2D grid of continuous anchor coordinates (Fourier features + local
              5-point terrain/visibility/creep/pathability stencil via grid_sample).
           3. 6 spatiotemporal transformer blocks over entities + continuous field tokens:
              - Spatial self-attention with 2D RoPE (rotating Q, K, V and inverse-rotating
                outputs into each token's local frame).
              - Causal temporal self-attention along each token's timeline with 1D RoPE
                over game loops (and 2D spatial value rotation across frames).
           4. Contextualized entity tokens project to r = 8 latent tactical modes
              Z = tanh(W_z c), forming the per-player Gram matrix
              G_full = (1 / N_0) [Q | Z]^T [Q | Z] and ledger (minerals, vespene) in fp32.
           5. Per-player commands u_p are sum-pooled over the frame's active physical
              commands and order changes (exact 0 when idle).
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from . import energy as E
from .direct import RoPE2D
from .physics import (
    ROPE_SHORTEST,
    FeedForward,
    Heads,
    RoPE1D,
    cos_sin,
    rotate,
)
from .physics_data import C_LIVE, C_STATIC, NEUTRAL, QUEUE, RAW_NUMS
from .tokenize import N_EFFECT_IDS, PC_KEYS

I = {n: i for i, n in enumerate(RAW_NUMS)}
PC = {n: i for i, n in enumerate(PC_KEYS)}
C_PLY = 14
MAP_WAVES = 6  # wavelengths from 4 to 256 map units
EFF_WAVES = 6  # wavelengths from 4 to 1024 game loops
N_RAW_TYPES = 4096
LOOPS_PER_SECOND_EXPORT = 16.0
INSTANT_ACCEL = 100.0
SPEED_AT, ACCEL_AT = 17, 18
CLIFF_WALKERS = (49, 4)
FLYING_BIT = 1
# Raw numbers read into the entity token context (including contents for neutral patches)
EXTRAS = (
    "contents",
    "harvesters",
    "ideal_harvesters",
    "cargo_used",
    "cargo_max",
    "buff_remain",
    "buff_max",
    "detect_range",
    "radar_range",
    "atk_up",
    "arm_up",
    "shd_up",
    "flags",
    "flags2",
    "display",
    "cloak",
    "add_on",
    "z",
    "weapon_cooldown",
    "build",
    "radius",
)
N_RACES = 5
UNKNOWN_ABILITY = 2**31 - 1


def _parse_single_ruleset(g, rels, widths):
    w_u, w_w, w_a, w_g = (
        widths["unit"],
        widths["weapon"],
        widths["ability"],
        widths["upgrade"],
    )
    unit_feat_dim = w_u + w_w
    speed = np.zeros(N_RAW_TYPES, np.float32)
    mass = np.zeros(N_RAW_TYPES, np.float32)
    unit_feat = np.zeros((N_RAW_TYPES, unit_feat_dim), np.float32)
    for u in g["units"]:
        sp = u["scal"][SPEED_AT] * 5 / LOOPS_PER_SECOND_EXPORT
        ac = float(np.expm1(u["scal"][ACCEL_AT] * 8))
        m = (
            0.0
            if ac >= INSTANT_ACCEL or ac <= 0
            else LOOPS_PER_SECOND_EXPORT**2 / ac
        )
        scal = np.asarray(u["scal"], dtype=np.float32)
        wpn = (
            np.sum(u["weapons"], axis=0, dtype=np.float32)
            if u["weapons"]
            else np.zeros(w_w, np.float32)
        )
        uf = np.concatenate([scal, wpn])
        for a in u["api_ids"]:
            if 0 <= a < N_RAW_TYPES:
                speed[a], mass[a] = sp, m
                unit_feat[a] = uf

    rel_u = rels.index("a_makes_u")
    rel_g = rels.index("a_makes_g")
    makes_u = {s: d_ for r, s, d_ in g["edges"] if r == rel_u}
    makes_g = {s: d_ for r, s, d_ in g["edges"] if r == rel_g}
    abil_by_raw = {}
    for i, ab in enumerate(g["abilities"]):
        af = np.asarray(ab["feat"], dtype=np.float32)
        u_sc = (
            np.asarray(g["units"][makes_u[i]]["scal"], dtype=np.float32)
            if i in makes_u
            else np.zeros(w_u, np.float32)
        )
        g_ft = (
            np.asarray(g["upgrades"][makes_g[i]]["feat"], dtype=np.float32)
            if i in makes_g
            else np.zeros(w_g, np.float32)
        )
        vec = np.concatenate([af, u_sc, g_ft])
        for a_id in ab["api_ids"]:
            if a_id is not None and a_id >= 0:
                abil_by_raw[int(a_id)] = vec
    return speed, mass, unit_feat, abil_by_raw


class Tables:
    """Per raw unit type (API id): value, build_time (loops), race, speed (map units a
    loop), mass (loops² a map unit: the inverse of acceleration; 0 when instant),
    energy_max (0: no pool), flying, cliff, plus static ruleset feature vectors for
    units (unit_scal + summed weapons) and abilities (abil_feat + produced unit_scal +
    researched upg_feat) across both the primary `base_build` and all rulesets in
    `rulesets.json`."""

    def __init__(
        self,
        base_build,
        unit_data="data/worldmodel/unit_data.json",
        rulesets="data/worldmodel/rulesets.json",
        energy_types="data/worldmodel/energy_types.json",
    ):
        value, build, race = (np.zeros(N_RAW_TYPES, np.float32) for _ in range(3))
        is_struct = np.zeros(N_RAW_TYPES, np.float32)
        is_worker = np.zeros(N_RAW_TYPES, np.float32)
        for t, u in json.load(open(unit_data)).items():
            t = int(t)
            if 0 <= t < N_RAW_TYPES:
                value[t] = u["minerals"] + u["vespene"]
                build[t] = u["build_time"]
                race[t] = u["race"]
                is_struct[t] = 1.0 if u.get("structure", False) else 0.0
                is_worker[t] = 1.0 if u.get("worker", False) else 0.0
        rs_doc = json.load(open(rulesets))
        widths = rs_doc["widths"]
        rels = list(rs_doc["relations"])
        rs = rs_doc["rulesets"]
        self.ruleset_names = sorted(rs.keys())
        self.ruleset_to_id = {k: i for i, k in enumerate(self.ruleset_names)}
        keys = [k for k in self.ruleset_names if k.split("_")[0] == str(base_build)]
        self.ruleset = keys[0] if keys else self.ruleset_names[-1]

        w_u, w_w, w_a, w_g = (
            widths["unit"],
            widths["weapon"],
            widths["ability"],
            widths["upgrade"],
        )
        self.unit_feat_dim = w_u + w_w
        self.abil_feat_dim = w_a + w_u + w_g

        energy_max, flying = (np.zeros(N_RAW_TYPES, np.float32) for _ in range(2))
        if os.path.exists(energy_types):
            et = json.load(open(energy_types))
            for t, e in et["energy_max"].items():
                energy_max[int(t)] = e
            for t in et["flying"]:
                flying[t] = 1.0
        cliff = np.zeros(N_RAW_TYPES, np.float32)
        cliff[list(CLIFF_WALKERS)] = 1.0

        all_k_list = []
        all_uf_list = []
        self._all_abil_by_raw = []
        for rk in self.ruleset_names:
            sp_r, m_r, uf_r, ab_r = _parse_single_ruleset(rs[rk], rels, widths)
            all_k_list.append(
                np.stack([value, m_r, sp_r, build, energy_max, flying, cliff], 1)
            )
            all_uf_list.append(uf_r)
            self._all_abil_by_raw.append(ab_r)

        def_idx = self.ruleset_to_id[self.ruleset]
        self.abil_by_raw = self._all_abil_by_raw[def_idx]
        self.race = torch.from_numpy(race.astype(np.int64))
        self.is_struct = torch.from_numpy(is_struct)
        self.is_worker = torch.from_numpy(is_worker)
        self.k = torch.from_numpy(all_k_list[def_idx])
        self.unit_feat = torch.from_numpy(all_uf_list[def_idx])
        self.all_k = torch.from_numpy(np.stack(all_k_list, axis=0))
        self.all_unit_feat = torch.from_numpy(np.stack(all_uf_list, axis=0))

    def abil_feat(self, vocab):
        """[vocab.n_abilities, abil_feat_dim] static feature tensor indexed by vocab ability id."""
        out = np.zeros((vocab.n_abilities, self.abil_feat_dim), np.float32)
        for raw_a, idx in vocab._abil.items():
            if raw_a in self.abil_by_raw and 0 <= idx < vocab.n_abilities:
                out[idx] = self.abil_by_raw[raw_a]
        return torch.from_numpy(out)

    def all_abil_feat(self, vocab):
        """[N_RS, vocab.n_abilities, abil_feat_dim] static ability features across all rulesets."""
        out = np.zeros(
            (len(self.ruleset_names), vocab.n_abilities, self.abil_feat_dim), np.float32
        )
        for ri, ab_map in enumerate(self._all_abil_by_raw):
            for raw_a, idx in vocab._abil.items():
                if raw_a in ab_map and 0 <= idx < vocab.n_abilities:
                    out[ri, idx] = ab_map[raw_a]
        return torch.from_numpy(out)


def extract_entity_perception(
    b: dict[str, torch.Tensor],
    f_idx: torch.Tensor | list[int],
    is_struct_buf: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Extracts exact SC2 engine per-player entity & effect perception at frames `f_idx`:
      - `gt_vis_live`: `[B, nf, 2, R]` bool (player `p` has full live sight & detection on `r`)
      - `gt_shimmer`: `[B, nf, 2, R]` bool (player `p` sees translucent cloak shimmer at `r`, `display=3, cloak=1`)
      - `gt_seen_struct`: `[B, nf, 2, R]` bool (player `p` has fog-snapshot memory of structure `r`)
      - `onscreen`: `[B, nf, R]` bool (`F2_ONSCREEN` bit in owner's camera view)
      - `g_vis`: `[B, nf, 2, R]` bool (raw `vis == 2` cell lookup at entity `(x, y)`)
      - `eff_vis`: `[B, nf, 2, X]` bool (raw `vis == 2` cell lookup at effect `(x, y)`)
    """
    dev = b["mask"].device
    f_t = torch.as_tensor(f_idx, device=dev, dtype=torch.long)
    B = b["mask"].shape[0]
    nf = len(f_t)
    R = b["mask"].shape[2]
    X = b["eff_mask"].shape[2] if "eff_mask" in b else 0

    alive_map = b["mask"][:, f_t] & b["alive"][:, f_t] & (b["inside"][:, f_t] == 0)
    num = b["num"][:, f_t]
    xy_raw = torch.stack([b["x"][:, f_t], b["y"][:, f_t]], dim=-1).float()

    cloak = num[..., I["cloak"]].round().long()
    flags = num[..., I["flags"]].round().long()
    onscreen = ((num[..., I["flags2"]].round().long() & 4) != 0) & alive_map
    cloaked_undetected = (cloak == 4) | (cloak == 1)
    burrowed = (flags & 2) != 0

    if is_struct_buf is not None:
        rtype = b["rtype"][:, f_t].clamp(0, is_struct_buf.shape[0] - 1)
        is_st = is_struct_buf.to(dev)[rtype] > 0.5
    else:
        is_st = torch.zeros_like(alive_map)

    if "grid" in b and "origin" in b:
        with torch.autocast(device_type=dev.type, enabled=False):
            gr = b["grid"][:, f_t].float()  # [B, nf, gh, gw, C_LIVE]
            gh, gw = gr.shape[2:4]
            gr_chw = gr.reshape(B * nf, gh, gw, C_LIVE).permute(0, 3, 1, 2)
            origin = b["origin"][:, None, None, :].float()
            wh = xy_raw.new_tensor([float(gw), float(gh)])
            at_ent = (((xy_raw - origin) / wh) * 2.0 - 1.0).reshape(B * nf, R, 1, 2)
            samp_ent = (
                F.grid_sample(gr_chw, at_ent, mode="nearest", align_corners=False)
                .squeeze(-1)
                .permute(0, 2, 1)
                .reshape(B, nf, R, C_LIVE)
            )
            g_vis = torch.stack(
                [samp_ent[..., 0] > 0.5, samp_ent[..., 5] > 0.5], dim=2
            )  # [B, nf, 2, R]
            g_seen = torch.stack(
                [samp_ent[..., 1] > 0.5, samp_ent[..., 6] > 0.5], dim=2
            )  # [B, nf, 2, R]

            if X > 0 and "eff_xy" in b and "center" in b:
                eff_xy_abs = b["eff_xy"][:, f_t].float() + b["center"][:, None, None, :].float()
                at_eff = (((eff_xy_abs - origin) / wh) * 2.0 - 1.0).reshape(B * nf, X, 1, 2)
                samp_eff = (
                    F.grid_sample(gr_chw, at_eff, mode="nearest", align_corners=False)
                    .squeeze(-1)
                    .permute(0, 2, 1)
                    .reshape(B, nf, X, C_LIVE)
                )
                eff_vis = torch.stack(
                    [samp_eff[..., 0] > 0.5, samp_eff[..., 5] > 0.5], dim=2
                )
            else:
                eff_vis = torch.ones(B, nf, 2, X, dtype=torch.bool, device=dev)
    else:
        g_vis = alive_map.unsqueeze(2).expand(B, nf, 2, R)
        g_seen = g_vis
        eff_vis = torch.ones(B, nf, 2, X, dtype=torch.bool, device=dev)

    # Burrowed detection check via own detectors' detect_range (cloaked units already flip u_cloak 4 -> 2)
    det_r = num[..., I["detect_range"]].float()
    owner_raw = b["owner"][:, None, :].expand(B, nf, R)  # 0=P1, 1=P2, 2=NEUTRAL
    det_by_p = []
    has_burrow = bool(burrowed.any()) if not torch._dynamo.is_compiling() else True
    if has_burrow:
        with torch.autocast(device_type=dev.type, enabled=False):
            for p_i in (0, 1):
                is_det_p = alive_map & (owner_raw == p_i) & (det_r > 0.5)
                r_allow = (det_r + 1.0) * is_det_p.float()  # [B, nf, R_det]
                # Pairwise squared distance [B, nf, R_det, R_target] only if burrowed
                dist2 = (xy_raw.unsqueeze(-2) - xy_raw.unsqueeze(-3)).square().sum(dim=-1)
                in_det = is_det_p.unsqueeze(-1) & (dist2 <= r_allow.unsqueeze(-1).square())
                det_by_p.append(in_det.any(dim=-2))
        det_p = torch.stack(det_by_p, dim=2)  # [B, nf, 2, R]
    else:
        det_p = torch.zeros(B, nf, 2, R, dtype=torch.bool, device=dev)

    vis_list, shim_list, snap_list = [], [], []
    for p_i in (0, 1):
        gv = g_vis[:, :, p_i]
        gs = g_seen[:, :, p_i]
        dp = det_p[:, :, p_i]
        v_live = alive_map & gv & (~cloaked_undetected) & ((~burrowed) | dp)
        v_shim = alive_map & gv & cloaked_undetected
        v_snap = alive_map & (~gv) & gs & is_st & (~cloaked_undetected) & (~burrowed)
        vis_list.append(v_live)
        shim_list.append(v_shim)
        snap_list.append(v_snap)

    return dict(
        gt_vis_live=torch.stack(vis_list, dim=2),
        gt_shimmer=torch.stack(shim_list, dim=2),
        gt_seen_struct=torch.stack(snap_list, dim=2),
        onscreen=onscreen,
        g_vis=g_vis,
        eff_vis=eff_vis,
    )


def _compute_dtype(ref: torch.Tensor) -> torch.dtype:
    return (
        torch.get_autocast_dtype("cuda")
        if torch.is_autocast_enabled("cuda")
        else ref.dtype
    )


class ContinuousPotentialField(nn.Module):
    """Continuous localized potential/terrain field V_loc(x, y) over the map.
    Instead of a full-map 2D Conv ResNet tower or discrete map anchor tokens, evaluates
    multi-scale 2D Fourier coordinate features in physical map units (wavelengths 4 to
    256 map units) plus a 5-point local stencil (center and +-delta cardinal offsets in
    map units) sampled via bilinear grid_sample from the live visibility, creep,
    pathability, buildability, and static terrain height layers strictly at continuous
    entity positions."""

    def __init__(self, d, n_freqs=MAP_WAVES, grid_size=0, stencil_step=2.0):
        super().__init__()
        self.grid_size = 0
        self.n_freqs = n_freqs
        # Physical map-unit wavelengths from 4 to 256 map units (same as entity_encoder)
        self.register_buffer(
            "omega",
            2.0
            * math.pi
            / torch.logspace(math.log10(4.0), math.log10(256.0), n_freqs),
            persistent=False,
        )
        # 5-point local stencil in map units: center, E, W, N, S
        offsets = torch.tensor(
            [
                [0.0, 0.0],
                [stencil_step, 0.0],
                [-stencil_step, 0.0],
                [0.0, stencil_step],
                [0.0, -stencil_step],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("offsets", offsets, persistent=False)
        c_map = C_LIVE + C_STATIC
        in_dim = (4 * n_freqs) + len(offsets) * c_map
        self.net = E.mlp(in_dim, d, d)

    def forward(self, b, f_idx, pos, owner=None):
        """pos: [B, nf, K, 2] in map units relative to map center -> [B, nf, K, d].
        When `owner` (`[B, nf, K]`: `E.P1=1`, `E.P2=2`, `E.GROUND=0`) is supplied, each
        player's entities read ONLY their own 5 live visibility/creep channels (zeroing
        opponent vision/seen channels so invisible enemy observers cannot leak!)."""
        B, nf, K, _ = pos.shape
        dev = pos.device
        dt = _compute_dtype(self.net[0].weight)

        # Fourier-transformed physical map units (wavelengths 4 to 256 map units)
        ang = pos.float().unsqueeze(-1) * self.omega  # [B, nf, K, 2, n_freqs]
        fourier = torch.cat([ang.sin().flatten(-2), ang.cos().flatten(-2)], dim=-1)

        n_pts = self.offsets.shape[0]
        c_map = C_LIVE + C_STATIC
        if "grid" in b and "static_grid" in b and "origin" in b and "center" in b:
            st = b["static_grid"].to(dt)
            st = torch.cat([st[..., :2] / 255.0, st[..., 2:]], dim=-1).permute(
                0, 3, 1, 2
            )
            gr = b["grid"][:, f_idx]
            gh, gw = gr.shape[2:4]
            gr = gr.reshape(B * nf, gh, gw, C_LIVE).permute(0, 3, 1, 2).to(dt)
            off = (b["center"] - b["origin"])[:, None, None, None, :].to(pos.dtype)
            wh = torch.tensor([gw, gh], device=dev, dtype=pos.dtype)
            # [B, nf, K, 5, 2] normalized to [-1, 1] for grid_sample
            at = (((pos.unsqueeze(-2) + self.offsets + off) / wh) * 2.0 - 1.0).to(dt)
            samp_live = (
                F.grid_sample(
                    gr,
                    at.reshape(B * nf, K, n_pts, 2),
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .reshape(B, nf, K, n_pts, C_LIVE)
            )
            if owner is not None:
                keep_p1 = (owner == E.P1).to(dt)[..., None, None]
                keep_p2 = (owner == E.P2).to(dt)[..., None, None]
                samp_live = torch.cat(
                    [samp_live[..., 0:5] * keep_p1, samp_live[..., 5:10] * keep_p2],
                    dim=-1,
                )
            samp_live = samp_live.reshape(B, nf, K, n_pts * C_LIVE)
            samp_stat = (
                F.grid_sample(
                    st,
                    at.reshape(B, nf * K, n_pts, 2),
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .reshape(B, nf, K, n_pts * C_STATIC)
            )
            sampled = torch.cat([samp_live, samp_stat], dim=-1)
        else:
            sampled = torch.zeros(B, nf, K, n_pts * c_map, device=dev, dtype=dt)

        inp = torch.cat([fourier.to(dt), sampled], dim=-1)
        return self.net(inp)


def _rotary_qkv_core(x, w_n, b_n, w_q, w_k, w_v, cs_q, cs_k, cs_v, H: int, KV: int, dh: int):
    """Fused pre-norm + QKV projections + 1D/2D RoPE rotation + L2 normalization."""
    B, T, d = x.shape
    y = F.layer_norm(x, (d,), w_n, b_n)
    q = F.linear(y, w_q).view(B, T, H, dh).transpose(1, 2)
    q = F.normalize(rotate(q, cs_q), dim=-1)

    k = F.linear(y, w_k).view(B, T, KV, dh).transpose(1, 2)
    k = F.normalize(rotate(k, cs_k), dim=-1).repeat_interleave(H // KV, 1)

    v = F.linear(y, w_v).view(B, T, KV, dh).transpose(1, 2)
    v = rotate(v, cs_v).repeat_interleave(H // KV, 1)
    return q, k, v


def _rotary_qkv(x, w_n, b_n, w_q, w_k, w_v, cs_q, cs_k, cs_v, scale, H: int, KV: int, dh: int):
    q, k, v = _compiled_qkv_core(
        x, w_n, b_n, w_q, w_k, w_v, cs_q, cs_k, cs_v, H, KV, dh
    )
    q = q * (scale.exp().to(q.dtype) * (dh**0.5))
    return q, k, v


def _rotary_out_ff(x, a, cs, w_o, w_fn, b_fn, w1, w2, w3):
    """Fused inverse RoPE rotation + output projection + residual + pre-norm SwiGLU."""
    d = x.shape[-1]
    dt = x.dtype
    a_rot = rotate(a, cs, back=True).transpose(1, 2).flatten(2)
    x = (x + F.linear(a_rot, w_o)).to(dt)
    y = F.layer_norm(x, (d,), w_fn, b_fn)
    return (x + F.linear(F.silu(F.linear(y, w1)) * F.linear(y, w2), w3)).to(dt)


def _rotary_out_ff_masked(x_t, a_t, p_t, t_mask, w_o, w_fn, b_fn, w1, w2, w3):
    """Fused inverse RoPE rotation + masked output residual + masked pre-norm SwiGLU."""
    d = x_t.shape[-1]
    dt = x_t.dtype
    a_rot = rotate(a_t, p_t, back=True).transpose(1, 2).flatten(2)
    x_t = torch.where(t_mask, (x_t + F.linear(a_rot, w_o)).to(dt), x_t)
    y = F.layer_norm(x_t, (d,), w_fn, b_fn)
    ff_out = (x_t + F.linear(F.silu(F.linear(y, w1)) * F.linear(y, w2), w3)).to(dt)
    return torch.where(t_mask, ff_out, x_t)


_compiled_qkv_core = _rotary_qkv_core
_compiled_out_ff = _rotary_out_ff
_compiled_out_ff_masked = _rotary_out_ff_masked


class SpatioTemporalBlock(nn.Module):
    """Pre-norm spatiotemporal block over entities and continuous potential field tokens:
    1. Spatial self-attention with 2D RoPE (rotating Q, K, V and inverse-rotating output)
       followed by SwiGLU feed-forward.
    2. Causal temporal self-attention along each persistent token's timeline with 1D RoPE
       over game loops (and 2D spatial rotation on values across frames) followed by SwiGLU."""

    def __init__(self, d, heads=4, kv_heads=2, ff=None):
        super().__init__()
        ff = ff or 3 * d
        self.ns = nn.LayerNorm(d)
        self.nt = nn.LayerNorm(d)
        self.space = Heads(d, heads, kv_heads)
        self.time = Heads(d, heads, kv_heads)
        self.ff_space = FeedForward(d, ff)
        self.ff_time = FeedForward(d, ff)

    def forward(self, x, valid, cs_2d, allow_t, t_mask, cs_1d, p_t):
        """x: [B * nf, T, d], valid: [B * nf, 1, 1, T], cs_2d: [B * nf, T, dh/2, 2],
        allow_t: [B * T_t, 1, nf, nf], t_mask: [B * T_t, nf, 1],
        cs_1d: [B * T_t, nf, dh/2, 2], p_t: [B * T_t, nf, dh/2, 2]."""
        d = x.shape[2]
        nf = cs_1d.shape[1]
        B = x.shape[0] // nf
        T_t = cs_1d.shape[0] // B

        # 1. 2D-RoPE Spatial self-attention across all tokens (globals, entities, field, effects)
        sp, ffs = self.space, self.ff_space
        qs, ks, vs = _rotary_qkv(
            x,
            self.ns.weight,
            self.ns.bias,
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
        a_s = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=valid)
        x = _compiled_out_ff(
            x,
            a_s,
            cs_2d,
            sp.o.weight,
            ffs.norm.weight,
            ffs.norm.bias,
            ffs.w1.weight,
            ffs.w2.weight,
            ffs.w3.weight,
        )

        # 2. 1D-RoPE Causal Temporal self-attention on persistent tokens [:T_t] only
        x_p = x[:, :T_t]
        x_t = x_p.view(B, nf, T_t, d).transpose(1, 2).reshape(B * T_t, nf, d)
        tm, fft = self.time, self.ff_time
        qt, kt, vt = _rotary_qkv(
            x_t,
            self.nt.weight,
            self.nt.bias,
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
        a_t = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=allow_t)
        x_t = _compiled_out_ff_masked(
            x_t,
            a_t,
            p_t,
            t_mask,
            tm.o.weight,
            fft.norm.weight,
            fft.norm.bias,
            fft.w1.weight,
            fft.w2.weight,
            fft.w3.weight,
        )
        x_p = x_t.view(B, T_t, nf, d).transpose(1, 2).reshape(B * nf, T_t, d)
        return torch.cat([x_p, x[:, T_t:]], dim=1)


class Frames(nn.Module):
    """Encodes collated physics batches into fixed-dimension Gram phase-space States."""

    def __init__(
        self,
        vocab,
        tables,
        d,
        u_dim,
        rules=None,
        r=E.R_LATENT,
        attn_layers=6,
        field_grid=8,
        dh=64,
        ff=None,
    ):
        super().__init__()
        self.tables = tables
        self.r = r
        self.u_dim = u_dim
        self.register_buffer("k_type", tables.k)
        self.register_buffer("race", tables.race)
        self.register_buffer("unit_feat", tables.unit_feat)
        self.register_buffer("abil_feat", tables.abil_feat(vocab))
        self.register_buffer("all_k_type", tables.all_k, persistent=False)
        self.register_buffer("all_unit_feat", tables.all_unit_feat, persistent=False)
        self.register_buffer("all_abil_feat", tables.all_abil_feat(vocab), persistent=False)
        self.register_buffer("is_struct_type", tables.is_struct, persistent=False)
        self.register_buffer("extras_at", torch.tensor([I[n] for n in EXTRAS]))
        self.register_buffer(
            "map_omega",
            2.0
            * math.pi
            / torch.logspace(math.log10(4.0), math.log10(256.0), MAP_WAVES),
            persistent=False,
        )
        self.register_buffer(
            "eff_omega",
            2.0
            * math.pi
            / torch.logspace(math.log10(4.0), math.log10(1024.0), EFF_WAVES),
            persistent=False,
        )

        # Global (CLS + 2 player) token encoders (matching entity_encoder.GlobalEncoder)
        self.cls_tok = nn.Parameter(torch.zeros(1, d))
        nn.init.normal_(self.cls_tok, std=0.02)
        self.view_in = nn.Linear(2, d)
        self.map_wh = nn.Linear(2, d)
        self.race_emb = nn.Embedding(N_RACES, d)
        self.ply_in = E.mlp(C_PLY, d, d)
        self.player_emb = nn.Embedding(2, d)

        # Learned unit/ability embeddings + light MLPs over static ruleset features
        self.unit_emb = nn.Embedding(N_RAW_TYPES, d)
        nn.init.normal_(self.unit_emb.weight, std=0.02)
        self.unit_mlp = E.mlp(tables.unit_feat_dim, d, d)
        self.abil_emb = nn.Embedding(vocab.n_abilities, d, padding_idx=0)
        nn.init.normal_(self.abil_emb.weight, std=0.02)
        with torch.no_grad():
            self.abil_emb.weight[0].zero_()
        self.abil_mlp = E.mlp(tables.abil_feat_dim, d, d)

        # Entity token encoders (with Fourier-transformed map-unit positions)
        self.unit_c = nn.Linear(d, d)
        self.abil_c = nn.Linear(d, d)
        self.prod_c = nn.Linear(d, d)
        self.owner_emb = nn.Embedding(E.OWNERS, d)
        self.kin_c = nn.Linear(E.K_Q + 4 * MAP_WAVES + 2 + 4, d)
        self.extras = E.mlp(len(EXTRAS) + 7, d, d)
        # Control-group membership (0..9 + active selection 10) + F2_ONSCREEN bit (zero-init for exact warm-start)
        self.grp_c = nn.Linear(12, d, bias=False)
        nn.init.zeros_(self.grp_c.weight)

        # Effect token encoders (matching entity_encoder.EffectEncoder)
        self.eff_type = nn.Embedding(N_EFFECT_IDS, d)
        self.eff_num = nn.Linear(2 + 2 * EFF_WAVES, d)

        # Continuous localized potential field (evaluated at entities + anchor grid)
        self.field = ContinuousPotentialField(d, grid_size=field_grid)

        # 2D Spatial RoPE (map units) and 1D Temporal RoPE (game loops)
        heads = max(1, d // dh)
        kv_heads = max(1, heads // 2)
        dh = d // heads
        assert dh % 4 == 0, f"head dim dh={dh} must be divisible by 4 for 2D RoPE"
        self.rope_2d = RoPE2D(dh, lo=ROPE_SHORTEST, hi=256.0)
        self.rope_1d = RoPE1D(dh)

        # 6 Spatiotemporal blocks across globals, entities, and active effects (zero discrete map tokens)
        self.blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(d, heads=heads, kv_heads=kv_heads, ff=ff)
                for _ in range(attn_layers)
            ]
        )
        self.z_norm = nn.LayerNorm(d)
        self.z_proj = nn.Linear(d, r)
        E.small_init(self.z_proj, std=2e-2)
        with torch.no_grad():
            self.z_proj.bias.fill_(-1.0)  # sigmoid(-1.0) ≈ 0.27 at init for non-negative modes
        # Learned positive per-entity amplitude (mass/weight) m_u > 0 from static unit ruleset
        # embedding + contextual token, avoiding equal-norm clamping across Probes and high-tier units
        self.mass_type = nn.Linear(d, 1, bias=False)
        self.mass_ctx = nn.Linear(d, 1, bias=True)
        E.small_init(self.mass_type, std=1e-2)
        E.small_init(self.mass_ctx, std=1e-2)
        with torch.no_grad():
            self.mass_ctx.bias.fill_(math.log(math.e - 1.0))  # softplus(0.5413) == 1.0 at init

        self.z_ply_norm = nn.LayerNorm(d)
        self.z_ply_proj = nn.Linear(d, r)
        E.small_init(self.z_ply_proj, std=1e-2)
        self.grad_ckpt = True

        # Command port encoder (per-row order changes + raw action tokens)
        self.abil_u = nn.Linear(d, u_dim, bias=False)
        self.reg_u = nn.Linear(4, u_dim, bias=False)
        self.act_kind_u = nn.Embedding(16, u_dim, padding_idx=0)
        self.act_num_u = nn.Linear(14, u_dim, bias=False)

    def compile_parts(self):
        """Compiles the shared rotary QKV and SwiGLU kernels once across all 6 blocks."""
        global _compiled_qkv_core, _compiled_out_ff, _compiled_out_ff_masked
        torch._dynamo.config.recompile_limit = 256
        torch._functorch.config.donated_buffer = False
        _compiled_qkv_core = torch.compile(_rotary_qkv_core, dynamic=True)
        _compiled_out_ff = torch.compile(_rotary_out_ff, dynamic=True)
        _compiled_out_ff_masked = torch.compile(_rotary_out_ff_masked, dynamic=True)
        self.field.net.compile(dynamic=True)
        self.extras.compile(dynamic=True)

    def emb_norms(self):
        """RMS norms of the learned unit and ability embeddings."""
        return {
            "unit": float(self.unit_emb.weight.detach().float().pow(2).mean().sqrt()),
            "ability": float(self.abil_emb.weight.detach().float().pow(2).mean().sqrt()),
        }

    def nodes(self, rtype, abil, prod_abil, rs_id=None):
        """Evaluates learned embeddings + light MLPs on the static ruleset tables once per
        pass (indexing exact per-game rulesets when `rs_id` is supplied) and gathers."""
        dt = _compute_dtype(self.cls_tok)
        if rs_id is not None and hasattr(self, "all_unit_feat"):
            B, nf, R = rtype.shape
            d = self.cls_tok.shape[-1]
            n_rs = self.all_unit_feat.shape[0]
            ri = torch.where(rs_id >= 0, rs_id, 0).clamp(0, n_rs - 1)  # [B]
            uf_b = self.all_unit_feat[ri].to(dt)  # [B, N_RAW_TYPES, uf_dim]
            af_b = self.all_abil_feat[ri].to(dt)  # [B, n_abil, af_dim]
            u_tab = self.unit_emb.weight.to(dt)[None] + self.unit_mlp(uf_b)
            a_tab = self.abil_emb.weight.to(dt)[None] + self.abil_mlp(af_b)
            u_c_b = self.unit_c(u_tab)  # [B, N_RAW_TYPES, d]
            z_a = torch.zeros(B, 1, d, device=a_tab.device, dtype=dt)
            a_c_b = torch.cat([z_a, self.abil_c(a_tab[:, 1:])], dim=1)
            a_p_b = torch.cat([z_a, self.prod_c(a_tab[:, 1:])], dim=1)
            z_u = torch.zeros(B, 1, self.u_dim, device=a_tab.device, dtype=dt)
            a_u_b = torch.cat([z_u, self.abil_u(a_tab[:, 1:])], dim=1)

            r_idx = rtype.clamp(0, u_c_b.shape[1] - 1).reshape(B, nf * R, 1)
            a_idx = abil.clamp(0, a_c_b.shape[1] - 1).reshape(B, nf * R, 1)
            p_idx = prod_abil.clamp(0, a_p_b.shape[1] - 1).reshape(B, nf * R, 1)

            unit_c = torch.gather(u_c_b, 1, r_idx.expand(-1, -1, d)).view(B, nf, R, d)
            ord_c = torch.gather(a_c_b, 1, a_idx.expand(-1, -1, d)).view(B, nf, R, d)
            prod_c = torch.gather(a_p_b, 1, p_idx.expand(-1, -1, d)).view(B, nf, R, d)
            ord_u = torch.gather(a_u_b, 1, a_idx.expand(-1, -1, self.u_dim)).view(
                B, nf, R, self.u_dim
            )
            return unit_c, ord_c, prod_c, ord_u

        u_tab = self.unit_emb.weight.to(dt) + self.unit_mlp(self.unit_feat.to(dt))
        a_tab = self.abil_emb.weight.to(dt) + self.abil_mlp(self.abil_feat.to(dt))
        u_c = self.unit_c(u_tab)
        z_a = torch.zeros(1, a_tab.shape[1], device=a_tab.device, dtype=dt)
        a_c = torch.cat([z_a, self.abil_c(a_tab[1:])], dim=0)
        a_p = torch.cat([z_a, self.prod_c(a_tab[1:])], dim=0)
        z_u = torch.zeros(1, self.u_dim, device=a_tab.device, dtype=dt)
        a_u = torch.cat([z_u, self.abil_u(a_tab[1:])], dim=0)
        r_idx = rtype.clamp(0, len(u_c) - 1)
        a_idx = abil.clamp(0, len(a_c) - 1)
        p_idx = prod_abil.clamp(0, len(a_p) - 1)
        unit_c = F.embedding(r_idx, u_c)
        ord_c = F.embedding(a_idx, a_c, padding_idx=0)
        prod_c = F.embedding(p_idx, a_p, padding_idx=0)
        ord_u = F.embedding(a_idx, a_u, padding_idx=0)
        return unit_c, ord_c, prod_c, ord_u

    def project_entities(self, c_ent_flat, rtype, mask, rs_id=None):
        """Projects contextualized entity tokens c_ent_flat [M, R, d] to amplitude-scaled
        latent modes z_flat [M, R, r] = m_u * softsign(W_z LayerNorm(c_u)), where m_u > 0
        is anchored to the unit's physical log-cost log(1 + value_u / 100) + 0.5 and
        modulated by learned static ruleset + contextual factors."""
        M, R, _ = c_ent_flat.shape
        dt = c_ent_flat.dtype
        m_flat = mask.reshape(M, R, 1).to(dt)
        c_normed = self.z_norm(c_ent_flat) * m_flat

        if rs_id is not None and hasattr(self, "all_unit_feat"):
            B = rs_id.shape[0]
            nf = M // B
            ri = torch.where(rs_id >= 0, rs_id, 0).clamp(0, self.all_unit_feat.shape[0] - 1)
            u_tab_b = self.unit_emb.weight.to(dt)[None] + self.unit_mlp(self.all_unit_feat[ri].to(dt))
            m_tab_b = self.mass_type(u_tab_b)  # [B, N_RAW_TYPES, 1]
            r_idx_b = rtype.reshape(B, nf * R, 1).clamp(0, m_tab_b.shape[1] - 1)
            m_type = torch.gather(m_tab_b, 1, r_idx_b).reshape(M, R, 1)
        else:
            u_tab = self.unit_emb.weight.to(dt) + self.unit_mlp(self.unit_feat.to(dt))
            m_type_tab = self.mass_type(u_tab)  # [N_RAW_TYPES, 1]
            r_idx = rtype.reshape(M, R).clamp(0, len(m_type_tab) - 1)
            m_type = F.embedding(r_idx, m_type_tab)  # [M, R, 1]

        r_idx = rtype.reshape(M, R).clamp(0, len(self.k_type) - 1)
        # Physical unit cost anchor from self.k_type[:, 0] (minerals + vespene)
        cost_u = self.k_type[r_idx, 0:1].to(dt).clamp(min=0.0)
        m_base = torch.log1p(cost_u / 100.0) + 0.5  # [M, R, 1] >= 0.5

        m_u = m_base * F.softplus(m_type + self.mass_ctx(c_normed)) * m_flat  # [M, R, 1] > 0
        z_dir = torch.sigmoid(self.z_proj(c_normed)) * m_flat  # [M, R, r] >= 0 (non-negative modes)
        z_flat = m_u * z_dir
        return z_flat, c_normed

    def coords(self, b, ff):
        """[B, nf, R, K_Q] grounded per-unit coordinates in full fp32:
        (1, x_norm, y_norm, vx_norm, vy_norm, hp_frac, shield_frac, energy_frac),
        where (x_norm, y_norm) are normalized by the fixed physical MAP_SCALE = 100.0
        map units so spatial wavelengths are isotropic and map-size independent."""
        with torch.autocast(device_type=b["x"].device.type, enabled=False):
            x = b["x"][:, ff].float()
            y = b["y"][:, ff].float()
            num = b["num"][:, ff].float()
            px = b["x"][:, ff - 1].float()
            py = b["y"][:, ff - 1].float()
            both = (b["mask"][:, ff] & b["mask"][:, ff - 1]).float()

            if "center" in b:
                cx = b["center"][:, 0, None, None].float()
                cy = b["center"][:, 1, None, None].float()
            else:
                cx, cy = 100.0, 100.0

            if "loops" in b:
                f_t = torch.as_tensor(ff, device=x.device)
                f_p = (f_t - 1).clamp(min=0)
                dt_loops = (
                    (b["loops"][:, f_t] - b["loops"][:, f_p])
                    .clamp(min=1)
                    .float()
                    .unsqueeze(-1)
                )
            else:
                dt_loops = 1.0

            x_norm = ((x - cx) / E.MAP_SCALE).clamp(-2.0, 2.0)
            y_norm = ((y - cy) / E.MAP_SCALE).clamp(-2.0, 2.0)
            vx_norm = (((x - px) * both) / (E.SPEED_SCALE * dt_loops)).clamp(-4.0, 4.0)
            vy_norm = (((y - py) * both) / (E.SPEED_SCALE * dt_loops)).clamp(-4.0, 4.0)

            hp_max = num[..., I["hp_max"]]
            sh_max = num[..., I["shield_max"]]
            hp_frac = (num[..., I["hp"]] / hp_max.clamp(min=1.0)) * (hp_max > 0)
            sh_frac = (num[..., I["shield"]] / sh_max.clamp(min=1.0)) * (sh_max > 0)
            en_frac = (num[..., I["energy"]] / E.ENERGY_UNIT).clamp(0.0, 1.5)
            ones = torch.ones_like(x_norm)

            return torch.stack(
                [ones, x_norm, y_norm, vx_norm, vy_norm, hp_frac, sh_frac, en_frac], dim=-1
            )

    def ledger(self, b, ff):
        """[B, nf, 2, 2] per-player (minerals, vespene) / 1000 in full fp32."""
        with torch.autocast(device_type=b["ply_num"].device.type, enabled=False):
            pl = b["ply_num"].float()
            v = torch.expm1(pl[:, ff, :, : len(PC_KEYS)] * 8.0)
            mins = v[..., PC["minerals"]] / E.BANK_SCALE
            vesp = v[..., PC["vespene"]] / E.BANK_SCALE
            return torch.stack([mins, vesp], dim=-1)

    def _embed_globals(self, b, f_idx):
        """[B, nf, 3, d] global tokens: [CLS, Player 1, Player 2]."""
        B = b["mask"].shape[0]
        nf = len(f_idx)
        dev = b["mask"].device
        dt = _compute_dtype(self.cls_tok)
        pl_num = b["ply_num"][:, f_idx, :, :C_PLY].to(dt)
        if "players" in b:
            races = b["players"].clamp(0, N_RACES - 1)
        else:
            races = torch.zeros(B, 2, dtype=torch.long, device=dev)
        p_tok = (
            self.ply_in(pl_num)
            + self.race_emb(races)[:, None].to(dt)
            + self.player_emb.weight[None, None].to(dt)
        )
        view = (
            b["view"].to(dt)
            if "view" in b
            else torch.ones(B, 2, device=dev, dtype=dt)
        )
        half = (
            (b["half"] / 100.0).to(dt)
            if "half" in b
            else torch.ones(B, 2, device=dev, dtype=dt)
        )
        cls = (self.cls_tok.to(dt) + self.view_in(view) + self.map_wh(half))[
            :, None, None, :
        ].expand(B, nf, 1, -1)
        return torch.cat([cls, p_tok], dim=2)

    def _embed_effects(self, b, f_idx):
        """[B, nf, X, d], [B, nf, X], [B, nf, X, 2] effect tokens, mask, and map-unit xy."""
        B = b["mask"].shape[0]
        nf = len(f_idx)
        dev = b["mask"].device
        dt = _compute_dtype(self.cls_tok)
        if "eff_mask" not in b or b["eff_mask"].shape[2] == 0:
            return (
                torch.zeros(B, nf, 0, self.cls_tok.shape[-1], device=dev, dtype=dt),
                torch.zeros(B, nf, 0, dtype=torch.bool, device=dev),
                torch.zeros(B, nf, 0, 2, device=dev, dtype=torch.float32),
            )
        eff_mask = b["eff_mask"][:, f_idx]
        eff_type = b["eff_type"][:, f_idx].clamp(0, N_EFFECT_IDS - 1)
        eff_own_raw = b["eff_owner"][:, f_idx]
        eff_own = torch.where(eff_own_raw == NEUTRAL, E.GROUND, eff_own_raw + 1).clamp(
            0, E.OWNERS - 1
        )
        rad = b["eff_radius"][:, f_idx].float()
        age = (
            b["eff_age"][:, f_idx].float()
            if "eff_age" in b
            else torch.zeros_like(rad)
        )
        a = age.unsqueeze(-1) * self.eff_omega
        eff_feat = torch.cat(
            [rad.unsqueeze(-1), torch.log1p(age).unsqueeze(-1) / 8.0, a.sin(), a.cos()],
            dim=-1,
        ).to(dt)
        eff_tok = (
            self.eff_type(eff_type).to(dt)
            + self.owner_emb(eff_own).to(dt)
            + self.eff_num(eff_feat)
        )
        eff_xy = b["eff_xy"][:, f_idx].float()
        return eff_tok, eff_mask, eff_xy

    def _encode_sequence(self, b, f_idx):
        """Encodes frame indices f_idx -> (q, z, z_ply, owner, mask, p, u, u_active)."""
        f_idx = torch.as_tensor(f_idx, device=b["mask"].device)
        B, F_total, R = b["mask"].shape
        nf = len(f_idx)

        q = self.coords(b, f_idx)  # [B, nf, R, K_Q]
        p = self.ledger(b, f_idx)  # [B, nf, 2, 2]

        rtype = b["rtype"][:, f_idx].clamp(0, N_RAW_TYPES - 1)
        owner = b["owner"][:, None].expand(B, nf, R)
        owner = torch.where(owner == NEUTRAL, E.GROUND, owner + 1)
        on_map = b["mask"][:, f_idx] & (b["inside"][:, f_idx] == 0)
        mask = on_map & b["alive"][:, f_idx]

        num = b["num"][:, f_idx]
        rs_id = b.get("ruleset_id", None)
        if rs_id is not None and hasattr(self, "all_k_type"):
            ri = torch.where(rs_id >= 0, rs_id, 0).clamp(0, self.all_k_type.shape[0] - 1)
            kt_b = self.all_k_type[ri]  # [B, N_RAW_TYPES, 7]
            kt = torch.gather(
                kt_b, 1, rtype.reshape(B, nf * R, 1).expand(-1, -1, 7)
            ).view(B, nf, R, 7)
        else:
            kt = self.k_type[rtype]  # [B, nf, R, 7]
        kt_comp = torch.sign(kt) * torch.log1p(kt.abs()) / 8.0

        abil = b["slot_abil"][:, f_idx, :, 0]
        prod_abil = b["slot_abil"][:, f_idx, :, QUEUE]

        extras = num[..., self.extras_at]
        extras = torch.sign(extras) * torch.log1p(extras.abs()) / 8.0
        ext_in = torch.cat([extras, kt_comp], dim=-1)

        # Fourier-transformed map-unit coordinates (wavelengths 4 to 256 map units)
        xy_raw = torch.stack([b["x"][:, f_idx], b["y"][:, f_idx]], dim=-1).float()
        xy_ang = xy_raw.unsqueeze(-1) * self.map_omega  # [B, nf, R, 2, MAP_WAVES]
        xy_fourier = torch.cat(
            [xy_ang.sin().flatten(-2), xy_ang.cos().flatten(-2)], dim=-1
        ).to(q.dtype)

        facing = num[..., I["facing"]]
        ord_reg = b["slot_reg"][:, f_idx, :, 0, :]
        kin_in = torch.cat(
            [
                q,
                xy_fourier,
                facing.sin().unsqueeze(-1),
                facing.cos().unsqueeze(-1),
                ord_reg,
            ],
            dim=-1,
        )

        # Control groups [B, nf, R, 11] + F2_ONSCREEN (bit 4 of flags2)
        onscreen = ((num[..., I["flags2"]].long() & 4) != 0).to(q.dtype).unsqueeze(-1)
        if "groups" in b:
            grp_in = torch.cat([b["groups"][:, f_idx].to(q.dtype), onscreen], dim=-1)
        else:
            grp_in = torch.cat(
                [torch.zeros(B, nf, R, 11, device=q.device, dtype=q.dtype), onscreen],
                dim=-1,
            )

        # Continuous localized potential field evaluated strictly at entity coordinates (no discrete map tokens)
        dev = b["mask"].device
        if "center" in b:
            center = b["center"][:, None, None, :].float()
        else:
            center = torch.full((B, 1, 1, 2), 100.0, device=dev)
        pos_ent = xy_raw - center
        ckpt = self.grad_ckpt and torch.is_grad_enabled()

        perc = extract_entity_perception(
            b, f_idx, getattr(self, "is_struct_type", None)
        )

        def _embed_x():
            unit_c, ord_c, prod_c, _ = self.nodes(rtype, abil, prod_abil, rs_id=rs_id)
            field_ent = self.field(b, f_idx, pos_ent, owner=owner)
            dt = _compute_dtype(self.cls_tok)
            c_ent = (
                unit_c
                + ord_c
                + prod_c
                + self.owner_emb(owner).to(dt)
                + self.kin_c(kin_in)
                + self.extras(ext_in)
                + self.grp_c(grp_in.to(dt))
                + field_ent
            ).to(dt)
            g_tok = self._embed_globals(b, f_idx).to(dt)
            return torch.cat(
                [g_tok, c_ent, eff_tok.to(dt)], dim=2
            ).reshape(B * nf, T, -1)

        eff_tok, eff_mask, eff_xy = self._embed_effects(b, f_idx)  # [B, nf, X, d]
        g0, X = 3, eff_tok.shape[2]
        T_t = g0 + R
        T = T_t + X

        if ckpt:
            x_tok = checkpoint(_embed_x, use_reentrant=False)
        else:
            x_tok = _embed_x()
        valid_g = torch.ones(B, nf, g0, dtype=torch.bool, device=dev)
        valid_all = torch.cat([valid_g, mask, eff_mask], dim=2)

        # Perspective-isolated spatial attention mask [B * nf, 1, T, T]:
        # - CLS (0) is omniscient and attends to all valid tokens; no token attends to CLS.
        # - P1 (1) and P1-owned entities attend strictly to P1 + P1-visible neutrals/effects.
        # - P2 (2) and P2-owned entities attend strictly to P2 + P2-visible neutrals/effects.
        # - Neutrals and effects attend only to themselves (zero 2-hop cross-player relay!).
        p1_ent_k = mask & ((owner == E.P1) | ((owner == E.GROUND) & perc["g_vis"][:, :, 0]))
        p2_ent_k = mask & ((owner == E.P2) | ((owner == E.GROUND) & perc["g_vis"][:, :, 1]))
        p1_eff_k = eff_mask & perc["eff_vis"][:, :, 0]
        p2_eff_k = eff_mask & perc["eff_vis"][:, :, 1]

        z_bool = torch.zeros(B, nf, 1, dtype=torch.bool, device=dev)
        o_bool = torch.ones(B, nf, 1, dtype=torch.bool, device=dev)
        p1_keys = torch.cat([z_bool, o_bool, z_bool, p1_ent_k, p1_eff_k], dim=2)  # [B, nf, T]
        p2_keys = torch.cat([z_bool, z_bool, o_bool, p2_ent_k, p2_eff_k], dim=2)  # [B, nf, T]

        q_is_cls = torch.zeros(B, nf, T, dtype=torch.bool, device=dev)
        q_is_cls[:, :, 0] = True
        q_is_p1 = torch.cat(
            [
                z_bool,
                o_bool,
                z_bool,
                owner == E.P1,
                torch.zeros(B, nf, X, dtype=torch.bool, device=dev),
            ],
            dim=2,
        )
        q_is_p2 = torch.cat(
            [
                z_bool,
                z_bool,
                o_bool,
                owner == E.P2,
                torch.zeros(B, nf, X, dtype=torch.bool, device=dev),
            ],
            dim=2,
        )
        eye_T = torch.eye(T, dtype=torch.bool, device=dev)[None, None]
        allow_s = (
            (q_is_cls.unsqueeze(-1) & valid_all.unsqueeze(-2))
            | (q_is_p1.unsqueeze(-1) & p1_keys.unsqueeze(-2))
            | (q_is_p2.unsqueeze(-1) & p2_keys.unsqueeze(-2))
            | eye_T
        )
        valid_flat = allow_s.reshape(B * nf, 1, T, T)

        t_valid = torch.cat([valid_g, mask], dim=2)
        t_mask = t_valid.transpose(1, 2).reshape(B * T_t, nf)
        tri = torch.tril(torch.ones(nf, nf, dtype=torch.bool, device=dev))
        eye = torch.eye(nf, dtype=torch.bool, device=dev)
        allow_t = ((tri[None] & t_mask[:, None, :]) | eye[None])[:, None]
        t_mask_3d = t_mask.unsqueeze(-1)

        # Global tokens sit at origin (0, 0) unrotated; entities and effects at map-unit pos
        pos_g = torch.zeros(B, nf, g0, 2, device=dev, dtype=torch.float32)
        pos_all = torch.cat([pos_g, pos_ent, eff_xy], dim=2)

        # 2D Spatial RoPE (map units) and 1D Temporal RoPE (game loops), shared across blocks
        cs_2d = cos_sin(self.rope_2d.angles(pos_all.reshape(B * nf, T, 2)))
        h2 = cs_2d.shape[-2]
        p_t = (
            cs_2d[:, :T_t]
            .view(B, nf, T_t, h2, 2)
            .transpose(1, 2)
            .reshape(B * T_t, nf, h2, 2)
        )
        if "loops" in b:
            loops = b["loops"][:, f_idx] - b["loops"][:, f_idx[:1]]
        else:
            loops = torch.arange(nf, device=dev)[None].expand(B, -1)
        t_cs = cos_sin(self.rope_1d.angles(loops))
        cs_1d = (
            t_cs[:, None, :, :, :]
            .expand(B, T_t, nf, h2, 2)
            .reshape(B * T_t, nf, h2, 2)
        )

        ckpt = self.grad_ckpt and torch.is_grad_enabled()
        for blk in self.blocks:
            if ckpt:
                x_tok = checkpoint(
                    blk,
                    x_tok,
                    valid_flat,
                    cs_2d,
                    allow_t,
                    t_mask_3d,
                    cs_1d,
                    p_t,
                    use_reentrant=True,
                )
            else:
                x_tok = blk(
                    x_tok,
                    valid_flat,
                    cs_2d,
                    allow_t,
                    t_mask_3d,
                    cs_1d,
                    p_t,
                )

        cls_out = x_tok[:, 0:1]  # [B * nf, 1, d]
        ply_out = x_tok[:, 1:3]  # [B * nf, 2, d]
        z_ply_flat = F.softsign(
            self.z_ply_proj(self.z_ply_norm(ply_out + cls_out))
        )  # [B * nf, 2, r]

        c_ent_raw = x_tok[:, g0 : g0 + R]
        z_flat, c_ent_normed = self.project_entities(c_ent_raw, rtype, mask, rs_id=rs_id)
        u, u_active, q_changed = self._encode_commands(
            b, f_idx, abil, prod_abil, ord_reg, owner, mask, return_changed=True
        )
        if getattr(self, "_return_ctx", False):
            return dict(
                q=q,
                z=z_flat.view(B, nf, R, self.r),
                z_ply=z_ply_flat.view(B, nf, 2, self.r),
                c_ent=c_ent_normed.view(B, nf, R, -1),
                ply_out=ply_out.view(B, nf, 2, -1),
                cls_out=cls_out.view(B, nf, -1),
                pos_ent=pos_ent,
                rtype=rtype,
                num=num,
                owner=owner,
                mask=mask,
                p=p,
                u=u,
                u_active=u_active,
                q_changed=q_changed,
                perception=perc,
            )
        return (
            q,
            z_flat.view(B, nf, R, self.r),
            z_ply_flat.view(B, nf, 2, self.r),
            owner,
            mask,
            p,
            u,
            u_active,
        )

    def encode_window_context(self, b, f_idx):
        """Encodes frame indices f_idx and returns full unprojected contextualized
        entity, player, and scene representations alongside grounded coordinates."""
        prev = getattr(self, "_return_ctx", False)
        self._return_ctx = True
        try:
            return self._encode_sequence(b, f_idx)
        finally:
            self._return_ctx = prev

    def _encode_commands(
        self, b, f_idx, abil, prod_abil, ord_reg, owner, mask, return_changed=False
    ):
        """Encodes per-player control impulse u [B, nf, 2, u_dim] and u_active [B, nf, 2]
        for the transition f_idx -> f_idx + 1.

        Reads the newly issued order and production queue abilities at frame f_idx + 1
        (excluding passive order completions where the ability slot returns to 0)."""
        F_total = b["mask"].shape[1]
        f_next = (f_idx + 1).clamp(max=F_total - 1)
        has_next = (f_idx + 1 < F_total)[None, :, None]

        abil_next = b["slot_abil"][:, f_next, :, 0]
        prod_next = b["slot_abil"][:, f_next, :, QUEUE]
        reg_next = b["slot_reg"][:, f_next, :, 0, :]

        # Genuine newly issued / retargeted orders (excluding passive completion to 0)
        reg_diff = (reg_next - ord_reg).abs().sum(dim=-1) > 1e-4
        ord_started = ((abil_next != abil) | reg_diff) & (abil_next > 0) & has_next & mask
        prod_started = (prod_next != prod_abil) & (prod_next > 0) & has_next & mask
        q_changed = ord_started | prod_started

        dt_u = _compute_dtype(self.cls_tok)
        a_tab_u = self.abil_emb.weight.to(dt_u) + self.abil_mlp(self.abil_feat.to(dt_u))
        z_u = torch.zeros(1, self.u_dim, device=a_tab_u.device, dtype=dt_u)
        a_u = torch.cat([z_u, self.abil_u(a_tab_u[1:])], dim=0)

        ord_u = F.embedding(abil_next.clamp(0, len(a_u) - 1), a_u, padding_idx=0)
        prod_u = F.embedding(prod_next.clamp(0, len(a_u) - 1), a_u, padding_idx=0)
        row_u = (
            (ord_u + self.reg_u(reg_next.to(dt_u))) * ord_started.unsqueeze(-1).to(dt_u)
            + prod_u * prod_started.unsqueeze(-1).to(dt_u)
        )

        u_list, act_list = [], []
        for pl in (E.P1, E.P2):
            pl_chg = q_changed & (owner == pl)
            u_pl = (row_u * (owner == pl).unsqueeze(-1).to(row_u.dtype)).sum(dim=2)
            active_pl = pl_chg.any(dim=2)
            if "act_mask" in b and "act_kind" in b:
                # Only physical command events (A_CMD=1, A_REISSUE=2, A_CANCEL=11),
                # excluding passive camera pans and unit selections
                kind = b["act_kind"][:, f_idx]
                is_cmd = (kind == 1) | (kind == 2) | (kind == 11)
                am = (
                    b["act_mask"][:, f_idx]
                    & (b["act_player"][:, f_idx] == (pl - 1))
                    & is_cmd
                )
                ak = self.act_kind_u(kind.clamp(0, 15))
                if "act_num" in b and "act_xy" in b:
                    an_in = torch.cat(
                        [b["act_num"][:, f_idx], b["act_xy"][:, f_idx] / 100.0], dim=-1
                    )
                    ak = ak + self.act_num_u(an_in.to(ak.dtype))
                u_pl = u_pl + (ak * am.unsqueeze(-1).to(ak.dtype)).sum(dim=2) / 10.0
                active_pl = active_pl | am.any(dim=2)
            u_list.append(u_pl / 10.0)
            act_list.append(active_pl)

        u = torch.stack(u_list, dim=2)  # [B, nf, 2, u_dim]
        u_active = torch.stack(act_list, dim=2)  # [B, nf, 2]
        if return_changed:
            return u, u_active, q_changed
        return u, u_active

    def frames(self, b, ff):
        """Encodes transitions from frames ff to ff + 1 -> (State_t, State_next)."""
        ff = list(ff)
        # If ff is contiguous [1, 2, ..., K], encode [1, ..., K + 1] in one pass
        if ff and ff == list(range(ff[0], ff[-1] + 1)):
            all_f = list(range(ff[0], ff[-1] + 2))
            q, z, z_ply, owner, mask, p, u, u_active = self._encode_sequence(b, all_f)
            q_t, q_n = q[:, :-1], q[:, 1:]
            z_t, z_n = z[:, :-1], z[:, 1:]
            zp_t, zp_n = z_ply[:, :-1], z_ply[:, 1:]
            own_t, own_n = owner[:, :-1], owner[:, 1:]
            msk_t, msk_n = mask[:, :-1], mask[:, 1:]
            p_t, p_n = p[:, :-1], p[:, 1:]
            u_t, u_n = u[:, :-1], u[:, 1:]
            act_t, act_n = u_active[:, :-1], u_active[:, 1:]
        else:
            f_next = [f + 1 for f in ff]
            q_t, z_t, zp_t, own_t, msk_t, p_t, u_t, act_t = self._encode_sequence(b, ff)
            q_n, z_n, zp_n, own_n, msk_n, p_n, u_n, act_n = self._encode_sequence(
                b, f_next
            )

        B, nf = q_t.shape[:2]
        flat = lambda t: t.reshape(B * nf, *t.shape[2:])

        s_t = E.State(
            q=flat(q_t).float(),
            z=flat(z_t).float(),
            owner=flat(own_t),
            mask=flat(msk_t),
            p=flat(p_t).float(),
            u=flat(u_t).float(),
            u_active=flat(act_t),
            z_ply=flat(zp_t).float(),
        )
        s_t.x_obs, s_t.x_lat = E.build_gram_state(
            s_t.q, s_t.z, s_t.owner, s_t.mask, s_t.p, s_t.z_ply, r=self.r
        )

        s_next = E.State(
            q=flat(q_n).float(),
            z=flat(z_n).float(),
            owner=flat(own_n),
            mask=flat(msk_n),
            p=flat(p_n).float(),
            u=flat(u_n).float(),
            u_active=flat(act_n),
            z_ply=flat(zp_n).float(),
        )
        s_next.x_obs, s_next.x_lat = E.build_gram_state(
            s_next.q, s_next.z, s_next.owner, s_next.mask, s_next.p, s_next.z_ply, r=self.r
        )
        return s_t, s_next


BATCH_KEYS = (
    "mask",
    "alive",
    "inside",
    "x",
    "y",
    "num",
    "groups",
    "rtype",
    "owner",
    "players",
    "view",
    "ply_num",
    "slot_abil",
    "slot_reg",
    "queue_changed",
    "center",
    "half",
    "side",
    "origin",
    "grid",
    "static_grid",
    "loops",
    "eff_mask",
    "eff_type",
    "eff_owner",
    "eff_xy",
    "eff_radius",
    "eff_age",
    "act_mask",
    "act_player",
    "act_kind",
    "act_group",
    "act_gop",
    "act_num",
    "act_xy",
    "act_placed",
    "act_target",
    "death_frame",
    "parent_row",
    "ruleset_id",
    "birth_count",
    "birth_b",
    "birth_f",
    "birth_r",
    "birth_p",
)


def to_device(b, device):
    """The batch keys the energy model reads, on the device."""
    return {k: b[k].to(device, non_blocking=True) for k in BATCH_KEYS if k in b}
