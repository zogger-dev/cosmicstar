"""Continuous-Field Port-Hamiltonian World Model (Zero-Gram Architecture).

No per-entity prediction heads, no discrete map anchor tokens, no hard-coded sub-system
slots, and zero unlocalized Gram matrices (no Q^T Q, no Q^T Z, no Z^T Z). Instead:

1. **Continuous Localized 2D Polar Gauss-Rayleigh Field**:
   Each live entity u at map position r_u = (x_u, y_u) with velocity v_u = (vx_u, vy_u)
   has a learned positive amplitude/mass m_u > 0 (anchored to unit log-cost) and an
   r-dimensional mode vector z_hat_u in [-1, 1]^r (via unclamped softsign), forming
   z_u = m_u * z_hat_u in R^r.
   Using an isotropic 2D Polar Gauss-Rayleigh quadrature of M_w = 80 wavevectors
   (10 radial rings x 8 golden-ratio staggered orientations), each entity paints a
   localized spatial wave packet onto the continuous 2D map:
       psi(r_u)       = sqrt(w_k) * [cos(k . r_u), sin(k . r_u)]                 in R^{2 M_w}
       psi_dot(r_u)   = sqrt(w_k) * (k . v_u) * [-sin(k . r_u), cos(k . r_u)]    in R^{2 M_w}
   Summing over player p's units yields player p's continuous position field F_p and
   momentum field Pi_p in wave-packet space:
       F_p  = (1 / sqrt(N_0)) sum_{u in P} psi(r_u) z_u^T       in R^{2 M_w x r}
       Pi_p = (1 / sqrt(N_0)) sum_{u in P} psi_dot(r_u) z_u^T   in R^{2 M_w x r}
   By Parseval's theorem, inner products in wave-packet space equal spatial integrals of
   localized fields over the map:
       <F_p, F_q> = (1 / N_0) sum_{u in P, v in Q} K(r_u - r_v) z_u z_v^T

2. **Canonical Configuration (q) and Momentum (p) Phase-Space State**:
   Each player's state x_p = (q_p, p_p) in R^{D_p} (D_p = 150 for r = 8) consists strictly
   of translation-invariant Grounded Physical Totals (`x_obs`, 10 dims) and Continuous
   Spatial Field Integrals (`x_lat`, 140 dims):
   - **Configuration q_p** (d_q = 122):
       * Grounded totals q_obs (6):
           [0] minerals   (bank / 1000)
           [1] vespene    (bank / 1000)
           [2] count      (sum_{u in P} 1 / N_0)
           [3] hp         (sum_{u in P} hp_u / N_0)
           [4] shield     (sum_{u in P} shield_u / N_0)
           [5] energy     (sum_{u in P} energy_u / N_0)
       * Continuous field configuration q_lat (116):
           - F_self   (36): upper triangle of F_p^T F_p (localized self-field energy & synergy)
           - F_cross  (64): full r x r matrix F_p^T F_{-p} (localized P1-P2 contestation)
           - F_env     (8): diag(F_p^T F_env) (localized overlap with neutral resources/map)
           - z_ply     (8): global player + CLS macro/tech state
   - **Momentum p_p** (d_mom = 28 = 18 linear O(v) + 10 quadratic O(v^2)):
       * Grounded motion p_obs (4):
           [6] vx         (sum_{u in P} vx_u / N_0, linear O(v) in p_lin)
           [7] vy         (sum_{u in P} vy_u / N_0, linear O(v) in p_lin)
           [8] vx2        (sum_{u in P} vx_u^2 / N_0, quadratic O(v^2) >= 0 in p_quad)
           [9] vy2        (sum_{u in P} vy_u^2 / N_0, quadratic O(v^2) >= 0 in p_quad)
       * Continuous field momentum & flux p_lat (24):
           - T_field   (8): diag(Pi_p^T Pi_p) >= 0 (localized field kinetic energy, in p_quad)
           - A_self    (8): diag(F_p^T Pi_p) (localized self-compression/expansion flux, in p_lin)
           - A_cross   (8): diag(Pi_p^T F_{-p}) (localized closing/retreating flux vs enemy, in p_lin)

3. **Explicit Potential + Kinetic Hamiltonian H_p(q, p) = V_p(q) + T_p(q, p)**:
   - Potential Energy V_p(q_1, q_2) = V_own(q_{own, p}) + s_p * V_tug(q_1, q_2)
   - Kinetic Energy T_p(q_{own, p}, p_p) >= 0 (purely O(v^2), time-reversal symmetric):
         T_p(q, p) = 0.5 * ||L_T(q_{own, p})^T p_{lin, p}||^2
                   + softplus(w_T(q_{own, p}))^T p_{quad, p} >= 0

4. **Canonical Block-Structured Port-Hamiltonian Dynamics**:
       [ dq ]   [  -R_qq(x)     J_qp(x)  ] [ grad_q V ]   [ B_q(x) ]
       [ dp ] = [ -J_qp(x)^T   -R_pp(x)  ] [ grad_p T ] + [ B_p(x) ] u
   coupled across players by skew-symmetric J_12(x_1, x_2) and dissipative R_12(x_1, x_2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# Per-unit grounded physical columns q_u in R^8 (kept for unit tensor extraction & L2 compat)
Q_COLS = ("1", "x", "y", "vx", "vy", "hp", "shield", "energy")
Q = {name: i for i, name in enumerate(Q_COLS)}
K_Q = len(Q_COLS)  # 8
N_QQ = K_Q * (K_Q + 1) // 2  # 36 (kept for Layer 2 backward compatibility)

POS_COLS = ("x", "y")
VEL_COLS = ("vx", "vy")
VAL_COLS = ("1", "hp", "shield", "energy")
KIN_COLS = ("x", "y", "vx", "vy")
ACTUATED_Q = ("vx", "vy", "energy")

POS_IDX = tuple(Q[c] for c in POS_COLS)  # (1, 2)
VEL_IDX = tuple(Q[c] for c in VEL_COLS)  # (3, 4)
VAL_IDX = tuple(Q[c] for c in VAL_COLS)  # (0, 5, 6, 7)

# Player resource ledger p_p = (minerals, vespene) / 1000
PLY_COLS = ("minerals", "vespene")
PL = {name: i for i, name in enumerate(PLY_COLS)}
N_LEDGER = len(PLY_COLS)  # 2

# 10 Grounded Physical Totals per player (`x_obs`, zero Gram matrices, 100% translation-invariant)
OBS_COLS = (
    "minerals",  # 0: q_obs
    "vespene",   # 1: q_obs
    "count",     # 2: q_obs (sum 1 / N0)
    "hp",        # 3: q_obs (sum hp / N0)
    "shield",    # 4: q_obs (sum shield / N0)
    "energy",    # 5: q_obs (sum energy / N0)
    "vx",        # 6: p_obs_lin (sum vx / N0)
    "vy",        # 7: p_obs_lin (sum vy / N0)
    "vx2",       # 8: p_obs_quad (sum vx^2 / N0)
    "vy2",       # 9: p_obs_quad (sum vy^2 / N0)
)
OBS = {name: i for i, name in enumerate(OBS_COLS)}
N_OBS_P = len(OBS_COLS)  # 10 grounded physical observables per player
N_OBS_Q = 6              # indices 0..5 are configuration q_obs
N_OBS_MOM = 4            # indices 6..9 are momentum/kinetic p_obs

GROUND, P1, P2 = 0, 1, 2
OWNERS = 3

# Normalization constants
N0 = 100.0          # canonical reference unit count
BANK_SCALE = 1000.0  # mineral / vespene bank normalization
MAP_SCALE = 100.0   # physical map coordinate scale (100 map units = 1 normalized unit)
ENERGY_UNIT = 200.0  # spellcaster energy normalization (200 max energy)
FPS = 22.4          # Faster game loops per second
SPEED_SCALE = 5.0 / FPS  # canonical unit speed (5 map units/s) per Faster game loop
R_LATENT = 16       # 16 non-negative continuous field channels per entity (D_p = 482)

# Continuous 2D Polar Gauss-Rayleigh Wave-Packet Quadrature (10 rings x 8 angles = 80 wavevectors)
N_WAVE_SCALES = 10
N_WAVE_ANGLES = 8
N_WAVES = N_WAVE_SCALES * N_WAVE_ANGLES  # 80 wavevectors -> 160 sin/cos features
WAVE_SIGMA_MAP = 10.0  # characteristic local combat/base interaction Gaussian sigma (map units)


def _build_wavevectors(
    n_scales: int = N_WAVE_SCALES,
    n_angles: int = N_WAVE_ANGLES,
    sigma_map: float = WAVE_SIGMA_MAP,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constructs an isotropic 2D Polar Gauss-Rayleigh quadrature of M_w = n_scales * n_angles
    wavevectors k_m in R^{M_w x 2} and positive quadrature weights w_m in R^{M_w} (sum w_m = 1)
    approximating the 2D Gaussian interaction kernel:
        K(Delta r) = exp(-||Delta r||^2 / (2 * sigma_norm^2))
    where sigma_norm = sigma_map / MAP_SCALE (0.10 in normalized map units = 10 map units).

    Golden-ratio angular staggering across radial rings eliminates cardinal/diagonal X-ray
    ridges and achieves 4-digit rotational isotropy across all combat/map distances.
    """
    sigma_norm = sigma_map / MAP_SCALE
    u_nodes = (torch.arange(n_scales, device=device, dtype=dtype) + 0.5) / n_scales
    k_mags = (1.0 / sigma_norm) * torch.sqrt(-2.0 * torch.log(1.0 - u_nodes))

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    i_idx = torch.arange(n_scales, device=device, dtype=dtype)[:, None]
    j_idx = torch.arange(n_angles, device=device, dtype=dtype)[None, :]
    thetas = math.pi * (j_idx + (i_idx / phi) % 1.0) / n_angles

    kx = (k_mags[:, None] * torch.cos(thetas)).reshape(-1)
    ky = (k_mags[:, None] * torch.sin(thetas)).reshape(-1)
    k_vecs = torch.stack([kx, ky], dim=-1)
    w_vecs = torch.full((n_scales * n_angles,), 1.0 / (n_scales * n_angles), device=device, dtype=dtype)
    return k_vecs, w_vecs


def _triu_indices(n: int, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Upper-triangle row and column indices (i <= j) for an n x n symmetric matrix."""
    return torch.triu_indices(n, n, offset=0, device=device)


def gram_layout(r: int = R_LATENT) -> dict:
    """Computes the canonical (q, p) phase-space layout for one player's state x_p in R^{D_p}.

    Ordering of x_p = [x_obs (10) | q_lat (116) | p_lat (24)] (total D_p = 150 for r = 8):
      - x_obs [0 : 10]:
          * q_obs [0 : 6]:   minerals, vespene, count, hp, shield, energy
          * p_obs [6 : 10]:  vx (lin), vy (lin), vx2 (quad), vy2 (quad)
      - q_lat [10 : 126] (n_q_lat = 116):
          * F_self  [10 : 46]:   r*(r+1)//2 = 36 localized self-field overlap entries
          * F_cross [46 : 110]:  r*r = 64 localized P1-P2 cross-field contest entries
          * F_env   [110 : 118]: r = 8 localized neutral/resource field overlap entries
          * z_ply   [118 : 126]: r = 8 global player + CLS macro state entries
      - p_lat [126 : 150] (n_p_lat = 24):
          * T_field [126 : 134]: r = 8 localized field kinetic energy diag(Pi_p^T Pi_p) (quad)
          * A_self  [134 : 142]: r = 8 localized self-advection flux diag(F_p^T Pi_p) (lin)
          * A_cross [142 : 150]: r = 8 localized enemy-closing flux diag(Pi_p^T F_{-p}) (lin)
    """
    ii_z, jj_z = _triu_indices(r)
    n_self = len(ii_z)   # 36
    n_cross = r * r      # 64
    n_env = r            # 8
    n_ply = r            # 8
    n_q_lat = n_self + n_cross + n_env + n_ply  # 116

    n_t_field = r        # 8
    n_a_self = r         # 8
    n_a_cross = r        # 8
    n_p_lat = n_t_field + n_a_self + n_a_cross  # 24

    n_lat = n_q_lat + n_p_lat  # 140
    d_p = N_OBS_P + n_lat      # 150

    # Configuration q_indices (122 dims): q_obs [0..5] + q_lat [10..125]
    q_indices = list(range(0, N_OBS_Q)) + list(range(N_OBS_P, N_OBS_P + n_q_lat))

    # Momentum p_indices (28 dims): p_obs [6..9] + p_lat [126..149]
    p_indices = list(range(N_OBS_Q, N_OBS_P)) + list(range(N_OBS_P + n_q_lat, d_p))

    # Split p_indices into odd linear O(v) momenta (p_lin_sub, 18 dims) and
    # non-negative quadratic O(v^2) kinetic densities (p_quad_sub, 10 dims: vx2, vy2, T_field)
    p_lin_in_p = []
    p_quad_in_p = []
    t_field_start = N_OBS_P + n_q_lat
    t_field_end = t_field_start + n_t_field
    for p_pos, idx in enumerate(p_indices):
        if idx in (OBS["vx2"], OBS["vy2"]) or (t_field_start <= idx < t_field_end):
            p_quad_in_p.append(p_pos)
        else:
            p_lin_in_p.append(p_pos)

    q_idx = torch.tensor(q_indices, dtype=torch.long)
    p_idx = torch.tensor(p_indices, dtype=torch.long)
    p_lin_sub = torch.tensor(p_lin_in_p, dtype=torch.long)
    p_quad_sub = torch.tensor(p_quad_in_p, dtype=torch.long)

    # q_own_mask: masks out F_cross inside q_idx so V_own(q_p) and T_p(q_p, p_p) depend
    # strictly on Player p's own configuration, while V_tug(q_1, q_2) sees F_cross!
    f_cross_start = N_OBS_P + n_self
    f_cross_end = f_cross_start + n_cross
    q_own_mask = torch.tensor(
        [0.0 if (f_cross_start <= idx < f_cross_end) else 1.0 for idx in q_indices],
        dtype=torch.float32,
    )

    # Every coordinate in x_p = (q_p, p_p) is translation-invariant (zero kin_idx!)
    w_adv = torch.ones(d_p, dtype=torch.float32)

    # b_mask: which coordinates can be directly actuated by player commands B(x) u_p
    # Allowed (1.0):
    #   - resource ledger (minerals, vespene): spending on production/tech
    #   - grounded energy & velocities (energy, vx, vy, vx2, vy2): spellcasting & move/attack
    #   - global player command/macro state z_ply
    #   - all latent field momentum/flux coordinates p_lat (T_field, A_self, A_cross)
    # Blocked (0.0):
    #   - grounded unit count, hp, shield (must evolve through (J - R) grad H!)
    #   - spatial field configuration F_self, F_cross, F_env (must evolve via J_qp grad_p T!)
    b_mask = torch.zeros(d_p, dtype=torch.float32)
    b_mask[OBS["minerals"]] = 1.0
    b_mask[OBS["vespene"]] = 1.0
    b_mask[OBS["energy"]] = 1.0
    b_mask[OBS["vx"] : N_OBS_P] = 1.0
    z_ply_start = N_OBS_P + n_self + n_cross + n_env
    p_lat_start = N_OBS_P + n_q_lat
    b_mask[z_ply_start:p_lat_start] = 1.0  # z_ply
    b_mask[p_lat_start:d_p] = 1.0          # p_lat

    # goal_mask: aligns active command impulses with advantage gradients on actuated
    # non-ledger coordinates (excluding raw bank drain [0, 1] so spending isn't penalized)
    goal_mask = b_mask.clone()
    goal_mask[:N_LEDGER] = 0.0

    # Per-coordinate step weights for 1-frame deltas in loss_obs
    w_obs = torch.tensor(
        [
            20.0,  # 0: minerals
            20.0,  # 1: vespene
            45.0,  # 2: count
            20.0,  # 3: hp
            20.0,  # 4: shield
            15.0,  # 5: energy
            12.0,  # 6: vx
            12.0,  # 7: vy
            8.0,   # 8: vx2
            8.0,   # 9: vy2
        ],
        dtype=torch.float32,
    )

    return dict(
        r=r,
        n_self=n_self,
        n_cross=n_cross,
        n_env=n_env,
        n_ply=n_ply,
        n_q_lat=n_q_lat,
        n_t_field=n_t_field,
        n_a_self=n_a_self,
        n_a_cross=n_a_cross,
        n_p_lat=n_p_lat,
        n_lat=n_lat,
        d_p=d_p,
        d_q=len(q_indices),
        d_mom=len(p_indices),
        d_p_lin=len(p_lin_in_p),
        d_p_quad=len(p_quad_in_p),
        q_idx=q_idx,
        p_idx=p_idx,
        p_lin_sub=p_lin_sub,
        p_quad_sub=p_quad_sub,
        q_own_mask=q_own_mask,
        f_self_slice=(N_OBS_P, N_OBS_P + n_self),
        f_cross_slice=(N_OBS_P + n_self, N_OBS_P + n_self + n_cross),
        f_env_slice=(N_OBS_P + n_self + n_cross, z_ply_start),
        z_ply_slice=(z_ply_start, p_lat_start),
        t_field_slice=(p_lat_start, p_lat_start + n_t_field),
        a_self_slice=(p_lat_start + n_t_field, p_lat_start + n_t_field + n_a_self),
        a_cross_slice=(p_lat_start + n_t_field + n_a_self, d_p),
        w_adv=w_adv,
        goal_mask=goal_mask,
        b_mask=b_mask,
        w_obs=w_obs,
    )


@dataclass
class State:
    """M frames of fixed-dimension continuous-field phase-space coordinates and control inputs."""

    q: torch.Tensor  # [M, R, K_Q] fp32: per-unit grounded columns
    z: torch.Tensor  # [M, R, r] fp32: per-unit mass-scaled latent modes
    owner: torch.Tensor  # [M, R] long: GROUND (0), P1 (1), P2 (2)
    mask: torch.Tensor  # [M, R] bool: live on-map row
    p: torch.Tensor  # [M, 2, N_LEDGER] fp32: per-player (minerals, vespene) / 1000
    u: torch.Tensor  # [M, 2, u_dim] fp32: per-player sum-pooled commands at frame t
    u_active: torch.Tensor  # [M, 2] bool: whether player p issued an action / order change
    z_ply: torch.Tensor | None = None  # [M, 2, r] fp32: per-player + CLS latent
    x_obs: torch.Tensor | None = None  # [M, 2, N_OBS_P] fp32 cached grounded totals
    x_lat: torch.Tensor | None = None  # [M, 2, n_lat] fp32 cached continuous field integrals
    loop: float = 1.0


_WAVE_CACHE: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_wavevectors(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device, dtype)
    if key not in _WAVE_CACHE:
        _WAVE_CACHE[key] = _build_wavevectors(device=device, dtype=dtype)
    return _WAVE_CACHE[key]


def build_gram_state(q, z, owner, mask, p, z_ply=None, r=R_LATENT):
    """Computes (x_obs [M, 2, 10], x_lat [M, 2, 140]) in full fp32 precision (with
    autocast disabled so matmuls never downcast to bfloat16) using Grounded Physical
    Totals and the Continuous Localized 2D Polar Gauss-Rayleigh Field."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        q = q.float()
        z = z.float()
        p = p.float()
        M = q.shape[0]
        if z_ply is None:
            z_ply = torch.zeros(M, 2, r, device=q.device, dtype=torch.float32)
        else:
            z_ply = z_ply.float()

        ii_z, jj_z = _triu_indices(r, device=q.device)
        k_vecs, w_vecs = _get_wavevectors(q.device, torch.float32)  # [M_w, 2], [M_w]
        sqrt_w = w_vecs.sqrt()[None, None, :]  # [1, 1, M_w]

        # Extract unit positions (normalized by MAP_SCALE = 100.0) and velocities
        pos = q[..., 1:3]  # [M, R, 2]
        vel = q[..., 3:5]  # [M, R, 2]

        # Wave-packet phase angles theta_{u, m} = k_m . r_u and angular rates omega_{u, m} = k_m . v_u
        theta = torch.matmul(pos, k_vecs.t())  # [M, R, M_w]
        omega = torch.matmul(vel, k_vecs.t())  # [M, R, M_w]

        cos_t = torch.cos(theta) * sqrt_w
        sin_t = torch.sin(theta) * sqrt_w

        # Position wave packet psi(r_u) in R^{2 M_w} and velocity-advected psi_dot(r_u, v_u) in R^{2 M_w}
        psi = torch.cat([cos_t, sin_t], dim=-1)  # [M, R, 2*M_w]
        psi_dot = torch.cat([-omega * sin_t, omega * cos_t], dim=-1)  # [M, R, 2*M_w]

        scale = 1.0 / (N0**0.5)

        # Build per-owner continuous fields F_owner, Pi_owner and grounded physical totals
        f_fields = {}
        pi_fields = {}
        obs_by_p = {}

        for own_id in (GROUND, P1, P2):
            m = (mask & (owner == own_id)).unsqueeze(-1).to(torch.float32)  # [M, R, 1]
            zp = z * (m * scale)  # [M, R, r]
            psi_m = psi * m       # [M, R, 2*M_w]
            f_fields[own_id] = torch.matmul(psi_m.transpose(-1, -2), zp)  # [M, 2*M_w, r]

            if own_id in (P1, P2):
                psi_dot_m = psi_dot * m
                pi_fields[own_id] = torch.matmul(psi_dot_m.transpose(-1, -2), zp)  # [M, 2*M_w, r]

                # 8 unit-summed grounded totals: count, hp, shield, energy, vx, vy, vx2, vy2
                qm = q * m  # [M, R, 8]
                count_tot = qm[..., Q["1"]].sum(dim=1, keepdim=True) / N0
                hp_tot = qm[..., Q["hp"]].sum(dim=1, keepdim=True) / N0
                sh_tot = qm[..., Q["shield"]].sum(dim=1, keepdim=True) / N0
                en_tot = qm[..., Q["energy"]].sum(dim=1, keepdim=True) / N0
                vx_tot = qm[..., Q["vx"]].sum(dim=1, keepdim=True) / N0
                vy_tot = qm[..., Q["vy"]].sum(dim=1, keepdim=True) / N0
                vx2_tot = (qm[..., Q["vx"]].square()).sum(dim=1, keepdim=True) / N0
                vy2_tot = (qm[..., Q["vy"]].square()).sum(dim=1, keepdim=True) / N0

                obs_by_p[own_id] = torch.cat(
                    [
                        p[:, own_id - 1],  # [M, 2]: minerals, vespene
                        count_tot,
                        hp_tot,
                        sh_tot,
                        en_tot,
                        vx_tot,
                        vy_tot,
                        vx2_tot,
                        vy2_tot,
                    ],
                    dim=-1,
                )  # [M, 10]

        obs_list = []
        lat_list = []
        for pl, opp in ((P1, P2), (P2, P1)):
            F_p = f_fields[pl]
            F_opp = f_fields[opp]
            F_env = f_fields[GROUND]
            Pi_p = pi_fields[pl]

            # 1. Localized Self-Field Overlap: F_self = F_p^T F_p [M, r, r] -> upper triangle (36)
            g_self_mat = torch.matmul(F_p.transpose(-1, -2), F_p)
            f_self_vec = g_self_mat[:, ii_z, jj_z]  # [M, 36]

            # 2. Localized Cross-Player Contest Field: F_cross = F_p^T F_{-p} [M, r, r] -> flat (64)
            f_cross_vec = torch.matmul(F_p.transpose(-1, -2), F_opp).reshape(M, r * r)  # [M, 64]

            # 3. Localized Neutral/Resource Overlap: F_env = diag(F_p^T F_env) [M, r] (8)
            f_env_vec = (F_p * F_env).sum(dim=1)  # [M, 8]

            # 4. Global player/scene macro latent: z_ply [M, r] (8)
            z_ply_vec = z_ply[:, pl - 1]  # [M, 8]

            # 5. Localized Field Kinetic Energy: T_field = diag(Pi_p^T Pi_p) >= 0 [M, r] (8)
            t_field_vec = Pi_p.square().sum(dim=1)  # [M, 8]

            # 6. Localized Self-Advection Flux: A_self = diag(F_p^T Pi_p) [M, r] (8)
            a_self_vec = (F_p * Pi_p).sum(dim=1)  # [M, 8]

            # 7. Localized Cross-Player Closing Flux: A_cross = diag(Pi_p^T F_{-p}) [M, r] (8)
            a_cross_vec = (Pi_p * F_opp).sum(dim=1)  # [M, 8]

            q_lat = torch.cat([f_self_vec, f_cross_vec, f_env_vec, z_ply_vec], dim=-1)  # [M, 116]
            p_lat = torch.cat([t_field_vec, a_self_vec, a_cross_vec], dim=-1)           # [M, 24]

            obs_list.append(obs_by_p[pl])
            lat_list.append(torch.cat([q_lat, p_lat], dim=-1))

        x_obs = torch.stack(obs_list, dim=1)  # [M, 2, 10]
        x_lat = torch.stack(lat_list, dim=1)  # [M, 2, 140]
        return x_obs, x_lat


def evaluate_spatial_fields_grid(
    q: torch.Tensor,
    z: torch.Tensor,
    owner: torch.Tensor,
    mask: torch.Tensor,
    gx: torch.Tensor,
    gy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Evaluates the continuous 2D painted wave-packet fields F_1(x,y), F_2(x,y), F_env(x,y),
    Pi_1(x,y), Pi_2(x,y) on a 2D map coordinate grid (gx, gy in map units centered at map center)
    for interactive visualization in the WorldModel Inspector.

    Args:
        q: [R, 8] fp32 grounded unit columns at frame t
        z: [R, r] fp32 mass-scaled unit modes at frame t
        owner: [R] long unit owner (0, 1, 2)
        mask: [R] bool live unit mask
        gx, gy: [G, G] fp32 map coordinates relative to map center (in map units)

    Returns:
        dict with:
          - F_p1: [r, G, G] Player 1 painted spatial field modes
          - F_p2: [r, G, G] Player 2 painted spatial field modes
          - F_diff_norm: [G, G] ||F_1(x,y)||_2 - ||F_2(x,y)||_2 (territorial dominance)
          - F_contest: [G, G] F_1(x,y) . F_2(x,y) (local combat/contestation integrand)
          - T_density: [G, G] ||Pi_1(x,y)||_2^2 + ||Pi_2(x,y)||_2^2 (local kinetic field density)
    """
    with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
        q = q.float()
        z = z.float()
        G = gx.shape[0]
        r = z.shape[-1]
        k_vecs, w_vecs = _get_wavevectors(q.device, torch.float32)  # [M_w, 2], [M_w]
        sqrt_w = w_vecs.sqrt()[None, :]  # [1, M_w]

        # Grid wave packets Psi_grid [G*G, 2*M_w]
        pos_grid = torch.stack([gx / MAP_SCALE, gy / MAP_SCALE], dim=-1).reshape(G * G, 2)
        theta_g = pos_grid @ k_vecs.t()  # [G*G, M_w]
        psi_grid = torch.cat([torch.cos(theta_g) * sqrt_w, torch.sin(theta_g) * sqrt_w], dim=-1)

        # Unit wave packets psi_u [R, 2*M_w] and psi_dot_u [R, 2*M_w]
        pos_u = q[:, 1:3]
        vel_u = q[:, 3:5]
        theta_u = pos_u @ k_vecs.t()
        omega_u = vel_u @ k_vecs.t()
        cos_u = torch.cos(theta_u) * sqrt_w
        sin_u = torch.sin(theta_u) * sqrt_w
        psi_u = torch.cat([cos_u, sin_u], dim=-1)
        psi_dot_u = torch.cat([-omega_u * sin_u, omega_u * cos_u], dim=-1)

        scale = 1.0 / (N0**0.5)
        F_grid = {}
        Pi_grid = {}
        for own_id in (GROUND, P1, P2):
            m = (mask & (owner == own_id)).unsqueeze(-1).float()
            zp = z * (m * scale)
            F_wave = (psi_u * m).t() @ zp       # [2*M_w, r]
            F_grid[own_id] = (psi_grid @ F_wave).reshape(G, G, r)   # [G, G, r]
            if own_id in (P1, P2):
                Pi_wave = (psi_dot_u * m).t() @ zp  # [2*M_w, r]
                Pi_grid[own_id] = (psi_grid @ Pi_wave).reshape(G, G, r) # [G, G, r]

        F_env = F_grid[GROUND]
        F1 = F_grid[P1]
        F2 = F_grid[P2]
        Pi1 = Pi_grid[P1]
        Pi2 = Pi_grid[P2]

        f_diff_norm = F1.norm(dim=-1) - F2.norm(dim=-1)
        f_contest = (F1 * F2).sum(dim=-1)
        t_density = Pi1.square().sum(dim=-1) + Pi2.square().sum(dim=-1)

        return {
            "F_p1": F1.permute(2, 0, 1),    # [r, G, G]
            "F_p2": F2.permute(2, 0, 1),    # [r, G, G]
            "F_env": F_env.permute(2, 0, 1),# [r, G, G]
            "Pi_p1": Pi1.permute(2, 0, 1),  # [r, G, G]
            "Pi_p2": Pi2.permute(2, 0, 1),  # [r, G, G]
            "F_diff_norm": f_diff_norm,     # [G, G]
            "F_contest": f_contest,         # [G, G]
            "T_density": t_density,         # [G, G]
        }


def mlp(d_in: int, d_out: int, hidden: int, bias: bool = True) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, hidden),
        nn.LayerNorm(hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, d_out, bias=bias),
    )


def _phd_mlp(d_in: int, d_hid: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hid),
        nn.SiLU(),
        nn.Linear(d_hid, d_out),
    )


def small_init(layer: nn.Linear, std: float = 1e-3) -> nn.Linear:
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def _smooth_mlp(d_in: int, d_hid: int, d_out: int) -> nn.Sequential:
    """3-layer smooth MLP with LayerNorm + SiLU for continuous C^2 energy landscapes."""
    return nn.Sequential(
        nn.Linear(d_in, d_hid),
        nn.LayerNorm(d_hid),
        nn.SiLU(),
        nn.Linear(d_hid, d_hid),
        nn.LayerNorm(d_hid),
        nn.SiLU(),
        nn.Linear(d_hid, d_out),
    )


class Hamiltonian(nn.Module):
    """Explicit Potential + Kinetic Hamiltonian H_p(q, p) = V_p(q_1, q_2) + T_p(q_p, p_p).

    1. **Potential Energy V_p(q_1, q_2)**:
         V_1(q_1, q_2) = V_own(q_{own, 1}) + V_tug(q_1, q_2)
         V_2(q_1, q_2) = V_own(q_{own, 2}) - V_tug(q_1, q_2)
       where V_tug(q_1, q_2) = 0.5 * (v_tug(q_1, q_2) - v_tug(q_2, q_1)) is strictly
       antisymmetric under player swap, and q_{own, p} masks out F_cross so V_own measures
       purely Player p's own economy/army/field configuration.

    2. **Kinetic Energy T_p(q_{own, p}, p_p) >= 0 (Purely Quadratic O(v^2))**:
         T_p(q, p) = 0.5 * ||L_T(q_{own, p})^T p_{lin, p}||^2
                   + softplus(w_T(q_{own, p}))^T p_{quad, p} >= 0
       Because p_{lin, p} is strictly O(v) and p_{quad, p} is strictly O(v^2) >= 0,
       T_p(q, p) is purely quadratic O(v^2), satisfies time-reversal symmetry
       T_p(q, -v) == T_p(q, +v), and vanishes identically at zero velocity T_p(q, 0) == 0.

    3. **Player Advantage**:
         A_1(x_1, x_2) = H_1 - H_2 = (V_own(q_{own, 1}) - V_own(q_{own, 2}))
                                   + 2 * V_tug(q_1, q_2)
                                   + (T_1(q_{own, 1}, p_1) - T_2(q_{own, 2}, p_2))
    """

    def __init__(self, d: int = 256, r: int = R_LATENT, rank_t: int = 16):
        super().__init__()
        layout = gram_layout(r)
        self.d_p = layout["d_p"]
        self.d_q = layout["d_q"]
        self.d_mom = layout["d_mom"]
        self.d_p_lin = layout["d_p_lin"]
        self.d_p_quad = layout["d_p_quad"]
        self.rank_t = rank_t

        self.register_buffer("q_idx", layout["q_idx"])
        self.register_buffer("p_idx", layout["p_idx"])
        self.register_buffer("p_lin_sub", layout["p_lin_sub"])
        self.register_buffer("p_quad_sub", layout["p_quad_sub"])
        self.register_buffer("q_own_mask", layout["q_own_mask"])

        # Potential energy networks V_own(q_{own, p}) and V_tug(q_1, q_2)
        self.v_own_net = _smooth_mlp(self.d_q, d, 1)
        self.v_tug_net = _smooth_mlp(2 * self.d_q, d, 1)

        # Kinetic energy networks:
        #   - l_t_net(q_{own, p}) -> [d_p_lin, rank_t] inverse-mass factor on linear O(v) momenta
        #   - w_t_net(q_{own, p}) -> [d_p_quad] positive weights on quadratic O(v^2) kinetic densities
        self.l_t_net = nn.Sequential(
            nn.Linear(self.d_q, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, self.d_p_lin * rank_t),
        )
        self.w_t_net = nn.Sequential(
            nn.Linear(self.d_q, d // 2),
            nn.LayerNorm(d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, self.d_p_quad),
        )

    def _kinetic_energy(self, q_own: torch.Tensor, p_p: torch.Tensor) -> torch.Tensor:
        B = q_own.shape[0]
        p_lin = p_p[:, self.p_lin_sub]    # [B, d_p_lin]
        p_quad = p_p[:, self.p_quad_sub]  # [B, d_p_quad] >= 0

        l_mat = self.l_t_net(q_own).view(B, self.d_p_lin, self.rank_t) * (
            1.0 / ((self.d_p_lin * self.rank_t) ** 0.5)
        )
        proj = torch.bmm(p_lin.unsqueeze(1), l_mat).squeeze(1)  # [B, rank_t]
        t_lin = 0.5 * proj.square().sum(dim=-1)

        w_quad = F.softplus(self.w_t_net(q_own)) * (0.1 / self.d_p_quad)
        t_quad = (w_quad * p_quad).sum(dim=-1)
        return t_lin + t_quad

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Evaluates H_1, H_2, V_1, V_2, T_1, T_2, A_1 for batch of states x1, x2: [B, D_p]."""
        q1 = x1[:, self.q_idx]
        q2 = x2[:, self.q_idx]
        p1 = x1[:, self.p_idx]
        p2 = x2[:, self.p_idx]

        q1_own = q1 * self.q_own_mask
        q2_own = q2 * self.q_own_mask

        # 1. Potential Energy V_p(q)
        v_own1 = self.v_own_net(q1_own).squeeze(-1)
        v_own2 = self.v_own_net(q2_own).squeeze(-1)

        t12 = self.v_tug_net(torch.cat([q1, q2], dim=-1)).squeeze(-1)
        t21 = self.v_tug_net(torch.cat([q2, q1], dim=-1)).squeeze(-1)
        v_tug = 0.5 * (t12 - t21)

        V1 = v_own1 + v_tug
        V2 = v_own2 - v_tug

        # 2. Kinetic Energy T_p(q, p) >= 0 (purely O(v^2))
        T1 = self._kinetic_energy(q1_own, p1)
        T2 = self._kinetic_energy(q2_own, p2)

        # 3. Total Hamiltonian H_p = V_p + T_p (zero kin_idx!)
        H1 = V1 + T1
        H2 = V2 + T2
        A1 = H1 - H2

        zeros = torch.zeros_like(H1)
        return {
            "H1": H1,
            "H2": H2,
            "V1": V1,
            "V2": V2,
            "T1": T1,
            "T2": T2,
            "A1": A1,
            "own1": v_own1,
            "own2": v_own2,
            "tug": v_tug,
            "kin1": zeros,
            "kin2": zeros,
        }


class PortHamiltonianDynamics(nn.Module):
    """Canonical Block-Structured 2-Player Port-Hamiltonian System:

        [ dq_p ]   [  -R_qq(x_p)      J_qp(x_p)  ] [ grad_{q_p} V_p ]   [ B_q(x_p) ]
        [ dp_p ] = [ -J_qp(x_p)^T    -R_pp(x_p)  ] [ grad_{p_p} T_p ] + [ B_p(x_p) ] u_p
                   + cross-player (J_12 - R_12) coupling

    Guarantees by construction:
      1. Canonical symplectic coupling between configuration q and momentum p:
           dq_p = +J_qp grad_{p_p} T_p  (advection of configuration by velocity/momentum)
           dp_p = -J_qp^T grad_{q_p} V_p (spatial force -grad_q V accelerating momentum)
      2. Exact energy conservation of J:
           <grad H_1, j_1> + <grad H_2, j_2> == 0
      3. Non-negative energy dissipation of R:
           <grad H_p, r_p> >= 0  (combat attrition, friction, decay)
    """

    def __init__(self, d_p: int, u_dim: int, d_hid: int = 256, rank: int = 16, r: int = R_LATENT):
        super().__init__()
        self.d_p = d_p
        self.u_dim = u_dim
        self.rank = rank

        layout = gram_layout(r)
        self.d_q = layout["d_q"]
        self.d_mom = layout["d_mom"]

        self.register_buffer("q_idx", layout["q_idx"])
        self.register_buffer("p_idx", layout["p_idx"])
        self.register_buffer("w_adv", layout["w_adv"])
        self.register_buffer("b_mask", layout["b_mask"])

        # Canonical symplectic coupling J_qp(x_p): maps grad_p T -> dq and -grad_q V -> dp
        # Factored as U_q(x_p) V_p(x_p)^T in R^{d_q x d_mom} plus internal skew J_qq, J_pp
        self.j_qp_net = _phd_mlp(d_p, d_hid, (self.d_q + self.d_mom) * rank)
        self.j_qq_net = _phd_mlp(d_p, d_hid, 2 * self.d_q * rank)
        self.j_pp_net = _phd_mlp(d_p, d_hid, 2 * self.d_mom * rank)

        # Cross-player conservative exchange J_12(x_1, x_2)
        self.j_cross_net = _phd_mlp(2 * d_p, d_hid, 4 * d_p * rank)

        # Block-structured PSD dissipation R_qq(x_p) >= 0 and R_pp(x_p) >= 0
        self.r_qq_net = _phd_mlp(d_p, d_hid, self.d_q * rank + self.d_q)
        self.r_pp_net = _phd_mlp(d_p, d_hid, self.d_mom * rank + self.d_mom)

        # Cross-player mutual combat attrition R_cross(x_1, x_2) >= 0
        self.r_cross_net = _phd_mlp(2 * d_p, d_hid, 2 * d_p * rank)

        # State-dependent input port matrix B(x_p): [B, d_p, u_dim]
        self.b_net = _phd_mlp(d_p, d_hid, d_p * u_dim)

    def _apply_skew(self, ab: torch.Tensor, g: torch.Tensor, dim: int) -> torch.Tensor:
        """Applies (A B^T - B A^T) g in O(dim * rank) without materializing dim x dim."""
        B = g.shape[0]
        r = self.rank
        ab = ab.view(B, 2, dim, r) * (1.0 / ((dim * r) ** 0.5))
        a, b = ab[:, 0], ab[:, 1]
        bt_g = torch.bmm(b.transpose(1, 2), g.unsqueeze(-1))
        at_g = torch.bmm(a.transpose(1, 2), g.unsqueeze(-1))
        return (torch.bmm(a, bt_g) - torch.bmm(b, at_g)).squeeze(-1)

    def _apply_j_qp(
        self, uv: torch.Tensor, g_q: torch.Tensor, g_p: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies canonical block [[0, J_qp], [-J_qp^T, 0]] to [g_q, g_p] where J_qp = U_q V_p^T."""
        B = g_q.shape[0]
        r = self.rank
        scale = 1.0 / (((self.d_q + self.d_mom) * r) ** 0.5)
        uv = uv * scale
        u_q = uv[:, : self.d_q * r].view(B, self.d_q, r)
        v_p = uv[:, self.d_q * r :].view(B, self.d_mom, r)

        # dq = +U_q (V_p^T g_p), dp = -V_p (U_q^T g_q)
        vt_gp = torch.bmm(v_p.transpose(1, 2), g_p.unsqueeze(-1))
        dq = torch.bmm(u_q, vt_gp).squeeze(-1)

        ut_gq = torch.bmm(u_q.transpose(1, 2), g_q.unsqueeze(-1))
        dp = -torch.bmm(v_p, ut_gq).squeeze(-1)
        return dq, dp

    def _apply_psd(self, ld: torch.Tensor, g: torch.Tensor, dim: int) -> torch.Tensor:
        """Applies (L L^T + diag(softplus(d))) g >= 0 in O(dim * rank)."""
        B = g.shape[0]
        r = self.rank
        l_flat = ld[:, : dim * r].view(B, dim, r) * (1.0 / ((dim * r) ** 0.5))
        d_diag = F.softplus(ld[:, dim * r :]) * (1.0 / dim)
        lt_g = torch.bmm(l_flat.transpose(1, 2), g.unsqueeze(-1))
        ll_g = torch.bmm(l_flat, lt_g).squeeze(-1)
        return ll_g + d_diag * g

    def _scatter_qp(self, q_vec: torch.Tensor, p_vec: torch.Tensor) -> torch.Tensor:
        B = q_vec.shape[0]
        out = q_vec.new_zeros((B, self.d_p))
        out.index_copy_(1, self.q_idx, q_vec)
        out.index_copy_(1, self.p_idx, p_vec)
        return out

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        u1: torch.Tensor,
        u2: torch.Tensor,
        H_fn: Hamiltonian,
        dt: float | torch.Tensor = 1.0,
    ):
        """Computes next-state predictions and Port-Hamiltonian flow pieces."""
        with torch.enable_grad():
            x1_g = x1.detach().requires_grad_(True)
            x2_g = x2.detach().requires_grad_(True)
            h_dict = H_fn(x1_g, x2_g)
            H1, H2 = h_dict["H1"], h_dict["H2"]

            g1 = torch.autograd.grad(H1.sum(), x1_g, retain_graph=True)[0]
            g2 = torch.autograd.grad(H2.sum(), x2_g, retain_graph=False)[0]

        g1_q, g1_p = g1[:, self.q_idx], g1[:, self.p_idx]
        g2_q, g2_p = g2[:, self.q_idx], g2[:, self.p_idx]

        # 1. Canonical Conservative Flow J(x) grad H
        uv1 = self.j_qp_net(x1)
        uv2 = self.j_qp_net(x2)
        dq1_qp, dp1_qp = self._apply_j_qp(uv1, g1_q, g1_p)
        dq2_qp, dp2_qp = self._apply_j_qp(uv2, g2_q, g2_p)

        dq1 = dq1_qp + self._apply_skew(self.j_qq_net(x1), g1_q, self.d_q)
        dp1 = dp1_qp + self._apply_skew(self.j_pp_net(x1), g1_p, self.d_mom)
        dq2 = dq2_qp + self._apply_skew(self.j_qq_net(x2), g2_q, self.d_q)
        dp2 = dp2_qp + self._apply_skew(self.j_pp_net(x2), g2_p, self.d_mom)

        j1_self = self._scatter_qp(dq1, dp1)
        j2_self = self._scatter_qp(dq2, dp2)

        # Cross-player conservative exchange J_12
        x12 = torch.cat([x1, x2], dim=-1)
        x21 = torch.cat([x2, x1], dim=-1)
        g12 = torch.cat([g1, g2], dim=-1)
        g21 = torch.cat([g2, g1], dim=-1)
        j_c12 = self._apply_skew(self.j_cross_net(x12), g12, 2 * self.d_p)
        j_c21 = self._apply_skew(self.j_cross_net(x21), g21, 2 * self.d_p)
        j1_cross = 0.5 * (j_c12[:, : self.d_p] + j_c21[:, self.d_p :])
        j2_cross = 0.5 * (j_c12[:, self.d_p :] + j_c21[:, : self.d_p])

        j1 = j1_self + j1_cross
        j2 = j2_self + j2_cross

        # 2. Block-Structured PSD Dissipative Flow R(x) grad H (combat attrition & friction)
        rq1 = self._apply_psd(self.r_qq_net(x1), g1_q, self.d_q)
        rp1 = self._apply_psd(self.r_pp_net(x1), g1_p, self.d_mom)
        rq2 = self._apply_psd(self.r_qq_net(x2), g2_q, self.d_q)
        rp2 = self._apply_psd(self.r_pp_net(x2), g2_p, self.d_mom)

        r1_self = self._scatter_qp(rq1, rp1)
        r2_self = self._scatter_qp(rq2, rp2)

        B = x1.shape[0]
        l_c12 = self.r_cross_net(x12).view(B, 2 * self.d_p, self.rank) * (
            1.0 / ((2 * self.d_p * self.rank) ** 0.5)
        )
        l_c21_raw = self.r_cross_net(x21).view(B, 2, self.d_p, self.rank) * (
            1.0 / ((2 * self.d_p * self.rank) ** 0.5)
        )
        l_c21 = torch.cat([l_c21_raw[:, 1], l_c21_raw[:, 0]], dim=1)
        l_c = 0.5 * (l_c12 + l_c21)
        rc_full = torch.bmm(l_c, torch.bmm(l_c.transpose(1, 2), g12.unsqueeze(-1))).squeeze(-1)

        r1 = r1_self + rc_full[:, : self.d_p]
        r2 = r2_self + rc_full[:, self.d_p :]

        # 3. Control Input Port B(x_p) u_p (masked by b_mask)
        b1_mat = self.b_net(x1).view(B, self.d_p, self.u_dim) * (1.0 / (self.u_dim**0.5))
        b2_mat = self.b_net(x2).view(B, self.d_p, self.u_dim) * (1.0 / (self.u_dim**0.5))
        bu1 = torch.bmm(b1_mat, u1.unsqueeze(-1)).squeeze(-1) * self.b_mask
        bu2 = torch.bmm(b2_mat, u2.unsqueeze(-1)).squeeze(-1) * self.b_mask

        # 4. Symplectic goal-gradient pull from q to p so commands issued from rest (p=0)
        # immediately receive advantage gradient via J_qp!
        g1_goal = g1 - g2
        g2_goal = g2 - g1
        _, dp1_goal_pull = self._apply_j_qp(
            uv1, g1_goal[:, self.q_idx], torch.zeros_like(g1_p)
        )
        _, dp2_goal_pull = self._apply_j_qp(
            uv2, g2_goal[:, self.q_idx], torch.zeros_like(g2_p)
        )
        g1_goal_eff = g1_goal + self._scatter_qp(torch.zeros_like(g1_q), dp1_goal_pull)
        g2_goal_eff = g2_goal + self._scatter_qp(torch.zeros_like(g2_q), dp2_goal_pull)

        if isinstance(dt, torch.Tensor):
            dt_col = dt.view(-1, 1)
        else:
            dt_col = dt

        dx1 = (j1 - r1 + bu1) * dt_col
        dx2 = (j2 - r2 + bu2) * dt_col
        x1_next = x1 + dx1
        x2_next = x2 + dx2

        diss1 = (g1 * r1).sum(dim=-1)
        diss2 = (g2 * r2).sum(dim=-1)

        return (
            x1_next,
            x2_next,
            {
                **h_dict,
                "g1": g1,
                "g2": g2,
                "g1_goal": g1_goal,
                "g2_goal": g2_goal,
                "g1_goal_eff": g1_goal_eff,
                "g2_goal_eff": g2_goal_eff,
                "j1": j1,
                "j2": j2,
                "r1": r1,
                "r2": r2,
                "bu1": bu1,
                "bu2": bu2,
                "dx1": dx1,
                "dx2": dx2,
                "diss1": diss1,
                "diss2": diss2,
            },
        )


def _compiled_dynamics_step(dyn: PortHamiltonianDynamics, H_fn: Hamiltonian, x1, x2, u1, u2, dt):
    return dyn(x1, x2, u1, u2, H_fn, dt=dt)


if hasattr(torch, "_functorch") and hasattr(torch._functorch, "config"):
    torch._functorch.config.donated_buffer = False


class EnergyModel(nn.Module):
    """Zero-Gram Continuous-Field Port-Hamiltonian World Model."""

    def __init__(
        self,
        d: int = 256,
        u_dim: int = 32,
        hidden: int = 256,
        r: int = R_LATENT,
        rank: int = 16,
    ):
        super().__init__()
        self.d = d
        self.hidden = hidden
        self.r = r
        self.u_dim = u_dim
        self._step_fn = _compiled_dynamics_step

        layout = gram_layout(r)
        self.d_p = layout["d_p"]
        self.n_lat = layout["n_lat"]

        self.register_buffer("w_obs", layout["w_obs"])
        self.register_buffer("w_adv", layout["w_adv"])
        self.register_buffer("goal_mask", layout["goal_mask"])

        self.H = Hamiltonian(d=hidden, r=r, rank_t=rank)
        self.dynamics = PortHamiltonianDynamics(
            d_p=self.d_p, u_dim=u_dim, d_hid=hidden, rank=rank, r=r
        )

    def compile_parts(self):
        """Compiles the Port-Hamiltonian dynamics step with donated_buffer=False for create_graph=True."""
        if hasattr(torch, "_functorch") and hasattr(torch._functorch, "config"):
            torch._functorch.config.donated_buffer = False
        if hasattr(torch, "compile"):
            self._step_fn = torch.compile(_compiled_dynamics_step, dynamic=True)
        return self

    def _goal_loss(
        self, g_goal: torch.Tensor, bu: torch.Tensor, active: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aligns active control impulse B(x_p) u_p with -grad_{x_p} A_p on non-ledger
        actuated coordinates (including symplectic pull from q to p)."""
        if not active.any():
            z = g_goal.sum() * 0.0
            return z, z.detach()

        target = (-g_goal.detach()) * self.goal_mask
        bu_eff = bu * self.goal_mask
        t_norm = target.norm(dim=-1)
        b_norm = bu_eff.norm(dim=-1)
        valid = active & (t_norm > 1e-6) & (b_norm > 1e-6)
        if not valid.any():
            z = g_goal.sum() * 0.0
            return z, z.detach()

        cos = (target[valid] * bu_eff[valid]).sum(dim=-1) / (
            t_norm[valid] * b_norm[valid] + 1e-8
        )
        loss = (1.0 - cos).mean()
        return loss, cos.detach().mean()

    @staticmethod
    def _variance_floor_loss(
        z: torch.Tensor, mask: torch.Tensor, z_ply: torch.Tensor | None, target_std: float = 0.12
    ) -> torch.Tensor:
        """VICReg variance floor + off-diagonal decorrelation preventing radial collapse
        and encouraging the r non-negative field modes z_u >= 0 to specialize into
        distinct, non-overlapping tactical archetypes."""
        live_z = z[mask]
        if live_z.shape[0] > 4:
            std_z = (live_z.var(dim=0, unbiased=False) + 1e-6).sqrt()
            loss_var = F.relu(target_std - std_z).mean()

            # Off-diagonal correlation penalty on normalized channels
            z_cent = (live_z - live_z.mean(dim=0, keepdim=True)) / std_z.unsqueeze(0)
            corr = (z_cent.t() @ z_cent) / live_z.shape[0]
            r_dim = corr.shape[0]
            off_diag = corr - torch.diag(torch.diag(corr))
            loss_cov = (off_diag.square().sum()) / max(1, r_dim * (r_dim - 1))
            loss_z = loss_var + 0.25 * loss_cov
        else:
            loss_z = z.sum() * 0.0

        if z_ply is not None and z_ply.shape[0] > 2:
            zp_flat = z_ply.reshape(-1, z_ply.shape[-1])
            std_zp = (zp_flat.var(dim=0, unbiased=False) + 1e-6).sqrt()
            loss_zp = F.relu(target_std - std_zp).mean()
        else:
            loss_zp = z.sum() * 0.0

        return loss_z + loss_zp

    def forward(
        self,
        states: list[State] | State,
        s_next: State | dict[str, float] | None = None,
        lat_weight: float = 0.1,
        goal_weight: float = 0.05,
        reg_weight: float = 0.02,
    ):
        paired_call = isinstance(s_next, State)
        if paired_call:
            assert isinstance(states, State)
            s_t = states
            if s_t.x_obs is not None and s_t.x_lat is not None:
                x_obs_t, x_lat_t = s_t.x_obs, s_t.x_lat
            else:
                x_obs_t, x_lat_t = build_gram_state(
                    s_t.q, s_t.z, s_t.owner, s_t.mask, s_t.p, s_t.z_ply, r=self.r
                )
            if s_next.x_obs is not None and s_next.x_lat is not None:
                x_obs_next, x_lat_next = s_next.x_obs, s_next.x_lat
            else:
                x_obs_next, x_lat_next = build_gram_state(
                    s_next.q, s_next.z, s_next.owner, s_next.mask, s_next.p, s_next.z_ply, r=self.r
                )
            u_t = s_t.u
            act_t = s_t.u_active
            dt_vec = torch.full(
                (s_t.q.shape[0],), float(s_t.loop), device=s_t.q.device, dtype=torch.float32
            )
            loss_reg = 0.5 * (
                self._variance_floor_loss(s_t.z, s_t.mask, s_t.z_ply)
                + self._variance_floor_loss(s_next.z, s_next.mask, s_next.z_ply)
            )
        else:
            if isinstance(states, State):
                states = [states]

            all_obs_t, all_lat_t = [], []
            all_obs_n, all_lat_n = [], []
            all_u_t, all_act_t, all_dt = [], [], []
            reg_losses = []

            for s in states:
                M = s.q.shape[0]
                if M < 2:
                    continue
                if s.x_obs is not None and s.x_lat is not None:
                    x_obs, x_lat = s.x_obs, s.x_lat
                else:
                    x_obs, x_lat = build_gram_state(
                        s.q, s.z, s.owner, s.mask, s.p, s.z_ply, r=self.r
                    )
                reg_losses.append(self._variance_floor_loss(s.z, s.mask, s.z_ply))

                all_obs_t.append(x_obs[:-1])
                all_lat_t.append(x_lat[:-1])
                all_obs_n.append(x_obs[1:])
                all_lat_n.append(x_lat[1:])
                all_u_t.append(s.u[:-1])
                all_act_t.append(s.u_active[:-1])
                all_dt.append(
                    torch.full(
                        (M - 1,), float(s.loop), device=s.q.device, dtype=torch.float32
                    )
                )

            x_obs_t = torch.cat(all_obs_t, dim=0)
            x_lat_t = torch.cat(all_lat_t, dim=0)
            x_obs_next = torch.cat(all_obs_n, dim=0)
            x_lat_next = torch.cat(all_lat_n, dim=0)
            u_t = torch.cat(all_u_t, dim=0)
            act_t = torch.cat(all_act_t, dim=0)
            dt_vec = torch.cat(all_dt, dim=0)
            loss_reg = torch.stack(reg_losses).mean() if reg_losses else x_obs_t.sum() * 0.0

        x_t = torch.cat([x_obs_t, x_lat_t], dim=-1)  # [B_tot, 2, 150]
        x1_t, x2_t = x_t[:, 0], x_t[:, 1]
        u1_t, u2_t = u_t[:, 0], u_t[:, 1]

        x1_pred, x2_pred, pieces = self._step_fn(
            self.dynamics, self.H, x1_t, x2_t, u1_t, u2_t, dt_vec
        )
        x_pred = torch.stack([x1_pred, x2_pred], dim=1)  # [B_tot, 2, 150]

        pred_obs = x_pred[:, :, :N_OBS_P]
        pred_lat = x_pred[:, :, N_OBS_P:]

        # 1. Grounded Physical Totals MSE Loss (10 clean physical observables per player)
        w = self.w_obs.view(1, 1, N_OBS_P)
        diff_obs = (pred_obs - x_obs_next) * w
        copy_diff_obs = (x_obs_t - x_obs_next) * w

        loss_obs = F.smooth_l1_loss(diff_obs, torch.zeros_like(diff_obs), beta=0.5)
        copy_loss_obs = F.smooth_l1_loss(
            copy_diff_obs, torch.zeros_like(copy_diff_obs), beta=0.5
        ).detach()

        # 2. Latent Continuous Field Dynamics Consistency Loss (with stop-gradient on target)
        loss_lat = F.smooth_l1_loss(pred_lat, x_lat_next.detach(), beta=0.2)

        # 3. Goal-Directed Control Port Alignment Loss (including symplectic pull from q to p)
        g_loss1, cos1 = self._goal_loss(pieces["g1_goal_eff"], pieces["bu1"], act_t[:, 0])
        g_loss2, cos2 = self._goal_loss(pieces["g2_goal_eff"], pieces["bu2"], act_t[:, 1])
        loss_goal = 0.5 * (g_loss1 + g_loss2)
        goal_cos = 0.5 * (cos1 + cos2)

        total_loss = (
            loss_obs
            + lat_weight * loss_lat
            + goal_weight * loss_goal
            + reg_weight * loss_reg
        )

        # Per-column physical RMSE diagnostics on the 10 grounded totals
        with torch.no_grad():
            raw_err = (pred_obs - x_obs_next).square().mean(dim=(0, 1)).sqrt()
            copy_err = (x_obs_t - x_obs_next).square().mean(dim=(0, 1)).sqrt()

            diag = {
                "loss": total_loss,
                "loss_obs": loss_obs.detach(),
                "loss_lat": loss_lat.detach(),
                "loss_reg": loss_reg.detach(),
                "loss_goal": loss_goal.detach(),
                "goal_cos": goal_cos.detach(),
                "copy_loss_obs": copy_loss_obs,
                "rmse_minerals": raw_err[OBS["minerals"]],
                "rmse_vespene": raw_err[OBS["vespene"]],
                "copy_rmse_minerals": copy_err[OBS["minerals"]],
                "copy_rmse_vespene": copy_err[OBS["vespene"]],
                "res_1": raw_err[OBS["count"]],
                "copy_res_1": copy_err[OBS["count"]],
                "res_x": raw_err[OBS["vx2"]],
                "copy_res_x": copy_err[OBS["vx2"]],
                "res_y": raw_err[OBS["vy2"]],
                "copy_res_y": copy_err[OBS["vy2"]],
                "res_vx": raw_err[OBS["vx"]],
                "copy_res_vx": copy_err[OBS["vx"]],
                "res_vy": raw_err[OBS["vy"]],
                "copy_res_vy": copy_err[OBS["vy"]],
                "rmse_hp": raw_err[OBS["hp"]],
                "copy_rmse_hp": copy_err[OBS["hp"]],
                "rmse_shield": raw_err[OBS["shield"]],
                "copy_rmse_shield": copy_err[OBS["shield"]],
                "res_energy": raw_err[OBS["energy"]],
                "copy_res_energy": copy_err[OBS["energy"]],
                "H_1": pieces["H1"].detach().mean(),
                "H_2": pieces["H2"].detach().mean(),
                "V_1": pieces["V1"].detach().mean(),
                "V_2": pieces["V2"].detach().mean(),
                "T_1": pieces["T1"].detach().mean(),
                "T_2": pieces["T2"].detach().mean(),
                "advantage": pieces["A1"].detach().mean(),
                "tug": pieces["tug"].detach().mean(),
                "dissipated_1": pieces["diss1"].detach().mean(),
                "dissipated_2": pieces["diss2"].detach().mean(),
            }
        if paired_call:
            return total_loss, diag
        return diag
