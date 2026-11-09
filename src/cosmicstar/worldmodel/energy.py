"""The Port-Hamiltonian energy model in fixed-dimension Gram phase space.

Each frame's variable-sized entity table is mapped to a fixed-dimension per-player
phase-space state x_p in fp32:

    x_obs_p (38)   player ledger (minerals, vespene) + upper triangle of the grounded
                   physical Gram matrix G_qq = (1 / N_0) Q_p^T Q_p over the 8 raw
                   per-unit columns Q_COLS = (1, x, y, vx, vy, hp, shield, energy)
    x_lat_p (100)  cross-Gram G_qz = (1 / N_0) Q_p^T Z_p (64) and upper triangle of the
                   latent tactical Gram matrix G_zz = (1 / N_0) Z_p^T Z_p (36), where
                   Z_p in R^{N_p x r} (r = 8) is projected from the attended entity
                   tokens (which attend over both players' entities, neutral resource
                   patches with their remaining contents, and the patch's ruleset)

Together, x = (x_1, x_2) has fixed dimension D = 276 (138 per player: 76 grounded,
200 latent/cross) regardless of the number of live entities N.

Port-Hamiltonian dynamics on x = (x_1, x_2):

    ẋ = (J(x) - R(x)) g(x) + B(x) u

where:
    H_1, H_2       two player-symmetric instances of one Hamiltonian:
                       H_1(x) = E_own(x̃_1) + E_tug(x̃_1, x̃_2)
                       H_2(x) = E_own(x̃_2) - E_tug(x̃_1, x̃_2)
                   with E_tug(a, b) = -E_tug(b, a) antisymmetric (tug-of-war).
                   In H_1 and H_2 (and the advantage A_1 = H_1 - H_2), the pure
                   kinematic dimensions (x, y, vx, vy) are weighted to 0 via w_adv, so
                   raw unit motion adds zero advantage unless it improves tactical
                   posture Z_p or resource/health flows. A separate local kinematic
                   potential H_kin(x_p) drives the kinematic entries of G_qq.
    g(x)           the per-owner effort vector [∇_{x_1}(H_1 + H_kin), ∇_{x_2}(H_2 + H_kin)]
    J(x) = A - A^T skew-symmetric conservative router (g^T J g = 0 by construction)
    R(x) = L L^T   positive semi-definite dissipative sink (g^T R g >= 0 by construction)
    B(x) u         block-diagonal per-player control port: u_p is sum-pooled over frame
                   commands (exact 0 when idle) and masked so commands can directly push
                   bank, velocities (vx, vy), caster energy, and latent intent Z_p, but
                   never directly teleport (1, x, y, hp, shield).

Self-supervised training objective:
    1. Grounded velocity residual L_obs on the 76 grounded coordinates x_obs(t+1)
    2. Stop-gradient latent consistency L_lat on the 200 latent coordinates x_lat(t+1)
    3. Adversarial goal alignment L_goal: when player p acts (u_p != 0), their port
       push B_p u_p on the value-bearing coordinates aligns with their advantage
       gradient ∇_{x_p} A_p = ∇_{x_p} (H_p - H_{-p}).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------- phase-space layout

GROUND, P1, P2 = 0, 1, 2
OWNERS = 3
PLAYERS = 2

# Per-entity grounded columns Q_p: [M, R, K_Q] for playing players (P1, P2)
Q_COLS = ("1", "x", "y", "vx", "vy", "hp", "shield", "energy")
Q = {n: i for i, n in enumerate(Q_COLS)}
K_Q = len(Q_COLS)  # 8

# Kinematic columns: weighted to 0 in the strategic/tactical advantage Hamiltonian H_p
KIN_COLS = ("x", "y", "vx", "vy")
VAL_COLS = ("1", "hp", "shield", "energy")
ACTUATED_Q = ("vx", "vy", "energy")

# Player ledger coordinates in x_obs_p
LEDGER_COLS = ("minerals", "vespene")
PL = {n: i for i, n in enumerate(LEDGER_COLS)}
N_LEDGER = len(LEDGER_COLS)  # 2

# Fixed Gram normalization constant N_0 (never live N, so unit counts are preserved)
N0 = 100.0
BANK_SCALE = 1000.0
SPEED_SCALE = 0.29
ENERGY_UNIT = 200.0

# Default latent tactical mode rank r
R_LATENT = 8
N_QQ = K_Q * (K_Q + 1) // 2  # 36 grounded Gram entries per player
N_OBS_P = N_LEDGER + N_QQ  # 38 grounded entries per player


def _triu_indices(n):
    return torch.triu_indices(n, n)


def gram_layout(r=R_LATENT):
    """Computes sizes and masks for one player's phase-space vector x_p:
        [0 : N_LEDGER]                          minerals, vespene (2)
        [N_LEDGER : N_OBS_P]                    upper triangle of G_qq (36)
        [N_OBS_P : N_OBS_P + K_Q * r]           flattened G_qz (8 * r)
        [N_OBS_P + K_Q * r : D_p - r]           upper triangle of G_zz (r * (r + 1) // 2)
        [D_p - r : D_p]                         player + CLS global latent z_ply (r)
    Returns a dict of index slices, masks, and per-coordinate loss weights.
    """
    n_qz = K_Q * r
    n_zz = r * (r + 1) // 2
    n_ply = r
    d_p = N_OBS_P + n_qz + n_zz + n_ply

    ii_q, jj_q = _triu_indices(K_Q)
    kin_set = {Q[n] for n in KIN_COLS}
    act_set = {Q[n] for n in ACTUATED_Q}

    # w_adv: 1.0 on value/posture dimensions, 0.0 on pure kinematic dimensions (x, y, vx, vy)
    w_adv = torch.ones(d_p, dtype=torch.float32)
    for k in range(N_QQ):
        i, j = int(ii_q[k]), int(jj_q[k])
        if i in kin_set or j in kin_set:
            w_adv[N_LEDGER + k] = 0.0
    for i in range(K_Q):
        if i in kin_set:
            w_adv[N_OBS_P + i * r : N_OBS_P + (i + 1) * r] = 0.0

    # b_mask: which coordinates a player's command port B_p u_p may push directly.
    # Allowed: ledger (spending/cancels), any G_qq / G_qz entry involving actuated
    # (vx, vy, energy), and latent posture (G_qz, G_zz).
    # Blocked (0): G_qq entries over only (1, x, y, hp, shield).
    b_mask = torch.zeros(d_p, dtype=torch.float32)
    b_mask[:N_LEDGER] = 1.0
    for k in range(N_QQ):
        i, j = int(ii_q[k]), int(jj_q[k])
        if i in act_set or j in act_set:
            b_mask[N_LEDGER + k] = 1.0
    b_mask[N_OBS_P:] = 1.0

    # Named diagonal/first-moment indices inside x_obs_p for clear logging & weighting
    diag_idx = {}
    first_idx = {}
    for k in range(N_QQ):
        i, j = int(ii_q[k]), int(jj_q[k])
        if i == j:
            diag_idx[Q_COLS[i]] = N_LEDGER + k
        if i == 0:
            first_idx[Q_COLS[j]] = N_LEDGER + k

    # Per-coordinate step weight on x_obs_p so fast-changing velocity moments and
    # small-delta position/HP/ledger moments have balanced gradient scale
    col_step_weight = {
        "1": 20.0,
        "x": 40.0,
        "y": 40.0,
        "vx": 5.0,
        "vy": 5.0,
        "hp": 25.0,
        "shield": 25.0,
        "energy": 20.0,
    }
    w_obs = torch.ones(N_OBS_P, dtype=torch.float32)
    w_obs[PL["minerals"]] = 20.0  # 5-mineral trip * 20 / 1000 = 0.1
    w_obs[PL["vespene"]] = 20.0
    for k in range(N_QQ):
        i, j = int(ii_q[k]), int(jj_q[k])
        wi = col_step_weight[Q_COLS[i]]
        wj = col_step_weight[Q_COLS[j]]
        w_obs[N_LEDGER + k] = (wi * wj) ** 0.5

    return dict(
        r=r,
        n_qz=n_qz,
        n_zz=n_zz,
        n_ply=n_ply,
        d_p=d_p,
        ii_q=ii_q,
        jj_q=jj_q,
        w_adv=w_adv,
        b_mask=b_mask,
        diag_idx=diag_idx,
        first_idx=first_idx,
        w_obs=w_obs,
    )


@dataclass
class State:
    """M frames of fixed-dimension Gram phase-space coordinates and control inputs."""

    q: torch.Tensor  # [M, R, K_Q] fp32: per-unit grounded columns
    z: torch.Tensor  # [M, R, r] fp32: per-unit projected latent tactical modes
    owner: torch.Tensor  # [M, R] long: GROUND (0), P1 (1), P2 (2)
    mask: torch.Tensor  # [M, R] bool: live on-map row
    p: torch.Tensor  # [M, 2, N_LEDGER] fp32: per-player (minerals, vespene) / 1000
    u: torch.Tensor  # [M, 2, u_dim] fp32: per-player sum-pooled commands at frame t
    u_active: torch.Tensor  # [M, 2] bool: whether player p issued an action / order change
    z_ply: torch.Tensor | None = None  # [M, 2, r] fp32: per-player + CLS latent
    x_obs: torch.Tensor | None = None  # [M, 2, N_OBS_P] fp32 cached grounded state
    x_lat: torch.Tensor | None = None  # [M, 2, D_p - N_OBS_P] fp32 cached latent state
    loop: float = 1.0


def build_gram_state(q, z, owner, mask, p, z_ply=None, r=R_LATENT):
    """Computes (x_obs [M, 2, 38], x_lat [M, 2, 108]) in fp32 from per-unit Q, Z
    and per-player global latent z_ply.

    For each player pl in (P1, P2):
        Y_pl = [Q_pl | Z_pl] masked to live rows of player pl
        G_pl = (1 / N0) Y_pl^T Y_pl
    """
    q = q.float()
    z = z.float()
    p = p.float()
    M = q.shape[0]
    if z_ply is None:
        z_ply = torch.zeros(M, 2, r, dtype=torch.float32, device=q.device)
    else:
        z_ply = z_ply.float()
    ii_q, jj_q = _triu_indices(K_Q)
    ii_z, jj_z = _triu_indices(r)

    obs_list, lat_list = [], []
    for pl in (P1, P2):
        m = (mask & (owner == pl)).unsqueeze(-1).float()  # [M, R, 1]
        qp = q * m  # [M, R, K_Q]
        zp = z * m  # [M, R, r]
        g_qq = torch.matmul(qp.transpose(-1, -2), qp) / N0  # [M, K_Q, K_Q]
        g_qz = torch.matmul(qp.transpose(-1, -2), zp) / N0  # [M, K_Q, r]
        g_zz = torch.matmul(zp.transpose(-1, -2), zp) / N0  # [M, r, r]

        qq_vec = g_qq[:, ii_q, jj_q]  # [M, 36]
        qz_vec = g_qz.flatten(-2, -1)  # [M, 8 * r]
        zz_vec = g_zz[:, ii_z, jj_z]  # [M, r * (r + 1) // 2]
        zp_vec = z_ply[:, pl - 1]  # [M, r]

        obs_list.append(torch.cat([p[:, pl - 1, :N_LEDGER], qq_vec], dim=-1))
        lat_list.append(torch.cat([qz_vec, zz_vec, zp_vec], dim=-1))

    return torch.stack(obs_list, dim=1), torch.stack(lat_list, dim=1)


def mlp(d_in, d_out, hidden, bias=True):
    return nn.Sequential(
        nn.Linear(d_in, hidden),
        nn.LayerNorm(hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, d_out, bias=bias),
    )


def small_init(linear, std=1e-3):
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


# ---------------------------------------------------------------- the Hamiltonian


class Hamiltonian(nn.Module):
    """Player-symmetric Tug-of-War Hamiltonian over x = (x_1, x_2) in fp32.

    Strategic/Tactical Energy (with kinematic coordinates (x, y, vx, vy) weighted to 0
    via w_adv):
        H_1(x) = E_own(x̃_1) + E_tug(x̃_1, x̃_2)
        H_2(x) = E_own(x̃_2) - E_tug(x̃_1, x̃_2)
    where E_tug(a, b) = 0.5 * (psi(a, b) - psi(b, a)) is strictly antisymmetric.
    The zero-sum Advantage metric is:
        A_1(x) = H_1(x) - H_2(x) = (E_own(x̃_1) - E_own(x̃_2)) + 2 E_tug(x̃_1, x̃_2)

    A separate local kinematic potential H_kin(x_p) drives only the kinematic entries
    of x_p so pathing kinematics are predicted without polluting A_1(x).
    """

    def __init__(self, r=R_LATENT, hidden=256):
        super().__init__()
        layout = gram_layout(r)
        self.d_p = layout["d_p"]
        self.register_buffer("w_adv", layout["w_adv"])
        self.register_buffer("w_kin", 1.0 - layout["w_adv"])

        self.own_net = mlp(self.d_p, 1, hidden, bias=False)
        self.tug_net = mlp(2 * self.d_p, 1, hidden, bias=False)
        self.kin_net = mlp(self.d_p, 1, hidden, bias=False)
        small_init(self.own_net[-1], std=1e-3)
        small_init(self.tug_net[-1], std=1e-3)
        small_init(self.kin_net[-1], std=1e-3)

    def forward(self, x1, x2):
        """x1, x2: [M, D_p] fp32 -> dict with H1, H2, A1, E_own, E_tug, H_kin."""
        x1_adv = x1 * self.w_adv
        x2_adv = x2 * self.w_adv

        own1 = self.own_net(x1_adv).squeeze(-1)
        own2 = self.own_net(x2_adv).squeeze(-1)

        tug_12 = self.tug_net(torch.cat([x1_adv, x2_adv], dim=-1)).squeeze(-1)
        tug_21 = self.tug_net(torch.cat([x2_adv, x1_adv], dim=-1)).squeeze(-1)
        tug = 0.5 * (tug_12 - tug_21)

        H1 = own1 + tug
        H2 = own2 - tug
        A1 = H1 - H2

        # Local kinematic potential: its gradient is projected onto w_kin so it only
        # generates flows on the kinematic coordinates (x, y, vx, vy)
        kin1 = self.kin_net(x1).squeeze(-1)
        kin2 = self.kin_net(x2).squeeze(-1)

        return dict(
            H1=H1,
            H2=H2,
            A1=A1,
            own1=own1,
            own2=own2,
            tug=tug,
            kin1=kin1,
            kin2=kin2,
        )


# ---------------------------------------------------------------- the Dynamics


class PortHamiltonianDynamics(nn.Module):
    """One step of ẋ = (J(x) - R(x)) g(x) + B(x) u in fixed dimension D = 2 * D_p.

    Player-symmetric construction:
        For player 1 with perspective s_1 = [x_1, x_2] and player 2 with s_2 = [x_2, x_1]:
        - J_own(s_p) = A(s_p) - A(s_p)^T          (skew-symmetric within player p)
        - J_cross(x_1, x_2) = C(s_1) - C(s_2)^T   (skew-symmetric across players)
        - R_own(s_p) = L(s_p) L(s_p)^T            (PSD dissipation on player p)
        - B(s_p) u_p                              (masked control port, 0 when u_p = 0)
    """

    def __init__(self, u_dim, r=R_LATENT, hidden=256, rank=16):
        super().__init__()
        layout = gram_layout(r)
        self.d_p = layout["d_p"]
        self.rank = rank
        self.register_buffer("w_adv", layout["w_adv"])
        self.register_buffer("w_kin", 1.0 - layout["w_adv"])
        self.register_buffer("b_mask", layout["b_mask"])

        # Base O(0.1) factors + state-dependent low-rank factors for J_own, J_cross, R_own
        # so J and R are O(0.01) at step 0 (instead of 1e-6), allowing gradients to flow
        # immediately into H and the encoder while H's 1e-4 init starts at copy baseline.
        scale = (self.d_p * rank) ** -0.25 * 0.2
        self.j_own_u_base = nn.Parameter(torch.randn(self.d_p, rank) * scale)
        self.j_own_v_base = nn.Parameter(torch.randn(self.d_p, rank) * scale)
        self.j_cross_u_base = nn.Parameter(torch.randn(self.d_p, rank) * scale)
        self.j_cross_v_base = nn.Parameter(torch.randn(self.d_p, rank) * scale)
        self.r_own_l_base = nn.Parameter(torch.randn(self.d_p, rank) * scale)

        self.j_own_u = mlp(2 * self.d_p, self.d_p * rank, hidden)
        self.j_own_v = mlp(2 * self.d_p, self.d_p * rank, hidden)
        self.j_cross_u = mlp(2 * self.d_p, self.d_p * rank, hidden)
        self.j_cross_v = mlp(2 * self.d_p, self.d_p * rank, hidden)
        self.r_own_l = mlp(2 * self.d_p, self.d_p * rank, hidden)

        small_init(self.j_own_u[-1], std=1e-3)
        small_init(self.j_own_v[-1], std=1e-3)
        small_init(self.j_cross_u[-1], std=1e-3)
        small_init(self.j_cross_v[-1], std=1e-3)
        small_init(self.r_own_l[-1], std=1e-3)

        # Control port B(s_p) u_p: linear in u_p (bias=False so u_p = 0 -> B u_p = 0)
        self.b_state = mlp(2 * self.d_p, self.d_p * u_dim, hidden)
        self.b_base = nn.Linear(u_dim, self.d_p, bias=False)
        small_init(self.b_state[-1], std=0.0)
        small_init(self.b_base, std=0.0)

    def _skew_apply(self, u_net, v_net, u_base, v_base, s, g):
        """Applies (U V^T - V U^T) g without materializing D_p x D_p if not needed."""
        M = s.shape[0]
        U = u_base + u_net(s).view(M, self.d_p, self.rank)
        V = v_base + v_net(s).view(M, self.d_p, self.rank)
        # (U V^T - V U^T) g = U (V^T g) - V (U^T g)
        vt_g = torch.matmul(V.transpose(-1, -2), g.unsqueeze(-1))
        ut_g = torch.matmul(U.transpose(-1, -2), g.unsqueeze(-1))
        return (torch.matmul(U, vt_g) - torch.matmul(V, ut_g)).squeeze(-1)

    def _cross_apply(self, s1, s2, g1, g2):
        """Applies J_cross to g2 and -J_cross^T to g1 where J_cross = C(s1) - C(s2)^T."""
        M = s1.shape[0]
        U1 = self.j_cross_u_base + self.j_cross_u(s1).view(M, self.d_p, self.rank)
        V1 = self.j_cross_v_base + self.j_cross_v(s1).view(M, self.d_p, self.rank)
        U2 = self.j_cross_u_base + self.j_cross_u(s2).view(M, self.d_p, self.rank)
        V2 = self.j_cross_v_base + self.j_cross_v(s2).view(M, self.d_p, self.rank)
        f1 = torch.matmul(U1, torch.matmul(V1.transpose(-1, -2), g2.unsqueeze(-1))) - (
            torch.matmul(V2, torch.matmul(U2.transpose(-1, -2), g2.unsqueeze(-1)))
        )
        f2 = torch.matmul(U2, torch.matmul(V2.transpose(-1, -2), g1.unsqueeze(-1))) - (
            torch.matmul(V1, torch.matmul(U1.transpose(-1, -2), g1.unsqueeze(-1)))
        )
        return f1.squeeze(-1), f2.squeeze(-1)

    def _psd_apply(self, s, g):
        """Applies R(s) g = L(s) L(s)^T g, guaranteed >= 0 in inner product with g."""
        M = s.shape[0]
        L = self.r_own_l_base + self.r_own_l(s).view(M, self.d_p, self.rank)
        lt_g = torch.matmul(L.transpose(-1, -2), g.unsqueeze(-1))
        r_g = torch.matmul(L, lt_g).squeeze(-1)
        dissipated = lt_g.squeeze(-1).square().sum(-1)  # ||L^T g||^2 >= 0
        return r_g, dissipated

    def port_push(self, s, u):
        """Computes B(s) u [M, D_p], exact 0 when u == 0, masked by b_mask."""
        M, u_dim = u.shape
        B_dyn = self.b_state(s).view(M, self.d_p, u_dim)
        bu = self.b_base(u) + torch.matmul(B_dyn, u.unsqueeze(-1)).squeeze(-1)
        return bu * self.b_mask

    def _apply_all(self, s1, s2, g1, g2, u1, u2):
        """Batches all J, R, and B evaluations across both players [2 * M, 2 * d_p] to
        halve kernel launches and double GEMM batch sizes."""
        M, u_dim = u1.shape
        s_both = torch.cat([s1, s2], dim=0)
        g_both = torch.cat([g1, g2], dim=0).unsqueeze(-1)
        g_swap = torch.cat([g2, g1], dim=0).unsqueeze(-1)
        u_both = torch.cat([u1, u2], dim=0)

        # 1. J_own(s_p) g_p = (U_own V_own^T - V_own U_own^T) g_p
        Uo = self.j_own_u_base + self.j_own_u(s_both).view(2 * M, self.d_p, self.rank)
        Vo = self.j_own_v_base + self.j_own_v(s_both).view(2 * M, self.d_p, self.rank)
        j_own = (
            torch.matmul(Uo, torch.matmul(Vo.transpose(-1, -2), g_both))
            - torch.matmul(Vo, torch.matmul(Uo.transpose(-1, -2), g_both))
        ).squeeze(-1)
        j1_own, j2_own = j_own.chunk(2, dim=0)

        # 2. J_cross: f1 = (U1 V1^T - V2 U2^T) g2, f2 = (U2 V2^T - V1 U1^T) g1
        Uc = self.j_cross_u_base + self.j_cross_u(s_both).view(
            2 * M, self.d_p, self.rank
        )
        Vc = self.j_cross_v_base + self.j_cross_v(s_both).view(
            2 * M, self.d_p, self.rank
        )
        Uc_swap = torch.cat([Uc[M:], Uc[:M]], dim=0)
        Vc_swap = torch.cat([Vc[M:], Vc[:M]], dim=0)
        j_cross = (
            torch.matmul(Uc, torch.matmul(Vc.transpose(-1, -2), g_swap))
            - torch.matmul(Vc_swap, torch.matmul(Uc_swap.transpose(-1, -2), g_swap))
        ).squeeze(-1)
        j1_cross, j2_cross = j_cross.chunk(2, dim=0)

        # 3. R(s_p) g_p = L L^T g_p
        L = self.r_own_l_base + self.r_own_l(s_both).view(2 * M, self.d_p, self.rank)
        lt_g = torch.matmul(L.transpose(-1, -2), g_both)
        r_both = torch.matmul(L, lt_g).squeeze(-1)
        diss_both = lt_g.squeeze(-1).square().sum(-1)
        r1, r2 = r_both.chunk(2, dim=0)
        diss1, diss2 = diss_both.chunk(2, dim=0)

        # 4. Control port B(s_p) u_p
        B_dyn = self.b_state(s_both).view(2 * M, self.d_p, u_dim)
        bu_both = (
            self.b_base(u_both) + torch.matmul(B_dyn, u_both.unsqueeze(-1)).squeeze(-1)
        ) * self.b_mask
        bu1, bu2 = bu_both.chunk(2, dim=0)

        return j1_own + j1_cross, j2_own + j2_cross, r1, r2, diss1, diss2, bu1, bu2

    def forward(self, x1, x2, u1, u2, H_mod, dt=1.0):
        """Steps (x1, x2) by dt -> (x1_next, x2_next, pieces).
        Keeps x1, x2 connected to the encoder graph during training."""
        x1_req = x1 if x1.requires_grad else x1.detach().requires_grad_(True)
        x2_req = x2 if x2.requires_grad else x2.detach().requires_grad_(True)

        h_out = H_mod(x1_req, x2_req)
        cg = self.training
        # 3 non-overlapping autograd.grad calls instead of 6:
        # H1 = own1 + tug, H2 = own2 - tug, A1 = own1 - own2 + 2 * tug
        g_own1, g_own2 = torch.autograd.grad(
            h_out["own1"].sum() + h_out["own2"].sum(),
            (x1_req, x2_req),
            create_graph=cg,
            retain_graph=True,
        )
        g_tug1, g_tug2 = torch.autograd.grad(
            h_out["tug"].sum(),
            (x1_req, x2_req),
            create_graph=cg,
            retain_graph=True,
        )
        g_kin1, g_kin2 = torch.autograd.grad(
            h_out["kin1"].sum() + h_out["kin2"].sum(),
            (x1_req, x2_req),
            create_graph=cg,
            retain_graph=cg,
        )

        g1_adv = g_own1 + g_tug1
        g2_adv = g_own2 - g_tug2
        g1_goal = g_own1 + 2.0 * g_tug1
        g2_goal = g_own2 - 2.0 * g_tug2
        g1_kin = g_kin1 * self.w_kin
        g2_kin = g_kin2 * self.w_kin

        g1 = g1_adv + g1_kin
        g2 = g2_adv + g2_kin

        s1 = torch.cat([x1_req, x2_req], dim=-1)
        s2 = torch.cat([x2_req, x1_req], dim=-1)

        j1, j2, r1, r2, diss1, diss2, bu1, bu2 = self._apply_all(
            s1, s2, g1, g2, u1, u2
        )

        dx1 = (j1 - r1) + bu1
        dx2 = (j2 - r2) + bu2

        x1_next = x1_req + dx1 * dt
        x2_next = x2_req + dx2 * dt

        return (
            x1_next,
            x2_next,
            dict(
                dx1=dx1,
                dx2=dx2,
                bu1=bu1,
                bu2=bu2,
                g1=g1,
                g2=g2,
                g1_goal=g1_goal,
                g2_goal=g2_goal,
                j1=j1,
                j2=j2,
                r1=r1,
                r2=r2,
                diss1=diss1,
                diss2=diss2,
                **h_out,
            ),
        )


# ---------------------------------------------------------------- the Objective


class EnergyModel(nn.Module):
    """Port-Hamiltonian model in fixed-dimension Gram phase space."""

    def __init__(self, d, u_dim, hidden=256, r=R_LATENT, rank=16, w_lat=0.1, w_goal=0.05):
        super().__init__()
        self.r = r
        self.w_lat = w_lat
        self.w_goal = w_goal
        layout = gram_layout(r)
        self.d_p = layout["d_p"]
        self.diag_idx = layout["diag_idx"]
        self.first_idx = layout["first_idx"]
        self.register_buffer("w_obs", layout["w_obs"])
        self.register_buffer("w_adv", layout["w_adv"])

        self.H = Hamiltonian(r=r, hidden=hidden)
        self.dynamics = PortHamiltonianDynamics(
            u_dim=u_dim, r=r, hidden=hidden, rank=rank
        )

    def compile_parts(self):
        """Compiles the batched J, R, B dynamics operators."""
        self.dynamics._apply_all = torch.compile(self.dynamics._apply_all, dynamic=True)

    def encode_phase_space(self, s):
        if s.x_obs is not None and s.x_lat is not None:
            return s.x_obs, s.x_lat
        return build_gram_state(s.q, s.z, s.owner, s.mask, s.p, s.z_ply, r=self.r)

    def step(self, s):
        x_obs, x_lat = self.encode_phase_space(s)
        x = torch.cat([x_obs, x_lat], dim=-1)  # [M, 2, D_p]
        x1, x2 = x[:, 0], x[:, 1]
        u1, u2 = s.u[:, 0].float(), s.u[:, 1].float()
        x1_next, x2_next, pieces = self.dynamics(x1, x2, u1, u2, self.H, dt=s.loop)
        x_next = torch.stack([x1_next, x2_next], dim=1)
        return x_next[:, :, :N_OBS_P], x_next[:, :, N_OBS_P:], pieces

    def _goal_loss(self, g_goal, bu, active):
        """Adversarial goal alignment: when a player acts (active == True), the energy-
        carrying part of B_p u_p (detached so B is grounded strictly by physical state
        deltas in L_obs) aligns positively with ∇_{x_p} A_p."""
        bu_adv = (bu.detach()) * self.w_adv
        # Cosine similarity on value-bearing dimensions
        gn = g_goal.norm(dim=-1)
        bn = bu_adv.norm(dim=-1)
        valid = active & (gn > 1e-8) & (bn > 1e-8)
        cos = (g_goal * bu_adv).sum(dim=-1) / (gn * bn).clamp(min=1e-8)
        # Softplus margin encouraging positive alignment (> 0), without CPU-GPU sync (.any())
        per_frame = F.softplus(-2.0 * cos)
        vf = valid.float()
        denom = vf.sum().clamp(min=1.0)
        loss = (per_frame * vf).sum() / denom
        mean_cos = (cos * vf).sum() / denom
        return loss, mean_cos.detach()

    def forward(self, s, s_next):
        """s: current frame State; s_next: next frame State -> (loss, logs)."""
        x_obs, x_lat = self.encode_phase_space(s)
        x_obs_next, x_lat_next = self.encode_phase_space(s_next)

        x_obs_pred, x_lat_pred, pieces = self.step(s)

        # 1. Grounded phase-space velocity loss (76 coordinates across both players)
        diff_obs = (x_obs_pred - x_obs_next) * self.w_obs
        per_coord_obs = 0.5 * diff_obs.square().mean(dim=(0, 1))  # [N_OBS_P]
        loss_obs = per_coord_obs.sum()

        # Copy baseline (predicting x_obs_next = x_obs)
        diff_copy = (x_obs - x_obs_next) * self.w_obs
        per_coord_copy = 0.5 * diff_copy.square().mean(dim=(0, 1))

        # 2. Stop-gradient latent consistency loss (200 coordinates across both players)
        loss_lat = F.smooth_l1_loss(x_lat_pred, x_lat_next.detach())

        # 3. Adversarial goal alignment loss on active command frames
        l_goal1, cos1 = self._goal_loss(
            pieces["g1_goal"], pieces["bu1"], s.u_active[:, 0]
        )
        l_goal2, cos2 = self._goal_loss(
            pieces["g2_goal"], pieces["bu2"], s.u_active[:, 1]
        )
        loss_goal = 0.5 * (l_goal1 + l_goal2)
        goal_cos = 0.5 * (cos1 + cos2)

        loss = loss_obs + self.w_lat * loss_lat + self.w_goal * loss_goal

        # Detailed per-axis diagnostics (against copy baseline)
        logs = {
            "loss_obs": loss_obs.detach(),
            "loss_lat": loss_lat.detach(),
            "loss_goal": loss_goal.detach(),
            "goal_cos": goal_cos,
            "copy_loss_obs": per_coord_copy.sum().detach(),
            "mse_minerals": 2.0 * per_coord_obs[PL["minerals"]].detach(),
            "mse_vespene": 2.0 * per_coord_obs[PL["vespene"]].detach(),
            "copy_mse_minerals": 2.0 * per_coord_copy[PL["minerals"]].detach(),
            "copy_mse_vespene": 2.0 * per_coord_copy[PL["vespene"]].detach(),
        }
        for col in Q_COLS:
            idx = self.first_idx[col]
            d_idx = self.diag_idx[col]
            val = 0.5 * (per_coord_obs[idx] + per_coord_obs[d_idx]).detach()
            cval = 0.5 * (per_coord_copy[idx] + per_coord_copy[d_idx]).detach()
            if col in ("hp", "shield"):
                logs[f"mse_{col}"] = 2.0 * val
                logs[f"copy_mse_{col}"] = 2.0 * cval
            else:
                logs[f"res_{col}"] = val
                logs[f"copy_res_{col}"] = cval

        logs["H_1"] = pieces["H1"].mean().detach()
        logs["H_2"] = pieces["H2"].mean().detach()
        logs["advantage"] = pieces["A1"].mean().detach()
        logs["tug"] = pieces["tug"].mean().detach()
        logs["dissipated_1"] = pieces["diss1"].mean().detach()
        logs["dissipated_2"] = pieces["diss2"].mean().detach()
        return loss, logs
