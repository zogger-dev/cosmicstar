"""Layer 2 (5-minute = 60 x 5s stride) Hierarchical Port-Hamiltonian World Model.

Couples additively onto Layer 1's fast tactical energy field:
    H_{p,total}(tau) = H_p^(1)(x_tau) + H_p^(2)(X_tau)
    A_{1,total}(tau) = A_1^(1)(x_tau) + A_1^(2)(X_tau)

Consumes strictly the un-bottlenecked latents from Layer 1 (no manual macro ledgers):
    - cls_out:   [B, T, 256]          (Layer 1 global scene token before r=8 projection)
    - ply_out:   [B, T, 2, 256]       (Layer 1 player tokens before r=8 projection)
    - G_qq:      [B, T, 2, 36]        (grounded 8x8 unit Gram Q_p^T Q_p / N0)
    - G_qc:      [B, T, 2, 8, 256]    (full-rank 8x256 Cross-Gram Q_p^T C_p / N0)
    - G_cc_triu: [B, T, 2, 32896]     (full-rank 256x256 unit Gram C_p^T C_p / N0)
    - u_win, act_feat, u_active       (5-second window control impulses)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmicstar.worldmodel import energy as E

N_OBS_L2 = E.N_QQ  # 36 grounded unit Gram entries (no manual macro ledgers)
D_L1 = 256
R_GRAM_L2 = 16     # learned bilinear congruence rank over full 256x256 G_cc
R_MACRO_L2 = 32    # causal 60-step (5-min) history latent channels per player
E_SCALE_L1 = 1e-3  # normalizes Layer 1 H^(1) (~1e-3) to O(1) dimensionless units


def l2_layout(r_gram=R_GRAM_L2, r_macro=R_MACRO_L2):
    """Phase-space layout for one player's Layer 2 coordinate X_{tau, p}:
    [ G_qq (36) | G_qc_proj (8 * r_gram) | G_cc_proj (r_gram*(r_gram+1)//2) | z_macro (r_macro) ].
    """
    ii_q, jj_q = E._triu_indices(E.K_Q)
    ii_g, jj_g = E._triu_indices(r_gram)
    n_qc = E.K_Q * r_gram
    n_cc = len(ii_g)
    n_lat = n_qc + n_cc + r_macro
    d_p = N_OBS_L2 + n_lat

    kin_set = {E.Q_COLS.index(c) for c in E.KIN_COLS}
    act_set = {E.Q_COLS.index(c) for c in E.ACTUATED_Q}

    # Zero out purely kinematic (x, y, vx, vy) entries in strategic Hamiltonian H^(2)
    w_adv = torch.ones(d_p, dtype=torch.float32)
    for k in range(N_OBS_L2):
        i, j = int(ii_q[k]), int(jj_q[k])
        if i in kin_set or j in kin_set:
            w_adv[k] = 0.0
    for row in range(E.K_Q):
        if row in kin_set:
            w_adv[N_OBS_L2 + row * r_gram : N_OBS_L2 + (row + 1) * r_gram] = 0.0

    # Block direct control-port teleportation of (1, x, y, hp, shield) in G_qq, matching Layer 1:
    # player commands u_p inject impulse into (vx, vy, energy) and latent strategic posture x_lat,
    # while all unit count, position, HP, and shield evolution must flow through (J - R) grad H!
    b_mask = torch.zeros(d_p, dtype=torch.float32)
    for k in range(N_OBS_L2):
        i, j = int(ii_q[k]), int(jj_q[k])
        if i in act_set or j in act_set:
            b_mask[k] = 1.0
    b_mask[N_OBS_L2:] = 1.0

    diag_idx = {}
    first_idx = {}
    for k in range(N_OBS_L2):
        i, j = int(ii_q[k]), int(jj_q[k])
        if i == j:
            diag_idx[E.Q_COLS[i]] = k
        if i == 0:
            first_idx[E.Q_COLS[j]] = k

    # Step weights for 5-second deltas on G_qq
    col_step_weight = {
        "1": 10.0,
        "x": 15.0,
        "y": 15.0,
        "vx": 4.0,
        "vy": 4.0,
        "hp": 12.0,
        "shield": 12.0,
        "energy": 10.0,
    }
    w_obs = torch.ones(N_OBS_L2, dtype=torch.float32)
    for k in range(N_OBS_L2):
        i, j = int(ii_q[k]), int(jj_q[k])
        wi = col_step_weight[E.Q_COLS[i]]
        wj = col_step_weight[E.Q_COLS[j]]
        w_obs[k] = (wi * wj) ** 0.5

    return dict(
        r_gram=r_gram,
        r_macro=r_macro,
        n_qc=n_qc,
        n_cc=n_cc,
        n_lat=n_lat,
        d_p=d_p,
        ii_q=ii_q,
        jj_q=jj_q,
        ii_g=ii_g,
        jj_g=jj_g,
        w_adv=w_adv,
        b_mask=b_mask,
        diag_idx=diag_idx,
        first_idx=first_idx,
        w_obs=w_obs,
    )


class FullRankGramOperator(nn.Module):
    """Unpacks the full-rank 256x256 unit Gram matrix G_cc from its upper triangle
    [B, T, 2, 32896] and performs:
      1) Exact full-rank matrix-vector operator queries:
           - diag(G_cc)                [B, T, 2, 256]
           - G_cc @ ply_out_p          [B, T, 2, 256]
           - G_cc @ ply_out_opp        [B, T, 2, 256]
           - G_cc @ cls_out            [B, T, 2, 256]
      2) Learned strategic bilinear congruence W_c^T G_cc W_c [B, T, 2, r_gram, r_gram]
         and Cross-Gram projection G_qc @ W_c [B, T, 2, 8, r_gram].
    """

    def __init__(self, d_l1=D_L1, r_gram=R_GRAM_L2):
        super().__init__()
        self.d_l1 = d_l1
        self.r_gram = r_gram

        ii_c, jj_c = E._triu_indices(d_l1)
        sym_idx = torch.zeros((d_l1, d_l1), dtype=torch.long)
        arange_k = torch.arange(len(ii_c), dtype=torch.long)
        sym_idx[ii_c, jj_c] = arange_k
        sym_idx[jj_c, ii_c] = arange_k
        diag_c_idx = sym_idx[torch.arange(d_l1), torch.arange(d_l1)]

        ii_g, jj_g = E._triu_indices(r_gram)
        self.register_buffer("sym_idx", sym_idx, persistent=False)
        self.register_buffer("diag_c_idx", diag_c_idx, persistent=False)
        self.register_buffer("ii_g", ii_g, persistent=False)
        self.register_buffer("jj_g", jj_g, persistent=False)

        # Orthogonal strategic probe basis W_c in R^{256 x r_gram}
        self.w_c = nn.Parameter(torch.empty(d_l1, r_gram))
        nn.init.orthogonal_(self.w_c)

    def forward(self, g_qc, g_cc_triu, cls_out, ply_out):
        """
        g_qc:      [B, T, 2, 8, 256]
        g_cc_triu: [B, T, 2, 32896]
        cls_out:   [B, T, 256]
        ply_out:   [B, T, 2, 256]
        """
        B, T, _, _ = g_cc_triu.shape
        # Materialize full symmetric [B, T, 2, 256, 256] G_cc in bf16/fp32
        G_cc = g_cc_triu[..., self.sym_idx].float()
        g_qc_f = g_qc.float()
        cls_f = cls_out.float()
        ply_f = ply_out.float()

        # 1. Exact full-rank 256-dim operator invariants
        cc_diag = g_cc_triu[..., self.diag_c_idx].float()  # [B, T, 2, 256]
        v_ply = torch.matmul(G_cc, ply_f.unsqueeze(-1)).squeeze(-1)  # [B, T, 2, 256]
        ply_opp = ply_f.flip(dims=[2])
        v_opp = torch.matmul(G_cc, ply_opp.unsqueeze(-1)).squeeze(-1)  # [B, T, 2, 256]
        v_cls = torch.matmul(
            G_cc, cls_f.unsqueeze(2).expand(B, T, 2, -1).unsqueeze(-1)
        ).squeeze(-1)

        # 2. Learned strategic bilinear congruence W_c^T G_cc W_c and G_qc W_c
        # Normalize columns of W_c so congruence stays well-scaled
        w = F.normalize(self.w_c, dim=0) * (self.d_l1 / self.r_gram) ** 0.25
        qc_proj = torch.matmul(g_qc_f, w)  # [B, T, 2, 8, r_gram]
        cc_w = torch.matmul(G_cc, w)       # [B, T, 2, 256, r_gram]
        cc_proj = torch.matmul(w.t(), cc_w)  # [B, T, 2, r_gram, r_gram]

        qc_vec = torch.tanh(qc_proj.flatten(-2, -1))
        cc_vec = torch.tanh(cc_proj[..., self.ii_g, self.jj_g])

        return dict(
            cc_diag=cc_diag,
            v_ply=v_ply,
            v_opp=v_opp,
            v_cls=v_cls,
            qc_vec=qc_vec,
            cc_vec=cc_vec,
        )


class CausalMacroBlock(nn.Module):
    """One layer of player-symmetric causal temporal attention over T <= 60 (5-minute)
    summary steps + cross-player interaction at each step tau."""

    def __init__(self, d=256, heads=8, ff=768, dropout=0.1):
        super().__init__()
        self.d = d
        self.heads = heads
        self.dh = d // heads

        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)

        self.norm_cross = nn.LayerNorm(d)
        self.cross_mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d, bias=False),
        )
        E.small_init(self.cross_mlp[-1], std=1e-2)

        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff, d, bias=False),
        )

    def forward(self, x, rope_cos, rope_sin):
        """x: [B, T, 2, d] -> [B, T, 2, d]. Strictly causal along T."""
        B, T, P, d = x.shape
        # 1. Causal temporal self-attention per player stream [B * 2, T, d]
        xp = x.transpose(1, 2).reshape(B * P, T, d)
        h = self.norm1(xp)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B * P, T, self.heads, self.dh).transpose(1, 2)
        k = k.view(B * P, T, self.heads, self.dh).transpose(1, 2)
        v = v.view(B * P, T, self.heads, self.dh).transpose(1, 2)

        # Apply 1D RoPE in macro step index tau
        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]
        q = torch.stack([q1 * rope_cos - q2 * rope_sin, q1 * rope_sin + q2 * rope_cos], dim=-1).flatten(-2)
        k = torch.stack([k1 * rope_cos - k2 * rope_sin, k1 * rope_sin + k2 * rope_cos], dim=-1).flatten(-2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B * P, T, d)
        xp = xp + self.drop(self.proj(attn_out))
        x = xp.view(B, P, T, d).transpose(1, 2)

        # 2. Symmetric same-step cross-player interaction at step tau
        xn = self.norm_cross(x)
        x_opp = xn.flip(dims=[2])
        x = x + self.cross_mlp(torch.cat([xn, x_opp], dim=-1))

        # 3. FFN
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class MacroSequenceEncoder(nn.Module):
    """Encodes a 5-minute sequence of 60 x 5s summary frames into Layer 2 phase-space
    coordinates X_{tau, p} and macro control vectors U_{tau, p}."""

    def __init__(self, d=256, layers=4, heads=8, r_gram=R_GRAM_L2, r_macro=R_MACRO_L2, u_dim=32):
        super().__init__()
        self.d = d
        self.r_gram = r_gram
        self.r_macro = r_macro
        self.u_dim = u_dim

        self.gram_op = FullRankGramOperator(d_l1=D_L1, r_gram=r_gram)

        # Per-step input features per player:
        # ply_out (256) + cls_out (256) + cc_diag (256) + v_ply (256) + v_opp (256) + v_cls (256)
        # + G_qc[0] unit-sum (256) + G_qc[5] hp-sum (256) + G_qc[6] shd-sum (256)
        # + G_qq (36) + qc_vec (8*r_gram) + cc_vec (r_gram*(r_gram+1)//2)
        n_qc = E.K_Q * r_gram
        n_cc = r_gram * (r_gram + 1) // 2
        in_dim = (9 * D_L1) + N_OBS_L2 + n_qc + n_cc
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        self.blocks = nn.ModuleList(
            [CausalMacroBlock(d=d, heads=heads, ff=3 * d) for _ in range(layers)]
        )
        self.out_norm = nn.LayerNorm(d)
        self.z_macro_proj = nn.Linear(d, r_macro)

        # Macro 5s window control encoder: combines Layer 1 u_win [32] + 5s act_feat [24]
        self.u_proj = nn.Sequential(
            nn.Linear(32 + 24, d // 2),
            nn.LayerNorm(d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, u_dim, bias=False),
        )

        # 1D RoPE frequencies for macro time steps tau = 0 .. 255
        dh = d // heads
        inv_freq = 1.0 / (1000.0 ** (torch.arange(0, dh, 2, dtype=torch.float32) / dh))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, batch):
        """Returns:
        x_obs: [B, T, 2, 36] fp32
        x_lat: [B, T, 2, n_lat] fp32
        u:     [B, T, 2, u_dim] fp32 (exact 0 when u_active is False)
        u_active: [B, T, 2] bool
        z_macro:  [B, T, 2, r_macro] fp32
        """
        g_qq = batch["G_qq"].float()          # [B, T, 2, 36]
        g_qc = batch["G_qc"].float()          # [B, T, 2, 8, 256]
        g_cc_triu = batch["G_cc_triu"]        # [B, T, 2, 32896]
        cls_out = batch["cls_out"].float()    # [B, T, 256]
        ply_out = batch["ply_out"].float()    # [B, T, 2, 256]
        u_win = batch["u_win"].float()        # [B, T, 2, 32]
        act_feat = batch["act_feat"].float()  # [B, T, 2, 24]
        u_active = batch["u_active"].bool()   # [B, T, 2]

        B, T, _, _ = g_qq.shape
        g_out = self.gram_op(g_qc, g_cc_triu, cls_out, ply_out)

        cls_exp = cls_out.unsqueeze(2).expand(B, T, 2, -1)
        # Rows 0 ('1'), 5 ('hp'), 6 ('shield') of full-rank Cross-Gram G_qc [B, T, 2, 8, 256]
        qc_row0 = g_qc[:, :, :, 0, :]
        qc_row5 = g_qc[:, :, :, 5, :]
        qc_row6 = g_qc[:, :, :, 6, :]

        tok_in = torch.cat(
            [
                ply_out,
                cls_exp,
                g_out["cc_diag"],
                g_out["v_ply"],
                g_out["v_opp"],
                g_out["v_cls"],
                qc_row0,
                qc_row5,
                qc_row6,
                g_qq,
                g_out["qc_vec"],
                g_out["cc_vec"],
            ],
            dim=-1,
        )
        h = self.in_proj(tok_in)  # [B, T, 2, d]

        t_pos = torch.arange(T, device=h.device, dtype=torch.float32)
        angles = torch.outer(t_pos, self.inv_freq)[None, None, :, :]  # [1, 1, T, dh//2]
        rope_cos = angles.cos()
        rope_sin = angles.sin()

        for blk in self.blocks:
            h = blk(h, rope_cos, rope_sin)

        # Soft-bounded strategic latent (scaled tanh in (-2, 2) so std=0.4 stays strictly linear)
        z_macro = 2.0 * torch.tanh(0.5 * self.z_macro_proj(self.out_norm(h)))

        x_obs = g_qq
        x_lat = torch.cat([g_out["qc_vec"], g_out["cc_vec"], z_macro], dim=-1)

        u_raw = self.u_proj(torch.cat([u_win, act_feat], dim=-1))
        u = u_raw * u_active.unsqueeze(-1).float()

        return x_obs, x_lat, u, u_active, z_macro


class EnergyModelL2(nn.Module):
    """Layer 2 Additive Port-Hamiltonian Model over 60 x 5s (5-minute) sequences.

    Total Multi-Scale Energy Field:
        H_{1,total}(tau) = H_1^(1)(tau) + H_1^(2)(X_tau)
        H_{2,total}(tau) = H_2^(1)(tau) + H_2^(2)(X_tau)
        A_{1,total}(tau) = A_1^(1)(tau) + A_1^(2)(X_tau)
    """

    def __init__(
        self,
        d=256,
        layers=4,
        heads=8,
        r_gram=R_GRAM_L2,
        r_macro=R_MACRO_L2,
        u_dim=32,
        hidden=256,
        rank=16,
        rollout_steps=4,
        w_lat=0.1,
        w_goal=0.05,
        w_reg=0.02,
        w_td=0.0,
    ):
        super().__init__()
        self.r_gram = r_gram
        self.r_macro = r_macro
        self.rollout_steps = rollout_steps
        self.w_lat = w_lat
        self.w_goal = w_goal
        self.w_reg = w_reg
        self.w_td = w_td

        layout = l2_layout(r_gram=r_gram, r_macro=r_macro)
        self.d_p = layout["d_p"]
        self.diag_idx = layout["diag_idx"]
        self.first_idx = layout["first_idx"]
        self.register_buffer("w_obs", layout["w_obs"])
        self.register_buffer("w_adv", layout["w_adv"])
        self.register_buffer("w_kin", 1.0 - layout["w_adv"])
        self.register_buffer("b_mask", layout["b_mask"])

        self.seq_enc = MacroSequenceEncoder(
            d=d, layers=layers, heads=heads, r_gram=r_gram, r_macro=r_macro, u_dim=u_dim
        )

        # Layer 2 Hamiltonian H^(2)(X_1, X_2)
        self.own_net = E.mlp(self.d_p, 1, hidden, bias=False)
        self.tug_net = E.mlp(2 * self.d_p, 1, hidden, bias=False)
        self.kin_net = E.mlp(self.d_p, 1, hidden, bias=False)
        E.small_init(self.own_net[-1], std=1e-3)
        E.small_init(self.tug_net[-1], std=1e-3)
        E.small_init(self.kin_net[-1], std=1e-3)

        # Layer 2 Port-Hamiltonian Operators J^(2), R^(2), B^(2)
        scale = (self.d_p * rank) ** -0.25 * 0.2
        self.rank = rank
        self.j_own_u_base = nn.Parameter(torch.randn(self.d_p, rank) * scale)
        self.j_own_v_base = nn.Parameter(torch.randn(self.d_p, rank) * scale)
        self.j_cross_u_base = nn.Parameter(torch.randn(self.d_p, rank) * scale)
        self.j_cross_v_base = nn.Parameter(torch.randn(self.d_p, rank) * scale)
        self.r_own_l_base = nn.Parameter(torch.randn(self.d_p, rank) * scale)

        self.j_own_u = E.mlp(2 * self.d_p, self.d_p * rank, hidden)
        self.j_own_v = E.mlp(2 * self.d_p, self.d_p * rank, hidden)
        self.j_cross_u = E.mlp(2 * self.d_p, self.d_p * rank, hidden)
        self.j_cross_v = E.mlp(2 * self.d_p, self.d_p * rank, hidden)
        self.r_own_l = E.mlp(2 * self.d_p, self.d_p * rank, hidden)
        E.small_init(self.j_own_u[-1], std=1e-3)
        E.small_init(self.j_own_v[-1], std=1e-3)
        E.small_init(self.j_cross_u[-1], std=1e-3)
        E.small_init(self.j_cross_v[-1], std=1e-3)
        E.small_init(self.r_own_l[-1], std=1e-3)

        self.b_state = E.mlp(2 * self.d_p, self.d_p * u_dim, hidden)
        self.b_base = nn.Linear(u_dim, self.d_p, bias=False)
        E.small_init(self.b_state[-1], std=0.0)
        E.small_init(self.b_base, std=0.0)

    def H_l2(self, x1, x2):
        """Evaluates Layer 2 slow Hamiltonian H^(2)(X_1, X_2)."""
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

        kin1 = self.kin_net(x1).squeeze(-1)
        kin2 = self.kin_net(x2).squeeze(-1)

        return dict(
            H1=H1, H2=H2, A1=A1, own1=own1, own2=own2, tug=tug, kin1=kin1, kin2=kin2
        )

    def _apply_operators(self, s1, s2, g1, g2, u1, u2):
        M, u_dim = u1.shape
        s_both = torch.cat([s1, s2], dim=0)
        g_both = torch.cat([g1, g2], dim=0).unsqueeze(-1)
        g_swap = torch.cat([g2, g1], dim=0).unsqueeze(-1)
        u_both = torch.cat([u1, u2], dim=0)

        Uo = self.j_own_u_base + self.j_own_u(s_both).view(2 * M, self.d_p, self.rank)
        Vo = self.j_own_v_base + self.j_own_v(s_both).view(2 * M, self.d_p, self.rank)
        j_own = (
            torch.matmul(Uo, torch.matmul(Vo.transpose(-1, -2), g_both))
            - torch.matmul(Vo, torch.matmul(Uo.transpose(-1, -2), g_both))
        ).squeeze(-1)
        j1_own, j2_own = j_own.chunk(2, dim=0)

        Uc = self.j_cross_u_base + self.j_cross_u(s_both).view(2 * M, self.d_p, self.rank)
        Vc = self.j_cross_v_base + self.j_cross_v(s_both).view(2 * M, self.d_p, self.rank)
        Uc_swap = torch.cat([Uc[M:], Uc[:M]], dim=0)
        Vc_swap = torch.cat([Vc[M:], Vc[:M]], dim=0)
        j_cross = (
            torch.matmul(Uc, torch.matmul(Vc.transpose(-1, -2), g_swap))
            - torch.matmul(Vc_swap, torch.matmul(Uc_swap.transpose(-1, -2), g_swap))
        ).squeeze(-1)
        j1_cross, j2_cross = j_cross.chunk(2, dim=0)

        L = self.r_own_l_base + self.r_own_l(s_both).view(2 * M, self.d_p, self.rank)
        lt_g = torch.matmul(L.transpose(-1, -2), g_both)
        r_both = torch.matmul(L, lt_g).squeeze(-1)
        diss_both = lt_g.squeeze(-1).square().sum(-1)
        r1, r2 = r_both.chunk(2, dim=0)
        diss1, diss2 = diss_both.chunk(2, dim=0)

        B_dyn = self.b_state(s_both).view(2 * M, self.d_p, u_dim)
        bu_both = (
            self.b_base(u_both) + torch.matmul(B_dyn, u_both.unsqueeze(-1)).squeeze(-1)
        ) * self.b_mask
        bu1, bu2 = bu_both.chunk(2, dim=0)

        return j1_own + j1_cross, j2_own + j2_cross, r1, r2, diss1, diss2, bu1, bu2

    def step_dynamics(self, x1, x2, u1, u2):
        """One 5-second Port-Hamiltonian macro step on (x1, x2) [M, d_p]."""
        x1_req = x1 if x1.requires_grad else x1.detach().requires_grad_(True)
        x2_req = x2 if x2.requires_grad else x2.detach().requires_grad_(True)

        h_out = self.H_l2(x1_req, x2_req)
        cg = self.training
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

        j1, j2, r1, r2, diss1, diss2, bu1, bu2 = self._apply_operators(
            s1, s2, g1, g2, u1, u2
        )

        dx1 = (j1 - r1) + bu1
        dx2 = (j2 - r2) + bu2

        return (
            x1_req + dx1,
            x2_req + dx2,
            dict(
                dx1=dx1,
                dx2=dx2,
                bu1=bu1,
                bu2=bu2,
                g1_goal=g1_goal,
                g2_goal=g2_goal,
                diss1=diss1,
                diss2=diss2,
                **h_out,
            ),
        )

    def _goal_loss(self, g_goal, target_dir, active):
        """Stored-energy injection alignment: in Layer 2 (where irreversible combat
        attrition -R grad H_p < 0 drains stored army energy H_p > 0 toward 0), stronger
        positions have higher stored energy H_p^(2) > 0 (+A_1^(2) = H_1^(2) - H_2^(2) > 0),
        so active player commands u_p inject energy along +g_goal = +grad_{X_p} A_p^(2)."""
        t_adv = (target_dir.detach()) * self.w_adv
        gn = g_goal.norm(dim=-1)
        tn = t_adv.norm(dim=-1)
        valid = active & (gn > 1e-8) & (tn > 1e-8)
        cos = (g_goal * t_adv).sum(dim=-1) / (gn * tn).clamp(min=1e-8)
        per_frame = F.softplus(-2.0 * cos)
        vf = valid.float()
        denom = vf.sum().clamp(min=1.0)
        return (per_frame * vf).sum() / denom, (cos * vf).sum().detach() / denom

    def _td_energy_loss(self, x, batch, gamma=0.95):
        """Multi-horizon telescopic energy bootstrapping across m in (2, 4, 8, 16, 24)
        steps (10s, 20s, 40s, 80s, 120s) in shared native 1e-3 Joule units.
        Oriented Layer 1 stored energy is E_p^(1) = -H_p^(1) (with A_{1,oriented}^(1) = -A_1^(1)),
        and Layer 2 stored energy is E_p^(2) = +H_p^(2) (with A_1^(2) = H_1^(2) - H_2^(2)):
            E_p^(1)(tau) + E_p^(2)(X_tau) = E_p^(1)(tau + m) + gamma^m * sg[E_p^(2)(X_{tau+m})]
        Both H^(1) and H^(2) remain at their natural O(1e-3) Port-Hamiltonian scale so
        (J - R) grad H^(2) matches phase-space velocities Delta X, while residuals are
        evaluated in normalized O(1) units (/ E_SCALE_L1).
        """
        B, T, _, _ = x.shape
        x_flat = x.reshape(B * T, 2, self.d_p)
        hout = self.H_l2(x_flat[:, 0], x_flat[:, 1])
        h1_l2 = hout["H1"].view(B, T) / E_SCALE_L1
        h2_l2 = hout["H2"].view(B, T) / E_SCALE_L1
        a1_l2 = hout["A1"].view(B, T) / E_SCALE_L1

        # Oriented Layer 1 stored energy (-H1_l1, -H2_l1, -A1_l1 > 0 for P1 advantage)
        e1_l1 = -batch["H1_l1"].float() / E_SCALE_L1
        e2_l1 = -batch["H2_l1"].float() / E_SCALE_L1
        a1_l1 = -batch["A1_l1"].float() / E_SCALE_L1

        horizons = [m for m in (2, 4, 8, 16, 24) if m < T]
        if not horizons:
            return h1_l2.sum() * 0.0

        td_loss = h1_l2.sum() * 0.0
        for m in horizons:
            disc = gamma**m
            tgt_h1 = (e1_l1[:, m:] - e1_l1[:, :-m]) + disc * h1_l2[:, m:].detach()
            tgt_h2 = (e2_l1[:, m:] - e2_l1[:, :-m]) + disc * h2_l2[:, m:].detach()
            tgt_a1 = (a1_l1[:, m:] - a1_l1[:, :-m]) + disc * a1_l2[:, m:].detach()

            l_m = (
                0.5 * F.smooth_l1_loss(h1_l2[:, :-m], tgt_h1)
                + 0.5 * F.smooth_l1_loss(h2_l2[:, :-m], tgt_h2)
                + F.smooth_l1_loss(a1_l2[:, :-m], tgt_a1)
            )
            td_loss = td_loss + l_m / len(horizons)

        return td_loss

    def _vicreg_loss(self, z_macro):
        """Prevents dimensional/constant collapse on z_macro [B, T, 2, r_macro]."""
        z_flat = z_macro.reshape(-1, self.r_macro)
        if z_flat.shape[0] < 4:
            return z_flat.sum() * 0.0
        std = torch.sqrt(z_flat.var(dim=0) + 1e-4)
        var_loss = F.relu(0.25 - std).mean()
        z_cent = z_flat - z_flat.mean(dim=0, keepdim=True)
        cov = (z_cent.t() @ z_cent) / max(1, z_flat.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = off_diag.square().mean()
        return var_loss + cov_loss

    def forward(self, batch):
        """Evaluates multi-horizon (1..K steps = 5s..20s) Port-Hamiltonian rollouts,
        goal alignment, and additive energy logs on a [B, T, ...] batch of 5s summaries."""
        x_obs, x_lat, u, u_active, z_macro = self.seq_enc(batch)
        x = torch.cat([x_obs, x_lat], dim=-1)  # [B, T, 2, d_p]
        B, T, _, _ = x.shape

        K = min(self.rollout_steps, T - 1)
        # 1-step (5s) transition across all tau = 0 .. T - 2
        M1 = B * (T - 1)
        x_t = x[:, :-1].reshape(M1, 2, self.d_p)
        x_n1 = x[:, 1:].reshape(M1, 2, self.d_p)
        u_t = u[:, :-1].reshape(M1, 2, -1)
        act_t = u_active[:, :-1].reshape(M1, 2)

        x1_p1, x2_p1, pieces = self.step_dynamics(
            x_t[:, 0], x_t[:, 1], u_t[:, 0], u_t[:, 1]
        )
        x_pred1 = torch.stack([x1_p1, x2_p1], dim=1)

        diff_obs1 = (x_pred1[:, :, :N_OBS_L2] - x_n1[:, :, :N_OBS_L2]) * self.w_obs
        per_coord_obs = 0.5 * diff_obs1.square().mean(dim=(0, 1))
        loss_obs_1 = per_coord_obs.sum()

        diff_copy1 = (x_t[:, :, :N_OBS_L2] - x_n1[:, :, :N_OBS_L2]) * self.w_obs
        per_coord_copy = 0.5 * diff_copy1.square().mean(dim=(0, 1))

        loss_lat_1 = F.smooth_l1_loss(x_pred1[:, :, N_OBS_L2:], x_n1[:, :, N_OBS_L2:].detach())

        # Multi-step rollout k = 2 .. K (+10s, +15s, +20s) to force pre-combat & macro foresight
        loss_obs_multi = loss_obs_1 * 0.0
        loss_lat_multi = loss_lat_1 * 0.0
        if K >= 2 and T > K:
            MK = B * (T - K)
            xk_1 = x[:, : T - K, 0].reshape(MK, self.d_p)
            xk_2 = x[:, : T - K, 1].reshape(MK, self.d_p)
            for k in range(1, K + 1):
                uk = u[:, k - 1 : T - K + k - 1].reshape(MK, 2, -1)
                xk_1, xk_2, _ = self.step_dynamics(xk_1, xk_2, uk[:, 0], uk[:, 1])
                if k >= 2:
                    xk_pred = torch.stack([xk_1, xk_2], dim=1)
                    xk_true = x[:, k : T - K + k].reshape(MK, 2, self.d_p)
                    w_k = 0.5 ** (k - 1)
                    d_obs_k = (xk_pred[:, :, :N_OBS_L2] - xk_true[:, :, :N_OBS_L2]) * self.w_obs
                    loss_obs_multi = loss_obs_multi + w_k * 0.5 * d_obs_k.square().mean(dim=(0, 1)).sum()
                    loss_lat_multi = loss_lat_multi + w_k * F.smooth_l1_loss(
                        xk_pred[:, :, N_OBS_L2:], xk_true[:, :, N_OBS_L2:].detach()
                    )

        loss_obs = loss_obs_1 + 0.5 * loss_obs_multi
        loss_lat = loss_lat_1 + 0.5 * loss_lat_multi

        # Goal alignment on 5s control port push
        l_g1, cos1 = self._goal_loss(pieces["g1_goal"], pieces["bu1"], act_t[:, 0])
        l_g2, cos2 = self._goal_loss(pieces["g2_goal"], pieces["bu2"], act_t[:, 1])
        loss_goal = 0.5 * (l_g1 + l_g2)
        goal_cos = 0.5 * (cos1 + cos2)

        loss_td = self._td_energy_loss(x, batch)
        loss_reg = self._vicreg_loss(z_macro)

        loss = (
            loss_obs
            + self.w_lat * loss_lat
            + self.w_goal * loss_goal
            + self.w_td * loss_td
            + self.w_reg * loss_reg
        )

        # Additive EBM Stored Energies in shared 1e-3 Joule units:
        # E_{p,total} = -H_p^(1) + H_p^(2), A_{1,total} = -A_1^(1) + A_1^(2)
        e1_l1 = -batch["H1_l1"][:, :-1].reshape(M1).float()
        e2_l1 = -batch["H2_l1"][:, :-1].reshape(M1).float()
        a1_l1 = -batch["A1_l1"][:, :-1].reshape(M1).float()
        h1_tot = e1_l1 + pieces["H1"].detach()
        h2_tot = e2_l1 + pieces["H2"].detach()
        a1_tot = a1_l1 + pieces["A1"].detach()

        logs = {
            "loss_obs": loss_obs_1.detach(),
            "loss_obs_multi": loss_obs_multi.detach(),
            "loss_lat": loss_lat.detach(),
            "loss_goal": loss_goal.detach(),
            "loss_td": loss_td.detach(),
            "loss_reg": loss_reg.detach(),
            "goal_cos": goal_cos,
            "copy_loss_obs": per_coord_copy.sum().detach(),
            "H1_l2": pieces["H1"].mean().detach(),
            "H2_l2": pieces["H2"].mean().detach(),
            "A1_l2": pieces["A1"].mean().detach(),
            "A1_total": a1_tot.mean().detach(),
            "z_macro_std": z_macro.std().detach(),
        }
        for col in E.Q_COLS:
            idx = self.first_idx[col]
            d_idx = self.diag_idx[col]
            val = 0.5 * (per_coord_obs[idx] + per_coord_obs[d_idx]).detach()
            cval = 0.5 * (per_coord_copy[idx] + per_coord_copy[d_idx]).detach()
            logs[f"res_{col}"] = val
            logs[f"copy_res_{col}"] = cval

        return loss, logs
