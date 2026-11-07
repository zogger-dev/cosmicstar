import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedSpatioTemporalBlock(nn.Module):
    """
    Tensors:
      x_entities: [B, T, N, D]       (B: Batch, T: Time Horizon, N: Entities, D: Latent Dim)
      x_actions:  [B, T, M, D]       (M: Active player commands/order queue tokens)
      map_tokens: [B, T, K_map, D]   (K_map: Sampled coordinate points on continuous field)
    """
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        # 1. Intra-Order Attention: Orders attend to each other bidirectionally
        self.order_self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_orders = nn.LayerNorm(d_model)

        # 2. Entity-to-Action Cross Attention: Entities query their incoming commands
        self.entity_action_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ea = nn.LayerNorm(d_model)

        # 3. Spatial Entity Self-Attention: Entities interact with each other (crowding, targeting)
        self.entity_spatial_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_es = nn.LayerNorm(d_model)

        # 4. Map Grounding Cross-Attention: Entities query the continuous terrain/threat field
        self.entity_map_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_em = nn.LayerNorm(d_model)

        # 5. Temporal Attention: Tracks each entity's trajectory across the time horizon T
        self.entity_temporal_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_et = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x_entities, x_actions, map_tokens, entity_mask=None, action_mask=None):
        B, T, N, D = x_entities.shape
        _, _, M, _ = x_actions.shape
        _, _, K_map, _ = map_tokens.shape

        # Flatten B and T to process each time-slice spatially: [B*T, ..., D]
        e_s = x_entities.view(B * T, N, D)
        a_s = x_actions.view(B * T, M, D)
        m_s = map_tokens.view(B * T, K_map, D)

        # --- Step 1: Orders attend bidirectionally to other orders ---
        a_ctx, _ = self.order_self_attn(query=a_s, key=a_s, value=a_s, key_padding_mask=action_mask)
        a_s = self.norm_orders(a_s + a_ctx)

        # --- Step 2: Entities attend to contextualized Orders ---
        ea_ctx, _ = self.entity_action_attn(query=e_s, key=a_s, value=a_s, key_padding_mask=action_mask)
        e_s = self.norm_ea(e_s + ea_ctx)

        # --- Step 3: Entities attend to each other in 2D space ---
        es_ctx, _ = self.entity_spatial_attn(query=e_s, key=e_s, value=e_s, key_padding_mask=entity_mask)
        e_s = self.norm_es(e_s + es_ctx)

        # --- Step 4: Entities attend to Continuous Map Field (Map-to-Entity Grounding) ---
        em_ctx, _ = self.entity_map_attn(query=e_s, key=m_s, value=m_s)
        e_s = self.norm_em(e_s + em_ctx)

        # Reshape back to sequence form: [B, T, N, D]
        x_entities = e_s.view(B, T, N, D)

        # --- Step 5: Temporal Attention across time horizon T per entity ---
        # Permute to [B*N, T, D] to run 1D causal/windowed attention along the timeline
        e_t = x_entities.permute(0, 2, 1, 3).reshape(B * N, T, D)
        et_ctx, _ = self.entity_temporal_attn(query=e_t, key=e_t, value=e_t)
        e_t = self.norm_et(e_t + et_ctx)
        
        # Restore shape: [B, T, N, D]
        x_entities = e_t.view(B, N, T, D).permute(0, 2, 1, 3)

        # Final MLP projection
        out = self.norm_out(x_entities + self.mlp(x_entities))
        return out, a_s.view(B, T, M, D)


class FactorizedSpatioTemporalBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        # 1. Spatial Attention across entities at time t
        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_s = nn.LayerNorm(d_model)
        
        # 2. Cross-Attention to continuous map field
        self.map_cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_m = nn.LayerNorm(d_model)
        
        # 3. Temporal Attention across time steps T for entity i
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_t = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x, map_tokens):
        """
        x: [B, T, N, D]
        map_tokens: [B, T, K_map, D] (Spatial field samples)
        """
        B, T, N, D = x.shape
        
        # --- Stage 1: Spatial Self-Attention (over N entities) ---
        x_s = x.view(B * T, N, D)
        s_out, _ = self.spatial_attn(x_s, x_s, x_s)
        x_s = self.norm_s(x_s + s_out)
        
        # --- Stage 2: Spatial Map Cross-Attention ---
        m_tokens = map_tokens.view(B * T, -1, D)
        m_out, _ = self.map_cross_attn(query=x_s, key=m_tokens, value=m_tokens)
        x_s = self.norm_m(x_s + m_out)
        x = x_s.view(B, T, N, D)
        
        # --- Stage 3: Temporal Self-Attention (over T time steps) ---
        x_t = x.permute(0, 2, 1, d).reshape(B * N, T, D) # [B*N, T, D]
        t_out, _ = self.temporal_attn(x_t, x_t, x_t)
        x_t = self.norm_t(x_t + t_out)
        x = x_t.view(B, N, T, D).permute(0, 2, 1, 3) # Back to [B, T, N, D]
        
        # Feedforward
        out = self.norm_out(x + self.mlp(x))
        return out


class ContinuousMapNeuralField(nn.Module):
    """Continuous coordinate neural field V_θ(x | E) for terrain and threat."""
    def __init__(self, d_model=256, n_freqs=16):
        super().__init__()
        self.n_freqs = n_freqs
        in_dim = 2 + 4 * n_freqs # [x, y] + sin/cos embeddings
        self.field_mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, d_model)
        )
        self.energy_head = nn.Linear(d_model, 1)

    def fourier_embed(self, coords):
        # coords: [..., 2]
        freq_bands = torch.linspace(1.0, 32.0, self.n_freqs, device=coords.device)
        args = coords.unsqueeze(-1) * freq_bands * torch.pi
        sin_f = torch.sin(args).flatten(-2)
        cos_f = torch.cos(args).flatten(-2)
        return torch.cat([coords, sin_f, cos_f], dim=-1)

    def forward(self, sample_coords):
        # Returns continuous map tokens for cross-attention
        embedded = self.fourier_embed(sample_coords)
        features = self.field_mlp(embedded)
        potential = F.softplus(self.energy_head(features))
        return features, potential


class ThermodynamicEBM(nn.Module):
    def __init__(self, d_model=256, k_int=5):
        super().__init__()
        self.map_field = ContinuousMapNeuralField(d_model=d_model)
        self.block1 = UnifiedSpatioTemporalBlock(d_model=d_model)
        self.block2 = UnifiedSpatioTemporalBlock(d_model=d_model)
        
        # Generalized Mass Metrics: M_spatial, M_internal
        self.mass_head = nn.Linear(d_model, 2 + k_int)
        
        # Spring Potential: U_internal(q_int | semantics)
        self.spring_head = nn.Sequential(
            nn.Linear(d_model + k_int, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        # Power & Force Heads
        self.p_internal_head = nn.Linear(d_model, 1)
        self.player_power_head = nn.Linear(64, 1)
        self.control_force_head = nn.Linear(d_model + 64, 2 + k_int)

    def compute_energy_state(self, tokens, s_x, s_v, q_int, v_int, map_potential):
        """
        Computes total H = T_spatial + T_internal + U_internal + V_spatial
        """
        masses = F.softplus(self.mass_head(tokens)) + 1e-4
        m_x = masses[..., :2]
        m_int = masses[..., 2:]
        
        # Kinetic Energies
        T_spatial = 0.5 * torch.sum(m_x * (s_v ** 2), dim=-1)
        T_internal = 0.5 * torch.sum(m_int * (v_int ** 2), dim=-1)
        
        # Internal Spring Enthalpy
        q_features = torch.cat([tokens, q_int], dim=-1)
        U_internal = F.softplus(self.spring_head(q_features)).squeeze(-1)
        
        H_entities = torch.sum(T_spatial + T_internal + U_internal, dim=-1)
        H_map = torch.sum(map_potential, dim=-1) # Continuous field integral
        
        return H_entities + H_map, (T_spatial + T_internal + U_internal)

    def forward_loss(self, traj, actions, sample_map_coords, casualties_mask, dt):
        """
        traj: Dict containing sequences across T=2:
          'tokens': [B, T, N, D], 'x': [B, T, N, 2], 'v': [B, T, N, 2],
          'q_int': [B, T, N, K], 'v_int': [B, T, N, K]
        casualties_mask: [B, N] (1 if entity died between t and t+dt)
        """
        B, T, N, D = traj['tokens'].shape
        
        # Evaluate map field
        map_toks, map_pot = self.map_field(sample_map_coords) # [B, T, K_pts, D]
        
        # Pass through Spatio-Temporal Factorized Blocks
        h = self.block1(traj['tokens'], map_toks)
        h = self.block2(h, map_toks)
        
        # Compute H at t=0 and t=1
        H_t, e_energies_t = self.compute_energy_state(
            h[:, 0], traj['x'][:, 0], traj['v'][:, 0],
            traj['q_int'][:, 0], traj['v_int'][:, 0], map_pot[:, 0]
        )
        H_next, _ = self.compute_energy_state(
            h[:, 1], traj['x'][:, 1], traj['v'][:, 1],
            traj['q_int'][:, 1], traj['v_int'][:, 1], map_pot[:, 1]
        )
        
        # 1. Internal Metabolic Work Rate
        P_int = torch.sum(self.p_internal_head(h[:, 0]).squeeze(-1), dim=-1)
        
        # 2. External Player Injected Power
        P_ext = F.softplus(self.player_power_head(actions['dense'])).squeeze(-1)
        
        # 3. True Physical Dissipative Sinks:
        # A) Casualty Dissipation (Dead units' total energy destroyed)
        D_internal = torch.sum(e_energies_t * casualties_mask, dim=-1) / dt
        
        # B) Control Work Dissipation (Inefficient APM)
        q_all_dot = torch.cat([traj['v'][:, 0], traj['v_int'][:, 0]], dim=-1)
        act_conditioned = torch.cat([h[:, 0], actions['dense'].unsqueeze(1).expand(-1, N, -1)], dim=-1)
        F_control = self.control_force_head(act_conditioned)
        W_effective = torch.sum(F_control * q_all_dot, dim=[-1, -2])
        D_control = F.relu(P_ext - W_effective) # Non-negative wasted power
        
        Total_Dissipation = D_internal + D_control
        
        # First Law: dH/dt = P_ext + P_int - Total_Dissipation
        dH_dt = (H_next - H_t) / dt
        loss_first_law = F.mse_loss(dH_dt, (P_ext + P_int - Total_Dissipation))
        
        return loss_first_law