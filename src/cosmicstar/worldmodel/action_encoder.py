"""Action encoder: a window's action tokens -> one summary per (segment, player).

Token input (one per action, see action_tokens.py): its parts are concatenated and mixed
by an MLP into d:

    kind, ability family, button, target kind, control group, group op, removal kind,
    actor (team + player code), recipient (trades), point (map position),
    target unit (pointer: the state encoder's embedding of that unit's state),
    snapshot type (fogged targets), selected units (mean of their state embeddings),
    numbers (presses, counts, amounts, flags), time relative to the sample's present

Owner codes, positions and unit embeddings are the state encoder's own tables, so a player
code, a map position or a unit means the same thing in actions and in states.

A causal transformer runs over the window's tokens in time order (a retarget sees the
command it retargets; a selection is seen by the commands after it). Then each (segment,
player) gets one summary by attention pooling over that player's tokens in that segment; a
learned "nothing" key is always there, so a player who did nothing still gets a summary
(doing nothing is information too). The summaries are the abstract actions: what each player
did in each segment. The world model predicts given them; a policy would later act in this
space, with a separate decoder down to raw interface events.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .action_tokens import C_ACTION, N_ACT_KINDS, N_GROUP_OPS, N_GROUPS, N_REMOVE
from .encoder import mlp


class RelTime(nn.Module):
    """Seconds relative to the present (past < 0 < future) -> sin/cos from 0.25 s to ~1 min,
    plus the signed value: action timing matters at the scale of a fraction of a second."""

    def __init__(self, n_periods=8, shortest=0.25, longest=64.0):
        super().__init__()
        periods = torch.logspace(math.log10(shortest), math.log10(longest), n_periods)
        self.register_buffer("omega", 2 * math.pi / periods, persistent=False)

    def out_dim(self):
        return 2 * len(self.omega) + 1

    def forward(self, t):
        a = t[..., None] * self.omega
        return torch.cat([a.sin(), a.cos(), (t / 30.0)[..., None]], -1)


class ActionEncoder(nn.Module):
    N_PARTS = 15

    def __init__(self, state_encoder, avocab, d=256, depth=2, heads=8, mlp_ratio=4.0):
        super().__init__()
        self._se = [state_encoder]  # shared tables; a list so its parameters aren't registered twice
        self.heads = heads
        self.kind = nn.Embedding(N_ACT_KINDS, d)
        self.family = nn.Embedding(avocab.n_families, d, padding_idx=0)
        self.button = nn.Embedding(avocab.n_buttons, d, padding_idx=0)
        self.tkind = nn.Embedding(4, d)
        self.group = nn.Embedding(N_GROUPS, d, padding_idx=0)
        self.gop = nn.Embedding(N_GROUP_OPS, d, padding_idx=0)
        self.remove = nn.Embedding(N_REMOVE, d, padding_idx=0)
        self.num = mlp(C_ACTION, d)
        self.time = RelTime()
        self.time_in = mlp(self.time.out_dim(), d)
        self.fuse = mlp(self.N_PARTS * d, d, 2 * d)
        self.start = nn.Parameter(torch.zeros(1, 1, d))  # always-present first token
        layer = nn.TransformerEncoderLayer(d, heads, int(d * mlp_ratio), 0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        # summaries
        self.sum_query = nn.Parameter(torch.zeros(1, 1, 1, d))
        self.sum_time = mlp(self.time.out_dim(), d)
        self.hor_time = mlp(self.time.out_dim(), d)  # how long the summarized interval is (variable horizons)
        self.nothing = nn.Parameter(torch.zeros(1, 1, d))
        self.sq, self.skv, self.so = nn.LayerNorm(d), nn.LayerNorm(d), nn.Linear(d, d)
        self.q, self.kv = nn.Linear(d, d), nn.Linear(d, 2 * d)
        self.sf, self.sff = nn.LayerNorm(d), mlp(d, d, int(d * mlp_ratio))
        self.out_norm = nn.LayerNorm(d)
        for p in (self.start, self.sum_query, self.nothing):
            nn.init.normal_(p, std=0.02)

    @property
    def se(self):
        return self._se[0]

    def tokens(self, b):
        """[B, N, d] input embedding of each action token."""
        se = self.se
        B, N = b["act_kind"].shape
        num = b["act_num"]
        has_xy, has_rec = num[..., :1], num[..., 9:10]
        tgt_ok = b["act_tgt_ok"][..., None].to(num.dtype)
        sel_emb = se.entity_tokens(b, "sel")
        sums = torch.zeros(B * N, sel_emb.shape[-1], device=num.device, dtype=sel_emb.dtype)
        sums.index_add_(0, b["sel_event"], sel_emb)
        cnt = torch.bincount(b["sel_event"], minlength=B * N).clamp(min=1).to(sums.dtype)[:, None]
        sel = (sums / cnt).view(B, N, -1)
        parts = [
            self.kind(b["act_kind"]), self.family(b["act_family"]), self.button(b["act_button"]),
            self.tkind(b["act_tkind"]), self.group(b["act_group"]), self.gop(b["act_gop"]),
            self.remove(b["act_remove"]),
            se.team(b["act_team"]) + se.pcode(b["act_pcode"]),
            (se.team(b["act_rteam"]) + se.pcode(b["act_rpcode"])) * has_rec,
            se.pos(b["act_xy"]) * has_xy,
            se.entity_tokens(b, "tgt") * tgt_ok,
            se.unit(b["act_snap"]),
            sel,
            self.num(num),
            self.time_in(self.time(b["act_t"])),
        ]
        return self.fuse(torch.cat(parts, -1))

    def forward(self, b, ahead=None, horizon_s=None):
        """Returns summaries [B, S, P, d], their mask [B, S, P] (real players) and the
        contextualized tokens [B, N, d]. horizon_s [B]: the summarized interval's length in
        seconds, when it varies between batches.

        ahead (in segments): instead of one summary per segment, one per (frame, player) over
        the `ahead` segments that follow the frame, i.e. the actions in the interval the
        predictor is asked about. The window's bounds then run `ahead` segments past its last
        frame (seg_t holds T + ahead bounds; queries are made for the first T)."""
        x = self.tokens(b)
        B, N, d = x.shape
        x = torch.cat([self.start.expand(B, -1, -1).to(x.dtype), x], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~b["act_mask"]], 1)
        causal = torch.triu(torch.ones(N + 1, N + 1, dtype=torch.bool, device=x.device), 1)
        h = self.norm(self.blocks(x, mask=causal, src_key_padding_mask=pad))[:, 1:]

        # one summary per (segment, player)
        se = self.se
        S, P = b["seg_t"].shape[1], b["sum_team"].shape[1]
        if ahead:
            S -= ahead
        q = (self.sum_query + self.sum_time(self.time(b["seg_t"][:, :S]))[:, :, None]
             + (se.team(b["sum_team"]) + se.pcode(b["sum_pcode"]))[:, None])  # [B, S, P, d]
        if horizon_s is not None:
            q = q + self.hor_time(self.time(horizon_s))[:, None, None]
        q = q.reshape(B, S * P, d)
        kv_src = torch.cat([self.nothing.expand(B, -1, -1).to(h.dtype), h], 1)
        seg = torch.arange(S, device=x.device).repeat_interleave(P)          # query -> segment
        ply = torch.arange(P, device=x.device).repeat(S)                     # query -> player
        a_seg = b["act_seg"][:, None]
        in_seg = (a_seg == seg[None, :, None]) if not ahead else \
            (a_seg >= seg[None, :, None]) & (a_seg < seg[None, :, None] + ahead)
        allowed = in_seg & (b["act_player"][:, None] == ply[None, :, None]) & b["act_mask"][:, None]  # [B, S*P, N]
        allowed = torch.cat([torch.ones_like(allowed[..., :1]), allowed], -1)
        H = self.heads
        qq = self.q(self.sq(q)).view(B, S * P, H, -1).transpose(1, 2)
        k, v = self.kv(self.skv(kv_src)).view(B, N + 1, 2, H, -1).unbind(2)
        a = F.scaled_dot_product_attention(qq, k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed[:, None])
        s = q + self.so(a.transpose(1, 2).reshape(B, S * P, d))
        s = s + self.sff(self.sf(s))
        summaries = self.out_norm(s).view(B, S, P, d)
        mask = b["sum_mask"][:, None].expand(B, S, P)
        return summaries, mask, h
