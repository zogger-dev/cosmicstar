"""Grouped-query attention projections, pointer-biased SDPA, and FlexAttention kernels."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

from .rope import rotate


def attend_pairs(q, k, v, mask, n_u: int = 0, edges=None, bias=None):
    """Masked SDPA with optional sparse pointer-edge additive score biases."""
    gqa = q.shape[1] != k.shape[1]
    if edges is None:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[:, None, None], enable_gqa=gqa
        )
    q_b, k_b, v_b, m_b = q[n_u:], k[n_u:], v[n_u:], mask[n_u:]
    pi, qi, ki, ci = edges
    am = torch.zeros(
        q_b.shape[0], q.shape[2], k.shape[2], q.shape[1], dtype=q.dtype, device=q.device
    )
    am = am.index_put((pi, qi, ki), bias.to(q.dtype)[ci])
    am = am.permute(0, 3, 1, 2).masked_fill(~m_b[:, None, None], -float("inf"))
    r_b = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=am, enable_gqa=gqa)
    if n_u == 0:
        return r_b
    r_u = F.scaled_dot_product_attention(
        q[:n_u], k[:n_u], v[:n_u], attn_mask=mask[:n_u, None, None], enable_gqa=gqa
    )
    return torch.cat([r_u, r_b], dim=0)


def _flex_spatial_raw(q, k, v, mask, eng_tgt, par_tgt, ord_tgt, bd):
    """FlexAttention with detached `[6, H]` unit-pointer edge biases -> `(out, lse)`."""
    b0, b1, b2, b3, b4, b5 = bd.unbind(0)

    def score_mod(score, b, hh, qi, ki):
        s = torch.where(mask[b, ki], score, -float("inf"))
        add = (
            torch.where(eng_tgt[b, qi] == ki, b0[hh], 0.0)
            + torch.where(eng_tgt[b, ki] == qi, b1[hh], 0.0)
            + torch.where(par_tgt[b, qi] == ki, b2[hh], 0.0)
            + torch.where(par_tgt[b, ki] == qi, b3[hh], 0.0)
            + torch.where(ord_tgt[b, qi] == ki, b4[hh], 0.0)
            + torch.where(ord_tgt[b, ki] == qi, b5[hh], 0.0)
        )
        return s + add.to(s.dtype)

    return flex_attention(
        q,
        k,
        v,
        score_mod=score_mod,
        return_lse=True,
        enable_gqa=(q.shape[1] != k.shape[1]),
    )


class _EdgeBiasGrad(torch.autograd.Function):
    """Exact analytical gradient `dL/dbias` on `O(R)` pointer edges from LSE."""

    @staticmethod
    def forward(ctx, out, lse, q, k, v, mask, eng, par, ord0, bias):
        ctx.save_for_backward(
            out.detach(), lse, q, k, v, mask, eng, par, ord0, bias.detach()
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        out, lse, q, k, v, mask, eng, par, ord0, bd = ctx.saved_tensors
        D = (grad_out.float() * out.float()).sum(-1)
        scale = 1.0 / math.sqrt(q.shape[-1])
        rep = q.shape[1] // k.shape[1]
        gb = torch.zeros_like(bd, dtype=torch.float32)
        ptrs = ((0, eng), (2, par), (4, ord0))
        for k_idx, ptr in ptrs:
            bi0, idx_a0 = torch.nonzero(ptr >= 0, as_tuple=True)
            if not len(bi0):
                continue
            idx_b0 = ptr[bi0, idx_a0].long()
            for rev, slot in ((False, k_idx), (True, k_idx + 1)):
                qi, ki = (idx_b0, idx_a0) if rev else (idx_a0, idx_b0)
                ok = mask[bi0, ki]
                bi, q_i, k_i = bi0[ok], qi[ok], ki[ok]
                if not len(bi):
                    continue
                ks = k[bi, :, k_i].float()
                vs = v[bi, :, k_i].float()
                if rep > 1:
                    ks = ks.repeat_interleave(rep, 1)
                    vs = vs.repeat_interleave(rep, 1)
                s = (q[bi, :, q_i].float() * ks).sum(-1) * scale
                for kk, pp in ptrs:
                    s = s + (pp[bi, q_i] == k_i)[:, None].float() * bd[kk].float()[None]
                    s = (
                        s
                        + (pp[bi, k_i] == q_i)[:, None].float()
                        * bd[kk + 1].float()[None]
                    )
                p = torch.exp(s - lse[bi, :, q_i].float())
                dov = (grad_out[bi, :, q_i].float() * vs).sum(-1)
                gb[slot] = (p * (dov - D[bi, :, q_i])).sum(0)
        return (
            grad_out,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            gb.to(bd.dtype),
        )


class FlexSpatial(nn.Module):
    """Compiled inner FlexAttention wrapper (keeps `_EdgeBiasGrad` outside Dynamo)."""

    def __init__(self):
        super().__init__()
        self.is_compiled = False

    def forward(self, q, k, v, mask, eng_tgt, par_tgt, ord_tgt, bd):
        return _flex_spatial_raw(q, k, v, mask, eng_tgt, par_tgt, ord_tgt, bd)


def attend_spatial(q, k, v, mask, eng_tgt, par_tgt, ord_tgt, bias, flex_mod=None):
    """Spatial self-attention with directed `[6, H]` unit-pointer edge biases."""
    compiled_flex = flex_mod is not None and getattr(flex_mod, "is_compiled", False)
    if not q.is_cuda or (not compiled_flex and not torch.is_grad_enabled()):
        B, H, T, _ = q.shape
        am = torch.zeros(B, H, T, T, dtype=q.dtype, device=q.device)
        sb = bias.to(q.dtype)
        for k_idx, ptr in ((0, eng_tgt), (2, par_tgt), (4, ord_tgt)):
            has = ptr >= 0
            idx_f = ptr.clamp(min=0).long()[:, None, :, None].expand(B, H, T, 1)
            w_f = has[:, None, :, None].to(q.dtype) * sb[k_idx][None, :, None, None]
            am = am.scatter_add(3, idx_f, w_f)
            idx_r = ptr.clamp(min=0).long()[:, None, None, :].expand(B, H, 1, T)
            w_r = has[:, None, None, :].to(q.dtype) * sb[k_idx + 1][None, :, None, None]
            am = am.scatter_add(2, idx_r, w_r)
        am = am.masked_fill(~mask[:, None, None], -float("inf"))
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=am, enable_gqa=(q.shape[1] != k.shape[1])
        )

    fn = flex_mod if flex_mod is not None else _flex_spatial_raw
    out, lse = fn(q, k, v, mask, eng_tgt, par_tgt, ord_tgt, bias.detach())
    if torch.is_grad_enabled() and bias.requires_grad:
        out = _EdgeBiasGrad.apply(
            out,
            lse.detach(),
            q.detach(),
            k.detach(),
            v.detach(),
            mask,
            eng_tgt,
            par_tgt,
            ord_tgt,
            bias,
        )
    return out


class Attend(nn.Module):
    """Spatial softmax over allowed keys with unit-pointer edge biases."""

    def __init__(self):
        super().__init__()
        self.flex = FlexSpatial()

    def forward(self, q, k, v, mask, eng_tgt, par_tgt, ord_tgt, bias):
        return attend_spatial(
            q, k, v, mask, eng_tgt, par_tgt, ord_tgt, bias, flex_mod=self.flex
        )


class PairAttend(nn.Module):
    """Masked attention with action-pointer pair biases."""

    def forward(self, q, k, v, mask, n_u, edges, bias):
        return attend_pairs(q, k, v, mask, n_u=n_u, edges=edges, bias=bias)


class Heads(nn.Module):
    """Grouped-query attention projections with normalized Q/K and learned logit scale.

    If `dh` is omitted, defaults cleanly to `d // heads`.
    """

    def __init__(self, d: int, heads: int, kv_heads: int, dh: int | None = None):
        super().__init__()
        dh = dh if dh is not None else (d // heads)
        self.H, self.KV, self.dh = heads, kv_heads, dh
        self.q = nn.Linear(d, heads * dh, bias=False)
        self.k = nn.Linear(d, kv_heads * dh, bias=False)
        self.v = nn.Linear(d, kv_heads * dh, bias=False)
        self.o = nn.Linear(heads * dh, d, bias=False)
        self.scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def _kv(self, x, layer, cs=None):
        B, T = x.shape[:2]
        z = layer(x).view(B, T, self.KV, self.dh).transpose(1, 2)
        return rotate(z, cs) if cs is not None else z

    def qkv(self, x, cs=None):
        B, T = x.shape[:2]
        w = torch.cat([self.q.weight, self.k.weight, self.v.weight], 0)
        z = F.linear(x, w).view(B, T, self.H + 2 * self.KV, self.dh).transpose(1, 2)
        if cs is not None:
            z = rotate(z, cs)
        q, k, v = z.split([self.H, self.KV, self.KV], 1)
        q = F.normalize(q, dim=-1) * self.scale.exp().to(q.dtype) * math.sqrt(self.dh)
        return q, F.normalize(k, dim=-1), v

    def kv(self, x, cs=None):
        B, T = x.shape[:2]
        w = torch.cat([self.k.weight, self.v.weight], 0)
        z = F.linear(x, w).view(B, T, 2 * self.KV, self.dh).transpose(1, 2)
        if cs is not None:
            z = rotate(z, cs)
        k, v = z.split([self.KV, self.KV], 1)
        return F.normalize(k, dim=-1), v

    def query(self, x, cs=None):
        B, T = x.shape[:2]
        q = self.q(x).view(B, T, self.H, self.dh).transpose(1, 2)
        q = rotate(q, cs) if cs is not None else q
        return (
            F.normalize(q, dim=-1) * self.scale.exp().to(q.dtype) * math.sqrt(self.dh)
        )

    def key(self, x, cs=None):
        return F.normalize(self._kv(x, self.k, cs), dim=-1)

    def value(self, x, cs=None):
        return self._kv(x, self.v, cs)

    def out(self, a, cs=None):
        a = rotate(a, cs, back=True) if cs is not None else a
        return self.o(a.transpose(1, 2).flatten(2))
