"""WindowBlock's FlexAttention (block mask + validity) against a plain masked softmax. Run directly: python tests/test_dense_window.py (CUDA)."""

import torch

from cosmicstar.worldmodel.dense import WIN, WindowBlock, _rot


def reference(blk, x, ctx):
    B, L, d = x.shape
    n0 = ctx["n0"]
    h = torch.cat([blk.ne(x[:, :n0]), blk.nc(x[:, n0:])], 1)
    q, k, v = blk.qkv(h).view(B, L, 3, blk.H, blk.dh).unbind(2)
    q, k = _rot(q, ctx["ang"]), _rot(k, ctx["ang"])
    q, k, v = (t.transpose(1, 2).float() for t in (q, k, v))  # [B, H, L, dh]
    sc = q @ k.transpose(-1, -2) / blk.dh**0.5
    i = torch.arange(L, device=x.device)
    win = (i - n0) // (WIN * WIN)
    allowed = (i[:, None] < n0) | (i[None, :] < n0) | (win[:, None] == win[None, :])
    allowed = allowed[None, None] & ctx["valid"][:, None, None, :]
    a = torch.softmax(sc.masked_fill(~allowed, float("-inf")), -1) @ v
    a = a.transpose(1, 2).reshape(B, L, d)
    x = x + blk.o(a.to(x.dtype))
    return torch.cat(
        [
            x[:, :n0] + blk.me(blk.men(x[:, :n0])),
            x[:, n0:] + blk.mc(blk.mcn(x[:, n0:])),
        ],
        1,
    )


def main():
    torch.manual_seed(0)
    dev = "cuda"
    d, H, B, g, ne, nc, s = 128, 4, 2, 3, 37, 8 * 16, 2.0
    blk = WindowBlock(d, H).to(dev)
    from cosmicstar.worldmodel.dense import ROPE_HI, ROPE_LO
    from cosmicstar.worldmodel.direct import RoPE2D
    from torch.nn.attention.flex_attention import create_block_mask

    L, n0 = g + ne + nc, g + ne
    x = torch.randn(B, L, d, device=dev)
    pos = torch.randn(B, L, 2, device=dev) * 6
    valid = torch.rand(B, L, device=dev) > 0.15
    valid[:, 0] = True
    mask_mod = (
        lambda b, h, qi, ki: (qi < n0)
        | (ki < n0)
        | ((qi - n0) // (WIN * WIN) == (ki - n0) // (WIN * WIN))
    )
    ctx = dict(
        n0=n0,
        ang=RoPE2D(d // H, ROPE_LO, ROPE_HI).to(dev).angles(pos),
        valid=valid,
        block_mask=create_block_mask(mask_mod, None, None, L, L, device=dev),
    )
    out = blk(x, ctx)
    ref = reference(blk, x, ctx)
    err = (out - ref).abs().max().item()
    out.sum().backward()
    gq = blk.qkv.weight.grad.clone()
    blk.zero_grad()
    ref.sum().backward()
    gerr = (
        (gq - blk.qkv.weight.grad).abs().max() / blk.qkv.weight.grad.abs().max()
    ).item()
    print(f"max abs error {err:.2e}; qkv grad max rel error {gerr:.2e}")
    assert err < 1e-3 and gerr < 1e-2
    print("ok")


if __name__ == "__main__":
    main()
