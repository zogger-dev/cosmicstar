"""A frame's encoding must not depend on its batch-mates: encode a frame from a small
map alone, then batched with a frame from a much larger map (more canvas padding, more
entity padding), and compare its outputs. Run directly: python
tests/test_dense_batch_independence.py (CUDA).
"""

import json

import torch

from cosmicstar.worldmodel import vocab as vocab_mod
from cosmicstar.worldmodel.action_tokens import ActionVocab
from cosmicstar.worldmodel.data import to
from cosmicstar.worldmodel.dense import DenseEncoder
from cosmicstar.worldmodel.tokenize import GameFrames, collate

# Taito Citadel, 138 x 138 playable
SMALL = "data/games/1v1/95841/ea9d28118a7d7253/051a0621324c4123.p1.cst"
# Mothership, 166 x 166 playable
LARGE = "data/games/1v1/95841/e854e11f57f13367/014daa87317de353.p1.cst"
VOCAB = "data/worldmodel/runs/direct_20261031_132847/vocab.json"


def main():
    torch.manual_seed(0)
    torch.backends.cudnn.allow_tf32 = (
        False  # exact comparison: no shape-dependent TF32 rounding
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    v = vocab_mod.Vocab(json.load(open(VOCAB)))
    a, b = GameFrames(SMALL, v), GameFrames(LARGE, v)
    fa = a.frame(2000, a.owner_map("player", me_slot=a.viewer))
    fb = b.frame(2500, b.owner_map("player", me_slot=b.viewer))
    enc = DenseEncoder(v, ActionVocab().n_families, 128, heads=4).cuda().eval()
    with torch.no_grad():
        alone = enc(to(collate([fa]), "cuda"))
        mixed = enc(to(collate([fa, fb]), "cuda"))
    n = int(alone["mask"][0].sum())
    ea, em = alone["tokens"][0][alone["mask"][0]], mixed["tokens"][0][mixed["mask"][0]]
    print(
        f"canvas alone {tuple(collate([fa])['grid'].shape[2:])}, mixed"
        f" {tuple(collate([fa, fb])['grid'].shape[2:])}; real tokens {n} /"
        f" {int(mixed['mask'][0].sum())}"
    )
    err = (ea - em).abs().max().item()
    print(f"max abs difference of the small map's real output tokens: {err:.2e}")
    assert ea.shape == em.shape and err < 1e-3, "the encoding depends on the batch"
    print("ok")


if __name__ == "__main__":
    main()
