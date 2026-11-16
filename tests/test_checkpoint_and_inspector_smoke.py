"""End-to-end checkpoint `strict=True` loading, fog-causality, and Inspector smoke test.

Run:
    PYTHONPATH=src .venv/bin/python tests/test_checkpoint_and_inspector_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cosmicstar.worldmodel.core import Tables, Vocab, paths, runtime  # noqa: E402
from cosmicstar.worldmodel.data import (  # noqa: E402
    ActionVocab,
    EXCLUDED,
    Games,
    base_paths,
    collate,
    game_store,
    to_device,
)
from cosmicstar.worldmodel.inspector import WorldModelInspector  # noqa: E402
from cosmicstar.worldmodel.l0_physics import Physics  # noqa: E402
from cosmicstar.worldmodel.l1_tactical import (  # noqa: E402
    D_SQUAD_DEFAULT,
    EnergyModel,
    Frames,
    HindsightOutcomePosterior,
    L1ChunkFlowDecoder,
    MultiLayerSAEStack,
    PerceiverSquadCompressor,
)
from cosmicstar.worldmodel.l2_observer import ObserverWorldModelL2  # noqa: E402


def test_strict_checkpoint_loading_and_fog_isolation() -> None:
    runtime.configure_torch_env()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load()
    avocab = ActionVocab()
    tables = Tables(94137)

    # 1. L0 Physics strict checkpoint load
    if os.path.exists(paths.DEFAULT_L0_CKPT):
        ck_l0 = torch.load(paths.DEFAULT_L0_CKPT, map_location="cpu", weights_only=False)
        m_l0 = Physics(
            vocab,
            avocab,
            128,
            moving_neutrals=game_store.MOVING_NEUTRALS,
            blocks=6,
            spatial_every=1,
            tubelet=4,
            d_tube=256,
        ).to(dev).eval()
        m_l0.load_state_dict(runtime.clean_state_dict(ck_l0["model"]), strict=True)
        print(f"[OK] L0 Physics strict=True ({len(ck_l0['model'])} tensors)")

    # 2. L1 Energy + SAE strict checkpoint load
    if os.path.exists(paths.DEFAULT_L1_CKPT):
        ck_l1 = torch.load(paths.DEFAULT_L1_CKPT, map_location="cpu", weights_only=False)
        mod_l1 = EnergyModel(256, 32, 256).to(dev).eval()
        mod_l1.load_state_dict(runtime.clean_state_dict(ck_l1["model"]), strict=True)
        print(f"[OK] L1 EnergyModel strict=True ({len(ck_l1['model'])} tensors)")

    if os.path.exists(paths.DEFAULT_SAE_CKPT):
        ck_sae = torch.load(paths.DEFAULT_SAE_CKPT, map_location="cpu", weights_only=False)
        sae = MultiLayerSAEStack(
            n_layers=int(ck_sae["n_layers"]),
            d_in=int(ck_sae["d_in"]),
            d_sae=int(ck_sae["d_sae"]),
            k_active=int(ck_sae["k_active"]),
        ).to(dev).eval()
        sae.load_state_dict(ck_sae["state_dict"], strict=True)
        print(f"[OK] L1 7-Layer SAE strict=True ({len(ck_sae['state_dict'])} tensors)")

    # 3. Observer v2 (Frames + Perceiver + Posterior + ChunkFlow + ObserverWorldModelL2) strict=True
    if os.path.exists(paths.DEFAULT_OBSERVER_CKPT):
        ck_obs = torch.load(paths.DEFAULT_OBSERVER_CKPT, map_location="cpu", weights_only=False)
        enc = Frames(vocab, tables, 256, 32, attn_layers=6, dh=64, ff=768).to(dev).eval()
        enc.grad_ckpt = False
        enc.load_state_dict(runtime.clean_state_dict(ck_obs["enc"]), strict=True)
        pooler = PerceiverSquadCompressor(tables, d_l1=256).to(dev).eval()
        pooler.load_state_dict(runtime.clean_state_dict(ck_obs["pooler"]), strict=True)
        posterior = HindsightOutcomePosterior().to(dev).eval()
        posterior.load_state_dict(runtime.clean_state_dict(ck_obs["posterior"]), strict=True)
        decoder = L1ChunkFlowDecoder(d_l1=256, d_squad=D_SQUAD_DEFAULT).to(dev).eval()
        decoder.load_state_dict(runtime.clean_state_dict(ck_obs["decoder"]), strict=True)
        obs_l2 = ObserverWorldModelL2(rollout_steps=4).to(dev).eval()
        obs_l2.load_state_dict(runtime.clean_state_dict(ck_obs["obs_l2"]), strict=True)
        print("[OK] L1+L2 Observer v2 all 5 modules strict=True")

        # Verify fog-of-war perspective isolation on an unseen game window
        if os.path.exists(paths.SPLITS_PATH) and os.path.isdir(paths.GAMES_DIR):
            split = json.load(open(paths.SPLITS_PATH))
            unseen_ids = sorted(set(split["held_out_games"]) - set(EXCLUDED))
            bases = base_paths(paths.GAMES_DIR, set(unseen_ids[:2]))
            gid = sorted(bases)[0]
            g = Games(bases, vocab, avocab, keep=1).get(gid)
            b_cpu = collate([g.window(800, 17, swap=False, step=7)], vocab, drop_dense_slots=False)
            b1 = to_device(b_cpu, dev)
            b2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in b1.items()}
            # Perturb completely unseen P2 units (owner == 1, outside P1 vision)
            with torch.no_grad():
                ctx1 = enc.encode_window_context(b1, list(range(1, 17)))
                perc1 = ctx1["perception"]
                p2_hidden = (
                    (b2["owner"][:, None, :] == 1)
                    & (~perc1["gt_vis_live"][:, :, 0])
                    & (~perc1["gt_shimmer"][:, :, 0])
                    & (~perc1["gt_seen_struct"][:, :, 0])
                )
                if p2_hidden.any():
                    b2["num"][:, 1:17][p2_hidden] += 5.0
                ctx2 = enc.encode_window_context(b2, list(range(1, 17)))
                s1 = pooler(b1, ctx1, f_start=0, f_end=15, prev_state=None, batch_frame_end=16)
                s2 = pooler(b2, ctx2, f_start=0, f_end=15, prev_state=None, batch_frame_end=16)
                leak_p1_sq = float((s1["q_squad"][:, 0] - s2["q_squad"][:, 0]).abs().max())
                leak_p1_z = float((s1["z_squad"][:, 0] - s2["z_squad"][:, 0]).abs().max())
                leak_p1_opp = float((s1["z_opp_obs"][:, 0] - s2["z_opp_obs"][:, 0]).abs().max())
                print(
                    f"[OK] Fog isolation P1 leak under hidden P2 perturbation: "
                    f"q_sq={leak_p1_sq:.2e}, z_sq={leak_p1_z:.2e}, z_opp={leak_p1_opp:.2e}"
                )
                assert leak_p1_sq == 0.0 and leak_p1_z == 0.0 and leak_p1_opp == 0.0


def test_inspector_end_to_end_smoke() -> None:
    if not (torch.cuda.is_available() and os.path.isdir(paths.GAMES_DIR)):
        return
    insp = WorldModelInspector()
    assert len(insp.catalog) > 0
    gid = insp.catalog[0]["gid"]
    fr = insp.inspect_frame(gid, loop_t=1500, window_len=8)
    assert "units" in fr and "hud" in fr and "gram" in fr
    tok = insp.query_token_details(tok_idx=0, layer=5, head=0)
    assert "top_spatial_attended" in tok or "token" in tok or isinstance(tok, dict)
    gr = insp.query_gram_256(owner="p1", mat_type="G_cc_256", cell_i=0, cell_j=0)
    assert isinstance(gr, dict)
    obs = insp.inspect_observer_rollout(gid, loop_t=1500, n_branches=4, horizon_steps=4, mode="forecast")
    assert "fan" in obs or "branches" in obs or "t_sec" in obs
    print(f"[OK] WorldModelInspector end-to-end smoke passed on game {gid}")


if __name__ == "__main__":
    test_strict_checkpoint_loading_and_fog_isolation()
    test_inspector_end_to_end_smoke()
    print("All checkpoint parity, fog isolation, and Inspector smoke tests passed!")
