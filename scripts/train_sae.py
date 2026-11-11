"""Trains the 7-Layer Post-Hoc Top-K Sparse Autoencoder (SAE) Stack on frozen `Frames`
entity representations across Layer 0 (Input Embedding) and Transformer Blocks 1..6,
auto-labels every dictionary feature by its top activating unit types and orders, and
saves the stack to `data/worldmodel/sae/sae_7layer.pt`.

Usage:
    .venv/bin/python scripts/train_sae.py --ckpt <path_to_model_step.pt>
"""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch
import torch.nn.functional as F
import train_physics as tp

from cosmicstar.worldmodel import energy as E, energy_data as ED, physics_data as PD
from cosmicstar.worldmodel.action_tokens import ActionVocab
from cosmicstar.worldmodel.sae import MultiLayerSAEStack
from cosmicstar.worldmodel.vocab import Vocab


def _find_latest_l1_ckpt() -> str:
    ckpts = sorted(
        [
            p
            for p in glob.glob("data/worldmodel/runs/energy_*/model_step*.pt")
            if "energy_l2" not in p
        ],
        key=lambda p: (
            os.path.getmtime(p),
            int(re.findall(r"step(\d+)", os.path.basename(p))[0])
            if re.findall(r"step(\d+)", os.path.basename(p))
            else 0,
        ),
        reverse=True,
    )
    return ckpts[0] if ckpts else ""


def _build_unit_and_abil_names(
    vocab: Vocab,
    rules_path: str,
    base_build: int,
    unit_data_path: str = "data/worldmodel/unit_data.json",
):
    unit_names = {}
    if os.path.exists(unit_data_path):
        raw_ud = json.load(open(unit_data_path))
        for k, u in raw_ud.items():
            unit_names[int(k)] = u.get("name", f"Unit_{k}")

    raw = json.load(open(rules_path))
    rs = None
    for k, v in raw.get("rulesets", {}).items():
        if k.startswith(f"{base_build}_"):
            rs = v
            break
    if rs is None and raw.get("rulesets"):
        rs = next(iter(raw["rulesets"].values()))

    if rs and "units" in rs and not unit_names:
        units_obj = rs["units"]
        if isinstance(units_obj, dict):
            for uid_str, u in units_obj.items():
                unit_names[int(uid_str)] = u.get("name", f"Unit_{uid_str}")
        elif isinstance(units_obj, list):
            for u in units_obj:
                if "id" in u:
                    unit_names[int(u["id"])] = u.get("name", f"Unit_{u['id']}")

    raw_abil_name = {0: "Idle"}
    if rs and "abilities" in rs:
        for ab in rs["abilities"]:
            nm = ab.get("name", "")
            short = nm.split("/")[-1] if "/" in nm else nm
            for aid in ab.get("api_ids", []):
                if aid is not None and aid >= 0:
                    raw_abil_name[int(aid)] = short

    abil_names = {0: "Idle", 1: "Idle"}
    for raw_a, v_idx in getattr(vocab, "_abil", {}).items():
        abil_names[int(v_idx)] = raw_abil_name.get(int(raw_a), f"Abil_{raw_a}")
    return unit_names, abil_names


@torch.no_grad()
def collect_layer_activations(
    enc: ED.Frames,
    games: tp.Games,
    windows: list[tuple],
    vocab: Vocab,
    n_frames: int = 8,
    max_entities: int = 50000,
    device: str = "cuda",
):
    """Runs `enc` over sampled game windows and collects 7-layer entity states [N, 7, 256]
    along with unit rtype, active order ability, and owner metadata for auto-labeling."""
    enc.eval()
    all_states = []  # list of [N_i, 7, 256]
    all_rtypes = []
    all_abils = []
    all_owners = []
    total_n = 0

    for w_idx, (gid, t0, swap) in enumerate(windows):
        try:
            w = games.read(gid, t0, n_frames + 2, swap)
            b_cpu = {k: v for k, v in PD.collate([w], vocab).items() if k in ED.BATCH_KEYS}
            b = ED.to_device(b_cpu, device)
        except Exception:
            continue

        all_f = list(range(1, n_frames + 1))
        f_idx = torch.as_tensor(all_f, device=device)
        B, _, R = b["mask"].shape
        nf = len(all_f)

        q = enc.coords(b, f_idx)
        rtype = b["rtype"][:, f_idx].clamp(0, ED.N_RAW_TYPES - 1)
        owner = b["owner"][:, None].expand(B, nf, R)
        owner = torch.where(owner == ED.NEUTRAL, E.GROUND, owner + 1)
        on_map = b["mask"][:, f_idx] & (b["inside"][:, f_idx] == 0)
        mask = on_map & b["alive"][:, f_idx]

        num = b["num"][:, f_idx]
        kt = enc.k_type[rtype]
        kt_comp = torch.sign(kt) * torch.log1p(kt.abs()) / 8.0
        abil = b["slot_abil"][:, f_idx, :, 0]
        prod_abil = b["slot_abil"][:, f_idx, :, PD.QUEUE]

        extras = num[..., enc.extras_at]
        extras = torch.sign(extras) * torch.log1p(extras.abs()) / 8.0
        ext_in = torch.cat([extras, kt_comp], dim=-1)

        xy_raw = torch.stack([b["x"][:, f_idx], b["y"][:, f_idx]], dim=-1).float()
        xy_ang = xy_raw.unsqueeze(-1) * enc.map_omega
        xy_fourier = torch.cat(
            [xy_ang.sin().flatten(-2), xy_ang.cos().flatten(-2)], dim=-1
        ).to(q.dtype)

        facing = num[..., ED.I["facing"]]
        ord_reg = b["slot_reg"][:, f_idx, :, 0, :]
        kin_in = torch.cat(
            [q, xy_fourier, facing.sin().unsqueeze(-1), facing.cos().unsqueeze(-1), ord_reg],
            dim=-1,
        )

        center = (
            b["center"][:, None, None, :].float()
            if "center" in b
            else torch.full((B, 1, 1, 2), 100.0, device=device)
        )
        pos_ent = xy_raw - center

        eff_tok, eff_mask, eff_xy = enc._embed_effects(b, f_idx)
        g0, X = 3, eff_tok.shape[2]
        T_t = g0 + R
        T = T_t + X

        with torch.autocast("cuda", dtype=torch.bfloat16):
            unit_c, ord_c, prod_c, _ = enc.nodes(rtype, abil, prod_abil)
            field_ent = enc.field(b, f_idx, pos_ent)
            dt = ED._compute_dtype(enc.cls_tok)

            own_c = enc.owner_emb(owner).to(dt)
            kin_c = enc.kin_c(kin_in).to(dt)
            ext_c = enc.extras(ext_in).to(dt)

            c_ent = (unit_c + ord_c + prod_c + own_c + kin_c + ext_c + field_ent).to(dt)
            g_tok = enc._embed_globals(b, f_idx).to(dt)
            x_tok = torch.cat([g_tok, c_ent, eff_tok.to(dt)], dim=2).reshape(B * nf, T, -1)

            valid_g = torch.ones(B, nf, g0, dtype=torch.bool, device=device)
            valid_all = torch.cat([valid_g, mask, eff_mask], dim=2)
            valid_flat = valid_all.reshape(B * nf, 1, 1, T)

            t_valid = torch.cat([valid_g, mask], dim=2)
            t_mask = t_valid.transpose(1, 2).reshape(B * T_t, nf)
            tri = torch.tril(torch.ones(nf, nf, dtype=torch.bool, device=device))
            eye = torch.eye(nf, dtype=torch.bool, device=device)
            allow_t = ((tri[None] & t_mask[:, None, :]) | eye[None])[:, None]
            t_mask_3d = t_mask.unsqueeze(-1)

            pos_g = torch.zeros(B, nf, g0, 2, device=device, dtype=torch.float32)
            pos_all = torch.cat([pos_g, pos_ent, eff_xy], dim=2)
            cs_2d = ED.cos_sin(enc.rope_2d.angles(pos_all.reshape(B * nf, T, 2)))
            h2 = cs_2d.shape[-2]
            p_t = (
                cs_2d[:, :T_t]
                .view(B, nf, T_t, h2, 2)
                .transpose(1, 2)
                .reshape(B * T_t, nf, h2, 2)
            )
            loops = b["loops"][:, f_idx] - b["loops"][:, f_idx[:1]]
            t_cs = ED.cos_sin(enc.rope_1d.angles(loops))
            cs_1d = t_cs[:, None, :, :, :].expand(B, T_t, nf, h2, 2).reshape(B * T_t, nf, h2, 2)

            # Take the target frame (last frame of window), keeping all Player 1 & Player 2 units
            # plus up to 3 neutral entities so the 128 SAE features specialize in combat & economy units
            t_pick = nf - 1
            live_m = mask[0, t_pick]  # [R]
            own_t = owner[0, t_pick]  # [R]
            ply_rows = torch.nonzero(live_m & ((own_t == 1) | (own_t == 2))).squeeze(-1)
            neu_rows = torch.nonzero(live_m & ~((own_t == 1) | (own_t == 2))).squeeze(-1)[:3]
            live_rows = torch.cat([ply_rows, neu_rows], dim=0)
            if len(live_rows) == 0:
                continue

            tok_rows = g0 + live_rows
            layer_list = [MultiLayerSAEStack.preprocess_states(x_tok[t_pick, tok_rows])]

            for blk in enc.blocks:
                x_tok = blk(x_tok, valid_flat, cs_2d, allow_t, t_mask_3d, cs_1d, p_t)
                layer_list.append(MultiLayerSAEStack.preprocess_states(x_tok[t_pick, tok_rows]))

        st_7 = torch.stack(layer_list, dim=1).cpu()  # [N_live, 7, 256]
        all_states.append(st_7)
        all_rtypes.append(rtype[0, t_pick, live_rows].cpu().numpy())
        all_abils.append(abil[0, t_pick, live_rows].cpu().numpy())
        all_owners.append(owner[0, t_pick, live_rows].cpu().numpy())

        total_n += len(live_rows)
        if total_n >= max_entities:
            break

    states_cat = torch.cat(all_states, dim=0)[:max_entities]
    rtypes_cat = np.concatenate(all_rtypes, axis=0)[:max_entities]
    abils_cat = np.concatenate(all_abils, axis=0)[:max_entities]
    owners_cat = np.concatenate(all_owners, axis=0)[:max_entities]
    return states_cat, rtypes_cat, abils_cat, owners_cat


def auto_label_sae_features(
    sae_stack: MultiLayerSAEStack,
    states_cat: torch.Tensor,  # [N, 7, 256]
    rtypes_cat: np.ndarray,
    abils_cat: np.ndarray,
    owners_cat: np.ndarray,
    unit_names: dict[int, str],
    abil_names: dict[int, str],
    device: str = "cuda",
):
    """Computes human-readable monosemantic labels for every SAE feature (j = 0..d_sae-1)
    across all 7 layers (l = 0..6)."""
    N = states_cat.shape[0]
    d_sae = sae_stack.d_sae
    meta_all = []

    for l_idx in range(sae_stack.n_layers):
        sae = sae_stack.layers[l_idx]
        f_chunks = []
        for i in range(0, N, 4096):
            x_b = states_cat[i : i + 4096, l_idx].to(device)
            with torch.no_grad():
                f_b, _, _ = sae.encode(x_b)
            f_chunks.append(f_b.cpu().numpy())
        f_mat = np.concatenate(f_chunks, axis=0)  # [N, d_sae]

        layer_meta = []
        for j in range(d_sae):
            col = f_mat[:, j]
            active_idx = np.nonzero(col > 1e-3)[0]
            freq = float(len(active_idx)) / max(1, N)
            if len(active_idx) == 0:
                layer_meta.append(
                    {
                        "id": j,
                        "label": f"L{l_idx}·F{j:03d} (Unused)",
                        "top_units": "—",
                        "top_orders": "—",
                        "freq": 0.0,
                        "mean_act": 0.0,
                    }
                )
                continue

            mean_act = float(col[active_idx].mean())
            # Weight top activating instances by activation strength
            order_sort = active_idx[np.argsort(-col[active_idx])[:60]]
            u_counts = Counter(unit_names.get(int(rtypes_cat[idx]), f"U{rtypes_cat[idx]}") for idx in order_sort)
            top1_cnt = u_counts.most_common(1)[0][1] if u_counts else 0
            top_u_list = [
                k for k, cnt in u_counts.most_common(2)
                if cnt >= max(3, int(0.20 * len(order_sort))) and cnt >= int(0.35 * top1_cnt)
            ]
            if not top_u_list and u_counts:
                top_u_list = [u_counts.most_common(1)[0][0]]

            # Only count orders belonging to the top unit types, and require >=25% share to avoid minority noise
            top_u_set = set(top_u_list)
            a_counts = Counter(
                abil_names.get(int(abils_cat[idx]), f"A{abils_cat[idx]}")
                for idx in order_sort
                if unit_names.get(int(rtypes_cat[idx]), f"U{rtypes_cat[idx]}") in top_u_set
            )
            min_a_cnt = max(4, int(0.25 * max(1, sum(a_counts.values()))))
            top_a_list = [k for k, cnt in a_counts.most_common(2) if k != "Idle" and cnt >= min_a_cnt]
            top_u_str = "/".join(top_u_list) if top_u_list else "Entity"
            top_a_str = "/".join(top_a_list) if top_a_list else "Idle"

            short_label = f"F{j:03d}: {top_u_str} ({top_a_str})"
            layer_meta.append(
                {
                    "id": j,
                    "label": short_label,
                    "top_units": ", ".join(f"{k}({v})" for k, v in u_counts.most_common(3)),
                    "top_orders": ", ".join(f"{k}({v})" for k, v in a_counts.most_common(3)),
                    "freq": round(freq, 4),
                    "mean_act": round(mean_act, 3),
                }
            )
        meta_all.append(layer_meta)

    sae_stack.feature_meta = meta_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="data/worldmodel/sae/sae_7layer.pt")
    ap.add_argument("--games", default="data/games_v3")
    ap.add_argument("--vocab", default="data/worldmodel/frame_cache_seq/vocab.json")
    ap.add_argument("--split", default="data/games/splits/default.json")
    ap.add_argument("--rules", default="data/worldmodel/rulesets.json")
    ap.add_argument("--base_build", type=int, default=94137)
    ap.add_argument("--patch", default="5.0.14")
    ap.add_argument("--d_sae", type=int, default=128)
    ap.add_argument("--k_active", type=int, default=4)
    ap.add_argument("--windows", type=int, default=140)
    ap.add_argument("--max_entities", type=int, default=45000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    t0 = time.time()
    ckpt_path = args.ckpt or _find_latest_l1_ckpt()
    print(f"[SAE] Using encoder checkpoint: {ckpt_path or '(fresh init)'}", flush=True)

    vocab = Vocab(json.load(open(args.vocab)))
    avocab = ActionVocab()
    tables = ED.Tables(args.base_build, rulesets=args.rules)
    enc = ED.Frames(vocab, tables, 256, 32, attn_layers=6, dh=64, ff=768).cuda().eval()

    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        e_sd = enc.state_dict()
        ck_e = {k: v for k, v in ck["enc"].items() if k in e_sd and e_sd[k].shape == v.shape}
        enc.load_state_dict(ck_e, strict=False)
        print(f"[SAE] Loaded {len(ck_e)}/{len(e_sd)} encoder tensors from {ckpt_path}", flush=True)

    split = json.load(open(args.split))
    train_ids = set(split["train"]) - set(tp.EXCLUDED)
    bases = tp.of_patch(PD.base_paths(args.games, train_ids), args.patch)
    map_ids = {m: i for i, m in enumerate(sorted({tp.map_of(b) for b in bases.values()}))}
    wins = tp.fixed_windows(bases, args.windows, 12)
    games = tp.Games(bases, vocab, avocab, keep=16, map_ids=map_ids)

    print(f"[SAE] Collecting 7-layer entity activations across {len(wins)} windows...", flush=True)
    states_cat, rtypes_cat, abils_cat, owners_cat = collect_layer_activations(
        enc, games, wins, vocab, n_frames=8, max_entities=args.max_entities
    )
    N = states_cat.shape[0]
    print(f"[SAE] Collected {N} live entity states across 7 layers in {time.time() - t0:.1f}s", flush=True)

    sae_stack = MultiLayerSAEStack(
        n_layers=7, d_in=256, d_sae=args.d_sae, k_active=args.k_active
    ).cuda()

    # Initialize b_pre to empirical mean of each layer
    with torch.no_grad():
        for l_idx in range(7):
            sae_stack.layers[l_idx].b_pre.data.copy_(states_cat[:, l_idx].mean(dim=0).cuda())

    opt = torch.optim.Adam(sae_stack.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)

    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(N)
        r2_sums = np.zeros(7)
        n_batches = 0
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            x_batch = states_cat[idx].cuda(non_blocking=True)  # [B, 7, 256]
            loss_tot = 0.0
            for l_idx in range(7):
                out = sae_stack.layers[l_idx](x_batch[:, l_idx])
                loss_tot = loss_tot + out["loss"]
                r2_sums[l_idx] += float(out["r2"].item())
            opt.zero_grad(set_to_none=True)
            loss_tot.backward()
            sae_stack.normalize_decoders_()
            opt.step()
            n_batches += 1
        sched.step()
        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            r2_avg = [round(v / max(1, n_batches), 4) for v in r2_sums]
            print(f"[SAE] Epoch {ep:02d}/{args.epochs} • Layer R² (L0..L6): {r2_avg}", flush=True)

    unit_names, abil_names = _build_unit_and_abil_names(vocab, args.rules, args.base_build)
    auto_label_sae_features(
        sae_stack, states_cat, rtypes_cat, abils_cat, owners_cat, unit_names, abil_names
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(
        {
            "state_dict": sae_stack.state_dict(),
            "n_layers": sae_stack.n_layers,
            "d_in": sae_stack.d_in,
            "d_sae": sae_stack.d_sae,
            "k_active": sae_stack.k_active,
            "feature_meta": sae_stack.feature_meta,
            "source_ckpt": ckpt_path,
        },
        args.out,
    )
    print(f"[SAE] Saved 7-layer SAE stack + auto-labels to {args.out} ({time.time() - t0:.1f}s total)", flush=True)
    print("[SAE] Sample L6 (Final Layer) Top Features:")
    for m in sorted(sae_stack.feature_meta[6], key=lambda x: -x["freq"])[:10]:
        print(f"  {m['label']:34s} | freq={m['freq']*100:5.1f}% | units: {m['top_units']}")


if __name__ == "__main__":
    main()
