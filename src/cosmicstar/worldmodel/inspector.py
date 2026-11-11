"""Interactive Replay & Neural World Model Inspector Backend.

Provides deep, on-demand introspection of:
  1. StarCraft II replay states, terrain grids, and .s2ma map objects (NoFlyZone air
     blockers, XelNagaHealingShrines, cliffs, vision, creep, pathing).
  2. Layer 1 (`Frames` + `EnergyModel`) spatiotemporal representations across all 6
     transformer blocks:
       - Additive input embedding decomposition (unit, order, prod, owner, kin, extras, field)
       - Dense 2D evaluation of the learned `ContinuousPotentialField` V_loc(x, y)
       - Exact 2D-RoPE spatial attention weights [L=6, H=4, T, T] and 1D-RoPE causal
         temporal attention weights [L=6, H=4, T_t, nf]
       - Layer-by-layer token norms, cosine evolution, and 3D PCA -> RGB projections
  3. Gram Phase-Space & Spectral Attribution:
       - Layer 1 16x16 Gram matrices G^(p) = (1/N0) [Q_p | Z_p]^T [Q_p | Z_p] for
         P1, P2, and Neutral, plus autograd sensitivity grad_{G^(p)} (-A_1)
       - Layer 2 full-rank 256x256 G_cc^(p) = (1/N0) C_p^T C_p singular value spectrum,
         effective rank, top eigenmode unit loadings, and learned W_c projections
  4. Port-Hamiltonian Flow & End-to-End Autograd Force Fields:
       - Decomposition of dx into Conservative J(x) grad H, Dissipative -R(x) grad H,
         and Control B(x) u vs ground-truth dx
       - Per-unit marginal strategic advantage d(-A_1)/dw_i
       - End-to-end spatial force vectors -grad_{(x_i, y_i)} A_1 differentiated all the
         way through the 6-layer 2D-RoPE transformer and ContinuousPotentialField
  5. Interactive Counterfactual Sandbox:
       - Delete units, move/drag units to new (x, y) map positions, scale HP/shields,
         or adjust player minerals/vespene and immediately compute Delta H_1, H_2, A_1.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any

import mpyq
import numpy as np
import torch
import torch.nn.functional as F

from . import energy as E
from . import energy_data as ED
from . import energy_l2 as EL2
from . import physics_data as PD
from . import statefile
from .action_tokens import ActionVocab
from .sae import MultiLayerSAEStack
from .vocab import Vocab


def _trim_batch(b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {k: b[k] for k in ED.BATCH_KEYS if k in b}
    if "slot_reg" in out:
        out["slot_reg"] = out["slot_reg"][..., :1, :].contiguous()
    if "slot_abil" in out:
        out["slot_abil"] = out["slot_abil"][..., : PD.QUEUE + 1].contiguous()
    return out


def _pca_rgb(vecs: np.ndarray) -> np.ndarray:
    """Projects [N, D] vectors onto their top 3 principal components and normalizes to [0, 255] RGB."""
    N, D = vecs.shape
    if N == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if N < 3:
        out = np.full((N, 3), 160, dtype=np.uint8)
        return out
    centered = vecs - vecs.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ vt[:3].T  # [N, 3]
        lo = np.percentile(proj, 5, axis=0, keepdims=True)
        hi = np.percentile(proj, 95, axis=0, keepdims=True)
        normed = np.clip((proj - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
        return (normed * 215.0 + 40.0).astype(np.uint8)
    except Exception:
        return np.full((N, 3), 160, dtype=np.uint8)


class S2MAMapCache:
    """Parses .s2ma MPQ map archives from the Battle.net Cache to extract exact
    NoFlyZone air blockers, XelNagaHealingShrines, and special terrain objects."""

    def __init__(self, cache_glob: str | None = None):
        if cache_glob is None:
            cache_glob = os.path.expanduser(
                "~/Games/battlenet/drive_c/ProgramData/Blizzard Entertainment/Battle.net/Cache/*/*/*.s2ma"
            )
        self.by_hash: dict[str, str] = {}
        self.by_name: dict[str, str] = {}
        self._parsed: dict[str, dict[str, Any]] = {}

        for p in glob.glob(cache_glob):
            h = os.path.splitext(os.path.basename(p))[0]
            self.by_hash[h] = p
            self.by_hash[h[:16]] = p
            try:
                a = mpyq.MPQArchive(p, listfile=False)
                dh = a.read_file("DocumentHeader")
                if dh:
                    strings = [s.decode("utf-8", errors="ignore") for s in re.findall(b"[ -~]{4,}", dh)]
                    for idx, s in enumerate(strings):
                        if "DocInfo/Name" in s and idx + 1 < len(strings):
                            name = strings[idx + 1].strip()
                            self.by_name[name] = p
                            break
            except Exception:
                continue

    def get_map_objects(self, map_hash: str, map_name: str) -> dict[str, Any]:
        p = (
            self.by_hash.get(map_hash)
            or self.by_hash.get(map_hash[:16])
            or self.by_name.get(map_name)
        )
        if not p:
            return {"found_s2ma": False, "no_fly_zones": [], "healing_shrines": [], "objects": []}
        if p in self._parsed:
            return self._parsed[p]

        no_fly_zones = []
        healing_shrines = []
        other_objs = []
        try:
            a = mpyq.MPQArchive(p, listfile=False)
            objs_raw = a.read_file("Objects")
            if objs_raw:
                root = ET.fromstring(objs_raw.decode("utf-8", errors="ignore"))
                for el in root:
                    pos_str = el.attrib.get("Position", "")
                    parts = [float(x) for x in pos_str.split(",") if x.strip()]
                    if len(parts) < 2:
                        continue
                    x, y = parts[0], parts[1]
                    z = parts[2] if len(parts) > 2 else 0.0
                    t = el.attrib.get("Type", "") or el.attrib.get("UnitType", "")
                    if t == "NoFlyZone":
                        r_hard = float(el.attrib.get("PathingRadiusHard") or 3.0)
                        r_soft = float(el.attrib.get("PathingRadiusSoft") or r_hard)
                        no_fly_zones.append(
                            {
                                "x": round(x, 2),
                                "y": round(y, 2),
                                "z": round(z, 2),
                                "r_hard": round(r_hard, 2),
                                "r_soft": round(r_soft, 2),
                            }
                        )
                    elif "HealingShrine" in t:
                        healing_shrines.append(
                            {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2), "type": t}
                        )
                    elif any(
                        k in t
                        for k in (
                            "InhibitorZone",
                            "AccelerationZone",
                            "XelNagaTower",
                            "Destructible",
                            "Unbuildable",
                        )
                    ):
                        other_objs.append(
                            {"x": round(x, 2), "y": round(y, 2), "type": t}
                        )
        except Exception:
            pass

        res = {
            "found_s2ma": True,
            "s2ma_path": os.path.basename(p),
            "no_fly_zones": no_fly_zones,
            "healing_shrines": healing_shrines,
            "objects": other_objs[:120],
        }
        self._parsed[p] = res
        return res


class WorldModelInspector:
    """Stateful GPU inspector engine for Layer 1 (`Frames` + `EnergyModel`) and
    Layer 2 (`EnergyModelL2`) over StarCraft II replays."""

    def __init__(
        self,
        games_dir: str = "data/games_v3",
        vocab_path: str = "data/worldmodel/frame_cache_seq/vocab.json",
        split_path: str = "data/games/splits/default.json",
        unit_data_path: str = "data/worldmodel/unit_data.json",
        rulesets_path: str = "data/worldmodel/rulesets.json",
        l1_ckpt: str = "data/worldmodel/runs/energy_r16_zerogram/model_step120.pt",
        l2_ckpt: str = "data/worldmodel/runs/energy_l2_535g_cont_5k/model_l2_step5000.pt",
        l2_summary_dir: str = "data/worldmodel/l2_summaries_5014_cont",
        l2_burst_summary_dir: str = "data/worldmodel/l2_summaries_5014",
        patch: str = "5.0.14",
        base_build: int = 94137,
        device: str = "cuda",
    ):
        self.lock = threading.Lock()
        self.device = device
        self.games_dir = games_dir
        self.patch = patch
        self.l2_summary_dir = l2_summary_dir
        self.l2_burst_summary_dir = l2_burst_summary_dir

        self.vocab = Vocab(json.load(open(vocab_path)))
        self.avocab = ActionVocab()
        self.tables = ED.Tables(base_build, unit_data=unit_data_path, rulesets=rulesets_path)
        self.s2ma_cache = S2MAMapCache()

        # Unit metadata lookup
        self.unit_meta: dict[int, dict[str, Any]] = {}
        raw_ud = json.load(open(unit_data_path))
        for k, u in raw_ud.items():
            uid = int(k)
            cat = "other"
            if u.get("worker"):
                cat = "worker"
            elif u.get("structure"):
                cat = "structure"
            elif u.get("food", 0) > 0:
                cat = "army"
            self.unit_meta[uid] = {
                "name": u.get("name", f"Unit_{uid}"),
                "race": u.get("race", 0),
                "minerals": u.get("minerals", 0),
                "vespene": u.get("vespene", 0),
                "food": u.get("food", 0.0),
                "structure": bool(u.get("structure", False)),
                "worker": bool(u.get("worker", False)),
                "category": cat,
            }

        # Ability name lookup (indexed by vocab ability index and raw ability id)
        rs_doc = json.load(open(rulesets_path))
        g_rs = rs_doc["rulesets"][self.tables.ruleset]
        self.raw_abil_name: dict[int, str] = {0: "None"}
        for ab in g_rs["abilities"]:
            nm = ab.get("name", "")
            for aid in ab.get("api_ids", []):
                if aid is not None and aid >= 0:
                    self.raw_abil_name[int(aid)] = nm
        self.vocab_abil_name: dict[int, str] = {0: "None", 1: "Unknown"}
        for raw_a, v_idx in self.vocab._abil.items():
            self.vocab_abil_name[int(v_idx)] = self.raw_abil_name.get(int(raw_a), f"Abil_{raw_a}")

        # Discover games (unseen_games + train)
        split = json.load(open(split_path))
        import train_physics as tp

        unseen_ids = set(split["held_out_games"]) - set(tp.EXCLUDED)
        train_ids = set(split["train"]) - set(tp.EXCLUDED)
        self.unseen_bases = tp.of_patch(PD.base_paths(games_dir, unseen_ids), patch)
        self.train_bases = tp.of_patch(PD.base_paths(games_dir, train_ids), patch)
        self.all_bases = {**self.unseen_bases, **self.train_bases}
        self.game_subset = {
            **{gid: "unseen_games" for gid in self.unseen_bases},
            **{gid: "train" for gid in self.train_bases},
        }

        # Build lightweight game catalog
        self.catalog: list[dict[str, Any]] = []
        self.catalog_by_id: dict[str, dict[str, Any]] = {}
        self._build_catalog()

        # Loaded physics_data.Game LRU cache
        self._open_games: dict[str, PD.Game] = {}

        # Initialize Layer 1 and Layer 2 models
        self.model_l1 = E.EnergyModel(256, 32, 256).to(device).eval()
        self.enc_l1 = ED.Frames(
            self.vocab, self.tables, 256, 32, attn_layers=6, dh=64, ff=768
        ).to(device).eval()
        self.enc_l1.grad_ckpt = False

        self.model_l2 = EL2.EnergyModelL2(
            d=256, layers=4, heads=8, r_gram=16, r_macro=32, u_dim=32, hidden=256, rank=16
        ).to(device).eval()

        # Cache of last evaluated frame for <1ms interactive token/Gram drilldowns
        self._last_frame_cache: dict[str, Any] = {}

        self.active_l1_ckpt = ""
        self.active_l2_ckpt = ""
        if os.path.exists(l1_ckpt):
            self.load_l1_checkpoint(l1_ckpt)
        if os.path.exists(l2_ckpt):
            self.load_l2_checkpoint(l2_ckpt)

        # Optional 7-layer post-hoc Top-K SAE stack (auto-reloaded if updated on disk)
        self.sae_path = "data/worldmodel/sae/sae_7layer.pt"
        self.sae_mtime = 0.0
        self.sae_stack: MultiLayerSAEStack | None = None
        self._maybe_load_sae()

    def _maybe_load_sae(self):
        if not os.path.exists(self.sae_path):
            return
        try:
            mt = os.path.getmtime(self.sae_path)
            if self.sae_stack is not None and mt <= self.sae_mtime:
                return
            ck = torch.load(self.sae_path, map_location=self.device, weights_only=False)
            stack = MultiLayerSAEStack(
                n_layers=int(ck.get("n_layers", 7)),
                d_in=int(ck.get("d_in", 256)),
                d_sae=int(ck.get("d_sae", 128)),
                k_active=int(ck.get("k_active", 4)),
            ).to(self.device).eval()
            stack.load_state_dict(ck["state_dict"])
            if "feature_meta" in ck:
                stack.feature_meta = ck["feature_meta"]
            self.sae_stack = stack
            self.sae_mtime = mt
        except Exception:
            pass

    def _build_catalog(self):
        # Prioritize all 30 unseen games + first 80 train games for instant startup,
        # while allowing any gid in all_bases to be queried
        ordered_gids = sorted(self.unseen_bases.keys()) + sorted(self.train_bases.keys())[:100]
        for gid in ordered_gids:
            base = self.all_bases[gid]
            subset = self.game_subset[gid]
            try:
                d = statefile.load(f"{base}.p1.cst")
                meta = json.loads(str(d["meta"]))
                players = meta.get("players", [{}, {}])
                p1 = players[0] if len(players) > 0 else {}
                p2 = players[1] if len(players) > 1 else {}
                res = meta.get("end", {}).get("results", {})
                winner = 1 if res.get("1") == 1 else (2 if res.get("2") == 1 else 0)
                loops = int(meta.get("duration_loops", 0))
                info = {
                    "gid": gid,
                    "subset": subset,
                    "map": meta.get("map", "Unknown Map"),
                    "map_hash": meta.get("map_hash", ""),
                    "p1_name": p1.get("name", "Player 1"),
                    "p1_race": p1.get("race", "Unknown"),
                    "p2_name": p2.get("name", "Player 2"),
                    "p2_race": p2.get("race", "Unknown"),
                    "winner": winner,
                    "duration_loops": loops,
                    "duration_s": round(loops / 22.4, 1),
                }
                self.catalog.append(info)
                self.catalog_by_id[gid] = info
            except Exception:
                continue

    def list_checkpoints(self) -> dict[str, list[str]]:
        l1_ckpts = sorted(
            [
                p
                for p in glob.glob("data/worldmodel/runs/energy_*/model_step*.pt")
                if "energy_l2" not in p
            ],
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        l2_ckpts = sorted(
            glob.glob("data/worldmodel/runs/energy_l2_535g_cont_5k/model_l2_step*.pt")
            + glob.glob("data/worldmodel/runs/energy_l2_535g_5k/model_l2_step*.pt"),
            key=lambda p: (
                os.path.dirname(p),
                int(re.findall(r"step(\d+)", os.path.basename(p))[0])
                if re.findall(r"step(\d+)", os.path.basename(p))
                else 0,
            ),
            reverse=True,
        )
        return {"l1": l1_ckpts[:25], "l2": l2_ckpts[:20]}

    def load_l1_checkpoint(self, path: str):
        with self.lock:
            ck = torch.load(path, map_location=self.device, weights_only=False)
            m_sd = self.model_l1.state_dict()
            ck_m = {k: v for k, v in ck["model"].items() if k in m_sd and m_sd[k].shape == v.shape}
            self.model_l1.load_state_dict(ck_m, strict=False)
            e_sd = self.enc_l1.state_dict()
            ck_e = {k: v for k, v in ck["enc"].items() if k in e_sd and e_sd[k].shape == v.shape}
            self.enc_l1.load_state_dict(ck_e, strict=False)
            self.model_l1.eval()
            self.enc_l1.eval()
            self.active_l1_ckpt = path
            self._last_frame_cache.clear()

    def load_l2_checkpoint(self, path: str):
        with self.lock:
            ck = torch.load(path, map_location=self.device, weights_only=False)
            self.model_l2.load_state_dict(ck["model"])
            self.model_l2.eval()
            self.active_l2_ckpt = path

    def _get_game(self, gid: str) -> PD.Game:
        if gid not in self._open_games:
            if len(self._open_games) >= 6:
                oldest = next(iter(self._open_games))
                del self._open_games[oldest]
            base = self.all_bases[gid]
            g = PD.Game(base, self.vocab)
            g._load_actions(self.avocab)
            self._open_games[gid] = g
        return self._open_games[gid]

    def _reconstruct_gcc_triu(self, d: dict[str, Any]) -> torch.Tensor:
        """If the summary .pt stores sparse unit latents `u_c` instead of `G_cc_triu`,
        materializes `G_cc_triu` [T, 2, 32896] on GPU."""
        if "G_cc_triu" in d:
            return d["G_cc_triu"]
        u_c = d["u_c"].to(self.device).float()
        u_owner = d["u_owner"].to(self.device)
        fptr = d["frame_ptr"].numpy()
        T = len(d["loops"])
        ii, jj = E._triu_indices(256)
        out = torch.zeros((T, 2, len(ii)), dtype=torch.float16, device="cpu")
        for t in range(T):
            a, b = int(fptr[t]), int(fptr[t + 1])
            if b <= a:
                continue
            ct = u_c[a:b]
            ot = u_owner[a:b]
            for p_idx, p_val in enumerate((1, 2)):
                cp = ct[ot == p_val]
                if len(cp) > 0:
                    gcc = (cp.t() @ cp) / E.N0
                    out[t, p_idx] = gcc[ii, jj].half().cpu()
        d["G_cc_triu"] = out
        return out

    def get_timeline(self, gid: str, mode: str = "cont") -> dict[str, Any]:
        """Returns the full-game 5s timeline for `gid`, including Layer 1 and Layer 2
        Hamiltonian energies, advantages, dissipation, ground-truth vitals/counts, and
        Layer 2 causal temporal attention maps."""
        subset = self.game_subset.get(gid, "unseen_games")
        sdir = self.l2_summary_dir if mode == "cont" else self.l2_burst_summary_dir
        pt_path = os.path.join(sdir, subset, f"{gid}.pt")
        if not os.path.exists(pt_path):
            pt_path = os.path.join(self.l2_burst_summary_dir, subset, f"{gid}.pt")

        g_game = self._get_game(gid)
        meta_info = self.catalog_by_id.get(gid, {"gid": gid, "duration_loops": g_game.F})

        if not os.path.exists(pt_path):
            return {"error": f"Summary file not found for {gid}", "meta": meta_info}

        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        self._reconstruct_gcc_triu(d)

        T = len(d["loops"])
        loops = d["loops"].numpy().astype(int).tolist()
        t_sec = [round(l / 22.4, 2) for l in loops]

        # Run Layer 2 model + capture its 4 causal attention layers
        batch = {
            k: d[k].unsqueeze(0).to(self.device)
            for k in (
                "cls_out",
                "ply_out",
                "G_qq",
                "G_qc",
                "G_cc_triu",
                "H1_l1",
                "H2_l1",
                "A1_l1",
                "u_win",
                "act_feat",
                "u_active",
            )
        }
        with self.lock, torch.no_grad():
            x_obs, x_lat, u_l2, u_act_l2, z_macro = self._run_l2_with_attn(batch)
            x_all = torch.cat([x_obs, x_lat], dim=-1)[0]  # [T, 2, d_p]
            hout_l2 = self.model_l2.H_l2(x_all[:, 0], x_all[:, 1])

        # Also step L2 dynamics with grad to get L2 dissipation & goal alignment
        with self.lock, torch.enable_grad():
            _, _, pieces_l2 = self.model_l2.step_dynamics(
                x_all[:, 0].detach(),
                x_all[:, 1].detach(),
                u_l2[0, :, 0].detach(),
                u_l2[0, :, 1].detach(),
            )

        # Oriented energies (> 0 means stronger / deeper potential attractor well)
        h1_l1 = (-d["H1_l1"].float().numpy() * 1000.0).tolist()
        h2_l1 = (-d["H2_l1"].float().numpy() * 1000.0).tolist()
        a1_l1 = (-d["A1_l1"].float().numpy() * 1000.0).tolist()

        h1_l2 = (-hout_l2["H1"].detach().cpu().numpy() * 1000.0).tolist()
        h2_l2 = (-hout_l2["H2"].detach().cpu().numpy() * 1000.0).tolist()
        a1_l2 = (-hout_l2["A1"].detach().cpu().numpy() * 1000.0).tolist()

        a1_tot = [round(v1 + v2, 4) for v1, v2 in zip(a1_l1, a1_l2)]
        diss1_l2 = (pieces_l2["diss1"].detach().cpu().numpy() * 1000.0).tolist()
        diss2_l2 = (pieces_l2["diss2"].detach().cpu().numpy() * 1000.0).tolist()

        # Ground-truth unit counts and HP+Shield pools from G_qq (N0 = 100)
        # In G_qq (36 upper-triangle of 8x8):
        # index 0 is (1, 1) -> count / 100
        # index 5 is (1, hp) -> sum(hp_frac) / 100
        # index 6 is (1, shield) -> sum(shield_frac) / 100
        # index 7 is (1, energy) -> sum(energy_frac) / 100
        g_qq_np = d["G_qq"].float().numpy() * E.N0  # [T, 2, 36]
        p1_units = g_qq_np[:, 0, 0].round(1).tolist()
        p2_units = g_qq_np[:, 1, 0].round(1).tolist()
        p1_hp_eq = (g_qq_np[:, 0, 5] + g_qq_np[:, 0, 6]).round(2).tolist()
        p2_hp_eq = (g_qq_np[:, 1, 5] + g_qq_np[:, 1, 6]).round(2).tolist()

        return {
            "gid": gid,
            "mode": mode,
            "meta": meta_info,
            "T": T,
            "loops": loops,
            "t_sec": t_sec,
            "H1_l1": h1_l1,
            "H2_l1": h2_l1,
            "A1_l1": a1_l1,
            "H1_l2": h1_l2,
            "H2_l2": h2_l2,
            "A1_l2": a1_l2,
            "A1_tot": a1_tot,
            "diss1_l2": diss1_l2,
            "diss2_l2": diss2_l2,
            "p1_units": p1_units,
            "p2_units": p2_units,
            "p1_hp_eq": p1_hp_eq,
            "p2_hp_eq": p2_hp_eq,
            "z_macro": z_macro[0].detach().cpu().numpy().round(3).tolist(),  # [T, 2, 32]
        }

    def _run_l2_with_attn(self, batch: dict[str, torch.Tensor]):
        """Runs MacroSequenceEncoder.forward(batch) to obtain x_obs, x_lat, u, u_active, z_macro."""
        return self.model_l2.seq_enc(batch)

    def inspect_frame(
        self,
        gid: str,
        loop_t: int,
        window_len: int = 8,
        counterfactual: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Runs a deep hooked forward + autograd pass of Layer 1 (`Frames` + `EnergyModel`)
        at game loop `loop_t` (with causal window `window_len` frames ending at `loop_t`,
        plus `loop_t + 1` for ground-truth 1-step transition comparison)."""
        with self.lock:
            t_start = time.time()
            g = self._get_game(gid)
            meta_info = self.catalog_by_id.get(gid, {"gid": gid, "duration_loops": g.F})

            # Clamp loop_t into valid range [window_len + 1, g.F - 3]
            nf = max(4, min(24, int(window_len)))
            loop_t = max(nf + 1, min(g.F - 3, int(loop_t)))
            # We read L = nf + 2 frames so ff = [1 .. nf] encodes transitions [1..nf] -> [2..nf+1],
            # where index nf (i.e. frame index -2 of the nf+1 encoded frames) is loop_t!
            L_read = nf + 2
            t0 = loop_t - nf

            w = g.window(t0, L_read, swap=False)
            b_cpu = _trim_batch(PD.collate([w], self.vocab))
            b = ED.to_device(b_cpu, self.device)

            # Target frame index in `b` is `f_curr = nf` (which corresponds to loop_t),
            # and `f_next = nf + 1` (which corresponds to loop_t + 1).
            all_f = list(range(1, nf + 2))  # length nf_enc = nf + 1; index -2 is f_curr=nf, -1 is f_next=nf+1
            target_idx = nf - 1  # index inside all_f corresponding to b[:, nf] (loop_t)

            # Save baseline un-edited evaluation if counterfactual is requested
            base_scalars = None
            if counterfactual:
                with torch.enable_grad():
                    base_res = self._forward_and_analyze(b, all_f, target_idx, full_analysis=False)
                    base_scalars = base_res["scalars"]
                b = self._apply_counterfactual(b, nf, counterfactual)

            with torch.enable_grad():
                analysis = self._forward_and_analyze(b, all_f, target_idx, full_analysis=True)

            # Extract map grid, .s2ma overlays, and painted wave-packet fields at f_curr = nf
            map_payload = self._extract_map_payload(g, b, nf)

            elapsed_ms = round((time.time() - t_start) * 1000.0, 1)
            out = {
                "gid": gid,
                "loop_t": int(loop_t),
                "time_s": round(loop_t / 22.4, 2),
                "window_len": nf,
                "elapsed_ms": elapsed_ms,
                "active_l1_ckpt": os.path.basename(self.active_l1_ckpt),
                "active_l2_ckpt": os.path.basename(self.active_l2_ckpt),
                "meta": meta_info,
                "map": map_payload,
                "base_scalars": base_scalars,
                **analysis,
            }
            return out

    def _apply_counterfactual(
        self, b: dict[str, torch.Tensor], f_curr: int, cf: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Clones batch `b` and applies counterfactual edits (delete units, move units,
        scale HP/shield, or adjust player resources) at `f_curr`."""
        b = {k: v.clone() for k, v in b.items()}
        for row_idx in cf.get("deleted_rows", []):
            r = int(row_idx)
            if 0 <= r < b["mask"].shape[2]:
                b["alive"][0, f_curr:, r] = False
                b["mask"][0, f_curr:, r] = False

        for row_str, xy in cf.get("moved_rows", {}).items():
            r = int(row_str)
            if 0 <= r < b["x"].shape[2] and len(xy) == 2:
                dx = float(xy[0]) - float(b["x"][0, f_curr, r].item())
                dy = float(xy[1]) - float(b["y"][0, f_curr, r].item())
                b["x"][0, f_curr, r] = float(xy[0])
                b["y"][0, f_curr, r] = float(xy[1])
                # Also shift previous frame so velocity vx/vy doesn't spike artificially unless desired
                if not cf.get("keep_velocity_spike", False) and f_curr > 0:
                    b["x"][0, f_curr - 1, r] = b["x"][0, f_curr - 1, r] + dx
                    b["y"][0, f_curr - 1, r] = b["y"][0, f_curr - 1, r] + dy

        for row_str, hp_mult in cf.get("hp_scale", {}).items():
            r = int(row_str)
            if 0 <= r < b["num"].shape[2]:
                mult = max(0.0, min(1.0, float(hp_mult)))
                b["num"][0, f_curr, r, ED.I["hp"]] *= mult
                b["num"][0, f_curr, r, ED.I["shield"]] *= mult

        return b

    def _forward_and_analyze(
        self,
        b: dict[str, torch.Tensor],
        all_f: list[int],
        target_idx: int,
        full_analysis: bool = True,
    ) -> dict[str, Any]:
        """Executes `Frames` + `EnergyModel` while capturing all intermediate transformer
        layer states, exact 2D-RoPE spatial and 1D-RoPE temporal attention matrices,
        additive embedding norms, Gram matrices, SVD spectra, and end-to-end autograd
        marginal unit values and spatial force fields."""
        enc = self.enc_l1
        model = self.model_l1
        dev = self.device

        # Make (x, y) leaf tensors with requires_grad=True for end-to-end spatial force field
        b_x = b["x"].float().detach().requires_grad_(True)
        b_y = b["y"].float().detach().requires_grad_(True)
        b_mod = {**b, "x": b_x, "y": b_y}

        f_idx = torch.as_tensor(all_f, device=dev)
        B, F_total, R = b_mod["mask"].shape
        nf = len(all_f)

        q = enc.coords(b_mod, f_idx)  # [B, nf, R, 8]
        p = enc.ledger(b_mod, f_idx)  # [B, nf, 2, 2]

        rtype = b_mod["rtype"][:, f_idx].clamp(0, ED.N_RAW_TYPES - 1)
        owner = b_mod["owner"][:, None].expand(B, nf, R)
        owner = torch.where(owner == ED.NEUTRAL, E.GROUND, owner + 1)
        on_map = b_mod["mask"][:, f_idx] & (b_mod["inside"][:, f_idx] == 0)
        mask = on_map & b_mod["alive"][:, f_idx]

        num = b_mod["num"][:, f_idx]
        kt = enc.k_type[rtype]
        kt_comp = torch.sign(kt) * torch.log1p(kt.abs()) / 8.0

        abil = b_mod["slot_abil"][:, f_idx, :, 0]
        prod_abil = b_mod["slot_abil"][:, f_idx, :, PD.QUEUE]

        extras = num[..., enc.extras_at]
        extras = torch.sign(extras) * torch.log1p(extras.abs()) / 8.0
        ext_in = torch.cat([extras, kt_comp], dim=-1)

        xy_raw = torch.stack([b_mod["x"][:, f_idx], b_mod["y"][:, f_idx]], dim=-1).float()
        xy_ang = xy_raw.unsqueeze(-1) * enc.map_omega
        xy_fourier = torch.cat(
            [xy_ang.sin().flatten(-2), xy_ang.cos().flatten(-2)], dim=-1
        ).to(q.dtype)

        facing = num[..., ED.I["facing"]]
        ord_reg = b_mod["slot_reg"][:, f_idx, :, 0, :]
        kin_in = torch.cat(
            [q, xy_fourier, facing.sin().unsqueeze(-1), facing.cos().unsqueeze(-1), ord_reg],
            dim=-1,
        )

        center = (
            b_mod["center"][:, None, None, :].float()
            if "center" in b_mod
            else torch.full((B, 1, 1, 2), 100.0, device=dev)
        )
        pos_ent = xy_raw - center
        S = 0

        eff_tok, eff_mask, eff_xy = enc._embed_effects(b_mod, f_idx)
        g0, X = 3, eff_tok.shape[2]
        T_t = g0 + R
        T = T_t + X

        unit_c, ord_c, prod_c, _ = enc.nodes(rtype, abil, prod_abil)
        field_ent = enc.field(b_mod, f_idx, pos_ent)
        dt = ED._compute_dtype(enc.cls_tok)

        own_c = enc.owner_emb(owner).to(dt)
        kin_c = enc.kin_c(kin_in).to(dt)
        ext_c = enc.extras(ext_in).to(dt)

        c_ent = (unit_c + ord_c + prod_c + own_c + kin_c + ext_c + field_ent).to(dt)
        g_tok = enc._embed_globals(b_mod, f_idx).to(dt)
        x_tok = torch.cat([g_tok, c_ent, eff_tok.to(dt)], dim=2).reshape(
            B * nf, T, -1
        )

        valid_g = torch.ones(B, nf, g0, dtype=torch.bool, device=dev)
        valid_all = torch.cat([valid_g, mask, eff_mask], dim=2)
        valid_flat = valid_all.reshape(B * nf, 1, 1, T)

        t_valid = torch.cat([valid_g, mask], dim=2)
        t_mask = t_valid.transpose(1, 2).reshape(B * T_t, nf)
        tri = torch.tril(torch.ones(nf, nf, dtype=torch.bool, device=dev))
        eye = torch.eye(nf, dtype=torch.bool, device=dev)
        allow_t = ((tri[None] & t_mask[:, None, :]) | eye[None])[:, None]
        t_mask_3d = t_mask.unsqueeze(-1)

        pos_g = torch.zeros(B, nf, g0, 2, device=dev, dtype=torch.float32)
        pos_all = torch.cat([pos_g, pos_ent, eff_xy], dim=2)

        cs_2d = ED.cos_sin(enc.rope_2d.angles(pos_all.reshape(B * nf, T, 2)))
        h2 = cs_2d.shape[-2]
        p_t = (
            cs_2d[:, :T_t]
            .view(B, nf, T_t, h2, 2)
            .transpose(1, 2)
            .reshape(B * T_t, nf, h2, 2)
        )
        loops = b_mod["loops"][:, f_idx] - b_mod["loops"][:, f_idx[:1]]
        t_cs = ED.cos_sin(enc.rope_1d.angles(loops))
        cs_1d = (
            t_cs[:, None, :, :, :]
            .expand(B, T_t, nf, h2, 2)
            .reshape(B * T_t, nf, h2, 2)
        )

        # Capture layer trajectories & exact spatial/temporal attention at `target_idx`
        layer_states = [x_tok[target_idx].detach().float()]
        spatial_attns = []   # list of 6 tensors [H=4, T, T] at target_idx
        temporal_attns = []  # list of 6 tensors [T_t, H=4, nf] at target_idx
        spatial_deltas = []  # [6, T] L2 norm of spatial sub-block update
        temporal_deltas = [] # [6, T] L2 norm of temporal sub-block update

        for blk in enc.blocks:
            d_dim = x_tok.shape[2]
            sp, ffs = blk.space, blk.ff_space
            qs, ks, vs = ED._rotary_qkv(
                x_tok,
                blk.ns.weight,
                blk.ns.bias,
                sp.q.weight,
                sp.k.weight,
                sp.v.weight,
                cs_2d,
                cs_2d,
                cs_2d,
                sp.scale,
                sp.H,
                sp.KV,
                sp.dh,
            )
            if full_analysis:
                with torch.no_grad():
                    qt_s = qs[target_idx].float()
                    kt_s = ks[target_idx].float()
                    logits_s = torch.matmul(qt_s, kt_s.transpose(-1, -2)) / (sp.dh**0.5)
                    logits_s = logits_s.masked_fill(~valid_flat[target_idx], float("-inf"))
                    attn_s = torch.softmax(logits_s, dim=-1)  # [H, T, T]
                    spatial_attns.append(attn_s.detach())

            a_s = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=valid_flat)
            x_after_space = ED._compiled_out_ff(
                x_tok,
                a_s,
                cs_2d,
                sp.o.weight,
                ffs.norm.weight,
                ffs.norm.bias,
                ffs.w1.weight,
                ffs.w2.weight,
                ffs.w3.weight,
            )
            if full_analysis:
                with torch.no_grad():
                    spatial_deltas.append(
                        (x_after_space[target_idx].float() - x_tok[target_idx].float())
                        .norm(dim=-1)
                        .detach()
                    )

            x_p = x_after_space[:, :T_t]
            x_t = x_p.view(B, nf, T_t, d_dim).transpose(1, 2).reshape(B * T_t, nf, d_dim)
            tm, fft = blk.time, blk.ff_time
            qt, kt, vt = ED._rotary_qkv(
                x_t,
                blk.nt.weight,
                blk.nt.bias,
                tm.q.weight,
                tm.k.weight,
                tm.v.weight,
                cs_1d,
                cs_1d,
                p_t,
                tm.scale,
                tm.H,
                tm.KV,
                tm.dh,
            )
            if full_analysis:
                with torch.no_grad():
                    # Target frame query is index `target_idx` along nf
                    qt_t = qt[:, :, target_idx : target_idx + 1, :].float()
                    kt_t = kt.float()
                    logits_t = torch.matmul(qt_t, kt_t.transpose(-1, -2)) / (tm.dh**0.5)
                    logits_t = logits_t.masked_fill(
                        ~allow_t[:, :, target_idx : target_idx + 1, :], float("-inf")
                    )
                    attn_t = torch.softmax(logits_t, dim=-1).squeeze(2)  # [T_t, H, nf]
                    temporal_attns.append(attn_t.detach())

            a_t = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=allow_t)
            x_t_out = ED._compiled_out_ff_masked(
                x_t,
                a_t,
                p_t,
                t_mask_3d,
                tm.o.weight,
                fft.norm.weight,
                fft.norm.bias,
                fft.w1.weight,
                fft.w2.weight,
                fft.w3.weight,
            )
            x_p_out = x_t_out.view(B, T_t, nf, d_dim).transpose(1, 2).reshape(B * nf, T_t, d_dim)
            x_tok = torch.cat([x_p_out, x_after_space[:, T_t:]], dim=1)

            if full_analysis:
                with torch.no_grad():
                    temporal_deltas.append(
                        (x_tok[target_idx].float() - x_after_space[target_idx].float())
                        .norm(dim=-1)
                        .detach()
                    )
                    layer_states.append(x_tok[target_idx].detach().float())

        cls_out = x_tok[:, 0:1]
        ply_out = x_tok[:, 1:3]
        z_ply_flat = F.softsign(enc.z_ply_proj(enc.z_ply_norm(ply_out + cls_out)))

        z_flat, c_normed = enc.project_entities(x_tok[:, g0 : g0 + R], rtype, mask)

        # Control port u (post-command abilities at f_idx + 1)
        u_all, u_act_all = enc._encode_commands(
            b_mod, f_idx, abil, prod_abil, ord_reg, owner, mask
        )

        # Extract frame `t` (target_idx) and frame `t+1` (target_idx + 1)
        q_t = q[:, target_idx].float()              # [1, R, 8]
        z_t = z_flat[target_idx : target_idx + 1].float()  # [1, R, 8]
        zp_t = z_ply_flat[target_idx : target_idx + 1].float()  # [1, 2, 8]
        own_t = owner[:, target_idx]                # [1, R]
        msk_t = mask[:, target_idx]                 # [1, R]
        p_t_led = p[:, target_idx].float()          # [1, 2, 2]
        u_t = u_all[:, target_idx].float()          # [1, 2, u_dim]
        act_t = u_act_all[:, target_idx]            # [1, 2]

        # Introducing per-unit existence weight `unit_w = 1.0` (with requires_grad=True)
        # so d(-A_1)/d(unit_w_i) gives the exact marginal advantage of unit i!
        unit_w = torch.ones((1, R, 1), dtype=torch.float32, device=dev, requires_grad=True)
        q_weighted = q_t * unit_w
        z_weighted = z_t * unit_w

        x_obs_t, x_lat_t = E.build_gram_state(
            q_weighted, z_weighted, own_t, msk_t, p_t_led, zp_t, r=enc.r
        )
        x_t_full = torch.cat([x_obs_t, x_lat_t], dim=-1)  # [1, 2, 138]
        x1_t, x2_t = x_t_full[:, 0], x_t_full[:, 1]

        # Run Port-Hamiltonian dynamics on detached x1_t, x2_t so its internal
        # autograd.grad(..., retain_graph=False) does not free the encoder graph!
        x1_next_pred, x2_next_pred, pieces = model.dynamics(
            x1_t.detach(), x2_t.detach(), u_t[:, 0], u_t[:, 1], model.H, dt=1.0
        )

        # Next frame ground-truth state x_{t+1}
        with torch.no_grad():
            n_idx = min(nf - 1, target_idx + 1)
            x_obs_n, x_lat_n = E.build_gram_state(
                q[:, n_idx].float(),
                z_flat[n_idx : n_idx + 1].float(),
                owner[:, n_idx],
                mask[:, n_idx],
                p[:, n_idx].float(),
                z_ply_flat[n_idx : n_idx + 1].float(),
                r=enc.r,
            )
            x_next_true = torch.cat([x_obs_n, x_lat_n], dim=-1)

        # Goal cosine alignment at frame t (including symplectic pull from q to p)
        _, cos1 = model._goal_loss(pieces["g1_goal_eff"], pieces["bu1"], act_t[:, 0])
        _, cos2 = model._goal_loss(pieces["g2_goal_eff"], pieces["bu2"], act_t[:, 1])

        neg_A1_val = float(-pieces["A1"][0].item() * 1000.0)
        H1_val = float(-pieces["H1"][0].item() * 1000.0)
        H2_val = float(-pieces["H2"][0].item() * 1000.0)
        V1_val = float(-pieces["V1"][0].item() * 1000.0)
        V2_val = float(-pieces["V2"][0].item() * 1000.0)
        T1_val = float(pieces["T1"][0].item() * 1000.0)
        T2_val = float(pieces["T2"][0].item() * 1000.0)

        scalars = {
            "H1": round(H1_val, 4),
            "H2": round(H2_val, 4),
            "V1": round(V1_val, 4),
            "V2": round(V2_val, 4),
            "T1": round(T1_val, 4),
            "T2": round(T2_val, 4),
            "A1": round(neg_A1_val, 4),
            "raw_A1": round(float(pieces["A1"][0].item() * 1000.0), 4),
            "own1": round(float(-pieces["own1"][0].item() * 1000.0), 4),
            "own2": round(float(-pieces["own2"][0].item() * 1000.0), 4),
            "tug": round(float(-pieces["tug"][0].item() * 1000.0), 4),
            "kin1": round(float(pieces["kin1"][0].item() * 1000.0), 4),
            "kin2": round(float(pieces["kin2"][0].item() * 1000.0), 4),
            "diss1": round(float(pieces["diss1"][0].item() * 1000.0), 4),
            "diss2": round(float(pieces["diss2"][0].item() * 1000.0), 4),
            "goal_cos1": round(float(cos1.item()), 4),
            "goal_cos2": round(float(cos2.item()), 4),
            "u1_active": bool(act_t[0, 0].item()),
            "u2_active": bool(act_t[0, 1].item()),
            "u1_norm": round(float(u_t[0, 0].norm().item()), 4),
            "u2_norm": round(float(u_t[0, 1].norm().item()), 4),
        }

        if not full_analysis:
            return {"scalars": scalars}

        # Evaluate model.H(x1_t, x2_t) connected to the encoder graph (unit_w, b_x, b_y)
        h_graph = model.H(x1_t, x2_t)
        neg_A1 = -h_graph["A1"][0]
        t_tot = h_graph["T1"][0] + h_graph["T2"][0]

        # Fast kinetic gradient wrt q_t
        grad_q_kin = torch.autograd.grad(t_tot, q_t, retain_graph=True, allow_unused=True)[0]
        if grad_q_kin is not None:
            force_kin_x = (grad_q_kin[0, :, 1] / E.MAP_SCALE * 1000.0).detach().cpu().numpy()
            force_kin_y = (grad_q_kin[0, :, 2] / E.MAP_SCALE * 1000.0).detach().cpu().numpy()
        else:
            force_kin_x = np.zeros(R, dtype=np.float32)
            force_kin_y = np.zeros(R, dtype=np.float32)

        # Single end-to-end backward pass through all 6 transformer blocks (retain_graph=False)
        grad_w, grad_bx, grad_by = torch.autograd.grad(
            neg_A1,
            (unit_w, b_x, b_y),
            retain_graph=False,
        )
        marginal_adv_unit = (0.5 * grad_w[0, :, 0] * 1000.0).detach().cpu().numpy()
        force_adv_x = (grad_bx[0, all_f[target_idx]] * 1000.0).detach().cpu().numpy()
        force_adv_y = (grad_by[0, all_f[target_idx]] * 1000.0).detach().cpu().numpy()

        # Build detailed unit, token, Field/Mode, and Port-Hamiltonian flow payloads
        with torch.no_grad():
            gram_payload = self._build_gram_payload(
                q_t[0],
                z_t[0],
                c_normed[target_idx].float() / (256.0**0.5),
                own_t[0],
                msk_t[0],
                p_t_led[0],
                zp_t[0],
                x_obs_t[0],
                x_lat_t[0],
                pieces["g1_goal"][0],
                pieces["g2_goal"][0],
            )

            phd_payload = self._build_phd_payload(
                x1_t[0],
                x2_t[0],
                x_next_true[0, 0],
                x_next_true[0, 1],
                pieces,
            )

            units_payload, tokens_payload, attn_payload = self._build_tokens_and_units(
                b_mod,
                all_f[target_idx],
                target_idx,
                q_t[0],
                z_t[0],
                c_normed[target_idx].float() / (256.0**0.5),
                own_t[0],
                msk_t[0],
                eff_mask[0, target_idx],
                pos_all[0, target_idx],
                center[0, 0, 0],
                marginal_adv_unit,
                force_adv_x,
                force_adv_y,
                force_kin_x,
                force_kin_y,
                (
                    unit_c[0, target_idx],
                    ord_c[0, target_idx],
                    prod_c[0, target_idx],
                    own_c[0, target_idx],
                    kin_c[0, target_idx],
                    ext_c[0, target_idx],
                    field_ent[0, target_idx],
                ),
                layer_states,
                spatial_attns,
                temporal_attns,
                spatial_deltas,
                temporal_deltas,
                loops[0].cpu().numpy().tolist(),
            )

        return {
            "scalars": scalars,
            "gram": gram_payload,
            "phd": phd_payload,
            "units": units_payload,
            "tokens": tokens_payload,
            "attention": attn_payload,
        }

    def _build_gram_payload(
        self,
        q: torch.Tensor,         # [R, 8]
        z: torch.Tensor,         # [R, 8]
        c_norm: torch.Tensor,    # [R, 256]
        owner: torch.Tensor,     # [R]
        mask: torch.Tensor,      # [R]
        p_led: torch.Tensor,     # [2, 2]
        z_ply: torch.Tensor,     # [2, 8]
        x_obs: torch.Tensor,     # [2, 10]
        x_lat: torch.Tensor,     # [2, 140]
        g1_goal: torch.Tensor,   # [150]
        g2_goal: torch.Tensor,   # [150]
    ) -> dict[str, Any]:
        """Constructs 8x8 Continuous Field Overlap matrices (F_self, F_cross) and their
        Hamiltonian sensitivities, per-mode field/flux vectors (F_env, T_field, A_self,
        A_cross, z_ply), 10 grounded physical totals, and 256x256 SVD spectra."""
        r = z.shape[-1]
        labels_8 = [f"z{i}" for i in range(r)]
        labels_16 = list(E.Q_COLS) + labels_8
        yz = torch.cat([q, z], dim=-1)  # [R, 16]

        layout = E.gram_layout(r)
        ii_z, jj_z = E._triu_indices(r)
        n_self = layout["n_self"]    # 36
        n_cross = layout["n_cross"]  # 64
        n_env = layout["n_env"]      # 8
        n_ply = layout["n_ply"]      # 8
        n_q_lat = layout["n_q_lat"]  # 116

        def _unpack_self_8x8(vec36: np.ndarray, is_grad: bool = False) -> np.ndarray:
            mat = np.zeros((r, r), dtype=np.float32)
            for k in range(n_self):
                i, j = int(ii_z[k]), int(jj_z[k])
                val = float(vec36[k]) * (1.0 if (not is_grad or i == j) else 0.5)
                mat[i, j] = val
                mat[j, i] = val
            return np.round(mat, 4 if is_grad else 5)

        def _unpack_player_field(pl_idx: int, g_vec: torch.Tensor) -> dict[str, Any]:
            # Note: -A1 = H2 - H1, so d(-A1)/dx_p = -g_goal_p
            g_np = (-g_vec.detach().cpu().numpy()) * 1000.0
            lat_np = x_lat[pl_idx].detach().cpu().numpy()
            obs_np = x_obs[pl_idx].detach().cpu().numpy()
            g_obs_np = g_np[: E.N_OBS_P]
            g_lat_np = g_np[E.N_OBS_P :]

            f_self_8 = _unpack_self_8x8(lat_np[:n_self], is_grad=False)
            grad_self_8 = _unpack_self_8x8(g_lat_np[:n_self], is_grad=True)

            off_c = n_self
            f_cross_8 = np.round(lat_np[off_c : off_c + n_cross].reshape(r, r), 5)
            grad_cross_8 = np.round(g_lat_np[off_c : off_c + n_cross].reshape(r, r), 4)

            off_env = off_c + n_cross
            f_env_v = np.round(lat_np[off_env : off_env + n_env], 5).tolist()
            grad_env_v = np.round(g_lat_np[off_env : off_env + n_env], 4).tolist()

            off_ply = off_env + n_env
            z_ply_v = np.round(lat_np[off_ply : off_ply + n_ply], 4).tolist()
            grad_ply_v = np.round(g_lat_np[off_ply : off_ply + n_ply], 4).tolist()

            off_t = n_q_lat
            t_field_v = np.round(lat_np[off_t : off_t + r], 5).tolist()
            grad_t_v = np.round(g_lat_np[off_t : off_t + r], 4).tolist()

            off_as = off_t + r
            a_self_v = np.round(lat_np[off_as : off_as + r], 5).tolist()
            grad_as_v = np.round(g_lat_np[off_as : off_as + r], 4).tolist()

            off_ac = off_as + r
            a_cross_v = np.round(lat_np[off_ac : off_ac + r], 5).tolist()
            grad_ac_v = np.round(g_lat_np[off_ac : off_ac + r], 4).tolist()

            # Embed into (8+r)x(8+r) combined view (grounded totals on top-left diagonal, F_self on bottom-right)
            g_16 = np.zeros((8 + r, 8 + r), dtype=np.float32)
            grad_16 = np.zeros((8 + r, 8 + r), dtype=np.float32)
            q_to_obs = {0: 2, 3: 6, 4: 7, 5: 3, 6: 4, 7: 5}
            for q_i, obs_i in q_to_obs.items():
                g_16[q_i, q_i] = float(obs_np[obs_i])
                grad_16[q_i, q_i] = float(g_obs_np[obs_i])
            g_16[8:, 8:] = f_self_8
            grad_16[8:, 8:] = grad_self_8

            return {
                "F_self_8x8": f_self_8.tolist(),
                "grad_F_self_8x8": grad_self_8.tolist(),
                "F_cross_8x8": f_cross_8.tolist(),
                "grad_F_cross_8x8": grad_cross_8.tolist(),
                "G_16x16": np.round(g_16, 5).tolist(),
                "grad_G_16x16": np.round(grad_16, 4).tolist(),
                "obs_totals": np.round(obs_np, 5).tolist(),
                "grad_obs_totals": np.round(g_obs_np, 4).tolist(),
                "mode_vectors": {
                    "F_env": f_env_v,
                    "grad_F_env": grad_env_v,
                    "T_field": t_field_v,
                    "grad_T_field": grad_t_v,
                    "A_self": a_self_v,
                    "grad_A_self": grad_as_v,
                    "A_cross": a_cross_v,
                    "grad_A_cross": grad_ac_v,
                    "z_ply": z_ply_v,
                    "grad_z_ply": grad_ply_v,
                },
            }

        pf_p1 = _unpack_player_field(0, g1_goal)
        pf_p2 = _unpack_player_field(1, g2_goal)

        # Layer 2 learned strategic basis W_c [256, 16]
        w_c = F.normalize(self.model_l2.seq_enc.gram_op.w_c.detach(), dim=0) * (256.0 / 16.0) ** 0.25

        owners_out = {}
        for name, pl_id in (("p1", E.P1), ("p2", E.P2), ("neutral", E.GROUND)):
            m = mask & (owner == pl_id)
            idx_live = torch.nonzero(m).squeeze(-1)
            yz_p = yz[idx_live]  # [N_p, 16]

            # Full-rank 256x256 G_cc SVD via thin SVD of unit-normalized centered C_p / sqrt(N0)
            cp = c_norm[idx_live] / (E.N0**0.5)  # [N_p, 256]
            if len(idx_live) > 0:
                try:
                    cp_unit = F.normalize(c_norm[idx_live], dim=-1)
                    cp_cent = (cp_unit - cp_unit.mean(dim=0, keepdim=True)) if len(idx_live) > 1 else cp_unit
                    u_svd, s_svd, vt_svd = torch.linalg.svd(cp_cent, full_matrices=False)
                    eig_vals = s_svd.square() / E.N0
                    top_k = min(16, len(eig_vals))
                    sv_list = eig_vals[:top_k].cpu().numpy().round(5).tolist()
                    p_prob = eig_vals / eig_vals.sum().clamp(min=1e-12)
                    ent = -(p_prob * (p_prob.clamp(min=1e-12).log())).sum().item()
                    eff_rank = round(math.exp(ent), 2)
                    k8 = min(8, len(s_svd))
                    loadings = ((u_svd[:, :k8] * s_svd[:k8]) / 100.0).t().cpu().numpy().round(6).tolist()
                except Exception:
                    sv_list, eff_rank, loadings = [], 0.0, []
            else:
                sv_list, eff_rank, loadings = [], 0.0, []

            if len(idx_live) > 0:
                cp_w = cp @ w_c  # [N_p, 16]
                l2_cc_16 = torch.tanh(cp_w.t() @ cp_w).cpu().numpy().round(4).tolist()
                qp_s = q[idx_live] / (E.N0**0.5)
                l2_qc_8x16 = torch.tanh(qp_s.t() @ cp_w).cpu().numpy().round(4).tolist()
            else:
                l2_cc_16 = np.zeros((16, 16)).tolist()
                l2_qc_8x16 = np.zeros((8, 16)).tolist()

            if name == "p1":
                pf = pf_p1
            elif name == "p2":
                pf = pf_p2
            else:
                g_16_n = ((yz_p.t() @ yz_p) / E.N0).cpu().numpy().round(5) if len(idx_live) > 0 else np.zeros((8 + r, 8 + r))
                pf = {
                    "F_self_8x8": g_16_n[8:, 8:].tolist(),
                    "grad_F_self_8x8": np.zeros((r, r)).tolist(),
                    "F_cross_8x8": np.zeros((r, r)).tolist(),
                    "grad_F_cross_8x8": np.zeros((r, r)).tolist(),
                    "G_16x16": g_16_n.tolist(),
                    "grad_G_16x16": np.zeros((8 + r, 8 + r)).tolist(),
                    "obs_totals": [0.0] * E.N_OBS_P,
                    "grad_obs_totals": [0.0] * E.N_OBS_P,
                    "mode_vectors": {
                        "F_env": [0.0] * r,
                        "grad_F_env": [0.0] * r,
                        "T_field": [0.0] * r,
                        "grad_T_field": [0.0] * r,
                        "A_self": [0.0] * r,
                        "grad_A_self": [0.0] * r,
                        "A_cross": [0.0] * r,
                        "grad_A_cross": [0.0] * r,
                        "z_ply": [0.0] * r,
                        "grad_z_ply": [0.0] * r,
                    },
                }

            owners_out[name] = {
                "n_units": int(len(idx_live)),
                "unit_rows": idx_live.cpu().numpy().astype(int).tolist(),
                **pf,
                "svd_eigenvalues": sv_list,
                "eff_rank": eff_rank,
                "eigenmode_unit_loadings": loadings,
                "l2_cc_16x16": l2_cc_16,
                "l2_qc_8x16": l2_qc_8x16,
                "ledger": (
                    (p_led[pl_id - 1].cpu().numpy() * E.BANK_SCALE).round(1).tolist()
                    if pl_id in (E.P1, E.P2)
                    else [0.0, 0.0]
                ),
                "z_ply": (
                    z_ply[pl_id - 1].cpu().numpy().round(4).tolist()
                    if pl_id in (E.P1, E.P2)
                    else [0.0] * r
                ),
            }

        return {
            "labels_8": labels_8,
            "labels_16": labels_16,
            "obs_cols": list(E.OBS_COLS),
            "q_cols": list(E.Q_COLS),
            "owners": owners_out,
        }

    def _build_phd_payload(
        self,
        x1_t: torch.Tensor,
        x2_t: torch.Tensor,
        x1_next_true: torch.Tensor,
        x2_next_true: torch.Tensor,
        pieces: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Extracts Port-Hamiltonian flow decomposition:
            dx = J(x) grad H (conservative) - R(x) grad H (dissipative) + B(x) u (control)
        vs ground-truth dx = x_{t+1} - x_t across all 10 Grounded Physical Totals and
        key Continuous Spatial Field coordinates."""
        layout = E.gram_layout(self.model_l1.r)

        axes = [
            ("minerals (q)", E.OBS["minerals"]),
            ("vespene (q)", E.OBS["vespene"]),
            ("count (q)", E.OBS["count"]),
            ("hp (q)", E.OBS["hp"]),
            ("shield (q)", E.OBS["shield"]),
            ("energy (q)", E.OBS["energy"]),
            ("vel_x (p_lin)", E.OBS["vx"]),
            ("vel_y (p_lin)", E.OBS["vy"]),
            ("vx^2 (p_quad)", E.OBS["vx2"]),
            ("vy^2 (p_quad)", E.OBS["vy2"]),
            ("F_self[0,0] (q_lat)", layout["f_self_slice"][0]),
            ("F_cross[0,0] (q_lat)", layout["f_cross_slice"][0]),
            ("F_env[0] (q_lat)", layout["f_env_slice"][0]),
            ("T_field[0] (p_quad)", layout["t_field_slice"][0]),
            ("A_self[0] (p_lin)", layout["a_self_slice"][0]),
            ("A_cross[0] (p_lin)", layout["a_cross_slice"][0]),
        ]

        def _extract_player(
            xt: torch.Tensor,
            xn: torch.Tensor,
            j_vec: torch.Tensor,
            r_vec: torch.Tensor,
            bu_vec: torch.Tensor,
            dx_vec: torch.Tensor,
            g_adv: torch.Tensor,
        ):
            true_dx = (xn - xt).detach().cpu().numpy()
            j_np = j_vec.detach().cpu().numpy()
            neg_r_np = (-r_vec).detach().cpu().numpy()
            bu_np = bu_vec.detach().cpu().numpy()
            dx_np = dx_vec.detach().cpu().numpy()
            ga_np = (-g_adv).detach().cpu().numpy()

            rows = []
            for label, idx in axes:
                w = float(layout["w_obs"][idx].item()) if idx < E.N_OBS_P else 1.0
                rows.append(
                    {
                        "label": label,
                        "idx": idx,
                        "val": round(float(xt[idx].item()), 5),
                        "cons": round(float(j_np[idx] * w), 5),
                        "diss": round(float(neg_r_np[idx] * w), 5),
                        "ctrl": round(float(bu_np[idx] * w), 5),
                        "pred_dx": round(float(dx_np[idx] * w), 5),
                        "true_dx": round(float(true_dx[idx] * w), 5),
                        "grad_adv": round(float(ga_np[idx] * 1000.0), 4),
                    }
                )
            return rows

        return {
            "p1": _extract_player(
                x1_t,
                x1_next_true,
                pieces["j1"][0],
                pieces["r1"][0],
                pieces["bu1"][0],
                pieces["dx1"][0],
                pieces["g1_goal"][0],
            ),
            "p2": _extract_player(
                x2_t,
                x2_next_true,
                pieces["j2"][0],
                pieces["r2"][0],
                pieces["bu2"][0],
                pieces["dx2"][0],
                pieces["g2_goal"][0],
            ),
        }

    def _build_tokens_and_units(
        self,
        b: dict[str, torch.Tensor],
        f_curr: int,
        target_idx: int,
        q: torch.Tensor,             # [R, 8]
        z: torch.Tensor,             # [R, 8]
        c_norm: torch.Tensor,        # [R, 256]
        owner: torch.Tensor,         # [R]
        mask: torch.Tensor,          # [R]
        eff_mask: torch.Tensor,      # [X]
        pos_rel: torch.Tensor,       # [T, 2] relative to center
        center_xy: torch.Tensor,     # [2]
        marginal_adv: np.ndarray,    # [R]
        force_adv_x: np.ndarray,     # [R]
        force_adv_y: np.ndarray,     # [R]
        force_kin_x: np.ndarray,     # [R]
        force_kin_y: np.ndarray,     # [R]
        emb_parts_tuple: tuple[torch.Tensor, ...],
        layer_states: list[torch.Tensor],   # 7 tensors [T, 256]
        spatial_attns: list[torch.Tensor],  # 6 tensors [H=4, T, T]
        temporal_attns: list[torch.Tensor], # 6 tensors [T_t, H=4, nf]
        spatial_deltas: list[torch.Tensor], # 6 tensors [T]
        temporal_deltas: list[torch.Tensor],# 6 tensors [T]
        window_loops: list[int],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """Builds the active unit list, active transformer token catalog, per-layer
        representation trajectories, and compact spatial/temporal attention structures."""
        R = mask.shape[0]
        S = 0
        g0 = 3
        T_t = g0 + R
        cx, cy = float(center_xy[0].item()), float(center_xy[1].item())

        # Compute 3D PCA RGB across live entities for all 7 layers in one GPU batch
        live_rows = torch.nonzero(mask).squeeze(-1).cpu().numpy().astype(int).tolist()
        live_tok_indices = [g0 + r for r in live_rows]
        live_tok_t = torch.as_tensor(live_tok_indices, dtype=torch.long, device=self.device)

        st_stack = torch.stack(layer_states, dim=0)  # [7, T, 256]
        layer_rgb_by_row: dict[int, list[list[int]]] = {r: [] for r in live_rows}
        if len(live_rows) >= 3:
            v_live = st_stack[:, live_tok_t, :]  # [7, N_live, 256]
            v_cent = v_live - v_live.mean(dim=1, keepdim=True)
            try:
                _, _, vt = torch.linalg.svd(v_cent, full_matrices=False)  # vt: [7, K, 256]
                proj = torch.matmul(v_cent, vt[:, :3, :].transpose(-1, -2)).cpu().numpy()  # [7, N_live, 3]
                lo = np.percentile(proj, 5, axis=1, keepdims=True)
                hi = np.percentile(proj, 95, axis=1, keepdims=True)
                rgb_all = (np.clip((proj - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0) * 215.0 + 40.0).astype(int)
                for idx_i, r in enumerate(live_rows):
                    layer_rgb_by_row[r] = rgb_all[:, idx_i, :].tolist()
            except Exception:
                for r in live_rows:
                    layer_rgb_by_row[r] = [[160, 160, 160]] * 7
        else:
            for r in live_rows:
                layer_rgb_by_row[r] = [[160, 160, 160]] * 7

        # Additive input embedding component norms for live rows
        part_names = ("unit", "order", "prod", "owner", "kin", "extras", "field")
        part_norms = [p.detach().float().norm(dim=-1).cpu().numpy() for p in emb_parts_tuple]

        # Layer 2 learned basis W_c [256, 16] projections per unit
        w_c = F.normalize(self.model_l2.seq_enc.gram_op.w_c.detach(), dim=0)
        l2_proj_all = (c_norm @ w_c).detach().cpu().numpy()  # [R, 16]

        rtype_np = b["rtype"][0, f_curr].detach().cpu().numpy().astype(int)
        x_np = b["x"][0, f_curr].detach().cpu().numpy()
        y_np = b["y"][0, f_curr].detach().cpu().numpy()
        num_np = b["num"][0, f_curr].detach().cpu().numpy()
        abil_np = b["slot_abil"][0, f_curr, :, 0].detach().cpu().numpy().astype(int)
        prod_np = b["slot_abil"][0, f_curr, :, PD.QUEUE].detach().cpu().numpy().astype(int)
        q_np = q.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        own_np = owner.detach().cpu().numpy().astype(int)

        # Stack layer norms [7, T] and cosine similarity to final layer l=6 [7, T]
        st_norms = st_stack.norm(dim=-1).cpu().numpy()  # [7, T]
        final_st = st_stack[-1:]  # [1, T, 256]
        cos_to_final = F.cosine_similarity(st_stack, final_st, dim=-1).cpu().numpy()  # [7, T]
        sp_del_np = torch.stack(spatial_deltas, dim=0).cpu().numpy()   # [6, T]
        tm_del_np = torch.stack(temporal_deltas, dim=0).cpu().numpy()  # [6, T]

        units_out = []
        for r in live_rows:
            tok_i = g0 + r
            utype = int(rtype_np[r])
            um = self.unit_meta.get(
                utype,
                {
                    "name": f"Unit_{utype}",
                    "category": "other",
                    "structure": False,
                    "worker": False,
                    "minerals": 0,
                    "vespene": 0,
                    "food": 0.0,
                },
            )
            a_id = int(abil_np[r])
            p_id = int(prod_np[r])
            flying = bool(self.tables.k[utype, 5].item() > 0.5) if utype < len(self.tables.k) else False

            units_out.append(
                {
                    "row": r,
                    "tok_idx": tok_i,
                    "owner": int(own_np[r]),  # 1=P1, 2=P2, 0=Neutral
                    "rtype": utype,
                    "name": um["name"],
                    "category": um["category"],
                    "structure": um["structure"],
                    "worker": um["worker"],
                    "flying": flying,
                    "value": int(um["minerals"] + um["vespene"]),
                    "x": round(float(x_np[r]), 2),
                    "y": round(float(y_np[r]), 2),
                    "vx": round(float(q_np[r, 3] * E.SPEED_SCALE), 3),
                    "vy": round(float(q_np[r, 4] * E.SPEED_SCALE), 3),
                    "mass": round(float(np.linalg.norm(z_np[r])), 3),
                    "hp": round(float(num_np[r, ED.I["hp"]]), 1),
                    "hp_max": round(float(num_np[r, ED.I["hp_max"]]), 1),
                    "shield": round(float(num_np[r, ED.I["shield"]]), 1),
                    "shield_max": round(float(num_np[r, ED.I["shield_max"]]), 1),
                    "energy": round(float(num_np[r, ED.I["energy"]]), 1),
                    "contents": round(float(num_np[r, ED.I["contents"]]), 0),
                    "radius": round(max(0.35, float(num_np[r, ED.I["radius"]])), 2),
                    "facing": round(float(num_np[r, ED.I["facing"]]), 2),
                    "weapon_cd": round(float(num_np[r, ED.I["weapon_cooldown"]]), 1),
                    "order_abil": self.vocab_abil_name.get(a_id, f"Abil_{a_id}"),
                    "prod_abil": self.vocab_abil_name.get(p_id, f"Abil_{p_id}"),
                    "q": q_np[r].round(4).tolist(),
                    "z": z_np[r].round(4).tolist(),
                    "l2_w_proj": l2_proj_all[r].round(4).tolist(),
                    "marginal_adv": round(float(marginal_adv[r]), 4),
                    "force_adv": [round(float(force_adv_x[r]), 4), round(float(force_adv_y[r]), 4)],
                    "force_kin": [round(float(force_kin_x[r]), 4), round(float(force_kin_y[r]), 4)],
                    "emb_parts": {
                        k: round(float(part_norms[idx][r]), 3) for idx, k in enumerate(part_names)
                    },
                    "layer_norms": st_norms[:, tok_i].round(3).tolist(),
                    "layer_cos_final": cos_to_final[:, tok_i].round(4).tolist(),
                    "spatial_deltas": sp_del_np[:, tok_i].round(3).tolist(),
                    "temporal_deltas": tm_del_np[:, tok_i].round(3).tolist(),
                    "layer_pca_rgb": layer_rgb_by_row[r],
                }
            )

        # Build compact active token catalog:
        # [CLS (0), P1 (1), P2 (2)] + live units + 64 field anchors + active effects
        tokens_out = []
        active_tok_ids = [0, 1, 2]
        tokens_out.append({"tok_idx": 0, "kind": "global", "label": "[CLS]", "x": round(cx, 1), "y": round(cy, 1), "owner": 0})
        tokens_out.append({"tok_idx": 1, "kind": "global", "label": "[P1]", "x": round(cx, 1), "y": round(cy, 1), "owner": 1})
        tokens_out.append({"tok_idx": 2, "kind": "global", "label": "[P2]", "x": round(cx, 1), "y": round(cy, 1), "owner": 2})

        for u in units_out:
            active_tok_ids.append(u["tok_idx"])
            tokens_out.append(
                {
                    "tok_idx": u["tok_idx"],
                    "kind": "unit",
                    "row": u["row"],
                    "label": f"{u['name']} #{u['row']}",
                    "x": u["x"],
                    "y": u["y"],
                    "owner": u["owner"],
                }
            )

        pos_np = pos_rel.detach().cpu().numpy()
        for s_i in range(S):
            tok_i = g0 + R + s_i
            active_tok_ids.append(tok_i)
            tokens_out.append(
                {
                    "tok_idx": tok_i,
                    "kind": "field",
                    "anchor_idx": s_i,
                    "label": f"FieldAnchor ({s_i // 8},{s_i % 8})",
                    "x": round(cx + float(pos_np[tok_i, 0]), 2),
                    "y": round(cy + float(pos_np[tok_i, 1]), 2),
                    "owner": 0,
                }
            )

        eff_live = torch.nonzero(eff_mask).squeeze(-1).cpu().numpy().astype(int).tolist()
        for e_i in eff_live:
            tok_i = T_t + e_i
            active_tok_ids.append(tok_i)
            tokens_out.append(
                {
                    "tok_idx": tok_i,
                    "kind": "effect",
                    "label": f"Effect #{e_i}",
                    "x": round(cx + float(pos_np[tok_i, 0]), 2),
                    "y": round(cy + float(pos_np[tok_i, 1]), 2),
                    "owner": 0,
                }
            )

        # Cache full active attention sub-matrices in RAM for <1ms interactive queries
        act_t_tensor = torch.as_tensor(active_tok_ids, dtype=torch.long, device=self.device)
        tok_id_to_compact = {tid: idx for idx, tid in enumerate(active_tok_ids)}

        n_act = len(active_tok_ids)
        sp_cache = np.zeros((6, 5, n_act, n_act), dtype=np.float32)

        for l_idx in range(6):
            att_l = spatial_attns[l_idx][:, act_t_tensor][:, :, act_t_tensor]  # [4, n_act, n_act]
            att_mean = att_l.mean(dim=0, keepdim=True)  # [1, n_act, n_act]
            att_5 = torch.cat([att_mean, att_l], dim=0).cpu().numpy()  # [5, n_act, n_act]
            sp_cache[l_idx] = att_5

        # Temporal attention for active persistent tokens (< T_t)
        persist_compact = [tid for tid in active_tok_ids if tid < T_t]
        persist_t = torch.as_tensor(persist_compact, dtype=torch.long, device=self.device)
        # [6, len(persist), 4, nf]
        tm_cache = np.stack(
            [temporal_attns[l_idx][persist_t].cpu().numpy() for l_idx in range(6)],
            axis=0,
        )

        # Cache in memory for instant (<1ms) token/layer/head/Gram-256 drilldowns
        self._last_frame_cache = {
            "active_tok_ids": active_tok_ids,
            "tok_id_to_compact": tok_id_to_compact,
            "sp_cache": sp_cache,  # [6, 5, n_act, n_act] (0=mean, 1..4=H0..H3)
            "persist_tok_ids": persist_compact,
            "persist_to_idx": {tid: i for i, tid in enumerate(persist_compact)},
            "tm_cache": tm_cache,  # [6, n_persist, 4, nf]
            "window_loops": window_loops,
            "layer_states": [st[act_t_tensor].cpu().numpy() for st in layer_states],  # 7 x [n_act, 256]
            "layer_states_gpu": [st.detach() for st in layer_states],                 # 7 x [T, 256]
            "c_norm_np": c_norm.detach().cpu().numpy(),  # [R, 256]
            "q_np": q_np,                                # [R, 8]
            "z_np": z_np,                                # [R, r]
            "q_tensor": q.detach(),                      # [R, 8]
            "z_tensor": z.detach(),                      # [R, r]
            "owner_tensor": owner.detach(),              # [R]
            "mask_tensor": mask.detach(),                # [R]
            "own_np": own_np,                            # [R]
            "live_rows": live_rows,
            "units_out": units_out,
        }

        attn_summary = {
            "window_loops": window_loops,
            "global_temporal": {
                "cls": tm_cache[:, 0, :, :].mean(axis=1).round(4).tolist(),  # [6, nf]
                "p1": tm_cache[:, 1, :, :].mean(axis=1).round(4).tolist(),
                "p2": tm_cache[:, 2, :, :].mean(axis=1).round(4).tolist(),
            },
        }
        return units_out, tokens_out, attn_summary

    def query_token_details(
        self, tok_idx: int, layer: int = 5, head: int = 0
    ) -> dict[str, Any]:
        """Instant (<1ms) drilldown for any clicked token `tok_idx` from the cached frame:
        returns its full spatial attention distribution across all active tokens for the
        requested layer (0..5) and head (0=Mean, 1..4=H0..H3), its full causal temporal
        attention across all 6 layers & 4 heads, and its cosine similarity to all other
        active tokens at layer `layer`."""
        with self.lock:
            cache = self._last_frame_cache
            if not cache or tok_idx not in cache["tok_id_to_compact"]:
                return {"error": f"Token {tok_idx} not in active cache"}

            c_idx = cache["tok_id_to_compact"][tok_idx]
            active_ids = cache["active_tok_ids"]
            l_clamped = max(0, min(5, int(layer)))
            h_clamped = max(0, min(4, int(head)))

            # 1. Outgoing spatial attention weights (what `tok_idx` attends to)
            out_weights = cache["sp_cache"][l_clamped, h_clamped, c_idx]  # [n_act]
            # 2. Incoming spatial attention weights (who attends to `tok_idx`)
            in_weights = cache["sp_cache"][l_clamped, h_clamped, :, c_idx]  # [n_act]

            # Per-head breakdown for top 15 outgoing targets
            top_idx = np.argsort(-out_weights)[:25]
            outgoing = [
                {
                    "tok_idx": int(active_ids[i]),
                    "weight": round(float(out_weights[i]), 4),
                    "heads": cache["sp_cache"][l_clamped, 1:, c_idx, i].round(4).tolist(),
                }
                for i in top_idx
                if out_weights[i] > 0.002
            ]

            top_in_idx = np.argsort(-in_weights)[:25]
            incoming = [
                {
                    "tok_idx": int(active_ids[i]),
                    "weight": round(float(in_weights[i]), 4),
                }
                for i in top_in_idx
                if in_weights[i] > 0.002
            ]

            # 3. Cosine similarity between `tok_idx` and all other active tokens at `layer + 1`
            st_l = cache["layer_states"][l_clamped + 1]  # [n_act, 256]
            v_self = st_l[c_idx]
            norms = np.linalg.norm(st_l, axis=-1) * max(1e-8, float(np.linalg.norm(v_self)))
            cos_sim = (st_l @ v_self) / np.maximum(norms, 1e-8)
            cos_by_tok = {
                int(active_ids[i]): round(float(cos_sim[i]), 4) for i in range(len(active_ids))
            }

            # 4. Temporal attention [6, 4, nf]
            if tok_idx in cache["persist_to_idx"]:
                p_idx = cache["persist_to_idx"][tok_idx]
                temp_attn = cache["tm_cache"][:, p_idx, :, :].round(4).tolist()  # [6, 4, nf]
            else:
                temp_attn = []

            return {
                "tok_idx": tok_idx,
                "layer": l_clamped,
                "head": h_clamped,
                "outgoing": outgoing,
                "incoming": incoming,
                "cosine_similarity": cos_by_tok,
                "temporal_attn": temp_attn,
                "window_loops": cache["window_loops"],
            }

    def query_gram_256(
        self,
        owner: str = "p1",
        mat_type: str = "G_cc_256",
        cell_i: int = 0,
        cell_j: int = 0,
    ) -> dict[str, Any]:
        """Instant (<3ms) computation of full-rank 256x256 G_cc, Centered Covariance Cov(C_p),
        Clustered Covariance, 8x256 Cross-Gram G_qc, or N_live x N_live unit-to-unit
        Gram matrix K_uu = C C^T, plus interpretable feature correlations and signed dipole
        per-unit attribution for cell (cell_i, cell_j)."""
        from collections import Counter

        with self.lock:
            cache = self._last_frame_cache
            if not cache or "c_norm_np" not in cache:
                return {"error": "No cached frame available"}

            c_norm = cache["c_norm_np"]          # [R, 256] (unit L2 norm ~ 1)
            c_ln = c_norm * 16.0                 # [R, 256] (LayerNorm O(1) per channel)
            q_np = cache["q_np"]                 # [R, 8]
            own_np = cache["own_np"]             # [R]
            live_rows = cache["live_rows"]
            units_out = cache["units_out"]
            u_by_row = {u["row"]: u for u in units_out}

            def _rows_for(pl_code: int) -> list[int]:
                return [r for r in live_rows if int(own_np[r]) == pl_code]

            def _channel_correlations(vec: np.ndarray, rows_sub: list[int]) -> list[list[Any]]:
                if len(rows_sub) < 3 or float(np.std(vec)) < 1e-7:
                    return []
                us = [u_by_row[r] for r in rows_sub]
                cand: dict[str, np.ndarray] = {
                    "is_structure": np.array([1.0 if u["structure"] else 0.0 for u in us]),
                    "is_worker": np.array([1.0 if u["worker"] else 0.0 for u in us]),
                    "is_army": np.array([1.0 if u["category"] == "army" else 0.0 for u in us]),
                    "is_flying": np.array([1.0 if u["flying"] else 0.0 for u in us]),
                    "hp": np.array([float(u["hp"]) for u in us]),
                    "shield": np.array([float(u["shield"]) for u in us]),
                    "value": np.array([float(u["value"]) for u in us]),
                    "speed": np.array([float(np.hypot(u["vx"], u["vy"])) for u in us]),
                    "d(-A1)": np.array([float(u["marginal_adv"]) for u in us]),
                }
                for k in range(len(us[0]["z"])):
                    cand[f"z{k}"] = np.array([float(u["z"][k]) for u in us])
                for nm in set(u["name"] for u in us):
                    cand[f"type:{nm}"] = np.array([1.0 if u["name"] == nm else 0.0 for u in us])
                for ab in set(u["order_abil"].split("/")[-1] for u in us):
                    cand[f"ord:{ab}"] = np.array(
                        [1.0 if u["order_abil"].split("/")[-1] == ab else 0.0 for u in us]
                    )
                v_c = vec - vec.mean()
                v_norm = float(np.linalg.norm(v_c))
                if v_norm < 1e-7:
                    return []
                out_c = []
                for k_name, f_arr in cand.items():
                    f_c = f_arr - f_arr.mean()
                    f_norm = float(np.linalg.norm(f_c))
                    if f_norm < 1e-7:
                        continue
                    r_val = float(np.dot(v_c, f_c) / (v_norm * f_norm))
                    out_c.append([k_name, round(r_val, 3)])
                out_c.sort(key=lambda x: -abs(x[1]))
                return out_c[:5]

            if mat_type in ("K_uu", "K_spatial"):
                own_order = {1: 0, 2: 1, 0: 2}
                sorted_u = sorted(
                    units_out,
                    key=lambda u: (own_order.get(u["owner"], 3), u["category"], u["rtype"], u["row"]),
                )
                s_rows = [u["row"] for u in sorted_u]
                s_labels = [f"P{u['owner']} {u['name']} #{u['row']}" for u in sorted_u]
                s_owners = [u["owner"] for u in sorted_u]
                if len(s_rows) > 0:
                    if mat_type == "K_spatial":
                        with torch.no_grad():
                            k_vecs, w_vecs = E._get_wavevectors(torch.device("cpu"), torch.float32)
                            sqrt_w = w_vecs.sqrt()[None, :]
                            pos_s = torch.from_numpy(q_np[s_rows, 1:3]).float()
                            th = pos_s @ k_vecs.t()
                            psi_s = torch.cat([torch.cos(th) * sqrt_w, torch.sin(th) * sqrt_w], dim=-1)
                            k_uu = (psi_s @ psi_s.t()).numpy().round(4)
                    else:
                        c_mat = c_norm[s_rows]  # [N, 256]
                        c_unit = c_mat / np.maximum(np.linalg.norm(c_mat, axis=-1, keepdims=True), 1e-8)
                        k_uu = (c_unit @ c_unit.T).round(4)  # [N, N] true cosine in [-1, +1]
                    ci = max(0, min(len(s_rows) - 1, int(cell_i)))
                    cj = max(0, min(len(s_rows) - 1, int(cell_j)))
                    unit_scores = {
                        int(s_rows[idx]): round(float(k_uu[ci, idx]) / 100.0, 6)
                        for idx in range(len(s_rows))
                    }
                    mat_list = k_uu.tolist()
                    u_i = sorted_u[ci]
                    u_j = sorted_u[cj]
                    dist_ij = round(float(np.hypot(u_i["x"] - u_j["x"], u_i["y"] - u_j["y"])), 1)
                    cell_corrs = {
                        "row_label": s_labels[ci],
                        "col_label": s_labels[cj],
                        "row_corrs": [[f"Mass:{u_i.get('mass', 1.0):.2f}", 1.0], [f"Dist:{dist_ij}u", 1.0]],
                        "col_corrs": [[f"Mass:{u_j.get('mass', 1.0):.2f}", 1.0], [f"Cat:{u_j['category']}", 1.0]],
                        "pos_pole": f"Spatial/Latent Neighbor of {s_labels[ci]}",
                        "neg_pole": f"Distant / Opposite to {s_labels[ci]}",
                    }
                else:
                    mat_list, unit_scores, ci, cj, cell_corrs = [], {}, 0, 0, {}
                return {
                    "mat_type": mat_type,
                    "owner": owner,
                    "shape": [len(s_rows), len(s_rows)],
                    "matrix": mat_list,
                    "unit_labels": s_labels,
                    "unit_rows": s_rows,
                    "unit_owners": s_owners,
                    "cell": [ci, cj],
                    "unit_scores": unit_scores,
                    "cell_corrs": cell_corrs,
                }

            pl_code = 1 if owner == "p1" else (2 if owner == "p2" else 0)
            rp = live_rows if owner == "diff" else _rows_for(pl_code)

            if mat_type == "G_qc_8x256":
                ci = max(0, min(7, int(cell_i)))
                cj = max(0, min(255, int(cell_j)))
                if owner == "diff":
                    r1, r2 = _rows_for(1), _rows_for(2)
                    m1 = (q_np[r1].T @ c_ln[r1]) / E.N0 if r1 else np.zeros((8, 256))
                    m2 = (q_np[r2].T @ c_ln[r2]) / E.N0 if r2 else np.zeros((8, 256))
                    mat = (m1 - m2).round(4)
                else:
                    mat = ((q_np[rp].T @ c_ln[rp]) / E.N0).round(4) if rp else np.zeros((8, 256))
                unit_scores = {
                    int(r): round(float(q_np[r, ci] * c_ln[r, cj] / E.N0), 6)
                    for r in live_rows
                }
                q_name = E.Q_COLS[ci]
                cell_corrs = {
                    "row_label": f"Q[{q_name}]",
                    "col_label": f"c_{cj}",
                    "row_corrs": [[q_name, 1.0]],
                    "col_corrs": _channel_correlations(c_ln[rp, cj], rp) if rp else [],
                    "pos_pole": f"+{q_name} & +c_{cj}",
                    "neg_pole": f"+{q_name} & -c_{cj}",
                }
                return {
                    "mat_type": "G_qc_8x256",
                    "owner": owner,
                    "shape": [8, 256],
                    "matrix": mat.tolist(),
                    "cell": [ci, cj],
                    "unit_scores": unit_scores,
                    "cell_corrs": cell_corrs,
                }

            # 256x256 Modes: G_cc_active (W_z 8D Rowspace), G_cc_256 (Raw), G_cc_cov (Centered), G_cc_clustered
            ci = max(0, min(255, int(cell_i)))
            cj = max(0, min(255, int(cell_j)))
            centered = mat_type in ("G_cc_active", "G_cc_cov", "G_cc_clustered")

            # Unit-L2-normalized embeddings (scaled by 16 so channel std ~ 1) prevent
            # LayerNorm outlier-denominator imbalance where Structures have 5x the norm of Mobile units
            if mat_type == "G_cc_256":
                c_base = c_ln.copy()
            else:
                norms_r = np.maximum(np.linalg.norm(c_ln, axis=-1, keepdims=True), 1e-8)
                c_base = (c_ln / norms_r) * 16.0

            if mat_type == "G_cc_active":
                # Project out the 248D nullspace ker(W_z), keeping the 8D active subspace rowspace(W_z)
                w_z = self.enc_l1.z_proj.weight.detach().float().cpu().numpy()  # [8, 256]
                _, _, vt_z = np.linalg.svd(w_z, full_matrices=False)
                p_row = vt_z.T @ vt_z  # [256, 256]
                c_base = (c_base @ p_row) * (256.0 / 8.0) ** 0.5

            def _player_c(rows_p: list[int]) -> np.ndarray:
                if not rows_p:
                    return np.zeros((0, 256), dtype=np.float32)
                arr = c_base[rows_p]
                if centered and len(rows_p) > 1:
                    arr = arr - arr.mean(axis=0, keepdims=True)
                return arr

            if owner == "diff":
                r1, r2 = _rows_for(1), _rows_for(2)
                c1, c2 = _player_c(r1), _player_c(r2)
                m1 = (c1.T @ c1) / E.N0 if len(r1) else np.zeros((256, 256))
                m2 = (c2.T @ c2) / E.N0 if len(r2) else np.zeros((256, 256))
                mat_full = m1 - m2
            else:
                cp = _player_c(rp)
                mat_full = (cp.T @ cp) / E.N0 if len(rp) else np.zeros((256, 256))

            ch_order = list(range(256))
            if mat_type == "G_cc_clustered":
                diag = np.maximum(np.abs(np.diag(mat_full)), 1e-12)
                norms = np.sqrt(diag)
                corr = mat_full / (norms[:, None] * norms[None, :])
                visited = np.zeros(256, dtype=bool)
                order_idx = []
                curr = int(np.argmax(diag))
                for _ in range(256):
                    order_idx.append(curr)
                    visited[curr] = True
                    if len(order_idx) == 256:
                        break
                    sc = np.where(visited, -1e9, corr[curr] * (norms**0.25))
                    curr = int(np.argmax(sc))
                ch_order = order_idx
                mat_full = mat_full[np.ix_(ch_order, ch_order)]

            real_i = ch_order[ci]
            real_j = ch_order[cj]
            cell_val = float(mat_full[ci, cj])
            cell_sign = 1.0 if cell_val >= 0 else -1.0

            mu_by_own: dict[int, np.ndarray] = {}
            for pl_k in (1, 2, 0):
                rk = _rows_for(pl_k)
                if centered and len(rk) > 1:
                    mu_by_own[pl_k] = c_base[rk].mean(axis=0)
                else:
                    mu_by_own[pl_k] = np.zeros(256, dtype=np.float32)

            unit_scores = {}
            pos_names, neg_names = [], []
            for r in live_rows:
                pl_k = int(own_np[r])
                mu_k = mu_by_own.get(pl_k, np.zeros(256, dtype=np.float32))
                vi = float(c_base[r, real_i] - mu_k[real_i])
                vj = float(c_base[r, real_j] - mu_k[real_j])
                prod = vi * vj
                # Only units whose product (vi * vj) has the SAME sign as `cell_val` are driving this cell!
                if prod * cell_sign <= 0:
                    unit_scores[int(r)] = 0.0
                    continue
                mag = abs(prod) / E.N0
                sgn = 1.0 if vi >= 0 else -1.0
                unit_scores[int(r)] = round(sgn * mag, 6)
                if r in rp and abs(prod) > 0.25:
                    tag = f"{u_by_row[r]['name']}:{u_by_row[r]['order_abil'].split('/')[-1]}"
                    if sgn > 0:
                        pos_names.append(tag)
                    else:
                        neg_names.append(tag)

            pos_top = ", ".join(f"{k}({v})" for k, v in Counter(pos_names).most_common(3)) or "—"
            neg_top = ", ".join(f"{k}({v})" for k, v in Counter(neg_names).most_common(3)) or "—"
            vec_i = np.array([c_base[r, real_i] - mu_by_own[int(own_np[r])][real_i] for r in rp]) if rp else np.zeros(0)
            vec_j = np.array([c_base[r, real_j] - mu_by_own[int(own_np[r])][real_j] for r in rp]) if rp else np.zeros(0)

            cell_corrs = {
                "row_label": f"c_{real_i}" + (f" (pos #{ci})" if mat_type == "G_cc_clustered" else ""),
                "col_label": f"c_{real_j}" + (f" (pos #{cj})" if mat_type == "G_cc_clustered" else ""),
                "row_corrs": _channel_correlations(vec_i, rp),
                "col_corrs": _channel_correlations(vec_j, rp),
                "pos_pole": pos_top,
                "neg_pole": neg_top,
            }

            return {
                "mat_type": mat_type,
                "owner": owner,
                "shape": [256, 256],
                "matrix": mat_full.round(4).tolist(),
                "channel_order": ch_order,
                "cell": [ci, cj],
                "real_cell": [real_i, real_j],
                "unit_scores": unit_scores,
                "cell_corrs": cell_corrs,
            }

    def _extract_map_payload(
        self, g: PD.Game, b: dict[str, torch.Tensor], f_curr: int
    ) -> dict[str, Any]:
        """Extracts terrain height, pathability, buildability, visibility, creep, .s2ma
        NoFlyZones / HealingShrines, and evaluates `ContinuousPotentialField` on a dense
        64x64 grid over the playable map."""
        x0, y0, x1, y1 = [int(v) for v in g.playable]
        pw, ph = x1 - x0, y1 - y0
        cx, cy = float(g.center[0]), float(g.center[1])
        hx, hy = float(g.half[0]), float(g.half[1])

        # Static terrain height [ph, pw]
        h_grid = g.static_grid[:, :, 0].astype(np.uint8)
        # Live grid at f_curr [ph, pw, 10]: (vis, seen, creep, pathable, buildable) x 2 players
        live = b["grid"][0, f_curr].detach().cpu().numpy()

        # Downsample large grids by 2x if > 128 for crisp & fast JSON transport, or keep exact
        # Actually ph, pw is typically ~140x140 = 19,600 ints; let's downsample to 80x80 for
        # compact JSON (<40 KB) while preserving exact playable bounds!
        G_vis = 80
        ys = np.linspace(0, ph - 1, G_vis).astype(int)
        xs = np.linspace(0, pw - 1, G_vis).astype(int)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")

        height_ds = h_grid[yy, xx].tolist()
        path_ds = live[yy, xx, 3].astype(int).tolist()
        build_ds = live[yy, xx, 4].astype(int).tolist()
        creep_ds = np.maximum(live[yy, xx, 2], live[yy, xx, 7]).astype(int).tolist()
        p1_vis_ds = live[yy, xx, 0].astype(int).tolist()
        p2_vis_ds = live[yy, xx, 5].astype(int).tolist()

        # Evaluate ContinuousPotentialField AND painted 2D wave-packet fields on a 44x44 grid
        G_pot = 44
        r_modes = self.enc_l1.r
        sae_features = []
        sae_layer_r2 = []
        with torch.no_grad():
            px_lin = torch.linspace(-hx, hx, G_pot, device=self.device)
            py_lin = torch.linspace(-hy, hy, G_pot, device=self.device)
            gy, gx = torch.meshgrid(py_lin, px_lin, indexing="ij")
            pos_grid = torch.stack([gx, gy], dim=-1).reshape(1, 1, G_pot * G_pot, 2)
            f_idx_t = torch.tensor([f_curr], device=self.device)
            v_loc = self.enc_l1.field(b, f_idx_t, pos_grid)[0, 0].float()  # [G_pot*G_pot, 256]
            v_norm = v_loc.norm(dim=-1).reshape(G_pot, G_pot).cpu().numpy().round(2).tolist()
            # Project V_loc through z_norm + z_proj (sigmoid) to see the r tactical field modes!
            v_z = torch.sigmoid(self.enc_l1.z_proj(self.enc_l1.z_norm(v_loc.to(self.enc_l1.cls_tok.dtype))).float())
            v_z_modes = v_z.reshape(G_pot, G_pot, r_modes).permute(2, 0, 1).cpu().numpy().round(3).tolist()

            # Evaluate painted entity wave-packet fields F_1(x,y), F_2(x,y), F_env(x,y), Pi_p(x,y)
            cache = self._last_frame_cache
            if cache and "q_tensor" in cache:
                sf = E.evaluate_spatial_fields_grid(
                    cache["q_tensor"],
                    cache["z_tensor"],
                    cache["owner_tensor"],
                    cache["mask_tensor"],
                    gx,
                    gy,
                )
                f_diff_norm = sf["F_diff_norm"].cpu().numpy().round(4).tolist()
                f_contest = sf["F_contest"].cpu().numpy().round(4).tolist()
                t_density = sf["T_density"].cpu().numpy().round(4).tolist()
                f_p1_modes = sf["F_p1"].cpu().numpy().round(4).tolist()
                f_p2_modes = sf["F_p2"].cpu().numpy().round(4).tolist()
                f_env_modes = sf["F_env"].cpu().numpy().round(4).tolist()
                pi_p1_modes = sf["Pi_p1"].cpu().numpy().round(4).tolist()
                pi_p2_modes = sf["Pi_p2"].cpu().numpy().round(4).tolist()

                # Evaluate 7-Layer Post-Hoc Top-K SAE Stack if available
                self._maybe_load_sae()
                if self.sae_stack is not None and "layer_states_gpu" in cache:
                    sae_res = self.sae_stack.analyze_frame_entities(
                        cache["layer_states_gpu"],
                        cache["live_rows"],
                        3,
                        cache["q_tensor"],
                        cache["z_tensor"],
                        cache["owner_tensor"],
                        cache["mask_tensor"],
                        gx,
                        gy,
                        top_n_map_features=16,
                    )
                    sae_features = sae_res["active_map_features"]
                    sae_layer_r2 = sae_res["layer_r2"]
                    by_row = sae_res["unit_sae_by_row"]
                    for u_dict in cache["units_out"]:
                        u_dict["sae_layers"] = by_row.get(int(u_dict["row"]), [])
            else:
                z_grid = np.zeros((G_pot, G_pot)).tolist()
                f_diff_norm = z_grid
                f_contest = z_grid
                t_density = z_grid
                f_p1_modes = [z_grid] * r_modes
                f_p2_modes = [z_grid] * r_modes
                f_env_modes = [z_grid] * r_modes
                pi_p1_modes = [z_grid] * r_modes
                pi_p2_modes = [z_grid] * r_modes

        map_hash = g.meta.get("map_hash", "")
        map_name = g.meta.get("map", "")
        s2ma_info = self.s2ma_cache.get_map_objects(map_hash, map_name)

        return {
            "map_name": map_name,
            "map_size": [int(g.W), int(g.H)],
            "playable": [x0, y0, x1, y1],
            "center": [round(cx, 2), round(cy, 2)],
            "half": [round(hx, 2), round(hy, 2)],
            "grid_res": G_vis,
            "height": height_ds,
            "pathable": path_ds,
            "buildable": build_ds,
            "creep": creep_ds,
            "p1_vis": p1_vis_ds,
            "p2_vis": p2_vis_ds,
            "pot_res": G_pot,
            "pot_norm": v_norm,
            "pot_z_modes": v_z_modes,       # [r, G_pot, G_pot]
            "field_diff_norm": f_diff_norm, # [G_pot, G_pot] ||F_1|| - ||F_2||
            "field_contest": f_contest,     # [G_pot, G_pot] F_1 . F_2
            "field_t_density": t_density,   # [G_pot, G_pot] ||Pi_1||^2 + ||Pi_2||^2
            "field_p1_modes": f_p1_modes,   # [r, G_pot, G_pot]
            "field_p2_modes": f_p2_modes,   # [r, G_pot, G_pot]
            "field_env_modes": f_env_modes, # [r, G_pot, G_pot]
            "field_pi1_modes": pi_p1_modes, # [r, G_pot, G_pot]
            "field_pi2_modes": pi_p2_modes, # [r, G_pot, G_pot]
            "sae_features": sae_features,   # Top active L6 SAE features painted on 2D map
            "sae_layer_r2": sae_layer_r2,   # 7-layer SAE explained variance R^2
            "s2ma": s2ma_info,
        }
