"""Top-level stateful GPU `WorldModelInspector` service composing L1, SAE, L2, and HUD views."""

from __future__ import annotations

import glob
import json
import logging
import os
import threading
from typing import Any

import torch

from ..core import paths, runtime
from ..core.tables import (
    EXCLUDE_FROM_CASUALTIES,
    Tables,
    ZERG_BUILD_COSTS,
    load_ability_name_maps,
    load_unit_metadata,
)
from ..core.vocab import Vocab
from ..data import statefile
from ..data.action_tokens import ActionVocab
from ..data.game_store import Game, base_paths
from ..data.splits import EXCLUDED, of_patch
from ..l1_tactical import (
    D_SQUAD_DEFAULT,
    EnergyModel,
    Frames,
    HindsightOutcomePosterior,
    L1ChunkFlowDecoder,
    MultiLayerSAEStack,
    PerceiverSquadCompressor,
)
from ..l2_observer import ObserverWorldModelL2
from .l1_frame import L1FrameMixin
from .l1_gram_sae import L1GramSaeMixin
from .map_s2ma import S2MAMapCache, clean_player_name
from .observer_view import ObserverViewMixin
from .replay_hud import ReplayHudMixin

logger = logging.getLogger(__name__)


class WorldModelInspector(
    ObserverViewMixin,
    L1FrameMixin,
    L1GramSaeMixin,
    ReplayHudMixin,
):
    """Stateful GPU inspector engine for `L1` (`Frames` + `EnergyModel` + `SAE`) and
    `L2` (`ObserverWorldModelL2` + `L1ChunkFlowDecoder`) over StarCraft II replays."""

    def __init__(
        self,
        games_dir: str = paths.GAMES_DIR,
        vocab_path: str = paths.VOCAB_PATH,
        split_path: str = paths.SPLITS_PATH,
        unit_data_path: str = paths.UNIT_DATA_PATH,
        rulesets_path: str = paths.RULESETS_PATH,
        l1_ckpt: str = paths.DEFAULT_L1_CKPT,
        observer_ckpt: str = paths.DEFAULT_OBSERVER_CKPT,
        patch: str = "",
        base_build: int = 94137,
        device: str = "cuda",
    ):
        self.lock = threading.Lock()
        self.device = device
        self.games_dir = games_dir
        self.patch = patch

        self.vocab = Vocab.load(vocab_path)
        self.avocab = ActionVocab()
        self.tables = Tables(base_build, unit_data=unit_data_path, rulesets=rulesets_path)
        self.s2ma_cache = S2MAMapCache()

        # Unified unit & ability metadata from core.tables
        self.unit_meta: dict[int, dict[str, Any]] = load_unit_metadata(unit_data_path)
        self._unit_costs: dict[str, tuple[int, int]] = {
            u["name"].lower(): (int(u.get("minerals", 0)), int(u.get("vespene", 0)))
            for u in self.unit_meta.values()
            if "name" in u
        }
        self._replay_path_map: dict[str, str] = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(f"{paths.REPLAYS_DIR}/**/*.SC2Replay", recursive=True)
        }
        self.exclude_from_dead: frozenset[str] = EXCLUDE_FROM_CASUALTIES
        self._structure_names: set[str] = {
            u["name"].lower()
            for u in self.unit_meta.values()
            if u.get("structure") and "name" in u
        }
        self.zerg_build_costs: dict[str, tuple[int, int]] = dict(ZERG_BUILD_COSTS)
        self.raw_abil_name, self.vocab_abil_name = load_ability_name_maps(
            self.vocab, self.tables.ruleset, rulesets_path=rulesets_path
        )

        # Discover extracted games across all 5 SC2 builds (~932 games)
        with open(split_path) as f:
            split = json.load(f)
        unseen_ids = set(split["held_out_games"]) - set(EXCLUDED)
        map_val_ids = set(split.get("held_out_map_games", [])) - set(EXCLUDED)
        all_found = {
            k: v for k, v in base_paths(games_dir, None).items() if k not in set(EXCLUDED)
        }
        if patch:
            all_found = of_patch(all_found, patch)
        self.unseen_bases = {k: all_found[k] for k in unseen_ids if k in all_found}
        self.map_val_bases = {k: all_found[k] for k in map_val_ids if k in all_found}
        self.train_bases = {
            k: v
            for k, v in all_found.items()
            if k not in self.unseen_bases and k not in self.map_val_bases
        }
        self.all_bases = {**self.unseen_bases, **self.map_val_bases, **self.train_bases}
        self.game_subset = {
            **{gid: "unseen_games" for gid in self.unseen_bases},
            **{gid: "held_out_map_games" for gid in self.map_val_bases},
            **{gid: "train" for gid in self.train_bases},
        }

        self.catalog: list[dict[str, Any]] = []
        self.catalog_by_id: dict[str, dict[str, Any]] = {}
        self._build_catalog()

        self._open_games: dict[str, Game] = {}

        # Layer 1 Energy & Frames inspector stack
        self.model_l1 = EnergyModel(256, 32, 256).to(device).eval()
        self.enc_l1 = Frames(
            self.vocab, self.tables, 256, 32, attn_layers=6, dh=64, ff=768
        ).to(device).eval()
        self.enc_l1.grad_ckpt = False

        # Multi-Scale Stochastic Observer v2 stack
        self.obs_enc = Frames(
            self.vocab, self.tables, 256, 32, attn_layers=6, dh=64, ff=768
        ).to(device).eval()
        self.obs_enc.grad_ckpt = False
        self.obs_pooler = PerceiverSquadCompressor(self.tables, d_l1=256).to(device).eval()
        self.obs_posterior = HindsightOutcomePosterior().to(device).eval()
        self.obs_decoder = L1ChunkFlowDecoder(d_l1=256, d_squad=D_SQUAD_DEFAULT).to(device).eval()
        self.obs_l2 = ObserverWorldModelL2(rollout_steps=4).to(device).eval()
        self._obs_game_cache: dict[str, dict[str, Any]] = {}
        self.observer_ckpt_default = observer_ckpt
        self.active_observer_ckpt = ""
        self._observer_mtime = 0.0

        self._last_frame_cache: dict[str, Any] = {}
        self._cst_cache: dict[str, Any] = {}

        self.active_l1_ckpt = ""
        if os.path.exists(l1_ckpt):
            self.load_l1_checkpoint(l1_ckpt)
            self.obs_enc.load_state_dict(self.enc_l1.state_dict(), strict=False)
        self._maybe_reload_observer(observer_ckpt)

        self.sae_path = paths.DEFAULT_SAE_CKPT
        self.sae_mtime = 0.0
        self.sae_stack: MultiLayerSAEStack | None = None
        self._maybe_load_sae()

    def _maybe_load_sae(self) -> None:
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

    def _build_catalog(self) -> None:
        race_short = {"Terran": "T", "Zerg": "Z", "Protoss": "P", "Random": "R"}
        items = []
        for gid, base in self.all_bases.items():
            try:
                cst = statefile.load(f"{base}.p1.cst")
                meta = json.loads(str(cst["meta"]))
                players = meta.get("players", [])
                p1 = players[0] if len(players) > 0 else {}
                p2 = players[1] if len(players) > 1 else {}
                r1 = race_short.get(p1.get("race", ""), "?")
                r2 = race_short.get(p2.get("race", ""), "?")
                n1 = clean_player_name(p1.get("name", "Player 1")) or "Player 1"
                n2 = clean_player_name(p2.get("name", "Player 2")) or "Player 2"
                dur = int(meta.get("duration_loops", 0))
                map_name = meta.get("map_name", os.path.basename(os.path.dirname(base)))
                item = {
                    "gid": gid,
                    "subset": self.game_subset.get(gid, "train"),
                    "map_name": map_name,
                    "matchup": f"{r1}v{r2}",
                    "p1_name": n1,
                    "p2_name": n2,
                    "p1_race": p1.get("race", ""),
                    "p2_race": p2.get("race", ""),
                    "p1_mmr": p1.get("mmr", 0),
                    "p2_mmr": p2.get("mmr", 0),
                    "p1_result": p1.get("result", ""),
                    "p2_result": p2.get("result", ""),
                    "duration_loops": dur,
                    "duration_sec": round(dur / 22.4, 1),
                }
                items.append(item)
                self.catalog_by_id[gid] = item
            except Exception:
                continue
        order = {"unseen_games": 0, "held_out_map_games": 1, "train": 2}
        items.sort(
            key=lambda d: (
                order.get(d["subset"], 9),
                -max(d.get("p1_mmr", 0) or 0, d.get("p2_mmr", 0) or 0),
                d["gid"],
            )
        )
        self.catalog = items

    def list_checkpoints(self) -> dict[str, list[str]]:
        l1_ckpts = sorted(
            glob.glob("data/worldmodel/runs/energy_*/model_step*.pt"),
            key=os.path.getmtime,
            reverse=True,
        )
        obs_ckpts = sorted(
            glob.glob("data/worldmodel/runs/observer_v2*/model_*.pt"),
            key=os.path.getmtime,
            reverse=True,
        )
        seen_obs = []
        for p in ([self.active_observer_ckpt] if self.active_observer_ckpt else []) + obs_ckpts:
            if p and os.path.exists(p) and p not in seen_obs:
                seen_obs.append(p)
        return {"l1": l1_ckpts[:25], "observer": seen_obs[:10]}

    def _maybe_reload_observer(self, preferred_path: str | None = None) -> None:
        candidates = []
        if preferred_path:
            candidates.append(preferred_path)
        if self.observer_ckpt_default:
            candidates.append(self.observer_ckpt_default)
            candidates.append(
                os.path.join(os.path.dirname(self.observer_ckpt_default), "model_latest.pt")
            )
        chosen = next((p for p in candidates if p and os.path.exists(p)), None)
        if not chosen:
            return
        try:
            mt = os.path.getmtime(chosen)
            if chosen == self.active_observer_ckpt and mt <= self._observer_mtime:
                return
            self.load_observer_checkpoint(chosen)
        except Exception as e:
            logger.warning("Could not load observer checkpoint %s: %s", chosen, e)

    def load_observer_checkpoint(self, path: str) -> None:
        with self.lock:
            ck = torch.load(path, map_location=self.device, weights_only=False)
            for key, mod in (
                ("enc", self.obs_enc),
                ("pooler", self.obs_pooler),
                ("posterior", self.obs_posterior),
                ("decoder", self.obs_decoder),
                ("obs_l2", self.obs_l2),
            ):
                if key in ck:
                    mod.load_state_dict(runtime.clean_state_dict(ck[key]), strict=False)
                mod.eval()
            self.active_observer_ckpt = path
            self._observer_mtime = os.path.getmtime(path)
            self._obs_game_cache.clear()

    def load_l1_checkpoint(self, path: str) -> None:
        with self.lock:
            ck = torch.load(path, map_location=self.device, weights_only=False)
            m_sd = self.model_l1.state_dict()
            ck_m = {
                k: v for k, v in runtime.clean_state_dict(ck["model"]).items()
                if k in m_sd and m_sd[k].shape == v.shape
            }
            self.model_l1.load_state_dict(ck_m, strict=False)
            e_sd = self.enc_l1.state_dict()
            ck_e = {
                k: v for k, v in runtime.clean_state_dict(ck["enc"]).items()
                if k in e_sd and e_sd[k].shape == v.shape
            }
            self.enc_l1.load_state_dict(ck_e, strict=False)
            self.model_l1.eval()
            self.enc_l1.eval()
            self.active_l1_ckpt = path
            self._last_frame_cache.clear()

    def _get_game(self, gid: str) -> Game:
        if gid not in self._open_games:
            if len(self._open_games) >= 6:
                oldest = next(iter(self._open_games))
                del self._open_games[oldest]
            base = self.all_bases[gid]
            g = Game(base, self.vocab)
            g._load_actions(self.avocab)
            self._open_games[gid] = g
        return self._open_games[gid]
