"""Canonical dataset, ruleset, vocabulary, asset, and checkpoint paths."""

from __future__ import annotations

import os

GAMES_DIR = "data/games_v3"
REPLAYS_DIR = "data/replays"
SPLITS_PATH = "data/games/splits/default.json"

VOCAB_PATH = "data/worldmodel/vocab.json"
LEGACY_VOCAB_PATH = "data/worldmodel/frame_cache_seq/vocab.json"

UNIT_DATA_PATH = "data/worldmodel/unit_data.json"
RULESETS_PATH = "data/worldmodel/rulesets.json"
RULES_DIR = "data/worldmodel/rules"
BALANCE_DIR = "data/worldmodel/balance"
ABILITY_LINKS_PATH = "data/worldmodel/ability_links.json"
ENERGY_TYPES_PATH = "data/worldmodel/energy_types.json"
GAMEDATA_DIR = "data/worldmodel/gamedata"
ASSETS_DIR = "data/worldmodel/assets"

DEFAULT_L0_CKPT = (
    "data/worldmodel/runs/physics_l0_tubelet_d256_4pillars_10k/model_step10000.pt"
)
DEFAULT_L1_CKPT = "data/worldmodel/runs/energy_r16_zerogram/model_step44285.pt"
DEFAULT_OBSERVER_CKPT = "data/worldmodel/runs/observer_v2/model_best.pt"
DEFAULT_SAE_CKPT = "data/worldmodel/sae/sae_7layer.pt"
DEFAULT_OBSERVER_CACHE = "data/worldmodel/observer_cache_v2"


def resolve_vocab_path(path: str | None = None) -> str:
    """Resolve `vocab.json`, falling back transparently if only the legacy path exists."""
    if path and os.path.exists(path):
        return path
    if os.path.exists(VOCAB_PATH):
        return VOCAB_PATH
    if os.path.exists(LEGACY_VOCAB_PATH):
        return LEGACY_VOCAB_PATH
    return path or VOCAB_PATH
