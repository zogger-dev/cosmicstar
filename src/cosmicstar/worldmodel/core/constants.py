"""Shared StarCraft II engine schema constants and physical normalization scales."""

from __future__ import annotations

import numpy as np

# Engine player / team codes
N_CODES = 8  # team / player codes per game (max 8 players)
CODE_NEUTRAL = N_CODES
NEUTRAL_PLAYER = 16  # engine owner id of neutral units
N_RACES = 5

# Player-common vector columns (from pysc2 Observation.player_common)
PC_KEYS = (
    "minerals",
    "vespene",
    "food_cap",
    "food_used",
    "food_army",
    "food_workers",
    "idle_worker_count",
    "army_count",
    "warp_gate_count",
    "larva_count",
)
PC = {name: i for i, name in enumerate(PC_KEYS)}

# Engine unit / effect / buff / control-group schema capacities
GAS_TYPES = (20, 61, 88, 1943, 1980, 1981)  # refinery, assimilator, extractor (+ rich)
WORKER_TYPES = (45, 84, 104)  # SCV, probe, drone
GAS_GAP_LOOPS = 64  # longest out-of-sight gas harvest gap bridged (~2.9 s)
CLIFF_WALKERS = (49, 4)  # Reaper, Colossus
FLYING_BIT = 1
MULE_RTYPE = 268

N_RAW_TYPES = 4096
N_BUFF_IDS = 512
N_RAW_ABILITY_IDS = 4096
N_EFFECT_IDS = 64
C_GROUPS = 11  # control groups 0..9 + active selection (10)
UNKNOWN_ABILITY = 2**31 - 1

# Physical & tactical coordinate normalization scales (shared across L0 / L1 / L2)
GROUND, P1, P2 = 0, 1, 2
OWNERS = 3

Q_COLS = ("1", "x", "y", "vx", "vy", "hp", "shield", "energy")
Q = {name: i for i, name in enumerate(Q_COLS)}
K_Q = len(Q_COLS)  # 8

R_LATENT = 16
N0 = 100.0  # canonical reference unit count (100 units == 1.0)
BANK_SCALE = 1000.0  # mineral / vespene / HP / cost normalization
MAP_SCALE = 100.0  # physical map coordinate scale (100 map units == 1.0)
ENERGY_UNIT = 200.0  # spellcaster max energy normalization
FPS = 22.4  # Faster game loops per second
LOOPS_PER_SECOND = FPS
SPEED_SCALE = 5.0 / FPS  # canonical unit speed (5 map units/s) per Faster game loop
LOOPS_PER_SECOND_EXPORT = 16.0  # Editor XML export speed/accel timebase
INSTANT_ACCEL = 100.0
SPEED_AT, ACCEL_AT = 17, 18


def log_scale(x, scale: float = 10.0):
    """Smooth non-negative log1p scaling (`log(1 + max(x, 0)) / scale`)."""
    return np.log1p(np.maximum(x, 0)) / scale
