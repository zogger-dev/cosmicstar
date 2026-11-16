# CosmicStar World Model Architecture

## 1. Strict Acyclic Package Graph

All code under `src/cosmicstar/worldmodel/` follows an acyclic dependency hierarchy enforced by [`tests/test_architecture_imports.py`](../../../tests/test_architecture_imports.py):

- **`core/`** *(zero internal dependencies)*:
  - `paths.py`: canonical dataset, asset, and checkpoint locations (`resolve_vocab_path`).
  - `constants.py`: SC2 engine capacities, player-common keys (`PC_KEYS`), and physical normalization scales (`MAP_SCALE`, `SPEED_SCALE`, `BANK_SCALE`, `N0`, `Q_COLS`).
  - `vocab.py`: discrete concept vocabulary (`Vocab`).
  - `tables.py`: unified multi-patch static balance & unit metadata tables (`Tables`, `load_unit_metadata`, `build_unit_meta_tensors`, `EXCLUDE_FROM_CASUALTIES`).
  - `runtime.py`: deterministic Torch/Inductor/TF32 environment configuration, `_orig_mod` `state_dict` cleaning, atomic checkpoint saving, and host-RAM watchdog.
- **`nn/`** *(depends only on PyTorch)*:
  - `layers.py`: `mlp`, `norm_silu_mlp`, `small_init`, SwiGLU `FeedForward`, `CellNorm`, `ResBlock`, `MapFourier`, `MapDecoder`.
  - `rope.py`: 1D loop-time (`RoPE1D`) and 2D map-plane (`RoPE2D`, `apply_rope_2d`, `cos_sin`, `rotate`) rotary embeddings.
  - `attention.py`: grouped-query `Heads`, pointer-biased `PairAttend`, and analytical-gradient `FlexSpatial` / `Attend`.
- **`data/`** *(depends on `core/` and `cosmicstar.envs`)*:
  - `statefile.py`, `arrow_state.py`, `corpus.py`, `gamedata.py`: replay metadata, CASC XML catalogs, and binary/Arrow state stores.
  - `replay_actions.py`, `selection_tracker.py`, `command_join.py`, `action_tokens.py`: lossless interface action parsing, burst folding, selection tracking, and executed-command matching.
  - `game_store.py`: multi-view `Game` window reader, neutral reconciliation, gas/transport containment, `effect_ages`, `collate`, `trim_batch`, and `to_device`.
  - `splits.py` & `epochs.py`: `EXCLUDED` filter, patch/map split helpers, thread-safe LRU `Games` pool, and bucketed epoch window planner.
- **`l0_physics/`** *(depends on `core/`, `nn/`, `data/`)*:
  - `encoders.py`: entity, action, effect, and global token encoders + `5x5` 1-cell terrain stencil.
  - `trunk.py`: `Tower`, causal spatiotemporal `Block`, `PhysicsEncoder` (`k = 4` tubelet compression at `d_tube = 256`), and `pointer_pairs`.
  - `heads.py`: `RowHead` (`f_src` logit/stay residuals), `BirthHead`, `EffPersistHead`, `EffSpawnHead`, and `RecipientHead`.
  - `losses.py` & `model.py`: `Physics` orchestrator + Hungarian birth/AoE matching, snapshot-masked `REG` Huber losses, map residual decoding, and detached diagnostic probes.
- **`l1_tactical/`** *(depends on `core/`, `nn/`, `data/`)*:
  - `frames.py`: `Frames` (`step = 7` / `0.3125 s` perspective-isolated fog-safe entity & `ContinuousPotentialField` encoder).
  - `squad_pooler.py`: `PerceiverSquadCompressor` (`M = 64` smooth squad tokens + `M_opp = 32` perceived-enemy tokens) & `HindsightOutcomePosterior`.
  - `chunk_decoder.py`: `L1ChunkFlowDecoder` (`16`-frame parallel flow-matching micro-trajectory, order & spawn decoder).
  - `energy.py` & `sae.py`: Port-Hamiltonian `EnergyModel` (`Gram` phase space) and 7-layer post-hoc Top-K Sparse Autoencoder (`MultiLayerSAEStack`).
- **`l2_observer/`** *(depends on `core/`, `nn/`, `data/`, `l1_tactical/`)*:
  - `perception.py`: `StaticTerrainStencil`, squad visibility extraction, and `SightFootprintHead`.
  - `encoders.py`: 3-stream Theory-of-Mind (`PerspectiveMacroEncoder`, `OpponentBeliefHead`, `OmniscientObserverEncoder`).
  - `priors.py`: `ConditionalFlowPrior` (dual-mode CFG `Delta A` tactical intents) & `WorldOutcomeFlowPrior`.
  - `dynamics.py` & `model.py`: `SquadPortHamiltonian` and `ObserverWorldModelL2` training + closed-loop multi-branch `rollout_observer()`.
- **`inspector/`** *(composes `core/`, `data/`, `l0_physics/`, `l1_tactical/`, `l2_observer/`)*:
  - `service.py`, `observer_view.py`, `l1_frame.py`, `l1_gram_sae.py`, `replay_hud.py`, `map_s2ma.py`, `icons.py`, plus zero-build static UI under `ui/` (`index.html`, `styles.css`, `js/*.js`).

## 2. Historical Design Notes

- [`README.md`](README.md): empirical replay extraction notes, Editor balance export quirks, and selection-tracker measurements.
- [`REPLAY_EVENTS.md`](REPLAY_EVENTS.md): annotated catalog of raw `s2protocol` game events.
- [`DESIGN_physics.md`](DESIGN_physics.md) & [`DESIGN_energy.md`](DESIGN_energy.md): original design notes for `L0` physics and `L1` Port-Hamiltonian energy dynamics (superseded early drafts live in [`docs/archive/`](../../../docs/archive/)).
