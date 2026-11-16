# CosmicStar

A hierarchical learned world model of StarCraft II trained on ladder replays across multiple game builds (`5.0.14`–`5.0.17`), designed as the simulation and belief-state foundation for an AlphaStar-scale agent.

Detailed package boundaries, temporal scales, and module responsibilities are documented in [`src/cosmicstar/worldmodel/ARCHITECTURE.md`](src/cosmicstar/worldmodel/ARCHITECTURE.md).

## Multi-Scale World Model Stack

| Level | Package | Step / Horizon | Role |
|---|---|---|---|
| **`L0` Physics** | `cosmicstar.worldmodel.l0_physics` | `k = 4` loops (`0.18 s`) / `224`-loop windows (`10 s`) | Causal entity, terrain, pointer, birth, discrete AoE effect, visibility/creep, and economy simulator |
| **`L1` Tactical** | `cosmicstar.worldmodel.l1_tactical` | `step = 7` loops (`0.3125 s`) / `5.0 s` chunks | Perspective-isolated `Frames` encoder, `PerceiverSquadCompressor`, `HindsightOutcomePosterior`, `L1ChunkFlowDecoder`, plus `EnergyModel` & 7-layer Top-K `SAE` |
| **`L2` Observer** | `cosmicstar.worldmodel.l2_observer` | `112` loops (`5.0 s`) / `120 s` rollouts | Fog-of-war 3-stream Theory-of-Mind (`h_1, h_2, h_obs`), `OpponentBeliefHead`, conditional flow priors, and `SquadPortHamiltonian` rollouts |
| **Inspector** | `cosmicstar.worldmodel.inspector` | Real-time interactive GPU server | 2D battlefield canvas, broadcast HUD, counterfactual sandbox, SAE/Gram attribution, and multi-branch `L2` fan charts & squad threat maps |

## Repository Layout

| Path | Description |
|---|---|
| `src/cosmicstar/envs/` | Running the Windows SC2 client on Linux via Wine/Proton (`wine_run_config.py`), CDN build installer (`tact.py`), CASC storage reader (`casc.py`), and Battle.net depot (`depot.py`) |
| `src/cosmicstar/worldmodel/core/` | Canonical paths, SC2 engine constants & physical scales, `Vocab`, unified multi-patch `Tables` / unit metadata, and runtime/checkpoint utilities |
| `src/cosmicstar/worldmodel/nn/` | Reusable neural primitives (`RoPE1D`, `RoPE2D`, `Heads`, `FlexSpatial`, `PairAttend`, `FeedForward`, `ResBlock`, `MapFourier`, `MapDecoder`) |
| `src/cosmicstar/worldmodel/data/` | `.cst` / Arrow state stores, replay action & deterministic selection trackers, command matching, `Game` window reader & collator, and epoch/split samplers |
| `src/cosmicstar/worldmodel/{l0_physics,l1_tactical,l2_observer,inspector}/` | `L0`, `L1`, `L2` world model packages and interactive GPU web inspector (`ui/`) |
| `scripts/` | Active training, evaluation, and inspection entrypoints (`train_physics.py`, `train_observer.py`, `train_energy.py`, `train_sae.py`, `inspect_worldmodel.py`, `physics_baselines.py`) |
| `scripts/pipeline/` | One-time replay ingestion, engine state extraction, CASC GameData/balance export, ruleset graph building, and split generation utilities |
| `tests/` | Causal masking & gradient isolation (`test_physics_causality.py`), DAG architecture boundaries (`test_architecture_imports.py`), and checkpoint/Inspector smoke tests (`test_checkpoint_and_inspector_smoke.py`) |

`data/` (gitignored) stores replays, extracted Arrow/state views (`data/games_v3`), static ruleset catalogs, icons, and training runs.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -e . --all-extras
.venv/bin/python scripts/pipeline/regen_s2proto.py
```

## Common Commands

```bash
# Train L0 causal tubelet physics simulator
PYTHONPATH=src .venv/bin/python scripts/train_physics.py --seqs 6 --frames 224 --accum 4 --steps 10000

# Co-train L1 (Frames + Perceiver + Chunk Flow) and L2 (Stochastic Observer) across all 5 builds
PYTHONPATH=src .venv/bin/python scripts/train_observer.py --steps 4000 --batch 24 --live_batch 6

# Launch the interactive WorldModel Inspector web UI
PYTHONPATH=src .venv/bin/python scripts/inspect_worldmodel.py --port 8765

# Run architectural, causality, and checkpoint verification tests
PYTHONPATH=src .venv/bin/python tests/test_architecture_imports.py
PYTHONPATH=src .venv/bin/python tests/test_physics_causality.py
PYTHONPATH=src .venv/bin/python tests/test_checkpoint_and_inspector_smoke.py
```

## Running the SC2 Engine on Linux (Wine/Proton)

Blizzard's headless Linux builds stop at `4.10`; all `5.0.x` replays require the Windows client. `src/cosmicstar/envs/wine_run_config.py` registers a `pysc2` `RunConfig` (`WineLinux`) that launches `SC2_x64.exe` inside a Wine/GE-Proton prefix via `umu-run` (`$SC2PATH`), selecting the exact engine build per replay (`python -m cosmicstar.envs.tact <build>` installs older builds from Blizzard's CDN).
