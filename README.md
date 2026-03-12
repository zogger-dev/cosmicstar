# CosmicStar: StarCraft II AlphaStar Implementation

## Ultimate Goal
To build a high-performance StarCraft II agent using Deep Reinforcement Learning and Imitation Learning, replicating the architecture of DeepMind's AlphaStar (Transformer + LSTM). The agent aims to play the full game (1v1) by learning from human replays and self-play.

## Current State (December 2025)
**Status:** Infrastructure & Data Pipeline Complete. Model Implementation In-Progress.

### ✅ Infrastructure Solved
*   **Modern Replay Support**: Patched `pysc2` to support StarCraft II v5.0.15+ (Builds 95841 & 95299).
*   **Integrated Downloader**: `scripts/download_maps.py` handles standard packs and extracts missing maps directly from Battle.net depots.
*   **Environment**: Validated `pysc2` launch and action execution.

### ✅ Data Pipeline
*   **Parser**: `cosmicstar.parsing.replay.ReplayParser` extracts rich observations and actions.
*   **Processor**: `data_processor.py` converts replays into PyTorch tensors.
*   **Package**: All core logic consolidated into the `cosmicstar/` package.

### 🚧 In Progress
*   **Model Architecture**: Currently designing the **ModelTorso** and **Heads** in `cosmicstar/models/` using JAX/Flax.

## Key Scripts
*   `scripts/download_maps.py`: **[Tool]** Sync maps from ladder or Battle.net depots.
*   `scripts/download_replays.py`: **[Tool]** Download replay packs.
*   `scripts/run_loop.py`: **[Env]** Validates installation by running an agent loop.
*   `data_processor.py`: **[Data]** Bulk processing of replays into `.pt` files.
*   `parse_replay.py`: **[Tool]** High-level entry point to test/debug replay parsing.

## Setup Requirements
*   StarCraft II (v5.0.15+)
*   Python 3.10+
*   Dependencies in `requirements.txt` (`pysc2`, `torch`, `jax`, `flax`, etc.)
