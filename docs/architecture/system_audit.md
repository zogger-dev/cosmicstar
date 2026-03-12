# System Audit & Architecture Review

## 1. Overview
**Date**: 2025-12-20
**Auditor**: Antigravity (Staff SWE)
**Scope**: `docs/research/policy_architecture.md`, `parse_replay.py`, `data_processor.py`, and general repository structure.

## 2. Research Plan Review (`policy_architecture.md`)
The ML Researcher's plan covers the AlphaStar architecture comprehensively.
*   **Strengths**: Clear separation of Torso, Core, and Head. Good alignment with the Nature 2019 paper (Transformer for units, ResNet for spatial).
*   **Gaps**:
    *   **Data Pipeline Integration**: The plan mentions JAX/XLA but the current codebase creates PyTorch `.pt` files. We need to clarify if the training loop will load PyTorch tensors into JAX (via DLPack or numpy) or if we should switch the data processor to save `.npy` / `tfrecord`.
    *   **Observation Space details**: "Screen 64x64 or 24x24" - needs standardization. The current code uses 84x84.

## 3. Codebase Audit

### 3.1. Replay Parsing (`parse_replay.py` & `data_processor.py`)
**Current Status**:
*   Scripts are monolithic and rely heavily on global state (`absl.flags`).
*   Significant code duplication between `parse_replay.py` (viewer) and `data_processor.py` (extractor).
*   Hardcoded paths and magic numbers (e.g., `controller.step(8)`, resolution 84x84) separate from any configuration source.

**Thread Safety & Concurrency Issues**:
1.  **Global Flags**: `absl.flags` are process-global. Importing these modules in a multi-threaded worker environment will cause collisions or inability to configure different replays for different threads purely via flags.
2.  **Process Management**: `run_configs.get().start()` launches a new SC2 process.
    *   *Risk*: If parallelized naively, multiple SC2 instances might try to bind to default Loopback ports (standard SC2 behavior). We need to explicitly manage port assignment for concurrent instances.
3.  **File I/O**: Direct `print` statements will cause interleaved garbage output in a parallel run.

### 3.2. Data Pipeline
*   **Format**: Saves List[Dict] via `torch.save`. This is decent for prototyping but inefficient for large-scale distributed training (loading full files at once).
*   **Versioning**: No metadata versioning in the saved files. If the extraction logic changes, old data becomes incompatible without warning.

## 4. Refactoring Recommendations (Review Mode)

### Refactor 1: Decouple Logic from Flags
**Goal**: Make replay parsing a pure function/class that can be called by a worker pool.
**Change**:
*   Create `cosmicstar/parsing/replay.py`.
*   Implement `ReplayParser` class.
*   Pass configuration (replay path, map path, resolution) via `__init__` or method arguments, NOT `FLAGS`.

### Refactor 2: Thread-Safe Port Management
**Goal**: Allow multiple SC2 instances to run in parallel.
**Change**:
*   Implement a `PortManager` or use `pysc2`'s existing helpers if available to assign unique ports (game port, base port) for each worker process.

### Refactor 3: Unify Feature Extraction
**Goal**: Single source of truth for observation generation.
**Change**:
*   Extract the observation processing (Steps "1. EXTRACT STATE" and "2. EXTRACT ACTIONS" in `data_processor.py`) into a shared utility `extract_features.py` or `feature_builder.py`.
*   Ensure both the debugger (`parse_replay.py`) and processor (`data_processor.py`) use this same builder.

### Refactor 4: Configuration Object
**Goal**: Centralize magic numbers.
**Change**:
*   Create `cosmicstar/config.py`.
*   Define `InterfaceConfig` (screen dims, minimap dims) and `GameLoopConfig` (step size).

## 5. Next Steps
1.  Execute Refactor 1 & 3 immediately to fix the "Critical" thread-safety blocker for data scaling.
2.  Update `data_processor.py` to use `multiprocessing` with the new refactored class.
