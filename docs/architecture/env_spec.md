# CosmicStar Environment Specification

This document defines the interface between the StarCraft II environment (pysc2) and the CosmicStar agent. It details the observation tensors, action space, and reward signals.

## 1. Observation Space

The agent receives a dictionary of tensors at each timestep.

### Scalar Features
Global economy and game state information.
- **Shape**: `[Batch, 15]`
- **Features**:
    - Player Id
    - Minerals
    - Vespene
    - Food Used
    - Food Cap
    - Food Army
    - Food Workers
    - Idle Worker Count
    - Army Count
    - Warp Gate Count
    - Larva Count
    - Game Loop (normalized)
    - Score (normalized)
    - ... (Other scalar stats)

### Entity Features (Units)
List of all visible units (allies, enemies, neutral).
- **Shape**: `[Batch, Max_Units, Feature_Dim]` (e.g., `[B, 512, 18]`)
- **Feature_Dim Includes**:
    - Unit Type ID (one-hot or embedding index)
    - Owner (Self, Enemy, Neutral)
    - Health / Max Health
    - Shield / Max Shield
    - Energy / Max Energy
    - X, Y Position (normalized)
    - Facing Angle
    - Is Flying
    - Is Cloaked
    - Is Powered
    - Is Selected
    - Order ID (current action)
    - Buff IDs

### Spatial Features (Minimap/Screen)
Grid-based features from feature layers (optional, focusing on Entity approach).
- **Shape**: `[Batch, Channels, Height, Width]`
- **Channels**: HeightMap, Visibility, Creep, Camera, etc.

### Available Actions (Mask)
Binary mask indicating which action functions are valid in the current state.
- **Shape**: `[Batch, Total_Action_Functions]` (Size: ~573 for standard pysc2)
- **Values**: 1 if available, 0 if not.

## 2. Action Space

The action is a structured argument policy.

### Function ID
The main action identifier (e.g., Move, Attack, Build_Marine).
- **Shape**: `[Batch]`
- **Discrete**: `[0, Total_Action_Functions - 1]`

### Arguments
Depending on the Function ID, specific arguments are required.
- **Delay**: `[Batch]` (Discrete delay steps)
- **Queued**: `[Batch]` (Binary: Queue action or Now)
- **Units**: `[Batch, Max_Selected]` (Indices into Entity Features for selection)
- **Target Unit**: `[Batch]` (Index into Entity Features for targeting)
- **Target Point**: `[Batch, 2]` (X, Y coordinates)

## 3. Action Masking

To ensure the agent only outputs valid actions, we apply masking:
1.  **Function Mask**: Multiplied with the logits of the Function ID head. Derived from `pysc2`'s `available_actions`.
2.  **Argument Masking**: (Advanced) Masking invalid targets or units based on the selected function (e.g., can't target an enemy for "Heal" if "Heal" is only for allies).

## 4. Reward Shaping

The environment provides a scalar reward at each step.
- **Base Reward**: Win/Loss/Tie (+1, -1, 0) at terminal step.
- **Auxiliary Rewards** (shaped):
    - **Resource Collection**: +0.001 per mineral/gas mined (encourages expansion).
    - **Unit Production**: +0.01 per supply created (encourages macro).
    - **Combat**: +0.01 * (Enemy Resources Destroyed) - 0.01 * (Own Resources Lost).
    - **Map Control**: +0.001 for exploring new areas (fog of war lifting).

## 5. Implementation Details

- **Wrapper Class**: `CosmicEnvWrapper` (wraps `sc2_env.SC2Env`)
- **Bridge**: Converts `pysc2` Timestep -> `TensorDict` and `AgentAction` -> `pysc2.actions.FunctionCall`.
