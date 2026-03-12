# Model Design: Entity-Based Architecture

To mimic AlphaStar's performance, we move beyond simple screen screenshots and use a structured **Entity-Based** approach.

## 1. Inputs (Observations)

### A. Scalar Inputs (Global State)
*Shape: (Batch, N_Scalars)*
A vector of global game statistics.
- **Resources**: Minerals, Vespene Gas.
- **Supply**: Supply Used, Supply Cap, Supply Army, Supply Workers.
- **Time**: Game Loop (normalized).
- **Player**: Race (One-hot).

### B. Entity Inputs (Unit List)
*Shape: (Batch, Max_Units, N_Unit_Features)*
A list of all visible units (friendly and enemy). We pad this to a fixed maximum (e.g., 512 units) and mask the rest.
**Features per Unit:**
1.  **Unit Type**: One-hot (e.g., Marine, Zergling).
2.  **Alliance**: One-hot (Self, Ally, Neutral, Enemy).
3.  **Health**: Current HP / Max HP.
4.  **Shields**: Current Shields / Max Shields.
5.  **Energy**: Current Energy / Max Energy.
6.  **Position**: X, Y coordinates (normalized 0-1).
7.  **Build Progress**: 0-1 (if building).
8.  **Is Powered**: Boolean (for Protoss).

### C. Spatial Inputs (Minimap)
*Shape: (Batch, Channels, Height, Width)*
Since `sc2reader` doesn't render frames, we might initially skip this or reconstruct a basic 2D grid from Unit positions for spatial context.
- **Channels**: HeightMap, Visibility, Creep, Camera Position.

---

## 2. Transformer Torso (The "Brain")
Instead of a CNN, we use a **Transformer Encoder** to process the Entity List.
- **Input**: List of Unit Embeddings.
- **Mechanism**: Self-Attention allows the model to understand relationships (e.g., "My Medivac [A] is close to my Marine [B]").
- **Output**: Contextualized Unit Embeddings.

---

## 3. Outputs (Action Space)

We treat actions as a **Function Call** with **Arguments**.
`Action = (Function_ID, Args...)`

### A. Action Type Head
*Shape: (Batch, N_Functions)*
- **Logits**: Probability distribution over all possible command types (e.g., `Attack`, `Move`, `Train_Marine`, `Build_SupplyDepot`).

### B. Argument Heads
Depending on the chosen Function ID, we sample arguments from specific heads:

1.  **Delay Head**: How many frames to wait before this action (for APM control).
2.  **Queued Head**: Boolean (Queue action vs Immediate).
3.  **Target Unit Head (Pointer Network)**:
    - Instead of predicting (X,Y), we predict an **Index** into the Input Entity List.
    - "Attack Unit #42".
    - Uses **Attention** between the Action vector and the Entity Embeddings.
4.  **Target Point Head**:
    - Predicts (X, Y) coordinates on the map.
    - Discrete grid (e.g., 64x64 or 128x128).

---

## 4. Derived Data Pipeline (sc2reader -> Tensor)

We need a converter to turn a `sc2reader.Replay` frame into this format.

**Challenge**: `sc2reader` gives us *Events* (Action logs), but not the full *State* (Unit positions) at every tick easily.
- **Solution**: We must replicate the state tracking. `sc2reader` tracks unit creation/death. We need to strictly track their updates to generate the "Entity List" at any frame `t`.
