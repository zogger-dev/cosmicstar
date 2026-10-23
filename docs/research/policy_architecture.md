# AlphaStar Policy Architecture Research

## 1. Overview
This document details the neural network architecture for the `cosmicstar` agent, based on the AlphaStar (Nature 2019) specification. The architecture is designed to handle the complex, multi-modal state space of StarCraft II, combining spatial inputs (minimap, screen), non-spatial entity lists, and scalar game data.

The architecture consists of three main stages:
1.  **Torso (Encoders & Fusion)**: Processes raw observations into a unified latent representation.
2.  **Core (Deep LSTM)**: Maintains a long-term memory of the game state.
3.  **Head (Action Decoders)**: Autoregressively generates actions using Pointer Networks.

## 2. Torso Architecture
The "Torso" fuses three distinct input streams.

### 2.1. Input Streams
1.  **Spatial Features**:
    *   Inputs: Minimap ($64 \times 64$) and Screen ($64 \times 64$ or $24 \times 24$ downsampled) categorical features (height_map, visibility, creep, player_relative, etc.).
    *   Processing: **ResNet**.
        *   Structure: Scatter connection or standard ResNet blocks.
        *   Output: downsampled spatial map embedding (e.g., $16 \times 16 \times 256$).
2.  **Entity List**:
    *   Inputs: List of visible units (up to 512 entities). Each unit has attributes: `[unit_type, owner, health, shield, energy, x, y, ...]`
    *   Processing: **Transformer Encoder**.
        *   Mechanism: Self-attention over the entity list to learn unit-unit interactions.
        *   Output: Entity embeddings (e.g., tensor of shape `[N_entities, embed_dim]`).
        *   *Fusion Note*: These embeddings are averaged (mean pooling) to form a global entities vector, AND used directly by the Pointer Network in the head.
3.  **Scalar Features**:
    *   Inputs: Minerals, Vespene, Food Cap, Game Loop, etc.
    *   Processing: **MLP** (Multi-Layer Perceptron).
        *   Output: Scalar embedding vector.

### 2.2. Fusion (The "Torso" Output)
The outputs from the three streams are fused to form the input to the Core LSTM.
*   **Method**: Concatenation of:
    1.  Flattened or Averaged Spatial Embedding.
    2.  Averaged Entity Embedding (Global context).
    3.  Scalar Embedding.
*   **Result**: A 1D vector representation $e_t$ representing the global state at time $t$.

## 3. Core Architecture (Deep LSTM)
The core handles temporal dependencies, critical for StarCraft II (hidden information, strategy evolution).

*   **Type**: Deep LSTM (Long Short-Term Memory).
*   **Hidden Size**: **384** units per layer.
*   **Depth**: **3** stacked LSTM layers.
*   **Connections**: Standard stacked LSTM connections (Layer 1 output -> Layer 2 input).
*   **Total State Size**: $3 \times 2 \times 384$ (Hidden + Cell states for 3 layers).
*   **Delay**: The LSTM receives the observation $O_t$ and the previous action $A_{t-1}$.

## 4. Head Architecture (Pointer Network)
The action head is **autoregressive**. It produces a structured action: `Func(Args...)`.

### 4.1. Action Type Selection
*   First, the agent selects the `Function ID` (e.g., `Move_screen`, `Train_Marine`).
*   **Input**: LSTM output $h_t$.
*   **Mechanism**: Softmax over available action IDs (masked by validity).

### 4.2. Argument Selection (Pointer Network)
Many arguments (like `target_unit_tag`) refer to specific entities in the observation. This is solved via a **Pointer Network**.

*   **Query ($Q$)**: The LSTM output $h_t$ (projected via a linear layer).
*   **Keys ($K$)**: The **Entity Embeddings** from the Torso (Transformer output).
*   **Mechanism**:
    $$ \text{Attention}(Q, K) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) $$
*   **Output**: A distribution over the N entities. The agent samples an index corresponding to the target unit.

### 4.3. Spatial Arguments
For arguments like `target_point (x, y)`:
*   The head attends to the **Spatial Embeddings** (from the ResNet) or uses a Deconv tower.
*   AlphaStar typically uses a mechanism to select a location $(x, y)$ from the spatial feature map, often conditioned on the Pointer Network output if a unit was selected first (autoregressive dependencies).

## 5. Implementation Considerations (PyTorch)
*Originally written for JAX/XLA; the codebase settled on PyTorch for both the world model and the policy, and JAX/Flax were removed in October 2026.*
*   **V-trace**: For off-policy correction during training.
*   **UPGO**: Upgoing Policy Gradient Optimization for value estimation.
*   **Batching**: The LSTM state must be managed carefully across batches (an explicit unroll loop over the time axis, or `torch.compile` over the step function).
*   **Scatter/Gather**: Efficient implementation of the Pointer Network requires efficient scatter/gather ops (`index_select`, `gather`, `scatter_add`), which should be kept out of Python loops so they fuse under `torch.compile`.
