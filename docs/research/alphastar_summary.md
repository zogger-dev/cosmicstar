# AlphaStar Architecture Summary

## Overview

AlphaStar is a multi-agent Reinforcement Learning (RL) system that achieved
Grandmaster level in StarCraft II. It combines supervised learning,
reinforcement learning, and a novel league training setup.

## Neural Network Architecture

### 1. Inputs (Observation Processing)

- **Entities**: Unit list (position, type, health, etc.) is processed by a
  **Transformer Torso** to create entity embeddings.
  - _What is a Torso?_: In RL, the "torso" is the shared main body of the
    network that extracts features before splitting into different "heads"
    (policy, value).
  - _Same as GPT?_: Yes, it uses the same **Self-Attention** mechanism found in
    LLMs like GPT or Gemini. Instead of attending to words in a sentence to
    understand context, it attends to **other units** in the list to understand
    the battle configuration (e.g., "My Medic should heal that Marine").
- **Spatial Data**: Minimap and screen features are processed by **ResNet-style
  Convolutional Neural Networks (CNNs)**.
- **Scalar Data**: Resources, time, supply, etc., are encoded via simple **MLPs
  (Multi-Layer Perceptrons)**.
  - _Note_: An MLP is a standard feed-forward neural network (layers of nodes
    connected to the next layer), used here to process simple numbers into a
    feature vector.

### 2. The Core (Reasoning & Memory)

- **Deep LSTM**: A large Long Short-Term Memory network integrates the processed
  embeddings. This provides the agent with "memory" to handle partial
  observability and long-term strategy (e.g., remembering a scouted enemy base
  from minutes ago).

### 3. Output (Action Generation)

#### A. Auto-regressive Policy Head

Actions are generated sequentially, not simultaneously. The network models the
conditional probability: $P(Args | Action)$.

- **Why Sequential?**
  - **Dependency**: You cannot meaningfully choose a target until you know
    _what_ you are doing. If you choose `Train Marine`, picking a map location
    is nonsensical; you must target the Barracks. If you choose `Attack`, you
    _must_ target a Unit or Location. Masking irrelevant arguments based on the
    chosen Action is crucial.
  - **Search Space**: Predicting every combination of Action + Unit + Target
    simultaneously would result in billions of output possibilities
    ($500 \times 500 \times 65,000$). Breaking it down ($500 \rightarrow 500$)
    makes it tractable.

#### B. Pointer Networks

Crucial for selecting specific units from the variable-length entity list.
Instead of predicting a fixed coordinate, it "points" to a specific entity
embedding from the input.

- **The Problem**: Standard classifiers have fixed classes (e.g., "Dog", "Cat").
  In StarCraft, the "classes" (units) change every frame (3 Marines vs 50
  Zerglings). "Output Node #5" has no consistent meaning.
- **The Solution**: Use the **Input** as the **Output**.
  1.  **Key**: The Transformer generates embeddings for all visible units (e.g.,
      `[u1, u2, u3]`).
  2.  **Query**: The network generates a "Query" vector based on its intent
      (e.g., "I want to attack high-value targets").
  3.  **Selection**: It computes the dot product (similarity) between the Query
      and every Key. The Softmax of these scores determines which specific unit
      in the list is selected (e.g., "Attack Unit #2").
- **Target Point Head**:
  - Predicts (X, Y) coordinates on the map.
  - Discrete grid (e.g., 64x64 or 128x128).
- **Benefit**: This allows **generalization**. The agent learns to select
  targets based on their _properties_ (Health, Type), regardless of how many
  units are present or where they are in the list.

## Training Pipeline

### Phase 1: Supervised Learning (Imitation)

- Trained on **971,000 matches** played by the top 22% of human players
  (versions 4.8.2-4.8.6).
- **Availability**: Blizzard releases anonymized replay packs (s2client-proto).
  While the specific filtered training set isn't a single "download link", the
  raw data comes from Blizzard's public replay releases.
- **Transition**: This supervised network serves as the **initialization**
  (starting point) for the Reinforcement Learning phase. Without this
  pre-training, the agent would flail randomly and never learn complex
  strategies.
- Result: Platinum/Diamond level performance.

### Phase 2: AlphaStar League (Reinforcement Learning)

- **Multi-Agent RL**: Agents play against each other.
- **The League**: A dynamic population of agents.
  - **Frozen Copies**: Every few billion steps, a "snapshot" of an agent is
    **frozen** and added to the league. These frozen agents do not learn
    anymore; they exist solely as opponents. This ever-growing pool of past
    versions ensures the learning agents can't just "switch" strategies; they
    must be able to beat _everything_ they (and others) have ever been.
  - **Main Agents**: The primary agents being trained to win. They play against
    the frozen pool and current exploiters.
  - **Exploiters**:
    - **Reward**: Their reward is simply **Winning**. The difference is **who
      they play against**.
    - **Main Exploiters**: Play _only_ against the Main Agents. If they find a
      cheesy strategy that beats the Main Agent 100% of the time (but loses to
      everyone else), they get a huge reward. This forces the Main Agent to
      patch that hole.
    - **League Exploiters**: Play against _everyone_ in the league (past and
      present). They find systemic weaknesses common to the population.
- **Prioritized Fictitious Self-Play (PFSP)**: Agents play against a **sample**
  of past versions of themselves and other league members. They do not play
  against every single version; they prioritize playing against opponents that
  they are currently struggling with or that are similar in skill level (plus
  some older ones to prevent "unlearning" basics).

## Key Takeaways for Our Project

- **Architecture**: A simple CNN is not enough. We need to handle unit lists
  (Entities). The `pysc2` feature layer API gives us spatial layers, but the raw
  unit interface (structured) is what AlphaStar heavily relied on.
- **Action Space**: We need to handle arguments. "Attack" is not just one
  button; it's `Function(Attack) -> Args(Target Unit / Location)`.
- **Advanced Micro**: Consider using specialized sub-agents (e.g., Reaper units)
  as critics or policy guides to accelerate high-skill learning. See
  [hierarchical_micro.md](file:///c:/Users/prade/Documents/workspace/cosmicstar/docs/research/hierarchical_micro.md)
  for details.
