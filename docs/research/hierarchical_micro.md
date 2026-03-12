# Research: Hierarchical Micro & Specialized Sub-Agents

## Concept

Instead of training a single monolithic brain to handle everything from base
building (Macro) to individual unit dancing (Micro), we can train specialized
**Micro-Agents** for high-skill tasks (e.g., Reaper vs. Zergling, Blink Stalker
kiting). These agents then serve as "Experts" or "Critics" to guide the primary
AlphaStar-like model.

## Why it's Worthwhile

1. **Sparsity of Reinforcement**: In a full game, "perfect micro" might only
   happen 10 minutes in. The reward signal for that micro is delayed and
   diluted. Training on a dedicated "Micro Map" provides dense, immediate
   rewards.
2. **Domain Knowledge Injection**: It allows us to encode human tactical
   knowledge (e.g., "how to split against banelings") into a neural policy
   without brittle hard-coding.
3. **Modular Improvement**: We can upgrade the Reaper-micro agent independently
   as the meta shifts, without retraining the entire 100M+ parameter main model.

## Integration Strategies

### 1. Reward Shaping (The "Critic" Approach)

The main agent receives an auxiliary reward based on the assessment of the
Micro-Agent.

- **Formula**: $R_{combined} = R_{env} + \lambda \cdot (Value_{Micro}(obs))$
- **Pros**: Simple to implement in the environment wrapper.
- **Cons**: Can lead to "Reward Hacking" where the agent over-focuses on micro
  even when it's strategically bad to fight.

### 2. Policy Distillation (The "Teacher" Approach)

Add a loss term (KL Divergence) during training that encourages the main agent's
policy ($\pi_{main}$) to match the Micro-Agent's policy ($\pi_{micro}$) when
relevant units are selected.

- **Trigger**: Only active when the `selected_units` contains the specialized
  unit type (e.g., Reaper).
- **Pros**: Injects the skill directly into the main brain's weights.

### 3. Hierarchical Switching (The "Manager" Approach)

The high-level policy outputs a "Macro Action" or a "Call Sub-Policy" action.

- **Mechanism**: Control is handed over to the specialized sub-network for a
  fixed number of steps or until a condition is met.
- **Pros**: Guaranteed high-performance micro.
- **Cons**: Complexity in "handoff" logic ensuring the sub-agent doesn't walk
  the Reaper into a known trap the Macro-agent scouted.

## Proposed Experiment: "Reaper vs. Ling" Micro

1. **Environment**: Create or use a small `pysc2` map with 1 Reaper and 4
   Zerglings.
2. **Specialized Agent**: Train a small MLP-based agent to maximize "Damage
   Dealt vs. Damage Taken".
3. **Integration**: Use this agent to generate **Action Suggestions** in the
   `CosmicAgent`'s decision loop, which are then fused with the main model's
   logits.

## Potential Risks

- **Context Loss**: A micro-agent trained in a vacuum might not know about the
  "hidden" Lurker just outside its vision range that the Macro-agent scouted 30
  seconds ago.
- **Over-Optimization**: Training an agent to be "perfect" at micro might make
  it too predictable or easy to exploit by a meta-level strategy.
