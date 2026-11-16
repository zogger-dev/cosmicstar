# Latent world model: design draft (2026-10-31)

Status: built (2026-11-01) as `train_direct.py --model latent` (worldmodel/latent_dynamics.py), except
the engine-command teacher (needs the per-loop re-extraction). The target encoder is a flag:
online + stop-gradient (default) or a moving-average copy. See the README section.

## Why

The direct model (`train_direct.py`: encoder, then a predictor that outputs positions, HP and
orders at t+k) is now fast (~1.15 s/step at 16 frames x 4 horizons, dense encoder) and beats
persistence well on held-out maps (moving units −68% at 4 loops, −42% at 56). Two things fail:

1. **Commands are ignored.** Predictions are identical with commands given or withheld, in every
   view. Units a command touched are predicted worse than persistence at short horizons in the
   combined view (0.195 vs 0.086 map units at 4 loops).
2. **The latent z is unused.** KL is ~0 for every slot, the enemy's included, even with the
   posterior reading the enemy's true orders and commands.

Working diagnosis: the information is present but reachable only by a multi-hop route (command
token, then the selection or control group it applies to, then those units, then a new order),
learned from ~4% of units while current motion and current orders explain the other 96%. The
action encoder gets almost no gradient (the predictor ignores it), so it stays weak, so it stays
ignored. The posterior has no objective that needs it.

## Principles

- **The world model is the latent.** A representation whose future is predictable from its present
  and the actions, with the dynamics learned in latent space (JEPA view), not pixel or value
  prediction.
- **Decoders are probes.** Heads that read positions, HP, orders and deaths out of latents run on
  stop-gradient features (or at a very low weight), to measure what the latent holds without
  shaping it.
- **Actions are a residual.** The change a command causes is a separate, identifiable term.

## Architecture

Per unit token (and per cell and global token), with f the encoder:

```
s_t            = f(frame_t)                                   tokens: globals, entities, cells
s_{t+k}^target = f_ema(frame_{t+k})                           slow-moving copy of f, stop-gradient

ŝ_{t+k} = s_t + h(s_t) + g(s_t, A)
          │      │        └─ action delta: state tokens attend to the action tokens A
          │      └─ passive dynamics: what happens with no new commands (k given)
          └─ the present
```

- **h (passive):** a transformer over the state tokens (the current predictor's shape, without
  command memory), conditioned on the horizon k.
- **g (action):** cross-attention, queries = state tokens (after h), keys and values = action
  tokens; output added to each state token. Actions do not attend to the state by default.
  Option: a joint [state, action] embedding whose state outputs are discarded, keeping
  contextualized action tokens (see Anti-collapse for the guard this needs).
- **A (action tokens):** what the player actually issued (below). Not the engine's resolved
  commands: which units received an order is a consequence, part of the next state.

## Actions

Commands play three roles; they need not share one representation.

| role | question | representation |
|---|---|---|
| condition the dynamics (g) | what did the player issue? | humans: replay commands + selections + hotkeys (L1 + L0); bots: raw API commands with their unit lists |
| prior / future policy | what will this player do next? | per-player command sequence (+ the input stream for humans) |
| inverse model / z | what happened that the present didn't predict? | per-player embedding aligned with the residual transition |

Levels of description:
- **L0** input stream: keys, selections, hotkeys, camera (humans only; how a player plays)
- **L1** player commands: ability + target, no recipients (what we feed today)
- **L2** engine commands with recipients: ability, target, loop, the units that received it
  (the extraction's `cmd_*` tables). **Part of the transition, not an input**: the engine's
  resolution of the player's input (selection, hotkeys) into recipients. A target / probe.
- **L3** order changes on units: the outcome (a probe target, not an input)

Humans: the model must learn the selection-to-recipient resolution as part of the dynamics.

**Engine commands as a training-time teacher** (never an input):

```
teacher (training only): e_eng[unit] = Enc_eng(the engine commands that unit received in [t, t+k))
student (always):        e_rep[unit] = the unit's state token attending to the replay actions
                                       (its selection / control-group bits let it resolve whether
                                        "select 1, Move" applies to it): g's cross-attention output
loss: Barlow Twins(e_rep, e_eng), per unit, over the batch
```

A dense, direct gradient for the selection-to-recipient resolution (every unit gets a pair, not
just the ~4% whose position error reflects a command); no leak, since g uses e_rep at inference.
Most units receive nothing in a window: weight commanded units up in this term, or sample a
balanced subset. Option: a slow-moving or frozen-after-warm-up teacher.
Bots: the raw API action names its units, so for them the action legitimately carries recipients
and the action-to-effect mapping is direct (one reason bot replays should help).
Player views: the viewer's own actions (the enemy's true orders and commands go to the posterior
only). Combined view: both players'. Withheld: none (flag).

## The latent z

With residual dynamics, z has a natural home: **an inferred action embedding**, in the same space
as the action encoder's output.

- **Posterior** q(z | s_t, s_{t+k}): reads the residual the actions must explain (and, in player
  views, the enemy's true orders and commands, as `EnemyIntent` does now).
- **Prior** p(z | s_t): what a player is likely to do, from the present (later: player
  embeddings for individual styles).
- **Used where commands are missing:** commands withheld, or the enemy in a player view: g reads z
  in place of (or next to) the action tokens.
- **Aligned with the real action embedding** where commands exist (below), so z means "what was
  done", not an arbitrary code.

## Anti-collapse

- **Target encoder:** f_ema, an exponential moving average of f; latent targets are stop-gradient.
- **Latent prediction loss** between ŝ_{t+k} and s_{t+k}^target, per token, over tokens that exist at
  both times (deaths: an exit probe; births: see open questions).
- **Variance and decorrelation** on s_t (VICReg: per-dimension variance above a floor, small
  off-diagonal covariance), so the latent can't shrink to a constant.
- **Decorrelation between the branches** (the identifiability term): the batch cross-correlation
  between h(s_t) and g(s_t, e(A)) (e: the action encoder) pushed toward **zero**, so g carries
  only what h doesn't; likewise between g's output and the encoder's s_t, so the action branch
  doesn't re-encode the state. g = 0 is trivially uncorrelated, so it needs a partner: a variance
  floor on g's output over samples **with actions only** (no noise injected when there are none),
  plus the prediction loss. Linear decorrelation leaves nonlinear dependence; HSIC if that shows.
- **Inverse alignment (Barlow Twins) between the residual and the actions** (the posterior / z side):
  `Δ_res = s_{t+k}^target − s_t − h(s_t)` (stop-gradient through h) vs the action embedding (per
  player; per unit for L2 commands). Cross-correlation over the batch toward the identity:
  the part of the transition h can't explain must match what the commands say, and neither side
  may collapse. This also gives the action encoder a direct training signal.
  Using the residual, not the whole transition, is the guard if action tokens are contextualized
  by the state: state information can't help match what h already explained.
- **h trains alone on withheld samples** (g off), so h learns passive dynamics specifically and
  with-command samples can only improve through g.

## Balancing the losses

Terms: latent prediction, VICReg, branch decorrelation, inverse alignment (z), teacher alignment,
KL; probes are stop-gradient (free). Log each term's loss and gradient norm on the shared
parameters; balance with learned uncertainty weights or GradNorm rather than hand-tuned
coefficients; stage it (the action encoder against the teacher first, a fast well-posed
problem, then the dynamics), adding each loss once the parts it depends on are meaningful.

## Probes (stop-gradient)

On s_t and on ŝ_{t+k}: entity position (exact, map units), HP and shield fraction, order, exit
(present / died / gone); cells: input layers, occupancy. Read against persistence and against the
current direct model. Diagnostics: probe error with g on vs off for commanded units (the command
effect); attention from commanded units to their command tokens.

## Data

- Human pro replays (current corpus) plus **AI Arena bot replays**: explicit, dense commands with
  recipients (raw API, no selection), mechanics-heavy. A source token (bot / human) so the prior
  doesn't learn inhuman micro as human. Larger and slower to extract (action counts).
- Later: player embeddings (individual styles).

## Open questions

1. **Births:** units appearing in [t, t+k) have no present token. Map cells (occupancy after a
   U-Net expansion of the per-level maps) or spawn tokens from producing structures.
2. **Rollouts:** feeding ŝ back in for multi-step prediction; how errors compound; train on
   rollouts (k1 then k2) or only single steps.
3. **Horizon k:** h and g conditioned on k (as now) or fixed-step dynamics rolled out.
4. **Contextualized actions:** joint [state, action] embedding (keep action tokens) vs plain
   action tokens.
5. **Per-player vs per-unit Barlow Twins:** per unit gives many more aligned pairs; per player is
   what the prior and z need.
6. **What survives:** with decoders detached, which details the latent drops (probes show it);
   whether a small decoder weight is needed.
7. **Cells:** whether cells get an explicit position embedding (today: rotary only, inside attention).

## Plan

1. Probes on the current checkpoint: what the latent holds now.
2. Command diagnostics on the current model: attention from commanded units to their command
   tokens; the command effect (given vs withheld) on commanded units. Engine commands stay a
   target, not an input.
3. Build the latent model (f, f_ema, h, g, probes, VICReg, residual Barlow Twins), compare with the
   direct model on the same probes and on the command effect.
4. Bot replays.
