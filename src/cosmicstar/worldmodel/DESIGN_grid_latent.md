# Grid latent: design (2026-11-01)

Status: first version built as `train_direct.py --model grid` (worldmodel/grid_latent.py); run
`grid16_20261101_1019`. Follows DESIGN_latent_dynamics.md (h / g residual dynamics, EMA target,
categorical z, energy score are unchanged).

## Why

The latent model kept one token per entity and a loss per token: a re-encoded entity list, a
step away from predicting attributes, and its size depended on the frame and the map. A world
model's state should be a fixed-size representation of the whole game, with entities recovered
by a decoder.

## Latent

- **Fixed grid on every map.** S x S cells (default 16) over the playable area, centered,
  scaled by its longest side: cells are square in map units, the shorter side is padded evenly
  (positions in [-1, 1] on the long axis, about ±0.95 on the short one). No coverage channel for
  partly padded border cells (left out to see what happens).
- **Scatter, not resize.** The dense encoder runs as before; its entity and effect tokens and
  the real cells of one map level (`--pool_level`, default 2 map units per cell), each at its true
  center in map units, are scattered onto the grid. Each latent cell pools what fell into it with
  one attention softmax: query = the cell's center, keys / values = the token plus its offset
  from the cell center, so the real lattice and its scale stay visible. Cells hold ~20 real-cell
  tokens plus the entities.
- **Mixer.** `--mix_depth` transformer blocks over [CLS, players, S² cells] (2D RoPE at the cells'
  centers). The latent is those 1 + P + S² tokens, d = 128: 16 x 16 x 128 ≈ 33k floats against
  ~0.6M raw inputs.
- **Dynamics** run on the latent tokens unchanged (h: self-attention; g: attends to the action
  tokens and z). The latent loss covers every token: fixed shapes, no entity padding.

## Decoder (a probe: `--dec_grad` 0.1 of its gradient reaches the latent)

Occupancy at 16 x 16 (150 player views): 68% of cells empty; a non-empty cell holds 2 entities at
the median, 21 at p99, 98 at most; ~260 entities per frame (623 max). Fixed slots per cell
(DETR) would be ~3% used, so queries are placed by counts:

1. **Counts per ((owner, type), cell).** Owner ∈ {neutral, the frame's player 0, player 1}.
   Poisson log-rates per cell for every class, a dense fixed-shape loss.
2. **One query per entity**: its cell's latent vector + its (owner, type) + its slot within the
   (cell, owner, type) group. Queries attend to each other (RoPE at the cell centers) and to the
   latent, and read out the offset within the cell, the entity numbers (tokenize.entity_features)
   and the order. Training places queries from the true counts; inference will use the count head.
3. **Order-free loss.** Within a group, predictions and targets are both sorted by y offset and
   paired in that order (exact in 1-D, on the GPU, no assignment solver). Evaluation compares the
   position error under this pairing with a global Hungarian assignment per (owner, type),
   ignoring cells: the gap measures mismatches at cell borders.

Targets are the raw features, not the encoder's entity tokens. The decoder runs on the present
latent (truth at t) and on ŝ_{t+k} (truth at t+k); evaluation also decodes the target latent
(what the latent holds) and compares with persistence (the units' own displacement, the present's
counts).

## Losses

Latent prediction, VICReg on the latent, decorrelation of g, Barlow z / commands, KL, energy
score (as before); the decoder losses (`--w_ent`); and second moments, predicted vs target
(`--w_gram` 0.1): MᵀM of the latent cells, EᵀE of the decoded entity embeddings against the
target encoder's entity tokens, and the entity x cell cross term. Gram matrices also punish
averaging (averaging shrinks second moments).

## The inverse (2026-11-01, run `grid16_inv_20261101_1108`)

The first run (`grid16_20261101_1019`, stopped at step 4,525) collapsed in time: the latent's
change between t and t+k fell from 0.42 to 0.027 and the dynamics ended 2.5x worse than copying
the present latent; decoding the true latent placed entities only within ~2.5 map units and
missed ~170 of ~260 entities in its counts. Changes:

- **The posterior is the inverse of the action delta.** δ_a = E(s_{t+k}) − E(s_t) − sg(h(E(s_t), k)),
  with E(s_{t+k}) from the online encoder on the real future frame (with gradient; the latent
  loss's targets still come from the EMA copy). The posterior reads δ_a and k only, never the
  state, and outputs each player's categorical z, the inferred action embedding; Barlow Twins
  aligns z with the action encoder's summary where commands exist. In player views the enemy's
  true orders and commands on the units the viewer sees are pooled into those units' latent cells
  and added to δ_a (action information, not state). The prior predicts z from the present; g reads
  z where actions are missing.
- **h alone on withheld samples.** A latent loss on s + h (no g) on samples with commands withheld
  (`--w_halone` 1), so h is the best prediction from the state alone and δ_a is what the state
  can't explain. g still runs there, reading z (turning g off would leave z nothing to do).
- The decoder keeps `--dec_grad` 0.1, to see whether the inverse alone holds detail; if decoding
  the true latent stays coarse by step 2,000, raising it is the next decision.

Cost: the second (online) future encoding: 26 GB instead of 12.

### Pairing z with the commands (2026-11-01, run `grid16_vic_20261101_1214`)

`grid16_inv_20261101_1108` (stopped at step 2,250) collapsed like the first run, and its Barlow
term never moved off ~64 (zero correlation): about 43 (sample, player) pairs per batch for a 64 x
64 cross-correlation, at weight 0.05. Changes:

- **VICReg pairing** (`--inv_pair vicreg`, `--inv_dim` 32, `--w_inv` 1): per-pair MSE between the
  projections of z and of the command summary, plus each side's variance hinge and covariance.
- **Targets on every sample.** Action windows are always built in full; withholding is only a
  mask (has_act) on what g and the prior read, so withheld samples still give the inverse a pair.
- **The enemy's commands as targets in player views**, encoded from the enemy's own view
  (training only, like the enemy intent).
- **A queue** of the last 8 batches' projections (detached) in the variance / covariance
  statistics (`--inv_queue`).

About 160 pairs per batch instead of 43, statistics over ~1,000 rows.

## Rollouts instead of direct multi-horizon (2026-11-01, `--seq_len 14`)

The direct samples (one jump to t+k, k drawn 4-224 loops and given as an input) plausibly feed
the collapse: one model predicting 0.2 s and 10 s ahead from the same latent is pulled toward an
averaged, slow latent, and g / the inverse see one summary of up to 10 s of commands. Instead:

- Each sequence draws a step Δ ∈ {4, 8, 12, 16} loops (`--deltas 1,2,3,4` frames) and always runs
  n = 14 steps: 2.5, 5, 7.5 or 10 s in 0.18-0.71 s increments. Δ is still an input to h and g,
  over a 4x range instead of 56x.
- Per step: that step's commands (full window, has_act mask), z_i from the inverse on that step's
  action residual δ_a,i = E(s_{i+1}) − E(s_i) − sg(h(E(s_i), Δ)), the prior from s_i, KL per step.
- Batch: `--seqs` 5 sequences x 15 frames, each frame encoded once online (a step's future is the
  next step's present) and once by the EMA copy (targets): 70 transitions per batch at about the
  previous cost.
- Losses: everything per step, teacher-forced (each step from the real latent); plus the open-loop
  rollout from each sequence's first latent with its predictions fed back (posterior z per step),
  a latent loss at every step, weight `--w_ol` ramped over `--ol_ramp` 2,000 steps; the decoder on
  the rollout's last prediction.
- g gets self-attention over the state tokens (`--g_self`), so a command's effect can move units
  between cells.
- Evaluation: open-loop rollouts; horizon K uses the smallest Δ with K / Δ <= 14 (4, 16, 56 loops
  from Δ = 4; 112 from 8; 168 from 12; 224 from 16), z from the posterior or sampled from the prior
  on the predicted latent; same columns (model, true latent, persistence).

## Identity

Left out for now. Matching is within a frame only; nothing links a decoded unit at t to t+k.

## Actions (decided, not built)

A predicted action latent is decoded to exact units by pointing into the real unit list: each
unit (features + position) queries the action latent at its position and says whether it's a
recipient and of what; trained on the engine's executed commands (cmd_* tables), which stay
outputs, never inputs.

## Open

- Latent width / resolution: sweep d ∈ {128, 256, 512} at 16 x 16 on the fullest cells; 32 x 32
  halves the clump tail.
- The level above: the count tensor or a learned compression of the latent.
- The action latent's form (commands scattered onto the grid).
- Fallbacks: a field (CenterNet) companion head; autoregressive decoding if parallel queries blend
  clumps.
- Inference-time decoding from predicted counts (not yet evaluated).
