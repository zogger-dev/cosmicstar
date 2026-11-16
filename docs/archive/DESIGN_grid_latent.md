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

## Pretraining (2026-11-01, `scripts/train_ae.py`, run `ae16_20261101_1254`)

The grid runs stacked dynamics, z, the inverse and anti-collapse terms on a latent nobody had shown
can hold the game, and all collapsed. Step back: train the encoder and the decoders alone, one frame
in, the same frame out, gradients fully on. If it trains, a latent exists that holds the whole state
in a decodable form; the dynamics come after, on a pretrained encoder.

- Encoder: GridEncoder unchanged (16 x 16 x 128 + CLS and players).
- Decoders: entities (counts per (cell, owner, type), one query per entity: offset, numbers,
  order; order-free matching); map layers at 1 map unit (MapDecoder: each real cell from its latent
  cell's vector and its offset; visible / seen / creep per player, pathable, placeable, height; 4,096
  sampled cells per frame in training); players' numbers and upgrades from their tokens.
- Not decoded yet: order queues beyond the current order, production, buffs, effects.
- 64 frames per batch (a third from the combined view), 20,000 steps, ~0.9 s/step, 20 GB.
- Evaluation: matched position error (mean, median, share within 0.5 map units), count precision /
  recall from the predicted counts, count error per frame, HP, numbers, orders, map IoU per layer,
  height, player numbers, upgrades; per (set, view).

The rollout mode (`--seq_len`) is built and smoke-tested, parked until the encoder is pretrained.

**v2 (run `ae16v2_20261101_1347`).** `ae16_20261101_1254` (stopped at step 3,425) was decoding with
counts and type-given queries; on unseen maps it stalled (median position error 2.3 map units,
count recall falling 0.49 -> 0.36 by step 3,000) while seen games reached 1.1 at step 1,000.
Changes:
- No counts. Queries are (cell, slot) only, one per true entity; type (CE over the unit vocabulary)
  and owner (neutral / player 0 / 1) are decoded; matching within each cell by y offset.
- The 49 entity numbers by kind: 27 continuous (smooth L1), 13 binary flags (BCE), 3 categorical
  groups (display, cloak, inside; CE with a "none" class).
- The order queue: the first 4 orders per entity (ability, with an "empty" class; target offset;
  targets a unit; progress), replacing the single current order.
- All 11 map channels (10 BCE, height MSE).
- Loss groups balanced by learned uncertainty weights (exp(-s) L + s, 14 groups).
- Batches are moments in all three views (p1, p2, combined; shared player codes), 21 per batch.
- The decomposition check: a head predicts each player's view latent from the combined view's latent
  of the same moment (stop-gradient targets; a probe, `--decomp_grad` 0); evaluation reports the
  unexplained variance against copying the combined latent.
- Evaluation per attribute: each continuous number's error, each flag's F1, each categorical's
  accuracy, type / owner accuracy, order queue (order accuracy, length, target error), IoU per map
  channel.

**v3 (run `ae16v3_20261101_1412`).** v2 stopped at step 1,475 to add:
- the order queue at 8 slots plus the queue's full length (log; orders can queue up to ~19, rarely;
  20 sampled views reach 7, p99.9 is 3), and the production queue at 8 slots (ability, progress:
  a reactored facility's full queue);
- the container pointer: each entity points to its container (transport, bunker, command center,
  gas building, an add-on's building) among the frame's queries, or none; cross entropy against the
  query matched to the true parent's row;
- a per-cell entity count (0-127, cross entropy) from the latent: in the loss, used to place the
  queries at inference (training places them from the truth);
- on_map and playable out of the map loss (structure: on_map is the mask, playable ~1 in the crop).
19 loss groups under the uncertainty weights. v2's decomposition check at step 1,000: the head
barely beat copying the combined latent (unexplained 0.26 vs 0.25 on seen games, 0.23 vs 0.30 on
unseen maps); still a probe.

**v4 (run `ae16v4_20261101_1435`).** v3 at step 1,000 generalized better than v1 (unseen-map median
position error 1.68 vs 2.33; per-cell count accuracy 0.62 on non-empty cells; type 0.55, owner 0.64)
and its decomposition probe beat copying (unexplained 0.27 vs 0.51-0.60). But production (0.000 on
occupied slots), containers (0.005) and the rare flags (flying, burrowed, cloaked, selected, ...: F1
0) stuck at the majority class: their losses averaged over every entity / slot, where "empty",
"none" and "false" are almost always right. v4 balances them: the mean over occupied slots / units
with a container / a flag's positives plus the mean over the rest (order and production queues,
container pointer, each binary flag).

**v5 (run `ae16v5_20261101_1504`).** v4 at step 1,000 learned what v3 had stuck (unseen maps:
production 0.74, order queue 0.71, containers 0.26, flying F1 0.41, add-on 0.44, selected 0.35), at
a small early cost (median position error 1.87 vs 1.68). But the categoricals' 0.98-0.99 accuracy was
the majority class: cloak is 46% unknown and 53% not cloaked, with cloaked, detected and allied at
0.04%, 0.06% and 0.004% of rows; inside is ~1% passengers and in-gas workers. v5: the categoricals'
cross entropy is averaged per class (over the classes present) and then over classes; evaluation
adds each state's recall and share, and type's balanced accuracy (mean recall over the types
present). Display only takes visible / snapshot in player views (hidden doesn't occur), so it
duplicates the snapshot flag.

**v6 (run `ae16v6_20261101_1657`): the frame cache.** v5 reached step 6,800 (unseen maps at step
5,000: median position 1.68, per-cell count 0.71, type balanced 0.42, production 0.90, containers
0.40; seen games much better: count 0.95, position 1.03) and was OOM-killed: the in-memory pool (12
rotating games + 6 evaluation games, three views each) grew to ~47 GB. Batches now come from the frame
cache (worldmodel/frame_cache.py): all 629 extracted games, 80 moments each in all three views,
50,320 moments / 150,960 frames / 1.5 GB on disk; train 448 games (14 maps), held out 23 games on the
training maps and 158 games on 7 maps never trained on. Every batch draws its 21 moments from all 448
training games instead of 12. Validation every 100 steps: the reconstruction loss on fixed batches
from the held-out games and the held-out maps, next to the training average. Same model and losses
as v5; RAM ~a few GB.

### Player identity and map views (2026-11-01, after v6 at step 10,400)

v6's player-view vision and creep scored 0.85-0.88 on seen games and 0.19-0.35 on held-out data, while
the combined view held 0.76-0.78 everywhere. A player's frame filled `visible_0 / seen_0 / creep_0` or
the `_1` set by slot, but player codes were random per sample and player tokens carry no position, so
the encoder wasn't directly told which slot the viewer had. Suspected, not proven: the decoder had to
pick a set it couldn't know, except by memorising seen games. Why the combined view scored well under
the same setup wasn't explained.

Changed (the cache stays; all of it happens when reading and collating):

- **Player index:** the player code is the player's index, 0 or 1 (neutral its own), the same in all
  three views. Team codes are no longer read by the grid encoder; the relation stays. One entity has one
  owner and one source view, so these stay one-hot.
- **Swap:** the reader exchanges the two players at random per sample (codes, token order, map layers,
  all frames and views alike), so an index identifies a player and says nothing else.
- **Viewer code:** multi-hot, once per frame, on CLS through a bias-free linear layer: `[1,0]`, `[0,1]`,
  and `[1,1]` for the observer, whose embedding is therefore the sum of the players'.
- **Map views:** no `_0 / _1` fields past collation. `map_views` splits a batch into the static layers
  per frame and one row per player view: one in a player's frame, two in a combined frame. The encoder
  sums the rows it has. One decoder reads a view from a latent cell, an offset and which player; another
  reads pathable, placeable and height. The map metrics are now `visible`, `seen`, `creep`.

Checkpoints from before this don't load.

### Fixed pyramid (2026-11-01, `worldmodel/quad_latent.py`)

The real-unit pyramid and the 16 x 16 latent were two grids that didn't line up: only level 1's cells were
scattered, so levels 2-4 reached the latent only through entity and global tokens. Now one grid, by
halving the map: 128, 64, 32, 16 norm tokens over a square of the longest playable side (132-166 map
units across the 629 cached games, so a norm cell is 1.03-1.30 units and always holds a real cell). The
16 x 16 level is the latent.

| Level | Tokens | Attention |
|---|---|---|
| 128, 64, 32 | norm + lattice (1, 2, 4 units) + entities, reading the globals | 3 blocks in 4 x 4 windows, the middle block's slid by 2 |
| 16 | norm + entities + globals | 3 full blocks |

- 3 ResNet blocks on the norm tokens start each level; a 2 x 2 merge (`Down`) ends it. Norm tokens start
  from their place and the mean of the lattice cells inside them.
- Square cells: circles stay circles. Lattice cells of the square past the playable area stay tokens with
  the playable layer false (real terrain inside the cached crop, zeros past it). 9% of the cells on average.
- Globals read in every window and write through one attention over the level's tokens.
- Attention is the only way entities reach cells: more entities take more of a joint softmax, so counts are
  visible (saturating). No summed scatter unless counts stall.
- Windows are padded by size group, never to the fullest; a large group is split.

Tested against brute force: window membership, one block's output (1e-15, double precision), lattice
placement. A frame's latent differs by ~0.005 (values up to ~4.6) with other batch-mates, the same as running
one batch twice: conv rounding, not padding.

Speed: the first version, plain SDPA with the block uncompiled, took 1.49 s/step at 48 frames, a third of it
in SDPA's backward, and compiling the whole block crashed on fresh batches (checkpoint recompute mismatch).
Now FlexAttention when training (SDPA without gradient: FlexAttention's no-gradient path doesn't compile inside
these parts, and the two agree to 5e-7), and the block split into parts that take only tensors (`QuadPre`,
`WinAttend`, `GlobalAttend`, `QuadPost`), so a window group's size is a dynamic shape. Encoder alone: 0.74 s/step
and 20.3 GB on fresh batches, with compile errors not suppressed. Left uncompiled because they don't matter:
window grouping (1.1% of a step; its output shapes depend on the data), token embedding (0.3%), decoder targets (0.1%).

### Hungarian matching (2026-11-02)

Pretraining paired predictions and entities by sorting both on y within a cell. Entities tied in y (26% within
0.05 units: rows of buildings, mineral lines) got an arbitrary slot, so their x couldn't be learned: at step 8,000
of `ae_20261101_223407`, x error 1.44 tied against 0.25 not tied, with y fine for both (12 frames, unseen games).
Passengers and workers in gas sit exactly at their container, so they tie in x as well, and 5% of entities share
an exact position; no chain of position tie-breakers helps those.

`grid_latent.hungarian` now finds the cheapest pairing over each whole frame, not within cells: a latent cell's edge is
a line we drew, and a unit just across it shouldn't be scored as a total miss. Cost: position error in map units + cross
entropy of type and owner + mean error of the numbers. Offset targets are measured from the querying cell's center, so a
cross-cell pair costs its real error. Slots and true counts still place the queries; they no longer define the pairing.
Not done yet: soft counts at cell edges, and metrics for how fuzzy the boundaries are. Evaluation uses the same pairing,
so scores from earlier runs (y-sorted) read lower for the same model. Set accuracy of types per cell (0.90 against 0.85
paired, same checkpoint) suggests pairing cost type about 6 points; type also falls with crowding (1.00 in a cell of one,
0.69 in a cell of 16-31), which matching won't fix.

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
