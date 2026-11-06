# Physics model: design draft (2026-11-02, revised 2026-11-03)

Status: draft for review. Nothing here is built except a memory prototype. Items marked **Open** need a
decision; everything else was agreed in discussion.

## Why

Pretraining by autoencoding (`scripts/train_ae.py`, commit `543db84`) was the wrong first step. It is a
compression task: each token can carry its own information to the output, so nothing rewards attention for
exchanging anything. The audit of its 40,000-step model showed what it learned instead:

- Neutrals are 65% of entities: 0.19 map units on known maps, 1.26 on unseen maps. They look memorized.
- Player entities sit near 0.9 units, moving or not, and error tracks how many entities share a latent cell.
- A bigger decoder on the frozen encoder gained about 8% and told no more positions apart.

What matters is that a ball of marines at a choke point understands what that means, and that needs dynamics.
So: keep the entity list, and learn how sequences of game states change.

Long horizons and belief about hidden information belong to levels above this one, propagated up the
hierarchy. This level is physics: 2.5 s at a time, at the game's full time resolution.

## What's different from the earlier models

| | Direct pilot | Latent, grid | Autoencoder | This design |
|---|---|---|---|---|
| Entity list | Kept | Scattered or latent tokens | Dropped at the output | **Kept** |
| Prediction | One jump, horizon as input | Latent, then rollouts | None | **Next state at every step of a sequence** |
| Target | Raw state | Learned latent | Raw state | **EMA latent, raw state as an auxiliary head** |
| Anti-collapse terms | None needed | VICReg, Barlow Twins, EMA, ... | None needed | **Barlow Twins, EMA, auxiliary raw head** |
| Entity pairing | By unit id | Query slots, matching | Query slots, Hungarian | **By unit id; births by matching** |

Tokens predict their own future by row. Only births need query slots and Hungarian matching.

## Architecture: the 2026-11-04 redesign

This replaces the pyramid and windows described under Architecture and Memory below; those parts are history. Actions, heads and targets stand, except as noted here.

- **Sizes.** d = 512, 6 blocks of 5 attention layers (state spatial, state temporal, action history, actions reading, state reading actions), one softmax each, 8 query heads over 4 key and value heads, SwiGLU feed-forward of 1365 after state spatial, after state temporal and after action history. Queries and keys normalized with a learned scale. The action history has attention and feed-forward weights of its own (2026-11-05: they were shared with the state timeline, a choice made without discussion; actions and entities are different kinds of sequence). The feed-forward after spatial (2026-11-05) gives temporal attention computed features to compare, not raw gathered ones: 14% slower, peak 27.1 GB. 80.8M parameters with 5 maps.
- **Input.** 14 layers at the playable area's own size: per player, from that player's view: visible, seen, creep, pathable, buildable (per frame) and height (the map's); then playable and known, which is zero past the edge in a batch. Pathing changes with buildings and rocks, so nothing here is assumed static.
- **Tower.** A plain ResNet, 3 strides: 128 channels at half resolution, 256 at a quarter, 512 at an eighth, 1 residual block at each.
- **Grid.** Each map has 256 learned tokens of d, one per cell of a 16 x 16 grid over the square of the longest side. A frame's cell token is the map's token plus the tower at the cell's center, by `grid_sample`. A cell past the map's edge reads zero there and keeps its learned token. Maps with fewer than 5 training games are left out: 10 maps, by the earlier count. No generalization across maps for now; a model that predicts the tokens from raw map data can be distilled from them later. There is no cross-attention, no skip and no per-cell table: if fine structure such as walls is lost, the grid or the dimensions grow.
- **Entities.** Rows read the map only through attention.
- **Spatial.** Every token (globals, rows, effects, cells) has one softmax over all tokens of the frame. Whether cells crowd out entities, or the reverse, is a thing to watch: the split into two softmaxes was the earlier answer to that. No action of the frame is present.
- **Temporal.** Rows, cells and globals: a query at t reads earlier frames' action-updated tokens, with a key at t taken from t - 1's update, so a clean state at t holds nothing of t's actions. Rows read only frames they exist in. Effects have no timeline. Plain masked SDPA: every timeline has few keys. Time is rotary over game loops (2026-11-05, replacing a learned embedding of the frame's place in the window): wavelengths 3 to 672 loops (30 s), so a score depends only on how many loops apart two frames are, whatever the step, and a key-value cache stays valid as the window slides. Entity and action timelines rotate the same way.
- **Actions.** After the feed-forward, frames with actions only: actions read the state, both players' tokens, but not each other; the state reads the actions; the pointer bias is on the pair. A player's own earlier actions, in the same frame or before, reach an action through its causal timeline, which is one stream per player in issue order. The two players' actions in one frame are simultaneous, so they don't read each other; each reaches the other only through the state. The result goes to a running change kept apart from the state, starting at zero.
- **Positions.** RoPE at wavelengths from 1/32 of a map unit to 256. Tokens without a place sit at the origin. Values rotate too (2026-11-05, after Geometric Transform Attention, Miyato et al. 2023): every value is rotated by its token's place and every output back by the query's, all 64 dimensions of every head, so what a token reads is each key's content turned by the key's offset from it. Before this, position reached only the scores: a unit learned how much of its neighbours were near, not in which direction, and nothing it read moved when it moved. In spatial attention a unit now reads the direction to its neighbours; in the state timeline each earlier frame's value turns by where the token was then and comes back by where it is now, so it reads how far it has moved since each frame, and velocity and acceleration are differences of those (map cells sit still, so theirs cancel); between entities and actions each reads the other's place relative to its own. The action history keeps rotating by time only: an action's point over time is not a moving body. Shifting a whole frame or a whole track leaves every output unchanged, to rounding. The value projection chooses which bands carry content: fine bands scramble with a neighbour's offset, coarse ones barely turn within a window. Keys and values rotate at the 4 key-value heads, before they are shared out. 2.60 to 2.77 s a pass (6%), peak 27.8 to 28.2 GB, on the same 80 passes.
- **Place on the map.** Rows also carry sin and cos of their map x and y at 6 wavelengths from 4 to 256 map units (2026-11-05): map knowledge such as ramps and chokes is tied to place, which rotations, being relative, cannot say. Coarse only, since bfloat16 resolves a raw coordinate near 64 to a quarter of a unit and fine motion now comes through the rotated values.
- **Effect age.** Effect tokens carry their age in loops (2026-11-05), as log and as sin and cos at 6 wavelengths from 4 to 1024 loops, so delays such as storm ticks or a bile's landing read exactly. Effects have no id: an entry continues the previous frame's of the same type, owner and place, or for moving ones (guardian shield, the colossus beam) the nearest unmatched one within 1.5 units. Ages are counted over the whole game, so one running when the window opens keeps its age; one cast out of a view's sight is as old as its sighting there. Measured maxima: storm 54, bile 39, scan 196 loops.
- **Selections and recipients (2026-11-05).** A selection token points at the units it adds (up to 32), by the same pair bias as a command's target, with bias values of its own (codes 3 and 4 of 5: none, action to target, target to action, selection to added unit, added unit to selection). Before, a selection carried only counts, so a command right after a selection in the same loop had no way to know its units: 12% of recipient rows in sampled windows lack the selection bit. Removals are positions in the current selection, not units, so they stay counts. Who carries out a command is a target, not an input: the engine's executed commands (`cmd_*` in each player's view) are joined to replay rows by `command_join.match` (37,540 of 37,557 matched on 10 games) and to tokens through each raw row's token (a held hotkey's presses share one); 92% of command tokens get recipients. The loss: each command token's final state, projected, against each live row of its frame's clean state, one binary logit per row; positives and negatives averaged separately, then the two halves. Logged with `recip_f1` against `recip_sel_f1`, the F1 of taking the acting player's selected rows, about 0.92: the baseline to beat.
- **Progress and flags as inputs (2026-11-05).** Build, order and production progress enter as 1 - progress: bfloat16 resolves 0.004 between 0.5 and 1 and finely near 0, and completion is the moment that matters (a marine moves 0.0025 a loop). The progress targets are unchanged. The 16 bits of flags and flags2 (hallucinated, powered, burrowed, selected and the rest) are inputs too: they were targets with no input. The selected bit is the engine's selection, the observed player's only. Each player also gets supply left (cap - used, / 16, below 0 over the cap) beside the log of supply used, which resolves only 1.5 supply at 150: being supply blocked stalls production, and bfloat16 is fine near 0.
- **Fixes (2026-11-05).** Deaths: a row was absent at its death frame (the first it is gone in) and dead only from the next, so no step with a death was ever scored (1 in 600,501 rows); a row is now dead from its death frame, and 24 deaths are scored on the same windows. Facing: its change is wrapped to [-pi, pi), since a turn across 0 counted as about 6.3 (2,276 times on 30 windows). Two neutral types (raw ids 149, 639) read facing 7.0 from the game; harmless once wrapped.
- **Attention kernels.** FlexAttention for anything over entities (a mask per frame), Flash where nothing is masked, masked SDPA for timelines. Flash takes no `attn_mask`.
- **Map decoder.** Reads the sampled cell's token alone.
- **Batches.** A pass holds as many windows as fit a budget of 3,900 tokens (windows x (most rows + 259)), at most 16: 4 to 5 typical windows, 3 to 4 of the largest, 24.5 GB at most and 3 s a pass. That holds given two things: a garbage collection after each backward, since a pass's checkpoints sit in a reference cycle and take GBs from the next pass, and compiling on two small windows at the start, since compiling holds several GB until the next collection. `gc.freeze()` after that keeps the collection to 0.03 s. Loss per pass is unweighted by rows: whether to weight it is open. Those numbers are from before the learned tokens and the 14 layers: to be measured again.
- **Neutrals.** From the two player views, not an observer: one row per neutral by type and place, since one out of sight is a snapshot under another tag, and critters by tag, only while seen. Contents are as last seen by either player, backfilled to the game's start; one never seen reads 0 all game, so a nonzero value at frame 0 says it will be seen. On one game this gave the observer's 151 rows and its contents except on the never-seen rows.
- **Extraction.** `data/games_v3`, layout 3, no observer. Crashes on start under Proton are common and are retried; views that fail 3 times need `--retry_failed` runs afterwards (last extraction: 181 of 1,887 views on the first run).

`tests/test_physics_causality.py` proves the dependencies exactly on a real window: changing an action leaves clean states through its frame unchanged and updated states before it; changing a row, the map or a player's numbers leaves earlier frames unchanged; an absent row changes nothing else.

## Architecture (before the redesign)

```
frames t-55 .. t       each: entities, globals, the map
  per frame            map tokens at 64 x 64, from raw cells
  for each level (64, 32, 16):
    repeat:
      timeline         each entity along its own past; map cells too, at 32 and 16 (causal)
      current frame    entities, map cells and globals attend within the frame
    merge 2 x 2 to the next level (conv blocks on map cells before attention)
  heads                each entity's token -> its current state and its next state
```

**Sequences.** 56 frames, one game loop apart, for 55 predictions: 2.5 s. Each position predicts the next frame
from everything up to and including itself, as a language model does. No horizon input. No added motion or
history features: earlier frames arrive through the timelines.

Why 56, measured on 40 views. Frames differ more than assumed: 39% of own units move per loop, moving units step 0.137 units per
loop, and 68% of them change velocity by over 5%. Workers and air units take 11-27 loops to speed up or slow down (Oracle
20 and 27, Medivac 17 and 15, workers 15 and 12), which a 15-frame window can't hold along with rest and cruise; ground army is
nearly instant. Starts and stops per frame are the same at any length, so long sequences lose no events for the memory
and add continuity from command to ramp to arrival.

**Timeline, then current frame.** Each block lets a token pick up its history, then talk to its neighbors, and
the last block is within-frame, so timeline information spreads across the state. Timelines are causal, and
within-frame attention never crosses frames, so nothing sees the future.

**Entities.** A grid per sequence, frames x rows. A unit keeps the same row index across the whole sequence, so reading
its history is cheap and simple. Frames before a row's entity appears are masked out completely: no token is read or written, so
nothing about the future shows. Snapshots are their own rows, and the
model learns the link to the real unit; there is no hand linking.

**Rows that are hidden but alive.** Rows are masked only before birth. Gas workers cause over 99% of the
times a player's own row disappears and returns (median 32 loops), and the observation doesn't report them: the building shows no
passengers or activity, though its contents drop by 4 as a worker enters. `_gas_gaps` keeps them as rows at the building, and
transports use their exact passenger lists, with an "inside" state and a pointer to the container either way. Querying the
engine by tag answers for a hidden worker, confirming it exists but not naming the building, so that query can verify
`_gas_gaps` on sample games. `_gas_gaps` needs porting to Arrow and checking. A Neural'd unit's owner flips to the controller;
what its owner sees out of vision is untested.

**Births: predicted count plus queries (decided).** Each frame predicts how many units appear, a classifier over 0 to about
64, against a measured maximum of 26. That sets how many output-only queries run, plus a small margin. They read all tokens
at the end of the pass, nothing reads them, and each outputs real or empty, with type, parent, position and other properties.
Hungarian matching pairs them with that frame's births, so order doesn't matter. The cost is small: most tables are 1 x 1, and only
frames with births need one. Each query makes at most one entity, so the classifier's size is the limit per frame; the number
running varies with the prediction. Parent is a pointer to the producer's row, using the actions' pair attention: about 99% of
`Born` events name their creator, and buildings' `Init` events use "no parent". In training the matched unit is in its row,
masked until it appears. In a rollout, a filled query adds a row at the next step.

Measured on 80 replay logs, without `InvisibleTargetDummy` (38% of logged births, absent from rows in 12 games): births on 3.8%
of loops, about 1 per second, 74% of 56-frame windows, 2.6 per window; about 355 tokens per 6 x 56 pass against about 16,500 for
50 fixed queries. Larva to egg keeps its id; units from eggs are new ids. Feeding earlier birth tokens forward, so a query can see
that a barracks produced one marine and not two, is a later test. So is training on predicted frames over several steps.

**Deaths.** An alive/dead field on each row, in both the input and the next-state output, so predicting a death is predicting
that one value flip. From the frame after a death, the row keeps its last values, including its position, with hit points and shields
at 0 and the field set to dead, so a burst of deaths nearby is something other tokens can read, in the right window. A separate learned DEATH token was
rejected: it could sit far from the row's token in embedding space, and a flip is more likely to learn strongly. A dead row stays
until the sequence ends, and only the field is scored once, on the frame it flips, with no raw losses afterwards. Units that died
before a window starts have no row in it, which is accepted. Dead rows cost little, about 2 per 56-frame window on
average, 83 at most. Deaths fall on 2.8% of loops and are bursty: half of 56-frame windows have none, and the busiest 10% hold 58%.

The log's deaths aren't all combat. On 80 games, 23% of 50,093 are rows logged as `Larva` with no killer, which are eggs hatching: in 5 Zerg
views, 1,617 of 1,622 eggs that ended had a new row at the same spot within 2 loops. MULEs and depleted mineral fields also log a
killer player, so a killer doesn't prove combat. Half of Drone deaths have no killer, cause unchecked. Those rates include these
non-combat ends. A hatch stays an ordinary death plus a birth, with no special rule, and combat scores exclude these types.

**Map.** A 64 x 64 / 32 / 16 pyramid over a square of the map's longest playable side, without lattice tokens.
This is the existing `QuadEncoder` with its lattice removed. Fine levels keep attention by default; conv-only at
64 is an experiment. Padded cells stay tokens with playable false. Static terrain is recomputed every frame for
now; caching it is an optimization to make only if needed. Visibility, seen and creep are never cached.

**The real grid without lattice tokens.**
- Each 64-level token reads a fixed window of real cells around its center, 3 x 3 at 1 unit, through a linear
  layer. Cells there are 2.1-2.6 units wide, so this covers them with slight overlap, and the window is the
  same real size on every map.
- Attention positions (RoPE) are in map units, and each map token embeds its position.
- The global token gets the map's real width and height.
- Entities and map tokens share 4 x 4 windows and attend jointly, so a unit reads the terrain around it from its
  window's tokens. Each 64-level token's linear patch keeps its 3 x 3 cells' values available. A separate crop per
  entity is an experiment only if a probe can't recover nearby pathing from entity tokens.

**Fine structure.** Without a U-Net, fine map detail may never be learned. To inspect during training, not to
solve now: see Evaluation.

## Actions

Part of the physics model from the start: it learns the engine's deterministic response to commands (minus minor
engine randomness), and it can't do that without them. Both players' actions are given, including camera movement. The camera has no head: hypothesis, it can be decoded from
the latent without being asked for, tested with a probe on frozen weights.

**Proposal, for review.** Both players' replay events (what a player issues, the same space an agent would later act
in), one token per event, in the frame of its loop:

- **Content:** player, kind (command, repeat, select, group, camera), ability, flags. No copy of any unit's embedding.
- **Pointer to the target:** an action naming a unit in the grid carries a pointer to that unit's row, and the two can
  attend to each other: a learned bias on the attention score for that pair, one for each direction. An entity knows
  nothing about being targeted or commanded until attention with the action tells it.
- **Position:** the target point for point commands and camera moves, the target unit's position for unit commands, and
  a global position otherwise, so targeted actions land in that area's windows and RoPE sees the offset.
- **When actions attend (user decision, replaces the earlier staging and the separate D step):** state at t -> action predictor -> actions at t -> updated tokens ->
  t + 1. Training runs those dependencies in one pass with the real actions. Each block, at every level with sliding windows: (1) timelines, where a state token
  reads earlier frames' action-updated tokens from the previous block, and action tokens read each player's earlier actions; (2) state attention with no action of this
  frame, whose output is the clean state; (3) action attention, the same windows with this frame's actions, starting from the identity. The action side
  carries on as a delta on the state, cells averaged down between levels, so the state is never altered; convs are state-only. Attention decides which actions matter.
  Next-state heads read the updated tokens; a later action predictor reads the clean state; the target is the teacher's clean state at t + 1. Blocks are 2, 2, 2.
  Actions with no place are in no spatial window: per frame they share one window, reading each other and the globals, and reach other tokens through the globals.
  Globals have a causal timeline, so player tokens can hold what's learned from earlier frames, such as control groups. Checked exactly: changing an action at t leaves
  the clean state at frames up to t unchanged and changes the updated tokens at t and the clean state at t + 1. Measured on the largest windows (6 x 56, checkpointing,
  attention compiled): 27.2 GB, about 6.5 s per pass.
- **Timeline:** per player, causal, over that player's action tokens so far in the window: "select, then command".
- **Effects persist through entity and player timelines,** which is how a command keeps mattering once its token is gone.
- **Size:** 0.89 events per loop, about 50 per 56-frame window, against thousands of other tokens, so memory should barely
  change.
- **Test:** zero out actions at evaluation, and compare order-change accuracy and camera tracking.
- **Fog:** a command targeting something fogged has tag 0 plus a snapshot; it uses the point and type only.

Engine commands (`cmd_*`) with recipients stay for evaluation, not input. Timing, measured on 10 games: of a point command's recipients, none show the order at the command's own loop, and
about 69% do one loop later (the remainder not yet explained). So an action at loop L belongs to the
transition L -> L+1. Engine and replay loops agree exactly for 88% of commands.

## Heads and targets

Few heads: one entity head (all raw fields below, shared by rows and birth queries), the birth count and real-or-empty
outputs, and the map decoder. As many of an entity's attributes as possible are next-state targets with gradient (user decision, 2026-11-03), buffs included,
since carry buffs are how a worker shows what it holds. The camera stays a probe.

Discrete outputs are classifiers: birth count, flags, and pointers for type and parent. Large values use log-spaced bins plus an
offset, or regression on a compressed scale.

Per entity, by row, for the entity's current frame and for the next one:

| Target | Next-state form | Note |
|---|---|---|
| Alive/dead | Field on each row, in input and output, scored on the flip and on rows still alive | From the replay's death records; temporary absences are not deaths |
| Position | dx, dy, regression | The boosted term; scaled so a fast unit's per-loop move is about 1, since raw moves are about 0.14 units |
| hp, shield, energy, build | Change, regression | |
| Order queue | 8 slots | Ability, target, progress, on a sample of tokens, changed rows first |
| Production queue | 8 slots | Ability, progress |
| Buffs, available abilities | Multi-hot | Abilities on a sample of tokens, since the table is large |
| Control groups, add-on, engaged, rally | Bits, plus rally offset | |
| Engaged target, container, order targets | Pointers over rows, across views by tag | On a sample of tokens |
| Flags, display, cloak, type, other numbers | Classes, or change for numbers | Includes radius, height, upgrades, detection, radar, cargo |

Map: next-frame visibility, seen and creep, per sampled cell, with the existing `MapDecoder`.

**Births and deaths.** See Entities. A birth takes type, parent, position and other properties. A death is the alive/dead field flipping.

**Prediction and collapse (decided 2026-11-03).** The main objective is latent: each token at t predicts the next
frame's token as a residual, `s_t + g(s_t)`, with g a small zero-initialized per-token head, so a fresh model starts
at persistence. Targets come from a moving-average (EMA) copy of the encoder, for a teacher more stable than the
student, with Barlow Twins per entity token. The whole sequence goes through once, as in a language model: one online
pass gives all 55 predictions, and one EMA pass without gradient gives all targets. g adds almost no cost.

The raw next-state heads are an auxiliary loss with gradient into the network. Position is boosted over the other raw
terms, because it's what changes most at one loop. The auxiliary weight starts high, about 1.0, and decays over
training to about 0.1. Decaying to 0 is a later experiment: earlier latent runs drifted under detached probes.

The risk is collapse in time. Identity, owner and terrain alone can satisfy Barlow Twins, whose samples are entity tokens
(about 86,000 in a 6 x 56 pass, about 29,000 non-neutral), so its correlations are dominated by differences between
entities. Earlier runs collapsed this way, including with a decoder at weight 0.1. So stage 1 tracks how far latents change
between frames and raw-head skill against copying, and stops decaying the weight if these worsen. It also compares the correlation matrix between passes, since each pass has only about 6 games and a
running average may be needed. A further remedy, now the leading option: compute the correlations after removing each row's
mean over time. Rows that don't change go quiet, and mined fields and attacked rocks keep their signal.

**Discounted positions.** Critters' whole loss is a separate term with a heavy discount. Larva position takes the same discount,
including at birth; larva births and every other larva property keep full weight, because larvae spawn. There is no motion
term: the loss scores position, and motion is what the model works out to predict it. Critters stay tokens, since they block
pathing and building placement and move aside for a build order. Neutrals stay in every other loss.

Loss weights are fixed, not learned: learned uncertainty weights may favor easy terms and shrink next-state weight.

## Ways it can stop learning

Raw targets close representation collapse, but these remain. The pilot hit several.

| Failure | Guard |
|---|---|
| Do-nothing predictor | Score against copying; sample busy moments more often; report them separately |
| Actions ignored | Accuracy on rows whose order actually changed |
| Attention ignored: each unit follows its own order | No-attention baseline; fights and crowds reported separately |
| Rare events ignored | Scores only on frames where they happen; see Evaluation. Reweight or oversample only if those stay at the copy baseline |
| Loss weights shut off dynamics | See above; log them |

## Evaluation

During training (scripts/train_physics.py): every 25 updates a quick validation on 6 fixed windows each from unseen games and unseen maps, in the log. Every 500 updates a checkpoint, then
training pauses for a full evaluation on 144 fixed windows each from train, unseen games and unseen maps, with position error by kind of row against copying, in the run's eval.jsonl. Windows and
the sampling inside the losses are fixed, so numbers compare across updates.

**Baselines, all on the same frames:** copy the last frame; follow the current order at unit speed; the same
model with no attention between entities, timelines only.

**Usefulness of exchange.** Remove some attention and see whether next-state error gets worse: all of it, or
only enemy-to-ally, or only beyond some range. Fights and crowds should suffer most. Shift tests show that
information moved, not that it mattered, so they don't answer this alone. Cosine between pre- and post-attention
tokens is a cheap alarm only: exactly 1 means attention does nothing, but a token can shift by a fixed amount
for its own type, with nothing read from its neighbors.

**Slices,** on seen games, unseen games and unseen maps, each view: moving vs still, by entity type, entities per
cell, neutral vs player, in fights, moments with many changes. Position: mean, median, RMSE and tail percentiles,
never one average over everything; neutrals are 65% of rows. Rare changes are scored only where they happen, against copying,
plus invented ones: neutral contents and health, births, deaths and player health. Deaths are bursty, so they're also
reported in fight windows. Critters and larvae are reported apart. The earlier `tight_units` and `touched` metrics
aren't reused: the user doesn't consider them good measures.

**Physics checks:** the share of predicted positions on unpathable terrain, and pathing near narrow chokes.

**Free-running check.** Training always starts from real frames. A short check feeds predictions back for a few
steps and compares against the truth and against copying, since error compounds when the model runs on its own
output. It's a sanity check, not a training method.

**The long-run test (the user's goal for physics).** Start a replay from its first frame, give the model every action,
feed its own predictions back, and run to the end of the game. No error would mean it has learned the engine. Report
how error grows with time, against copying, at 100, 1,000 and 10,000 loops and at the end, plus time to divergence
and game-level outcomes (supply, structures, who wins fights). Critter and larva positions are effectively random, since
the model can't recover the engine's seed, so the bar is no error in what's predictable and no spread from what isn't.
Predicted states must hold every input the model reads.

**Comparison.** The pilot on unseen games, 0.18 s: position 0.041 (copying 0.108), moving units 0.31 (0.83),
orders .990 (.985). 1.43 s: 0.354 (0.729), 1.54 (3.40), .958 (.880). That used the observer extraction and 2v2
games, so it isn't directly comparable.

## Data

**Extraction: done 2026-11-03.** Every game loop, in `data/games_arrow`, 43 GB: 1,882 of 1,887 views, all 629 games. 624 games
have all 3 views. 5 miss one view and are set aside for training: `982fc2368b80adb3`, `1ff02cd0dd2751fa`,
`c50626c6c77288c6`, `5bf221dbd7045ab1` (P2), `d2de5d4aef655894` (P1). At 14 workers, about 9 hours, 46 GB peak, about 10% launch
failures. Retrying 181 views at 10 workers recovered 176 with no connection failures or crashes, so 10 workers is the
better setting. Reading all views back hasn't been verified.

**The desync check is not an engine error.** `DesyncCheck` compares rows with the replay's unit log. All 5 failures tripped its
phantom rule on a tag with high word 2, which marks a snapshot. All 5 stop at the same game time when retried. The 3 inspected were
snapshots of the player's own structures, in all three races, with a logged live structure at the same position under an unrelated tag.
Tags change between runs and flipping the high bit matches nothing. Why the engine shows these is unknown. The check stays as it
is by decision. A real client desync should hang the client and surface as a timeout, which is the user's recollection and
unreproduced. No game had more than one view time out, and 1,289 of 1,297 audited views ended exactly at the first player leave.
`extract` keeps a view if the engine turns `Status.ended` mid-game, so a real early stop needs comparing with that loop.

Already extracted, tokenizer work only: buffs (`b_id`), add-ons (`add_on`), containers (`pass_*`, `parent_row`), command
recipients (`cmd_*`, for evaluation).

**Training store: Arrow, seek friendly.** Written directly by the extractor, replacing the frame cache: `<view>.units.arrow`
and `<view>.frames.arrow` in 64-frame batches, zstd 3. Each batch's first frame is a keyframe and other rows are deltas, so a
window needs only its batch. Longest game, player view: 283 MB (159 MB `.cst`, 477 MB plain Arrow), about 4.5 ms to
decode a batch. Everything a frame needs still has to be stored per frame, so any window can be read without whole-game
preprocessing: control-group bits, latest stats, containers and gas workers.

Training store: add `ent_id` per frame and what `make_sample` derives from ids and deaths; 56-frame windows; store a
per-window change count, and flag windows with fights and births. Measure peak RAM on the largest games before scaling.
Same split: the 7 held-out maps stay held out.

**The state: both player views, not merged.** The physics model reads each player's own units from that player's view, with
a two-element observer vector marking which player sees a row, `[1, 1]` for observer mode. Neutrals (critters, resources,
rocks) come from an observer pass as a third channel; that pass keeps nothing else. Measured on one 56-minute game, 26,041
own-unit rows at 76 sampled frames: every observer row was found in its owner's view, none extra, with identical
positions, types and hit points. Observer mode alone won't do, because it reports no orders, production or passengers
(all empty). The owner's view adds those, plus selection, camera, creep and executed commands. Outputs are point
predictions. Fog and uncertainty are a second model on top.

**Effects** (Psi Storm, Corrosive Bile, nukes, Lurker spines) appear from actions, with no unit id. The observer file has
none, so they come from the player views (each view has only some). To decide: merging them across views. In the measured
game they're on about 10% of frames, up to 128 per frame, about 97% Lurker spines. **Proposal:** inputs as tokens, and
next-frame targets as map channels per effect type, using the same map decoder, so there's no matching and no new head.

**Engine placeholders** (a building's ghost after a build command) are off in the extractor (`show_placeholders=False`),
so none appear in the state. The worker's order already holds the target, so no action head is needed.

**Maps and patches.** Decided: unchanged for now, expecting physics to generalize better than the autoencoder, and the
held-out maps will test that. 448 training games on 11 maps (14 map files), 23 unseen games on those maps, and 158 games on
7 held-out maps; patches 5.0.14 and 5.0.15. If error on held-out maps concentrates near terrain, the first addition is 122
games on 8 more layouts in 5.0.14, the same patch as 436 current games. 5.0.17 has 14 games on 5 maps (raw ability ids shift
there, including Baneling morph, so those need care). 5.0.16 stays a held-out test: it changed economy and pacing.
Patch-agnostic rule embeddings with pointer outputs are a separate design.

## Memory

Measured with a prototype: the architecture above with 9.5M parameters, a backward pass and an optimizer step.
Timelines followed rows, not unit ids (the same token counts). "Largest" is both players' rows together at the biggest
moments, up to 1,041 entities per frame.

| Sequences x frames | Moments | Peak GPU | s/step |
|---|---|---|---|
| 16 x 8 | Typical / largest | 8.0 / 9.1 GB | 0.36 / 0.46 |
| 32 x 8 | Typical / largest | 15.8 / 17.9 GB | 0.73 / 0.90 |
| 48 x 8 | Typical / largest | 24.0 / 26.6 GB | 1.15 / 1.36 |
| 24 x 15 | Largest | 25.2 GB | 1.24 |

Those exclude decoders, actions and the EMA pass. **Measured on the real model, 2026-11-03** (`scripts/train_physics.py --measure`, 9.3M
parameters, attention compiled):

| Sequences x frames per pass | Checkpointing | Windows | Peak GPU | s/pass |
|---|---|---|---|---|
| 6 x 56 | Yes | Largest (808 rows) | 19.0 GB | 3.7 |
| 6 x 56 | Yes | Random | 18.5 GB | 2.9 |
| 2 x 56 | No | Largest | 24.6 GB | 0.9 |

**Plan: 6 x 56 per pass with checkpointing, 4 passes per update:** about 12 s per update on random windows, so 40,000 updates would take about 5.5 days.
Only attention compiles: compiling the convs and MLPs too fails at 40 frames and up (Inductor's `CantSplit`). A new specialization can't compile during a
checkpoint's recomputation, so calls keep one shape: at least 2 rows, always a pair table. Attention is already local (4 x 4 windows); the 64 level's cost is its
count, 4,096 tokens in each of 336 frames per pass, about 1.4M, so a conv-only 64 level is still only an experiment.

**Epochs (2026-11-05, `epochs.py`):** every pass is planned ahead from a seed, so a run's data order is reproducible and resumes exactly
(pass `step x accum`). Training and validation use one patch (`--patch`, default 5.0.14), since the model has no rules input and two patches'
numbers would blend.
- An epoch starts a window every `--stride` (14) frames of each game, from an offset of the game's own; offsets permute per game, so 14 epochs
  start a window at every frame exactly once and nothing repeats before. The model sees no absolute time, so the stride is invisible to it, and
  a frame sits at 4 depths of its window per epoch. 5.0.14: 341,705 windows, 62,088 passes, about 35 hours an epoch.
- A window's rows (units alive in any of its frames) come from each game's index without reading it (0.4 s for all games, exact). Buckets of 32
  rows, the thin ends merged until each has windows from 40 games: 15 buckets, a pass 3 to 8 windows within the 3,900-token budget. 97% of
  entity slots are live, against 70% with random passes: 51.5k live rows/s against 42.1k.
- Spread through an epoch: each bucket's passes evenly, the passes of an update from different buckets, each game's windows evenly through its
  bucket's passes and never twice in a pass, distinct maps where the bucket allows, and from a game the window whose kind (third of the game,
  effects or not) was seen longest ago. The player swap is drawn per window from the seed.
- Each pass's loss counts by its windows (`windows / windows in the update`): every loss term is a mean over its pass, so unweighted a pass of
  3 large windows would count as much as one of 8 small ones. All windows have the same frames, so counting each frame's prediction as a
  sample gives the same weights.
- To watch: Barlow standardizes over the pass, and a pass now holds windows of one size (mostly one stage of the game). If its loss differs by
  bucket, standardize with running statistics instead.

Lattice tokens were most of the old cost: the old encoder used 26 GB for 48 frames. Entity-count batch caps are
pinned for later.

FlexAttention failed with a driver error around 65,000 windows in one call (inferred, not confirmed). Splitting
window calls fixed it, so `MAX_SCORES` needs lowering, or a cap on rows.

## Stages

Each stage adds one thing, and the next doesn't start until it passes. Training everything at once is what made
the earlier models impossible to read.

| Stage | Adds | Must show |
|---|---|---|
| 0 | Every-loop extraction (done); Arrow training store with unit ids; baselines on it | Baselines reproduce; RAM measured |
| 1. Physics | Both player views plus observer neutrals: encoder, timelines, both players' actions, EMA latent, raw heads, births and deaths | Beats copying and order-following; attention beats no attention in fights and crowds; latents keep changing |
| 2. Uncertainty | A second model: player views, latents, distributions | Latents get used, without leaking the future |
| 3. Inverse | Which commands caused a change | Recovers commands from state changes |
| 4. Next level | Hierarchy above physics | Decided once physics works |

## Later: latents and belief

Not part of stages 0-1; tokens only need a place to receive a latent later.

- **Why.** The game has imperfect information. Your own units' outcomes are determined given your commands and
  what the opponent does, so latents carry what the opponent might do. Regression to the mean is the problem
  they exist to solve: "spire or hydra den", not a blend.
- **Where.** A latent on every entity except your own units (an unseen mineral field has a belief about what's
  left), on map cells (a hidden base has no entity token), and per player. Added to their tokens, so attention
  cost barely changes.
- **Ghost tokens.** A unit entering fog keeps its token, marked unobserved with the time since last seen. Its
  prior is the belief; samples give possible positions. Training has the truth from replays.
- **Unsolved.** How entity, cell and player latents condition on each other (they should attend to each other: a
  scan that finds no third base collapses the belief); whether attention along latent timelines learns what
  persists, a hidden base but not a unit's micro; keeping every level of a hierarchy in use.
- **Prior vs posterior.** Today there's only the KL between them, with free bits and balancing. A low KL can mean
  they agree, or that z carries nothing. To tell apart: swap in z from another sample and see if prediction
  changes; check z varies across states; train a probe after the fact for enemy commands, on both prior and
  posterior. Judge with prior samples, since a posterior reading the future can leak it.
- **Long-lived belief.** A 56-frame timeline is 2.5 s of memory, and a scan's result must last minutes. That
  belongs to the levels above, with evidence going up and belief coming back down.
- **Engine randomness.** Not a reason to add latents yet. A per-entity `z` in the next-state head is cheap to build, but its
  posterior sees the answer, so it can hide physics the model hasn't learned. If tried, it must improve rollouts and leave
  deterministic units no worse with `z` set to zero.

Energy models and diffusion are parked until there's a reason to need them.

## Reused and retired

**Reused:** token embeddings, `QuadBlock` and window grouping (`quad_latent.py`), `ResBlock` and `Down`
(`dense.py`), `MapDecoder`, `ActionEncoder` and `GameActions`, the rare-state set, the
split, player views and swapping, and two-hot targets (`direct.py`) for the uncertainty model.

**Retired:** lattice tokens; horizon inputs; h / g latent dynamics; the autoencoder as a goal. Hungarian matching stays, for
births only. The 40k model stays at `data/worldmodel/runs/ae_20261102_063320/model_step40000.pt`.

## Open decisions

1. Pointer bias between actions and targets: how it fits the compiled attention, and its cost.
2. Barlow Twins correlations: whether removing each row's mean over time is the default. It's the leading option, not yet decided.
3. Heavy discount values for critter and larva positions, and the query margin for births.
4. Whether the count in training comes from real births or from the prediction.

Settled: map-cell timelines at 32 and 16 only; no with / without action pairs; every-loop extraction; EMA latent
prediction as a residual, with Barlow Twins and a raw auxiliary head that decays; boosted position; both player views
plus observer neutrals, with no merge; point predictions; no entity crop by default; 56-frame sequences; births by
count plus queries with Hungarian matching; an alive/dead field on each row; `_gas_gaps` for workers in gas; critters kept as tokens;
map pool unchanged for now.

Rotary cos and sin are computed once per forward and shared by every block (2026-11-05, done once time became rotary too): bit-identical, 5.58 to 5.41 s an update, about 3%.
