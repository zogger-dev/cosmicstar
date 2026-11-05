# World model data notes

> Cleanup, 2026-10-30: the earlier JEPA trainers and their checks (`train_stage1.py`,
> `train_window.py`, `train_pairs.py`, `action_check.py`, `eval_units.py`, `probe_frozen.py`,
> `spatial_check.py`, `worldmodel/predictor.py`, `window_predictor.py`, `probes.py`), the 2v2
> slot-4 composition (`compose_view.py`, `validate_compose.py`), the engine re-matching ability
> table (`build_ability_table.py`), the scripted build-order bench (`zerg_bench*.py`,
> `bo_search.py`, `replay_build_order.py`, `builds/`), the mining probes and the original
> March scaffold were removed from the tree. The notes below still name them; they are in git
> history before commit "Clean up the tree".

## Replay ability ids → ability names

A replay command names its ability as `(abilLink, cmdIndex)`:

- `abilLink` is the ability's position in that build's ability catalog. It is **not stable
  across patches** (e.g. link 731 is `MothershipCloak` in 5.0.14–5.0.15 and `MorphToBaneling`
  in 5.0.17).
- `cmdIndex` is the button within the ability. It equals the API's `AbilityData.link_index`.

So the stable identity of a command is `(link_name, link_index)`, e.g. `("BarracksTrain", 0)`
for a Marine. The only per-build unknown is `abilLink → link_name`.

### Getting the link numbers for a build: the Editor export

The StarCraft II Editor can export the exact rules of the build a replay was played on,
including old builds (it downloads the replay's dependencies itself):

```
"C:\Program Files (x86)\StarCraft II\Support64\SC2Editor_x64.exe" -exportreplay "<replay>.SC2Replay" -exportreplaydir "<output dir>"
```

- No GUI interaction. It takes about 5 minutes the first time a build's data is fetched and
  about 10 seconds after that.
- The menu's *File → Export Balance Data…* only exports the **currently installed** build.
  Use the command line for anything else.
- `scripts/export_balance.py` runs it once per ruleset (build + data version + mods, from
  `scripts/dump_rules.py`) into `data/worldmodel/balance/<ruleset>/`. It records the outcome in
  `data/worldmodel/balance/status.json`.
- Errors land in `Documents\StarCraft II\EditorLogs\<time> ExportReplayErrors.txt`.
- Pass **absolute paths**: the Editor runs from `Support64\`. A relative output dir gets
  created under `C:\Program Files (x86)\StarCraft II\Support64\` and the export then fails
  with "Unable to export data [The system cannot find the file specified.]".
- **Some dependencies can't be fetched.** The error is "Unable to download replay dependency
  [hash]", even when the file is in the local Battle.net cache and Blizzard's depot serves it.
  The cause is unknown.
  - When it's the replay's **map**, another replay of the same ruleset on a different map
    works. `export_balance.py` tries one replay per map.
  - When it's the **balance mod**, the whole ruleset is out. This happens to two 95299 mods,
    `a825…` (60 games) and `3d46…` (5 games).
- `extract_states.build_manifest` skips replays whose ruleset `status.json` doesn't mark ok.
  As of 2026-10-19 that's 66 games that can't be exported (the two mods above, plus two
  one-game rulesets whose only map fails). Another 9 games are skipped because their ruleset
  was never dumped: they're custom-lobby games with a non-standard 6th dependency (`b6cc…`),
  and the engine refuses to launch them.
- Output: one XML file per unit. `<ability id="BarracksLiftOff" index="150">` gives a
  catalog name and its link directly. `<unit id="Marine" ability="161" index="0">` and
  `<upgrade ... ability= index=>` give the link of the ability that builds, trains or
  researches it, without the ability's name.

`scripts/build_link_table.py` turns the exports into `data/worldmodel/ability_links.json`:

1. It names every link, either directly or through what it produces.
   - Several abilities can make the same thing (`GatewayTrain` and `WarpGateTrain` both make
     Zealots), so a produce-only link takes the producing unit's own ability whose buttons hold
     the same items.
   - As a last resort it accepts a button name that merely contains the item's name (the
     Nuke's button is `NukeArm`). That match is limited to melee-available abilities whose
     names no other link already has.
2. It expands each link to all its buttons with the API's `AbilityData`.
3. It checks the result against the replay-vote table from `scripts/build_ability_table.py`.

For a new patch, rerun `scripts/dump_rules.py`, then `scripts/export_balance.py`, then
`scripts/build_link_table.py`.

## Training runs: checkpoints, pausing, memory

`scripts/train_window.py`:

- **Checkpoints:**
  - `model_latest.pt` is rewritten every `--save_minutes` (default 20), so at most that much
    work is lost.
  - `model_step<N>.pt` milestones are kept every `--save_every` steps.
  - Writes go to a temp file first, so a checkpoint is never half-written.
- **Pause:** create an empty file `PAUSE` in the run folder. The run saves and exits after the
  current step.
- **Resume:** `--resume <run folder>` uses the same settings, games and vocabulary, and appends
  to the logs. Checkpoints written before resume existed hold weights only; they resume "warm"
  (optimizer statistics rebuilt).
- **One data-heavy job at a time.**
  - A run holds its game pool in RAM (~180 MB per game). A second job loading games made
    Windows page memory, and GPU utilization went from flat to spiky.
  - On Windows, an over-full GPU is paged to system RAM. That cost ~40%, so the script sets
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Evaluation** (`eval.csv`, offline scripts):
  - Probes are evaluation only; they never train the model.
  - `scripts/probe_frozen.py` trains readouts from scratch on a checkpoint's frozen
    encodings. The probes trained during the first runs had a 7-unit bottleneck and understated
    the model.
  - `scripts/eval_units.py` reports forecasts in game units.

## Replay command events: what they mean

This was checked against the engine: replays were played back and each player's commands
reported by the engine were matched to the replay events (`scripts/build_ability_table.py`,
saved in `data/worldmodel/engine_actions/`).

| Replay pattern (same 8-loop window) | Commands the engine executed |
|---|---|
| `CMD` | 1 (100%) |
| `CMD_POINT` + `REPEAT` | 1 (99%) |
| `CMD_UNIT` + `REPEAT` | 1 (84%) |
| `REPEAT` alone | 1 (67%), 0 (33%) |
| `CMD` `REPEAT` / `REPEAT` `REPEAT` | 2 (most) |
| `CMD_POINT` alone | never occurs |

- `CMD` is a new command.
- `CMD_POINT` / `CMD_UNIT` changes the target of the player's last command and executes
  nothing on its own.
- `REPEAT` (`SCommandManagerStateEvent`, always state 1) executes the last command again.
  - After a retarget, it fires the command at the new target (spam right-clicks, shift-clicked
    waypoints).
  - On its own, it repeats the same command at the same target. A held or mashed hotkey is one
    `CMD` and then one `REPEAT` per key repeat, usually 0–2 loops apart (e.g. 9 Drones = `CMD`
    + 8 `REPEAT`s).
  - The ~33% of lone `REPEAT`s that execute nothing are attempts the game rejects (no larva,
    minerals or supply).
- `action_tokens.py` folds these into one token per executed or attempted command
  (bursts become a press count). The raw events on disk stay lossless.

### Things that did not work (don't retry)

- **Reverse-engineering link numbers from the catalog XML** (CASC data plus the depot balance
  mod). Links are handed out at an id's *first mention* in load order, not its declaration.
  First mentions include typed ability fields, literal `##id##`/`##abil##` templates, and some
  actor event strings. The best model reached about 71% of the known links.
- **Matching replay commands to engine-reported actions** (`scripts/build_ability_table.py`).
  It works for abilities players actually use, but rejected or reordered commands create stray
  votes (about 0.05%), and keys with a single vote can be wrong (95841 link 260).
- **sc2reader's datapacks** come from the same Editor export, but their names for
  build/train/research abilities are made up (e.g. `BarracksFlyingBuild`), and new builds lag.

## Player views (`extract_states.py --views observer,players`)

Each game is extracted from five viewpoints: the observer (slot 0, no fog) and each player
(fog, snapshots of remembered structures, visibility and creep maps per frame).

- **The engine's slot-4 view crashes** (ACCESS_VIOLATION) at the game's first research order,
  by any player. This held in 31 of 31 replays, on every build. It is not caused by the
  WCS interface or by any view option.
- **Slot 4 is composed instead** (`scripts/compose_view.py`): slot 3's view, with slot 4's own
  units taken from the observer's rows, and slot 3's units shown as an ally sees them.
  - Teammates share vision, creep and snapshots. Snapshot content is identical; only their
    tags differ per viewer.
  - An ally's units hide the following (`ALLY_HIDDEN`):
    - harvesters
    - weapon_cooldown
    - engaged target
    - add-on reference
    - cargo_used
    - production queue
    - transport passengers
- **Validation** (`scripts/validate_compose.py`, every frame). The composed view matches the
  engine's real view:
  - Rhoskallian LE (88), T+P vs Z+Z: p2 composed from p1, and p1 from p2, are 100% identical.
    So is p4 from p3, compared up to the crash.
  - Yellowjacket (100), Zerg: 99.999% identical. The only differences are `z` (unit height)
    of moving units, which differs between separate engine runs.

## Does the predictor use the actions? (2026-10-20)

`scripts/action_check.py` compares, for held-out windows, the prediction made with the real
action stream against predictions made with action chunks borrowed from other games (same
seats) and with no actions at all. Both the 30s observer run and the player-view run scored
at chance: recall@1 ≈ 1/(K+1), losses with real / borrowed / no actions equal to three or
four decimals, and the prediction moving ~0.1% of the residual when the actions change.

Why, by construction: at position k the predictor only received actions up to k, whose
effects are already visible in the frames up to k, while the actions during the interval it
was predicting (k → k+h) were never inputs. A ridge probe (scratchpad) confirmed the
information exists: hand features of the actions *during* the interval raise the held-out
R² of the 10 s target change from 0.74 (state only) to 0.80.

### Latent-action mode (`train_window.py --latent_action_s 5`)

The predictor is asked one step (D seconds) ahead at every frame, given a latent action per
player for the interval after that frame. The latent action has two sources that must agree:

- the command encoder's summary of the commands in [k, k+D) (`ActionEncoder(..., ahead=)`);
- an inverse model (`inverse_model.py`) reading the frame and the frame D later: the change.

They're tied by InfoNCE over the batch (the same window's overlapping intervals are not
negatives), so the latent action must explain the change and be recoverable from the
commands alone (it can't copy the next frame). Either source drives the predictor in training
(`--z_source mix|cmd|inv`). Evaluation prints, per set: agreement top-1, and the
counterfactual check (predictor given the real commands vs another window's: recall@1 vs
chance 1/B, MRR; loss with real / borrowed / none / the inverse model as oracle).

### The EMA target over training (frozen readers, game units)

`probe_frozen.py` on the player-view run's checkpoints: per-player quantities read better and
better (army value from unreadable to below the spread by step 3000), while per-cell counts
read steadily worse (units per cell RMSE 1.00 → 1.38 over steps 1000 → 3000, monotone on both
held-out sets): the moving target trades spatial detail for per-player content. The in-run
"now" baseline (probes trained on observer encodings applied to player-view encodings) is
invalid once the views diverge; only readers trained per representation compare fairly.

### Pair-based simulator (`scripts/train_pairs.py`, 2026-10-20)

Windows are gone for the observer simulator. A sample is a pair of moments D seconds apart:
3 context frames ending at t (0.36 s apart, to read motion, since frames hold no orders), all
players' commands in [t, t+D), and the frame at t+D. D is log-uniform in [1, 15] s per batch
and conditions the command summaries, the inverse model and the predictor's single head. The
batch is 32 pairs from random pool games, so the agreement InfoNCE has 32 genuinely different
moments as negatives (the window version's negatives were overlapping intervals of 4 games).
The prediction loss is divided by the copy-the-present loss so all horizons weigh the same;
command pointers are static (`GameActions.window(static_pointers=True)`: no future unit
state); the predictor is driven by the command-side latent only (`--z_source cmd`), the
inverse model learns through agreement and serves as the oracle in evaluation.

It is ~20x faster per step than the window version (0.32 s vs 7 s, 4 GB vs 22 GB) because the
block-causal predictor runs over 3 frames instead of 169.

Next on this piece: re-extract the observer frames with unit orders and use them as state
instead of the context frames (orders are the already-decided change; the latent action is
then only the new commands).


### Stochastic latent actions (`worldmodel/latent.py`, 2026-10-23)

Why: a predictor given only the present and trained with a regression loss predicts the
average of the futures that present allows (a building that is half spire, half hydralisk
den), and the JEPA encoder then learns to stop representing whatever was unpredictable, so
the averaging is baked into the representation, not just the prediction. In the observer view
with both players' commands the game is deterministic and this never shows; it shows the
moment commands are withheld (the opponent's, under fog; everyone's, in the test below).

What: the latent action z per (pair, player) becomes a random variable with three estimates
that differ only in what they see: the posterior q(z | present, future) (the inverse model,
now a Gaussian), the command distribution q(z | present, commands) (the command encoder, now a
Gaussian; nearly a delta for the observer) and the prior p(z | present) (new head on the CLS
and the player's slot: a mixture of `--prior_k` diagonal Gaussians, so it can hold several
branches of the future from one state). The predictor is driven by a *sample* (`--z_source`:
posterior, commands, or a coin per pass). Losses: the prediction loss as before;
`--beta` x KL(posterior || prior) per nat above `--free_nats` (one-sample Monte Carlo, the
sample that drove the predictor; the floor keeps the decision itself free of charge; beta
ramps over `--beta_warmup` steps); `--align_weight` x KL(posterior || commands) in closed form;
`--usage_weight` x (log K - entropy of the batch-mean component responsibilities), the
mixture-of-experts load-balancing idea, so components do not die; `--div_weight` x mean
|cosine| between a state's component means, a repulsion, off by default. Capacity is set by
beta and the floor, not by `--d_z`: unused dimensions collapse onto the prior for free, too
few dimensions bring the averaging back; 128-256 is the planned size once 10 s chunks carry
positions (drops, nydus) rather than just discrete choices.

Diagnostics, printed per evaluation set and horizon: the loss with the prior's mean vs one
prior sample vs the best of `--eval_samples` samples vs the posterior (oracle) and vs copying
the present; KL in nats; active units (posterior means that vary across data); effective
components used (exp of the entropy of the mean responsibility). A gap between the prior's
mean and best-of-K says the future was multimodal and the prior spans it; active units
climbing to d_z with KL at the floor says the latent is saturated; components used falling
to 1-2 says the mixture collapsed (raise `--usage_weight`, or `--div_weight`).

The experiment this is for: `--no_cmd` withholds everyone's commands, so the predictor sees
the present and z only, the posterior drives training and the prior drives evaluation. The
observer view is the clean testbed: all ambiguity is decisions, none is hidden state. The
calibration check on top: a frozen probe that reads "spire present" from a representation, run
on each sampled prediction 60 s before the tech choice, should agree with the empirical
frequency in the replays for that state. Old checkpoints load as `--prior none` (point
latents, the previous model).

### Two things that broke the action channel, and the first result (2026-10-20)

1. **Owner codes as a signature.** With random owner codes drawn per sample, the in-batch
   agreement and counterfactual losses matched samples by their codes (84–100% in training,
   chance on every held-out set including seen games). Fix: one code draw per batch
   (`GameFrames.owner_map(codes=...)`). Any in-batch comparison needs this.
2. **Soft negatives.** With negatives from other games only, identity (map, races) discriminates.
   Fix: batches of 8 games x 4 moments, plus a counterfactual loss on the predictor (the real
   latent action must give the best prediction among borrowed ones, softmax on the loss relative
   to copying the present).

With both (`pairs_20261020_220134`, 55 games, 5000 steps, 36 min): on unseen games the 15 s
loss with the real commands is 0.049 vs 0.181 with borrowed commands and 0.192 with none
(unseen maps 0.052 / 0.173 / 0.226); recall@1 of the real commands among 8 candidates 53–59%
(chance 12.5%); loss 17–19% of copy-the-present. Agreement between the inverse model and the
command encoder is still weak (7–13% top-1). `--eval_only` on a checkpoint also prints the
split against same-game vs other-game borrows.

### Unit orders in the state (extraction launched 2026-10-20)

`extract_states.py` now keeps every unit's orders (`ord_*`, ragged like production: ability,
target point, target unit id, progress). The tokenizer adds the first order to the entity
token (`ent_order` ability embedding + `ent_ord`: has one, target offset, targets a unit,
progress, queue length; zeros for older files); the encoder's `use_orders` flag is turned off
when loading checkpoints trained without them. Orders are the already-decided part of the
coming change, so a single frame with them is close to a Markov state and the latent action
covers only new commands. The two-pass extraction (all missing games, then the 80 old files
redone; old copies in `obs_noorders/`) logs to `data/worldmodel/obs/extract_orders.log`.

**Correction (same night):** against borrowed chunks from the same game a few intervals away
(`NEAR` offsets in `--eval_only`), the real commands barely win: adjacent chunks score within
6–8% of the real one at 5 s and 15 s, and real beats both adjacent chunks only 30–42% of the
time (chance 33%). The channel had learned the player's game stage (random same-game moments
are minutes apart), not the specific commands. `--adjacent 2` puts consecutive intervals of
the same game in every batch so the agreement and counterfactual losses can only be won by the
specific commands; the near-moment check is the metric that counts from here on.
With `--adjacent 2` (`pairs_20261020_224801`, to step 3831) the near-moment check did not
move: adjacent chunks score within 1–2% of the real one and "beats both adjacent" stays at
chance, while the agreement between the inverse model and the command encoder improved to
10–20% top-1. A likely confound: commands lag their effects, so the previous interval's orders
execute during the predicted interval and genuinely predict it; with orders in the state that
part is explained by the state, and only new commands are left for the latent action. That is
what the orders extraction is for.

### Orders as state: first result (2026-10-21)

Same configuration as the adjacent-negatives run, on 302 games extracted with orders
(`pairs_20261021_141458`): the near-moment check moved off chance for the first time. Real
commands beat both adjacent same-game chunks 45–56% of the time (chance 33%; 27–45% without
orders), beat all chunks within ±3 intervals 25–34% (chance 14%), with chunks 2–3 intervals
away scoring 6–19% worse than the real one. Two caveats drive the follow-up runs: the run also
had 302 games instead of 55 (`--no_orders` on the same games attributes the gain), and the
loss-vs-copy ratio got worse (46% vs 26% at 5 s) because orders at t+D are partly the players'
future commands, i.e. policy in the target (`--target_no_orders` keeps orders in the input
state only). `--context 1` tests whether orders make the context frames unnecessary.

**Attribution (same day, four runs on the same 302 games):** the specificity came from orders
in the *target*. With orders kept in the input but removed from the target
(`--target_no_orders`, so the target is the physical state), the near-moment check is back
at chance (real beats both adjacent chunks 25–44%, chance 33%); without orders on 302 games
it is at chance as well, so the data increase alone did nothing; and orders in the input do
not improve physical prediction over three context frames (loss-vs-copy 28% / 17% either
way). Commands in [t, t+D) determine the orders at t+D almost by definition, so an
order-bearing target lets the model separate chunks by predicting what the player will be
ordering (policy), not what happens. On the physical 5–15 s future the interval's specific
commands are still not separable from the adjacent interval's in any configuration tried.

### Per-cell diagnostic and the count head (2026-10-21, evening)

`scripts/spatial_check.py` compares, in the model's own space per grid cell, the prediction
driven by the real commands with the adjacent intervals' commands and with copying the
present's slot, on the cells whose true unit count changes. Readers trained on true
encodings extrapolate wildly on predicted vectors, so only the embedding-space block is
trustworthy. Findings: with orders in the input the real chunk beats the adjacent chunks on
changing cells at 15 s (70% / 62% of pairs on unseen games), nothing at 5 s; but no
physical-target model beats copying the present's slot on the grid (worse on static cells,
parity on changing ones): all the measured gains were in the CLS and player slots.

`train_pairs.py --cnt_weight` adds a head on the predictor's grid outputs predicting each
cell's unit / army / structure counts at t+D (log1p, from probe_targets), evaluated in game
units with no readers. Run `pairs_counts_v1` (502 games): army counts per cell on changing
cells 4.1–4.6 RMSE vs 4.8–5.5 for copying the present, the first spatial improvement over
"nothing changes", from momentum (context frames + orders); the commands add nothing specific
(for army, real = adjacent = no actions). The head decodes absolute counts, so static cells
inherit ~2 units of read-out error where copying is exact; a residual form on the present's
true counts is the fix.

## Vision after death (player views, 2026-10-22)

Measured on the 33-game player-view sample (`data/worldmodel/pov`) and a 1-loop-step
extraction of one game: when a vision source dies, the engine's visibility map goes dark in
the same game loop (65 of 65 isolated overlord deaths lose the whole sight disc in the same
step; loop-exact at step 1), structures and resources flip to snapshots in that loop, and the
dead unit has no row in its death frame (it is present with hp > 0 the frame before and then
appears only in `dead_units`). Enemy mobile units that were being seen, however, stay in the
observation as visible units, moving and updating, for 47 *observation steps*: the same queen
after the same overlord death vanished after 47 loops (2.1 s) in the step-1 extraction and
after 47 frames = 188 loops (8.4 s) in the step-4 extraction. A human at normal speed gets
about 2 s of this "death vision"; the step-4 player views show four times too much (a step-8
run still tracked the queen 14 s later, and stepping the engine one loop at a time without
observing did not help: the timer counts observation requests). Fix: `extract_states.py
--substep 1` (now the default for player views) requests an observation every loop and
discards it unparsed; the kept frames then match a 1-loop-step extraction exactly at every
loop (visible units identical, all three views, 200 s) for about 15% more time. The 33-game
`data/worldmodel/pov` sample predates the fix and leaks; re-extract it before reuse.

Correction (2026-10-23): the linger is the game's **death reveal**, a real mechanic
(Liquipedia, Sight: vision "continues to be provided for a brief time after the unit or
structure blinks, is warped, or is destroyed"; the Editor exposes Death Reveal Duration /
Radius / Type per unit). In a live game through the API with two participants, an overlord
killed in combat keeps its killers visible for 39–40 game loops (about 1.8 s) whether the
engine is stepped 1 or 4 loops per observation: game time, as it should be. A debug kill
bypasses the reveal, which is why an earlier test wrongly saw nothing. The API's visibility
map does not include the reveal (the sight disc goes dark at once while the units stay
reported), so a visible unit on a dark cell is the reveal's signature in the data, and a mask
on map visibility would wrongly delete it. Only replay playback mis-times the reveal, counting
observation requests instead of loops; one request per loop (`--substep 1`) restores ~2.1 s.
Refined in a live game: the reveal is "vision through the fog", not true vision. The
visibility map goes dark at once and nothing can be targeted, but units keep being reported
with live positions for about 40 loops (1.8 s). Revealed are all enemy units within about 6
cells of the death spot (passive units at 2, 5 and 6 cells stayed; 7, 7.5, 8 and 10.5 vanished
at once) plus the attackers at any distance (vikings firing from 8.5 cells stayed). Replay
playback counts 47 observation requests instead of ~40 loops, so per-loop requests overshoot by
about 0.3 s. In the data the reveal's signature is a visible unit on a dark cell; it is real
information and must not be masked away.

Two reveal rules, measured in live games with two API participants (2-loop polls):
**attack reveal**: a unit that attacks one of the player's units is reported as visible for
46–48 loops (2.1 s) after each shot, even from cells the player has never seen (sieged tanks
12 cells from a queen, map value 0 at their cells, visible while firing and 46 loops after
she died); the map stays dark, so these units are seen but not targetable. **Death area
reveal**: units within a per-type radius of the death spot stay visible for a while (drone
death: 4 cells stayed, 5 vanished, 64 loops; overlord death: 6 stayed, 7 vanished). The
Editor's balance export has none of these fields (no Death Time, no Death Reveal); they are
in the game's UnitData.xml inside the CASC archives.

## Deferred: the game's unit data as a model input (2026-10-23)

Idea (to revisit, not to build now): feed the game's own unit data into the model's inputs
rather than letting it absorb unit properties into a learned type embedding, so that changes in
timings, ranges, speeds and the like change the model's behaviour immediately and the entity
network can attend to units by their properties. Concretely, the entity token's type input
would carry (or be replaced by) a per-type stat vector.

What exists: the Editor's balance exports per ruleset in `data/worldmodel/balance/<build>_<hash>/`
(one XML per unit: life, armor, shields, cost and build time, movement speed / acceleration /
turn rate, sight radius, cargo, supply, attributes, weapons, abilities with their costs and
requirements; produced by `scripts/export_balance.py`). The extraction already keys each game to
its ruleset, so the vector can be looked up per game.

What is missing: everything the balance export leaves out of CUnit, notably Death Time and the
Death Reveal fields (Duration, Radius, Type, Filters), plus ability and effect data. Those are in
UnitData.xml (and the other GameData XMLs) inside the game's CASC archives under
`C:\Program Files (x86)\StarCraft II\SC2Data`. Reading them needs a CASC reader (CascLib /
CascView); an earlier attempt to install CascLib on Windows failed. The reveal mechanics
themselves need no table to be learned: they show up in the player views as visible units on
dark cells (attackers for 2.1 s after each shot, units near a death for a per-type radius), so
the encoder can learn them from the data; the table would make them explicit.

## Replay sources and their terms (checked 2026-10-23)

- **drop.sc** is a front end of Sc2ReplayStats (downloads are served by sc2replaystats.com).
  It lists only the 25 most recent public uploads (about 8 new games a day in October 2026,
  mostly 2.5–3k MMR), replay pages are `/replay/<id>`, and it has no terms or robots.txt of
  its own. The governing text is Sc2ReplayStats' Terms and Conditions
  (`sc2replaystats.com/account/termsAndConditions`, "No data scraping or automated
  collection"): no scraping, crawling, downloading or collecting by scripts, browser
  automation, headless browsers or bulk tools without prior written permission, no bypassing
  of rate limits or bot checks; permitted access "may be subject to separate limits, fees,
  API terms". Replay pages for anonymous visitors sit behind a proof-of-work "human
  validation" page. So: manual downloads only, or written permission
  (contact_us@sc2replaystats.com, Discord linked from the site).
- **Spawning Tool** (`lotv.spawningtool.com`): ~85k replays with patch / matchup / pro-only
  filters, but robots.txt disallows `/*/download/` and `/zip/`, so no automated downloading of
  the listing. Its **Replay Packs** page (`/replaypacks/`) links 113 tournament packs
  (2015–2026, Google Drive) meant for download; the 2025–2026 packs (HomeStory Cup XXVII–XXIX,
  EWC 2025, FEL Cracow 2025, DreamHack Dallas 2025) are the ones likely to match the installed
  base builds (94137, 95299, 95841, 96883, 98310; older patches need binaries we don't have).
- **SC2ReSet** (Zenodo, doi 10.5281/zenodo.5575796, v2.0.0 March 2025): 71 tournament replay
  archives 2016–2024, 4.5 GB, licence "Other (Non-Commercial)" under Blizzard's EULA; the
  companion SC2EGSet (parsed game states) is CC BY 4.0. Mostly pre-5.0.14 patches.
- **Blizzard replay packs** (s2client-proto README; much more via the replay API, see [REPLAY_CORPUS_blizzard.md](REPLAY_CORPUS_blizzard.md)): only 3.16.1 packs remain listed; Linux
  headless builds exist for 3.16–4.10 only. Not usable with current-patch models.

### Downloaded tournament packs (2026-10-23)

57 zips and 5 rar archives in `C:/Users/prade/Downloads/sc2 replays`, about 13.4k replays
(2015–2026). Installed base builds: 93333, 94137, 95299, 95841, 96883, 98310. On installed
builds right now, ~606 replays: HomeStory Cup XXVII (176, 5.0.14 / 94137), EWC 2025 (113,
94137), FEL Cracow 2025 (77, 94137), DreamHack Dallas 2025 (64, 93333), HomeStory Cup XXVIII
(176, 5.0.15 / 95841). HomeStory Cup XXIX (175) is on 5.0.16 (97425), the excluded, reverted
patch. Everything 2024 and earlier (5.0.13 and below, ~12k replays) needs older builds from the
Blizzard depot; deferred. Seven 2015–2016 packs (~650 replays) fail the header parse (older
replay format); deferred with the builds.

### Storage plan for every-loop extraction (measured on one 15-min game)

Observer unit arrays: raw 298 MB per game at step 4; the current `.npz` (deflate) 42 MB; the same
arrays with xz 31 MB; rows regrouped into per-unit time series with positions quantized to
1/256 cell and delta-coded along time, then xz: **11.7 MB** (3.6x smaller than today, stdlib
`lzma`, no new dependency). Player views: the visibility map is 209 MB raw, 4.6 MB deflate,
**0.41 MB xz** (0.28 MB as frame deltas); the unit arrays (20 MB deflate) can be replaced by a
per-frame visible-unit bitmask over the observer's rows (~0.4 MB) plus last-seen frames for
snapshots and the hidden-field rules already used by `compose_view.py`, i.e. a player view at
~1–2 MB instead of 25 MB. Projected at one frame per loop: ~50 MB per game for the observer and
all player views, so ~30 GB for the 606 playable replays and ~700 GB for all 13.4k. Engine
time, not disk, is the binding constraint for the old packs (weeks of extraction).

### Discord (PiG's server replays)

Discord Terms of Service, "Restrictions on your use of Discord's services": no "scraping our
services without our written consent, including by using any robot, spider, crawler, scraper,
or other automatic device, process, or software"; no "selling, licensing, or otherwise
commercializing content or data obtained from our services"; "Other people's content. You may
not use this content without that person's consent, or as allowed by law." Automating a normal
user account ("self-bots") "is forbidden, and can result in an account termination". A bot
account via the API falls under the Developer Policy: #20 "Do not mine or scrape any data,
content, or information available on or through Discord services"; #21 "Do not use message
content obtained through the APIs to train machine learning or AI models ... unless express
permission is granted by Discord"; #2 no processes on a server's behalf without permission.
So: no self-bot, no client automation, and a bot that collects attachments for training needs
Discord's express permission as well as the server's. The workable route is asking PiG's team
for their replay archives (they already distribute batches via Google Drive, e.g. the
`pigosaur 62` folder export) with consent for research use, or manual downloads by a member.

### Protobuf parser (2026-10-23)

The s2clientprotocol wheel's `*_pb2.py` modules are old-style (direct descriptor construction),
which the C++ `upb` implementation refuses, so the extraction ran the pure-Python parser (44%
of frame time). Each old module embeds the serialized FileDescriptorProto of its `.proto`, so
`scripts/regen_s2proto.py` rewrites them in the modern builder style from that blob alone: no
`.proto` sources, no protoc, no download; originals kept as `*.py.orig`, `--restore` puts them
back, `--check` verifies. A 500-unit observation parses in 0.09 ms (upb) vs 13.7 ms (Python).
End to end on the 200 s test game: step 1 per view 34–45 s vs 84–100 s; output identical
(every array equal with NaN-aware compare; only snapshot units' client-assigned tags differ
between any two runs). `extract_states.py` now picks upb automatically when the bindings are
modern (`protobuf_impl()`); the environment variable still overrides.

## Linux plan (2026-10-23, to test on a Linux box)

Why: training (torch.compile / Triton, prebuilt flash-attention, fork dataloaders), and the
player side: several SC2 clients each with its own virtual display and virtual keyboard /
mouse (one X server per client, Xvfb for headless API runs, Xorg dummy or a Wine virtual
desktop `explorer /desktop=<name>,<WxH>` for rendered play; input per display via XTEST /
xdotool or uinput). Windows has one input queue per session, so the same needs VMs there.
Also the generalization test the user wants: hold out every game played on a community
balance mod (new and redesigned abilities) and test the model on it; the extraction already
keys each game to its ruleset and skipped 76 replays with non-standard rulesets, which are
the candidates.

Engine facts: Blizzard's headless Linux builds stop at 4.10; every 5.0.x client is Windows
only, so on Linux it runs under Wine/Proton (python-sc2 documents this: `SC2PF=WineLinux`,
`WINE=<wine binary>`, `SC2PATH=<install>`; Lutris or Battle.net-under-Proton installs).
pysc2's run configs have no Wine variant; a small one (launch `wine SC2_x64.exe -listen
127.0.0.1 -port N -displayMode 0` from the right `Versions/Base<build>` folder) is needed
for the extractor there.

What to verify on the box, in order:
1. Battle.net + SC2 install under Proton/Wine; the game starts and plays (rendered).
2. Headless API run under Wine: start the exe with `-listen/-port/-displayMode 0`, connect
   with pysc2's RemoteController, play back one of our 5.0.17 replays as observer; compare
   the extracted arrays with the Windows extraction of the same replay (must be identical).
3. Several base builds: the Windows install keeps `Versions/Base93333 .. Base98310`; copy
   them over (or fetch older ones from the depot) and confirm each launches under Wine.
4. A modded game (the balance mod) plays back: needs the mod's data in the Battle.net cache
   (`Cache/..`), which the client fetches on first open.
5. Throughput: loops per second per client vs Windows (~22 loops/s of simulation per 4-loop
   frame costs 20–50 ms here), and how many clients fit in RAM/CPU.
6. Two clients on two X displays with independent xdotool input, both rendered.

WSL2 is the fallback for training only: Triton runs there as far as community reports go
(the usual snag is `LD_LIBRARY_PATH=/usr/lib/wsl/lib` for libcuda.so), but reports of CUDA
init failures on new GPUs exist and nothing official confirms support; a native box avoids it.

### Linux results (2026-10-23, RTX 5090 box, Ubuntu, Lutris + GE-Proton 11-11)

Steps 1 and 2 of the plan pass; 3 and 4 wait on files from the Windows machine.

- Install: Battle.net under Proton via Lutris (prefix `~/Games/battlenet`, launched through
  `umu-run`); the game plays rendered. The prefix has one build, `Versions/Base98310` (5.0.17).
- Headless API run: `cosmicstar/envs/wine_run_config.py` is a pysc2 run config that starts
  `umu-run .../Base<build>/SC2_x64.exe -listen 127.0.0.1 -port N -displayMode 0 ...` from
  `Support64`, with `-dataDir`/`-tempDir` rewritten to `C:\...`/`Z:\...`. Importing it
  registers it (priority 2 on Linux); `extract_states.py` and `ReplayParser` import it.
  API port open 6 s after launch; a 5.0.17 replay as observer with fog off steps at
  ~1500 loops/s raw (step(16)+observe, 215 units: 11 ms per frame).
- Full extraction (`extract_states.py --worker` on a 6.3-min 5.0.17 game, 530 frames,
  124k unit rows): 18 s, 11 s of it waiting on the engine (13 ms per 16-loop frame, on par
  with Windows). The very first launch took 96 s (Proton/umu and DXVK shader-cache warm-up);
  every launch after that was fast. Two runs of the same game are identical in all 67 arrays.
- The 12 build-98310 games of `my-recent-games` with 2 workers (`extract_states.py`, observer
  view): 11 passed first try, 15,920 frames / 10.4M unit rows / 155 MB in 1276 worker-seconds
  (12.5 frames/s = 200 loops/s per worker, games of 18-212 s). One game failed with
  "Websocket timed out" during an observation and passed alone (96 s); the worker now retries
  timeouts like it retries InvalidReplayData.
- Maps: the API client does not fetch a missing map (`ResponseReplayInfo ParsingError:
  Failed to open the map archive ...Cache\\xx\\yy\\<hash>.s2ma`); only the game UI does. Blizzard's
  depot hosts that every tool uses (`http://<region>.depot.battle.net:1119/<hash>.<type>`, in
  `download_maps.py` too) no longer resolve, but `SC2_x64.exe` 5.0.17 carries the current one:
  `https://<region>-s2-depot.classic.blizzard.com/<hash>.<type>` (strings: "-s2-depot.classic").
  It serves cache handles by SHA-256 (the handle = 4 bytes type + 4 bytes region + 32 bytes
  hash). `cosmicstar/envs/depot.py` fetches into the prefix's cache; the six maps of the 13
  build-98310 games took 3 s. `extract_states.py` now fetches a replay's missing map and mods
  while building the manifest (`--no_fetch` turns it off): on `pig-viewer-replays/
  replay-analysis` it pulled 4 maps, after which its 13 build-98310 games were skipped by the
  next gate, "ruleset not exported" (their mod set has no Editor balance export, which needs
  Windows). Older builds still need `Versions/Base<build>` from a Windows install (plan step 3); the Battle.net product `s2` has one build on TACT (5.0.17) and the
  `Versions` folders are not on the CDN as such.
- Windows vs Linux (plan step 2, partial): the only Windows extractions of cached-map games
  predate the current extractor (45 arrays, observed as slot 1, `--until 600`). Re-extracted
  on Linux with `--until 600`: `frame_loop`, `frame_start` (all but the final total) and every
  `p_*` tracker array identical; over the first 393 frames (137,626 unit rows) `u_type`,
  `u_owner`, `u_x`, `u_y`, `u_hp`, `u_hp_max`, `u_shield`, `u_shield_max`, `u_energy`,
  `u_build`, `u_contents` identical. Differences are extractor-version changes (`u_id`
  numbering, `u_flags` bits, `u_harvesters` int8 vs int16, the upgrade filter) plus two extra
  rows in the final frame (observer vs slot-1 view) and 11 `u_z` rows off by <= 0.09 (zergling,
  roach; cosmetic). A same-version Windows extraction is still needed for a bit-exact check.
- Editor balance export (`export_balance.py`) under Proton: `SC2Editor_x64.exe -exportreplay
  <Z:\\...replay> -exportreplaydir <Z:\\...out>` through `umu-run` from `Support64` works; the
  Editor's logs land in `<prefix>/drive_c/users/<user>/Documents/StarCraft II/EditorLogs`.
  End to end on `my-recent-games`: `dump_rules.py` (engine, 2 rulesets) then `export_balance.py`
  (Editor, 8 s and 20 s per ruleset after the first warm-up run of ~90 s). The Linux export of
  98310_52f09e46 against the Windows export of the same ruleset: same 1597 files, 1577
  byte-identical, the other 20 differ only in the `source="<map>..."` attribute (a different
  replay was exported). The locale `S2Strings.xml` files come from the map/mod data, not the
  install's locale: a ruleset exported from a 1-locale game had 11 fewer. So the export gate
  in `build_manifest` can be cleared on Linux.
- Python 3.12: `s2protocol.versions` imports the removed `imp` module; `regen_s2proto.py`
  now patches it (importlib) alongside the protobuf bindings. s2protocol is at
  5.0.17.98310.0 so `versions.build(98310)` exists.
- Open: `minimize_client_window` is a no-op on Linux (the 20% win from minimizing is not
  taken; needs xdotool or an Xvfb display per client), and plan steps 3-6.

### Request costs and the headless question (2026-10-23)

The Windows client has no headless mode: `-displayMode 0` (what pysc2 passes) means windowed,
not off; only Blizzard's Linux build is headless. Minimizing the window after launch made the
engine step about 20% faster (5.8 vs 7.1 ms per 4-loop frame); a tiny window changed nothing.
The extractor now minimizes its clients.

Per 200 s of the test game (player view, upb parser, nothing stored): step(4)+observe 8.6 s;
4 x step(1)+observe 8.9 s (step calls are nearly free); 4 x [step(1) + unparsed observation]
15.7 s (each observation request costs ~2 ms of engine serialization) and keeps the fog timers
right; 4 x [step(1) + RequestPing] 9.3 s but LEAKS (a ping does not advance the client's
timers); step(1)+observe every loop 16.0 s, also right. So with the fast parser, extracting
every loop costs the same engine time as the fog fix already does, and parsing all of it is
~0.1 ms per observation. Pings are now skipped for the observer view (no fog).

## Design note: skill / APM conditioning (2026-10-23)

Requirement (user): the action decoder (and possibly the models before it) takes an APM
limiter as an input, so the generated actions differ with the intended "skill level" rather
than always imitating the data's average player.

What the data already carries per player and game: MMR (replay metadata, in `meta.players`),
race, result, and the full command stream with loop stamps (`data/worldmodel/actions/`:
`loop`, `player`, `kind`, abilities, targets, selections, control groups, camera), from which
realized APM/EPM per player and per time window follow directly (raw actions vs effective
ones: selection spam and repeated commands are a large share of high APM).

How it would enter the models:
- Conditioning features on each player token: MMR and realized APM over the recent window
  (and optionally the whole-game average), given during training as the true values of the
  player who played; at inference the knob is set to the desired level. APM is a consequence
  of skill, not its cause, so conditioning on it reproduces the action distribution of that
  level, mistakes included, which is the intent.
- In the simulator / belief models ("even before action decoding"): the same features on the
  player tokens let predicted dynamics reflect the players' skill (how well fights are
  microed, how fast production is re-queued). For the belief model the opponent's APM is not
  observable in-game; it is a hidden variable the model estimates from the observed play, while
  MMR is shown at the loading screen and is a fair input.
- In the decoder, as a hard budget as well as a condition: actions per window with explicit
  "no action" steps, and minimum spacing between actions, so a limiter cannot be exceeded even
  when the conditioning is off-distribution.

Open: whether to count raw or effective actions for the limiter, how to window it (per minute
vs per fight), and how the inverse model's latent action (per interval, all players) relates
to a per-player budget.

## Direct state-space pilot (`scripts/train_direct.py`, `worldmodel/direct.py`, 2026-10-30)

Design decision: the lowest level of the world model predicts *state*, not embeddings. SC2
state is already an abstraction (entities with exact attributes), so regressing it wastes no
capacity on unpredictable detail, needs no target encoder, no EMA and no collapse guard, and
the targets come free from the extraction. Given everyone's commands the engine is
deterministic; the only uncertainty in the observer view is the commands themselves. JEPA
(learned targets) is kept for the levels above this one, where unit motion is effectively
random at the horizon in question.

The pilot runs on the existing observer extraction (`data/worldmodel/obs`, 4 loops per frame,
commands per loop in `data/worldmodel/actions`): horizons k = 1..14 frames (0.18-2.5 s).

Model (`worldmodel/direct.py`):
- `RoPEEncoder`: the state encoder's token embedding (unit / owner / position / numbers /
  production / order tables, reused) under self-attention with **2D rotary positions in map
  units** (half of each head's pairs rotate with x, half with y, wavelengths 2-256 units), so
  a score depends on the signed offset between two things. CLS and player tokens sit at the
  origin. No grid or window readouts.
- `ActionEncoder` (reused): the window's commands [t, t+k) in loop order through a causal
  transformer; the predictor cross-attends to these tokens, and one summary per player feeds
  the prior.
- `DirectPredictor`: the encoded present + a horizon embedding + z, through blocks of RoPE
  self-attention and cross-attention to the action tokens (a learned "no commands" memory
  token is always present).
- `Probes`, per entity at t+k: exit (present / died / gone), position offset per axis (two-hot,
  81 bins over +-20 map units), change in hp / shield / energy / build fraction (two-hot, 65
  bins over +-1), the current order's ability (softmax over the vocabulary). Continuous
  targets are changes from t, so "nothing happens" is the same target for every unit and
  persistence is the zero bin. Two-hot (MuZero) keeps a left-or-right choice bimodal.
- z per (sample, player), `--d_z` 32: posterior = `InverseModel` over the present's and the
  future's tokens (the oracle), prior = `PriorHead` over the present's CLS, the player slot
  and the player's command summary or a learned "withheld" embedding. KL(posterior || prior)
  above `--free_nats` is charged with `--beta` (warm-up). The predictor is driven by a
  posterior sample in training; evaluation runs it with the posterior mean, a prior sample
  and the prior's mean.
- Action dropout (`--act_drop` 0.5): half the samples get an empty command window and the
  prior is told so. **The test of the design**: with commands given the future is determined
  and KL should fall to ~0; with them withheld z must carry what the players did. Both KLs
  are printed per horizon.

Batches: `--frames` present frames x `--hor` horizons each; the present is encoded once per
frame and shared by its horizons. Births (units at t+k that did not exist at t) are not
predicted yet; their count per window is printed so the ignored mass is visible. 32 x 2 at
d=256 runs at ~0.25 s/step and 7.4 GB.

### First result (2026-10-30, two 4000-step runs, 32 frames x 2 horizons, pool 48 games)

Run 1: `--beta 0.05 --free_nats 0.5 --beta_warmup 300` (`data/worldmodel/direct_pilot.log`).
Run 2: `--beta 0.02 --free_nats 4 --beta_warmup 1500` (`direct_pilot2.log`, checkpoint
`runs/direct_20261030_123137`). Both ~0.25 s/step, 7.7 GB. Unseen games, run 2, with the
same states evaluated with their commands and with commands withheld:

| horizon | pos err (persist) | moving units (persist) | commanded units (persist) | order acc (persist) | exit acc (persist) | died recall |
|---|---|---|---|---|---|---|
| 0.18 s | 0.041 (0.108) | 0.31 (0.83) | 0.04 (0.10) | .990 (.985) | .999 (.998) | - |
| 0.71 s | 0.179 (0.428) | 0.85 (2.15) | 0.54 (0.84) | .969 (.930) | .998 (.990) | .08 |
| 1.43 s | 0.354 (0.729) | 1.54 (3.40) | 1.60 (1.90) | .958 (.880) | .997 (.981) | .28 |
| 2.50 s | 0.581 (1.094) | 2.31 (4.59) | - | .948 (.851) | .993 (.979) | .17 |

(map units; "commanded units" = selected or targeted by a command in the window, 1-3% of the
units present; unseen maps and seen games are within a few percent of these.)

What this says:
- Predicting state directly works: movement error halves against persistence at every
  horizon, orders and exits improve, deaths start to be called (recall 0.2-0.3 at 1.4 s).
  Most of this comes from the orders already in the state (a unit with a move order moves
  along it); the model has learned the engine's kinematics from a single frame.
- **The commands are not used.** With and without them the errors are identical to three
  digits, including on the units the commands touched. The window's SELECT / CMD tokens bind
  to units only softly (pointer embeddings of type, owner, position); in 4000 steps the
  predictor did not learn to match an entity against them. Same story as the pair model's
  first version (above). The fix is hard binding: add each command's embedding to the tokens
  of the units it was issued to, which needs the recipients per command (see next).
- **z is not used.** Posterior-driven and prior-driven predictions are identical; KL fell to
  0.02 nats in run 1 and sits at 0.5-0.75 (under the 4-nat floor) in run 2. With the commands
  unread there is nothing for z to add either (it would have to carry the same thing). So the
  with/without KL test is not yet informative; it becomes so once commands bind.
- Births are 1 per window at 0.18 s and ~10-20 at 2.5 s (2v2 games included); not predicted.
- hp / shield / energy / build changes are tiny on average (MAE 0.001-0.005 either way); a
  fight-only subset is the metric to add.

Next (in order): record command recipients and control-group contents in the extraction,
bind commands to recipients in the predictor, evaluate on commanded units and in fights,
then the KL test; then births (child slots); then one-loop extraction.

### Command recipients and control groups (2026-10-30)

The API does not expose control-group contents: `ui_data.groups` is index + leader unit type
+ count per group, and the selection panel (`ui_data.multi`) lists type / hp / energy per
selected unit without tags. What it does expose, in a player-view playback, is every command
the engine executed with its resolved recipients:
`observation.actions[].action_raw.unit_command.unit_tags` (plus ability, target point / tag).
`scripts/build_ability_table.py` already reads these for the ability table and drops the tags.

Plan: (1) in the player-view extraction, store per command loop, ability, target and recipient
unit ids (via `unit_tag`), one playback per player as now; join to the observer's command
stream on loop / ability / target for observer-view training. (2) Control-group contents as
part of the player's state: a selection tracker over the replay's selection-delta and
control-group events (`worldmodel/replay_actions.py` already parses the deltas; sc2reader's
tracker is the reference for the index/mask semantics), validated per frame against
`ui_data.groups` (count, leader type) and at every command against the engine's recipients
(a command issued to group 1 reveals group 1 exactly). Stored per frame as ragged group
membership; in the player view each entity token gets an 11-bit "in groups / selected"
feature, so "select group 1, snipe" needs no long-term memory in the model.

Built 2026-10-30. Validated on the 13 build-98310 1v1 games of `data/replays/pig-viewer-replays`
(26 player views + 13 observer views, `data/worldmodel/pov_1v1` / `obs_1v1`, extracted with
`--any_ruleset` since their ruleset has no Editor export) and, first, on four 2v2 games of
`my-recent-games` (`pov_cmd` / `obs_cmd_test`). Three pieces:

**1. Command recipients** (`extract_states.py --views players`, per-view file). With the
smallest feature layer on (16x16; the engine fills `ui_data` only then; cost unmeasurable, 4.4 vs
4.1 s per 3000 loops) the player view keeps every command the engine executed, with the
exact loop it was issued at (`Action.game_loop`) and its recipients:

    cmd_loop[C], cmd_frame[C]      issue loop (exact); the frame whose observation window reported it
    cmd_abil[C]                    API ability id
    cmd_target[C], cmd_tx, cmd_ty, cmd_target_id
                                   0 none / 1 point / 2 unit; the point (NaN if none), the target's
                                   state unit id (-1 if none)
    cmd_queued[C]                  the shift-queue flag
    cmd_start[C+1], cmd_unit[M]    recipients of command c: cmd_unit[cmd_start[c]:cmd_start[c+1]],
                                   state unit ids (unit_tag maps them to engine tags), in the
                                   engine's order (ascending tag)

The per-loop observations that `--substep 1` requests are no longer thrown away unparsed: their
commands, deaths and the player's selection are read (0.1 ms to parse with upb, plus a Python
pass over the units for `is_selected`). Before this, player views missed the commands and
`dead_units` of 3 of every 4 loops; the 33-game `pov` sample's `dead_id` lists are incomplete
for that reason (the observer view, stepped without substep, is unaffected). Two more things
the player view now keeps from the engine, both the observed player's only (the API fills
`is_selected` and `ui_data` for the observed player alone, so the observer view has neither):

    u_flags bit 64 (F_SELECTED)    the unit is in the player's selection, in the kept frames
    sel_loop[S], sel_start[S+1], sel_id
                                   the selection as a set of state unit ids at every loop it
                                   changed (sel_loop[k] <= L < sel_loop[k+1]: the selection as
                                   of loop L), read from every per-loop observation

**2. The join to the replay's command stream** (`worldmodel/command_join.py`): each engine
command pairs with the action-table row (`data/worldmodel/actions/<name>.npz`, the same rows for
every view) that caused it, by exact loop and target, REPEAT rows by the last paired API ability,
each engine command used once (the `build_ability_table.match` rules, with the exact loop
instead of an 8-loop window). Observer-view training can take the recipients from the player
views through this: 5698 of 5703 engine commands (99.9%) matched a row, and 1299 of 1304 CMD
rows (99.6%) matched an engine command (a CMD row with no engine command is a rejected
command: no larva, minerals, supply; the 5 unmatched engine commands are in the two long games).

**3. Control groups and selection as state** (`worldmodel/selection_tracker.py`), per frame, in
the observer files for every player and in each player view for its player:

    grp_player[P], grp_start[F*P*11+1], grp_id[...]
                                     group g (0-9; 10 = the active selection) of player k (slot
                                     grp_player[k]) at frame f: grp_id[grp_start[(f*P+k)*11+g]:
                                     grp_start[(f*P+k)*11+g+1]], state unit ids in the game's
                                     order; dead members left out
    ui_grp_count, ui_grp_leader[F, 10], ui_sel_panel[F], ui_sel_type / ui_sel_start[F+1]
                                     with --ui: the engine's own summary (ui_data; needs the
                                     feature layer on), kept for validation

The replay's selection events are the only source that covers every player in the observer
view (`is_selected`, `ui_data` and the recipients exist only in a player's own playback), so the
tracker is primary and the engine's data validates it: per frame against `ui_data.groups`, at
every selection / group event against the `is_selected` set one loop later, and at every
command against the recipients. Slot 4's composed view (`compose_view.py`) carries none of the
per-player arrays (its groups are in the observer file).

The tracker replays the replay's SELECT / GROUP rows (sc2reader's SelectionTracker is the
reference for the event structure; it sorts by tag as an approximation and ignores deaths). What
had to be found out, each checked against the engine's recipients and group counts:

- **Order.** A list (group or selection) is in ascending unit-tag order; the removal masks index
  that order, the engine's recipients come in it, and the selection panel lists units in it
  (every multi-unit recipient list in the sample is in tag order). The subgroup priorities in the
  deltas describe the units; they do not order the list (sorting by them, in either direction,
  broke the masks).
- **Mask bits.** s2protocol reads a removal mask as one big-endian integer, and the game writes it
  byte by byte, so for masks over more than 8 units the bytes come out swapped (position 0 of a
  16-unit mask is bit 8). `replay_actions._mask_positions` now decodes position p as bit
  (w-1-p%8) of the (p//8)-th 8-bit group from the top; a 23-unit cleanup mask decoded this way
  removed exactly the 12 members that had died the loop before. **Action tables written before
  this (`data/worldmodel/actions`, all 573) have wrong `rem_pos` for masks over 8 units and lack
  the group masks and subgroup priorities; re-run `extract_actions.py` before using them for
  groups** (`check_player_commands.py` refuses the old layout).
- **Deaths.** The game does not remove a unit from a group or the selection when it dies: it stays
  as a stale entry that the masks of later recalls and deltas account for (a baneling that
  exploded at loop 5076 was still position 1 of its group's recall mask at loop 5233), and the
  engine's counts, panel and recipients just skip it. Pruning at the death loop (the first
  version) shifted every later mask. So the tracker never prunes; the per-frame output drops
  members whose death loop (the replay's tracker events) is <= the frame loop. An egg hatching
  is a death of the egg plus, two loops later, a delta replacing it by the hatched unit.
- **Steal.** Set/add with steal (ops 4, 5) need no side effect: the game logs the removal from
  every other group itself (a burst of recall / mask / set rows per affected group, then the
  selection's restoration, all in one loop). Doing it in the tracker double-removed.
- **Group masks.** `SControlGroupUpdateEvent` carries a mask too (dropped before): on a recall
  it is the members not selectable right now (dead, in a transport, ...), on add the positions of
  the group's current list. Now in the GROUP rows' `select_remove` / `rem_pos`.

**1v1 results** (`scripts/check_player_commands.py data/worldmodel/pov_1v1/*.npz --replays
data/replays/pig-viewer-replays`). Columns: (frame, group 0-9) cells whose tracked alive count
equals `ui_grp_count`; frames whose tracked selection has the selection panel's unit types;
selection / group events after which the tracked selection equals the engine's `is_selected` set
one loop later; engine commands whose recipients are exactly the tracked selection at the
command's row / a proper subset of it (the game's own filtering, below) / neither ("off": part of
the recipients outside the tracked selection, i.e. the tracker is wrong there).

| game (p1 v p2), loops | view | group cells | selection frames | events | recipients exact / subset / off |
|---|---|---|---|---|---|
| ZvP cannon rush (Rainfall), 8992 | p1 Z | 99.99% | 99.6% | 100% | 72 / 28 / 0 |
| | p2 P | 100.00% | 99.7% | 100% | 98 / 2 / 0 |
| CryHavoc v PrinceXizor, TvZ, 16963 | p1 T | 95.19% | 99.7% | 98.9% | 49 / 51 / 0 |
| | p2 Z | 100.00% | 96.5% | 95.6% | 46 / 53 / 1 |
| Netslayer v PrinceXizor, TvZ, 17844 | p1 T | 99.00% | 96.8% | 96.2% | 72 / 25 / 3 |
| | p2 Z | 100.00% | 95.6% | 97.1% | 45 / 52 / 3 |
| keyz v PrinceXizor, TvZ, 27608 | p1 T | 100.00% | 98.8% | 99.2% | 59 / 40 / 1 |
| | p2 Z | 99.99% | 95.8% | 96.0% | 35 / 62 / 3 |
| PrinceXizor v GatoRyup, ZvT, 30856 | p1 Z | 99.83% | 94.2% | 93.3% | 38 / 59 / 3 |
| | p2 T | 93.39% | 99.2% | 99.9% | 37 / 63 / 0 |
| PrinceXizor v Matallamas, ZvT, 19038 | p1 Z | 100.00% | 98.5% | 99.0% | 51 / 48 / 1 |
| | p2 T | 99.89% | 95.4% | 98.2% | 70 / 25 / 5 |
| At Eternity's Edge (105), TvP, 40076 | p1 T | 96.76% | 96.6% | 94.2% | 47 / 51 / 2 |
| | p2 P | 100.00% | 98.4% | 99.7% | 50 / 50 / 0 |
| Blackrock (72), TvP, 17930 | p1 T | 99.26% | 99.3% | 99.4% | 61 / 39 / 0 |
| | p2 P | 100.00% | 99.1% | 99.7% | 83 / 17 / 0 |
| Blackrock (77), PvZ, 10669 | p1 P | 100.00% | 98.3% | 99.3% | 95 / 5 / 0 |
| | p2 Z | 100.00% | 99.2% | 99.8% | 72 / 28 / 0 |
| Getting Reaper Cheesed, TvT, 8789 | p1 T | 100.00% | 99.8% | 100% | 90 / 10 / 0 |
| | p2 T | 100.00% | 99.8% | 99.2% | 82 / 18 / 0 |
| Rainfall (84), PvT, 24356 | p1 P | 100.00% | 99.6% | 99.9% | 62 / 38 / 0 |
| | p2 T | 99.92% | 98.9% | 98.4% | 57 / 43 / 0 |
| nydus worms forever, ZvT, 16587 | p1 Z | 100.00% | 95.4% | 95.3% | 45 / 54 / 1 |
| | p2 T | 100.00% | 97.2% | 92.7% | 39 / 61 / 0 |
| out macro'd, TvT, 26187 | p1 T | 100.00% | 98.5% | 98.0% | 70 / 30 / 0 |
| | p2 T | 99.67% | 96.6% | 96.7% | 30 / 68 / 2 |
| all 26 views | | 99.13% | 97.7% | 97.6% | 54 / 44 / 1.3 |

Join: 22,666 of 22,740 engine commands (99.7%) matched a replay row, 6167 of 6257 CMD rows
(98.6%). Group ops (set / add / recall / steal) agree with the engine's selection one loop later
in 99.6% of 5529 cases; the remaining event mismatches are mostly single-unit SELECT rows where
`is_selected` lags or the unit is gone a loop later. The engine's own selection agrees with its
recipients at the same rate as the tracker's (recipients = `is_selected` set exactly for 55% of
commands, a subset for 44%), so the subset column is the game filtering the selection, not
tracking error. Over the non-empty group cells alone the count agreement is 97.4%; the leader
check (first unit's type = `ui_grp_leader`) is 88.5%, see the note under the 2v2 table. All 39
views went through on the first try with 8 workers and `PROTON_VERB=run` (a 2248-frame player
view in 51 s, the 10,019-frame one in 368-409 s: 100-180 loops/s per worker; observer views
28-223 s).

**2v2 results**, the first test set, kept for the record (the 2v2 games have bigger armies and
the two long Zerg views carry the open drift described below). Measured per view (`scripts/check_player_commands.py data/worldmodel/pov_cmd/*.npz --replays
data/replays/my-recent-games`; the extraction prints the same counts per view in its `ok` line),
all 12 views, 8 workers (the run that also wrote the observer files); same columns.

| view | group cells | selection frames | events | recipients exact / subset / off | loops |
|---|---|---|---|---|---|
| New Bed of Chaos (50) p1, ZZvZZ | 99.98% | 99.4% | 99.7% | 63 / 37 / 0 | 6292 |
| New Bed of Chaos (50) p2 | 99.98% | 98.9% | 100% | 78 / 22 / 0 | |
| New Bed of Chaos (50) p3 | 99.99% | 99.4% | 100% | 81 / 19 / 0 | |
| Fields of Death (53) p1, Zerg | 99.25% | 94.8% | 95.3% | 55 / 41 / 4 | 17166 |
| Fields of Death (53) p2, Zerg | 97.94% | 92.0% | 91.4% | 61 / 32 / 7 | |
| Fields of Death (53) p3, Terran | 100.00% | 99.6% | 99.9% | 78 / 22 / 0 | |
| Rhoskallian (88) p1, Terran | 95.73% | 87.3% | 91.9% | 56 / 29 / 15 | 24642 |
| Rhoskallian (88) p2, Protoss | 100.00% | 99.8% | 100% | 68 / 32 / 0 | |
| Rhoskallian (88) p3, Zerg | 99.16% | 94.5% | 96.7% | 60 / 36 / 4 | |
| Yellowjacket (100) p1, Zerg | 96.46% | 83.3% | 85.7% | 52 / 32 / 16 | 24068 |
| Yellowjacket (100) p2, Zerg | 98.28% | 87.6% | 92.3% | 60 / 30 / 10 | |
| Yellowjacket (100) p3, Terran | 100.00% | 96.3% | 98.5% | 54 / 43 / 3 | |
| all twelve | 98.61% | 93.1% | 94.4% | 60 / 32 / 7 | |

Join: 13,426 of 13,440 engine commands (99.9%) matched a replay row, 3281 of 3300 CMD rows
(99.4%). The engine's own selection agrees with its recipients at the same rate as the tracker's
(recipients = `is_selected` set exactly for 65.5% of commands, a subset for 32%): the subset
cases are the game filtering the selection, not tracking error. Empty cells count as agreeing in
the group column; over the non-empty cells alone the count agreement is 96.6%. The leader check
(first tracked unit's type = `ui_grp_leader`) only works for single-type groups (98% in the ZvZ
game, 87% overall): the panel shows the highest-priority subgroup first, and ordering the list by
the deltas' priorities made the masks worse, so the list order and the panel's order are
different things. The observer file's groups for a player are identical to that player's view's
(17,303 of 17,303 lists in New Bed p2). Speed with 8 workers: a 6292-loop player view in 58 s,
a 24,642-loop one in 361-392 s (65-110 loops/s per worker; the per-loop parse and the Python
pass over the units for `is_selected` are part of it), observer views 24 s and 233 s.

The subset cases are the game's own filtering: a train from larvae takes one larva per press
(abilities 1342-1345, 1351), a morph one unit per press (4124 baneling), and attack / smart-click
on a mixed selection skips the units that cannot do it (eggs, cocoons). In the short game the
remaining mismatches are single frames at a morph (the delta lands 1-3 loops after the engine
shows the new type) and the selection panel's own quirk of keeping a unit that died while
selected until the next change.

**Open: drift in the long Zerg views.** From the first big fight on, the tracked groups of
Yellowjacket p1 run 1-3 units short of the engine's count for long stretches, and 16% of its
commands have a recipient outside the tracked selection (Terran and Protoss views, and the Zerg
views of the short game, are at 0-4%). The masks are the symptom: of the 149 group recall / add
masks in the first six views, 122 drop exactly the tracker's dead entries, 10 a subset of them,
and 17 (all in the long games, after fights) drop one entry the replay's death log calls alive
and leave one it calls dead, at positions 4-9 apart in a list the tracker keeps in tag order.
`check_player_commands.py --verbose` prints the events after which the tracker and the engine's
`is_selected` set differ (loop, rows with masks, the units each side has alone), for this. The same list recalled a few loops apart shows the same shifted position, so
the game's list differs from the tracker's by one entry's place, after many units of the list
died at once. Not understood yet; candidates are a death the tracker log reports late or not at
all (a unit killed by a baneling the same loop it was added?), or dead entries changing place in
the game's list. Until then, per-frame groups are right to ~99% of cells in games without big
fights and ~96% in ones with them; the per-command recipients (part 1) are exact regardless.

**Launch failures, not crashes.** Two runs with four workers lost 6 of 12 views each: the SC2
websocket closed or never came up (`WebSocketConnectionClosed`, `ConnectError`,
`ConnectionError`), never the same views twice, and a probe launching two clients at once had
both fail while the same client launched alone passed. So it is concurrent launching under
Proton, not the feature layer (which is optional anyway: `--ui`, needed only for the `ui_*`
validation arrays; the recipients come from `action_raw`). The sibling session's
`PROTON_VERB=run` in `wine_run_config.wine_env` is meant to fix it, and the worker now retries
these errors like `InvalidReplayData` (45 s, 90 s). The 8-worker rerun below is the test.

### Older builds from the CDN (2026-10-30)

The replay header's `m_ngdpRootKey` is the TACT **root** content key of the game-data build
the replay was played on, and it equals the `root =` line of that build's BuildConfig (checked
for 93333, 94137, 95841, 97364, 97563 and the current 98310). The BuildConfig hash itself is
not in the replay and Blizzard's version endpoint (`us.patch.battle.net:1119/s2/versions`)
lists only the current build per region (CN is still on 5.0.14.93333, same CDNConfig). BlizzTrack
archives every version manifest: `https://blizztrack.com/api/manifest/s2/seqn?file=versions`
lists the snapshots, `https://blizztrack.com/api/manifest/s2/versions?seqn=<seqn>` gives each
build's BuildConfig / CDNConfig. The US CDN (`http://level3.blizzard.com/tpr/sc2/config/xx/yy/<hash>`)
still served all of these on 2026-10-30:

| build | version | BuildConfig | CDNConfig (one of) | replays here |
|---|---|---|---|---|
| 93333 | 5.0.14 | 8453c2f1c98b955334c7284215429c36 | e4f2e2da310c08495f2701d401d14659 | DH Dallas 2025 (64) |
| 94137 | 5.0.14 | 05cce99544c8b80c07f6eb3250b957a9 | 7df6dfb5f6256b47c5ab86bff1b1b1eb | HSC XXVII, EWC, FEL 2025 (~366) |
| 95841 | 5.0.15 | f48701eb2ef86e687658d3deb0fc1bc8 | 77abccbe50b29b03be4d4cfca8b217ad | HSC XXVIII (176) |
| 97364 | 5.0.16 | 2e257e413468e65da51630c5080a6168 | 14cac7d294ebc69e790d53ea7688effa | viewer tournaments (61) |
| 97563 | 5.0.16 | 5bc8dcc1fade3a320c12936586b7c0ed | d8aae9fce802373d9da50a2bec7b6040 | viewer tournaments (84) |

Also on BlizzTrack: 95248, 95299, 96592, 96826, 96883, 97425.

**What a build needs.** `Versions/Base<build>/SC2_x64.exe` and `SC2.exe` (the build's install
manifest; the "code"-tagged files are loose on the CDN and the Agent never keeps them in the
storage), the build's data in the install's CASC storage (`SC2Data/data`), the BuildConfig under
`SC2Data/config/xx/yy/`, and a row in `.build.info` (Active 0, otherwise a copy of the current
row with the build's key and version). The engine reads every row, loads its BuildConfig and
takes the one whose root equals `-dataVersion` (pysc2 passes the replay's). For this install's
tags ("Windows ... enUS speech" / "enUS text") a build's download manifest selects 188 573 of its
608 4xx files, 26.3 GB, and a 5.0.17 install already has all but 145-190 of them (63-66 MB);
with the build's own encoding (46 MB), root, install, download and size files the delta is
83-134 MB per build. Nothing else changes: the 98310 row stays active, old `.idx` versions and
archives are untouched, the new files go into the free tail of the last archive (new `data.NNN`
when full, with the Agent's 16 marker records at its start) and every bucket gets a new `.idx`
version, `shmem` is rewritten (bucket versions, free-space list). Formats and the checksum /
hash algorithms (lookup3 record checksums, chained hashlittle2 entry hash, bucket = xor of the
EKey's first 9 bytes folded to a nibble) were verified against the existing storage first.

**Implementation: `cosmicstar/envs/tact.py`** (requests only). `plan()` fetches BuildConfig,
encoding, install and download manifests and lists what the storage lacks; `install_build()`
fetches the missing BLTE files from the CDN archives (`CDNConfig` -> 149 `.index` files, range
requests; loose otherwise) into `data/tact/blobs/` (verified against their EKey), backs up
`.build.info`, `shmem` and the live `.idx` files to `SC2Data/backup-<time>/` (with a manifest of
the archive sizes), appends the files to the storage, writes the `.idx` files and `shmem`, the
binaries, the BuildConfig copy and the `.build.info` row. The engine rewrites `shmem` when it
starts, so `--force` (needed while any engine or the Agent runs) waits for a gap between engine
launches and re-checks `shmem` afterwards. `--revert <backup dir>` undoes an install exactly
(restores the files, deletes the newer `.idx`, truncates/deletes the archives). Measured here:

| build | files added | written to the storage | wall time |
|---|---|---|---|
| 93333 | 190 | 134.0 MB (46 MB of it the encoding file) | 29 s |
| 94137 | 8 + metadata | 106.1 MB | 30 s |
| 95841 | 27 + metadata | 106.8 MB | 30 s |
| 97364 | 23 + metadata | 83.4 MB | 30 s |
| 97563 | 15 + metadata | 83.0 MB | 29 s |

(25 s of each is the post-write `shmem` check; the fetch itself was 1-2 s per build, the
binaries 115 MB per build on disk.) Usage, from the repo root:

```bash
.venv/bin/python -m cosmicstar.envs.tact 93333 --dry-run        # what is missing
.venv/bin/python -m cosmicstar.envs.tact 93333 --force          # install (also while engines run)
.venv/bin/python -m cosmicstar.envs.tact 93333 --install "<dir>" --force   # into another install
.venv/bin/python -m cosmicstar.envs.tact 93333 --revert "<dir>/SC2Data/backup-..."
```

**Where the builds live.** The real install (`~/Games/battlenet/.../StarCraft II`) stays as
Battle.net left it, so its updates never meet our archive and index additions: 93333 was
written into it once (14:27 on 2026-10-30, the engine then played a DreamHack replay) and
reverted the same afternoon with `--revert` (backups `SC2Data/backup-20261030-14*`; a 98310
replay plays as before). The older builds are in a side install,
`.../Program Files (x86)/StarCraft II old`: `SC2Data/data/data.001-023` are hardlinks to the real
archives (the engine never writes those; it rewrites only `shmem` and `data.000` when it starts),
`data.000`, `data.024`, the `.idx` files, `shmem`, `indices`, `config`, `Support`, `Support64`,
`Interfaces` and `Versions/Base98310` are copies (2.1 GB extra), and the five builds were
installed into it with `--install` (table above; `data.024` filled up and `data.025` holds the
rest, 391 MB). `wine_run_config` knows it: `installs()` lists the real install and the
`SIDE_INSTALLS` that exist (just `$SC2PATH` when that is set), `install_for_build()` picks the
first one with `Versions/Base<build>/SC2_x64.exe`, `installed_builds()` is the union, and the
run config takes the install from the requested version, so `extract_states.py`, `dump_rules.py`
and `export_balance.py` need no environment juggling (the Editor runs from the real install).

**End to end** (2026-10-30, run config picking the install): `dump_rules.py` dumped the 93333
ruleset (12 s), `export_balance.py` exported it (10 s), `extract_states.py` processed one replay
each of 93333 (50 s), 94137 (64 s) and 95841 (85 s) from `data/replays/older-builds-test3`, and
earlier 3 of the 4 DreamHack Dallas 2025 replays in `data/replays/older-builds-test` (G4
Ultralove fails the exact desync check at 558 s: a unit the replay logs as dead at 555 s is
alive - open, not seen in the other 9 games) and the 94137 / 95841 replays in
`older-builds-test2`. 97364 and 97563 are installed (the engine starts) but
`extract_states.EXCLUDED_BUILDS` skips 5.0.16 replays by design.

**Revert and shmem.** `--revert` refuses a backup folder from another install (the shell keeps
an exported `SC2PATH`, which once pointed a revert of the real install's backup at the side
install; the side install was simply reinstalled, 30 s per build). The engine rewrites `shmem`
at every start, sometimes in its own layout (header block type 5, the free-space block moved);
the reader takes both and the writer always produces the Agent's layout (byte-identical to the
Agent's file when nothing changed).

**The client's own route (SC2Switcher) does not install builds here.** With the Battle.net app
open and logged in in the same prefix, `Support64/SC2Switcher_x64.exe <replay>` only starts
`Versions/Base98310/SC2_x64.exe <replay>`: no Agent operation, no download. A client started
outside the Battle.net app never gets its session ("CONNECTING TO BLIZZARD SERVICES", Play
Offline / Cancel, for minutes) and offline mode cannot download (replay_fix says the same). The
remaining in-game path (Battle.net Play -> Replays -> old replay -> download prompt) needs mouse
clicks that cannot be injected on this box (no xdotool; XTest pointer input is ignored under
Xwayland; `XGetImage` screenshots of the Wine windows do work).

**Gotchas found on the way.** Proton's default verb `waitforexitandrun` first waits for every
Wine process in the prefix to exit, so with the Battle.net app open no engine launch ever
started (`wine_run_config.wine_env` now sets `PROTON_VERB=run`; launches also no longer
serialize behind each other). The Windows-side launcher (`umu.exe`) can outlive its child while
another Wine process keeps the wineserver up, so `export_balance.py` now waits for the Editor
process itself. The Editor's command line parser takes " -" inside a path for an option: a
replay named `QM - ShoWTimE vs SHIN - G1 - Ley Lines.SC2Replay` fails with "Unable to load
replay file [Core: file not found]", so `export_balance.py` exports from a copy with a plain
name; it also needs every dependency in the Battle.net cache (`depot.ensure_cached`), since it
cannot download them while the Battle.net app runs.
### Oracle = the players' views concatenated (design decision, 2026-10-30)

Focus is 1v1 (pro tournament packs; no slot-4 crash). The oracle state for a 1v1 game is the
two player views **concatenated, not unioned**: every entity appears once per viewer who
perceives it, each row tagged with the viewer and showing what that viewer sees (full detail
for the owner, a shadow for a cloaked unit without detection, a snapshot for a remembered
structure, nothing in fog). The duplicates sit at the same position, so attention binds them;
the state then holds what each player *knows*, not only what is true. Targets per row are that
viewer's observation of the unit at t+k, including whether they still see it; masking one
viewer's rows out of the input turns the same network into the belief (player-view) model, so
stage two is a training-time mask (like the action dropout), not a second architecture. The
concatenation is done in the loader; extraction is unchanged (views are already stepped on the
same loops). Token count ~1.3-1.6x a single view. Snapshot rows have client-assigned tags and
link to their counterpart only by type and position. The observer playback becomes optional for
1v1 once a few games confirm the owners' rows reproduce the observer's (compose_view.py's
measurement says they do).

## Ruleset encoder: types from the balance data (`worldmodel/ruleset.py`, 2026-10-30)

**Parked the same day** (`train_direct.py --types ruleset`; the default is `table`, learned
embeddings per id): the decision is to let the model learn the type embeddings from the
replays first, then use the game data in two later roles: linear probes from the learned
embeddings to the catalog fields (which values did the model discover from dynamics alone),
and distillation, where this encoder becomes the student that reproduces the learned
embeddings from the catalogs, so new patches and units get embeddings from numbers alone. A
shared-per-type plus per-ruleset-residual decomposition of the learned tables (mirroring base
data plus balance patch) was discussed and deferred. The parser validation against the export
and the rules dump (below) is due when the probes and the student need trustworthy inputs.

Original rationale: unit types, ability commands and upgrades embedded from the game's own
data, so that a balance change moves the model's behaviour at once and a type the model
never saw enters from its numbers. `scripts/build_ruleset_tables.py`
turns each ruleset's Editor balance export (`data/worldmodel/balance/<ruleset>/*.xml`) and
engine rules dump (`data/worldmodel/rules/<ruleset>.json`) into `data/worldmodel/rulesets.json`:
per ruleset a graph of ~1060 unit nodes, ~440 ability-command nodes and ~90 upgrade-level nodes.

Features, fixed width, every optional element as its numbers plus a presence bit (zeros when
absent): units 46 numbers (life / shields / energy with regen, armor, costs, movement type and
speed, radius, sight, cargo, supply, race, the 11 attribute tags); weapons 36 each (range,
period, count, damage and max, splash, target class, kind, bonus damage and bonus max per
attribute tag, so the marauder's `<bonus type="Armored">` is one slot and a unit without a
bonus is zeros); ability commands 15 (type, range, costs, target kind from the rules dump);
upgrade levels 4. A unit's weapons are a set: each through an MLP, summed, so any count works
(2 is the most in these exports). Blizzard's hand-written strengths / weaknesses lists are
left out: interface hints, not mechanics.

Joins: API unit type id -> XML unit by name (1058 of 1061 mapped; 1063 of 1063 on 98310);
API ability id -> (link name, command index) (410 of 442 commands); upgrade name -> level id
(the tracker's names, case-insensitive fallback). Edges, 16 typed relations: unit has ability /
upgrade, command makes unit / upgrade (from the producer's `<trains>` / `<builds>` /
`<researches>` catalog indices), command or upgrade requires unit / upgrade, unit produced by
unit. ~4100 edges per ruleset.

`RulesetEncoder`: item MLPs -> node tokens (+ a kind embedding, + a zero-initialized residual
per stable name, `--no_type_residual` turns it off) -> `--rs_depth` 3 attention layers over
all ~1600 nodes with an additive bias per edge relation -> tables of unit / ability / upgrade
embeddings (+ an "unknown" row each). Runs once per ruleset per batch (cached, `clear()` per
step); entity tokens, order and production features, fogged command targets and the players'
completed upgrades gather their rows (`lookup`). 3.1M parameters at d=256. Games whose ruleset
has no balance export are skipped by `train_direct.py --types ruleset` (the default; `table`
is the old learned embedding); the 1v1 viewer games on ruleset 98310_4e2c79d1 need
`dump_rules.py` + `export_balance.py` first. The residual norms are printed per evaluation:
a name whose residual grows is one the data describes badly.

What this is for, beyond patches: the attention over the ruleset lets a type's vector hold
what is true in every game (a bonus tag meets the attribute tags of the types it applies to,
a target class meets movement types, a train command meets its product), so the entity-level
attention over the frame only has to pair those properties with the situation.

### Upgrade effects from the game data (2026-10-30)

The balance export never says what an upgrade does: every weapon `<bonus>` has `max` equal to
`damage`, and the per-unit `<upgrades>` levels carry only cost and requirements. The Marauder's
+1 vs Armored per weapons level (20/22/24/26) is nowhere in it. The numbers live in the game's
own catalogs, `GameData/*.xml` inside the mods, which sit in the install's CASC storage:

- **Root format.** SC2's TACT root is an MNDX file (`MNDX` magic): three compressed name
  databases (a path-fragment trie over rank/select bit arrays) plus the CKey entries. It is
  not a text manifest. Rather than port CascLib's ~3000-line parser, `envs/casc.py` loads
  CascLib itself through ctypes (`scripts/build_casclib.sh` builds `lib/libcasc.so` with
  plain gcc/g++, no cmake; compiled by the user, since the sandbox refuses to build downloaded
  code). `Storage(install, build_key=...)` opens the storage at a build (the `.build.info`
  row's Build Key; the Lutris install has no Product column, so no code name is passed),
  `find(mask)` lists names (lower-case, backslashes, e.g.
  `mods\liberty.sc2mod\base.sc2data\gamedata\upgradedata.xml`), `read(name)` returns bytes.
- **Extraction.** `scripts/extract_gamedata.py` writes the ladder mods' catalogs per build to
  `data/worldmodel/gamedata/<build>/mods/<mod>.sc2mod/base.sc2data/gamedata/*.xml`: 552 files,
  19.8 MB per build, for core, liberty, swarm, void and voidmulti (the ladder dependency chain;
  the storage also holds libertymulti / swarmmulti, the balance-test mods, coop and campaign
  data). A build has to be installed (real or side install) to be read; the script refuses a
  build it cannot find in `.build.info` rather than fall back to the active one.
- **Layering and inheritance** (`worldmodel/gamedata.py`): mods are applied in load order; a
  later entry with the same id sets scalar fields, patches array entries given by `index`
  (voidmulti patches `TerranInfantryWeaponsLevel1`'s inherited `EffectArray[10]` to +2 for the
  Ghost's rifle), appends entries without one, removes with `removed="1"`. `parent=` copies the
  resolved parent, so the level entries (`...Level1`, parent `TerranInfantryWeapons`) inherit
  the template's per-level effect list; `default="1"` entries are templates.
- **What is read.** `CUpgrade.EffectArray` references `Catalog,Id,Field[Key]` with a value,
  applied once per level: `Effect,PunisherGrenadesU,Amount +1`,
  `Effect,PunisherGrenadesU,AttributeBonus[Armored] +1`, `Unit,Marine,LifeArmor +1`,
  `Weapon,Claws,RateMultiplier +0.2` twice (adrenal glands, liberty and void each add 0.2:
  +40% attack rate). Icons, tooltips and actors are dropped (`MECHANICS` lists the fields
  kept). `AffectedUnitArray` names the units an upgrade touches.
- **Into the tables.** Each weapon vector gains per-level increments (damage, attack-rate
  multiplier, range, period, bonus per attribute tag) summed over the upgrade families that
  touch its weapon or effect id, one representative level per family; each upgrade level gets
  what it does (counts and means of damage / bonus / armor / life / speed / rate / range
  deltas, affected-unit count) after the export's costs; `g_affects_u` edges link an upgrade to
  the units it changes. A `has_gamedata` bit says whether the build's catalogs were available.
  Verified on 98310: Marauder +1 and +1 vs Armored, Marine +1, Zergling +1 and rate +0.4,
  Ghost +2 (voidmulti), infantry armor +1.
- **Production commands.** The export's per-unit `<abilities>` lists leave out train / build /
  research abilities (only `<trains>` / `<builds>` / `<researches>` entries with a catalog
  index). `ability_links.json` names those indices, so each product becomes an ability-command
  node (`BarracksTrain/Marine`, API id 560, type train / build / research, the product's cost)
  with `a_makes_u` / `a_makes_g` edges; 641 commands per ruleset, 609 with API ids (was 442 /
  410), so production orders now resolve to real rows instead of the unknown one.

## Control-group memory in the state, player views as training data (2026-10-30)

Decision (user): the model predicts at most 2.5 s ahead, so control-group contents cannot be
rebuilt from events inside a window; they are state, encoded at extraction time. Each entity
token now carries `ent_groups`: 11 bits, membership in its owner's control groups 0-9 and the
active selection (10), read from the tracker's per-frame `grp_*` tables (`GameFrames.groups`),
plus the engine's selected flag as the 7th flag bit of the entity numbers (`F_SELECTED` in
`u_flags`). "Select group 1, cast Snipe" is then a predictable state change: the units whose
group-1 bit is set get the selected bit, and the command's recipients are the selected ones.
No hard binding of commands to the engine's recipients: the model is to learn it from the
select / control-group / command tokens and these bits; `cmd_*` stays for evaluation (the
"touched by a command" metric). On the 1v1 views ~4% of a player's unit rows are selected at
any moment and most army units sit in some group.

`train_direct.py` now trains on player-view files directly (`--data data/worldmodel/pov_1v1
--splits data/worldmodel/splits_1v1.json`): the action table is `<game>.npz` for a view
`<game>_p<slot>.npz`, the window keeps only the viewer's commands (`only=viewer`; the enemy's
are unknown to them), the owner map uses me / enemy relations, and the per-player command
summaries are scattered to the frame's player order (the enemy's summary is zero and masked,
so the prior sees "no commands" for the enemy, which is the point of the player view). The
oracle as two concatenated views is not built yet; this is the single-view (belief) setting.
The pro 1v1 extraction (two views per game) is running in its own session.

## Open item: desyncs may be wrong map versions (noted 2026-10-30)

Several replays fail the extractor's exact desync check (e.g. one DreamHack Dallas 2025 game at
558 s; a share of the pro 1v1 set is expected to). Hypothesis (user): the map loaded for the
playback is not the version the game was played on. `depot.py` fetches a map by the content
hash in the replay's cache handles, so the file should be exact, but the engine may pick a map
or mod from the Battle.net cache by name / another handle, or a dependency (mod) may resolve
to a newer version. To check when it comes up: for a desyncing replay, compare the handles in
`replay.details` with the files actually opened (the engine's log, or `-dataDir` cache
contents), re-run with only the fetched files present, and see whether the desync moves.

Update, same day: a concrete candidate. Older-build replays were started with `map_data`,
which makes SC2 write a shared temp map file; `launch_lock` existed because concurrent
launches could load each other's map (InvalidReplayData or a desync). That lock also
serialized every start of the pro 1v1 extraction (one engine loading, seven frozen). Older
builds find the map in the Battle.net cache by handle just like the current one (a 93333
replay starts in 13 s without map_data), so `extract_states` now starts without map_data and
without the lock, with map_data as the fallback. If desyncs drop with this, the shared temp
map was the wrong-version path.

## Pro 1v1 extraction: what broke and what fixed it (2026-10-30 evening)

Two player views per game (no observer; the oracle is the two views concatenated), 629 games,
1258 views, builds 93333 / 94137 / 95841 (+2 on 95299 / 96883), `extract_states.py --views
players --step 4 --workers 12`. Ten runs went wrong before one worked; the causes, in the
order they were found, all in `scripts/extract_states.py` and `envs/wine_run_config.py`:

1. **Finished views were discarded.** When a replay ends, the engine's status turns `ended` and
   the next step is refused with pysc2's "`step` called while in state: Status.ended"; the end
   handler only knew the older wording "already ended", so a view that had run the whole game
   was logged as a `fail` and its frames dropped. Now: that error with frames extracted ends
   the view; with no frames it is "replay ended at start" and retried (some starts do end the
   game at once, intermittently per start, with or without map_data; a retry gets through).
2. **Concurrent Proton launches.** Eight `umu-run` launches at once each run Proton's prefix
   setup, which restarts the prefix's wineserver under the others; the losers hang at the
   umu.exe stage on a dead fsync futex, forever. Launches are serialized behind the file lock,
   held from `run_config.start` through `start_replay` (~20-25 s each); stepping is concurrent.
3. **No wineserver between engines.** With the Battle.net app closed the wineserver exits with
   the last Wine process and the next launch reboots the prefix. `keep_prefix_alive()` starts an
   idle Notepad before the workers (awaited until the wineserver exists) and terminates it at
   the end. Killing a keeper by hand kills the wineserver it owns and every engine on it.
4. **Six-hour timeout.** pysc2's controller timeout here is 6 h, so a hung engine parked its
   worker silently. `ENGINE_TIMEOUT = 90` s for launch + replay start, `STEP_TIMEOUT = 30` s per
   request afterwards (set on the websocket), both turned into retried "timed out" errors.
5. **Launcher trees left behind.** pysc2 terminates only the direct child; `WineProcess` now
   launches in its own process group and `_shutdown` kills the group.
6. **Raw websocket reads.** The per-loop fog-path observation used `c._client._sock.recv()`
   directly; one stall was caught with the worker blocked there. It goes through `c.observe()`
   now (same parse cost).
7. Diagnostics: workers' stderr in `<out>/workers.log`, `kill -USR1 <worker>` dumps its Python
   stacks there, `retry` / `fail` lines carry view, error and elapsed seconds.
8. **The shared temp map.** A start with map_data writes the map to the Windows user's temp
   folder (`AppData\Local\Temp\StarCraft II\TempLaunchMap.SC2Map`, one overflow slot `~00`,
   plus TempKeyFrameP*), which the engine keeps reading after `start_replay` returns; starts
   queued back-to-back behind the lock collided with the previous engine's load and were
   refused with `InvalidReplayData` (bursts after many workers finished together, a cascade
   through the retry sleeps; 100 spread-out starts in between had none). A private Windows
   %TEMP% per engine did nothing (Wine ignores TEMP/TMP; the engine writes to the profile's Local
   AppData), and a Windows user per engine is not possible under Proton (the user is pinned to
   steamuser). Run 13 settles instead: the launch lock is held 8 s after `start_replay` returns
   (`LAUNCH_SETTLE`) and the temp folder is emptied before a run. 0 fails, 0 retries, but the
   serialized starts cap the run at ~2.5 views/min with 12 workers.
10. **A Wine prefix per worker** (`wine_run_config.ensure_pool_prefix`, `~/Games/sc2-prefixes/<k>`):
   a copy of the base prefix (~600 MB: registry, `drive_c/windows`, the user profile, dosdevices,
   the `pfx -> .` link) with the two game installs and the Battle.net map cache symlinked back,
   the launcher dropped and the copied temp folder emptied. Worker k sets `WINEPREFIX` to prefix k
   before building its run config; `installs()` then resolves the installs through that prefix's
   drive_c so the engine's C: paths are unchanged. Each prefix has its own wineserver, temp slots
   and registry, so the launch lock is per prefix (no contention), there is no settle and no
   keeper. Tested by hand first (`~/Games/sc2-prefix-test`: own wineserver, temp pair in its own
   folder, a view extracted); run 14 is the first run on the pool. `--no_prefix_pool` restores
   the shared-prefix path. `SC2PATH` must stay unset (it would bypass the prefix-relative lookup).
9. The selection tracker sliced `start[i + 1]` on the action table's last row (no trailing
   sentinel in the stored `*_start` columns); only reached once views ran to the end. Fixed.

Measured with 12 workers (run 11): 115-135% CPU per engine, 2.2-3.0 GB RSS each, one
wineserver, 68-577 s per view (median 217 s for a median 13.9-min game), ~2.6 views/min, no
failures or retries in the first 11 minutes; alone a 10-min game took 73 s, so the engines
contend about 2x. About 8 hours for the set. The RAM-backed `/tmp` holds the sessions' scratch
folders; with 12 engines that memory matters (2-3 GB were freed).

## Extractor layout 2: everything the raw API reports (2026-10-30 night)

Taking stock of the state before the pro set is redone: the model saw 18 numbers per entity,
nine tracker stats per player (every 160 loops) and a 16x16 visibility map; the files also held
facing, weapon cooldown, engaged target, cargo, add-on, buffs, passengers, the full order queue,
effects, creep and the map grids, none of which reached the model; and the extractor never read
rally points, exact per-frame resources, buff timers, score details or anything about ability
cooldowns. The extractor now takes everything the observation offers (`meta["layout"] == 2`,
the layout list in scripts/extract_states.py), so the redo is the last one for missing data:

- raw Unit: rally points (`rally_*`, ragged; town halls have two), upgrade levels, the buff
  timer (`u_buff_remain/max`), is_active / blip / on-screen (`u_flags2`), ideal harvesters,
  cargo capacity, detect and radar range.
- the observed player per frame: `pc_*` (player_common: exact minerals, gas, food, idle workers,
  army count, warp gates, larvae), `sc_*` (score details: collected / spent / killed / lost /
  friendly-fire resources per category, unit and structure value, damage dealt / taken / healed
  per vital, idle times, apm) and the camera. Zeros in the observer view.
- the raw player block: power sources (`pw_*`), radar rings (`rd_*`), upgrade ids (`pu_*`).
- the tracker: all 39 PlayerStats fields, not nine (`p_minerals_lost_army`, ...).
- **cooldowns**: the API has no cooldown field, but `RequestQueryAvailableAbilities` lists what a
  unit can use right now; a used Blink, Stim or bile is absent until its cooldown ends. One query
  per frame for the observed player's visible units (`ab_id`, `ab_point`, ragged per unit row).
  Costs ~13% per view (17 s vs 15 s for 90 s of game at step 4). The observer view gets no
  answer for anyone (tested: 0 rows for all three owners) and skips the query; it also gets no
  rally points and no power sources, private details like production and cargo, which is one
  more reason the oracle is the two player views concatenated. Still to confirm on a full game:
  that a unit on cooldown really drops the ability (the 90 s test used none).
- not taken: alerts, chat, the feature-layer / render data.
- the map's start locations in `meta`.
- **interface memory from the replay**, per frame like the control groups
  (`selection_tracker.frame_memory`): camera hotkey positions (`cam_save[F, P, 8, 2]`; all
  eight slots are used in pro games), the selection's active subgroup (`subgroup[F, P]`, the
  tab index; 515 tab presses in 25 replays, reset to 0 on a group recall by assumption), the
  player's last command (`lc_*[F, P]`: what a repeat re-fires), and autocast toggles as an
  event list with their recipients (`ac_*`; command flag 0x40 with 0x80 = on; rare, 3 in 25 pro
  replays, e.g. Repair on for two SCVs). Also units lost / killed and resources lost: not
  stored as such because the per-frame death list, the dead units' rows and the tracker's lost
  / killed fields already give them exactly; the tokenizer accumulates them.

Measured on 90 s of a pro game (both player views): all new arrays filled where the game had the
thing (889 rally rows, 689 of them onto a mineral patch; 17 nonzero score fields; exact
minerals per frame), file +15% (591 KB vs 516 KB; the abilities table compresses to 31 KB).
Nothing is featurized yet: the tokenizer still reads the layout-1 columns and the first order
only; the full order queue, rally, buffs, cooldowns, weapon cooldown / engaged target,
passengers and the map grids are the model-side work that follows.

## The corpus layout: replays and games in mirrored trees (2026-10-31)

Outputs used to be flat folders of `<sha1 of file name>.npz` (`obs_pro`, `pov_pro`,
`actions`, splits of file names). Now (worldmodel/corpus.py, scripts/ingest_replays.py):

    data/replays/<format>/<build>/<map>/<id>.SC2Replay + <id>.json     the corpus, one tree for
                                                                       every source
    data/replays/index.json, maps.json                                 every sidecar; the map
                                                                       directories by name; rebuilt
    data/games/<format>/<build>/<map>/<id>.p1.cst .p2.cst              the player views
                                        <id>.acts.npz                  the action table
                                        <id>.meta.json                 per view: layout, step,
                                                                       substep, stats, commit
                                        <id>.p<slot>.fail.json         a failed view
    data/games/splits/<name>.json                                      game ids
    data/games/logs/<run>.log, .json, .jobs.json                       per extraction run

`<id>` is the content hash of the replay (first 16 hex of sha1), so a game in two packs is one
game with both packs in its sidecar, and a renamed file is not a new game. The format (1v1,
2v2, ..., ffa) comes from the replay's teams, the build is the base build, the map is the first
16 hex digits of the map file's hash (the replay's cache handle, what the engine loaded), not
its name: a tournament's own version of a ladder map or a map's next revision is a different
map to the engine and to the model, and the same file under two names is one. The sidecar has
the name, players, races, MMR, result, duration and date; `maps.json` lists the directories by
name with their game counts and builds. Pruning: a format, a patch or a map is one `rm -rf` in both trees; anything
else (a tournament, a player, games under five minutes, failed views) is a query over the
sidecars or `index.json` that yields ids, then those ids' files. `jq` examples:

    jq -r 'to_entries[] | select(.value.sources[0].pack | test("HSC")) | .key' data/replays/index.json
    find data/games -name '*.fail.json'

There is no dataset level: one extraction per game, and the extractor compares each view's
sidecar (layout, step, substep) with its settings, so a layout change is an incremental redo
in place (`--redo` forces one; `--retry_failed` retries views with a fail file). The observer
view is gone from the outputs (the oracle is the two player views concatenated). The action
table is written by the extractor itself, once per game, next to the views
(scripts/extract_actions.py is gone). The source pack is in the sidecar, not the path, since
it means nothing to training.

`--maps` on the extractor and make_splits take names, slugs or hash prefixes; a name selects
every version of that map (`maps.json` shows them: in the pro pack "Tokamak" and "Tokamak LE"
are different files, and "[TLMC21] Tourmaline" is not "Tourmaline LE").

Ingest: `python scripts/ingest_replays.py <dir | archive | file>...` sorts replays into the
tree (loose files are moved, archives unpacked with 7z and left in place, duplicates
collapsed, unreadable files listed and left). The pro 1v1 pack went in on 2026-10-31: 629
games, builds 93333 (65), 94137 (371), 95299 (1), 95841 (191), 96883 (1). The older 2v2
extractions (`obs_1v1`, `pov_1v1`, `actions`, `splits_1v1.json`) stay as a legacy flat layout
the loaders still read.

## Packed state files (.cst, 2026-10-31)

State files were `np.savez_compressed` (zlib on each raw column). On a median pro player view
(8.6 MB from 253 MB raw) the bytes went to the visibility maps (20%), unit x / y (23%), ids,
facing, hp and the ragged start indices. Measured alternatives on that file, all bit-exact:

| scheme | size | write | read |
|---|---|---|---|
| zlib on raw columns (before) | 8.6 MB | 1.6 s | 0.3 s |
| xz on raw columns | 5.2 MB | 10.6 s | 0.3 s |
| zstd 9 on raw columns | 6.9 MB | 0.5 s | 0.06 s |
| transforms + zlib | 2.8 MB | 0.6 s | 0.4 s |
| transforms + zstd 9 (now) | 3.2 MB | 0.4 s | 0.4 s |
| transforms + zstd 19 | 2.3 MB | 5 s | 0.05 s |

The transforms (worldmodel/statefile.py) are what matter, the codec is second: per-unit deltas
(each unit column minus the same unit's value in the previous frame, matched by id, so a unit
standing still is zeros; float32 columns are exact multiples of 1/4096 and go to int32 fixed
point, float16 columns are delta'd as bit patterns), XOR of consecutive visibility / creep
frames, and consecutive differences of the monotonic start arrays. Parquet and Arrow IPC were
tried too (pyarrow, since removed): on raw rows they match zstd (6.9 / 6.7 MB) and their own
delta encodings make things worse because consecutive rows are different units; on rows sorted
unit-major Parquet's delta gets to 3.4 MB; on the transformed columns they match the per-array
blobs. So the container stayed a per-array one (JSON header + one zstd blob per array, arrays
decoded lazily on access; `statefile.load` also opens the old .npz files). `u_id` is the one
unit column kept raw: decoding the deltas needs it, and the engine's row order is only half
stable between frames, so it is 20% of the file. Sorting rows by id within each frame would
remove most of that but changes the engine's order, which was left as is. The 3.2 MB here vs
2.6 MB in the first measurement is that: the experiment delta'd u_id by id, which cannot be
decoded. With 100k replays the difference between the old and the new format is tens of TB.

`zstandard` is a project dependency now. Loaders: `tokenize.GameFrames`, the training scripts,
`vocab.build` and `check_player_commands` go through `statefile.load`; the action tables
(replay_actions / extract_actions output) stay .npz.

## Confirmed: the API does not report a worker carrying rich vespene (2026-10-30)

The game data defines distinct carry buffs for every resource (`CarryMineralFieldMinerals`,
`CarryHighYieldMineralFieldMinerals`, `CarryHarvestableVespeneGeyserGas[Protoss|Zerg]`,
`CarryHarvestableRichVespeneGeyserGas[Protoss|Zerg]`, Terrazine variants), but the engine's API
buff table (RequestData, `data/worldmodel/rules/<ruleset>.json` `buffs`) lists 186 of the 552
`CBehaviorBuff` ids in the catalogs, and none of the Rich or Terrazine carry buffs. Checked in
the extracted 2v2 games: next to regular geysers workers show the vespene carry buffs by the
hundreds (604 Zerg, 424 Protoss, 154 Terran worker-frames); next to built rich extractors /
refineries / assimilators (12 games, 2002 worker-frames within 3 units) no carry buff appears
at all, only 33 mineral-carry buffs of workers passing by. So a worker returning rich gas
carries no reported buff; the only cues are the gather target's unit type (RichVespeneGeyser /
ExtractorRich / RefineryRich / AssimilatorRich) and the player's vespene rate. The wider point
for buff features: the API exposes about a third of the game's buffs; the rest are invisible
in the observation.

### Deferred: isolating engines from each other (containerizing), sketch (2026-10-30)

Every engine-vs-engine problem tonight came from sharing one Wine prefix: the wineserver
(Proton's setup restarting it under concurrent launches), the Windows user's temp folder (the
numbered TempLaunchMap / TempStartReplay slots that run out), and the registry. Options, cheapest
first:

1. **Per-engine Windows user in one prefix.** Wine names the profile after the Unix user; a
   per-engine `USER` / `WINEUSERNAME` would give each engine its own `drive_c/users/<name>`
   (Local AppData, Documents). Being tested in the extraction session. Keeps one wineserver.
2. **Per-engine Wine prefix, shared install.** A pool of prefixes `~/Games/sc2-prefix-<k>`, each
   a copy of the base prefix's small parts (`drive_c/windows`, `users`, registry files) with
   `drive_c/Program Files (x86)/StarCraft II[ old]` and `ProgramData/.../Battle.net/Cache`
   symlinked to the shared ones (Wine follows symlinks). Each prefix has its own wineserver, temp
   folder and registry, so no launch lock, no keeper and no slot walk; Proton runs its prefix
   setup once per prefix. Cost: ~100-200 MB RAM per wineserver + services, a few seconds of
   prefix boot per engine (once), ~100 MB disk per prefix. `wine_run_config` would hand worker k
   prefix k (WINEPREFIX per launch); the extractor's workers are already one engine each.
   **Done** (item 10 above): `ensure_pool_prefix` / `--slot`, one prefix per worker.
3. **A bind mount per engine** of a private directory over `drive_c/users/steamuser` (bwrap /
   unshare -Ur around umu-run). Nests a user namespace inside pressure-vessel's; works on
   Ubuntu if unprivileged user namespaces are allowed, but fragile across umu updates.
4. **Full containers** (podman with the GPU and the X socket passed through, Wine inside):
   complete isolation, heaviest to set up and to keep in step with Proton / DXVK.

Option 2 is the one to build if option 1 fails: it removes the whole class of problems rather
than the current instance, and it costs little.

**Longer term (user's direction):** full containers with input isolation and a scheduler.
Each container runs a headless Wayland compositor (Weston headless backend on the GPU's render
node) with Xwayland for Wine, so DXVK presents as it does on the desktop but no screen or input
device is involved; the container holds Wine, a private prefix, the mounted install with every
build and a small agent that launches an engine on request and publishes its API port. The
Python that drives replays stays on the host (pysc2's controller connects to a remote
host:port), so the containers are pure simulation nodes. A host controller keeps a job queue
(replay + seat + build now; game lobbies later) and a pool of nodes. For self-play the API's
multiplayer path applies directly: one engine creates the game with participant slots, the
others join with agreed ports, so container c1 serves seat 1 of game g1 and c2 seat 2 on a
shared pod network, and the agents sit outside as API clients, which lets policy inference be
one batched GPU process over many games. Memory rises by a wineserver and compositor per
container; GPU access through the NVIDIA container toolkit.

**Input isolation, refined (user):** agents are eventually to play through simulated keyboard
and mouse, so each container needs its own seat and a way to inject into it. Pick a wlroots
compositor (Cage or Sway, headless backend on the GPU render node) rather than Weston: its
virtual-keyboard and virtual-pointer protocols let a client in the container type and move the
mouse as ordinary Wayland requests, Xwayland forwards them to Wine and the game, and the
screencopy protocol gives the agent's frames at render resolution. Alternatives: XTest against
the container's Xwayland (compositor-dependent; GNOME's ignored it today), or a uinput virtual
device (most faithful, needs the device and permissions inside the container). One seat per
container means physical devices never reach the game and injected input cannot cross
containers. The API stays alongside for rewards and ground truth.

## Pro 1v1 set: extracted (2026-10-31, run 16)

The 629 tournament games (two player views each, no observer) are in `data/games/1v1/` as
layout 2, step 4, substep 1: 1258 `.cst` views, 629 `.acts.npz` action tables, 629 `.meta.json`
sidecars, 6.2 GB. Run `20261031-000244` (`extract_states.py --format 1v1 --step 4 --substep 1
--workers 12 --retry_failed`, logs in `data/games/logs/`) took 6 h 16 min on the 12-prefix pool:
0 failed views, 1 retry (a `start_replay` websocket timeout, passed on the second try), no
`InvalidReplayData`, no "ended at start", no duplicate views. The only tracebacks in the run log
are umu-run's `ProcessLookupError` at engine shutdown.

| build | games | views | ruleset(s) |
|---|---|---|---|
| 93333 (5.0.14, DH Dallas 2025) | 65 | 130 | 4742e079 56 games, 01973d3e 7, two singletons |
| 94137 (HSC 27, EWC 2025, FEL Cracov 2025) | 371 | 742 | 715f8b55 343, 690bc661 9, bf25efab 7, fc4e1bac 6, 5d683877 5, one singleton |
| 95299 | 1 | 2 | 39c4524e |
| 95841 (HSC 28, pigosaur 62, random-replays) | 191 | 382 | b982a64c 151, 1f369ff1 19, e25ffd56 9, be1c3a89 6, 018b3cff 5, one singleton |
| 96883 | 1 | 2 | 27099e7b |

Every one of the 18 rulesets has its balance export (`data/worldmodel/balance/status.json`), so
every view's abilities are named. Per view: mean 206 s, median 169 s, max 1520 s (a 25-minute
game); 93333 games are the longest (median 203 s), 95841 the shortest (158 s). Output 5.2 MB per
view on average (median 3.9 MB). Throughput with 12 engines 3.6-3.7 views/min on 94137/95841
and 2.6/min in the first hour of 93333 games; each build changeover costs a few minutes (its
warm-up game runs alone, and a single-game build runs both views alone). Memory with 12 pool
engines: 37-51 GB used (engines 2.5-3.3 GB each, workers 0.8-0.9 GB each), lowest 7 GB
available on a 64 GB box; 10 workers would be the safe setting for longer games. Rerunning the
same command does nothing (every sidecar matches); `--redo` re-extracts, `--retry_failed`
retries fail files. The pool prefixes stay in `~/Games/sc2-prefixes/00-11` (5.6 GB, remade on
first use if deleted); `~/Games/sc2-prefix-test` (529 MB) was the hand-made trial copy and can go.

## Horizons and times in game loops (2026-10-31)

Decision (user): the model is told game loops, never frame numbers. `train_direct.py` draws
horizons from the multiples of the extraction's frame step up to `--max_loops` (default 56 =
2.5 s): on the step-4 pro set that is 4, 8, ..., 56 (a quarter of the per-loop horizons, fine for
a first run); a step-1 extraction gives every loop 1..56 with no code change. `--eval_loops`
(default `4,16,32,56`) must be multiples of the step. Each sample's horizon is the loop difference
between its two frames, and the predictor, the action encoder's summary query and the inverse
model read it through `LoopTime` (action_encoder.py: sin/cos at periods 2-2048 loops plus the
signed value / 56). Command times and segment bounds inside the action window are loops
relative to the present as well (they were seconds). Checkpoints from before this change don't
load (the time features' width changed). The run's `config.json` records `step_loops` and
`hor_loops`; each view's extraction settings are in its `<id>.meta.json` sidecar (layout, step,
substep, until, abilities, extractor commit, stats).

Smoke test (200 steps, pool 8, a scratch split holding out Ruby Rock LE): runs end to end on the
pro player views and evaluates at 4 / 16 / 32 / 56 loops; at 200 steps it sits at persistence, as
expected.

**Step-1 extraction, measured** on one median game (14b0d10112accb33, 10.7 min, build 94137, two
views, 2 workers on an idle box): 14,105 frames per view, 11.4 MB and 9.2 MB (3.1x the step-4
files, 3.7 / 2.9 MB), 101-105 s per view against 170-176 s at step 4 in run 16 (12 workers,
loaded box; the engine already stepped every loop at `--substep 1`, so writing every frame costs
little). So the full pro set at step 1 is roughly 19 GB and a run no longer than run 16.

## Pyramid encoder and the layout-2 inputs (`worldmodel/pyramid.py`, 2026-10-31)

Decisions (user): every field layout 2 extracts becomes a model input; the encoder is the
scatter-and-local-attention pyramid discussed on 2026-10-30, with static and dynamic map layers
not yet separated (what that split should mean is still open).

**Inputs** (`tokenize.py`):
- Entity numbers grow from 18 to 49 (`ENTITY_NUMS`): facing (sin, cos), radius, weapon cooldown,
  engaged target (flag + offset), attack / armor / shield levels, buff remaining fraction and
  duration, ideal harvesters, cargo used / max, detection and radar range, display state (3),
  cloak state (4), the second flag set (active, blip, on screen), passenger, in gas, abilities
  known (the viewer's own units when abilities were queried), has an add-on.
- List-valued state pooled into the entity token (`BAGS`): every order of the queue (ability +
  target offset, targets a unit, progress, position), the production queue (ability, progress,
  position), buffs (raw id, 512-row table), available abilities (raw id, 4096-row table, point
  flag), rally points (offset, targets a unit). Per list: mean of the items' embeddings and the
  log count, through a linear map. (A plain sum made a unit with 40 available abilities 40x
  louder; the mean + log count keeps the count without the magnitude.)
- Units inside units are rows of their own with a pointer (parent) to the container: passengers
  from the extraction's passenger lists (transports, bunkers, command centers; same unit id as on
  the map, so identity survives loading and unloading; position = the container's; max hp from
  the unit's own map rows or the type's largest), and workers inside gas buildings, bridged while
  out of sight for at most 64 loops when their order before the gap was the gas building. Add-ons
  point at their building. The pointer adds the container's token through a linear map. On the
  median test game: 1,232 passenger rows, 5,696 in-gas rows, 6,989 add-on links over 3,526 frames.
  A loaded unit is now "present" at t + k (it was "gone"), so the targets follow it through a pickup.
- Effects (storm, bile, ...) are tokens: effect table (64) + owner + position + radius. Not predicted.
- Player tokens: tracker stats (9) + player_common (10) + score details (78) + units lost (from the
  deaths and owners) + known flag + camera xy + subgroup + last command (age, point, x, y, targets
  a unit) = 107 numbers; plus position embeddings of the camera and the mean of the saved camera
  positions, and the last command's ability family. Private parts are the viewer's only.
- Map grids at one cell per map unit: visible, seen, creep, pathable, placeable, height, on map,
  playable (`GRID_CHANNELS`). Orientation checked: rows are y, no flip (own units on visible cells
  100% vs 0% flipped; hatcheries on creep 100% vs 0%). The batch canvas is the largest map padded
  to a multiple of 16 (Pylon LE 248 -> 256).

**Encoder** (levels by cell size in map units):
- Stem: two 3x3 convs over the grid channels -> width 32.
- Every level (decision, user, same day): 3 conv blocks on the map (depthwise 3x3 + LayerNorm +
  MLP, residual; `--conv_blocks`), then 3 attention blocks (`--attn_blocks`).
- One width throughout (decision, user, same day): map cells have the units' width d at every level,
  so units and cells attend together at full width with no projection. d is 128 everywhere (units,
  cells, players, the action encoder, predictor and latent heads; `--d 128`), 8 heads of 16 (`--head_dim`;
  the predictor gets the encoder's head count, since it reads the encoder's rotations). At head width
  16 each head's 2D RoPE has 4 wavelengths per axis (1 to 256 map units) instead of 8.
  (The first version grew the cells 32 / 64 / 128 at the local levels and projected units down into
  them, which left a unit's exchange with the map at the 256-cell grid one 32-wide head.)
- s = 1, 2, 4: each attention block is one round of local joint attention: each cell attends to its sliding 3x3 neighborhood of cells, the units / effects in
  those cells, CLS + players and a learned null key; each unit attends to the 3x3 cells around its
  cell, the units in them, the globals and the null key; the globals attend to every cell and unit.
  No map-to-map attention beyond the neighborhood. Units keep width d (projected into the level's
  width for attention and back); cells pool by 2x2 patch merge; units are reassigned to the coarser
  cells. Implementation: neighbors as shifted views of the padded map, unit keys as sparse pairs,
  one softmax per query merged across the dense and sparse key sets (`_segment_softmax_merge`);
  checked against a dense masked-softmax reference to 2e-7 (`tests/test_pyramid_local.py`).
- s = 8, 16 (width d): full joint attention over CLS, players, a null token, every map cell and
  every unit / effect.
- No additive scatter of units into cells: cells learn about units only through attention, and the
  null key is their only density signal (an additive scatter per local level is the open option).
- RoPE in map units everywhere (centered on the playable area), CLS / players / null at the origin.
- Output to the predictor: [CLS, players, units, effects, s = 16 cells]; probes and z read the
  units' rows (`unit_slice`).

`train_direct.py --encoder pyramid` (default; `--encoder flat` keeps the October units-only encoder,
which also reads the new entity numbers and player numbers but not the lists, pointers, effects or
grids). With 1 conv + 1 attention block per local level and 2 full blocks per full level (the first
version): encoder 8.5 M, 0.51 s/step at 16 frames x 2 horizons, 8.6 GB; at 400 steps (same seed)
training loss 1.36 vs the flat encoder's 1.41, both still at persistence on held-out positions.
With 3 + 3 at every level and cells 32 / 64 / 128: encoder 17.5 M (30.3 M in all), 1.35 s/step, 6.8 GB
with activation checkpointing per block. With 3 + 3 and 256 everywhere: encoder 28.9 M (41.8 M in all);
16 frames x 2 horizons runs out of the 32 GB; 8 frames: 3.3-3.75 s/step, 21.5 GB; 4 frames: 1.7-2.3 s/step,
11.1 GB (about 2.4 frames/s, against 11.8 frames/s for the narrower version). With d = 128 everywhere
(the default): encoder 7.7 M (11.0 M in all), 16 frames x 2 horizons at 3.2-3.45 s/step, 21.4 GB
(about 5 frames/s); the 256-cell grid at full width is most of the cost. Flat (d 256): 0.10 s/step, 2.2 GB.

## Combined view, enemy intent for the posterior, longer horizons (2026-10-31)

Run 2026-10-31 13:28 (pyramid, d 128, stopped at step 2000) beat persistence clearly on held-out
maps (moving units 2.56 vs 4.28 map units at 56 loops; orders 0.94 vs 0.87), but z collapsed: KL
~0 with commands and without. Two causes in the code: KL was averaged (and charged) over the
summarized players only, the viewer in a player view, so the enemy's z was never regularized
or measured; and nothing pushed z to carry the enemy's choices, which at these horizons are a
small share of the loss. Changes for the next run:

- **Views.** A game is its two player views plus their combined view (`tokenize.CombinedFrames`).
  The combined view is the two views side by side, not merged: every token of both (a unit both
  players see is two tokens; a snapshot one player keeps of minerals the other saw destroyed
  stays), each marked with its source view (`ent_src` / `eff_src`, that viewer's player code,
  embedded) and with relations as that viewer sees them. Unit ids stay per view (the second
  view's offset), so each token's target is its own view's future; replay pointers resolve to
  either copy (`alias`). Both players' commands are given. `--obs_share` (1/3) of present frames
  come from it.
- **Minimap.** Visibility, seen and creep layers per player (slot order); the encoder embeds each
  player's layers with that player's code and sums them into one grid before the stem
  (`PyramidEncoder.minimap`), so "seen by this player, not that one" is learnable. `ply_gridok`
  says whose layers are present (the viewer's in a player view, both in the combined view).
- **Enemy intent, posterior only** (`inverse_model.EnemyIntent`, `train_direct.enemy_info`). In a
  player view, for each enemy unit visible at t (not a snapshot, not inside anything): its true
  order queue at t and the enemy's commands in [t, t+k) that selected it (stim, blink, moves),
  read from the enemy's own view (units joined by engine tag: visible units share tags across
  views, snapshots don't). Pooled per unit and added to the present tokens the posterior reads.
  The prior never sees it, so KL charges for the enemy intent the prior can't predict.
- **KL** is charged and reported for every player; the prior's commands-given flag is per player
  (a player view never gives the enemy's). Evaluation splits KL into own (summarized players)
  and enemy, adds `pos err enemy mv` (moving enemy units), and evaluates player views and the
  combined view separately (`<set>` and `<set> obs`).
- **Horizons** up to `--max_loops 224` (10 s), drawn log-uniformly (`--hor_dist log`); evaluated
  at 4, 16, 56, 112, 224 loops. Births per window grow with the horizon (~35 at 224) and aren't
  predicted.
- Pool: `--pool 12` games (~1.75 GB each with three views); `--val_games 2`.

Smoke test (50 steps): 5.2 s/step, 27 GB peak.

## Dense windowed encoder (worldmodel/dense.py, 2026-10-31)

The pyramid's sparse local attention kept the GPU memory-bound: ~300 W of a 600 W 5090 at
"100%" utilization, 5.3 s/step (torch.compile: 4.1 s, 20 GB instead of 27). Neighbor shifts,
pair searches, gathers, segment softmaxes and depthwise convs each touch every number once or
twice. `DenseEncoder` (now `--encoder dense`, the default) rebuilds the map side from dense parts:

- **Canvas cropped to the playable area** (`GameFrames.grid_origin`: the playable corner rounded
  down to a multiple of 16; `collate` passes `grid_origin`). Borders were 40% of a typical full
  map and 70% of Pylon's; playable areas sit ~6 units low on most maps.
- **Minimap embedding** as before (static layers + each player's layers with their code, summed).
- **Dense ResNet blocks** (3 x 3 convs, per-cell LayerNorm, identity skips): 2 in the stem, 3 per
  level. Cells outside the frame's own cropped map are zeroed after every map step and between
  a block's two convs. With GroupNorm (statistics over the whole canvas) and unmasked padding, a
  frame's encoding depended on the largest map in its batch: a Taito Citadel frame batched with a
  Mothership frame changed by up to 3.9 (`tests/test_dense_batch_independence.py`; now 1e-5 with
  TF32 off). Zeroing between the convs matters because a norm turns a zero cell into its bias,
  which the second conv reads at the map's edge.
- **Levels at 1, 2, 4, 8, 16 map units per cell**, each with 3 conv blocks then 3 attention
  blocks; 2 x 2 downsampling between them with a projection shortcut (learned 2 x 2 stride-2
  merge + linear of the 2 x 2 average). Every level's map is returned in `maps` for a later
  full-resolution decoder (U-Net skips).
- **Attention** (FlexAttention, one softmax per query over [globals, entities, cells]): a cell
  sees the 16 cells of its fixed 4 x 4 window, every entity and the globals; entities and
  globals see everything. Cells are laid out window-major so the block mask is block-sparse.
- **No pair biases.** A learned geometric bias inside the kernel (the gap between bodies, and
  between a body and a cell's square) was built and dropped: even its cheapest form (fixed
  per-head scales, learned per-query strengths) was ~40% of the full-resolution block, scalar
  math on CUDA cores for ~1.5 billion scores per block. Measured at level 1 (16 frames, 26k
  tokens, backward): no bias 8 ms, fixed bias 14 ms, one per-query strength 16 ms, two 32 ms,
  a learned lookup table 401 ms, a small learned network 3.7 s (every score adds into a few
  weights with atomics). Rotary angles to 1/8 map unit with learned q / k already give a learned
  function of the offset, and the radius is in each entity's features; the tight-passage metric
  checks what's lost. (FlexAttention notes: per-head weights must be 1-D tensors indexed by the
  head; a captured Python float turns symbolic under dynamic shapes and breaks the backward.)
- **Block mask** built from the structure (`DenseEncoder._block_mask`: the dense part and the cells
  padded to 128-token blocks; cell blocks see the dense blocks fully and themselves through the
  window rule). The generic builder evaluated every pair: 18 s per new entity count.
- **Positions** at 0.125-512 map units: rotary angles (encoder and predictor) and the entity
  position embedding (`MapFourier`, from map units instead of fractions of the playable area).
- d 128 everywhere, 4 heads of 32 (`--head_dim` now defaults to 32).
- Tests: `tests/test_dense_window.py` (the attention block vs a dense masked softmax: 5e-7;
  gradients 2e-7).
- Encoder alone, 16 frames, activation checkpointing: forward 0.13 s, forward + backward 0.81 s,
  5.3 GB (after a one-time ~15 s compile).

**Fine-scale passage evaluation** (`train_direct.tight_units`): moving army ground units whose
body is within 1 map unit of unpathable terrain (a distance map per game) or a structure's
edge, split by radius (`pos err tight sm`: < 0.5, zerglings and marines; `pos err tight lg`:
>= 0.75, tanks, archons, ultralisks). Gaps formed by units holding a position (a stalker in a
wall) aren't flagged yet.

**Step time** (`train_direct.py`, 16 frames x 2 horizons, the same 100 steps and data):

| encoder | s/step (steps 50-100) | peak memory |
|---|---|---|
| pyramid (sparse local attention) | 5.27 | 27.3 GB |
| pyramid, its blocks compiled | 4.13 | 20.0 GB |
| dense, learned pair-bias network in the kernel | 3.7-3.9 | 13.8 GB |
| dense, pair bias as 2 per-query strengths | 2.1 | 13.7 GB |
| **dense, no pair bias, everything compiled (default)** | **0.74** | **13.4 GB** |

Everything compilable is compiled by default (`--no_compile` turns it off): the encoder's ResNet,
downsample and attention blocks, the predictor, posterior, prior, probes and the enemy-intent
pooling. The first ~25 steps include compilation (~5 s/step). Without activation checkpointing
the step runs out of memory (32 GB). `--profile N` prints one step's GPU time by kernel; at
2.1 s/step the attention backward was 41% and its forward (run twice under checkpointing) 20%.

## Latent dynamics model (`--model latent`, worldmodel/latent_dynamics.py, 2026-11-01)

The design is in `DESIGN_latent_dynamics.md`; this is what's built (everything except the
engine-command teacher, which waits for the per-loop re-extraction):

- **Residual dynamics in the encoder's space:** `ŝ_{t+k} = s_t + h(s_t, k) + g(s_t + h, A, z)`.
  h (`PassiveDynamics`, `--h_depth 4`): self-attention over the state tokens with the encoder's
  rotary angles, conditioned on the horizon. g (`ActionDynamics`, `--g_depth 2`): the state tokens
  attend to the replay action tokens and the players' z tokens (queries only; actions never read
  the state). Both end in zero-initialized projections: a fresh model predicts persistence.
- **Targets:** every present token's counterpart in the encoding of frame t+k (globals by
  position, units by id via `fut_row`, cells by grid position; effects have none), stop-gradient,
  from the online encoder (`--target online`, the default) or a moving-average copy
  (`--target ema`, `--ema_tau 0.996`). VICReg and stop-gradient already guard against collapse;
  the moving average adds stability if it's needed. Either way the future frame is encoded
  without a backward pass.
- **Losses:** latent prediction (per-token MSE, `--w_lat`), VICReg on s_t (variance floor 1,
  off-diagonal covariance; `--w_var`, `--w_cov`), decorrelation of g's delta from h's delta and
  from s_t (cross-correlation to zero, `--w_dec`), a floor on g's spread over samples with
  commands (`--g_floor 0.05`, `--w_gvar`; g = 0 is trivially uncorrelated), Barlow Twins between
  the posterior's z and the players' command summaries (`--w_inv`, `--inv_off`), the KL.
- **Probes** decode ŝ_{t+k}'s unit tokens on detached features, so the evaluation tables still
  read positions, orders and deaths against persistence; `--probe_weight w` lets a share w of the
  probes' gradient reach the latent (the "small decoder weight" escape hatch; the probes still
  train at full strength). The train-vs-validation loss is the
  latent loss.
- **Speed:** 0.52 s/step at 16 frames x 4 horizons, 10.7 GB (the direct model: 1.15 s, 24 GB).
- **Watch for:** collapse in time rather than across tokens. VICReg keeps tokens distinct but
  doesn't stop the encoder from encoding only what doesn't change (type, owner, terrain), which
  makes persistence a perfect latent prediction. The detached probes show it: position errors at
  or above persistence while the latent loss looks small. Remedies: `--probe_weight` at a low
  value (0.05-0.1), or a variance floor on the temporal differences s_{t+k} − s_t.

**Latent model, runs of 2026-11-01** (all 16 frames x 4 horizons, same seed and data order):

1. Plain latent model: latent loss flat at ~0.02 from step 25; probes at persistence. Read at
   the time as collapse in time; see 3: the probes couldn't have done better.
2. `--w_dinv 1` with yes/no targets (moved, touched, order changed, new order) from the latent
   change Δ = s_{t+k} − s_t only, with gradient through the future encoding (~1.0 s/step, 24 GB):
   the latent change rose (0.03 → 0.09-0.17), the dynamics beat persistence in latent space
   (latent loss / change ≈ 0.6), but the probes stayed at persistence.
3. **The probes were mis-posed.** Their targets are changes from t (dx, dy, ΔHP, ...), but they
   read only the future token, which doesn't say where the unit is now: the best they could do
   was "no change", i.e. persistence. Probes on the true future latent were at persistence too,
   while the Δ inverse head decoded displacement from the latent change with a low loss. Fix:
   the probes read [future token, present token] (`probe_in`, `probe_true_in`).
4. Displacement as **Barlow Twins** between the latent change and the unit's raw feature change
   (position as a fraction of the playable area + the 49 entity numbers, which are fractions,
   logs and /max by construction, so no batch scaling; `--w_dbar 0.05`,
   `--dbar_off`), instead of regressing dx, dy: every attribute that changed, aligned without
   decoding it.

Diagnostics logged every 25 steps: `chg` (how much the latent changes from t to t+k: persistence's
error in latent space) and `ratio` (latent loss / chg; below 1: the dynamics beat persistence in
latent space). Evaluation row `pos mv truelat`: moving-unit position error read from the true
future latent (with the present token), against the predicted one in `pos err moving`.
5. Run 4 at step 2,000 (held-out maps, moving units vs persistence): predicted latent −33% at
   16 loops, −13% at 56 (the true latent: −45%, −29%), but worse than persistence at 4 loops
   after being −31% at step 1,000 (the true-latent probes too); likely the representation
   drifting under detached probes, worst where changes are tiny. z unused (KL ~0.1, flat over
   horizons).

**Categorical z** (`--z categorical`, now the default; latent.py `cat_*`, `CategoricalPrior`):
32 one-hot variables of 32 classes per (sample, player) (~2^160 states; the design discussed
earlier, which the Gaussian z had stood in for). Straight-through samples, 1% uniform mixed into
both distributions (`--unimix`), DreamerV3's KL balancing with free bits: KL(sg(post) || prior)
at `--kl_dyn 0.5` trains the prior, KL(post || sg(prior)) at `--kl_rep 0.1` regularizes the
posterior, each free below `--free_bits 1` nat. Flattened to 1,024 and projected to the z token.
Evaluation: the posterior's mode, a prior sample, the prior's class probabilities ("mean").
`--z gauss` keeps the earlier diagonal Gaussian.

**Energy score** (`--es_k K`, `--w_es`): K futures from prior samples (straight-through, so the
prior learns), scored per token as mean distance to the true future latent minus half the mean
distance between samples (/ sqrt(d)). A proper scoring rule: averaging two futures (pressing vs
pulling back) is penalized, and so is a spread that doesn't match the real one. h runs once,
g K times.

**`--target ema` with `--w_dinv`:** the futures are encoded twice, by the online encoder (with
gradient, for the Δ head only) and by the moving-average copy (no gradient, for the latent
targets and the posterior): slowly changing targets against the drift seen in run 4.

**Run 6: categorical z + energy score + moving-average target, 17,000 steps** (`--model latent
--target ema --w_dinv 1 --z categorical --es_k 4`; 2,000 steps, then resumed to 17,000 on the
20,000-step cosine; `data/worldmodel/runs/direct_20261101_011415`). ~1.3 s/step, 25 GB.

- Training: latent loss 0.05 (step 500) → 0.017 validation (step 17,000), no train/validation gap.
  KL held at 0.7-0.9 nats per player throughout (the Gaussian z had ~0.1), higher with commands
  withheld and for the enemy. Online-probe readings: the predicted latent plateaued from ~step
  10,000 (−45% at 16 loops, −25% at 56); the true latent kept improving to the end (−49% at 56,
  −35% at 224 at step 16,000); 4-loop readings swung ±30% between evaluations (probe lag: the
  probes chase a changing representation, have no horizon input, and 4-loop moves are ~0.7 map
  units).
- **Probe refit** (`--eval_only --refit_probes 1000 --val_states 256 --eval_sets unseen-maps`):
  fresh probes with a horizon input (`probe_hor`, `probe_true_hor`) trained on the frozen
  step-17,000 model, evaluated on 256 states per horizon. Held-out maps, moving units vs
  persistence, posterior z:

  | horizon | player views: predicted / true latent | combined view: predicted / true |
  |---|---|---|
  | 4 loops | −45% / −53% | −41% / −52% |
  | 16 loops | −40% / −50% | −37% / −50% |
  | 56 loops | −19% / −33% | −19% / −33% |
  | 112 loops | −8% / −30% | −7% / −28% |
  | 224 loops | −5% / −23% | −4% / −23% |

  Player views, 4 / 16 / 56 loops: commanded units −28% / −17% / −11% (the first model better
  than persistence on them at every horizon; in the combined view still worse at 4 loops), small
  units near walls −56% / −44% / −20%, large −17% / −15% / −11%, enemy units −42% / −24% / −7%;
  orders better than persistence everywhere. The posterior's z beats a prior sample by ~1-3% at
  16-224 loops. The refit probes were still improving at step 1,000, so the true-latent numbers
  are likely understated at long horizons.
- Reading: the encoder holds far more than the dynamics predict (at 56 loops, −33% readable vs
  −19% predicted; online probes showed −49%), and the dynamics plateaued while the encoder kept
  improving. The direct model (step 2,000) is still ahead at 16-56 loops (−66%, −42% on 48
  states). Next lever: the dynamics model (h, g, how z enters), not the representation.

## Grid latent (`--model grid`, worldmodel/grid_latent.py, 2026-11-01)

Design: `DESIGN_grid_latent.md`. The latent is now a fixed 16 x 16 grid (plus CLS and players)
on every map, not one token per entity: the dense encoder's entity / effect tokens and its
2-map-unit cells are scattered onto the grid (playable area centered, scaled by its longest side)
and pooled per latent cell by attention with each token's offset from the cell center; 4
transformer blocks mix the 259 tokens. The h / g dynamics, EMA target, categorical z and energy
score run on them unchanged, with a latent loss on every token.

Entities come back through a decoder (0.1 of its gradient reaches the latent): Poisson counts per
(cell, owner, type), one query per counted entity (cell vector + class + slot in its group),
readouts of the offset in the cell, the entity numbers and the order, paired with the truth
within each (cell, owner, type) group by y offset (order-free, on the GPU). Second-moment losses
(latent cells, decoded entities vs the target's entity tokens, entities x cells) at 0.1.
Evaluation decodes ŝ_{t+k} (model), the target latent (what the latent holds) and compares with
persistence; it also reports the position error under a global Hungarian assignment, the
cell-border mismatch.

Occupancy that set the decoder (150 player views, every 16th frame): 261 entities per frame
(median; 623 max). At 16 x 16 (8.8 map units per cell) 68% of cells are empty, a non-empty cell
holds 2 / 7 / 21 / 98 entities (p50 / p90 / p99 / max); at 32 x 32, 1 / 4 / 10 / 50.

Also: `torch._dynamo.config.recompile_limit` raised to 64. The dense blocks see float32 and
bfloat16 inputs, with and without gradient, at several shapes, and past the default 8 variants
dynamo had been running the rest uncompiled in every earlier run. Compile errors (an inductor
tiling assertion on rare shapes) now fall back to eager instead of crashing.

Smoke run (100 steps, 16 frames x 4 horizons): ~0.5 s/step (the per-entity latent model ~1.15),
11.9 GB. Run: `grid16_20261101_1019`, 20,000 steps, evaluation every 1,000.

Result of `grid16_20261101_1019` (stopped at step 4,525): collapse in time. The latent's change
t → t+k fell 0.42 → 0.027 and the latent ratio rose 0.50 → 2.5 (worse than copying the present);
on held-out maps (step 2,000) decoding the true latent placed entities within ~2.5 map units
(moving units 2.1 vs persistence 0.7 at 4 loops) with a count error of ~170-180 per frame against
~260 entities; z unused (posterior = prior, commands given = withheld). The cell-border gap was
small (in-cell pairing 2.54 vs global Hungarian 2.45 map units).

Restart `grid16_inv_20261101_1108`: the posterior becomes the inverse of the action delta (from
E_online(s_{t+k}) − E(s_t) − sg(h) only, Barlow-aligned with the action encoder), h trains alone on
withheld samples, decoder gradient unchanged (0.1). See DESIGN_grid_latent.md, "The inverse".

`grid16_inv_20261101_1108` (stopped at step 2,250): the same collapse (latent ratio 1.8 on held-out
maps at step 1,000; true-latent decode ~2.8 map units, count error ~300 per frame), and the
inverse's Barlow term stuck at ~64, no correlation: only ~43 pairs per batch for a 64-dimension
Barlow at weight 0.05. Restart `grid16_vic_20261101_1214`: VICReg pairing (32 dims, weight 1),
action windows always built in full (withholding is a mask on g and the prior, so every sample is
a pair), the enemy's commands as targets in player views, a queue of 8 batches for the pairing's
statistics. Training windows are now full in every mode (has_act masks them everywhere they were
read); evaluation still builds empty windows for "withheld".

`grid16_vic_20261101_1214` (stopped at step ~1,000): the VICReg pairing didn't move either (flat at
~1.9) though the inverse gets ~140 pairs per batch; collapse as before (latent ratio 2.0-2.4 on
held-out maps). Rollout training built (`--seq_len 14`, steps of 4 / 8 / 12 / 16 loops, open-loop
loss, `--g_self`), smoke-tested, parked.

## Grid latent pretraining (`scripts/train_ae.py`, 2026-11-01)

Encoder and decoders alone: one frame -> grid latent -> the whole frame back (entities, map layers
at 1 map unit, players' numbers and upgrades), gradients fully on, no dynamics. The test of whether a
16 x 16 x 128 latent can hold the game state. Run `ae16_20261101_1254`: 64 frames per batch, 20,000
steps, ~0.9 s/step, 20 GB. Code committed before it as 2fec194 (branch `grid-latent`).

`ae16_20261101_1254` (stopped at step 3,425): training losses fell fast, but unseen maps stalled
(median position error 2.30, count recall 0.36 at step 3,000) while seen games were at 1.09 by step
1,000. Restart `ae16v2_20261101_1347`: no counts, (cell, slot) queries with type and owner decoded,
numbers by kind, the order queue (4 slots), all 11 map channels, uncertainty weighting, batches of
moments in all three views, and the decomposition check (player latents from the combined latent).
See DESIGN_grid_latent.md, "Pretraining".

v3 `ae16v3_20261101_1412` (v2 stopped at step 1,475): 8-slot order queue plus its length, 8-slot
production queue, the container pointer, a per-cell entity count in the loss, on_map / playable out
of the map loss.

v4 `ae16v4_20261101_1435` (v3 stopped at step ~1,300): class-balanced losses for the order and
production queues, the container pointer and the binary flags (v3 had production at 0.000, containers
at 0.005 and the rare flags at F1 0 on unseen maps at step 1,000, all predicting the majority class).

v5 `ae16v5_20261101_1504` (v4 stopped at step ~1,700): class-balanced cross entropy for display /
cloak / inside (their 0.98-0.99 accuracy was the majority class: cloaked / detected are ~1 row in
1,000), per-state recall in evaluation, type balanced accuracy.

## Add-on captures: a test set to extract (2026-11-01)

A Terran add-on can change player owner (the other Terran lands on an abandoned one). None of the
extracted games has one; the tracker events (`SUnitOwnerChangeEvent`) of all 946 TvT games in the
replay packs (18,162 replays) record exactly 4, every one an add-on, all 3.5-5 min in. To be
extracted later as a test set (kept out of training):

| pack (`data/replays/`) | game | owner changes |
|---|---|---|
| `2021_03-DH_SC2_Masters_Fall.zip` | Ro8, MarineLorD vs Clem G1, Lightshade LE | FactoryTechLab → p1 at 4.83 min |
| `2022_02-DH_SC2_Masters_Valencia_2022.zip` | Ro16, Clem vs SpeCial G1, Cosmic Sapphire | BarracksTechLab → p1 at 3.93, → p2 at 4.10 min |
| `2024_02-ESL_SC2_Masters_Spring_2024_Online.zip` | EU groups, Milkicow vs Clem G2, Crimson Court | BarracksReactor → p2 at 3.67, → p1 at 4.42 min |
| `HSC XIII Replay Pack.zip` | group A, uThermal vs Cure, Lost and Found LE | BarracksReactor → p1 at 4.05 min |

Archive member paths:
- `2 - Fall/4 - Europe/Main Event/Playoffs/4 - Ro8 #4 - MarineLorD vs Clem/20210810 - GAME 1 - MarineLorD vs Clem - T vs T - Lightshade LE.SC2Replay`
- `DH SC2 Masters - Valencia 2022 Festival/Playoffs/9 - Ro16 #1 - Clem vs SpeCial/20220702 - GAME 1 - Clem vs SpeCial - T vs T - Cosmic Sapphire.SC2Replay`
- `2024_02-ESL_SC2_Masters_Spring_2024_Online/2 - Europe/1 - Group Stage/A - R1M1 - Clem vs Milkicow/20240410 - Game 2 - Milkicow vs Clem - TvT - Crimson Court.SC2Replay`
- `HSC XIII Replay Pack/Day 1/Group A/uThermal vs. Cure/Lost and Found LE.SC2Replay`

Note (2026-11-01, deferred): during a Neural Parasite the raw API reports the **controller** as the
unit's owner (8 extracted views: an Archon / Carrier shows the Zerg player while it carries buff 22
NeuralParasite, its own player before and after). The real owner is not a unit field; only the buff
and history mark the transfer as temporary. The tracker's `SUnitOwnerChangeEvent` has both the control
and the upkeep player (Neural changes control only). Judged trivially learnable (the buff is an
input); if it ever matters, extract the upkeep owner and decode it with a neural flag / the buffs.

## Frame cache (`worldmodel/frame_cache.py`, `scripts/build_frame_cache.py`, 2026-11-01)

The pretraining run held 12 rotating games plus 6 evaluation games in RAM, each with its three views
(~146k view-frames in the pool, ~2.3 GB per game), and the process grew to ~47 GB and was OOM-killed
at step 6,800 of v5 (checkpoint at step 6,000 kept). Every batch also drew from only those 12 games,
and the training split has only 14 maps.

A state file can't be read frame by frame (delta-coded units, XOR'd maps) and part of a frame comes
from whole-game preprocessing, so the cache is built once: each game is loaded with its three views,
80 moments are chosen while it's decoded (one per equal slice of the game, jittered), and each
moment's frame is written in all three views with fixed player codes (the player's index, 0 or 1).
The reader swaps the two players at random per sample, every frame and view alike; collation then
splits the map into static layers and one row per player view (`DESIGN_grid_latent.md`, "Player
identity and map views"). ~30 KB per moment compressed (all three
views), ~2.5 MB per game; reading 21 moments takes ~10 ms and batching ~10 ms (building frames from
whole games took 0.3-0.8 s per batch). Layout: `<root>/vocab.json`, `<root>/<split>/<game id>.fcache`.

Built: 629 games, 50,320 moments (150,960 frames), 1.5 GB, 7 minutes with 6 workers. Splits as
directories: train 448 games (14 maps), held_out_games 23, held_out_map_games 158 (7 maps). v6
`ae16v6_20261101_1657` trains from it (0.85 s/step, 20 GB GPU, RAM ~a few GB), with a train /
validation loss every 100 steps on the held-out games and held-out maps.

## Rare-state evaluation set (2026-11-01)

Random evaluation frames rarely hold the rare entity states, so their F1 swung on a handful of units
(blip F1 0-0.17). Measured in the training cache: blips are 0.025% of entities, in 0.6% of player
view-frames (~11 blips each when present); in raw data 7 of 8 sampled Terran views with a sensor
tower have them. Training sees ~3 blip entities per batch, which the balanced flag loss weighs fully,
so this was an evaluation problem. `scripts/build_rare_eval.py` picks 270 fixed held-out frames
(held-out games and maps) containing each state: blip (310 units), cloaked (325), burrowed (1,710),
hallucination (58), engaged (1,400), has add-on (931), cloak cloaked (66) / detected (86), passenger
(472), in gas (1,493), display hidden (909), units with a container (2,896). `train_ae.py` reports each
state's recall / precision / F1 on them at every evaluation (`worldmodel/rare_eval.py`);
`--eval_ckpt <model_latest.pt>` evaluates a saved model once.

First reading, v6's step-6,000 checkpoint (`--eval_ckpt`): every rare state had recall 0.95-1.00 (blip
1.00, cloaked 1.00, hallucination 0.98, detected 0.95, passenger 1.00) but precision 0.08-0.37: the
fully balanced losses (mean over positives + mean over negatives) train as if each state were present
half the time, so the decoder over-predicts them. Container pointer 0.41. Change: class weights
count ** -0.5 (square root; `--balance_power`, 1 = fully balanced, 0 = plain mean) in every balanced
loss (queues, container pointer, flags, categoricals); at 1% positives they carry ~10% of the weight
instead of 50%. v6 resumed from step 6,000 with it (`--resume`), same run directory and logs.

### Window cache (2026-11-01)

v6 at step 7,000 began to overfit the 80-moment cache: validation losses split (train 1.87, held-out
games 2.43, held-out maps 3.55; level at step 4,000), and vision / creep IoU reached 0.78 / 0.67 on
the cached training moments but 0.22 / 0.11-0.21 on held-out games of the same maps (each of the 35,840
training moments had been seen ~4 times). Rebuilt as windows for dynamics too: 300 windows per game,
each 15 consecutive frames (a moment and the 14 after it: 56 game loops, 2.5 s at the 4-loop frame
step, the rollout mode's Δ = 4 x 14 steps; longer horizons to come from a hierarchical level), every
frame in all three views, compressed per window (5.8 KB per view-frame, ~78 MB per game, ~49 GB for
629 games, ~23 s per game to build). One code permutation per sampled window. `FrameCache.sample`
draws a random frame of a random window (encoder pretraining); `sample_windows` returns whole windows
(dynamics). Directory: `data/worldmodel/frame_cache_seq`.

The first full build crashed the machine (6 workers, every uncompressed frame of a game held until
the write). The builder now compresses each window as it's built. One worker peaks at 13.6 GB on the
largest games (state files vary 5-10x). A second attempt at 4 workers hit a 35 GB `systemd-run` cap in
2 minutes (the cap held; the machine was untouched): a GameSet's views reference each other, so a
finished game lived on until the cycle collector ran, under the next one. Each job now ends with
`gc.collect()` and `malloc_trim`: 10.7 GB worst case per worker; 3 workers can't exceed the cap.

Built: 629 games, 188,700 windows, 8.5M view-frames, 43 GB, 17 minutes for the last 409 games at 12
workers (38 GB, under a 56 GB cap; the earlier 3-worker estimate guarded against an all-largest case
that doesn't happen). v6 resumed from step 8,000 on it (`--cache data/worldmodel/frame_cache_seq`).

## Fixed map pyramid (`worldmodel/quad_latent.py`, 2026-11-01)

`scripts/train_ae.py` now encodes with `QuadEncoder`: one grid from 128 x 128 norm tokens down to the 16 x 16
latent, real map cells and entities attending in sliding 4 x 4 windows, full attention at the latent's level.
`GridEncoder` remains for the parked `train_direct.py --model grid`. Earlier pretraining checkpoints don't
load. See `DESIGN_grid_latent.md`, "Fixed pyramid".
