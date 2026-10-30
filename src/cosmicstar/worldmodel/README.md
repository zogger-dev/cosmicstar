# World model data notes

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
- **Blizzard replay packs** (s2client-proto README): only 3.16.1 packs remain listed; Linux
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

**Two places to install to.** (A) The real install (`~/Games/battlenet/.../StarCraft II`): done
once for 93333 at 14:27 on 2026-10-30 (backup `SC2Data/backup-20261030-142746`), after which
the 93333 engine played a DreamHack Dallas replay; the user asked for the real install to stay
unmodified, so whether that write is kept or reverted (one command) is theirs to decide. (B) A
side install, `.../Program Files (x86)/StarCraft II old`, which needs no write to the real one:
`data.001-023` are hardlinks to the real archives (the engine only rewrites `shmem` and `data.000`
at start, with identical content), `data.000`, `data.024`, the `.idx` files, `shmem`, `indices`,
`config`, `Support`, `Support64`, `Interfaces` and `Versions/Base98310` are copies (2.1 GB extra
plus the builds), and the five builds were installed into it (30 s each). Scripts use it through
`SC2PATH="$HOME/Games/battlenet/drive_c/Program Files (x86)/StarCraft II old"`.

**End to end** (side install, `SC2PATH` set, 2026-10-30): `dump_rules.py` dumped the 93333
ruleset (12 s), `export_balance.py` exported it (10 s), `extract_states.py` processed 3 of the 4
DreamHack Dallas 2025 replays in `data/replays/older-builds-test` (38-50 s each; G4 Ultralove
fails the exact desync check at 558 s: a unit the replay logs as dead at 555 s is alive - open,
not seen in the other 6 games), and in `data/replays/older-builds-test2` one 94137 and two 95841
replays (16-63 s each; the other 94137 game is skipped as "ruleset not exported"). 97364 and 97563
are installed but `extract_states.EXCLUDED_BUILDS` still skips 5.0.16 replays.

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
