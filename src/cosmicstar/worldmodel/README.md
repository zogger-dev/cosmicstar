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
