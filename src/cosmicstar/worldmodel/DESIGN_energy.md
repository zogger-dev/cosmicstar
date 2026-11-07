# Energy model: design discussion checkpoint (2026-11-06)

Status: discussion only, nothing built, nothing approved for building. This records where the
conversation stood when it was checkpointed. Items marked **Open** have no decision yet. The
untracked draft `thermodynamic_model.py` from earlier the same day predates this discussion and is
superseded by it (see "Why the first draft could not train").

## The picture

StarCraft II as a closed, dissipative, zero-sum energy system.

- **Energy is value in resource units.** A map's endowment is the total energy: minerals and gas in
  the ground. Nothing enters the system afterwards. Energy moves between stages and only leaves by
  dissipation.
- **Stages.** In the ground → bank → in production → units and structures → engaged → damage dealt.
  Workers move energy from the ground to the bank; spending moves it into production; completion
  moves it into units; fights convert it into damage; deaths and destruction dissipate it. A cancel
  is a transfer with a loss (150 in a barracks under construction, 112 back to the bank, 38
  dissipated).
- **Ground state.** A single surviving building. The whole game is a relaxation toward it. Both
  players' energy dissipates monotonically; the player whose energy dissipates last wins. The
  strategic quantity is the ratio of the two dissipation rates, and both rates are observed every
  frame.
- **Kinetic energy is imbalance of flow.** The coordinates are the ledger stages per player; the
  velocities are the flows between stages. An income advantage is a velocity difference at the first
  stage; spending while the opponent banks is one at the second. Kinetic energy is on the difference
  of the two flow vectors and is zero when the flows match stage for stage: the even game.
- **Potential is position.** Each entity is a source with a shape (range, splash, air or ground,
  vision over cliffs) and the terrain modulates how the shape propagates (cliffs block, chokes
  concentrate, high ground extends). The field at a point is the sum of sources as seen through the
  map; move a tank and its well moves with it. A sieged tank line is a well: a force needs more
  kinetic energy than the well is deep to break it. Lanchester's square law is the quantitative
  analog: fighting strength is quadratic in count, like kinetic energy in velocity.
- **Map control falls out.** Unmined minerals are stored energy nobody owns yet; an expansion is a
  claim on it. Macro position and tactical position live in the same field.
- **F = ma.** One field, zero-sum, sign = player 1's advantage. A player's force is the gradient of
  the field along the directions they control (their orders, their spending); the net force is the
  difference, zero in a balanced game. Mass is the value in play: the same push moves a 40-supply
  game more than a 200-supply game. Being cornered is positive feedback: a losing player's available
  force shrinks with their units and income.
- **Attention (APM)** is a second budget, distinct from income. It bounds how much force a player can
  apply per unit time and in how many directions at once. It constrains which trajectories are
  reachable; it does not shape the field. Efficiency means energy set in motion per unit of
  attention (one held key filling eight barracks), not fewer actions. Not thought through yet.
- **Leaks**, treated as noise for now: hp, shield and unit-energy regeneration are free energy
  entering the system; they are finite and dissipate again.

## The form

A port-Hamiltonian system: a Hamiltonian with dissipation and external inputs built in.

    ẋ = (J − R) ∇H(x) + B u

- x: the observed coordinates. Unit positions, hp, shields, energy, cooldowns; per player bank,
  supply, production, unit and structure value; per patch minerals and gas remaining.
- J: the conservative exchange between stages (the symplectic part).
- R ⪰ 0: dissipation (deaths, destruction, cancel losses). Non-negative by construction.
- H: the energy. The ledger part is defined, not learned, in resource units. The learned part is the
  positional potential: how a configuration of entities and terrain converts into flows.
- B u: the players' input, the force applied through their actions. u is observed in replays; at
  play time it is what the planner chooses.

**Why it cannot collapse.** Coordinates, actions and velocities are all observed. At every visited
state the model must account for the observed ẋ through (J − R)∇H and the observed u. A constant H
would attribute every change to u alone, which fails as soon as a unit keeps walking, a worker keeps
mining or a tank line keeps killing with no new command. The force residual, observed ẋ minus the
model's accounting, forces H to be non-constant. There is no learned encoder in that loop. This is why
Hamiltonian neural networks work: their coordinates are measured and their velocities are regressed.
The guarantee comes from the observed coordinates and velocities; the Hamiltonian form is the
structure on top.

**Division of labor.** The laws fix the form of the energy: a potential whose gradient is the force,
non-negative dissipation, accounting that closes, a ground state. They cannot pick the trajectory:
the trajectory where nobody does anything obeys every law. The replays fix the landscape: the chunks
that happened are low energy, and denoising supplies "everywhere else is higher" without constructed
negatives.

## Levels

Non-autoregressive at every level. A level generates a chunk of trajectory by denoising, with the
energy as guidance; it never steps frame by frame, so errors do not compound. Play-time performance is
not a concern yet; distill to faster models later if needed. Time scales, all **Open** as to exact
values: about 1.25 s, 30 s, and 2–5 minutes.

### Physics (1.25 s, 28 loops)

An energy over (start state, actions, chunk) trained by denoising on observer-view replays. The
engine's own pathing is gradient descent on a distance field and crowding is repulsion, so this is
the level where the potential picture is nearly literal. Current windows are 56 loops, so the data
pipeline already fits.

One energy, any conditioning direction: condition on actions to sample a chunk (forward, prediction);
condition on an intended chunk to sample actions (inverse, execution). Efficiency enters at
execution as a cost, physics energy plus λ·power(a), not in the representation. Spam does not change
the outcome, so the power term removes it.

### Intent (one latent per 1.25 s chunk)

The interface between levels. The upper level generates a trajectory of chunk latents; the physics
level decodes each into actions.

- z holds intent only: what the player wants to do, conditioned on their uncertainty. It carries no
  beliefs about the opponent. The uncertainty lives in the context (the player's view history), not
  in z.
- The encoder that infers z from a replay chunk sees the player's view and their actions, never the
  observer view, so z is intent under the information the player had.
- z is not pinned to the actions: that would be imitation, and the equivalence class of action
  sequences with the same effect is the point. It is pinned through the physics energy to the
  observed chunk, with the opponent's actions left out of the condition, so the energy marginalizes
  the opponent over what opponents did in situations that looked like this one.
- A bottleneck (dimension or codebook) so z holds the predictable part, the intent, and sheds micro
  noise the upper level cannot predict. **Open**: which bottleneck.
- Barlow Twins teacher-student as the latent term. It prevents dimensional collapse (constant or
  low-rank latents). It does not by itself prevent collapse in time, which is what the grid and
  latent runs did with Barlow and VICReg in place: they kept identity, owner and terrain and dropped
  what changes. The observed-velocity pin is what prevents that, and it stays at full weight.
- An action energy: train the decoder so its actions score lower than the replay's actions for the
  same intended chunk, physics energy plus power. The humans' own inefficiencies are the negatives;
  none are constructed. Guard: the physics energy was trained on human actions, so the decoder can
  exploit its errors off the manifold; the engine stays the judge of anything it finds.

### Belief

Under fog the enemy's half of x is not observed. A module produces it from the player's view history;
this is the one learned representation between observations and coordinates. It is safe exactly when
its gradient flows to observed velocities, and it can additionally be trained against the observer
view of the same game (both views exist in games_v3), with no labels. It must not get a loss of its
own in latent space with nothing observed downstream; that is the only configuration that collapsed
before.

### 30 s and above

Energies over sequences of z, conditioned on the belief, generated by denoising. The targets are the
ledger velocities at that scale, which are observed: at 30 s fights resolve and production completes
(stored energy releases); at minutes, economy and tech. The win is the limit of the longest scale,
not a target; it is too far away to learn from.

**Open**: how the 30 s level reads the 1.25 s level. One candidate is the entity Gram matrix as a
space-invariant tactical posture (parameter-free, permutation-invariant, keeps counts and
co-occurrence; loses location unless the tokens carry it; 512² numbers, so low-rank or per region).
A learned summary is the alternative and is safe as long as the 30 s targets are observed. The
choice is inductive bias, not safety. Decodability: the physics level must be able to move the real
configuration toward a target expressed in the summary; checkable on real frames 30 s apart.

### Value of information

The planner's energy is not the dissipation ratio read off directly. With two possibilities A and B
and an unknown opponent, the mean path is nonsense or merely safe; a scout that reveals which yields a
lower-energy path than the mean. The comparison is

    cost of scouting + E[ min over z of energy | observation ]   vs   min over z of energy | nothing

To compute it the planner needs a distribution over what is hidden and a model of what a scout would
observe: the belief. VOI is then a computation in the planner, exact given the belief and
transferable to situations no pro ever scouted, rather than a learned habit. The amortized version
also appears: the 30 s energy rates scouting intents low in uncertain contexts because in the data
those were followed by better dissipation ratios. AlphaStar's critic saw both sides in training, so
it never represented not knowing.

## Data

- Pros only cover the ridge of good play; nobody at that level attacks into a sieged tank line, so
  the well's depth is visible only through avoidance. Lower-level players walk into the well and the
  ledger shows what it cost. Diverse skill levels give the field's shape where pros never go, with
  observed consequences. Skill is at most a conditioning variable, never a label on the energy, or
  the model learns rank, not physics.
- Practical catch: the pro packs are 5.0.14; the Blizzard ladder corpus is 5.0.2 and 5.0.4, which are
  held (see the patch-scope note). **Open**: where lower-level 5.0.14 games come from.
- Every term of H's ledger part is in the engine's counters, mostly already in the Arrow store:
  per-patch contents, banks, `sc_collected_*`, `sc_spent_*`, `sc_lost_*`, `sc_killed_*`,
  `sc_total_damage_*`, food, counts, APM and effective APM, idle worker and production time. The
  physics window reads only the `pc_*` numbers today.

## Why the first draft could not train

`thermodynamic_model.py` (2026-11-06, untracked) defined H = learned masses × v² + learned spring
potential + a map field, and a single loss: MSE between (H_next − H_t)/dt and P_ext + P_int −
dissipation, all terms learned, nothing compared to an observation. Trained on random tensors (no
game data) with its two call-site faults patched, the loss fell from 3486 to 2.7 in 1,500 CPU steps
with the mean mass shrinking from 0.66 to 0.02: the homogeneous all-zero solution. P_int is an
unconstrained linear head that absorbs any residual, the temporal attention over T = 2 is not causal
so H_t can read frame t + 1, position is never read, and the map term cancels in the difference. The
missing ingredient is the one above: observed coordinates and velocities on one side of the
equation. The port-Hamiltonian form has them.

## Relation to the current physics model

`physics.py` is already "grounded plus latent": `loss = lat + aux_weight × (raw + births + map +
player + recipients)` at `physics.py:862`, with `lat` Barlow Twins against an EMA teacher and
`aux_weight` fixed at 1.0 (`DESIGN_physics.md` says it decays to 0.1; the code never changes it). The
raw heads on observed deltas are the pin; the Barlow term is the one collapse-capable piece and its
contribution is untested. Its backbone (entity encoders, map tower, spatial and temporal attention)
is the natural parameterization of the positional potential and of the physics-level denoiser.

## Not decided, not approved

- Nothing here is approved for implementation.
- Exact time scales; the intent bottleneck; the 30 s summary (Gram or learned); the source of
  lower-level games; how APM constrains the sampler; whether regeneration is ever modeled.
- Whether to ablate the Barlow term in the current physics model.
