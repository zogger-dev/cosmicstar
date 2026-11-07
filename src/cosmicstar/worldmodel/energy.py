"""The energy model (DESIGN_energy.md): StarCraft II as a closed, dissipative, zero-sum
system in port-Hamiltonian form,

    ẋ = J ∇H(x) + B u

with no dissipation term: energy leaves the system only when a row dies, which is the row
vanishing and its value with it, not a rate the model predicts.

On state vectors handed in (energy_data.py reads them from the physics batches): the
semantic context c of each row arrives as a vector. Every tensor carries a leading
frame dimension M, so a chunk of frames runs in one call.

What moves, and what moves it:

    position, velocity, heading    orders and the motion tables only (second order: the
                                   input is a force, mass from the type's acceleration).
                                   The field exerts no spatial force: where units go in
                                   the field is the player's choice, pinned at the
                                   action level, not here
    hp, shield, contents           first order: a gradient flow in the field. Damage is
                                   the enemy's projection meeting a row's presence, so
                                   the field climbs with the row's hp and hp runs down.
                                   No inertia: hp moves at the rate the field sets.
                                   Zero hp is death
    energy, cooldown, progress     known rates from the tables; resets are the engine's
                                   events, which the model does not predict
    the ledger                     first order too: bank and supply flow in the field
                                   the player's own rows make (workers mining, depots
                                   finishing) plus the orders' spending. The observed
                                   rates are read, not integrated

Coordinates are raw (map units, loops, resource units) and the networks read them
through fixed scales (Q_SCALE, K_SCALE, P_SCALE), so no axis dwarfs another at a linear
layer. The gradient flows run with unit mobility in those scaled coordinates: a change
of a coordinate's unit changes nothing but a constant.

The pieces, in the order a step runs them:

    State          frames: rows (q, k, c, owner, mask), players (p, cp), inputs u
    Interaction    the cross-player pair term e(probe i, source j): source j's
                   projection meeting probe i's presence, one scalar a pair, added to
                   i's owner and subtracted from j's. So V_1 + V_2 = 0 pair by pair:
                   adversarial by construction, and every term is grounded, since each
                   is an expected damage flow. Same-owner pairs contribute nothing, so
                   nothing heals by sign. ψ, pairs with the ground, rides on the same
                   hidden layer: patches pull workers, workers drain patches
    PlayerPotential a player's ledger in the field of its own rows
    Hamiltonian    H = T + V per owner: T the observed flows read out, V the terms above
    Dynamics       one step of J ∇H + B u, as laid out above
    ForceResidual  the only objective: the step's coordinates against the observed next
                   ones, per axis, rows and players
    EnergyModel    the three together: forward(state, next coordinates) -> loss, logs
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------- the state's layout

# q: a row's observed coordinates. Map units and loops; heading in radians; hp, shield
# and energy as fractions of their maxima; contents the resources left in a patch
COORDS = (
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "heading",
    "turn",
    "hp",
    "shield",
    "energy",
    "cooldown",
    "progress",
    "contents",
)
Q = {n: i for i, n in enumerate(COORDS)}
POS, VEL = slice(Q["x"], Q["z"] + 1), slice(Q["vx"], Q["vz"] + 1)
HP = slice(Q["hp"], Q["shield"] + 1)
# what a term may read of a row's own state: everything but where it is, so the only
# place a position enters is the geometry between rows (translation invariance)
PHYS = [i for n, i in Q.items() if n not in ("x", "y", "z")]
# what a row shows as a source: its state without hp and shield. A tank at 10 hp fires
# the same shell, and a projection that read its own hp would move that hp for firing
SRC = [i for n, i in Q.items() if n not in ("x", "y", "z", "hp", "shield")]
# the coordinates the field moves (gradient flows)
FLOW = ("hp", "shield", "contents")
# the coordinates an order pushes on: motion and heading (as forces), energy (a cast),
# progress (a cancel). Nothing else is a player's to move directly
ACTUATED = ("vx", "vy", "vz", "turn", "energy", "progress")
# the scale of each coordinate as a network reads it: a fast unit's move a loop, a
# radian's π, a long cooldown, a patch's contents
Q_SCALE = (1, 1, 1, 0.29, 0.29, 0.29, math.pi, math.pi, 1, 1, 1, 100, 1, 1000)

# p: a player's observed coordinates, the ledger: bank and supply, and their observed
# one-loop rates. Resource units, food, per loop. The rates are read (a trip landing, a
# purchase) but the flows are the field's and the orders'
PLAYER = (
    "minerals",
    "vespene",
    "food_used",
    "food_cap",
    "food_free",
    "d_minerals",
    "d_vespene",
    "d_food_used",
    "d_food_cap",
    "d_food_free",
)
PL = {n: i for i, n in enumerate(PLAYER)}
LEDGER, LRATES = slice(0, 5), slice(5, 10)
P_SCALE = (1000, 1000, 100, 100, 100, 5, 5, 1, 1, 1)

# k: a row's constants from the patch's tables: value its cost in resource units (a
# patch: its contents' worth), mass the inverse of its acceleration (0: instant), speed
# its cap, the maxima, build_time in loops (0: not built). Mass spans 26 to 4096, so
# it is read as log1p; the rest by a scale
CONSTS = ("value", "mass", "speed", "hp_max", "shield_max", "energy_max", "build_time")
K = {n: i for i, n in enumerate(CONSTS)}
K_SCALE = (500, 8, 0.29, 1000, 500, 200, 1000)

GROUND, P1, P2 = 0, 1, 2  # owners: the ground (neutral), the players
OWNERS = 3
GEOM = 4  # Δx, Δy, Δz, distance between a probe and a source
SCALE = 8.0  # map units the geometry is divided by
ROWS_SCALE = 64.0  # the summed row contexts of a player are divided by this


@dataclass
class State:
    """M frames of N rows; d the context width; P = 2 players; u_dim the input width."""

    q: torch.Tensor  # [M, N, len(COORDS)] float: the rows' observed coordinates
    k: torch.Tensor  # [M, N, len(CONSTS)] float: the types' constants
    c: torch.Tensor  # [M, N, d] float: the rows' semantic context
    owner: torch.Tensor  # [M, N] long: GROUND, P1 or P2
    mask: torch.Tensor  # [M, N] bool: a real row
    p: torch.Tensor  # [M, P, len(PLAYER)] float: the players' observed coordinates
    cp: torch.Tensor  # [M, P, d] float: the players' context (race, upgrades, ...)
    u: torch.Tensor  # [M, N, u_dim] float: this frame's inputs to each row (commands)
    loop: float = 1.0  # loops a step

    def replace(self, q=None, p=None):
        return State(
            self.q if q is None else q,
            self.k,
            self.c,
            self.owner,
            self.mask,
            self.p if p is None else p,
            self.cp,
            self.u,
            self.loop,
        )


def by_owner(values, owner, mask):
    """[M, N] per-row values -> [M, OWNERS] sums over real rows of each owner."""
    M = values.shape[0]
    out = torch.zeros(M * OWNERS, dtype=values.dtype, device=values.device)
    at = (torch.arange(M, device=owner.device)[:, None] * OWNERS + owner).reshape(-1)
    return out.index_add(0, at, (values * mask.to(values.dtype)).reshape(-1)).view(
        M, OWNERS
    )


def mlp(d_in, d_out, hidden, bias=True):
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out, bias=bias)
    )


def zero(linear):
    """A final layer that starts at zero: the potential and the inputs begin flat, so
    the first steps score the copy baseline and every term grows from the residual,
    not from the init's noise summed over N² pairs. A potential's final layer has no
    bias: a constant there, times the number of pairs, is a free drift of V that
    nothing in the residual sees."""
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


def scaled(t, scale):
    """t divided by a per-column scale (a tuple)."""
    return t / t.new_tensor(scale)


def constants(k):
    """[..., len(CONSTS)] the constants as a network reads them."""
    k = torch.cat([k[..., :1], torch.log1p(k[..., 1:2]), k[..., 2:]], -1)
    return scaled(k, K_SCALE)


def probe(s):
    """[M, N, d + len(PHYS) + len(CONSTS)] what a row is as a probe: its context, its
    whole state but position (hp and shields: what there is to lose), its constants."""
    return torch.cat([s.c, scaled(s.q, Q_SCALE)[..., PHYS], constants(s.k)], -1)


def source(s):
    """[M, N, d + len(SRC) + len(CONSTS)] what a row is as a source: its context, its
    state without hp and shields (cooldown, energy, progress... a disruptor that has
    fired is a different source from one that has not) and its constants."""
    return torch.cat([s.c, scaled(s.q, Q_SCALE)[..., SRC], constants(s.k)], -1)


def geometry(x, s):
    """x [M, X, 3] points -> [M, X, N, GEOM] Δ and distance to every row, scaled."""
    delta = s.q[:, None, :, POS] - x[:, :, None]
    return torch.cat([delta, delta.norm(dim=-1, keepdim=True)], -1) / SCALE


class PairMLP(nn.Module):
    """g(a_i, b_j, geom_ij) over all pairs -> [M, X, N], without materializing the pair
    inputs: the first layer is linear, so W [a; b; geom] = W_a a + W_b b + W_g geom,
    and only the hidden layer is pair-sized."""

    def __init__(self, d_a, d_b, hidden, outputs=1):
        super().__init__()
        self.a = nn.Linear(d_a, hidden, bias=False)
        self.b = nn.Linear(d_b, hidden, bias=False)
        self.g = nn.Linear(GEOM, hidden)
        self.out = zero(nn.Linear(hidden, outputs, bias=False))

    def forward(self, a, b, geom, extra=None):
        """a [M, X, d_a], b [M, N, d_b], geom [M, X, N, GEOM], extra [M, X, N, hidden]
        added to the hidden layer (a relation embedding) -> [M, X, N, outputs]."""
        h = self.a(a)[:, :, None] + self.b(b)[:, None] + self.g(geom)
        if extra is not None:
            h = h + extra
        return self.out(F.gelu(h))


# ---------------------------------------------------------------- the fields

ENEMY, NEUTRAL_REL = 0, 1


class Interaction(nn.Module):
    """The pair terms, two outputs of one pair-sized hidden layer (the pair terms are
    the memory, so they share it), with the relation enemy / neutral embedded into it.

    e   the cross-player term: e(probe i, source j, geometry) for every pair of rows of
        different players, source j's projection (its type, cooldown, energy, the
        range and the angle) meeting probe i's presence (its type, hp, shields). It
        enters i's owner's V with + and j's owner's with −, so the two players' sums
        cancel pair by pair. Its gradient on i's hp is the damage; on i's position it
        is repulsion from incoming fire, on j's position attraction toward what j can
        hit, both unused here (no spatial force) and the action level's to read.
        Same-owner pairs contribute nothing: no friendly fire, no healing by sign
    ψ   the neutral term at each row from pairs with the ground, one function either
        way: a patch pulls the workers that can mine it, the workers mining it drain
        its contents (the gradient on contents). Exactly one side the ground
    """

    def __init__(self, d, hidden):
        super().__init__()
        self.g = PairMLP(
            d + len(PHYS) + len(CONSTS), d + len(SRC) + len(CONSTS), hidden, outputs=2
        )
        self.rel = nn.Embedding(2, hidden)

    def forward(self, s):
        """-> (absorbed [M, N]: Σ_j e_ij at each probe, projected [M, N]: Σ_i e_ij at
        each source, ψ [M, N])."""
        geom = geometry(s.q[..., POS], s)
        ground = s.owner == GROUND
        gi, gj = ground[:, :, None], ground[:, None, :]
        neutral = gi ^ gj
        cross = (s.owner[:, :, None] != s.owner[:, None, :]) & ~gi & ~gj
        both = s.mask[:, :, None] & s.mask[:, None, :]
        rel = torch.where(neutral, NEUTRAL_REL, ENEMY)
        out = self.g(probe(s), source(s), geom, self.rel(rel))
        e = out[..., 0] * (both & cross)
        psi = (out[..., 1] * (both & neutral)).sum(-1)
        return e.sum(-1), e.sum(-2), psi


class PlayerPotential(nn.Module):
    """A player's ledger in the field of its own rows: h(cp, p, the sum of the player's
    row contexts). Its gradient on the bank and supply is what the rows do to income
    and supply: workers mining, a depot finishing, an army being built. The sum over
    rows is the plainest posture summary; a Gram or an attention pooling later."""

    def __init__(self, d, hidden):
        super().__init__()
        self.h = mlp(2 * d + len(PLAYER), 1, hidden, bias=False)
        zero(self.h[2])

    def forward(self, s):
        """-> [M, P] the potential of each player's ledger."""
        M, _, d = s.c.shape
        P = s.p.shape[1]
        at = (
            torch.arange(M, device=s.owner.device)[:, None] * OWNERS + s.owner
        ).reshape(-1)
        rows = torch.zeros(M * OWNERS, d, device=s.c.device, dtype=s.c.dtype)
        rows = rows.index_add(
            0, at, (s.c * s.mask[..., None].to(s.c.dtype)).reshape(-1, d)
        ).view(M, OWNERS, d)
        x = torch.cat(
            [s.cp, scaled(s.p, P_SCALE), rows[:, P1 : P1 + P] / ROWS_SCALE], -1
        )
        return self.h(x).squeeze(-1)


# ---------------------------------------------------------------- the Hamiltonian


class Hamiltonian(nn.Module):
    """H = T + V per owner.

    T   kinetic, read out of the observed velocities: ½ mass |v|² per row (the mass
        from the tables, 0 for instant types), ½ |rates|² per player in the scaled
        units: the flow through its ledger. Every velocity is observed, so T is never
        differentiated
    V   learned: a self term f(c, q without position, k) per row; the interaction, +
        at the probe's owner and − at the source's; the neutral term; the player term
    """

    def __init__(self, d, hidden=256):
        super().__init__()
        self.f = mlp(d + len(PHYS) + len(CONSTS), 1, hidden, bias=False)
        zero(self.f[2])
        self.interaction = Interaction(d, hidden)
        self.player = PlayerPotential(d, hidden)

    def kinetic(self, s):
        """[M, OWNERS]."""
        t = by_owner(
            0.5 * s.k[..., K["mass"]] * s.q[..., VEL].square().sum(-1), s.owner, s.mask
        )
        rates = 0.5 * scaled(s.p, P_SCALE)[..., LRATES].square().sum(-1)
        return torch.cat([t[:, :P1], t[:, P1:] + rates], 1)

    def potential(self, s):
        """-> (V [M, OWNERS], its parts by owner). `absorbed` is what each player's rows
        take, `interaction` the zero-sum difference (absorbed minus projected)."""
        own = self.f(probe(s)).squeeze(-1)
        absorbed, projected, psi = self.interaction(s)
        absorbed = by_owner(absorbed, s.owner, s.mask)
        parts = {
            "own": by_owner(own, s.owner, s.mask),
            "interaction": absorbed - by_owner(projected, s.owner, s.mask),
            "neutral": by_owner(psi, s.owner, s.mask),
        }
        v = sum(parts.values())
        v = torch.cat([v[:, :P1], v[:, P1:] + self.player(s)], 1)
        parts["absorbed"] = absorbed
        return v, parts

    def forward(self, s):
        """-> dict: H [M, OWNERS], T, V, and V's parts; total their sum."""
        T = self.kinetic(s)
        V, parts = self.potential(s)
        H = T + V
        return dict(H=H, total=H.sum(), T=T, V=V, **parts)

    def zero_sum_error(self, s):
        """max |interaction_1 + interaction_2| over the frames: zero up to float error,
        by construction. A check."""
        v = self.potential(s)[1]["interaction"]
        return (v[:, P1] + v[:, P2]).abs().max()


# ---------------------------------------------------------------- the dynamics


class Input(nn.Module):
    """B u: what this frame's commands do: a force on the row's actuated coordinates
    (motion, heading, a cast, a cancel), and spending on its player's ledger. A
    player's input reaches only that player's rows (B is block diagonal by owner, which
    the per-row u already is)."""

    def __init__(self, u_dim, hidden=256):
        super().__init__()
        self.rows = mlp(u_dim, len(COORDS), hidden)
        self.player = mlp(u_dim, len(PLAYER), hidden)
        zero(self.rows[2]), zero(self.player[2])
        act = torch.zeros(len(COORDS))
        act[[Q[n] for n in ACTUATED]] = 1.0
        self.register_buffer("act", act)
        led = torch.zeros(len(PLAYER))
        led[LEDGER] = 1.0
        self.register_buffer("led", led)

    def forward(self, s):
        """-> (on rows [M, N, len(COORDS)], on players [M, P, len(PLAYER)])."""
        M = s.mask.shape[0]
        m = s.mask[..., None].to(s.u.dtype)
        on_rows = self.rows(s.u) * m * self.act
        at = (
            torch.arange(M, device=s.owner.device)[:, None] * OWNERS + s.owner
        ).reshape(-1)
        on_players = torch.zeros(
            M * OWNERS, len(PLAYER), device=s.u.device, dtype=on_rows.dtype
        ).index_add(0, at, (self.player(s.u) * m * self.led).reshape(-1, len(PLAYER)))
        return on_rows, on_players.view(M, OWNERS, -1)[:, P1 : P1 + s.p.shape[1]]


class Dynamics(nn.Module):
    """One step of ẋ = J ∇H + B u, as the module docstring lays out.

    The force is −∇V by autograd on the rows' q and the players' p. It moves hp, shield
    and contents, and the bank and supply, as gradient flows with unit mobility in the
    scaled coordinates (Δq = −scale² ∂V/∂q: the flow a network sees in its own units);
    nothing else. Position, velocity and heading answer to the inputs and the motion
    tables only. Energy regenerates, the cooldown counts down, progress runs at the
    type's build time; their resets are the engine's. A row whose hp reaches zero is
    dead; nothing is subtracted for it, its value leaves with the row.
    """

    def __init__(self, d, u_dim, hidden=256):
        super().__init__()
        self.B = Input(u_dim, hidden)
        self.energy_regen = 0.7875 / 22.4 / 200  # per loop, as a fraction of 200
        mob = torch.zeros(len(COORDS))
        for n in FLOW:
            mob[Q[n]] = Q_SCALE[Q[n]] ** 2
        self.register_buffer("mobility", mob)
        mob_p = torch.zeros(len(PLAYER))
        mob_p[LEDGER] = torch.tensor(P_SCALE[LEDGER], dtype=torch.float) ** 2
        self.register_buffer("mobility_p", mob_p)

    def forward(self, s, H):
        """s the frames, H the Hamiltonian -> (q_next [M, N, len(COORDS)], p_next
        [M, P, len(PLAYER)], pieces)."""
        q = s.q.detach().requires_grad_(True)
        p = s.p.detach().requires_grad_(True)
        out = H(s.replace(q=q, p=p))
        # training differentiates through the force (double backward); evaluation
        # needs the force but no graph behind it, and runs under enable_grad
        grad_q, grad_p = torch.autograd.grad(
            out["V"].sum(), (q, p), create_graph=self.training
        )
        flow, flow_p = -grad_q * self.mobility, -grad_p * self.mobility_p
        k, dt = s.k, s.loop
        mass, speed = k[..., K["mass"]], k[..., K["speed"]]
        bu, bu_p = self.B(s)
        # motion and heading: the input is the force; instant types take it as their
        # velocity, the rest accelerate by force / mass; the cap is the type's speed
        instant = (mass <= 0)[..., None]
        m = mass.clamp(min=1e-6)[..., None]
        v_next = torch.where(instant, bu[..., VEL], q[..., VEL] + bu[..., VEL] / m * dt)
        v_next = v_next * (
            speed[..., None] / v_next.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ).clamp(max=1.0)
        pos_next = q[..., POS] + v_next * dt
        turn_next = torch.where(
            instant[..., 0],
            bu[..., Q["turn"]],
            q[..., Q["turn"]] + bu[..., Q["turn"]] / m[..., 0] * dt,
        )
        heading = q[..., Q["heading"]] + turn_next * dt
        # gradient flows: hp and shield down the interaction, contents down ψ
        hp = (q[..., HP] + flow[..., HP] * dt).clamp(0, 1)
        alive = hp[..., 0] > 0
        contents = (q[..., Q["contents"]] + flow[..., Q["contents"]] * dt).clamp(min=0)
        # clocks, at known rates
        energy = (
            q[..., Q["energy"]] + self.energy_regen * dt + bu[..., Q["energy"]] * dt
        ).clamp(0, 1)
        cooldown = (q[..., Q["cooldown"]] - dt).clamp(min=0)
        build = k[..., K["build_time"]]
        progress = (
            q[..., Q["progress"]]
            + torch.where(build > 0, dt / build.clamp(min=1), 0.0)
            + bu[..., Q["progress"]] * dt
        ).clamp(0, 1)
        q_next = torch.cat(
            [
                pos_next,
                v_next,
                heading[..., None],
                turn_next[..., None],
                hp,
                energy[..., None],
                cooldown[..., None],
                progress[..., None],
                contents[..., None],
            ],
            -1,
        )
        # the ledger: bank and supply flow in the player term's field and move by the
        # spending; the observed rates are carried, not predicted
        ledger = p[..., LEDGER] + flow_p[..., LEDGER] * dt + bu_p[..., LEDGER] * dt
        p_next = torch.cat([ledger, p[..., LRATES]], -1)
        return q_next, p_next, dict(flow=flow, flow_p=flow_p, alive=alive, **out)


# ---------------------------------------------------------------- the objective


class ForceResidual(nn.Module):
    """The one training signal: the step's coordinates against the observed next ones,
    rows and players. Observed coordinates, observed inputs, observed changes: a
    constant H attributes every change to the inputs alone and fails where hp keeps
    falling and workers keep mining, so H is forced to be non-constant
    (DESIGN_energy.md, why it cannot collapse). Scales put the axes on one footing.
    Scored but not trained on: the cooldown, whose residual is the fire events, which
    this model has no head for; the ledger's rates, which are read, not predicted."""

    def __init__(self):
        super().__init__()
        w = torch.ones(len(COORDS))
        w[POS] = w[VEL] = 1 / 0.29  # a fast unit's move a loop, map units
        w[Q["hp"]] = w[Q["shield"]] = w[Q["progress"]] = 10.0  # fractions a loop
        w[Q["contents"]] = 1 / 5  # a worker's trip
        self.register_buffer("w", w)
        use = torch.ones(len(COORDS))
        use[Q["cooldown"]] = 0.0
        self.register_buffer("use", use)
        wp = torch.ones(len(PLAYER))
        wp[PL["minerals"]] = wp[PL["vespene"]] = 1 / 50
        wp[PL["food_used"]] = wp[PL["food_cap"]] = wp[PL["food_free"]] = 1 / 8
        self.register_buffer("wp", wp)
        use_p = torch.ones(len(PLAYER))
        use_p[LRATES] = 0.0
        self.register_buffer("use_p", use_p)

    def forward(self, q_pred, q_next, mask, p_pred, p_next):
        """mask [M, N] the rows scored -> (loss, per-axis residuals [len(COORDS)],
        [len(PLAYER)])."""
        r = F.smooth_l1_loss(q_pred * self.w, q_next * self.w, reduction="none")
        rows = (r * mask[..., None]).sum((0, 1)) / mask.sum().clamp(min=1)
        players = F.smooth_l1_loss(
            p_pred * self.wp, p_next * self.wp, reduction="none"
        ).mean((0, 1))
        return (rows * self.use).sum() + (players * self.use_p).sum(), rows, players


# ---------------------------------------------------------------- together


class EnergyModel(nn.Module):
    """Hamiltonian, dynamics and residual. forward(state, q_next, p_next, scored) ->
    (loss, logs); step(state) -> the next coordinates. The two players are two
    instances of the same Hamiltonian read out per owner, coupled through the
    interaction; at play time the opponent's rows and ledger come from the belief
    instead of the frame, and nothing else changes."""

    def __init__(self, d, u_dim, hidden=256):
        super().__init__()
        self.H = Hamiltonian(d, hidden)
        self.dynamics = Dynamics(d, u_dim, hidden)
        self.residual = ForceResidual()

    def step(self, s):
        return self.dynamics(s, self.H)

    def forward(self, s, q_next, p_next, scored=None):
        """scored [M, N] bool: the rows with an observed next frame (default: the
        state's mask)."""
        scored = s.mask if scored is None else scored
        q_pred, p_pred, pieces = self.step(s)
        loss, rows, players = self.residual(q_pred, q_next, scored, p_pred, p_next)
        logs = {f"res_{n}": rows[i].detach() for n, i in Q.items()}
        logs.update({f"res_{n}": players[i].detach() for n, i in PL.items()})
        # each trained axis's share of the loss
        total = loss.detach().clamp(min=1e-12)
        use, use_p = self.residual.use, self.residual.use_p
        logs.update(
            {f"share_{n}": rows[i].detach() / total for n, i in Q.items() if use[i] > 0}
        )
        logs.update(
            {
                f"share_{n}": players[i].detach() / total
                for n, i in PL.items()
                if use_p[i] > 0
            }
        )
        # the copy baseline: nothing moves. What a constant H would score
        _, crows, cplayers = self.residual(s.q, q_next, scored, s.p, p_next)
        logs.update({f"copy_{n}": crows[i].detach() for n, i in Q.items()})
        logs.update({f"copy_{n}": cplayers[i].detach() for n, i in PL.items()})
        for name in ("H", "T", "V", "own", "interaction", "absorbed", "neutral"):
            logs.update(
                {
                    f"{name}_{o}": pieces[name][:, o].mean().detach()
                    for o in range(OWNERS)
                }
            )
        logs["advantage"] = (pieces["H"][:, P1] - pieces["H"][:, P2]).mean().detach()
        # the accounting, from what happened: value lost to damage this step, and the
        # value of the rows that died (the dissipation)
        value = s.k[..., K["value"]]
        lost = -(q_next[..., HP] - s.q[..., HP]).sum(-1) * value
        died = ((s.q[..., Q["hp"]] > 0) & (q_next[..., Q["hp"]] <= 0)).to(value.dtype)
        lost, died = (
            by_owner(lost, s.owner, scored),
            by_owner(died * value, s.owner, scored),
        )
        for o in range(OWNERS):
            logs[f"lost_{o}"] = lost[:, o].mean().detach()
            logs[f"dissipated_{o}"] = died[:, o].mean().detach()
        return loss, logs
