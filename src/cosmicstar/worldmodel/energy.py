"""The energy model (DESIGN_energy.md): StarCraft II as a closed, dissipative, zero-sum
system in port-Hamiltonian form,

    ẋ = J ∇H(x) + B u

with no dissipation term: energy leaves the system only when a row dies, which is the row
vanishing and its value with it, not a rate the model predicts.

On state vectors handed in (energy_data.py reads them from the physics batches): the
semantic context c of each row arrives as a vector. Every tensor carries a leading
frame dimension M, so a chunk of frames runs in one call.

A row's state comes in three columns:

    integrated     q: position and velocity in the plane, hp, shield, energy, and a
                   patch's contents. What the dynamics steps and the residual scores
    given          g: height, heading (as cosine and sine), weapon cooldown, build
                   progress. Read by the networks, never stepped or scored: the engine
                   derives them (z from the terrain and whether the unit flies, the
                   heading from the order and the target) or runs them at known rates
                   and resets them on its own events
    constant       k: the type's value, mass, speed, maxima, build time, radius, whether
                   it flies or walks cliffs

What moves, and what moves it:

    position, velocity             orders and the motion tables only (second order: the
                                   input is a force, mass from the type's acceleration).
                                   The field exerts no spatial force: where units go in
                                   the field is the player's choice, pinned at the
                                   action level, not here
    hp, shield, contents           first order: a gradient flow in the field. Damage is
                                   the enemy's projection meeting a row's presence, so
                                   the field climbs with the row's hp and hp runs down.
                                   No inertia: hp moves at the rate the field sets.
                                   Zero hp is death
    energy                         regenerates at the known rate where the type has a
                                   pool; a cast is an input
    the ledger                     first order too: bank and supply flow in the field
                                   the player's own rows make (workers mining, depots
                                   finishing) plus the orders' spending. The observed
                                   rates are read, not integrated

Energy is value in resource units, and so is V. Every axis the field moves is measured
in resource units: a row's hp bar is worth its value's hp share (value × hp_max / (hp_max
+ shield_max)), its shield bar the rest, a patch's contents are resource units already,
so is the bank, and a point of supply is worth what a depot charges for it. The residual
scores all of them with one weight (a worker's trip), and the gradient flows run with
unit mobility in those units: ΔE = −∂V/∂E. A zergling's bar and a battlecruiser's bar are
then different amounts of energy, not the same unit, and a 28-mineral loss is the same
loss whichever row it happens to. Position and velocity have their own scale (a fast
unit's move a loop) and are not energy.

Coordinates are stored raw and the networks read them through fixed scales (Q_SCALE,
G_SCALE, K_SCALE, P_SCALE), so no axis dwarfs another at a linear layer.

Each row flows down its owner's potential, not the total: the interaction cancels in
V_1 + V_2 exactly, so its gradient is read from the probe's side of the plain sum of the
pair term over all cross-player pairs, which is the same for both owners. The other
terms (own, ψ, player) sit in the probe's owner's V. So one gradient, on the rows as
probes, with the rows as sources held fixed. Each term reads only the coordinates it
may move: the interaction and the own term never see a patch's contents, ψ never sees hp
or shields, so the interaction is the only damage, ψ the only mining, and the own term
holds what a row does to itself (regeneration).

The pieces, in the order a step runs them:

    State          frames: rows (q, g, k, c, owner, mask), players (p, cp), inputs u
    Interaction    the cross-player pair term e(probe i, source j): source j's
                   projection meeting probe i's presence, one scalar a pair, added to
                   i's owner and subtracted from j's. So V_1 + V_2 = 0 pair by pair:
                   adversarial by construction, and every term is grounded, since each
                   is an expected damage flow. Same-owner pairs contribute nothing, so
                   nothing heals by sign
    Neutral        ψ, pairs with the ground: patches pull workers, workers drain patches
    PlayerPotential a player's ledger in the field of its own rows
    Hamiltonian    H = T + V per owner: T the observed flows read out, V the terms above
    Dynamics       one step of J ∇H + B u, as laid out above
    ForceResidual  the only objective: the step's coordinates against the observed next
                   ones, per axis, rows and players
    EnergyModel    the three together: forward(state, next coordinates) -> loss, logs
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------- the state's layout

# q: a row's integrated coordinates. Map units and loops; hp, shield and energy as
# fractions (of the maxima; energy of 200); contents the resources left in a patch
COORDS = ("x", "y", "vx", "vy", "hp", "shield", "energy", "contents")
Q = {n: i for i, n in enumerate(COORDS)}
POS, VEL = slice(Q["x"], Q["y"] + 1), slice(Q["vx"], Q["vy"] + 1)
HP = slice(Q["hp"], Q["shield"] + 1)
# what a term may read of a row's own state, by term: everything but where it is, so
# the only place a position enters is the geometry between rows (translation
# invariance), and only the flows it may move
PHYS = [i for n, i in Q.items() if n not in ("x", "y")]
PROBE_E = [i for n, i in Q.items() if n not in ("x", "y", "contents")]
PROBE_PSI = [i for n, i in Q.items() if n not in ("x", "y", "hp", "shield")]
# what a row shows as a source: its state without hp and shield. A tank at 10 hp fires
# the same shell, and a projection that read its own hp would move that hp for firing
SRC = [i for n, i in Q.items() if n not in ("x", "y", "hp", "shield")]
# the coordinates the field moves (gradient flows), in resource units
FLOW = ("hp", "shield", "contents")
# the coordinates an order pushes on: motion (as a force) and energy (a cast). Nothing
# else is a player's to move directly
ACTUATED = ("vx", "vy", "energy")
# the kinematic axes: not scored on the frame a row dies in, where the engine freezes
# them, while the killing blow on hp and shield is scored
KINEMATIC = ("x", "y", "vx", "vy", "energy")
# the scale of each coordinate as a network reads it: a fast unit's move a loop, a
# patch's contents
Q_SCALE = (1, 1, 0.29, 0.29, 1, 1, 1, 1000)

# g: a row's given coordinates, read and never stepped: height, heading as cosine and
# sine, weapon cooldown (loops), build progress (0..1)
GIVEN = ("z", "cos", "sin", "cooldown", "progress")
G = {n: i for i, n in enumerate(GIVEN)}
G_SCALE = (16, 1, 1, 100, 1)

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
FOOD_VALUE = 100.0 / 8.0  # resource units a point of supply: a depot
# a ledger coordinate's worth in resource units
P_VALUE = (
    1,
    1,
    FOOD_VALUE,
    FOOD_VALUE,
    FOOD_VALUE,
    1,
    1,
    FOOD_VALUE,
    FOOD_VALUE,
    FOOD_VALUE,
)

# k: a row's constants from the patch's tables: value its cost in resource units (a
# patch: its contents' worth), mass the inverse of its acceleration (0: instant), speed
# its cap, the maxima (energy_max 0: no pool), build_time in loops (0: not built), its
# collision radius, whether it flies, whether it walks cliffs. Mass spans 26 to 4096,
# so it is read as log1p; the rest by a scale
CONSTS = (
    "value",
    "mass",
    "speed",
    "hp_max",
    "shield_max",
    "energy_max",
    "build_time",
    "radius",
    "flying",
    "cliff",
)
K = {n: i for i, n in enumerate(CONSTS)}
K_SCALE = (500, 8, 0.29, 1000, 500, 200, 1000, 4, 1, 1)

# the residual's metric. Energy axes (hp, shield, contents, the ledger) in resource
# units, one weight: a worker's trip. Position and velocity: a fast unit's move a loop.
# Unit energy: a fraction of its pool
W_ENERGY = 1 / 5
W_MOVE = 1 / 0.29
W_Q = {
    n: W_ENERGY if n in FLOW else W_MOVE if n in ("x", "y", "vx", "vy") else 1.0
    for n in COORDS
}

GROUND, P1, P2 = 0, 1, 2  # owners: the ground (neutral), the players
OWNERS = 3
# Δx, Δy, distance in the plane, Δz, and the bearing of the probe from the source's
# heading (cosine, sine) between a probe and a source
GEOM = 6
SCALE = 8.0  # map units the plane geometry is divided by
Z_SCALE = 2.0  # a cliff level
ROWS_SCALE = 64.0  # the summed row contexts of a player are divided by this
ENERGY_REGEN = 0.7875 / 22.4  # energy a loop, where the type has a pool
ENERGY_UNIT = 200.0  # the energy coordinate is a fraction of this


@dataclass
class State:
    """M frames of N rows; d the context width; P = 2 players; u_dim the input width."""

    q: torch.Tensor  # [M, N, len(COORDS)] float: the rows' integrated coordinates
    g: torch.Tensor  # [M, N, len(GIVEN)] float: the rows' given coordinates
    k: torch.Tensor  # [M, N, len(CONSTS)] float: the types' constants
    c: torch.Tensor  # [M, N, d] float: the rows' semantic context
    owner: torch.Tensor  # [M, N] long: GROUND, P1 or P2
    mask: torch.Tensor  # [M, N] bool: a real row
    p: torch.Tensor  # [M, P, len(PLAYER)] float: the players' observed coordinates
    cp: torch.Tensor  # [M, P, d] float: the players' context (race, upgrades, ...)
    u: torch.Tensor  # [M, N, u_dim] float: this frame's inputs to each row (commands)
    loop: float = 1.0  # loops a step
    # the rows' coordinates as sources, when they are to be told apart from the rows'
    # coordinates as probes (the dynamics differentiates the probe side only); the
    # same values as q
    qs: torch.Tensor | None = None

    @property
    def q_source(self):
        return self.q if self.qs is None else self.qs

    def replace(self, q=None, p=None, qs=None):
        return State(
            self.q if q is None else q,
            self.g,
            self.k,
            self.c,
            self.owner,
            self.mask,
            self.p if p is None else p,
            self.cp,
            self.u,
            self.loop,
            self.qs if qs is None else qs,
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


def row_values(k):
    """[M, N, len(COORDS)] what a unit of each coordinate is worth in resource units:
    hp and shield by the value's share, contents 1, the rest 0 (not energy)."""
    hm, sm = k[..., K["hp_max"]], k[..., K["shield_max"]]
    share = k[..., K["value"]] / (hm + sm).clamp(min=1)
    v = torch.zeros(*k.shape[:-1], len(COORDS), device=k.device, dtype=k.dtype)
    v[..., Q["hp"]] = share * hm
    v[..., Q["shield"]] = share * sm
    v[..., Q["contents"]] = 1.0
    return v


def features(s, phys):
    """What a row is to a term: its context, the coordinates `phys` of its state, its
    givens, its constants."""
    return torch.cat(
        [s.c, scaled(s.q, Q_SCALE)[..., phys], scaled(s.g, G_SCALE), constants(s.k)],
        -1,
    )


def source(s):
    """What a row is as a source: its context, its state without hp and shields
    (energy, contents), its givens (cooldown: a disruptor that has fired is a different
    source from one that has not; heading) and its constants."""
    return torch.cat(
        [
            s.c,
            scaled(s.q_source, Q_SCALE)[..., SRC],
            scaled(s.g, G_SCALE),
            constants(s.k),
        ],
        -1,
    )


def d_features(phys):
    return len(phys) + len(GIVEN) + len(CONSTS)


def geometry(s):
    """[M, N, N, GEOM] from every probe (rows as probes, q) to every source (rows as
    sources, q_source): Δ and distance in the plane, Δz, and the bearing of the probe
    seen from the source's heading, as cosine and sine. Scaled."""
    x, xs = s.q[..., POS], s.q_source[..., POS]
    delta = xs[:, None] - x[:, :, None]  # probe -> source
    dist = delta.norm(dim=-1, keepdim=True)
    dz = (s.g[:, None, :, G["z"]] - s.g[:, :, None, G["z"]])[..., None]
    # the direction from the source to the probe against the source's heading
    d = -delta / dist.clamp(min=1e-6)
    hc, hs = s.g[:, None, :, G["cos"]], s.g[:, None, :, G["sin"]]
    cos_b = d[..., 0] * hc + d[..., 1] * hs
    sin_b = hc * d[..., 1] - hs * d[..., 0]
    return torch.cat(
        [delta / SCALE, dist / SCALE, dz / Z_SCALE, cos_b[..., None], sin_b[..., None]],
        -1,
    )


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

    def forward(self, a, b, geom):
        """a [M, X, d_a], b [M, N, d_b], geom [M, X, N, GEOM] -> [M, X, N, outputs]."""
        h = self.a(a)[:, :, None] + self.b(b)[:, None] + self.g(geom)
        return self.out(F.gelu(h))


# ---------------------------------------------------------------- the fields


class Interaction(nn.Module):
    """The cross-player pair term: e(probe i, source j, geometry) for every pair of rows
    of different players, source j's projection (its type, cooldown, energy, the range,
    the bearing) meeting probe i's presence (its type, hp, shields). It enters i's
    owner's V with + and j's owner's with −, so the two players' sums cancel pair by
    pair. Its gradient on i's hp is the damage; on i's position it is repulsion from
    incoming fire, on j's position attraction toward what j can hit, both unused here
    (no spatial force) and the action level's to read. Same-owner pairs contribute
    nothing: no friendly fire, no healing by sign. The probe's contents are not read:
    this term moves hp and shields only."""

    def __init__(self, d, hidden):
        super().__init__()
        self.e = PairMLP(d + d_features(PROBE_E), d + d_features(SRC), hidden)

    def forward(self, s, geom):
        """-> (absorbed [M, N]: Σ_j e_ij at each probe, projected [M, N]: Σ_i e_ij at
        each source)."""
        ground = s.owner == GROUND
        gi, gj = ground[:, :, None], ground[:, None, :]
        cross = (s.owner[:, :, None] != s.owner[:, None, :]) & ~gi & ~gj
        both = s.mask[:, :, None] & s.mask[:, None, :]
        e = self.e(features(s, PROBE_E), source(s), geom).squeeze(-1) * (both & cross)
        return e.sum(-1), e.sum(-2)


class Neutral(nn.Module):
    """ψ, the neutral term at each row from its pairs with the ground, one function
    either way: a patch pulls the workers that can mine it, the workers mining it drain
    its contents (the gradient on the patch's contents, with the patch as the probe).
    Exactly one side the ground. The probe's hp and shields are not read: the ground
    does no damage."""

    def __init__(self, d, hidden):
        super().__init__()
        self.psi = PairMLP(d + d_features(PROBE_PSI), d + d_features(SRC), hidden)

    def forward(self, s, geom):
        """-> ψ [M, N] at each probe."""
        ground = s.owner == GROUND
        neutral = ground[:, :, None] ^ ground[:, None, :]
        both = s.mask[:, :, None] & s.mask[:, None, :]
        psi = self.psi(features(s, PROBE_PSI), source(s), geom).squeeze(-1)
        return (psi * (both & neutral)).sum(-1)


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
    """H = T + V per owner, in resource units.

    T   kinetic, read out of the observed velocities: ½ value (|v| / speed)² per row,
        the value in motion (mass is the dynamics' business: how fast an order becomes
        motion); ½ |rates|² per player, the ledger's one-loop flows in resource units.
        Every velocity is observed, so T is never differentiated
    V   learned: a self term f(c, q without position or contents, g, k) per row; the
        interaction, + at the probe's owner and − at the source's; the neutral term;
        the player term
    """

    def __init__(self, d, hidden=256):
        super().__init__()
        self.f = mlp(d + d_features(PROBE_E), 1, hidden, bias=False)
        zero(self.f[2])
        self.interaction = Interaction(d, hidden)
        self.neutral = Neutral(d, hidden)
        self.player = PlayerPotential(d, hidden)

    def kinetic(self, s):
        """[M, OWNERS]."""
        speed = s.k[..., K["speed"]]
        frac = s.q[..., VEL].norm(dim=-1) / speed.clamp(min=1e-6) * (speed > 0)
        t = by_owner(0.5 * s.k[..., K["value"]] * frac.square(), s.owner, s.mask)
        rates = 0.5 * (s.p * s.p.new_tensor(P_VALUE))[..., LRATES].square().sum(-1)
        return torch.cat([t[:, :P1], t[:, P1:] + rates], 1)

    def potential(self, s):
        """-> (V [M, OWNERS], its parts by owner). `absorbed` is what each player's rows
        take, `interaction` the zero-sum difference (absorbed minus projected),
        `player` the ledger term (0 for the ground)."""
        geom = geometry(s)
        own = self.f(features(s, PROBE_E)).squeeze(-1)
        absorbed, projected = self.interaction(s, geom)
        absorbed = by_owner(absorbed, s.owner, s.mask)
        player = self.player(s)
        player = torch.cat([torch.zeros_like(player[:, :1]), player], 1)
        parts = {
            "own": by_owner(own, s.owner, s.mask),
            "interaction": absorbed - by_owner(projected, s.owner, s.mask),
            "neutral": by_owner(self.neutral(s, geom), s.owner, s.mask),
            "player": player,
        }
        v = sum(parts.values())
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
    (motion, a cast), and spending on its player's ledger. A player's input reaches
    only that player's rows (B is block diagonal by owner, which the per-row u already
    is)."""

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

    The force on a row is −∇ of its owner's V by autograd on the rows' q (as probes;
    the rows as sources are held fixed) and the players' p: one gradient of own + ψ +
    player + the plain sum of the cross-player pair term. It moves hp, shield and
    contents, and the bank and supply, as gradient flows with unit mobility in
    resource units (ΔE = −∂V/∂E, so Δq = −∂V/∂q / value²); nothing else. Position and
    velocity answer to the inputs and the motion tables only. Energy regenerates where
    the type has a pool. A row whose hp reaches zero is dead; nothing is subtracted for
    it, its value leaves with the row.
    """

    def __init__(self, d, u_dim, hidden=256):
        super().__init__()
        self.B = Input(u_dim, hidden)
        mob_p = torch.zeros(len(PLAYER))
        for n in PLAYER[LEDGER]:
            mob_p[PL[n]] = 1 / P_VALUE[PL[n]] ** 2
        self.register_buffer("mobility_p", mob_p)

    @staticmethod
    def mobility(k):
        """[M, N, len(COORDS)] 1 / value² on the flow axes, 0 elsewhere."""
        v = row_values(k)
        return torch.where(v > 0, 1 / v.clamp(min=1e-6).square(), torch.zeros_like(v))

    def forward(self, s, H):
        """s the frames, H the Hamiltonian -> (q_next [M, N, len(COORDS)], p_next
        [M, P, len(PLAYER)], pieces)."""
        q = s.q.detach().requires_grad_(True)
        p = s.p.detach().requires_grad_(True)
        out = H(s.replace(q=q, p=p, qs=s.q.detach()))
        # the owner's potential, differentiated on the probe side. Training
        # differentiates through the force (double backward); evaluation needs the
        # force but no graph behind it, and runs under enable_grad
        own_side = (out["own"] + out["neutral"] + out["player"]).sum() + out[
            "absorbed"
        ].sum()
        grad_q, grad_p = torch.autograd.grad(
            own_side, (q, p), create_graph=self.training
        )
        flow, flow_p = -grad_q * self.mobility(s.k), -grad_p * self.mobility_p
        k, dt = s.k, s.loop
        mass, speed = k[..., K["mass"]], k[..., K["speed"]]
        bu, bu_p = self.B(s)
        # motion: the input is the force; instant types take it as their velocity, the
        # rest accelerate by force / mass; the cap is the type's speed
        instant = (mass <= 0)[..., None]
        m = mass.clamp(min=1e-6)[..., None]
        v_next = torch.where(instant, bu[..., VEL], q[..., VEL] + bu[..., VEL] / m * dt)
        v_next = v_next * (
            speed[..., None] / v_next.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ).clamp(max=1.0)
        pos_next = q[..., POS] + v_next * dt
        # gradient flows: hp and shield down the interaction, contents down ψ
        hp = (q[..., HP] + flow[..., HP] * dt).clamp(0, 1)
        alive = hp[..., 0] > 0
        contents = (q[..., Q["contents"]] + flow[..., Q["contents"]] * dt).clamp(min=0)
        # energy: the known regeneration where the type has a pool, and the casts
        e_max = k[..., K["energy_max"]]
        has_pool = e_max > 0
        energy = (
            q[..., Q["energy"]]
            + ENERGY_REGEN / ENERGY_UNIT * dt * has_pool
            + bu[..., Q["energy"]] * dt
        ).clamp(min=0)
        energy = torch.minimum(energy, e_max / ENERGY_UNIT)
        q_next = torch.cat(
            [pos_next, v_next, hp, energy[..., None], contents[..., None]], -1
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
    (DESIGN_energy.md, why it cannot collapse). The energy axes are scored in resource
    units with one weight, so the loss on a bar is proportional to what the bar is
    worth; position and velocity in a fast unit's move. The kinematic axes are not
    scored on the frame a row dies in (the engine freezes them there); the ledger's
    rates are read, not predicted, so they are logged but not trained on."""

    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.tensor([W_Q[n] for n in COORDS]))
        kin = torch.zeros(len(COORDS), dtype=torch.bool)
        kin[[Q[n] for n in KINEMATIC]] = True
        self.register_buffer("kinematic", kin)
        self.register_buffer(
            "wp", torch.tensor([W_ENERGY * v for v in P_VALUE], dtype=torch.float)
        )
        use_p = torch.ones(len(PLAYER))
        use_p[LRATES] = 0.0
        self.register_buffer("use_p", use_p)

    def forward(self, q_pred, q_next, mask, alive_next, k, p_pred, p_next):
        """mask [M, N] the rows scored, alive_next [M, N] the rows alive in the next
        frame, k the constants -> (loss, per-axis residuals [len(COORDS)],
        [len(PLAYER)])."""
        v = row_values(k)
        w = torch.where(v > 0, v, torch.ones_like(v)) * self.w
        r = F.smooth_l1_loss(q_pred * w, q_next * w, reduction="none")
        m = mask[..., None] & (~self.kinematic | alive_next[..., None])
        rows = (r * m).sum((0, 1)) / m.sum((0, 1)).clamp(min=1)
        players = F.smooth_l1_loss(
            p_pred * self.wp, p_next * self.wp, reduction="none"
        ).mean((0, 1))
        return rows.sum() + (players * self.use_p).sum(), rows, players


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
        alive_next = q_next[..., Q["hp"]] > 0
        q_pred, p_pred, pieces = self.step(s)
        loss, rows, players = self.residual(
            q_pred, q_next, scored, alive_next, s.k, p_pred, p_next
        )
        logs = {f"res_{n}": rows[i].detach() for n, i in Q.items()}
        logs.update({f"res_{n}": players[i].detach() for n, i in PL.items()})
        # each trained axis's share of the loss
        total = loss.detach().clamp(min=1e-12)
        use_p = self.residual.use_p
        logs.update({f"share_{n}": rows[i].detach() / total for n, i in Q.items()})
        logs.update(
            {
                f"share_{n}": players[i].detach() / total
                for n, i in PL.items()
                if use_p[i] > 0
            }
        )
        # the copy baseline: nothing moves. What a constant H would score
        _, crows, cplayers = self.residual(
            s.q, q_next, scored, alive_next, s.k, s.p, p_next
        )
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
        # the accounting, from what happened: value lost to damage this step (hp and
        # shield in resource units), and the value of the rows that died (the
        # dissipation)
        v = row_values(s.k)
        lost = -((q_next[..., HP] - s.q[..., HP]) * v[..., HP]).sum(-1)
        died = ((s.q[..., Q["hp"]] > 0) & ~alive_next).to(v.dtype)
        lost, died = (
            by_owner(lost, s.owner, scored),
            by_owner(died * s.k[..., K["value"]], s.owner, scored),
        )
        for o in range(OWNERS):
            logs[f"lost_{o}"] = lost[:, o].mean().detach()
            logs[f"dissipated_{o}"] = died[:, o].mean().detach()
        return loss, logs
