"""The energy model (DESIGN_energy.md): StarCraft II as a closed, dissipative, zero-sum
system in port-Hamiltonian form,

    ẋ = J ∇H(x) + B u

with no dissipation term: energy leaves the system only when a row dies, which is the row
vanishing and its value with it, not a rate the model predicts. Damage is the opponent's
field acting on a row's hp coordinate, a force like any other.

A skeleton: the pieces and the shapes that pass between them, on state vectors handed in.
No entity encoding (the semantic context c of each row arrives as a vector), no training
loop, nothing tested yet. Each class says which part of the design it is.

The pieces, in the order a step runs them:

    State        what one frame is. Rows: coordinates q (observed, differentiable),
                 constants k (from the patch's tables), context c (semantic: type,
                 orders, buffs... as a vector), owner, mask. Players: coordinates p (the
                 ledger: bank, income, supply, value; observed, differentiable) and
                 context cp. The inputs u (what this frame's commands do, as vectors)
    Hamiltonian  H = T + V, read out per owner: H_0 the ground, H_1, H_2 the players.
                 One function, two instances: the owner enters only as a relation (same,
                 enemy, neutral) in the pair term, so the two players share every weight
                 (swap symmetry). The pair term is the port: the field one player's
                 sources create at the other's rows. The player term couples a player's
                 ledger to its own rows
    Dynamics     one step of J ∇H + B u: ∇V by autograd on rows and players (T is read
                 out, not a force); J the position-velocity pairing (second order with
                 the type's acceleration, overdamped when it is instant) and a plain
                 gradient flow on hp, shield and the ledger; a row whose hp reaches zero
                 is dead and leaves; B u the inputs, acting only on the owner's own rows
    ForceResidual the only objective: the step's coordinates against the observed next
                 ones, per axis, rows and players
    EnergyModel  the three together: forward(state, next coordinates) -> loss, logs;
                 step(state) -> next coordinates
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------- the state's layout

# q: a row's observed coordinates, map units and loops; hp, shield and energy as
# fractions of their maxima; contents the resources left in a patch
COORDS = (
    "x",
    "y",
    "vx",
    "vy",
    "hp",
    "shield",
    "energy",
    "cooldown",
    "progress",
    "contents",
)
Q = {n: i for i, n in enumerate(COORDS)}
POS, VEL = slice(Q["x"], Q["y"] + 1), slice(Q["vx"], Q["vy"] + 1)
HP = slice(Q["hp"], Q["shield"] + 1)  # flow down the field's gradient; zero hp is death
CLOCKS = ("cooldown", "progress")  # known rates, reset by events the model predicts
MOBILITY = 1.0  # fraction of max hp a loop per unit of the field's gradient

# p: a player's observed coordinates, the ledger, in resource units and loops: the
# bank, its rates (the flows: income less spending), supply, the value standing in
# units and structures, workers. Their velocities are the imbalance of flow between
# the players; the player's own rows are their sources
PLAYER = (
    "minerals",
    "vespene",
    "mineral_rate",
    "vespene_rate",
    "food_used",
    "food_cap",
    "army_value",
    "structure_value",
    "workers",
)
PL = {n: i for i, n in enumerate(PLAYER)}
BANK, RATES = (
    slice(PL["minerals"], PL["vespene"] + 1),
    slice(PL["mineral_rate"], PL["vespene_rate"] + 1),
)

# k: a row's constants from the patch's tables (physics_data.motion_table and the cost
# table, later): value its cost in resource units (a patch: its contents' worth), mass
# the inverse of its acceleration (0: instant), speed its cap, the maxima
CONSTS = ("value", "mass", "speed", "hp_max", "shield_max", "energy_max")
K = {n: i for i, n in enumerate(CONSTS)}

GROUND, P1, P2 = 0, 1, 2  # owners: the ground (neutral), the players
OWNERS = 3


@dataclass
class State:
    """One frame. N rows; d the context width; P = 2 players; u_dim the input width."""

    q: torch.Tensor  # [N, len(COORDS)] float: the rows' observed coordinates
    k: torch.Tensor  # [N, len(CONSTS)] float: the types' constants
    c: torch.Tensor  # [N, d] float: the rows' semantic context, from an encoder later
    owner: torch.Tensor  # [N] long: GROUND, P1 or P2
    mask: torch.Tensor  # [N] bool: a real row
    p: torch.Tensor  # [P, len(PLAYER)] float: the players' observed coordinates
    cp: torch.Tensor  # [P, d] float: the players' context (race, upgrades, ...)
    u: torch.Tensor  # [N, u_dim] float: this frame's inputs to each row (commands)
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
    """[N] per-row values -> [OWNERS] sums over real rows of each owner."""
    out = torch.zeros(OWNERS, dtype=values.dtype, device=values.device)
    return out.index_add(0, owner, values * mask.to(values.dtype))


def mlp(d_in, d_out, hidden):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))


# ---------------------------------------------------------------- the Hamiltonian


class PairPotential(nn.Module):
    """The port between the players: the field each row's sources create at every other
    row, summed. g(c_i, c_j, q_j − q_i, relation) for all real pairs i, j, with the
    relation same / enemy / neutral the only place the owner enters, so swapping the
    players changes nothing. O(N²) here; attention with a pair bias later."""

    def __init__(self, d, hidden):
        super().__init__()
        self.rel = nn.Embedding(3, hidden)  # same, enemy, neutral
        self.g = mlp(2 * d + 3, 1, hidden)  # both contexts, dx, dy, distance

    def forward(self, s):
        """-> [N] the potential at each row from all the others."""
        N = s.q.shape[0]
        d = s.q[None, :, POS] - s.q[:, None, POS]  # [N, N, 2] from i to j
        dist = d.norm(dim=-1, keepdim=True)
        same = s.owner[:, None] == s.owner[None, :]
        neutral = (s.owner[:, None] == GROUND) | (s.owner[None, :] == GROUND)
        rel = torch.where(neutral, 2, torch.where(same, 0, 1))
        pair = torch.cat(
            [
                s.c[:, None].expand(N, N, -1),
                s.c[None, :].expand(N, N, -1),
                torch.cat([d / 8, dist / 8], -1),
            ],
            -1,
        )
        h = self.g[0](pair) + self.rel(rel)
        v = self.g[2](F.gelu(h)).squeeze(-1)  # [N, N]
        valid = (
            s.mask[:, None]
            & s.mask[None, :]
            & ~torch.eye(N, dtype=torch.bool, device=v.device)
        )
        return (v * valid).sum(1)


class PlayerPotential(nn.Module):
    """A player's ledger in the field of its own rows: h(cp, p, the sum of the player's
    row contexts). Its gradient on p is the flow of the bank, supply and value that the
    rows cause: workers mining, structures producing, an army standing. The sum over
    rows is the plainest posture summary; a Gram or an attention pooling later."""

    def __init__(self, d, hidden):
        super().__init__()
        self.h = mlp(2 * d + len(PLAYER), 1, hidden)

    def forward(self, s):
        """-> [P] the potential of each player's ledger."""
        P = s.p.shape[0]
        rows = torch.zeros(OWNERS, s.c.shape[-1], device=s.c.device, dtype=s.c.dtype)
        rows = rows.index_add(0, s.owner, s.c * s.mask[:, None].to(s.c.dtype))
        return self.h(torch.cat([s.cp, s.p, rows[P1 : P1 + P]], -1)).squeeze(-1)


class Hamiltonian(nn.Module):
    """H = T + V per owner.

    T   kinetic: ½ mass |v|² per row, mass from the tables (0 for instant types); for a
        player, ½ |rates|², the energy of the flow through its ledger
    V   learned: a self term f(c, q) per row, the pair term across rows, the player term
        on the ledger
    """

    def __init__(self, d, hidden=256):
        super().__init__()
        self.f = mlp(d + len(COORDS), 1, hidden)
        self.pair = PairPotential(d, hidden)
        self.player = PlayerPotential(d, hidden)

    def kinetic(self, s):
        """[OWNERS] ½ m |v|² over each owner's rows, and ½ |rates|² for each player."""
        t = by_owner(
            0.5 * s.k[:, K["mass"]] * s.q[:, VEL].square().sum(-1), s.owner, s.mask
        )
        t[P1:] = t[P1:] + 0.5 * s.p[:, RATES].square().sum(-1)
        return t

    def potential(self, s):
        """[OWNERS] the learned part: each row's own term, what every other row's field
        is worth at it (the coupling: a player's rows sit in the field the other player's
        sources make), and the player term on its ledger."""
        own = self.f(torch.cat([s.c, s.q], -1)).squeeze(-1)
        v = by_owner(own + self.pair(s), s.owner, s.mask)
        v[P1:] = v[P1:] + self.player(s)
        return v

    def forward(self, s):
        """-> dict: H [OWNERS] and its parts; total their sum."""
        parts = dict(T=self.kinetic(s), V=self.potential(s))
        H = parts["T"] + parts["V"]
        return dict(H=H, total=H.sum(), **parts)


# ---------------------------------------------------------------- the dynamics


class Input(nn.Module):
    """B u: what this frame's commands do, as a force on the row's own coordinates and
    on its player's ledger (a spend, a queued unit). A player's input reaches only that
    player's rows (B is block diagonal by owner, which the per-row u already is)."""

    def __init__(self, u_dim, hidden=256):
        super().__init__()
        self.rows = mlp(u_dim, len(COORDS), hidden)
        self.player = mlp(u_dim, len(PLAYER), hidden)

    def forward(self, s):
        """-> (on rows [N, len(COORDS)], on players [P, len(PLAYER)])."""
        m = s.mask[:, None].to(s.u.dtype)
        on_rows = self.rows(s.u) * m
        on_players = torch.zeros(
            OWNERS, len(PLAYER), device=s.u.device, dtype=on_rows.dtype
        ).index_add(0, s.owner, self.player(s.u) * m)
        return on_rows, on_players[P1 : P1 + s.p.shape[0]]


class Dynamics(nn.Module):
    """One step of ẋ = J ∇H + B u.

    The force is −∇V, the learned potential's gradient by autograd on the rows' q and
    the players' p; T is read out of H but not differentiated. J pairs position with
    velocity: ẋ = v, and v̇ = force / mass, capped at the type's speed; a type with
    mass 0 (instant acceleration) is the overdamped limit, its velocity set directly by
    the force. On hp and shield the step is a plain gradient flow: the opponent's
    sources make the field climb with a row's hp, so it flows down, and a same-owner
    healer makes it fall, so it flows up. Nothing is subtracted for dissipation: a row
    whose hp reaches zero is dead, and its value leaves with it. The ledger flows the
    same way: the player term's gradient on p is income and growth, the inputs are
    spending. Energy (mana) regenerates by a known rate and is spent by inputs; the
    clocks run at known rates.
    """

    def __init__(self, d, u_dim, hidden=256):
        super().__init__()
        self.B = Input(u_dim, hidden)
        self.cooldown_rate, self.progress_rate = (
            1.0,
            0.0,
        )  # loops per loop; progress' rate is a type's build time, from k later
        self.energy_regen = 0.7875 / 22.4 / 200  # per loop, as a fraction of 200

    def forward(self, s, H):
        """s the frame, H the Hamiltonian -> (q_next [N, len(COORDS)], p_next
        [P, len(PLAYER)], pieces)."""
        q = s.q.detach().requires_grad_(True)
        p = s.p.detach().requires_grad_(True)
        sq = s.replace(q=q, p=p)
        out = H(sq)
        grad_q, grad_p = torch.autograd.grad(
            out["V"].sum(), (q, p), create_graph=torch.is_grad_enabled()
        )
        force, force_p = -grad_q, -grad_p  # −∇V on every coordinate
        k, dt = s.k, s.loop
        mass, speed = k[:, K["mass"]], k[:, K["speed"]]
        # J: position and velocity. Instant types take the force as their velocity;
        # the rest accelerate by force / mass. The cap is the type's speed.
        v = q[:, VEL]
        instant = mass <= 0
        v_next = torch.where(
            instant[:, None],
            force[:, POS],
            v + force[:, POS] / mass.clamp(min=1e-6)[:, None] * dt,
        )
        v_next = v_next * (
            speed[:, None] / v_next.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ).clamp(max=1.0)
        pos_next = q[:, POS] + v_next * dt
        # hp and shield: gradient flow in the field; at zero hp the row is dead
        hp = (q[:, HP] + MOBILITY * force[:, HP] * dt).clamp(0, 1)
        alive = hp[:, 0] > 0
        energy = (q[:, Q["energy"]] + self.energy_regen * dt).clamp(0, 1)
        cooldown = (q[:, Q["cooldown"]] - self.cooldown_rate * dt).clamp(min=0)
        progress = (q[:, Q["progress"]] + self.progress_rate * dt).clamp(0, 1)
        contents = q[:, Q["contents"]]
        bu, bu_p = self.B(s)  # the inputs: a command's push on every coordinate
        q_next = (
            torch.cat(
                [
                    pos_next,
                    v_next,
                    hp,
                    energy[:, None],
                    cooldown[:, None],
                    progress[:, None],
                    contents[:, None],
                ],
                -1,
            )
            + bu * dt
        )
        # the ledger: the bank moves by its rates, everything flows in the field
        p_next = p + (force_p + bu_p) * dt
        p_next = torch.cat(
            [p_next[:, BANK] + p[:, RATES] * dt, p_next[:, PL["mineral_rate"] :]], -1
        )
        return q_next, p_next, dict(force=force, force_p=force_p, alive=alive, **out)


# ---------------------------------------------------------------- the objective


class ForceResidual(nn.Module):
    """The one training signal: the step's coordinates against the observed next ones,
    rows and players. Observed coordinates, observed inputs, observed changes: a
    constant H attributes every change to the inputs alone and fails where units keep
    walking and workers keep mining, so H is forced to be non-constant
    (DESIGN_energy.md, why it cannot collapse). Per-axis weights put position, hp, the
    clocks and the ledger on one scale."""

    def __init__(self):
        super().__init__()
        w = torch.ones(len(COORDS))
        w[POS] = 1 / 0.29  # a fast unit's move a loop, map units
        w[VEL] = 1 / 0.29
        w[Q["hp"]] = w[Q["shield"]] = 10.0  # fractions a loop are small
        self.register_buffer("w", w)
        wp = torch.ones(len(PLAYER)) / 50  # resource units; a worker trip is 5
        wp[RATES] = 1.0
        wp[PL["food_used"]] = wp[PL["food_cap"]] = wp[PL["workers"]] = 1 / 8
        self.register_buffer("wp", wp)

    def forward(self, q_pred, q_next, mask, p_pred, p_next):
        """-> (loss, per-axis residuals [len(COORDS)], [len(PLAYER)])."""
        r = F.smooth_l1_loss(q_pred * self.w, q_next * self.w, reduction="none")
        rows = (r * mask[:, None]).sum(0) / mask.sum().clamp(min=1)
        players = F.smooth_l1_loss(
            p_pred * self.wp, p_next * self.wp, reduction="none"
        ).mean(0)
        return rows.sum() + players.sum(), rows, players


# ---------------------------------------------------------------- together


class EnergyModel(nn.Module):
    """Hamiltonian, dynamics and residual. forward(state, q_next, p_next) -> (loss,
    logs); step(state) -> the next coordinates. The two players are two instances of
    the same Hamiltonian read out per owner, coupled only through the pair term; at
    play time the opponent's rows and ledger come from the belief instead of the frame,
    and nothing else changes."""

    def __init__(self, d, u_dim, hidden=256):
        super().__init__()
        self.H = Hamiltonian(d, hidden)
        self.dynamics = Dynamics(d, u_dim, hidden)
        self.residual = ForceResidual()

    def step(self, s):
        return self.dynamics(s, self.H)

    def forward(self, s, q_next, p_next):
        q_pred, p_pred, pieces = self.step(s)
        loss, rows, players = self.residual(q_pred, q_next, s.mask, p_pred, p_next)
        logs = {f"res_{n}": rows[i].detach() for n, i in Q.items()}
        logs.update({f"res_{n}": players[i].detach() for n, i in PL.items()})
        logs.update(
            {f"H_{o}": pieces["H"][o].detach() for o in range(OWNERS)},
            T=pieces["T"].detach(),
            V=pieces["V"].detach(),
            advantage=(pieces["H"][P1] - pieces["H"][P2]).detach(),
        )
        return loss, logs
