"""Stochastic latent actions: the distribution over what happens next.

A predictor that gets only the present and is trained with a regression loss predicts the
average of the futures that present allows (spire and hydralisk den at once), and a JEPA
encoder then learns to stop representing whatever was unpredictable. The fix is a latent z
per (pair, player) that the predictor is conditioned on, estimated three ways that differ
only in what they see:

    posterior  q(z | present, future)    the inverse model; training signal only
    commands   q(z | present, commands)  the command encoder; the controllable input
    prior      p(z | present)            what players tend to do from here; sampled at
                                         inference, a mixture so it can hold several branches

Losses (train_pairs.py): the predictor is driven by a sample of the posterior (or of the
command distribution); KL(posterior || prior) is charged per nat above --free_nats (the
decision itself is always affordable) and trains the prior; KL(posterior || commands) ties
the command encoder to what actually happened; an entropy bonus on the prior's batch-mean
component usage and an optional repulsion between component means keep the mixture's
components distinct (the same job as load balancing in a mixture of experts).

Diagnostics: KL in nats, active units (posterior means that vary across data, Burda et al.),
effective number of components used, and in evaluation the prediction error with the prior's
mean vs one sample vs the best of K samples vs the posterior (the oracle). A gap between the
mean and best-of-K says the future was multimodal and the prior spans it.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

LOG2PI = math.log(2 * math.pi)
LV_MIN, LV_MAX = -9.0, 4.0


def split_gauss(x):
    """[..., 2 d] -> mean [..., d], log-variance [..., d] (clamped), in float32."""
    mu, lv = x.float().chunk(2, -1)
    return mu, lv.clamp(LV_MIN, LV_MAX)


def rsample(mu, lv):
    return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)


def gauss_logp(z, mu, lv):
    """log N(z; mu, diag exp(lv)), summed over the last dimension."""
    return -0.5 * (lv + (z - mu) ** 2 / lv.exp() + LOG2PI).sum(-1)


def kl_gauss(mu_q, lv_q, mu_p, lv_p):
    """KL(N_q || N_p) for diagonal Gaussians, summed over the last dimension (nats)."""
    return 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1).sum(-1)


class MixturePrior(nn.Module):
    """p(z | present) per player: a mixture of K diagonal Gaussians whose weights, means and
    variances are functions of the present frame's CLS slot and the player's slot. K = 1 is a
    plain Gaussian prior."""

    def __init__(self, d, d_z, K=8, max_players=8):
        super().__init__()
        self.K, self.d_z = K, d_z
        self.player = nn.Embedding(max_players, d)
        self.net = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, 4 * d), nn.GELU(), nn.Linear(4 * d, K + 2 * K * d_z))
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():  # spread the components' means a little at init so they can specialize
            self.net[-1].weight.mul_(0.5)

    def forward(self, cls, ply):
        """cls [B, d], ply [B, P, d] -> dict(logits [B, P, K], mu [B, P, K, d_z], lv [B, P, K, d_z])."""
        B, P, _ = ply.shape
        x = torch.cat([cls[:, None].expand(-1, P, -1), ply + self.player.weight[:P][None].to(ply.dtype)], -1)
        o = self.net(x).float()
        logits, rest = o[..., :self.K], o[..., self.K:].view(B, P, self.K, 2, self.d_z)
        return dict(logits=logits, mu=rest[..., 0, :], lv=rest[..., 1, :].clamp(LV_MIN, LV_MAX))


def mix_component_logp(z, p):
    """log pi_k + log N(z; mu_k, lv_k) per component: z [B, P, d_z] -> [B, P, K]."""
    return F.log_softmax(p["logits"], -1) + gauss_logp(z[:, :, None], p["mu"], p["lv"])


def mix_logp(z, p):
    return torch.logsumexp(mix_component_logp(z, p), -1)


def mix_resp(z, p):
    """Responsibility of each component for z, [B, P, K]."""
    return F.softmax(mix_component_logp(z, p), -1)


def mix_mean(p):
    return (F.softmax(p["logits"], -1)[..., None] * p["mu"]).sum(-2)


@torch.no_grad()
def mix_sample(p):
    """One z per (pair, player) from the mixture: pick a component, then its Gaussian."""
    k = torch.distributions.Categorical(logits=p["logits"]).sample()                     # [B, P]
    idx = k[..., None, None].expand(-1, -1, 1, p["mu"].shape[-1])
    mu, lv = p["mu"].gather(2, idx)[:, :, 0], p["lv"].gather(2, idx)[:, :, 0]
    return rsample(mu, lv)


def kl_mc(z, mu_q, lv_q, p):
    """A one-sample estimate of KL(q || p) for a Gaussian q and a mixture p, with z ~ q."""
    return gauss_logp(z, mu_q, lv_q) - mix_logp(z, p)


def usage(resp, mask):
    """How evenly the prior's components are used over a batch: (loss = log K - H(mean
    responsibility), effective number of components = exp H). resp [B, P, K], mask [B, P]."""
    m = mask.float()[..., None]
    r = (resp * m).sum((0, 1)) / m.sum().clamp(min=1)
    H = -(r * (r + 1e-8).log()).sum()
    return math.log(resp.shape[-1]) - H, H.exp()


def diversity(p, mask):
    """Repulsion between a state's component means: mean |cosine| over component pairs."""
    mu = F.normalize(p["mu"], dim=-1)                                                   # [B, P, K, d_z]
    K = mu.shape[-2]
    if K < 2:
        return mu.sum() * 0
    cos = mu @ mu.transpose(-1, -2)                                                      # [B, P, K, K]
    off = cos.abs().sum((-1, -2)) - K                                                    # minus the diagonal
    per = off / (K * (K - 1))
    m = mask.float()
    return (per * m).sum() / m.sum().clamp(min=1)


def active_units(post_mu, threshold=0.01):
    """Burda et al.: dimensions whose posterior mean varies across data. post_mu [N, d_z]."""
    if len(post_mu) < 2:
        return 0
    return int((post_mu.var(0) > threshold).sum())
