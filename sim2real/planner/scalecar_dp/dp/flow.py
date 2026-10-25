"""Flow matching with a per-token noise ceiling (training-time soft RTC), x0 parameterisation.

Convention: x_tau = (1 - tau) x0 + tau eps, tau = 1 is noise. Each token i has a ceiling tau_bar_i:
0 = hard prefix (input is the given prefix), (0,1) = soft prior (integration starts from
(1 - tau_bar) prior + tau_bar eps and the token stays frozen until the global time drops below tau_bar),
1 = free. The network sees tau_i = min(tau, tau_bar_i) and predicts x0.
"""
from __future__ import annotations

import torch


def sample_tau(B: int, device, a: float = 1.5, b: float = 1.0) -> torch.Tensor:
    """tau ~ Beta(1.5, 1): biased towards the noisy end (pi0 / LeRobot flow-matching default)."""
    return torch.distributions.Beta(torch.tensor(a, device=device), torch.tensor(b, device=device)).sample((B,))


def training_inputs(x0: torch.Tensor, prior: torch.Tensor, tau_bar: torch.Tensor, tau: torch.Tensor):
    """x0, prior (B,T,D) normalized; tau_bar (B,T); tau (B,) -> (x, tau_i)."""
    eps = torch.randn_like(x0)
    t = tau.view(-1, 1)
    tau_i = torch.minimum(t.expand_as(tau_bar), tau_bar)
    frozen = (t >= tau_bar).unsqueeze(-1)            # inference would still hold the prior start state
    start = (1 - tau_bar).unsqueeze(-1) * prior + tau_bar.unsqueeze(-1) * eps
    moving = (1 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * eps
    return torch.where(frozen, start, moving), tau_i


def loss(pred_x0: torch.Tensor, x0: torch.Tensor, prefix: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """MSE on non-prefix, valid frames (normalized units)."""
    w = (~prefix & valid).to(pred_x0.dtype).unsqueeze(-1)
    return ((pred_x0.float() - x0.float()) ** 2 * w).sum() / (w.sum() * x0.shape[-1]).clamp(min=1)


@torch.no_grad()
def sample(denoise, prior: torch.Tensor, tau_bar: torch.Tensor, steps: int = 10, generator=None,
           eps: torch.Tensor | None = None) -> torch.Tensor:
    """Euler/DDIM integration from tau = 1 to 0. ``denoise(x, tau_i) -> x0_hat``; prior (B,T,D) normalized.
    ``eps`` (B,T,D): the initial noise; drawn from ``generator`` when omitted (callers that capture the
    sampler in a CUDA graph pass it explicitly)."""
    if eps is None:
        eps = torch.randn(prior.shape, device=prior.device, generator=generator, dtype=prior.dtype)
    x = (1 - tau_bar).unsqueeze(-1) * prior + tau_bar.unsqueeze(-1) * eps
    grid = torch.linspace(1, 0, steps + 1, device=prior.device)
    for a, b in zip(grid[:-1], grid[1:]):
        tau_cur = torch.minimum(torch.full_like(tau_bar, float(a)), tau_bar)
        x0 = denoise(x, tau_cur)
        move = (tau_cur > b).unsqueeze(-1)
        tc = tau_cur.clamp(min=1e-6).unsqueeze(-1)
        e_hat = (x - (1 - tc) * x0) / tc
        x = torch.where(move, (1 - b) * x0 + b * e_hat, x)
    return x
