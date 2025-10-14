"""Loss terms for training physics-informed neural networks."""

from __future__ import annotations

import torch

from .preprocessing import NormalizationConfig, normalize_inputs


def compute_pde_residual(
    model,
    S: torch.Tensor,
    t: torch.Tensor,
    sigma: torch.Tensor,
    r: float = 0.05,
    *,
    features: torch.Tensor | None = None,
    model_output: torch.Tensor | None = None,
    config: NormalizationConfig | None = None,
):
    """Compute the Black-Scholes PDE residual for the current network."""
    if features is None:
        features = normalize_inputs(S, t, sigma, config=config)

    if model_output is None:
        V = model(features).squeeze(-1)
    else:
        V = model_output.squeeze(-1)

    ones = torch.ones_like(V)
    dV_dS = torch.autograd.grad(V, S, ones, create_graph=True, retain_graph=True)[0]
    dV_dt = torch.autograd.grad(V, t, ones, create_graph=True, retain_graph=True)[0]
    d2V_dS2 = torch.autograd.grad(dV_dS, S, torch.ones_like(dV_dS), create_graph=True)[
        0
    ]

    pde = dV_dt + 0.5 * sigma**2 * S**2 * d2V_dS2 + r * S * dV_dS - r * V
    return pde


def pinn_loss(
    model,
    S,
    t,
    sigma,
    target,
    *,
    r: float = 0.05,
    λ: float = 0.01,
    boundary_weight: float = 1.0,
    config: NormalizationConfig | None = None,
):
    """Combine data-fit, PDE, and smoothness losses for PINN training.

    Returns the total loss along with each component so callers can log them.
    """
    S = S.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    sigma = sigma.clone().detach()

    features = normalize_inputs(S, t, sigma, config=config)
    pred = model(features).squeeze(-1)
    L_price = torch.mean((pred - target) ** 2)

    residual = compute_pde_residual(
        model,
        S,
        t,
        sigma,
        r,
        features=features,
        model_output=pred,
        config=config,
    )
    L_PDE = torch.mean(residual**2)

    ones = torch.ones_like(pred)
    delta = torch.autograd.grad(pred, S, ones, create_graph=True, retain_graph=True)[0]
    gamma = torch.autograd.grad(delta, S, torch.ones_like(delta), create_graph=True)[0]
    L_reg = λ * torch.mean(gamma**2)

    L_boundary = torch.tensor(0.0, device=pred.device)
    if boundary_weight > 0.0:
        cfg = config or NormalizationConfig()
        payoff = torch.clamp(S - cfg.K, min=0.0)
        t_terminal = torch.full_like(t, cfg.t_max)
        terminal_features = normalize_inputs(S, t_terminal, sigma, config=cfg)
        V_terminal = model(terminal_features).squeeze(-1)

        S_low = torch.full_like(S, cfg.S_min)
        low_features = normalize_inputs(S_low, t, sigma, config=cfg)
        V_low = model(low_features).squeeze(-1)

        S_high = torch.full_like(S, cfg.S_max)
        high_features = normalize_inputs(S_high, t, sigma, config=cfg)
        tau = torch.clamp(cfg.T - t, min=0.0)
        payoff_high = S_high - cfg.K * torch.exp(-cfg.r * tau)
        V_high = model(high_features).squeeze(-1)

        L_boundary = (
            torch.mean((V_terminal - payoff) ** 2)
            + torch.mean(V_low**2)
            + torch.mean((V_high - payoff_high) ** 2)
        ) / 3.0

    total = L_price + L_PDE + L_reg + boundary_weight * L_boundary
    return total, (L_price, L_PDE, L_reg, L_boundary)
