"""Loss terms for training physics-informed neural networks."""

from __future__ import annotations

import torch

from .preprocessing import normalize_inputs


def compute_pde_residual(
    model,
    S: torch.Tensor,
    t: torch.Tensor,
    sigma: torch.Tensor,
    r: float = 0.05,
    *,
    features: torch.Tensor | None = None,
    model_output: torch.Tensor | None = None,
):
    """Compute the Black-Scholes PDE residual for the current network."""
    if features is None:
        features = normalize_inputs(S, t, sigma)

    if model_output is None:
        V = model(features).squeeze(-1)
    else:
        V = model_output.squeeze(-1)

    ones = torch.ones_like(V)
    dV_dS = torch.autograd.grad(V, S, ones, create_graph=True, retain_graph=True)[0]
    dV_dt = torch.autograd.grad(V, t, ones, create_graph=True, retain_graph=True)[0]
    d2V_dS2 = torch.autograd.grad(dV_dS, S, torch.ones_like(dV_dS), create_graph=True)[0]

    pde = dV_dt + 0.5 * sigma**2 * S**2 * d2V_dS2 + r * S * dV_dS - r * V
    return pde


def pinn_loss(model, S, t, sigma, target, r=0.05, λ=0.01):
    """Combine data-fit, PDE, and smoothness losses for PINN training.

    Returns the total loss along with each component so callers can log them.
    """
    S = S.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    sigma = sigma.clone().detach()

    features = normalize_inputs(S, t, sigma)
    pred = model(features).squeeze(-1)
    L_price = torch.mean((pred - target)**2)

    residual = compute_pde_residual(
        model, S, t, sigma, r, features=features, model_output=pred
    )
    L_PDE = torch.mean(residual**2)

    grads = torch.autograd.grad(
        pred, S, torch.ones_like(pred), create_graph=True
    )[0]
    L_reg = λ * torch.mean(grads**2)
    return L_price + L_PDE + L_reg, (L_price, L_PDE, L_reg)
