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
    cfg = config or NormalizationConfig()

    if features is None:
        features = normalize_inputs(S, t, sigma, config=cfg)

    if not features.requires_grad:
        features = features.detach().clone().requires_grad_(True)

    if model_output is None:
        V = model(features).squeeze(-1)
    else:
        V = model_output.squeeze(-1)

    ones = torch.ones_like(V)
    grad_feats = torch.autograd.grad(
        V, features, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]

    dV_dx_norm = grad_feats[:, 0]
    dV_dtau_norm = grad_feats[:, 1]

    # Compute second derivative w.r.t. features (not a slice)
    d2_feats = torch.autograd.grad(
        dV_dx_norm,
        features,
        grad_outputs=torch.ones_like(dV_dx_norm),
        create_graph=True,
        retain_graph=True,
    )[0]
    d2V_dx_norm2 = d2_feats[:, 0]

    x_range = max(cfg.x_max - cfg.x_min, 1e-6)
    tau_range = max(cfg.tau_range, 1e-6)

    dx_norm_dx = 2.0 / x_range
    dtau_norm_dtau = 2.0 / tau_range

    dV_dx = dV_dx_norm * dx_norm_dx
    dV_dS = dV_dx / S

    d2V_dx2 = d2V_dx_norm2 * (dx_norm_dx**2)
    d2V_dS2 = d2V_dx2 * (1.0 / (S**2)) + dV_dx * (-1.0 / (S**2))

    dV_dtau = dV_dtau_norm * dtau_norm_dtau
    dV_dt = -dV_dtau

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
    cfg = config or NormalizationConfig()

    # Create leaf tensors with gradients enabled
    S_leaf = S.detach().clone().requires_grad_(True)
    t_leaf = t.detach().clone().requires_grad_(True)
    sigma_leaf = sigma.detach().clone()

    # Forward pass
    features = normalize_inputs(S_leaf, t_leaf, sigma_leaf, config=cfg)
    pred = model(features).squeeze(-1)

    # Price loss (data fitting)
    L_price = torch.mean((pred - target) ** 2)

    # PDE residual loss - let it recompute to avoid graph conflicts
    residual = compute_pde_residual(
        model,
        S_leaf,
        t_leaf,
        sigma_leaf,
        r,
        features=None,
        model_output=None,
        config=cfg,
    )
    L_PDE = torch.mean(residual**2)

    # Gamma regularization (smoothness penalty)
    ones = torch.ones_like(pred)
    delta = torch.autograd.grad(
        pred, S_leaf, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]
    gamma = torch.autograd.grad(
        delta,
        S_leaf,
        grad_outputs=torch.ones_like(delta),
        create_graph=True,
        retain_graph=True,
    )[0]

    gamma_var = torch.mean(gamma**2)
    price_scale = L_price.detach()
    reg_scale = price_scale / (gamma_var.detach() + 1e-8)
    L_reg = λ * gamma_var * reg_scale

    # Boundary condition losses
    L_boundary = torch.tensor(0.0, device=pred.device)
    if boundary_weight > 0.0:
        # Terminal condition: V(T, S, σ) = max(S - K, 0)
        payoff = torch.clamp(S_leaf - cfg.K, min=0.0)
        t_terminal = torch.full_like(t_leaf, cfg.t_max)
        terminal_features = normalize_inputs(S_leaf, t_terminal, sigma_leaf, config=cfg)
        V_terminal = model(terminal_features).squeeze(-1)
        L_terminal = torch.mean((V_terminal - payoff) ** 2)

        # Lower boundary: V(t, S→0, σ) → 0
        S_low = torch.full_like(S_leaf, cfg.S_min)
        low_features = normalize_inputs(S_low, t_leaf, sigma_leaf, config=cfg)
        V_low = model(low_features).squeeze(-1)
        L_low = torch.mean(V_low**2)

        # Upper boundary: V(t, S→∞, σ) → S - K*exp(-r*τ)
        S_high = torch.full_like(S_leaf, cfg.S_max)
        high_features = normalize_inputs(S_high, t_leaf, sigma_leaf, config=cfg)
        tau = torch.clamp(cfg.T - t_leaf, min=0.0)
        payoff_high = S_high - cfg.K * torch.exp(-cfg.r * tau)
        V_high = model(high_features).squeeze(-1)
        L_high = torch.mean((V_high - payoff_high) ** 2)

        L_boundary = (L_terminal + L_low + L_high) / 3.0

    total = L_price + L_PDE + L_reg + boundary_weight * L_boundary
    return total, (L_price, L_PDE, L_reg, L_boundary)
