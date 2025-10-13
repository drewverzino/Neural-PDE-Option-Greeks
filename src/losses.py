"""Loss terms for training physics-informed neural networks."""

import torch


def _prepare_inputs(S: torch.Tensor, t: torch.Tensor, sigma: torch.Tensor):
    """Return a concatenated input tensor with gradients enabled."""
    coords = torch.stack([S, t, sigma], dim=1)
    return coords.clone().detach().requires_grad_(True)


def compute_pde_residual(model, S, t, sigma, r=0.05):
    """Compute the Black-Scholes PDE residual for the current network."""
    coords = _prepare_inputs(S, t, sigma)
    V = model(coords).squeeze(-1)

    grads = torch.autograd.grad(
        V, coords, grad_outputs=torch.ones_like(V), create_graph=True
    )[0]
    dV_dS = grads[:, 0]
    dV_dt = grads[:, 1]

    second_grads = torch.autograd.grad(
        dV_dS, coords, grad_outputs=torch.ones_like(dV_dS), create_graph=True
    )[0]
    d2V_dS2 = second_grads[:, 0]

    S_flat = coords[:, 0]
    sigma_flat = coords[:, 2]
    pde = dV_dt + 0.5 * sigma_flat**2 * S_flat**2 * d2V_dS2 + r * S_flat * dV_dS - r * V
    return pde


def pinn_loss(model, S, t, sigma, target, r=0.05, λ=0.01):
    """Combine data-fit, PDE, and smoothness losses for PINN training.

    Returns the total loss along with each component so callers can log them.
    """
    coords = _prepare_inputs(S, t, sigma)
    pred = model(coords).squeeze(-1)
    L_price = torch.mean((pred - target)**2)
    L_PDE = torch.mean(compute_pde_residual(model, S, t, sigma, r)**2)
    grads = torch.autograd.grad(
        pred, coords, grad_outputs=torch.ones_like(pred), create_graph=True
    )[0]
    dV_dS = grads[:, 0]
    L_reg = λ * torch.mean(dV_dS**2)
    return L_price + L_PDE + L_reg, (L_price, L_PDE, L_reg)
