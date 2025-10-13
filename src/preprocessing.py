"""Feature normalization helpers for PINN training."""

from __future__ import annotations

from typing import Iterable

import torch

# Reference contract parameters used throughout the project.
K_REF = 100.0
T_REF = 2.0

# Sampling ranges used during synthetic data generation.
S_MIN, S_MAX = 20.0, 200.0
T_MIN, T_MAX = 0.0, T_REF
SIGMA_MIN, SIGMA_MAX = 0.05, 0.6

# Derived ranges for transformed coordinates.
X_MIN, X_MAX = torch.log(torch.tensor(S_MIN / K_REF)), torch.log(torch.tensor(S_MAX / K_REF))
TAU_MIN, TAU_MAX = T_MIN, T_REF - 0.01  # training samples obey t ∈ [0.01, T]


def _to_tensor(value: float | Iterable[float], template: torch.Tensor) -> torch.Tensor:
    """Create a tensor on the same device/dtype as the template."""
    return torch.as_tensor(value, dtype=template.dtype, device=template.device)


def _scale_tensor(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Map a value into [-1, 1] given lower/upper bounds."""
    lo = _to_tensor(lower, value)
    hi = _to_tensor(upper, value)
    return 2.0 * (value - lo) / (hi - lo) - 1.0


def normalize_inputs(
    S: torch.Tensor,
    t: torch.Tensor,
    sigma: torch.Tensor,
    *,
    K: float = K_REF,
    T: float = T_REF,
) -> torch.Tensor:
    """Return normalized PINN features (log-moneyness, time-to-maturity, volatility).

    The transformation matches the proposal: x = ln(S / K), τ = T - t, followed by
    min–max scaling to [-1, 1] using the synthetic data bounds.
    """
    x = torch.log(S / _to_tensor(K, S))
    tau = _to_tensor(T, t) - t
    x_norm = _scale_tensor(x, float(X_MIN), float(X_MAX))
    tau_norm = _scale_tensor(tau, TAU_MIN, TAU_MAX)
    sigma_norm = _scale_tensor(sigma, SIGMA_MIN, SIGMA_MAX)
    return torch.stack([x_norm, tau_norm, sigma_norm], dim=-1)


__all__ = [
    "normalize_inputs",
    "K_REF",
    "T_REF",
    "S_MIN",
    "S_MAX",
    "SIGMA_MIN",
    "SIGMA_MAX",
]
