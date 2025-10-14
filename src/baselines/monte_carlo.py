"""Monte-Carlo baselines for option sensitivities."""

import numpy as np


def mc_pathwise_greeks(
    S0,
    K=100,
    T=1.0,
    r=0.05,
    sigma=0.2,
    N=10_000,
    *,
    seed=None,
    rng=None,
):
    """Estimate call-option Greeks with pathwise derivatives under Black–Scholes.

    Returns a dictionary with delta, theta, vega, and rho estimates (gamma is
    omitted due to instability in second-order pathwise estimators).
    """
    if rng is not None and seed is not None:
        raise ValueError("Provide either `rng` or `seed`, not both.")
    if rng is None:
        rng = np.random.default_rng(seed)

    sqrt_T = np.sqrt(T)
    Z = rng.standard_normal(N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * sqrt_T * Z)
    payoff = np.maximum(ST - K, 0.0)
    indicator = (ST > K).astype(float)
    discount = np.exp(-r * T)

    delta = discount * np.mean(indicator * ST / S0)

    # Guard against T≈0 when computing theta
    theta_factor = r - 0.5 * sigma**2
    sqrt_term = np.where(sqrt_T > 0.0, 0.5 * sigma * Z / sqrt_T, 0.0)
    theta_tau = discount * np.mean(-r * payoff + indicator * ST * (theta_factor + sqrt_term))
    theta = -theta_tau

    vega = discount * np.mean(indicator * ST * (sqrt_T * Z - sigma * T))

    rho = discount * np.mean(T * (indicator * ST - payoff))

    return {
        "delta": float(delta),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
    }


def mc_pathwise_delta(
    S0,
    K=100,
    T=1.0,
    r=0.05,
    sigma=0.2,
    N=10_000,
    *,
    seed=None,
    rng=None,
):
    """Backward-compatible wrapper returning only the pathwise delta estimate."""
    return mc_pathwise_greeks(S0, K=K, T=T, r=r, sigma=sigma, N=N, seed=seed, rng=rng)["delta"]
