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
    """Estimate call-option Greeks with pathwise derivatives under Black–Scholes."""
    if rng is not None and seed is not None:
        raise ValueError("Provide either `rng` or `seed`, not both.")
    if rng is None:
        rng = np.random.default_rng(seed)

    sqrt_T = np.sqrt(T)
    Z = rng.standard_normal(N)
    exponent = (r - 0.5 * sigma**2) * T + sigma * sqrt_T * Z
    growth = np.exp(exponent)
    ST = S0 * growth
    payoff = np.maximum(ST - K, 0.0)
    indicator = (ST > K).astype(float)
    discount = np.exp(-r * T)

    price = discount * np.mean(payoff)
    delta = discount * np.mean(indicator * ST / S0)

    # Guard against T≈0 when computing theta
    theta_factor = r - 0.5 * sigma**2
    sqrt_term = np.where(sqrt_T > 0.0, 0.5 * sigma * Z / sqrt_T, 0.0)
    theta = discount * np.mean(
        -r * payoff + indicator * ST * (theta_factor + sqrt_term)
    )

    vega = discount * np.mean(indicator * ST * (sqrt_T * Z - sigma * T))

    rho = discount * np.mean(T * (indicator * ST - payoff))

    # Finite-difference gamma with common random numbers to stabilise variance.
    eps_rel = 1e-3
    eps_min = 1e-4
    eps_gamma = max(eps_rel * S0, eps_min)
    if eps_gamma >= S0:
        if S0 <= 0.0:
            raise ValueError("`S0` must be positive for gamma estimation.")
        eps_gamma = 0.5 * S0

    S_up = S0 + eps_gamma
    S_down = S0 - eps_gamma
    ST_up = S_up * growth
    ST_down = S_down * growth
    payoff_up = np.maximum(ST_up - K, 0.0)
    payoff_down = np.maximum(ST_down - K, 0.0)
    price_up = discount * np.mean(payoff_up)
    price_down = discount * np.mean(payoff_down)
    gamma = (price_up - 2.0 * price + price_down) / (eps_gamma**2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
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
    return mc_pathwise_greeks(S0, K=K, T=T, r=r, sigma=sigma, N=N, seed=seed, rng=rng)[
        "delta"
    ]
