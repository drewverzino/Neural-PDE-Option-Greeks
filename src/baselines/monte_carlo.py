"""Monte-Carlo baselines for option sensitivities."""

import numpy as np


def mc_pathwise_delta(
    S0,
    K=100,
    T=1,
    r=0.05,
    sigma=0.2,
    N=10_000,
    *,
    seed=None,
    rng=None,
):
    """Estimate the call delta with the pathwise derivative estimator.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    K : float
        Strike price.
    T : float
        Time horizon (years).
    r : float
        Risk-free rate.
    sigma : float
        Volatility of the underlying asset.
    N : int
        Number of Monte Carlo samples.
    seed : int, optional
        Seed forwarded to NumPy's default RNG (ignored if rng is provided).
    rng : numpy.random.Generator, optional
        Custom random generator for reproducibility and batched sampling.
    """
    if rng is not None and seed is not None:
        raise ValueError("Provide either `rng` or `seed`, not both.")
    if rng is None:
        rng = np.random.default_rng(seed)
    Z = rng.standard_normal(N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    dS = ST / S0
    delta = np.exp(-r * T) * np.mean(dS * (ST > K))
    return float(delta)
