"""Finite-difference baselines for approximating option Greeks."""

import numpy as np

from ..utils import bs_price


def finite_diff_greeks(
    S,
    K=100,
    T=2.0,
    t=1.0,
    sigma=0.2,
    r=0.05,
    *,
    eps_S=1e-2,
    eps_t=1e-3,
    eps_sigma=1e-3,
    eps_r=1e-4,
):
    """Approximate Black–Scholes Greeks via finite differences.

    All inputs mirror the analytic pricing helper. Individual step sizes are
    used for each parameter to balance bias/variance per derivative.
    Returns a dictionary with estimates for delta, gamma, theta, vega, and rho.
    """
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    r = np.asarray(r, dtype=float)

    V0 = bs_price(S, K, T, t, sigma, r)

    # Stock-price derivatives (central difference)
    V_up_S = bs_price(S + eps_S, K, T, t, sigma, r)
    V_down_S = bs_price(S - eps_S, K, T, t, sigma, r)
    delta = (V_up_S - V_down_S) / (2.0 * eps_S)
    gamma = (V_up_S - 2.0 * V0 + V_down_S) / (eps_S**2)

    # Calendar-time derivative (theta)
    t_max = np.nextafter(T, 0.0)  # ensure strictly less than T
    t_min = 1e-6
    t_up = np.clip(t + eps_t, t_min, t_max)
    t_down = np.clip(t - eps_t, t_min, t_max)
    V_up_t = bs_price(S, K, T, t_up, sigma, r)
    V_down_t = bs_price(S, K, T, t_down, sigma, r)
    denom_t = np.where(t_up != t_down, t_up - t_down, np.nan)
    theta = (V_up_t - V_down_t) / denom_t

    # Volatility derivative (vega)
    sigma_min = 1e-6
    sigma_up = np.clip(sigma + eps_sigma, sigma_min, None)
    sigma_down = np.clip(sigma - eps_sigma, sigma_min, None)
    V_up_sigma = bs_price(S, K, T, t, sigma_up, r)
    V_down_sigma = bs_price(S, K, T, t, sigma_down, r)
    denom_sigma = np.where(sigma_up != sigma_down,
                           sigma_up - sigma_down, np.nan)
    vega = (V_up_sigma - V_down_sigma) / denom_sigma

    # Rate derivative (rho)
    r_up = r + eps_r
    r_down = r - eps_r
    V_up_r = bs_price(S, K, T, t, sigma, r_up)
    V_down_r = bs_price(S, K, T, t, sigma, r_down)
    denom_r = np.where(r_up != r_down, r_up - r_down, np.nan)
    rho = (V_up_r - V_down_r) / denom_r

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
    }
