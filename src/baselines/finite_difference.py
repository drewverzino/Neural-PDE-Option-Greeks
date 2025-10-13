"""Finite-difference baselines for approximating option Greeks."""

import numpy as np

from ..utils import bs_price


def finite_diff_greeks(S, K=100, T=2.0, t=1.0, sigma=0.2, r=0.05, eps=1e-2):
    """Approximate delta and gamma via central finite differences.

    Parameters mirror those in the Black-Scholes model. The finite step size
    `eps` perturbs the underlying price and controls the approximation bias.
    Returns an ordered pair (delta, gamma).
    """
    V0 = bs_price(S, K, T, t, sigma, r)
    V_up = bs_price(S + eps, K, T, t, sigma, r)
    V_down = bs_price(S - eps, K, T, t, sigma, r)
    delta = (V_up - V_down) / (2 * eps)
    gamma = (V_up - 2*V0 + V_down) / (eps**2)
    return delta, gamma
