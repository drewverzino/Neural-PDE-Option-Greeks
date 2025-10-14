"""Closed-form Black-Scholes pricing utilities."""

import numpy as np
from scipy.stats import norm


def bs_price(S, K, T, t, sigma, r=0.05, option_type="call"):
    """Return the Black-Scholes option price for a European call or put.

    Parameters
    ----------
    S : float or np.ndarray
        Current underlying asset price.
    K : float
        Strike price.
    T : float
        Expiration time.
    t : float or np.ndarray
        Evaluation time (0 at issuance, T at maturity).
    sigma : float or np.ndarray
        Volatility of the underlying asset.
    r : float, default 0.05
        Risk-free interest rate.
    option_type : {"call", "put"}, default "call"
        Which payoff to evaluate.
    """
    tau = T - t
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    else:
        return K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, t, sigma, r=0.05):
    """Compute the five primary Greeks for a European call option.

    Returns a dictionary with delta, gamma, theta, vega, and rho, each
    evaluated under the Black-Scholes assumptions for the supplied inputs.
    """
    tau = T - t
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(tau))
    price = bs_price(S, K, T, t, sigma, r)
    theta = -(0.5 * sigma**2 * S**2 * gamma + r * S * delta - r * price)
    vega = S * norm.pdf(d1) * np.sqrt(tau)
    rho = K * tau * np.exp(-r * tau) * norm.cdf(d2)
    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
