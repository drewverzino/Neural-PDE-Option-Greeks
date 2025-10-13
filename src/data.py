"""Synthetic data generation for training and validation splits."""

from pathlib import Path
import numpy as np

from . import DATA_DIR
from .utils import bs_price


def generate_dataset(
    n_train: int = 1_000_000,
    n_val: int = 100_000,
    K: float = 100,
    r: float = 0.05,
    seed: int = 42,
    output_dir: Path = DATA_DIR,
):
    """Draw samples across Black-Scholes inputs and store priced datasets.

    Parameters
    ----------
    n_train, n_val : int
        Number of Monte-Carlo samples for the training and validation splits.
    K : float
        Strike price used for all synthetic contracts.
    r : float
        Risk-free rate for discounting.
    seed : int
        Random seed for reproducibility.
    output_dir : pathlib.Path
        Destination directory for the generated NumPy files.
    """
    np.random.seed(seed)

    def sample(n):
        """Return an array with columns [S, t, sigma, price] for n draws."""
        S = np.random.uniform(20, 200, n)
        t = np.random.uniform(0.01, 2.0, n)
        sigma = np.random.uniform(0.05, 0.6, n)
        V = bs_price(S, K, T=2.0, t=t, sigma=sigma, r=r)
        return np.stack([S, t, sigma, V], axis=1)

    train, val = sample(n_train), sample(n_val)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "synthetic_train.npy", train)
    np.save(output_dir / "synthetic_val.npy", val)
    print(
        f"Generated {len(train)} training and {len(val)} validation samples.")


if __name__ == "__main__":
    generate_dataset()
