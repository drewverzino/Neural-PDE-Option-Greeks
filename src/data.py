"""Synthetic data generation for training, validation, and test splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import DATA_DIR
from .utils import bs_price


def generate_dataset(
    n_train: int = 1_000_000,
    n_val: int = 100_000,
    n_test: int = 100_000,
    *,
    K: float = 100.0,
    T: float = 2.0,
    r: float = 0.05,
    seed: int = 42,
    output_dir: Path = DATA_DIR,
    t_min: float = 0.01,
    t_max: float | None = None,
    s_bounds: tuple[float, float] = (20.0, 200.0),
    sigma_bounds: tuple[float, float] = (0.05, 0.6),
) -> dict[str, np.ndarray]:
    """Draw samples across Black-Scholes inputs and store priced datasets.

    Parameters
    ----------
    n_train, n_val, n_test : int
        Number of Monte-Carlo samples for the training/validation/test splits.
    K, T, r : float
        Contract strike, maturity, and risk-free rate used for pricing.
    seed : int
        Random seed for reproducibility.
    output_dir : pathlib.Path
        Destination directory for the generated NumPy files.
    t_min, t_max : float
        Minimum and maximum calendar time. If `t_max` is None we use
        `T - 1e-3` to avoid zero time-to-maturity.
    s_bounds, sigma_bounds : tuple
        (min, max) ranges for the underlying price and volatility draws.
    """
    rng = np.random.default_rng(seed)
    t_upper = T - 1e-3 if t_max is None else min(t_max, T - 1e-6)

    def sample(n: int) -> np.ndarray:
        """Return an array with columns [S, t, sigma, price] for n draws."""
        S = rng.uniform(s_bounds[0], s_bounds[1], n)
        t = rng.uniform(t_min, t_upper, n)
        sigma = rng.uniform(sigma_bounds[0], sigma_bounds[1], n)
        V = bs_price(S, K, T=T, t=t, sigma=sigma, r=r)
        return np.stack([S, t, sigma, V], axis=1)

    splits = {
        "train": sample(n_train),
        "val": sample(n_val),
        "test": sample(n_test),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in splits.items():
        np.save(output_dir / f"synthetic_{name}.npy", array)
    metadata = {
        "K": K,
        "T": T,
        "r": r,
        "t_min": t_min,
        "t_max": float(t_upper),
        "s_bounds": list(s_bounds),
        "sigma_bounds": list(sigma_bounds),
    }
    with open(output_dir / "synthetic_meta.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
    print(
        "Generated datasets:",
        ", ".join(f"{name}={len(arr)}" for name, arr in splits.items()),
    )
    return splits


def _visualize_splits(splits: dict[str, np.ndarray], output_dir: Path) -> None:
    """Create interactive sanity plots for the generated datasets."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for name, data in splits.items():
        S, t, sigma, V = data.T
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f"{name} · Stock price S",
                f"{name} · Calendar time t",
                f"{name} · Volatility σ",
                f"{name} · Price vs S",
            ),
        )
        fig.add_trace(
            go.Histogram(x=S, nbinsx=50, name="S", opacity=0.75), row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=t, nbinsx=50, name="t", opacity=0.75), row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=sigma, nbinsx=50, name="sigma", opacity=0.75), row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=S, y=V, mode="markers", name="price", opacity=0.3),
            row=2,
            col=2,
        )
        fig.update_layout(height=700, width=900, template="plotly_white")
        fig.update_xaxes(title_text="S", row=1, col=1)
        fig.update_xaxes(title_text="t", row=1, col=2)
        fig.update_xaxes(title_text="σ", row=2, col=1)
        fig.update_xaxes(title_text="S", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=2, col=2)
        fig.write_html(figures_dir / f"{name}_summary.html")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Black-Scholes datasets."
    )
    parser.add_argument(
        "--n-train", type=int, default=1_000_000, help="Number of training samples."
    )
    parser.add_argument(
        "--n-val", type=int, default=100_000, help="Number of validation samples."
    )
    parser.add_argument(
        "--n-test", type=int, default=100_000, help="Number of test samples."
    )
    parser.add_argument("--strike", type=float, default=100.0, help="Strike price K.")
    parser.add_argument(
        "--maturity", type=float, default=2.0, help="Maturity T (years)."
    )
    parser.add_argument("--rate", type=float, default=0.05, help="Risk-free rate r.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory to store .npy files.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Produce exploratory plots of the generated datasets.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    splits = generate_dataset(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        K=args.strike,
        T=args.maturity,
        r=args.rate,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    if args.visualize:
        _visualize_splits(splits, args.output_dir)


if __name__ == "__main__":
    main()
