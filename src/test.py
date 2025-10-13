"""Out-of-sample evaluation against analytic and baseline benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import torch

from . import DATA_DIR, RESULTS_DIR
from .baselines import finite_diff_greeks, mc_pathwise_delta
from .models import PINNModel
from .preprocessing import normalize_inputs
from .utils.black_scholes import bs_greeks


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def evaluate_oos(
    data_path: Path | str,
    model_path: Path | str,
    *,
    device: str | torch.device = "cpu",
    sample_size: int | None = None,
    mc_paths: int = 50_000,
    seed: int = 0,
    fig_dir: Path | None = None,
    surface_grid: int | None = None,
) -> Dict[str, float]:
    """Compare the PINN vs analytic Black–Scholes, finite differences, and Monte Carlo."""
    rng = np.random.default_rng(seed)

    data = np.load(data_path)
    if sample_size is not None and sample_size < len(data):
        idx = rng.choice(len(data), size=sample_size, replace=False)
        data = data[idx]
    elif sample_size is not None and sample_size > len(data):
        print(f"[warn] sample_size={sample_size} exceeds dataset size {len(data)}; using full dataset.")

    S_np, t_np, sigma_np, price_np = data.T
    K, T, r = 100.0, 2.0, 0.05

    device = torch.device(device)
    model = PINNModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    S = torch.tensor(S_np, dtype=torch.float32, device=device, requires_grad=True)
    t = torch.tensor(t_np, dtype=torch.float32, device=device, requires_grad=True)
    sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)

    features = normalize_inputs(S, t, sigma)
    pred_price = model(features).squeeze()

    ones = torch.ones_like(pred_price)
    delta_pred = torch.autograd.grad(pred_price, S, grad_outputs=ones, create_graph=True)[0]
    gamma_pred = torch.autograd.grad(delta_pred, S, grad_outputs=torch.ones_like(delta_pred), create_graph=False)[0]

    pred_price_np = pred_price.detach().cpu().numpy()
    delta_pred_np = delta_pred.detach().cpu().numpy()
    gamma_pred_np = gamma_pred.detach().cpu().numpy()

    analytic = bs_greeks(S_np, K, T, t_np, sigma_np, r)
    delta_true = analytic["delta"]
    gamma_true = analytic["gamma"]
    price_true = price_np

    delta_fd, gamma_fd = finite_diff_greeks(S_np, K=K, T=T, t=t_np, sigma=sigma_np, r=r)

    mc_delta = np.array(
        [
            mc_pathwise_delta(s, K=K, T=tn, r=r, sigma=vol, N=mc_paths, seed=seed + i)
            for i, (s, tn, vol) in enumerate(zip(S_np, t_np, sigma_np))
        ]
    )

    metrics = {
        "pinn_price_rmse": rmse(pred_price_np, price_true),
        "pinn_delta_mae": mae(delta_pred_np, delta_true),
        "pinn_gamma_mae": mae(gamma_pred_np, gamma_true),
        "fd_delta_mae": mae(delta_fd, delta_true),
        "fd_gamma_mae": mae(gamma_fd, gamma_true),
        "mc_delta_mae": mae(mc_delta, delta_true),
    }

    if fig_dir is not None:
        _visualize_results(
            fig_dir,
            S_np,
            sigma_np,
            pred_price_np,
            price_true,
            delta_pred_np,
            gamma_pred_np,
            delta_fd,
            gamma_fd,
            mc_delta,
            delta_true,
            gamma_true,
            surface_grid=surface_grid,
        )

    return metrics


def _visualize_results(
    fig_dir: Path,
    S: np.ndarray,
    sigma: np.ndarray,
    price_pred: np.ndarray,
    price_true: np.ndarray,
    delta_pinn: np.ndarray,
    gamma_pinn: np.ndarray,
    delta_fd: np.ndarray,
    gamma_fd: np.ndarray,
    delta_mc: np.ndarray,
    delta_true: np.ndarray,
    gamma_true: np.ndarray,
    *,
    surface_grid: int | None = None,
) -> None:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_count = min(len(S), 5000)
    if len(S) > plot_count:
        idx = np.linspace(0, len(S) - 1, plot_count, dtype=int)
    else:
        idx = np.arange(len(S))

    diag_price = np.linspace(price_true.min(), price_true.max(), 100)
    diag_delta = np.linspace(delta_true.min(), delta_true.max(), 100)
    diag_gamma = np.linspace(gamma_true.min(), gamma_true.max(), 100)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.scatter(price_true[idx], price_pred[idx], s=6, alpha=0.4, label="PINN")
    ax.plot(diag_price, diag_price, color="black", linestyle="--", linewidth=1)
    ax.set_title("Price: analytic vs PINN")
    ax.set_xlabel("Analytic price")
    ax.set_ylabel("PINN price")
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(delta_true[idx], delta_pinn[idx], s=6, alpha=0.4, label="PINN")
    ax.scatter(delta_true[idx], delta_fd[idx], s=6, alpha=0.4, label="Finite diff")
    ax.scatter(delta_true[idx], delta_mc[idx], s=6, alpha=0.2, label="Monte Carlo")
    ax.plot(diag_delta, diag_delta, color="black", linestyle="--", linewidth=1)
    ax.set_title("Delta: analytic vs predictions")
    ax.set_xlabel("Analytic Δ")
    ax.set_ylabel("Predicted Δ")
    ax.legend()

    ax = axes[1, 0]
    delta_err_pinn = delta_pinn - delta_true
    delta_err_fd = delta_fd - delta_true
    delta_err_mc = delta_mc - delta_true
    bins = 60
    ax.hist(delta_err_pinn, bins=bins, alpha=0.6, label="PINN")
    ax.hist(delta_err_fd, bins=bins, alpha=0.6, label="Finite diff")
    ax.hist(delta_err_mc, bins=bins, alpha=0.6, label="Monte Carlo")
    ax.set_title("Delta error distribution")
    ax.set_xlabel("Δ prediction error")
    ax.set_ylabel("Frequency")
    ax.legend()

    ax = axes[1, 1]
    ax.scatter(gamma_true[idx], gamma_pinn[idx], s=6, alpha=0.4, label="PINN")
    ax.scatter(gamma_true[idx], gamma_fd[idx], s=6, alpha=0.4, label="Finite diff")
    ax.plot(diag_gamma, diag_gamma, color="black", linestyle="--", linewidth=1)
    ax.set_title("Gamma: analytic vs predictions")
    ax.set_xlabel("Analytic Γ")
    ax.set_ylabel("Predicted Γ")
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "oos_summary.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(S[idx], price_true[idx] - price_pred[idx], s=6, alpha=0.5, label="Price error")
    ax.set_xlabel("Stock price S")
    ax.set_ylabel("Price error (analytic - PINN)")
    ax.set_title("Price error vs stock price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "price_error_vs_S.png", dpi=200)
    plt.close(fig)

    if surface_grid is not None and surface_grid > 1:
        _plot_surface(fig_dir, S, sigma, price_true, price_pred, surface_grid)


def _plot_surface(
    fig_dir: Path,
    S: np.ndarray,
    sigma: np.ndarray,
    price_true: np.ndarray,
    price_pred: np.ndarray,
    grid: int,
) -> None:
    grid = int(grid)
    s_bins = np.linspace(S.min(), S.max(), grid + 1)
    sigma_bins = np.linspace(sigma.min(), sigma.max(), grid + 1)
    s_centers = 0.5 * (s_bins[:-1] + s_bins[1:])
    sigma_centers = 0.5 * (sigma_bins[:-1] + sigma_bins[1:])

    def grid_average(values: np.ndarray) -> np.ndarray:
        surface = np.full((grid, grid), np.nan, dtype=float)
        counts = np.zeros((grid, grid), dtype=int)
        s_idx = np.clip(np.digitize(S, s_bins) - 1, 0, grid - 1)
        sigma_idx = np.clip(np.digitize(sigma, sigma_bins) - 1, 0, grid - 1)
        for i, (si, sj) in enumerate(zip(s_idx, sigma_idx)):
            surface[si, sj] = np.nan_to_num(surface[si, sj], nan=0.0) + values[i]
            counts[si, sj] += 1
        mask = counts > 0
        surface[mask] = surface[mask] / counts[mask]
        surface[~mask] = np.nan
        return surface

    true_surface = grid_average(price_true)
    pred_surface = grid_average(price_pred)
    error_surface = grid_average(price_pred - price_true)

    S_grid, Sigma_grid = np.meshgrid(s_centers, sigma_centers, indexing="ij")

    def save_surface(data: np.ndarray, title: str, filename: str) -> None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        filled = data.copy()
        if np.isnan(filled).all():
            return
        nan_mask = np.isnan(filled)
        if np.any(~nan_mask):
            filled[nan_mask] = np.nanmean(filled[~nan_mask])
        ax.plot_surface(
            S_grid,
            Sigma_grid,
            filled,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
        )
        ax.set_xlabel("Stock price S")
        ax.set_ylabel("Volatility σ")
        ax.set_zlabel("Price")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=200)
        plt.close(fig)

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_surface(true_surface, "Analytic price surface", "analytic_surface.png")
    save_surface(pred_surface, "PINN price surface", "pinn_surface.png")
    save_surface(error_surface, "Price error surface (PINN - analytic)", "price_error_surface.png")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Out-of-sample evaluation script.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_DIR / "synthetic_val.npy",
        help="Path to OOS dataset (.npy).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=RESULTS_DIR / "pinn_checkpoint.pt",
        help="Trained PINN checkpoint.",
    )
    parser.add_argument("--device", default="cpu", help='Evaluation device ("cpu" or "cuda").')
    parser.add_argument("--sample-size", type=int, default=None, help="Optional subsample size.")
    parser.add_argument("--mc-paths", type=int, default=50_000, help="Monte Carlo paths per evaluation.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "oos_metrics.json",
        help="Where to store the JSON metrics.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate diagnostic plots comparing models and baselines.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=RESULTS_DIR / "figures" / "oos",
        help="Directory to store visualizations (used with --visualize).",
    )
    parser.add_argument(
        "--surface-grid",
        type=int,
        default=40,
        help="Number of bins per axis for 3D surfaces (used with --visualize).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    metrics = evaluate_oos(
        data_path=args.data_path,
        model_path=args.model_path,
        device=args.device,
        sample_size=args.sample_size,
        mc_paths=args.mc_paths,
        seed=args.seed,
        fig_dir=args.fig_dir if args.visualize else None,
        surface_grid=args.surface_grid if args.visualize else None,
    )
    print(json.dumps(metrics, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
