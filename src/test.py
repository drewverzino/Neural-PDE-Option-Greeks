"""Out-of-sample evaluation against analytic and baseline benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from . import DATA_DIR, RESULTS_DIR
from .baselines import finite_diff_greeks, mc_pathwise_greeks
from .models import PINNModel
from .preprocessing import (
    NormalizationConfig,
    load_normalization_config,
    normalize_inputs,
)
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
        print(
            f"[warn] sample_size={sample_size} exceeds dataset size {len(data)}; using full dataset."
        )

    S_np, t_np, sigma_np, price_np = data.T
    config = load_normalization_config(Path(data_path).parent)
    K, T, r = config.K, config.T, config.r

    device = torch.device(device)
    model = PINNModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    S = torch.tensor(S_np, dtype=torch.float32, device=device)
    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)

    features = normalize_inputs(S, t, sigma, config=config)
    features = features.detach().clone().requires_grad_(True)
    pred_price = model(features).squeeze()

    ones = torch.ones_like(pred_price)
    grad_feats = torch.autograd.grad(
        pred_price,
        features,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_x_norm = grad_feats[:, 0]
    grad_tau_norm = grad_feats[:, 1]
    grad_sigma_norm = grad_feats[:, 2]

    # ✅ FIXED: Compute second derivative w.r.t. features
    d2_feats = torch.autograd.grad(
        grad_x_norm,
        features,
        grad_outputs=torch.ones_like(grad_x_norm),
        create_graph=True,
        retain_graph=True,
    )[0]
    d2V_dx_norm2 = d2_feats[:, 0]

    x_range = max(config.x_max - config.x_min, 1e-6)
    tau_range = max(config.tau_range, 1e-6)
    sigma_range = max(config.sigma_range, 1e-6)

    dx_norm_dx = 2.0 / x_range
    dtau_norm_dtau = 2.0 / tau_range
    dsigma_norm_dsigma = 2.0 / sigma_range

    dV_dx = grad_x_norm * dx_norm_dx
    delta_pred = dV_dx / S

    d2V_dx2 = d2V_dx_norm2 * (dx_norm_dx**2)
    gamma_pred = d2V_dx2 * (1.0 / (S**2)) + dV_dx * (-1.0 / (S**2))

    theta_pred = -(grad_tau_norm * dtau_norm_dtau)
    vega_pred = grad_sigma_norm * dsigma_norm_dsigma

    pred_price_np = pred_price.detach().cpu().numpy()
    delta_pred_np = delta_pred.detach().cpu().numpy()
    gamma_pred_np = gamma_pred.detach().cpu().numpy()
    theta_pred_np = theta_pred.detach().cpu().numpy()
    vega_pred_np = vega_pred.detach().cpu().numpy()

    analytic = bs_greeks(S_np, K, T, t_np, sigma_np, r)
    delta_true = analytic["delta"]
    gamma_true = analytic["gamma"]
    theta_true = analytic["theta"]
    vega_true = analytic["vega"]
    rho_true = analytic["rho"]
    price_true = price_np

    fd_greeks = finite_diff_greeks(S_np, K=K, T=T, t=t_np, sigma=sigma_np, r=r)
    delta_fd = fd_greeks["delta"]
    gamma_fd = fd_greeks["gamma"]
    theta_fd = fd_greeks["theta"]
    vega_fd = fd_greeks["vega"]
    rho_fd = fd_greeks["rho"]

    mc_delta: list[float] = []
    mc_theta: list[float] = []
    mc_vega: list[float] = []
    mc_rho: list[float] = []
    for i, (s, tn, vol) in enumerate(zip(S_np, t_np, sigma_np)):
        tau = max(1e-6, T - tn)
        mc_estimates = mc_pathwise_greeks(
            s, K=K, T=tau, r=r, sigma=vol, N=mc_paths, seed=seed + i
        )
        mc_delta.append(mc_estimates["delta"])
        mc_theta.append(mc_estimates["theta"])
        mc_vega.append(mc_estimates["vega"])
        mc_rho.append(mc_estimates["rho"])

    mc_delta = np.asarray(mc_delta)
    mc_theta = np.asarray(mc_theta)
    mc_vega = np.asarray(mc_vega)
    mc_rho = np.asarray(mc_rho)

    tau_np = np.clip(T - t_np, 1e-6, None)
    # ρ = τ (S Δ - V) for European calls
    rho_pred_np = tau_np * (S_np * delta_pred_np - pred_price_np)

    metrics = {
        "pinn_price_rmse": rmse(pred_price_np, price_true),
        "pinn_delta_mae": mae(delta_pred_np, delta_true),
        "pinn_gamma_mae": mae(gamma_pred_np, gamma_true),
        "pinn_theta_mae": mae(theta_pred_np, theta_true),
        "pinn_vega_mae": mae(vega_pred_np, vega_true),
        "pinn_rho_mae": mae(rho_pred_np, rho_true),
        "fd_delta_mae": mae(delta_fd, delta_true),
        "fd_gamma_mae": mae(gamma_fd, gamma_true),
        "fd_theta_mae": mae(theta_fd, theta_true),
        "fd_vega_mae": mae(vega_fd, vega_true),
        "fd_rho_mae": mae(rho_fd, rho_true),
        "mc_delta_mae": mae(mc_delta, delta_true),
        "mc_theta_mae": mae(mc_theta, theta_true),
        "mc_vega_mae": mae(mc_vega, vega_true),
        "mc_rho_mae": mae(mc_rho, rho_true),
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
            theta_pred_np,
            vega_pred_np,
            rho_pred_np,
            delta_fd,
            gamma_fd,
            theta_fd,
            vega_fd,
            rho_fd,
            mc_delta,
            mc_theta,
            mc_vega,
            mc_rho,
            delta_true,
            gamma_true,
            theta_true,
            vega_true,
            rho_true,
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
    theta_pinn: np.ndarray,
    vega_pinn: np.ndarray,
    rho_pinn: np.ndarray,
    delta_fd: np.ndarray,
    gamma_fd: np.ndarray,
    theta_fd: np.ndarray,
    vega_fd: np.ndarray,
    rho_fd: np.ndarray,
    delta_mc: np.ndarray,
    theta_mc: np.ndarray,
    vega_mc: np.ndarray,
    rho_mc: np.ndarray,
    delta_true: np.ndarray,
    gamma_true: np.ndarray,
    theta_true: np.ndarray,
    vega_true: np.ndarray,
    rho_true: np.ndarray,
    *,
    surface_grid: int | None = None,
) -> None:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_count = min(len(S), 5000)
    idx = (
        np.linspace(0, len(S) - 1, plot_count, dtype=int)
        if len(S) > plot_count
        else np.arange(len(S))
    )

    diag_price = np.linspace(price_true.min(), price_true.max(), 100)
    diag_delta = np.linspace(delta_true.min(), delta_true.max(), 100)
    diag_gamma = np.linspace(gamma_true.min(), gamma_true.max(), 100)

    summary_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Price: analytic vs PINN",
            "Delta: analytic vs predictions",
            "Delta error distribution",
            "Gamma: analytic vs predictions",
        ),
    )
    summary_fig.add_trace(
        go.Scatter(x=price_true[idx], y=price_pred[idx], mode="markers", name="PINN"),
        row=1,
        col=1,
    )
    summary_fig.add_trace(
        go.Scatter(
            x=diag_price,
            y=diag_price,
            mode="lines",
            name="diag",
            line=dict(dash="dash"),
        ),
        row=1,
        col=1,
    )

    summary_fig.add_trace(
        go.Scatter(x=delta_true[idx], y=delta_pinn[idx], mode="markers", name="PINN"),
        1,
        2,
    )
    summary_fig.add_trace(
        go.Scatter(
            x=delta_true[idx], y=delta_fd[idx], mode="markers", name="Finite diff"
        ),
        1,
        2,
    )
    summary_fig.add_trace(
        go.Scatter(
            x=delta_true[idx], y=delta_mc[idx], mode="markers", name="Monte Carlo"
        ),
        1,
        2,
    )
    summary_fig.add_trace(
        go.Scatter(
            x=diag_delta,
            y=diag_delta,
            mode="lines",
            name="diag",
            line=dict(dash="dash"),
        ),
        1,
        2,
    )

    summary_fig.add_trace(
        go.Histogram(x=delta_pinn - delta_true, name="PINN", opacity=0.6), 2, 1
    )
    summary_fig.add_trace(
        go.Histogram(x=delta_fd - delta_true, name="Finite diff", opacity=0.6), 2, 1
    )
    summary_fig.add_trace(
        go.Histogram(x=delta_mc - delta_true, name="Monte Carlo", opacity=0.6), 2, 1
    )
    summary_fig.update_yaxes(title_text="Count", row=2, col=1)
    summary_fig.update_xaxes(title_text="Δ prediction error", row=2, col=1)

    summary_fig.add_trace(
        go.Scatter(x=gamma_true[idx], y=gamma_pinn[idx], mode="markers", name="PINN"),
        2,
        2,
    )
    summary_fig.add_trace(
        go.Scatter(
            x=gamma_true[idx], y=gamma_fd[idx], mode="markers", name="Finite diff"
        ),
        2,
        2,
    )
    summary_fig.add_trace(
        go.Scatter(
            x=diag_gamma,
            y=diag_gamma,
            mode="lines",
            name="diag",
            line=dict(dash="dash"),
        ),
        2,
        2,
    )

    summary_fig.update_xaxes(title_text="Analytic price", row=1, col=1)
    summary_fig.update_yaxes(title_text="PINN price", row=1, col=1)
    summary_fig.update_xaxes(title_text="Analytic Δ", row=1, col=2)
    summary_fig.update_yaxes(title_text="Predicted Δ", row=1, col=2)
    summary_fig.update_xaxes(title_text="Analytic Γ", row=2, col=2)
    summary_fig.update_yaxes(title_text="Predicted Γ", row=2, col=2)
    summary_fig.update_layout(height=800, width=1000, template="plotly_white")
    summary_fig.write_html(fig_dir / "oos_summary.html")

    error_fig = go.Figure()
    error_fig.add_trace(
        go.Scatter(
            x=S[idx],
            y=price_true[idx] - price_pred[idx],
            mode="markers",
            name="Price error",
        )
    )
    error_fig.update_layout(
        title="Price error vs stock price",
        xaxis_title="Stock price S",
        yaxis_title="Price error (analytic - PINN)",
        template="plotly_white",
    )
    error_fig.write_html(fig_dir / "price_error_vs_S.html")

    if surface_grid is not None and surface_grid > 1:
        components = [
            ("price", price_true, price_pred),
            ("delta", delta_true, delta_pinn),
            ("gamma", gamma_true, gamma_pinn),
            ("theta", theta_true, theta_pinn),
            ("vega", vega_true, vega_pinn),
            ("rho", rho_true, rho_pinn),
        ]
        _plot_component_surfaces(fig_dir, S, sigma, components, surface_grid)


def _plot_component_surfaces(
    fig_dir: Path,
    S: np.ndarray,
    sigma: np.ndarray,
    components: list[tuple[str, np.ndarray, np.ndarray]],
    grid: int,
) -> None:
    """Render 3D surfaces for analytic vs. PINN components (price and Greeks)."""

    grid = int(grid)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    s_bins = np.linspace(S.min(), S.max(), grid + 1)
    sigma_bins = np.linspace(sigma.min(), sigma.max(), grid + 1)
    s_centers = 0.5 * (s_bins[:-1] + s_bins[1:])
    sigma_centers = 0.5 * (sigma_bins[:-1] + sigma_bins[1:])
    S_grid, Sigma_grid = np.meshgrid(s_centers, sigma_centers, indexing="ij")

    s_idx = np.clip(np.digitize(S, s_bins) - 1, 0, grid - 1)
    sigma_idx = np.clip(np.digitize(sigma, sigma_bins) - 1, 0, grid - 1)

    def grid_average(values: np.ndarray) -> np.ndarray:
        surface = np.full((grid, grid), np.nan, dtype=float)
        counts = np.zeros((grid, grid), dtype=int)
        for idx_i, idx_j, val in zip(s_idx, sigma_idx, values):
            if np.isnan(surface[idx_i, idx_j]):
                surface[idx_i, idx_j] = 0.0
            surface[idx_i, idx_j] += val
            counts[idx_i, idx_j] += 1
        mask = counts > 0
        surface[mask] /= counts[mask]
        surface[~mask] = np.nan
        return surface

    def save_surface(data: np.ndarray, title: str, filename: str, zlabel: str) -> None:
        if np.isnan(data).all():
            return
        filled = data.copy()
        nan_mask = np.isnan(filled)
        if np.any(~nan_mask):
            filled[nan_mask] = np.nanmean(filled[~nan_mask])
        fig = go.Figure(
            data=[go.Surface(x=S_grid, y=Sigma_grid, z=filled, colorscale="Viridis")]
        )
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Stock price S",
                yaxis_title="Volatility σ",
                zaxis_title=zlabel,
            ),
        )
        fig.write_html(fig_dir / filename)

    for name, true_vals, pred_vals in components:
        true_surface = grid_average(true_vals)
        pred_surface = grid_average(pred_vals)
        error_surface = grid_average(pred_vals - true_vals)
        capital = name.capitalize()
        save_surface(
            true_surface,
            f"Analytic {capital} surface",
            f"analytic_{name}_surface.html",
            capital,
        )
        save_surface(
            pred_surface,
            f"PINN {capital} surface",
            f"pinn_{name}_surface.html",
            capital,
        )
        save_surface(
            error_surface,
            f"{capital} error surface (PINN - analytic)",
            f"{name}_error_surface.html",
            capital,
        )


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
    parser.add_argument(
        "--device", default="cpu", help='Evaluation device ("cpu" or "cuda").'
    )
    parser.add_argument(
        "--sample-size", type=int, default=None, help="Optional subsample size."
    )
    parser.add_argument(
        "--mc-paths", type=int, default=50_000, help="Monte Carlo paths per evaluation."
    )
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
