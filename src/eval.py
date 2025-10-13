"""Plot routines to inspect the trained PINN price surface."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import torch

from . import FIGURES_DIR, RESULTS_DIR
from .models import PINNModel
from .preprocessing import normalize_inputs


def _configure_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        dev = torch.device("cpu")
    return dev


def evaluate(
    model_path: Path | str = RESULTS_DIR / "pinn_checkpoint.pt",
    output_path: Path | str = FIGURES_DIR / "final_results" / "pinn_surface.png",
    *,
    device: str | torch.device = "cpu",
    grid_points: int = 100,
    s_min: float = 20.0,
    s_max: float = 200.0,
    sigma_min: float = 0.05,
    sigma_max: float = 0.6,
    time_value: float = 1.0,
) -> dict[str, str | float]:
    """Load a saved checkpoint and render the implied price surface."""
    device = _configure_device(device)
    model = PINNModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    S = torch.linspace(s_min, s_max, grid_points, device=device)
    sigma = torch.linspace(sigma_min, sigma_max, grid_points, device=device)
    Sg, Sigmag = torch.meshgrid(S, sigma, indexing="ij")
    t_flat = torch.full_like(Sg.flatten(), time_value)
    features = normalize_inputs(Sg.flatten(), t_flat, Sigmag.flatten())
    with torch.no_grad():
        preds = model(features).reshape(grid_points, grid_points).cpu().numpy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 2D contour plot
    plt.figure(figsize=(6, 4))
    plt.contourf(S.cpu().numpy(), sigma.cpu().numpy(), preds.T, levels=50, cmap="viridis")
    plt.colorbar(label="Predicted Price")
    plt.xlabel("Stock Price S")
    plt.ylabel("Volatility σ")
    plt.title("PINN Option Surface (Contour)")
    contour_path = output_path.with_name(output_path.stem + "_contour.png")
    plt.savefig(contour_path, dpi=300)

    # 3D surface plot
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        Sg.cpu().numpy(),
        Sigmag.cpu().numpy(),
        preds,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
    )
    ax.set_xlabel("Stock Price S")
    ax.set_ylabel("Volatility σ")
    ax.set_zlabel("Predicted Price")
    ax.set_title("PINN Option Surface (3D)")
    surface_path = output_path.with_name(output_path.stem + "_3d.png")
    plt.savefig(surface_path, dpi=300)

    return {
        "contour_path": str(contour_path),
        "surface_path": str(surface_path),
        "grid_points": grid_points,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the PINN price surface.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=RESULTS_DIR / "pinn_checkpoint.pt",
        help="Checkpoint to load.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=FIGURES_DIR / "final_results" / "pinn_surface.png",
        help="Destination for the rendered contour plot.",
    )
    parser.add_argument("--device", default="cpu", help='Computation device ("cpu" or "cuda").')
    parser.add_argument("--grid-points", type=int, default=100, help="Number of grid points per axis.")
    parser.add_argument("--s-min", type=float, default=20.0, help="Minimum stock price.")
    parser.add_argument("--s-max", type=float, default=200.0, help="Maximum stock price.")
    parser.add_argument("--sigma-min", type=float, default=0.05, help="Minimum volatility.")
    parser.add_argument("--sigma-max", type=float, default=0.6, help="Maximum volatility.")
    parser.add_argument("--time", type=float, default=1.0, help="Evaluation time (years).")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    evaluate(
        model_path=args.model_path,
        output_path=args.output_path,
        device=args.device,
        grid_points=args.grid_points,
        s_min=args.s_min,
        s_max=args.s_max,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        time_value=args.time,
    )


if __name__ == "__main__":
    main()
