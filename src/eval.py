"""Plot routines to inspect the trained PINN price surface."""

from __future__ import annotations

import argparse
from pathlib import Path

import plotly.graph_objects as go
import torch

from . import FIGURES_DIR, RESULTS_DIR
from .models import PINNModel, load_pinn_checkpoint
from .preprocessing import normalize_inputs


def _configure_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        dev = torch.device("cpu")
    return dev


def evaluate(
    model_path: Path | str = RESULTS_DIR / "pinn_checkpoint.pt",
    output_path: Path | str = FIGURES_DIR / "final_results" / "pinn_surface.html",
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
    load_pinn_checkpoint(model, torch.load(model_path, map_location=device), strict=False)

    S = torch.linspace(s_min, s_max, grid_points, device=device)
    sigma = torch.linspace(sigma_min, sigma_max, grid_points, device=device)
    Sg, Sigmag = torch.meshgrid(S, sigma, indexing="ij")
    t_flat = torch.full_like(Sg.flatten(), time_value)
    features = normalize_inputs(Sg.flatten(), t_flat, Sigmag.flatten())
    with torch.no_grad():
        preds = model(features).reshape(grid_points, grid_points).cpu().numpy()

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".html":
        output_path = output_path.with_suffix(".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    surface = go.Surface(
        x=S.cpu().numpy(),
        y=sigma.cpu().numpy(),
        z=preds.T,
        colorscale="Viridis",
    )
    fig = go.Figure(surface)
    fig.update_layout(
        title="PINN price surface",
        scene=dict(
            xaxis_title="Stock Price S",
            yaxis_title="Volatility σ",
            zaxis_title="Predicted Price",
        ),
    )
    fig.write_html(output_path)

    return {
        "surface_path": str(output_path),
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
        default=FIGURES_DIR / "final_results" / "pinn_surface.html",
        help="Destination for the rendered contour plot.",
    )
    parser.add_argument(
        "--device", default="cpu", help='Computation device ("cpu" or "cuda").'
    )
    parser.add_argument(
        "--grid-points", type=int, default=100, help="Number of grid points per axis."
    )
    parser.add_argument(
        "--s-min", type=float, default=20.0, help="Minimum stock price."
    )
    parser.add_argument(
        "--s-max", type=float, default=200.0, help="Maximum stock price."
    )
    parser.add_argument(
        "--sigma-min", type=float, default=0.05, help="Minimum volatility."
    )
    parser.add_argument(
        "--sigma-max", type=float, default=0.6, help="Maximum volatility."
    )
    parser.add_argument(
        "--time", type=float, default=1.0, help="Evaluation time (years)."
    )
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
