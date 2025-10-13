"""Plot routines to inspect the trained PINN price surface."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from . import FIGURES_DIR, RESULTS_DIR
from .models import PINNModel


def evaluate(
    model_path: Path | str = RESULTS_DIR / "pinn_checkpoint.pt",
    output_path: Path | str = FIGURES_DIR / "final_results" / "pinn_surface.png",
):
    """Load a saved checkpoint and render the implied price surface."""
    model = PINNModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    S = torch.linspace(20, 200, 100)
    sigma = torch.linspace(0.05, 0.6, 100)
    # Evaluate the model on a grid over underlying price and volatility.
    Sg, Sigmag = torch.meshgrid(S, sigma, indexing="ij")
    inputs = torch.stack([Sg.flatten(), torch.full_like(Sg.flatten(), 1.0), Sigmag.flatten()], dim=1)
    with torch.no_grad():
        preds = model(inputs).reshape(100, 100)
    plt.figure(figsize=(6, 4))
    plt.contourf(S, sigma, preds.numpy().T, levels=50, cmap="viridis")
    plt.colorbar(label="Predicted Price")
    plt.xlabel("Stock Price S")
    plt.ylabel("Volatility σ")
    plt.title("PINN Option Surface")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)


if __name__ == "__main__":
    evaluate()
