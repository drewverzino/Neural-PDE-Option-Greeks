"""Training script for the physics-informed PINN model."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import DATA_DIR, RESULTS_DIR
from .losses import pinn_loss
from .models import PINNModel


def load_data(path: Path | str = DATA_DIR / "synthetic_train.npy", batch_size: int = 4096):
    """Load the synthetic training set and wrap it in a DataLoader."""
    data = np.load(path)
    S, t, sigma, V = [torch.tensor(data[:, i], dtype=torch.float32) for i in range(4)]
    dataset = TensorDataset(S, t, sigma, V)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(epochs: int = 50, lr: float = 1e-3, checkpoint_path: Path | str = RESULTS_DIR / "pinn_checkpoint.pt"):
    """Train the PINN on mini-batches of synthetic Black-Scholes data."""
    model = PINNModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = load_data()
    for epoch in range(epochs):
        total_loss = 0.0
        for S, t, sigma, V in loader:
            opt.zero_grad()
            loss, (_Lp, _Lpde, _Lr) = pinn_loss(model, S, t, sigma, V)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        # Report the average loss so users can monitor convergence.
        print(f"Epoch {epoch + 1:03d}: Loss={total_loss / len(loader):.6f}")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    train()
