"""Training script for the physics-informed PINN model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import DATA_DIR, FIGURES_DIR, RESULTS_DIR
from .losses import compute_pde_residual, pinn_loss
from .models import PINNModel
from .preprocessing import K_REF, S_MAX, S_MIN, SIGMA_MAX, SIGMA_MIN, T_REF
from .utils import bs_price


def _build_dataloader(data: np.ndarray, batch_size: int, *, shuffle: bool) -> DataLoader:
    tensors = [torch.tensor(data[:, i], dtype=torch.float32) for i in range(4)]
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_data(
    path: Path | str = DATA_DIR / "synthetic_train.npy",
    batch_size: int = 4096,
    *,
    return_numpy: bool = False,
) -> DataLoader | Tuple[DataLoader, np.ndarray]:
    """Load the synthetic training set and wrap it in a DataLoader."""
    data = np.load(path)
    loader = _build_dataloader(data, batch_size, shuffle=True)
    if return_numpy:
        return loader, data
    return loader


def _adaptive_resample(
    model: torch.nn.Module,
    data: np.ndarray,
    *,
    n_points: int = 10_000,
    radius: float = 0.1,
    eval_samples: int = 50_000,
    device: torch.device,
) -> np.ndarray:
    """Augment the dataset near regions with the highest PDE residual."""
    model.eval()
    if len(data) == 0:
        return data

    rng = np.random.default_rng()
    sample_size = min(eval_samples, len(data))
    idx = rng.choice(len(data), size=sample_size, replace=False)
    sample = data[idx]

    S = torch.tensor(sample[:, 0], dtype=torch.float32, device=device, requires_grad=True)
    t = torch.tensor(sample[:, 1], dtype=torch.float32, device=device, requires_grad=True)
    sigma = torch.tensor(sample[:, 2], dtype=torch.float32, device=device)

    residual = compute_pde_residual(model, S, t, sigma).detach().cpu().numpy()
    top_k = np.argsort(np.abs(residual))[-min(n_points, sample_size):]
    anchors = sample[top_k]

    # Jitter around the anchors while respecting domain bounds.
    S_anchor, t_anchor, sigma_anchor = anchors[:, 0], anchors[:, 1], anchors[:, 2]
    S_new = np.clip(S_anchor * (1.0 + rng.normal(scale=radius, size=S_anchor.shape)), S_MIN, S_MAX)
    t_noise = rng.normal(scale=radius * 0.2, size=t_anchor.shape)
    t_new = np.clip(t_anchor + t_noise, 0.01, T_REF - 1e-3)
    sigma_new = np.clip(
        sigma_anchor * (1.0 + rng.normal(scale=radius, size=sigma_anchor.shape)),
        SIGMA_MIN,
        SIGMA_MAX,
    )

    V_new = bs_price(S_new, K_REF, T=T_REF, t=t_new, sigma=sigma_new)
    new_samples = np.stack([S_new, t_new, sigma_new, V_new], axis=1)
    augmented = np.concatenate([data, new_samples], axis=0)
    return augmented


def _evaluate_set(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute average loss components over a dataloader."""
    model.eval()
    total_loss = total_price = total_pde = total_reg = 0.0
    steps = 0
    for batch in loader:
        S, t, sigma, V = [x.to(device) for x in batch]
        loss, (L_price, L_PDE, L_reg) = pinn_loss(model, S, t, sigma, V)
        total_loss += loss.item()
        total_price += L_price.item()
        total_pde += L_PDE.item()
        total_reg += L_reg.item()
        steps += 1
    return {
        "loss": total_loss / steps,
        "price": total_price / steps,
        "pde": total_pde / steps,
        "reg": total_reg / steps,
    }


def _configure_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        dev = torch.device("cpu")
    return dev


def train(
    epochs: int = 50,
    lr: float = 1e-3,
    checkpoint_path: Path | str = RESULTS_DIR / "pinn_checkpoint.pt",
    *,
    batch_size: int = 4096,
    data_path: Path | str = DATA_DIR / "synthetic_train.npy",
    val_path: Path | str | None = DATA_DIR / "synthetic_val.npy",
    device: str | torch.device = "cpu",
    adaptive_sampling: bool = False,
    adaptive_every: int = 10,
    adaptive_points: int = 10_000,
    adaptive_radius: float = 0.1,
    adaptive_eval_samples: int = 50_000,
    use_warmup: bool = True,
    warmup_steps: int = 500,
    grad_clip: float | None = None,
    load_checkpoint: bool = False,
    save_checkpoint: bool = True,
    plot_losses: bool = True,
    plot_path: Path | str = FIGURES_DIR / "training_curves" / "loss_curves.png",
    log_path: Path | str = RESULTS_DIR / "training_history.json",
) -> Tuple[torch.nn.Module, List[dict[str, float]]]:
    """Train the PINN on mini-batches of synthetic Black-Scholes data."""
    device = _configure_device(device)
    model = PINNModel().to(device)

    checkpoint_path = Path(checkpoint_path)
    if load_checkpoint and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
        return model, []

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    base_lr = lr
    global_step = 0

    loader, data = load_data(data_path, batch_size=batch_size, return_numpy=True)
    val_loader: DataLoader | None = None
    if val_path is not None and Path(val_path).exists():
        val_data = np.load(val_path)
        val_loader = _build_dataloader(val_data, batch_size, shuffle=False)
    history: List[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        total_loss = total_price = total_pde = total_reg = 0.0
        steps = 0
        for batch in loader:
            S, t, sigma, V = [x.to(device) for x in batch]
            opt.zero_grad()
            loss, (L_price, L_PDE, L_reg) = pinn_loss(model, S, t, sigma, V)

            if use_warmup and warmup_steps > 0:
                lr_scale = min(1.0, (global_step + 1) / warmup_steps)
                for group in opt.param_groups:
                    group["lr"] = base_lr * lr_scale

            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            total_loss += loss.item()
            total_price += L_price.item()
            total_pde += L_PDE.item()
            total_reg += L_reg.item()
            steps += 1
            global_step += 1

        log_entry = {
            "epoch": epoch + 1,
            "loss": total_loss / steps,
            "price": total_price / steps,
            "pde": total_pde / steps,
            "reg": total_reg / steps,
            "lr": opt.param_groups[0]["lr"],
        }
        if val_loader is not None:
            val_metrics = _evaluate_set(model, val_loader, device)
            log_entry.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_price": val_metrics["price"],
                    "val_pde": val_metrics["pde"],
                    "val_reg": val_metrics["reg"],
                }
            )
        history.append(log_entry)
        print(
            f"Epoch {log_entry['epoch']:03d} | "
            f"loss={log_entry['loss']:.6f} | "
            f"price={log_entry['price']:.6f} | "
            f"pde={log_entry['pde']:.6f} | "
            f"reg={log_entry['reg']:.6f}"
        )

        if adaptive_sampling and (epoch + 1) % adaptive_every == 0:
            data = _adaptive_resample(
                model,
                data,
                n_points=adaptive_points,
                radius=adaptive_radius,
                eval_samples=adaptive_eval_samples,
                device=device,
            )
            loader = _build_dataloader(data, batch_size, shuffle=True)

    if save_checkpoint:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    if history:
        if plot_losses:
            _plot_losses(history, plot_path)
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    return model, history


def _plot_losses(history: List[dict[str, float]], plot_path: Path | str) -> None:
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["loss"] for entry in history]
    train_price = [entry["price"] for entry in history]
    train_pde = [entry["pde"] for entry in history]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, train_price, label="Train price")
    plt.plot(epochs, train_pde, label="Train PDE")

    if "val_loss" in history[0]:
        val_loss = [entry["val_loss"] for entry in history]
        val_pde = [entry["val_pde"] for entry in history]
        plt.plot(epochs, val_loss, label="Val loss", linestyle="--")
        plt.plot(epochs, val_pde, label="Val PDE", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN training curves")
    plt.legend()
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the PINN model for option pricing and Greeks."
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Mini-batch size.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_DIR / "synthetic_train.npy",
        help="Path to the training dataset (.npy).",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=DATA_DIR / "synthetic_val.npy",
        help="Optional validation dataset (.npy).",
    )
    parser.add_argument(
        "--no-val",
        dest="use_val",
        action="store_false",
        help="Disable validation evaluation each epoch.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=RESULTS_DIR / "pinn_checkpoint.pt",
        help="Where to write/read the model checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Training device ("cpu" or "cuda").',
    )

    parser.add_argument("--no-warmup", dest="use_warmup", action="store_false", help="Disable LR warmup.")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Number of warmup steps if enabled.")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping norm (None to disable).")

    parser.add_argument(
        "--adaptive-sampling",
        action="store_true",
        help="Enable adaptive sampling of high residual regions.",
    )
    parser.add_argument(
        "--adaptive-every",
        type=int,
        default=10,
        help="Epoch interval for adaptive sampling.",
    )
    parser.add_argument(
        "--adaptive-points",
        type=int,
        default=10_000,
        help="Number of anchor points to resample during adaptive sampling.",
    )
    parser.add_argument(
        "--adaptive-radius",
        type=float,
        default=0.1,
        help="Noise scale around anchors when generating new samples.",
    )
    parser.add_argument(
        "--adaptive-eval-samples",
        type=int,
        default=50_000,
        help="Evaluation sample count when ranking residuals.",
    )

    parser.add_argument(
        "--load-checkpoint",
        action="store_true",
        help="Load an existing checkpoint and skip training if found.",
    )
    parser.add_argument(
        "--no-save-checkpoint",
        dest="save_checkpoint",
        action="store_false",
        help="Do not write a checkpoint after training.",
    )

    parser.add_argument(
        "--plot-path",
        type=Path,
        default=FIGURES_DIR / "training_curves" / "loss_curves.png",
        help="Where to save the loss curve plot.",
    )
    parser.add_argument(
        "--no-plots",
        dest="plot_losses",
        action="store_false",
        help="Skip plotting loss curves.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=RESULTS_DIR / "training_history.json",
        help="Where to save per-epoch metrics (JSON).",
    )

    parser.set_defaults(use_warmup=True, save_checkpoint=True, use_val=True, plot_losses=True)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    model, history = train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        data_path=args.data_path,
        val_path=args.val_path if args.use_val else None,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        adaptive_sampling=args.adaptive_sampling,
        adaptive_every=args.adaptive_every,
        adaptive_points=args.adaptive_points,
        adaptive_radius=args.adaptive_radius,
        adaptive_eval_samples=args.adaptive_eval_samples,
        use_warmup=args.use_warmup,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        load_checkpoint=args.load_checkpoint,
        save_checkpoint=args.save_checkpoint,
        plot_losses=args.plot_losses,
        plot_path=args.plot_path,
        log_path=args.log_path,
    )

    if history:
        print(json.dumps(history[-1], indent=2))
    else:
        print("Training skipped (checkpoint loaded).")


if __name__ == "__main__":
    main()
