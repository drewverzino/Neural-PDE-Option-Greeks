"""Training script for the physics-informed PINN model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import plotly.graph_objects as go
from . import DATA_DIR, FIGURES_DIR, RESULTS_DIR
from .losses import compute_pde_residual, pinn_loss
from .models import PINNModel
from .preprocessing import NormalizationConfig, load_normalization_config
from .utils import bs_price


def _build_dataloader(
    data: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int | None = None,
    pin_memory: bool = False,
) -> DataLoader:
    tensors = [torch.tensor(data[:, i], dtype=torch.float32) for i in range(4)]
    dataset = TensorDataset(*tensors)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
    }
    if num_workers is not None:
        loader_kwargs.update(
            {
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "persistent_workers": num_workers > 0,
            }
        )
    return DataLoader(dataset, **loader_kwargs)


def load_data(
    path: Path | str = DATA_DIR / "synthetic_train.npy",
    batch_size: int = 4096,
    *,
    return_numpy: bool = False,
    num_workers: int | None = None,
    pin_memory: bool = False,
) -> DataLoader | Tuple[DataLoader, np.ndarray]:
    """Load the synthetic training set and wrap it in a DataLoader."""
    data = np.load(path)
    loader = _build_dataloader(
        data,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
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
    config: NormalizationConfig,
) -> np.ndarray:
    """Augment the dataset near regions with the highest PDE residual."""
    model.eval()
    if len(data) == 0:
        return data

    rng = np.random.default_rng()
    sample_size = min(eval_samples, len(data))
    idx = rng.choice(len(data), size=sample_size, replace=False)
    sample = data[idx]

    S = torch.tensor(
        sample[:, 0], dtype=torch.float32, device=device, requires_grad=True
    )
    t = torch.tensor(
        sample[:, 1], dtype=torch.float32, device=device, requires_grad=True
    )
    sigma = torch.tensor(sample[:, 2], dtype=torch.float32, device=device)

    residual = (
        compute_pde_residual(model, S, t, sigma, r=config.r, config=config)
        .detach()
        .cpu()
        .numpy()
    )
    top_k = np.argsort(np.abs(residual))[-min(n_points, sample_size) :]
    anchors = sample[top_k]

    # Jitter around the anchors while respecting domain bounds.
    S_anchor, t_anchor, sigma_anchor = anchors[:, 0], anchors[:, 1], anchors[:, 2]
    S_new = np.clip(
        S_anchor * (1.0 + rng.normal(scale=radius, size=S_anchor.shape)),
        config.S_min,
        config.S_max,
    )
    t_noise = rng.normal(scale=radius * 0.2, size=t_anchor.shape)
    t_new = np.clip(t_anchor + t_noise, config.t_min, config.t_max)
    sigma_new = np.clip(
        sigma_anchor * (1.0 + rng.normal(scale=radius, size=sigma_anchor.shape)),
        config.sigma_min,
        config.sigma_max,
    )

    V_new = bs_price(S_new, config.K, T=config.T, t=t_new, sigma=sigma_new, r=config.r)
    new_samples = np.stack([S_new, t_new, sigma_new, V_new], axis=1)
    augmented = np.concatenate([data, new_samples], axis=0)
    return augmented


def _evaluate_set(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    lambda_reg: float,
    boundary_weight: float,
    config: NormalizationConfig,
) -> dict[str, float]:
    """Compute average loss components over a dataloader."""
    model.eval()
    total_loss = total_price = total_pde = total_reg = total_boundary = 0.0
    steps = 0
    for batch in loader:
        S, t, sigma, V = [x.to(device) for x in batch]
        loss, (L_price, L_PDE, L_reg, L_boundary) = pinn_loss(
            model,
            S,
            t,
            sigma,
            V,
            r=config.r,
            λ=lambda_reg,
            boundary_weight=boundary_weight,
            config=config,
        )
        total_loss += loss.item()
        total_price += L_price.item()
        total_pde += L_PDE.item()
        total_reg += L_reg.item()
        total_boundary += L_boundary.item()
        steps += 1
    return {
        "loss": total_loss / steps,
        "price": total_price / steps,
        "pde": total_pde / steps,
        "reg": total_reg / steps,
        "boundary": total_boundary / steps,
    }


def _configure_device(device: str | torch.device) -> torch.device:
    if isinstance(device, str):
        device_lower = device.lower()
        if device_lower == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        device = device_lower

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        dev = torch.device("cpu")
    elif dev.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            print("MPS requested but not available. Falling back to CPU.")
            dev = torch.device("cpu")
    return dev


def train(
    epochs: int = 50,
    lr: float = 5e-4,
    checkpoint_path: Path | str = RESULTS_DIR / "pinn_checkpoint.pt",
    *,
    batch_size: int = 4096,
    data_path: Path | str = DATA_DIR / "synthetic_train.npy",
    val_path: Path | str | None = DATA_DIR / "synthetic_val.npy",
    device: str | torch.device = "auto",
    adaptive_sampling: bool = False,
    adaptive_every: int = 5,
    adaptive_points: int = 10_000,
    adaptive_radius: float = 0.1,
    adaptive_eval_samples: int = 50_000,
    use_warmup: bool = True,
    warmup_steps: int = 500,
    warmup_base_lr: float = 1e-5,
    grad_clip: float | None = 1.0,
    load_checkpoint: bool = False,
    save_checkpoint: bool = True,
    plot_losses: bool = True,
    plot_path: Path | str = FIGURES_DIR / "training_curves" / "loss_curves.png",
    log_path: Path | str = RESULTS_DIR / "training_history.json",
    lambda_reg: float = 0.01,
    boundary_weight: float = 1.0,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    boundary_warmup: int = 10,
) -> Tuple[torch.nn.Module, List[dict[str, float]]]:
    """Train the PINN on mini-batches of synthetic Black-Scholes data."""
    device = _configure_device(device)
    model = PINNModel().to(device)

    if pin_memory is None:
        pin_memory = device.type == "cuda"

    checkpoint_path = Path(checkpoint_path)
    if load_checkpoint and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
        return model, []

    if use_warmup and warmup_steps > 0:
        init_lr = min(warmup_base_lr, lr)
    else:
        init_lr = lr
    opt = torch.optim.Adam(model.parameters(), lr=init_lr)
    target_lr = lr
    base_warmup_lr = min(warmup_base_lr, target_lr)
    global_step = 0

    use_cuda = device.type == "cuda"
    loader, data = load_data(
        data_path,
        batch_size=batch_size,
        return_numpy=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    config = load_normalization_config(Path(data_path).parent)
    val_loader: DataLoader | None = None
    if val_path is not None and Path(val_path).exists():
        val_data = np.load(val_path)
        val_loader = _build_dataloader(
            val_data,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    history: List[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        total_loss = total_price = total_pde = total_reg = total_boundary = 0.0
        steps = 0
        for batch in loader:
            S, t, sigma, V = [x.to(device) for x in batch]
            opt.zero_grad()
            if boundary_weight > 0.0 and boundary_warmup > 0:
                boundary_scale = min(1.0, (epoch + 1) / boundary_warmup)
            else:
                boundary_scale = 1.0
            current_boundary_weight = boundary_weight * boundary_scale
            loss, (L_price, L_PDE, L_reg, L_boundary) = pinn_loss(
                model,
                S,
                t,
                sigma,
                V,
                r=config.r,
                λ=lambda_reg,
                boundary_weight=current_boundary_weight,
                config=config,
            )

            if use_warmup and warmup_steps > 0:
                progress = min(1.0, (global_step + 1) / warmup_steps)
                current_lr = base_warmup_lr + (target_lr - base_warmup_lr) * progress
                for group in opt.param_groups:
                    group["lr"] = current_lr

            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            total_loss += loss.item()
            total_price += L_price.item()
            total_pde += L_PDE.item()
            total_reg += L_reg.item()
            total_boundary += L_boundary.item()
            steps += 1
            global_step += 1

        log_entry = {
            "epoch": epoch + 1,
            "loss": total_loss / steps,
            "price": total_price / steps,
            "pde": total_pde / steps,
            "reg": total_reg / steps,
            "boundary": total_boundary / steps,
            "lr": opt.param_groups[0]["lr"],
            "boundary_weight": current_boundary_weight,
        }
        if val_loader is not None:
            val_metrics = _evaluate_set(
                model,
                val_loader,
                device,
                lambda_reg=lambda_reg,
                boundary_weight=boundary_weight,
                config=config,
            )
            log_entry.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_price": val_metrics["price"],
                    "val_pde": val_metrics["pde"],
                    "val_reg": val_metrics["reg"],
                    "val_boundary": val_metrics.get("boundary", 0.0),
                }
            )
        history.append(log_entry)
        print(
            f"Epoch {log_entry['epoch']:03d} | "
            f"loss={log_entry['loss']:.6f} | "
            f"price={log_entry['price']:.6f} | "
            f"pde={log_entry['pde']:.6f} | "
            f"reg={log_entry['reg']:.6f} | "
            f"boundary={log_entry['boundary']:.6f} "
            f"(scaled: {current_boundary_weight * log_entry['boundary']:.6f})"
        )

        if adaptive_sampling and (epoch + 1) % adaptive_every == 0:
            data = _adaptive_resample(
                model,
                data,
                n_points=adaptive_points,
                radius=adaptive_radius,
                eval_samples=adaptive_eval_samples,
                device=device,
                config=config,
            )
            loader = _build_dataloader(
                data,
                batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

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
    train_reg = [entry["reg"] for entry in history]
    train_boundary = [entry.get("boundary", 0.0) for entry in history]

    plot_path = Path(plot_path)
    if plot_path.suffix.lower() != ".html":
        plot_path = plot_path.with_suffix(".html")
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()
    val_loss = val_price = val_pde = val_reg = val_boundary = None
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train loss"))
    fig.add_trace(go.Scatter(x=epochs, y=train_price, name="Train L_price"))
    fig.add_trace(go.Scatter(x=epochs, y=train_pde, name="Train L_PDE"))
    fig.add_trace(go.Scatter(x=epochs, y=train_reg, name="Train L_reg"))
    fig.add_trace(go.Scatter(x=epochs, y=train_boundary, name="Train L_boundary"))

    if "val_loss" in history[0]:
        val_loss = [entry["val_loss"] for entry in history]
        val_pde = [entry["val_pde"] for entry in history]
        val_price = [entry["val_price"] for entry in history]
        val_reg = [entry["val_reg"] for entry in history]
        val_boundary = [entry.get("val_boundary", 0.0) for entry in history]
        fig.add_trace(
            go.Scatter(x=epochs, y=val_loss, name="Val loss", line=dict(dash="dash"))
        )
        fig.add_trace(
            go.Scatter(
                x=epochs, y=val_price, name="Val L_price", line=dict(dash="dash")
            )
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_pde, name="Val L_PDE", line=dict(dash="dash"))
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_reg, name="Val L_reg", line=dict(dash="dash"))
        )
        fig.add_trace(
            go.Scatter(
                x=epochs, y=val_boundary, name="Val L_boundary", line=dict(dash="dash")
            )
        )

    fig.update_layout(
        title="PINN training curves",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_white",
    )
    fig.write_html(plot_path)

    if "val_loss" in history[0]:
        val_price = [entry["val_price"] for entry in history]
        val_pde = [entry["val_pde"] for entry in history]
        val_reg = [entry["val_reg"] for entry in history]

    log_path = plot_path.with_name(plot_path.stem + "_log.html")
    fig_log = go.Figure()
    eps = 1e-12
    fig_log.add_trace(
        go.Scatter(x=epochs, y=np.array(train_price) + eps, name="Train L_price")
    )
    fig_log.add_trace(
        go.Scatter(x=epochs, y=np.array(train_pde) + eps, name="Train L_PDE")
    )
    fig_log.add_trace(
        go.Scatter(x=epochs, y=np.array(train_reg) + eps, name="Train L_reg")
    )
    fig_log.add_trace(
        go.Scatter(x=epochs, y=np.array(train_boundary) + eps, name="Train L_boundary")
    )
    if "val_loss" in history[0]:
        fig_log.add_trace(
            go.Scatter(
                x=epochs,
                y=np.array(val_price) + eps,
                name="Val L_price",
                line=dict(dash="dash"),
            )
        )
        fig_log.add_trace(
            go.Scatter(
                x=epochs,
                y=np.array(val_pde) + eps,
                name="Val L_PDE",
                line=dict(dash="dash"),
            )
        )
        fig_log.add_trace(
            go.Scatter(
                x=epochs,
                y=np.array(val_reg) + eps,
                name="Val L_reg",
                line=dict(dash="dash"),
            )
        )
        fig_log.add_trace(
            go.Scatter(
                x=epochs,
                y=np.array(val_boundary) + eps,
                name="Val L_boundary",
                line=dict(dash="dash"),
            )
        )
    fig_log.update_layout(
        title="PINN component losses (log scale)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_white",
    )
    fig_log.write_html(log_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the PINN model for option pricing and Greeks."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Target learning rate after warmup.",
    )
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
        default="auto",
        help='Training device ("cpu", "cuda", "mps", or "auto").',
    )

    parser.add_argument(
        "--no-warmup",
        dest="use_warmup",
        action="store_false",
        help="Disable LR warmup.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps if enabled.",
    )
    parser.add_argument(
        "--warmup-base-lr",
        type=float,
        default=1e-5,
        help="Starting learning rate used at the beginning of warmup.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm (set <=0 to disable).",
    )

    parser.add_argument(
        "--adaptive-sampling",
        action="store_true",
        help="Enable adaptive sampling of high residual regions.",
    )
    parser.add_argument(
        "--adaptive-every",
        type=int,
        default=5,
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
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=0.01,
        help="Weight for the gradient regularization term in the loss.",
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=1.0,
        help="Weight for the terminal/boundary condition loss.",
    )
    parser.add_argument(
        "--boundary-warmup",
        type=int,
        default=10,
        help="Epochs over which to ramp the boundary weight (0 to disable).",
    )

    parser.set_defaults(
        use_warmup=True, save_checkpoint=True, use_val=True, plot_losses=True
    )
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
        warmup_base_lr=args.warmup_base_lr,
        grad_clip=args.grad_clip,
        load_checkpoint=args.load_checkpoint,
        save_checkpoint=args.save_checkpoint,
        plot_losses=args.plot_losses,
        plot_path=args.plot_path,
        log_path=args.log_path,
        lambda_reg=args.lambda_reg,
        boundary_weight=args.boundary_weight,
        boundary_warmup=args.boundary_warmup,
    )

    if history:
        print(json.dumps(history[-1], indent=2))
    else:
        print("Training skipped (checkpoint loaded).")


if __name__ == "__main__":
    main()
