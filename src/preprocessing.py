"""Feature normalization helpers for PINN training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any

import json
import torch

# Reference contract parameters used throughout the project.
K_REF = 100.0
T_REF = 2.0

# Sampling ranges used during synthetic data generation (defaults).
S_MIN, S_MAX = 20.0, 200.0
T_MIN, T_MAX = 0.01, T_REF
SIGMA_MIN, SIGMA_MAX = 0.05, 0.6

# Derived ranges for transformed coordinates.
X_MIN, X_MAX = torch.log(torch.tensor(S_MIN / K_REF)), torch.log(
    torch.tensor(S_MAX / K_REF)
)
TAU_MIN = 1e-3  # τ = T - t, t ≤ T - 1e-3 during data generation
TAU_MAX = T_REF - T_MIN


@dataclass(frozen=True)
class NormalizationConfig:
    """Describe contract parameters and sampling bounds used for normalisation."""

    K: float = K_REF
    T: float = T_REF
    r: float = 0.05
    S_min: float = S_MIN
    S_max: float = S_MAX
    t_min: float = T_MIN
    t_max: float = T_MAX
    sigma_min: float = SIGMA_MIN
    sigma_max: float = SIGMA_MAX

    @property
    def tau_min(self) -> float:
        return max(1e-6, self.T - self.t_max)

    @property
    def tau_max(self) -> float:
        return max(self.tau_min, self.T - self.t_min)

    @property
    def x_min(self) -> float:
        return float(torch.log(torch.tensor(self.S_min / self.K)))

    @property
    def x_max(self) -> float:
        return float(torch.log(torch.tensor(self.S_max / self.K)))


def _to_tensor(value: float | Iterable[float], template: torch.Tensor) -> torch.Tensor:
    """Create a tensor on the same device/dtype as the template."""
    return torch.as_tensor(value, dtype=template.dtype, device=template.device)


def _scale_tensor(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Map a value into [-1, 1] given lower/upper bounds."""
    lo = _to_tensor(lower, value)
    hi = _to_tensor(upper, value)
    return 2.0 * (value - lo) / (hi - lo) - 1.0


def normalize_inputs(
    S: torch.Tensor,
    t: torch.Tensor,
    sigma: torch.Tensor,
    *,
    K: float = K_REF,
    T: float = T_REF,
    config: NormalizationConfig | None = None,
) -> torch.Tensor:
    """Return normalized PINN features (log-moneyness, time-to-maturity, volatility).

    The transformation matches the proposal: x = ln(S / K), τ = T - t, followed by
    min–max scaling to [-1, 1] using the synthetic data bounds.
    """
    cfg = config or NormalizationConfig(K=K, T=T)

    x = torch.log(S / _to_tensor(cfg.K, S))
    tau = _to_tensor(cfg.T, t) - t
    x_norm = _scale_tensor(x, cfg.x_min, cfg.x_max)
    tau_norm = _scale_tensor(tau, cfg.tau_min, cfg.tau_max)
    sigma_norm = _scale_tensor(sigma, cfg.sigma_min, cfg.sigma_max)
    return torch.stack([x_norm, tau_norm, sigma_norm], dim=-1)


def load_normalization_config(path: Path | str | None) -> NormalizationConfig:
    """Load bounds/contract parameters from a JSON metadata file."""
    if path is None:
        return NormalizationConfig()
    meta_path = Path(path)
    if meta_path.is_dir():
        meta_path = meta_path / "synthetic_meta.json"
    if not meta_path.exists():
        return NormalizationConfig()
    with open(meta_path, "r", encoding="utf-8") as fp:
        raw: dict[str, Any] = json.load(fp)
    return NormalizationConfig(
        K=raw.get("K", K_REF),
        T=raw.get("T", T_REF),
        r=raw.get("r", 0.05),
        S_min=raw.get("s_bounds", [S_MIN, S_MAX])[0],
        S_max=raw.get("s_bounds", [S_MIN, S_MAX])[1],
        t_min=raw.get("t_min", T_MIN),
        t_max=raw.get("t_max", T_MAX),
        sigma_min=raw.get("sigma_bounds", [SIGMA_MIN, SIGMA_MAX])[0],
        sigma_max=raw.get("sigma_bounds", [SIGMA_MIN, SIGMA_MAX])[1],
    )


__all__ = [
    "normalize_inputs",
    "K_REF",
    "T_REF",
    "S_MIN",
    "S_MAX",
    "SIGMA_MIN",
    "SIGMA_MAX",
    "NormalizationConfig",
    "load_normalization_config",
]
