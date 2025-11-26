"""Physics-informed neural network architectures for option pricing."""

import os
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class TanhBlock(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        # optional: mild stabilization without changing the design
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        return self.norm(x + h)  # residual + light norm


class PINNModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, n_layers=5):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [TanhBlock(hidden_dim) for _ in range(n_layers)])
        self.output = nn.Linear(hidden_dim, 1)
        # good inits for smooth nets
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        h = torch.tanh(self.input(x))
        for b in self.blocks:
            h = b(h)
        return self.output(h).squeeze(-1)


def load_pinn_checkpoint(
    model: nn.Module,
    state_dict_or_path: str | Mapping[str, Any],
    *,
    map_location: str | torch.device | None = None,
    strict: bool = False,
) -> nn.Module:
    """Load a checkpoint while mapping legacy layer names.

    Older checkpoints used ``resblocks.*``. Newer models use ``blocks.*`` with
    an added LayerNorm; we keep the old weights, allow missing norm params, and
    avoid unexpected-key errors.
    """
    if isinstance(state_dict_or_path, (str, bytes, os.PathLike)):  # type: ignore[arg-type]
        state = torch.load(state_dict_or_path, map_location=map_location)
    else:
        state = state_dict_or_path

    remapped = {}
    for k, v in state.items():
        new_k = k.replace("resblocks.", "blocks.")
        remapped[new_k] = v

    model.load_state_dict(remapped, strict=strict)
    return model


# class ResidualBlock(nn.Module):
#     """Two-layer fully connected residual block with ReLU activations."""

#     def __init__(self, dim=128):
#         super().__init__()
#         self.fc1, self.fc2 = nn.Linear(dim, dim), nn.Linear(dim, dim)

#     def forward(self, x):
#         """Return x + f(x) to preserve gradient flow in deep stacks."""
#         return F.relu(self.fc2(F.relu(self.fc1(x))) + x)


# class PINNModel(nn.Module):
#     """Simple PINN encoder with a residual trunk and scalar output head."""

#     def __init__(self, input_dim=3, hidden_dim=128, n_layers=5):
#         super().__init__()
#         self.input = nn.Linear(input_dim, hidden_dim)
#         self.resblocks = nn.ModuleList(
#             [ResidualBlock(hidden_dim) for _ in range(n_layers)])
#         self.output = nn.Linear(hidden_dim, 1)

#     def forward(self, x):
#         """Predict the option value for concatenated (S, t, sigma) inputs."""
#         x = F.relu(self.input(x))
#         for block in self.resblocks:
#             x = block(x)
#         return self.output(x)
