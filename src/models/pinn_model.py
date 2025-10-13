"""Physics-informed neural network architectures for option pricing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two-layer fully connected residual block with ReLU activations."""

    def __init__(self, dim=128):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, x):
        """Return x + f(x) to preserve gradient flow in deep stacks."""
        return F.relu(self.fc2(F.relu(self.fc1(x))) + x)


class PINNModel(nn.Module):
    """Simple PINN encoder with a residual trunk and scalar output head."""

    def __init__(self, input_dim=3, hidden_dim=128, n_layers=5):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(n_layers)])
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """Predict the option value for concatenated (S, t, sigma) inputs."""
        x = F.relu(self.input(x))
        for block in self.resblocks:
            x = block(x)
        return self.output(x)
