"""PyTorch Q-network used by the Unitree G1 DQN agent."""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """Map the four-value observation to one Q-value per action."""

    def __init__(
        self,
        observation_size: int = 4,
        action_size: int = 3,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()

        if observation_size <= 0:
            raise ValueError("observation_size must be positive.")
        if action_size <= 0:
            raise ValueError("action_size must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")

        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Apply a reproducible initialization suited to ReLU layers."""

        linear_layers = [
            module
            for module in self.network
            if isinstance(module, nn.Linear)
        ]

        for layer in linear_layers[:-1]:
            nn.init.kaiming_uniform_(
                layer.weight,
                nonlinearity="relu",
            )
            nn.init.zeros_(layer.bias)

        # A smaller final-layer initialization avoids very large
        # untrained Q-values while leaving the outputs unconstrained.
        nn.init.uniform_(
            linear_layers[-1].weight,
            a=-0.03,
            b=0.03,
        )
        nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return unnormalized action values; DQN does not use softmax."""

        return self.network(observations)
