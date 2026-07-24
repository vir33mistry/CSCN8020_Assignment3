"""Student-written Deep Q-Network agent for Unitree G1 elbow control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from .q_network import QNetwork
from .replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class DQNHyperparameters:
    """Assignment baseline hyperparameters."""

    gamma: float = 0.95
    learning_rate: float = 0.001
    batch_size: int = 64
    replay_capacity: int = 50_000
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    target_update_steps: int = 250
    warmup_transitions: int = 500
    gradient_clip_norm: float = 10.0
    hidden_size: int = 64

    def validate(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between zero and one.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.replay_capacity < self.batch_size:
            raise ValueError(
                "replay_capacity must be at least batch_size."
            )
        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ValueError(
                "epsilon values must satisfy "
                "0 <= epsilon_min <= epsilon_start <= 1."
            )
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError(
                "epsilon_decay must be in the interval (0, 1]."
            )
        if self.target_update_steps <= 0:
            raise ValueError("target_update_steps must be positive.")
        if self.warmup_transitions < self.batch_size:
            raise ValueError(
                "warmup_transitions must be at least batch_size."
            )
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")


class DQNAgent:
    """Online/target-network DQN with replay and epsilon-greedy actions."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        observation_size: int = 4,
        action_size: int = 3,
        hyperparameters: DQNHyperparameters | None = None,
        seed: int = 8020,
        device: str | torch.device = "cpu",
    ) -> None:
        self.hyperparameters = (
            hyperparameters or DQNHyperparameters()
        )
        self.hyperparameters.validate()

        if observation_size <= 0:
            raise ValueError("observation_size must be positive.")
        if action_size <= 0:
            raise ValueError("action_size must be positive.")

        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        self.seed = int(seed)
        self.device = torch.device(device)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self._python_rng = random.Random(self.seed)

        self.online_network = QNetwork(
            observation_size=self.observation_size,
            action_size=self.action_size,
            hidden_size=self.hyperparameters.hidden_size,
        ).to(self.device)

        self.target_network = QNetwork(
            observation_size=self.observation_size,
            action_size=self.action_size,
            hidden_size=self.hyperparameters.hidden_size,
        ).to(self.device)
        self.sync_target_network()
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.hyperparameters.learning_rate,
        )
        self.loss_function = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(
            capacity=self.hyperparameters.replay_capacity,
            seed=self.seed,
        )

        self.epsilon = self.hyperparameters.epsilon_start
        self.optimization_steps = 0

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        state_array = np.asarray(state, dtype=np.float32)

        if state_array.shape != (self.observation_size,):
            raise ValueError(
                "State shape must be "
                f"({self.observation_size},), got "
                f"{state_array.shape}."
            )

        return torch.as_tensor(
            state_array,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return online-network Q-values without tracking gradients."""

        self.online_network.eval()
        with torch.no_grad():
            values = self.online_network(
                self._state_tensor(state)
            )
        self.online_network.train()
        return values.squeeze(0).cpu().numpy()

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float | None = None,
    ) -> int:
        """Choose a random or greedy action using epsilon-greedy policy."""

        effective_epsilon = (
            self.epsilon if epsilon is None else float(epsilon)
        )

        if not 0.0 <= effective_epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one.")

        if self._python_rng.random() < effective_epsilon:
            return self._python_rng.randrange(self.action_size)

        q_values = self.q_values(state)
        return int(np.argmax(q_values))

    def greedy_action(self, state: np.ndarray) -> int:
        """Select a deterministic greedy action for evaluation."""

        return self.select_action(state, epsilon=0.0)

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.replay_buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            truncated=truncated,
        )

    def ready_to_optimize(self) -> bool:
        minimum_size = max(
            self.hyperparameters.batch_size,
            self.hyperparameters.warmup_transitions,
        )
        return len(self.replay_buffer) >= minimum_size

    def optimize_model(self) -> float | None:
        """Apply one Bellman update and return the scalar loss."""

        if not self.ready_to_optimize():
            return None

        batch = self.replay_buffer.sample(
            self.hyperparameters.batch_size
        ).to_tensors(self.device)

        selected_q_values = self.online_network(
            batch.states
        ).gather(1, batch.actions)

        with torch.no_grad():
            next_q_values = self.target_network(
                batch.next_states
            ).max(dim=1, keepdim=True).values

            # A true task termination stops Bellman bootstrapping.
            # A time-limit truncation does not: the final physical
            # state is still a valid non-terminal state.
            bootstrap_mask = (~batch.terminated).float()

            bellman_targets = (
                batch.rewards
                + self.hyperparameters.gamma
                * bootstrap_mask
                * next_q_values
            )

        loss = self.loss_function(
            selected_q_values,
            bellman_targets,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=self.hyperparameters.gradient_clip_norm,
        )
        self.optimizer.step()

        self.optimization_steps += 1

        if (
            self.optimization_steps
            % self.hyperparameters.target_update_steps
            == 0
        ):
            self.sync_target_network()

        return float(loss.detach().cpu().item())

    def sync_target_network(self) -> None:
        """Copy all online-network weights to the target network."""

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    def decay_epsilon(self) -> float:
        """Decay epsilon once after an episode, respecting epsilon_min."""

        self.epsilon = max(
            self.hyperparameters.epsilon_min,
            self.epsilon * self.hyperparameters.epsilon_decay,
        )
        return self.epsilon

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save networks, optimizer, hyperparameters, and run metadata."""

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "checkpoint_version": self.CHECKPOINT_VERSION,
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "seed": self.seed,
                "device_used_for_training": str(self.device),
                "hyperparameters": asdict(self.hyperparameters),
                "online_network_state_dict": (
                    self.online_network.state_dict()
                ),
                "target_network_state_dict": (
                    self.target_network.state_dict()
                ),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "optimization_steps": self.optimization_steps,
                "metadata": metadata or {},
            },
            checkpoint_path,
        )

        return checkpoint_path

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        load_optimizer: bool = False,
    ) -> tuple["DQNAgent", dict[str, Any]]:
        """Load a trusted assignment checkpoint on CPU or another device."""

        checkpoint_path = Path(path)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint was not found: {checkpoint_path}"
            )

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=torch.device(device),
                weights_only=False,
            )
        except TypeError:
            # Compatibility with PyTorch releases that predate
            # the weights_only keyword.
            checkpoint = torch.load(
                checkpoint_path,
                map_location=torch.device(device),
            )

        required_keys = {
            "observation_size",
            "action_size",
            "seed",
            "hyperparameters",
            "online_network_state_dict",
            "target_network_state_dict",
        }
        missing_keys = required_keys - checkpoint.keys()
        if missing_keys:
            raise ValueError(
                "Checkpoint is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        hyperparameters = DQNHyperparameters(
            **checkpoint["hyperparameters"]
        )
        agent = cls(
            observation_size=int(checkpoint["observation_size"]),
            action_size=int(checkpoint["action_size"]),
            hyperparameters=hyperparameters,
            seed=int(checkpoint["seed"]),
            device=device,
        )

        agent.online_network.load_state_dict(
            checkpoint["online_network_state_dict"]
        )
        agent.target_network.load_state_dict(
            checkpoint["target_network_state_dict"]
        )

        if (
            load_optimizer
            and "optimizer_state_dict" in checkpoint
        ):
            agent.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        agent.epsilon = float(
            checkpoint.get(
                "epsilon",
                hyperparameters.epsilon_min,
            )
        )
        agent.optimization_steps = int(
            checkpoint.get("optimization_steps", 0)
        )
        agent.online_network.eval()
        agent.target_network.eval()

        return agent, dict(checkpoint.get("metadata", {}))
