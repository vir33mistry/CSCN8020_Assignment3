"""Bounded experience replay for the student-written DQN."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch


@dataclass(frozen=True)
class TensorBatch:
    """A sampled transition batch converted to PyTorch tensors."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


@dataclass(frozen=True)
class ReplayBatch:
    """A sampled transition batch stored as NumPy arrays."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    def to_tensors(self, device: torch.device) -> TensorBatch:
        """Convert the complete batch to tensors on the requested device."""

        return TensorBatch(
            states=torch.as_tensor(
                self.states,
                dtype=torch.float32,
                device=device,
            ),
            actions=torch.as_tensor(
                self.actions,
                dtype=torch.int64,
                device=device,
            ).unsqueeze(1),
            rewards=torch.as_tensor(
                self.rewards,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1),
            next_states=torch.as_tensor(
                self.next_states,
                dtype=torch.float32,
                device=device,
            ),
            terminated=torch.as_tensor(
                self.terminated,
                dtype=torch.bool,
                device=device,
            ).unsqueeze(1),
            truncated=torch.as_tensor(
                self.truncated,
                dtype=torch.bool,
                device=device,
            ).unsqueeze(1),
        )


@dataclass(frozen=True)
class Transition:
    """One environment interaction stored by replay memory."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool


class ReplayBuffer:
    """A fixed-capacity replay buffer with seeded random sampling."""

    def __init__(
        self,
        capacity: int = 50_000,
        seed: int = 8020,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")

        self.capacity = int(capacity)
        self._memory: Deque[Transition] = deque(
            maxlen=self.capacity
        )
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._memory)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Store an independent copy of one transition."""

        state_array = np.asarray(state, dtype=np.float32)
        next_state_array = np.asarray(
            next_state,
            dtype=np.float32,
        )

        if state_array.ndim != 1:
            raise ValueError("state must be a one-dimensional vector.")
        if next_state_array.shape != state_array.shape:
            raise ValueError(
                "next_state must have the same shape as state."
            )
        if not np.isfinite(state_array).all():
            raise ValueError("state contains a non-finite value.")
        if not np.isfinite(next_state_array).all():
            raise ValueError("next_state contains a non-finite value.")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite.")

        self._memory.append(
            Transition(
                state=state_array.copy(),
                action=int(action),
                reward=float(reward),
                next_state=next_state_array.copy(),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        )

    def sample(self, batch_size: int) -> ReplayBatch:
        """Sample a mini-batch uniformly without replacement."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > len(self._memory):
            raise ValueError(
                "Cannot sample more transitions than are stored."
            )

        indices = self._rng.choice(
            len(self._memory),
            size=batch_size,
            replace=False,
        )
        transitions = [self._memory[int(index)] for index in indices]

        return ReplayBatch(
            states=np.stack(
                [transition.state for transition in transitions]
            ),
            actions=np.asarray(
                [transition.action for transition in transitions],
                dtype=np.int64,
            ),
            rewards=np.asarray(
                [transition.reward for transition in transitions],
                dtype=np.float32,
            ),
            next_states=np.stack(
                [
                    transition.next_state
                    for transition in transitions
                ]
            ),
            terminated=np.asarray(
                [
                    transition.terminated
                    for transition in transitions
                ],
                dtype=np.bool_,
            ),
            truncated=np.asarray(
                [
                    transition.truncated
                    for transition in transitions
                ],
                dtype=np.bool_,
            ),
        )
