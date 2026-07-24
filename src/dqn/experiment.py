"""Shared evaluation and reproducibility helpers for Assignment 3."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import random
from typing import Any

import gymnasium as gym
import numpy as np
import torch


BENCHMARK_GOALS: tuple[float, ...] = (-0.8, -0.4, 0.4, 0.8)
EVALUATION_EPISODES_PER_GOAL = 5

ActionFunction = Callable[[np.ndarray, dict[str, Any]], int]


@dataclass(frozen=True)
class EpisodeMetrics:
    """Metrics required for one training or evaluation episode."""

    policy: str
    goal_angle: float
    episode_within_goal: int
    seed: int
    success: bool
    cumulative_reward: float
    episode_length: int
    final_absolute_error: float
    final_angle: float
    terminated: bool
    truncated: bool
    decrease_actions: int
    hold_actions: int
    increase_actions: int
    action_changes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_reproducible_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a repeatable CPU run."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

    # Small fully-connected networks are faster and more repeatable
    # with one CPU worker than with a large thread pool.
    torch.set_num_threads(1)


def run_evaluation_episode(
    env: gym.Env,
    *,
    policy_name: str,
    action_function: ActionFunction,
    goal_angle: float,
    episode_within_goal: int,
    seed: int,
) -> EpisodeMetrics:
    """Run one greedy benchmark episode and collect common metrics."""

    observation, info = env.reset(
        seed=seed,
        options={"goal_angle": goal_angle},
    )

    cumulative_reward = 0.0
    actions: list[int] = []
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = int(action_function(observation, info))
        actions.append(action)

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        cumulative_reward += float(reward)

    action_changes = sum(
        current != previous
        for previous, current in zip(actions, actions[1:])
    )

    return EpisodeMetrics(
        policy=policy_name,
        goal_angle=float(goal_angle),
        episode_within_goal=int(episode_within_goal),
        seed=int(seed),
        success=bool(info.get("is_success", False)),
        cumulative_reward=float(cumulative_reward),
        episode_length=int(info["episode_step"]),
        final_absolute_error=float(info["absolute_error"]),
        final_angle=float(info["elbow_angle"]),
        terminated=bool(terminated),
        truncated=bool(truncated),
        decrease_actions=actions.count(0),
        hold_actions=actions.count(1),
        increase_actions=actions.count(2),
        action_changes=int(action_changes),
    )


def evaluate_policy(
    env: gym.Env,
    *,
    policy_name: str,
    action_function: ActionFunction,
    goals: Sequence[float] = BENCHMARK_GOALS,
    episodes_per_goal: int = EVALUATION_EPISODES_PER_GOAL,
    base_seed: int = 18_020,
) -> list[dict[str, Any]]:
    """Evaluate one policy on the required common 20 episodes."""

    if episodes_per_goal <= 0:
        raise ValueError("episodes_per_goal must be positive.")

    rows: list[dict[str, Any]] = []

    for goal_index, goal_angle in enumerate(goals):
        for episode_index in range(episodes_per_goal):
            seed = (
                base_seed
                + goal_index * 100
                + episode_index
            )
            metrics = run_evaluation_episode(
                env,
                policy_name=policy_name,
                action_function=action_function,
                goal_angle=float(goal_angle),
                episode_within_goal=episode_index + 1,
                seed=seed,
            )
            rows.append(metrics.to_dict())

    return rows
