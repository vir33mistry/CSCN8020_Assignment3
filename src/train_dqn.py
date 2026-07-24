"""Train the two required Unitree G1 epsilon-decay configurations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from dqn import DQNAgent, DQNHyperparameters
from dqn.experiment import set_reproducible_seeds
from g1_rl import G1ElbowTargetEnv


CONFIGURATIONS = {
    "config_a": {
        "epsilon_decay": 0.995,
        "description": "Baseline - longer exploration period",
    },
    "config_b": {
        "epsilon_decay": 0.985,
        "description": "Faster transition toward exploitation",
    },
}


def train_configuration(
    *,
    configuration_name: str,
    episodes: int,
    seed: int,
    device: str,
    max_minutes: float,
    results_root: Path,
    models_root: Path,
) -> dict[str, object]:
    """Train one controlled exploration-decay configuration."""

    configuration = CONFIGURATIONS[configuration_name]
    set_reproducible_seeds(seed)

    hyperparameters = DQNHyperparameters(
        epsilon_decay=float(
            configuration["epsilon_decay"]
        )
    )
    agent = DQNAgent(
        hyperparameters=hyperparameters,
        seed=seed,
        device=device,
    )

    env = G1ElbowTargetEnv(
        goal_angle=None,
        goal_range=(-0.8, 0.8),
    )

    output_dir = results_root / configuration_name
    output_dir.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []
    start_time = time.perf_counter()
    stopped_reason = "episode_limit"

    print(
        f"\n=== Training {configuration_name} ===\n"
        f"Epsilon decay: {hyperparameters.epsilon_decay}\n"
        f"Episodes:      {episodes}\n"
        f"Device:        {device}\n"
        f"Seed:          {seed}"
    )

    try:
        for episode_number in range(1, episodes + 1):
            elapsed_minutes = (
                time.perf_counter() - start_time
            ) / 60.0
            if elapsed_minutes >= max_minutes:
                stopped_reason = "time_limit"
                print(
                    f"Stopping at {elapsed_minutes:.2f} minutes "
                    "to respect the configured time limit."
                )
                break

            episode_start = time.perf_counter()
            epsilon_used = float(agent.epsilon)
            observation, info = env.reset(
                seed=seed + episode_number - 1
            )

            cumulative_reward = 0.0
            episode_losses: list[float] = []
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.select_action(observation)
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                agent.remember(
                    state=observation,
                    action=action,
                    reward=reward,
                    next_state=next_observation,
                    terminated=terminated,
                    truncated=truncated,
                )

                loss = agent.optimize_model()
                if loss is not None:
                    episode_losses.append(loss)
                    loss_rows.append(
                        {
                            "optimization_step": (
                                agent.optimization_steps
                            ),
                            "episode": episode_number,
                            "loss": loss,
                        }
                    )

                cumulative_reward += float(reward)
                observation = next_observation

            epsilon_after_decay = agent.decay_epsilon()
            cumulative_seconds = (
                time.perf_counter() - start_time
            )
            mean_loss = (
                float(np.mean(episode_losses))
                if episode_losses
                else np.nan
            )

            episode_rows.append(
                {
                    "configuration": configuration_name,
                    "episode": episode_number,
                    "goal_angle": float(info["goal_angle"]),
                    "cumulative_reward": cumulative_reward,
                    "success": bool(
                        info.get("is_success", False)
                    ),
                    "episode_length": int(
                        info["episode_step"]
                    ),
                    "final_absolute_error": float(
                        info["absolute_error"]
                    ),
                    "epsilon_used": epsilon_used,
                    "epsilon_after_decay": (
                        epsilon_after_decay
                    ),
                    "mean_loss": mean_loss,
                    "replay_size": len(agent.replay_buffer),
                    "optimization_steps": (
                        agent.optimization_steps
                    ),
                    "episode_wall_clock_seconds": (
                        time.perf_counter() - episode_start
                    ),
                    "cumulative_wall_clock_seconds": (
                        cumulative_seconds
                    ),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )

            if (
                episode_number == 1
                or episode_number % 25 == 0
            ):
                recent = pd.DataFrame(episode_rows[-50:])
                print(
                    f"episode={episode_number:4d} | "
                    f"epsilon={agent.epsilon:.4f} | "
                    f"reward={cumulative_reward:+8.3f} | "
                    f"success50={recent['success'].mean():.1%} | "
                    f"steps={int(info['episode_step']):3d} | "
                    f"buffer={len(agent.replay_buffer):5d}"
                )
    finally:
        env.close()

    wall_clock_seconds = time.perf_counter() - start_time
    episodes_frame = pd.DataFrame(episode_rows)
    losses_frame = pd.DataFrame(
        loss_rows,
        columns=["optimization_step", "episode", "loss"],
    )

    episodes_frame.to_csv(
        output_dir / "training_metrics.csv",
        index=False,
    )
    losses_frame.to_csv(
        output_dir / "loss_metrics.csv",
        index=False,
    )

    final_20 = episodes_frame.tail(20)
    final_50 = episodes_frame.tail(50)
    summary: dict[str, object] = {
        "configuration": configuration_name,
        "description": configuration["description"],
        "seed": seed,
        "device": str(agent.device),
        "requested_episodes": episodes,
        "completed_episodes": int(len(episodes_frame)),
        "stopped_reason": stopped_reason,
        "wall_clock_seconds": wall_clock_seconds,
        "wall_clock_minutes": wall_clock_seconds / 60.0,
        "final_epsilon": float(agent.epsilon),
        "mean_reward_final_20": float(
            final_20["cumulative_reward"].mean()
        ),
        "training_success_rate_final_50": float(
            final_50["success"].mean()
        ),
        "total_optimization_steps": (
            agent.optimization_steps
        ),
        "hyperparameters": asdict(hyperparameters),
        "goal_range": [-0.8, 0.8],
        "truncation_treatment": (
            "Episodes stop on truncation, but Bellman targets "
            "bootstrap from truncated states; only true "
            "terminated states are masked."
        ),
    }

    checkpoint_path = (
        models_root / f"{configuration_name}_final.pt"
    )
    agent.save_checkpoint(
        checkpoint_path,
        metadata=summary,
    )

    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(
        f"Finished {configuration_name}: "
        f"{len(episodes_frame)} episodes in "
        f"{wall_clock_seconds / 60.0:.2f} minutes."
    )
    print(f"Checkpoint: {checkpoint_path}")

    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Configuration A (0.995 decay), "
            "Configuration B (0.985 decay), or both."
        )
    )
    parser.add_argument(
        "--config",
        choices=["config_a", "config_b", "all"],
        default="all",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=600,
        help="Episode cap for each configuration.",
    )
    parser.add_argument("--seed", type=int, default=8020)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--max-minutes-per-config",
        type=float,
        default=140.0,
        help=(
            "Safety stop per experiment. The default keeps both "
            "experiments below the assignment's five-hour limit."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("models"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.max_minutes_per_config <= 0:
        raise ValueError(
            "--max-minutes-per-config must be positive."
        )

    selected_configurations = (
        list(CONFIGURATIONS)
        if args.config == "all"
        else [args.config]
    )

    summaries = []
    for configuration_name in selected_configurations:
        summaries.append(
            train_configuration(
                configuration_name=configuration_name,
                episodes=args.episodes,
                seed=args.seed,
                device=args.device,
                max_minutes=args.max_minutes_per_config,
                results_root=args.results_root,
                models_root=args.models_root,
            )
        )

    print("\n=== Training summary ===")
    for summary in summaries:
        print(
            f"{summary['configuration']}: "
            f"episodes={summary['completed_episodes']}, "
            f"time={summary['wall_clock_minutes']:.2f} min, "
            f"epsilon={summary['final_epsilon']:.4f}, "
            f"reward20={summary['mean_reward_final_20']:.3f}, "
            f"success50="
            f"{summary['training_success_rate_final_50']:.1%}"
        )


if __name__ == "__main__":
    main()
