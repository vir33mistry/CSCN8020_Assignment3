"""Greedy 20-episode evaluation and rule-based/DQN comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from dqn import DQNAgent
from dqn.experiment import evaluate_policy, set_reproducible_seeds
from g1_rl import G1ElbowTargetEnv
from test_g1_elbow_env import choose_rule_based_action


def summarize_episodes(
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    by_goal = (
        episodes.groupby("goal_angle", as_index=False)
        .agg(
            episodes=("success", "size"),
            successes=("success", "sum"),
            success_rate=("success", "mean"),
            mean_reward=("cumulative_reward", "mean"),
            mean_episode_length=("episode_length", "mean"),
            mean_final_absolute_error=(
                "final_absolute_error",
                "mean",
            ),
            mean_hold_actions=("hold_actions", "mean"),
            mean_action_changes=("action_changes", "mean"),
        )
        .sort_values("goal_angle")
    )

    total_actions = (
        episodes["decrease_actions"]
        + episodes["hold_actions"]
        + episodes["increase_actions"]
    )
    summary = {
        "episodes": int(len(episodes)),
        "successes": int(episodes["success"].sum()),
        "success_rate": float(episodes["success"].mean()),
        "mean_cumulative_reward": float(
            episodes["cumulative_reward"].mean()
        ),
        "mean_episode_length": float(
            episodes["episode_length"].mean()
        ),
        "mean_final_absolute_error": float(
            episodes["final_absolute_error"].mean()
        ),
        "hold_action_fraction": float(
            episodes["hold_actions"].sum()
            / max(int(total_actions.sum()), 1)
        ),
        "mean_action_changes": float(
            episodes["action_changes"].mean()
        ),
    }
    return by_goal, summary


def evaluate_checkpoint(
    *,
    configuration_name: str,
    checkpoint_path: Path,
    output_dir: Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    agent, checkpoint_metadata = DQNAgent.load_checkpoint(
        checkpoint_path,
        device="cpu",
    )
    env = G1ElbowTargetEnv(
        goal_angle=None,
        goal_range=(-0.8, 0.8),
    )

    try:
        rows = evaluate_policy(
            env,
            policy_name=configuration_name,
            action_function=lambda observation, _info: (
                agent.greedy_action(observation)
            ),
        )
    finally:
        env.close()

    episodes = pd.DataFrame(rows)
    by_goal, summary = summarize_episodes(episodes)
    summary.update(
        {
            "configuration": configuration_name,
            "checkpoint": str(checkpoint_path),
            "evaluation_epsilon": 0.0,
            "seed_policy": (
                "Fixed benchmark seeds 18020-18324; identical "
                "for every policy."
            ),
            "checkpoint_training_metadata": (
                checkpoint_metadata
            ),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(
        output_dir / f"{configuration_name}_episodes.csv",
        index=False,
    )
    by_goal.to_csv(
        output_dir / f"{configuration_name}_by_goal.csv",
        index=False,
    )
    (output_dir / f"{configuration_name}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return episodes, by_goal, summary


def evaluate_rule_based(
    *,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    env = G1ElbowTargetEnv(
        goal_angle=None,
        goal_range=(-0.8, 0.8),
    )

    def baseline_action(
        observation: np.ndarray,
        info: dict[str, Any],
    ) -> int:
        return choose_rule_based_action(
            observation=observation,
            controller_target=float(info["controller_target"]),
            action_increment=env.action_increment,
        )

    try:
        rows = evaluate_policy(
            env,
            policy_name="rule_based",
            action_function=baseline_action,
        )
    finally:
        env.close()

    episodes = pd.DataFrame(rows)
    by_goal, summary = summarize_episodes(episodes)
    summary.update(
        {
            "configuration": "rule_based",
            "evaluation_epsilon": 0.0,
            "sample_efficiency": (
                "No learned samples; uses explicit task knowledge."
            ),
        }
    )

    episodes.to_csv(
        output_dir / "rule_based_episodes.csv",
        index=False,
    )
    by_goal.to_csv(
        output_dir / "rule_based_by_goal.csv",
        index=False,
    )
    (output_dir / "rule_based_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return episodes, by_goal, summary


def selection_key(summary: dict[str, Any]) -> tuple[float, ...]:
    """Rank policies by consistency, then reward and control quality."""

    return (
        float(summary["success_rate"]),
        float(summary["mean_cumulative_reward"]),
        -float(summary["mean_final_absolute_error"]),
        -float(summary["mean_action_changes"]),
        -float(summary["mean_episode_length"]),
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate both DQN configurations greedily and "
            "compare the selected policy with the rule-based baseline."
        )
    )
    parser.add_argument(
        "--config-a",
        type=Path,
        default=Path("models/config_a_final.pt"),
    )
    parser.add_argument(
        "--config-b",
        type=Path,
        default=Path("models/config_b_final.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evaluation"),
    )
    parser.add_argument(
        "--selected-checkpoint",
        type=Path,
        default=Path("models/selected_dqn.pt"),
    )
    parser.add_argument("--seed", type=int, default=8020)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    set_reproducible_seeds(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_episodes: list[pd.DataFrame] = []
    all_by_goal: list[pd.DataFrame] = []
    summaries: dict[str, dict[str, Any]] = {}

    for configuration_name, checkpoint_path in [
        ("config_a", args.config_a),
        ("config_b", args.config_b),
    ]:
        episodes, by_goal, summary = evaluate_checkpoint(
            configuration_name=configuration_name,
            checkpoint_path=checkpoint_path,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        all_episodes.append(episodes)
        all_by_goal.append(
            by_goal.assign(policy=configuration_name)
        )
        summaries[configuration_name] = summary

    (
        baseline_episodes,
        baseline_by_goal,
        baseline_summary,
    ) = evaluate_rule_based(output_dir=args.output_dir)
    all_episodes.append(baseline_episodes)
    all_by_goal.append(
        baseline_by_goal.assign(policy="rule_based")
    )
    summaries["rule_based"] = baseline_summary

    selected_configuration = max(
        ("config_a", "config_b"),
        key=lambda name: selection_key(summaries[name]),
    )
    selected_source = (
        args.config_a
        if selected_configuration == "config_a"
        else args.config_b
    )
    args.selected_checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copy2(selected_source, args.selected_checkpoint)

    combined_episodes = pd.concat(
        all_episodes,
        ignore_index=True,
    )
    combined_episodes.to_csv(
        args.output_dir / "all_policy_episodes.csv",
        index=False,
    )

    combined_by_goal = pd.concat(
        all_by_goal,
        ignore_index=True,
    )
    combined_by_goal.to_csv(
        args.output_dir / "all_policies_by_goal.csv",
        index=False,
    )

    comparison_rows = []
    for policy_name, summary in summaries.items():
        comparison_rows.append(
            {
                "policy": policy_name,
                "successes_out_of_20": summary["successes"],
                "success_rate": summary["success_rate"],
                "mean_cumulative_reward": (
                    summary["mean_cumulative_reward"]
                ),
                "mean_episode_length": (
                    summary["mean_episode_length"]
                ),
                "mean_final_absolute_error": (
                    summary["mean_final_absolute_error"]
                ),
                "hold_action_fraction": (
                    summary["hold_action_fraction"]
                ),
                "mean_action_changes": (
                    summary["mean_action_changes"]
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        args.output_dir / "policy_comparison.csv",
        index=False,
    )

    selection_summary = {
        "selected_configuration": selected_configuration,
        "selected_source_checkpoint": str(selected_source),
        "selected_checkpoint": str(
            args.selected_checkpoint
        ),
        "selection_method": (
            "Lexicographic evidence: success rate, mean reward, "
            "lower final error, fewer action changes, then shorter "
            "episodes."
        ),
        "required_threshold": 0.80,
        "selected_meets_threshold": bool(
            summaries[selected_configuration]["success_rate"]
            >= 0.80
        ),
        "summaries": summaries,
    }
    (
        args.output_dir / "selection_summary.json"
    ).write_text(
        json.dumps(selection_summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== Greedy evaluation (epsilon = 0.0) ===")
    print(comparison.to_string(index=False))
    print(
        f"\nSelected configuration: {selected_configuration}"
    )
    print(
        f"Saved selected checkpoint: "
        f"{args.selected_checkpoint}"
    )


if __name__ == "__main__":
    main()
