"""Validate the approved environment and record the rule-based baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gymnasium.utils.env_checker import check_env
import numpy as np
import pandas as pd

from dqn.experiment import evaluate_policy, set_reproducible_seeds
from g1_rl import G1ElbowTargetEnv
from test_g1_elbow_env import choose_rule_based_action


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the environment checker and required "
            "20-episode rule-based baseline."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/baseline"),
    )
    parser.add_argument("--seed", type=int, default=8020)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducible_seeds(args.seed)

    env = G1ElbowTargetEnv(
        goal_angle=None,
        goal_range=(-0.8, 0.8),
    )

    try:
        print("Running Gymnasium environment checker...")
        check_env(env, skip_render_check=True)
        print("Environment checker passed.")

        def baseline_action(
            observation: np.ndarray,
            info: dict[str, object],
        ) -> int:
            return choose_rule_based_action(
                observation=observation,
                controller_target=float(
                    info["controller_target"]
                ),
                action_increment=env.action_increment,
            )

        rows = evaluate_policy(
            env,
            policy_name="rule_based",
            action_function=baseline_action,
        )
    finally:
        env.close()

    episodes = pd.DataFrame(rows)
    episodes_path = args.output_dir / "episodes.csv"
    episodes.to_csv(episodes_path, index=False)

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
        )
        .sort_values("goal_angle")
    )
    by_goal.to_csv(
        args.output_dir / "summary_by_goal.csv",
        index=False,
    )

    summary = {
        "environment_checker_passed": True,
        "seed": args.seed,
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
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nRule-based benchmark")
    print(by_goal.to_string(index=False))
    print(
        f"\nOverall: {summary['successes']}/"
        f"{summary['episodes']} successes "
        f"({summary['success_rate']:.1%})"
    )


if __name__ == "__main__":
    main()
