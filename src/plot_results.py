"""Generate every plot required by the Assignment 3 rubric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "config_a": "#176B87",
    "config_b": "#D97706",
    "rule_based": "#64748B",
}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )


def save_figure(
    figure: plt.Figure,
    output_path: Path,
) -> None:
    figure.tight_layout()
    figure.savefig(
        output_path,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def load_training_frame(
    results_root: Path,
    configuration: str,
) -> pd.DataFrame:
    path = (
        results_root
        / configuration
        / "training_metrics.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Training metrics were not found: {path}"
        )
    return pd.read_csv(path)


def plot_training_rewards(
    frames: dict[str, pd.DataFrame],
    plots_dir: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
    )

    for axis, (configuration, frame) in zip(
        axes,
        frames.items(),
    ):
        moving_average = (
            frame["cumulative_reward"]
            .rolling(window=20, min_periods=1)
            .mean()
        )
        axis.plot(
            frame["episode"],
            frame["cumulative_reward"],
            color=COLORS[configuration],
            alpha=0.25,
            linewidth=0.8,
            label="Raw episode reward",
        )
        axis.plot(
            frame["episode"],
            moving_average,
            color=COLORS[configuration],
            linewidth=2.2,
            label="20-episode moving average",
        )
        axis.set_title(
            f"{configuration.replace('_', ' ').title()} "
            "- training reward"
        )
        axis.set_ylabel("Cumulative reward")
        axis.legend(loc="best")

    axes[-1].set_xlabel("Episode")
    figure.suptitle(
        "Raw and moving-average training reward",
        fontsize=14,
        fontweight="bold",
    )
    save_figure(
        figure,
        plots_dir / "training_reward.png",
    )


def plot_training_success(
    frames: dict[str, pd.DataFrame],
    plots_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.8))

    for configuration, frame in frames.items():
        rolling_success = (
            frame["success"]
            .astype(float)
            .rolling(window=50, min_periods=1)
            .mean()
        )
        axis.plot(
            frame["episode"],
            rolling_success * 100,
            color=COLORS[configuration],
            linewidth=2,
            label=(
                f"{configuration.replace('_', ' ').title()} "
                f"(decay "
                f"{frame['configuration'].map({'config_a': 0.995, 'config_b': 0.985}).iloc[0]:.3f})"
            ),
        )

    axis.axhline(
        80,
        color="#B91C1C",
        linestyle="--",
        linewidth=1.2,
        label="80% assignment target",
    )
    axis.set_ylim(-2, 102)
    axis.set_xlabel("Episode")
    axis.set_ylabel("Rolling success rate (%)")
    axis.set_title(
        "Training success rate - 50-episode rolling window",
        fontweight="bold",
    )
    axis.legend(loc="lower right")
    save_figure(
        figure,
        plots_dir / "training_success_rate.png",
    )


def plot_epsilon(
    frames: dict[str, pd.DataFrame],
    plots_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.8))

    for configuration, frame in frames.items():
        axis.plot(
            frame["episode"],
            frame["epsilon_used"],
            color=COLORS[configuration],
            linewidth=2,
            label=configuration.replace("_", " ").title(),
        )

    axis.axhline(
        0.05,
        color="#475569",
        linestyle="--",
        label="Minimum epsilon = 0.05",
    )
    axis.set_xlabel("Episode")
    axis.set_ylabel("Epsilon")
    axis.set_ylim(0, 1.03)
    axis.set_title(
        "Epsilon-greedy exploration schedule",
        fontweight="bold",
    )
    axis.legend()
    save_figure(
        figure,
        plots_dir / "epsilon_decay.png",
    )


def plot_losses(
    results_root: Path,
    plots_dir: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=False,
    )

    for axis, configuration in zip(
        axes,
        ("config_a", "config_b"),
    ):
        path = (
            results_root
            / configuration
            / "loss_metrics.csv"
        )
        frame = pd.read_csv(path)
        if frame.empty:
            axis.text(
                0.5,
                0.5,
                "No optimization losses recorded",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            continue

        # Draw a light, down-sampled raw trace so large runs stay
        # readable, plus a 500-step moving average.
        stride = max(len(frame) // 4_000, 1)
        raw = frame.iloc[::stride]
        smoothed = (
            frame["loss"]
            .rolling(window=500, min_periods=1)
            .mean()
        )
        axis.plot(
            raw["optimization_step"],
            raw["loss"],
            color=COLORS[configuration],
            alpha=0.15,
            linewidth=0.6,
            label="Raw loss (display sample)",
        )
        axis.plot(
            frame["optimization_step"],
            smoothed,
            color=COLORS[configuration],
            linewidth=1.8,
            label="500-update moving average",
        )
        axis.set_title(
            configuration.replace("_", " ").title()
        )
        axis.set_ylabel("Huber loss")
        axis.set_yscale("log")
        axis.legend(loc="best")

    axes[-1].set_xlabel("Optimization step")
    figure.suptitle(
        "Temporal-difference optimization loss",
        fontsize=14,
        fontweight="bold",
    )
    save_figure(
        figure,
        plots_dir / "loss_curve.png",
    )


def plot_configuration_comparison(
    results_root: Path,
    evaluation_dir: Path,
    plots_dir: Path,
) -> None:
    training_summaries = {}
    evaluation_summaries = {}

    for configuration in ("config_a", "config_b"):
        training_summaries[configuration] = json.loads(
            (
                results_root
                / configuration
                / "training_summary.json"
            ).read_text(encoding="utf-8")
        )
        evaluation_summaries[configuration] = json.loads(
            (
                evaluation_dir
                / f"{configuration}_summary.json"
            ).read_text(encoding="utf-8")
        )

    metrics = [
        "Final-50 training\nsuccess (%)",
        "Greedy evaluation\nsuccess (%)",
        "Mean evaluation\nreward",
    ]
    config_a_values = [
        100
        * training_summaries["config_a"][
            "training_success_rate_final_50"
        ],
        100
        * evaluation_summaries["config_a"]["success_rate"],
        evaluation_summaries["config_a"][
            "mean_cumulative_reward"
        ],
    ]
    config_b_values = [
        100
        * training_summaries["config_b"][
            "training_success_rate_final_50"
        ],
        100
        * evaluation_summaries["config_b"]["success_rate"],
        evaluation_summaries["config_b"][
            "mean_cumulative_reward"
        ],
    ]

    x = np.arange(len(metrics))
    width = 0.34
    figure, axis = plt.subplots(figsize=(10, 5))
    bars_a = axis.bar(
        x - width / 2,
        config_a_values,
        width,
        color=COLORS["config_a"],
        label="Config A - epsilon decay 0.995",
    )
    bars_b = axis.bar(
        x + width / 2,
        config_b_values,
        width,
        color=COLORS["config_b"],
        label="Config B - epsilon decay 0.985",
    )
    axis.bar_label(bars_a, fmt="%.2f", padding=3)
    axis.bar_label(bars_b, fmt="%.2f", padding=3)
    axis.set_xticks(x, metrics)
    axis.set_ylabel("Metric value")
    axis.set_title(
        "Controlled epsilon-decay comparison",
        fontweight="bold",
    )
    axis.legend()
    save_figure(
        figure,
        plots_dir / "configuration_comparison.png",
    )


def plot_evaluation_by_goal(
    evaluation_dir: Path,
    plots_dir: Path,
) -> None:
    frame = pd.read_csv(
        evaluation_dir / "all_policies_by_goal.csv"
    )
    policy_order = ["rule_based", "config_a", "config_b"]
    goals = sorted(frame["goal_angle"].unique())
    x = np.arange(len(goals))
    width = 0.24

    figure, axis = plt.subplots(figsize=(10, 5.2))

    for policy_index, policy in enumerate(policy_order):
        subset = (
            frame[frame["policy"] == policy]
            .set_index("goal_angle")
            .reindex(goals)
        )
        offsets = x + (
            policy_index - (len(policy_order) - 1) / 2
        ) * width
        bars = axis.bar(
            offsets,
            subset["success_rate"].to_numpy() * 100,
            width,
            label=policy.replace("_", " ").title(),
            color=COLORS[policy],
        )
        axis.bar_label(
            bars,
            fmt="%.0f%%",
            padding=2,
            fontsize=8,
        )

    axis.axhline(
        80,
        color="#B91C1C",
        linestyle="--",
        linewidth=1.2,
        label="80% required overall threshold",
    )
    axis.set_xticks(
        x,
        [f"{goal:+.1f}" for goal in goals],
    )
    axis.set_ylim(0, 108)
    axis.set_xlabel("Target angle (rad)")
    axis.set_ylabel("Greedy success rate (%)")
    axis.set_title(
        "Evaluation success rate by target angle "
        "(5 episodes per goal)",
        fontweight="bold",
    )
    axis.legend(ncol=2, loc="lower center")
    save_figure(
        figure,
        plots_dir / "evaluation_success_by_goal.png",
    )


def plot_rule_based_vs_selected(
    evaluation_dir: Path,
    plots_dir: Path,
) -> None:
    comparison = pd.read_csv(
        evaluation_dir / "policy_comparison.csv"
    )
    selection = json.loads(
        (
            evaluation_dir / "selection_summary.json"
        ).read_text(encoding="utf-8")
    )
    selected = selection["selected_configuration"]
    subset = (
        comparison[
            comparison["policy"].isin(
                ["rule_based", selected]
            )
        ]
        .set_index("policy")
        .loc[["rule_based", selected]]
    )

    metrics = [
        ("success_rate", "Success rate (%)", 100.0),
        ("mean_episode_length", "Mean episode length", 1.0),
        (
            "mean_final_absolute_error",
            "Mean final |error| (rad)",
            1.0,
        ),
        ("mean_action_changes", "Mean action changes", 1.0),
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10, 7),
    )
    for axis, (column, title, scale) in zip(
        axes.flat,
        metrics,
    ):
        values = subset[column].to_numpy() * scale
        bars = axis.bar(
            ["Rule-based", selected.replace("_", " ").title()],
            values,
            color=[
                COLORS["rule_based"],
                COLORS[selected],
            ],
            width=0.6,
        )
        axis.bar_label(bars, fmt="%.3f", padding=3)
        axis.set_title(title)

    figure.suptitle(
        "Rule-based baseline versus selected DQN",
        fontsize=14,
        fontweight="bold",
    )
    save_figure(
        figure,
        plots_dir / "rule_based_vs_selected_dqn.png",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all Assignment 3 plots."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path("results/evaluation"),
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("plots"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    frames = {
        configuration: load_training_frame(
            args.results_root,
            configuration,
        )
        for configuration in ("config_a", "config_b")
    }

    plot_training_rewards(frames, args.plots_dir)
    plot_training_success(frames, args.plots_dir)
    plot_epsilon(frames, args.plots_dir)
    plot_losses(args.results_root, args.plots_dir)
    plot_configuration_comparison(
        args.results_root,
        args.evaluation_dir,
        args.plots_dir,
    )
    plot_evaluation_by_goal(
        args.evaluation_dir,
        args.plots_dir,
    )
    plot_rule_based_vs_selected(
        args.evaluation_dir,
        args.plots_dir,
    )

    print("Generated plots:")
    for path in sorted(args.plots_dir.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
