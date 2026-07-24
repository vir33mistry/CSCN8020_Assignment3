"""Interactive MuJoCo-viewer demonstration of the saved DQN policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from dqn import DQNAgent
from dqn.experiment import set_reproducible_seeds
from g1_rl import G1ElbowTargetEnv


ACTION_NAMES = {
    G1ElbowTargetEnv.ACTION_DECREASE: "DECREASE",
    G1ElbowTargetEnv.ACTION_HOLD: "HOLD",
    G1ElbowTargetEnv.ACTION_INCREASE: "INCREASE",
}


def run_rendered_episode(
    env: G1ElbowTargetEnv,
    agent: DQNAgent,
    *,
    goal_angle: float,
    seed: int,
) -> dict[str, float | int | bool]:
    observation, info = env.reset(
        seed=seed,
        options={"goal_angle": goal_angle},
    )
    cumulative_reward = 0.0
    terminated = False
    truncated = False

    print(
        f"\n=== GREEDY DQN EPISODE: "
        f"goal {goal_angle:+.1f} rad ==="
    )

    while not (terminated or truncated):
        q_values = agent.q_values(observation)
        action = int(np.argmax(q_values))
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)
        cumulative_reward += float(reward)

        print(
            f"step={int(info['episode_step']):3d} | "
            f"action={action} "
            f"({ACTION_NAMES[action]:8s}) | "
            f"Q=[{q_values[0]:+.3f}, "
            f"{q_values[1]:+.3f}, "
            f"{q_values[2]:+.3f}] | "
            f"target={float(info['controller_target']):+.3f} | "
            f"angle={float(info['elbow_angle']):+.3f} | "
            f"error={float(info['angle_error']):+.3f} | "
            f"streak={int(info['success_streak']):2d} | "
            f"reward={reward:+.3f}"
        )

        if (
            env.viewer is not None
            and not env.viewer.is_running()
        ):
            print("Viewer closed before the episode finished.")
            break

    success = bool(info.get("is_success", False))
    print("\nEpisode result")
    print(f"  Success:             {success}")
    print(f"  Terminated:          {terminated}")
    print(f"  Truncated:           {truncated}")
    print(f"  Episode length:      {info['episode_step']}")
    print(f"  Cumulative reward:   {cumulative_reward:.4f}")
    print(
        f"  Final absolute error: "
        f"{float(info['absolute_error']):.4f} rad"
    )

    return {
        "success": success,
        "episode_length": int(info["episode_step"]),
        "cumulative_reward": cumulative_reward,
        "final_absolute_error": float(
            info["absolute_error"]
        ),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the selected DQN checkpoint and demonstrate "
            "greedy control in the MuJoCo viewer."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/selected_dqn.pt"),
    )
    parser.add_argument(
        "--goals",
        nargs="+",
        type=float,
        default=[-0.8, 0.8],
        help="Two or more demonstration goals in radians.",
    )
    parser.add_argument("--seed", type=int, default=28020)
    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Use timed pauses rather than pressing Enter "
            "between episodes."
        ),
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=3.0,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if len(args.goals) < 2:
        raise ValueError(
            "Provide at least two goals for the assignment demo."
        )
    if any(not -0.8 <= goal <= 0.8 for goal in args.goals):
        raise ValueError(
            "Every demonstration goal must be in [-0.8, +0.8]."
        )
    if args.pause_seconds < 0:
        raise ValueError("--pause-seconds cannot be negative.")

    set_reproducible_seeds(args.seed)
    agent, metadata = DQNAgent.load_checkpoint(
        args.checkpoint,
        device="cpu",
    )

    env = G1ElbowTargetEnv(
        goal_angle=None,
        goal_range=(-0.8, 0.8),
        render_mode="human",
    )

    print("Loaded saved DQN checkpoint.")
    print(f"Checkpoint: {args.checkpoint}")
    print(
        "Training configuration: "
        f"{metadata.get('configuration', 'unknown')}"
    )
    print("Evaluation epsilon: 0.0 (greedy)")
    print("No retraining is performed by this script.")

    results = []
    try:
        # Open the viewer and expose the initial model before
        # the recording begins.
        env.reset(
            seed=args.seed,
            options={"goal_angle": args.goals[0]},
        )
        env.render()

        if args.auto:
            print(
                f"Starting automatically in "
                f"{args.pause_seconds:.1f} seconds."
            )
            time.sleep(args.pause_seconds)
        else:
            input(
                "Position the MuJoCo camera, begin screen "
                "recording, then press Enter..."
            )

        for goal_index, goal_angle in enumerate(args.goals):
            result = run_rendered_episode(
                env,
                agent,
                goal_angle=goal_angle,
                seed=args.seed + goal_index,
            )
            results.append(result)

            if goal_index < len(args.goals) - 1:
                if args.auto:
                    time.sleep(args.pause_seconds)
                else:
                    input(
                        "Press Enter to start the next goal..."
                    )

        successful = sum(
            bool(result["success"]) for result in results
        )
        print(
            f"\nDemonstration complete: "
            f"{successful}/{len(results)} successful goals."
        )

        if not args.auto:
            input(
                "Stop screen recording, then press Enter "
                "to close the viewer..."
            )
        else:
            time.sleep(args.pause_seconds)
    finally:
        env.close()


if __name__ == "__main__":
    main()
