from __future__ import annotations

import argparse
import time

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np

from g1_rl import G1ElbowTargetEnv


def choose_rule_based_action(
    observation: np.ndarray,
    controller_target: float,
    action_increment: float,
) -> int:
    """
    Rule-based policy used to validate the environment.

    The policy moves the internal controller target toward the
    final goal. Once the controller target is sufficiently close
    to the goal, it selects HOLD and allows the low-level PD
    controller to settle the physical elbow.

    Observation:
        0: current elbow angle
        1: current elbow velocity
        2: goal angle
        3: angle error
    """

    goal_angle = float(observation[2])
    target_error = goal_angle - controller_target
    target_tolerance = action_increment / 2.0

    if target_error < -target_tolerance:
        return G1ElbowTargetEnv.ACTION_DECREASE

    if target_error > target_tolerance:
        return G1ElbowTargetEnv.ACTION_INCREASE

    return G1ElbowTargetEnv.ACTION_HOLD


def run_episode(
    env: gym.Env,
    seed: int,
) -> None:
    observation, info = env.reset(seed=seed)

    cumulative_reward = 0.0

    print("\n=== EPISODE START ===")
    print(f"Initial observation: {observation}")
    print(f"Goal angle: {info['goal_angle']:.4f} rad")

    while True:
        action = choose_rule_based_action(
            observation=observation,
            controller_target=float(
                info["controller_target"]
            ),
            action_increment=0.08,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        cumulative_reward += reward

        print(
            f"step={info['episode_step']:3d} | "
            f"action={action} | "
            f"angle={info['elbow_angle']:+.4f} | "
            f"velocity={info['elbow_velocity']:+.4f} | "
            f"controller_target="
            f"{info['controller_target']:+.4f} | "
            f"goal={info['goal_angle']:+.4f} | "
            f"error={info['angle_error']:+.4f} | "
            f"streak={info['success_streak']:2d} | "
            f"reward={reward:+.4f}"
        )

        if terminated or truncated:
            break

    print("\n=== EPISODE END ===")
    print(f"Terminated:        {terminated}")
    print(f"Truncated:         {truncated}")
    print(f"Success:           {info['is_success']}")
    print(f"Episode steps:     {info['episode_step']}")
    print(f"Final angle:       {info['elbow_angle']:.4f}")
    print(f"Goal angle:        {info['goal_angle']:.4f}")
    print(f"Final error:       {info['angle_error']:.4f}")
    print(f"Cumulative reward: {cumulative_reward:.4f}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the G1 elbow Gymnasium environment."
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Open the MuJoCo viewer.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    env = G1ElbowTargetEnv(
        render_mode="human" if args.render else None,
        goal_angle=-0.8,
    )

    try:
        print("Running Gymnasium environment checker.")
        check_env(
            env,
            skip_render_check=True,
        )
        print("Environment checker passed.")

        run_episode(
            env=env,
            seed=42,
        )

        if args.render:
            print("Close the MuJoCo viewer window to finish.")

            while (
                env.viewer is not None
                and env.viewer.is_running()
            ):
                time.sleep(0.05)
    finally:
        env.close()


if __name__ == "__main__":
    main()
