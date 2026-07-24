from __future__ import annotations

import argparse
import time

import mujoco.viewer

from g1_rl import G1ElbowTargetEnv
from test_g1_elbow_env import choose_rule_based_action


def wait_for_student(
    env: G1ElbowTargetEnv,
) -> bool:
    """
    Keep the viewer open without advancing physics.

    The student can position the camera and inspect the
    control panel before pressing Enter in the terminal.
    """

    print()
    print("=== CAMERA SETUP ===")
    print("The MuJoCo viewer is open.")
    print("Use the mouse to rotate, pan, and zoom.")
    print("Open the Control panel on the right if needed.")
    print("Return to the terminal and press Enter to begin.")
    print()

    input("Press Enter to start the Gymnasium demonstration...")

    return (
        env.viewer is not None
        and env.viewer.is_running()
    )


def run_demo(
    env: G1ElbowTargetEnv,
    seed: int,
    startup_countdown: int,
) -> None:
    observation, info = env.reset(seed=seed)

    # Ensure the initial state is visible before waiting.
    env.render()

    if not wait_for_student(env):
        print("The viewer was closed before the demo started.")
        return

    print()
    print("Starting in:")

    for seconds_remaining in range(
        startup_countdown,
        0,
        -1,
    ):
        print(seconds_remaining)
        time.sleep(1.0)

    print("GO")
    print()

    cumulative_reward = 0.0

    while True:
        action = choose_rule_based_action(
            observation=observation,
            controller_target=float(
                info["controller_target"]
            ),
            action_increment=env.action_increment,
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
            f"angle={info['elbow_angle']:+.4f} rad | "
            f"velocity={info['elbow_velocity']:+.4f} rad/s | "
            f"controller_target="
            f"{info['controller_target']:+.4f} rad | "
            f"goal={info['goal_angle']:+.4f} rad | "
            f"error={info['angle_error']:+.4f} rad | "
            f"streak={info['success_streak']:2d} | "
            f"reward={reward:+.4f}"
        )

        if env.viewer is not None:
            if not env.viewer.is_running():
                print(
                    "The viewer was closed before the "
                    "episode finished."
                )
                break

        if terminated or truncated:
            print()
            print("=== DEMONSTRATION COMPLETE ===")
            print(f"Terminated:        {terminated}")
            print(f"Truncated:         {truncated}")
            print(f"Success:           {info['is_success']}")
            print(f"Episode steps:     {info['episode_step']}")
            print(f"Final angle:       {info['elbow_angle']:.4f}")
            print(f"Goal angle:        {info['goal_angle']:.4f}")
            print(f"Final error:       {info['angle_error']:.4f}")
            print(
                f"Cumulative reward: "
                f"{cumulative_reward:.4f}"
            )
            break

    print()
    input(
        "Press Enter to close the viewer and end the demo..."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive visual demonstration of the "
            "Unitree G1 elbow Gymnasium environment."
        )
    )

    parser.add_argument(
        "--goal",
        type=float,
        default=-0.8,
        help="Elbow goal angle in radians.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Gymnasium reset seed.",
    )

    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Seconds to count down before the episode starts.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.countdown < 0:
        raise ValueError(
            "--countdown must be zero or greater."
        )

    env = G1ElbowTargetEnv(
        render_mode="human",
        goal_angle=args.goal,
    )

    try:
        run_demo(
            env=env,
            seed=args.seed,
            startup_countdown=args.countdown,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
