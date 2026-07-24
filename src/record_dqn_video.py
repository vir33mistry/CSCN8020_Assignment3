"""Create a 2-3 minute MP4 from actual headless MuJoCo/DQN states.

The interactive viewer script remains the authoritative viewer demo. This
recorder exists so the repository also contains a portable, saved-model video
even on machines where an OpenGL window cannot be captured automatically.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dqn import DQNAgent
from dqn.experiment import BENCHMARK_GOALS, set_reproducible_seeds
from g1_rl import G1ElbowTargetEnv


WIDTH = 1280
HEIGHT = 720
FPS = 24
ACTION_NAMES = {0: "DECREASE", 1: "HOLD", 2: "INCREASE"}


@dataclass(frozen=True)
class RecordedStep:
    body_positions: np.ndarray
    goal: float
    step: int
    action: int
    q_values: np.ndarray
    angle: float
    velocity: float
    controller_target: float
    error: float
    streak: int
    reward: float
    cumulative_reward: float
    success: bool


@dataclass(frozen=True)
class RecordedEpisode:
    goal: float
    steps: list[RecordedStep]
    success: bool
    episode_length: int
    cumulative_reward: float
    final_absolute_error: float


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = (
        "DejaVuSans-Bold.ttf"
        if bold
        else "DejaVuSans.ttf"
    )
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size)


FONTS = {
    "title": load_font(34, bold=True),
    "heading": load_font(23, bold=True),
    "body": load_font(18),
    "body_bold": load_font(18, bold=True),
    "small": load_font(14),
    "large": load_font(54, bold=True),
}


def gradient_background() -> Image.Image:
    top = np.asarray([7, 20, 38], dtype=np.float32)
    bottom = np.asarray([18, 48, 67], dtype=np.float32)
    rows = np.linspace(top, bottom, HEIGHT).astype(np.uint8)
    image_array = np.repeat(
        rows[:, np.newaxis, :],
        WIDTH,
        axis=1,
    )
    return Image.fromarray(image_array, mode="RGB")


def draw_card(
    title: str,
    lines: list[str],
    *,
    accent: str = "#38BDF8",
) -> np.ndarray:
    image = gradient_background()
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (120, 90, WIDTH - 120, HEIGHT - 90),
        radius=28,
        fill="#F8FAFC",
        outline=accent,
        width=5,
    )
    draw.text(
        (WIDTH // 2, 155),
        title,
        font=FONTS["title"],
        fill="#0F172A",
        anchor="mm",
    )
    y = 245
    for line in lines:
        draw.text(
            (WIDTH // 2, y),
            line,
            font=FONTS["heading"],
            fill="#334155",
            anchor="mm",
        )
        y += 55
    return np.asarray(image)


def project_bodies(body_positions: np.ndarray) -> np.ndarray:
    """Project MuJoCo body centres with a fixed isometric camera."""

    look_at = np.asarray([0.0, 0.0, 0.9])
    azimuth = np.deg2rad(135.0)
    elevation = np.deg2rad(8.0)
    camera = look_at + 4.0 * np.asarray(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )

    forward = look_at - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    relative = body_positions - look_at
    projected_x = relative @ right
    projected_y = relative @ up

    scale = 330.0
    screen_x = 365.0 + projected_x * scale
    screen_y = 395.0 - projected_y * scale
    return np.column_stack([screen_x, screen_y])


def draw_metric_bar(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    label: str,
    value: float,
    minimum: float,
    maximum: float,
    color: str,
) -> None:
    draw.text(
        (x, y),
        f"{label}: {value:+.3f}",
        font=FONTS["small"],
        fill="#CBD5E1",
    )
    bar_y = y + 23
    draw.rounded_rectangle(
        (x, bar_y, x + width, bar_y + 12),
        radius=6,
        fill="#334155",
    )
    ratio = float(
        np.clip(
            (value - minimum) / (maximum - minimum),
            0.0,
            1.0,
        )
    )
    draw.rounded_rectangle(
        (x, bar_y, x + int(width * ratio), bar_y + 12),
        radius=6,
        fill=color,
    )


def draw_step_frame(
    step: RecordedStep,
    *,
    selected_configuration: str,
    body_parent_ids: np.ndarray,
    body_names: list[str],
) -> np.ndarray:
    image = gradient_background()
    draw = ImageDraw.Draw(image)

    draw.rectangle(
        (0, 0, WIDTH, 76),
        fill="#020617",
    )
    draw.text(
        (32, 22),
        "Unitree G1 - Saved DQN Policy",
        font=FONTS["heading"],
        fill="#F8FAFC",
    )
    draw.text(
        (WIDTH - 32, 22),
        (
            f"{selected_configuration.replace('_', ' ').title()} "
            "| epsilon = 0.0 | CPU"
        ),
        font=FONTS["body"],
        fill="#7DD3FC",
        anchor="ra",
    )

    draw.rounded_rectangle(
        (28, 96, 690, 650),
        radius=20,
        fill="#0F2638",
        outline="#1E526B",
        width=2,
    )
    draw.text(
        (52, 115),
        "Headless MuJoCo physics state",
        font=FONTS["body_bold"],
        fill="#E2E8F0",
    )

    projected = project_bodies(step.body_positions)
    left_arm_ids = {
        index
        for index, name in enumerate(body_names)
        if name
        in {
            "left_shoulder_pitch_link",
            "left_shoulder_roll_link",
            "left_shoulder_yaw_link",
            "left_elbow_link",
            "left_wrist_roll_link",
            "left_wrist_pitch_link",
            "left_wrist_yaw_link",
        }
    }

    # A soft ground reference makes movement easier to interpret.
    draw.line(
        (85, 575, 630, 575),
        fill="#37647A",
        width=2,
    )

    for body_id in range(1, len(body_parent_ids)):
        parent_id = int(body_parent_ids[body_id])
        if parent_id <= 0:
            continue
        start = tuple(projected[parent_id])
        end = tuple(projected[body_id])
        is_left_arm = (
            body_id in left_arm_ids
            or parent_id in left_arm_ids
        )
        draw.line(
            (start[0], start[1], end[0], end[1]),
            fill="#38BDF8" if is_left_arm else "#94A3B8",
            width=8 if is_left_arm else 6,
        )

    for body_id in range(1, len(body_parent_ids)):
        x, y = projected[body_id]
        name = body_names[body_id]
        radius = 10 if name == "left_elbow_link" else 6
        color = (
            "#F97316"
            if name == "left_elbow_link"
            else (
                "#7DD3FC"
                if body_id in left_arm_ids
                else "#E2E8F0"
            )
        )
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline="#0F172A",
            width=2,
        )

    elbow_id = body_names.index("left_elbow_link")
    elbow_x, elbow_y = projected[elbow_id]
    draw.text(
        (elbow_x + 16, elbow_y - 12),
        "controlled elbow",
        font=FONTS["small"],
        fill="#FDBA74",
    )

    draw.rounded_rectangle(
        (720, 96, 1252, 650),
        radius=20,
        fill="#0F172A",
        outline="#1E526B",
        width=2,
    )
    draw.text(
        (748, 122),
        f"Goal {step.goal:+.1f} rad | Step {step.step}",
        font=FONTS["heading"],
        fill="#F8FAFC",
    )

    action_color = {
        0: "#F97316",
        1: "#22C55E",
        2: "#38BDF8",
    }[step.action]
    draw.rounded_rectangle(
        (748, 172, 1218, 228),
        radius=12,
        fill=action_color,
    )
    draw.text(
        (983, 200),
        f"Action {step.action}: {ACTION_NAMES[step.action]}",
        font=FONTS["heading"],
        fill="#07111F",
        anchor="mm",
    )

    draw.text(
        (748, 253),
        (
            f"Q-values  [ {step.q_values[0]:+.2f}, "
            f"{step.q_values[1]:+.2f}, "
            f"{step.q_values[2]:+.2f} ]"
        ),
        font=FONTS["body"],
        fill="#CBD5E1",
    )

    draw_metric_bar(
        draw,
        x=748,
        y=298,
        width=420,
        label="Actual elbow angle",
        value=step.angle,
        minimum=-1.05,
        maximum=2.09,
        color="#38BDF8",
    )
    draw_metric_bar(
        draw,
        x=748,
        y=355,
        width=420,
        label="Controller target",
        value=step.controller_target,
        minimum=-1.05,
        maximum=2.09,
        color="#F97316",
    )
    draw_metric_bar(
        draw,
        x=748,
        y=412,
        width=420,
        label="Goal angle",
        value=step.goal,
        minimum=-1.05,
        maximum=2.09,
        color="#22C55E",
    )

    draw.text(
        (748, 482),
        f"Angle error:       {step.error:+.4f} rad",
        font=FONTS["body"],
        fill="#E2E8F0",
    )
    draw.text(
        (748, 518),
        f"Angular velocity:  {step.velocity:+.4f} rad/s",
        font=FONTS["body"],
        fill="#E2E8F0",
    )
    draw.text(
        (748, 554),
        f"Success streak:    {step.streak}/8",
        font=FONTS["body"],
        fill="#E2E8F0",
    )
    draw.text(
        (748, 590),
        (
            f"Step reward: {step.reward:+.3f}   "
            f"Total: {step.cumulative_reward:+.3f}"
        ),
        font=FONTS["body"],
        fill="#E2E8F0",
    )

    draw.text(
        (WIDTH // 2, 684),
        (
            "Actual MuJoCo simulation state | "
            "Saved checkpoint loaded | No retraining"
        ),
        font=FONTS["small"],
        fill="#94A3B8",
        anchor="mm",
    )
    return np.asarray(image)


def simulate_episodes(
    agent: DQNAgent,
    goals: list[float],
    seed: int,
) -> tuple[
    list[RecordedEpisode],
    np.ndarray,
    list[str],
]:
    env = G1ElbowTargetEnv(
        goal_angle=None,
        goal_range=(-0.8, 0.8),
    )
    body_names = [
        mujoco.mj_id2name(
            env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )
        or f"body_{body_id}"
        for body_id in range(env.model.nbody)
    ]
    body_parent_ids = env.model.body_parentid.copy()
    episodes: list[RecordedEpisode] = []

    try:
        for goal_index, goal in enumerate(goals):
            observation, info = env.reset(
                seed=seed + goal_index,
                options={"goal_angle": goal},
            )
            cumulative_reward = 0.0
            steps: list[RecordedStep] = []
            terminated = False
            truncated = False

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

                steps.append(
                    RecordedStep(
                        body_positions=env.data.xpos.copy(),
                        goal=float(goal),
                        step=int(info["episode_step"]),
                        action=action,
                        q_values=q_values.copy(),
                        angle=float(info["elbow_angle"]),
                        velocity=float(info["elbow_velocity"]),
                        controller_target=float(
                            info["controller_target"]
                        ),
                        error=float(info["angle_error"]),
                        streak=int(info["success_streak"]),
                        reward=float(reward),
                        cumulative_reward=cumulative_reward,
                        success=bool(
                            info.get("is_success", False)
                        ),
                    )
                )

            episodes.append(
                RecordedEpisode(
                    goal=float(goal),
                    steps=steps,
                    success=bool(
                        info.get("is_success", False)
                    ),
                    episode_length=int(info["episode_step"]),
                    cumulative_reward=cumulative_reward,
                    final_absolute_error=float(
                        info["absolute_error"]
                    ),
                )
            )
    finally:
        env.close()

    return episodes, body_parent_ids, body_names


def append_repeated(
    writer: Any,
    frame: np.ndarray,
    count: int,
) -> None:
    for _ in range(count):
        writer.append_data(frame)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 2-3 minute MP4 using the selected saved DQN "
            "and actual headless MuJoCo states."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/selected_dqn.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("video/selected_dqn_demo.mp4"),
    )
    parser.add_argument(
        "--goals",
        nargs="+",
        type=float,
        default=list(BENCHMARK_GOALS),
    )
    parser.add_argument("--seed", type=int, default=28020)
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=135,
        help="Target duration; must be between 120 and 180.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not 120 <= args.duration_seconds <= 180:
        raise ValueError(
            "--duration-seconds must be between 120 and 180."
        )
    if len(args.goals) < 2:
        raise ValueError("The video must show at least two goals.")
    if any(not -0.8 <= goal <= 0.8 for goal in args.goals):
        raise ValueError("Video goals must be in [-0.8, +0.8].")

    set_reproducible_seeds(args.seed)
    agent, metadata = DQNAgent.load_checkpoint(
        args.checkpoint,
        device="cpu",
    )
    selected_configuration = str(
        metadata.get("configuration", "selected DQN")
    )

    episodes, parent_ids, body_names = simulate_episodes(
        agent,
        [float(goal) for goal in args.goals],
        args.seed,
    )
    total_steps = sum(
        len(episode.steps) for episode in episodes
    )
    successes = sum(episode.success for episode in episodes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    title_seconds = 6
    result_seconds = 4
    summary_seconds = 8
    card_frames = FPS * (
        title_seconds
        + result_seconds * len(episodes)
        + summary_seconds
    )
    target_frames = args.duration_seconds * FPS
    movement_frames = max(target_frames - card_frames, total_steps)
    base_repeats = movement_frames // total_steps
    extra_repeats = movement_frames % total_steps
    step_counter = 0

    try:
        title = draw_card(
            "CSCN8020 Assignment 3",
            [
                "Deep Q-Network Control of the Unitree G1 Left Elbow",
                "Viraj Dipakkumar Mistry | Student ID 9088985",
                (
                    f"Saved {selected_configuration.replace('_', ' ').title()} "
                    "| Greedy policy (epsilon = 0.0)"
                ),
                "Four benchmark goals | No retraining",
            ],
        )
        append_repeated(writer, title, title_seconds * FPS)

        for episode in episodes:
            for step in episode.steps:
                frame = draw_step_frame(
                    step,
                    selected_configuration=selected_configuration,
                    body_parent_ids=parent_ids,
                    body_names=body_names,
                )
                repeats = base_repeats
                if step_counter < extra_repeats:
                    repeats += 1
                append_repeated(writer, frame, repeats)
                step_counter += 1

            result = draw_card(
                (
                    f"Goal {episode.goal:+.1f} rad - "
                    f"{'SUCCESS' if episode.success else 'NOT SOLVED'}"
                ),
                [
                    f"Episode length: {episode.episode_length} steps",
                    (
                        f"Cumulative reward: "
                        f"{episode.cumulative_reward:.4f}"
                    ),
                    (
                        f"Final absolute error: "
                        f"{episode.final_absolute_error:.4f} rad"
                    ),
                    "Policy: saved DQN, epsilon = 0.0",
                ],
                accent=(
                    "#22C55E" if episode.success else "#EF4444"
                ),
            )
            append_repeated(
                writer,
                result,
                result_seconds * FPS,
            )

        summary = draw_card(
            "Demonstration Summary",
            [
                f"Successful goals: {successes}/{len(episodes)}",
                (
                    "Goals shown: "
                    + ", ".join(
                        f"{episode.goal:+.1f}"
                        for episode in episodes
                    )
                    + " rad"
                ),
                "Checkpoint loaded from models/selected_dqn.pt",
                (
                    "Run src/render_dqn_policy.py for the "
                    "interactive MuJoCo viewer"
                ),
            ],
            accent="#22C55E",
        )
        append_repeated(
            writer,
            summary,
            summary_seconds * FPS,
        )
    finally:
        writer.close()

    print(f"Created video: {args.output}")
    print(
        f"Duration: approximately "
        f"{args.duration_seconds // 60}:"
        f"{args.duration_seconds % 60:02d}"
    )
    print(f"Successful goals: {successes}/{len(episodes)}")


if __name__ == "__main__":
    main()
