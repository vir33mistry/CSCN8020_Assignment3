from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np


RenderMode = Literal["human"]


class G1ElbowTargetEnv(gym.Env[np.ndarray, int]):
    """
    Gymnasium environment for discrete control of the Unitree G1
    left elbow.

    The agent changes an intermediate controller target using three
    discrete actions. A PD controller with MuJoCo bias-force
    compensation applies torque to the robot.

    Observation:
        [
            current_elbow_angle,
            current_elbow_velocity,
            goal_angle,
            goal_angle - current_elbow_angle,
        ]

    Actions:
        0: decrease the controller target
        1: hold the controller target
        2: increase the controller target
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 50,
    }

    ACTION_DECREASE = 0
    ACTION_HOLD = 1
    ACTION_INCREASE = 2

    CONTROLLED_JOINT = "left_elbow_joint"
    CONTROLLED_ACTUATOR = "left_elbow"

    def __init__(
        self,
        scene_path: str | Path = (
            "assets/g1_fixed_base/"
            "scene_29dof_fixed_base.xml"
        ),
        render_mode: RenderMode | None = None,
        goal_angle: float | None = None,
        goal_range: tuple[float, float] = (-0.9, 1.5),
        action_increment: float = 0.08,
        frame_skip: int = 10,
        maximum_episode_steps: int = 150,
        success_tolerance: float = 0.04,
        required_success_steps: int = 8,
        controlled_kp: float = 20.0,
        controlled_kd: float = 2.0,
        hold_kp: float = 30.0,
        hold_kd: float = 3.0,
    ) -> None:
        super().__init__()

        self.scene_path = Path(scene_path).expanduser().resolve()

        if not self.scene_path.is_file():
            raise FileNotFoundError(
                f"G1 scene file was not found: {self.scene_path}"
            )

        if render_mode not in {None, "human"}:
            raise ValueError(
                "render_mode must be None or 'human'."
            )

        if not np.isfinite(action_increment) or action_increment <= 0:
            raise ValueError(
                "action_increment must be greater than zero."
            )

        if not np.isfinite(frame_skip) or frame_skip <= 0:
            raise ValueError(
                "frame_skip must be greater than zero."
            )

        if (
            not np.isfinite(maximum_episode_steps)
            or maximum_episode_steps <= 0
        ):
            raise ValueError(
                "maximum_episode_steps must be greater than zero."
            )

        if not np.isfinite(success_tolerance) or success_tolerance <= 0:
            raise ValueError(
                "success_tolerance must be greater than zero."
            )

        if (
            not np.isfinite(required_success_steps)
            or required_success_steps <= 0
        ):
            raise ValueError(
                "required_success_steps must be greater than zero."
            )

        gain_values = {
            "controlled_kp": controlled_kp,
            "controlled_kd": controlled_kd,
            "hold_kp": hold_kp,
            "hold_kd": hold_kd,
        }

        for gain_name, gain_value in gain_values.items():
            if not np.isfinite(gain_value) or gain_value < 0:
                raise ValueError(
                    f"{gain_name} must be greater than or equal to zero."
                )

        self.render_mode = render_mode
        self.fixed_goal_angle = goal_angle
        self.goal_range = goal_range
        self.action_increment = action_increment
        self.frame_skip = frame_skip
        self.maximum_episode_steps = maximum_episode_steps
        self.success_tolerance = success_tolerance
        self.required_success_steps = required_success_steps

        self.controlled_kp = controlled_kp
        self.controlled_kd = controlled_kd
        self.hold_kp = hold_kp
        self.hold_kd = hold_kd

        self.model = mujoco.MjModel.from_xml_path(
            str(self.scene_path)
        )
        self.data = mujoco.MjData(self.model)

        self.joint_id = self._require_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            self.CONTROLLED_JOINT,
        )

        self.actuator_id = self._require_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            self.CONTROLLED_ACTUATOR,
        )

        self.qpos_index = int(
            self.model.jnt_qposadr[self.joint_id]
        )

        self.qvel_index = int(
            self.model.jnt_dofadr[self.joint_id]
        )

        joint_low, joint_high = self.model.jnt_range[
            self.joint_id
        ]

        self.joint_low = float(joint_low)
        self.joint_high = float(joint_high)

        self._validate_goal_range()

        self.joint_actuator_mapping = (
            self._build_joint_actuator_mapping()
        )

        self.action_space = spaces.Discrete(3)

        float_limit = np.finfo(np.float32).max

        self.observation_space = spaces.Box(
            low=np.array(
                [
                    self.joint_low,
                    -float_limit,
                    self.joint_low,
                    self.joint_low - self.joint_high,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.joint_high,
                    float_limit,
                    self.joint_high,
                    self.joint_high - self.joint_low,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.goal_angle = 0.0
        self.controller_target = 0.0
        self.hold_targets = np.zeros(
            self.model.nq,
            dtype=np.float64,
        )

        self.episode_step = 0
        self.success_streak = 0

        self.viewer = None

    def _require_id(
        self,
        object_type: mujoco.mjtObj,
        name: str,
    ) -> int:
        object_id = mujoco.mj_name2id(
            self.model,
            object_type,
            name,
        )

        if object_id < 0:
            raise ValueError(
                f"MuJoCo object was not found: {name}"
            )

        return object_id

    def _validate_goal_range(self) -> None:
        if len(self.goal_range) != 2:
            raise ValueError(
                "goal_range must contain exactly two values."
            )

        goal_low, goal_high = self.goal_range

        if not np.isfinite([goal_low, goal_high]).all():
            raise ValueError(
                "goal_range bounds must be finite."
            )

        if goal_low >= goal_high:
            raise ValueError(
                "goal_range lower bound must be less than "
                "the upper bound."
            )

        if goal_low < self.joint_low:
            raise ValueError(
                "goal_range lower bound is outside the "
                "elbow joint range."
            )

        if goal_high > self.joint_high:
            raise ValueError(
                "goal_range upper bound is outside the "
                "elbow joint range."
            )

        if self.fixed_goal_angle is not None:
            if not np.isfinite(self.fixed_goal_angle):
                raise ValueError(
                    "goal_angle must be finite."
                )

            if not (
                self.joint_low
                <= self.fixed_goal_angle
                <= self.joint_high
            ):
                raise ValueError(
                    "goal_angle is outside the elbow joint range."
                )

    def _build_joint_actuator_mapping(
        self,
    ) -> list[tuple[int, int, int]]:
        mapping: list[tuple[int, int, int]] = []

        for actuator_id in range(self.model.nu):
            joint_id = int(
                self.model.actuator_trnid[actuator_id, 0]
            )

            if joint_id < 0:
                raise RuntimeError(
                    f"Actuator {actuator_id} is not attached "
                    "to a joint."
                )

            qpos_index = int(
                self.model.jnt_qposadr[joint_id]
            )

            qvel_index = int(
                self.model.jnt_dofadr[joint_id]
            )

            mapping.append(
                (
                    actuator_id,
                    qpos_index,
                    qvel_index,
                )
            )

        return mapping

    @staticmethod
    def _calculate_pd_torque(
        target_angle: float,
        current_angle: float,
        current_velocity: float,
        kp: float,
        kd: float,
    ) -> float:
        return (
            kp * (target_angle - current_angle)
            - kd * current_velocity
        )

    def _get_observation(self) -> np.ndarray:
        elbow_angle = float(
            self.data.qpos[self.qpos_index]
        )

        elbow_velocity = float(
            self.data.qvel[self.qvel_index]
        )

        angle_error = self.goal_angle - elbow_angle

        return np.array(
            [
                elbow_angle,
                elbow_velocity,
                self.goal_angle,
                angle_error,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        elbow_angle = float(
            self.data.qpos[self.qpos_index]
        )

        angle_error = self.goal_angle - elbow_angle

        return {
            "elbow_angle": elbow_angle,
            "elbow_velocity": float(
                self.data.qvel[self.qvel_index]
            ),
            "goal_angle": self.goal_angle,
            "controller_target": self.controller_target,
            "angle_error": angle_error,
            "absolute_error": abs(angle_error),
            "success_streak": self.success_streak,
            "episode_step": self.episode_step,
        }

    def _select_goal(self) -> float:
        if self.fixed_goal_angle is not None:
            return float(self.fixed_goal_angle)

        low, high = self.goal_range

        return float(
            self.np_random.uniform(low, high)
        )

    def _apply_action(self, action: int) -> None:
        if action == self.ACTION_DECREASE:
            self.controller_target -= self.action_increment
        elif action == self.ACTION_HOLD:
            pass
        elif action == self.ACTION_INCREASE:
            self.controller_target += self.action_increment
        else:
            raise ValueError(
                f"Invalid action: {action}"
            )

        self.controller_target = float(
            np.clip(
                self.controller_target,
                self.joint_low,
                self.joint_high,
            )
        )

    def _apply_controller(self) -> None:
        for (
            actuator_id,
            qpos_index,
            qvel_index,
        ) in self.joint_actuator_mapping:
            current_position = float(
                self.data.qpos[qpos_index]
            )

            current_velocity = float(
                self.data.qvel[qvel_index]
            )

            if actuator_id == self.actuator_id:
                desired_position = self.controller_target
                kp = self.controlled_kp
                kd = self.controlled_kd
            else:
                desired_position = float(
                    self.hold_targets[qpos_index]
                )
                kp = self.hold_kp
                kd = self.hold_kd

            pd_torque = self._calculate_pd_torque(
                target_angle=desired_position,
                current_angle=current_position,
                current_velocity=current_velocity,
                kp=kp,
                kd=kd,
            )

            bias_torque = float(
                self.data.qfrc_bias[qvel_index]
            )

            commanded_torque = (
                pd_torque + bias_torque
            )

            control_low, control_high = (
                self.model.actuator_ctrlrange[
                    actuator_id
                ]
            )

            self.data.ctrl[actuator_id] = np.clip(
                commanded_torque,
                control_low,
                control_high,
            )

    def _calculate_reward(
        self,
        absolute_error: float,
        action: int,
    ) -> float:
        # Main shaping term: closer to the goal is better.
        reward = -absolute_error

        # Small bonus for entering the success region.
        if absolute_error <= self.success_tolerance:
            reward += 1.0

        # Discourage unnecessary changes once close to the goal.
        if (
            absolute_error <= self.success_tolerance
            and action != self.ACTION_HOLD
        ):
            reward -= 0.05

        return float(reward)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        self.hold_targets = self.data.qpos.copy()

        if options is not None and "goal_angle" in options:
            selected_goal = float(
                options["goal_angle"]
            )

            if not np.isfinite(selected_goal) or not (
                self.joint_low
                <= selected_goal
                <= self.joint_high
            ):
                raise ValueError(
                    "The reset goal_angle is outside "
                    "the elbow joint range."
                )

            self.goal_angle = selected_goal
        else:
            self.goal_angle = self._select_goal()

        self.controller_target = float(
            self.data.qpos[self.qpos_index]
        )

        self.episode_step = 0
        self.success_streak = 0

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        if not self.action_space.contains(action):
            raise ValueError(
                f"Action {action} is not in the action space."
            )

        self._apply_action(int(action))

        for _ in range(self.frame_skip):
            self._apply_controller()
            mujoco.mj_step(
                self.model,
                self.data,
            )

        self.episode_step += 1

        observation = self._get_observation()

        absolute_error = abs(
            float(observation[3])
        )

        if absolute_error <= self.success_tolerance:
            self.success_streak += 1
        else:
            self.success_streak = 0

        terminated = (
            self.success_streak
            >= self.required_success_steps
        )

        truncated = (
            self.episode_step
            >= self.maximum_episode_steps
        )

        reward = self._calculate_reward(
            absolute_error=absolute_error,
            action=int(action),
        )

        if terminated:
            reward += 10.0

        info = self._get_info()
        info["is_success"] = terminated

        if self.render_mode == "human":
            self.render()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> None:
        if self.render_mode != "human":
            return

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
            )

        if not self.viewer.is_running():
            return

        self.viewer.sync()

        render_interval = 1.0 / self.metadata[
            "render_fps"
        ]

        time.sleep(render_interval)

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
