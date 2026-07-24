from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


DEFAULT_SCENE = Path(
    "assets/g1_fixed_base/"
    "scene_29dof_fixed_base.xml"
)

CONTROLLED_JOINT = "left_elbow_joint"
CONTROLLED_ACTUATOR = "left_elbow"

TARGET_ANGLE_RAD = -0.8
KP = 20.0
KD = 2.0
HOLD_KP = 30.0
HOLD_KD = 3.0

SIMULATION_DURATION_SECONDS = 8.0
LOG_INTERVAL_STEPS = 5


def require_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)

    if object_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")

    return object_id


def set_initial_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    """
    Set a simple initial pose.

    The robot is still dynamically simulated and is not balanced by
    a locomotion controller. This function only creates a repeatable
    starting configuration.
    """
    mujoco.mj_resetData(model, data)

    # Floating-base position: x, y, z.
    data.qpos[0:3] = np.array([0.0, 0.0, 0.80])

    # Floating-base quaternion: w, x, y, z.
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

    mujoco.mj_forward(model, data)


def calculate_pd_torque(
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control one Unitree G1 arm joint with a PD controller."
    )

    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help="Path to the G1 scene XML.",
    )

    parser.add_argument(
        "--target",
        type=float,
        default=TARGET_ANGLE_RAD,
        help="Target joint angle in radians.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=SIMULATION_DURATION_SECONDS,
        help="Simulation duration in seconds.",
    )

    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run without opening the viewer.",
    )

    return parser.parse_args()

def build_joint_actuator_mapping(
    model: mujoco.MjModel,
) -> list[tuple[int, int, int]]:
    """
    Build mappings between actuators, joint position indices,
    and joint velocity indices.

    The returned tuple is:
        actuator_id, qpos_index, qvel_index
    """
    mapping: list[tuple[int, int, int]] = []

    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])

        if joint_id < 0:
            raise RuntimeError(
                f"Actuator {actuator_id} is not connected to a joint."
            )

        qpos_index = int(model.jnt_qposadr[joint_id])
        qvel_index = int(model.jnt_dofadr[joint_id])

        mapping.append(
            (
                actuator_id,
                qpos_index,
                qvel_index,
            )
        )

    return mapping


def run_simulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_id: int,
    qpos_index: int,
    qvel_index: int,
    target: float,
    duration: float,
    output_path: Path,
    joint_actuator_mapping: list[tuple[int, int, int]],
    hold_targets: np.ndarray,
    viewer: mujoco.viewer.Handle | None = None,
) -> None:
    start_sim_time = data.time
    step_number = 0

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "simulation_time",
                "target_angle_rad",
                "joint_angle_rad",
                "joint_velocity_rad_s",
                "angle_error_rad",
                "commanded_torque_nm",
            ]
        )

        while data.time - start_sim_time < duration:
            wall_step_start = time.perf_counter()

            for (
                mapped_actuator_id,
                mapped_qpos_index,
                mapped_qvel_index,
            ) in joint_actuator_mapping:
                current_position = float(
                    data.qpos[mapped_qpos_index]
                )

                current_velocity = float(
                    data.qvel[mapped_qvel_index]
                )

                if mapped_actuator_id == actuator_id:
                    desired_position = target
                    kp = KP
                    kd = KD
                else:
                    desired_position = float(
                        hold_targets[mapped_qpos_index]
                    )
                    kp = HOLD_KP
                    kd = HOLD_KD

                pd_torque = calculate_pd_torque(
                    target_angle=desired_position,
                    current_angle=current_position,
                    current_velocity=current_velocity,
                    kp=kp,
                    kd=kd,
                )

                # Compensate for gravity and velocity-dependent bias forces.
                bias_torque = float(
                    data.qfrc_bias[mapped_qvel_index]
                )

                commanded_torque = pd_torque + bias_torque
                mapped_ctrl_low, mapped_ctrl_high = (
                    model.actuator_ctrlrange[mapped_actuator_id]
                )

                data.ctrl[mapped_actuator_id] = np.clip(
                    commanded_torque,
                    mapped_ctrl_low,
                    mapped_ctrl_high,
                )

            torque = float(data.ctrl[actuator_id])
            mujoco.mj_step(model, data)

            if step_number % LOG_INTERVAL_STEPS == 0:
                logged_angle = float(data.qpos[qpos_index])
                logged_velocity = float(data.qvel[qvel_index])
                logged_error = target - logged_angle

                writer.writerow(
                    [
                        f"{data.time:.6f}",
                        f"{target:.6f}",
                        f"{logged_angle:.6f}",
                        f"{logged_velocity:.6f}",
                        f"{logged_error:.6f}",
                        f"{torque:.6f}",
                    ]
                )

            if viewer is not None:
                if not viewer.is_running():
                    break

                viewer.sync()

                elapsed = time.perf_counter() - wall_step_start
                remaining = model.opt.timestep - elapsed

                if remaining > 0:
                    time.sleep(remaining)

            step_number += 1


def main() -> None:
    args = parse_arguments()
    if args.duration <= 0:
        raise ValueError("--duration must be greater than zero.")
    scene_path = args.scene.resolve()

    if not scene_path.is_file():
        raise FileNotFoundError(
            f"Scene file not found: {scene_path}"
        )

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    joint_id = require_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        CONTROLLED_JOINT,
    )

    actuator_id = require_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        CONTROLLED_ACTUATOR,
    )

    qpos_index = int(model.jnt_qposadr[joint_id])
    qvel_index = int(model.jnt_dofadr[joint_id])

    joint_low, joint_high = model.jnt_range[joint_id]
    ctrl_low, ctrl_high = model.actuator_ctrlrange[actuator_id]

    target = float(
        np.clip(args.target, joint_low, joint_high)
    )

    print("=== SINGLE-JOINT CONTROL ===")
    print(f"Scene:             {scene_path}")
    print(f"Joint:             {CONTROLLED_JOINT}")
    print(f"Actuator:          {CONTROLLED_ACTUATOR}")
    print(f"Joint ID:          {joint_id}")
    print(f"Actuator ID:       {actuator_id}")
    print(f"qpos index:        {qpos_index}")
    print(f"qvel index:        {qvel_index}")
    print(
        f"Joint range:       "
        f"[{joint_low:.3f}, {joint_high:.3f}] rad"
    )
    print(
        f"Torque range:      "
        f"[{ctrl_low:.3f}, {ctrl_high:.3f}] Nm"
    )
    print(f"Requested target:  {args.target:.3f} rad")
    print(f"Applied target:    {target:.3f} rad")

    set_initial_pose(model, data)
    joint_actuator_mapping = build_joint_actuator_mapping(
        model
    )

    hold_targets = data.qpos.copy()

    Path("results").mkdir(exist_ok=True)
    output_path = Path("results/single_joint_control.csv")

    if args.no_viewer:
        run_simulation(
            model=model,
            data=data,
            actuator_id=actuator_id,
            qpos_index=qpos_index,
            qvel_index=qvel_index,
            target=target,
            duration=args.duration,
            output_path=output_path,
            joint_actuator_mapping=joint_actuator_mapping,
            hold_targets=hold_targets,
        )
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run_simulation(
                model=model,
                data=data,
                actuator_id=actuator_id,
                qpos_index=qpos_index,
                qvel_index=qvel_index,
                target=target,
                duration=args.duration,
                output_path=output_path,
                joint_actuator_mapping=joint_actuator_mapping,
                hold_targets=hold_targets,
                viewer=viewer,
            )

            if viewer.is_running():
                print(
                    "Simulation complete. Close the MuJoCo "
                    "viewer window to finish."
                )

                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)

    final_angle = float(data.qpos[qpos_index])
    final_error = target - final_angle

    print("\n=== RESULT ===")
    print(f"Final angle:       {final_angle:.4f} rad")
    print(f"Final error:       {final_error:.4f} rad")
    print(f"Logged data:       {output_path}")


if __name__ == "__main__":
    main()
