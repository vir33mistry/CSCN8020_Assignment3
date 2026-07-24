from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer


def get_name(model, object_type, object_id):
    name = mujoco.mj_id2name(model, object_type, object_id)
    return name if name is not None else f"<unnamed-{object_id}>"


def print_model_summary(model):
    print("\n=== G1 MODEL SUMMARY ===")
    print(f"Bodies:              {model.nbody}")
    print(f"Joints:              {model.njnt}")
    print(f"Degrees of freedom:  {model.nv}")
    print(f"Position variables:  {model.nq}")
    print(f"Actuators:           {model.nu}")
    print(f"Sensors:             {model.nsensor}")
    print(f"Simulation timestep: {model.opt.timestep:.6f} seconds")


def print_joint_information(model):
    joint_type_names = {
        int(mujoco.mjtJoint.mjJNT_FREE): "free",
        int(mujoco.mjtJoint.mjJNT_BALL): "ball",
        int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
        int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
    }

    print("\n=== JOINTS ===")
    for joint_id in range(model.njnt):
        name = get_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = int(model.jnt_type[joint_id])
        print(
            f"{joint_id:2d} | {name:35s} | "
            f"type={joint_type_names.get(joint_type, str(joint_type)):6s} | "
            f"qpos_address={model.jnt_qposadr[joint_id]:2d} | "
            f"dof_address={model.jnt_dofadr[joint_id]:2d}"
        )


def print_actuator_information(model):
    print("\n=== ACTUATORS ===")
    for actuator_id in range(model.nu):
        name = get_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        low, high = model.actuator_ctrlrange[actuator_id]
        print(
            f"{actuator_id:2d} | {name:35s} | "
            f"control range=[{low:9.3f}, {high:9.3f}]"
        )


def print_sensor_information(model):
    print("\n=== SENSORS ===")
    for sensor_id in range(model.nsensor):
        name = get_name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        print(
            f"{sensor_id:2d} | {name:35s} | "
            f"dimension={model.sensor_dim[sensor_id]:2d} | "
            f"data_address={model.sensor_adr[sensor_id]:3d}"
        )


def run_viewer(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()

            delay = model.opt.timestep - (
                time.perf_counter() - step_start
            )
            if delay > 0:
                time.sleep(delay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    scene_path = args.scene.expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)

    model = mujoco.MjModel.from_xml_path(str(scene_path))

    print_model_summary(model)
    print_joint_information(model)
    print_actuator_information(model)
    print_sensor_information(model)

    if not args.no_viewer:
        run_viewer(model)


if __name__ == "__main__":
    main()
