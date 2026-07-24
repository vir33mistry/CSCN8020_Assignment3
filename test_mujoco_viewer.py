import time

import mujoco
import mujoco.viewer


MODEL_XML = """
<mujoco model="viewer_test">
    <worldbody>
        <light pos="0 0 3"/>
        <geom
            name="floor"
            type="plane"
            size="2 2 0.1"
            rgba="0.8 0.8 0.8 1"
        />
        <body name="box" pos="0 0 1">
            <freejoint/>
            <geom
                type="box"
                size="0.1 0.1 0.1"
                mass="1"
                rgba="0.2 0.5 0.8 1"
            />
        </body>
    </worldbody>
</mujoco>
"""


def main() -> None:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()

            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()