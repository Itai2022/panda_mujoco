import mujoco
import numpy as np
import pinocchio as pin

from panda_mujoco.ik_solver import IKConfig
from panda_mujoco.ompl_planner import PlanConfig
from panda_mujoco.tasks.avoidance import AvoidanceTask
from panda_mujoco.utils import rotation_around_x, rotation_around_z


def main() -> None:
    at = AvoidanceTask(
        "./models/panda_obstacle_avoidance.xml",
        "./models/panda_urdf/panda.urdf",
    )
    target_frame = "gripper"
    target_rotation = rotation_around_x(np.pi) @ rotation_around_z(-np.pi / 2)
    target_pos = np.array([0.2, 0.4, 0.4])
    T_tar = pin.SE3(translation=target_pos, rotation=target_rotation)

    body_id = at.mj_model.body("target_mocap").id
    at.mj_data.mocap_pos[at.mj_model.body_mocapid[body_id]] = target_pos
    mujoco.mj_forward(at.mj_model, at.mj_data)

    ik_config = IKConfig(max_iter=200, max_retry=20)
    plan_config = PlanConfig(acc_limit=4.0, vel_limit=1.0, sample_range=0.5)
    at.get_collision_free_path(
        target_frame=target_frame,
        T_tar=T_tar,
        current_q=at.mj_data.qpos[:],
        plan_config=plan_config,
        ik_config=ik_config,
    )
    at.run()


if __name__ == "__main__":
    main()
