from pathlib import Path
from typing import Union

import mujoco
import numpy as np
import pinocchio as pin

from ..ik_solver import IKConfig, IKSolver
from ..ompl_planner import OMPLPlanner, PlanConfig
from ..env import MujocoEnv
from ..utils import rotation_around_x, rotation_around_z


class AvoidanceTask(MujocoEnv):
    def __init__(
        self,
        mjcf_path: Union[str, Path],
        urdf_path: Union[str, Path],
        planning_dim: int = 7,
        launch_viewer: bool = True,
        height: int = 256,
        width: int = 256,
    ) -> None:
        super().__init__(mjcf_path, launch_viewer, height, width)
        self.ik_solver = IKSolver(mjcf_path, urdf_path)
        self.ompl_planner = OMPLPlanner(mjcf_path, planning_dim)
        self.solution_path = None

    def get_collision_free_path(
        self,
        target_frame: str,
        T_tar: pin.SE3,
        current_q: np.ndarray,
        plan_config: PlanConfig | None = None,
        ik_config: IKConfig | None = None,
    ):
        if not ik_config:
            ik_config = IKConfig()
        if not plan_config:
            plan_config = PlanConfig()
        target_q = self.ik_solver.solve_ik(target_frame, T_tar, ik_config)
        path = self.ompl_planner.generate_path(
            current_q,
            target_q,  # pyright:ignore
            plan_config,
        )
        self.solution_path = path
        self.idx = 0

    def step(self):
        if self.idx < len(self.solution_path):  # pyright:ignore
            with self.viewer_handle.lock():  # pyright: ignore
                self.mj_data.ctrl[:7] = self.solution_path[self.idx]  # pyright:ignore
            self.idx += 1

    def run(self) -> None:
        try:
            while self.is_running():
                self.step()
                mujoco.mj_step(self.mj_model, self.mj_data)
                self.sync()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    pa = AvoidanceTask(
        "./models/panda_obstacle_avoidance.xml", "./models/panda_urdf/panda.urdf"
    )
    target_frame = "gripper"
    target_rotation = rotation_around_x(np.pi) @ rotation_around_z(-np.pi / 2)
    target_pos = np.array([0.2, 0.4, 0.4])
    T_tar = pin.SE3(translation=target_pos, rotation=target_rotation)
    body_id = pa.mj_model.body("target_mocap").id
    pa.mj_data.mocap_pos[pa.mj_model.body_mocapid[body_id]] = target_pos
    mujoco.mj_fwdKinematics(pa.mj_model, pa.mj_data)
    ik_config = IKConfig(max_iter=200, max_retry=20)
    plan_config = PlanConfig(acc_limit=4.0, sample_range=0.5)
    pa.get_collision_free_path(
        target_frame=target_frame,
        T_tar=T_tar,
        current_q=pa.mj_data.qpos[:],
        plan_config=plan_config,
        ik_config=ik_config,
    )
    pa.run()
