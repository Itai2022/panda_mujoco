import time
from pathlib import Path
from typing import Union

import mujoco
import numpy as np
import pinocchio as pin

from .ik_solver import IKSolver, IKConfig
from .env import MujocoEnv
from .ompl_planner import OMPLPlanner, PlanConfig
from .utils import motion_from_rotation_x


class PandaArm(MujocoEnv):
    def __init__(
        self,
        mjcf_path: Union[str, Path],
        urdf_path: Union[str, Path],
        launch_viewer: bool = True,
        target_frame: str = "gripper",
        height: int = 256,
        width: int = 256,
    ) -> None:
        super().__init__(mjcf_path, launch_viewer, height, width)
        self.ik_solver = IKSolver(mjcf_path, urdf_path)
        self.planner = OMPLPlanner(mjcf_path)
        self.gripper_opened = False
        self.gripper_speed = 1.6e-4
        self.attach_obj: str | None = None
        self.target_frame = target_frame

    def ee_pose(self) -> pin.SE3:
        site = self.mj_data.site("gripper")
        return pin.SE3(
            translation=site.xpos.copy(),
            rotation=site.xmat.reshape(3, 3).copy(),
        )

    def fk(self, target_q, plan_config=PlanConfig(), ignore_pair=None):
        current_q = self.mj_data.qpos[:9].copy()
        return self.planner.generate_path(
            current_q,
            target_q,
            plan_config,
            ignore_pair=ignore_pair,
            full_qpos=self.mj_data.qpos.copy(),
        )

    def ik(self, target_frame, T_tar, ik_config=IKConfig(), q_init=None):
        return self.ik_solver.solve_ik(target_frame, T_tar, ik_config, q_init)

    def move_to(
        self,
        target_frame,
        T_tar,
        ik_config=IKConfig(),
        plan_config=PlanConfig(),
        ignore_pair=None,
    ):
        target_q = self.ik(target_frame, T_tar, ik_config)
        if target_q is not None:
            target_q[7] = self.mj_data.qpos[7]
        return self.planner.generate_path(
            self.mj_data.qpos[: len(target_q)],  # pyright: ignore
            target_q,  # pyright: ignore
            plan_config,
            ignore_pair,
            self.attach_obj,
            target_frame,
            full_qpos=self.mj_data.qpos.copy(),
        )

    def set_ctrl(self, ctrl) -> None:
        with self.viewer_handle.lock():  # pyright: ignore
            self.mj_data.ctrl[:7] = ctrl

    def get_contact_force(self, block_name: str = "blue_block") -> float:
        total_force = 0.0
        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            b1 = mujoco.mj_id2name(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.mj_model.geom_bodyid[contact.geom1],
            )
            b2 = mujoco.mj_id2name(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.mj_model.geom_bodyid[contact.geom2],
            )
            is_block = block_name in b1 or block_name in b2
            is_finger_pad = any(
                kw in b for b in (b1, b2) for kw in ("fingertip_pad", "finger")
            )
            if is_block and is_finger_pad:
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.mj_model, self.mj_data, i, force)
                total_force += abs(force[0])
        self.current_force = total_force
        return total_force

    def open_gripper(self) -> None:
        if self.mj_data.qpos[7] <= 0.038:
            self.mj_data.ctrl[7] += self.gripper_speed
        else:
            self.gripper_opened = True

    def close_gripper(self, target_body: str | None = None) -> None:
        if target_body is not None:
            if self.get_contact_force(target_body) <= 20.0:
                self.mj_data.ctrl[7] -= self.gripper_speed
            else:
                self.gripper_opened = False
        else:
            if self.mj_data.qpos[7] > 0.002:
                self.mj_data.ctrl[7] -= self.gripper_speed
            else:
                self.gripper_opened = False


if __name__ == "__main__":
    arm = PandaArm(
        "./models/panda_simulation.xml", "./models/panda_urdf/panda_z_offset.urdf"
    )
    target_frame = "gripper"
    A = pin.SE3(translation=np.array([-0.5, -0.2, 0.1]), rotation=np.eye(3))
    T_ee = pin.SE3(
        translation=arm.mj_data.site("gripper").xpos,
        rotation=arm.mj_data.site("gripper").xmat.reshape(3, 3),
    )
    rotation = motion_from_rotation_x(-np.pi / 2)
    T_tar = A * T_ee * rotation
    T_tar = pin.SE3(translation=np.array([0.5, 0.5, 0.8]), rotation=np.eye(3))
    ik_config = IKConfig()
    plan_config = PlanConfig(acc_limit=0.5)
    path = arm.move_to(
        target_frame=target_frame,
        T_tar=T_tar,
        ik_config=ik_config,
        plan_config=plan_config,
    )
    idx = 0
    try:
        while arm.is_running():
            # if not path_executed:
            if idx < len(path):
                with arm.viewer_handle.lock():  # pyright:ignore
                    arm.mj_data.ctrl[:7] = path[min(idx, len(path))]
                arm.print_collision()
                idx += 5
            mujoco.mj_step(arm.mj_model, arm.mj_data)
            arm.sync()
            if idx > len(path):
                time.sleep(0.01)
                arm.open_gripper()
                arm.sync()
                mujoco.mj_step(arm.mj_model, arm.mj_data)
    except KeyboardInterrupt:
        pass
    finally:
        arm.close()
