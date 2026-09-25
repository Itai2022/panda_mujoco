from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
import pinocchio as pin

from ..ik_solver import IKConfig
from ..ompl_planner import PlanConfig
from ..robot import PandaArm
from ..utils import (
    motion_from_translation,
    rotation_around_x,
    rotation_around_z,
)
from .task_utils import PickPlaceState, Trajectory


@dataclass
class PickPlaceConfig:
    vel_limit: float = 1.0
    plan_acc_limit: float = 2.0
    wait_duration: int = 150
    pregrab_height_offset: float = 0.05
    grab_approach_offset: tuple[float, float, float] = (0.0, 0.0, 0.05)
    lift_offset: tuple[float, float, float] = (0.0, 0.0, -0.1)
    target_safety_margin: float = 0.02
    lift_threshold: float = 0.02
    interp_steps: int = 100


class PickPlaceTask:
    def __init__(
        self,
        arm: PandaArm,
        target_body: str,
        config: PickPlaceConfig | None = None,
    ) -> None:
        self.arm = arm
        self.target_body = target_body
        self.config = config or PickPlaceConfig()
        self.body_size = self._measure_body_size(target_body)

        self.ik_config = IKConfig(verbose=False)
        self.plan_config = PlanConfig(vel_limit=self.config.vel_limit)
        self.qpos_init = arm.mj_data.qpos[:9].copy()

        self.traj = Trajectory()
        self.wait_steps = 0
        self.initial_z = 0.0
        self.state = PickPlaceState.MOVE_TO_PREGRAB

        self._handlers: dict[PickPlaceState, Callable[[], None]] = {
            PickPlaceState.MOVE_TO_PREGRAB: self._step_move_to_pregrab,
            PickPlaceState.TO_GRAB: self._step_to_grab,
            PickPlaceState.PICK_UP: self._step_pick_up,
            PickPlaceState.MOVE_TO_TARGET: self._step_move_to_target,
            PickPlaceState.RESET: self._step_reset,
        }

        self.video = []

    def _measure_body_size(self, name: str) -> float:
        geom_id = self.arm.mj_data.geom(name).id
        geom_type = self.arm.mj_model.geom_type[geom_id]
        size = self.arm.mj_model.geom_size[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return size[2]
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return size[1]
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return size[0]
        return size[-1]

    def _current_q(self) -> np.ndarray:
        return self.arm.mj_data.qpos[:9].copy()

    def _interpolate_to(self, target_q: np.ndarray) -> np.ndarray:
        return np.linspace(
            self._current_q()[:7], target_q[:7], self.config.interp_steps
        )

    def _enter(self, state: PickPlaceState) -> None:
        self.state = state
        self.traj.reset()
        self.wait_steps = 0

    def _tick_wait(self) -> bool:
        if self.wait_steps < self.config.wait_duration:
            self.wait_steps += 1
            return False
        self.wait_steps = 0
        return True

    def _send_next_ctrl(self) -> bool:
        ctrl = self.traj.pop_ctrl()
        if ctrl is None:
            return False
        self.arm.set_ctrl(ctrl)
        return True

    def _step_move_to_pregrab(self) -> None:
        if self.traj.is_empty():
            self.traj.path = self._plan_pregrab()
            return
        if self._send_next_ctrl():
            self.arm.open_gripper()
            return
        if self._tick_wait():
            self._enter(PickPlaceState.TO_GRAB)

    def _plan_pregrab(self) -> np.ndarray:
        geom = self.arm.mj_data.geom(self.target_body)
        pos = geom.xpos.copy()
        pos[2] += self.body_size + self.config.pregrab_height_offset
        angle_z = pin.rpy.matrixToRpy(geom.xmat.reshape(3, 3).copy())[2]  # pyright:ignore
        target_rot = rotation_around_z(angle_z) @ self.arm.ee_pose().rotation
        T_tar = pin.SE3(translation=pos, rotation=target_rot)
        return self.arm.move_to(
            self.arm.target_frame, T_tar, self.ik_config, self.plan_config
        )

    def _step_to_grab(self) -> None:
        if self.traj.is_empty():
            T_offset = motion_from_translation(
                np.array(self.config.grab_approach_offset)
            )
            target_q = self.arm.ik(
                self.arm.target_frame,
                self.arm.ee_pose() * T_offset,
                self.ik_config,
                q_init=self._current_q(),
            )
            if target_q is not None:
                self.traj.path = self._interpolate_to(target_q)
            return

        if self._send_next_ctrl():
            return

        if self.wait_steps < self.config.wait_duration:
            self.wait_steps += 1
            return

        self.arm.close_gripper(self.target_body)
        if not self.arm.gripper_opened:
            self._enter(PickPlaceState.PICK_UP)

    def _step_pick_up(self) -> None:
        if self.traj.is_empty():
            if self.arm.gripper_opened:
                return
            self.initial_z = self.arm.mj_data.body(self.target_body).xpos[2]
            T_offset = motion_from_translation(np.array(self.config.lift_offset))
            target_q = self.arm.ik(
                self.arm.target_frame,
                self.arm.ee_pose() * T_offset,
                self.ik_config,
                q_init=self._current_q(),
            )
            if target_q is not None:
                self.traj.path = self._interpolate_to(target_q)
            return

        if self._send_next_ctrl():
            return

        lifted = (
            self.arm.mj_data.body(self.target_body).xpos[2] - self.initial_z
            > self.config.lift_threshold
        )
        if lifted:
            self.arm.attach_obj = self.target_body
            self._enter(PickPlaceState.MOVE_TO_TARGET)
        else:
            print("did not pick up")
            self.arm.open_gripper()
            if self.arm.gripper_opened:
                print("gripper opened")
                self._enter(PickPlaceState.MOVE_TO_PREGRAB)

    def _step_move_to_target(self) -> None:
        if self.traj.is_empty():
            translation = self.arm.mj_data.site("target_circle").xpos[:].copy()
            translation[2] += 2 * self.body_size + self.config.target_safety_margin
            rotation = rotation_around_x(np.pi) @ rotation_around_z(-np.pi / 2)
            T_target = pin.SE3(rotation, translation)
            plan_config = PlanConfig(
                acc_limit=self.config.plan_acc_limit, vel_limit=self.config.vel_limit
            )
            self.traj.path = self.arm.move_to(
                self.arm.target_frame,
                T_target,
                self.ik_config,
                plan_config,
                ["left_finger", "right_finger", self.target_body],
            )
            return

        if self._send_next_ctrl():
            return

        if self.wait_steps < self.config.wait_duration:
            self.wait_steps += 1
            return

        self.arm.open_gripper()
        if self.arm.gripper_opened:
            self.arm.attach_obj = None
            self._enter(PickPlaceState.RESET)

    def _step_reset(self) -> None:
        if self.traj.is_empty():
            self.traj.path = self.arm.fk(self.qpos_init)
            return

        if not self.traj.is_exhausted():
            self._send_next_ctrl()
        else:
            self.arm.close_gripper()

    def step(self) -> None:
        self._handlers[self.state]()

    def run(self, record_video=False, fps=30) -> None:
        sim_dt = self.arm.mj_model.opt.timestep
        render_interval = max(1, int(1.0 / (fps * sim_dt)))
        step = 0
        try:
            while self.arm.is_running():
                self.step()
                mujoco.mj_step(self.arm.mj_model, self.arm.mj_data)
                if record_video and step % render_interval == 0:
                    self.video.append(self.arm.render())
                step += 1
                self.arm.sync()
        except KeyboardInterrupt:
            pass
        finally:
            import imageio.v3 as iio

            docs = Path(__file__).resolve().parents[2] / "docs"
            docs.mkdir(exist_ok=True, parents=True)

            print(f"Saving video with {len(self.video)} frames...")
            iio.imwrite(docs / "demo.gif", self.video, fps=fps, loop=0)
            print("Video Saved.")
            self.video = []
