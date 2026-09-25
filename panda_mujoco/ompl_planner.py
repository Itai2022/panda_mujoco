from functools import partial
from pathlib import Path
from typing import Union

import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algos
import mujoco
import numpy as np
from ompl import base as ob
from ompl import geometric as og
from toppra.constraint import JointAccelerationConstraint, JointVelocityConstraint
from dataclasses import dataclass


@dataclass
class PlanConfig:
    acc_limit: float = 8.0
    vel_limit: float | np.ndarray | None = 2.178
    time_out: float = 1.0
    sample_range: float = 0.5
    max_retry: int = 10


class OMPLPlanner:
    def __init__(self, mjcf_path: Union[str, Path], ndim: int = 7) -> None:
        super().__init__()
        if isinstance(mjcf_path, str):
            mjcf_path = Path(mjcf_path)
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path.resolve()))
        self.collision_data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.collision_data, 0)
        mujoco.mj_forward(self.model, self.collision_data)
        self.ndim = ndim
        self.planning_space = ob.RealVectorStateSpace(self.ndim)
        # space bounds
        self.bound = ob.RealVectorBounds(self.ndim)

        self.ss = og.SimpleSetup(self.planning_space)

        # joint range
        self.set_bounds()

    def set_bounds(self):
        jnt_range = self.model.jnt_range
        self.q_min = jnt_range[:, 0]
        self.q_max = jnt_range[:, 1]
        for i in range(self.ndim):
            self.bound.setLow(i, self.q_min[i])
            self.bound.setHigh(i, self.q_max[i])
        self.planning_space.setBounds(self.bound)

    def smooth_path(self, ompl_path, vel_limit: np.ndarray):
        diffs = np.linalg.norm(np.diff(ompl_path, axis=0), axis=1)
        s = np.insert(np.cumsum(diffs), 0, 0.0)
        s = s / s[-1]
        path = ta.SplineInterpolator(s, ompl_path)
        # vel constraints
        jv_contraint = JointVelocityConstraint(vel_limit)
        acc_limit = np.array([8.0] * self.ndim)

        # acc constraints
        # pyright:ignore       acc_limit = np.stack([-limit, limit]).T
        ja_contraint = JointAccelerationConstraint(
            acc_limit, discretization_scheme=constraint.DiscretizationType.Interpolation
        )
        instance = algos.TOPPRA(
            [jv_contraint, ja_contraint], path, solver_wrapper="seidel"
        )
        jnt_traj = instance.compute_trajectory()
        if not jnt_traj:
            print("TOPPRA Failed. Original Path without smoothing resturned.")
            return ompl_path
        ts_sample = np.arange(
            0,
            jnt_traj.duration,  # pyright:ignore
            self.model.opt.timestep,
        )
        return jnt_traj(ts_sample)

    def get_planning_path(
        self,
        current_q,
        target_q,
        plan_setting: PlanConfig,
        ignore_pair=None,
        attach_obj=None,
        target_frame=None,
    ):
        self.ss.clear()
        # start & goal state
        start_state = self.planning_space.allocState()
        goal_state = self.planning_space.allocState()
        for i in range(self.ndim):
            start_state[i] = current_q[i]  # pyright:ignore
            goal_state[i] = target_q[i]  # pyright:ignore

        # set validity checker
        def isStateValid(
            state,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            ndim: int,
            target_frame,
            ignore_pair=None,
            attach_obj=None,
        ):
            qpos = [state[i] for i in range(ndim)]
            data.qpos[:ndim] = qpos
            mujoco.mj_fwdKinematics(model, data)
            if attach_obj is not None:
                target_id = data.site(target_frame).id
                gripper_xpos = data.site(target_id).xpos.copy()
                gripper_xmat = data.site(target_id).xmat.copy().reshape(3, 3)
                geom_id = data.geom(attach_obj).id
                geom_type = model.geom_type[geom_id]
                if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                    body_size = model.geom_size[geom_id][2]
                elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    body_size = model.geom_size[geom_id][1]
                elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    body_size = model.geom_size[geom_id][0]
                else:
                    body_size = model.geom_size[geom_id][-1]
                # attach_xpos = data.geom(attach_obj).xpos
                t_rel = np.array([0, 0, body_size])
                adr = model.body(attach_obj).jntadr[0]
                qpos_start = model.jnt_qposadr[adr]
                data.qpos[qpos_start : qpos_start + 3] = (
                    gripper_xmat @ t_rel + gripper_xpos
                )
                data.qpos[qpos_start + 3 : qpos_start + 7] = data.xquat[
                    target_id
                ].copy()
                mujoco.mj_fwdKinematics(model, data)

            mujoco.mj_fwdPosition(model, data)
            ignored = set(ignore_pair) if ignore_pair else set()
            for i in range(data.ncon):
                contact = data.contact[i]

                id1 = model.geom_bodyid[contact.geom[0]]
                id2 = model.geom_bodyid[contact.geom[1]]

                body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, id1)
                body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, id2)

                active1 = ("hand" in body1 or "finger" in body1 or "link" in body1) or (
                    attach_obj is not None and body1 == attach_obj
                )
                active2 = ("hand" in body2 or "finger" in body2 or "link" in body2) or (
                    attach_obj is not None and body2 == attach_obj
                )

                # environments ignore e.g. non target body collides with table
                if not (active1 or active2):
                    continue

                if body1 in ignored and body2 in ignored:
                    continue
                return False
            return True

        self.ss.setStateValidityChecker(
            partial(
                isStateValid,
                model=self.model,
                data=self.collision_data,
                ndim=self.ndim,
                target_frame=target_frame,
                ignore_pair=ignore_pair,
                attach_obj=attach_obj,
            )
        )
        self.ss.setStartAndGoalStates(start_state, goal_state)

        # set planner
        planner = og.RRTConnect(self.ss.getSpaceInformation())
        planner.setRange(plan_setting.sample_range)
        # planner = og.InformedRRTstar(self.ss.getSpaceInformation())
        self.ss.setPlanner(planner)

        solved = self.ss.solve(plan_setting.time_out)

        if solved and str(self.ss.getLastPlannerStatus()) == "Exact solution":
            solution_path = self.ss.getSolutionPath()
            num_state = solution_path.getStateCount()
            simplifer = og.PathSimplifier(self.ss.getSpaceInformation())

            simplifer.ropeShortcutPath(solution_path)
            num_state = solution_path.getStateCount()
            planned_path = np.empty((num_state, self.ndim))
            for i in range(num_state):
                state = solution_path.getState(i)
                for j in range(self.ndim):
                    planned_path[i, j] = state[j]  # pyright:ignore

            if plan_setting.vel_limit:
                if isinstance(plan_setting.vel_limit, float):
                    limit = np.array([plan_setting.vel_limit] * self.ndim)
                    vel_limit = np.stack([-limit, limit]).T
                return self.smooth_path(planned_path, vel_limit)  # pyright:ignore

            return planned_path
        else:
            print(
                f"OMPL Planning failed, planner status: {self.ss.getLastPlannerStatus()}"
            )

        return np.array([])

    def generate_path(
        self,
        current_q: np.ndarray,
        target_q: np.ndarray,
        plan_setting: PlanConfig,
        ignore_pair=None,
        attach_obj=None,
        target_frame=None,
        full_qpos=None,
    ):

        assert len(target_q) == len(current_q)

        if full_qpos is not None:
            self.collision_data.qpos[:] = full_qpos
        else:
            self.collision_data.qpos[: len(target_q)] = current_q
        for _ in range(plan_setting.max_retry):
            # rrt to find a collision free path from current joint to target joint
            rrt_path = self.get_planning_path(
                current_q, target_q, plan_setting, ignore_pair, attach_obj, target_frame
            )
            status = self.ss.getLastPlannerStatus()
            if str(status) == "Exact solution":
                return rrt_path
            elif str(status) == "Approximate solution":
                print("Approximate solution will not be returned.")
            else:
                print("Failed.")

        return np.array([])
