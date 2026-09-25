from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, no_type_check

import mujoco
import numpy as np
import osqp
import pinocchio as pin
import scipy.sparse as sparse


@dataclass
class IKConfig:
    max_iter: int = 200
    max_retry: int = 10
    sample_retry: int = 10
    W_J: np.ndarray = field(default_factory=lambda: np.eye(6))
    damp: float = 1e-4
    gain: float = 1.0
    configuration_limit_gain: float = 0.5
    dt: float = 0.1
    verbose: bool = False


class IKSolver:
    def __init__(
        self,
        mjcf_path: Union[str, Path],
        urdf_path: Union[str, Path],
    ) -> None:
        if isinstance(urdf_path, str):
            urdf_path = Path(urdf_path)

        if isinstance(mjcf_path, str):
            mjcf_path = Path(mjcf_path)
        # pinocchio model and data
        self.pin_model = pin.buildModelFromUrdf(str(urdf_path.resolve()))
        self.pin_data = self.pin_model.createData()
        self.q_min = self.pin_model.lowerPositionLimit
        self.q_max = self.pin_model.upperPositionLimit
        self.v_max = self.pin_model.upperVelocityLimit
        self.v_min = -self.v_max  # pyright: ignore

        # data for collision checking
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path.resolve()))
        self.collision_data = mujoco.MjData(self.model)
        # self.model.geom_margin[self.model.geom("obstacle").id] = 0.015

        # qp solver
        self.solver = osqp.OSQP()

    def solve_ik(
        self,
        target_frame: str,
        T_tar: pin.SE3,
        ik_config: IKConfig | None = None,
        q_init=None,
    ):
        goal_state_in_collision = []
        if ik_config is None:
            ik_config = IKConfig()
        for i in range(ik_config.max_retry):
            if q_init is not None:
                qpos = q_init
            else:
                qpos = self.random_valid(ik_config.sample_retry)
            # sample a collision free state
            if not qpos.shape[0]:
                if ik_config.verbose:
                    print("Start state in collision. Skip")
                continue

            it_count = 0
            if ik_config.verbose:
                print(f"Computing IK to the following pose:\n {T_tar}")

            translation_err = float("inf")
            rotation_err = float("inf")
            while it_count < ik_config.max_iter:
                it_count += 1
                pin.framesForwardKinematics(self.pin_model, self.pin_data, qpos)
                T_cur = self.pin_data.oMf[self.pin_model.getFrameId(target_frame)]  # pyright: ignore
                T_bt = T_cur.actInv(T_tar)
                err = pin.log6(T_bt).vector

                translation_err = np.linalg.norm(err[:3])  # pyright: ignore
                rotation_err = np.linalg.norm(err[3:])  # pyright: ignore
                if translation_err < 1e-3 and rotation_err < 1e-3:
                    if ik_config.verbose:
                        print(
                            f"Current translational errror: {translation_err:.3f}, roration error: {rotation_err:.3f}"
                        )
                        print(
                            f"Ik converged after {i + 1} trail with {it_count} iteration steps."
                        )
                    if self.collide(qpos):
                        if ik_config.verbose:
                            print("But final state is in collision.")
                        goal_state_in_collision.append(True)
                        break
                    return qpos
                if ik_config.verbose:
                    print(
                        f"Current translational errror: {translation_err:.3f}, roration error: {rotation_err:.3f}"
                    )

                J = self.get_jocobian(target_frame, qpos, T_bt)
                # passed in error instead of error / t, hence this is the
                # displacement of ee
                P, q_qp, A, v_lower, v_upper = self._build_qp(
                    J,
                    err,
                    ik_config.gain,
                    ik_config.W_J,
                    qpos,
                    ik_config.dt,
                    ik_config.damp,
                    ik_config.configuration_limit_gain,
                )  # noqa
                self.solver.setup(P, q_qp, A, v_lower, v_upper, verbose=False)
                result = self.solver.solve()
                qpos = pin.integrate(self.pin_model, qpos, result.x)
            if it_count >= ik_config.max_iter and ik_config.verbose:
                print(
                    f"IK did not converge after {ik_config.max_iter} steps. Pose may not be reachable or increase maxmimal number of iteration."
                )
                print(
                    f"Fianl translational error norm: {translation_err}, rotational error: {rotation_err}"
                )
                print(f"Final qpos: {qpos}")
        if all(goal_state_in_collision) and len(goal_state_in_collision):
            print(
                "All trail ended in collision state. Please check if state is actually reachable."
            )
        return None

    @no_type_check
    def random_valid(self, sample_retry: int = 10):
        for _ in range(sample_retry):
            q = np.random.uniform(low=self.q_min, high=self.q_max)
            if self.collide(q):
                continue
            else:
                return q
        return np.array([])

    def collide(
        self,
        qpos,
    ):
        self.collision_data.qpos[: len(qpos)] = qpos
        mujoco.mj_fwdPosition(self.model, self.collision_data)
        # if self.collision_data.ncon:
        #     for i in range(self.collision_data.ncon):
        #         contact = self.collision_data.contact[i]
        #         id1 = self.model.geom_bodyid[contact.geom1]
        #         id2 = self.model.geom_bodyid[contact.geom2]
        #         body1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, id1)
        #         body2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, id2)
        #         print(f"{body1} and {body2} collide with each other.")
        return self.collision_data.ncon > 0

    @no_type_check
    def _build_qp(self, J, error, gain, W_J, qpos, dt, damp, config_limit_gain):
        minus_error = -gain * error
        P = sparse.csc_matrix(J.T @ W_J @ J + damp * np.eye(J.shape[1]))
        q_qp = -J.T @ W_J @ minus_error

        if np.isinf(self.v_min).any() or np.isinf(self.v_max).any():
            # configuration limit: q_min <= qpos + \delta q <= q_max <==>
            # q_min - qpos <= delta_q <= q_max + qpos
            #  (self.q_min - qpos)
            v_lower = config_limit_gain * pin.difference(
                self.pin_model, qpos, self.q_min
            )
            #  (self.q_max - qpos)
            v_upper = config_limit_gain * pin.difference(
                self.pin_model, qpos, self.q_max
            )
        else:
            # delta_q >= q_min * dt
            v_lower = np.maximum(
                self.v_min * dt,
                config_limit_gain * pin.difference(self.pin_model, qpos, self.q_min),
            )
            # delta_q <= q_max * dt
            v_upper = np.minimum(
                self.v_max * dt,
                config_limit_gain * pin.difference(self.pin_model, qpos, self.q_max),
            )

        A = sparse.csc_matrix(np.eye(J.shape[1]))
        return P, q_qp, A, v_lower, v_upper

    def solve_ik_dls(self, target_frame: str, T_tar: pin.SE3, damp=1e-3, dt=0.01):
        q = self.random_valid()
        while True:
            pin.framesForwardKinematics(self.pin_model, self.pin_data, q)
            T_cur = self.pin_data.oMf[self.pin_model.getFrameId(target_frame)]  # pyright: ignore
            T_bt = T_cur.actInv(T_tar)
            err = pin.log6(T_bt).vector
            translation_err = np.linalg.norm(err[:3])  # pyright: ignore
            rotation_err = np.linalg.norm(err[3:])  # pyright: ignore
            if translation_err < 1e-3 and rotation_err < 1e-3:
                if (q < self.q_min).any() or (q >= self.q_max).any():  # pyright: ignore
                    print("Joint Out Of Limits")
                break
            print(
                f"Current translational errror: {translation_err:.3f}, roration error: {rotation_err:.3f}"
            )
            J = self.get_jocobian(target_frame, q, T_bt)
            v = -J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)  # pyright: ignore
            q = pin.integrate(self.pin_model, q, v * dt)
        return q

    def get_jocobian(self, target_frame: str, q: np.ndarray, T_bt: pin.SE3):
        J_raw = pin.computeFrameJacobian(
            self.pin_model,
            self.pin_data,
            q,
            self.pin_model.getFrameId(target_frame),
            pin.ReferenceFrame.LOCAL,
        )
        J = -pin.Jlog6(T_bt.inverse()) @ J_raw
        return J
