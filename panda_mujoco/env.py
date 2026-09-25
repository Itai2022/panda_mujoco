from pathlib import Path
from typing import Union
import mujoco
from mujoco.viewer import launch_passive
import mujoco.renderer
import numpy as np


class MujocoEnv:
    def __init__(
        self,
        model_path: Union[str, Path],
        launch_viewer: bool = True,
        height: int = 256,
        width: int = 256,
    ) -> None:
        if isinstance(model_path, str):
            model_path = Path(model_path)
        self.mj_model = mujoco.MjModel.from_xml_path(str(model_path.absolute()))
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.viewer_handle = (
            launch_passive(
                self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
            )
            if launch_viewer
            else None
        )

        if self.viewer_handle:
            #  distance to the "look-at" point
            self.viewer_handle.cam.distance = 2.8
            self.viewer_handle.cam.elevation = -45.0  # look down
            self.viewer_handle.cam.azimuth = 210.0  # rotation around Z
        self.renderer = mujoco.renderer.Renderer(self.mj_model, height, width)

    def is_running(self):
        if self.viewer_handle:
            return self.viewer_handle.is_running()
        else:
            return False

    def sync(self):
        if self.viewer_handle:
            return self.viewer_handle.sync()

    def close(self):
        if self.viewer_handle and self.viewer_handle.is_running():
            self.viewer_handle.close()
            self.viewer_handle = None
        if self.renderer:
            self.renderer.close()

    def render(
        self,
        camera_id: Union[int, str, mujoco.MjvCamera] = -1,
        mode="rgb",
    ) -> np.ndarray:
        out = None
        self.renderer.update_scene(self.mj_data, camera_id)
        if mode == "seg":
            self.renderer.enable_segmentation_rendering()
            out = self.renderer.render()
            self.renderer.disable_segmentation_rendering()
        elif mode == "depth":
            self.renderer.enable_depth_rendering()
            out = self.renderer.render()
            self.renderer.disable_depth_rendering()
        else:
            out = self.renderer.render()
        # out[:] = np.flipud(out)
        return out

    def print_collision(self):
        mujoco.mj_fwdPosition(self.mj_model, self.mj_data)
        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]

            id1 = self.mj_model.geom_bodyid[contact.geom[0]]
            id2 = self.mj_model.geom_bodyid[contact.geom[1]]

            body1 = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, id1)
            body2 = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, id2)

            print(
                f"Collision between body: {body1} and {body2} with distance: {contact.dist}."
            )
