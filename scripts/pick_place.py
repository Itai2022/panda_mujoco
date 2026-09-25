from panda_mujoco.tasks.pick_place import PickPlaceTask
from panda_mujoco.robot import PandaArm


def main() -> None:
    target_body = input("What object do you want to grab?\n")
    arm = PandaArm(
        "./models/panda_simulation.xml",
        "./models/panda_urdf/panda_z_offset.urdf",
        target_frame="gripper",
    )
    try:
        PickPlaceTask(arm, target_body).run()
    finally:
        arm.close()


if __name__ == "__main__":
    main()
