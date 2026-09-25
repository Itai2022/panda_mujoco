from pinocchio import SE3
import numpy as np


def motion_from_translation(translation: np.ndarray):
    return SE3(translation=translation, rotation=np.eye(3))


def motion_from_rotation_x(angle: float):
    rotation = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(-angle)],
        ]
    )
    return SE3(translation=np.array([0.0, 0.0, 0.0]), rotation=rotation)


def motion_from_rotation_y(angle: float):
    rotation = np.array(
        [
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)],
        ]
    )
    return SE3(translation=np.array([0.0, 0.0, 0.0]), rotation=rotation)


def motion_from_rotation_z(angle: float):
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    return SE3(translation=np.array([0.0, 0.0, 0.0]), rotation=rotation)


def rotation_around_x(angle: float):
    rotation = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(-angle)],
        ]
    )
    return rotation


def rotation_around_y(angle: float):
    rotation = np.array(
        [
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)],
        ]
    )
    return rotation


def rotation_around_z(angle: float):
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    return rotation
