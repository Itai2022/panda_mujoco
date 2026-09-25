from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np


@dataclass
class Trajectory:
    path: np.ndarray = field(default_factory=lambda: np.array([]))
    idx: int = 0

    def is_empty(self) -> bool:
        return len(self.path) == 0

    def is_exhausted(self) -> bool:
        return self.idx >= len(self.path)

    def reset(self) -> None:
        self.path = np.array([])
        self.idx = 0

    def pop_ctrl(self) -> np.ndarray | None:
        if self.is_exhausted():
            return None
        ctrl = self.path[self.idx]
        self.idx += 1
        return ctrl


class PickPlaceState(Enum):
    MOVE_TO_PREGRAB = auto()
    TO_GRAB = auto()
    PICK_UP = auto()
    MOVE_TO_TARGET = auto()
    RESET = auto()
