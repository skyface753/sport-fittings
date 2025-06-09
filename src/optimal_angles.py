
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class OptimalAngles:
    knee_extension_bottom: tuple[int, int] = (140, 150)
    knee_extension_top: tuple[int, int] = (70, 80)
    torso_to_horizontal: tuple[int, int] = (40, 50)
    elbow_angle: tuple[int, int] = (150, 170)
    shoulder_angle: tuple[int, int] = (85, 90)
