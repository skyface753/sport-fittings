
from dataclasses import dataclass
from enum import Enum


# @dataclass(frozen=True)
# class OptimalAngles:
#     knee_extension_bottom: tuple[int, int] = (140, 150)
#     knee_extension_top: tuple[int, int] = (70, 80)
#     torso_to_horizontal: tuple[int, int] = (40, 50)
#     elbow_angle: tuple[int, int] = (150, 170)
#     shoulder_angle: tuple[int, int] = (85, 90)


# @dataclass(frozen=True)
# class OptimalAnglesOnTheDrops(OptimalAngles):
#     torso_to_horizontal: tuple[int, int] = (30, 40)


# @dataclass(frozen=True)
# class OptimalAnglesOnTheAeros:
#     knee_extension_bottom: tuple[int, int] = (140, 150)
#     knee_extension_top: tuple[int, int] = (70, 80)
#     # torso_to_horizontal: tuple[int, int] = (40, 50)
#     elbow_angle: tuple[int, int] = (150, 170)
#     shoulder_angle: tuple[int, int] = (85, 90)
#     hip_angle: tuple[int, int] = (40, 50)


from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class AngleSpec:
    label: str
    name: str
    points: Tuple[str, str, str]
    # mode -> list of (min, max)
    optimal_ranges: Dict[str, List[Tuple[int, int]]]
    modes: List[str]  # modes where this angle is measured


ANGLE_SPECS = [
    AngleSpec(
        label="Knee Angle (Hip-Knee-Ankle)",
        name="knee_angle",
        points=("right_hip", "right_knee", "right_ankle"),
        optimal_ranges={
            "hood": [(140, 150), (70, 80)],
            "drop": [(140, 150), (70, 80)],
            "aero": [(140, 150), (70, 80)],
        },
        modes=["hood", "drop", "aero"],
    ),
    AngleSpec(
        label="Torso Angle (Shoulder-Hip-Horizontal)",
        name="torso_to_horizontal",
        points=("right_shoulder", "right_hip", "horizontal_reference_point"),
        optimal_ranges={
            "hood": [(40, 50)],
            "drop": [(30, 40)],
        },
        modes=["hood", "drop"],
    ),
    AngleSpec(
        label="Elbow Angle (Shoulder-Elbow-Wrist)",
        name="elbow_angle",
        points=("right_shoulder", "right_elbow", "right_wrist"),
        optimal_ranges={
            "hood": [(150, 170)],
            "drop": [(150, 170)],
            # "aero": [(150, 170)], # TODO: prüfen, ob das so passt
        },
        modes=["hood", "drop"]  # , "aero"],
    ),
    AngleSpec(
        label="Shoulder Angle (Hip-Shoulder-Elbow)",
        name="shoulder_angle",
        points=("right_hip", "right_shoulder", "right_elbow"),
        optimal_ranges={
            "hood": [(85, 90)],
            "drop": [(85, 90)],
        },
        modes=["hood", "drop"]
    ),
    AngleSpec(
        label="Hip Angle (Shoulder-Hip-Knee)",
        name="hip_angle",
        points=("right_shoulder", "right_hip", "right_knee"),
        optimal_ranges={
            # "hood": [(40, 50)],
            # "drop": [(40, 50)],
            "aero": [(40, 50)],
        },
        modes=["aero"]  # "hood", "drop", "aero"],
    )
]
