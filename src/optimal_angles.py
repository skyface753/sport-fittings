import json
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class AngleSpec:
    label: str
    name: str
    points: Tuple[str, str, str]
    optimal_ranges: Dict[str, List[Tuple[int, int]]]
    modes: List[str]


def load_angle_specs_from_json(filepath: str, body_side: str = "right") -> List[AngleSpec]:
    with open(filepath, 'r') as f:
        data = json.load(f)

    angle_specs = []
    for item in data:
        # Convert points list back to tuple if necessary, though dataclasses handles lists fine for Tuple hints
        item['points'] = tuple(item['points'])
        # add the body side as the prefix to the points. expect for the horizontal_reference_point
        item['points'] = tuple(
            f"{body_side}_{point}" if point != "horizontal_reference_point" else point for point in item['points'])
        # Convert inner lists of optimal_ranges to tuples if strict Tuple[int, int] is needed,
        # otherwise List[List[int]] works with List[Tuple[int, int]] hint.
        # Here we assume List[Tuple[int, int]] so we convert the inner lists
        for mode, ranges in item['optimal_ranges'].items():
            item['optimal_ranges'][mode] = [tuple(r) for r in ranges]

        angle_specs.append(AngleSpec(**item))
    return angle_specs
