
from src.models import AllLandmarks
from collections import defaultdict
from scipy.signal import find_peaks
import numpy as np
from src.models import Point


def calculate_angle(a: Point, b: Point, c: Point) -> float:
    """Calculates the angle (in degrees) between three points, with b as the vertex."""
    # Create vectors from the vertex point b
    vec1 = a - b
    vec2 = c - b

    # Calculate dot product
    dot_product = vec1.x * vec2.x + vec1.y * vec2.y

    # Calculate magnitudes
    mag1 = np.sqrt(vec1.x**2 + vec1.y**2)
    mag2 = np.sqrt(vec2.x**2 + vec2.y**2)

    if mag1 == 0 or mag2 == 0:
        return 0.0  # Avoid division by zero

    cosine_angle = dot_product / (mag1 * mag2)
    # Ensure cosine_angle is within [-1, 1] to prevent arccos errors due to floating point inaccuracies
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def calculate_seat_height_adjustment(
    current_angle: float,
    # optimal_range: tuple[float, float],
    next_optimal_angle: float,
    # Degrees per cm of seat height change (example value)
    sensitivity_factor: float = 0.5
) -> float:
    """
    Calculates the required seat height adjustment (in cm) based on the current angle
    and an optimal range.

    Args:
        current_angle: The currently measured angle in degrees.
        optimal_range: A tuple (min_angle, max_angle) representing the desired
                       range for the angle in degrees.
        sensitivity_factor: How many degrees the angle changes for every 1 cm
                            of seat height adjustment. This value is CRITICAL
                            and needs to be determined empirically or via a
                            biomechanical model. A positive value means
                            increasing seat height increases the angle.

    Returns:
        The required seat height adjustment in centimeters.
        - Positive value: Increase seat height.
        - Negative value: Decrease seat height.
        - 0.0: Angle is within the optimal range.
    """
    # min_optimal_angle, max_optimal_angle = optimal_range
    print(f"Current angle: {current_angle:.2f}°")
    print(f"Next optimal angle: {next_optimal_angle:.2f}°")

    # if min_optimal_angle <= current_angle <= max_optimal_angle:
    #     return 0.0  # Angle is already within the optimal range

    if sensitivity_factor == 0:
        print("Warning: Sensitivity factor is zero, cannot calculate adjustment.")
        return 0.0

    angle_difference = next_optimal_angle - current_angle
    adjustment = angle_difference / sensitivity_factor
    return adjustment


def print_angle_stats(angles: list[float], angle_name: str):
    """Prints statistics for a list of angles."""
    angle_stats = []
    if not angles:
        # print(f"No {angle_name} angles calculated.")
        angle_stats.append(f"No {angle_name} angles calculated.")
        return
    # print(f"\n--- {angle_name} Angle Statistics ---")
    # print(f"Min angle: {min(angles):.2f}°")
    # print(f"Max angle: {max(angles):.2f}°")
    # print(f"Mean angle: {np.mean(angles):.2f}°")
    # print(f"Median angle: {np.median(angles):.2f}°")
    # print(f"Standard deviation: {np.std(angles):.2f}°")
    # print("-" * (len(angle_name) + 22))
    angle_stats.append(f"{angle_name} Angle Statistics:")
    angle_stats.append(f"Min: {min(angles):.2f}°")
    angle_stats.append(f"Max: {max(angles):.2f}°")
    angle_stats.append(f"Mean: {np.mean(angles):.2f}°")
    angle_stats.append(f"Median: {np.median(angles):.2f}°")
    angle_stats.append(f"Std Dev: {np.std(angles):.2f}°")
    angle_stats.append("-" * (len(angle_name) + 22))
    return angle_stats


def calc_angles(all_frames_landmarks: AllLandmarks, mode: str, angle_specs) -> dict[str, list[float]]:
    dynamic_angles = defaultdict(list)

    for frame_lm in all_frames_landmarks.frames_landmarks:
        for spec in angle_specs:
            if mode not in spec.modes:
                continue  # skip angles not relevant for this mode

            p1, p2, p3 = spec.points
            if not hasattr(frame_lm, p1) or not hasattr(frame_lm, p2):
                # print(f"Missing points for angle {spec.label} in frame. "
                #       f"Points: {p1}, {p2}, {p3}")
                continue
            if p3 == "horizontal_reference_point":
                if hasattr(frame_lm, p2) and getattr(frame_lm, p2) is not None:
                    p3 = Point(
                        getattr(frame_lm, p2).x + 100, getattr(frame_lm, p2).y)
                    if all(hasattr(frame_lm, p) and getattr(frame_lm, p) for p in (p1, p2)):
                        angle = calculate_angle(
                            getattr(frame_lm, p1),
                            getattr(frame_lm, p2),
                            p3,
                        )
                        dynamic_angles[spec.label].append(angle)
                else:
                    continue
                    # print(
                    #     f"Horizontal reference point {p2} not found in frame for angle {spec.label}.")
            elif all(hasattr(frame_lm, p) and getattr(frame_lm, p) for p in (p1, p2, p3)):
                angle = calculate_angle(
                    getattr(frame_lm, p1),
                    getattr(frame_lm, p2),
                    getattr(frame_lm, p3),
                )
                dynamic_angles[spec.label].append(angle)
            else:
                print(f"Missing points for angle {spec.label} in frame. "
                      f"Points: {p1}, {p2}, {p3}")

    return dynamic_angles


def calc_peaks(dynamic_angles: dict[str, list[float]], angle_name="Knee Angle (Hip-Knee-Ankle)") -> tuple[list[int], np.ndarray, list[int], np.ndarray]:
    # Parameters
    peak_detection_distance = 20

    # Get the angle data
    angles = np.array(dynamic_angles[angle_name])

    # ------------------------
    # Find high peaks (maxima)
    peaks_high, _ = find_peaks(angles, distance=peak_detection_distance)
    mean_angle = np.mean(angles)
    peaks_high = [p for p in peaks_high if angles[p] > mean_angle]
    max_angles = angles[peaks_high]

    # ------------------------
    # Find low peaks (minima)
    peaks_low, _ = find_peaks(-angles, distance=peak_detection_distance)
    peaks_low = [p for p in peaks_low if angles[p] < mean_angle]
    min_angles = angles[peaks_low]

    return peaks_high, max_angles, peaks_low, min_angles


def angles_summary(dynamic_angles: dict[str, list[float]], avg_knee_angle_peaks_high: float,
                   avg_knee_angle_peaks_low: float):
    """
    Generates a summary of cycling angles as a list of formatted strings.

    Args:
        dynamic_angles: A dictionary where keys are angle labels (strings) and
                        values are lists of angle measurements (floats).
        avg_knee_angle_peaks_high: The average of the high peaks of the knee angle.
        avg_knee_angle_peaks_low: The average of the low peaks of the knee angle.

    Returns:
        A list of strings, each representing a line of the cycling angle summary.
    """
    summary_lines = []
    summary_lines.append("\n--- Cycling Angle Summary ---")

    # Calculate average angles (handle cases where angle list might be empty)
    # Using .get with a default empty list and then checking if it's empty is more robust
    shoulder_angles = dynamic_angles.get(
        "Shoulder Angle (Hip-Shoulder-Elbow)", [])
    avg_shoulder_angle = np.mean(shoulder_angles) if shoulder_angles else 0

    elbow_angles = dynamic_angles.get("Elbow Angle (Shoulder-Elbow-Wrist)", [])
    avg_elbow_angle = np.mean(elbow_angles) if elbow_angles else 0

    torso_angles = dynamic_angles.get(
        "Torso Angle (Shoulder-Hip-Horizontal)", [])
    avg_torso_angle = np.mean(torso_angles) if torso_angles else 0

    knee_angles = dynamic_angles.get("Knee Angle (Hip-Knee-Ankle)", [])
    avg_knee_angle = np.mean(knee_angles) if knee_angles else 0

    hip_angles = dynamic_angles.get("Hip Angle (Shoulder-Hip-Knee)", [])
    avg_hip_angle = np.mean(hip_angles) if hip_angles else 0

    # Populate the summary lines
    summary_lines.append(f'Knee Angle (Avg): {avg_knee_angle:.2f}°')
    summary_lines.append(
        f'Knee Angle Peaks (High): {avg_knee_angle_peaks_high:.2f}°')
    summary_lines.append(
        f'Knee Angle Peaks (Low): {avg_knee_angle_peaks_low:.2f}°')
    summary_lines.append(f'Shoulder Angle (Avg): {avg_shoulder_angle:.2f}°')
    summary_lines.append(f'Elbow Angle (Avg): {avg_elbow_angle:.2f}°')
    summary_lines.append(f'Torso Angle (Avg): {avg_torso_angle:.2f}°')
    summary_lines.append(f'Hip Angle (Avg): {avg_hip_angle:.2f}°')
    summary_lines.append("-----------------------------\n")

    return summary_lines
